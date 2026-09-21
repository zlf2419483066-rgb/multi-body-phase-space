"""Append-only per-topology candidate identity checkpoints.

The normal five-body runner writes a complete catalog after the search.  This
module is an intentionally separate persistence primitive for long searches:
each topology is one gzip member in an append-only data log, and a later index
line commits that member only after its bytes have been flushed.  A truncated
data tail without a commit line is ignored; a committed block whose bytes or
identity evidence no longer validate is a corruption error.

This module does not resume a search and does not perform drawing.  It also
does not treat a source record's original vertex numbering as identity data:
every source is independently reviewed and must rederive the stored canonical
identity, while equivalent sources are allowed to differ in representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
import zlib

from .domain import CountMode
from .label_catalog import CandidateCatalog, CatalogEntry
from .label_identity import CandidateIdentity, canonicalize_candidate_labeling
from .physical_labels import CandidateMomentumRecord, review_candidate_record


CHECKPOINT_SCHEMA_VERSION = "F5-catalog-checkpoint-v1"
SCOPE_FILENAME = "scope.json"
INDEX_FILENAME = "index.jsonl"
MODE_NAMES = tuple(mode.value for mode in CountMode)
DATA_FORMAT = "gzip-member-jsonl-v1"
DATA_FILENAME = "data.jsonl.gz"


class CheckpointError(ValueError):
    """A checkpoint is invalid, inconsistent, or cannot be read safely."""


class CheckpointCorruptionError(CheckpointError):
    """A committed index block no longer matches its data bytes/evidence."""


def _jsonable(value: Any) -> Any:
    # These exact immutable/scalar cases occur throughout checkpoint rows.
    # Keeping subclasses on the historical fallback path preserves the
    # Mapping/Enum behavior tested by the public serialization boundary.
    exact_type = type(value)
    if exact_type in (str, int, float, bool, type(None)):
        return value
    if exact_type is dict:
        return {str(key): _jsonable(item) for key, item in value.items()}
    if exact_type is tuple:
        return [_jsonable(item) for item in value]
    if exact_type is list:
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _key_token(key: Any) -> str:
    return _json_bytes(key).decode("utf-8")


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _topology_digest(topology_key: Any) -> str:
    return _digest_bytes(_json_bytes(topology_key))


def _validate_scope_topology(scope: Mapping[str, Any], topology_key: Any) -> None:
    """Check the canonical topology header against optional n/loop scope."""

    key = topology_key
    if not isinstance(key, (list, tuple)) or len(key) < 3 or key[0] != "topology":
        raise CheckpointCorruptionError("topology key is not a canonical topology key")
    scope_n = scope.get("n")
    if scope_n is not None and key[1] != scope_n:
        raise CheckpointCorruptionError("topology key n disagrees with checkpoint scope")
    scope_loop = scope.get("loop_count")
    if scope_loop is not None:
        if scope_n is None:
            raise CheckpointCorruptionError("loop_count scope requires n")
        if key[2] != scope_n - 1 + scope_loop:
            raise CheckpointCorruptionError("topology key vertex count disagrees with checkpoint n/loop_count")


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise CheckpointError(f"{name} must be a boolean")
    return value


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointError(f"{name} must be a non-negative integer")
    return value


def _complete_index_lines(raw: bytes) -> tuple[bytes, ...]:
    """Return newline-terminated index records, ignoring only EOF truncation."""

    if not raw:
        return ()
    lines = raw.splitlines(keepends=True)
    complete: list[bytes] = []
    for line in lines:
        if not line.endswith(b"\n"):
            # The writer always appends a newline in one write.  A final line
            # without it is therefore an uncommitted/truncated index record.
            break
        if line.strip():
            complete.append(line)
    return tuple(complete)


def _write_fsync(handle: Any, payload: bytes) -> None:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise CheckpointError(f"immutable checkpoint header differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            _write_fsync(handle, payload)
    except FileExistsError:
        existing = path.read_bytes()
        if existing != payload:
            raise CheckpointError(f"immutable checkpoint header differs: {path}")


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    mode: str
    identity: CandidateIdentity
    source_record: CandidateMomentumRecord
    source_index: int

    @property
    def digest(self) -> str:
        return self.identity.digest

    @property
    def key(self) -> tuple[Any, ...]:
        return self.identity.key


@dataclass(frozen=True, slots=True)
class TopologyCheckpoint:
    topology_key: tuple[Any, ...]
    topology_digest: str
    identities_by_mode: Mapping[str, tuple[CheckpointIdentity, ...]]
    label_enumeration_complete: Mapping[str, bool]
    identity_save_complete: Mapping[str, bool]
    topology_complete: bool
    data_offset: int
    data_length: int
    data_sha256: str
    audit: Mapping[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return (
            self.topology_complete
            and all(self.label_enumeration_complete.get(mode, False) for mode in MODE_NAMES)
            and all(self.identity_save_complete.get(mode, False) for mode in MODE_NAMES)
        )


@dataclass(frozen=True, slots=True)
class CheckpointReadback:
    root: Path
    scope: Mapping[str, Any]
    expected_topology_keys: tuple[Any, ...] | None
    records: tuple[TopologyCheckpoint, ...]

    @property
    def topology_keys(self) -> tuple[Any, ...]:
        return tuple(record.topology_key for record in self.records)

    @property
    def complete(self) -> bool:
        if self.expected_topology_keys is None:
            return False
        expected = {_key_token(key) for key in self.expected_topology_keys}
        actual = {_key_token(key) for key in self.topology_keys}
        return expected == actual and len(expected) == len(actual) and all(
            record.complete for record in self.records
        )

    @property
    def identity_keys_by_mode(self) -> dict[str, frozenset[str]]:
        result = {mode: set() for mode in MODE_NAMES}
        for record in self.records:
            for mode, identities in record.identities_by_mode.items():
                result.setdefault(mode, set()).update(_key_token(item.key) for item in identities)
        return {mode: frozenset(values) for mode, values in result.items()}

    @property
    def identity_digests_by_mode(self) -> dict[str, frozenset[str]]:
        result = {mode: set() for mode in MODE_NAMES}
        for record in self.records:
            for mode, identities in record.identities_by_mode.items():
                result.setdefault(mode, set()).update(item.digest for item in identities)
        return {mode: frozenset(values) for mode, values in result.items()}


def _scope_payload(scope: Mapping[str, Any], expected_topology_keys: Iterable[Any] | None) -> dict[str, Any]:
    if not isinstance(scope, Mapping):
        raise TypeError("scope must be a mapping")
    expected = None if expected_topology_keys is None else [_jsonable(key) for key in expected_topology_keys]
    if expected is not None:
        tokens = [_key_token(key) for key in expected]
        if len(tokens) != len(set(tokens)):
            raise CheckpointError("expected_topology_keys contains duplicates")
    base = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "scope": _jsonable(scope),
        "expected_topology_keys": expected,
        "expected_scope_complete": expected is not None,
    }
    base["scope_sha256"] = _digest_bytes(_json_bytes(base))
    return base


def _read_scope(root: Path) -> tuple[dict[str, Any], tuple[Any, ...] | None, str]:
    path = root / SCOPE_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read checkpoint scope: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("unsupported checkpoint scope schema")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise CheckpointError("checkpoint scope must be a mapping")
    expected = payload.get("expected_topology_keys")
    if expected is not None and not isinstance(expected, list):
        raise CheckpointError("expected_topology_keys must be an array or null")
    expected_complete = payload.get("expected_scope_complete")
    if type(expected_complete) is not bool or expected_complete != (expected is not None):
        raise CheckpointError("expected_scope_complete disagrees with expected_topology_keys")
    scope_sha256 = payload.get("scope_sha256")
    base = {
        "schema_version": payload.get("schema_version"),
        "scope": _jsonable(scope),
        "expected_topology_keys": expected,
        "expected_scope_complete": expected_complete,
    }
    if scope_sha256 != _digest_bytes(_json_bytes(base)):
        raise CheckpointCorruptionError("checkpoint scope checksum mismatch")
    expected_tuple = None if expected is None else tuple(expected)
    if expected_tuple is not None:
        tokens = [_key_token(key) for key in expected_tuple]
        if len(tokens) != len(set(tokens)):
            raise CheckpointError("checkpoint scope contains duplicate topology keys")
    return dict(scope), expected_tuple, str(scope_sha256)


def _identity_source(identity_payload: Mapping[str, Any], source_payload: Mapping[str, Any], source_index: Any) -> CheckpointIdentity:
    try:
        identity = CandidateIdentity.from_dict(identity_payload)
        source = CandidateMomentumRecord.from_dict(source_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointCorruptionError("invalid checkpoint identity/source payload") from exc
    index = _strict_int(source_index, "source_index")
    try:
        review_candidate_record(source)
        rebuilt = canonicalize_candidate_labeling(source, identity.mode)
    except (TypeError, ValueError) as exc:
        raise CheckpointCorruptionError("source record failed candidate identity review") from exc
    if rebuilt.digest != identity.digest or rebuilt.key != identity.key or rebuilt.topology_key != identity.topology_key:
        raise CheckpointCorruptionError("source record does not rederive the stored canonical identity")
    if (
        identity.vertex_map != rebuilt.vertex_map
        or identity.particle_map != rebuilt.particle_map
        or identity.edge_occurrence_map != rebuilt.edge_occurrence_map
        or identity.final_slot_map != rebuilt.final_slot_map
    ):
        raise CheckpointCorruptionError("stored identity maps do not match source rederivation")
    if rebuilt.canonical_record.to_dict() != identity.canonical_record.to_dict():
        raise CheckpointCorruptionError("stored canonical identity record differs from source rederivation")
    return CheckpointIdentity(identity.mode.value, identity, source, index)


def _decode_data_block(payload: bytes, expected_topology_key: Any, expected_digest: str,
                       scope: Mapping[str, Any], scope_sha256: str) -> TopologyCheckpoint:
    try:
        row = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptionError("committed checkpoint data is not valid JSON") from exc
    if (not isinstance(row, Mapping) or row.get("record_type") != "topology"
            or row.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or row.get("data_format") != DATA_FORMAT
            or row.get("scope_sha256") != scope_sha256):
        raise CheckpointCorruptionError("committed checkpoint data row is not a topology record")
    topology_key = row.get("topology_key")
    topology_digest = row.get("topology_digest")
    if _key_token(topology_key) != _key_token(expected_topology_key) or topology_digest != expected_digest:
        raise CheckpointCorruptionError("index topology key/digest disagrees with data")
    _validate_scope_topology(scope, topology_key)
    if _topology_digest(topology_key) != topology_digest:
        raise CheckpointCorruptionError("topology digest does not match full topology key")
    completion = row.get("completion")
    if not isinstance(completion, Mapping):
        raise CheckpointCorruptionError("topology checkpoint is missing completion metadata")
    audit = row.get("audit", {})
    if not isinstance(audit, Mapping):
        raise CheckpointCorruptionError("topology checkpoint audit must be a mapping")
    topology_complete = _strict_bool(completion.get("topology_complete"), "topology_complete")
    label_complete_raw = completion.get("label_enumeration_complete")
    identity_complete_raw = completion.get("identity_save_complete")
    if not isinstance(label_complete_raw, Mapping) or not isinstance(identity_complete_raw, Mapping):
        raise CheckpointCorruptionError("completion ranges must be per-mode mappings")
    label_complete = {mode: _strict_bool(label_complete_raw.get(mode), f"label_enumeration_complete[{mode}]") for mode in MODE_NAMES}
    identity_complete = {mode: _strict_bool(identity_complete_raw.get(mode), f"identity_save_complete[{mode}]") for mode in MODE_NAMES}
    if audit.get("stop_reason") is not None and (
        topology_complete or any(label_complete.values()) or any(identity_complete.values())
    ):
        raise CheckpointCorruptionError(
            "a topology with a stop reason cannot claim complete checkpoint coverage"
        )
    modes = row.get("modes")
    if not isinstance(modes, Mapping):
        raise CheckpointCorruptionError("topology checkpoint is missing mode rows")
    identities_by_mode: dict[str, tuple[CheckpointIdentity, ...]] = {}
    for mode in MODE_NAMES:
        rows = modes.get(mode, ())
        if not isinstance(rows, list):
            raise CheckpointCorruptionError(f"checkpoint rows for {mode} must be an array")
        decoded: list[CheckpointIdentity] = []
        seen: set[str] = set()
        for item in rows:
            if not isinstance(item, Mapping):
                raise CheckpointCorruptionError("checkpoint identity row must be a mapping")
            value = _identity_source(item.get("identity", {}), item.get("source_record", {}), item.get("source_index"))
            if value.mode != mode or _key_token(value.identity.topology_key) != _key_token(topology_key):
                raise CheckpointCorruptionError("identity mode/topology disagrees with topology checkpoint")
            if value.digest in seen:
                raise CheckpointCorruptionError("duplicate identity digest in one topology checkpoint")
            seen.add(value.digest)
            decoded.append(value)
        identities_by_mode[mode] = tuple(sorted(decoded, key=lambda item: item.digest))
    declared_identity_counts = audit.get("identity_counts")
    if declared_identity_counts is not None:
        if not isinstance(declared_identity_counts, Mapping):
            raise CheckpointCorruptionError("audit identity_counts must be a mapping")
        for mode in MODE_NAMES:
            declared_count = _strict_int(
                declared_identity_counts.get(mode),
                f"audit.identity_counts[{mode}]",
            )
            actual_count = len(identities_by_mode[mode])
            if declared_count != actual_count:
                raise CheckpointCorruptionError(
                    f"audit identity_counts[{mode}] disagrees with checkpoint rows"
                )
    complete_coverage = (
        topology_complete
        and all(label_complete.get(mode, False) for mode in MODE_NAMES)
        and all(identity_complete.get(mode, False) for mode in MODE_NAMES)
    )
    declared_pass_count = audit.get("momentum_pass_count")
    if declared_pass_count is not None:
        declared_pass_count = _strict_int(
            declared_pass_count, "audit.momentum_pass_count"
        )
    if complete_coverage and declared_pass_count is not None:
        nonempty_modes = {
            mode: bool(identities_by_mode[mode]) for mode in MODE_NAMES
        }
        if declared_pass_count > 0 and not all(nonempty_modes.values()):
            raise CheckpointCorruptionError(
                "complete positive-momentum checkpoint must contain identities in every mode"
            )
        if declared_pass_count == 0 and any(nonempty_modes.values()):
            raise CheckpointCorruptionError(
                "complete zero-momentum checkpoint must not contain identities"
            )
    return TopologyCheckpoint(tuple(topology_key) if isinstance(topology_key, list) else topology_key,
                              topology_digest, identities_by_mode, label_complete, identity_complete,
                              topology_complete, 0, 0, "", dict(audit))


class CatalogCheckpointWriter:
    """Append topology data rows and commit them with a flushed index line."""

    def __init__(self, root: str | Path, *, scope: Mapping[str, Any],
                 expected_topology_keys: Iterable[Any] | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._scope = _scope_payload(scope, expected_topology_keys)
        _write_immutable(self.root / SCOPE_FILENAME, (_json_bytes(self._scope) + b"\n"))
        (self.root / DATA_FILENAME).touch(exist_ok=True)
        (self.root / INDEX_FILENAME).touch(exist_ok=True)
        existing = read_checkpoint(self.root)
        index_bytes = self.index_path.read_bytes()
        if index_bytes and not index_bytes.endswith(b"\n"):
            raise CheckpointError(
                "checkpoint index has an uncommitted truncated tail; refusing to append"
            )
        self._committed = {_key_token(key) for key in existing.topology_keys}
        expected = self._scope.get("expected_topology_keys")
        self._expected_tokens = None if expected is None else {_key_token(key) for key in expected}

    @property
    def scope_sha256(self) -> str:
        return str(self._scope["scope_sha256"])

    @property
    def data_path(self) -> Path:
        return self.root / DATA_FILENAME

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILENAME

    def append_topology(
        self,
        topology_key: Any,
        *,
        identities_by_mode: Mapping[str, Iterable[CandidateIdentity | CatalogEntry]],
        source_records_by_mode: Mapping[str, Mapping[str, CandidateMomentumRecord]] | None = None,
        label_enumeration_complete: Mapping[str, bool] | None = None,
        identity_save_complete: Mapping[str, bool] | None = None,
        topology_complete: bool = False,
        audit: Mapping[str, Any] | None = None,
    ) -> TopologyCheckpoint:
        return self._append_topology(
            topology_key,
            identities_by_mode=identities_by_mode,
            source_records_by_mode=source_records_by_mode,
            label_enumeration_complete=label_enumeration_complete,
            identity_save_complete=identity_save_complete,
            topology_complete=topology_complete,
            audit=audit,
            _verify=True,
        )

    def _append_verified_topology(
        self,
        topology_key: Any,
        *,
        identities_by_mode: Mapping[str, Iterable[CandidateIdentity | CatalogEntry]],
        source_records_by_mode: Mapping[str, Mapping[str, CandidateMomentumRecord]] | None = None,
        label_enumeration_complete: Mapping[str, bool] | None = None,
        identity_save_complete: Mapping[str, bool] | None = None,
        topology_complete: bool = False,
        audit: Mapping[str, Any] | None = None,
    ) -> TopologyCheckpoint:
        """Internal path for objects already validated by public canonical APIs."""

        return self._append_topology(
            topology_key,
            identities_by_mode=identities_by_mode,
            source_records_by_mode=source_records_by_mode,
            label_enumeration_complete=label_enumeration_complete,
            identity_save_complete=identity_save_complete,
            topology_complete=topology_complete,
            audit=audit,
            _verify=False,
        )

    def _append_topology(
        self,
        topology_key: Any,
        *,
        identities_by_mode: Mapping[str, Iterable[CandidateIdentity | CatalogEntry]],
        source_records_by_mode: Mapping[str, Mapping[str, CandidateMomentumRecord]] | None = None,
        label_enumeration_complete: Mapping[str, bool] | None = None,
        identity_save_complete: Mapping[str, bool] | None = None,
        topology_complete: bool = False,
        audit: Mapping[str, Any] | None = None,
        _verify: bool,
    ) -> TopologyCheckpoint:
        key_token = _key_token(topology_key)
        if key_token in self._committed:
            raise CheckpointError("topology key is already committed")
        if self._expected_tokens is not None and key_token not in self._expected_tokens:
            raise CheckpointError("topology key is outside immutable checkpoint scope")
        _validate_scope_topology(self._scope["scope"], topology_key)
        if not isinstance(identities_by_mode, Mapping):
            raise TypeError("identities_by_mode must be a mapping")
        source_records_by_mode = source_records_by_mode or {}
        labels = {mode: _strict_bool((label_enumeration_complete or {}).get(mode, False), f"label_enumeration_complete[{mode}]") for mode in MODE_NAMES}
        saved = {mode: _strict_bool((identity_save_complete or {}).get(mode, False), f"identity_save_complete[{mode}]") for mode in MODE_NAMES}
        if audit is not None and not isinstance(audit, Mapping):
            raise TypeError("audit must be a mapping or None")
        audit_payload = _jsonable(audit or {})
        if audit_payload.get("stop_reason") is not None:
            # A stopped labeling stage cannot certify full topology coverage,
            # even when a caller supplied optimistic completion flags.
            topology_complete = False
            labels = {mode: False for mode in MODE_NAMES}
            saved = {mode: False for mode in MODE_NAMES}
        complete_topology_key = _jsonable(topology_key)
        topology_digest = _topology_digest(complete_topology_key)
        mode_rows: dict[str, list[dict[str, Any]]] = {}
        decoded_by_mode: dict[str, tuple[CheckpointIdentity, ...]] = {}
        for mode in MODE_NAMES:
            raw_items = identities_by_mode.get(mode, ())
            rows: list[dict[str, Any]] = []
            decoded: list[CheckpointIdentity] = []
            seen: set[str] = set()
            source_map = source_records_by_mode.get(mode, {})
            if not isinstance(source_map, Mapping):
                raise TypeError(f"source_records_by_mode[{mode}] must be a mapping")
            for source_index, raw in enumerate(raw_items):
                entry_source = raw.source_record if isinstance(raw, CatalogEntry) else None
                if isinstance(raw, CatalogEntry):
                    # Catalog reloads expose CatalogIdentity, whose mode is a
                    # string and whose source is already retained by the
                    # entry.  Normalize through the public CandidateIdentity
                    # parser so the same map/key checks apply to both APIs.
                    try:
                        identity = CandidateIdentity.from_dict(raw.identity.to_dict())
                    except (KeyError, TypeError, ValueError) as exc:
                        raise CheckpointError("catalog entry identity cannot be normalized") from exc
                else:
                    identity = raw
                if not isinstance(identity, CandidateIdentity):
                    raise TypeError("identities_by_mode values must be CandidateIdentity or CatalogEntry")
                if identity.mode.value != mode or _key_token(identity.topology_key) != key_token:
                    raise CheckpointError("identity mode/topology disagrees with append topology")
                if identity.digest in seen:
                    raise CheckpointError("duplicate identity digest in append topology")
                seen.add(identity.digest)
                source = entry_source or source_map.get(identity.digest, identity.canonical_record)
                if not isinstance(source, CandidateMomentumRecord):
                    raise TypeError("source record must be CandidateMomentumRecord")
                checked = (_identity_source(identity.to_dict(), source.to_dict(), source_index)
                           if _verify else CheckpointIdentity(mode, identity, source, source_index))
                decoded.append(checked)
                rows.append({"identity": identity.to_dict(), "source_record": source.to_dict(), "source_index": source_index})
            mode_rows[mode] = rows
            decoded_by_mode[mode] = tuple(sorted(decoded, key=lambda item: item.digest))
        row = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "record_type": "topology",
            "scope_sha256": self.scope_sha256,
            "data_format": DATA_FORMAT,
            "topology_key": complete_topology_key,
            "topology_digest": topology_digest,
            "completion": {
                "topology_complete": _strict_bool(topology_complete, "topology_complete"),
                "label_enumeration_complete": labels,
                "identity_save_complete": saved,
            },
            "audit": audit_payload,
            "modes": mode_rows,
        }
        payload = _json_bytes(row) + b"\n"
        compressed = gzip.compress(payload, compresslevel=1, mtime=0)
        offset = self.data_path.stat().st_size
        with self.data_path.open("ab") as handle:
            _write_fsync(handle, compressed)
        index_row = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "record_type": "commit",
            "scope_sha256": self.scope_sha256,
            "data_format": DATA_FORMAT,
            "topology_key": complete_topology_key,
            "topology_digest": topology_digest,
            "data_offset": offset,
            "data_length": len(compressed),
            "data_sha256": _digest_bytes(compressed),
            "data_uncompressed_size": len(payload),
            "entry_counts": {mode: len(mode_rows[mode]) for mode in MODE_NAMES},
            "topology_complete": bool(topology_complete),
            "label_enumeration_complete": labels,
            "identity_save_complete": saved,
            "audit": _jsonable(audit or {}),
        }
        index_payload = _json_bytes(index_row) + b"\n"
        with self.index_path.open("ab") as handle:
            _write_fsync(handle, index_payload)
        self._committed.add(key_token)
        return TopologyCheckpoint(tuple(complete_topology_key), topology_digest, decoded_by_mode,
                                  labels, saved, bool(topology_complete), offset, len(compressed),
                                  _digest_bytes(compressed), dict(audit or {}))


def _iter_index_lines(path: Path) -> Iterable[bytes]:
    try:
        with path.open("rb") as handle:
            for line in handle:
                if not line.endswith(b"\n"):
                    # Only an EOF-truncated final index record is ignored.
                    break
                if line.strip():
                    yield line
    except OSError as exc:
        raise CheckpointError(f"cannot read checkpoint index: {path}") from exc


def _decompress_member(payload: bytes, expected_size: int) -> bytes:
    decoder = zlib.decompressobj(16 + 15)
    try:
        plain = decoder.decompress(payload) + decoder.flush()
    except zlib.error as exc:
        raise CheckpointCorruptionError("committed gzip member is invalid") from exc
    if not decoder.eof or decoder.unused_data:
        raise CheckpointCorruptionError("committed data range is not exactly one gzip member")
    if len(plain) != expected_size:
        raise CheckpointCorruptionError("gzip member uncompressed size mismatch")
    return plain


def _iter_checkpoint_records(root: Path) -> tuple[dict[str, Any], tuple[Any, ...] | None, str, Iterable[TopologyCheckpoint]]:
    scope, expected, scope_sha256 = _read_scope(root)
    data_path = root / DATA_FILENAME
    index_path = root / INDEX_FILENAME
    try:
        data_size = data_path.stat().st_size
        data_handle = data_path.open("rb")
    except OSError as exc:
        raise CheckpointError(f"cannot read checkpoint data: {root}") from exc
    seen_topologies: set[str] = set()
    expected_tokens = None if expected is None else {_key_token(key) for key in expected}
    last_committed_end = 0

    def records() -> Iterable[TopologyCheckpoint]:
        nonlocal last_committed_end
        try:
            for raw_line in _iter_index_lines(index_path):
                try:
                    commit = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise CheckpointCorruptionError("committed index line is invalid JSON") from exc
                if (not isinstance(commit, Mapping)
                        or commit.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
                        or commit.get("record_type") != "commit"
                        or commit.get("scope_sha256") != scope_sha256
                        or commit.get("data_format") != DATA_FORMAT):
                    raise CheckpointCorruptionError("invalid committed checkpoint index row")
                topology_key = commit.get("topology_key")
                topology_digest = commit.get("topology_digest")
                token = _key_token(topology_key)
                if token in seen_topologies:
                    raise CheckpointCorruptionError("duplicate committed topology key")
                if _topology_digest(topology_key) != topology_digest:
                    raise CheckpointCorruptionError("committed topology digest does not match full key")
                if expected_tokens is not None and token not in expected_tokens:
                    raise CheckpointCorruptionError("committed topology is outside immutable scope")
                offset = _strict_int(commit.get("data_offset"), "data_offset")
                length = _strict_int(commit.get("data_length"), "data_length")
                uncompressed_size = _strict_int(commit.get("data_uncompressed_size"), "data_uncompressed_size")
                if length <= 0 or offset + length > data_size:
                    raise CheckpointCorruptionError("committed data range exceeds data log")
                current_range = (offset, offset + length)
                if offset < last_committed_end:
                    raise CheckpointCorruptionError("committed data ranges overlap or are out of order")
                data_handle.seek(offset)
                compressed = data_handle.read(length)
                if len(compressed) != length:
                    raise CheckpointCorruptionError("committed data range was truncated")
                if _digest_bytes(compressed) != commit.get("data_sha256"):
                    raise CheckpointCorruptionError("committed data block checksum mismatch")
                payload = _decompress_member(compressed, uncompressed_size)
                record = _decode_data_block(payload, topology_key, topology_digest, scope, scope_sha256)
                counts = commit.get("entry_counts")
                if (not isinstance(counts, Mapping)
                        or any(_strict_int(counts.get(mode), f"entry_counts[{mode}]") != len(record.identities_by_mode[mode]) for mode in MODE_NAMES)):
                    raise CheckpointCorruptionError("committed entry counts disagree with data")
                for name, values in (("label_enumeration_complete", record.label_enumeration_complete),
                                     ("identity_save_complete", record.identity_save_complete)):
                    raw_values = commit.get(name)
                    if (not isinstance(raw_values, Mapping)
                            or any(_strict_bool(raw_values.get(mode), f"{name}[{mode}]") != values[mode] for mode in MODE_NAMES)):
                        raise CheckpointCorruptionError(f"committed {name} disagrees with data")
                if _strict_bool(commit.get("topology_complete"), "topology_complete") != record.topology_complete:
                    raise CheckpointCorruptionError("committed topology completion disagrees with data")
                if _jsonable(commit.get("audit", {})) != _jsonable(record.audit):
                    raise CheckpointCorruptionError("committed topology audit disagrees with data")
                record = TopologyCheckpoint(record.topology_key, record.topology_digest, record.identities_by_mode,
                                            record.label_enumeration_complete, record.identity_save_complete,
                                            record.topology_complete, offset, length, commit["data_sha256"], record.audit)
                records_seen = record
                last_committed_end = current_range[1]
                seen_topologies.add(token)
                yield records_seen
        finally:
            data_handle.close()

    return scope, expected, scope_sha256, records()


def iter_checkpoint_records(root: str | Path) -> Iterable[TopologyCheckpoint]:
    """Stream committed topology records without loading the data log."""

    _, _, _, records = _iter_checkpoint_records(Path(root))
    return records


def read_checkpoint(root: str | Path) -> CheckpointReadback:
    """Read committed blocks; ignore only an uncommitted EOF index tail."""

    root = Path(root)
    scope, expected, _, records = _iter_checkpoint_records(root)
    return CheckpointReadback(root, scope, expected, tuple(records))


def assert_identity_key_equivalence(
    checkpoint: CheckpointReadback,
    catalogs: Mapping[str, CandidateCatalog | str | Path],
) -> None:
    """Require checkpoint and full catalogs to contain the same canonical keys.

    Source records are deliberately excluded from this comparison: equivalent
    identities may retain different original vertex numbering or other source
    representation details.
    """

    from .label_catalog import load_catalog

    expected: dict[str, set[str]] = {mode: set() for mode in MODE_NAMES}
    for mode, catalog in catalogs.items():
        loaded = load_catalog(catalog) if isinstance(catalog, (str, Path)) else catalog
        if not isinstance(loaded, CandidateCatalog):
            raise TypeError("catalogs must contain CandidateCatalog or catalog paths")
        expected.setdefault(mode, set()).update(_key_token(entry.identity.key) for entry in loaded.entries)
    actual = {mode: set(values) for mode, values in checkpoint.identity_keys_by_mode.items()}
    for mode in set(expected) | set(actual):
        if expected.get(mode, set()) != actual.get(mode, set()):
            raise CheckpointError(f"identity key set mismatch for mode {mode}")


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION", "DATA_FORMAT", "SCOPE_FILENAME", "DATA_FILENAME", "INDEX_FILENAME",
    "CheckpointError", "CheckpointCorruptionError", "CheckpointIdentity",
    "TopologyCheckpoint", "CheckpointReadback", "CatalogCheckpointWriter",
    "read_checkpoint", "iter_checkpoint_records", "assert_identity_key_equivalence",
]
