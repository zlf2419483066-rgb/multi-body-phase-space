"""Streaming adapters for one-witness-per-topology candidate logs.

The L4/L5 content audits persist one JSON object per positive topology.  A
witness row is useful evidence that *one* momentum marking exists, but it is
not a complete fixed-label identity archive.  This module therefore keeps the
row/file binding explicit and never reports full identity coverage.

The adapter is deliberately separate from the candidate schema and from the
production label generators.  It reads JSONL or gzip JSONL, reconstructs the
existing ``CandidateMomentumRecord``, repeats the physical review, derives a
fixed-label ``CandidateIdentity``, and checks the outer topology/key payload.
Malformed rows, review failures, binding failures, and source changes are
structured errors for callers to map to ``unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ..label_identity import CandidateIdentity, canonicalize_candidate_labeling
from ..physical_labels import CandidateMomentumRecord, review_candidate_record
from .sources import FIXED_MODE, RELABELING_MODE, VerifiedSource


WITNESS_SOURCE_SCHEMA_VERSION = "F5-I-witness-source-v1"


class WitnessSourceError(ValueError):
    """A witness row or its immutable source binding cannot be accepted."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
    ) -> None:
        self.code = str(code)
        self.path = path
        self.line = line
        location = ""
        if path is not None:
            location = path
            if line is not None:
                location += f":{line}"
        elif line is not None:
            location = f"line {line}"
        prefix = f"{self.code}"
        if location:
            prefix += f" at {location}"
        super().__init__(f"{prefix}: {message}")


class WitnessSourceChangedError(WitnessSourceError):
    """The witness file changed while it was being read."""


@dataclass(frozen=True, slots=True)
class WitnessRow:
    """One raw witness row with its immutable source location binding.

    ``file_sha256`` is the caller-supplied expected digest when one was
    supplied.  The actual digest is available in ``WitnessFileBinding`` after
    a complete read; a partially consumed iterator cannot claim a final digest.
    """

    source_path: str
    source_index: int
    raw: Mapping[str, Any]
    file_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, str) or not self.source_path:
            raise TypeError("source_path must be a non-empty string")
        if isinstance(self.source_index, bool) or not isinstance(self.source_index, int):
            raise TypeError("source_index must be an integer")
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        if not isinstance(self.raw, Mapping):
            raise TypeError("raw must be a mapping")
        if self.file_sha256 is not None:
            _validate_sha256(self.file_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WITNESS_SOURCE_SCHEMA_VERSION,
            "source_path": self.source_path,
            "source_index": self.source_index,
            "file_sha256": self.file_sha256,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True, slots=True)
class WitnessFileBinding:
    """The digest and line count established by a complete source read."""

    source_path: str
    size: int
    line_count: int
    sha256: str
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.size < 0 or self.line_count < 0:
            raise ValueError("size and line_count must be non-negative")
        _validate_sha256(self.sha256)
        if self.expected_sha256 is not None:
            _validate_sha256(self.expected_sha256)
            if self.sha256 != self.expected_sha256:
                raise WitnessSourceError(
                    "file_sha256_mismatch",
                    f"expected {self.expected_sha256}, got {self.sha256}",
                    path=self.source_path,
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WITNESS_SOURCE_SCHEMA_VERSION,
            "source_path": self.source_path,
            "size": self.size,
            "line_count": self.line_count,
            "sha256": self.sha256,
            "expected_sha256": self.expected_sha256,
        }


@dataclass(frozen=True, slots=True)
class WitnessReadResult:
    """Complete read result; coverage is intentionally never complete."""

    records: tuple[VerifiedSource, ...]
    rows: int
    file: WitnessFileBinding
    topology_keys: tuple[Any, ...]
    identity_keys: tuple[Any, ...]
    expected_topology_keys: tuple[Any, ...] | None
    expected_identity_keys: tuple[Any, ...] | None
    coverage_complete: bool = False
    coverage_reason: str = "one_witness_per_topology_is_not_full_identity_coverage"

    def __post_init__(self) -> None:
        if self.rows != len(self.records):
            raise ValueError("rows must equal the number of verified records")
        if self.coverage_complete:
            raise ValueError("witness reads cannot claim full identity coverage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WITNESS_SOURCE_SCHEMA_VERSION,
            "rows": self.rows,
            "file": self.file.to_dict(),
            "topology_keys": _jsonable(self.topology_keys),
            "identity_keys": _jsonable(self.identity_keys),
            "expected_topology_keys": _jsonable(self.expected_topology_keys),
            "expected_identity_keys": _jsonable(self.expected_identity_keys),
            "coverage_complete": self.coverage_complete,
            "coverage_reason": self.coverage_reason,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def _json_bytes(value: Any, *, sort_keys: bool = True) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=sort_keys,
        separators=(",", ":"),
    ).encode("utf-8")


def _key_token(value: Any) -> str:
    return _json_bytes(value).decode("utf-8")


def _tuplify(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_tuplify(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _tuplify(item) for key, item in value.items()}
    return value


def _validate_sha256(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("sha256 must be a 64-character hexadecimal string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("sha256 must be hexadecimal") from exc
    return value.lower()


def _normalize_expected(values: Iterable[Any] | None, *, name: str) -> tuple[Any, ...] | None:
    if values is None:
        return None
    items = tuple(_tuplify(value) for value in values)
    by_token: dict[str, Any] = {}
    for item in items:
        token = _key_token(item)
        if token in by_token:
            raise WitnessSourceError("duplicate_expected_key", f"{name} contains duplicate keys")
        by_token[token] = item
    return tuple(by_token[token] for token in sorted(by_token))


def _check_stat(path: Path, before: tuple[int, int]) -> None:
    try:
        after_stat = path.stat()
    except OSError as exc:
        raise WitnessSourceChangedError("source_stat_failed", str(exc), path=str(path)) from exc
    after = (after_stat.st_size, after_stat.st_mtime_ns)
    if after != before:
        raise WitnessSourceChangedError(
            "source_changed",
            "source size or modification time changed during read",
            path=str(path),
        )


class _HashingReader:
    """File-like reader that hashes compressed/source bytes as they are read."""

    def __init__(self, handle: Any, digest: Any) -> None:
        self._handle = handle
        self._digest = digest

    def read(self, size: int = -1) -> bytes:
        value = self._handle.read(size)
        if value:
            self._digest.update(value)
        return value

    def read1(self, size: int = -1) -> bytes:
        reader = getattr(self._handle, "read1", None)
        value = reader(size) if reader is not None else self._handle.read(size)
        if value:
            self._digest.update(value)
        return value

    def readinto(self, buffer: Any) -> int:
        count = self._handle.readinto(buffer)
        if count:
            self._digest.update(memoryview(buffer)[:count])
        return count

    def readinto1(self, buffer: Any) -> int:
        reader = getattr(self._handle, "readinto1", None)
        count = reader(buffer) if reader is not None else self._handle.readinto(buffer)
        if count:
            self._digest.update(memoryview(buffer)[:count])
        return count

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return self._handle.seekable()

    def tell(self) -> int:
        return self._handle.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._handle.seek(offset, whence)

    def close(self) -> None:
        # The owner closes the underlying file.  gzip.GzipFile must not close
        # the caller's source handle when it is passed as fileobj.
        return None

    @property
    def closed(self) -> bool:
        return bool(self._handle.closed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def _parse_json_row(value: Any, source: Path, index: int, expected: str | None) -> WitnessRow:
    if not isinstance(value, Mapping):
        raise WitnessSourceError(
            "row_not_object", "each witness row must be a JSON object", path=str(source), line=index + 1
        )
    return WitnessRow(str(source), index, value, expected)


def _iter_witness_rows_internal(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Iterator[WitnessRow]:
    """Internal iterator whose StopIteration value is WitnessFileBinding."""

    source = Path(path).expanduser().resolve()
    expected = _validate_sha256(expected_sha256) if expected_sha256 is not None else None
    try:
        stat = source.stat()
    except OSError as exc:
        raise WitnessSourceError("source_stat_failed", str(exc), path=str(source)) from exc
    before = (stat.st_size, stat.st_mtime_ns)
    digest = hashlib.sha256()
    line_count = 0
    try:
        # The L3 drawing artifact is a small JSON manifest with a ``rows``
        # array; L4/L5 logs are JSONL or gzip JSONL.  Keep the manifest path in
        # the same row/file binding API while retaining its metadata untouched.
        if source.suffix.lower() == ".json":
            try:
                payload = source.read_bytes()
                digest.update(payload)
                document = json.loads(payload.decode("utf-8-sig"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise WitnessSourceError("manifest_read_failed", str(exc), path=str(source)) from exc
            if not isinstance(document, Mapping) or not isinstance(document.get("rows"), list):
                raise WitnessSourceError(
                    "manifest_rows_missing", "JSON witness manifest must contain a rows array", path=str(source)
                )
            for index, value in enumerate(document["rows"]):
                line_count = index + 1
                yield _parse_json_row(value, source, index, expected)
        else:
            with source.open("rb") as raw_handle:
                hashed = _HashingReader(raw_handle, digest)
                binary = gzip.GzipFile(fileobj=hashed, mode="rb") if source.suffix.lower() == ".gz" else hashed
                text_stream = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
                try:
                    for line_number, line in enumerate(text_stream, 1):
                        line_count = line_number
                        if not line.strip():
                            raise WitnessSourceError(
                                "blank_line", "blank JSONL lines are not permitted", path=str(source), line=line_number
                            )
                        try:
                            value = json.loads(line)
                        except (TypeError, ValueError) as exc:
                            raise WitnessSourceError(
                                "invalid_json", str(exc), path=str(source), line=line_number
                            ) from exc
                        yield _parse_json_row(value, source, line_number - 1, expected)
                finally:
                    text_stream.close()
    except WitnessSourceError:
        raise
    except (OSError, UnicodeError, EOFError) as exc:
        raise WitnessSourceError("source_read_failed", str(exc), path=str(source), line=line_count or None) from exc
    _check_stat(source, before)
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise WitnessSourceError(
            "file_sha256_mismatch", f"expected {expected}, got {actual}", path=str(source)
        )
    return WitnessFileBinding(str(source), stat.st_size, line_count, actual, expected)


def iter_witness_rows(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Iterator[WitnessRow]:
    """Stream raw witness rows and verify the final file binding on exhaustion."""

    yield from _iter_witness_rows_internal(path, expected_sha256=expected_sha256)


def _record_from_row(
    row: WitnessRow,
) -> tuple[CandidateMomentumRecord, CandidateIdentity, CandidateIdentity]:
    raw = row.raw
    payload = raw.get("normalized_record")
    if not isinstance(payload, Mapping):
        normalized = raw.get("normalized")
        if isinstance(normalized, Mapping):
            payload = normalized.get("record")
    if not isinstance(payload, Mapping):
        payload = raw.get("source_record")
    if not isinstance(payload, Mapping):
        raise WitnessSourceError(
            "missing_normalized_record",
            "witness row has no normalized_record, normalized.record, or source_record mapping",
            path=row.source_path,
            line=row.source_index + 1,
        )
    try:
        record = CandidateMomentumRecord.from_dict(payload)
    except (TypeError, ValueError, KeyError) as exc:
        raise WitnessSourceError(
            "candidate_record_invalid",
            str(exc),
            path=row.source_path,
            line=row.source_index + 1,
        ) from exc
    try:
        review_candidate_record(record)
    except (TypeError, ValueError, KeyError) as exc:
        raise WitnessSourceError(
            "candidate_record_review_failed",
            str(exc),
            path=row.source_path,
            line=row.source_index + 1,
        ) from exc
    try:
        fixed_identity = canonicalize_candidate_labeling(record, FIXED_MODE)
        identity_mode = row.raw.get("identity_mode", FIXED_MODE)
        if identity_mode not in (FIXED_MODE, RELABELING_MODE):
            raise ValueError(f"unsupported witness identity_mode: {identity_mode!r}")
        source_identity = canonicalize_candidate_labeling(record, identity_mode)
    except (TypeError, ValueError, KeyError) as exc:
        raise WitnessSourceError(
            "identity_rebuild_failed",
            str(exc),
            path=row.source_path,
            line=row.source_index + 1,
        ) from exc
    _check_outer_payload(row, record, fixed_identity, source_identity, payload)
    return record, fixed_identity, source_identity


def _check_outer_payload(
    row: WitnessRow,
    record: CandidateMomentumRecord,
    fixed_identity: CandidateIdentity,
    source_identity: CandidateIdentity,
    payload: Mapping[str, Any],
) -> None:
    raw = row.raw
    expected_topology = record.topology.to_dict()
    for field in ("topology", "source_topology"):
        outer_topology = raw.get(field)
        if outer_topology is not None and _json_bytes(outer_topology) != _json_bytes(expected_topology):
            raise WitnessSourceError(
                "topology_mismatch",
                f"outer {field} differs from the candidate source topology",
                path=row.source_path,
                line=row.source_index + 1,
            )
    if _json_bytes(payload.get("topology")) != _json_bytes(expected_topology):
        raise WitnessSourceError(
            "record_topology_mismatch",
            "candidate source record topology does not rederive from its CandidateMomentumRecord",
            path=row.source_path,
            line=row.source_index + 1,
        )
    normalized = raw.get("normalized")
    normalized_payload = raw.get("normalized_record")
    if isinstance(normalized_payload, Mapping) and isinstance(normalized, Mapping) and isinstance(normalized.get("record"), Mapping):
        if _json_bytes(normalized["record"]) != _json_bytes(normalized_payload):
            raise WitnessSourceError(
                "normalized_alias_mismatch",
                "normalized.record differs from normalized_record",
                path=row.source_path,
                line=row.source_index + 1,
            )
    outer_key_value = raw.get("key", raw.get("topology_key"))
    if outer_key_value is None:
        raise WitnessSourceError(
            "missing_outer_key", "witness row has no topology/full key", path=row.source_path, line=row.source_index + 1
        )
    outer_key = _tuplify(outer_key_value)
    if outer_key not in (fixed_identity.topology_key, fixed_identity.key, source_identity.key):
        raise WitnessSourceError(
            "key_mismatch",
            "outer key is neither the canonical topology key nor the rebuilt source identity key",
            path=row.source_path,
            line=row.source_index + 1,
        )
    if "topology_key" in raw and _tuplify(raw["topology_key"]) != source_identity.topology_key:
        raise WitnessSourceError(
            "topology_key_mismatch", "outer topology_key disagrees with the rebuilt source identity", path=row.source_path, line=row.source_index + 1
        )
    for field in ("identity_key", "full_key", "fullkey"):
        if field in raw and _tuplify(raw[field]) != source_identity.key:
            raise WitnessSourceError(
                "identity_key_mismatch", f"outer {field} disagrees with the rebuilt source identity", path=row.source_path, line=row.source_index + 1
            )
    if "identity_mode" in raw and raw["identity_mode"] != source_identity.mode.value:
        raise WitnessSourceError(
            "identity_mode_mismatch", "outer identity_mode disagrees with the rebuilt source identity", path=row.source_path, line=row.source_index + 1
        )
    if "identity_digest" in raw and raw["identity_digest"] != source_identity.digest:
        raise WitnessSourceError(
            "identity_digest_mismatch", "outer identity_digest disagrees with the rebuilt source identity", path=row.source_path, line=row.source_index + 1
        )
    if "source_record_digest" in raw:
        source_digest = hashlib.sha256(_json_bytes(payload, sort_keys=False)).hexdigest()
        if raw["source_record_digest"] != source_digest:
            raise WitnessSourceError(
                "source_record_digest_mismatch", f"expected {source_digest}, got {raw['source_record_digest']!r}", path=row.source_path, line=row.source_index + 1
            )
    if "canonical_record_digest" in raw:
        canonical_digest = hashlib.sha256(_json_bytes(source_identity.canonical_record.to_dict(), sort_keys=False)).hexdigest()
        if raw["canonical_record_digest"] != canonical_digest:
            raise WitnessSourceError(
                "canonical_record_digest_mismatch", f"expected {canonical_digest}, got {raw['canonical_record_digest']!r}", path=row.source_path, line=row.source_index + 1
            )
    if "key_sha256" in raw:
        actual = hashlib.sha256(_json_bytes(raw["key"], sort_keys=False)).hexdigest()
        if raw["key_sha256"] != actual:
            raise WitnessSourceError(
                "key_sha256_mismatch",
                f"expected {actual}, got {raw['key_sha256']!r}",
                path=row.source_path,
                line=row.source_index + 1,
            )
    if "key_blob_hex" in raw:
        blob = raw["key_blob_hex"]
        if not isinstance(blob, str) or not blob or len(blob) % 2 or any(ch not in "0123456789abcdefABCDEF" for ch in blob):
            raise WitnessSourceError(
                "key_blob_invalid", "key_blob_hex must be non-empty hexadecimal", path=row.source_path, line=row.source_index + 1
            )


def verify_witness_row(row: WitnessRow) -> VerifiedSource:
    """Review one already-read witness row.

    This is intentionally a row-level operation: it rechecks the candidate
    record and rebuilds the fixed-label identity, but it does not make any
    claim about the enclosing file's SHA, completeness, or identity coverage.
    The streaming adapter below delegates to this same constructor so a
    bounded reader and a full reader cannot drift in their row semantics.
    """

    record, identity, _ = _record_from_row(row)
    return VerifiedSource(identity, record, "witness", row.source_index, row.source_path)


def _iter_verified_sources_internal(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_topology_keys: Iterable[Any] | None = None,
    expected_full_keys: Iterable[Any] | None = None,
) -> Iterator[VerifiedSource]:
    """Internal iterator whose StopIteration value is WitnessFileBinding."""

    expected_topology = _normalize_expected(expected_topology_keys, name="expected_topology_keys")
    expected_full = _normalize_expected(expected_full_keys, name="expected_full_keys")
    actual_topology: dict[str, Any] = {}
    actual_full: dict[str, Any] = {}
    raw_rows = _iter_witness_rows_internal(path, expected_sha256=expected_sha256)
    while True:
        try:
            row = next(raw_rows)
        except StopIteration as stop:
            binding = stop.value
            break
        verified = verify_witness_row(row)
        topology_token = _key_token(verified.topology_key)
        identity_token = _key_token(verified.key)
        if topology_token in actual_topology:
            raise WitnessSourceError("duplicate_topology_key", "witness log contains duplicate topology keys", path=row.source_path, line=row.source_index + 1)
        if identity_token in actual_full:
            raise WitnessSourceError("duplicate_identity_key", "witness log contains duplicate identity keys", path=row.source_path, line=row.source_index + 1)
        actual_topology[topology_token] = verified.topology_key
        actual_full[identity_token] = verified.key
        yield verified
    if expected_topology is not None and set(actual_topology) != {_key_token(item) for item in expected_topology}:
        raise WitnessSourceError("topology_coverage_mismatch", "actual and expected witness topology sets differ", path=str(Path(path).expanduser().resolve()))
    if expected_full is not None and set(actual_full) != {_key_token(item) for item in expected_full}:
        raise WitnessSourceError("identity_coverage_mismatch", "actual and expected witness identity sets differ", path=str(Path(path).expanduser().resolve()))
    return binding


def iter_verified_sources(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_topology_keys: Iterable[Any] | None = None,
    expected_full_keys: Iterable[Any] | None = None,
) -> Iterator[VerifiedSource]:
    """Stream reviewed fixed-label sources from a witness log.

    Expected key sets are checked only after EOF.  A caller that stops early
    receives provenance-bearing records but must not use that partial stream as
    a certificate.
    """

    yield from _iter_verified_sources_internal(
        path,
        expected_sha256=expected_sha256,
        expected_topology_keys=expected_topology_keys,
        expected_full_keys=expected_full_keys,
    )


def read_verified_sources(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_topology_keys: Iterable[Any] | None = None,
    expected_full_keys: Iterable[Any] | None = None,
) -> WitnessReadResult:
    """Consume a witness file and return reviewed records plus file binding."""

    expected_topology = _normalize_expected(expected_topology_keys, name="expected_topology_keys")
    expected_full = _normalize_expected(expected_full_keys, name="expected_full_keys")
    source = Path(path).expanduser().resolve()
    verified = _iter_verified_sources_internal(
        source,
        expected_sha256=expected_sha256,
        expected_topology_keys=expected_topology,
        expected_full_keys=expected_full,
    )
    records_list: list[VerifiedSource] = []
    while True:
        try:
            records_list.append(next(verified))
        except StopIteration as stop:
            binding = stop.value
            break
    if not isinstance(binding, WitnessFileBinding):
        raise WitnessSourceError("missing_file_binding", "witness iterator returned no final file binding", path=str(source))
    records = tuple(records_list)
    topology_keys = tuple(sorted((item.topology_key for item in records), key=_key_token))
    identity_keys = tuple(sorted((item.key for item in records), key=_key_token))
    return WitnessReadResult(
        records=records,
        rows=len(records),
        file=binding,
        topology_keys=topology_keys,
        identity_keys=identity_keys,
        expected_topology_keys=expected_topology,
        expected_identity_keys=expected_full,
    )


__all__ = [
    "WITNESS_SOURCE_SCHEMA_VERSION",
    "WitnessSourceError",
    "WitnessSourceChangedError",
    "WitnessRow",
    "WitnessFileBinding",
    "WitnessReadResult",
    "iter_witness_rows",
    "verify_witness_row",
    "iter_verified_sources",
    "read_verified_sources",
]
