"""Read-only adapters for the existing five-body candidate archives.

This module is deliberately kept outside the candidate schema.  A candidate
record remains a momentum-only ``IM-5-candidate-v1`` record; this adapter only
rechecks that record and exposes an immutable view for the independent
invariant analysis.  It never enumerates labels, writes to a source archive,
or transports a particle relabeling result to a fixed-particle profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..catalog_checkpoint import (
    CheckpointIdentity,
    CheckpointReadback,
    MODE_NAMES,
    TopologyCheckpoint,
    read_checkpoint,
)
from ..domain import CountMode
from ..label_catalog import CandidateCatalog, CatalogEntry, load_catalog
from ..label_identity import CandidateIdentity, canonicalize_candidate_labeling
from ..physical_labels import CandidateMomentumRecord, review_candidate_record
from .contracts import EvaluationRequest, InvariantOccurrence, PhysicsProfile, ReviewedCandidate


SOURCE_SCHEMA_VERSION = "F5-I-source-v1"
FIXED_MODE = CountMode.FIXED_LABELS.value
RELABELING_MODE = CountMode.PARTICLE_RELABELING.value


class SourceError(ValueError):
    """A source cannot be accepted as an independence input."""


class SourceChangedError(SourceError):
    """A source changed after its immutable read snapshot was captured."""


class UnsupportedSourceError(SourceError):
    """The requested source mode or archive format is not supported."""


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


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _key_token(value: Any) -> str:
    return _json_bytes(value).decode("utf-8")


def _ordered_keys(values: Iterable[Any]) -> tuple[Any, ...]:
    """Return unique complete keys in deterministic serialized-key order."""

    by_token: dict[str, Any] = {}
    for value in values:
        by_token.setdefault(_key_token(value), value)
    return tuple(by_token[token] for token in sorted(by_token))


@dataclass(frozen=True, slots=True)
class SourceFile:
    """The read-only binding of one explicitly named source file."""

    name: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Immutable source binding captured before an evaluation is persisted."""

    source_kind: str
    source_path: str | None
    files: tuple[SourceFile, ...]
    content_sha256: str

    @property
    def fingerprint(self) -> str:
        return _sha256(
            {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "source_kind": self.source_kind,
                "source_path": self.source_path,
                "files": [item.to_dict() for item in self.files],
                "content_sha256": self.content_sha256,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "files": [item.to_dict() for item in self.files],
            "content_sha256": self.content_sha256,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class VerifiedSource:
    """A reviewed candidate identity and its original source record."""

    identity: CandidateIdentity
    source_record: CandidateMomentumRecord
    source_kind: str
    source_index: int | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CandidateIdentity):
            raise TypeError("identity must be CandidateIdentity")
        if not isinstance(self.source_record, CandidateMomentumRecord):
            raise TypeError("source_record must be CandidateMomentumRecord")
        if self.identity.mode.value != FIXED_MODE:
            raise UnsupportedSourceError(
                "only fixed_labels identities are supported; particle relabeling is unsupported"
            )
        # CandidateIdentity.from_dict performs complete key, digest, topology,
        # and map reconstruction.  review_candidate_record recomputes the
        # momentum evidence instead of trusting a persisted verified flag.
        review_candidate_record(self.source_record)
        rebuilt = canonicalize_candidate_labeling(self.source_record, FIXED_MODE)
        if rebuilt.key != self.identity.key or rebuilt.digest != self.identity.digest:
            raise SourceError("source record does not rederive the identity key")
        if rebuilt.canonical_record.to_dict() != self.identity.canonical_record.to_dict():
            raise SourceError("source identity canonical record disagrees with source record")
        if (
            rebuilt.vertex_map != self.identity.vertex_map
            or rebuilt.particle_map != self.identity.particle_map
            or rebuilt.edge_occurrence_map != self.identity.edge_occurrence_map
            or rebuilt.final_slot_map != self.identity.final_slot_map
        ):
            raise SourceError("source identity maps disagree with source rederivation")

    @property
    def mode(self) -> str:
        return self.identity.mode.value

    @property
    def key(self) -> tuple[Any, ...]:
        return self.identity.key

    @property
    def digest(self) -> str:
        return self.identity.digest

    @property
    def topology_key(self) -> tuple[Any, ...]:
        return self.identity.topology_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "source_index": self.source_index,
            "mode": self.mode,
            "identity": self.identity.to_dict(),
            "source_record": self.source_record.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    """Coverage evidence kept separate from per-record validity.

    ``expected_identity_keys`` is populated only after the historical source
    completion contract has been rechecked.  For a partial source it stays
    ``None`` even when a caller supplies an optimistic boolean.
    """

    source_kind: str
    mode: str
    actual_identity_keys: tuple[Any, ...]
    expected_identity_keys: tuple[Any, ...] | None
    actual_topology_keys: tuple[Any, ...]
    expected_topology_keys: tuple[Any, ...] | None
    source_topology_keys: tuple[Any, ...]
    expected_source_topology_keys: tuple[Any, ...] | None
    source_complete: bool
    identity_complete: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode != FIXED_MODE:
            raise UnsupportedSourceError(
                "coverage for particle_relabeling cannot be used for fixed profiles"
            )
        actual = {_key_token(item) for item in self.actual_identity_keys}
        if len(actual) != len(self.actual_identity_keys):
            raise SourceError("source coverage contains duplicate identity keys")
        if self.expected_identity_keys is not None:
            expected = {_key_token(item) for item in self.expected_identity_keys}
            if expected != actual:
                raise SourceError("expected and actual source identity key sets disagree")
            if not self.identity_complete:
                raise SourceError("identity coverage cannot be complete without expected keys")
        if self.expected_topology_keys is not None:
            expected_topologies = {_key_token(item) for item in self.expected_topology_keys}
            actual_topologies = {_key_token(item) for item in self.actual_topology_keys}
            if expected_topologies != actual_topologies:
                raise SourceError("expected and actual source topology key sets disagree")
        source_topologies = _ordered_keys(self.source_topology_keys)
        if self.expected_source_topology_keys is not None:
            expected_source = {_key_token(item) for item in self.expected_source_topology_keys}
            actual_source = {_key_token(item) for item in source_topologies}
            if expected_source != actual_source:
                raise SourceError("expected and actual source topology coverage disagree")
        object.__setattr__(self, "actual_identity_keys", _ordered_keys(self.actual_identity_keys))
        object.__setattr__(self, "actual_topology_keys", _ordered_keys(self.actual_topology_keys))
        object.__setattr__(self, "source_topology_keys", source_topologies)
        if self.expected_identity_keys is not None:
            object.__setattr__(self, "expected_identity_keys", _ordered_keys(self.expected_identity_keys))
        if self.expected_topology_keys is not None:
            object.__setattr__(self, "expected_topology_keys", _ordered_keys(self.expected_topology_keys))
        if self.expected_source_topology_keys is not None:
            object.__setattr__(self, "expected_source_topology_keys", _ordered_keys(self.expected_source_topology_keys))
        object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons))

    @property
    def complete(self) -> bool:
        return self.source_complete and self.identity_complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "source_kind": self.source_kind,
            "mode": self.mode,
            "actual_identity_keys": _jsonable(self.actual_identity_keys),
            "expected_identity_keys": _jsonable(self.expected_identity_keys),
            "actual_topology_keys": _jsonable(self.actual_topology_keys),
            "expected_topology_keys": _jsonable(self.expected_topology_keys),
            "source_topology_keys": _jsonable(self.source_topology_keys),
            "expected_source_topology_keys": _jsonable(self.expected_source_topology_keys),
            "source_complete": self.source_complete,
            "identity_complete": self.identity_complete,
            "complete": self.complete,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class SourceBatch:
    """Immutable source batch consumed by later verification/aggregation."""

    records: tuple[VerifiedSource, ...]
    coverage: SourceCoverage
    snapshot: SourceSnapshot

    def __post_init__(self) -> None:
        if any(not isinstance(item, VerifiedSource) for item in self.records):
            raise TypeError("records must contain VerifiedSource values")
        actual = _ordered_keys(item.key for item in self.records)
        if tuple(_key_token(item) for item in actual) != tuple(
            _key_token(item) for item in self.coverage.actual_identity_keys
        ):
            raise SourceError("source coverage does not match reviewed records")

    @property
    def source_complete(self) -> bool:
        return self.coverage.complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "records": [item.to_dict() for item in self.records],
            "coverage": self.coverage.to_dict(),
            "snapshot": self.snapshot.to_dict(),
        }

    def requests(self, profile: PhysicsProfile) -> tuple[EvaluationRequest, ...]:
        """Build explicit fixed-profile requests while retaining this batch.

        ``EvaluationRequest`` intentionally carries only the compact contract
        fields.  Callers must retain this ``SourceBatch`` alongside the
        request: it is the authoritative full identity/source-record binding
        used by aggregation and certificate verification.
        """

        return tuple(make_evaluation_request(item, profile) for item in self.records)


def _file_snapshot(root: Path, names: Iterable[str]) -> tuple[SourceFile, ...]:
    result: list[SourceFile] = []
    for name in names:
        path = root / name
        try:
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise SourceError(f"cannot read source file: {path}") from exc
        result.append(SourceFile(name, size, digest.hexdigest()))
    return tuple(result)


def _content_digest(records: Iterable[VerifiedSource]) -> str:
    rows = [
        {
            "key": _jsonable(item.key),
            "digest": item.digest,
            "topology_key": _jsonable(item.topology_key),
            "source_record": item.source_record.to_dict(),
        }
        for item in records
    ]
    rows.sort(key=lambda row: _key_token(row["key"]))
    return _sha256(rows)


def _identity_from_public(identity: Any) -> CandidateIdentity:
    if isinstance(identity, CandidateIdentity):
        candidate = CandidateIdentity.from_dict(identity.to_dict())
    elif hasattr(identity, "to_dict"):
        candidate = CandidateIdentity.from_dict(identity.to_dict())
    else:
        raise TypeError("source identity must be a CandidateIdentity-like value")
    if candidate.mode.value != FIXED_MODE:
        raise UnsupportedSourceError(
            "particle_relabeling source identities are unsupported in the first version"
        )
    return candidate


def _verified_identity(identity: Any, source_record: CandidateMomentumRecord,
                       *, source_kind: str, source_index: int | None,
                       source_path: str | None) -> VerifiedSource:
    if not isinstance(source_record, CandidateMomentumRecord):
        raise TypeError("source record must be CandidateMomentumRecord")
    review_candidate_record(source_record)
    candidate = _identity_from_public(identity)
    return VerifiedSource(candidate, source_record, source_kind, source_index, source_path)


def _source_digest(record: CandidateMomentumRecord) -> str:
    return _sha256(record.to_dict())


def _reviewed_candidate(item: VerifiedSource) -> ReviewedCandidate:
    if not isinstance(item, VerifiedSource):
        raise TypeError("item must be a VerifiedSource")
    # Do not use the compact contract's review_status as evidence.  The full
    # source was already re-reviewed in VerifiedSource.__post_init__; repeat
    # it at this boundary so callers cannot manufacture a reviewed wrapper.
    review_candidate_record(item.source_record)
    occurrences = tuple(
        InvariantOccurrence(
            tuple(variable),
            ordinal=index,
            occurrence_id=f"{label.edge_index}:{label.occurrence}",
        )
        for index, (variable, label) in enumerate(
            zip(item.source_record.variables, item.source_record.complete_labeling.double_lines)
        )
    )
    return ReviewedCandidate(
        topology_key=_key_token(item.topology_key),
        identity_key=_key_token(item.key),
        mode=item.mode,
        source_digest=_source_digest(item.source_record),
        occurrences=occurrences,
    )


def make_evaluation_request(item: VerifiedSource, profile: PhysicsProfile) -> EvaluationRequest:
    """Create the foundation request for one fully reviewed source identity."""

    if not isinstance(profile, PhysicsProfile):
        raise TypeError("profile must be PhysicsProfile")
    return EvaluationRequest(profile=profile, candidate=_reviewed_candidate(item))


def _record_batch(records: Iterable[VerifiedSource], *, source_kind: str,
                  source_path: str | None, files: tuple[SourceFile, ...],
                  complete: bool, expected_identity_keys: tuple[Any, ...] | None,
                  expected_topology_keys: tuple[Any, ...] | None,
                  source_topology_keys: tuple[Any, ...] | None = None,
                  expected_source_topology_keys: tuple[Any, ...] | None = None,
                  reasons: Iterable[str] = ()) -> SourceBatch:
    ordered = tuple(sorted(records, key=lambda item: (_key_token(item.topology_key), item.digest)))
    actual_keys = _ordered_keys(item.key for item in ordered)
    actual_topologies = _ordered_keys(item.topology_key for item in ordered)
    identity_complete = complete and expected_identity_keys is not None
    coverage = SourceCoverage(
        source_kind,
        FIXED_MODE,
        actual_keys,
        expected_identity_keys if identity_complete else None,
        actual_topologies,
        expected_topology_keys if complete else None,
        tuple(source_topology_keys or actual_topologies),
        expected_source_topology_keys if complete else None,
        bool(complete),
        bool(identity_complete),
        tuple(reasons),
    )
    snapshot = SourceSnapshot(
        source_kind,
        source_path,
        files,
        _content_digest(ordered),
    )
    return SourceBatch(ordered, coverage, snapshot)


def _catalog_batch(catalog: CandidateCatalog, *, source_path: str | None,
                   files: tuple[SourceFile, ...]) -> SourceBatch:
    if catalog.manifest.count_mode != FIXED_MODE:
        raise UnsupportedSourceError(
            "catalog particle_relabeling mode cannot be transported to fixed particle identities"
        )
    records: list[VerifiedSource] = []
    for entry in catalog.entries:
        if not isinstance(entry, CatalogEntry):
            raise SourceError("candidate catalog contains a non-catalog entry")
        if entry.identity.mode != FIXED_MODE:
            continue
        records.append(
            _verified_identity(
                entry.identity,
                entry.source_record,
                source_kind="catalog",
                source_index=entry.source_index,
                source_path=source_path,
            )
        )
    manifest = catalog.manifest
    actual_digests = {item.digest for item in records}
    expected_digests = set(manifest.expected_identity_digests)
    stored_digests = set(manifest.stored_identity_digests)
    contract = bool(
        source_path is not None
        and manifest.run_status == "complete"
        and manifest.topology_enumeration_complete
        and manifest.label_enumeration_complete
        and manifest.identity_save_complete
        and not manifest.failures
        and len(manifest.expected_identity_digests) == len(expected_digests)
        and len(manifest.stored_identity_digests) == len(stored_digests)
        and expected_digests == stored_digests == actual_digests
    )
    reasons: list[str] = []
    if source_path is None:
        reasons.append("in_memory_source_has_no_immutable_archive_binding")
    if not contract:
        reasons.append("catalog_completion_contract_not_proven")
    expected_keys = _ordered_keys(item.key for item in records) if contract else None
    return _record_batch(
        records,
        source_kind="catalog",
        source_path=source_path,
        files=files,
        complete=contract,
        expected_identity_keys=expected_keys,
        expected_topology_keys=None,
        reasons=reasons,
    )


def _checkpoint_batch(checkpoint: CheckpointReadback, *, source_path: str | None,
                      files: tuple[SourceFile, ...]) -> SourceBatch:
    records: list[VerifiedSource] = []
    for topology in checkpoint.records:
        values = topology.identities_by_mode.get(FIXED_MODE, ())
        for item in values:
            if not isinstance(item, CheckpointIdentity):
                raise SourceError("checkpoint contains an invalid identity row")
            records.append(
                _verified_identity(
                    item.identity,
                    item.source_record,
                    source_kind="checkpoint",
                    source_index=item.source_index,
                    source_path=source_path,
                )
            )
    source_topology_keys = _ordered_keys(item.topology_key for item in checkpoint.records)
    identity_topology_keys = _ordered_keys(item.topology_key for item in records)
    expected_topology_keys = (
        _ordered_keys(checkpoint.expected_topology_keys)
        if checkpoint.expected_topology_keys is not None
        else None
    )
    # ``CheckpointReadback.complete`` is itself recomputed by strict readback
    # from the immutable scope, committed blocks, checksums and identity rows.
    # We additionally require a real archive path; an object assembled by a
    # caller cannot turn an optimistic completion flag into coverage proof.
    contract = bool(
        source_path is not None
        and checkpoint.complete
        and expected_topology_keys is not None
        and {_key_token(item) for item in expected_topology_keys}
        == {_key_token(item) for item in source_topology_keys}
    )
    reasons: list[str] = []
    if source_path is None:
        reasons.append("in_memory_source_has_no_immutable_archive_binding")
    if checkpoint.expected_topology_keys is None:
        reasons.append("checkpoint_has_no_immutable_expected_topology_set")
    if not contract:
        reasons.append("checkpoint_completion_contract_not_proven")
    expected_keys = _ordered_keys(item.key for item in records) if contract else None
    return _record_batch(
        records,
        source_kind="checkpoint",
        source_path=source_path,
        files=files,
        complete=contract,
        expected_identity_keys=expected_keys,
        expected_topology_keys=identity_topology_keys if contract else None,
        source_topology_keys=source_topology_keys,
        expected_source_topology_keys=expected_topology_keys if contract else None,
        reasons=reasons,
    )


def _load_path(path: Path) -> SourceBatch:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise SourceError(f"source path must be an explicit archive directory: {root}")
    catalog_names = ("manifest.json", "identities.jsonl", "identity_index.json")
    checkpoint_names = ("scope.json", "data.jsonl.gz", "index.jsonl")
    present_catalog = all((root / name).is_file() for name in catalog_names)
    present_checkpoint = all((root / name).is_file() for name in checkpoint_names)
    if present_catalog and present_checkpoint:
        raise SourceError("source path matches both catalog and checkpoint formats")
    if present_catalog:
        before = _file_snapshot(root, catalog_names)
        catalog = load_catalog(root)
        after = _file_snapshot(root, catalog_names)
        if before != after:
            raise SourceChangedError("candidate catalog changed during read")
        return _catalog_batch(catalog, source_path=str(root), files=after)
    if present_checkpoint:
        before = _file_snapshot(root, checkpoint_names)
        checkpoint = read_checkpoint(root)
        after = _file_snapshot(root, checkpoint_names)
        if before != after:
            raise SourceChangedError("catalog checkpoint changed during read")
        return _checkpoint_batch(checkpoint, source_path=str(root), files=after)
    raise SourceError(
        "source path does not contain the explicit catalog or checkpoint file set"
    )


def read_source(source: Any, *, mode: str = FIXED_MODE) -> SourceBatch:
    """Read and verify one explicit source.

    The function accepts an already loaded catalog/checkpoint or an explicit
    archive directory.  It does not search parent directories or infer a
    source archive from a run identifier.  A single record/identity is useful
    for local evidence and is always marked incomplete for topology purposes.
    """

    if mode != FIXED_MODE:
        raise UnsupportedSourceError(
            "particle_relabeling is unsupported; no certificate is transported across particle identities"
        )
    if isinstance(source, (str, Path)):
        return _load_path(Path(source))
    if isinstance(source, CandidateCatalog):
        # Only an explicit path goes through the immutable archive reader.  A
        # loaded object is treated as in-memory input even when it carries a
        # source_path field; the path field alone is not a source proof.
        return _catalog_batch(source, source_path=None, files=())
    if isinstance(source, CheckpointReadback):
        return _checkpoint_batch(source, source_path=None, files=())
    if isinstance(source, CandidateMomentumRecord):
        identity = canonicalize_candidate_labeling(source, FIXED_MODE)
        record = _verified_identity(identity, source, source_kind="record", source_index=None, source_path=None)
        return _record_batch(
            (record,),
            source_kind="record",
            source_path=None,
            files=(),
            complete=False,
            expected_identity_keys=None,
            expected_topology_keys=None,
            reasons=("single_record_is_not_a_complete_source",),
        )
    if isinstance(source, CandidateIdentity) or isinstance(source, CatalogEntry):
        identity = source if isinstance(source, CandidateIdentity) else source.identity
        source_record = (
            identity.canonical_record
            if isinstance(source, CandidateIdentity)
            else source.source_record
        )
        record = _verified_identity(
            identity,
            source_record,
            source_kind="identity",
            source_index=getattr(source, "source_index", None),
            source_path=None,
        )
        return _record_batch(
            (record,),
            source_kind="identity",
            source_path=None,
            files=(),
            complete=False,
            expected_identity_keys=None,
            expected_topology_keys=None,
            reasons=("single_identity_is_not_a_complete_source",),
        )
    raise TypeError(
        "source must be an explicit catalog/checkpoint path, CandidateCatalog, "
        "CheckpointReadback, CandidateMomentumRecord, CandidateIdentity, or CatalogEntry"
    )


def verify_source_unchanged(batch: SourceBatch) -> None:
    """Recheck the exact source files and reviewed content fingerprint."""

    if not isinstance(batch, SourceBatch):
        raise TypeError("batch must be a SourceBatch")
    if batch.snapshot.source_path is not None:
        root = Path(batch.snapshot.source_path)
        names = tuple(item.name for item in batch.snapshot.files)
        if not names:
            raise SourceChangedError("file-backed source has no file binding")
        current = _file_snapshot(root, names)
        if current != batch.snapshot.files:
            raise SourceChangedError("source archive files changed after read")
    current_content = _content_digest(batch.records)
    if current_content != batch.snapshot.content_sha256:
        raise SourceChangedError("reviewed source content changed after read")


__all__ = [
    "SOURCE_SCHEMA_VERSION", "FIXED_MODE", "RELABELING_MODE",
    "SourceError", "SourceChangedError", "UnsupportedSourceError",
    "SourceFile", "SourceSnapshot", "VerifiedSource", "SourceCoverage",
    "SourceBatch", "read_source", "verify_source_unchanged",
    "make_evaluation_request",
]
