"""Durable candidate identity catalog for the five-body workflow.

The catalog stores every fixed-label candidate identity that the caller gives
it.  It also keeps the source record with its original vertex numbering so a
canonical identity can be indexed without forcing drawings to use canonical
coordinates.  No independence or G07 calculation is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from .domain import CountMode
from .physical_labels import CandidateMomentumRecord, review_candidate_record
from .label_identity import CandidateIdentity, canonicalize_candidate_labeling


CATALOG_SCHEMA_VERSION = "F5-candidate-catalog-v1"
IDENTITY_SCHEMA_VERSION = "IM-5-candidate-v1"


class CatalogError(ValueError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    return value


def _digest(key: Any) -> str:
    payload = json.dumps(_jsonable(key), ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_key(record: CandidateMomentumRecord) -> tuple[Any, ...]:
    """Fallback payload key used only for loaded metadata checks.

    The identity module remains the authority for canonicalization.  This
    payload is intentionally complete and preserves edge occurrences.
    """

    topology = record.topology
    labeling = record.complete_labeling
    final = tuple((item.vertex, item.slot, item.particle_id) for item in labeling.final_particles)
    singles = tuple((item.edge_index, item.occurrence, item.particle_id) for item in labeling.single_lines)
    doubles = tuple((item.edge_index, item.occurrence, tuple(item.particle_ids)) for item in labeling.double_lines)
    return (
        IDENTITY_SCHEMA_VERSION, "fixed_labels",
        topology.to_dict(), final, singles, doubles,
    )


@dataclass(frozen=True, slots=True)
class CatalogIdentity:
    """Serializable identity view used after catalog reload."""

    schema_version: str
    mode: str
    topology_key: tuple[Any, ...]
    key: tuple[Any, ...]
    digest: str
    canonical_record: CandidateMomentumRecord
    source_record: CandidateMomentumRecord
    vertex_map: tuple[tuple[int, int], ...] = ()
    particle_map: tuple[tuple[int, int], ...] = ()
    edge_occurrence_map: tuple[tuple[Any, Any], ...] = ()
    final_slot_map: tuple[tuple[Any, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != IDENTITY_SCHEMA_VERSION:
            raise CatalogError(f"unsupported identity schema: {self.schema_version!r}")
        if self.mode not in {item.value for item in CountMode}:
            raise CatalogError(f"unsupported identity count mode: {self.mode!r}")
        if not isinstance(self.canonical_record, CandidateMomentumRecord):
            raise TypeError("canonical_record must be CandidateMomentumRecord")
        if not isinstance(self.source_record, CandidateMomentumRecord):
            raise TypeError("source_record must be CandidateMomentumRecord")
        source_review = review_candidate_record(self.source_record)
        canonical_review = review_candidate_record(self.canonical_record)
        if source_review.record is None or canonical_review.record is None:
            raise CatalogError("catalog identity contains an unreviewed candidate record")
        if _digest(self.key) != self.digest:
            raise CatalogError("identity digest does not match identity key")

    @property
    def original_record(self) -> CandidateMomentumRecord:
        return self.source_record

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "topology_key": _jsonable(self.topology_key),
            "key": _jsonable(self.key),
            "digest": self.digest,
            "canonical_record": self.canonical_record.to_dict(),
            "source_record": self.source_record.to_dict(),
            "vertex_map": [{"original": old, "canonical": new} for old, new in self.vertex_map],
            "particle_map": [{"original": old, "canonical": new} for old, new in self.particle_map],
            "edge_occurrence_map": [{"old": list(old), "new": list(new)} for old, new in self.edge_occurrence_map],
            "final_slot_map": [{"old": list(old), "new": list(new)} for old, new in self.final_slot_map],
        }


@dataclass(frozen=True, slots=True)
class CatalogManifest:
    schema_version: str = CATALOG_SCHEMA_VERSION
    identity_schema: str = IDENTITY_SCHEMA_VERSION
    scope: dict[str, Any] = field(default_factory=dict)
    count_mode: str = CountMode.FIXED_LABELS.value
    marker_scope: str = "all_stored"
    topology_scope: str = "explicit"
    independence: dict[str, Any] = field(default_factory=lambda: {"status": "not_evaluated", "reason": "criterion_deferred"})
    run_status: str = "incomplete"
    topology_enumeration_complete: bool = False
    label_enumeration_complete: bool = False
    identity_save_complete: bool = False
    drawing_selection_complete: bool = False
    expected_identity_digests: tuple[str, ...] = ()
    stored_identity_digests: tuple[str, ...] = ()
    stop_reason: str | None = None
    failures: tuple[dict[str, Any], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CATALOG_SCHEMA_VERSION:
            raise CatalogError(f"unsupported catalog schema: {self.schema_version!r}")
        if self.identity_schema != IDENTITY_SCHEMA_VERSION:
            raise CatalogError(f"unsupported identity schema: {self.identity_schema!r}")
        if self.count_mode not in {mode.value for mode in CountMode}:
            raise CatalogError(f"unsupported catalog count mode: {self.count_mode!r}")
        if self.marker_scope not in {"one", "specified", "all_stored", "selected"}:
            raise CatalogError(f"unsupported marker scope: {self.marker_scope!r}")
        if self.topology_scope not in {"explicit", "fixture", "enumerated", "catalog_subset"}:
            raise CatalogError(f"unsupported topology scope: {self.topology_scope!r}")
        if self.run_status not in {"complete", "incomplete", "failed"}:
            raise CatalogError(f"unsupported catalog run status: {self.run_status!r}")
        if self.run_status == "complete" and self.stop_reason is not None:
            raise CatalogError("complete catalog cannot have a stop_reason")
        if not isinstance(self.independence, Mapping):
            raise CatalogError("catalog independence state must be a mapping")
        if self.independence.get("status") != "not_evaluated":
            raise CatalogError("candidate catalog cannot claim an independence evaluation")
        if self.independence.get("rank") is not None or self.independence.get("certificate") is not None:
            raise CatalogError("candidate catalog cannot contain rank or certificate placeholders")
        for name in (
            "topology_enumeration_complete", "label_enumeration_complete",
            "identity_save_complete", "drawing_selection_complete",
        ):
            if type(getattr(self, name)) is not bool:
                raise CatalogError(f"catalog coverage field {name} must be a boolean")
        for name in ("expected_identity_digests", "stored_identity_digests"):
            values = getattr(self, name)
            if any(type(value) is not str for value in values):
                raise CatalogError(f"catalog {name} must contain strings")
        if not isinstance(self.failures, (tuple, list)):
            raise CatalogError("catalog failures must be a tuple or list of mappings")
        if any(not isinstance(item, Mapping) for item in self.failures):
            raise CatalogError("catalog failures must contain only mappings")
        object.__setattr__(self, "expected_identity_digests", tuple(self.expected_identity_digests))
        object.__setattr__(self, "stored_identity_digests", tuple(self.stored_identity_digests))
        object.__setattr__(self, "failures", tuple(self.failures))

    @property
    def complete(self) -> bool:
        expected = set(self.expected_identity_digests)
        stored = set(self.stored_identity_digests)
        return (
            self.run_status == "complete"
            and self.topology_enumeration_complete
            and self.label_enumeration_complete
            and self.identity_save_complete
            and expected == stored
            and len(expected) == len(self.expected_identity_digests)
            and len(stored) == len(self.stored_identity_digests)
            and not self.failures
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "identity_schema": self.identity_schema,
            "scope": _jsonable(self.scope),
            "count_mode": self.count_mode,
            "marker_scope": self.marker_scope,
            "topology_scope": self.topology_scope,
            "independence": _jsonable(self.independence),
            "run_status": self.run_status,
            "coverage": {
                "topology_enumeration_complete": self.topology_enumeration_complete,
                "label_enumeration_complete": self.label_enumeration_complete,
                "identity_save_complete": self.identity_save_complete,
                "drawing_selection_complete": self.drawing_selection_complete,
            },
            "expected_identity_digests": list(self.expected_identity_digests),
            "stored_identity_digests": list(self.stored_identity_digests),
            "stop_reason": self.stop_reason,
            "failures": _jsonable(self.failures),
        }
        result.update(_jsonable(self.extra))
        return result


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    identity: CatalogIdentity
    source_record: CandidateMomentumRecord
    source_index: int

    @property
    def key(self) -> tuple[Any, ...]:
        return self.identity.key

    @property
    def digest(self) -> str:
        return self.identity.digest

    @property
    def topology_key(self) -> tuple[Any, ...]:
        return self.identity.topology_key


@dataclass(frozen=True, slots=True)
class CandidateCatalog:
    manifest: CatalogManifest
    entries: tuple[CatalogEntry, ...]
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, CatalogManifest):
            raise TypeError("catalog manifest must be CatalogManifest")
        seen: set[str] = set()
        for entry in self.entries:
            if not isinstance(entry, CatalogEntry):
                raise TypeError("catalog entries must be CatalogEntry objects")
            if (not isinstance(entry.source_record, CandidateMomentumRecord)
                    or entry.source_record.to_dict() != entry.identity.source_record.to_dict()):
                raise CatalogError("catalog entry source_record disagrees with identity source_record")
            if entry.digest in seen:
                raise CatalogError(f"duplicate identity digest: {entry.digest}")
            seen.add(entry.digest)
        stored = tuple(self.manifest.stored_identity_digests)
        if set(stored) != seen or len(stored) != len(seen):
            raise CatalogError("manifest stored identity coverage does not match catalog entries")
        if self.manifest.identity_save_complete:
            expected = tuple(self.manifest.expected_identity_digests)
            if set(expected) != seen or len(expected) != len(seen):
                raise CatalogError("manifest expected identity coverage does not match catalog entries")

    @property
    def complete(self) -> bool:
        return self.manifest.complete

    def by_digest(self, digest: str) -> CatalogEntry:
        for entry in self.entries:
            if entry.digest == digest:
                return entry
        raise KeyError(digest)

    def to_dict(self) -> dict[str, Any]:
        return {"manifest": self.manifest.to_dict(), "entries": [entry.identity.to_dict() for entry in self.entries]}


def _identity_fields(identity: Any, source_record: CandidateMomentumRecord) -> CatalogIdentity:
    if not isinstance(identity, CandidateIdentity):
        raise TypeError("identity must be CandidateIdentity")
    # Reparse the public identity payload so map truncation or a fabricated
    # canonical record cannot enter the catalog through build_catalog.
    validated = CandidateIdentity.from_dict(identity.to_dict())
    canonical = getattr(identity, "canonical_record", None)
    if canonical is None:
        canonical = source_record
    if not isinstance(canonical, CandidateMomentumRecord):
        raise TypeError("identity canonical_record must be CandidateMomentumRecord")
    checked = review_candidate_record(canonical)
    if checked.record is None:
        raise CatalogError("identity canonical record failed candidate review")
    schema = str(getattr(identity, "schema_version", ""))
    if schema != IDENTITY_SCHEMA_VERSION:
        raise CatalogError(f"unsupported identity schema: {schema!r}")
    key = tuple(getattr(identity, "key", _record_key(canonical)))
    if not hasattr(identity, "digest") or not hasattr(identity, "mode") or not hasattr(identity, "topology_key"):
        raise CatalogError("identity is missing canonical identity fields")
    digest = str(identity.digest)
    mode = getattr(identity.mode, "value", identity.mode)
    if mode not in {item.value for item in CountMode}:
        raise CatalogError(f"unsupported identity count mode: {mode!r}")
    topology_key = tuple(identity.topology_key)
    if _digest(key) != digest:
        raise CatalogError("identity digest does not match key")
    from .label_identity import canonicalize_candidate_labeling
    rebuilt = canonicalize_candidate_labeling(source_record, mode)
    if rebuilt.key != key or rebuilt.digest != digest or rebuilt.topology_key != topology_key:
        raise CatalogError("identity does not rebuild from its source candidate record")
    if rebuilt.canonical_record.to_dict() != canonical.to_dict():
        raise CatalogError("identity canonical record differs from rebuilt source identity")
    if (tuple(rebuilt.vertex_map) != tuple(identity.vertex_map)
            or tuple(rebuilt.particle_map) != tuple(identity.particle_map)
            or tuple(rebuilt.edge_occurrence_map) != tuple(identity.edge_occurrence_map)
            or tuple(rebuilt.final_slot_map) != tuple(identity.final_slot_map)):
        raise CatalogError("identity source-to-canonical maps differ from rebuilt identity")
    return CatalogIdentity(validated.schema_version, str(mode), topology_key, key, digest,
                           validated.canonical_record, source_record,
                           tuple(validated.vertex_map), tuple(validated.particle_map),
                           tuple(validated.edge_occurrence_map), tuple(validated.final_slot_map))


def build_catalog(identities: Iterable[Any], *, source_records: Mapping[str, CandidateMomentumRecord] | None = None,
                  manifest: CatalogManifest | None = None) -> CandidateCatalog:
    entries: list[CatalogEntry] = []
    source_records = source_records or {}
    for index, identity in enumerate(identities):
        canonical = getattr(identity, "canonical_record", None)
        if canonical is None:
            raise CatalogError("identity lacks canonical_record")
        source = source_records.get(str(getattr(identity, "digest", "")), canonical)
        if not isinstance(source, CandidateMomentumRecord):
            raise TypeError("source record must be CandidateMomentumRecord")
        reviewed = review_candidate_record(source)
        if reviewed.record is None:
            raise CatalogError("source record failed candidate review")
        value = _identity_fields(identity, source)
        entries.append(CatalogEntry(value, source, index))
    entries.sort(key=lambda entry: (entry.topology_key, entry.digest))
    if manifest is None:
        digests = tuple(entry.digest for entry in entries)
        # An explicit list of records does not prove that the requested search
        # domain was exhausted.  Callers must provide the enumeration coverage
        # manifest when they want a catalog to be complete.
        manifest = CatalogManifest(run_status="incomplete", marker_scope="selected",
                                   topology_scope="explicit", expected_identity_digests=digests,
                                   stored_identity_digests=digests,
                                   stop_reason="explicit_identity_selection")
    return CandidateCatalog(manifest, tuple(entries))


def write_catalog(catalog: CandidateCatalog, output_dir: str | Path, *, overwrite: bool = False) -> Path:
    if not isinstance(catalog, CandidateCatalog):
        raise TypeError("catalog must be CandidateCatalog")
    stored = tuple(entry.digest for entry in catalog.entries)
    if set(catalog.manifest.stored_identity_digests) != set(stored) or len(catalog.manifest.stored_identity_digests) != len(stored):
        raise CatalogError("manifest stored identity coverage does not match records")
    root = Path(output_dir)
    if root.exists() and not overwrite:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=overwrite)
    records_path = root / "identities.jsonl"
    index_path = root / "identity_index.json"
    manifest_path = root / "manifest.json"
    lines = []
    index = []
    for entry in catalog.entries:
        row = entry.identity.to_dict()
        row["source_index"] = entry.source_index
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        index.append({"digest": entry.digest, "key": _jsonable(entry.key), "line": len(index)})
    payloads = {
        records_path: "\n".join(lines) + ("\n" if lines else ""),
        index_path: json.dumps(index, ensure_ascii=False, indent=2),
        manifest_path: json.dumps(catalog.manifest.to_dict(), ensure_ascii=False, indent=2),
    }
    for target, payload in payloads.items():
        # Do not rewrite stable identity/index checkpoints when only drawing
        # coverage changed.  Every changed file is replaced atomically from a
        # same-directory temporary file, so interruption cannot truncate the
        # previous complete file.
        if target.exists() and target.read_text(encoding="utf-8") == payload:
            continue
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                             prefix=f".{target.name}.", suffix=".tmp",
                                             delete=False) as handle:
                temporary_name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
    return root


def _manifest(data: Mapping[str, Any]) -> CatalogManifest:
    if not isinstance(data, Mapping):
        raise CatalogError("manifest must be a mapping")
    coverage = data.get("coverage", {})
    if not isinstance(coverage, Mapping):
        raise CatalogError("manifest coverage must be a mapping")
    def strict_bool(name: str, default: bool = False) -> bool:
        value = coverage.get(name, default)
        if type(value) is not bool:
            raise CatalogError(f"manifest coverage field {name} must be a boolean")
        return value
    scope = data.get("scope", {})
    independence = data.get("independence", {})
    if not isinstance(scope, Mapping) or not isinstance(independence, Mapping):
        raise CatalogError("manifest scope and independence must be mappings")
    expected = data.get("expected_identity_digests", ())
    stored = data.get("stored_identity_digests", ())
    if not isinstance(expected, list) or not isinstance(stored, list):
        raise CatalogError("manifest identity digest lists must be arrays")
    if any(type(item) is not str for item in (*expected, *stored)):
        raise CatalogError("manifest identity digests must be strings")
    return CatalogManifest(
        schema_version=data.get("schema_version", ""), identity_schema=data.get("identity_schema", ""),
        scope=dict(scope), count_mode=data.get("count_mode", CountMode.FIXED_LABELS.value),
        marker_scope=data.get("marker_scope", "selected"), topology_scope=data.get("topology_scope", "explicit"),
        independence=dict(independence), run_status=data.get("run_status", "incomplete"),
        topology_enumeration_complete=strict_bool("topology_enumeration_complete"),
        label_enumeration_complete=strict_bool("label_enumeration_complete"),
        identity_save_complete=strict_bool("identity_save_complete"),
        drawing_selection_complete=strict_bool("drawing_selection_complete"),
        expected_identity_digests=tuple(expected), stored_identity_digests=tuple(stored),
        stop_reason=data.get("stop_reason"), failures=data.get("failures", ()),
        extra={key: value for key, value in data.items() if key not in {"schema_version", "identity_schema", "scope", "count_mode", "marker_scope", "topology_scope", "independence", "run_status", "coverage", "expected_identity_digests", "stored_identity_digests", "stop_reason", "failures"}},
    )


def load_catalog(path: str | Path) -> CandidateCatalog:
    root = Path(path)
    try:
        manifest = _manifest(json.loads((root / "manifest.json").read_text(encoding="utf-8")))
        raw_rows = (root / "identities.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in raw_rows if line.strip()]
        index = json.loads((root / "identity_index.json").read_text(encoding="utf-8"))
        if not isinstance(index, list) or len(index) != len(rows):
            raise CatalogError("identity index length does not match identities.jsonl")
    except (OSError, json.JSONDecodeError, CatalogError, KeyError) as exc:
        raise CatalogError(f"cannot read candidate catalog: {root}") from exc
    entries = []
    for row_index, row in enumerate(rows):
        try:
            if not isinstance(row, Mapping):
                raise CatalogError("identity row must be a mapping")
            index_row = index[row_index]
            if (not isinstance(index_row, Mapping) or type(index_row.get("line")) is not int
                    or index_row.get("line") != row_index):
                raise CatalogError(f"identity index line mismatch at row {row_index}")
            canonical = CandidateMomentumRecord.from_dict(row["canonical_record"])
            if "source_record" not in row:
                raise CatalogError("identity row is missing required source_record")
            source = CandidateMomentumRecord.from_dict(row["source_record"])
            source_review = review_candidate_record(source)
            if source_review.record is None:
                raise CatalogError("source record failed candidate review")
            parsed = CandidateIdentity.from_dict(row)
            identity = CatalogIdentity(
                parsed.schema_version, parsed.mode.value, parsed.topology_key,
                parsed.key, parsed.digest, parsed.canonical_record, source,
                parsed.vertex_map, parsed.particle_map,
                parsed.edge_occurrence_map, parsed.final_slot_map,
            )
            rebuilt = canonicalize_candidate_labeling(identity.source_record, identity.mode)
            if rebuilt.key != identity.key or rebuilt.digest != identity.digest:
                raise CatalogError(f"identity row {row_index} does not rebuild from source record")
            if rebuilt.canonical_record.to_dict() != identity.canonical_record.to_dict():
                raise CatalogError(f"identity row {row_index} canonical record mismatch")
            if (tuple(rebuilt.vertex_map) != tuple(identity.vertex_map)
                    or tuple(rebuilt.particle_map) != tuple(identity.particle_map)
                    or tuple(rebuilt.edge_occurrence_map) != tuple(identity.edge_occurrence_map)
                    or tuple(rebuilt.final_slot_map) != tuple(identity.final_slot_map)):
                raise CatalogError(f"identity row {row_index} source-to-canonical maps mismatch")
            if index_row.get("digest") != identity.digest or _freeze(index_row.get("key")) != identity.key:
                raise CatalogError(f"identity index row {row_index} does not match identity")
            source_index = row.get("source_index", row_index)
            if type(source_index) is not int or source_index < 0:
                raise CatalogError("negative source index")
            entries.append(CatalogEntry(identity, source, source_index))
        except (KeyError, TypeError, ValueError, IndexError, CatalogError) as exc:
            raise CatalogError(f"invalid candidate identity row {row_index}") from exc
    stored = tuple(entry.digest for entry in entries)
    if len(set(stored)) != len(stored):
        raise CatalogError("duplicate identity digest in records")
    if set(manifest.stored_identity_digests) != set(stored) or len(manifest.stored_identity_digests) != len(stored):
        raise CatalogError("manifest stored identity coverage does not match records")
    if manifest.identity_save_complete and (set(manifest.expected_identity_digests) != set(stored) or len(manifest.expected_identity_digests) != len(stored)):
        raise CatalogError("identity completeness is claimed but expected identities are missing")
    return CandidateCatalog(manifest, tuple(entries), str(root))


__all__ = [
    "CATALOG_SCHEMA_VERSION", "IDENTITY_SCHEMA_VERSION", "CatalogError", "CatalogIdentity",
    "CatalogManifest", "CatalogEntry", "CandidateCatalog", "build_catalog", "write_catalog", "load_catalog",
]
