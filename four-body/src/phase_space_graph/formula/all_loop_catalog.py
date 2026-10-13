"""Read-only fixed-identity catalog for the archived four-body L0--L3 sources.

This module reads frozen G08 audits for L0/L1/L2 and the frozen fixed-identity
gzip archive for L3.  It does not run topology or labeling enumeration.  The
only identity rules used here are the existing physical normalizer and
``canonicalize_physical_labeling`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..domain import CountMode, Labeling, Topology
from ..drawing.archive import ArchivedIdentity, load_identity_archive
from ..label_identity import MarkedIdentity, canonicalize_physical_labeling
from ..physical_labels import PhysicalLabelRecord, normalize_physical_labeling
from . import d5_catalog as _d5


G08_AUDIT_RELATIVE_TEMPLATE = Path(
    "文档/验证与测试/结果图表/四体图枚举/g08_四体_l{loop}_串联审计.json"
)
L3_ARCHIVE_RELATIVE_PATH = _d5.D5_ARCHIVE_RELATIVE_PATH

EXPECTED_TOPOLOGY_COUNTS: dict[int, int] = {0: 2, 1: 5, 2: 14, 3: 33}
EXPECTED_FIXED_COUNTS: dict[int, int] = {0: 15, 1: 120, 2: 684, 3: 2544}
EXPECTED_CLASS_COUNTS: dict[int, int] = {0: 2, 1: 6, 2: 29, 3: 106}
EXPECTED_MEMBER_DISTRIBUTIONS: dict[int, tuple[int, ...]] = {
    0: (3, 12),
    1: (12, 12, 24, 24, 24, 24),
    2: (12,) + (24,) * 28,
    3: (24,) * 106,
}
EXPECTED_SOURCE_SHA256: dict[int, str] = {
    0: "a2cbe8ed1d425c674f1afdb5f5a10b7e5101fe3efec29929b44f07b4c97ad8f3",
    1: "0bca1d4119efaa3b2142c679255ff2948020e55a702aaf0ebe755488d91cc60f",
    2: "282328dc1e48c23f8b28ec323121cf5d4e14214715f0043bb5a2931dfd147c0c",
    3: "4f1c2253ef098e4c02bd929986bc5c0e8e6707f99d8500ad516121671aa06878",
}


def _jsonable(value: Any) -> Any:
    return _d5._jsonable(value)


def _canonical_json(value: Any) -> str:
    return _d5._canonical_json(value)


def _loop_count(topology: Topology) -> int:
    return topology.edge_count - len(topology.vertices) + 1


def _raw_topology_key(topology: Topology) -> str:
    return _canonical_json(topology.to_dict())


def default_g08_audit_path(loop_count: int) -> Path:
    if isinstance(loop_count, bool) or not isinstance(loop_count, int) or loop_count not in (0, 1, 2):
        raise ValueError("G08 audit paths exist only for loop counts 0, 1, and 2")
    demo_root = Path(__file__).resolve().parents[3]
    return demo_root / str(G08_AUDIT_RELATIVE_TEMPLATE).format(loop=loop_count)


def default_l3_archive_path() -> Path:
    demo_root = Path(__file__).resolve().parents[3]
    return demo_root / L3_ARCHIVE_RELATIVE_PATH


@dataclass(frozen=True, slots=True)
class AllLoopSourceIdentity:
    """One source fixed identity with its canonical form and archive locator."""

    archived: ArchivedIdentity
    fixed_identity: MarkedIdentity
    source_locator: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AllLoopCatalogMember:
    """An archived fixed identity plus its explicit map to a class representative."""

    archived: ArchivedIdentity
    source_locator: dict[str, Any]
    map_to_representative: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.archived.program_id,
            "fixed_key": _jsonable(self.archived.key),
            "fixed_digest": self.archived.digest,
            "source_locator": _jsonable(self.source_locator),
            "map_to_representative": self.map_to_representative,
        }


@dataclass(frozen=True, slots=True)
class AllLoopMarkedClass:
    """One particle-relabeling class with every archived fixed member."""

    class_id: str
    loop_count: int
    topology_id: str
    topology_key: tuple[Any, ...]
    topology_digest: str
    particle_identity: MarkedIdentity
    representative: ArchivedIdentity
    representative_source_locator: dict[str, Any]
    representative_particle_identity: MarkedIdentity
    members: tuple[AllLoopCatalogMember, ...]
    legacy_d5_class_id: str | None = None

    @property
    def representative_record(self) -> PhysicalLabelRecord:
        return self.representative.canonical_record

    @property
    def representative_fixed_digest(self) -> str:
        return self.representative.digest

    @property
    def fixed_member_count(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "legacy_d5_class_id": self.legacy_d5_class_id,
            "loop_count": self.loop_count,
            "topology_id": self.topology_id,
            "topology_key": _jsonable(self.topology_key),
            "topology_digest": self.topology_digest,
            "particle_relabeling_key": _jsonable(self.particle_identity.key),
            "particle_relabeling_digest": self.particle_identity.digest,
            "representative_policy": (
                "archived fixed record matching the canonical particle-relabeling record; "
                "deterministic archived member fallback if absent"
            ),
            "representative_program_id": self.representative.program_id,
            "representative_source_locator": _jsonable(self.representative_source_locator),
            "representative_fixed_key": _jsonable(self.representative.key),
            "representative_fixed_digest": self.representative.digest,
            "representative_record": self.representative.canonical_record.to_dict(),
            "fixed_member_count": self.fixed_member_count,
            "members": [member.to_dict() for member in self.members],
        }


@dataclass(frozen=True, slots=True)
class AllLoopCatalog:
    """Validated grouping of the frozen L0/L1/L2 G08 and L3 identity sources."""

    archive_paths: tuple[tuple[int, Path], ...]
    archive_sha256: tuple[tuple[int, str], ...]
    fixed_record_count: int
    topology_count: int
    classes: tuple[AllLoopMarkedClass, ...]

    def to_dict(self) -> dict[str, Any]:
        hashes = dict(self.archive_sha256)
        loop_metadata = []
        for loop, _ in self.archive_paths:
            loop_classes = tuple(item for item in self.classes if item.loop_count == loop)
            loop_metadata.append({
                "loop_count": loop,
                "fixed_identity_count": sum(item.fixed_member_count for item in loop_classes),
                "topology_count": len({item.topology_id for item in loop_classes}),
                "particle_relabeling_class_count": len(loop_classes),
            })
        return {
            "artifact_type": "four_body_all_loop_identity_catalog",
            "schema_version": "all-loop-catalog-v1",
            "sources": [
                {"loop_count": loop, "path": str(path), "sha256": hashes[loop]}
                for loop, path in self.archive_paths
            ],
            "loop_counts": loop_metadata,
            "grouping": (
                "canonicalize_physical_labeling(record, CountMode.PARTICLE_RELABELING); "
                "full canonical key is the group identity"
            ),
            "fixed_identity_count": self.fixed_record_count,
            "topology_count": self.topology_count,
            "particle_relabeling_class_count": len(self.classes),
            "representative_map_direction": (
                "each archived fixed record → the selected archived fixed representative "
                "of its particle-relabeling class"
            ),
            "classes": [item.to_dict() for item in self.classes],
        }


def _read_g08_fixed_identities(
    loop_count: int,
    payload: Mapping[str, Any],
    source_path: str | Path,
) -> tuple[AllLoopSourceIdentity, ...]:
    """Rebuild fixed identities from saved G08 valid rows without enumeration."""

    if loop_count not in (0, 1, 2):
        raise ValueError("G08 fixed-identity reader supports loop counts 0, 1, and 2")
    if not isinstance(payload, Mapping):
        raise ValueError("G08 audit root must be a mapping")
    if payload.get("status") != "complete" or payload.get("complete") is not True:
        raise ValueError(f"L{loop_count} G08 audit is not complete")

    topology_report = payload.get("topology_report")
    if not isinstance(topology_report, Mapping):
        raise ValueError("G08 audit is missing its topology report")
    if (topology_report.get("status") != "complete"
            or topology_report.get("n") != 4
            or topology_report.get("loop_count") != loop_count):
        raise ValueError(f"L{loop_count} G08 topology scope is incomplete or inconsistent")
    raw_topologies = topology_report.get("topologies")
    reports = payload.get("labeling_reports")
    if not isinstance(raw_topologies, list) or not isinstance(reports, list):
        raise ValueError("G08 audit must contain topology and labeling-report lists")

    topologies: dict[str, Topology] = {}
    for raw_topology in raw_topologies:
        if not isinstance(raw_topology, Mapping):
            raise ValueError("G08 topology scope contains a malformed topology")
        topology = Topology.from_dict(raw_topology)
        if topology.n != 4 or _loop_count(topology) != loop_count:
            raise ValueError(f"L{loop_count} topology scope contains another n or loop count")
        key = _raw_topology_key(topology)
        if key in topologies:
            raise ValueError("G08 topology scope contains a duplicate topology record")
        topologies[key] = topology

    if len(reports) != len(topologies):
        raise ValueError("G08 labeling reports contain duplicate, repeated, or omitted topology records")
    report_topologies: dict[str, Topology] = {}
    for report in reports:
        if not isinstance(report, Mapping) or not isinstance(report.get("topology"), Mapping):
            raise ValueError("G08 labeling report has no topology record")
        topology = Topology.from_dict(report["topology"])
        key = _raw_topology_key(topology)
        if key in report_topologies:
            raise ValueError("G08 labeling reports contain a duplicate topology report")
        report_topologies[key] = topology
    if set(report_topologies) != set(topologies):
        raise ValueError("G08 labeling reports do not match the declared topology scope")

    # These aggregate fields are source completeness checks; the catalog's
    # topology count is separately measured from canonical identity keys.
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("G08 audit is missing aggregate counts")
    if counts.get("topologies") != len(topologies):
        raise ValueError("G08 topology count does not match its saved topology records")
    if counts.get("fixed_label_classes") != EXPECTED_FIXED_COUNTS[loop_count]:
        raise ValueError("G08 fixed-label count differs from the frozen full-scope count")

    identities: list[AllLoopSourceIdentity] = []
    seen_keys: set[tuple[Any, ...]] = set()
    seen_digests: set[str] = set()
    source = Path(source_path).resolve()
    for report_index, report in enumerate(reports, start=1):
        topology = report_topologies[_raw_topology_key(Topology.from_dict(report["topology"]))]
        if (_loop_count(topology) != loop_count or topology.n != 4
                or report.get("status") != "complete"
                or report.get("count_mode") != CountMode.FIXED_LABELS.value):
            raise ValueError(f"L{loop_count} labeling report {report_index} is incomplete or has the wrong mode")
        raw_rows = report.get("valid_results")
        report_counts = report.get("counts")
        if not isinstance(raw_rows, list) or not isinstance(report_counts, Mapping):
            raise ValueError(f"L{loop_count} labeling report {report_index} lacks saved valid rows/counts")
        if report_counts.get("unique") != len(raw_rows):
            raise ValueError(f"L{loop_count} labeling report {report_index} has an incomplete valid-row list")

        for row_index, raw_row in enumerate(raw_rows, start=1):
            if not isinstance(raw_row, Mapping) or raw_row.get("valid") is not True:
                raise ValueError(f"L{loop_count} report {report_index} has a non-valid row in valid_results")
            try:
                labeling = Labeling.from_dict(raw_row["labeling"])
                normalization = normalize_physical_labeling(
                    topology, labeling, CountMode.FIXED_LABELS
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"L{loop_count} report {report_index} valid row {row_index} cannot be normalized"
                ) from exc
            if normalization.stage != "accepted" or normalization.record is None:
                raise ValueError(
                    f"L{loop_count} report {report_index} valid row {row_index} failed current normalization"
                )
            fixed_identity = canonicalize_physical_labeling(
                normalization.record, CountMode.FIXED_LABELS
            )
            stored_canonical = raw_row.get("canonical")
            if not isinstance(stored_canonical, Mapping):
                raise ValueError(f"L{loop_count} valid row is missing its saved canonical identity")
            # G08's saved ``canonical`` field is the older G04 labeled-topology
            # identity, while this catalog uses the existing IM-4 identity.
            # Keep the G08 evidence as provenance, but do not compare keys or
            # digests across those distinct schemas.
            if (stored_canonical.get("mode") != CountMode.FIXED_LABELS.value
                    or not isinstance(stored_canonical.get("digest"), str)
                    or "canonical_key" not in stored_canonical):
                raise ValueError(f"L{loop_count} valid row has malformed G08 canonical evidence")
            if fixed_identity.key in seen_keys or fixed_identity.digest in seen_digests:
                raise ValueError(
                    f"L{loop_count} G08 source repeats fixed identity {fixed_identity.digest}"
                )
            seen_keys.add(fixed_identity.key)
            seen_digests.add(fixed_identity.digest)
            program_id = f"G08-L{loop_count}-R{report_index:04d}-V{row_index:04d}"
            archived = ArchivedIdentity(
                program_id=program_id,
                topology_key=fixed_identity.topology_key,
                key=fixed_identity.key,
                digest=fixed_identity.digest,
                canonical_record=fixed_identity.canonical_record,
                source_path=str(source),
            )
            identities.append(AllLoopSourceIdentity(
                archived=archived,
                fixed_identity=fixed_identity,
                source_locator={
                    "kind": "g08_valid_result",
                    "path": str(source),
                    "loop_count": loop_count,
                    "labeling_report_index": report_index,
                    "valid_result_index": row_index,
                    "source_topology": topology.to_dict(),
                    "g08_canonical_digest": stored_canonical["digest"],
                    "g08_canonical_key": stored_canonical["canonical_key"],
                },
            ))

    if len(identities) != EXPECTED_FIXED_COUNTS[loop_count]:
        raise ValueError(
            f"L{loop_count} expected {EXPECTED_FIXED_COUNTS[loop_count]} fixed identities, "
            f"got {len(identities)}"
        )
    return tuple(identities)


def _read_frozen_g08(loop_count: int, path: Path) -> tuple[tuple[AllLoopSourceIdentity, ...], str]:
    source = path.resolve()
    raw = source.read_bytes()
    digest_before = hashlib.sha256(raw).hexdigest()
    if digest_before != EXPECTED_SOURCE_SHA256[loop_count]:
        raise ValueError(f"L{loop_count} G08 audit SHA-256 differs from the frozen coverage source")
    try:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"L{loop_count} G08 audit is not valid UTF-8 JSON") from exc
        result = _read_g08_fixed_identities(loop_count, payload, source)
    finally:
        digest_after = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest_before != digest_after:
            raise ValueError(f"L{loop_count} G08 audit changed while the catalog was being built")
    return result, digest_before


def _read_frozen_l3(path: Path) -> tuple[tuple[AllLoopSourceIdentity, ...], str]:
    source = path.resolve()
    digest_before = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest_before != EXPECTED_SOURCE_SHA256[3]:
        raise ValueError("L3 fixed-identity archive SHA-256 differs from the frozen coverage source")
    try:
        archived_rows = load_identity_archive(source)
    finally:
        digest_after = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest_before != digest_after:
            raise ValueError("L3 fixed-identity archive changed while the catalog was being built")
    if len(archived_rows) != EXPECTED_FIXED_COUNTS[3]:
        raise ValueError(f"expected {EXPECTED_FIXED_COUNTS[3]} L3 fixed records, got {len(archived_rows)}")
    if (len({row.key for row in archived_rows}) != len(archived_rows)
            or len({row.digest for row in archived_rows}) != len(archived_rows)):
        raise ValueError("L3 fixed-identity archive contains a repeated full fixed identity")

    result: list[AllLoopSourceIdentity] = []
    for row_index, archived in enumerate(archived_rows, start=1):
        fixed_identity = canonicalize_physical_labeling(
            archived.canonical_record, CountMode.FIXED_LABELS
        )
        if (fixed_identity.key != archived.key or fixed_identity.digest != archived.digest
                or fixed_identity.topology_key != archived.topology_key):
            raise ValueError(f"L3 fixed identity mismatch for {archived.program_id}/{archived.digest}")
        result.append(AllLoopSourceIdentity(
            archived=archived,
            fixed_identity=fixed_identity,
            source_locator={
                "kind": "fixed_identity_archive_row",
                "path": str(source),
                "loop_count": 3,
                "archive_row_index": row_index,
                "program_id": archived.program_id,
            },
        ))
    return tuple(result), digest_before


def _normalize_loops(loops: int | Iterable[int] | None) -> tuple[int, ...]:
    if loops is None:
        return (0, 1, 2, 3)
    if isinstance(loops, bool):
        raise TypeError("loops must be an integer or iterable of integers")
    if isinstance(loops, int):
        values = (loops,)
    else:
        if isinstance(loops, (str, bytes)):
            raise TypeError("loops must contain integer loop counts")
        try:
            values = tuple(loops)
        except TypeError as exc:
            raise TypeError("loops must be an integer or iterable of integers") from exc
    if not values:
        raise ValueError("loops must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2, 3)
           for value in values):
        raise ValueError("loop counts must be integers from 0 through 3")
    if len(set(values)) != len(values):
        raise ValueError("loop counts must not contain duplicates")
    return tuple(sorted(values))


def _source_path_map(
    archive_paths: Mapping[int, str | Path] | None,
    loops: tuple[int, ...],
) -> dict[int, Path]:
    supplied = dict(archive_paths or {})
    if any(isinstance(loop, bool) or not isinstance(loop, int) or loop not in (0, 1, 2, 3)
           for loop in supplied):
        raise ValueError("archive_paths keys must be loop counts from 0 through 3")
    paths: dict[int, Path] = {}
    for loop in loops:
        if loop in supplied:
            paths[loop] = Path(supplied[loop])
        elif loop == 3:
            paths[loop] = default_l3_archive_path()
        else:
            paths[loop] = default_g08_audit_path(loop)
    return paths


def _build_loop_classes(
    loop_count: int,
    source_identities: tuple[AllLoopSourceIdentity, ...],
) -> tuple[tuple[AllLoopMarkedClass, ...], int]:
    if len(source_identities) != EXPECTED_FIXED_COUNTS[loop_count]:
        raise ValueError(f"L{loop_count} fixed identity count differs from the frozen coverage list")

    groups: dict[tuple[Any, ...], list[tuple[AllLoopSourceIdentity, MarkedIdentity]]] = {}
    topology_keys: set[tuple[Any, ...]] = set()
    for source_identity in source_identities:
        archived = source_identity.archived
        fixed_identity = canonicalize_physical_labeling(
            archived.canonical_record, CountMode.FIXED_LABELS
        )
        if (fixed_identity.key != archived.key or fixed_identity.digest != archived.digest
                or fixed_identity.topology_key != archived.topology_key):
            raise ValueError(f"L{loop_count} fixed identity changed for {archived.program_id}/{archived.digest}")
        if (source_identity.fixed_identity.key != fixed_identity.key
                or source_identity.fixed_identity.digest != fixed_identity.digest
                or source_identity.fixed_identity.topology_key != fixed_identity.topology_key):
            raise ValueError(f"L{loop_count} source fixed identity disagrees with its stored record")
        topology_keys.add(fixed_identity.topology_key)
        particle_identity = canonicalize_physical_labeling(
            archived.canonical_record, CountMode.PARTICLE_RELABELING
        )
        groups.setdefault(particle_identity.key, []).append((source_identity, particle_identity))

    if len(topology_keys) != EXPECTED_TOPOLOGY_COUNTS[loop_count]:
        raise ValueError(
            f"L{loop_count} expected {EXPECTED_TOPOLOGY_COUNTS[loop_count]} topologies, "
            f"got {len(topology_keys)}"
        )
    if len(groups) != EXPECTED_CLASS_COUNTS[loop_count]:
        raise ValueError(
            f"L{loop_count} expected {EXPECTED_CLASS_COUNTS[loop_count]} particle classes, "
            f"got {len(groups)}"
        )

    topology_ids = {
        key: f"L{loop_count}-T{index:02d}"
        for index, key in enumerate(sorted(topology_keys, key=_canonical_json), start=1)
    }
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            _canonical_json(item[1][0][1].topology_key),
            item[1][0][1].digest,
            _canonical_json(item[0]),
        ),
    )

    classes: list[AllLoopMarkedClass] = []
    for index, (particle_key, group) in enumerate(ordered_groups, start=1):
        group.sort(key=lambda pair: (pair[0].archived.program_id, pair[0].archived.digest))
        particle_identity = group[0][1]
        canonical_particle_record = particle_identity.canonical_record.to_dict()
        if any(identity.key != particle_key
               or identity.canonical_record.to_dict() != canonical_particle_record
               for _, identity in group):
            raise ValueError("particle-relabeling group has inconsistent canonical records")

        fixed_canonical = canonicalize_physical_labeling(
            particle_identity.canonical_record, CountMode.FIXED_LABELS
        )
        representative_candidates = [
            (source_identity, particle_member_identity)
            for source_identity, particle_member_identity in group
            if (source_identity.archived.key == fixed_canonical.key
                and source_identity.archived.digest == fixed_canonical.digest)
        ]
        representative_source, representative_particle_identity = (
            representative_candidates[0] if representative_candidates else group[0]
        )
        representative = representative_source.archived
        members = tuple(
            AllLoopCatalogMember(
                archived=source_identity.archived,
                source_locator=source_identity.source_locator,
                map_to_representative=_d5._compose_to_representative(
                    member_particle_identity,
                    representative_particle_identity,
                    source_identity.archived,
                ),
            )
            for source_identity, member_particle_identity in group
        )
        topology_key = representative.topology_key
        topology_digest = _d5._digest(topology_key)
        classes.append(AllLoopMarkedClass(
            class_id=f"L{loop_count}-C{index:03d}",
            loop_count=loop_count,
            topology_id=topology_ids[topology_key],
            topology_key=topology_key,
            topology_digest=topology_digest,
            particle_identity=particle_identity,
            representative=representative,
            representative_source_locator=representative_source.source_locator,
            representative_particle_identity=representative_particle_identity,
            members=members,
            legacy_d5_class_id=f"D5-{index:03d}" if loop_count == 3 else None,
        ))

    member_distribution = tuple(sorted(len(item.members) for item in classes))
    if member_distribution != EXPECTED_MEMBER_DISTRIBUTIONS[loop_count]:
        raise ValueError(f"L{loop_count} fixed-to-class member distribution differs from the frozen list")
    return tuple(classes), len(topology_keys)


def build_all_loop_catalog(
    loops: int | Iterable[int] | None = None,
    archive_paths: Mapping[int, str | Path] | None = None,
) -> AllLoopCatalog:
    """Load frozen records and group their fixed identities by particle relabeling.

    ``archive_paths`` may replace selected file locations, but every source is
    still required to match the SHA-256 recorded in the reviewed coverage list.
    No enumeration entry point is called.
    """

    selected_loops = _normalize_loops(loops)
    sources = _source_path_map(archive_paths, selected_loops)
    all_classes: list[AllLoopMarkedClass] = []
    source_hashes: list[tuple[int, str]] = []
    total_topologies = 0
    total_fixed = 0
    for loop_count in selected_loops:
        if loop_count == 3:
            source_identities, source_sha = _read_frozen_l3(sources[loop_count])
        else:
            source_identities, source_sha = _read_frozen_g08(loop_count, sources[loop_count])
        classes, topology_count = _build_loop_classes(loop_count, source_identities)
        all_classes.extend(classes)
        total_fixed += len(source_identities)
        total_topologies += topology_count
        source_hashes.append((loop_count, source_sha))
    if len({item.representative_fixed_digest for item in all_classes}) != len(all_classes):
        # Fixed representative digests can legitimately not be global class
        # identifiers, so this guards only accidental duplicate class leaders.
        raise ValueError("all-loop catalog contains duplicate fixed class representatives")
    return AllLoopCatalog(
        archive_paths=tuple((loop, sources[loop].resolve()) for loop in selected_loops),
        archive_sha256=tuple(source_hashes),
        fixed_record_count=total_fixed,
        topology_count=total_topologies,
        classes=tuple(all_classes),
    )


def select_all_loop_classes(
    catalog: AllLoopCatalog,
    loops: int | Iterable[int] | None = None,
    topology: str | None = None,
    class_selector: str | None = None,
) -> tuple[AllLoopMarkedClass, ...]:
    """Filter catalog classes by loop, topology ID/digest, and class ID/digest."""

    if not isinstance(catalog, AllLoopCatalog):
        raise TypeError("select_all_loop_classes expects an AllLoopCatalog")
    selected_loops = None if loops is None else set(_normalize_loops(loops))
    candidates = tuple(
        item for item in catalog.classes
        if selected_loops is None or item.loop_count in selected_loops
    )

    if topology is not None:
        if not isinstance(topology, str):
            raise TypeError("topology selector must be a string")
        value = topology.strip()
        if not value:
            raise ValueError("topology selector must not be empty")
        matched_topologies = tuple(
            item for item in candidates
            if item.topology_id.casefold() == value.casefold()
            or item.topology_digest.casefold() == value.casefold()
            or item.topology_digest.casefold().startswith(value.casefold())
        )
        if not matched_topologies:
            raise ValueError(f"unknown topology selector {topology!r}")
        topology_ids = {item.topology_id for item in matched_topologies}
        if len(topology_ids) != 1:
            raise ValueError(f"topology digest prefix {topology!r} is ambiguous")
        candidates = matched_topologies

    if class_selector is not None:
        if not isinstance(class_selector, str):
            raise TypeError("class selector must be a string")
        value = class_selector.strip()
        if not value:
            raise ValueError("class selector must not be empty")
        matched_classes = tuple(
            item for item in candidates
            if item.class_id.casefold() == value.casefold()
            or (item.legacy_d5_class_id is not None
                and item.legacy_d5_class_id.casefold() == value.casefold())
            or item.particle_identity.digest.casefold() == value.casefold()
            or item.particle_identity.digest.casefold().startswith(value.casefold())
        )
        if not matched_classes:
            raise ValueError(f"unknown class selector {class_selector!r}")
        if len(matched_classes) != 1:
            raise ValueError(f"class digest prefix {class_selector!r} is ambiguous")
        candidates = matched_classes
    elif topology is None and loops is None and not candidates:
        return ()
    return candidates


__all__ = [
    "G08_AUDIT_RELATIVE_TEMPLATE",
    "L3_ARCHIVE_RELATIVE_PATH",
    "EXPECTED_TOPOLOGY_COUNTS",
    "EXPECTED_FIXED_COUNTS",
    "EXPECTED_CLASS_COUNTS",
    "EXPECTED_MEMBER_DISTRIBUTIONS",
    "EXPECTED_SOURCE_SHA256",
    "AllLoopCatalogMember",
    "AllLoopMarkedClass",
    "AllLoopCatalog",
    "default_g08_audit_path",
    "default_l3_archive_path",
    "build_all_loop_catalog",
    "select_all_loop_classes",
]
