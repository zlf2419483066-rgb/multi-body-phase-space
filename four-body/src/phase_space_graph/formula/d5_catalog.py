"""Read-only catalog for the archived 106 three-loop D5 marked classes.

The catalog consumes only the frozen 2,544-row fixed-identity archive.  It
does not enumerate labelings or alter topology, DAG, or IKV rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from ..domain import CountMode
from ..drawing.archive import ArchivedIdentity, load_identity_archive
from ..invariants import invariant_label
from ..label_identity import MarkedIdentity, canonicalize_physical_labeling


D5_ARCHIVE_RELATIVE_PATH = Path(
    "文档/验证与测试/误差分析/三圈来源集合审查_2026-09-16/run01/"
    "2544个固定标记完整身份.json.gz"
)
D5_EXPECTED_FIXED_COUNT = 2544
D5_EXPECTED_TOPOLOGY_COUNT = 33
D5_EXPECTED_CLASS_COUNT = 106
D5_EXPECTED_ARCHIVE_SHA256 = "4f1c2253ef098e4c02bd929986bc5c0e8e6707f99d8500ad516121671aa06878"


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=False,
        sort_keys=True,
        separators=None if indent is not None else (",", ":"),
        indent=indent,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _pairs_map(pairs: tuple[tuple[Any, Any], ...]) -> dict[Any, Any]:
    mapping = dict(pairs)
    if len(mapping) != len(pairs) or len(set(mapping.values())) != len(mapping):
        raise ValueError("canonical identity map must be a bijection")
    return mapping


def _inverse(mapping: dict[Any, Any]) -> dict[Any, Any]:
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("canonical identity map must be invertible")
    return {value: key for key, value in mapping.items()}


def _compose_to_representative(
    member: MarkedIdentity,
    representative: MarkedIdentity,
    member_record: ArchivedIdentity,
) -> dict[str, Any]:
    """Compose source→shared-canonical with shared-canonical→representative."""

    member_vertices = _pairs_map(member.vertex_map)
    representative_vertices = _inverse(_pairs_map(representative.vertex_map))
    member_particles = _pairs_map(member.particle_map)
    representative_particles = _inverse(_pairs_map(representative.particle_map))
    member_edges = _pairs_map(member.edge_occurrence_map)
    representative_edges = _inverse(_pairs_map(representative.edge_occurrence_map))
    member_slots = _pairs_map(member.final_slot_map)
    representative_slots = _inverse(_pairs_map(representative.final_slot_map))

    if member.canonical_record.to_dict() != representative.canonical_record.to_dict():
        raise ValueError("fixed records do not share the expected particle-relabeling representative")

    vertex_map = {source: representative_vertices[canonical]
                  for source, canonical in member_vertices.items()}
    particle_map = {source: representative_particles[canonical]
                    for source, canonical in member_particles.items()}
    edge_map = {source: representative_edges[canonical]
                for source, canonical in member_edges.items()}
    slot_map = {source: representative_slots[canonical]
                for source, canonical in member_slots.items()}
    return {
        "direction": "source_fixed_record_to_representative_fixed_record",
        "vertex_map": [
            {"source_vertex": source, "representative_vertex": target}
            for source, target in sorted(vertex_map.items())
        ],
        "particle_map": [
            {
                "source_particle": source,
                "representative_particle": target,
                "momentum_label_substitution": f"p_source[{source}] -> p_rep[{target}]",
                "mass_symbol_substitution": f"m_source[{source}] -> m_rep[{target}]",
                "representative_mass_assignment": f"m_rep[{target}] = m_source[{source}]",
                "reverse_substitution_to_source": f"m_rep[{target}] -> m_source[{source}]",
            }
            for source, target in sorted(particle_map.items())
        ],
        "invariant_map": [
            {
                "source_ikv": invariant_label(source_subset),
                "representative_ikv": invariant_label(tuple(sorted(particle_map[p] for p in source_subset))),
                "source_particles": list(source_subset),
                "representative_particles": sorted(particle_map[p] for p in source_subset),
            }
            for source_subset in member_record.canonical_record.momentum.variables
        ],
        "edge_occurrence_map": [
            {
                "source_edge_index": source[0],
                "source_occurrence": source[1],
                "representative_edge_index": target[0],
                "representative_occurrence": target[1],
            }
            for source, target in sorted(edge_map.items())
        ],
        "final_slot_map": [
            {
                "source_vertex": source[0],
                "source_slot": source[1],
                "representative_vertex": target[0],
                "representative_slot": target[1],
            }
            for source, target in sorted(slot_map.items())
        ],
    }


@dataclass(frozen=True, slots=True)
class D5CatalogMember:
    """One archived fixed identity and its explicit map to a class representative."""

    archived: ArchivedIdentity
    map_to_representative: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.archived.program_id,
            "fixed_key": _jsonable(self.archived.key),
            "fixed_digest": self.archived.digest,
            "map_to_representative": self.map_to_representative,
        }


@dataclass(frozen=True, slots=True)
class D5MarkedClass:
    """One particle-relabeling class with an archived fixed representative."""

    class_id: str
    topology_id: str
    topology_key: tuple[Any, ...]
    topology_digest: str
    particle_identity: MarkedIdentity
    representative: ArchivedIdentity
    representative_particle_identity: MarkedIdentity
    members: tuple[D5CatalogMember, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "topology_id": self.topology_id,
            "topology_key": _jsonable(self.topology_key),
            "topology_digest": self.topology_digest,
            "particle_relabeling_key": _jsonable(self.particle_identity.key),
            "particle_relabeling_digest": self.particle_identity.digest,
            "representative_policy": "archived fixed record whose fixed identity matches the canonical particle-relabeling record; deterministic archived member fallback if absent",
            "representative_program_id": self.representative.program_id,
            "representative_fixed_key": _jsonable(self.representative.key),
            "representative_fixed_digest": self.representative.digest,
            "representative_record": self.representative.canonical_record.to_dict(),
            "members": [member.to_dict() for member in self.members],
        }


@dataclass(frozen=True, slots=True)
class D5IdentityCatalog:
    """Validated grouping of the fixed historical archive."""

    archive_path: Path
    archive_sha256: str
    fixed_record_count: int
    topology_count: int
    classes: tuple[D5MarkedClass, ...]

    def select_topology(self, selector: str) -> tuple[D5MarkedClass, ...]:
        value = selector.strip()
        if not value:
            raise ValueError("topology selector must not be empty")
        matched = tuple(
            item for item in self.classes
            if item.topology_id.casefold() == value.casefold()
            or item.topology_digest.casefold() == value.casefold()
            or item.topology_digest.casefold().startswith(value.casefold())
        )
        if not matched:
            valid = ", ".join(sorted({item.topology_id for item in self.classes}))
            raise ValueError(f"unknown topology selector {selector!r}; topology IDs: {valid}")
        if len({item.topology_id for item in matched}) != 1:
            raise ValueError(f"topology digest prefix {selector!r} is ambiguous")
        return matched

    def select_class(self, selector: str) -> tuple[D5MarkedClass, ...]:
        value = selector.strip()
        if not value:
            raise ValueError("class selector must not be empty")
        matched = tuple(
            item for item in self.classes
            if item.class_id.casefold() == value.casefold()
            or item.particle_identity.digest.casefold() == value.casefold()
            or item.particle_identity.digest.casefold().startswith(value.casefold())
        )
        if not matched:
            raise ValueError(f"unknown class selector {selector!r}; use D5-001–D5-106 or a digest prefix")
        if len(matched) != 1:
            raise ValueError(f"class digest prefix {selector!r} is ambiguous")
        return matched

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "d5_three_loop_identity_catalog",
            "schema_version": "D5-catalog-v1",
            "source_archive": str(self.archive_path),
            "source_archive_sha256": self.archive_sha256,
            "grouping": "canonicalize_physical_labeling(record, CountMode.PARTICLE_RELABELING); full canonical key is the group identity",
            "fixed_identity_count": self.fixed_record_count,
            "topology_count": self.topology_count,
            "particle_relabeling_class_count": len(self.classes),
            "representative_map_direction": "each archived fixed record → the selected archived fixed representative of its particle-relabeling class",
            "classes": [item.to_dict() for item in self.classes],
        }


def default_d5_archive_path() -> Path:
    demo_root = Path(__file__).resolve().parents[3]
    return demo_root / D5_ARCHIVE_RELATIVE_PATH


def build_d5_catalog(archive_path: str | Path | None = None) -> D5IdentityCatalog:
    """Load and group archived fixed records without re-enumeration."""

    source = Path(archive_path) if archive_path is not None else default_d5_archive_path()
    source = source.resolve()
    archive_sha_before = hashlib.sha256(source.read_bytes()).hexdigest()
    if archive_path is None and archive_sha_before != D5_EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            "the default historical fixed-identity archive SHA-256 differs from the reviewed source"
        )
    archived = load_identity_archive(source)
    archive_sha_after = hashlib.sha256(source.read_bytes()).hexdigest()
    if archive_sha_before != archive_sha_after:
        raise ValueError("fixed-identity archive changed while the D5 catalog was being built")
    if len(archived) != D5_EXPECTED_FIXED_COUNT:
        raise ValueError(
            f"expected {D5_EXPECTED_FIXED_COUNT} archived fixed identities, got {len(archived)}"
        )
    if len({entry.key for entry in archived}) != len(archived):
        raise ValueError("historical fixed-identity archive contains a repeated full fixed key")
    if len({entry.digest for entry in archived}) != len(archived):
        raise ValueError("historical fixed-identity archive contains a repeated fixed digest")

    groups: dict[tuple[Any, ...], list[tuple[ArchivedIdentity, MarkedIdentity]]] = {}
    topology_keys: set[tuple[Any, ...]] = set()
    for fixed in archived:
        fixed_identity = canonicalize_physical_labeling(fixed.canonical_record, CountMode.FIXED_LABELS)
        if fixed_identity.key != fixed.key or fixed_identity.digest != fixed.digest:
            raise ValueError(f"fixed identity mismatch for {fixed.program_id}/{fixed.digest}")
        if fixed.topology_key != fixed_identity.topology_key:
            raise ValueError(f"topology key mismatch for {fixed.program_id}/{fixed.digest}")
        topology_keys.add(fixed.topology_key)
        particle_identity = canonicalize_physical_labeling(
            fixed.canonical_record, CountMode.PARTICLE_RELABELING
        )
        groups.setdefault(particle_identity.key, []).append((fixed, particle_identity))

    if len(topology_keys) != D5_EXPECTED_TOPOLOGY_COUNT:
        raise ValueError(f"expected {D5_EXPECTED_TOPOLOGY_COUNT} topologies, got {len(topology_keys)}")
    if len(groups) != D5_EXPECTED_CLASS_COUNT:
        raise ValueError(f"expected {D5_EXPECTED_CLASS_COUNT} particle classes, got {len(groups)}")

    sorted_topologies = sorted(topology_keys, key=_canonical_json)
    topology_ids = {key: f"T{index:02d}" for index, key in enumerate(sorted_topologies, start=1)}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            _canonical_json(item[1][0][1].topology_key),
            item[1][0][1].digest,
            _canonical_json(item[0]),
        ),
    )

    classes: list[D5MarkedClass] = []
    for index, (particle_key, group) in enumerate(ordered_groups, start=1):
        group.sort(key=lambda pair: (pair[0].program_id, pair[0].digest))
        particle_identity = group[0][1]
        canonical_record = particle_identity.canonical_record.to_dict()
        if any(identity.key != particle_key or identity.canonical_record.to_dict() != canonical_record
               for _, identity in group):
            raise ValueError("particle-relabeling group has inconsistent canonical records")

        fixed_canonical = canonicalize_physical_labeling(
            particle_identity.canonical_record, CountMode.FIXED_LABELS
        )
        representative_candidates = [
            (fixed, particle_map)
            for fixed, particle_map in group
            if fixed.digest == fixed_canonical.digest and fixed.key == fixed_canonical.key
        ]
        representative, representative_particle = (
            representative_candidates[0] if representative_candidates else group[0]
        )

        members = tuple(
            D5CatalogMember(
                archived=fixed,
                map_to_representative=_compose_to_representative(
                    member_particle, representative_particle, fixed
                ),
            )
            for fixed, member_particle in group
        )
        topology_key = representative.topology_key
        topology_digest = _digest(topology_key)
        classes.append(D5MarkedClass(
            class_id=f"D5-{index:03d}",
            topology_id=topology_ids[topology_key],
            topology_key=topology_key,
            topology_digest=topology_digest,
            particle_identity=particle_identity,
            representative=representative,
            representative_particle_identity=representative_particle,
            members=members,
        ))

    return D5IdentityCatalog(
        archive_path=source,
        archive_sha256=archive_sha_before,
        fixed_record_count=len(archived),
        topology_count=len(topology_keys),
        classes=tuple(classes),
    )


__all__ = [
    "D5_ARCHIVE_RELATIVE_PATH",
    "D5_EXPECTED_FIXED_COUNT",
    "D5_EXPECTED_TOPOLOGY_COUNT",
    "D5_EXPECTED_CLASS_COUNT",
    "D5_EXPECTED_ARCHIVE_SHA256",
    "D5CatalogMember",
    "D5MarkedClass",
    "D5IdentityCatalog",
    "default_d5_archive_path",
    "build_d5_catalog",
]
