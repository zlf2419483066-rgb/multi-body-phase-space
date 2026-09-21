"""Whole-graph relabeling and topology automorphisms (LM-03).

This module is deliberately a representation layer.  A vertex permutation is
applied to the complete graph first, then its external slots, internal edge
occurrences and particle labels are rebuilt together.  Momentum and rank
certificates are recomputed by :func:`normalize_physical_labeling`; no old
mass symbol or edge index is copied into a new topology.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Any, Iterable, Mapping

from .canonical import canonicalize_topology
from .domain import (
    CountMode,
    DoubleLineLabel,
    EdgeType,
    FinalParticleLabel,
    InternalEdge,
    Labeling,
    SingleLineLabel,
    Topology,
)
from .physical_labels import (
    NormalizationResult,
    PhysicalLabelRecord,
    normalize_physical_labeling,
)


def _mapping_items(mapping: Mapping[int, int] | Iterable[tuple[int, int]], name: str) -> dict[int, int]:
    if isinstance(mapping, Mapping):
        items = tuple(mapping.items())
    else:
        try:
            items = tuple(mapping)
        except TypeError as exc:
            raise TypeError(f"{name} must be a mapping or iterable of pairs") from exc
    if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in items):
        raise ValueError(f"{name} must contain (source, target) pairs")
    result: dict[int, int] = {}
    for source, target in items:
        if isinstance(source, bool) or not isinstance(source, int):
            raise TypeError(f"{name} sources must be integers")
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError(f"{name} targets must be integers")
        if source in result:
            raise ValueError(f"{name} contains a repeated source: {source!r}")
        result[source] = target
    return result


def _complete_bijection(
    mapping: Mapping[int, int] | Iterable[tuple[int, int]],
    domain: Iterable[int],
    name: str,
) -> dict[int, int]:
    expected = set(domain)
    result = _mapping_items(mapping, name)
    if set(result) != expected:
        raise ValueError(f"{name} must cover exactly {sorted(expected)!r}")
    if len(set(result.values())) != len(result):
        raise ValueError(f"{name} must be one-to-one")
    return result


def _aggregate_topology(
    topology: Topology,
    vertex_map: Mapping[int, int],
) -> tuple[Topology, dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    """Map and aggregate edges, returning temporary occurrence and slot maps."""

    groups: dict[tuple[int, int, EdgeType], list[tuple[int, int]]] = {}
    for edge_index, edge in enumerate(topology.edges):
        if edge.multiplicity <= 0:
            continue
        key = (vertex_map[edge.source], vertex_map[edge.target], edge.edge_type)
        groups.setdefault(key, []).extend((edge_index, occurrence) for occurrence in range(edge.multiplicity))

    sorted_groups = sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value))
    new_edges = tuple(
        InternalEdge(source, target, edge_type, len(refs))
        for (source, target, edge_type), refs in sorted_groups
    )
    mapped_slots = {vertex_map[vertex]: count for vertex, count in topology.final_slots}
    new_topology = Topology(
        n=topology.n,
        vertices=tuple(sorted(vertex_map.values())),
        edges=new_edges,
        initial_vertex=vertex_map[topology.initial_vertex],
        final_slots=mapped_slots,
    )
    # Topology keeps the sorted group order, so these are temporary occurrence
    # positions.  They are reordered after particle labels are known.
    occurrence_map: dict[tuple[int, int], tuple[int, int]] = {}
    for new_index, (_, refs) in enumerate(sorted_groups):
        for new_occurrence, old_ref in enumerate(refs):
            occurrence_map[old_ref] = (new_index, new_occurrence)
    slot_map = {
        (vertex, slot): (vertex_map[vertex], slot)
        for vertex, count in topology.final_slots
        for slot in range(max(count, 0))
    }
    return new_topology, occurrence_map, slot_map


def _topology_signature(topology: Topology) -> tuple[Any, ...]:
    return (
        topology.n,
        topology.initial_vertex,
        tuple(topology.vertices),
        tuple(topology.final_slots),
        tuple((edge.source, edge.target, edge.edge_type.value, edge.multiplicity) for edge in topology.edges if edge.multiplicity > 0),
    )


@dataclass(frozen=True, slots=True)
class TopologyAlignment:
    """Canonical topology, original-to-canonical map, and all root-fixed symmetries."""

    canonical_topology: Topology
    topology_key: tuple[Any, ...]
    vertex_map: tuple[tuple[int, int], ...]
    automorphisms: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_topology, Topology):
            raise TypeError("canonical_topology must be a Topology")
        if not isinstance(self.topology_key, tuple):
            raise TypeError("topology_key must be a tuple")
        object.__setattr__(self, "vertex_map", tuple(tuple(pair) for pair in self.vertex_map))
        object.__setattr__(self, "automorphisms", tuple(tuple(image) for image in self.automorphisms))

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_topology": self.canonical_topology.to_dict(),
            "topology_key": _jsonable(self.topology_key),
            "vertex_map": [{"original": a, "canonical": b} for a, b in self.vertex_map],
            "automorphisms": [list(image) for image in self.automorphisms],
        }


@dataclass(frozen=True, slots=True)
class RelabeledPhysicalRecord:
    """A rebuilt record plus every old-reference to new-reference map."""

    record: PhysicalLabelRecord
    vertex_map: tuple[tuple[int, int], ...]
    particle_map: tuple[tuple[int, int], ...]
    edge_occurrence_map: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    final_slot_map: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    normalization: NormalizationResult

    def __post_init__(self) -> None:
        if not isinstance(self.record, PhysicalLabelRecord):
            raise TypeError("record must be a PhysicalLabelRecord")
        if not isinstance(self.normalization, NormalizationResult):
            raise TypeError("normalization must be a NormalizationResult")

    def to_dict(self) -> dict[str, object]:
        return {
            "record": self.record.to_dict(),
            "vertex_map": [list(pair) for pair in self.vertex_map],
            "particle_map": [list(pair) for pair in self.particle_map],
            "edge_occurrence_map": [
                {"old": list(old), "new": list(new)} for old, new in self.edge_occurrence_map
            ],
            "final_slot_map": [
                {"old": list(old), "new": list(new)} for old, new in self.final_slot_map
            ],
            "normalization_stage": self.normalization.stage,
        }


@dataclass(frozen=True, slots=True)
class MarkedIdentity:
    """Stable identity of one complete physical labeling in one mode."""

    schema_version: str
    mode: CountMode
    topology_key: tuple[Any, ...]
    key: tuple[Any, ...]
    digest: str
    canonical_record: PhysicalLabelRecord
    vertex_map: tuple[tuple[int, int], ...]
    particle_map: tuple[tuple[int, int], ...]
    edge_occurrence_map: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    final_slot_map: tuple[tuple[tuple[int, int], tuple[int, int]], ...]

    def __post_init__(self) -> None:
        if self.schema_version != "IM-4-v1":
            raise ValueError(f"unsupported marked-identity schema: {self.schema_version!r}")
        if not isinstance(self.mode, CountMode):
            raise TypeError("mode must be a CountMode")
        if not isinstance(self.canonical_record, PhysicalLabelRecord):
            raise TypeError("canonical_record must be a PhysicalLabelRecord")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "topology_key": _jsonable(self.topology_key),
            "key": _jsonable(self.key),
            "digest": self.digest,
            "canonical_record": self.canonical_record.to_dict(),
            "vertex_map": [{"original": a, "canonical": b} for a, b in self.vertex_map],
            "particle_map": [{"original": a, "canonical": b} for a, b in self.particle_map],
            "edge_occurrence_map": [
                {"old": list(old), "new": list(new)} for old, new in self.edge_occurrence_map
            ],
            "final_slot_map": [
                {"old": list(old), "new": list(new)} for old, new in self.final_slot_map
            ],
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _identity_digest(key: tuple[Any, ...]) -> str:
    payload = json.dumps(_jsonable(key), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _marked_payload(
    record: PhysicalLabelRecord,
    vertex_map: Mapping[int, int],
    particle_map: Mapping[int, int] | None = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Build the LM-04 F,E payload without solving physics again."""

    particle_map = particle_map or {particle: particle for particle in range(1, record.topology.n + 1)}
    final_by_vertex: dict[int, list[int]] = {new: [] for new in set(vertex_map.values())}
    for label in record.complete_labeling.final_particles:
        final_by_vertex[vertex_map[label.vertex]].append(particle_map[label.particle_id])
    final_payload = tuple(tuple(sorted(final_by_vertex[index])) for index in sorted(final_by_vertex))

    single_by_ref = {
        (label.edge_index, label.occurrence): (particle_map[label.particle_id],)
        for label in record.complete_labeling.single_lines
    }
    double_by_ref = {
        (label.edge_index, label.occurrence): tuple(sorted(particle_map[particle] for particle in label.particle_ids))
        for label in record.complete_labeling.double_lines
    }
    edge_payload: list[tuple[int, int, int, tuple[int, ...]]] = []
    for edge_index, edge in enumerate(record.topology.edges):
        if edge.multiplicity <= 0:
            continue
        kind = 0 if edge.edge_type is EdgeType.DOUBLE else 1
        lookup = double_by_ref if kind == 0 else single_by_ref
        for occurrence in range(edge.multiplicity):
            edge_payload.append(
                (
                    vertex_map[edge.source],
                    vertex_map[edge.target],
                    kind,
                    lookup.get((edge_index, occurrence), ()),
                )
            )
    return final_payload, tuple(sorted(edge_payload))


def canonicalize_physical_labeling(
    record: PhysicalLabelRecord,
    mode: CountMode | str = CountMode.FIXED_LABELS,
) -> MarkedIdentity:
    """Choose the smallest complete ``(F,E)`` payload over topology automorphisms."""

    if not isinstance(record, PhysicalLabelRecord):
        raise TypeError("canonicalize_physical_labeling expects a PhysicalLabelRecord")
    canonical_mode = mode if isinstance(mode, CountMode) else CountMode(mode)
    alignment = align_topology(record.topology)
    initial_map = dict(alignment.vertex_map)
    candidates: list[tuple[tuple[Any, ...], tuple[int, ...], dict[int, int]]] = []
    particle_permutations = (
        (tuple(range(1, record.topology.n + 1)),)
        if canonical_mode is CountMode.FIXED_LABELS
        else itertools.permutations(range(1, record.topology.n + 1))
    )
    for permutation in particle_permutations:
        particle_map = {particle: permutation[particle - 1] for particle in range(1, record.topology.n + 1)}
        for automorphism in alignment.automorphisms:
            mapping = {original: automorphism[canonical] for original, canonical in initial_map.items()}
            payload = _marked_payload(record, mapping, particle_map)
            candidates.append((payload, automorphism, mapping, particle_map, permutation))
    payload, _, winning_map, winning_particles, _ = min(candidates, key=lambda item: (item[0], item[1], item[4]))
    key = ("IM-4-v1", canonical_mode.value, alignment.topology_key, payload[0], payload[1])
    digest = _identity_digest(key)
    transformed = relabel_physical_record(record, winning_map, winning_particles)
    return MarkedIdentity(
        schema_version="IM-4-v1",
        mode=canonical_mode,
        topology_key=alignment.topology_key,
        key=key,
        digest=digest,
        canonical_record=transformed.record,
        vertex_map=transformed.vertex_map,
        particle_map=transformed.particle_map,
        edge_occurrence_map=transformed.edge_occurrence_map,
        final_slot_map=transformed.final_slot_map,
    )


def align_topology(topology: Topology) -> TopologyAlignment:
    """Return the existing G04 canonical topology and all root-fixed automorphisms."""

    canonical = canonicalize_topology(topology)
    original_map = dict(canonical.original_to_canonical)
    canonical_topology, _, _ = _aggregate_topology(topology, original_map)
    expected = _topology_signature(canonical_topology)
    vertices = tuple(canonical_topology.vertices)
    if canonical_topology.initial_vertex != 0:
        raise ValueError("canonical topology must map the initial vertex to 0")

    automorphisms: list[tuple[int, ...]] = []
    rest = tuple(vertex for vertex in vertices if vertex != 0)
    for permutation in itertools.permutations(rest):
        image = (0,) + permutation
        mapping = {vertex: image[vertex] for vertex in vertices}
        transformed, _, _ = _aggregate_topology(canonical_topology, mapping)
        if _topology_signature(transformed) == expected:
            automorphisms.append(image)
    identity = tuple(vertices)
    if identity not in automorphisms:
        automorphisms.insert(0, identity)
    automorphisms.sort()
    return TopologyAlignment(
        canonical_topology=canonical_topology,
        topology_key=canonical.key,
        vertex_map=tuple(sorted(original_map.items())),
        automorphisms=tuple(automorphisms),
    )


def _relabel_record(
    record: PhysicalLabelRecord,
    vertex_map: Mapping[int, int],
    particle_map: Mapping[int, int],
) -> RelabeledPhysicalRecord:
    topology = record.topology
    fields = topology.validate_fields()
    labels = record.complete_labeling.validate_fields(topology)
    if not fields.valid or not labels.valid:
        raise ValueError("record must contain field-valid topology and complete labeling")

    mapped_topology, temporary_edges, _ = _aggregate_topology(topology, vertex_map)
    # Collect mapped labels by structural edge and sort occurrences by their
    # physical payload, as required by the complete identity contract.
    single_entries: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for label in record.complete_labeling.single_lines:
        old_ref = (label.edge_index, label.occurrence)
        new_ref = temporary_edges[old_ref]
        single_entries.append((old_ref, new_ref[0], particle_map[label.particle_id]))
    double_entries: list[tuple[tuple[int, int], int, tuple[int, ...]]] = []
    for label in record.complete_labeling.double_lines:
        old_ref = (label.edge_index, label.occurrence)
        new_ref = temporary_edges[old_ref]
        subset = tuple(sorted(particle_map[particle] for particle in label.particle_ids))
        double_entries.append((old_ref, new_ref[0], subset))

    final_labels_by_vertex: dict[int, list[tuple[FinalParticleLabel, int]]] = {}
    for label in record.complete_labeling.final_particles:
        final_vertex = vertex_map[label.vertex]
        final_labels_by_vertex.setdefault(final_vertex, []).append((label, particle_map[label.particle_id]))
    final_slot_map: dict[tuple[int, int], tuple[int, int]] = {}
    final_particles: list[FinalParticleLabel] = []
    for vertex in sorted(final_labels_by_vertex):
        entries = sorted(final_labels_by_vertex[vertex], key=lambda item: (item[1], item[0].vertex, item[0].slot))
        for new_slot, (old_label, new_particle) in enumerate(entries):
            final_slot_map[(old_label.vertex, old_label.slot)] = (vertex, new_slot)
            final_particles.append(FinalParticleLabel(vertex, new_slot, new_particle))

    edge_occurrence_map: dict[tuple[int, int], tuple[int, int]] = {}
    for edge_index in range(len(mapped_topology.edges)):
        singles = sorted(
            ((old_ref, particle) for old_ref, new_index, particle in single_entries if new_index == edge_index),
            key=lambda item: (item[1], item[0]),
        )
        doubles = sorted(
            ((old_ref, subset) for old_ref, new_index, subset in double_entries if new_index == edge_index),
            key=lambda item: (item[1], item[0]),
        )
        entries = singles if mapped_topology.edges[edge_index].edge_type is EdgeType.SINGLE else doubles
        for new_occurrence, (old_ref, _) in enumerate(entries):
            edge_occurrence_map[old_ref] = (edge_index, new_occurrence)

    single_labels = [
        SingleLineLabel(edge_occurrence_map[(label.edge_index, label.occurrence)][0],
                        edge_occurrence_map[(label.edge_index, label.occurrence)][1],
                        particle_map[label.particle_id])
        for label in record.complete_labeling.single_lines
    ]
    double_labels = [
        DoubleLineLabel(edge_occurrence_map[(label.edge_index, label.occurrence)][0],
                        edge_occurrence_map[(label.edge_index, label.occurrence)][1],
                        tuple(sorted(particle_map[particle] for particle in label.particle_ids)))
        for label in record.complete_labeling.double_lines
    ]
    transformed_labeling = Labeling(final_particles, single_labels, double_labels)
    normalization = normalize_physical_labeling(topology=mapped_topology, labeling=transformed_labeling)
    if normalization.stage not in {"accepted", "dependent"} or normalization.record is None:
        raise ValueError(f"relabeling did not preserve a valid physical record: {normalization.to_dict()}")
    mapped_slots = {
        (label.vertex, label.slot): final_slot_map[(label.vertex, label.slot)]
        for label in record.complete_labeling.final_particles
    }
    return RelabeledPhysicalRecord(
        record=normalization.record,
        vertex_map=tuple(sorted(vertex_map.items())),
        particle_map=tuple(sorted(particle_map.items())),
        edge_occurrence_map=tuple(sorted(edge_occurrence_map.items())),
        final_slot_map=tuple(sorted(mapped_slots.items())),
        normalization=normalization,
    )


def relabel_physical_record(
    record: PhysicalLabelRecord,
    vertex_map: Mapping[int, int] | Iterable[tuple[int, int]],
    particle_map: Mapping[int, int] | Iterable[tuple[int, int]] | None = None,
) -> RelabeledPhysicalRecord:
    """Apply complete vertex and particle bijections to one physical record."""

    if not isinstance(record, PhysicalLabelRecord):
        raise TypeError("relabel_physical_record expects a PhysicalLabelRecord")
    vertices = _complete_bijection(vertex_map, record.topology.vertices, "vertex_map")
    particles = _complete_bijection(
        particle_map if particle_map is not None else ((p, p) for p in range(1, record.topology.n + 1)),
        range(1, record.topology.n + 1),
        "particle_map",
    )
    return _relabel_record(record, vertices, particles)


__all__ = [
    "TopologyAlignment",
    "RelabeledPhysicalRecord",
    "MarkedIdentity",
    "align_topology",
    "canonicalize_physical_labeling",
    "relabel_physical_record",
]
