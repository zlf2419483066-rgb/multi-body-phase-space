"""Complete five-body candidate identities (IM-5-candidate-v1).

Identity is representation data only.  It canonicalizes the complete graph
and all particle occurrences, then rechecks the transformed record through the
candidate structure-plus-momentum normalizer.  No invariant or formula module
is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from collections import OrderedDict
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
    CandidateMomentumRecord,
    CandidateNormalizationResult,
    normalize_candidate_labeling,
    review_candidate_record,
)


SCHEMA_VERSION = "IM-5-candidate-v1"
_ALIGNMENT_CACHE_MAXSIZE = 64
_alignment_cache: OrderedDict[tuple[Any, ...], "TopologyAlignment"] = OrderedDict()
_alignment_cache_hits = 0
_alignment_cache_misses = 0


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


def _digest(key: tuple[Any, ...]) -> str:
    payload = json.dumps(_jsonable(key), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping_items(mapping: Mapping[int, int] | Iterable[tuple[int, int]], name: str) -> dict[int, int]:
    items = tuple(mapping.items()) if isinstance(mapping, Mapping) else tuple(mapping)
    if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in items):
        raise ValueError(f"{name} must contain (source, target) pairs")
    result: dict[int, int] = {}
    for source, target in items:
        if isinstance(source, bool) or not isinstance(source, int):
            raise TypeError(f"{name} sources must be integers")
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError(f"{name} targets must be integers")
        if source in result:
            raise ValueError(f"{name} contains a repeated source")
        result[source] = target
    return result


def _complete_bijection(
    mapping: Mapping[int, int] | Iterable[tuple[int, int]],
    domain: Iterable[int],
    name: str,
) -> dict[int, int]:
    expected = set(domain)
    result = _mapping_items(mapping, name)
    if set(result) != expected or len(set(result.values())) != len(result):
        raise ValueError(f"{name} must be a bijection over {sorted(expected)!r}")
    return result


def _aggregate_topology(
    topology: Topology,
    vertex_map: Mapping[int, int],
) -> tuple[Topology, dict[tuple[int, int], tuple[int, int]]]:
    groups: dict[tuple[int, int, EdgeType], list[tuple[int, int]]] = {}
    for edge_index, edge in enumerate(topology.edges):
        if edge.multiplicity <= 0:
            continue
        key = (vertex_map[edge.source], vertex_map[edge.target], edge.edge_type)
        groups.setdefault(key, []).extend(
            (edge_index, occurrence) for occurrence in range(edge.multiplicity)
        )
    sorted_groups = sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value))
    new_edges = tuple(
        InternalEdge(source, target, edge_type, len(refs))
        for (source, target, edge_type), refs in sorted_groups
    )
    new_topology = Topology(
        n=topology.n,
        vertices=tuple(sorted(vertex_map.values())),
        edges=new_edges,
        initial_vertex=vertex_map[topology.initial_vertex],
        final_slots={vertex_map[vertex]: count for vertex, count in topology.final_slots},
    )
    occurrence_map: dict[tuple[int, int], tuple[int, int]] = {}
    for new_index, (_, refs) in enumerate(sorted_groups):
        for new_occurrence, old_ref in enumerate(refs):
            occurrence_map[old_ref] = (new_index, new_occurrence)
    return new_topology, occurrence_map


def _topology_signature(topology: Topology) -> tuple[Any, ...]:
    return (
        topology.n,
        topology.initial_vertex,
        tuple(topology.vertices),
        tuple(topology.final_slots),
        tuple(
            (edge.source, edge.target, edge.edge_type.value, edge.multiplicity)
            for edge in topology.edges
            if edge.multiplicity > 0
        ),
    )


@dataclass(frozen=True, slots=True)
class TopologyAlignment:
    canonical_topology: Topology
    topology_key: tuple[Any, ...]
    vertex_map: tuple[tuple[int, int], ...]
    automorphisms: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class RelabeledCandidateRecord:
    record: CandidateMomentumRecord
    vertex_map: tuple[tuple[int, int], ...]
    particle_map: tuple[tuple[int, int], ...]
    edge_occurrence_map: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    final_slot_map: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    normalization: CandidateNormalizationResult


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    schema_version: str
    mode: CountMode
    topology_key: tuple[Any, ...]
    key: tuple[Any, ...]
    digest: str
    canonical_record: CandidateMomentumRecord
    vertex_map: tuple[tuple[int, int], ...]
    particle_map: tuple[tuple[int, int], ...]
    edge_occurrence_map: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    final_slot_map: tuple[tuple[tuple[int, int], tuple[int, int]], ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported candidate identity schema: {self.schema_version!r}")
        if not isinstance(self.mode, CountMode):
            raise TypeError("mode must be a CountMode")
        if not isinstance(self.canonical_record, CandidateMomentumRecord):
            raise TypeError("canonical_record must be CandidateMomentumRecord")

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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateIdentity":
        """Parse and fully rederive an identity; parsing alone is not review."""
        if not isinstance(data, Mapping):
            raise TypeError("candidate identity must be a mapping")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported candidate identity schema")
        mode = CountMode(data["mode"])
        record = CandidateMomentumRecord.from_dict(data["canonical_record"])
        # Parsing a record checks its shape; identity read-back additionally
        # rechecks the declared momentum certificate against the labeling.
        review_candidate_record(record)
        rebuilt = canonicalize_candidate_labeling(record, mode)
        if _restore_tuples(data.get("key")) != rebuilt.key:
            raise ValueError("stored candidate identity key differs from rebuilt key")
        if data.get("digest") != rebuilt.digest:
            raise ValueError("stored candidate identity digest differs from rebuilt key")
        if _restore_tuples(data.get("topology_key")) != rebuilt.topology_key:
            raise ValueError("stored candidate topology key differs from rebuilt key")
        if record.to_dict() != rebuilt.canonical_record.to_dict():
            raise ValueError("stored canonical record is not the canonical representative")
        # Maps are evidence attached to the identity.  They are not part of
        # the physical key, but read-back must still reject truncated,
        # duplicated, or ill-typed maps instead of silently discarding them.
        _validate_identity_maps(data, rebuilt.canonical_record)
        vertex_map = tuple(_pair_list(data, "vertex_map", "original", "canonical"))
        particle_map = tuple(_pair_list(data, "particle_map", "original", "canonical"))
        edge_map = tuple(_pair_list(data, "edge_occurrence_map", "old", "new"))
        final_map = tuple(_pair_list(data, "final_slot_map", "old", "new"))
        return cls(
            schema_version=SCHEMA_VERSION,
            mode=mode,
            topology_key=rebuilt.topology_key,
            key=rebuilt.key,
            digest=rebuilt.digest,
            canonical_record=record,
            vertex_map=vertex_map,
            particle_map=particle_map,
            edge_occurrence_map=edge_map,
            final_slot_map=final_map,
        )


def _validate_alignment_input(topology: Topology) -> None:
    """Validate before deriving a cache key or consulting the cache."""
    if not isinstance(topology, Topology):
        raise TypeError("align_topology expects a Topology object")
    report = topology.validate_fields()
    if not report.valid:
        raise ValueError(f"Topology fields must be valid before alignment: {report.to_dict()}")
    if topology.n != 5:
        raise ValueError("five-body alignment requires n=5")


def _exact_topology_cache_key(topology: Topology) -> tuple[Any, ...]:
    """Include every input field that can affect maps or occurrence order."""
    return (
        type(topology.n), topology.n,
        tuple((type(vertex), vertex) for vertex in topology.vertices),
        (type(topology.initial_vertex), topology.initial_vertex),
        tuple(
            (type(vertex), vertex, type(count), count)
            for vertex, count in topology.final_slots
        ),
        tuple(
            (
                type(edge.source), edge.source,
                type(edge.target), edge.target,
                edge.edge_type.value,
                type(edge.multiplicity), edge.multiplicity,
            )
            for edge in topology.edges
        ),
    )


def clear_alignment_cache() -> None:
    global _alignment_cache_hits, _alignment_cache_misses
    _alignment_cache.clear()
    _alignment_cache_hits = 0
    _alignment_cache_misses = 0


def alignment_cache_info() -> dict[str, int]:
    return {
        "size": len(_alignment_cache),
        "maxsize": _ALIGNMENT_CACHE_MAXSIZE,
        "hits": _alignment_cache_hits,
        "misses": _alignment_cache_misses,
    }


def _align_topology_reference(topology: Topology) -> TopologyAlignment:
    canonical = canonicalize_topology(topology)
    original_map = dict(canonical.original_to_canonical)
    canonical_topology, _ = _aggregate_topology(topology, original_map)
    expected = _topology_signature(canonical_topology)
    vertices = tuple(canonical_topology.vertices)
    if canonical_topology.initial_vertex != 0:
        raise ValueError("canonical topology must map initial vertex to zero")
    automorphisms: list[tuple[int, ...]] = []
    rest = tuple(vertex for vertex in vertices if vertex != 0)
    for permutation in itertools.permutations(rest):
        image = (0,) + permutation
        mapping = {vertex: image[vertex] for vertex in vertices}
        transformed, _ = _aggregate_topology(canonical_topology, mapping)
        if _topology_signature(transformed) == expected:
            automorphisms.append(image)
    identity = tuple(vertices)
    if identity not in automorphisms:
        automorphisms.insert(0, identity)
    return TopologyAlignment(
        canonical_topology=canonical_topology,
        topology_key=canonical.key,
        vertex_map=tuple(sorted(original_map.items())),
        automorphisms=tuple(sorted(automorphisms)),
    )


def align_topology_reference(topology: Topology) -> TopologyAlignment:
    """Reference full root-fixed permutation path for A/B verification."""
    _validate_alignment_input(topology)
    return _align_topology_reference(topology)


def _alignment_invariant_groups(topology: Topology) -> tuple[tuple[tuple[Any, ...], tuple[int, ...]], ...]:
    stats: dict[int, list[int | None]] = {
        vertex: [topology.slot_count(vertex), 0, 0, 0, 0]
        for vertex in topology.vertices
    }
    for edge in topology.edges:
        if edge.multiplicity <= 0:
            continue
        if edge.edge_type is EdgeType.SINGLE:
            stats[edge.source][2] = stats[edge.source][2] + edge.multiplicity
            stats[edge.target][1] = stats[edge.target][1] + edge.multiplicity
        else:
            stats[edge.source][4] = stats[edge.source][4] + edge.multiplicity
            stats[edge.target][3] = stats[edge.target][3] + edge.multiplicity
    groups: dict[tuple[Any, ...], list[int]] = {}
    for vertex in topology.vertices:
        if vertex == topology.initial_vertex:
            continue
        key = tuple(stats[vertex])
        groups.setdefault(key, []).append(vertex)
    return tuple(
        (key, tuple(vertices))
        for key, vertices in sorted(groups.items(), key=lambda item: item[0])
    )


def _align_topology_grouped(topology: Topology) -> TopologyAlignment:
    canonical = canonicalize_topology(topology)
    original_map = dict(canonical.original_to_canonical)
    canonical_topology, _ = _aggregate_topology(topology, original_map)
    expected = _topology_signature(canonical_topology)
    vertices = tuple(canonical_topology.vertices)
    if canonical_topology.initial_vertex != 0:
        raise ValueError("canonical topology must map initial vertex to zero")
    groups = _alignment_invariant_groups(canonical_topology)
    automorphisms: list[tuple[int, ...]] = []
    group_permutations = tuple(
        tuple(itertools.permutations(group_vertices))
        for _, group_vertices in groups
    )
    for selected in itertools.product(*group_permutations):
        mapping = {canonical_topology.initial_vertex: canonical_topology.initial_vertex}
        for (_, group_vertices), image_vertices in zip(groups, selected):
            mapping.update(dict(zip(group_vertices, image_vertices)))
        image = tuple(mapping[vertex] for vertex in vertices)
        transformed, _ = _aggregate_topology(canonical_topology, mapping)
        if _topology_signature(transformed) == expected:
            automorphisms.append(image)
    identity = tuple(vertices)
    if identity not in automorphisms:
        automorphisms.insert(0, identity)
    return TopologyAlignment(
        canonical_topology=canonical_topology,
        topology_key=canonical.key,
        vertex_map=tuple(sorted(original_map.items())),
        automorphisms=tuple(sorted(automorphisms)),
    )


def align_topology(
    topology: Topology,
    *,
    use_cache: bool = True,
    use_grouping: bool = True,
) -> TopologyAlignment:
    """Align with the grouped path, or use the independent reference path."""
    global _alignment_cache_hits, _alignment_cache_misses
    _validate_alignment_input(topology)
    if type(use_cache) is not bool:
        raise TypeError("use_cache must be a boolean")
    if type(use_grouping) is not bool:
        raise TypeError("use_grouping must be a boolean")
    if not use_grouping:
        return _align_topology_reference(topology)
    key = _exact_topology_cache_key(topology)
    if use_cache:
        cached = _alignment_cache.get(key)
        if cached is not None:
            _alignment_cache.move_to_end(key)
            _alignment_cache_hits += 1
            return cached
        _alignment_cache_misses += 1
    alignment = _align_topology_grouped(topology)
    if use_cache:
        _alignment_cache[key] = alignment
        _alignment_cache.move_to_end(key)
        while len(_alignment_cache) > _ALIGNMENT_CACHE_MAXSIZE:
            _alignment_cache.popitem(last=False)
    return alignment


def _marked_payload(
    record: CandidateMomentumRecord,
    vertex_map: Mapping[int, int],
    particle_map: Mapping[int, int] | None = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    particle_map = particle_map or {
        particle: particle for particle in range(1, record.topology.n + 1)
    }
    final_by_vertex: dict[int, list[int]] = {new: [] for new in set(vertex_map.values())}
    for label in record.complete_labeling.final_particles:
        final_by_vertex[vertex_map[label.vertex]].append(particle_map[label.particle_id])
    final_payload = tuple(
        tuple(sorted(final_by_vertex[index])) for index in sorted(final_by_vertex)
    )
    single_by_ref = {
        (label.edge_index, label.occurrence): (particle_map[label.particle_id],)
        for label in record.complete_labeling.single_lines
    }
    double_by_ref = {
        (label.edge_index, label.occurrence): tuple(
            sorted(particle_map[particle] for particle in label.particle_ids)
        )
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


def _particle_prefix_maps(
    record: CandidateMomentumRecord,
    vertex_map: Mapping[int, int],
) -> tuple[dict[int, int], ...]:
    """Return the exact block-minimal particle maps for one vertex map.

    The marked payload compares final-particle blocks before edge payloads.
    For a fixed vertex automorphism, any particle map whose block is not the
    smallest remaining particle block loses lexicographically at the first
    such vertex.  The only maps that can attain the minimum therefore assign
    consecutive target blocks to canonical vertices and enumerate every
    within-block permutation.  This is a representation optimization only;
    the payload, tie-break, relabeling, and identity serialization remain
    unchanged.
    """

    particles = tuple(range(1, record.topology.n + 1))
    groups: dict[int, list[int]] = {
        vertex: [] for vertex in sorted(set(vertex_map.values()))
    }
    for label in record.complete_labeling.final_particles:
        try:
            target_vertex = vertex_map[label.vertex]
        except KeyError as exc:
            raise ValueError("vertex_map does not cover the candidate final labels") from exc
        groups.setdefault(target_vertex, []).append(label.particle_id)
    source_particles = sorted(
        particle for group in groups.values() for particle in group
    )
    if tuple(source_particles) != particles:
        raise ValueError("candidate final particles must partition the particle IDs")

    remaining = list(particles)
    block_choices: list[tuple[tuple[int, ...], ...]] = []
    source_groups: list[tuple[int, ...]] = []
    for target_vertex in sorted(groups):
        source_group = tuple(sorted(groups[target_vertex]))
        target_block = tuple(remaining[:len(source_group)])
        del remaining[:len(source_group)]
        source_groups.append(source_group)
        block_choices.append(tuple(itertools.permutations(target_block)))
    if remaining:
        raise ValueError("candidate final particles do not cover all target blocks")

    maps: list[dict[int, int]] = []
    for selected in itertools.product(*block_choices):
        particle_map: dict[int, int] = {}
        for source_group, target_permutation in zip(source_groups, selected):
            particle_map.update(zip(source_group, target_permutation))
        maps.append(particle_map)
    return tuple(maps)


def _relabel_record(
    record: CandidateMomentumRecord,
    vertex_map: Mapping[int, int],
    particle_map: Mapping[int, int],
) -> RelabeledCandidateRecord:
    topology = record.topology
    fields = topology.validate_fields()
    labels = record.complete_labeling.validate_fields(topology)
    if not fields.valid or not labels.valid:
        raise ValueError("record must contain field-valid topology and complete labeling")
    mapped_topology, temporary_edges = _aggregate_topology(topology, vertex_map)
    single_entries: list[tuple[tuple[int, int], int, int]] = []
    double_entries: list[tuple[tuple[int, int], int, tuple[int, ...]]] = []
    for label in record.complete_labeling.single_lines:
        old_ref = (label.edge_index, label.occurrence)
        new_ref = temporary_edges[old_ref]
        single_entries.append((old_ref, new_ref[0], particle_map[label.particle_id]))
    for label in record.complete_labeling.double_lines:
        old_ref = (label.edge_index, label.occurrence)
        new_ref = temporary_edges[old_ref]
        subset = tuple(sorted(particle_map[particle] for particle in label.particle_ids))
        double_entries.append((old_ref, new_ref[0], subset))
    final_labels_by_vertex: dict[int, list[tuple[FinalParticleLabel, int]]] = {}
    for label in record.complete_labeling.final_particles:
        final_labels_by_vertex.setdefault(vertex_map[label.vertex], []).append(
            (label, particle_map[label.particle_id])
        )
    final_slot_map: dict[tuple[int, int], tuple[int, int]] = {}
    final_particles: list[FinalParticleLabel] = []
    for vertex in sorted(final_labels_by_vertex):
        entries = sorted(final_labels_by_vertex[vertex], key=lambda item: (item[1], item[0].slot))
        for new_slot, (old_label, new_particle) in enumerate(entries):
            final_slot_map[(old_label.vertex, old_label.slot)] = (vertex, new_slot)
            final_particles.append(FinalParticleLabel(vertex, new_slot, new_particle))
    edge_occurrence_map: dict[tuple[int, int], tuple[int, int]] = {}
    for edge_index, edge in enumerate(mapped_topology.edges):
        entries = (
            sorted(
                ((old_ref, particle) for old_ref, new_index, particle in single_entries if new_index == edge_index),
                key=lambda item: (item[1], item[0]),
            )
            if edge.edge_type is EdgeType.SINGLE
            else sorted(
                ((old_ref, subset) for old_ref, new_index, subset in double_entries if new_index == edge_index),
                key=lambda item: (item[1], item[0]),
            )
        )
        for new_occurrence, (old_ref, _) in enumerate(entries):
            edge_occurrence_map[old_ref] = (edge_index, new_occurrence)
    transformed_labeling = Labeling(
        final_particles,
        tuple(
            SingleLineLabel(
                edge_occurrence_map[(label.edge_index, label.occurrence)][0],
                edge_occurrence_map[(label.edge_index, label.occurrence)][1],
                particle_map[label.particle_id],
            )
            for label in record.complete_labeling.single_lines
        ),
        tuple(
            DoubleLineLabel(
                edge_occurrence_map[(label.edge_index, label.occurrence)][0],
                edge_occurrence_map[(label.edge_index, label.occurrence)][1],
                tuple(sorted(particle_map[particle] for particle in label.particle_ids)),
            )
            for label in record.complete_labeling.double_lines
        ),
    )
    normalization = normalize_candidate_labeling(mapped_topology, transformed_labeling)
    if normalization.stage != "momentum_candidate" or normalization.record is None:
        raise ValueError(f"relabeling did not preserve candidate record: {normalization.to_dict()}")
    return RelabeledCandidateRecord(
        record=normalization.record,
        vertex_map=tuple(sorted(vertex_map.items())),
        particle_map=tuple(sorted(particle_map.items())),
        edge_occurrence_map=tuple(sorted(edge_occurrence_map.items())),
        final_slot_map=tuple(sorted(final_slot_map.items())),
        normalization=normalization,
    )


def relabel_candidate_record(
    record: CandidateMomentumRecord,
    vertex_map: Mapping[int, int] | Iterable[tuple[int, int]],
    particle_map: Mapping[int, int] | Iterable[tuple[int, int]] | None = None,
) -> RelabeledCandidateRecord:
    if not isinstance(record, CandidateMomentumRecord):
        raise TypeError("relabel_candidate_record expects CandidateMomentumRecord")
    # A dataclass instance can still carry a certificate that is internally
    # well-shaped but no longer agrees with its complete labeling.  Recheck
    # before using it as identity evidence; otherwise relabeling could turn a
    # damaged record into a seemingly valid canonical candidate.
    review_candidate_record(record)
    vertices = _complete_bijection(vertex_map, record.topology.vertices, "vertex_map")
    particles = _complete_bijection(
        particle_map if particle_map is not None else ((p, p) for p in range(1, record.topology.n + 1)),
        range(1, record.topology.n + 1),
        "particle_map",
    )
    return _relabel_record(record, vertices, particles)


def canonicalize_candidate_labeling(
    record: CandidateMomentumRecord,
    mode: CountMode | str = CountMode.FIXED_LABELS,
    *,
    use_particle_prefix_pruning: bool = True,
) -> CandidateIdentity:
    if not isinstance(record, CandidateMomentumRecord):
        raise TypeError("canonicalize_candidate_labeling expects CandidateMomentumRecord")
    # Canonicalization is a public identity boundary, so it must not trust a
    # certificate merely because its serialized field shape is valid.
    review_candidate_record(record)
    canonical_mode = mode if isinstance(mode, CountMode) else CountMode(mode)
    if type(use_particle_prefix_pruning) is not bool:
        raise TypeError("use_particle_prefix_pruning must be a boolean")
    alignment = align_topology(record.topology)
    initial_map = dict(alignment.vertex_map)
    candidates: list[tuple[Any, tuple[int, ...], dict[int, int], dict[int, int], tuple[int, ...]]] = []
    if canonical_mode is CountMode.FIXED_LABELS:
        particle_permutations = (tuple(range(1, record.topology.n + 1)),)
        for permutation in particle_permutations:
            particle_map = {
                particle: permutation[particle - 1]
                for particle in range(1, record.topology.n + 1)
            }
            for automorphism in alignment.automorphisms:
                mapping = {
                    original: automorphism[canonical]
                    for original, canonical in initial_map.items()
                }
                payload = _marked_payload(record, mapping, particle_map)
                candidates.append((payload, automorphism, mapping, particle_map, permutation))
    elif not use_particle_prefix_pruning:
        # Preserve the complete 5! reference route as an explicit oracle.
        for permutation in itertools.permutations(range(1, record.topology.n + 1)):
            particle_map = {
                particle: permutation[particle - 1]
                for particle in range(1, record.topology.n + 1)
            }
            for automorphism in alignment.automorphisms:
                mapping = {
                    original: automorphism[canonical]
                    for original, canonical in initial_map.items()
                }
                payload = _marked_payload(record, mapping, particle_map)
                candidates.append((payload, automorphism, mapping, particle_map, permutation))
    else:
        for automorphism in alignment.automorphisms:
            mapping = {
                original: automorphism[canonical]
                for original, canonical in initial_map.items()
            }
            for particle_map in _particle_prefix_maps(record, mapping):
                permutation = tuple(
                    particle_map[particle]
                    for particle in range(1, record.topology.n + 1)
                )
                payload = _marked_payload(record, mapping, particle_map)
                candidates.append((payload, automorphism, mapping, particle_map, permutation))
    payload, _, winning_map, winning_particles, _ = min(
        candidates, key=lambda item: (item[0], item[1], item[4])
    )
    key = (SCHEMA_VERSION, canonical_mode.value, alignment.topology_key, payload[0], payload[1])
    digest = _digest(key)
    # The source record was reviewed above; call the internal transformer here
    # to avoid repeating the same evidence check for every winning mapping.
    transformed = _relabel_record(record, winning_map, winning_particles)
    return CandidateIdentity(
        schema_version=SCHEMA_VERSION,
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


# The shorter name is convenient for catalog callers while retaining the
# explicit candidate name for new code.
MarkedIdentity = CandidateIdentity


def _restore_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_restore_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _restore_tuples(item) for key, item in value.items()}
    return value


def _pair_list(
    data: Mapping[str, Any], field: str, left: str, right: str
) -> tuple[tuple[Any, Any], ...]:
    raw = data.get(field)
    if not isinstance(raw, list):
        raise TypeError(f"candidate identity {field} must be a list")
    pairs: list[tuple[Any, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping) or left not in item or right not in item:
            raise TypeError(
                f"candidate identity {field} entries must contain {left!r} and {right!r}"
            )
        old = item[left]
        new = item[right]
        if isinstance(old, list):
            old = tuple(old)
        if isinstance(new, list):
            new = tuple(new)
        pairs.append((old, new))
    return tuple(pairs)


def _validate_identity_maps(data: Mapping[str, Any], record: CandidateMomentumRecord) -> None:
    vertices = tuple(record.topology.vertices)
    vertex_items = _pair_list(data, "vertex_map", "original", "canonical")
    if len(vertex_items) != len(vertices):
        raise ValueError("vertex_map must cover every source vertex")
    if any(
        type(old) is not int or type(new) is not int
        for old, new in vertex_items
    ):
        raise TypeError("vertex_map entries must contain integer vertex IDs")
    if len({old for old, _ in vertex_items}) != len(vertex_items):
        raise ValueError("vertex_map sources must be unique")
    if {new for _, new in vertex_items} != set(vertices):
        raise ValueError("vertex_map targets must be a vertex bijection")

    particle_items = _pair_list(data, "particle_map", "original", "canonical")
    _complete_bijection(particle_items, range(1, record.topology.n + 1), "particle_map")

    edge_items = _pair_list(data, "edge_occurrence_map", "old", "new")
    if len(edge_items) != record.topology.edge_count:
        raise ValueError("edge_occurrence_map must cover every expanded internal edge")
    old_edges = [old for old, _ in edge_items]
    new_edges = [new for _, new in edge_items]
    if any(
        not isinstance(old, tuple) or len(old) != 2 or
        not isinstance(new, tuple) or len(new) != 2 or
        any(type(value) is not int for value in (*old, *new))
        for old, new in edge_items
    ):
        raise TypeError("edge_occurrence_map entries must contain integer pairs")
    if len(set(old_edges)) != len(old_edges) or len(set(new_edges)) != len(new_edges):
        raise ValueError("edge_occurrence_map must be one-to-one")
    expected_edges = {
        (index, occurrence)
        for index, edge in enumerate(record.topology.edges)
        if edge.multiplicity > 0
        for occurrence in range(edge.multiplicity)
    }
    if set(new_edges) != expected_edges:
        raise ValueError(
            "edge_occurrence_map targets must cover the canonical edge occurrences"
        )

    slot_items = _pair_list(data, "final_slot_map", "old", "new")
    expected_slots = {
        (vertex, slot)
        for vertex, count in record.topology.final_slots
        for slot in range(count)
    }
    if len(slot_items) != len(expected_slots):
        raise ValueError("final_slot_map must cover every final slot")
    old_slots = [old for old, _ in slot_items]
    new_slots = [new for _, new in slot_items]
    if any(
        not isinstance(old, tuple) or len(old) != 2 or
        not isinstance(new, tuple) or len(new) != 2 or
        any(type(value) is not int for value in (*old, *new))
        for old, new in slot_items
    ):
        raise TypeError("final_slot_map entries must contain integer pairs")
    if len(set(old_slots)) != len(old_slots) or len(set(new_slots)) != len(new_slots):
        raise ValueError("final_slot_map must be one-to-one")
    if set(new_slots) != expected_slots:
        raise ValueError("final_slot_map targets must cover the canonical final slots")


__all__ = [
    "SCHEMA_VERSION", "TopologyAlignment", "RelabeledCandidateRecord",
    "CandidateIdentity", "MarkedIdentity", "align_topology",
    "align_topology_reference", "clear_alignment_cache", "alignment_cache_info",
    "canonicalize_candidate_labeling", "relabel_candidate_record", "_marked_payload",
]
