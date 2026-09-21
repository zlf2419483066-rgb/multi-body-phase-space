"""Complete topology and marked-label canonical keys for the five-body package."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Any, Iterable, Mapping

from .domain import CountMode, EdgeType, Labeling, Topology


def _mode(value: CountMode | str) -> CountMode:
    if isinstance(value, CountMode):
        return value
    try:
        return CountMode(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown canonicalization mode: {value!r}") from exc


def _validate_topology(topology: Topology) -> tuple[int, ...]:
    if not isinstance(topology, Topology):
        raise TypeError("canonicalization expects a Topology object")
    report = topology.validate_fields()
    if not report.valid:
        raise ValueError(f"Topology fields must be valid before canonicalization: {report.to_dict()}")
    if topology.n != 5:
        raise ValueError("five-body canonicalization requires n=5")
    if len(set(topology.vertices)) != len(topology.vertices):
        raise ValueError("Topology vertices must be unique before canonicalization")
    return tuple(topology.vertices)


def _effective_edges(topology: Topology) -> tuple[tuple[int, Any], ...]:
    return tuple((index, edge) for index, edge in enumerate(topology.edges) if edge.multiplicity > 0)


def _particle_permutations(n: int, mode: CountMode) -> tuple[tuple[int, ...], ...]:
    return (tuple(range(1, n + 1)),) if mode is CountMode.FIXED_LABELS else tuple(itertools.permutations(range(1, n + 1)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _digest(key: tuple[Any, ...]) -> str:
    payload = json.dumps(_jsonable(key), ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalizationResult:
    key: tuple[Any, ...]
    digest: str
    mode: CountMode
    original_to_canonical: tuple[tuple[int, int], ...]
    particle_mapping: tuple[tuple[int, int], ...]

    @property
    def canonical_key(self) -> tuple[Any, ...]:
        return self.key

    @property
    def vertex_mapping(self) -> tuple[tuple[int, int], ...]:
        return self.original_to_canonical

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_key": _jsonable(self.key),
            "digest": self.digest,
            "mode": self.mode.value,
            "original_to_canonical": [{"original": a, "canonical": b} for a, b in self.original_to_canonical],
            "particle_mapping": [{"original": a, "canonical": b} for a, b in self.particle_mapping],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _topology_key_for_order(topology: Topology, order: tuple[int, ...]) -> tuple[Any, ...]:
    mapping = {original: canonical for canonical, original in enumerate(order)}
    slots = tuple(topology.slot_count(original) for original in order)
    aggregate: dict[tuple[int, int, str], int] = {}
    for _, edge in _effective_edges(topology):
        key = (mapping[edge.source], mapping[edge.target], edge.edge_type.value)
        aggregate[key] = aggregate.get(key, 0) + edge.multiplicity
    edges = tuple((source, target, kind, count) for (source, target, kind), count in sorted(aggregate.items()))
    return ("topology", topology.n, len(order), mapping[topology.initial_vertex], slots, edges)


def canonicalize_topology_reference(topology: Topology) -> CanonicalizationResult:
    vertices = _validate_topology(topology)
    candidates = tuple((_topology_key_for_order(topology, order), order) for order in itertools.permutations(vertices))
    key, order = min(candidates, key=lambda item: item[0])
    mapping = {original: canonical for canonical, original in enumerate(order)}
    return CanonicalizationResult(key, _digest(key), CountMode.FIXED_LABELS, tuple(sorted(mapping.items())), tuple((p, p) for p in range(1, topology.n + 1)))


def canonicalize_topology(topology: Topology) -> CanonicalizationResult:
    vertices = _validate_topology(topology)
    slots = dict(topology.final_slots)
    groups: dict[int, list[int]] = {}
    for vertex in vertices:
        if vertex != topology.initial_vertex:
            groups.setdefault(slots[vertex], []).append(vertex)
    aggregate: dict[tuple[int, int, str], int] = {}
    for _, edge in _effective_edges(topology):
        key = (edge.source, edge.target, edge.edge_type.value)
        aggregate[key] = aggregate.get(key, 0) + edge.multiplicity
    records = tuple((*key, count) for key, count in aggregate.items())
    best_key: tuple[Any, ...] | None = None
    best_order: tuple[int, ...] | None = None
    for parts in itertools.product(*(itertools.permutations(groups[key]) for key in sorted(groups))):
        order = (topology.initial_vertex,) + tuple(vertex for part in parts for vertex in part)
        mapping = {vertex: index for index, vertex in enumerate(order)}
        edges = tuple(sorted((mapping[source], mapping[target], kind, count) for source, target, kind, count in records))
        key = ("topology", topology.n, len(vertices), 0, tuple(slots[v] for v in order), edges)
        if best_key is None or key < best_key:
            best_key, best_order = key, order
    assert best_key is not None and best_order is not None
    return CanonicalizationResult(best_key, _digest(best_key), CountMode.FIXED_LABELS, tuple(sorted({v: i for i, v in enumerate(best_order)}.items())), tuple((p, p) for p in range(1, topology.n + 1)))


def _marked_key_for_order(topology: Topology, labeling: Labeling, order: tuple[int, ...], particle_map: Mapping[int, int]) -> tuple[Any, ...]:
    vertex_map = {original: canonical for canonical, original in enumerate(order)}
    slots = tuple(topology.slot_count(original) for original in order)
    final: dict[int, list[int]] = {vertex_map[vertex]: [] for vertex in order}
    for label in labeling.final_particles:
        if label.vertex in vertex_map:
            final[vertex_map[label.vertex]].append(particle_map.get(label.particle_id, -1))
    final_labels = tuple(tuple(sorted(final[index])) for index in range(len(order)))
    single_by_ref = {(label.edge_index, label.occurrence): particle_map.get(label.particle_id, -1) for label in labeling.single_lines}
    double_by_ref = {(label.edge_index, label.occurrence): tuple(sorted(particle_map.get(p, -1) for p in label.particle_ids)) for label in labeling.double_lines}
    aggregate: dict[tuple[int, int, str], dict[str, Any]] = {}
    for edge_index, edge in _effective_edges(topology):
        structural_key = (vertex_map[edge.source], vertex_map[edge.target], edge.edge_type.value)
        entry = aggregate.setdefault(structural_key, {"multiplicity": 0, "single": [], "double": []})
        entry["multiplicity"] += edge.multiplicity
        target = entry["single"] if edge.edge_type is EdgeType.SINGLE else entry["double"]
        lookup = single_by_ref if edge.edge_type is EdgeType.SINGLE else double_by_ref
        target.extend(lookup.get((edge_index, occurrence), -1 if edge.edge_type is EdgeType.SINGLE else (-1,)) for occurrence in range(edge.multiplicity))
    edges = []
    for (source, target, kind), entry in sorted(aggregate.items()):
        labels = tuple(sorted(entry["single"])) if kind == EdgeType.SINGLE.value else tuple(sorted(entry["double"]))
        edges.append((source, target, kind, entry["multiplicity"], labels))
    return ("labeled_topology", topology.n, len(order), vertex_map[topology.initial_vertex], slots, final_labels, tuple(edges))


def _validate_labeling(topology: Topology, labeling: Labeling) -> None:
    _validate_topology(topology)
    if not isinstance(labeling, Labeling):
        raise TypeError("canonicalize_labeling expects a Labeling object")
    report = labeling.validate_fields(topology)
    if not report.valid:
        raise ValueError(f"Labeling fields must be valid before canonicalization: {report.to_dict()}")


def canonicalize_labeling_reference(topology: Topology, labeling: Labeling, mode: CountMode | str = CountMode.FIXED_LABELS) -> CanonicalizationResult:
    _validate_labeling(topology, labeling)
    canonical_mode = _mode(mode)
    candidates = []
    for order in itertools.permutations(topology.vertices):
        for permutation in _particle_permutations(topology.n, canonical_mode):
            particle_map = {original: permutation[original - 1] for original in range(1, topology.n + 1)}
            candidates.append((_marked_key_for_order(topology, labeling, order, particle_map), order, permutation))
    key, order, permutation = min(candidates, key=lambda item: item[0])
    return CanonicalizationResult(key, _digest(key), canonical_mode, tuple(sorted({v: i for i, v in enumerate(order)}.items())), tuple(enumerate(permutation, 1)))


def canonicalize_labeling(topology: Topology, labeling: Labeling, mode: CountMode | str = CountMode.FIXED_LABELS) -> CanonicalizationResult:
    _validate_labeling(topology, labeling)
    canonical_mode = _mode(mode)
    ranks = {vertex: index for index, vertex in enumerate(topology.vertices)}
    best = None
    for permutation in _particle_permutations(topology.n, canonical_mode):
        particle_map = dict(enumerate(permutation, 1))
        final = {vertex: [] for vertex in topology.vertices}
        for label in labeling.final_particles:
            final[label.vertex].append(particle_map[label.particle_id])
        groups: dict[tuple[int | None, tuple[int, ...]], list[int]] = {}
        for vertex in topology.vertices:
            if vertex != topology.initial_vertex:
                prefix = (topology.slot_count(vertex), tuple(sorted(final[vertex])))
                groups.setdefault(prefix, []).append(vertex)
        for parts in itertools.product(*(itertools.permutations(groups[key]) for key in sorted(groups))):
            order = (topology.initial_vertex,) + tuple(vertex for part in parts for vertex in part)
            key = _marked_key_for_order(topology, labeling, order, particle_map)
            candidate = (key, tuple(ranks[v] for v in order), permutation, order)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise ValueError("cannot canonicalize a labeling without vertices")
    key, _, permutation, order = best
    return CanonicalizationResult(key, _digest(key), canonical_mode, tuple(sorted({v: i for i, v in enumerate(order)}.items())), tuple(enumerate(permutation, 1)))


__all__ = ["CanonicalizationResult", "canonicalize_labeling", "canonicalize_labeling_reference", "canonicalize_topology", "canonicalize_topology_reference"]

