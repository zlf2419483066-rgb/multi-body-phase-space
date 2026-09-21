"""Exact canonical encodings for graph and variable equivalence (G04).

The current scope has at most six internal vertices and four final-state
particles.  Exhaustively trying internal-vertex permutations (and, for the
relabeling mode, all particle permutations) is therefore small enough to be a
transparent reference algorithm. Topology production keys minimize initial
and slot prefixes first; canonicalize_topology_reference retains all orders.
Equality is decided by the complete tuple
key; the optional SHA-256 digest is only a stable display/cache summary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Any, Iterable, Mapping

from .domain import CountMode, EdgeType, Labeling, Topology


Variable = tuple[int, ...]
"""Canonical particle subset used by a variable combination."""


def _canonical_mode(mode: CountMode | str) -> CountMode:
    if isinstance(mode, CountMode):
        return mode
    try:
        return CountMode(mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown canonicalization mode: {mode!r}") from exc


def _validate_topology_for_canonicalization(topology: Topology) -> tuple[int, ...]:
    if not isinstance(topology, Topology):
        raise TypeError("canonicalization expects a Topology object")
    report = topology.validate_fields()
    if not report.valid:
        raise ValueError(f"Topology fields must be valid before canonicalization: {report.to_dict()}")
    if topology.n not in (3, 4):
        raise ValueError("G04 currently supports n=3 and n=4 only")
    vertices = tuple(topology.vertices)
    if len(set(vertices)) != len(vertices):
        raise ValueError("Topology vertices must be unique before canonicalization")
    return vertices


def _effective_edge_records(topology: Topology) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (index, edge)
        for index, edge in enumerate(topology.edges)
        if edge.multiplicity > 0
    )


def _particle_permutations(n: int, mode: CountMode) -> tuple[tuple[int, ...], ...]:
    if mode is CountMode.FIXED_LABELS:
        return (tuple(range(1, n + 1)),)
    return tuple(itertools.permutations(range(1, n + 1)))


def _mapping_tuple(mapping: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(mapping.items()))


def _stable_digest(key: tuple[Any, ...]) -> str:
    payload = json.dumps(_jsonable(key), ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class CanonicalizationResult:
    """A complete canonical key and the transformations that produced it."""

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
            "original_to_canonical": [
                {"original": original, "canonical": canonical}
                for original, canonical in self.original_to_canonical
            ],
            "particle_mapping": [
                {"original": original, "canonical": canonical}
                for original, canonical in self.particle_mapping
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class VariableCombinationResult:
    """S4-canonical key for an unordered variable multiset."""

    n: int
    variables: tuple[Variable, ...]
    key: tuple[Any, ...]
    digest: str
    particle_mapping: tuple[tuple[int, int], ...]

    @property
    def canonical_key(self) -> tuple[Any, ...]:
        return self.key

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "variables": [list(variable) for variable in self.variables],
            "canonical_key": _jsonable(self.key),
            "digest": self.digest,
            "particle_mapping": [
                {"original": original, "canonical": canonical}
                for original, canonical in self.particle_mapping
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _topology_key_for_order(topology: Topology, order: tuple[int, ...]) -> tuple[Any, ...]:
    mapping = {original: canonical for canonical, original in enumerate(order)}
    slots = tuple(topology.slot_count(original) for original in order)
    aggregate: dict[tuple[int, int, str], int] = {}
    for _, edge in _effective_edge_records(topology):
        edge_key = (mapping[edge.source], mapping[edge.target], edge.edge_type.value)
        aggregate[edge_key] = aggregate.get(edge_key, 0) + edge.multiplicity
    edges = tuple(
        (source, target, edge_type, multiplicity)
        for (source, target, edge_type), multiplicity in sorted(aggregate.items())
    )
    return (
        "topology",
        topology.n,
        len(order),
        mapping[topology.initial_vertex],
        slots,
        edges,
    )


def canonicalize_topology_reference(topology: Topology) -> CanonicalizationResult:
    """Canonicalize an unmarked topology over all internal-vertex permutations."""

    vertices = _validate_topology_for_canonicalization(topology)
    candidates = tuple(
        (_topology_key_for_order(topology, order), order)
        for order in itertools.permutations(vertices)
    )
    key, order = min(candidates, key=lambda item: item[0])
    mapping = {original: canonical for canonical, original in enumerate(order)}
    identity = tuple((particle, particle) for particle in range(1, topology.n + 1))
    return CanonicalizationResult(
        key=key,
        digest=_stable_digest(key),
        mode=CountMode.FIXED_LABELS,
        original_to_canonical=_mapping_tuple(mapping),
        particle_mapping=identity,
    )


def canonicalize_topology(topology: Topology) -> CanonicalizationResult:
    """Minimize the unchanged key, fixing its initial and slot prefixes first.

    The initial vertex must map to zero. Remaining slots must be ascending;
    only permutations inside equal-slot groups can improve the edge suffix.
    Groups and permutations retain reference iteration order for mapping ties.
    """
    vertices = _validate_topology_for_canonicalization(topology)
    slots = dict(topology.final_slots)
    groups: dict[int, list[int]] = {}
    for vertex in vertices:
        if vertex != topology.initial_vertex:
            groups.setdefault(slots[vertex], []).append(vertex)
    aggregate: dict[tuple[int, int, str], int] = {}
    for _, edge in _effective_edge_records(topology):
        k = (edge.source, edge.target, edge.edge_type.value)
        aggregate[k] = aggregate.get(k, 0) + edge.multiplicity
    records = tuple((*k, count) for k, count in aggregate.items())
    best_key = None
    best_order = None
    for parts in itertools.product(*(itertools.permutations(groups[k]) for k in sorted(groups))):
        order = (topology.initial_vertex,) + tuple(v for part in parts for v in part)
        mapping = {v: i for i, v in enumerate(order)}
        edges = tuple(sorted((mapping[s], mapping[t], kind, count) for s, t, kind, count in records))
        key = ('topology', topology.n, len(vertices), 0, tuple(slots[v] for v in order), edges)
        if best_key is None or key < best_key:
            best_key, best_order = key, order
    return CanonicalizationResult(
        key=best_key, digest=_stable_digest(best_key), mode=CountMode.FIXED_LABELS,
        original_to_canonical=_mapping_tuple({v: i for i, v in enumerate(best_order)}),
        particle_mapping=tuple((p, p) for p in range(1, topology.n + 1)),
    )


def _normalize_particle_subset(subset: Iterable[int], n: int) -> Variable:
    try:
        values = tuple(subset)
    except TypeError as exc:
        raise TypeError("a variable subset must be iterable") from exc
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("variable particle IDs must be integers")
    if len(set(values)) != len(values):
        raise ValueError("a variable subset cannot repeat a particle")
    normalized = tuple(sorted(values))
    if len(normalized) < 2 or len(normalized) >= n:
        raise ValueError("variable subsets must have size 2..n-1")
    if any(value < 1 or value > n for value in normalized):
        raise ValueError(f"variable particle IDs must lie in 1..{n}")
    return normalized


def canonicalize_variable_set(
    variables: Iterable[Iterable[int]],
    n: int,
) -> VariableCombinationResult:
    """Return the canonical key under one common S_n particle permutation.

    The input order is ignored, but repeated subsets remain repeated so a
    downstream independence check cannot lose multiplicity information.
    """

    if n not in (3, 4):
        raise ValueError("G04 currently supports n=3 and n=4 only")
    normalized = tuple(_normalize_particle_subset(variable, n) for variable in variables)
    candidates: list[tuple[tuple[Any, ...], tuple[int, ...]]] = []
    for permutation in itertools.permutations(range(1, n + 1)):
        particle_map = {original: permutation[original - 1] for original in range(1, n + 1)}
        transformed = tuple(
            sorted(
                tuple(sorted(particle_map[particle] for particle in variable))
                for variable in normalized
            )
        )
        key = ("variable_combination", n, transformed)
        candidates.append((key, permutation))
    key, permutation = min(candidates, key=lambda item: item[0])
    mapping = {original: permutation[original - 1] for original in range(1, n + 1)}
    return VariableCombinationResult(
        n=n,
        variables=normalized,
        key=key,
        digest=_stable_digest(key),
        particle_mapping=_mapping_tuple(mapping),
    )


def _marked_key_for_order(
    topology: Topology,
    labeling: Labeling,
    order: tuple[int, ...],
    particle_map: Mapping[int, int],
) -> tuple[Any, ...]:
    vertex_map = {original: canonical for canonical, original in enumerate(order)}
    slots = tuple(topology.slot_count(original) for original in order)

    final_by_vertex: dict[int, list[int]] = {vertex_map[vertex]: [] for vertex in order}
    for label in labeling.final_particles:
        if label.vertex in vertex_map:
            final_by_vertex[vertex_map[label.vertex]].append(particle_map.get(label.particle_id, -1))
    final_labels = tuple(tuple(sorted(final_by_vertex[index])) for index in range(len(order)))

    single_by_ref = {
        (label.edge_index, label.occurrence): particle_map.get(label.particle_id, -1)
        for label in labeling.single_lines
    }
    double_by_ref = {
        (label.edge_index, label.occurrence): tuple(
            sorted(particle_map.get(particle, -1) for particle in label.particle_ids)
        )
        for label in labeling.double_lines
    }
    aggregate: dict[tuple[int, int, str], dict[str, Any]] = {}
    for edge_index, edge in _effective_edge_records(topology):
        structural_key = (vertex_map[edge.source], vertex_map[edge.target], edge.edge_type.value)
        entry = aggregate.setdefault(
            structural_key,
            {"multiplicity": 0, "single": [], "double": []},
        )
        entry["multiplicity"] += edge.multiplicity
        if edge.edge_type is EdgeType.SINGLE:
            entry["single"].extend(
                single_by_ref.get((edge_index, occurrence), -1)
                for occurrence in range(edge.multiplicity)
            )
        else:
            entry["double"].extend(
                double_by_ref.get((edge_index, occurrence), (-1,))
                for occurrence in range(edge.multiplicity)
            )
    edges = []
    for (source, target, edge_type), entry in sorted(aggregate.items()):
        labels = (
            tuple(sorted(entry["single"]))
            if edge_type == EdgeType.SINGLE.value
            else tuple(sorted(entry["double"]))
        )
        edges.append((source, target, edge_type, entry["multiplicity"], labels))
    return (
        "labeled_topology",
        topology.n,
        len(order),
        vertex_map[topology.initial_vertex],
        slots,
        final_labels,
        tuple(edges),
    )


def canonicalize_labeling_reference(
    topology: Topology,
    labeling: Labeling,
    mode: CountMode | str = CountMode.FIXED_LABELS,
) -> CanonicalizationResult:
    """Canonicalize a marked graph under vertex and optional particle permutations."""

    vertices = _validate_topology_for_canonicalization(topology)
    if not isinstance(labeling, Labeling):
        raise TypeError("canonicalize_labeling expects a Labeling object")
    labeling_report = labeling.validate_fields(topology)
    if not labeling_report.valid:
        raise ValueError(f"Labeling fields must be valid before canonicalization: {labeling_report.to_dict()}")
    canonical_mode = _canonical_mode(mode)
    candidates: list[tuple[tuple[Any, ...], tuple[int, ...], tuple[int, ...]]] = []
    for order in itertools.permutations(vertices):
        for permutation in _particle_permutations(topology.n, canonical_mode):
            particle_map = {
                original: permutation[original - 1]
                for original in range(1, topology.n + 1)
            }
            key = _marked_key_for_order(topology, labeling, order, particle_map)
            candidates.append((key, order, permutation))
    key, order, permutation = min(candidates, key=lambda item: item[0])
    vertex_map = {original: canonical for canonical, original in enumerate(order)}
    particle_map = {
        original: permutation[original - 1]
        for original in range(1, topology.n + 1)
    }
    return CanonicalizationResult(
        key=key,
        digest=_stable_digest(key),
        mode=canonical_mode,
        original_to_canonical=_mapping_tuple(vertex_map),
        particle_mapping=_mapping_tuple(particle_map),
    )


def canonicalize_labeling(
    topology: Topology,
    labeling: Labeling,
    mode: CountMode | str = CountMode.FIXED_LABELS,
) -> CanonicalizationResult:
    """Minimize the original marked key by its initial/slot/label prefixes.

    The marked key places final labels BEFORE internal edges. Restricting to
    unmarked canonical orders or topology automorphisms alone would therefore
    change the persisted key. Instead, for each particle permutation sort the
    remaining vertices by (slot count, final-label tuple), retaining all orders
    inside ties. Select ties in the original exhaustive iteration order.
    """
    vertices = _validate_topology_for_canonicalization(topology)
    if not isinstance(labeling, Labeling):
        raise TypeError('canonicalize_labeling expects a Labeling object')
    report = labeling.validate_fields(topology)
    if not report.valid:
        raise ValueError(f'Labeling fields must be valid before canonicalization: {report.to_dict()}')
    canonical_mode = _canonical_mode(mode)
    ranks = {vertex: index for index, vertex in enumerate(vertices)}
    best = None
    for permutation in _particle_permutations(topology.n, canonical_mode):
        particle_map = dict(enumerate(permutation, 1))
        final = {vertex: [] for vertex in vertices}
        for label in labeling.final_particles:
            final[label.vertex].append(particle_map[label.particle_id])
        groups = {}
        for vertex in vertices:
            if vertex != topology.initial_vertex:
                prefix = (topology.slot_count(vertex), tuple(sorted(final[vertex])))
                groups.setdefault(prefix, []).append(vertex)
        for parts in itertools.product(*(itertools.permutations(groups[k]) for k in sorted(groups))):
            order = (topology.initial_vertex,) + tuple(v for part in parts for v in part)
            key = _marked_key_for_order(topology, labeling, order, particle_map)
            candidate = (key, tuple(ranks[v] for v in order), permutation, order)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    key, _, permutation, order = best
    return CanonicalizationResult(
        key, _stable_digest(key), canonical_mode,
        _mapping_tuple({v: i for i, v in enumerate(order)}),
        tuple(enumerate(permutation, 1)),
    )
