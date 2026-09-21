"""Structure-only checks for five-body complete-expansion graphs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from typing import Any, Iterable

from .domain import EdgeType, Topology, ValidationReport


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _safe_multiplicity(edge: Any) -> int:
    value = getattr(edge, "multiplicity", 0)
    return value if _is_int(value) and value > 0 else 0


@dataclass(frozen=True, slots=True)
class TopologyMetrics:
    n: int
    v: int
    d: int
    l: int
    edge_count: int
    loop_count: int
    full_components: int
    double_components: int
    double_cycle_rank: int
    final_slot_total: int
    outdegree_by_vertex: tuple[tuple[int, int], ...]

    @property
    def expected_d(self) -> int:
        return self.n - 2 + self.l

    @property
    def L(self) -> int:
        return self.loop_count

    @property
    def E(self) -> int:
        return self.edge_count

    @property
    def expected_v(self) -> int:
        return self.expected_d + 1

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n, "v": self.v, "d": self.d, "l": self.l,
            "edge_count": self.edge_count, "L": self.loop_count,
            "full_components": self.full_components,
            "double_components": self.double_components,
            "double_cycle_rank": self.double_cycle_rank,
            "final_slot_total": self.final_slot_total,
            "outdegree_by_vertex": [{"vertex": v, "outdegree": degree} for v, degree in self.outdegree_by_vertex],
            "expected_d": self.expected_d, "expected_v": self.expected_v,
        }


@dataclass(frozen=True, slots=True)
class StructuralCheck:
    code: str
    passed: bool
    message: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "passed": self.passed, "message": self.message, "details": self.details}


@dataclass(frozen=True, slots=True)
class TopologyStructureReport:
    metrics: TopologyMetrics
    field_report: ValidationReport
    checks: tuple[StructuralCheck, ...]

    @property
    def valid(self) -> bool:
        return self.field_report.valid and all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[StructuralCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def failed_codes(self) -> tuple[str, ...]:
        return tuple(check.code for check in self.failures)

    def all_failure_codes(self) -> tuple[str, ...]:
        return self.field_report.error_codes() + self.failed_codes()

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "metrics": self.metrics.to_dict(),
            "field_validation": self.field_report.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "failed_codes": list(self.failed_codes()),
            "all_failure_codes": list(self.all_failure_codes()),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class _DisjointSet:
    def __init__(self, vertices: Iterable[int]) -> None:
        self.parent = {vertex: vertex for vertex in vertices}
        self.rank = {vertex: 0 for vertex in vertices}

    def find(self, vertex: int) -> int:
        parent = self.parent[vertex]
        if parent != vertex:
            self.parent[vertex] = self.find(parent)
        return self.parent[vertex]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def component_count(self) -> int:
        return len({self.find(vertex) for vertex in self.parent})


def _valid_vertices(topology: Topology) -> tuple[int, ...]:
    return tuple(vertex for vertex in topology.vertices if _is_int(vertex))


def _edge_counts(topology: Topology) -> tuple[int, int]:
    doubles = sum(_safe_multiplicity(edge) for edge in topology.edges if edge.edge_type is EdgeType.DOUBLE)
    singles = sum(_safe_multiplicity(edge) for edge in topology.edges if edge.edge_type is not EdgeType.DOUBLE)
    return doubles, singles


def _undirected_components(vertices: tuple[int, ...], edges: Iterable[Any], edge_type: EdgeType | None = None) -> tuple[int, int]:
    vertex_set = set(vertices)
    dsu = _DisjointSet(vertices)
    count = 0
    for edge in edges:
        if edge_type is not None and edge.edge_type is not edge_type:
            continue
        multiplicity = _safe_multiplicity(edge)
        count += multiplicity
        if multiplicity and edge.source in vertex_set and edge.target in vertex_set:
            dsu.union(edge.source, edge.target)
    return dsu.component_count(), count


def _final_slot_counts(topology: Topology) -> tuple[int, dict[int, int], tuple[dict[str, object], ...]]:
    counts: dict[int, int] = {}
    invalid: list[dict[str, object]] = []
    total = 0
    for index, pair in enumerate(topology.final_slots):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            invalid.append({"path": f"final_slots[{index}]", "entry": repr(pair)})
            continue
        vertex, count = pair
        if not _is_int(vertex) or not _is_int(count) or count < 0:
            invalid.append({"path": f"final_slots[{index}]", "vertex": repr(vertex), "count": repr(count)})
            continue
        counts[vertex] = counts.get(vertex, 0) + count
        total += count
    return total, counts, tuple(invalid)


def _outdegree_data(topology: Topology, vertices: tuple[int, ...], slot_counts: dict[int, int]) -> tuple[tuple[tuple[int, int], ...], tuple[dict[str, object], ...]]:
    outgoing = {vertex: 0 for vertex in vertices}
    vertex_set = set(vertices)
    for edge in topology.edges:
        count = _safe_multiplicity(edge)
        if count and edge.source in vertex_set:
            outgoing[edge.source] += count
    result = tuple((vertex, outgoing[vertex] + slot_counts.get(vertex, 0)) for vertex in vertices)
    violations = tuple({"vertex": vertex, "actual": degree, "expected": 2} for vertex, degree in result if degree != 2)
    return result, violations


def _reachable_vertices(topology: Topology, vertices: tuple[int, ...]) -> tuple[int, ...]:
    vertex_set = set(vertices)
    if topology.initial_vertex not in vertex_set:
        return ()
    adjacency = {vertex: set() for vertex in vertices}
    for edge in topology.edges:
        if _safe_multiplicity(edge) and edge.source in vertex_set and edge.target in vertex_set:
            adjacency[edge.source].add(edge.target)
    queue: deque[int] = deque([topology.initial_vertex])
    reached = {topology.initial_vertex}
    while queue:
        source = queue.popleft()
        for target in sorted(adjacency[source]):
            if target not in reached:
                reached.add(target)
                queue.append(target)
    return tuple(sorted(reached))


def _dag_data(topology: Topology, vertices: tuple[int, ...]) -> tuple[bool, tuple[int, ...], tuple[tuple[int, int], ...]]:
    vertex_set = set(vertices)
    adjacency: dict[int, list[tuple[int, int]]] = {vertex: [] for vertex in vertices}
    indegree = {vertex: 0 for vertex in vertices}
    self_loops: list[tuple[int, int]] = []
    for edge in topology.edges:
        count = _safe_multiplicity(edge)
        if not count:
            continue
        if edge.source == edge.target:
            self_loops.append((edge.source, count))
        if edge.source not in vertex_set or edge.target not in vertex_set:
            continue
        adjacency[edge.source].append((edge.target, count))
        indegree[edge.target] += count
    queue: deque[int] = deque(sorted(vertex for vertex, degree in indegree.items() if degree == 0))
    while queue:
        source = queue.popleft()
        for target, count in adjacency[source]:
            indegree[target] -= count
            if indegree[target] == 0:
                queue.append(target)
    cyclic = tuple(sorted(vertex for vertex, degree in indegree.items() if degree > 0))
    return not self_loops and not cyclic, cyclic, tuple(self_loops)


def analyze_topology(topology: Topology) -> TopologyStructureReport:
    if not isinstance(topology, Topology):
        raise TypeError("analyze_topology expects a Topology object")
    field_report = topology.validate_fields()
    vertices = _valid_vertices(topology)
    vertex_set = set(vertices)
    v = len(topology.vertices)
    d, l = _edge_counts(topology)
    edge_count = d + l
    full_components, _ = _undirected_components(vertices, topology.edges)
    double_components, double_edge_count = _undirected_components(vertices, topology.edges, EdgeType.DOUBLE)
    double_cycle_rank = double_edge_count - len(vertex_set) + double_components
    loop_count = edge_count - len(vertex_set) + full_components
    final_slot_total, slot_counts, invalid_slots = _final_slot_counts(topology)
    outdegrees, outdegree_violations = _outdegree_data(topology, vertices, slot_counts)
    reachable = _reachable_vertices(topology, vertices)
    unreachable = tuple(sorted(vertex_set - set(reachable)))
    is_dag, cyclic_vertices, self_loops = _dag_data(topology, vertices)
    supported_scope = topology.n == 5
    metrics = TopologyMetrics(topology.n, v, d, l, edge_count, loop_count, full_components, double_components, double_cycle_rank, final_slot_total, outdegrees)
    checks = (
        StructuralCheck("supported_scope", supported_scope, "The five-body structure layer supports n=5.", {"n": topology.n, "supported_n": [5]}),
        StructuralCheck("double_subgraph_tree", bool(vertex_set) and double_components == 1 and double_edge_count == len(vertex_set) - 1 and double_cycle_rank == 0, "The undirected double-line subgraph must be a tree covering all internal vertices.", {"vertices": list(vertices), "double_edge_count": double_edge_count, "components": double_components, "cycle_rank": double_cycle_rank}),
        StructuralCheck("complete_expansion_counts", topology.n + d + l == 2 * v and d - l == topology.n - 2 and v == d + 1 and final_slot_total == topology.n, "The complete-expansion edge, vertex, and boundary counts must satisfy all defining equations.", {"n_plus_d_plus_l": topology.n + d + l, "two_v": 2 * v, "d_minus_l": d - l, "n_minus_2": topology.n - 2, "v": v, "expected_v": d + 1, "final_slot_total": final_slot_total, "expected_final_slot_total": topology.n, "invalid_final_slots": list(invalid_slots)}),
        StructuralCheck("outdegree_two", not outdegree_violations and not invalid_slots and set(slot_counts) == vertex_set, "Every internal vertex must have total outgoing count two, including final-state slots.", {"outdegree_by_vertex": [{"vertex": vertex, "actual": degree} for vertex, degree in outdegrees], "violations": list(outdegree_violations), "invalid_final_slots": list(invalid_slots)}),
        StructuralCheck("directed_acyclic", is_dag, "The complete internal directed graph must be a DAG without self-loops.", {"cyclic_vertices": list(cyclic_vertices), "self_loops": [{"vertex": vertex, "multiplicity": count} for vertex, count in self_loops]}),
        StructuralCheck("initial_reachability", bool(vertex_set) and not unreachable and topology.initial_vertex in vertex_set, "Every internal vertex must be reachable from the initial-state attachment vertex.", {"initial_vertex": topology.initial_vertex, "reachable": list(reachable), "unreachable": list(unreachable)}),
        StructuralCheck("loop_count_matches_single_lines", loop_count == l and full_components == 1, "The undirected independent loop count L must match the internal single-line count l.", {"L": loop_count, "l": l, "edge_count": edge_count, "v": len(vertex_set), "components": full_components}),
    )
    return TopologyStructureReport(metrics, field_report, checks)


__all__ = ["StructuralCheck", "TopologyMetrics", "TopologyStructureReport", "analyze_topology"]

