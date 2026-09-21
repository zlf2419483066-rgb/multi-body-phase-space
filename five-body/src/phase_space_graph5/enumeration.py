"""Canonical structure enumeration for the public five-body L0--L3 scope.

The generator keeps the four-body reference route: Prüfer tree shapes, every
orientation, every initial attachment, every directed single-line multiset,
then structural validation and complete topology canonicalization.  The
public scope is deliberately opened one loop at a time through L3;
``structure.analyze_topology`` remains available for hand-built n=5 fixtures
with higher loop counts, while public L4/L5 enumeration remains closed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
import json
import math
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .canonical import canonicalize_topology, canonicalize_topology_reference
from .domain import DiagramSpec, EdgeType, InternalEdge, RunStatus, Topology, ValidationIssue
from .structure import analyze_topology


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class EnumerationLimits:
    max_candidates: int | None = None
    max_topologies: int | None = None
    max_skeletons: int | None = None
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_candidates", "max_topologies", "max_skeletons"):
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.max_seconds is not None:
            if isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float)):
                raise ValueError("max_seconds must be a non-negative number or None")
            if not math.isfinite(self.max_seconds) or self.max_seconds < 0:
                raise ValueError("max_seconds must be a non-negative number or None")

    def to_dict(self) -> dict[str, int | float | None]:
        return {"max_candidates": self.max_candidates, "max_topologies": self.max_topologies, "max_skeletons": self.max_skeletons, "max_seconds": self.max_seconds}


@dataclass(frozen=True, slots=True)
class PruningLedger:
    counts: tuple[tuple[str, int], ...] = ()

    def count(self, rule: str) -> int:
        return dict(self.counts).get(rule, 0)

    def to_dict(self) -> dict[str, int]:
        return dict(self.counts)


@dataclass(frozen=True, slots=True)
class EnumerationReport:
    n: int
    loop_count: int
    status: RunStatus
    expected_vertices: int
    skeleton_count: int
    labeled_tree_count: int
    orientation_count: int
    initial_position_count: int
    single_multiset_count: int
    raw_candidate_count: int
    structural_valid_count: int
    unique_count: int
    topologies: tuple[Topology, ...]
    canonical_keys: tuple[tuple[Any, ...], ...]
    ledger: PruningLedger
    issues: tuple[ValidationIssue, ...]
    limits: EnumerationLimits
    pruning_enabled: bool
    completed_skeleton_count: int = 0
    stop_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.status is RunStatus.COMPLETE

    @property
    def accepted_count(self) -> int:
        return self.unique_count

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n, "loop_count": self.loop_count, "status": self.status.value,
            "complete": self.complete, "expected_vertices": self.expected_vertices,
            "counts": {
                "skeletons": self.skeleton_count, "labeled_trees": self.labeled_tree_count,
                "orientations": self.orientation_count, "initial_positions": self.initial_position_count,
                "single_multisets": self.single_multiset_count, "raw_candidates": self.raw_candidate_count,
                "structural_valid": self.structural_valid_count, "unique": self.unique_count,
            },
            "topologies": [topology.to_dict() for topology in self.topologies],
            "canonical_keys": [_jsonable(key) for key in self.canonical_keys],
            "pruning_ledger": self.ledger.to_dict(), "issues": [issue.to_dict() for issue in self.issues],
            "limits": self.limits.to_dict(), "pruning_enabled": self.pruning_enabled,
            "completed_skeleton_count": self.completed_skeleton_count, "stop_reason": self.stop_reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class _MutableLedger:
    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def record(self, rule: str, amount: int = 1) -> None:
        self._counts[rule] += amount

    def freeze(self) -> PruningLedger:
        return PruningLedger(tuple(sorted(self._counts.items())))


def _decode_prufer(sequence: Sequence[int], vertex_count: int) -> tuple[tuple[int, int], ...]:
    degree = [1] * vertex_count
    for value in sequence:
        degree[value] += 1
    edges: list[tuple[int, int]] = []
    for value in sequence:
        leaf = min(index for index, count in enumerate(degree) if count == 1)
        edges.append((min(leaf, value), max(leaf, value)))
        degree[leaf] -= 1
        degree[value] -= 1
    remaining = [index for index, count in enumerate(degree) if count == 1]
    if len(remaining) != 2:
        raise RuntimeError("Prüfer decoding did not leave exactly two leaves")
    edges.append((min(remaining), max(remaining)))
    return tuple(sorted(edges))


def prufer_labeled_trees(vertex_count: int) -> Iterator[tuple[tuple[int, int], ...]]:
    if not _is_int(vertex_count) or vertex_count < 2:
        raise ValueError("vertex_count must be an integer >= 2")
    if vertex_count == 2:
        yield ((0, 1),)
        return
    for sequence in itertools.product(range(vertex_count), repeat=vertex_count - 2):
        yield _decode_prufer(sequence, vertex_count)


def _canonical_tree_key(edges: Iterable[tuple[int, int]], vertex_count: int, checkpoint: Callable[[], None] = lambda: None) -> tuple[tuple[int, int], ...]:
    edge_tuple = tuple(edges)
    best: tuple[tuple[int, int], ...] | None = None
    for order in itertools.permutations(range(vertex_count)):
        checkpoint()
        mapped = tuple(sorted((min(order[left], order[right]), max(order[left], order[right])) for left, right in edge_tuple))
        if best is None or mapped < best:
            best = mapped
    if best is None:
        raise ValueError("a tree must contain at least one edge")
    return best


def _tree_shape_signature(edges: Iterable[tuple[int, int]], vertex_count: int) -> str:
    adjacency = [[] for _ in range(vertex_count)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)

    def rooted(vertex: int, parent: int) -> str:
        return "(" + "".join(sorted(rooted(child, vertex) for child in adjacency[vertex] if child != parent)) + ")"

    return min(rooted(vertex, -1) for vertex in range(vertex_count))


def tree_skeletons(vertex_count: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    unique: dict[str, tuple[tuple[int, int], ...]] = {}
    for edges in prufer_labeled_trees(vertex_count):
        signature = _tree_shape_signature(edges, vertex_count)
        if signature not in unique:
            unique[signature] = _canonical_tree_key(edges, vertex_count)
    return tuple(sorted(unique.values()))


def _oriented_tree(skeleton: tuple[tuple[int, int], ...], mask: int) -> tuple[InternalEdge, ...]:
    result: list[InternalEdge] = []
    for index, (left, right) in enumerate(skeleton):
        if mask & (1 << index):
            left, right = right, left
        result.append(InternalEdge(left, right, EdgeType.DOUBLE))
    return tuple(result)


def _adjacency(vertices: Iterable[int], edges: Iterable[InternalEdge]) -> dict[int, set[int]]:
    result = {vertex: set() for vertex in vertices}
    for edge in edges:
        if edge.multiplicity > 0 and edge.source in result and edge.target in result:
            result[edge.source].add(edge.target)
    return result


def _would_create_cycle(adjacency: Mapping[int, set[int]], source: int, target: int) -> bool:
    if source == target:
        return True
    pending = [target]
    reached = {target}
    while pending:
        current = pending.pop()
        for child in adjacency.get(current, ()):
            if child == source:
                return True
            if child not in reached:
                reached.add(child)
                pending.append(child)
    return False


def _reachable(adjacency: Mapping[int, set[int]], initial: int) -> set[int]:
    pending = [initial]
    reached = {initial}
    while pending:
        current = pending.pop()
        for child in adjacency.get(current, ()):
            if child not in reached:
                reached.add(child)
                pending.append(child)
    return reached


def _unique_zero_indegree_vertex(adjacency: Mapping[int, set[int]], vertices: tuple[int, ...]) -> int | None:
    """Return the only zero-indegree vertex, or None when there is not exactly one.

    The pruned generator has already constructed a weakly connected DAG: the
    oriented double tree is acyclic and each selected single is rejected if it
    closes a directed cycle.  On that path, a unique zero-indegree vertex is
    exactly the vertex that reaches every vertex.  This helper must not be used
    by the unpruned route, which keeps its explicit BFS semantics.
    """
    indegree = {vertex: 0 for vertex in vertices}
    for source in vertices:
        for target in adjacency.get(source, ()):
            if target in indegree:
                indegree[target] += 1
    sources = tuple(vertex for vertex in vertices if indegree[vertex] == 0)
    return sources[0] if len(sources) == 1 else None


def _single_multisets(vertices: tuple[int, ...], base_edges: tuple[InternalEdge, ...], loop_count: int, ledger: _MutableLedger, prune: bool, checkpoint: Callable[[], None] = lambda: None) -> Iterator[tuple[tuple[int, int], ...]]:
    if loop_count == 0:
        yield ()
        return
    outgoing = Counter(edge.source for edge in base_edges for _ in range(edge.multiplicity))
    capacity = {vertex: 2 - outgoing.get(vertex, 0) for vertex in vertices}
    if prune and any(value < 0 for value in capacity.values()):
        ledger.record("outdegree_capacity")
        return
    pairs = tuple((source, target) for source in vertices for target in vertices if source != target)
    adjacency = _adjacency(vertices, base_edges)
    indegree = {vertex: 0 for vertex in vertices}
    if prune:
        for edge in base_edges:
            if edge.target in indegree:
                indegree[edge.target] += edge.multiplicity
    source_count = sum(1 for vertex in vertices if indegree[vertex] == 0) if prune else 0
    used_outgoing: Counter[int] = Counter()
    selected: list[tuple[int, int]] = []

    def visit(start: int, remaining: int) -> Iterator[tuple[tuple[int, int], ...]]:
        nonlocal source_count
        if prune and source_count > remaining + 1:
            ledger.record("initial_source_bound")
            return
        if remaining == 0:
            yield tuple(selected)
            return
        for index in range(start, len(pairs)):
            checkpoint()
            source, target = pairs[index]
            if prune and used_outgoing[source] >= capacity[source]:
                ledger.record("outdegree_capacity")
                continue
            if prune and _would_create_cycle(adjacency, source, target):
                ledger.record("directed_cycle")
                continue
            used_outgoing[source] += 1
            was_present = target in adjacency[source]
            if prune and indegree[target] == 0:
                source_count -= 1
            if prune:
                indegree[target] += 1
            adjacency[source].add(target)
            selected.append((source, target))
            yield from visit(index, remaining - 1)
            selected.pop()
            if not was_present:
                adjacency[source].remove(target)
            if prune:
                indegree[target] -= 1
                if indegree[target] == 0:
                    source_count += 1
            used_outgoing[source] -= 1

    yield from visit(0, loop_count)


def _build_topology(n: int, vertices: tuple[int, ...], double_edges: tuple[InternalEdge, ...], singles: tuple[tuple[int, int], ...], initial_vertex: int) -> Topology:
    single_counts = Counter(singles)
    edges = list(double_edges)
    edges.extend(InternalEdge(source, target, EdgeType.SINGLE, multiplicity=count) for (source, target), count in sorted(single_counts.items()))
    outgoing = Counter(edge.source for edge in edges for _ in range(edge.multiplicity))
    final_slots = {vertex: 2 - outgoing.get(vertex, 0) for vertex in vertices}
    return Topology(n, vertices, tuple(edges), initial_vertex, final_slots)


def _canonical_representative(topology: Topology, *, reference: bool = False) -> tuple[Topology, tuple[Any, ...]]:
    canonicalizer = canonicalize_topology_reference if reference else canonicalize_topology
    result = canonicalizer(topology)
    vertex_map = dict(result.original_to_canonical)
    edges = tuple(InternalEdge(vertex_map[edge.source], vertex_map[edge.target], edge.edge_type, edge.multiplicity) for edge in topology.edges)
    slots = {vertex_map[vertex]: count for vertex, count in topology.final_slots if vertex in vertex_map}
    representative = Topology(topology.n, tuple(range(len(vertex_map))), edges, vertex_map[topology.initial_vertex], slots)
    return representative, result.key


def _empty_report(n: int, loop_count: int, status: RunStatus, expected_vertices: int, limits: EnumerationLimits, issues: tuple[ValidationIssue, ...] = (), pruning_enabled: bool = True) -> EnumerationReport:
    return EnumerationReport(n, loop_count, status, expected_vertices, 0, 0, 0, 0, 0, 0, 0, 0, (), (), PruningLedger(), issues, limits, pruning_enabled)


def enumerate_topologies(n: int = 5, loop_count: int = 0, limits: EnumerationLimits | None = None, *, use_pruning: bool = True, use_reference_canonical: bool = False, cancel_requested: Callable[[], bool] | None = None) -> EnumerationReport:
    if not _is_int(n):
        raise TypeError("n must be an integer")
    if not _is_int(loop_count):
        raise TypeError("loop_count must be an integer")
    limits = EnumerationLimits() if limits is None else limits
    if not isinstance(limits, EnumerationLimits):
        raise TypeError("limits must be an EnumerationLimits object or None")
    if not isinstance(use_pruning, bool):
        raise TypeError("use_pruning must be a boolean")
    if not isinstance(use_reference_canonical, bool):
        raise TypeError("use_reference_canonical must be a boolean")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None")
    expected_vertices = n - 1 + loop_count
    spec = DiagramSpec(n=n, loop_count=loop_count)
    field_report = spec.validate_fields()
    scope_issues = list(field_report.issues)
    if field_report.valid and (n != 5 or loop_count not in (0, 1, 2, 3)):
        scope_issues.append(ValidationIssue("unsupported_enumeration_scope", "Topology enumeration currently opens only n=5, loop_count=0..3; structure fixtures may use higher loop counts, but public L4/L5 enumeration remains closed.", "scope"))
    if scope_issues:
        return _empty_report(n, loop_count, RunStatus.FAILED, expected_vertices, limits, tuple(scope_issues), use_pruning)
    if expected_vertices < 2:
        return _empty_report(n, loop_count, RunStatus.FAILED, expected_vertices, limits, (ValidationIssue("unsupported_vertex_count", "Enumeration requires at least two internal vertices.", "v"),), use_pruning)

    ledger = _MutableLedger()
    started = time.monotonic()
    stop_reason: str | None = None
    skeleton_count = completed_skeleton_count = labeled_tree_count = orientation_count = initial_position_count = single_multiset_count = raw_candidate_count = structural_valid_count = 0
    representatives: dict[tuple[Any, ...], Topology] = {}

    class StopSearch(Exception):
        def __init__(self, reason: str) -> None:
            self.reason = reason

    def checkpoint() -> None:
        if cancel_requested is not None and cancel_requested():
            raise StopSearch("cancel_requested")
        if limits.max_seconds is not None and time.monotonic() - started >= limits.max_seconds:
            raise StopSearch("limit_max_seconds")

    vertices = tuple(range(expected_vertices))
    try:
        checkpoint()
        for name in ("max_candidates", "max_topologies", "max_skeletons"):
            if getattr(limits, name) == 0:
                raise StopSearch("limit_" + name)
        shapes: set[tuple[tuple[int, int], ...]] = set()
        signatures: set[str] = set()
        for edges in prufer_labeled_trees(expected_vertices):
            checkpoint()
            signature = _tree_shape_signature(edges, expected_vertices)
            if signature not in signatures:
                signatures.add(signature)
                shapes.add(_canonical_tree_key(edges, expected_vertices, checkpoint))
            labeled_tree_count += 1
        for skeleton in sorted(shapes):
            checkpoint()
            if limits.max_skeletons is not None and skeleton_count >= limits.max_skeletons:
                raise StopSearch("limit_max_skeletons")
            skeleton_count += 1
            for mask in range(1 << len(skeleton)):
                checkpoint()
                orientation_count += 1
                double_edges = _oriented_tree(skeleton, mask)
                base_outgoing = Counter(edge.source for edge in double_edges)
                if use_pruning and any(base_outgoing[v] > 2 for v in vertices):
                    ledger.record("outdegree_capacity")
                    continue
                for singles in _single_multisets(vertices, double_edges, loop_count, ledger, use_pruning, checkpoint):
                    checkpoint()
                    single_multiset_count += 1
                    base = _build_topology(n, vertices, double_edges, singles, vertices[0])
                    adjacency = _adjacency(vertices, base.edges)
                    unique_source = _unique_zero_indegree_vertex(adjacency, vertices) if use_pruning else None
                    for initial_vertex in vertices:
                        checkpoint()
                        if limits.max_candidates is not None and raw_candidate_count >= limits.max_candidates:
                            raise StopSearch("limit_max_candidates")
                        initial_position_count += 1
                        raw_candidate_count += 1
                        if use_pruning and unique_source != initial_vertex:
                            ledger.record("initial_reachability")
                            continue
                        topology = Topology(n, vertices, base.edges, initial_vertex, base.final_slots)
                        report = analyze_topology(topology)
                        if not report.valid:
                            ledger.record("structural_invalid")
                            for code in report.all_failure_codes():
                                ledger.record(f"structural_{code}")
                            continue
                        structural_valid_count += 1
                        representative, key = _canonical_representative(topology, reference=use_reference_canonical)
                        if key in representatives:
                            ledger.record("canonical_duplicate")
                            continue
                        representatives[key] = representative
                        if limits.max_topologies is not None and len(representatives) >= limits.max_topologies:
                            raise StopSearch("limit_max_topologies")
            completed_skeleton_count += 1
    except StopSearch as exc:
        stop_reason = exc.reason
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    if stop_reason is not None:
        ledger.record(stop_reason)
    ordered = sorted(representatives.items())
    return EnumerationReport(n, loop_count, RunStatus.INCOMPLETE if stop_reason else RunStatus.COMPLETE, expected_vertices, skeleton_count, labeled_tree_count, orientation_count, initial_position_count, single_multiset_count, raw_candidate_count, structural_valid_count, len(ordered), tuple(topology for _, topology in ordered), tuple(key for key, _ in ordered), ledger.freeze(), (), limits, use_pruning, completed_skeleton_count, stop_reason)


__all__ = ["EnumerationLimits", "EnumerationReport", "PruningLedger", "enumerate_topologies", "prufer_labeled_trees", "tree_skeletons"]
