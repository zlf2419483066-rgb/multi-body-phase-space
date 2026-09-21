from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase_space_graph5 import EnumerationLimits, RunStatus, enumerate_topologies  # noqa: E402
from phase_space_graph5.domain import EdgeType, InternalEdge  # noqa: E402
import phase_space_graph5.enumeration as enumeration  # noqa: E402


class UniqueSourcePruningTests(unittest.TestCase):
    @staticmethod
    def _topology_projection(report):
        return tuple(topology.to_dict() for topology in report.topologies)

    def test_dag_unique_source_theorem_and_non_dag_boundary(self) -> None:
        dag = {0: {1, 2}, 1: {3}, 2: {3}, 3: set()}
        self.assertEqual(enumeration._unique_zero_indegree_vertex(dag, (0, 1, 2, 3)), 0)
        self.assertEqual(enumeration._reachable(dag, 0), {0, 1, 2, 3})

        multiple_sources = {0: {2}, 1: {2}, 2: set()}
        self.assertIsNone(enumeration._unique_zero_indegree_vertex(multiple_sources, (0, 1, 2)))
        self.assertNotEqual(enumeration._reachable(multiple_sources, 0), {0, 1, 2})

        # Unique zero indegree is unsafe without the DAG/connected-generator
        # precondition: a disconnected cyclic component has no source of its own.
        non_dag = {0: {1}, 1: set(), 2: {3}, 3: {2}}
        self.assertEqual(enumeration._unique_zero_indegree_vertex(non_dag, (0, 1, 2, 3)), 0)
        self.assertEqual(enumeration._reachable(non_dag, 0), {0, 1})

    def test_parallel_duplicate_edge_backtracking_keeps_source_count(self) -> None:
        vertices = (0, 1, 2, 3, 4)
        # Directed double tree with sources 1 and 4.
        base_edges = tuple(
            InternalEdge(source, target, EdgeType.DOUBLE)
            for source, target in ((1, 2), (1, 3), (4, 2), (4, 0))
        )
        ledger = enumeration._MutableLedger()
        rows = tuple(enumeration._single_multisets(vertices, base_edges, 2, ledger, True))

        # 0->1 twice makes source 1 positive while source 4 stays unique;
        # repeating 0->3 cannot remove either source 1 or 4.
        self.assertIn(((0, 1), (0, 1)), rows)
        self.assertNotIn(((0, 3), (0, 3)), rows)
        self.assertGreater(ledger.freeze().count("initial_source_bound"), 0)

    @staticmethod
    def _independent_valid_multiset(vertices, base_edges, singles) -> bool:
        outgoing = Counter()
        indegree = {vertex: 0 for vertex in vertices}
        adjacency = {vertex: set() for vertex in vertices}
        for edge in base_edges:
            outgoing[edge.source] += edge.multiplicity
            indegree[edge.target] += edge.multiplicity
            adjacency[edge.source].add(edge.target)
        for source, target in singles:
            outgoing[source] += 1
            indegree[target] += 1
            adjacency[source].add(target)
        if any(outgoing[vertex] > 2 for vertex in vertices):
            return False

        # Independent DFS cycle check.
        colors = {vertex: 0 for vertex in vertices}

        def visit(vertex: int) -> bool:
            colors[vertex] = 1
            for child in adjacency[vertex]:
                if colors[child] == 1:
                    return False
                if colors[child] == 0 and not visit(child):
                    return False
            colors[vertex] = 2
            return True

        if any(colors[vertex] == 0 and not visit(vertex) for vertex in vertices):
            return False
        sources = [vertex for vertex in vertices if indegree[vertex] == 0]
        if len(sources) != 1:
            return False
        reached = {sources[0]}
        pending = [sources[0]]
        while pending:
            source = pending.pop()
            for target in adjacency[source]:
                if target not in reached:
                    reached.add(target)
                    pending.append(target)
        return reached == set(vertices)

    def test_small_pair_multiset_domain_matches_independent_filter(self) -> None:
        vertices = (0, 1, 2, 3, 4)
        base_edges = tuple(
            InternalEdge(source, target, EdgeType.DOUBLE)
            for source, target in ((1, 2), (1, 3), (4, 2), (4, 0))
        )
        all_rows = tuple(enumeration._single_multisets(vertices, base_edges, 2, enumeration._MutableLedger(), False))
        pruned_rows = tuple(enumeration._single_multisets(vertices, base_edges, 2, enumeration._MutableLedger(), True))
        expected = tuple(
            row for row in all_rows if self._independent_valid_multiset(vertices, base_edges, row)
        )
        self.assertEqual(len(all_rows), 210)
        self.assertEqual(pruned_rows, expected)
        self.assertEqual(len(pruned_rows), len(set(pruned_rows)))

    def test_pruned_and_unpruned_l1_keep_full_outputs_but_real_counts_differ(self) -> None:
        pruned = enumerate_topologies(5, 1, use_pruning=True)
        unpruned = enumerate_topologies(5, 1, use_pruning=False)
        self.assertEqual(pruned.status, RunStatus.COMPLETE, pruned.to_dict())
        self.assertEqual(unpruned.status, RunStatus.COMPLETE, unpruned.to_dict())
        self.assertEqual(pruned.canonical_keys, unpruned.canonical_keys)
        self.assertEqual(self._topology_projection(pruned), self._topology_projection(unpruned))
        self.assertEqual(pruned.structural_valid_count, unpruned.structural_valid_count)
        self.assertEqual(pruned.unique_count, unpruned.unique_count)
        self.assertEqual(pruned.single_multiset_count, 144)
        self.assertEqual(pruned.raw_candidate_count, 720)
        self.assertEqual(pruned.initial_position_count, 720)
        self.assertEqual(pruned.ledger.count("initial_source_bound"), 198)
        self.assertEqual(unpruned.single_multiset_count, 960)
        self.assertEqual(unpruned.raw_candidate_count, 4800)
        self.assertEqual(unpruned.initial_position_count, 4800)
        self.assertEqual(unpruned.ledger.count("initial_source_bound"), 0)

    def test_unpruned_route_does_not_call_unique_source_helper(self) -> None:
        original = enumeration._unique_zero_indegree_vertex
        enumeration._unique_zero_indegree_vertex = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unpruned helper call"))
        try:
            report = enumerate_topologies(5, 0, EnumerationLimits(max_candidates=1), use_pruning=False)
        finally:
            enumeration._unique_zero_indegree_vertex = original
        self.assertEqual(report.status, RunStatus.INCOMPLETE, report.to_dict())
        self.assertEqual(report.raw_candidate_count, 1)

    def test_cancel_and_hard_candidate_limits_remain_actual(self) -> None:
        limited = enumerate_topologies(5, 1, EnumerationLimits(max_candidates=1), use_pruning=True)
        self.assertEqual(limited.status, RunStatus.INCOMPLETE, limited.to_dict())
        self.assertEqual(limited.stop_reason, "limit_max_candidates")
        self.assertEqual(limited.raw_candidate_count, 1)
        self.assertEqual(limited.initial_position_count, 1)
        self.assertEqual(limited.ledger.count("limit_max_candidates"), 1)

        cancelled = enumerate_topologies(5, 1, cancel_requested=lambda: True)
        self.assertEqual(cancelled.status, RunStatus.INCOMPLETE, cancelled.to_dict())
        self.assertEqual(cancelled.stop_reason, "cancel_requested")
        self.assertEqual(cancelled.raw_candidate_count, 0)
        self.assertEqual(cancelled.initial_position_count, 0)
        self.assertEqual(cancelled.ledger.count("cancel_requested"), 1)


if __name__ == "__main__":
    unittest.main()
