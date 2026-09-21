import json
import unittest

from phase_space_graph import EdgeType, InternalEdge, Topology, analyze_topology


def f01_tree() -> Topology:
    return Topology(
        n=4,
        vertices=(0, 1, 2),
        edges=(
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.DOUBLE),
        ),
        initial_vertex=0,
        final_slots={0: 1, 1: 1, 2: 2},
    )


def f02_branching_tree() -> Topology:
    return Topology(
        n=4,
        vertices=(0, 1, 2),
        edges=(
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(0, 2, EdgeType.DOUBLE),
        ),
        initial_vertex=0,
        final_slots={0: 0, 1: 2, 2: 2},
    )


def f03_single_loop() -> Topology:
    return Topology(
        n=4,
        vertices=(0, 1, 2, 3),
        edges=(
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(0, 2, EdgeType.DOUBLE),
            InternalEdge(2, 3, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.SINGLE),
        ),
        initial_vertex=0,
        final_slots={0: 0, 1: 1, 2: 1, 3: 2},
    )


def f04_same_endpoint_lines() -> Topology:
    return Topology(
        n=3,
        vertices=(0, 1, 2),
        edges=(
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.DOUBLE),
            InternalEdge(0, 1, EdgeType.SINGLE),
        ),
        initial_vertex=0,
        final_slots={0: 0, 1: 1, 2: 2},
    )


class ValidFixtureTests(unittest.TestCase):
    def test_tree_and_branching_tree_pass_without_indegree_rule(self) -> None:
        for topology in (f01_tree(), f02_branching_tree()):
            with self.subTest(topology=topology):
                report = analyze_topology(topology)
                self.assertTrue(report.valid, report.to_dict())
                self.assertEqual(report.failed_codes(), ())
                self.assertEqual(report.metrics.L, 0)
                self.assertEqual(report.metrics.l, 0)
                self.assertEqual(report.metrics.d, 2)
                self.assertEqual(report.metrics.v, 3)
                self.assertEqual(report.metrics.outdegree_by_vertex, ((0, 2), (1, 2), (2, 2)))

    def test_single_loop_with_two_incoming_lines_passes(self) -> None:
        report = analyze_topology(f03_single_loop())
        self.assertTrue(report.valid, report.to_dict())
        self.assertEqual(report.metrics.n, 4)
        self.assertEqual(report.metrics.v, 4)
        self.assertEqual(report.metrics.d, 3)
        self.assertEqual(report.metrics.l, 1)
        self.assertEqual(report.metrics.L, 1)
        self.assertEqual(report.metrics.double_components, 1)
        self.assertEqual(report.metrics.double_cycle_rank, 0)
        self.assertNotIn("indegree_one", report.failed_codes())

    def test_same_endpoint_single_and_double_lines_pass(self) -> None:
        report = analyze_topology(f04_same_endpoint_lines())
        self.assertTrue(report.valid, report.to_dict())
        self.assertEqual(report.metrics.edge_count, 3)
        self.assertEqual(report.metrics.L, 1)
        self.assertEqual(report.metrics.l, 1)

    def test_report_is_json_serializable(self) -> None:
        report = analyze_topology(f03_single_loop())
        encoded = report.to_json()
        self.assertEqual(json.loads(encoded)["valid"], True)


class StructuralFailureTests(unittest.TestCase):
    def test_directed_cycle_is_reported_even_when_count_relations_hold(self) -> None:
        topology = Topology(
            n=3,
            vertices=(0, 1, 2),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(2, 1, EdgeType.SINGLE),
            ),
            initial_vertex=0,
            final_slots={0: 1, 1: 1, 2: 1},
        )
        report = analyze_topology(topology)
        self.assertFalse(report.valid)
        self.assertIn("directed_acyclic", report.failed_codes())
        self.assertEqual(report.metrics.L, 1)
        self.assertEqual(report.metrics.l, 1)
        dag = next(check for check in report.checks if check.code == "directed_acyclic")
        self.assertEqual(dag.details["cyclic_vertices"], [1, 2])

    def test_self_loop_is_reported_separately(self) -> None:
        topology = Topology(
            n=3,
            vertices=(0, 1, 2),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(1, 1, EdgeType.SINGLE),
            ),
            initial_vertex=0,
            final_slots={0: 1, 1: 1, 2: 1},
        )
        report = analyze_topology(topology)
        dag = next(check for check in report.checks if check.code == "directed_acyclic")
        self.assertFalse(dag.passed)
        self.assertEqual(dag.details["self_loops"], [{"vertex": 1, "multiplicity": 1}])

    def test_disconnected_double_subgraph_and_reachability_are_located(self) -> None:
        topology = Topology(
            n=4,
            vertices=(0, 1, 2),
            edges=(InternalEdge(0, 1, EdgeType.DOUBLE),),
            initial_vertex=0,
            final_slots={0: 1, 1: 1, 2: 2},
        )
        report = analyze_topology(topology)
        self.assertFalse(report.valid)
        self.assertIn("double_subgraph_tree", report.failed_codes())
        self.assertIn("initial_reachability", report.failed_codes())
        tree = next(check for check in report.checks if check.code == "double_subgraph_tree")
        self.assertEqual(tree.details["components"], 2)
        reachability = next(check for check in report.checks if check.code == "initial_reachability")
        self.assertEqual(reachability.details["unreachable"], [2])

    def test_parallel_double_edges_count_as_two_graph_edges(self) -> None:
        topology = Topology(
            n=4,
            vertices=(0, 1, 2),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE, multiplicity=2),
                InternalEdge(1, 2, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 0, 1: 1, 2: 2},
        )
        report = analyze_topology(topology)
        self.assertEqual(report.metrics.d, 3)
        self.assertEqual(report.metrics.edge_count, 3)
        self.assertIn("double_subgraph_tree", report.failed_codes())
        self.assertEqual(report.metrics.double_cycle_rank, 1)

    def test_changed_final_slots_fail_count_and_outdegree_rules(self) -> None:
        topology = Topology(
            n=4,
            vertices=(0, 1, 2),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 1, 1: 1, 2: 1},
        )
        report = analyze_topology(topology)
        self.assertIn("complete_expansion_counts", report.failed_codes())
        self.assertIn("outdegree_two", report.failed_codes())
        self.assertEqual(report.metrics.final_slot_total, 3)

    def test_field_errors_are_retained_without_crashing_structural_checks(self) -> None:
        topology = Topology(
            n=4,
            vertices=(0, 1),
            edges=(InternalEdge(0, 9, EdgeType.SINGLE), InternalEdge(0, 0, EdgeType.SINGLE)),
            initial_vertex=0,
            final_slots={0: 2, 1: 2},
        )
        report = analyze_topology(topology)
        self.assertFalse(report.valid)
        self.assertFalse(report.field_report.valid)
        self.assertIn("unknown_edge_vertex", report.field_report.error_codes())
        self.assertIn("directed_acyclic", report.failed_codes())

    def test_general_n_is_rejected_by_current_scope(self) -> None:
        topology = Topology(
            n=5,
            vertices=(0, 1, 2, 3),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(2, 3, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 1, 1: 1, 2: 1, 3: 2},
        )
        report = analyze_topology(topology)
        self.assertFalse(report.valid)
        self.assertIn("supported_scope", report.failed_codes())


if __name__ == "__main__":
    unittest.main()
