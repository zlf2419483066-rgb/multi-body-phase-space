import json
import unittest

from phase_space_graph import (
    EnumerationLimits,
    RunStatus,
    analyze_topology,
    enumerate_topologies,
    prufer_labeled_trees,
    tree_skeletons,
)


class TreeGeneratorTests(unittest.TestCase):
    def test_prufer_counts_and_unlabeled_shape_counts(self) -> None:
        # Cayley's labeled counts and the known small unlabeled tree counts are
        # an independent check on the skeleton service used by G05.
        expected_labeled = {2: 1, 3: 3, 4: 16, 5: 125, 6: 1296}
        expected_shapes = {2: 1, 3: 1, 4: 2, 5: 3, 6: 6}
        for vertex_count in expected_labeled:
            self.assertEqual(sum(1 for _ in prufer_labeled_trees(vertex_count)), expected_labeled[vertex_count])
            self.assertEqual(len(tree_skeletons(vertex_count)), expected_shapes[vertex_count])

    def test_skeleton_edges_are_trees_on_contiguous_vertices(self) -> None:
        for skeleton in tree_skeletons(6):
            self.assertEqual(len(skeleton), 5)
            vertices = {vertex for edge in skeleton for vertex in edge}
            self.assertEqual(vertices, set(range(6)))


class EnumerationTests(unittest.TestCase):
    def test_four_body_tree_run_is_complete_and_deduplicated(self) -> None:
        report = enumerate_topologies(4, 0)
        self.assertEqual(report.status, RunStatus.COMPLETE)
        self.assertEqual(report.expected_vertices, 3)
        self.assertEqual(report.unique_count, 2)
        self.assertEqual(len(report.canonical_keys), report.unique_count)
        self.assertEqual(len(set(report.canonical_keys)), report.unique_count)
        self.assertTrue(all(analyze_topology(topology).valid for topology in report.topologies))

    def test_reference_route_and_pruned_route_have_same_keys(self) -> None:
        optimized = enumerate_topologies(3, 1)
        reference = enumerate_topologies(3, 1, use_pruning=False)
        self.assertEqual(optimized.status, RunStatus.COMPLETE)
        self.assertEqual(reference.status, RunStatus.COMPLETE)
        self.assertEqual(optimized.canonical_keys, reference.canonical_keys)
        self.assertLess(optimized.raw_candidate_count, reference.raw_candidate_count)
        self.assertGreater(optimized.ledger.count("directed_cycle"), 0)

    def test_candidate_limit_is_reported_as_incomplete(self) -> None:
        report = enumerate_topologies(4, 3, EnumerationLimits(max_candidates=10))
        self.assertEqual(report.status, RunStatus.INCOMPLETE)
        self.assertEqual(report.raw_candidate_count, 10)
        self.assertEqual(report.ledger.count("limit_max_candidates"), 1)
        self.assertFalse(report.complete)

    def test_time_limit_is_reported_as_incomplete(self) -> None:
        report = enumerate_topologies(4, 3, EnumerationLimits(max_seconds=0))
        self.assertEqual(report.status, RunStatus.INCOMPLETE)
        self.assertEqual(report.raw_candidate_count, 0)
        self.assertEqual(report.ledger.count("limit_max_seconds"), 1)

    def test_skeleton_limit_is_reported_as_incomplete(self) -> None:
        report = enumerate_topologies(4, 1, EnumerationLimits(max_skeletons=1))
        self.assertEqual(report.status, RunStatus.INCOMPLETE)
        self.assertEqual(report.skeleton_count, 1)
        self.assertEqual(report.ledger.count("limit_max_skeletons"), 1)

    def test_topology_limit_is_a_hard_upper_bound(self) -> None:
        report = enumerate_topologies(4, 0, EnumerationLimits(max_topologies=0))
        self.assertEqual(report.status, RunStatus.INCOMPLETE)
        self.assertEqual(report.unique_count, 0)
        self.assertEqual(report.ledger.count("limit_max_topologies"), 1)

    def test_invalid_scope_returns_failed_report(self) -> None:
        unsupported_n = enumerate_topologies(5, 0)
        unsupported_l = enumerate_topologies(4, 4)
        self.assertEqual(unsupported_n.status, RunStatus.FAILED)
        self.assertEqual(unsupported_l.status, RunStatus.FAILED)
        self.assertIn("unsupported_n", {issue.code for issue in unsupported_n.issues})
        self.assertIn("unsupported_loop_count", {issue.code for issue in unsupported_l.issues})

    def test_report_json_contains_auditable_counts_and_topologies(self) -> None:
        report = enumerate_topologies(3, 0)
        restored = json.loads(report.to_json())
        self.assertEqual(restored["status"], RunStatus.COMPLETE.value)
        self.assertEqual(restored["counts"]["unique"], report.unique_count)
        self.assertEqual(len(restored["topologies"]), report.unique_count)
        self.assertIn("pruning_ledger", restored)


if __name__ == "__main__":
    unittest.main()
