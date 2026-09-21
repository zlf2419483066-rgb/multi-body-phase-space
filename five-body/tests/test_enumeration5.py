from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase_space_graph5 import (  # noqa: E402
    EnumerationLimits,
    RunStatus,
    analyze_topology,
    enumerate_topologies,
    prufer_labeled_trees,
    tree_skeletons,
)


class FiveBodyEnumerationTests(unittest.TestCase):
    def test_prufer_and_shape_counts_for_four_internal_vertices(self) -> None:
        self.assertEqual(sum(1 for _ in prufer_labeled_trees(4)), 16)
        self.assertEqual(len(tree_skeletons(4)), 2)

    def test_zero_loop_scope_has_three_unlabeled_structure_candidates(self) -> None:
        report = enumerate_topologies(5, 0)
        self.assertEqual(report.status, RunStatus.COMPLETE, report.to_dict())
        self.assertEqual(report.expected_vertices, 4)
        # The hand-derived 105 count belongs to the later fixed-label
        # expansion (60+15+30).  G05 itself emits the three unlabeled root
        # split/topology classes; F5-03 owns the particle-label count.
        self.assertEqual(report.unique_count, 3)
        self.assertEqual(len(set(report.canonical_keys)), 3)
        self.assertTrue(all(analyze_topology(topology).valid for topology in report.topologies))

    def test_unpruned_reference_route_has_same_complete_keys(self) -> None:
        optimized = enumerate_topologies(5, 0)
        reference = enumerate_topologies(5, 0, use_pruning=False)
        self.assertEqual(optimized.status, RunStatus.COMPLETE)
        self.assertEqual(reference.status, RunStatus.COMPLETE)
        self.assertEqual(optimized.canonical_keys, reference.canonical_keys)
        self.assertEqual(optimized.unique_count, reference.unique_count)

    def test_three_loop_public_entry_supports_bounded_stop(self) -> None:
        report = enumerate_topologies(5, 3, EnumerationLimits(max_seconds=0))
        self.assertEqual(report.status, RunStatus.INCOMPLETE, report.to_dict())
        self.assertEqual(report.expected_vertices, 7)
        self.assertEqual(report.stop_reason, "limit_max_seconds")
        self.assertEqual(report.ledger.count("limit_max_seconds"), 1)
        self.assertFalse(report.issues)

    def test_public_scope_rejects_l4_and_l5(self) -> None:
        for loop_count in (4, 5):
            with self.subTest(loop_count=loop_count):
                report = enumerate_topologies(5, loop_count)
                self.assertEqual(report.status, RunStatus.FAILED)
                self.assertIn("unsupported_enumeration_scope", {issue.code for issue in report.issues})

    def test_candidate_and_time_limits_are_incomplete(self) -> None:
        candidate_limited = enumerate_topologies(5, 0, EnumerationLimits(max_candidates=1))
        self.assertEqual(candidate_limited.status, RunStatus.INCOMPLETE)
        self.assertEqual(candidate_limited.raw_candidate_count, 1)
        self.assertEqual(candidate_limited.ledger.count("limit_max_candidates"), 1)
        time_limited = enumerate_topologies(5, 0, EnumerationLimits(max_seconds=0))
        self.assertEqual(time_limited.status, RunStatus.INCOMPLETE)
        self.assertEqual(time_limited.raw_candidate_count, 0)
        self.assertEqual(time_limited.ledger.count("limit_max_seconds"), 1)

    def test_report_serializes_scope_and_pruning_evidence(self) -> None:
        report = enumerate_topologies(5, 0, EnumerationLimits(max_topologies=0))
        data = report.to_dict()
        self.assertEqual(data["status"], RunStatus.INCOMPLETE.value)
        self.assertIn("pruning_ledger", data)
        self.assertEqual(data["limits"]["max_topologies"], 0)


if __name__ == "__main__":
    unittest.main()
