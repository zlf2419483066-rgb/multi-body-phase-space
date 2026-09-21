from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase_space_graph5 import (  # noqa: E402
    RunStatus,
    analyze_topology,
    canonicalize_topology_reference,
    enumerate_topologies,
)


class FiveBodyLowLoopExpansionTests(unittest.TestCase):
    def test_one_loop_pruned_structure_complete(self) -> None:
        report = enumerate_topologies(5, 1, use_pruning=True)

        self.assertEqual(report.status, RunStatus.COMPLETE, report.to_dict())
        self.assertEqual(report.expected_vertices, 5)
        self.assertEqual(report.skeleton_count, 3)
        self.assertEqual(report.labeled_tree_count, 125)
        self.assertEqual(report.orientation_count, 48)
        self.assertEqual(report.single_multiset_count, 144)
        self.assertEqual(report.raw_candidate_count, 720)
        self.assertEqual(report.structural_valid_count, 144)
        self.assertEqual(report.unique_count, 69)
        self.assertEqual(len(set(report.canonical_keys)), report.unique_count)
        self.assertTrue(all(analyze_topology(topology).valid for topology in report.topologies))
        self.assertTrue(all(analyze_topology(topology).metrics.loop_count == 1 for topology in report.topologies))

    def test_one_loop_unpruned_route_keeps_the_complete_structural_domain(self) -> None:
        report = enumerate_topologies(5, 1, use_pruning=False)

        self.assertEqual(report.status, RunStatus.COMPLETE, report.to_dict())
        self.assertEqual(report.single_multiset_count, 960)
        self.assertEqual(report.raw_candidate_count, 4800)
        self.assertEqual(report.structural_valid_count, 144)
        self.assertEqual(report.unique_count, 69)

    def test_two_loop_pruned_structure_complete(self) -> None:
        report = enumerate_topologies(5, 2, use_pruning=True)

        self.assertEqual(report.status, RunStatus.COMPLETE, report.to_dict())
        self.assertEqual(report.expected_vertices, 6)
        self.assertEqual(report.skeleton_count, 6)
        self.assertEqual(report.labeled_tree_count, 1296)
        self.assertEqual(report.orientation_count, 192)
        self.assertEqual(report.single_multiset_count, 4083)
        self.assertEqual(report.raw_candidate_count, 24498)
        self.assertEqual(report.structural_valid_count, 4083)
        self.assertEqual(report.unique_count, 1731)
        self.assertEqual(len(set(report.canonical_keys)), report.unique_count)
        self.assertTrue(all(analyze_topology(topology).valid for topology in report.topologies))
        self.assertTrue(all(analyze_topology(topology).metrics.loop_count == 2 for topology in report.topologies))

    def test_pruning_and_full_reference_canonicalization_have_same_l1_keys(self) -> None:
        reports = {
            (pruning, reference): enumerate_topologies(
                5,
                1,
                use_pruning=pruning,
                use_reference_canonical=reference,
            )
            for pruning in (True, False)
            for reference in (True, False)
        }
        for report in reports.values():
            self.assertEqual(report.status, RunStatus.COMPLETE, report.to_dict())

        key_sets = {report.canonical_keys for report in reports.values()}
        self.assertEqual(len(key_sets), 1)
        keys = next(iter(key_sets))
        self.assertEqual(len(keys), 69)

        recomputed_reference_keys = tuple(
            sorted(canonicalize_topology_reference(topology).key for topology in reports[(True, False)].topologies)
        )
        self.assertEqual(recomputed_reference_keys, keys)


if __name__ == "__main__":
    unittest.main()
