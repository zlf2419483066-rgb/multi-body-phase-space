"""Reference comparisons for the untested symmetry optimization checkpoint.

No expected scientific count is taken from the source slides.
"""

import json
import unittest
from unittest.mock import patch

from phase_space_graph import (
    CountMode, LabelingLimits, enumerate_labelings, enumerate_topologies,
    build_audit_report, enumerate_invariant_combinations, run_audit,
)
from phase_space_graph.canonical import (
    canonicalize_labeling, canonicalize_labeling_reference,
)
from phase_space_graph.enumeration import (
    _canonical_tree_key, prufer_labeled_trees, tree_skeletons,
)
from test_g06_labeling import f01, f02, f03, f04


class SymmetryOptimizationTests(unittest.TestCase):
    def assert_same_search(self, topology, mode, limits=None):
        reference = enumerate_labelings(topology, mode, limits, use_symmetry=False)
        optimized = enumerate_labelings(topology, mode, limits, use_symmetry=True)
        left, right = reference.to_dict(), optimized.to_dict()
        for value in (left, right):
            value.pop('representative_evaluations')
            value.pop('search_strategy')
        # Includes representative labeling, exact certificates, failure samples,
        # raw counters, limits, and stop reason, not just final counts.
        self.assertEqual(left, right)
        singles = sum(edge.multiplicity for edge in topology.edges
                      if edge.edge_type.value == 'single')
        self.assertLessEqual(optimized.representative_evaluations, topology.n ** singles)
        return reference, optimized

    def test_fixture_reports_match_reference_in_both_modes(self):
        for topology in (f01(), f02(), f03(), f04()):
            for mode in (CountMode.FIXED_LABELS, CountMode.PARTICLE_RELABELING):
                with self.subTest(topology=topology, mode=mode):
                    self.assert_same_search(topology, mode)

    def test_raw_budget_and_result_budget_keep_reference_prefix(self):
        for limits in (
            LabelingLimits(max_labelings=0), LabelingLimits(max_labelings=1),
            LabelingLimits(max_labelings=5), LabelingLimits(max_labelings=95),
            LabelingLimits(max_labelings=96), LabelingLimits(max_results=1),
            LabelingLimits(max_results=0),
        ):
            with self.subTest(limits=limits):
                self.assert_same_search(f03(), CountMode.FIXED_LABELS, limits)

    def test_zero_time_and_immediate_cancellation(self):
        for symmetry in (False, True):
            report = enumerate_labelings(f03(), limits=LabelingLimits(max_seconds=0),
                                         use_symmetry=symmetry)
            self.assertFalse(report.complete)
            self.assertEqual(report.raw_labeling_count, 0)
            report = enumerate_labelings(f03(), cancel_requested=lambda: True,
                                         use_symmetry=symmetry)
            self.assertEqual(report.stop_reason, 'cancel_requested')
            self.assertEqual(report.raw_labeling_count, 0)

    def test_momentum_recovery_only_runs_for_orbit_representatives(self):
        from phase_space_graph import labeling as module
        with patch.object(module, '_recover_momenta', wraps=module._recover_momenta) as recover:
            report = enumerate_labelings(f03())
        self.assertEqual(report.raw_labeling_count, 96)
        self.assertEqual(recover.call_count, 4)
        self.assertEqual(report.representative_evaluations, 4)

    def test_marked_keys_and_tie_mappings_keep_exhaustive_definition(self):
        for topology in (f01(), f02(), f03(), f04()):
            report = enumerate_labelings(topology, use_symmetry=False)
            # Includes symmetric tree ends and repeated single/double endpoints.
            for result in report.valid_results:
                for mode in (CountMode.FIXED_LABELS, CountMode.PARTICLE_RELABELING):
                    self.assertEqual(
                        canonicalize_labeling(topology, result.labeling, mode),
                        canonicalize_labeling_reference(topology, result.labeling, mode),
                    )

    def test_tree_signatures_preserve_old_representative_keys(self):
        # Bounded test; full v=6 legacy comparison belongs in the later audit.
        for vertices in range(2, 6):
            reference = {_canonical_tree_key(edges, vertices)
                         for edges in prufer_labeled_trees(vertices)}
            self.assertEqual(set(tree_skeletons(vertices)), reference)
        self.assertEqual(len(tree_skeletons(6)), 6)


class AuditMetadataRegressionTests(unittest.TestCase):
    def test_aggregation_does_not_invent_times_or_unrequested_counts(self):
        topologies = enumerate_topologies(4, 0)
        combinations = enumerate_invariant_combinations(4, sizes=(2,))
        for mode in (CountMode.FIXED_LABELS, CountMode.PARTICLE_RELABELING):
            reports = [enumerate_labelings(t, mode) for t in topologies.topologies]
            audit = build_audit_report(topologies, reports, combinations)
            data = json.loads(audit.to_json())
            self.assertIsNone(data['started_at'])
            self.assertIsNone(data['finished_at'])
            missing = ('particle_relabeling_classes' if mode is CountMode.FIXED_LABELS
                       else 'fixed_label_classes')
            self.assertIsNone(data['counts'][missing])
            self.assertIn('未计算', audit.to_markdown())

    def test_orchestrator_records_start_before_computation(self):
        from phase_space_graph import reporting as module
        events = []
        original = module.enumerate_topologies

        def timestamp():
            events.append('timestamp')
            return f'time-{len(events)}'

        def enumerate_recorded(*args, **kwargs):
            events.append('enumeration')
            return original(*args, **kwargs)

        with patch.object(module, '_timestamp', side_effect=timestamp), \
             patch.object(module, 'enumerate_topologies', side_effect=enumerate_recorded):
            audit = run_audit(4, 0, combination_sizes=(2,))
        self.assertEqual(events, ['timestamp', 'enumeration', 'timestamp'])
        self.assertEqual(audit.started_at, 'time-1')
        self.assertEqual(audit.finished_at, 'time-3')


if __name__ == '__main__':
    unittest.main()
