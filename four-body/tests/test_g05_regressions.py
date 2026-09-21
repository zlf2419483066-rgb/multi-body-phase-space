import unittest
from unittest.mock import patch
import random

from phase_space_graph import EnumerationLimits, enumerate_topologies, Topology, InternalEdge
from phase_space_graph.canonical import canonicalize_topology, canonicalize_topology_reference


class RegressionTests(unittest.TestCase):
    def test_selected_skeleton_is_fully_processed(self):
        r = enumerate_topologies(4, 1, EnumerationLimits(max_skeletons=1))
        self.assertEqual(r.orientation_count, 8)
        self.assertEqual(r.completed_skeleton_count, 1)
        self.assertEqual(r.skeleton_count, 1)
        self.assertEqual(r.stop_reason, 'limit_max_skeletons')
        self.assertGreater(r.raw_candidate_count, 4)

    def test_exact_skeleton_budget_can_complete(self):
        r = enumerate_topologies(4, 1, EnumerationLimits(max_skeletons=2))
        self.assertTrue(r.complete)
        self.assertEqual(r.completed_skeleton_count, 2)

    def test_zero_time_skips_tree_generation(self):
        with patch('phase_space_graph.enumeration.prufer_labeled_trees', side_effect=AssertionError):
            r = enumerate_topologies(4, 3, EnumerationLimits(max_seconds=0))
        self.assertEqual(r.stop_reason, 'limit_max_seconds')
        self.assertEqual(r.skeleton_count, 0)

    def test_setup_can_be_cancelled(self):
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            return checks == 20
        r = enumerate_topologies(4, 3, cancel_requested=cancel)
        self.assertEqual(r.stop_reason, 'cancel_requested')
        self.assertEqual(r.skeleton_count, 0)

    def test_keyboard_interrupt_retains_results(self):
        from phase_space_graph.enumeration import _canonical_representative
        calls = 0
        def interrupt(topology):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return _canonical_representative(topology)
        with patch('phase_space_graph.enumeration._canonical_representative', side_effect=interrupt):
            r = enumerate_topologies(4, 0)
        self.assertEqual(r.stop_reason, 'keyboard_interrupt')
        self.assertEqual(r.unique_count, 1)
        self.assertEqual(r.ledger.count('canonical_unfinished'), 1)

    def test_nonfinite_time_rejected(self):
        for value in (float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError):
                EnumerationLimits(max_seconds=value)

    def test_candidate_budget_exact_boundary(self):
        full = enumerate_topologies(4, 0)
        exact = enumerate_topologies(4, 0, EnumerationLimits(max_candidates=full.raw_candidate_count))
        partial = enumerate_topologies(4, 0, EnumerationLimits(max_candidates=full.raw_candidate_count-1))
        self.assertTrue(exact.complete)
        self.assertEqual(exact.canonical_keys, full.canonical_keys)
        self.assertFalse(partial.complete)

    def test_general_field_valid_topologies_match_reference(self):
        rng = random.Random(402)
        for size in range(2, 7):
            for _ in range(6):
                vertices = tuple(rng.sample(range(-30, 50), size))
                slots = dict.fromkeys(vertices, 0)
                for _ in range(4):
                    slots[rng.choice(vertices)] += 1
                edges = [InternalEdge(rng.choice(vertices), rng.choice(vertices), rng.choice(('single', 'double')), rng.randrange(3)) for _ in range(8)]
                t = Topology(4, vertices, edges, rng.choice(vertices), slots)
                a, b = canonicalize_topology(t), canonicalize_topology_reference(t)
                self.assertEqual(a, b)

    def test_low_runs_match_unpruned_reference(self):
        for n, l in ((3, 0), (3, 1), (4, 0), (4, 1)):
            a = enumerate_topologies(n, l)
            b = enumerate_topologies(n, l, use_pruning=False)
            self.assertEqual(a.canonical_keys, b.canonical_keys)


if __name__ == '__main__':
    unittest.main()
