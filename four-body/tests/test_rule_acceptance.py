"""External acceptance and rule regression tests; no source count drives generation."""

from copy import deepcopy
from dataclasses import replace
import unittest

from phase_space_graph import (
    CountMode, DoubleLineLabel, FinalParticleLabel, InternalEdge, Labeling,
    RunStatus, SourceDiagram, SourceReference, Topology, build_audit_report,
    compare_reference_count, enumerate_invariant_combinations, enumerate_labelings,
    enumerate_topologies, evaluate_labeling, trace_source_catalog,
)
from test_g06_labeling import f01, f02, f03


class RuleAcceptanceTests(unittest.TestCase):
    def test_every_physical_failure_has_a_replayable_witness(self):
        for strategy in (False, True):
            topology = f03()
            report = enumerate_labelings(topology, use_symmetry=strategy)
            witnesses = dict(report.failure_witnesses)
            for code, _ in report.failure_counts:
                self.assertIn(code, witnesses)
                witness = witnesses[code]
                replayed = evaluate_labeling(topology, witness.labeling)
                self.assertEqual(witness.to_dict(), replayed.to_dict())
                self.assertIn(code, replayed.failure_codes)
            certificate = witnesses['dependent_invariants'].independence.certificate
            self.assertTrue(certificate.verified)

    def inputs(self):
        topology = enumerate_topologies(4, 0)
        reports = tuple(enumerate_labelings(t) for t in topology.topologies)
        combinations = enumerate_invariant_combinations(4, sizes=(2,))
        return topology, reports, combinations

    def test_expected_count_cannot_change_generation_or_screening(self):
        inputs = self.inputs()
        ordinary = build_audit_report(*inputs)
        mismatched = build_audit_report(*inputs, source_references=(
            SourceReference('external', 'synthetic incorrect expectation', 999, verified=True, n=4, loop_count=0),))
        self.assertEqual(ordinary.topology_report, mismatched.topology_report)
        self.assertEqual(ordinary.labeling_reports, mismatched.labeling_reports)
        self.assertEqual(ordinary.counts, mismatched.counts)
        self.assertTrue(mismatched.complete)
        self.assertEqual(mismatched.source_status, 'count_mismatch')
        self.assertEqual(mismatched.to_dict()['acceptance']['catalog_status'], 'not_checked')

    def test_partial_count_match_cannot_pass_external_acceptance(self):
        result = compare_reference_count(2, 2, 'incomplete')
        self.assertEqual(result['status'], 'inconclusive')

    def test_reference_scope_is_not_guessed_from_text(self):
        audit = build_audit_report(*self.inputs(), source_references=(
            SourceReference('other loop', 'arbitrary description', 999, n=4, loop_count=3),))
        self.assertEqual(audit.source_status, 'out_of_scope')

    def test_missing_required_combination_size_is_incomplete(self):
        topology, reports, _ = self.inputs()
        wrong_size = enumerate_invariant_combinations(4, sizes=(3,))
        audit = build_audit_report(topology, reports, wrong_size)
        self.assertFalse(audit.complete)
        self.assertIn('missing_required_combination_scope', {i.code for i in audit.issues})

    def test_failed_combination_search_is_failed_not_incomplete(self):
        topology, reports, combinations = self.inputs()
        audit = build_audit_report(topology, reports, replace(combinations, status=RunStatus.FAILED))
        self.assertEqual(audit.status, RunStatus.FAILED)

    def test_full_combination_scope_records_uncovered_members(self):
        audit = build_audit_report(*self.inputs())
        comparison = audit.combination_comparison
        self.assertEqual(comparison['status'], 'checked')
        self.assertEqual(comparison['graph_only'], [])
        self.assertTrue(comparison['combination_only'])
        for record in comparison['combination_only']:
            self.assertTrue(record['independence']['independent'])
            self.assertIn('canonical', record)

    def test_graph_key_outside_supplied_complete_universe_fails(self):
        topology, reports, combinations = self.inputs()
        broken = replace(combinations, unique_records=())
        audit = build_audit_report(topology, reports, broken)
        self.assertEqual(audit.status, RunStatus.FAILED)
        self.assertTrue(audit.combination_comparison['graph_only'])

    def test_explicit_double_subset_must_match_conservation(self):
        topology = f01()
        finals = (FinalParticleLabel(0, 0, 4), FinalParticleLabel(1, 0, 3),
                  FinalParticleLabel(2, 0, 1), FinalParticleLabel(2, 1, 2))
        correct = Labeling(finals, double_lines=(DoubleLineLabel(0, 0, (1, 2, 3)),))
        wrong = Labeling(finals, double_lines=(DoubleLineLabel(0, 0, (1, 2)),))
        self.assertTrue(evaluate_labeling(topology, correct).valid)
        self.assertIn('declared_double_momentum_mismatch', evaluate_labeling(topology, wrong).failure_codes)

    def test_source_graphs_match_after_arbitrary_vertex_renumbering(self):
        audit = build_audit_report(*self.inputs()).to_dict()
        original = f01()
        mapping = {0: 20, 1: -2, 2: 7}
        renamed = Topology(4, tuple(mapping[v] for v in original.vertices),
                           tuple(InternalEdge(mapping[e.source], mapping[e.target], e.edge_type, e.multiplicity)
                                 for e in original.edges), mapping[original.initial_vertex],
                           {mapping[v]: count for v, count in original.final_slots})
        entries = (SourceDiagram('hand derivation', 1, 'chain', renamed, True),
                   SourceDiagram('hand derivation', 1, 'fork', f02(), True))
        trace = trace_source_catalog(audit, entries, expected_count=2)
        self.assertEqual(trace['status'], 'catalog_match')
        self.assertTrue(all(row['stage'] == 'accepted' for row in trace['rows']))
        self.assertIn('witness_topology', trace['rows'][0])

    def test_equal_counts_and_duplicate_source_shapes_do_not_pass(self):
        audit = build_audit_report(*self.inputs()).to_dict()
        entries = (SourceDiagram('fixture', 1, 'a', f01(), True),
                   SourceDiagram('fixture', 1, 'b', f01(), True))
        trace = trace_source_catalog(audit, entries, expected_count=2)
        self.assertEqual(trace['count_comparison']['status'], 'count_match')
        self.assertEqual(trace['status'], 'catalog_mismatch')
        self.assertTrue(trace['same_topology_source_groups'])
        self.assertTrue(trace['unmatched_effective_keys'])

    def test_untranscribed_and_unverified_entries_cannot_pass(self):
        audit = build_audit_report(*self.inputs()).to_dict()
        entries = (SourceDiagram('fixture', 1, 'a', f01()), SourceDiagram('fixture', 1, 'b'))
        self.assertEqual(trace_source_catalog(audit, entries, expected_count=2)['status'], 'inconclusive')

    def test_missing_structure_has_different_meaning_for_partial_search(self):
        audit = build_audit_report(*self.inputs()).to_dict()
        # Deliberately remove all generated records to test the diagnostic boundary.
        broken = deepcopy(audit)
        broken['topology_report']['topologies'] = []
        entry = SourceDiagram('fixture', 1, 'chain', f01(), True)
        complete = trace_source_catalog(broken, (entry,), expected_count=2)
        self.assertEqual(complete['rows'][0]['stage'], 'missing_from_complete_search')
        broken['topology_report']['status'] = 'incomplete'
        broken['status'] = 'incomplete'
        partial = trace_source_catalog(broken, (entry,), expected_count=2)
        self.assertEqual(partial['rows'][0]['stage'], 'unknown_in_partial_search')


if __name__ == '__main__':
    unittest.main()
