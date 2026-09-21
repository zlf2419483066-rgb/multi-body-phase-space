from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase_space_graph5.domain import (
    CountMode,
    DoubleLineLabel,
    EdgeType,
    InternalEdge,
    Labeling,
    Topology,
)
from phase_space_graph5.enumeration import enumerate_topologies
from phase_space_graph5.fixtures import fixture_a, fixture_b
from phase_space_graph5.label_identity import canonicalize_candidate_labeling, relabel_candidate_record
from phase_space_graph5.labeling import (
    LabelingLimits,
    MomentumCertificate,
    enumerate_labelings,
    evaluate_labeling,
)
from phase_space_graph5.physical_labels import CandidateMomentumRecord, normalize_candidate_labeling


class FiveBodyMomentumCandidateTests(unittest.TestCase):
    @staticmethod
    def _identity_digests(records, mode: CountMode) -> set[str]:
        return {canonicalize_candidate_labeling(record, mode).digest for record in records}

    def test_zero_loop_shapes_match_hand_count_in_both_identity_modes(self) -> None:
        topologies = enumerate_topologies().topologies
        self.assertEqual(len(topologies), 3)
        expected_fixed = (30, 15, 60)
        for topology, fixed_expected in zip(topologies, expected_fixed):
            report = enumerate_labelings(topology, CountMode.FIXED_LABELS, limits=LabelingLimits(max_seconds=540))
            self.assertTrue(report.complete)
            self.assertEqual(report.raw_assignment_count, 120)
            self.assertEqual(report.momentum_pass_count, 120)
            self.assertEqual(len(report.candidate_records), report.momentum_pass_count)
            self.assertEqual(
                len(self._identity_digests(report.candidate_records, CountMode.FIXED_LABELS)),
                fixed_expected,
            )
            self.assertEqual(
                len(self._identity_digests(report.candidate_records, CountMode.PARTICLE_RELABELING)),
                1,
            )

    def test_one_loop_fixture_matches_raw_pass_and_identity_counts(self) -> None:
        topology, _ = fixture_b(3)
        report = enumerate_labelings(topology, CountMode.FIXED_LABELS, limits=LabelingLimits(max_seconds=540))
        self.assertTrue(report.complete)
        self.assertEqual(report.raw_assignment_count, 600)
        self.assertEqual(report.momentum_pass_count, 480)
        self.assertEqual(len(report.candidate_records), report.momentum_pass_count)
        self.assertEqual(
            len(self._identity_digests(report.candidate_records, CountMode.FIXED_LABELS)),
            240,
        )
        self.assertEqual(
            len(self._identity_digests(report.candidate_records, CountMode.PARTICLE_RELABELING)),
            3,
        )

    def test_enumerator_rejects_structure_before_search_counts(self) -> None:
        topology, _ = fixture_a()
        invalid = Topology(
            n=5,
            vertices=topology.vertices,
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE, multiplicity=2),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(2, 3, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 1, 1: 1, 2: 1, 3: 2},
        )
        report = enumerate_labelings(invalid)
        self.assertFalse(report.complete)
        self.assertEqual(report.raw_assignment_count, 0)
        self.assertEqual(report.momentum_pass_count, 0)
        self.assertEqual(report.candidate_records, ())
        self.assertEqual(report.failure_count("double_subgraph_tree"), 1)

    def test_zero_loop_candidate_keeps_tree_cut_variables_and_deferred_state(self) -> None:
        topology, labeling = fixture_a()
        checked = normalize_candidate_labeling(topology, labeling)
        self.assertEqual(checked.stage, "momentum_candidate")
        record = checked.record
        assert record is not None
        self.assertEqual(record.variables, ((2, 3, 4, 5), (3, 4, 5), (4, 5)))
        self.assertEqual(record.independence_status, "not_evaluated")
        self.assertIsNone(record.independence)
        self.assertEqual(record.effectiveness_status, "not_evaluated")
        self.assertIsNone(record.effectiveness)

    def test_empty_partial_and_full_correct_double_declarations_normalize_identically(self) -> None:
        topology, labeling = fixture_a()
        baseline = normalize_candidate_labeling(topology, labeling)
        assert baseline.record is not None
        complete = baseline.record.complete_labeling
        partial = Labeling(
            final_particles=labeling.final_particles,
            double_lines=(complete.double_lines[0],),
        )
        for declared in (labeling, partial, complete):
            checked = normalize_candidate_labeling(topology, declared)
            self.assertEqual(checked.stage, "momentum_candidate")
            assert checked.record is not None
            self.assertEqual(checked.record.to_dict(), baseline.record.to_dict())

        wrong = Labeling(
            final_particles=labeling.final_particles,
            double_lines=(DoubleLineLabel(0, 0, (2, 3, 4)),),
        )
        rejected = normalize_candidate_labeling(topology, wrong)
        self.assertEqual(rejected.stage, "momentum_rejected")
        self.assertIn(
            "declared_double_momentum_mismatch",
            {issue.code for issue in rejected.issues},
        )

    def test_one_loop_subcases_preserve_repeated_variable_and_reject_negative(self) -> None:
        passing = normalize_candidate_labeling(*fixture_b(3))
        self.assertEqual(passing.stage, "momentum_candidate")
        assert passing.record is not None
        self.assertEqual(
            passing.record.variables,
            ((1, 2, 4, 5), (2, 4, 5), (3, 4, 5), (4, 5)),
        )

        repeated = normalize_candidate_labeling(*fixture_b(2))
        self.assertEqual(repeated.stage, "momentum_candidate")
        assert repeated.record is not None
        self.assertGreaterEqual(repeated.record.variables.count((3, 4, 5)), 2)

        rejected = normalize_candidate_labeling(*fixture_b(1))
        self.assertEqual(rejected.stage, "momentum_rejected")
        self.assertIn(
            "negative_composite_coefficient",
            {issue.code for issue in rejected.issues},
        )

    def test_public_evaluation_checks_structure_before_momentum(self) -> None:
        topology, labeling = fixture_a()
        invalid_topology = Topology(
            n=5,
            vertices=topology.vertices,
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE, multiplicity=2),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(2, 3, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 0, 1: 1, 2: 1, 3: 3},
        )
        result = evaluate_labeling(invalid_topology, labeling)
        self.assertFalse(result.valid)
        self.assertIsNone(result.momentum)
        self.assertIn("double_subgraph_tree", result.failure_codes)

    def test_certificate_readback_rejects_coercion_and_variable_tampering(self) -> None:
        topology, labeling = fixture_a()
        checked = normalize_candidate_labeling(topology, labeling)
        assert checked.record is not None
        payload = checked.record.momentum.to_dict()

        for value in (1, "yes"):
            tampered = deepcopy(payload)
            tampered["verified"] = value
            with self.assertRaises(TypeError):
                MomentumCertificate.from_dict(tampered)

        tampered = deepcopy(payload)
        tampered["double_momenta"][0]["coefficients"][0] = 1.2
        with self.assertRaises(TypeError):
            MomentumCertificate.from_dict(tampered)

        tampered = deepcopy(payload)
        tampered["variables"][0] = [1, 2]
        with self.assertRaises(ValueError):
            MomentumCertificate.from_dict(tampered)

        tampered = deepcopy(payload)
        del tampered["double_momenta"][0]["occurrence"]
        with self.assertRaises(ValueError):
            MomentumCertificate.from_dict(tampered)

    def test_record_certificate_tampering_is_rejected_by_identity_entry_points(self) -> None:
        topology, labeling = fixture_a()
        checked = normalize_candidate_labeling(topology, labeling)
        assert checked.record is not None
        payload = deepcopy(checked.record.to_dict())
        payload["momentum"]["double_momenta"][0]["coefficients"][-1] = 0
        payload["momentum"]["variables"][0] = [2, 3, 4]
        damaged = CandidateMomentumRecord.from_dict(payload)
        with self.assertRaises(ValueError):
            canonicalize_candidate_labeling(damaged)
        with self.assertRaises(ValueError):
            relabel_candidate_record(damaged, {0: 0, 1: 1, 2: 2, 3: 3})


if __name__ == "__main__":
    unittest.main()
