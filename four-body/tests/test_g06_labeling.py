import json
import unittest

from phase_space_graph import (
    CountMode,
    EdgeType,
    FinalParticleLabel,
    InternalEdge,
    Labeling,
    LabelingLimits,
    RunStatus,
    SingleLineLabel,
    Topology,
    enumerate_labelings,
    evaluate_labeling,
)


def f01() -> Topology:
    return Topology(
        4,
        (0, 1, 2),
        (InternalEdge(0, 1, EdgeType.DOUBLE), InternalEdge(1, 2, EdgeType.DOUBLE)),
        0,
        {0: 1, 1: 1, 2: 2},
    )


def f02() -> Topology:
    return Topology(
        4,
        (0, 1, 2),
        (InternalEdge(0, 1, EdgeType.DOUBLE), InternalEdge(0, 2, EdgeType.DOUBLE)),
        0,
        {0: 0, 1: 2, 2: 2},
    )


def f03() -> Topology:
    return Topology(
        4,
        (0, 1, 2, 3),
        (
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(0, 2, EdgeType.DOUBLE),
            InternalEdge(2, 3, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.SINGLE),
        ),
        0,
        {0: 0, 1: 1, 2: 1, 3: 2},
    )


def f04() -> Topology:
    return Topology(
        3,
        (0, 1, 2),
        (
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.DOUBLE),
            InternalEdge(0, 1, EdgeType.SINGLE),
        ),
        0,
        {0: 0, 1: 1, 2: 2},
    )


class MomentumRecoveryTests(unittest.TestCase):
    def test_f01_chain_recovers_nested_subsets(self) -> None:
        labeling = Labeling(
            (
                FinalParticleLabel(0, 0, 4),
                FinalParticleLabel(1, 0, 3),
                FinalParticleLabel(2, 0, 1),
                FinalParticleLabel(2, 1, 2),
            )
        )
        result = evaluate_labeling(f01(), labeling)
        self.assertTrue(result.valid)
        self.assertEqual(result.variables, ((1, 2, 3), (1, 2)))
        self.assertTrue(result.momentum.verified)
        self.assertTrue(all(residual == (0, 0, 0, 0) for _, residual in result.momentum.vertex_residuals))
        self.assertEqual(result.independence.rank, 2)

    def test_f02_branching_recovers_partition(self) -> None:
        labeling = Labeling(
            (
                FinalParticleLabel(1, 0, 1),
                FinalParticleLabel(1, 1, 2),
                FinalParticleLabel(2, 0, 3),
                FinalParticleLabel(2, 1, 4),
            )
        )
        result = evaluate_labeling(f02(), labeling)
        self.assertTrue(result.valid)
        self.assertEqual(result.variables, ((1, 2), (3, 4)))

    def test_f03_circle_uses_internal_single_in_conservation(self) -> None:
        labeling = Labeling(
            (
                FinalParticleLabel(1, 0, 1),
                FinalParticleLabel(2, 0, 4),
                FinalParticleLabel(3, 0, 2),
                FinalParticleLabel(3, 1, 3),
            ),
            (SingleLineLabel(2, 0, 2),),
        )
        result = evaluate_labeling(f03(), labeling)
        self.assertTrue(result.valid)
        self.assertEqual(result.variables, ((1, 2), (3, 4), (2, 3)))
        self.assertEqual(result.total_measure_cancellations, 1)
        self.assertEqual(result.independence.rank, 3)

    def test_f04_keeps_same_endpoint_single_and_double(self) -> None:
        labeling = Labeling(
            (
                FinalParticleLabel(1, 0, 1),
                FinalParticleLabel(2, 0, 2),
                FinalParticleLabel(2, 1, 3),
            ),
            (SingleLineLabel(1, 0, 3),),
        )
        result = evaluate_labeling(f04(), labeling)
        self.assertTrue(result.valid)
        self.assertEqual(result.variables, ((1, 2), (2, 3)))
        self.assertEqual(result.total_measure_cancellations, 1)

    def test_wrong_single_label_reports_composite_failures(self) -> None:
        labeling = Labeling(
            (
                FinalParticleLabel(1, 0, 1),
                FinalParticleLabel(2, 0, 4),
                FinalParticleLabel(3, 0, 2),
                FinalParticleLabel(3, 1, 3),
            ),
            (SingleLineLabel(2, 0, 1),),
        )
        result = evaluate_labeling(f03(), labeling)
        self.assertFalse(result.valid)
        self.assertIn("composite_coefficient_gt_one", result.failure_codes)
        self.assertIn("negative_composite_coefficient", result.failure_codes)


class LabelingSearchTests(unittest.TestCase):
    def test_f01_search_is_complete_and_deduplicated(self) -> None:
        report = enumerate_labelings(f01())
        self.assertEqual(report.status, RunStatus.COMPLETE)
        self.assertEqual(report.raw_labeling_count, 24)
        self.assertTrue(report.effective)
        self.assertEqual(report.unique_count, len(report.valid_results))
        self.assertTrue(all(result.total_measure_cancellations == 0 for result in report.valid_results))

    def test_common_particle_relabeling_is_a_separate_mode(self) -> None:
        fixed = enumerate_labelings(f01(), CountMode.FIXED_LABELS)
        relabeled = enumerate_labelings(f01(), CountMode.PARTICLE_RELABELING)
        self.assertEqual(fixed.status, relabeled.status)
        self.assertGreaterEqual(fixed.unique_count, relabeled.unique_count)
        self.assertEqual(relabeled.count_mode, CountMode.PARTICLE_RELABELING)

    def test_limit_and_json_report(self) -> None:
        report = enumerate_labelings(f03(), limits=LabelingLimits(max_labelings=1))
        self.assertEqual(report.status, RunStatus.INCOMPLETE)
        self.assertEqual(report.raw_labeling_count, 1)
        self.assertEqual(report.stop_reason, "limit_max_labelings")
        restored = json.loads(report.to_json())
        self.assertEqual(restored["counts"]["raw_labelings"], 1)
        self.assertIn("failure_samples", restored)

    def test_invalid_labeling_is_rejected_before_momentum(self) -> None:
        invalid = Labeling((FinalParticleLabel(0, 0, 1),))
        result = evaluate_labeling(f01(), invalid)
        self.assertFalse(result.valid)
        self.assertIn("final_particle_bijection_mismatch", result.failure_codes)

    def test_invalid_topology_is_rejected_before_momentum(self) -> None:
        invalid_topology = Topology(
            4,
            (0, 1, 2),
            (InternalEdge(0, 1, EdgeType.DOUBLE),),
            0,
            {0: 1, 1: 1, 2: 1},
        )
        labeling = Labeling(
            (
                FinalParticleLabel(0, 0, 1),
                FinalParticleLabel(1, 0, 2),
                FinalParticleLabel(2, 0, 3),
                FinalParticleLabel(2, 1, 4),
            )
        )
        with self.assertRaisesRegex(ValueError, "Topology fields must be valid"):
            evaluate_labeling(invalid_topology, labeling)


if __name__ == "__main__":
    unittest.main()
