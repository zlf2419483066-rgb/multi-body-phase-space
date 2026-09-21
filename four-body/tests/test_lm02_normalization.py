import unittest

from phase_space_graph import DoubleLineLabel, FinalParticleLabel, Labeling, SingleLineLabel
from phase_space_graph.physical_labels import normalize_physical_labeling
from im_fixtures import all_fixtures


class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures = {fixture.name: fixture for fixture in all_fixtures()}

    def test_blank_partial_and_complete_b_have_one_physical_record(self) -> None:
        fixture = self.fixtures["B"]
        blank = normalize_physical_labeling(fixture.topology, fixture.labeling)
        partial = normalize_physical_labeling(
            fixture.topology,
            Labeling(
                fixture.labeling.final_particles,
                fixture.labeling.single_lines,
                [DoubleLineLabel(0, 0, (1, 2))],
            ),
        )
        complete = normalize_physical_labeling(fixture.topology, fixture.record.complete_labeling)
        self.assertEqual(blank.stage, "accepted")
        self.assertEqual(partial.stage, "accepted")
        self.assertEqual(complete.stage, "accepted")
        self.assertEqual(blank.record, partial.record)
        self.assertEqual(blank.record, complete.record)
        self.assertEqual(blank.record.momentum.variables, fixture.expected_variables)
        self.assertEqual(blank.record.measure_cancellations, ((1, 0), (2, 1), (3, 0), (4, 0)))

    def test_wrong_explicit_double_line_is_rejected(self) -> None:
        fixture = self.fixtures["B"]
        wrong = Labeling(
            fixture.labeling.final_particles,
            fixture.labeling.single_lines,
            [DoubleLineLabel(1, 0, (1, 3))],
        )
        result = normalize_physical_labeling(fixture.topology, wrong)
        self.assertEqual(result.stage, "momentum_rejected")
        self.assertIn("declared_double_momentum_mismatch", [issue.code for issue in result.issues])
        self.assertIsNotNone(result.legacy_evaluation)
        self.assertIsNone(result.record)

    def test_dependent_record_keeps_duplicate_variables(self) -> None:
        fixture = self.fixtures["D"]
        result = normalize_physical_labeling(fixture.topology, fixture.labeling)
        self.assertEqual(result.stage, "dependent")
        self.assertIsNotNone(result.record)
        self.assertEqual(result.record.momentum.variables, ((2, 3, 4), (2, 4), (2, 4)))
        self.assertEqual(result.record.independence.rank, 2)
        self.assertEqual([issue.code for issue in result.issues], ["dependent_invariants"])

    def test_structure_is_checked_before_momentum(self) -> None:
        fixture = self.fixtures["E"]
        result = normalize_physical_labeling(fixture.topology, fixture.labeling)
        self.assertEqual(result.stage, "structure_rejected")
        self.assertIn("outdegree_two", [issue.code for issue in result.issues])
        self.assertIsNone(result.legacy_evaluation)
        self.assertIsNone(result.record)

    def test_missing_single_and_invalid_particle_are_diagnostics(self) -> None:
        fixture = self.fixtures["B"]
        missing_single = Labeling(fixture.labeling.final_particles, ())
        missing = normalize_physical_labeling(fixture.topology, missing_single)
        self.assertEqual(missing.stage, "momentum_rejected")
        self.assertIn("missing_single_label", [issue.code for issue in missing.issues])

        invalid_particle = Labeling(
            [FinalParticleLabel(1, 0, 5), *fixture.labeling.final_particles[1:]],
            fixture.labeling.single_lines,
        )
        invalid = normalize_physical_labeling(fixture.topology, invalid_particle)
        self.assertEqual(invalid.stage, "field_rejected")
        self.assertIn("particle_id_out_of_range", [issue.code for issue in invalid.issues])

    def test_input_labeling_is_not_mutated(self) -> None:
        fixture = self.fixtures["B"]
        original = fixture.labeling
        result = normalize_physical_labeling(fixture.topology, original)
        self.assertEqual(result.stage, "accepted")
        self.assertEqual(original, fixture.labeling)
        self.assertEqual(original.double_lines, ())


if __name__ == "__main__":
    unittest.main()
