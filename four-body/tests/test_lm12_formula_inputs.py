import json
import unittest

from im_fixtures import _fixture_c, fixture_b, fixture_d, fixture_e
from phase_space_graph import (
    build_formula_input,
    normalize_physical_labeling,
    relabel_physical_record,
    render_formula_input_markdown,
)


class LM12FormulaInputTests(unittest.TestCase):
    def test_b_lists_the_four_local_factors_with_total_incoming_at_vertex_two(self):
        fixture = fixture_b()
        card = build_formula_input(
            normalize_physical_labeling(fixture.topology, fixture.labeling),
            measure_convention="jing_psf",
        )
        self.assertEqual(card.status, "accepted")
        self.assertEqual(card.availability, "ready")
        self.assertEqual(len(card.factors), 4)
        self.assertEqual(
            [factor.expression for factor in card.factors],
            [
                "dΦ₂(M; m_12, m_34)",
                "dΦ₂(m_12; m_1, m_2)",
                "dΦ₂(m_234; m_4, m_23)",
                "dΦ₂(m_23; m_2, m_3)",
            ],
        )
        vertex_two = card.factors[2]
        self.assertEqual(vertex_two.incoming_coefficients, (0, 1, 1, 1))
        self.assertEqual(vertex_two.parent_mass_squared, "s_234")
        self.assertEqual(vertex_two.outgoing_mass_squared, ("m_4^2", "s_23"))

    def test_b_keeps_three_ds_and_one_particle_two_cancellation(self):
        fixture = fixture_b()
        card = build_formula_input(
            normalize_physical_labeling(fixture.topology, fixture.labeling),
            measure_convention="jing_psf",
        )
        self.assertEqual([item.variable_label for item in card.double_measures], ["s12", "s34", "s23"])
        self.assertEqual([item.differential for item in card.double_measures], [
            "(2π)^3 ds_{12}", "(2π)^3 ds_{34}", "(2π)^3 ds_{23}"
        ])
        counts = {item.particle_id: item.cancellation_count for item in card.particle_measures}
        self.assertEqual(counts[2], 1)
        self.assertEqual(sum(counts.values()), 1)
        self.assertTrue(card.factors_ready)
        self.assertFalse(card.formula_verified)
        self.assertTrue({"reference_frames", "angles", "measure_cancellation", "constraints_eliminated", "jacobian", "branches", "domain"} <= set(card.missing_steps))

    def test_c_alpha_and_beta_keep_their_own_double_variables(self):
        alpha = _fixture_c(3)
        beta = _fixture_c(2)
        alpha_card = build_formula_input(
            normalize_physical_labeling(alpha.topology, alpha.labeling),
            measure_convention="standard_2pi4",
        )
        beta_card = build_formula_input(
            normalize_physical_labeling(beta.topology, beta.labeling),
            measure_convention="standard_2pi4",
        )
        self.assertEqual(alpha_card.variable_labels, ("s124", "s123", "s12"))
        self.assertEqual(beta_card.variable_labels, ("s134", "s123", "s12"))
        self.assertEqual(alpha_card.double_measures[0].differential, "ds_{124}/(2π)")
        self.assertEqual(beta_card.double_measures[0].differential, "ds_{134}/(2π)")

    def test_dependent_d_is_unavailable_and_retains_relation(self):
        fixture = fixture_d()
        card = build_formula_input(
            normalize_physical_labeling(fixture.topology, fixture.labeling),
            measure_convention="jing_psf",
        )
        self.assertEqual(card.status, "dependent")
        self.assertEqual(card.availability, "unavailable")
        self.assertEqual(card.factors, ())
        self.assertIn("s24 - s24 = 0", card.unavailable_reason)
        self.assertFalse(any(card.formula_status.values()))
        self.assertIn("s24 - s24 = 0", card.constraints)

    def test_unknown_convention_and_failure_do_not_fallback(self):
        fixture = fixture_b()
        with self.assertRaises(ValueError):
            build_formula_input(
                normalize_physical_labeling(fixture.topology, fixture.labeling),
                measure_convention="unknown",
            )
        rejected = build_formula_input(
            normalize_physical_labeling(fixture_e().topology, fixture_e().labeling),
            measure_convention="jing_psf",
        )
        self.assertEqual(rejected.availability, "unavailable")
        self.assertEqual(rejected.factors, ())
        self.assertFalse(rejected.formula_verified)

    def test_particle_relabeling_updates_momenta_mass_symbols_and_measure_groups(self):
        fixture = _fixture_c(3)
        moved = relabel_physical_record(
            fixture.record,
            {0: 0, 1: 1, 2: 2, 3: 3},
            {1: 2, 2: 1, 3: 4, 4: 3},
        ).record
        card = build_formula_input(moved, measure_convention="jing_psf")
        self.assertEqual(card.variable_labels, ("s123", "s124", "s12"))
        counts = {item.particle_id: item.cancellation_count for item in card.particle_measures}
        self.assertEqual(counts[4], 1)
        self.assertEqual(counts[3], 0)
        self.assertIn("p_4", card.particle_measures[3].differential)
        self.assertIn("m_4", card.factors[0].expression)
        self.assertIn("p_4", card.factors[0].outgoing[1].momentum_expression)

    def test_normalization_input_and_machine_view_are_round_trippable(self):
        fixture = fixture_b()
        card = build_formula_input(
            normalize_physical_labeling(fixture.topology, fixture.labeling),
            measure_convention="jing_psf",
            provenance={"source": "B <source>"},
        )
        payload = json.loads(card.to_json())
        self.assertEqual(payload["artifact_type"], "formula_input_card")
        self.assertEqual(payload["status"]["measure_convention"], "jing_psf")
        self.assertEqual(payload["factors"][2]["incoming_coefficients"], [0, 1, 1, 1])
        markdown = render_formula_input_markdown(card)
        self.assertIn("公式已验证：否", markdown)
        self.assertIn("jacobian", markdown)


if __name__ == "__main__":
    unittest.main()
