import json
import unittest

from im_fixtures import fixture_b, fixture_f
from phase_space_graph import build_formula_input, normalize_physical_labeling
from phase_space_graph.formula.expanded import build_expanded_f0_expression
from test_fm_formula import pair_fixture


def card_from_b(convention="standard_2pi4"):
    fixture = fixture_b()
    record = normalize_physical_labeling(fixture.topology, fixture.labeling)
    return build_formula_input(record, measure_convention=convention)


class ExpandedF0Tests(unittest.TestCase):
    def test_standard_convention_expands_each_local_factor_and_retains_sources(self):
        card = build_formula_input(pair_fixture(), measure_convention="standard_2pi4")
        expanded = build_expanded_f0_expression(card)

        self.assertEqual(expanded.status, "expanded_pre_cancellation")
        self.assertEqual(len(expanded.factors), len(card.factors))
        self.assertEqual(len(expanded.double_line_factors), len(card.double_measures))
        self.assertTrue(all(factor.normalization_2pi_power == 4 for factor in expanded.factors))
        self.assertTrue(all(factor.normalization_prefactor == "(2*pi)^4" for factor in expanded.factors))
        self.assertEqual(sum(len(factor.outgoing_measures) for factor in expanded.factors),
                         2 * len(card.factors))
        for source_factor, result_factor in zip(card.factors, expanded.factors):
            self.assertEqual(result_factor.vertex, source_factor.vertex)
            self.assertEqual(result_factor.outgoing_support_momenta,
                             tuple(line.momentum_expression for line in source_factor.outgoing))
            expected_integration_momenta = tuple(
                f"q_{{{line.edge_index + 1},{line.occurrence + 1}}}"
                if line.role == "double" else line.momentum_expression
                for line in source_factor.outgoing
            )
            self.assertEqual(result_factor.outgoing_integration_momenta,
                             expected_integration_momenta)
            self.assertEqual(len(result_factor.incoming_references),
                             len(source_factor.incoming_references))
            self.assertEqual(len(result_factor.outgoing_measures), 2)
            self.assertTrue(all(item.denominator == "(2*pi)^3"
                                for item in result_factor.outgoing_measures))
            self.assertIn("delta^4(", result_factor.momentum_delta)

        for source_measure, result_measure in zip(card.double_measures,
                                                   expanded.double_line_factors):
            self.assertEqual(result_measure.edge_index, source_measure.edge_index)
            self.assertEqual(result_measure.occurrence, source_measure.occurrence)
            self.assertEqual(result_measure.differential, source_measure.differential)

        payload = json.loads(expanded.to_json())
        self.assertEqual(payload["input_card_digest"], expanded.input_card_digest)
        self.assertEqual(payload["factors"][0]["outgoing_measures"][0]["kind"],
                         "on_shell_measure")
        self.assertFalse(payload["cancellation_performed"])
        self.assertFalse(payload["integration_performed"])
        self.assertFalse(payload["covariant_reduction_performed"])
        self.assertFalse(payload["formula_verified"])
        self.assertNotIn("CancelCopies", expanded.expression)

    def test_jing_convention_has_no_local_two_pi4_prefactor(self):
        card = build_formula_input(pair_fixture(), measure_convention="jing_psf")
        expanded = build_expanded_f0_expression(card)

        self.assertTrue(all(factor.normalization_2pi_power == 0 for factor in expanded.factors))
        self.assertTrue(all(factor.normalization_prefactor == "1" for factor in expanded.factors))
        self.assertNotIn("(2*pi)^4", expanded.expression)
        self.assertTrue(all(measure.denominator == "(2*pi)^3"
                            for factor in expanded.factors
                            for measure in factor.outgoing_measures))
        self.assertEqual(
            [item.differential for item in expanded.double_line_factors],
            [item.differential for item in card.double_measures],
        )

    def test_b_parent_mass_does_not_create_extra_double_line_measure_or_cancel_p2(self):
        card = card_from_b()
        expanded = build_expanded_f0_expression(card)
        parent_block = next(item for item in card.factors if item.vertex == 2)
        self.assertIn("p_2", parent_block.incoming_momentum)
        self.assertEqual(parent_block.parent_mass_squared, "s_234")

        # The expanded tree retains both occurrences of p2's one-particle
        # measure. The shared-particle ledger is deliberately not applied here.
        p2_measures = [
            measure
            for factor in expanded.factors
            for measure in factor.outgoing_measures
            if measure.particle_ids == (2,)
        ]
        self.assertEqual(len(p2_measures), 2)
        self.assertEqual(len(expanded.double_line_factors), len(card.double_measures))
        self.assertEqual(
            [item.variable_label for item in expanded.double_line_factors],
            [item.variable_label for item in card.double_measures],
        )
        self.assertNotIn("s234", [item.variable_label for item in expanded.double_line_factors])
        self.assertFalse(expanded.cancellation_performed)
        self.assertEqual(
            sorted((measure.source_reference.vertex, measure.source_reference.role)
                   for measure in p2_measures),
            [(1, "single"), (3, "external")],
        )

    def test_b_vertex_zero_integrates_double_line_q_and_retains_particle_sum_as_support(self):
        card = card_from_b()
        expanded = build_expanded_f0_expression(card)
        source_factor = next(item for item in card.factors if item.vertex == 0)
        factor = next(item for item in expanded.factors if item.vertex == 0)

        self.assertEqual(factor.outgoing_integration_momenta, ("q_{1,1}", "q_{2,1}"))
        self.assertEqual(
            factor.outgoing_support_momenta,
            tuple(line.momentum_expression for line in source_factor.outgoing),
        )
        self.assertEqual(factor.outgoing_support_momenta, ("p_1 + p_2", "p_3 + p_4"))
        self.assertIn("delta^4((P) - (q_{1,1}) - (q_{2,1}))", factor.momentum_delta)
        for measure, source in zip(factor.outgoing_measures, source_factor.outgoing):
            self.assertEqual(measure.momentum_expression,
                             f"q_{{{source.edge_index + 1},{source.occurrence + 1}}}")
            self.assertEqual(measure.mass_squared, source.mass_squared)
            self.assertIn(measure.momentum_expression, measure.differential)
            self.assertNotIn(source.momentum_expression, measure.differential)
            self.assertEqual(measure.support_momentum_expression, source.momentum_expression)
            self.assertEqual(measure.source_reference.edge_index, source.edge_index)
            self.assertEqual(measure.source_reference.occurrence, source.occurrence)
        self.assertEqual(
            [item.mass_squared for item in factor.outgoing_measures],
            ["s_12", "s_34"],
        )

    def test_parallel_outgoing_occurrences_keep_distinct_references(self):
        fixture = fixture_f()
        card = build_formula_input(
            normalize_physical_labeling(fixture.topology, fixture.labeling),
            measure_convention="jing_psf",
        )
        expanded = build_expanded_f0_expression(card)
        factor = next(item for item in expanded.factors if item.vertex == 2)
        parallel = [
            item for item in factor.outgoing_measures
            if item.source_reference.edge_index == 4
        ]

        self.assertEqual(
            [(item.source_reference.occurrence, item.particle_ids) for item in parallel],
            [(0, (1,)), (1, (2,))],
        )
        self.assertEqual(factor.outgoing_integration_momenta, ("p_1", "p_2"))
        self.assertEqual(factor.outgoing_support_momenta, ("p_1", "p_2"))

    def test_unavailable_or_tampered_cards_are_rejected(self):
        from dataclasses import replace

        card = build_formula_input(pair_fixture(), measure_convention="standard_2pi4")
        with self.assertRaises(TypeError):
            build_expanded_f0_expression(object())
        with self.assertRaises(ValueError):
            build_expanded_f0_expression(replace(card, double_measures=()))


if __name__ == "__main__":
    unittest.main()
