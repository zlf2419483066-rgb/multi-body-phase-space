"""Occurrence-bound local measure reduction tests; no global formula claims."""

from dataclasses import replace
import unittest

from im_fixtures import fixture_b, fixture_f
from phase_space_graph import (
    CancellationPlanOverride,
    OccurrenceLocator,
    build_cancellation_plan,
    build_formula_input,
    canonicalize_physical_labeling,
    reduce_local_measures,
)
from phase_space_graph.formula.expanded import build_expanded_f0_expression


def _card(fixture, convention="jing_psf"):
    identity = canonicalize_physical_labeling(fixture.record)
    return build_formula_input(fixture.record, measure_convention=convention, identity=identity)


class LocalReductionTests(unittest.TestCase):
    def test_b_external_override_binds_and_reduces_only_its_local_measure(self):
        card = _card(fixture_b())
        plan = build_cancellation_plan(
            card,
            overrides=(CancellationPlanOverride(
                particle_id=2,
                cancellation_occurrences=(OccurrenceLocator("external", 3, 0),),
                reason="HAPOF p.28 B source choice: cancel terminal external p2.",
            ),),
        )
        expanded = build_expanded_f0_expression(card)
        trace = reduce_local_measures(plan, expanded)

        self.assertEqual(trace.status, "partial")
        self.assertEqual(trace.input_card_digest, expanded.input_card_digest)
        self.assertEqual(trace.measure_convention, "jing_psf")
        self.assertEqual(trace.covariant_status, "pending")
        self.assertFalse(trace.covariant_certificate_built)
        self.assertFalse(trace.full_covariant_reduction_performed)
        self.assertFalse(trace.formula_verified)
        self.assertEqual(
            trace.preserved_double_line_factors,
            tuple(item.to_dict() for item in expanded.double_line_factors),
        )

        by_vertex = {item.vertex: item for item in trace.factors}
        local = by_vertex[3]
        self.assertEqual(local.status, "locally_reduced")
        self.assertEqual(local.canceled_occurrences[0]["occurrence_id"], "p2:external:v3:slot0")
        self.assertEqual(local.removed_measures[0]["measure"]["particle_ids"], [2])
        self.assertEqual(local.integrated_variable, "p_3")
        self.assertEqual(local.integrated_measure["source_reference"]["slot"], 1)
        self.assertEqual(local.delta4_jacobian, "1")
        self.assertEqual(local.two_pi_accounting.net_local_power, -3)
        self.assertEqual(local.two_pi_accounting.canceled_measure_denominator_powers, (3,))
        self.assertEqual(local.two_pi_accounting.residual_measure_denominator_powers, (3,))
        constraint = local.remaining_constraint
        self.assertEqual(constraint.kind, "on_shell_residual")
        self.assertEqual(constraint.delta, "delta(((q_{4,1}) - (p_2))^2 - m_3^2)")
        self.assertEqual(constraint.positive_energy, "theta(((q_{4,1}) - (p_2))^0)")
        self.assertEqual(constraint.denominator_power, 3)
        self.assertEqual(constraint.source_measure["source_reference"]["particle_ids"], [3])
        self.assertEqual(
            [item.status for item in trace.factors if item.vertex != 3],
            ["preserved", "preserved", "preserved"],
        )

    def test_standard_convention_keeps_prefactor_and_residual_denominator_exponents(self):
        card = _card(fixture_b(), "standard_2pi4")
        plan = build_cancellation_plan(
            card,
            overrides=(CancellationPlanOverride(
                particle_id=2,
                cancellation_occurrences=(OccurrenceLocator("external", 3, 0),),
                reason="formula-level local reduction test",
            ),),
        )
        trace = reduce_local_measures(plan, build_expanded_f0_expression(card))
        local = next(item for item in trace.factors if item.vertex == 3)

        self.assertEqual(trace.measure_convention, "standard_2pi4")
        self.assertEqual(local.two_pi_accounting.local_prefactor_power, 4)
        self.assertEqual(local.two_pi_accounting.net_local_power, 1)
        self.assertIn("(2*pi)^4", local.after_expression)
        self.assertIn("/ (2*pi)^3", local.after_expression)

    def test_existing_f_fixture_both_canceled_vertex_stays_pending_with_delta4(self):
        card = _card(fixture_f())
        plan = build_cancellation_plan(card)
        expanded = build_expanded_f0_expression(card)
        trace = reduce_local_measures(plan, expanded)

        pending = [item for item in trace.factors if item.status == "pending_cross_vertex_integration"]
        self.assertTrue(pending)
        local = pending[0]
        self.assertEqual(len(local.canceled_occurrences), 2)
        self.assertEqual(len(local.removed_measures), 2)
        self.assertIsNone(local.integrated_measure)
        self.assertIsNone(local.integrated_variable)
        self.assertIsNone(local.delta4_jacobian)
        self.assertEqual(local.remaining_constraint.kind, "local_delta4_pending")
        self.assertEqual(local.remaining_constraint.awaiting, "cross_vertex_integration")
        self.assertEqual(local.after_expression, local.source_factor["momentum_delta"])
        self.assertEqual(local.two_pi_accounting.net_local_power, 0)
        self.assertEqual(trace.status, "pending")
        self.assertTrue(trace.pending_reasons)
        self.assertNotIn("1/delta", trace.to_json())
        self.assertEqual(
            trace.preserved_double_line_factors,
            tuple(item.to_dict() for item in expanded.double_line_factors),
        )

    def test_rejects_digest_identity_and_convention_mismatches(self):
        card = _card(fixture_b())
        plan = build_cancellation_plan(card)
        expanded = build_expanded_f0_expression(card)

        with self.assertRaisesRegex(ValueError, "input-card digests disagree"):
            reduce_local_measures(plan, replace(expanded, input_card_digest="0" * 64))
        changed_identity = dict(expanded.identity)
        changed_identity["digest"] = "mismatched"
        with self.assertRaisesRegex(ValueError, "marked identities disagree"):
            reduce_local_measures(plan, replace(expanded, identity=changed_identity))
        with self.assertRaisesRegex(ValueError, "measure conventions disagree"):
            reduce_local_measures(plan, replace(expanded, measure_convention="standard_2pi4"))


if __name__ == "__main__":
    unittest.main()
