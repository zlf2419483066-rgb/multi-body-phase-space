"""B/F03 fixed-label global elimination trace tests."""

from dataclasses import replace
import json
import unittest

from im_fixtures import fixture_b, fixture_f
from phase_space_graph import (
    CancellationPlanOverride,
    OccurrenceLocator,
    build_b_covariant_certificate,
    build_cancellation_plan,
    build_formula_input,
    canonicalize_physical_labeling,
    reduce_local_measures,
)
from phase_space_graph.formula.expanded import build_expanded_f0_expression


def _card(fixture, convention="jing_psf"):
    identity = canonicalize_physical_labeling(fixture.record)
    return build_formula_input(fixture.record, measure_convention=convention, identity=identity)


def _b_inputs(convention="jing_psf"):
    card = _card(fixture_b(), convention)
    plan = build_cancellation_plan(
        card,
        overrides=(CancellationPlanOverride(
            particle_id=2,
            cancellation_occurrences=(OccurrenceLocator("external", 3, 0),),
            reason="HAPOF p.28 B source choice: cancel terminal external p2.",
        ),),
    )
    expanded = build_expanded_f0_expression(card)
    local = reduce_local_measures(plan, expanded)
    return card, expanded, plan, local


class BFCovariantCertificateTests(unittest.TestCase):
    def test_b_trace_records_ordered_conservation_elimination_and_unique_delta_use(self):
        card, expanded, plan, local = _b_inputs()
        certificate = build_b_covariant_certificate(card, expanded, plan, local)

        self.assertEqual(certificate.status, "built")
        self.assertFalse(certificate.formula_verified)
        self.assertEqual(certificate.source_match_status, "not_checked")
        self.assertEqual(certificate.input_card_digest, expanded.input_card_digest)
        p2_plan = next(item for item in plan.particles if item.particle_id == 2)
        self.assertEqual(p2_plan.retained_occurrence.locator, OccurrenceLocator("internal", 2, 0))
        self.assertEqual(
            tuple(item.locator for item in p2_plan.cancellation_occurrences),
            (OccurrenceLocator("external", 3, 0),),
        )
        self.assertEqual(certificate.measure_convention, "jing_psf")
        self.assertEqual(certificate.ordered_variables, ("s12", "s34", "s23"))
        self.assertEqual(
            tuple(item.equation for item in certificate.conservation_equations),
            (
                "P = q_{1,1} + q_{2,1}",
                "q_{1,1} = p_1 + p_2",
                "q_{2,1} + p_2 = p_4 + q_{4,1}",
                "q_{4,1} = p_2 + p_3",
            ),
        )

        steps = certificate.elimination_steps
        self.assertEqual([item.pair_momentum_label for item in steps], ["q12", "q23", "q34"])
        self.assertEqual([item.relation for item in steps], [
            "q_{1,1} = p_1 + p_2",
            "q_{4,1} = p_2 + p_3",
            "q_{2,1} = p_3 + p_4",
        ])
        self.assertEqual([item.absolute_jacobian for item in steps], ["1", "1", "1"])
        self.assertEqual([item.shell_transfer.variable_label for item in steps], ["s12", "s23", "s34"])
        self.assertEqual(
            [item.shell_transfer.result_delta for item in steps],
            [
                "delta(s12 - (p_1 + p_2)^2)",
                "delta(s23 - (p_2 + p_3)^2)",
                "delta(s34 - (p_3 + p_4)^2)",
            ],
        )
        self.assertTrue(all(item.shell_transfer.positive_energy_implied for item in steps))
        self.assertTrue(all(item.shell_transfer.double_line_differential for item in steps))

        q23_step = steps[1]
        self.assertIsNone(q23_step.source_conservation_delta4)
        self.assertEqual(q23_step.source_local_constraint["kind"], "on_shell_residual")
        self.assertIn("already consumed", q23_step.delta4_consumption)
        self.assertEqual(q23_step.substitutions[0]["to"], "p_3")
        dispositions = {item.vertex: item for item in certificate.delta4_dispositions}
        self.assertEqual(set(dispositions), {0, 1, 2, 3})
        self.assertEqual(dispositions[0].disposition, "retained_global_conservation")
        self.assertEqual(dispositions[1].consumed_by, "q_elimination_1")
        self.assertEqual(dispositions[2].consumed_by, "q_elimination_3")
        self.assertEqual(dispositions[3].consumed_by, "LocalReductionTrace.vertex_3")

        retained = {item.particle_id: item for item in certificate.retained_particle_measures}
        self.assertEqual(set(retained), {1, 2, 3, 4})
        self.assertEqual(retained[2].retained_occurrence["kind"], "internal")
        self.assertEqual(retained[2].retained_occurrence["edge_index"], 2)
        self.assertEqual(retained[3].representation, "recovered_from_local_residual_constraint_by_q23_translation")
        self.assertEqual(
            certificate.preserved_double_line_factors,
            tuple(item.to_dict() for item in expanded.double_line_factors),
        )
        self.assertEqual(certificate.residual_global_equation, "P = p_1 + p_2 + p_3 + p_4")
        self.assertEqual(certificate.residual_global_delta4, "delta^4((P) - (p_1) - (p_2) - (p_3) - (p_4))")
        self.assertEqual(json.loads(certificate.to_json()), certificate.to_dict())

    def test_jing_and_standard_two_pi_ledgers_differ_by_global_four(self):
        jing = build_b_covariant_certificate(*_b_inputs("jing_psf"))
        standard = build_b_covariant_certificate(*_b_inputs("standard_2pi4"))

        self.assertEqual(jing.normalization.global_coefficient_power, 0)
        self.assertEqual(jing.normalization.final_total_power, -12)
        self.assertEqual(jing.normalization.double_line_differential_power, 9)
        self.assertEqual(standard.normalization.global_coefficient_power, 4)
        self.assertEqual(standard.normalization.final_total_power, -8)
        self.assertEqual(standard.normalization.double_line_differential_power, -3)
        self.assertEqual(standard.normalization.standard_minus_jing_power, 4)
        self.assertEqual(standard.normalization.total_local_prefactor_power, 16)
        self.assertEqual(standard.normalization.q_shell_denominator_power, -9)

    def test_rejects_default_lm_plan_wrong_digest_and_wrong_structure(self):
        card, expanded, plan, local = _b_inputs()
        default_plan = build_cancellation_plan(card)
        with self.assertRaisesRegex(ValueError, "retaining internal p2 and canceling terminal external p2"):
            build_b_covariant_certificate(card, expanded, default_plan, reduce_local_measures(default_plan, expanded))

        with self.assertRaisesRegex(ValueError, "digest/convention does not match"):
            build_b_covariant_certificate(card, expanded, replace(plan, input_card_digest="0" * 64), local)

        wrong_expanded = build_expanded_f0_expression(_card(fixture_f()))
        with self.assertRaisesRegex(ValueError, "disagrees with the validated B/F03 card"):
            build_b_covariant_certificate(card, wrong_expanded, plan, local)


if __name__ == "__main__":
    unittest.main()
