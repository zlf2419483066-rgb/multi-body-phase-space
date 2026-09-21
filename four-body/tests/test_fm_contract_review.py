import json
import math
import unittest
from dataclasses import replace

from im_fixtures import fixture_b, fixture_d
from test_fm_formula import pair_fixture
from phase_space_graph import (
    build_formula_input, generate_formula, FormulaOptions, normalize_physical_labeling,
    canonicalize_physical_labeling, validate_artifact,
)
from phase_space_graph.formula.raw import reduce_covariant


class FormulaContractReviewTests(unittest.TestCase):
    def setUp(self):
        self.card = build_formula_input(pair_fixture(), measure_convention="standard_2pi4")

    def test_conflicting_conventions_fail_for_cards_and_records(self):
        with self.assertRaises(ValueError):
            generate_formula(self.card, options=FormulaOptions(normalization="jing_psf"))
        with self.assertRaises(ValueError):
            generate_formula(pair_fixture(), measure_convention="standard_2pi4", options=FormulaOptions())

    def test_tampered_ready_input_cannot_bypass_lm_review(self):
        for changed in (
            replace(self.card, independent_variables=((1, 3), (2, 4))),
            replace(self.card, factors=()),
            replace(self.card, double_measures=()),
            replace(self.card, n=5),
            replace(self.card, formula_status={"formula_verified": True}),
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    generate_formula(changed)

    def test_default_identity_and_source_roundtrip_are_present(self):
        result = generate_formula(self.card, run_benchmark=False)
        expected = canonicalize_physical_labeling(pair_fixture()).to_dict()
        self.assertEqual(result.identity["key"], expected["key"])
        self.assertEqual(result.identity["mode"], "fixed_labels")
        self.assertEqual(len(result.input_digest), 64)
        loaded = json.loads(result.to_json())
        self.assertEqual(loaded["input_card"]["record"], result.input_card["record"])
        self.assertEqual(loaded["statuses"]["source"], "pending")

    def test_tampered_identity_rejected(self):
        identity = canonicalize_physical_labeling(pair_fixture())
        card = build_formula_input(pair_fixture(), measure_convention="standard_2pi4", identity=identity)
        bad = dict(card.identity, digest="wrong")
        with self.assertRaises(ValueError):
            generate_formula(replace(card, identity=bad))

    def test_b_raw_keeps_all_incoming_references(self):
        fixture = fixture_b()
        card = build_formula_input(normalize_physical_labeling(fixture.topology, fixture.labeling),
                                   measure_convention="jing_psf")
        from phase_space_graph.formula.raw import build_f0_expression
        raw = build_f0_expression(card)
        source = next(f for f in raw.factor_sources if f["vertex"] == 2)
        self.assertEqual(source["incoming_coefficients"], [0, 1, 1, 1])
        self.assertEqual(len(source["incoming_references"]), 2)
        self.assertIn("jing_psf", raw.expression)
        self.assertIn("CancelCopies", raw.expression)

    def test_covariant_target_has_ds_and_concrete_constraints_but_no_fake_proof(self):
        result = reduce_covariant(self.card)
        self.assertEqual(result.status, "pending")
        self.assertIn("ds12 delta(s12-(p_1+p_2)^2)", result.expression)
        self.assertIn("ds34 delta(s34-(p_3+p_4)^2)", result.expression)
        self.assertNotIn("COV-tree-elimination", [step.rule_id for step in result.steps])
        self.assertTrue(result.diagnostics)

    def test_dependent_input_stays_blocked(self):
        fx = fixture_d()
        result = generate_formula(normalize_physical_labeling(fx.topology, fx.labeling))
        self.assertEqual(result.statuses["direct"], "blocked")
        self.assertIsNone(result.direct_formula)
        self.assertIsNotNone(result.input_card)

    def test_unsupported_options_are_not_silently_ignored(self):
        for key, value in (("reference_profile", "missing"), ("direct_profile_policy", "lookup"),
                           ("schema_version", "unknown")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                FormulaOptions(**{key: value})

    def test_corrupted_status_and_source_digest_fail_validation(self):
        result = generate_formula(self.card, run_benchmark=False)
        bad = replace(result, statuses=dict(result.statuses, direct="verified", benchmark="verified"), input_digest="bad")
        self.assertFalse(bad.direct_verified)
        failures = {f.check for f in validate_artifact(bad) if f.status == "failed"}
        self.assertTrue({"direct-integrity", "benchmark-integrity", "input-provenance"} <= failures)

    def test_massless_density_has_closed_form_normalization_not_just_path_agreement(self):
        # Approved reference formula, evaluated independently by hand at
        # M=4, s12=s34=1, mi=0: sqrt(lambda)/M^2=sqrt(3)/2.
        # Three solid angles are retained; this is per ds12 ds34 dOmega^3.
        point = {"s12": 1., "s34": 1., "theta0": 1., "phi0": .2,
                 "thetaA": .7, "phiA": 1.2, "thetaB": 2., "phiB": 2.7}
        expected = math.sqrt(3.) / (2**18 * math.pi**8)
        for convention, multiplier in (("standard_2pi4", 1.), ("jing_psf", (2*math.pi)**-4)):
            result = generate_formula(pair_fixture(), measure_convention=convention, run_benchmark=False)
            direct_value = result.direct_formula.numeric_density(point, (0.,)*4, 4.)
            ref_point = {"s12": 1., "s34": 1., "theta0": 1., "phi0": .2,
                         "theta12": .7, "phi12": 1.2, "theta34": 2., "phi34": 2.7}
            ref_value = result.bridge_formula.numeric_density(ref_point, (0.,)*4, 4.)
            self.assertAlmostEqual(direct_value / (expected*multiplier), 1., places=13)
            self.assertAlmostEqual(ref_value / (expected*multiplier), 1., places=13)


if __name__ == "__main__":
    unittest.main()
