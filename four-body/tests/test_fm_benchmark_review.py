"""Regression tests for the FM-12 benchmark's evidence boundary."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

from phase_space_graph import build_formula_input
from phase_space_graph.formula.benchmark import compare_paths, map_reference_to_target
from phase_space_graph.formula.direct_charts import build_direct_formula, select_direct_chart
from phase_space_graph.formula.reference import build_bridge_formula, build_reference_chart

from test_fm_formula import pair_fixture


MASS_VALUES = (1.0, 1.5, 0.8, 1.2)
TOTAL_MASS = 6.0


def _paths():
    card = build_formula_input(pair_fixture(), measure_convention="standard_2pi4")
    direct = build_direct_formula(card, select_direct_chart(card))
    bridge = build_bridge_formula(card, build_reference_chart(card))
    direct_attrs = {
        "profile": direct.profile,
        "target_variables": direct.target_variables,
        "angle_variables": direct.angle_variables,
        "branches": direct.branches,
        "status": direct.status,
        "numeric_density": direct.numeric_density,
        "numeric_momenta": direct.numeric_momenta,
        "normalization": "standard_2pi4",
        "differential_measure": "solid_angle",
        "angle_definitions": direct.angle_definitions,
    }
    bridge_attrs = {
        "profile": bridge.profile,
        "variables": bridge.variables,
        "status": bridge.status,
        "numeric_density": bridge.numeric_density,
        "numeric_momenta": bridge.numeric_momenta,
        "normalization": "standard_2pi4",
        "differential_measure": "solid_angle",
        "angle_definitions": bridge.angle_definitions,
    }
    return SimpleNamespace(**direct_attrs), SimpleNamespace(**bridge_attrs)


def _first_point():
    # This is the documented deterministic point family, copied as a concrete
    # replay point so error-injection tests do not depend on a private helper.
    return {
        "s12": 2.68**2,
        "s34": 2.15**2,
        "theta0": 0.61,
        "phi0": 0.23,
        "theta12": 0.82,
        "phi12": 1.17,
        "theta34": 1.46,
        "phi34": 3.21,
    }


class FormulaBenchmarkReviewTests(unittest.TestCase):
    def test_default_sample_is_replayable_and_has_evidence(self):
        direct, bridge = _paths()
        report = compare_paths(direct, bridge, masses=MASS_VALUES, total_mass=TOTAL_MASS)
        self.assertEqual(report.status, "verified")
        self.assertGreaterEqual(report.points, 8)
        self.assertEqual(report.points, report.compared)
        evidence = getattr(report, "evidence", None)
        if evidence is not None:
            self.assertEqual(len(evidence["replay_points"]), report.points)
            self.assertEqual(tuple(evidence["masses"]), MASS_VALUES)
            self.assertEqual(evidence["scope"], "local_numeric_consistency")
            self.assertEqual(evidence["global_claim"], "not_a_global_formula_proof")
        else:
            self.assertTrue(any("replay_evidence=" in item for item in report.diagnostics))

    def test_empty_sample_cannot_verify(self):
        direct, bridge = _paths()
        report = compare_paths(direct, bridge, masses=MASS_VALUES, total_mass=TOTAL_MASS, points=())
        self.assertNotEqual(report.status, "verified")
        self.assertEqual(report.compared, 0)

    def test_invalid_input_evidence_is_strict_json_safe(self):
        direct, bridge = _paths()
        report = compare_paths(
            direct,
            bridge,
            masses=(float("nan"), 1.5, 0.8, 1.2),
            total_mass=float("nan"),
            relative_tolerance=float("nan"),
        )
        self.assertNotEqual(report.status, "verified")
        json.dumps(report.to_dict(), allow_nan=False)

    def test_nan_density_fails_and_json_is_finite(self):
        direct, bridge = _paths()
        direct.numeric_density = lambda values, masses, total_mass: float("nan")
        report = compare_paths(direct, bridge, masses=MASS_VALUES, total_mass=TOTAL_MASS, points=(_first_point(),))
        self.assertNotEqual(report.status, "verified")
        self.assertIn("strictly positive", report.failures[0]["error"])
        json.dumps(report.to_dict(), allow_nan=False)

    def test_missing_and_short_momentum_adapters_cannot_pass(self):
        direct, bridge = _paths()
        direct.numeric_momenta = None
        missing = compare_paths(direct, bridge, masses=MASS_VALUES, total_mass=TOTAL_MASS, points=(_first_point(),))
        self.assertNotEqual(missing.status, "verified")

        direct, bridge = _paths()
        direct.numeric_momenta = lambda values, masses, total_mass: ((1.0, 0.0, 0.0, 0.0),)
        short = compare_paths(direct, bridge, masses=MASS_VALUES, total_mass=TOTAL_MASS, points=(_first_point(),))
        self.assertNotEqual(short.status, "verified")
        self.assertIn("expected 4", short.failures[0]["error"])

    def test_same_wrong_momenta_are_checked_against_physics(self):
        direct, bridge = _paths()
        direct_original = direct.numeric_momenta
        reference_original = bridge.numeric_momenta

        def wrong_direct_momenta(values, masses, total_mass):
            valid = direct_original(values, masses, total_mass)
            return tuple((vector[0] + 0.1, vector[1], vector[2], vector[3]) for vector in valid)

        def wrong_reference_momenta(values, masses, total_mass):
            valid = reference_original(values, masses, total_mass)
            return tuple((vector[0] + 0.1, vector[1], vector[2], vector[3]) for vector in valid)

        direct.numeric_momenta = wrong_direct_momenta
        bridge.numeric_momenta = wrong_reference_momenta
        report = compare_paths(direct, bridge, masses=MASS_VALUES, total_mass=TOTAL_MASS, points=(_first_point(),))
        self.assertNotEqual(report.status, "verified")
        error = report.failures[0]["error"]
        self.assertIn("direct", error)
        self.assertIn("reference", error)

    def test_physical_residuals_scale_with_parent_mass(self):
        direct, bridge = _paths()
        direct_original = direct.numeric_momenta
        reference_original = bridge.numeric_momenta

        def wrong_direct_momenta(values, masses, total_mass):
            valid = direct_original(values, masses, total_mass)
            return tuple((vector[0] + 1e-8, vector[1], vector[2], vector[3]) for vector in valid)

        def wrong_reference_momenta(values, masses, total_mass):
            valid = reference_original(values, masses, total_mass)
            return tuple((vector[0] + 1e-8, vector[1], vector[2], vector[3]) for vector in valid)

        direct.numeric_momenta = wrong_direct_momenta
        bridge.numeric_momenta = wrong_reference_momenta
        report = compare_paths(
            direct,
            bridge,
            masses=(1e-6, 1.5e-6, 0.8e-6, 1.2e-6),
            total_mass=6e-6,
        )
        self.assertNotEqual(report.status, "verified")
        self.assertTrue(any("energy conservation" in row["error"] for row in report.failures))

    def test_branch_metadata_and_normalization_must_be_supported(self):
        direct, bridge = _paths()
        direct.branches = ()
        unsupported = compare_paths(direct, bridge, masses=MASS_VALUES, total_mass=TOTAL_MASS, points=(_first_point(),))
        self.assertNotEqual(unsupported.status, "verified")

        direct, bridge = _paths()
        direct.branches = (replace(direct.branches[0], coefficient="2"),)
        coefficient = compare_paths(direct, bridge, masses=MASS_VALUES, total_mass=TOTAL_MASS, points=(_first_point(),))
        self.assertNotEqual(coefficient.status, "verified")

        direct, bridge = _paths()
        bridge.normalization = "jing_psf"
        mismatch = compare_paths(direct, bridge, masses=MASS_VALUES, total_mass=TOTAL_MASS, points=(_first_point(),))
        self.assertNotEqual(mismatch.status, "verified")

    def test_reference_map_renames_angles_without_leaking_source_names(self):
        mapped = map_reference_to_target(_first_point(), target=("s12", "s34"))
        self.assertIsNotNone(mapped)
        self.assertEqual(mapped["thetaA"], _first_point()["theta12"])
        self.assertEqual(mapped["phiB"], _first_point()["phi34"])
        self.assertNotIn("theta12", mapped)
        self.assertNotIn("phi34", mapped)
        self.assertIsNone(map_reference_to_target(_first_point(), target=("s13", "s24")))


if __name__ == "__main__":
    unittest.main()
