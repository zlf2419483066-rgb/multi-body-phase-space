"""Independent local evidence for the d=5 direct/reference comparison."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

from phase_space_graph import generate_formula
from phase_space_graph.formula.benchmark import compare_paths
from phase_space_graph.formula.d5_direct import build_d5_formula
from phase_space_graph.formula.records import FormulaOptions
from phase_space_graph.formula.reference import build_bridge_formula, build_reference_chart
from phase_space_graph.formula.validation import validate_artifact

from test_tree_covariant import _d5_card


MASSES = (1.0, 1.5, 0.8, 1.2)
TOTAL_MASS = 6.0


def _paths(*, density_mode: str = "per_branch"):
    card = _d5_card("standard_2pi4")
    options = FormulaOptions(normalization="standard_2pi4", density_mode=density_mode)
    direct = build_d5_formula(card, options=options)
    bridge = build_bridge_formula(card, build_reference_chart(card, options=options), options=options)
    mutable_direct = SimpleNamespace(
        profile=direct.profile,
        target_variables=direct.target_variables,
        angle_variables=direct.angle_variables,
        branches=direct.branches,
        status=direct.status,
        numeric_density=direct.numeric_density,
        numeric_momenta=direct.numeric_momenta,
        normalization=direct.normalization,
        differential_measure=direct.differential_measure,
        angle_definitions=direct.angle_definitions,
    )
    mutable_bridge = SimpleNamespace(
        profile=bridge.profile,
        variables=bridge.variables,
        status=bridge.status,
        numeric_density=bridge.numeric_density,
        numeric_momenta=bridge.numeric_momenta,
        normalization=bridge.normalization,
        differential_measure=bridge.differential_measure,
        angle_definitions=bridge.angle_definitions,
    )
    return mutable_direct, mutable_bridge


class D5BenchmarkTests(unittest.TestCase):
    def test_default_sample_checks_both_mirrors_and_two_fd_steps(self):
        direct, bridge = _paths()
        report = compare_paths(direct, bridge, masses=MASSES, total_mass=TOTAL_MASS)

        self.assertEqual(report.status, "verified_local")
        self.assertEqual(report.points, 8)
        self.assertEqual(report.compared, 8)
        self.assertEqual(report.branch_status, "both_mirrors_checked")
        evidence = report.evidence
        self.assertEqual(evidence["target_profile"], "direct_gram_d5")
        self.assertEqual(evidence["finite_difference_steps"], (2.0e-5, 1.0e-5))
        self.assertEqual(evidence["effective_density_tolerance"], 5.0e-6)
        self.assertEqual(evidence["effective_kinematic_residual_tolerance"], 1.0e-8)
        self.assertEqual(evidence["density_divisor_for_summed_mode"], 1.0)
        self.assertEqual(len(evidence["point_evidence"]), 8)
        for point in evidence["point_evidence"]:
            self.assertEqual(len(point["jacobians"]), 2)
            self.assertEqual(len(point["sine_weighted_density_comparison"]), 2)
            self.assertEqual(
                {row["branch_sigma"] for row in point["mirror_branches"]},
                {-1, 1},
            )
            self.assertEqual(sum(row["matching_reference_sigma"] for row in point["mirror_branches"]), 1)
            self.assertTrue(all(
                set(row["direct_quality"]["source_pair_relative_residuals"]) == {"s12", "s34"}
                and set(row["reference_mirror_quality"]["source_pair_relative_residuals"]) == {"s12", "s34"}
                for row in point["mirror_branches"]
            ))
        json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_summed_density_is_divided_by_two_before_branch_comparison(self):
        direct, bridge = _paths(density_mode="summed")
        report = compare_paths(direct, bridge, masses=MASSES, total_mass=TOTAL_MASS)

        self.assertEqual(report.status, "verified_local")
        self.assertEqual(report.evidence["density_divisor_for_summed_mode"], 2.0)
        for row in report.evidence["point_evidence"]:
            self.assertAlmostEqual(
                row["density_direct_per_branch"] * 2.0,
                row["density_direct_total"],
            )

    def test_equal_positive_masses_are_a_supported_benchmark_input(self):
        direct, bridge = _paths()
        report = compare_paths(direct, bridge, masses=(1.0, 1.0, 1.0, 1.0), total_mass=6.0)
        self.assertEqual(report.status, "verified_local")

    def test_wrong_density_coefficient_fails_finite_point_comparison(self):
        direct, bridge = _paths()
        exact_density = direct.numeric_density
        direct.numeric_density = lambda values, masses, total_mass: 1.2 * exact_density(values, masses, total_mass)

        report = compare_paths(direct, bridge, masses=MASSES, total_mass=TOTAL_MASS)

        self.assertEqual(report.status, "failed")
        self.assertTrue(report.failures)
        self.assertTrue(any("density residual" in row["error"] for row in report.failures))

    def test_wrong_mirror_adapter_fails_sigma_and_momentum_checks(self):
        direct, bridge = _paths()
        exact_momenta = direct.numeric_momenta
        direct.numeric_momenta = lambda values, masses, total_mass, branch=1: exact_momenta(
            values, masses, total_mass, branch=-branch,
        )

        report = compare_paths(direct, bridge, masses=MASSES, total_mass=TOTAL_MASS)

        self.assertEqual(report.status, "failed")
        self.assertTrue(report.failures)
        self.assertTrue(any("sigma=" in row["error"] or "momentum mismatch" in row["error"] for row in report.failures))

    def test_shared_invalid_momentum_adapters_do_not_pass(self):
        direct, bridge = _paths()
        reference_momenta = bridge.numeric_momenta
        common_by_invariants = {}

        def invariant_key(momenta):
            values = []
            for label in direct.target_variables:
                digits = label.removeprefix("s").removeprefix("_")
                selected = [int(digit) - 1 for digit in digits]
                summed = tuple(sum(momenta[index][axis] for index in selected) for axis in range(4))
                values.append(summed[0] ** 2 - sum(component ** 2 for component in summed[1:]))
            return tuple(values)

        def add_energy_offset(momenta):
            return tuple((item[0] + 0.1, *item[1:]) for item in momenta)

        def invalid_reference_adapter(values, masses, total_mass):
            invalid = add_energy_offset(reference_momenta(values, masses, total_mass))
            common_by_invariants[invariant_key(invalid)] = invalid
            return invalid

        def common_invalid_direct_adapter(values, masses, total_mass, branch=1):
            key = tuple(float(values[label]) for label in direct.target_variables)
            return common_by_invariants[key]

        bridge.numeric_momenta = invalid_reference_adapter
        direct.numeric_momenta = common_invalid_direct_adapter
        direct.numeric_density = lambda values, masses, total_mass: 1.0

        report = compare_paths(direct, bridge, masses=MASSES, total_mass=TOTAL_MASS)

        self.assertEqual(report.status, "failed")
        self.assertTrue(report.failures)
        self.assertTrue(any(
            "energy conservation" in row["error"] for row in report.failures
        ))

    def test_missing_adapter_and_short_sample_cannot_claim_local_verification(self):
        direct, bridge = _paths()
        direct.numeric_momenta = None
        blocked = compare_paths(direct, bridge, masses=MASSES, total_mass=TOTAL_MASS)
        self.assertEqual(blocked.status, "blocked")

        direct, bridge = _paths()
        from phase_space_graph.formula.benchmark import _default_points

        short = compare_paths(
            direct,
            bridge,
            masses=MASSES,
            total_mass=TOTAL_MASS,
            points=_default_points(MASSES, TOTAL_MASS)[:7],
        )
        self.assertEqual(short.status, "unsupported")
        self.assertEqual(short.compared, 0)

    def test_validation_rejects_empty_or_nonfinite_verified_local_evidence(self):
        direct, bridge = _paths()
        report = compare_paths(direct, bridge, masses=MASSES, total_mass=TOTAL_MASS)
        self.assertEqual(report.status, "verified_local")
        artifact = generate_formula(_d5_card("standard_2pi4"), run_benchmark=False)
        valid_artifact = replace(
            artifact,
            internal_benchmark=report,
            statuses={**artifact.statuses, "benchmark": "verified_local"},
        )
        self.assertFalse(valid_artifact.direct_verified)
        self.assertFalse(any(
            finding.check == "benchmark-evidence" and finding.status == "failed"
            for finding in validate_artifact(valid_artifact)
        ))

        missing_artifact = replace(
            artifact,
            internal_benchmark=None,
            statuses={**artifact.statuses, "benchmark": "verified_local"},
        )
        missing_findings = validate_artifact(missing_artifact)
        self.assertTrue(any(
            finding.check == "benchmark-integrity" and finding.status == "failed"
            for finding in missing_findings
        ))

        empty_report = replace(report, evidence={})
        empty_artifact = replace(
            artifact,
            internal_benchmark=empty_report,
            statuses={**artifact.statuses, "benchmark": "verified_local"},
        )
        empty_findings = validate_artifact(empty_artifact)
        self.assertTrue(any(
            finding.check == "benchmark-evidence" and finding.status == "failed"
            for finding in empty_findings
        ))

        nan_report = replace(report, evidence={**report.evidence, "injected_nonfinite": float("nan")})
        nan_artifact = replace(
            artifact,
            internal_benchmark=nan_report,
            statuses={**artifact.statuses, "benchmark": "verified_local"},
        )
        nan_findings = validate_artifact(nan_artifact)
        self.assertTrue(any(
            finding.check == "benchmark-evidence" and finding.status == "failed"
            for finding in nan_findings
        ))
        self.assertFalse(nan_artifact.direct_verified)


if __name__ == "__main__":
    unittest.main()
