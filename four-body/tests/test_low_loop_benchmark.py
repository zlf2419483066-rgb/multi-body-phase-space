"""Finite-point checks for the independent low-loop mixed-coordinate benchmark."""

from __future__ import annotations

import json
import math
import unittest
from functools import lru_cache
from types import SimpleNamespace

from phase_space_graph import build_formula_input
from phase_space_graph.formula.all_loop_catalog import build_all_loop_catalog
from phase_space_graph.formula.low_loop_benchmark import _independent_internal_coordinates, compare_low_loop_paths
from phase_space_graph.formula.low_loop_direct import build_low_loop_chart_data_from_labels, build_low_loop_formula
from phase_space_graph.formula.records import FormulaOptions
from phase_space_graph.formula.reference import build_bridge_formula, build_reference_chart


MASSES = (1.0, 1.5, 0.8, 1.2)
TOTAL_MASS = 6.0


@lru_cache(maxsize=1)
def _catalog():
    return build_all_loop_catalog()


def _paths(loop_count: int, *, density_mode: str = "per_branch", phi_eliminated: bool | None = None):
    candidates = tuple(item for item in _catalog().classes if item.loop_count == loop_count)
    selected = None
    for item in candidates:
        labels = tuple("s" + "".join(str(i) for i in row) for row in item.representative_record.independence.variables)
        chart = build_low_loop_chart_data_from_labels(labels)
        if phi_eliminated is None or bool(chart["elimination"]["phi_eliminated"]) == phi_eliminated:
            selected = (item, labels, chart)
            break
    if selected is None:
        raise AssertionError(f"no L={loop_count} chart found for phi_eliminated={phi_eliminated}")
    item, labels, chart_data = selected
    card = build_formula_input(item.representative_record, measure_convention="standard_2pi4")
    options = FormulaOptions(normalization="standard_2pi4", density_mode=density_mode)
    direct_record = build_low_loop_formula(card, options=options)
    bridge_record = build_bridge_formula(card, build_reference_chart(card, options=options), options=options)
    direct = SimpleNamespace(
        profile=direct_record.profile,
        target_variables=direct_record.target_variables,
        angle_variables=direct_record.angle_variables,
        branches=direct_record.branches,
        status=direct_record.status,
        numeric_density=direct_record.numeric_density,
        numeric_momenta=direct_record.numeric_momenta,
        normalization=direct_record.normalization,
    )
    bridge = SimpleNamespace(
        profile=bridge_record.profile,
        variables=bridge_record.variables,
        status=bridge_record.status,
        numeric_density=bridge_record.numeric_density,
        numeric_momenta=bridge_record.numeric_momenta,
        normalization=bridge_record.normalization,
    )
    return direct, bridge, chart_data, labels


def _sample_with_phi_above_pi(bridge, chart, base_sample):
    from phase_space_graph.formula.benchmark import _default_points

    sample = list(base_sample or _default_points(MASSES, TOTAL_MASS))
    for axis in ("phi34", "phi12"):
        for index in range(1, 128):
            candidate = dict(sample[0])
            candidate[axis] = (index + 0.37) * 2.0 * math.pi / 128.0
            momenta = bridge.numeric_momenta(candidate, MASSES, TOTAL_MASS)
            internal = _independent_internal_coordinates(momenta, chart["geometry"])
            if internal["internal_phi"] > math.pi:
                sample[0] = candidate
                return tuple(sample)
    raise AssertionError("could not construct a retained internal_phi > pi replay point")


class LowLoopBenchmarkTests(unittest.TestCase):
    def test_explicit_phi_chart_has_eight_dimensional_two_step_evidence(self):
        direct, bridge, chart, _ = _paths(0, phi_eliminated=False)
        from phase_space_graph.formula.benchmark import _default_points
        points = _sample_with_phi_above_pi(bridge, chart, _default_points(MASSES, TOTAL_MASS))
        called_branches = []
        exact_density = direct.numeric_density

        def record_density_branch(values, masses, total_mass, branch=1):
            called_branches.append(branch)
            return exact_density(values, masses, total_mass, branch=branch)

        direct.numeric_density = record_density_branch

        report = compare_low_loop_paths(direct, bridge, chart_data=chart, masses=MASSES, total_mass=TOTAL_MASS, points=points)

        self.assertEqual(report.status, "verified_local")
        self.assertEqual(report.points, 8)
        self.assertEqual(report.compared, 8)
        self.assertEqual(report.branch_status, "explicit_internal_phi_mirror_checked")
        evidence = report.evidence
        self.assertEqual(evidence["finite_difference_steps"], (2.0e-5, 1.0e-5))
        self.assertTrue(all(branch == 1 for branch in called_branches))
        self.assertTrue(any(
            point["independent_extraction"]["internal"]["internal_phi"] > math.pi
            for point in evidence["point_evidence"]
        ))
        self.assertEqual(evidence["finite_difference_protocol"]["source_dimension"], 8)
        self.assertEqual(evidence["finite_difference_protocol"]["target_dimension"], 8)
        for point in evidence["point_evidence"]:
            self.assertEqual(len(point["jacobians"]), 2)
            self.assertTrue(all(len(row) == 8 for row in point["jacobians"][0]["matrix"]))
            self.assertEqual(len(point["sine_weighted_density_comparison"]), 2)
            self.assertEqual({row["kind"] for row in point["mirrors_and_branches"]}, {"source", "p1p2_plane_reflection"})
            self.assertLess(point["mirror_coordinates"]["phi_sum_residual"], 1.0e-8)
        json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_eliminated_phi_summed_density_checks_both_internal_roots(self):
        direct, bridge, chart, _ = _paths(2, density_mode="summed", phi_eliminated=True)

        report = compare_low_loop_paths(direct, bridge, chart_data=chart, masses=MASSES, total_mass=TOTAL_MASS)

        self.assertEqual(report.status, "verified_local")
        self.assertEqual(report.branch_status, "both_internal_phi_roots_and_mirrors_checked")
        self.assertEqual(report.evidence["density_divisor_for_summed_mode"], 2.0)
        for point in report.evidence["point_evidence"]:
            self.assertEqual({row["branch_argument"] for row in point["mirrors_and_branches"]}, {-1, 1})
            self.assertEqual(
                {row["branch_from_sin_internal_phi"] for row in point["mirrors_and_branches"]},
                {-1, 1},
            )
            self.assertAlmostEqual(
                point["density_direct_per_branch"] * 2.0,
                point["density_direct_total"],
            )

    def test_wrong_density_coefficient_fails(self):
        direct, bridge, chart, _ = _paths(0, phi_eliminated=False)
        exact = direct.numeric_density
        direct.numeric_density = lambda values, masses, total_mass, branch=1: 1.2 * exact(values, masses, total_mass, branch=branch)

        report = compare_low_loop_paths(direct, bridge, chart_data=chart, masses=MASSES, total_mass=TOTAL_MASS)

        self.assertEqual(report.status, "failed")
        self.assertTrue(any("density residual" in row["error"] for row in report.failures))

    def test_wrong_internal_phi_branch_fails_momentum_and_mirror_checks(self):
        direct, bridge, chart, _ = _paths(2, phi_eliminated=True)
        exact = direct.numeric_momenta
        direct.numeric_momenta = lambda values, masses, total_mass, branch=1: exact(
            values, masses, total_mass, branch=-branch,
        )

        report = compare_low_loop_paths(direct, bridge, chart_data=chart, masses=MASSES, total_mass=TOTAL_MASS)

        self.assertEqual(report.status, "failed")
        self.assertTrue(report.failures)
        self.assertTrue(any("momentum mismatch" in row["error"] for row in report.failures))

    def test_same_invalid_momenta_from_both_paths_do_not_pass(self):
        direct, bridge, chart, labels = _paths(0, phi_eliminated=False)
        reference_momenta = bridge.numeric_momenta
        by_target = {}

        def invariant_key(momenta):
            from phase_space_graph.formula import d5_benchmark as helper
            return tuple(helper._invariants_from_momenta(momenta, labels)[name] for name in labels)

        def invalid_reference(values, masses, total_mass):
            momenta = reference_momenta(values, masses, total_mass)
            invalid = tuple((row[0] + 0.1, *row[1:]) for row in momenta)
            by_target[invariant_key(invalid)] = invalid
            return invalid

        def common_invalid(values, masses, total_mass, branch=1):
            return by_target[tuple(float(values[label]) for label in labels)]

        bridge.numeric_momenta = invalid_reference
        direct.numeric_momenta = common_invalid
        direct.numeric_density = lambda values, masses, total_mass, branch=1: 1.0

        report = compare_low_loop_paths(direct, bridge, chart_data=chart, masses=MASSES, total_mass=TOTAL_MASS)

        self.assertEqual(report.status, "failed")
        self.assertTrue(any("energy conservation" in row["error"] for row in report.failures))

    def test_empty_short_duplicate_and_nonfinite_samples_are_rejected(self):
        direct, bridge, chart, _ = _paths(0, phi_eliminated=False)
        empty = compare_low_loop_paths(direct, bridge, chart_data=chart, masses=MASSES, total_mass=TOTAL_MASS, points=())
        self.assertEqual(empty.status, "unsupported")

        from phase_space_graph.formula.benchmark import _default_points
        sample = _default_points(MASSES, TOTAL_MASS)
        short = compare_low_loop_paths(direct, bridge, chart_data=chart, masses=MASSES, total_mass=TOTAL_MASS, points=sample[:7])
        self.assertEqual(short.status, "unsupported")
        duplicate = compare_low_loop_paths(direct, bridge, chart_data=chart, masses=MASSES, total_mass=TOTAL_MASS, points=sample[:7] + (sample[0], sample[1]))
        self.assertEqual(duplicate.status, "unsupported")
        nonfinite = dict(sample[0], phi12=math.nan)
        rejected = compare_low_loop_paths(direct, bridge, chart_data=chart, masses=MASSES, total_mass=TOTAL_MASS, points=(nonfinite,) + sample[1:])
        self.assertEqual(rejected.status, "failed")


if __name__ == "__main__":
    unittest.main()
