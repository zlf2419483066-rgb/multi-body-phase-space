"""Public complete-formula routing and low-loop benchmark evidence checks."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from phase_space_graph import (
    FormulaOptions,
    build_all_loop_catalog,
    build_formula_input,
    generate_complete_formula,
    generate_complete_formulas,
    validate_artifact,
)
from phase_space_graph.formula.benchmark import compare_paths
from phase_space_graph.formula.records import BenchmarkReport


class CompleteFormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_all_loop_catalog(loops=(0, 1, 2, 3))

    def _class_for_loop(self, loop):
        return tuple(item for item in self.catalog.classes if item.loop_count == loop)

    def test_complete_entry_routes_tree_nested_d3_d4_and_d5(self):
        tree_and_nested = self._class_for_loop(0)
        self.assertEqual(len(tree_and_nested), 2)
        cards = tuple(
            build_formula_input(item.representative_record, measure_convention="jing_psf")
            for item in tree_and_nested
        )
        artifacts = generate_complete_formulas(cards, run_benchmark=False)
        self.assertEqual(len(artifacts), 2)
        self.assertTrue(all(item.direct_formula.profile == "direct_gram_low_loop" for item in artifacts))
        self.assertEqual(len({item.topology_id for item in tree_and_nested}), 2)

        for loop, expected_profile in ((1, "direct_gram_low_loop"),
                                       (2, "direct_gram_low_loop"),
                                       (3, "direct_gram_d5")):
            marked_class = self._class_for_loop(loop)[0]
            card = build_formula_input(marked_class.representative_record, measure_convention="jing_psf")
            artifact = generate_complete_formula(card, run_benchmark=False)
            self.assertEqual(artifact.direct_formula.profile, expected_profile)
            self.assertEqual(artifact.statuses["covariant"], "built")
            self.assertEqual(artifact.covariant_result.status, "built")
            self.assertEqual(artifact.statuses["bridge"], "built")
            self.assertEqual(len(artifact.direct_formula.target_variables), 2 + loop)

    def test_invalid_card_and_conflicting_normalization_are_rejected(self):
        marked_class = self._class_for_loop(0)[0]
        card = build_formula_input(marked_class.representative_record, measure_convention="jing_psf")
        with self.assertRaises(ValueError):
            generate_complete_formula(
                card,
                measure_convention="standard_2pi4",
                options=FormulaOptions(normalization="jing_psf"),
                run_benchmark=False,
            )
        with self.assertRaises(ValueError):
            generate_complete_formula(replace(card, n=5), run_benchmark=False)

    def test_low_loop_benchmark_dispatch_forwards_points_and_tolerances(self):
        marked_class = self._class_for_loop(1)[0]
        card = build_formula_input(marked_class.representative_record, measure_convention="jing_psf")
        artifact = generate_complete_formula(card, run_benchmark=False)
        direct = artifact.direct_formula
        bridge = artifact.bridge_formula
        point = {"s12": 9.0, "s34": 10.0}
        points = (point,)
        chart = {"status": "built", "profile": "direct_gram_low_loop"}
        expected = object()
        with patch(
            "phase_space_graph.formula.low_loop_direct.build_low_loop_chart_data_from_labels",
            return_value=chart,
        ) as build_chart, patch(
            "phase_space_graph.formula.low_loop_benchmark.compare_low_loop_paths",
            return_value=expected,
        ) as compare_low_loop:
            result = compare_paths(
                direct,
                bridge,
                masses=(0.7, 1.0, 0.8, 1.1),
                total_mass=8.0,
                points=points,
                relative_tolerance=2.0e-8,
                fd_relative_tolerance=3.0e-6,
                kinematic_tolerance=4.0e-9,
            )

        self.assertIs(result, expected)
        build_chart.assert_called_once_with(tuple(direct.target_variables))
        kwargs = compare_low_loop.call_args.kwargs
        self.assertEqual(kwargs["chart_data"], chart)
        self.assertEqual(kwargs["masses"], (0.7, 1.0, 0.8, 1.1))
        self.assertEqual(kwargs["total_mass"], 8.0)
        self.assertIs(kwargs["points"], points)
        self.assertEqual(kwargs["relative_tolerance"], 2.0e-8)
        self.assertEqual(kwargs["fd_relative_tolerance"], 3.0e-6)
        self.assertEqual(kwargs["kinematic_tolerance"], 4.0e-9)

    def test_low_loop_point_evidence_is_required_for_verified_local_status(self):
        marked_class = self._class_for_loop(1)[0]
        card = build_formula_input(marked_class.representative_record, measure_convention="jing_psf")
        artifact = generate_complete_formula(card, run_benchmark=False)
        replay_points = tuple({"point": index} for index in range(8))
        evidence = {
            "target_profile": "direct_gram_low_loop",
            "replay_points": replay_points,
            "point_evidence": tuple({"point": index, "errors": []} for index in range(8)),
        }
        report = BenchmarkReport(
            status="verified_local",
            profile="sequential_12_34",
            points=8,
            compared=8,
            max_absolute_residual=0.0,
            max_relative_residual=0.0,
            coordinate_residual=0.0,
            branch_status="explicit_internal_phi_mirror_checked",
            evidence=evidence,
        )
        complete = replace(
            artifact,
            internal_benchmark=report,
            statuses=dict(artifact.statuses, benchmark="verified_local"),
        )
        self.assertFalse(any(item.status == "failed" for item in validate_artifact(complete)))

        damaged_report = replace(report, evidence={**evidence, "point_evidence": ()})
        damaged = replace(
            complete,
            internal_benchmark=damaged_report,
            statuses=dict(complete.statuses, benchmark="verified_local"),
        )
        failed_checks = {item.check for item in validate_artifact(damaged) if item.status == "failed"}
        self.assertIn("benchmark-evidence", failed_checks)


if __name__ == "__main__":
    unittest.main()
