"""Independent reference-point and export-evidence checks for Yu2022 Tables 5/7."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from functools import lru_cache
from types import SimpleNamespace

from phase_space_graph.formula.benchmark import _default_points
from phase_space_graph.formula.d5_benchmark import _momenta
from phase_space_graph.formula.records import FormulaOptions
from phase_space_graph.formula.reference import build_bridge_formula, build_reference_chart
from phase_space_graph.formula.yu2022_check import (
    FD_STEPS,
    _boost,
    check_yu2022_reference,
    compare_all_loop_export_to_yu2022,
    source_discrepancies,
    table5_coordinates,
    table5_density,
    table7_coordinates,
    table7_density,
)

from test_tree_covariant import _d5_card


MASSES = (1.0, 1.5, 0.8, 1.2)
TOTAL_MASS = 6.0


@lru_cache(maxsize=2)
def _bridge(normalization: str = "standard_2pi4"):
    card = _d5_card(normalization)
    options = FormulaOptions(normalization=normalization)
    return build_bridge_formula(card, build_reference_chart(card, options=options), options=options)


class Yu2022CheckTests(unittest.TestCase):
    def test_explicit_audit_lorentz_boost_preserves_mass_and_inverts(self):
        vector = (2.4, 0.3, -0.7, 1.1)
        beta = (0.21, -0.16, 0.08)
        boosted = _boost(vector, beta)
        restored = _boost(boosted, tuple(-value for value in beta))

        boosted_mass2 = boosted[0] ** 2 - sum(value**2 for value in boosted[1:])
        original_mass2 = vector[0] ** 2 - sum(value**2 for value in vector[1:])
        self.assertAlmostEqual(boosted_mass2, original_mass2, places=12)
        for actual, expected in zip(restored, vector):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_table_coordinate_definitions_and_density_normalizations(self):
        bridge = _bridge()
        point = _default_points(MASSES, TOTAL_MASS)[0]
        momenta = _momenta(bridge.numeric_momenta(point, MASSES, TOTAL_MASS), "reference event")
        values5, output5, evidence5 = table5_coordinates(momenta, TOTAL_MASS)
        values7, output7, evidence7 = table7_coordinates(momenta, MASSES, TOTAL_MASS)

        self.assertEqual(set(values5), {"s12", "s13", "s14", "s23", "s124", "theta", "phi", "psi"})
        self.assertEqual(set(values7), {"s12", "s13", "s24", "s34", "c1star", "theta", "phi", "psi"})
        self.assertEqual(len(output5), 8)
        self.assertEqual(len(output7), 8)
        self.assertGreater(evidence5["orientation"]["normalized_triple_product"] ** 2, 0.0)
        self.assertAlmostEqual(
            evidence7["external_kinematics"]["k1star_from_boost"],
            evidence7["external_kinematics"]["k1star_from_kallen"],
        )
        rho5_jing = table5_density(momenta, MASSES, TOTAL_MASS, normalization="jing_psf")["rho_per_mirror"]
        rho5_standard = table5_density(momenta, MASSES, TOTAL_MASS, normalization="standard_2pi4")["rho_per_mirror"]
        rho7_jing = table7_density(momenta, MASSES, TOTAL_MASS, normalization="jing_psf")["rho_per_mirror"]
        rho7_standard = table7_density(momenta, MASSES, TOTAL_MASS, normalization="standard_2pi4")["rho_per_mirror"]
        self.assertAlmostEqual(rho5_standard / rho5_jing, (2.0 * math.pi) ** 4)
        self.assertAlmostEqual(rho7_standard / rho7_jing, (2.0 * math.pi) ** 4)

    def test_reference_report_keeps_two_fd_levels_mirrors_and_source_discrepancies(self):
        report = check_yu2022_reference(
            _bridge(), masses=MASSES, total_mass=TOTAL_MASS, normalization="standard_2pi4",
        )

        self.assertEqual(report["status"], "verified_local")
        self.assertEqual(report["coverage"], "density-coefficient-and-coordinate-pullback only")
        self.assertEqual(report["whole_table"], "pending/source_discrepancy")
        self.assertEqual(report["finite_difference_steps"], list(FD_STEPS))
        self.assertEqual(report["source"]["citation"], "Yu et al. (2022)")
        self.assertEqual(report["source"]["tables"]["table5"]["pdf_page"], 8)
        self.assertEqual(report["source"]["tables"]["table5"]["journal_page"], "083101-7")
        self.assertEqual(report["source"]["tables"]["table7"]["pdf_page"], 9)
        self.assertEqual(report["source"]["tables"]["table7"]["journal_page"], "083101-8")
        self.assertEqual(report["tables"]["table5"]["source"], report["source"]["tables"]["table5"])
        self.assertEqual(report["tables"]["table7"]["source"], report["source"]["tables"]["table7"])
        self.assertEqual(report["tables"]["table5"]["dimensionful_jacobian_factor"], "M^6")
        self.assertEqual(report["tables"]["table7"]["dimensionful_jacobian_factor"], "M^4")
        self.assertEqual(len(report["point_evidence"]), 8)
        for point in report["point_evidence"]:
            self.assertEqual(set(point["mirrors"]), {"source", "p1p2_plane_reflection"})
            for table in ("table5", "table7"):
                self.assertEqual(set(point["tables"][table]["mirrors"]), {"source", "p1p2_plane_reflection"})
                source_mirror = point["tables"][table]["mirrors"]["source"]
                reflected_mirror = point["tables"][table]["mirrors"]["p1p2_plane_reflection"]
                self.assertAlmostEqual(
                    source_mirror["external_density_data"]["rho_per_mirror"],
                    reflected_mirror["external_density_data"]["rho_per_mirror"],
                )
                for name in source_mirror["coordinates"]:
                    self.assertAlmostEqual(source_mirror["coordinates"][name], reflected_mirror["coordinates"][name])
                for mirror in point["tables"][table]["mirrors"].values():
                    self.assertEqual(len(mirror["jacobians"]), 2)
                    self.assertEqual([item["step"] for item in mirror["jacobians"]], list(FD_STEPS))
                    self.assertTrue(all(len(item["matrix"]) == 8 for item in mirror["jacobians"]))
                    self.assertEqual(len(mirror["density_pullback_comparisons"]), 2)
        discrepancy5, discrepancy7 = source_discrepancies(MASSES)
        self.assertEqual(discrepancy5["source"]["pdf_page"], 8)
        self.assertEqual(discrepancy7["source"]["pdf_page"], 9)
        self.assertAlmostEqual(discrepancy5["absolute_numerator_residual"], 2.0 * MASSES[2] ** 2)
        self.assertAlmostEqual(discrepancy7["absolute_numerator_residual"], 2.0 * sum(value**2 for value in MASSES))
        json.dumps(report, ensure_ascii=False, allow_nan=False)

    def test_fixed_reference_rejects_empty_duplicate_nan_and_wrong_density(self):
        bridge = _bridge()
        defaults = _default_points(MASSES, TOTAL_MASS)
        empty = check_yu2022_reference(bridge, masses=MASSES, total_mass=TOTAL_MASS, points=())
        self.assertEqual(empty["status"], "unsupported")
        duplicate = check_yu2022_reference(
            bridge, masses=MASSES, total_mass=TOTAL_MASS,
            points=defaults[:7] + (defaults[0], defaults[1]),
        )
        self.assertEqual(duplicate["status"], "unsupported")
        nan_point = dict(defaults[0], phi0=math.nan)
        invalid = check_yu2022_reference(
            bridge, masses=MASSES, total_mass=TOTAL_MASS, points=(nan_point,) + defaults[1:],
        )
        self.assertEqual(invalid["status"], "failed")

        exact_density = bridge.numeric_density
        wrong = SimpleNamespace(
            profile=bridge.profile, variables=bridge.variables, status=bridge.status,
            normalization=bridge.normalization, numeric_momenta=bridge.numeric_momenta,
            numeric_density=lambda values, masses, total_mass: 1.1 * exact_density(values, masses, total_mass),
        )
        changed = check_yu2022_reference(wrong, masses=MASSES, total_mass=TOTAL_MASS)
        self.assertEqual(changed["status"], "failed")

    def test_all_loop_export_pullback_preserves_ids_and_checks_4576_scale(self):
        paper = check_yu2022_reference(_bridge(), masses=MASSES, total_mass=TOTAL_MASS)
        evidence_rows = []
        for point in paper["point_evidence"]:
            source_table5 = point["tables"]["table5"]["mirrors"]["source"]
            direct_rows = [
                {"step": item["step"], "lhs": item["external_pullback_density"],
                 "rhs": point["reference_sine_weighted_density"]}
                for item in source_table5["density_pullback_comparisons"]
            ]
            evidence_rows.append({
                "reference_coordinates": point["reference_coordinates"],
                "target_coordinates": source_table5["coordinates"],
                "sine_weighted_density_comparison": direct_rows,
            })
        index = {"schema_version": "test", "classes": [{
            "class_id": "L0-C001", "loop_count": 0, "topology_id": "L0-T01",
            "representative_fixed_digest": "fixed-digest", "evidence": "class.evidence.json",
            "benchmark_status": "verified_local", "source_status": "not_checked",
        }]}
        payload = {
            "class_id": "L0-C001",
            "loop_count": 0,
            "topology_id": "L0-T01",
            "representative_fixed_digest": "fixed-digest",
            "normalization": paper["normalization"],
            "benchmark": {
                "status": "verified_local",
                "evidence": {"masses": MASSES, "total_mass": TOTAL_MASS, "point_evidence": evidence_rows},
            },
        }

        def summarize(index_data, evidence_data, *, paper_data=paper, tolerance=None):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "index.json").write_text(json.dumps(index_data), encoding="utf-8")
                (root / "class.evidence.json").write_text(json.dumps(evidence_data), encoding="utf-8")
                return compare_all_loop_export_to_yu2022(root, paper_data, tolerance=tolerance)

        summary = summarize(index, payload)

        self.assertEqual(summary["status"], "partial_or_pending")
        self.assertEqual(summary["expected_pullback_comparisons_per_class"], 32)
        self.assertEqual(summary["expected_pullback_comparisons_all_143"], 4576)
        self.assertEqual(summary["class_results"][0]["class_id"], "L0-C001")
        self.assertEqual(summary["class_results"][0]["source_status"], "not_checked")
        self.assertEqual(summary["class_results"][0]["compared_pullbacks"], 32)
        self.assertFalse(summary["all_143_present"])
        self.assertFalse(summary["all_expected_class_ids_present"])

        repeated = json.loads(json.dumps(payload))
        repeated["benchmark"]["evidence"]["point_evidence"] = [evidence_rows[0]] * 8
        repeated_result = summarize(index, repeated)
        self.assertEqual(repeated_result["class_results"][0]["status"], "failed_or_incomplete")
        self.assertTrue(any("repeats a reference-coordinate key" in error
                            for error in repeated_result["class_results"][0]["errors"]))

        zero_density = json.loads(json.dumps(payload))
        zero_density["benchmark"]["evidence"]["point_evidence"][0]["sine_weighted_density_comparison"][0]["lhs"] = 0.0
        zero_result = summarize(index, zero_density)
        self.assertTrue(any("density must be positive and finite" in error
                            for error in zero_result["class_results"][0]["errors"]))

        identity_mismatch = json.loads(json.dumps(payload))
        identity_mismatch["representative_fixed_digest"] = "tampered-digest"
        identity_result = summarize(index, identity_mismatch)
        self.assertTrue(any("representative_fixed_digest mismatch" in error
                            for error in identity_result["class_results"][0]["errors"]))

        missing_normalization = json.loads(json.dumps(payload))
        missing_normalization.pop("normalization")
        normalization_result = summarize(index, missing_normalization)
        self.assertTrue(any("normalization is missing" in error
                            for error in normalization_result["class_results"][0]["errors"]))

        failed_paper = json.loads(json.dumps(paper))
        failed_paper["status"] = "failed"
        self.assertEqual(summarize(index, payload, paper_data=failed_paper)["status"], "failed")
        self.assertEqual(summarize(index, payload, tolerance=0.0)["status"], "failed")

        unknown_index = json.loads(json.dumps(index))
        unknown_index["classes"][0].update(class_id="L9-C001", loop_count=9, topology_id="L9-T01")
        unknown_evidence = json.loads(json.dumps(payload))
        unknown_evidence.update(class_id="L9-C001", loop_count=9, topology_id="L9-T01")
        unknown_result = summarize(unknown_index, unknown_evidence)
        self.assertFalse(unknown_result["all_expected_class_ids_present"])
        self.assertTrue(any("outside the required" in error
                            for error in unknown_result["class_results"][0]["errors"]))


if __name__ == "__main__":
    unittest.main()
