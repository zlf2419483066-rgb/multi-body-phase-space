"""B/F03 dossier composition, serialization, and Markdown output tests."""

import json
from pathlib import Path
import tempfile
import unittest

from im_fixtures import fixture_b
from phase_space_graph import build_formula_input, canonicalize_physical_labeling
from phase_space_graph.formula import build_b_f03_dossier


def _card(convention="jing_psf"):
    record = fixture_b().record
    identity = canonicalize_physical_labeling(record)
    return build_formula_input(record, measure_convention=convention, identity=identity)


def _chart_parameters(normalization=None):
    parameters = {
        "total_mass": 5.0,
        "masses": (0.5, 0.6, 0.4, 0.7),
        "s12": 2.56,
        "s34": 2.89,
        "s23": 2.0,
        "theta0": 0.8,
        "phi0": 0.4,
        "theta1": 1.1,
        "phi1": 1.7,
        "phi4_star": 0.9,
    }
    if normalization is not None:
        parameters["normalization"] = normalization
    return parameters


class BF03DossierTests(unittest.TestCase):
    def test_dossier_composes_all_traces_and_renders_sources_and_limits(self):
        dossier = build_b_f03_dossier(_card())

        self.assertEqual(dossier.status, "assembled")
        self.assertEqual(dossier.covariant_certificate.status, "built")
        self.assertFalse(dossier.formula_verified)
        self.assertFalse(dossier.covariant_certificate.formula_verified)
        self.assertEqual(dossier.source_match_status, "not_checked")
        self.assertEqual(dossier.covariant_certificate.source_match_status, "not_checked")
        self.assertEqual(dossier.local_reduction.status, "partial")
        payload = dossier.to_dict()
        self.assertEqual(payload["covariant_certificate"]["status"], "built")
        self.assertFalse(payload["formula_verified"])
        json.loads(dossier.to_json())

        markdown = dossier.to_markdown()
        for phrase in (
            "原始局部表达式", "入射来源", "出射测度与来源", "出现级抵消计划",
            "末顶点外部出现 v3/slot0", "残余约束", "四个顶点守恒式",
            "delta⁴ 的唯一用途", "q12", "q23", "q34", "全局协变消元证书",
            "直接密度与 chart 定义域", "not_sampled", "formula_verified = false",
            "not_checked", "未验证事项", "overall Euler chart",
            "完整机器可读协变证书", "four_body_total_mass",
            "三整体角＋两内部角的 helicity 换图册", "u=cos(alpha)",
            "boxcar(c0; -1, 1)", "product_i theta(p_i^0)",
        ):
            self.assertIn(phrase, markdown)
        self.assertIn("delta(s23 - (p_2 + p_3)^2)", markdown)
        self.assertIn("source_reference", markdown)

    def test_optional_chart_sample_is_bound_to_card_normalization_and_written_exclusively(self):
        dossier = build_b_f03_dossier(_card(), direct_chart_parameters=_chart_parameters())

        self.assertIsNotNone(dossier.direct_sample)
        self.assertEqual(dossier.direct_sample.status, "physical")
        self.assertTrue(dossier.direct_sample.all_shells_checked)
        self.assertEqual(dossier.direct_sample.normalization, "jing_psf")
        markdown = dossier.to_markdown()
        self.assertIn("数值样本状态：`physical`", markdown)
        self.assertIn("rho_BF03", markdown)
        self.assertIn("root_cosine", markdown)

        standard = build_b_f03_dossier(
            _card("standard_2pi4"),
            direct_chart_parameters=_chart_parameters(),
        )
        self.assertEqual(standard.direct_sample.status, "physical")
        self.assertEqual(standard.direct_sample.normalization, "standard_2pi4")
        self.assertIn("(2*pi)^8", standard.to_markdown())

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "b-f03-dossier.md"
            written = dossier.write_markdown_new_file(path)
            self.assertEqual(written, path)
            self.assertEqual(path.read_text(encoding="utf-8"), markdown)
            with self.assertRaises(FileExistsError):
                dossier.write_markdown_new_file(path)

    def test_direct_sample_rejects_normalization_conflicting_with_card(self):
        with self.assertRaisesRegex(ValueError, "must match the formula card"):
            build_b_f03_dossier(
                _card("standard_2pi4"),
                direct_chart_parameters=_chart_parameters("jing_psf"),
            )


if __name__ == "__main__":
    unittest.main()
