"""Coverage for six-step all-loop formula export and non-overwrite behavior."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phase_space_graph.formula.all_loop_export as all_loop_export
from phase_space_graph.formula.all_loop_catalog import build_all_loop_catalog
from phase_space_graph.formula_inputs import build_formula_input


class AllLoopFormulaExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This reads frozen L0-L2 G08 witnesses and canonicalizes them; it does not enumerate.
        cls.catalog = build_all_loop_catalog(loops=(0, 1, 2))

    def test_one_l0_class_exports_six_steps_and_actual_low_loop_evidence(self):
        marked_class = self.catalog.classes[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "export"
            with patch.object(all_loop_export, "build_all_loop_catalog", return_value=self.catalog):
                result = all_loop_export.export_all_four_body_formulas(
                    class_selector=marked_class.class_id,
                    loops=(0,),
                    output_dir=output,
                )

            self.assertEqual(result["selected_class_count"], 1)
            self.assertEqual(result["markdown_count"], 1)
            self.assertEqual(result["failed_class_count"], 0)
            stem = f"{marked_class.class_id}_{marked_class.topology_id}"
            markdown = (output / f"{stem}.md").read_text(encoding="utf-8")
            evidence = json.loads((output / f"{stem}.evidence.json").read_text(encoding="utf-8"))
            for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
                self.assertIn(heading, markdown)
            self.assertIn("显式 $h(S)$", markdown)
            self.assertIn("显式 $t(S)$", markdown)
            self.assertIn(r"\det K", markdown)
            self.assertIn(r"J_x", markdown)
            self.assertIn(r"v_0", markdown)
            self.assertIn(r"J_{\rm pull}", markdown)
            self.assertIn(r"\begin{pmatrix}", markdown)
            self.assertIn(r"\varphi", markdown)
            self.assertIn(r"\det H>0", markdown)
            self.assertNotIn("sqrt(", markdown)
            self.assertIn(r"\sqrt", markdown)
            self.assertIn(r"\rho_{\rm low}=", markdown)
            self.assertIn("对应 $p=12$", markdown)
            self.assertIn(r"(2\pi)^{12}", markdown)
            self.assertIn("输入测度约定：`jing_psf`", markdown)
            for constraint in evidence["tree_covariant_trace"]["remaining_constraints_latex"]:
                self.assertIn(constraint, markdown)
            self.assertEqual(evidence["source_status"], "not_checked")
            self.assertEqual(evidence["benchmark"]["status"], "pending")
            self.assertEqual(evidence["tree_covariant_trace"]["scope"]["d"], 2)
            self.assertEqual(evidence["identity_class"]["class_id"], marked_class.class_id)

            index = json.loads((output / "index.json").read_text(encoding="utf-8"))
            coverage = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
            self.assertEqual(index["source_status"], "not_checked")
            self.assertEqual(index["benchmark_status"], "pending")
            self.assertEqual(coverage["by_loop"]["0"]["exported_classes"], 1)

    def test_serialized_marker_graph_reads_both_measure_conventions_without_unknown_fallback(self):
        marked_class = self.catalog.classes[0]
        for convention in ("jing_psf", "standard_2pi4"):
            with self.subTest(convention=convention):
                # Ordinary formula-input serialization carries the convention in status.
                card = build_formula_input(
                    marked_class.representative_record,
                    measure_convention=convention,
                )
                serialized = card.to_dict()
                self.assertEqual(serialized["status"]["measure_convention"], convention)
                rendered = "\n".join(all_loop_export._render_marker_graph(serialized))
                self.assertIn(f"输入测度约定：`{convention}`", rendered)

                # Some serialized artifacts also carry this value under provenance.
                provenance_card = build_formula_input(
                    marked_class.representative_record,
                    measure_convention=convention,
                    provenance={"measure_convention": convention},
                ).to_dict()
                self.assertEqual(provenance_card["provenance"]["measure_convention"], convention)
                rendered = "\n".join(all_loop_export._render_marker_graph(provenance_card))
                self.assertIn(f"输入测度约定：`{convention}`", rendered)

        with self.assertRaisesRegex(ValueError, "recognized measure convention"):
            all_loop_export._render_marker_graph({"record": {}, "variable_labels": []})

    def test_l1_l2_render_actual_constraints_roots_or_phi_inputs(self):
        for loop in (1, 2):
            with self.subTest(loop=loop):
                marked_class = next(item for item in self.catalog.classes if item.loop_count == loop)
                evidence = all_loop_export.build_all_loop_formula_evidence(marked_class)
                markdown = all_loop_export._render_markdown(marked_class, evidence)
                chart = evidence["chart_data"]
                trace = evidence["tree_covariant_trace"]

                self.assertEqual(trace["scope"]["d"], 2 + loop)
                self.assertEqual(trace["scope"]["vertex_count"], 3 + loop)
                self.assertEqual(len(trace["remaining_constraints_latex"]), 3 + loop)
                self.assertEqual(len(chart["constraint_equations"]), loop)
                self.assertIn(r"C=\begin{pmatrix}", markdown)
                self.assertIn(r"C_E=\begin{pmatrix}", markdown)
                step_five = "\n".join(all_loop_export._render_low_loop_step_five(chart))
                self.assertIn(r"\chi_{\rm open}=\mathbf 1_{\mathcal D}", step_five)
                self.assertIn(r"\Theta_+(u)=\mathbf 1_{u>0}", step_five)
                self.assertIn(r"B_{(a,b)}(u)=\Theta_+(u-a)\Theta_+(b-u)", step_five)
                self.assertIn(r"B_{(-1,1)}(x)B_{(-1,1)}(y)", step_five)
                for row in chart["constraint_equations"]:
                    self.assertIn(row["C_equation_latex"], markdown)
                    self.assertIn(row["u0_latex"], markdown)
                if chart["elimination"]["phi_eliminated"]:
                    self.assertIn(chart["elimination"]["phi_roots_latex"], markdown)
                    self.assertIn(r"B_{(-1,1)}(c_\varphi)", step_five)
                else:
                    self.assertIn(chart["elimination"]["retained_phi_input_latex"], markdown)
                    self.assertIn(r"B_{(0,2\pi)}(\varphi)", step_five)
                self.assertIn(r"J_{\rm pull}", markdown)
                self.assertIn(r"\sqrt", markdown)
                self.assertNotIn("sqrt(", markdown)

    def test_standard_normalization_uses_its_fixed_power_in_rendered_jacobians(self):
        marked_class = self.catalog.classes[0]
        evidence = all_loop_export.build_all_loop_formula_evidence(
            marked_class,
            normalization="standard_2pi4",
        )
        step_six = "\n".join(all_loop_export._render_low_loop_step_six(
            evidence["chart_data"], evidence["direct_formula"], "standard_2pi4"
        ))
        self.assertIn("对应 $p=8$", step_six)
        self.assertIn(r"(2\pi)^{8}", step_six)
        self.assertNotIn(r"(2\pi)^p", step_six)
        self.assertIn(r"\rho_{\rm low}=", step_six)

    def test_pair_and_nested_local_frames_thresholds_and_canonical_body_are_explicit(self):
        for marked_class in (item for item in self.catalog.classes if item.loop_count == 0):
            evidence = all_loop_export.build_all_loop_formula_evidence(marked_class)
            chart = evidence["chart_data"]
            step_three = "\n".join(all_loop_export._render_low_loop_step_three(chart))
            step_five = "\n".join(all_loop_export._render_low_loop_step_five(chart))
            mode = chart["geometry"]["mode"]
            self.assertIn("m_a,m_b,m_c,m_d", step_three)
            self.assertIn(r"\mathbf e_z", step_five)
            self.assertIn(r"\mathbf e_x", step_five)
            self.assertIn(r"\mathbf e_y", step_five)
            self.assertIn("R_{zyz}", step_five)
            self.assertIn(r"\lambda(x,y,z)=", step_five)
            self.assertIn(r"J_{\rm rot}=1", step_five)
            if mode == "pair":
                self.assertIn("q_{ab}\\parallel +\\hat z", step_three)
                self.assertIn("q_{cd}\\parallel-\\hat z", step_three)
                self.assertIn("H>m_a+m_b", step_five)
                self.assertIn("T>m_c+m_d", step_five)
                self.assertIn("M>H+T", step_five)
            else:
                self.assertIn("p_c\\parallel +\\hat z", step_three)
                self.assertIn("q_{ab}\\parallel-\\hat z", step_three)
                self.assertIn("H>m_a+m_b", step_five)
                self.assertIn("T>H+m_c", step_five)
                self.assertIn(r"\beta=-\frac{\mathbf p_d^{(abc)}}{T+E_d}", step_five)
                self.assertIn(r"E'=\gamma(E+\boldsymbol\beta\!\cdot\mathbf p)", step_five)
                self.assertIn(r"\boldsymbol\beta=0\Rightarrow\operatorname{Boost}_{\boldsymbol\beta}=I", step_five)
                self.assertIn("M>T+m_d", step_five)

    def test_multi_particle_invariant_names_are_subscripted_in_s_and_b_vectors(self):
        marked_class = next(
            item for item in self.catalog.classes
            if item.loop_count == 2
            if (1, 2, 3) in item.representative_record.independence.variables
        )
        evidence = all_loop_export.build_all_loop_formula_evidence(marked_class)
        rendered = "\n".join(all_loop_export._render_low_loop_step_three(evidence["chart_data"]))
        self.assertIn("s_{123}", rendered)
        self.assertNotRegex(rendered, r"\bs123\b")

    def test_low_loop_markdown_math_delimiters_and_u0_rows_for_all_37_classes(self):
        for marked_class in self.catalog.classes:
            with self.subTest(class_id=marked_class.class_id):
                evidence = all_loop_export.build_all_loop_formula_evidence(marked_class)
                markdown = all_loop_export._render_markdown(marked_class, evidence)
                lines = markdown.splitlines()
                display_delimiters = [line for line in lines if line.strip() == "$$"]
                self.assertEqual(len(display_delimiters) % 2, 0)
                for line in lines:
                    if "$$" in line:
                        self.assertEqual(line.strip(), "$$")

                chart = evidence["chart_data"]
                u0_lines = [line for line in lines if line.startswith("- $u_{0,")]
                self.assertEqual(len(u0_lines), len(chart["constraint_equations"]))
                for line in u0_lines:
                    self.assertEqual(line.count("$"), 2)
                    self.assertTrue(line.endswith("$。"))

    def test_requested_benchmark_report_is_preserved_without_upgrading_pending(self):
        marked_class = self.catalog.classes[0]

        class PendingReport:
            def to_dict(self):
                return {"status": "pending", "diagnostics": ["fixture has no comparison points"]}

        with patch(
            "phase_space_graph.formula.benchmark.compare_paths",
            return_value=PendingReport(),
        ) as compare_paths:
            evidence = all_loop_export.build_all_loop_formula_evidence(
                marked_class,
                benchmark_masses=(1.0, 1.5, 0.8, 1.2),
                benchmark_total_mass=6.0,
            )

        self.assertEqual(evidence["benchmark"]["status"], "pending")
        self.assertIn("fixture has no comparison points", evidence["benchmark"]["diagnostics"])
        compare_paths.assert_called_once()

    def test_l3_steps_three_through_six_delegate_to_existing_d5_renderers(self):
        with patch.object(all_loop_export, "_render_step_three", return_value=["## 3. D5"]) as step_three, \
             patch.object(all_loop_export, "_render_step_four", return_value=["## 4. D5"]) as step_four, \
             patch.object(all_loop_export, "_render_step_five", return_value=["## 5. D5"]) as step_five, \
             patch.object(all_loop_export, "_render_step_six", return_value=["## 6. D5"]) as step_six:
            rendered = all_loop_export._render_steps_three_to_six({"chart_data": {}}, loop_count=3)

        self.assertEqual(rendered, ["## 3. D5", "## 4. D5", "## 5. D5", "## 6. D5"])
        step_three.assert_called_once()
        step_four.assert_called_once()
        step_five.assert_called_once()
        step_six.assert_called_once()

    def test_loop_filter_and_failure_continue_are_reported(self):
        fake_evidence = {
            "formula_build_status": "built",
            "tree_covariant_trace": {"status": "built"},
            "chart_data": {"status": "built"},
            "benchmark": {"status": "pending"},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "export"
            with patch.object(all_loop_export, "build_all_loop_catalog", return_value=self.catalog), \
                 patch.object(
                     all_loop_export,
                     "build_all_loop_formula_evidence",
                     side_effect=[RuntimeError("fixture build failure"), fake_evidence],
                 ), \
                 patch.object(all_loop_export, "_render_markdown", return_value="# fixture\n"):
                result = all_loop_export.export_all_four_body_formulas(
                    all_classes=True,
                    loops=0,
                    output_dir=output,
                )

            self.assertEqual(result["selected_class_count"], 2)
            self.assertEqual(result["markdown_count"], 1)
            self.assertEqual(result["failed_class_count"], 1)
            coverage = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
            self.assertEqual(
                coverage["by_loop"]["0"]["selected_classes"],
                sum(item.loop_count == 0 for item in self.catalog.classes),
            )
            self.assertEqual(coverage["by_loop"]["0"]["failed_classes"], 1)
            failure = coverage["failures"][0]
            self.assertIn("fixture build failure", failure["error"])
            self.assertTrue((output / failure["failure_json"]).is_file())

    def test_output_collision_refuses_to_replace_existing_files(self):
        marked_class = self.catalog.classes[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            readme = output / "README.md"
            readme.write_text("preserve", encoding="utf-8")
            with patch.object(all_loop_export, "build_all_loop_catalog", return_value=self.catalog):
                with self.assertRaises(FileExistsError):
                    all_loop_export.export_all_four_body_formulas(
                        class_selector=marked_class.class_id,
                        loops=0,
                        output_dir=output,
                    )
            self.assertEqual(readme.read_text(encoding="utf-8"), "preserve")

    def test_selector_requirements_and_loop_validation(self):
        with self.assertRaisesRegex(ValueError, "select all_classes"):
            all_loop_export.export_all_four_body_formulas()
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            all_loop_export.export_all_four_body_formulas(all_classes=True, class_selector="L0-C001")
        with self.assertRaisesRegex(ValueError, "loop counts"):
            all_loop_export.export_all_four_body_formulas(loops=(0, 4))
        with self.assertRaisesRegex(ValueError, "requires benchmark_masses"):
            all_loop_export.export_all_four_body_formulas(all_classes=True, run_benchmark=True)
        with self.assertRaisesRegex(ValueError, "require run_benchmark"):
            all_loop_export.export_all_four_body_formulas(all_classes=True, benchmark_masses=(1, 1, 1, 1), benchmark_total_mass=6)


if __name__ == "__main__":
    unittest.main()
