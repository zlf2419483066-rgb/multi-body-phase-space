"""Coverage for the archived D5 class catalog and six-step exporter.

These tests only use the read-only identity archive; they do not enumerate
candidate graph labelings or run the batch export.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phase_space_graph.formula.d5_catalog as d5_catalog_module
import phase_space_graph.formula.d5_export as d5_export_module
from phase_space_graph.drawing.archive import load_identity_archive
from phase_space_graph.domain import CountMode
from phase_space_graph.formula.d5_catalog import (
    D5_EXPECTED_ARCHIVE_SHA256,
    D5_EXPECTED_CLASS_COUNT,
    D5_EXPECTED_FIXED_COUNT,
    D5_EXPECTED_TOPOLOGY_COUNT,
    build_d5_catalog,
    default_d5_archive_path,
)
from phase_space_graph.formula.d5_export import export_d5_formulas
from phase_space_graph.invariants import invariant_label
from phase_space_graph.label_identity import canonicalize_physical_labeling, relabel_physical_record


ARCHIVE = default_d5_archive_path()


def _mapping(rows, source_key, target_key):
    return {row[source_key]: row[target_key] for row in rows}


def _write_archive(path: Path, payload) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def _six_step_fixture(marked_class):
    """Small stable interface fixture for Markdown rendering tests."""
    from types import SimpleNamespace

    record = marked_class.representative.canonical_record.to_dict()
    card = SimpleNamespace(
        record=record,
        variable_labels=("s12", "s13", "s14", "s23", "s34"),
        measure_convention="jing_psf",
    )
    local_factors = [
        {
            "vertex": index,
            "expanded_expression_latex": rf"\delta^{{(4)}}(P_{{{index}}}-q_{{{index}}})",
            "conservation_equation_latex": rf"P_{{{index}}}=q_{{{index}}}",
            "incoming_momentum_latex": rf"P_{{{index}}}",
            "outgoing_integration_momenta_latex": [rf"q_{{{index}}}"],
            "outgoing_support_momenta_latex": [rf"q_{{{index}}}"],
            "momentum_delta_latex": rf"\delta^{{(4)}}(P_{{{index}}}-q_{{{index}}})",
            "normalization_2pi_power": 0,
        }
        for index in range(1, 7)
    ]
    trace = {
        "status": "built",
        "f0": {
            "factors": local_factors,
            "double_line_factors": [
                {"edge_index": index, "occurrence": 0} for index in range(5)
            ],
            "double_line_factors_latex": [rf"\mathrm{{d}}s_{{{index + 1}}}" for index in range(5)],
        },
        "normalization": {
            "normalization_identity_latex": r"[(2\pi)^3(2\pi)^{-3}]^5=1",
            "per_tree_integration_balance_latex": r"(2\pi)^3(2\pi)^{-3}=1",
            "final_external_measure_factor_latex": r"\prod_i d\mu(p_i)=(2\pi)^{-12}\prod_i d^4p_i\delta(p_i^2-m_i^2)\theta(p_i^0)",
        },
        "cancellation": {
            "policy": "D0 test policy",
            "cancelled_measures": [
                {"locator": {"edge_index": 0, "occurrence": 0}, "measure": {"particle_ids": [1]}}
            ],
            "retained_external_measures": [
                {"particle_ids": [1], "source_reference": {"vertex": 2, "slot": 0}}
            ],
        },
        "elimination": {
            "steps": [
                {
                    "leaf_vertex": index,
                    "parent_vertex": index + 1,
                    "delta4_source_latex": rf"\delta^{{(4)}}(q_{{{index}}}-p_{{{index}}})",
                    "current_equation_latex": rf"q_{{{index}}}-p_{{{index}}}=0",
                    "integrated_q": rf"q_{{{index}}}",
                    "substitution_latex": rf"q_{{{index}}}=p_{{{index}}}",
                    "abs_4d_jacobian_latex": "1",
                }
                for index in range(1, 6)
            ],
            "root_vertex": 1,
            "root_equation": "P-p_1-p_2-p_3-p_4=0",
            "root_equation_latex": r"P-p_1-p_2-p_3-p_4=0",
            "root_residual": "0",
            "root_residual_latex": "0",
            "remaining_q_variables": [],
        },
        "q_shell_transfers": [
            {
                "integrated_q": rf"q_{{{index}}}",
                "derived_solution_latex": rf"q_{{{index}}}=p_{{{index}}}",
                "global_support_solution_latex": rf"q_{{{index}}}=p_{{{index}}}",
                "shell_delta_latex": rf"\delta(s_{{{index}}}-m_{{{index}}}^2)",
                "positive_energy_transfer_latex": rf"\theta(p_{{{index}}}^0)",
                "ds_factor_in_final_expression_latex": rf"\mathrm{{d}}s_{{{index}}}",
            }
            for index in range(1, 6)
        ],
        "remaining_constraints_latex": [r"\delta^{(4)}(P-\sum_i p_i)"]
        + [rf"\delta(s_{{{index}}}-S_{{{index}}})" for index in range(1, 6)],
    }
    chart = {
        "profile": "direct_gram_d5",
        "status": "built",
        "ordered_s": ["s12", "s13", "s14", "s23", "s34"],
        "basis_variables": ["s234", "s13", "s24", "s12", "s34"],
        "mass_offset_latex": [rf"b_{{{index}}}=M^2-m_{{{index}}}^2" for index in range(1, 6)],
        "L_latex": r"L=\begin{pmatrix}1&0\\0&1\end{pmatrix}",
        "inverse_L_latex": r"L^{-1}=\begin{pmatrix}1&0\\0&1\end{pmatrix}",
        "det_L_latex": r"\det L=1",
        "abs_det_L_latex": r"|\det L|=1",
        "forward_equations_latex": [r"s_{12}=t", r"s_{13}=v"],
        "inverse_equations_latex": [r"t=s_{12}", r"v=s_{13}"],
        "energy_expressions_latex": [r"E_1=\cdots", r"E_2=\cdots"],
        "completed_pair_invariants_latex": {"s23": r"s_{23}=u-w-z+\cdots"},
        "gram_matrix_latex": [r"H_{ii}=E_i^2-m_i^2"],
        "angle_variables_latex": [r"\theta", r"\phi", r"\psi"],
        "angle_measure_latex": r"dR=\sin\theta\,d\theta\,d\phi\,d\psi",
        "rotation_latex": r"\mathbf{k}^{\rm COM}=R\mathbf{k}^{\rm body}",
        "coordinate_map": {
            "constraint_delta_change_latex": r"\delta^5[S-F_S]=|\det L|^{-1}\delta^5[X-F_X]"
        },
        "root": {
            "body_frame_momenta_latex": [r"\mathbf{k}_1=(0,0,k_1)"],
            "mirror_root_constraint_latex": r"f_y=\det H-H_{11}x_2^2k_{3y}^2=0",
            "mirror_root_absolute_derivative_latex": r"|\partial f_y/\partial k_{3y}|=2H_{11}x_2^2|k_{3y}|",
            "mirror_root_latex": [r"k_{3y}=+\sqrt{\det H}/(k_1x_2)", r"k_{3y}=-\sqrt{\det H}/(k_1x_2)"],
        },
        "scalar_elimination": [
            {
                "eliminated_variable": f"x{index}",
                "constraint_latex": rf"f_{{{index}}}=x_{{{index}}}-a_{{{index}}}=0",
                "root_latex": rf"x_{{{index}}}=a_{{{index}}}",
                "root_derivative_latex": rf"|\partial f_{{{index}}}/\partial x_{{{index}}}|=1",
            }
            for index in range(1, 7)
        ],
        "scalar_elimination_jacobian": {
            "total_jacobian_latex": r"J_{scalar}=64M^3(k_1k_2k_3)^2",
            "constraint_group_change_latex": r"f_2+f_4,\quad f_3+f_5",
            "constraint_group_determinants_latex": [r"|\det T_1|=1", r"|\det T_2|=1"],
            "angle_change_per_mirror_branch_latex": r"d\Omega_1d\Omega_2d\Omega_3=dR\,dc_{12}dc_{13}dc_{23}/\sqrt{\Delta}",
        },
        "branches": [
            {"label": "sigma=+1", "sigma": 1, "root_latex": r"k_{3y}=+a", "momentum_reconstruction_latex": [r"p_1=(E_1,Rk_1)"], "density_weight": "one branch"},
            {"label": "sigma=-1", "sigma": -1, "root_latex": r"k_{3y}=-a", "momentum_reconstruction_latex": [r"p_1=(E_1,Rk_1)"], "density_weight": "one branch"},
        ],
        "root_conditions_latex": [r"E_i>0\quad(i=1,2,3,4)", r"\det H>0"],
        "domain_latex": [r"m_i>0,\quad M>\sum_i m_i", r"0<\theta<\pi"],
        "chi_open_definition_latex": r"\chi_{open}=\prod_i\Theta_+(E_i)\Theta_+(\det H),\quad\Theta_+(x)=1_{x>0}",
        "density_expression_latex": r"\rho_{S,\sigma}=\chi_{open}/(512(2\pi)^{12}M^3\sqrt{\det H}|\det L|)",
        "differential_measure_latex": r"d\Phi_4=\rho_{S,\sigma}(S)d^5S\,dR",
        "normalization_derivation_latex": [r"J_{scalar}=64M^3(k_1k_2k_3)^2", r"d^5X=|\det L|^{-1}d^5S"],
        "normalization_expressions_latex": {
            "jing_psf_per_branch": r"\rho_{J,\sigma}=\chi_{open}/(512(2\pi)^{12}M^3\sqrt{\det H}|\det L|)",
            "standard_2pi4_per_branch": r"\rho_{S,\sigma}=\chi_{open}/(512(2\pi)^8M^3\sqrt{\det H}|\det L|)",
            "jing_psf_summed": r"\rho_{J,sum}=2\rho_{J,\sigma}",
            "standard_2pi4_summed": r"\rho_{S,sum}=2\rho_{S,\sigma}",
        },
    }
    formula_artifact = {
        "statuses": {"direct": "built", "source": "not_checked"},
        "internal_benchmark": {"status": "pending"},
    }
    return card, trace, chart, formula_artifact


class D5ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One archive load/grouping pass is shared by all tests in this class.
        cls.catalog = build_d5_catalog()

    def test_catalog_counts_group_sizes_topology_distribution_and_pinned_archive(self):
        self.assertEqual(self.catalog.fixed_record_count, D5_EXPECTED_FIXED_COUNT)
        self.assertEqual(self.catalog.topology_count, D5_EXPECTED_TOPOLOGY_COUNT)
        self.assertEqual(len(self.catalog.classes), D5_EXPECTED_CLASS_COUNT)
        self.assertEqual(self.catalog.archive_sha256, D5_EXPECTED_ARCHIVE_SHA256)
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), D5_EXPECTED_ARCHIVE_SHA256)
        self.assertEqual(Counter(len(item.members) for item in self.catalog.classes), {24: 106})
        topology_distribution = Counter(item.topology_id for item in self.catalog.classes)
        self.assertEqual(len(topology_distribution), D5_EXPECTED_TOPOLOGY_COUNT)
        self.assertEqual(sum(topology_distribution.values()), D5_EXPECTED_CLASS_COUNT)
        for topology_id, class_count in topology_distribution.items():
            member_count = sum(
                len(item.members) for item in self.catalog.classes if item.topology_id == topology_id
            )
            self.assertEqual(member_count, 24 * class_count, topology_id)

    def test_every_archived_map_replays_to_the_real_fixed_representative(self):
        for marked_class in self.catalog.classes:
            representative = marked_class.representative.canonical_record
            member_keys = {member.archived.key for member in marked_class.members}
            canonical_fixed = canonicalize_physical_labeling(
                marked_class.particle_identity.canonical_record, CountMode.FIXED_LABELS
            )
            if canonical_fixed.key in member_keys:
                self.assertEqual(marked_class.representative.key, canonical_fixed.key)
            else:
                first_archived = min(
                    (member.archived for member in marked_class.members),
                    key=lambda item: (item.program_id, item.digest),
                )
                self.assertEqual(marked_class.representative.key, first_archived.key)
            for member in marked_class.members:
                data = member.map_to_representative
                vertex_map = _mapping(data["vertex_map"], "source_vertex", "representative_vertex")
                particle_map = _mapping(data["particle_map"], "source_particle", "representative_particle")
                replay = relabel_physical_record(
                    member.archived.canonical_record, vertex_map, particle_map
                )
                self.assertEqual(
                    replay.record.to_dict(), representative.to_dict(),
                    f"{marked_class.class_id}/{member.archived.program_id}: replayed record differs",
                )
                self.assertEqual(dict(replay.vertex_map), vertex_map)
                self.assertEqual(dict(replay.particle_map), particle_map)

                explicit_edges = {
                    (row["source_edge_index"], row["source_occurrence"]):
                    (row["representative_edge_index"], row["representative_occurrence"])
                    for row in data["edge_occurrence_map"]
                }
                self.assertEqual(dict(replay.edge_occurrence_map), explicit_edges)
                explicit_slots = {
                    (row["source_vertex"], row["source_slot"]):
                    (row["representative_vertex"], row["representative_slot"])
                    for row in data["final_slot_map"]
                }
                self.assertEqual(dict(replay.final_slot_map), explicit_slots)

                source_masses = {particle: 101.0 + particle for particle in range(1, 5)}
                representative_masses = {
                    target: source_masses[source] for source, target in particle_map.items()
                }
                for row in data["particle_map"]:
                    source = row["source_particle"]
                    target = row["representative_particle"]
                    self.assertEqual(row["momentum_label_substitution"],
                                     f"p_source[{source}] -> p_rep[{target}]")
                    self.assertEqual(row["mass_symbol_substitution"],
                                     f"m_source[{source}] -> m_rep[{target}]")
                    self.assertEqual(row["representative_mass_assignment"],
                                     f"m_rep[{target}] = m_source[{source}]")
                    self.assertEqual(representative_masses[target], source_masses[source])

                mapped_subsets = {
                    tuple(sorted(particle_map[particle] for particle in subset))
                    for subset in member.archived.canonical_record.momentum.variables
                }
                self.assertEqual(mapped_subsets, set(representative.momentum.variables))
                invariant_rows = {
                    row["source_ikv"]: row["representative_ikv"]
                    for row in data["invariant_map"]
                }
                for subset in member.archived.canonical_record.momentum.variables:
                    expected_label = invariant_label(tuple(sorted(particle_map[p] for p in subset)))
                    self.assertEqual(invariant_rows[invariant_label(subset)], expected_label)

    def test_duplicate_fixed_key_is_rejected_without_reenumeration(self):
        with gzip.open(ARCHIVE, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload[1] = dict(payload[0])
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "duplicate.json.gz"
            _write_archive(altered, payload)
            with self.assertRaisesRegex(ValueError, "repeated full fixed key"):
                build_d5_catalog(altered)

    def test_archive_hash_is_checked_again_after_loader_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "archive.json.gz"
            source.write_bytes(ARCHIVE.read_bytes())

            def mutate_during_load(path):
                path.write_bytes(path.read_bytes() + b"changed")
                return ()

            with patch.object(d5_catalog_module, "load_identity_archive", side_effect=mutate_during_load):
                with self.assertRaisesRegex(ValueError, "changed while"):
                    build_d5_catalog(source)

    def test_selector_exclusivity_and_existing_output_refusal(self):
        with self.assertRaisesRegex(ValueError, "select exactly one"):
            export_d5_formulas()
        with self.assertRaisesRegex(ValueError, "select exactly one"):
            export_d5_formulas(all_classes=True, class_selector="D5-001")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            marker = output / "README.md"
            marker.write_text("preserve this file", encoding="utf-8")
            with patch.object(d5_export_module, "build_d5_catalog", return_value=self.catalog):
                with self.assertRaises(FileExistsError):
                    export_d5_formulas(class_selector="D5-001", output_dir=output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve this file")

    def test_failure_coverage_survives_and_is_linked_from_readme(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "export"
            with patch.object(d5_export_module, "build_d5_catalog", return_value=self.catalog), \
                 patch.object(d5_export_module, "build_d5_formula_evidence", side_effect=RuntimeError("fixture build failure")):
                result = export_d5_formulas(class_selector="D5-001", output_dir=output)
            self.assertEqual(result["markdown_count"], 0)
            self.assertEqual(result["failed_class_count"], 1)
            coverage = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
            self.assertEqual(coverage["failed_class_count"], 1)
            self.assertIn("fixture build failure", coverage["failures"][0]["error"])
            failure_path = output / coverage["failures"][0]["failure_json"]
            self.assertTrue(failure_path.is_file())
            readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("## 构建失败", readme)
            self.assertIn(f"({failure_path.name})", readme)

    def test_six_step_markdown_uses_latex_without_raw_json_and_has_valid_tables(self):
        marked_class = self.catalog.classes[0]
        card, trace, chart, formula_artifact = _six_step_fixture(marked_class)
        markdown = d5_export_module._render_markdown(
            marked_class, card, trace, chart, formula_artifact
        )
        for heading in (
            "## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.",
        ):
            self.assertIn(heading, markdown)
        self.assertNotIn("未提供", markdown)
        self.assertNotIn(r"\mathtt{\detokenize{", markdown)
        self.assertNotIn("{'", markdown)
        self.assertIn("jing_psf_per_branch", markdown)
        self.assertIn("standard_2pi4_per_branch", markdown)
        self.assertIn("jing_psf_summed", markdown)
        self.assertIn("standard_2pi4_summed", markdown)

        current_table_columns = None
        for line in markdown.splitlines():
            if line.startswith("|") and line.endswith("|"):
                columns = len(line.split("|")) - 2
                if current_table_columns is None:
                    current_table_columns = columns
                self.assertEqual(columns, current_table_columns)
            else:
                current_table_columns = None

    def test_benchmark_status_summary_preserves_local_status_semantics(self):
        self.assertEqual(
            d5_export_module._benchmark_status_counts(
                [
                    {"benchmark_status": "verified_local"},
                    {"benchmark_status": "failed"},
                    {"benchmark_status": "blocked"},
                ],
                [{"class_id": "D5-failed-build"}],
            ),
            {"pass": 1, "fail": 1, "pending": 2},
        )

    def test_math_vertical_bar_escape_terminates_before_following_variable(self):
        expression = r"\left|k\right|k_2"
        rendered = (
            d5_export_module._inline_math(expression),
            d5_export_module._display_latex(expression),
            d5_export_module._latex(expression),
        )
        for output in rendered:
            self.assertIn(r"\left\vert k\right\vert k_2", output)
            self.assertNotIn(r"\vertk", output)


if __name__ == "__main__":
    unittest.main()
