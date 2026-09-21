import math
import tempfile
import unittest
from pathlib import Path

from phase_space_graph import (
    DoubleLineLabel, EdgeType, FinalParticleLabel, InternalEdge, Labeling,
    MomentumCertificate, Topology, analyze_invariants, normalize_physical_labeling,
    build_formula_input, generate_formula, FormulaSelection, export_markdown,
)


def pair_fixture():
    topology = Topology(
        4, (0, 1, 2),
        (InternalEdge(0, 1, EdgeType.DOUBLE), InternalEdge(0, 2, EdgeType.DOUBLE)),
        0, {0: 0, 1: 2, 2: 2},
    )
    labeling = Labeling((
        FinalParticleLabel(1, 0, 1), FinalParticleLabel(1, 1, 2),
        FinalParticleLabel(2, 0, 3), FinalParticleLabel(2, 1, 4),
    ))
    vectors = ((0, 0, 0, 0), ((1, 1, 0, 0)), ((0, 0, 1, 1)))
    momentum = MomentumCertificate(
        ((0, 0, vectors[1]), (1, 0, vectors[2])),
        ((0, vectors[0]), (1, vectors[0]), (2, vectors[0])),
        True,
    )
    indep = analyze_invariants(momentum.variables, 4)
    # The certificate represents the two pair invariants and is independent.
    from phase_space_graph.physical_labels import PhysicalLabelRecord, single_occurrence_groups, measure_cancellation_counts
    record = PhysicalLabelRecord(topology, Labeling(labeling.final_particles, (), (
        DoubleLineLabel(0, 0, (1, 2)), DoubleLineLabel(1, 0, (3, 4)),
    )), momentum, indep, single_occurrence_groups(topology, labeling), measure_cancellation_counts(topology, labeling))
    return record


class FormulaModuleTests(unittest.TestCase):
    def test_pair_chart_builds_both_paths_and_benchmark(self):
        card = build_formula_input(pair_fixture(), measure_convention="standard_2pi4")
        self.assertEqual(card.availability, "ready")
        artifact = generate_formula(card, masses=(1.0, 1.0, 1.0, 1.0), total_mass=8.0)
        self.assertEqual(artifact.statuses["f0"], "built")
        self.assertEqual(artifact.statuses["bridge"], "built")
        self.assertEqual(artifact.statuses["direct"], "built")
        self.assertEqual(artifact.statuses["benchmark"], "verified")
        self.assertLess(artifact.internal_benchmark.max_relative_residual, 1e-12)

    def test_default_benchmark_points_follow_non_equal_mass_thresholds(self):
        card = build_formula_input(pair_fixture(), measure_convention="standard_2pi4")
        artifact = generate_formula(card, masses=(1.0, 1.5, 0.8, 1.2), total_mass=6.0)
        self.assertEqual(artifact.statuses["benchmark"], "verified")
        self.assertGreaterEqual(artifact.internal_benchmark.points, 8)

    def test_non_equal_mass_momenta_are_on_shell_and_sum_to_parent(self):
        card = build_formula_input(pair_fixture(), measure_convention="standard_2pi4")
        artifact = generate_formula(card, run_benchmark=False)
        chart = artifact.direct_formula
        values = {"s12": 8.41, "s34": 4.84, "theta0": 1.1, "phi0": .3,
                  "thetaA": .8, "phiA": 1.7, "thetaB": 2.0, "phiB": 4.2}
        masses = (1.0, 1.5, .8, 1.2)
        momenta = chart.numeric_momenta(values, masses, 6.0)
        self.assertEqual(len(momenta), 4)
        for p, mass in zip(momenta, masses):
            self.assertAlmostEqual(p[0]**2 - sum(x*x for x in p[1:]), mass**2, places=9)
        self.assertAlmostEqual(sum(p[0] for p in momenta), 6.0, places=9)
        for component in range(1, 4):
            self.assertAlmostEqual(sum(p[component] for p in momenta), 0.0, places=9)

    def test_default_benchmark_reports_empty_domain(self):
        card = build_formula_input(pair_fixture(), measure_convention="standard_2pi4")
        artifact = generate_formula(card, masses=(1.0, 1.5, 0.8, 1.2), total_mass=3.5)
        self.assertEqual(artifact.statuses["benchmark"], "unsupported")
        self.assertIn("no non-degenerate four-body", artifact.internal_benchmark.diagnostics[0])

    def test_unsupported_variable_family_is_explicit(self):
        from im_fixtures import fixture_b
        artifact = generate_formula(build_formula_input(normalize_physical_labeling(fixture_b().topology, fixture_b().labeling), measure_convention="jing_psf"), run_benchmark=False)
        self.assertEqual(artifact.statuses["direct"], "unsupported")
        self.assertEqual(artifact.statuses["bridge"], "built")
        self.assertFalse(artifact.direct_verified)

    def test_markdown_export_preserves_pending_status(self):
        card = build_formula_input(pair_fixture(), measure_convention="jing_psf")
        artifact = generate_formula(card, run_benchmark=False)
        with tempfile.TemporaryDirectory() as folder:
            output = export_markdown((artifact,), Path(folder) / "formula.md", selection=FormulaSelection((0,), "all"))
            text = output.read_text(encoding="utf-8")
            self.assertIn("F1_direct", text)
            self.assertIn("benchmark", text)
            self.assertIn("pending", text)


if __name__ == "__main__":
    unittest.main()
