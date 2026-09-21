import math
import unittest
from dataclasses import replace

from phase_space_graph import (
    DoubleLineLabel,
    EdgeType,
    FinalParticleLabel,
    FormulaOptions,
    InternalEdge,
    Labeling,
    MomentumCertificate,
    Topology,
    analyze_invariants,
    build_formula_input,
)
from phase_space_graph.formula.direct_charts import (
    build_direct_formula,
    select_direct_chart,
)
from phase_space_graph.formula.reference import build_bridge_formula, build_reference_chart
from phase_space_graph.physical_labels import (
    PhysicalLabelRecord,
    measure_cancellation_counts,
    single_occurrence_groups,
)


def pair_card(convention="standard_2pi4"):
    topology = Topology(
        4,
        (0, 1, 2),
        (InternalEdge(0, 1, EdgeType.DOUBLE), InternalEdge(0, 2, EdgeType.DOUBLE)),
        0,
        {0: 0, 1: 2, 2: 2},
    )
    labeling = Labeling(
        (
            FinalParticleLabel(1, 0, 1),
            FinalParticleLabel(1, 1, 2),
            FinalParticleLabel(2, 0, 3),
            FinalParticleLabel(2, 1, 4),
        )
    )
    momentum = MomentumCertificate(
        ((0, 0, (1, 1, 0, 0)), (1, 0, (0, 0, 1, 1))),
        ((0, (0, 0, 0, 0)), (1, (0, 0, 0, 0)), (2, (0, 0, 0, 0))),
        True,
    )
    independence = analyze_invariants(momentum.variables, 4)
    record = PhysicalLabelRecord(
        topology,
        Labeling(
            labeling.final_particles,
            (),
            (DoubleLineLabel(0, 0, (1, 2)), DoubleLineLabel(1, 0, (3, 4))),
        ),
        momentum,
        independence,
        single_occurrence_groups(topology, labeling),
        measure_cancellation_counts(topology, labeling),
    )
    return build_formula_input(record, measure_convention=convention)


def direct_values():
    return {
        "s12": 4.0,
        "s34": 4.0,
        "theta0": 1.1,
        "phi0": 0.3,
        "thetaA": 0.8,
        "phiA": 1.7,
        "thetaB": 2.0,
        "phiB": 4.2,
    }


def reference_values():
    values = direct_values()
    return {
        "s12": values["s12"],
        "s34": values["s34"],
        "theta0": values["theta0"],
        "phi0": values["phi0"],
        "theta12": values["thetaA"],
        "phi12": values["phiA"],
        "theta34": values["thetaB"],
        "phi34": values["phiB"],
    }


class FormulaDomainReviewTests(unittest.TestCase):
    def setUp(self):
        self.card = pair_card()
        self.direct_chart = select_direct_chart(self.card)
        self.direct = build_direct_formula(self.card, self.direct_chart)
        self.reference_chart = build_reference_chart(self.card)
        self.bridge = build_bridge_formula(self.card, self.reference_chart)

    def test_parent_sum_domain_rejects_point_with_nonnegative_kallen(self):
        # lambda(M^2, s12, s34) is non-negative here, but the subsystem roots
        # violate sqrt(s12)+sqrt(s34) <= M.
        for formula, values in ((self.direct, direct_values()), (self.bridge, reference_values())):
            with self.subTest(profile=formula.profile):
                bad = dict(values)
                bad["s12"] = 36.0
                bad["s34"] = 1.0
                with self.assertRaises(ValueError):
                    formula.numeric_density(bad, (0.0, 0.0, 0.0, 0.0), 3.0)
                with self.assertRaises(ValueError):
                    formula.numeric_momenta(bad, (0.0, 0.0, 0.0, 0.0), 3.0)

    def test_nonfinite_and_angle_inputs_are_rejected(self):
        for formula, values in ((self.direct, direct_values()), (self.bridge, reference_values())):
            for key, bad_value in (("s12", math.nan), ("s34", math.inf), ("theta0", -1e-3), ("phi0", 2 * math.pi)):
                with self.subTest(profile=formula.profile, key=key):
                    bad = dict(values)
                    bad[key] = bad_value
                    with self.assertRaises(ValueError):
                        formula.numeric_density(bad, (0.0, 0.0, 0.0, 0.0), 6.0)
                    with self.assertRaises(ValueError):
                        formula.numeric_momenta(bad, (0.0, 0.0, 0.0, 0.0), 6.0)
        with self.assertRaises(ValueError):
            self.direct.numeric_density(direct_values(), (-1.0, 0.0, 0.0, 0.0), 6.0)
        with self.assertRaises(ValueError):
            self.bridge.numeric_momenta(reference_values(), (0.0, math.inf, 0.0, 0.0), 6.0)

    def test_massless_particle_and_massless_pair_interior_are_allowed(self):
        direct_momenta = self.direct.numeric_momenta(direct_values(), (0.0, 0.0, 0.0, 0.0), 6.0)
        bridge_momenta = self.bridge.numeric_momenta(reference_values(), (0.0, 0.0, 0.0, 0.0), 6.0)
        self.assertTrue(math.isfinite(self.direct.numeric_density(direct_values(), (0.0, 0.0, 0.0, 0.0), 6.0)))
        self.assertTrue(math.isfinite(self.bridge.numeric_density(reference_values(), (0.0, 0.0, 0.0, 0.0), 6.0)))
        for momenta in (direct_momenta, bridge_momenta):
            self.assertEqual(len(momenta), 4)
            for p in momenta:
                self.assertTrue(all(math.isfinite(component) for component in p))
                self.assertGreaterEqual(p[0], 0.0)

    def test_zero_subsystem_is_explicit_degenerate_error(self):
        for formula, values in ((self.direct, direct_values()), (self.bridge, reference_values())):
            bad = dict(values)
            bad["s12"] = 0.0
            with self.subTest(profile=formula.profile):
                with self.assertRaisesRegex(ValueError, "degenerate"):
                    formula.numeric_density(bad, (0.0, 0.0, 0.0, 0.0), 6.0)
                with self.assertRaisesRegex(ValueError, "degenerate"):
                    formula.numeric_momenta(bad, (0.0, 0.0, 0.0, 0.0), 6.0)

    def test_options_and_chart_binding_are_checked(self):
        with self.assertRaises(ValueError):
            build_direct_formula(self.card, self.direct_chart, options=FormulaOptions(normalization="jing_psf"))
        with self.assertRaises(ValueError):
            build_bridge_formula(self.card, self.reference_chart, options=FormulaOptions(normalization="jing_psf"))
        wrong_chart = replace(self.direct_chart, target_variables=("s13", "s24"))
        with self.assertRaises(ValueError):
            build_direct_formula(self.card, wrong_chart)
        with self.assertRaises(ValueError):
            build_bridge_formula(self.card, replace(self.reference_chart, status="unsupported"))
        summed = build_direct_formula(
            self.card,
            self.direct_chart,
            options=FormulaOptions(normalization=self.card.measure_convention, density_mode="summed"),
        )
        self.assertTrue(summed.branches[0].summed)

    def test_expression_and_bridge_measure_metadata_are_explicit(self):
        self.assertIn("dsA dsB dOmega0 dOmegaA dOmegaB", self.direct.differential_measure)
        self.assertIn("sin(theta)dtheta dphi", self.direct.differential_measure)
        self.assertNotIn("sA", self.direct.density_expression)
        self.assertNotIn("m_i", self.direct.density_expression)
        self.assertTrue(all("p_i(y)" not in expression for expression in self.direct.momentum_expressions))
        direct_momentum = " ".join(self.direct.momentum_expressions)
        self.assertIn("beta_s12=+k_M/E_s12^P", direct_momentum)
        self.assertIn("p_2(y)=Boost_{beta_s12}((E_2^{s12}, -k_{s12}", direct_momentum)
        reference_momentum = " ".join(self.bridge.momentum_expressions)
        self.assertIn("beta12=+k_M/E12^P", reference_momentum)
        self.assertIn("p_2(x)=Boost_{beta12}((E_2^{t12}, -k_{t12}", reference_momentum)
        self.assertIn("local auxiliary coordinates", self.bridge.constraints[0])
        for convention in ("standard_2pi4", "jing_psf"):
            card = pair_card(convention)
            bridge = build_bridge_formula(card, build_reference_chart(card))
            for item in card.double_measures:
                rows = [row for row in bridge.constraints if item.variable_label in row and "delta" in row]
                self.assertTrue(rows)
                self.assertTrue(all("2*pi" not in row for row in rows))


if __name__ == "__main__":
    unittest.main()
