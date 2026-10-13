import json
import math
import unittest

from im_fixtures import fixture_a, fixture_b
from phase_space_graph.formula.reference import (
    build_bridge_formula,
    build_reference_chart,
    reconstruct_reference_momenta,
)
from phase_space_graph.formula_inputs import build_formula_input
from phase_space_graph.formula.records import FormulaOptions
from phase_space_graph.formula.b_direct import (
    B_F03_DIFFERENTIAL_MEASURE,
    B_F03_DENSITY_EXPRESSION,
    B_F03_NORMALIZATION_IDENTITY,
    SUPPORTED_NORMALIZATIONS,
    check_b_f03_eligibility,
    sample_b_f03,
)


class BDirectSampleTests(unittest.TestCase):
    def setUp(self):
        self.record = fixture_b().record
        self.masses = (0.5, 0.6, 0.4, 0.7)

    def sample(self, **overrides):
        coordinates = {
            "total_mass": 5.0,
            "masses": self.masses,
            "s12": 2.56,
            "s34": 2.89,
            "s23": 2.0,
            "theta0": 0.8,
            "phi0": 0.4,
            "theta1": 1.1,
            "phi1": 1.7,
            "phi4_star": 0.9,
        }
        coordinates.update(overrides)
        return sample_b_f03(self.record, **coordinates)

    def test_fixture_b_is_selected_by_structure_and_ordered_invariants(self):
        eligibility = check_b_f03_eligibility(self.record)
        self.assertTrue(eligibility.eligible, eligibility.reasons)
        self.assertFalse(check_b_f03_eligibility(fixture_a().record).eligible)

    def test_reconstructed_sample_is_on_shell_conserved_and_has_requested_invariants(self):
        sample = self.sample()
        self.assertEqual(sample.status, "physical", sample.diagnostics)
        self.assertEqual(sample.branch_count, 1)
        self.assertTrue(sample.positive_energy)
        self.assertTrue(sample.all_shells_checked)
        self.assertTrue(sample.conservation_checked)
        self.assertEqual(len(sample.momenta), 4)

        masses = self.masses
        p1, p2, p3, p4 = sample.momenta
        for momentum, mass in zip(sample.momenta, masses):
            self.assertGreater(momentum[0], 0.0)
            self.assertAlmostEqual(self.mass_squared(momentum), mass * mass, places=9)

        self.assert_four_close(self.add(p1, p2), sample.q12)
        self.assert_four_close(self.add(p3, p4), sample.q34)
        self.assert_four_close(self.add(p2, p3), sample.q23)
        self.assert_four_close(self.add(sample.q34, p2), sample.Q)
        self.assertAlmostEqual(self.mass_squared(self.add(p1, p2)), 2.56, places=9)
        self.assertAlmostEqual(self.mass_squared(self.add(p3, p4)), 2.89, places=9)
        self.assertAlmostEqual(self.mass_squared(self.add(p2, p3)), 2.0, places=9)
        self.assert_four_close(
            tuple(sum(momentum[i] for momentum in sample.momenta) for i in range(4)),
            (5.0, 0.0, 0.0, 0.0),
        )
        payload = sample.to_dict()
        self.assertEqual(payload["coordinates"]["invariant_values"], [2.56, 2.89, 2.0])
        self.assertEqual(payload["momenta"][0]["components"], list(p1))
        self.assertIn("no-rotation boost", " ".join(payload["angle_definitions"]))
        json.dumps(payload)

    def test_unique_cosine_root_and_analytic_measure_normalization(self):
        sample = self.sample()
        self.assertEqual(sample.status, "physical", sample.diagnostics)
        self.assertTrue(-1.0 <= sample.root_cosine <= 1.0)
        self.assertAlmostEqual(sample.root_residual, 0.0, places=11)
        self.assertAlmostEqual(sample.c_reconstructed, sample.root_cosine, places=11)

        M = 5.0
        m12 = math.sqrt(2.56)
        sqrt_S = math.sqrt(sample.S)
        left = (
            (2.0 * math.pi) ** -6 * sample.k12 / (4.0 * M)
            * (2.0 * math.pi) ** -6 * sample.k1 / (4.0 * m12)
            * (2.0 * math.pi) ** -6 * sample.k4 / (4.0 * sqrt_S)
            * (2.0 * math.pi) ** 9
            * (2.0 * math.pi) ** -3
            / (2.0 * sample.k2 * sample.k4)
        )
        self.assertAlmostEqual(left, sample.density, delta=abs(sample.density) * 1e-13)
        self.assertIn("ds12 ds34 ds23", B_F03_DIFFERENTIAL_MEASURE)
        self.assertIn("dOmega_q12", B_F03_DIFFERENTIAL_MEASURE)
        self.assertIn("dOmega_p1", B_F03_DIFFERENTIAL_MEASURE)
        self.assertIn("dphi4*", B_F03_DIFFERENTIAL_MEASURE)
        self.assertIn("128", B_F03_DENSITY_EXPRESSION)
        self.assertIn("(2*pi)^12", B_F03_DENSITY_EXPRESSION)
        self.assertEqual(SUPPORTED_NORMALIZATIONS, ("jing_psf", "standard_2pi4"))
        self.assertIn("(2*pi)^9", B_F03_NORMALIZATION_IDENTITY)
        self.assertIn("(2*pi)^-3", B_F03_NORMALIZATION_IDENTITY)

    def test_standard_sample_density_is_jing_density_times_global_factor(self):
        jing = self.sample()
        standard = self.sample(normalization="standard_2pi4")
        expected_ratio = (2.0 * math.pi) ** 4
        self.assertEqual(jing.status, "physical", jing.diagnostics)
        self.assertEqual(standard.status, "physical", standard.diagnostics)
        self.assertAlmostEqual(standard.density / jing.density, expected_ratio, places=11)
        self.assertEqual(standard.momenta, jing.momenta)
        standard_dict = standard.to_dict()
        self.assertEqual(standard_dict["normalization"], "standard_2pi4")
        self.assertIn("dPhi4^S", standard_dict["differential_measure"])
        self.assertIn("dPhi4^S=(2*pi)^4 dPhi4^J", standard_dict["differential_measure"])
        self.assertIn("(2*pi)^8", standard_dict["density_expression"])
        self.assertIn("rho_BF03^S", standard_dict["normalization_identity"])

    def test_reference_normalization_ratio_uses_its_own_identical_angle_measure(self):
        # Compare the two conventions on the same six-angle reference chart;
        # this does not compare its density to the distinct five-angle B chart.
        card_jing = build_formula_input(self.record, measure_convention="jing_psf")
        card_standard = build_formula_input(self.record, measure_convention="standard_2pi4")
        chart = build_reference_chart(card_jing)
        reference_jing = build_bridge_formula(
            card_jing,
            chart,
            options=FormulaOptions(normalization="jing_psf"),
        )
        reference_standard = build_bridge_formula(
            card_standard,
            chart,
            options=FormulaOptions(normalization="standard_2pi4"),
        )
        values = {
            "s12": 2.56,
            "s34": 2.89,
            "theta0": 0.8,
            "phi0": 0.4,
            "theta12": 1.1,
            "phi12": 1.7,
            "theta34": 0.9,
            "phi34": 2.2,
        }
        rho_jing = reference_jing.numeric_density(values, self.masses, 5.0)
        rho_standard = reference_standard.numeric_density(values, self.masses, 5.0)
        self.assertAlmostEqual(
            rho_standard / rho_jing,
            (2.0 * math.pi) ** 4,
            places=11,
        )

    def test_roundtrip_against_independent_sequential_reference_coordinates(self):
        card = build_formula_input(self.record, measure_convention="jing_psf")
        chart = build_reference_chart(card)
        reference_values = {
            "s12": 2.56,
            "s34": 2.89,
            "theta0": 0.8,
            "phi0": 0.4,
            "theta12": 1.1,
            "phi12": 1.7,
            "theta34": 0.9,
            "phi34": 2.2,
        }
        reference = reconstruct_reference_momenta(
            chart, reference_values, self.masses, 5.0
        )
        p1, p2, p3, p4 = reference
        q34 = self.add(p3, p4)
        Q = self.add(q34, p2)
        beta = tuple(Q[i] / Q[0] for i in range(1, 4))
        p1_star = self.boost(p1, tuple(-item for item in beta))
        p2_star = self.boost(p2, tuple(-item for item in beta))
        p4_star = self.boost(p4, tuple(-item for item in beta))

        p2_spatial = p2_star[1:]
        ez = self.unit(p2_spatial)
        p1_spatial = p1_star[1:]
        projection = tuple(
            p1_spatial[i] - self.dot3(p1_spatial, ez) * ez[i]
            for i in range(3)
        )
        ex = self.unit(projection)
        ey = self.cross(ez, ex)
        phi4_star = math.atan2(
            self.dot3(p4_star[1:], ey), self.dot3(p4_star[1:], ex)
        ) % (2.0 * math.pi)
        s23 = self.mass_squared(self.add(p2, p3))

        direct = sample_b_f03(
            self.record,
            5.0,
            self.masses,
            2.56,
            2.89,
            s23,
            0.8,
            0.4,
            1.1,
            1.7,
            phi4_star,
        )
        self.assertEqual(direct.status, "physical", direct.diagnostics)
        for expected, actual in zip(reference, direct.momenta):
            self.assert_four_close(actual, expected)

    def test_local_change_of_variables_matches_same_measure_densities_at_eight_points(self):
        """Compare densities after the local (u34,phi34)->(s23,phi4*) map.

        Here u=cos(theta) is the numerical angular variable, so
        dOmega=dphi |du|=sin(theta)dtheta dphi.  All phi coordinates are
        periodic modulo 2*pi; finite differences use the wrapped shortest
        angular difference.  Thus the determinant is for (u34,phi34) to
        (s23,phi4*), and rho_B = rho_ref/|det| with no extra sine factor.
        Raw rho_B and rho_ref are not compared because their coordinate
        measures differ.
        """

        masses = (0.5, 0.6, 0.4, 0.7)  # Unequal masses throughout this sweep.
        card = build_formula_input(self.record, measure_convention="jing_psf")
        chart = build_reference_chart(card)
        reference_density = build_bridge_formula(
            card, chart, options=FormulaOptions(normalization="jing_psf")
        )
        points = (
            (0.23, 0.4, 0.15, 1.7, 0.31, 2.2),
            (-0.48, 1.3, -0.6, 2.3, 0.52, 4.1),
            (0.7, 2.5, 0.61, 4.4, -0.36, 0.8),
            (-0.25, 4.8, 0.42, 0.9, 0.68, 3.0),
            (0.53, 5.7, -0.31, 5.1, -0.58, 1.1),
            (-0.71, 3.2, 0.22, 2.9, 0.14, 5.5),
            (0.36, 1.9, -0.72, 0.6, 0.76, 4.8),
            (-0.12, 5.1, 0.81, 3.7, -0.22, 2.6),
            (0.12, 2.7, -0.48, 4.9, 0.45, 0.2),
            (0.68, 0.9, 0.3, 3.3, -0.7, 5.0),
        )
        passed = []
        excluded = []
        for index, point in enumerate(points):
            reference_values = self.reference_values_from_cosines(point)
            mapped = self.map_reference_to_b(chart, masses, 5.0, point)
            s23, phi4_star, cosine, reference = mapped
            direct = sample_b_f03(
                self.record,
                5.0,
                masses,
                2.56,
                2.89,
                s23,
                reference_values["theta0"],
                reference_values["phi0"],
                reference_values["theta12"],
                reference_values["phi12"],
                phi4_star,
                normalization="jing_psf",
            )
            if direct.status != "physical" or abs(cosine) >= 1.0 - 1e-5:
                excluded.append((index, "physical or cosine-boundary degeneracy"))
                continue

            jacobian_coarse = self.local_map_jacobian(chart, masses, point, 1.0e-4)
            jacobian_fine = self.local_map_jacobian(chart, masses, point, 5.0e-5)
            determinant_coarse = self.determinant(jacobian_coarse)
            determinant_fine = self.determinant(jacobian_fine)
            column1 = math.hypot(jacobian_fine[0][0], jacobian_fine[1][0])
            column2 = math.hypot(jacobian_fine[0][1], jacobian_fine[1][1])
            conditioning = abs(determinant_fine) / max(column1 * column2, 1e-300)
            step_stability = abs(determinant_coarse - determinant_fine) / max(
                abs(determinant_fine), 1e-300
            )
            if conditioning < 1.0e-4 or step_stability > 2.0e-5:
                excluded.append((index, "ill-conditioned or finite-difference unstable"))
                continue

            rho_ref = reference_density.numeric_density(reference_values, masses, 5.0)
            rho_b_from_reference = rho_ref / abs(determinant_fine)
            relative_residual = abs(direct.density - rho_b_from_reference) / abs(direct.density)
            self.assertLess(
                relative_residual,
                2.0e-5,
                msg=f"point {index} failed: det={determinant_fine}, residual={relative_residual}",
            )
            for expected, actual in zip(reference, direct.momenta):
                self.assert_four_close(actual, expected)
            passed.append(index)

        self.assertGreaterEqual(
            len(passed),
            8,
            msg=f"only {len(passed)} nondegenerate, well-conditioned points passed; excluded={excluded}",
        )

    def test_unsupported_degenerate_and_unphysical_are_distinct(self):
        unsupported = sample_b_f03(
            fixture_a().record,
            5.0,
            self.masses,
            2.56,
            2.89,
            2.0,
            0.8,
            0.4,
            1.1,
            1.7,
            0.9,
        )
        unknown_normalization = sample_b_f03(
            self.record,
            5.0,
            self.masses,
            2.56,
            2.89,
            2.0,
            0.8,
            0.4,
            1.1,
            1.7,
            0.9,
            normalization="unregistered_convention",
        )
        degenerate = self.sample(s12=(self.masses[0] + self.masses[1]) ** 2)
        unphysical = self.sample(s23=1.01)
        self.assertEqual(unsupported.status, "unsupported")
        self.assertEqual(degenerate.status, "degenerate")
        self.assertEqual(unphysical.status, "unphysical")
        self.assertEqual(unknown_normalization.status, "unsupported")
        self.assertEqual(unknown_normalization.to_dict()["normalization"], "unregistered_convention")

    def test_polar_angles_and_cosine_boundary_are_chart_degeneracies(self):
        theta0_pole = self.sample(theta0=0.0)
        theta1_pole = self.sample(theta1=math.pi)
        outside_angle_domain = self.sample(theta0=-1e-3)
        self.assertEqual(theta0_pole.status, "degenerate")
        self.assertEqual(theta1_pole.status, "degenerate")
        self.assertEqual(outside_angle_domain.status, "unphysical")

        # At this fixed physical setup c0 crosses -1 near the lower s23 edge.
        # Approach the root boundary from inside; phi4* becomes redundant there.
        lower, upper = 1.01, 1.2
        for _ in range(60):
            middle = 0.5 * (lower + upper)
            point = self.sample(s23=middle)
            if point.root_cosine is not None and point.root_cosine < -1.0:
                lower = middle
            else:
                upper = middle
        root_boundary = self.sample(s23=upper)
        self.assertEqual(root_boundary.status, "degenerate", root_boundary.diagnostics)
        self.assertAlmostEqual(abs(root_boundary.root_cosine), 1.0, places=8)

    @staticmethod
    def mass_squared(momentum):
        return momentum[0] ** 2 - sum(component * component for component in momentum[1:])

    @staticmethod
    def add(left, right):
        return tuple(left[i] + right[i] for i in range(4))

    def assert_four_close(self, actual, expected):
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=9)

    @staticmethod
    def reference_values_from_cosines(point):
        u0, phi0, u1, phi1, u34, phi34 = point
        return {
            "s12": 2.56,
            "s34": 2.89,
            "theta0": math.acos(u0),
            "phi0": phi0 % (2.0 * math.pi),
            "theta12": math.acos(u1),
            "phi12": phi1 % (2.0 * math.pi),
            "theta34": math.acos(u34),
            "phi34": phi34 % (2.0 * math.pi),
        }

    @classmethod
    def map_reference_to_b(cls, chart, masses, total_mass, point):
        values = cls.reference_values_from_cosines(point)
        p1, p2, p3, p4 = reconstruct_reference_momenta(chart, values, masses, total_mass)
        q23 = cls.add(p2, p3)
        q34 = cls.add(p3, p4)
        Q = cls.add(q34, p2)
        beta = tuple(Q[i] / Q[0] for i in range(1, 4))
        inverse_beta = tuple(-component for component in beta)
        p1_star = cls.boost(p1, inverse_beta)
        p2_star = cls.boost(p2, inverse_beta)
        p4_star = cls.boost(p4, inverse_beta)
        ez = cls.unit(p2_star[1:])
        p1_spatial = p1_star[1:]
        projection = tuple(
            p1_spatial[i] - cls.dot3(p1_spatial, ez) * ez[i]
            for i in range(3)
        )
        ex = cls.unit(projection)
        ey = cls.cross(ez, ex)
        phi4_star = math.atan2(
            cls.dot3(p4_star[1:], ey), cls.dot3(p4_star[1:], ex)
        ) % (2.0 * math.pi)
        cosine = cls.dot3(p4_star[1:], p2_star[1:]) / (
            math.sqrt(cls.dot3(p4_star[1:], p4_star[1:]))
            * math.sqrt(cls.dot3(p2_star[1:], p2_star[1:]))
        )
        return cls.mass_squared(q23), phi4_star, cosine, (p1, p2, p3, p4)

    @classmethod
    def local_map_jacobian(cls, chart, masses, point, step):
        """Central-difference d(s23,phi4*)/d(u34,phi34)."""

        columns = []
        for coordinate_index in (4, 5):
            plus = list(point)
            minus = list(point)
            plus[coordinate_index] += step
            minus[coordinate_index] -= step
            if coordinate_index == 5:
                plus[coordinate_index] %= 2.0 * math.pi
                minus[coordinate_index] %= 2.0 * math.pi
            map_plus = cls.map_reference_to_b(chart, masses, 5.0, tuple(plus))
            map_minus = cls.map_reference_to_b(chart, masses, 5.0, tuple(minus))
            columns.append((
                (map_plus[0] - map_minus[0]) / (2.0 * step),
                cls.periodic_difference(map_plus[1], map_minus[1]) / (2.0 * step),
            ))
        return (
            (columns[0][0], columns[1][0]),
            (columns[0][1], columns[1][1]),
        )

    @staticmethod
    def periodic_difference(left, right):
        return (left - right + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def determinant(matrix):
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    @staticmethod
    def dot3(left, right):
        return sum(left[i] * right[i] for i in range(3))

    @classmethod
    def unit(cls, vector):
        norm = math.sqrt(cls.dot3(vector, vector))
        return tuple(component / norm for component in vector)

    @staticmethod
    def cross(left, right):
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    @staticmethod
    def boost(momentum, beta):
        beta2 = sum(item * item for item in beta)
        if beta2 < 1e-30:
            return momentum
        gamma = 1.0 / math.sqrt(1.0 - beta2)
        dot = sum(beta[i] * momentum[i + 1] for i in range(3))
        factor = (gamma - 1.0) * dot / beta2 + gamma * momentum[0]
        return (
            gamma * (momentum[0] + dot),
            momentum[1] + factor * beta[0],
            momentum[2] + factor * beta[1],
            momentum[3] + factor * beta[2],
        )


if __name__ == "__main__":
    unittest.main()
