import json
import math
import unittest

from im_fixtures import fixture_b
from phase_space_graph.formula import (
    BHelicitySample,
    fixed_to_helicity_angles,
    helicity_to_fixed_angles,
    sample_b_f03_helicity,
)
from phase_space_graph.formula.b_direct import sample_b_f03


class BHelicityChartTests(unittest.TestCase):
    def setUp(self):
        self.record = fixture_b().record
        self.masses = (0.5, 0.6, 0.4, 0.7)

    def test_forward_and_inverse_angle_maps_round_trip(self):
        points = (
            (0.8, 0.4, 0.35, 1.7),
            (1.2, 5.6, -0.42, 4.4),
            (math.pi / 2.0, 1.4, 0.1, 5.8),
        )
        for theta0, phi0, u, psi in points:
            with self.subTest(point=(theta0, phi0, u, psi)):
                mapped = helicity_to_fixed_angles(theta0, phi0, u, psi)
                self.assertEqual(mapped.status, "mapped", mapped.diagnostics)
                self.assertAlmostEqual(mapped.measure_jacobian_abs, 1.0, places=14)
                inverse = fixed_to_helicity_angles(
                    theta0,
                    phi0,
                    mapped.theta1,
                    mapped.phi1,
                )
                self.assertEqual(inverse.status, "mapped", inverse.diagnostics)
                self.assertAlmostEqual(inverse.u, u, places=12)
                self.assertAlmostEqual(
                    self.periodic_difference(inverse.psi, psi),
                    0.0,
                    places=12,
                )

    def test_local_numeric_solid_angle_jacobian_has_unit_absolute_value(self):
        # The finite-difference output coordinates are cos(theta1),phi1;
        # both source and target use dOmega=d(cos(theta))d(phi) in magnitude.
        for theta0, phi0, u, psi in (
            (0.8, 0.4, 0.35, 1.7),
            (1.2, 5.6, -0.42, 4.4),
            (math.pi / 2.0, 1.4, 0.1, 5.8),
        ):
            jacobian = self.numeric_angle_jacobian(theta0, phi0, u, psi, 1e-6)
            determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
            self.assertAlmostEqual(abs(determinant), 1.0, places=8)

    def test_local_helicity_basis_is_right_handed(self):
        theta0, phi0 = 0.8, 0.4
        n0 = self.direction(theta0, phi0)
        e_theta = helicity_to_fixed_angles(theta0, phi0, 0.0, 0.0).n1
        e_phi = helicity_to_fixed_angles(theta0, phi0, 0.0, math.pi / 2.0).n1
        cross = (
            e_theta[1] * e_phi[2] - e_theta[2] * e_phi[1],
            e_theta[2] * e_phi[0] - e_theta[0] * e_phi[2],
            e_theta[0] * e_phi[1] - e_theta[1] * e_phi[0],
        )
        for actual, expected in zip(cross, n0):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_wrapper_reuses_direct_sample_momenta_and_density(self):
        fixed = sample_b_f03(
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
            normalization="standard_2pi4",
        )
        helicity_angles = fixed_to_helicity_angles(0.8, 0.4, 1.1, 1.7)
        wrapped = sample_b_f03_helicity(
            self.record,
            5.0,
            self.masses,
            2.56,
            2.89,
            2.0,
            0.8,
            0.4,
            helicity_angles.psi,
            helicity_angles.u,
            0.9,
            normalization="standard_2pi4",
        )
        self.assertIsInstance(wrapped, BHelicitySample)
        self.assertEqual(wrapped.status, "physical", wrapped.diagnostics)
        self.assertEqual(wrapped.direct_sample.normalization, "standard_2pi4")
        self.assertAlmostEqual(wrapped.density, fixed.density, delta=abs(fixed.density) * 1e-12)
        for expected, actual in zip(fixed.momenta, wrapped.momenta):
            for left, right in zip(expected, actual):
                self.assertAlmostEqual(left, right, places=11)
        payload = wrapped.to_dict()
        self.assertEqual(payload["coordinate_status"], "mapped")
        json.dumps(payload)

    def test_chart_boundaries_are_degenerate_not_physical_no_solutions(self):
        overall_pole = helicity_to_fixed_angles(0.0, 0.4, 0.35, 1.7)
        parallel = helicity_to_fixed_angles(0.8, 0.4, 1.0, 1.7)
        inverse_parallel = fixed_to_helicity_angles(0.8, 0.4, 0.8, 0.4)
        invalid_u = helicity_to_fixed_angles(0.8, 0.4, 1.01, 1.7)
        self.assertEqual(overall_pole.status, "degenerate")
        self.assertEqual(parallel.status, "degenerate")
        self.assertTrue(any("psi redundant" in item for item in parallel.diagnostics))
        self.assertEqual(inverse_parallel.status, "degenerate")
        self.assertIsNone(inverse_parallel.psi)
        self.assertEqual(invalid_u.status, "unphysical")

        # The helicity coordinate is degenerate at u=1, but the wrapped direct
        # kinematics still exists. Report that chart boundary without turning
        # it into an unphysical point.
        boundary_sample = sample_b_f03_helicity(
            self.record,
            5.0,
            self.masses,
            2.56,
            2.89,
            2.0,
            0.8,
            0.4,
            1.7,
            1.0,
            0.9,
        )
        self.assertEqual(boundary_sample.coordinate_status, "degenerate")
        self.assertEqual(boundary_sample.status, "degenerate")
        if boundary_sample.direct_sample is not None:
            self.assertIn(boundary_sample.direct_sample.status, {"physical", "degenerate"})

        overall_pole_sample = sample_b_f03_helicity(
            self.record,
            5.0,
            self.masses,
            2.56,
            2.89,
            2.0,
            0.0,
            0.4,
            1.7,
            0.35,
            0.9,
        )
        self.assertEqual(overall_pole_sample.coordinate_status, "degenerate")
        self.assertEqual(overall_pole_sample.status, "degenerate")
        self.assertTrue(any("theta0 is at a polar" in item for item in overall_pole_sample.diagnostics))

    @classmethod
    def fixed_cos_theta_phi(cls, theta0, phi0, u, psi):
        result = helicity_to_fixed_angles(theta0, phi0, u, psi)
        if result.status != "mapped":
            raise AssertionError(result.diagnostics)
        return math.cos(result.theta1), result.phi1

    @classmethod
    def numeric_angle_jacobian(cls, theta0, phi0, u, psi, step):
        columns = []
        for index in (0, 1):
            plus = [u, psi]
            minus = [u, psi]
            plus[index] += step
            minus[index] -= step
            if index == 1:
                plus[index] %= 2.0 * math.pi
                minus[index] %= 2.0 * math.pi
            plus_value = cls.fixed_cos_theta_phi(theta0, phi0, *plus)
            minus_value = cls.fixed_cos_theta_phi(theta0, phi0, *minus)
            columns.append((
                (plus_value[0] - minus_value[0]) / (2.0 * step),
                cls.periodic_difference(plus_value[1], minus_value[1]) / (2.0 * step),
            ))
        return (
            (columns[0][0], columns[1][0]),
            (columns[0][1], columns[1][1]),
        )

    @staticmethod
    def direction(theta, phi):
        return (
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        )

    @staticmethod
    def periodic_difference(left, right):
        return (left - right + math.pi) % (2.0 * math.pi) - math.pi


if __name__ == "__main__":
    unittest.main()
