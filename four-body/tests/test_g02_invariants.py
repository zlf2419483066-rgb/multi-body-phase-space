import unittest
from fractions import Fraction

from phase_space_graph import (
    affine_expression,
    analyze_invariants,
    base_pair_invariants,
    candidate_invariants,
    invariant_label,
)


class InvariantGenerationTests(unittest.TestCase):
    def test_candidate_order_and_count(self) -> None:
        self.assertEqual(len(candidate_invariants(4)), 10)
        self.assertEqual(
            [invariant_label(item) for item in candidate_invariants(4)],
            [
                "s12",
                "s13",
                "s14",
                "s23",
                "s24",
                "s34",
                "s123",
                "s124",
                "s134",
                "s234",
            ],
        )
        self.assertEqual(len(candidate_invariants(3)), 3)

    def test_triple_affine_expression(self) -> None:
        expression = affine_expression((3, 1, 2), 4)
        self.assertEqual(expression.variables, base_pair_invariants(4))
        self.assertEqual(
            expression.coefficients,
            (Fraction(1), Fraction(1), Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        )
        self.assertEqual(expression.mass_constant.parent_coefficient, Fraction(0))
        self.assertEqual(
            expression.mass_constant.particle_coefficients,
            (Fraction(-1), Fraction(-1), Fraction(-1), Fraction(0)),
        )
        self.assertEqual(expression.format(), "s12 + s13 + s23 - m1^2 - m2^2 - m3^2")
        self.assertEqual(
            expression.evaluate(
                [10, 20, 30, 40, 50, 60],
                parent_mass_squared=100,
                particle_mass_squares=[1, 4, 9, 16],
            ),
            Fraction(56),
        )


class IndependenceAnalysisTests(unittest.TestCase):
    def test_independent_examples(self) -> None:
        cases = [
            ([(1, 2), (3, 4)], 2),
            ([(1, 2), (1, 3), (2, 3)], 3),
            ([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)], 5),
            ([(1, 2), (3, 4), (2, 3)], 3),
        ]
        for variables, expected_rank in cases:
            with self.subTest(variables=variables):
                result = analyze_invariants(variables, 4)
                self.assertTrue(result.independent)
                self.assertEqual(result.rank, expected_rank)
                self.assertIsNone(result.certificate)

    def test_duplicate_pair_gets_zero_relation_certificate(self) -> None:
        result = analyze_invariants([(1, 2), (1, 2)], 4)
        self.assertFalse(result.independent)
        self.assertEqual(result.rank, 1)
        self.assertIsNotNone(result.certificate)
        certificate = result.certificate
        assert certificate is not None
        self.assertTrue(certificate.verified)
        self.assertEqual(certificate.multipliers, (Fraction(1), Fraction(-1)))
        self.assertTrue(certificate.rhs_mass.is_zero)
        self.assertEqual(certificate.format(), "s12 - s12 = 0")

    def test_triple_local_identity_certificate(self) -> None:
        result = analyze_invariants([(1, 2), (1, 3), (2, 3), (1, 2, 3)], 4)
        self.assertFalse(result.independent)
        self.assertEqual(result.rank, 3)
        certificate = result.certificate
        self.assertIsNotNone(certificate)
        assert certificate is not None
        self.assertTrue(certificate.verified)
        self.assertEqual(certificate.multipliers, (Fraction(1), Fraction(1), Fraction(1), Fraction(-1)))
        self.assertEqual(
            certificate.rhs_mass.particle_coefficients,
            (Fraction(1), Fraction(1), Fraction(1), Fraction(0)),
        )
        self.assertEqual(certificate.format(), "s12 + s13 + s23 - s123 = m1^2 + m2^2 + m3^2")

    def test_all_six_pairs_have_five_independent_directions(self) -> None:
        result = analyze_invariants(candidate_invariants(4)[:6], 4)
        self.assertFalse(result.independent)
        self.assertEqual(result.rank, 5)
        certificate = result.certificate
        self.assertIsNotNone(certificate)
        assert certificate is not None
        self.assertTrue(certificate.verified)
        self.assertEqual(certificate.constraint_multiplier, Fraction(1))
        self.assertEqual(certificate.rhs_mass.parent_coefficient, Fraction(1))
        self.assertEqual(
            certificate.rhs_mass.particle_coefficients,
            (Fraction(2), Fraction(2), Fraction(2), Fraction(2)),
        )

    def test_three_body_constraint(self) -> None:
        result = analyze_invariants(candidate_invariants(3), 3)
        self.assertFalse(result.independent)
        self.assertEqual(result.rank, 2)
        certificate = result.certificate
        self.assertIsNotNone(certificate)
        assert certificate is not None
        self.assertTrue(certificate.verified)
        self.assertEqual(certificate.constraint_multiplier, Fraction(1))
        self.assertEqual(
            certificate.rhs_mass.particle_coefficients,
            (Fraction(1), Fraction(1), Fraction(1)),
        )

    def test_exact_fraction_coefficients_are_supported(self) -> None:
        result = analyze_invariants([(1, 2), (1, 2)], 4)
        self.assertEqual(result.to_dict()["certificate"]["multipliers"], ["1", "-1"])
        with self.assertRaises(TypeError):
            analyze_invariants([(1.0, 2)], 4)


if __name__ == "__main__":
    unittest.main()
