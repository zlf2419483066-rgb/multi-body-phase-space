from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.independence.contracts import (  # noqa: E402
    InvariantOccurrence,
    PhysicsProfile,
    ReviewedCandidate,
    as_fraction,
)
from phase_space_graph5.independence.coordinates import (  # noqa: E402
    SUBSET_ORDER,
    X_ORDER,
    Z_ORDER,
    build_label_affine_map,
    full_subset_affine_map,
    pair_coordinate_map,
)
from phase_space_graph5.independence.exact_algebra import (  # noqa: E402
    SparsePolynomial,
    compose_linear,
    determinant,
    deserialize_polynomial,
    left_nullspace,
    matmul,
    nullspace,
    polynomial_constant,
    polynomial_gcd,
    polynomial_gradient,
    rank,
    right_inverse,
    serialize_polynomial,
    square_free_part,
)


def _minkowski_square(momentum: tuple[int, int, int, int]) -> Fraction:
    """Exact (+---) square for the independent affine-coordinate oracle."""

    energy, px, py, pz = momentum
    return Fraction(energy * energy - px * px - py * py - pz * pz)


def _add_momenta(momentum_list: tuple[tuple[int, int, int, int], ...], subset: tuple[int, ...]) -> tuple[int, int, int, int]:
    return tuple(
        sum(momentum_list[index - 1][component] for index in subset)
        for component in range(4)
    )  # type: ignore[return-value]


def _direct_subset_values(
    momentum_list: tuple[tuple[int, int, int, int], ...],
) -> tuple[PhysicsProfile, dict[str, Fraction], dict[str, Fraction], dict[str, Fraction]]:
    m2 = tuple(_minkowski_square(momentum) for momentum in momentum_list)
    total = _add_momenta(momentum_list, (1, 2, 3, 4, 5))
    profile = PhysicsProfile.regular_3_plus_1(_minkowski_square(total), m2)
    values = {
        "s" + "".join(map(str, subset)): _minkowski_square(_add_momenta(momentum_list, subset))
        for subset in SUBSET_ORDER
    }
    x_values = {label: values[label] for label in X_ORDER}
    z_values = {label: values[label] for label in Z_ORDER}
    return profile, x_values, z_values, values


def _evaluate_affine(expression: object, values: dict[str, Fraction]) -> Fraction:
    # Keep this evaluator independent of the coordinate construction: the
    # row is evaluated from its public constant/coefficient payload only.
    affine = expression
    coefficients = affine.coefficients  # type: ignore[attr-defined]
    return affine.constant + sum(  # type: ignore[attr-defined]
        coefficient * values[name]
        for coefficient, name in zip(coefficients, Z_ORDER)
    )


def _reconstruct_z_from_pairs(profile: PhysicsProfile, x_values: dict[str, Fraction]) -> dict[str, Fraction]:
    masses = profile.m2
    return {
        "s12": x_values["s12"],
        "s23": x_values["s23"],
        "s34": x_values["s34"],
        "s45": x_values["s45"],
        "s1234": sum((x_values[label] for label in ("s12", "s13", "s14", "s23", "s24", "s34")), Fraction(0)) - 2 * sum(masses[:4], Fraction(0)),
        "s2345": sum((x_values[label] for label in ("s23", "s24", "s25", "s34", "s35", "s45")), Fraction(0)) - 2 * sum(masses[1:], Fraction(0)),
        "s123": x_values["s12"] + x_values["s13"] + x_values["s23"] - sum(masses[:3], Fraction(0)),
        "s234": x_values["s23"] + x_values["s24"] + x_values["s34"] - sum(masses[1:4], Fraction(0)),
        "s345": x_values["s34"] + x_values["s35"] + x_values["s45"] - sum(masses[2:], Fraction(0)),
    }


class IndependenceFoundationTests(unittest.TestCase):
    @staticmethod
    def profile() -> PhysicsProfile:
        return PhysicsProfile.regular_3_plus_1(
            "100",
            ("1", "4", "9", "16", "25"),
        )

    def test_exact_profile_rejects_binary_values_and_round_trips(self) -> None:
        profile = self.profile()
        self.assertEqual(PhysicsProfile.from_dict(profile.to_dict()), profile)
        for value in (0.5, True, False):
            with self.assertRaises((TypeError, ValueError)):
                as_fraction(value)  # type: ignore[arg-type]
        with self.assertRaises((TypeError, ValueError)):
            PhysicsProfile.regular_3_plus_1(100.0, (1, 4, 9, 16, 25))  # type: ignore[arg-type]

    def test_all_25_subset_rows_keep_fixed_order_and_exact_mass_corrections(self) -> None:
        affine = full_subset_affine_map(self.profile())
        self.assertEqual(len(SUBSET_ORDER), 25)
        self.assertEqual(affine.output_order, tuple("s" + "".join(map(str, subset)) for subset in SUBSET_ORDER))
        self.assertEqual(affine.z_order, Z_ORDER)
        self.assertEqual(affine.shape, (25, 9))
        self.assertTrue(all(len(row.coefficients) == 9 for row in affine.rows))
        # Pair and chosen z coordinates have zero affine mass offset by
        # construction; the remaining subset formulas retain exact unequal
        # mass corrections rather than silently dropping them.
        self.assertEqual(affine.rows[0].constant, Fraction(0))
        self.assertEqual(affine.rows[10].constant, Fraction(0))  # s123 = z7
        self.assertEqual(affine.rows[24].constant, Fraction(0))  # the last listed four-body subset is s2345 = z6
        nonzero_constants = [row.constant for row in affine.rows if row.constant]
        self.assertTrue(nonzero_constants)
        self.assertGreater(len(set(nonzero_constants)), 1)
        self.assertIn(Fraction(17), nonzero_constants)  # m1² + m4² in s14
        self.assertIn(Fraction(29), nonzero_constants)  # m2² + m5² in s25

    def test_all_25_affine_rows_match_direct_minkowski_squares_for_two_exact_fixtures(self) -> None:
        fixtures = (
            ((3, 1, 0, 0), (4, 0, 1, 0), (5, 0, 0, 2), (6, 1, 1, 1), (7, 2, 0, 1)),
            ((5, 1, 1, 0), (6, 0, 2, 1), (7, 2, 1, 1), (8, 1, 3, 0), (9, 0, 1, 2)),
        )
        for momenta in fixtures:
            profile, x_values, z_values, direct_values = _direct_subset_values(momenta)
            self.assertEqual(len(set(profile.m2)), 5)
            self.assertTrue(all(value > 0 for value in profile.m2))
            affine = full_subset_affine_map(profile)
            for subset, label, row in zip(SUBSET_ORDER, affine.output_order, affine.rows):
                self.assertEqual(label, "s" + "".join(map(str, subset)))
                self.assertEqual(_evaluate_affine(row, z_values), direct_values[label], label)

            pair_map = pair_coordinate_map(profile)
            reconstructed_x = {
                label: _evaluate_affine(row, z_values)
                for label, row in zip(X_ORDER, pair_map.rows)
            }
            self.assertEqual(reconstructed_x, x_values)

            pair_sum_coefficients = tuple(
                sum((row.coefficients[index] for row in pair_map.rows), Fraction(0))
                for index in range(9)
            )
            self.assertEqual(pair_sum_coefficients, (Fraction(0),) * 9)
            self.assertEqual(
                sum(pair_map.b, Fraction(0)),
                profile.M2 + 3 * sum(profile.m2, Fraction(0)),
            )
            self.assertEqual(
                sum(reconstructed_x.values(), Fraction(0)),
                profile.M2 + 3 * sum(profile.m2, Fraction(0)),
            )

            reconstructed_z = _reconstruct_z_from_pairs(profile, x_values)
            self.assertEqual(reconstructed_z, z_values)
            self.assertEqual(
                tuple(_evaluate_affine(row, reconstructed_z) for row in pair_map.rows),
                tuple(x_values[label] for label in X_ORDER),
            )

    def test_pair_map_has_the_declared_ten_by_nine_shape_and_exact_constants(self) -> None:
        pair_map = pair_coordinate_map(self.profile())
        self.assertEqual(pair_map.output_order, X_ORDER)
        self.assertEqual(pair_map.shape, (10, 9))
        # Simplified constants from the reviewed affine reverse map.
        self.assertEqual(pair_map.rows[2].constant, Fraction(17))
        self.assertEqual(pair_map.rows[6].constant, Fraction(29))
        self.assertEqual(pair_map.rows[3].constant, Fraction(126))

    def test_repeated_occurrence_is_preserved_in_label_map(self) -> None:
        repeated = (1, 2, 3)
        candidate = ReviewedCandidate(
            topology_key="topology-fixture",
            identity_key="identity-fixture",
            mode="fixed_labels",
            source_digest="source-fixture",
            occurrences=(
                InvariantOccurrence(repeated, 0, "edge0:0"),
                InvariantOccurrence(repeated, 1, "edge1:0"),
            ),
        )
        affine = build_label_affine_map(candidate, self.profile())
        self.assertEqual(affine.output_order, ("s123", "s123"))
        self.assertEqual(affine.source_occurrences, candidate.occurrences)
        self.assertEqual(affine.B[0], affine.B[1])
        self.assertEqual(affine.b[0], affine.b[1])

    def test_non_first_pivot_right_inverse_and_complete_column_kernel(self) -> None:
        matrix = ((0, 1, 0), (0, 0, 1))
        self.assertEqual(rank(matrix), 2)
        inverse = right_inverse(matrix)
        self.assertEqual(inverse, ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))))
        self.assertEqual(matmul(matrix, inverse), ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))))
        kernel = nullspace(matrix)
        self.assertEqual(kernel, ((Fraction(1), Fraction(0), Fraction(0)),))
        self.assertEqual(matmul(matrix, tuple(zip(*kernel))), ((Fraction(0),), (Fraction(0),)))

    def test_left_kernel_and_explicit_empty_matrix_cases(self) -> None:
        matrix = ((1, 0), (0, 1), (1, 1))
        self.assertEqual(left_nullspace(matrix), ((Fraction(-1), Fraction(-1), Fraction(1)),))
        empty_kernel = nullspace((), ncols=3)
        self.assertEqual(len(empty_kernel), 3)
        self.assertEqual(empty_kernel[0], (Fraction(1), Fraction(0), Fraction(0)))
        self.assertEqual(right_inverse((), ncols=3), ((), (), ()))

    def test_sparse_polynomial_exact_serialization_and_determinant(self) -> None:
        variables = ("x", "y")
        polynomial = SparsePolynomial(
            variables,
            (((2, 0), Fraction(3, 2)), ((0, 1), Fraction(-5, 3)), ((0, 0), Fraction(7))),
        )
        self.assertEqual(deserialize_polynomial(serialize_polynomial(polynomial)), polynomial)
        with self.assertRaises((TypeError, ValueError)):
            SparsePolynomial.from_dict({"variables": ["x"], "terms": {"1": 0.5}})  # type: ignore[arg-type]
        one = polynomial_constant(variables, 1)
        zero = polynomial_constant(variables, 0)
        self.assertEqual(determinant(((polynomial, zero), (zero, one))), polynomial)

    def test_synthetic_gcd_families_preserve_unsupported_boundary_or_exact_result(self) -> None:
        x_y = ("x", "y")
        x = SparsePolynomial(x_y, (((1, 0), Fraction(1)),))
        x_plus_y = SparsePolynomial(x_y, (((1, 0), Fraction(1)), ((0, 1), Fraction(1))))
        cylinder = SparsePolynomial(x_y, (((2, 0), Fraction(1)), ((1, 0), Fraction(1))))
        non_cylinder = SparsePolynomial(x_y, (((1, 0), Fraction(1)), ((0, 2), Fraction(1))))
        mixed = x.multiply(x_plus_y)
        repeated = x_plus_y.multiply(x_plus_y)

        gcd_cases = (
            ("cylinder", (cylinder, SparsePolynomial.zero(x_y)), cylinder),
            ("non_cylinder", (non_cylinder, SparsePolynomial(x_y, (((0, 1), Fraction(1)),))), SparsePolynomial(x_y, (((0, 0), Fraction(1)),))),
            ("mixed", (mixed, x), x),
            ("repeated", (repeated, x_plus_y), x_plus_y),
        )
        for name, inputs, expected in gcd_cases:
            result = polynomial_gcd(inputs)
            if result.status == "unsupported":
                self.assertIn("SymPy", result.reason or "", name)
            else:
                self.assertEqual(result.status, "available", name)
                self.assertEqual(result.value, expected, name)

        gradient_result = polynomial_gradient(repeated)
        square_free_result = square_free_part(repeated)
        composed = compose_linear(x_plus_y, ((1, 0), (0, 1)), target_variables=("u", "v"))
        cas_results = (gradient_result, square_free_result, composed)
        for result in cas_results:
            self.assertIn(result.status, {"available", "unsupported"})
            if result.status == "unsupported":
                self.assertIn("SymPy", result.reason or "")
        if all(result.status == "unsupported" for result in cas_results):
            self.skipTest(
                "SKIP: bundled Python has no SymPy; exact gradient/sqf/compose expectations were not executable"
            )
        self.assertEqual(gradient_result.status, "available")
        self.assertEqual(
            gradient_result.value,
            (
                SparsePolynomial(x_y, (((1, 0), Fraction(2)), ((0, 1), Fraction(2)))),
                SparsePolynomial(x_y, (((1, 0), Fraction(2)), ((0, 1), Fraction(2)))),
            ),
        )
        self.assertEqual(square_free_result.status, "available")
        self.assertEqual(square_free_result.value, x_plus_y)
        self.assertEqual(composed.status, "available")
        self.assertEqual(
            composed.value,
            SparsePolynomial(("u", "v"), (((1, 0), Fraction(1)), ((0, 1), Fraction(1)))),
        )


if __name__ == "__main__":
    unittest.main()
