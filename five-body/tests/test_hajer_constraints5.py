from __future__ import annotations

from fractions import Fraction
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase_space_graph5.independence.hajer_constraints import (
    HAJER_VARIABLES,
    build_hajer_constraint,
    check_hajer_constraint,
    inspect_invariant_set,
)
from phase_space_graph5.independence.invariant_sets import extract_invariant_set
from phase_space_graph5.independence.sources import VerifiedSource
from phase_space_graph5.fixtures import fixture_b
from phase_space_graph5.label_identity import canonicalize_candidate_labeling
from phase_space_graph5.physical_labels import normalize_candidate_labeling


def _dot(left: tuple[int, ...], right: tuple[int, ...]) -> Fraction:
    return Fraction(left[0] * right[0] - sum(a * b for a, b in zip(left[1:], right[1:])))


def _determinant(matrix: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    rows = [list(row) for row in matrix]
    result = Fraction(1)
    for column in range(len(rows)):
        pivot = next((index for index in range(column, len(rows)) if rows[index][column]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            result = -result
        value = rows[column][column]
        result *= value
        for index in range(column + 1, len(rows)):
            factor = rows[index][column] / value
            rows[index] = [left - factor * right for left, right in zip(rows[index], rows[column])]
    return result


def _evaluate(entry: object, values: dict[str, Fraction]) -> Fraction:
    total = Fraction(0)
    for monomial, coefficient in entry.terms:  # type: ignore[attr-defined]
        term = coefficient
        for variable, exponent in zip(entry.variables, monomial):  # type: ignore[attr-defined]
            term *= values[variable] ** exponent
        total += term
    return total


def _numeric_matrix(matrix: object, values: dict[str, Fraction]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(_evaluate(entry, values) for entry in row) for row in matrix)  # type: ignore[union-attr]


def _point_values() -> tuple[dict[str, Fraction], tuple[tuple[int, ...], ...]]:
    momenta = (
        (2, 1, 0, 0),
        (3, -1, 0, 0),
        (4, 0, 1, 0),
        (5, 0, 0, 1),
        (6, 0, -1, -1),
    )
    values: dict[str, Fraction] = {
        "M2": _dot(tuple(sum(momentum[index] for momentum in momenta) for index in range(4)),
                    tuple(sum(momentum[index] for momentum in momenta) for index in range(4)))
    }
    for index, momentum in enumerate(momenta, start=1):
        values[f"m{index}_2"] = _dot(momentum, momentum)
    for size in (2, 3, 4):
        for subset in itertools.combinations(range(1, 6), size):
            label = "s" + "".join(str(index) for index in subset)
            total = tuple(sum(momenta[index - 1][component] for index in subset) for component in range(4))
            if label in HAJER_VARIABLES:
                values[label] = _dot(total, total)
    return values, momenta


class HajerConstraintTests(unittest.TestCase):
    def test_descriptor_has_fixed_fifteen_variables_and_immutable_shape(self) -> None:
        descriptor = build_hajer_constraint()
        self.assertEqual(descriptor.variables, HAJER_VARIABLES)
        self.assertEqual(len(descriptor.q_matrix), 5)
        self.assertEqual(len(descriptor.cm_matrix), 7)
        self.assertEqual(descriptor.reference["equation"], "Eq. 7.5")
        with self.assertRaises(TypeError):
            descriptor.reference["equation"] = "changed"  # type: ignore[index]

    def test_exact_four_momentum_point_satisfies_and_matches_direct_gram(self) -> None:
        values, momenta = _point_values()
        descriptor = build_hajer_constraint()
        result = check_hajer_constraint(values)
        self.assertEqual(result["status"], "satisfied")
        self.assertEqual(result["rawdet"], "0")
        self.assertEqual(result["CMdet"], "0")
        self.assertTrue(result["identitycheck"])
        self.assertEqual(result["independence_status"], "not_evaluated")

        cumulative = tuple(
            tuple(sum(momenta[index][component] for index in range(end)) for component in range(4))
            for end in range(1, 6)
        )
        q_matrix = tuple(tuple(_dot(left, right) for right in cumulative) for left in cumulative)
        direct_cm_points = ((0, 0, 0, 0),) + cumulative
        cm_matrix = tuple(
            tuple(
                Fraction(0) if row == column == 0 else
                Fraction(1) if row == 0 or column == 0 else
                Fraction(0) if row == column else
                _dot(tuple(a - b for a, b in zip(direct_cm_points[row - 1], direct_cm_points[column - 1])),
                     tuple(a - b for a, b in zip(direct_cm_points[row - 1], direct_cm_points[column - 1])))
                for column in range(7)
            )
            for row in range(7)
        )
        self.assertEqual(_numeric_matrix(descriptor.q_matrix, values), q_matrix)
        self.assertEqual(_numeric_matrix(descriptor.cm_matrix, values), cm_matrix)
        self.assertEqual(_determinant(q_matrix), Fraction(0))
        self.assertEqual(_determinant(cm_matrix), 32 * _determinant(q_matrix))

    def test_identity_negative_point_is_violated_without_physical_claim(self) -> None:
        values = {
            "M2": 1,
            "m1_2": 1,
            "m2_2": 2,
            "m3_2": 2,
            "m4_2": 2,
            "m5_2": 2,
            "s12": 1,
            "s23": 2,
            "s34": 2,
            "s45": 2,
            "s1234": 1,
            "s2345": 2,
            "s123": 1,
            "s234": 2,
            "s345": 2,
        }
        result = check_hajer_constraint(values)
        self.assertEqual(result["status"], "violated")
        self.assertEqual(result["rawdet"], "1")
        self.assertEqual(result["CMdet"], "32")
        self.assertTrue(result["identitycheck"])
        self.assertEqual(result["independence_status"], "not_evaluated")

    def test_missing_and_non_exact_values_are_not_defaulted(self) -> None:
        values, _ = _point_values()
        missing = dict(values)
        del missing["s345"]
        result = check_hajer_constraint(missing)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["missing"], ["s345"])
        self.assertNotIn("rawdet", result)
        with self.assertRaises(TypeError):
            check_hajer_constraint({**values, "s345": True})
        with self.assertRaises(TypeError):
            check_hajer_constraint({**values, "s345": 1.5})
        with self.assertRaises(ValueError):
            check_hajer_constraint({**values, "extra": 1})

    def test_b2_occurrences_bind_to_cqct_and_preserve_duplicate_group(self) -> None:
        topology, labeling = fixture_b(2)
        checked = normalize_candidate_labeling(topology, labeling)
        self.assertIsNotNone(checked.record)
        record = checked.record
        identity = canonicalize_candidate_labeling(record, "fixed_labels")
        source = VerifiedSource(identity, record, "fixture-b2", 0, None)
        invariant_set = extract_invariant_set(source)
        inspection = inspect_invariant_set(invariant_set)
        self.assertEqual(
            [row["particle_subset"] for row in inspection["occurrences"]],
            [[1, 3, 4, 5], [3, 4, 5], [3, 4, 5], [4, 5]],
        )
        self.assertEqual(inspection["duplicate_groups"], [[1, 2]])
        self.assertEqual(inspection["affine_rank"], 3)
        self.assertEqual(inspection["constraint_status"], "requires_completion")
        self.assertEqual(inspection["independence_status"], "not_evaluated")
        values, _ = _point_values()
        expected = [288, 225, 225, 120]
        for row, target in zip(inspection["occurrences"], expected):
            affine_value = sum(
                Fraction(coefficient) * values[name]
                for name, coefficient in zip(HAJER_VARIABLES, row["affine_coefficients"])
            )
            self.assertEqual(affine_value, target)


if __name__ == "__main__":
    unittest.main()
