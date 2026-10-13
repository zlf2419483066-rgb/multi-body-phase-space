"""Hajer 2025 five-body Gram-constraint descriptor.

This module records the symbolic constraint used as an invariant-kinematics
diagnostic.  It does not select masses, impose a threshold, or evaluate the
old independence criterion.  The descriptor keeps the nine existing
``Z_ORDER`` variables and six squared-mass variables (``M2`` plus five
particle masses) separate, and only
evaluates the two small numeric determinants when an explicit complete point
is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .coordinates import Z_ORDER
from .contracts import as_fraction, fraction_text
from .exact_algebra import SparsePolynomial, polynomial_constant, rank
from .invariant_sets import InvariantSet


HAJER_SCHEMA_VERSION = "F5-Hajer-constraint-v1"
MASS_VARIABLES = ("M2", "m1_2", "m2_2", "m3_2", "m4_2", "m5_2")
HAJER_VARIABLES = tuple(Z_ORDER) + MASS_VARIABLES
REFERENCE = {
    "author": "Jan Hajer",
    "title": "Invariant relativistic kinematics: Phase space triangulation",
    "doi": "10.1103/g4ck-sk4s",
    "equation": "Eq. 7.5",
    "zeta": "+1",
    "constraint": "det(Q)=0",
    "identity": "det(CM)=32 det(Q)",
    "boundary": "diagnostic Gram constraint only; independence_status remains not_evaluated",
}


def _zero(variables: Sequence[str] = HAJER_VARIABLES) -> SparsePolynomial:
    return polynomial_constant(tuple(variables), 0)


def _one(variables: Sequence[str] = HAJER_VARIABLES) -> SparsePolynomial:
    return polynomial_constant(tuple(variables), 1)


def _variable(name: str) -> SparsePolynomial:
    if name not in HAJER_VARIABLES:
        raise ValueError(f"unknown Hajer variable: {name}")
    exponent = [0] * len(HAJER_VARIABLES)
    exponent[HAJER_VARIABLES.index(name)] = 1
    return SparsePolynomial(HAJER_VARIABLES, ((tuple(exponent), Fraction(1)),))


def _add(left: SparsePolynomial, right: SparsePolynomial) -> SparsePolynomial:
    return left.add(right)


def _sub(left: SparsePolynomial, right: SparsePolynomial) -> SparsePolynomial:
    return left.subtract(right)


def _scale(value: SparsePolynomial, scalar: int | Fraction) -> SparsePolynomial:
    return value.scale(scalar)


def _subset_name(subset: Sequence[int]) -> str:
    values = tuple(subset)
    if not values:
        raise ValueError("an invariant subset cannot be empty")
    if values == tuple(range(1, 6)):
        return "M2"
    if len(values) == 1:
        return f"m{values[0]}_2"
    return "s" + "".join(str(value) for value in values)


def _subset_square(subset: Sequence[int]) -> SparsePolynomial:
    values = tuple(subset)
    if values != tuple(range(values[0], values[-1] + 1)):
        raise ValueError(f"Hajer canonical subset must be contiguous: {values!r}")
    return _variable(_subset_name(values))


def _cumulative_square(index: int) -> SparsePolynomial:
    if index < 1 or index > 5:
        raise ValueError("cumulative momentum index must lie in 1..5")
    return _subset_square(tuple(range(1, index + 1)))


def _distance_square(left: int, right: int) -> SparsePolynomial:
    if left == right:
        return _zero()
    if left == 0:
        return _cumulative_square(right)
    if right == 0:
        return _cumulative_square(left)
    lower, upper = sorted((left, right))
    if lower < 1 or upper > 5:
        raise ValueError("cumulative momentum index must lie in 0..5")
    return _subset_square(tuple(range(lower + 1, upper + 1)))


def _build_q_matrix() -> tuple[tuple[SparsePolynomial, ...], ...]:
    rows: list[tuple[SparsePolynomial, ...]] = []
    for i in range(1, 6):
        row: list[SparsePolynomial] = []
        qi2 = _cumulative_square(i)
        for j in range(1, 6):
            if i == j:
                row.append(qi2)
                continue
            qj2 = _cumulative_square(j)
            dij2 = _distance_square(i, j)
            row.append(_scale(_sub(_add(qi2, qj2), dij2), Fraction(1, 2)))
        rows.append(tuple(row))
    return tuple(rows)


def _build_cm_matrix() -> tuple[tuple[SparsePolynomial, ...], ...]:
    rows: list[tuple[SparsePolynomial, ...]] = []
    for i in range(7):
        row: list[SparsePolynomial] = []
        for j in range(7):
            if i == 0 and j == 0:
                row.append(_zero())
            elif i == 0 or j == 0:
                row.append(_one())
            elif i == j:
                row.append(_zero())
            else:
                # CM row/column 1 is the physical zero point; rows 2..6 are
                # q1..q5.  The leading row/column 0 is the Cayley--Menger
                # affine border and is not a physical point.
                row.append(_distance_square(i - 1, j - 1))
        rows.append(tuple(row))
    return tuple(rows)


def _matrix_to_dict(matrix: Sequence[Sequence[SparsePolynomial]]) -> list[list[dict[str, object]]]:
    return [[entry.to_dict() for entry in row] for row in matrix]


@dataclass(frozen=True, slots=True)
class HajerConstraint:
    """Immutable symbolic CM/Q descriptor for the fixed Hajer convention."""

    variables: tuple[str, ...]
    q_matrix: tuple[tuple[SparsePolynomial, ...], ...]
    cm_matrix: tuple[tuple[SparsePolynomial, ...], ...]
    reference: Mapping[str, str]
    schema_version: str = HAJER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "q_matrix", tuple(tuple(row) for row in self.q_matrix))
        object.__setattr__(self, "cm_matrix", tuple(tuple(row) for row in self.cm_matrix))
        if self.schema_version != HAJER_SCHEMA_VERSION:
            raise ValueError("unsupported Hajer constraint schema")
        if self.variables != HAJER_VARIABLES:
            raise ValueError("Hajer variables must preserve Z_ORDER followed by five squared masses")
        if len(self.q_matrix) != 5 or any(len(row) != 5 for row in self.q_matrix):
            raise ValueError("Q must be a 5x5 matrix")
        if len(self.cm_matrix) != 7 or any(len(row) != 7 for row in self.cm_matrix):
            raise ValueError("CM must be a 7x7 matrix")
        if any(entry.variables != HAJER_VARIABLES for matrix in (self.q_matrix, self.cm_matrix) for row in matrix for entry in row):
            raise ValueError("symbolic entries must use the fixed fifteen-variable order")
        object.__setattr__(self, "reference", MappingProxyType(dict(self.reference)))

    @property
    def Q(self) -> tuple[tuple[SparsePolynomial, ...], ...]:
        return self.q_matrix

    @property
    def CM(self) -> tuple[tuple[SparsePolynomial, ...], ...]:
        return self.cm_matrix

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "variables": list(self.variables),
            "reference": dict(self.reference),
            "Q": _matrix_to_dict(self.q_matrix),
            "CM": _matrix_to_dict(self.cm_matrix),
        }


def build_hajer_constraint() -> HajerConstraint:
    """Build the symbolic Eq.7.5 constraint without a mass profile."""

    return HajerConstraint(
        variables=HAJER_VARIABLES,
        q_matrix=_build_q_matrix(),
        cm_matrix=_build_cm_matrix(),
        reference=REFERENCE,
    )


def _determinant_exact(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    rows = [list(row) for row in matrix]
    size = len(rows)
    if size == 0 or any(len(row) != size for row in rows):
        raise ValueError("exact determinant requires a non-empty square matrix")
    determinant = Fraction(1)
    for column in range(size):
        pivot = next((index for index in range(column, size) if rows[index][column]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        pivot_value = rows[column][column]
        determinant *= pivot_value
        for index in range(column + 1, size):
            factor = rows[index][column] / pivot_value
            if factor:
                rows[index] = [left - factor * right for left, right in zip(rows[index], rows[column])]
    return determinant


def _evaluate(polynomial: SparsePolynomial, values: Mapping[str, Fraction]) -> Fraction:
    total = Fraction(0)
    for monomial, coefficient in polynomial.terms:
        term = coefficient
        for variable, exponent in zip(polynomial.variables, monomial):
            term *= values[variable] ** exponent
        total += term
    return total


def _evaluate_matrix(matrix: Sequence[Sequence[SparsePolynomial]], values: Mapping[str, Fraction]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(_evaluate(entry, values) for entry in row) for row in matrix)


def _subset_expression(descriptor: HajerConstraint, subset: Sequence[int]) -> SparsePolynomial:
    values = tuple(subset)
    if len(values) not in {2, 3, 4} or tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("invariant subsets must be sorted unique two-, three-, or four-particle tuples")
    if values[0] < 1 or values[-1] > 5:
        raise ValueError("invariant subset particle IDs must lie in 1..5")
    indicator = tuple(1 if particle in values else 0 for particle in range(1, 6))
    coefficients = tuple(indicator[index] - indicator[index + 1] for index in range(4)) + (indicator[4],)
    expression = _zero()
    for row, left in enumerate(coefficients):
        for column, right in enumerate(coefficients):
            if left and right:
                expression = expression.add(descriptor.q_matrix[row][column].scale(left * right))
    return expression


def _linear_coefficients(polynomial: SparsePolynomial) -> tuple[Fraction, ...]:
    if any(sum(exponent) > 1 for exponent, _ in polynomial.terms):
        raise ValueError("Hajer invariant expression is not affine")
    zero = (0,) * len(HAJER_VARIABLES)
    by_exponent = dict(polynomial.terms)
    return tuple(by_exponent.get(tuple(1 if index == position else 0 for index in range(len(HAJER_VARIABLES))), Fraction(0))
                   for position in range(len(HAJER_VARIABLES)))


def inspect_invariant_set(invariant_set: InvariantSet) -> dict[str, object]:
    """Bind ordered candidate occurrences to Hajer's symbolic Q descriptor.

    This is a structural handoff only.  It preserves duplicates and source
    identity, emits the nine-column ``z`` coefficient matrix, and reports
    ``requires_completion`` for a set shorter than the nine variables.  It
    never labels a set independent or dependent.
    """

    if not isinstance(invariant_set, InvariantSet):
        raise TypeError("inspect_invariant_set expects an InvariantSet")
    descriptor = build_hajer_constraint()
    rows: list[dict[str, object]] = []
    z_rows: list[tuple[str, ...]] = []
    affine_rows: list[tuple[str, ...]] = []
    for occurrence in invariant_set.occurrences:
        expression = _subset_expression(descriptor, occurrence.particle_subset)
        coefficients = _linear_coefficients(expression)
        z_coefficients = coefficients[:len(Z_ORDER)]
        rows.append({
            **occurrence.to_dict(),
            "variables": list(HAJER_VARIABLES),
            "expression": expression.to_dict(),
            "affine_coefficients": [fraction_text(value) for value in coefficients],
            "z_coefficients": [fraction_text(value) for value in z_coefficients],
        })
        z_rows.append(tuple(z_coefficients))
        affine_rows.append(tuple(coefficients))
    occurrence_count = len(rows)
    return {
        "schema_version": HAJER_SCHEMA_VERSION,
        "source_digest": invariant_set.source_digest,
        "identity_digest": invariant_set.identity_digest,
        "mode": invariant_set.mode,
        "topology_key": invariant_set.topology_key,
        "identity_key": invariant_set.identity_key,
        "occurrences": rows,
        "unique_subsets": [list(item) for item in invariant_set.unique_subsets],
        "duplicate_groups": [list(item) for item in invariant_set.duplicate_groups],
        "variables": list(HAJER_VARIABLES),
        "z_order": list(Z_ORDER),
        "z_coefficient_matrix": [[fraction_text(value) for value in row] for row in z_rows],
        "affine_coefficient_matrix": [[fraction_text(value) for value in row] for row in affine_rows],
        "affine_rank": rank(z_rows, ncols=len(Z_ORDER)),
        "gram_schema_version": descriptor.schema_version,
        "gram_reference": dict(descriptor.reference),
        "constraint_status": "requires_completion" if occurrence_count < len(Z_ORDER) else "not_evaluated",
        "independence_status": "not_evaluated",
    }


def check_hajer_constraint(
    values: Mapping[str, int | str | Fraction],
) -> dict[str, object]:
    """Evaluate ``det(Q)=0`` for one complete exact fifteen-variable point.

    Missing variables produce an explicit ``unknown`` result.  Booleans,
    floats, malformed rational text, and unexpected fields are rejected as
    invalid input.  The result deliberately keeps ``independence_status`` at
    ``not_evaluated``: the Gram relation is not an independence or physical
    feasibility verdict.
    """

    if not isinstance(values, Mapping):
        raise TypeError("Hajer values must be a mapping")
    descriptor = build_hajer_constraint()
    missing = [name for name in descriptor.variables if name not in values]
    extra = sorted(str(name) for name in values if name not in descriptor.variables)
    base: dict[str, object] = {
        "schema_version": descriptor.schema_version,
        "variables": list(descriptor.variables),
        "independence_status": "not_evaluated",
    }
    if extra:
        raise ValueError(f"unexpected Hajer variables: {extra!r}")
    if missing:
        return {**base, "status": "unknown", "reason": "missing exact variables", "missing": missing, "extra": []}
    exact_values = {name: as_fraction(values[name]) for name in descriptor.variables}
    q_numeric = _evaluate_matrix(descriptor.q_matrix, exact_values)
    cm_numeric = _evaluate_matrix(descriptor.cm_matrix, exact_values)
    rawdet = _determinant_exact(q_numeric)
    cm_det = _determinant_exact(cm_numeric)
    identitycheck = cm_det == 32 * rawdet
    result = {
        **base,
        "status": ("satisfied" if rawdet == 0 else "violated") if identitycheck else "unknown",
        "reason": None if identitycheck else "CM/Q determinant identity failed",
        "rawdet": fraction_text(rawdet),
        "CMdet": fraction_text(cm_det),
        "detQ": fraction_text(rawdet),
        "detCM": fraction_text(cm_det),
        "identitycheck": identitycheck,
        "missing": [],
        "extra": [],
    }
    return result


__all__ = [
    "HAJER_SCHEMA_VERSION", "MASS_VARIABLES", "HAJER_VARIABLES", "REFERENCE",
    "HajerConstraint", "build_hajer_constraint", "check_hajer_constraint", "inspect_invariant_set",
]
