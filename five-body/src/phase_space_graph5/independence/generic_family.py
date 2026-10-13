"""Generic positive-mass five-body independence family.

The family is parameterized by the nine fixed ``z`` coordinates and the six
squared-mass parameters.  It deliberately does not construct a
``PhysicsProfile`` or substitute numerical masses.  A source record supplies
the ordered invariant occurrences; the generic theorem supplies the
dimension bound and the exact affine rank certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Any, Mapping, Sequence

from .exact_algebra import columns_payload, left_nullspace, rank
from .coordinates import Z_ORDER
from .hajer_constraints import HAJER_VARIABLES, MASS_VARIABLES, inspect_invariant_set
from .invariant_sets import InvariantSet, extract_invariant_set
from .sources import FIXED_MODE, VerifiedSource, _jsonable, _key_token


GENERIC_FAMILY_SCHEMA_VERSION = "F5-I-generic-family-v1"
GENERIC_EVIDENCE_SCHEMA_VERSION = "F5-I-generic-evidence-v1"
GENERIC_THEOREM_VERSION = "F5-I-positive-mass-compact-projection-v1"
PARAMETER_ORDER = tuple(HAJER_VARIABLES)
Z_PARAMETER_ORDER = tuple(Z_ORDER)
GENERIC_MAX_DIMENSION = 8


@dataclass(frozen=True, slots=True)
class GenericFamilyDescriptor:
    """Closed descriptor for the strict positive-mass theorem domain."""

    spacetime: str = "3+1"
    metric: str = "+---"
    mass_condition: str = "mi>0"
    pairwise_distinct: bool = True
    threshold_condition: str = "M>sum(mi)"
    physical_scope: str = "full_regular_physical"
    claim: str = "generic_independence"
    quantifier: str = "all_declared_parameters_same_label"
    theorem_version: str = GENERIC_THEOREM_VERSION
    schema_version: str = GENERIC_FAMILY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.spacetime != "3+1":
            raise ValueError("generic family requires spacetime 3+1")
        if self.metric != "+---":
            raise ValueError("generic family requires metric +---")
        if self.mass_condition != "mi>0":
            raise ValueError("generic family requires strictly positive final-state masses")
        if self.pairwise_distinct is not True:
            raise ValueError("generic family requires pairwise-distinct masses")
        if self.threshold_condition != "M>sum(mi)":
            raise ValueError("generic family requires M>sum(mi)")
        if self.physical_scope != "full_regular_physical":
            raise ValueError("generic family requires the full regular physical scope")
        if self.claim != "generic_independence":
            raise ValueError("generic family claim is fixed")
        if self.quantifier != "all_declared_parameters_same_label":
            raise ValueError("generic family quantifier is fixed")
        if self.theorem_version != GENERIC_THEOREM_VERSION:
            raise ValueError("unsupported generic theorem version")
        if self.schema_version != GENERIC_FAMILY_SCHEMA_VERSION:
            raise ValueError("unsupported generic family schema version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "theorem_version": self.theorem_version,
            "spacetime": self.spacetime,
            "metric": self.metric,
            "mass_condition": self.mass_condition,
            "pairwise_distinct": self.pairwise_distinct,
            "threshold_condition": self.threshold_condition,
            "physical_scope": self.physical_scope,
            "claim": self.claim,
            "quantifier": self.quantifier,
        }


GENERIC_FAMILY = GenericFamilyDescriptor()


def _source_binding(source: VerifiedSource, invariant_set: InvariantSet) -> dict[str, object]:
    return {
        "source_kind": source.source_kind,
        "source_index": source.source_index,
        "mode": source.mode,
        "topology_key": _jsonable(source.topology_key),
        "source_digest": invariant_set.source_digest,
    }


def _identity_binding(source: VerifiedSource, invariant_set: InvariantSet) -> dict[str, object]:
    return {
        "identity_key": _jsonable(source.key),
        "identity_digest": invariant_set.identity_digest,
    }


def _determinant(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    rows = [list(row) for row in matrix]
    size = len(rows)
    if size == 0:
        return Fraction(1)
    if any(len(row) != size for row in rows):
        raise ValueError("minor must be square")
    result = Fraction(1)
    for column in range(size):
        pivot = next((index for index in range(column, size) if rows[index][column]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            result = -result
        value = rows[column][column]
        result *= value
        for index in range(column + 1, size):
            factor = rows[index][column] / value
            if factor:
                rows[index] = [left - factor * right for left, right in zip(rows[index], rows[column])]
    return result


def _nonzero_minor(
    matrix: tuple[tuple[Fraction, ...], ...], order: int,
) -> dict[str, object]:
    if order == 0:
        return {"order": 0, "row_indices": [], "column_indices": [], "value": "1"}
    row_count = len(matrix)
    column_count = len(matrix[0]) if matrix else len(PARAMETER_ORDER)
    for row_indices in combinations(range(row_count), order):
        for column_indices in combinations(range(column_count), order):
            minor = tuple(
                tuple(matrix[row][column] for column in column_indices)
                for row in row_indices
            )
            value = _determinant(minor)
            if value:
                return {
                    "order": order,
                    "row_indices": list(row_indices),
                    "column_indices": list(column_indices),
                    "value": str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}",
                }
    raise ValueError(f"no nonzero minor found for claimed rank {order}")


def _mass_constant_relations(
    matrix: tuple[tuple[Fraction, ...], ...], kernel: tuple[tuple[Fraction, ...], ...],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for weights in kernel:
        weighted = tuple(
            sum((weights[row] * matrix[row][column] for row in range(len(matrix))), Fraction(0))
            for column in range(len(PARAMETER_ORDER))
        )
        result.append({
            "weights": [str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}" for value in weights],
            "parameter_coefficients": [str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}" for value in weighted],
            "mass_variables": list(MASS_VARIABLES),
            "mass_coefficients": [
                str(weighted[index].numerator) if weighted[index].denominator == 1 else f"{weighted[index].numerator}/{weighted[index].denominator}"
                for index in range(len(PARAMETER_ORDER) - len(MASS_VARIABLES), len(PARAMETER_ORDER))
            ],
        })
    return result


def _certificate(
    matrix: tuple[tuple[Fraction, ...], ...],
    affine_parameter_matrix: tuple[tuple[Fraction, ...], ...],
    dimension: int,
    affine_rank: int,
) -> dict[str, object]:
    if dimension > GENERIC_MAX_DIMENSION:
        return {
            "kind": "dimension_bound",
            "requested_dimension": dimension,
            "maximum_regular_dimension": GENERIC_MAX_DIMENSION,
            "theorem_version": GENERIC_THEOREM_VERSION,
        }
    if affine_rank < dimension:
        kernel = left_nullspace(matrix, ncols=len(Z_PARAMETER_ORDER))
        return {
            "kind": "left_kernel",
            "left_kernel": columns_payload(kernel),
            "mass_constant_relations": _mass_constant_relations(affine_parameter_matrix, kernel),
        }
    return {
        "kind": "nonzero_minor",
        "minor": _nonzero_minor(matrix, affine_rank),
    }


def evaluate_generic_label(source: VerifiedSource) -> dict[str, object]:
    """Produce generic-family evidence for one reviewed fixed-label source."""

    if not isinstance(source, VerifiedSource):
        raise TypeError("source must be a VerifiedSource")
    if source.mode != FIXED_MODE:
        raise ValueError("generic positive-mass family supports fixed_labels only")
    invariant_set = extract_invariant_set(source)
    inspected = inspect_invariant_set(invariant_set)
    if tuple(inspected["variables"]) != PARAMETER_ORDER:
        raise ValueError("Hajer inspection returned an unexpected parameter order")
    raw_parameter_matrix = inspected["affine_coefficient_matrix"]
    affine_parameter_matrix = tuple(
        tuple(Fraction(value) for value in row) for row in raw_parameter_matrix  # type: ignore[arg-type]
    )
    matrix = tuple(row[:len(Z_PARAMETER_ORDER)] for row in affine_parameter_matrix)
    dimension = len(matrix)
    affine_rank = rank(matrix, ncols=len(Z_PARAMETER_ORDER))
    generic_rank = min(affine_rank, GENERIC_MAX_DIMENSION)
    status = "pass" if dimension == affine_rank and dimension <= GENERIC_MAX_DIMENSION else "reject"
    checks = [
        "source_revalidated",
        "ordered_occurrences_preserved",
        "hajer_cqct_parameter_basis",
        "exact_affine_rank",
        "generic_dimension_bound",
    ]
    return {
        "schema_version": GENERIC_EVIDENCE_SCHEMA_VERSION,
        "family": GENERIC_FAMILY.to_dict(),
        "theorem_version": GENERIC_THEOREM_VERSION,
        "source": _source_binding(source, invariant_set),
        "identity": _identity_binding(source, invariant_set),
        "topology_key": _key_token(source.topology_key),
        "identity_key": _key_token(source.key),
        "source_digest": invariant_set.source_digest,
        "identity_digest": invariant_set.identity_digest,
        "occurrences": [item.to_dict() for item in invariant_set.occurrences],
        "unique_subsets": [list(item) for item in invariant_set.unique_subsets],
        "duplicate_groups": [list(item) for item in invariant_set.duplicate_groups],
        "parameter_order": list(PARAMETER_ORDER),
        "B": [[str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}" for value in row] for row in matrix],
        "affine_parameter_matrix": [
            [str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}" for value in row]
            for row in affine_parameter_matrix
        ],
        "dimension": dimension,
        "affine_rank": affine_rank,
        "generic_rank": generic_rank,
        "status": status,
        "input_status": "valid",
        "execution_status": "complete",
        "checks": checks,
        "certificate": _certificate(matrix, affine_parameter_matrix, dimension, affine_rank),
    }


__all__ = [
    "GENERIC_FAMILY_SCHEMA_VERSION", "GENERIC_EVIDENCE_SCHEMA_VERSION",
    "GENERIC_THEOREM_VERSION", "PARAMETER_ORDER", "Z_PARAMETER_ORDER", "GENERIC_MAX_DIMENSION",
    "GenericFamilyDescriptor", "GENERIC_FAMILY", "evaluate_generic_label",
]
