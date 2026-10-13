"""Independent verifier for generic positive-mass evidence.

The verifier re-derives ordered occurrences from the reviewed source and
rebuilds the 15-column symbolic affine formulas from the pair-coordinate
identities.  Only the first nine ``z`` columns are the kinematic matrix whose
rank is certified; the six mass columns are symbolic constants and are used
only to check the corresponding affine relation.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Any, Mapping, Sequence

from .coordinates import Z_ORDER
from .contracts import as_fraction, fraction_text
from .exact_algebra import rank
from .hajer_constraints import HAJER_VARIABLES, MASS_VARIABLES, inspect_invariant_set
from .invariant_sets import extract_invariant_set
from .sources import FIXED_MODE, VerifiedSource, _jsonable, _key_token
from .generic_family import (
    GENERIC_EVIDENCE_SCHEMA_VERSION,
    GENERIC_FAMILY,
    GENERIC_MAX_DIMENSION,
    GENERIC_THEOREM_VERSION,
    PARAMETER_ORDER,
    Z_PARAMETER_ORDER,
)


def _zero() -> list[Fraction]:
    return [Fraction(0)] * len(PARAMETER_ORDER)


def _basis(index: int) -> list[Fraction]:
    result = _zero()
    result[index] = Fraction(1)
    return result


def _add(left: Sequence[Fraction], right: Sequence[Fraction], scale: int = 1) -> list[Fraction]:
    return [a + Fraction(scale) * b for a, b in zip(left, right)]


def _mass_sum(particles: Sequence[int]) -> list[Fraction]:
    result = _zero()
    for particle in particles:
        index = PARAMETER_ORDER.index(f"m{particle}_2")
        result[index] += 1
    return result


def _pair_rows() -> dict[str, tuple[Fraction, ...]]:
    rows: dict[str, list[Fraction]] = {}
    rows["s12"] = _basis(0)
    rows["s23"] = _basis(1)
    rows["s34"] = _basis(2)
    rows["s45"] = _basis(3)
    rows["s13"] = _add(_add(_basis(6), _basis(0), -1), _basis(1), -1)
    rows["s13"] = _add(rows["s13"], _mass_sum((1, 2, 3)))
    rows["s24"] = _add(_add(_basis(7), _basis(1), -1), _basis(2), -1)
    rows["s24"] = _add(rows["s24"], _mass_sum((2, 3, 4)))
    rows["s35"] = _add(_add(_basis(8), _basis(2), -1), _basis(3), -1)
    rows["s35"] = _add(rows["s35"], _mass_sum((3, 4, 5)))

    row = _basis(4)
    row = _add(row, _mass_sum((1, 2, 3, 4)), 2)
    for label in ("s12", "s13", "s23", "s24", "s34"):
        row = _add(row, rows[label], -1)
    rows["s14"] = row

    row = _basis(5)
    row = _add(row, _mass_sum((2, 3, 4, 5)), 2)
    for label in ("s23", "s24", "s34", "s35", "s45"):
        row = _add(row, rows[label], -1)
    rows["s25"] = row

    row = _basis(9)
    row = _add(row, _mass_sum((1, 2, 3, 4, 5)), 3)
    for label in ("s12", "s13", "s14", "s23", "s24", "s25", "s34", "s35", "s45"):
        row = _add(row, rows[label], -1)
    rows["s15"] = row
    return {label: tuple(value) for label, value in rows.items()}


def _subset_row(subset: Sequence[int], pairs: Mapping[str, tuple[Fraction, ...]]) -> tuple[Fraction, ...]:
    values = tuple(subset)
    if len(values) not in {2, 3, 4} or tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("invariant subsets must be sorted unique two-, three-, or four-particle tuples")
    if values[0] < 1 or values[-1] > 5:
        raise ValueError("invariant subset particle IDs must lie in 1..5")
    result = _zero()
    for left, right in combinations(values, 2):
        result = _add(result, pairs[f"s{left}{right}"])
    result = _add(result, _mass_sum(values), -(len(values) - 2))
    return tuple(result)


def _independent_parameter_matrix(invariant_set: Any) -> tuple[tuple[Fraction, ...], ...]:
    pairs = _pair_rows()
    return tuple(_subset_row(item.particle_subset, pairs) for item in invariant_set.occurrences)


def _parse_matrix(value: object, *, rows: int, columns: int) -> tuple[tuple[Fraction, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("matrix must be a sequence of rows")
    if len(value) != rows:
        raise ValueError("matrix row count mismatch")
    parsed: list[tuple[Fraction, ...]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != columns:
            raise ValueError("matrix column count mismatch")
        parsed_row: list[Fraction] = []
        for item in row:
            parsed_value = as_fraction(item)  # rejects float/bool
            if isinstance(item, str) and fraction_text(parsed_value) != item:
                raise ValueError("matrix rational text is not canonical")
            parsed_row.append(parsed_value)
        parsed.append(tuple(parsed_row))
    return tuple(parsed)


def _determinant(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    rows = [list(row) for row in matrix]
    size = len(rows)
    if size == 0:
        return Fraction(1)
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


def _expected_minor(matrix: tuple[tuple[Fraction, ...], ...], order: int) -> dict[str, object]:
    if order == 0:
        return {"order": 0, "row_indices": [], "column_indices": [], "value": "1"}
    for row_indices in combinations(range(len(matrix)), order):
        for column_indices in combinations(range(len(Z_PARAMETER_ORDER)), order):
            value = _determinant(tuple(tuple(matrix[row][column] for column in column_indices) for row in row_indices))
            if value:
                return {
                    "order": order,
                    "row_indices": list(row_indices),
                    "column_indices": list(column_indices),
                    "value": fraction_text(value),
                }
    raise ValueError("no nonzero minor for recomputed rank")


def _binding(source: VerifiedSource, invariant_set: Any) -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "source_kind": source.source_kind,
            "source_index": source.source_index,
            "mode": source.mode,
            "topology_key": _jsonable(source.topology_key),
            "source_digest": invariant_set.source_digest,
        },
        {
            "identity_key": _jsonable(source.key),
            "identity_digest": invariant_set.identity_digest,
        },
    )


def _invalid(errors: list[str], *, checks: list[str] | None = None) -> dict[str, object]:
    return {
        "valid": False,
        "status": "invalid",
        "checks": checks or [],
        "errors": errors,
    }


def _strict_equal(left: object, right: object) -> bool:
    """Compare JSON-shaped evidence without Python bool/int coercion."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, int) or isinstance(right, int):
        return type(left) is type(right) and left == right
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, type(right)) or len(left) != len(right):
            return False
        return all(_strict_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _expected_certificate(
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
    if affine_rank == dimension:
        return {"kind": "nonzero_minor", "minor": _expected_minor(matrix, affine_rank)}
    # Recompute a basis independently using a small exact elimination in the
    # shared backend only for the 9-column kinematic matrix.
    from .exact_algebra import left_nullspace

    kernel = left_nullspace(matrix, ncols=len(Z_PARAMETER_ORDER))
    relations: list[dict[str, object]] = []
    for weights in kernel:
        weighted = tuple(
            sum((weights[row] * affine_parameter_matrix[row][column] for row in range(dimension)), Fraction(0))
            for column in range(len(PARAMETER_ORDER))
        )
        relations.append({
            "weights": [fraction_text(value) for value in weights],
            "parameter_coefficients": [fraction_text(value) for value in weighted],
            "mass_variables": list(MASS_VARIABLES),
            "mass_coefficients": [fraction_text(weighted[index]) for index in range(9, 15)],
        })
    return {
        "kind": "left_kernel",
        "left_kernel": [[fraction_text(value) for value in column] for column in kernel],
        "mass_constant_relations": relations,
    }


def verify_generic_label(source: VerifiedSource, evidence: Mapping[str, object]) -> dict[str, object]:
    """Verify generic evidence without using the producer as an oracle."""

    if not isinstance(source, VerifiedSource):
        return _invalid(["source must be a VerifiedSource"])
    if not isinstance(evidence, Mapping):
        return _invalid(["evidence must be a mapping"])
    required = {
        "schema_version", "family", "theorem_version", "source", "identity",
        "topology_key", "identity_key", "source_digest", "identity_digest",
        "occurrences", "unique_subsets", "duplicate_groups", "parameter_order",
        "B", "affine_parameter_matrix", "dimension", "affine_rank", "generic_rank",
        "status", "input_status", "execution_status", "checks", "certificate",
    }
    errors: list[str] = []
    missing = sorted(required - set(evidence))
    unknown = sorted(set(evidence) - required)
    if missing:
        errors.append(f"missing required evidence fields: {missing}")
    if unknown:
        errors.append(f"unexpected evidence fields: {unknown}")
    if errors:
        return _invalid(errors)
    try:
        invariant_set = extract_invariant_set(source)
        expected_source, expected_identity = _binding(source, invariant_set)
        if evidence["schema_version"] != GENERIC_EVIDENCE_SCHEMA_VERSION:
            errors.append("generic evidence schema mismatch")
        if not _strict_equal(evidence["family"], GENERIC_FAMILY.to_dict()):
            errors.append("generic family descriptor mismatch")
        if evidence["theorem_version"] != GENERIC_THEOREM_VERSION:
            errors.append("theorem version mismatch")
        if not _strict_equal(evidence["source"], expected_source):
            errors.append("source key/digest binding mismatch")
        if not _strict_equal(evidence["identity"], expected_identity):
            errors.append("identity key/digest binding mismatch")
        if evidence["topology_key"] != _key_token(source.topology_key):
            errors.append("topology key binding mismatch")
        if evidence["identity_key"] != _key_token(source.key):
            errors.append("identity key token mismatch")
        if evidence["source_digest"] != invariant_set.source_digest:
            errors.append("source digest token mismatch")
        if evidence["identity_digest"] != invariant_set.identity_digest:
            errors.append("identity digest token mismatch")
        expected_occurrences = [item.to_dict() for item in invariant_set.occurrences]
        if not _strict_equal(evidence["occurrences"], expected_occurrences):
            errors.append("ordered occurrence binding mismatch")
        if not _strict_equal(evidence["unique_subsets"], [list(item) for item in invariant_set.unique_subsets]):
            errors.append("unique subset binding mismatch")
        if not _strict_equal(evidence["duplicate_groups"], [list(item) for item in invariant_set.duplicate_groups]):
            errors.append("duplicate occurrence grouping mismatch")
        if not _strict_equal(evidence["parameter_order"], list(PARAMETER_ORDER)):
            errors.append("15-column parameter order mismatch")
        dimension = len(invariant_set.occurrences)
        expected_parameter_matrix = _independent_parameter_matrix(invariant_set)
        expected_z_matrix = tuple(row[:len(Z_PARAMETER_ORDER)] for row in expected_parameter_matrix)
        parsed_parameter_matrix = _parse_matrix(evidence["affine_parameter_matrix"], rows=dimension, columns=len(PARAMETER_ORDER))
        parsed_z_matrix = _parse_matrix(evidence["B"], rows=dimension, columns=len(Z_PARAMETER_ORDER))
        if parsed_parameter_matrix != expected_parameter_matrix:
            errors.append("independent 15-column affine reconstruction mismatch")
        if parsed_z_matrix != expected_z_matrix:
            errors.append("independent 9-column B reconstruction mismatch")
        inspected = inspect_invariant_set(invariant_set)
        inspected_parameter_matrix = _parse_matrix(inspected["affine_coefficient_matrix"], rows=dimension, columns=len(PARAMETER_ORDER))
        if inspected_parameter_matrix != expected_parameter_matrix:
            errors.append("Hajer cQct cross-check mismatch")
        if evidence["dimension"] != dimension or isinstance(evidence["dimension"], bool) or not isinstance(evidence["dimension"], int):
            errors.append("dimension mismatch")
        expected_rank = rank(expected_z_matrix, ncols=len(Z_PARAMETER_ORDER))
        if evidence["affine_rank"] != expected_rank or isinstance(evidence["affine_rank"], bool) or not isinstance(evidence["affine_rank"], int):
            errors.append("affine rank mismatch")
        expected_generic_rank = min(expected_rank, GENERIC_MAX_DIMENSION)
        if evidence["generic_rank"] != expected_generic_rank or isinstance(evidence["generic_rank"], bool) or not isinstance(evidence["generic_rank"], int):
            errors.append("generic rank mismatch")
        expected_status = "pass" if dimension == expected_rank and dimension <= GENERIC_MAX_DIMENSION else "reject"
        if evidence["status"] != expected_status:
            errors.append("status mismatch")
        if evidence["input_status"] != "valid":
            errors.append("input_status must be valid")
        if evidence["execution_status"] != "complete":
            errors.append("execution_status must be complete")
        checks = evidence["checks"]
        if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)) or any(not isinstance(item, str) for item in checks):
            errors.append("checks must be a string sequence")
        expected_certificate = _expected_certificate(expected_z_matrix, expected_parameter_matrix, dimension, expected_rank)
        if not _strict_equal(evidence["certificate"], expected_certificate):
            errors.append("certificate mismatch")
        if errors:
            return _invalid(errors, checks=["source_reextracted", "independent_reconstruction"])
        return {
            "valid": True,
            "status": expected_status,
            "checks": [
                "source_reextracted", "identity_binding", "ordered_occurrences_binding",
                "independent_pair_affine_reconstruction", "hajer_cqct_crosscheck",
                "z_rank_recomputed", "certificate_recomputed",
            ],
            "errors": [],
            "schema_version": GENERIC_EVIDENCE_SCHEMA_VERSION,
            "input_status": "valid",
            "execution_status": "complete",
        }
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return _invalid([f"evidence verification failed: {type(exc).__name__}: {exc}"])


__all__ = ["verify_generic_label"]
