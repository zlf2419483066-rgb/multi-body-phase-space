"""Five-body fixed-profile Gram construction.

The coordinate map and exact algebra backend are deliberately injected.  This
module owns the five-body Gram convention and strict rational threshold check;
it does not define a second polynomial representation or coordinate schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Any, Mapping, Sequence

from .contracts import PhysicsProfile, as_fraction, fraction_text


Z_VARIABLES = (
    "s12", "s23", "s34", "s45", "s1234", "s2345", "s123", "s234", "s345",
)
PAIR_VARIABLES = (
    "s12", "s13", "s14", "s15", "s23", "s24", "s25", "s34", "s35", "s45",
)
GRAM_SCHEMA_VERSION = "F5-I-gram-v1"


class GramUnsupported(ValueError):
    """The requested profile or backend is outside the implemented domain."""


def _backend(module: Any | None = None) -> Any:
    if module is not None:
        return module
    from . import exact_algebra  # type: ignore[import-not-found]
    return exact_algebra


def _rational(value: Any) -> Fraction:
    try:
        return as_fraction(value)
    except (TypeError, ValueError) as exc:
        raise GramUnsupported(str(exc)) from exc


def _mass_squares(profile: Any) -> tuple[Fraction, tuple[Fraction, ...]]:
    if not isinstance(profile, PhysicsProfile):
        raise GramUnsupported("profile must be the fixed PhysicsProfile contract")
    return _rational(profile.M2), tuple(_rational(value) for value in profile.m2)


def _sqrt_interval(value: Fraction, bits: int) -> tuple[Fraction, Fraction]:
    if value < 0:
        raise GramUnsupported("square-root interval requires a non-negative rational")
    if value == 0:
        return Fraction(0), Fraction(0)
    if bits < 1:
        raise ValueError("bits must be positive")
    scale = 1 << bits
    k = isqrt((value.numerator * scale * scale) // value.denominator)
    lower = Fraction(k, scale)
    if k * k * value.denominator == value.numerator * scale * scale:
        return lower, lower
    return lower, Fraction(k + 1, scale)


def validate_profile(profile: Any, *, bits: int = 128) -> dict[str, Any]:
    """Validate the supported fixed-QQ profile without floating tolerances.

    Threshold ambiguity is reported as ``unknown``.  It is never converted to
    a physical rejection, because a failed interval separation is not a proof
    that the requested profile is outside the physical domain.
    """

    try:
        total, masses = _mass_squares(profile)
        if profile.domain.threshold_status not in {"unsupported", "verified"}:
            return {"status": "unknown", "input_status": "unsupported",
                    "reason": "profile threshold_status is outside the contract"}
        if profile.domain.spacetime != "3+1" or profile.domain.metric != "(+---)" or profile.domain.kind != "full_regular_physical":
            return {"status": "unknown", "input_status": "unsupported",
                    "reason": "profile domain is outside the fixed regular 3+1 contract"}
        if profile.domain.dimension != 8 or profile.domain.nonempty_evidence != "analytic_t_family":
            return {"status": "unknown", "input_status": "unsupported",
                    "reason": "profile domain is not the declared non-empty eight-dimensional domain"}
        if profile.quantifier != "fixed_profile" or profile.coefficient_field != "Q" or profile.P_convention != "fixed_future_timelike_total":
            return {"status": "unknown", "input_status": "unsupported",
                    "reason": "profile quantifier, coefficient field, or P convention is unsupported"}
        if total <= 0 or any(value < 0 for value in masses):
            return {"status": "unknown", "input_status": "unsupported",
                    "reason": "mass-square sign conditions are not supported"}
        m_low, m_high = zip(*(_sqrt_interval(value, bits) for value in masses))
        total_low, total_high = _sqrt_interval(total, bits)
        lower_sum = sum(m_low, Fraction(0))
        upper_sum = sum(m_high, Fraction(0))
        if total_low > upper_sum:
            status = "valid"
            reason = "strict threshold separated by rational intervals"
        elif total_high <= lower_sum:
            status = "unknown"
            reason = "profile fails the strict threshold but is not a physical reject"
        else:
            status = "unknown"
            reason = "threshold interval remains unresolved at the requested precision"
        return {
            "status": status if status == "valid" else "unknown",
            "input_status": status if status == "valid" else "unsupported",
            "reason": reason,
            "bits": bits,
            "M2": total,
            "mass_squares": masses,
            "M_interval": (total_low, total_high),
            "mass_intervals": tuple(zip(m_low, m_high)),
            "sum_mass_lower": lower_sum,
            "sum_mass_upper": upper_sum,
        }
    except (GramUnsupported, TypeError, ValueError, ZeroDivisionError) as exc:
        return {"status": "unknown", "input_status": "unsupported", "reason": str(exc)}


def _constant(module: Any, value: Any, variables: Sequence[str] = Z_VARIABLES) -> Any:
    return module.polynomial_constant(tuple(variables), value)


def _add(module: Any, left: Any, right: Any) -> Any:
    return module.polynomial_add(left, right)


def _sub(module: Any, left: Any, right: Any) -> Any:
    return module.polynomial_subtract(left, right)


def _mul(module: Any, left: Any, right: Any) -> Any:
    return module.polynomial_multiply(left, right)


def _scale(module: Any, value: Any, scalar: Any) -> Any:
    return module.polynomial_scale(value, scalar)


def _determinant(module: Any, matrix: Sequence[Sequence[Any]]) -> Any:
    return module.determinant(matrix)


def _pair_key(key: Any) -> tuple[int, int]:
    if isinstance(key, str) and key.startswith("s") and len(key) == 3:
        return int(key[1]), int(key[2])
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return int(key[0]), int(key[1])
    raise GramUnsupported(f"unsupported pair coordinate key: {key!r}")


def gram_matrix_from_pairs(pair_coordinates: Mapping[Any, Any], mass_squares: Sequence[Any], *, algebra: Any | None = None) -> tuple[tuple[Any, ...], ...]:
    """Build G from all ten pair coordinates, preserving every occurrence map.

    ``pair_coordinates`` is supplied by ``coordinates.py``; this function does
    not infer or deduplicate variables.  Keys may be ``s12`` strings or integer
    pairs.  The diagonal convention is ``Gii=m_i²`` and the off-diagonal
    convention is ``Gij=(sij-mi²-mj²)/2``.
    """

    module = _backend(algebra)
    masses = tuple(_constant(module, _rational(value)) for value in mass_squares)
    if len(masses) != 5:
        raise GramUnsupported("Gram construction requires five mass squares")
    pairs = {_pair_key(key): value for key, value in pair_coordinates.items()}
    matrix: list[list[Any]] = []
    for i in range(1, 6):
        row: list[Any] = []
        for j in range(1, 6):
            if i == j:
                row.append(masses[i - 1])
            else:
                key = (min(i, j), max(i, j))
                if key not in pairs:
                    raise GramUnsupported(f"missing pair coordinate s{key[0]}{key[1]}")
                value = pairs[key]
                row.append(_scale(module, _sub(module, _sub(module, value, masses[i - 1]), masses[j - 1]), Fraction(1, 2)))
        matrix.append(row)
    return tuple(tuple(row) for row in matrix)


@dataclass(frozen=True, slots=True)
class GramData:
    matrix: tuple[tuple[Any, ...], ...]
    H: Any
    gradient: tuple[Any, ...]
    variables: tuple[Any, ...]
    profile_check: Mapping[str, Any]
    schema_version: str = GRAM_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "matrix": self.matrix,
            "H": self.H,
            "gradient": self.gradient,
            "variables": self.variables,
            "profile_check": dict(self.profile_check),
        }


def build_gram_data(
    profile: Any,
    pair_coordinates: Mapping[Any, Any],
    variables: Sequence[Any] = Z_VARIABLES,
    *,
    algebra: Any | None = None,
    threshold_bits: int = 128,
) -> GramData:
    """Construct G, H and ∇H for a fixed supported profile.

    The caller supplies the coordinate map generated by the foundation
    coordinate layer.  A threshold-unknown profile still produces algebraic
    data for inspection, but downstream physical conclusions must remain
    ``unknown``.
    """

    module = _backend(algebra)
    total, masses = _mass_squares(profile)
    del total
    check = validate_profile(profile, bits=threshold_bits)
    matrix = gram_matrix_from_pairs(pair_coordinates, masses, algebra=module)
    H = _determinant(module, matrix)
    gradient_result = module.gradient(H)
    if gradient_result.status != "available":
        raise GramUnsupported(gradient_result.reason or "exact polynomial gradient unavailable")
    gradient = tuple(gradient_result.value)
    if len(gradient) != len(tuple(variables)):
        raise GramUnsupported("gradient length does not match the supplied coordinate variables")
    return GramData(matrix, H, gradient, tuple(variables), check)


def build_gram_data_from_affine(
    profile: PhysicsProfile,
    pair_map: Any,
    *,
    algebra: Any | None = None,
    threshold_bits: int = 128,
) -> GramData:
    """Build Gram data from the foundation ``coordinates.AffineMap``.

    The affine map remains the sole owner of the x/z formulas.  This adapter
    only embeds each exact affine row into the backend's sparse polynomial
    type, preserving the declared nine-variable order and all constants.
    """

    module = _backend(algebra)
    from .exact_algebra import SparsePolynomial  # type: ignore[import-not-found]
    variables = tuple(getattr(pair_map, "z_order", Z_VARIABLES))
    if variables != tuple(Z_VARIABLES):
        raise GramUnsupported("five-body Gram construction requires the fixed z variable order")
    pair_coordinates: dict[str, Any] = {}
    for name, expression in zip(pair_map.output_order, pair_map.rows):
        terms: list[tuple[tuple[int, ...], Fraction]] = []
        if expression.constant:
            terms.append(((0,) * len(variables), expression.constant))
        for index, coefficient in enumerate(expression.coefficients):
            if coefficient:
                exponent = [0] * len(variables)
                exponent[index] = 1
                terms.append((tuple(exponent), coefficient))
        pair_coordinates[name] = SparsePolynomial(variables, tuple(terms))
    return build_gram_data(profile, pair_coordinates, variables, algebra=module,
                           threshold_bits=threshold_bits)


def check_gram_point(profile: PhysicsProfile, matrix: Sequence[Sequence[Any]], *, algebra: Any | None = None) -> dict[str, Any]:
    """Check all explicit Gram conditions for a fixed profile.

    This helper is an exact point check only.  It never upgrades a point to a
    general-domain conclusion; absent inertia support is explicitly unknown.
    """

    module = _backend(algebra)
    profile_check = validate_profile(profile)
    if profile_check.get("status") != "valid":
        return {"status": "unknown", "reason": "profile threshold/domain is not proven",
                "profile_check": profile_check}
    if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
        return {"status": "unknown", "reason": "Gram matrix must be 5x5"}
    expected_diagonal = tuple(profile.m2)
    diagonal_ok = tuple(matrix[index][index] for index in range(5)) == expected_diagonal
    total_ok = sum((sum(row) for row in matrix), Fraction(0)) == profile.M2
    rank = int(module.rank(matrix))
    inertia_result = module.inertia(matrix)
    if inertia_result.status != "available":
        return {"status": "unknown", "reason": inertia_result.reason or "exact Gram inertia is unsupported",
                "rank": rank, "profile_check": profile_check}
    inertia = tuple(inertia_result.value)
    row_sums = tuple(sum(row, Fraction(0)) for row in matrix)
    positive_rows = all(value > 0 for value in row_sums)
    valid = diagonal_ok and total_ok and rank == 4 and inertia == (1, 3, 1) and positive_rows
    return {"status": "pass" if valid else "unknown", "rank": rank,
            "inertia": inertia, "row_sums": row_sums, "positive_row_sums": positive_rows,
            "diagonal_ok": diagonal_ok, "total_ok": total_ok, "profile_check": profile_check}


__all__ = [
    "Z_VARIABLES", "PAIR_VARIABLES", "GRAM_SCHEMA_VERSION", "GramUnsupported",
    "GramData", "validate_profile", "gram_matrix_from_pairs", "build_gram_data",
    "build_gram_data_from_affine",
    "check_gram_point",
]
