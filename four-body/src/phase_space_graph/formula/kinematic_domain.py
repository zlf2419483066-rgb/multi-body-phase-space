"""Input-domain checks shared by the direct and reference kinematic paths.

This module deliberately contains no density formula, momentum inverse, or
Jacobian code.  It only validates the finite numerical inputs needed by the
two independently implemented paths.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


ANGLE_PAIRS = (("theta0", "phi0"), ("thetaA", "phiA"), ("thetaB", "phiB"))
DOMAIN_TOLERANCE = 1e-12


def _finite_float(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _safe_square(name: str, value: float) -> float:
    result = value * value
    if not math.isfinite(result):
        raise ValueError(f"{name} is outside finite floating-point range")
    return result


def kallen(a: float, b: float, c: float) -> float:
    """Return lambda(a,b,c), rejecting non-representable finite inputs."""

    try:
        result = a * a + b * b + c * c - 2.0 * (a * b + a * c + b * c)
    except OverflowError as exc:
        raise ValueError("Kallen function is outside finite floating-point range") from exc
    if not math.isfinite(result):
        raise ValueError("Kallen function is outside finite floating-point range")
    return result


def kallen_tolerance(*values: float) -> float:
    scale = max((abs(value) for value in values), default=1.0)
    return DOMAIN_TOLERANCE * scale**4


def kallen_root(value: float, *scale_values: float) -> float:
    """Take a Kallen square root after an explicit domain check.

    A tiny negative caused by floating-point cancellation is treated as the
    exact boundary only when it is within the scale-relative tolerance.
    Materially negative values always raise instead of becoming an artificial
    zero.
    """

    if not math.isfinite(value):
        raise ValueError("Kallen function is not finite")
    if value < -kallen_tolerance(*scale_values):
        raise ValueError("Kallen function is negative outside the physical domain")
    return math.sqrt(0.0 if value < 0.0 else value)


def validate_masses(masses: Sequence[float], total_mass: float) -> tuple[tuple[float, ...], float]:
    try:
        mass_values = tuple(masses)
    except (TypeError, ValueError) as exc:
        raise ValueError("exactly four particle masses are required") from exc
    if len(mass_values) != 4:
        raise ValueError("exactly four particle masses are required")
    checked = tuple(_finite_float(f"m_{index}", value) for index, value in enumerate(mass_values, 1))
    if any(value < 0.0 for value in checked):
        raise ValueError("particle masses must be non-negative")
    parent = _finite_float("total_mass", total_mass)
    if parent <= 0.0:
        raise ValueError("total_mass must be positive; a zero parent is degenerate")
    threshold = sum(checked)
    if parent < threshold - DOMAIN_TOLERANCE * max(threshold, 1e-300):
        raise ValueError("total_mass is below the four-particle threshold")
    _safe_square("total_mass", parent)
    for index, value in enumerate(checked, 1):
        _safe_square(f"m_{index}", value)
    return checked, parent


def validate_angles(values: Mapping[str, float], angle_names: Sequence[str]) -> None:
    names = tuple(angle_names)
    if len(names) != 6:
        raise ValueError("a four-body chart requires three theta/phi angle pairs")
    for name in names:
        _finite_float(name, values[name])
    pairs = tuple(zip(names[::2], names[1::2]))
    for theta_name, phi_name in pairs:
        if theta_name not in values or phi_name not in values:
            raise ValueError(f"missing angle pair: {theta_name}, {phi_name}")
        theta = _finite_float(theta_name, values[theta_name])
        phi = _finite_float(phi_name, values[phi_name])
        if not 0.0 <= theta <= math.pi:
            raise ValueError(f"{theta_name} must satisfy 0 <= theta <= pi")
        if not 0.0 <= phi < 2.0 * math.pi:
            raise ValueError(f"{phi_name} must satisfy 0 <= phi < 2*pi")


def validate_pair_domain(
    values: Mapping[str, float],
    pair_names: Sequence[str],
    masses: Sequence[float],
    total_mass: float,
    angle_names: Sequence[str],
) -> dict[str, float]:
    """Validate the pair chart's complete numerical input domain.

    Threshold equalities are retained as explicit zero-density boundaries.
    A zero invariant mass is rejected because the corresponding rest frame
    and angular chart do not exist.  Massless particles and positive-mass
    massless-pair interior points remain valid.
    """

    checked_masses, parent = validate_masses(masses, total_mass)
    data: dict[str, float] = {}
    for name in tuple(pair_names) + tuple(angle_names):
        if name not in values:
            raise ValueError(f"missing chart coordinate: {name}")
        data[name] = _finite_float(name, values[name])
    validate_angles(data, angle_names)
    if len(pair_names) != 2:
        raise ValueError("the pair chart requires exactly two invariant coordinates")
    first, second = tuple(pair_names)
    sa, sb = data[first], data[second]
    if sa <= 0.0 or sb <= 0.0:
        raise ValueError("zero or negative subsystem invariant is a degenerate chart")
    ma, mb = math.sqrt(sa), math.sqrt(sb)
    pair_a = tuple(int(value) for value in first[1:])
    pair_b = tuple(int(value) for value in second[1:])
    if len(pair_a) != 2 or len(pair_b) != 2 or any(index < 1 or index > 4 for index in pair_a + pair_b):
        raise ValueError("pair chart labels must be s_ij with particle indices 1..4")
    threshold_a = checked_masses[pair_a[0] - 1] + checked_masses[pair_a[1] - 1]
    threshold_b = checked_masses[pair_b[0] - 1] + checked_masses[pair_b[1] - 1]
    threshold_scale = max(threshold_a, threshold_b, ma, mb, 1e-300)
    if ma < threshold_a - DOMAIN_TOLERANCE * threshold_scale:
        raise ValueError(f"{first} is below its two-particle threshold")
    if mb < threshold_b - DOMAIN_TOLERANCE * threshold_scale:
        raise ValueError(f"{second} is below its two-particle threshold")
    if ma + mb > parent + DOMAIN_TOLERANCE * max(parent, ma + mb, 1e-300):
        raise ValueError(f"sqrt({first})+sqrt({second}) exceeds total_mass")
    parent_sq = _safe_square("total_mass", parent)
    sa_sq = _safe_square(first, ma)
    sb_sq = _safe_square(second, mb)
    for label, value, roots in (
        ("parent", kallen(parent_sq, sa, sb), (parent, ma, mb)),
        (first, kallen(sa, checked_masses[pair_a[0] - 1] ** 2, checked_masses[pair_a[1] - 1] ** 2), (ma, checked_masses[pair_a[0] - 1], checked_masses[pair_a[1] - 1])),
        (second, kallen(sb, checked_masses[pair_b[0] - 1] ** 2, checked_masses[pair_b[1] - 1] ** 2), (mb, checked_masses[pair_b[0] - 1], checked_masses[pair_b[1] - 1])),
    ):
        if value < -kallen_tolerance(*roots):
            raise ValueError(f"{label} Kallen function is negative outside the physical domain")
    # Keep these local checks visible in the validation path; the values are
    # used by the independent density and inverse implementations separately.
    if not math.isfinite(sa_sq) or not math.isfinite(sb_sq):
        raise ValueError("subsystem invariant is outside finite floating-point range")
    return data


__all__ = ["ANGLE_PAIRS", "DOMAIN_TOLERANCE", "kallen", "kallen_tolerance", "kallen_root", "validate_masses", "validate_angles", "validate_pair_domain"]
