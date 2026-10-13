"""Helicity-style 3+2 coordinates wrapping the bounded B/F03 sample chart.

For generic points, the three overall angles are ``(theta0,phi0,psi)`` and
the two internal coordinates are ``(u,phi4_star)``, with
``u=cos(alpha)=p1hat(q12-rest) dot n0``.  This is a local chart on the same
five-dimensional angular space as :mod:`b_direct`; it is not a conversion to
the particle-1/2 Euler chart used in some source descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

from .b_direct import BDirectSample, sample_b_f03

Vector3 = tuple[float, float, float]
HelicityMapStatus = Literal["mapped", "degenerate", "unphysical"]
HelicitySampleStatus = Literal["physical", "unsupported", "degenerate", "unphysical"]

B_HELICITY_PROFILE = "b_f03_helicity_3plus2"
B_HELICITY_VARIABLES = ("theta0", "phi0", "psi", "u", "phi4_star")
B_HELICITY_MEASURE_IDENTITY = (
    "u=cos(alpha); dOmega_1=dpsi |du|=sin(alpha)dalpha dpsi; "
    "|d(cos(theta1),phi1)/d(u,psi)|=1 away from chart degeneracies"
)
B_HELICITY_ANGLE_DEFINITIONS = (
    "n0 is the q12 direction in the P rest frame",
    "u=cos(alpha) is the q12-rest p1 direction cosine relative to n0",
    "e_theta0,e_phi0,n0 is a right-handed frame; psi is p1 azimuth about n0 in that frame",
    "phi4_star is inherited from the B/F03 Q-rest p2* axis and p1*-projection zero axis",
    "This 3+2 chart is not the particle-1/2 overall Euler chart",
)

_TOL = 2.0e-12
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True, slots=True)
class BHelicityMapResult:
    """Explicit map between helicity ``(u,psi)`` and fixed ``(theta1,phi1)``."""

    status: HelicityMapStatus
    diagnostics: tuple[str, ...]
    theta0: float | None = None
    phi0: float | None = None
    u: float | None = None
    psi: float | None = None
    theta1: float | None = None
    phi1: float | None = None
    n0: Vector3 | None = None
    n1: Vector3 | None = None
    measure_jacobian_abs: float | None = None

    @property
    def mapped(self) -> bool:
        return self.status in {"mapped", "degenerate"} and self.theta1 is not None and self.phi1 is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": B_HELICITY_PROFILE,
            "status": self.status,
            "diagnostics": list(self.diagnostics),
            "coordinates": {
                "theta0": self.theta0,
                "phi0": self.phi0,
                "psi": self.psi,
                "u": self.u,
                "theta1": self.theta1,
                "phi1": self.phi1,
            },
            "n0": list(self.n0) if self.n0 is not None else None,
            "n1": list(self.n1) if self.n1 is not None else None,
            "measure_jacobian_abs": self.measure_jacobian_abs,
            "measure_identity": B_HELICITY_MEASURE_IDENTITY,
        }


@dataclass(frozen=True, slots=True)
class BHelicitySample:
    """A helicity-coordinate point and its underlying B/F03 sample result."""

    status: HelicitySampleStatus
    coordinate_status: HelicityMapStatus
    diagnostics: tuple[str, ...]
    angle_map: BHelicityMapResult
    phi4_star: float
    direct_sample: BDirectSample | None
    normalization: str

    @property
    def density(self) -> float | None:
        return self.direct_sample.density if self.direct_sample is not None else None

    @property
    def momenta(self):
        return self.direct_sample.momenta if self.direct_sample is not None else None

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "b_helicity_sample",
            "profile": B_HELICITY_PROFILE,
            "status": self.status,
            "coordinate_status": self.coordinate_status,
            "normalization": self.normalization,
            "diagnostics": list(self.diagnostics),
            "angle_variables": list(B_HELICITY_VARIABLES),
            "angle_definitions": list(B_HELICITY_ANGLE_DEFINITIONS),
            "angle_map": self.angle_map.to_dict(),
            "phi4_star": self.phi4_star,
            "density": self.density,
            "direct_sample": self.direct_sample.to_dict() if self.direct_sample else None,
            "measure_identity": B_HELICITY_MEASURE_IDENTITY,
        }


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _basis(theta0: float, phi0: float) -> tuple[Vector3, Vector3, Vector3]:
    st, ct = math.sin(theta0), math.cos(theta0)
    sp, cp = math.sin(phi0), math.cos(phi0)
    n0 = (st * cp, st * sp, ct)
    e_theta0 = (ct * cp, ct * sp, -st)
    e_phi0 = (-sp, cp, 0.0)
    return n0, e_theta0, e_phi0


def _dot3(left: Vector3, right: Vector3) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _norm3(vector: Vector3) -> float:
    return math.sqrt(max(0.0, _dot3(vector, vector)))


def _unit_direction(theta: float, phi: float) -> Vector3:
    return (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )


def _fixed_angle_degeneracies(theta0: float, theta1: float, u: float) -> tuple[str, ...]:
    reasons: list[str] = []
    if abs(math.sin(theta0)) <= _TOL:
        reasons.append("theta0 is at a polar overall-angle boundary; phi0 is redundant")
    if abs(math.sin(theta1)) <= _TOL:
        reasons.append("theta1 is at a fixed-axis polar boundary; phi1 is redundant")
    if 1.0 - abs(u) <= _TOL:
        reasons.append("u=+/-1 makes psi redundant because p1 is parallel to n0")
    return tuple(reasons)


def helicity_to_fixed_angles(
    theta0: float,
    phi0: float,
    u: float,
    psi: float,
) -> BHelicityMapResult:
    """Map ``(u,psi)`` to fixed-axis q12-rest ``(theta1,phi1)`` explicitly."""

    values = tuple(_finite(item) for item in (theta0, phi0, u, psi))
    if any(item is None for item in values):
        return BHelicityMapResult("unphysical", ("all helicity angles must be finite real numbers",))
    theta0, phi0, u, psi = values  # type: ignore[misc]
    if not 0.0 <= theta0 <= math.pi:
        return BHelicityMapResult("unphysical", ("theta0 must lie in [0,pi]",))
    if not 0.0 <= phi0 < _TWO_PI:
        return BHelicityMapResult("unphysical", ("phi0 must lie in [0,2*pi)",))
    if not -1.0 <= u <= 1.0:
        return BHelicityMapResult("unphysical", ("u=cos(alpha) must lie in [-1,1]",))
    if not 0.0 <= psi < _TWO_PI:
        return BHelicityMapResult("unphysical", ("psi must lie in [0,2*pi)",))

    n0, e_theta0, e_phi0 = _basis(theta0, phi0)
    transverse = math.sqrt(max(0.0, 1.0 - u * u))
    n1 = tuple(
        u * n0[index]
        + transverse * (math.cos(psi) * e_theta0[index] + math.sin(psi) * e_phi0[index])
        for index in range(3)
    )
    n1 = tuple(component / _norm3(n1) for component in n1)
    theta1 = math.acos(min(1.0, max(-1.0, n1[2])))
    xy_norm = math.hypot(n1[0], n1[1])
    phi1 = 0.0 if xy_norm <= _TOL else math.atan2(n1[1], n1[0]) % _TWO_PI
    degeneracies = _fixed_angle_degeneracies(theta0, theta1, u)
    return BHelicityMapResult(
        "degenerate" if degeneracies else "mapped",
        degeneracies,
        theta0,
        phi0,
        u,
        psi,
        theta1,
        phi1,
        n0,
        n1,
        None if degeneracies else 1.0,
    )


def fixed_to_helicity_angles(
    theta0: float,
    phi0: float,
    theta1: float,
    phi1: float,
) -> BHelicityMapResult:
    """Invert fixed-axis q12-rest ``(theta1,phi1)`` into ``(u,psi)``."""

    values = tuple(_finite(item) for item in (theta0, phi0, theta1, phi1))
    if any(item is None for item in values):
        return BHelicityMapResult("unphysical", ("all fixed and overall angles must be finite real numbers",))
    theta0, phi0, theta1, phi1 = values  # type: ignore[misc]
    if not 0.0 <= theta0 <= math.pi or not 0.0 <= theta1 <= math.pi:
        return BHelicityMapResult("unphysical", ("theta0 and theta1 must lie in [0,pi]",))
    if not 0.0 <= phi0 < _TWO_PI or not 0.0 <= phi1 < _TWO_PI:
        return BHelicityMapResult("unphysical", ("phi0 and phi1 must lie in [0,2*pi)",))

    n0, e_theta0, e_phi0 = _basis(theta0, phi0)
    n1 = _unit_direction(theta1, phi1)
    u = min(1.0, max(-1.0, _dot3(n1, n0)))
    transverse_theta = _dot3(n1, e_theta0)
    transverse_phi = _dot3(n1, e_phi0)
    transverse_norm = math.hypot(transverse_theta, transverse_phi)
    degeneracies = list(_fixed_angle_degeneracies(theta0, theta1, u))
    if transverse_norm <= _TOL:
        psi = None
    else:
        psi = math.atan2(transverse_phi, transverse_theta) % _TWO_PI
    return BHelicityMapResult(
        "degenerate" if degeneracies else "mapped",
        tuple(degeneracies),
        theta0,
        phi0,
        u,
        psi,
        theta1,
        phi1,
        n0,
        n1,
        None if degeneracies else 1.0,
    )


def sample_b_f03_helicity(
    record: object,
    total_mass: float,
    masses: Sequence[float],
    s12: float,
    s34: float,
    s23: float,
    theta0: float,
    phi0: float,
    psi: float,
    u: float,
    phi4_star: float,
    *,
    normalization: str = "jing_psf",
) -> BHelicitySample:
    """Evaluate the B/F03 sample using three overall and two internal angles."""

    angle_map = helicity_to_fixed_angles(theta0, phi0, u, psi)
    if not angle_map.mapped:
        return BHelicitySample(
            "unphysical",
            angle_map.status,
            angle_map.diagnostics,
            angle_map,
            phi4_star,
            None,
            normalization,
        )
    if angle_map.theta1 is None or angle_map.phi1 is None:
        return BHelicitySample(
            "unphysical",
            "unphysical",
            ("helicity map did not produce fixed-axis angles",),
            angle_map,
            phi4_star,
            None,
            normalization,
        )

    direct = sample_b_f03(
        record,
        total_mass,
        masses,
        s12,
        s34,
        s23,
        theta0,
        phi0,
        angle_map.theta1,
        angle_map.phi1,
        phi4_star,
        normalization=normalization,
    )
    if direct.status in {"unsupported", "unphysical"}:
        status: HelicitySampleStatus = direct.status
    elif angle_map.status == "degenerate" or direct.status == "degenerate":
        status = "degenerate"
    else:
        status = "physical"
    return BHelicitySample(
        status,
        angle_map.status,
        angle_map.diagnostics + direct.diagnostics,
        angle_map,
        phi4_star,
        direct,
        normalization,
    )


__all__ = [
    "BHelicityMapResult",
    "BHelicitySample",
    "B_HELICITY_ANGLE_DEFINITIONS",
    "B_HELICITY_MEASURE_IDENTITY",
    "B_HELICITY_PROFILE",
    "B_HELICITY_VARIABLES",
    "fixed_to_helicity_angles",
    "helicity_to_fixed_angles",
    "sample_b_f03_helicity",
]
