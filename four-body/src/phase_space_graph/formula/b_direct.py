"""Bounded B/F03 single-loop direct analytic IKV sample profile.

This module implements only the locally reduced B/F03 chart described by the
2026-09-23 implementation specification.  It does not build a FormulaArtifact
or certify a global covariant identity.  The physical record eligibility check
uses the B connectivity, particle assignments, and ordered invariants; it does
not use a graph number or an identity digest.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Literal, Sequence

from ..domain import EdgeType
from ..physical_labels import PhysicalLabelRecord
from .kinematic_domain import kallen

Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]
BDirectStatus = Literal["physical", "unsupported", "degenerate", "unphysical"]

B_F03_PROFILE = "b_f03_single_loop_local_sample"
B_F03_ORDERED_INVARIANTS = ((1, 2), (3, 4), (2, 3))
B_F03_ANGLE_VARIABLES = ("theta0", "phi0", "theta1", "phi1", "phi4_star")
B_F03_ANGLE_DEFINITIONS = (
    "(theta0,phi0): q12 direction in the P rest frame",
    "(theta1,phi1): p1 direction in the q12 rest frame, using axes transported from the P rest frame by the no-rotation boost",
    "phi4_star: p4 azimuth in the Q rest frame about p2*, with zero axis from boosted p1* projected perpendicular to p2*",
    "theta0/theta1 poles are spherical-coordinate degeneracies; |c0|=1 makes phi4_star physically redundant",
    "This profile does not convert these five angles to a particle-1/2 overall Euler chart.",
)
B_F03_DENSITY_EXPRESSION = (
    "k12*k1 / ((2*pi)^12 * 128 * M * m12 * sqrt(S) * k2)"
)
B_F03_STANDARD_DENSITY_EXPRESSION = (
    "k12*k1 / ((2*pi)^8 * 128 * M * m12 * sqrt(S) * k2) "
    "= (2*pi)^4 * rho_BF03^J"
)
B_F03_DIFFERENTIAL_MEASURE = (
    "dPhi4^J = rho_BF03 ds12 ds34 ds23 "
    "dOmega_q12(theta0,phi0) dOmega_p1(theta1,phi1) dphi4*; "
    "dOmega=sin(theta)dtheta dphi; phi4* is the azimuth about p2* "
    "with zero axis from boosted p1 projected perpendicular to p2*"
)
B_F03_NORMALIZATION_IDENTITY = (
    "[(2*pi)^-6*k12/(4*M)]*[(2*pi)^-6*k1/(4*m12)]*"
    "[(2*pi)^-6*k4/(4*sqrt(S))]*(2*pi)^9*(2*pi)^-3/(2*k2*k4) "
    "= k12*k1/((2*pi)^12*128*M*m12*sqrt(S)*k2)"
)
B_F03_STANDARD_DIFFERENTIAL_MEASURE = (
    "dPhi4^S = rho_BF03^S ds12 ds34 ds23 "
    "dOmega_q12(theta0,phi0) dOmega_p1(theta1,phi1) dphi4*; "
    "dOmega=sin(theta)dtheta dphi; dPhi4^S=(2*pi)^4 dPhi4^J; "
    "phi4* is the azimuth about p2* with zero axis from boosted p1 projected perpendicular to p2*"
)
B_F03_STANDARD_NORMALIZATION_IDENTITY = (
    "rho_BF03^S=(2*pi)^4*rho_BF03^J "
    "= k12*k1/((2*pi)^8*128*M*m12*sqrt(S)*k2)"
)
SUPPORTED_NORMALIZATIONS = ("jing_psf", "standard_2pi4")

_REL_TOL = 2.0e-10
_DOMAIN_TOL = 2.0e-12


@dataclass(frozen=True, slots=True)
class BDirectEligibility:
    """Structural eligibility result for the exact B/F03 marking."""

    eligible: bool
    reasons: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return "eligible" if self.eligible else "unsupported"

    def to_dict(self) -> dict[str, object]:
        return {"eligible": self.eligible, "status": self.status, "reasons": list(self.reasons)}


@dataclass(frozen=True, slots=True)
class BDirectSample:
    """One chart point, including explicit branch and domain diagnostics."""

    profile: str
    status: BDirectStatus
    diagnostics: tuple[str, ...]
    ordered_invariants: tuple[str, ...] = ("s12", "s34", "s23")
    angle_variables: tuple[str, ...] = B_F03_ANGLE_VARIABLES
    differential_measure: str = B_F03_DIFFERENTIAL_MEASURE
    density_expression: str = B_F03_DENSITY_EXPRESSION
    normalization_identity: str = B_F03_NORMALIZATION_IDENTITY
    angle_definitions: tuple[str, ...] = B_F03_ANGLE_DEFINITIONS
    normalization: str = "jing_psf"
    total_mass: float | None = None
    masses: tuple[float, float, float, float] | None = None
    invariant_values: tuple[float, float, float] | None = None
    angle_values: tuple[float, float, float, float, float] | None = None
    branch_count: int = 0
    root_condition: str = "-1 <= c0 <= 1 and F(c0)=0"
    root_cosine: float | None = None
    root_residual: float | None = None
    c_reconstructed: float | None = None
    S: float | None = None
    k12: float | None = None
    k1: float | None = None
    k2: float | None = None
    k4: float | None = None
    density: float | None = None
    momenta: tuple[Vector4, Vector4, Vector4, Vector4] | None = None
    q12: Vector4 | None = None
    q34: Vector4 | None = None
    q23: Vector4 | None = None
    Q: Vector4 | None = None
    positive_energy: bool | None = None
    all_shells_checked: bool = False
    conservation_checked: bool = False

    @property
    def physical(self) -> bool:
        return self.status == "physical"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready sample record with vectors as component lists."""

        return {
            "artifact_type": "b_direct_sample",
            "profile": self.profile,
            "status": self.status,
            "physical": self.physical,
            "diagnostics": list(self.diagnostics),
            "normalization": self.normalization,
            "coordinates": {
                "total_mass": self.total_mass,
                "masses": list(self.masses) if self.masses is not None else None,
                "ordered_invariants": list(self.ordered_invariants),
                "invariant_values": list(self.invariant_values) if self.invariant_values is not None else None,
                "angle_variables": list(self.angle_variables),
                "angle_values": list(self.angle_values) if self.angle_values is not None else None,
            },
            "differential_measure": self.differential_measure,
            "density_expression": self.density_expression,
            "normalization_identity": self.normalization_identity,
            "angle_definitions": list(self.angle_definitions),
            "branch": {
                "count": self.branch_count,
                "root_condition": self.root_condition,
                "root_cosine": self.root_cosine,
                "root_residual": self.root_residual,
                "c_reconstructed": self.c_reconstructed,
            },
            "kinematics": {
                "S": self.S,
                "k12": self.k12,
                "k1": self.k1,
                "k2": self.k2,
                "k4": self.k4,
                "density": self.density,
                "positive_energy": self.positive_energy,
                "all_shells_checked": self.all_shells_checked,
                "conservation_checked": self.conservation_checked,
            },
            "intermediates": {
                "q12": list(self.q12) if self.q12 is not None else None,
                "q34": list(self.q34) if self.q34 is not None else None,
                "q23": list(self.q23) if self.q23 is not None else None,
                "Q": list(self.Q) if self.Q is not None else None,
            },
            "momenta": (
                [
                    {"particle_id": particle, "components": list(vector)}
                    for particle, vector in enumerate(self.momenta, 1)
                ]
                if self.momenta is not None
                else None
            ),
        }


def check_b_f03_eligibility(record: object) -> BDirectEligibility:
    """Check B connectivity, labels, and ordered ``(s12,s34,s23)`` directly.

    The accepted structure is expressed using its directed incidence and
    particle labels, not a fixture name, source graph number, or digest.  Edge
    storage order may differ; each double line is matched by its endpoints.
    """

    if not isinstance(record, PhysicalLabelRecord):
        return BDirectEligibility(False, ("a complete PhysicalLabelRecord is required",))

    reasons: list[str] = []
    topology = record.topology
    if topology.n != 4 or set(topology.vertices) != {0, 1, 2, 3}:
        reasons.append("requires four particles and the four B internal vertices")
    if topology.initial_vertex != 0:
        reasons.append("the initial attachment must be at vertex 0")

    expected_edges = Counter(
        {
            (0, 1, EdgeType.DOUBLE, 1): 1,
            (0, 2, EdgeType.DOUBLE, 1): 1,
            (2, 3, EdgeType.DOUBLE, 1): 1,
            (1, 2, EdgeType.SINGLE, 1): 1,
        }
    )
    actual_edges = Counter(
        (edge.source, edge.target, edge.edge_type, edge.multiplicity)
        for edge in topology.edges
    )
    if actual_edges != expected_edges:
        reasons.append("directed line connectivity must match the B/F03 single-loop structure")

    if dict(topology.final_slots) != {0: 0, 1: 1, 2: 1, 3: 2}:
        reasons.append("final slots must be distributed as 0,1,1,2 over vertices 0..3")

    edge_indices = {
        (edge.source, edge.target, edge.edge_type): index
        for index, edge in enumerate(topology.edges)
    }
    if len(edge_indices) != len(topology.edges):
        reasons.append("B/F03 connectivity must not contain duplicate edge records")

    labeling = record.complete_labeling
    final_assignments = {
        (item.vertex, item.slot, item.particle_id) for item in labeling.final_particles
    }
    if final_assignments != {
        (1, 0, 1),
        (2, 0, 4),
        (3, 0, 2),
        (3, 1, 3),
    }:
        reasons.append("external particle labels must be p1@v1, p4@v2, and p2,p3@v3")

    expected_double_by_edge = {
        (0, 1, EdgeType.DOUBLE): (1, 2),
        (0, 2, EdgeType.DOUBLE): (3, 4),
        (2, 3, EdgeType.DOUBLE): (2, 3),
    }
    actual_doubles = {
        (item.edge_index, item.occurrence): tuple(item.particle_ids)
        for item in labeling.double_lines
    }
    if len(actual_doubles) != len(labeling.double_lines):
        reasons.append("each B/F03 double-line occurrence must have one unambiguous label")
    for edge_key, particles in expected_double_by_edge.items():
        edge_index = edge_indices.get(edge_key)
        if edge_index is None or actual_doubles.get((edge_index, 0)) != particles:
            reasons.append(
                f"double line {edge_key[0]}->{edge_key[1]} must carry "
                f"the ordered particle subset {particles}"
            )
    if len(actual_doubles) != len(expected_double_by_edge):
        reasons.append("B/F03 requires exactly three labelled double-line occurrences")

    expected_single_edge = edge_indices.get((1, 2, EdgeType.SINGLE))
    actual_singles = tuple(
        (item.edge_index, item.occurrence, item.particle_id)
        for item in labeling.single_lines
    )
    if actual_singles != ((expected_single_edge, 0, 2),):
        reasons.append("the sole single line 1->2 must be the internal p2 occurrence")

    if tuple(record.momentum.variables) != B_F03_ORDERED_INVARIANTS:
        reasons.append("independent invariants must be ordered exactly as (s12,s34,s23)")
    if not record.momentum.verified or not record.independence.independent:
        reasons.append("verified momentum and independent-invariant evidence are required")
    if tuple(record.independence.variables) != B_F03_ORDERED_INVARIANTS:
        reasons.append("the independence record must use ordered variables (s12,s34,s23)")
    if dict(record.measure_cancellations) != {1: 0, 2: 1, 3: 0, 4: 0}:
        reasons.append("the B/F03 measure marking requires exactly one internal p2 cancellation")

    return BDirectEligibility(not reasons, tuple(reasons))


def _finite(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _add(left: Vector4, right: Vector4) -> Vector4:
    return tuple(left[i] + right[i] for i in range(4))  # type: ignore[return-value]


def _sub(left: Vector4, right: Vector4) -> Vector4:
    return tuple(left[i] - right[i] for i in range(4))  # type: ignore[return-value]


def _dot(left: Vector4, right: Vector4) -> float:
    return left[0] * right[0] - sum(left[i] * right[i] for i in range(1, 4))


def _spatial_dot(left: Vector3, right: Vector3) -> float:
    return sum(left[i] * right[i] for i in range(3))


def _spatial_norm(vector: Vector3) -> float:
    return math.sqrt(max(0.0, _spatial_dot(vector, vector)))


def _scale3(vector: Vector3, scale: float) -> Vector3:
    return (scale * vector[0], scale * vector[1], scale * vector[2])


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _unit3(vector: Vector3, *, name: str) -> Vector3:
    norm = _spatial_norm(vector)
    if norm <= 0.0:
        raise ValueError(f"{name} has zero length")
    return _scale3(vector, 1.0 / norm)


def _direction(theta: float, phi: float) -> Vector3:
    return (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )


def _boost(momentum: Vector4, beta: Vector3) -> Vector4:
    """Boost a rest-frame vector into a frame moving with velocity beta."""

    bx, by, bz = beta
    beta2 = bx * bx + by * by + bz * bz
    if beta2 < 1.0e-30:
        return momentum
    if beta2 >= 1.0:
        raise ValueError("boost velocity must be subluminal")
    gamma = 1.0 / math.sqrt(1.0 - beta2)
    spatial = (momentum[1], momentum[2], momentum[3])
    beta_dot_p = bx * spatial[0] + by * spatial[1] + bz * spatial[2]
    factor = ((gamma - 1.0) * beta_dot_p / beta2) + gamma * momentum[0]
    return (
        gamma * (momentum[0] + beta_dot_p),
        spatial[0] + factor * bx,
        spatial[1] + factor * by,
        spatial[2] + factor * bz,
    )


def _sqrt_lambda(a: float, b: float, c: float, description: str) -> tuple[float | None, str | None]:
    value = kallen(a, b, c)
    scale = max(abs(a), abs(b), abs(c), 1.0)
    tolerance = _DOMAIN_TOL * scale * scale
    if value < -tolerance:
        return None, f"{description} is outside the Kallen physical domain"
    return math.sqrt(0.0 if value < 0.0 else value), None


def _close(left: float, right: float, scale: float) -> bool:
    return abs(left - right) <= _REL_TOL * max(abs(scale), abs(left), abs(right), 1.0)


def _normalization_record(normalization: str) -> tuple[str, str, str]:
    if normalization == "standard_2pi4":
        return (
            B_F03_STANDARD_DIFFERENTIAL_MEASURE,
            B_F03_STANDARD_DENSITY_EXPRESSION,
            B_F03_STANDARD_NORMALIZATION_IDENTITY,
        )
    return (
        B_F03_DIFFERENTIAL_MEASURE,
        B_F03_DENSITY_EXPRESSION,
        B_F03_NORMALIZATION_IDENTITY,
    )


def _failure(status: BDirectStatus, message: str, **fields: object) -> BDirectSample:
    normalization = str(fields.pop("normalization", "jing_psf"))
    measure, density, identity = _normalization_record(normalization)
    return BDirectSample(
        profile=B_F03_PROFILE,
        status=status,
        diagnostics=(message,),
        normalization=normalization,
        differential_measure=measure,
        density_expression=density,
        normalization_identity=identity,
        **fields,
    )


def sample_b_f03(
    record: object,
    total_mass: float,
    masses: Sequence[float],
    s12: float,
    s34: float,
    s23: float,
    theta0: float,
    phi0: float,
    theta1: float,
    phi1: float,
    phi4_star: float,
    *,
    normalization: str = "jing_psf",
) -> BDirectSample:
    """Reconstruct one B/F03 sample from three invariants and five angles.

    Angles ``(theta0,phi0)`` orient q12 in the P rest frame.  Angles
    ``(theta1,phi1)`` orient p1 in the q12 rest frame, using axes transported
    from the P frame by a no-rotation boost.  The final azimuth is measured
    about p2 in the Q rest frame, with its zero axis obtained by projecting
    boosted p1 perpendicular to p2.  This natural local chart is not converted
    to an overall particle-1/2 Euler chart.  The eliminated cosine has one
    analytic root; no inverse solver or reference density is used.
    """

    eligibility = check_b_f03_eligibility(record)
    if not eligibility.eligible:
        return _failure(
            "unsupported",
            "; ".join(eligibility.reasons),
            normalization=normalization,
        )
    if normalization not in SUPPORTED_NORMALIZATIONS:
        return _failure(
            "unsupported",
            f"normalization {normalization!r} is unsupported; supported values are {SUPPORTED_NORMALIZATIONS}",
            normalization=normalization,
        )

    def fail(status: BDirectStatus, message: str, **fields: object) -> BDirectSample:
        return _failure(status, message, normalization=normalization, **fields)

    try:
        M = _finite("total_mass", total_mass)
        mass_values = tuple(_finite(f"m_{index}", value) for index, value in enumerate(masses, 1))
    except TypeError as exc:
        raise ValueError("exactly four particle masses are required") from exc
    if len(mass_values) != 4:
        raise ValueError("exactly four particle masses are required")
    m1, m2, m3, m4 = mass_values
    s12 = _finite("s12", s12)
    s34 = _finite("s34", s34)
    s23 = _finite("s23", s23)
    angles = tuple(
        _finite(name, value)
        for name, value in zip(B_F03_ANGLE_VARIABLES, (theta0, phi0, theta1, phi1, phi4_star))
    )
    theta0, phi0, theta1, phi1, phi4_star = angles

    if M <= 0.0 or any(mass < 0.0 for mass in mass_values):
        return fail("unphysical", "M must be positive and all four masses must be non-negative")
    if M < sum(mass_values) - _DOMAIN_TOL * max(M, sum(mass_values), 1.0):
        return fail("unphysical", "M is below the four-particle threshold")
    if any(not 0.0 <= theta <= math.pi for theta in (theta0, theta1)):
        return fail("unphysical", "theta0 and theta1 must lie in [0, pi]")
    if any(not 0.0 <= phi < 2.0 * math.pi for phi in (phi0, phi1, phi4_star)):
        return fail("unphysical", "phi0, phi1, and phi4_star must lie in [0, 2*pi)")

    for name, value, threshold in (
        ("s12", s12, (m1 + m2) ** 2),
        ("s34", s34, (m3 + m4) ** 2),
        ("s23", s23, (m2 + m3) ** 2),
    ):
        if value < threshold - _DOMAIN_TOL * max(value, threshold, 1.0):
            return fail("unphysical", f"{name} is below its two-particle threshold")
    if s12 <= 0.0:
        return fail("degenerate", "s12=0 has no q12 rest frame")
    if s34 <= 0.0:
        return fail("degenerate", "s34=0 has no q34 rest frame")
    if s23 < 0.0:
        return fail("unphysical", "s23 must be non-negative")

    sqrt_s12 = math.sqrt(s12)
    sqrt_s34 = math.sqrt(s34)
    sqrt_s23 = math.sqrt(s23)
    if sqrt_s12 + sqrt_s34 > M + _DOMAIN_TOL * max(M, 1.0):
        return fail("unphysical", "sqrt(s12)+sqrt(s34) exceeds M")

    lambda_parent, problem = _sqrt_lambda(M * M, s12, s34, "P -> q12+q34")
    if problem:
        return fail("unphysical", problem)
    lambda_12, problem = _sqrt_lambda(s12, m1 * m1, m2 * m2, "q12 -> p1+p2")
    if problem:
        return fail("unphysical", problem)
    assert lambda_parent is not None and lambda_12 is not None
    if lambda_parent <= math.sqrt(_DOMAIN_TOL) * max(M * M, s12, s34, 1.0):
        return fail("degenerate", "P -> q12+q34 is at the production threshold")
    if lambda_12 <= math.sqrt(_DOMAIN_TOL) * max(s12, m1 * m1, m2 * m2, 1.0):
        return fail("degenerate", "q12 -> p1+p2 is at the two-body threshold")
    k12 = lambda_parent / (2.0 * M)
    m12 = sqrt_s12
    k1 = lambda_12 / (2.0 * m12)
    if k12 <= _REL_TOL * max(M, 1.0):
        return fail("degenerate", "P -> q12+q34 is at the production threshold")
    if k1 <= _REL_TOL * max(m12, 1.0):
        return fail("degenerate", "q12 -> p1+p2 is at the two-body threshold")

    e12 = (M * M + s12 - s34) / (2.0 * M)
    e34 = (M * M + s34 - s12) / (2.0 * M)
    if e12 <= 0.0 or e34 <= 0.0:
        return fail("unphysical", "the initial two-body block has a non-positive energy")

    n0 = _direction(theta0, phi0)
    q12: Vector4 = (e12, *(k12 * component for component in n0))
    q34: Vector4 = (e34, *(-k12 * component for component in n0))
    e1_rest = (s12 + m1 * m1 - m2 * m2) / (2.0 * m12)
    e2_rest = (s12 + m2 * m2 - m1 * m1) / (2.0 * m12)
    n1 = _direction(theta1, phi1)
    p1_rest: Vector4 = (e1_rest, *(k1 * component for component in n1))
    p2_rest: Vector4 = (e2_rest, *(-k1 * component for component in n1))
    beta12: Vector3 = (q12[1] / q12[0], q12[2] / q12[0], q12[3] / q12[0])
    p1 = _boost(p1_rest, beta12)
    p2_initial = _boost(p2_rest, beta12)

    Q = _add(q34, p2_initial)
    S = _dot(Q, Q)
    energy_scale = max(M * M, s12, s34, 1.0)
    if Q[0] <= 0.0 or S <= _DOMAIN_TOL * energy_scale:
        return fail("degenerate", "Q=q34+p2 does not define a future timelike rest frame", S=S)
    sqrt_S = math.sqrt(S)
    beta_Q: Vector3 = (Q[1] / Q[0], Q[2] / Q[0], Q[3] / Q[0])
    if sum(component * component for component in beta_Q) >= 1.0:
        return fail("unphysical", "Q has no subluminal rest-frame boost", S=S)

    p2_star = _boost(p2_initial, tuple(-item for item in beta_Q))  # type: ignore[arg-type]
    p1_star = _boost(p1, tuple(-item for item in beta_Q))  # type: ignore[arg-type]
    e2_star = (S + m2 * m2 - s34) / (2.0 * sqrt_S)
    lambda_2, problem = _sqrt_lambda(S, m2 * m2, s34, "Q -> p2+q34")
    if problem:
        return fail("unphysical", problem, S=S)
    lambda_4, problem = _sqrt_lambda(S, s23, m4 * m4, "Q -> q23+p4")
    if problem:
        return fail("unphysical", problem, S=S)
    assert lambda_2 is not None and lambda_4 is not None
    if lambda_2 <= math.sqrt(_DOMAIN_TOL) * max(S, m2 * m2, s34, 1.0):
        return fail("degenerate", "k2*=0 makes the eliminated-cosine chart singular", S=S)
    if lambda_4 <= math.sqrt(_DOMAIN_TOL) * max(S, s23, m4 * m4, 1.0):
        return fail("degenerate", "k4*=0 makes the eliminated-cosine chart singular", S=S)
    k2 = lambda_2 / (2.0 * sqrt_S)
    k4 = lambda_4 / (2.0 * sqrt_S)
    if k2 <= _REL_TOL * max(sqrt_S, 1.0):
        return fail("degenerate", "k2*=0 makes the eliminated-cosine chart singular", S=S, k2=k2, k4=k4)
    if k4 <= _REL_TOL * max(sqrt_S, 1.0):
        return fail("degenerate", "k4*=0 makes the eliminated-cosine chart singular", S=S, k2=k2, k4=k4)
    e23_star = (S + s23 - m4 * m4) / (2.0 * sqrt_S)
    c0 = (s23 + m2 * m2 - m3 * m3 - 2.0 * e23_star * e2_star) / (2.0 * k2 * k4)
    if not math.isfinite(c0):
        return fail("unphysical", "the analytic cosine root is not finite", S=S, k2=k2, k4=k4)
    if c0 < -1.0 - _DOMAIN_TOL or c0 > 1.0 + _DOMAIN_TOL:
        return fail(
            "unphysical",
            "the unique analytic cosine root lies outside [-1,1]",
            S=S,
            k2=k2,
            k4=k4,
            root_cosine=c0,
        )
    cosine = min(1.0, max(-1.0, c0))
    root_residual = (
        s23 + m2 * m2 - m3 * m3
        - 2.0 * e23_star * e2_star
        - 2.0 * k2 * k4 * cosine
    )
    if abs(root_residual) > _REL_TOL * max(S, s23, m2 * m2, m3 * m3, 1.0):
        return fail(
            "unphysical",
            "the unsquared on-shell equation does not vanish at the analytic root",
            S=S,
            k2=k2,
            k4=k4,
            root_cosine=c0,
            root_residual=root_residual,
        )
    if abs(math.sin(theta0)) <= _REL_TOL:
        return fail(
            "degenerate",
            "theta0 is at a polar chart boundary where phi0 is redundant",
            S=S,
            k2=k2,
            k4=k4,
            root_cosine=c0,
            root_residual=root_residual,
            branch_count=1,
        )
    if abs(math.sin(theta1)) <= _REL_TOL:
        return fail(
            "degenerate",
            "theta1 is at a polar chart boundary where phi1 is redundant",
            S=S,
            k2=k2,
            k4=k4,
            root_cosine=c0,
            root_residual=root_residual,
            branch_count=1,
        )
    if 1.0 - abs(cosine) <= _REL_TOL:
        return fail(
            "degenerate",
            "the cosine root is at |c0|=1, so phi4_star does not distinguish a physical orientation",
            S=S,
            k2=k2,
            k4=k4,
            root_cosine=c0,
            root_residual=root_residual,
            branch_count=1,
        )

    p2_direction: Vector3 = _unit3(
        (p2_star[1], p2_star[2], p2_star[3]), name="p2* direction"
    )
    p1_star_spatial: Vector3 = (p1_star[1], p1_star[2], p1_star[3])
    p1_projection = _sub3(
        p1_star_spatial,
        _scale3(p2_direction, _spatial_dot(p1_star_spatial, p2_direction)),
    )
    projection_norm = _spatial_norm(p1_projection)
    if projection_norm <= _REL_TOL * max(_spatial_norm(p1_star_spatial), 1.0):
        return fail(
            "degenerate",
            "boosted p1 is collinear with p2*; the phi4* zero axis is undefined",
            S=S,
            k2=k2,
            k4=k4,
            root_cosine=c0,
            root_residual=root_residual,
        )
    ex = _unit3(p1_projection, name="projected p1* zero axis")
    ey = _unit3(_cross(p2_direction, ex), name="right-handed phi4* axis")

    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    azimuth = _add3(_scale3(ex, math.cos(phi4_star)), _scale3(ey, math.sin(phi4_star)))
    p4_direction = _add3(_scale3(p2_direction, cosine), _scale3(azimuth, sine))
    e4_star = (S + m4 * m4 - s23) / (2.0 * sqrt_S)
    p4_star: Vector4 = (e4_star, *(k4 * component for component in p4_direction))
    q23_star: Vector4 = (e23_star, *(-k4 * component for component in p4_direction))
    p3_star = _sub(q23_star, p2_star)
    p2 = _boost(p2_star, beta_Q)
    p3 = _boost(p3_star, beta_Q)
    p4 = _boost(p4_star, beta_Q)
    q23 = _boost(q23_star, beta_Q)

    momenta = (p1, p2, p3, p4)
    parent: Vector4 = (M, 0.0, 0.0, 0.0)
    if any(momentum[0] <= 0.0 for momentum in momenta):
        return fail(
            "unphysical",
            "one or more final particles have non-positive energy",
            S=S,
            k2=k2,
            k4=k4,
            root_cosine=c0,
            root_residual=root_residual,
            positive_energy=False,
        )

    shells = tuple(_dot(momentum, momentum) for momentum in momenta)
    expected_shells = tuple(mass * mass for mass in mass_values)
    if not all(_close(actual, expected, energy_scale) for actual, expected in zip(shells, expected_shells)):
        return fail(
            "unphysical",
            "reconstructed final momenta failed an on-shell check",
            S=S,
            k2=k2,
            k4=k4,
            root_cosine=c0,
            root_residual=root_residual,
            positive_energy=True,
        )
    summed = tuple(sum(momentum[index] for momentum in momenta) for index in range(4))
    if not all(_close(actual, expected, M) for actual, expected in zip(summed, parent)):
        return fail(
            "unphysical",
            "reconstructed final momenta do not conserve P=(M,0,0,0)",
            S=S,
            k2=k2,
            k4=k4,
            root_cosine=c0,
            root_residual=root_residual,
            positive_energy=True,
        )

    q12_check = _add(p1, p2)
    q34_check = _add(p3, p4)
    q23_check = _add(p2, p3)
    if not (
        _close(_dot(q12_check, q12_check), s12, energy_scale)
        and _close(_dot(q34_check, q34_check), s34, energy_scale)
        and _close(_dot(q23_check, q23_check), s23, energy_scale)
        and _close(_dot(q23, q23), s23, energy_scale)
        and _close(_dot(Q, Q), S, energy_scale)
    ):
        return fail(
            "unphysical",
            "reconstructed momenta failed one or more requested-invariant checks",
            S=S,
            k2=k2,
            k4=k4,
            root_cosine=c0,
            root_residual=root_residual,
            positive_energy=True,
        )

    p3_shell_residual = _dot(p3_star, p3_star) - m3 * m3
    if abs(p3_shell_residual) > _REL_TOL * max(energy_scale, 1.0):
        return fail(
            "unphysical",
            "the eliminated p3 shell equation failed after reconstruction",
            S=S,
            k2=k2,
            k4=k4,
            root_cosine=c0,
            root_residual=p3_shell_residual,
            positive_energy=True,
        )

    density_jing = (
        k12 * k1
        / ((2.0 * math.pi) ** 12 * 128.0 * M * m12 * sqrt_S * k2)
    )
    density = (
        density_jing * (2.0 * math.pi) ** 4
        if normalization == "standard_2pi4"
        else density_jing
    )
    measure_record, density_record, identity_record = _normalization_record(normalization)
    c_reconstructed = (
        _spatial_dot(
            (p4_star[1], p4_star[2], p4_star[3]),
            (p2_star[1], p2_star[2], p2_star[3]),
        )
        / (k2 * k4)
    )
    return BDirectSample(
        profile=B_F03_PROFILE,
        status="physical",
        diagnostics=(
            "one simple analytic cosine root; local sample only, not a global covariant certificate",
            "theta1,phi1 use q12-rest axes transported from the P frame by a no-rotation boost",
            "the particle-1/2 overall Euler chart conversion is not implemented",
        ),
        normalization=normalization,
        differential_measure=measure_record,
        density_expression=density_record,
        normalization_identity=identity_record,
        total_mass=M,
        masses=mass_values,
        invariant_values=(s12, s34, s23),
        angle_values=(theta0, phi0, theta1, phi1, phi4_star),
        branch_count=1,
        root_cosine=c0,
        root_residual=root_residual,
        c_reconstructed=c_reconstructed,
        S=S,
        k12=k12,
        k1=k1,
        k2=k2,
        k4=k4,
        density=density,
        momenta=momenta,
        q12=q12,
        q34=q34,
        q23=q23,
        Q=Q,
        positive_energy=True,
        all_shells_checked=True,
        conservation_checked=True,
    )


def _add3(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub3(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


__all__ = [
    "BDirectEligibility",
    "BDirectSample",
    "B_F03_ANGLE_DEFINITIONS",
    "B_F03_ANGLE_VARIABLES",
    "B_F03_DENSITY_EXPRESSION",
    "B_F03_DIFFERENTIAL_MEASURE",
    "B_F03_NORMALIZATION_IDENTITY",
    "B_F03_ORDERED_INVARIANTS",
    "B_F03_PROFILE",
    "B_F03_STANDARD_DENSITY_EXPRESSION",
    "B_F03_STANDARD_DIFFERENTIAL_MEASURE",
    "B_F03_STANDARD_NORMALIZATION_IDENTITY",
    "SUPPORTED_NORMALIZATIONS",
    "check_b_f03_eligibility",
    "sample_b_f03",
]
