"""Independent local density pullback checks against Yu (2022), Tables 5 and 7.

The external variables and Gram factor are rebuilt from Euclidean geometry of
the fixed sequential-reference four-vectors.  No production Gram or density
helper is used.  This check covers density coefficients and coordinate
pullbacks only; it cannot validate every printed kinematic relation in either
table.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import d5_benchmark as _d5


YU2022_PDF_SHA256 = "e0cad84ae75b0ed93911f1614e874b363a08c448b53a71341f27e02b4aab30b9"
YU2022_SOURCE = "Yu et al. (2022)"
YU2022_TABLE_SOURCES = {
    "table5": {
        "citation": YU2022_SOURCE,
        "table": 5,
        "pdf_page": 8,
        "journal_page": "083101-7",
        "pdf_sha256": YU2022_PDF_SHA256,
    },
    "table7": {
        "citation": YU2022_SOURCE,
        "table": 7,
        "pdf_page": 9,
        "journal_page": "083101-8",
        "pdf_sha256": YU2022_PDF_SHA256,
    },
}
REFERENCE_VARIABLES = (
    "s12", "s34", "theta0", "phi0", "theta12", "phi12", "theta34", "phi34",
)
TABLE5_INVARIANTS = ("s12", "s13", "s14", "s23", "s124")
TABLE7_INVARIANTS = ("s12", "s13", "s24", "s34")
FD_STEPS = (2.0e-5, 1.0e-5)
EXPECTED_CLASS_COUNTS_BY_LOOP = (2, 6, 29, 106)
SUPPORTED_NORMALIZATIONS = ("jing_psf", "standard_2pi4")
TWO_PI = 2.0 * math.pi
YU2022_SCHEMA_VERSION = "yu2022-density-pullback-v1"


def _report_source() -> dict[str, Any]:
    return {
        "citation": YU2022_SOURCE,
        "pdf_sha256": YU2022_PDF_SHA256,
        "tables": {name: dict(source) for name, source in YU2022_TABLE_SOURCES.items()},
    }


def _expected_all_loop_class_ids() -> set[str]:
    return {
        f"L{loop_count}-C{index:03d}"
        for loop_count, class_count in enumerate(EXPECTED_CLASS_COUNTS_BY_LOOP)
        for index in range(1, class_count + 1)
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _norm3(a: Sequence[float]) -> float:
    return math.sqrt(_dot3(a, a))


def _boost(vector: Sequence[float], beta: Sequence[float]) -> tuple[float, float, float, float]:
    """Apply an explicit Lorentz boost using the independent audit convention."""

    if len(vector) != 4 or len(beta) != 3:
        raise ValueError("Lorentz boost requires a four-vector and a three-velocity")
    energy, *momentum = (float(value) for value in vector)
    velocity = tuple(float(value) for value in beta)
    if not all(math.isfinite(value) for value in (energy, *momentum, *velocity)):
        raise ValueError("Lorentz boost inputs must be finite")
    beta2 = _dot3(velocity, velocity)
    if beta2 < 0.0 or beta2 >= 1.0:
        raise ValueError("Lorentz boost velocity must satisfy 0 <= beta^2 < 1")
    gamma = 1.0 / math.sqrt(1.0 - beta2)
    beta_dot_p = _dot3(velocity, momentum)
    spatial_factor = gamma * energy
    if beta2 > 0.0:
        spatial_factor += (gamma - 1.0) * beta_dot_p / beta2
    boosted_spatial = tuple(momentum[i] + spatial_factor * velocity[i] for i in range(3))
    boosted_energy = gamma * (energy + beta_dot_p)
    return (boosted_energy, *boosted_spatial)


def _lambda(a: float, b: float, c: float) -> float:
    return a * a + b * b + c * c - 2.0 * (a * b + a * c + b * c)


def _positive_sqrt(value: float, label: str, *, scale: float = 1.0) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be finite and strictly positive")
    result = math.sqrt(value)
    if not math.isfinite(result) or result <= 0.0 or result <= 1.0e-14 * max(1.0, scale):
        raise ValueError(f"{label} is zero or numerically degenerate")
    return result


def _normalization_factor(normalization: str) -> float:
    if normalization == "jing_psf":
        return 1.0
    if normalization == "standard_2pi4":
        return TWO_PI**4
    raise ValueError("normalization must be jing_psf or standard_2pi4")


def _source_kinematics(
    momenta: Sequence[Sequence[float]], masses: Sequence[float], total_mass: float,
) -> dict[str, Any]:
    """Compute the paper's k_i, c_ij and A_g directly from spatial vectors."""

    vectors = _d5._momenta(momenta, "Yu2022 reference momenta")
    mass_values = tuple(float(value) for value in masses)
    if len(mass_values) != 4 or any(not math.isfinite(value) or value <= 0.0 for value in mass_values):
        raise ValueError("Yu2022 check requires four finite, strictly positive masses")
    parent_mass = float(total_mass)
    if not math.isfinite(parent_mass) or parent_mass <= sum(mass_values):
        raise ValueError("total_mass must be strictly above the four-body threshold")

    spatial = tuple(tuple(vectors[i][axis + 1] for axis in range(3)) for i in range(3))
    k = tuple(_positive_sqrt(_dot3(row, row), f"k{i+1}^2", scale=parent_mass) for i, row in enumerate(spatial))
    cosine: dict[str, float] = {}
    for first, second in ((1, 2), (1, 3), (2, 3)):
        value = _dot3(spatial[first - 1], spatial[second - 1]) / (k[first - 1] * k[second - 1])
        if not math.isfinite(value) or abs(value) > 1.0 + 1.0e-12:
            raise ValueError(f"c{first}{second} is outside the Euclidean cosine range")
        cosine[f"c{first}{second}"] = min(1.0, max(-1.0, value))
    c12, c13, c23 = cosine["c12"], cosine["c13"], cosine["c23"]
    gram = 1.0 + 2.0 * c12 * c13 * c23 - c12 * c12 - c13 * c13 - c23 * c23
    if not math.isfinite(gram) or gram <= 0.0:
        raise ValueError("external angular Gram determinant is not strictly positive")
    a_g = 1.0 / math.sqrt(gram)

    q12 = _d5._sum_vectors((vectors[0], vectors[1]))
    s12 = _d5._minkowski_square(q12)
    s34 = _d5._minkowski_square(_d5._sum_vectors((vectors[2], vectors[3])))
    q_norm = _norm3(q12[1:])
    if q_norm <= 1.0e-14 * parent_mass:
        raise ValueError("q12 direction is degenerate; c1star is undefined")
    axis = tuple(value / q_norm for value in q12[1:])
    beta = tuple(-q12[i + 1] / q12[0] for i in range(3))
    p1_star = _boost(vectors[0], beta)
    k1_star = _positive_sqrt(_dot3(p1_star[1:], p1_star[1:]), "k1star^2", scale=parent_mass)
    c1_star = -_dot3(p1_star[1:], axis) / k1_star
    if not math.isfinite(c1_star) or abs(c1_star) > 1.0 + 1.0e-12:
        raise ValueError("c1star is outside the Euclidean cosine range")
    c1_star = min(1.0, max(-1.0, c1_star))
    lambda_parent = _lambda(parent_mass * parent_mass, s12, s34)
    lambda_pair = _lambda(s12, mass_values[0] ** 2, mass_values[1] ** 2)
    if lambda_parent <= 0.0 or lambda_pair <= 0.0:
        raise ValueError("Table 7 momentum factors require strict open-domain Kallen roots")
    gamma_beta = math.sqrt(lambda_parent) / (2.0 * parent_mass * math.sqrt(s12))
    expected_k1_star = math.sqrt(lambda_pair) / (2.0 * math.sqrt(s12))
    if abs(k1_star - expected_k1_star) > 1.0e-10 * max(1.0, expected_k1_star):
        raise ValueError("boosted p1 momentum disagrees with the independent two-body k1star root")
    return {
        "k1": k[0], "k2": k[1], "k3": k[2],
        "c12": c12, "c13": c13, "c23": c23,
        "angular_gram_determinant": gram,
        "A_g": a_g,
        "s12": s12,
        "s34": s34,
        "c1star": c1_star,
        "k1star_from_boost": k1_star,
        "k1star_from_kallen": expected_k1_star,
        "gamma_beta": gamma_beta,
    }


def table5_coordinates(
    momenta: Sequence[Sequence[float]], total_mass: float,
) -> tuple[dict[str, float], tuple[float, ...], dict[str, Any]]:
    total_mass = float(total_mass)
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total_mass must be finite and positive")
    vectors = _d5._momenta(momenta, "Table 5 momenta")
    invariants = _d5._invariants_from_momenta(vectors, TABLE5_INVARIANTS)
    angles, _sigma, orientation = _d5._recover_euler_and_sigma(vectors)
    values = {**invariants, **angles}
    output = tuple(invariants[name] / (total_mass * total_mass) for name in TABLE5_INVARIANTS) + (
        angles["theta"], angles["phi"], angles["psi"],
    )
    if len(output) != 8 or not all(math.isfinite(value) for value in output):
        raise ValueError("Table 5 coordinates are not a finite eight-coordinate vector")
    return values, output, {"invariants": invariants, "euler_angles": angles, "orientation": orientation}


def table7_coordinates(
    momenta: Sequence[Sequence[float]], masses: Sequence[float], total_mass: float,
) -> tuple[dict[str, float], tuple[float, ...], dict[str, Any]]:
    total_mass = float(total_mass)
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total_mass must be finite and positive")
    vectors = _d5._momenta(momenta, "Table 7 momenta")
    invariants = _d5._invariants_from_momenta(vectors, TABLE7_INVARIANTS)
    euler, _sigma, orientation = _d5._recover_euler_and_sigma(vectors)
    kinematics = _source_kinematics(vectors, masses, total_mass)
    values = {**invariants, "c1star": kinematics["c1star"], **euler}
    output = tuple(invariants[name] / (total_mass * total_mass) for name in TABLE7_INVARIANTS) + (
        kinematics["c1star"], euler["theta"], euler["phi"], euler["psi"],
    )
    if len(output) != 8 or not all(math.isfinite(value) for value in output):
        raise ValueError("Table 7 coordinates are not a finite eight-coordinate vector")
    return values, output, {"invariants": invariants, "euler_angles": euler, "orientation": orientation,
                            "external_kinematics": kinematics}


def table5_density(
    momenta: Sequence[Sequence[float]], masses: Sequence[float], total_mass: float, *, normalization: str = "jing_psf",
) -> dict[str, Any]:
    kinematics = _source_kinematics(momenta, masses, total_mass)
    parent_mass = float(total_mass)
    factor = _normalization_factor(normalization)
    denominator = 512.0 * (2.0 * math.pi) ** 12 * parent_mass**3 * kinematics["k1"] * kinematics["k2"] * kinematics["k3"]
    density = factor * kinematics["A_g"] / denominator
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("Table 5 external density is non-finite or non-positive")
    return {"rho_per_mirror": density, "normalization_factor": factor, "denominator": denominator, **kinematics}


def table7_density(
    momenta: Sequence[Sequence[float]], masses: Sequence[float], total_mass: float, *, normalization: str = "jing_psf",
) -> dict[str, Any]:
    kinematics = _source_kinematics(momenta, masses, total_mass)
    parent_mass = float(total_mass)
    factor = _normalization_factor(normalization)
    denominator = 256.0 * (2.0 * math.pi) ** 12 * parent_mass**2 * kinematics["k1"] * kinematics["k2"] * kinematics["k3"]
    density = factor * kinematics["A_g"] * kinematics["gamma_beta"] * kinematics["k1star_from_boost"] / denominator
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("Table 7 external density is non-finite or non-positive")
    return {"rho_per_mirror": density, "normalization_factor": factor, "denominator": denominator, **kinematics}


def _validate_reference_point(point: Any) -> dict[str, float]:
    if not isinstance(point, Mapping):
        raise ValueError("reference point must be a coordinate mapping")
    try:
        values = {name: float(point[name]) for name in REFERENCE_VARIABLES}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reference point must contain all eight sequential variables") from exc
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("reference point contains NaN or infinity")
    if values["s12"] <= 0.0 or values["s34"] <= 0.0:
        raise ValueError("source invariant squares must be positive")
    for name in ("theta0", "theta12", "theta34"):
        if not 0.0 < values[name] < math.pi:
            raise ValueError(f"{name} must be strictly between 0 and pi")
    for name in ("phi0", "phi12", "phi34"):
        if not 0.0 <= values[name] < TWO_PI:
            raise ValueError(f"{name} must lie inside [0,2*pi)")
    return values


def _point_from_dimensionless(values: Sequence[float], total_mass: float) -> dict[str, float]:
    if len(values) != 8:
        raise ValueError("dimensionless reference vector must have eight coordinates")
    return {
        "s12": float(values[0]) * total_mass**2,
        "s34": float(values[1]) * total_mass**2,
        "theta0": float(values[2]), "phi0": float(values[3]) % TWO_PI,
        "theta12": float(values[4]), "phi12": float(values[5]) % TWO_PI,
        "theta34": float(values[6]), "phi34": float(values[7]) % TWO_PI,
    }


def _event_coordinates(
    momenta: Sequence[Sequence[float]], *, table: str, masses: Sequence[float], total_mass: float,
) -> tuple[dict[str, float], tuple[float, ...], dict[str, Any]]:
    if table == "table5":
        return table5_coordinates(momenta, total_mass)
    if table == "table7":
        return table7_coordinates(momenta, masses, total_mass)
    raise ValueError("table must be table5 or table7")


def _map_reference(
    values: Sequence[float], *, bridge: Any, table: str, masses: Sequence[float], total_mass: float,
    mirrored: bool,
) -> tuple[tuple[float, ...], tuple[tuple[float, float, float, float], ...]]:
    point = _point_from_dimensionless(values, total_mass)
    momenta = _d5._momenta(bridge.numeric_momenta(point, masses, total_mass), "reference momenta")
    if mirrored:
        momenta = _d5._reflect_mirror(momenta)
    _coordinates, output, _evidence = _event_coordinates(
        momenta, table=table, masses=masses, total_mass=total_mass,
    )
    return output, momenta


def _determinant(matrix: Sequence[Sequence[float]]) -> float:
    width = len(matrix)
    if width != 8 or any(len(row) != width for row in matrix):
        raise ValueError("Yu2022 finite-difference Jacobian must be 8 by 8")
    work = [list(map(float, row)) for row in matrix]
    result = 1.0
    for column in range(width):
        pivot = max(range(column, width), key=lambda row: abs(work[row][column]))
        pivot_value = work[pivot][column]
        if not math.isfinite(pivot_value) or pivot_value == 0.0:
            return 0.0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            result = -result
        pivot_value = work[column][column]
        result *= pivot_value
        for row in range(column + 1, width):
            multiplier = work[row][column] / pivot_value
            for item in range(column + 1, width):
                work[row][item] -= multiplier * work[column][item]
            work[row][column] = 0.0
    return result


def _wrap_delta(value: float) -> float:
    return (value + math.pi) % TWO_PI - math.pi


def _finite_difference_jacobian(
    values: Sequence[float], step: float, *, bridge: Any, table: str, masses: Sequence[float],
    total_mass: float, mirrored: bool,
) -> dict[str, Any]:
    center, _ = _map_reference(values, bridge=bridge, table=table, masses=masses, total_mass=total_mass, mirrored=mirrored)
    matrix = [[0.0] * 8 for _ in range(8)]
    for column in range(8):
        plus, minus = list(values), list(values)
        plus[column] += step
        minus[column] -= step
        y_plus, _ = _map_reference(plus, bridge=bridge, table=table, masses=masses, total_mass=total_mass, mirrored=mirrored)
        y_minus, _ = _map_reference(minus, bridge=bridge, table=table, masses=masses, total_mass=total_mass, mirrored=mirrored)
        for row in range(8):
            delta = _wrap_delta(y_plus[row] - y_minus[row]) if row in (6, 7) else y_plus[row] - y_minus[row]
            matrix[row][column] = delta / (2.0 * step)
    det = _determinant(matrix)
    exponent = 6 if table == "table5" else 4
    real_jacobian = abs(det) * total_mass**exponent
    if not math.isfinite(det) or not math.isfinite(real_jacobian):
        raise ValueError("Yu2022 finite-difference Jacobian is non-finite")
    return {
        "step": step,
        "matrix": tuple(tuple(row) for row in matrix),
        "dimensionless_determinant_signed": det,
        "dimensionless_determinant_absolute": abs(det),
        "real_dimension_jacobian_absolute": real_jacobian,
        "dimension_factor": f"M^{exponent}",
        "output_center": center,
        "periodic_output_rows": (6, 7),
    }


def _physical_snapshot(
    momenta: Sequence[Sequence[float]], masses: Sequence[float], total_mass: float,
) -> dict[str, Any]:
    vectors = _d5._momenta(momenta, "physical-check momenta")
    total = _d5._sum_vectors(vectors)
    energy_residual = abs(total[0] - total_mass) / max(total_mass, 1.0e-300)
    spatial_residual = max(abs(total[i]) for i in range(1, 4)) / max(total_mass, 1.0e-300)
    shell = tuple(
        abs(_d5._minkowski_square(vector) - mass * mass)
        / max(_d5._minkowski_square(vector), mass * mass, total_mass * total_mass, 1.0e-300)
        for vector, mass in zip(vectors, masses)
    )
    return {
        "positive_energies": tuple(vector[0] for vector in vectors),
        "energy_conservation_relative_residual": energy_residual,
        "spatial_conservation_relative_residual": spatial_residual,
        "on_shell_relative_residual_by_particle": shell,
    }


def _relative_residual(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-300)


def source_discrepancies(masses: Sequence[float]) -> tuple[dict[str, Any], ...]:
    mass2 = tuple(float(value) ** 2 for value in masses)
    if len(mass2) != 4 or any(not math.isfinite(value) for value in mass2):
        raise ValueError("source discrepancy metadata requires four finite masses")
    return (
        {
            "source": dict(YU2022_TABLE_SOURCES["table5"]),
            "table": 5,
            "quantity": "E3 printed relation",
            "status": "source_discrepancy_recorded_not_used_in_density",
            "absolute_numerator_residual": 2.0 * mass2[2],
            "residual_expression": "2*m3^2",
            "note": "The Table 5 E3 line is inconsistent with the momentum-conservation relation; this audit records the identified residual and does not use that line to evaluate the density.",
        },
        {
            "source": dict(YU2022_TABLE_SOURCES["table7"]),
            "table": 7,
            "quantity": "s23 printed relation",
            "status": "source_discrepancy_recorded_not_used_in_density",
            "printed_relation": "s23 = 2*M*E2 - sum_i(m_i^2) - s12 - s24",
            "momentum_conservation_relation": "s23 = 2*M*E2 + sum_i(m_i^2) - s12 - s24",
            "absolute_numerator_residual": 2.0 * sum(mass2),
            "residual_expression": "2*sum_i(m_i^2)",
            "note": "The page image was visually checked; the mass-squared sum sign in the printed relation disagrees with conservation.",
        },
    )


def _safe_report(report: Mapping[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(item) for item in value]
        return value
    return convert(report)


def check_yu2022_reference(
    bridge: Any,
    *,
    masses: Sequence[float],
    total_mass: float,
    points: Iterable[Mapping[str, float]] | None = None,
    normalization: str | None = None,
    relative_tolerance: float = 1.0e-9,
    fd_relative_tolerance: float = 5.0e-6,
    kinematic_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Compare the two printed external density coefficients after pullback.

    Only the density coefficient and its coordinate pullback are checked.  A
    finite sample cannot resolve the source discrepancies or validate each
    printed kinematic equation, so whole_table remains pending.
    """

    try:
        mass_values = tuple(float(value) for value in masses)
        parent_mass = float(total_mass)
        requested_tolerance = float(relative_tolerance)
        requested_fd_tolerance = float(fd_relative_tolerance)
        kin_tolerance = float(kinematic_tolerance)
    except (TypeError, ValueError, OverflowError):
        mass_values, parent_mass = (), math.nan
        requested_tolerance = requested_fd_tolerance = kin_tolerance = math.nan
    effective_tolerance = max(requested_tolerance, requested_fd_tolerance) if all(
        math.isfinite(value) for value in (requested_tolerance, requested_fd_tolerance)
    ) else math.nan
    try:
        if points is None:
            from .benchmark import _default_points
            generated = _default_points(mass_values, parent_mass)
            sample: tuple[Any, ...] = () if generated is None else tuple(generated)
        else:
            sample = tuple(points)
    except TypeError:
        sample = ()
    convention = normalization or getattr(bridge, "normalization", "")
    discrepancies = source_discrepancies(mass_values) if len(mass_values) == 4 else ()

    def base_report(status: str, message: str | None = None) -> dict[str, Any]:
        return _safe_report({
            "artifact_type": "yu2022_external_density_pullback_check",
            "schema_version": YU2022_SCHEMA_VERSION,
            "source": _report_source(),
            "status": status,
            "coverage": "density-coefficient-and-coordinate-pullback only",
            "whole_table": "pending/source_discrepancy",
            "source_status": "not_checked",
            "normalization": convention,
            "masses": mass_values,
            "total_mass": parent_mass,
            "requested_relative_tolerance": requested_tolerance,
            "requested_fd_relative_tolerance": requested_fd_tolerance,
            "effective_density_tolerance": effective_tolerance,
            "kinematic_tolerance": kin_tolerance,
            "effective_kinematic_residual_tolerance": 10.0 * kin_tolerance,
            "finite_difference_steps": FD_STEPS,
            "source_discrepancies": discrepancies,
            "diagnostics": (message,) if message else (),
            "point_count": len(sample),
            "unique_point_count": len(sample),
            "point_evidence": (),
            "tables": {
                "table5": {"source": dict(YU2022_TABLE_SOURCES["table5"]), "status": status},
                "table7": {"source": dict(YU2022_TABLE_SOURCES["table7"]), "status": status},
            },
            "scope": "local_density_coefficient_and_coordinate_pullback",
            "global_claim": "not_a_full_table_or_global_formula_proof",
        })

    if len(mass_values) != 4 or any(not math.isfinite(value) or value <= 0.0 for value in mass_values):
        return base_report("unsupported", "four finite, strictly positive masses are required")
    if not math.isfinite(parent_mass) or parent_mass <= sum(mass_values):
        return base_report("unsupported", "total_mass must be strictly above the four-body threshold")
    if not all(math.isfinite(value) and value > 0.0 for value in (requested_tolerance, requested_fd_tolerance, kin_tolerance)):
        return base_report("unsupported", "all tolerances must be finite and positive")
    if convention not in {"jing_psf", "standard_2pi4"}:
        return base_report("unsupported", "normalization must be jing_psf or standard_2pi4")
    if getattr(bridge, "profile", None) != "sequential_12_34" or getattr(bridge, "status", None) != "built":
        return base_report("unsupported", "bridge must be a built sequential_12_34 reference formula")
    if tuple(getattr(bridge, "variables", ())) != REFERENCE_VARIABLES:
        return base_report("unsupported", "bridge variables do not match the fixed sequential 12/34 reference")
    if getattr(bridge, "normalization", None) != convention:
        return base_report("failed", "requested normalization disagrees with the bridge normalization")
    if not callable(getattr(bridge, "numeric_momenta", None)) or not callable(getattr(bridge, "numeric_density", None)):
        return base_report("blocked", "reference momentum or density adapter is missing")
    if len(sample) < 8:
        return base_report("unsupported", "at least eight unique reference points are required")
    normalized: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
    try:
        for raw in sample:
            point = _validate_reference_point(raw)
            key = tuple(point[name] for name in REFERENCE_VARIABLES)
            if key in seen:
                return base_report("unsupported", "reference sample contains duplicate points")
            seen.add(key)
            normalized.append(point)
    except (TypeError, ValueError, OverflowError) as exc:
        return base_report("failed", f"invalid reference point: {exc}")
    if len(seen) < 8:
        return base_report("unsupported", "at least eight unique reference points are required")

    point_evidence: list[dict[str, Any]] = []
    table_residuals: dict[str, list[float]] = {"table5": [], "table7": []}
    table_failures: dict[str, list[str]] = {"table5": [], "table7": []}
    for point_index, point in enumerate(normalized):
        x = (
            point["s12"] / parent_mass**2, point["s34"] / parent_mass**2,
            point["theta0"], point["phi0"], point["theta12"], point["phi12"],
            point["theta34"], point["phi34"],
        )
        row: dict[str, Any] = {"point_index": point_index, "reference_coordinates": point, "tables": {}}
        try:
            reference_momenta = _d5._momenta(bridge.numeric_momenta(point, mass_values, parent_mass), "reference momenta")
            rho_reference = float(bridge.numeric_density(point, mass_values, parent_mass))
            if not math.isfinite(rho_reference) or rho_reference <= 0.0:
                raise ValueError("reference density must be finite and strictly positive")
            source_sine = math.prod(math.sin(point[name]) for name in ("theta0", "theta12", "theta34"))
            source_weighted = rho_reference * source_sine
            if not math.isfinite(source_weighted) or source_weighted <= 0.0:
                raise ValueError("reference sine-weighted density must be finite and positive")
            mirror_events = (
                ("source", reference_momenta),
                ("p1p2_plane_reflection", _d5._reflect_mirror(reference_momenta)),
            )
            row["reference_density"] = rho_reference
            row["reference_sine_weighted_density"] = source_weighted
            row["source_sine_factor"] = source_sine
            row["mirrors"] = {}
            for mirror_name, mirror_momenta in mirror_events:
                physical = _physical_snapshot(mirror_momenta, mass_values, parent_mass)
                physical_tolerance = 10.0 * kin_tolerance
                if (
                    physical["energy_conservation_relative_residual"] > physical_tolerance
                    or physical["spatial_conservation_relative_residual"] > physical_tolerance
                    or any(value > physical_tolerance for value in physical["on_shell_relative_residual_by_particle"])
                    or any(value <= 0.0 for value in physical["positive_energies"])
                ):
                    raise ValueError(f"reference {mirror_name} momenta fail independent shell/positive-energy/conservation checks")
                row["mirrors"][mirror_name] = {
                    "momenta": mirror_momenta,
                    "physical_snapshot": physical,
                }
                for table in ("table5", "table7"):
                    table_row = row["tables"].setdefault(table, {"mirrors": {}})
                    try:
                        values, _target_center, coordinate_evidence = _event_coordinates(
                            mirror_momenta, table=table, masses=mass_values, total_mass=parent_mass,
                        )
                        density_data = (
                            table5_density(mirror_momenta, mass_values, parent_mass, normalization=convention)
                            if table == "table5" else
                            table7_density(mirror_momenta, mass_values, parent_mass, normalization=convention)
                        )
                        jacobians = tuple(_finite_difference_jacobian(
                            x, step, bridge=bridge, table=table, masses=mass_values,
                            total_mass=parent_mass, mirrored=mirror_name != "source",
                        ) for step in FD_STEPS)
                        comparisons = []
                        for jacobian in jacobians:
                            lhs = density_data["rho_per_mirror"] * math.sin(values["theta"]) * jacobian["real_dimension_jacobian_absolute"]
                            residual = _relative_residual(lhs, source_weighted)
                            passed = math.isfinite(residual) and residual <= effective_tolerance
                            table_residuals[table].append(residual)
                            if not passed:
                                table_failures[table].append(
                                    f"point {point_index} {mirror_name} h={jacobian['step']:.1e}: residual {residual:.6g}"
                                )
                            comparisons.append({
                                "step": jacobian["step"],
                                "external_pullback_density": lhs,
                                "reference_pullback_density": source_weighted,
                                "absolute_residual": abs(lhs - source_weighted),
                                "relative_residual": residual,
                                "passes_effective_tolerance": passed,
                            })
                        table_row["mirrors"][mirror_name] = {
                            "coordinates": values,
                            "coordinate_evidence": coordinate_evidence,
                            "external_density_data": density_data,
                            "jacobians": jacobians,
                            "density_pullback_comparisons": tuple(comparisons),
                        }
                    except Exception as exc:
                        table_failures[table].append(f"point {point_index} {mirror_name}: {type(exc).__name__}: {exc}")
                        table_row["mirrors"][mirror_name] = {"error": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:
            message = f"point {point_index}: {type(exc).__name__}: {exc}"
            for table in ("table5", "table7"):
                table_failures[table].append(message)
        point_evidence.append(row)

    table_summaries = {}
    for table in ("table5", "table7"):
        residuals = table_residuals[table]
        table_summaries[table] = {
            "source": dict(YU2022_TABLE_SOURCES[table]),
            "status": "verified_local" if not table_failures[table] and residuals else "failed",
            "compared": len(residuals),
            "expected_comparisons": len(normalized) * 2 * len(FD_STEPS),
            "max_relative_residual": max(residuals, default=None),
            "failures": tuple(table_failures[table]),
            "paper_density_per_mirror": (
                "A_g/[512*(2*pi)^12*M^3*k1*k2*k3]" if table == "table5" else
                "A_g*gamma_beta*k1star/[256*(2*pi)^12*M^2*k1*k2*k3]"
            ),
            "coordinate_variables": (
                TABLE5_INVARIANTS + ("theta", "phi", "psi") if table == "table5" else
                TABLE7_INVARIANTS + ("c1star", "theta", "phi", "psi")
            ),
            "dimensionful_jacobian_factor": "M^6" if table == "table5" else "M^4",
        }
    overall = "verified_local" if all(item["status"] == "verified_local" for item in table_summaries.values()) else "failed"
    report = {
        "artifact_type": "yu2022_external_density_pullback_check",
        "schema_version": YU2022_SCHEMA_VERSION,
        "source": _report_source(),
        "status": overall,
        "coverage": "density-coefficient-and-coordinate-pullback only",
        "whole_table": "pending/source_discrepancy",
        "source_status": "not_checked",
        "normalization": convention,
        "masses": mass_values,
        "total_mass": parent_mass,
        "requested_relative_tolerance": requested_tolerance,
        "requested_fd_relative_tolerance": requested_fd_tolerance,
        "effective_density_tolerance": effective_tolerance,
        "kinematic_tolerance": kin_tolerance,
        "effective_kinematic_residual_tolerance": 10.0 * kin_tolerance,
        "finite_difference_steps": FD_STEPS,
        "finite_difference_protocol": {
            "source_coordinates": "(s12/M^2,s34/M^2,theta0,phi0,theta12,phi12,theta34,phi34)",
            "table5_target_coordinates": "(s12,s13,s14,s23,s124)/M^2,theta,phi,psi",
            "table7_target_coordinates": "(s12,s13,s24,s34)/M^2,c1star,theta,phi,psi",
            "periodic_output_rows": (6, 7),
            "table5_real_jacobian": "abs(det(J_dimensionless))*M^6",
            "table7_real_jacobian": "abs(det(J_dimensionless))*M^4",
        },
        "density_comparison": "rho_Yu*sin(theta)*abs(J_real) versus rho_reference*sin(theta0)*sin(theta12)*sin(theta34)",
        "source_discrepancies": source_discrepancies(mass_values),
        "tables": table_summaries,
        "point_count": len(normalized),
        "unique_point_count": len(seen),
        "expected_pullback_comparisons_per_table": len(normalized) * 2 * len(FD_STEPS),
        "point_evidence": tuple(point_evidence),
        "scope": "local_density_coefficient_and_coordinate_pullback",
        "global_claim": "not_a_full_table_or_global_formula_proof",
    }
    return _safe_report(report)


def compare_all_loop_export_to_yu2022(
    export_directory: str | Path, paper_report: Mapping[str, Any], *, tolerance: float | None = None,
) -> dict[str, Any]:
    """Match saved per-class benchmark evidence to the fixed-reference paper pullback.

    The helper consumes only exported `.evidence.json` files and never rebuilds
    a graph, formula, or direct-path numeric adapter.
    """

    root = Path(export_directory)
    index_path = root / "index.json"
    base = {
        "artifact_type": "yu2022_all_loop_export_crosscheck",
        "schema_version": YU2022_SCHEMA_VERSION,
        "export_directory": str(root),
        "source": paper_report.get("source", {}),
        "coverage": "density-coefficient-and-coordinate-pullback only",
        "whole_table": "pending/source_discrepancy",
        "source_status": "not_checked",
    }
    def global_failure(message: str) -> dict[str, Any]:
        return {**base, "status": "failed", "class_count": 0, "class_results": [],
                "diagnostics": [message]}

    if paper_report.get("status") != "verified_local":
        return global_failure(
            f"fixed-reference paper report status is {paper_report.get('status', 'missing')!r}, not verified_local"
        )
    normalization = paper_report.get("normalization")
    if normalization not in SUPPORTED_NORMALIZATIONS:
        return global_failure("fixed-reference normalization is missing or unsupported")
    try:
        global_tolerance = float(
            tolerance if tolerance is not None else paper_report.get("effective_density_tolerance")
        )
    except (TypeError, ValueError, OverflowError):
        global_tolerance = math.nan
    if not math.isfinite(global_tolerance) or global_tolerance <= 0.0:
        return global_failure("global density tolerance must be finite and strictly positive")

    if not index_path.is_file():
        return {**base, "status": "pending_export", "class_count": 0, "class_results": [],
                "diagnostics": ["all-loop export index.json is not present; only the fixed-reference check is available"]}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {**base, "status": "failed", "class_count": 0, "class_results": [],
                "diagnostics": [f"cannot read all-loop index: {exc}"]}
    classes = index.get("classes", [])
    if not isinstance(classes, list):
        return global_failure("all-loop index classes field is not an array")
    paper_points = paper_report.get("point_evidence", [])
    if not isinstance(paper_points, list):
        return global_failure("fixed-reference point_evidence field is not an array")
    paper_by_key: dict[tuple[float, ...], Mapping[str, Any]] = {}
    paper_key_errors = []
    for row in paper_points:
        if not isinstance(row, Mapping):
            paper_key_errors.append("fixed-reference point entry is not an object")
            continue
        point = row.get("reference_coordinates", {})
        try:
            key = tuple(float(point[name]) for name in REFERENCE_VARIABLES)
        except (KeyError, TypeError, ValueError, OverflowError):
            paper_key_errors.append("fixed-reference point has malformed coordinates")
            continue
        if any(not math.isfinite(value) for value in key):
            paper_key_errors.append("fixed-reference point has nonfinite coordinates")
            continue
        if key in paper_by_key:
            paper_key_errors.append("fixed-reference point_evidence repeats a reference-coordinate key")
            continue
        paper_by_key[key] = row
    if paper_key_errors or len(paper_points) != 8 or len(paper_by_key) != 8:
        details = paper_key_errors or ["fixed-reference point_evidence must contain 8 unique reference-coordinate keys"]
        return global_failure("; ".join(details))
    results: list[dict[str, Any]] = []
    expected_loop_by_id = {
        f"L{loop_count}-C{index:03d}": loop_count
        for loop_count, class_count in enumerate(EXPECTED_CLASS_COUNTS_BY_LOOP)
        for index in range(1, class_count + 1)
    }
    expected_ids = set(expected_loop_by_id)
    index_class_ids = [str(entry.get("class_id", "")) if isinstance(entry, Mapping) else "" for entry in classes]
    index_ids_match = (
        len(classes) == len(expected_ids)
        and len(set(index_class_ids)) == len(expected_ids)
        and set(index_class_ids) == expected_ids
    )

    def positive_pullbacks_by_step(
        rows: Any, *, value_field: str, label: str,
    ) -> dict[float, float]:
        if not isinstance(rows, list):
            raise ValueError(f"{label} pullback evidence is not an array")
        values: dict[float, float] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} pullback entry is not an object")
            try:
                step = float(row["step"])
                value = float(row[value_field])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{label} pullback entry is malformed") from exc
            if not math.isfinite(step) or step <= 0.0:
                raise ValueError(f"{label} pullback finite-difference step must be positive and finite")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} pullback density must be positive and finite")
            if step in values:
                raise ValueError(f"{label} pullback evidence repeats h={step:.1e}")
            values[step] = value
        if set(values) != set(FD_STEPS):
            raise ValueError(f"{label} pullback evidence must contain exactly the two configured FD steps")
        return values

    for entry in classes:
        if not isinstance(entry, Mapping):
            results.append({"class_id": "", "status": "failed_or_incomplete", "compared_points": 0,
                            "compared_pullbacks": 0, "errors": ["index class entry is not an object"]})
            continue
        class_id = str(entry.get("class_id", ""))
        evidence_name = entry.get("evidence")
        class_row = {
            "class_id": class_id, "loop_count": entry.get("loop_count"),
            "topology_id": entry.get("topology_id"),
            "representative_fixed_digest": entry.get("representative_fixed_digest"),
            "benchmark_status": entry.get("benchmark_status", "pending"),
            "source_status": entry.get("source_status", "not_checked"),
            "evidence_file": str(evidence_name),
        }
        if not evidence_name:
            results.append({**class_row, "status": "pending_schema", "compared": 0,
                            "diagnostic": "index entry has no evidence path"})
            continue
        evidence_path = root / str(evidence_name)
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            results.append({**class_row, "status": "pending_schema", "compared": 0,
                            "diagnostic": f"cannot read class evidence: {exc}"})
            continue
        if not isinstance(evidence, Mapping):
            results.append({**class_row, "status": "pending_schema", "compared": 0,
                            "diagnostic": "class evidence top-level value is not an object"})
            continue
        benchmark = evidence.get("benchmark", {})
        benchmark_evidence = benchmark.get("evidence", {}) if isinstance(benchmark, Mapping) else {}
        point_rows = benchmark_evidence.get("point_evidence", ()) if isinstance(benchmark_evidence, Mapping) else ()
        if not isinstance(point_rows, list) or not point_rows:
            results.append({**class_row, "status": "pending_benchmark", "compared": 0,
                            "diagnostic": "class benchmark has no replayable point_evidence"})
            continue
        errors: list[str] = []
        expected_loop_count = expected_loop_by_id.get(class_id)
        if class_id not in expected_ids:
            errors.append("index class_id is outside the required L0-C001..L3-C106 coverage")
        if expected_loop_count is None or entry.get("loop_count") != expected_loop_count:
            errors.append("index class_id and loop_count are inconsistent")
        if not isinstance(entry.get("topology_id"), str) or not entry.get("topology_id"):
            errors.append("index topology_id is missing")
        elif expected_loop_count is not None and not entry["topology_id"].startswith(f"L{expected_loop_count}-T"):
            errors.append("index topology_id and loop_count are inconsistent")
        if not isinstance(entry.get("representative_fixed_digest"), str) or not entry.get("representative_fixed_digest"):
            errors.append("index representative_fixed_digest is missing")
        for identity_field in ("class_id", "loop_count", "topology_id", "representative_fixed_digest"):
            index_value = entry.get(identity_field)
            evidence_value = evidence.get(identity_field)
            if index_value is None or evidence_value is None or index_value != evidence_value:
                errors.append(f"index/evidence {identity_field} mismatch or missing value")
        try:
            class_masses = tuple(float(value) for value in benchmark_evidence["masses"])
            class_total_mass = float(benchmark_evidence["total_mass"])
        except (KeyError, TypeError, ValueError, OverflowError):
            class_masses, class_total_mass = (), math.nan
        expected_masses = tuple(float(value) for value in paper_report.get("masses", ()))
        expected_total_mass = float(paper_report.get("total_mass", math.nan))
        if class_masses != expected_masses or class_total_mass != expected_total_mass:
            errors.append("class benchmark masses/total_mass do not match the fixed-reference paper report")
        class_normalization = evidence.get("normalization")
        if class_normalization not in SUPPORTED_NORMALIZATIONS:
            errors.append("class export normalization is missing or unsupported")
        elif class_normalization != normalization:
            errors.append("class export normalization does not match the fixed-reference paper report")
        if benchmark.get("status") != "verified_local":
            errors.append(f"class direct benchmark status is {benchmark.get('status', 'missing')!r}, not verified_local")
        replay_rows: list[dict[str, Any]] = []
        if len(point_rows) != 8:
            errors.append(f"class benchmark must contain exactly 8 reference points, found {len(point_rows)}")
        seen_class_keys: set[tuple[float, ...]] = set()
        for point_row in point_rows:
            if not isinstance(point_row, Mapping):
                errors.append("class benchmark point entry is not an object")
                continue
            reference_point = point_row.get("reference_coordinates", {})
            try:
                key = tuple(float(reference_point[name]) for name in REFERENCE_VARIABLES)
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append("class benchmark point has malformed reference coordinates")
                continue
            if any(not math.isfinite(value) for value in key):
                errors.append("class benchmark point has nonfinite reference coordinates")
                continue
            if key in seen_class_keys:
                errors.append("class benchmark repeats a reference-coordinate key")
                continue
            seen_class_keys.add(key)
            paper_point = paper_by_key.get(key)
            if paper_point is None:
                errors.append("class benchmark point has no matching fixed-reference paper point")
                continue
            direct_pullbacks = point_row.get("sine_weighted_density_comparison", ())
            try:
                direct_by_step = positive_pullbacks_by_step(
                    direct_pullbacks, value_field="lhs", label="direct",
                )
            except ValueError as exc:
                errors.append(str(exc))
                direct_by_step = {}
            table_values = {}
            for table in ("table5", "table7"):
                mirror = paper_point.get("tables", {}).get(table, {}).get("mirrors", {}).get("source", {})
                paper_steps = mirror.get("density_pullback_comparisons", ())
                try:
                    paper_by_step = positive_pullbacks_by_step(
                        paper_steps, value_field="external_pullback_density", label=f"paper {table}",
                    )
                except ValueError as exc:
                    errors.append(str(exc))
                    paper_by_step = {}
                comparisons = []
                for step in FD_STEPS:
                    if step not in direct_by_step or step not in paper_by_step:
                        errors.append(f"class benchmark or {table} evidence is missing h={step:.1e}")
                        continue
                    lhs = direct_by_step[step]
                    rhs = paper_by_step[step]
                    residual = _relative_residual(lhs, rhs)
                    comparisons.append({"step": step, "direct_pullback": lhs, "paper_pullback": rhs,
                                        "relative_residual": residual, "passes_tolerance": residual <= global_tolerance})
                    if not math.isfinite(residual) or residual > global_tolerance:
                        errors.append(f"{table} direct-to-paper pullback residual {residual:.6g} at h={step:.1e}")
                table_values[table] = comparisons
            replay_rows.append({"reference_coordinates": reference_point,
                                "direct_coordinates": point_row.get("target_coordinates") or point_row.get("recovered_d5_coordinates"),
                                "direct_sine_weighted_density_by_step": direct_by_step,
                                "table_pullback_comparisons": table_values})
        if seen_class_keys != set(paper_by_key):
            errors.append("class benchmark reference-coordinate keys do not cover the 8 fixed-reference points exactly")
        pullback_count = sum(
            len(comparisons)
            for row in replay_rows
            for comparisons in row["table_pullback_comparisons"].values()
        )
        class_complete = len(replay_rows) == 8 and pullback_count == 8 * len(FD_STEPS) * 2 and not errors
        results.append({**class_row, "status": "verified_local" if class_complete else "failed_or_incomplete",
                        "compared_points": len(replay_rows), "compared_pullbacks": pullback_count,
                        "expected_pullbacks": 8 * len(FD_STEPS) * 2, "max_relative_residual": max((
                            item["relative_residual"] for row in replay_rows
                            for comparisons in row["table_pullback_comparisons"].values() for item in comparisons
                        ), default=None), "errors": errors, "point_evidence": replay_rows})
    for failure in index.get("failures", ()):
        results.append({
            "class_id": str(failure.get("class_id", "")),
            "loop_count": failure.get("loop_count"),
            "topology_id": failure.get("topology_id"),
            "representative_fixed_digest": failure.get("representative_fixed_digest"),
            "benchmark_status": failure.get("benchmark_status", "pending"),
            "source_status": failure.get("source_status", "not_checked"),
            "evidence_file": failure.get("evidence_json"),
            "status": "export_build_failed",
            "compared_points": 0,
            "compared_pullbacks": 0,
            "expected_pullbacks": 8 * len(FD_STEPS) * 2,
            "errors": [str(failure.get("error", "class export build failed"))],
        })
    expected_count = 143
    class_ids = [row.get("class_id") for row in results]
    all_classes = (
        index_ids_match
        and len(results) == expected_count
        and len(set(class_ids)) == expected_count
        and set(class_ids) == expected_ids
    )
    complete = all_classes and all(row["status"] == "verified_local" for row in results)
    diagnostics = []
    if not index_ids_match:
        diagnostics.append("all-loop index class IDs must match the exact L0-C001..L3-C106 coverage")
    if not complete:
        diagnostics.append("export schema/evidence is missing, partial, inconsistent, or not locally verified for all 143 classes")
    return _safe_report({
        **base,
        "status": "verified_local" if complete else "partial_or_pending",
        "export_schema_version": index.get("schema_version"),
        "exported_class_count": len(classes),
        "export_build_failure_count": len(index.get("failures", ())),
        "expected_all_loop_class_count": expected_count,
        "all_143_present": all_classes,
        "expected_class_counts_by_loop": {
            f"L{loop_count}": class_count
            for loop_count, class_count in enumerate(EXPECTED_CLASS_COUNTS_BY_LOOP)
        },
        "all_expected_class_ids_present": index_ids_match,
        "class_count": len(results),
        "global_tolerance": global_tolerance,
        "normalization": normalization,
        "expected_points_per_class": 8,
        "finite_difference_steps_per_table": len(FD_STEPS),
        "tables_compared_per_point": 2,
        "expected_pullback_comparisons_per_class": 8 * len(FD_STEPS) * 2,
        "expected_pullback_comparisons_all_143": expected_count * 8 * len(FD_STEPS) * 2,
        "actual_pullback_comparisons": sum(row.get("compared_pullbacks", 0) for row in results),
        "class_status_counts": {
            status: sum(row["status"] == status for row in results)
            for status in sorted({row["status"] for row in results})
        },
        "class_results": results,
        "diagnostics": diagnostics,
    })


def render_yu2022_markdown(report: Mapping[str, Any]) -> str:
    tables = report.get("tables", {})
    lines = [
        "# Yu2022 Table 5/7 外部密度校对",
        "",
        f"- 状态：`{report.get('status', 'unknown')}`。",
        f"- 范围：`{report.get('coverage', 'unknown')}`。",
        f"- 整表状态：`{report.get('whole_table', 'pending/source_discrepancy')}`。",
        f"- Source status：`{report.get('source_status', 'not_checked')}`。",
        f"- 来源：{report.get('source', {}).get('citation', YU2022_SOURCE)}；PDF SHA-256 `{report.get('source', {}).get('pdf_sha256', YU2022_PDF_SHA256)}`。",
        f"- 归一化：`{report.get('normalization', 'unknown')}`；点数：{report.get('point_count', 0)}。",
        f"- 两档有限差分：`{report.get('finite_difference_steps', FD_STEPS)}`；比较阈值：`{report.get('effective_density_tolerance', 'unknown')}`。",
        "",
        "本校对只检验密度系数与坐标拉回，不表示整张表的所有印刷关系均已验证。",
        "",
        "| Table | PDF 页 | 期刊页 | 状态 | 比较数 | 最大相对残差 | 变量 | Jacobian 因子 |",
        "|---:|---:|---|---|---:|---:|---|---|",
    ]
    for table in ("table5", "table7"):
        row = tables.get(table, {})
        table_source = row.get("source", YU2022_TABLE_SOURCES[table])
        lines.append(
            f"| {table.removeprefix('table')} | {table_source.get('pdf_page', '—')} | "
            f"{table_source.get('journal_page', '—')} | `{row.get('status', 'pending')}` | {row.get('compared', 0)} | "
            f"{row.get('max_relative_residual', '—')} | `{row.get('coordinate_variables', ())}` | "
            f"`{row.get('dimensionful_jacobian_factor', '—')}` |"
        )
    lines.extend(["", "## 来源印式偏差", ""])
    for item in report.get("source_discrepancies", ()):
        lines.append(
            f"- Table {item.get('table')} (PDF page {item.get('source', {}).get('pdf_page', 'unknown')}, "
            f"journal page {item.get('source', {}).get('journal_page', 'unknown')}), {item.get('quantity')}："
            f"残差量级 `{item.get('residual_expression')}`；"
            f"质量取值对应的分子残差 `{item.get('absolute_numerator_residual')}`。{item.get('note', '')}"
        )
    lines.extend(["", "## 固定参考点逐点证据", ""])
    for point in report.get("point_evidence", ()):
        source = point.get("reference_coordinates", {})
        lines.extend([
            f"### Point {point.get('point_index', '?')}",
            "",
            f"- Source coordinates: `{json.dumps(source, ensure_ascii=False, sort_keys=True)}`",
            f"- Reference pullback: `{point.get('reference_sine_weighted_density', 'unavailable')}`",
            "",
        ])
        for table in ("table5", "table7"):
            mirrors = point.get("tables", {}).get(table, {}).get("mirrors", {})
            for mirror_name in ("source", "p1p2_plane_reflection"):
                mirror = mirrors.get(mirror_name, {})
                if "error" in mirror:
                    lines.append(f"- {table} / {mirror_name}: error `{mirror['error']}`")
                    continue
                lines.extend([
                    f"#### {table.removeprefix('table')} / {mirror_name}",
                    "",
                    f"- Coordinates: `{json.dumps(mirror.get('coordinates', {}), ensure_ascii=False, sort_keys=True)}`",
                    f"- Per-mirror density: `{mirror.get('external_density_data', {}).get('rho_per_mirror', 'unavailable')}`",
                    "",
                    "| h | external pullback | reference pullback | relative residual | pass |",
                    "|---:|---:|---:|---:|---|",
                ])
                for item in mirror.get("density_pullback_comparisons", ()):
                    lines.append(
                        f"| {item.get('step')} | {item.get('external_pullback_density')} | "
                        f"{item.get('reference_pullback_density')} | {item.get('relative_residual')} | "
                        f"{item.get('passes_effective_tolerance')} |"
                    )
                for jacobian in mirror.get("jacobians", ()):
                    lines.extend([
                        "",
                        f"Finite-difference matrix, h={jacobian.get('step')}:",
                        "",
                        "```json",
                        json.dumps(jacobian.get("matrix", ()), ensure_ascii=False, separators=(",", ":")),
                        "```",
                    ])
                lines.append("")
    diagnostics = report.get("diagnostics", ())
    if diagnostics:
        lines.extend(["", "## 诊断", "", *(f"- {item}" for item in diagnostics)])
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "YU2022_PDF_SHA256", "YU2022_SOURCE", "YU2022_TABLE_SOURCES", "REFERENCE_VARIABLES", "TABLE5_INVARIANTS",
    "TABLE7_INVARIANTS", "FD_STEPS", "table5_coordinates", "table7_coordinates",
    "table5_density", "table7_density", "source_discrepancies", "check_yu2022_reference",
    "compare_all_loop_export_to_yu2022", "render_yu2022_markdown",
]
