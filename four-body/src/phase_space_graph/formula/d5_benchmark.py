"""Independent finite-point check for the four-body direct Gram chart.

This module reconstructs the target invariants and Euler angles from the
sequential reference momenta.  It does not use the direct Gram map to produce
those target coordinates.  The report is strictly local numerical evidence;
it never promotes a formula or source status to ``verified``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .records import BenchmarkReport


_REFERENCE_VARIABLES = (
    "s12", "s34", "theta0", "phi0", "theta12", "phi12", "theta34", "phi34",
)
_TWO_PI = 2.0 * math.pi
_FD_STEPS = (2.0e-5, 1.0e-5)
_LABEL = re.compile(r"^s_?(?:\{)?([1-4]{2,4})(?:\})?$")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _display(value: Any) -> Any:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return repr(value)
    return result if math.isfinite(result) else repr(result)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_json_safe(item) for item in value)
    return value


def _make_report(
    status: str,
    profile: str,
    sample: Sequence[Any],
    *,
    compared: int = 0,
    max_absolute: float | None = None,
    max_relative: float | None = None,
    max_coordinate: float | None = None,
    branch_status: str = "not_checked",
    failures: Sequence[Mapping[str, Any]] = (),
    diagnostics: Sequence[str] = (),
    evidence: Mapping[str, Any] | None = None,
) -> BenchmarkReport:
    def finite_or_none(value: float | None) -> float | None:
        return value if value is None or math.isfinite(value) else None

    return BenchmarkReport(
        status=status,
        profile=profile,
        points=len(sample),
        compared=compared,
        max_absolute_residual=finite_or_none(max_absolute),
        max_relative_residual=finite_or_none(max_relative),
        coordinate_residual=finite_or_none(max_coordinate),
        branch_status=branch_status,
        failures=tuple(_json_safe(dict(row)) for row in failures),
        diagnostics=tuple(str(item) for item in diagnostics),
        evidence=_json_safe(dict(evidence or {})),
    )


def _vector(value: Any, label: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} is not a four-vector")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not a finite numeric four-vector") from exc
    if len(result) != 4 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain four finite components")
    if result[0] <= 0.0:
        raise ValueError(f"{label} has non-positive energy")
    return result  # type: ignore[return-value]


def _momenta(value: Any, label: str) -> tuple[tuple[float, float, float, float], ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} is not a four-particle momentum collection")
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{label} is not a momentum collection") from exc
    if len(rows) != 4:
        raise ValueError(f"{label} has {len(rows)} vectors; expected 4")
    return tuple(_vector(row, f"{label}[{index + 1}]") for index, row in enumerate(rows))


def _dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def _cross3(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm3(a: Sequence[float]) -> float:
    return math.sqrt(_dot3(a, a))


def _scale3(a: Sequence[float], factor: float) -> tuple[float, float, float]:
    return tuple(factor * a[index] for index in range(3))  # type: ignore[return-value]


def _minkowski_square(vector: Sequence[float]) -> float:
    return vector[0] * vector[0] - sum(component * component for component in vector[1:])


def _sum_vectors(vectors: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    return tuple(sum(vector[axis] for vector in vectors) for axis in range(4))  # type: ignore[return-value]


def _subset_from_label(label: Any) -> tuple[int, ...]:
    match = _LABEL.fullmatch(str(label))
    if match is None:
        raise ValueError(f"unsupported target invariant label {label!r}")
    digits = tuple(int(item) for item in match.group(1))
    if len(set(digits)) != len(digits):
        raise ValueError(f"target invariant label repeats a particle index: {label!r}")
    return tuple(index - 1 for index in digits)


def _invariants_from_momenta(
    momenta: Sequence[Sequence[float]], labels: Sequence[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for label in labels:
        selected = _subset_from_label(label)
        total = _sum_vectors(tuple(momenta[index] for index in selected))
        result[str(label)] = _minkowski_square(total)
    return result


def _recover_euler_and_sigma(
    momenta: Sequence[Sequence[float]],
) -> tuple[dict[str, float], int, dict[str, Any]]:
    p1 = tuple(momenta[0][axis] for axis in range(1, 4))
    p2 = tuple(momenta[1][axis] for axis in range(1, 4))
    p3 = tuple(momenta[2][axis] for axis in range(1, 4))
    norm1 = _norm3(p1)
    if not math.isfinite(norm1) or norm1 <= 0.0:
        raise ValueError("cannot recover the body z axis from p1")
    z_axis = _scale3(p1, 1.0 / norm1)
    transverse = tuple(p2[index] - _dot3(p2, z_axis) * z_axis[index] for index in range(3))
    norm_transverse = _norm3(transverse)
    if not math.isfinite(norm_transverse) or norm_transverse <= 0.0:
        raise ValueError("cannot recover the body x axis: p2 is collinear with p1")
    x_axis = _scale3(transverse, 1.0 / norm_transverse)
    y_axis = _cross3(z_axis, x_axis)

    # The columns are the rotated body axes in lab coordinates.
    rotation = tuple(
        (x_axis[row], y_axis[row], z_axis[row]) for row in range(3)
    )
    cosine_theta = max(-1.0, min(1.0, rotation[2][2]))
    theta = math.acos(cosine_theta)
    phi = math.atan2(rotation[1][2], rotation[0][2]) % _TWO_PI
    psi = math.atan2(rotation[2][1], -rotation[2][0]) % _TWO_PI
    triple = _dot3(_cross3(p1, p2), p3)
    triple_scale = _norm3(p1) * _norm3(p2) * _norm3(p3)
    if (
        not math.isfinite(triple)
        or not math.isfinite(triple_scale)
        or triple_scale <= 0.0
        or abs(triple) / triple_scale <= 1.0e-14
    ):
        raise ValueError("the oriented triple product is zero or numerically degenerate")
    sigma = 1 if triple > 0.0 else -1
    return (
        {"theta": theta, "phi": phi, "psi": psi},
        sigma,
        {
            "rotation_matrix": rotation,
            "triple_product": triple,
            "normalized_triple_product": triple / triple_scale,
            "sigma": sigma,
            "body_axes_lab": {"x": x_axis, "y": y_axis, "z": z_axis},
        },
    )


def _wrap_delta(value: float) -> float:
    return (value + math.pi) % _TWO_PI - math.pi


def _dimensionless_source(point: Mapping[str, Any], total_mass: float) -> tuple[float, ...]:
    values = tuple(float(point[name]) for name in _REFERENCE_VARIABLES)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("reference point contains a non-finite coordinate")
    if values[0] <= 0.0 or values[1] <= 0.0:
        raise ValueError("reference pair invariant squares must be positive")
    for name in ("theta0", "theta12", "theta34"):
        if not 0.0 <= float(point[name]) <= math.pi:
            raise ValueError(f"{name} lies outside [0, pi]")
    for name in ("phi0", "phi12", "phi34"):
        if not 0.0 <= float(point[name]) < _TWO_PI:
            raise ValueError(f"{name} lies outside [0, 2 pi)")
    scale = total_mass * total_mass
    return (values[0] / scale, values[1] / scale, *values[2:])


def _point_from_dimensionless(x: Sequence[float], total_mass: float) -> dict[str, float]:
    return {
        "s12": float(x[0]) * total_mass * total_mass,
        "s34": float(x[1]) * total_mass * total_mass,
        "theta0": float(x[2]),
        "phi0": float(x[3]) % _TWO_PI,
        "theta12": float(x[4]),
        "phi12": float(x[5]) % _TWO_PI,
        "theta34": float(x[6]),
        "phi34": float(x[7]) % _TWO_PI,
    }


def _map_reference_point(
    x: Sequence[float],
    *,
    bridge: Any,
    labels: Sequence[str],
    masses: Sequence[float],
    total_mass: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, float, float, float], ...]]:
    point = _point_from_dimensionless(x, total_mass)
    momenta = _momenta(bridge.numeric_momenta(point, masses, total_mass), "reference momenta")
    invariants = _invariants_from_momenta(momenta, labels)
    euler, _, _ = _recover_euler_and_sigma(momenta)
    output = tuple(invariants[label] / (total_mass * total_mass) for label in labels) + (
        euler["theta"], euler["phi"], euler["psi"],
    )
    if not all(math.isfinite(value) for value in output):
        raise ValueError("reference-to-target coordinate map produced non-finite values")
    return output, momenta


def _determinant(matrix: Sequence[Sequence[float]]) -> float:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Jacobian matrix must be square")
    work = [list(map(float, row)) for row in matrix]
    determinant = 1.0
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        pivot_value = work[pivot][column]
        if not math.isfinite(pivot_value) or pivot_value == 0.0:
            return 0.0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, n):
            factor = work[row][column] / pivot_value
            for item in range(column + 1, n):
                work[row][item] -= factor * work[column][item]
            work[row][column] = 0.0
    return determinant


def _finite_difference_jacobian(
    x: Sequence[float],
    step: float,
    *,
    bridge: Any,
    labels: Sequence[str],
    masses: Sequence[float],
    total_mass: float,
) -> dict[str, Any]:
    center, _ = _map_reference_point(
        x, bridge=bridge, labels=labels, masses=masses, total_mass=total_mass,
    )
    rows = [[0.0] * 8 for _ in range(8)]
    for column in range(8):
        plus = list(x)
        minus = list(x)
        plus[column] += step
        minus[column] -= step
        y_plus, _ = _map_reference_point(
            plus, bridge=bridge, labels=labels, masses=masses, total_mass=total_mass,
        )
        y_minus, _ = _map_reference_point(
            minus, bridge=bridge, labels=labels, masses=masses, total_mass=total_mass,
        )
        for row in range(8):
            if row in (6, 7):
                difference = _wrap_delta(y_plus[row] - y_minus[row])
            else:
                difference = y_plus[row] - y_minus[row]
            rows[row][column] = difference / (2.0 * step)
    determinant = _determinant(rows)
    if not math.isfinite(determinant):
        raise ValueError("finite-difference Jacobian determinant is non-finite")
    real_determinant = abs(determinant) * total_mass**6
    if not math.isfinite(real_determinant):
        raise ValueError("dimensionful finite-difference Jacobian determinant is non-finite")
    return {
        "step": step,
        "matrix": tuple(tuple(row) for row in rows),
        "dimensionless_determinant_signed": determinant,
        "dimensionless_determinant_absolute": abs(determinant),
        "real_dimension_jacobian_absolute": real_determinant,
        "output_center": center,
    }


def _relative_residual(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-300)


def _validate_momenta(
    momenta: Sequence[Sequence[float]],
    *,
    masses: Sequence[float],
    total_mass: float,
    labels: Sequence[str],
    expected_invariants: Mapping[str, float],
    expected_source_pair: Mapping[str, float],
    expected_sigma: int,
    tolerance: float,
    label: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    errors: list[str] = []
    parent_scale = total_mass * total_mass
    total = _sum_vectors(momenta)
    energy_residual = abs(total[0] - total_mass) / max(abs(total_mass), 1.0e-300)
    momentum_residual = max(abs(total[axis]) for axis in range(1, 4)) / max(abs(total_mass), 1.0e-300)
    if energy_residual > 10.0 * tolerance:
        errors.append(f"{label} energy conservation residual={energy_residual:.6g}")
    if momentum_residual > 10.0 * tolerance:
        errors.append(f"{label} spatial conservation residual={momentum_residual:.6g}")
    shell_residuals = []
    for index, (vector, mass) in enumerate(zip(momenta, masses), start=1):
        actual = _minkowski_square(vector)
        expected = mass * mass
        residual = abs(actual - expected) / max(abs(actual), abs(expected), parent_scale, 1.0e-300)
        shell_residuals.append(residual)
        if not math.isfinite(residual) or residual > 10.0 * tolerance:
            errors.append(f"{label} p{index} on-shell residual={residual:.6g}")
    actual_invariants = _invariants_from_momenta(momenta, labels)
    invariant_residuals: dict[str, float] = {}
    for name in labels:
        expected = float(expected_invariants[name])
        actual = actual_invariants[name]
        residual = abs(actual - expected) / max(abs(actual), abs(expected), parent_scale, 1.0e-300)
        invariant_residuals[name] = residual
        if not math.isfinite(residual) or residual > 10.0 * tolerance:
            errors.append(f"{label} {name} residual={residual:.6g}")
    source_pair_residuals: dict[str, float] = {}
    for name, particle_indices in (("s12", (0, 1)), ("s34", (2, 3))):
        actual = _minkowski_square(_sum_vectors(tuple(momenta[index] for index in particle_indices)))
        expected = float(expected_source_pair[name])
        residual = abs(actual - expected) / max(abs(actual), abs(expected), parent_scale, 1.0e-300)
        source_pair_residuals[name] = residual
        if not math.isfinite(residual) or residual > 10.0 * tolerance:
            errors.append(f"{label} source {name} residual={residual:.6g}")
    try:
        _, sigma, orientation = _recover_euler_and_sigma(momenta)
    except ValueError as exc:
        sigma = 0
        orientation = {"error": str(exc)}
        errors.append(f"{label} sigma cannot be determined: {exc}")
    if sigma != expected_sigma:
        errors.append(f"{label} sigma={sigma}, expected {expected_sigma}")
    quality = {
        "energy_conservation_relative_residual": energy_residual,
        "spatial_conservation_relative_residual": momentum_residual,
        "on_shell_relative_residual_by_particle": tuple(shell_residuals),
        "target_invariant_relative_residuals": invariant_residuals,
        "target_invariants": actual_invariants,
        "source_pair_relative_residuals": source_pair_residuals,
        "sigma": sigma,
        "orientation": orientation,
    }
    return quality, tuple(errors)


def _reflect_mirror(momenta: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float, float], ...]:
    first = tuple(momenta[0][axis] for axis in range(1, 4))
    second = tuple(momenta[1][axis] for axis in range(1, 4))
    normal = _cross3(first, second)
    norm = _norm3(normal)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot reflect a degenerate p1,p2 plane")
    normal = _scale3(normal, 1.0 / norm)
    reflected = []
    for vector in momenta:
        spatial = tuple(
            vector[axis + 1] - 2.0 * _dot3(tuple(vector[1:]), normal) * normal[axis]
            for axis in range(3)
        )
        reflected.append((vector[0], *spatial))
    return tuple(reflected)


def _max_momentum_residual(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]],
) -> tuple[float, float]:
    maximum_absolute = 0.0
    scale = 0.0
    for a, b in zip(first, second):
        for x, y in zip(a, b):
            maximum_absolute = max(maximum_absolute, abs(x - y))
            scale = max(scale, abs(x), abs(y))
    return maximum_absolute, maximum_absolute / max(scale, 1.0e-300)


def _parsed_point(point: Any) -> dict[str, float]:
    if not isinstance(point, Mapping):
        raise ValueError("reference point must be a mapping")
    try:
        result = {name: float(point[name]) for name in _REFERENCE_VARIABLES}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reference point does not contain all eight sequential coordinates") from exc
    _dimensionless_source(result, 1.0)
    return result


def compare_d5_paths(
    direct: Any,
    bridge: Any,
    *,
    masses: Sequence[float],
    total_mass: float,
    points: Iterable[Mapping[str, float]] | None = None,
    relative_tolerance: float = 1.0e-9,
    fd_relative_tolerance: float = 5.0e-6,
    kinematic_tolerance: float = 1.0e-9,
) -> BenchmarkReport:
    """Compare D5 and sequential paths through a momentum-derived map.

    All five direct invariant coordinates are extracted from the reference
    four-vectors.  An 8x8 central-difference Jacobian is evaluated at two fixed
    dimensionless step sizes and each direct mirror is checked independently.
    """

    try:
        mass_values = tuple(float(value) for value in masses)
        parent_mass = float(total_mass)
        requested_density_tolerance = float(relative_tolerance)
        fd_tolerance = float(fd_relative_tolerance)
        kin_tolerance = float(kinematic_tolerance)
    except (TypeError, ValueError, OverflowError):
        mass_values, parent_mass = (), math.nan
        requested_density_tolerance = fd_tolerance = kin_tolerance = math.nan
    effective_density_tolerance = max(requested_density_tolerance, fd_tolerance) if (
        _finite(requested_density_tolerance) and _finite(fd_tolerance)
    ) else math.nan

    try:
        if points is None:
            from .benchmark import _default_points

            generated = _default_points(mass_values, parent_mass)
            sample: tuple[Any, ...] = () if generated is None else generated
        else:
            sample = tuple(points)
    except TypeError:
        sample = ()
        input_failure = "benchmark points are not iterable"
    else:
        input_failure = ""

    def report(status: str, message: str, *, rows: Sequence[Mapping[str, Any]] = ()) -> BenchmarkReport:
        failure_rows = tuple(dict(row) for row in rows)
        if message and not failure_rows:
            failure_rows = ({"error": message},)
        evidence = {
            "replay_points": tuple(
                {str(key): _display(value) for key, value in item.items()} if isinstance(item, Mapping) else repr(item)
                for item in sample
            ),
            "masses": tuple(_display(value) for value in mass_values),
            "total_mass": _display(parent_mass),
            "requested_relative_tolerance": _display(requested_density_tolerance),
            "requested_fd_relative_tolerance": _display(fd_tolerance),
            "effective_density_tolerance": _display(effective_density_tolerance),
            "kinematic_tolerance": _display(kin_tolerance),
            "effective_kinematic_residual_tolerance": _display(10.0 * kin_tolerance),
            "finite_difference_steps": _FD_STEPS,
            "scope": "local_numeric_consistency",
            "global_claim": "not_a_global_formula_proof",
            "source_profile": getattr(bridge, "profile", ""),
            "target_profile": getattr(direct, "profile", ""),
            "source_variables": tuple(getattr(bridge, "variables", ())),
            "target_variables": tuple(getattr(direct, "target_variables", ())),
            "normalization": {
                "direct": getattr(direct, "normalization", ""),
                "reference": getattr(bridge, "normalization", ""),
            },
            "coordinate_map": "reference momenta -> five subset squares and Euler ZYZ angles",
            "point_evidence": tuple(rows),
            "failed": len(failure_rows),
        }
        return _make_report(
            status,
            getattr(bridge, "profile", ""),
            sample,
            branch_status="both_mirrors_checked" if status == "verified_local" else "incomplete",
            failures=failure_rows,
            diagnostics=(message,) if message else (),
            evidence=evidence,
        )

    if input_failure:
        return report("unsupported", input_failure)
    if not sample:
        return report("unsupported", "at least eight replay points are required")
    if len(mass_values) != 4 or any(not math.isfinite(value) or value <= 0.0 for value in mass_values):
        return report("unsupported", "D5 benchmark requires four finite, strictly positive masses")
    if not math.isfinite(parent_mass) or parent_mass <= sum(mass_values):
        return report("unsupported", "total_mass must be finite and strictly above the four-body threshold")
    if any(not math.isfinite(value) or value <= 0.0 for value in (
        requested_density_tolerance, fd_tolerance, kin_tolerance,
    )):
        return report("unsupported", "all benchmark tolerances must be finite and positive")
    if len(sample) < 8:
        return report("unsupported", "at least eight replay points are required for local evidence")
    if getattr(direct, "status", None) != "built" or getattr(bridge, "status", None) != "built":
        return report("blocked", "one numeric path is not built")
    if getattr(direct, "profile", None) != "direct_gram_d5" or getattr(bridge, "profile", None) != "sequential_12_34":
        return report("unsupported", "supported profiles are direct_gram_d5 and sequential_12_34")
    labels = tuple(getattr(direct, "target_variables", ()))
    if len(labels) != 5 or len(set(labels)) != 5 or tuple(getattr(direct, "angle_variables", ())) != ("theta", "phi", "psi"):
        return report("unsupported", "D5 target chart must declare five unique invariants and theta/phi/psi")
    if tuple(getattr(bridge, "variables", ())) != _REFERENCE_VARIABLES:
        return report("unsupported", "reference chart variables do not match the sequential 12/34 coordinates")
    normalization = getattr(direct, "normalization", None)
    if normalization not in {"jing_psf", "standard_2pi4"}:
        return report("unsupported", "normalization must be jing_psf or standard_2pi4")
    if normalization != getattr(bridge, "normalization", None):
        return report("failed", "normalization conventions do not match")
    if not callable(getattr(direct, "numeric_density", None)) or not callable(getattr(bridge, "numeric_density", None)):
        return report("blocked", "one numeric density adapter is missing")
    if not callable(getattr(direct, "numeric_momenta", None)) or not callable(getattr(bridge, "numeric_momenta", None)):
        return report("blocked", "one numeric momentum adapter is missing")
    branches = tuple(getattr(direct, "branches", ()))
    branch_keys = {getattr(item, "key", "") for item in branches}
    if len(branches) != 2 or branch_keys != {"sigma=+1", "sigma=-1"} or any(
        getattr(item, "multiplicity", None) != 1
        or getattr(item, "coefficient", None) != "one mirror branch"
        or not getattr(item, "equations", ())
        or not getattr(item, "root_condition", "")
        or not getattr(item, "domain", ())
        for item in branches
    ):
        return report("unsupported", "D5 branch metadata must expose both unit-multiplicity mirror roots")
    summed_flags = {bool(getattr(item, "summed", False)) for item in branches}
    if len(summed_flags) != 1:
        return report("unsupported", "D5 branch metadata has inconsistent density mode")
    density_divisor = 2.0 if True in summed_flags else 1.0
    try:
        for label in labels:
            _subset_from_label(label)
    except ValueError as exc:
        return report("unsupported", str(exc))

    normalized_sample: list[dict[str, float]] = []
    seen_points: set[tuple[float, ...]] = set()
    for index, raw_point in enumerate(sample):
        try:
            point = _parsed_point(raw_point)
            x = _dimensionless_source(point, parent_mass)
        except (TypeError, ValueError, OverflowError, KeyError) as exc:
            return report("failed", "invalid reference replay point", rows=({"point": index, "error": str(exc)},))
        key = tuple(float(point[name]) for name in _REFERENCE_VARIABLES)
        if key in seen_points:
            return report("unsupported", "replay sample must contain at least eight unique points")
        seen_points.add(key)
        normalized_sample.append(point)
    if len(seen_points) < 8:
        return report("unsupported", "replay sample has fewer than eight unique points")

    point_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    density_absolute_residuals: list[float] = []
    density_relative_residuals: list[float] = []
    coordinate_residuals: list[float] = []
    compared = 0
    for point_index, reference_point in enumerate(normalized_sample):
        errors: list[str] = []
        row: dict[str, Any] = {"point": point_index, "reference_coordinates": dict(reference_point)}
        try:
            reference_momenta = _momenta(
                bridge.numeric_momenta(reference_point, mass_values, parent_mass),
                "reference momenta",
            )
            reference_target = _invariants_from_momenta(reference_momenta, labels)
            source_pair = {"s12": reference_point["s12"], "s34": reference_point["s34"]}
            direct_angles, reference_sigma, orientation = _recover_euler_and_sigma(reference_momenta)
            direct_values = {**reference_target, **direct_angles}
            x = _dimensionless_source(reference_point, parent_mass)
            jacobians = tuple(
                _finite_difference_jacobian(
                    x, step, bridge=bridge, labels=labels,
                    masses=mass_values, total_mass=parent_mass,
                )
                for step in _FD_STEPS
            )
            rho_reference = float(bridge.numeric_density(reference_point, mass_values, parent_mass))
            rho_direct_total = float(direct.numeric_density(direct_values, mass_values, parent_mass))
            if not all(math.isfinite(value) and value > 0.0 for value in (rho_reference, rho_direct_total)):
                raise ValueError("both path densities must be finite and strictly positive")
            rho_direct_branch = rho_direct_total / density_divisor
            source_sine = (
                math.sin(reference_point["theta0"])
                * math.sin(reference_point["theta12"])
                * math.sin(reference_point["theta34"])
            )
            direct_sine = math.sin(direct_angles["theta"])
            rhs = rho_reference * source_sine
            if not math.isfinite(rhs) or rhs <= 0.0:
                raise ValueError("reference sine-weighted density is not finite and positive")
            step_density_rows = []
            for jacobian in jacobians:
                lhs = rho_direct_branch * direct_sine * jacobian["real_dimension_jacobian_absolute"]
                if not math.isfinite(lhs) or lhs <= 0.0:
                    raise ValueError("direct sine-weighted density/Jacobian is not finite and positive")
                residual = _relative_residual(lhs, rhs)
                density_absolute_residuals.append(abs(lhs - rhs))
                density_relative_residuals.append(residual)
                step_density_rows.append({
                    "step": jacobian["step"],
                    "lhs": lhs,
                    "rhs": rhs,
                    "absolute_residual": abs(lhs - rhs),
                    "relative_residual": residual,
                    "passes_effective_tolerance": residual <= effective_density_tolerance,
                })
                if residual > effective_density_tolerance:
                    errors.append(
                        f"density residual at h={jacobian['step']:.1e} is {residual:.6g} "
                        f"> {effective_density_tolerance:.6g}"
                    )
            jacobian_step_difference = _relative_residual(
                jacobians[0]["real_dimension_jacobian_absolute"],
                jacobians[1]["real_dimension_jacobian_absolute"],
            )
            row.update({
                "reference_sigma": reference_sigma,
                "recovered_d5_coordinates": direct_values,
                "target_invariants_from_reference_momenta": reference_target,
                "reference_source_pair_inputs": source_pair,
                "reference_orientation": orientation,
                "density_reference": rho_reference,
                "density_direct_total": rho_direct_total,
                "density_direct_per_branch": rho_direct_branch,
                "sine_weighted_density_comparison": tuple(step_density_rows),
                "jacobians": jacobians,
                "relative_difference_between_fd_steps": jacobian_step_difference,
            })

            # Each mirror is checked from its own reconstructed four-vectors.
            reflected_reference = _reflect_mirror(reference_momenta)
            branch_rows = []
            source_quality_cache: dict[int, dict[str, Any]] = {}
            for branch in (1, -1):
                reconstructed = _momenta(
                    direct.numeric_momenta(direct_values, mass_values, parent_mass, branch=branch),
                    f"direct sigma={branch:+d} momenta",
                )
                expected_mirror = reference_momenta if branch == reference_sigma else reflected_reference
                mirror_euler, mirror_sigma, mirror_orientation = _recover_euler_and_sigma(expected_mirror)
                mirror_angle_residuals = {
                    "theta": abs(mirror_euler["theta"] - direct_angles["theta"]),
                    "phi": abs(_wrap_delta(mirror_euler["phi"] - direct_angles["phi"])),
                    "psi": abs(_wrap_delta(mirror_euler["psi"] - direct_angles["psi"])),
                }
                mirror_angle_residual = max(mirror_angle_residuals.values())
                if mirror_sigma != branch:
                    errors.append(f"reference mirror sigma={mirror_sigma}, expected {branch}")
                if mirror_angle_residual > 10.0 * kin_tolerance:
                    errors.append(
                        f"reference mirror Euler-angle residual={mirror_angle_residual:.6g}"
                    )
                target_quality_direct, target_errors_direct = _validate_momenta(
                    reconstructed,
                    masses=mass_values,
                    total_mass=parent_mass,
                    labels=labels,
                    expected_invariants=reference_target,
                    expected_source_pair=source_pair,
                    expected_sigma=branch,
                    tolerance=kin_tolerance,
                    label=f"direct sigma={branch:+d}",
                )
                expected_reference_sigma = branch
                target_quality_reference, target_errors_reference = _validate_momenta(
                    expected_mirror,
                    masses=mass_values,
                    total_mass=parent_mass,
                    labels=labels,
                    expected_invariants=reference_target,
                    expected_source_pair=source_pair,
                    expected_sigma=expected_reference_sigma,
                    tolerance=kin_tolerance,
                    label=f"reference mirror sigma={branch:+d}",
                )
                coordinate_absolute, coordinate_scaled = _max_momentum_residual(reconstructed, expected_mirror)
                coordinate_residuals.append(coordinate_scaled)
                if coordinate_scaled > 10.0 * kin_tolerance:
                    errors.append(
                        f"direct sigma={branch:+d} momentum mismatch={coordinate_scaled:.6g} "
                        f"(absolute={coordinate_absolute:.6g})"
                    )
                errors.extend(target_errors_direct)
                errors.extend(target_errors_reference)
                branch_rows.append({
                    "branch_sigma": branch,
                    "matching_reference_sigma": branch == reference_sigma,
                    "reference_mirror_euler": mirror_euler,
                    "reference_mirror_orientation": mirror_orientation,
                    "reference_mirror_euler_angle_residuals": mirror_angle_residuals,
                    "reference_mirror_euler_max_residual": mirror_angle_residual,
                    "direct_quality": target_quality_direct,
                    "reference_mirror_quality": target_quality_reference,
                    "momentum_absolute_residual": coordinate_absolute,
                    "momentum_relative_residual": coordinate_scaled,
                    "reference_momenta_after_p1p2_plane_reflection": expected_mirror,
                    "direct_momenta": reconstructed,
                })
            row["mirror_branches"] = tuple(branch_rows)
            compared += 1
        except (ArithmeticError, TypeError, ValueError, OverflowError, KeyError) as exc:
            errors.append(str(exc))
        except Exception as exc:  # Numeric adapters are external inputs to this check.
            errors.append(f"numeric adapter raised {type(exc).__name__}: {exc}")
        row["errors"] = tuple(errors)
        point_rows.append(row)
        if errors:
            failures.append({"point": point_index, "coordinates": reference_point, "error": "; ".join(errors)})

    status = "verified_local" if compared == len(sample) and not failures else "failed"
    max_absolute = max(density_absolute_residuals, default=None)
    max_relative = max(density_relative_residuals, default=None)
    max_coordinate = max(coordinate_residuals, default=None)
    return _make_report(
        status,
        getattr(bridge, "profile", ""),
        sample,
        compared=compared,
        max_absolute=max_absolute,
        max_relative=max_relative,
        max_coordinate=max_coordinate,
        branch_status="both_mirrors_checked" if compared == len(sample) and not failures else "failed_or_incomplete",
        failures=failures,
        diagnostics=(
            "D5 finite-point benchmark only: both sigma mirrors, independent physical constraints, momentum-derived coordinates, and two central-difference steps were checked",
            "verified_local records this finite sample and does not prove the full formula or match external source material",
        ),
        evidence={
            "replay_points": tuple(normalized_sample),
            "masses": mass_values,
            "total_mass": parent_mass,
            "requested_relative_tolerance": requested_density_tolerance,
            "requested_fd_relative_tolerance": fd_tolerance,
            "effective_density_tolerance": effective_density_tolerance,
            "kinematic_tolerance": kin_tolerance,
            "effective_kinematic_residual_tolerance": 10.0 * kin_tolerance,
            "finite_difference_steps": _FD_STEPS,
            "source_profile": getattr(bridge, "profile", ""),
            "target_profile": getattr(direct, "profile", ""),
            "source_variables": tuple(getattr(bridge, "variables", ())),
            "target_variables": labels,
            "normalization": {
                "direct": getattr(direct, "normalization", ""),
                "reference": getattr(bridge, "normalization", ""),
            },
            "differential_measure": {
                "direct": getattr(direct, "differential_measure", ""),
                "reference": getattr(bridge, "differential_measure", ""),
            },
            "coordinate_map": "reference p1..p4 -> subset-square target invariants and ZYZ Euler angles; no direct coordinate map used to generate targets",
            "finite_difference_protocol": {
                "source_coordinates": "(s12/M^2,s34/M^2,theta0,phi0,theta12,phi12,theta34,phi34)",
                "target_coordinates": "(five momentum-derived target invariants/M^2,theta,phi,psi)",
                "angle_unwrap": "shortest signed difference for output phi and psi",
                "real_jacobian_dimension_factor": "dimensionless determinant * M^6",
            },
            "density_comparison": "rho_direct_per_branch*sin(theta)*abs(J_real) versus rho_reference*sin(theta0)*sin(theta12)*sin(theta34)",
            "kinematic_acceptance": "each residual is required to be <= effective_kinematic_residual_tolerance=10*kinematic_tolerance",
            "density_divisor_for_summed_mode": density_divisor,
            "point_evidence": tuple(point_rows),
            "failed": len(failures),
            "scope": "local_numeric_consistency",
            "global_claim": "not_a_global_formula_proof",
        },
    )


__all__ = ["compare_d5_paths"]
