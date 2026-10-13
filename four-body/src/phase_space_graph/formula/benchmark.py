"""Independent local comparison of direct and recursive formula paths.

The benchmark is deliberately a local, finite-point check.  It does not turn
finite numerical agreement into a proof of the full formula.  In particular,
both momentum adapters are checked independently before their outputs are
compared: two adapters returning the same physically invalid vectors must
still fail.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .records import SCHEMA_VERSION, BenchmarkReport, BridgeFormula, DirectFormula, FormulaOptions


_TARGET_VARIABLES = ("s12", "s34")
_DIRECT_ANGLES = ("theta0", "phi0", "thetaA", "phiA", "thetaB", "phiB")
_REFERENCE_VARIABLES = (
    "s12", "s34", "theta0", "phi0", "theta12", "phi12", "theta34", "phi34",
)
_PI2 = 2.0 * math.pi


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _display_value(value: Any) -> Any:
    """Return a JSON-safe value while retaining NaN/inf as visible strings."""

    if isinstance(value, bool):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return repr(value)
    if not math.isfinite(numeric):
        return repr(value)
    return numeric


def _display_point(point: Any) -> Any:
    if isinstance(point, Mapping):
        return {str(key): _display_value(value) for key, value in point.items()}
    return repr(point)


def _default_points(masses: Sequence[float], total_mass: float) -> tuple[dict[str, float], ...] | None:
    """Make at least eight deterministic non-degenerate interior points."""

    try:
        values = tuple(float(mass) for mass in masses)
        parent_mass = float(total_mass)
    except (TypeError, ValueError, OverflowError):
        return None
    # Zero masses are allowed by the current chart (the pair invariant is kept
    # strictly positive); negative, non-finite masses are not physical input.
    if len(values) != 4 or any(not math.isfinite(mass) or mass < 0.0 for mass in values):
        return None
    if not math.isfinite(parent_mass) or parent_mass <= 0.0:
        return None
    lower_a = values[0] + values[1]
    lower_b = values[2] + values[3]
    slack = parent_mass - lower_a - lower_b
    if not math.isfinite(slack) or slack <= 0.0:
        return None

    # Keep both pair masses away from threshold and the parent boundary.  The
    # table is fixed so a report can be reproduced without a random seed.
    fractions = (
        (0.12, 0.10), (0.12, 0.28), (0.18, 0.08), (0.18, 0.24),
        (0.25, 0.06), (0.25, 0.21), (0.32, 0.05), (0.32, 0.18),
    )
    angles = (
        (0.61, 0.23, 0.82, 1.17, 1.46, 3.21),
        (0.74, 1.31, 1.05, 2.44, 1.83, 4.16),
        (0.93, 2.07, 1.28, 0.58, 2.16, 5.03),
        (1.16, 2.91, 0.67, 3.12, 2.42, 0.91),
        (1.39, 3.47, 1.51, 4.02, 0.88, 2.73),
        (1.67, 4.28, 1.92, 5.11, 1.14, 4.87),
        (1.94, 5.19, 2.21, 1.06, 1.72, 0.37),
        (2.23, 0.87, 2.47, 2.18, 2.03, 5.61),
    )
    return tuple(
        {
            "s12": (lower_a + first * slack) ** 2,
            "s34": (lower_b + second * slack) ** 2,
            "theta0": values_[0],
            "phi0": values_[1],
            "theta12": values_[2],
            "phi12": values_[3],
            "theta34": values_[4],
            "phi34": values_[5],
        }
        for (first, second), values_ in zip(fractions, angles)
    )


def _as_numeric_vector(value: Any, *, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} is not a numeric vector")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not a numeric vector") from exc
    if len(vector) != 4:
        raise ValueError(f"{label} has length {len(vector)}; expected 4")
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{label} contains a non-finite component")
    if vector[0] <= 0.0:
        raise ValueError(f"{label} has non-positive energy")
    return vector


def _normalise_momenta(value: Any, *, label: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} is not a momentum collection")
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{label} is not a momentum collection") from exc
    if len(rows) != 4:
        raise ValueError(f"{label} has {len(rows)} vectors; expected 4")
    return tuple(_as_numeric_vector(row, label=f"{label}[{index}]") for index, row in enumerate(rows))


def _scaled_residual(value: float, scale: float) -> float:
    denominator = abs(scale)
    if denominator == 0.0:
        denominator = math.nextafter(0.0, 1.0)
    return abs(value) / denominator


def _minkowski_sum(first: Sequence[float], second: Sequence[float]) -> tuple[float, float, float, float]:
    result = tuple(first[index] + second[index] for index in range(4))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("momentum sum is non-finite")
    return result  # type: ignore[return-value]


def _mass_squared(vector: Sequence[float]) -> float:
    return vector[0] * vector[0] - sum(component * component for component in vector[1:])


def _validate_momenta(
    momenta: tuple[tuple[float, ...], ...],
    *,
    masses: Sequence[float],
    total_mass: float,
    point: Mapping[str, Any],
    tolerance: float,
    label: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Check one adapter against physical constraints and requested invariants."""

    issues: list[str] = []
    parent_mass_squared = total_mass * total_mass
    if not _finite(parent_mass_squared):
        return (
            {
                "energy_conservation": None,
                "spatial_conservation": None,
                "on_shell": None,
                "pair_invariants": None,
            },
            (f"{label} parent mass scale is non-finite",),
        )
    total_energy = sum(vector[0] for vector in momenta)
    spatial_sums = tuple(sum(vector[axis] for vector in momenta) for axis in range(1, 4))
    energy_difference = total_energy - total_mass if _finite(total_energy) else math.inf
    if _finite(energy_difference):
        energy_residual = _scaled_residual(energy_difference, total_mass)
    else:
        energy_residual = None
        issues.append(f"{label} energy conservation is non-finite")
    if all(_finite(value) for value in spatial_sums):
        momentum_residual = max((_scaled_residual(value, total_mass) for value in spatial_sums), default=0.0)
    else:
        momentum_residual = None
        issues.append(f"{label} spatial conservation is non-finite")
    if energy_residual is not None and energy_residual > 10.0 * tolerance:
        issues.append(f"{label} energy conservation residual={energy_residual:.6g}")
    if momentum_residual is not None and momentum_residual > 10.0 * tolerance:
        issues.append(f"{label} spatial conservation residual={momentum_residual:.6g}")

    on_shell_residual = 0.0
    for index, (vector, mass) in enumerate(zip(momenta, masses)):
        actual_mass_squared = _mass_squared(vector)
        expected_mass_squared = mass * mass
        if not _finite(actual_mass_squared) or not _finite(expected_mass_squared):
            issues.append(f"{label} p{index + 1} has a non-finite mass square")
            continue
        shell = actual_mass_squared - expected_mass_squared
        if not _finite(shell):
            issues.append(f"{label} p{index + 1} on-shell difference is non-finite")
            continue
        scale = max(abs(actual_mass_squared), abs(expected_mass_squared), parent_mass_squared)
        residual = _scaled_residual(shell, scale)
        on_shell_residual = max(on_shell_residual, residual)
        if residual > 10.0 * tolerance:
            issues.append(f"{label} p{index + 1} on-shell residual={residual:.6g}")

    pair12 = _minkowski_sum(momenta[0], momenta[1])
    pair34 = _minkowski_sum(momenta[2], momenta[3])
    calculated = (_mass_squared(pair12), _mass_squared(pair34))
    invariant_residual = 0.0
    for name, actual, expected in zip(("s12", "s34"), calculated, (point["s12"], point["s34"])):
        expected_float = float(expected)
        if not _finite(actual) or not _finite(expected_float):
            issues.append(f"{label} {name} is non-finite")
            continue
        difference = actual - expected_float
        if not _finite(difference):
            issues.append(f"{label} {name} difference is non-finite")
            continue
        scale = max(abs(actual), abs(expected_float), parent_mass_squared)
        residual = _scaled_residual(difference, scale)
        invariant_residual = max(invariant_residual, residual)
        if residual > 10.0 * tolerance:
            issues.append(f"{label} {name} residual={residual:.6g}")

    quality: dict[str, Any] = {
        "energy_conservation": energy_residual,
        "spatial_conservation": momentum_residual,
        "on_shell": on_shell_residual,
        "pair_invariants": invariant_residual,
    }
    return quality, tuple(issues)


def _point_values(point: Any) -> tuple[dict[str, float], dict[str, float]] | None:
    if not isinstance(point, Mapping):
        return None
    try:
        reference = {name: float(point[name]) for name in _REFERENCE_VARIABLES}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in reference.values()):
        return None
    if reference["s12"] <= 0.0 or reference["s34"] <= 0.0:
        return None
    if not 0.0 <= reference["theta0"] <= math.pi or not 0.0 <= reference["theta12"] <= math.pi or not 0.0 <= reference["theta34"] <= math.pi:
        return None
    for name in ("phi0", "phi12", "phi34"):
        if not 0.0 <= reference[name] < _PI2:
            return None
    target = {
        "s12": reference["s12"],
        "s34": reference["s34"],
        "theta0": reference["theta0"],
        "phi0": reference["phi0"],
        "thetaA": reference["theta12"],
        "phiA": reference["phi12"],
        "thetaB": reference["theta34"],
        "phiB": reference["phi34"],
    }
    return reference, target


def _max_vector_diff(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> tuple[float, float]:
    if len(a) != len(b):
        raise ValueError(f"momentum collections have lengths {len(a)} and {len(b)}")
    raw = 0.0
    scale = 0.0
    for index, (pa, pb) in enumerate(zip(a, b)):
        if len(pa) != len(pb):
            raise ValueError(f"momentum vector {index} has incompatible lengths")
        for x, y in zip(pa, pb):
            difference = abs(x - y)
            if not math.isfinite(difference):
                raise ValueError("momentum difference is non-finite")
            raw = max(raw, difference)
            scale = max(scale, abs(x), abs(y))
    scaled = _scaled_residual(raw, scale)
    if not _finite(scaled):
        raise ValueError("scaled momentum difference is non-finite")
    return raw, scaled


def _evidence(
    sample: Sequence[Any],
    *,
    masses: Sequence[float],
    total_mass: float,
    tolerance: float,
    direct: DirectFormula,
    bridge: BridgeFormula,
    compared: int,
    failures: Sequence[Mapping[str, Any]],
    max_absolute: float | None,
    max_relative: float | None,
    coordinate_residual: float | None,
) -> dict[str, Any]:
    return {
        "replay_points": tuple(_display_point(point) for point in sample),
        "masses": tuple(_display_value(mass) for mass in masses),
        "total_mass": _display_value(total_mass),
        "relative_tolerance": _display_value(tolerance),
        "scales": {
            "density": "max(abs(density_direct), abs(density_reference))",
            "momentum": "max(abs(all momentum components))",
            "conservation": "total_mass",
            "mass_squared_and_invariants": "max(total_mass^2, abs(actual), abs(expected))",
        },
        "source_profile": bridge.profile,
        "target_profile": direct.profile,
        "source_variables": tuple(bridge.variables),
        "target_variables": tuple(direct.target_variables),
        "normalization": {
            "direct": getattr(direct, "normalization", ""),
            "reference": getattr(bridge, "normalization", ""),
        },
        "differential_measure": {
            "direct": getattr(direct, "differential_measure", ""),
            "reference": getattr(bridge, "differential_measure", ""),
        },
        "angle_definitions": {
            "direct": tuple(getattr(direct, "angle_definitions", ())),
            "reference": tuple(getattr(bridge, "angle_definitions", ())),
        },
        "schema_version": SCHEMA_VERSION,
        "rule_versions": tuple(
            getattr(direct, "rule_versions", ())
            or getattr(bridge, "rule_versions", ())
            or FormulaOptions().rule_versions
        ),
        "coordinate_map": "s12/s34 identity; theta12/phi12 -> thetaA/phiA; theta34/phi34 -> thetaB/phiB",
        "branch_mode": "single_branch",
        "scope": "local_numeric_consistency",
        "global_claim": "not_a_global_formula_proof",
        "compared": compared,
        "failed": len(failures),
        "max_absolute_residual": max_absolute,
        "max_relative_residual": max_relative,
        "coordinate_residual": coordinate_residual,
    }


def _make_report(
    status: str,
    profile: str,
    points: int,
    compared: int,
    max_absolute: float | None,
    max_relative: float | None,
    coordinate_residual: float | None,
    branch_status: str,
    failures: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[str],
    evidence: Mapping[str, Any],
) -> BenchmarkReport:
    failure_rows = tuple(dict(row) for row in failures)
    diagnostic_rows = tuple(str(row) for row in diagnostics)
    return BenchmarkReport(
        status,
        profile,
        points,
        compared,
        max_absolute,
        max_relative,
        coordinate_residual,
        branch_status,
        failure_rows,
        diagnostic_rows,
        dict(evidence),
    )


def _unsupported_report(
    status: str,
    bridge: BridgeFormula,
    sample: Sequence[Any],
    *,
    masses: Sequence[float],
    total_mass: float,
    tolerance: float,
    direct: DirectFormula,
    failures: Sequence[Mapping[str, Any]] = (),
    diagnostics: Sequence[str] = (),
    branch_status: str = "not_checked",
) -> BenchmarkReport:
    evidence = _evidence(
        sample, masses=masses, total_mass=total_mass, tolerance=tolerance,
        direct=direct, bridge=bridge, compared=0, failures=failures,
        max_absolute=None, max_relative=None, coordinate_residual=None,
    )
    return _make_report(
        status, bridge.profile, len(sample), 0, None, None, None,
        branch_status, failures, diagnostics, evidence,
    )


def compare_paths(
    direct: DirectFormula,
    bridge: BridgeFormula,
    *,
    masses: Sequence[float],
    total_mass: float,
    points: Iterable[Mapping[str, float]] | None = None,
    relative_tolerance: float = 1e-9,
    fd_relative_tolerance: float = 5e-6,
    kinematic_tolerance: float = 1e-9,
) -> BenchmarkReport:
    """Compare matching physical points, never raw densities in unlike charts."""

    if getattr(direct, "profile", None) == "direct_gram_d5":
        # Lazy import keeps the dedicated benchmark isolated from this legacy
        # pair-chart implementation and avoids a module import cycle.
        from .d5_benchmark import compare_d5_paths

        return compare_d5_paths(
            direct,
            bridge,
            masses=masses,
            total_mass=total_mass,
            points=points,
            relative_tolerance=relative_tolerance,
            fd_relative_tolerance=fd_relative_tolerance,
            kinematic_tolerance=kinematic_tolerance,
        )

    if getattr(direct, "profile", None) == "direct_gram_low_loop":
        # Build the exact chart from the direct formula's declared target
        # variable order. Both imports stay lazy so legacy pair/D5 callers do
        # not load the low-loop numerical implementation.
        from .low_loop_benchmark import compare_low_loop_paths
        from .low_loop_direct import build_low_loop_chart_data_from_labels

        labels = tuple(getattr(direct, "target_variables", ()))
        chart_data = build_low_loop_chart_data_from_labels(labels)
        return compare_low_loop_paths(
            direct,
            bridge,
            chart_data=chart_data,
            masses=masses,
            total_mass=total_mass,
            points=points,
            relative_tolerance=relative_tolerance,
            fd_relative_tolerance=fd_relative_tolerance,
            kinematic_tolerance=kinematic_tolerance,
        )

    try:
        tolerance = float(relative_tolerance)
    except (TypeError, ValueError, OverflowError):
        tolerance = math.nan
    try:
        mass_values = tuple(float(mass) for mass in masses)
        parent_mass = float(total_mass)
    except (TypeError, ValueError, OverflowError):
        mass_values = ()
        parent_mass = math.nan

    if points is None:
        generated = _default_points(mass_values, parent_mass)
        sample: tuple[Any, ...] = () if generated is None else generated
        if generated is None:
            return _unsupported_report(
                "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
                tolerance=tolerance, direct=direct,
                diagnostics=("no non-degenerate four-body phase-space domain for the supplied masses and total_mass",),
            )
    else:
        try:
            sample = tuple(points)
        except TypeError:
            sample = ()
            return _unsupported_report(
                "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
                tolerance=tolerance, direct=direct,
                diagnostics=("benchmark points are not iterable",),
            )

    if not sample:
        return _unsupported_report(
            "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("empty benchmark sample is not evidence and cannot be verified",),
        )
    if len(mass_values) != 4 or any(not math.isfinite(mass) or mass < 0.0 for mass in mass_values):
        return _unsupported_report(
            "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("masses must be four finite non-negative values",),
        )
    if not math.isfinite(parent_mass) or parent_mass <= sum(mass_values):
        return _unsupported_report(
            "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("total_mass must be finite and above the four-body threshold",),
        )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        return _unsupported_report(
            "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("relative_tolerance must be finite and positive",),
        )

    if direct.status != "built" or bridge.status != "built":
        return _unsupported_report(
            "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("one path is unavailable",),
        )
    if direct.profile != "direct_pair_recursive" or bridge.profile != "sequential_12_34":
        return _unsupported_report(
            "pending", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("only the direct_pair_recursive versus sequential_12_34 profile is supported",),
        )
    if tuple(direct.target_variables) != _TARGET_VARIABLES:
        return _unsupported_report(
            "pending", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("no supported direct chart beyond target variables s12,s34",),
        )
    if tuple(direct.angle_variables) != _DIRECT_ANGLES or tuple(bridge.variables) != _REFERENCE_VARIABLES:
        return _unsupported_report(
            "pending", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("coordinate angle variables do not match the supported 12/34 to A/B map",),
        )

    branches = getattr(direct, "branches", ())
    if not isinstance(branches, (tuple, list)) or len(branches) != 1:
        return _unsupported_report(
            "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct, branch_status="unsupported",
            diagnostics=("current benchmark supports exactly one direct branch",),
        )
    branch = branches[0]
    if (
        getattr(branch, "multiplicity", None) != 1
        or bool(getattr(branch, "summed", True))
        or getattr(branch, "coefficient", None) != "1"
        or not getattr(branch, "key", "")
        or not getattr(branch, "equations", ())
        or not getattr(branch, "root_condition", "")
        or not getattr(branch, "domain", ())
    ):
        return _unsupported_report(
            "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct, branch_status="unsupported",
            diagnostics=("direct root/branch metadata is unsupported for the single-branch benchmark",),
        )

    direct_normalization = getattr(direct, "normalization", "")
    bridge_normalization = getattr(bridge, "normalization", "")
    if not isinstance(direct_normalization, str) or not isinstance(bridge_normalization, str) or not direct_normalization or not bridge_normalization:
        return _unsupported_report(
            "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("both paths must declare a non-empty normalization before comparison",),
        )
    if direct_normalization != bridge_normalization:
        return _unsupported_report(
            "failed", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=(f"normalization mismatch: direct={direct_normalization!r}, reference={bridge_normalization!r}",),
        )
    direct_measure = getattr(direct, "differential_measure", "")
    bridge_measure = getattr(bridge, "differential_measure", "")
    if not direct_measure or not bridge_measure:
        return _unsupported_report(
            "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("both paths must declare their differential measure before comparison",),
        )
    if direct_measure and bridge_measure and direct_measure != bridge_measure:
        return _unsupported_report(
            "failed", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=(f"differential-measure mismatch: direct={direct_measure!r}, reference={bridge_measure!r}",),
        )
    direct_angles = tuple(getattr(direct, "angle_definitions", ()))
    bridge_angles = tuple(getattr(bridge, "angle_definitions", ()))
    if not direct_angles or not bridge_angles:
        return _unsupported_report(
            "unsupported", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("both paths must declare angle definitions before comparison",),
        )

    if not callable(direct.numeric_density) or not callable(bridge.numeric_density):
        return _unsupported_report(
            "blocked", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("numeric density adapters are missing",),
        )
    if not callable(direct.numeric_momenta) or not callable(bridge.numeric_momenta):
        return _unsupported_report(
            "blocked", bridge, sample, masses=mass_values, total_mass=parent_mass,
            tolerance=tolerance, direct=direct,
            diagnostics=("numeric momentum adapters are missing",),
        )

    absolute_residuals: list[float] = []
    relative_residuals: list[float] = []
    coordinate_residuals: list[float] = []
    failures: list[dict[str, Any]] = []
    compared = 0
    quality_rows: list[dict[str, Any]] = []
    for index, original_point in enumerate(sample):
        display_point = _display_point(original_point)
        parsed = _point_values(original_point)
        if parsed is None:
            failures.append({"point": index, "coordinates": display_point, "error": "invalid or non-finite reference coordinates"})
            continue
        reference_point, direct_point = parsed
        try:
            rho_d = float(direct.numeric_density(direct_point, mass_values, parent_mass))
            rho_r = float(bridge.numeric_density(reference_point, mass_values, parent_mass))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            failures.append({"point": index, "coordinates": display_point, "error": f"density adapter: {exc}"})
            continue
        if not _finite(rho_d) or not _finite(rho_r) or rho_d <= 0.0 or rho_r <= 0.0:
            failures.append({
                "point": index, "coordinates": display_point,
                "density_direct": _display_value(rho_d), "density_reference": _display_value(rho_r),
                "error": "densities must be finite and strictly positive",
            })
            continue
        try:
            momenta_d = _normalise_momenta(direct.numeric_momenta(direct_point, mass_values, parent_mass), label="direct momenta")
            momenta_r = _normalise_momenta(bridge.numeric_momenta(reference_point, mass_values, parent_mass), label="reference momenta")
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            failures.append({"point": index, "coordinates": display_point, "error": f"momentum adapter: {exc}"})
            continue

        compared += 1
        density_absolute = abs(rho_d - rho_r)
        density_relative = density_absolute / max(abs(rho_d), abs(rho_r))
        absolute_residuals.append(density_absolute)
        relative_residuals.append(density_relative)
        try:
            raw_coordinate, scaled_coordinate = _max_vector_diff(momenta_d, momenta_r)
            quality_d, issues_d = _validate_momenta(
                momenta_d, masses=mass_values, total_mass=parent_mass,
                point=reference_point, tolerance=tolerance, label="direct",
            )
            quality_r, issues_r = _validate_momenta(
                momenta_r, masses=mass_values, total_mass=parent_mass,
                point=reference_point, tolerance=tolerance, label="reference",
            )
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            failures.append({
                "point": index, "coordinates": display_point,
                "error": f"physical validation: {exc}",
            })
            continue
        coordinate_residuals.append(scaled_coordinate)
        quality_rows.append({"point": index, "direct": quality_d, "reference": quality_r})
        errors = list(issues_d) + list(issues_r)
        if density_relative > tolerance:
            errors.append(f"density relative residual={density_relative:.6g}")
        if scaled_coordinate > 10.0 * tolerance:
            errors.append(f"coordinate residual={scaled_coordinate:.6g} (raw={raw_coordinate:.6g})")
        if errors:
            failures.append({
                "point": index, "coordinates": display_point,
                "density_direct": rho_d, "density_reference": rho_r,
                "relative_residual": density_relative,
                "coordinate_residual": scaled_coordinate,
                "quality": {"direct": quality_d, "reference": quality_r},
                "error": "; ".join(errors),
            })

    status = "verified" if compared == len(sample) and not failures else "failed"
    maximum_absolute = max(absolute_residuals, default=None)
    maximum_relative = max(relative_residuals, default=None)
    maximum_coordinate = max(coordinate_residuals, default=None)
    evidence = _evidence(
        sample, masses=mass_values, total_mass=parent_mass, tolerance=tolerance,
        direct=direct, bridge=bridge, compared=compared, failures=failures,
        max_absolute=maximum_absolute, max_relative=maximum_relative,
        coordinate_residual=maximum_coordinate,
    )
    evidence["point_quality"] = tuple(quality_rows)
    diagnostics = (
        "single-branch local point comparison; finite agreement is not a global formula proof",
    )
    return _make_report(
        status, bridge.profile, len(sample), compared, maximum_absolute,
        maximum_relative, maximum_coordinate, "single_branch", failures,
        diagnostics, evidence,
    )


def benchmark_formula(*args: Any, **kwargs: Any) -> BenchmarkReport:
    return compare_paths(*args, **kwargs)


def map_reference_to_target(point: Mapping[str, float], *, target: Sequence[str]) -> dict[str, float] | None:
    """Map supported reference coordinates to the direct chart's names."""

    if tuple(target) != _TARGET_VARIABLES or not isinstance(point, Mapping):
        return None
    try:
        theta_a = point["theta12"] if "theta12" in point else point["thetaA"]
        phi_a = point["phi12"] if "phi12" in point else point["phiA"]
        theta_b = point["theta34"] if "theta34" in point else point["thetaB"]
        phi_b = point["phi34"] if "phi34" in point else point["phiB"]
        return {
            "s12": point["s12"],
            "s34": point["s34"],
            "theta0": point["theta0"],
            "phi0": point["phi0"],
            "thetaA": theta_a,
            "phiA": phi_a,
            "thetaB": theta_b,
            "phiB": phi_b,
        }
    except (KeyError, TypeError):
        return None


__all__ = ["compare_paths", "benchmark_formula", "map_reference_to_target"]
