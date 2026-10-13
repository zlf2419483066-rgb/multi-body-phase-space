"""Independent local numerical checks for the low-loop mixed-coordinate charts.

The target invariants and the selected pair/nested angular coordinates are
recovered directly from the fixed sequential reference four-vectors.  The
production inverse and production coordinate extractor are intentionally not
used to construct benchmark coordinates.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from . import d5_benchmark as _d5
from .records import BenchmarkReport


_REFERENCE_VARIABLES = (
    "s12", "s34", "theta0", "phi0", "theta12", "phi12", "theta34", "phi34",
)
_FD_STEPS = (2.0e-5, 1.0e-5)
_TWO_PI = 2.0 * math.pi


def _boost(vector: Sequence[float], beta: Sequence[float]) -> tuple[float, float, float, float]:
    b2 = sum(float(value) ** 2 for value in beta)
    if not math.isfinite(b2) or b2 >= 1.0:
        raise ValueError("Lorentz boost velocity is non-finite or superluminal")
    if b2 == 0.0:
        return tuple(float(value) for value in vector)  # type: ignore[return-value]
    gamma = 1.0 / math.sqrt(1.0 - b2)
    dot = sum(float(vector[i + 1]) * float(beta[i]) for i in range(3))
    factor = ((gamma - 1.0) * dot / b2) + gamma * float(vector[0])
    result = (gamma * (float(vector[0]) + dot),) + tuple(
        float(vector[i + 1]) + factor * float(beta[i]) for i in range(3)
    )
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Lorentz boost produced a non-finite four-vector")
    return result  # type: ignore[return-value]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Sequence[float], label: str) -> tuple[float, float, float]:
    size = _norm(a)
    if not math.isfinite(size) or size <= 1.0e-14:
        raise ValueError(f"{label} is zero or geometrically degenerate")
    return tuple(float(value) / size for value in a)  # type: ignore[return-value]


def _sum(vectors: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    if not vectors:
        raise ValueError("cannot sum an empty four-vector collection")
    return tuple(sum(float(vector[axis]) for vector in vectors) for axis in range(4))  # type: ignore[return-value]


def _rest_boost(vector: Sequence[float], system: Sequence[float]) -> tuple[float, float, float, float]:
    if system[0] <= 0.0 or not all(math.isfinite(float(value)) for value in system):
        raise ValueError("rest-frame system has invalid energy or components")
    beta = tuple(float(system[i + 1]) / float(system[0]) for i in range(3))
    return _boost(vector, tuple(-value for value in beta))


def _transverse(vector: Sequence[float], axis: Sequence[float]) -> tuple[float, float, float]:
    projection = _dot(vector, axis)
    return tuple(float(vector[i]) - projection * float(axis[i]) for i in range(3))  # type: ignore[return-value]


def _independent_internal_coordinates(
    momenta: Sequence[Sequence[float]], geometry: Mapping[str, Any],
) -> dict[str, float]:
    """Extract x,y,phi by the physical boosts, independent of production code."""

    if len(momenta) != 4:
        raise ValueError("internal-angle extraction requires four momenta")
    a, b, c, d = tuple(int(value) - 1 for value in geometry["particles"])
    if geometry.get("mode") == "pair":
        q_ab = _sum((momenta[a], momenta[b]))
        q_cd = _sum((momenta[c], momenta[d]))
        axis = _unit(q_ab[1:], "pair axis")
        pa = _rest_boost(momenta[a], q_ab)
        pc = _rest_boost(momenta[c], q_cd)
        ka, kc = _norm(pa[1:]), _norm(pc[1:])
        if ka <= 1.0e-14 or kc <= 1.0e-14:
            raise ValueError("pair-frame daughter momentum is zero")
        x, y = _dot(pa[1:], axis) / ka, _dot(pc[1:], axis) / kc
        transverse_a = _transverse(momenta[a][1:], axis)
        transverse_c = _transverse(momenta[c][1:], axis)
    elif geometry.get("mode") == "nested":
        q_abc = _sum((momenta[a], momenta[b], momenta[c]))
        p_abc = tuple(_rest_boost(momentum, q_abc) for momentum in momenta)
        axis = _unit(p_abc[c][1:], "nested c axis")
        q_ab = _sum((p_abc[a], p_abc[b]))
        pa_ab = _rest_boost(p_abc[a], q_ab)
        ka, kd = _norm(pa_ab[1:]), _norm(p_abc[d][1:])
        if ka <= 1.0e-14 or kd <= 1.0e-14:
            raise ValueError("nested-frame daughter momentum is zero")
        x, y = _dot(pa_ab[1:], axis) / ka, _dot(p_abc[d][1:], axis) / kd
        transverse_a = _transverse(p_abc[a][1:], axis)
        transverse_c = _transverse(p_abc[d][1:], axis)
    else:
        raise ValueError("chart geometry mode must be pair or nested")
    ex = _unit(transverse_a, "first decay-plane transverse direction")
    ey = _unit(_cross(axis, ex), "oriented transverse direction")
    phi = math.atan2(_dot(transverse_c, ey), _dot(transverse_c, ex)) % _TWO_PI
    if not all(math.isfinite(value) for value in (x, y, phi)):
        raise ValueError("independently extracted internal coordinate is non-finite")
    if not abs(x) < 1.0 or not abs(y) < 1.0 or not 0.0 < phi < _TWO_PI or abs(phi - math.pi) <= 1.0e-14:
        raise ValueError("independently extracted internal coordinate is on an excluded boundary")
    return {"x": x, "y": y, "internal_phi": phi}


def _target_state(
    momenta: Sequence[Sequence[float]], *, labels: Sequence[str], geometry: Mapping[str, Any],
    retained_internal: Sequence[str], total_mass: float,
) -> tuple[dict[str, float], tuple[float, ...], dict[str, Any]]:
    invariants = _d5._invariants_from_momenta(momenta, labels)
    internal = _independent_internal_coordinates(momenta, geometry)
    euler, _sigma, orientation = _d5._recover_euler_and_sigma(momenta)
    values = {**invariants, **internal, **euler}
    target = tuple(invariants[name] / (total_mass * total_mass) for name in labels)
    retained_values = tuple(internal[name] for name in retained_internal)
    output = target + retained_values + (euler["theta"], euler["phi"], euler["psi"])
    if len(output) != 8 or not all(math.isfinite(value) for value in output):
        raise ValueError("momentum-derived target map is not a finite eight-coordinate map")
    return values, output, {"target_invariants": invariants, "internal": internal, "euler": euler, "orientation": orientation}


def _map_reference_point(
    x: Sequence[float], *, bridge: Any, labels: Sequence[str], geometry: Mapping[str, Any],
    retained_internal: Sequence[str], masses: Sequence[float], total_mass: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, float, float, float], ...]]:
    point = {
        name: (float(x[index]) * total_mass * total_mass if index < 2 else float(x[index]))
        for index, name in enumerate(_REFERENCE_VARIABLES)
    }
    point["phi0"] %= _TWO_PI
    point["phi12"] %= _TWO_PI
    point["phi34"] %= _TWO_PI
    momenta = _d5._momenta(bridge.numeric_momenta(point, masses, total_mass), "reference momenta")
    _values, output, _evidence = _target_state(
        momenta, labels=labels, geometry=geometry, retained_internal=retained_internal, total_mass=total_mass,
    )
    return output, momenta


def _determinant(matrix: Sequence[Sequence[float]]) -> float:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("finite-difference Jacobian must be square")
    work = [list(map(float, row)) for row in matrix]
    determinant = 1.0
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        value = work[pivot][column]
        if not math.isfinite(value) or value == 0.0:
            return 0.0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        value = work[column][column]
        determinant *= value
        for row in range(column + 1, n):
            factor = work[row][column] / value
            for item in range(column + 1, n):
                work[row][item] -= factor * work[column][item]
            work[row][column] = 0.0
    return determinant


def _wrap(value: float) -> float:
    return (value + math.pi) % _TWO_PI - math.pi


def _jacobian(
    x: Sequence[float], step: float, *, bridge: Any, labels: Sequence[str], geometry: Mapping[str, Any],
    retained_internal: Sequence[str], masses: Sequence[float], total_mass: float, dimension: int,
) -> dict[str, Any]:
    center, _ = _map_reference_point(
        x, bridge=bridge, labels=labels, geometry=geometry, retained_internal=retained_internal,
        masses=masses, total_mass=total_mass,
    )
    matrix = [[0.0] * 8 for _ in range(8)]
    periodic_rows = {dimension + len(retained_internal) + 1, dimension + len(retained_internal) + 2}
    if "internal_phi" in retained_internal:
        periodic_rows.add(dimension + tuple(retained_internal).index("internal_phi"))
    for column in range(8):
        plus, minus = list(x), list(x)
        plus[column] += step
        minus[column] -= step
        yp, _ = _map_reference_point(
            plus, bridge=bridge, labels=labels, geometry=geometry, retained_internal=retained_internal,
            masses=masses, total_mass=total_mass,
        )
        ym, _ = _map_reference_point(
            minus, bridge=bridge, labels=labels, geometry=geometry, retained_internal=retained_internal,
            masses=masses, total_mass=total_mass,
        )
        for row in range(8):
            delta = _wrap(yp[row] - ym[row]) if row in periodic_rows else yp[row] - ym[row]
            matrix[row][column] = delta / (2.0 * step)
    det = _determinant(matrix)
    real = abs(det) * total_mass ** (2 * dimension - 4)
    if not math.isfinite(det) or not math.isfinite(real):
        raise ValueError("finite-difference Jacobian is non-finite")
    return {
        "step": step,
        "matrix": tuple(tuple(row) for row in matrix),
        "dimensionless_determinant_signed": det,
        "dimensionless_determinant_absolute": abs(det),
        "real_dimension_jacobian_absolute": real,
        "output_center": center,
        "periodic_output_rows": tuple(sorted(periodic_rows)),
    }


def _physical_check(
    momenta: Sequence[Sequence[float]], *, masses: Sequence[float], total_mass: float,
    labels: Sequence[str], expected_invariants: Mapping[str, float], expected_source: Mapping[str, float],
    tolerance: float, label: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    errors: list[str] = []
    parent_scale = total_mass * total_mass
    energies = tuple(float(vector[0]) for vector in momenta)
    for index, energy in enumerate(energies, 1):
        if not math.isfinite(energy) or energy <= 0.0:
            errors.append(f"{label} p{index} energy is not finite and positive")
    total = _d5._sum_vectors(momenta)
    energy_res = abs(total[0] - total_mass) / max(abs(total_mass), 1.0e-300)
    momentum_res = max(abs(total[axis]) for axis in range(1, 4)) / max(abs(total_mass), 1.0e-300)
    if not math.isfinite(energy_res) or energy_res > 10.0 * tolerance:
        errors.append(f"{label} energy conservation residual={energy_res:.6g}")
    if not math.isfinite(momentum_res) or momentum_res > 10.0 * tolerance:
        errors.append(f"{label} momentum conservation residual={momentum_res:.6g}")
    shell: list[float] = []
    for i, (vector, mass) in enumerate(zip(momenta, masses), 1):
        actual = _d5._minkowski_square(vector)
        residual = abs(actual - mass * mass) / max(abs(actual), mass * mass, parent_scale, 1.0e-300)
        shell.append(residual)
        if not math.isfinite(residual) or residual > 10.0 * tolerance:
            errors.append(f"{label} p{i} shell residual={residual:.6g}")
    invariants = _d5._invariants_from_momenta(momenta, labels)
    inv_res = {}
    for name in labels:
        residual = abs(invariants[name] - expected_invariants[name]) / max(
            abs(invariants[name]), abs(expected_invariants[name]), parent_scale, 1.0e-300,
        )
        inv_res[name] = residual
        if not math.isfinite(residual) or residual > 10.0 * tolerance:
            errors.append(f"{label} {name} residual={residual:.6g}")
    source_res = {}
    for name, indices in (("s12", (0, 1)), ("s34", (2, 3))):
        actual = _d5._minkowski_square(_d5._sum_vectors(tuple(momenta[i] for i in indices)))
        residual = abs(actual - expected_source[name]) / max(abs(actual), abs(expected_source[name]), parent_scale, 1e-300)
        source_res[name] = residual
        if not math.isfinite(residual) or residual > 10.0 * tolerance:
            errors.append(f"{label} source {name} residual={residual:.6g}")
    return ({
        "energy_conservation_relative_residual": energy_res,
        "positive_energy_by_particle": energies,
        "spatial_conservation_relative_residual": momentum_res,
        "on_shell_relative_residual_by_particle": tuple(shell),
        "target_invariant_relative_residuals": inv_res,
        "target_invariants": invariants,
        "source_pair_relative_residuals": source_res,
    }, tuple(errors))


def _componentwise_momentum_residuals(
    actual: Sequence[Sequence[float]], expected: Sequence[Sequence[float]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    absolute = tuple(
        tuple(abs(float(actual[i][axis]) - float(expected[i][axis])) for axis in range(4))
        for i in range(4)
    )
    relative = tuple(
        tuple(
            difference / max(abs(float(actual[i][axis])), abs(float(expected[i][axis])), 1.0e-300)
            for axis, difference in enumerate(row)
        )
        for i, row in enumerate(absolute)
    )
    return absolute, relative


def _parsed_point(point: Any) -> dict[str, float]:
    if not isinstance(point, Mapping):
        raise ValueError("reference replay point must be a mapping")
    try:
        values = {name: float(point[name]) for name in _REFERENCE_VARIABLES}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reference replay point must contain all eight sequential coordinates") from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("reference replay point contains NaN or infinity")
    if values["s12"] <= 0.0 or values["s34"] <= 0.0:
        raise ValueError("source pair invariant squares must be positive")
    for name in ("theta0", "theta12", "theta34"):
        if not 0.0 < values[name] < math.pi:
            raise ValueError(f"{name} must lie strictly inside (0, pi)")
    for name in ("phi0", "phi12", "phi34"):
        if not 0.0 <= values[name] < _TWO_PI:
            raise ValueError(f"{name} must lie inside [0, 2 pi)")
    return values


def compare_low_loop_paths(
    direct: Any,
    bridge: Any,
    *,
    chart_data: Mapping[str, Any],
    masses: Sequence[float],
    total_mass: float,
    points: Iterable[Mapping[str, float]] | None = None,
    relative_tolerance: float = 1.0e-9,
    fd_relative_tolerance: float = 5.0e-6,
    kinematic_tolerance: float = 1.0e-9,
) -> BenchmarkReport:
    """Compare the direct mixed chart with the fixed sequential 12/34 path."""

    try:
        mass_values = tuple(float(value) for value in masses)
        parent_mass = float(total_mass)
        requested_tol, fd_tol, kin_tol = map(float, (relative_tolerance, fd_relative_tolerance, kinematic_tolerance))
    except (TypeError, ValueError, OverflowError):
        mass_values, parent_mass = (), math.nan
        requested_tol = fd_tol = kin_tol = math.nan
    effective_tol = max(requested_tol, fd_tol) if all(math.isfinite(v) for v in (requested_tol, fd_tol)) else math.nan
    try:
        if points is None:
            from .benchmark import _default_points
            sample = _default_points(mass_values, parent_mass) or ()
        else:
            sample = tuple(points)
    except TypeError:
        sample = ()
    labels = tuple(getattr(direct, "target_variables", ()))
    retained = tuple(chart_data.get("internal_variables", ())) if isinstance(chart_data, Mapping) else ()
    geometry = chart_data.get("geometry", {}) if isinstance(chart_data, Mapping) else {}
    dimension = len(labels)
    phi_eliminated = bool(chart_data.get("elimination", {}).get("phi_eliminated", False)) if isinstance(chart_data, Mapping) else False
    branches = tuple(getattr(direct, "branches", ()))

    def report(status: str, message: str, rows: Sequence[Mapping[str, Any]] = ()) -> BenchmarkReport:
        return _d5._make_report(
            status,
            getattr(bridge, "profile", ""),
            sample,
            compared=sum(1 for row in rows if not row.get("errors")),
            branch_status=("both_internal_phi_roots_and_mirrors_checked" if status == "verified_local" and phi_eliminated else "explicit_internal_phi_mirror_checked" if status == "verified_local" else "failed_or_incomplete"),
            failures=tuple({"point": row.get("point"), "error": "; ".join(row.get("errors", ())) } for row in rows if row.get("errors")),
            diagnostics=(message,) if message else (),
            evidence={
                "replay_points": tuple(sample), "masses": mass_values, "total_mass": parent_mass,
                "requested_relative_tolerance": requested_tol,
                "requested_fd_relative_tolerance": fd_tol,
                "effective_density_tolerance": effective_tol,
                "kinematic_tolerance": kin_tol,
                "effective_kinematic_residual_tolerance": 10.0 * kin_tol if math.isfinite(kin_tol) else math.nan,
                "finite_difference_steps": _FD_STEPS,
                "source_profile": getattr(bridge, "profile", ""),
                "target_profile": getattr(direct, "profile", ""),
                "target_variables": labels, "retained_internal_variables": retained,
                "geometry": geometry, "phi_eliminated": phi_eliminated,
                "density_divisor_for_summed_mode": 2.0 if phi_eliminated and any(getattr(b, "summed", False) for b in branches) else 1.0,
                "coordinate_map": "sequential reference momenta -> subset invariant squares, independently boosted internal coordinates, and Euler ZYZ angles",
                "finite_difference_protocol": {
                    "source_coordinates": "(s12/M^2,s34/M^2,theta0,phi0,theta12,phi12,theta34,phi34)",
                    "target_coordinates": "(d target S/M^2, retained internal x/y/phi, theta,phi,psi)",
                    "real_jacobian_dimension_factor": "dimensionless determinant * M^(2*d-4)",
                },
                "density_comparison": "rho_direct_per_branch*sin(theta)*abs(J_real) versus rho_reference*product(source sin angles)",
                "point_evidence": tuple(rows), "failed": sum(bool(row.get("errors")) for row in rows),
                "scope": "local_numeric_consistency", "global_claim": "not_a_global_formula_proof",
            },
        )

    if not sample or len(sample) < 8:
        return report("unsupported", "at least eight replay points are required")
    if len(mass_values) != 4 or any(not math.isfinite(value) or value <= 0.0 for value in mass_values):
        return report("unsupported", "low-loop benchmark requires four finite, strictly positive masses")
    if not math.isfinite(parent_mass) or parent_mass <= sum(mass_values):
        return report("unsupported", "total_mass must be strictly above the four-body threshold")
    if any(not math.isfinite(value) or value <= 0.0 for value in (requested_tol, fd_tol, kin_tol)):
        return report("unsupported", "all benchmark tolerances must be finite and positive")
    if getattr(direct, "status", None) != "built" or getattr(bridge, "status", None) != "built":
        return report("blocked", "one numeric formula path is not built")
    if getattr(direct, "profile", None) != "direct_gram_low_loop" or getattr(bridge, "profile", None) != "sequential_12_34":
        return report("unsupported", "supported profiles are direct_gram_low_loop and sequential_12_34")
    if not isinstance(chart_data, Mapping) or chart_data.get("status") != "built" or chart_data.get("profile") != "direct_gram_low_loop":
        return report("unsupported", "chart_data must be a built direct_gram_low_loop chart")
    if dimension not in (2, 3, 4) or len(set(labels)) != dimension or tuple(chart_data.get("target_variables", ())) != labels:
        return report("unsupported", "direct/chart target labels must be matching unique d=2,3,4 variables")
    if tuple(chart_data.get("angle_variables", ())) != retained + ("theta", "phi", "psi"):
        return report("unsupported", "chart angle variables must be retained internal coordinates followed by Euler angles")
    if len(retained) != 5 - dimension or tuple(getattr(direct, "angle_variables", ())) != retained + ("theta", "phi", "psi"):
        return report("unsupported", "low-loop target map must have eight independent coordinates")
    if tuple(getattr(bridge, "variables", ())) != _REFERENCE_VARIABLES:
        return report("unsupported", "bridge variables must be the fixed sequential 12/34 source coordinates")
    if getattr(direct, "normalization", None) not in {"jing_psf", "standard_2pi4"} or direct.normalization != getattr(bridge, "normalization", None):
        return report("failed", "direct and reference normalization conventions must match")
    if not all(callable(getattr(path, attr, None)) for path in (direct, bridge) for attr in ("numeric_density", "numeric_momenta")):
        return report("blocked", "one numeric density or momentum adapter is missing")
    if phi_eliminated:
        if len(branches) != 2 or {getattr(item, "key", "") for item in branches} != {"internal_phi_branch=+1", "internal_phi_branch=-1"}:
            return report("unsupported", "eliminated internal_phi must declare both signed roots")
    elif len(branches) != 1 or getattr(branches[0], "key", "") != "internal_phi=explicit-coordinate":
        return report("unsupported", "retained internal_phi must declare one explicit-coordinate branch")
    try:
        for label in labels:
            _d5._subset_from_label(label)
    except ValueError as exc:
        return report("unsupported", str(exc))

    normalized: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
    try:
        for raw in sample:
            point = _parsed_point(raw)
            key = tuple(point[name] for name in _REFERENCE_VARIABLES)
            if key in seen:
                return report("unsupported", "replay sample must contain at least eight unique points")
            seen.add(key)
            normalized.append(point)
    except (TypeError, ValueError, OverflowError) as exc:
        return report("failed", f"invalid replay point: {exc}")
    if len(normalized) < 8:
        return report("unsupported", "replay sample has fewer than eight unique points")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    density_abs: list[float] = []
    density_rel: list[float] = []
    coord_residuals: list[float] = []
    for index, point in enumerate(normalized):
        errors: list[str] = []
        row: dict[str, Any] = {"point": index, "reference_coordinates": point}
        try:
            source_momenta = _d5._momenta(bridge.numeric_momenta(point, mass_values, parent_mass), "reference momenta")
            values, _target_center, extraction = _target_state(
                source_momenta, labels=labels, geometry=geometry, retained_internal=retained, total_mass=parent_mass,
            )
            # Fill only the independent internal variables; values also retain x/y/phi for evidence.
            direct_values = {name: values[name] for name in labels + tuple(retained) + ("theta", "phi", "psi")}
            internal_phi = extraction["internal"]["internal_phi"]
            branch = 1 if math.sin(internal_phi) > 0.0 else -1
            x = (point["s12"] / parent_mass**2, point["s34"] / parent_mass**2,
                 point["theta0"], point["phi0"], point["theta12"], point["phi12"], point["theta34"], point["phi34"])
            jacobians = tuple(_jacobian(
                x, step, bridge=bridge, labels=labels, geometry=geometry, retained_internal=retained,
                masses=mass_values, total_mass=parent_mass, dimension=dimension,
            ) for step in _FD_STEPS)
            rho_ref = float(bridge.numeric_density(point, mass_values, parent_mass))
            density_branch = branch if phi_eliminated else 1
            rho_direct_total = float(direct.numeric_density(direct_values, mass_values, parent_mass, branch=density_branch))
            density_divisor = 2.0 if phi_eliminated and any(getattr(b, "summed", False) for b in branches) else 1.0
            rho_direct_branch = rho_direct_total / density_divisor
            if not all(math.isfinite(value) and value > 0.0 for value in (rho_ref, rho_direct_branch)):
                raise ValueError("path densities must be finite and strictly positive")
            source_sine = math.prod(math.sin(point[name]) for name in ("theta0", "theta12", "theta34"))
            lhs_rows = []
            for jacobian in jacobians:
                lhs = rho_direct_branch * math.sin(values["theta"]) * jacobian["real_dimension_jacobian_absolute"]
                rhs = rho_ref * source_sine
                residual = _d5._relative_residual(lhs, rhs)
                density_abs.append(abs(lhs-rhs)); density_rel.append(residual)
                lhs_rows.append({"step": jacobian["step"], "lhs": lhs, "rhs": rhs,
                                 "absolute_residual": abs(lhs-rhs), "relative_residual": residual,
                                 "passes_effective_tolerance": residual <= effective_tol})
                if not math.isfinite(residual) or residual > effective_tol:
                    errors.append(f"density residual at h={jacobian['step']:.1e} is {residual:.6g} > {effective_tol:.6g}")
            step_residual = _d5._relative_residual(jacobians[0]["real_dimension_jacobian_absolute"], jacobians[1]["real_dimension_jacobian_absolute"])
            row.update({
                "target_coordinates": direct_values,
                "independent_extraction": extraction,
                "internal_phi_branch": branch,
                "density_branch_argument": density_branch,
                "density_reference": rho_ref,
                "density_direct_total": rho_direct_total,
                "density_direct_per_branch": rho_direct_branch,
                "density_divisor_for_summed_mode": density_divisor,
                "sine_weighted_density_comparison": tuple(lhs_rows),
                "jacobians": jacobians,
                "relative_difference_between_fd_steps": step_residual,
            })

            reflected = _d5._reflect_mirror(source_momenta)
            _, _, reflected_extraction = _target_state(
                reflected, labels=labels, geometry=geometry, retained_internal=retained, total_mass=parent_mass,
            )
            reflected_branch = 1 if math.sin(reflected_extraction["internal"]["internal_phi"]) > 0.0 else -1
            x_res = abs(reflected_extraction["internal"]["x"] - extraction["internal"]["x"])
            y_res = abs(reflected_extraction["internal"]["y"] - extraction["internal"]["y"])
            phi_sum_res = abs(_wrap(reflected_extraction["internal"]["internal_phi"] + internal_phi))
            if x_res > 10.0 * kin_tol or y_res > 10.0 * kin_tol or phi_sum_res > 10.0 * kin_tol:
                errors.append(f"reflected internal coordinates fail mirror rule: x={x_res:.6g}, y={y_res:.6g}, phi-sum={phi_sum_res:.6g}")
            if phi_eliminated and reflected_branch != -branch:
                errors.append("reflected event did not switch sign(sin(internal_phi)) branch")

            if phi_eliminated:
                branch_candidates = ((branch, source_momenta, extraction), (reflected_branch, reflected, reflected_extraction))
                if branch_candidates[0][0] == branch_candidates[1][0]:
                    raise ValueError("reference and reflected events did not select distinct internal_phi roots")
            else:
                branch_candidates = ((1, source_momenta, extraction), (1, reflected, reflected_extraction))
            mirror_rows = []
            seen_expected: set[tuple[int, str]] = set()
            for selected_branch, expected, expected_extraction in branch_candidates:
                kind = "source" if expected is source_momenta else "p1p2_plane_reflection"
                marker = (selected_branch, kind)
                if marker in seen_expected:
                    raise ValueError("duplicate branch/mirror evidence row")
                seen_expected.add(marker)
                expected_values = {
                    name: expected_extraction["target_invariants"][name] for name in labels
                }
                expected_values.update({name: expected_extraction["internal"][name] for name in retained})
                expected_values.update(expected_extraction["euler"])
                source_quality, source_errors = _physical_check(
                    expected, masses=mass_values, total_mass=parent_mass, labels=labels,
                    expected_invariants=extraction["target_invariants"], expected_source={"s12": point["s12"], "s34": point["s34"]},
                    tolerance=kin_tol, label=f"reference {kind}",
                )
                errors.extend(source_errors)
                reconstructed = _d5._momenta(
                    direct.numeric_momenta(expected_values, mass_values, parent_mass, branch=selected_branch),
                    f"direct {kind} branch={selected_branch:+d} momenta",
                )
                quality, physical_errors = _physical_check(
                    reconstructed, masses=mass_values, total_mass=parent_mass, labels=labels,
                    expected_invariants=extraction["target_invariants"], expected_source={"s12": point["s12"], "s34": point["s34"]},
                    tolerance=kin_tol, label=f"direct {kind} branch={selected_branch:+d}",
                )
                errors.extend(physical_errors)
                absolute, relative = _d5._max_momentum_residual(reconstructed, expected)
                component_absolute, component_relative = _componentwise_momentum_residuals(reconstructed, expected)
                coord_residuals.append(relative)
                if relative > 10.0 * kin_tol:
                    errors.append(f"direct {kind} momentum mismatch={relative:.6g} (absolute={absolute:.6g})")
                mirror_rows.append({
                    "kind": kind, "branch_argument": selected_branch,
                    "branch_from_sin_internal_phi": 1 if math.sin(expected_extraction["internal"]["internal_phi"]) > 0.0 else -1,
                    "internal_coordinates": expected_extraction["internal"],
                    "euler_angles": expected_extraction["euler"],
                    "reference_momenta": expected, "reference_quality": source_quality,
                    "momentum_absolute_residual": absolute, "momentum_relative_residual": relative,
                    "momentum_component_absolute_residuals": component_absolute,
                    "momentum_component_relative_residuals": component_relative,
                    "direct_quality": quality, "direct_momenta": reconstructed,
                })
            row["mirror_coordinates"] = {
                "source": extraction["internal"], "reflected": reflected_extraction["internal"],
                "x_residual": x_res, "y_residual": y_res, "phi_sum_residual": phi_sum_res,
                "source_branch": branch, "reflected_branch": reflected_branch,
            }
            row["mirrors_and_branches"] = tuple(mirror_rows)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        row["errors"] = tuple(errors)
        rows.append(row)
        if errors:
            failures.append({"point": index, "coordinates": point, "error": "; ".join(errors)})

    status = "verified_local" if len(rows) == len(normalized) and not failures else "failed"
    report = _d5._make_report(
        status, getattr(bridge, "profile", ""), normalized,
        compared=sum(not row["errors"] for row in rows),
        max_absolute=max(density_abs, default=None), max_relative=max(density_rel, default=None),
        max_coordinate=max(coord_residuals, default=None),
        branch_status=("both_internal_phi_roots_and_mirrors_checked" if status == "verified_local" and phi_eliminated else "explicit_internal_phi_mirror_checked" if status == "verified_local" else "failed_or_incomplete"),
        failures=failures,
        diagnostics=("low-loop finite-point benchmark checks independent boosted coordinates, both finite-difference steps, physical constraints, and the reflected event", "verified_local is finite-sample evidence, not a global formula proof"),
        evidence={
            "replay_points": tuple(normalized), "masses": mass_values, "total_mass": parent_mass,
            "requested_relative_tolerance": requested_tol, "requested_fd_relative_tolerance": fd_tol,
            "effective_density_tolerance": effective_tol, "kinematic_tolerance": kin_tol,
            "effective_kinematic_residual_tolerance": 10.0 * kin_tol,
            "finite_difference_steps": _FD_STEPS, "target_variables": labels,
            "retained_internal_variables": retained, "geometry": geometry,
            "phi_eliminated": phi_eliminated,
            "density_divisor_for_summed_mode": 2.0 if phi_eliminated and any(getattr(b, "summed", False) for b in branches) else 1.0,
            "finite_difference_protocol": {"source_dimension": 8, "target_dimension": 8,
                "real_jacobian_dimension_factor": "dimensionless determinant * M^(2*d-4)"},
            "density_comparison": "rho_direct_per_branch*sin(theta)*abs(J_real) versus rho_reference*sin(theta0)*sin(theta12)*sin(theta34)",
            "point_evidence": tuple(rows), "failed": len(failures),
            "scope": "local_numeric_consistency", "global_claim": "not_a_global_formula_proof",
        },
    )
    return report


__all__ = ["compare_low_loop_paths"]
