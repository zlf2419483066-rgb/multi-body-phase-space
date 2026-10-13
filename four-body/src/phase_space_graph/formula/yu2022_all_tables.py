"""Independent numerical pullback checks for Yu (2022), Tables 5--26.

This module evaluates only the source expressions stored in the primary-audit
specification.  Momentum-derived coordinates and the angular Gram factor are
rebuilt from the Euclidean COM vectors; production Gram and density helpers are
not used.  Reports are finite-sample evidence and do not promote source status.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import d5_benchmark as _d5
from .yu2022_check import YU2022_PDF_SHA256, _boost, _determinant, _lambda


_DEMO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    _DEMO_ROOT / "文档" / "验证与测试" / "误差分析"
    / "Yu2022全22表核对_2026-10-09" / "主审22表密度与坐标规格.json"
)
_SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
TABLES: dict[int, dict[str, Any]] = {
    int(item["table"]): item for item in _SPEC["tables"]
}
if tuple(sorted(TABLES)) != tuple(range(5, 27)):
    raise RuntimeError("the primary-audit specification must contain Tables 5--26")

_TWO_PI = 2.0 * math.pi
_SOURCE_SCALE = (2.0 * math.pi) ** -12
_SUPPORTED_NORMALIZATIONS = {"jing_psf", "standard_2pi4"}
_FUNCTIONS = {"sqrt": math.sqrt, "sin": math.sin, "abs": abs, "lam": _lambda}
_BINARY_NODES = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_UNARY_NODES = (ast.UAdd, ast.USub)


def _as_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _norm3(a: Sequence[float]) -> float:
    return math.sqrt(_dot3(a, a))


def _unit_cosine(value: float, label: str) -> float:
    if not math.isfinite(value) or abs(value) > 1.0 + 1.0e-11:
        raise ValueError(f"{label} is outside the cosine range")
    return min(1.0, max(-1.0, value))


def _subset(label: str) -> tuple[int, ...]:
    if not label.startswith("s"):
        raise ValueError(f"invalid invariant label {label!r}")
    digits = label[1:]
    if len(digits) not in (2, 3, 4) or any(char not in "1234" for char in digits):
        raise ValueError(f"invalid invariant label {label!r}")
    result = tuple(int(char) - 1 for char in digits)
    if len(set(result)) != len(result):
        raise ValueError(f"invariant label repeats a particle: {label!r}")
    return result


def _safe_expression(expression: str, values: Mapping[str, float]) -> float:
    """Evaluate only arithmetic, approved names, and approved math calls."""

    tree = ast.parse(expression, mode="eval")

    def check(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            check(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("source expression contains a non-numeric constant")
        elif isinstance(node, ast.Name):
            if node.id not in values and node.id not in _FUNCTIONS:
                raise ValueError(f"source expression uses unknown name {node.id!r}")
            if node.id.startswith("__"):
                raise ValueError("dunder names are forbidden in source expressions")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, _BINARY_NODES):
            check(node.left)
            check(node.right)
            if isinstance(node.op, ast.Pow) and not (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, (int, float))
                and abs(float(node.right.value)) <= 12.0
            ):
                raise ValueError("source-expression powers must be small numeric constants")
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY_NODES):
            check(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _FUNCTIONS or node.keywords:
                raise ValueError("source expression contains an unsupported function call")
            for argument in node.args:
                check(argument)
        else:
            raise ValueError(f"source expression contains unsupported syntax: {type(node).__name__}")

    check(tree)
    environment = {"__builtins__": {}, **_FUNCTIONS, **values}
    result = float(eval(compile(tree, "<Yu2022 source density>", "eval"), environment, {}))
    if not math.isfinite(result):
        raise ValueError("source density expression returned a non-finite value")
    return result


def _event_validation(
    vectors: Sequence[Sequence[float]], mass_values: Sequence[float], parent_mass: float,
) -> dict[str, Any]:
    total = _d5._sum_vectors(vectors)
    scale = max(1.0, parent_mass)
    energy_residual = abs(total[0] - parent_mass) / scale
    spatial_residual = max(abs(total[i]) for i in range(1, 4)) / scale
    shell_residuals = tuple(
        abs(_d5._minkowski_square(vector) - mass * mass)
        / max(1.0, parent_mass * parent_mass, mass * mass)
        for vector, mass in zip(vectors, mass_values)
    )
    if energy_residual > 1.0e-8 or spatial_residual > 1.0e-8:
        raise ValueError("input momenta are not a COM event with the supplied total mass")
    if any(residual > 1.0e-8 for residual in shell_residuals):
        raise ValueError("input momenta are not on shell for the supplied masses")
    return {
        "energy_conservation_relative_residual": energy_residual,
        "spatial_conservation_relative_residual": spatial_residual,
        "on_shell_relative_residual_by_particle": shell_residuals,
    }


def event_data(
    momenta: Sequence[Sequence[float]], masses: Sequence[float], total_mass: float,
) -> dict[str, Any]:
    """Rebuild all table inputs from four COM momenta and positive masses.

    For a starred variable, Q is the named subset sum in the parent COM frame;
    the selected p_i is explicitly boosted with beta=-Q_vec/Q_0 into the Q rest
    frame.  Its cosine is minus the dot product of the boosted spatial vector
    and the unit parent-COM Q direction, divided by the boosted momentum norm.
    The unstarred theta_i_subset uses the parent-COM p_i and Q vectors.
    """

    vectors = _d5._momenta(momenta, "Yu2022 table-check momenta")
    mass_values = tuple(_as_float(value, f"m{index + 1}") for index, value in enumerate(masses))
    if len(mass_values) != 4 or any(value <= 0.0 for value in mass_values):
        raise ValueError("Yu2022 tables require four strictly positive masses")
    parent_mass = _as_float(total_mass, "total_mass")
    if parent_mass <= sum(mass_values):
        raise ValueError("total_mass must be strictly above the four-body threshold")
    quality = _event_validation(vectors, mass_values, parent_mass)

    data: dict[str, Any] = {f"m{i + 1}": value for i, value in enumerate(mass_values)}
    data["M"] = parent_mass
    for i, vector in enumerate(vectors):
        data[f"E{i + 1}"] = vector[0]
        data[f"k{i + 1}"] = _norm3(vector[1:])
        if data[f"k{i + 1}"] <= 0.0:
            raise ValueError(f"k{i + 1} must be strictly positive")

    invariant_names = tuple(
        "s" + "".join(str(index + 1) for index in subset)
        for width in (2, 3, 4)
        for subset in __import__("itertools").combinations(range(4), width)
    )
    invariant_subsets = {name: _subset(name) for name in invariant_names}
    subset_vectors: dict[str, tuple[float, float, float, float]] = {}
    for name, subset in invariant_subsets.items():
        total = _d5._sum_vectors(tuple(vectors[index] for index in subset))
        subset_vectors[name] = total
        data[name] = _d5._minkowski_square(total)

    pair_cosines: dict[tuple[int, int], float] = {}
    for first, second in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)):
        p_first, p_second = vectors[first - 1][1:], vectors[second - 1][1:]
        cosine = _unit_cosine(
            _dot3(p_first, p_second) / (data[f"k{first}"] * data[f"k{second}"]),
            f"c{first}{second}",
        )
        pair_cosines[(first, second)] = cosine
        data[f"c{first}{second}"] = cosine
        data[f"theta{first}{second}"] = math.acos(cosine)
    for (first, second), cosine in tuple(pair_cosines.items()):
        data[f"c{second}{first}"] = cosine
        data[f"theta{second}{first}"] = math.acos(cosine)

    # The angular Gram factor is deliberately formed from Euclidean k and c.
    c12, c13, c23 = data["c12"], data["c13"], data["c23"]
    angular_gram = 1.0 + 2.0 * c12 * c13 * c23 - c12 * c12 - c13 * c13 - c23 * c23
    if not math.isfinite(angular_gram) or angular_gram <= 0.0:
        raise ValueError("Euclidean external angular Gram determinant is not strictly positive")
    data["Ag"] = 1.0 / math.sqrt(angular_gram)
    data["angular_gram_determinant_euclidean"] = angular_gram

    needed_starred: set[str] = set()
    for table in TABLES.values():
        expression = table["density_without_2pi"]
        for node in ast.walk(ast.parse(expression, mode="eval")):
            if isinstance(node, ast.Name) and node.id.startswith("c_"):
                needed_starred.add(node.id)
        for variable in table["variables"]:
            if variable.startswith("c_"):
                needed_starred.add(variable)

    for name in sorted(needed_starred):
        _, particle_text, subset_text = name.split("_")
        particle = int(particle_text)
        subset = tuple(int(char) - 1 for char in subset_text)
        q = _d5._sum_vectors(tuple(vectors[index] for index in subset))
        q2 = _d5._minkowski_square(q)
        qnorm = _norm3(q[1:])
        if q2 <= 0.0 or qnorm <= 1.0e-14 * parent_mass:
            raise ValueError(f"{name} has a degenerate subset direction")
        axis = tuple(component / qnorm for component in q[1:])
        beta = tuple(-q[i + 1] / q[0] for i in range(3))
        boosted = _boost(vectors[particle - 1], beta)
        kstar = _norm3(boosted[1:])
        if kstar <= 1.0e-14 * parent_mass:
            raise ValueError(f"{name} has a degenerate boosted momentum")
        data[name] = _unit_cosine(-_dot3(boosted[1:], axis) / kstar, name)
        ks_name = f"ks_{particle}_{subset_text}"
        data[ks_name] = kstar
        subset_mass = math.sqrt(q2)
        data[f"g{subset_text}"] = qnorm / subset_mass

    # q_S denotes the parent-COM spatial norm of the named subset momentum.
    for suffix in ("12", "34"):
        q = subset_vectors[f"s{suffix}"]
        qnorm = _norm3(q[1:])
        data[f"q{suffix}"] = qnorm
        if qnorm <= 1.0e-14 * parent_mass:
            raise ValueError(f"q{suffix} direction is degenerate")
        for particle in range(1, 5):
            p = vectors[particle - 1][1:]
            data[f"theta{particle}_{suffix}"] = math.acos(
                _unit_cosine(_dot3(p, q[1:]) / (data[f"k{particle}"] * qnorm), f"theta{particle}_{suffix}")
            )

    # The specification's D22 is kept as written, using its COM angle inputs.
    data["D22"] = (
        data["E4"] * (data["E1"] + data["E2"]) * math.sin(data["theta4_12"]) ** 2
        + data["E3"] * (data["E1"] + data["E2"])
        * math.sin(data["theta34"] + data["theta4_12"]) ** 2
        + data["E3"] * data["E4"] * math.sin(data["theta34"]) ** 2
    )
    if data["D22"] == 0.0 or not math.isfinite(data["D22"]):
        raise ValueError("D22 denominator is zero or non-finite")
    data["event_quality"] = quality
    return data


def _table_spec(table: int | str) -> dict[str, Any]:
    if isinstance(table, str):
        lowered = table.lower().removeprefix("table")
        try:
            table = int(lowered)
        except ValueError as exc:
            raise ValueError(f"unknown Yu2022 table {table!r}") from exc
    try:
        return TABLES[int(table)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Yu2022 table {table!r} is outside 5--26") from exc


def _coordinate_variables(spec: Mapping[str, Any], variant: str | None) -> tuple[str, ...]:
    if variant in (None, "row", "row_variables"):
        return tuple(spec["variables"])
    if variant in ("alternative", "alternative_variables"):
        if "alternative_variables" not in spec:
            raise ValueError(f"Table {spec['table']} has no alternative variable caption")
        return tuple(spec["alternative_variables"])
    raise ValueError("variant must be row or alternative_variables")


def _coordinate_value(name: str, data: Mapping[str, Any], parent_mass: float) -> float:
    if name.startswith("s"):
        return float(data[name]) / (parent_mass * parent_mass)
    if name.startswith("c"):
        return float(data[name])
    if name.startswith("theta"):
        return float(data[name])
    raise ValueError(f"unsupported Yu2022 coordinate {name!r}")


def table_coordinates(
    table: int | str,
    momenta: Sequence[Sequence[float]],
    masses: Sequence[float],
    total_mass: float,
    *,
    variant: str | None = None,
) -> dict[str, Any]:
    """Return the row's five coordinates followed by ZYZ Euler angles.

    Every invariant-square output is dimensionless s/M^2, and only the last
    two Euler outputs are periodic rows in the finite-difference map.
    """

    spec = _table_spec(table)
    parent_mass = _as_float(total_mass, "total_mass")
    data = event_data(momenta, masses, parent_mass)
    variables = _coordinate_variables(spec, variant)
    euler, sigma, orientation = _d5._recover_euler_and_sigma(momenta)
    names = variables + ("thetaEuler", "phiEuler", "psiEuler")
    values = tuple(_coordinate_value(name, data, parent_mass) for name in variables) + (
        euler["theta"], euler["phi"], euler["psi"],
    )
    if len(values) != 8 or any(not math.isfinite(value) for value in values):
        raise ValueError(f"Table {spec['table']} coordinates are not a finite eight-vector")
    return {
        "table": int(spec["table"]),
        "variant": "alternative_variables" if variant in ("alternative", "alternative_variables") else "row_variables",
        "coordinate_names": names,
        "coordinate_values": values,
        "external_values": {name: data[name] for name in variables},
        "euler_angles": euler,
        "orientation_sigma": sigma,
        "orientation": orientation,
        "event_data": data,
    }


def table_density(
    table: int | str,
    momenta: Sequence[Sequence[float]],
    masses: Sequence[float],
    total_mass: float,
    *,
    normalization: str = "jing_psf",
) -> dict[str, Any]:
    """Evaluate a transcribed row in the restricted source-expression sandbox."""

    spec = _table_spec(table)
    if normalization not in _SUPPORTED_NORMALIZATIONS:
        raise ValueError("normalization must be jing_psf or standard_2pi4")
    data = event_data(momenta, masses, total_mass)
    unscaled = _safe_expression(spec["density_without_2pi"], data)
    if unscaled <= 0.0:
        raise ValueError(f"Table {spec['table']} source density is non-positive")
    normalization_factor = 1.0 if normalization == "jing_psf" else _TWO_PI**4
    density = unscaled * _SOURCE_SCALE * normalization_factor
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError(f"Table {spec['table']} normalized source density is invalid")
    return {
        "table": int(spec["table"]),
        "density": density,
        "density_without_2pi": unscaled,
        "source_scale": _SOURCE_SCALE,
        "normalization": normalization,
        "normalization_factor": normalization_factor,
        "source_expression": spec["density_without_2pi"],
        "event_data": data,
    }


def _point_from_dimensionless(values: Sequence[float], total_mass: float) -> dict[str, float]:
    if len(values) != 8:
        raise ValueError("reference coordinates must have eight components")
    return {
        "s12": float(values[0]) * total_mass**2,
        "s34": float(values[1]) * total_mass**2,
        "theta0": float(values[2]),
        "phi0": float(values[3]) % _TWO_PI,
        "theta12": float(values[4]),
        "phi12": float(values[5]) % _TWO_PI,
        "theta34": float(values[6]),
        "phi34": float(values[7]) % _TWO_PI,
    }


def _wrap_delta(value: float) -> float:
    return (value + math.pi) % _TWO_PI - math.pi


def _relative_residual(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1.0e-300)


def _reference_momenta(
    values: Sequence[float], bridge: Any, masses: Sequence[float], total_mass: float,
    mirror: int,
) -> tuple[tuple[float, float, float, float], ...]:
    point = _point_from_dimensionless(values, total_mass)
    vectors = _d5._momenta(
        bridge.numeric_momenta(point, masses, total_mass), "fixed sequential reference momenta",
    )
    if mirror == -1:
        vectors = _d5._reflect_mirror(vectors)
    elif mirror != 1:
        raise ValueError("mirror must be +1 or -1")
    return vectors


def _mapped_coordinates(
    values: Sequence[float], *, bridge: Any, table: int, variant: str,
    masses: Sequence[float], total_mass: float, mirror: int,
) -> tuple[float, ...]:
    vectors = _reference_momenta(values, bridge, masses, total_mass, mirror)
    return table_coordinates(
        table, vectors, masses, total_mass, variant=variant,
    )["coordinate_values"]


def _jacobian(
    values: Sequence[float], step: float, *, bridge: Any, table: int, variant: str,
    masses: Sequence[float], total_mass: float, mirror: int,
) -> dict[str, Any]:
    if step <= 0.0 or not math.isfinite(step):
        raise ValueError("finite-difference step must be finite and positive")
    center = _mapped_coordinates(
        values, bridge=bridge, table=table, variant=variant,
        masses=masses, total_mass=total_mass, mirror=mirror,
    )
    matrix = [[0.0] * 8 for _ in range(8)]
    for column in range(8):
        plus, minus = list(values), list(values)
        plus[column] += step
        minus[column] -= step
        y_plus = _mapped_coordinates(
            plus, bridge=bridge, table=table, variant=variant,
            masses=masses, total_mass=total_mass, mirror=mirror,
        )
        y_minus = _mapped_coordinates(
            minus, bridge=bridge, table=table, variant=variant,
            masses=masses, total_mass=total_mass, mirror=mirror,
        )
        for row in range(8):
            delta = _wrap_delta(y_plus[row] - y_minus[row]) if row in (6, 7) else y_plus[row] - y_minus[row]
            matrix[row][column] = delta / (2.0 * step)
    det = _determinant(matrix)
    spec = TABLES[table]
    dimensions = sum(str(name).startswith("s") for name in spec["variables"])
    exponent = 2 * dimensions - 4
    if exponent != int(spec["mass_dimension_factor_exponent"]):
        raise ValueError(f"Table {table} dimension count conflicts with the primary-audit specification")
    real_absdet = abs(det) * total_mass**exponent
    if not math.isfinite(det) or not math.isfinite(real_absdet):
        raise ValueError("finite-difference Jacobian is non-finite")
    return {
        "step": step,
        "dimensionless_determinant_signed": det,
        "dimensionless_determinant_absolute": abs(det),
        "real_jacobian_absolute": real_absdet,
        "mass_dimension_factor": f"M^{exponent}",
        "periodic_output_rows": (6, 7),
        "matrix": tuple(tuple(row) for row in matrix),
        "output_center": center,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def run_check(
    bridge: Any,
    *,
    masses: Sequence[float],
    total_mass: float,
    points: Iterable[Mapping[str, float]] | None = None,
    normalization: str = "jing_psf",
    steps: Sequence[float] = (2.0e-5, 1.0e-5),
) -> dict[str, Any]:
    """Compare all 22 rows, both mirrors, and each requested FD step.

    Each exception is retained in a comparison row and marks that table entry
    ``error``; no point or mirror is silently omitted.  Table 23 and 24 caption
    alternatives are evaluated as separately named variants.
    """

    if not callable(getattr(bridge, "numeric_momenta", None)) or not callable(
        getattr(bridge, "numeric_density", None)
    ):
        raise ValueError("bridge must provide numeric_momenta and numeric_density")
    mass_values = tuple(_as_float(value, f"m{index + 1}") for index, value in enumerate(masses))
    parent_mass = _as_float(total_mass, "total_mass")
    if len(mass_values) != 4 or any(value <= 0.0 for value in mass_values):
        raise ValueError("run_check requires four strictly positive masses")
    if parent_mass <= sum(mass_values):
        raise ValueError("total_mass must be strictly above threshold")
    if normalization not in _SUPPORTED_NORMALIZATIONS:
        raise ValueError("normalization must be jing_psf or standard_2pi4")
    step_values = tuple(_as_float(value, "finite-difference step") for value in steps)
    if not step_values or any(value <= 0.0 for value in step_values):
        raise ValueError("steps must contain one or more finite positive values")
    sample = tuple(points) if points is not None else _d5_default_points(mass_values, parent_mass)
    if len(sample) < 8:
        raise ValueError("run_check requires at least eight reference points")

    table_reports: list[dict[str, Any]] = []
    for table_number in sorted(TABLES):
        spec = TABLES[table_number]
        variants = ["row_variables"]
        if "alternative_variables" in spec:
            variants.append("alternative_variables")
        for variant in variants:
            coordinate_variant = "row" if variant == "row_variables" else "alternative_variables"
            comparisons: list[dict[str, Any]] = []
            table_errors = 0
            for point_index, reference_point in enumerate(sample):
                try:
                    point = {name: float(reference_point[name]) for name in _d5._REFERENCE_VARIABLES}
                    x = (
                        point["s12"] / parent_mass**2,
                        point["s34"] / parent_mass**2,
                        point["theta0"], point["phi0"], point["theta12"], point["phi12"],
                        point["theta34"], point["phi34"],
                    )
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    for mirror in (1, -1):
                        for step in step_values:
                            comparisons.append({
                                "point_index": point_index,
                                "reference_coordinates": _json_safe(reference_point),
                                "mirror": mirror,
                                "step": step,
                                "status": "error",
                                "error": f"invalid reference point: {exc}",
                            })
                            table_errors += 1
                    continue

                for mirror in (1, -1):
                    for step in step_values:
                        row: dict[str, Any] = {
                            "point_index": point_index,
                            "reference_coordinates": point,
                            "mirror": mirror,
                            "step": step,
                            "status": "evaluated",
                        }
                        try:
                            vectors = _reference_momenta(x, bridge, mass_values, parent_mass, mirror)
                            coordinate_result = table_coordinates(
                                table_number, vectors, mass_values, parent_mass,
                                variant=coordinate_variant,
                            )
                            density_result = table_density(
                                table_number, vectors, mass_values, parent_mass,
                                normalization=normalization,
                            )
                            jacobian = _jacobian(
                                x, step, bridge=bridge, table=table_number,
                                variant=coordinate_variant, masses=mass_values,
                                total_mass=parent_mass, mirror=mirror,
                            )
                            rho_reference = float(bridge.numeric_density(point, mass_values, parent_mass))
                            source_sine = (
                                math.sin(point["theta0"])
                                * math.sin(point["theta12"])
                                * math.sin(point["theta34"])
                            )
                            target_sine = math.sin(coordinate_result["euler_angles"]["theta"])
                            lhs = density_result["density"] * target_sine * jacobian["real_jacobian_absolute"]
                            rhs = rho_reference * source_sine
                            if not all(math.isfinite(value) and value > 0.0 for value in (lhs, rhs)):
                                raise ValueError("sine-weighted pullback densities must be finite and positive")
                            row.update({
                                "source_expression": spec["density_without_2pi"],
                                "table_coordinates": coordinate_result["coordinate_values"],
                                "coordinate_names": coordinate_result["coordinate_names"],
                                "external_density": density_result["density"],
                                "reference_density": rho_reference,
                                "external_pullback_density": lhs,
                                "reference_pullback_density": rhs,
                                "absolute_error": abs(lhs - rhs),
                                "relative_residual": _relative_residual(lhs, rhs),
                                "ratio": lhs / rhs,
                                "jacobian": jacobian,
                                "external_density_evidence": {
                                    key: value for key, value in density_result.items()
                                    if key != "event_data"
                                },
                            })
                        except Exception as exc:
                            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                            table_errors += 1
                        comparisons.append(_json_safe(row))
            table_reports.append({
                "table": table_number,
                "variant": variant,
                "status": "error" if table_errors else "evaluated",
                "source_expression": spec["density_without_2pi"],
                "variables": list(spec["variables"] if variant == "row_variables" else spec["alternative_variables"]),
                "interpretation": spec.get("interpretation"),
                "source_measure_conflict": spec.get("source_measure_conflict"),
                "comparison_count": len(comparisons),
                "error_count": table_errors,
                "comparisons": comparisons,
            })

    case = {
        "masses": mass_values,
        "total_mass": parent_mass,
        "normalization": normalization,
        "finite_difference_steps": step_values,
        "reference_profile": getattr(bridge, "profile", ""),
        "reference_density_expression": getattr(bridge, "density_expression", ""),
        "reference_coordinates": list(_d5._REFERENCE_VARIABLES),
        "coordinate_map": "reference dimensionless (s12/M2,s34/M2,angles) -> five table row variables with every s/M2 -> ZYZ Euler(theta,phi,psi)",
        "real_jacobian_conversion": "abs(dimless det)*M^(2*d-4), d is count of invariant-square row variables",
        "normalization_rule": _SPEC["normalization"],
        "density_comparison": "A_table*sin(thetaEuler)*absJ versus reference_density*sin(theta0)*sin(theta12)*sin(theta34)",
        "branches": _SPEC["branches"],
        "tables": table_reports,
    }
    return {"schema_version": "yu2022-all-tables-pullback-v1", "cases": [_json_safe(case)]}


def _d5_default_points(masses: Sequence[float], total_mass: float) -> tuple[dict[str, float], ...]:
    # Reuse only the deterministic sequential reference point grid; no formula
    # catalog or production density helper participates in this audit.
    from .benchmark import _default_points

    result = _default_points(masses, total_mass)
    if result is None or len(result) < 8:
        raise ValueError("could not build eight physical default reference points")
    return result


__all__ = ["event_data", "table_coordinates", "table_density", "run_check", "TABLES", "SPEC_PATH"]
