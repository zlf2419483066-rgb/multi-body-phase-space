"""Direct analytic mixed-coordinate charts for four-body L=0,1,2 graphs.

The exact affine layer in this module maps the graph IKVs to a selected
two-subsystem mass pair and a full five-invariant geometric basis.  The
continuous angle layer then removes exactly the graph constraints described
in ``低圈直接混合坐标_实施规格_2026-10-09.md``.  This is a separate chart
from both the recursive pair chart and the d=5 Gram chart.
"""

from __future__ import annotations

from fractions import Fraction
import itertools
import math
from typing import Any, Mapping, Sequence

from ..formula_inputs import FormulaInputCard
from .d5_direct import (
    D5_BASIS,
    _MASS_SYMBOLS,
    _fraction_text,
    _fraction_latex,
    _matrix_latex,
    _matrix_det,
    _matrix_inverse,
    _subset_form,
)
from .inputs import validate_formula_card
from .kinematic_domain import validate_masses
from .records import BranchRecord, CoordinateMap, DirectFormula, FormulaOptions

Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]
MatrixQ = tuple[tuple[Fraction, ...], ...]
LOW_LOOP_EULER_VARIABLES = ("theta", "phi", "psi")
_MASS_SYMBOL_INDEX = {name: index for index, name in enumerate(_MASS_SYMBOLS)}


def _parse_label(label: str) -> tuple[int, ...]:
    if not isinstance(label, str) or not label.startswith("s"):
        raise ValueError(f"invalid low-loop invariant label: {label!r}")
    try:
        subset = tuple(int(char) for char in label[1:])
    except ValueError as exc:
        raise ValueError(f"invalid low-loop invariant label: {label!r}") from exc
    if (
        len(subset) not in (2, 3)
        or tuple(sorted(subset)) != subset
        or len(set(subset)) != len(subset)
        or any(item not in (1, 2, 3, 4) for item in subset)
        or "s" + "".join(str(item) for item in subset) != label
    ):
        raise ValueError(f"low-loop labels must be increasing two-/three-particle subsets: {label!r}")
    return subset


def _label(subset: Sequence[int]) -> str:
    return "s" + "".join(str(item) for item in sorted(subset))


def _rank(rows: Sequence[Sequence[Fraction]]) -> int:
    if not rows:
        return 0
    work = [[Fraction(value) for value in row] for row in rows]
    width = len(work[0])
    if any(len(row) != width for row in work):
        raise ValueError("matrix rows have inconsistent widths")
    pivot_row = 0
    for column in range(width):
        pivot = next((i for i in range(pivot_row, len(work)) if work[i][column]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        value = work[pivot_row][column]
        work[pivot_row] = [item / value for item in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or not work[row][column]:
                continue
            multiplier = work[row][column]
            work[row] = [a - multiplier * b for a, b in zip(work[row], work[pivot_row])]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def _solve_rectangular(
    matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]
) -> tuple[Fraction, ...] | None:
    """Solve A x=b exactly, returning None unless the solution is unique."""

    rows = len(matrix)
    if rows != len(rhs):
        raise ValueError("linear system row count mismatch")
    cols = len(matrix[0]) if rows else 0
    if any(len(row) != cols for row in matrix):
        raise ValueError("linear system has ragged rows")
    work = [list(map(Fraction, row)) + [Fraction(value)] for row, value in zip(matrix, rhs)]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(cols):
        pivot = next((i for i in range(pivot_row, rows) if work[i][column]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [item / scale for item in work[pivot_row]]
        for i in range(rows):
            if i == pivot_row or not work[i][column]:
                continue
            scale = work[i][column]
            work[i] = [a - scale * b for a, b in zip(work[i], work[pivot_row])]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    if any(not any(row[:cols]) and row[cols] for row in work):
        return None
    if len(pivot_columns) != cols:
        return None
    solution = [Fraction(0)] * cols
    for i, column in enumerate(pivot_columns):
        solution[column] = work[i][cols]
    return tuple(solution)


def _json_matrix(matrix: Sequence[Sequence[Fraction]]) -> list[list[str]]:
    return [[_fraction_text(Fraction(value)) for value in row] for row in matrix]


def _json_vector(vector: Sequence[Fraction]) -> list[str]:
    return [_fraction_text(Fraction(value)) for value in vector]


def _row_add_scaled(left: Sequence[Fraction], right: Sequence[Fraction], scale: Fraction) -> tuple[Fraction, ...]:
    return tuple(a + scale * b for a, b in zip(left, right))


def _span_coordinates(rows: MatrixQ, target: Sequence[Fraction]) -> tuple[Fraction, ...] | None:
    # The requested coefficients satisfy sum_i c_i rows[i] = target.
    return _solve_rectangular(tuple(zip(*rows)), target)


def _candidate_subsystem_pairs() -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for permutation in itertools.permutations((1, 2, 3, 4)):
        a, b, c, d = permutation
        result.append({
            "mode": "pair",
            "permutation": list(permutation),
            "h_label": _label((a, b)),
            "t_label": _label((c, d)),
            "jx_label": _label((b, c, d)),
            "jy_label": _label((a, b, d)),
            "v_label": _label((a, c)),
            "particles": list(permutation),
        })
        result.append({
            "mode": "nested",
            "permutation": list(permutation),
            "h_label": _label((a, b)),
            "t_label": _label((a, b, c)),
            "jx_label": _label((a, c)),
            "jy_label": _label((a, b, d)),
            "v_label": _label((a, d)),
            "particles": list(permutation),
        })
    return tuple(result)


def _forms_for_labels(labels: Sequence[str]) -> tuple[MatrixQ, MatrixQ]:
    forms = tuple(_subset_form(_parse_label(name)) for name in labels)
    return tuple(item[0] for item in forms), tuple(item[1] for item in forms)


def _choose_chart_basis(labels: tuple[str, ...]) -> dict[str, Any]:
    d = len(labels)
    s_matrix, s_offsets = _forms_for_labels(labels)
    if _rank(s_matrix) != d:
        raise ValueError("low-loop target IKVs are dependent in the common d=5 basis")

    candidates = _candidate_subsystem_pairs()
    for candidate in candidates:
        h_row, h_offset = _subset_form(_parse_label(candidate["h_label"]))
        t_row, t_offset = _subset_form(_parse_label(candidate["t_label"]))
        h_coordinates = _span_coordinates(s_matrix, h_row)
        t_coordinates = _span_coordinates(s_matrix, t_row)
        if h_coordinates is None or t_coordinates is None:
            continue
        first_two = (h_coordinates, t_coordinates)
        if _rank(first_two) != 2:
            continue
        chosen_u: list[int] = []
        current = list(first_two)
        for index in range(d):
            trial = current + [tuple(Fraction(int(column == index)) for column in range(d))]
            if _rank(trial) > len(current):
                chosen_u.append(index)
                current = trial
            if len(chosen_u) == d - 2:
                break
        if len(chosen_u) != d - 2:
            continue
        k_rows = [h_coordinates, t_coordinates]
        for index in chosen_u:
            k_rows.append(tuple(Fraction(int(column == index)) for column in range(d)))
        k_matrix = tuple(k_rows)
        det_k = _matrix_det(k_matrix)
        if not det_k:
            continue
        h_affine = tuple(h_offset[j] - sum(h_coordinates[i] * s_offsets[i][j] for i in range(d)) for j in range(5))
        t_affine = tuple(t_offset[j] - sum(t_coordinates[i] * s_offsets[i][j] for i in range(d)) for j in range(5))
        u_labels = [labels[index] for index in chosen_u]
        u_rows = [s_matrix[index] for index in chosen_u]
        u_offsets = [s_offsets[index] for index in chosen_u]
        z_labels = (candidate["h_label"], candidate["t_label"], candidate["jx_label"], candidate["jy_label"], candidate["v_label"])
        z_matrix, z_offsets = _forms_for_labels(z_labels)
        det_az = _matrix_det(z_matrix)
        if not det_az:
            continue
        inverse_az = _matrix_inverse(z_matrix)
        inverse_k = _matrix_inverse(k_matrix)
        identity_k = tuple(tuple(Fraction(int(i == j)) for j in range(d)) for i in range(d))
        identity_z = tuple(tuple(Fraction(int(i == j)) for j in range(5)) for i in range(5))
        if _matrix_product(k_matrix, inverse_k) != identity_k:
            raise ArithmeticError("exact K inverse certificate failed")
        if _matrix_product(z_matrix, inverse_az) != identity_z:
            raise ArithmeticError("exact A_Z inverse certificate failed")
        return {
            "candidate": candidate,
            "d": d,
            "S_matrix": s_matrix,
            "S_mass_offsets": s_offsets,
            "h_coefficients": h_coordinates,
            "t_coefficients": t_coordinates,
            "h_mass_offset": h_affine,
            "t_mass_offset": t_affine,
            "u_indices": tuple(chosen_u),
            "u_labels": tuple(u_labels),
            "U_matrix": tuple(u_rows),
            "U_mass_offsets": tuple(u_offsets),
            "K": k_matrix,
            "det_K": det_k,
            "inverse_K": inverse_k,
            "Z_matrix": z_matrix,
            "Z_mass_offsets": z_offsets,
            "A_Z": z_matrix,
            "inverse_A_Z": inverse_az,
            "D": abs(det_az),
        }
    raise ValueError(
        "no exact pair/nested subsystem mass pair lies in the target IKV row space "
        "with a nonsingular exact completion; report this card to the formula owner"
    )


def _matrix_product(left: Sequence[Sequence[Fraction]], right: Sequence[Sequence[Fraction]]) -> MatrixQ:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("matrix product dimensions do not match")
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(len(right))) for j in range(len(right[0])))
        for i in range(len(left))
    )


def _select_elimination_columns(c_base: MatrixQ, k: int) -> tuple[int, ...]:
    """Pivot from exact structural columns, before jx1/jy1 scale factors."""

    if k == 0:
        return ()
    if k == 1:
        if any(row[2] for row in c_base):
            return (2,)
        for column in (0, 1):
            if any(row[column] for row in c_base):
                return (column,)
        raise ValueError("one eliminated variable has no exact structural pivot")
    if k == 2:
        for columns in ((0, 2), (1, 2)):
            if _matrix_det(tuple(tuple(row[column] for column in columns) for row in c_base)):
                return columns
        if all(not row[2] for row in c_base):
            columns = (0, 1)
            if _matrix_det(tuple(tuple(row[column] for column in columns) for row in c_base)):
                return columns
        raise ValueError(
            "exact constraint matrix has no prescribed (x,v)/(y,v) pivot and the v column is not identically zero"
        )
    raise ValueError("low-loop charts eliminate at most two angular constraints")


def build_low_loop_chart_data_from_labels(labels: Sequence[str]) -> dict[str, Any]:
    """Build the complete exact chart plan from ordered graph labels.

    This entry point is intended for downstream formula/benchmark code that
    has only the public ``DirectFormula.target_variables`` sequence.
    """

    ordered = tuple(labels)
    d = len(ordered)
    if d not in (2, 3, 4):
        raise ValueError("low-loop direct chart requires d=2,3,4 (L=0,1,2)")
    if len(set(ordered)) != d:
        raise ValueError("low-loop target labels must be distinct")
    for label in ordered:
        _parse_label(label)
    basis = _choose_chart_basis(ordered)
    k = d - 2
    candidate = basis["candidate"]
    z_labels = (
        candidate["h_label"], candidate["t_label"], candidate["jx_label"],
        candidate["jy_label"], candidate["v_label"],
    )
    inverse_az = basis["inverse_A_Z"]
    # Express each retained original U row in the exact Z basis.
    c_base_rows: list[tuple[Fraction, ...]] = []
    u_mass_z_offsets: list[tuple[Fraction, ...]] = []
    u_in_z_coefficients: list[tuple[Fraction, ...]] = []
    for row, mass_offset in zip(basis["U_matrix"], basis["U_mass_offsets"]):
        z_coefficients = tuple(sum(row[x] * inverse_az[x][z] for x in range(5)) for z in range(5))
        mapped_offset = tuple(
            mass_offset[m] - sum(z_coefficients[z] * basis["Z_mass_offsets"][z][m] for z in range(5))
            for m in range(5)
        )
        c_base_rows.append((z_coefficients[2], z_coefficients[3], z_coefficients[4]))
        u_in_z_coefficients.append(z_coefficients)
        u_mass_z_offsets.append(mapped_offset)
    c_base: MatrixQ = tuple(c_base_rows)
    solved_columns = _select_elimination_columns(c_base, k)
    retained_columns = tuple(index for index in range(3) if index not in solved_columns)
    c_e_base = tuple(tuple(row[column] for column in solved_columns) for row in c_base)
    det_c_e_base = _matrix_det(c_e_base) if k else Fraction(1)
    if k and not det_c_e_base:
        raise ArithmeticError("selected exact elimination pivot unexpectedly has zero determinant")

    shape_names = ("x", "y", "v")
    physical_names = ("x", "y", "internal_phi")
    retained_internal = tuple(physical_names[index] for index in retained_columns)
    angle_variables = retained_internal + LOW_LOOP_EULER_VARIABLES
    jx_label, jy_label, v_label = z_labels[2:]
    geometry = {
        "mode": candidate["mode"],
        "particles": list(candidate["particles"]),
        "h_label": candidate["h_label"],
        "t_label": candidate["t_label"],
        "jx_label": jx_label,
        "jy_label": jy_label,
        "v_label": v_label,
        "variables": ["x", "y", "internal_phi"],
        "coordinate_definitions": {
            "x": "cosine of the specified first daughter direction relative to the local +z axis",
            "y": "cosine of the specified second daughter direction relative to the local +z axis",
            "internal_phi": "oriented relative azimuth of the two decay planes, in (0,2*pi), excluding pi",
        },
    }
    internal_coordinate_definition = {
        "mode": candidate["mode"],
        "particles_abcd": list(candidate["particles"]),
        "h_label": candidate["h_label"],
        "t_label": candidate["t_label"],
        "jx_label": candidate["jx_label"],
        "jy_label": candidate["jy_label"],
        "v_label": candidate["v_label"],
        "x": geometry["coordinate_definitions"]["x"],
        "y": geometry["coordinate_definitions"]["y"],
        "internal_phi": geometry["coordinate_definitions"]["internal_phi"],
        "phi_eliminated": 2 in solved_columns,
        "branch_semantics": "branch=sign(sin(internal_phi)) when internal_phi is eliminated; otherwise only branch=+1 is allowed and the retained internal_phi selects the mirror",
    }
    affine_rows = []
    for i, label in enumerate(ordered):
        affine_rows.append({
            "label": label,
            "X_coefficients": _json_vector(basis["S_matrix"][i]),
            "mass_offset_coefficients": {symbol: _fraction_text(basis["S_mass_offsets"][i][j]) for j, symbol in enumerate(_MASS_SYMBOLS)},
        })
    h_t_u = {
        "h": {
            "label": candidate["h_label"],
            "S_coefficients": _json_vector(basis["h_coefficients"]),
            "mass_offset_coefficients": {symbol: _fraction_text(value) for symbol, value in zip(_MASS_SYMBOLS, basis["h_mass_offset"])},
        },
        "t": {
            "label": candidate["t_label"],
            "S_coefficients": _json_vector(basis["t_coefficients"]),
            "mass_offset_coefficients": {symbol: _fraction_text(value) for symbol, value in zip(_MASS_SYMBOLS, basis["t_mass_offset"])},
        },
        "U": [{
            "label": name,
            "source_index": index,
            "S_coefficients": _json_vector(tuple(Fraction(int(j == index)) for j in range(d))),
            "mass_offset_coefficients": {symbol: "0" for symbol in _MASS_SYMBOLS},
        } for index, name in zip(basis["u_indices"], basis["u_labels"])],
    }
    z_form_records = []
    for name, row, offset in zip(z_labels, basis["Z_matrix"], basis["Z_mass_offsets"]):
        z_form_records.append({
            "label": name,
            "X_coefficients": _json_vector(row),
            "mass_offset_coefficients": {symbol: _fraction_text(value) for symbol, value in zip(_MASS_SYMBOLS, offset)},
        })
    return {
        "profile": "direct_gram_low_loop",
        "status": "built",
        "loop_count": d - 2,
        "dimension": d,
        "target_variables": list(ordered),
        "basis_variables": list(D5_BASIS),
        "internal_variables": list(retained_internal),
        "angle_variables": list(angle_variables),
        "euler_variables": list(LOW_LOOP_EULER_VARIABLES),
        "S_affine_map": affine_rows,
        "S_matrix": _json_matrix(basis["S_matrix"]),
        "S_mass_offset_coefficients": [
            {symbol: _fraction_text(value) for symbol, value in zip(_MASS_SYMBOLS, row)}
            for row in basis["S_mass_offsets"]
        ],
        "h_t_U": h_t_u,
        "h": h_t_u["h"],
        "t": h_t_u["t"],
        "U": h_t_u["U"],
        "K": _json_matrix(basis["K"]),
        "K_latex": "K=" + _matrix_latex(basis["K"]),
        "inverse_K": _json_matrix(basis["inverse_K"]),
        "inverse_K_latex": "K^{-1}=" + _matrix_latex(basis["inverse_K"]),
        "det_K": _fraction_text(basis["det_K"]),
        "abs_det_K": _fraction_text(abs(basis["det_K"])),
        "B_variables": [candidate["h_label"], candidate["t_label"], *basis["u_labels"]],
        "B_affine_map": "B=K*S+b at fixed (M^2,m_1^2,...,m_4^2); the first rows define h,t and remaining rows are selected original U variables",
        "B_mass_offset_coefficients": [
            {symbol: _fraction_text(value) for symbol, value in zip(_MASS_SYMBOLS, row)}
            for row in (basis["h_mass_offset"], basis["t_mass_offset"], *((Fraction(0),) * 5 for _ in basis["U_matrix"]))
        ],
        "Z_variables": list(z_labels),
        "Z_forms": z_form_records,
        "A_Z": _json_matrix(basis["A_Z"]),
        "A_Z_latex": "A_Z=" + _matrix_latex(basis["A_Z"]),
        "inverse_A_Z": _json_matrix(inverse_az),
        "inverse_A_Z_latex": "A_Z^{-1}=" + _matrix_latex(inverse_az),
        "Z_mass_offset_coefficients": [
            {symbol: _fraction_text(value) for symbol, value in zip(_MASS_SYMBOLS, row)}
            for row in basis["Z_mass_offsets"]
        ],
        "det_A_Z": _fraction_text(_matrix_det(basis["A_Z"])),
        "D": _fraction_text(basis["D"]),
        "abs_det_A_Z": _fraction_text(basis["D"]),
        "determinant_certificate": {
            "method": "exact Fraction Gaussian elimination",
            "K": _fraction_text(basis["det_K"]),
            "A_Z": _fraction_text(_matrix_det(basis["A_Z"])),
            "inverse_A_Z": _json_matrix(inverse_az),
            "K_inverse_product": _json_matrix(_matrix_product(basis["K"], basis["inverse_K"])),
            "A_Z_inverse_product": _json_matrix(_matrix_product(basis["A_Z"], inverse_az)),
            "identity_check": "passed exactly: K*K_inverse=I and A_Z*inverse_A_Z=I",
        },
        "verification": {
            "affine_rank": "passed exactly by Fraction row reduction",
            "B_matrix_inverse": "exact inverse constructed; det_K is nonzero",
            "Z_matrix_inverse": "exact inverse constructed; det_A_Z is nonzero",
            "constraint_pivot": "selected from exact structural rational columns before jx1/jy1 factors",
            "numeric_event": "pending; chart construction does not imply an evaluated physical point",
            "independent_formula_check": "pending",
        },
        "geometry": geometry,
        "angular_expansion": _angular_expansion_records(candidate),
        "angular_expansion_latex": _angular_expansion_latex(candidate),
        "internal_coordinate_definition": internal_coordinate_definition,
        "U_in_Z_coefficients": _json_matrix(tuple(u_in_z_coefficients)),
        "constraint_matrix_base": _json_matrix(c_base),
        "constraint_matrix_base_latex": "C_{base}=" + _matrix_latex(c_base),
        "C": [
            [
                f"{_fraction_text(row[0])}*jx1",
                f"{_fraction_text(row[1])}*jy1",
                _fraction_text(row[2]),
            ]
            for row in c_base
        ],
        "C_latex": _constraint_matrix_latex(c_base),
        "constraint_mass_offsets": [
            {symbol: _fraction_text(value) for symbol, value in zip(_MASS_SYMBOLS, row)}
            for row in u_mass_z_offsets
        ],
        "constraint_equations": [{
            "U_label": name,
            "U_in_Z_coefficients": _json_vector(coefficients),
            "mass_offset_coefficients": {symbol: _fraction_text(value) for symbol, value in zip(_MASS_SYMBOLS, offset)},
            "u0_definition": "q_h*h+q_t*t+q_Jx*jx0+q_Jy*jy0+sum_m q_mass[m]*mass_squared[m]",
            "C_equation": [
                f"{_fraction_text(base_row[0])}*jx1*x",
                f"{_fraction_text(base_row[1])}*jy1*y",
                f"{_fraction_text(base_row[2])}*v",
            ],
            "equation": f"{name}-u0=C_row*(x,y,v)^T",
            "equation_latex": _constraint_equation_latex(name, coefficients, offset, base_row),
            "C_equation_latex": _constraint_rhs_latex(base_row),
            "u0_latex": _u0_latex(coefficients, offset),
        } for name, coefficients, offset, base_row in zip(basis["u_labels"], u_in_z_coefficients, u_mass_z_offsets, c_base)],
        "C_base_columns": ["Jx coefficient", "Jy coefficient", "v coefficient"],
        "solved_columns": [shape_names[index] for index in solved_columns],
        "solved_column_indices": list(solved_columns),
        "retained_column_indices": list(retained_columns),
        "C_E_base": _json_matrix(c_e_base),
        "C_E_base_latex": "C_{E,base}=" + _matrix_latex(c_e_base),
        "C_E": [
            [
                (f"{_fraction_text(row[column])}*" + ("jx1" if column == 0 else "jy1")
                 if column in (0, 1) else _fraction_text(row[column]))
                for column in solved_columns
            ]
            for row in c_base
        ],
        "C_E_latex": _constraint_elimination_matrix_latex(c_base, solved_columns),
        "det_C_E_base": _fraction_text(det_c_e_base),
        "det_C_E_scale": {
            "jx1_power": sum(column == 0 for column in solved_columns),
            "jy1_power": sum(column == 1 for column in solved_columns),
            "expression": (
                _fraction_text(det_c_e_base)
                + ("*jx1" if sum(column == 0 for column in solved_columns) == 1 else "*jx1^2" if sum(column == 0 for column in solved_columns) == 2 else "")
                + ("*jy1" if sum(column == 1 for column in solved_columns) == 1 else "*jy1^2" if sum(column == 1 for column in solved_columns) == 2 else "")
            ),
        },
        "elimination": {
            "constraint_equation": "U-u0=C*(x,y,v)^T",
            "pivot_policy": "exact rational structural coefficients before multiplying the nonzero jx1/jy1 scale factors",
            "retained_internal_variables": list(retained_internal),
            "solved_shape_variables": [shape_names[index] for index in solved_columns],
            "det_C_E_base": _fraction_text(det_c_e_base),
            "phi_roots": (
                "internal_phi=acos(cos_internal_phi) for branch=+1; "
                "internal_phi=2*pi-acos(cos_internal_phi) for branch=-1; "
                "cos_internal_phi=(v-v0-vx*x-vy*y-vxy*x*y)/(vw*sqrt(1-x^2)*sqrt(1-y^2))"
                if 2 in solved_columns else "internal_phi is retained as a coordinate; its 0..2*pi value already selects the mirror"
            ),
            "phi_roots_latex": _phi_roots_latex(2 in solved_columns),
            "retained_phi_input_latex": (
                r"\varphi\text{ is an explicit coordinate on }(0,2\pi),\quad{}\varphi\ne\pi"
                if "internal_phi" in retained_internal else None
            ),
            "phi_eliminated": 2 in solved_columns,
        },
        "jacobian": {
            "rho_X": "chi_open/(512*(2*pi)^p*M^3*sqrt(det(H(X))))",
            "p_by_normalization": {"jing_psf": 12, "standard_2pi4": 8},
            "pullback": "abs(jx1*jy1)*abs(det_K)/(D*abs(det_C_E))*W",
            "W_when_internal_phi_eliminated": "1",
            "W_when_internal_phi_retained": "abs(vw)*sqrt(1-x^2)*sqrt(1-y^2)*abs(sin(internal_phi))",
            "full_density": "rho_X*abs(jx1*jy1)*abs(det_K)/(D*abs(det_C_E))*W",
            "no_extra_S_jacobian": True,
            "empty_determinant": "1",
            "rho_X_latex": r"\rho_X=\frac{\chi_{\rm open}}{512(2\pi)^pM^3\sqrt{\det H}}",
            "pullback_latex": r"J_{\rm pull}=\frac{|j_{x1}j_{y1}\det K|}{D|\det C_E|}W",
            "full_density_latex": r"\rho_{\rm low}=\frac{\chi_{\rm open}|j_{x1}j_{y1}\det K|W}{512(2\pi)^pM^3\sqrt{\det H}\,D|\det C_E|}",
        },
        "density": {
            "rho_X": "chi_open/(512*(2*pi)^p*M^3*sqrt(det(H(X))))",
            "p_by_normalization": {"jing_psf": 12, "standard_2pi4": 8},
            "pullback_factors": ["abs(jx1*jy1)", "abs(det_K)", "1/D", "1/abs(det_C_E)", "W"],
            "pullback": "abs(jx1*jy1)*abs(det_K)/(D*abs(det_C_E))*W",
            "W_phi_eliminated": "1",
            "W_phi_retained": "abs(vw)*sqrt(1-x^2)*sqrt(1-y^2)*abs(sin(internal_phi))",
            "expression_by_normalization": {
                "jing_psf": "chi_open*abs(jx1*jy1*det_K)*W/(512*(2*pi)^12*M^3*sqrt(det(H))*D*abs(det_C_E))",
                "standard_2pi4": "chi_open*abs(jx1*jy1*det_K)*W/(512*(2*pi)^8*M^3*sqrt(det(H))*D*abs(det_C_E))",
            },
            "expression_by_normalization_latex": {
                "jing_psf": _density_latex(12, 1),
                "standard_2pi4": _density_latex(8, 1),
            },
            "expression_by_normalization_summed_latex": {
                "jing_psf": _density_latex(12, 2 if 2 in solved_columns else 1),
                "standard_2pi4": _density_latex(8, 2 if 2 in solved_columns else 1),
            },
            "W_definition_latex": (
                r"W=1\quad(\varphi\text{ eliminated})"
                if 2 in solved_columns else
                r"W=|v_w|\sqrt{1-x^2}\sqrt{1-y^2}|\sin\varphi|\quad(\varphi\text{ retained})"
            ),
            "summed_multiplier": 2 if 2 in solved_columns else 1,
            "summed_density_note": (
                "two distinct internal_phi roots are summed"
                if 2 in solved_columns else
                "internal_phi is an explicit 0..2*pi coordinate; no duplicate mirror multiplier is applied"
            ),
        },
        "density_expression_latex": {
            "jing_psf_per_branch": _density_latex(12, 1),
            "standard_2pi4_per_branch": _density_latex(8, 1),
            "jing_psf_summed": _density_latex(12, 2 if 2 in solved_columns else 1),
            "standard_2pi4_summed": _density_latex(8, 2 if 2 in solved_columns else 1),
        },
        "gram_formula": {
            "X_from_Z": "X=inverse_A_Z*(Z-b_Z) with Z=(h,t,Jx,Jy,v); exact affine offsets are in Z_mass_offset_coefficients",
            "energies": [
                "E1=(M^2+m1^2-s234)/(2M)",
                "E2=(s12+s234-s34-m1^2)/(2M)",
                "E3=(s13+s234-s24-m1^2)/(2M)",
                "E4=M-E1-E2-E3",
            ],
            "pair_completion": "s23=s234-s24-s34+m2^2+m3^2+m4^2",
            "H_ii": "E_i^2-m_i^2, i=1,2,3",
            "H_ij": "E_i*E_j-(s_ij-m_i^2-m_j^2)/2, i<j in {12,13,23}",
            "strict_support": ["E_i>0 for every i", "H11>0", "H11*H22-H12^2>0", "det(H)>0"],
        },
        "gram_formula_latex": _gram_formula_latex(),
        "momenta": _momentum_formula_records(candidate),
        "domain": _domain_records(retained_internal, 2 in solved_columns),
        "domain_latex": _domain_latex(retained_internal, 2 in solved_columns),
        "angle_measure": "dR=sin(theta)dtheta dphi dpsi; internal cosines use dx and dy, not polar-angle measures",
        "angle_measure_latex": r"dR=\sin\theta\,d\theta\,d\phi\,d\psi",
        "differential_measure": "dPhi4=rho_low*d^dS*prod(d retained internal coordinates)*dR; all internal delta constraints are eliminated",
        "differential_measure_latex": _differential_measure_latex(retained_internal),
        "euler_definitions": [
            "The canonical body frame fixes p1 along +z and p2 in xz with px>0.",
            "Parent-rest momenta are actively rotated by Rz(phi) Ry(theta) Rz(psi).",
            "0<theta<pi, 0<=phi<2*pi, 0<=psi<2*pi; dR=sin(theta)dtheta dphi dpsi.",
        ],
        "diagnostics": ["status=built; no independent formula verification is implied"],
    }


def _momentum_formula_records(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate["mode"] == "pair":
        equations = [
            "H=sqrt(h), T=sqrt(t), q=sqrt(lambda(M^2,h,t))/(2M)",
            "EA=(M^2+h-t)/(2M), EB=(M^2+t-h)/(2M)",
            "ea=(h+ma^2-mb^2)/(2H), ka=sqrt(ea^2-ma^2); ec=(t+mc^2-md^2)/(2T), kc=sqrt(ec^2-mc^2)",
            "pa=((EA*ea+q*ka*x)/H, ka*sqrt(1-x^2), 0, (q*ea+EA*ka*x)/H); pb=(EA,0,0,q)-pa",
            "pc=((EB*ec-q*kc*y)/T, kc*sqrt(1-y^2)*cos(internal_phi), kc*sqrt(1-y^2)*sin(internal_phi), (-q*ec+EB*kc*y)/T); pd=(EB,0,0,-q)-pc",
        ]
        equations_latex = [
            r"H=\sqrt{h},\quad T=\sqrt{t},\quad q=\frac{\sqrt{\lambda(M^2,h,t)}}{2M}",
            r"E_A=\frac{M^2+h-t}{2M},\quad E_B=\frac{M^2+t-h}{2M}",
            r"e_a=\frac{h+m_a^2-m_b^2}{2H},\quad k_a=\sqrt{e_a^2-m_a^2},\quad e_c=\frac{t+m_c^2-m_d^2}{2T},\quad k_c=\sqrt{e_c^2-m_c^2}",
            r"p_a=\left(\frac{E_Ae_a+qk_ax}{H},\ k_a\sqrt{1-x^2},\ 0,\ \frac{qe_a+E_Ak_ax}{H}\right)",
            r"p_b=(E_A,0,0,q)-p_a",
            r"p_c=\left(\frac{E_Be_c-qk_cy}{T},\ k_c\sqrt{1-y^2}\cos\varphi,\ k_c\sqrt{1-y^2}\sin\varphi,\ \frac{-qe_c+E_Bk_cy}{T}\right)",
            r"p_d=(E_B,0,0,-q)-p_c",
        ]
        return {"mode": "pair", "particle_order": list(candidate["particles"]), "equations": equations, "equations_latex": equations_latex}
    equations = [
        "H=sqrt(h), T=sqrt(t), k=sqrt(lambda(t,h,mc^2))/(2T)",
        "Eq=(t+h-mc^2)/(2T), Ec=(t+mc^2-h)/(2T), Ed=(M^2-t-md^2)/(2T), kd=sqrt(Ed^2-md^2)",
        "ea=(h+ma^2-mb^2)/(2H), ka=sqrt(ea^2-ma^2)",
        "pa=((Eq*ea-k*ka*x)/H, ka*sqrt(1-x^2), 0, (Eq*ka*x-k*ea)/H); pb=(Eq,0,0,-k)-pa; pc=(Ec,0,0,k)",
        "pd=(Ed, kd*sqrt(1-y^2)*cos(internal_phi), kd*sqrt(1-y^2)*sin(internal_phi), kd*y)",
        "Boost all abc-system four-vectors by beta=-pd_spatial/(T+Ed) into the parent rest frame.",
    ]
    equations_latex = [
        r"H=\sqrt{h},\quad T=\sqrt{t},\quad k=\frac{\sqrt{\lambda(t,h,m_c^2)}}{2T}",
        r"E_q=\frac{t+h-m_c^2}{2T},\quad E_c=\frac{t+m_c^2-h}{2T},\quad E_d=\frac{M^2-t-m_d^2}{2T},\quad k_d=\sqrt{E_d^2-m_d^2}",
        r"e_a=\frac{h+m_a^2-m_b^2}{2H},\quad k_a=\sqrt{e_a^2-m_a^2}",
        r"p_a^{(abc)}=\left(\frac{E_qe_a-k k_ax}{H},\ k_a\sqrt{1-x^2},\ 0,\ \frac{E_qk_ax-ke_a}{H}\right)",
        r"p_b^{(abc)}=(E_q,0,0,-k)-p_a^{(abc)},\quad p_c^{(abc)}=(E_c,0,0,k)",
        r"p_d^{(abc)}=(E_d,k_d\sqrt{1-y^2}\cos\varphi,k_d\sqrt{1-y^2}\sin\varphi,k_dy)",
        r"\beta=-\frac{\mathbf p_d^{(abc)}}{T+E_d},\quad p_i^{(P)}=\operatorname{Boost}_{\beta}(p_i^{(abc)})\quad(i=a,b,c,d)",
    ]
    return {"mode": "nested", "particle_order": list(candidate["particles"]), "equations": equations, "equations_latex": equations_latex}


def _gram_formula_latex() -> dict[str, Any]:
    return {
        "energies": [
            r"E_1=\frac{M^2+m_1^2-s_{234}}{2M}",
            r"E_2=\frac{s_{12}+s_{234}-s_{34}-m_1^2}{2M}",
            r"E_3=\frac{s_{13}+s_{234}-s_{24}-m_1^2}{2M}",
            r"E_4=M-E_1-E_2-E_3",
        ],
        "pair_completion": r"s_{23}=s_{234}-s_{24}-s_{34}+m_2^2+m_3^2+m_4^2",
        "H": r"H_{ii}=E_i^2-m_i^2,\quad{}H_{ij}=E_iE_j-\frac{s_{ij}-m_i^2-m_j^2}{2}\quad(i<j\in\{12,13,23\})",
        "strict_support": r"E_i>0,\quad{}H_{11}>0,\quad{}H_{11}H_{22}-H_{12}^2>0,\quad{}\det H>0",
        "X_from_Z": r"X=A_Z^{-1}(Z-b_Z),\quad{}Z=(h,t,J_x,J_y,v)",
    }


def _domain_latex(retained: Sequence[str], phi_eliminated: bool) -> list[str]:
    domain = [
        r"m_i>0\ (i=1,2,3,4),\quad{}M>\sum_{i=1}^4m_i",
        r"E_i>0\ (i=1,2,3,4),\quad{}H_{11}>0,\quad{}H_{11}H_{22}-H_{12}^2>0,\quad{}\det H>0",
        r"|x|<1,\quad{}|y|<1,\quad{}j_{x1}j_{y1}\ne0,\quad{}\det C_E\ne0",
        r"0<\theta<\pi,\quad{}0\le\phi<2\pi,\quad{}0\le\psi<2\pi",
    ]
    if phi_eliminated:
        domain.append(r"|c_\varphi|<1,\quad{}\varphi_{\pm}\in(0,2\pi)\setminus\{\pi\}")
    else:
        domain.append(r"0<\varphi<2\pi,\quad{}\varphi\ne\pi")
    domain.append(r"\lambda>0\text{ for every selected strict two-body subsystem decay}")
    return domain


def _angular_expansion_records(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate["mode"] == "pair":
        equations = {
            "Jx": "jx0+jx1*x",
            "Jy": "jy0+jy1*y",
            "v": "v0+vx*x+vy*y+vxy*x*y+vw*sqrt(1-x^2)*sqrt(1-y^2)*cos(internal_phi)",
            "coefficients": {
                "jx0": "M^2+ma^2-2*M*EA*ea/H",
                "jx1": "-2*M*q*ka/H",
                "jy0": "M^2+mc^2-2*M*EB*ec/T",
                "jy1": "+2*M*q*kc/T",
                "v0": "ma^2+mc^2+2*ea*ec*(EA*EB+q^2)/(H*T)",
                "vx": "2*ka*ec*q*M/(H*T)",
                "vy": "-2*ea*kc*q*M/(H*T)",
                "vxy": "-2*ka*kc*(EA*EB+q^2)/(H*T)",
                "vw": "-2*ka*kc",
            },
            "auxiliary": [
                "H=sqrt(h), T=sqrt(t), q=sqrt(lambda(M^2,h,t))/(2*M)",
                "EA=(M^2+h-t)/(2*M), EB=(M^2+t-h)/(2*M)",
                "ea=(h+ma^2-mb^2)/(2*H), ka=sqrt(ea^2-ma^2)",
                "ec=(t+mc^2-md^2)/(2*T), kc=sqrt(ec^2-mc^2)",
            ],
        }
    else:
        equations = {
            "Jx": "jx0+jx1*x",
            "Jy": "jy0+jy1*y",
            "v": "v0+vx*x+vy*y+vxy*x*y+vw*sqrt(1-x^2)*sqrt(1-y^2)*cos(internal_phi)",
            "coefficients": {
                "jx0": "ma^2+mc^2+2*ea*(Eq*Ec+k^2)/H",
                "jx1": "-2*ka*k*T/H",
                "jy0": "h+md^2+2*Eq*Ed",
                "jy1": "+2*k*kd",
                "v0": "ma^2+md^2+2*Eq*ea*Ed/H",
                "vx": "-2*k*ka*Ed/H",
                "vy": "2*k*ea*kd/H",
                "vxy": "-2*Eq*ka*kd/H",
                "vw": "-2*ka*kd",
            },
            "auxiliary": [
                "H=sqrt(h), T=sqrt(t), k=sqrt(lambda(t,h,mc^2))/(2*T)",
                "Eq=(t+h-mc^2)/(2*T), Ec=(t+mc^2-h)/(2*T)",
                "Ed=(M^2-t-md^2)/(2*T), kd=sqrt(Ed^2-md^2)",
                "ea=(h+ma^2-mb^2)/(2*H), ka=sqrt(ea^2-ma^2)",
            ],
        }
    equations.update({
        "selected_labels": {
            "h": candidate["h_label"], "t": candidate["t_label"],
            "Jx": candidate["jx_label"], "Jy": candidate["jy_label"], "v": candidate["v_label"],
        },
        "particle_order_abcd": list(candidate["particles"]),
    })
    return equations


def _angular_expansion_latex(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate["mode"] == "pair":
        coefficients = (
            r"j_{x0}=M^2+m_a^2-\frac{2ME_Ae_a}{H}",
            r"j_{x1}=-\frac{2Mqk_a}{H}",
            r"j_{y0}=M^2+m_c^2-\frac{2ME_Be_c}{T}",
            r"j_{y1}=\frac{2Mqk_c}{T}",
            r"v_0=m_a^2+m_c^2+\frac{2e_ae_c(E_AE_B+q^2)}{HT}",
            r"v_x=\frac{2k_ae_cqM}{HT}",
            r"v_y=-\frac{2e_ak_cqM}{HT}",
            r"v_{xy}=-\frac{2k_ak_c(E_AE_B+q^2)}{HT}",
            r"v_w=-2k_ak_c",
        )
        auxiliaries = (
            r"H=\sqrt{h},\quad T=\sqrt{t},\quad q=\frac{\sqrt{\lambda(M^2,h,t)}}{2M}",
            r"E_A=\frac{M^2+h-t}{2M},\quad E_B=\frac{M^2+t-h}{2M}",
            r"e_a=\frac{h+m_a^2-m_b^2}{2H},\quad k_a=\sqrt{e_a^2-m_a^2}",
            r"e_c=\frac{t+m_c^2-m_d^2}{2T},\quad k_c=\sqrt{e_c^2-m_c^2}",
        )
    else:
        coefficients = (
            r"j_{x0}=m_a^2+m_c^2+\frac{2e_a(E_qE_c+k^2)}{H}",
            r"j_{x1}=-\frac{2k_akT}{H}",
            r"j_{y0}=h+m_d^2+2E_qE_d",
            r"j_{y1}=2kk_d",
            r"v_0=m_a^2+m_d^2+\frac{2E_qe_aE_d}{H}",
            r"v_x=-\frac{2kk_aE_d}{H}",
            r"v_y=\frac{2ke_ak_d}{H}",
            r"v_{xy}=-\frac{2E_qk_ak_d}{H}",
            r"v_w=-2k_ak_d",
        )
        auxiliaries = (
            r"H=\sqrt{h},\quad T=\sqrt{t},\quad k=\frac{\sqrt{\lambda(t,h,m_c^2)}}{2T}",
            r"E_q=\frac{t+h-m_c^2}{2T},\quad E_c=\frac{t+m_c^2-h}{2T}",
            r"E_d=\frac{M^2-t-m_d^2}{2T},\quad k_d=\sqrt{E_d^2-m_d^2}",
            r"e_a=\frac{h+m_a^2-m_b^2}{2H},\quad k_a=\sqrt{e_a^2-m_a^2}",
        )
    return {
        "Jx": r"J_x=j_{x0}+j_{x1}x",
        "Jy": r"J_y=j_{y0}+j_{y1}y",
        "v": r"v=v_0+v_xx+v_yy+v_{xy}xy+v_w\sqrt{1-x^2}\sqrt{1-y^2}\cos\varphi",
        "coefficients": list(coefficients),
        "auxiliary": list(auxiliaries),
        "selected_subsystems": f"h={candidate['h_label']},\\quad t={candidate['t_label']},\\quad J_x={candidate['jx_label']},\\quad J_y={candidate['jy_label']},\\quad v={candidate['v_label']}",
    }


def _constraint_rhs_latex(row: Sequence[Fraction]) -> str:
    return _signed_latex_sum(tuple(zip(row, (r"j_{x1}x", r"j_{y1}y", "v"))))


def _constraint_equation_latex(
    name: str, coefficients: Sequence[Fraction], offset: Sequence[Fraction], row: Sequence[Fraction]
) -> str:
    del coefficients, offset
    return rf"{name}-u_0={_constraint_rhs_latex(row)}"


def _u0_latex(coefficients: Sequence[Fraction], offset: Sequence[Fraction]) -> str:
    terms = (
        *zip(coefficients[:4], ("h", "t", r"j_{x0}", r"j_{y0}")),
        *zip(offset, ("M^2", "m_1^2", "m_2^2", "m_3^2", "m_4^2")),
    )
    return _signed_latex_sum(terms)


def _monomial_latex(value: Fraction, symbol: str = "") -> str:
    """Render one signed coefficient times a symbol without redundant unit factors."""
    if not value:
        return "0"
    if not symbol:
        return _fraction_latex(value)
    magnitude = abs(value)
    coefficient = "" if magnitude == 1 else _fraction_latex(magnitude)
    return ("-" if value < 0 else "") + coefficient + symbol


def _signed_latex_sum(terms: Sequence[tuple[Fraction, str]]) -> str:
    pieces: list[str] = []
    for value, symbol in terms:
        if not value:
            continue
        magnitude = _monomial_latex(abs(value), symbol)
        if not pieces:
            pieces.append(("-" if value < 0 else "") + magnitude)
        else:
            pieces.append(("-" if value < 0 else "+") + magnitude)
    return "".join(pieces) if pieces else "0"


def _constraint_matrix_latex(matrix: Sequence[Sequence[Fraction]]) -> str:
    rows = r" \\ ".join(
        " & ".join(
            _monomial_latex(row[column], r"j_{x1}" if column == 0 else r"j_{y1}" if column == 1 else "")
            for column in range(3)
        )
        for row in matrix
    )
    return r"C=\begin{pmatrix}" + rows + r"\end{pmatrix}"


def _constraint_elimination_matrix_latex(matrix: Sequence[Sequence[Fraction]], columns: Sequence[int]) -> str:
    symbols = (r"j_{x1}", r"j_{y1}", "")
    rows = r" \\ ".join(
        " & ".join(_monomial_latex(row[column], symbols[column]) for column in columns)
        for row in matrix
    )
    return r"C_E=\begin{pmatrix}" + rows + r"\end{pmatrix}"


def _phi_roots_latex(phi_eliminated: bool) -> str:
    if not phi_eliminated:
        return r"\varphi\text{ is retained on }(0,2\pi),\quad{}\varphi\ne\pi"
    return (
        r"c_\varphi=\frac{v-v_0-v_xx-v_yy-v_{xy}xy}{v_w\sqrt{1-x^2}\sqrt{1-y^2}},\quad{}"
        r"\varphi_{+}=\arccos(c_\varphi),\quad{}\varphi_{-}=2\pi-\arccos(c_\varphi),\quad{}|c_\varphi|<1"
    )


def _density_latex(power: int, multiplier: int) -> str:
    prefix = "2\\," if multiplier == 2 else ""
    return (
        prefix
        + r"\frac{\chi_{\rm open}|j_{x1}j_{y1}\det K|W}"
        + rf"{{512(2\pi)^{{{power}}}M^3\sqrt{{\det H}}D|\det C_E|}}"
    )


def _differential_measure_latex(retained: Sequence[str]) -> str:
    differentials = {"x": r"dx", "y": r"dy", "internal_phi": r"d\varphi"}
    internal = " ".join(differentials[name] for name in retained)
    middle = (internal + r"\,") if internal else ""
    return (
        r"d\Phi_4=\rho_{\rm low}\,d^dS\," + middle
        + r"dR,\quad{}dR=\sin\theta\,d\theta\,d\phi\,d\psi"
    )


def _domain_records(retained: Sequence[str], phi_eliminated: bool) -> list[str]:
    domain = [
        "m_i>0 for i=1,2,3,4; M>sum_i m_i (strict threshold)",
        "all reconstructed particle energies are strictly positive",
        "strict subsystem two-body interiors for the selected pair/nested mode; all Kallen square roots are positive",
        "|x|<1 and |y|<1; angular endpoints are excluded",
        "jx1!=0, jy1!=0, and det(C_E)!=0",
        "H11>0; H11*H22-H12^2>0; det(H)>0",
        "0<theta<pi; 0<=phi<2*pi; 0<=psi<2*pi",
    ]
    if phi_eliminated:
        domain += ["|cos(internal_phi)|<1; both acos mirror roots are strict interior roots"]
    else:
        domain += ["0<internal_phi<2*pi and internal_phi!=pi"]
    return domain


def build_low_loop_chart_data(card: FormulaInputCard) -> dict[str, Any]:
    """Validate and build a low-loop chart from a ready formula-input card."""

    if not isinstance(card, FormulaInputCard):
        raise TypeError("build_low_loop_chart_data expects a FormulaInputCard")
    if card.availability != "ready":
        raise ValueError(card.unavailable_reason or "formula input is unavailable")
    validate_formula_card(card)
    if card.n != 4 or card.loop_count not in (0, 1, 2):
        raise ValueError("low-loop direct chart supports only four-body L=0,1,2 cards")
    if len(card.independent_variables) != 2 + card.loop_count:
        raise ValueError("low-loop card must have d=2+L independent invariant variables")
    expected_labels = tuple(_label(item) for item in card.independent_variables)
    if tuple(card.variable_labels) != expected_labels:
        raise ValueError("low-loop card variable labels disagree with its ordered subsets")
    data = build_low_loop_chart_data_from_labels(card.variable_labels)
    data["measure_convention"] = card.measure_convention
    return data


def _mass_squares(masses: Sequence[float], total_mass: float) -> tuple[float, ...]:
    return (total_mass * total_mass, *(mass * mass for mass in masses))


def _fraction(value: str | Fraction | int) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _float_matrix(matrix: Sequence[Sequence[str | Fraction]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(_fraction(value)) for value in row) for row in matrix)


def _offset_values(rows: Sequence[Mapping[str, str]], masses: Sequence[float], total_mass: float) -> tuple[float, ...]:
    squares = _mass_squares(masses, total_mass)
    return tuple(
        sum(float(_fraction(row[symbol])) * squares[index] for symbol, index in _MASS_SYMBOL_INDEX.items())
        for row in rows
    )


def _finite(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _read_chart_values(
    chart_data: Mapping[str, Any], values: Mapping[str, float] | Sequence[float]
) -> dict[str, float]:
    names = tuple(chart_data["target_variables"]) + tuple(chart_data["angle_variables"])
    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        extra = [name for name in values if name not in names]
        if missing:
            raise ValueError(f"missing low-loop chart coordinates: {missing}")
        if extra:
            raise ValueError(f"unexpected low-loop chart coordinates outside this profile scope: {extra}")
        return {name: _finite(name, values[name]) for name in names}
    try:
        sequence = tuple(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("low-loop chart coordinates must be a mapping or finite sequence") from exc
    if len(sequence) != len(names):
        raise ValueError(f"expected {len(names)} low-loop chart coordinates, got {len(sequence)}")
    return {name: _finite(name, value) for name, value in zip(names, sequence)}


def _as_chart_data(card_or_data: FormulaInputCard | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(card_or_data, FormulaInputCard):
        return build_low_loop_chart_data(card_or_data)
    if not isinstance(card_or_data, Mapping) or card_or_data.get("profile") != "direct_gram_low_loop":
        raise TypeError("expected a FormulaInputCard or direct low-loop chart-data mapping")
    return dict(card_or_data)


def _strict_sqrt(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be strictly positive in the open low-loop domain")
    return math.sqrt(value)


def _two_body_momentum(parent: float, first: float, second: float, name: str) -> float:
    if not all(math.isfinite(item) for item in (parent, first, second)):
        raise ValueError(f"{name} masses must be finite")
    if parent <= first + second:
        raise ValueError(f"{name} threshold boundary or subthreshold point is rejected")
    # Factorized Kallen form avoids accepting a negative value by tolerance.
    factors = (
        parent - first - second,
        parent + first + second,
        parent - first + second,
        parent + first - second,
    )
    product = math.prod(factors)
    return _strict_sqrt(f"{name} Kallen function", product) / (2.0 * parent)


def _geometry_coefficients(
    chart_data: Mapping[str, Any], h: float, t: float, masses: Sequence[float], total_mass: float
) -> dict[str, float]:
    mode = chart_data["geometry"]["mode"]
    a, b, c, d = chart_data["geometry"]["particles"]
    ma, mb, mc, md = (masses[index - 1] for index in (a, b, c, d))
    if h <= 0.0 or t <= 0.0:
        raise ValueError("selected subsystem invariant masses must be strictly positive")
    H, T = math.sqrt(h), math.sqrt(t)
    result: dict[str, float] = {"H": H, "T": T}
    if mode == "pair":
        q = _two_body_momentum(total_mass, H, T, "parent-to-pair")
        EA = (total_mass * total_mass + h - t) / (2.0 * total_mass)
        EB = (total_mass * total_mass + t - h) / (2.0 * total_mass)
        ea = (h + ma * ma - mb * mb) / (2.0 * H)
        ec = (t + mc * mc - md * md) / (2.0 * T)
        ka = _two_body_momentum(H, ma, mb, "ab pair")
        kc = _two_body_momentum(T, mc, md, "cd pair")
        jx0 = total_mass * total_mass + ma * ma - 2.0 * total_mass * EA * ea / H
        jx1 = -2.0 * total_mass * q * ka / H
        jy0 = total_mass * total_mass + mc * mc - 2.0 * total_mass * EB * ec / T
        jy1 = 2.0 * total_mass * q * kc / T
        v0 = ma * ma + mc * mc + 2.0 * ea * ec * (EA * EB + q * q) / (H * T)
        vx = 2.0 * ka * ec * q * total_mass / (H * T)
        vy = -2.0 * ea * kc * q * total_mass / (H * T)
        vxy = -2.0 * ka * kc * (EA * EB + q * q) / (H * T)
        vw = -2.0 * ka * kc
        result.update({"q": q, "EA": EA, "EB": EB, "ea": ea, "ec": ec, "ka": ka, "kc": kc})
    elif mode == "nested":
        k = _two_body_momentum(T, H, mc, "abc nested decay")
        _two_body_momentum(total_mass, T, md, "parent-to-abc+d")
        Eq = (t + h - mc * mc) / (2.0 * T)
        Ec = (t + mc * mc - h) / (2.0 * T)
        Ed = (total_mass * total_mass - t - md * md) / (2.0 * T)
        kd = _two_body_momentum(total_mass, T, md, "parent-to-abc+d") * total_mass / T
        ea = (h + ma * ma - mb * mb) / (2.0 * H)
        ka = _two_body_momentum(H, ma, mb, "ab pair")
        jx0 = ma * ma + mc * mc + 2.0 * ea * (Eq * Ec + k * k) / H
        jx1 = -2.0 * ka * k * T / H
        jy0 = h + md * md + 2.0 * Eq * Ed
        jy1 = 2.0 * k * kd
        v0 = ma * ma + md * md + 2.0 * Eq * ea * Ed / H
        vx = -2.0 * k * ka * Ed / H
        vy = 2.0 * k * ea * kd / H
        vxy = -2.0 * Eq * ka * kd / H
        vw = -2.0 * ka * kd
        result.update({"k": k, "Eq": Eq, "Ec": Ec, "Ed": Ed, "kd": kd, "ea": ea, "ka": ka, "mc": mc, "md": md})
    else:
        raise ValueError(f"unknown low-loop geometry mode: {mode!r}")
    result.update({"jx0": jx0, "jx1": jx1, "jy0": jy0, "jy1": jy1,
                   "v0": v0, "vx": vx, "vy": vy, "vxy": vxy, "vw": vw})
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("non-finite low-loop angular-expansion coefficient")
    if jx1 == 0.0 or jy1 == 0.0:
        raise ValueError("jx1 and jy1 must be nonzero in the strict chart domain")
    if vw == 0.0:
        raise ValueError("the internal relative-azimuth coefficient vw must be nonzero")
    return result


def _solve_float_square(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> tuple[float, ...]:
    n = len(matrix)
    if n == 0:
        return ()
    if n != len(rhs) or any(len(row) != n for row in matrix):
        raise ValueError("angular elimination matrix must be square")
    work = [[float(value) for value in row] + [float(rhs[i])] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = next((row for row in range(column, n) if work[row][column] != 0.0), None)
        if pivot is None:
            raise ValueError("det(C_E)=0 at the supplied low-loop chart point")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(n):
            if row == column or work[row][column] == 0.0:
                continue
            scale = work[row][column]
            work[row] = [left - scale * right for left, right in zip(work[row], work[column])]
    return tuple(work[row][-1] for row in range(n))


def _rotation_zyz(theta: float, phi: float, psi: float) -> tuple[tuple[float, ...], ...]:
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(phi), math.sin(phi)
    cs, ss = math.cos(psi), math.sin(psi)
    rz_phi = ((cp, -sp, 0.0), (sp, cp, 0.0), (0.0, 0.0, 1.0))
    ry_theta = ((ct, 0.0, st), (0.0, 1.0, 0.0), (-st, 0.0, ct))
    rz_psi = ((cs, -ss, 0.0), (ss, cs, 0.0), (0.0, 0.0, 1.0))
    return _matmul3(_matmul3(rz_phi, ry_theta), rz_psi)


def _matmul3(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: Sequence[float], b: Sequence[float]) -> Vector3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: Sequence[float], name: str) -> Vector3:
    size = _norm(vector)
    if not math.isfinite(size) or size == 0.0:
        raise ValueError(f"{name} is degenerate in the strict chart domain")
    return tuple(value / size for value in vector)  # type: ignore[return-value]


def _boost(momentum: Vector4, beta: Sequence[float]) -> Vector4:
    b2 = _dot(beta, beta)
    if not math.isfinite(b2) or b2 >= 1.0:
        raise ValueError("Lorentz boost velocity must satisfy |beta|<1")
    if b2 == 0.0:
        return momentum
    gamma = 1.0 / math.sqrt(1.0 - b2)
    spatial = momentum[1:]
    projection = _dot(beta, spatial)
    factor = (gamma - 1.0) * projection / b2 + gamma * momentum[0]
    boosted = tuple(spatial[i] + factor * beta[i] for i in range(3))
    return (gamma * (momentum[0] + projection), *boosted)


def _dot4_sum(momenta: Sequence[Vector4]) -> Vector4:
    return tuple(sum(item[axis] for item in momenta) for axis in range(4))  # type: ignore[return-value]


def _invariant(momentum: Sequence[float]) -> float:
    return momentum[0] * momentum[0] - sum(value * value for value in momentum[1:])


def _subset_invariant(momenta: Sequence[Vector4], subset: Sequence[int]) -> float:
    return _invariant(_dot4_sum(tuple(momenta[index - 1] for index in subset)))


def _internal_v(geometry: Mapping[str, float], x: float, y: float, internal_phi: float) -> float:
    rx = _strict_sqrt("1-x^2", 1.0 - x * x)
    ry = _strict_sqrt("1-y^2", 1.0 - y * y)
    return (
        geometry["v0"] + geometry["vx"] * x + geometry["vy"] * y
        + geometry["vxy"] * x * y
        + geometry["vw"] * rx * ry * math.cos(internal_phi)
    )


def _coordinate_state(
    chart_data: Mapping[str, Any], values: Mapping[str, float] | Sequence[float],
    masses: Sequence[float], total_mass: float, branch: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    data = _read_chart_values(chart_data, values)
    masses, total_mass = validate_masses(masses, total_mass)
    if any(mass <= 0.0 for mass in masses):
        raise ValueError("low-loop direct profile requires strictly positive particle masses")
    if total_mass <= sum(masses):
        raise ValueError("low-loop open domain requires M>sum(m_i); threshold boundary is rejected")
    if isinstance(branch, bool) or not isinstance(branch, int) or branch not in (-1, 1):
        raise ValueError("branch must be +1 or -1")
    theta, phi, psi = (data[name] for name in LOW_LOOP_EULER_VARIABLES)
    if not 0.0 < theta < math.pi:
        raise ValueError("Euler boundary theta=0 or pi is rejected; 0<theta<pi is required")
    if not 0.0 <= phi < 2.0 * math.pi or not 0.0 <= psi < 2.0 * math.pi:
        raise ValueError("Euler angles require 0<=phi<2*pi and 0<=psi<2*pi")
    phi_eliminated = bool(chart_data["elimination"]["phi_eliminated"])
    if not phi_eliminated and branch != 1:
        raise ValueError("branch must be +1 when internal_phi is retained; its coordinate selects the mirror")

    target = tuple(chart_data["target_variables"])
    graph = {name: data[name] for name in target}
    s_matrix = _float_matrix(chart_data["S_matrix"])
    # Exact offsets are evaluated using the fixed physical masses.
    s_offsets = _offset_values(chart_data["S_mass_offset_coefficients"], masses, total_mass)
    b_values = [
        sum(float(_fraction(chart_data["K"][row][column])) * graph[target[column]] for column in range(len(target)))
        + _offset_values((chart_data["B_mass_offset_coefficients"][row],), masses, total_mass)[0]
        for row in range(chart_data["dimension"])
    ]
    h, t = b_values[0], b_values[1]
    geometry = _geometry_coefficients(chart_data, h, t, masses, total_mass)
    u_values = b_values[2:]
    q_z = _float_matrix(chart_data["U_in_Z_coefficients"])
    mass_z_offsets = _offset_values(chart_data["constraint_mass_offsets"], masses, total_mass)
    c_base = _float_matrix(chart_data["constraint_matrix_base"])
    c_numeric = tuple(
        (row[0] * geometry["jx1"], row[1] * geometry["jy1"], row[2])
        for row in c_base
    )
    u0 = tuple(
        q_z[index][0] * h + q_z[index][1] * t
        + q_z[index][2] * geometry["jx0"]
        + q_z[index][3] * geometry["jy0"]
        + mass_z_offsets[index]
        for index in range(len(u_values))
    )
    rhs = tuple(value - constant for value, constant in zip(u_values, u0))
    solved_columns = tuple(chart_data["solved_column_indices"])
    retained_columns = tuple(chart_data["retained_column_indices"])
    internal = {name: data[name] for name in chart_data["internal_variables"]}
    internal_names = ("x", "y", "v")
    angle_map = {"x": "x", "y": "y", "v": "internal_phi"}
    z_values: dict[int, float] = {}
    for column in retained_columns:
        coordinate_name = angle_map[internal_names[column]]
        if column == 2:
            # The exact pivot policy retains v only when its constraint column vanishes.
            z_values[column] = 0.0
        else:
            z_values[column] = internal[coordinate_name]
    matrix_e = tuple(tuple(row[column] for column in solved_columns) for row in c_numeric)
    rhs_e = tuple(
        rhs[i] - sum(c_numeric[i][column] * z_values[column] for column in retained_columns)
        for i in range(len(rhs))
    )
    solved = _solve_float_square(matrix_e, rhs_e) if solved_columns else ()
    z_values.update({column: value for column, value in zip(solved_columns, solved)})
    x, y = z_values[0], z_values[1]
    if not math.isfinite(x) or not math.isfinite(y) or not abs(x) < 1.0 or not abs(y) < 1.0:
        raise ValueError("internal cosine domain requires |x|<1 and |y|<1")
    internal_phi: float
    if phi_eliminated:
        v = z_values[2]
        rx = _strict_sqrt("1-x^2", 1.0 - x * x)
        ry = _strict_sqrt("1-y^2", 1.0 - y * y)
        cosine = (v - geometry["v0"] - geometry["vx"] * x - geometry["vy"] * y - geometry["vxy"] * x * y) / (geometry["vw"] * rx * ry)
        if not math.isfinite(cosine) or not abs(cosine) < 1.0:
            raise ValueError("internal-phi root requires |cos(internal_phi)|<1")
        principal = math.acos(cosine)
        internal_phi = principal if branch == 1 else 2.0 * math.pi - principal
        if not 0.0 < internal_phi < 2.0 * math.pi or internal_phi == math.pi:
            raise ValueError("internal-phi mirror root lies on an excluded boundary")
    else:
        internal_phi = internal["internal_phi"]
        if not 0.0 < internal_phi < 2.0 * math.pi or internal_phi == math.pi:
            raise ValueError("internal_phi must satisfy 0<internal_phi<2*pi and internal_phi!=pi")
        v = _internal_v(geometry, x, y, internal_phi)
    jx = geometry["jx0"] + geometry["jx1"] * x
    jy = geometry["jy0"] + geometry["jy1"] * y
    z = (h, t, jx, jy, v)
    z_offsets = _offset_values(chart_data["Z_mass_offset_coefficients"], masses, total_mass)
    inverse_az = _float_matrix(chart_data["inverse_A_Z"])
    x_common = tuple(sum(inverse_az[i][j] * (z[j] - z_offsets[j]) for j in range(5)) for i in range(5))
    reconstructed_s = tuple(
        sum(s_matrix[i][j] * x_common[j] for j in range(5)) + s_offsets[i]
        for i in range(len(target))
    )
    scale = max(1.0, total_mass * total_mass, *(abs(value) for value in graph.values()))
    for name, expected, actual in zip(target, graph.values(), reconstructed_s):
        if abs(expected - actual) > 2.0e-8 * scale:
            raise ValueError(f"analytic angular back-substitution does not recover graph invariant {name}")
    angles = dict(data)
    angles.update({"x": x, "y": y, "v": v, "internal_phi": internal_phi})
    return angles, {
        "masses": masses,
        "total_mass": total_mass,
        "h": h,
        "t": t,
        "U": u_values,
        "geometry": geometry,
        "Z": z,
        "X": x_common,
        "H_gram": None,
        "branch": branch,
        "phi_eliminated": phi_eliminated,
        "C_numeric": c_numeric,
        "det_C_E": _float_det(matrix_e),
    }


def _float_det(matrix: Sequence[Sequence[float]]) -> float:
    n = len(matrix)
    if n == 0:
        return 1.0
    if any(len(row) != n for row in matrix):
        raise ValueError("determinant matrix must be square")
    work = [list(map(float, row)) for row in matrix]
    result = 1.0
    for column in range(n):
        pivot = next((row for row in range(column, n) if work[row][column] != 0.0), None)
        if pivot is None:
            return 0.0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            result = -result
        value = work[column][column]
        result *= value
        for row in range(column + 1, n):
            if work[row][column] == 0.0:
                continue
            multiplier = work[row][column] / value
            for j in range(column + 1, n):
                work[row][j] -= multiplier * work[column][j]
            work[row][column] = 0.0
    return result


def _gram_from_common_x(x: Sequence[float], masses: Sequence[float], total_mass: float) -> tuple[tuple[float, ...], ...]:
    u, v, w, s12, s34 = x
    m1, m2, m3, m4 = masses
    e1 = (total_mass * total_mass + m1 * m1 - u) / (2.0 * total_mass)
    e2 = (s12 + u - s34 - m1 * m1) / (2.0 * total_mass)
    e3 = (v + u - w - m1 * m1) / (2.0 * total_mass)
    e4 = total_mass - e1 - e2 - e3
    energies = (e1, e2, e3, e4)
    if any(not math.isfinite(energy) or energy <= 0.0 for energy in energies):
        raise ValueError("low-loop strict support requires E_i>0 for every particle")
    s23 = u - w - s34 + m2 * m2 + m3 * m3 + m4 * m4
    h = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        h[i][i] = energies[i] * energies[i] - masses[i] * masses[i]
    for i, j, sij in ((0, 1, s12), (0, 2, v), (1, 2, s23)):
        h[i][j] = h[j][i] = energies[i] * energies[j] - (sij - masses[i] ** 2 - masses[j] ** 2) / 2.0
    h11, h22, h33 = h[0][0], h[1][1], h[2][2]
    h12, h13, h23 = h[0][1], h[0][2], h[1][2]
    minor2 = h11 * h22 - h12 * h12
    det_h = h11 * h22 * h33 + 2.0 * h12 * h13 * h23 - h11 * h23 * h23 - h22 * h13 * h13 - h33 * h12 * h12
    if h11 <= 0.0:
        raise ValueError("strict Gram domain requires H11>0")
    if minor2 <= 0.0:
        raise ValueError("strict Gram domain requires H11*H22-H12^2>0")
    if not math.isfinite(det_h) or det_h <= 0.0:
        raise ValueError("strict Gram domain requires det(H)>0")
    return tuple(tuple(row) for row in h)


def _shape_momenta(
    chart_data: Mapping[str, Any], state: Mapping[str, Any], angles: Mapping[str, float]
) -> tuple[Vector4, ...]:
    geometry = state["geometry"]
    x, y, internal_phi = angles["x"], angles["y"], angles["internal_phi"]
    sx, sy = _strict_sqrt("1-x^2", 1.0 - x * x), _strict_sqrt("1-y^2", 1.0 - y * y)
    a, b, c, d = chart_data["geometry"]["particles"]
    by_particle: dict[int, Vector4] = {}
    if chart_data["geometry"]["mode"] == "pair":
        H, T = geometry["H"], geometry["T"]
        q, EA, EB = geometry["q"], geometry["EA"], geometry["EB"]
        ea, ec, ka, kc = geometry["ea"], geometry["ec"], geometry["ka"], geometry["kc"]
        pa = (
            (EA * ea + q * ka * x) / H,
            ka * sx,
            0.0,
            (q * ea + EA * ka * x) / H,
        )
        q_ab = (EA, 0.0, 0.0, q)
        pb = tuple(q_ab[i] - pa[i] for i in range(4))
        pc = (
            (EB * ec - q * kc * y) / T,
            kc * sy * math.cos(internal_phi),
            kc * sy * math.sin(internal_phi),
            (-q * ec + EB * kc * y) / T,
        )
        q_cd = (EB, 0.0, 0.0, -q)
        pd = tuple(q_cd[i] - pc[i] for i in range(4))
        by_particle.update({a: pa, b: pb, c: pc, d: pd})
    else:
        H, T = geometry["H"], geometry["T"]
        k, Eq, Ec, Ed, kd = (geometry[name] for name in ("k", "Eq", "Ec", "Ed", "kd"))
        ea, ka = geometry["ea"], geometry["ka"]
        pa = (
            (Eq * ea - k * ka * x) / H,
            ka * sx,
            0.0,
            (Eq * ka * x - k * ea) / H,
        )
        q_ab = (Eq, 0.0, 0.0, -k)
        pb = tuple(q_ab[i] - pa[i] for i in range(4))
        pc = (Ec, 0.0, 0.0, k)
        pd = (Ed, kd * sy * math.cos(internal_phi), kd * sy * math.sin(internal_phi), kd * y)
        beta = tuple(-pd[i + 1] / (T + Ed) for i in range(3))
        pa, pb, pc, pd = (_boost(momentum, beta) for momentum in (pa, pb, pc, pd))
        by_particle.update({a: pa, b: pb, c: pc, d: pd})
    return tuple(by_particle[index] for index in range(1, 5))


def _canonical_body(momenta: Sequence[Vector4]) -> tuple[tuple[Vector3, ...], tuple[Vector3, Vector3, Vector3]]:
    p1, p2 = momenta[0][1:], momenta[1][1:]
    ez = _unit(p1, "p1 body axis")
    p2_transverse = tuple(p2[i] - _dot(p2, ez) * ez[i] for i in range(3))
    ex = _unit(p2_transverse, "p2 body x-axis")
    ey = _unit(_cross(ez, ex), "right-handed body y-axis")
    axes = (ex, ey, ez)
    body = tuple(tuple(_dot(momentum[1:], axis) for axis in axes) for momentum in momenta)
    return body, axes


def _rotate_body(
    body: Sequence[Sequence[float]], angles: Mapping[str, float]
) -> tuple[Vector4, ...]:
    rotation = _rotation_zyz(angles["theta"], angles["phi"], angles["psi"])
    out = []
    for vector in body:
        rotated = tuple(sum(rotation[i][j] * vector[j] for j in range(3)) for i in range(3))
        out.append((0.0, *rotated))
    return tuple(out)


def _attach_energies(
    momenta: Sequence[Vector4], masses: Sequence[float]
) -> tuple[Vector4, ...]:
    return tuple((momentum[0], momentum[1], momentum[2], momentum[3]) for momentum in momenta)


def _check_event(
    momenta: Sequence[Vector4], chart_data: Mapping[str, Any], expected: Mapping[str, float],
    masses: Sequence[float], total_mass: float,
) -> None:
    if len(momenta) != 4:
        raise ArithmeticError("low-loop reconstruction did not produce four momenta")
    scale = max(1.0, total_mass * total_mass)
    for index, (momentum, mass) in enumerate(zip(momenta, masses), 1):
        if momentum[0] <= 0.0 or not math.isfinite(momentum[0]):
            raise ValueError(f"strict support requires E_{index}>0")
        residual = _invariant(momentum) - mass * mass
        if abs(residual) > 5.0e-9 * scale:
            raise ArithmeticError(f"low-loop reconstruction failed p_{index} on-shell check")
    total = _dot4_sum(momenta)
    if abs(total[0] - total_mass) > 5.0e-9 * max(1.0, total_mass):
        raise ArithmeticError("low-loop reconstruction failed total-energy conservation")
    if any(abs(component) > 5.0e-9 * max(1.0, total_mass) for component in total[1:]):
        raise ArithmeticError("low-loop reconstruction failed total-momentum conservation")
    for label, subset in zip(chart_data["target_variables"], chart_data["S_affine_map"]):
        subset_ids = _parse_label(label)
        actual = _subset_invariant(momenta, subset_ids)
        if abs(actual - expected[label]) > 2.0e-8 * scale:
            raise ArithmeticError(f"low-loop reconstruction failed graph invariant {label}")


def _kinematics(
    chart_data: Mapping[str, Any], values: Mapping[str, float] | Sequence[float],
    masses: Sequence[float], total_mass: float, branch: int = 1,
) -> tuple[tuple[Vector4, ...], dict[str, Any]]:
    angles, state = _coordinate_state(chart_data, values, masses, total_mass, branch)
    shape = _shape_momenta(chart_data, state, angles)
    body, _axes = _canonical_body(shape)
    rotated_spatial = _rotate_body(body, angles)
    momenta = tuple(
        (shape[index][0], rotated_spatial[index][1], rotated_spatial[index][2], rotated_spatial[index][3])
        for index in range(4)
    )
    _check_event(momenta, chart_data, {name: angles[name] for name in chart_data["target_variables"]}, state["masses"], state["total_mass"])
    h_gram = _gram_from_common_x(state["X"], state["masses"], state["total_mass"])
    det_h = _float_det(h_gram)
    if not math.isfinite(det_h) or det_h <= 0.0:
        raise ValueError("strict support requires det(H)>0")
    state = {**state, "H_gram": h_gram, "det_H": det_h, "resolved_angles": angles, "shape_momenta": shape}
    return momenta, state


def reconstruct_low_loop_momenta(
    card_or_data: FormulaInputCard | Mapping[str, Any],
    values: Mapping[str, float] | Sequence[float],
    masses: Sequence[float],
    total_mass: float,
    branch: int = 1,
) -> tuple[Vector4, ...]:
    """Reconstruct parent-rest p1..p4 from graph S and retained coordinates."""

    chart_data = _as_chart_data(card_or_data)
    return _kinematics(chart_data, values, masses, total_mass, branch=branch)[0]


def numeric_density(
    card_or_data: FormulaInputCard | Mapping[str, Any],
    values: Mapping[str, float] | Sequence[float],
    masses: Sequence[float],
    total_mass: float,
    branch: int = 1,
    normalization: str = "jing_psf",
    density_mode: str = "per_branch",
) -> float:
    """Evaluate the analytic Gram pullback density on one branch."""

    if normalization not in {"jing_psf", "standard_2pi4"}:
        raise ValueError(f"unknown normalization: {normalization!r}")
    if density_mode not in {"per_branch", "summed"}:
        raise ValueError(f"unknown density_mode: {density_mode!r}")
    chart_data = _as_chart_data(card_or_data)
    _momenta, state = _kinematics(chart_data, values, masses, total_mass, branch=branch)
    geometry = state["geometry"]
    angles = state["resolved_angles"]
    if state["phi_eliminated"]:
        w_factor = 1.0
        summed_multiplier = 2.0 if density_mode == "summed" else 1.0
    else:
        w_factor = abs(geometry["vw"]) * math.sqrt(1.0 - angles["x"] ** 2) * math.sqrt(1.0 - angles["y"] ** 2) * abs(math.sin(angles["internal_phi"]))
        summed_multiplier = 1.0
    solved = tuple(chart_data["solved_column_indices"])
    c_e = tuple(tuple(state["C_numeric"][i][column] for column in solved) for i in range(len(solved)))
    determinant_c_e = abs(_float_det(c_e))
    if not math.isfinite(determinant_c_e) or determinant_c_e == 0.0:
        raise ValueError("density pullback requires det(C_E)!=0")
    determinant_k = abs(float(Fraction(chart_data["det_K"])))
    determinant_z = float(Fraction(chart_data["D"]))
    power = 12 if normalization == "jing_psf" else 8
    rho_x = 1.0 / (
        512.0 * (2.0 * math.pi) ** power * state["total_mass"] ** 3 * math.sqrt(state["det_H"])
    )
    pullback = (
        abs(geometry["jx1"] * geometry["jy1"]) * determinant_k
        / (determinant_z * determinant_c_e) * w_factor
    )
    result = rho_x * pullback * summed_multiplier
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("low-loop density is non-finite or non-positive in the strict open domain")
    return result


def build_low_loop_formula(
    card: FormulaInputCard, options: FormulaOptions | None = None
) -> DirectFormula:
    """Build the direct low-loop formula record and numeric closures."""

    chart_data = build_low_loop_chart_data(card)
    options = options or FormulaOptions(normalization=card.measure_convention)
    if options.normalization != card.measure_convention:
        raise ValueError("FormulaOptions.normalization disagrees with the FormulaInputCard measure convention")
    target_variables = tuple(chart_data["target_variables"])
    angle_variables = tuple(chart_data["angle_variables"])
    power = 12 if options.normalization == "jing_psf" else 8
    density_expression = chart_data["density"]["expression_by_normalization"][options.normalization]
    if options.density_mode == "summed" and chart_data["elimination"]["phi_eliminated"]:
        density_expression = "2*(" + density_expression + ")"
    branches = tuple(
        BranchRecord(
            key=(f"internal_phi_branch={sign:+d}" if chart_data["elimination"]["phi_eliminated"] else "internal_phi=explicit-coordinate"),
            equations=(
                "U-u0=C*(x,y,v)^T",
                chart_data["elimination"]["phi_roots"],
                "Z=(h,t,Jx,Jy,v); X=inverse_A_Z*(Z-b_Z)",
            ),
            root_condition=(
                "|cos(internal_phi)|<1; branch=sign(sin(internal_phi))"
                if chart_data["elimination"]["phi_eliminated"] else "0<internal_phi<2*pi and internal_phi!=pi; branch=+1"
            ),
            back_substitution=tuple(chart_data["momenta"]["equations"]),
            domain=tuple(chart_data["domain"]),
            multiplicity=1,
            coefficient=("one root; summed mode includes both roots" if chart_data["elimination"]["phi_eliminated"] and options.density_mode == "summed" else "one physical chart point"),
            summed=options.density_mode == "summed" and chart_data["elimination"]["phi_eliminated"],
        )
        for sign in ((1, -1) if chart_data["elimination"]["phi_eliminated"] else (1,))
    )
    coordinate_map = CoordinateMap(
        source=tuple(chart_data["basis_variables"]),
        target=target_variables + angle_variables,
        forward=tuple(
            f"{row['label']} = linear form in X plus exact mass offset"
            for row in chart_data["S_affine_map"]
        ),
        inverse=(
            "B=K*S+b with exact K and mass offsets",
            "U-u0=C*(x,y,v)^T; solve the exact-selected C_E subsystem",
            "if v is solved, reconstruct both acos mirror roots; otherwise internal_phi is explicit",
            "Z=(h,t,Jx,Jy,v); X=inverse_A_Z*(Z-b_Z)",
        ),
        jacobian=chart_data["jacobian"]["full_density"] + f"; normalization power p={power}",
    )

    def density_closure(values: Mapping[str, float] | Sequence[float], masses: Sequence[float], total_mass: float, branch: int = 1) -> float:
        return numeric_density(
            chart_data, values, masses, total_mass, branch=branch,
            normalization=options.normalization, density_mode=options.density_mode,
        )

    def momenta_closure(values: Mapping[str, float] | Sequence[float], masses: Sequence[float], total_mass: float, branch: int = 1) -> tuple[Vector4, ...]:
        return _kinematics(chart_data, values, masses, total_mass, branch=branch)[0]

    momentum_expressions = tuple(chart_data["momenta"]["equations"]) + (
        "Canonicalize by p1,p2 body axes, then apply active Rz(phi) Ry(theta) Rz(psi).",
    )
    return DirectFormula(
        profile="direct_gram_low_loop",
        target_variables=target_variables,
        angle_variables=angle_variables,
        density_expression=density_expression,
        momentum_expressions=momentum_expressions,
        jacobian=chart_data["jacobian"]["full_density"],
        branches=branches,
        domain=tuple(chart_data["domain"]),
        status="built",
        coordinate_map=coordinate_map,
        diagnostics=("built low-loop direct mixed-coordinate formula; numeric verification is pending",),
        numeric_density=density_closure,
        numeric_momenta=momenta_closure,
        normalization=options.normalization,
        differential_measure=chart_data["differential_measure"],
        angle_definitions=tuple(chart_data["euler_definitions"]) + tuple(chart_data["geometry"]["coordinate_definitions"][name] for name in chart_data["internal_variables"]),
    )


def _normalize_angle(value: float) -> float:
    result = value % (2.0 * math.pi)
    return 0.0 if result == 2.0 * math.pi else result


def _euler_from_axes(axes: Sequence[Sequence[float]]) -> dict[str, float]:
    # axes are the body basis vectors expressed in the parent-rest laboratory frame.
    rotation = tuple(tuple(axes[column][row] for column in range(3)) for row in range(3))
    cos_theta = rotation[2][2]
    if not math.isfinite(cos_theta) or not -1.0 <= cos_theta <= 1.0:
        raise ValueError("body orientation has invalid Euler cosine")
    theta = math.acos(cos_theta)
    if not 0.0 < theta < math.pi:
        raise ValueError("Euler body orientation lies on the excluded theta boundary")
    phi = _normalize_angle(math.atan2(rotation[1][2], rotation[0][2]))
    psi = _normalize_angle(math.atan2(rotation[2][1], -rotation[2][0]))
    return {"theta": theta, "phi": phi, "psi": psi}


def extract_low_loop_coordinates(
    card_or_data: FormulaInputCard | Mapping[str, Any],
    momenta: Sequence[Sequence[float]] | Mapping[int, Sequence[float]],
    masses: Sequence[float],
    total_mass: float,
) -> dict[str, Any]:
    """Read target IKVs and direct-geometry coordinates from an event.

    Internal angles are recovered by actual Lorentz boosts into the selected
    pair/nested subsystem rest frames.  No production inverse solver is used.
    """

    chart_data = _as_chart_data(card_or_data)
    masses, total_mass = validate_masses(masses, total_mass)
    if any(mass <= 0.0 for mass in masses):
        raise ValueError("low-loop direct profile requires strictly positive particle masses")
    if total_mass <= sum(masses):
        raise ValueError("low-loop open domain requires M>sum(m_i); threshold boundary is rejected")
    if isinstance(momenta, Mapping):
        try:
            vectors = tuple(tuple(_finite(f"p{particle}[{axis}]", momenta[particle][axis]) for axis in range(4))
                            for particle in range(1, 5))
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("momenta mapping must contain p1..p4 as four-vectors") from exc
    else:
        try:
            vectors = tuple(tuple(_finite(f"p{particle}[{axis}]", item[axis]) for axis in range(4))
                            for particle, item in enumerate(momenta, 1))
        except (TypeError, IndexError) as exc:
            raise ValueError("momenta must be four four-vectors") from exc
    if len(vectors) != 4 or any(len(item) != 4 for item in vectors):
        raise ValueError("momenta must contain exactly four four-vectors")
    total = _dot4_sum(vectors)
    scale = max(1.0, total_mass)
    if abs(total[0] - total_mass) > 5.0e-9 * scale or any(abs(value) > 5.0e-9 * scale for value in total[1:]):
        raise ValueError("coordinate extraction expects parent-rest COM momenta")
    for index, (momentum, mass) in enumerate(zip(vectors, masses), 1):
        if momentum[0] <= 0.0 or abs(_invariant(momentum) - mass * mass) > 5.0e-9 * max(1.0, total_mass * total_mass):
            raise ValueError(f"coordinate extraction requires positive on-shell p{index}")

    target_values = {
        label: _subset_invariant(vectors, _parse_label(label))
        for label in chart_data["target_variables"]
    }
    a, b, c, d = chart_data["geometry"]["particles"]
    mode = chart_data["geometry"]["mode"]
    if mode == "pair":
        q_ab = _dot4_sum((vectors[a - 1], vectors[b - 1]))
        q_cd = _dot4_sum((vectors[c - 1], vectors[d - 1]))
        ez = _unit(q_ab[1:], "Q_ab parent-rest direction")
        beta_ab = tuple(q_ab[i + 1] / q_ab[0] for i in range(3))
        beta_cd = tuple(q_cd[i + 1] / q_cd[0] for i in range(3))
        pa_rest = _boost(vectors[a - 1], tuple(-value for value in beta_ab))
        pc_rest = _boost(vectors[c - 1], tuple(-value for value in beta_cd))
        ka = _norm(pa_rest[1:])
        kc = _norm(pc_rest[1:])
        if ka == 0.0 or kc == 0.0:
            raise ValueError("pair-frame daughter momentum is zero on the excluded threshold boundary")
        x = _dot(pa_rest[1:], ez) / ka
        y = _dot(pc_rest[1:], ez) / kc
        pa_trans = tuple(vectors[a - 1][i + 1] - _dot(vectors[a - 1][1:], ez) * ez[i] for i in range(3))
        pc_trans = tuple(vectors[c - 1][i + 1] - _dot(vectors[c - 1][1:], ez) * ez[i] for i in range(3))
        ex = _unit(pa_trans, "first pair-plane transverse direction")
        ey = _unit(_cross(ez, ex), "pair-plane oriented transverse direction")
        internal_phi = _normalize_angle(math.atan2(_dot(pc_trans, ey), _dot(pc_trans, ex)))
    else:
        q_abc = _dot4_sum(tuple(vectors[index - 1] for index in (a, b, c)))
        beta_abc = tuple(-q_abc[i + 1] / q_abc[0] for i in range(3))
        p_abc = tuple(_boost(vectors[index - 1], beta_abc) for index in (a, b, c, d))
        pa_abc, pb_abc, pc_abc, pd_abc = p_abc
        ez = _unit(pc_abc[1:], "c direction in abc rest frame")
        q_ab_abc = _dot4_sum((pa_abc, pb_abc))
        beta_ab_abc = tuple(q_ab_abc[i + 1] / q_ab_abc[0] for i in range(3))
        pa_ab = _boost(pa_abc, tuple(-value for value in beta_ab_abc))
        ka = _norm(pa_ab[1:])
        kd = _norm(pd_abc[1:])
        if ka == 0.0 or kd == 0.0:
            raise ValueError("nested-frame daughter momentum is zero on an excluded threshold boundary")
        x = _dot(pa_ab[1:], ez) / ka
        y = _dot(pd_abc[1:], ez) / kd
        pa_trans = tuple(pa_abc[i + 1] - _dot(pa_abc[1:], ez) * ez[i] for i in range(3))
        pd_trans = tuple(pd_abc[i + 1] - _dot(pd_abc[1:], ez) * ez[i] for i in range(3))
        ex = _unit(pa_trans, "nested first-plane transverse direction")
        ey = _unit(_cross(ez, ex), "nested oriented transverse direction")
        internal_phi = _normalize_angle(math.atan2(_dot(pd_trans, ey), _dot(pd_trans, ex)))
    if not abs(x) < 1.0 or not abs(y) < 1.0:
        raise ValueError("extracted internal cosines lie on or outside excluded boundaries")
    if not 0.0 < internal_phi < 2.0 * math.pi or internal_phi == math.pi:
        raise ValueError("extracted internal_phi lies on an excluded angular boundary")
    body, axes = _canonical_body(vectors)
    del body
    euler = _euler_from_axes(axes)
    values = {**target_values, "x": x, "y": y, "internal_phi": internal_phi, **euler}
    phi_eliminated = bool(chart_data["elimination"]["phi_eliminated"])
    branch = (1 if math.sin(internal_phi) > 0.0 else -1) if phi_eliminated else 1
    chart_values = {name: values[name] for name in chart_data["target_variables"] + chart_data["angle_variables"]}
    return {
        "values": chart_values,
        "target_variables": target_values,
        "internal_coordinates": {"x": x, "y": y, "internal_phi": internal_phi},
        "euler_angles": euler,
        "branch": branch,
        "profile": chart_data["profile"],
    }


__all__ = [
    "build_low_loop_chart_data",
    "build_low_loop_chart_data_from_labels",
    "build_low_loop_formula",
    "reconstruct_low_loop_momenta",
    "numeric_density",
    "extract_low_loop_coordinates",
]
