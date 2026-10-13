"""Direct analytic chart for the four-body, three-loop (d=5) invariants.

The graph variables are related exactly to the common basis
``X=(s234,s13,s24,s12,s34)``.  The kinematic inverse uses the two mirror
roots of the spatial Gram matrix and the Jing density quoted in P02.
"""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Any, Mapping, Sequence

from ..formula_inputs import FormulaInputCard
from .inputs import validate_formula_card
from .kinematic_domain import validate_masses
from .records import BranchRecord, CoordinateMap, DirectFormula, FormulaOptions

Vector4 = tuple[float, float, float, float]
MatrixF = tuple[tuple[Fraction, ...], ...]

D5_BASIS = ("s234", "s13", "s24", "s12", "s34")
D5_ANGLE_VARIABLES = ("theta", "phi", "psi")
D5_ANGLE_DEFINITIONS = (
    "The body frame fixes k_1 along +z and k_2 in the xz plane with x_2>0; sigma=±1 selects the sign of k_{3y}.",
    "The parent-rest COM frame momenta are obtained from the body frame by active R_z(phi) R_y(theta) R_z(psi).",
    "0<theta<pi, 0<=phi<2*pi, 0<=psi<2*pi, and dR=sin(theta) dtheta dphi dpsi; theta endpoints are rejected.",
)
D5_DIFFERENTIAL_MEASURE = (
    "dPhi4 = rho_{S,sigma}(S) d^5S dR, with dR=sin(theta)dtheta dphi dpsi; "
    "rho is the coefficient per ordered graph-IKV measure d^5S and complete Euler-angle measure dR."
)
D5_DIFFERENTIAL_MEASURE_LATEX = (
    r"d\Phi_4=\rho_{S,\sigma}(S)\,d^5S\,dR,\quad{}"
    r"dR=\sin\theta\,d\theta\,d\phi\,d\psi"
)

_MASS_SYMBOLS = ("M^2", "m_1^2", "m_2^2", "m_3^2", "m_4^2")


def _fraction(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _format_linear(coefficients: Sequence[Fraction], symbols: Sequence[str]) -> str:
    terms: list[str] = []
    for coefficient, symbol in zip(coefficients, symbols):
        if not coefficient:
            continue
        magnitude = abs(coefficient)
        factor = symbol if magnitude == 1 else f"{_fraction_text(magnitude)} {symbol}"
        if not terms:
            terms.append(f"-{factor}" if coefficient < 0 else factor)
        else:
            terms.append(("- " if coefficient < 0 else "+ ") + factor)
    return " ".join(terms) if terms else "0"


def _latex_symbol(symbol: str) -> str:
    if symbol == "M^2":
        return "M^2"
    if symbol.startswith("m_"):
        particle = symbol[2:-2]
        return f"m_{{{particle}}}^2"
    if symbol.startswith("s") and symbol[1:].isdigit():
        return f"s_{{{symbol[1:]}}}"
    if symbol.startswith("s"):
        return f"s_{{{symbol[1:]}}}"
    return symbol


def _format_linear_latex(coefficients: Sequence[Fraction], symbols: Sequence[str]) -> str:
    terms: list[str] = []
    for coefficient, symbol in zip(coefficients, symbols):
        if not coefficient:
            continue
        magnitude = abs(coefficient)
        latex_name = _latex_symbol(symbol)
        factor = latex_name if magnitude == 1 else (
            f"\\frac{{{magnitude.numerator}}}{{{magnitude.denominator}}}{latex_name}"
            if magnitude.denominator != 1 else f"{magnitude.numerator}{latex_name}"
        )
        if not terms:
            terms.append(f"-{factor}" if coefficient < 0 else factor)
        else:
            terms.append(("- " if coefficient < 0 else "+ ") + factor)
    return " ".join(terms) if terms else "0"


def _fraction_latex(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    sign = "-" if value < 0 else ""
    return f"{sign}\\frac{{{abs(value.numerator)}}}{{{value.denominator}}}"


def _matrix_latex(matrix: Sequence[Sequence[Fraction]]) -> str:
    rows = r" \\ ".join(" & ".join(_fraction_latex(value) for value in row) for row in matrix)
    return r"\begin{pmatrix}" + rows + r"\end{pmatrix}"


def _matrix_det(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("the d=5 coordinate matrix must be square")
    work = [[_fraction(value) for value in row] for row in matrix]
    determinant = Fraction(1)
    for column in range(n):
        pivot = next((row for row in range(column, n) if work[row][column]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[pivot], work[column] = work[column], work[pivot]
            determinant = -determinant
        value = work[column][column]
        determinant *= value
        for row in range(column + 1, n):
            if not work[row][column]:
                continue
            multiplier = work[row][column] / value
            for col in range(column + 1, n):
                work[row][col] -= multiplier * work[column][col]
            work[row][column] = Fraction(0)
    return determinant


def _matrix_inverse(matrix: Sequence[Sequence[Fraction]]) -> MatrixF:
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("the d=5 coordinate matrix must be square")
    work = [
        [_fraction(value) for value in row] + [Fraction(int(i == j)) for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for column in range(n):
        pivot = next((row for row in range(column, n) if work[row][column]), None)
        if pivot is None:
            raise ValueError("the five graph IKVs are not independent in the d=5 basis")
        work[column], work[pivot] = work[pivot], work[column]
        value = work[column][column]
        work[column] = [item / value for item in work[column]]
        for row in range(n):
            if row == column or not work[row][column]:
                continue
            multiplier = work[row][column]
            work[row] = [left - multiplier * right for left, right in zip(work[row], work[column])]
    return tuple(tuple(row[n:]) for row in work)


def _pair_forms() -> dict[tuple[int, int], tuple[tuple[Fraction, ...], tuple[Fraction, ...]]]:
    """Return each sij as coefficients of X and (M²,m1²,...,m4²)."""

    zero = (Fraction(0),) * 5
    forms: dict[tuple[int, int], tuple[tuple[Fraction, ...], tuple[Fraction, ...]]] = {}
    # Basis entries: u=s234, v=s13, w=s24, t=s12, z=s34.
    forms[(1, 2)] = ((Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(0)), zero)
    forms[(1, 3)] = ((Fraction(0), Fraction(1), Fraction(0), Fraction(0), Fraction(0)), zero)
    forms[(2, 4)] = ((Fraction(0), Fraction(0), Fraction(1), Fraction(0), Fraction(0)), zero)
    forms[(3, 4)] = ((Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1)), zero)
    # s234=s23+s24+s34-(m2²+m3²+m4²).
    forms[(2, 3)] = (
        (Fraction(1), Fraction(0), Fraction(-1), Fraction(0), Fraction(-1)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(1)),
    )
    # Sum_{i<j}sij=M²+2 sum_i mi².
    forms[(1, 4)] = (
        (Fraction(-1), Fraction(-1), Fraction(0), Fraction(-1), Fraction(0)),
        (Fraction(1), Fraction(2), Fraction(1), Fraction(1), Fraction(1)),
    )
    return forms


_PAIR_FORMS = _pair_forms()


def _subset_form(subset: Sequence[int]) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    particles = tuple(subset)
    if len(set(particles)) != len(particles) or any(particle not in (1, 2, 3, 4) for particle in particles):
        raise ValueError(f"invalid d=5 invariant subset: {particles!r}")
    if len(particles) not in (2, 3):
        raise ValueError("d=5 graph IKVs must be two- or three-particle invariants")
    x = [Fraction(0)] * 5
    mass = [Fraction(0)] * 5
    for left_index, left in enumerate(particles):
        for right in particles[left_index + 1:]:
            pair = tuple(sorted((left, right)))
            pair_x, pair_mass = _PAIR_FORMS[pair]
            x = [old + item for old, item in zip(x, pair_x)]
            mass = [old + item for old, item in zip(mass, pair_mass)]
    if len(particles) == 3:
        for particle in particles:
            mass[particle] -= 1
    return tuple(x), tuple(mass)


def _subset_label(subset: Sequence[int]) -> str:
    return "s" + "".join(str(item) for item in subset)


def _parse_labels(labels: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    subsets: list[tuple[int, ...]] = []
    for label in labels:
        if not isinstance(label, str) or not label.startswith("s"):
            raise ValueError(f"invalid d=5 invariant label: {label!r}")
        try:
            subset = tuple(int(char) for char in label[1:])
        except ValueError as exc:
            raise ValueError(f"invalid d=5 invariant label: {label!r}") from exc
        if len(subset) not in (2, 3) or len(set(subset)) != len(subset) or tuple(sorted(subset)) != subset:
            raise ValueError(f"d=5 invariant labels must list two or three unique particle indices in increasing order: {label!r}")
        if any(particle not in (1, 2, 3, 4) for particle in subset) or _subset_label(subset) != label:
            raise ValueError(f"d=5 invariant labels must use increasing particle indices: {label!r}")
        subsets.append(subset)
    return tuple(subsets)


def _chart_data_from_subsets(
    subsets: Sequence[Sequence[int]],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    ordered_subsets = tuple(tuple(item) for item in subsets)
    ordered_labels = tuple(labels) if labels is not None else tuple(_subset_label(item) for item in ordered_subsets)
    if len(ordered_subsets) != 5 or len(ordered_labels) != 5:
        raise ValueError("the direct d=5 chart requires exactly five ordered graph IKVs")
    if len(set(ordered_labels)) != 5:
        raise ValueError("the five d=5 graph IKVs must be distinct")
    for subset, label in zip(ordered_subsets, ordered_labels):
        if len(subset) not in (2, 3) or len(set(subset)) != len(subset) or tuple(sorted(subset)) != subset:
            raise ValueError("d=5 invariant subsets must contain two or three unique increasing particle indices")
        if _subset_label(subset) != label:
            raise ValueError("d=5 variable labels disagree with their ordered particle subsets")

    forms = tuple(_subset_form(subset) for subset in ordered_subsets)
    matrix: MatrixF = tuple(row[0] for row in forms)
    mass_offset: MatrixF = tuple(row[1] for row in forms)
    determinant = _matrix_det(matrix)
    if determinant == 0:
        raise ValueError("the five graph IKVs are dependent in the common d=5 basis")
    inverse = _matrix_inverse(matrix)
    inverse_product = tuple(
        tuple(sum(matrix[row][k] * inverse[k][column] for k in range(5)) for column in range(5))
        for row in range(5)
    )
    identity = tuple(tuple(Fraction(int(row == column)) for column in range(5)) for row in range(5))
    if inverse_product != identity:
        raise ArithmeticError("exact d=5 affine inverse failed its L*L_inverse=I certificate")

    forward_equations = tuple(
        f"{label} = {_format_linear(x_coeff + mass_coeff, D5_BASIS + _MASS_SYMBOLS)}"
        for label, (x_coeff, mass_coeff) in zip(ordered_labels, forms)
    )
    forward_equations_latex = tuple(
        f"{_latex_symbol(label)} = {_format_linear_latex(x_coeff + mass_coeff, D5_BASIS + _MASS_SYMBOLS)}"
        for label, (x_coeff, mass_coeff) in zip(ordered_labels, forms)
    )
    inverse_equations: list[str] = []
    inverse_mass: list[tuple[Fraction, ...]] = []
    for row_index, basis_name in enumerate(D5_BASIS):
        s_coeff = tuple(inverse[row_index][column] for column in range(5))
        mass_coeff = tuple(
            -sum(inverse[row_index][column] * mass_offset[column][symbol] for column in range(5))
            for symbol in range(5)
        )
        inverse_mass.append(mass_coeff)
        inverse_equations.append(
            f"{basis_name} = {_format_linear(s_coeff + mass_coeff, ordered_labels + _MASS_SYMBOLS)}"
        )
    inverse_equations_latex = tuple(
        f"{_latex_symbol(basis_name)} = "
        + _format_linear_latex(
            tuple(inverse[row_index][column] for column in range(5)) + inverse_mass[row_index],
            ordered_labels + _MASS_SYMBOLS,
        )
        for row_index, basis_name in enumerate(D5_BASIS)
    )

    pair_expressions = {
        "s23": "s234 - s24 - s34 + m_2^2 + m_3^2 + m_4^2",
        "s14": "M^2 + 2(m_1^2+m_2^2+m_3^2+m_4^2) - (s12+s13+s23+s24+s34)",
    }
    energy_expressions = (
        "E_1=(M^2+m_1^2-s234)/(2M)",
        "E_2=(s12+s234-s34-m_1^2)/(2M)",
        "E_3=(s13+s234-s24-m_1^2)/(2M)",
        "E_4=M-E_1-E_2-E_3",
    )
    energy_expressions_latex = (
        r"E_1=\frac{M^2+m_1^2-u}{2M}",
        r"E_2=\frac{t+u-z-m_1^2}{2M}",
        r"E_3=\frac{v+u-w-m_1^2}{2M}",
        r"E_4=M-E_1-E_2-E_3",
    )
    scalar_elimination = (
        {"constraint": "f1=s234-M^2-m_1^2+2M E_1", "constraint_latex": r"f_1=u-M^2-m_1^2+2ME_1=0", "eliminated_variable": "E_1", "root": "E_1=(M^2+m_1^2-s234)/(2M)", "root_latex": r"E_1=\frac{M^2+m_1^2-u}{2M}", "absolute_derivative": "2M", "absolute_derivative_latex": r"\left|\frac{\partial f_1}{\partial E_1}\right|=2M"},
        {"constraint": "f2=s34-M^2-s12+2M(E_1+E_2)", "constraint_latex": r"f_2=z-M^2-t+2M(E_1+E_2)=0", "eliminated_variable": "E_2", "root": "E_2=(M^2+s12-s34)/(2M)-E_1=(s12+s234-s34-m_1^2)/(2M)", "root_latex": r"E_2=\frac{M^2+t-z}{2M}-E_1=\frac{t+u-z-m_1^2}{2M}", "absolute_derivative": "2M", "absolute_derivative_latex": r"\left|\frac{\partial f_2}{\partial E_2}\right|=2M"},
        {"constraint": "f3=s24-M^2-s13+2M(E_1+E_3)", "constraint_latex": r"f_3=w-M^2-v+2M(E_1+E_3)=0", "eliminated_variable": "E_3", "root": "E_3=(M^2+s13-s24)/(2M)-E_1=(s13+s234-s24-m_1^2)/(2M)", "root_latex": r"E_3=\frac{M^2+v-w}{2M}-E_1=\frac{v+u-w-m_1^2}{2M}", "absolute_derivative": "2M", "absolute_derivative_latex": r"\left|\frac{\partial f_3}{\partial E_3}\right|=2M"},
        {"constraint": "f4=s12-m_1^2-m_2^2-2E_1E_2+2k_1k_2 c12", "constraint_latex": r"f_4=t-m_1^2-m_2^2-2E_1E_2+2k_1k_2c_{12}=0", "eliminated_variable": "c12", "root": "c12=(m_1^2+m_2^2+2E_1E_2-s12)/(2k_1k_2)", "root_latex": r"c_{12}=\frac{m_1^2+m_2^2+2E_1E_2-t}{2k_1k_2}", "absolute_derivative": "2k_1k_2", "absolute_derivative_latex": r"\left|\frac{\partial f_4}{\partial c_{12}}\right|=2k_1k_2"},
        {"constraint": "f5=s13-m_1^2-m_3^2-2E_1E_3+2k_1k_3 c13", "constraint_latex": r"f_5=v-m_1^2-m_3^2-2E_1E_3+2k_1k_3c_{13}=0", "eliminated_variable": "c13", "root": "c13=(m_1^2+m_3^2+2E_1E_3-s13)/(2k_1k_3)", "root_latex": r"c_{13}=\frac{m_1^2+m_3^2+2E_1E_3-v}{2k_1k_3}", "absolute_derivative": "2k_1k_3", "absolute_derivative_latex": r"\left|\frac{\partial f_5}{\partial c_{13}}\right|=2k_1k_3"},
        {"constraint": "F=(M-E_1-E_2-E_3)^2-m_4^2-k_1^2-k_2^2-k_3^2-2k_1k_2 c12-2k_1k_3 c13-2k_2k_3 c23", "constraint_latex": r"F=(M-E_1-E_2-E_3)^2-m_4^2-k_1^2-k_2^2-k_3^2-2k_1k_2c_{12}-2k_1k_3c_{13}-2k_2k_3c_{23}=0", "eliminated_variable": "c23", "root": "c23=((M-E_1-E_2-E_3)^2-m_4^2-k_1^2-k_2^2-k_3^2-2k_1k_2 c12-2k_1k_3 c13)/(2k_2k_3)", "root_latex": r"c_{23}=\frac{(M-E_1-E_2-E_3)^2-m_4^2-k_1^2-k_2^2-k_3^2-2k_1k_2c_{12}-2k_1k_3c_{13}}{2k_2k_3}", "absolute_derivative": "2k_2k_3", "absolute_derivative_latex": r"\left|\frac{\partial F}{\partial c_{23}}\right|=2k_2k_3"},
    )
    scalar_elimination = tuple(
        {**step, "root_derivative_latex": step["absolute_derivative_latex"]}
        for step in scalar_elimination
    )
    domain = (
        "m_i>0 for i=1,2,3,4; M>m_1+m_2+m_3+m_4; all E_i>0",
        "H_11>0; H_11 H_22-H_12^2>0; det(H)>0 (strict open Gram-positive domain)",
        "0<theta<pi; 0<=phi<2*pi; 0<=psi<2*pi; dR=sin(theta)dtheta dphi dpsi; theta endpoints are rejected",
        "finite inputs only; physical-domain equalities and Euler-coordinate boundaries with zero angular measure are rejected by numeric reconstruction",
    )
    cosine_gram = (
        "G_c=[[1,c12,c13],[c12,1,c23],[c13,c23,1]]",
        "Delta=det(G_c)=1+2c12 c13 c23-c12^2-c13^2-c23^2",
        "det(H)=(k_1 k_2 k_3)^2 Delta; sqrt(det(H))=k_1 k_2 k_3 sqrt(Delta)",
    )
    root_data = {
        "energies": list(energy_expressions),
        "energies_latex": list(energy_expressions_latex),
        "magnitudes": ("k_i=sqrt(E_i^2-m_i^2), i=1,2,3",),
        "magnitudes_latex": (r"k_i=\sqrt{E_i^2-m_i^2},\quad i=1,2,3",),
        "cosines": (
            "c12=(m_1^2+m_2^2+2E_1E_2-s12)/(2k_1k_2)",
            "c13=(m_1^2+m_3^2+2E_1E_3-s13)/(2k_1k_3)",
            "c23=((M-E_1-E_2-E_3)^2-m_4^2-k_1^2-k_2^2-k_3^2-2k_1k_2 c12-2k_1k_3 c13)/(2k_2k_3)",
        ),
        "cosines_latex": (
            r"c_{12}=\frac{m_1^2+m_2^2+2E_1E_2-t}{2k_1k_2}",
            r"c_{13}=\frac{m_1^2+m_3^2+2E_1E_3-v}{2k_1k_3}",
            r"c_{23}=\frac{E_4^2-m_4^2-k_1^2-k_2^2-k_3^2-2k_1k_2c_{12}-2k_1k_3c_{13}}{2k_2k_3}",
        ),
        "mirror_root": "k_{3y}=sigma sqrt(det(H))/(k_1 x_2), sigma in {+1,-1}",
        "mirror_root_latex": r"k_{3y}=\sigma\frac{\sqrt{\det H}}{k_1x_2},\quad\sigma=\pm1",
        "mirror_root_constraint": "f_y=det(H)-H_11 x_2^2 k_{3y}^2=0",
        "mirror_root_constraint_latex": r"f_y=\det H-H_{11}x_2^2k_{3y}^2=0",
        "mirror_root_absolute_derivative": "|df_y/dk_{3y}|=2 H_11 x_2^2 |k_{3y}|=2 k_1 x_2 sqrt(det(H))",
        "mirror_root_absolute_derivative_latex": r"\left|\frac{\partial f_y}{\partial k_{3y}}\right|=2H_{11}x_2^2|k_{3y}|=2k_1x_2\sqrt{\det H}",
        "mirror_root_derivative_role": "geometric mirror-root reconstruction only; this is not an additional delta-integration Jacobian",
        "mirror_root_derivative_not_extra_jacobian": True,
        "body_frame_momenta": (
            "k_1=(0,0,k_1)",
            "k_2=(x_2,0,z_2), z_2=H_12/k_1, x_2=sqrt(H_22-z_2^2)>0",
            "k_3=(x_3,sigma sqrt(det(H))/(k_1 x_2),z_3), z_3=H_13/k_1, x_3=(H_23-z_2 z_3)/x_2",
            "k_4=-(k_1+k_2+k_3)",
        ),
        "body_frame_momenta_latex": (
            r"\mathbf{k}_1=(0,0,k_1),\quad k_1=\sqrt{H_{11}}",
            r"z_2=H_{12}/k_1,\quad x_2=\sqrt{H_{22}-z_2^2}>0,\quad\mathbf{k}_2=(x_2,0,z_2)",
            r"z_3=H_{13}/k_1,\quad x_3=(H_{23}-z_2z_3)/x_2,\quad\mathbf{k}_3=(x_3,\sigma\sqrt{\det H}/(k_1x_2),z_3)",
            r"\mathbf{k}_4=-(\mathbf{k}_1+\mathbf{k}_2+\mathbf{k}_3)",
        ),
        "four_momenta_latex": tuple(
            rf"p_{{{particle}}}^{{\rm COM}}=(E_{{{particle}}},R_z(\phi)R_y(\theta)R_z(\psi)\mathbf{{k}}_{{{particle}}}^{{\rm body}})"
            for particle in range(1, 5)
        ),
        "rotation": "k_i^COM=R_z(phi)R_y(theta)R_z(psi) k_i^body; parent-rest COM frame, active rotations",
        "rotation_latex": r"\mathbf{k}_i^{\rm COM}=R_z(\phi)R_y(\theta)R_z(\psi)\mathbf{k}_i^{\rm body}\quad(\text{parent rest frame; active})",
    }
    chi_open = (
        "Theta(E1)Theta(E2)Theta(E3)Theta(E4)Theta(H11)"
        "Theta(H11*H22-H12^2)Theta(det(H)); strict open support indicator"
    )
    chi_open_latex = (
        r"\chi_{\rm open}=\Theta_+(E_1)\Theta_+(E_2)\Theta_+(E_3)\Theta_+(E_4)"
        r"\Theta_+(H_{11})\Theta_+(H_{11}H_{22}-H_{12}^2)\Theta_+(\det H),\quad"
        r"\quad\Theta_+(x)=\mathbf{1}_{x>0}"
    )
    density_base = "chi_open(S)/(512 (2*pi)^12 M^3 sqrt(det(H(X(S)))) abs(det(L)))"
    normalization_expressions = {
        "jing_psf_per_branch": f"rho_sigma={density_base}",
        "standard_2pi4_per_branch": "rho_sigma=chi_open(S)/(512 (2*pi)^8 M^3 sqrt(det(H(X(S)))) abs(det(L)))",
        "jing_psf_summed": f"rho_sum=2*{density_base}",
        "standard_2pi4_summed": "rho_sum=2*chi_open(S)/(512 (2*pi)^8 M^3 sqrt(det(H(X(S)))) abs(det(L)))",
    }
    return {
        "profile": "direct_gram_d5",
        "status": "built",
        "target_variables": list(ordered_labels),
        "ordered_s": list(ordered_labels),
        "basis_variables": list(D5_BASIS),
        "L": [[_fraction_text(value) for value in row] for row in matrix],
        "inverse_L": [[_fraction_text(value) for value in row] for row in inverse],
        "L_latex": "L=" + _matrix_latex(matrix),
        "inverse_L_latex": "L^{-1}=" + _matrix_latex(inverse),
        "det_L": _fraction_text(determinant),
        "abs_det_L": _fraction_text(abs(determinant)),
        "det_L_latex": r"\det L=" + _fraction_latex(determinant),
        "abs_det_L_latex": r"|\det L|=" + _fraction_latex(abs(determinant)),
        "mass_offset": [
            _format_linear(row, _MASS_SYMBOLS) for row in mass_offset
        ],
        "mass_offset_latex": [
            f"b_{{{index + 1}}}={_format_linear_latex(row, _MASS_SYMBOLS)}"
            for index, row in enumerate(mass_offset)
        ],
        "mass_offset_coefficients": [
            {symbol: _fraction_text(value) for symbol, value in zip(_MASS_SYMBOLS, row)}
            for row in mass_offset
        ],
        "forward_equations": list(forward_equations),
        "forward_equations_latex": list(forward_equations_latex),
        "inverse_equations": inverse_equations,
        "inverse_equations_latex": list(inverse_equations_latex),
        "coordinate_map": {
            "source": list(D5_BASIS),
            "target": list(ordered_labels),
            "forward": list(forward_equations),
            "forward_latex": list(forward_equations_latex),
            "inverse": inverse_equations,
            "inverse_latex": list(inverse_equations_latex),
            "jacobian": f"d^5S=abs(det(L)) d^5X={_fraction_text(abs(determinant))} d^5X; d^5X=abs(det(L))^(-1) d^5S",
            "constraint_delta_change": "delta^5[S-F_S(p)]=abs(det(L))^(-1) delta^5[X-F_X(p)] at fixed M,m_i",
            "constraint_delta_change_latex": r"\delta^5[S-F_S(p)]=|\det L|^{-1}\delta^5[X-F_X(p)]\quad(M,m_i\text{ fixed})",
        },
        "determinant_certificate": {
            "method": "exact Fraction Gaussian elimination",
            "det_L": _fraction_text(determinant),
            "inverse_L": [[_fraction_text(value) for value in row] for row in inverse],
            "L_inverse_product": [[_fraction_text(value) for value in row] for row in inverse_product],
            "identity_check": "passed exactly: L*inverse_L=I",
        },
        "completed_pair_invariants": pair_expressions,
        "completed_pair_invariants_latex": {
            "s23": r"s_{23}=u-w-z+m_2^2+m_3^2+m_4^2",
            "s14": r"s_{14}=M^2+2\sum_{i=1}^4m_i^2-(s_{12}+s_{13}+s_{23}+s_{24}+s_{34})",
        },
        "energy_expressions": list(energy_expressions),
        "energy_expressions_latex": list(energy_expressions_latex),
        "scalar_elimination": list(scalar_elimination),
        "scalar_elimination_jacobian": {
            "factorized_absolute_derivatives": ["2M", "2M", "2M", "2k_1k_2", "2k_1k_3", "2k_2k_3"],
            "total": "J_scalar=(2M)^3*(2k_1k_2)*(2k_1k_3)*(2k_2k_3)",
            "total_latex": r"J_{\rm scalar}=(2M)^3(2k_1k_2)(2k_1k_3)(2k_2k_3)=64M^3(k_1k_2k_3)^2",
            "total_jacobian_latex": r"J_{\rm scalar}=(2M)^3(2k_1k_2)(2k_1k_3)(2k_2k_3)=64M^3(k_1k_2k_3)^2",
            "constraint_group_change": "s34-(P-p_1-p_2)^2=f2+f4; s24-(P-p_1-p_3)^2=f3+f5; the two row operations together have absolute determinant 1",
            "constraint_group_change_latex": r"s_{34}-(P-p_1-p_2)^2=f_2+f_4,\quad s_{24}-(P-p_1-p_3)^2=f_3+f_5,\quad|\det T|=1",
            "constraint_group_determinants_latex": (
                r"\det\frac{\partial(f_2,f_2+f_4)}{\partial(f_2,f_4)}=1",
                r"\det\frac{\partial(f_3,f_3+f_5)}{\partial(f_3,f_5)}=1",
                r"|\det T|=1\quad\text{for both grouped constraint changes}",
            ),
            "angle_change_per_mirror_branch": "dOmega_1 dOmega_2 dOmega_3=dR dc12 dc13 dc23/sqrt(Delta)",
            "angle_change_per_mirror_branch_latex": r"d\Omega_1d\Omega_2d\Omega_3=\frac{dR\,dc_{12}dc_{13}dc_{23}}{\sqrt{\Delta}}\quad\text{(each mirror branch)}",
            "gram_relation": list(cosine_gram),
            "gram_relation_latex": (
                r"G_c=\begin{pmatrix}1&c_{12}&c_{13}\\c_{12}&1&c_{23}\\c_{13}&c_{23}&1\end{pmatrix}",
                r"\Delta=\det G_c=1+2c_{12}c_{13}c_{23}-c_{12}^2-c_{13}^2-c_{23}^2",
                r"\det H=(k_1k_2k_3)^2\Delta,\quad\sqrt{\det H}=k_1k_2k_3\sqrt{\Delta}",
            ),
        },
        "normalization_derivation_latex": (
            r"\frac{k_1k_2k_3}{8(2\pi)^{12}}\,dE_1dE_2dE_3\,d\Omega_1d\Omega_2d\Omega_3\,\delta(F)\prod_{i=1}^{5}\delta(f_i)\,d^5X",
            r"J_{\rm scalar}=(2M)^3(2k_1k_2)(2k_1k_3)(2k_2k_3)=64M^3(k_1k_2k_3)^2",
            r"d\Omega_1d\Omega_2d\Omega_3=\frac{dR\,dc_{12}dc_{13}dc_{23}}{\sqrt{\Delta}},\quad\sqrt{\det H}=k_1k_2k_3\sqrt{\Delta}",
            r"\frac{k_1k_2k_3}{8(2\pi)^{12}}\frac{1}{J_{\rm scalar}\sqrt{\Delta}}=\frac{1}{512(2\pi)^{12}M^3\sqrt{\det H}}\quad\text{per mirror branch}",
            r"\delta^5[S-F_S(p)]=|\det L|^{-1}\delta^5[X-F_X(p)]\quad(M,m_i\text{ fixed})",
            r"d^5X=|\det L|^{-1}d^5S\quad\Longrightarrow\quad\rho_{S,\sigma}=\frac{\chi_{\rm open}(S)}{512(2\pi)^{12}M^3\sqrt{\det H(X(S))}}|\det L|^{-1}",
            r"\rho_{\rm standard\_2pi4}=(2\pi)^4\rho_{\rm jing\_psf}\quad\text{and}\quad\rho_{\rm summed}=2\rho_\sigma\text{ with both momentum branches retained}",
        ),
        "root": root_data,
        "domain": list(domain),
        "domain_latex": (
            r"m_i>0\ (i=1,2,3,4),\quad M>\sum_{i=1}^4m_i,\quad E_i>0\ (i=1,2,3,4)",
            r"H_{11}>0,\quad H_{11}H_{22}-H_{12}^2>0,\quad\det H>0",
            r"0<\theta<\pi,\quad0\le\phi<2\pi,\quad0\le\psi<2\pi",
        ),
        "root_conditions": (
            "E_i>0 for i=1,2,3,4",
            "H_11>0; H_11 H_22-H_12^2>0; det(H)>0",
            "k_1>0, k_2>0, k_3>0 and Delta>0 follow from positive-definite H",
        ),
        "root_conditions_latex": (
            r"E_i>0\quad(i=1,2,3,4)",
            r"H_{11}>0,\quad H_{11}H_{22}-H_{12}^2>0,\quad\det H>0",
            r"k_1,k_2,k_3>0,\quad\Delta>0",
        ),
        "gram_matrix": (
            "H_ii=E_i^2-m_i^2 (i=1,2,3)",
            "H_ij=E_i E_j-(s_ij-m_i^2-m_j^2)/2 (i!=j; s12,s13,s23 from the explicit expressions above)",
        ),
        "gram_matrix_latex": (
            r"H_{ii}=E_i^2-m_i^2\quad(i=1,2,3)",
            r"H_{ij}=E_iE_j-\frac{s_{ij}-m_i^2-m_j^2}{2}\quad(i\ne j),\quad s_{12}=t,\ s_{13}=v,\ s_{23}=u-w-z+m_2^2+m_3^2+m_4^2",
        ),
        "branches": [
            {"label": "sigma=+1", "sigma": 1, "root": "k_{3y}=+sqrt(det(H))/(k_1 x_2)", "root_latex": r"k_{3y}=+\sqrt{\det H}/(k_1x_2)", "momentum_reconstruction_latex": [*root_data["energies_latex"], *root_data["magnitudes_latex"], *root_data["cosines_latex"], *root_data["body_frame_momenta_latex"], *root_data["four_momenta_latex"]], "density_weight": "one branch"},
            {"label": "sigma=-1", "sigma": -1, "root": "k_{3y}=-sqrt(det(H))/(k_1 x_2)", "root_latex": r"k_{3y}=-\sqrt{\det H}/(k_1x_2)", "momentum_reconstruction_latex": [*root_data["energies_latex"], *root_data["magnitudes_latex"], *root_data["cosines_latex"], *root_data["body_frame_momenta_latex"], *root_data["four_momenta_latex"]], "density_weight": "one branch"},
        ],
        "density_expression": density_base,
        "density_expression_latex": r"\rho_{S,\sigma}(S)=\frac{\chi_{\rm open}(S)}{512(2\pi)^{12}M^3\sqrt{\det H(X(S))}}\,|\det L|^{-1}",
        "chi_open_definition": chi_open,
        "chi_open_definition_latex": chi_open_latex,
        "normalization_expressions": normalization_expressions,
        "normalization_expressions_latex": {
            "jing_psf_per_branch": r"\rho_\sigma=\frac{\chi_{\rm open}(S)}{512(2\pi)^{12}M^3\sqrt{\det H(X(S))}}|\det L|^{-1}",
            "standard_2pi4_per_branch": r"\rho_\sigma=\frac{\chi_{\rm open}(S)}{512(2\pi)^8M^3\sqrt{\det H(X(S))}}|\det L|^{-1}",
            "jing_psf_summed": r"\rho_{\rm sum}=\frac{2\chi_{\rm open}(S)}{512(2\pi)^{12}M^3\sqrt{\det H(X(S))}}|\det L|^{-1}\quad(\text{both mirror branches})",
            "standard_2pi4_summed": r"\rho_{\rm sum}=\frac{2\chi_{\rm open}(S)}{512(2\pi)^8M^3\sqrt{\det H(X(S))}}|\det L|^{-1}\quad(\text{both mirror branches})",
        },
        "angle_variables": list(D5_ANGLE_VARIABLES),
        "angle_variables_latex": [r"\theta", r"\phi", r"\psi"],
        "angle_definitions": list(D5_ANGLE_DEFINITIONS),
        "rotation_latex": root_data["rotation_latex"],
        "angle_measure": "dR=sin(theta)dtheta dphi dpsi",
        "angle_measure_latex": r"dR=\sin\theta\,d\theta\,d\phi\,d\psi",
        "differential_measure": D5_DIFFERENTIAL_MEASURE,
        "differential_measure_latex": D5_DIFFERENTIAL_MEASURE_LATEX,
        "diagnostics": ["status=built; no independent formula verification is implied"],
    }


def _checked_card(card: FormulaInputCard) -> tuple[tuple[tuple[int, ...], ...], dict[str, Any]]:
    if not isinstance(card, FormulaInputCard):
        raise TypeError("d=5 direct formula expects a FormulaInputCard")
    validate_formula_card(card)
    if card.loop_count != 3 or len(card.independent_variables) != 5:
        raise ValueError("the direct Gram chart supports only four-body three-loop d=5 cards")
    labels = tuple(card.variable_labels)
    subsets = tuple(tuple(group) for group in card.independent_variables)
    if len(labels) != 5 or len(subsets) != 5:
        raise ValueError("the direct Gram chart requires exactly five ordered independent IKVs")
    data = _chart_data_from_subsets(subsets, labels)
    return subsets, data


def build_d5_chart_data(card: FormulaInputCard) -> dict[str, Any]:
    """Return exact affine-map and d=5 Gram derivation data for export."""

    _, data = _checked_card(card)
    return data


def _read_values(target_variables: Sequence[str], values: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    names = tuple(target_variables) + D5_ANGLE_VARIABLES
    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(f"missing d=5 chart coordinates: {missing}")
        try:
            result = {name: float(values[name]) for name in names}
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("d=5 chart coordinates must be finite real numbers") from exc
    else:
        try:
            sequence = tuple(values)
        except (TypeError, ValueError) as exc:
            raise ValueError("d=5 chart coordinates must be a mapping or a finite sequence") from exc
        if len(sequence) != len(names):
            raise ValueError(f"expected {len(names)} d=5 chart coordinates, got {len(sequence)}")
        try:
            result = dict(zip(names, map(float, sequence)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("d=5 chart coordinates must be finite real numbers") from exc
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("non-finite d=5 chart coordinates are rejected")
    return result


def _evaluate_linear(coefficients: Sequence[str], values: Mapping[str, float]) -> float:
    result = 0.0
    for coefficient, name in zip(coefficients, D5_BASIS):
        result += float(Fraction(coefficient)) * values[name]
    return result


def _coordinate_values(chart_data: Mapping[str, Any], graph_values: Mapping[str, float]) -> dict[str, float]:
    masses_squared = [graph_values["_M2"]] + [graph_values[f"_m{index}2"] for index in range(1, 5)]
    result: dict[str, float] = {}
    for basis_index, basis_name in enumerate(D5_BASIS):
        row = chart_data["inverse_L"][basis_index]
        offset = chart_data["mass_offset_coefficients"]
        value = sum(float(Fraction(row[column])) * graph_values[name]
                    for column, name in enumerate(chart_data["ordered_s"]))
        value -= sum(
            float(Fraction(chart_data["inverse_L"][basis_index][column]))
            * sum(
                float(Fraction(offset[column][_MASS_SYMBOLS[symbol]])) * masses_squared[symbol]
                for symbol in range(5)
            )
            for column in range(5)
        )
        result[basis_name] = value
    return result


def _pair_invariants(x: Mapping[str, float], masses: Sequence[float], total_mass: float) -> dict[str, float]:
    u, v, w, t, z = (x[name] for name in D5_BASIS)
    m1, m2, m3, m4 = masses
    m234 = m2 * m2 + m3 * m3 + m4 * m4
    s23 = u - w - z + m234
    s14 = total_mass * total_mass + 2.0 * sum(mass * mass for mass in masses) - (t + v + s23 + w + z)
    return {"s12": t, "s13": v, "s14": s14, "s23": s23, "s24": w, "s34": z}


def _kinematics(
    chart_data: Mapping[str, Any],
    target_variables: Sequence[str],
    values: Mapping[str, float] | Sequence[float],
    masses: Sequence[float],
    total_mass: float,
    branch: int = 1,
) -> tuple[tuple[Vector4, ...], dict[str, Any]]:
    data = _read_values(target_variables, values)
    masses, total_mass = validate_masses(masses, total_mass)
    if any(mass <= 0.0 for mass in masses):
        raise ValueError("the d=5 direct profile requires strictly positive particle masses")
    threshold = sum(masses)
    if total_mass <= threshold:
        raise ValueError("d=5 open domain requires total_mass > sum(masses); threshold boundary is rejected")
    theta, phi, psi = (data[name] for name in D5_ANGLE_VARIABLES)
    if not 0.0 < theta < math.pi:
        raise ValueError("Euler boundary theta=0 or pi is rejected; 0<theta<pi is required")
    if not 0.0 <= phi < 2.0 * math.pi or not 0.0 <= psi < 2.0 * math.pi:
        raise ValueError("Euler angles require 0<=phi<2*pi and 0<=psi<2*pi")

    graph_with_mass = dict(data)
    graph_with_mass["_M2"] = total_mass * total_mass
    graph_with_mass.update({f"_m{index}2": mass * mass for index, mass in enumerate(masses, 1)})
    x = _coordinate_values(chart_data, graph_with_mass)
    u, v, w, t, z = (x[name] for name in D5_BASIS)
    pairs = _pair_invariants(x, masses, total_mass)

    # These are the six scalar roots f1..f6 used in the P02 chart.
    e1 = (total_mass * total_mass + masses[0] ** 2 - u) / (2.0 * total_mass)
    e2 = (t + u - z - masses[0] ** 2) / (2.0 * total_mass)
    e3 = (v + u - w - masses[0] ** 2) / (2.0 * total_mass)
    e4 = total_mass - e1 - e2 - e3
    energies = (e1, e2, e3, e4)
    if any(not math.isfinite(value) for value in energies):
        raise ValueError("non-finite reconstructed energy is rejected")
    if any(value <= 0.0 for value in energies):
        raise ValueError("d=5 open domain requires E_i>0 for every particle")

    h = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        h[i][i] = energies[i] * energies[i] - masses[i] * masses[i]
    for (i, j), key in (((0, 1), "s12"), ((0, 2), "s13"), ((1, 2), "s23")):
        hij = energies[i] * energies[j] - (pairs[key] - masses[i] ** 2 - masses[j] ** 2) / 2.0
        h[i][j] = h[j][i] = hij
    if any(not math.isfinite(value) for row in h for value in row):
        raise ValueError("non-finite Gram matrix is rejected")
    h11 = h[0][0]
    h22 = h[1][1]
    h33 = h[2][2]
    h12, h13, h23 = h[0][1], h[0][2], h[1][2]
    minor2 = h11 * h22 - h12 * h12
    det_h = (
        h11 * h22 * h33 + 2.0 * h12 * h13 * h23
        - h11 * h23 * h23 - h22 * h13 * h13 - h33 * h12 * h12
    )
    if not math.isfinite(det_h) or not math.isfinite(minor2):
        raise ValueError("non-finite Gram determinant is rejected")
    if h11 <= 0.0:
        raise ValueError("d=5 open domain requires H_11>0; Gram boundary is rejected")
    if minor2 <= 0.0:
        raise ValueError("d=5 open domain requires H_11 H_22-H_12^2>0; Gram boundary is rejected")
    if det_h <= 0.0:
        raise ValueError("d=5 open domain requires det(H)>0; Gram boundary is rejected")

    k1 = math.sqrt(h11)
    z2 = h12 / k1
    x2_sq = h22 - z2 * z2
    if x2_sq <= 0.0:
        raise ValueError("d=5 open body frame requires x_2^2>0; Gram boundary is rejected")
    x2 = math.sqrt(x2_sq)
    z3 = h13 / k1
    x3 = (h23 - z2 * z3) / x2
    if isinstance(branch, bool) or not isinstance(branch, int) or branch not in (-1, 1):
        raise ValueError("branch must be +1 or -1")
    sigma = 1 if branch == 1 else -1
    y3 = sigma * math.sqrt(det_h) / (k1 * x2)
    body = (
        (0.0, 0.0, k1),
        (x2, 0.0, z2),
        (x3, y3, z3),
    )
    body4 = tuple(-sum(body[i][axis] for i in range(3)) for axis in range(3))
    body_all = body + (body4,)
    energies_all = energies

    ctheta, stheta = math.cos(theta), math.sin(theta)
    cphi, sphi = math.cos(phi), math.sin(phi)
    cpsi, spsi = math.cos(psi), math.sin(psi)
    rz_phi = ((cphi, -sphi, 0.0), (sphi, cphi, 0.0), (0.0, 0.0, 1.0))
    ry_theta = ((ctheta, 0.0, stheta), (0.0, 1.0, 0.0), (-stheta, 0.0, ctheta))
    rz_psi = ((cpsi, -spsi, 0.0), (spsi, cpsi, 0.0), (0.0, 0.0, 1.0))

    def matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3)) for i in range(3))

    rotation = matmul(matmul(rz_phi, ry_theta), rz_psi)
    rotated: list[Vector4] = []
    for particle, vector in enumerate(body_all):
        lab = tuple(sum(rotation[i][j] * vector[j] for j in range(3)) for i in range(3))
        rotated.append((energies_all[particle], lab[0], lab[1], lab[2]))

    cosines = {
        "c12": (masses[0] ** 2 + masses[1] ** 2 + 2.0 * e1 * e2 - t) / (2.0 * math.sqrt(h11 * h22)),
        "c13": (masses[0] ** 2 + masses[2] ** 2 + 2.0 * e1 * e3 - v) / (2.0 * math.sqrt(h11 * h33)),
    }
    k1_mag, k2_mag, k3_mag = math.sqrt(h11), math.sqrt(h22), math.sqrt(h33)
    cosines["c23"] = (
        e4 * e4 - masses[3] ** 2 - h11 - h22 - h33
        - 2.0 * k1_mag * k2_mag * cosines["c12"]
        - 2.0 * k1_mag * k3_mag * cosines["c13"]
    ) / (2.0 * k2_mag * k3_mag)
    result_info = {
        "x": x,
        "pairs": pairs,
        "energies": energies,
        "H": tuple(tuple(row) for row in h),
        "det_H": det_h,
        "minor2": minor2,
        "cosines": cosines,
        "body_momenta": body_all,
        "branch": sigma,
        "theta": theta,
    }
    return tuple(rotated), result_info


def reconstruct_d5_momenta(
    target_variables: Sequence[str],
    values: Mapping[str, float] | Sequence[float],
    masses: Sequence[float],
    total_mass: float,
    branch: int = 1,
) -> tuple[Vector4, ...]:
    """Reconstruct p1..p4; branch is the explicit sigma=+1/-1 Gram root."""

    subsets = _parse_labels(target_variables)
    chart_data = _chart_data_from_subsets(subsets, target_variables)
    momenta, _ = _kinematics(chart_data, target_variables, values, masses, total_mass, branch=branch)
    return momenta


def _density(
    chart_data: Mapping[str, Any],
    target_variables: Sequence[str],
    values: Mapping[str, float] | Sequence[float],
    masses: Sequence[float],
    total_mass: float,
    normalization: str,
    density_mode: str,
) -> float:
    _, info = _kinematics(chart_data, target_variables, values, masses, total_mass)
    det_h = info["det_H"]
    determinant = float(Fraction(chart_data["abs_det_L"]))
    power = 12 if normalization == "jing_psf" else 8
    multiplier = 2.0 if density_mode == "summed" else 1.0
    denominator = (
        512.0 * (2.0 * math.pi) ** power * total_mass * total_mass * total_mass
        * math.sqrt(det_h) * determinant
    )
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("d=5 direct density denominator is outside finite positive floating-point range")
    result = multiplier / denominator
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("d=5 direct density is non-finite for the supplied open-domain point")
    return result


def build_d5_formula(card: FormulaInputCard, options: FormulaOptions | None = None) -> DirectFormula:
    """Build a ``DirectFormula`` for one accepted four-body d=5 card."""

    _, chart_data = _checked_card(card)
    options = options or FormulaOptions(normalization=card.measure_convention)
    if options.normalization != card.measure_convention:
        raise ValueError("FormulaOptions.normalization disagrees with the FormulaInputCard measure convention")
    target_variables = tuple(chart_data["ordered_s"])
    coordinate_map = CoordinateMap(
        source=D5_BASIS,
        target=target_variables,
        forward=tuple(chart_data["forward_equations"]),
        inverse=tuple(chart_data["inverse_equations"]),
        jacobian=(
            f"d^5S=abs(det(L))d^5X={chart_data['abs_det_L']}d^5X; "
            f"rho_S=rho_X*abs(det(L))^(-1)={chart_data['abs_det_L']}^(-1)rho_X"
        ),
    )
    base = (
        f"chi_open(S)/(512*(2*pi)^{12 if options.normalization == 'jing_psf' else 8}*"
        f"M^3*sqrt(det(H(X(S))))*{chart_data['abs_det_L']})"
    )
    density_expression = f"2*{base}" if options.density_mode == "summed" else base
    branches = tuple(
        BranchRecord(
            key=f"sigma={sigma:+d}",
            equations=(
                "f1=s234-M^2-m_1^2+2M E_1=0",
                "f2=s34-M^2-s12+2M(E_1+E_2)=0",
                "f3=s24-M^2-s13+2M(E_1+E_3)=0",
                "f4=s12-m_1^2-m_2^2-2E_1E_2+2k_1k_2 c12=0",
                "f5=s13-m_1^2-m_3^2-2E_1E_3+2k_1k_3 c13=0",
                "F=(M-E_1-E_2-E_3)^2-m_4^2-k_1^2-k_2^2-k_3^2-2k_1k_2c12-2k_1k_3c13-2k_2k_3c23=0",
            ),
            root_condition=(
                f"{chart_data['domain'][0]}; {chart_data['domain'][1]}; "
                f"k_{{3y}}={sigma:+d}sqrt(det(H))/(k_1*x_2)"
            ),
            back_substitution=(
                tuple(chart_data["root"]["energies"])
                + tuple(chart_data["root"]["cosines"])
                + tuple(chart_data["root"]["body_frame_momenta"])
                + (chart_data["root"]["rotation"],)
            ),
            domain=tuple(chart_data["domain"]),
            multiplicity=1,
            coefficient="one mirror branch",
            summed=options.density_mode == "summed",
        )
        for sigma in (1, -1)
    )

    def numeric_density(values: Mapping[str, float] | Sequence[float], masses: Sequence[float], total_mass: float) -> float:
        return _density(chart_data, target_variables, values, masses, total_mass, options.normalization, options.density_mode)

    def numeric_momenta(
        values: Mapping[str, float] | Sequence[float],
        masses: Sequence[float],
        total_mass: float,
        branch: int = 1,
    ) -> tuple[Vector4, ...]:
        return _kinematics(chart_data, target_variables, values, masses, total_mass, branch=branch)[0]

    diagnostics = (
        "built direct d=5 formula; the chart status is built and does not imply independent verification",
        "numeric density rejects non-finite inputs, physical-domain points outside the strict open Gram domain, and Euler boundaries",
        "density_mode=summed multiplies the density by two while retaining both sigma momentum branches",
    )
    return DirectFormula(
        profile="direct_gram_d5",
        target_variables=target_variables,
        angle_variables=D5_ANGLE_VARIABLES,
        density_expression=density_expression,
        momentum_expressions=(
            tuple(chart_data["root"]["energies"])
            + tuple(chart_data["root"]["magnitudes"])
            + tuple(chart_data["gram_matrix"])
            + tuple(chart_data["root"]["cosines"])
            + tuple(chart_data["root"]["body_frame_momenta"])
            + (chart_data["root"]["rotation"],)
        ),
        jacobian=(
            f"J_scalar=(2M)^3*(2k_1k_2)*(2k_1k_3)*(2k_2k_3); "
            f"dOmega_1 dOmega_2 dOmega_3=dR dc12 dc13 dc23/sqrt(Delta); "
            f"sqrt(det(H))=k_1k_2k_3 sqrt(Delta); d^5X/d^5S=abs(det(L))^(-1)={chart_data['abs_det_L']}^(-1)"
        ),
        branches=branches,
        domain=tuple(chart_data["domain"]),
        status="built",
        coordinate_map=coordinate_map,
        diagnostics=diagnostics,
        normalization=options.normalization,
        differential_measure=D5_DIFFERENTIAL_MEASURE,
        angle_definitions=D5_ANGLE_DEFINITIONS,
        numeric_density=numeric_density,
        numeric_momenta=numeric_momenta,
    )


__all__ = [
    "D5_BASIS",
    "D5_ANGLE_VARIABLES",
    "D5_ANGLE_DEFINITIONS",
    "D5_DIFFERENTIAL_MEASURE",
    "D5_DIFFERENTIAL_MEASURE_LATEX",
    "build_d5_chart_data",
    "build_d5_formula",
    "reconstruct_d5_momenta",
]
