"""Finite direct IKV charts for the pair and d=5 Gram profiles."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from ..formula_inputs import FormulaInputCard
from .inputs import validate_formula_card
from .kinematic_domain import kallen, kallen_root, validate_masses, validate_pair_domain
from .records import BranchRecord, CoordinateMap, DirectFormula, FormulaOptions
from .d5_direct import (
    D5_ANGLE_VARIABLES,
    build_d5_chart_data,
    build_d5_formula,
    reconstruct_d5_momenta,
)

Vector4 = tuple[float, float, float, float]

ANGLE_VARIABLES = ("theta0", "phi0", "thetaA", "phiA", "thetaB", "phiB")
ANGLE_DEFINITIONS = (
    "(theta0, phi0) is the direction of subsystem A in the parent rest frame; subsystem B is opposite.",
    "(thetaA, phiA) is the first listed particle direction in subsystem A rest frame before the no-rotation boost.",
    "(thetaB, phiB) is the first listed particle direction in subsystem B rest frame before the no-rotation boost.",
    "The six angles are three solid angles; this profile does not implement a separate Yu 3+3 angular chart.",
)
DIFFERENTIAL_MEASURE = (
    "dPhi4 = rho(s) dsA dsB dOmega0 dOmegaA dOmegaB, "
    "where dOmega=sin(theta)dtheta dphi; rho is the coefficient per complete measure."
)


@dataclass(frozen=True, slots=True)
class DirectChart:
    profile: str
    target_variables: tuple[str, ...]
    angle_variables: tuple[str, ...]
    domain: tuple[str, ...]
    status: str
    coordinate_map: CoordinateMap | None = None
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "target_variables": list(self.target_variables),
            "angle_variables": list(self.angle_variables),
            "domain": list(self.domain),
            "status": self.status,
            "coordinate_map": self.coordinate_map.to_dict() if self.coordinate_map else None,
            "diagnostics": list(self.diagnostics),
        }


def _pairs(variables: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    normalized = tuple(tuple(v) for v in variables)
    if len(normalized) != 2 or any(len(v) != 2 for v in normalized):
        return None
    if set(normalized[0]).isdisjoint(normalized[1]) and set(normalized[0]) | set(normalized[1]) == {1, 2, 3, 4}:
        return normalized[0], normalized[1]
    return None


def select_direct_chart(card: FormulaInputCard, *, options: FormulaOptions | None = None) -> DirectChart:
    if not isinstance(card, FormulaInputCard):
        raise TypeError("select_direct_chart expects a FormulaInputCard")
    validate_formula_card(card)
    if options is not None and options.normalization != card.measure_convention:
        raise ValueError(
            "FormulaOptions.normalization disagrees with the FormulaInputCard measure convention"
        )
    variables = tuple(card.variable_labels)
    if card.loop_count == 3 and len(card.independent_variables) == 5:
        data = build_d5_chart_data(card)
        cmap = CoordinateMap(
            source=tuple(data["basis_variables"]),
            target=variables,
            forward=tuple(data["forward_equations"]),
            inverse=tuple(data["inverse_equations"]),
            jacobian=(
                f"d^5S=abs(det(L))d^5X={data['abs_det_L']}d^5X; "
                f"d^5X=abs(det(L))^(-1)d^5S"
            ),
        )
        return DirectChart(
            "direct_gram_d5",
            variables,
            D5_ANGLE_VARIABLES,
            tuple(data["domain"]),
            "built",
            cmap,
            diagnostics=("built direct d=5 chart; independent formula verification is pending",),
        )
    pair = _pairs(card.independent_variables)
    angles = ANGLE_VARIABLES
    if pair is None:
        return DirectChart(
            "unsupported",
            variables,
            angles,
            (),
            "unsupported",
            diagnostics=("direct analytic chart currently covers two disjoint pair invariants only",),
        )
    names = tuple("s" + "".join(map(str, group)) for group in pair)
    cmap = CoordinateMap(
        source=names + angles,
        target=names + angles,
        forward=tuple(f"{name} -> {name}" for name in names + angles),
        inverse=tuple(f"{name} -> {name}" for name in names + angles),
        jacobian="1",
    )
    first, second = names
    i, j = _pair_indices(first)
    k, l = _pair_indices(second)
    domain = (
        f"{first} >= (m_{i} + m_{j})^2; {second} >= (m_{k} + m_{l})^2",
        f"sqrt({first}) + sqrt({second}) <= M",
        "positive subsystem invariants; threshold equalities are zero-density boundaries",
        "0 <= theta <= pi; 0 <= phi < 2*pi",
    )
    return DirectChart("direct_pair_recursive", names, angles, domain, "built", cmap)


def _lambda(a: float, b: float, c: float) -> float:
    return kallen(a, b, c)


def _unit(theta: float, phi: float) -> tuple[float, float, float]:
    return math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)


def _boost(p: Vector4, beta: tuple[float, float, float]) -> Vector4:
    bx, by, bz = beta
    b2 = bx * bx + by * by + bz * bz
    if b2 < 1e-30:
        return p
    gamma = 1.0 / math.sqrt(1.0 - b2)
    dot = bx * p[1] + by * p[2] + bz * p[3]
    factor = ((gamma - 1.0) * dot / b2 + gamma * p[0])
    return (gamma * (p[0] + dot), p[1] + factor * bx, p[2] + factor * by, p[3] + factor * bz)


def _read_values(chart: DirectChart, values: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    if isinstance(values, Mapping):
        missing = [name for name in chart.target_variables + chart.angle_variables if name not in values]
        if missing:
            raise ValueError(f"missing chart coordinates: {missing}")
        return {name: float(values[name]) for name in chart.target_variables + chart.angle_variables}
    try:
        values = tuple(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("chart coordinates must be a mapping or a finite sequence") from exc
    names = chart.target_variables + chart.angle_variables
    if len(values) != len(names):
        raise ValueError(f"expected {len(names)} chart coordinates, got {len(values)}")
    return dict(zip(names, map(float, values)))


def reconstruct_direct_momenta(
    chart: DirectChart,
    values: Mapping[str, float] | Sequence[float],
    masses: Sequence[float],
    total_mass: float,
    branch: int = 1,
) -> tuple[Vector4, ...]:
    """Return p1..p4 in the parent rest frame; d=5 accepts branch=+1/-1."""

    if chart.status != "built":
        raise ValueError("the selected direct chart is unsupported")
    if chart.profile == "direct_gram_d5":
        if tuple(chart.angle_variables) != D5_ANGLE_VARIABLES:
            raise ValueError("the selected d=5 direct chart has inconsistent Euler coordinates")
        return reconstruct_d5_momenta(
            chart.target_variables, values, masses, total_mass, branch=branch
        )
    if chart.profile != "direct_pair_recursive":
        raise ValueError("the selected direct chart is unsupported")
    data = _read_values(chart, values)
    masses, total_mass = validate_masses(masses, total_mass)
    validate_pair_domain(data, chart.target_variables, masses, total_mass, chart.angle_variables)
    pair_a = tuple(int(x) for x in chart.target_variables[0][1:])
    pair_b = tuple(int(x) for x in chart.target_variables[1][1:])
    sa, sb = data[chart.target_variables[0]], data[chart.target_variables[1]]
    ma = math.sqrt(sa)
    mb = math.sqrt(sb)
    lam_parent = _lambda(total_mass * total_mass, sa, sb)
    p0 = kallen_root(lam_parent, total_mass, ma, mb) / (2.0 * total_mass)
    ea = (total_mass * total_mass + sa - sb) / (2.0 * total_mass)
    eb = (total_mass * total_mass + sb - sa) / (2.0 * total_mass)
    n0 = _unit(data["theta0"], data["phi0"])
    qa = (ea, p0*n0[0], p0*n0[1], p0*n0[2])
    qb = (eb, -p0*n0[0], -p0*n0[1], -p0*n0[2])
    out: dict[int, Vector4] = {}
    for group, parent, theta_key, phi_key in ((pair_a, qa, "thetaA", "phiA"), (pair_b, qb, "thetaB", "phiB")):
        i, j = group
        mi, mj = masses[i-1], masses[j-1]
        # parent[0] is the energy of a moving subsystem in the parent rest
        # frame.  The two-body decay formula requires that subsystem's rest
        # mass, while the boost below still uses its lab-frame energy.
        parent_mass = math.sqrt(sa if group == pair_a else sb)
        lam = _lambda(parent_mass * parent_mass, mi * mi, mj * mj)
        q = kallen_root(lam, parent_mass, mi, mj) / (2.0 * parent_mass)
        ei = (parent_mass * parent_mass + mi * mi - mj * mj) / (2.0 * parent_mass)
        ej = (parent_mass * parent_mass + mj * mj - mi * mi) / (2.0 * parent_mass)
        n = _unit(data[theta_key], data[phi_key])
        pi_rest = (ei, q*n[0], q*n[1], q*n[2])
        pj_rest = (ej, -q*n[0], -q*n[1], -q*n[2])
        beta = (parent[1]/parent[0], parent[2]/parent[0], parent[3]/parent[0])
        out[i] = _boost(pi_rest, beta)
        out[j] = _boost(pj_rest, beta)
    return tuple(out[index] for index in range(1, 5))


def _density(value: Mapping[str, float], masses: Sequence[float], total_mass: float, convention: str, pair_names: tuple[str, str]) -> float:
    masses, total_mass = validate_masses(masses, total_mass)
    value = validate_pair_domain(value, pair_names, masses, total_mass, ANGLE_VARIABLES)
    sa, sb = value[pair_names[0]], value[pair_names[1]]
    def two(parent_s: float, first: float, second: float) -> float:
        lam = _lambda(parent_s, first * first, second * second)
        return kallen_root(lam, math.sqrt(parent_s), first, second) / (32.0 * math.pi**2 * parent_s)
    pair_a = tuple(int(x) for x in pair_names[0][1:])
    pair_b = tuple(int(x) for x in pair_names[1][1:])
    rho = two(total_mass**2, math.sqrt(sa), math.sqrt(sb)) * two(sa, masses[pair_a[0]-1], masses[pair_a[1]-1]) * two(sb, masses[pair_b[0]-1], masses[pair_b[1]-1]) / (2.0*math.pi)**2
    result = rho / (2.0*math.pi)**4 if convention == "jing_psf" else rho
    if not math.isfinite(result):
        raise ValueError("direct density is not finite for the supplied physical point")
    return result


def _pair_indices(name: str) -> tuple[int, int]:
    return tuple(int(item) for item in name[1:])  # type: ignore[return-value]


def _momentum_expressions(pair_names: tuple[str, str]) -> tuple[str, ...]:
    first, second = pair_names
    i, j = _pair_indices(first)
    k, l = _pair_indices(second)
    parent_definitions = (
        f"Q_{first}=(E_{first}^P,+k_M n(theta0,phi0)); "
        f"Q_{second}=(E_{second}^P,-k_M n(theta0,phi0)); "
        f"E_{first}^P=(M^2+{first}-{second})/(2*M); "
        f"E_{second}^P=(M^2+{second}-{first})/(2*M); "
        f"k_M=sqrt(lambda(M^2,{first},{second}))/(2*M); "
        f"beta_{first}=+k_M/E_{first}^P n(theta0,phi0); "
        f"beta_{second}=-k_M/E_{second}^P n(theta0,phi0)"
    )
    result = []
    for particle, other, pair, theta, phi, beta, direction in (
        (i, j, first, "thetaA", "phiA", f"beta_{first}", "+"),
        (j, i, first, "thetaA", "phiA", f"beta_{first}", "-"),
        (k, l, second, "thetaB", "phiB", f"beta_{second}", "+"),
        (l, k, second, "thetaB", "phiB", f"beta_{second}", "-"),
    ):
        result.append(
            f"p_{particle}(y)=Boost_{{{beta}}}((E_{particle}^{{{pair}}}, "
            f"{direction}k_{{{pair}}} n({theta},{phi}))); "
            f"E_{particle}^{{{pair}}}=({pair}+m_{particle}^2-m_{other}^2)/(2*sqrt({pair})); "
            f"k_{{{pair}}}=sqrt(lambda({pair},m_{particle}^2,m_{other}^2))/(2*sqrt({pair})); "
            f"{parent_definitions}"
        )
    return tuple(result)


def _density_expression(pair_names: tuple[str, str]) -> str:
    first, second = pair_names
    i, j = _pair_indices(first)
    k, l = _pair_indices(second)
    return (
        f"rho_direct({first},{second}) = "
        f"[sqrt(lambda(M^2,{first},{second}))/(32*pi^2*M^2)] "
        f"*[sqrt(lambda({first},m_{i}^2,m_{j}^2))/(32*pi^2*{first})] "
        f"*[sqrt(lambda({second},m_{k}^2,m_{l}^2))/(32*pi^2*{second})]/(2*pi)^2"
    )


def build_direct_formula(card: FormulaInputCard, chart: DirectChart, *, options: FormulaOptions | None = None) -> DirectFormula:
    options = options or FormulaOptions(normalization=card.measure_convention)
    validate_formula_card(card)
    if options.normalization != card.measure_convention:
        raise ValueError(
            "FormulaOptions.normalization disagrees with the FormulaInputCard measure convention"
        )
    if chart.profile == "direct_gram_d5":
        if chart.status != "built" or tuple(chart.target_variables) != tuple(card.variable_labels):
            raise ValueError("direct d=5 chart is not bound to the supplied FormulaInputCard")
        if tuple(chart.angle_variables) != D5_ANGLE_VARIABLES:
            raise ValueError("direct d=5 chart has inconsistent Euler coordinates")
        return build_d5_formula(card, options=options)
    expected_pair = _pairs(card.independent_variables)
    expected_names = tuple("s" + "".join(map(str, group)) for group in expected_pair) if expected_pair else ()
    if chart.status == "built":
        if chart.profile != "direct_pair_recursive" or tuple(chart.target_variables) != expected_names or tuple(chart.angle_variables) != ANGLE_VARIABLES:
            raise ValueError("direct chart is not bound to the supplied FormulaInputCard")
    if chart.status != "built":
        return DirectFormula(
            profile=chart.profile,
            target_variables=chart.target_variables,
            angle_variables=chart.angle_variables,
            density_expression="",
            momentum_expressions=(),
            jacobian="",
            branches=(),
            domain=(),
            status="unsupported",
            coordinate_map=chart.coordinate_map,
            diagnostics=chart.diagnostics,
            normalization=options.normalization,
            differential_measure=DIFFERENTIAL_MEASURE,
            angle_definitions=ANGLE_DEFINITIONS,
        )
    pair_names = (chart.target_variables[0], chart.target_variables[1])
    density = _density_expression(pair_names)
    if options.normalization == "jing_psf":
        density += " /(2*pi)^4"
    first, second = pair_names
    i, j = _pair_indices(first)
    k, l = _pair_indices(second)
    root_condition = (
        f"lambda(M^2,{first},{second}) >= 0; "
        f"lambda({first},m_{i}^2,m_{j}^2) >= 0; "
        f"lambda({second},m_{k}^2,m_{l}^2) >= 0; "
        f"sqrt({first})+sqrt({second}) <= M; finite positive-energy roots"
    )
    branch = BranchRecord(
        "pair-decay",
        (f"p_{i}^2=m_{i}^2", f"p_{j}^2=m_{j}^2", f"p_{k}^2=m_{k}^2", f"p_{l}^2=m_{l}^2", "sum_{r=1}^4 p_r=P"),
        root_condition,
        tuple(f"p_{particle} from explicit Q,k,E,Boost expressions" for particle in (i, j, k, l)),
        chart.domain,
        summed=options.density_mode == "summed",
    )
    return DirectFormula(
        profile=chart.profile,
        target_variables=chart.target_variables,
        angle_variables=chart.angle_variables,
        density_expression=density,
        momentum_expressions=_momentum_expressions(pair_names),
        jacobian=f"rho coefficient from three two-body factors for {first},{second}; coordinate map det=1",
        branches=(branch,),
        domain=chart.domain,
        status="built",
        coordinate_map=chart.coordinate_map,
        diagnostics=tuple(chart.diagnostics) + (
            "three solid-angle variables are retained; a separate Yu 3+3 angular conversion is not implemented",
        ),
        normalization=options.normalization,
        differential_measure=DIFFERENTIAL_MEASURE,
        angle_definitions=ANGLE_DEFINITIONS,
        numeric_density=lambda values, masses, total_mass: _density(_read_values(chart, values), masses, total_mass, options.normalization, pair_names),
        numeric_momenta=lambda values, masses, total_mass, branch=1: reconstruct_direct_momenta(
            chart, values, masses, total_mass, branch=branch
        ),
    )


def build_direct_density(card: FormulaInputCard, chart: DirectChart, *, options: FormulaOptions | None = None) -> DirectFormula:
    return build_direct_formula(card, chart, options=options)


__all__ = ["DirectChart", "select_direct_chart", "reconstruct_direct_momenta", "build_direct_formula", "build_direct_density"]
