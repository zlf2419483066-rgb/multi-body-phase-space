"""Independent sequential reference chart used by the formula benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from ..formula_inputs import FormulaInputCard
from .inputs import validate_formula_card
from .kinematic_domain import kallen, kallen_root, validate_masses, validate_pair_domain
from .records import BridgeFormula, CoordinateMap, FormulaOptions

Vector4 = tuple[float, float, float, float]

ANGLE_VARIABLES = ("theta0", "phi0", "theta12", "phi12", "theta34", "phi34")
ANGLE_DEFINITIONS = (
    "(theta0, phi0) is the direction of the 12 auxiliary subsystem in the parent rest frame; the 34 subsystem is opposite.",
    "(theta12, phi12) and (theta34, phi34) are first-particle directions in the corresponding subsystem rest frames before the no-rotation boosts.",
    "The reference path uses three solid angles and is an internal sequential benchmark; it does not claim a separate Yu 3+3 angular chart.",
)
DIFFERENTIAL_MEASURE = (
    "dPhi4 = rho(s) dsA dsB dOmega0 dOmegaA dOmegaB, "
    "where dOmega=sin(theta)dtheta dphi; rho is the coefficient per complete measure."
)


@dataclass(frozen=True, slots=True)
class ReferenceChart:
    profile: str
    variables: tuple[str, ...]
    domain: tuple[str, ...]
    status: str = "built"
    coordinate_map: CoordinateMap | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "variables": list(self.variables),
            "domain": list(self.domain),
            "status": self.status,
            "coordinate_map": self.coordinate_map.to_dict() if self.coordinate_map else None,
        }


def build_reference_chart(card: FormulaInputCard, *, options: FormulaOptions | None = None) -> ReferenceChart:
    if not isinstance(card, FormulaInputCard):
        raise TypeError("build_reference_chart expects a FormulaInputCard")
    validate_formula_card(card)
    if options is not None and options.normalization != card.measure_convention:
        raise ValueError(
            "FormulaOptions.normalization disagrees with the FormulaInputCard measure convention"
        )
    variables = ("s12", "s34", "theta0", "phi0", "theta12", "phi12", "theta34", "phi34")
    domain = (
        "s12 >= (m_1+m_2)^2; s34 >= (m_3+m_4)^2",
        "sqrt(s12)+sqrt(s34) <= M",
        "positive auxiliary subsystem invariants; threshold equalities are zero-density boundaries",
        "0 <= theta <= pi; 0 <= phi < 2*pi",
    )
    cmap = CoordinateMap(variables, variables, tuple(f"{x} -> {x}" for x in variables), tuple(f"{x} -> {x}" for x in variables), "1")
    return ReferenceChart("sequential_12_34", variables, domain, "built", cmap)


def _read(chart: ReferenceChart, values: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    if isinstance(values, Mapping):
        missing = [n for n in chart.variables if n not in values]
        if missing:
            raise ValueError(f"missing reference coordinates: {missing}")
        return {n: float(values[n]) for n in chart.variables}
    try:
        values = tuple(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference coordinates must be a mapping or a finite sequence") from exc
    if len(values) != len(chart.variables):
        raise ValueError(f"expected {len(chart.variables)} reference coordinates")
    return dict(zip(chart.variables, map(float, values)))


def _lam(a: float, b: float, c: float) -> float:
    return kallen(a, b, c)


def _n(theta: float, phi: float) -> tuple[float, float, float]:
    return math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)


def _boost(p: Vector4, beta: tuple[float, float, float]) -> Vector4:
    b2 = sum(x*x for x in beta)
    if b2 < 1e-30:
        return p
    gamma = 1/math.sqrt(1-b2)
    dot = sum(beta[i]*p[i+1] for i in range(3))
    fac = (gamma-1)*dot/b2 + gamma*p[0]
    return gamma*(p[0]+dot), p[1]+fac*beta[0], p[2]+fac*beta[1], p[3]+fac*beta[2]


def reconstruct_reference_momenta(chart: ReferenceChart, values: Mapping[str, float] | Sequence[float], masses: Sequence[float], total_mass: float) -> tuple[Vector4, ...]:
    if chart.status != "built":
        raise ValueError("reference chart is unavailable")
    v = _read(chart, values)
    masses, total_mass = validate_masses(masses, total_mass)
    validate_pair_domain(v, ("s12", "s34"), masses, total_mass, ANGLE_VARIABLES)
    s12, s34 = v["s12"], v["s34"]
    m12, m34 = math.sqrt(s12), math.sqrt(s34)
    lp = _lam(total_mass**2, s12, s34)
    q = kallen_root(lp, total_mass, m12, m34)/(2*total_mass)
    e12 = (total_mass**2+s12-s34)/(2*total_mass)
    e34 = (total_mass**2+s34-s12)/(2*total_mass)
    n0 = _n(v["theta0"],v["phi0"])
    q12=(e12,q*n0[0],q*n0[1],q*n0[2]); q34=(e34,-q*n0[0],-q*n0[1],-q*n0[2])
    result: dict[int,Vector4] = {}
    for pair,parent,tkey,pkey in (((1,2),q12,"theta12","phi12"),((3,4),q34,"theta34","phi34")):
        i,j=pair; mi,mj=masses[i-1],masses[j-1]
        # parent[0] is the moving subsystem energy; use sqrt(s_pair) for
        # its rest-frame decay and retain parent[0] only for the boost.
        parent_mass = m12 if pair == (1,2) else m34
        ld=_lam(parent_mass**2,mi*mi,mj*mj)
        k=kallen_root(ld, parent_mass, mi, mj)/(2*parent_mass)
        ei=(parent_mass**2+mi*mi-mj*mj)/(2*parent_mass); ej=(parent_mass**2+mj*mj-mi*mi)/(2*parent_mass)
        ni=_n(v[tkey],v[pkey]); pi=(ei,k*ni[0],k*ni[1],k*ni[2]); pj=(ej,-k*ni[0],-k*ni[1],-k*ni[2])
        beta=(parent[1]/parent[0],parent[2]/parent[0],parent[3]/parent[0])
        result[i]=_boost(pi,beta); result[j]=_boost(pj,beta)
    return tuple(result[i] for i in range(1,5))


def _rho(v: Mapping[str,float], masses: Sequence[float], total_mass: float, convention: str) -> float:
    masses, total_mass = validate_masses(masses, total_mass)
    v = validate_pair_domain(v, ("s12", "s34"), masses, total_mass, ANGLE_VARIABLES)
    def two(parent: float, a: float, b: float) -> float:
        lam = _lam(parent, a*a, b*b)
        return kallen_root(lam, math.sqrt(parent), a, b)/(32*math.pi**2*parent)
    value=two(total_mass**2,math.sqrt(v["s12"]),math.sqrt(v["s34"]))
    value*=two(v["s12"],masses[0],masses[1])*two(v["s34"],masses[2],masses[3])/(2*math.pi)**2
    result = value/(2*math.pi)**4 if convention=="jing_psf" else value
    if not math.isfinite(result):
        raise ValueError("reference density is not finite for the supplied physical point")
    return result


def _reference_density_expression() -> str:
    return (
        "rho_ref(x; t12=s12^(x), t34=s34^(x)) = "
        "[sqrt(lambda(M^2,t12,t34))/(32*pi^2*M^2)] "
        "*[sqrt(lambda(t12,m_1^2,m_2^2))/(32*pi^2*t12)] "
        "*[sqrt(lambda(t34,m_3^2,m_4^2))/(32*pi^2*t34)]/(2*pi)^2"
    )


def _reference_momentum_expressions() -> tuple[str, ...]:
    result = []
    parent_definitions = (
        "Q12=(E12^P,+k_M n(theta0,phi0)); Q34=(E34^P,-k_M n(theta0,phi0)); "
        "E12^P=(M^2+t12-t34)/(2*M); E34^P=(M^2+t34-t12)/(2*M); "
        "k_M=sqrt(lambda(M^2,t12,t34))/(2*M); "
        "beta12=+k_M/E12^P n(theta0,phi0); beta34=-k_M/E34^P n(theta0,phi0)"
    )
    for particle, other, angle_theta, angle_phi, subsystem, beta, direction in (
        (1, 2, "theta12", "phi12", "t12", "beta12", "+"),
        (2, 1, "theta12", "phi12", "t12", "beta12", "-"),
        (3, 4, "theta34", "phi34", "t34", "beta34", "+"),
        (4, 3, "theta34", "phi34", "t34", "beta34", "-"),
    ):
        result.append(
            f"p_{particle}(x)=Boost_{{{beta}}}((E_{particle}^{{{subsystem}}}, "
            f"{direction}k_{{{subsystem}}} n({angle_theta},{angle_phi}))); "
            f"E_{particle}^{{{subsystem}}}=({subsystem}+m_{particle}^2-m_{other}^2)/(2*sqrt({subsystem})); "
            f"k_{{{subsystem}}}=sqrt(lambda({subsystem},m_{particle}^2,m_{other}^2))/(2*sqrt({subsystem})); "
            f"{parent_definitions}"
        )
    return tuple(result)


def _bridge_constraints(card: FormulaInputCard) -> tuple[str, ...]:
    import re

    constraints = [
        "reference local auxiliary coordinates x=(s12^(x),s34^(x),Omega0,Omega12,Omega34); target graph variables are local to each delta",
    ]
    for item in card.double_measures:
        momentum_x = re.sub(r"p_(\d+)", r"p_\1(x)", item.momentum_expression)
        constraints.append(
            f"edge {item.edge_index + 1}, occurrence {item.occurrence + 1}: "
            f"ds_{{{''.join(str(index) for index in item.variable)}}} "
            f"delta({item.variable_label} - ({momentum_x})^2)"
        )
    return tuple(constraints)


def build_bridge_formula(card: FormulaInputCard, chart: ReferenceChart, *, options: FormulaOptions | None = None) -> BridgeFormula:
    options = options or FormulaOptions(normalization=card.measure_convention)
    validate_formula_card(card)
    if options.normalization != card.measure_convention:
        raise ValueError(
            "FormulaOptions.normalization disagrees with the FormulaInputCard measure convention"
        )
    if chart.status != "built" or chart.profile != "sequential_12_34" or tuple(chart.variables) != (
        "s12", "s34", "theta0", "phi0", "theta12", "phi12", "theta34", "phi34"
    ):
        raise ValueError("reference chart is not bound to the supported sequential profile")
    expr = _reference_density_expression()
    if options.normalization == "jing_psf":
        expr += " /(2*pi)^4"
    return BridgeFormula(
        profile=chart.profile, variables=chart.variables, density_expression=expr,
        momentum_expressions=_reference_momentum_expressions(),
        constraints=_bridge_constraints(card) + chart.domain,
        status="built", coordinate_map=chart.coordinate_map,
        normalization=options.normalization,
        differential_measure=DIFFERENTIAL_MEASURE,
        angle_definitions=ANGLE_DEFINITIONS,
        numeric_density=lambda values,m,total_mass: _rho(_read(chart,values),m,total_mass,options.normalization),
        numeric_momenta=lambda values,m,total_mass: reconstruct_reference_momenta(chart,values,m,total_mass),
    )


def build_reference_density(card: FormulaInputCard, chart: ReferenceChart, *, options: FormulaOptions | None = None) -> BridgeFormula:
    return build_bridge_formula(card, chart, options=options)


__all__=["ReferenceChart","build_reference_chart","reconstruct_reference_momenta","build_bridge_formula","build_reference_density"]
