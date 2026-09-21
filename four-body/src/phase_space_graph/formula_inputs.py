"""Formula-input records derived from complete physical labelings.

LM-12 is a boundary between the physical-record layer and a future formula
derivation layer.  It records local two-body blocks and measure bookkeeping;
it deliberately does not choose a reference frame, solve a Jacobian, choose
branches, or claim that a complete phase-space formula has been verified.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from html import escape
from typing import Any, Iterable, Mapping

from .domain import EdgeType
from .label_identity import MarkedIdentity, canonicalize_physical_labeling
from .physical_labels import NormalizationResult, PhysicalLabelRecord, Provenance, review_physical_record


FORMULA_INPUT_SCHEMA_VERSION = "IM-4-formula-v1"
MEASURE_CONVENTIONS = {
    "jing_psf": {
        "label": "资料约定 jing_psf",
        "double_line_factor": "(2π)^3 ds_S",
        "definition": "dΦ_n = δ⁴(P-Σp_i) ∏[d³p_i/((2π)³ 2E_i)]；双线因子 (2π)^3 ds_S，不在本模块转换常数。",
    },
    "standard_2pi4": {
        "label": "标准约定 standard_2pi4",
        "double_line_factor": "ds_S/(2π)",
        "definition": "dΦ_n = (2π)⁴ δ⁴(P-Σp_i) ∏[d³p_i/((2π)³ 2E_i)]；双线因子 ds_S/(2π)，不在本模块转换常数。",
    },
}
UNVERIFIED_STEPS = ("reference_frames", "angles", "measure_cancellation", "constraints_eliminated", "jacobian", "branches", "domain")


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def _fraction_text(value: Any) -> str:
    numerator = getattr(value, "numerator", value)
    denominator = getattr(value, "denominator", 1)
    return str(numerator) if denominator == 1 else f"{numerator}/{denominator}"


def _particle_symbol(particle: int) -> str:
    return f"p_{particle}"


def _vector_expression(vector: Iterable[Any]) -> str:
    """Render an exact particle-basis vector without dropping repetitions."""

    terms: list[str] = []
    for particle, coefficient in enumerate(vector, 1):
        if not coefficient:
            continue
        magnitude = abs(coefficient)
        factor = (
            _particle_symbol(particle)
            if magnitude == 1
            else f"{_fraction_text(magnitude)} p_{particle}"
        )
        if not terms:
            terms.append(f"-{factor}" if coefficient < 0 else factor)
        else:
            terms.append(("- " if coefficient < 0 else "+ ") + factor)
    return " ".join(terms) or "0"


def _support(vector: Iterable[Any]) -> tuple[int, ...]:
    return tuple(index for index, coefficient in enumerate(vector, 1) if coefficient)


def _mass_symbols(vector: Iterable[Any], n: int, *, squared: bool) -> tuple[str, str | None]:
    """Return a readable mass and its squared invariant, preserving bad vectors."""

    values = tuple(vector)
    if len(values) != n or any(value not in (0, 1) for value in values):
        expression = _vector_expression(values)
        return f"sqrt(({expression})^2)", f"({expression})^2"
    subset = _support(values)
    if not subset:
        return "0", "0"
    if len(subset) == n:
        return "M" if not squared else "M^2", "M^2"
    if len(subset) == 1:
        particle = subset[0]
        return (f"m_{particle}" if not squared else f"m_{particle}^2"), (
            f"m_{particle}^2"
        )
    label = "".join(str(item) for item in subset)
    return (f"m_{label}" if not squared else f"s_{label}"), f"s_{label}"


def _mass_for_particle(particle: int) -> tuple[str, str]:
    return f"m_{particle}", f"m_{particle}^2"


def _symbol_for_double(subset: tuple[int, ...], n: int) -> tuple[str, str | None]:
    vector = tuple(1 if index in subset else 0 for index in range(1, n + 1))
    return _mass_symbols(vector, n, squared=False)


@dataclass(frozen=True, slots=True)
class FormulaConnection:
    """One outgoing line used by a local factor, with its graph reference."""

    role: str
    edge_index: int | None
    occurrence: int | None
    source: int
    target: int | None
    momentum_expression: str
    coefficients: tuple[int, ...] | None
    mass: str
    mass_squared: str
    particle_ids: tuple[int, ...]
    slot: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "edge_index": self.edge_index,
            "occurrence": self.occurrence,
            "source": self.source,
            "target": self.target,
            "momentum_expression": self.momentum_expression,
            "coefficients": list(self.coefficients) if self.coefficients is not None else None,
            "mass": self.mass,
            "mass_squared": self.mass_squared,
            "particle_ids": list(self.particle_ids),
            "slot": self.slot,
        }


@dataclass(frozen=True, slots=True)
class TwoBodyFactor:
    """A local dPhi_2 input; this is not a completed analytic factor."""

    vertex: int
    incoming_momentum: str
    incoming_coefficients: tuple[int, ...]
    parent_mass: str
    parent_mass_squared: str
    outgoing: tuple[FormulaConnection, ...]
    connection_references: tuple[dict[str, object], ...]
    reference_frame: str | None
    angles: tuple[str, ...] | None
    expression: str
    status: str = "ready"
    diagnostics: tuple[str, ...] = ()
    incoming_references: tuple[dict[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outgoing", tuple(self.outgoing))
        object.__setattr__(self, "connection_references", tuple(dict(item) for item in self.connection_references))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def incoming_vector(self) -> tuple[int, ...]:
        """Alias exposing the integer coefficients of the total incoming Qv."""

        return self.incoming_coefficients

    @property
    def outgoing_mass_squared(self) -> tuple[str, ...]:
        return tuple(item.mass_squared for item in self.outgoing)

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex": self.vertex,
            "incoming_momentum": self.incoming_momentum,
            "incoming_coefficients": list(self.incoming_coefficients),
            "incoming_references": _jsonable(self.incoming_references),
            "incoming_recovered": _vector_expression(self.incoming_coefficients),
            "parent_mass": self.parent_mass,
            "parent_mass_squared": self.parent_mass_squared,
            "outgoing": [item.to_dict() for item in self.outgoing],
            "connection_references": _jsonable(self.connection_references),
            "reference_frame": self.reference_frame,
            "angles": list(self.angles) if self.angles is not None else None,
            "expression": self.expression,
            "status": self.status,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class DoubleMeasureInput:
    edge_index: int
    occurrence: int
    variable: tuple[int, ...]
    variable_label: str
    differential: str
    convention: str
    momentum_expression: str

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_index": self.edge_index,
            "occurrence": self.occurrence,
            "variable": list(self.variable),
            "variable_label": self.variable_label,
            "differential": self.differential,
            "convention": self.convention,
            "momentum_expression": self.momentum_expression,
        }


@dataclass(frozen=True, slots=True)
class ParticleMeasureInput:
    particle_id: int
    occurrences: tuple[tuple[str, int, int], ...]
    cancellation_count: int
    differential: str

    def to_dict(self) -> dict[str, object]:
        return {
            "particle_id": self.particle_id,
            "occurrences": [list(item) for item in self.occurrences],
            "cancellation_count": self.cancellation_count,
            "differential": self.differential,
            "retained_occurrences": [list(item) for item in self.occurrences if item[0] == "external"],
            "cancellation_occurrences": [list(item) for item in self.occurrences if item[0] == "internal"],
        }


@dataclass(frozen=True, slots=True)
class FormulaInputCard:
    """Machine-readable input card for a future formula derivation."""

    schema_version: str
    status: str
    availability: str
    unavailable_reason: str | None
    n: int | None
    loop_count: int | None
    mode: str | None
    measure_convention: str
    measure_definition: str
    factors: tuple[TwoBodyFactor, ...]
    double_measures: tuple[DoubleMeasureInput, ...]
    particle_measures: tuple[ParticleMeasureInput, ...]
    independent_variables: tuple[tuple[int, ...], ...]
    variable_labels: tuple[str, ...]
    independence: dict[str, object] | None
    constraints: tuple[str, ...]
    formula_status: dict[str, bool]
    missing_steps: tuple[str, ...]
    diagnostics: tuple[dict[str, object], ...]
    identity: dict[str, object] | None
    record: dict[str, object] | None
    provenance: dict[str, object] | None

    def __post_init__(self) -> None:
        if self.schema_version != FORMULA_INPUT_SCHEMA_VERSION:
            raise ValueError(f"unsupported formula-input schema: {self.schema_version!r}")
        if self.measure_convention not in MEASURE_CONVENTIONS:
            raise ValueError(f"unknown measure convention: {self.measure_convention!r}")
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "double_measures", tuple(self.double_measures))
        object.__setattr__(self, "particle_measures", tuple(self.particle_measures))
        object.__setattr__(self, "independent_variables", tuple(tuple(item) for item in self.independent_variables))
        object.__setattr__(self, "variable_labels", tuple(self.variable_labels))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "missing_steps", tuple(self.missing_steps))
        object.__setattr__(self, "diagnostics", tuple(dict(item) for item in self.diagnostics))
        object.__setattr__(self, "formula_status", dict(self.formula_status))

    @property
    def factors_ready(self) -> bool:
        return bool(self.formula_status.get("factors_ready", False))

    @property
    def formula_verified(self) -> bool:
        return bool(self.formula_status.get("formula_verified", False))

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "formula_input_card",
            "schema_version": self.schema_version,
            "status": {
                "stage": self.status,
                "availability": self.availability,
                "unavailable_reason": self.unavailable_reason,
                "n": self.n,
                "loop_count": self.loop_count,
                "mode": self.mode,
                "measure_convention": self.measure_convention,
                "measure_definition": self.measure_definition,
            },
            "factors": [item.to_dict() for item in self.factors],
            "double_measures": [item.to_dict() for item in self.double_measures],
            "particle_measures": [item.to_dict() for item in self.particle_measures],
            "independent_variables": [list(item) for item in self.independent_variables],
            "variable_labels": list(self.variable_labels),
            "independence": _jsonable(self.independence),
            "constraints": list(self.constraints),
            "formula_status": dict(self.formula_status),
            "missing_steps": list(self.missing_steps),
            "diagnostics": _jsonable(self.diagnostics),
            "identity": _jsonable(self.identity),
            "record": _jsonable(self.record),
            "provenance": _jsonable(self.provenance),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_dict(provenance: Provenance | Mapping[str, Any] | None) -> dict[str, object] | None:
    if provenance is None:
        return None
    if isinstance(provenance, Provenance):
        return provenance.to_dict()
    if isinstance(provenance, Mapping):
        return {str(key): _jsonable(value) for key, value in provenance.items()}
    raise TypeError("provenance must be Provenance, a mapping, or None")


def _independence_payload(record: PhysicalLabelRecord) -> dict[str, object]:
    result = record.independence
    certificate = result.certificate
    return {
        "variables": [list(variable) for variable in result.variables],
        "labels": ["s" + "".join(str(item) for item in variable) for variable in result.variables],
        "rank": result.rank,
        "variable_count": result.variable_count,
        "independent": result.independent,
        "constraint": result.constraint.to_dict(),
        "certificate": certificate.to_dict() if certificate else None,
        "relation": certificate.format() if certificate else None,
    }


def _factor_diagnostic(record: PhysicalLabelRecord, vertex: int, text: str) -> TwoBodyFactor:
    return TwoBodyFactor(
        vertex=vertex,
        incoming_momentum="未恢复",
        incoming_coefficients=(),
        parent_mass="未定义",
        parent_mass_squared="未定义",
        outgoing=(),
        connection_references=(),
        reference_frame=None,
        angles=None,
        expression="未生成 dΦ₂（" + text + "）",
        status="unavailable",
        diagnostics=(text,),
    )


def _build_factors(record: PhysicalLabelRecord) -> tuple[TwoBodyFactor, ...]:
    topology = record.topology
    n = topology.n
    vectors = {
        (edge_index, occurrence): tuple(vector)
        for edge_index, occurrence, vector in record.momentum.double_momenta
    }
    doubles = {
        (label.edge_index, label.occurrence): tuple(label.particle_ids)
        for label in record.complete_labeling.double_lines
    }
    singles = {
        (label.edge_index, label.occurrence): label.particle_id
        for label in record.complete_labeling.single_lines
    }
    edges: list[tuple[int, int, EdgeType, int, int]] = []
    for edge_index, edge in enumerate(topology.edges):
        for occurrence in range(max(edge.multiplicity, 0)):
            edges.append((edge_index, occurrence, edge.edge_type, edge.source, edge.target))
    finals: dict[int, list[tuple[int, int]]] = {vertex: [] for vertex in topology.vertices}
    for label in record.complete_labeling.final_particles:
        finals.setdefault(label.vertex, []).append((label.slot, label.particle_id))

    result: list[TwoBodyFactor] = []
    for vertex in topology.vertices:
        incoming_vector = [0] * n
        incoming_terms: list[str] = []
        incoming_refs = []
        if vertex == topology.initial_vertex:
            incoming_terms.append("P")
            incoming_vector = [1] * n
            incoming_refs.append({"role": "initial", "vertex": vertex})
        for edge_index, occurrence, edge_type, source, target in edges:
            if target != vertex:
                continue
            ref = (edge_index, occurrence)
            incoming_refs.append({"role": edge_type.value, "edge_index": edge_index, "occurrence": occurrence, "source": source, "target": target})
            particle = singles.get(ref)
            if edge_type is EdgeType.DOUBLE:
                vector = vectors.get(ref)
                if vector is None:
                    incoming_terms.append(f"q_{{{edge_index + 1},{occurrence + 1}}}")
                else:
                    incoming_terms.append(f"q_{{{edge_index + 1},{occurrence + 1}}}")
                    for index, value in enumerate(vector):
                        incoming_vector[index] += value
            else:
                if particle is None:
                    incoming_terms.append(f"x_{{{edge_index + 1},{occurrence + 1}}}")
                else:
                    incoming_terms.append(_particle_symbol(particle))
                    incoming_vector[particle - 1] += 1

        # A graph vertex is a two-body block only when exactly two outgoing
        # occurrences exist after final-state slots are included.  Final lines
        # are listed first to retain the conventional B ordering.
        outgoing: list[FormulaConnection] = []
        for slot, particle in sorted(finals.get(vertex, ())):
            mass, mass_squared = _mass_for_particle(particle)
            outgoing.append(FormulaConnection(
                role="external",
                edge_index=None,
                occurrence=None,
                source=vertex,
                target=None,
                momentum_expression=_particle_symbol(particle),
                coefficients=tuple(1 if index + 1 == particle else 0 for index in range(n)),
                mass=mass,
                mass_squared=mass_squared,
                particle_ids=(particle,),
                slot=slot,
            ))
        for edge_index, occurrence, edge_type, source, target in edges:
            if source != vertex:
                continue
            ref = (edge_index, occurrence)
            if edge_type is EdgeType.DOUBLE:
                vector = vectors.get(ref)
                particles = doubles.get(ref, ())
                if vector is not None:
                    mass, mass_squared = _mass_symbols(vector, n, squared=False)
                    momentum = _vector_expression(vector)
                else:
                    mass, mass_squared = _symbol_for_double(particles, n)
                    momentum = "未恢复"
                outgoing.append(FormulaConnection(
                    role="double",
                    edge_index=edge_index,
                    occurrence=occurrence,
                    source=source,
                    target=target,
                    momentum_expression=momentum,
                    coefficients=tuple(vector) if vector is not None else None,
                    mass=mass,
                    mass_squared=mass_squared or "未定义",
                    particle_ids=tuple(particles),
                ))
            else:
                particle = singles.get(ref)
                if particle is None:
                    mass, mass_squared, momentum, particles = "未定义", "未定义", "未恢复", ()
                else:
                    mass, mass_squared = _mass_for_particle(particle)
                    momentum, particles = _particle_symbol(particle), (particle,)
                outgoing.append(FormulaConnection(
                    role="single",
                    edge_index=edge_index,
                    occurrence=occurrence,
                    source=source,
                    target=target,
                    momentum_expression=momentum,
                    coefficients=tuple(1 if index + 1 == particle else 0 for index in range(n)) if particle is not None else None,
                    mass=mass,
                    mass_squared=mass_squared,
                    particle_ids=tuple(particles),
                ))

        incoming_expression = " + ".join(incoming_terms) or "0"
        parent_mass, parent_mass_squared = _mass_symbols(incoming_vector, n, squared=False)
        references = tuple({
            "role": item.role,
            "edge_index": item.edge_index,
            "occurrence": item.occurrence,
            "source": item.source,
            "target": item.target,
            "slot": item.slot,
        } for item in outgoing)
        diagnostics: list[str] = []
        status = "ready"
        if len(outgoing) != 2:
            status = "unavailable"
            diagnostics.append(f"顶点 {vertex} 的出射出现数为 {len(outgoing)}，不是二体的 2 条。")
        if len(outgoing) == 2:
            expression = f"dΦ₂({parent_mass}; {outgoing[0].mass}, {outgoing[1].mass})"
        else:
            expression = "未生成 dΦ₂"
        result.append(TwoBodyFactor(
            vertex=vertex,
            incoming_momentum=incoming_expression if incoming_terms else "0",
            incoming_coefficients=tuple(incoming_vector),
            parent_mass=parent_mass,
            parent_mass_squared=parent_mass_squared or "未定义",
            outgoing=tuple(outgoing),
            connection_references=references,
            reference_frame=None,
            angles=None,
            expression=expression,
            status=status,
            diagnostics=tuple(diagnostics),
            incoming_references=tuple(incoming_refs),
        ))
    return tuple(result)


def _build_double_measures(record: PhysicalLabelRecord, convention: str) -> tuple[DoubleMeasureInput, ...]:
    config = MEASURE_CONVENTIONS[convention]
    result: list[DoubleMeasureInput] = []
    for edge_index, occurrence, vector in record.momentum.double_momenta:
        values = tuple(vector)
        particles = _support(values)
        label = "s" + "".join(str(item) for item in particles) if particles else "未定义"
        differential = config["double_line_factor"].replace("ds_S", f"ds_{{{''.join(str(item) for item in particles)}}}")
        result.append(DoubleMeasureInput(
            edge_index=edge_index,
            occurrence=occurrence,
            variable=particles,
            variable_label=label,
            differential=differential,
            convention=convention,
            momentum_expression=_vector_expression(values),
        ))
    return tuple(result)


def _build_particle_measures(record: PhysicalLabelRecord) -> tuple[ParticleMeasureInput, ...]:
    counts = dict(record.measure_cancellations)
    return tuple(
        ParticleMeasureInput(
            particle_id=particle,
            occurrences=tuple(occurrences),
            cancellation_count=counts.get(particle, 0),
            differential=f"粒子 p_{particle} 测度抵消 {counts.get(particle, 0)} 次",
        )
        for particle, occurrences in record.single_groups
    )


def _diagnostic_rows(stage: str, reason: str | None = None) -> tuple[dict[str, object], ...]:
    if reason is None:
        return ()
    return ({"code": stage, "message": reason},)


def build_formula_input(
    record: PhysicalLabelRecord | NormalizationResult,
    *,
    measure_convention: str,
    identity: MarkedIdentity | None = None,
    provenance: Provenance | Mapping[str, Any] | None = None,
) -> FormulaInputCard:
    """Build a formula input card from a physical record or its normalization.

    The accepted path produces local factors.  A dependent normalization (or
    a dependent record passed directly) is explicitly unavailable, retaining
    its rank certificate instead of producing a misleading ready card.
    """

    if measure_convention not in MEASURE_CONVENTIONS:
        raise ValueError(
            f"unknown measure convention {measure_convention!r}; choose one of {tuple(MEASURE_CONVENTIONS)}"
        )
    normalization: NormalizationResult | None = record if isinstance(record, NormalizationResult) else None
    physical = normalization.record if normalization is not None else record
    if not isinstance(record, (PhysicalLabelRecord, NormalizationResult)):
        raise TypeError("build_formula_input requires a complete record or a normalization result")
    if physical is not None and not isinstance(physical, PhysicalLabelRecord):
        raise TypeError("build_formula_input expects a PhysicalLabelRecord or NormalizationResult")
    if physical is not None:
        if physical.topology.n != 4:
            raise ValueError("formula inputs currently support n=4 only")
        checked = review_physical_record(physical)
        if normalization is not None and normalization.stage != checked.stage:
            raise ValueError("normalization stage disagrees with the physical record")
    if identity is not None:
        if not isinstance(identity, MarkedIdentity):
            raise TypeError("identity must be a MarkedIdentity or None")
        if physical is None:
            raise ValueError("an identity cannot supply a missing record")
        expected = canonicalize_physical_labeling(physical, identity.mode)
        if (identity.key, identity.digest, identity.canonical_record) != (expected.key, expected.digest, expected.canonical_record):
            raise ValueError("identity does not belong to the supplied physical record")
        physical = identity.canonical_record

    config = MEASURE_CONVENTIONS[measure_convention]
    n = physical.topology.n if physical is not None else None
    loop_count = None
    if normalization is not None and normalization.structure_report is not None:
        loop_count = normalization.structure_report.metrics.loop_count
    elif physical is not None:
        loop_count = physical.topology.edge_count - len(physical.topology.vertices) + 1
    mode = identity.mode.value if identity is not None else None
    identity_dict = identity.to_dict() if identity is not None else None
    source = _source_dict(provenance)

    status = normalization.stage if normalization is not None else (
        "accepted" if physical is not None and physical.independence.independent else "dependent"
    )
    independent = physical is not None and physical.independence.independent
    reason: str | None = None
    if physical is None:
        reason = "没有完整物理记录；公式输入卡不能从裸拓扑或失败结果猜测标记。"
    elif not independent:
        certificate = physical.independence.certificate
        reason = "所选双线不变量不独立，不能进入已就绪公式输入。"
        if certificate is not None:
            reason += f" 依赖证书：{certificate.format()}"
        status = "dependent"
    elif normalization is not None and normalization.stage != "accepted":
        reason = f"记录阶段为 {normalization.stage}，没有可用的 accepted 公式输入。"
        independent = False

    if independent and physical is not None and (normalization is None or normalization.stage == "accepted"):
        factors = _build_factors(physical)
        factor_ready = all(item.status == "ready" for item in factors)
        double_measures = _build_double_measures(physical, measure_convention)
        particle_measures = _build_particle_measures(physical)
        diagnostics = tuple(
            {"code": "factor_unavailable", "message": message}
            for factor in factors
            for message in factor.diagnostics
        )
        availability = "ready" if factor_ready else "unavailable"
        if not factor_ready and reason is None:
            reason = "至少一个顶点不能构造成恰好两条出射的局部二体因子。"
        status = "accepted" if factor_ready else "factor_unavailable"
    else:
        factors = ()
        double_measures = ()
        particle_measures = ()
        diagnostics = _diagnostic_rows(status, reason)
        availability = "unavailable"

    independence_payload = _independence_payload(physical) if physical is not None else None
    constraints: tuple[str, ...] = ()
    if physical is not None:
        constraint = physical.independence.constraint
        constraints = (constraint.equation_name,)
        if physical.independence.certificate is not None:
            constraints += (physical.independence.certificate.format(),)
    formula_status = {
        "factors_ready": bool(availability == "ready"),
        "constraints_eliminated": False,
        "jacobian_checked": False,
        "branches_checked": False,
        "domain_checked": False,
        "formula_verified": False,
    }
    variables = physical.independence.variables if physical is not None else ()
    labels = tuple("s" + "".join(str(item) for item in variable) for variable in variables)
    return FormulaInputCard(
        schema_version=FORMULA_INPUT_SCHEMA_VERSION,
        status=status,
        availability=availability,
        unavailable_reason=reason,
        n=n,
        loop_count=loop_count,
        mode=mode,
        measure_convention=measure_convention,
        measure_definition=config["definition"],
        factors=factors,
        double_measures=double_measures,
        particle_measures=particle_measures,
        independent_variables=tuple(tuple(variable) for variable in variables),
        variable_labels=labels,
        independence=independence_payload,
        constraints=constraints,
        formula_status=formula_status,
        missing_steps=UNVERIFIED_STEPS,
        diagnostics=diagnostics,
        identity=identity_dict,
        record=physical.to_dict() if physical is not None else None,
        provenance=source,
    )


# Descriptive aliases used by the task notes and future callers.
build_formula_input_card = build_formula_input
FormulaInput = FormulaInputCard


def render_formula_input_markdown(card: FormulaInputCard) -> str:
    """Render a compact Chinese study view without inventing missing physics."""

    if not isinstance(card, FormulaInputCard):
        raise TypeError("render_formula_input_markdown expects a FormulaInputCard")
    lines = [
        "## 公式输入卡",
        "",
        f"- 状态：{card.status}；可用性：{card.availability}",
        f"- 体系：n={card.n if card.n is not None else '未知'}，圈数 l={card.loop_count if card.loop_count is not None else '未知'}",
        f"- 测度约定：{card.measure_convention}（{card.measure_definition}）",
        f"- 公式已验证：{'是' if card.formula_verified else '否'}",
    ]
    if card.unavailable_reason:
        lines += [f"- 不可用原因：{card.unavailable_reason}"]
    if card.factors:
        lines += ["", "### 局部二体因子", "", "| 顶点 | 总入射动量 | 父质量平方 | 出射质量平方 | 输入表达式 |", "| --- | --- | --- | --- | --- |"]
        for factor in card.factors:
            outgoing = ", ".join(item.mass_squared for item in factor.outgoing)
            lines.append(f"| {factor.vertex} | {factor.incoming_momentum} | {factor.parent_mass_squared} | {outgoing} | {factor.expression} |")
    if card.double_measures:
        lines += ["", "### 双线微分输入", "", "| 双线 | 变量 | 微分因子 | 动量 |", "| --- | --- | --- | --- |"]
        for item in card.double_measures:
            lines.append(f"| 边 {item.edge_index + 1}，出现 {item.occurrence + 1} | {item.variable_label} | {item.differential} | {item.momentum_expression} |")
    if card.particle_measures:
        lines += ["", "### 粒子测度出现与抵消", "", "| 粒子 | 抵消次数 | 全部出现 |", "| --- | ---: | --- |"]
        for item in card.particle_measures:
            occurrences = "; ".join(str(entry) for entry in item.occurrences) or "无"
            lines.append(f"| p_{item.particle_id} | {item.cancellation_count} | {occurrences} |")
    if card.independence:
        relation = card.independence.get("relation") or "无依赖证书"
        lines += ["", "### 独立变量与约束证据", "", f"- 变量：{', '.join(card.variable_labels) or '无'}", f"- 秩：{card.independence.get('rank')} / {card.independence.get('variable_count')}", f"- 证书：{relation}"]
    lines += ["", "### 尚未核验", "", *[f"- {step}" for step in card.missing_steps], "", "<details><summary>机器记录</summary>", "", "```json", card.to_json(), "```", "</details>"]
    return "\n".join(lines)


def render_formula_input_html(card: FormulaInputCard) -> str:
    if not isinstance(card, FormulaInputCard):
        raise TypeError("render_formula_input_html expects a FormulaInputCard")
    visible, _, _ = render_formula_input_markdown(card).partition("<details><summary>")
    return (
        '<article class="formula-input-card"><pre>'
        + escape(visible.rstrip())
        + '</pre><details><summary>机器记录</summary><pre>'
        + escape(card.to_json())
        + '</pre></details></article>'
    )


__all__ = [
    "FORMULA_INPUT_SCHEMA_VERSION",
    "MEASURE_CONVENTIONS",
    "UNVERIFIED_STEPS",
    "FormulaConnection",
    "TwoBodyFactor",
    "DoubleMeasureInput",
    "ParticleMeasureInput",
    "FormulaInputCard",
    "FormulaInput",
    "build_formula_input",
    "build_formula_input_card",
    "render_formula_input_markdown",
    "render_formula_input_html",
]
