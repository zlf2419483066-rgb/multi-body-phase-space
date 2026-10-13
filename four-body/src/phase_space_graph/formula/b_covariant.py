"""B/F03-specific replayable covariant elimination trace.

The implementation is deliberately restricted to the frozen B/F03 identity.
It replays source-bound local and global steps and emits a ``built`` record;
it does not modify FormulaArtifact/CovariantResult verification status.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from ..formula_inputs import FormulaInputCard
from .cancellations import CancellationPlan, OccurrenceLocator, ParticleOccurrence
from .expanded import ExpandedF0Expression, OnShellMeasureExpansion, build_expanded_f0_expression
from .inputs import validate_formula_card
from .local_reduction import (
    LocalReductionTrace,
    reduce_local_measures,
)


B_F03_CERTIFICATE_SCHEMA_VERSION = "FM-BF03-covariant-trace-v1"
B_F03_IDENTITY_DIGEST = "6662211ad03a46d6aa972d9904e26e40a0eedf788f5851bbde6b57abe4e60119"
B_F03_VARIABLES = ((1, 2), (3, 4), (2, 3))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sum_momenta(particles: tuple[int, ...]) -> str:
    return " + ".join(f"p_{particle}" for particle in particles)


@dataclass(frozen=True, slots=True)
class VertexConservationTrace:
    vertex: int
    lhs: str
    rhs: str
    equation: str
    incoming_sources: tuple[dict[str, Any], ...]
    outgoing_sources: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "incoming_sources", tuple(dict(item) for item in self.incoming_sources))
        object.__setattr__(self, "outgoing_sources", tuple(dict(item) for item in self.outgoing_sources))

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex": self.vertex,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "equation": self.equation,
            "incoming_sources": [dict(item) for item in self.incoming_sources],
            "outgoing_sources": [dict(item) for item in self.outgoing_sources],
        }


@dataclass(frozen=True, slots=True)
class ShellDeltaTransfer:
    edge_index: int
    occurrence: int
    variable_label: str
    support_momentum: str
    source_shell_delta: str
    source_positive_energy: str
    result_delta: str
    result_positive_energy: str
    delta_sign_rule: str
    q_measure_denominator_power: int
    double_line_differential: str
    double_line_source: dict[str, Any]
    shell_measure_source: dict[str, Any]
    positive_energy_sources: tuple[dict[str, Any], ...]
    positive_energy_implied: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "double_line_source", dict(self.double_line_source))
        object.__setattr__(self, "shell_measure_source", dict(self.shell_measure_source))
        object.__setattr__(self, "positive_energy_sources", tuple(dict(item) for item in self.positive_energy_sources))

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_index": self.edge_index,
            "occurrence": self.occurrence,
            "variable_label": self.variable_label,
            "support_momentum": self.support_momentum,
            "source_shell_delta": self.source_shell_delta,
            "source_positive_energy": self.source_positive_energy,
            "result_delta": self.result_delta,
            "result_positive_energy": self.result_positive_energy,
            "delta_sign_rule": self.delta_sign_rule,
            "q_measure_denominator_power": self.q_measure_denominator_power,
            "double_line_differential": self.double_line_differential,
            "double_line_source": dict(self.double_line_source),
            "shell_measure_source": dict(self.shell_measure_source),
            "positive_energy_sources": [dict(item) for item in self.positive_energy_sources],
            "positive_energy_implied": self.positive_energy_implied,
        }


@dataclass(frozen=True, slots=True)
class MomentumEliminationStep:
    step_id: str
    operation: str
    pair_momentum_label: str
    q_momentum: str
    edge_index: int
    occurrence: int
    source_vertex: int
    target_vertex: int
    relation: str
    substitutions: tuple[dict[str, str], ...]
    source_conservation_vertex: int | None
    source_conservation_delta4: dict[str, Any] | None
    source_local_constraint: dict[str, Any] | None
    source_q_measure: dict[str, Any]
    absolute_jacobian: str
    jacobian_reason: str
    delta4_consumption: str
    shell_transfer: ShellDeltaTransfer

    def __post_init__(self) -> None:
        object.__setattr__(self, "substitutions", tuple(dict(item) for item in self.substitutions))
        if self.source_conservation_delta4 is not None:
            object.__setattr__(self, "source_conservation_delta4", dict(self.source_conservation_delta4))
        if self.source_local_constraint is not None:
            object.__setattr__(self, "source_local_constraint", dict(self.source_local_constraint))
        object.__setattr__(self, "source_q_measure", dict(self.source_q_measure))

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "pair_momentum_label": self.pair_momentum_label,
            "q_momentum": self.q_momentum,
            "edge_index": self.edge_index,
            "occurrence": self.occurrence,
            "source_vertex": self.source_vertex,
            "target_vertex": self.target_vertex,
            "relation": self.relation,
            "substitutions": [dict(item) for item in self.substitutions],
            "source_conservation_vertex": self.source_conservation_vertex,
            "source_conservation_delta4": dict(self.source_conservation_delta4) if self.source_conservation_delta4 else None,
            "source_local_constraint": dict(self.source_local_constraint) if self.source_local_constraint else None,
            "source_q_measure": dict(self.source_q_measure),
            "absolute_jacobian": self.absolute_jacobian,
            "jacobian_reason": self.jacobian_reason,
            "delta4_consumption": self.delta4_consumption,
            "shell_transfer": self.shell_transfer.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ParticleMeasureSource:
    particle_id: int
    retained_occurrence: dict[str, Any]
    representation: str
    measure_expression: str
    source: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "retained_occurrence", dict(self.retained_occurrence))
        object.__setattr__(self, "source", dict(self.source))

    def to_dict(self) -> dict[str, object]:
        return {
            "particle_id": self.particle_id,
            "retained_occurrence": dict(self.retained_occurrence),
            "representation": self.representation,
            "measure_expression": self.measure_expression,
            "source": dict(self.source),
        }


@dataclass(frozen=True, slots=True)
class Delta4Disposition:
    vertex: int
    disposition: str
    source_delta4: str
    source_references: tuple[dict[str, Any], ...]
    consumed_by: str | None
    result: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_references", tuple(dict(item) for item in self.source_references))

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex": self.vertex,
            "disposition": self.disposition,
            "source_delta4": self.source_delta4,
            "source_references": [dict(item) for item in self.source_references],
            "consumed_by": self.consumed_by,
            "result": self.result,
        }


@dataclass(frozen=True, slots=True)
class CovariantNormalizationLedger:
    convention: str
    local_vertex_prefactor_power: int
    local_vertex_count: int
    total_local_prefactor_power: int
    outgoing_onshell_measure_count: int
    outgoing_measure_denominator_power: int
    duplicate_p2_measure_denominator_removed: int
    double_q_measure_count: int
    q_shell_denominator_power: int
    double_line_differential_power: int
    raw_f0_total_power: int
    post_cancellation_total_power: int
    retained_particle_measure_count: int
    retained_particle_measure_denominator_power: int
    global_coefficient_power: int
    final_total_power: int
    standard_minus_jing_power: int
    accounting_steps: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "accounting_steps", tuple(self.accounting_steps))

    def to_dict(self) -> dict[str, object]:
        return {
            "convention": self.convention,
            "local_vertex_prefactor_power": self.local_vertex_prefactor_power,
            "local_vertex_count": self.local_vertex_count,
            "total_local_prefactor_power": self.total_local_prefactor_power,
            "outgoing_onshell_measure_count": self.outgoing_onshell_measure_count,
            "outgoing_measure_denominator_power": self.outgoing_measure_denominator_power,
            "duplicate_p2_measure_denominator_removed": self.duplicate_p2_measure_denominator_removed,
            "double_q_measure_count": self.double_q_measure_count,
            "q_shell_denominator_power": self.q_shell_denominator_power,
            "double_line_differential_power": self.double_line_differential_power,
            "raw_f0_total_power": self.raw_f0_total_power,
            "post_cancellation_total_power": self.post_cancellation_total_power,
            "retained_particle_measure_count": self.retained_particle_measure_count,
            "retained_particle_measure_denominator_power": self.retained_particle_measure_denominator_power,
            "global_coefficient_power": self.global_coefficient_power,
            "final_total_power": self.final_total_power,
            "standard_minus_jing_power": self.standard_minus_jing_power,
            "accounting_steps": list(self.accounting_steps),
        }


@dataclass(frozen=True, slots=True)
class BFCovariantCertificate:
    schema_version: str
    artifact_type: str
    status: str
    identity_json: str
    input_card_digest: str
    measure_convention: str
    ordered_variables: tuple[str, ...]
    source_scope: str
    conservation_equations: tuple[VertexConservationTrace, ...]
    elimination_steps: tuple[MomentumEliminationStep, ...]
    delta4_dispositions: tuple[Delta4Disposition, ...]
    retained_particle_measures: tuple[ParticleMeasureSource, ...]
    preserved_double_line_factors: tuple[dict[str, Any], ...]
    residual_global_equation: str
    residual_global_delta4: str
    invariant_deltas: tuple[dict[str, Any], ...]
    normalization: CovariantNormalizationLedger
    formula_verified: bool = False
    source_match_status: str = "not_checked"

    def __post_init__(self) -> None:
        if self.schema_version != B_F03_CERTIFICATE_SCHEMA_VERSION or self.artifact_type != "b_f03_covariant_elimination_trace":
            raise ValueError("unsupported B/F03 covariant trace schema")
        if self.status not in {"built", "derived"}:
            raise ValueError("B/F03 trace status must remain built or derived")
        object.__setattr__(self, "ordered_variables", tuple(self.ordered_variables))
        object.__setattr__(self, "conservation_equations", tuple(self.conservation_equations))
        object.__setattr__(self, "elimination_steps", tuple(self.elimination_steps))
        object.__setattr__(self, "delta4_dispositions", tuple(self.delta4_dispositions))
        object.__setattr__(self, "retained_particle_measures", tuple(self.retained_particle_measures))
        object.__setattr__(self, "preserved_double_line_factors", tuple(dict(item) for item in self.preserved_double_line_factors))
        object.__setattr__(self, "invariant_deltas", tuple(dict(item) for item in self.invariant_deltas))
        if self.formula_verified or self.source_match_status != "not_checked":
            raise ValueError("this certificate cannot claim formula verification or source matching")

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(self.identity_json)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "status": self.status,
            "identity": self.identity,
            "input_card_digest": self.input_card_digest,
            "measure_convention": self.measure_convention,
            "ordered_variables": list(self.ordered_variables),
            "source_scope": self.source_scope,
            "conservation_equations": [item.to_dict() for item in self.conservation_equations],
            "elimination_steps": [item.to_dict() for item in self.elimination_steps],
            "delta4_dispositions": [item.to_dict() for item in self.delta4_dispositions],
            "retained_particle_measures": [item.to_dict() for item in self.retained_particle_measures],
            "preserved_double_line_factors": [dict(item) for item in self.preserved_double_line_factors],
            "residual_global_equation": self.residual_global_equation,
            "residual_global_delta4": self.residual_global_delta4,
            "invariant_deltas": [dict(item) for item in self.invariant_deltas],
            "normalization": self.normalization.to_dict(),
            "formula_verified": self.formula_verified,
            "source_match_status": self.source_match_status,
        }

    def to_json(self) -> str:
        return _json(self.to_dict())


def _check_b_card(card: FormulaInputCard) -> None:
    validate_formula_card(card)
    if card.availability != "ready" or card.identity is None:
        raise ValueError("B/F03 covariant replay requires a ready card with complete identity")
    if card.identity.get("digest") != B_F03_IDENTITY_DIGEST or card.identity.get("mode") != "fixed_labels":
        raise ValueError("card is not the frozen fixed-label B/F03 identity")
    if card.independent_variables != B_F03_VARIABLES or card.variable_labels != ("s12", "s34", "s23"):
        raise ValueError("B/F03 requires the ordered variables (s12,s34,s23)")
    record = card.record
    if record is None:
        raise ValueError("B/F03 card is missing its complete physical record")
    topology = record["topology"]
    expected_edges = [
        {"source": 0, "target": 1, "edge_type": "double", "multiplicity": 1},
        {"source": 0, "target": 2, "edge_type": "double", "multiplicity": 1},
        {"source": 1, "target": 2, "edge_type": "single", "multiplicity": 1},
        {"source": 2, "target": 3, "edge_type": "double", "multiplicity": 1},
    ]
    if (topology.get("n"), topology.get("vertices"), topology.get("initial_vertex"), topology.get("edges")) != (
        4, [0, 1, 2, 3], 0, expected_edges
    ):
        raise ValueError("card graph does not match the B/F03 four-vertex, three-double-line structure")
    labeling = record["complete_labeling"]
    expected_finals = [
        {"vertex": 1, "slot": 0, "particle_id": 1},
        {"vertex": 2, "slot": 0, "particle_id": 4},
        {"vertex": 3, "slot": 0, "particle_id": 2},
        {"vertex": 3, "slot": 1, "particle_id": 3},
    ]
    if labeling.get("final_particles") != expected_finals or labeling.get("single_lines") != [
        {"edge_index": 2, "occurrence": 0, "particle_id": 2}
    ]:
        raise ValueError("card final/single particle occurrences do not match B/F03")
    if [(item.vertex, tuple(item.incoming_coefficients)) for item in card.factors] != [
        (0, (1, 1, 1, 1)), (1, (1, 1, 0, 0)), (2, (0, 1, 1, 1)), (3, (0, 1, 1, 0))
    ]:
        raise ValueError("card local incoming systems do not match B/F03")


def _check_hapof_plan(plan: CancellationPlan) -> None:
    if plan.formula_identity.get("digest") != B_F03_IDENTITY_DIGEST:
        raise ValueError("cancellation plan has the wrong B/F03 identity")
    by_particle = {item.particle_id: item for item in plan.particles}
    p2 = by_particle.get(2)
    if p2 is None:
        raise ValueError("B/F03 cancellation plan is missing p2")
    expected_internal = OccurrenceLocator("internal", 2, 0)
    expected_external = OccurrenceLocator("external", 3, 0)
    if p2.retained_occurrence.locator != expected_internal or tuple(item.locator for item in p2.cancellation_occurrences) != (expected_external,):
        raise ValueError("B/F03 requires retaining internal p2 and canceling terminal external p2")
    if p2.selection_policy != "formula_level_cancellation_override":
        raise ValueError("B/F03 p2 cancellation must be an explicit formula-level override")
    if any(item.cancellation_occurrences for item in plan.particles if item.particle_id != 2):
        raise ValueError("B/F03 permits only the p2 terminal external occurrence cancellation")


def _factor_map(expanded: ExpandedF0Expression) -> dict[int, Any]:
    return {factor.vertex: factor for factor in expanded.factors}


def _conservation_trace(factor: Any) -> VertexConservationTrace:
    rhs = " + ".join(factor.outgoing_integration_momenta)
    outgoing_sources = tuple(measure.source_reference.to_dict() for measure in factor.outgoing_measures)
    incoming_sources = tuple(reference.to_dict() for reference in factor.incoming_references)
    return VertexConservationTrace(
        vertex=factor.vertex,
        lhs=factor.incoming_momentum,
        rhs=rhs,
        equation=f"{factor.incoming_momentum} = {rhs}",
        incoming_sources=incoming_sources,
        outgoing_sources=outgoing_sources,
    )


def _single_measure_map(expanded: ExpandedF0Expression) -> dict[OccurrenceLocator, tuple[Any, OnShellMeasureExpansion]]:
    result: dict[OccurrenceLocator, tuple[Any, OnShellMeasureExpansion]] = {}
    for factor in expanded.factors:
        for measure in factor.outgoing_measures:
            source = measure.source_reference
            if source.role == "external":
                locator = OccurrenceLocator("external", source.vertex, source.slot)  # type: ignore[arg-type]
            elif source.role == "single":
                locator = OccurrenceLocator("internal", source.edge_index, source.occurrence)  # type: ignore[arg-type]
            else:
                continue
            if locator in result:
                raise ValueError(f"expanded formula repeats occurrence location {locator}")
            result[locator] = (factor, measure)
    return result


def _q_measure_map(expanded: ExpandedF0Expression) -> dict[tuple[int, int], tuple[Any, OnShellMeasureExpansion]]:
    result: dict[tuple[int, int], tuple[Any, OnShellMeasureExpansion]] = {}
    for factor in expanded.factors:
        for measure in factor.outgoing_measures:
            source = measure.source_reference
            if source.role != "double":
                continue
            if source.edge_index is None or source.occurrence is None:
                raise ValueError("B/F03 q measure is missing its edge occurrence")
            key = (source.edge_index, source.occurrence)
            if key in result:
                raise ValueError(f"expanded formula repeats q measure {key}")
            result[key] = (factor, measure)
    return result


def _ds_map(expanded: ExpandedF0Expression) -> dict[tuple[int, int], Any]:
    result = {(item.edge_index, item.occurrence): item for item in expanded.double_line_factors}
    if len(result) != len(expanded.double_line_factors):
        raise ValueError("expanded F0 repeats a double-line differential occurrence")
    return result


def _shell_transfer(
    key: tuple[int, int],
    q_measure: OnShellMeasureExpansion,
    ds_factor: Any,
    support_ids: tuple[int, ...],
    retained_measures: dict[int, tuple[Any, OnShellMeasureExpansion]],
) -> ShellDeltaTransfer:
    support = _sum_momenta(support_ids)
    if q_measure.support_momentum_expression != support or ds_factor.variable != support_ids:
        raise ValueError(f"q measure {key} support does not match the B/F03 particle sum")
    expected_mass_squared = "s_" + "".join(str(item) for item in support_ids)
    if q_measure.mass_squared != expected_mass_squared:
        raise ValueError(f"q measure {key} shell mass disagrees with its ds variable")
    particle_sources: list[dict[str, Any]] = []
    for particle in support_ids:
        retained = retained_measures.get(particle)
        if retained is None:
            raise ValueError(f"q measure {key} support particle p{particle} has no retained positive-energy measure")
        _, measure = retained
        particle_sources.append({
            "particle_id": particle,
            "positive_energy": measure.positive_energy,
            "source_reference": measure.source_reference.to_dict(),
        })
    variable = q_measure.momentum_expression
    return ShellDeltaTransfer(
        edge_index=key[0],
        occurrence=key[1],
        variable_label=ds_factor.variable_label,
        support_momentum=support,
        source_shell_delta=q_measure.shell_delta,
        source_positive_energy=q_measure.positive_energy,
        result_delta=f"delta({ds_factor.variable_label} - ({support})^2)",
        result_positive_energy=f"theta(({support})^0)",
        delta_sign_rule="delta(x-s)=delta(s-x); absolute linear sign Jacobian is 1",
        q_measure_denominator_power=3,
        double_line_differential=ds_factor.differential,
        double_line_source=ds_factor.source_reference.to_dict(),
        shell_measure_source={
            "source_layer": "ExpandedTwoBodyFactor.outgoing_measures",
            "vertex": q_measure.source_reference.vertex,
            "momentum_expression": variable,
            "source_reference": q_measure.source_reference.to_dict(),
        },
        positive_energy_sources=tuple(particle_sources),
        positive_energy_implied=True,
    )


def _normalization(expanded: ExpandedF0Expression) -> CovariantNormalizationLedger:
    local_power = 4 if expanded.measure_convention == "standard_2pi4" else 0
    local_total = local_power * len(expanded.factors)
    outgoing_count = sum(len(factor.outgoing_measures) for factor in expanded.factors)
    outgoing_power = -3 * outgoing_count
    ds_power_per_line = 3 if expanded.measure_convention == "jing_psf" else -1
    ds_total = ds_power_per_line * len(expanded.double_line_factors)
    raw = local_total + outgoing_power + ds_total
    duplicate_removed = 3
    post_cancel = raw + duplicate_removed
    q_count = len(expanded.double_line_factors)
    q_power = -3 * q_count
    final_particle_count = 4
    final_particle_power = -3 * final_particle_count
    coefficient = local_total + q_power + ds_total
    final_total = final_particle_power + coefficient
    standard_minus_jing = 4
    if (outgoing_count, len(expanded.factors), q_count) != (8, 4, 3):
        raise ValueError("B/F03 normalization ledger requires four vertices, eight dmu factors, and three q lines")
    if post_cancel != final_total:
        raise ValueError("B/F03 normalization ledger does not close after the duplicate p2 measure is removed")
    return CovariantNormalizationLedger(
        convention=expanded.measure_convention,
        local_vertex_prefactor_power=local_power,
        local_vertex_count=len(expanded.factors),
        total_local_prefactor_power=local_total,
        outgoing_onshell_measure_count=outgoing_count,
        outgoing_measure_denominator_power=outgoing_power,
        duplicate_p2_measure_denominator_removed=duplicate_removed,
        double_q_measure_count=q_count,
        q_shell_denominator_power=q_power,
        double_line_differential_power=ds_total,
        raw_f0_total_power=raw,
        post_cancellation_total_power=post_cancel,
        retained_particle_measure_count=final_particle_count,
        retained_particle_measure_denominator_power=final_particle_power,
        global_coefficient_power=coefficient,
        final_total_power=final_total,
        standard_minus_jing_power=standard_minus_jing,
        accounting_steps=(
            "F0: 4 local vertex prefactors + 8 outgoing dmu denominators + 3 double-line differentials.",
            "Cancel the duplicate external p2 dmu at vertex 3; removing its denominator changes the exponent by +3.",
            "The three q shell deltas keep their individual (2*pi)^-3 factors through substitution.",
            "Pair each q shell denominator with its same-edge ds factor; Jing contributes -3+3=0 per line.",
            "Standard contributes local +16, q shells -9, and ds -3, hence a global coefficient (2*pi)^4.",
            "The four retained final-particle measures contribute (2*pi)^-12 in both conventions.",
        ),
    )


def build_b_covariant_certificate(
    card: FormulaInputCard,
    expanded: ExpandedF0Expression,
    plan: CancellationPlan,
    local_trace: LocalReductionTrace,
) -> BFCovariantCertificate:
    """Replay the frozen B/F03 covariant elimination from bound source data."""

    if not isinstance(card, FormulaInputCard):
        raise TypeError("build_b_covariant_certificate expects a FormulaInputCard")
    if not isinstance(expanded, ExpandedF0Expression):
        raise TypeError("build_b_covariant_certificate expects an ExpandedF0Expression")
    if not isinstance(plan, CancellationPlan):
        raise TypeError("build_b_covariant_certificate expects a CancellationPlan")
    if not isinstance(local_trace, LocalReductionTrace):
        raise TypeError("build_b_covariant_certificate expects a LocalReductionTrace")
    _check_b_card(card)
    _check_hapof_plan(plan)

    expected_expanded = build_expanded_f0_expression(card)
    if expanded.to_dict() != expected_expanded.to_dict():
        raise ValueError("ExpandedF0Expression disagrees with the validated B/F03 card")
    if plan.input_card_digest != expanded.input_card_digest or plan.measure_convention != expanded.measure_convention:
        raise ValueError("CancellationPlan digest/convention does not match ExpandedF0Expression")
    if plan.formula_identity_json != _json(expanded.identity):
        raise ValueError("CancellationPlan identity does not match ExpandedF0Expression")
    expected_local_trace = reduce_local_measures(plan, expanded)
    if local_trace.to_dict() != expected_local_trace.to_dict():
        raise ValueError("LocalReductionTrace does not replay from the supplied plan and expanded F0")

    factors = _factor_map(expanded)
    if set(factors) != {0, 1, 2, 3}:
        raise ValueError("B/F03 requires exactly one local factor at each vertex 0,1,2,3")
    equations = tuple(_conservation_trace(factors[vertex]) for vertex in range(4))
    expected_equations = (
        "P = q_{1,1} + q_{2,1}",
        "q_{1,1} = p_1 + p_2",
        "q_{2,1} + p_2 = p_4 + q_{4,1}",
        "q_{4,1} = p_2 + p_3",
    )
    if tuple(item.equation for item in equations) != expected_equations:
        raise ValueError("actual vertex conservation equations do not match the B/F03 specification")

    single_measures = _single_measure_map(expanded)
    plan_by_particle = {item.particle_id: item for item in plan.particles}
    retained_measures: dict[int, tuple[Any, OnShellMeasureExpansion]] = {}
    particle_rows: list[ParticleMeasureSource] = []
    for particle in range(1, 5):
        particle_plan = plan_by_particle.get(particle)
        if particle_plan is None:
            raise ValueError(f"cancellation plan is missing p{particle}")
        occurrence = particle_plan.retained_occurrence
        source_pair = single_measures.get(occurrence.locator)
        if source_pair is None:
            raise ValueError(f"retained p{particle} occurrence is absent from expanded F0")
        factor, measure = source_pair
        retained_measures[particle] = source_pair
        if particle == 3:
            local = next((item for item in local_trace.factors if item.vertex == 3), None)
            if local is None or local.integrated_measure != measure.to_dict() or local.remaining_constraint is None:
                raise ValueError("p3 dmu must be the measure integrated into the vertex-3 residual constraint")
            representation = "recovered_from_local_residual_constraint_by_q23_translation"
            source = {
                "source_layer": "LocalReductionTrace.remaining_constraint",
                "constraint": local.remaining_constraint.to_dict(),
                "integrated_measure": local.integrated_measure,
                "original_occurrence_source": measure.source_reference.to_dict(),
            }
        else:
            if local_trace.factors[factor.vertex].status != "preserved":
                raise ValueError(f"retained p{particle} measure is not in a preserved local factor")
            representation = "retained_expanded_outgoing_measure"
            source = {"source_layer": "ExpandedF0Expression", "measure": measure.to_dict()}
        particle_rows.append(ParticleMeasureSource(
            particle_id=particle,
            retained_occurrence=occurrence.to_dict(),
            representation=representation,
            measure_expression=measure.expression,
            source=source,
        ))

    q_measures = _q_measure_map(expanded)
    ds_factors = _ds_map(expanded)
    q_specs = (
        {"key": (0, 0), "pair_label": "q12", "support": (1, 2), "solved": "q_{1,1} = p_1 + p_2", "source_vertex": 1, "operation": "integrate_q_with_vertex_delta4"},
        {"key": (3, 0), "pair_label": "q23", "support": (2, 3), "solved": "q_{4,1} = p_2 + p_3", "source_vertex": 3, "operation": "translate_q23_to_p3"},
        {"key": (1, 0), "pair_label": "q34", "support": (3, 4), "solved": "q_{2,1} = p_3 + p_4", "source_vertex": 2, "operation": "integrate_q_with_vertex_delta4"},
    )
    if set(q_measures) != {(0, 0), (1, 0), (3, 0)} or set(ds_factors) != set(q_measures):
        raise ValueError("B/F03 requires exactly the three specified q/ds edge occurrences")

    transfers: dict[tuple[int, int], ShellDeltaTransfer] = {}
    steps: list[MomentumEliminationStep] = []
    for index, spec in enumerate(q_specs, 1):
        key = spec["key"]
        q_factor, q_measure = q_measures[key]
        ds_factor = ds_factors[key]
        support_ids = spec["support"]
        if q_measure.source_reference.source != (0 if key in {(0, 0), (1, 0)} else 2):
            raise ValueError(f"q measure {key} has the wrong source vertex")
        expected_target = {(0, 0): 1, (1, 0): 2, (3, 0): 3}[key]
        if q_measure.source_reference.target != expected_target:
            raise ValueError(f"q measure {key} has the wrong target vertex")
        transfer = _shell_transfer(key, q_measure, ds_factor, support_ids, retained_measures)
        transfers[key] = transfer
        source_vertex = spec["source_vertex"]
        if spec["operation"] == "translate_q23_to_p3":
            local = next(item for item in local_trace.factors if item.vertex == 3)
            source_local_constraint = local.remaining_constraint.to_dict()
            source_delta4 = None
            substitutions = ({"from": "q_{4,1} - p_2", "to": "p_3", "source_step": "vertex_3_local_p3_integration"},)
            jacobian_reason = "平移变量p3=q23-p2的线性部分为I_4，绝对行列式为1。"
            delta4_consumption = "vertex 3 delta^4 was already consumed by LocalReductionTrace when integrating p3; not reused here."
        else:
            source_local_constraint = None
            source_delta4 = {
                "source_layer": "ExpandedTwoBodyFactor.momentum_delta",
                "vertex": source_vertex,
                "expression": factors[source_vertex].momentum_delta,
            }
            substitutions = (
                ({"from": "q_{4,1}", "to": "p_2 + p_3", "source_step": "q23_translation"},)
                if key == (1, 0) else ()
            )
            jacobian_reason = "对被积q四动量的delta^4线性系数为±I_4，绝对行列式为1。"
            delta4_consumption = f"vertex_{source_vertex}_delta4_integrated_{q_measure.momentum_expression}"
        relation = spec["solved"]
        if q_measure.momentum_expression not in relation:
            raise ValueError(f"q measure {key} variable is absent from its recorded substitution")
        steps.append(MomentumEliminationStep(
            step_id=f"q_elimination_{index}",
            operation=spec["operation"],
            pair_momentum_label=spec["pair_label"],
            q_momentum=q_measure.momentum_expression,
            edge_index=key[0],
            occurrence=key[1],
            source_vertex=q_measure.source_reference.source,  # type: ignore[arg-type]
            target_vertex=q_measure.source_reference.target,  # type: ignore[arg-type]
            relation=relation,
            substitutions=substitutions,
            source_conservation_vertex=source_vertex if source_delta4 is not None else None,
            source_conservation_delta4=source_delta4,
            source_local_constraint=source_local_constraint,
            source_q_measure=q_measure.to_dict(),
            absolute_jacobian="1",
            jacobian_reason=jacobian_reason,
            delta4_consumption=delta4_consumption,
            shell_transfer=transfer,
        ))

    residual_equation = "P = p_1 + p_2 + p_3 + p_4"
    residual_delta4 = "delta^4((P) - (p_1) - (p_2) - (p_3) - (p_4))"
    delta4_dispositions = (
        Delta4Disposition(0, "retained_global_conservation", factors[0].momentum_delta,
                          tuple(ref.to_dict() for ref in factors[0].incoming_references)
                          + tuple(measure.source_reference.to_dict() for measure in factors[0].outgoing_measures),
                          None, residual_delta4),
        Delta4Disposition(1, "integrated_q12", factors[1].momentum_delta,
                          tuple(ref.to_dict() for ref in factors[1].incoming_references)
                          + tuple(measure.source_reference.to_dict() for measure in factors[1].outgoing_measures),
                          "q_elimination_1", None),
        Delta4Disposition(2, "integrated_q34_after_q23_substitution", factors[2].momentum_delta,
                          tuple(ref.to_dict() for ref in factors[2].incoming_references)
                          + tuple(measure.source_reference.to_dict() for measure in factors[2].outgoing_measures),
                          "q_elimination_3", None),
        Delta4Disposition(3, "consumed_once_by_local_p3_integration", factors[3].momentum_delta,
                          tuple(ref.to_dict() for ref in factors[3].incoming_references)
                          + tuple(measure.source_reference.to_dict() for measure in factors[3].outgoing_measures),
                          "LocalReductionTrace.vertex_3", None),
    )
    invariant_deltas = tuple({
        "variable_label": transfers[key].variable_label,
        "expression": transfers[key].result_delta,
        "source_shell_transfer": transfers[key].to_dict(),
        "double_line_differential_preserved": transfers[key].double_line_differential,
    } for key in ((0, 0), (1, 0), (3, 0)))

    ledger = _normalization(expanded)
    return BFCovariantCertificate(
        schema_version=B_F03_CERTIFICATE_SCHEMA_VERSION,
        artifact_type="b_f03_covariant_elimination_trace",
        status="built",
        identity_json=_json(card.identity),
        input_card_digest=expanded.input_card_digest,
        measure_convention=expanded.measure_convention,
        ordered_variables=card.variable_labels,
        source_scope="single fixed-label B/F03 identity only; HAPOF source matching not checked",
        conservation_equations=equations,
        elimination_steps=tuple(steps),
        delta4_dispositions=delta4_dispositions,
        retained_particle_measures=tuple(particle_rows),
        preserved_double_line_factors=tuple(item.to_dict() for item in expanded.double_line_factors),
        residual_global_equation=residual_equation,
        residual_global_delta4=residual_delta4,
        invariant_deltas=invariant_deltas,
        normalization=ledger,
        formula_verified=False,
        source_match_status="not_checked",
    )


__all__ = [
    "B_F03_CERTIFICATE_SCHEMA_VERSION",
    "B_F03_IDENTITY_DIGEST",
    "B_F03_VARIABLES",
    "VertexConservationTrace",
    "ShellDeltaTransfer",
    "MomentumEliminationStep",
    "ParticleMeasureSource",
    "Delta4Disposition",
    "CovariantNormalizationLedger",
    "BFCovariantCertificate",
    "build_b_covariant_certificate",
]
