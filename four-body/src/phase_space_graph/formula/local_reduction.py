"""Auditable local measure cancellation and delta-four integration traces.

This module applies only the one-vertex rules represented by an explicit
``CancellationPlan``.  A returned trace is partial bookkeeping; it is not a
global covariant reduction or a verified phase-space formula.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .cancellations import CancellationPlan, OccurrenceLocator, ParticleOccurrence
from .expanded import (
    EXPANDED_F0_SCHEMA_VERSION,
    ExpandedF0Expression,
    ExpandedTwoBodyFactor,
    OnShellMeasureExpansion,
)


LOCAL_REDUCTION_SCHEMA_VERSION = "FM-local-reduction-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class TwoPiAccounting:
    local_prefactor_power: int
    original_measure_denominator_powers: tuple[int, int]
    canceled_measure_denominator_powers: tuple[int, ...]
    residual_measure_denominator_powers: tuple[int, ...]
    net_local_power: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_measure_denominator_powers", tuple(self.original_measure_denominator_powers))
        object.__setattr__(self, "canceled_measure_denominator_powers", tuple(self.canceled_measure_denominator_powers))
        object.__setattr__(self, "residual_measure_denominator_powers", tuple(self.residual_measure_denominator_powers))

    def to_dict(self) -> dict[str, object]:
        return {
            "local_prefactor_power": self.local_prefactor_power,
            "original_measure_denominator_powers": list(self.original_measure_denominator_powers),
            "canceled_measure_denominator_powers": list(self.canceled_measure_denominator_powers),
            "residual_measure_denominator_powers": list(self.residual_measure_denominator_powers),
            "net_local_power": self.net_local_power,
        }


@dataclass(frozen=True, slots=True)
class RemainingConstraintTrace:
    kind: str
    delta: str
    positive_energy: str | None
    denominator_power: int | None
    source_local_delta4: dict[str, Any]
    source_measure: dict[str, Any] | None
    awaiting: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "delta": self.delta,
            "positive_energy": self.positive_energy,
            "denominator_power": self.denominator_power,
            "source_local_delta4": dict(self.source_local_delta4),
            "source_measure": dict(self.source_measure) if self.source_measure is not None else None,
            "awaiting": self.awaiting,
        }


@dataclass(frozen=True, slots=True)
class LocalFactorTrace:
    vertex: int
    status: str
    source_factor: dict[str, Any]
    canceled_occurrences: tuple[dict[str, object], ...]
    removed_measures: tuple[dict[str, Any], ...]
    integrated_measure: dict[str, Any] | None
    integrated_variable: str | None
    delta4_jacobian: str | None
    delta4_jacobian_reason: str | None
    remaining_constraint: RemainingConstraintTrace | None
    two_pi_accounting: TwoPiAccounting
    after_expression: str
    source_references: tuple[dict[str, Any], ...]
    note: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "canceled_occurrences", tuple(dict(item) for item in self.canceled_occurrences))
        object.__setattr__(self, "removed_measures", tuple(dict(item) for item in self.removed_measures))
        object.__setattr__(self, "source_references", tuple(dict(item) for item in self.source_references))

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex": self.vertex,
            "status": self.status,
            "source_factor": dict(self.source_factor),
            "canceled_occurrences": [dict(item) for item in self.canceled_occurrences],
            "removed_measures": [dict(item) for item in self.removed_measures],
            "integrated_measure": dict(self.integrated_measure) if self.integrated_measure is not None else None,
            "integrated_variable": self.integrated_variable,
            "delta4_jacobian": self.delta4_jacobian,
            "delta4_jacobian_reason": self.delta4_jacobian_reason,
            "remaining_constraint": self.remaining_constraint.to_dict() if self.remaining_constraint else None,
            "two_pi_accounting": self.two_pi_accounting.to_dict(),
            "after_expression": self.after_expression,
            "source_references": [dict(item) for item in self.source_references],
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class LocalReductionTrace:
    schema_version: str
    input_card_digest: str
    measure_convention: str
    formula_identity_json: str
    status: str
    factors: tuple[LocalFactorTrace, ...]
    preserved_double_line_factors: tuple[dict[str, Any], ...]
    pending_reasons: tuple[str, ...]
    covariant_status: str = "pending"
    covariant_certificate_built: bool = False
    full_covariant_reduction_performed: bool = False
    formula_verified: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != LOCAL_REDUCTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported local-reduction schema: {self.schema_version!r}")
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "preserved_double_line_factors", tuple(dict(item) for item in self.preserved_double_line_factors))
        object.__setattr__(self, "pending_reasons", tuple(self.pending_reasons))
        if self.covariant_status != "pending" or self.covariant_certificate_built or self.full_covariant_reduction_performed or self.formula_verified:
            raise ValueError("a local reduction trace cannot claim a covariant certificate or verified formula")

    @property
    def formula_identity(self) -> dict[str, Any]:
        return json.loads(self.formula_identity_json)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "local_reduction_trace",
            "schema_version": self.schema_version,
            "input_card_digest": self.input_card_digest,
            "measure_convention": self.measure_convention,
            "formula_identity": self.formula_identity,
            "status": self.status,
            "factors": [item.to_dict() for item in self.factors],
            "preserved_double_line_factors": [dict(item) for item in self.preserved_double_line_factors],
            "pending_reasons": list(self.pending_reasons),
            "covariant_status": self.covariant_status,
            "covariant_certificate_built": self.covariant_certificate_built,
            "full_covariant_reduction_performed": self.full_covariant_reduction_performed,
            "formula_verified": self.formula_verified,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


def _prefactor(power: int) -> str:
    return "(2*pi)^4" if power == 4 else "1"


def _locator_for_measure(measure: OnShellMeasureExpansion) -> OccurrenceLocator | None:
    source = measure.source_reference
    if source.role == "external":
        if source.vertex is None or source.slot is None:
            raise ValueError("external expanded measure is missing vertex/slot source coordinates")
        return OccurrenceLocator("external", source.vertex, source.slot)
    if source.role == "single":
        if source.edge_index is None or source.occurrence is None:
            raise ValueError("internal expanded measure is missing its edge occurrence")
        return OccurrenceLocator("internal", source.edge_index, source.occurrence)
    return None


def _reference_matches(occurrence: ParticleOccurrence, measure: OnShellMeasureExpansion, factor: ExpandedTwoBodyFactor) -> bool:
    source = measure.source_reference
    if source.source_layer != "TwoBodyFactor.outgoing" or source.particle_ids != (occurrence.particle_id,):
        return False
    if occurrence.kind == "external":
        return (
            source.role == "external"
            and source.vertex == factor.vertex == occurrence.vertex
            and source.slot == occurrence.slot
            and source.edge_index is None
            and source.occurrence is None
            and source.source == factor.vertex
            and source.target is None
            and source.momentum_expression == measure.momentum_expression == measure.support_momentum_expression
        )
    return (
        source.role == "single"
        and source.vertex == factor.vertex == occurrence.source_vertex
        and source.edge_index == occurrence.edge_index
        and source.occurrence == occurrence.line_occurrence
        and source.source == occurrence.source_vertex
        and source.target == occurrence.target_vertex
        and source.slot is None
        and source.momentum_expression == measure.momentum_expression == measure.support_momentum_expression
    )


def _validate_inputs(plan: CancellationPlan, expanded: ExpandedF0Expression) -> dict[OccurrenceLocator, ParticleOccurrence]:
    if not isinstance(plan, CancellationPlan):
        raise TypeError("reduce_local_measures expects a CancellationPlan")
    if not isinstance(expanded, ExpandedF0Expression):
        raise TypeError("reduce_local_measures expects an ExpandedF0Expression")
    if expanded.schema_version != EXPANDED_F0_SCHEMA_VERSION or expanded.status != "expanded_pre_cancellation":
        raise ValueError("local reduction requires a fresh FM-03 pre-cancellation expansion")
    if expanded.cancellation_performed or expanded.integration_performed or expanded.covariant_reduction_performed or expanded.formula_verified:
        raise ValueError("local reduction cannot consume an already canceled or reduced expansion")
    if expanded.input_card_digest != plan.input_card_digest:
        raise ValueError("CancellationPlan and ExpandedF0Expression input-card digests disagree")
    if expanded.measure_convention != plan.measure_convention:
        raise ValueError("CancellationPlan and ExpandedF0Expression measure conventions disagree")
    if expanded.identity is None or _canonical_json(expanded.identity) != plan.formula_identity_json:
        raise ValueError("CancellationPlan and ExpandedF0Expression marked identities disagree")

    expected_prefactor = 4 if expanded.measure_convention == "standard_2pi4" else 0
    if expanded.measure_convention not in {"jing_psf", "standard_2pi4"}:
        raise ValueError(f"unsupported measure convention: {expanded.measure_convention!r}")
    if not expanded.factors:
        raise ValueError("expanded F0 must contain local two-body factors")
    vertices = [factor.vertex for factor in expanded.factors]
    if len(vertices) != len(set(vertices)):
        raise ValueError("expanded F0 contains duplicate local factors for a vertex")

    by_locator: dict[OccurrenceLocator, ParticleOccurrence] = {}
    canceled_locators: set[OccurrenceLocator] = set()
    for particle_plan in plan.particles:
        for occurrence in particle_plan.occurrences:
            if occurrence.locator in by_locator:
                raise ValueError(f"cancellation plan assigns occurrence {occurrence.occurrence_id} more than once")
            by_locator[occurrence.locator] = occurrence
        for occurrence in particle_plan.cancellation_occurrences:
            if occurrence.locator in canceled_locators:
                raise ValueError(f"cancellation plan cancels occurrence {occurrence.occurrence_id} more than once")
            canceled_locators.add(occurrence.locator)

    seen: dict[OccurrenceLocator, int] = {}
    for factor in expanded.factors:
        if factor.measure_convention != expanded.measure_convention or factor.normalization_2pi_power != expected_prefactor:
            raise ValueError(f"vertex {factor.vertex} normalization disagrees with its measure convention")
        if (len(factor.outgoing_measures) != 2 or len(factor.outgoing_integration_momenta) != 2
                or len(factor.outgoing_support_momenta) != 2):
            raise ValueError(f"vertex {factor.vertex} must have exactly two outgoing measures")
        for index, measure in enumerate(factor.outgoing_measures):
            if measure.denominator != "(2*pi)^3":
                raise ValueError(f"vertex {factor.vertex} has an unsupported outgoing measure denominator")
            if measure.momentum_expression != factor.outgoing_integration_momenta[index]:
                raise ValueError(f"vertex {factor.vertex} outgoing measure variable disagrees with its source factor")
            if measure.support_momentum_expression != factor.outgoing_support_momenta[index]:
                raise ValueError(f"vertex {factor.vertex} outgoing measure support disagrees with its source factor")
            locator = _locator_for_measure(measure)
            if locator is None:
                if measure.source_reference.role != "double":
                    raise ValueError(f"vertex {factor.vertex} has an unsupported outgoing measure role")
                continue
            occurrence = by_locator.get(locator)
            if occurrence is None or not _reference_matches(occurrence, measure, factor):
                raise ValueError(f"expanded outgoing measure {locator} does not bind to the cancellation plan")
            seen[locator] = seen.get(locator, 0) + 1
    if set(seen) != set(by_locator) or any(count != 1 for count in seen.values()):
        raise ValueError("expanded single/external measures do not cover plan occurrences exactly once")
    for item in expanded.double_line_factors:
        if item.convention != expanded.measure_convention:
            raise ValueError("double-line differential convention disagrees with expanded F0")
    return by_locator


def _accounting(prefactor_power: int, canceled_count: int, remaining_count: int) -> TwoPiAccounting:
    original = (3, 3)
    canceled = (3,) * canceled_count
    remaining = (3,) * remaining_count
    return TwoPiAccounting(
        local_prefactor_power=prefactor_power,
        original_measure_denominator_powers=original,
        canceled_measure_denominator_powers=canceled,
        residual_measure_denominator_powers=remaining,
        net_local_power=prefactor_power - sum(original) + sum(canceled),
    )


def reduce_local_measures(plan: CancellationPlan, expanded: ExpandedF0Expression) -> LocalReductionTrace:
    """Apply occurrence-bound local rules and return a pending/partial trace.

    Zero canceled measures preserve a vertex factor.  One canceled measure is
    removed, then the other outgoing four-momentum is integrated against the
    vertex's ``delta^4`` with unit absolute Jacobian.  Two canceled measures
    leave the local ``delta^4`` untouched and explicitly await cross-vertex
    integration.  Double-line ``ds`` factors are copied verbatim.
    """

    occurrence_by_locator = _validate_inputs(plan, expanded)
    canceled_locators = {
        occurrence.locator
        for particle_plan in plan.particles
        for occurrence in particle_plan.cancellation_occurrences
    }
    traces: list[LocalFactorTrace] = []
    pending: list[str] = []
    reduced_count = 0

    for factor in expanded.factors:
        canceled_pairs: list[tuple[ParticleOccurrence, OnShellMeasureExpansion]] = []
        for measure in factor.outgoing_measures:
            locator = _locator_for_measure(measure)
            if locator is not None and locator in canceled_locators:
                canceled_pairs.append((occurrence_by_locator[locator], measure))
        if len(canceled_pairs) > 2:
            raise ValueError(f"vertex {factor.vertex} has more canceled measures than outgoing slots")

        prefactor_power = factor.normalization_2pi_power
        prefix = _prefactor(prefactor_power)
        source_factor = factor.to_dict()
        source_delta4 = {
            "source_layer": "ExpandedTwoBodyFactor.momentum_delta",
            "vertex": factor.vertex,
            "expression": factor.momentum_delta,
        }
        canceled_occurrences = tuple(item.to_dict() for item, _ in canceled_pairs)
        removed_measures = tuple({
            "canceled_occurrence": item.to_dict(),
            "measure": measure.to_dict(),
        } for item, measure in canceled_pairs)
        all_sources = tuple(
            [ref.to_dict() for ref in factor.incoming_references]
            + [measure.source_reference.to_dict() for measure in factor.outgoing_measures]
        )

        if not canceled_pairs:
            accounting = _accounting(prefactor_power, 0, 2)
            traces.append(LocalFactorTrace(
                vertex=factor.vertex,
                status="preserved",
                source_factor=source_factor,
                canceled_occurrences=(),
                removed_measures=(),
                integrated_measure=None,
                integrated_variable=None,
                delta4_jacobian=None,
                delta4_jacobian_reason=None,
                remaining_constraint=None,
                two_pi_accounting=accounting,
                after_expression=factor.expression,
                source_references=all_sources,
                note="该顶点没有选中抵消出现；局部因子原样保留。",
            ))
            continue

        if len(canceled_pairs) == 1:
            canceled_occurrence, removed_measure = canceled_pairs[0]
            other_measures = [measure for measure in factor.outgoing_measures if measure is not removed_measure]
            if len(other_measures) != 1:
                raise ValueError(f"vertex {factor.vertex} cannot uniquely bind the other outgoing measure")
            integrated = other_measures[0]
            q = factor.incoming_momentum
            p = removed_measure.momentum_expression
            difference = f"(({q}) - ({p}))"
            residual_delta = f"delta({difference}^2 - {integrated.mass_squared})"
            residual_theta = f"theta({difference}^0)"
            residual_constraint = RemainingConstraintTrace(
                kind="on_shell_residual",
                delta=residual_delta,
                positive_energy=residual_theta,
                denominator_power=3,
                source_local_delta4=source_delta4,
                source_measure=integrated.to_dict(),
            )
            local_piece = f"{residual_delta} * {residual_theta} / (2*pi)^3"
            after = local_piece if prefix == "1" else f"{prefix} * {local_piece}"
            accounting = _accounting(prefactor_power, 1, 1)
            integrated_source = integrated.source_reference.to_dict()
            traces.append(LocalFactorTrace(
                vertex=factor.vertex,
                status="locally_reduced",
                source_factor=source_factor,
                canceled_occurrences=canceled_occurrences,
                removed_measures=removed_measures,
                integrated_measure=integrated.to_dict(),
                integrated_variable=integrated.momentum_expression,
                delta4_jacobian="1",
                delta4_jacobian_reason="对被积四动量的delta^4线性系数为−I_4，绝对行列式为1。",
                remaining_constraint=residual_constraint,
                two_pi_accounting=accounting,
                after_expression=after,
                source_references=(source_delta4, removed_measure.source_reference.to_dict(), integrated_source),
                note="移除指定的dmu出现，并用局部delta^4积分另一出射四动量；未消去的约束仍显式保留。",
            ))
            reduced_count += 1
            continue

        residual_constraint = RemainingConstraintTrace(
            kind="local_delta4_pending",
            delta=factor.momentum_delta,
            positive_energy=None,
            denominator_power=None,
            source_local_delta4=source_delta4,
            source_measure=None,
            awaiting="cross_vertex_integration",
        )
        after = factor.momentum_delta if prefix == "1" else f"{prefix} * {factor.momentum_delta}"
        accounting = _accounting(prefactor_power, 2, 0)
        pending.append(f"顶点 {factor.vertex} 两个出射测度均被取消；保留局部delta^4，等待跨顶点安排积分。")
        traces.append(LocalFactorTrace(
            vertex=factor.vertex,
            status="pending_cross_vertex_integration",
            source_factor=source_factor,
            canceled_occurrences=canceled_occurrences,
            removed_measures=removed_measures,
            integrated_measure=None,
            integrated_variable=None,
            delta4_jacobian=None,
            delta4_jacobian_reason=None,
            remaining_constraint=residual_constraint,
            two_pi_accounting=accounting,
            after_expression=after,
            source_references=(source_delta4,) + tuple(measure.source_reference.to_dict() for measure in factor.outgoing_measures),
            note="两个出射测度都已被取消，没有可称为‘另一保留测度’的局部积分变量。",
        ))

    if pending:
        status = "pending"
    elif reduced_count:
        status = "partial"
    else:
        status = "pending"
        pending.append("没有单顶点守恒积分可执行；本产物保留为待处理局部账本。")

    return LocalReductionTrace(
        schema_version=LOCAL_REDUCTION_SCHEMA_VERSION,
        input_card_digest=expanded.input_card_digest,
        measure_convention=expanded.measure_convention,
        formula_identity_json=_canonical_json(expanded.identity),
        status=status,
        factors=tuple(traces),
        preserved_double_line_factors=tuple(item.to_dict() for item in expanded.double_line_factors),
        pending_reasons=tuple(pending),
    )


__all__ = [
    "LOCAL_REDUCTION_SCHEMA_VERSION",
    "TwoPiAccounting",
    "RemainingConstraintTrace",
    "LocalFactorTrace",
    "LocalReductionTrace",
    "reduce_local_measures",
]
