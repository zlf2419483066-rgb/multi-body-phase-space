"""F0 and covariant intermediate expression builders."""

from __future__ import annotations

from ..formula_inputs import FormulaInputCard
from .records import CovariantResult, RawExpression, RuleStep
from .inputs import validate_formula_card


def _require_card(card: FormulaInputCard) -> FormulaInputCard:
    if not isinstance(card, FormulaInputCard):
        raise TypeError("formula generation expects a FormulaInputCard")
    if card.availability != "ready":
        raise ValueError(card.unavailable_reason or "formula input is unavailable")
    return validate_formula_card(card)


def build_f0_expression(card: FormulaInputCard) -> RawExpression:
    """Build the auditable, uncollapsed graph expression from LM-12."""

    card = _require_card(card)
    factors = tuple(
        f"dPhi2[{card.measure_convention}](Q_v{factor.vertex}={factor.incoming_momentum}; "
        + ", ".join(line.momentum_expression for line in factor.outgoing) + ")"
        for factor in card.factors
    )
    measures = tuple(item.differential for item in card.double_measures)
    cancellations = tuple(
        f"CancelCopies[dmu(p_{item.particle_id})^{item.cancellation_count}]"
        for item in card.particle_measures if item.cancellation_count
    )
    product = " * ".join(factors + measures + cancellations) or "1"
    steps = (
        RuleStep(
            "F0-local-factors",
            "complete physical labeling",
            product,
            evidence="LM-12 formula input card",
        ),
    )
    return RawExpression(
        expression=product,
        factors=factors,
        double_measures=measures,
        cancellations=cancellations,
        steps=steps,
        factor_sources=tuple(factor.to_dict() for factor in card.factors),
        measure_sources=tuple(item.to_dict() for item in card.double_measures),
        cancellation_sources=tuple(item.to_dict() for item in card.particle_measures),
    )


def reduce_covariant(card: FormulaInputCard, raw: RawExpression | None = None) -> CovariantResult:
    """Show the specified target, pending the FM-04 elimination certificate.

    A structural tree test does not execute measure cancellation, elimination,
    or the normalization ledger.  Never emit a fabricated completed RuleStep.
    """

    card = _require_card(card)
    expected_raw = build_f0_expression(card)
    if raw is not None and raw != expected_raw:
        raise ValueError("raw expression does not belong to the supplied formula card")
    raw = expected_raw
    loop = card.loop_count if card.loop_count is not None else 0
    if loop < 0:
        return CovariantResult(raw.expression, (), "blocked", diagnostics=("invalid loop count",))
    constraints = tuple(
        f"delta(s{''.join(map(str, variable))}-(" + "+".join(f"p_{i}" for i in variable) + ")^2)"
        for variable in card.independent_variables
    )
    expression = (
        f"dPhi4[{card.measure_convention}](P; p_1,p_2,p_3,p_4) * "
        + " * ".join(
            f"ds{''.join(map(str, variable))} {constraint}"
            for variable, constraint in zip(card.independent_variables, constraints)
        )
    )
    return CovariantResult(
        expression=expression,
        remaining_constraints=constraints,
        status="pending",
        steps=raw.steps,
        diagnostics=("FM-04 target expression only: tree elimination, repeated-measure cancellation "
                     "and normalization certificates are not implemented; no graph-derived F1 claim",),
    )


build_raw_expression = build_f0_expression


__all__ = ["build_f0_expression", "build_raw_expression", "reduce_covariant"]
