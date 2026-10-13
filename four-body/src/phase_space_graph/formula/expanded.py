"""Pre-cancellation expansion of the graph-local F0 expression.

This module expands each LM-12 two-body block into its four-momentum delta
and two explicit positive-energy on-shell measures.  It preserves repeated
occurrences.  It does not cancel measures, integrate constraints, or reduce to
a covariant formula.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from ..formula_inputs import FormulaInputCard
from .inputs import validate_formula_card


EXPANDED_F0_SCHEMA_VERSION = "FM-F0-expanded-v1"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class ExpansionSourceReference:
    """A graph/input-card location retained by an expanded factor."""

    source_layer: str
    role: str
    vertex: int | None = None
    edge_index: int | None = None
    occurrence: int | None = None
    source: int | None = None
    target: int | None = None
    slot: int | None = None
    particle_ids: tuple[int, ...] = ()
    momentum_expression: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "particle_ids", tuple(self.particle_ids))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_layer": self.source_layer,
            "role": self.role,
            "vertex": self.vertex,
            "edge_index": self.edge_index,
            "occurrence": self.occurrence,
            "source": self.source,
            "target": self.target,
            "slot": self.slot,
            "particle_ids": list(self.particle_ids),
            "momentum_expression": self.momentum_expression,
        }


@dataclass(frozen=True, slots=True)
class OnShellMeasureExpansion:
    """One explicit ``dmu(k)`` belonging to one outgoing occurrence.

    ``momentum_expression`` is the independent integration variable.  For a
    double line, ``support_momentum_expression`` retains the input card's
    particle sum without treating that sum as the integration variable.
    """

    momentum_expression: str
    support_momentum_expression: str
    mass: str
    mass_squared: str
    particle_ids: tuple[int, ...]
    differential: str
    shell_delta: str
    positive_energy: str
    denominator: str
    source_reference: ExpansionSourceReference

    def __post_init__(self) -> None:
        object.__setattr__(self, "particle_ids", tuple(self.particle_ids))

    @property
    def expression(self) -> str:
        return (
            f"{self.differential} {self.shell_delta} {self.positive_energy}"
            f"/{self.denominator}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "on_shell_measure",
            "definition": "d^4k delta(k^2-m^2) theta(k^0)/(2*pi)^3",
            "momentum_expression": self.momentum_expression,
            "support_momentum_expression": self.support_momentum_expression,
            "mass": self.mass,
            "mass_squared": self.mass_squared,
            "particle_ids": list(self.particle_ids),
            "differential": self.differential,
            "shell_delta": self.shell_delta,
            "positive_energy": self.positive_energy,
            "denominator": self.denominator,
            "expression": self.expression,
            "source_reference": self.source_reference.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExpandedTwoBodyFactor:
    """One fully expanded local ``dPhi2`` factor before any cancellation."""

    vertex: int
    measure_convention: str
    incoming_momentum: str
    outgoing_integration_momenta: tuple[str, str]
    outgoing_support_momenta: tuple[str, str]
    momentum_delta: str
    normalization_2pi_power: int
    incoming_references: tuple[ExpansionSourceReference, ...]
    outgoing_measures: tuple[OnShellMeasureExpansion, OnShellMeasureExpansion]

    def __post_init__(self) -> None:
        if self.measure_convention not in {"jing_psf", "standard_2pi4"}:
            raise ValueError(f"unknown measure convention: {self.measure_convention!r}")
        if self.normalization_2pi_power not in (0, 4):
            raise ValueError("a local two-body normalization must be 1 or (2*pi)^4")
        object.__setattr__(self, "outgoing_integration_momenta", tuple(self.outgoing_integration_momenta))
        object.__setattr__(self, "outgoing_support_momenta", tuple(self.outgoing_support_momenta))
        object.__setattr__(self, "incoming_references", tuple(self.incoming_references))
        object.__setattr__(self, "outgoing_measures", tuple(self.outgoing_measures))
        if (len(self.outgoing_integration_momenta) != 2
                or len(self.outgoing_support_momenta) != 2
                or len(self.outgoing_measures) != 2):
            raise ValueError("an expanded dPhi2 must retain exactly two outgoing measures")

    @property
    def normalization_prefactor(self) -> str:
        return "(2*pi)^4" if self.normalization_2pi_power == 4 else "1"

    @property
    def expression(self) -> str:
        pieces = [self.momentum_delta]
        pieces.extend(item.expression for item in self.outgoing_measures)
        if self.normalization_2pi_power:
            pieces.insert(0, self.normalization_prefactor)
        return " * ".join(pieces)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "two_body_factor",
            "vertex": self.vertex,
            "measure_convention": self.measure_convention,
            "incoming_momentum": self.incoming_momentum,
            "outgoing_integration_momenta": list(self.outgoing_integration_momenta),
            "outgoing_support_momenta": list(self.outgoing_support_momenta),
            "momentum_delta": self.momentum_delta,
            "normalization_2pi_power": self.normalization_2pi_power,
            "normalization_prefactor": self.normalization_prefactor,
            "outgoing_measures": [item.to_dict() for item in self.outgoing_measures],
            "incoming_references": [item.to_dict() for item in self.incoming_references],
            "expression": self.expression,
        }


@dataclass(frozen=True, slots=True)
class ExpandedDoubleLineFactor:
    """A card-provided double-line differential with its graph occurrence."""

    edge_index: int
    occurrence: int
    variable: tuple[int, ...]
    variable_label: str
    differential: str
    convention: str
    momentum_expression: str
    source_reference: ExpansionSourceReference

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable", tuple(self.variable))

    @property
    def expression(self) -> str:
        return self.differential

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "double_line_measure",
            "edge_index": self.edge_index,
            "occurrence": self.occurrence,
            "variable": list(self.variable),
            "variable_label": self.variable_label,
            "differential": self.differential,
            "convention": self.convention,
            "momentum_expression": self.momentum_expression,
            "expression": self.expression,
            "source_reference": self.source_reference.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExpandedF0Expression:
    """Serializable full local F0 product at the pre-cancellation stage."""

    measure_convention: str
    expression: str
    factors: tuple[ExpandedTwoBodyFactor, ...]
    double_line_factors: tuple[ExpandedDoubleLineFactor, ...]
    input_card_digest: str
    identity: dict[str, Any] | None
    provenance: dict[str, Any] | None
    schema_version: str = EXPANDED_F0_SCHEMA_VERSION
    status: str = "expanded_pre_cancellation"
    cancellation_performed: bool = False
    integration_performed: bool = False
    covariant_reduction_performed: bool = False
    formula_verified: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != EXPANDED_F0_SCHEMA_VERSION:
            raise ValueError(f"unsupported expanded F0 schema: {self.schema_version!r}")
        if self.cancellation_performed or self.integration_performed or self.covariant_reduction_performed:
            raise ValueError("FM-03 expanded F0 cannot include cancellation, integration, or reduction")
        if self.formula_verified:
            raise ValueError("an expanded F0 expression is not a verified full formula")
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "double_line_factors", tuple(self.double_line_factors))
        if self.identity is not None:
            object.__setattr__(self, "identity", dict(self.identity))
        if self.provenance is not None:
            object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "expanded_f0_expression",
            "schema_version": self.schema_version,
            "status": self.status,
            "measure_convention": self.measure_convention,
            "expression": self.expression,
            "factors": [item.to_dict() for item in self.factors],
            "double_line_factors": [item.to_dict() for item in self.double_line_factors],
            "input_card_digest": self.input_card_digest,
            "identity": _jsonable(self.identity),
            "provenance": _jsonable(self.provenance),
            "cancellation_performed": self.cancellation_performed,
            "integration_performed": self.integration_performed,
            "covariant_reduction_performed": self.covariant_reduction_performed,
            "formula_verified": self.formula_verified,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)


def _incoming_source(ref: dict[str, object], vertex: int) -> ExpansionSourceReference:
    return ExpansionSourceReference(
        source_layer="TwoBodyFactor.incoming_references",
        role=str(ref.get("role", "unknown")),
        vertex=vertex,
        edge_index=ref.get("edge_index"),  # type: ignore[arg-type]
        occurrence=ref.get("occurrence"),  # type: ignore[arg-type]
        source=ref.get("source"),  # type: ignore[arg-type]
        target=ref.get("target"),  # type: ignore[arg-type]
    )


def _outgoing_source(factor_vertex: int, line: Any) -> ExpansionSourceReference:
    return ExpansionSourceReference(
        source_layer="TwoBodyFactor.outgoing",
        role=str(line.role),
        vertex=factor_vertex,
        edge_index=line.edge_index,
        occurrence=line.occurrence,
        source=line.source,
        target=line.target,
        slot=line.slot,
        particle_ids=tuple(line.particle_ids),
        momentum_expression=line.momentum_expression,
    )


def _outgoing_integration_momentum(line: Any) -> str:
    if line.role == "double":
        if line.edge_index is None or line.occurrence is None:
            raise ValueError("a double-line integration variable requires its edge occurrence")
        return f"q_{{{line.edge_index + 1},{line.occurrence + 1}}}"
    if line.role in {"single", "external"}:
        return line.momentum_expression
    raise ValueError(f"unsupported outgoing connection role: {line.role!r}")


def _expand_two_body_factor(card: FormulaInputCard, factor: Any) -> ExpandedTwoBodyFactor:
    if factor.status != "ready" or len(factor.outgoing) != 2:
        raise ValueError(f"vertex {factor.vertex} is not a ready two-body factor")
    sources = tuple(_outgoing_source(factor.vertex, line) for line in factor.outgoing)
    integration_momenta = tuple(_outgoing_integration_momentum(line) for line in factor.outgoing)
    measures = tuple(
        OnShellMeasureExpansion(
            momentum_expression=integration_momentum,
            support_momentum_expression=line.momentum_expression,
            mass=line.mass,
            mass_squared=line.mass_squared,
            particle_ids=tuple(line.particle_ids),
            differential=f"d^4({integration_momentum})",
            shell_delta=f"delta(({integration_momentum})^2 - {line.mass_squared})",
            positive_energy=f"theta(({integration_momentum})^0)",
            denominator="(2*pi)^3",
            source_reference=source,
        )
        for line, integration_momentum, source in zip(factor.outgoing, integration_momenta, sources)
    )
    support_momenta = tuple(line.momentum_expression for line in factor.outgoing)
    prefactor_power = 4 if card.measure_convention == "standard_2pi4" else 0
    return ExpandedTwoBodyFactor(
        vertex=factor.vertex,
        measure_convention=card.measure_convention,
        incoming_momentum=factor.incoming_momentum,
        outgoing_integration_momenta=integration_momenta,
        outgoing_support_momenta=support_momenta,
        momentum_delta=(
            f"delta^4(({factor.incoming_momentum}) - "
            f"({integration_momenta[0]}) - ({integration_momenta[1]}))"
        ),
        normalization_2pi_power=prefactor_power,
        incoming_references=tuple(
            _incoming_source(ref, factor.vertex) for ref in factor.incoming_references
        ),
        outgoing_measures=measures,
    )


def _expand_double_line(card: FormulaInputCard, item: Any) -> ExpandedDoubleLineFactor:
    return ExpandedDoubleLineFactor(
        edge_index=item.edge_index,
        occurrence=item.occurrence,
        variable=tuple(item.variable),
        variable_label=item.variable_label,
        differential=item.differential,
        convention=item.convention,
        momentum_expression=item.momentum_expression,
        source_reference=ExpansionSourceReference(
            source_layer="FormulaInputCard.double_measures",
            role="double_measure",
            edge_index=item.edge_index,
            occurrence=item.occurrence,
            particle_ids=tuple(item.variable),
            momentum_expression=item.momentum_expression,
        ),
    )


def build_expanded_f0_expression(card: FormulaInputCard) -> ExpandedF0Expression:
    """Expand all graph-local two-body factors before cancellation/reduction.

    Each outgoing occurrence receives one explicit ``dmu`` according to
    ``dmu(k)=d^4k delta(k^2-m^2) theta(k^0)/(2*pi)^3``.  For
    ``standard_2pi4``, each local two-body factor also receives its specified
    ``(2*pi)^4`` prefactor.  Double-line differentials are copied from the
    validated input card unchanged.
    """

    if not isinstance(card, FormulaInputCard):
        raise TypeError("build_expanded_f0_expression expects a FormulaInputCard")
    card = validate_formula_card(card)
    factors = tuple(_expand_two_body_factor(card, item) for item in card.factors)
    doubles = tuple(_expand_double_line(card, item) for item in card.double_measures)
    expression = " * ".join(
        tuple(item.expression for item in factors)
        + tuple(item.expression for item in doubles)
    ) or "1"
    input_digest = hashlib.sha256(card.to_json().encode("utf-8")).hexdigest()
    return ExpandedF0Expression(
        measure_convention=card.measure_convention,
        expression=expression,
        factors=factors,
        double_line_factors=doubles,
        input_card_digest=input_digest,
        identity=_jsonable(card.identity),
        provenance=_jsonable(card.provenance),
    )


__all__ = [
    "EXPANDED_F0_SCHEMA_VERSION",
    "ExpansionSourceReference",
    "OnShellMeasureExpansion",
    "ExpandedTwoBodyFactor",
    "ExpandedDoubleLineFactor",
    "ExpandedF0Expression",
    "build_expanded_f0_expression",
]
