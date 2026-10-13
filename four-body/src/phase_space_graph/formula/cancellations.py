"""Occurrence-aware particle-measure cancellation plans.

The plan records which concrete occurrence of each particle is retained and
which are canceled.  It is a formula-layer choice only: it never rewrites the
physical record, marked identity, or drawing input, and it performs no
integration or physical elimination.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from ..formula_inputs import FormulaInputCard, ParticleMeasureInput
from .inputs import validate_formula_card


CANCELLATION_PLAN_SCHEMA_VERSION = "FM-cancellation-plan-v1"
LM12_DEFAULT_POLICY = "lm12_external_retained_internal_cancelled"
FORMULA_OVERRIDE_POLICY = "formula_level_cancellation_override"


def _is_coordinate(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True, slots=True)
class OccurrenceLocator:
    """A stable selector using the coordinates stored by ``ParticleMeasureInput``."""

    kind: str
    first: int
    second: int

    def __post_init__(self) -> None:
        if self.kind not in {"external", "internal"}:
            raise ValueError("occurrence kind must be 'external' or 'internal'")
        if not _is_coordinate(self.first) or not _is_coordinate(self.second):
            raise ValueError("occurrence coordinates must be non-negative integers")

    @classmethod
    def from_tuple(cls, value: tuple[str, int, int] | list[object]) -> OccurrenceLocator:
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise TypeError("an occurrence locator must be (kind, first, second)")
        kind, first, second = value
        if not isinstance(kind, str):
            raise TypeError("occurrence kind must be a string")
        return cls(kind, first, second)  # type: ignore[arg-type]

    def sort_key(self) -> tuple[int, int, int]:
        return (0 if self.kind == "external" else 1, self.first, self.second)

    def to_dict(self) -> dict[str, object]:
        if self.kind == "external":
            return {"kind": self.kind, "vertex": self.first, "slot": self.second}
        return {"kind": self.kind, "edge_index": self.first, "occurrence": self.second}


@dataclass(frozen=True, slots=True)
class ParticleOccurrence:
    """One particle occurrence bound to its actual graph location."""

    particle_id: int
    kind: str
    vertex: int | None = None
    slot: int | None = None
    edge_index: int | None = None
    line_occurrence: int | None = None
    source_vertex: int | None = None
    target_vertex: int | None = None

    def __post_init__(self) -> None:
        if not _is_coordinate(self.particle_id) or self.particle_id == 0:
            raise ValueError("particle_id must be a positive integer")
        if self.kind == "external":
            values = (self.vertex, self.slot)
            if any(value is None or not _is_coordinate(value) for value in values):
                raise ValueError("external occurrences require non-negative vertex and slot")
            if any(value is not None for value in (self.edge_index, self.line_occurrence, self.source_vertex, self.target_vertex)):
                raise ValueError("external occurrences cannot carry internal-line coordinates")
        elif self.kind == "internal":
            values = (self.edge_index, self.line_occurrence, self.source_vertex, self.target_vertex)
            if any(value is None or not _is_coordinate(value) for value in values):
                raise ValueError("internal occurrences require edge, line occurrence, source, and target")
            if self.slot is not None or self.vertex is not None:
                raise ValueError("internal occurrences cannot carry external vertex/slot coordinates")
        else:
            raise ValueError("occurrence kind must be 'external' or 'internal'")

    @property
    def locator(self) -> OccurrenceLocator:
        if self.kind == "external":
            assert self.vertex is not None and self.slot is not None
            return OccurrenceLocator("external", self.vertex, self.slot)
        assert self.edge_index is not None and self.line_occurrence is not None
        return OccurrenceLocator("internal", self.edge_index, self.line_occurrence)

    @property
    def occurrence_id(self) -> str:
        if self.kind == "external":
            return f"p{self.particle_id}:external:v{self.vertex}:slot{self.slot}"
        return f"p{self.particle_id}:internal:e{self.edge_index}:o{self.line_occurrence}"

    def to_dict(self) -> dict[str, object]:
        return {
            "occurrence_id": self.occurrence_id,
            "particle_id": self.particle_id,
            **self.locator.to_dict(),
            "source_vertex": self.source_vertex,
            "target_vertex": self.target_vertex,
        }


@dataclass(frozen=True, slots=True)
class CancellationPlanOverride:
    """Formula-level request naming every occurrence to cancel for one particle.

    Exactly one occurrence must remain after applying the request.  Selecting
    an external occurrence here makes that occurrence the canceled one while
    retaining the unique remainder, without changing LM-12 defaults.
    """

    particle_id: int
    cancellation_occurrences: tuple[OccurrenceLocator, ...]
    reason: str

    def __post_init__(self) -> None:
        if not _is_coordinate(self.particle_id) or self.particle_id == 0:
            raise ValueError("particle_id must be a positive integer")
        occurrences = tuple(self.cancellation_occurrences)
        if any(not isinstance(item, OccurrenceLocator) for item in occurrences):
            raise TypeError("cancellation_occurrences must contain OccurrenceLocator values")
        if len(set(occurrences)) != len(occurrences):
            raise ValueError("cancellation override contains a duplicate occurrence")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("a formula-level cancellation override needs a non-empty reason")
        object.__setattr__(self, "cancellation_occurrences", tuple(sorted(occurrences, key=OccurrenceLocator.sort_key)))
        object.__setattr__(self, "reason", self.reason.strip())

    def to_dict(self) -> dict[str, object]:
        return {
            "particle_id": self.particle_id,
            "cancellation_occurrences": [item.to_dict() for item in self.cancellation_occurrences],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ParticleCancellationPlan:
    particle_id: int
    occurrences: tuple[ParticleOccurrence, ...]
    retained_occurrence: ParticleOccurrence
    cancellation_occurrences: tuple[ParticleOccurrence, ...]
    input_cancellation_count: int
    selection_policy: str
    selection_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurrences", tuple(self.occurrences))
        object.__setattr__(self, "cancellation_occurrences", tuple(self.cancellation_occurrences))
        if self.retained_occurrence.particle_id != self.particle_id:
            raise ValueError("retained occurrence belongs to a different particle")
        if any(item.particle_id != self.particle_id for item in self.occurrences + self.cancellation_occurrences):
            raise ValueError("plan contains an occurrence belonging to a different particle")
        if len(self.occurrences) != len(self.cancellation_occurrences) + 1:
            raise ValueError("exactly one occurrence must be retained")
        if len(set(self.occurrences)) != len(self.occurrences):
            raise ValueError("particle plan contains duplicate occurrences")
        if self.retained_occurrence in self.cancellation_occurrences:
            raise ValueError("retained occurrence is also marked for cancellation")
        if set(self.occurrences) != {self.retained_occurrence, *self.cancellation_occurrences}:
            raise ValueError("retained and canceled occurrences must cover every occurrence exactly once")
        if self.input_cancellation_count < 0:
            raise ValueError("input_cancellation_count cannot be negative")
        if self.input_cancellation_count != sum(item.kind == "internal" for item in self.occurrences):
            raise ValueError("input cancellation count disagrees with internal occurrence metadata")
        if not self.selection_policy or not self.selection_reason.strip():
            raise ValueError("selection policy and reason must be non-empty")

    @property
    def cancellation_count(self) -> int:
        return len(self.cancellation_occurrences)

    def to_dict(self) -> dict[str, object]:
        return {
            "particle_id": self.particle_id,
            "occurrences": [item.to_dict() for item in self.occurrences],
            "retained_occurrence": self.retained_occurrence.to_dict(),
            "cancellation_occurrences": [item.to_dict() for item in self.cancellation_occurrences],
            "input_cancellation_count": self.input_cancellation_count,
            "cancellation_count": self.cancellation_count,
            "selection_policy": self.selection_policy,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True, slots=True)
class CancellationPlan:
    schema_version: str
    input_schema_version: str
    input_card_digest: str
    measure_convention: str
    formula_identity_json: str
    selection_policy: str
    particles: tuple[ParticleCancellationPlan, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CANCELLATION_PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported cancellation-plan schema: {self.schema_version!r}")
        object.__setattr__(self, "particles", tuple(self.particles))
        particle_ids = [item.particle_id for item in self.particles]
        if particle_ids != sorted(set(particle_ids)):
            raise ValueError("particle plans must be unique and ordered by particle_id")
        if not self.formula_identity_json:
            raise ValueError("cancellation plan must be bound to a complete formula identity")
        if len(self.input_card_digest) != 64 or any(char not in "0123456789abcdef" for char in self.input_card_digest):
            raise ValueError("input_card_digest must be a lowercase SHA-256 hex digest")
        if self.measure_convention not in {"jing_psf", "standard_2pi4"}:
            raise ValueError(f"unknown cancellation-plan measure convention: {self.measure_convention!r}")

    @property
    def formula_identity(self) -> dict[str, Any]:
        return json.loads(self.formula_identity_json)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "cancellation_plan",
            "schema_version": self.schema_version,
            "input_schema_version": self.input_schema_version,
            "input_card_digest": self.input_card_digest,
            "measure_convention": self.measure_convention,
            "formula_identity": self.formula_identity,
            "selection_policy": self.selection_policy,
            "particles": [item.to_dict() for item in self.particles],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _particle_occurrence(item: ParticleMeasureInput, locator: OccurrenceLocator, factors: tuple[Any, ...]) -> ParticleOccurrence:
    matches: list[ParticleOccurrence] = []
    if locator.kind == "external":
        for factor in factors:
            if factor.vertex != locator.first:
                continue
            for connection in factor.outgoing:
                if connection.role != "external" or connection.slot != locator.second:
                    continue
                if connection.particle_ids != (item.particle_id,):
                    raise ValueError(f"external occurrence {locator} has inconsistent particle identity")
                if any(value is not None for value in (connection.edge_index, connection.occurrence, connection.target)):
                    raise ValueError(f"external occurrence {locator} has invalid line coordinates")
                matches.append(ParticleOccurrence(item.particle_id, "external", vertex=factor.vertex, slot=connection.slot))
    else:
        for factor in factors:
            for connection in factor.outgoing:
                if connection.role != "single" or (connection.edge_index, connection.occurrence) != (locator.first, locator.second):
                    continue
                if connection.particle_ids != (item.particle_id,):
                    raise ValueError(f"internal occurrence {locator} has inconsistent particle identity")
                if connection.source != factor.vertex or connection.target is None:
                    raise ValueError(f"internal occurrence {locator} is missing its endpoint vertices")
                target_refs = [
                    reference
                    for target_factor in factors
                    if target_factor.vertex == connection.target
                    for reference in target_factor.incoming_references
                    if reference.get("role") == "single"
                    and reference.get("edge_index") == locator.first
                    and reference.get("occurrence") == locator.second
                    and reference.get("source") == connection.source
                    and reference.get("target") == connection.target
                ]
                if len(target_refs) != 1:
                    raise ValueError(f"internal occurrence {locator} is not uniquely bound at its target vertex")
                matches.append(ParticleOccurrence(
                    item.particle_id,
                    "internal",
                    edge_index=locator.first,
                    line_occurrence=locator.second,
                    source_vertex=connection.source,
                    target_vertex=connection.target,
                ))
    if len(matches) != 1:
        raise ValueError(f"occurrence {locator} resolves to {len(matches)} formula connections; expected exactly one")
    return matches[0]


def _validated_occurrences(card: FormulaInputCard) -> tuple[tuple[ParticleMeasureInput, tuple[ParticleOccurrence, ...]], ...]:
    if card.availability != "ready" or not card.factors:
        raise ValueError("cancellation planning requires a ready FormulaInputCard")
    particle_ids = [item.particle_id for item in card.particle_measures]
    if particle_ids != sorted(set(particle_ids)):
        raise ValueError("particle-measure groups must be unique and ordered by particle_id")

    actual: dict[OccurrenceLocator, int] = {}
    for factor in card.factors:
        for connection in factor.outgoing:
            if connection.role == "external":
                if connection.slot is None:
                    raise ValueError("external formula connection is missing its slot")
                locator = OccurrenceLocator("external", factor.vertex, connection.slot)
                actual[locator] = actual.get(locator, 0) + 1
            elif connection.role == "single":
                if connection.edge_index is None or connection.occurrence is None:
                    raise ValueError("internal single-line connection is missing its edge occurrence")
                locator = OccurrenceLocator("internal", connection.edge_index, connection.occurrence)
                actual[locator] = actual.get(locator, 0) + 1

    seen: set[OccurrenceLocator] = set()
    normalized: list[tuple[ParticleMeasureInput, tuple[ParticleOccurrence, ...]]] = []
    for item in card.particle_measures:
        if not isinstance(item, ParticleMeasureInput):
            raise TypeError("particle_measures must contain ParticleMeasureInput values")
        occurrences: list[ParticleOccurrence] = []
        locators = tuple(OccurrenceLocator.from_tuple(value) for value in item.occurrences)
        if len(set(locators)) != len(locators):
            raise ValueError(f"particle p_{item.particle_id} lists an occurrence more than once")
        for locator in locators:
            if locator in seen:
                raise ValueError(f"formula occurrence {locator} is assigned more than once")
            seen.add(locator)
            if actual.get(locator) != 1:
                raise ValueError(f"formula occurrence {locator} resolves to {actual.get(locator, 0)} outgoing connections")
            occurrences.append(_particle_occurrence(item, locator, card.factors))
        expected_internal = sum(locator.kind == "internal" for locator in locators)
        if item.cancellation_count != expected_internal:
            raise ValueError(
                f"p_{item.particle_id} LM-12 cancellation_count {item.cancellation_count} "
                f"does not match {expected_internal} internal occurrences"
            )
        normalized.append((item, tuple(sorted(occurrences, key=lambda occurrence: occurrence.locator.sort_key()))))
    if seen != set(actual):
        missing = sorted(set(actual) - seen, key=OccurrenceLocator.sort_key)
        stale = sorted(seen - set(actual), key=OccurrenceLocator.sort_key)
        raise ValueError(f"particle-measure metadata does not cover formula connections exactly once (missing={missing}, stale={stale})")
    if any(count != 1 for count in actual.values()):
        raise ValueError("a formula occurrence is represented by multiple outgoing connections")
    return tuple(normalized)


def build_cancellation_plan(
    card: FormulaInputCard,
    *,
    overrides: Iterable[CancellationPlanOverride] = (),
) -> CancellationPlan:
    """Build a deterministic plan from LM-12 occurrence metadata.

    By default, the unique external occurrence is retained and all internal
    occurrences are canceled, matching LM-12.  Formula-level overrides name
    all occurrences to cancel for selected particles; exactly one unlisted
    occurrence must remain.  The function only records and validates choices.
    """

    if not isinstance(card, FormulaInputCard):
        raise TypeError("build_cancellation_plan expects a FormulaInputCard")
    validate_formula_card(card)
    if card.identity is None:
        raise ValueError("cancellation plan requires a FormulaInputCard bound to a complete MarkedIdentity")
    if not isinstance(card.identity, Mapping):
        raise TypeError("FormulaInputCard.identity must be a mapping")
    identity_json = json.dumps(card.identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    input_card_digest = hashlib.sha256(card.to_json().encode("utf-8")).hexdigest()

    override_rows = tuple(overrides)
    if any(not isinstance(item, CancellationPlanOverride) for item in override_rows):
        raise TypeError("overrides must contain CancellationPlanOverride values")
    override_by_particle = {item.particle_id: item for item in override_rows}
    if len(override_by_particle) != len(override_rows):
        raise ValueError("at most one formula-level override is allowed per particle")

    particle_rows = _validated_occurrences(card)
    known_particle_ids = {item.particle_id for item, _ in particle_rows}
    unknown_overrides = sorted(set(override_by_particle) - known_particle_ids)
    if unknown_overrides:
        raise ValueError(f"formula-level overrides name unknown particles: {unknown_overrides}")

    plans: list[ParticleCancellationPlan] = []
    any_override = False
    for item, occurrences in particle_rows:
        if not occurrences:
            raise ValueError(f"particle p_{item.particle_id} has no occurrence to retain")
        override = override_by_particle.get(item.particle_id)
        by_locator = {occurrence.locator: occurrence for occurrence in occurrences}
        if override is None:
            external = [occurrence for occurrence in occurrences if occurrence.kind == "external"]
            if len(external) != 1:
                raise ValueError(
                    f"LM-12 default requires exactly one external occurrence for p_{item.particle_id}; "
                    "supply a formula-level cancellation override"
                )
            retained = external[0]
            canceled = tuple(occurrence for occurrence in occurrences if occurrence != retained)
            policy = LM12_DEFAULT_POLICY
            reason = "LM-12 default retains the unique external occurrence and cancels all internal occurrences."
        else:
            any_override = True
            missing = [locator for locator in override.cancellation_occurrences if locator not in by_locator]
            if missing:
                raise ValueError(f"formula-level override for p_{item.particle_id} names unknown occurrences: {missing}")
            if len(override.cancellation_occurrences) != len(occurrences) - 1:
                raise ValueError(
                    f"formula-level override for p_{item.particle_id} must cancel exactly {len(occurrences) - 1} occurrences"
                )
            canceled = tuple(by_locator[locator] for locator in override.cancellation_occurrences)
            retained_candidates = [occurrence for occurrence in occurrences if occurrence.locator not in set(override.cancellation_occurrences)]
            if len(retained_candidates) != 1:
                raise ValueError(f"formula-level override for p_{item.particle_id} leaves {len(retained_candidates)} retained occurrences")
            retained = retained_candidates[0]
            policy = FORMULA_OVERRIDE_POLICY
            reason = override.reason
        plans.append(ParticleCancellationPlan(
            particle_id=item.particle_id,
            occurrences=occurrences,
            retained_occurrence=retained,
            cancellation_occurrences=tuple(sorted(canceled, key=lambda occurrence: occurrence.locator.sort_key())),
            input_cancellation_count=item.cancellation_count,
            selection_policy=policy,
            selection_reason=reason,
        ))

    return CancellationPlan(
        schema_version=CANCELLATION_PLAN_SCHEMA_VERSION,
        input_schema_version=card.schema_version,
        input_card_digest=input_card_digest,
        measure_convention=card.measure_convention,
        formula_identity_json=identity_json,
        selection_policy="lm12_default_with_formula_overrides" if any_override else LM12_DEFAULT_POLICY,
        particles=tuple(plans),
    )


__all__ = [
    "CANCELLATION_PLAN_SCHEMA_VERSION",
    "LM12_DEFAULT_POLICY",
    "FORMULA_OVERRIDE_POLICY",
    "OccurrenceLocator",
    "ParticleOccurrence",
    "CancellationPlanOverride",
    "ParticleCancellationPlan",
    "CancellationPlan",
    "build_cancellation_plan",
]
