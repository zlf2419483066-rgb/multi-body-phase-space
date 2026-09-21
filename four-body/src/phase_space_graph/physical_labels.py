"""Immutable physical records for the independent-labeling workflow.

This first layer stores a completed labeling and the evidence already
produced by existing G02/G06 modules. It deliberately does not recover
momenta, classify a record, or start an enumeration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .domain import CountMode, DoubleLineLabel, Labeling, Topology, ValidationIssue
from .invariants import IndependenceResult
from .labeling import LabelingResult, MomentumCertificate, evaluate_labeling
from .structure import TopologyStructureReport, analyze_topology


SCHEMA_VERSION = "IM-4-v1"
NORMALIZATION_STAGES = (
    "field_rejected",
    "structure_rejected",
    "momentum_rejected",
    "dependent",
    "accepted",
)

# The first item identifies the occurrence kind. The two integers then mean
# (vertex, slot) or (edge_index, occurrence), respectively.
Occurrence = tuple[str, int, int]
SingleGroups = tuple[tuple[int, tuple[Occurrence, ...]], ...]
ParticleCounts = tuple[tuple[int, int], ...]


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


def _normalize_groups(groups: Iterable[tuple[int, Iterable[Occurrence]]]) -> SingleGroups:
    normalized: list[tuple[int, tuple[Occurrence, ...]]] = []
    for particle_id, occurrences in groups:
        if isinstance(particle_id, bool) or not isinstance(particle_id, int):
            raise TypeError("single-group particle IDs must be integers")
        occurrence_values: list[Occurrence] = []
        for occurrence in occurrences:
            if not isinstance(occurrence, (tuple, list)) or len(occurrence) != 3:
                raise TypeError("each occurrence must be (kind, first, second)")
            kind, first, second = occurrence
            if kind not in {"external", "internal"}:
                raise ValueError("occurrence kind must be 'external' or 'internal'")
            if any(isinstance(item, bool) or not isinstance(item, int) for item in (first, second)):
                raise TypeError("occurrence coordinates must be integers")
            occurrence_values.append((kind, first, second))
        normalized.append((particle_id, tuple(occurrence_values)))
    return tuple(normalized)


def _normalize_counts(counts: Iterable[tuple[int, int]]) -> ParticleCounts:
    normalized: list[tuple[int, int]] = []
    for particle_id, count in counts:
        if (
            isinstance(particle_id, bool)
            or not isinstance(particle_id, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise TypeError("measure cancellation entries must be (integer particle, non-negative integer count)")
        normalized.append((particle_id, count))
    return tuple(normalized)


def single_occurrence_groups(topology: Topology, labeling: Labeling) -> SingleGroups:
    """Group every external and internal single-line occurrence by particle."""

    if not isinstance(topology, Topology) or not isinstance(labeling, Labeling):
        raise TypeError("single_occurrence_groups expects a Topology and Labeling")
    grouped: dict[int, list[Occurrence]] = {particle: [] for particle in range(1, topology.n + 1)}
    for item in labeling.final_particles:
        if item.particle_id in grouped:
            grouped[item.particle_id].append(("external", item.vertex, item.slot))
    for item in labeling.single_lines:
        if item.particle_id in grouped:
            grouped[item.particle_id].append(("internal", item.edge_index, item.occurrence))
    return tuple((particle, tuple(grouped[particle])) for particle in range(1, topology.n + 1))


def measure_cancellation_counts(topology: Topology, labeling: Labeling) -> ParticleCounts:
    """Return the number of internal repeated appearances for each particle."""

    if not isinstance(topology, Topology) or not isinstance(labeling, Labeling):
        raise TypeError("measure_cancellation_counts expects a Topology and Labeling")
    counts = Counter(item.particle_id for item in labeling.single_lines)
    return tuple((particle, counts.get(particle, 0)) for particle in range(1, topology.n + 1))


def complete_labeling_from_evaluation(topology: Topology, result: LabelingResult) -> Labeling:
    """Fill every double-line label from one successful G06 momentum result."""

    if not isinstance(topology, Topology) or not isinstance(result, LabelingResult):
        raise TypeError("complete_labeling_from_evaluation expects a Topology and LabelingResult")
    if result.momentum is None or not result.momentum.verified:
        raise ValueError("a verified momentum certificate is required to complete a labeling")
    double_labels = [
        DoubleLineLabel(
            edge_index,
            occurrence,
            tuple(index + 1 for index, coefficient in enumerate(vector) if coefficient == 1),
        )
        for edge_index, occurrence, vector in result.momentum.double_momenta
    ]
    return Labeling(result.labeling.final_particles, result.labeling.single_lines, double_labels)


def physical_record_from_evaluation(topology: Topology, result: LabelingResult) -> PhysicalLabelRecord:
    """Build a record from verified momentum and independence evidence."""

    if result.momentum is None or not result.momentum.verified or result.independence is None:
        raise ValueError("a verified momentum and independence certificate are required")
    complete = complete_labeling_from_evaluation(topology, result)
    return PhysicalLabelRecord(
        topology=topology,
        complete_labeling=complete,
        momentum=result.momentum,
        independence=result.independence,
        single_groups=single_occurrence_groups(topology, complete),
        measure_cancellations=measure_cancellation_counts(topology, complete),
    )


def review_physical_record(record: PhysicalLabelRecord) -> NormalizationResult:
    """Recheck a complete record; status flags are not physical evidence."""
    if not isinstance(record, PhysicalLabelRecord):
        raise TypeError("expected a complete PhysicalLabelRecord")
    checked = normalize_physical_labeling(record.topology, record.complete_labeling)
    if checked.record is None or checked.record != record:
        raise ValueError("physical record is incomplete or its momentum/rank/measure evidence is inconsistent")
    return checked


def _issues_from_codes(codes: tuple[str, ...], messages: tuple[str, ...]) -> tuple[ValidationIssue, ...]:
    return tuple(
        ValidationIssue(code, messages[index] if index < len(messages) else code)
        for index, code in enumerate(codes)
    )


def normalize_physical_labeling(
    topology: Topology,
    labeling: Labeling,
    count_mode: CountMode | str = CountMode.FIXED_LABELS,
) -> NormalizationResult:
    """Validate one input and return a complete physical record when possible.

    This is the first public single-record entry point. It intentionally does
    not search over particle assignments or read audit files.
    """

    if not isinstance(topology, Topology):
        raise TypeError("normalize_physical_labeling expects a Topology object")
    if not isinstance(labeling, Labeling):
        raise TypeError("normalize_physical_labeling expects a Labeling object")
    topology_fields = topology.validate_fields()
    if not topology_fields.valid:
        return NormalizationResult("field_rejected", issues=topology_fields.issues)
    labeling_fields = labeling.validate_fields(topology)
    if not labeling_fields.valid:
        return NormalizationResult("field_rejected", issues=labeling_fields.issues)

    structure_report = analyze_topology(topology)
    if not structure_report.valid:
        structural_issues = tuple(
            ValidationIssue(check.code, check.message)
            for check in structure_report.failures
        )
        return NormalizationResult(
            "structure_rejected",
            structure_report=structure_report,
            issues=structural_issues,
        )

    evaluation = evaluate_labeling(topology, labeling, count_mode)
    if evaluation.momentum is not None and evaluation.independence is not None:
        if "dependent_invariants" in evaluation.failure_codes:
            record = physical_record_from_evaluation(topology, evaluation)
            return NormalizationResult(
                "dependent",
                record=record,
                structure_report=structure_report,
                legacy_evaluation=evaluation,
                issues=_issues_from_codes(evaluation.failure_codes, evaluation.failure_messages),
            )
        if evaluation.valid:
            record = physical_record_from_evaluation(topology, evaluation)
            return NormalizationResult(
                "accepted",
                record=record,
                structure_report=structure_report,
                legacy_evaluation=evaluation,
            )

    return NormalizationResult(
        "momentum_rejected",
        structure_report=structure_report,
        legacy_evaluation=evaluation,
        issues=_issues_from_codes(evaluation.failure_codes, evaluation.failure_messages),
    )


@dataclass(frozen=True, slots=True)
class PhysicalLabelRecord:
    """A complete physical labeling together with audit certificates.

    Source page, drawing coordinates, old keys, and transcription details do
    not belong here because they must not affect physical identity.
    """

    topology: Topology
    complete_labeling: Labeling
    momentum: MomentumCertificate
    independence: IndependenceResult
    single_groups: SingleGroups
    measure_cancellations: ParticleCounts
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported physical-label schema: {self.schema_version!r}")
        if not isinstance(self.topology, Topology):
            raise TypeError("PhysicalLabelRecord.topology must be a Topology")
        if not isinstance(self.complete_labeling, Labeling):
            raise TypeError("PhysicalLabelRecord.complete_labeling must be a Labeling")
        if not isinstance(self.momentum, MomentumCertificate):
            raise TypeError("PhysicalLabelRecord.momentum must be a MomentumCertificate")
        if not isinstance(self.independence, IndependenceResult):
            raise TypeError("PhysicalLabelRecord.independence must be an IndependenceResult")
        if self.independence.n != self.topology.n:
            raise ValueError("momentum and independence evidence must use the topology particle count")
        object.__setattr__(self, "single_groups", _normalize_groups(self.single_groups))
        object.__setattr__(self, "measure_cancellations", _normalize_counts(self.measure_cancellations))

    def to_dict(self) -> dict[str, object]:
        """Serialize physical content without provenance or presentation fields."""

        return {
            "schema_version": self.schema_version,
            "topology": self.topology.to_dict(),
            "complete_labeling": self.complete_labeling.to_dict(),
            "momentum": self.momentum.to_dict(),
            "independence": self.independence.to_dict(),
            "single_groups": [
                {
                    "particle_id": particle,
                    "occurrences": [
                        (
                            {"kind": "external", "vertex": first, "slot": second}
                            if kind == "external"
                            else {"kind": "internal", "edge_index": first, "occurrence": second}
                        )
                        for kind, first, second in occurrences
                    ],
                }
                for particle, occurrences in self.single_groups
            ],
            "measure_cancellations": [
                {"particle_id": particle, "count": count}
                for particle, count in self.measure_cancellations
            ],
        }


@dataclass(frozen=True, slots=True)
class Provenance:
    """Source and legacy information kept outside the physical identity."""

    source: tuple[tuple[str, Any], ...] = ()
    raw_labeling: Labeling | None = None
    legacy_key: Any = None

    def __init__(
        self,
        source: Mapping[str, Any] | Iterable[tuple[str, Any]] = (),
        raw_labeling: Labeling | None = None,
        legacy_key: Any = None,
    ) -> None:
        items = tuple(source.items()) if isinstance(source, Mapping) else tuple(source)
        if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in items):
            raise TypeError("Provenance.source must contain key/value pairs")
        if any(not isinstance(key, str) or not key for key, _ in items):
            raise TypeError("Provenance source keys must be non-empty strings")
        if raw_labeling is not None and not isinstance(raw_labeling, Labeling):
            raise TypeError("Provenance.raw_labeling must be a Labeling or None")
        object.__setattr__(self, "source", tuple((key, value) for key, value in items))
        object.__setattr__(self, "raw_labeling", raw_labeling)
        object.__setattr__(self, "legacy_key", legacy_key)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": {key: _jsonable(value) for key, value in self.source},
            "raw_labeling": self.raw_labeling.to_dict() if self.raw_labeling else None,
            "legacy_key": _jsonable(self.legacy_key),
        }


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """One physical review result, with a complete record when available."""

    stage: str
    record: PhysicalLabelRecord | None = None
    structure_report: TopologyStructureReport | None = None
    legacy_evaluation: LabelingResult | None = None
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in NORMALIZATION_STAGES:
            raise ValueError(f"unsupported normalization stage: {self.stage!r}")
        if self.stage in {"accepted", "dependent"} and self.record is None:
            raise ValueError(f"{self.stage} results require a PhysicalLabelRecord")
        if self.record is not None and not isinstance(self.record, PhysicalLabelRecord):
            raise TypeError("NormalizationResult.record must be a PhysicalLabelRecord or None")
        if self.structure_report is not None and not isinstance(self.structure_report, TopologyStructureReport):
            raise TypeError("structure_report must be a TopologyStructureReport or None")
        if self.legacy_evaluation is not None and not isinstance(self.legacy_evaluation, LabelingResult):
            raise TypeError("legacy_evaluation must be a LabelingResult or None")
        if not all(isinstance(issue, ValidationIssue) for issue in self.issues):
            raise TypeError("issues must contain ValidationIssue objects")
        object.__setattr__(self, "issues", tuple(self.issues))

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "record": self.record.to_dict() if self.record else None,
            "structure_report": self.structure_report.to_dict() if self.structure_report else None,
            "legacy_evaluation": self.legacy_evaluation.to_dict() if self.legacy_evaluation else None,
            "issues": [issue.to_dict() for issue in self.issues],
        }


__all__ = [
    "SCHEMA_VERSION",
    "NORMALIZATION_STAGES",
    "Occurrence",
    "SingleGroups",
    "ParticleCounts",
    "PhysicalLabelRecord",
    "Provenance",
    "NormalizationResult",
    "complete_labeling_from_evaluation",
    "physical_record_from_evaluation",
    "normalize_physical_labeling",
    "single_occurrence_groups",
    "measure_cancellation_counts",
]
