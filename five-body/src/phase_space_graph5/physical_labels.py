"""Five-body momentum candidate records.

The four-body ``PhysicalLabelRecord`` couples normalization to an invariant
independence certificate.  This module intentionally defines a separate
record: five-body candidates stop at structure plus momentum conservation and
keep independence/effectiveness explicitly unevaluated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .domain import DoubleLineLabel, EdgeType, Labeling, Topology, ValidationIssue
from .labeling import LabelingResult, MomentumCertificate, _expanded_edges, evaluate_labeling
from .structure import TopologyStructureReport, analyze_topology


SCHEMA_VERSION = "IM-5-candidate-v1"
NORMALIZATION_STAGES = (
    "field_rejected",
    "structure_rejected",
    "momentum_rejected",
    "momentum_candidate",
)

Occurrence = tuple[str, int, int]
SingleGroups = tuple[tuple[int, tuple[Occurrence, ...]], ...]


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
        values: list[Occurrence] = []
        for occurrence in occurrences:
            if not isinstance(occurrence, (tuple, list)) or len(occurrence) != 3:
                raise TypeError("each occurrence must be (kind, first, second)")
            kind, first, second = occurrence
            if kind not in {"external", "internal"}:
                raise ValueError("occurrence kind must be 'external' or 'internal'")
            if any(isinstance(item, bool) or not isinstance(item, int) for item in (first, second)):
                raise TypeError("occurrence coordinates must be integers")
            values.append((kind, first, second))
        normalized.append((particle_id, tuple(values)))
    return tuple(normalized)


def single_occurrence_groups(topology: Topology, labeling: Labeling) -> SingleGroups:
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


def complete_labeling_from_evaluation(topology: Topology, result: LabelingResult) -> Labeling:
    if not isinstance(topology, Topology) or not isinstance(result, LabelingResult):
        raise TypeError("complete_labeling_from_evaluation expects a Topology and LabelingResult")
    if result.momentum is None or not result.momentum.verified:
        raise ValueError("a verified momentum certificate is required")
    double_labels = [
        DoubleLineLabel(
            edge_index,
            occurrence,
            tuple(index + 1 for index, coefficient in enumerate(vector) if coefficient == 1),
        )
        for edge_index, occurrence, vector in result.momentum.double_momenta
    ]
    return Labeling(result.labeling.final_particles, result.labeling.single_lines, double_labels)


@dataclass(frozen=True, slots=True)
class CandidateMomentumRecord:
    """Persisted structure-plus-momentum candidate.

    Construction and :meth:`from_dict` validate field shape and strict scalar
    types.  They intentionally do not recompute the momentum certificate;
    callers that accept a record as evidence must call
    :func:`review_candidate_record`.
    """

    topology: Topology
    complete_labeling: Labeling
    momentum: MomentumCertificate
    single_groups: SingleGroups
    independence_status: str = "not_evaluated"
    independence: None = None
    effectiveness_status: str = "not_evaluated"
    effectiveness: None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported candidate schema: {self.schema_version!r}")
        if not isinstance(self.topology, Topology) or self.topology.n != 5:
            raise ValueError("CandidateMomentumRecord requires a five-body topology")
        if not isinstance(self.complete_labeling, Labeling):
            raise TypeError("complete_labeling must be a Labeling")
        label_report = self.complete_labeling.validate_fields(self.topology)
        if not label_report.valid:
            raise ValueError("complete_labeling has invalid fields")
        if not isinstance(self.momentum, MomentumCertificate) or not self.momentum.verified:
            raise ValueError("candidate record requires a verified momentum certificate")
        if self.independence_status != "not_evaluated" or self.effectiveness_status != "not_evaluated":
            raise ValueError("five-body candidate records only support unevaluated status")
        if self.independence is not None or self.effectiveness is not None:
            raise ValueError("candidate records cannot contain future independence/effectiveness evidence")
        expected_double_refs = tuple(
            (index, occurrence)
            for index, occurrence, _ in _expanded_edges(self.topology, EdgeType.DOUBLE)
        )
        actual_double_refs = tuple(
            (label.edge_index, label.occurrence) for label in self.complete_labeling.double_lines
        )
        if actual_double_refs != expected_double_refs:
            raise ValueError("complete_labeling must contain every double-line occurrence exactly once")
        if tuple((index, occurrence) for index, occurrence, _ in self.momentum.double_momenta) != expected_double_refs:
            raise ValueError("momentum certificate references do not match topology double lines")
        object.__setattr__(self, "single_groups", _normalize_groups(self.single_groups))

    @property
    def variables(self) -> tuple[tuple[int, ...], ...]:
        return self.momentum.variables

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "topology": self.topology.to_dict(),
            "complete_labeling": self.complete_labeling.to_dict(),
            "momentum": self.momentum.to_dict(),
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
            "independence_status": self.independence_status,
            "independence": None,
            "effectiveness_status": self.effectiveness_status,
            "effectiveness": None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateMomentumRecord":
        if not isinstance(data, Mapping):
            raise TypeError("candidate record must be a mapping")
        required = {
            "schema_version", "topology", "complete_labeling", "momentum",
            "single_groups", "independence_status", "independence",
            "effectiveness_status", "effectiveness",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"candidate record is missing {sorted(missing)!r}")
        if data["independence_status"] != "not_evaluated" or data["effectiveness_status"] != "not_evaluated":
            raise ValueError("unknown candidate independence/effectiveness state")
        if data["independence"] is not None or data["effectiveness"] is not None:
            raise ValueError("candidate record contains deferred evidence")
        groups = []
        for group in data.get("single_groups", ()):
            occurrences = []
            for item in group.get("occurrences", ()):
                if item.get("kind") == "external":
                    occurrences.append(("external", item["vertex"], item["slot"]))
                elif item.get("kind") == "internal":
                    occurrences.append(("internal", item["edge_index"], item["occurrence"]))
                else:
                    raise ValueError("unknown single occurrence kind")
            groups.append((group["particle_id"], tuple(occurrences)))
        return cls(
            topology=Topology.from_dict(data["topology"]),
            complete_labeling=Labeling.from_dict(data["complete_labeling"]),
            momentum=MomentumCertificate.from_dict(data["momentum"]),
            single_groups=tuple(groups),
            independence_status=data["independence_status"],
            independence=data["independence"],
            effectiveness_status=data["effectiveness_status"],
            effectiveness=data["effectiveness"],
            schema_version=data["schema_version"],
        )


def candidate_record_from_evaluation(
    topology: Topology, result: LabelingResult
) -> CandidateMomentumRecord:
    if result.momentum is None or not result.momentum.verified or not result.valid:
        raise ValueError("a valid momentum evaluation is required")
    complete = complete_labeling_from_evaluation(topology, result)
    return CandidateMomentumRecord(
        topology=topology,
        complete_labeling=complete,
        momentum=result.momentum,
        single_groups=single_occurrence_groups(topology, complete),
    )


@dataclass(frozen=True, slots=True)
class CandidateNormalizationResult:
    stage: str
    record: CandidateMomentumRecord | None = None
    structure_report: TopologyStructureReport | None = None
    evaluation: LabelingResult | None = None
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in NORMALIZATION_STAGES:
            raise ValueError(f"unsupported candidate stage: {self.stage!r}")
        if self.stage == "momentum_candidate" and self.record is None:
            raise ValueError("momentum_candidate requires a record")
        if self.record is not None and not isinstance(self.record, CandidateMomentumRecord):
            raise TypeError("record must be CandidateMomentumRecord")
        object.__setattr__(self, "issues", tuple(self.issues))

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "record": self.record.to_dict() if self.record else None,
            "structure_report": self.structure_report.to_dict() if self.structure_report else None,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _issues(codes: Iterable[str], messages: Iterable[str]) -> tuple[ValidationIssue, ...]:
    codes = tuple(codes)
    messages = tuple(messages)
    return tuple(ValidationIssue(code, messages[index] if index < len(messages) else code) for index, code in enumerate(codes))


def normalize_candidate_labeling(
    topology: Topology, labeling: Labeling
) -> CandidateNormalizationResult:
    if not isinstance(topology, Topology):
        raise TypeError("normalize_candidate_labeling expects a Topology")
    if not isinstance(labeling, Labeling):
        raise TypeError("normalize_candidate_labeling expects a Labeling")
    fields = topology.validate_fields()
    if not fields.valid:
        return CandidateNormalizationResult("field_rejected", issues=tuple(fields.issues))
    labeling_fields = labeling.validate_fields(topology)
    if not labeling_fields.valid:
        return CandidateNormalizationResult("field_rejected", issues=tuple(labeling_fields.issues))
    structure = analyze_topology(topology)
    if not structure.valid:
        return CandidateNormalizationResult(
            "structure_rejected",
            structure_report=structure,
            issues=tuple(ValidationIssue(check.code, check.message) for check in structure.failures),
        )
    evaluation = evaluate_labeling(topology, labeling)
    if evaluation.valid and evaluation.momentum is not None:
        return CandidateNormalizationResult(
            "momentum_candidate",
            record=candidate_record_from_evaluation(topology, evaluation),
            structure_report=structure,
            evaluation=evaluation,
        )
    return CandidateNormalizationResult(
        "momentum_rejected",
        structure_report=structure,
        evaluation=evaluation,
        issues=_issues(evaluation.failure_codes, evaluation.failure_messages),
    )


def review_candidate_record(record: CandidateMomentumRecord) -> CandidateNormalizationResult:
    """Recompute structure and momentum, then compare every stored field."""
    if not isinstance(record, CandidateMomentumRecord):
        raise TypeError("expected a CandidateMomentumRecord")
    checked = normalize_candidate_labeling(record.topology, record.complete_labeling)
    if checked.record is None or checked.record.to_dict() != record.to_dict():
        raise ValueError("candidate record is incomplete or its momentum evidence is inconsistent")
    return checked


__all__ = [
    "SCHEMA_VERSION", "NORMALIZATION_STAGES", "Occurrence", "SingleGroups",
    "CandidateMomentumRecord", "CandidateNormalizationResult",
    "candidate_record_from_evaluation", "complete_labeling_from_evaluation",
    "normalize_candidate_labeling", "review_candidate_record",
    "single_occurrence_groups",
]
