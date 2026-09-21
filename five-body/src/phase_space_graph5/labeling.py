"""Five-body candidate labeling and tree-cut momentum recovery.

This module deliberately stops after structural and momentum checks.  It does
not import the four-body invariant module and never turns a repeated variable
into a rejection.  A momentum-passing assignment is a candidate even when a
future independence analysis may later classify it differently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
import json
import math
import time
from typing import Any, Callable, Iterable, Mapping

from .domain import (
    CountMode,
    DoubleLineLabel,
    EdgeType,
    FinalParticleLabel,
    Labeling,
    RunStatus,
    SingleLineLabel,
    Topology,
    ValidationIssue,
)
from .structure import analyze_topology


Vector = tuple[int, ...]


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


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _zero(n: int) -> Vector:
    return (0,) * n


def _unit(n: int, particle: int) -> Vector:
    return tuple(1 if index + 1 == particle else 0 for index in range(n))


def _add(left: Vector, right: Vector) -> Vector:
    return tuple(a + b for a, b in zip(left, right))


def _sub(left: Vector, right: Vector) -> Vector:
    return tuple(a - b for a, b in zip(left, right))


def _expanded_edges(topology: Topology, edge_type: EdgeType) -> tuple[tuple[int, int, Any], ...]:
    return tuple(
        (edge_index, occurrence, edge)
        for edge_index, edge in enumerate(topology.edges)
        if edge.edge_type is edge_type and edge.multiplicity > 0
        for occurrence in range(edge.multiplicity)
    )


@dataclass(frozen=True, slots=True)
class LabelingLimits:
    """Cooperative limits; ``deadline`` is an absolute monotonic timestamp."""

    max_labelings: int | None = None
    max_results: int | None = None
    max_seconds: float | None = None
    deadline: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_labelings", "max_results"):
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        for name in ("max_seconds", "deadline"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be a finite number or None")
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be finite")
                if name == "max_seconds" and value < 0:
                    raise ValueError("max_seconds must be non-negative")

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "max_labelings": self.max_labelings,
            "max_results": self.max_results,
            "max_seconds": self.max_seconds,
            "deadline": self.deadline,
        }


@dataclass(frozen=True, slots=True)
class FailureRecord:
    code: str
    message: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "details": _jsonable(self.details)}


@dataclass(frozen=True, slots=True)
class MomentumCertificate:
    """Recovered double-line vectors and exact vertex residuals."""

    double_momenta: tuple[tuple[int, int, Vector], ...]
    vertex_residuals: tuple[tuple[int, Vector], ...]
    verified: bool

    def __post_init__(self) -> None:
        # ``bool`` is intentionally exact here.  In particular, accepting 1
        # or "yes" would let a damaged persisted certificate become a
        # seemingly verified candidate on read-back.
        if type(self.verified) is not bool:
            raise TypeError("MomentumCertificate.verified must be a boolean")

    @property
    def variables(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(index + 1 for index, coefficient in enumerate(vector) if coefficient == 1)
            for _, _, vector in self.double_momenta
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "double_momenta": [
                {"edge_index": edge_index, "occurrence": occurrence, "coefficients": list(vector)}
                for edge_index, occurrence, vector in self.double_momenta
            ],
            "variables": [list(variable) for variable in self.variables],
            "vertex_residuals": [
                {"vertex": vertex, "residual": list(residual)}
                for vertex, residual in self.vertex_residuals
            ],
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MomentumCertificate":
        if not isinstance(data, Mapping):
            raise TypeError("momentum certificate must be a mapping")
        required = {"double_momenta", "variables", "vertex_residuals", "verified"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"momentum certificate is missing {sorted(missing)!r}")
        verified = data["verified"]
        if type(verified) is not bool:
            raise TypeError("momentum.verified must be a boolean")

        def strict_int(value: Any, name: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            return value

        def strict_list(value: Any, name: str) -> list[Any]:
            if not isinstance(value, list):
                raise TypeError(f"{name} must be a list")
            return value

        double_rows = strict_list(data["double_momenta"], "double_momenta")
        residual_rows = strict_list(data["vertex_residuals"], "vertex_residuals")
        variable_rows = strict_list(data["variables"], "variables")

        parsed_double: list[tuple[int, int, Vector]] = []
        for index, row in enumerate(double_rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"double_momenta[{index}] must be a mapping")
            required_row = {"edge_index", "occurrence", "coefficients"}
            missing_row = required_row - set(row)
            if missing_row:
                raise ValueError(
                    f"double_momenta[{index}] is missing {sorted(missing_row)!r}"
                )
            coefficients = strict_list(row["coefficients"], f"double_momenta[{index}].coefficients")
            parsed_double.append(
                (
                    strict_int(row["edge_index"], "edge_index"),
                    strict_int(row["occurrence"], "occurrence"),
                    tuple(strict_int(value, "coefficient") for value in coefficients),
                )
            )

        parsed_residuals: list[tuple[int, Vector]] = []
        for index, row in enumerate(residual_rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"vertex_residuals[{index}] must be a mapping")
            required_row = {"vertex", "residual"}
            missing_row = required_row - set(row)
            if missing_row:
                raise ValueError(
                    f"vertex_residuals[{index}] is missing {sorted(missing_row)!r}"
                )
            residual = strict_list(row["residual"], f"vertex_residuals[{index}].residual")
            parsed_residuals.append(
                (
                    strict_int(row["vertex"], "vertex"),
                    tuple(strict_int(value, "residual component") for value in residual),
                )
            )

        declared: list[tuple[int, ...]] = []
        for index, variable in enumerate(variable_rows):
            if not isinstance(variable, list):
                raise TypeError(f"variables[{index}] must be a list")
            declared.append(
                tuple(strict_int(value, "declared variable particle") for value in variable)
            )

        result = cls(
            tuple(parsed_double),
            tuple(parsed_residuals),
            verified,
        )
        if tuple(declared) != result.variables:
            raise ValueError("momentum.variables disagrees with coefficient vectors")
        return result


@dataclass(frozen=True, slots=True)
class LabelingResult:
    """One candidate labeling evaluation, with no independence field semantics."""

    labeling: Labeling
    valid: bool
    failure_codes: tuple[str, ...]
    failure_messages: tuple[str, ...]
    momentum: MomentumCertificate | None
    independence: None = None
    canonical: None = None
    single_occurrence_counts: tuple[tuple[int, int], ...] = ()

    @property
    def variables(self) -> tuple[tuple[int, ...], ...]:
        return self.momentum.variables if self.momentum else ()

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "failure_codes": list(self.failure_codes),
            "failure_messages": list(self.failure_messages),
            "labeling": self.labeling.to_dict(),
            "momentum": self.momentum.to_dict() if self.momentum else None,
            "independence": None,
            "canonical": None,
            "single_occurrence_counts": [
                {"particle": particle, "count": count}
                for particle, count in self.single_occurrence_counts
            ],
        }


@dataclass(frozen=True, slots=True)
class CandidateSearchReport:
    topology: Topology
    status: RunStatus
    count_mode: CountMode
    raw_assignment_count: int
    momentum_pass_count: int
    candidate_records: tuple[Any, ...]
    failure_counts: tuple[tuple[str, int], ...]
    failure_samples: tuple[FailureRecord, ...]
    failure_witnesses: tuple[tuple[str, LabelingResult], ...]
    limits: LabelingLimits
    stop_reason: str | None
    # Number of actual evaluate_labeling calls.  In the reference path this
    # equals raw assignments; with caching it counts representative misses.
    # ``raw_assignment_count`` remains the number of assignments traversed.
    representative_evaluations: int = 0

    @property
    def complete(self) -> bool:
        return self.status is RunStatus.COMPLETE

    def failure_count(self, code: str) -> int:
        return dict(self.failure_counts).get(code, 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology.to_dict(),
            "status": self.status.value,
            "complete": self.complete,
            "count_mode": self.count_mode.value,
            "counts": {
                "raw_assignment_count": self.raw_assignment_count,
                "momentum_pass_count": self.momentum_pass_count,
                "candidate_records": len(self.candidate_records),
            },
            "representative_evaluations": self.representative_evaluations,
            "candidate_records": [record.to_dict() for record in self.candidate_records],
            "failure_counts": dict(self.failure_counts),
            "failure_samples": [sample.to_dict() for sample in self.failure_samples],
            "failure_witnesses": {
                code: result.to_dict() for code, result in self.failure_witnesses
            },
            "limits": self.limits.to_dict(),
            "stop_reason": self.stop_reason,
        }


def _slot_order(topology: Topology) -> tuple[tuple[int, int], ...]:
    slots: list[tuple[int, int]] = []
    for vertex, count in topology.final_slots:
        if _is_int(vertex) and _is_int(count) and count >= 0:
            slots.extend((vertex, slot) for slot in range(count))
    return tuple(slots)


def _labeling_from_assignments(
    topology: Topology,
    slots: tuple[tuple[int, int], ...],
    final_particles: tuple[int, ...],
    single_refs: tuple[tuple[int, int, Any], ...],
    single_particles: tuple[int, ...],
) -> Labeling:
    return Labeling(
        final_particles=tuple(
            FinalParticleLabel(vertex, slot, particle)
            for (vertex, slot), particle in zip(slots, final_particles)
        ),
        single_lines=tuple(
            SingleLineLabel(edge_index, occurrence, particle)
            for (edge_index, occurrence, _), particle in zip(single_refs, single_particles)
        ),
    )


def _side_vertices(
    vertices: tuple[int, ...],
    double_refs: tuple[tuple[int, int, Any], ...],
    removed: tuple[int, int],
    target: int,
) -> set[int]:
    adjacency = {vertex: set() for vertex in vertices}
    for edge_index, occurrence, edge in double_refs:
        if (edge_index, occurrence) == removed:
            continue
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    pending = [target]
    reached = {target}
    while pending:
        vertex = pending.pop()
        for child in adjacency[vertex]:
            if child not in reached:
                reached.add(child)
                pending.append(child)
    return reached


def _recover_momenta(
    topology: Topology,
    labeling: Labeling,
    prepared_sides: Mapping[tuple[int, int], set[int]] | None = None,
) -> tuple[MomentumCertificate | None, tuple[str, ...], tuple[str, ...]]:
    n = topology.n
    vertices = tuple(topology.vertices)
    double_refs = _expanded_edges(topology, EdgeType.DOUBLE)
    if len(double_refs) != len(vertices) - 1:
        return None, ("double_tree_required",), ("Double-line occurrences must form a tree.",)
    single_by_ref = {
        (label.edge_index, label.occurrence): label.particle_id
        for label in labeling.single_lines
    }
    final_by_vertex: dict[int, Vector] = {vertex: _zero(n) for vertex in vertices}
    for label in labeling.final_particles:
        final_by_vertex[label.vertex] = _add(
            final_by_vertex[label.vertex], _unit(n, label.particle_id)
        )
    single_in: dict[int, Vector] = {vertex: _zero(n) for vertex in vertices}
    single_out: dict[int, Vector] = {vertex: _zero(n) for vertex in vertices}
    for edge_index, occurrence, edge in _expanded_edges(topology, EdgeType.SINGLE):
        particle = single_by_ref.get((edge_index, occurrence))
        if particle is None:
            return None, ("missing_single_label",), (
                "Every internal single-line occurrence needs a label.",
            )
        vector = _unit(n, particle)
        single_out[edge.source] = _add(single_out[edge.source], vector)
        single_in[edge.target] = _add(single_in[edge.target], vector)
    initial = tuple(1 for _ in range(n))
    base = {
        vertex: _sub(
            _add(final_by_vertex[vertex], single_out[vertex]),
            _add(single_in[vertex], initial if vertex == topology.initial_vertex else _zero(n)),
        )
        for vertex in vertices
    }
    double_momenta: list[tuple[int, int, Vector]] = []
    for edge_index, occurrence, edge in double_refs:
        side = (
            prepared_sides[(edge_index, occurrence)]
            if prepared_sides is not None
            else _side_vertices(vertices, double_refs, (edge_index, occurrence), edge.target)
        )
        vector = _zero(n)
        for vertex in side:
            vector = _add(vector, base[vertex])
        double_momenta.append((edge_index, occurrence, vector))

    residuals: dict[int, Vector] = {vertex: _zero(n) for vertex in vertices}
    for vertex in vertices:
        residuals[vertex] = _sub(
            _add(single_in[vertex], initial if vertex == topology.initial_vertex else _zero(n)),
            _add(final_by_vertex[vertex], single_out[vertex]),
        )
    for edge_index, occurrence, edge in double_refs:
        vector = next(
            vector
            for index, item, vector in double_momenta
            if (index, item) == (edge_index, occurrence)
        )
        residuals[edge.source] = _sub(residuals[edge.source], vector)
        residuals[edge.target] = _add(residuals[edge.target], vector)
    residual_tuple = tuple((vertex, residuals[vertex]) for vertex in vertices)
    failures: list[str] = []
    messages: list[str] = []
    for edge_index, occurrence, vector in double_momenta:
        if any(coefficient < 0 for coefficient in vector):
            failures.append("negative_composite_coefficient")
            messages.append(f"Double edge {(edge_index, occurrence)} has a negative coefficient.")
        if any(coefficient > 1 for coefficient in vector):
            failures.append("composite_coefficient_gt_one")
            messages.append(f"Double edge {(edge_index, occurrence)} has a coefficient greater than one.")
        support_size = sum(coefficient == 1 for coefficient in vector)
        if support_size == 0:
            failures.append("empty_composite")
            messages.append(f"Double edge {(edge_index, occurrence)} carries an empty subset.")
        elif support_size == 1:
            failures.append("singleton_composite")
            messages.append(f"Double edge {(edge_index, occurrence)} carries a singleton subset.")
        elif support_size == n:
            failures.append("fixed_total_composite")
            messages.append(f"Double edge {(edge_index, occurrence)} carries the fixed total momentum.")
        elif not 2 <= support_size <= n - 1:
            failures.append("composite_support_size")
            messages.append(f"Double edge {(edge_index, occurrence)} has an unsupported subset size.")
    if any(any(component != 0 for component in residual) for _, residual in residual_tuple):
        failures.append("conservation_mismatch")
        messages.append("At least one vertex conservation residual is non-zero.")
    certificate = MomentumCertificate(tuple(double_momenta), residual_tuple, not failures)
    return certificate, tuple(dict.fromkeys(failures)), tuple(dict.fromkeys(messages))


def evaluate_labeling(
    topology: Topology,
    labeling: Labeling,
    *,
    _prepared_sides: Mapping[tuple[int, int], set[int]] | None = None,
) -> LabelingResult:
    """Evaluate one explicitly supplied labeling through momentum only."""
    if not isinstance(topology, Topology):
        raise TypeError("evaluate_labeling expects a Topology object")
    if not isinstance(labeling, Labeling):
        raise TypeError("evaluate_labeling expects a Labeling object")
    topology_fields = topology.validate_fields()
    if not topology_fields.valid:
        raise ValueError(f"Topology fields must be valid: {topology_fields.to_dict()}")
    structure = analyze_topology(topology)
    if not structure.valid:
        codes = structure.all_failure_codes()
        messages = tuple(
            check.message for check in structure.failures
        ) + tuple(issue.message for issue in structure.field_report.issues)
        return LabelingResult(
            labeling, False, codes, messages, None, None, None, ()
        )
    single_counts = Counter(label.particle_id for label in labeling.single_lines)
    fields = labeling.validate_fields(topology)
    if not fields.valid:
        return LabelingResult(
            labeling, False, fields.error_codes(),
            tuple(issue.message for issue in fields.issues if issue.severity == "error"),
            None, None, None, tuple(sorted(single_counts.items())),
        )
    momentum, failures, messages = _recover_momenta(
        topology, labeling, prepared_sides=_prepared_sides
    )
    if momentum is not None and not failures and labeling.double_lines:
        recovered = {
            (index, occurrence): tuple(index + 1 for index, coefficient in enumerate(vector) if coefficient == 1)
            for index, occurrence, vector in momentum.double_momenta
        }
        mismatched = [
            label for label in labeling.double_lines
            if tuple(sorted(label.particle_ids)) != recovered.get((label.edge_index, label.occurrence))
        ]
        if mismatched:
            failures = ("declared_double_momentum_mismatch",)
            messages = ("Explicit double-line subsets disagree with vertex momentum conservation.",)
    return LabelingResult(
        labeling, not failures, failures, messages, momentum, None, None,
        tuple(sorted(single_counts.items())),
    )


def _candidate_record(topology: Topology, result: LabelingResult) -> Any:
    from .physical_labels import candidate_record_from_evaluation
    return candidate_record_from_evaluation(topology, result)


def _rename_vector(vector: Vector, final_particles: tuple[int, ...]) -> Vector:
    """Transport representative particle coordinates under a final permutation."""
    inverse = {new: old for old, new in enumerate(final_particles, 1)}
    return tuple(vector[inverse[new] - 1] for new in range(1, len(vector) + 1))


def _transform_result(
    representative: LabelingResult,
    actual_labeling: Labeling,
    final_particles: tuple[int, ...],
) -> LabelingResult:
    """Apply the particle permutation to a cached representative result."""
    momentum = representative.momentum
    transformed_momentum = None
    if momentum is not None:
        transformed_momentum = type(momentum)(
            tuple(
                (edge_index, occurrence, _rename_vector(vector, final_particles))
                for edge_index, occurrence, vector in momentum.double_momenta
            ),
            tuple(
                (vertex, _rename_vector(vector, final_particles))
                for vertex, vector in momentum.vertex_residuals
            ),
            momentum.verified,
        )
    counts = tuple(
        sorted(
            (final_particles[particle - 1], count)
            for particle, count in representative.single_occurrence_counts
        )
    )
    return LabelingResult(
        labeling=actual_labeling,
        valid=representative.valid,
        failure_codes=representative.failure_codes,
        failure_messages=representative.failure_messages,
        momentum=transformed_momentum,
        independence=None,
        canonical=None,
        single_occurrence_counts=counts,
    )


def enumerate_labelings(
    topology: Topology,
    count_mode: CountMode | str = CountMode.FIXED_LABELS,
    limits: LabelingLimits | None = None,
    *,
    cancel_requested: Callable[[], bool] | None = None,
    deadline: float | None = None,
    use_symmetry: bool = True,
) -> CandidateSearchReport:
    """Enumerate all raw assignments, optionally reusing particle orbits.

    ``use_symmetry=False`` is the direct reference path.  The optimized path
    still traverses every raw assignment in the same order and only reuses the
    momentum classification of a fixed-final representative.
    """
    if not isinstance(topology, Topology):
        raise TypeError("enumerate_labelings expects a Topology object")
    mode = count_mode if isinstance(count_mode, CountMode) else CountMode(count_mode)
    limits = limits or LabelingLimits()
    if not isinstance(limits, LabelingLimits):
        raise TypeError("limits must be a LabelingLimits object or None")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None")
    if type(use_symmetry) is not bool:
        raise TypeError("use_symmetry must be a boolean")
    structure = analyze_topology(topology)
    if not structure.valid:
        return CandidateSearchReport(
            topology=topology,
            status=RunStatus.FAILED,
            count_mode=mode,
            raw_assignment_count=0,
            momentum_pass_count=0,
            candidate_records=(),
            failure_counts=tuple((code, 1) for code in structure.all_failure_codes()),
            failure_samples=(),
            failure_witnesses=(),
            limits=limits,
            stop_reason=None,
            representative_evaluations=0,
        )
    failures: Counter[str] = Counter()
    samples: list[FailureRecord] = []
    witnesses: dict[str, LabelingResult] = {}
    records: list[Any] = []
    raw_count = 0
    momentum_pass = 0
    representative_evaluations = 0
    started = time.monotonic()
    effective_deadline = deadline if deadline is not None else limits.deadline
    if limits.max_seconds is not None:
        local_deadline = started + limits.max_seconds
        effective_deadline = local_deadline if effective_deadline is None else min(effective_deadline, local_deadline)
    stop_reason: str | None = None

    def checkpoint() -> None:
        if cancel_requested is not None and cancel_requested():
            raise _Stop("cancel_requested")
        if effective_deadline is not None and time.monotonic() >= effective_deadline:
            raise _Stop("deadline_exceeded")
        if limits.max_labelings is not None and raw_count >= limits.max_labelings:
            raise _Stop("limit_max_labelings")
        if limits.max_results is not None and len(records) >= limits.max_results:
            raise _Stop("limit_max_results")

    class _Stop(Exception):
        def __init__(self, reason: str):
            self.reason = reason

    try:
        if limits.max_labelings == 0 or limits.max_results == 0:
            raise _Stop("limit_zero_budget")
        slots = _slot_order(topology)
        single_refs = _expanded_edges(topology, EdgeType.SINGLE)
        particle_values = tuple(range(1, topology.n + 1))
        prepared_sides: dict[tuple[int, int], set[int]] = {}
        representative_cache: dict[tuple[int, ...], LabelingResult] = {}
        if use_symmetry:
            # Double-line cuts depend only on topology.  Compute them once per
            # call; they are then shared by every representative evaluation.
            double_refs = _expanded_edges(topology, EdgeType.DOUBLE)
            vertices = tuple(topology.vertices)
            for edge_index, occurrence, edge in double_refs:
                prepared_sides[(edge_index, occurrence)] = _side_vertices(
                    vertices, double_refs, (edge_index, occurrence), edge.target
                )
        for final_particles in itertools.permutations(particle_values):
            inverse = {
                particle: index + 1
                for index, particle in enumerate(final_particles)
            }
            for single_particles in itertools.product(particle_values, repeat=len(single_refs)):
                checkpoint()
                raw_count += 1
                labeling = _labeling_from_assignments(
                    topology, slots, final_particles, single_refs, single_particles
                )
                if not use_symmetry:
                    representative_evaluations += 1
                    result = evaluate_labeling(topology, labeling)
                else:
                    assignment = tuple(inverse[particle] for particle in single_particles)
                    representative = representative_cache.get(assignment)
                    if representative is None:
                        base_labeling = _labeling_from_assignments(
                            topology,
                            slots,
                            particle_values,
                            single_refs,
                            assignment,
                        )
                        representative = evaluate_labeling(
                            topology,
                            base_labeling,
                            _prepared_sides=prepared_sides,
                        )
                        representative_cache[assignment] = representative
                        representative_evaluations += 1
                    result = _transform_result(
                        representative, labeling, final_particles
                    )
                if result.valid and result.momentum is not None and result.momentum.verified:
                    momentum_pass += 1
                    records.append(_candidate_record(topology, result))
                    continue
                for code, message in zip(result.failure_codes, result.failure_messages):
                    failures[code] += 1
                    witnesses.setdefault(code, result)
                    if len(samples) < 8:
                        samples.append(FailureRecord(code, message, {"labeling": labeling.to_dict()}))
    except _Stop as exc:
        stop_reason = exc.reason
        failures[exc.reason] += 1
    status = RunStatus.INCOMPLETE if stop_reason else RunStatus.COMPLETE
    return CandidateSearchReport(
        topology=topology,
        status=status,
        count_mode=mode,
        raw_assignment_count=raw_count,
        momentum_pass_count=momentum_pass,
        candidate_records=tuple(records),
        failure_counts=tuple(sorted(failures.items())),
        failure_samples=tuple(samples),
        failure_witnesses=tuple(sorted(witnesses.items())),
        limits=limits,
        stop_reason=stop_reason,
        representative_evaluations=representative_evaluations,
    )


__all__ = [
    "Vector", "LabelingLimits", "FailureRecord", "MomentumCertificate",
    "LabelingResult", "CandidateSearchReport", "evaluate_labeling",
    "enumerate_labelings", "_expanded_edges", "_side_vertices", "_recover_momenta",
]
