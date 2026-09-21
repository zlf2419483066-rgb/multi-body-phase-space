"""G06 exhaustive particle labels, momentum recovery, and validity filters.

The default search caches fixed-final particle-orbit representatives and tree
cuts while preserving reference traversal, raw counters, and budget semantics.
Only successful representatives are expanded into full result certificates.
Use use_symmetry=False to compare with the exhaustive evaluation route.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
import json
import math
import time
from typing import Any, Callable, Iterable, Mapping

from .canonical import CanonicalizationResult, canonicalize_labeling
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
from .invariants import IndependenceResult, analyze_invariants
from .structure import analyze_topology


Vector = tuple[int, ...]


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
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


def _negate(value: Vector) -> Vector:
    return tuple(-item for item in value)


def _expanded_edges(topology: Topology, edge_type: EdgeType) -> tuple[tuple[int, int, Any], ...]:
    return tuple(
        (edge_index, occurrence, edge)
        for edge_index, edge in enumerate(topology.edges)
        if edge.edge_type is edge_type and edge.multiplicity > 0
        for occurrence in range(edge.multiplicity)
    )


@dataclass(frozen=True, slots=True)
class LabelingLimits:
    """Cooperative limits for one topology's labeling search."""

    max_labelings: int | None = None
    max_results: int | None = None
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_labelings", "max_results"):
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.max_seconds is not None:
            if isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float)):
                raise ValueError("max_seconds must be a non-negative finite number or None")
            if not math.isfinite(self.max_seconds) or self.max_seconds < 0:
                raise ValueError("max_seconds must be a non-negative finite number or None")

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "max_labelings": self.max_labelings,
            "max_results": self.max_results,
            "max_seconds": self.max_seconds,
        }


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """One compact example of a rejected labeling."""

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


@dataclass(frozen=True, slots=True)
class LabelingResult:
    """One labeling evaluation, valid or rejected with explicit reasons."""

    labeling: Labeling
    valid: bool
    failure_codes: tuple[str, ...]
    failure_messages: tuple[str, ...]
    momentum: MomentumCertificate | None
    independence: IndependenceResult | None
    canonical: CanonicalizationResult | None
    single_occurrence_counts: tuple[tuple[int, int], ...]
    measure_cancellations: tuple[tuple[int, int], ...]

    @property
    def variables(self) -> tuple[tuple[int, ...], ...]:
        return self.momentum.variables if self.momentum else ()

    @property
    def total_measure_cancellations(self) -> int:
        return sum(count for _, count in self.measure_cancellations)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "failure_codes": list(self.failure_codes),
            "failure_messages": list(self.failure_messages),
            "labeling": self.labeling.to_dict(),
            "momentum": self.momentum.to_dict() if self.momentum else None,
            "independence": self.independence.to_dict() if self.independence else None,
            "canonical": self.canonical.to_dict() if self.canonical else None,
            "single_occurrence_counts": [
                {"particle": particle, "count": count}
                for particle, count in self.single_occurrence_counts
            ],
            "measure_cancellations": [
                {"particle": particle, "count": count}
                for particle, count in self.measure_cancellations
            ],
        }


@dataclass(frozen=True, slots=True)
class LabelingSearchReport:
    """Auditable G06 search result for one topology."""

    topology: Topology
    status: RunStatus
    count_mode: CountMode
    raw_labeling_count: int
    momentum_valid_count: int
    independent_count: int
    unique_count: int
    valid_results: tuple[LabelingResult, ...]
    failure_counts: tuple[tuple[str, int], ...]
    failure_samples: tuple[FailureRecord, ...]
    issues: tuple[ValidationIssue, ...]
    limits: LabelingLimits
    stop_reason: str | None
    representative_evaluations: int = 0
    search_strategy: str = "reference"
    failure_witnesses: tuple[tuple[str, LabelingResult], ...] = ()

    @property
    def complete(self) -> bool:
        return self.status is RunStatus.COMPLETE

    @property
    def effective(self) -> bool:
        return bool(self.valid_results)

    def failure_count(self, code: str) -> int:
        return dict(self.failure_counts).get(code, 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology.to_dict(),
            "status": self.status.value,
            "complete": self.complete,
            "effective": self.effective,
            "count_mode": self.count_mode.value,
            "counts": {
                "raw_labelings": self.raw_labeling_count,
                "momentum_valid": self.momentum_valid_count,
                "independent": self.independent_count,
                "unique": self.unique_count,
            },
            "valid_results": [result.to_dict() for result in self.valid_results],
            "failure_counts": dict(self.failure_counts),
            "failure_samples": [sample.to_dict() for sample in self.failure_samples],
            "issues": [issue.to_dict() for issue in self.issues],
            "limits": self.limits.to_dict(),
            "stop_reason": self.stop_reason,
            "representative_evaluations": self.representative_evaluations,
            "search_strategy": self.search_strategy,
            'failure_witnesses': {code: result.to_dict() for code, result in self.failure_witnesses},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    final_labels = tuple(
        FinalParticleLabel(vertex, slot, particle)
        for (vertex, slot), particle in zip(slots, final_particles)
    )
    single_labels = tuple(
        SingleLineLabel(edge_index, occurrence, particle)
        for (edge_index, occurrence, _), particle in zip(single_refs, single_particles)
    )
    return Labeling(final_particles=final_labels, single_lines=single_labels)


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


def _recover_momenta(topology: Topology, labeling: Labeling, prepared_sides=None) -> tuple[MomentumCertificate | None, tuple[str, ...], tuple[str, ...]]:
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
        final_by_vertex[label.vertex] = _add(final_by_vertex[label.vertex], _unit(n, label.particle_id))
    single_in: dict[int, Vector] = {vertex: _zero(n) for vertex in vertices}
    single_out: dict[int, Vector] = {vertex: _zero(n) for vertex in vertices}
    for edge_index, occurrence, edge in _expanded_edges(topology, EdgeType.SINGLE):
        particle = single_by_ref.get((edge_index, occurrence))
        if particle is None:
            return None, ("missing_single_label",), ("Every internal single-line occurrence needs a label.",)
        vector = _unit(n, particle)
        single_out[edge.source] = _add(single_out[edge.source], vector)
        single_in[edge.target] = _add(single_in[edge.target], vector)
    initial = tuple(1 for _ in range(n))
    base: dict[int, Vector] = {
        vertex: _sub(_add(final_by_vertex[vertex], single_out[vertex]), _add(single_in[vertex], initial if vertex == topology.initial_vertex else _zero(n)))
        for vertex in vertices
    }
    # base is final + single_out - single_in - initial, i.e. -b in the
    # vertex conservation equation.
    double_momenta: list[tuple[int, int, Vector]] = []
    for edge_index, occurrence, edge in double_refs:
        side = (prepared_sides[(edge_index, occurrence)] if prepared_sides is not None
                else _side_vertices(vertices, double_refs, (edge_index, occurrence), edge.target))
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
        vector = next(vector for index, item, vector in double_momenta if (index, item) == (edge_index, occurrence))
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
    if failures:
        return certificate, tuple(dict.fromkeys(failures)), tuple(dict.fromkeys(messages))
    return certificate, (), ()


def evaluate_labeling(
    topology: Topology,
    labeling: Labeling,
    count_mode: CountMode | str = CountMode.FIXED_LABELS,
) -> LabelingResult:
    """Evaluate one explicitly supplied labeling without search limits."""

    if not isinstance(topology, Topology):
        raise TypeError("evaluate_labeling expects a Topology object")
    if not isinstance(labeling, Labeling):
        raise TypeError("evaluate_labeling expects a Labeling object")
    topology_report = topology.validate_fields()
    if not topology_report.valid:
        raise ValueError(
            "Topology fields must be valid before labeling evaluation: "
            f"{topology_report.to_dict()}"
        )
    try:
        mode = count_mode if isinstance(count_mode, CountMode) else CountMode(count_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown labeling count mode: {count_mode!r}") from exc
    single_counts = Counter(label.particle_id for label in labeling.single_lines)
    cancellations = tuple(sorted((particle, single_counts.get(particle, 0)) for particle in range(1, topology.n + 1)))
    field_report = labeling.validate_fields(topology)
    if not field_report.valid:
        codes = field_report.error_codes()
        messages = tuple(issue.message for issue in field_report.issues if issue.severity == "error")
        return LabelingResult(
            labeling=labeling,
            valid=False,
            failure_codes=codes,
            failure_messages=messages,
            momentum=None,
            independence=None,
            canonical=None,
            single_occurrence_counts=tuple(sorted(single_counts.items())),
            measure_cancellations=cancellations,
        )
    momentum, failures, messages = _recover_momenta(topology, labeling)
    if momentum is not None and not failures and labeling.double_lines:
        recovered = {(index, occurrence): tuple(i + 1 for i, c in enumerate(vector) if c == 1)
                     for index, occurrence, vector in momentum.double_momenta}
        mismatched = [label for label in labeling.double_lines
                      if tuple(sorted(label.particle_ids)) != recovered.get((label.edge_index, label.occurrence))]
        if mismatched:
            failures = ('declared_double_momentum_mismatch',)
            messages = ('Explicit double-line subsets disagree with vertex momentum conservation.',)
    independence: IndependenceResult | None = None
    canonical: CanonicalizationResult | None = None
    if momentum is not None and not failures:
        independence = analyze_invariants(momentum.variables, topology.n)
        if not independence.independent:
            failures = ("dependent_invariants",)
            messages = ("Recovered double-line variables are dependent under the total-mass constraint.",)
    if not failures:
        try:
            canonical = canonicalize_labeling(topology, labeling, mode)
        except ValueError as exc:
            failures = ("canonicalization_error",)
            messages = (str(exc),)
    return LabelingResult(
        labeling=labeling,
        valid=not failures,
        failure_codes=failures,
        failure_messages=messages,
        momentum=momentum,
        independence=independence,
        canonical=canonical,
        single_occurrence_counts=tuple(sorted(single_counts.items())),
        measure_cancellations=cancellations,
    )


def _failed_report(topology: Topology, mode: CountMode, limits: LabelingLimits, codes: tuple[str, ...]) -> LabelingSearchReport:
    return LabelingSearchReport(
        topology=topology,
        status=RunStatus.FAILED,
        count_mode=mode,
        raw_labeling_count=0,
        momentum_valid_count=0,
        independent_count=0,
        unique_count=0,
        valid_results=(),
        failure_counts=tuple((code, 1) for code in codes),
        failure_samples=(),
        issues=tuple(ValidationIssue(code, f"Topology failed before labeling search: {code}.") for code in codes),
        limits=limits,
        stop_reason=None,
    )


def enumerate_labelings(
    topology: Topology,
    count_mode: CountMode | str = CountMode.FIXED_LABELS,
    limits: LabelingLimits | None = None,
    *,
    cancel_requested: Callable[[], bool] | None = None,
    use_symmetry: bool = True,
) -> LabelingSearchReport:
    """Enumerate all labelings for one structurally valid topology."""

    if not isinstance(topology, Topology):
        raise TypeError("enumerate_labelings expects a Topology object")
    try:
        mode = count_mode if isinstance(count_mode, CountMode) else CountMode(count_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown labeling count mode: {count_mode!r}") from exc
    if limits is None:
        limits = LabelingLimits()
    if not isinstance(limits, LabelingLimits):
        raise TypeError("limits must be a LabelingLimits object or None")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None")
    if not isinstance(use_symmetry, bool):
        raise TypeError('use_symmetry must be a boolean')
    structure = analyze_topology(topology)
    if not structure.valid:
        return _failed_report(topology, mode, limits, structure.all_failure_codes())
    slots = _slot_order(topology)
    single_refs = _expanded_edges(topology, EdgeType.SINGLE)
    started = time.monotonic()
    raw = momentum_valid = independent_count = 0
    results: dict[tuple[Any, ...], LabelingResult] = {}
    failures: Counter[str] = Counter()
    samples: list[FailureRecord] = []
    witnesses: dict[str, LabelingResult] = {}
    stop_reason: str | None = None
    representative_evaluations = 0
    representatives = {}
    independence_cache = {}
    prepared_sides = None

    class StopSearch(Exception):
        def __init__(self, reason: str) -> None:
            self.reason = reason

    def checkpoint() -> None:
        if cancel_requested is not None and cancel_requested():
            raise StopSearch("cancel_requested")
        if limits.max_seconds is not None and time.monotonic() - started >= limits.max_seconds:
            raise StopSearch("limit_max_seconds")

    try:
        checkpoint()
        if limits.max_labelings == 0 or limits.max_results == 0:
            raise StopSearch("limit_zero_budget")
        particle_values = tuple(range(1, topology.n + 1))
        if use_symmetry:
            double_refs = _expanded_edges(topology, EdgeType.DOUBLE)
            prepared_sides = {}
            for edge_index, occurrence, edge in double_refs:
                checkpoint()
                prepared_sides[(edge_index, occurrence)] = _side_vertices(
                    tuple(topology.vertices), double_refs, (edge_index, occurrence), edge.target)
        single_choices = itertools.product(particle_values, repeat=len(single_refs))
        for final_particles in itertools.permutations(particle_values):
            inverse = {particle: index + 1 for index, particle in enumerate(final_particles)}
            for single_particles in single_choices:
                checkpoint()
                if limits.max_labelings is not None and raw >= limits.max_labelings:
                    raise StopSearch("limit_max_labelings")
                raw += 1
                if not use_symmetry:
                    labeling = _labeling_from_assignments(
                        topology, slots, final_particles, single_refs, tuple(single_particles))
                    representative_evaluations += 1
                    result = evaluate_labeling(topology, labeling, mode)
                else:
                    # Keep the reference traversal and raw-limit semantics, but
                    # solve only one fixed-final representative per particle orbit.
                    assignment = tuple(inverse[p] for p in single_particles)
                    if assignment not in representatives:
                        representative_evaluations += 1
                        base_labeling = _labeling_from_assignments(
                            topology, slots, particle_values, single_refs, assignment)
                        momentum, codes, messages = _recover_momenta(
                            topology, base_labeling, prepared_sides)
                        independence = None
                        if momentum is not None and not codes:
                            variables = momentum.variables
                            if variables not in independence_cache:
                                independence_cache[variables] = analyze_invariants(variables, topology.n)
                            independence = independence_cache[variables]
                            if not independence.independent:
                                codes = ('dependent_invariants',)
                                messages = ('Recovered double-line variables are dependent under the total-mass constraint.',)
                        representatives[assignment] = (momentum, codes, messages, independence)
                    momentum, codes, messages, independence = representatives[assignment]
                    if codes:
                        if any(code not in witnesses for code in codes):
                            # This is a fixed-final representative, with its own
                            # labeling and exact certificate, not a mislabeled
                            # copy of the current raw assignment. No new solve.
                            base_labeling = _labeling_from_assignments(
                                topology, slots, particle_values, single_refs, assignment)
                            base_counts = Counter(assignment)
                            witness = LabelingResult(
                                base_labeling, False, codes, messages, momentum, independence, None,
                                tuple(sorted(base_counts.items())),
                                tuple((p, base_counts.get(p, 0)) for p in particle_values))
                            for code in codes:
                                witnesses.setdefault(code, witness)
                        # Failure categories are invariant under a common ID
                        # permutation. Keep raw counts and reference sample order
                        # without allocating millions of rejected result objects.
                        if momentum is not None and momentum.verified:
                            momentum_valid += 1
                        for code, message in zip(codes, messages):
                            failures[code] += 1
                            if len(samples) < 8:
                                labeling = _labeling_from_assignments(
                                    topology, slots, final_particles, single_refs, tuple(single_particles))
                                samples.append(FailureRecord(code, message, {'labeling': labeling.to_dict()}))
                        continue
                    labeling = _labeling_from_assignments(
                        topology, slots, final_particles, single_refs, tuple(single_particles))
                    canonical = None
                    if not codes:
                        def rename_vector(vector):
                            return tuple(vector[inverse[p] - 1] for p in particle_values)
                        momentum = MomentumCertificate(
                            tuple((i, occurrence, rename_vector(vector))
                                  for i, occurrence, vector in momentum.double_momenta),
                            tuple((v, rename_vector(vector)) for v, vector in momentum.vertex_residuals),
                            momentum.verified,
                        )
                        # Rebuild exact expressions/certificates for actual IDs;
                        # do not reuse a certificate with unrenamed mass symbols.
                        variables = momentum.variables
                        if variables not in independence_cache:
                            independence_cache[variables] = analyze_invariants(variables, topology.n)
                        independence = independence_cache[variables]
                        canonical = canonicalize_labeling(topology, labeling, mode)
                    single_counts = Counter(single_particles)
                    result = LabelingResult(
                        labeling, not codes, codes, messages, momentum, independence, canonical,
                        tuple(sorted(single_counts.items())),
                        tuple((p, single_counts.get(p, 0)) for p in particle_values),
                    )
                if result.momentum is not None and result.momentum.verified:
                    momentum_valid += 1
                if result.independence is not None and result.independence.independent:
                    independent_count += 1
                if result.valid:
                    key = result.canonical.key if result.canonical else result.labeling.to_json()
                    if key not in results:
                        if limits.max_results is not None and len(results) >= limits.max_results:
                            raise StopSearch("limit_max_results")
                        results[key] = result
                else:
                    for code in result.failure_codes:
                        witnesses.setdefault(code, result)
                    for code, message in zip(result.failure_codes, result.failure_messages):
                        failures[code] += 1
                        if len(samples) < 8:
                            samples.append(FailureRecord(code, message, {"labeling": result.labeling.to_dict()}))
            # product() iterators are exhausted after the first final assignment;
            # create a fresh product for each final permutation.
            single_choices = itertools.product(particle_values, repeat=len(single_refs))
    except StopSearch as exc:
        stop_reason = exc.reason
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    if stop_reason:
        failures[stop_reason] += 1
    ordered = tuple(results[key] for key in sorted(results, key=repr))
    return LabelingSearchReport(
        topology=topology,
        status=RunStatus.INCOMPLETE if stop_reason else RunStatus.COMPLETE,
        count_mode=mode,
        raw_labeling_count=raw,
        momentum_valid_count=momentum_valid,
        independent_count=independent_count,
        unique_count=len(ordered),
        valid_results=ordered,
        failure_counts=tuple(sorted(failures.items())),
        failure_samples=tuple(samples),
        issues=(),
        limits=limits,
        stop_reason=stop_reason,
        representative_evaluations=representative_evaluations,
        search_strategy='particle_orbit_cache' if use_symmetry else 'reference',
        failure_witnesses=tuple(sorted(witnesses.items())),
    )


__all__ = [
    "FailureRecord",
    "LabelingLimits",
    "LabelingResult",
    "LabelingSearchReport",
    "MomentumCertificate",
    "enumerate_labelings",
    "evaluate_labeling",
]
