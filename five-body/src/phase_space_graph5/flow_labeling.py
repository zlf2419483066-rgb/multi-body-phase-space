"""Optional scalar-flow/DAG partition labeling for five-body topologies.

The legacy :func:`phase_space_graph5.labeling.enumerate_labelings` path keeps
its original final-permutation and single-assignment traversal.  This module
exposes a separate strategy which first computes the exact scalar flow on
every double occurrence, then partitions particle IDs along the directed DAG.
The default path sends every generated leaf through public physical
normalization.  Its explicit particle-orbit path evaluates one representative
per relative assignment and transports its certificates, while checking the
complete generated labeling against the transported result.  Neither path
replaces the physical rules or independence semantics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
import math
import time
from typing import Any, Callable, Iterable, Iterator, Mapping

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
from .labeling import (
    FailureRecord,
    LabelingLimits,
    LabelingResult,
    _expanded_edges,
    _side_vertices,
    _slot_order,
)
from .physical_labels import (
    CandidateMomentumRecord,
    CandidateNormalizationResult,
    candidate_record_from_evaluation,
    complete_labeling_from_evaluation,
    normalize_candidate_labeling,
)
from .structure import analyze_topology


FLOW_STRATEGY = "scalar_flow_partition"
_MAX_FAILURE_SAMPLES = 8


class _FlowStop(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class _FlowOrbitInputError(ValueError):
    """Flow-generator contract validation failure for orbit reuse."""

    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        message = "; ".join(issue.message for issue in self.issues)
        super().__init__(message or "flow orbit input is invalid")


@dataclass(frozen=True, slots=True)
class FlowLabelingReport:
    """Report returned by :func:`enumerate_flow_labelings`.

    ``flow_partition_generated_labelings`` is the number of complete DAG
    partition leaves actually generated.  ``traditional_search_space_size``
    is the size of the legacy ``factorial(n) * n**single_occurrences`` space;
    it is reported separately and is never used as an actual visit count.
    """

    topology: Topology
    status: RunStatus
    count_mode: CountMode
    strategy: str
    record_order: str
    scalar_flow: tuple[tuple[tuple[int, int], int], ...]
    scalar_flow_legal: bool
    scalar_flow_rejection: str | None
    traditional_search_space_size: int
    flow_partition_generated_labelings: int
    flow_partition_normalized_labelings: int
    flow_partition_branch_count: int
    momentum_pass_count: int
    candidate_records: tuple[CandidateMomentumRecord, ...]
    failure_counts: tuple[tuple[str, int], ...]
    failure_samples: tuple[FailureRecord, ...]
    failure_witnesses: tuple[tuple[str, LabelingResult], ...]
    limits: LabelingLimits
    stop_reason: str | None
    representative_evaluations: int = 0
    certificate_transports: int = 0
    particle_orbit_reuse: bool = False

    @property
    def complete(self) -> bool:
        return self.status is RunStatus.COMPLETE

    @property
    def raw_assignment_count(self) -> int:
        """Compatibility alias for actual generated flow leaves.

        This is deliberately not the legacy permutation-product space.  Use
        ``traditional_search_space_size`` for that theoretical size.
        """

        return self.flow_partition_generated_labelings

    @property
    def normalized_labeling_count(self) -> int:
        return self.flow_partition_normalized_labelings

    def failure_count(self, code: str) -> int:
        return dict(self.failure_counts).get(code, 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology.to_dict(),
            "status": self.status.value,
            "complete": self.complete,
            "count_mode": self.count_mode.value,
            "strategy": self.strategy,
            "record_order": self.record_order,
            "scalar_flow": [
                {"edge_index": ref[0], "occurrence": ref[1], "value": value}
                for ref, value in self.scalar_flow
            ],
            "scalar_flow_legal": self.scalar_flow_legal,
            "scalar_flow_rejection": self.scalar_flow_rejection,
            "counts": {
                "traditional_search_space_size": self.traditional_search_space_size,
                "flow_partition_generated_labelings": self.flow_partition_generated_labelings,
                "flow_partition_normalized_labelings": self.flow_partition_normalized_labelings,
                "flow_partition_branch_count": self.flow_partition_branch_count,
                "momentum_pass_count": self.momentum_pass_count,
                "candidate_records": len(self.candidate_records),
                "representative_evaluations": self.representative_evaluations,
                "certificate_transports": self.certificate_transports,
                "particle_orbit_reuse": self.particle_orbit_reuse,
            },
            "candidate_records": [record.to_dict() for record in self.candidate_records],
            "failure_counts": dict(self.failure_counts),
            "failure_samples": [sample.to_dict() for sample in self.failure_samples],
            "failure_witnesses": {
                code: result.to_dict() for code, result in self.failure_witnesses
            },
            "limits": self.limits.to_dict(),
            "stop_reason": self.stop_reason,
        }


def _traditional_search_space_size(topology: Topology) -> int:
    try:
        if isinstance(topology.n, bool) or not isinstance(topology.n, int) or topology.n < 0:
            return 0
        return math.factorial(topology.n) * topology.n ** len(
            _expanded_edges(topology, EdgeType.SINGLE)
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0


def _scalar_base(topology: Topology) -> dict[int, int]:
    values = {
        vertex: int(topology.slot_count(vertex) or 0)
        for vertex in topology.vertices
    }
    for _, _, edge in _expanded_edges(topology, EdgeType.SINGLE):
        values[edge.source] += 1
        values[edge.target] -= 1
    values[topology.initial_vertex] -= topology.n
    return values


def _scalar_double_flows(
    topology: Topology,
) -> tuple[tuple[tuple[int, int], int], ...]:
    refs = _expanded_edges(topology, EdgeType.DOUBLE)
    base = _scalar_base(topology)
    vertices = tuple(topology.vertices)
    return tuple(
        (
            (edge_index, occurrence),
            sum(
                base[vertex]
                for vertex in _side_vertices(
                    vertices, refs, (edge_index, occurrence), edge.target
                )
            ),
        )
        for edge_index, occurrence, edge in refs
    )


def _flow_is_legal(flows: Iterable[tuple[tuple[int, int], int]]) -> bool:
    return all(2 <= value <= 4 for _, value in flows)


def _topological_order(topology: Topology) -> tuple[int, ...]:
    adjacency = {vertex: set() for vertex in topology.vertices}
    indegree = {vertex: 0 for vertex in topology.vertices}
    refs = _expanded_edges(topology, EdgeType.SINGLE) + _expanded_edges(
        topology, EdgeType.DOUBLE
    )
    for _, _, edge in refs:
        if edge.target not in adjacency[edge.source]:
            adjacency[edge.source].add(edge.target)
            indegree[edge.target] += 1
    ready = sorted(vertex for vertex, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while ready:
        vertex = ready.pop(0)
        order.append(vertex)
        for target in sorted(adjacency[vertex]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
        ready.sort()
    if len(order) != len(topology.vertices):
        raise ValueError("scalar_flow_partition requires a directed acyclic topology")
    if not order or order[0] != topology.initial_vertex:
        raise ValueError(
            "scalar_flow_partition requires the initial vertex to be the DAG root"
        )
    return tuple(order)


def _channels_by_vertex(
    topology: Topology,
    scalar_flows: Mapping[tuple[int, int], int],
) -> dict[int, tuple[tuple[str, int, int, int], ...]]:
    channels: dict[int, list[tuple[str, int, int, int]]] = {
        vertex: [] for vertex in topology.vertices
    }
    for edge_index, occurrence, edge in _expanded_edges(topology, EdgeType.SINGLE):
        channels[edge.source].append(("single", edge_index, occurrence, 1))
    for edge_index, occurrence, edge in _expanded_edges(topology, EdgeType.DOUBLE):
        ref = (edge_index, occurrence)
        if ref not in scalar_flows:
            raise ValueError(f"missing scalar flow for double occurrence {ref!r}")
        channels[edge.source].append(("double", edge_index, occurrence, scalar_flows[ref]))
    for vertex, count in topology.final_slots:
        for slot in range(count):
            channels[vertex].append(("final", vertex, slot, 1))
    result: dict[int, tuple[tuple[str, int, int, int], ...]] = {}
    for vertex, items in channels.items():
        items.sort(key=lambda item: (item[0] == "final", item[1], item[2]))
        if len(items) != 2:
            raise ValueError(
                "scalar_flow_partition requires exactly two outgoing channels "
                "including final slots at every vertex"
            )
        result[vertex] = tuple(items)
    return result


def _incoming_refs(topology: Topology) -> dict[int, tuple[tuple[int, int], ...]]:
    incoming: dict[int, list[tuple[int, int]]] = {
        vertex: [] for vertex in topology.vertices
    }
    for edge_index, occurrence, edge in _expanded_edges(topology, EdgeType.SINGLE):
        incoming[edge.target].append((edge_index, occurrence))
    for edge_index, occurrence, edge in _expanded_edges(topology, EdgeType.DOUBLE):
        incoming[edge.target].append((edge_index, occurrence))
    return {vertex: tuple(sorted(refs)) for vertex, refs in incoming.items()}


def _partitions(
    items: tuple[int, ...], sizes: tuple[int, ...]
) -> Iterator[tuple[tuple[int, ...], ...]]:
    if len(items) != sum(sizes):
        return
    if not sizes:
        yield ()
        return
    size = sizes[0]
    for selected in itertools.combinations(items, size):
        selected_set = set(selected)
        remainder = tuple(item for item in items if item not in selected_set)
        for tail in _partitions(remainder, sizes[1:]):
            yield (tuple(selected),) + tail


def _iter_flow_labelings(
    topology: Topology,
    scalar_flows: Mapping[tuple[int, int], int],
    checkpoint: Callable[[], None],
    branch_counter: list[int],
) -> Iterator[Labeling]:
    order = _topological_order(topology)
    channels = _channels_by_vertex(topology, scalar_flows)
    incoming = _incoming_refs(topology)
    single_refs = _expanded_edges(topology, EdgeType.SINGLE)
    double_refs = _expanded_edges(topology, EdgeType.DOUBLE)

    def visit(
        position: int,
        edge_sets: dict[tuple[int, int], tuple[int, ...]],
        final_particles: dict[tuple[int, int], int],
        single_particles: dict[tuple[int, int], int],
        double_particles: dict[tuple[int, int], tuple[int, ...]],
    ) -> Iterator[Labeling]:
        if position == len(order):
            checkpoint()
            yield Labeling(
                final_particles=tuple(
                    FinalParticleLabel(vertex, slot, final_particles[(vertex, slot)])
                    for vertex, count in topology.final_slots
                    for slot in range(count)
                ),
                single_lines=tuple(
                    SingleLineLabel(edge_index, occurrence, single_particles[(edge_index, occurrence)])
                    for edge_index, occurrence, _ in single_refs
                ),
                double_lines=tuple(
                    DoubleLineLabel(
                        edge_index,
                        occurrence,
                        double_particles[(edge_index, occurrence)],
                    )
                    for edge_index, occurrence, _ in double_refs
                ),
            )
            return

        vertex = order[position]
        if vertex == topology.initial_vertex:
            available = tuple(range(1, topology.n + 1))
        else:
            values: list[int] = []
            for ref in incoming[vertex]:
                if ref not in edge_sets:
                    return
                values.extend(edge_sets[ref])
            if len(values) != len(set(values)):
                return
            available = tuple(sorted(values))
        output_channels = channels[vertex]
        sizes = tuple(channel[3] for channel in output_channels)
        if len(available) != sum(sizes):
            return
        for groups in _partitions(available, sizes):
            checkpoint()
            branch_counter[0] += 1
            next_edges = dict(edge_sets)
            next_finals = dict(final_particles)
            next_singles = dict(single_particles)
            next_doubles = dict(double_particles)
            for channel, group in zip(output_channels, groups):
                kind, first, second, _ = channel
                if kind == "final":
                    next_finals[(first, second)] = group[0]
                elif kind == "single":
                    next_edges[(first, second)] = group
                    next_singles[(first, second)] = group[0]
                else:
                    next_edges[(first, second)] = group
                    next_doubles[(first, second)] = group
            yield from visit(
                position + 1,
                next_edges,
                next_finals,
                next_singles,
                next_doubles,
            )

    yield from visit(0, {}, {}, {}, {})


def _reference_order_key(
    topology: Topology, record: CandidateMomentumRecord
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    complete = record.complete_labeling
    final_by_slot = {
        (label.vertex, label.slot): label.particle_id
        for label in complete.final_particles
    }
    single_by_ref = {
        (label.edge_index, label.occurrence): label.particle_id
        for label in complete.single_lines
    }
    return (
        tuple(final_by_slot[slot] for slot in _slot_order(topology)),
        tuple(
            single_by_ref[(edge_index, occurrence)]
            for edge_index, occurrence, _ in _expanded_edges(topology, EdgeType.SINGLE)
        ),
    )


def _traditional_order_records(
    topology: Topology, records: list[CandidateMomentumRecord]
) -> None:
    records.sort(key=lambda record: _reference_order_key(topology, record))


def _add_normalization_failures(
    normalized: Any,
    labeling: Labeling,
    failures: Counter[str],
    samples: list[FailureRecord],
    witnesses: dict[str, LabelingResult],
) -> None:
    issues = tuple(normalized.issues)
    if not issues:
        issues = (ValidationIssue(normalized.stage, normalized.stage),)
    for issue in issues:
        failures[issue.code] += 1
        evaluation = normalized.evaluation
        if isinstance(evaluation, LabelingResult):
            witnesses.setdefault(issue.code, evaluation)
        if len(samples) < _MAX_FAILURE_SAMPLES:
            samples.append(
                FailureRecord(
                    issue.code,
                    issue.message,
                    {"stage": normalized.stage, "labeling": labeling.to_dict()},
                )
            )


def _relative_orbit_labeling(
    topology: Topology, labeling: Labeling
) -> tuple[tuple[int, ...], tuple[int, ...], Labeling]:
    """Build the fixed-final representative used by particle orbit reuse."""

    fields = labeling.validate_fields(topology)
    if not fields.valid:
        raise _FlowOrbitInputError(fields.issues)

    expected_final_refs = _slot_order(topology)
    actual_final_refs = tuple(
        (label.vertex, label.slot) for label in labeling.final_particles
    )
    if actual_final_refs != expected_final_refs:
        raise _FlowOrbitInputError(
            (
                ValidationIssue(
                    "flow_orbit_incomplete_final_slots",
                    "Flow orbit input final slots do not cover the topology exactly: "
                    f"expected {expected_final_refs!r}, got {actual_final_refs!r}.",
                    "final_particles",
                ),
            )
        )
    expected_single_refs = tuple(
        (edge_index, occurrence)
        for edge_index, occurrence, _ in _expanded_edges(topology, EdgeType.SINGLE)
    )
    actual_single_refs = tuple(
        (label.edge_index, label.occurrence) for label in labeling.single_lines
    )
    if actual_single_refs != expected_single_refs:
        raise _FlowOrbitInputError(
            (
                ValidationIssue(
                    "flow_orbit_incomplete_single_occurrences",
                    "Flow orbit input single-line occurrences do not cover the topology "
                    f"exactly: expected {expected_single_refs!r}, got {actual_single_refs!r}.",
                    "single_lines",
                ),
            )
        )
    expected_double_refs = tuple(
        (edge_index, occurrence)
        for edge_index, occurrence, _ in _expanded_edges(topology, EdgeType.DOUBLE)
    )
    actual_double_refs = tuple(
        (label.edge_index, label.occurrence) for label in labeling.double_lines
    )
    if actual_double_refs != expected_double_refs:
        raise _FlowOrbitInputError(
            (
                ValidationIssue(
                    "flow_orbit_incomplete_double_occurrences",
                    "Flow orbit input double-line occurrences do not cover the topology "
                    f"exactly: expected {expected_double_refs!r}, got {actual_double_refs!r}.",
                    "double_lines",
                ),
            )
        )

    final_by_slot = {
        (label.vertex, label.slot): label.particle_id
        for label in labeling.final_particles
    }
    slots = _slot_order(topology)
    final_particles = tuple(final_by_slot[slot] for slot in slots)
    expected = tuple(range(1, topology.n + 1))
    if tuple(sorted(final_particles)) != expected:
        raise ValueError("flow final labels are not a particle permutation")
    inverse = {particle: index + 1 for index, particle in enumerate(final_particles)}
    single_by_ref = {
        (label.edge_index, label.occurrence): label.particle_id
        for label in labeling.single_lines
    }
    single_refs = _expanded_edges(topology, EdgeType.SINGLE)
    actual_singles = tuple(
        single_by_ref[(edge_index, occurrence)]
        for edge_index, occurrence, _ in single_refs
    )
    relative_singles = tuple(inverse[particle] for particle in actual_singles)
    representative = Labeling(
        tuple(
            FinalParticleLabel(vertex, slot, particle)
            for (vertex, slot), particle in zip(slots, expected)
        ),
        tuple(
            SingleLineLabel(edge_index, occurrence, particle)
            for (edge_index, occurrence, _), particle in zip(single_refs, relative_singles)
        ),
    )
    return final_particles, relative_singles, representative


def _orbit_normalization(
    topology: Topology,
    labeling: Labeling,
    representative_cache: dict[tuple[int, ...], LabelingResult],
    prepared_sides: Mapping[tuple[int, int], set[int]],
) -> tuple[CandidateNormalizationResult, bool, bool]:
    """Evaluate/transport one flow leaf through a representative cache.

    The booleans report cache miss and certificate transport respectively.
    A rejected representative is returned as a failed normalization result so
    the caller can preserve the normal failure witness and stop immediately.
    """

    from .labeling import _transform_result, evaluate_labeling

    try:
        final_particles, relative_singles, representative_labeling = _relative_orbit_labeling(
            topology, labeling
        )
    except _FlowOrbitInputError as exc:
        return (
            CandidateNormalizationResult(
                "field_rejected",
                issues=exc.issues,
            ),
            False,
            False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return (
            CandidateNormalizationResult(
                "field_rejected",
                issues=(ValidationIssue("flow_orbit_input_invalid", str(exc)),),
            ),
            False,
            False,
        )
    representative = representative_cache.get(relative_singles)
    miss = representative is None
    if miss:
        representative = evaluate_labeling(
            topology, representative_labeling, _prepared_sides=prepared_sides
        )
        representative_cache[relative_singles] = representative
    if not representative.valid or representative.momentum is None:
        transformed = _transform_result(representative, labeling, final_particles)
        issues = tuple(
            ValidationIssue(code, message)
            for code, message in zip(
                transformed.failure_codes, transformed.failure_messages
            )
        )
        return (
            CandidateNormalizationResult(
                "momentum_rejected", evaluation=transformed, issues=issues
            ),
            miss,
            False,
        )
    transformed = _transform_result(representative, labeling, final_particles)
    if not transformed.valid or transformed.momentum is None:
        issues = tuple(
            ValidationIssue(code, message)
            for code, message in zip(
                transformed.failure_codes, transformed.failure_messages
            )
        )
        return (
            CandidateNormalizationResult(
                "momentum_rejected", evaluation=transformed, issues=issues
            ),
            miss,
            True,
        )
    try:
        reconstructed = complete_labeling_from_evaluation(topology, transformed)
    except (TypeError, ValueError) as exc:
        return (
            CandidateNormalizationResult(
                "field_rejected",
                evaluation=transformed,
                issues=(
                    ValidationIssue(
                        "flow_orbit_complete_labeling_mismatch",
                        f"Transported certificate could not reconstruct complete labeling: {exc}",
                        "complete_labeling",
                    ),
                ),
            ),
            miss,
            True,
        )
    if reconstructed.to_dict() != labeling.to_dict():
        return (
            CandidateNormalizationResult(
                "field_rejected",
                evaluation=transformed,
                issues=(
                    ValidationIssue(
                        "flow_orbit_complete_labeling_mismatch",
                        "Transported certificate complete labeling differs from the "
                        "generated labeling.",
                        "complete_labeling",
                    ),
                ),
            ),
            miss,
            True,
        )
    return (
        CandidateNormalizationResult(
            "momentum_candidate",
            record=candidate_record_from_evaluation(topology, transformed),
            evaluation=transformed,
        ),
        miss,
        True,
    )


def enumerate_flow_labelings(
    topology: Topology,
    count_mode: CountMode | str = CountMode.FIXED_LABELS,
    limits: LabelingLimits | None = None,
    *,
    cancel_requested: Callable[[], bool] | None = None,
    deadline: float | None = None,
    strategy: str = FLOW_STRATEGY,
    record_order: str = "flow",
    use_particle_orbit: bool = False,
) -> FlowLabelingReport:
    """Enumerate legal labelings using exact scalar-flow partitions.

    ``strategy`` is explicit so callers cannot silently select this path.
    ``record_order='flow'`` retains deterministic DAG partition order;
    ``record_order='reference'`` sorts complete candidate records using the
    legacy final-permutation then single-assignment lexicographic order.
    """

    started = time.monotonic()
    if not isinstance(topology, Topology):
        raise TypeError("enumerate_flow_labelings expects a Topology object")
    if strategy != FLOW_STRATEGY:
        raise ValueError(f"unsupported flow labeling strategy: {strategy!r}")
    if record_order not in {"flow", "reference"}:
        raise ValueError("record_order must be 'flow' or 'reference'")
    if type(use_particle_orbit) is not bool:
        raise TypeError("use_particle_orbit must be a boolean")
    mode = count_mode if isinstance(count_mode, CountMode) else CountMode(count_mode)
    limits = limits or LabelingLimits()
    if not isinstance(limits, LabelingLimits):
        raise TypeError("limits must be a LabelingLimits object or None")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None")
    if deadline is not None and (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("deadline must be a finite number or None")

    deadline_values = [
        value for value in (deadline, limits.deadline) if value is not None
    ]
    effective_deadline = min(deadline_values) if deadline_values else None
    if limits.max_seconds is not None:
        local_deadline = started + limits.max_seconds
        effective_deadline = (
            local_deadline
            if effective_deadline is None
            else min(effective_deadline, local_deadline)
        )

    traditional_space = _traditional_search_space_size(topology)

    def preflight_stop_reason() -> str | None:
        if cancel_requested is not None and cancel_requested():
            return "cancel_requested"
        if limits.max_labelings == 0 or limits.max_results == 0:
            return "limit_zero_budget"
        if effective_deadline is not None and time.monotonic() >= effective_deadline:
            return "deadline_exceeded"
        return None

    def incomplete_report(reason: str) -> FlowLabelingReport:
        return FlowLabelingReport(
            topology,
            RunStatus.INCOMPLETE,
            mode,
            FLOW_STRATEGY,
            record_order,
            (),
            False,
            None,
            traditional_space,
            0,
            0,
            0,
            0,
            (),
            (),
            (),
            (),
            limits,
            reason,
            0,
            0,
            use_particle_orbit,
        )

    initial_stop = preflight_stop_reason()
    if initial_stop is not None:
        return incomplete_report(initial_stop)

    structure = analyze_topology(topology)
    if not structure.valid:
        return FlowLabelingReport(
            topology,
            RunStatus.FAILED,
            mode,
            FLOW_STRATEGY,
            record_order,
            (),
            False,
            "structure_rejected",
            traditional_space,
            0,
            0,
            0,
            0,
            (),
            tuple((code, 1) for code in structure.all_failure_codes()),
            (),
            (),
            limits,
            None,
            0,
            0,
            use_particle_orbit,
        )

    scalar_flow = _scalar_double_flows(topology)
    scalar_stop = preflight_stop_reason()
    if scalar_stop is not None:
        return incomplete_report(scalar_stop)
    scalar_legal = _flow_is_legal(scalar_flow)
    if not scalar_legal:
        return FlowLabelingReport(
            topology,
            RunStatus.COMPLETE,
            mode,
            FLOW_STRATEGY,
            record_order,
            scalar_flow,
            False,
            "scalar_flow_rejected",
            traditional_space,
            0,
            0,
            0,
            0,
            (),
            (),
            (),
            (),
            limits,
            None,
            0,
            0,
            use_particle_orbit,
        )

    raw_generated = 0
    normalized_count = 0
    representative_evaluations = 0
    certificate_transports = 0
    branch_counter = [0]
    momentum_pass = 0
    records: list[CandidateMomentumRecord] = []
    failures: Counter[str] = Counter()
    samples: list[FailureRecord] = []
    witnesses: dict[str, LabelingResult] = {}
    stop_reason: str | None = None
    fatal_reason: str | None = None
    representative_cache: dict[tuple[int, ...], LabelingResult] = {}
    prepared_sides: dict[tuple[int, int], set[int]] = {}
    if use_particle_orbit:
        double_refs = _expanded_edges(topology, EdgeType.DOUBLE)
        vertices = tuple(topology.vertices)
        prepared_sides = {
            (edge_index, occurrence): _side_vertices(
                vertices, double_refs, (edge_index, occurrence), edge.target
            )
            for edge_index, occurrence, edge in double_refs
        }

    def time_checkpoint() -> None:
        if cancel_requested is not None and cancel_requested():
            raise _FlowStop("cancel_requested")
        if effective_deadline is not None and time.monotonic() >= effective_deadline:
            raise _FlowStop("deadline_exceeded")

    def checkpoint() -> None:
        time_checkpoint()
        if limits.max_labelings is not None and raw_generated >= limits.max_labelings:
            raise _FlowStop("limit_max_labelings")
        if limits.max_results is not None and len(records) >= limits.max_results:
            raise _FlowStop("limit_max_results")

    try:
        if limits.max_labelings == 0 or limits.max_results == 0:
            raise _FlowStop("limit_zero_budget")
        scalar_map = dict(scalar_flow)
        for labeling in _iter_flow_labelings(
            topology, scalar_map, checkpoint, branch_counter
        ):
            raw_generated += 1
            if use_particle_orbit:
                normalized, cache_miss, transported = _orbit_normalization(
                    topology, labeling, representative_cache, prepared_sides
                )
                representative_evaluations += int(cache_miss)
                certificate_transports += int(transported)
            else:
                normalized_count += 1
                normalized = normalize_candidate_labeling(topology, labeling)
            if normalized.record is not None:
                momentum_pass += 1
                records.append(normalized.record)
            else:
                _add_normalization_failures(
                    normalized, labeling, failures, samples, witnesses
                )
                fatal_reason = (
                    "flow_orbit_rejected"
                    if use_particle_orbit
                    else "flow_normalization_rejected"
                )
                raise _FlowStop(fatal_reason)
    except _FlowStop as exc:
        stop_reason = exc.reason

    if stop_reason is None and raw_generated == 0:
        fatal_reason = "flow_partition_empty"
        stop_reason = fatal_reason

    if record_order == "reference":
        try:
            time_checkpoint()
            _traditional_order_records(topology, records)
            time_checkpoint()
        except _FlowStop as exc:
            stop_reason = stop_reason or exc.reason

    if fatal_reason is not None:
        status = RunStatus.FAILED
    else:
        status = RunStatus.INCOMPLETE if stop_reason else RunStatus.COMPLETE
    return FlowLabelingReport(
        topology,
        status,
        mode,
        FLOW_STRATEGY,
        record_order,
        scalar_flow,
        True,
        None,
        traditional_space,
        raw_generated,
        normalized_count,
        branch_counter[0],
        momentum_pass,
        tuple(records),
        tuple(sorted(failures.items())),
        tuple(samples),
        tuple(sorted(witnesses.items())),
        limits,
        stop_reason,
        representative_evaluations,
        certificate_transports,
        use_particle_orbit,
    )


__all__ = [
    "FLOW_STRATEGY",
    "FlowLabelingReport",
    "enumerate_flow_labelings",
]
