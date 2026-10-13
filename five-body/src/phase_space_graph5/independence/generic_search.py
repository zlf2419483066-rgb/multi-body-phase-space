"""Bounded common-particle-renaming search for the generic five-body label.

The search keeps one fixed final-state slot ordering and enumerates only the
``5**L`` assignments of particle ids to expanded single-line occurrences.
It deliberately delegates momentum recovery to the existing public labeling
layer and delegates the generic independence claim to the independent
generic-family verifier.  It never enumerates topologies or changes canonical
identity rules.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
import hashlib
import itertools
import json
import math
import time
from typing import Any

from ..domain import EdgeType, FinalParticleLabel, Labeling, SingleLineLabel, Topology
from ..label_identity import canonicalize_candidate_labeling
from ..labeling import evaluate_labeling
from ..physical_labels import (
    CandidateMomentumRecord,
    candidate_record_from_evaluation,
    normalize_candidate_labeling,
)
from .coordinates import Z_ORDER
from .exact_algebra import rank
from ..structure import analyze_topology
from .generic_family import GENERIC_THEOREM_VERSION, evaluate_generic_label
from .generic_verification import verify_generic_label
from .generic_verification import _pair_rows, _subset_row
from .hajer_constraints import _linear_coefficients, _subset_expression, build_hajer_constraint
from .sources import FIXED_MODE, VerifiedSource


GENERIC_SEARCH_SCHEMA_VERSION = "F5-I-generic-search-v1"
GENERIC_SEARCH_THEOREM_VERSION = GENERIC_THEOREM_VERSION
GENERIC_SEARCH_COVERAGE_RULE = "spec_section_12_global_S5_fixed_final"
GENERIC_SEARCH_SCOPE = "fixed_final_slots_witnesses_only"
GENERIC_SEARCH_BACKENDS = ("reference", "optimized")


def _hajer_subset_rows() -> dict[tuple[int, ...], tuple[Fraction, ...]]:
    """Precompute producer rows from the Hajer symbolic ``cQ c^T`` path."""

    descriptor = build_hajer_constraint()
    return {
        subset: tuple(_linear_coefficients(_subset_expression(descriptor, subset))[:len(Z_ORDER)])
        for size in (2, 3, 4)
        for subset in itertools.combinations(range(1, 6), size)
    }


def _replay_pair_subset_rows() -> dict[tuple[int, ...], tuple[Fraction, ...]]:
    """Build the verifier's independent pair-invariant rows separately."""

    pairs = _pair_rows()
    return {
        subset: tuple(_subset_row(subset, pairs)[:len(Z_ORDER)])
        for size in (2, 3, 4)
        for subset in itertools.combinations(range(1, 6), size)
    }


_OPTIMIZED_SUBSET_Z_ROWS = _hajer_subset_rows()
_REPLAY_PAIR_SUBSET_Z_ROWS = _replay_pair_subset_rows()


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _strict_json_equal(left: object, right: object) -> bool:
    try:
        return _json_bytes(left) == _json_bytes(right)
    except (TypeError, ValueError):
        return False


def _single_occurrences(topology: Topology) -> tuple[tuple[int, int], ...]:
    """Match the existing labeling module's deterministic expanded order."""

    return tuple(
        (edge_index, occurrence)
        for edge_index, edge in enumerate(topology.edges)
        if edge.edge_type is EdgeType.SINGLE and edge.multiplicity > 0
        for occurrence in range(edge.multiplicity)
    )


def _final_slots(topology: Topology) -> tuple[tuple[int, int], ...]:
    return tuple(
        (vertex, slot)
        for vertex, count in sorted(topology.final_slots, key=lambda item: (item[0], item[1]))
        for slot in range(count)
    )


def _topology_payload(topology: Topology) -> dict[str, object]:
    return topology.to_dict()


def _topology_binding(topology: Topology) -> dict[str, object]:
    payload = _topology_payload(topology)
    return {"topology": payload, "topology_digest": _sha256(payload)}


def _fixed_labeling(topology: Topology, slots: Sequence[tuple[int, int]], assignment: Sequence[int]) -> Labeling:
    return Labeling(
        final_particles=tuple(
            FinalParticleLabel(vertex, slot, particle)
            for (vertex, slot), particle in zip(slots, range(1, 6))
        ),
        single_lines=tuple(
            SingleLineLabel(edge_index, occurrence, particle)
            for (edge_index, occurrence), particle in zip(
                _single_occurrences(topology), assignment
            )
        ),
    )


def _make_source(topology: Topology, labeling: Labeling) -> VerifiedSource:
    normalized = normalize_candidate_labeling(topology, labeling)
    if normalized.stage != "momentum_candidate" or normalized.record is None:
        raise ValueError("a momentum-passing labeling did not normalize to a source record")
    identity = canonicalize_candidate_labeling(normalized.record, FIXED_MODE)
    return VerifiedSource(
        identity,
        normalized.record,
        "generic-search",
        None,
        None,
    )


def _make_source_from_record(record: CandidateMomentumRecord) -> VerifiedSource:
    """Build the full source boundary from an already evaluated record."""

    identity = canonicalize_candidate_labeling(record, FIXED_MODE)
    return VerifiedSource(identity, record, "generic-search", None, None)


def _record_from_evaluation(
    topology: Topology, labeling: Labeling, evaluation: Any,
) -> CandidateMomentumRecord:
    """Reuse one momentum evaluation without dropping assignment bindings."""

    if getattr(evaluation, "labeling", None) != labeling:
        raise ValueError("evaluation labeling does not match the generated assignment")
    record = candidate_record_from_evaluation(topology, evaluation)
    if record.topology != topology:
        raise ValueError("candidate record topology does not match the search topology")
    if record.complete_labeling.final_particles != labeling.final_particles:
        raise ValueError("candidate record final slots do not match the generated assignment")
    if record.complete_labeling.single_lines != labeling.single_lines:
        raise ValueError("candidate record single-line assignment does not match the generated assignment")
    expected_double_refs = tuple(
        (edge_index, occurrence)
        for edge_index, edge in enumerate(topology.edges)
        if edge.edge_type is EdgeType.DOUBLE and edge.multiplicity > 0
        for occurrence in range(edge.multiplicity)
    )
    actual_double_refs = tuple(
        (label.edge_index, label.occurrence)
        for label in record.complete_labeling.double_lines
    )
    if actual_double_refs != expected_double_refs:
        raise ValueError("candidate record does not preserve complete double-line occurrence order")
    return record


def _rank_status_from_rows(
    record: CandidateMomentumRecord,
    table: Mapping[tuple[int, ...], tuple[Fraction, ...]],
) -> tuple[str, int, int]:
    rows: list[tuple[Fraction, ...]] = []
    for label in record.complete_labeling.double_lines:
        subset = tuple(sorted(label.particle_ids))
        try:
            rows.append(table[subset])
        except KeyError as exc:
            raise ValueError(f"unsupported double-line particle subset: {subset!r}") from exc
    dimension = len(rows)
    affine_rank = rank(rows, ncols=len(Z_ORDER))
    status = "pass" if dimension == affine_rank and dimension <= 8 else "reject"
    return status, dimension, affine_rank


def _optimized_rank_status(record: CandidateMomentumRecord) -> tuple[str, int, int]:
    """Return status, occurrence dimension, and exact nine-column rank."""

    return _rank_status_from_rows(record, _OPTIMIZED_SUBSET_Z_ROWS)


def _replay_rank_status(record: CandidateMomentumRecord) -> tuple[str, int, int]:
    """Use an independently constructed pair-invariant table for replay."""

    return _rank_status_from_rows(record, _REPLAY_PAIR_SUBSET_Z_ROWS)


def _record_digest(record: CandidateMomentumRecord) -> str:
    return _sha256(record.to_dict())


def _momentum_decision(evaluation: Any) -> str:
    """Classify a labeling result while rejecting internally inconsistent states.

    ``evaluate_labeling`` returns a field object rather than a constructor with
    invariants, so callers must not silently turn a damaged result into a
    momentum rejection.  A valid result needs a verified momentum certificate
    and no failure codes; an invalid result needs at least one failure code and
    an explicitly unverified certificate.  Missing or verified certificates on
    the rejection path are treated as internal errors rather than momentum
    rejections.
    """

    valid = getattr(evaluation, "valid", None)
    if type(valid) is not bool:
        raise ValueError("inconsistent labeling evaluation: valid must be boolean")
    failure_codes = getattr(evaluation, "failure_codes", None)
    if not isinstance(failure_codes, (tuple, list)):
        raise ValueError("inconsistent labeling evaluation: failure_codes must be a sequence")
    if any(not isinstance(code, str) or not code for code in failure_codes):
        raise ValueError("inconsistent labeling evaluation: failure_codes must contain non-empty strings")
    if valid:
        if failure_codes:
            raise ValueError("inconsistent labeling evaluation: valid result has failure codes")
        momentum = getattr(evaluation, "momentum", None)
        if momentum is None or type(getattr(momentum, "verified", None)) is not bool or not momentum.verified:
            raise ValueError("inconsistent labeling evaluation: valid result lacks a verified momentum certificate")
        return "pass"
    if not failure_codes:
        raise ValueError("inconsistent labeling evaluation: invalid result has no failure codes")
    momentum = getattr(evaluation, "momentum", None)
    if momentum is None or type(getattr(momentum, "verified", None)) is not bool or momentum.verified:
        raise ValueError("inconsistent labeling evaluation: rejected result lacks an unverified momentum certificate")
    return "reject"


def _event(
    assignment: Sequence[int],
    *,
    momentum: str,
    failures: Sequence[str] = (),
    generic_status: str | None = None,
    record_digest: str | None = None,
) -> dict[str, object]:
    return {
        "assignment": list(assignment),
        "momentum": momentum,
        "failures": list(failures),
        "generic_status": generic_status,
        "record_digest": record_digest,
    }


def _stream_update(stream: hashlib._Hash, event: Mapping[str, object]) -> None:
    stream.update(_json_bytes(event))
    stream.update(b"\n")


def _stream_digest(events: Iterable[Mapping[str, object]]) -> str:
    stream = hashlib.sha256()
    for event in events:
        _stream_update(stream, event)
    return stream.hexdigest()


def _base_result(
    topology: Topology,
    *,
    expected: int,
    slots: Sequence[tuple[int, int]],
    single_refs: Sequence[tuple[int, int]],
    attempted: int,
    momentum_pass: int,
    momentum_reject: int,
    independent_reject: int,
    status: str,
    complete: bool,
    reason: str,
    stream_sha256: str,
    witness: Mapping[str, object] | None = None,
    errors: Sequence[Mapping[str, object]] = (),
    max_seconds: float | int | None = None,
    max_assignments: int | None = None,
) -> dict[str, object]:
    binding = _topology_binding(topology)
    return {
        "schema_version": GENERIC_SEARCH_SCHEMA_VERSION,
        "coverage_theorem_version": GENERIC_SEARCH_THEOREM_VERSION,
        "coverage_rule": GENERIC_SEARCH_COVERAGE_RULE,
        "coverage_scope": GENERIC_SEARCH_SCOPE,
        "fixed_identity_catalog_saved": False,
        **binding,
        "final_slot_order": [
            {"vertex": vertex, "slot": slot} for vertex, slot in slots
        ],
        "fixed_final_labeling": [
            {"vertex": vertex, "slot": slot, "particle_id": particle}
            for (vertex, slot), particle in zip(slots, range(1, 6))
        ],
        "single_occurrence_order": [
            {"edge_index": edge_index, "occurrence": occurrence}
            for edge_index, occurrence in single_refs
        ],
        "assignment_order": "product(range(1,6), repeat=L)",
        "L": len(single_refs),
        "expected": expected,
        "attempted": attempted,
        "momentum_pass": momentum_pass,
        "momentum_reject": momentum_reject,
        "independent_reject": independent_reject,
        "status": status,
        "complete": complete,
        "reason": reason,
        "stream_sha256": stream_sha256,
        "witness": dict(witness) if witness is not None else None,
        "errors": [dict(error) for error in errors],
        "budget": {
            "max_seconds": max_seconds,
            "max_assignments": max_assignments,
        },
    }


def _unknown(
    topology: Topology,
    *,
    reason: str,
    expected: int = 0,
    slots: Sequence[tuple[int, int]] = (),
    single_refs: Sequence[tuple[int, int]] = (),
    attempted: int = 0,
    momentum_pass: int = 0,
    momentum_reject: int = 0,
    independent_reject: int = 0,
    events: Sequence[Mapping[str, object]] = (),
    errors: Sequence[Mapping[str, object]] = (),
    max_seconds: float | int | None = None,
    max_assignments: int | None = None,
) -> dict[str, object]:
    return _base_result(
        topology,
        expected=expected,
        slots=slots,
        single_refs=single_refs,
        attempted=attempted,
        momentum_pass=momentum_pass,
        momentum_reject=momentum_reject,
        independent_reject=independent_reject,
        status="unknown",
        complete=False,
        reason=reason,
        stream_sha256=_stream_digest(events),
        errors=errors,
        max_seconds=max_seconds,
        max_assignments=max_assignments,
    )


def _validate_budget(max_seconds: object, max_assignments: object) -> tuple[float | int, int | None]:
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)):
        raise TypeError("max_seconds must be a finite non-negative number")
    if not math.isfinite(float(max_seconds)) or max_seconds < 0:
        raise ValueError("max_seconds must be a finite non-negative number")
    if max_assignments is not None and (not _strict_int(max_assignments) or max_assignments < 0):
        raise TypeError("max_assignments must be a non-negative integer or None")
    return max_seconds, max_assignments


def _validate_backend(backend: object) -> str:
    if not isinstance(backend, str) or backend not in GENERIC_SEARCH_BACKENDS:
        raise ValueError(f"backend must be one of {GENERIC_SEARCH_BACKENDS!r}")
    return backend


def _validate_topology(topology: object) -> tuple[Topology, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], int]:
    if not isinstance(topology, Topology):
        raise TypeError("topology must be a Topology")
    report = topology.validate_fields()
    if not report.valid:
        raise ValueError("topology fields are invalid")
    structure = analyze_topology(topology)
    if not structure.valid:
        raise ValueError("topology structure is invalid")
    if topology.n != 5:
        raise ValueError("generic five-body search requires n=5")
    slots = _final_slots(topology)
    if len(slots) != 5:
        raise ValueError("generic five-body search requires exactly five final slots")
    single_refs = _single_occurrences(topology)
    return topology, slots, single_refs, 5 ** len(single_refs)


def search_generic_topology(
    topology: Topology,
    *,
    max_seconds: float,
    max_assignments: int | None = None,
    backend: str = "reference",
) -> dict[str, object]:
    """Search one topology through its fixed-final ``5**L`` space.

    ``reference`` preserves the original full producer/verifier path.  The
    bounded ``optimized`` path uses the exact precomputed subset-to-Z table to
    reject rank-deficient momentum candidates early; a potential pass still
    crosses the complete source/evidence/verifier boundary.
    """

    try:
        max_seconds, max_assignments = _validate_budget(max_seconds, max_assignments)
        backend = _validate_backend(backend)
        topology, slots, single_refs, expected = _validate_topology(topology)
    except Exception as exc:
        if isinstance(topology, Topology):
            try:
                binding_slots = _final_slots(topology)
                binding_refs = _single_occurrences(topology)
                expected = 5 ** len(binding_refs)
            except Exception:
                binding_slots, binding_refs, expected = (), (), 0
            return _unknown(
                topology,
                reason="invalid_input",
                expected=expected,
                slots=binding_slots,
                single_refs=binding_refs,
                errors=({"type": type(exc).__name__, "message": str(exc)},),
                max_seconds=max_seconds if "max_seconds" in locals() and isinstance(max_seconds, (int, float)) else None,
                max_assignments=max_assignments if "max_assignments" in locals() and (_strict_int(max_assignments) or max_assignments is None) else None,
            )
        raise

    events: list[dict[str, object]] = []
    started = time.monotonic()
    attempted = momentum_pass = momentum_reject = independent_reject = 0
    deadline = started + float(max_seconds)
    assignments = itertools.product(range(1, 6), repeat=len(single_refs))

    for assignment in assignments:
        if attempted >= (max_assignments if max_assignments is not None else expected):
            return _unknown(
                topology,
                reason="assignment_budget_exhausted",
                expected=expected,
                slots=slots,
                single_refs=single_refs,
                attempted=attempted,
                momentum_pass=momentum_pass,
                momentum_reject=momentum_reject,
                independent_reject=independent_reject,
                events=events,
                max_seconds=max_seconds,
                max_assignments=max_assignments,
            )
        if time.monotonic() >= deadline:
            return _unknown(
                topology,
                reason="time_budget_exhausted",
                expected=expected,
                slots=slots,
                single_refs=single_refs,
                attempted=attempted,
                momentum_pass=momentum_pass,
                momentum_reject=momentum_reject,
                independent_reject=independent_reject,
                events=events,
                max_seconds=max_seconds,
                max_assignments=max_assignments,
            )
        assignment = tuple(assignment)
        attempted += 1
        try:
            labeling = _fixed_labeling(topology, slots, assignment)
            evaluation = evaluate_labeling(topology, labeling)
            momentum_decision = _momentum_decision(evaluation)
            if momentum_decision == "reject":
                momentum_reject += 1
                events.append(_event(assignment, momentum="reject", failures=evaluation.failure_codes))
                continue
            record: CandidateMomentumRecord
            fast_status: str | None = None
            if backend == "optimized":
                record = _record_from_evaluation(topology, labeling, evaluation)
                fast_status, _, _ = _optimized_rank_status(record)
                if fast_status == "reject":
                    momentum_pass += 1
                    independent_reject += 1
                    events.append(
                        _event(
                            assignment,
                            momentum="pass",
                            generic_status="reject",
                            record_digest=_record_digest(record),
                        )
                    )
                    continue
                source = _make_source_from_record(record)
            else:
                source = _make_source(topology, labeling)
            evidence = evaluate_generic_label(source)
            verifier = verify_generic_label(source, evidence)
            if not verifier.get("valid") or verifier.get("status") != evidence.get("status"):
                raise ValueError("generic evidence failed its independent verifier")
            momentum_pass += 1
            generic_status = str(evidence.get("status"))
            if fast_status is not None and generic_status != fast_status:
                raise ValueError("optimized fast/full status conflict")
            record = source.source_record
            events.append(
                _event(
                    assignment,
                    momentum="pass",
                    generic_status=generic_status,
                    record_digest=_record_digest(record),
                )
            )
            if generic_status == "pass":
                witness = {
                    "assignment": list(assignment),
                    "record": record.to_dict(),
                    "evidence": evidence,
                    "verifier": verifier,
                }
                return _base_result(
                    topology,
                    expected=expected,
                    slots=slots,
                    single_refs=single_refs,
                    attempted=attempted,
                    momentum_pass=momentum_pass,
                    momentum_reject=momentum_reject,
                    independent_reject=independent_reject,
                    status="pass",
                    complete=attempted == expected,
                    reason="generic_witness_found",
                    stream_sha256=_stream_digest(events),
                    witness=witness,
                    max_seconds=max_seconds,
                    max_assignments=max_assignments,
                )
            if generic_status != "reject":
                raise ValueError("generic verifier returned unsupported status")
            independent_reject += 1
        except Exception as exc:
            return _unknown(
                topology,
                reason="evaluation_error",
                expected=expected,
                slots=slots,
                single_refs=single_refs,
                attempted=attempted,
                momentum_pass=momentum_pass,
                momentum_reject=momentum_reject,
                independent_reject=independent_reject,
                events=events,
                errors=({"type": type(exc).__name__, "message": str(exc)},),
                max_seconds=max_seconds,
                max_assignments=max_assignments,
            )

    return _base_result(
        topology,
        expected=expected,
        slots=slots,
        single_refs=single_refs,
        attempted=attempted,
        momentum_pass=momentum_pass,
        momentum_reject=momentum_reject,
        independent_reject=independent_reject,
        status="reject",
        complete=True,
        reason="all_momentum_candidates_generic_reject",
        stream_sha256=_stream_digest(events),
        max_seconds=max_seconds,
        max_assignments=max_assignments,
    )


def _replay_search(
    topology: Topology,
    *,
    deadline: float | None = None,
    max_assignments: int | None = None,
    backend: str = "reference",
) -> tuple[dict[str, object], list[dict[str, object]], dict[tuple[int, ...], CandidateMomentumRecord]]:
    backend = _validate_backend(backend)
    _, slots, single_refs, expected = _validate_topology(topology)
    events: list[dict[str, object]] = []
    records: dict[tuple[int, ...], CandidateMomentumRecord] = {}
    attempted = momentum_pass = momentum_reject = independent_reject = 0
    for raw_assignment in itertools.product(range(1, 6), repeat=len(single_refs)):
        if max_assignments is not None and attempted >= max_assignments:
            break
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("verification time budget exhausted")
        assignment = tuple(raw_assignment)
        attempted += 1
        labeling = _fixed_labeling(topology, slots, assignment)
        evaluation = evaluate_labeling(topology, labeling)
        momentum_decision = _momentum_decision(evaluation)
        if momentum_decision == "reject":
            momentum_reject += 1
            events.append(_event(assignment, momentum="reject", failures=evaluation.failure_codes))
            continue
        record: CandidateMomentumRecord
        fast_status: str | None = None
        if backend == "optimized":
            record = _record_from_evaluation(topology, labeling, evaluation)
            fast_status, _, _ = _replay_rank_status(record)
            if fast_status == "pass":
                source = _make_source_from_record(record)
                evidence = evaluate_generic_label(source)
                verifier = verify_generic_label(source, evidence)
                if not verifier.get("valid") or verifier.get("status") != evidence.get("status"):
                    raise ValueError("generic evidence failed its independent verifier")
                generic_status = str(evidence.get("status"))
                if generic_status != fast_status:
                    raise ValueError("optimized fast/full status conflict")
            else:
                generic_status = fast_status
        else:
            source = _make_source(topology, labeling)
            record = source.source_record
            evidence = evaluate_generic_label(source)
            verifier = verify_generic_label(source, evidence)
            if not verifier.get("valid") or verifier.get("status") != evidence.get("status"):
                raise ValueError("generic evidence failed its independent verifier")
            generic_status = str(evidence.get("status"))
        momentum_pass += 1
        status = generic_status
        records[assignment] = record
        events.append(
            _event(
                assignment,
                momentum="pass",
                generic_status=status,
                record_digest=_record_digest(record),
            )
        )
        if status == "reject":
            independent_reject += 1
    stats = {
        "expected": expected,
        "attempted": attempted,
        "momentum_pass": momentum_pass,
        "momentum_reject": momentum_reject,
        "independent_reject": independent_reject,
        "complete": attempted == expected,
        "slots": slots,
        "single_refs": single_refs,
    }
    return stats, events, records


def _invalid_verification(errors: Sequence[str], *, checks: Sequence[str] = ()) -> dict[str, object]:
    return {"valid": False, "status": "unknown", "checks": list(checks), "errors": list(errors)}


def verify_generic_search(
    topology: Topology,
    result: Mapping[str, object],
    *,
    max_seconds: float | None = None,
    backend: str = "reference",
) -> dict[str, object]:
    """Recheck one result with a full reject replay or a pass prefix replay.

    A reject claims universal coverage and therefore replays all ``5**L``
    assignments.  A pass only claims the serialized witness and its producer
    prefix, so verification replays through the reported attempt and checks
    that the witness is the first generic pass at the end of that prefix.
    """

    if not isinstance(result, Mapping):
        return _invalid_verification(["result must be a mapping"])
    try:
        backend = _validate_backend(backend)
        if max_seconds is not None:
            max_seconds, _ = _validate_budget(max_seconds, None)
            deadline = time.monotonic() + float(max_seconds)
        else:
            deadline = None
        topology, slots, single_refs, expected = _validate_topology(topology)
        raw_status = result.get("status")
        if raw_status not in {"pass", "reject", "unknown"}:
            return _invalid_verification(["unsupported_status"])
        if raw_status == "unknown":
            return _invalid_verification(["unknown_search_result_is_not_a_certificate"])
        pass_limit: int | None = None
        if raw_status == "pass":
            if not _strict_int(result.get("attempted")) or not 1 <= result["attempted"] <= expected:
                return _invalid_verification(["invalid_pass_attempted"])
            pass_limit = result["attempted"]
        stats, events, records = _replay_search(
            topology,
            deadline=deadline,
            max_assignments=pass_limit,
            backend=backend,
        )
    except TimeoutError:
        return _invalid_verification(["verification_time_budget_exhausted"])
    except Exception as exc:
        return _invalid_verification([f"verification_error:{type(exc).__name__}:{exc}"])

    required = {
        "schema_version", "coverage_theorem_version", "coverage_rule", "coverage_scope",
        "fixed_identity_catalog_saved", "topology", "topology_digest",
        "final_slot_order", "fixed_final_labeling", "single_occurrence_order",
        "assignment_order", "L", "expected", "attempted", "momentum_pass",
        "momentum_reject", "independent_reject", "status", "complete", "reason",
        "stream_sha256", "witness", "errors", "budget",
    }
    missing = sorted(required - set(result))
    if missing:
        return _invalid_verification([f"missing_fields:{','.join(missing)}"])
    checks: list[str] = ["topology_replayed"]
    checks.append(
        "full_assignment_space_replayed"
        if stats["complete"]
        else "assignment_prefix_replayed"
    )
    checks.append("stream_replayed")
    expected_result = _base_result(
        topology,
        expected=expected,
        slots=slots,
        single_refs=single_refs,
        attempted=stats["attempted"],
        momentum_pass=stats["momentum_pass"],
        momentum_reject=stats["momentum_reject"],
        independent_reject=stats["independent_reject"],
        status="reject",
        complete=True,
        reason="all_momentum_candidates_generic_reject",
        stream_sha256=_stream_digest(events),
    )

    for key in ("schema_version", "coverage_theorem_version", "coverage_rule", "coverage_scope", "assignment_order"):
        if not _strict_json_equal(result.get(key), expected_result[key]):
            return _invalid_verification([f"binding_mismatch:{key}"], checks=checks)
    if result.get("fixed_identity_catalog_saved") is not False:
        return _invalid_verification(["fixed_identity_catalog_saved_must_be_false"], checks=checks)
    if not _strict_json_equal(result.get("topology"), expected_result["topology"]):
        return _invalid_verification(["topology_mismatch"], checks=checks)
    if not _strict_json_equal(result.get("topology_digest"), expected_result["topology_digest"]):
        return _invalid_verification(["topology_digest_mismatch"], checks=checks)
    for key in ("final_slot_order", "fixed_final_labeling", "single_occurrence_order", "L", "expected"):
        if not _strict_json_equal(result.get(key), expected_result[key]):
            return _invalid_verification([f"binding_mismatch:{key}"], checks=checks)

    if raw_status == "reject":
        for key in ("attempted", "momentum_pass", "momentum_reject", "independent_reject", "complete", "stream_sha256"):
            if not _strict_json_equal(result.get(key), expected_result[key]):
                return _invalid_verification([f"replay_mismatch:{key}"], checks=checks)
        if result.get("witness") is not None:
            return _invalid_verification(["reject_must_not_have_witness"], checks=checks)
        if stats["independent_reject"] != stats["momentum_pass"]:
            return _invalid_verification(["replay_contains_generic_pass"], checks=checks)
        checks.append("reject_full_cover_verified")
        return {"valid": True, "status": "reject", "checks": checks, "errors": []}

    # A pass may stop at the first generic witness.  Recompute its exact
    # prefix and then independently verify the serialized witness source.
    attempted = result.get("attempted")
    if not _strict_int(attempted) or not 1 <= attempted <= expected:
        return _invalid_verification(["invalid_pass_attempted"], checks=checks)
    prefix = events[:attempted]
    prefix_stats = {
        "momentum_pass": sum(event["momentum"] == "pass" for event in prefix),
        "momentum_reject": sum(event["momentum"] == "reject" for event in prefix),
    }
    first_pass = next((index for index, event in enumerate(events) if event["generic_status"] == "pass"), None)
    if first_pass is None or first_pass != attempted - 1:
        return _invalid_verification(["pass_witness_not_last_prefix_event"], checks=checks)
    if not _strict_json_equal(result.get("stream_sha256"), _stream_digest(prefix)):
        return _invalid_verification(["pass_prefix_stream_mismatch"], checks=checks)
    for key, value in (
        ("momentum_pass", prefix_stats["momentum_pass"]),
        ("momentum_reject", prefix_stats["momentum_reject"]),
        ("independent_reject", sum(event["generic_status"] == "reject" for event in prefix)),
        ("complete", attempted == expected),
    ):
        if not _strict_json_equal(result.get(key), value):
            return _invalid_verification([f"pass_prefix_mismatch:{key}"], checks=checks)
    witness = result.get("witness")
    if not isinstance(witness, Mapping):
        return _invalid_verification(["pass_witness_required"], checks=checks)
    if not _strict_int(witness.get("assignment")):
        # Assignment itself is a list; this branch only catches scalar forgery.
        if not isinstance(witness.get("assignment"), list):
            return _invalid_verification(["invalid_witness_assignment"], checks=checks)
    assignment = witness.get("assignment")
    if (
        not isinstance(assignment, list)
        or len(assignment) != len(single_refs)
        or any(not _strict_int(item) or not 1 <= item <= 5 for item in assignment)
        or tuple(assignment) not in records
    ):
        return _invalid_verification(["witness_assignment_not_replayed"], checks=checks)
    record_data = witness.get("record")
    evidence = witness.get("evidence")
    stored_verifier = witness.get("verifier")
    if not isinstance(record_data, Mapping) or not isinstance(evidence, Mapping) or not isinstance(stored_verifier, Mapping):
        return _invalid_verification(["incomplete_witness"], checks=checks)
    try:
        record = CandidateMomentumRecord.from_dict(record_data)
        if not _strict_json_equal(record.topology.to_dict(), topology.to_dict()):
            return _invalid_verification(["witness_topology_mismatch"], checks=checks)
        if not _strict_json_equal(record.to_dict(), records[tuple(assignment)].to_dict()):
            return _invalid_verification(["witness_record_mismatch"], checks=checks)
        identity = canonicalize_candidate_labeling(record, FIXED_MODE)
        source = VerifiedSource(identity, record, "generic-search", None, None)
        verifier = verify_generic_label(source, evidence)
    except Exception as exc:
        return _invalid_verification([f"witness_reconstruction_error:{type(exc).__name__}:{exc}"], checks=checks)
    if not verifier.get("valid") or verifier.get("status") != "pass":
        return _invalid_verification(["witness_generic_verification_failed"], checks=checks)
    if not _strict_json_equal(stored_verifier, verifier):
        return _invalid_verification(["witness_verifier_mismatch"], checks=checks)
    checks.extend(["pass_prefix_verified", "witness_topology_verified", "witness_generic_certificate_verified"])
    return {"valid": True, "status": "pass", "checks": checks, "errors": []}


__all__ = [
    "GENERIC_SEARCH_SCHEMA_VERSION",
    "GENERIC_SEARCH_THEOREM_VERSION",
    "GENERIC_SEARCH_COVERAGE_RULE",
    "GENERIC_SEARCH_SCOPE",
    "search_generic_topology",
    "verify_generic_search",
]
