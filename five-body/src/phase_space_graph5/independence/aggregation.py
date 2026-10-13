"""Conservative topology aggregation for independently verified labels.

The existing S/P source archive and the new independence certificates have
different coverage meanings.  A reliable PASS may come from a partial source;
REJECT requires a complete fixed-label identity key set and a separately
verified reject for every key in that set.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Iterable, Mapping

from ..physical_labels import review_candidate_record
from .certificates import EvidenceCertificate, VerificationReport, parse_certificate
from .contracts import EvaluationRequest, LabelEvaluation, PhysicsProfile, TopologyEvaluation
from .sources import FIXED_MODE, SourceBatch, VerifiedSource, _jsonable, _key_token


class AggregationError(ValueError):
    """A source/evaluation collection has incompatible fixed contracts."""


Verifier = Callable[[VerifiedSource, EvaluationRequest, LabelEvaluation], VerificationReport]


@dataclass(frozen=True, slots=True)
class VerifiedLabelResult:
    """One compact evaluation plus its separately verifier-owned report."""

    evaluation: LabelEvaluation
    certificate: EvidenceCertificate | None
    verification: VerificationReport | None
    verified: bool = False
    reason: str | None = None


def _canonical(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_evaluation(value: LabelEvaluation | Mapping[str, Any]) -> LabelEvaluation:
    if isinstance(value, LabelEvaluation):
        return value
    if isinstance(value, Mapping):
        return LabelEvaluation.from_dict(value)
    raise TypeError("evaluations must contain LabelEvaluation or serialized mappings")


def _source_maps(batch: SourceBatch) -> dict[str, VerifiedSource]:
    result: dict[str, VerifiedSource] = {}
    for item in batch.records:
        token = _key_token(item.key)
        if token in result:
            raise AggregationError("source batch contains duplicate complete identity keys")
        result[token] = item
    return result


def _request_maps(requests: Iterable[EvaluationRequest], source: Mapping[str, VerifiedSource]) -> tuple[
    PhysicsProfile | None, dict[str, EvaluationRequest], list[str]
]:
    request_by_key: dict[str, EvaluationRequest] = {}
    profile: PhysicsProfile | None = None
    reasons: list[str] = []
    for request in requests:
        if not isinstance(request, EvaluationRequest):
            reasons.append("request_is_not_an_EvaluationRequest")
            continue
        if profile is None:
            profile = request.profile
        elif _canonical(profile.to_dict()) != _canonical(request.profile.to_dict()):
            reasons.append("requests_have_conflicting_profiles")
        key = request.candidate.identity_key
        if key in request_by_key:
            reasons.append("duplicate_request_identity_key")
            continue
        request_by_key[key] = request
        item = source.get(key)
        if item is None:
            reasons.append("request_identity_key_missing_from_source")
            continue
        # Compact ReviewedCandidate fields are only an index.  Full record and
        # identity equality is checked later against certificate evidence.
        if request.candidate.mode != FIXED_MODE:
            reasons.append("request_mode_is_not_fixed_labels")
        if request.candidate.topology_key != _key_token(item.topology_key):
            reasons.append("request_topology_key_disagrees_with_source")
        if request.candidate.identity_key != _key_token(item.key):
            reasons.append("request_identity_key_disagrees_with_source")
    return profile, request_by_key, reasons


def _certificate_matches_source(certificate: EvidenceCertificate,
                                item: VerifiedSource,
                                request: EvaluationRequest,
                                evaluation: LabelEvaluation) -> tuple[bool, str | None]:
    if certificate.outcome != evaluation.status:
        return False, "certificate_outcome_disagrees_with_evaluation"
    if _canonical(certificate.profile) != _canonical(request.profile.to_dict()):
        return False, "certificate_profile_disagrees_with_request"
    if _canonical(certificate.request) != _canonical(request.to_dict()):
        return False, "certificate_request_disagrees_with_request"
    if _canonical(certificate.source_record) != _canonical(item.source_record.to_dict()):
        return False, "certificate_source_record_disagrees_with_source"
    if _canonical(certificate.identity) != _canonical(item.identity.to_dict()):
        return False, "certificate_identity_disagrees_with_source"
    if evaluation.profile.to_dict() != request.profile.to_dict():
        return False, "evaluation_profile_disagrees_with_request"
    if evaluation.mode != FIXED_MODE:
        return False, "evaluation_mode_is_not_fixed_labels"
    if evaluation.topology_key != request.candidate.topology_key:
        return False, "evaluation_topology_key_disagrees_with_request"
    if evaluation.identity_key != request.candidate.identity_key:
        return False, "evaluation_identity_key_disagrees_with_request"
    if tuple(evaluation.occurrences) != tuple(request.candidate.occurrences):
        return False, "evaluation_occurrences_disagree_with_request"
    review_candidate_record(item.source_record)
    return True, None


def _verified_result(item: VerifiedSource, request: EvaluationRequest,
                    evaluation: LabelEvaluation,
                    verifier: Verifier) -> VerifiedLabelResult:
    certificate: EvidenceCertificate | None = None
    if evaluation.certificate is not None:
        try:
            certificate = parse_certificate(evaluation.certificate)
        except (TypeError, ValueError, KeyError):
            return VerifiedLabelResult(evaluation, None, None, reason="malformed_certificate")
    if certificate is None:
        return VerifiedLabelResult(evaluation, None, None, reason="missing_certificate")
    try:
        report = verifier(item, request, evaluation)
    except Exception:
        return VerifiedLabelResult(evaluation, certificate, None, reason="verifier_failed")
    if not isinstance(report, VerificationReport):
        return VerifiedLabelResult(evaluation, certificate, None, reason="verifier_returned_invalid_report")
    try:
        valid, reason = _certificate_matches_source(certificate, item, request, evaluation)
    except (TypeError, ValueError, KeyError):
        valid, reason = False, "certificate_recheck_failed"
    if not valid:
        return VerifiedLabelResult(evaluation, certificate, None, reason=reason)
    if report is None or not report.accepted or report.status != evaluation.status or report.errors:
        return VerifiedLabelResult(evaluation, certificate, None, reason="verification_report_not_accepted")
    if certificate.execution_status != "complete" or certificate.input_status != "valid":
        return VerifiedLabelResult(evaluation, certificate, None, reason="certificate_execution_not_complete")
    return VerifiedLabelResult(evaluation, certificate, report, verified=True)


def aggregate_topology_evaluations(
    source: SourceBatch,
    requests: EvaluationRequest | Iterable[EvaluationRequest],
    evaluations: Iterable[LabelEvaluation | Mapping[str, Any]],
    *,
    topology_key: Any | None = None,
) -> TopologyEvaluation:
    """Aggregate independently verified labels under the fixed-profile rules.

    The independent verifier is called for every candidate.  Serialized
    producer flags or injected verification report objects are never used.
    """

    if not isinstance(source, SourceBatch):
        raise TypeError("source must be a SourceBatch")
    try:
        from .verification import verify_label_evidence
    except (ImportError, AttributeError) as exc:
        raise AggregationError("the independent evidence verifier is unavailable") from exc
    verifier = verify_label_evidence
    disk_coverage_verified = False
    try:
        from .sources import read_source, verify_source_unchanged
        verify_source_unchanged(source)
    except Exception as exc:
        source_binding_ok = False
        source_binding_reason = f"source_changed_or_unreadable:{type(exc).__name__}"
    else:
        source_binding_ok = True
        source_binding_reason = None
    if source_binding_ok and source.snapshot.source_path is not None:
        try:
            reread = read_source(source.snapshot.source_path)
            if (
                reread.snapshot.fingerprint != source.snapshot.fingerprint
                or tuple(_canonical(item.to_dict()) for item in reread.records)
                != tuple(_canonical(item.to_dict()) for item in source.records)
                or reread.coverage.to_dict() != source.coverage.to_dict()
            ):
                source_binding_ok = False
                source_binding_reason = "source_reread_does_not_match_snapshot"
        except Exception as exc:
            source_binding_ok = False
            source_binding_reason = f"source_reread_failed:{type(exc).__name__}"
        else:
            disk_coverage_verified = True
    source_by_key = _source_maps(source)
    if isinstance(requests, EvaluationRequest):
        request_values = (requests,)
    else:
        request_values = tuple(requests)
    profile, request_by_key, reasons = _request_maps(request_values, source_by_key)
    if not source_binding_ok and source_binding_reason is not None:
        reasons.insert(0, source_binding_reason)
    if "requests_have_conflicting_profiles" in reasons:
        raise AggregationError("requests with different profiles cannot be aggregated")
    if profile is None:
        raise AggregationError("at least one valid EvaluationRequest is required")

    evaluations_by_key: dict[str, LabelEvaluation] = {}
    for raw in evaluations:
        try:
            evaluation = _as_evaluation(raw)
        except (TypeError, ValueError, KeyError):
            reasons.append("malformed_label_evaluation")
            continue
        if evaluation.identity_key in evaluations_by_key:
            if _canonical(evaluations_by_key[evaluation.identity_key].to_dict()) != _canonical(evaluation.to_dict()):
                raise AggregationError("conflicting evaluations share one identity key")
            reasons.append("duplicate_label_evaluation_identity_key")
            continue
        evaluations_by_key[evaluation.identity_key] = evaluation
    for key in set(evaluations_by_key) - set(source_by_key):
        reasons.append(f"evaluation_identity_key_missing_from_source:{key}")

    topology_token = None if topology_key is None else (_key_token(topology_key) if not isinstance(topology_key, str) else topology_key)
    if topology_token is None:
        topology_values = {
            request.candidate.topology_key for request in request_by_key.values()
        }
        if len(topology_values) != 1:
            raise AggregationError("topology scope is ambiguous; provide topology_key explicitly")
        else:
            topology_token = next(iter(topology_values))

    expected_keys = {
        _key_token(item.key)
        for item in source.records
        if _key_token(item.topology_key) == topology_token
    }
    if source.coverage.expected_identity_keys is not None:
        expected_keys = {
            _key_token(key)
            for key in source.coverage.expected_identity_keys
            if any(
                _key_token(item.key) == _key_token(key)
                and _key_token(item.topology_key) == topology_token
                for item in source.records
            )
        }
    actual_keys = {
        key for key, item in source_by_key.items() if _key_token(item.topology_key) == topology_token
    }
    source_complete = bool(
        source_binding_ok
        and disk_coverage_verified
        and source.coverage.complete
        and expected_keys == actual_keys
    )
    evaluated: list[LabelEvaluation] = []
    valid_outcomes: dict[str, str] = {}
    evaluated_keys: set[str] = set()
    for key in sorted(actual_keys):
        item = source_by_key[key]
        request = request_by_key.get(key)
        evaluation = evaluations_by_key.get(key)
        if request is None or evaluation is None:
            reasons.append(f"missing_request_or_evaluation:{key}")
            continue
        evaluated.append(evaluation)
        result = _verified_result(item, request, evaluation, verifier)
        cert = result.certificate
        report_value = result.verification
        if (
            result.verified
            and cert is not None
            and report_value is not None
            and report_value.accepted
            and not report_value.errors
            and report_value.status == evaluation.status
            and cert.outcome == evaluation.status
        ):
            # _verified_result has already compared full source/identity
            # payloads; this branch records only genuinely usable evidence.
            try:
                valid, reason = _certificate_matches_source(cert, item, request, evaluation)
            except (TypeError, ValueError, KeyError):
                valid, reason = False, "certificate_recheck_failed"
            if valid:
                valid_outcomes[key] = evaluation.status
                evaluated_keys.add(key)
            elif reason:
                reasons.append(f"{reason or 'certificate_recheck_failed'}:{key}")
        else:
            reasons.append(f"{result.reason or 'certificate_not_verified'}:{key}")

    pass_keys = {key for key, value in valid_outcomes.items() if value == "pass"}
    reject_keys = {key for key, value in valid_outcomes.items() if value == "reject"}
    if not source_binding_ok:
        status = "unknown"
    elif pass_keys:
        status = "pass"
    elif source_complete and expected_keys and reject_keys == expected_keys:
        status = "reject"
    else:
        status = "unknown"

    if status == "unknown" and not reasons:
        reasons.append("no_verified_pass_and_reject_coverage_incomplete")
    source_coverage = source.coverage.to_dict()
    source_coverage.update({
        "topology_key": topology_token,
        "topology_actual_identity_keys": sorted(actual_keys),
        "topology_expected_identity_keys": sorted(expected_keys) if source_complete else None,
        "source_complete_for_topology": source_complete,
    })
    independence_coverage = {
        "evaluated_identity_keys": sorted(evaluated_keys),
        "verified_pass_keys": sorted(pass_keys),
        "verified_reject_keys": sorted(reject_keys),
        "missing_identity_keys": sorted(expected_keys - evaluated_keys),
        "unknown_identity_keys": sorted(
            (actual_keys - evaluated_keys)
            | {key for key, value in valid_outcomes.items() if value == "unknown"}
        ),
        "complete_reject_coverage": bool(source_complete and reject_keys == expected_keys),
    }
    return TopologyEvaluation(
        topology_key=topology_token,
        mode=FIXED_MODE,
        profile=profile,
        status=status,
        source_coverage=source_coverage,
        independence_coverage=independence_coverage,
        label_evaluations=tuple(evaluated),
        witness_identity_key=next(iter(sorted(pass_keys)), None),
        unknown_reasons=tuple(dict.fromkeys(reasons)),
    )


__all__ = [
    "AggregationError", "VerifiedLabelResult", "aggregate_topology_evaluations",
]
