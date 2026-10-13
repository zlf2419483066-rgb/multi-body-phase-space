"""Closed aggregation for the generic five-body independence evaluation.

The generic path intentionally has a smaller public surface than the first
fixed-profile implementation.  It accepts a reviewed :class:`SourceBatch`
and the dictionaries returned by ``generic_family.evaluate_generic_label``.
Every usable conclusion is rechecked through the real
``generic_verification.verify_generic_label`` function.  There is no verifier
callback argument: callers cannot replace the mathematical verifier with a
function that merely returns ``valid=True``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from typing import Any

from .sources import (
    SourceBatch,
    SourceChangedError,
    SourceError,
    VerifiedSource,
    _key_token,
    _source_digest,
    read_source,
    verify_source_unchanged,
)


GENERIC_AGGREGATION_SCHEMA = "F5-I-generic-aggregation-v1"
_STATUSES = frozenset({"pass", "reject", "unknown"})


class GenericAggregationError(ValueError):
    """The input batch/evaluation collection is not a valid aggregation."""


def _jsonable(value: Any) -> Any:
    """Make a detached JSON-shaped copy without changing integer/bool types."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    raise TypeError(f"value is not JSON-shaped: {type(value).__name__}")


def _digest_json(value: Any) -> str:
    raw = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _key_from_value(value: Any) -> str | None:
    """Read a key in either its complete JSON-token or structured form."""

    if isinstance(value, str):
        return value
    if value is None:
        return None
    return _key_token(value)


def _field(payload: Mapping[str, Any], name: str) -> Any:
    """Read a binding field from the strict evidence containers.

    Generic evidence keeps identity bindings under ``identity`` and source
    bindings under ``source``.  The top-level form remains accepted for the
    small foundation dictionaries and for the verifier's certificate wrapper;
    no recursive search is used because an unrelated nested value must not
    accidentally bind a label.
    """

    containers: list[Mapping[str, Any]] = [payload]
    certificate = payload.get("certificate")
    if isinstance(certificate, Mapping):
        containers.append(certificate)
    for container_name in ("identity", "source"):
        nested = payload.get(container_name)
        if isinstance(nested, Mapping):
            containers.append(nested)
        if isinstance(certificate, Mapping):
            nested_certificate = certificate.get(container_name)
            if isinstance(nested_certificate, Mapping):
                containers.append(nested_certificate)
    for container in containers:
        if name in container:
            return container[name]
    return None


def _status(payload: Mapping[str, Any]) -> str | None:
    value = _field(payload, "status")
    return value if isinstance(value, str) and value in _STATUSES else None


def _evaluation_payload(entry: Any) -> tuple[Mapping[str, Any] | None, str | None]:
    """Unwrap one evaluator row and return its explicit identity token."""

    if not isinstance(entry, Mapping):
        return None, None
    payload: Mapping[str, Any] = entry
    nested = entry.get("evidence")
    if isinstance(nested, Mapping):
        payload = nested
    identity = _key_from_value(_field(payload, "identity_key"))
    if identity is None and payload is not entry:
        identity = _key_from_value(_field(entry, "identity_key"))
    return payload, identity


def _source_row(item: VerifiedSource) -> dict[str, Any]:
    return {
        "identity_key": _key_token(item.key),
        "topology_key": _key_token(item.topology_key),
        "source_digest": _source_digest(item.source_record),
        "identity_digest": item.digest,
    }


def _batch_shape(batch: SourceBatch) -> tuple[dict[str, Any], ...]:
    return tuple(_source_row(item) for item in batch.records)


def _same_batch(left: SourceBatch, right: SourceBatch) -> bool:
    """Compare all reviewed source content and coverage, not only a digest."""

    return (
        tuple(item.to_dict() for item in left.records)
        == tuple(item.to_dict() for item in right.records)
        and left.coverage.to_dict() == right.coverage.to_dict()
        and left.snapshot.to_dict() == right.snapshot.to_dict()
    )


def _fresh_source_status(batch: SourceBatch) -> tuple[bool, bool, list[str], SourceBatch | None]:
    """Return (binding_ok, disk_coverage_verified, reasons, fresh_batch)."""

    reasons: list[str] = []
    try:
        verify_source_unchanged(batch)
    except (SourceChangedError, SourceError, OSError, TypeError, ValueError) as exc:
        return False, False, [f"source_unchanged_check_failed:{exc}"], None

    source_path = batch.snapshot.source_path
    if source_path is None:
        reasons.append("in_memory_source_has_no_disk_coverage_proof")
        return True, False, reasons, None

    try:
        fresh = read_source(source_path)
    except (SourceChangedError, SourceError, OSError, TypeError, ValueError) as exc:
        return False, False, [f"fresh_source_read_failed:{exc}"], None
    if not _same_batch(batch, fresh):
        return False, False, ["fresh_source_read_does_not_match_initial_read"], fresh
    if not fresh.coverage.complete:
        reasons.append("fresh_source_completion_contract_not_proven")
        return True, False, reasons, fresh
    return True, True, reasons, fresh


def _final_source_status(
    batch: SourceBatch, initial_fresh: SourceBatch | None
) -> tuple[bool, list[str]]:
    """Check the immutable source again after all label verifications."""

    try:
        verify_source_unchanged(batch)
        if batch.snapshot.source_path is None:
            return True, []
        final = read_source(batch.snapshot.source_path)
        if not _same_batch(batch, final):
            return False, ["source_changed_during_evaluation"]
        if initial_fresh is not None and not _same_batch(initial_fresh, final):
            return False, ["source_changed_between_fresh_reads"]
        return True, []
    except (SourceChangedError, SourceError, OSError, TypeError, ValueError) as exc:
        return False, [f"final_source_check_failed:{exc}"]


def _normalise_entries(evaluations: Any) -> tuple[Any, ...]:
    if isinstance(evaluations, Mapping):
        # A mapping from a complete identity key to an evidence dictionary is
        # accepted, but the dictionary's own identity_key remains mandatory.
        return tuple(evaluations.values())
    if isinstance(evaluations, Sequence) and not isinstance(evaluations, (str, bytes)):
        return tuple(evaluations)
    if isinstance(evaluations, Iterable) and not isinstance(evaluations, (str, bytes)):
        return tuple(evaluations)
    raise GenericAggregationError("evaluations must be a mapping or iterable of evidence dictionaries")


def _call_verifier(source: VerifiedSource, evidence: Mapping[str, Any]) -> dict[str, Any]:
    # Import lazily so source adapters and static tooling remain usable while
    # the foundation owner is updating the generic verifier module.
    from .generic_verification import verify_generic_label

    try:
        report = verify_generic_label(source, evidence)
    except Exception as exc:  # a malformed certificate is an UNKNOWN input
        return {
            "valid": False,
            "status": "unknown",
            "checks": {},
            "errors": [f"verifier_exception:{type(exc).__name__}:{exc}"],
        }
    if not isinstance(report, Mapping):
        return {
            "valid": False,
            "status": "unknown",
            "checks": {},
            "errors": ["verifier_returned_non_mapping"],
        }
    result = _jsonable(report)
    if not isinstance(result, dict):  # guarded by the Mapping check above
        raise GenericAggregationError("verifier result is not a JSON object")
    return result


def _report_usable(
    source: VerifiedSource,
    evidence: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[bool, str | None, list[str]]:
    errors: list[str] = []
    status = _status(evidence)
    report_status = report.get("status")
    if status is None:
        errors.append("evidence_status_missing_or_invalid")
    if report_status not in _STATUSES:
        errors.append("verifier_status_missing_or_invalid")
    if report.get("valid") is not True:
        errors.append("verifier_did_not_accept_evidence")
    if status is not None and report_status in _STATUSES and status != report_status:
        errors.append("evidence_and_verifier_status_disagree")

    expected = _source_row(source)
    for field in ("identity_key", "topology_key", "source_digest"):
        actual = _field(evidence, field)
        if field.endswith("key"):
            actual = _key_from_value(actual)
        if actual != expected[field]:
            errors.append(f"{field}_does_not_bind_source")

    # A verifier may expose its own binding checks.  A report with an explicit
    # false check is never allowed to produce a physical conclusion.
    checks = report.get("checks")
    if isinstance(checks, Mapping):
        false_checks = [str(key) for key, value in checks.items() if value is False]
        if false_checks:
            errors.append("verifier_checks_failed:" + ",".join(sorted(false_checks)))
    if errors:
        return False, None, errors
    return True, status, []


def _topology_status(
    rows: list[dict[str, Any]],
    *,
    source_complete: bool,
) -> str:
    valid_rows = [row for row in rows if row["valid"]]
    if any(row["status"] == "pass" for row in valid_rows):
        return "pass"
    expected_count = len(rows)
    rejects = [row for row in valid_rows if row["status"] == "reject"]
    if source_complete and expected_count > 0 and len(rejects) == expected_count:
        return "reject"
    return "unknown"


def aggregate_generic(batch: SourceBatch, evaluations: Any) -> dict[str, Any]:
    """Reverify and aggregate generic evidence for one reviewed source batch.

    A PASS can be contributed by a verified subset of a source.  A REJECT is
    emitted only for a disk-backed, freshly reread complete source where every
    identity key in the topology has a valid reject report.  Any malformed,
    missing, conflicting, or out-of-range evidence remains UNKNOWN.
    """

    if not isinstance(batch, SourceBatch):
        raise TypeError("batch must be a SourceBatch")
    source_rows = _batch_shape(batch)
    by_identity = {row["identity_key"]: (item, row) for item, row in zip(batch.records, source_rows)}
    if len(by_identity) != len(batch.records):
        raise GenericAggregationError("source batch contains duplicate complete identity keys")

    source_binding_ok, disk_coverage_verified, source_reasons, fresh = _fresh_source_status(batch)
    source_complete = bool(source_binding_ok and disk_coverage_verified and batch.coverage.complete)

    entries = _normalise_entries(evaluations)
    rows_by_identity: dict[str, dict[str, Any]] = {}
    duplicate_keys: set[str] = set()
    out_of_range: list[str] = []
    for entry in entries:
        evidence, identity_key = _evaluation_payload(entry)
        if evidence is None or identity_key is None:
            out_of_range.append("evidence_missing_complete_identity_key")
            continue
        if identity_key not in by_identity:
            out_of_range.append(identity_key)
            continue
        if identity_key in rows_by_identity:
            duplicate_keys.add(identity_key)
            continue
        item, expected = by_identity[identity_key]
        report = _call_verifier(item, evidence)
        usable, status, errors = _report_usable(item, evidence, report)
        rows_by_identity[identity_key] = {
            "identity_key": identity_key,
            "topology_key": expected["topology_key"],
            "source_digest": expected["source_digest"],
            "status": status if usable else "unknown",
            "valid": usable,
            "evidence": _jsonable(evidence),
            "verification": report,
            "errors": errors,
            "witness_identity_key": _jsonable(_field(evidence, "witness_identity_key")),
        }

    final_binding_ok, final_source_reasons = _final_source_status(batch, fresh)
    if not final_binding_ok:
        source_binding_ok = False
        disk_coverage_verified = False
        source_reasons.extend(final_source_reasons)
    source_complete = bool(source_binding_ok and disk_coverage_verified and batch.coverage.complete)

    if duplicate_keys:
        for identity_key in duplicate_keys:
            row = rows_by_identity.get(identity_key)
            if row is not None:
                row["valid"] = False
                row["status"] = "unknown"
                row["errors"].append("duplicate_evaluation_for_identity_key")

    # A changed or otherwise unbound source invalidates every physical
    # conclusion, including a PASS that happened to verify algebraically.
    # Keep the evidence and verifier report for audit, while exposing only
    # UNKNOWN as the aggregate status.
    if not source_binding_ok:
        for row in rows_by_identity.values():
            row["reported_status"] = row["status"]
            row["valid"] = False
            row["status"] = "unknown"
            row["errors"].append("source_binding_not_verified")

    rows_by_topology: dict[str, list[dict[str, Any]]] = {}
    for item, expected in zip(batch.records, source_rows):
        row = rows_by_identity.get(expected["identity_key"])
        if row is None:
            row = {
                "identity_key": expected["identity_key"],
                "topology_key": expected["topology_key"],
                "source_digest": expected["source_digest"],
                "status": "unknown",
                "valid": False,
                "evidence": None,
                "verification": None,
                "errors": ["missing_evaluation_for_identity_key"],
                "witness_identity_key": None,
            }
        rows_by_topology.setdefault(expected["topology_key"], []).append(row)

    topology_results: list[dict[str, Any]] = []
    for topology_key in sorted(rows_by_topology):
        rows = rows_by_topology[topology_key]
        status = _topology_status(rows, source_complete=source_complete)
        topology_results.append(
            {
                "topology_key": topology_key,
                "status": status,
                "identity_keys": [row["identity_key"] for row in rows],
                "pass_identity_keys": [row["identity_key"] for row in rows if row["valid"] and row["status"] == "pass"],
                "reject_identity_keys": [row["identity_key"] for row in rows if row["valid"] and row["status"] == "reject"],
                "unknown_identity_keys": [row["identity_key"] for row in rows if not row["valid"] or row["status"] == "unknown"],
                "witness_identity_keys": [
                    row["witness_identity_key"] or row["identity_key"]
                    for row in rows
                    if row["valid"] and row["status"] == "pass"
                ],
                "evaluations": rows,
            }
        )

    topology_statuses = [item["status"] for item in topology_results]
    if not source_binding_ok:
        overall_status = "unknown"
    elif "pass" in topology_statuses:
        overall_status = "pass"
    elif topology_statuses and source_complete and all(status == "reject" for status in topology_statuses):
        overall_status = "reject"
    else:
        overall_status = "unknown"

    evaluated = (
        not out_of_range
        and not duplicate_keys
        and len(rows_by_identity) == len(batch.records)
        and all(row["valid"] for row in rows_by_identity.values())
    )
    return {
        "schema_version": GENERIC_AGGREGATION_SCHEMA,
        "status": overall_status,
        "source_binding_ok": source_binding_ok,
        "disk_coverage_verified": disk_coverage_verified,
        "source_complete": source_complete,
        "evaluation_complete": evaluated,
        "source": {
            "snapshot": batch.snapshot.to_dict(),
            "coverage": batch.coverage.to_dict(),
            "fresh_read": fresh.snapshot.to_dict() if fresh is not None else None,
            "record_count": len(batch.records),
            "content_digest": batch.snapshot.content_sha256,
        },
        "source_reasons": source_reasons,
        "out_of_range_evaluations": sorted(out_of_range),
        "duplicate_identity_keys": sorted(duplicate_keys),
        "topologies": topology_results,
        "counts": {
            "source_records": len(batch.records),
            "evaluations_received": len(entries),
            "evaluations_bound": len(rows_by_identity),
            "valid_pass": sum(row["valid"] and row["status"] == "pass" for row in rows_by_identity.values()),
            "valid_reject": sum(row["valid"] and row["status"] == "reject" for row in rows_by_identity.values()),
            "unknown": sum(not row["valid"] or row["status"] == "unknown" for row in rows_by_identity.values())
            + (len(batch.records) - len(rows_by_identity)),
        },
    }


__all__ = ["GENERIC_AGGREGATION_SCHEMA", "GenericAggregationError", "aggregate_generic"]
