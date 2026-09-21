"""Opt-in compact checkpoint runner for long five-body searches.

The ordinary :mod:`reporting` path keeps its historical report and catalog
behavior.  This module is an explicit checkpoint-only path: it processes one
topology at a time, strictly saves both identity modes to the append-only
checkpoint, then releases topology-local candidate records and identity
lists.  Its aggregate report contains audit rows only and never claims to be
the ordinary raw-candidate report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Iterable, Mapping

from .canonical import canonicalize_topology
from .catalog_checkpoint import CatalogCheckpointWriter, CheckpointError, read_checkpoint
from .domain import CountMode, RunStatus
from .enumeration import EnumerationLimits, EnumerationReport, enumerate_topologies
from .label_catalog import (CATALOG_SCHEMA_VERSION, IDENTITY_SCHEMA_VERSION, CandidateCatalog,
                            CatalogEntry, CatalogIdentity, CatalogManifest, build_catalog, write_catalog)
from .label_identity import (CandidateIdentity, _marked_payload, align_topology,
                              canonicalize_candidate_labeling)
from .labeling import CandidateSearchReport, LabelingLimits, enumerate_labelings
from .reporting import (_ATOMIC_REPLACE_ATTEMPTS, _ATOMIC_REPLACE_INITIAL_DELAY,
                        _ATOMIC_REPLACE_MAX_SECONDS, _write_json_atomic, source_fingerprint)


_DEFAULT_MAX_SECONDS = 540.0
_MAX_BUDGET_SECONDS = 1200.0
_COMPACT_SCHEMA_VERSION = "F5-compact-run-report-v1"
_CHECKPOINT_SCHEMA_VERSION = "F5-catalog-checkpoint-v1"
FlowProvider = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class CompactRunReport:
    status: str
    n: int
    loop_count: int
    started_at: str
    finished_at: str
    deadline_seconds: float
    stop_reason: str | None
    topology_report: EnumerationReport
    compact_audit: tuple[dict[str, Any], ...]
    checkpoint_directory: str | None
    output_directory: str | None
    stage_timings: Mapping[str, float] = field(default_factory=dict)
    progress: Mapping[str, Any] = field(default_factory=dict)
    failures: tuple[dict[str, Any], ...] = ()
    label_strategy: str = "default"
    flow_record_order: str = "reference"
    flow_particle_orbit: bool = False
    aggregate_catalog_materialized: bool = False
    benchmark: str = "compact_checkpoint_only"
    context: Mapping[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.status == RunStatus.COMPLETE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _COMPACT_SCHEMA_VERSION,
            "status": self.status,
            "complete": self.complete,
            "n": self.n,
            "loop_count": self.loop_count,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "deadline_seconds": self.deadline_seconds,
            "stop_reason": self.stop_reason,
            "topology_report": self.topology_report.to_dict(),
            "compact_audit": list(self.compact_audit),
            "checkpoint": {
                "path": self.checkpoint_directory,
                "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "scope_complete": self.topology_report.complete,
                "committed_topologies": sum(bool(row.get("checkpoint_committed")) for row in self.compact_audit),
                "complete_topologies": sum(bool(row.get("complete")) for row in self.compact_audit),
                "expected_topologies": self.topology_report.unique_count if self.topology_report.complete else None,
            },
            "output_directory": self.output_directory,
            "stage_timings": dict(self.stage_timings),
            "timings": dict(self.stage_timings),
            "progress": dict(self.progress),
            "failures": list(self.failures),
            "label_strategy": self.label_strategy,
            "flow_record_order": self.flow_record_order,
            "flow_particle_orbit": self.flow_particle_orbit,
            "context": dict(self.context),
            "aggregate_catalog_materialized": self.aggregate_catalog_materialized,
            "benchmark": self.benchmark,
            "independence": {"status": "not_evaluated", "reason": "criterion_deferred"},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _validate_budget(max_seconds: float) -> float:
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)):
        raise TypeError("max_seconds must be a finite number")
    value = float(max_seconds)
    if not math.isfinite(value) or value < 0.0 or value > _MAX_BUDGET_SECONDS:
        raise ValueError(f"max_seconds must be finite and between 0 and {_MAX_BUDGET_SECONDS:g} seconds")
    return value


def _output_root(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        root = root / f"run-{time.time_ns()}-{os.getpid()}"
        root.mkdir(parents=True)
    return root


def _scope(n: int, loop_count: int, topology_report: EnumerationReport, label_strategy: str,
           flow_record_order: str, flow_particle_orbit: bool) -> dict[str, Any]:
    return {
        "n": n,
        "loop_count": loop_count,
        "rules_schema": "F5-plan-v1",
        "source_revision": f"sha256:{source_fingerprint()}",
        "baseline": "64f9a26",
        "runner_mode": "compact_checkpoint_only",
        "label_strategy": label_strategy,
        "flow_record_order": flow_record_order,
        "flow_particle_orbit": flow_particle_orbit,
        "topology_count": topology_report.unique_count if topology_report.complete else None,
    }


def _run_context(*, max_seconds: float, topology_limits: EnumerationLimits,
                 label_strategy: str, flow_particle_orbit: bool) -> dict[str, Any]:
    return {
        "rules_schema": "F5-plan-v1",
        "catalog_schema": CATALOG_SCHEMA_VERSION,
        "identity_schema": IDENTITY_SCHEMA_VERSION,
        "source_revision": f"sha256:{source_fingerprint()}",
        "baseline": "64f9a26",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "limits": {"max_seconds": max_seconds, "dot_seconds": 60.0,
                    "topology": topology_limits.to_dict()},
        "execution": {"render": False, "benchmark": "compact_checkpoint_only",
                       "label_strategy": label_strategy,
                       "flow_particle_orbit": flow_particle_orbit},
    }


def _failure_summary(report: Any) -> dict[str, Any]:
    return {
        "failure_counts": dict(report.failure_counts),
        "failure_sample_codes": [sample.code for sample in report.failure_samples],
        "failure_witness_codes": [code for code, _ in report.failure_witnesses],
        "failure_witness_digest": hashlib.sha256(
            json.dumps({code: value.to_dict() for code, value in report.failure_witnesses},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _scalar_flow_evidence(report: Any) -> dict[str, Any]:
    """Return the flow proof fields without changing the legacy report shape.

    The scalar-flow strategy can complete a topology with zero momentum-pass
    records.  Keeping its actual cut values and rejection reason in the
    compact audit makes that zero result reviewable; counts alone cannot
    distinguish a valid empty partition from a failed preflight.
    """

    scalar_flow = getattr(report, "scalar_flow", None)
    if scalar_flow is not None:
        scalar_flow = [
            {"edge_index": int(ref[0]), "occurrence": int(ref[1]), "value": int(value)}
            for ref, value in scalar_flow
        ]
    return {
        "scalar_flow": scalar_flow,
        "scalar_flow_legal": getattr(report, "scalar_flow_legal", None),
        "scalar_flow_rejection": getattr(report, "scalar_flow_rejection", None),
    }


def _resolve_flow_provider(provider: FlowProvider | None) -> FlowProvider:
    if provider is not None:
        if not callable(provider):
            raise TypeError("flow_provider must be callable")
        return provider
    try:
        from .flow_labeling import enumerate_flow_labelings  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("label_strategy='flow' requires phase_space_graph5.flow_labeling.enumerate_flow_labelings") from exc
    return enumerate_flow_labelings


def _call_flow_provider(provider: FlowProvider, topology: Any, *, limits: LabelingLimits,
                        cancel_requested: Callable[[], bool], deadline: float | None,
                        record_order: str, flow_particle_orbit: bool) -> Any:
    """Call the owner-supplied flow strategy through its stable keyword boundary."""

    signature = inspect.signature(provider)
    kwargs: dict[str, Any] = {}
    available = signature.parameters
    for name, value in (("limits", limits), ("cancel_requested", cancel_requested),
                        ("deadline", deadline), ("record_order", record_order)):
        if name in available or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in available.values()):
            kwargs[name] = value
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in available.values()
    )
    if "flow_particle_orbit" in available:
        kwargs["flow_particle_orbit"] = flow_particle_orbit
    elif "use_particle_orbit" in available or accepts_kwargs:
        # The owned flow provider keeps its lower-level keyword name; the
        # compact runner exposes the explicit flow-scoped public option.
        kwargs["use_particle_orbit"] = flow_particle_orbit
    elif flow_particle_orbit:
        # An explicitly requested optimization must never be silently
        # dropped at a custom-provider boundary.  The default False path
        # remains compatible with older providers that predate this option.
        raise TypeError(
            "flow_particle_orbit=True requires a flow provider accepting "
            "flow_particle_orbit, use_particle_orbit, or **kwargs"
        )
    report = provider(topology, **kwargs)
    required = ("topology", "candidate_records", "failure_counts", "failure_samples",
                "failure_witnesses", "limits", "stop_reason", "complete", "status")
    if any(not hasattr(report, name) for name in required):
        raise TypeError("flow provider must return a CandidateSearchReport-compatible FlowLabelingReport")
    if flow_particle_orbit:
        confirmed = getattr(report, "particle_orbit_reuse", None)
        if type(confirmed) is not bool or not confirmed:
            raise TypeError(
                "flow provider did not confirm particle_orbit_reuse=True "
                "for flow_particle_orbit=True"
            )
    return report


def _run_label_strategy(topology: Any, *, limits: LabelingLimits, cancel_requested: Callable[[], bool],
                       deadline: float | None, label_strategy: str,
                       flow_provider: FlowProvider | None,
                       flow_record_order: str, flow_particle_orbit: bool) -> Any:
    if label_strategy == "default":
        return enumerate_labelings(topology, count_mode=CountMode.FIXED_LABELS, limits=limits,
                                   cancel_requested=cancel_requested, deadline=deadline)
    if label_strategy == "flow":
        return _call_flow_provider(_resolve_flow_provider(flow_provider), topology,
                                   limits=limits, cancel_requested=cancel_requested, deadline=deadline,
                                   record_order=flow_record_order,
                                   flow_particle_orbit=flow_particle_orbit)
    raise ValueError("label_strategy must be 'default' or 'flow'")


def _identity_mode_lists(report: Any, *, cancel_requested: Callable[[], bool],
                         deadline: float, failure_is_error: bool) -> tuple[dict[str, list[CandidateIdentity]],
                                                   dict[str, dict[str, Any]], list[dict[str, Any]], bool, str | None,
                                                   dict[str, dict[str, int]]]:
    identities: dict[str, list[CandidateIdentity]] = {mode.value: [] for mode in CountMode}
    sources: dict[str, dict[str, Any]] = {mode.value: {} for mode in CountMode}
    prekey_identity: dict[str, dict[Any, CandidateIdentity]] = {mode.value: {} for mode in CountMode}
    prekey_stats: dict[str, dict[str, int]] = {
        mode.value: {"prekey_hits": 0, "canonicalization_calls": 0}
        for mode in CountMode
    }
    failures: list[dict[str, Any]] = []
    status_value = getattr(report.status, "value", report.status)
    report_failures = dict(getattr(report, "failure_counts", ()))
    failed = (failure_is_error and bool(report_failures)) or status_value == RunStatus.FAILED.value
    stop_reason: str | None = report.stop_reason
    seen: dict[str, set[str]] = {mode.value: set() for mode in CountMode}
    for record in report.candidate_records:
        if cancel_requested() or time.monotonic() >= deadline:
            stop_reason = stop_reason or "deadline_exceeded"
            failed = True
            break
        try:
            alignment = align_topology(record.topology)
            vertex_map = dict(alignment.vertex_map)
            final_particles = sorted(record.complete_labeling.final_particles,
                                     key=lambda item: (item.vertex, item.slot))
            particle_map = {item.particle_id: index + 1 for index, item in enumerate(final_particles)}
            for mode in CountMode:
                mode_name = mode.value
                prekey = _marked_payload(
                    record,
                    vertex_map,
                    None if mode is CountMode.FIXED_LABELS else particle_map,
                )
                cached = prekey_identity[mode_name].get(prekey)
                if cached is not None:
                    prekey_stats[mode_name]["prekey_hits"] += 1
                    continue
                prekey_stats[mode_name]["canonicalization_calls"] += 1
                identity = canonicalize_candidate_labeling(record, mode)
                prekey_identity[mode_name][prekey] = identity
                if identity.digest not in seen[mode_name]:
                    seen[mode_name].add(identity.digest)
                    identities[mode_name].append(identity)
                    sources[mode_name][identity.digest] = record
        except (TypeError, ValueError) as exc:
            failed = True
            failures.append({"stage": "identity", "error": str(exc), "topology": report.topology.to_dict()})
    if report.stop_reason:
        failed = True
    if failed and stop_reason is None:
        stop_reason = "labeling_error"
    return identities, sources, failures, failed, stop_reason, prekey_stats


def run_compact_candidates(*, n: int = 5, loop_count: int = 0,
                           max_seconds: float = _DEFAULT_MAX_SECONDS,
                           topology_limits: EnumerationLimits | None = None,
                           labeling_limits: LabelingLimits | None = None,
                           use_pruning: bool = True,
                           output_dir: str | Path | None = None,
                           label_strategy: str = "default",
                           flow_record_order: str = "reference",
                           flow_particle_orbit: bool = False,
                           flow_provider: FlowProvider | None = None,
                           cancel_requested: Callable[[], bool] | None = None,
                           checkpoint_writer_factory: Callable[..., CatalogCheckpointWriter] | None = None,
                           write_report: bool = True) -> CompactRunReport:
    """Run explicit compact checkpoint mode.

    `flow_provider` is an internal owner boundary for the flow-labeling
    implementation.  The default path remains the existing public labeling
    search.  This function never builds aggregate catalogs and never stores a
    raw candidate-record sequence in its report.
    """

    max_seconds = _validate_budget(max_seconds)
    if output_dir is None:
        raise ValueError("compact checkpoint mode requires output_dir")
    if label_strategy not in {"default", "flow"}:
        raise ValueError("label_strategy must be 'default' or 'flow'")
    if flow_record_order not in {"flow", "reference"}:
        raise ValueError("flow_record_order must be 'flow' or 'reference'")
    if type(flow_particle_orbit) is not bool:
        raise TypeError("flow_particle_orbit must be a boolean")
    if flow_particle_orbit and label_strategy != "flow":
        raise ValueError("flow_particle_orbit requires label_strategy='flow'")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None")
    external_cancel = cancel_requested or (lambda: False)
    if label_strategy == "flow" and flow_provider is None:
        # Fail before making output or enumerating so an unavailable optional
        # strategy cannot create a misleading partial run.
        _resolve_flow_provider(None)
    started_at = _timestamp()
    run_started = time.monotonic()
    deadline = run_started + max_seconds
    root = _output_root(output_dir)
    failures: list[dict[str, Any]] = []
    stop_reason: str | None = None
    status = RunStatus.INCOMPLETE.value
    stage_timings: dict[str, float] = {
        "topology_enumeration": 0.0,
        "labeling_enumeration": 0.0,
        "identity_canonicalization": 0.0,
        "topology_checkpoint": 0.0,
        "compact_report_persistence": 0.0,
        "total": 0.0,
    }
    progress: dict[str, Any] = {
        "schema_version": "F5-compact-progress-v1",
        "status": "running",
        "stage": "initializing",
        "stop_reason": None,
        "started_at": started_at,
        "updated_at": _timestamp(),
        "elapsed_seconds": 0.0,
        "stage_timings": stage_timings,
        "progress": {
            "topologies_discovered": 0,
            "topologies_total": None,
            "topologies_committed": 0,
            "topologies_complete": 0,
            "raw_assignments": 0,
            "momentum_pass": 0,
            "representative_evaluations": 0,
            "certificate_transports": 0,
            "identity_counts": {mode.value: 0 for mode in CountMode},
        },
        "benchmark": "compact_checkpoint_only",
        "render": False,
        "label_strategy": label_strategy,
        "flow_particle_orbit": flow_particle_orbit,
        "resume": {"supported": False, "reason": "single_pass_run"},
    }

    def persist(stage: str, *, progress_status: str = "running", reason: str | None = None) -> None:
        progress["stage"] = stage
        progress["status"] = progress_status
        progress["stop_reason"] = reason
        progress["updated_at"] = _timestamp()
        progress["elapsed_seconds"] = max(0.0, time.monotonic() - run_started)
        progress["stage_timings"] = dict(stage_timings)
        if root is not None and write_report:
            try:
                _write_json_atomic(root / "run_progress.json", progress, deadline=deadline)
            except OSError as exc:
                failures.append({"stage": "progress_persistence", "error": str(exc)})
                progress["status"] = "incomplete"
                progress["stop_reason"] = "progress_persistence_failed"

    persist("initializing")
    topology_limits_value = topology_limits or EnumerationLimits()
    if topology_limits_value.max_seconds is None or topology_limits_value.max_seconds > max_seconds:
        topology_limits_value = EnumerationLimits(topology_limits_value.max_candidates,
                                                  topology_limits_value.max_topologies,
                                                  topology_limits_value.max_skeletons,
                                                  max_seconds)
    run_context = _run_context(
        max_seconds=max_seconds,
        topology_limits=topology_limits_value,
        label_strategy=label_strategy,
        flow_particle_orbit=flow_particle_orbit,
    )

    def cancelled() -> bool:
        return external_cancel() or _remaining(deadline) <= 0 or bool(failures and any(item.get("stage") == "progress_persistence" for item in failures))

    def persistence_failed() -> bool:
        return any(item.get("stage") == "progress_persistence" for item in failures)

    topology_started = time.monotonic()
    persist("topology_enumeration")
    topology_report = enumerate_topologies(n, loop_count, topology_limits_value,
                                           use_pruning=use_pruning, cancel_requested=cancelled)
    stage_timings["topology_enumeration"] = max(0.0, time.monotonic() - topology_started)
    progress["progress"]["topologies_discovered"] = len(topology_report.topologies)
    progress["progress"]["topologies_total"] = topology_report.unique_count if topology_report.complete else None
    stop_reason = topology_report.stop_reason
    if _remaining(deadline) <= 0:
        stop_reason = "deadline_exceeded"
    elif external_cancel() and stop_reason is None:
        stop_reason = "cancel_requested"
    persist("labeling_enumeration", reason=stop_reason)

    checkpoint_root = root / "catalog" / "checkpoints" if root is not None else None
    checkpoint_writer: CatalogCheckpointWriter | None = None
    if checkpoint_root is not None:
        factory = checkpoint_writer_factory or CatalogCheckpointWriter
        try:
            checkpoint_writer = factory(
                checkpoint_root,
                scope=_scope(n, loop_count, topology_report, label_strategy, flow_record_order,
                             flow_particle_orbit),
                expected_topology_keys=(topology_report.canonical_keys if topology_report.complete else None),
            )
            if checkpoint_writer is None or not callable(
                getattr(checkpoint_writer, "_append_verified_topology", None)
            ):
                raise TypeError(
                    "checkpoint_writer_factory must return a writer with "
                    "_append_verified_topology; an empty writer cannot claim committed coverage"
                )
        except (OSError, CheckpointError, TypeError, ValueError) as exc:
            error = {"stage": "checkpoint_writer", "error": str(exc)}
            failures.append(error)
            stop_reason = "checkpoint_writer_invalid"
            checkpoint_writer = None

    compact_rows: list[dict[str, Any]] = []
    for topology in topology_report.topologies:
        if checkpoint_root is not None and checkpoint_writer is None:
            # The writer is mandatory for compact mode.  In particular, do
            # not take the historical no-writer test seam as evidence that a
            # topology was durably committed.
            break
        if cancelled():
            stop_reason = stop_reason or (
                "progress_persistence_failed" if persistence_failed()
                else "cancel_requested" if external_cancel() else "deadline_exceeded"
            )
            break
        topology_key = tuple(canonicalize_topology(topology).key)
        limits = labeling_limits or LabelingLimits()
        remaining = _remaining(deadline)
        effective_seconds = remaining if limits.max_seconds is None else min(remaining, limits.max_seconds)
        effective_limits = LabelingLimits(limits.max_labelings, limits.max_results,
                                          effective_seconds, deadline)
        labeling_started = time.monotonic()
        try:
            report = _run_label_strategy(topology, limits=effective_limits, cancel_requested=cancelled,
                                         deadline=deadline, label_strategy=label_strategy,
                                         flow_provider=flow_provider,
                                         flow_record_order=flow_record_order,
                                         flow_particle_orbit=flow_particle_orbit)
        except Exception as exc:
            failures.append({"stage": "labeling", "error": str(exc), "topology": topology.to_dict()})
            stop_reason = stop_reason or "labeling_error"
            report = None
        stage_timings["labeling_enumeration"] += max(0.0, time.monotonic() - labeling_started)
        if report is None:
            audit = {
                "topology_key": list(topology_key), "status": "failed", "complete": False,
                "checkpoint_committed": False, "raw_assignment_count": 0,
                "momentum_pass_count": 0, "candidate_count": 0,
                "failure_counts": {}, "failure_sample_codes": [], "failure_witness_codes": [],
                "stop_reason": stop_reason or "labeling_error",
                "error": failures[-1], "raw_records_retained": 0,
            }
            compact_rows.append(audit)
            break
        progress["progress"]["raw_assignments"] += report.raw_assignment_count
        progress["progress"]["momentum_pass"] += report.momentum_pass_count
        progress["progress"]["representative_evaluations"] += getattr(report, "representative_evaluations", 0)
        progress["progress"]["certificate_transports"] += getattr(report, "certificate_transports", 0)
        persist("identity_canonicalization", reason=report.stop_reason)
        identity_started = time.monotonic()
        identities, sources, identity_failures, identity_failed, identity_stop, prekey_stats = _identity_mode_lists(
            report, cancel_requested=cancelled, deadline=deadline,
            failure_is_error=(label_strategy == "flow"),
        )
        stage_timings["identity_canonicalization"] += max(0.0, time.monotonic() - identity_started)
        failures.extend(identity_failures)
        if identity_stop:
            stop_reason = stop_reason or identity_stop
        report_status = getattr(report.status, "value", report.status)
        report_failure_counts = dict(getattr(report, "failure_counts", ()))
        report_failed = ((label_strategy == "flow" and bool(report_failure_counts))
                         or report_status == RunStatus.FAILED.value)
        topology_complete = bool(report.complete and not report_failed and not identity_failed
                                 and identity_stop is None and not cancelled())
        flow_counts = {
            name: getattr(report, name, None)
            for name in (
                "traditional_search_space_size", "flow_partition_generated_labelings",
                "flow_partition_normalized_labelings", "flow_partition_branch_count",
                "representative_evaluations", "certificate_transports",
            )
        }
        audit = {
            "topology_key": list(topology_key),
            "status": "complete" if topology_complete else "incomplete",
            "complete": topology_complete,
            "checkpoint_committed": False,
            "raw_assignment_count": report.raw_assignment_count,
            "momentum_pass_count": report.momentum_pass_count,
            "candidate_count": len(report.candidate_records),
            "representative_evaluations": getattr(report, "representative_evaluations", None),
            "certificate_transports": getattr(report, "certificate_transports", None),
            "particle_orbit_reuse": getattr(report, "particle_orbit_reuse", flow_particle_orbit),
            "flow_particle_orbit": flow_particle_orbit,
            "strategy": label_strategy,
            "flow_record_order": flow_record_order,
            "raw_assignment_count_semantics": (
                "flow_partition_generated_labelings" if label_strategy == "flow" else "legacy_raw_assignments"
            ),
            "flow_counts": flow_counts,
            **_scalar_flow_evidence(report),
            "failure_counts": report_failure_counts,
            "failure_sample_codes": [sample.code for sample in report.failure_samples],
            "failure_witness_codes": [code for code, _ in report.failure_witnesses],
            "failure_witness_digest": _failure_summary(report)["failure_witness_digest"],
            "stop_reason": identity_stop or report.stop_reason or ("labeling_error" if report_failed else None),
            "identity_counts": {mode: len(values) for mode, values in identities.items()},
            "identity_prekey_stats": prekey_stats,
            "identity_error_count": len(identity_failures),
            "raw_records_retained": 0,
        }
        try:
            if checkpoint_writer is not None:
                checkpoint_started = time.monotonic()
                # Every identity here was just produced by the public
                # canonicalize_candidate_labeling path from this topology's
                # reviewed CandidateMomentumRecord.  The checkpoint writer's
                # private verified append preserves those strict objects and
                # avoids re-canonicalizing the same high-circle payload.  Its
                # public append path remains strict for external callers.
                checkpoint_writer._append_verified_topology(
                    topology_key,
                    identities_by_mode=identities,
                    source_records_by_mode=sources,
                    label_enumeration_complete={mode: bool(report.complete and not report_failed and not identity_failed) for mode in identities},
                    identity_save_complete={mode: bool(topology_complete) for mode in identities},
                    topology_complete=topology_complete,
                    audit={
                        **audit,
                        "stop_reason": audit["stop_reason"],
                        "p_witness_present": report.momentum_pass_count > 0,
                    },
                )
                stage_timings["topology_checkpoint"] += max(0.0, time.monotonic() - checkpoint_started)
                audit["checkpoint_committed"] = True
                progress["progress"]["topologies_committed"] += 1
                progress["progress"]["topologies_complete"] += int(topology_complete)
                progress["progress"]["identity_counts"] = {
                    mode: progress["progress"]["identity_counts"].get(mode, 0) + len(values)
                    for mode, values in identities.items()
                }
                persist("labeling_enumeration", reason=stop_reason)
            else:
                audit["checkpoint_committed"] = True
        except (OSError, CheckpointError, TypeError, ValueError) as exc:
            error = {"stage": "checkpoint", "error": str(exc), "topology_key": list(topology_key)}
            failures.append(error)
            audit["status"] = "incomplete"
            audit["complete"] = False
            audit["stop_reason"] = "checkpoint_persistence_failed"
            audit["error"] = error
            stop_reason = stop_reason or "checkpoint_persistence_failed"
        # The next iteration must not retain this topology's raw sequence.
        del identities, sources, report
        compact_rows.append(audit)
        if audit["stop_reason"] is not None or not audit["checkpoint_committed"]:
            break

    expected_complete = topology_report.complete and len(compact_rows) == topology_report.unique_count
    all_complete = expected_complete and all(row.get("complete") and row.get("checkpoint_committed") for row in compact_rows)
    if not expected_complete and stop_reason is None:
        stop_reason = "incomplete_topology_scope"
    if failures and stop_reason is None:
        stop_reason = "incomplete"
    status = RunStatus.COMPLETE.value if all_complete and not failures and stop_reason is None else RunStatus.INCOMPLETE.value
    if topology_report.status is RunStatus.FAILED:
        status = RunStatus.FAILED.value
    stage_timings["total"] = max(0.0, time.monotonic() - run_started)
    if _remaining(deadline) <= 0 and status == RunStatus.COMPLETE.value:
        status = RunStatus.INCOMPLETE.value
        stop_reason = stop_reason or "deadline_exceeded"
    progress["status"] = status
    progress["stage"] = "finished"
    progress["stop_reason"] = stop_reason
    progress["elapsed_seconds"] = stage_timings["total"]
    persist("finished", progress_status=status, reason=stop_reason)
    if any(item.get("stage") == "progress_persistence" for item in failures):
        status = RunStatus.INCOMPLETE.value
        stop_reason = stop_reason or "progress_persistence_failed"
        progress["status"] = status
        progress["stop_reason"] = stop_reason
    if root is not None and write_report:
        report_started = time.monotonic()
        provisional = CompactRunReport(status, n, loop_count, started_at, _timestamp(), max_seconds,
                                       stop_reason, topology_report, tuple(compact_rows),
                                       str(checkpoint_root) if checkpoint_root else None, str(root),
                                       dict(stage_timings), dict(progress), tuple(failures), label_strategy,
                                       flow_record_order, flow_particle_orbit, context=run_context)
        try:
            (root / "run_report.json").write_text(provisional.to_json() + "\n", encoding="utf-8")
            (root / "run_manifest.json").write_text(json.dumps(provisional.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            # A late sidecar failure must downgrade the returned report and
            # leave a truthful incomplete marker when the directory permits
            # one more write.  This is deliberately bounded: no retry loop
            # can extend the search past its global deadline.
            error = {"stage": "compact_report_persistence", "error": str(exc)}
            failures.append(error)
            status = RunStatus.INCOMPLETE.value
            stop_reason = stop_reason or "compact_report_persistence_failed"
            progress["status"] = status
            progress["stop_reason"] = stop_reason
            provisional = CompactRunReport(status, n, loop_count, started_at, _timestamp(), max_seconds,
                                           stop_reason, topology_report, tuple(compact_rows),
                                           str(checkpoint_root) if checkpoint_root else None, str(root),
                                           dict(stage_timings), dict(progress), tuple(failures), label_strategy,
                                           flow_record_order, flow_particle_orbit, context=run_context)
            try:
                (root / "run_report.json").write_text(provisional.to_json() + "\n", encoding="utf-8")
                (root / "run_manifest.json").write_text(json.dumps(provisional.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                # The caller still receives the downgraded object and the
                # checkpoint remains the durable source of committed blocks.
                pass
        if status == RunStatus.COMPLETE.value and _remaining(deadline) <= 0:
            # Report serialization itself is part of the bounded run.  A
            # complete search whose final sidecars crossed the wall-clock
            # deadline must be exposed as incomplete, even if the checkpoint
            # blocks are all present.
            status = RunStatus.INCOMPLETE.value
            stop_reason = stop_reason or "deadline_exceeded"
            progress["status"] = status
            progress["stop_reason"] = stop_reason
            failures.append({"stage": "compact_report_persistence",
                             "error": "final report persistence crossed deadline"})
            provisional = CompactRunReport(status, n, loop_count, started_at, _timestamp(), max_seconds,
                                           stop_reason, topology_report, tuple(compact_rows),
                                           str(checkpoint_root) if checkpoint_root else None, str(root),
                                           dict(stage_timings), dict(progress), tuple(failures), label_strategy,
                                           flow_record_order, flow_particle_orbit, context=run_context)
            try:
                (root / "run_report.json").write_text(provisional.to_json() + "\n", encoding="utf-8")
                (root / "run_manifest.json").write_text(json.dumps(provisional.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                # One bounded correction attempt is enough; the returned
                # object and worker exit status remain authoritative if the
                # directory is still locked.
                pass
        stage_timings["compact_report_persistence"] = max(0.0, time.monotonic() - report_started)
    return CompactRunReport(status, n, loop_count, started_at, _timestamp(), max_seconds,
                            stop_reason, topology_report, tuple(compact_rows),
                            str(checkpoint_root) if checkpoint_root else None, str(root) if root else None,
                            dict(stage_timings), dict(progress), tuple(failures), label_strategy,
                            flow_record_order, flow_particle_orbit, context=run_context)


def export_checkpoint_catalog(checkpoint_directory: str | Path, output_directory: str | Path,
                              *, mode: CountMode | str,
                              topology_keys: Iterable[Any] | None = None) -> CandidateCatalog:
    """Export selected complete checkpoint topologies to the old catalog schema.

    Every requested key must be inside the immutable scope and every selected
    block must be complete for both modes.  The checkpoint reader performs the
    strict source/key/map verification before this function constructs the
    compatibility catalog.
    """

    mode_value = mode.value if isinstance(mode, CountMode) else str(mode)
    if mode_value not in {item.value for item in CountMode}:
        raise ValueError(f"unknown catalog mode: {mode_value}")
    checkpoint = read_checkpoint(checkpoint_directory)
    expected = checkpoint.expected_topology_keys
    if expected is None:
        raise ValueError("checkpoint scope has no complete expected topology key list")
    expected_by_token = {json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":")): key for key in expected}
    selected = tuple(expected if topology_keys is None else topology_keys)
    selected_tokens = [json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for key in selected]
    if len(selected_tokens) != len(set(selected_tokens)):
        raise ValueError("selected topology keys contain duplicates")
    missing_scope = [token for token in selected_tokens if token not in expected_by_token]
    if missing_scope:
        raise ValueError("selected topology key is outside immutable checkpoint scope")
    records = {json.dumps(record.topology_key, ensure_ascii=False, sort_keys=True, separators=(",", ":")): record for record in checkpoint.records}
    candidate_identities: list[CandidateIdentity] = []
    source_records: dict[str, Any] = {}
    for token in selected_tokens:
        record = records.get(token)
        if record is None or not record.complete:
            raise ValueError("selected topology checkpoint is missing or incomplete")
        for item in record.identities_by_mode[mode_value]:
            identity = CandidateIdentity.from_dict(item.identity.to_dict())
            candidate_identities.append(identity)
            source_records[identity.digest] = item.source_record
    candidate_identities.sort(key=lambda identity: (identity.topology_key, identity.digest))
    digests = tuple(identity.digest for identity in candidate_identities)
    manifest = CatalogManifest(
        scope={"checkpoint_directory": str(Path(checkpoint_directory).resolve()),
               "selected_topology_keys": [list(key) for key in selected],
               "source_scope": dict(checkpoint.scope)},
        count_mode=mode_value, marker_scope="selected", topology_scope="catalog_subset",
        run_status="complete", topology_enumeration_complete=True,
        label_enumeration_complete=True, identity_save_complete=True,
        drawing_selection_complete=False, expected_identity_digests=digests,
        stored_identity_digests=digests, stop_reason=None, failures=(),
        extra={"selection": "checkpoint_topology_export", "aggregate_catalog_materialized": True},
    )
    catalog = build_catalog(candidate_identities, source_records=source_records, manifest=manifest)
    write_catalog(catalog, output_directory, overwrite=True)
    return catalog


def write_compact_run_report(report: CompactRunReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json() + "\n", encoding="utf-8")
    return path


__all__ = ["CompactRunReport", "run_compact_candidates", "export_checkpoint_catalog",
           "write_compact_run_report"]
