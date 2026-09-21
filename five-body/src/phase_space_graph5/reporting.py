"""Bounded candidate search, catalog persistence, and drawing output.

This module is the five-body orchestration boundary.  It stops at S/P,
never imports an invariant or combination-search module, and records partial
work before returning on a deadline or drawing failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import html
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

from .canonical import canonicalize_topology
from .catalog_checkpoint import CatalogCheckpointWriter
from .domain import CountMode, RunStatus
from .enumeration import EnumerationLimits, EnumerationReport, enumerate_topologies
from .label_catalog import (CandidateCatalog, CatalogEntry, CatalogManifest, build_catalog,
                            write_catalog)
from .label_identity import canonicalize_candidate_labeling
from .labeling import CandidateSearchReport, LabelingLimits, enumerate_labelings
from .drawing import (LayoutError, build_scene, layout_graph, prepare_drawing, render_drawing,
                      validate_drawing, verify_dot)


# The five-body workflow keeps the historical 540 s default for ordinary
# calls, while allowing the overnight runner to request the full 20 minute
# wall-clock budget.  The outer CLI supervisor owns the hard deadline and
# still reserves a small window for its final status checkpoint.
_DEFAULT_MAX_SECONDS = 540.0
_MAX_BUDGET_SECONDS = 1200.0
_PROGRESS_SCHEMA_VERSION = "F5-run-progress-v1"
_ATOMIC_REPLACE_ATTEMPTS = 4
_ATOMIC_REPLACE_MAX_SECONDS = 0.20
_ATOMIC_REPLACE_INITIAL_DELAY = 0.025


@dataclass(frozen=True, slots=True)
class DrawingOutcome:
    digest: str
    mode: str
    topology_key: tuple[Any, ...]
    status: str
    view: str
    images: tuple[str, ...] = ()
    validation: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    original_vertex_map: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"digest": self.digest, "mode": self.mode, "topology_key": list(self.topology_key),
                "status": self.status, "view": self.view, "images": list(self.images),
                "validation": self.validation, "error": self.error,
                "original_vertex_map": [list(item) for item in self.original_vertex_map]}


@dataclass(frozen=True, slots=True)
class CandidateRunReport:
    status: str
    n: int
    loop_count: int
    count_mode: str
    started_at: str
    finished_at: str
    deadline_seconds: float
    stop_reason: str | None
    topology_report: EnumerationReport
    labeling_reports: tuple[CandidateSearchReport, ...]
    catalog: CandidateCatalog | None
    catalogs: Mapping[str, CandidateCatalog] = field(default_factory=dict)
    drawing_outcomes: tuple[DrawingOutcome, ...] = ()
    failures: tuple[dict[str, Any], ...] = ()
    output_directory: str | None = None
    stage_timings: Mapping[str, float] = field(default_factory=dict)
    progress: Mapping[str, Any] = field(default_factory=dict)
    benchmark: str = "search_and_render"
    checkpoint_directory: str | None = None

    @property
    def complete(self) -> bool:
        return self.status == RunStatus.COMPLETE.value and self.catalog is not None and self.catalog.complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "complete": self.complete, "n": self.n, "loop_count": self.loop_count,
            "count_mode": self.count_mode, "started_at": self.started_at, "finished_at": self.finished_at,
            "deadline_seconds": self.deadline_seconds, "stop_reason": self.stop_reason,
            "topology_report": self.topology_report.to_dict(),
            "labeling_reports": [report.to_dict() for report in self.labeling_reports],
            "catalog": self.catalog.manifest.to_dict() if self.catalog else None,
            "catalogs": {key: value.manifest.to_dict() for key, value in self.catalogs.items()},
            "drawing_outcomes": [outcome.to_dict() for outcome in self.drawing_outcomes],
            "failures": list(self.failures), "output_directory": self.output_directory,
            # These are additive orchestration metadata.  Existing physical,
            # identity, and catalog keys above retain their established
            # meaning and schema.
            "stage_timings": dict(self.stage_timings),
            "timings": dict(self.stage_timings),
            "progress": dict(self.progress),
            "benchmark": self.benchmark,
            "checkpoint_directory": self.checkpoint_directory,
            "independence": {"status": "not_evaluated", "reason": "criterion_deferred",
                             "rank": None, "certificate": None},
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


def _validate_max_drawings(max_drawings: int | None) -> None:
    if max_drawings is not None and (isinstance(max_drawings, bool) or
                                     not isinstance(max_drawings, int) or max_drawings < 0):
        raise ValueError("max_drawings must be a non-negative integer or None")


def _deadline_reason(reason: str | None, deadline: float) -> str | None:
    if reason == "cancel_requested" and _remaining(deadline) <= 0:
        return "deadline_exceeded"
    return reason


def _drawing_failure_status(exc: BaseException) -> str:
    return "incomplete" if isinstance(exc, TimeoutError) or (
        isinstance(exc, LayoutError) and str(exc).startswith("timeout:")
    ) else "failed"


def _scope(n: int, loop_count: int, mode: CountMode, *, topology_report: EnumerationReport,
           marker_scope: str) -> dict[str, Any]:
    return {"n": n, "loop_count": loop_count, "count_mode": mode.value,
            "topology_count": topology_report.unique_count, "marker_scope": marker_scope,
            "independence": "not_evaluated", "rules_schema": "F5-plan-v1",
            "source_revision": f"sha256:{source_fingerprint()}", "baseline": "64f9a26"}


def _run_context(*, dot_path: str | None, max_seconds: float,
                 topology_limits: EnumerationLimits, render: bool = True) -> dict[str, Any]:
    return {
        "rules_schema": "F5-plan-v1",
        "catalog_schema": "F5-candidate-catalog-v1",
        "identity_schema": "IM-5-candidate-v1",
        "source_revision": f"sha256:{source_fingerprint()}",
        "baseline": "64f9a26",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "graphviz_dot": dot_path or os.environ.get("PSF_GRAPHVIZ_DOT") or "auto",
        },
        "limits": {
            "max_seconds": max_seconds,
            "dot_seconds": 60.0,
            "topology": topology_limits.to_dict(),
        },
        "execution": {
            "render": render,
            "benchmark": "search_only" if not render else "search_and_render",
        },
    }


def source_fingerprint() -> str:
    """Hash the five-body source files used by a run for reproducibility."""
    root = Path(__file__).resolve().parent
    rows = []
    for path in sorted(root.rglob("*.py")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{path.relative_to(root).as_posix()}:{digest}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _output_root(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    root = Path(path)
    if not root.exists():
        root.mkdir(parents=True)
        return root
    if not any(root.iterdir()):
        return root
    stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    child = root / stamp
    child.mkdir(parents=True)
    return child


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=list), encoding="utf-8")


def _write_json_atomic(path: Path, payload: Any, *, deadline: float | None = None) -> None:
    """Persist a small progress JSON with a bounded Windows sharing-lock retry.

    Only the progress sidecar uses this helper.  Catalog data and checkpoint
    index commits retain their own append/fsync protocol.  A failed replace
    leaves the previous target untouched and removes only this call's temp
    file before raising a bounded, inspectable error.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    payload_bytes = json.dumps(payload, ensure_ascii=False, indent=2, default=list)
    try:
        temporary.write_text(payload_bytes, encoding="utf-8")
        started = time.monotonic()
        last_error: PermissionError | None = None
        attempts_made = 0
        for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
            attempts_made += 1
            try:
                temporary.replace(path)
                return
            except PermissionError as exc:
                last_error = exc
                if attempt + 1 >= _ATOMIC_REPLACE_ATTEMPTS:
                    break
                elapsed = time.monotonic() - started
                remaining_cap = _ATOMIC_REPLACE_MAX_SECONDS - elapsed
                if deadline is not None:
                    remaining_cap = min(remaining_cap, max(0.0, deadline - time.monotonic()))
                if remaining_cap <= 0.0:
                    break
                delay = min(_ATOMIC_REPLACE_INITIAL_DELAY * (2 ** attempt), remaining_cap)
                if delay <= 0.0:
                    break
                time.sleep(delay)
        assert last_error is not None
        raise PermissionError(
            f"atomic progress replace failed after {attempts_made} attempts: {path}"
        ) from last_error
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                # The target is never removed during cleanup.  A locked temp
                # is left for postmortem inspection rather than risking an
                # unrelated file operation.
                pass


def _render_one(entry: CatalogEntry, output: Path, *, dot_path: str | None, deadline: float,
                view: str) -> DrawingOutcome:
    try:
        if _remaining(deadline) <= 0:
            raise TimeoutError("candidate run deadline exceeded before drawing")
        inp = prepare_drawing(record=entry.source_record, identity=entry.identity)
        layout = layout_graph(inp, dot_path=dot_path, timeout=min(60.0, _remaining(deadline)), deadline=deadline)
        scene = build_scene(inp, layout)
        validation = validate_drawing(inp, scene)
        if not validation.valid:
            return DrawingOutcome(entry.digest, entry.identity.mode, entry.topology_key, "needs_review", view,
                                  validation=validation.to_dict(), error="; ".join(validation.issues),
                                  original_vertex_map=entry.identity.vertex_map)
        stem = f"P_{entry.digest[:16]}_{entry.identity.mode}"
        images = []
        for fmt in ("svg", "png"):
            if _remaining(deadline) <= 0:
                raise TimeoutError("candidate run deadline exceeded during drawing")
            target = output / f"{stem}.{fmt}"
            target.write_bytes(render_drawing(scene, fmt=fmt))
            images.append(target.name)
        _write_json(output / f"{stem}.json", {
            "digest": entry.digest, "identity_key": list(entry.key), "view": view,
            "mode": entry.identity.mode, "evidence_stage": inp.evidence_stage,
            "independence": "not_evaluated", "validation": validation.to_dict(),
            "source_record_vertex_map": [list(item) for item in entry.identity.vertex_map],
            "images": images,
        })
        return DrawingOutcome(entry.digest, entry.identity.mode, entry.topology_key, "completed", view,
                              tuple(images), validation.to_dict(), original_vertex_map=entry.identity.vertex_map)
    except (LayoutError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        return DrawingOutcome(entry.digest, entry.identity.mode, entry.topology_key,
                              _drawing_failure_status(exc), view,
                              error=str(exc), original_vertex_map=entry.identity.vertex_map)


def _render_structure_input(topology: Any, topology_key: tuple[Any, ...], output: Path,
                            *, dot_path: str | None, deadline: float,
                            mode: str = "fixed_labels",
                            original_vertex_map: tuple[tuple[int, int], ...] = ()) -> DrawingOutcome:
    """Render one S structure without requiring a P identity."""
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(topology_key, ensure_ascii=False,
                                       separators=(",", ":"), default=list).encode("utf-8")).hexdigest()
    try:
        if _remaining(deadline) <= 0:
            raise TimeoutError("candidate run deadline exceeded before drawing")
        inp = prepare_drawing(topology=topology)
        layout = layout_graph(inp, dot_path=dot_path, timeout=min(60.0, _remaining(deadline)), deadline=deadline)
        scene = build_scene(inp, layout)
        validation = validate_drawing(inp, scene)
        if not validation.valid:
            return DrawingOutcome(digest, mode, topology_key, "needs_review", "structure",
                                  validation=validation.to_dict(), error="; ".join(validation.issues),
                                  original_vertex_map=original_vertex_map)
        stem = f"S_{digest[:16]}"
        images = []
        for fmt in ("svg", "png"):
            if _remaining(deadline) <= 0:
                raise TimeoutError("candidate run deadline exceeded during drawing")
            target = output / f"{stem}.{fmt}"
            target.write_bytes(render_drawing(scene, fmt=fmt))
            images.append(target.name)
        _write_json(output / f"{stem}.json", {
            "topology_key": list(topology_key), "view": "structure", "status_layer": "S",
            "independence": "not_evaluated", "validation": validation.to_dict(), "images": images,
            "original_vertex_map": [list(item) for item in original_vertex_map],
        })
        return DrawingOutcome(digest, mode, topology_key, "completed", "structure",
                              tuple(images), validation.to_dict(), original_vertex_map=original_vertex_map)
    except (LayoutError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        return DrawingOutcome(digest, mode, topology_key,
                              _drawing_failure_status(exc), "structure",
                              error=str(exc), original_vertex_map=original_vertex_map)


def _render_structure(entry: CatalogEntry, output: Path, *, dot_path: str | None,
                      deadline: float) -> DrawingOutcome:
    """Render one explicitly selected S structure from a catalog entry."""
    return _render_structure_input(entry.source_record.topology, entry.topology_key, output,
                                   dot_path=dot_path, deadline=deadline,
                                   mode=entry.identity.mode,
                                   original_vertex_map=entry.identity.vertex_map)


def _write_index(output: Path, outcomes: Iterable[DrawingOutcome], *, scope: Mapping[str, Any]) -> None:
    rows = []
    for outcome in outcomes:
        links = " ".join(f'<a href="{html.escape(name)}">{html.escape(name)}</a>' for name in outcome.images)
        rows.append("<tr>" + "".join((f"<td>{html.escape(outcome.digest)}</td>",
                                      f"<td>{html.escape(outcome.mode)}</td>",
                                      f"<td>{html.escape(outcome.status)}</td>",
                                      f"<td>{html.escape(outcome.view)}</td>",
                                      f"<td>{links}</td>")) + "</tr>")
    document = "<!doctype html><meta charset='utf-8'><title>五体候选图集</title>" \
               "<style>body{font-family:sans-serif}table{border-collapse:collapse}td,th{padding:.35rem;border:1px solid #ccc}</style>" \
               "<h1>五体候选图集</h1><pre>" + html.escape(json.dumps(scope, ensure_ascii=False, indent=2)) + \
               "</pre><table><thead><tr><th>digest</th><th>mode</th><th>status</th><th>view</th><th>files</th></tr></thead><tbody>" \
               + "".join(rows) + "</tbody></table>"
    (output / "index.html").write_text(document, encoding="utf-8")


def _write_combined_index(output: Path, groups: Mapping[str, Iterable[DrawingOutcome]]) -> None:
    rows = []
    for label, outcomes in groups.items():
        for outcome in outcomes:
            links = " ".join(f'<a href="{html.escape(label)}/{html.escape(name)}">{html.escape(name)}</a>' for name in outcome.images)
            rows.append(f"<tr><td>{html.escape(label)}</td><td>{html.escape(outcome.digest)}</td>"
                        f"<td>{html.escape(outcome.status)}</td><td>{links}</td></tr>")
    document = "<!doctype html><meta charset='utf-8'><title>五体候选图集</title>" \
               "<style>body{font-family:sans-serif}table{border-collapse:collapse}td,th{padding:.35rem;border:1px solid #ccc}</style>" \
               "<h1>五体候选图集</h1><table><thead><tr><th>view</th><th>digest</th><th>status</th><th>files</th></tr></thead><tbody>" \
               + "".join(rows) + "</tbody></table>"
    (output / "index.html").write_text(document, encoding="utf-8")


def render_catalog(catalog: CandidateCatalog, output_dir: str | Path, *, dot_path: str | None = None,
                   max_items: int | None = None, deadline: float | None = None,
                   view: str = "marked") -> tuple[DrawingOutcome, ...]:
    """Render selected stored identities without searching the catalog."""

    if view not in {"marked", "structure"}:
        raise ValueError("view must be 'marked' or 'structure'")
    if not isinstance(catalog, CandidateCatalog):
        raise TypeError("catalog must be CandidateCatalog")
    _validate_max_drawings(max_items)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    end = deadline if deadline is not None else time.monotonic() + 540.0
    if view == "structure":
        grouped: dict[tuple[Any, ...], CatalogEntry] = {}
        for entry in catalog.entries:
            grouped.setdefault(entry.topology_key, entry)
        selected = tuple(grouped[key] for key in sorted(grouped))
    else:
        selected = catalog.entries
    if max_items is not None:
        selected = selected[:max_items]
    outcomes = []
    for entry in selected:
        if _remaining(end) <= 0:
            outcomes.append(DrawingOutcome(entry.digest, entry.identity.mode, entry.topology_key,
                                           "incomplete", view, error="candidate run deadline exceeded",
                                           original_vertex_map=entry.identity.vertex_map))
            continue
        outcomes.append(_render_structure(entry, output, dot_path=dot_path, deadline=end)
                         if view == "structure"
                         else _render_one(entry, output, dot_path=dot_path, deadline=end, view=view))
    marker_scope = "all_stored" if max_items is None else "selected"
    selection_complete = bool(outcomes) and all(item.status == "completed" for item in outcomes)
    if not selected:
        selection_complete = max_items is None
    scope = {"marker_scope": marker_scope, "view": view,
             "selection_complete": selection_complete,
             "completeness_claim": "all_stored_identities" if max_items is None else "selected_items_only",
             "independence": "not_evaluated"}
    _write_index(output, outcomes, scope=scope)
    _write_json(output / "drawing_manifest.json", {"scope": scope,
                                                    "outcomes": [item.to_dict() for item in outcomes]})
    return tuple(outcomes)


def run_candidates(*, n: int = 5, loop_count: int = 0,
                   count_mode: CountMode | str = CountMode.FIXED_LABELS,
                   max_seconds: float = _DEFAULT_MAX_SECONDS, topology_limits: EnumerationLimits | None = None,
                   labeling_limits: LabelingLimits | None = None, use_pruning: bool = True,
                   output_dir: str | Path | None = None, dot_path: str | None = None,
                   render: bool = True, max_drawings: int | None = None) -> CandidateRunReport:
    mode = count_mode if isinstance(count_mode, CountMode) else CountMode(count_mode)
    max_seconds = _validate_budget(max_seconds)
    _validate_max_drawings(max_drawings)
    started = _timestamp()
    run_started = time.monotonic()
    deadline = time.monotonic() + max_seconds
    failures: list[dict[str, Any]] = []
    root = _output_root(output_dir)
    benchmark = "search_only" if not render else "search_and_render"
    # A progress-sidecar failure is itself an incomplete-run condition.  Keep
    # the in-memory status conservative even if the failure happens late.
    status = "incomplete"
    progress_persistence_errors: list[str] = []
    stage_timings: dict[str, float] = {
        "topology_enumeration": 0.0,
        "labeling_enumeration": 0.0,
        "identity_canonicalization": 0.0,
        "topology_checkpoint": 0.0,
        "catalog_persistence": 0.0,
        "report_persistence": 0.0,
        "drawing": 0.0,
        "total": 0.0,
    }
    progress_counts: dict[str, Any] = {
        "topologies_discovered": 0,
        "topologies_total": None,
        "labeling_reports_completed": 0,
        "identity_counts": {CountMode.FIXED_LABELS.value: 0,
                             CountMode.PARTICLE_RELABELING.value: 0},
        "drawings_completed": 0,
        "drawings_requested": 0,
        "checkpoint_completed_topologies": 0,
        "checkpoint_committed_topologies": 0,
        "checkpoint_complete_topologies": 0,
        "checkpoint_last_topology_key": None,
    }
    progress_snapshot: dict[str, Any] = {}

    def persist_progress(stage: str, *, progress_status: str = "running",
                         stop_reason: str | None = None) -> None:
        """Write one durable stage boundary, never once per candidate row."""

        nonlocal progress_snapshot, status
        payload = {
            "schema_version": _PROGRESS_SCHEMA_VERSION,
            "status": progress_status,
            "stage": stage,
            "stop_reason": stop_reason,
            "started_at": started,
            "updated_at": _timestamp(),
            "elapsed_seconds": max(0.0, time.monotonic() - run_started),
            "stage_timings": dict(stage_timings),
            "timings": dict(stage_timings),
            "progress": dict(progress_counts),
            "progress_persistence_errors": list(progress_persistence_errors),
            "benchmark": benchmark,
            "render": render,
            "resume": {"supported": False, "reason": "single_pass_run"},
        }
        progress_snapshot = payload
        if root is not None:
            progress_path = root / "run_progress.json"
            try:
                _write_json_atomic(progress_path, payload, deadline=deadline)
            except OSError as exc:
                message = f"{progress_path}: {exc}"
                if message not in progress_persistence_errors:
                    progress_persistence_errors.append(message)
                    failures.append({"stage": "progress_persistence", "error": message})
                payload["status"] = "incomplete"
                payload["stop_reason"] = payload.get("stop_reason") or "progress_persistence_failed"
                payload["progress_persistence_errors"] = list(progress_persistence_errors)
                progress_snapshot = payload
                status = "incomplete"

    # This initial record is useful when a hard supervisor timeout arrives
    # before the first enumeration call returns.  It is intentionally one
    # write per stage boundary, so a benchmark is not dominated by I/O.
    persist_progress("initializing")
    cancel = lambda: _remaining(deadline) <= 0 or bool(progress_persistence_errors)
    topo_limits = topology_limits or EnumerationLimits()
    if topo_limits.max_seconds is None or topo_limits.max_seconds > max_seconds:
        topo_limits = EnumerationLimits(topo_limits.max_candidates, topo_limits.max_topologies,
                                        topo_limits.max_skeletons, max_seconds)
    topology_started = time.monotonic()
    persist_progress("topology_enumeration")
    topology_report = enumerate_topologies(n, loop_count, topo_limits, use_pruning=use_pruning,
                                           cancel_requested=cancel)
    stage_timings["topology_enumeration"] = max(0.0, time.monotonic() - topology_started)
    progress_counts["topologies_discovered"] = len(topology_report.topologies)
    progress_counts["topologies_total"] = (
        topology_report.unique_count if topology_report.complete else None
    )
    persist_progress("labeling_enumeration", stop_reason=_deadline_reason(topology_report.stop_reason, deadline))
    checkpoint_root: Path | None = None
    checkpoint_writer: CatalogCheckpointWriter | None = None
    checkpoint_completed_topologies = 0
    checkpoint_complete_topologies = 0
    checkpoint_last_topology_key: list[Any] | None = None
    if root is not None:
        checkpoint_root = root / "catalog" / "checkpoints"
        checkpoint_writer = CatalogCheckpointWriter(
            checkpoint_root,
            scope={"n": n, "loop_count": loop_count, "rules_schema": "F5-plan-v1",
                   "source_revision": f"sha256:{source_fingerprint()}", "baseline": "64f9a26"},
            expected_topology_keys=(topology_report.canonical_keys if topology_report.complete else None),
        )
    labeling_reports: list[CandidateSearchReport] = []
    identities_by_mode: dict[CountMode, list[Any]] = {
        CountMode.FIXED_LABELS: [], CountMode.PARTICLE_RELABELING: []
    }
    source_records_by_mode: dict[CountMode, dict[str, Any]] = {
        CountMode.FIXED_LABELS: {}, CountMode.PARTICLE_RELABELING: {}
    }
    topology_audit_by_mode: dict[CountMode, list[dict[str, Any]]] = {
        CountMode.FIXED_LABELS: [], CountMode.PARTICLE_RELABELING: []
    }
    stop_reason = _deadline_reason(topology_report.stop_reason, deadline)
    for topology in topology_report.topologies:
        if cancel():
            stop_reason = stop_reason or "deadline_exceeded"
            break
        labeling_started = time.monotonic()
        limits = labeling_limits or LabelingLimits()
        remaining = _remaining(deadline)
        if limits.max_seconds is None or limits.max_seconds > remaining:
            limits = LabelingLimits(limits.max_labelings, limits.max_results, remaining, deadline)
        else:
            limits = LabelingLimits(limits.max_labelings, limits.max_results, limits.max_seconds, deadline)
        report = enumerate_labelings(topology, count_mode=CountMode.FIXED_LABELS, limits=limits,
                                     cancel_requested=cancel, deadline=deadline)
        stage_timings["labeling_enumeration"] += max(0.0, time.monotonic() - labeling_started)
        labeling_reports.append(report)
        progress_counts["labeling_reports_completed"] = len(labeling_reports)
        persist_progress("identity_canonicalization",
                         stop_reason=_deadline_reason(report.stop_reason, deadline))
        if report.stop_reason:
            stop_reason = stop_reason or _deadline_reason(report.stop_reason, deadline)
        topology_key = tuple(canonicalize_topology(topology).key)
        topology_identity_digests: dict[CountMode, set[str]] = {
            CountMode.FIXED_LABELS: set(), CountMode.PARTICLE_RELABELING: set()
        }
        topology_identities_by_mode: dict[str, list[Any]] = {
            CountMode.FIXED_LABELS.value: [], CountMode.PARTICLE_RELABELING.value: []
        }
        topology_sources_by_mode: dict[str, dict[str, Any]] = {
            CountMode.FIXED_LABELS.value: {}, CountMode.PARTICLE_RELABELING.value: {}
        }
        identity_stage_error = False
        identity_started = time.monotonic()
        for record in report.candidate_records:
            if cancel():
                stop_reason = stop_reason or "deadline_exceeded"
                break
            try:
                for identity_mode in (CountMode.FIXED_LABELS, CountMode.PARTICLE_RELABELING):
                    if cancel():
                        stop_reason = stop_reason or "deadline_exceeded"
                        break
                    identity = canonicalize_candidate_labeling(record, identity_mode)
                    source_records = source_records_by_mode[identity_mode]
                    topology_identity_digests[identity_mode].add(identity.digest)
                    mode_name = identity_mode.value
                    if identity.digest not in topology_sources_by_mode[mode_name]:
                        topology_identities_by_mode[mode_name].append(identity)
                        topology_sources_by_mode[mode_name][identity.digest] = record
                    if identity.digest not in source_records:
                        identities_by_mode[identity_mode].append(identity)
                        source_records[identity.digest] = record
            except (TypeError, ValueError) as exc:
                failures.append({"stage": "identity", "error": str(exc), "topology": topology.to_dict()})
                identity_stage_error = True
            if stop_reason == "deadline_exceeded":
                break
        stage_timings["identity_canonicalization"] += max(0.0, time.monotonic() - identity_started)
        progress_counts["identity_counts"] = {
            CountMode.FIXED_LABELS.value: len(source_records_by_mode[CountMode.FIXED_LABELS]),
            CountMode.PARTICLE_RELABELING.value: len(source_records_by_mode[CountMode.PARTICLE_RELABELING]),
        }
        persist_progress("labeling_enumeration", stop_reason=stop_reason)
        for audit_mode in (CountMode.FIXED_LABELS, CountMode.PARTICLE_RELABELING):
            mode_digests = topology_identity_digests[audit_mode]
            topology_audit_by_mode[audit_mode].append({
                "mode": audit_mode.value,
                "topology_key": list(topology_key),
                "raw_assignment_count": report.raw_assignment_count,
                "momentum_pass_count": report.momentum_pass_count,
                "unique_identity_count": len(mode_digests),
                "stored_identity_count": len(mode_digests),
                "label_enumeration_complete": report.complete,
            })
        if checkpoint_writer is not None:
            checkpoint_started = time.monotonic()
            checkpoint_writer._append_verified_topology(
                topology_key,
                identities_by_mode=topology_identities_by_mode,
                source_records_by_mode=topology_sources_by_mode,
                label_enumeration_complete={mode: report.complete for mode in topology_identities_by_mode},
                identity_save_complete={
                    mode: (report.complete and not identity_stage_error and stop_reason is None)
                    for mode in topology_identities_by_mode
                },
                topology_complete=True,
                audit={
                    "raw_assignment_count": report.raw_assignment_count,
                    "momentum_pass_count": report.momentum_pass_count,
                    "candidate_count": len(report.candidate_records),
                    "label_enumeration_complete": report.complete,
                    "stop_reason": report.stop_reason,
                },
            )
            stage_timings["topology_checkpoint"] += max(0.0, time.monotonic() - checkpoint_started)
            checkpoint_completed_topologies += 1
            checkpoint_complete_topologies += int(
                report.complete and not identity_stage_error and stop_reason is None
            )
            checkpoint_last_topology_key = list(topology_key)
            progress_counts["checkpoint_committed_topologies"] = checkpoint_completed_topologies
            progress_counts["checkpoint_complete_topologies"] = checkpoint_complete_topologies
            progress_counts["checkpoint_completed_topologies"] = checkpoint_complete_topologies
            progress_counts["checkpoint_last_topology_key"] = checkpoint_last_topology_key
            persist_progress("labeling_enumeration", stop_reason=stop_reason)
        if stop_reason == "deadline_exceeded":
            break

    label_enumeration_complete = (
        topology_report.complete
        and len(labeling_reports) == topology_report.unique_count
        and len(labeling_reports) == len(topology_report.topologies)
        and all(report.complete for report in labeling_reports)
    )
    status = "complete" if (topology_report.complete and labeling_reports
                             and all(report.complete for report in labeling_reports)
                             and stop_reason is None and not failures
                             and not progress_persistence_errors) else "incomplete"
    if progress_persistence_errors and stop_reason is None:
        stop_reason = "progress_persistence_failed"
    if topology_report.status is RunStatus.FAILED or any(report.status is RunStatus.FAILED for report in labeling_reports):
        status = "failed"
    catalogs: dict[str, CandidateCatalog] = {}
    catalog_started = time.monotonic()
    for identity_mode in (CountMode.FIXED_LABELS, CountMode.PARTICLE_RELABELING):
        source_records = source_records_by_mode[identity_mode]
        digests = tuple(sorted(source_records))
        manifest = CatalogManifest(
            scope=_scope(n, loop_count, identity_mode, topology_report=topology_report, marker_scope="all_stored"),
            count_mode=identity_mode.value, marker_scope="all_stored", topology_scope="enumerated",
            run_status=status, topology_enumeration_complete=topology_report.complete,
             label_enumeration_complete=label_enumeration_complete,
             identity_save_complete=not failures,
            expected_identity_digests=digests, stored_identity_digests=digests,
            stop_reason=stop_reason, failures=tuple(failures),
            extra={"topology_audit": topology_audit_by_mode[identity_mode],
                   "selection": "all_stored_identities"},
        )
        catalogs[identity_mode.value] = build_catalog(identities_by_mode[identity_mode],
                                                       source_records=source_records, manifest=manifest)
    catalog = catalogs[mode.value]
    search_status = status
    # Save the candidate records before any Graphviz work.  A drawing timeout
    # must never erase the already audited S/P identities.
    if root is not None:
        persist_progress("catalog_persistence", stop_reason=stop_reason)
        if progress_persistence_errors:
            status = "incomplete"
            stop_reason = stop_reason or "progress_persistence_failed"
        else:
            for catalog_mode, mode_catalog in catalogs.items():
                write_catalog(mode_catalog, root / "catalog" / catalog_mode, overwrite=False)
        stage_timings["catalog_persistence"] += max(0.0, time.monotonic() - catalog_started)
        if not progress_persistence_errors:
            persist_progress("drawing" if render else "drawing_skipped", stop_reason=stop_reason)
    else:
        stage_timings["catalog_persistence"] += max(0.0, time.monotonic() - catalog_started)
    search_status = status
    drawing_outcomes: tuple[DrawingOutcome, ...] = ()
    drawing_complete = True
    if root is not None:
        if render and not progress_persistence_errors:
            drawing_started = time.monotonic()
            # S structures are selected from the independently enumerated
            # topology report, so a structure with no P witness is still
            # drawable and no topology is silently inferred from P.
            (root / "drawings" / "structure").mkdir(parents=True, exist_ok=True)
            (root / "drawings" / "marked").mkdir(parents=True, exist_ok=True)
            selected_topologies = tuple(zip(topology_report.canonical_keys, topology_report.topologies))
            if max_drawings is not None:
                selected_topologies = selected_topologies[:max_drawings]
            structure_outcomes = tuple(
                _render_structure_input(topology, tuple(key), root / "drawings" / "structure",
                                        dot_path=dot_path, deadline=deadline, mode=mode.value)
                for key, topology in selected_topologies
            ) if selected_topologies else ()
            marked_outcomes = render_catalog(catalog, root / "drawings" / "marked", dot_path=dot_path,
                                             max_items=max_drawings, deadline=deadline, view="marked")
            _write_index(root / "drawings" / "structure", structure_outcomes,
                         scope={"topology_scope": "enumerated", "marker_scope": "one_structure_per_topology",
                                "view": "structure", "independence": "not_evaluated"})
            _write_json(root / "drawings" / "structure" / "drawing_manifest.json",
                        {"scope": {"topology_scope": "enumerated", "marker_scope": "one_structure_per_topology",
                                   "view": "structure"},
                         "outcomes": [item.to_dict() for item in structure_outcomes]})
            _write_combined_index(root / "drawings", {"structure": structure_outcomes, "marked": marked_outcomes})
            drawing_outcomes = structure_outcomes + marked_outcomes
            stage_timings["drawing"] += max(0.0, time.monotonic() - drawing_started)
            progress_counts["drawings_requested"] = len(drawing_outcomes)
            progress_counts["drawings_completed"] = sum(
                item.status == "completed" for item in drawing_outcomes
            )
            drawing_complete = bool(drawing_outcomes) and all(item.status == "completed" for item in drawing_outcomes)
            if not selected_topologies and not catalog.entries:
                drawing_complete = topology_report.complete
            if not drawing_complete:
                status = "incomplete" if status == "complete" else status
        elif render:
            drawing_complete = False
        for catalog_mode, mode_catalog in tuple(catalogs.items()):
            manifest = replace(mode_catalog.manifest, run_status=search_status,
                               drawing_selection_complete=drawing_complete if catalog_mode == mode.value else False)
            catalogs[catalog_mode] = CandidateCatalog(manifest, mode_catalog.entries, mode_catalog.source_path)
        catalog = catalogs[mode.value]
        catalog_started = time.monotonic()
        for catalog_mode, mode_catalog in catalogs.items():
            write_catalog(mode_catalog, root / "catalog" / catalog_mode, overwrite=True)
        stage_timings["catalog_persistence"] += max(0.0, time.monotonic() - catalog_started)
        stage_timings["total"] = max(0.0, time.monotonic() - run_started)
        persist_progress("finished", progress_status=status, stop_reason=stop_reason)
        if progress_persistence_errors:
            status = "incomplete"
            stop_reason = stop_reason or "progress_persistence_failed"
            search_status = status
            for catalog_mode, mode_catalog in tuple(catalogs.items()):
                manifest = replace(mode_catalog.manifest, run_status="incomplete",
                                   stop_reason=stop_reason,
                                   failures=tuple(failures))
                catalogs[catalog_mode] = CandidateCatalog(manifest, mode_catalog.entries,
                                                          mode_catalog.source_path)
            catalog = catalogs[mode.value]
            for catalog_mode, mode_catalog in catalogs.items():
                write_catalog(mode_catalog, root / "catalog" / catalog_mode, overwrite=True)
        report_started = time.monotonic()
        checkpoint_info = {
            "path": str(checkpoint_root) if checkpoint_root is not None else None,
            "data_format": "gzip-member-jsonl-v1",
            "scope_complete": topology_report.complete,
            "committed_topologies": checkpoint_completed_topologies,
            "complete_topologies": checkpoint_complete_topologies,
            "completed_topologies": checkpoint_complete_topologies,
            "expected_topologies": topology_report.unique_count if topology_report.complete else None,
        }
        _write_json(root / "run_manifest.json", {
            "status": status, "stop_reason": stop_reason,
            "run_context": _run_context(dot_path=dot_path, max_seconds=max_seconds,
                                         topology_limits=topo_limits, render=render),
            "topology_report": topology_report.to_dict(),
            "labeling_reports": [report.to_dict() for report in labeling_reports],
            "catalogs": {key: value.manifest.to_dict() for key, value in catalogs.items()},
            "drawings": [outcome.to_dict() for outcome in drawing_outcomes],
            "stage_timings": dict(stage_timings),
            "timings": dict(stage_timings),
            "progress": dict(progress_snapshot),
            "benchmark": benchmark,
            "checkpoint": checkpoint_info,
        })
        stage_timings["report_persistence"] += max(0.0, time.monotonic() - report_started)
        stage_timings["total"] = max(0.0, time.monotonic() - run_started)
        # The finished sidecar is committed before run_manifest.  Do not add
        # a second replace after the manifest: a late sharing-lock failure
        # must not leave a complete manifest beside an incomplete sidecar.
    else:
        # Keep the in-memory report useful for callers that deliberately run
        # without persistence (e.g. a microbenchmark).
        stage_timings["total"] = max(0.0, time.monotonic() - run_started)
        persist_progress("finished", progress_status=status, stop_reason=stop_reason)
        checkpoint_info = {
            "path": None,
            "data_format": "gzip-member-jsonl-v1",
            "scope_complete": topology_report.complete,
            "committed_topologies": 0,
            "complete_topologies": 0,
            "completed_topologies": 0,
            "expected_topologies": topology_report.unique_count if topology_report.complete else None,
        }
    finished = _timestamp()
    return CandidateRunReport(status, n, loop_count, mode.value, started, finished, max_seconds,
                              stop_reason, topology_report, tuple(labeling_reports), catalog, catalogs,
                              drawing_outcomes, tuple(failures), str(root) if root else None,
                              dict(stage_timings), dict(progress_snapshot), benchmark,
                              str(checkpoint_root) if checkpoint_root is not None else None)


def write_run_report(report: CandidateRunReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json() + "\n", encoding="utf-8")
    return path


__all__ = ["DrawingOutcome", "CandidateRunReport", "render_catalog", "run_candidates", "write_run_report", "source_fingerprint"]
