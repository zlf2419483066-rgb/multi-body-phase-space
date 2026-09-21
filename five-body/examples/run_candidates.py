"""Reproducible bounded five-body candidate and drawing runner.

Examples::

    python examples/run_candidates.py --fixture A
    python examples/run_candidates.py --fixture B2
    python examples/run_candidates.py --loop-count 0 --max-seconds 30

The fixture A path deliberately exports all 60 fixed-label identities.  The
fixture B2 path draws the repeated ``p345`` double-line variable twice.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import itertools
import json
import math
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_space_graph5.domain import FinalParticleLabel, Labeling
from phase_space_graph5.fixtures import fixture_a, fixture_b
from phase_space_graph5.label_catalog import CatalogManifest, build_catalog, write_catalog
from phase_space_graph5.domain import CountMode
from phase_space_graph5.label_identity import canonicalize_candidate_labeling
from phase_space_graph5.physical_labels import normalize_candidate_labeling
from phase_space_graph5.reporting import render_catalog, run_candidates, source_fingerprint, write_run_report
from phase_space_graph5.compact_reporting import run_compact_candidates
from phase_space_graph5.data_paths import default_output


_CLI_SCHEMA_VERSION = "F5-cli-worker-v1"
# Keep 540 s as the CLI default for short, reproducible calls while allowing
# an overnight candidate run to use the full 20-minute wall-clock budget.
_DEFAULT_MAX_SECONDS = 540.0
_HARD_BUDGET_SECONDS = 1200.0
_CHECKPOINT_RESERVE_SECONDS = 1.0
_HARD_TIMEOUT_EXIT = 124


def _prepare_worker_output(requested: Path) -> Path:
    """Choose one fresh run directory without touching an older checkpoint."""

    output = Path(requested).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        child = output / f"run-{time.time_ns()}-{os.getpid()}"
        child.mkdir()
        return child
    return output


def _checkpoint_paths(output: Path) -> list[str]:
    candidates = (
        "worker_status.json", "run_progress.json", "fixture_manifest.json", "run_manifest.json", "run_report.json",
        "catalog/fixed_labels/manifest.json", "catalog/particle_relabeling/manifest.json",
        "catalog/checkpoints/scope.json", "catalog/checkpoints/index.jsonl", "catalog/checkpoints/data.jsonl.gz",
    )
    return [name for name in candidates if (output / name).is_file()]


def _read_progress(output: Path) -> dict[str, object] | None:
    path = output / "run_progress.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _mark_timeout_progress(output: Path) -> dict[str, object] | None:
    """Turn the last durable running snapshot into an honest incomplete one."""

    progress = _read_progress(output)
    if progress is None:
        return None
    progress["status"] = "incomplete"
    progress["stage"] = progress.get("stage", "unknown")
    progress["stop_reason"] = "deadline_exceeded"
    progress["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    temporary = output / f".run_progress.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output / "run_progress.json")
    except OSError:
        return progress
    return progress


def _write_worker_status(output: Path, *, status: str, stop_reason: str | None,
                         requested_seconds: float, worker_seconds: float,
                         started_at: float, worker_pid: int | None,
                         worker_exit_code: int | None, error: str | None = None,
                         run_status: str | None = None,
                         run_stop_reason: str | None = None) -> None:
    payload = {
        "schema_version": _CLI_SCHEMA_VERSION,
        "status": status,
        "process_status": status,
        "stop_reason": stop_reason,
        "run_status": run_status,
        "run_stop_reason": run_stop_reason,
        "requested_max_seconds": requested_seconds,
        "worker_budget_seconds": worker_seconds,
        "hard_budget_seconds": _HARD_BUDGET_SECONDS,
        "elapsed_seconds": max(0.0, time.monotonic() - started_at),
        "worker_pid": worker_pid,
        "worker_exit_code": worker_exit_code,
        "run_directory": str(output),
        "checkpoint_paths": _checkpoint_paths(output),
        "error": error,
    }
    progress = _read_progress(output)
    if progress is not None:
        payload["progress"] = progress
        timings = progress.get("stage_timings")
        if isinstance(timings, dict):
            payload["stage_timings"] = timings
    target = output / "worker_status.json"
    temporary = output / f".worker_status.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def _terminate_worker(process: subprocess.Popen[bytes]) -> None:
    if sys.platform == "win32":
        try:
            subprocess.run(("taskkill", "/PID", str(process.pid), "/T", "/F"),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False, timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass


def _actual_run_status(output: Path) -> tuple[str | None, str | None]:
    """Read the worker's own status without treating process exit as run success."""

    for name in ("fixture_manifest.json", "run_manifest.json", "run_report.json",
                 "catalog/fixed_labels/manifest.json", "catalog/particle_relabeling/manifest.json"):
        path = output / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data.get("status"), data.get("stop_reason")
    return None, None


def _worker_argv(args: argparse.Namespace, output: Path, worker_seconds: float) -> list[str]:
    command = [sys.executable, str(HERE), "--_worker", "--max-seconds", repr(worker_seconds),
               "--output", str(output), "--mode", args.mode, "--loop-count", str(args.loop_count)]
    if args.fixture:
        command.extend(("--fixture", args.fixture))
    if args.dot is not None:
        command.extend(("--dot", args.dot))
    if args.max_drawings is not None:
        command.extend(("--max-drawings", str(args.max_drawings)))
    if args.no_render:
        command.append("--no-render")
    if args.compact:
        command.append("--compact")
    if args.label_strategy != "default":
        command.extend(("--label-strategy", args.label_strategy))
    if getattr(args, "flow_particle_orbit", False):
        command.append("--flow-particle-orbit")
    return command


def _run_outer_worker(args: argparse.Namespace) -> int:
    requested = float(args.max_seconds)
    output = _prepare_worker_output(args.output)
    started = time.monotonic()
    # Keep a small parent-side window for killing the worker and persisting a
    # truthful timeout record.  The worker receives the reduced cooperative
    # budget, so its normal checkpoint writes finish before the hard deadline.
    reserve = min(_CHECKPOINT_RESERVE_SECONDS, requested * 0.25)
    worker_seconds = max(0.0, requested - reserve)
    command = _worker_argv(args, output, worker_seconds)
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process: subprocess.Popen[bytes] | None = None
    try:
        popen_kwargs = {
            "cwd": str(HERE.parents[1]),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": None,
            "creationflags": creationflags,
        }
        if sys.platform != "win32":
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **popen_kwargs)
        # A zero budget still gets a short startup window to emit its empty,
        # deadline-exceeded checkpoint.  Positive budgets are hard bounded by
        # the caller's requested duration.
        wait_seconds = (max(1.0, requested) if requested == 0.0
                        else max(0.0, requested - (time.monotonic() - started)))
        try:
            process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            _terminate_worker(process)
            _mark_timeout_progress(output)
            run_status, run_stop_reason = _actual_run_status(output)
            # A hard kill can arrive before the child has emitted its final
            # run manifest.  The supervisor still has enough evidence to
            # report the run as incomplete and deadline-exceeded.
            # A hard kill always wins over any stale complete manifest found
            # in the run directory.  Committed blocks may remain, but the
            # run as a whole is incomplete at this boundary.
            run_status = "incomplete"
            run_stop_reason = "deadline_exceeded"
            _write_worker_status(output, status="hard_timeout", stop_reason="deadline_exceeded",
                                 requested_seconds=requested, worker_seconds=worker_seconds,
                                 started_at=started, worker_pid=process.pid,
                                 worker_exit_code=process.returncode,
                                 run_status=run_status, run_stop_reason=run_stop_reason)
            return _HARD_TIMEOUT_EXIT
        exit_code = process.returncode
        run_status, run_stop_reason = _actual_run_status(output)
        if exit_code == 0:
            _write_worker_status(output, status="completed", stop_reason=None,
                                 requested_seconds=requested, worker_seconds=worker_seconds,
                                 started_at=started, worker_pid=process.pid,
                                 worker_exit_code=exit_code,
                                 run_status=run_status, run_stop_reason=run_stop_reason)
            return 0
        if run_status == "complete":
            # Never let a non-zero worker exit preserve a complete status
            # from an earlier sidecar or a partially written final report.
            run_status = "incomplete"
        run_stop_reason = run_stop_reason or "worker_failed"
        _write_worker_status(output, status="failed", stop_reason="worker_failed",
                             requested_seconds=requested, worker_seconds=worker_seconds,
                             started_at=started, worker_pid=process.pid,
                             worker_exit_code=exit_code,
                             run_status=run_status, run_stop_reason=run_stop_reason)
        return 1
    except (OSError, ValueError) as exc:
        failed_run_status, failed_run_reason = _actual_run_status(output)
        if failed_run_status == "complete":
            failed_run_status = "incomplete"
            failed_run_reason = failed_run_reason or "worker_start_failed"
        _write_worker_status(output, status="failed", stop_reason="worker_start_failed",
                             requested_seconds=requested, worker_seconds=worker_seconds,
                             started_at=started, worker_pid=process.pid if process else None,
                             worker_exit_code=None, error=str(exc),
                             run_status=failed_run_status,
                             run_stop_reason=failed_run_reason)
        return 1


def _fixture_records(name: str, *, deadline: float | None = None):
    identities_by_mode = {"fixed_labels": [], "particle_relabeling": []}
    sources_by_mode = {"fixed_labels": {}, "particle_relabeling": {}}
    stop_reason = "deadline_exceeded" if deadline is not None and time.monotonic() >= deadline else None
    raw_assignment_count = 0
    if name == "A":
        topology, _ = fixture_a()
        slots = tuple((vertex, slot) for vertex, count in topology.final_slots for slot in range(count))
        identities = []
        sources = {}
        for permutation in itertools.permutations(range(1, 6)):
            if deadline is not None and time.monotonic() >= deadline:
                stop_reason = "deadline_exceeded"
                break
            raw_assignment_count += 1
            labeling = Labeling(tuple(FinalParticleLabel(vertex, slot, particle)
                                      for (vertex, slot), particle in zip(slots, permutation)))
            checked = normalize_candidate_labeling(topology, labeling)
            if checked.record is None:
                raise RuntimeError(f"fixture A labeling unexpectedly failed: {checked.to_dict()}")
            identity = canonicalize_candidate_labeling(checked.record, "fixed_labels")
            if identity.digest not in sources:
                identities.append(identity)
                sources[identity.digest] = checked.record
            for mode in ("fixed_labels", "particle_relabeling"):
                relabeled = canonicalize_candidate_labeling(checked.record, mode)
                if relabeled.digest not in sources_by_mode[mode]:
                    identities_by_mode[mode].append(relabeled)
                    sources_by_mode[mode][relabeled.digest] = checked.record
        expected = 60
        if stop_reason is None and len(sources) != expected:
            raise RuntimeError(f"fixture A expected {expected} fixed identities, got {len(sources)}")
    elif name == "B2" and stop_reason is None:
        topology, labeling = fixture_b(2)
        checked = normalize_candidate_labeling(topology, labeling)
        if checked.record is None:
            raise RuntimeError(f"fixture B(a=2) unexpectedly failed: {checked.to_dict()}")
        raw_assignment_count = 1
        identity = canonicalize_candidate_labeling(checked.record, "fixed_labels")
        identities, sources = [identity], {identity.digest: checked.record}
        for mode in ("fixed_labels", "particle_relabeling"):
            relabeled = canonicalize_candidate_labeling(checked.record, mode)
            identities_by_mode[mode] = [relabeled]
            sources_by_mode[mode] = {relabeled.digest: checked.record}
    elif name != "B2":
        raise ValueError(name)
    run_status = "incomplete" if stop_reason else "complete"
    labels_complete = name == "A" and not stop_reason
    catalogs = {}
    for mode in ("fixed_labels", "particle_relabeling"):
        mode_digests = tuple(sorted(sources_by_mode[mode]))
        manifest = CatalogManifest(scope={"fixture": name, "selection": "all_labels" if labels_complete else "single_selected_label"}, count_mode=mode,
                                   marker_scope="all_stored" if labels_complete else "specified",
                                   topology_scope="fixture",
                                   run_status=run_status, topology_enumeration_complete=not stop_reason,
                                   label_enumeration_complete=labels_complete, identity_save_complete=not stop_reason,
                                   expected_identity_digests=mode_digests, stored_identity_digests=mode_digests,
                                   stop_reason=stop_reason,
                                   extra={"selection": "all_labels" if labels_complete else "single_selected_label",
                                          "topology_audit": [{"topology_scope": "fixture",
                                                               "raw_assignment_count": raw_assignment_count,
                                                               "momentum_pass_count": raw_assignment_count,
                                                               "unique_identity_count": len(mode_digests),
                                                               "stored_identity_count": len(mode_digests),
                                                               "label_enumeration_complete": labels_complete}]})
        catalogs[mode] = build_catalog(identities_by_mode[mode], source_records=sources_by_mode[mode], manifest=manifest)
    return catalogs, stop_reason


def _run_worker(args: argparse.Namespace) -> int:
    """Run the search in the child process selected by the outer supervisor."""

    if args.fixture:
        deadline = time.monotonic() + args.max_seconds
        catalogs, stop_reason = _fixture_records(args.fixture, deadline=deadline)
        output = args.output
        if output.exists() and any(output.iterdir()):
            output = output / f"run-{time.time_ns()}"
        output.mkdir(parents=True, exist_ok=True)
        # Persist the identity checkpoint before Graphviz work.  If the outer
        # worker is terminated while drawing, the search result remains usable.
        for mode, value in catalogs.items():
            write_catalog(value, output / "catalog" / mode)
        catalog = catalogs[args.mode]
        if not args.no_render:
            outcomes = render_catalog(catalog, output / "drawings", dot_path=args.dot,
                                      max_items=args.max_drawings, view="marked", deadline=deadline)
        else:
            outcomes = ()
        drawing_complete = (not outcomes) if args.no_render else bool(outcomes) and all(
            item.status == "completed" for item in outcomes
        )
        fixture_status = "incomplete" if stop_reason or (not args.no_render and not drawing_complete) else "complete"
        catalogs = {
            mode: type(value)(replace(value.manifest, run_status=fixture_status,
                                      identity_save_complete=value.manifest.identity_save_complete and fixture_status == "complete",
                                      drawing_selection_complete=(drawing_complete if mode == args.mode else False)),
                              value.entries, value.source_path)
            for mode, value in catalogs.items()
        }
        catalog = catalogs[args.mode]
        for mode, value in catalogs.items():
            write_catalog(value, output / "catalog" / mode, overwrite=True)
        (output / "fixture_manifest.json").write_text(
            json.dumps({"fixture": args.fixture, "status": fixture_status,
                        "stop_reason": stop_reason, "deadline_seconds": args.max_seconds,
                        "run_context": {"rules_schema": "F5-plan-v1",
                                         "catalog_schema": "F5-candidate-catalog-v1",
                                         "identity_schema": "IM-5-candidate-v1",
                                         "source_revision": f"sha256:{source_fingerprint()}",
                                         "baseline": "64f9a26",
                                         "environment": {"python": sys.version.split()[0],
                                                         "platform": platform.platform(),
                                                         "graphviz_dot": args.dot or "auto"},
                                         "limits": {"max_seconds": args.max_seconds,
                                                    "dot_seconds": 60.0,
                                                    "marker_max": args.max_drawings}},
                        "catalogs": {key: value.manifest.to_dict() for key, value in catalogs.items()},
                        "catalog": catalog.manifest.to_dict(),
                        "drawings": [item.to_dict() for item in outcomes],
                        "independence": {"status": "not_evaluated", "reason": "criterion_deferred"}},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if fixture_status in {"complete", "incomplete"} else 1

    if args.compact:
        report = run_compact_candidates(loop_count=args.loop_count, max_seconds=args.max_seconds,
                                        output_dir=args.output, label_strategy=args.label_strategy,
                                        flow_particle_orbit=getattr(args, "flow_particle_orbit", False))
        # run_compact_candidates persists its explicit compact schema itself;
        # this call is intentionally not routed through write_run_report,
        # whose ordinary report schema includes raw candidate records.
        if report.status == "incomplete" and report.stop_reason in {
            "progress_persistence_failed", "checkpoint_persistence_failed",
            "checkpoint_writer_invalid", "compact_report_persistence_failed",
        }:
            # A persistence failure must reach the outer supervisor as a
            # process failure.  Otherwise it could read an older complete
            # sidecar and report success after the new run lost its marker.
            return 1
        return 0 if report.status in {"complete", "incomplete"} else 1
    report = run_candidates(loop_count=args.loop_count, max_seconds=args.max_seconds,
                            count_mode=args.mode,
                            output_dir=args.output, dot_path=args.dot, render=not args.no_render,
                            max_drawings=args.max_drawings)
    write_run_report(report, Path(report.output_directory or args.output) / "run_report.json")
    return 0 if report.status in {"complete", "incomplete"} else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("A", "B2"), help="run a hand-written fixture")
    parser.add_argument("--loop-count", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=_DEFAULT_MAX_SECONDS)
    parser.add_argument("--max-drawings", type=int)
    parser.add_argument(
        "--output", type=Path, default=default_output("runs", "serial"),
        help="output directory (default: PSF_DATA_ROOT/Demo案例/5体/runs/serial)",
    )
    parser.add_argument("--dot", type=str, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--mode", choices=("fixed_labels", "particle_relabeling"), default="fixed_labels")
    parser.add_argument("--compact", action="store_true",
                        help="opt into checkpoint-only compact reporting (both identity modes are saved)")
    parser.add_argument("--label-strategy", choices=("default", "flow"), default="default",
                        help="label strategy for --compact; flow uses scalar-flow legal label generation")
    parser.add_argument("--flow-particle-orbit", action="store_true",
                        help="opt into flow particle-orbit certificate reuse; requires --compact --label-strategy flow")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds < 0 or args.max_seconds > _HARD_BUDGET_SECONDS:
        parser.error(f"--max-seconds must be finite and between 0 and {_HARD_BUDGET_SECONDS:g} seconds")
    if args.max_drawings is not None and args.max_drawings < 0:
        parser.error("--max-drawings must be a non-negative integer")
    if args.fixture and args.compact:
        parser.error("--compact is available for enumerated searches, not fixtures")
    if args.label_strategy == "flow" and not args.compact:
        parser.error("--label-strategy flow requires --compact")
    if args.flow_particle_orbit and not args.compact:
        parser.error("--flow-particle-orbit requires --compact")
    if args.flow_particle_orbit and args.label_strategy != "flow":
        parser.error("--flow-particle-orbit requires --label-strategy flow")
    if args._worker:
        return _run_worker(args)
    return _run_outer_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
