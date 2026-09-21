"""Four-process checkpoint orchestration for five-body candidate searches.

This module is an opt-in coordinator.  The ordinary reporting and compact
paths remain unchanged.  The parent enumerates the complete topology scope
once, partitions the immutable canonical topology keys, and gives each worker
only its assigned source representatives.  Workers use the existing labeling,
physical normalization, identity, and checkpoint APIs; this module does not
define any physical or canonical rules.

The manifest verifies committed index coverage and shard disjointness without
claiming strict source/map readback.  That expensive strict readback is an
independent follow-up operation and is recorded as ``not_run`` here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping

from .canonical import canonicalize_topology
from .catalog_checkpoint import (CHECKPOINT_SCHEMA_VERSION, DATA_FORMAT, MODE_NAMES,
                                 CatalogCheckpointWriter)
from .compact_reporting import _identity_mode_lists
from .domain import CountMode, RunStatus, Topology
from .enumeration import EnumerationLimits, enumerate_topologies
from .flow_labeling import enumerate_flow_labelings
from .labeling import LabelingLimits, enumerate_labelings


_SCHEMA_VERSION = "F5-parallel-run-v1"
_MAX_BUDGET_SECONDS = 1200.0
_DEFAULT_MAX_SECONDS = 540.0
_DEFAULT_WORKERS = 4
_CLEANUP_SECONDS = 5.0


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


def _token(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _validate_budget(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("max_seconds must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or value > _MAX_BUDGET_SECONDS:
        raise ValueError(f"max_seconds must be between 0 and {_MAX_BUDGET_SECONDS:g}")
    return value


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    files = sorted(
        path for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    return {
        path.relative_to(root).as_posix(): _sha256(path.read_bytes())
        for path in files
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(_jsonable(payload), ensure_ascii=False,
                                    sort_keys=True, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _fresh_output(path: str | Path) -> Path:
    root = Path(path).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"output path is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        root = root / f"run-{time.time_ns()}-{os.getpid()}"
        root.mkdir()
    return root


def _partition_scope(loop_count: int, keys: list[Any], *, worker_count: int,
                     worker_index: int | None, source_hashes: Mapping[str, str],
                     label_strategy: str, use_particle_orbit: bool) -> dict[str, Any]:
    return {
        "n": 5,
        "loop_count": loop_count,
        "runner_mode": "parallel_checkpoint",
        "label_strategy": label_strategy,
        "use_particle_orbit": use_particle_orbit,
        "partition_count": worker_count,
        "partition_index": worker_index,
        "expected_topology_count": len(keys),
        "source_hashes": dict(source_hashes),
    }


def _topology_payload(topology: Topology, key: Any) -> dict[str, Any]:
    return {"topology": topology.to_dict(), "topology_key": _jsonable(key)}


def _run_one_topology(item: Mapping[str, Any], writer: CatalogCheckpointWriter,
                      *, deadline: float, label_strategy: str,
                      use_particle_orbit: bool) -> dict[str, Any]:
    topology = Topology.from_dict(item["topology"])
    topology_key = tuple(item["topology_key"])
    if time.monotonic() >= deadline:
        raise TimeoutError("deadline_exceeded_before_topology")
    remaining = max(0.0, deadline - time.monotonic())
    limits = LabelingLimits(max_seconds=remaining, deadline=deadline)
    cancelled = lambda: time.monotonic() >= deadline
    if label_strategy == "flow":
        report = enumerate_flow_labelings(
            topology, limits=limits, cancel_requested=cancelled,
            deadline=deadline, record_order="reference",
            use_particle_orbit=use_particle_orbit,
        )
    else:
        report = enumerate_labelings(
            topology, count_mode=CountMode.FIXED_LABELS, limits=limits,
            cancel_requested=cancelled, deadline=deadline,
            use_symmetry=False,
        )
    report_status = getattr(report.status, "value", report.status)
    report_failures = dict(getattr(report, "failure_counts", ()))
    identity_stop = report.stop_reason
    identities, sources, identity_failures, identity_failed, identity_stop, prekey_stats = _identity_mode_lists(
        report, cancel_requested=cancelled, deadline=deadline,
        failure_is_error=(label_strategy == "flow"),
    )
    report_failed = report_status == RunStatus.FAILED.value or (
        label_strategy == "flow" and bool(report_failures)
    )
    stop_reason = identity_stop or report.stop_reason
    topology_complete = bool(
        report.complete and not report_failed and not identity_failed
        and not identity_failures and stop_reason is None and not cancelled()
    )
    flow_counts = {
        name: getattr(report, name, None)
        for name in (
            "traditional_search_space_size", "flow_partition_generated_labelings",
            "flow_partition_normalized_labelings", "flow_partition_branch_count",
        )
    }
    representative_evaluations = getattr(report, "representative_evaluations", 0)
    certificate_transports = getattr(report, "certificate_transports", 0)
    particle_orbit_reuse = bool(getattr(report, "particle_orbit_reuse", False))
    audit = {
        "strategy": label_strategy,
        "flow_record_order": "reference" if label_strategy == "flow" else None,
        "status": "complete" if topology_complete else "incomplete",
        "complete": topology_complete,
        "stop_reason": stop_reason,
        "raw_assignment_count": report.raw_assignment_count,
        "momentum_pass_count": report.momentum_pass_count,
        "candidate_count": len(report.candidate_records),
        "raw_assignment_count_semantics": (
            "flow_partition_generated_labelings" if label_strategy == "flow"
            else "legacy_raw_assignments"
        ),
        "flow_counts": flow_counts,
        "representative_evaluations": representative_evaluations,
        "certificate_transports": certificate_transports,
        "particle_orbit_reuse": particle_orbit_reuse,
        "failure_counts": report_failures,
        "failure_sample_codes": [sample.code for sample in report.failure_samples],
        "identity_counts": {mode: len(values) for mode, values in identities.items()},
        "identity_prekey_stats": prekey_stats,
        "identity_error_count": len(identity_failures),
        "raw_records_retained": 0,
        "p_witness_present": report.momentum_pass_count > 0,
    }
    writer._append_verified_topology(
        topology_key,
        identities_by_mode=identities,
        source_records_by_mode=sources,
        label_enumeration_complete={mode: bool(report.complete and not report_failed and not identity_failed)
                                    for mode in MODE_NAMES},
        identity_save_complete={mode: bool(topology_complete) for mode in MODE_NAMES},
        topology_complete=topology_complete,
        audit=audit,
    )
    return {
        "topology_key": _jsonable(topology_key),
        "raw_assignment_count": int(report.raw_assignment_count),
        "momentum_pass_count": int(report.momentum_pass_count),
        "candidate_count": len(report.candidate_records),
        "identity_counts": audit["identity_counts"],
        "complete": topology_complete,
        "stop_reason": stop_reason,
        "flow_counts": flow_counts,
        "representative_evaluations": representative_evaluations,
        "certificate_transports": certificate_transports,
        "particle_orbit_reuse": particle_orbit_reuse,
    }


def _process_snapshot_raw(pid: int | None, *, known_descendants: list[Mapping[str, Any]] | None = None,
                          known_create_time: float | None = None) -> dict[str, Any]:
    """Return observable process/tree liveness evidence for cleanup reports."""
    if pid is None:
        return {"psutil_available": False, "pid": None,
                "pid_alive": False, "descendant_pids": [],
                "descendant_alive": [], "descendant_records": [],
                "snapshot_complete": False, "error": "missing_pid"}
    try:
        import psutil  # type: ignore
    except ImportError:
        return {"psutil_available": False, "pid": pid,
                "pid_alive": None, "descendant_pids": [],
                "descendant_alive": [], "descendant_records": [],
                "snapshot_complete": False, "error": "psutil_unavailable"}

    def probe(candidate_pid: int, create_time: float | None) -> bool | None:
        try:
            candidate = psutil.Process(candidate_pid)
            if create_time is not None:
                try:
                    if abs(candidate.create_time() - create_time) > 1e-3:
                        return False
                except psutil.NoSuchProcess:
                    return False
                except Exception:
                    return None
            try:
                return bool(candidate.is_running() and candidate.status() != psutil.STATUS_ZOMBIE)
            except psutil.NoSuchProcess:
                return False
            except Exception:
                return None
        except psutil.NoSuchProcess:
            return False
        except Exception:
            return None
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        try:
            records = [dict(item) for item in (known_descendants or ())]
            pids = [int(item["pid"]) for item in records]
            alive = [probe(item_pid, item.get("create_time"))
                     for item_pid, item in zip(pids, records)]
        except Exception as exc:
            return {"psutil_available": True, "pid": pid,
                    "pid_alive": False, "descendant_pids": [],
                    "descendant_alive": [], "descendant_records": [],
                    "snapshot_complete": False,
                    "error": f"{type(exc).__name__}:{exc}"}
        return {"psutil_available": True, "pid": pid,
                "pid_alive": False, "descendant_pids": pids,
                "descendant_alive": alive, "descendant_records": records,
                "snapshot_complete": known_descendants is not None
                and all(value is not None for value in alive)}
    except Exception as exc:
        return {"psutil_available": True, "pid": pid,
                "pid_alive": None, "descendant_pids": [],
                "descendant_alive": [], "descendant_records": [],
                "snapshot_complete": False, "error": f"{type(exc).__name__}:{exc}"}
    if known_descendants is None:
        try:
            descendants = process.children(recursive=True)
            records = []
            for child in descendants:
                try:
                    records.append({"pid": child.pid, "create_time": child.create_time()})
                except psutil.NoSuchProcess:
                    return {"psutil_available": True, "pid": pid,
                            "pid_alive": probe(pid, known_create_time), "descendant_pids": [],
                            "descendant_alive": [], "descendant_records": [],
                            "snapshot_complete": False,
                            "error": "descendant_exited_during_snapshot"}
                except Exception as exc:
                    return {"psutil_available": True, "pid": pid,
                            "pid_alive": None, "descendant_pids": [],
                            "descendant_alive": [], "descendant_records": [],
                            "snapshot_complete": False,
                            "error": f"{type(exc).__name__}:{exc}"}
        except psutil.NoSuchProcess:
            return {"psutil_available": True, "pid": pid,
                    "pid_alive": False, "descendant_pids": [],
                    "descendant_alive": [], "descendant_records": [],
                    "snapshot_complete": False,
                    "error": "root_exited_during_snapshot"}
        except Exception as exc:
            return {"psutil_available": True, "pid": pid,
                    "pid_alive": None, "descendant_pids": [],
                    "descendant_alive": [], "descendant_records": [],
                    "snapshot_complete": False,
                    "error": f"{type(exc).__name__}:{exc}"}
    else:
        try:
            records = [dict(item) for item in known_descendants]
        except Exception as exc:
            return {"psutil_available": True, "pid": pid,
                    "pid_alive": None, "descendant_pids": [],
                    "descendant_alive": [], "descendant_records": [],
                    "snapshot_complete": False,
                    "error": f"{type(exc).__name__}:{exc}"}
    try:
        descendant_pids = [int(item["pid"]) for item in records]
    except Exception as exc:
        return {"psutil_available": True, "pid": pid,
                "pid_alive": None, "descendant_pids": [],
                "descendant_alive": [], "descendant_records": records,
                "snapshot_complete": False,
                "error": f"{type(exc).__name__}:{exc}"}
    alive = [probe(pid_value, item.get("create_time"))
             for pid_value, item in zip(descendant_pids, records)]
    try:
        create_time = (known_create_time if known_create_time is not None
                       else process.create_time())
    except psutil.NoSuchProcess:
        return {"psutil_available": True, "pid": pid,
                "pid_alive": False, "descendant_pids": descendant_pids,
                "descendant_alive": alive, "descendant_records": records,
                "snapshot_complete": False,
                "error": "root_exited_during_snapshot"}
    except Exception as exc:
        return {"psutil_available": True, "pid": pid,
                "pid_alive": None, "descendant_pids": descendant_pids,
                "descendant_alive": alive, "descendant_records": records,
                "snapshot_complete": False,
                "error": f"{type(exc).__name__}:{exc}"}
    try:
        pid_alive = probe(pid, create_time)
    except psutil.NoSuchProcess:
        pid_alive = False
    return {"psutil_available": True, "pid": pid,
            "pid_alive": pid_alive,
            "create_time": create_time,
            "descendant_pids": descendant_pids,
            "descendant_alive": alive,
            "descendant_records": records,
            "snapshot_complete": pid_alive is not None
            and all(value is not None for value in alive)}


def _process_handle_state(process_handle: Any) -> tuple[bool | None, int | None]:
    if process_handle is None:
        return None, None
    try:
        if hasattr(process_handle, "poll"):
            return process_handle.poll() is not None, process_handle.returncode
        if hasattr(process_handle, "is_alive"):
            alive = bool(process_handle.is_alive())
            return not alive, getattr(process_handle, "exitcode", None)
    except Exception:
        return None, None
    return None, None


def _process_snapshot(pid: int | None, *, known_descendants: list[Mapping[str, Any]] | None = None,
                      known_create_time: float | None = None,
                      process_handle: Any = None) -> dict[str, Any]:
    snapshot = _process_snapshot_raw(
        pid, known_descendants=known_descendants, known_create_time=known_create_time,
    )
    process_exited, process_returncode = _process_handle_state(process_handle)
    snapshot["pid_visible"] = snapshot.get("pid_alive")
    snapshot["process_exited"] = process_exited
    snapshot["process_returncode"] = process_returncode
    if process_exited is True:
        snapshot["pid_alive"] = False
    elif process_exited is False:
        snapshot["pid_alive"] = True
    if process_handle is not None:
        snapshot["snapshot_complete"] = (
            snapshot.get("snapshot_complete") is True
            and snapshot.get("pid_alive") is not None
        )
    return snapshot


def _terminate_tree(process: mp.Process) -> dict[str, Any]:
    """Stop a worker tree and return command plus before/after evidence."""
    pid = process.pid
    evidence: dict[str, Any] = {
        "pid": pid,
        "before": None,
        "taskkill_returncode": None,
        "taskkill_error": None,
        "fallback_process_kill": False,
    }
    try:
        evidence["before"] = _process_snapshot(pid, process_handle=process)
    except BaseException as exc:
        evidence["before"] = {"psutil_available": False, "pid": pid,
                               "pid_alive": None, "descendant_pids": [],
                               "descendant_alive": [], "descendant_records": [],
                               "snapshot_complete": False,
                               "error": f"{type(exc).__name__}:{exc}"}
    if process.pid is None:
        evidence["after"] = _process_snapshot(None)
        evidence["all_stopped"] = False
        return evidence
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False, capture_output=True, text=True,
                timeout=_CLEANUP_SECONDS,
            )
            evidence["taskkill_returncode"] = result.returncode
            if result.stderr:
                evidence["taskkill_error"] = result.stderr.strip()
            elif result.returncode != 0:
                evidence["taskkill_error"] = f"taskkill_returncode:{result.returncode}"
        except Exception as exc:
            evidence["taskkill_error"] = f"{type(exc).__name__}:{exc}"
    else:
        try:
            os.kill(process.pid, 9)
        except ProcessLookupError:
            pass
        except OSError as exc:
            evidence["taskkill_error"] = f"{type(exc).__name__}:{exc}"
    try:
        process.join(timeout=2.0)
    except Exception:
        pass
    if process.is_alive():
        evidence["fallback_process_kill"] = True
        try:
            process.kill()
        except Exception:
            process.terminate()
        process.join(timeout=1.0)
    before = evidence["before"]
    try:
        evidence["after"] = _process_snapshot(
            pid,
            known_descendants=before.get("descendant_records")
            if isinstance(before, Mapping) and before.get("snapshot_complete") is True else None,
            known_create_time=before.get("create_time")
            if isinstance(before, Mapping) else None,
            process_handle=process,
        )
    except BaseException as exc:
        evidence["after"] = {"psutil_available": False, "pid": pid,
                              "pid_alive": None, "descendant_pids": [],
                              "descendant_alive": [], "descendant_records": [],
                              "snapshot_complete": False,
                              "error": f"{type(exc).__name__}:{exc}"}
    after = evidence["after"]
    probe_deadline = time.monotonic() + 1.0
    while (time.monotonic() < probe_deadline
           and not (
               after.get("snapshot_complete") is True
               and after.get("process_exited") is True
               and after.get("pid_alive") is False
               and all(value is False for value in after.get("descendant_alive", ()))
           )):
        time.sleep(0.05)
        try:
            after = _process_snapshot(
                pid,
                known_descendants=before.get("descendant_records")
                if isinstance(before, Mapping) and before.get("snapshot_complete") is True else None,
                known_create_time=before.get("create_time")
                if isinstance(before, Mapping) else None,
                process_handle=process,
            )
        except BaseException:
            break
    evidence["after"] = after
    evidence["all_stopped"] = (
        isinstance(before, Mapping) and before.get("snapshot_complete") is True
        and after.get("psutil_available") is True
        and after.get("process_exited") is True
        and after.get("pid_alive") is False
        and all(value is False for value in after.get("descendant_alive", ()))
        and after.get("snapshot_complete") is True
    )
    return evidence


def _worker_entry(worker_index: int, loop_count: int, items: list[dict[str, Any]],
                  root: str, deadline: float, worker_count: int,
                  label_strategy: str, use_particle_orbit: bool,
                  parent_source_hashes: Mapping[str, str]) -> None:
    output = Path(root)
    worker_started = time.monotonic()
    actual_hashes = _source_hashes()
    status: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "worker_index": worker_index,
        "worker_pid": os.getpid(),
        "status": "running",
        "process_status": "running",
        "run_status": "running",
        "started_monotonic": worker_started,
        "topology_count": len(items),
        "source_hashes": actual_hashes,
        "parent_source_hashes": dict(parent_source_hashes),
        "topologies_committed": 0,
        "aggregate": {"raw_assignment_count": 0, "momentum_pass_count": 0,
                       "candidate_count": 0, "topologies": 0,
                       "identity_counts": {mode: 0 for mode in MODE_NAMES},
                       "representative_evaluations": 0,
                       "certificate_transports": 0,
                       "particle_orbit_reuse": False},
    }
    _write_json_atomic(output / "worker_status.json", status)
    try:
        keys = [item["topology_key"] for item in items]
        writer = CatalogCheckpointWriter(
            output / "catalog",
            scope=_partition_scope(loop_count, keys, worker_count=worker_count,
                                   worker_index=worker_index, source_hashes=actual_hashes,
                                   label_strategy=label_strategy,
                                   use_particle_orbit=use_particle_orbit),
            expected_topology_keys=keys,
        )
        for item in items:
            row = _run_one_topology(item, writer, deadline=deadline,
                                    label_strategy=label_strategy,
                                    use_particle_orbit=use_particle_orbit)
            status["aggregate"]["topologies"] += 1
            for name in ("raw_assignment_count", "momentum_pass_count", "candidate_count"):
                status["aggregate"][name] += row[name]
            status["aggregate"]["representative_evaluations"] += row["representative_evaluations"]
            status["aggregate"]["certificate_transports"] += row["certificate_transports"]
            status["aggregate"]["particle_orbit_reuse"] = (
                status["aggregate"]["particle_orbit_reuse"] or row["particle_orbit_reuse"]
            )
            for mode in MODE_NAMES:
                status["aggregate"]["identity_counts"][mode] += row["identity_counts"][mode]
            status["topologies_committed"] = status["aggregate"]["topologies"]
            _write_json_atomic(output / "worker_status.json", status)
            if not row["complete"]:
                status.update({"status": "incomplete", "process_status": "completed",
                               "run_status": "incomplete", "stop_reason": row["stop_reason"]})
                _write_json_atomic(output / "worker_status.json", status)
                return
        status.update({"status": "completed", "process_status": "completed",
                       "run_status": "complete", "stop_reason": None,
                       "finished_monotonic": time.monotonic(),
                       "worker_wall_seconds": time.monotonic() - worker_started})
    except BaseException as exc:
        status.update({"status": "failed", "process_status": "failed",
                       "run_status": "incomplete", "stop_reason": "worker_failed",
                       "error_type": type(exc).__name__, "error": str(exc),
                       "finished_monotonic": time.monotonic()})
        _write_json_atomic(output / "worker_status.json", status)
        raise
    _write_json_atomic(output / "worker_status.json", status)


def _start_workers(ctx: Any, specs: list[tuple[Any, ...]], deadline: float) -> list[mp.Process]:
    processes: list[mp.Process] = []
    try:
        for spec in specs:
            if time.monotonic() >= deadline:
                raise TimeoutError("deadline_exceeded_during_worker_start")
            process = ctx.Process(target=_worker_entry, args=spec)
            process.start()
            processes.append(process)
    except BaseException:
        for process in processes:
            if process.is_alive():
                _terminate_tree(process)
        for process in processes:
            process.join(timeout=2.0)
        raise
    return processes


def _monitor_workers(processes: list[mp.Process], deadline: float,
                     cancel_requested: Callable[[], bool] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    timed_out = False
    failed_early = False
    cancelled = False
    termination_events: list[dict[str, Any]] = []

    def stop_live_workers() -> None:
        for index, process in enumerate(processes):
            if not process.is_alive():
                continue
            try:
                event = _terminate_tree(process)
            except BaseException as exc:
                event = {"pid": process.pid, "all_stopped": False,
                         "error": f"{type(exc).__name__}:{exc}"}
            event["worker_index"] = index
            termination_events.append(event)

    try:
        while any(process.is_alive() for process in processes):
            if any((not process.is_alive()) and process.exitcode not in (0, None)
                   for process in processes):
                failed_early = True
                stop_live_workers()
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                stop_live_workers()
                break
            if cancel_requested is not None and cancel_requested():
                cancelled = True
                stop_live_workers()
                break
            time.sleep(min(0.25, remaining))
    finally:
        # Any exception from monitoring or status collection must not orphan
        # workers that were already spawned.
        stop_live_workers()
        for process in processes:
            process.join(timeout=2.0)
    return {
        "wall_seconds": time.monotonic() - started,
        "timed_out": timed_out,
        "failed_early": failed_early,
        "cancelled": cancelled,
        "all_children_stopped": all(not process.is_alive() for process in processes),
        "exitcodes": [process.exitcode for process in processes],
        "termination_events": termination_events,
        "all_termination_events_stopped": all(
            event.get("all_stopped") is True for event in termination_events
        ),
    }


def _read_status(root: Path) -> dict[str, Any] | None:
    path = root / "worker_status.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _verify_shard_index(root: Path, expected_keys: list[Any],
                        source_hashes: Mapping[str, str]) -> dict[str, Any]:
    """Verify durable scope/index coverage only; this is not strict readback."""
    scope_path = root / "catalog" / "scope.json"
    index_path = root / "catalog" / "index.jsonl"
    data_path = root / "catalog" / "data.jsonl.gz"
    try:
        scope_doc = json.loads(scope_path.read_text(encoding="utf-8"))
        raw = index_path.read_bytes()
        data = data_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "error": str(exc), "expected_count": len(expected_keys),
                "committed_count": 0, "committed_keys": [], "entry_count_totals": {mode: 0 for mode in MODE_NAMES}}
    expected = scope_doc.get("expected_topology_keys") if isinstance(scope_doc, Mapping) else None
    expected_tokens = {_token(key) for key in expected_keys}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    entry_count_totals = {mode: 0 for mode in MODE_NAMES}
    last_end = 0
    if (not isinstance(scope_doc, Mapping)
            or scope_doc.get("expected_scope_complete") is not True
            or not isinstance(expected, list)
            or {_token(key) for key in expected} != expected_tokens):
        errors.append("scope_expected_keys_mismatch")
    scope_sha = scope_doc.get("scope_sha256") if isinstance(scope_doc, Mapping) else None
    if isinstance(scope_doc, Mapping):
        scope_base = {
            "schema_version": scope_doc.get("schema_version"),
            "scope": scope_doc.get("scope"),
            "expected_topology_keys": scope_doc.get("expected_topology_keys"),
            "expected_scope_complete": scope_doc.get("expected_scope_complete"),
        }
        if scope_doc.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            errors.append("scope_schema_invalid")
        if scope_sha != _sha256(_token(scope_base).encode()):
            errors.append("scope_sha256_mismatch")
    if raw and not raw.endswith(b"\n"):
        errors.append("index_truncated_tail")
    for line in raw.splitlines():
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            errors.append("index_invalid_json")
            continue
        if isinstance(value, dict) and value.get("record_type") == "commit":
            rows.append(value)
        else:
            errors.append("index_non_commit_row")
    tokens = [_token(row.get("topology_key")) for row in rows]
    if len(tokens) != len(set(tokens)):
        errors.append("duplicate_committed_topology")
    if set(tokens) != expected_tokens:
        errors.append("committed_key_coverage_mismatch")
    for row in rows:
        if (row.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
                or row.get("record_type") != "commit"
                or row.get("data_format") != DATA_FORMAT):
            errors.append("commit_schema_invalid")
        if row.get("scope_sha256") != scope_sha:
            errors.append("commit_scope_sha_mismatch")
        offset = row.get("data_offset")
        length = row.get("data_length")
        if (isinstance(offset, bool) or not isinstance(offset, int)
                or isinstance(length, bool) or not isinstance(length, int)
                or offset < 0 or length <= 0 or offset < last_end
                or offset + length > len(data)):
            errors.append("data_range_invalid")
        else:
            block = data[offset:offset + length]
            if _sha256(block) != row.get("data_sha256"):
                errors.append("data_sha256_mismatch")
            last_end = offset + length
        if row.get("topology_complete") is not True:
            errors.append("topology_not_complete")
        for name in ("label_enumeration_complete", "identity_save_complete"):
            values = row.get(name)
            if not isinstance(values, Mapping):
                errors.append(f"{name}_invalid")
                continue
            if any(values.get(mode) is not True for mode in MODE_NAMES):
                errors.append("mode_completion_not_complete")
        counts = row.get("entry_counts")
        if not isinstance(counts, Mapping) or any(
                isinstance(counts.get(mode), bool) or not isinstance(counts.get(mode), int)
                or counts.get(mode) < 0 for mode in MODE_NAMES):
            errors.append("entry_counts_invalid")
        else:
            for mode in MODE_NAMES:
                entry_count_totals[mode] += counts[mode]
    scope_payload = scope_doc.get("scope") if isinstance(scope_doc, Mapping) else None
    actual_source = (scope_payload.get("source_hashes")
                     if isinstance(scope_payload, Mapping) else None)
    if actual_source != dict(source_hashes):
        errors.append("scope_source_hash_mismatch")
    return {
        "status": "complete" if not errors else "incomplete",
        "expected_count": len(expected_tokens),
        "committed_count": len(tokens),
        "committed_keys": [row.get("topology_key") for row in rows],
        "errors": errors,
        "source_hash_match": actual_source == dict(source_hashes),
        "entry_count_totals": entry_count_totals,
    }


@dataclass(frozen=True, slots=True)
class ParallelRunReport:
    status: str
    n: int
    loop_count: int
    output_directory: str
    manifest_path: str
    started_at: str
    finished_at: str
    deadline_seconds: float
    stop_reason: str | None
    topology_count: int
    stage_timings: Mapping[str, float] = field(default_factory=dict)
    manifest: Mapping[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "status": self.status,
            "complete": self.complete,
            "n": self.n,
            "loop_count": self.loop_count,
            "output_directory": self.output_directory,
            "manifest_path": self.manifest_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "deadline_seconds": self.deadline_seconds,
            "stop_reason": self.stop_reason,
            "topology_count": self.topology_count,
            "stage_timings": dict(self.stage_timings),
            "parallel_manifest": dict(self.manifest),
            "strict_readback": {"status": "not_run", "reason": "independent_follow_up"},
        }


def run_parallel_candidates(*, n: int = 5, loop_count: int = 0,
                            max_seconds: float = _DEFAULT_MAX_SECONDS,
                            output_dir: str | Path | None = None,
                            workers: int = _DEFAULT_WORKERS,
                            label_strategy: str = "default",
                            use_particle_orbit: bool = False,
                            topology_limits: EnumerationLimits | None = None,
                            use_pruning: bool = True,
                            cancel_requested: Callable[[], bool] | None = None) -> ParallelRunReport:
    max_seconds = _validate_budget(max_seconds)
    if output_dir is None:
        raise ValueError("parallel checkpoint mode requires output_dir")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1 or workers > 32:
        raise ValueError("workers must be an integer between 1 and 32")
    if label_strategy not in {"default", "flow"}:
        raise ValueError("label_strategy must be 'default' or 'flow'")
    if label_strategy == "default" and use_particle_orbit:
        raise ValueError("use_particle_orbit requires label_strategy='flow'")
    if type(use_particle_orbit) is not bool:
        raise TypeError("use_particle_orbit must be a boolean")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None")
    root = _fresh_output(output_dir)
    started_wall = time.monotonic()
    deadline = started_wall + max_seconds
    started_at = _timestamp()
    source_hashes = _source_hashes()
    stage_timings: dict[str, float] = {"topology_enumeration": 0.0, "workers": 0.0,
                                       "index_coverage": 0.0, "manifest_persistence": 0.0,
                                       "total": 0.0}
    cancelled = lambda: (cancel_requested is not None and cancel_requested()) or time.monotonic() >= deadline
    enum_started = time.monotonic()
    topology_limits = topology_limits or EnumerationLimits()
    effective_topology_limits = EnumerationLimits(
        topology_limits.max_candidates, topology_limits.max_topologies,
        topology_limits.max_skeletons,
        max(0.0, deadline - time.monotonic()) if topology_limits.max_seconds is None
        else min(max(0.0, deadline - time.monotonic()), topology_limits.max_seconds),
    )
    topology_report = enumerate_topologies(
        n, loop_count, effective_topology_limits, use_pruning=use_pruning,
        cancel_requested=cancelled,
    )
    stage_timings["topology_enumeration"] = time.monotonic() - enum_started
    topologies = sorted(topology_report.topologies,
                        key=lambda topology: _token(canonicalize_topology(topology).key))
    full_keys = [list(canonicalize_topology(topology).key) for topology in topologies]
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "status": "incomplete",
        "stop_reason": topology_report.stop_reason,
        "n": n,
        "loop_count": loop_count,
        "started_at": started_at,
        "deadline_seconds": max_seconds,
        "workers": workers,
        "label_strategy": label_strategy,
        "use_particle_orbit": use_particle_orbit,
        "source_hashes": source_hashes,
        "topology_enumeration": {
            "status": topology_report.status.value,
            "complete": topology_report.complete,
            "unique_count": topology_report.unique_count,
            "stored_count": len(topologies),
            "stop_reason": topology_report.stop_reason,
        },
        "full_topology_keys": full_keys,
        "full_topology_key_sha256": _sha256(_token(full_keys).encode()),
        "partition_rule": "sorted canonical topology key index modulo workers",
        "shards": [],
        "union_coverage": None,
        "actual_counts": None,
        "strict_readback": {"status": "not_run", "reason": "independent_follow_up"},
    }
    if not topology_report.complete or len(topologies) != topology_report.unique_count:
        manifest["stop_reason"] = manifest["stop_reason"] or "incomplete_topology_scope"
        manifest["status"] = "incomplete"
        _write_json_atomic(root / "parallel_manifest.json", manifest)
        report = ParallelRunReport("incomplete", n, loop_count, str(root),
                                   str(root / "parallel_manifest.json"), started_at,
                                   _timestamp(), max_seconds, manifest["stop_reason"],
                                   len(topologies), stage_timings, manifest)
        _write_json_atomic(root / "run_manifest.json", report.to_dict())
        return report

    partitions: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    for index, topology in enumerate(topologies):
        partitions[index % workers].append(_topology_payload(topology, full_keys[index]))
    run_started_workers = time.monotonic()
    ctx = mp.get_context("spawn")
    specs: list[tuple[Any, ...]] = []
    for index, items in enumerate(partitions):
        shard = root / f"shard-{index}"
        shard.mkdir()
        specs.append((index, loop_count, items, str(shard), deadline, workers,
                      label_strategy, use_particle_orbit, source_hashes))
    processes: list[mp.Process] = []
    start_error: str | None = None
    try:
        processes = _start_workers(ctx, specs, deadline)
        monitor = _monitor_workers(processes, deadline, cancel_requested=cancel_requested)
    except BaseException as exc:
        start_error = f"{type(exc).__name__}:{exc}"
        monitor = {"wall_seconds": time.monotonic() - run_started_workers,
                   "timed_out": False, "failed_early": True,
                   "cancelled": False,
                   "all_children_stopped": all(not process.is_alive() for process in processes),
                   "exitcodes": [process.exitcode for process in processes],
                   "termination_events": [],
                   "all_termination_events_stopped": False}
    stage_timings["workers"] = time.monotonic() - run_started_workers
    statuses = [_read_status(root / f"shard-{index}") for index in range(workers)]
    shard_checks: list[dict[str, Any]] = []
    index_started = time.monotonic()
    for index, items in enumerate(partitions):
        if time.monotonic() >= deadline:
            break
        shard_checks.append(_verify_shard_index(root / f"shard-{index}",
                                                [item["topology_key"] for item in items],
                                                source_hashes))
    stage_timings["index_coverage"] = time.monotonic() - index_started
    for index, check in enumerate(shard_checks):
        status = statuses[index] if index < len(statuses) else None
        aggregate_value = status.get("aggregate") if isinstance(status, Mapping) else None
        declared = (aggregate_value.get("identity_counts")
                    if isinstance(aggregate_value, Mapping) else None)
        actual = check.get("entry_count_totals")
        check["worker_identity_counts_match"] = isinstance(declared, Mapping) and declared == actual
        if not check["worker_identity_counts_match"]:
            check.setdefault("errors", []).append("entry_counts_worker_aggregate_mismatch")
    all_source_hashes_match = all(
        isinstance(status, Mapping) and status.get("source_hashes") == source_hashes
        for status in statuses
    )
    all_status_complete = all(
        isinstance(status, Mapping) and status.get("status") == "completed"
        and status.get("run_status") == "complete" for status in statuses
    )
    index_complete = len(shard_checks) == workers and all(
        check.get("status") == "complete" and not check.get("errors")
        and check.get("worker_identity_counts_match") is True
        for check in shard_checks
    )
    committed_tokens: list[str] = []
    committed_token_set: set[str] = set()
    duplicate_tokens: list[str] = []
    for check in shard_checks:
        for key in check.get("committed_keys", ()):
            token = _token(key)
            if token in committed_token_set:
                duplicate_tokens.append(token)
            committed_tokens.append(token)
            committed_token_set.add(token)
    expected_tokens = {_token(key) for key in full_keys}
    committed_set = committed_token_set
    union_exact = committed_set == expected_tokens and not duplicate_tokens
    aggregate = {"topologies": 0, "raw_assignment_count": 0,
                 "momentum_pass_count": 0, "candidate_count": 0,
                 "identity_counts": {mode: 0 for mode in MODE_NAMES},
                 "representative_evaluations": 0,
                 "certificate_transports": 0,
                 "particle_orbit_reuse": False}
    aggregate_error: str | None = None
    for status in statuses:
        values = status.get("aggregate") if isinstance(status, Mapping) else None
        if isinstance(values, Mapping):
            try:
                for name in ("topologies", "raw_assignment_count", "momentum_pass_count",
                             "candidate_count", "representative_evaluations",
                             "certificate_transports"):
                    value = values.get(name, 0)
                    if not _is_nonnegative_int(value):
                        raise ValueError(f"{name}_invalid")
                    aggregate[name] += value
                counts = values.get("identity_counts")
                if not isinstance(counts, Mapping):
                    raise ValueError("identity_counts_missing")
                for mode in MODE_NAMES:
                    value = counts.get(mode)
                    if not _is_nonnegative_int(value):
                        raise ValueError(f"identity_count_{mode}_invalid")
                    aggregate["identity_counts"][mode] += value
                orbit_reuse = values.get("particle_orbit_reuse")
                if type(orbit_reuse) is not bool:
                    raise ValueError("particle_orbit_reuse_invalid")
                aggregate["particle_orbit_reuse"] = (
                    aggregate["particle_orbit_reuse"] or orbit_reuse
                )
            except (TypeError, ValueError):
                aggregate_error = "worker_aggregate_invalid"
        else:
            aggregate_error = (
                "worker_aggregate_missing" if values is None
                else "worker_aggregate_invalid"
            )
    manifest["worker_monitor"] = monitor
    manifest["worker_start_error"] = start_error
    manifest["worker_statuses"] = statuses
    manifest["shards"] = [
        {"index": index, "expected_count": len(items),
         "expected_keys": [item["topology_key"] for item in items],
         "checkpoint_root": str(root / f"shard-{index}" / "catalog"),
         "index_coverage": shard_checks[index] if index < len(shard_checks) else {"status": "not_run"},
         "status": statuses[index].get("run_status") if isinstance(statuses[index], Mapping) else None}
        for index, items in enumerate(partitions)
    ]
    manifest["union_coverage"] = {
        "expected_count": len(expected_tokens),
        "committed_count": len(committed_set),
        "missing_count": len(expected_tokens - committed_set),
        "unexpected_count": len(committed_set - expected_tokens),
        "duplicate_count": len(duplicate_tokens),
        "shards_pairwise_disjoint": not duplicate_tokens,
        "coverage_exact": union_exact,
    }
    manifest["actual_counts"] = aggregate
    manifest["actual_counts_error"] = aggregate_error
    manifest["worker_source_hashes_match"] = all_source_hashes_match
    manifest["status_checks"] = {
        "all_workers_completed": all_status_complete,
        "index_coverage_complete": index_complete,
        "children_stopped": monitor["all_children_stopped"],
        "termination_events_stopped": monitor.get("all_termination_events_stopped", False),
        "cancelled": monitor.get("cancelled", False),
        "strict_readback_not_run": True,
    }
    stop_reason: str | None = None
    if start_error:
        stop_reason = "worker_start_failed"
    elif monitor.get("cancelled"):
        stop_reason = "cancelled"
    elif monitor["timed_out"]:
        stop_reason = "deadline_exceeded"
    elif monitor["failed_early"] or any(code not in (0, None) for code in monitor["exitcodes"]):
        stop_reason = "worker_failed"
    elif (not monitor["all_children_stopped"]
          or not monitor.get("all_termination_events_stopped", False)
          or aggregate_error is not None
          or not all_source_hashes_match or not all_status_complete
          or not index_complete or not union_exact):
        stop_reason = "incomplete_shard_coverage"
    elif time.monotonic() >= deadline:
        stop_reason = "deadline_exceeded"
    manifest["stop_reason"] = stop_reason
    manifest["status"] = "complete" if stop_reason is None else "incomplete"
    manifest["finished_at"] = _timestamp()
    manifest["stage_timings"] = dict(stage_timings)
    persist_started = time.monotonic()
    _write_json_atomic(root / "parallel_manifest.json", manifest)
    stage_timings["manifest_persistence"] = time.monotonic() - persist_started
    stage_timings["total"] = time.monotonic() - started_wall
    manifest["stage_timings"] = dict(stage_timings)
    manifest["actual_wall_seconds"] = stage_timings["total"]
    manifest["status_checks"]["final_save_within_deadline"] = time.monotonic() < deadline
    if manifest["status"] == "complete" and not manifest["status_checks"]["final_save_within_deadline"]:
        manifest["status"] = "incomplete"
        manifest["stop_reason"] = "deadline_exceeded"
        stop_reason = "deadline_exceeded"
    _write_json_atomic(root / "parallel_manifest.json", manifest)
    report = ParallelRunReport(manifest["status"], n, loop_count, str(root),
                               str(root / "parallel_manifest.json"), started_at,
                               _timestamp(), max_seconds, stop_reason,
                               len(topologies), stage_timings, manifest)
    _write_json_atomic(root / "run_manifest.json", report.to_dict())
    if manifest["status"] == "complete" and time.monotonic() >= deadline:
        # Final persistence is part of the global budget.  Correct both
        # sidecars if serialization itself crossed the deadline.
        manifest["status"] = "incomplete"
        manifest["stop_reason"] = "deadline_exceeded"
        stop_reason = "deadline_exceeded"
        manifest["status_checks"]["final_save_within_deadline"] = False
        manifest["finished_at"] = _timestamp()
        _write_json_atomic(root / "parallel_manifest.json", manifest)
        report = ParallelRunReport("incomplete", n, loop_count, str(root),
                                   str(root / "parallel_manifest.json"), started_at,
                                   _timestamp(), max_seconds, "deadline_exceeded",
                                   len(topologies), stage_timings, manifest)
        _write_json_atomic(root / "run_manifest.json", report.to_dict())
    return report


__all__ = ["ParallelRunReport", "run_parallel_candidates"]
