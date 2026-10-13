"""Run the opt-in four-process five-body checkpoint coordinator.

The ordinary ``run_candidates.py`` remains the default serial CLI.  This
entry point adds an outer Windows process supervisor so its requested budget
includes Python startup, worker startup, strict-free index coverage checks,
and final manifest persistence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping


HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_space_graph5.parallel_reporting import run_parallel_candidates  # noqa: E402
from phase_space_graph5.data_paths import default_output  # noqa: E402


_DEFAULT_MAX_SECONDS = 540.0
_HARD_BUDGET_SECONDS = 1200.0
_RESERVE_SECONDS = 1.0
_TIMEOUT_EXIT = 124


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _prepare_output(path: Path) -> Path:
    output = path.resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        output = output / f"run-{time.time_ns()}-{os.getpid()}"
        output.mkdir()
    return output


def _read_actual_status(output: Path) -> tuple[str | None, str | None]:
    for name in ("run_manifest.json", "parallel_manifest.json"):
        path = output / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value.get("status"), value.get("stop_reason")
    return None, None


def _write_status(output: Path, *, status: str, process_status: str,
                  run_status: str | None, stop_reason: str | None,
                  requested_seconds: float, worker_seconds: float,
                  started: float, worker_pid: int | None,
                  worker_exit_code: int | None, error: str | None = None,
                  termination: Mapping[str, Any] | None = None) -> None:
    payload = {
        "schema_version": "F5-parallel-cli-v1",
        "status": status,
        "process_status": process_status,
        "run_status": run_status,
        "stop_reason": stop_reason,
        "requested_max_seconds": requested_seconds,
        "worker_budget_seconds": worker_seconds,
        "hard_budget_seconds": _HARD_BUDGET_SECONDS,
        "started_at_monotonic": started,
        "elapsed_seconds": max(0.0, time.monotonic() - started),
        "worker_pid": worker_pid,
        "worker_exit_code": worker_exit_code,
        "error": error,
        "termination": dict(termination) if termination is not None else None,
        "tree_stopped": (termination.get("all_stopped")
                         if isinstance(termination, Mapping) else None),
    }
    temporary = output / f".worker_status.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        for attempt in range(20):
            try:
                temporary.replace(output / "worker_status.json")
                return
            except PermissionError as exc:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _process_snapshot_raw(pid: int | None, *, known_descendants: list[Mapping[str, Any]] | None = None,
                          known_create_time: float | None = None) -> dict[str, Any]:
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
        current = psutil.Process(pid)
    except psutil.NoSuchProcess:
        try:
            records = [dict(item) for item in (known_descendants or ())]
            pids = [int(item["pid"]) for item in records]
            alive = [probe(item_pid, item.get("create_time"))
                     for item_pid, item in zip(pids, records)]
        except Exception as exc:
            return {"psutil_available": True, "pid": pid,
                    "pid_alive": False, "descendant_pids": [],
                    "descendant_alive": [], "descendant_records": records,
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
            descendants = current.children(recursive=True)
            descendant_records = []
            for child in descendants:
                try:
                    descendant_records.append(
                        {"pid": child.pid, "create_time": child.create_time()}
                    )
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
            descendant_records = [dict(item) for item in known_descendants]
        except Exception as exc:
            return {"psutil_available": True, "pid": pid,
                    "pid_alive": None, "descendant_pids": [],
                    "descendant_alive": [], "descendant_records": [],
                    "snapshot_complete": False,
                    "error": f"{type(exc).__name__}:{exc}"}
    try:
        descendant_pids = [int(item["pid"]) for item in descendant_records]
    except Exception as exc:
        return {"psutil_available": True, "pid": pid,
                "pid_alive": None, "descendant_pids": [],
                "descendant_alive": [], "descendant_records": descendant_records,
                "snapshot_complete": False,
                "error": f"{type(exc).__name__}:{exc}"}
    descendant_alive = [probe(pid_value, item.get("create_time"))
                        for pid_value, item in zip(descendant_pids, descendant_records)]
    try:
        create_time = (known_create_time if known_create_time is not None
                       else current.create_time())
    except psutil.NoSuchProcess:
        return {"psutil_available": True, "pid": pid,
                "pid_alive": False, "descendant_pids": descendant_pids,
                "descendant_alive": descendant_alive, "descendant_records": descendant_records,
                "snapshot_complete": False,
                "error": "root_exited_during_snapshot"}
    except Exception as exc:
        return {"psutil_available": True, "pid": pid,
                "pid_alive": None, "descendant_pids": descendant_pids,
                "descendant_alive": descendant_alive, "descendant_records": descendant_records,
                "snapshot_complete": False,
                "error": f"{type(exc).__name__}:{exc}"}
    pid_alive = probe(pid, create_time)
    return {"psutil_available": True, "pid": pid,
            "pid_alive": pid_alive,
            "create_time": create_time,
            "descendant_pids": descendant_pids,
            "descendant_alive": descendant_alive,
            "descendant_records": descendant_records,
            "snapshot_complete": pid_alive is not None
            and all(value is not None for value in descendant_alive)}


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


def _terminate_tree(process: subprocess.Popen[bytes]) -> dict[str, Any]:
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
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                capture_output=True, text=True, check=False, timeout=5.0,
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
            worker_group = os.getpgid(process.pid)
            if worker_group != os.getpgrp():
                os.killpg(worker_group, signal.SIGKILL)
            else:
                evidence["taskkill_error"] = "shared_process_group"
        except (OSError, ProcessLookupError):
            evidence["taskkill_error"] = "killpg_failed"
    try:
        process.kill()
        evidence["fallback_process_kill"] = True
    except ProcessLookupError:
        pass
    except OSError as exc:
        evidence["taskkill_error"] = evidence["taskkill_error"] or f"{type(exc).__name__}:{exc}"
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        evidence["taskkill_error"] = evidence["taskkill_error"] or "process_wait_timeout"
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


def _worker_argv(args: argparse.Namespace, output: Path, worker_seconds: float) -> list[str]:
    command = [sys.executable, str(HERE), "--_worker",
               "--max-seconds", repr(worker_seconds), "--output", str(output),
               "--loop-count", str(args.loop_count), "--workers", str(args.workers),
               "--label-strategy", args.label_strategy]
    if args.orbit:
        command.append("--orbit")
    return command


def _run_worker(args: argparse.Namespace) -> int:
    started = time.monotonic()
    try:
        report = run_parallel_candidates(
            loop_count=args.loop_count, max_seconds=args.max_seconds,
            output_dir=args.output, workers=args.workers,
            label_strategy=args.label_strategy,
            use_particle_orbit=args.orbit,
        )
        status = report.status
        _write_status(args.output, status="completed", process_status="completed",
                      run_status=status, stop_reason=report.stop_reason,
                      requested_seconds=args.max_seconds,
                      worker_seconds=args.max_seconds, started=started,
                      worker_pid=os.getpid(), worker_exit_code=0)
        return 0 if status in {"complete", "incomplete"} else 1
    except Exception as exc:
        _write_status(args.output, status="failed", process_status="failed",
                      run_status="incomplete", stop_reason="worker_failed",
                      requested_seconds=args.max_seconds,
                      worker_seconds=args.max_seconds, started=started,
                      worker_pid=os.getpid(), worker_exit_code=1,
                      error=f"{type(exc).__name__}: {exc}")
        return 1


def _run_outer(args: argparse.Namespace) -> int:
    requested = float(args.max_seconds)
    output = _prepare_output(args.output)
    started = time.monotonic()
    reserve = min(_RESERVE_SECONDS, requested * 0.25)
    worker_seconds = max(0.0, requested - reserve)
    command = _worker_argv(args, output, worker_seconds)
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process: subprocess.Popen[bytes] | None = None
    termination: dict[str, Any] | None = None
    try:
        kwargs: dict[str, object] = {
            "cwd": str(HERE.parents[1]),
            "stdin": subprocess.DEVNULL,
            "stdout": None,
            "stderr": None,
            "creationflags": creationflags,
        }
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
        wait_seconds = max(1.0, requested) if requested == 0 else max(
            0.0, requested - (time.monotonic() - started)
        )
        try:
            process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            termination = _terminate_tree(process)
            _write_status(output, status="hard_timeout", process_status="hard_timeout",
                          run_status="incomplete", stop_reason="deadline_exceeded",
                          requested_seconds=requested, worker_seconds=worker_seconds,
                          started=started, worker_pid=process.pid,
                          worker_exit_code=process.returncode,
                          termination=termination)
            return _TIMEOUT_EXIT
        exit_code = process.returncode
        run_status, run_stop_reason = _read_actual_status(output)
        if exit_code == 0:
            _write_status(output, status="completed", process_status="completed",
                          run_status=run_status, stop_reason=run_stop_reason,
                          requested_seconds=requested, worker_seconds=worker_seconds,
                          started=started, worker_pid=process.pid,
                          worker_exit_code=exit_code)
            return 0
        _write_status(output, status="failed", process_status="failed",
                      run_status="incomplete", stop_reason=run_stop_reason or "worker_failed",
                      requested_seconds=requested, worker_seconds=worker_seconds,
                      started=started, worker_pid=process.pid,
                      worker_exit_code=exit_code)
        return 1
    except (OSError, ValueError) as exc:
        if process is not None and process.poll() is None:
            termination = _terminate_tree(process)
        _write_status(output, status="failed", process_status="failed",
                      run_status="incomplete", stop_reason="worker_start_failed",
                      requested_seconds=requested, worker_seconds=worker_seconds,
                      started=started, worker_pid=process.pid if process else None,
                      worker_exit_code=process.returncode if process else None,
                      error=str(exc), termination=termination if process else None)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-count", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=_DEFAULT_MAX_SECONDS)
    parser.add_argument(
        "--output", type=Path, default=default_output("runs", "parallel"),
        help="output directory (default: PSF_DATA_ROOT/Demo案例/5体/runs/parallel)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label-strategy", choices=("default", "flow"), default="default")
    parser.add_argument("--orbit", action="store_true",
                        help="enable existing flow particle-orbit reuse (requires --label-strategy flow)")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or not 0 <= args.max_seconds <= _HARD_BUDGET_SECONDS:
        parser.error(f"--max-seconds must be between 0 and {_HARD_BUDGET_SECONDS:g}")
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")
    if args.orbit and args.label_strategy != "flow":
        parser.error("--orbit requires --label-strategy flow")
    if args._worker:
        return _run_worker(args)
    return _run_outer(args)


if __name__ == "__main__":
    raise SystemExit(main())
