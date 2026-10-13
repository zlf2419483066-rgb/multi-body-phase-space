"""Bounded generic witness/search batch runner.

This runner consumes only explicitly supplied witness files (L3/L4/L5) or a
single explicitly supplied L2 drawing manifest.  It does not enumerate
topologies.  Witness rows are reviewed one at a time; a verified generic PASS
stops that topology, while a verified generic REJECT invokes the bounded
assignment search and its independent replay verifier.  A topology REJECT is
therefore possible only after a complete replay.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phase_space_graph5.canonical import canonicalize_topology
from phase_space_graph5.domain import EdgeType, InternalEdge, Topology
from phase_space_graph5.independence.generic_family import evaluate_generic_label
from phase_space_graph5.independence.generic_search import (
    GENERIC_SEARCH_BACKENDS,
    search_generic_topology,
    verify_generic_search,
)
from phase_space_graph5.independence.generic_verification import verify_generic_label
from phase_space_graph5.independence.sources import VerifiedSource, _key_token
from phase_space_graph5.independence.witness_sources import (
    WitnessSourceError,
    iter_witness_rows,
)


RUNNER_SCHEMA = "F5-I-generic-witness-batch-v1"
_STATUSES = frozenset({"pass", "reject", "unknown"})
_IMPLEMENTATION_FILES = (
    "generic_family.py", "generic_verification.py", "generic_search.py",
    "witness_sources.py", "sources.py", "invariant_sets.py", "hajer_constraints.py",
    "exact_algebra.py", "contracts.py", "coordinates.py",
    "domain.py", "physical_labels.py", "label_identity.py", "canonical.py",
    "labeling.py", "catalog_checkpoint.py", "label_catalog.py",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _implementation_hashes() -> dict[str, str | None]:
    package = _SRC / "phase_space_graph5"
    result: dict[str, str | None] = {}
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        result[str(path.relative_to(_SRC))] = _sha256_file(path)[0]
    result[str(Path(__file__).resolve().relative_to(_ROOT))] = _sha256_file(Path(__file__))[0]
    return result


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    temporary.replace(path)


class _GzipJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = gzip.open(path, "wt", encoding="utf-8", newline="\n")
        self.rows = 0

    def write(self, value: Any) -> int:
        self.handle.write(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True))
        self.handle.write("\n")
        self.rows += 1
        return self.rows - 1

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "_GzipJsonlWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _safe_source_rows(iterator: Iterable[Any], binding: dict[str, Any], errors: list[dict[str, Any]]) -> Iterable[Any]:
    try:
        yield from iterator
    except WitnessSourceError as exc:
        binding["iterator_error"] = True
        errors.append({"path": binding["path"], "error": f"source:{exc}"})


def _prepare_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    else:
        path.mkdir(parents=True)


def _theorem_binding(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"theorem file not found: {resolved}")
    return {"path": str(resolved), "sha256": _sha256_file(resolved)[0]}


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _invalid_report(error: str) -> dict[str, Any]:
    return {"valid": False, "status": "unknown", "checks": [], "errors": [error]}


def _witness_verifier(row: Any) -> VerifiedSource:
    """Use the witness adapter's public single-row review API."""

    from phase_space_graph5.independence.witness_sources import verify_witness_row

    return verify_witness_row(row)


def _source_row(path: str, index: int, file_sha256: str, record: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "path": path,
        "index": index,
        "file_sha256": file_sha256,
        # One source record is retained per evidence row.  Raw witness JSON is
        # deliberately omitted so topology/index rows stay lightweight.
        "record": dict(record) if record is not None else None,
    }


def _copy_search_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Retain the independently found witness record for replay.

    A generic-label REJECT source record and the later search PASS witness are
    different records.  Both are required to audit the transition, so the
    search result is intentionally kept complete.
    """

    return copy.deepcopy(dict(result))


def _search_witness_reference(
    result: Mapping[str, Any], *, source_path: str, source_index: int
) -> dict[str, Any]:
    witness = result.get("witness")
    evidence = witness.get("evidence") if isinstance(witness, Mapping) else None
    identity_key = evidence.get("identity_key") if isinstance(evidence, Mapping) else None
    return {
        "kind": "generic_search",
        "identity_key": identity_key,
        "source_path": source_path,
        "source_index": source_index,
        "assignment": witness.get("assignment") if isinstance(witness, Mapping) else None,
    }


def _topology_state(token: str, topology: Any) -> dict[str, Any]:
    return {
        "topology_key": token,
        "status": "unknown",
        "source_row_count": 0,
        "evaluated_count": 0,
        "pass_count": 0,
        "label_reject_count": 0,
        "search_reject_count": 0,
        "unknown_count": 0,
        "certificate_rows": [],
        "witness_references": [],
        "terminal": False,
        "input_error": False,
        "topology": topology.to_dict() if isinstance(topology, Topology) else None,
    }


def _evaluate_witness(
    source: VerifiedSource,
    *,
    source_path: str,
    source_index: int,
    file_sha256: str,
    deadline: float,
    per_topology_seconds: float,
    search_backend: str,
) -> tuple[dict[str, Any], str]:
    started = time.monotonic()
    identity_key = _key_token(source.key)
    topology_key = _key_token(source.topology_key)
    record = source.source_record.to_dict()
    output: dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA,
        "source_row": _source_row(source_path, source_index, file_sha256, record),
        "topology_key": topology_key,
        "identity_key": identity_key,
        "generic_evidence": None,
        "generic_verifier": None,
        "search_result": None,
        "search_verifier": None,
        "generic_status": "unknown",
        "status": "unknown",
        "witness_reference": None,
        "errors": [],
    }
    if _remaining(deadline) <= 0:
        output["errors"].append("max_seconds_exhausted_before_evaluation")
        return output, topology_key
    try:
        evidence = evaluate_generic_label(source)
        verifier = verify_generic_label(source, evidence)
        output["generic_evidence"] = _jsonable(evidence)
        output["generic_verifier"] = _jsonable(verifier)
        if (
            not isinstance(evidence, Mapping)
            or not isinstance(verifier, Mapping)
            or verifier.get("valid") is not True
            or evidence.get("status") not in _STATUSES
            or verifier.get("status") != evidence.get("status")
        ):
            output["errors"].append("generic_evidence_or_verifier_invalid")
            return output, topology_key
        generic_status = str(evidence["status"])
        output["generic_status"] = generic_status
        if generic_status == "pass":
            output["status"] = "pass"
            output["witness_reference"] = {
                "kind": "generic_label",
                "identity_key": identity_key,
                "source_path": source_path,
                "source_index": source_index,
            }
            return output, topology_key
        if generic_status != "reject":
            output["errors"].append("generic_verifier_returned_unknown_status")
            return output, topology_key
    except Exception as exc:
        output["errors"].append(f"generic_evaluation_error:{type(exc).__name__}:{exc}")
        return output, topology_key

    search_budget = min(per_topology_seconds - (time.monotonic() - started), _remaining(deadline))
    if search_budget <= 0:
        output["errors"].append("per_topology_budget_exhausted_before_search")
        return output, topology_key
    try:
        search_result = search_generic_topology(
            source.source_record.topology,
            max_seconds=search_budget,
            backend=search_backend,
        )
        output["search_result"] = _copy_search_result(search_result)
        verify_budget = min(per_topology_seconds - (time.monotonic() - started), _remaining(deadline))
        if verify_budget <= 0:
            output["errors"].append("per_topology_budget_exhausted_before_search_verification")
            return output, topology_key
        search_verifier = verify_generic_search(
            source.source_record.topology,
            search_result,
            max_seconds=verify_budget,
            backend=search_backend,
        )
        output["search_verifier"] = _jsonable(search_verifier)
        if search_verifier.get("valid") is not True:
            output["errors"].append("generic_search_verifier_invalid")
            return output, topology_key
        if search_verifier.get("status") == "pass":
            output["status"] = "pass"
            output["witness_reference"] = _search_witness_reference(
                search_result, source_path=source_path, source_index=source_index
            )
        elif (
            search_verifier.get("status") == "reject"
            and search_result.get("status") == "reject"
            and search_result.get("complete") is True
        ):
            output["status"] = "reject"
        else:
            output["errors"].append("search_result_not_complete_for_conclusion")
    except Exception as exc:
        output["errors"].append(f"generic_search_error:{type(exc).__name__}:{exc}")
    return output, topology_key


def _restore_l2_topology(key: Any) -> Topology:
    """Restore a canonical topology key without manufacturing a source record."""

    value = key
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, (tuple, list)) or len(value) != 6 or value[0] != "topology":
        raise ValueError("L2 row topology_key is not a canonical topology key")
    n, vertex_count, initial, slots, edges = value[1:]
    if not isinstance(n, int) or isinstance(n, bool) or not isinstance(vertex_count, int):
        raise ValueError("L2 topology key has invalid integer fields")
    if not isinstance(slots, (tuple, list)) or len(slots) != vertex_count:
        raise ValueError("L2 topology key slot order does not match vertex count")
    restored_edges = []
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 4:
            raise ValueError("L2 topology key has an invalid edge")
        source, target, kind, multiplicity = edge
        restored_edges.append(InternalEdge(source, target, EdgeType(kind), multiplicity))
    topology = Topology(
        n,
        tuple(range(vertex_count)),
        tuple(restored_edges),
        initial,
        tuple((index, count) for index, count in enumerate(slots)),
    )
    canonical = canonicalize_topology(topology)
    if _key_token(canonical.key) != _key_token(value):
        raise ValueError("L2 topology key failed canonical topology replay")
    return topology


def _iter_l2_rows(path: Path) -> Iterable[tuple[int, Mapping[str, Any], Topology]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("rows"), list):
        raise ValueError("L2 drawing manifest must contain a rows array")
    for index, row in enumerate(payload["rows"]):
        if not isinstance(row, Mapping):
            continue
        if row.get("kind") != "marked":
            continue
        validation = row.get("validation")
        if row.get("status") != "complete" or isinstance(validation, Mapping) and validation.get("valid") is False:
            raise ValueError("L2 marked row is not a valid complete structure")
        yield index, row, _restore_l2_topology(row.get("topology_key"))


def _evaluate_l2(
    row: Mapping[str, Any], topology: Topology, *, source_path: str, source_index: int,
    file_sha256: str, deadline: float, per_topology_seconds: float,
    search_backend: str,
) -> tuple[dict[str, Any], str]:
    started = time.monotonic()
    topology_key = _key_token(topology_key_value := canonicalize_topology(topology).key)
    output: dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA,
        "source_row": _source_row(source_path, source_index, file_sha256, None),
        "topology_key": topology_key,
        "identity_key": None,
        "generic_evidence": None,
        "generic_verifier": None,
        "search_result": None,
        "search_verifier": None,
        "generic_status": "not_applicable_l2",
        "status": "unknown",
        "witness_reference": None,
        "errors": [],
    }
    budget = min(per_topology_seconds, _remaining(deadline))
    if budget <= 0:
        output["errors"].append("budget_exhausted_before_l2_search")
        return output, topology_key
    try:
        result = search_generic_topology(topology, max_seconds=budget, backend=search_backend)
        output["search_result"] = _jsonable(result)
        verify_budget = min(per_topology_seconds - (time.monotonic() - started), _remaining(deadline))
        if verify_budget <= 0:
            output["errors"].append("budget_exhausted_before_l2_search_verification")
            return output, topology_key
        report = verify_generic_search(
            topology, result, max_seconds=verify_budget, backend=search_backend
        )
        output["search_verifier"] = _jsonable(report)
        if report.get("valid") is True and report.get("status") in _STATUSES:
            if report.get("status") == "pass":
                output["status"] = "pass"
                output["witness_reference"] = {"kind": "generic_search", "source_index": source_index}
            elif result.get("complete") is True and report.get("status") == "reject":
                output["status"] = "reject"
            else:
                output["errors"].append("l2_search_not_complete_for_conclusion")
        else:
            output["errors"].append("l2_search_verifier_invalid")
    except Exception as exc:
        output["errors"].append(f"l2_search_error:{type(exc).__name__}:{exc}")
    return output, topology_key


def _update_state(state: dict[str, Any], row: Mapping[str, Any], line_number: int) -> None:
    state["evaluated_count"] += 1
    state["certificate_rows"].append(line_number)
    status = row.get("status")
    if status == "pass":
        state["pass_count"] += 1
        state["status"] = "pass"
        state["terminal"] = True
    elif status == "reject":
        state["search_reject_count"] += 1
        if state["status"] != "pass":
            state["status"] = "reject"
            state["terminal"] = True
    else:
        state["unknown_count"] += 1
        if state["status"] not in {"pass", "reject"}:
            state["status"] = "unknown"
    witness = row.get("witness_reference")
    if witness is not None:
        state["witness_references"].append(witness)


def run_batch(
    sources: Iterable[Path], output: Path, *, mode: str, offset: int, limit: int | None,
    per_topology_seconds: float, max_seconds: float, theorem: Path, expected_count: int | None,
    search_backend: str = "reference",
) -> int:
    started = time.monotonic()
    deadline = started + max_seconds
    _prepare_output(output)
    if search_backend not in GENERIC_SEARCH_BACKENDS:
        raise ValueError(f"search_backend must be one of {GENERIC_SEARCH_BACKENDS!r}")
    theorem_before: dict[str, str] | None = None
    implementation_before: dict[str, str | None] = {}
    labels_writer = _GzipJsonlWriter(output / "labels.jsonl.gz")
    topology_writer = _GzipJsonlWriter(output / "topology_index.jsonl.gz")
    source_bindings: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    seen_topologies: set[str] = set()
    source_errors: list[dict[str, Any]] = []
    unclassified_unknown_rows = 0
    rows_seen = evaluations = 0
    timed_out = False
    failed = False
    error: str | None = None
    try:
        theorem_before = _theorem_binding(theorem)
        implementation_before = _implementation_hashes()
        source_list = tuple(Path(item).expanduser().resolve() for item in sources)
        if mode == "l2" and len(source_list) != 1:
            raise ValueError("L2 mode requires exactly one drawing manifest source")
        for path in source_list:
            if _remaining(deadline) <= 0:
                timed_out = True
                break
            before_sha, before_size = _sha256_file(path)
            binding = {
                "path": str(path), "sha256_before": before_sha, "size_before": before_size,
                "rows_seen": 0, "evaluation_rows": 0, "read_to_eof": False,
                "iterator_error": False,
            }
            source_bindings.append(binding)
            if mode == "l2":
                iterator: Iterable[Any] = _iter_l2_rows(path)
            else:
                # A full read verifies this same-stream digest at EOF. Partial
                # reads close before EOF and remain bounded by the runner's
                # independent before/after source hashes.
                iterator = iter_witness_rows(path, expected_sha256=before_sha)
            stopped_early = False
            for item in _safe_source_rows(iterator, binding, source_errors):
                if mode == "l2":
                    source_index, raw_row, topology = item
                    row_path = str(path)
                    row_key = _key_token(canonicalize_topology(topology).key)
                    source_obj = None
                else:
                    raw_row = item.raw
                    source_index = item.source_index
                    row_path = item.source_path
                    row_key = None
                    source_obj = None
                if (
                    (limit is not None and evaluations >= limit and rows_seen >= offset)
                    or _remaining(deadline) <= 0
                ):
                    stopped_early = True
                    if _remaining(deadline) <= 0:
                        timed_out = True
                    break
                rows_seen += 1
                binding["rows_seen"] += 1
                _write_json_atomic(output / "progress.json", {
                    "schema_version": RUNNER_SCHEMA,
                    "status": "running",
                    "mode": mode,
                    "rows_seen": rows_seen,
                    "evaluations": evaluations,
                    "source_path": row_path,
                    "source_index": source_index,
                    "elapsed_seconds": time.monotonic() - started,
                })
                selected = rows_seen - 1 >= offset and (limit is None or evaluations < limit)
                if not selected or _remaining(deadline) <= 0:
                    if _remaining(deadline) <= 0:
                        timed_out = True
                    continue
                if mode == "witness":
                    try:
                        source_obj = _witness_verifier(item)
                        row_key = _key_token(source_obj.topology_key)
                    except Exception as exc:
                        source_errors.append({"path": row_path, "index": source_index, "error": f"witness:{type(exc).__name__}:{exc}"})
                        unclassified_unknown_rows += 1
                        labels_writer.write({
                            "schema_version": RUNNER_SCHEMA,
                            "source_row": _source_row(row_path, source_index, before_sha, None),
                            "topology_key": None,
                            "identity_key": None,
                            "generic_evidence": None,
                            "generic_verifier": None,
                            "search_result": None,
                            "search_verifier": None,
                            "generic_status": "unknown",
                            "status": "unknown",
                            "witness_reference": None,
                            "errors": [f"witness_row_invalid:{type(exc).__name__}:{exc}"],
                        })
                        evaluations += 1
                        binding["evaluation_rows"] += 1
                        continue
                state = states.setdefault(
                    row_key,
                    _topology_state(row_key, topology if mode == "l2" else source_obj.source_record.topology),
                )
                state["source_row_count"] += 1
                duplicate_topology = row_key in seen_topologies
                seen_topologies.add(row_key)
                if duplicate_topology or state["terminal"]:
                    state["input_error"] = True
                    state["status"] = "unknown"
                    state["terminal"] = False
                    duplicate_row = {
                        "schema_version": RUNNER_SCHEMA,
                        "source_row": _source_row(
                            row_path, source_index, before_sha,
                            source_obj.source_record.to_dict() if source_obj is not None else None,
                        ),
                        "topology_key": row_key,
                        "identity_key": _key_token(source_obj.key) if source_obj is not None else None,
                        "generic_evidence": None, "generic_verifier": None,
                        "search_result": None, "search_verifier": None,
                        "generic_status": "unknown", "status": "unknown",
                        "witness_reference": None,
                        "errors": ["duplicate_complete_topology_key"],
                    }
                    labels_writer.write(duplicate_row)
                    evaluations += 1
                    binding["evaluation_rows"] += 1
                    source_errors.append({"path": row_path, "index": source_index, "error": "duplicate_complete_topology_key"})
                    continue
                if mode == "witness":
                    evidence_row, topology_key = _evaluate_witness(
                        source_obj, source_path=row_path, source_index=source_index,
                        file_sha256=before_sha, deadline=deadline,
                        per_topology_seconds=per_topology_seconds,
                        search_backend=search_backend,
                    )
                else:
                    evidence_row, topology_key = _evaluate_l2(
                        raw_row, topology, source_path=row_path, source_index=source_index,
                        file_sha256=before_sha, deadline=deadline,
                        per_topology_seconds=per_topology_seconds,
                        search_backend=search_backend,
                    )
                evaluations += 1
                binding["evaluation_rows"] += 1
                line = labels_writer.write(evidence_row)
                state = states.setdefault(topology_key, _topology_state(topology_key, topology if mode == "l2" else source_obj.source_record.topology))
                _update_state(state, evidence_row, line)
                if _remaining(deadline) <= 0:
                    timed_out = True
            if hasattr(iterator, "close"):
                iterator.close()
            binding["read_to_eof"] = not stopped_early and not binding["iterator_error"]
            after_sha, after_size = _sha256_file(path)
            binding["sha256_after"] = after_sha
            binding["size_after"] = after_size
            binding["stable"] = after_sha == before_sha and after_size == before_size
            if not binding["stable"]:
                source_errors.append({"path": str(path), "error": "source_changed_between_hashes"})
        if _remaining(deadline) <= 0:
            timed_out = True
        if expected_count is not None and offset == 0 and limit is None and rows_seen != expected_count:
            source_errors.append({
                "error": "expected_count_mismatch",
                "expected": expected_count,
                "actual": rows_seen,
            })
        if expected_count is not None and offset == 0 and limit is None and len(states) != expected_count:
            source_errors.append({
                "error": "expected_unique_topology_count_mismatch",
                "expected": expected_count,
                "actual": len(states),
            })
    except Exception as exc:
        failed = True
        error = f"{type(exc).__name__}:{exc}"
    finally:
        labels_writer.close()
        topology_writer.close()

    source_stable = bool(source_bindings) and all(item.get("stable") is True for item in source_bindings)
    partial = offset > 0 or limit is not None or timed_out or bool(source_errors)
    topology_results: list[dict[str, Any]] = []
    for state in states.values():
        result = dict(state)
        result.pop("terminal", None)
        if not source_stable or result.get("input_error"):
            result["status"] = "unknown"
        topology_results.append(result)
        topology_writer = None
    theorem_after: dict[str, str] | None = None
    implementation_after: dict[str, str | None] | None = None
    runner_sha256: str | None = None
    binding_check_failed = False
    try:
        theorem_after = _theorem_binding(theorem)
    except Exception as exc:
        binding_check_failed = True
        failed = True
        error = error or f"theorem_binding_hash_failed:{type(exc).__name__}:{exc}"
        source_errors.append({"error": "theorem_binding_check_failed", "detail": error})
    try:
        implementation_after = _implementation_hashes()
    except Exception as exc:
        binding_check_failed = True
        failed = True
        error = error or f"implementation_binding_hash_failed:{type(exc).__name__}:{exc}"
        source_errors.append({"error": "implementation_binding_check_failed", "detail": error})
    try:
        runner_sha256 = _sha256_file(Path(__file__))[0]
    except Exception as exc:
        binding_check_failed = True
        failed = True
        error = error or f"runner_binding_hash_failed:{type(exc).__name__}:{exc}"
        source_errors.append({"error": "runner_binding_check_failed", "detail": error})
    binding_changed = theorem_after != theorem_before or implementation_after != implementation_before
    if binding_changed:
        source_errors.append({"error": "source_or_theorem_code_changed_during_run"})
    partial = offset > 0 or limit is not None or timed_out or bool(source_errors)
    integrity_ok = bool(source_stable and not binding_changed and not binding_check_failed and not failed)
    if not integrity_ok:
        for result in topology_results:
            result["status"] = "unknown"
    # Publish topology conclusions only after all source/code/theorem bindings
    # have been checked.  A binding failure therefore cannot leave a PASS in
    # the lightweight index while the manifest says incomplete.
    with _GzipJsonlWriter(output / "topology_index.jsonl.gz") as as_writer:
        for result in topology_results:
            with_result = dict(result)
            with_result["schema_version"] = RUNNER_SCHEMA
            with_result["source_binding_ok"] = integrity_ok
            with_result.pop("terminal", None)
            with_result.pop("topology", None)
            as_writer.write(with_result)
    topology_evaluation_complete = bool(
        topology_results and not partial and integrity_ok
        and all(item["status"] in {"pass", "reject"} for item in topology_results)
    )
    run_status = "failed" if failed else (
        "incomplete" if timed_out or source_errors or binding_changed or not topology_evaluation_complete else "complete"
    )
    manifest = {
        "schema_version": RUNNER_SCHEMA,
        "status": run_status,
        "failed": failed,
        "error": error,
        "runtime": {
            "mode": mode, "offset": offset, "limit": limit,
            "per_topology_seconds": per_topology_seconds, "max_seconds": max_seconds,
            "search_backend": search_backend,
            "elapsed_seconds": time.monotonic() - started,
            "timed_out": timed_out, "binding_changed": binding_changed,
            "python": sys.version.split()[0], "platform": platform.platform(),
        },
        "coverage": {
            "rows_seen": rows_seen, "evaluations": evaluations,
            "unclassified_unknown_rows": unclassified_unknown_rows,
            "expected_count": expected_count,
            "expected_count_checked": bool(expected_count is not None and not partial),
            "expected_count_match_rows": (rows_seen == expected_count) if expected_count is not None and not partial else None,
            "expected_count_match_unique_topologies": (len(states) == expected_count) if expected_count is not None and not partial else None,
            "source_record_coverage_complete": bool(source_stable and not partial),
            "topology_source_complete": bool(source_stable and not partial),
            "all_identity_coverage": False,
            "source_complete": False,
            "topology_evaluation_complete": topology_evaluation_complete,
            "evaluation_complete": topology_evaluation_complete,
        },
        "sources": source_bindings,
        "source_errors": source_errors,
        "topologies": {
            "count": len(topology_results),
            "pass": sum(item["status"] == "pass" for item in topology_results),
            "reject": sum(item["status"] == "reject" for item in topology_results),
            "unknown": sum(item["status"] == "unknown" for item in topology_results),
        },
        "bindings": {
            "theorem_before": theorem_before,
            "theorem_after": theorem_after,
            "implementation_sha256_before": implementation_before,
            "implementation_sha256_after": implementation_after,
            "binding_check_failed": binding_check_failed,
            "runner_sha256": runner_sha256,
        },
        "files": {
            "labels": "labels.jsonl.gz", "topology_index": "topology_index.jsonl.gz",
            "progress": "progress.json", "manifest": "manifest.json",
        },
    }
    _write_json_atomic(output / "manifest.json", manifest)
    _write_json_atomic(output / "progress.json", {
        "schema_version": RUNNER_SCHEMA, "status": run_status,
        "rows_seen": rows_seen, "evaluations": evaluations,
        "elapsed_seconds": manifest["runtime"]["elapsed_seconds"],
    })
    return 1 if failed else (2 if run_status == "incomplete" else 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--mode", choices=("witness", "l2"), default="witness")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-topology-seconds", type=float, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--theorem", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--search-backend", choices=GENERIC_SEARCH_BACKENDS, default="reference")
    args = parser.parse_args(argv)
    if args.offset < 0 or args.limit is not None and args.limit < 0:
        parser.error("offset and limit must be non-negative")
    if not math.isfinite(args.per_topology_seconds) or args.per_topology_seconds < 0:
        parser.error("--per-topology-seconds must be finite and non-negative")
    if not math.isfinite(args.max_seconds) or args.max_seconds < 0:
        parser.error("--max-seconds must be finite and non-negative")
    if args.expected_count is not None and args.expected_count < 0:
        parser.error("--expected-count must be non-negative")
    return run_batch(
        args.source, args.output, mode=args.mode, offset=args.offset, limit=args.limit,
        per_topology_seconds=args.per_topology_seconds, max_seconds=args.max_seconds,
        theorem=args.theorem, expected_count=args.expected_count,
        search_backend=args.search_backend,
    )


if __name__ == "__main__":
    raise SystemExit(main())
