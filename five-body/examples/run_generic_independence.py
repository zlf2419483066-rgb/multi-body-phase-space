"""Run the explicit generic independence evaluation for one source archive.

The runner reads only the directory supplied by ``--source``.  It never
searches for catalogues, enumerates labels, renders graphs, or selects a
high-circle source implicitly.  Its output directory is a new directory and
is never silently overwritten.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phase_space_graph5.independence.generic_aggregation import aggregate_generic
from phase_space_graph5.independence.sources import SourceBatch, _key_token, read_source


RUNNER_SCHEMA = "F5-I-generic-run-v1"
DEFAULT_MAX_SECONDS = 300.0
_IMPLEMENTATION_FILES = (
    "generic_family.py", "generic_verification.py", "generic_aggregation.py",
    "sources.py", "invariant_sets.py", "hajer_constraints.py", "exact_algebra.py",
    "contracts.py", "coordinates.py", "domain.py", "physical_labels.py",
    "label_identity.py", "catalog_checkpoint.py", "label_catalog.py",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def _prepare_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    else:
        path.mkdir(parents=True)


def _implementation_hashes() -> dict[str, str | None]:
    package = _SRC / "phase_space_graph5" / "independence"
    result: dict[str, str | None] = {}
    independence_names = {
        "generic_family.py", "generic_verification.py", "generic_aggregation.py",
        "sources.py", "invariant_sets.py", "hajer_constraints.py", "exact_algebra.py",
        "contracts.py", "coordinates.py",
    }
    for name in _IMPLEMENTATION_FILES:
        path = (package / name) if name in independence_names else (_SRC / "phase_space_graph5" / name)
        result[str(path.relative_to(_SRC))] = _sha256_file(path) if path.is_file() else None
    # The runner is outside the package but is part of the executable
    # evidence-producing code path.
    result[str(Path(__file__).resolve().relative_to(_HERE.parent))] = _sha256_file(Path(__file__))
    return result


def _theorem_binding(path: Path | None) -> dict[str, str | None]:
    if path is None:
        raise ValueError("an explicit theorem/spec path is required for a production run")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"theorem binding is not a file: {resolved}")
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _label_row_from_aggregation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity_key": row.get("identity_key"),
        "topology_key": row.get("topology_key"),
        "source_digest": row.get("source_digest"),
        "status": row.get("status", "unknown"),
        "valid": row.get("valid", False),
        "evidence": row.get("evidence"),
        "verification": row.get("verification"),
        "errors": row.get("errors", []),
        "witness_identity_key": row.get("witness_identity_key"),
    }


def _flatten_label_rows(aggregation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topology in aggregation.get("topologies", ()):
        if not isinstance(topology, Mapping):
            continue
        for row in topology.get("evaluations", ()):
            if isinstance(row, Mapping):
                rows.append(_label_row_from_aggregation(row))
    return rows


def _source_index(batch: SourceBatch) -> dict[str, Any]:
    """Persist the complete key index once, without copying source records."""

    return {
        "schema_version": "F5-I-generic-source-index-v1",
        "snapshot": batch.snapshot.to_dict(),
        "coverage": batch.coverage.to_dict(),
        "records": [
            {
                "identity_key": _key_token(item.key),
                "topology_key": _key_token(item.topology_key),
                "identity_digest": item.digest,
            }
            for item in batch.records
        ],
    }


def _aggregation_summary(aggregation: Mapping[str, Any]) -> dict[str, Any]:
    """Remove per-label evidence from the topology/index summary."""

    source = aggregation.get("source")
    source_summary = {
        "record_count": source.get("record_count", 0) if isinstance(source, Mapping) else 0,
        "content_digest": source.get("content_digest") if isinstance(source, Mapping) else None,
        "snapshot_fingerprint": (
            source.get("snapshot", {}).get("fingerprint")
            if isinstance(source, Mapping) and isinstance(source.get("snapshot"), Mapping)
            else None
        ),
        "index_file": "source_index.json",
    }
    topologies = []
    for topology in aggregation.get("topologies", ()):
        if not isinstance(topology, Mapping):
            continue
        topologies.append(
            {
                "topology_key": topology.get("topology_key"),
                "status": topology.get("status", "unknown"),
                "identity_keys": topology.get("identity_keys", []),
                "pass_identity_keys": topology.get("pass_identity_keys", []),
                "reject_identity_keys": topology.get("reject_identity_keys", []),
                "unknown_identity_keys": topology.get("unknown_identity_keys", []),
                "witness_identity_keys": topology.get("witness_identity_keys", []),
                "evaluation_count": len(topology.get("evaluations", ())),
            }
        )
    return {
        "schema_version": aggregation.get("schema_version"),
        "status": aggregation.get("status", "unknown"),
        "source_binding_ok": aggregation.get("source_binding_ok", False),
        "disk_coverage_verified": aggregation.get("disk_coverage_verified", False),
        "source_complete": aggregation.get("source_complete", False),
        "evaluation_complete": aggregation.get("evaluation_complete", False),
        "source": source_summary,
        "source_reasons": aggregation.get("source_reasons", []),
        "out_of_range_evaluations": aggregation.get("out_of_range_evaluations", []),
        "duplicate_identity_keys": aggregation.get("duplicate_identity_keys", []),
        "topologies": topologies,
        "counts": aggregation.get("counts", {}),
    }


def run_generic(source: Path, output: Path, max_seconds: float, theorem: Path | None = None) -> int:
    started = time.monotonic()
    deadline = started + max_seconds
    _prepare_output(output)
    labels: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    evaluation_errors: list[dict[str, Any]] = []
    aggregation: dict[str, Any] | None = None
    batch: SourceBatch | None = None
    timed_out = False
    binding_changed = False
    failed = False
    error: str | None = None
    theorem_before: dict[str, str | None] = {"path": None, "sha256": None}
    implementation_before: dict[str, str | None] = {}
    try:
        theorem_before = _theorem_binding(theorem)
        implementation_before = _implementation_hashes()
        batch = read_source(source)
        for item in batch.records:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            try:
                from phase_space_graph5.independence.generic_family import evaluate_generic_label

                evidence = evaluate_generic_label(item)
                if not isinstance(evidence, Mapping):
                    raise TypeError("evaluate_generic_label must return a mapping")
                evaluations.append(dict(evidence))
            except Exception as exc:  # preserve the failed attempt in the manifest
                evaluation_errors.append(
                    {
                        "identity_key": _key_token(item.key),
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
        if batch is not None:
            aggregation = aggregate_generic(batch, evaluations)
            labels = _flatten_label_rows(aggregation)
        if time.monotonic() >= deadline:
            timed_out = True
    except Exception as exc:
        failed = True
        error = f"{type(exc).__name__}:{exc}"

    elapsed = time.monotonic() - started
    if batch is not None and aggregation is None:
        aggregation = {
            "schema_version": "F5-I-generic-aggregation-v1",
            "status": "unknown",
            "source_binding_ok": False,
            "disk_coverage_verified": False,
            "source_complete": False,
            "evaluation_complete": False,
            "source": {"snapshot": batch.snapshot.to_dict(), "coverage": batch.coverage.to_dict()},
            "source_reasons": ["runner_failed_before_aggregation"],
            "topologies": [],
        }
    if aggregation is not None:
        _write_jsonl_atomic(output / "labels.jsonl", labels)
        if batch is not None:
            _write_atomic(output / "source_index.json", _source_index(batch))
        _write_atomic(output / "aggregation.json", _aggregation_summary(aggregation))

    try:
        theorem_after = _theorem_binding(theorem)
        implementation_after = _implementation_hashes()
    except Exception as exc:
        theorem_after = theorem_before
        implementation_after = implementation_before
        failed = True
        error = error or f"binding_hash_failed:{type(exc).__name__}:{exc}"
    if theorem_after != theorem_before or implementation_after != implementation_before:
        error = error or "source_or_theorem_changed_during_run"
        binding_changed = True

    source_snapshot = batch.snapshot.to_dict() if batch is not None else None
    evaluation_complete = bool(aggregation and aggregation.get("evaluation_complete") is True)
    source_complete = bool(aggregation and aggregation.get("source_complete") is True)
    run_status = "failed" if failed else (
        "incomplete"
        if timed_out
        or binding_changed
        or evaluation_errors
        or (aggregation is not None and aggregation.get("evaluation_complete") is not True)
        or (aggregation is not None and aggregation.get("source_binding_ok") is not True)
        else "complete"
    )
    manifest = {
        "schema_version": RUNNER_SCHEMA,
        "status": run_status,
        "failed": failed,
        "error": error,
        "runtime": {
            "attempted": bool(batch is not None),
            "elapsed_seconds": elapsed,
            "max_seconds": max_seconds,
            "timed_out": timed_out,
            "binding_changed": binding_changed,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "source": {
            "requested_path": str(source.expanduser().resolve()),
            "snapshot": source_snapshot,
            "source_complete": source_complete,
            "evaluation_complete": evaluation_complete,
            "record_count": len(batch.records) if batch is not None else 0,
            "evaluation_count": len(evaluations),
            "evaluation_errors": evaluation_errors,
        },
        "aggregation": {
            "status": aggregation.get("status") if aggregation else "unknown",
            "source_binding_ok": aggregation.get("source_binding_ok") if aggregation else False,
            "disk_coverage_verified": aggregation.get("disk_coverage_verified") if aggregation else False,
            "counts": aggregation.get("counts", {}) if aggregation else {},
            "topology_results": [
                {
                    "topology_key": item.get("topology_key"),
                    "status": item.get("status"),
                    "witness_identity_keys": item.get("witness_identity_keys", []),
                }
                for item in (aggregation.get("topologies", ()) if aggregation else ())
                if isinstance(item, Mapping)
            ],
        },
        "bindings": {
            "implementation_sha256_before": implementation_before,
            "implementation_sha256_after": implementation_after,
            "theorem_before": theorem_before,
            "theorem_after": theorem_after,
            "runner_sha256": _sha256_file(Path(__file__)),
        },
        "files": {
            "labels": "labels.jsonl" if aggregation is not None else None,
            "aggregation": "aggregation.json" if aggregation is not None else None,
            "source_index": "source_index.json" if batch is not None else None,
        },
    }
    _write_atomic(output / "manifest.json", manifest)
    return 1 if failed else (2 if run_status == "incomplete" else 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument(
        "--theorem",
        type=Path,
        required=True,
        help="explicit theorem/spec file whose SHA-256 is recorded and rechecked",
    )
    args = parser.parse_args(argv)
    if not (0 <= args.max_seconds <= 86400):
        parser.error("--max-seconds must be between 0 and 86400 seconds")
    return run_generic(args.source, args.output, args.max_seconds, args.theorem)


if __name__ == "__main__":
    raise SystemExit(main())
