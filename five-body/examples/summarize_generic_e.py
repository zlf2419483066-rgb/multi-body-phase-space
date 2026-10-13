"""Summarize explicitly planned generic-E artifacts without mathematical replay.

The summary checks artifact bindings, certificate structure, and exact topology
key coverage.  It does not replay the mathematical certificates or enumerate
topologies.  See ``audit_level`` in the emitted summary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phase_space_graph5.canonical import canonicalize_topology
from phase_space_graph5.domain import EdgeType, InternalEdge, Topology
from phase_space_graph5.independence.generic_family import GENERIC_THEOREM_VERSION
from phase_space_graph5.independence.generic_search import (
    GENERIC_SEARCH_COVERAGE_RULE,
    GENERIC_SEARCH_SCOPE,
    GENERIC_SEARCH_THEOREM_VERSION,
)
from phase_space_graph5.independence.sources import _key_token
from phase_space_graph5.label_identity import canonicalize_candidate_labeling
from phase_space_graph5.physical_labels import CandidateMomentumRecord


SCHEMA_VERSION = "F5-I-generic-E-summary-v1"
AUDIT_LEVEL = "artifact_bindings_and_certificate_structure"
_ALLOWED = frozenset({"pass", "reject", "unknown"})
_KEY_SET_ALGORITHM = "sha256(sorted(_key_token(key)) UTF-8 tokens, each followed by LF)"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _resolved(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _same_path(left: str | Path, right: str | Path, base: Path) -> bool:
    try:
        return os.path.normcase(str(_resolved(left, base))) == os.path.normcase(str(_resolved(right, base)))
    except (OSError, RuntimeError, ValueError):
        return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _key_and_token(value: Any) -> tuple[Any, str]:
    """Accept a structured complete topology key or its existing JSON token."""

    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("topology key string is not a serialized complete key") from exc
        value = decoded
    if not isinstance(value, (list, tuple)) or not value or value[0] != "topology":
        raise ValueError("expected a structured complete topology key")
    return value, _key_token(value)


def _key_for_token(token: str) -> Any:
    return json.loads(token)


def _loop_from_key(value: Any) -> int:
    key, _ = _key_and_token(value)
    if len(key) != 6 or not isinstance(key[2], int) or isinstance(key[2], bool) or not isinstance(key[5], (list, tuple)):
        raise ValueError("topology key does not carry a complete loop-order structure")
    internal_edge_count = 0
    for edge in key[5]:
        if not isinstance(edge, (list, tuple)) or len(edge) != 4:
            raise ValueError("topology key has a malformed internal edge")
        multiplicity = edge[3]
        if not isinstance(multiplicity, int) or isinstance(multiplicity, bool) or multiplicity < 0:
            raise ValueError("topology key has an invalid edge multiplicity")
        internal_edge_count += multiplicity
    return internal_edge_count - key[2] + 1


def _canonical_topology_token(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("certificate topology is not an object")
    vertices = value.get("vertices")
    raw_edges = value.get("edges")
    final_slots = value.get("final_slots")
    if not isinstance(vertices, (list, tuple)) or not isinstance(raw_edges, (list, tuple)):
        raise ValueError("certificate topology lacks vertices or edges")
    edges = tuple(InternalEdge.from_dict(edge) for edge in raw_edges)
    if isinstance(final_slots, Mapping):
        slots: Any = final_slots
    elif isinstance(final_slots, (list, tuple)):
        slots = []
        for slot in final_slots:
            if isinstance(slot, Mapping):
                slots.append((slot.get("vertex"), slot.get("count")))
            elif isinstance(slot, (list, tuple)) and len(slot) == 2:
                slots.append(tuple(slot))
            else:
                raise ValueError("certificate topology has a malformed final slot")
    else:
        raise ValueError("certificate topology lacks final slots")
    topology = Topology(
        value.get("n"), vertices, edges, value.get("initial_vertex"), slots
    )
    return _key_token(canonicalize_topology(topology).key)


def _identity_topology_token(identity_key: Any) -> str | None:
    if isinstance(identity_key, str):
        try:
            identity = json.loads(identity_key)
        except json.JSONDecodeError:
            return None
    else:
        identity = identity_key
    if not isinstance(identity, (list, tuple)) or len(identity) < 3:
        return None
    try:
        return _key_and_token(identity[2])[1]
    except Exception:
        return None


def _key_set_sha256(tokens: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for token in sorted(tokens):
        digest.update(token.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class InputTracker:
    """Hash every explicitly opened file before and after the audit."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, Any]] = {}
        self._cache: dict[tuple[str, str], Any] = {}

    def _entry(self, path: Path, kind: str) -> dict[str, Any]:
        resolved = path.resolve()
        key = str(resolved)
        if key not in self.files:
            digest, size = _sha256_file(resolved)
            self.files[key] = {
                "path": key,
                "kind": kind,
                "sha256_before": digest,
                "sha256_after": None,
                "bytes_before": size,
                "bytes_after": None,
                "rows": None,
                "status": "pass",
                "errors": [],
            }
        elif self.files[key]["kind"] != kind and kind not in self.files[key]["kind"]:
            self.files[key]["kind"] = f"{self.files[key]['kind']},{kind}"
        return self.files[key]

    def read_json(self, path: Path, *, kind: str) -> Any:
        key = (str(path.resolve()), kind)
        if key in self._cache:
            return self._cache[key]
        entry = self._entry(path, kind)
        try:
            if str(path).lower().endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
                    value = json.load(stream)
            else:
                value = json.loads(path.read_text(encoding="utf-8-sig"))
            entry["rows"] = _json_record_count(value)
            self._cache[key] = value
            return value
        except Exception as exc:  # Preserve a reviewable report for malformed input.
            entry["status"] = "unknown"
            entry["errors"].append(f"read_json:{type(exc).__name__}:{exc}")
            raise

    def read_jsonl(self, path: Path, *, kind: str, compressed: bool = False) -> list[Any]:
        key = (str(path.resolve()), kind)
        if key in self._cache:
            return self._cache[key]
        entry = self._entry(path, kind)
        rows: list[Any] = []
        opener = gzip.open if compressed or str(path).lower().endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8-sig") as stream:
                for line_number, raw in enumerate(stream):
                    if not raw.strip():
                        continue
                    try:
                        rows.append(json.loads(raw))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSONL at zero-based line {line_number}: {exc}") from exc
            entry["rows"] = len(rows)
            self._cache[key] = rows
            return rows
        except Exception as exc:
            entry["status"] = "unknown"
            entry["errors"].append(f"read_jsonl:{type(exc).__name__}:{exc}")
            raise

    def read_jsonl_selected(self, path: Path, *, kind: str, selected: set[int], compressed: bool = False) -> dict[int, Any]:
        """Scan an indexed JSONL artifact while retaining only referenced rows."""

        key = (str(path.resolve()), kind)
        if key in self._cache:
            cached = self._cache[key]
            if isinstance(cached, dict):
                return cached
        entry = self._entry(path, kind)
        result: dict[int, Any] = {}
        records = 0
        opener = gzip.open if compressed or str(path).lower().endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8-sig") as stream:
                for line_number, raw in enumerate(stream):
                    if not raw.strip():
                        continue
                    records += 1
                    if line_number not in selected:
                        continue
                    try:
                        result[line_number] = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid referenced JSONL at zero-based line {line_number}: {exc}") from exc
            entry["rows"] = records
            self._cache[key] = result
            return result
        except Exception as exc:
            entry["status"] = "unknown"
            entry["errors"].append(f"read_jsonl_selected:{type(exc).__name__}:{exc}")
            raise

    def read_jsonl_identity_keys(self, path: Path, *, kind: str, selected: set[str]) -> dict[str, list[tuple[int, Any]]]:
        """Find explicit identity-key references while retaining only matches."""

        key = (str(path.resolve()), kind)
        if key in self._cache:
            cached = self._cache[key]
            if isinstance(cached, dict):
                return cached
        entry = self._entry(path, kind)
        result: dict[str, list[tuple[int, Any]]] = defaultdict(list)
        records = 0
        opener = gzip.open if str(path).lower().endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8-sig") as stream:
                for line_number, raw in enumerate(stream):
                    if not raw.strip():
                        continue
                    records += 1
                    row = json.loads(raw)
                    identity_key = row.get("identity_key") if isinstance(row, Mapping) else None
                    if isinstance(identity_key, str) and identity_key in selected:
                        result[identity_key].append((line_number, row))
            entry["rows"] = records
            self._cache[key] = dict(result)
            return dict(result)
        except Exception as exc:
            entry["status"] = "unknown"
            entry["errors"].append(f"read_jsonl_identity_keys:{type(exc).__name__}:{exc}")
            raise

    def finalize(self) -> set[str]:
        changed: set[str] = set()
        for path, entry in self.files.items():
            try:
                digest, size = _sha256_file(Path(path))
                entry["sha256_after"] = digest
                entry["bytes_after"] = size
                if digest != entry["sha256_before"] or size != entry["bytes_before"]:
                    entry["status"] = "unknown"
                    entry["errors"].append("input_changed_during_summary")
                    changed.add(path)
            except Exception as exc:
                entry["status"] = "unknown"
                entry["errors"].append(f"post_hash:{type(exc).__name__}:{exc}")
                changed.add(path)
        return changed


def _json_record_count(value: Any) -> int:
    if isinstance(value, Mapping):
        for name in ("loops", "topologies", "topology_results", "cases", "rows"):
            rows = value.get(name)
            if isinstance(rows, list):
                return len(rows)
        return 1
    if isinstance(value, list):
        return len(value)
    return 1


def _lookup_expected(loop: int, expected_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in expected_rows:
        if isinstance(row, Mapping) and row.get("loop") == loop:
            return dict(row)
    return None


def _identity_key_token(identity_key: Any) -> str | None:
    """Normalize an identity key that may be a JSON token or a JSON value."""

    if isinstance(identity_key, str):
        try:
            identity_key = json.loads(identity_key)
        except json.JSONDecodeError:
            return None
    if not isinstance(identity_key, (list, tuple)) or len(identity_key) < 3:
        return None
    try:
        return _key_token(identity_key)
    except Exception:
        return None


def _expected_rows(plan: Mapping[str, Any], base: Path, tracker: InputTracker) -> tuple[list[dict[str, Any]], set[str], list[str]]:
    errors: list[str] = []
    global_inputs: set[str] = set()
    plan_expected = plan.get("expected_p_manifest", plan.get("expected_manifest"))
    rows: list[dict[str, Any]] = []
    if isinstance(plan_expected, Mapping) and plan_expected.get("path"):
        path = _resolved(plan_expected["path"], base)
        global_inputs.add(str(path))
        try:
            value = tracker.read_json(path, kind="expected_p_manifest")
            entry = tracker.files[str(path)]
            required_sha = plan_expected.get("sha256")
            if not isinstance(required_sha, str) or len(required_sha) != 64:
                errors.append("expected_p_manifest_sha256_required")
            elif entry["sha256_before"] != required_sha.lower():
                errors.append("expected_p_manifest_sha256_mismatch")
            if not isinstance(value, Mapping) or not isinstance(value.get("loops"), list):
                errors.append("expected_p_manifest_must_contain_loops_array")
            else:
                if value.get("schema") != "F5-P-expected-key-inventory-v1":
                    errors.append("expected_p_manifest_schema_unsupported")
                algorithm = value.get("key_set_digest_algorithm")
                rows = [dict(item) for item in value["loops"] if isinstance(item, Mapping)]
                for row in rows:
                    if algorithm is not None:
                        row.setdefault("key_set_digest_algorithm", algorithm)
                if len(rows) != len(value["loops"]):
                    errors.append("expected_p_manifest_has_non_object_loop")
                if value.get("errors"):
                    errors.append("expected_p_manifest_reports_errors")
                for row in rows:
                    errors.extend(_validate_expected_row(row))
        except Exception as exc:
            errors.append(f"expected_p_manifest_read_failed:{type(exc).__name__}:{exc}")
    elif isinstance(plan.get("loops"), list) and all(isinstance(row, Mapping) and isinstance(row.get("expected_keys"), list) for row in plan["loops"]):
        rows = [{"loop": row.get("loop"), "topology_keys": row.get("expected_keys")} for row in plan["loops"]]
        for row in rows:
            errors.extend(_validate_expected_row(row))
    else:
        errors.append("expected_p_manifest_or_explicit_expected_keys_required")
    return rows, global_inputs, errors


def _validate_expected_row(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    loop = row.get("loop")
    if row.get("errors"):
        errors.append(f"loop_{loop}:expected_inventory_reports_errors")
    if not isinstance(loop, int) or isinstance(loop, bool):
        errors.append("expected_loop_number_invalid")
        return errors
    keys = row.get("topology_keys", row.get("expected_keys"))
    if not isinstance(keys, list):
        return [f"loop_{loop}:expected_topology_keys_missing"]
    tokens: list[str] = []
    try:
        key_pairs = [_key_and_token(key) for key in keys]
        tokens = [token for _, token in key_pairs]
    except Exception as exc:
        errors.append(f"loop_{loop}:invalid_expected_topology_key:{type(exc).__name__}:{exc}")
        key_pairs = []
    unique = set(tokens)
    for name, actual in (("expected_count", len(keys)), ("observed_unique", len(unique))):
        declared = row.get(name)
        if declared is not None and (
            not isinstance(declared, int) or isinstance(declared, bool) or declared != actual
        ):
            errors.append(f"loop_{loop}:{name}_mismatch")
    if len(unique) != len(tokens):
        errors.append(f"loop_{loop}:duplicate_key_in_expected_manifest")
    for key, _ in key_pairs:
        try:
            if _loop_from_key(key) != loop:
                errors.append(f"loop_{loop}:expected_topology_key_loop_mismatch")
        except Exception as exc:
            errors.append(f"loop_{loop}:expected_topology_key_loop_invalid:{type(exc).__name__}:{exc}")
    declared_algorithm = row.get("key_set_digest_algorithm")
    declared_digest = row.get("key_set_sha256", row.get("key_set_digest"))
    if declared_digest is not None:
        if declared_algorithm is not None and declared_algorithm != _KEY_SET_ALGORITHM:
            errors.append(f"loop_{loop}:unsupported_key_set_digest_algorithm")
        elif declared_digest != _key_set_sha256(tokens):
            errors.append(f"loop_{loop}:key_set_sha256_mismatch")
    source_files = row.get("source_files", ())
    if "source_files" in row and not isinstance(source_files, list):
        errors.append(f"loop_{loop}:expected_source_files_not_array")
        source_files = ()
    if "source_files" in row and not source_files:
        errors.append(f"loop_{loop}:expected_source_files_empty")
    if "source_files" in row and (
        not isinstance(row.get("expected_count"), int) or isinstance(row.get("expected_count"), bool)
        or not isinstance(row.get("observed_unique"), int) or isinstance(row.get("observed_unique"), bool)
    ):
        errors.append(f"loop_{loop}:expected_counts_missing_or_invalid")
    for source in source_files:
        if not isinstance(source, Mapping):
            errors.append(f"loop_{loop}:invalid_source_file_binding")
            continue
        before = source.get("sha256_before")
        after = source.get("sha256_after")
        if not isinstance(source.get("path"), str) or not source.get("path"):
            errors.append(f"loop_{loop}:expected_source_path_missing")
        if not _is_sha256(before) or not _is_sha256(after):
            errors.append(f"loop_{loop}:expected_source_sha256_invalid")
        size_before = source.get("size_before", source.get("size_bytes", source.get("size")))
        size_after = source.get("size_after", source.get("size_bytes", source.get("size")))
        if (
            not isinstance(size_before, int) or isinstance(size_before, bool) or size_before < 0
            or not isinstance(size_after, int) or isinstance(size_after, bool) or size_after < 0
        ):
            errors.append(f"loop_{loop}:expected_source_size_invalid")
        if not isinstance(before, str) or before != after:
            errors.append(f"loop_{loop}:unstable_expected_source_sha256")
        if source.get("source_sha_stable") is not True:
            errors.append(f"loop_{loop}:expected_source_sha_not_stable")
        if size_before is not None and size_after is not None and size_before != size_after:
            errors.append(f"loop_{loop}:unstable_expected_source_size")
    return errors


def _manifest_path(config: Mapping[str, Any], key: str, base: Path) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str):
        return None
    return _resolved(value, base)


def _check_explicit_file_sha(config: Mapping[str, Any], name: str, path: Path, tracker: InputTracker) -> list[str]:
    required = config.get(f"{name}_sha256")
    if required is None and isinstance(config.get(name), Mapping):
        required = config[name].get("sha256")
    if required is None:
        return []
    actual = tracker.files[str(path.resolve())]["sha256_before"]
    return [] if isinstance(required, str) and required.lower() == actual else [f"{name}_sha256_mismatch"]


def _bindings_ok(manifest: Mapping[str, Any], required_theorem_sha: str | None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping):
        return False, ["manifest_bindings_missing"]
    theorem_before = bindings.get("theorem_before")
    theorem_after = bindings.get("theorem_after")
    if not isinstance(theorem_before, Mapping) or not isinstance(theorem_after, Mapping):
        errors.append("manifest_theorem_before_after_missing")
    elif theorem_before != theorem_after:
        errors.append("manifest_theorem_binding_changed")
    elif not isinstance(theorem_before.get("path"), str) or not theorem_before.get("path") or not _is_sha256(theorem_before.get("sha256")):
        errors.append("manifest_theorem_binding_invalid")
    elif required_theorem_sha is None or theorem_before.get("sha256") != required_theorem_sha.lower():
        errors.append("manifest_theorem_sha256_missing_or_mismatch")
    code_before = bindings.get("implementation_sha256_before")
    code_after = bindings.get("implementation_sha256_after")
    if not isinstance(code_before, Mapping) or not isinstance(code_after, Mapping):
        errors.append("manifest_implementation_sha256_before_after_missing")
    elif code_before != code_after:
        errors.append("manifest_implementation_binding_changed")
    elif not code_before or not all(_is_sha256(value) for value in code_before.values()):
        errors.append("manifest_implementation_sha256_invalid_or_empty")
    runtime = manifest.get("runtime")
    if isinstance(runtime, Mapping) and runtime.get("binding_changed") is True:
        errors.append("manifest_runtime_binding_changed")
    elif not isinstance(runtime, Mapping):
        errors.append("manifest_runtime_binding_missing")
    if bindings.get("binding_check_failed") is True:
        errors.append("manifest_binding_check_failed")
    return not errors, errors


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _check_batch_manifest(manifest: Mapping[str, Any], required_theorem_sha: str | None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if manifest.get("schema_version") != "F5-I-generic-witness-batch-v1":
        errors.append("batch_manifest_schema_unsupported")
    run_state = manifest.get("status")
    runtime = manifest.get("runtime")
    if run_state not in {"complete", "incomplete"}:
        errors.append("batch_run_failed_or_unrecognized_state")
    if manifest.get("failed") is True:
        errors.append("batch_manifest_failed")
    if not isinstance(runtime, Mapping):
        errors.append("batch_runtime_binding_missing")
    elif runtime.get("binding_changed") is True:
        errors.append("batch_binding_changed")
    source_errors = manifest.get("source_errors", [])
    if source_errors:
        errors.append("batch_source_errors_present")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("batch_source_bindings_missing")
    else:
        for i, source in enumerate(sources):
            if not isinstance(source, Mapping):
                errors.append(f"batch_source_binding_{i}_invalid")
                continue
            if source.get("stable") is not True:
                errors.append(f"batch_source_binding_{i}_unstable")
            if source.get("sha256_before") != source.get("sha256_after"):
                errors.append(f"batch_source_binding_{i}_sha256_changed")
            if source.get("size_before") != source.get("size_after"):
                errors.append(f"batch_source_binding_{i}_size_changed")
    binding_ok, binding_errors = _bindings_ok(manifest, required_theorem_sha)
    errors.extend(binding_errors)
    return not errors and binding_ok, errors


def _check_legacy_manifest(manifest: Mapping[str, Any], required_theorem_sha: str | None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if manifest.get("status") != "complete":
        errors.append("legacy_run_not_complete")
    if manifest.get("failed") is True or manifest.get("error"):
        errors.append("legacy_manifest_failed")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or source.get("source_complete") is not True or source.get("evaluation_complete") is not True:
        errors.append("legacy_source_or_evaluation_incomplete")
    snapshot = source.get("snapshot") if isinstance(source, Mapping) else None
    if not isinstance(snapshot, Mapping) or not snapshot.get("content_sha256"):
        errors.append("legacy_source_snapshot_binding_missing")
    binding_ok, binding_errors = _bindings_ok(manifest, required_theorem_sha)
    errors.extend(binding_errors)
    return not errors and binding_ok, errors


def _legacy_rows(
    run: Mapping[str, Any], loop: int, base: Path, tracker: InputTracker, theorem_sha: str | None
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    manifest_path = _manifest_path(run, "manifest", base)
    aggregation_path = _manifest_path(run, "aggregation", base)
    if manifest_path is None or aggregation_path is None:
        return [], ["legacy_manifest_and_aggregation_paths_required"]
    try:
        manifest = tracker.read_json(manifest_path, kind="legacy_manifest")
        aggregation = tracker.read_json(aggregation_path, kind="legacy_aggregation")
        errors.extend(_check_explicit_file_sha(run, "manifest", manifest_path, tracker))
        errors.extend(_check_explicit_file_sha(run, "aggregation", aggregation_path, tracker))
    except Exception as exc:
        return [], errors + [f"legacy_input_read_failed:{type(exc).__name__}:{exc}"]
    if not isinstance(manifest, Mapping):
        return [], errors + ["legacy_manifest_not_object"]
    valid_manifest, binding_errors = _check_legacy_manifest(manifest, theorem_sha)
    errors.extend(binding_errors)
    if not isinstance(aggregation, Mapping):
        return [], errors + ["legacy_aggregation_not_object"]
    if aggregation.get("schema_version") != "F5-I-generic-aggregation-v1":
        errors.append("legacy_aggregation_schema_unsupported")
    manifest_files = manifest.get("files", {})
    declared_aggregation = manifest_files.get("aggregation") if isinstance(manifest_files, Mapping) else None
    declared_labels = manifest_files.get("labels") if isinstance(manifest_files, Mapping) else None
    if declared_aggregation and not _same_path(declared_aggregation, aggregation_path, manifest_path.parent):
        errors.append("legacy_aggregation_path_does_not_match_manifest")
    labels_path = _manifest_path(run, "labels", base)
    if labels_path is None and isinstance(declared_labels, str):
        labels_path = _resolved(declared_labels, manifest_path.parent)
    if labels_path is None:
        errors.append("legacy_labels_path_required_for_certificate_refs")
    elif declared_labels and not _same_path(declared_labels, labels_path, manifest_path.parent):
        errors.append("legacy_labels_path_does_not_match_manifest")
    if aggregation.get("status") != "pass" or aggregation.get("disk_coverage_verified") is not True or aggregation.get("source_binding_ok") is not True or aggregation.get("evaluation_complete") is not True:
        errors.append("legacy_aggregation_not_complete_and_bound")
    source_snapshot = manifest.get("source", {}).get("snapshot", {}) if isinstance(manifest.get("source"), Mapping) else {}
    source_record = aggregation.get("source", {})
    if isinstance(source_snapshot, Mapping) and isinstance(source_record, Mapping):
        if source_record.get("content_digest") != source_snapshot.get("content_sha256"):
            errors.append("legacy_aggregation_source_digest_mismatch")
        if source_record.get("snapshot_fingerprint") != source_snapshot.get("fingerprint"):
            errors.append("legacy_aggregation_source_fingerprint_mismatch")
    rows = aggregation.get("topologies", aggregation.get("topology_results"))
    if not isinstance(rows, list):
        return [], errors + ["legacy_aggregation_topologies_missing"]
    selected_identities: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        identities = row.get("witness_identity_keys", [])
        if isinstance(identities, list):
            selected_identities.update(item for item in identities if isinstance(item, str))
    label_rows: dict[str, list[tuple[int, Any]]] = {}
    if labels_path is not None:
        try:
            label_rows = tracker.read_jsonl_identity_keys(
                labels_path, kind="legacy_labels", selected=selected_identities
            )
            errors.extend(_check_explicit_file_sha(run, "labels", labels_path, tracker))
        except Exception as exc:
            errors.append(f"legacy_labels_read_failed:{type(exc).__name__}:{exc}")
    output: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"legacy_topology_row_{i}_invalid")
            continue
        try:
            key, token = _key_and_token(row.get("topology_key"))
        except Exception as exc:
            errors.append(f"legacy_topology_row_{i}_invalid_key:{type(exc).__name__}:{exc}")
            continue
        witnesses = row.get("witness_identity_keys", [])
        identities = row.get("identity_keys", row.get("pass_identity_keys", []))
        if not isinstance(witnesses, list):
            witnesses = []
        if not isinstance(identities, list):
            identities = []
        witnesses_valid = bool(witnesses) and all(item in identities for item in witnesses)
        witnesses_digest = hashlib.sha256(
            b"".join(str(item).encode("utf-8") + b"\n" for item in sorted(witnesses))
        ).hexdigest() if witnesses else None
        status = "unknown"
        reject_basis = None
        row_errors: list[str] = []
        actual_certificate: tuple[int, Mapping[str, Any], str] | None = None
        duplicated_witness_rows = [
            witness_identity for witness_identity in witnesses
            if isinstance(witness_identity, str) and len(label_rows.get(witness_identity, [])) > 1
        ]
        if duplicated_witness_rows:
            row_errors.append("legacy_witness_identity_has_duplicate_labels_rows")
        if row.get("status") == "pass" and witnesses_valid and labels_path is not None:
            valid_matches: list[tuple[int, Mapping[str, Any], str]] = []
            for witness_identity in witnesses:
                identity_rows = label_rows.get(witness_identity, []) if isinstance(witness_identity, str) else []
                if len(identity_rows) != 1:
                    continue
                label_line, label = identity_rows[0]
                if not isinstance(label, Mapping):
                    continue
                label_key = label.get("topology_key")
                try:
                    _, label_token = _key_and_token(label_key)
                except Exception:
                    continue
                evidence = label.get("evidence")
                verification = label.get("verification")
                identity = label.get("identity_key")
                if (
                    label_token != token
                    or identity != witness_identity
                    or label.get("status") != "pass"
                    or label.get("valid") is not True
                    or label.get("errors")
                    or not isinstance(evidence, Mapping)
                    or not isinstance(verification, Mapping)
                    or evidence.get("identity_key") != witness_identity
                    or label.get("source_digest") != evidence.get("source_digest")
                ):
                    continue
                adapted = {
                    "status": label.get("status"),
                    "generic_status": label.get("status"),
                    "identity_key": identity,
                    "generic_evidence": evidence,
                    "generic_verifier": verification,
                }
                certificate_status, _, _, certificate_errors = _certificate_status(adapted, token, loop)
                if certificate_status == "pass" and not certificate_errors:
                    valid_matches.append((label_line, label, witness_identity))
            if valid_matches and not duplicated_witness_rows:
                # Aggregation may name many distinct passing identities.  One
                # complete bound certificate suffices existentially; choose
                # the same reference deterministically on every run.
                actual_certificate = min(valid_matches, key=lambda item: (item[2], item[0]))
            elif not valid_matches:
                row_errors.append("legacy_aggregation_witness_has_no_valid_labels_certificate")
        if not valid_manifest or errors:
            row_errors.extend(errors)
        elif row.get("status") == "pass" and witnesses_valid and actual_certificate is not None and not row_errors:
            status = "pass"
        elif row.get("status") == "reject":
            row_errors.append("legacy_candidate_rejects_do_not_prove_topology_reject")
            reject_basis = {"kind": "legacy_aggregation_candidate_rejects_not_full_search"}
        else:
            if not row_errors:
                row_errors.append("legacy_aggregation_has_no_valid_topology_witness")
        certificate_refs = []
        if status == "pass" and actual_certificate is not None and labels_path is not None:
            label_line, _, witness_identity = actual_certificate
            certificate_refs.append({
                "kind": "legacy_generic_label",
                "status": "pass",
                "path": str(labels_path),
                "row": label_line,
                "identity_key": witness_identity,
                "labels_sha256_before": tracker.files[str(labels_path.resolve())]["sha256_before"],
            })
        output.append({
            "loop": loop,
            "topology_key": key,
            "topology_key_token": token,
            "status": status,
            "source_type": "legacy",
            "certificate_refs": certificate_refs,
            "reject_basis": reject_basis,
            "errors": sorted(set(row_errors)),
            "input_files": [str(manifest_path), str(aggregation_path)] + ([str(labels_path)] if labels_path is not None else []),
        })
    return output, errors


def _l2_rows(run: Mapping[str, Any], loop: int, base: Path, tracker: InputTracker, theorem_sha: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    report_path = _manifest_path(run, "report", base)
    if report_path is None:
        return [], ["l2_report_path_required"]
    try:
        report = tracker.read_json(report_path, kind="l2_report")
        errors.extend(_check_explicit_file_sha(run, "report", report_path, tracker))
    except Exception as exc:
        return [], [f"l2_report_read_failed:{type(exc).__name__}:{exc}"]
    if not isinstance(report, Mapping) or not isinstance(report.get("cases"), list):
        return [], errors + ["l2_report_cases_missing"]
    source = report.get("source", {})
    implementation = report.get("implementation", {})
    if isinstance(source, Mapping):
        before = source.get("manifest_sha256_before")
        after = source.get("manifest_sha256_after")
        if before is not None and after is not None and before != after:
            errors.append("l2_source_manifest_binding_changed")
    if isinstance(implementation, Mapping):
        before = implementation.get("sha256")
        after = implementation.get("sha256_after")
        if before is not None and after is not None and before != after:
            errors.append("l2_implementation_binding_changed")
    report_theorem_sha = None
    bindings = report.get("bindings")
    if isinstance(bindings, Mapping):
        before = bindings.get("theorem_before")
        after = bindings.get("theorem_after")
        if isinstance(before, Mapping) and isinstance(after, Mapping) and before == after:
            report_theorem_sha = before.get("sha256")
    theorem_bound = isinstance(report_theorem_sha, str) and theorem_sha is not None and report_theorem_sha == theorem_sha.lower()
    if not theorem_bound:
        errors.append("l2_report_theorem_sha256_unavailable_or_mismatched")
    code_stable = not (isinstance(implementation, Mapping) and isinstance(implementation.get("sha256"), Mapping) and isinstance(implementation.get("sha256_after"), Mapping) and implementation["sha256"] != implementation["sha256_after"])
    source_stable = not (isinstance(source, Mapping) and source.get("manifest_sha256_before") is not None and source.get("manifest_sha256_before") != source.get("manifest_sha256_after"))
    report_ok = report.get("status") == "pass" and code_stable and source_stable and theorem_bound and not errors
    output: list[dict[str, Any]] = []
    for i, case in enumerate(report["cases"]):
        if not isinstance(case, Mapping):
            errors.append(f"l2_case_{i}_invalid")
            continue
        try:
            key, token = _key_and_token(case.get("topology_key"))
        except Exception as exc:
            errors.append(f"l2_case_{i}_invalid_key:{type(exc).__name__}:{exc}")
            continue
        search = case.get("search", {})
        verifier = case.get("verifier", {})
        status = "unknown"
        cert_ref = {
            "kind": "l2_full_report_case",
            "path": str(report_path),
            "row": i,
            "manifest_row_number": case.get("manifest_row_number"),
        }
        row_errors: list[str] = []
        if not report_ok:
            row_errors.extend(errors)
        elif case.get("canonical_matches_source") is not True:
            row_errors.append("l2_case_canonical_source_mismatch")
        elif not isinstance(search, Mapping) or not isinstance(verifier, Mapping):
            row_errors.append("l2_case_certificate_missing")
        elif search.get("status") == "pass" and verifier.get("valid") is True and verifier.get("status") == "pass":
            status = "pass"
        elif (
            search.get("status") == "reject"
            and search.get("complete") is True
            and isinstance(search.get("L"), int)
            and not isinstance(search.get("L"), bool)
            and search.get("L") >= 0
            and search.get("attempted") == 5 ** search.get("L")
            and search.get("expected") == 5 ** search.get("L")
            and verifier.get("valid") is True
            and verifier.get("status") == "reject"
        ):
            status = "reject"
        else:
            row_errors.append("l2_case_certificate_not_conclusive")
        output.append({
            "loop": loop,
            "topology_key": key,
            "topology_key_token": token,
            "status": status,
            "source_type": "l2",
            "certificate_refs": [cert_ref] if status in {"pass", "reject"} else [cert_ref],
            "reject_basis": ({
                "kind": "verified_complete_5_power_L_search",
                "L": search.get("L"), "attempted": search.get("attempted"),
                "expected": search.get("expected"),
            } if status == "reject" else None),
            "errors": sorted(set(row_errors)),
            "input_files": [str(report_path)],
        })
    return output, errors


def _source_row_matches(source_row: Any, source_bindings: list[Any]) -> tuple[bool, str | None]:
    if not isinstance(source_row, Mapping):
        return False, "certificate_source_row_missing"
    source_path = source_row.get("path")
    source_sha = source_row.get("file_sha256")
    if not isinstance(source_path, str) or not isinstance(source_sha, str):
        return False, "certificate_source_binding_missing"
    for binding in source_bindings:
        if not isinstance(binding, Mapping):
            continue
        bound_path = binding.get("path")
        if not isinstance(bound_path, str):
            continue
        if os.path.normcase(source_path) == os.path.normcase(bound_path) and source_sha == binding.get("sha256_before"):
            return True, None
    return False, "certificate_source_row_not_bound_by_manifest"


def _evidence_binds_topology(evidence: Mapping[str, Any], token: str) -> bool:
    try:
        _, evidence_token = _key_and_token(evidence.get("topology_key"))
    except Exception:
        return False
    identity_key = evidence.get("identity_key")
    if not isinstance(identity_key, str) or _identity_topology_token(identity_key) != token:
        return False
    identity = evidence.get("identity")
    if not isinstance(identity, Mapping):
        return False
    nested_identity_key = identity.get("identity_key")
    if _identity_key_token(nested_identity_key) != _identity_key_token(identity_key):
        return False
    if _identity_topology_token(nested_identity_key) != token:
        return False
    source = evidence.get("source")
    try:
        _, source_token = _key_and_token(source.get("topology_key")) if isinstance(source, Mapping) else (None, None)
    except Exception:
        source_token = None
    return evidence_token == token and source_token == token


def _search_binding_errors(search: Mapping[str, Any], token: str, loop: int) -> list[str]:
    errors: list[str] = []
    if search.get("schema_version") != "F5-I-generic-search-v1":
        errors.append("search_schema_unsupported")
    search_topology = search.get("topology")
    search_topology_matches = False
    try:
        if _canonical_topology_token(search_topology) != token:
            errors.append("search_topology_key_mismatch")
        else:
            search_topology_matches = True
    except Exception as exc:
        errors.append(f"search_topology_invalid:{type(exc).__name__}:{exc}")
    try:
        if _loop_from_key(_key_for_token(token)) != loop:
            errors.append("search_topology_loop_mismatch")
    except Exception as exc:
        errors.append(f"search_topology_loop_invalid:{type(exc).__name__}:{exc}")
    if search.get("coverage_theorem_version") != GENERIC_SEARCH_THEOREM_VERSION:
        errors.append("search_theorem_version_mismatch")
    if search.get("coverage_rule") != GENERIC_SEARCH_COVERAGE_RULE:
        errors.append("search_coverage_rule_mismatch")
    if search.get("coverage_scope") != GENERIC_SEARCH_SCOPE:
        errors.append("search_coverage_scope_mismatch")
    search_l = search.get("L")
    single_order = search.get("single_occurrence_order")
    if not isinstance(search_l, int) or isinstance(search_l, bool) or search_l < 0:
        errors.append("search_L_invalid")
    elif not isinstance(single_order, list) or len(single_order) != search_l:
        errors.append("search_L_does_not_match_single_occurrence_order")
    expected_order = "product(range(1,6), repeat=L)" if isinstance(search_l, int) and not isinstance(search_l, bool) and search_l >= 0 else None
    if search.get("assignment_order") != expected_order:
        errors.append("search_assignment_order_mismatch")
    if search_topology_matches:
        try:
            raw_edges = search_topology.get("edges") if isinstance(search_topology, Mapping) else None
            if not isinstance(raw_edges, list):
                raise ValueError("search topology edges are not an array")
            # The runner indexes this order before canonical relabeling can reorder edges.
            expected_single_order = []
            for edge_index, raw_edge in enumerate(raw_edges):
                edge = InternalEdge.from_dict(raw_edge)
                if edge.edge_type == EdgeType.SINGLE:
                    expected_single_order.extend(
                        {"edge_index": edge_index, "occurrence": occurrence}
                        for occurrence in range(edge.expanded_count)
                    )
            if single_order != expected_single_order:
                errors.append("search_single_occurrence_order_mismatch")
        except Exception:
            errors.append("search_single_occurrence_order_unverifiable")
    return errors


def _certificate_status(row: Mapping[str, Any], token: str, loop: int) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if row.get("errors"):
        return "unknown", None, None, ["certificate_row_errors_present"]
    if _loop_from_key(_key_for_token(token)) != loop:
        return "unknown", None, None, ["certificate_topology_loop_mismatch"]
    generic = row.get("generic_evidence")
    generic_verifier = row.get("generic_verifier")
    search = row.get("search_result")
    search_verifier = row.get("search_verifier")
    if isinstance(generic, Mapping) and isinstance(generic_verifier, Mapping):
        if not _evidence_binds_topology(generic, token):
            errors.append("generic_evidence_identity_or_topology_binding_mismatch")
        outer_identity_key = row.get("identity_key")
        if (
            outer_identity_key is not None
            and _identity_key_token(outer_identity_key) != _identity_key_token(generic.get("identity_key"))
        ):
            errors.append("generic_evidence_identity_does_not_match_label_row")
        if (
            row.get("status") == "pass"
            and row.get("generic_status") == "pass"
            and generic.get("schema_version") == "F5-I-generic-evidence-v1"
            and generic.get("theorem_version") == GENERIC_THEOREM_VERSION
            and generic.get("status") == "pass"
            and generic_verifier.get("valid") is True
            and generic_verifier.get("status") == "pass"
            and not generic_verifier.get("errors")
            and not errors
        ):
            return "pass", {"kind": "generic_label", "status": "pass"}, None, errors
    if isinstance(search, Mapping) and isinstance(search_verifier, Mapping):
        errors.extend(_search_binding_errors(search, token, loop))
        if search_verifier.get("valid") is not True:
            errors.append("search_verifier_invalid")
        if search_verifier.get("errors"):
            errors.append("search_verifier_errors_present")
        verifier_checks = set(search_verifier.get("checks", [])) if isinstance(search_verifier.get("checks"), list) else set()
        if row.get("status") == "pass" and search.get("status") == "pass" and search_verifier.get("status") == "pass":
            witness = search.get("witness")
            witness_errors: list[str] = []
            if not isinstance(witness, Mapping):
                witness_errors.append("search_pass_witness_missing")
            else:
                assignment = witness.get("assignment")
                evidence = witness.get("evidence")
                record = witness.get("record")
                witness_verifier = witness.get("verifier")
                if not isinstance(assignment, list) or len(assignment) != search.get("L"):
                    witness_errors.append("search_pass_assignment_missing_or_wrong_length")
                if not isinstance(evidence, Mapping) or not isinstance(record, Mapping) or not isinstance(witness_verifier, Mapping):
                    witness_errors.append("search_pass_witness_record_or_evidence_missing")
                else:
                    if (
                        evidence.get("schema_version") != "F5-I-generic-evidence-v1"
                        or evidence.get("status") != "pass"
                        or evidence.get("theorem_version") != GENERIC_THEOREM_VERSION
                        or not _evidence_binds_topology(evidence, token)
                    ):
                        witness_errors.append("search_pass_witness_evidence_binding_invalid")
                    try:
                        parsed_record = CandidateMomentumRecord.from_dict(record)
                        if _canonical_topology_token(record.get("topology")) != token:
                            witness_errors.append("search_pass_witness_record_topology_mismatch")
                        identity = canonicalize_candidate_labeling(parsed_record, "fixed_labels")
                        if _identity_key_token(evidence.get("identity_key")) != _key_token(identity.key):
                            witness_errors.append("search_pass_witness_identity_does_not_match_record")
                        record_assignment = [
                            item.particle_id for item in parsed_record.complete_labeling.single_lines
                        ]
                        if assignment != record_assignment:
                            witness_errors.append("search_pass_assignment_does_not_match_record")
                    except Exception:
                        witness_errors.append("search_pass_witness_record_topology_missing")
                    if (
                        witness_verifier.get("valid") is not True
                        or witness_verifier.get("status") != "pass"
                        or witness_verifier.get("errors")
                    ):
                        witness_errors.append("search_pass_stored_witness_verifier_invalid")
                required_checks = {
                    "topology_replayed", "assignment_prefix_replayed", "stream_replayed",
                    "pass_prefix_verified", "witness_topology_verified", "witness_generic_certificate_verified",
                }
                if not required_checks <= verifier_checks:
                    witness_errors.append("search_pass_verifier_checks_incomplete")
            if not errors and not witness_errors:
                return "pass", {"kind": "generic_search", "status": "pass"}, None, []
            errors.extend(witness_errors)
        if row.get("status") == "reject" and search.get("status") == "reject":
            search_l = search.get("L")
            full_space = (
                isinstance(search_l, int) and not isinstance(search_l, bool) and search_l >= 0
                and search.get("complete") is True
                and search.get("attempted") == 5 ** search_l
                and search.get("expected") == 5 ** search_l
            )
            momentum_pass = search.get("momentum_pass")
            momentum_reject = search.get("momentum_reject")
            independent_reject = search.get("independent_reject")
            counts_valid = (
                isinstance(momentum_pass, int) and not isinstance(momentum_pass, bool) and momentum_pass >= 0
                and isinstance(momentum_reject, int) and not isinstance(momentum_reject, bool) and momentum_reject >= 0
                and isinstance(independent_reject, int) and not isinstance(independent_reject, bool) and independent_reject >= 0
                and momentum_pass + momentum_reject == search.get("attempted")
                and independent_reject == momentum_pass
            )
            required_checks = {
                "topology_replayed", "full_assignment_space_replayed", "stream_replayed",
                "reject_full_cover_verified",
            }
            if (
                not errors and full_space and counts_valid
                and search_verifier.get("valid") is True
                and search_verifier.get("status") == "reject"
                and required_checks <= verifier_checks
                and search.get("witness") is None
                and not search.get("errors")
            ):
                basis = {
                    "kind": "verified_complete_5_power_L_search",
                    "L": search_l,
                    "attempted": search.get("attempted"),
                    "expected": search.get("expected"),
                    "momentum_pass": momentum_pass,
                    "momentum_reject": momentum_reject,
                    "independent_reject": independent_reject,
                }
                return "reject", {"kind": "generic_search", "status": "reject"}, basis, []
            errors.append("search_reject_not_full_verified_5_power_L")
    if row.get("status") in {"pass", "reject"}:
        errors.append("topology_index_claim_not_supported_by_certificate")
    return "unknown", None, None, sorted(set(errors))


def _batch_rows(run: Mapping[str, Any], loop: int, base: Path, tracker: InputTracker, theorem_sha: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    manifest_path = _manifest_path(run, "manifest", base)
    if manifest_path is None:
        return [], ["batch_manifest_path_required"]
    try:
        manifest = tracker.read_json(manifest_path, kind="batch_manifest")
        if not isinstance(manifest, Mapping):
            return [], ["batch_manifest_not_object"]
        files = manifest.get("files", {})
        index_path = _manifest_path(run, "topology_index", base)
        labels_path = _manifest_path(run, "labels", base)
        if index_path is None and isinstance(files, Mapping) and files.get("topology_index"):
            index_path = _resolved(files["topology_index"], manifest_path.parent)
        if labels_path is None and isinstance(files, Mapping) and files.get("labels"):
            labels_path = _resolved(files["labels"], manifest_path.parent)
        if index_path is None or labels_path is None:
            return [], ["batch_topology_index_and_labels_paths_required"]
        index_rows = tracker.read_jsonl(index_path, kind="batch_topology_index", compressed=True)
        label_path = labels_path
        selected_rows = {
            value
            for row in index_rows if isinstance(row, Mapping)
            for value in row.get("certificate_rows", [])
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        }
        label_rows = tracker.read_jsonl_selected(
            label_path, kind="batch_labels", selected=selected_rows, compressed=True
        )
        file_binding_errors = []
        file_binding_errors.extend(_check_explicit_file_sha(run, "manifest", manifest_path, tracker))
        file_binding_errors.extend(_check_explicit_file_sha(run, "topology_index", index_path, tracker))
        file_binding_errors.extend(_check_explicit_file_sha(run, "labels", label_path, tracker))
        for field, actual_path in (("topology_index", index_path), ("labels", label_path)):
            declared = files.get(field) if isinstance(files, Mapping) else None
            if not isinstance(declared, str) or not _same_path(declared, actual_path, manifest_path.parent):
                file_binding_errors.append(f"batch_{field}_path_does_not_match_manifest")
    except Exception as exc:
        return [], errors + [f"batch_input_read_failed:{type(exc).__name__}:{exc}"]
    manifest_ok, binding_errors = _check_batch_manifest(manifest, theorem_sha)
    binding_errors.extend(file_binding_errors)
    manifest_ok = manifest_ok and not file_binding_errors
    errors.extend(binding_errors)
    source_bindings = manifest.get("sources", []) if isinstance(manifest, Mapping) else []
    output: list[dict[str, Any]] = []
    for i, row in enumerate(index_rows):
        if not isinstance(row, Mapping):
            errors.append(f"batch_topology_index_row_{i}_invalid")
            continue
        try:
            key, token = _key_and_token(row.get("topology_key"))
        except Exception as exc:
            errors.append(f"batch_topology_index_row_{i}_invalid_key:{type(exc).__name__}:{exc}")
            continue
        row_errors: list[str] = []
        status = "unknown"
        cert_refs: list[dict[str, Any]] = []
        reject_basis = None
        if row.get("schema_version") != "F5-I-generic-witness-batch-v1":
            row_errors.append("topology_index_schema_unsupported")
        if not manifest_ok:
            row_errors.extend(binding_errors)
        if row.get("source_binding_ok") is not True:
            row_errors.append("topology_index_source_binding_not_ok")
        if row.get("input_error") is True:
            row_errors.append("topology_index_input_error")
        certificate_rows = row.get("certificate_rows", [])
        if not isinstance(certificate_rows, list) or not certificate_rows:
            row_errors.append("topology_index_certificate_rows_missing")
            certificate_rows = []
        valid_certificate_indices = [
            value for value in certificate_rows
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ]
        if len(valid_certificate_indices) != len(certificate_rows):
            row_errors.append("certificate_row_reference_not_nonnegative_integer")
        if len(valid_certificate_indices) != len(set(valid_certificate_indices)):
            row_errors.append("duplicate_certificate_row_reference")
        source_row_count = row.get("source_row_count", 1)
        if len(certificate_rows) > 1 or (
            isinstance(source_row_count, int) and not isinstance(source_row_count, bool) and source_row_count > 1
        ):
            row_errors.append("duplicate_complete_topology_key_within_batch")
        cert_outcomes: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]] = []
        for cert_index in certificate_rows:
            if not isinstance(cert_index, int) or isinstance(cert_index, bool) or cert_index < 0 or cert_index not in label_rows:
                row_errors.append(f"certificate_row_reference_out_of_range:{cert_index}")
                continue
            cert = label_rows[cert_index]
            if not isinstance(cert, Mapping):
                row_errors.append(f"certificate_row_{cert_index}_invalid")
                continue
            try:
                _, cert_token = _key_and_token(cert.get("topology_key"))
            except Exception:
                cert_token = None
            if cert_token != token:
                row_errors.append(f"certificate_row_{cert_index}_topology_key_mismatch")
                continue
            bound, source_error = _source_row_matches(cert.get("source_row"), source_bindings)
            if not bound:
                row_errors.append(source_error or "certificate_source_binding_invalid")
                continue
            cert_status, cert_ref_kind, cert_reject_basis, cert_errors = _certificate_status(cert, token, loop)
            row_errors.extend(cert_errors)
            cert_outcomes.append((cert_status, cert_ref_kind, cert_reject_basis))
            cert_refs.append({
                "kind": cert_ref_kind.get("kind") if cert_ref_kind else "generic_certificate",
                "status": cert_ref_kind.get("status") if cert_ref_kind else "unknown",
                "labels_path": str(labels_path),
                "index": cert_index,
                "source_path": cert.get("source_row", {}).get("path") if isinstance(cert.get("source_row"), Mapping) else None,
                "source_index": cert.get("source_row", {}).get("index") if isinstance(cert.get("source_row"), Mapping) else None,
            })
            if cert_reject_basis is not None:
                reject_basis = cert_reject_basis
        if manifest_ok and row.get("schema_version") == "F5-I-generic-witness-batch-v1" and row.get("source_binding_ok") is True and row.get("input_error") is not True and len(certificate_rows) == 1 and len(cert_outcomes) == 1:
            cert_status, _, cert_reject_basis = cert_outcomes[0]
            if row.get("status") == cert_status and cert_status in {"pass", "reject"} and not row_errors:
                status = cert_status
                reject_basis = cert_reject_basis
        if row.get("status") == "reject" and status != "reject" and not row_errors:
            row_errors.append("topology_index_reject_not_supported_by_full_search_certificate")
        output.append({
            "loop": loop,
            "topology_key": key,
            "topology_key_token": token,
            "status": status,
            "source_type": "batch",
            "certificate_refs": cert_refs,
            "reject_basis": reject_basis,
            "errors": sorted(set(row_errors)),
            "input_files": [str(manifest_path), str(index_path), str(labels_path)],
            "source_run_state": manifest.get("status") if isinstance(manifest, Mapping) else None,
        })
    return output, errors


def _normalize_runs(loop_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    runs = loop_config.get("runs")
    if isinstance(runs, list):
        return [dict(run) for run in runs if isinstance(run, Mapping)]
    return [dict(loop_config)]


def _build_loop(
    loop_config: Mapping[str, Any], expected_row: Mapping[str, Any] | None,
    base: Path, tracker: InputTracker, theorem_sha: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loop_value = loop_config.get("loop")
    if not isinstance(loop_value, int) or isinstance(loop_value, bool):
        return [], {"loop": loop_value, "status": "unknown", "errors": ["plan_loop_number_invalid"]}
    errors: list[str] = []
    if expected_row is None:
        expected_tokens: list[str] = []
        expected_keys: dict[str, Any] = {}
        errors.append("expected_loop_missing")
    else:
        keys = expected_row.get("topology_keys", expected_row.get("expected_keys", []))
        expected_tokens = []
        expected_keys = {}
        if not isinstance(keys, list):
            errors.append("expected_topology_keys_missing")
        else:
            for value in keys:
                try:
                    key, token = _key_and_token(value)
                    expected_tokens.append(token)
                    expected_keys[token] = key
                except Exception as exc:
                    errors.append(f"invalid_expected_key:{type(exc).__name__}:{exc}")
    source_type = loop_config.get("source_type", loop_config.get("result_source_type"))
    if source_type not in {"legacy", "l2", "batch"}:
        errors.append("unsupported_source_type")
    candidates: list[dict[str, Any]] = []
    for run in _normalize_runs(loop_config):
        if source_type == "legacy":
            rows, run_errors = _legacy_rows(run, loop_value, base, tracker, theorem_sha)
        elif source_type == "l2":
            rows, run_errors = _l2_rows(run, loop_value, base, tracker, theorem_sha)
        elif source_type == "batch":
            rows, run_errors = _batch_rows(run, loop_value, base, tracker, theorem_sha)
        else:
            rows, run_errors = [], []
        candidates.extend(rows)
        if run_errors:
            errors.extend(f"source_{len(candidates)}:{err}" for err in run_errors)
    by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_token[item["topology_key_token"]].append(item)
    all_source_tokens = set(by_token)
    expected_set = set(expected_tokens)
    missing_tokens = sorted(expected_set - all_source_tokens)
    extra_tokens = sorted(all_source_tokens - expected_set)
    duplicate_tokens = sorted(token for token, items in by_token.items() if len(items) > 1)
    result_rows: list[dict[str, Any]] = []
    changed_input_tokens: set[str] = set()
    for token in sorted(expected_set):
        found = by_token.get(token, [])
        if not found:
            result_rows.append({
                "loop": loop_value, "topology_key": expected_keys[token],
                "topology_key_token": token, "status": "unknown", "source_type": source_type,
                "certificate_refs": [], "reject_basis": None,
                "errors": ["expected_topology_not_observed"], "input_files": [],
            })
            continue
        if len(found) > 1:
            refs = [ref for item in found for ref in item.get("certificate_refs", [])]
            result_rows.append({
                "loop": loop_value, "topology_key": expected_keys[token],
                "topology_key_token": token, "status": "unknown", "source_type": source_type,
                "certificate_refs": refs, "reject_basis": None,
                "errors": ["duplicate_complete_topology_key_across_inputs"],
                "input_files": sorted({path for item in found for path in item.get("input_files", [])}),
            })
            continue
        item = dict(found[0])
        item["topology_key"] = expected_keys[token]
        if errors:
            # Source-local failures are retained as unknown; other disjoint,
            # independently bound shards remain usable.
            pass
        result_rows.append(item)
    duplicate_records = [{
        "topology_key": _key_for_token(token),
        "topology_key_token": token,
        "references": [{"source_type": item.get("source_type"), "input_files": item.get("input_files", [])} for item in by_token[token]],
    } for token in duplicate_tokens]
    extra_records = [{"topology_key": _key_for_token(token), "topology_key_token": token} for token in extra_tokens]
    missing_records = [{"topology_key": expected_keys[token], "topology_key_token": token} for token in missing_tokens]
    # Every observed key is counted, including extras, while only expected keys
    # are ever written to the evidence index.
    rows_by_status = {status: sum(row["status"] == status for row in result_rows) for status in _ALLOWED}
    expected_integrity_errors = _validate_expected_row(expected_row) if expected_row is not None else ["expected_loop_missing"]
    errors.extend(expected_integrity_errors)
    if extra_records:
        errors.append("extra_topology_keys_present")
    if duplicate_records:
        errors.append("duplicate_complete_topology_keys_present")
    if missing_records:
        errors.append("expected_topology_keys_missing")
    errors = sorted(set(errors))
    loop_status = "pass" if not errors and rows_by_status["unknown"] == 0 and rows_by_status["pass"] + rows_by_status["reject"] == len(expected_tokens) else "unknown"
    report = {
        "loop": loop_value,
        "source_type": source_type,
        "status": loop_status,
        "expected_count": len(expected_tokens),
        "expected_unique": len(expected_set),
        "observed_unique": len(all_source_tokens),
        "observed_expected_unique": len(expected_set & all_source_tokens),
        "key_set_sha256": _key_set_sha256(expected_tokens),
        "key_set_digest_algorithm": _KEY_SET_ALGORITHM,
        "counts": rows_by_status,
        "coverage": {
            "complete": not missing_records and not extra_records and not duplicate_records and rows_by_status["unknown"] == 0,
            "missing": missing_records,
            "duplicates": duplicate_records,
            "extra": extra_records,
        },
        "expected_source_bindings": expected_row.get("source_files", []) if expected_row is not None else [],
        "errors": errors,
    }
    return result_rows, report


def _prepare_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    else:
        path.mkdir(parents=True)


def _write_gzip_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(payload)


def _write_outputs(output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    _prepare_output(output)
    index = b"".join(_json_bytes(row) + b"\n" for row in rows)
    _write_gzip_exclusive(output / "topology_index.jsonl.gz", index)
    _write_gzip_exclusive(output / "summary.json.gz", _json_bytes(summary) + b"\n")


def summarize_plan(plan: Mapping[str, Any], *, plan_path: Path | None = None, tracker: InputTracker | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tracker = tracker or InputTracker()
    base = plan_path.resolve().parent if plan_path is not None else Path.cwd()
    errors: list[str] = []
    expected_rows, global_input_paths, expected_errors = _expected_rows(plan, base, tracker)
    errors.extend(expected_errors)
    required_theorem_sha = plan.get("required_theorem_sha256", plan.get("theorem_sha256"))
    if not _is_sha256(required_theorem_sha):
        required_theorem_sha = None
        errors.append("required_theorem_sha256_missing_or_invalid")
    loop_configs = plan.get("loops")
    if not isinstance(loop_configs, list):
        loop_configs = []
        errors.append("plan_loops_array_missing")
    expected_by_loop: dict[int, dict[str, Any]] = {}
    for row in expected_rows:
        loop = row.get("loop")
        if isinstance(loop, int) and not isinstance(loop, bool):
            if loop in expected_by_loop:
                errors.append(f"duplicate_expected_loop_{loop}")
            expected_by_loop[loop] = row
    plan_loop_counts: dict[int, int] = defaultdict(int)
    for row in loop_configs:
        if isinstance(row, Mapping) and isinstance(row.get("loop"), int) and not isinstance(row.get("loop"), bool):
            plan_loop_counts[row["loop"]] += 1
    duplicate_plan_loops = {loop for loop, count in plan_loop_counts.items() if count > 1}
    errors.extend(f"duplicate_plan_loop_{loop}" for loop in sorted(duplicate_plan_loops))
    index_rows: list[dict[str, Any]] = []
    loop_reports: list[dict[str, Any]] = []
    emitted_duplicate_loops: set[int] = set()
    for loop_config in loop_configs:
        if not isinstance(loop_config, Mapping):
            errors.append("plan_loop_entry_not_object")
            continue
        loop = loop_config.get("loop")
        if isinstance(loop, int) and not isinstance(loop, bool) and loop in duplicate_plan_loops:
            if loop in emitted_duplicate_loops:
                continue
            emitted_duplicate_loops.add(loop)
            expected_row = expected_by_loop.get(loop)
            keys = expected_row.get("topology_keys", expected_row.get("expected_keys", [])) if expected_row else []
            if isinstance(keys, list):
                for key_value in keys:
                    try:
                        key, token = _key_and_token(key_value)
                    except Exception:
                        continue
                    index_rows.append({
                        "loop": loop, "topology_key": key, "topology_key_token": token,
                        "status": "unknown", "source_type": "unknown", "certificate_refs": [],
                        "reject_basis": None, "errors": [f"duplicate_plan_loop_{loop}"], "input_files": [],
                    })
            loop_reports.append({
                "loop": loop, "source_type": "unknown", "status": "unknown",
                "expected_count": len(keys) if isinstance(keys, list) else 0,
                "expected_unique": len({_key_token(item) for item in keys if isinstance(item, (list, tuple))}) if isinstance(keys, list) else 0,
                "observed_unique": 0, "observed_expected_unique": 0,
                "counts": {"pass": 0, "reject": 0, "unknown": len(keys) if isinstance(keys, list) else 0},
                "coverage": {"complete": False, "missing": [], "duplicates": [], "extra": []},
                "errors": [f"duplicate_plan_loop_{loop}"],
            })
            continue
        expected_row = expected_by_loop.get(loop) if isinstance(loop, int) else None
        output_rows, report = _build_loop(loop_config, expected_row, base, tracker, required_theorem_sha)
        index_rows.extend(output_rows)
        loop_reports.append(report)
    # Require one source plan per expected loop and produce explicit unknown
    # rows if a loop is absent from the plan.
    plan_loop_numbers = {row.get("loop") for row in loop_configs if isinstance(row, Mapping)}
    for loop, expected_row in expected_by_loop.items():
        if loop in plan_loop_numbers:
            continue
        missing_records: list[dict[str, Any]] = []
        try:
            keys = expected_row.get("topology_keys", expected_row.get("expected_keys", []))
            for key_value in keys if isinstance(keys, list) else []:
                key, token = _key_and_token(key_value)
                index_rows.append({
                    "loop": loop, "topology_key": key, "topology_key_token": token,
                    "status": "unknown", "source_type": "unknown", "certificate_refs": [],
                    "reject_basis": None, "errors": ["loop_missing_from_plan"], "input_files": [],
                })
        except Exception:
            pass
        loop_reports.append({
            "loop": loop, "source_type": "unknown", "status": "unknown",
            "expected_count": expected_row.get("expected_count"), "expected_unique": expected_row.get("observed_unique"),
            "observed_unique": 0, "observed_expected_unique": 0,
            "coverage": {"complete": False, "missing": missing_records, "duplicates": [], "extra": []},
            "errors": ["loop_missing_from_plan"],
        })
    changed_inputs = tracker.finalize()
    if changed_inputs:
        if plan_path is not None and str(plan_path.resolve()) in changed_inputs:
            errors.append("plan_changed_during_summary")
        changed_expected = global_input_paths & changed_inputs
        if changed_expected:
            errors.append("expected_manifest_changed_during_summary")
        for row in index_rows:
            if changed_expected or any(str(Path(path).resolve()) in changed_inputs for path in row.get("input_files", [])):
                row["status"] = "unknown"
                row["errors"] = sorted(set(row.get("errors", []) + ["input_changed_during_summary"]))
                row["reject_basis"] = None
        for report in loop_reports:
            if changed_expected:
                report["errors"] = sorted(set(report.get("errors", []) + ["expected_manifest_changed_during_summary"]))
                report["status"] = "unknown"
            report["counts"] = {status: sum(row["loop"] == report.get("loop") and row["status"] == status for row in index_rows) for status in _ALLOWED}
            report["coverage"]["complete"] = report["coverage"].get("complete", False) and report["counts"]["unknown"] == 0
            if report["counts"]["unknown"] or not report["coverage"]["complete"]:
                report["status"] = "unknown"
    # Global plan, theorem, or expected-manifest errors invalidate all source
    # conclusions, while preserving their references and differential report.
    if errors:
        for row in index_rows:
            row["status"] = "unknown"
            row["reject_basis"] = None
            row["errors"] = sorted(set(row.get("errors", []) + ["global_plan_binding_invalid"]))
        for report in loop_reports:
            report["errors"] = sorted(set(report.get("errors", []) + errors))
            report["status"] = "unknown"
            report["counts"] = {status: sum(row["loop"] == report.get("loop") and row["status"] == status for row in index_rows) for status in _ALLOWED}
            coverage = report.setdefault("coverage", {"complete": False, "missing": [], "duplicates": [], "extra": []})
            coverage["complete"] = False
    overall_counts = {status: sum(row["status"] == status for row in index_rows) for status in _ALLOWED}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit_level": AUDIT_LEVEL,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass" if not errors and loop_reports and all(report.get("status") == "pass" for report in loop_reports) else "unknown",
        "required_theorem_sha256": required_theorem_sha,
        "key_set_digest_algorithm": _KEY_SET_ALGORITHM,
        "counts": overall_counts,
        "loops": loop_reports,
        "inputs": sorted(tracker.files.values(), key=lambda item: item["path"]),
        "errors": sorted(set(errors)),
        "limitations": ["No mathematical replay was performed; original runners are the replay boundary."],
    }
    return index_rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="explicit JSON plan containing expected-P and result paths")
    parser.add_argument("--output", required=True, type=Path, help="new or empty output directory; existing files are never overwritten")
    args = parser.parse_args(argv)
    tracker = InputTracker()
    plan_path = args.plan.expanduser().resolve()
    try:
        plan = tracker.read_json(plan_path, kind="plan")
        if not isinstance(plan, Mapping):
            raise ValueError("plan root must be an object")
        rows, summary = summarize_plan(plan, plan_path=plan_path, tracker=tracker)
    except Exception as exc:
        tracker.finalize()
        rows = []
        summary = {
            "schema_version": SCHEMA_VERSION,
            "audit_level": AUDIT_LEVEL,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "unknown",
            "required_theorem_sha256": None,
            "key_set_digest_algorithm": _KEY_SET_ALGORITHM,
            "counts": {"pass": 0, "reject": 0, "unknown": 0},
            "loops": [],
            "inputs": sorted(tracker.files.values(), key=lambda item: item["path"]),
            "errors": [f"plan_read_failed:{type(exc).__name__}:{exc}"],
            "limitations": ["No mathematical replay was performed; original runners are the replay boundary."],
        }
    _write_outputs(args.output.expanduser().resolve(), rows, summary)
    return 0 if summary.get("status") == "pass" else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
