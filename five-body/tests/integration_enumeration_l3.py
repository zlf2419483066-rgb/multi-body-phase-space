"""One-shot public L3 enumeration integration check.

This file is intentionally not named ``test_*.py``: ordinary unit tests use a
bounded L3 stop, while this script performs the one complete public L3 run and
compares the full canonical key sequence with the frozen structure probe.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time


TEST_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = TEST_ROOT
SRC_ROOT = PACKAGE_ROOT / "src"
EXPECTED_KEYS = PACKAGE_ROOT / "outputs" / "overnight_20260919" / "structure" / "l3_structure_run_20260920_keys" / "pruned_canonical_keys.json"
RESULT_PATH = PACKAGE_ROOT / "outputs" / "overnight_20260919" / "structure" / "l3_public_integration_20260920.json"
sys.path.insert(0, str(SRC_ROOT))

from phase_space_graph5 import EnumerationLimits, RunStatus, enumerate_topologies  # noqa: E402


MAX_SECONDS = 1100.0


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value


def _key_digest(keys) -> str:
    payload = json.dumps(keys, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    expected_payload = json.loads(EXPECTED_KEYS.read_text(encoding="utf-8"))
    expected_keys = tuple(_freeze(key) for key in expected_payload["keys"])
    started = time.perf_counter()
    report = enumerate_topologies(5, 3, EnumerationLimits(max_seconds=MAX_SECONDS))
    elapsed = time.perf_counter() - started
    actual_keys = tuple(report.canonical_keys)
    result = {
        "schema_version": "F5-public-L3-integration-v1",
        "status": getattr(report.status, "value", report.status),
        "elapsed_s": round(elapsed, 6),
        "budget_seconds": MAX_SECONDS,
        "expected_key_file": str(EXPECTED_KEYS),
        "expected_key_count": len(expected_keys),
        "actual_key_count": len(actual_keys),
        "expected_key_sha256": expected_payload["sha256"],
        "actual_key_sha256": _key_digest(actual_keys),
        "full_key_sequence_equal": actual_keys == expected_keys,
        "counts": report.to_dict()["counts"],
        "stop_reason": report.stop_reason,
        "scope_boundary": "public L3 enabled; public L4/L5 remain rejected by enumerate_topologies",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if (
        report.status is RunStatus.COMPLETE
        and actual_keys == expected_keys
        and elapsed <= MAX_SECONDS
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
