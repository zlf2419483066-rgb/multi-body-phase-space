"""Focused tests for the opt-in parallel checkpoint coordinator."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pathlib
import runpy
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.parallel_reporting import (  # noqa: E402
    _source_hashes,
    _verify_shard_index,
    run_parallel_candidates,
)


class ParallelReportingTests(unittest.TestCase):
    def test_l0_four_shards_have_exact_durable_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_parallel_candidates(
                loop_count=0, max_seconds=30.0, output_dir=Path(directory) / "l0",
                workers=4,
            )
            self.assertEqual(report.status, "complete")
            manifest = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["topology_enumeration"]["unique_count"], 3)
            self.assertTrue(manifest["union_coverage"]["coverage_exact"])
            self.assertTrue(manifest["union_coverage"]["shards_pairwise_disjoint"])
            self.assertEqual(manifest["union_coverage"]["duplicate_count"], 0)
            self.assertEqual(manifest["union_coverage"]["missing_count"], 0)
            self.assertTrue(manifest["worker_source_hashes_match"])
            self.assertGreater(len(manifest["source_hashes"]), 6)
            self.assertTrue(all("\\" not in name for name in manifest["source_hashes"]))
            self.assertTrue(manifest["status_checks"]["index_coverage_complete"])
            self.assertEqual(manifest["actual_counts"]["identity_counts"],
                             {"fixed_labels": 105, "particle_relabeling": 3})
            self.assertEqual(manifest["actual_counts"]["representative_evaluations"], 360)
            self.assertEqual(manifest["actual_counts"]["certificate_transports"], 0)
            self.assertFalse(manifest["actual_counts"]["particle_orbit_reuse"])
            self.assertTrue(all(item["index_coverage"]["worker_identity_counts_match"]
                                for item in manifest["shards"]))
            self.assertEqual(manifest["strict_readback"]["status"], "not_run")
            self.assertEqual([item["expected_count"] for item in manifest["shards"]], [1, 1, 1, 0])

    def test_flow_and_explicit_orbit_options_are_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            orbit = run_parallel_candidates(
                loop_count=0, max_seconds=30.0, output_dir=root / "orbit",
                workers=4, label_strategy="flow", use_particle_orbit=True,
            )
            flow = run_parallel_candidates(
                loop_count=0, max_seconds=30.0, output_dir=root / "flow",
                workers=4, label_strategy="flow",
            )
            self.assertTrue(orbit.complete)
            self.assertTrue(flow.complete)
            self.assertTrue(json.loads(Path(orbit.manifest_path).read_text(encoding="utf-8"))["use_particle_orbit"])
            self.assertEqual(json.loads(Path(flow.manifest_path).read_text(encoding="utf-8"))["label_strategy"], "flow")
            orbit_manifest = json.loads(Path(orbit.manifest_path).read_text(encoding="utf-8"))
            self.assertIn("representative_evaluations", orbit_manifest["actual_counts"])
            self.assertIn("certificate_transports", orbit_manifest["actual_counts"])
            self.assertTrue(orbit_manifest["actual_counts"]["particle_orbit_reuse"])
            with self.assertRaises(ValueError):
                run_parallel_candidates(loop_count=0, max_seconds=1.0,
                                        output_dir=root / "invalid-orbit",
                                        workers=4, use_particle_orbit=True)

    def test_zero_budget_is_incomplete_and_preserves_scope_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_parallel_candidates(
                loop_count=0, max_seconds=0.0, output_dir=Path(directory) / "zero",
                workers=4,
            )
            self.assertEqual(report.status, "incomplete")
            self.assertFalse(report.complete)
            payload = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))
            self.assertFalse(payload["topology_enumeration"]["complete"])
            self.assertEqual(payload["strict_readback"]["status"], "not_run")

    def test_index_verifier_rejects_source_mismatch_and_duplicate_or_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_parallel_candidates(
                loop_count=0, max_seconds=30.0, output_dir=Path(directory) / "run",
                workers=4,
            )
            payload = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))
            shard = Path(payload["shards"][0]["checkpoint_root"]).parent
            expected = payload["shards"][0]["expected_keys"]
            mismatched = _verify_shard_index(shard, expected, {"labeling.py": "wrong"})
            self.assertEqual(mismatched["status"], "incomplete")
            self.assertIn("scope_source_hash_mismatch", mismatched["errors"])
            index = shard / "catalog" / "index.jsonl"
            original = index.read_bytes()
            index.write_bytes(original + original)
            duplicate = _verify_shard_index(shard, expected, _source_hashes())
            self.assertEqual(duplicate["status"], "incomplete")
            self.assertIn("duplicate_committed_topology", duplicate["errors"])
            first = json.loads(original.splitlines()[0].decode("utf-8"))
            first["label_enumeration_complete"] = "bad"
            index.write_text(json.dumps(first) + "\n", encoding="utf-8")
            malformed = _verify_shard_index(shard, expected, _source_hashes())
            self.assertEqual(malformed["status"], "incomplete")
            self.assertIn("label_enumeration_complete_invalid", malformed["errors"])
            index.write_bytes(original)
            (shard / "catalog" / "data.jsonl.gz").unlink()
            missing_data = _verify_shard_index(shard, expected, _source_hashes())
            self.assertEqual(missing_data["status"], "invalid")

    def test_malformed_worker_aggregate_is_reported_without_type_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "malformed-aggregate"
            with patch("phase_space_graph5.parallel_reporting._read_status",
                       return_value={"aggregate": [], "source_hashes": {}}):
                report = run_parallel_candidates(loop_count=0, max_seconds=30.0,
                                                 output_dir=output, workers=4)
            self.assertEqual(report.status, "incomplete")
            manifest = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["actual_counts_error"], "worker_aggregate_invalid")

    def test_cli_outer_supervisor_reports_short_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cli"
            command = [sys.executable,
                       str(Path(__file__).resolve().parents[1] / "examples" / "run_parallel_candidates.py"),
                       "--loop-count", "0", "--max-seconds", "0", "--output", str(output)]
            completed = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[1]),
                                       check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0)
            status = json.loads((output / "worker_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["process_status"], "completed")
            self.assertEqual(status["run_status"], "incomplete")

    def test_cli_status_write_raises_after_permission_retry_exhaustion(self) -> None:
        cli_path = Path(__file__).resolve().parents[1] / "examples" / "run_parallel_candidates.py"
        namespace = runpy.run_path(str(cli_path), run_name="parallel_cli_test")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(pathlib.Path, "replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    namespace["_write_status"](
                        Path(directory), status="failed", process_status="failed",
                        run_status="incomplete", stop_reason="worker_failed",
                        requested_seconds=1.0, worker_seconds=0.5,
                        started=0.0, worker_pid=None, worker_exit_code=1,
                    )

    def test_cli_termination_returns_process_tree_evidence(self) -> None:
        cli_path = Path(__file__).resolve().parents[1] / "examples" / "run_parallel_candidates.py"
        namespace = runpy.run_path(str(cli_path), run_name="parallel_cli_test")
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            evidence = namespace["_terminate_tree"](child)
            self.assertIn("taskkill_returncode", evidence)
            self.assertIn("before", evidence)
            self.assertIn("after", evidence)
            self.assertFalse(evidence["after"]["pid_alive"])
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_dead_root_does_not_hide_live_captured_descendant(self) -> None:
        cli_path = Path(__file__).resolve().parents[1] / "examples" / "run_parallel_candidates.py"
        namespace = runpy.run_path(str(cli_path), run_name="parallel_cli_test")
        root = subprocess.Popen([
            sys.executable, "-c",
            "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)",
        ])
        descendant_pids = []
        try:
            before = None
            for _ in range(30):
                before = namespace["_process_snapshot"](root.pid)
                descendant_pids = list(before.get("descendant_pids", ()))
                if descendant_pids or before.get("psutil_available") is False:
                    break
                time.sleep(0.1)
            if before is None or before.get("psutil_available") is False:
                self.skipTest("psutil is unavailable for process-tree evidence")
            if not descendant_pids:
                self.skipTest("test child was not observable")
            root.kill()
            root.wait()
            after = namespace["_process_snapshot"](
                root.pid,
                known_descendants=before["descendant_records"],
                known_create_time=before.get("create_time"),
            )
            self.assertFalse(after["pid_alive"])
            self.assertTrue(any(after["descendant_alive"]))
        finally:
            if root.poll() is None:
                root.kill()
                root.wait()
            try:
                import psutil
                for pid in descendant_pids:
                    try:
                        psutil.Process(pid).kill()
                    except psutil.Error:
                        pass
            except ImportError:
                pass

    def test_snapshot_access_denied_is_unknown_and_cleanup_still_runs(self) -> None:
        cli_path = Path(__file__).resolve().parents[1] / "examples" / "run_parallel_candidates.py"
        namespace = runpy.run_path(str(cli_path), run_name="parallel_cli_test")
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil is unavailable for process-tree evidence")
        with patch.object(psutil.Process, "children", side_effect=psutil.AccessDenied()):
            snapshot = namespace["_process_snapshot"](os.getpid())
        self.assertIsNone(snapshot["pid_alive"])
        self.assertFalse(snapshot["snapshot_complete"])

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            with patch.object(psutil.Process, "children", side_effect=psutil.AccessDenied()):
                evidence = namespace["_terminate_tree"](child)
            self.assertIsNotNone(child.poll())
            self.assertFalse(evidence["all_stopped"])
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_final_manifest_save_failure_does_not_return_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "save-failure"
            with patch("phase_space_graph5.parallel_reporting._write_json_atomic",
                       side_effect=OSError("synthetic final save failure")):
                with self.assertRaises(OSError):
                    run_parallel_candidates(loop_count=0, max_seconds=30.0,
                                            output_dir=output, workers=4)
            self.assertTrue(any(output.iterdir()))


if __name__ == "__main__":
    unittest.main()
