from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.label_catalog import load_catalog
from phase_space_graph5 import reporting
from phase_space_graph5.drawing import LayoutError
from phase_space_graph5.catalog_checkpoint import read_checkpoint
from phase_space_graph5.physical_labels import CandidateMomentumRecord
from phase_space_graph5.reporting import run_candidates


class CandidateReportingTests(unittest.TestCase):
    def test_bounded_run_keeps_incomplete_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_candidates(max_seconds=0.0, output_dir=Path(directory) / "run", render=False)
            self.assertEqual(report.status, "incomplete")
            self.assertFalse(report.complete)
            self.assertEqual(report.stop_reason, "deadline_exceeded")

    def test_zero_loop_small_scope_writes_both_identity_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_candidates(max_seconds=20.0, output_dir=Path(directory) / "run", render=False)
            self.assertEqual(report.status, "complete")
            self.assertEqual(set(report.catalogs), {"fixed_labels", "particle_relabeling"})
            self.assertEqual(len(report.catalogs["fixed_labels"].entries), 105)
            self.assertEqual(len(report.catalogs["particle_relabeling"].entries), 3)
            loaded = load_catalog(Path(report.output_directory) / "catalog" / "fixed_labels")
            self.assertTrue(loaded.complete)

    def test_topology_audit_counts_match_each_identity_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_candidates(max_seconds=20.0, output_dir=Path(directory) / "run", render=False)
            for mode, expected in (("fixed_labels", 105), ("particle_relabeling", 3)):
                audits = report.catalogs[mode].manifest.extra["topology_audit"]
                self.assertTrue(audits)
                self.assertTrue(all(item["mode"] == mode for item in audits))
                self.assertEqual(sum(item["unique_identity_count"] for item in audits), expected)
                self.assertEqual(sum(item["stored_identity_count"] for item in audits), expected)

    def test_drawing_failure_cannot_leave_run_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(reporting, "layout_graph", side_effect=LayoutError("layout_error: synthetic")):
                report = run_candidates(max_seconds=20.0, output_dir=Path(directory) / "run",
                                        render=True, max_drawings=1)
            self.assertEqual(report.status, "incomplete")
            self.assertFalse(report.complete)
            self.assertTrue(report.drawing_outcomes)
            self.assertEqual(report.drawing_outcomes[0].status, "failed")

    def test_budget_rejects_values_that_break_global_deadline(self):
        with self.assertRaises(ValueError):
            run_candidates(max_seconds=1201.0, render=False)
        with self.assertRaises(ValueError):
            run_candidates(max_seconds=float("nan"), render=False)
        with self.assertRaises(ValueError):
            run_candidates(max_seconds=1.0, max_drawings=-1, render=False)

    def test_search_only_benchmark_persists_stage_timing_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_candidates(max_seconds=20.0, output_dir=Path(directory) / "run", render=False)
            self.assertEqual(report.benchmark, "search_only")
            self.assertIn("topology_enumeration", report.stage_timings)
            self.assertIn("labeling_enumeration", report.stage_timings)
            self.assertTrue(all(value >= 0.0 for value in report.stage_timings.values()))
            progress = report.progress
            self.assertEqual(progress["status"], "complete")
            self.assertFalse(progress["render"])
            self.assertFalse(progress["resume"]["supported"])
            run = self._read_json(Path(report.output_directory) / "run_manifest.json")
            self.assertEqual(run["benchmark"], "search_only")
            self.assertEqual(run["progress"]["status"], "complete")

    def test_identity_failure_cannot_claim_checkpoint_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            original = reporting.canonicalize_candidate_labeling

            def fail_identity(record, mode):
                if isinstance(record, CandidateMomentumRecord):
                    raise ValueError("synthetic identity failure")
                return original(record, mode)

            with patch.object(reporting, "canonicalize_candidate_labeling", side_effect=fail_identity):
                report = run_candidates(max_seconds=20.0, output_dir=Path(directory) / "run", render=False)
            self.assertEqual(report.status, "incomplete")
            self.assertFalse(report.complete)
            checkpoint = read_checkpoint(Path(report.checkpoint_directory))
            self.assertFalse(checkpoint.complete)
            self.assertTrue(checkpoint.records)
            self.assertTrue(all(not record.complete for record in checkpoint.records))
            self.assertTrue(all(not any(record.identity_save_complete.values())
                                for record in checkpoint.records))

    def test_atomic_progress_replace_retries_short_sharing_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run_progress.json"
            target.write_text('{"status":"old"}', encoding="utf-8")
            original_replace = Path.replace
            attempts = []

            def flaky_replace(source, destination):
                attempts.append(1)
                if len(attempts) < 3:
                    raise PermissionError("synthetic sharing lock")
                return original_replace(source, destination)

            with patch.object(Path, "replace", flaky_replace), \
                    patch.object(reporting.time, "sleep", return_value=None):
                reporting._write_json_atomic(target, {"status": "new"})
            self.assertEqual(len(attempts), 3)
            self.assertIn('"status": "new"', target.read_text(encoding="utf-8"))
            self.assertEqual(tuple(target.parent.glob(".run_progress.json.*.tmp")), ())

    def test_atomic_progress_persistent_lock_preserves_old_file_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run_progress.json"
            old = '{"status":"old"}'
            target.write_text(old, encoding="utf-8")
            attempts = []

            def locked_replace(source, destination):
                attempts.append(1)
                raise PermissionError("synthetic persistent lock")

            started = time.monotonic()
            with patch.object(Path, "replace", locked_replace), \
                    patch.object(reporting.time, "sleep", return_value=None):
                with self.assertRaises(PermissionError):
                    reporting._write_json_atomic(target, {"status": "new"})
            self.assertEqual(len(attempts), 4)
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertEqual(target.read_text(encoding="utf-8"), old)
            self.assertEqual(tuple(target.parent.glob(".run_progress.json.*.tmp")), ())

    def test_persistent_progress_failure_stops_run_as_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            def locked_replace(source, destination):
                raise PermissionError("synthetic persistent lock")

            with patch.object(Path, "replace", locked_replace), \
                    patch.object(reporting.time, "sleep", return_value=None):
                report = run_candidates(max_seconds=0.0, output_dir=Path(directory) / "run", render=False)
            self.assertEqual(report.status, "incomplete")
            self.assertFalse(report.complete)
            self.assertTrue(any(item["stage"] == "progress_persistence" for item in report.failures))
            self.assertTrue(report.progress["progress_persistence_errors"])

    def test_late_finished_progress_failure_downgrades_manifest_once(self):
        with tempfile.TemporaryDirectory() as directory:
            original = reporting._write_json_atomic
            stages = []

            def fail_only_finished(path, payload, *, deadline=None):
                stages.append(payload["stage"])
                if payload["stage"] == "finished":
                    raise PermissionError("synthetic late sharing lock")
                return original(path, payload, deadline=deadline)

            with patch.object(reporting, "_write_json_atomic", side_effect=fail_only_finished):
                report = run_candidates(max_seconds=20.0, output_dir=Path(directory) / "run", render=False)
            self.assertEqual(stages.count("finished"), 1)
            self.assertEqual(report.status, "incomplete")
            self.assertFalse(report.complete)
            import json
            manifest = json.loads((Path(report.output_directory) / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "incomplete")
            self.assertTrue(all(value["run_status"] == "incomplete"
                                for value in manifest["catalogs"].values()))

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        import json
        return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
