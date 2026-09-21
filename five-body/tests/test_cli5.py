from __future__ import annotations

import json
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.label_catalog import load_catalog


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "run_candidates.py"
sys.path.insert(0, str(SCRIPT.parent))
import run_candidates as cli  # noqa: E402


class CandidateCliTests(unittest.TestCase):
    def _run(self, output: Path, *arguments: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments, "--output", str(output)],
            cwd=str(SCRIPT.parents[1]), text=True, capture_output=True, timeout=timeout,
        )

    @staticmethod
    def _json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_zero_budget_reports_actual_incomplete_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "zero"
            result = self._run(output, "--loop-count", "0", "--max-seconds", "0", "--no-render")
            self.assertEqual(result.returncode, 0, result.stderr)
            worker = self._json(output / "worker_status.json")
            self.assertEqual(worker["process_status"], "completed")
            self.assertEqual(worker["run_status"], "incomplete")
            self.assertEqual(worker["run_stop_reason"], "deadline_exceeded")
            run = self._json(output / "run_manifest.json")
            self.assertEqual(run["status"], "incomplete")

    def test_fixture_a_exports_fixed_and_particle_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture-a"
            result = self._run(output, "--fixture", "A", "--max-seconds", "20", "--no-render")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = self._json(output / "fixture_manifest.json")
            self.assertEqual(manifest["status"], "complete")
            fixed = load_catalog(output / "catalog" / "fixed_labels")
            particle = load_catalog(output / "catalog" / "particle_relabeling")
            self.assertTrue(fixed.complete)
            self.assertTrue(particle.complete)
            self.assertEqual(len(fixed.entries), 60)
            self.assertEqual(len(particle.entries), 1)

    def test_fixture_b2_mode_and_selected_scope_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture-b2"
            result = self._run(output, "--fixture", "B2", "--mode", "particle_relabeling",
                               "--max-seconds", "20", "--no-render")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = self._json(output / "fixture_manifest.json")
            selected = manifest["catalog"]
            self.assertEqual(selected["count_mode"], "particle_relabeling")
            self.assertEqual(selected["scope"]["selection"], "single_selected_label")
            self.assertFalse(selected["coverage"]["label_enumeration_complete"])
            audit = selected["topology_audit"][0]
            self.assertEqual(audit["raw_assignment_count"], 1)
            self.assertEqual(audit["momentum_pass_count"], 1)
            self.assertEqual(audit["unique_identity_count"], 1)
            catalog = load_catalog(output / "catalog" / "particle_relabeling")
            self.assertFalse(catalog.complete)
            self.assertEqual(len(catalog.entries), 1)

    def test_compact_cli_writes_explicit_schema_and_checkpoint_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "compact"
            result = self._run(output, "--compact", "--loop-count", "0",
                               "--label-strategy", "flow", "--max-seconds", "20", "--no-render")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = self._json(output / "run_report.json")
            self.assertEqual(report["schema_version"], "F5-compact-run-report-v1")
            self.assertTrue(report["complete"])
            self.assertEqual(report["label_strategy"], "flow")
            self.assertFalse((output / "catalog" / "fixed_labels").exists())
            worker = self._json(output / "worker_status.json")
            self.assertIn("catalog/checkpoints/scope.json", worker["checkpoint_paths"])
            self.assertIn("catalog/checkpoints/index.jsonl", worker["checkpoint_paths"])
            self.assertIn("catalog/checkpoints/data.jsonl.gz", worker["checkpoint_paths"])

    def test_twenty_minute_budget_is_accepted_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "twenty-minute"
            result = self._run(output, "--fixture", "B2", "--max-seconds", "1200", "--no-render")
            self.assertEqual(result.returncode, 0, result.stderr)
            worker = self._json(output / "worker_status.json")
            self.assertEqual(worker["hard_budget_seconds"], 1200.0)
            self.assertEqual(worker["run_status"], "complete")

    def test_hard_timeout_persists_incomplete_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timeout-progress"
            result = self._run(output, "--loop-count", "0", "--max-seconds", "0.01", "--no-render")
            self.assertEqual(result.returncode, 124, result.stderr)
            progress_path = output / "run_progress.json"
            if not progress_path.is_file():
                children = [item for item in output.iterdir() if item.is_dir()]
                progress_path = children[0] / "run_progress.json"
            progress = self._json(progress_path)
            self.assertEqual(progress["status"], "incomplete")
            self.assertEqual(progress["stop_reason"], "deadline_exceeded")

    def test_hard_timeout_overrides_stale_complete_sidecar(self) -> None:
        class TimedOutProcess:
            pid = 12345
            returncode = -9

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("synthetic-worker", timeout)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hard-timeout-stale"
            args = argparse.Namespace(
                max_seconds=1.0, output=output, mode="fixed_labels", loop_count=0,
                fixture=None, dot=None, max_drawings=None, no_render=True,
                compact=True, label_strategy="default",
            )
            captured = {}
            with patch.object(cli.subprocess, "Popen", return_value=TimedOutProcess()), \
                 patch.object(cli, "_terminate_worker"), \
                 patch.object(cli, "_mark_timeout_progress"), \
                 patch.object(cli, "_actual_run_status", return_value=("complete", None)), \
                 patch.object(cli, "_write_worker_status",
                              side_effect=lambda *a, **kw: captured.update(kw)):
                result = cli._run_outer_worker(args)
            self.assertEqual(result, cli._HARD_TIMEOUT_EXIT)
            self.assertEqual(captured["run_status"], "incomplete")
            self.assertEqual(captured["run_stop_reason"], "deadline_exceeded")

    def test_nonzero_worker_cannot_preserve_complete_sidecar(self) -> None:
        class FailedProcess:
            pid = 12346
            returncode = 1

            def wait(self, timeout=None):
                return None

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "worker-failure-stale"
            args = argparse.Namespace(
                max_seconds=1.0, output=output, mode="fixed_labels", loop_count=0,
                fixture=None, dot=None, max_drawings=None, no_render=True,
                compact=True, label_strategy="default",
            )
            captured = {}
            with patch.object(cli.subprocess, "Popen", return_value=FailedProcess()), \
                 patch.object(cli, "_actual_run_status", return_value=("complete", None)), \
                 patch.object(cli, "_write_worker_status",
                              side_effect=lambda *a, **kw: captured.update(kw)):
                result = cli._run_outer_worker(args)
            self.assertEqual(result, 1)
            self.assertEqual(captured["run_status"], "incomplete")
            self.assertEqual(captured["run_stop_reason"], "worker_failed")

    def test_compact_persistence_failure_returns_nonzero_worker_exit(self) -> None:
        args = argparse.Namespace(
            max_seconds=1.0, output=Path("unused"), mode="fixed_labels", loop_count=0,
            fixture=None, dot=None, max_drawings=None, no_render=True,
            compact=True, label_strategy="default",
        )

        class FailedReport:
            status = "incomplete"
            stop_reason = "compact_report_persistence_failed"

        with patch.object(cli, "run_compact_candidates", return_value=FailedReport()):
            self.assertEqual(cli._run_worker(args), 1)

    def test_hard_timeout_is_reported_and_old_output_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            output.mkdir()
            sentinel = output / "old-checkpoint.txt"
            sentinel.write_text("keep", encoding="utf-8")
            run = self._run(output, "--loop-count", "0", "--max-seconds", "0.01", "--no-render")
            self.assertEqual(run.returncode, 124, run.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            children = [item for item in output.iterdir() if item.is_dir()]
            self.assertEqual(len(children), 1)
            worker = self._json(children[0] / "worker_status.json")
            self.assertEqual(worker["process_status"], "hard_timeout")
            self.assertEqual(worker["stop_reason"], "deadline_exceeded")
            self.assertTrue(Path(worker["run_directory"]).is_absolute())

    def test_drawing_failure_is_reported_in_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "draw-failure"
            result = self._run(output, "--fixture", "B2", "--max-seconds", "20",
                               "--dot", str(output / "missing-dot"))
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = self._json(output / "fixture_manifest.json")
            self.assertEqual(manifest["status"], "incomplete")
            self.assertTrue(manifest["drawings"])
            self.assertIn(manifest["drawings"][0]["status"], {"failed", "incomplete"})
            worker = self._json(output / "worker_status.json")
            self.assertEqual(worker["process_status"], "completed")
            self.assertEqual(worker["run_status"], "incomplete")

    def test_nonempty_output_uses_a_fresh_absolute_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            output.mkdir()
            (output / "old.txt").write_text("keep", encoding="utf-8")
            result = self._run(output, "--fixture", "B2", "--max-seconds", "20", "--no-render")
            self.assertEqual(result.returncode, 0, result.stderr)
            children = [item for item in output.iterdir() if item.is_dir()]
            self.assertEqual(len(children), 1)
            worker = self._json(children[0] / "worker_status.json")
            self.assertEqual(Path(worker["run_directory"]), children[0].resolve())
            self.assertEqual((output / "old.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
