from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "run_candidates.py"


class FlowParticleOrbitCliTests(unittest.TestCase):
    def _run(self, output: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments, "--output", str(output)],
            cwd=str(SCRIPT.parents[1]),
            text=True,
            capture_output=True,
            timeout=30.0,
        )

    def test_flow_particle_orbit_is_explicit_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "orbit"
            result = self._run(
                output,
                "--compact",
                "--label-strategy",
                "flow",
                "--flow-particle-orbit",
                "--loop-count",
                "0",
                "--max-seconds",
                "20",
                "--no-render",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["complete"])
            self.assertTrue(report["flow_particle_orbit"])
            self.assertTrue(report["context"]["execution"]["flow_particle_orbit"])
            self.assertEqual(report["label_strategy"], "flow")
            scope = json.loads(
                (output / "catalog" / "checkpoints" / "scope.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(scope["scope"]["flow_particle_orbit"])
            self.assertTrue(report["compact_audit"])
            self.assertTrue(all(row["flow_particle_orbit"] for row in report["compact_audit"]))
            self.assertTrue(all(row["particle_orbit_reuse"] for row in report["compact_audit"]))
            self.assertTrue(
                all(row["flow_counts"]["flow_partition_normalized_labelings"] == 0
                    for row in report["compact_audit"])
            )

    def test_flow_particle_orbit_rejects_meaningless_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            no_compact = self._run(output, "--flow-particle-orbit", "--max-seconds", "0")
            self.assertNotEqual(no_compact.returncode, 0)
            self.assertIn("requires --compact", no_compact.stderr)

            wrong_strategy = self._run(
                output,
                "--compact",
                "--flow-particle-orbit",
                "--max-seconds",
                "0",
            )
            self.assertNotEqual(wrong_strategy.returncode, 0)
            self.assertIn("requires --label-strategy flow", wrong_strategy.stderr)


if __name__ == "__main__":
    unittest.main()
