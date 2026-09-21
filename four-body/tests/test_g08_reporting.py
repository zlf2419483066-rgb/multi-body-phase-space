import json
import tempfile
import unittest
from pathlib import Path

from phase_space_graph import (
    EnumerationLimits,
    RuntimeMetadata,
    SourceReference,
    build_audit_report,
    collect_runtime_metadata,
    enumerate_invariant_combinations,
    enumerate_labelings,
    enumerate_topologies,
    run_audit,
    write_audit_bundle,
)


class AuditReportTests(unittest.TestCase):
    def _complete_tree_inputs(self):
        topology_report = enumerate_topologies(4, 0)
        labeling_reports = tuple(enumerate_labelings(topology) for topology in topology_report.topologies)
        combination_report = enumerate_invariant_combinations(4, sizes=(2,))
        return topology_report, labeling_reports, combination_report

    def test_complete_tree_audit_contains_layered_counts_and_json(self) -> None:
        topology_report, labeling_reports, combination_report = self._complete_tree_inputs()
        runtime = RuntimeMetadata(
            rule_version="test",
            python_version="3.10",
            platform="test-platform",
            package_version="0.1.0",
            git_commit="test-commit",
            git_dirty=False,
            test_command="test command",
        )
        audit = build_audit_report(
            topology_report,
            labeling_reports,
            combination_report,
            source_references=(SourceReference("fixture", "tree", 2, verified=True),),
            runtime=runtime,
            started_at="2026-09-10T00:00:00+00:00",
            finished_at="2026-09-10T00:00:01+00:00",
        )
        self.assertTrue(audit.complete)
        self.assertEqual(audit.counts["topologies"], 2)
        self.assertEqual(audit.counts["effective_topologies"], 2)
        self.assertEqual(audit.counts["variable_combination_classes"], 2)
        self.assertEqual(audit.source_status, "count_match")
        self.assertEqual(audit.to_dict()['acceptance']['catalog_status'], 'not_checked')
        restored = json.loads(audit.to_json())
        self.assertEqual(restored["status"], "complete")
        self.assertEqual(restored["runtime"]["git_commit"], "test-commit")
        self.assertIn("四体图枚举审计报告", audit.to_markdown())

        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = write_audit_bundle(audit, directory, stem="tree_audit")
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["status"], "complete")
            self.assertIn("分层计数", markdown_path.read_text(encoding="utf-8"))

    def test_missing_labeling_report_is_incomplete(self) -> None:
        topology_report, labeling_reports, combination_report = self._complete_tree_inputs()
        audit = build_audit_report(topology_report, labeling_reports[:1], combination_report)
        self.assertFalse(audit.complete)
        self.assertEqual(audit.status.value, "incomplete")
        self.assertIn("missing_labeling_reports", {issue.code for issue in audit.issues})

    def test_incomplete_topology_report_remains_incomplete(self) -> None:
        topology_report = enumerate_topologies(4, 0, limits=EnumerationLimits(max_topologies=1))
        labeling_reports = tuple(enumerate_labelings(topology) for topology in topology_report.topologies)
        combination_report = enumerate_invariant_combinations(4, sizes=(2,))
        audit = build_audit_report(topology_report, labeling_reports, combination_report)
        self.assertFalse(audit.complete)
        self.assertIn("topology_search_incomplete", {issue.code for issue in audit.issues})

    def test_runtime_collection_is_serializable(self) -> None:
        metadata = collect_runtime_metadata(Path.cwd())
        data = metadata.to_dict()
        self.assertIn("python_version", data)
        self.assertIn("git_commit", data)
        self.assertIsInstance(json.dumps(data), str)

    def test_run_audit_orchestrates_layers_and_preserves_limits(self) -> None:
        audit = run_audit(
            4,
            0,
            topology_limits=EnumerationLimits(max_topologies=1),
        )
        self.assertFalse(audit.complete)
        self.assertEqual(audit.topology_report.status.value, "incomplete")
        self.assertIn("topology_search_incomplete", {issue.code for issue in audit.issues})


if __name__ == "__main__":
    unittest.main()
