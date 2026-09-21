from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.catalog_checkpoint import read_checkpoint
from phase_space_graph5.compact_reporting import (
    export_checkpoint_catalog,
    run_compact_candidates,
)
from phase_space_graph5.label_catalog import load_catalog
from phase_space_graph5.reporting import run_candidates


def _projection(identity, source):
    identity_payload = identity.to_dict()
    # CatalogIdentity includes its validated source field; CandidateIdentity
    # stores that field beside the identity in checkpoint rows.
    identity_payload.pop("source_record", None)
    return {
        "identity": identity_payload,
        "source_record": source.to_dict(),
        "vertex_map": list(identity.vertex_map),
        "particle_map": list(identity.particle_map),
        "edge_occurrence_map": list(identity.edge_occurrence_map),
        "final_slot_map": list(identity.final_slot_map),
    }


def _checkpoint_maps(checkpoint):
    result = {"fixed_labels": {}, "particle_relabeling": {}}
    for record in checkpoint.records:
        for mode, rows in record.identities_by_mode.items():
            for item in rows:
                result[mode][item.digest] = _projection(item.identity, item.source_record)
    return result


def _catalog_maps(catalogs):
    result = {}
    for mode, catalog in catalogs.items():
        result[mode] = {
            entry.digest: _projection(entry.identity, entry.source_record)
            for entry in catalog.entries
        }
    return result


class CompactReportingTests(unittest.TestCase):
    def test_l0_and_l1_checkpoint_full_identity_source_and_maps_match_old_catalogs(self) -> None:
        # L1 is intentionally included: this is the small complete scope used
        # to prove the compact boundary before any high-circle run.
        for loop_count, seconds in ((0, 20.0), (1, 120.0)):
            with self.subTest(loop_count=loop_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                ordinary = run_candidates(loop_count=loop_count, max_seconds=seconds,
                                          output_dir=root / "ordinary", render=False)
                self.assertEqual(ordinary.status, "complete")
                for strategy in ("default", "flow"):
                    with self.subTest(strategy=strategy):
                        compact = run_compact_candidates(
                            loop_count=loop_count, max_seconds=seconds,
                            output_dir=root / f"compact-{strategy}",
                            label_strategy=strategy, flow_record_order="reference",
                        )
                        self.assertEqual(compact.status, "complete")
                        self.assertTrue(compact.complete)
                        checkpoint = read_checkpoint(compact.checkpoint_directory)
                        self.assertTrue(checkpoint.complete)
                        self.assertEqual(len(checkpoint.records), ordinary.topology_report.unique_count)
                        self.assertEqual(_checkpoint_maps(checkpoint), _catalog_maps(ordinary.catalogs))
                        self.assertTrue(all(row["checkpoint_committed"] and row["complete"]
                                            for row in compact.compact_audit))
                        self.assertFalse(any("candidate_records" in row for row in compact.compact_audit))
                        self.assertFalse((Path(compact.output_directory) / "catalog" / "fixed_labels").exists())

    def test_empty_p_topology_has_complete_zero_identity_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_compact_candidates(loop_count=1, max_seconds=120.0,
                                            output_dir=Path(directory) / "compact")
            self.assertEqual(report.status, "complete")
            empty = [row for row in report.compact_audit if row["momentum_pass_count"] == 0]
            self.assertTrue(empty)
            self.assertTrue(all(row["complete"] and row["checkpoint_committed"] for row in empty))
            self.assertTrue(all(row["identity_counts"] == {"fixed_labels": 0,
                                                             "particle_relabeling": 0}
                                for row in empty))

    def test_flow_strategy_records_actual_generated_count_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_compact_candidates(loop_count=0, max_seconds=20.0,
                                            output_dir=Path(directory) / "flow",
                                            label_strategy="flow")
            self.assertEqual(report.status, "complete")
            self.assertEqual(sum(row["momentum_pass_count"] for row in report.compact_audit), 360)
            self.assertTrue(all(row["strategy"] == "flow" for row in report.compact_audit))
            self.assertTrue(all(row["raw_assignment_count_semantics"] ==
                                "flow_partition_generated_labelings"
                                for row in report.compact_audit))
            self.assertTrue(all(row["flow_counts"]["flow_partition_generated_labelings"] is not None
                                for row in report.compact_audit))
            self.assertTrue(all(row["scalar_flow_legal"] is True for row in report.compact_audit))
            self.assertTrue(all(isinstance(row["scalar_flow"], list) for row in report.compact_audit))

    def test_flow_failure_counts_cannot_claim_complete_checkpoint(self) -> None:
        from phase_space_graph5.flow_labeling import enumerate_flow_labelings

        def failing_provider(topology, **kwargs):
            original = enumerate_flow_labelings(topology, **kwargs)
            return replace(original, failure_counts=(("synthetic_failure", 1),))

        with tempfile.TemporaryDirectory() as directory:
            report = run_compact_candidates(loop_count=0, max_seconds=20.0,
                                            output_dir=Path(directory) / "flow-failure",
                                            label_strategy="flow", flow_provider=failing_provider)
            self.assertEqual(report.status, "incomplete")
            self.assertFalse(report.complete)
            self.assertTrue(any(row["failure_counts"].get("synthetic_failure") == 1
                                for row in report.compact_audit))
            checkpoint = read_checkpoint(report.checkpoint_directory)
            self.assertFalse(checkpoint.complete)
            self.assertTrue(all(not record.complete for record in checkpoint.records))

    def test_selected_checkpoint_export_reloads_full_identity_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_compact_candidates(loop_count=0, max_seconds=20.0,
                                            output_dir=root / "compact")
            checkpoint = read_checkpoint(report.checkpoint_directory)
            selected = checkpoint.topology_keys[:1]
            output = root / "export"
            catalog = export_checkpoint_catalog(report.checkpoint_directory, output,
                                                mode="particle_relabeling",
                                                topology_keys=selected)
            loaded = load_catalog(output)
            self.assertTrue(catalog.complete)
            self.assertTrue(loaded.complete)
            expected = {
                item.digest: _projection(item.identity, item.source_record)
                for item in checkpoint.records[0].identities_by_mode["particle_relabeling"]
            }
            actual = {entry.digest: _projection(entry.identity, entry.source_record)
                      for entry in loaded.entries}
            self.assertEqual(actual, expected)
            with self.assertRaises(ValueError):
                export_checkpoint_catalog(report.checkpoint_directory, root / "missing",
                                          mode="fixed_labels", topology_keys=[("missing",)])

    def test_timeout_and_cancel_are_incomplete_without_false_checkpoint_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            timeout = run_compact_candidates(loop_count=0, max_seconds=0.0,
                                             output_dir=Path(directory) / "timeout")
            self.assertEqual(timeout.status, "incomplete")
            self.assertFalse(timeout.complete)
            self.assertEqual(timeout.stop_reason, "deadline_exceeded")
            self.assertFalse(read_checkpoint(timeout.checkpoint_directory).complete)
            cancelled = run_compact_candidates(loop_count=0, max_seconds=20.0,
                                               output_dir=Path(directory) / "cancel",
                                               cancel_requested=lambda: True)
            self.assertEqual(cancelled.status, "incomplete")
            self.assertFalse(cancelled.complete)
            self.assertEqual(cancelled.stop_reason, "cancel_requested")
            self.assertFalse(read_checkpoint(cancelled.checkpoint_directory).complete)

    def test_checkpoint_persistence_failure_stops_compact_run(self) -> None:
        class FailingWriter:
            def __init__(self, root, **_kwargs):
                self.root = Path(root)

            def append_topology(self, *_args, **_kwargs):
                raise OSError("synthetic checkpoint persistence failure")

            _append_verified_topology = append_topology

        with tempfile.TemporaryDirectory() as directory:
            report = run_compact_candidates(loop_count=0, max_seconds=20.0,
                                            output_dir=Path(directory) / "persist-failure",
                                            checkpoint_writer_factory=FailingWriter)
            self.assertEqual(report.status, "incomplete")
            self.assertFalse(report.complete)
            self.assertTrue(any(item["stage"] == "checkpoint" for item in report.failures))
            self.assertTrue(report.compact_audit)
            self.assertFalse(report.compact_audit[0]["checkpoint_committed"])
            self.assertFalse(report.compact_audit[0]["complete"])

    def test_empty_checkpoint_writer_cannot_claim_committed_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_compact_candidates(
                loop_count=0, max_seconds=20.0,
                output_dir=Path(directory) / "empty-writer",
                checkpoint_writer_factory=lambda *_args, **_kwargs: None,
            )
            self.assertEqual(report.status, "incomplete")
            self.assertFalse(report.complete)
            self.assertEqual(report.stop_reason, "checkpoint_writer_invalid")
            self.assertFalse(report.compact_audit)
            self.assertTrue(any(item["stage"] == "checkpoint_writer" for item in report.failures))

    def test_late_finished_progress_failure_downgrades_compact_report(self) -> None:
        import phase_space_graph5.compact_reporting as compact_reporting

        original = compact_reporting._write_json_atomic
        stages = []

        def fail_only_finished(path, payload, *, deadline=None):
            stages.append(payload["stage"])
            if payload["stage"] == "finished":
                raise PermissionError("synthetic late sharing lock")
            return original(path, payload, deadline=deadline)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(compact_reporting, "_write_json_atomic",
                              side_effect=fail_only_finished):
                report = run_compact_candidates(loop_count=0, max_seconds=20.0,
                                                output_dir=Path(directory) / "late-failure")
            self.assertEqual(stages.count("finished"), 1)
            self.assertEqual(report.status, "incomplete")
            self.assertFalse(report.complete)
            manifest = json.loads((Path(report.output_directory) / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(manifest["stop_reason"], "progress_persistence_failed")

    def test_final_manifest_failure_downgrades_already_written_report(self) -> None:
        original_write_text = Path.write_text

        def fail_complete_manifest(path, text, *args, **kwargs):
            if path.name == "run_manifest.json":
                payload = json.loads(text)
                if payload.get("status") == "complete":
                    raise PermissionError("synthetic final sharing lock")
            return original_write_text(path, text, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, "write_text", new=fail_complete_manifest):
                report = run_compact_candidates(loop_count=0, max_seconds=20.0,
                                                output_dir=Path(directory) / "final-failure")
            self.assertEqual(report.status, "incomplete")
            self.assertFalse(report.complete)
            self.assertEqual(report.stop_reason, "compact_report_persistence_failed")
            root = Path(report.output_directory)
            self.assertEqual(json.loads((root / "run_report.json").read_text(encoding="utf-8"))["status"],
                             "incomplete")
            self.assertEqual(json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))["status"],
                             "incomplete")

    def test_final_report_serialization_crossing_deadline_downgrades_sidecars(self) -> None:
        import phase_space_graph5.compact_reporting as compact_reporting

        real_monotonic = compact_reporting.time.monotonic
        report_started = False
        original_write_text = Path.write_text

        def monotonic_after_report():
            value = real_monotonic()
            return value + 100.0 if report_started else value

        def mark_report_started(path, text, *args, **kwargs):
            nonlocal report_started
            if path.name == "run_report.json":
                report_started = True
            return original_write_text(path, text, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(compact_reporting.time, "monotonic", side_effect=monotonic_after_report), \
                 patch.object(Path, "write_text", new=mark_report_started):
                report = run_compact_candidates(loop_count=0, max_seconds=20.0,
                                                output_dir=Path(directory) / "deadline-sidecars")
            self.assertEqual(report.status, "incomplete")
            self.assertEqual(report.stop_reason, "deadline_exceeded")
            root = Path(report.output_directory)
            self.assertEqual(json.loads((root / "run_report.json").read_text(encoding="utf-8"))["status"],
                             "incomplete")
            self.assertEqual(json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))["status"],
                             "incomplete")

    def test_compact_requires_explicit_output_and_known_strategy(self) -> None:
        with self.assertRaises(ValueError):
            run_compact_candidates(loop_count=0, max_seconds=1.0, output_dir=None)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_compact_candidates(loop_count=0, max_seconds=1.0,
                                       output_dir=Path(directory) / "bad",
                                       label_strategy="unknown")


if __name__ == "__main__":
    unittest.main()
