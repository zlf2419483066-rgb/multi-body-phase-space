from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.fixtures import fixture_a, fixture_b
from phase_space_graph5.label_identity import canonicalize_candidate_labeling
from phase_space_graph5.label_catalog import CatalogManifest, build_catalog, write_catalog
from phase_space_graph5.physical_labels import normalize_candidate_labeling
from phase_space_graph5.independence.generic_aggregation import aggregate_generic
from phase_space_graph5.independence.sources import VerifiedSource, read_source
from phase_space_graph5.independence.generic_family import evaluate_generic_label


def _source(which: str) -> VerifiedSource:
    topology, labeling = fixture_a() if which == "A" else fixture_b(2)
    checked = normalize_candidate_labeling(topology, labeling)
    if checked.record is None:
        raise AssertionError("fixture did not produce a momentum candidate")
    identity = canonicalize_candidate_labeling(checked.record, "fixed_labels")
    return VerifiedSource(identity, checked.record, f"generic-fixture-{which}")


def _batch(which: str = "A"):
    return read_source(_source(which).source_record)


def _nested_set(payload: dict, name: str, value: object) -> None:
    if name in payload:
        payload[name] = value
        return
    for group in ("identity", "source", "certificate"):
        nested = payload.get(group)
        if isinstance(nested, dict) and name in nested:
            nested[name] = value
            return
    raise AssertionError(f"evidence has no {name} binding")


def _runner_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "examples" / "run_generic_independence.py"
    spec = importlib.util.spec_from_file_location("run_generic_independence_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load generic runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GenericAggregationTests(unittest.TestCase):
    def test_real_evaluator_is_reverified_and_memory_source_cannot_reject(self) -> None:
        batch = _batch("A")
        evidence = evaluate_generic_label(batch.records[0])
        self.assertIsInstance(evidence, dict)
        self.assertEqual(evidence.get("status"), "pass")
        result = aggregate_generic(batch, [evidence])
        self.assertFalse(result["source_complete"])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["counts"]["valid_pass"], 1)
        self.assertEqual(result["counts"]["source_records"], 1)

    def test_B2_real_reject_is_not_full_source_reject(self) -> None:
        batch = _batch("B2")
        evidence = evaluate_generic_label(batch.records[0])
        self.assertEqual(evidence.get("status"), "reject")
        result = aggregate_generic(batch, [evidence])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["source_complete"])
        self.assertEqual(result["topologies"][0]["status"], "unknown")
        self.assertEqual(result["counts"]["valid_reject"], 1)

    def test_tampered_source_binding_is_unknown_even_when_real_evaluator_was_used(self) -> None:
        batch = _batch("B2")
        evidence = deepcopy(evaluate_generic_label(batch.records[0]))
        _nested_set(evidence, "source_digest", "tampered")
        result = aggregate_generic(batch, [evidence])
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["source_binding_ok"])
        self.assertTrue(all(item["status"] == "unknown" for item in result["topologies"]))

    def test_missing_and_out_of_range_evidence_cannot_complete_reject(self) -> None:
        batch = _batch("A")
        evidence = evaluate_generic_label(batch.records[0])
        extra = deepcopy(evidence)
        _nested_set(extra, "identity_key", "[\"outside\"]")
        result = aggregate_generic(batch, [evidence, extra])
        # The valid in-range PASS remains a usable partial contribution; the
        # out-of-range row still prevents a complete evaluation and cannot
        # turn the source into a REJECT.
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["evaluation_complete"])
        self.assertTrue(result["out_of_range_evaluations"])
        self.assertNotEqual(result["status"], "reject")

    def test_duplicate_evidence_is_not_silently_first_wins(self) -> None:
        batch = _batch("A")
        evidence = evaluate_generic_label(batch.records[0])
        result = aggregate_generic(batch, [evidence, deepcopy(evidence)])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["evaluation_complete"])
        self.assertTrue(result["duplicate_identity_keys"])
        row = result["topologies"][0]["evaluations"][0]
        self.assertEqual(row["status"], "unknown")

    def test_missing_status_is_unknown_and_not_a_physical_reject(self) -> None:
        batch = _batch("A")
        evidence = deepcopy(evaluate_generic_label(batch.records[0]))
        evidence.pop("status", None)
        result = aggregate_generic(batch, [evidence])
        self.assertEqual(result["status"], "unknown")
        self.assertNotEqual(result["status"], "reject")

    def test_changed_source_snapshot_invalidates_all_topology_conclusions(self) -> None:
        original = _batch("A")
        changed_snapshot = replace(original.snapshot, content_sha256="0" * 64)
        changed = replace(original, snapshot=changed_snapshot)
        result = aggregate_generic(changed, [evaluate_generic_label(changed.records[0])])
        self.assertFalse(result["source_binding_ok"])
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(all(item["status"] == "unknown" for item in result["topologies"]))

    def test_runner_reads_one_explicit_complete_catalog_and_writes_bound_output(self) -> None:
        source = _source("A")
        manifest = CatalogManifest(
            scope={"test": "generic-runner"},
            marker_scope="all_stored",
            topology_scope="explicit",
            run_status="complete",
            topology_enumeration_complete=True,
            label_enumeration_complete=True,
            identity_save_complete=True,
            expected_identity_digests=(source.digest,),
            stored_identity_digests=(source.digest,),
        )
        catalog = build_catalog(
            [source.identity], source_records={source.digest: source.source_record}, manifest=manifest
        )
        runner = _runner_module()
        with tempfile.TemporaryDirectory(prefix="f5-generic-run-") as temporary:
            root = Path(temporary)
            source_path = write_catalog(catalog, root / "source")
            theorem = root / "theorem.md"
            theorem.write_text("generic test theorem\n", encoding="utf-8")
            output = root / "output"
            code = runner.run_generic(source_path, output, 300.0, theorem)
            self.assertEqual(code, 0)
            payload = __import__("json").loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertTrue(payload["source"]["source_complete"])
            self.assertTrue(payload["source"]["evaluation_complete"])
            self.assertIsNotNone(payload["bindings"]["theorem_before"]["sha256"])
            self.assertEqual(payload["aggregation"]["status"], "pass")
            self.assertTrue((output / "labels.jsonl").is_file())
            self.assertTrue((output / "source_index.json").is_file())

            failed_output = root / "failed-output"
            code = runner.run_generic(source_path, failed_output, 300.0, root / "missing-theorem.md")
            self.assertEqual(code, 1)
            failed_manifest = __import__("json").loads(
                (failed_output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed_manifest["status"], "failed")
            self.assertIn("theorem", failed_manifest["error"])

            (root / "existing").mkdir()
            (root / "existing" / "sentinel.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                runner.run_generic(source_path, root / "existing", 300.0, theorem)


if __name__ == "__main__":
    unittest.main()
