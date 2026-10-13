from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.canonical import canonicalize_topology
from phase_space_graph5.fixtures import fixture_a, fixture_b
from phase_space_graph5.independence.sources import VerifiedSource, _key_token
from phase_space_graph5.label_identity import canonicalize_candidate_labeling
from phase_space_graph5.physical_labels import normalize_candidate_labeling


def _runner():
    path = Path(__file__).resolve().parents[1] / "examples" / "run_generic_witness_batch.py"
    spec = importlib.util.spec_from_file_location("generic_witness_batch_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError("runner module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(which: str = "A") -> VerifiedSource:
    topology, labeling = fixture_a() if which == "A" else fixture_b(2)
    checked = normalize_candidate_labeling(topology, labeling)
    if checked.record is None:
        raise AssertionError(checked.to_dict())
    identity = canonicalize_candidate_labeling(checked.record, "fixed_labels")
    return VerifiedSource(identity, checked.record, "test-witness")


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _json_bytes(value: object, *, sort_keys: bool = True) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":")
    ).encode("utf-8")


class GenericWitnessBatchTests(unittest.TestCase):
    def test_witness_mode_streams_one_real_row_and_stops_on_pass(self) -> None:
        source = _source()
        row = {
            "topology": source.source_record.topology.to_dict(),
            "key": _jsonable(source.topology_key),
            "key_sha256": hashlib.sha256(_json_bytes(source.topology_key, sort_keys=False)).hexdigest(),
            "normalized_record": source.source_record.to_dict(),
        }
        runner = _runner()
        with tempfile.TemporaryDirectory(prefix="f5-witness-batch-") as directory:
            root = Path(directory)
            source_path = root / "witness.jsonl"
            source_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            theorem = root / "theorem.md"
            theorem.write_text("test theorem\n", encoding="utf-8")
            output = root / "output"
            code = runner.run_batch(
                [source_path], output, mode="witness", offset=0, limit=None,
                per_topology_seconds=30.0, max_seconds=60.0, theorem=theorem,
                expected_count=1,
            )
            self.assertEqual(code, 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["runtime"]["search_backend"], "reference")
            self.assertTrue(manifest["coverage"]["topology_evaluation_complete"])
            self.assertFalse(manifest["coverage"]["all_identity_coverage"])
            self.assertFalse(manifest["coverage"]["source_complete"])
            with gzip.open(output / "labels.jsonl.gz", "rt", encoding="utf-8") as stream:
                label = json.loads(next(stream))
            self.assertEqual(label["status"], "pass")
            self.assertEqual(label["source_row"]["index"], 0)
            self.assertIsInstance(label["source_row"]["record"], dict)
            with gzip.open(output / "topology_index.jsonl.gz", "rt", encoding="utf-8") as stream:
                index = json.loads(next(stream))
            self.assertEqual(index["status"], "pass")
            self.assertNotIn("record", index)

    def test_l2_restores_topology_without_faking_a_source_witness(self) -> None:
        source = _source()
        topology_key = canonicalize_topology(source.source_record.topology).key
        runner = _runner()
        with tempfile.TemporaryDirectory(prefix="f5-l2-batch-") as directory:
            root = Path(directory)
            manifest_path = root / "drawing_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "test-l2",
                        "rows": [
                            {"kind": "structure", "status": "complete", "validation": {"valid": True}},
                            {
                            "kind": "marked",
                            "number": 1,
                            "topology_key": _jsonable(topology_key),
                            "status": "complete",
                            "validation": {"valid": True},
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            theorem = root / "theorem.md"
            theorem.write_text("test theorem\n", encoding="utf-8")
            output = root / "output"
            code = runner.run_batch(
                [manifest_path], output, mode="l2", offset=0, limit=None,
                per_topology_seconds=30.0, max_seconds=60.0, theorem=theorem,
                expected_count=1,
            )
            self.assertEqual(code, 0)
            with gzip.open(output / "labels.jsonl.gz", "rt", encoding="utf-8") as stream:
                label = json.loads(next(stream))
            self.assertIsNone(label["identity_key"])
            self.assertIsNone(label["source_row"]["record"])
            self.assertEqual(label["generic_status"], "not_applicable_l2")
            self.assertEqual(label["status"], "pass")

    def test_limit_is_incomplete_and_does_not_claim_source_coverage(self) -> None:
        source = _source()
        row = {
            "topology": source.source_record.topology.to_dict(),
            "key": _jsonable(source.topology_key),
            "normalized_record": source.source_record.to_dict(),
        }
        runner = _runner()
        with tempfile.TemporaryDirectory(prefix="f5-limit-batch-") as directory:
            root = Path(directory)
            source_path = root / "witness.jsonl"
            source_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            theorem = root / "theorem.md"
            theorem.write_text("test theorem\n", encoding="utf-8")
            output = root / "output"
            code = runner.run_batch(
                [source_path], output, mode="witness", offset=0, limit=0,
                per_topology_seconds=30.0, max_seconds=60.0, theorem=theorem,
                expected_count=1,
            )
            self.assertEqual(code, 2)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "incomplete")
            self.assertFalse(manifest["coverage"]["evaluation_complete"])
            self.assertFalse(manifest["coverage"]["source_complete"])

    def test_rejecting_representative_keeps_search_witness_record_for_replay(self) -> None:
        source = _source("B2")
        row = {
            "topology": source.source_record.topology.to_dict(),
            "key": _jsonable(source.topology_key),
            "key_sha256": hashlib.sha256(_json_bytes(source.topology_key, sort_keys=False)).hexdigest(),
            "normalized_record": source.source_record.to_dict(),
        }
        runner = _runner()
        with tempfile.TemporaryDirectory(prefix="f5-b2-search-") as directory:
            root = Path(directory)
            source_path = root / "witness.jsonl"
            source_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            theorem = root / "theorem.md"
            theorem.write_text("test theorem\n", encoding="utf-8")
            output = root / "output"
            with (
                patch.object(runner, "search_generic_topology", wraps=runner.search_generic_topology) as search_call,
                patch.object(runner, "verify_generic_search", wraps=runner.verify_generic_search) as verify_call,
            ):
                code = runner.run_batch(
                    [source_path], output, mode="witness", offset=0, limit=None,
                    per_topology_seconds=30.0, max_seconds=60.0, theorem=theorem,
                    expected_count=1, search_backend="reference",
                )
            self.assertEqual(code, 0)
            search_call.assert_called_once()
            verify_call.assert_called_once()
            self.assertEqual(search_call.call_args.kwargs["backend"], "reference")
            self.assertEqual(verify_call.call_args.kwargs["backend"], "reference")
            with gzip.open(output / "labels.jsonl.gz", "rt", encoding="utf-8") as stream:
                label = json.loads(next(stream))
            self.assertEqual(label["generic_status"], "reject")
            self.assertEqual(label["status"], "pass")
            witness = label["search_result"]["witness"]
            self.assertIsInstance(witness["record"], dict)
            self.assertNotEqual(witness["record"], label["source_row"]["record"])
            self.assertEqual(label["search_verifier"]["valid"], True)
            self.assertEqual(label["search_verifier"]["status"], "pass")
            self.assertEqual(label["witness_reference"]["kind"], "generic_search")
            self.assertEqual(
                label["witness_reference"]["identity_key"],
                witness["evidence"]["identity_key"],
            )
            replay = runner.verify_generic_search(
                source.source_record.topology,
                label["search_result"],
                max_seconds=30.0,
            )
            self.assertTrue(replay["valid"])
            self.assertEqual(replay["status"], "pass")

    def test_duplicate_complete_topology_key_forces_unknown_and_incomplete(self) -> None:
        source = _source()
        row = {
            "topology": source.source_record.topology.to_dict(),
            "key": _jsonable(source.topology_key),
            "normalized_record": source.source_record.to_dict(),
        }
        runner = _runner()
        with tempfile.TemporaryDirectory(prefix="f5-duplicate-topology-") as directory:
            root = Path(directory)
            source_path = root / "witness.jsonl"
            source_path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )
            theorem = root / "theorem.md"
            theorem.write_text("test theorem\n", encoding="utf-8")
            output = root / "output"
            code = runner.run_batch(
                [source_path], output, mode="witness", offset=0, limit=None,
                per_topology_seconds=30.0, max_seconds=60.0, theorem=theorem,
                expected_count=2,
            )
            self.assertEqual(code, 2)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "incomplete")
            self.assertFalse(manifest["coverage"]["evaluation_complete"])
            self.assertEqual(manifest["topologies"]["pass"], 0)
            self.assertEqual(manifest["topologies"]["unknown"], 1)
            self.assertTrue(any(
                item.get("error") == "duplicate_complete_topology_key"
                for item in manifest["source_errors"]
            ))
            with gzip.open(output / "topology_index.jsonl.gz", "rt", encoding="utf-8") as stream:
                index = json.loads(next(stream))
            self.assertEqual(index["status"], "unknown")
            self.assertTrue(index["source_binding_ok"])

    def test_theorem_binding_change_gates_every_topology_to_unknown(self) -> None:
        source = _source()
        row = {
            "topology": source.source_record.topology.to_dict(),
            "key": _jsonable(source.topology_key),
            "normalized_record": source.source_record.to_dict(),
        }
        runner = _runner()
        with tempfile.TemporaryDirectory(prefix="f5-theorem-change-") as directory:
            root = Path(directory)
            source_path = root / "witness.jsonl"
            source_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            theorem = root / "theorem.md"
            theorem.write_text("test theorem\n", encoding="utf-8")
            output = root / "output"
            theorem_before = {"path": str(theorem), "sha256": "before"}
            theorem_after = {"path": str(theorem), "sha256": "after"}
            with patch.object(runner, "_theorem_binding", side_effect=[theorem_before, theorem_after]):
                code = runner.run_batch(
                    [source_path], output, mode="witness", offset=0, limit=None,
                    per_topology_seconds=30.0, max_seconds=60.0, theorem=theorem,
                    expected_count=1,
                )
            self.assertEqual(code, 2)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["runtime"]["binding_changed"])
            self.assertEqual(manifest["topologies"]["pass"], 0)
            self.assertEqual(manifest["topologies"]["unknown"], 1)
            with gzip.open(output / "topology_index.jsonl.gz", "rt", encoding="utf-8") as stream:
                index = json.loads(next(stream))
            self.assertEqual(index["status"], "unknown")
            self.assertFalse(index["source_binding_ok"])

    def test_final_implementation_hash_failure_writes_failed_unknown_manifest(self) -> None:
        source = _source()
        row = {
            "topology": source.source_record.topology.to_dict(),
            "key": _jsonable(source.topology_key),
            "normalized_record": source.source_record.to_dict(),
        }
        runner = _runner()
        with tempfile.TemporaryDirectory(prefix="f5-binding-failure-") as directory:
            root = Path(directory)
            source_path = root / "witness.jsonl"
            source_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            theorem = root / "theorem.md"
            theorem.write_text("test theorem\n", encoding="utf-8")
            output = root / "output"
            with patch.object(
                runner,
                "_implementation_hashes",
                side_effect=[{"before.py": "before"}, RuntimeError("hash read failed")],
            ):
                code = runner.run_batch(
                    [source_path], output, mode="witness", offset=0, limit=None,
                    per_topology_seconds=30.0, max_seconds=60.0, theorem=theorem,
                    expected_count=1,
                )
            self.assertEqual(code, 1)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue(manifest["bindings"]["binding_check_failed"])
            self.assertEqual(manifest["topologies"]["pass"], 0)
            self.assertEqual(manifest["topologies"]["unknown"], 1)
            with gzip.open(output / "topology_index.jsonl.gz", "rt", encoding="utf-8") as stream:
                index = json.loads(next(stream))
            self.assertEqual(index["status"], "unknown")
            self.assertFalse(index["source_binding_ok"])


if __name__ == "__main__":
    unittest.main()
