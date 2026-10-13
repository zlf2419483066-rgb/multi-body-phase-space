from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT / "src"))

from phase_space_graph5.canonical import canonicalize_topology
from phase_space_graph5.domain import InternalEdge, Topology
from phase_space_graph5.independence.generic_family import GENERIC_THEOREM_VERSION
from phase_space_graph5.independence.generic_search import (
    GENERIC_SEARCH_COVERAGE_RULE,
    GENERIC_SEARCH_SCOPE,
    GENERIC_SEARCH_THEOREM_VERSION,
)


SCRIPT_PATH = DEMO_ROOT / "examples" / "summarize_generic_e.py"
SPEC = importlib.util.spec_from_file_location("summarize_generic_e", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
summarizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarizer)


THEOREM_SHA = "a" * 64


def topology_for_key(topology_key: list) -> Topology:
    return Topology(
        topology_key[1],
        range(topology_key[2]),
        [InternalEdge(edge[0], edge[1], edge[2], edge[3]) for edge in topology_key[5]],
        topology_key[3],
        list(enumerate(topology_key[4])),
    )


def key(number: int) -> list:
    chords = [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)]
    source, target = chords[(number - 1) % len(chords)]
    edges = [
        InternalEdge(0, 1, "double"),
        InternalEdge(1, 2, "double"),
        InternalEdge(2, 3, "double"),
        InternalEdge(3, 4, "double"),
        InternalEdge(source, target, "single"),
    ]
    topology = Topology(5, range(5), edges, 0, {0: 0, 1: 1, 2: 1, 3: 1, 4: 2})
    return json.loads(summarizer._key_token(canonicalize_topology(topology).key))


def search_certificate(topology_key: list, status: str = "reject") -> dict:
    topology = topology_for_key(topology_key)
    occurrences = [
        {"edge_index": index, "occurrence": occurrence}
        for index, edge in enumerate(topology_key[5]) if edge[2] == "single"
        for occurrence in range(edge[3])
    ]
    expected = 5 ** len(occurrences)
    return {
        "schema_version": "F5-I-generic-search-v1",
        "coverage_theorem_version": GENERIC_SEARCH_THEOREM_VERSION,
        "coverage_rule": GENERIC_SEARCH_COVERAGE_RULE,
        "coverage_scope": GENERIC_SEARCH_SCOPE,
        "topology": topology.to_dict(),
        "L": len(occurrences),
        "single_occurrence_order": occurrences,
        "assignment_order": "product(range(1,6), repeat=L)",
        "status": status,
        "complete": status == "reject",
        "attempted": expected if status == "reject" else 1,
        "expected": expected,
        "momentum_pass": 0 if status == "reject" else 1,
        "momentum_reject": expected if status == "reject" else 0,
        "independent_reject": 0,
        "witness": None,
        "errors": [],
    }


def noncanonical_equivalent_topology(topology_key: list) -> Topology:
    canonical_topology = topology_for_key(topology_key)
    permutation = (0, 2, 1, 3, 4)
    relabeled = Topology(
        canonical_topology.n,
        range(canonical_topology.n),
        [
            InternalEdge(
                permutation[edge.source], permutation[edge.target],
                edge.edge_type, edge.multiplicity,
            )
            for edge in canonical_topology.edges
        ],
        permutation[canonical_topology.initial_vertex],
        {permutation[vertex]: count for vertex, count in canonical_topology.final_slots},
    )
    assert summarizer._canonical_topology_token(relabeled.to_dict()) == summarizer._key_token(topology_key)
    return relabeled


def single_occurrence_order(topology: Topology) -> list[dict[str, int]]:
    return [
        {"edge_index": edge_index, "occurrence": occurrence}
        for edge_index, edge in enumerate(topology.edges)
        if edge.edge_type.value == "single"
        for occurrence in range(edge.expanded_count)
    ]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha(path: Path) -> tuple[str, int]:
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest(), len(content)


class GenericESummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def make_run(
        self,
        name: str,
        certificates: list[tuple[list, str]],
        *,
        run_state: str = "complete",
        bad_source_hash: bool = False,
        ref_indices: list[int] | None = None,
    ) -> dict:
        directory = self.base / name
        directory.mkdir(parents=True)
        source_file = directory / "source.jsonl"
        source_file.write_text("{}\n" * max(1, len(certificates)), encoding="utf-8")
        source_sha, source_size = sha(source_file)
        theorem_path = self.base / "theorem.md"
        theorem_path.write_text("fixture theorem binding\n", encoding="utf-8")
        labels = []
        index = []
        refs = ref_indices or list(range(len(certificates)))
        for i, (topology, expected_status) in enumerate(certificates):
            token = summarizer._key_token(topology)
            identity = ["IM-5-candidate-v1", "fixed_labels", topology, [], []]
            identity_token = summarizer._key_token(identity)
            label = {
                "schema_version": "F5-I-generic-witness-batch-v1",
                "topology_key": token,
                "identity_key": identity_token,
                "status": expected_status,
                "generic_status": "pass" if expected_status == "pass" else "reject",
                "generic_evidence": None,
                "generic_verifier": None,
                "search_result": None,
                "search_verifier": None,
                "errors": [],
                "source_row": {
                    "path": str(source_file.resolve()),
                    "index": i,
                    "file_sha256": source_sha,
                },
            }
            if expected_status == "pass":
                label["generic_evidence"] = {
                    "schema_version": "F5-I-generic-evidence-v1",
                    "theorem_version": GENERIC_THEOREM_VERSION,
                    "status": "pass", "topology_key": token,
                    "identity_key": identity_token,
                    "identity": {"identity_key": identity},
                    "source": {"topology_key": topology},
                }
                label["generic_verifier"] = {"valid": True, "status": "pass", "errors": []}
            elif expected_status == "reject":
                label["generic_status"] = "reject"
                label["search_result"] = search_certificate(topology, "reject")
                label["search_verifier"] = {
                    "valid": True, "status": "reject", "errors": [],
                    "checks": [
                        "topology_replayed", "full_assignment_space_replayed",
                        "stream_replayed", "reject_full_cover_verified",
                    ],
                }
            labels.append(label)
            index.append({
                "schema_version": "F5-I-generic-witness-batch-v1",
                "topology_key": token,
                "status": expected_status,
                "source_row_count": 1,
                "evaluated_count": 1,
                "certificate_rows": [refs[i]],
                "source_binding_ok": True,
                "input_error": False,
            })
        labels_path = directory / "labels.jsonl.gz"
        index_path = directory / "topology_index.jsonl.gz"
        write_gzip_jsonl(labels_path, labels)
        write_gzip_jsonl(index_path, index)
        source_after = "f" * 64 if bad_source_hash else source_sha
        manifest = {
            "schema_version": "F5-I-generic-witness-batch-v1",
            "status": run_state,
            "failed": False,
            "error": None,
            "runtime": {"mode": "l2", "offset": 0, "limit": 1, "binding_changed": False},
            "coverage": {"source_record_coverage_complete": run_state == "complete"},
            "sources": [{
                "path": str(source_file.resolve()),
                "sha256_before": source_sha,
                "sha256_after": source_after,
                "size_before": source_size,
                "size_after": source_size,
                "stable": not bad_source_hash,
                "read_to_eof": run_state == "complete",
            }],
            "source_errors": [],
            "bindings": {
                "theorem_before": {"path": str(theorem_path), "sha256": THEOREM_SHA},
                "theorem_after": {"path": str(theorem_path), "sha256": THEOREM_SHA},
                "implementation_sha256_before": {"runner.py": "b" * 64},
                "implementation_sha256_after": {"runner.py": "b" * 64},
                "binding_check_failed": False,
                "runner_sha256": "old-runner-hash",
            },
            "files": {"labels": labels_path.name, "topology_index": index_path.name},
        }
        manifest_path = directory / "manifest.json"
        write_json(manifest_path, manifest)
        return {
            "manifest": str(manifest_path),
            "topology_index": str(index_path),
            "labels": str(labels_path),
        }

    def run_plan(self, expected: list[list], runs: list[dict], *, source_type: str = "batch"):
        loop = {
            "loop": 1,
            "source_type": source_type,
            "expected_keys": expected,
        }
        if source_type == "batch":
            loop["runs"] = runs
        else:
            loop.update(runs[0])
        return summarizer.summarize_plan(
            {"required_theorem_sha256": THEOREM_SHA, "loops": [loop]},
            plan_path=self.base / "plan.json",
        )

    def make_legacy_run(
        self, candidate: list, status: str, *, include_label: bool,
        witness_count: int = 1, duplicate_first_identity: bool = False,
    ) -> dict:
        directory = self.base / f"legacy-{status}-{include_label}"
        directory.mkdir()
        token = summarizer._key_token(candidate)
        identities = [
            ["IM-5-candidate-v1", "fixed_labels", candidate, [index], []]
            for index in range(witness_count)
        ]
        identity_tokens = [summarizer._key_token(identity) for identity in identities]
        source = {"content_sha256": "content", "fingerprint": "fingerprint"}
        theorem = {"path": "theorem.md", "sha256": THEOREM_SHA}
        manifest_path = directory / "manifest.json"
        aggregation_path = directory / "aggregation.json"
        labels_path = directory / "labels.jsonl"
        label_rows = []
        if include_label:
            for identity, identity_token in zip(identities, identity_tokens):
                label_rows.append({
                    "topology_key": token,
                    "identity_key": identity_token,
                    "status": "pass",
                    "valid": True,
                    "errors": [],
                    "source_digest": "source-digest",
                    "evidence": {
                        "schema_version": "F5-I-generic-evidence-v1",
                        "theorem_version": GENERIC_THEOREM_VERSION,
                        "status": "pass",
                        "topology_key": token,
                        "identity_key": identity_token,
                        "identity": {"identity_key": identity},
                        "source": {"topology_key": candidate},
                        "source_digest": "source-digest",
                    },
                    "verification": {"valid": True, "status": "pass", "errors": []},
                })
            if duplicate_first_identity and label_rows:
                label_rows.append(dict(label_rows[0]))
        labels_path.write_text(
            "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in label_rows),
            encoding="utf-8",
        )
        write_json(manifest_path, {
            "status": "complete", "failed": False, "error": None,
            "runtime": {"binding_changed": False},
            "files": {"aggregation": aggregation_path.name, "labels": labels_path.name},
            "source": {"source_complete": True, "evaluation_complete": True, "snapshot": source},
            "bindings": {
                "theorem_before": theorem, "theorem_after": theorem,
                "implementation_sha256_before": {"runner.py": "c" * 64},
                "implementation_sha256_after": {"runner.py": "c" * 64},
            },
        })
        witnesses = identity_tokens if status == "pass" else []
        write_json(aggregation_path, {
            "schema_version": "F5-I-generic-aggregation-v1",
            "status": "pass", "disk_coverage_verified": True,
            "source_binding_ok": True, "evaluation_complete": True,
            "source": {"content_digest": "content", "snapshot_fingerprint": "fingerprint"},
            "topologies": [{
                "topology_key": token, "status": status,
                "identity_keys": identity_tokens,
                "pass_identity_keys": witnesses,
                "witness_identity_keys": witnesses,
                "reject_identity_keys": identity_tokens if status == "reject" else [],
            }],
        })
        return {
            "manifest": str(manifest_path),
            "aggregation": str(aggregation_path),
            "labels": str(labels_path),
        }

    def test_mixed_verified_pass_and_full_reject(self) -> None:
        first, second = key(1), key(2)
        run = self.make_run("mixed", [(first, "pass"), (second, "reject")])
        index, summary = self.run_plan([first, second], [run])
        self.assertEqual(summary["status"], "pass")
        self.assertEqual([row["status"] for row in index], ["pass", "reject"])
        self.assertEqual(summary["loops"][0]["counts"], {"pass": 1, "reject": 1, "unknown": 0})
        self.assertEqual(index[1]["reject_basis"]["attempted"], 5)

    def test_missing_duplicate_and_extra_keys_are_reported(self) -> None:
        first, missing, extra = key(1), key(2), key(3)
        run = self.make_run("coverage", [(first, "pass"), (first, "pass"), (extra, "pass")])
        index, summary = self.run_plan([first, missing], [run])
        loop = summary["loops"][0]
        self.assertEqual(summary["status"], "unknown")
        self.assertEqual([row["status"] for row in index], ["unknown", "unknown"])
        self.assertEqual(loop["coverage"]["missing"][0]["topology_key"], missing)
        self.assertEqual(loop["coverage"]["extra"][0]["topology_key"], extra)
        self.assertEqual(loop["coverage"]["duplicates"][0]["topology_key"], first)
        self.assertEqual(len(index), 2, "extra complete keys must not enter E")

    def test_bad_manifest_source_hash_cannot_retain_pass(self) -> None:
        candidate = key(1)
        run = self.make_run("bad-hash", [(candidate, "pass")], bad_source_hash=True)
        index, summary = self.run_plan([candidate], [run])
        self.assertEqual(summary["status"], "unknown")
        self.assertEqual(index[0]["status"], "unknown")
        self.assertIn("batch_source_binding_0_unstable", " ".join(summary["loops"][0]["errors"]))

    def test_reject_with_incomplete_assignment_count_is_unknown(self) -> None:
        candidate = key(1)
        run = self.make_run("fake-reject", [(candidate, "reject")])
        labels_path = Path(run["labels"])
        with gzip.open(labels_path, "rt", encoding="utf-8") as stream:
            label = json.loads(stream.readline())
        label["search_result"]["attempted"] = 4
        write_gzip_jsonl(labels_path, [label])
        index, _ = self.run_plan([candidate], [run])
        self.assertEqual(index[0]["status"], "unknown")
        self.assertIsNone(index[0]["reject_basis"])

    def test_certificate_reference_must_match_topology_key(self) -> None:
        first, second = key(1), key(2)
        run = self.make_run("wrong-ref", [(first, "pass"), (second, "pass")], ref_indices=[1, 1])
        index, _ = self.run_plan([first, second], [run])
        self.assertEqual(index[0]["status"], "unknown")
        self.assertIn("certificate_row_1_topology_key_mismatch", index[0]["errors"])

    def test_search_topology_must_match_index_key(self) -> None:
        candidate, other = key(1), key(2)
        run = self.make_run("swapped-search-topology", [(candidate, "reject")])
        labels_path = Path(run["labels"])
        with gzip.open(labels_path, "rt", encoding="utf-8") as stream:
            label = json.loads(stream.readline())
        label["search_result"]["topology"] = topology_for_key(other).to_dict()
        write_gzip_jsonl(labels_path, [label])
        index, _ = self.run_plan([candidate], [run])
        self.assertEqual(index[0]["status"], "unknown")
        self.assertIn("search_topology_key_mismatch", index[0]["errors"])

    def test_search_occurrence_order_uses_serialized_edge_indices(self) -> None:
        candidate = key(1)
        run = self.make_run("noncanonical-search-edge-order", [(candidate, "reject")])
        labels_path = Path(run["labels"])
        with gzip.open(labels_path, "rt", encoding="utf-8") as stream:
            label = json.loads(stream.readline())
        topology = noncanonical_equivalent_topology(candidate)
        serialized_order = single_occurrence_order(topology)
        self.assertNotEqual(serialized_order, label["search_result"]["single_occurrence_order"])
        label["search_result"]["topology"] = topology.to_dict()
        label["search_result"]["single_occurrence_order"] = serialized_order
        write_gzip_jsonl(labels_path, [label])

        index, _ = self.run_plan([candidate], [run])

        self.assertEqual(index[0]["status"], "reject")
        self.assertNotIn("search_single_occurrence_order_mismatch", index[0]["errors"])

    def test_wrong_occurrence_order_for_equivalent_topology_remains_unknown(self) -> None:
        candidate = key(1)
        run = self.make_run("wrong-noncanonical-search-edge-order", [(candidate, "reject")])
        labels_path = Path(run["labels"])
        with gzip.open(labels_path, "rt", encoding="utf-8") as stream:
            label = json.loads(stream.readline())
        topology = noncanonical_equivalent_topology(candidate)
        serialized_order = single_occurrence_order(topology)
        self.assertNotEqual(serialized_order, label["search_result"]["single_occurrence_order"])
        label["search_result"]["topology"] = topology.to_dict()
        write_gzip_jsonl(labels_path, [label])

        index, _ = self.run_plan([candidate], [run])

        self.assertEqual(index[0]["status"], "unknown")
        self.assertIn("search_single_occurrence_order_mismatch", index[0]["errors"])

    def test_search_pass_with_empty_witness_is_unknown(self) -> None:
        candidate = key(1)
        run = self.make_run("empty-search-witness", [(candidate, "reject")])
        labels_path = Path(run["labels"])
        with gzip.open(labels_path, "rt", encoding="utf-8") as stream:
            label = json.loads(stream.readline())
        label["search_result"].update({"status": "pass", "complete": False, "attempted": 1, "witness": {}})
        label["status"] = "pass"
        label["search_verifier"].update({
            "status": "pass",
            "checks": [
                "topology_replayed", "assignment_prefix_replayed", "stream_replayed",
                "pass_prefix_verified", "witness_topology_verified", "witness_generic_certificate_verified",
            ],
        })
        write_gzip_jsonl(labels_path, [label])
        index, _ = self.run_plan([candidate], [run])
        self.assertEqual(index[0]["status"], "unknown")
        self.assertIn("search_pass_witness_record_or_evidence_missing", index[0]["errors"])

    def test_incomplete_partial_shards_merge_against_full_expected_set(self) -> None:
        first, second = key(1), key(2)
        shard_a = self.make_run("shard-a", [(first, "pass")], run_state="incomplete")
        shard_b = self.make_run("shard-b", [(second, "reject")], run_state="incomplete")
        index, summary = self.run_plan([first, second], [shard_a, shard_b])
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["loops"][0]["coverage"]["complete"])
        self.assertEqual([row["status"] for row in index], ["pass", "reject"])

    def test_expected_manifest_hash_mismatch_forces_unknown(self) -> None:
        candidate = key(1)
        expected_path = self.base / "expected.json.gz"
        with gzip.open(expected_path, "wt", encoding="utf-8") as stream:
            json.dump({"loops": [{"loop": 1, "topology_keys": [candidate]}]}, stream)
        run = self.make_run("wrong-expected-sha", [(candidate, "pass")])
        plan = {
            "expected_p_manifest": {"path": str(expected_path), "sha256": "0" * 64},
            "required_theorem_sha256": THEOREM_SHA,
            "loops": [{"loop": 1, "source_type": "batch", "runs": [run]}],
        }
        index, summary = summarizer.summarize_plan(plan, plan_path=self.base / "plan.json")
        self.assertEqual(index[0]["status"], "unknown")
        self.assertIn("expected_p_manifest_sha256_mismatch", summary["errors"])

    def test_expected_manifest_mutation_forces_every_index_row_unknown(self) -> None:
        candidates = [key(1), key(2)]
        expected_path = self.base / "expected-mutable.json.gz"
        with gzip.open(expected_path, "wt", encoding="utf-8") as stream:
            json.dump({"loops": [{"loop": 1, "topology_keys": candidates}]}, stream)
        expected_sha, _ = sha(expected_path)
        run = self.make_run("expected-mutation", [(candidate, "pass") for candidate in candidates])

        class MutatingTracker(summarizer.InputTracker):
            def finalize(self):
                expected_path.write_bytes(expected_path.read_bytes() + b"changed")
                return super().finalize()

        plan = {
            "expected_p_manifest": {"path": str(expected_path), "sha256": expected_sha},
            "required_theorem_sha256": THEOREM_SHA,
            "loops": [{"loop": 1, "source_type": "batch", "runs": [run]}],
        }
        index, summary = summarizer.summarize_plan(
            plan, plan_path=self.base / "plan.json", tracker=MutatingTracker()
        )
        self.assertEqual([row["status"] for row in index], ["unknown", "unknown"])
        self.assertIn("expected_manifest_changed_during_summary", summary["errors"])

    def test_duplicate_plan_loop_is_rejected_without_duplicate_index_rows(self) -> None:
        candidate = key(1)
        run = self.make_run("duplicate-plan", [(candidate, "pass")])
        loop = {"loop": 1, "source_type": "batch", "expected_keys": [candidate], "runs": [run]}
        index, summary = summarizer.summarize_plan(
            {"required_theorem_sha256": THEOREM_SHA, "loops": [loop, dict(loop)]},
            plan_path=self.base / "plan.json",
        )
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0]["status"], "unknown")
        self.assertIn("duplicate_plan_loop_1", summary["errors"])

    def test_legacy_candidate_reject_is_not_topology_reject(self) -> None:
        candidate = key(1)
        run = self.make_legacy_run(candidate, "reject", include_label=False)
        loop = {
            "loop": 1, "source_type": "legacy", "expected_keys": [candidate],
            **run,
        }
        index, _ = summarizer.summarize_plan(
            {"required_theorem_sha256": THEOREM_SHA, "loops": [loop]},
            plan_path=self.base / "plan.json",
        )
        self.assertEqual(index[0]["status"], "unknown")
        self.assertEqual(index[0]["certificate_refs"], [])

    def test_legacy_aggregation_pass_without_real_label_certificate_is_unknown(self) -> None:
        candidate = key(1)
        run = self.make_legacy_run(candidate, "pass", include_label=False)
        loop = {"loop": 1, "source_type": "legacy", "expected_keys": [candidate], **run}
        index, _ = summarizer.summarize_plan(
            {"required_theorem_sha256": THEOREM_SHA, "loops": [loop]},
            plan_path=self.base / "plan.json",
        )
        self.assertEqual(index[0]["status"], "unknown")
        self.assertIn("legacy_aggregation_witness_has_no_valid_labels_certificate", index[0]["errors"])

    def test_legacy_pass_references_a_real_labels_row(self) -> None:
        candidate = key(1)
        run = self.make_legacy_run(candidate, "pass", include_label=True)
        loop = {"loop": 1, "source_type": "legacy", "expected_keys": [candidate], **run}
        index, _ = summarizer.summarize_plan(
            {"required_theorem_sha256": THEOREM_SHA, "loops": [loop]},
            plan_path=self.base / "plan.json",
        )
        self.assertEqual(index[0]["status"], "pass")
        self.assertEqual(index[0]["certificate_refs"][0]["row"], 0)
        self.assertEqual(index[0]["certificate_refs"][0]["kind"], "legacy_generic_label")

    def test_legacy_multiple_distinct_pass_identities_use_existential_witness(self) -> None:
        candidate = key(1)
        run = self.make_legacy_run(candidate, "pass", include_label=True, witness_count=3)
        loop = {"loop": 1, "source_type": "legacy", "expected_keys": [candidate], **run}
        index, _ = summarizer.summarize_plan(
            {"required_theorem_sha256": THEOREM_SHA, "loops": [loop]},
            plan_path=self.base / "plan.json",
        )
        self.assertEqual(index[0]["status"], "pass")
        self.assertEqual(index[0]["certificate_refs"][0]["row"], 0)

    def test_legacy_duplicate_labels_rows_for_same_identity_are_unknown(self) -> None:
        candidate = key(1)
        run = self.make_legacy_run(
            candidate, "pass", include_label=True, witness_count=2,
            duplicate_first_identity=True,
        )
        loop = {"loop": 1, "source_type": "legacy", "expected_keys": [candidate], **run}
        index, _ = summarizer.summarize_plan(
            {"required_theorem_sha256": THEOREM_SHA, "loops": [loop]},
            plan_path=self.base / "plan.json",
        )
        self.assertEqual(index[0]["status"], "unknown")
        self.assertIn("legacy_witness_identity_has_duplicate_labels_rows", index[0]["errors"])


if __name__ == "__main__":
    unittest.main()
