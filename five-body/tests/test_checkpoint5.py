from __future__ import annotations

import json
import gzip
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.catalog_checkpoint import (
    CheckpointCorruptionError,
    CheckpointError,
    CatalogCheckpointWriter,
    DATA_FILENAME,
    assert_identity_key_equivalence,
    read_checkpoint,
)
from phase_space_graph5.fixtures import fixture_a, fixture_b
from phase_space_graph5.label_catalog import CatalogManifest, build_catalog
from phase_space_graph5.label_identity import canonicalize_candidate_labeling, relabel_candidate_record
from phase_space_graph5.physical_labels import normalize_candidate_labeling


def _identity_pair(topology, labeling):
    record = normalize_candidate_labeling(topology, labeling).record
    assert record is not None
    return record, {
        "fixed_labels": canonicalize_candidate_labeling(record, "fixed_labels"),
        "particle_relabeling": canonicalize_candidate_labeling(record, "particle_relabeling"),
    }


def _full_catalog(identity, source, mode):
    manifest = CatalogManifest(
        scope={"test": "checkpoint"}, count_mode=mode, run_status="complete",
        topology_enumeration_complete=True, label_enumeration_complete=True,
        identity_save_complete=True, expected_identity_digests=(identity.digest,),
        stored_identity_digests=(identity.digest,),
    )
    return build_catalog([identity], source_records={identity.digest: source}, manifest=manifest)


class CatalogCheckpointTests(unittest.TestCase):
    def test_append_read_and_compare_full_identity_keys(self) -> None:
        topology, labeling = fixture_a()
        source, identities = _identity_pair(topology, labeling)
        key = identities["fixed_labels"].topology_key
        with tempfile.TemporaryDirectory() as directory:
            writer = CatalogCheckpointWriter(
                Path(directory) / "checkpoint", scope={"n": 5, "loop_count": 0},
                expected_topology_keys=(key,),
            )
            writer.append_topology(
                key, identities_by_mode={mode: (identity,) for mode, identity in identities.items()},
                source_records_by_mode={mode: {identity.digest: source} for mode, identity in identities.items()},
                label_enumeration_complete={mode: True for mode in identities},
                identity_save_complete={mode: True for mode in identities},
                topology_complete=True,
            )
            result = read_checkpoint(Path(directory) / "checkpoint")
            self.assertTrue(result.complete)
            self.assertEqual(result.identity_digests_by_mode["fixed_labels"], frozenset({identities["fixed_labels"].digest}))
            assert_identity_key_equivalence(result, {
                "fixed_labels": _full_catalog(identities["fixed_labels"], source, "fixed_labels"),
                "particle_relabeling": _full_catalog(identities["particle_relabeling"], source, "particle_relabeling"),
            })

    def test_equivalent_source_representation_is_allowed(self) -> None:
        topology, labeling = fixture_a()
        original, _ = _identity_pair(topology, labeling)
        alternate = relabel_candidate_record(original, {0: 10, 1: 20, 2: 30, 3: 40}).record
        identity = canonicalize_candidate_labeling(alternate, "fixed_labels")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=(identity.topology_key,))
            writer.append_topology(
                identity.topology_key,
                identities_by_mode={"fixed_labels": (identity,), "particle_relabeling": ()},
                source_records_by_mode={"fixed_labels": {identity.digest: alternate}, "particle_relabeling": {}},
                label_enumeration_complete={"fixed_labels": True, "particle_relabeling": True},
                identity_save_complete={"fixed_labels": True, "particle_relabeling": True},
                topology_complete=True,
            )
            loaded = read_checkpoint(root)
            self.assertTrue(loaded.complete)
            self.assertEqual(loaded.records[0].identities_by_mode["fixed_labels"][0].source_record.topology.vertices, (10, 20, 30, 40))

    def test_uncommitted_data_and_truncated_index_tail_are_ignored(self) -> None:
        first_topology, first_labeling = fixture_a()
        second_topology, second_labeling = fixture_b(3)
        first_source, first = _identity_pair(first_topology, first_labeling)
        second_source, second = _identity_pair(second_topology, second_labeling)
        keys = (first["fixed_labels"].topology_key, second["fixed_labels"].topology_key)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=keys)
            for source, values in ((first_source, first), (second_source, second)):
                writer.append_topology(
                    values["fixed_labels"].topology_key,
                    identities_by_mode={mode: (identity,) for mode, identity in values.items()},
                    source_records_by_mode={mode: {identity.digest: source} for mode, identity in values.items()},
                    label_enumeration_complete={mode: True for mode in values},
                    identity_save_complete={mode: True for mode in values},
                    topology_complete=True,
                )
            data_path = root / DATA_FILENAME
            data_path.write_bytes(data_path.read_bytes() + b'{"record_type":"uncommitted"}\n')
            index_path = root / "index.jsonl"
            index_bytes = index_path.read_bytes()
            second_newline = index_bytes.rfind(b"\n")
            previous_newline = index_bytes.rfind(b"\n", 0, second_newline)
            index_path.write_bytes(index_bytes[:previous_newline + 1] + index_bytes[previous_newline + 1:second_newline])
            loaded = read_checkpoint(root)
            self.assertFalse(loaded.complete)
            self.assertEqual(len(loaded.records), 1)

    def test_corruption_of_committed_data_fails_closed(self) -> None:
        topology, labeling = fixture_a()
        source, identities = _identity_pair(topology, labeling)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=(identities["fixed_labels"].topology_key,))
            writer.append_topology(
                identities["fixed_labels"].topology_key,
                identities_by_mode={mode: (identity,) for mode, identity in identities.items()},
                source_records_by_mode={mode: {identity.digest: source} for mode, identity in identities.items()},
                label_enumeration_complete={mode: True for mode in identities},
                identity_save_complete={mode: True for mode in identities},
                topology_complete=True,
            )
            index = json.loads((root / "index.jsonl").read_text(encoding="utf-8"))
            commit = index
            index_row = commit
            offset = index_row["data_offset"]
            data = bytearray((root / DATA_FILENAME).read_bytes())
            data[offset + 2] = ord("X")
            (root / DATA_FILENAME).write_bytes(bytes(data))
            with self.assertRaises(CheckpointError):
                read_checkpoint(root)

    def test_duplicate_or_out_of_scope_topology_is_rejected(self) -> None:
        topology, labeling = fixture_a()
        source, identities = _identity_pair(topology, labeling)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=(identities["fixed_labels"].topology_key,))
            kwargs = {
                "identities_by_mode": {mode: (identity,) for mode, identity in identities.items()},
                "source_records_by_mode": {mode: {identity.digest: source} for mode, identity in identities.items()},
                "label_enumeration_complete": {mode: True for mode in identities},
                "identity_save_complete": {mode: True for mode in identities},
                "topology_complete": True,
            }
            writer.append_topology(identities["fixed_labels"].topology_key, **kwargs)
            with self.assertRaises(CheckpointError):
                writer.append_topology(identities["fixed_labels"].topology_key, **kwargs)

    def test_empty_candidate_topology_keeps_audit_and_completion(self) -> None:
        topology, _ = fixture_a()
        source_topology, source_labeling = fixture_a()
        source, identities = _identity_pair(source_topology, source_labeling)
        del source, identities
        from phase_space_graph5.canonical import canonicalize_topology
        key = canonicalize_topology(topology).key
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5, "loop_count": 0}, expected_topology_keys=(key,))
            writer.append_topology(
                key,
                identities_by_mode={"fixed_labels": (), "particle_relabeling": ()},
                label_enumeration_complete={"fixed_labels": True, "particle_relabeling": True},
                identity_save_complete={"fixed_labels": True, "particle_relabeling": True},
                topology_complete=True,
                audit={"raw_assignment_count": 12, "momentum_pass_count": 0, "candidate_count": 0},
            )
            result = read_checkpoint(root)
            self.assertTrue(result.complete)
            self.assertEqual(result.records[0].audit["raw_assignment_count"], 12)
            self.assertEqual(result.records[0].audit["momentum_pass_count"], 0)

    def test_scope_checksum_and_n_loop_binding_fail_closed(self) -> None:
        topology, labeling = fixture_a()
        source, identities = _identity_pair(topology, labeling)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5, "loop_count": 1}, expected_topology_keys=(identities["fixed_labels"].topology_key,))
            with self.assertRaises(CheckpointError):
                writer.append_topology(
                    identities["fixed_labels"].topology_key,
                    identities_by_mode={mode: (identity,) for mode, identity in identities.items()},
                    source_records_by_mode={mode: {identity.digest: source} for mode, identity in identities.items()},
                    label_enumeration_complete={mode: True for mode in identities},
                    identity_save_complete={mode: True for mode in identities},
                    topology_complete=True,
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=(identities["fixed_labels"].topology_key,))
            writer.append_topology(
                identities["fixed_labels"].topology_key,
                identities_by_mode={mode: (identity,) for mode, identity in identities.items()},
                source_records_by_mode={mode: {identity.digest: source} for mode, identity in identities.items()},
                label_enumeration_complete={mode: True for mode in identities},
                identity_save_complete={mode: True for mode in identities},
                topology_complete=True,
            )
            payload = json.loads((root / "scope.json").read_text(encoding="utf-8"))
            payload["expected_scope_complete"] = False
            (root / "scope.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(CheckpointError):
                read_checkpoint(root)

    def test_stored_mapping_damage_fails_after_block_checksum_update(self) -> None:
        topology, labeling = fixture_a()
        source, identities = _identity_pair(topology, labeling)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=(identities["fixed_labels"].topology_key,))
            writer.append_topology(
                identities["fixed_labels"].topology_key,
                identities_by_mode={mode: (identity,) for mode, identity in identities.items()},
                source_records_by_mode={mode: {identity.digest: source} for mode, identity in identities.items()},
            )
            index = json.loads((root / "index.jsonl").read_text(encoding="utf-8"))
            data_path = root / DATA_FILENAME
            compressed = data_path.read_bytes()
            commit = index
            offset, length = commit["data_offset"], commit["data_length"]
            row = json.loads(gzip.decompress(compressed[offset:offset + length]).decode("utf-8"))
            row["modes"]["fixed_labels"][0]["identity"]["vertex_map"][0]["canonical"] += 1
            plain = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
            replacement = gzip.compress(plain, compresslevel=1, mtime=0)
            data_path.write_bytes(replacement)
            commit["data_length"] = len(replacement)
            commit["data_sha256"] = __import__("hashlib").sha256(replacement).hexdigest()
            commit["data_uncompressed_size"] = len(plain)
            (root / "index.jsonl").write_text(json.dumps(commit, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(CheckpointCorruptionError):
                read_checkpoint(root)

    def test_catalog_entries_are_normalized_to_checkpoint_identities(self) -> None:
        topology, labeling = fixture_a()
        source, identities = _identity_pair(topology, labeling)
        fixed_entry = _full_catalog(identities["fixed_labels"], source, "fixed_labels").entries[0]
        particle_entry = _full_catalog(identities["particle_relabeling"], source, "particle_relabeling").entries[0]
        key = identities["fixed_labels"].topology_key
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=(key,))
            writer.append_topology(
                key,
                identities_by_mode={"fixed_labels": (fixed_entry,), "particle_relabeling": (particle_entry,)},
                label_enumeration_complete={"fixed_labels": True, "particle_relabeling": True},
                identity_save_complete={"fixed_labels": True, "particle_relabeling": True},
                topology_complete=True,
            )
            self.assertTrue(read_checkpoint(root).complete)

    def test_public_defaults_do_not_claim_complete_coverage(self) -> None:
        topology, labeling = fixture_a()
        source, identities = _identity_pair(topology, labeling)
        key = identities["fixed_labels"].topology_key
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=(key,))
            writer.append_topology(
                key,
                identities_by_mode={mode: (identity,) for mode, identity in identities.items()},
                source_records_by_mode={mode: {identity.digest: source} for mode, identity in identities.items()},
                topology_complete=True,
            )
            record = read_checkpoint(root).records[0]
            self.assertFalse(record.complete)
            self.assertFalse(any(record.label_enumeration_complete.values()))
            self.assertFalse(any(record.identity_save_complete.values()))

    def test_stop_reason_forces_incomplete_topology_checkpoint(self) -> None:
        topology, labeling = fixture_a()
        source, identities = _identity_pair(topology, labeling)
        key = identities["fixed_labels"].topology_key
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=(key,))
            writer.append_topology(
                key,
                identities_by_mode={mode: (identity,) for mode, identity in identities.items()},
                source_records_by_mode={mode: {identity.digest: source} for mode, identity in identities.items()},
                label_enumeration_complete={mode: True for mode in identities},
                identity_save_complete={mode: True for mode in identities},
                topology_complete=True,
                audit={"stop_reason": "deadline_exceeded"},
            )
            record = read_checkpoint(root).records[0]
            self.assertFalse(record.complete)
            self.assertFalse(record.topology_complete)
            self.assertFalse(any(record.identity_save_complete.values()))

    def test_reopen_refuses_to_append_after_truncated_index_tail(self) -> None:
        topology, labeling = fixture_a()
        source, identities = _identity_pair(topology, labeling)
        key = identities["fixed_labels"].topology_key
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            writer = CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=(key,))
            writer.append_topology(
                key,
                identities_by_mode={mode: (identity,) for mode, identity in identities.items()},
                source_records_by_mode={mode: {identity.digest: source} for mode, identity in identities.items()},
                label_enumeration_complete={mode: True for mode in identities},
                identity_save_complete={mode: True for mode in identities},
                topology_complete=True,
            )
            index_path = root / "index.jsonl"
            index_path.write_bytes(index_path.read_bytes() + b'{"record_type":"truncated"}')
            with self.assertRaises(CheckpointError):
                CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=(key,))


if __name__ == "__main__":
    unittest.main()
