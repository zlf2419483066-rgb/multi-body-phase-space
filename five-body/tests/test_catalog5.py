from __future__ import annotations

import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.fixtures import fixture_a
from phase_space_graph5.label_catalog import CatalogError, CatalogManifest, build_catalog, load_catalog, write_catalog
from phase_space_graph5.label_identity import canonicalize_candidate_labeling
from phase_space_graph5.label_identity import relabel_candidate_record
from phase_space_graph5.physical_labels import normalize_candidate_labeling


class CandidateCatalogTests(unittest.TestCase):
    def test_roundtrip_preserves_candidate_schema_and_unrated_independence(self):
        topology, labeling = fixture_a()
        checked = normalize_candidate_labeling(topology, labeling)
        self.assertIsNotNone(checked.record)
        record = checked.record
        identity = canonicalize_candidate_labeling(record, "fixed_labels")
        manifest = CatalogManifest(
            scope={"fixture": "A"}, count_mode="fixed_labels", run_status="complete",
            topology_enumeration_complete=True, label_enumeration_complete=True,
            identity_save_complete=True, expected_identity_digests=(identity.digest,),
            stored_identity_digests=(identity.digest,),
        )
        catalog = build_catalog([identity], source_records={identity.digest: record}, manifest=manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = write_catalog(catalog, Path(directory) / "catalog")
            loaded = load_catalog(path)
        self.assertTrue(loaded.complete)
        self.assertEqual(loaded.entries[0].source_record.topology.vertices, (0, 1, 2, 3))
        self.assertEqual(loaded.manifest.independence["status"], "not_evaluated")
        self.assertIsNone(loaded.manifest.independence.get("rank"))

    def test_explicit_identity_list_is_not_a_claimed_enumeration(self):
        topology, labeling = fixture_a()
        record = normalize_candidate_labeling(topology, labeling).record
        identity = canonicalize_candidate_labeling(record, "fixed_labels")
        catalog = build_catalog([identity])
        self.assertFalse(catalog.complete)
        self.assertFalse(catalog.manifest.topology_enumeration_complete)
        self.assertEqual(catalog.manifest.stop_reason, "explicit_identity_selection")

    def test_load_rejects_complete_catalog_with_deadline_stop(self):
        topology, labeling = fixture_a()
        record = normalize_candidate_labeling(topology, labeling).record
        identity = canonicalize_candidate_labeling(record, "fixed_labels")
        manifest = CatalogManifest(
            scope={"fixture": "A"}, count_mode="fixed_labels", run_status="complete",
            topology_enumeration_complete=True, label_enumeration_complete=True,
            identity_save_complete=True, expected_identity_digests=(identity.digest,),
            stored_identity_digests=(identity.digest,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_catalog(build_catalog([identity], manifest=manifest), Path(directory) / "catalog")
            payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            payload["stop_reason"] = "deadline_exceeded"
            (path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(CatalogError):
                load_catalog(path)

    def test_manifest_rejects_non_mapping_failures(self):
        with self.assertRaises(CatalogError):
            CatalogManifest(failures="")
        topology, labeling = fixture_a()
        record = normalize_candidate_labeling(topology, labeling).record
        identity = canonicalize_candidate_labeling(record, "fixed_labels")
        manifest = CatalogManifest(
            scope={"fixture": "A"}, count_mode="fixed_labels", run_status="complete",
            topology_enumeration_complete=True, label_enumeration_complete=True,
            identity_save_complete=True, expected_identity_digests=(identity.digest,),
            stored_identity_digests=(identity.digest,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_catalog(build_catalog([identity], manifest=manifest), Path(directory) / "catalog")
            payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            payload["failures"] = ""
            (path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(CatalogError):
                load_catalog(path)

    def test_readback_rejects_index_or_mapping_damage(self):
        topology, labeling = fixture_a()
        record = normalize_candidate_labeling(topology, labeling).record
        identity = canonicalize_candidate_labeling(record, "fixed_labels")
        manifest = CatalogManifest(
            scope={"fixture": "A"}, count_mode="fixed_labels", run_status="complete",
            topology_enumeration_complete=True, label_enumeration_complete=True,
            identity_save_complete=True, expected_identity_digests=(identity.digest,),
            stored_identity_digests=(identity.digest,),
        )
        catalog = build_catalog([identity], manifest=manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = write_catalog(catalog, Path(directory) / "catalog")
            index_path = path / "identity_index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index[0]["digest"] = "0" * 64
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaises(CatalogError):
                load_catalog(path)

    def test_readback_requires_source_record_and_strict_index_line(self):
        topology, labeling = fixture_a()
        record = normalize_candidate_labeling(topology, labeling).record
        identity = canonicalize_candidate_labeling(record, "fixed_labels")
        manifest = CatalogManifest(
            scope={"fixture": "A"}, count_mode="fixed_labels", run_status="complete",
            topology_enumeration_complete=True, label_enumeration_complete=True,
            identity_save_complete=True, expected_identity_digests=(identity.digest,),
            stored_identity_digests=(identity.digest,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_catalog(build_catalog([identity], manifest=manifest), Path(directory) / "catalog")
            row_path = path / "identities.jsonl"
            row = json.loads(row_path.read_text(encoding="utf-8"))
            row.pop("source_record")
            row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(CatalogError):
                load_catalog(path)

            path = write_catalog(build_catalog([identity], manifest=manifest), Path(directory) / "catalog-index")
            index_path = path / "identity_index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index[0]["line"] = False
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaises(CatalogError):
                load_catalog(path)

    def test_readback_rejects_a_different_but_legal_particle_map(self):
        topology, labeling = fixture_a()
        record = normalize_candidate_labeling(topology, labeling).record
        identity = canonicalize_candidate_labeling(record, "fixed_labels")
        manifest = CatalogManifest(
            scope={"fixture": "A"}, count_mode="fixed_labels", run_status="complete",
            topology_enumeration_complete=True, label_enumeration_complete=True,
            identity_save_complete=True, expected_identity_digests=(identity.digest,),
            stored_identity_digests=(identity.digest,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_catalog(build_catalog([identity], manifest=manifest), Path(directory) / "catalog")
            row_path = path / "identities.jsonl"
            row = json.loads(row_path.read_text(encoding="utf-8"))
            row["particle_map"][0]["canonical"], row["particle_map"][1]["canonical"] = (
                row["particle_map"][1]["canonical"], row["particle_map"][0]["canonical"]
            )
            row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(CatalogError):
                load_catalog(path)

    def test_overwrite_keeps_identity_checkpoint_bytes(self):
        topology, labeling = fixture_a()
        record = normalize_candidate_labeling(topology, labeling).record
        identity = canonicalize_candidate_labeling(record, "fixed_labels")
        manifest = CatalogManifest(
            scope={"fixture": "A"}, count_mode="fixed_labels", run_status="complete",
            topology_enumeration_complete=True, label_enumeration_complete=True,
            identity_save_complete=True, expected_identity_digests=(identity.digest,),
            stored_identity_digests=(identity.digest,),
        )
        catalog = build_catalog([identity], manifest=manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = write_catalog(catalog, Path(directory) / "catalog")
            identity_bytes = (path / "identities.jsonl").read_bytes()
            index_bytes = (path / "identity_index.json").read_bytes()
            manifest_bytes = (path / "manifest.json").read_bytes()
            updated = replace(manifest, drawing_selection_complete=True)
            write_catalog(build_catalog([identity], manifest=updated), path, overwrite=True)
            self.assertEqual((path / "identities.jsonl").read_bytes(), identity_bytes)
            self.assertEqual((path / "identity_index.json").read_bytes(), index_bytes)
            self.assertNotEqual((path / "manifest.json").read_bytes(), manifest_bytes)
            self.assertTrue(load_catalog(path).complete)

    def test_roundtrip_keeps_original_record_alongside_canonical_identity(self):
        topology, labeling = fixture_a()
        record = normalize_candidate_labeling(topology, labeling).record
        source = relabel_candidate_record(record, {0: 10, 1: 20, 2: 30, 3: 40}).record
        identity = canonicalize_candidate_labeling(source, "fixed_labels")
        manifest = CatalogManifest(
            scope={"fixture": "renumbered-A"}, count_mode="fixed_labels", run_status="complete",
            topology_enumeration_complete=True, label_enumeration_complete=True,
            identity_save_complete=True, expected_identity_digests=(identity.digest,),
            stored_identity_digests=(identity.digest,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_catalog(build_catalog([identity], source_records={identity.digest: source}, manifest=manifest),
                                 Path(directory) / "catalog")
            loaded = load_catalog(path)
        self.assertEqual(loaded.entries[0].source_record.topology.vertices, (10, 20, 30, 40))
        self.assertEqual(loaded.entries[0].identity.canonical_record.topology.vertices, (0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
