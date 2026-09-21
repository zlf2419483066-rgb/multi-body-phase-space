import json
import tempfile
import unittest
from pathlib import Path

from im_fixtures import fixture_b, fixture_d
from phase_space_graph import (
    CountMode,
    LabelCatalogBundle,
    catalog_from_audit,
    normalize_physical_labeling,
    read_label_catalog,
    write_label_catalog,
)


def legacy_report(fixture, *, status="complete", stored_valid=True, failure_counts=None):
    checked = normalize_physical_labeling(fixture.topology, fixture.labeling)
    return {
        "topology": fixture.topology.to_dict(),
        "status": status,
        "count_mode": "fixed_labels",
        "counts": {"unique": 1 if checked.legacy_evaluation else 0},
        "valid_results": [
            dict(checked.legacy_evaluation.to_dict(), valid=stored_valid)
        ] if checked.legacy_evaluation else [],
        "failure_counts": failure_counts or {},
        "failure_samples": [],
        "failure_witnesses": {},
    }


class LM07LegacyImportTests(unittest.TestCase):
    def test_import_rechecks_records_and_merges_duplicate_reports(self) -> None:
        fixture = fixture_b()
        audit = {
            "status": "complete",
            "labeling_reports": [legacy_report(fixture), legacy_report(fixture)],
        }
        bundle = catalog_from_audit(
            audit,
            mode=CountMode.FIXED_LABELS,
            source_info={"source_sha256": "fixture-hash", "source_rule_version": "unknown"},
        )
        self.assertIsInstance(bundle, LabelCatalogBundle)
        self.assertEqual(len(bundle.catalogs), 1)
        self.assertEqual(bundle.catalogs[0].identity_count, 1)
        self.assertEqual(bundle.catalogs[0].existence, "yes")
        self.assertEqual(bundle.catalogs[0].search_status, "complete")
        self.assertEqual(bundle.catalogs[0].import_status, "complete")
        self.assertTrue(bundle.catalogs[0].enumeration_complete)
        self.assertEqual(dict(bundle.provenance)["source_sha256"], "fixture-hash")

    def test_stored_validity_conflict_is_kept_and_fails_import_status(self) -> None:
        fixture = fixture_b()
        audit = {"status": "complete", "labeling_reports": [legacy_report(fixture, stored_valid=False)]}
        catalog = catalog_from_audit(audit, mode="fixed_labels").catalogs[0]
        self.assertEqual(catalog.identity_count, 1)
        self.assertEqual(catalog.import_status, "failed")
        self.assertFalse(catalog.enumeration_complete)
        self.assertTrue(any(issue.code == "legacy_validity_conflict" for issue in catalog.issues))

    def test_missing_failure_witness_does_not_create_a_fake_label(self) -> None:
        fixture = fixture_b()
        audit = {
            "status": "complete",
            "labeling_reports": [legacy_report(fixture, failure_counts={"dependent_invariants": 2})],
        }
        catalog = catalog_from_audit(audit, mode=CountMode.FIXED_LABELS).catalogs[0]
        self.assertEqual(catalog.identity_count, 1)
        self.assertTrue(catalog.enumeration_complete)
        self.assertFalse(dict(catalog.search_evidence.provenance)["failure_witnesses_complete"])
        self.assertTrue(any(w.stage == "momentum_rejected" for w in catalog.rejected_witnesses))

    def test_incomplete_search_with_only_dependent_record_is_unknown(self) -> None:
        fixture = fixture_d()
        audit = {"status": "incomplete", "labeling_reports": [legacy_report(fixture, status="incomplete")]}
        catalog = catalog_from_audit(audit, mode=CountMode.FIXED_LABELS).catalogs[0]
        self.assertEqual(catalog.identity_count, 0)
        self.assertEqual(catalog.existence, "unknown")
        self.assertFalse(catalog.enumeration_complete)

    def test_round_trip_validates_key_digest_and_does_not_overwrite(self) -> None:
        fixture = fixture_b()
        bundle = catalog_from_audit(
            {"status": "complete", "labeling_reports": [legacy_report(fixture)]},
            mode=CountMode.FIXED_LABELS,
            source_info={"source_sha256": "roundtrip"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            write_label_catalog(bundle, path)
            loaded = read_label_catalog(path)
            self.assertEqual(loaded.catalogs[0].default_label_key, bundle.catalogs[0].default_label_key)
            self.assertEqual(loaded.catalogs[0].identities[0].key, bundle.catalogs[0].identities[0].key)
            with self.assertRaises(FileExistsError):
                write_label_catalog(bundle, path)
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["catalogs"])

    def test_source_and_old_audit_are_not_mutated(self) -> None:
        fixture = fixture_b()
        audit = {"status": "complete", "labeling_reports": [legacy_report(fixture)]}
        before = json.dumps(audit, sort_keys=True)
        catalog_from_audit(audit, mode=CountMode.FIXED_LABELS)
        self.assertEqual(json.dumps(audit, sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
