import unittest
from types import SimpleNamespace
from unittest.mock import patch

from im_fixtures import fixture_b
from phase_space_graph import (
    CountMode,
    EnumerationLimits,
    LabelingLimits,
    run_label_catalog,
)
from phase_space_graph.label_catalog import catalog_from_audit


class LM08FreshAdapterTests(unittest.TestCase):
    def test_fresh_tree_run_returns_legacy_and_catalog(self) -> None:
        result = run_label_catalog(4, 0, mode=CountMode.FIXED_LABELS)
        self.assertEqual(result.legacy_audit.topology_report.n, 4)
        self.assertEqual(result.legacy_audit.topology_report.loop_count, 0)
        self.assertTrue(result.label_catalog.catalogs)
        self.assertEqual(
            {identity.key for catalog in result.label_catalog.catalogs for identity in catalog.identities},
            {
                identity.key
                for catalog in catalog_from_audit(result.legacy_audit.to_dict(), mode=CountMode.FIXED_LABELS).catalogs
                for identity in catalog.identities
            },
        )

    def test_limits_mode_and_symmetry_are_passed_without_rewriting(self) -> None:
        fixture = fixture_b()
        checked = fixture.record
        audit_mapping = {
            "status": "incomplete",
            "labeling_reports": [
                {
                    "topology": fixture.topology.to_dict(),
                    "status": "incomplete",
                    "valid_results": [
                        dict(
                            {
                                "valid": True,
                                "labeling": checked.complete_labeling.to_dict(),
                            }
                        )
                    ],
                    "failure_counts": {},
                }
            ],
        }
        fake_audit = SimpleNamespace(
            status=SimpleNamespace(value="incomplete"),
            to_dict=lambda: audit_mapping,
        )
        topology_limits = object()
        labeling_limits = object()
        with patch("phase_space_graph.label_catalog.run_audit", return_value=fake_audit) as mocked:
            result = run_label_catalog(
                4,
                1,
                mode="fixed_labels",
                topology_limits=topology_limits,
                labeling_limits=labeling_limits,
                use_symmetry=False,
            )
        mocked.assert_called_once_with(
            n=4,
            loop_count=1,
            count_mode=CountMode.FIXED_LABELS,
            topology_limits=topology_limits,
            labeling_limits=labeling_limits,
            use_symmetry=False,
        )
        self.assertEqual(result.label_catalog.catalogs[0].search_status, "incomplete")
        self.assertFalse(result.label_catalog.catalogs[0].enumeration_complete)

    def test_conversion_conflict_is_returned_with_legacy_audit(self) -> None:
        fixture = fixture_b()
        audit_mapping = {
            "status": "complete",
            "labeling_reports": [
                {
                    "topology": fixture.topology.to_dict(),
                    "status": "complete",
                    "valid_results": [
                        {"valid": False, "labeling": fixture.labeling.to_dict()}
                    ],
                    "failure_counts": {},
                }
            ],
        }
        fake_audit = SimpleNamespace(
            status=SimpleNamespace(value="complete"),
            to_dict=lambda: audit_mapping,
        )
        with patch("phase_space_graph.label_catalog.run_audit", return_value=fake_audit):
            result = run_label_catalog(4, 0, mode=CountMode.FIXED_LABELS)
        catalog = result.label_catalog.catalogs[0]
        self.assertEqual(catalog.import_status, "failed")
        self.assertEqual(catalog.existence, "yes")
        self.assertFalse(catalog.enumeration_complete)

    def test_mode_is_explicit_and_unknown_values_fail(self) -> None:
        with self.assertRaises(TypeError):
            run_label_catalog(4, 0)
        with self.assertRaises(ValueError):
            run_label_catalog(4, 0, mode="not-a-mode")


if __name__ == "__main__":
    unittest.main()
