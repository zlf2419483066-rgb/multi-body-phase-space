import unittest

from im_fixtures import _fixture_c, fixture_b, fixture_d, fixture_e
from phase_space_graph import (
    CountMode,
    SearchEvidence,
    build_topology_catalog,
    normalize_physical_labeling,
)


class LM06CatalogTests(unittest.TestCase):
    def test_duplicate_inputs_are_one_identity_and_partial_search_is_not_complete(self) -> None:
        fixture = fixture_b()
        normalized = normalize_physical_labeling(fixture.topology, fixture.labeling)
        catalog = build_topology_catalog(
            fixture.topology,
            [normalized, normalized],
            mode=CountMode.FIXED_LABELS,
            search_evidence={"n": 4, "loop_count": 1, "search_status": "incomplete", "records_complete": False},
        )
        self.assertEqual(catalog.identity_count, 1)
        self.assertEqual(catalog.default_label_key, catalog.identities[0].key)
        self.assertEqual(catalog.existence, "yes")
        self.assertFalse(catalog.enumeration_complete)
        self.assertEqual(len(catalog.links), 1)

    def test_dependent_only_in_incomplete_search_is_unknown(self) -> None:
        fixture = fixture_d()
        normalized = normalize_physical_labeling(fixture.topology, fixture.labeling)
        catalog = build_topology_catalog(
            fixture.topology,
            [normalized],
            mode="fixed_labels",
            search_evidence=SearchEvidence(search_status="incomplete", records_complete=False),
        )
        self.assertEqual(catalog.identity_count, 0)
        self.assertEqual(catalog.existence, "unknown")
        self.assertFalse(catalog.enumeration_complete)
        self.assertEqual(len(catalog.rejected_witnesses), 1)

    def test_complete_zero_valid_records_is_no(self) -> None:
        fixture = fixture_e()
        normalized = normalize_physical_labeling(fixture.topology, fixture.labeling)
        catalog = build_topology_catalog(
            fixture.topology,
            [normalized],
            mode=CountMode.FIXED_LABELS,
            search_evidence={"search_status": "complete", "records_complete": True},
        )
        self.assertEqual(catalog.existence, "no")
        self.assertTrue(catalog.enumeration_complete)
        self.assertEqual(catalog.default_label_key, None)

    def test_import_conflict_prevents_complete_claim(self) -> None:
        fixture = fixture_b()
        normalized = normalize_physical_labeling(fixture.topology, fixture.labeling)
        catalog = build_topology_catalog(
            fixture.topology,
            [normalized],
            mode=CountMode.FIXED_LABELS,
            search_evidence={"search_status": "complete", "records_complete": True, "import_status": "failed"},
        )
        self.assertEqual(catalog.existence, "yes")
        self.assertFalse(catalog.enumeration_complete)
        self.assertEqual(catalog.import_status, "failed")

    def test_two_non_equivalent_c_records_are_both_retained(self) -> None:
        first = _fixture_c(3)
        second = _fixture_c(2)
        catalog = build_topology_catalog(
            first.topology,
            [
                normalize_physical_labeling(first.topology, first.labeling),
                normalize_physical_labeling(second.topology, second.labeling),
            ],
            mode=CountMode.PARTICLE_RELABELING,
            search_evidence={"search_status": "incomplete"},
        )
        self.assertEqual(catalog.identity_count, 2)
        self.assertEqual(len(catalog.links), 2)
        self.assertEqual(catalog.default_label_key, min(identity.key for identity in catalog.identities))

    def test_catalog_serializes_full_key_separately_from_digest(self) -> None:
        fixture = fixture_b()
        normalized = normalize_physical_labeling(fixture.topology, fixture.labeling)
        catalog = build_topology_catalog(
            fixture.topology,
            [normalized],
            mode=CountMode.FIXED_LABELS,
            search_evidence={"search_status": "incomplete"},
        ).to_dict()
        self.assertIn("key", catalog["identities"][0])
        self.assertIn("digest", catalog["identities"][0])
        self.assertNotEqual(catalog["identities"][0]["key"], catalog["identities"][0]["digest"])


if __name__ == "__main__":
    unittest.main()
