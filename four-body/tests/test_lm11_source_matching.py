import unittest

from im_fixtures import fixture_b, fixture_d, _fixture_c
from phase_space_graph import (
    CountMode,
    DoubleLineLabel,
    Labeling,
    SourceDiagram,
    build_topology_catalog,
    match_source_record,
    normalize_physical_labeling,
    relabel_physical_record,
)


def _complete_catalog(topology, *normalizations, complete=True):
    return build_topology_catalog(
        topology,
        normalizations,
        mode=CountMode.FIXED_LABELS,
        search_evidence={
            "search_status": "complete" if complete else "incomplete",
            "records_complete": complete,
            "import_status": "complete",
        },
    )


class LM11SourceMatchingTests(unittest.TestCase):
    def test_verified_b_matches_after_source_vertex_renumbering(self):
        fixture = fixture_b()
        normalization = normalize_physical_labeling(fixture.topology, fixture.labeling)
        catalog = _complete_catalog(fixture.topology, normalization)
        moved = relabel_physical_record(
            fixture.record,
            {0: 10, 1: 20, 2: 30, 3: 40},
        ).record
        source = SourceDiagram("PSF", 12, "B", moved.topology, True, moved.complete_labeling)
        result = match_source_record(source, [catalog], mode=CountMode.FIXED_LABELS)
        self.assertEqual(result["status"], "catalog_match")
        self.assertTrue(result["catalog_match"])
        self.assertEqual(result["topology_status"], "topology_match")
        self.assertEqual(result["label_status"], "label_match")

    def test_topology_only_and_unverified_source_remain_inconclusive(self):
        fixture = fixture_b()
        catalog = _complete_catalog(fixture.topology, normalize_physical_labeling(fixture.topology, fixture.labeling))
        topology_only = SourceDiagram("PSF", 12, "B", fixture.topology, True, None)
        result = match_source_record(topology_only, [catalog], mode="fixed_labels")
        self.assertEqual(result["topology_status"], "topology_match")
        self.assertEqual(result["label_status"], "not_provided")
        self.assertEqual(result["status"], "inconclusive")
        unverified = SourceDiagram("PSF", 12, "B", fixture.topology, False, fixture.labeling)
        self.assertEqual(match_source_record(unverified, [catalog], mode="fixed_labels")["status"], "inconclusive")

    def test_wrong_explicit_double_line_is_rejected_by_physics(self):
        fixture = fixture_b()
        checked = normalize_physical_labeling(fixture.topology, fixture.labeling)
        catalog = _complete_catalog(fixture.topology, checked)
        complete = fixture.record.complete_labeling
        doubles = [
            DoubleLineLabel(item.edge_index, item.occurrence, (1, 3) if item.edge_index == 1 else item.particle_ids)
            for item in complete.double_lines
        ]
        bad = Labeling(complete.final_particles, complete.single_lines, doubles)
        result = match_source_record(SourceDiagram("PSF", 12, "B", fixture.topology, True, bad), [catalog], mode="fixed_labels")
        self.assertEqual(result["label_status"], "momentum_rejected")
        self.assertIn("declared_double_momentum_mismatch", {item["code"] for item in result["diagnostics"]})
        self.assertFalse(result["catalog_match"])

    def test_partial_catalog_does_not_claim_program_only_or_complete_match(self):
        fixture = fixture_b()
        normalization = normalize_physical_labeling(fixture.topology, fixture.labeling)
        catalog = _complete_catalog(fixture.topology, normalization, complete=False)
        result = match_source_record(
            SourceDiagram("PSF", 12, "B", fixture.topology, True, fixture.labeling),
            [catalog],
            mode="fixed_labels",
        )
        self.assertEqual(result["label_status"], "label_match")
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["catalog_match"])
        self.assertTrue(result["undetermined"])

    def test_c_alpha_and_beta_have_separate_source_identities(self):
        alpha = _fixture_c(3)
        beta = _fixture_c(2)
        catalog = _complete_catalog(
            alpha.topology,
            normalize_physical_labeling(alpha.topology, alpha.labeling),
            normalize_physical_labeling(beta.topology, beta.labeling),
        )
        source_a = SourceDiagram("PSF", 12, "C-甲", alpha.topology, True, alpha.labeling)
        source_b = SourceDiagram("PSF", 12, "C-乙", beta.topology, True, beta.labeling)
        result_a = match_source_record(source_a, [catalog], mode="fixed_labels")
        result_b = match_source_record(source_b, [catalog], mode="fixed_labels")
        self.assertEqual(result_a["status"], "catalog_match")
        self.assertEqual(result_b["status"], "catalog_match")
        self.assertNotEqual(result_a["source_identity_key"], result_b["source_identity_key"])
        self.assertEqual(len(result_a["program_only"]), 1)


if __name__ == "__main__":
    unittest.main()
