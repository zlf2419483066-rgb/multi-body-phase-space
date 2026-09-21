import json
import unittest

from phase_space_graph.domain import ValidationIssue
from phase_space_graph.physical_labels import (
    NORMALIZATION_STAGES,
    NormalizationResult,
    PhysicalLabelRecord,
    Provenance,
)
from im_fixtures import all_fixtures


class PhysicalLabelRecordTests(unittest.TestCase):
    def test_hand_calculated_fixtures_preserve_physical_payload(self) -> None:
        for fixture in all_fixtures():
            with self.subTest(fixture=fixture.name):
                if fixture.record is None:
                    self.assertEqual(fixture.name, "E")
                    continue
                record = fixture.record
                self.assertIsInstance(record, PhysicalLabelRecord)
                self.assertEqual(record.momentum.variables, fixture.expected_variables)
                self.assertEqual(record.independence.rank, fixture.expected_rank)
                self.assertEqual(record.measure_cancellations, fixture.expected_cancellations)

    def test_duplicate_variables_are_not_collapsed(self) -> None:
        record = next(fixture.record for fixture in all_fixtures() if fixture.name == "D")
        assert record is not None
        self.assertEqual(record.momentum.variables, ((2, 3, 4), (2, 4), (2, 4)))
        self.assertEqual(record.independence.rank, 2)
        self.assertFalse(record.independence.independent)

    def test_parallel_single_occurrences_are_retained(self) -> None:
        record = next(fixture.record for fixture in all_fixtures() if fixture.name == "F")
        assert record is not None
        grouped = dict(record.single_groups)
        self.assertEqual(
            grouped[1],
            (("external", 4, 0), ("internal", 1, 0), ("internal", 4, 0)),
        )
        self.assertEqual(grouped[2], (("external", 5, 0), ("internal", 4, 1)))
        self.assertEqual(record.measure_cancellations, ((1, 2), (2, 1), (3, 0), (4, 0)))

    def test_to_dict_keeps_all_evidence_and_is_json_safe(self) -> None:
        record = next(fixture.record for fixture in all_fixtures() if fixture.name == "B")
        assert record is not None
        payload = record.to_dict()
        restored = json.loads(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(restored["schema_version"], "IM-4-v1")
        self.assertEqual(restored["momentum"]["double_momenta"][2]["coefficients"], [0, 1, 1, 0])
        self.assertEqual(restored["independence"]["labels"], ["s12", "s34", "s23"])
        self.assertEqual(len(restored["single_groups"]), 4)
        self.assertNotIn("source", restored)

    def test_provenance_is_separate_from_identity_payload(self) -> None:
        fixture = next(item for item in all_fixtures() if item.name == "B")
        assert fixture.record is not None
        first = Provenance({"source_file": "page-a", "figure": 3}, fixture.labeling, ("legacy", 1))
        second = Provenance({"source_file": "page-b", "figure": 8}, fixture.labeling, ("legacy", 2))
        self.assertNotEqual(first, second)
        self.assertEqual(fixture.record, fixture.record)
        self.assertNotIn("source", fixture.record.to_dict())
        self.assertEqual(first.to_dict()["raw_labeling"], fixture.labeling.to_dict())


class NormalizationRecordTests(unittest.TestCase):
    def test_stages_distinguish_dependent_from_rejected(self) -> None:
        fixtures = {fixture.name: fixture for fixture in all_fixtures()}
        accepted = NormalizationResult("accepted", fixtures["B"].record)
        dependent = NormalizationResult("dependent", fixtures["D"].record)
        rejected = NormalizationResult(
            "structure_rejected",
            issues=(ValidationIssue("outgoing_count", "The fixture violates the two-outgoing rule."),),
        )
        self.assertEqual(accepted.record.momentum.variables, ((1, 2), (3, 4), (2, 3)))
        self.assertEqual(dependent.record.independence.rank, 2)
        self.assertIsNone(rejected.record)
        self.assertEqual(rejected.to_dict()["issues"][0]["code"], "outgoing_count")

    def test_stage_vocabulary_is_closed(self) -> None:
        self.assertEqual(
            NORMALIZATION_STAGES,
            ("field_rejected", "structure_rejected", "momentum_rejected", "dependent", "accepted"),
        )
        with self.assertRaises(ValueError):
            NormalizationResult("unknown")


if __name__ == "__main__":
    unittest.main()
