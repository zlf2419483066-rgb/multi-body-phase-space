from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.fixtures import fixture_a, fixture_b
from phase_space_graph5.label_identity import canonicalize_candidate_labeling
from phase_space_graph5.physical_labels import normalize_candidate_labeling
from phase_space_graph5.independence.invariant_sets import (
    InvariantSet,
    InvariantSetOccurrence,
    extract_invariant_set,
)
from phase_space_graph5.independence.sources import SourceError, UnsupportedSourceError, VerifiedSource, _source_digest, read_source


def _verified_fixture(which: str) -> VerifiedSource:
    topology, labeling = fixture_a() if which == "A" else fixture_b(2)
    checked = normalize_candidate_labeling(topology, labeling)
    if checked.record is None:
        raise AssertionError("fixture did not produce a momentum candidate")
    identity = canonicalize_candidate_labeling(checked.record, "fixed_labels")
    return VerifiedSource(identity, checked.record, f"fixture-{which}")


class InvariantSetExtractionTests(unittest.TestCase):
    def test_A_preserves_order_edge_refs_and_full_keys(self) -> None:
        source = _verified_fixture("A")
        result = extract_invariant_set(source)
        self.assertIsInstance(result, InvariantSet)
        self.assertEqual(result.mode, "fixed_labels")
        self.assertEqual(result.source_digest, _source_digest(source.source_record))
        self.assertEqual(result.identity_digest, source.digest)
        self.assertEqual(result.topology_key, source.topology_key)
        self.assertEqual(result.identity_key, source.key)
        expected = [
            (label.edge_index, label.occurrence, tuple(sorted(label.particle_ids)))
            for label in source.source_record.complete_labeling.double_lines
        ]
        actual = [
            (item.edge_index, item.edge_occurrence, item.particle_subset)
            for item in result.occurrences
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(result.unique_subsets, tuple(item[2] for item in expected))
        self.assertEqual(result.duplicate_groups, ())

    def test_B2_keeps_duplicate_occurrences_and_separate_unique_view(self) -> None:
        result = extract_invariant_set(_verified_fixture("B2"))
        self.assertGreaterEqual(len(result.occurrences), 2)
        self.assertEqual(len(result.unique_subsets), len(set(result.unique_subsets)))
        self.assertEqual(len(result.duplicate_groups), 1)
        group = result.duplicate_groups[0]
        self.assertGreaterEqual(len(group), 2)
        self.assertEqual(
            len({result.occurrences[index].particle_subset for index in group}), 1
        )
        self.assertEqual(
            tuple(index for index in group),
            tuple(sorted(index for index in group)),
        )

    def test_to_dict_is_detached_and_keeps_integer_boolean_shapes(self) -> None:
        result = extract_invariant_set(_verified_fixture("B2"))
        payload = result.to_dict()
        payload["occurrences"][0]["particle_subset"][0] = 99
        payload["unique_subsets"][0][0] = 99
        fresh = result.to_dict()
        self.assertNotEqual(fresh["occurrences"][0]["particle_subset"][0], 99)
        self.assertNotEqual(fresh["unique_subsets"][0][0], 99)
        self.assertIsInstance(fresh["occurrences"][0]["occurrence_index"], int)
        self.assertNotIsInstance(fresh["occurrences"][0]["occurrence_index"], bool)

    def test_public_contract_rejects_boolean_indexes_and_inconsistent_groups(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            InvariantSetOccurrence(True, 0, 0, (1, 2))
        result = extract_invariant_set(_verified_fixture("B2"))
        with self.assertRaises(ValueError):
            replace(result, unique_subsets=result.unique_subsets + ((1, 2),))
        with self.assertRaises(ValueError):
            replace(result, duplicate_groups=())
        with self.assertRaises((TypeError, ValueError)):
            replace(result, unique_subsets=((True, 2),) + result.unique_subsets[1:])
        with self.assertRaises((TypeError, ValueError)):
            replace(result, duplicate_groups=((True, result.duplicate_groups[0][1]),))

    def test_tampered_identity_maps_fail_at_verified_source_boundary(self) -> None:
        source = _verified_fixture("A")
        bad_identity = replace(source.identity, vertex_map=source.identity.vertex_map[:-1])
        with self.assertRaises(SourceError):
            VerifiedSource(bad_identity, source.source_record, source.source_kind)

    def test_particle_relabeling_is_not_transported(self) -> None:
        source = _verified_fixture("A")
        relabeled = canonicalize_candidate_labeling(source.source_record, "particle_relabeling")
        with self.assertRaises(UnsupportedSourceError):
            VerifiedSource(relabeled, source.source_record, "fixture-particle-relabeling")

    def test_existing_l0_fixed_catalog_roundtrips_without_enumeration(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "文档" / "验证记录" / "首批产物_2026-09-19"
            / "final_l0" / "catalog" / "fixed_labels"
        )
        batch = read_source(source_path)
        checked = 0
        for item in batch.records:
            extracted = extract_invariant_set(item)
            expected = [
                (label.edge_index, label.occurrence, tuple(sorted(label.particle_ids)))
                for label in item.source_record.complete_labeling.double_lines
            ]
            actual = [
                (row.edge_index, row.edge_occurrence, row.particle_subset)
                for row in extracted.occurrences
            ]
            self.assertEqual(actual, expected)
            checked += 1
        def sample(which: str) -> dict[str, object]:
            extracted = extract_invariant_set(_verified_fixture(which))
            return {
                "occurrence_count": len(extracted.occurrences),
                "unique_subset_count": len(extracted.unique_subsets),
                "duplicate_groups": [list(group) for group in extracted.duplicate_groups],
                "source_digest": extracted.source_digest,
                "identity_digest": extracted.identity_digest,
            }
        summary = {
            "source_path": str(source_path.resolve()),
            "source_kind": batch.snapshot.source_kind,
            "source_fingerprint": batch.snapshot.fingerprint,
            "source_content_sha256": batch.snapshot.content_sha256,
            "source_files": [item.to_dict() for item in batch.snapshot.files],
            "record_count": len(batch.records),
            "topology_count": len({item.topology_key for item in batch.records}),
            "coverage_complete": batch.coverage.complete,
            "extraction_checked_records": checked,
            "formal_status": "not_evaluated",
            "certificates_written": 0,
            "certificate_bytes": 0,
            "samples": {"A": sample("A"), "B2": sample("B2")},
        }
        log_path = Path(r"D:\多体相空间算法数据\Demo案例\5体\independence\validation_20260922\io\l0_invariant_sets_summary.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
