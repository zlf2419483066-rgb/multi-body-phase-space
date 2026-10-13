from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.domain import CountMode  # noqa: E402
from phase_space_graph5.fixtures import fixture_b  # noqa: E402
from phase_space_graph5.independence.witness_sources import (  # noqa: E402
    WitnessSourceError,
    iter_verified_sources,
    iter_witness_rows,
    read_verified_sources,
    verify_witness_row,
)
from phase_space_graph5.label_identity import canonicalize_candidate_labeling  # noqa: E402
from phase_space_graph5.physical_labels import normalize_candidate_labeling  # noqa: E402


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


def _record_and_identity():
    topology, labeling = fixture_b(3)
    checked = normalize_candidate_labeling(topology, labeling)
    if checked.record is None:
        raise AssertionError(checked.to_dict())
    record = checked.record
    identity = canonicalize_candidate_labeling(record, CountMode.FIXED_LABELS)
    return record, identity


def _jsonl_row() -> tuple[dict[str, object], object, object]:
    record, identity = _record_and_identity()
    row: dict[str, object] = {
        "topology": record.topology.to_dict(),
        "key": _jsonable(identity.topology_key),
        "key_sha256": hashlib.sha256(_json_bytes(identity.topology_key, sort_keys=False)).hexdigest(),
        "normalized_record": record.to_dict(),
    }
    return row, record, identity


def _drawing_manifest_row() -> tuple[dict[str, object], object, object]:
    record, _ = _record_and_identity()
    relabeled = canonicalize_candidate_labeling(record, CountMode.PARTICLE_RELABELING)
    source_payload = record.to_dict()
    row: dict[str, object] = {
        "topology_key": _jsonable(relabeled.topology_key),
        "source_topology": record.topology.to_dict(),
        "source_record": source_payload,
        "source_record_digest": hashlib.sha256(_json_bytes(source_payload, sort_keys=False)).hexdigest(),
        "identity_mode": relabeled.mode.value,
        "identity_digest": relabeled.digest,
        "identity_key": _jsonable(relabeled.key),
        "canonical_record_digest": hashlib.sha256(
            _json_bytes(relabeled.canonical_record.to_dict(), sort_keys=False)
        ).hexdigest(),
    }
    return row, record, relabeled


class WitnessSourceTests(unittest.TestCase):
    def test_gzip_jsonl_rechecks_record_and_keeps_coverage_incomplete(self) -> None:
        row, _, identity = _jsonl_row()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "witness.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            raw_rows = iter_witness_rows(path)
            try:
                raw = next(raw_rows)
            finally:
                raw_rows.close()
            single = verify_witness_row(raw)
            result = read_verified_sources(
                path,
                expected_sha256=digest,
                expected_topology_keys=(identity.topology_key,),
                expected_full_keys=(identity.key,),
            )
        self.assertEqual(result.rows, 1)
        self.assertEqual(result.file.sha256, digest)
        self.assertEqual(result.records[0].mode, "fixed_labels")
        self.assertEqual(single.key, result.records[0].key)
        self.assertEqual(single.digest, result.records[0].digest)
        self.assertEqual(single.topology_key, result.records[0].topology_key)
        self.assertEqual(single.source_record.to_dict(), result.records[0].source_record.to_dict())
        self.assertFalse(result.coverage_complete)
        self.assertEqual(result.coverage_reason, "one_witness_per_topology_is_not_full_identity_coverage")

    def test_plain_jsonl_full_read_checks_expected_sha256(self) -> None:
        row, _, _ = _jsonl_row()
        payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plain.jsonl"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            rows = tuple(iter_witness_rows(path, expected_sha256=digest))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source_index, 0)
        self.assertEqual(rows[0].file_sha256, digest)

    def test_plain_jsonl_bom_crlf_and_multibuffer_full_read(self) -> None:
        row, _, _ = _jsonl_row()
        row["padding"] = "相空间" * 512
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        payload = b"\xef\xbb\xbf" + b"\r\n".join([line] * 24) + b"\r\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bom-crlf.jsonl"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            rows = tuple(iter_witness_rows(path, expected_sha256=digest))
        self.assertEqual(len(rows), 24)
        self.assertEqual([item.source_index for item in rows], list(range(24)))
        self.assertTrue(all(item.file_sha256 == digest for item in rows))

    def test_plain_jsonl_wrong_expected_sha256_is_rejected_at_eof(self) -> None:
        row, _, _ = _jsonl_row()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-sha.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(WitnessSourceError) as context:
                tuple(iter_witness_rows(path, expected_sha256="0" * 64))
        self.assertEqual(context.exception.code, "file_sha256_mismatch")

    def test_partial_plain_jsonl_read_does_not_claim_or_check_eof_binding(self) -> None:
        row, _, _ = _jsonl_row()
        payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n") * 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.jsonl"
            path.write_text(payload, encoding="utf-8")
            rows = iter_witness_rows(path, expected_sha256="0" * 64)
            first = next(rows)
            self.assertEqual(first.source_index, 0)
            self.assertEqual(first.file_sha256, "0" * 64)
            # A row carries the caller-supplied expectation, not proof that
            # the source was exhausted and matched it. Closing early skips
            # EOF validation and does not expose a complete file binding.
            rows.close()

    def test_drawing_manifest_source_record_rebuilds_fixed_source(self) -> None:
        row, _, relabeled = _drawing_manifest_row()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drawing_manifest.json"
            path.write_text(json.dumps({"schema_version": "fixture", "rows": [row]}), encoding="utf-8")
            result = read_verified_sources(path)
        self.assertEqual(result.rows, 1)
        self.assertEqual(result.records[0].mode, "fixed_labels")
        self.assertEqual(result.records[0].topology_key, relabeled.topology_key)
        self.assertFalse(result.coverage_complete)

    def test_outer_key_tampering_is_structured_error(self) -> None:
        row, _, _ = _jsonl_row()
        row["key"] = ["topology", 5, 999]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(WitnessSourceError) as context:
                tuple(iter_verified_sources(path))
        self.assertEqual(context.exception.code, "key_mismatch")
        self.assertEqual(context.exception.line, 1)

    def test_record_tampering_fails_review_before_identity(self) -> None:
        row, _, _ = _jsonl_row()
        damaged = deepcopy(row)
        momentum = damaged["normalized_record"]["momentum"]  # type: ignore[index]
        momentum["double_momenta"][0]["coefficients"][0] += 1  # type: ignore[index]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-record.jsonl"
            path.write_text(json.dumps(damaged) + "\n", encoding="utf-8")
            with self.assertRaises(WitnessSourceError) as context:
                tuple(iter_verified_sources(path))
        self.assertIn(context.exception.code, {"candidate_record_invalid", "candidate_record_review_failed"})

    def test_file_sha_mismatch_is_not_a_rejecting_candidate_result(self) -> None:
        row, _, _ = _jsonl_row()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sha.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(WitnessSourceError) as context:
                read_verified_sources(path, expected_sha256="0" * 64)
        self.assertEqual(context.exception.code, "file_sha256_mismatch")

    def test_real_sources_only_smoke_first_row(self) -> None:
        sources = (
            Path(r"D:\多体相空间算法数据\Demo案例\5体\appdata_psf-five-body\20260920-l4-staged-stream-v2\P_one_witness_each.jsonl.gz"),
            Path(r"D:\多体相空间算法数据\Demo案例\5体\runs\20260922-l5-content-audit-full-v1\worker-0.p_witness.jsonl"),
            Path(r"D:\多体相空间算法数据\Demo案例\5体\repo_outputs\overnight_20260919\drawing_l3_p\drawing_manifest.json"),
        )
        for path in sources:
            if not path.exists():
                continue
            rows = iter_witness_rows(path)
            try:
                first = next(rows)
            finally:
                rows.close()
            self.assertEqual(first.source_index, 0)
            self.assertIsInstance(first.raw, dict)
            verified_rows = iter_verified_sources(path)
            try:
                verified = next(verified_rows)
            finally:
                verified_rows.close()
            self.assertEqual(verified.source_index, 0)
            self.assertEqual(verified.mode, "fixed_labels")


if __name__ == "__main__":
    unittest.main()
