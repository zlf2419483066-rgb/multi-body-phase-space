from __future__ import annotations

from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys
from unittest.mock import patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.fixtures import fixture_a, fixture_b  # noqa: E402
from phase_space_graph5.domain import EdgeType, InternalEdge, Topology  # noqa: E402
from phase_space_graph5.labeling import LabelingResult, MomentumCertificate  # noqa: E402
import phase_space_graph5.independence.generic_search as generic_search_module  # noqa: E402
from phase_space_graph5.independence.generic_search import (  # noqa: E402
    GENERIC_SEARCH_COVERAGE_RULE,
    GENERIC_SEARCH_SCHEMA_VERSION,
    GENERIC_SEARCH_THEOREM_VERSION,
    _OPTIMIZED_SUBSET_Z_ROWS,
    _REPLAY_PAIR_SUBSET_Z_ROWS,
    search_generic_topology,
    verify_generic_search,
)
from phase_space_graph5.independence.generic_verification import _pair_rows, _subset_row  # noqa: E402
from phase_space_graph5.independence.generic_family import evaluate_generic_label  # noqa: E402
from phase_space_graph5.label_identity import canonicalize_candidate_labeling  # noqa: E402
from phase_space_graph5.physical_labels import normalize_candidate_labeling  # noqa: E402
from phase_space_graph5.independence.sources import VerifiedSource  # noqa: E402


def _source_for(which: int) -> VerifiedSource:
    topology, labeling = fixture_b(which)
    normalized = normalize_candidate_labeling(topology, labeling)
    if normalized.record is None:
        raise AssertionError(f"fixture B{which} did not produce a momentum record")
    identity = canonicalize_candidate_labeling(normalized.record, "fixed_labels")
    return VerifiedSource(identity, normalized.record, f"fixture-B{which}")


def _portable_real_reject_topology() -> Topology:
    """P0099 from the L2 pilot, kept independent of the D: pilot archive."""

    return Topology(
        n=5,
        vertices=tuple(range(6)),
        edges=(
            InternalEdge(0, 3, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.SINGLE),
            InternalEdge(2, 4, EdgeType.DOUBLE),
            InternalEdge(2, 4, EdgeType.SINGLE),
            InternalEdge(3, 1, EdgeType.DOUBLE),
            InternalEdge(4, 5, EdgeType.DOUBLE),
        ),
        initial_vertex=0,
        final_slots={0: 1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2},
    )


class GenericSearchTests(unittest.TestCase):
    def test_A_L0_is_one_assignment_and_verifies(self) -> None:
        topology, _ = fixture_a()
        result = search_generic_topology(topology, max_seconds=10.0)
        self.assertEqual(result["schema_version"], GENERIC_SEARCH_SCHEMA_VERSION)
        self.assertEqual(result["coverage_theorem_version"], GENERIC_SEARCH_THEOREM_VERSION)
        self.assertEqual(result["coverage_rule"], GENERIC_SEARCH_COVERAGE_RULE)
        self.assertEqual(result["L"], 0)
        self.assertEqual(result["expected"], 1)
        self.assertEqual(result["attempted"], 1)
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["complete"])
        report = verify_generic_search(topology, result)
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["status"], "pass")

    def test_same_topology_B2_rejection_does_not_hide_B3_pass(self) -> None:
        topology, _ = fixture_b(2)
        b2 = evaluate_generic_label(_source_for(2))
        b3 = evaluate_generic_label(_source_for(3))
        self.assertEqual(b2["status"], "reject")
        self.assertEqual(b3["status"], "pass")

        result = search_generic_topology(topology, max_seconds=10.0)
        self.assertEqual(result["expected"], 5)
        self.assertEqual(result["status"], "pass")
        self.assertGreaterEqual(result["attempted"], 2)
        witness = result["witness"]
        self.assertIsInstance(witness, dict)
        self.assertIn(3, witness["assignment"])
        report = verify_generic_search(topology, result)
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["status"], "pass")

    def test_zero_budget_is_unknown_and_cannot_be_relabelled_reject(self) -> None:
        topology, _ = fixture_b(2)
        result = search_generic_topology(topology, max_seconds=0.0)
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["complete"])
        self.assertEqual(result["expected"], 5)
        self.assertEqual(result["attempted"], 0)
        forged = deepcopy(result)
        forged["status"] = "reject"
        forged["complete"] = True
        report = verify_generic_search(topology, forged)
        self.assertFalse(report["valid"])

    def test_forced_assignment_budget_cannot_claim_reject(self) -> None:
        topology, _ = fixture_b(2)
        result = search_generic_topology(topology, max_seconds=10.0, max_assignments=1)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["attempted"], 1)
        self.assertFalse(result["complete"])
        forged = deepcopy(result)
        forged["status"] = "reject"
        forged["complete"] = True
        forged["attempted"] = forged["expected"]
        forged["stream_sha256"] = "0" * 64
        report = verify_generic_search(topology, forged)
        self.assertFalse(report["valid"])

    def test_field_valid_but_structure_invalid_is_unknown(self) -> None:
        topology = Topology(
            n=5,
            vertices=(0, 1, 2, 3),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(0, 2, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 1, 1: 1, 2: 1, 3: 2},
        )
        result = search_generic_topology(topology, max_seconds=10.0)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "invalid_input")
        report = verify_generic_search(topology, result)
        self.assertFalse(report["valid"])

    def test_pass_witness_and_stream_are_bound(self) -> None:
        topology, _ = fixture_b(2)
        result = search_generic_topology(topology, max_seconds=10.0)
        self.assertEqual(result["status"], "pass")
        valid = verify_generic_search(topology, result)
        self.assertTrue(valid["valid"], valid)

        tampered = deepcopy(result)
        tampered["stream_sha256"] = "0" * 64
        self.assertFalse(verify_generic_search(topology, tampered)["valid"])

        tampered = deepcopy(result)
        tampered["witness"]["assignment"] = [5]
        self.assertFalse(verify_generic_search(topology, tampered)["valid"])

        tampered = deepcopy(result)
        tampered["witness"]["assignment"] = [True]
        self.assertFalse(verify_generic_search(topology, tampered)["valid"])

        tampered = deepcopy(result)
        tampered["witness"]["assignment"] = [[3]]
        self.assertFalse(verify_generic_search(topology, tampered)["valid"])

    def test_portable_real_reject_json_roundtrip_and_count_tamper(self) -> None:
        topology = _portable_real_reject_topology()
        result = search_generic_topology(topology, max_seconds=10.0)
        self.assertEqual(result["status"], "reject")
        self.assertTrue(result["complete"])
        self.assertEqual(result["expected"], 25)
        self.assertEqual(result["attempted"], 25)
        self.assertEqual(result["momentum_pass"], 9)
        self.assertEqual(result["momentum_reject"], 16)
        self.assertEqual(result["independent_reject"], 9)

        decoded = json.loads(json.dumps(result, ensure_ascii=False, sort_keys=True))
        report = verify_generic_search(topology, decoded, max_seconds=10.0)
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["status"], "reject")

        for field, value in (("attempted", 24), ("momentum_pass", 8), ("independent_reject", 8)):
            tampered = deepcopy(decoded)
            tampered[field] = value
            checked = verify_generic_search(topology, tampered, max_seconds=10.0)
            self.assertFalse(checked["valid"], (field, checked))

    def test_inconsistent_labeling_results_are_unknown_in_producer_and_replay(self) -> None:
        topology, labeling = fixture_b(2)
        baseline = search_generic_topology(topology, max_seconds=10.0)
        self.assertEqual(baseline["status"], "pass")
        malformed = (
            LabelingResult(labeling, True, (), (), None),
            LabelingResult(labeling, True, (), (), MomentumCertificate((), (), False)),
            LabelingResult(labeling, False, (), (), None),
            LabelingResult(labeling, False, ("momentum_failure",), ("failure",), None),
            LabelingResult(labeling, False, ("momentum_failure",), ("failure",), MomentumCertificate((), (), True)),
        )
        for evaluation in malformed:
            with self.subTest(valid=evaluation.valid, momentum=evaluation.momentum):
                with patch.object(generic_search_module, "evaluate_labeling", return_value=evaluation):
                    produced = search_generic_topology(topology, max_seconds=10.0)
                self.assertEqual(produced["status"], "unknown")
                self.assertEqual(produced["reason"], "evaluation_error")
                self.assertEqual(produced["attempted"], 1)

                with patch.object(generic_search_module, "evaluate_labeling", return_value=evaluation):
                    replayed = verify_generic_search(topology, baseline, max_seconds=10.0)
                self.assertFalse(replayed["valid"], replayed)

    def test_all_25_fast_rows_match_independent_pair_formula(self) -> None:
        pairs = _pair_rows()
        expected_subsets = tuple(
            subset
            for size in (2, 3, 4)
            for subset in itertools.combinations(range(1, 6), size)
        )
        self.assertEqual(len(expected_subsets), 25)
        self.assertEqual(set(_OPTIMIZED_SUBSET_Z_ROWS), set(expected_subsets))
        self.assertEqual(set(_REPLAY_PAIR_SUBSET_Z_ROWS), set(expected_subsets))
        for subset in expected_subsets:
            expected = tuple(_subset_row(subset, pairs)[:9])
            self.assertEqual(_OPTIMIZED_SUBSET_Z_ROWS[subset], expected, subset)
            self.assertEqual(_REPLAY_PAIR_SUBSET_Z_ROWS[subset], expected, subset)

    def test_optimized_backend_matches_reference_on_L0_L1_and_real_L2_reject(self) -> None:
        topologies = [fixture_a()[0], fixture_b(2)[0], _portable_real_reject_topology()]
        for topology in topologies:
            with self.subTest(topology=topology.to_dict()):
                reference = search_generic_topology(topology, max_seconds=10.0, backend="reference")
                optimized = search_generic_topology(topology, max_seconds=10.0, backend="optimized")
                self.assertEqual(optimized, reference)
                self.assertEqual(
                    verify_generic_search(topology, optimized, max_seconds=10.0, backend="optimized"),
                    verify_generic_search(topology, reference, max_seconds=10.0, backend="reference"),
                )

    def test_optimized_verifier_rejects_corrupted_producer_subset_row(self) -> None:
        topology, _ = fixture_a()
        optimized_before = dict(generic_search_module._OPTIMIZED_SUBSET_Z_ROWS)
        replay_before = dict(_REPLAY_PAIR_SUBSET_Z_ROWS)

        baseline = search_generic_topology(topology, max_seconds=10.0, backend="optimized")
        self.assertEqual(baseline["status"], "pass")
        self.assertTrue(verify_generic_search(topology, baseline, max_seconds=10.0, backend="optimized")["valid"])

        first_double = baseline["witness"]["record"]["complete_labeling"]["double_lines"][0]
        subset = tuple(sorted(first_double["particle_ids"]))
        self.assertIn(subset, optimized_before)
        corrupted_rows = dict(optimized_before)
        corrupted_rows[subset] = (0,) * len(optimized_before[subset])
        self.assertNotEqual(corrupted_rows[subset], replay_before[subset])

        with patch.object(generic_search_module, "_OPTIMIZED_SUBSET_Z_ROWS", corrupted_rows):
            self.assertEqual(_REPLAY_PAIR_SUBSET_Z_ROWS, replay_before)
            fast_result = search_generic_topology(topology, max_seconds=10.0, backend="optimized")
            self.assertEqual(fast_result["status"], "reject")
            self.assertTrue(fast_result["complete"])

            replay_report = verify_generic_search(topology, fast_result, max_seconds=10.0, backend="optimized")
            self.assertFalse(replay_report["valid"], replay_report)
            self.assertEqual(replay_report["status"], "unknown")
            self.assertEqual(_REPLAY_PAIR_SUBSET_Z_ROWS, replay_before)

        self.assertEqual(generic_search_module._OPTIMIZED_SUBSET_Z_ROWS, optimized_before)
        self.assertEqual(_REPLAY_PAIR_SUBSET_Z_ROWS, replay_before)


if __name__ == "__main__":
    unittest.main()
