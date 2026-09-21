"""Equivalence and boundary tests for the optional scalar-flow labeling path."""

from __future__ import annotations

import sys
from pathlib import Path
import time
import unittest
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phase_space_graph5 import (  # noqa: E402
    FLOW_STRATEGY,
    CountMode,
    EdgeType,
    FinalParticleLabel,
    InternalEdge,
    RunStatus,
    Topology,
    enumerate_flow_labelings,
    enumerate_topologies,
)
import phase_space_graph5.flow_labeling as flow_module  # noqa: E402
from phase_space_graph5.flow_labeling import FlowLabelingReport  # noqa: E402
from phase_space_graph5.labeling import LabelingLimits, enumerate_labelings  # noqa: E402
from phase_space_graph5.physical_labels import (  # noqa: E402
    CandidateNormalizationResult,
    review_candidate_record,
)
from phase_space_graph5.domain import ValidationIssue  # noqa: E402


def _record_dicts(report: FlowLabelingReport | object) -> list[dict[str, object]]:
    return [record.to_dict() for record in report.candidate_records]


def _parallel_topology(split: bool) -> Topology:
    edges = [
        InternalEdge(0, 1, EdgeType.DOUBLE),
        InternalEdge(0, 2, EdgeType.DOUBLE),
        InternalEdge(2, 3, EdgeType.DOUBLE),
        InternalEdge(3, 4, EdgeType.DOUBLE),
        InternalEdge(4, 5, EdgeType.DOUBLE),
    ]
    if split:
        edges.extend(
            (
                InternalEdge(1, 2, EdgeType.SINGLE),
                InternalEdge(1, 2, EdgeType.SINGLE),
            )
        )
    else:
        edges.append(InternalEdge(1, 2, EdgeType.SINGLE, multiplicity=2))
    return Topology(
        5,
        (0, 1, 2, 3, 4, 5),
        tuple(edges),
        0,
        {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2},
    )


def _renumber(topology: Topology) -> Topology:
    rename = {vertex: (vertex + 1) * 11 for vertex in topology.vertices}
    return Topology(
        topology.n,
        tuple(rename[vertex] for vertex in topology.vertices),
        tuple(
            InternalEdge(
                rename[edge.source],
                rename[edge.target],
                edge.edge_type,
                edge.multiplicity,
            )
            for edge in topology.edges
        ),
        rename[topology.initial_vertex],
        {rename[vertex]: count for vertex, count in topology.final_slots},
    )


class FlowLabeling5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        l0 = enumerate_topologies(n=5, loop_count=0, use_pruning=True)
        l1 = enumerate_topologies(n=5, loop_count=1, use_pruning=True)
        if not l0.complete or len(l0.topologies) != 3:
            raise AssertionError("L0 fixture is incomplete")
        if not l1.complete or len(l1.topologies) != 69:
            raise AssertionError("L1 fixture is incomplete")
        cls.l0 = tuple(l0.topologies)
        cls.l1 = tuple(l1.topologies)

    def test_l0_l1_records_match_reference_in_legacy_order(self) -> None:
        for scope in (self.l0, self.l1):
            for topology in scope:
                with self.subTest(topology=topology.to_dict()):
                    reference = enumerate_labelings(
                        topology, CountMode.FIXED_LABELS, use_symmetry=False
                    )
                    flow = enumerate_flow_labelings(
                        topology,
                        CountMode.FIXED_LABELS,
                        record_order="reference",
                    )
                    self.assertEqual(flow.status, RunStatus.COMPLETE)
                    self.assertEqual(flow.strategy, FLOW_STRATEGY)
                    self.assertEqual(_record_dicts(flow), _record_dicts(reference))
                    self.assertEqual(flow.momentum_pass_count, reference.momentum_pass_count)
                    self.assertEqual(
                        flow.traditional_search_space_size,
                        reference.raw_assignment_count,
                    )
                    self.assertEqual(
                        flow.flow_partition_normalized_labelings,
                        flow.flow_partition_generated_labelings,
                    )
                    if flow.scalar_flow_legal:
                        self.assertEqual(
                            flow.flow_partition_generated_labelings,
                            reference.momentum_pass_count,
                        )
                    else:
                        self.assertEqual(flow.flow_partition_generated_labelings, 0)
                        self.assertEqual(flow.scalar_flow_rejection, "scalar_flow_rejected")

    def test_public_record_replay_keeps_certificates(self) -> None:
        for topology in self.l0:
            flow = enumerate_flow_labelings(topology, record_order="reference")
            for record in flow.candidate_records:
                with self.subTest(topology=topology.to_dict()):
                    checked = review_candidate_record(record)
                    self.assertEqual(checked.record.to_dict(), record.to_dict())

    def test_parallel_split_compact_and_renumbered_records_match(self) -> None:
        for split in (False, True):
            for topology in (_parallel_topology(split), _renumber(_parallel_topology(split))):
                with self.subTest(split=split, vertices=topology.vertices):
                    reference = enumerate_labelings(topology, use_symmetry=False)
                    flow = enumerate_flow_labelings(topology, record_order="reference")
                    self.assertEqual(_record_dicts(flow), _record_dicts(reference))
                    self.assertEqual(
                        flow.flow_partition_generated_labelings,
                        reference.momentum_pass_count,
                    )
                    self.assertEqual(flow.flow_partition_normalized_labelings, len(flow.candidate_records))
                    self.assertGreater(flow.flow_partition_branch_count, flow.flow_partition_generated_labelings)

    def test_counts_are_separate_from_legacy_search_space(self) -> None:
        flow = next(
            candidate
            for topology in self.l1
            for candidate in (enumerate_flow_labelings(topology),)
            if candidate.scalar_flow_legal
            and candidate.traditional_search_space_size
            > candidate.flow_partition_generated_labelings
        )
        self.assertGreater(flow.traditional_search_space_size, flow.flow_partition_generated_labelings)
        payload = flow.to_dict()
        counts = payload["counts"]
        self.assertEqual(counts["traditional_search_space_size"], flow.traditional_search_space_size)
        self.assertEqual(
            counts["flow_partition_generated_labelings"],
            flow.flow_partition_generated_labelings,
        )
        self.assertEqual(
            counts["flow_partition_normalized_labelings"],
            flow.flow_partition_normalized_labelings,
        )
        self.assertEqual(counts["flow_partition_branch_count"], flow.flow_partition_branch_count)

    def test_limits_cancel_and_deadline_report_actual_flow_visits(self) -> None:
        topology = self.l0[0]
        limited = enumerate_flow_labelings(
            topology, limits=LabelingLimits(max_labelings=1)
        )
        self.assertEqual(limited.status, RunStatus.INCOMPLETE)
        self.assertEqual(limited.stop_reason, "limit_max_labelings")
        self.assertEqual(limited.flow_partition_generated_labelings, 1)
        self.assertEqual(limited.flow_partition_normalized_labelings, 1)

        result_limited = enumerate_flow_labelings(
            topology, limits=LabelingLimits(max_results=1)
        )
        self.assertEqual(result_limited.stop_reason, "limit_max_results")
        self.assertEqual(result_limited.flow_partition_generated_labelings, 1)
        self.assertEqual(result_limited.momentum_pass_count, 1)

        zero = enumerate_flow_labelings(topology, limits=LabelingLimits(max_labelings=0))
        self.assertEqual(zero.stop_reason, "limit_zero_budget")
        self.assertEqual(zero.flow_partition_generated_labelings, 0)

        cancelled = enumerate_flow_labelings(topology, cancel_requested=lambda: True)
        self.assertEqual(cancelled.stop_reason, "cancel_requested")
        self.assertEqual(cancelled.flow_partition_generated_labelings, 0)

        expired = enumerate_flow_labelings(
            topology, deadline=time.monotonic() - 1
        )
        self.assertEqual(expired.stop_reason, "deadline_exceeded")
        self.assertEqual(expired.flow_partition_generated_labelings, 0)

    def test_scalar_rejection_is_complete_zero_result(self) -> None:
        topology = next(
            topology
            for topology in self.l1
            if not enumerate_flow_labelings(topology).scalar_flow_legal
        )
        flow = enumerate_flow_labelings(topology)
        self.assertEqual(flow.status, RunStatus.COMPLETE)
        self.assertTrue(flow.complete)
        self.assertFalse(flow.scalar_flow_legal)
        self.assertEqual(flow.scalar_flow_rejection, "scalar_flow_rejected")
        self.assertEqual(flow.flow_partition_generated_labelings, 0)
        self.assertEqual(flow.flow_partition_normalized_labelings, 0)
        self.assertEqual(flow.momentum_pass_count, 0)
        self.assertEqual(flow.candidate_records, ())

    def test_scalar_rejection_still_honors_preflight_stops(self) -> None:
        topology = next(
            topology
            for topology in self.l1
            if not enumerate_flow_labelings(topology).scalar_flow_legal
        )
        cancelled = enumerate_flow_labelings(topology, cancel_requested=lambda: True)
        self.assertEqual(cancelled.status, RunStatus.INCOMPLETE)
        self.assertEqual(cancelled.stop_reason, "cancel_requested")
        expired = enumerate_flow_labelings(topology, deadline=time.monotonic() - 1)
        self.assertEqual(expired.status, RunStatus.INCOMPLETE)
        self.assertEqual(expired.stop_reason, "deadline_exceeded")
        zero = enumerate_flow_labelings(topology, limits=LabelingLimits(max_results=0))
        self.assertEqual(zero.status, RunStatus.INCOMPLETE)
        self.assertEqual(zero.stop_reason, "limit_zero_budget")

    def test_invalid_deadline_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            enumerate_flow_labelings(self.l0[0], deadline=True)
        with self.assertRaises(ValueError):
            enumerate_flow_labelings(self.l0[0], deadline=float("nan"))

    def test_explicit_and_limit_deadlines_use_the_earliest(self) -> None:
        topology = self.l0[0]
        limits_deadline = time.monotonic() - 1
        report = enumerate_flow_labelings(
            topology,
            limits=LabelingLimits(deadline=limits_deadline),
            deadline=time.monotonic() + 60,
        )
        self.assertEqual(report.status, RunStatus.INCOMPLETE)
        self.assertEqual(report.stop_reason, "deadline_exceeded")
        self.assertEqual(report.flow_partition_generated_labelings, 0)

        explicit_deadline = time.monotonic() - 1
        report = enumerate_flow_labelings(
            topology,
            limits=LabelingLimits(deadline=time.monotonic() + 60),
            deadline=explicit_deadline,
        )
        self.assertEqual(report.status, RunStatus.INCOMPLETE)
        self.assertEqual(report.stop_reason, "deadline_exceeded")
        self.assertEqual(report.flow_partition_generated_labelings, 0)

    def test_normalization_rejection_is_failed_not_filtered(self) -> None:
        injected = CandidateNormalizationResult(
            "momentum_rejected",
            issues=(ValidationIssue("injected_rejection", "injected rejection"),),
        )
        with patch.object(flow_module, "normalize_candidate_labeling", return_value=injected):
            report = enumerate_flow_labelings(self.l0[0])
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertFalse(report.complete)
        self.assertEqual(report.stop_reason, "flow_normalization_rejected")
        self.assertEqual(report.failure_count("injected_rejection"), 1)
        self.assertEqual(report.flow_partition_generated_labelings, 1)

    def test_legal_empty_partition_generation_is_failed(self) -> None:
        with patch.object(flow_module, "_iter_flow_labelings", return_value=iter(())):
            report = enumerate_flow_labelings(self.l0[0])
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertFalse(report.complete)
        self.assertEqual(report.stop_reason, "flow_partition_empty")
        self.assertEqual(report.flow_partition_generated_labelings, 0)

    def test_invalid_strategy_and_invalid_structure_are_explicit(self) -> None:
        with self.assertRaises(ValueError):
            enumerate_flow_labelings(self.l0[0], strategy="legacy")

        cyclic = Topology(
            5,
            (0, 1, 2, 3),
            (
                InternalEdge(0, 1, EdgeType.DOUBLE),
                InternalEdge(1, 0, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
            ),
            0,
            {0: 1, 1: 1, 2: 1, 3: 2},
        )
        flow = enumerate_flow_labelings(cyclic)
        self.assertEqual(flow.status, RunStatus.FAILED)
        self.assertFalse(flow.complete)
        self.assertEqual(flow.flow_partition_generated_labelings, 0)


if __name__ == "__main__":
    unittest.main()
