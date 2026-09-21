"""Focused tests for the production five-body particle-orbit cache."""

from __future__ import annotations

import sys
from pathlib import Path
import time
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import phase_space_graph5.labeling as labeling_module
from phase_space_graph5.domain import EdgeType, InternalEdge, Topology
from phase_space_graph5.enumeration import enumerate_topologies
from phase_space_graph5.labeling import (
    LabelingLimits,
    _expanded_edges,
    _labeling_from_assignments,
    _slot_order,
    enumerate_labelings,
    evaluate_labeling,
)


class LabelingSymmetry5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        expected = (Path(__file__).resolve().parents[1] / "src" / "phase_space_graph5" / "labeling.py").resolve()
        cls.asserted_module_file = Path(labeling_module.__file__).resolve()
        if cls.asserted_module_file != expected:
            raise AssertionError(
                f"wrong labeling module: {cls.asserted_module_file} != {expected}"
            )
        report = enumerate_topologies(n=5, loop_count=1, use_pruning=True)
        if not report.complete or len(report.topologies) != 69:
            raise AssertionError("L1 topology fixture is incomplete")
        cls.topology = report.topologies[0]

    @staticmethod
    def _projection(report):
        payload = report.to_dict()
        payload.pop("representative_evaluations", None)
        return payload

    @staticmethod
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
                (InternalEdge(1, 2, EdgeType.SINGLE),
                 InternalEdge(1, 2, EdgeType.SINGLE))
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

    def test_reference_count_is_actual_evaluate_count(self) -> None:
        limits = LabelingLimits(max_labelings=30)
        reference = enumerate_labelings(
            self.topology, limits=limits, use_symmetry=False
        )
        optimized = enumerate_labelings(
            self.topology, limits=limits, use_symmetry=True
        )
        self.assertEqual(self._projection(reference), self._projection(optimized))
        self.assertEqual(reference.representative_evaluations, 30)
        self.assertEqual(reference.raw_assignment_count, 30)
        self.assertEqual(optimized.raw_assignment_count, 30)
        self.assertLess(optimized.representative_evaluations, 30)

    def test_limits_cancel_and_deadline_remain_incomplete(self) -> None:
        cases = (
            (LabelingLimits(max_labelings=7), None, "limit_max_labelings"),
            (LabelingLimits(max_results=3), None, "limit_max_results"),
            (LabelingLimits(max_labelings=0), None, "limit_zero_budget"),
            (None, lambda: True, "cancel_requested"),
            (None, None, "deadline_exceeded"),
        )
        for limits, cancel, expected_reason in cases:
            kwargs = {"limits": limits, "cancel_requested": cancel}
            if expected_reason == "deadline_exceeded":
                kwargs["deadline"] = time.monotonic() - 1
            reference = enumerate_labelings(
                self.topology, use_symmetry=False, **kwargs
            )
            optimized = enumerate_labelings(
                self.topology, use_symmetry=True, **kwargs
            )
            self.assertEqual(self._projection(reference), self._projection(optimized))
            self.assertFalse(optimized.complete)
            self.assertEqual(optimized.stop_reason, expected_reason)

    def test_public_evaluate_default_still_checks_momentum_certificate(self) -> None:
        topology = enumerate_topologies(n=5, loop_count=0, use_pruning=True).topologies[0]
        labeling = _labeling_from_assignments(
            topology,
            _slot_order(topology),
            (1, 2, 3, 4, 5),
            _expanded_edges(topology, EdgeType.SINGLE),
            (),
        )
        result = evaluate_labeling(topology, labeling)
        self.assertTrue(result.valid)
        self.assertIsNotNone(result.momentum)
        self.assertTrue(result.momentum.verified)
        self.assertTrue(
            all(
                all(component == 0 for component in residual)
                for _, residual in result.momentum.vertex_residuals
            )
        )

    def test_parallel_single_occurrence_forms_have_full_report_equivalence(self) -> None:
        for split in (False, True):
            topology = self._parallel_topology(split)
            reference = enumerate_labelings(topology, use_symmetry=False)
            optimized = enumerate_labelings(topology, use_symmetry=True)
            self.assertEqual(self._projection(reference), self._projection(optimized))
            self.assertTrue(optimized.complete)
            self.assertEqual(optimized.raw_assignment_count, 3000)
            self.assertEqual(reference.representative_evaluations, 3000)
            self.assertLess(optimized.representative_evaluations, 3000)


if __name__ == "__main__":
    unittest.main()
