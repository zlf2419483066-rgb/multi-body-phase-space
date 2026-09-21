"""Tests for the isolated, opt-in flow particle-orbit candidate."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phase_space_graph5 import CountMode, RunStatus  # noqa: E402
from phase_space_graph5 import flow_labeling as flow_module  # noqa: E402
from phase_space_graph5.compact_reporting import (  # noqa: E402
    _call_flow_provider,
    run_compact_candidates,
)
from phase_space_graph5.domain import (  # noqa: E402
    DoubleLineLabel,
    EdgeType,
    Labeling,
    ValidationIssue,
)
from phase_space_graph5.enumeration import EnumerationLimits, enumerate_topologies  # noqa: E402
from phase_space_graph5.label_identity import canonicalize_candidate_labeling  # noqa: E402
from phase_space_graph5.labeling import LabelingLimits  # noqa: E402
from phase_space_graph5.physical_labels import CandidateNormalizationResult  # noqa: E402


class FlowOrbitCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.l0 = enumerate_topologies(5, 0).topologies
        cls.l1 = enumerate_topologies(5, 1).topologies

    def test_default_reference_and_orbit_match_all_l0_l1_records_and_maps(self) -> None:
        for loop_count, topologies in ((0, self.l0), (1, self.l1)):
            for topology in topologies:
                reference = flow_module.enumerate_flow_labelings(
                    topology, record_order="flow"
                )
                orbit = flow_module.enumerate_flow_labelings(
                    topology, record_order="flow", use_particle_orbit=True
                )
                self.assertEqual(
                    [record.to_dict() for record in reference.candidate_records],
                    [record.to_dict() for record in orbit.candidate_records],
                    msg=f"record mismatch at L{loop_count} {topology.to_dict()}",
                )
                self.assertEqual(reference.raw_assignment_count, orbit.raw_assignment_count)
                self.assertEqual(reference.momentum_pass_count, orbit.momentum_pass_count)
                self.assertEqual(reference.flow_partition_normalized_labelings, reference.raw_assignment_count)
                self.assertEqual(orbit.flow_partition_normalized_labelings, 0)
                self.assertEqual(
                    orbit.certificate_transports,
                    orbit.momentum_pass_count,
                )
                for reference_record, orbit_record in zip(
                    reference.candidate_records, orbit.candidate_records
                ):
                    for mode in CountMode:
                        reference_identity = canonicalize_candidate_labeling(
                            reference_record, mode
                        )
                        orbit_identity = canonicalize_candidate_labeling(
                            orbit_record, mode
                        )
                        self.assertEqual(
                            reference_identity.to_dict(), orbit_identity.to_dict()
                        )

    def test_default_is_reference_and_counters_are_separate(self) -> None:
        topology = self.l0[0]
        implicit = flow_module.enumerate_flow_labelings(topology)
        explicit = flow_module.enumerate_flow_labelings(
            topology, use_particle_orbit=False
        )
        self.assertEqual(implicit.to_dict(), explicit.to_dict())
        orbit = flow_module.enumerate_flow_labelings(
            topology, use_particle_orbit=True
        )
        counts = orbit.to_dict()["counts"]
        self.assertEqual(counts["flow_partition_generated_labelings"], 120)
        self.assertEqual(counts["flow_partition_normalized_labelings"], 0)
        self.assertEqual(counts["representative_evaluations"], 1)
        self.assertEqual(counts["certificate_transports"], 120)
        self.assertTrue(counts["particle_orbit_reuse"])

    def test_cancel_deadline_limit_and_representative_failure(self) -> None:
        topology = self.l0[0]
        for kwargs, reason in (
            ({"cancel_requested": lambda: True}, "cancel_requested"),
            ({"deadline": time.monotonic() - 1}, "deadline_exceeded"),
            ({"limits": LabelingLimits(max_labelings=1)}, "limit_max_labelings"),
        ):
            reference = flow_module.enumerate_flow_labelings(topology, **kwargs)
            orbit = flow_module.enumerate_flow_labelings(
                topology, use_particle_orbit=True, **kwargs
            )
            self.assertEqual(reference.stop_reason, reason)
            self.assertEqual(orbit.stop_reason, reason)
            self.assertEqual(reference.raw_assignment_count, orbit.raw_assignment_count)
            self.assertEqual(reference.momentum_pass_count, orbit.momentum_pass_count)
        rejected = CandidateNormalizationResult(
            "momentum_rejected",
            issues=(ValidationIssue("injected_rejection", "injected representative"),),
        )
        with patch.object(
            flow_module,
            "_orbit_normalization",
            return_value=(rejected, True, False),
        ):
            failed = flow_module.enumerate_flow_labelings(
                topology, use_particle_orbit=True
            )
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(failed.stop_reason, "flow_orbit_rejected")
        self.assertEqual(failed.raw_assignment_count, 1)
        self.assertEqual(failed.flow_partition_normalized_labelings, 0)
        self.assertEqual(failed.representative_evaluations, 1)
        self.assertEqual(failed.certificate_transports, 0)
        self.assertEqual(failed.failure_count("injected_rejection"), 1)

    def test_orbit_rejects_incomplete_or_malformed_generated_labelings(self) -> None:
        topology = next(
            topology
            for topology in self.l1
            if any(edge.edge_type is EdgeType.SINGLE for edge in topology.edges)
        )
        valid = flow_module.enumerate_flow_labelings(
            topology, record_order="flow"
        ).candidate_records[0].complete_labeling
        first_double = valid.double_lines[0]
        replacement = next(
            particle
            for particle in range(1, topology.n + 1)
            if particle not in first_double.particle_ids
        )
        wrong_subset = DoubleLineLabel(
            first_double.edge_index,
            first_double.occurrence,
            tuple(sorted(first_double.particle_ids[1:] + (replacement,))),
        )
        variants = {
            "wrong_double_subset": Labeling(
                valid.final_particles,
                valid.single_lines,
                (wrong_subset,) + valid.double_lines[1:],
            ),
            "missing_double": Labeling(
                valid.final_particles,
                valid.single_lines,
                valid.double_lines[:-1],
            ),
            "duplicate_final": Labeling(
                valid.final_particles + (valid.final_particles[-1],),
                valid.single_lines,
                valid.double_lines,
            ),
            "duplicate_single": Labeling(
                valid.final_particles,
                valid.single_lines + (valid.single_lines[-1],),
                valid.double_lines,
            ),
            "extra_double_ref": Labeling(
                valid.final_particles,
                valid.single_lines,
                valid.double_lines
                + (DoubleLineLabel(first_double.edge_index, first_double.occurrence + 1, (1, 2)),),
            ),
        }
        for name, malformed in variants.items():
            with self.subTest(name=name):
                with patch.object(
                    flow_module,
                    "_iter_flow_labelings",
                    return_value=iter((malformed,)),
                ):
                    reference = flow_module.enumerate_flow_labelings(
                        topology, record_order="flow"
                    )
                with patch.object(
                    flow_module,
                    "_iter_flow_labelings",
                    return_value=iter((malformed,)),
                ):
                    orbit = flow_module.enumerate_flow_labelings(
                        topology, record_order="flow", use_particle_orbit=True
                    )
                self.assertEqual(orbit.raw_assignment_count, 1)
                self.assertFalse(orbit.complete)
                self.assertEqual(orbit.stop_reason, "flow_orbit_rejected")
                if name == "missing_double":
                    # The public normalizer permits omitted doubles and
                    # reconstructs them.  Orbit reuse additionally enforces
                    # the complete-labeling contract of its flow generator.
                    self.assertTrue(reference.complete)
                    self.assertEqual(
                        orbit.failure_count("flow_orbit_incomplete_double_occurrences"),
                        1,
                    )
                else:
                    self.assertFalse(reference.complete)
                    self.assertEqual(reference.stop_reason, "flow_normalization_rejected")
                    self.assertGreaterEqual(sum(dict(orbit.failure_counts).values()), 1)

    def test_compact_audit_accepts_orbit_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_compact_candidates(
                n=5,
                loop_count=0,
                max_seconds=30,
                topology_limits=EnumerationLimits(max_topologies=3),
                label_strategy="flow",
                flow_record_order="flow",
                flow_particle_orbit=True,
                output_dir=directory,
                write_report=False,
            )
            # The topology limit deliberately makes the compact run partial;
            # each committed per-topology audit row must still be complete.
            self.assertFalse(report.complete)
            self.assertEqual(report.stop_reason, "limit_max_topologies")
            self.assertTrue(report.to_dict()["flow_particle_orbit"])
            self.assertTrue(report.to_dict()["context"]["execution"]["flow_particle_orbit"])
            self.assertEqual(len(report.compact_audit), 3)
            for row in report.compact_audit:
                self.assertTrue(row["complete"])
                self.assertTrue(row["particle_orbit_reuse"])
                self.assertTrue(row["flow_particle_orbit"])
                self.assertIn("representative_evaluations", row["flow_counts"])
                self.assertIn("certificate_transports", row["flow_counts"])
                self.assertEqual(
                    row["flow_counts"]["flow_partition_normalized_labelings"], 0
                )
            progress = report.progress["progress"]
            self.assertEqual(progress["representative_evaluations"], 3)
            self.assertEqual(progress["certificate_transports"], 360)

    def test_custom_provider_cannot_silently_drop_requested_orbit(self) -> None:
        topology = self.l0[0]
        limits = LabelingLimits()

        def report_with(confirmed: object) -> SimpleNamespace:
            return SimpleNamespace(
                topology=topology,
                candidate_records=(),
                failure_counts=(),
                failure_samples=(),
                failure_witnesses=(),
                limits=limits,
                stop_reason=None,
                complete=True,
                status=RunStatus.COMPLETE,
                particle_orbit_reuse=confirmed,
            )

        def legacy_provider(topology, *, limits, cancel_requested, deadline, record_order):
            return report_with(False)

        with self.assertRaisesRegex(TypeError, "requires a flow provider accepting"):
            _call_flow_provider(
                legacy_provider,
                topology,
                limits=limits,
                cancel_requested=lambda: False,
                deadline=None,
                record_order="flow",
                flow_particle_orbit=True,
            )

        def kwargs_provider(topology, **kwargs):
            return report_with(False)

        with self.assertRaisesRegex(TypeError, "did not confirm particle_orbit_reuse"):
            _call_flow_provider(
                kwargs_provider,
                topology,
                limits=limits,
                cancel_requested=lambda: False,
                deadline=None,
                record_order="flow",
                flow_particle_orbit=True,
            )


if __name__ == "__main__":
    unittest.main()
