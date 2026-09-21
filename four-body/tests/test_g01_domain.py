import json
import unittest

from phase_space_graph import (
    CountMode,
    DiagramSpec,
    DoubleLineLabel,
    EdgeType,
    FinalParticleLabel,
    InternalEdge,
    Labeling,
    SingleLineLabel,
    Topology,
    issue_codes,
)


def f01_topology() -> Topology:
    """Four-body chain-tree fixture from the project acceptance checklist."""

    return Topology(
        n=4,
        vertices=(2, 0, 1),
        edges=(
            InternalEdge(1, 2, EdgeType.DOUBLE),
            InternalEdge(0, 1, EdgeType.DOUBLE),
        ),
        initial_vertex=0,
        final_slots={0: 1, 1: 1, 2: 2},
    )


class DiagramSpecTests(unittest.TestCase):
    def test_supported_four_and_three_body_scopes(self) -> None:
        for loop_count in range(4):
            report = DiagramSpec(n=4, loop_count=loop_count).validate_fields()
            self.assertTrue(report.valid, report.to_dict())
        for loop_count in range(2):
            report = DiagramSpec(n=3, loop_count=loop_count).validate_fields()
            self.assertTrue(report.valid, report.to_dict())

    def test_unsupported_scope_is_reported(self) -> None:
        self.assertIn("unsupported_n", issue_codes(DiagramSpec(n=5).validate_fields()))
        self.assertIn("unsupported_loop_count", issue_codes(DiagramSpec(n=4, loop_count=4).validate_fields()))
        self.assertIn("negative_loop_count", issue_codes(DiagramSpec(n=4, loop_count=-1).validate_fields()))

    def test_count_mode_is_normalized(self) -> None:
        spec = DiagramSpec(n=4, loop_count=0, count_mode="particle_relabeling")
        self.assertIs(spec.count_mode, CountMode.PARTICLE_RELABELING)
        self.assertEqual(spec.to_dict()["count_mode"], "particle_relabeling")


class TopologyTests(unittest.TestCase):
    def test_valid_fixture_has_stable_representation(self) -> None:
        topology = f01_topology()
        report = topology.validate_fields()
        self.assertTrue(report.valid, report.to_dict())
        self.assertEqual(topology.vertices, (0, 1, 2))
        self.assertEqual(topology.final_slots, ((0, 1), (1, 1), (2, 2)))
        self.assertEqual(topology.edge_count, 2)
        shuffled = Topology(
            n=4,
            vertices=(1, 2, 0),
            edges=tuple(reversed(topology.edges)),
            initial_vertex=0,
            final_slots=((2, 2), (0, 1), (1, 1)),
        )
        self.assertEqual(topology.to_dict(), shuffled.to_dict())
        self.assertEqual(topology.to_dict(), Topology.from_dict(topology.to_dict()).to_dict())

    def test_edge_order_is_canonical_and_parallel_multiplicity_is_preserved(self) -> None:
        topology = Topology(
            n=4,
            vertices=(0, 1),
            edges=(
                InternalEdge(1, 0, "single", 2),
                InternalEdge(0, 1, "double", 1),
                InternalEdge(0, 1, "single", 1),
            ),
            initial_vertex=0,
            final_slots={0: 2, 1: 2},
        )
        self.assertEqual(topology.edge_count, 4)
        self.assertEqual([edge.edge_type for edge in topology.edges], [EdgeType.DOUBLE, EdgeType.SINGLE, EdgeType.SINGLE])
        self.assertEqual(topology.to_json(), Topology.from_json(topology.to_json()).to_json())

    def test_field_errors_do_not_prevent_constructing_g03_counterexamples(self) -> None:
        topology = Topology(
            n=4,
            vertices=(0, 1),
            edges=(InternalEdge(0, 9, "single", 1), InternalEdge(0, 0, "single", 1)),
            initial_vertex=0,
            final_slots={0: 4},
        )
        codes = issue_codes(topology.validate_fields())
        self.assertIn("unknown_edge_vertex", codes)
        # The self-loop is intentionally not a G01 field error; G03 owns that rule.
        self.assertNotIn("self_loop", codes)

    def test_invalid_fields_are_explicit(self) -> None:
        topology = Topology(
            n=4,
            vertices=(0, 0, 1),
            edges=(InternalEdge(0, 1, "single", -1),),
            initial_vertex=9,
            final_slots={0: 3, 8: 1},
        )
        codes = issue_codes(topology.validate_fields())
        for code in (
            "duplicate_vertex_id",
            "unknown_initial_vertex",
            "negative_edge_multiplicity",
            "unknown_final_slot_vertex",
            "missing_final_slot_vertex",
        ):
            self.assertIn(code, codes)


class LabelingTests(unittest.TestCase):
    def test_final_particles_are_bijective_but_single_line_repetition_is_allowed(self) -> None:
        topology = Topology(
            n=4,
            vertices=(0, 1, 2, 3),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(2, 3, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.SINGLE),
            ),
            initial_vertex=0,
            final_slots={0: 0, 1: 1, 2: 1, 3: 2},
        )
        labeling = Labeling(
            final_particles=(
                FinalParticleLabel(1, 0, 4),
                FinalParticleLabel(2, 0, 3),
                FinalParticleLabel(3, 0, 1),
                FinalParticleLabel(3, 1, 2),
            ),
            # Topology normalizes the same-endpoint double edge before the
            # single edge, so the single edge is canonical index 2.
            single_lines=(SingleLineLabel(2, 0, 2),),
            double_lines=(
                DoubleLineLabel(0, 0, (1, 2, 3)),
                DoubleLineLabel(1, 0, (1, 2)),
                DoubleLineLabel(3, 0, (1, 2)),
            ),
        )
        report = labeling.validate_fields(topology)
        self.assertTrue(report.valid, report.to_dict())
        self.assertEqual(labeling.single_lines[0].particle_id, 2)

    def test_labeling_errors_distinguish_slot_and_particle_problems(self) -> None:
        topology = f01_topology()
        labeling = Labeling(
            final_particles=(
                FinalParticleLabel(0, 0, 1),
                FinalParticleLabel(0, 0, 2),
                FinalParticleLabel(1, 2, 3),
                FinalParticleLabel(2, 0, 9),
            ),
            double_lines=(DoubleLineLabel(0, 0, (1, 1)),),
        )
        codes = issue_codes(labeling.validate_fields(topology))
        for code in (
            "duplicate_final_slot_label",
            "slot_out_of_range",
            "particle_id_out_of_range",
            "duplicate_composite_particle",
        ):
            self.assertIn(code, codes)

    def test_json_roundtrip_keeps_duplicate_occurrences_visible(self) -> None:
        labeling = Labeling(
            single_lines=(
                SingleLineLabel(4, 1, 2),
                SingleLineLabel(4, 0, 2),
            ),
            double_lines=(DoubleLineLabel(3, 0, (3, 1)),),
        )
        data = labeling.to_dict()
        self.assertEqual(data["single_lines"][0]["occurrence"], 0)
        restored = Labeling.from_dict(json.loads(json.dumps(data)))
        self.assertEqual(restored.to_dict(), data)


if __name__ == "__main__":
    unittest.main()
