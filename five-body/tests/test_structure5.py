from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase_space_graph5 import (  # noqa: E402
    DiagramSpec,
    EdgeType,
    FinalParticleLabel,
    InternalEdge,
    Labeling,
    Topology,
    analyze_topology,
    canonicalize_labeling,
    canonicalize_labeling_reference,
    canonicalize_topology,
    canonicalize_topology_reference,
    fixture_a,
    fixture_b,
    issue_codes,
)


def _renumber_topology(topology: Topology, new_vertex_order: tuple[int, ...]) -> Topology:
    """Return the same topology after applying a complete vertex permutation."""

    mapping = dict(zip(topology.vertices, new_vertex_order))
    return Topology(
        n=topology.n,
        vertices=tuple(mapping[vertex] for vertex in topology.vertices),
        edges=tuple(
            InternalEdge(mapping[edge.source], mapping[edge.target], edge.edge_type, edge.multiplicity)
            for edge in topology.edges
        ),
        initial_vertex=mapping[topology.initial_vertex],
        final_slots={mapping[vertex]: count for vertex, count in topology.final_slots},
    )


class FiveBodyStructureTests(unittest.TestCase):
    def test_fixture_a_is_zero_loop_chain(self) -> None:
        topology, labeling = fixture_a()
        report = analyze_topology(topology)
        self.assertTrue(report.valid, report.to_dict())
        self.assertTrue(labeling.validate_fields(topology).valid)
        self.assertEqual((report.metrics.n, report.metrics.v, report.metrics.d, report.metrics.l), (5, 4, 3, 0))
        self.assertEqual(report.metrics.outdegree_by_vertex, ((0, 2), (1, 2), (2, 2), (3, 2)))
        self.assertEqual(report.metrics.final_slot_total, 5)

    def test_fixture_b_has_correct_external_positions_and_one_loop(self) -> None:
        topology, labeling = fixture_b(3)
        report = analyze_topology(topology)
        self.assertTrue(report.valid, report.to_dict())
        self.assertTrue(labeling.validate_fields(topology).valid)
        self.assertEqual(dict(topology.final_slots), {0: 0, 1: 1, 2: 1, 3: 1, 4: 2})
        self.assertEqual((report.metrics.v, report.metrics.d, report.metrics.l, report.metrics.L), (5, 4, 1, 1))
        self.assertEqual(labeling.single_lines[0].edge_index, 1)  # 0->2 follows 0->1 after edge normalization.

    def test_n5_structure_scope_does_not_reject_hand_built_high_loop_fixture(self) -> None:
        self.assertTrue(DiagramSpec(n=5, loop_count=5).validate_fields().valid)
        topology, _ = fixture_b()
        self.assertNotIn("supported_scope", analyze_topology(topology).failed_codes())

    def test_parallel_double_edges_fail_tree_check_by_multiplicity(self) -> None:
        topology = Topology(
            n=5,
            vertices=(0, 1, 2, 3),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE, multiplicity=2),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(2, 3, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 0, 1: 1, 2: 1, 3: 3},
        )
        report = analyze_topology(topology)
        self.assertFalse(report.valid)
        self.assertIn("double_subgraph_tree", report.failed_codes())
        self.assertEqual(report.metrics.double_cycle_rank, 1)

    def test_directed_cycle_is_rejected_even_when_counts_are_consistent(self) -> None:
        topology = Topology(
            n=5,
            vertices=(0, 1, 2, 3, 4),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(2, 3, EdgeType.DOUBLE),
                InternalEdge(3, 4, EdgeType.DOUBLE),
                InternalEdge(4, 1, EdgeType.SINGLE),
            ),
            initial_vertex=0,
            final_slots={0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        )
        report = analyze_topology(topology)
        self.assertFalse(report.valid)
        self.assertIn("directed_acyclic", report.failed_codes())

    def test_initial_reachability_is_checked_separately_from_dag(self) -> None:
        topology = Topology(
            n=5,
            vertices=(0, 1, 2, 3, 4),
            edges=(
                InternalEdge(1, 0, EdgeType.DOUBLE),
                InternalEdge(2, 1, EdgeType.DOUBLE),
                InternalEdge(3, 2, EdgeType.DOUBLE),
                InternalEdge(4, 3, EdgeType.DOUBLE),
                InternalEdge(1, 0, EdgeType.SINGLE),
            ),
            initial_vertex=0,
            final_slots={0: 2, 1: 0, 2: 1, 3: 1, 4: 1},
        )
        report = analyze_topology(topology)
        self.assertFalse(report.valid)
        self.assertIn("initial_reachability", report.failed_codes())
        self.assertNotIn("directed_acyclic", report.failed_codes())

    def test_outdegree_and_final_slot_total_failures_are_explicit(self) -> None:
        topology, _ = fixture_a()
        bad_slots = Topology(
            n=5,
            vertices=topology.vertices,
            edges=topology.edges,
            initial_vertex=topology.initial_vertex,
            final_slots={0: 2, 1: 1, 2: 1, 3: 1},
        )
        codes = set(analyze_topology(bad_slots).failed_codes())
        self.assertIn("outdegree_two", codes)
        self.assertNotIn("complete_expansion_counts", codes)
        wrong_total = Topology(
            n=5,
            vertices=topology.vertices,
            edges=topology.edges,
            initial_vertex=topology.initial_vertex,
            final_slots={0: 1, 1: 1, 2: 1, 3: 1},
        )
        self.assertIn("complete_expansion_counts", analyze_topology(wrong_total).failed_codes())

    def test_duplicate_particle_and_boolean_fields_are_rejected(self) -> None:
        topology, labeling = fixture_a()
        duplicate = Labeling(
            final_particles=tuple(
                FinalParticleLabel(item.vertex, item.slot, 4 if item.particle_id == 5 else item.particle_id)
                for item in labeling.final_particles
            )
        )
        codes = set(issue_codes(duplicate.validate_fields(topology)))
        self.assertIn("duplicate_final_particle_label", codes)
        self.assertIn("final_particle_bijection_mismatch", codes)
        with self.assertRaises(TypeError):
            InternalEdge(True, 1, EdgeType.DOUBLE)
        with self.assertRaises(TypeError):
            DiagramSpec(n=True)
        with self.assertRaises(TypeError):
            DiagramSpec(loop_count=True)
        bool_label = Labeling(final_particles=(FinalParticleLabel(0, 0, True),))
        self.assertIn("particle_id_out_of_range", issue_codes(bool_label.validate_fields(topology)))

    def test_duplicate_final_slot_vertex_is_reported(self) -> None:
        topology, _ = fixture_a()
        duplicate_slots = Topology(
            n=5,
            vertices=topology.vertices,
            edges=topology.edges,
            initial_vertex=topology.initial_vertex,
            final_slots=((0, 1), (0, 1), (1, 1), (2, 1), (3, 1)),
        )
        report = analyze_topology(duplicate_slots)
        self.assertFalse(report.field_report.valid)
        self.assertIn("duplicate_final_slot_vertex", report.field_report.error_codes())

    def test_structural_report_is_json_serializable(self) -> None:
        encoded = analyze_topology(fixture_b()[0]).to_json()
        self.assertTrue(json.loads(encoded)["valid"])

    def test_optimized_and_reference_topology_keys_match(self) -> None:
        topology, labeling = fixture_b(3)
        self.assertEqual(canonicalize_topology(topology).key, canonicalize_topology_reference(topology).key)
        self.assertEqual(canonicalize_labeling(topology, labeling).key, canonicalize_labeling_reference(topology, labeling).key)

    def test_canonical_topology_key_survives_non_contiguous_vertex_renumbering(self) -> None:
        topology, _ = fixture_a()
        rename = {0: 11, 1: 24, 2: 37, 3: 50}
        renumbered = Topology(
            n=5,
            vertices=tuple(rename[v] for v in topology.vertices),
            edges=tuple(InternalEdge(rename[e.source], rename[e.target], e.edge_type, e.multiplicity) for e in topology.edges),
            initial_vertex=rename[topology.initial_vertex],
            final_slots={rename[v]: count for v, count in topology.final_slots},
        )
        self.assertEqual(canonicalize_topology(topology).key, canonicalize_topology(renumbered).key)

    def test_optimized_and_reference_keys_match_for_all_a_and_b_vertex_permutations(self) -> None:
        for topology, expected_permutations in ((fixture_a()[0], 24), (fixture_b()[0], 120)):
            base_key = canonicalize_topology_reference(topology).key
            seen = 0
            for new_vertex_order in permutations(topology.vertices):
                renumbered = _renumber_topology(topology, new_vertex_order)
                optimized = canonicalize_topology(renumbered)
                reference = canonicalize_topology_reference(renumbered)
                self.assertEqual(optimized.key, reference.key)
                self.assertEqual(reference.key, base_key)
                seen += 1
            self.assertEqual(seen, expected_permutations)

    def test_labeling_field_validation_keeps_double_occurrence_identity(self) -> None:
        topology, _ = fixture_a()
        labeling = Labeling(
            final_particles=tuple(
                FinalParticleLabel(vertex, slot, particle)
                for vertex, slot, particle in ((0, 0, 1), (1, 0, 2), (2, 0, 3), (3, 0, 4), (3, 1, 5))
            ),
        )
        self.assertTrue(labeling.validate_fields(topology).valid)


if __name__ == "__main__":
    unittest.main()
