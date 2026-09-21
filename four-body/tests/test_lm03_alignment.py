import unittest

from im_fixtures import fixture_b, fixture_d, fixture_f
from phase_space_graph import (
    DoubleLineLabel,
    EdgeType,
    InternalEdge,
    Labeling,
    SingleLineLabel,
    Topology,
    align_topology,
    normalize_physical_labeling,
    relabel_physical_record,
)


class LM03AlignmentTests(unittest.TestCase):
    def test_noncontiguous_vertex_names_have_the_same_g04_key(self) -> None:
        base = fixture_b().topology
        renamed = Topology(
            n=4,
            vertices=(40, 7, 19, 88),
            edges=(
                InternalEdge(40, 19, EdgeType.DOUBLE),
                InternalEdge(7, 19, EdgeType.SINGLE),
                InternalEdge(40, 7, EdgeType.DOUBLE),
                InternalEdge(19, 88, EdgeType.DOUBLE),
            ),
            initial_vertex=40,
            final_slots=((88, 2), (19, 1), (7, 1), (40, 0)),
        )
        first = align_topology(base)
        second = align_topology(renamed)
        self.assertEqual(first.topology_key, second.topology_key)
        self.assertEqual(first.canonical_topology, second.canonical_topology)

    def test_vertex_and_particle_maps_rebuild_all_references(self) -> None:
        fixture = fixture_b()
        result = relabel_physical_record(
            fixture.record,
            vertex_map={0: 0, 1: 2, 2: 1, 3: 3},
            particle_map={1: 2, 2: 4, 3: 1, 4: 3},
        )
        self.assertEqual(result.normalization.stage, "accepted")
        self.assertEqual(
            {item.particle_id for item in result.record.complete_labeling.final_particles},
            {1, 2, 3, 4},
        )
        self.assertEqual(len(result.edge_occurrence_map), 4)
        self.assertEqual(len(result.final_slot_map), 4)
        self.assertTrue(result.record.momentum.verified)
        self.assertTrue(result.record.independence.independent)

    def test_dependent_record_stays_dependent_after_whole_graph_map(self) -> None:
        fixture = fixture_d()
        result = relabel_physical_record(fixture.record, {0: 0, 1: 2, 2: 1, 3: 3})
        self.assertEqual(result.normalization.stage, "dependent")
        self.assertFalse(result.record.independence.independent)
        self.assertEqual(result.record.independence.rank, fixture.record.independence.rank)

    def test_compact_and_split_parallel_single_edges_align_to_same_topology(self) -> None:
        compact = fixture_f()
        split_topology = Topology(
            n=4,
            vertices=compact.topology.vertices,
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(1, 3, EdgeType.DOUBLE),
                InternalEdge(0, 1, EdgeType.SINGLE),
                InternalEdge(2, 3, EdgeType.SINGLE),
                InternalEdge(2, 3, EdgeType.SINGLE),
                InternalEdge(3, 4, EdgeType.DOUBLE),
                InternalEdge(3, 5, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots=dict(compact.topology.final_slots),
        )
        split_labeling = Labeling(
            compact.labeling.final_particles,
            (
                SingleLineLabel(1, 0, 1),
                SingleLineLabel(4, 0, 1),
                SingleLineLabel(5, 0, 2),
            ),
            (
                DoubleLineLabel(0, 0, (2, 3, 4)),
                DoubleLineLabel(2, 0, (1, 2)),
                DoubleLineLabel(3, 0, (3, 4)),
                DoubleLineLabel(6, 0, (1, 3)),
                DoubleLineLabel(7, 0, (2, 4)),
            ),
        )
        split_result = normalize_physical_labeling(split_topology, split_labeling)
        self.assertEqual(split_result.stage, "accepted")
        self.assertEqual(
            align_topology(compact.topology).topology_key,
            align_topology(split_topology).topology_key,
        )
        mapped = relabel_physical_record(split_result.record, {v: v for v in split_topology.vertices})
        self.assertEqual(mapped.normalization.stage, "accepted")
        self.assertEqual(len(mapped.edge_occurrence_map), 8)

    def test_branching_tree_keeps_both_root_fixed_automorphisms(self) -> None:
        topology = Topology(
            n=4,
            vertices=(0, 1, 2),
            edges=(InternalEdge(0, 1, EdgeType.DOUBLE), InternalEdge(0, 2, EdgeType.DOUBLE)),
            initial_vertex=0,
            final_slots={0: 0, 1: 2, 2: 2},
        )
        alignment = align_topology(topology)
        self.assertIn((0, 1, 2), alignment.automorphisms)
        self.assertIn((0, 2, 1), alignment.automorphisms)
        self.assertEqual(len(alignment.automorphisms), 2)

    def test_incomplete_or_nonbijective_maps_are_rejected(self) -> None:
        record = fixture_b().record
        with self.assertRaises(ValueError):
            relabel_physical_record(record, {0: 0})
        with self.assertRaises(ValueError):
            relabel_physical_record(record, {0: 0, 1: 1, 2: 1, 3: 3})
        with self.assertRaises(ValueError):
            relabel_physical_record(record, {0: 0, 1: 1, 2: 2, 3: 3}, {1: 1, 2: 1, 3: 3, 4: 4})


if __name__ == "__main__":
    unittest.main()
