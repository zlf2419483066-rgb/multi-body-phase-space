from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase_space_graph5.domain import (
    CountMode,
    EdgeType,
    FinalParticleLabel,
    InternalEdge,
    Labeling,
    SingleLineLabel,
    Topology,
)
from phase_space_graph5.enumeration import enumerate_topologies
from phase_space_graph5.fixtures import fixture_a
from phase_space_graph5.label_identity import (
    align_topology,
    align_topology_reference,
    alignment_cache_info,
    canonicalize_candidate_labeling,
    clear_alignment_cache,
)
from phase_space_graph5.physical_labels import normalize_candidate_labeling


def _renumber(topology: Topology, offset: int) -> Topology:
    return Topology(
        topology.n,
        tuple(vertex + offset for vertex in topology.vertices),
        tuple(
            InternalEdge(edge.source + offset, edge.target + offset, edge.edge_type, edge.multiplicity)
            for edge in topology.edges
        ),
        topology.initial_vertex + offset,
        {vertex + offset: count for vertex, count in topology.final_slots},
    )


def _parallel_topology(split: bool) -> tuple[Topology, Labeling]:
    edges = [
        InternalEdge(0, 1, EdgeType.DOUBLE),
        InternalEdge(0, 2, EdgeType.DOUBLE),
        InternalEdge(2, 3, EdgeType.DOUBLE),
        InternalEdge(3, 4, EdgeType.DOUBLE),
        InternalEdge(4, 5, EdgeType.DOUBLE),
    ]
    if split:
        edges.extend((InternalEdge(1, 2, EdgeType.SINGLE), InternalEdge(1, 2, EdgeType.SINGLE)))
    else:
        edges.append(InternalEdge(1, 2, EdgeType.SINGLE, multiplicity=2))
    topology = Topology(
        5,
        (0, 1, 2, 3, 4, 5),
        tuple(edges),
        0,
        {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2},
    )
    single_indices = [
        index for index, edge in enumerate(topology.edges)
        if edge.edge_type is EdgeType.SINGLE
    ]
    single_lines = (
        (SingleLineLabel(single_indices[0], 0, 1), SingleLineLabel(single_indices[0], 1, 2))
        if not split
        else (SingleLineLabel(single_indices[0], 0, 1), SingleLineLabel(single_indices[1], 0, 2))
    )
    labeling = Labeling(
        final_particles=(
            FinalParticleLabel(2, 0, 1),
            FinalParticleLabel(3, 0, 2),
            FinalParticleLabel(4, 0, 3),
            FinalParticleLabel(5, 0, 4),
            FinalParticleLabel(5, 1, 5),
        ),
        single_lines=single_lines,
    )
    return topology, labeling


class AlignmentOptimizationTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_alignment_cache()

    def test_grouped_and_reference_alignment_match_for_all_l0_l1_topologies(self) -> None:
        total = 0
        for loop_count in (0, 1):
            report = enumerate_topologies(5, loop_count)
            for topology in report.topologies:
                reference = align_topology_reference(topology)
                grouped = align_topology(topology, use_cache=False)
                cached = align_topology(topology)
                self.assertEqual(grouped, reference)
                self.assertEqual(cached, reference)
                total += 1
        self.assertEqual(total, 72)

    def test_lru_is_bounded_and_repeated_exact_topology_hits(self) -> None:
        clear_alignment_cache()
        topologies = tuple(
            topology
            for loop_count in (0, 1)
            for topology in enumerate_topologies(5, loop_count).topologies
        )
        for topology in topologies:
            align_topology(topology)
        info = alignment_cache_info()
        self.assertEqual(info["maxsize"], 64)
        self.assertLessEqual(info["size"], 64)
        before = alignment_cache_info()
        align_topology(topologies[-1])
        after = alignment_cache_info()
        self.assertEqual(after["hits"], before["hits"] + 1)

    def test_noncontiguous_original_map_is_part_of_exact_cache_input(self) -> None:
        topology = enumerate_topologies(5, 1).topologies[0]
        renumbered = _renumber(topology, 10)
        clear_alignment_cache()
        original = align_topology(topology)
        renumbered_alignment = align_topology(renumbered)
        info = alignment_cache_info()
        self.assertEqual(info["misses"], 2)
        self.assertEqual(
            renumbered_alignment.vertex_map,
            tuple((vertex, vertex - 10) for vertex in renumbered.vertices),
        )
        self.assertNotEqual(original.vertex_map, renumbered_alignment.vertex_map)
        self.assertEqual(renumbered_alignment, align_topology_reference(renumbered))

    def test_edge_order_and_parallel_occurrence_keys_are_preserved(self) -> None:
        topology, _ = fixture_a()
        reordered = Topology(
            topology.n,
            topology.vertices,
            tuple(reversed(topology.edges)),
            topology.initial_vertex,
            dict(reversed(topology.final_slots)),
        )
        self.assertEqual(reordered, topology)
        self.assertEqual(align_topology(reordered), align_topology_reference(reordered))

        split_topology, split_labeling = _parallel_topology(split=True)
        compact_topology, compact_labeling = _parallel_topology(split=False)
        clear_alignment_cache()
        split_alignment = align_topology(split_topology)
        split_info = alignment_cache_info()
        compact_alignment = align_topology(compact_topology)
        compact_info = alignment_cache_info()
        self.assertEqual(split_alignment, align_topology_reference(split_topology))
        self.assertEqual(compact_alignment, align_topology_reference(compact_topology))
        self.assertEqual(compact_info["misses"], split_info["misses"] + 1)

        split_record = normalize_candidate_labeling(split_topology, split_labeling).record
        compact_record = normalize_candidate_labeling(compact_topology, compact_labeling).record
        self.assertIsNotNone(split_record)
        self.assertIsNotNone(compact_record)
        split_identity = canonicalize_candidate_labeling(split_record, CountMode.FIXED_LABELS)
        compact_identity = canonicalize_candidate_labeling(compact_record, CountMode.FIXED_LABELS)
        self.assertEqual(split_identity.key, compact_identity.key)
        self.assertEqual(
            {old for old, _ in split_identity.edge_occurrence_map},
            {(index, occurrence)
             for index, edge in enumerate(split_topology.edges)
             if edge.multiplicity > 0
             for occurrence in range(edge.multiplicity)},
        )

    def test_invalid_bool_fields_and_bool_int_flags_cannot_touch_cache(self) -> None:
        topology, _ = fixture_a()
        clear_alignment_cache()
        align_topology(topology)
        before = alignment_cache_info()
        invalid = Topology(
            topology.n,
            topology.vertices,
            topology.edges,
            topology.initial_vertex,
            {
                vertex: (False if vertex == topology.vertices[0] else count)
                for vertex, count in topology.final_slots
            },
        )
        with self.assertRaises(ValueError):
            align_topology(invalid)
        self.assertEqual(alignment_cache_info(), before)
        with self.assertRaises(TypeError):
            align_topology(topology, use_cache=1)
        with self.assertRaises(TypeError):
            align_topology(topology, use_grouping=0)
        self.assertEqual(alignment_cache_info(), before)


if __name__ == "__main__":
    unittest.main()
