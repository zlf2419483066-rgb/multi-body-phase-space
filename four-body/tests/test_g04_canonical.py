import json
import unittest

from phase_space_graph import (
    CountMode,
    DoubleLineLabel,
    EdgeType,
    FinalParticleLabel,
    InternalEdge,
    Labeling,
    SingleLineLabel,
    Topology,
    canonicalize_labeling,
    canonicalize_topology,
    canonicalize_variable_set,
)


def chain_tree(vertex_ids=(0, 1, 2)) -> Topology:
    first, second, third = vertex_ids
    return Topology(
        n=4,
        vertices=vertex_ids,
        edges=(
            InternalEdge(first, second, EdgeType.DOUBLE),
            InternalEdge(second, third, EdgeType.DOUBLE),
        ),
        initial_vertex=first,
        final_slots={first: 1, second: 1, third: 2},
    )


def branching_tree() -> Topology:
    return Topology(
        n=4,
        vertices=(0, 1, 2),
        edges=(
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(0, 2, EdgeType.DOUBLE),
        ),
        initial_vertex=0,
        final_slots={0: 0, 1: 2, 2: 2},
    )


def chain_labeling(particle_at_first: int = 4, particle_at_second: int = 3) -> Labeling:
    return Labeling(
        final_particles=(
            FinalParticleLabel(0, 0, particle_at_first),
            FinalParticleLabel(1, 0, particle_at_second),
            FinalParticleLabel(2, 0, 1),
            FinalParticleLabel(2, 1, 2),
        )
    )


class TopologyCanonicalizationTests(unittest.TestCase):
    def test_vertex_renumbering_and_insertion_order_are_ignored(self) -> None:
        first = chain_tree((0, 1, 2))
        second = Topology(
            n=4,
            vertices=(40, 7, 19),
            edges=(
                InternalEdge(7, 19, EdgeType.DOUBLE),
                InternalEdge(40, 7, EdgeType.DOUBLE),
            ),
            initial_vertex=40,
            final_slots=((19, 2), (40, 1), (7, 1)),
        )
        first_result = canonicalize_topology(first)
        second_result = canonicalize_topology(second)
        self.assertEqual(first_result.key, second_result.key)
        self.assertEqual(first_result.digest, second_result.digest)
        self.assertEqual(len(first_result.original_to_canonical), 3)
        self.assertEqual(len(second_result.original_to_canonical), 3)

    def test_structurally_different_tree_shape_has_different_key(self) -> None:
        self.assertNotEqual(
            canonicalize_topology(chain_tree()).key,
            canonicalize_topology(branching_tree()).key,
        )

    def test_direction_and_line_type_are_part_of_key(self) -> None:
        original = chain_tree()
        reversed_direction = Topology(
            n=4,
            vertices=(0, 1, 2),
            edges=(
                InternalEdge(1, 0, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 1, 1: 1, 2: 2},
        )
        changed_type = Topology(
            n=4,
            vertices=(0, 1, 2),
            edges=(
                InternalEdge(0, 1, EdgeType.SINGLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 1, 1: 1, 2: 2},
        )
        self.assertNotEqual(canonicalize_topology(original).key, canonicalize_topology(reversed_direction).key)
        self.assertNotEqual(canonicalize_topology(original).key, canonicalize_topology(changed_type).key)

    def test_compact_and_duplicate_parallel_records_have_same_unmarked_key(self) -> None:
        compact = Topology(
            n=3,
            vertices=(0, 1, 2),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE, multiplicity=2),
                InternalEdge(1, 2, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 0, 1: 1, 2: 2},
        )
        expanded_records = Topology(
            n=3,
            vertices=(0, 1, 2),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(0, 1, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 0, 1: 1, 2: 2},
        )
        self.assertEqual(canonicalize_topology(compact).key, canonicalize_topology(expanded_records).key)


class MarkedCanonicalizationTests(unittest.TestCase):
    def test_external_slot_order_does_not_create_a_new_marked_key(self) -> None:
        original = chain_labeling()
        swapped_slots = Labeling(
            final_particles=(
                FinalParticleLabel(0, 0, 4),
                FinalParticleLabel(1, 0, 3),
                FinalParticleLabel(2, 0, 2),
                FinalParticleLabel(2, 1, 1),
            )
        )
        first = canonicalize_labeling(chain_tree(), original)
        second = canonicalize_labeling(chain_tree(), swapped_slots)
        self.assertEqual(first.key, second.key)

    def test_fixed_labels_and_common_particle_relabeling_are_distinct_modes(self) -> None:
        original = canonicalize_labeling(chain_tree(), chain_labeling(), CountMode.FIXED_LABELS)
        swapped_particles = canonicalize_labeling(
            chain_tree(), chain_labeling(particle_at_first=3, particle_at_second=4), CountMode.FIXED_LABELS
        )
        self.assertNotEqual(original.key, swapped_particles.key)
        original_s4 = canonicalize_labeling(chain_tree(), chain_labeling(), CountMode.PARTICLE_RELABELING)
        swapped_s4 = canonicalize_labeling(
            chain_tree(), chain_labeling(particle_at_first=3, particle_at_second=4), CountMode.PARTICLE_RELABELING
        )
        self.assertEqual(original_s4.key, swapped_s4.key)
        self.assertEqual(original_s4.mode, CountMode.PARTICLE_RELABELING)

    def test_parallel_line_labels_are_unordered_and_compact_equivalent(self) -> None:
        compact = Topology(
            n=3,
            vertices=(0, 1, 2),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE, multiplicity=2),
                InternalEdge(1, 2, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 0, 1: 1, 2: 2},
        )
        compact_labels = Labeling(
            final_particles=(
                FinalParticleLabel(1, 0, 1),
                FinalParticleLabel(2, 0, 2),
                FinalParticleLabel(2, 1, 3),
            ),
            double_lines=(
                DoubleLineLabel(0, 0, (1, 2)),
                DoubleLineLabel(0, 1, (1, 3)),
                DoubleLineLabel(1, 0, (2, 3)),
            ),
        )
        expanded = Topology(
            n=3,
            vertices=(0, 1, 2),
            edges=(
                InternalEdge(0, 1, EdgeType.DOUBLE),
                InternalEdge(1, 2, EdgeType.DOUBLE),
                InternalEdge(0, 1, EdgeType.DOUBLE),
            ),
            initial_vertex=0,
            final_slots={0: 0, 1: 1, 2: 2},
        )
        expanded_labels = Labeling(
            final_particles=compact_labels.final_particles,
            double_lines=(
                DoubleLineLabel(0, 0, (1, 3)),
                DoubleLineLabel(1, 0, (1, 2)),
                DoubleLineLabel(2, 0, (2, 3)),
            ),
        )
        first = canonicalize_labeling(compact, compact_labels)
        second = canonicalize_labeling(expanded, expanded_labels)
        self.assertEqual(first.key, second.key)

    def test_marked_result_has_stable_json(self) -> None:
        result = canonicalize_labeling(chain_tree(), chain_labeling())
        restored = json.loads(result.to_json())
        self.assertEqual(restored["canonical_key"], result.to_dict()["canonical_key"])
        self.assertEqual(restored["mode"], CountMode.FIXED_LABELS.value)


class VariableCombinationCanonicalizationTests(unittest.TestCase):
    def test_common_particle_permutation_and_variable_order_are_ignored(self) -> None:
        variables = ((1, 2), (1, 3), (2, 3, 4))
        permuted_particles = ((2, 3), (2, 4), (3, 4, 1))
        first = canonicalize_variable_set(variables, 4)
        second = canonicalize_variable_set(tuple(reversed(permuted_particles)), 4)
        self.assertEqual(first.key, second.key)

    def test_dn_collision_examples_keep_distinct_variable_keys(self) -> None:
        u = canonicalize_variable_set(((1, 2), (1, 3), (2, 3, 4)), 4)
        w = canonicalize_variable_set(((1, 2), (3, 4), (1, 2, 3)), 4)
        self.assertNotEqual(u.key, w.key)

    def test_repeated_subset_is_preserved(self) -> None:
        result = canonicalize_variable_set(((1, 2), (1, 2)), 4)
        self.assertEqual(len(result.variables), 2)
        self.assertEqual(len(result.key[2]), 2)


if __name__ == "__main__":
    unittest.main()
