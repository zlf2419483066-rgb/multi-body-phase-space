import unittest

from im_fixtures import fixture_a, fixture_b, fixture_d, fixture_f
from phase_space_graph import (
    CountMode,
    DoubleLineLabel,
    EdgeType,
    InternalEdge,
    Labeling,
    SingleLineLabel,
    Topology,
    canonicalize_physical_labeling,
    normalize_physical_labeling,
    relabel_physical_record,
)
from phase_space_graph.label_identity import _marked_payload


class LM04FixedIdentityTests(unittest.TestCase):
    def test_blank_partial_and_complete_b_have_one_fixed_identity(self) -> None:
        fixture = fixture_b()
        complete = fixture.record.complete_labeling
        partial = Labeling(
            fixture.labeling.final_particles,
            fixture.labeling.single_lines,
            (complete.double_lines[0],),
        )
        records = [
            normalize_physical_labeling(fixture.topology, fixture.labeling).record,
            normalize_physical_labeling(fixture.topology, partial).record,
            normalize_physical_labeling(fixture.topology, complete).record,
        ]
        identities = [canonicalize_physical_labeling(record) for record in records]
        self.assertEqual(len({identity.key for identity in identities}), 1)
        self.assertEqual(len({identity.digest for identity in identities}), 1)

    def test_fixed_particle_swap_changes_a_identity(self) -> None:
        fixture = fixture_a()
        swapped = relabel_physical_record(
            fixture.record,
            {0: 0, 1: 1, 2: 2},
            {1: 1, 2: 2, 3: 4, 4: 3},
        ).record
        first = canonicalize_physical_labeling(fixture.record)
        second = canonicalize_physical_labeling(swapped)
        self.assertNotEqual(first.key, second.key)
        self.assertNotEqual(first.digest, second.digest)

    def test_split_parallel_single_edges_do_not_change_identity(self) -> None:
        fixture = fixture_f()
        split_topology = Topology(
            n=4,
            vertices=fixture.topology.vertices,
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
            final_slots=dict(fixture.topology.final_slots),
        )
        split_labeling = Labeling(
            fixture.labeling.final_particles,
            (SingleLineLabel(1, 0, 1), SingleLineLabel(4, 0, 1), SingleLineLabel(5, 0, 2)),
            (
                DoubleLineLabel(0, 0, (2, 3, 4)),
                DoubleLineLabel(2, 0, (1, 2)),
                DoubleLineLabel(3, 0, (3, 4)),
                DoubleLineLabel(6, 0, (1, 3)),
                DoubleLineLabel(7, 0, (2, 4)),
            ),
        )
        split_record = normalize_physical_labeling(split_topology, split_labeling).record
        self.assertEqual(
            canonicalize_physical_labeling(fixture.record).key,
            canonicalize_physical_labeling(split_record).key,
        )
        self.assertEqual(len(split_record.complete_labeling.single_lines), 3)

    def test_identity_is_idempotent_and_payload_matches_record(self) -> None:
        identity = canonicalize_physical_labeling(fixture_b().record)
        repeated = canonicalize_physical_labeling(identity.canonical_record)
        self.assertEqual(identity.key, repeated.key)
        self.assertEqual(identity.digest, repeated.digest)
        payload = _marked_payload(
            identity.canonical_record,
            {vertex: vertex for vertex in identity.canonical_record.topology.vertices},
        )
        self.assertEqual(identity.key[-2:], payload)

    def test_dependent_records_receive_fixed_identity(self) -> None:
        identity = canonicalize_physical_labeling(fixture_d().record)
        self.assertEqual(identity.canonical_record.independence.independent, False)
        self.assertEqual(identity.key[1], CountMode.FIXED_LABELS.value)

    def test_particle_relabeling_mode_is_available_after_lm05(self) -> None:
        identity = canonicalize_physical_labeling(fixture_b().record, CountMode.PARTICLE_RELABELING)
        self.assertEqual(identity.mode, CountMode.PARTICLE_RELABELING)


if __name__ == "__main__":
    unittest.main()
