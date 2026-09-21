from __future__ import annotations

from copy import deepcopy
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
from phase_space_graph5.fixtures import fixture_a
from phase_space_graph5.label_identity import CandidateIdentity, canonicalize_candidate_labeling
from phase_space_graph5.physical_labels import CandidateMomentumRecord, normalize_candidate_labeling


def _parallel_topology(split: bool) -> tuple[Topology, Labeling]:
    edges = [
        InternalEdge(0, 1, EdgeType.DOUBLE),
        InternalEdge(0, 2, EdgeType.DOUBLE),
        InternalEdge(2, 3, EdgeType.DOUBLE),
        InternalEdge(3, 4, EdgeType.DOUBLE),
        InternalEdge(4, 5, EdgeType.DOUBLE),
    ]
    if split:
        edges.extend(
            (InternalEdge(1, 2, EdgeType.SINGLE), InternalEdge(1, 2, EdgeType.SINGLE))
        )
    else:
        edges.append(InternalEdge(1, 2, EdgeType.SINGLE, multiplicity=2))
    topology = Topology(
        n=5,
        vertices=(0, 1, 2, 3, 4, 5),
        edges=tuple(edges),
        initial_vertex=0,
        final_slots={0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2},
    )
    single_indices = [
        index for index, edge in enumerate(topology.edges) if edge.edge_type is EdgeType.SINGLE
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


class FiveBodyIdentityTests(unittest.TestCase):
    def test_identity_roundtrip_rechecks_record_and_maps(self) -> None:
        topology, labeling = fixture_a()
        normalized = normalize_candidate_labeling(topology, labeling)
        assert normalized.record is not None
        identity = canonicalize_candidate_labeling(normalized.record, CountMode.FIXED_LABELS)
        loaded = CandidateIdentity.from_dict(identity.to_dict())
        self.assertEqual(loaded.key, identity.key)
        self.assertEqual(loaded.digest, identity.digest)

        tampered = deepcopy(identity.to_dict())
        tampered["canonical_record"]["momentum"]["double_momenta"][0]["coefficients"][0] = 1.2
        with self.assertRaises(TypeError):
            CandidateIdentity.from_dict(tampered)

        tampered = deepcopy(identity.to_dict())
        tampered["vertex_map"] = tampered["vertex_map"][:-1]
        with self.assertRaises(ValueError):
            CandidateIdentity.from_dict(tampered)

    def test_unknown_deferred_state_is_not_normalized_on_readback(self) -> None:
        topology, labeling = fixture_a()
        normalized = normalize_candidate_labeling(topology, labeling)
        assert normalized.record is not None
        payload = normalized.record.to_dict()
        payload["independence_status"] = "independent"
        with self.assertRaises(ValueError):
            CandidateMomentumRecord.from_dict(payload)

    def test_fixed_particle_positions_remain_distinct_but_slot_exchange_is_equivalent(self) -> None:
        topology, labeling = fixture_a()
        original = normalize_candidate_labeling(topology, labeling)
        assert original.record is not None

        swapped_particle_labeling = Labeling(
            final_particles=tuple(
                FinalParticleLabel(
                    item.vertex,
                    item.slot,
                    {1: 2, 2: 1}.get(item.particle_id, item.particle_id),
                )
                for item in labeling.final_particles
            )
        )
        swapped_particles = normalize_candidate_labeling(topology, swapped_particle_labeling)
        assert swapped_particles.record is not None
        self.assertNotEqual(
            canonicalize_candidate_labeling(original.record, CountMode.FIXED_LABELS).key,
            canonicalize_candidate_labeling(swapped_particles.record, CountMode.FIXED_LABELS).key,
        )
        self.assertEqual(
            canonicalize_candidate_labeling(original.record, CountMode.PARTICLE_RELABELING).key,
            canonicalize_candidate_labeling(swapped_particles.record, CountMode.PARTICLE_RELABELING).key,
        )

        swapped_slots = Labeling(
            final_particles=tuple(
                FinalParticleLabel(
                    item.vertex,
                    item.slot,
                    {4: 5, 5: 4}.get(item.particle_id, item.particle_id),
                )
                for item in labeling.final_particles
            )
        )
        slot_exchange = normalize_candidate_labeling(topology, swapped_slots)
        assert slot_exchange.record is not None
        self.assertEqual(
            canonicalize_candidate_labeling(original.record, CountMode.FIXED_LABELS).key,
            canonicalize_candidate_labeling(slot_exchange.record, CountMode.FIXED_LABELS).key,
        )

    def test_edge_order_relabeling_preserves_identity(self) -> None:
        topology, labeling = fixture_a()
        reordered = Topology(
            n=topology.n,
            vertices=topology.vertices,
            edges=tuple(reversed(topology.edges)),
            initial_vertex=topology.initial_vertex,
            final_slots=dict(topology.final_slots),
        )
        first = normalize_candidate_labeling(topology, labeling)
        second = normalize_candidate_labeling(reordered, labeling)
        assert first.record is not None and second.record is not None
        self.assertEqual(
            canonicalize_candidate_labeling(first.record).key,
            canonicalize_candidate_labeling(second.record).key,
        )

    def test_noncontiguous_vertex_renumbering_keeps_identity_and_readback(self) -> None:
        topology, labeling = fixture_a()
        rename = {0: 11, 1: 24, 2: 37, 3: 50}
        renumbered_topology = Topology(
            n=5,
            vertices=tuple(rename[v] for v in topology.vertices),
            edges=tuple(
                InternalEdge(rename[e.source], rename[e.target], e.edge_type, e.multiplicity)
                for e in topology.edges
            ),
            initial_vertex=rename[topology.initial_vertex],
            final_slots={rename[v]: count for v, count in topology.final_slots},
        )
        renumbered_labeling = Labeling(
            final_particles=tuple(
                FinalParticleLabel(rename[item.vertex], item.slot, item.particle_id)
                for item in labeling.final_particles
            )
        )
        original = normalize_candidate_labeling(topology, labeling)
        renamed = normalize_candidate_labeling(renumbered_topology, renumbered_labeling)
        assert original.record is not None and renamed.record is not None
        first = canonicalize_candidate_labeling(original.record)
        second = canonicalize_candidate_labeling(renamed.record)
        self.assertEqual(first.key, second.key)
        loaded = CandidateIdentity.from_dict(second.to_dict())
        self.assertEqual(loaded.key, second.key)

    def test_compact_and_split_parallel_single_occurrences_share_identity(self) -> None:
        compact_topology, compact_labeling = _parallel_topology(split=False)
        split_topology, split_labeling = _parallel_topology(split=True)
        compact = normalize_candidate_labeling(compact_topology, compact_labeling)
        split = normalize_candidate_labeling(split_topology, split_labeling)
        self.assertEqual(compact.stage, "momentum_candidate")
        self.assertEqual(split.stage, "momentum_candidate")
        assert compact.record is not None and split.record is not None
        self.assertEqual(len(compact.record.complete_labeling.single_lines), 2)
        self.assertEqual(len(split.record.complete_labeling.single_lines), 2)
        self.assertEqual(compact.record.variables, split.record.variables)

        compact_identity = canonicalize_candidate_labeling(compact.record)
        split_identity = canonicalize_candidate_labeling(split.record)
        self.assertEqual(compact_identity.key, split_identity.key)
        self.assertEqual(
            compact_identity.canonical_record.to_dict(),
            split_identity.canonical_record.to_dict(),
        )

        swapped_labeling = Labeling(
            final_particles=compact_labeling.final_particles,
            single_lines=(
                SingleLineLabel(
                    next(index for index, edge in enumerate(compact_topology.edges) if edge.edge_type.value == "single"),
                    0,
                    2,
                ),
                SingleLineLabel(
                    next(index for index, edge in enumerate(compact_topology.edges) if edge.edge_type.value == "single"),
                    1,
                    1,
                ),
            ),
        )
        swapped = normalize_candidate_labeling(compact_topology, swapped_labeling)
        self.assertEqual(swapped.stage, "momentum_candidate")
        assert swapped.record is not None
        self.assertEqual(
            compact_identity.key,
            canonicalize_candidate_labeling(swapped.record).key,
        )

    def test_direct_identity_entry_points_reject_changed_complete_labeling(self) -> None:
        topology, labeling = fixture_a()
        normalized = normalize_candidate_labeling(topology, labeling)
        assert normalized.record is not None
        payload = deepcopy(normalized.record.to_dict())
        payload["complete_labeling"]["final_particles"][0]["particle_id"] = 2
        payload["complete_labeling"]["final_particles"][1]["particle_id"] = 1
        damaged = CandidateMomentumRecord.from_dict(payload)
        with self.assertRaises(ValueError):
            canonicalize_candidate_labeling(damaged)


if __name__ == "__main__":
    unittest.main()
