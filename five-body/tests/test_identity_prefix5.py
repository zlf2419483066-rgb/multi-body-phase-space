from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5 import label_identity as li
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
from phase_space_graph5.fixtures import fixture_b
from phase_space_graph5.labeling import enumerate_labelings
from phase_space_graph5.physical_labels import CandidateMomentumRecord, normalize_candidate_labeling


FIELDS = (
    "schema_version", "mode", "topology_key", "key", "digest", "canonical_record",
    "vertex_map", "particle_map", "edge_occurrence_map", "final_slot_map",
)
ROOT = Path(__file__).resolve().parents[1]


def _l0_source_records() -> list[CandidateMomentumRecord]:
    """Build the 105 fixed-label source representatives from public APIs."""

    identities: dict[str, CandidateMomentumRecord] = {}
    report = enumerate_topologies(5, 0)
    if report.status.value != "complete":
        raise AssertionError(report.to_dict())
    for topology in report.topologies:
        labeling_report = enumerate_labelings(topology, CountMode.FIXED_LABELS)
        if not labeling_report.complete:
            raise AssertionError(labeling_report.to_dict())
        for record in labeling_report.candidate_records:
            identity = li.canonicalize_candidate_labeling(record, CountMode.FIXED_LABELS)
            identities.setdefault(identity.digest, record)
    result = list(identities.values())
    if len(result) != 105:
        raise AssertionError(f"expected 105 fixed-label L0 records, got {len(result)}")
    return result


def _l1_representative_records() -> list[CandidateMomentumRecord]:
    """Use three self-contained one-loop fixture records as L1 representatives."""

    records: list[CandidateMomentumRecord] = []
    for particle in (2, 3, 5):
        topology, labeling = fixture_b(particle)
        result = normalize_candidate_labeling(topology, labeling)
        if result.record is None:
            raise AssertionError(result.to_dict())
        records.append(result.record)
    return records


def _renumbered_fixture() -> CandidateMomentumRecord:
    topology, labeling = fixture_a()
    rename = {0: 11, 1: 24, 2: 37, 3: 50}
    renamed_topology = Topology(
        n=5,
        vertices=tuple(rename[vertex] for vertex in topology.vertices),
        edges=tuple(
            InternalEdge(rename[edge.source], rename[edge.target], edge.edge_type, edge.multiplicity)
            for edge in topology.edges
        ),
        initial_vertex=rename[topology.initial_vertex],
        final_slots={rename[vertex]: count for vertex, count in topology.final_slots},
    )
    renamed_labeling = Labeling(
        final_particles=tuple(
            FinalParticleLabel(rename[item.vertex], item.slot, item.particle_id)
            for item in labeling.final_particles
        )
    )
    result = normalize_candidate_labeling(renamed_topology, renamed_labeling)
    if result.record is None:
        raise AssertionError(result.to_dict())
    return result.record


def _slot_exchange_fixture() -> CandidateMomentumRecord:
    topology, labeling = fixture_a()
    swapped = Labeling(
        final_particles=tuple(
            FinalParticleLabel(item.vertex, item.slot, {4: 5, 5: 4}.get(item.particle_id, item.particle_id))
            for item in labeling.final_particles
        )
    )
    result = normalize_candidate_labeling(topology, swapped)
    if result.record is None:
        raise AssertionError(result.to_dict())
    return result.record


def _parallel_fixture(split: bool) -> CandidateMomentumRecord:
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
        5, (0, 1, 2, 3, 4, 5), tuple(edges), 0,
        {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2},
    )
    single_indices = [index for index, edge in enumerate(topology.edges)
                      if edge.edge_type is EdgeType.SINGLE]
    single_lines = (
        (SingleLineLabel(single_indices[0], 0, 1), SingleLineLabel(single_indices[0], 1, 2))
        if not split else
        (SingleLineLabel(single_indices[0], 0, 1), SingleLineLabel(single_indices[1], 0, 2))
    )
    labeling = Labeling(
        final_particles=(
            FinalParticleLabel(2, 0, 1), FinalParticleLabel(3, 0, 2),
            FinalParticleLabel(4, 0, 3), FinalParticleLabel(5, 0, 4),
            FinalParticleLabel(5, 1, 5),
        ),
        single_lines=single_lines,
    )
    result = normalize_candidate_labeling(topology, labeling)
    if result.record is None:
        raise AssertionError(result.to_dict())
    return result.record


def _counted_identity(record: CandidateMomentumRecord, *, pruned: bool) -> tuple[dict[str, object], int]:
    original = li._marked_payload
    calls = 0

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    with patch.object(li, "_marked_payload", side_effect=counted):
        identity = li.canonicalize_candidate_labeling(
            record, CountMode.PARTICLE_RELABELING,
            use_particle_prefix_pruning=pruned,
        )
    return identity.to_dict(), calls


class ParticlePrefixIdentityTests(unittest.TestCase):
    def test_all_l0_fields_match_reference_and_reduce_payload_evaluations(self) -> None:
        records = _l0_source_records()
        pruned_count = 0
        reference_count = 0
        for record in records:
            pruned, calls = _counted_identity(record, pruned=True)
            reference, reference_calls = _counted_identity(record, pruned=False)
            for field in FIELDS:
                self.assertEqual(pruned[field], reference[field], field)
            pruned_count += calls
            reference_count += reference_calls
        self.assertEqual(pruned_count, 360)
        self.assertEqual(reference_count, 14_400)

    def test_l1_representatives_match_all_identity_fields(self) -> None:
        records = _l1_representative_records()
        self.assertEqual(len(records), 3)
        for record in records:
            pruned = li.canonicalize_candidate_labeling(record, CountMode.PARTICLE_RELABELING)
            reference = li.canonicalize_candidate_labeling(
                record, CountMode.PARTICLE_RELABELING, use_particle_prefix_pruning=False,
            )
            self.assertEqual(pruned.to_dict(), reference.to_dict())

    def test_noncontiguous_slots_and_parallel_occurrences_match_reference(self) -> None:
        records = (_renumbered_fixture(), _slot_exchange_fixture(),
                   _parallel_fixture(True), _parallel_fixture(False))
        for record in records:
            pruned = li.canonicalize_candidate_labeling(record, CountMode.PARTICLE_RELABELING)
            reference = li.canonicalize_candidate_labeling(
                record, CountMode.PARTICLE_RELABELING, use_particle_prefix_pruning=False,
            )
            self.assertEqual(pruned.to_dict(), reference.to_dict())

    def test_fixed_labels_are_unchanged_and_reference_switch_is_strict(self) -> None:
        topology, labeling = fixture_a()
        normalized = normalize_candidate_labeling(topology, labeling)
        self.assertIsNotNone(normalized.record)
        record = normalized.record
        assert record is not None
        default = li.canonicalize_candidate_labeling(record, CountMode.FIXED_LABELS)
        reference = li.canonicalize_candidate_labeling(
            record, CountMode.FIXED_LABELS, use_particle_prefix_pruning=False,
        )
        self.assertEqual(default.to_dict(), reference.to_dict())
        with self.assertRaises(TypeError):
            li.canonicalize_candidate_labeling(
                record, CountMode.PARTICLE_RELABELING, use_particle_prefix_pruning=1,
            )

        payload = deepcopy(record.to_dict())
        payload["complete_labeling"]["final_particles"][0]["particle_id"] = 2
        payload["complete_labeling"]["final_particles"][1]["particle_id"] = 1
        damaged = CandidateMomentumRecord.from_dict(payload)
        for pruned in (True, False):
            with self.assertRaises(ValueError):
                li.canonicalize_candidate_labeling(
                    damaged, CountMode.PARTICLE_RELABELING,
                    use_particle_prefix_pruning=pruned,
                )


if __name__ == "__main__":
    unittest.main()
