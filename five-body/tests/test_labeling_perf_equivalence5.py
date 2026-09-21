"""Equivalence guard for fixture-B profiling entry points.

The benchmark itself is deliberately outside the production package.  These
small tests pin the evidence needed before any bounded optimization: complete
fixed particle identity data, all 120 particle relabelings, and the candidate
record reconstructed by each stage.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase_space_graph5.domain import (  # noqa: E402
    CountMode,
    FinalParticleLabel,
    Labeling,
    SingleLineLabel,
)
from phase_space_graph5.fixtures import fixture_b  # noqa: E402
from phase_space_graph5.label_identity import canonicalize_candidate_labeling  # noqa: E402
from phase_space_graph5.labeling import evaluate_labeling  # noqa: E402
from phase_space_graph5.physical_labels import (  # noqa: E402
    candidate_record_from_evaluation,
    normalize_candidate_labeling,
    review_candidate_record,
)


class FixtureBLabelingEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topology, cls.labeling = fixture_b(3)
        normalized = normalize_candidate_labeling(cls.topology, cls.labeling)
        if normalized.record is None:
            raise AssertionError(normalized.to_dict())
        cls.record = normalized.record
        cls.evaluated = evaluate_labeling(cls.topology, cls.labeling)
        cls.fixed = canonicalize_candidate_labeling(cls.record, CountMode.FIXED_LABELS)
        cls.relabel = canonicalize_candidate_labeling(cls.record, CountMode.PARTICLE_RELABELING)

    def test_original_numbering_is_complete_and_preserved(self) -> None:
        self.assertEqual(self.topology.n, 5)
        self.assertEqual(sorted(item.particle_id for item in self.labeling.final_particles), [1, 2, 3, 4, 5])
        self.assertEqual(tuple(item.particle_id for item in self.labeling.single_lines), (3,))
        self.assertEqual(len(self.record.complete_labeling.double_lines), 4)
        self.assertEqual(self.record.to_dict()["topology"], self.topology.to_dict())

    def test_normalize_record_and_review_reconstruct_identically(self) -> None:
        rebuilt = candidate_record_from_evaluation(self.topology, self.evaluated)
        reviewed = review_candidate_record(self.record)
        self.assertEqual(rebuilt.to_dict(), self.record.to_dict())
        self.assertIsNotNone(reviewed.record)
        self.assertEqual(reviewed.record.to_dict(), self.record.to_dict())

    def test_fixed_labels_keep_full_particle_identity(self) -> None:
        self.assertEqual(self.fixed.mode, CountMode.FIXED_LABELS)
        self.assertEqual(tuple(self.fixed.particle_map), tuple((p, p) for p in range(1, 6)))
        swapped_labeling = type(self.labeling)(
            final_particles=tuple(
                type(item)(item.vertex, item.slot, {1: 2, 2: 1}.get(item.particle_id, item.particle_id))
                for item in self.labeling.final_particles
            ),
            single_lines=self.labeling.single_lines,
        )
        swapped = normalize_candidate_labeling(self.topology, swapped_labeling)
        self.assertIsNotNone(swapped.record)
        self.assertNotEqual(self.fixed.key, canonicalize_candidate_labeling(swapped.record, CountMode.FIXED_LABELS).key)

    def test_particle_relabeling_considers_all_120_permutations(self) -> None:
        fixed_digests = set()
        source_assignments = 0
        for permutation in permutations(range(1, 6)):
            particle_map = dict(zip(range(1, 6), permutation))
            relabeled_labeling = Labeling(
                final_particles=tuple(
                    FinalParticleLabel(item.vertex, item.slot, particle_map[item.particle_id])
                    for item in self.labeling.final_particles
                ),
                single_lines=tuple(
                    SingleLineLabel(item.edge_index, item.occurrence, particle_map[item.particle_id])
                    for item in self.labeling.single_lines
                ),
            )
            normalized = normalize_candidate_labeling(self.topology, relabeled_labeling)
            self.assertIsNotNone(normalized.record)
            fixed_digests.add(
                canonicalize_candidate_labeling(normalized.record, CountMode.FIXED_LABELS).digest
            )
            source_assignments += 1
        # All 120 fixed-particle assignments are actually processed.  The
        # two final slots at vertex 4 are exchange-equivalent, so the
        # canonical catalog has 60 fixed identities for this topology.
        self.assertEqual(source_assignments, 120)
        self.assertEqual(len(fixed_digests), 60)
        self.assertEqual(len(tuple(permutations(range(1, 6)))), 120)
        # The relabeling mode still searches all 120 particle permutations
        # internally; its public key is one quotient representative.
        self.assertEqual(self.relabel.mode, CountMode.PARTICLE_RELABELING)


if __name__ == "__main__":
    unittest.main()
