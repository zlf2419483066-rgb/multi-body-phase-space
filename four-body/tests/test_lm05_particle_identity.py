import itertools
import unittest

from im_fixtures import _fixture_c, fixture_a, fixture_b, fixture_d
from phase_space_graph import (
    CountMode,
    canonicalize_physical_labeling,
    relabel_physical_record,
)


class LM05ParticleIdentityTests(unittest.TestCase):
    def test_a_fixed_particle_swap_differs_but_common_swap_agrees(self) -> None:
        fixture = fixture_a()
        swapped = relabel_physical_record(
            fixture.record,
            {0: 0, 1: 1, 2: 2},
            {1: 1, 2: 2, 3: 4, 4: 3},
        ).record
        self.assertNotEqual(
            canonicalize_physical_labeling(fixture.record, CountMode.FIXED_LABELS).key,
            canonicalize_physical_labeling(swapped, CountMode.FIXED_LABELS).key,
        )
        self.assertEqual(
            canonicalize_physical_labeling(fixture.record, CountMode.PARTICLE_RELABELING).key,
            canonicalize_physical_labeling(swapped, CountMode.PARTICLE_RELABELING).key,
        )

    def test_all_b_particle_permutations_share_one_common_key(self) -> None:
        record = fixture_b().record
        expected = canonicalize_physical_labeling(record, CountMode.PARTICLE_RELABELING)
        for permutation in itertools.permutations(range(1, 5)):
            particle_map = {particle: permutation[particle - 1] for particle in range(1, 5)}
            transformed = relabel_physical_record(
                record,
                {0: 0, 1: 1, 2: 2, 3: 3},
                particle_map,
            ).record
            actual = canonicalize_physical_labeling(transformed, CountMode.PARTICLE_RELABELING)
            self.assertEqual(actual.key, expected.key)
            self.assertEqual(actual.digest, expected.digest)

    def test_c_alpha_and_c_beta_remain_distinct(self) -> None:
        first = canonicalize_physical_labeling(_fixture_c(3).record, CountMode.PARTICLE_RELABELING)
        second = canonicalize_physical_labeling(_fixture_c(2).record, CountMode.PARTICLE_RELABELING)
        self.assertNotEqual(first.key, second.key)

    def test_dependent_record_keeps_dependency_under_common_renaming(self) -> None:
        record = fixture_d().record
        transformed = relabel_physical_record(
            record,
            {0: 0, 1: 2, 2: 1, 3: 3},
            {1: 2, 2: 4, 3: 1, 4: 3},
        ).record
        identity = canonicalize_physical_labeling(transformed, CountMode.PARTICLE_RELABELING)
        self.assertFalse(identity.canonical_record.independence.independent)
        self.assertEqual(identity.canonical_record.independence.rank, record.independence.rank)

    def test_common_identity_is_independent_of_input_permutation_order(self) -> None:
        record = fixture_b().record
        identities = {
            canonicalize_physical_labeling(record, CountMode.PARTICLE_RELABELING).key,
            canonicalize_physical_labeling(
                relabel_physical_record(record, {0: 0, 1: 1, 2: 2, 3: 3}, {1: 4, 2: 3, 3: 2, 4: 1}).record,
                CountMode.PARTICLE_RELABELING,
            ).key,
        }
        self.assertEqual(len(identities), 1)


if __name__ == "__main__":
    unittest.main()
