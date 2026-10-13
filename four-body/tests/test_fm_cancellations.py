"""Tests for formula-level, occurrence-aware measure cancellation plans."""

from dataclasses import replace
import hashlib
import json
import unittest

from im_fixtures import fixture_b
from phase_space_graph import (
    CancellationPlanOverride,
    OccurrenceLocator,
    build_cancellation_plan,
    build_formula_input,
    canonicalize_physical_labeling,
)


class CancellationPlanTests(unittest.TestCase):
    def setUp(self):
        fixture = fixture_b()
        identity = canonicalize_physical_labeling(fixture.record)
        self.card = build_formula_input(
            fixture.record,
            measure_convention="jing_psf",
            identity=identity,
        )

    def test_default_binds_all_occurrences_and_matches_lm12(self):
        plan = build_cancellation_plan(self.card)
        particle = next(item for item in plan.particles if item.particle_id == 2)

        self.assertEqual(particle.retained_occurrence.occurrence_id, "p2:external:v3:slot0")
        self.assertEqual(
            [item.occurrence_id for item in particle.cancellation_occurrences],
            ["p2:internal:e2:o0"],
        )
        internal = particle.cancellation_occurrences[0]
        self.assertEqual((internal.source_vertex, internal.target_vertex), (1, 2))
        self.assertEqual(particle.input_cancellation_count, 1)
        self.assertEqual(particle.cancellation_count, 1)

        for item in plan.particles:
            self.assertEqual(len(item.occurrences), item.cancellation_count + 1)
            self.assertEqual(
                set(item.occurrences),
                {item.retained_occurrence, *item.cancellation_occurrences},
            )

    def test_formula_override_can_cancel_external_p2_without_mutating_lm_input(self):
        input_before = self.card.to_dict()
        override = CancellationPlanOverride(
            particle_id=2,
            cancellation_occurrences=(OccurrenceLocator("external", 3, 0),),
            reason="HAPOF 第28页 B 图对应选择：取消末顶点外部 p2。",
        )

        plan = build_cancellation_plan(self.card, overrides=(override,))
        particle = next(item for item in plan.particles if item.particle_id == 2)

        self.assertEqual(particle.retained_occurrence.occurrence_id, "p2:internal:e2:o0")
        self.assertEqual((particle.retained_occurrence.source_vertex, particle.retained_occurrence.target_vertex), (1, 2))
        self.assertEqual(
            [item.occurrence_id for item in particle.cancellation_occurrences],
            ["p2:external:v3:slot0"],
        )
        self.assertEqual(particle.selection_policy, "formula_level_cancellation_override")
        self.assertIn("HAPOF", particle.selection_reason)
        self.assertEqual(self.card.to_dict(), input_before)

        default_again = build_cancellation_plan(self.card)
        default_p2 = next(item for item in default_again.particles if item.particle_id == 2)
        self.assertEqual(default_p2.retained_occurrence.occurrence_id, "p2:external:v3:slot0")

    def test_serialization_is_deterministic_and_contains_complete_identity(self):
        first = build_cancellation_plan(self.card)
        second = build_cancellation_plan(self.card)
        payload = json.loads(first.to_json())

        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(payload["artifact_type"], "cancellation_plan")
        self.assertEqual(payload["formula_identity"]["digest"], self.card.identity["digest"])
        self.assertEqual(first.input_card_digest, hashlib.sha256(self.card.to_json().encode("utf-8")).hexdigest())
        self.assertEqual(payload["input_card_digest"], first.input_card_digest)
        self.assertEqual(payload["measure_convention"], self.card.measure_convention)
        self.assertEqual([item["particle_id"] for item in payload["particles"]], [1, 2, 3, 4])

    def test_rejects_wrong_metadata_count_duplicate_and_unbound_occurrences(self):
        p2_index = next(index for index, item in enumerate(self.card.particle_measures) if item.particle_id == 2)
        p2 = self.card.particle_measures[p2_index]

        rows = list(self.card.particle_measures)
        rows[p2_index] = replace(p2, cancellation_count=0)
        with self.assertRaisesRegex(ValueError, "card factors/variables/status disagree"):
            build_cancellation_plan(replace(self.card, particle_measures=tuple(rows)))

        rows[p2_index] = replace(p2, occurrences=p2.occurrences + (p2.occurrences[0],))
        with self.assertRaisesRegex(ValueError, "card factors/variables/status disagree"):
            build_cancellation_plan(replace(self.card, particle_measures=tuple(rows)))

        rows[p2_index] = replace(p2, occurrences=(("external", 99, 0), p2.occurrences[1]))
        with self.assertRaisesRegex(ValueError, "card factors/variables/status disagree"):
            build_cancellation_plan(replace(self.card, particle_measures=tuple(rows)))

    def test_override_must_name_exactly_all_cancellations(self):
        override = CancellationPlanOverride(
            particle_id=2,
            cancellation_occurrences=(),
            reason="invalid incomplete override",
        )
        with self.assertRaisesRegex(ValueError, "must cancel exactly 1"):
            build_cancellation_plan(self.card, overrides=(override,))

    def test_requires_identity_and_rejects_unknown_override_occurrence(self):
        with self.assertRaisesRegex(ValueError, "complete MarkedIdentity"):
            build_cancellation_plan(replace(self.card, identity=None, mode=None))

        override = CancellationPlanOverride(
            particle_id=2,
            cancellation_occurrences=(OccurrenceLocator("external", 99, 0),),
            reason="unknown test occurrence",
        )
        with self.assertRaisesRegex(ValueError, "unknown occurrences"):
            build_cancellation_plan(self.card, overrides=(override,))

    def test_rejects_tampered_identity_and_physical_record(self):
        changed_identity = dict(self.card.identity)
        changed_identity["digest"] = "not-the-canonical-digest"
        with self.assertRaisesRegex(ValueError, "formula identity mismatch: digest"):
            build_cancellation_plan(replace(self.card, identity=changed_identity))

        changed_record = dict(self.card.record)
        changed_record["single_groups"] = []
        with self.assertRaisesRegex(ValueError, "record disagrees with recomputed physical evidence"):
            build_cancellation_plan(replace(self.card, record=changed_record))


if __name__ == "__main__":
    unittest.main()
