"""Frozen-archive tests for the four-body all-loop identity catalog.

These tests exercise archive loading and canonical identity grouping only;
they do not run G05/G06/G08 enumeration or formula generation.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
import unittest

from phase_space_graph.domain import CountMode
from phase_space_graph.formula.d5_catalog import build_d5_catalog
from phase_space_graph.formula.all_loop_catalog import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_FIXED_COUNTS,
    EXPECTED_TOPOLOGY_COUNTS,
    _read_g08_fixed_identities,
    build_all_loop_catalog,
    default_g08_audit_path,
    select_all_loop_classes,
)
from phase_space_graph.physical_labels import PhysicalLabelRecord
from phase_space_graph.invariants import invariant_label
from phase_space_graph.label_identity import canonicalize_physical_labeling, relabel_physical_record


def _mapping(rows, source_key, target_key):
    return {row[source_key]: row[target_key] for row in rows}


class AllLoopCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = build_all_loop_catalog()
        cls.d5_catalog = build_d5_catalog()

    def test_frozen_counts_cover_all_four_loops(self) -> None:
        self.assertEqual(self.catalog.fixed_record_count, 3363)
        self.assertEqual(self.catalog.topology_count, 54)
        self.assertEqual(len(self.catalog.classes), 143)
        self.assertEqual(
            {loop: sum(item.loop_count == loop for item in self.catalog.classes)
             for loop in range(4)},
            EXPECTED_CLASS_COUNTS,
        )
        for loop in range(4):
            classes = tuple(item for item in self.catalog.classes if item.loop_count == loop)
            self.assertEqual(sum(len(item.members) for item in classes), EXPECTED_FIXED_COUNTS[loop])
            self.assertEqual(len({item.topology_id for item in classes}), EXPECTED_TOPOLOGY_COUNTS[loop])
            self.assertTrue(all(item.class_id.startswith(f"L{loop}-C") for item in classes))

    def test_fixed_members_retain_representative_maps_and_archive_locations(self) -> None:
        for marked_class in self.catalog.classes:
            self.assertIsInstance(marked_class.representative_record, PhysicalLabelRecord)
            self.assertEqual(
                marked_class.representative_fixed_digest,
                marked_class.representative.digest,
            )
            self.assertTrue(marked_class.members)
            for member in marked_class.members:
                self.assertTrue(member.archived.source_path)
                self.assertTrue(member.source_locator)
                self.assertEqual(member.map_to_representative["direction"],
                                 "source_fixed_record_to_representative_fixed_record")
                self.assertTrue(member.map_to_representative["particle_map"])
                self.assertIn("mass_symbol_substitution",
                              member.map_to_representative["particle_map"][0])
                self.assertTrue(member.map_to_representative["vertex_map"])
                self.assertTrue(member.map_to_representative["edge_occurrence_map"])
                self.assertTrue(member.map_to_representative["final_slot_map"])
                self.assertIn("invariant_map", member.map_to_representative)

    def test_every_fixed_map_replays_to_its_real_representative(self) -> None:
        """Replay every archived source identity through its explicit representative map."""
        replayed_by_loop: Counter[int] = Counter()
        replayed_total = 0
        for marked_class in self.catalog.classes:
            representative = marked_class.representative.canonical_record
            map_members = marked_class.members
            member_keys = {member.archived.key for member in map_members}
            canonical_fixed = canonicalize_physical_labeling(
                marked_class.particle_identity.canonical_record, CountMode.FIXED_LABELS
            )
            if canonical_fixed.key in member_keys:
                self.assertEqual(marked_class.representative.key, canonical_fixed.key)
            else:
                first_archived = min(
                    (member.archived for member in map_members),
                    key=lambda item: (item.program_id, item.digest),
                )
                self.assertEqual(marked_class.representative.key, first_archived.key)

            for member in map_members:
                data = member.map_to_representative
                vertex_map = _mapping(data["vertex_map"], "source_vertex", "representative_vertex")
                particle_map = _mapping(
                    data["particle_map"], "source_particle", "representative_particle"
                )
                replay = relabel_physical_record(
                    member.archived.canonical_record, vertex_map, particle_map
                )
                self.assertEqual(
                    replay.record.to_dict(), representative.to_dict(),
                    f"{marked_class.class_id}/{member.archived.program_id}: replayed record differs",
                )
                self.assertEqual(dict(replay.vertex_map), vertex_map)
                self.assertEqual(dict(replay.particle_map), particle_map)

                explicit_edges = {
                    (row["source_edge_index"], row["source_occurrence"]):
                    (row["representative_edge_index"], row["representative_occurrence"])
                    for row in data["edge_occurrence_map"]
                }
                self.assertEqual(dict(replay.edge_occurrence_map), explicit_edges)
                explicit_slots = {
                    (row["source_vertex"], row["source_slot"]):
                    (row["representative_vertex"], row["representative_slot"])
                    for row in data["final_slot_map"]
                }
                self.assertEqual(dict(replay.final_slot_map), explicit_slots)

                source_masses = {particle: 101.0 + particle for particle in range(1, 5)}
                representative_masses = {
                    target: source_masses[source] for source, target in particle_map.items()
                }
                for row in data["particle_map"]:
                    source = row["source_particle"]
                    target = row["representative_particle"]
                    self.assertEqual(
                        row["momentum_label_substitution"],
                        f"p_source[{source}] -> p_rep[{target}]",
                    )
                    self.assertEqual(
                        row["mass_symbol_substitution"],
                        f"m_source[{source}] -> m_rep[{target}]",
                    )
                    self.assertEqual(
                        row["representative_mass_assignment"],
                        f"m_rep[{target}] = m_source[{source}]",
                    )
                    self.assertEqual(representative_masses[target], source_masses[source])

                mapped_subsets = {
                    tuple(sorted(particle_map[particle] for particle in subset))
                    for subset in member.archived.canonical_record.momentum.variables
                }
                self.assertEqual(mapped_subsets, set(representative.momentum.variables))
                invariant_rows = {
                    row["source_ikv"]: row["representative_ikv"]
                    for row in data["invariant_map"]
                }
                for subset in member.archived.canonical_record.momentum.variables:
                    expected_label = invariant_label(
                        tuple(sorted(particle_map[particle] for particle in subset))
                    )
                    self.assertEqual(invariant_rows[invariant_label(subset)], expected_label)

                replayed_by_loop[marked_class.loop_count] += 1
                replayed_total += 1

        self.assertEqual(dict(replayed_by_loop), EXPECTED_FIXED_COUNTS)
        self.assertEqual(replayed_total, 3363)
        self.assertEqual(sum(replayed_by_loop.values()), 3363)

    def test_three_loop_class_ids_keep_the_legacy_d5_correspondence(self) -> None:
        three_loop = tuple(item for item in self.catalog.classes if item.loop_count == 3)
        self.assertEqual(len(three_loop), len(self.d5_catalog.classes))
        by_legacy_id = {item.legacy_d5_class_id: item for item in three_loop}
        for old_class in self.d5_catalog.classes:
            item = by_legacy_id[old_class.class_id]
            self.assertEqual(item.particle_identity.key, old_class.particle_identity.key)
            self.assertEqual(item.representative_fixed_digest, old_class.representative.digest)
            self.assertEqual(
                tuple(member.archived.digest for member in item.members),
                tuple(member.archived.digest for member in old_class.members),
            )

    def test_loop_topology_and_class_selectors_compose(self) -> None:
        selected = select_all_loop_classes(self.catalog, loops=(0, 2))
        self.assertTrue(selected)
        self.assertEqual({item.loop_count for item in selected}, {0, 2})

        target = self.catalog.classes[0]
        self.assertEqual(
            select_all_loop_classes(self.catalog, loops=target.loop_count,
                                    topology=target.topology_id,
                                    class_selector=target.class_id),
            (target,),
        )

        old_d5 = select_all_loop_classes(self.catalog, loops=3, class_selector="D5-001")
        self.assertEqual(len(old_d5), 1)
        self.assertEqual(old_d5[0].legacy_d5_class_id, "D5-001")

    def test_incomplete_g08_source_is_rejected(self) -> None:
        source = default_g08_audit_path(0)
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["status"] = "incomplete"
        with self.assertRaisesRegex(ValueError, "complete"):
            _read_g08_fixed_identities(0, payload, source)

    def test_duplicate_g08_topology_report_is_rejected(self) -> None:
        source = default_g08_audit_path(0)
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["labeling_reports"].append(copy.deepcopy(payload["labeling_reports"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate|repeat"):
            _read_g08_fixed_identities(0, payload, source)

    def test_duplicate_fixed_identity_is_rejected(self) -> None:
        source = default_g08_audit_path(0)
        payload = json.loads(source.read_text(encoding="utf-8"))
        report = payload["labeling_reports"][0]
        report["valid_results"].append(copy.deepcopy(report["valid_results"][0]))
        report["counts"]["unique"] += 1
        with self.assertRaisesRegex(ValueError, "duplicate|repeat"):
            _read_g08_fixed_identities(0, payload, source)


if __name__ == "__main__":
    unittest.main()
