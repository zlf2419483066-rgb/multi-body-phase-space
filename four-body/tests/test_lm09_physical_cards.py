import unittest

from im_fixtures import fixture_b, fixture_d, fixture_f, _fixture_c, fixture_e
from phase_space_graph import (
    Labeling,
    Topology,
    build_physics_card,
    normalize_physical_labeling,
    render_card_html,
    render_card_markdown,
)


class LM09PhysicalCardTests(unittest.TestCase):
    def test_b_card_keeps_all_incoming_momentum_at_vertex_two(self) -> None:
        fixture = fixture_b()
        card = build_physics_card(normalize_physical_labeling(fixture.topology, fixture.labeling))
        row = next(item for item in card.vertex_equations if item.vertex == 2)
        self.assertIn("p_2 + p_3 + p_4", row.incoming_recovered)
        self.assertIn("q_{2,1}", row.equation)
        self.assertIn("p_2", row.equation)
        self.assertEqual([item.variable_label for item in card.invariants], ["s12", "s34", "s23"])

    def test_c_variants_and_dependent_d_are_not_folded(self) -> None:
        alpha = build_physics_card(normalize_physical_labeling(_fixture_c(3).topology, _fixture_c(3).labeling))
        beta = build_physics_card(normalize_physical_labeling(_fixture_c(2).topology, _fixture_c(2).labeling))
        dependent = build_physics_card(normalize_physical_labeling(fixture_d().topology, fixture_d().labeling))
        self.assertEqual([item.variable_label for item in alpha.invariants], ["s124", "s123", "s12"])
        self.assertEqual([item.variable_label for item in beta.invariants], ["s134", "s123", "s12"])
        self.assertFalse(dependent.independence["independent"])
        self.assertEqual([item.variable_label for item in dependent.invariants], ["s234", "s24", "s24"])
        self.assertIn("s24 - s24", dependent.independence["relation"])

    def test_parallel_single_occurrences_and_measure_cancellation_are_visible(self) -> None:
        fixture = fixture_f()
        card = build_physics_card(normalize_physical_labeling(fixture.topology, fixture.labeling))
        self.assertEqual(len(card.connections), 8)  # 5 double + 3 single occurrences
        parallel = [row for row in card.connections if row.source == 2 and row.target == 3 and row.line_type == "单线"]
        self.assertEqual([(row.occurrence, row.particle_ids) for row in parallel], [(0, (1,)), (1, (2,))])
        counts = {row.particle_id: row.measure_cancellations for row in card.particles}
        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[2], 1)

    def test_record_json_readback_preserves_identity(self) -> None:
        fixture = fixture_b()
        normalization = normalize_physical_labeling(fixture.topology, fixture.labeling)
        card = build_physics_card(normalization, provenance={"page": "<script>alert(1)</script>"})
        saved = card.record
        self.assertIsNotNone(saved)
        topology = Topology.from_dict(saved["topology"])
        labeling = Labeling.from_dict(saved["complete_labeling"])
        reread = build_physics_card(normalize_physical_labeling(topology, labeling))
        self.assertEqual(card.identity_key, reread.identity_key)
        self.assertEqual(card.identity_digest, reread.identity_digest)

    def test_failure_card_has_diagnostic_and_safe_html(self) -> None:
        fixture = fixture_e()
        card = build_physics_card(
            normalize_physical_labeling(fixture.topology, fixture.labeling),
            provenance={"source": "<script>alert(1)</script>"},
        )
        self.assertEqual(card.stage, "structure_rejected")
        self.assertTrue(any(row["code"] == "outdegree_two" for row in card.diagnostics))
        markdown = render_card_markdown(card)
        html = render_card_html(card)
        self.assertIn("outdegree_two", markdown)
        self.assertIn("默认折叠", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
