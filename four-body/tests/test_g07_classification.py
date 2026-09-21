import json
import unittest

from phase_space_graph import (
    EdgeType,
    FinalParticleLabel,
    InternalEdge,
    Labeling,
    LabelingLimits,
    Topology,
    build_coverage_index,
    classify_variable_combination,
    dn_signature,
    enumerate_invariant_combinations,
    enumerate_labelings,
    link_labeling_report,
)


def f03() -> Topology:
    return Topology(
        4,
        (0, 1, 2, 3),
        (
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(0, 2, EdgeType.DOUBLE),
            InternalEdge(2, 3, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.SINGLE),
        ),
        0,
        {0: 0, 1: 1, 2: 1, 3: 2},
    )


class CombinationRouteTests(unittest.TestCase):
    def test_four_body_counts_match_binomial_and_exact_rank(self) -> None:
        report = enumerate_invariant_combinations(4)
        self.assertTrue(report.complete)
        self.assertEqual(report.candidate_count, 10)
        self.assertEqual(report.raw_count, 627)
        self.assertEqual(report.independent_count, 522)
        self.assertEqual(report.dependent_count, 105)
        self.assertEqual(report.unique_count, 40)
        self.assertEqual(
            [(item.size, item.raw_count, item.independent_count, item.unique_count) for item in report.summaries],
            [(2, 45, 45, 5), (3, 120, 120, 11), (4, 210, 195, 13), (5, 252, 162, 11)],
        )
        restored = json.loads(report.to_json())
        self.assertEqual(restored["counts"]["raw"], 627)
        self.assertEqual(restored["counts"]["unique"], 40)

    def test_three_body_default_combination_scope_is_two_invariants(self) -> None:
        report = enumerate_invariant_combinations(3)
        self.assertTrue(report.complete)
        self.assertEqual(report.sizes, (2,))
        self.assertEqual((report.raw_count, report.independent_count, report.unique_count), (3, 3, 1))

    def test_dn_and_f05_collision_are_kept_distinct(self) -> None:
        first = classify_variable_combination(((1, 2), (1, 3), (2, 3, 4)), 4)
        second = classify_variable_combination(((1, 2), (3, 4), (1, 2, 3)), 4)
        self.assertEqual(first.dn.label, "(3;7;2221)")
        self.assertEqual(second.dn.label, "(3;7;2221)")
        self.assertNotEqual(first.canonical.key, second.canonical.key)

    def test_common_particle_permutation_is_the_only_variable_equivalence(self) -> None:
        original = classify_variable_combination(((1, 2), (1, 3), (2, 3)), 4)
        permuted = classify_variable_combination(((2, 4), (3, 4), (2, 3)), 4)
        self.assertEqual(original.canonical.key, permuted.canonical.key)
        with self.assertRaisesRegex(ValueError, "cannot repeat"):
            classify_variable_combination(((1, 2), (1, 2)), 4)


class GraphRouteCoverageTests(unittest.TestCase):
    def test_f03_valid_labels_link_to_one_variable_class(self) -> None:
        report = enumerate_labelings(f03())
        links = link_labeling_report(f03(), report)
        self.assertEqual(report.unique_count, 24)
        self.assertEqual(len(links), 24)
        self.assertTrue(all(link.combination.independence.independent for link in links))
        self.assertEqual({link.combination.dn.label for link in links}, {"(3;6;2211)"})
        coverage = build_coverage_index(links)
        self.assertEqual(len(coverage.topology_to_variables), 1)
        self.assertEqual(len(coverage.variable_to_topologies), 1)
        self.assertEqual(len(coverage.dn_to_variables), 1)
        self.assertEqual(len(json.loads(coverage.to_json())["links"]), 24)

    def test_incomplete_g06_report_remains_incomplete_after_linking(self) -> None:
        report = enumerate_labelings(f03(), limits=LabelingLimits(max_labelings=1))
        links = link_labeling_report(f03(), report)
        self.assertFalse(report.complete)
        self.assertEqual(report.stop_reason, "limit_max_labelings")
        self.assertIsInstance(links, tuple)


if __name__ == "__main__":
    unittest.main()
