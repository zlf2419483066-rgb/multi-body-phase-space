"""Graph-derived four-body three-loop tree-certificate regressions."""

from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import unittest

from phase_space_graph import (
    build_formula_input,
    canonicalize_physical_labeling,
    generate_formula,
)
from phase_space_graph.drawing.archive import load_identity_archive
from phase_space_graph.formula.raw import build_f0_expression, reduce_covariant
from phase_space_graph.formula.tree_covariant import build_tree_covariant_trace


ARCHIVE = (
    Path(__file__).parents[1]
    / "文档"
    / "验证与测试"
    / "误差分析"
    / "三圈来源集合审查_2026-09-16"
    / "run01"
    / "2544个固定标记完整身份.json.gz"
)


@lru_cache(maxsize=2)
def _d5_card(convention: str = "jing_psf"):
    fixed = next(
        item for item in load_identity_archive(ARCHIVE)
        if item.program_id == "P02"
    )
    identity = canonicalize_physical_labeling(fixed.canonical_record)
    return build_formula_input(
        fixed.canonical_record,
        measure_convention=convention,
        identity=identity,
    )


def _normalize_polynomial(polynomial):
    normalized = {}
    for symbol, coefficient in polynomial.items():
        assert isinstance(symbol, str)
        assert isinstance(coefficient, int) and not isinstance(coefficient, bool)
        if coefficient:
            normalized[symbol] = coefficient
    return normalized


def _substitute_polynomial(polynomial, symbol, replacement):
    source = _normalize_polynomial(polynomial)
    solution = _normalize_polynomial(replacement)
    assert symbol not in solution
    coefficient = source.pop(symbol, 0)
    for target, target_coefficient in solution.items():
        updated = source.get(target, 0) + coefficient * target_coefficient
        if updated:
            source[target] = updated
        else:
            source.pop(target, None)
    return _normalize_polynomial(source)


def _equation_map(rows):
    result = {int(row["vertex"]): _normalize_polynomial(row["coefficients"]) for row in rows}
    assert len(result) == len(rows), "equation rows repeat a vertex"
    return result


def _replay_integer_tree_elimination(trace):
    elimination = trace["elimination"]
    steps = elimination["steps"]
    assert len(steps) == 5
    replayed = None
    for index, step in enumerate(steps):
        before = _equation_map(step["equations_before"])
        if replayed is not None:
            assert replayed == before, "one elimination after-state differs from the next before-state"
        leaf = int(step["leaf_vertex"])
        assert leaf in before
        assert len(step["equations_after"]) == len(step["equations_before"]) - 1
        q = str(step["integrated_q"])
        leaf_equation = _normalize_polynomial(step["current_equation_coefficients"])
        assert leaf_equation == before[leaf], "current leaf equation differs from its before-state row"
        q_coefficient = leaf_equation.get(q, 0)
        assert q_coefficient in (-1, 1), "integrated q coefficient must be +1 or -1"
        assert q_coefficient == step["q_coefficient"]
        substitution = _normalize_polynomial(step["substitution_coefficients"])
        assert q not in substitution
        assert _substitute_polynomial(leaf_equation, q, substitution) == {}, (
            "recorded q solution does not zero the current leaf equation"
        )

        expected_after = {
            vertex: _substitute_polynomial(equation, q, substitution)
            for vertex, equation in before.items()
            if vertex != leaf
        }
        actual_after = _equation_map(step["equations_after"])
        assert len(actual_after) == len(before) - 1
        assert actual_after == expected_after, "recorded equations_after does not follow by exact q substitution"
        replayed = actual_after

    root = int(elimination["root_vertex"])
    assert set(replayed or {}) == {root}, "leaf elimination did not leave exactly the declared root"
    expected_root = {"P": 1, "p_1": -1, "p_2": -1, "p_3": -1, "p_4": -1}
    assert replayed[root] == expected_root
    assert _normalize_polynomial(elimination["root_residual_coefficients"]) == expected_root


class TreeCovariantCertificateTests(unittest.TestCase):
    def test_trace_replays_all_five_leaf_integrations_and_global_support(self):
        card = _d5_card()
        trace = build_tree_covariant_trace(card)

        self.assertEqual(trace.status, "built")
        self.assertFalse(trace.formula_verified)
        self.assertEqual(trace.source_match_status, "not_assessed")
        self.assertEqual(trace.scope["loop_count"], 3)
        self.assertEqual(trace.scope["double_line_tree_edge_count"], 5)
        self.assertTrue(trace.scope["double_line_tree_connected"])
        self.assertEqual(trace.elimination["remaining_q_variables"], [])
        self.assertEqual(
            trace.elimination["root_residual_coefficients"],
            {"P": 1, "p_1": -1, "p_2": -1, "p_3": -1, "p_4": -1},
        )

        steps = trace.elimination["steps"]
        self.assertEqual(len(steps), 5)
        self.assertTrue(all(item["q_coefficient"] in (-1, 1) for item in steps))
        self.assertTrue(all(item["abs_4d_jacobian"] == "1" for item in steps))
        self.assertTrue(all(item["current_equation"] for item in steps))
        self.assertTrue(all(item["substitution"] for item in steps))
        self.assertEqual(steps[-1]["remaining_q_variables"], [])

        transfers = trace.q_shell_transfers
        self.assertEqual(len(transfers), 5)
        self.assertTrue(all(item["support_match"] for item in transfers))
        self.assertEqual(
            {tuple(item["recorded_particle_ids"]) for item in transfers},
            {tuple(item) for item in card.independent_variables},
        )
        self.assertTrue(all(item["source_q_shell_delta"] for item in transfers))
        self.assertTrue(all(item["positive_energy_transfer"] for item in transfers))
        self.assertTrue(all(item["global_support_uses_root_equation"] for item in transfers))
        self.assertEqual(len(trace.remaining_constraints), 6)
        self.assertEqual(len(trace.remaining_constraints_latex), 6)
        self.assertGreaterEqual(len(trace.rule_steps), 12)
        json.dumps(trace.to_dict(), ensure_ascii=False, allow_nan=False)

        payload = trace.to_dict()
        _replay_integer_tree_elimination(payload)
        damaged = deepcopy(payload)
        damaged_after = damaged["elimination"]["steps"][0]["equations_after"][0]["coefficients"]
        damaged_after["_forged_term"] = 1
        with self.assertRaisesRegex(AssertionError, "equations_after"):
            _replay_integer_tree_elimination(damaged)

    def test_d0_cancellation_and_normalization_ledger_for_both_conventions(self):
        jing = build_tree_covariant_trace(_d5_card("jing_psf"))
        standard = build_tree_covariant_trace(_d5_card("standard_2pi4"))

        self.assertEqual(jing.cancellation["retained_external_measure_count"], 4)
        self.assertEqual(jing.cancellation["retained_q_measure_count"], 5)
        canceled = jing.cancellation["cancelled_measures"]
        self.assertGreater(len(canceled), 0)
        self.assertTrue(all(item["locator"]["kind"] == "internal" for item in canceled))
        self.assertEqual(jing.normalization["net_residual_power_including_four_external_measures"], -12)
        self.assertEqual(jing.normalization["root_global_delta4_prefactor_power"], 0)
        self.assertEqual(standard.normalization["net_residual_power_including_four_external_measures"], -8)
        self.assertEqual(standard.normalization["root_global_delta4_prefactor_power"], 4)
        self.assertTrue(standard.normalization["standard_2pi4_global_factor_applied_once"])
        self.assertEqual(
            standard.normalization["per_tree_integration_balance"]["net_power_per_tree_integration"],
            0,
        )

    def test_covariant_dispatch_builds_only_scoped_d5_loop3_and_keeps_pending_others(self):
        card = _d5_card()
        raw = build_f0_expression(card)
        result = reduce_covariant(card, raw)
        self.assertEqual(result.status, "built")
        self.assertEqual(len(result.steps), 13)
        self.assertIn("not a verified full formula", " ".join(result.diagnostics))

        artifact = generate_formula(card, run_benchmark=False)
        self.assertEqual(artifact.statuses["covariant"], "built")
        self.assertEqual(artifact.statuses["direct"], "built")
        self.assertEqual(artifact.statuses["benchmark"], "pending")
        self.assertEqual(artifact.statuses["source"], "pending")
        self.assertFalse(artifact.direct_verified)

        wrong_scope = _d5_card()
        wrong_scope = replace(wrong_scope, loop_count=2)
        with self.assertRaises(ValueError):
            reduce_covariant(wrong_scope)

        wrong_raw = replace(raw, expression=raw.expression + " * corrupted")
        with self.assertRaisesRegex(ValueError, "does not belong"):
            reduce_covariant(card, wrong_raw)


if __name__ == "__main__":
    unittest.main()
