"""G08-backed graph-derived covariant tree regressions for four-body L=0..2."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import unittest

from phase_space_graph import (
    Labeling,
    Topology,
    build_formula_input,
    canonicalize_physical_labeling,
    normalize_physical_labeling,
)
from phase_space_graph.formula.raw import build_f0_expression, reduce_covariant
from phase_space_graph.formula.tree_covariant import build_tree_covariant_trace


G08_DIR = (
    Path(__file__).parents[1]
    / "文档"
    / "验证与测试"
    / "结果图表"
    / "四体图枚举"
)


@lru_cache(maxsize=3)
def _normalized_g08_witness(loop_count: int):
    """Re-normalize one persisted G08 valid witness, without enumeration."""
    audit_path = G08_DIR / f"g08_四体_l{loop_count}_串联审计.json"
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("complete") is not True:
        raise AssertionError(f"G08 audit is not complete: {audit_path}")

    for report in payload.get("labeling_reports", ()):
        topology = Topology.from_dict(report["topology"])
        for witness in report.get("valid_results", ()):
            labeling = Labeling.from_dict(witness["labeling"])
            normalized = normalize_physical_labeling(topology, labeling)
            if normalized.stage != "accepted" or normalized.record is None:
                continue
            identity = canonicalize_physical_labeling(normalized.record)
            return normalized, identity
    raise AssertionError(f"G08 audit contains no re-normalizable valid witness for L={loop_count}")


@lru_cache(maxsize=6)
def _g08_card(loop_count: int, convention: str = "jing_psf"):
    normalized, identity = _normalized_g08_witness(loop_count)
    return build_formula_input(normalized, measure_convention=convention, identity=identity)


def _normalized(poly):
    return {symbol: coefficient for symbol, coefficient in poly.items() if coefficient}


def _substitute(poly, symbol, replacement):
    result = dict(_normalized(poly))
    coefficient = result.pop(symbol, 0)
    for target, target_coefficient in _normalized(replacement).items():
        value = result.get(target, 0) + coefficient * target_coefficient
        if value:
            result[target] = value
        else:
            result.pop(target, None)
    return result


def _equations(rows):
    result = {int(row["vertex"]): _normalized(row["coefficients"]) for row in rows}
    if len(result) != len(rows):
        raise AssertionError("equation rows repeat a vertex")
    return result


def _replay_tree(trace, d):
    elimination = trace["elimination"]
    steps = elimination["steps"]
    if len(steps) != d:
        raise AssertionError(f"expected {d} executed leaf eliminations, got {len(steps)}")

    replayed = None
    for step in steps:
        before = _equations(step["equations_before"])
        if replayed is not None and replayed != before:
            raise AssertionError("an elimination after-state differs from the next before-state")
        leaf = int(step["leaf_vertex"])
        q = str(step["integrated_q"])
        leaf_equation = _normalized(step["current_equation_coefficients"])
        if before.get(leaf) != leaf_equation:
            raise AssertionError("current leaf equation differs from its before-state row")
        q_coefficient = leaf_equation.get(q, 0)
        if q_coefficient not in (-1, 1) or q_coefficient != step["q_coefficient"]:
            raise AssertionError("integrated q coefficient must be the recorded +1 or -1")

        substitution = _normalized(step["substitution_coefficients"])
        if q in substitution or _substitute(leaf_equation, q, substitution):
            raise AssertionError("recorded q solution does not zero the current leaf equation")
        expected_after = {
            vertex: _substitute(equation, q, substitution)
            for vertex, equation in before.items()
            if vertex != leaf
        }
        actual_after = _equations(step["equations_after"])
        if actual_after != expected_after:
            raise AssertionError("equations_after does not follow by exact q substitution")
        replayed = actual_after

    root = int(elimination["root_vertex"])
    expected_root = {"P": 1, **{f"p_{particle}": -1 for particle in range(1, 5)}}
    if set(replayed or {}) != {root} or replayed[root] != expected_root:
        raise AssertionError("leaf replay did not leave P-sum_i(p_i)=0 at the root")
    if _normalized(elimination["root_residual_coefficients"]) != expected_root:
        raise AssertionError("recorded root equation differs from the replay")


class AllLoopCovariantTreeTests(unittest.TestCase):
    def test_g08_l0_l1_l2_cards_execute_d_leaf_integrations(self):
        for loop_count in (0, 1, 2):
            with self.subTest(loop_count=loop_count):
                d = 2 + loop_count
                card = _g08_card(loop_count)
                trace = build_tree_covariant_trace(card)
                payload = trace.to_dict()

                self.assertEqual(card.loop_count, loop_count)
                self.assertEqual(len(card.double_measures), d)
                self.assertEqual(trace.scope["loop_count"], loop_count)
                self.assertEqual(trace.scope["d"], d)
                self.assertEqual(trace.scope["vertex_count"], d + 1)
                self.assertEqual(trace.scope["double_line_tree_edge_count"], d)
                self.assertTrue(trace.scope["double_line_tree_connected"])
                self.assertEqual(len(trace.elimination["steps"]), d)
                self.assertEqual(len(trace.q_shell_transfers), d)
                self.assertTrue(all(item["support_match"] for item in trace.q_shell_transfers))
                self.assertEqual(len(trace.remaining_constraints), d + 1)
                self.assertEqual(trace.cancellation["retained_external_measure_count"], 4)
                self.assertEqual(trace.cancellation["retained_q_measure_count"], d)
                self.assertEqual(trace.cancellation["cancelled_measure_count"], loop_count)
                self.assertEqual(
                    trace.normalization["net_residual_power_including_four_external_measures"],
                    -12,
                )
                _replay_tree(payload, d)

                raw = build_f0_expression(card)
                result = reduce_covariant(card, raw)
                self.assertEqual(result.status, "built")
                self.assertEqual(len(result.steps), 3 + 2 * d)

    def test_standard_normalization_ledger_is_generalized_over_l0_l1_l2(self):
        for loop_count in (0, 1, 2):
            with self.subTest(loop_count=loop_count):
                trace = build_tree_covariant_trace(_g08_card(loop_count, "standard_2pi4"))
                d = 2 + loop_count
                self.assertEqual(trace.scope["d"], d)
                self.assertEqual(trace.normalization["tree_integrations"], d)
                self.assertEqual(
                    trace.normalization["net_residual_power_including_four_external_measures"],
                    -8,
                )
                self.assertTrue(trace.normalization["standard_2pi4_global_factor_applied_once"])
                self.assertEqual(
                    trace.normalization["per_tree_integration_balance"]["net_power_per_tree_integration"],
                    0,
                )


if __name__ == "__main__":
    unittest.main()
