import unittest
from types import SimpleNamespace
from unittest.mock import patch

from phase_space_graph.formula.batch import run_formula_batch


def artifact(*, benchmark="verified"):
    report = None
    if benchmark == "failed":
        report = SimpleNamespace(status="failed", diagnostics=("benchmark witness",), failures=())
    return SimpleNamespace(
        statuses={"direct": "built", "benchmark": benchmark},
        internal_benchmark=report,
        to_dict=lambda: {"statuses": {"direct": "built", "benchmark": benchmark}},
    )


class FormulaBatchReviewTests(unittest.TestCase):
    def test_max_items_does_not_consume_lookahead(self):
        consumed = []

        def records():
            for value in range(3):
                consumed.append(value)
                yield value

        with patch("phase_space_graph.formula.batch.generate_formula", return_value=artifact()):
            report = run_formula_batch(records(), max_items=1)
        self.assertEqual(consumed, [0])
        self.assertTrue(report.stopped)
        self.assertEqual(len(report.items), 1)

    def test_generation_failure_is_retained_and_stops_batch(self):
        values = [artifact(), RuntimeError("bad identity")]
        with patch("phase_space_graph.formula.batch.generate_formula", side_effect=values):
            report = run_formula_batch(("first", "second", "third"))
        self.assertEqual([item.status for item in report.items], ["built", "failed"])
        self.assertIsNone(report.items[1].artifact)
        self.assertIn("bad identity", report.items[1].error)
        self.assertTrue(report.stopped)

    def test_timeout_after_one_slow_item_does_not_start_next_item(self):
        calls = []

        def records():
            for value in (1, 2):
                calls.append(value)
                yield value

        clock = iter((0.0, 0.0, 0.0, 2.0))
        with patch("phase_space_graph.formula.batch.time.monotonic", side_effect=lambda: next(clock)):
            with patch("phase_space_graph.formula.batch.generate_formula", return_value=artifact()):
                report = run_formula_batch(records(), max_seconds=1)
        self.assertEqual(calls, [1])
        self.assertEqual(len(report.items), 1)
        self.assertEqual(report.items[0].status, "built")
        self.assertTrue(report.stopped)
        self.assertIn("max_seconds", report.stop_reason)

    def test_benchmark_failure_stops_before_next_identity(self):
        consumed = []

        def records():
            for value in (1, 2):
                consumed.append(value)
                yield value

        failed = artifact(benchmark="failed")
        with patch("phase_space_graph.formula.batch.generate_formula", side_effect=(failed, artifact())):
            report = run_formula_batch(records())
        self.assertEqual(consumed, [1])
        self.assertEqual(report.items[0].status, "failed")
        self.assertIs(report.items[0].artifact, failed)
        self.assertTrue(report.stopped)
        self.assertIn("benchmark", report.items[0].error)

    def test_max_seconds_is_finite_and_capped(self):
        for value in (-1, 541):
            with self.assertRaises(ValueError):
                run_formula_batch((), max_seconds=value)
        with self.assertRaises(ValueError):
            run_formula_batch((), max_seconds=float("inf"))
        with self.assertRaises(TypeError):
            run_formula_batch((), max_seconds=True)

    def test_failed_item_serializes_with_null_artifact(self):
        with patch("phase_space_graph.formula.batch.generate_formula", side_effect=RuntimeError("bad")):
            report = run_formula_batch(("one",))
        self.assertIsNone(report.to_dict()["items"][0]["artifact"])


if __name__ == "__main__":
    unittest.main()
