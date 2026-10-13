"""Focused numerical regression checks for the Yu2022 Tables 5--26 audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import unittest


DEMO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DEMO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase_space_graph.drawing.archive import _record_from_dict  # noqa: E402
from phase_space_graph.formula.benchmark import _default_points  # noqa: E402
from phase_space_graph.formula.d5_benchmark import _momenta  # noqa: E402
from phase_space_graph.formula.inputs import prepare_formula_input  # noqa: E402
from phase_space_graph.formula.records import FormulaOptions  # noqa: E402
from phase_space_graph.formula.reference import build_bridge_formula, build_reference_chart  # noqa: E402
from phase_space_graph.formula.yu2022_all_tables import (  # noqa: E402
    TABLES,
    _jacobian,
    _safe_expression,
    table_coordinates,
    table_density,
)
from phase_space_graph.formula.yu2022_check import (  # noqa: E402
    table5_density,
    table7_density,
)


MASSES = (1.0, 1.5, 0.8, 1.2)
TOTAL_MASS = 6.0
TOLERANCE = 5.0e-6
RUN04_FIRST_EVIDENCE = (
    DEMO_ROOT / "文档" / "验证与测试" / "误差分析" / "四体全圈143类公式_2026-10-09"
    / "run04" / "L0-C001_L0-T01.evidence.json"
)


def _relative_residual(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1.0e-300)


def _dimensionless_reference_point(point: dict[str, float]) -> tuple[float, ...]:
    return (
        point["s12"] / TOTAL_MASS**2,
        point["s34"] / TOTAL_MASS**2,
        point["theta0"], point["phi0"],
        point["theta12"], point["phi12"],
        point["theta34"], point["phi34"],
    )


def _table21_c13_from_first_three(
    coordinate_values: tuple[float, ...], masses: tuple[float, ...], total_mass: float,
) -> float:
    """Rebuild c13 from s13, s24 and the Q13-rest-frame cosine c_1_13."""

    s13 = coordinate_values[0] * total_mass**2
    s24 = coordinate_values[1] * total_mass**2
    c_1_13 = coordinate_values[2]
    m1, _m2, m3, _m4 = masses

    def kallen(first: float, second: float, third: float) -> float:
        return first**2 + second**2 + third**2 - 2.0 * (
            first * second + first * third + second * third
        )

    root_s13 = math.sqrt(s13)
    pair_energy = (total_mass**2 + s13 - s24) / (2.0 * total_mass)
    pair_momentum = math.sqrt(kallen(total_mass**2, s13, s24)) / (2.0 * total_mass)
    beta = pair_momentum / pair_energy
    gamma = pair_energy / root_s13

    k1_star = math.sqrt(kallen(s13, m1**2, m3**2)) / (2.0 * root_s13)
    e1_star = (s13 + m1**2 - m3**2) / (2.0 * root_s13)
    e3_star = (s13 + m3**2 - m1**2) / (2.0 * root_s13)
    # The table's starred cosine is defined with a minus sign relative to the
    # parent-COM Q13 axis.  Boosting Q13 to the parent frame gives these energies.
    e1 = gamma * (e1_star - beta * k1_star * c_1_13)
    e3 = gamma * (e3_star + beta * k1_star * c_1_13)
    k1 = math.sqrt(e1**2 - m1**2)
    k3 = math.sqrt(e3**2 - m3**2)
    p1_dot_p3 = e1 * e3 - 0.5 * (s13 - m1**2 - m3**2)
    return p1_dot_p3 / (k1 * k3)


class Yu2022AllTablesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with RUN04_FIRST_EVIDENCE.open("r", encoding="utf-8") as handle:
            evidence = json.load(handle)
        record = _record_from_dict(evidence["input_card"]["record"])
        card = prepare_formula_input(record, measure_convention="jing_psf")
        options = FormulaOptions(normalization="jing_psf")
        chart = build_reference_chart(card, options=options)
        cls.bridge = build_bridge_formula(card, chart, options=options)
        all_points = _default_points(MASSES, TOTAL_MASS)
        if all_points is None or len(all_points) != 8:
            raise RuntimeError("the first run04 mass group must provide exactly eight default points")
        cls.points = tuple(all_points[:2])

    def _momenta(self, point: dict[str, float]):
        return _momenta(
            self.bridge.numeric_momenta(point, MASSES, TOTAL_MASS),
            "Yu2022 all-tables regression event",
        )

    def test_registry_and_primary_source_specs(self):
        self.assertEqual(tuple(sorted(TABLES)), tuple(range(5, 27)))
        self.assertEqual(
            tuple(TABLES[8]["variables"]),
            ("s12", "s23", "s24", "s234", "c_3_234"),
        )
        self.assertIn("k4**2", TABLES[22]["density_without_2pi"])

    def test_table5_and_table7_density_match_legacy_and_pullback(self):
        for point in self.points:
            momenta = self._momenta(point)
            for table, legacy_density in (
                (5, table5_density(momenta, MASSES, TOTAL_MASS, normalization="jing_psf")["rho_per_mirror"]),
                (7, table7_density(momenta, MASSES, TOTAL_MASS, normalization="jing_psf")["rho_per_mirror"]),
            ):
                with self.subTest(point=point, table=table):
                    density = table_density(
                        table, momenta, MASSES, TOTAL_MASS, normalization="jing_psf",
                    )["density"]
                    self.assertLessEqual(_relative_residual(density, legacy_density), TOLERANCE)

                    coordinates = table_coordinates(table, momenta, MASSES, TOTAL_MASS)
                    jacobian = _jacobian(
                        _dimensionless_reference_point(point), 2.0e-5,
                        bridge=self.bridge, table=table, variant="row_variables",
                        masses=MASSES, total_mass=TOTAL_MASS, mirror=1,
                    )
                    source_sine = (
                        math.sin(point["theta0"])
                        * math.sin(point["theta12"])
                        * math.sin(point["theta34"])
                    )
                    external_pullback = (
                        density
                        * math.sin(coordinates["euler_angles"]["theta"])
                        * jacobian["real_jacobian_absolute"]
                    )
                    reference_pullback = (
                        self.bridge.numeric_density(point, MASSES, TOTAL_MASS) * source_sine
                    )
                    self.assertTrue(math.isfinite(external_pullback))
                    self.assertTrue(math.isfinite(reference_pullback))
                    self.assertLessEqual(
                        _relative_residual(external_pullback, reference_pullback), TOLERANCE,
                    )

    def test_all_table_coordinate_variants_are_finite_eight_vectors(self):
        for point in self.points:
            momenta = self._momenta(point)
            for table_number, spec in sorted(TABLES.items()):
                variants = ["row_variables"]
                if "alternative_variables" in spec:
                    variants.append("alternative_variables")
                for variant in variants:
                    with self.subTest(point=point, table=table_number, variant=variant):
                        result = table_coordinates(
                            table_number, momenta, MASSES, TOTAL_MASS, variant=variant,
                        )
                        values = result["coordinate_values"]
                        self.assertEqual(len(values), 8)
                        self.assertEqual(len(result["coordinate_names"]), 8)
                        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_table21_c13_is_analytically_rebuilt_from_first_three_coordinates(self):
        for point in self.points:
            result = table_coordinates(21, self._momenta(point), MASSES, TOTAL_MASS)
            rebuilt = _table21_c13_from_first_three(
                result["coordinate_values"], MASSES, TOTAL_MASS,
            )
            event_value = result["external_values"]["c13"]
            self.assertTrue(math.isfinite(rebuilt))
            self.assertLessEqual(abs(rebuilt - event_value), TOLERANCE)

    def test_table26_retains_transcribed_source_factor_and_conflicts_are_not_pass(self):
        point = self.points[0]
        source_expression = TABLES[26]["density_without_2pi"]
        result = table_density(26, self._momenta(point), MASSES, TOTAL_MASS)
        self.assertEqual(result["source_expression"], source_expression)
        self.assertIn("sin(theta12)**2", source_expression)
        # A candidate sin(theta13)/sin(theta12) rewrite is not part of the
        # transcribed Table 26 A expression and must not replace that source.
        self.assertNotIn("sin(theta13)", source_expression)

        for table_number in (21, 26):
            density_result = table_density(
                table_number, self._momenta(point), MASSES, TOTAL_MASS,
            )
            self.assertNotIn("status", density_result)
            self.assertNotIn("source_status", density_result)
        for table_number in (23, 24):
            conflict = TABLES[table_number].get("source_measure_conflict", "")
            self.assertIn("without automatic acceptance", conflict)
            self.assertNotIn("pass", conflict.lower())

    def test_safe_expression_rejects_unapproved_syntax(self):
        with self.assertRaises(ValueError):
            _safe_expression("__import__('os').system('whoami')", {})
        with self.assertRaises(ValueError):
            _safe_expression("open('not-a-real-file')", {})

    def test_both_cli_help_paths_import_without_running_comparisons(self):
        scripts = (
            DEMO_ROOT / "examples" / "check_yu2022_all_tables.py",
            DEMO_ROOT / "examples" / "link_yu2022_all_tables_catalog.py",
        )
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    cwd=DEMO_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
