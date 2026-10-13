"""Regression coverage for the direct four-body d=5 chart.

The chart inputs in these tests come from independently reconstructed
momenta and direct subset sums.  They are not generated from the production
affine matrix, so the forward/reconstruction round-trip is not self-certified.
"""

from __future__ import annotations

from fractions import Fraction
import math
import re
import unittest

from phase_space_graph.drawing.archive import load_identity_archive
from phase_space_graph.formula.d5_catalog import default_d5_archive_path
from phase_space_graph.formula.d5_direct import (
    D5_BASIS,
    _chart_data_from_subsets,
    _kinematics,
    _parse_labels,
    build_d5_chart_data,
    build_d5_formula,
    reconstruct_d5_momenta,
)
from phase_space_graph.formula.direct_charts import (
    build_direct_density,
    build_direct_formula,
    reconstruct_direct_momenta,
    select_direct_chart,
)
from phase_space_graph.formula.records import FormulaOptions
from phase_space_graph.formula.reference import (
    build_reference_chart,
    reconstruct_reference_momenta,
)
from phase_space_graph.formula_inputs import build_formula_input


MASSES = (0.7, 1.0, 0.8, 1.1)
TOTAL_MASS = 8.0
EULER = {"theta": 0.91, "phi": 0.47, "psi": 1.23}
_LATEX_COMMANDS = {
    "Delta", "Omega", "Phi", "Theta", "Longrightarrow", "begin", "chi", "delta",
    "det", "end", "frac", "le", "left", "mathbf", "ne", "partial", "pi", "pm",
    "pmatrix", "phi", "prod", "psi", "quad", "rho", "right", "rm", "sigma", "sin",
    "sqrt", "sum", "text", "theta",
}


def _mass_squared(momentum):
    return momentum[0] ** 2 - sum(component * component for component in momentum[1:])


def _subset_invariant(momenta, subset):
    total = tuple(sum(momenta[index - 1][axis] for index in subset) for axis in range(4))
    return _mass_squared(total)


def _pair_invariants(momenta):
    return {
        f"s{left}{right}": _subset_invariant(momenta, (left, right))
        for left in range(1, 5)
        for right in range(left + 1, 5)
    }


def _spatial_triple(left, middle, right):
    a, b, c = left[1:], middle[1:], right[1:]
    cross = (b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0])
    return sum(a[index] * cross[index] for index in range(3))


def _latex_strings(value):
    if isinstance(value, str):
        if "\\" in value:
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _latex_strings(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _latex_strings(item)


class D5DirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use only the frozen identity archive.  This does not invoke a graph
        # or labeling enumerator.
        entries = load_identity_archive(default_d5_archive_path())
        selected = []
        selected_records = []
        seen_variable_sets = set()
        for entry in entries:
            record = entry.canonical_record
            loop_count = record.topology.edge_count - len(record.topology.vertices) + 1
            if loop_count != 3 or not record.independence.independent:
                continue
            if len(record.independence.variables) != 5:
                continue
            card = build_formula_input(record, measure_convention="jing_psf")
            if card.availability != "ready":
                continue
            variable_set = tuple(sorted(card.variable_labels))
            if variable_set not in seen_variable_sets:
                seen_variable_sets.add(variable_set)
                selected.append(card)
                selected_records.append(record)
            if len(selected) == 2:
                break
        if len(selected) < 2:
            raise AssertionError("the archived d=5 identities must provide two ready, distinct five-IKV sets")
        cls.cards = tuple(selected)
        cls.records = tuple(selected_records)

    def reference_momenta(self, card=None):
        card = card or self.cards[0]
        chart = build_reference_chart(card)
        values = {
            "s12": 2.5 ** 2,
            "s34": 2.8 ** 2,
            "theta0": 0.74,
            "phi0": 0.39,
            "theta12": 1.03,
            "phi12": 2.11,
            "theta34": 0.88,
            "phi34": 4.21,
        }
        return reconstruct_reference_momenta(chart, values, MASSES, TOTAL_MASS)

    def d5_values_from_momenta(self, card, momenta):
        values = {
            label: _subset_invariant(momenta, tuple(int(char) for char in label[1:]))
            for label in card.variable_labels
        }
        return {**values, **EULER}

    def assert_event(self, momenta, masses=MASSES, total_mass=TOTAL_MASS):
        self.assertEqual(len(momenta), 4)
        for momentum, mass in zip(momenta, masses):
            self.assertGreater(momentum[0], 0.0)
            self.assertAlmostEqual(_mass_squared(momentum), mass * mass, places=8)
        total = tuple(sum(momentum[axis] for momentum in momenta) for axis in range(4))
        self.assertAlmostEqual(total[0], total_mass, places=8)
        for component in total[1:]:
            self.assertAlmostEqual(component, 0.0, places=8)

    def test_exact_affine_certificate_and_archived_variable_sets(self):
        self.assertNotEqual(set(self.cards[0].variable_labels), set(self.cards[1].variable_labels))
        for card in self.cards:
            chart = build_d5_chart_data(card)
            matrix = [[Fraction(value) for value in row] for row in chart["L"]]
            inverse = [[Fraction(value) for value in row] for row in chart["inverse_L"]]
            product = [
                [sum(matrix[row][k] * inverse[k][column] for k in range(5)) for column in range(5)]
                for row in range(5)
            ]
            self.assertEqual(product, [[Fraction(int(i == j)) for j in range(5)] for i in range(5)])
            self.assertEqual(chart["determinant_certificate"]["identity_check"], "passed exactly: L*inverse_L=I")
            self.assertEqual(chart["abs_det_L"], chart["determinant_certificate"]["det_L"].lstrip("-"))
            self.assertEqual(len(chart["mass_offset_coefficients"]), 5)
            self.assertTrue(all(set(row) == {"M^2", "m_1^2", "m_2^2", "m_3^2", "m_4^2"}
                                for row in chart["mass_offset_coefficients"]))
            self.assertIn("abs(det(L))", chart["density_expression"])
            jacobian = chart["scalar_elimination_jacobian"]
            self.assertIn("s34-(P-p_1-p_2)^2=f2+f4", jacobian["constraint_group_change"])
            self.assertIn("s24-(P-p_1-p_3)^2=f3+f5", jacobian["constraint_group_change"])
            self.assertTrue(chart["root"]["mirror_root_derivative_not_extra_jacobian"])
            self.assertIn(r"\Theta_+(H_{11})", chart["chi_open_definition_latex"])
            self.assertIn("d^5S", chart["differential_measure"])

    def test_export_latex_control_words_are_delimited(self):
        chart = build_d5_chart_data(self.cards[0])
        self.assertIn(r"\quad{}dR", chart["differential_measure_latex"])
        for expression in _latex_strings(chart):
            with self.subTest(expression=expression):
                commands = set(re.findall(r"(?<!\\)\\([A-Za-z]+)", expression))
                self.assertLessEqual(commands, _LATEX_COMMANDS)
                self.assertNotRegex(
                    expression,
                    r"\\(?:qquad|quad|enspace|thinspace|medspace|thickspace)(?=[A-Za-z])",
                )

    def test_reference_event_reconstructs_every_archived_graph_ikv(self):
        for card in self.cards:
            source = self.reference_momenta(card)
            values = self.d5_values_from_momenta(card, source)
            rebuilt = reconstruct_d5_momenta(card.variable_labels, values, MASSES, TOTAL_MASS)
            self.assert_event(rebuilt)
            for label, subset in zip(card.variable_labels, card.independent_variables):
                expected = _subset_invariant(source, subset)
                actual = _subset_invariant(rebuilt, subset)
                self.assertAlmostEqual(actual, expected, places=7, msg=label)
            for label, expected in _pair_invariants(source).items():
                self.assertAlmostEqual(_pair_invariants(rebuilt)[label], expected, places=7, msg=label)

    def test_mirror_branches_preserve_ikvs_and_flip_spatial_orientation(self):
        card = self.cards[0]
        source = self.reference_momenta(card)
        values = self.d5_values_from_momenta(card, source)
        plus = reconstruct_d5_momenta(card.variable_labels, values, MASSES, TOTAL_MASS, branch=1)
        minus = reconstruct_d5_momenta(card.variable_labels, values, MASSES, TOTAL_MASS, branch=-1)
        sequence = tuple(values[name] for name in (*card.variable_labels, "theta", "phi", "psi"))
        minus_from_sequence = reconstruct_d5_momenta(card.variable_labels, sequence, MASSES, TOTAL_MASS, branch=-1)
        self.assert_event(plus)
        self.assert_event(minus)
        self.assertEqual(minus, minus_from_sequence)
        self.assertGreater(_spatial_triple(*plus[:3]), 0.0)
        self.assertLess(_spatial_triple(*minus[:3]), 0.0)
        chart_data = build_d5_chart_data(card)
        _, info = _kinematics(chart_data, card.variable_labels, values, MASSES, TOTAL_MASS, branch=1)
        p2, p3 = plus[1], plus[2]
        k2 = math.sqrt(p2[0] ** 2 - MASSES[1] ** 2)
        k3 = math.sqrt(p3[0] ** 2 - MASSES[2] ** 2)
        c23_from_reconstructed_p = sum(p2[axis] * p3[axis] for axis in range(1, 4)) / (k2 * k3)
        self.assertAlmostEqual(info["cosines"]["c23"], c23_from_reconstructed_p, places=10)
        pair_values = _pair_invariants(source)
        for branch_momenta in (plus, minus):
            for label, expected in pair_values.items():
                self.assertAlmostEqual(_pair_invariants(branch_momenta)[label], expected, places=7, msg=label)
        for subset in card.independent_variables:
            self.assertAlmostEqual(_subset_invariant(plus, subset), _subset_invariant(minus, subset), places=7)

    def test_normalization_modes_and_two_retained_branches(self):
        card = self.cards[0]
        source = self.reference_momenta(card)
        values = self.d5_values_from_momenta(card, source)
        jing = build_d5_formula(card, FormulaOptions(normalization="jing_psf"))
        rho_jing = jing.numeric_density(values, MASSES, TOTAL_MASS)

        standard_card = build_formula_input(self.records[0], measure_convention="standard_2pi4")
        standard = build_d5_formula(standard_card, FormulaOptions(normalization="standard_2pi4"))
        rho_standard = standard.numeric_density(values, MASSES, TOTAL_MASS)
        self.assertAlmostEqual(rho_standard / rho_jing, (2.0 * math.pi) ** 4, places=10)

        summed = build_d5_formula(card, FormulaOptions(normalization="jing_psf", density_mode="summed"))
        self.assertTrue(math.isclose(
            summed.numeric_density(values, MASSES, TOTAL_MASS) / rho_jing,
            2.0,
            rel_tol=1e-14,
            abs_tol=0.0,
        ))
        self.assertEqual(len(summed.branches), 2)
        self.assertEqual({item.key for item in summed.branches}, {"sigma=+1", "sigma=-1"})
        self.assertTrue(all(item.multiplicity == 1 and item.summed for item in summed.branches))

    def test_public_direct_chart_dispatch_and_three_entry_points_agree(self):
        card = self.cards[0]
        source = self.reference_momenta(card)
        values = self.d5_values_from_momenta(card, source)
        chart = select_direct_chart(card)
        self.assertEqual(chart.profile, "direct_gram_d5")
        direct_formula = build_direct_formula(card, chart)
        density_formula = build_direct_density(card, chart)
        self.assertEqual(direct_formula.profile, density_formula.profile)
        self.assertEqual(direct_formula.target_variables, density_formula.target_variables)
        direct_momenta = reconstruct_direct_momenta(chart, values, MASSES, TOTAL_MASS, branch=-1)
        formula_momenta = direct_formula.numeric_momenta(values, MASSES, TOTAL_MASS, branch=-1)
        self.assertEqual(direct_momenta, formula_momenta)
        self.assert_event(direct_momenta)
        self.assertGreater(direct_formula.numeric_density(values, MASSES, TOTAL_MASS), 0.0)
        self.assertEqual(direct_formula.status, "built")

    def test_nonfinite_inputs_positive_mass_threshold_energy_gram_and_euler_domain(self):
        common = {"s234": 32.49, "s13": 10.0, "s24": 13.2, "s12": 20.0, "s34": 20.0, **EULER}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            reconstruct_d5_momenta(D5_BASIS, {**common, "s234": math.nan}, MASSES, TOTAL_MASS)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            reconstruct_d5_momenta(D5_BASIS, common, (0.0, *MASSES[1:]), TOTAL_MASS)
        with self.assertRaisesRegex(ValueError, "threshold boundary"):
            reconstruct_d5_momenta(D5_BASIS, common, MASSES, sum(MASSES))
        with self.assertRaisesRegex(ValueError, "E_i>0"):
            reconstruct_d5_momenta(D5_BASIS, {**common, "s234": TOTAL_MASS ** 2 + MASSES[0] ** 2}, MASSES, TOTAL_MASS)
        with self.assertRaisesRegex(ValueError, "H_11 H_22-H_12"):
            reconstruct_d5_momenta(D5_BASIS, common, MASSES, TOTAL_MASS)
        with self.assertRaisesRegex(ValueError, "theta=0 or pi"):
            reconstruct_d5_momenta(D5_BASIS, {**common, "theta": 0.0}, MASSES, TOTAL_MASS)
        with self.assertRaisesRegex(ValueError, "Euler angles require"):
            reconstruct_d5_momenta(D5_BASIS, {**common, "phi": 2.0 * math.pi}, MASSES, TOTAL_MASS)

    def test_singular_ikv_matrix_is_rejected(self):
        dependent_subsets = ((1, 2), (1, 3), (2, 3), (1, 2, 3), (2, 4))
        with self.assertRaisesRegex(ValueError, "dependent"):
            _chart_data_from_subsets(dependent_subsets)
        with self.assertRaisesRegex(ValueError, "increasing order"):
            _parse_labels(("s21", "s13", "s24", "s12", "s34"))

    def test_branch_argument_accepts_only_integer_plus_or_minus_one(self):
        source = self.reference_momenta(self.cards[0])
        values = {
            label: _subset_invariant(source, tuple(int(char) for char in label[1:]))
            for label in D5_BASIS
        }
        values.update(EULER)
        for invalid_branch in (0, 2, 1.5, True):
            with self.subTest(branch=invalid_branch):
                with self.assertRaisesRegex(ValueError, r"branch must be \+1 or -1"):
                    reconstruct_d5_momenta(D5_BASIS, values, MASSES, TOTAL_MASS, branch=invalid_branch)


if __name__ == "__main__":
    unittest.main()
