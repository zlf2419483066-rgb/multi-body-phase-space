"""Structural and independent-event coverage for the direct low-loop chart."""

from __future__ import annotations

from fractions import Fraction
import json
import math
import unittest

from phase_space_graph.formula.all_loop_catalog import build_all_loop_catalog
from phase_space_graph.formula_inputs import build_formula_input
from phase_space_graph.formula.low_loop_direct import (
    _constraint_elimination_matrix_latex,
    _constraint_matrix_latex,
    _u0_latex,
    build_low_loop_chart_data,
    build_low_loop_chart_data_from_labels,
    build_low_loop_formula,
    extract_low_loop_coordinates,
    reconstruct_low_loop_momenta,
)
from phase_space_graph.formula.reference import build_reference_chart, reconstruct_reference_momenta
from phase_space_graph.formula.records import FormulaOptions


MASSES = (0.7, 1.0, 0.8, 1.1)
TOTAL_MASS = 8.0
ANALYTIC_MASS_CASES = (
    ("unequal-set-A", (0.7, 1.0, 0.8, 1.1)),
    ("unequal-set-B", (0.55, 1.25, 0.91, 0.68)),
)
REFERENCE_VALUES = {
    "s12": 2.5 ** 2,
    "s34": 2.8 ** 2,
    "theta0": 0.74,
    "phi0": 0.39,
    "theta12": 1.03,
    "phi12": 2.11,
    "theta34": 0.88,
    "phi34": 4.21,
}


def _analytic_reference_points():
    """Eight interior reference events, with varying pair masses and angles."""
    for index in range(8):
        yield {
            "s12": (2.35 + 0.05 * index) ** 2,
            "s34": (2.65 + 0.05 * index) ** 2,
            "theta0": 0.55 + 0.09 * index,
            "phi0": 0.20 + 0.51 * index,
            "theta12": 0.62 + 0.08 * index,
            "phi12": 0.38 + 0.57 * index,
            "theta34": 0.73 + 0.05 * index,
            "phi34": 0.26 + 0.68 * index,
        }


def _analytic_two_body_momentum(parent_mass, first_mass, second_mass):
    """Hand-evaluated factorized Kallen momentum, independent of chart helpers."""
    product = (
        parent_mass - first_mass - second_mass
    ) * (
        parent_mass + first_mass + second_mass
    ) * (
        parent_mass - first_mass + second_mass
    ) * (
        parent_mass + first_mass - second_mass
    )
    if product <= 0.0:
        raise AssertionError("analytic regression event must be strictly above threshold")
    return math.sqrt(product) / (2.0 * parent_mass)


def _hand_derived_jing_density(class_id, target_values, masses, total_mass):
    """Closed-form PSF coefficients; deliberately independent of production Jacobians."""
    twopi_power = (2.0 * math.pi) ** 12
    if class_id == "L0-C001":
        h, t = target_values["s12"], target_values["s34"]
        root_h, root_t = math.sqrt(h), math.sqrt(t)
        q = _analytic_two_body_momentum(total_mass, root_h, root_t)
        ka = _analytic_two_body_momentum(root_h, masses[0], masses[1])
        kc = _analytic_two_body_momentum(root_t, masses[2], masses[3])
        return q * ka * kc / (64.0 * twopi_power * total_mass * root_h * root_t)
    if class_id == "L0-C002":
        # The fixed nested ordering is [a,b,c,d]=[3,4,2,1].
        t, h = target_values["s234"], target_values["s34"]
        root_t, root_h = math.sqrt(t), math.sqrt(h)
        q = _analytic_two_body_momentum(total_mass, root_t, masses[0])
        k = _analytic_two_body_momentum(root_t, root_h, masses[1])
        ka = _analytic_two_body_momentum(root_h, masses[2], masses[3])
        return q * k * ka / (64.0 * twopi_power * total_mass * root_t * root_h)
    if class_id == "L1-C001":
        h = target_values["s12"]
        root_h = math.sqrt(h)
        ka = _analytic_two_body_momentum(root_h, masses[0], masses[1])
        return ka / (128.0 * twopi_power * total_mass**2 * root_h)
    raise AssertionError(f"no hand-derived formula was specified for {class_id}")


def _mass_squared(momentum):
    return momentum[0] ** 2 - sum(component * component for component in momentum[1:])


def _subset_invariant(momenta, subset):
    total = tuple(sum(momenta[index - 1][axis] for index in subset) for axis in range(4))
    return _mass_squared(total)


def _identity(size):
    return [[Fraction(int(i == j)) for j in range(size)] for i in range(size)]


def _fraction_matrix(matrix):
    return [[Fraction(item) for item in row] for row in matrix]


class LowLoopDirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_all_loop_catalog(loops=(0, 1, 2))
        cls.cards_by_loop = {}
        cls.records_by_loop = {}
        for loop in (0, 1, 2):
            representatives = [item.representative_record for item in cls.catalog.classes if item.loop_count == loop]
            cls.records_by_loop[loop] = tuple(representatives)
            cls.cards_by_loop[loop] = tuple(
                build_formula_input(record, measure_convention="jing_psf") for record in representatives
            )

    def test_every_l0_l1_l2_class_has_a_jsonable_exact_chart_plan(self):
        self.assertEqual(tuple(map(len, (self.cards_by_loop[0], self.cards_by_loop[1], self.cards_by_loop[2]))), (2, 6, 29))
        for loop, cards in self.cards_by_loop.items():
            for card in cards:
                with self.subTest(loop=loop, labels=card.variable_labels):
                    data = build_low_loop_chart_data(card)
                    json.loads(json.dumps(data, ensure_ascii=False, allow_nan=False))
                    self.assertEqual(data["loop_count"], loop)
                    self.assertEqual(len(data["target_variables"]), loop + 2)
                    self.assertEqual(data["angle_variables"], data["internal_variables"] + ["theta", "phi", "psi"])
                    self.assertEqual(len(data["K"]), loop + 2)
                    self.assertEqual(abs(Fraction(data["det_K"])), Fraction(data["abs_det_K"]))
                    self.assertNotEqual(Fraction(data["D"]), 0)
                    self.assertNotEqual(Fraction(data["det_C_E_base"]), 0)
                    self.assertEqual(len(data["constraint_equations"]), loop)
                    self.assertEqual(len(data["constraint_mass_offsets"]), loop)
                    self.assertEqual(len(data["U_in_Z_coefficients"]), loop)
                    self.assertEqual(
                        data["constraint_mass_offsets"],
                        [equation["mass_offset_coefficients"] for equation in data["constraint_equations"]],
                    )
                    self.assertEqual(data["verification"]["independent_formula_check"], "pending")
                    self.assertEqual(data["verification"]["numeric_event"], "pending; chart construction does not imply an evaluated physical point")
                    k = _fraction_matrix(data["K"])
                    inverse_k = _fraction_matrix(data["inverse_K"])
                    k_product = [[sum(k[i][j] * inverse_k[j][col] for j in range(len(k))) for col in range(len(k))] for i in range(len(k))]
                    self.assertEqual(k_product, _identity(len(k)))
                    az = _fraction_matrix(data["A_Z"])
                    inv_az = _fraction_matrix(data["inverse_A_Z"])
                    az_product = [[sum(az[i][j] * inv_az[j][col] for j in range(5)) for col in range(5)] for i in range(5)]
                    self.assertEqual(az_product, _identity(5))
                    self.assertEqual(
                        data["internal_coordinate_definition"]["phi_eliminated"],
                        "v" in data["solved_columns"],
                    )
                    self.assertTrue(all(set(item["mass_offset_coefficients"].values()) == {"0"}
                                        for item in data["h_t_U"]["U"]))
                    self.assertTrue(all(set(row.values()) == {"0"}
                                        for row in data["B_mass_offset_coefficients"][2:]))
                    self.assertEqual(len(data["constraint_equations"]), len(data["U_in_Z_coefficients"]))
                    rebuilt_from_labels = build_low_loop_chart_data_from_labels(card.variable_labels)
                    self.assertEqual(rebuilt_from_labels["K"], data["K"])
                    self.assertEqual(rebuilt_from_labels["A_Z"], data["A_Z"])

    def test_independent_reference_event_round_trips_through_extracted_chart(self):
        compared = 0
        for loop, cards in self.cards_by_loop.items():
            for card in cards:
                with self.subTest(loop=loop, labels=card.variable_labels):
                    source = reconstruct_reference_momenta(build_reference_chart(card), REFERENCE_VALUES, MASSES, TOTAL_MASS)
                    data = build_low_loop_chart_data(card)
                    extracted = extract_low_loop_coordinates(data, source, MASSES, TOTAL_MASS)
                    rebuilt = reconstruct_low_loop_momenta(data, extracted["values"], MASSES, TOTAL_MASS, branch=extracted["branch"])
                    for index, (left, right) in enumerate(zip(source, rebuilt), 1):
                        with self.subTest(particle=index):
                            for axis in range(4):
                                self.assertAlmostEqual(left[axis], right[axis], places=7)
                            self.assertAlmostEqual(_mass_squared(right), MASSES[index - 1] ** 2, places=8)
                    formula = build_low_loop_formula(card)
                    density = formula.numeric_density(extracted["values"], MASSES, TOTAL_MASS, branch=extracted["branch"])
                    self.assertTrue(math.isfinite(density))
                    self.assertGreater(density, 0.0)
                    compared += 1
        self.assertEqual(compared, 37)

    def test_both_phi_elimination_modes_mirror_roots_and_normalizations(self):
        selected = {}
        for loop, cards in self.cards_by_loop.items():
            for index, card in enumerate(cards):
                data = build_low_loop_chart_data(card)
                selected.setdefault(data["elimination"]["phi_eliminated"], (loop, index, card, data))
        self.assertEqual(set(selected), {False, True})
        for phi_eliminated, (loop, index, card, data) in selected.items():
            record = self.records_by_loop[loop][index]
            source = reconstruct_reference_momenta(build_reference_chart(card), REFERENCE_VALUES, MASSES, TOTAL_MASS)
            extracted = extract_low_loop_coordinates(data, source, MASSES, TOTAL_MASS)
            jing_card = build_formula_input(record, measure_convention="jing_psf")
            standard_card = build_formula_input(record, measure_convention="standard_2pi4")
            jing = build_low_loop_formula(jing_card, FormulaOptions(normalization="jing_psf"))
            standard = build_low_loop_formula(standard_card, FormulaOptions(normalization="standard_2pi4"))
            branch = extracted["branch"]
            rho_jing = jing.numeric_density(extracted["values"], MASSES, TOTAL_MASS, branch=branch)
            rho_standard = standard.numeric_density(extracted["values"], MASSES, TOTAL_MASS, branch=branch)
            self.assertAlmostEqual(rho_standard / rho_jing, (2.0 * math.pi) ** 4, places=9)
            if phi_eliminated:
                self.assertEqual(len(jing.branches), 2)
                opposite = -branch
                other_momenta = reconstruct_low_loop_momenta(data, extracted["values"], MASSES, TOTAL_MASS, branch=opposite)
                for label in data["target_variables"]:
                    subset = tuple(int(character) for character in label[1:])
                    self.assertAlmostEqual(_subset_invariant(other_momenta, subset), extracted["target_variables"][label], places=7)
                other_density = jing.numeric_density(extracted["values"], MASSES, TOTAL_MASS, branch=opposite)
                self.assertAlmostEqual(other_density / rho_jing, 1.0, places=10)
                summed = build_low_loop_formula(jing_card, FormulaOptions(normalization="jing_psf", density_mode="summed"))
                summed_density = summed.numeric_density(extracted["values"], MASSES, TOTAL_MASS, branch=branch)
                self.assertAlmostEqual(summed_density / rho_jing, 2.0, places=10)
            else:
                self.assertEqual(len(jing.branches), 1)
                summed = build_low_loop_formula(jing_card, FormulaOptions(normalization="jing_psf", density_mode="summed"))
                summed_density = summed.numeric_density(extracted["values"], MASSES, TOTAL_MASS, branch=1)
                self.assertAlmostEqual(summed_density / rho_jing, 1.0, places=10)
                with self.assertRaisesRegex(ValueError, r"branch must be \+1"):
                    jing.numeric_density(extracted["values"], MASSES, TOTAL_MASS, branch=-1)
                with self.assertRaisesRegex(ValueError, r"branch must be \+1"):
                    jing.numeric_momenta(extracted["values"], MASSES, TOTAL_MASS, branch=True)

    def test_nonfinite_zero_mass_and_coordinate_boundaries_are_rejected(self):
        card = next(card for loop_cards in self.cards_by_loop.values() for card in loop_cards)
        data = build_low_loop_chart_data(card)
        source = reconstruct_reference_momenta(build_reference_chart(card), REFERENCE_VALUES, MASSES, TOTAL_MASS)
        values = extract_low_loop_coordinates(data, source, MASSES, TOTAL_MASS)["values"]
        with self.assertRaisesRegex(ValueError, "must be finite"):
            reconstruct_low_loop_momenta(data, {**values, data["target_variables"][0]: math.nan}, MASSES, TOTAL_MASS)
        with self.assertRaisesRegex(ValueError, "strictly positive particle masses"):
            reconstruct_low_loop_momenta(data, values, (0.0, *MASSES[1:]), TOTAL_MASS)
        with self.assertRaisesRegex(ValueError, "theta=0 or pi"):
            reconstruct_low_loop_momenta(data, {**values, "theta": 0.0}, MASSES, TOTAL_MASS)
        if not data["elimination"]["phi_eliminated"]:
            with self.assertRaisesRegex(ValueError, "internal_phi must satisfy"):
                reconstruct_low_loop_momenta(data, {**values, "internal_phi": math.pi}, MASSES, TOTAL_MASS)

        eliminated_card = next(
            candidate
            for loop_cards in self.cards_by_loop.values()
            for candidate in loop_cards
            if build_low_loop_chart_data(candidate)["elimination"]["phi_eliminated"]
        )
        eliminated_data = build_low_loop_chart_data(eliminated_card)
        eliminated_source = reconstruct_reference_momenta(
            build_reference_chart(eliminated_card), REFERENCE_VALUES, MASSES, TOTAL_MASS
        )
        eliminated_values = extract_low_loop_coordinates(
            eliminated_data, eliminated_source, MASSES, TOTAL_MASS
        )["values"]
        boundary_cosine = "x" if "x" in eliminated_data["internal_variables"] else "y"
        with self.assertRaisesRegex(ValueError, "internal cosine domain"):
            reconstruct_low_loop_momenta(
                eliminated_data,
                {**eliminated_values, boundary_cosine: 1.0},
                MASSES,
                TOTAL_MASS,
            )

    def test_latex_coefficients_omit_zero_and_unit_factors(self):
        matrix = (
            (Fraction(1), Fraction(-1), Fraction(1)),
            (Fraction(0), Fraction(2, 3), Fraction(0)),
        )
        self.assertEqual(
            _constraint_matrix_latex(matrix),
            r"C=\begin{pmatrix}j_{x1} & -j_{y1} & 1 \\ 0 & \frac{2}{3}j_{y1} & 0\end{pmatrix}",
        )
        self.assertEqual(
            _constraint_elimination_matrix_latex(matrix, (0, 1, 2)),
            r"C_E=\begin{pmatrix}j_{x1} & -j_{y1} & 1 \\ 0 & \frac{2}{3}j_{y1} & 0\end{pmatrix}",
        )
        self.assertEqual(
            _u0_latex(
                (Fraction(1), Fraction(-1), Fraction(0), Fraction(-1), Fraction(9)),
                (Fraction(1), Fraction(0), Fraction(0), Fraction(1), Fraction(1)),
            ),
            r"h-t-j_{y0}+M^2+m_3^2+m_4^2",
        )

    def test_hand_derived_closed_form_density_regressions(self):
        """Hand-derived analytic regressions, not an independent paper-level validation."""
        selected_ids = ("L0-C001", "L0-C002", "L1-C001")
        classes_by_id = {item.class_id: item for item in self.catalog.classes}
        self.assertTrue(set(selected_ids) <= set(classes_by_id))
        points = tuple(_analytic_reference_points())
        self.assertEqual(len(points), 8)

        for class_id in selected_ids:
            record = classes_by_id[class_id].representative_record
            jing_card = build_formula_input(record, measure_convention="jing_psf")
            standard_card = build_formula_input(record, measure_convention="standard_2pi4")
            data = build_low_loop_chart_data(jing_card)
            reference_chart = build_reference_chart(jing_card)
            jing_formula = build_low_loop_formula(jing_card, FormulaOptions(normalization="jing_psf"))
            standard_formula = build_low_loop_formula(
                standard_card, FormulaOptions(normalization="standard_2pi4")
            )

            if class_id == "L0-C001":
                self.assertEqual(data["target_variables"], ["s12", "s34"])
                self.assertEqual(data["geometry"]["mode"], "pair")
            elif class_id == "L0-C002":
                self.assertEqual(data["target_variables"], ["s234", "s34"])
                self.assertEqual(data["geometry"]["mode"], "nested")
                self.assertEqual(data["geometry"]["particles"], [3, 4, 2, 1])
            else:
                self.assertEqual(data["target_variables"], ["s123", "s12", "s34"])
                self.assertEqual(data["geometry"]["mode"], "pair")
                self.assertEqual(set(data["internal_variables"]), {"x", "internal_phi"})
                self.assertEqual(data["solved_columns"], ["y"])

            for mass_case, masses in ANALYTIC_MASS_CASES:
                for point_index, reference_values in enumerate(points):
                    source = reconstruct_reference_momenta(
                        reference_chart, reference_values, masses, TOTAL_MASS
                    )
                    extracted = extract_low_loop_coordinates(data, source, masses, TOTAL_MASS)
                    expected_jing = _hand_derived_jing_density(
                        class_id, extracted["target_variables"], masses, TOTAL_MASS
                    )
                    expected_standard = expected_jing * (2.0 * math.pi) ** 4
                    self.assertGreater(expected_jing, 0.0)
                    with self.subTest(
                        class_id=class_id, mass_case=mass_case, point=point_index,
                        normalization="jing_psf",
                    ):
                        actual = jing_formula.numeric_density(
                            extracted["values"], masses, TOTAL_MASS, branch=extracted["branch"]
                        )
                        self.assertAlmostEqual(actual / expected_jing, 1.0, places=8)
                    with self.subTest(
                        class_id=class_id, mass_case=mass_case, point=point_index,
                        normalization="standard_2pi4",
                    ):
                        actual = standard_formula.numeric_density(
                            extracted["values"], masses, TOTAL_MASS, branch=extracted["branch"]
                        )
                        self.assertAlmostEqual(actual / expected_standard, 1.0, places=8)

    def test_branch_scope_and_required_inputs_follow_elimination_plan(self):
        card = self.cards_by_loop[2][0]
        data = build_low_loop_chart_data(card)
        source = reconstruct_reference_momenta(build_reference_chart(card), REFERENCE_VALUES, MASSES, TOTAL_MASS)
        extracted = extract_low_loop_coordinates(data, source, MASSES, TOTAL_MASS)
        formula = build_low_loop_formula(card)
        if data["elimination"]["phi_eliminated"]:
            self.assertEqual(len(formula.branches), 2)
            self.assertEqual({item.key for item in formula.branches}, {"internal_phi_branch=+1", "internal_phi_branch=-1"})
            with self.assertRaisesRegex(ValueError, "missing low-loop chart coordinates"):
                formula.numeric_momenta({name: value for name, value in extracted["values"].items() if name != data["angle_variables"][0]}, MASSES, TOTAL_MASS)
        else:
            self.assertEqual(len(formula.branches), 1)
            self.assertEqual(formula.branches[0].key, "internal_phi=explicit-coordinate")
            with self.assertRaisesRegex(ValueError, r"branch must be \+1"):
                formula.numeric_momenta(extracted["values"], MASSES, TOTAL_MASS, branch=-1)


if __name__ == "__main__":
    unittest.main()
