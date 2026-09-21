import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from phase_space_graph.formula.export import FormulaSelection, export_markdown, select_artifacts


class DummyArtifact:
    def __init__(self, identity=None, *, input_card=None, direct=None, benchmark=None):
        self.identity = identity
        self.input_card = input_card
        self.input_digest = "input-digest" if input_card else None
        self.raw_expression = None
        self.covariant_result = None
        self.direct_formula = direct
        self.bridge_formula = None
        self.internal_benchmark = benchmark
        self.statuses = {"direct": "built", "benchmark": "verified"}
        self.diagnostics = ("artifact diagnostic",)

    @property
    def direct_verified(self):
        return False

    def to_dict(self):
        return {
            "identity": self.identity,
            "input_card": self.input_card,
            "statuses": self.statuses,
            "diagnostics": list(self.diagnostics),
        }


def _direct():
    branch = SimpleNamespace(
        key="pair-decay",
        root_condition="roots >= 0",
        equations=("p_i^2=m_i^2",),
        back_substitution=("p_j=P-p_i",),
        domain=("s12 > 0",),
        multiplicity=1,
        coefficient="1",
        summed=False,
    )
    return SimpleNamespace(
        profile="direct",
        target_variables=("s12",),
        angle_variables=("theta",),
        density_expression="rho",
        jacobian="J",
        status="built",
        normalization="standard_2pi4",
        differential_measure="ds12 dOmega",
        angle_definitions=("theta: polar angle",),
        domain=("s12 > 0",),
        branches=(branch,),
        diagnostics=("direct diagnostic",),
    )


class FormulaExportReviewTests(unittest.TestCase):
    def test_default_is_current_and_explicit_all_keeps_every_artifact(self):
        artifacts = (DummyArtifact({"mode": "fixed", "topology_key": [1]}), DummyArtifact({"mode": "fixed", "topology_key": [2]}))
        self.assertEqual(select_artifacts(artifacts), (artifacts[0],))
        self.assertEqual(select_artifacts(artifacts, FormulaSelection(marking="all")), artifacts)

    def test_representative_groups_by_identity_in_input_order(self):
        first = DummyArtifact({"mode": "fixed_labels", "topology_key": [1, 2]})
        duplicate = DummyArtifact({"mode": "fixed_labels", "topology_key": [1, 2]})
        other = DummyArtifact({"mode": "particle_relabeling", "topology_key": [1, 2]})
        selected = select_artifacts((first, duplicate, other), FormulaSelection(marking="representative"))
        self.assertEqual(selected, (first, other))
        with self.assertRaises(ValueError):
            select_artifacts((DummyArtifact(),), FormulaSelection(marking="representative"))
        with self.assertRaises(ValueError):
            select_artifacts((DummyArtifact({"mode": None, "topology_key": None}),), FormulaSelection(marking="representative"))

    def test_empty_explicit_selection_does_not_export_all(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "empty.md"
            with self.assertRaises(ValueError):
                export_markdown((DummyArtifact(),), destination, selection=FormulaSelection((), "all"))
            self.assertFalse(destination.exists())

    def test_current_rejects_multiple_indices_and_rejects_bool_or_float_indices(self):
        artifacts = (DummyArtifact(), DummyArtifact())
        with self.assertRaises(ValueError):
            select_artifacts(artifacts, FormulaSelection((0, 1), "current"))
        with self.assertRaises(TypeError):
            FormulaSelection((True,), "all")
        with self.assertRaises(TypeError):
            FormulaSelection((0.0,), "all")

    def test_export_collision_is_exclusive_and_preserves_old_content(self):
        artifact = DummyArtifact({"mode": "fixed", "topology_key": [1]})
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "formula.md"
            destination.write_text("old export", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_markdown((artifact,), destination, selection=FormulaSelection(marking="all"))
            self.assertEqual(destination.read_text(encoding="utf-8"), "old export")

    def test_render_contains_identity_branch_measure_source_diagnostics_and_machine_json(self):
        identity = {"mode": "fixed", "topology_key": [1, 2], "digest": "abc"}
        artifact = DummyArtifact(
            identity,
            input_card={"provenance": {"source": "fixture"}, "double_measures": [{"variable": "s12"}]},
            direct=_direct(),
        )
        with tempfile.TemporaryDirectory() as folder:
            output = export_markdown((artifact,), Path(folder) / "formula.md", selection=FormulaSelection(marking="all"))
            text = output.read_text(encoding="utf-8")
        self.assertIn("topology_key", text)
        self.assertIn("pair-decay", text)
        self.assertIn("ds12 dOmega", text)
        self.assertIn("fixture", text)
        self.assertIn("artifact diagnostic", text)
        machine = text.split("```json\n", 1)[1].split("\n```", 1)[0]
        self.assertEqual(json.loads(machine)["identity"], identity)


if __name__ == "__main__":
    unittest.main()
