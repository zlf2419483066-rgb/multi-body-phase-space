from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.fixtures import fixture_b
from phase_space_graph5.drawing import build_scene, layout_graph, prepare_drawing, render_drawing, validate_drawing, verify_dot
from phase_space_graph5.physical_labels import normalize_candidate_labeling
from phase_space_graph5.label_identity import canonicalize_candidate_labeling


class CandidateDrawingTests(unittest.TestCase):
    def test_b2_keeps_duplicate_p345_double_occurrences(self):
        topology, labeling = fixture_b(2)
        checked = normalize_candidate_labeling(topology, labeling)
        self.assertIsNotNone(checked.record)
        record = checked.record
        identity = canonicalize_candidate_labeling(record, "fixed_labels")
        drawing_input = prepare_drawing(record=record, identity=identity)
        q_labels = [edge.particle_ids for edge in drawing_input.edges if edge.edge_type == "double"]
        self.assertGreaterEqual(q_labels.count((3, 4, 5)), 2)
        dot, _ = verify_dot()
        layout = layout_graph(drawing_input, dot_path=dot)
        scene = build_scene(drawing_input, layout)
        report = validate_drawing(drawing_input, scene)
        self.assertTrue(report.valid, report.issues)
        self.assertTrue(render_drawing(scene, fmt="svg").startswith(b"<svg") or b"<svg" in render_drawing(scene, fmt="svg")[:200])
        self.assertGreater(len(render_drawing(scene, fmt="png")), 100)

    def test_marked_input_keeps_identity_and_original_vertex_metadata(self):
        topology, labeling = fixture_b(2)
        record = normalize_candidate_labeling(topology, labeling).record
        identity = canonicalize_candidate_labeling(record, "fixed_labels")
        drawing_input = prepare_drawing(record=record, identity=identity)
        self.assertEqual(drawing_input.identity_key, identity.key)
        self.assertEqual(drawing_input.metadata["identity_digest"], identity.digest)
        self.assertEqual(drawing_input.metadata["original_vertex_map"], [list(item) for item in identity.vertex_map])


if __name__ == "__main__":
    unittest.main()
