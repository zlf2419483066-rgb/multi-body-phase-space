from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.drawing import (  # noqa: E402
    LayoutError,
    LayoutResult,
    build_scene,
    layout_graph,
    prepare_drawing,
    render_drawing,
    validate_drawing,
    verify_dot,
)
from phase_space_graph5.drawing.geometry import DrawingScene, Stroke  # noqa: E402
from phase_space_graph5.drawing.records import EdgeRef  # noqa: E402
from phase_space_graph5.drawing.render import (  # noqa: E402
    _control_segments,
    _curve_path,
    _draw_status,
    _local_offset_points,
    _math_label,
    _point_and_tangent,
    _status_text,
)
from phase_space_graph5.fixtures import fixture_b  # noqa: E402
from phase_space_graph5.label_identity import canonicalize_candidate_labeling  # noqa: E402
from phase_space_graph5.physical_labels import normalize_candidate_labeling  # noqa: E402


class _Path:
    """Small Path-compatible test double for renderer geometry tests."""

    MOVETO = 1
    LINETO = 2
    CURVE4 = 4

    def __init__(self, vertices, codes):
        self.vertices = tuple(vertices)
        self.codes = tuple(codes)


@dataclass
class _Figure:
    calls: list[tuple[tuple[object, ...], dict[str, object]]]

    def text(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _candidate_scene_input():
    topology, labeling = fixture_b(2)
    checked = normalize_candidate_labeling(topology, labeling)
    if checked.record is None:
        raise AssertionError(checked.to_dict())
    identity = canonicalize_candidate_labeling(checked.record, "fixed_labels")
    return prepare_drawing(record=checked.record, identity=identity)


def _straight_layout(drawing_input):
    nodes = {f"v{vertex}": (float(vertex) * 90.0, 0.0) for vertex in drawing_input.topology.vertices}
    nodes["initial"] = (-70.0, 0.0)
    paths = {}
    for edge in drawing_input.edges:
        if edge.ref.kind == "initial":
            start = nodes["initial"]
            end = nodes[f"v{edge.target}"]
        elif edge.ref.kind == "external":
            start = nodes[f"v{edge.source}"]
            end = (start[0] + 30.0, 45.0 + 25.0 * float(edge.ref.slot or 0))
            nodes[f"x{edge.ref.vertex}_{edge.ref.slot}"] = end
        else:
            start = nodes[f"v{edge.source}"]
            end = nodes[f"v{edge.target}"]
        paths[edge.ref] = (start, end)
    return LayoutResult(nodes, paths, "", {}, None, "test")


class RendererGeometryTests(unittest.TestCase):
    def test_graphviz_cubic_controls_are_curves_and_endpoint_is_preserved(self):
        points = ((0.0, 0.0), (1.0, 3.0), (2.0, 3.0), (3.0, 0.0), (6.0, 0.0))
        segments, terminal = _control_segments(points)
        self.assertEqual(len(segments), 1)
        self.assertEqual(terminal, (6.0, 0.0))
        path = _curve_path(points, _Path)
        self.assertEqual(path.codes, (_Path.MOVETO, _Path.CURVE4, _Path.CURVE4, _Path.CURVE4, _Path.LINETO))
        self.assertEqual(path.vertices[-1], (6.0, 0.0))

    def test_multisegment_spline_has_two_cubic_segments(self):
        points = ((0.0, 0.0), (1.0, 3.0), (2.0, 3.0), (3.0, 0.0),
                  (4.0, -3.0), (5.0, -3.0), (6.0, 0.0), (9.0, 0.0))
        segments, terminal = _control_segments(points)
        self.assertEqual(len(segments), 2)
        self.assertEqual(terminal, (9.0, 0.0))
        path = _curve_path(points, _Path)
        self.assertEqual(path.codes.count(_Path.CURVE4), 6)
        self.assertEqual(path.vertices[-1], (9.0, 0.0))

    def test_local_normals_and_arclength_tangent_follow_curve(self):
        points = ((0.0, 0.0), (8.0, 12.0), (22.0, -8.0), (34.0, 4.0), (42.0, 10.0))
        point, tangent = _point_and_tangent(points, 0.5)
        self.assertTrue(point)
        self.assertGreater(tangent[0] * (points[-1][0] - points[0][0]) + tangent[1] * (points[-1][1] - points[0][1]), 0.0)
        base = _local_offset_points(points, 0.0)
        shifted = _local_offset_points(points, 2.0)
        self.assertEqual(len(base), len(shifted))
        self.assertNotAlmostEqual(shifted[0][0] - base[0][0], shifted[-1][0] - base[-1][0], places=3)

    def test_math_labels_are_strictly_record_generated(self):
        self.assertEqual(_math_label("p_{5}"), "$p_{5}$")
        self.assertEqual(_math_label("q_{345}"), "$q_{345}$")
        with self.assertRaises(ValueError):
            _math_label("candidate")


class RendererContractTests(unittest.TestCase):
    def test_candidate_scene_keeps_metadata_and_semantic_references(self):
        drawing_input = _candidate_scene_input()
        scene = build_scene(drawing_input, _straight_layout(drawing_input))
        report = validate_drawing(drawing_input, scene)
        self.assertTrue(report.valid, report.issues)
        self.assertEqual(scene.metadata["evidence_stage"], "momentum_candidate")
        self.assertIn("P: momentum candidate", _status_text(scene))
        self.assertTrue(any(stroke.dashed for stroke in scene.strokes))

    def test_status_and_dashed_legend_are_metadata_only(self):
        ref = EdgeRef("internal", edge_index=0, occurrence=0)
        scene = DrawingScene(
            nodes={},
            strokes=(Stroke(ref, ((0.0, 0.0), (1.0, 0.0)), 0, 1, "single", True),),
            labels=(),
            metadata={"evidence_stage": "momentum_candidate"},
        )
        figure = _Figure([])
        _draw_status(figure, scene)
        rendered_text = " ".join(" ".join(str(value) for value in args) for args, _kwargs in figure.calls)
        self.assertIn("momentum candidate", rendered_text)
        self.assertIn("dashed", rendered_text)

    def test_structure_only_status_does_not_claim_momentum_or_independence(self):
        scene = DrawingScene({}, (), (), {"evidence_stage": "structure_only"})
        self.assertEqual(_status_text(scene), "S: structure candidate")

    def test_missing_matplotlib_is_a_clear_dependency_error(self):
        ref = EdgeRef("internal", edge_index=0, occurrence=0)
        scene = DrawingScene({}, (Stroke(ref, ((0.0, 0.0), (1.0, 0.0)), 0, 1, "single", False),), (), {})
        with patch.dict(sys.modules, {"matplotlib": None}):
            with self.assertRaisesRegex(RuntimeError, "dependency_error: Matplotlib is required"):
                render_drawing(scene)

    def test_real_dot_scene_renders_svg_and_png_when_optional_dependencies_exist(self):
        try:
            dot, _version = verify_dot()
        except LayoutError as exc:
            self.skipTest(str(exc))
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Matplotlib is unavailable; renderer intentionally reports a dependency error")
        drawing_input = _candidate_scene_input()
        layout = layout_graph(drawing_input, dot_path=dot)
        self.assertEqual(set(layout.paths), {edge.ref for edge in drawing_input.edges})
        scene = build_scene(drawing_input, layout)
        self.assertTrue(validate_drawing(drawing_input, scene).valid)
        svg = render_drawing(scene, fmt="svg")
        png = render_drawing(scene, fmt="png")
        self.assertTrue(svg.startswith(b"<?xml") or b"<svg" in svg[:500])
        self.assertTrue(png.startswith(b"\x89PNG"))
        self.assertIn(b"momentum candidate", svg)
        self.assertIn(b"dashed", svg)


if __name__ == "__main__":
    unittest.main()
