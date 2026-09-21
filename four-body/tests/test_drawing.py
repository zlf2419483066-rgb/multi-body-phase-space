import tempfile
import unittest
import os
import shutil
from dataclasses import replace
from pathlib import Path

from phase_space_graph.drawing import (
    DrawingInput, LayoutError, LayoutResult, build_scene, layout_graph,
    prepare_drawing, render_drawing, validate_drawing,
    select_inputs,
)
from phase_space_graph.drawing.geometry import Stroke
from phase_space_graph.drawing.layout import _edge_pos
from phase_space_graph.drawing.render import _curve_path, _local_offset_points, _point_and_tangent
from im_fixtures import fixture_a


class DrawingTests(unittest.TestCase):
    def test_marked_input_preserves_edge_occurrences_and_external_slots(self):
        rec = fixture_a().record
        inp = prepare_drawing(rec)
        self.assertEqual(inp.mode, "marked")
        self.assertEqual(sum(e.edge_type == "double" for e in inp.edges), 2)
        self.assertEqual(sum(e.ref.kind == "external" for e in inp.edges), 4)
        self.assertEqual(inp.edges[-1].ref.kind, "initial")

    def test_scene_semantics_and_svg_render(self):
        inp = prepare_drawing(fixture_a().record)
        nodes = {"v0": (0., 0.), "v1": (80., 0.), "v2": (160., 0.),
                 "initial": (-50., 0.), "x0_0": (-20., -35.),
                 "x1_0": (100., -35.), "x2_0": (180., -35.), "x2_1": (180., 35.)}
        paths = {}
        for edge in inp.edges:
            a = nodes.get(f"v{edge.source}") if edge.source is not None else nodes["initial"]
            b = nodes.get(f"v{edge.target}") if edge.target is not None else nodes[f"x{edge.ref.vertex}_{edge.ref.slot}"]
            paths[edge.ref] = (a, b)
        layout = LayoutResult(nodes, paths, "", {}, "", "test")
        scene = build_scene(inp, layout)
        report = validate_drawing(inp, scene)
        self.assertTrue(report.valid, report.issues)
        data = render_drawing(scene, fmt="svg")
        self.assertTrue(data.startswith(b"<?xml") or b"<svg" in data[:500])

    def test_validation_rejects_direction_dash_and_label_changes(self):
        inp = prepare_drawing(fixture_a().record)
        nodes = {"v0": (0., 0.), "v1": (80., 0.), "v2": (160., 0.),
                 "initial": (-50., 0.), "x0_0": (-20., -35.),
                 "x1_0": (100., -35.), "x2_0": (180., -35.), "x2_1": (180., 35.)}
        paths = {}
        for edge in inp.edges:
            a = nodes.get(f"v{edge.source}") if edge.source is not None else nodes["initial"]
            b = nodes.get(f"v{edge.target}") if edge.target is not None else nodes[f"x{edge.ref.vertex}_{edge.ref.slot}"]
            paths[edge.ref] = (a, b)
        scene = build_scene(inp, LayoutResult(nodes, paths, "", {}, None, "test"))
        bad = replace(scene.strokes[0], source=scene.strokes[0].target, target=scene.strokes[0].source,
                      dashed=not scene.strokes[0].dashed, particle_id=99)
        report = validate_drawing(inp, replace(scene, strokes=(bad,) + scene.strokes[1:]))
        self.assertFalse(report.semantic_ok)
        self.assertTrue(any("direction_mismatch" in issue for issue in report.issues))
        self.assertTrue(any("particle_mapping_mismatch" in issue for issue in report.issues))

    def test_real_dot_layout_preserves_every_reference(self):
        dot = os.environ.get("PSF_GRAPHVIZ_DOT") or shutil.which("dot")
        if not dot:
            candidate = os.path.expanduser(r"~\Graphviz-16.1.0\bin\dot.exe")
            dot = candidate if os.path.isfile(candidate) else None
        if not dot:
            self.skipTest("Graphviz dot is unavailable; set PSF_GRAPHVIZ_DOT for integration testing")
        inp = prepare_drawing(fixture_a().record)
        layout = layout_graph(inp, dot_path=dot)
        self.assertEqual(set(layout.paths), {edge.ref for edge in inp.edges})
        self.assertTrue(layout.dot_source.startswith("digraph G"))
        self.assertIsNone(layout.dot_path)
        scene = build_scene(inp, layout)
        self.assertTrue(validate_drawing(inp, scene).valid)
        svg = render_drawing(scene, fmt="svg")
        self.assertIn(b"$p_{", svg)
        self.assertIn(b"$q_{", svg)

    def test_graphviz_edge_pos_keeps_head_endpoint_and_cubic_controls(self):
        dot = os.environ.get("PSF_GRAPHVIZ_DOT") or shutil.which("dot")
        if not dot:
            candidate = os.path.expanduser(r"~\Graphviz-16.1.0\bin\dot.exe")
            dot = candidate if os.path.isfile(candidate) else None
        if not dot:
            self.skipTest("Graphviz dot is unavailable; set PSF_GRAPHVIZ_DOT for integration testing")
        inp = prepare_drawing(fixture_a().record)
        layout = layout_graph(inp, dot_path=dot)
        raw_edges = {str(item.get("id")): item for item in layout.raw_json.get("edges", ())}
        for index, edge in enumerate(inp.edges):
            raw = raw_edges[f"e{index}"]
            parsed = _edge_pos(raw)
            self.assertEqual(layout.paths[edge.ref], parsed)
            self.assertGreaterEqual(len(parsed), 5)
            self.assertTrue(str(raw.get("pos", "")).startswith("e,"))
        from matplotlib.path import Path
        curve = _curve_path(layout.paths[inp.edges[0].ref], Path)
        self.assertEqual(curve.codes[0], Path.MOVETO)
        self.assertIn(Path.CURVE4, curve.codes)
        self.assertEqual(tuple(curve.vertices[-1]), layout.paths[inp.edges[0].ref][-1])

    def test_curved_direction_and_local_normal_offsets(self):
        points = ((0.0, 0.0), (8.0, 12.0), (22.0, -8.0), (34.0, 4.0), (42.0, 10.0))
        point, tangent = _point_and_tangent(points, 0.5)
        self.assertTrue(point)
        self.assertGreater(tangent[0] * (points[-1][0] - points[0][0]) +
                           tangent[1] * (points[-1][1] - points[0][1]), 0.0)
        base = _local_offset_points(points, 0.0)
        shifted = _local_offset_points(points, 2.0)
        self.assertEqual(len(base), len(shifted))
        first_delta = (shifted[0][0] - base[0][0], shifted[0][1] - base[0][1])
        last_delta = (shifted[-1][0] - base[-1][0], shifted[-1][1] - base[-1][1])
        self.assertNotAlmostEqual(first_delta[0], last_delta[0], places=3)
        self.assertNotAlmostEqual(first_delta[1], last_delta[1], places=3)

    def test_multisegment_spline_and_explicit_endpoint_are_rendered(self):
        raw = {
            "pos": "e,6,0 0,0 1,2 2,2 3,0 4,-2 5,-2 6,0",
            "_draw_": [{"op": "b", "points": [[0, 0], [1, 2], [2, 2], [3, 0],
                                                        [4, -2], [5, -2], [6, 0]]}],
        }
        points = _edge_pos(raw)
        self.assertEqual(points[-1], (6.0, 0.0))
        from matplotlib.path import Path
        curve = _curve_path(points, Path)
        self.assertEqual(sum(code == Path.CURVE4 for code in curve.codes), 6)
        midpoint, _tangent = _point_and_tangent(points, 0.75)
        self.assertGreater(midpoint[0], 3.5)

    def test_missing_layout_route_is_not_silently_fallback_drawn(self):
        inp = prepare_drawing(fixture_a().record)
        with self.assertRaises(ValueError):
            build_scene(inp, LayoutResult({}, {}, "", {}, None, "test"))

    def test_dot_physical_vertex_and_point_dimensions_match_renderer(self):
        dot = os.environ.get("PSF_GRAPHVIZ_DOT") or shutil.which("dot")
        if not dot:
            candidate = os.path.expanduser(r"~\Graphviz-16.1.0\bin\dot.exe")
            dot = candidate if os.path.isfile(candidate) else None
        if not dot:
            self.skipTest("Graphviz dot is unavailable; set PSF_GRAPHVIZ_DOT for integration testing")
        layout = layout_graph(prepare_drawing(fixture_a().record), dot_path=dot)
        objects = {str(item.get("name")): item for item in layout.raw_json.get("objects", ())}
        self.assertEqual((objects["v0"]["width"], objects["v0"]["height"]), ("0.22", "0.22"))
        self.assertEqual((objects["initial"]["width"], objects["initial"]["height"]), ("0.04", "0.04"))

    def test_marked_label_visibility_is_render_only(self):
        inp = prepare_drawing(fixture_a().record)
        nodes = {"v0": (0., 0.), "v1": (80., 0.), "v2": (160., 0.),
                 "initial": (-50., 0.), "x0_0": (-20., -35.),
                 "x1_0": (100., -35.), "x2_0": (180., -35.), "x2_1": (180., 35.)}
        paths = {}
        for edge in inp.edges:
            a = nodes.get(f"v{edge.source}") if edge.source is not None else nodes["initial"]
            b = nodes.get(f"v{edge.target}") if edge.target is not None else nodes[f"x{edge.ref.vertex}_{edge.ref.slot}"]
            paths[edge.ref] = (a, b)
        scene = build_scene(inp, LayoutResult(nodes, paths, "", {}, None, "test"))
        full = render_drawing(scene, fmt="svg")
        overview = render_drawing(scene, fmt="svg", show_internal_labels=False)
        self.assertIn(b"$q_{", full)
        self.assertNotIn(b"$q_{", overview)
        self.assertIn(b"$p_{", overview)
        self.assertEqual(len(scene.labels), 6)

    def test_missing_dot_is_dependency_error(self):
        with self.assertRaises(LayoutError):
            layout_graph(prepare_drawing(topology=fixture_a().topology, mode="topology"), dot_path="Z:\\missing\\dot.exe")

    def test_selection_is_explicit_and_does_not_enumerate(self):
        selection = select_inputs((prepare_drawing(topology=fixture_a().topology, mode="topology"),), scope="current")
        self.assertEqual(selection.scope, "current")
        self.assertEqual(len(selection.inputs), 1)


if __name__ == "__main__":
    unittest.main()
