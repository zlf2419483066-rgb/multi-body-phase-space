import unittest
from dataclasses import replace

from phase_space_graph.drawing import (
    LayoutResult,
    build_scene,
    prepare_drawing,
    validate_drawing,
)
from phase_space_graph.drawing.geometry import NODE_RADIUS, POINT_RADIUS
from phase_space_graph.drawing.validation import ENDPOINT_TOLERANCE

from im_fixtures import fixture_a


class DrawingGeometryValidationTests(unittest.TestCase):
    @staticmethod
    def _scene():
        inp = prepare_drawing(fixture_a().record)
        nodes = {
            "v0": (0.0, 0.0),
            "v1": (80.0, 0.0),
            "v2": (160.0, 0.0),
            "initial": (-50.0, 0.0),
            "x0_0": (-20.0, -35.0),
            "x1_0": (100.0, -35.0),
            "x2_0": (180.0, -35.0),
            "x2_1": (180.0, 35.0),
        }
        paths = {}
        for edge in inp.edges:
            source = nodes["initial"] if edge.source is None else nodes[f"v{edge.source}"]
            target = (
                nodes[f"v{edge.target}"]
                if edge.target is not None
                else nodes[f"x{edge.ref.vertex}_{edge.ref.slot}"]
            )
            paths[edge.ref] = (source, target)
        layout = LayoutResult(nodes, paths, "", {}, None, "test")
        return inp, build_scene(inp, layout)

    def test_center_to_center_fixture_paths_are_accepted(self):
        inp, scene = self._scene()
        report = validate_drawing(inp, scene)
        self.assertTrue(report.valid, report.issues)

    def test_reversed_path_fails_even_when_stroke_metadata_is_unchanged(self):
        inp, scene = self._scene()
        stroke = scene.strokes[0]
        bad = replace(stroke, points=tuple(reversed(stroke.points)))
        report = validate_drawing(inp, replace(scene, strokes=(bad,) + scene.strokes[1:]))
        self.assertFalse(report.geometry_ok)
        self.assertTrue(any("endpoint_source_mismatch" in issue for issue in report.issues))
        self.assertTrue(any("endpoint_target_mismatch" in issue for issue in report.issues))

    def test_swapped_routes_fail_endpoint_validation(self):
        inp, scene = self._scene()
        first, second = scene.strokes[:2]
        swapped = (
            replace(first, points=second.points),
            replace(second, points=first.points),
            *scene.strokes[2:],
        )
        report = validate_drawing(inp, replace(scene, strokes=swapped))
        self.assertFalse(report.geometry_ok)
        self.assertGreaterEqual(
            sum("endpoint_" in issue for issue in report.issues),
            2,
        )

    def test_endpoint_rounding_inside_radius_plus_tolerance_is_accepted(self):
        inp, scene = self._scene()
        stroke = scene.strokes[0]
        rounded = replace(
            stroke,
            points=((NODE_RADIUS + 1.5, 0.0), (80.0 - NODE_RADIUS - 1.5, 0.0)),
        )
        report = validate_drawing(inp, replace(scene, strokes=(rounded,) + scene.strokes[1:]))
        self.assertTrue(report.valid, report.issues)

    def test_endpoint_beyond_radius_plus_tolerance_is_rejected(self):
        inp, scene = self._scene()
        stroke = scene.strokes[0]
        distant = replace(
            stroke,
            points=((NODE_RADIUS + ENDPOINT_TOLERANCE + 0.1, 0.0), (80.0, 0.0)),
        )
        report = validate_drawing(inp, replace(scene, strokes=(distant,) + scene.strokes[1:]))
        self.assertFalse(report.geometry_ok)
        self.assertTrue(any("endpoint_source_mismatch" in issue for issue in report.issues))

    def test_external_endpoint_uses_small_auxiliary_radius(self):
        inp, scene = self._scene()
        index = next(i for i, edge in enumerate(inp.edges) if edge.ref.kind == "external")
        edge = inp.edges[index]
        source = scene.nodes[f"v{edge.source}"]
        target = scene.nodes[f"x{edge.ref.vertex}_{edge.ref.slot}"]
        dx, dy = target[0] - source[0], target[1] - source[1]
        length = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / length, dy / length
        stroke = scene.strokes[index]
        accepted = replace(
            stroke,
            points=(source, (target[0] - (POINT_RADIUS + 1.5) * ux,
                             target[1] - (POINT_RADIUS + 1.5) * uy)),
        )
        accepted_scene = replace(
            scene,
            strokes=scene.strokes[:index] + (accepted,) + scene.strokes[index + 1:],
        )
        self.assertTrue(validate_drawing(inp, accepted_scene).valid)

        rejected = replace(
            stroke,
            points=(source, (target[0] - (POINT_RADIUS + ENDPOINT_TOLERANCE + 0.1) * ux,
                             target[1] - (POINT_RADIUS + ENDPOINT_TOLERANCE + 0.1) * uy)),
        )
        rejected_scene = replace(
            scene,
            strokes=scene.strokes[:index] + (rejected,) + scene.strokes[index + 1:],
        )
        report = validate_drawing(inp, rejected_scene)
        self.assertFalse(report.geometry_ok)
        self.assertTrue(any("endpoint_target_mismatch" in issue for issue in report.issues))

    def test_missing_node_and_duplicate_stroke_are_rejected(self):
        inp, scene = self._scene()
        missing = replace(scene, nodes={name: point for name, point in scene.nodes.items() if name != "v1"})
        missing_report = validate_drawing(inp, missing)
        self.assertFalse(missing_report.geometry_ok)
        self.assertIn("missing_node:v1", missing_report.issues)

        duplicate = replace(scene, strokes=scene.strokes + (scene.strokes[0],))
        duplicate_report = validate_drawing(inp, duplicate)
        self.assertFalse(duplicate_report.semantic_ok)
        self.assertIn("stroke_count_mismatch", duplicate_report.issues)
        self.assertTrue(any("duplicate_stroke_reference" in issue for issue in duplicate_report.issues))


if __name__ == "__main__":
    unittest.main()
