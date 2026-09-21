"""GD-05: semantic and basic geometric checks independent of renderer."""
from __future__ import annotations
from dataclasses import dataclass
import math
from collections import Counter
from .records import DrawingInput, EdgeRef
from .geometry import DrawingScene, NODE_RADIUS, POINT_RADIUS


# Graphviz stores coordinates in points.  Its node clipping and JSON rounding
# can move a path endpoint by a couple of points, so endpoint checks allow a
# small margin beyond the configured node boundary.  This is an endpoint
# attachment check only; it intentionally makes no claim about crossings or
# global collision freedom.
ENDPOINT_TOLERANCE = 2.0

@dataclass(frozen=True, slots=True)
class ValidationReport:
    semantic_ok: bool
    geometry_ok: bool
    issues: tuple[str, ...] = ()
    @property
    def valid(self): return self.semantic_ok and self.geometry_ok

def validate_drawing(inp: DrawingInput, scene: DrawingScene) -> ValidationReport:
    issues = []
    expected_refs = tuple(edge.ref for edge in inp.edges)
    expected = set(expected_refs)
    actual_refs = tuple(stroke.ref for stroke in scene.strokes)
    actual = set(actual_refs)
    if expected != actual:
        issues.append("edge_reference_mismatch")
    for ref, count in Counter(actual_refs).items():
        if count > 1:
            issues.append(f"duplicate_stroke_reference:{ref.key()}")
    if len(actual_refs) != len(expected_refs):
        issues.append("stroke_count_mismatch")

    _validate_required_nodes(inp, scene, issues)

    expected_labels = {}
    for edge in inp.edges:
        if edge.particle_id is not None:
            expected_labels[edge.ref] = f"p_{{{edge.particle_id}}}"
        elif edge.particle_ids:
            expected_labels[edge.ref] = "q_{" + "".join(map(str, edge.particle_ids)) + "}"
    actual_labels = {ref: text for text, _position, ref in scene.labels}
    if set(actual_labels) != set(expected_labels):
        issues.append("label_reference_mismatch")
    for ref, text in expected_labels.items():
        if actual_labels.get(ref) != text:
            issues.append(f"label_mapping_mismatch:{ref.key()}")

    for edge, stroke in zip(inp.edges, scene.strokes):
        if edge.ref != stroke.ref:
            issues.append("scene_order_or_reference_mismatch")
        if len(stroke.points) < 2:
            issues.append(f"degenerate_path:{edge.ref.key()}")
        else:
            _validate_path_endpoints(edge, stroke, scene, issues)
        if edge.edge_type != stroke.edge_type:
            issues.append(f"line_type_mismatch:{edge.ref.key()}")
        if edge.source != stroke.source or edge.target != stroke.target:
            issues.append(f"direction_mismatch:{edge.ref.key()}")
        if edge.dashed != stroke.dashed:
            issues.append(f"dash_mapping_mismatch:{edge.ref.key()}")
        if edge.particle_id != stroke.particle_id or tuple(edge.particle_ids) != tuple(stroke.particle_ids):
            issues.append(f"particle_mapping_mismatch:{edge.ref.key()}")
        expected_count = 2 if edge.edge_type == "double" else 1
        if stroke.stroke_count != expected_count:
            issues.append(f"stroke_count_mismatch:{edge.ref.key()}")

    semantic_prefixes = (
        "edge_reference_mismatch",
        "duplicate_stroke_reference",
        "stroke_count_mismatch",
        "scene_order_or_reference_mismatch",
        "label_reference_mismatch",
        "label_mapping_mismatch",
        "line_type_mismatch",
        "direction_mismatch",
        "dash_mapping_mismatch",
        "particle_mapping_mismatch",
    )
    semantic_ok = not any(issue.startswith(semantic_prefixes) for issue in issues)
    geometry_prefixes = (
        "degenerate_path",
        "missing_node:",
        "invalid_node_position:",
        "endpoint_",
    )
    geometry_ok = not any(issue.startswith(geometry_prefixes) for issue in issues)
    return ValidationReport(semantic_ok, geometry_ok, tuple(issues))


def _required_node_names(inp: DrawingInput) -> tuple[str, ...]:
    names = [f"v{vertex}" for vertex in inp.topology.vertices]
    names.append("initial")
    for edge in inp.edges:
        if edge.ref.kind == "external":
            names.append(f"x{edge.ref.vertex}_{edge.ref.slot}")
    return tuple(dict.fromkeys(names))


def _validate_required_nodes(inp: DrawingInput, scene: DrawingScene, issues: list[str]) -> None:
    for name in _required_node_names(inp):
        if name not in scene.nodes:
            issues.append(f"missing_node:{name}")
            continue
        point = scene.nodes[name]
        if not _finite_point(point):
            issues.append(f"invalid_node_position:{name}")


def _finite_point(point) -> bool:
    try:
        return math.isfinite(float(point[0])) and math.isfinite(float(point[1]))
    except (IndexError, TypeError, ValueError):
        return False


def _node_names_for_edge(edge) -> tuple[str, str]:
    source = f"v{edge.source}" if edge.source is not None else "initial"
    target = (f"v{edge.target}" if edge.target is not None
              else f"x{edge.ref.vertex}_{edge.ref.slot}")
    return source, target


def _radius_for_node(scene: DrawingScene, name: str) -> float:
    key = "node_radius" if name.startswith("v") else "point_radius"
    default = NODE_RADIUS if key == "node_radius" else POINT_RADIUS
    try:
        value = float(scene.metadata.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value >= 0 else default


def _endpoint_near(point, node, radius: float) -> bool:
    if not _finite_point(point) or not _finite_point(node):
        return False
    distance = math.hypot(float(point[0]) - float(node[0]),
                         float(point[1]) - float(node[1]))
    return distance <= radius + ENDPOINT_TOLERANCE


def _validate_path_endpoints(edge, stroke, scene: DrawingScene, issues: list[str]) -> None:
    source_name, target_name = _node_names_for_edge(edge)
    if source_name not in scene.nodes or target_name not in scene.nodes:
        return
    points = stroke.points
    source = scene.nodes[source_name]
    target = scene.nodes[target_name]
    if not _endpoint_near(points[0], source, _radius_for_node(scene, source_name)):
        issues.append(f"endpoint_source_mismatch:{edge.ref.key()}")
    if not _endpoint_near(points[-1], target, _radius_for_node(scene, target_name)):
        issues.append(f"endpoint_target_mismatch:{edge.ref.key()}")
