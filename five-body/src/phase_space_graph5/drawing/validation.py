"""Semantic and endpoint checks for candidate drawings."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from .geometry import DrawingScene, NODE_RADIUS, POINT_RADIUS
from .records import DrawingInput

ENDPOINT_TOLERANCE = 2.0


@dataclass(frozen=True, slots=True)
class ValidationReport:
    semantic_ok: bool
    geometry_ok: bool
    issues: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.semantic_ok and self.geometry_ok

    def to_dict(self) -> dict[str, object]:
        return {"semantic_ok": self.semantic_ok, "geometry_ok": self.geometry_ok,
                "valid": self.valid, "issues": list(self.issues)}


def _finite(point) -> bool:
    try:
        return math.isfinite(float(point[0])) and math.isfinite(float(point[1]))
    except (IndexError, TypeError, ValueError):
        return False


def _node_names(edge) -> tuple[str, str]:
    source = f"v{edge.source}" if edge.source is not None else "initial"
    target = f"v{edge.target}" if edge.target is not None else f"x{edge.ref.vertex}_{edge.ref.slot}"
    return source, target


def _radius(scene, name: str) -> float:
    key = "node_radius" if name.startswith("v") else "point_radius"
    default = NODE_RADIUS if key == "node_radius" else POINT_RADIUS
    try:
        value = float(scene.metadata.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value >= 0 else default


def _near(point, node, radius: float) -> bool:
    return _finite(point) and _finite(node) and math.hypot(point[0] - node[0], point[1] - node[1]) <= radius + ENDPOINT_TOLERANCE


def validate_drawing(inp: DrawingInput, scene: DrawingScene) -> ValidationReport:
    issues: list[str] = []
    expected_refs = tuple(edge.ref for edge in inp.edges)
    actual_refs = tuple(stroke.ref for stroke in scene.strokes)
    if set(expected_refs) != set(actual_refs):
        issues.append("edge_reference_mismatch")
    for ref, count in Counter(actual_refs).items():
        if count > 1:
            issues.append(f"duplicate_stroke_reference:{ref.key()}")
    if len(expected_refs) != len(actual_refs):
        issues.append("stroke_count_mismatch")

    required = [f"v{vertex}" for vertex in inp.topology.vertices] + ["initial"]
    required.extend(f"x{edge.ref.vertex}_{edge.ref.slot}" for edge in inp.edges if edge.ref.kind == "external")
    for name in dict.fromkeys(required):
        if name not in scene.nodes:
            issues.append(f"missing_node:{name}")
        elif not _finite(scene.nodes[name]):
            issues.append(f"invalid_node_position:{name}")

    expected_labels = {}
    for edge in inp.edges:
        if edge.particle_id is not None:
            expected_labels[edge.ref] = f"p_{{{edge.particle_id}}}"
        elif edge.particle_ids:
            expected_labels[edge.ref] = "q_{" + "".join(map(str, edge.particle_ids)) + "}"
    actual_labels = {ref: text for text, _, ref in scene.labels}
    if set(expected_labels) != set(actual_labels):
        issues.append("label_reference_mismatch")
    for ref, text in expected_labels.items():
        if actual_labels.get(ref) != text:
            issues.append(f"label_mapping_mismatch:{ref.key()}")

    for expected, stroke in zip(inp.edges, scene.strokes):
        if expected.ref != stroke.ref:
            issues.append("scene_order_or_reference_mismatch")
        if len(stroke.points) < 2:
            issues.append(f"degenerate_path:{expected.ref.key()}")
        else:
            source_name, target_name = _node_names(expected)
            if source_name in scene.nodes and not _near(stroke.points[0], scene.nodes[source_name], _radius(scene, source_name)):
                issues.append(f"endpoint_source_mismatch:{expected.ref.key()}")
            if target_name in scene.nodes and not _near(stroke.points[-1], scene.nodes[target_name], _radius(scene, target_name)):
                issues.append(f"endpoint_target_mismatch:{expected.ref.key()}")
        for field, left, right in (
            ("line_type", expected.edge_type, stroke.edge_type),
            ("direction", (expected.source, expected.target), (stroke.source, stroke.target)),
            ("dash_mapping", expected.dashed, stroke.dashed),
            ("particle_mapping", (expected.particle_id, tuple(expected.particle_ids)), (stroke.particle_id, tuple(stroke.particle_ids))),
        ):
            if left != right:
                issues.append(f"{field}_mismatch:{expected.ref.key()}")
        if stroke.stroke_count != (2 if expected.edge_type == "double" else 1):
            issues.append(f"stroke_count_mismatch:{expected.ref.key()}")

    semantic_prefixes = ("edge_reference_mismatch", "duplicate_stroke_reference", "stroke_count_mismatch",
                         "scene_order_or_reference_mismatch", "label_reference_mismatch", "label_mapping_mismatch",
                         "line_type_mismatch", "direction_mismatch", "dash_mapping_mismatch", "particle_mapping_mismatch")
    geometry_prefixes = ("degenerate_path", "missing_node:", "invalid_node_position:", "endpoint_")
    semantic_ok = not any(issue.startswith(semantic_prefixes) for issue in issues)
    geometry_ok = not any(issue.startswith(geometry_prefixes) for issue in issues)
    return ValidationReport(semantic_ok, geometry_ok, tuple(issues))


__all__ = ["ENDPOINT_TOLERANCE", "ValidationReport", "validate_drawing"]
