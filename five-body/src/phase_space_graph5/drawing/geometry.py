"""Typed drawable primitives and conservative geometry repairs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .layout import LayoutResult
from .records import DrawingInput, EdgeRef

NODE_RADIUS = 0.22 * 72.0 / 2.0
POINT_RADIUS = 0.04 * 72.0 / 2.0


@dataclass(frozen=True, slots=True)
class Stroke:
    ref: EdgeRef
    points: tuple[tuple[float, float], ...]
    source: int | None
    target: int | None
    edge_type: str
    dashed: bool
    particle_id: int | None = None
    particle_ids: tuple[int, ...] = ()
    arrow: bool = True

    @property
    def stroke_count(self) -> int:
        return 2 if self.edge_type == "double" else 1


@dataclass(frozen=True, slots=True)
class DrawingScene:
    nodes: dict[str, tuple[float, float]]
    strokes: tuple[Stroke, ...]
    labels: tuple[tuple[str, tuple[float, float], EdgeRef], ...]
    metadata: dict[str, Any]


def _distance_to_segment(point, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if not length_sq:
        return ((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
    nearest = (start[0] + t * dx, start[1] + t * dy)
    return ((point[0] - nearest[0]) ** 2 + (point[1] - nearest[1]) ** 2) ** 0.5


def _straight_backbone_path(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    return (start, (start[0] + dx / 3.0, start[1] + dy / 3.0),
            (start[0] + 2.0 * dx / 3.0, start[1] + 2.0 * dy / 3.0), end)


def _straighten_single_double_edges(inp: DrawingInput, paths, nodes) -> int:
    groups: dict[tuple[int, int], list[Any]] = {}
    for edge in inp.edges:
        if edge.ref.kind == "internal":
            groups.setdefault((edge.source, edge.target), []).append(edge)
    changed = 0
    for (source, target), edges in groups.items():
        doubles = [edge for edge in edges if edge.edge_type == "double"]
        if len(doubles) != 1:
            continue
        double = doubles[0]
        if len(paths.get(double.ref, ())) < 2:
            continue
        source_center, target_center = nodes.get(f"v{source}"), nodes.get(f"v{target}")
        if source_center is None or target_center is None:
            continue
        dx, dy = target_center[0] - source_center[0], target_center[1] - source_center[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 2.0 * NODE_RADIUS:
            continue
        ux, uy = dx / length, dy / length
        start = (source_center[0] + NODE_RADIUS * ux, source_center[1] + NODE_RADIUS * uy)
        end = (target_center[0] - NODE_RADIUS * ux, target_center[1] - NODE_RADIUS * uy)
        if any(name.startswith("v") and int(name[1:]) not in {source, target}
               and _distance_to_segment(position, start, end) <= NODE_RADIUS + 2.0
               for name, position in nodes.items()):
            continue
        if any(any(len(path) >= 2 and min(_distance_to_segment(point, start, end)
                                      for point in path[1:-1] or path) <= 2.5
                   for path in (paths.get(edge.ref, ()),))
               for edge in edges if edge.edge_type == "single"):
            continue
        paths[double.ref] = _straight_backbone_path(start, end)
        changed += 1
    return changed


def build_scene(inp: DrawingInput, layout: LayoutResult) -> DrawingScene:
    paths = {}
    for edge in inp.edges:
        points = layout.paths.get(edge.ref)
        if points is None or len(points) < 2:
            raise ValueError(f"missing path for {edge.ref.key()}")
        paths[edge.ref] = points
    straightened = _straighten_single_double_edges(inp, paths, layout.nodes)
    strokes = tuple(
        Stroke(edge.ref, paths[edge.ref], edge.source, edge.target, edge.edge_type,
               edge.dashed, edge.particle_id, edge.particle_ids)
        for edge in inp.edges
    )
    labels = []
    for stroke in strokes:
        if stroke.particle_id is not None:
            text = f"p_{{{stroke.particle_id}}}"
        elif stroke.particle_ids:
            text = "q_{" + "".join(map(str, stroke.particle_ids)) + "}"
        else:
            continue
        labels.append((text, stroke.points[len(stroke.points) // 2], stroke.ref))
    return DrawingScene(layout.nodes, strokes, tuple(labels), {
        "mode": inp.mode,
        "status_layer": "P" if inp.mode == "marked" else "S",
        "record_digest": inp.record_digest,
        "identity_key": inp.identity_key,
        "display_convention": inp.display_convention,
        "evidence_stage": inp.evidence_stage,
        "independence": "not_evaluated",
        "dot_version": layout.dot_version,
        "node_radius": NODE_RADIUS,
        "point_radius": POINT_RADIUS,
        "straightened_backbone_count": straightened,
    })


__all__ = ["NODE_RADIUS", "POINT_RADIUS", "Stroke", "DrawingScene", "build_scene"]
