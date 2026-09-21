"""GD-03: map layout paths into typed drawable primitives."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .records import DrawingInput, EdgeRef
from .layout import LayoutResult

# Graphviz reports positions in points.  The DOT node width is 0.22in, so a
# physical vertex has a 7.92 point radius; keeping the same value here makes
# fallback paths and Matplotlib circles meet at the same boundary.
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

def _fallback_path(inp: DrawingInput, edge, nodes):
    a_name = f"v{edge.source}" if edge.source is not None else "initial"
    b_name = (f"v{edge.target}" if edge.target is not None
              else f"x{edge.ref.vertex}_{edge.ref.slot}")
    a = nodes.get(a_name)
    b = nodes.get(b_name)
    if a is None or b is None:
        return ()
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = (dx * dx + dy * dy) ** 0.5
    if not length:
        return (a, b)
    ux, uy = dx / length, dy / length
    start = NODE_RADIUS if a_name.startswith("v") else POINT_RADIUS
    end = NODE_RADIUS if b_name.startswith("v") else POINT_RADIUS
    if length <= start + end:
        return (a, b)
    return ((a[0] + start * ux, a[1] + start * uy),
            (b[0] - end * ux, b[1] - end * uy))


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
    return (start,
            (start[0] + dx / 3.0, start[1] + dy / 3.0),
            (start[0] + 2.0 * dx / 3.0, start[1] + 2.0 * dy / 3.0),
            end)


def _straighten_single_double_edges(inp: DrawingInput, paths, nodes):
    """Keep one double edge on a safe LR chord as the visual backbone.

    This is a generic layout repair: it applies only to an endpoint group with
    one double occurrence, and only when the chord misses other physical
    vertices and the same-group single route is not already on that chord.
    Particle IDs and edge references remain untouched.
    """
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
        path = paths.get(double.ref, ())
        if len(path) < 2:
            continue
        source_center = nodes.get(f"v{source}")
        target_center = nodes.get(f"v{target}")
        if source_center is None or target_center is None:
            continue
        dx, dy = target_center[0] - source_center[0], target_center[1] - source_center[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 2.0 * NODE_RADIUS:
            continue
        ux, uy = dx / length, dy / length
        start = (source_center[0] + NODE_RADIUS * ux, source_center[1] + NODE_RADIUS * uy)
        end = (target_center[0] - NODE_RADIUS * ux, target_center[1] - NODE_RADIUS * uy)
        if any(
            name.startswith("v") and int(name[1:]) not in {source, target}
            and _distance_to_segment(position, start, end) <= NODE_RADIUS + 2.0
            for name, position in nodes.items()
        ):
            continue
        single_paths = [paths.get(edge.ref, ()) for edge in edges if edge.edge_type == "single"]
        if any(
            len(single_path) >= 2
            and min(_distance_to_segment(point, start, end) for point in single_path[1:-1] or single_path) <= 2.5
            for single_path in single_paths
        ):
            continue
        paths[double.ref] = _straight_backbone_path(start, end)
        changed += 1
    return changed

def build_scene(inp: DrawingInput, layout: LayoutResult) -> DrawingScene:
    strokes=[]; labels=[]
    paths = {}
    for edge in inp.edges:
        points = layout.paths.get(edge.ref)
        if points is None or len(points) < 2:
            raise ValueError(f"missing path for {edge.ref.key()}")
        paths[edge.ref] = points
    straightened = _straighten_single_double_edges(inp, paths, layout.nodes)
    for edge in inp.edges:
        points = paths[edge.ref]
        strokes.append(Stroke(edge.ref, points, edge.source, edge.target, edge.edge_type,
                              edge.dashed, edge.particle_id, edge.particle_ids))
        text = None
        if edge.particle_id is not None: text = f"p_{{{edge.particle_id}}}"
        elif edge.particle_ids: text = "q_{" + "".join(map(str, edge.particle_ids)) + "}"
        if text:
            mid = points[len(points)//2]; labels.append((text, (mid[0], mid[1]), edge.ref))
    return DrawingScene(layout.nodes, tuple(strokes), tuple(labels), {
        "mode": inp.mode, "record_digest": inp.record_digest,
        "display_convention": inp.display_convention,
        "dot_version": layout.dot_version,
        "node_radius": NODE_RADIUS,
        "point_radius": POINT_RADIUS,
        "straightened_backbone_count": straightened,
    })
