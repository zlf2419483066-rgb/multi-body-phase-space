"""Bounded Graphviz layout for candidate drawings."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any

from .records import DrawingInput, EdgeRef


class LayoutError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LayoutResult:
    nodes: dict[str, tuple[float, float]]
    paths: dict[EdgeRef, tuple[tuple[float, float], ...]]
    dot_source: str
    raw_json: dict[str, Any]
    dot_path: str | None
    dot_version: str


def _dot_executable(path: str | None) -> str | None:
    if path:
        return path
    configured = os.environ.get("PSF_GRAPHVIZ_DOT")
    if configured:
        return configured
    discovered = shutil.which("dot")
    if discovered:
        return discovered
    candidates = (
        r"C:\Program Files\Graphviz\bin\dot.exe",
        r"C:\Program Files (x86)\Graphviz2.38\bin\dot.exe",
    )
    return next((value for value in candidates if os.path.isfile(value)), None)


def verify_dot(path: str | None = None) -> tuple[str, str]:
    """Resolve ``dot`` and return its executable path and version."""

    executable = _dot_executable(path)
    if executable is None:
        raise LayoutError("dependency_error: Graphviz dot executable was not found")
    try:
        result = subprocess.run([executable, "-V"], capture_output=True, text=True, shell=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LayoutError(f"dependency_error: cannot execute dot: {exc}") from exc
    if result.returncode:
        raise LayoutError(f"dependency_error: dot -V failed: {result.stderr.strip()}")
    return executable, (result.stdout or result.stderr).strip()


def _remaining(deadline: float | None, timeout: float) -> float:
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LayoutError("timeout: candidate run deadline exceeded")
    return min(timeout, remaining)


def _parse_points(value: str | None) -> tuple[tuple[float, float], ...]:
    if not value:
        return ()
    output = []
    for token in value.replace("\\", " ").split():
        if "," not in token or token.startswith(("e,", "s,")):
            continue
        try:
            x, y = token.split(",", 1)
            output.append((float(x), float(y)))
        except ValueError:
            continue
    return tuple(output)


def _parse_endpoint(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    for token in value.replace("\\", " ").split():
        if not token.startswith("e,"):
            continue
        try:
            x, y = token[2:].split(",", 1)
            return float(x), float(y)
        except ValueError:
            return None
    return None


def _edge_pos(item: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    endpoint = _parse_endpoint(item.get("pos"))
    drawn: list[tuple[float, float]] = []
    for operation in item.get("_draw_", ()):
        if operation.get("op") in {"b", "B"}:
            points = [(float(pair[0]), float(pair[1])) for pair in operation.get("points", ())]
            if drawn and points and points[0] == drawn[-1]:
                points = points[1:]
            drawn.extend(points)
    points = tuple(drawn) or _parse_points(item.get("pos"))
    if endpoint is not None and (not points or points[-1] != endpoint):
        points = (*points, endpoint)
    return points


def _make_dot(inp: DrawingInput, rankdir: str) -> str:
    lines = [
        "digraph G {",
        f'  graph [rankdir={rankdir}, splines=true, concentrate=false, outputorder=edgesfirst, ordering=out, nodesep=0.55, ranksep=0.8];',
        '  node [shape=circle, width=0.22, height=0.22, fixedsize=true, label=""];',
    ]
    for vertex in inp.topology.vertices:
        lines.append(f'  "v{int(vertex)}";')
    lines.append('  "initial" [shape=point, width=0.04, height=0.04];')
    for edge_number, edge in enumerate(inp.edges):
        if edge.ref.kind == "internal":
            source, target = f'"v{edge.source}"', f'"v{edge.target}"'
        elif edge.ref.kind == "external":
            source, target = f'"v{edge.source}"', f'"x{edge.ref.vertex}_{edge.ref.slot}"'
            lines.append(f'  {target} [shape=point, width=0.04, height=0.04];')
        else:
            source, target = '"initial"', f'"v{edge.target}"'
        weight = 100 if edge.edge_type == "double" or edge.ref.kind == "initial" else (25 if edge.ref.kind == "internal" else 1)
        lines.append(f'  {source} -> {target} [id="e{edge_number}", key="{edge_number}", weight={weight}, minlen=1];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def layout_graph(inp: DrawingInput, *, dot_path: str | None = None, rankdir: str = "LR",
                 timeout: float = 60.0, workdir: str | None = None,
                 deadline: float | None = None) -> LayoutResult:
    executable = _dot_executable(dot_path)
    if not executable:
        raise LayoutError("dependency_error: Graphviz dot executable was not found")
    dot_source = _make_dot(inp, rankdir)
    with tempfile.TemporaryDirectory(prefix="psf5-drawing-", dir=workdir) as directory:
        source = os.path.join(directory, "graph.dot")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(dot_source)
        try:
            result = subprocess.run(
                [executable, "-Tjson", source], capture_output=True, text=True,
                encoding="utf-8", check=False, shell=False,
                timeout=_remaining(deadline, min(60.0, timeout)),
            )
            version = subprocess.run(
                [executable, "-V"], capture_output=True, text=True, shell=False,
                timeout=_remaining(deadline, 10.0),
            )
        except subprocess.TimeoutExpired as exc:
            raise LayoutError("timeout: dot exceeded limit") from exc
        except OSError as exc:
            raise LayoutError(f"dependency_error: cannot execute dot: {exc}") from exc
    if result.returncode:
        raise LayoutError(f"layout_error: {result.stderr.strip()}")
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LayoutError("layout_error: invalid dot JSON") from exc
    nodes: dict[str, tuple[float, float]] = {}
    for item in raw.get("objects", ()):
        points = _parse_points(item.get("pos"))
        if points:
            nodes[str(item.get("name"))] = points[0]
    paths: dict[EdgeRef, tuple[tuple[float, float], ...]] = {}
    for item in raw.get("edges", ()):
        name = str(item.get("id", item.get("name", "")))
        if not name.startswith("e"):
            continue
        try:
            index = int(name[1:])
        except ValueError:
            continue
        if index < len(inp.edges):
            paths[inp.edges[index].ref] = _edge_pos(item)
    return LayoutResult(nodes, paths, dot_source, raw, executable, (version.stdout or version.stderr).strip())


__all__ = ["LayoutError", "LayoutResult", "layout_graph", "verify_dot"]
