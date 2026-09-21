"""GD-02: bounded Graphviz invocation and JSON layout parsing."""
from __future__ import annotations
from dataclasses import dataclass
import json, os, shutil, subprocess, tempfile
from typing import Any

from .records import DrawingInput, EdgeRef

class LayoutError(RuntimeError): pass

@dataclass(frozen=True, slots=True)
class LayoutResult:
    nodes: dict[str, tuple[float, float]]
    paths: dict[EdgeRef, tuple[tuple[float, float], ...]]
    dot_source: str
    raw_json: dict[str, Any]
    # The subprocess input is temporary; this is deliberately None after the
    # call.  ``dot_source`` is the durable reproducibility record.
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
    home = os.path.expanduser("~")
    return next((p for p in (
        os.path.join(home, "Graphviz-16.1.0", "bin", "dot.exe"),
        r"C:\Program Files\Graphviz\bin\dot.exe",
        r"C:\Program Files (x86)\Graphviz2.38\bin\dot.exe",
    ) if os.path.isfile(p)), None)

def _parse_points(value: str | None) -> tuple[tuple[float, float], ...]:
    """Parse ordinary points from a Graphviz ``pos`` value.

    Edge positions use marker-prefixed points such as ``e,90,26`` for the
    head endpoint.  Those markers are intentionally left out here; callers
    that need the endpoint must parse it explicitly so it cannot be confused
    with a cubic control point.
    """
    if not value:
        return ()
    out = []
    for token in value.replace("\\", " ").split():
        if "," not in token or token.startswith(("e,", "s,")):
            continue
        try:
            x, y = token.split(",", 1)
            out.append((float(x), float(y)))
        except ValueError:
            pass
    return tuple(out)


def _parse_endpoint(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    for token in value.replace("\\", " ").split():
        if not token.startswith("e,"):
            continue
        try:
            x, y = token[2:].split(",", 1)
            return (float(x), float(y))
        except ValueError:
            return None
    return None

def _edge_pos(item: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    endpoint = _parse_endpoint(item.get("pos"))
    drawn = []
    for op in item.get("_draw_", ()):
        if op.get("op") in {"b", "B"}:
            part = [(float(pair[0]), float(pair[1])) for pair in op.get("points", ())]
            if drawn and part and part[0] == drawn[-1]:
                part = part[1:]
            drawn.extend(part)
    pts = tuple(drawn)
    if not pts:
        pts = _parse_points(item.get("pos"))
    # ``e,x,y`` is the actual node-boundary endpoint.  The old renderer
    # silently dropped it, leaving a visible gap before the head node.
    if endpoint is not None and (not pts or pts[-1] != endpoint):
        pts = (*pts, endpoint)
    return pts

def _make_dot(inp: DrawingInput, rankdir: str) -> str:
    lines = ["digraph G {", f'  graph [rankdir={rankdir}, splines=true, concentrate=false, outputorder=edgesfirst, ordering=out, nodesep=0.55, ranksep=0.8];',
             '  node [shape=circle, width=0.22, height=0.22, fixedsize=true, label=""];']
    for v in inp.topology.vertices: lines.append(f'  "v{v}";')
    lines += ['  "initial" [shape=point, width=0.04];']
    for edge_no, edge in enumerate(inp.edges):
        if edge.ref.kind == "internal": src, dst = f'"v{edge.source}"', f'"v{edge.target}"'
        elif edge.ref.kind == "external": src, dst = f'"v{edge.source}"', f'"x{edge.ref.vertex}_{edge.ref.slot}"'
        else: src, dst = '"initial"', f'"v{edge.target}"'
        if edge.ref.kind == "external": lines.append(f'  {dst} [shape=point, width=0.04, height=0.04];')
        if edge.ref.kind == "internal":
            # Double lines carry the visual backbone.  Their larger weight
            # makes dot keep the physical vertices on one LR baseline while
            # still allowing single-line arcs to route above/below it.
            weight = 100 if edge.edge_type == "double" else 25
            lines.append(f'  {src} -> {dst} [id="e{edge_no}", key="{edge_no}", weight={weight}, minlen=1];')
        elif edge.ref.kind == "external":
            lines.append(f'  {src} -> {dst} [id="e{edge_no}", key="{edge_no}", weight=1, minlen=1];')
        else:
            lines.append(f'  {src} -> {dst} [id="e{edge_no}", key="{edge_no}", weight=100, minlen=1];')
    lines.append("}")
    return "\n".join(lines) + "\n"

def layout_graph(inp: DrawingInput, *, dot_path: str | None = None, rankdir: str = "LR",
                 timeout: float = 60.0, workdir: str | None = None) -> LayoutResult:
    exe = _dot_executable(dot_path)
    if not exe: raise LayoutError("dependency_error: Graphviz dot executable was not found")
    dot = _make_dot(inp, rankdir)
    with tempfile.TemporaryDirectory(prefix="psf-drawing-", dir=workdir) as tmp:
        src = os.path.join(tmp, "graph.dot")
        with open(src, "w", encoding="utf-8") as handle:
            handle.write(dot)
        try:
            proc = subprocess.run([exe, "-Tjson", src], capture_output=True, text=True,
                                  encoding="utf-8", check=False, shell=False, timeout=timeout)
            ver = subprocess.run([exe, "-V"], capture_output=True, text=True, shell=False, timeout=10)
        except subprocess.TimeoutExpired as exc: raise LayoutError("timeout: dot exceeded limit") from exc
        except OSError as exc: raise LayoutError(f"dependency_error: cannot execute dot: {exc}") from exc
        if proc.returncode: raise LayoutError(f"layout_error: {proc.stderr.strip()}")
        try: raw = json.loads(proc.stdout)
        except json.JSONDecodeError as exc: raise LayoutError("layout_error: invalid dot JSON") from exc
        nodes: dict[str, tuple[float, float]] = {}
        for item in raw.get("objects", ()):
            if item.get("pos"):
                p = _parse_points(item["pos"])
                if p: nodes[str(item.get("name"))] = p[0]
        paths: dict[EdgeRef, tuple[tuple[float, float], ...]] = {}
        for item in raw.get("edges", ()):
            # Graphviz versions differ: JSON may expose the DOT id as `id`
            # or retain it as `name`.  Never rely on edge array order.
            name = str(item.get("id", item.get("name", "")))
            if not name.startswith("e"): continue
            try: idx = int(name[1:])
            except ValueError: continue
            if idx < len(inp.edges): paths[inp.edges[idx].ref] = _edge_pos(item)
        return LayoutResult(nodes, paths, dot, raw, None, (ver.stdout or ver.stderr).strip())
