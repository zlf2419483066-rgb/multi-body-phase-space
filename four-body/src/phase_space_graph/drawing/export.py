"""GD-06: safe single-artifact export with an auditable manifest."""
from __future__ import annotations
import json
from pathlib import Path
from .geometry import DrawingScene
from .render import render_drawing

def export_drawing(scene: DrawingScene, output_dir: str | Path, *, stem: str = "diagram", fmt: str = "svg") -> Path:
    root=Path(output_dir); root.mkdir(parents=True, exist_ok=False)
    image=root/f"{stem}.{fmt}"; image.write_bytes(render_drawing(scene,fmt=fmt))
    manifest=root/f"{stem}.json"; manifest.write_text(json.dumps({"image":image.name,"scene":scene.metadata,"stroke_count":len(scene.strokes),"label_count":len(scene.labels)},ensure_ascii=False,indent=2),encoding="utf-8")
    return image
