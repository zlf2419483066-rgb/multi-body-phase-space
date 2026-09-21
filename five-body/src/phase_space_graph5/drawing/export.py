"""Single image export with a compact semantic manifest."""
from __future__ import annotations

import json
from pathlib import Path

from .geometry import DrawingScene
from .render import render_drawing


def export_drawing(scene: DrawingScene, output_dir: str | Path, *, stem: str = "diagram",
                   fmt: str = "svg", overwrite: bool = False) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    image = root / f"{stem}.{fmt}"
    if image.exists() and not overwrite:
        raise FileExistsError(image)
    image.write_bytes(render_drawing(scene, fmt=fmt))
    manifest = root / f"{stem}.json"
    if manifest.exists() and not overwrite:
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps({"image": image.name, "scene": scene.metadata,
                                   "stroke_count": len(scene.strokes), "label_count": len(scene.labels)},
                                  ensure_ascii=False, indent=2, default=list), encoding="utf-8")
    return image


__all__ = ["export_drawing"]
