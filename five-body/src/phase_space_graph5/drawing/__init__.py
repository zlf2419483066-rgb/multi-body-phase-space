"""Five body candidate drawing pipeline.

The package is downstream only: it consumes a validated topology or
``CandidateMomentumRecord`` and never runs labeling, independence, or G07.
"""

from .records import DrawingEdge, DrawingInput, EdgeRef, candidate_record_digest, prepare_drawing
from .layout import LayoutError, LayoutResult, layout_graph, verify_dot
from .geometry import DrawingScene, Stroke, build_scene
from .validation import ValidationReport, validate_drawing
from .render import render_drawing
from .export import export_drawing
from .selection import DrawingSelection, select_inputs

__all__ = [
    "DrawingEdge", "DrawingInput", "EdgeRef", "candidate_record_digest", "prepare_drawing",
    "LayoutError", "LayoutResult", "layout_graph", "verify_dot", "DrawingScene", "Stroke",
    "build_scene", "ValidationReport", "validate_drawing", "render_drawing", "export_drawing",
    "DrawingSelection", "select_inputs",
]
