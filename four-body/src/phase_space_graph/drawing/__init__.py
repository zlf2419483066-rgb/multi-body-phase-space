"""Auditable rendering of phase-space graph records.

The package is deliberately downstream of the physical identity modules: it
never enumerates, canonicalizes, or filters a topology.
"""

from .records import DrawingInput, EdgeRef, DrawingEdge, prepare_drawing
from .layout import LayoutError, LayoutResult, layout_graph
from .geometry import DrawingScene, build_scene
from .validation import ValidationReport, validate_drawing
from .render import render_drawing
from .export import export_drawing
from .selection import DrawingSelection, select_inputs
from .archive import ArchivedIdentity, ArchiveError, load_identity_archive, select_representatives, export_representatives

__all__ = [
    "DrawingInput", "EdgeRef", "DrawingEdge", "prepare_drawing",
    "LayoutError", "LayoutResult", "layout_graph", "DrawingScene",
    "build_scene", "ValidationReport", "validate_drawing", "render_drawing",
    "export_drawing", "DrawingSelection", "select_inputs", "ArchivedIdentity", "ArchiveError",
    "load_identity_archive", "select_representatives", "export_representatives",
]
