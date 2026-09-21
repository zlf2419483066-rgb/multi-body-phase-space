"""GD-06: explicit selection over already loaded drawing inputs.

This module intentionally has no catalog search. Callers provide the records
they have selected from an existing catalog or archive.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from .records import DrawingInput

@dataclass(frozen=True, slots=True)
class DrawingSelection:
    inputs: tuple[DrawingInput, ...]
    scope: str = "explicit"
    def __post_init__(self):
        if self.scope not in {"current", "explicit", "catalog_subset"}:
            raise ValueError("unsupported drawing selection scope")
        object.__setattr__(self, "inputs", tuple(self.inputs))

def select_inputs(inputs: Iterable[DrawingInput], *, scope: str = "explicit") -> DrawingSelection:
    values = tuple(inputs)
    if not all(isinstance(item, DrawingInput) for item in values):
        raise TypeError("drawing selection must contain DrawingInput objects")
    return DrawingSelection(values, scope)
