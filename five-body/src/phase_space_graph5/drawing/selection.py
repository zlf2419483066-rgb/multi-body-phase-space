"""Explicit drawing selection; this module never searches candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .records import DrawingInput


@dataclass(frozen=True, slots=True)
class DrawingSelection:
    inputs: tuple[DrawingInput, ...]
    scope: str = "explicit"
    marker_scope: str = "selected"
    topology_scope: str = "explicit"

    def __post_init__(self) -> None:
        if self.scope not in {"current", "explicit", "catalog_subset"}:
            raise ValueError("unsupported drawing selection scope")
        if self.marker_scope not in {"one", "specified", "all_stored", "selected"}:
            raise ValueError("unsupported marker selection scope")
        if self.topology_scope not in {"explicit", "enumerated", "catalog_subset", "fixture"}:
            raise ValueError("unsupported topology selection scope")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if not all(isinstance(item, DrawingInput) for item in self.inputs):
            raise TypeError("drawing selection must contain DrawingInput objects")


def select_inputs(inputs: Iterable[DrawingInput], *, scope: str = "explicit",
                  marker_scope: str = "selected", topology_scope: str = "explicit") -> DrawingSelection:
    return DrawingSelection(tuple(inputs), scope, marker_scope, topology_scope)


__all__ = ["DrawingSelection", "select_inputs"]
