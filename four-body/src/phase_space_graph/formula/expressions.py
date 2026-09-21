"""Small typed expression nodes for auditable formula snippets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ExpressionNode:
    kind: str
    value: str | None = None
    children: tuple["ExpressionNode", ...] = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("expression node kind cannot be empty")
        object.__setattr__(self, "children", tuple(self.children))

    def render(self) -> str:
        if self.kind in {"symbol", "differential", "delta", "theta", "constant"}:
            return self.value or self.kind
        if self.kind == "product":
            return " * ".join(x.render() for x in self.children) or "1"
        if self.kind == "sum":
            return " + ".join(x.render() for x in self.children) or "0"
        if self.kind == "cancel_copies":
            return "CancelCopies(" + ", ".join(x.render() for x in self.children) + ")"
        return self.value or "(" + ", ".join(x.render() for x in self.children) + ")"

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "value": self.value, "children": [x.to_dict() for x in self.children]}


def symbol(value: str) -> ExpressionNode:
    return ExpressionNode("symbol", value)


def differential(value: str) -> ExpressionNode:
    return ExpressionNode("differential", value)


def delta(value: str) -> ExpressionNode:
    return ExpressionNode("delta", value)


def product(nodes: Iterable[ExpressionNode]) -> ExpressionNode:
    return ExpressionNode("product", children=tuple(nodes))


__all__ = ["ExpressionNode", "symbol", "differential", "delta", "product"]
