"""Bounded formula batches retaining partial in-memory results."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from numbers import Real
from typing import Any, Iterable

from ..physical_labels import NormalizationResult, PhysicalLabelRecord
from .generate import generate_formula
from .records import FormulaArtifact


@dataclass(frozen=True, slots=True)
class BatchItem:
    index: int
    status: str
    artifact: FormulaArtifact | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "status": self.status,
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class FormulaBatchReport:
    items: tuple[BatchItem, ...]
    stopped: bool = False
    stop_reason: str | None = None

    @property
    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in self.items:
            result[item.status] = result.get(item.status, 0) + 1
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items], "stopped": self.stopped, "stop_reason": self.stop_reason, "counts": self.counts}


def _validate_max_items(max_items: int | None) -> int | None:
    if max_items is None:
        return None
    if isinstance(max_items, bool) or not isinstance(max_items, int):
        raise TypeError("max_items must be a non-negative integer or None")
    if max_items < 0:
        raise ValueError("max_items must be non-negative")
    return max_items


def _validate_max_seconds(max_seconds: float) -> float:
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, Real):
        raise TypeError("max_seconds must be a finite real number")
    value = float(max_seconds)
    if not math.isfinite(value):
        raise ValueError("max_seconds must be finite")
    if value < 0 or value > 540:
        raise ValueError("max_seconds must be between 0 and 540 seconds")
    return value


def _benchmark_failure(artifact: FormulaArtifact) -> str | None:
    report = artifact.internal_benchmark
    if artifact.statuses.get("benchmark") != "failed" and not (report and report.status == "failed"):
        return None
    if report is None:
        return "benchmark failed"
    details = list(report.diagnostics)
    details.extend(str(item) for item in report.failures)
    return "benchmark failed" + (": " + "; ".join(details) if details else "")


def run_formula_batch(
    records: Iterable[PhysicalLabelRecord | NormalizationResult],
    *,
    max_items: int | None = None,
    max_seconds: float = 540,
    **kwargs: Any,
) -> FormulaBatchReport:
    """Run explicit records within count and wall-clock budgets.

    The input iterator is advanced only after both budgets permit another
    item.  A formula failure is retained as a failed item so earlier results
    remain useful for a caller's in-memory checkpoint; this function does not
    claim to provide a disk checkpoint.
    """

    max_items = _validate_max_items(max_items)
    max_seconds = _validate_max_seconds(max_seconds)
    items: list[BatchItem] = []
    started = time.monotonic()
    iterator = iter(records)
    index = 0

    while True:
        # Check limits before touching the iterator.  In particular, reaching
        # max_items must not consume one look-ahead record.
        if max_items is not None and index >= max_items:
            return FormulaBatchReport(tuple(items), True, f"max_items={max_items}")
        if time.monotonic() - started >= max_seconds:
            return FormulaBatchReport(tuple(items), True, f"max_seconds={max_seconds}")

        try:
            record = next(iterator)
        except StopIteration:
            return FormulaBatchReport(tuple(items), False, None)
        except Exception as exc:
            error = f"input iterator failed at identity index {index}: {exc}"
            items.append(BatchItem(index, "failed", None, error))
            return FormulaBatchReport(tuple(items), True, f"failed_at_index={index}")

        # A slow iterator can cross the deadline while yielding an item.  Do
        # not start formula generation after that boundary; retain the exact
        # consumed position as a failure witness.
        if time.monotonic() - started >= max_seconds:
            error = f"timeout before starting identity index {index}"
            items.append(BatchItem(index, "failed", None, error))
            return FormulaBatchReport(tuple(items), True, f"max_seconds={max_seconds}")

        try:
            artifact = generate_formula(record, **kwargs)
        except Exception as exc:
            error = f"formula generation failed at identity index {index}: {exc}"
            items.append(BatchItem(index, "failed", None, error))
            return FormulaBatchReport(tuple(items), True, f"failed_at_index={index}")

        benchmark_error = _benchmark_failure(artifact)
        if benchmark_error is not None:
            items.append(BatchItem(index, "failed", artifact, benchmark_error))
            return FormulaBatchReport(tuple(items), True, f"benchmark_failed_at_index={index}")

        status = artifact.statuses.get("direct", "missing")
        items.append(BatchItem(index, status, artifact))
        index += 1

        # A single formula call cannot be interrupted.  Its returned artifact
        # is recorded, then the next boundary check prevents another call.
        if time.monotonic() - started >= max_seconds:
            return FormulaBatchReport(tuple(items), True, f"max_seconds={max_seconds}")


__all__ = ["BatchItem", "FormulaBatchReport", "run_formula_batch"]
