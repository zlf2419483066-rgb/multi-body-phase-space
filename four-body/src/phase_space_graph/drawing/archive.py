"""Read-only adapter for the saved independent-label archive.

The adapter reconstructs records from stored certificates.  It never invokes
the enumerator and refuses rows whose identity digest or physical record is
inconsistent with the archive payload.
"""
from __future__ import annotations
from dataclasses import dataclass
import gzip, hashlib, json
from pathlib import Path
from typing import Any, Iterable

from ..domain import Labeling, Topology
from ..invariants import analyze_invariants
from ..labeling import MomentumCertificate
from ..physical_labels import PhysicalLabelRecord, review_physical_record
from .layout import layout_graph
from .geometry import build_scene
from .records import prepare_drawing
from .render import render_drawing
from .validation import validate_drawing

class ArchiveError(ValueError):
    pass

def _freeze(value: Any) -> Any:
    if isinstance(value, list): return tuple(_freeze(item) for item in value)
    if isinstance(value, dict): return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    return value

def _digest(key: Any) -> str:
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

@dataclass(frozen=True, slots=True)
class ArchivedIdentity:
    program_id: str
    topology_key: tuple[Any, ...]
    key: tuple[Any, ...]
    digest: str
    canonical_record: PhysicalLabelRecord
    source_path: str

def _record_from_dict(data: dict[str, Any]) -> PhysicalLabelRecord:
    try:
        topology = Topology.from_dict(data["topology"])
        labeling = Labeling.from_dict(data["complete_labeling"])
        momentum_data = data["momentum"]
        momentum = MomentumCertificate(
            tuple((row["edge_index"], row["occurrence"], tuple(row["coefficients"]))
                  for row in momentum_data["double_momenta"]),
            tuple((row["vertex"], tuple(row["residual"]))
                  for row in momentum_data["vertex_residuals"]),
            bool(momentum_data["verified"]),
        )
        independence = analyze_invariants(momentum.variables, topology.n)
        groups = []
        for group in data.get("single_groups", ()):
            occurrences = []
            for item in group.get("occurrences", ()):
                if item.get("kind") == "external":
                    occurrences.append(("external", item["vertex"], item["slot"]))
                elif item.get("kind") == "internal":
                    occurrences.append(("internal", item["edge_index"], item["occurrence"]))
                else:
                    raise ArchiveError("unknown occurrence kind in archived record")
            groups.append((group["particle_id"], tuple(occurrences)))
        cancellations = tuple((row["particle_id"], row["count"])
                              for row in data.get("measure_cancellations", ()))
        record = PhysicalLabelRecord(topology, labeling, momentum, independence,
                                     tuple(groups), cancellations)
        checked = review_physical_record(record)
        if checked.stage not in {"accepted", "dependent"}:
            raise ArchiveError(f"archived physical record review failed: {checked.stage}")
        return record
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ArchiveError(f"malformed archived physical record: {exc}") from exc

def load_identity_archive(path: str | Path, *, max_rows: int | None = None) -> tuple[ArchivedIdentity, ...]:
    """Load saved identities without running G05/G06/G08 enumeration."""
    source = Path(path)
    try:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"cannot read identity archive: {source}") from exc
    if not isinstance(payload, list):
        raise ArchiveError("identity archive must be a JSON list")
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    rows = payload if max_rows is None else payload[:max_rows]
    result=[]
    for row in rows:
        if not isinstance(row, dict): raise ArchiveError("identity archive row must be an object")
        key = _freeze(row.get("identity_key"))
        topology_key = _freeze(row.get("topology_key"))
        digest = row.get("digest")
        if not isinstance(key, tuple) or not isinstance(topology_key, tuple) or not isinstance(digest, str):
            raise ArchiveError("identity archive row lacks identity keys or digest")
        if _digest(key) != digest:
            raise ArchiveError(f"identity digest mismatch for {row.get('program_id')!r}")
        record = _record_from_dict(row.get("canonical_record", {}))
        result.append(ArchivedIdentity(str(row.get("program_id", "")), topology_key, key,
                                       digest, record, str(source)))
    return tuple(result)

def select_representatives(entries: Iterable[ArchivedIdentity], *, max_items: int | None = None) -> tuple[ArchivedIdentity, ...]:
    """Choose one deterministic identity per topology from already loaded rows."""
    groups: dict[tuple[Any, ...], list[ArchivedIdentity]] = {}
    for entry in entries: groups.setdefault(entry.topology_key, []).append(entry)
    representatives = tuple(sorted((min(group, key=lambda item: json.dumps(item.key, ensure_ascii=False, separators=(",", ":")))
                                    for group in groups.values()),
                                  key=lambda item: (item.program_id, item.digest)))
    if max_items is not None:
        if max_items < 0: raise ValueError("max_items must be non-negative")
        return representatives[:max_items]
    return representatives

def export_representatives(archive_path: str | Path, output_dir: str | Path, *, max_items: int = 1,
                           dot_path: str | None = None, fmt: str = "svg") -> dict[str, Any]:
    """Export a bounded representative set and a manifest; never overwrites."""
    if max_items < 0: raise ValueError("max_items must be non-negative")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    entries = select_representatives(load_identity_archive(archive_path), max_items=max_items)
    manifest = {"archive": str(archive_path), "selected": [], "pending": [], "failed": []}
    for entry in entries:
        item = {"program_id": entry.program_id, "digest": entry.digest, "topology_key": entry.topology_key}
        try:
            drawing_input = prepare_drawing(identity=entry)
            layout = layout_graph(drawing_input, dot_path=dot_path)
            scene = build_scene(drawing_input, layout)
            validation = validate_drawing(drawing_input, scene)
            if not validation.valid:
                raise ArchiveError(f"drawing validation failed: {validation.issues}")
            name = entry.program_id or entry.digest[:12]
            image = root / f"{name}.{fmt}"
            image.write_bytes(render_drawing(scene, fmt=fmt))
            item.update({"status": "completed", "image": image.name, "issues": validation.issues})
            manifest["selected"].append(item)
        except Exception as exc:
            item.update({"status": "failed", "error": str(exc)})
            manifest["failed"].append(item)
    manifest["selected_count"] = len(manifest["selected"])
    manifest["failed_count"] = len(manifest["failed"])
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=list), encoding="utf-8")
    return manifest
