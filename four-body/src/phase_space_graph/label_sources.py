"""逐项比较人工资料记录与程序标记目录。

This module is an external acceptance layer.  It consumes ``SourceDiagram``
transcriptions and saved LM-06 catalogs, then reports what is known and what
is still unverified.  It never changes the generator, imports source counts
as constraints, or treats a partial catalog as a complete one.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .canonical import canonicalize_topology
from .domain import CountMode
from .label_catalog import LabelCatalogBundle, TopologyLabelCatalog
from .label_identity import MarkedIdentity, canonicalize_physical_labeling
from .physical_labels import NormalizationResult, normalize_physical_labeling
from .structure import analyze_topology
from .validation import SourceDiagram


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def _catalog_entries(catalogs: LabelCatalogBundle | Iterable[TopologyLabelCatalog]) -> tuple[TopologyLabelCatalog, ...]:
    if isinstance(catalogs, LabelCatalogBundle):
        return tuple(catalogs.catalogs)
    try:
        entries = tuple(catalogs)
    except TypeError as exc:
        raise TypeError("catalogs must be a LabelCatalogBundle or an iterable of topology catalogs") from exc
    if not all(isinstance(item, TopologyLabelCatalog) for item in entries):
        raise TypeError("catalogs must contain TopologyLabelCatalog objects")
    return entries


def _source_location(source: SourceDiagram) -> dict[str, object]:
    return {
        "source": source.source,
        "page": source.page,
        "figure": source.figure,
        "transcription_verified": source.transcription_verified,
    }


def _normalization_diagnostics(result: NormalizationResult) -> list[dict[str, object]]:
    rows = [issue.to_dict() for issue in result.issues]
    if result.legacy_evaluation is not None:
        for code, message in zip(result.legacy_evaluation.failure_codes, result.legacy_evaluation.failure_messages):
            if not any(item.get("code") == code for item in rows):
                rows.append({"code": code, "message": message})
    if result.structure_report is not None:
        rows.extend(
            {"code": check.code, "message": check.message, "details": _jsonable(check.details)}
            for check in result.structure_report.failures
            if not any(item.get("code") == check.code for item in rows)
        )
    return rows


def _mode(value: CountMode | str) -> CountMode:
    if isinstance(value, CountMode):
        return value
    try:
        return CountMode(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown count mode: {value!r}") from exc


def _catalog_identities(catalog: TopologyLabelCatalog) -> tuple[MarkedIdentity, ...]:
    return tuple(catalog.identities)


def match_source_record(
    source_diagram: SourceDiagram,
    catalogs: LabelCatalogBundle | Iterable[TopologyLabelCatalog],
    *,
    mode: CountMode | str,
) -> dict[str, object]:
    """Match one verified source transcription to saved physical identities.

    The returned mapping deliberately separates transcription, structure,
    topology and labeling statuses.  ``status='catalog_match'`` is reserved
    for an exact identity match in a complete catalog; an exact match inside
    a partial catalog remains explicitly ``inconclusive``.
    """

    if not isinstance(source_diagram, SourceDiagram):
        raise TypeError("source_diagram must be a SourceDiagram")
    selected_mode = _mode(mode)
    entries = _catalog_entries(catalogs)
    mode_entries = tuple(item for item in entries if item.mode is selected_mode)
    result: dict[str, object] = {
        **_source_location(source_diagram),
        "mode": selected_mode.value,
        "transcription_status": "verified" if source_diagram.transcription_verified else "unverified",
        "structure_status": "not_provided",
        "topology_status": "not_provided",
        "label_status": "not_provided",
        "status": "inconclusive",
        "catalog_match": False,
        "program_keys": [],
        "matched_keys": [],
        "source_only": [],
        "program_only": [],
        "undetermined": [],
        "differences": [],
        "diagnostics": [],
    }
    if not source_diagram.transcription_verified:
        result["undetermined"] = [{"reason": "transcription_unverified", "message": "资料转录尚未复核，不能授予 catalog_match。"}]
        return result
    if source_diagram.topology is None:
        result["structure_status"] = "not_transcribed"
        result["undetermined"] = [{"reason": "topology_not_provided", "message": "资料条目没有转录拓扑。"}]
        return result
    if not mode_entries:
        result["structure_status"] = "not_checked"
        result["topology_status"] = "mode_unavailable"
        result["undetermined"] = [{"reason": "mode_catalog_missing", "message": f"目录中没有 {selected_mode.value} 口径。"}]
        return result

    structure = analyze_topology(source_diagram.topology)
    result["structure_status"] = "passed" if structure.valid else "rejected"
    result["structure"] = structure.to_dict()
    diagnostics = [
        {"code": check.code, "message": check.message, "details": _jsonable(check.details)}
        for check in structure.failures
    ] if not structure.valid else []
    result["diagnostics"] = diagnostics
    if not structure.valid:
        result["topology_status"] = "structure_rejected"
        result["label_status"] = "not_checked"
        return result

    source_key = tuple(canonicalize_topology(source_diagram.topology).key)
    matching_catalogs = tuple(item for item in mode_entries if item.topology_key == source_key)
    result["program_keys"] = [_jsonable(item.topology_key) for item in matching_catalogs]
    result["topology_status"] = "topology_match" if matching_catalogs else "topology_not_found"
    if not matching_catalogs:
        scope = catalogs.topology_scope if isinstance(catalogs, LabelCatalogBundle) else {}
        if (isinstance(catalogs, LabelCatalogBundle) and catalogs.topology_scope_complete
                and scope.get("mode") == selected_mode.value
                and scope.get("n") == source_diagram.topology.n
                and scope.get("loop_count") == structure.metrics.loop_count):
            result["status"] = "catalog_mismatch"
            result["source_only"] = [_jsonable(source_key)]
        else:
            result["undetermined"] = [{"reason": "partial_program_catalog", "message": "程序目录未完整覆盖声明范围，不能把资料图判为程序独有。"}]
        return result

    catalog = matching_catalogs[0]
    program_identities = tuple({identity.key: identity for entry in matching_catalogs
                                for identity in entry.identities}.values())
    catalog_complete = (all(entry.enumeration_complete and not entry.issues
                            and entry.import_status in {"complete", "not_applicable"}
                            for entry in matching_catalogs)
                        and not (isinstance(catalogs, LabelCatalogBundle) and catalogs.issues))
    result["program_identity_keys"] = [_jsonable(identity.key) for identity in program_identities]
    if source_diagram.labeling is None:
        result["label_status"] = "not_provided"
        result["program_only"] = []
        result["undetermined"] = [{"reason": "source_label_not_provided", "message": "资料只提供拓扑，尚不能进行完整标记匹配。"}]
        return result

    normalization = normalize_physical_labeling(source_diagram.topology, source_diagram.labeling, selected_mode)
    result["source_label_evaluation"] = normalization.to_dict()
    result["diagnostics"] = _normalization_diagnostics(normalization)
    if normalization.record is None or normalization.stage not in {"accepted", "dependent"}:
        result["label_status"] = normalization.stage
        result["differences"] = [{"reason": "source_label_rejected", "stage": normalization.stage}]
        return result
    identity = canonicalize_physical_labeling(normalization.record, selected_mode)
    result["source_identity_key"] = _jsonable(identity.key)
    result["source_identity_digest"] = identity.digest
    if normalization.stage == "dependent":
        result["label_status"] = "dependent"
        result["differences"] = [{"reason": "dependent_invariants", "message": "资料标记通过守恒但双线不变量不独立。"}]
        return result
    matches = tuple(item for item in program_identities if item.key == identity.key)
    if matches:
        result["matched_keys"] = [_jsonable(identity.key)]
        result["label_status"] = "label_match"
        result["program_only"] = [_jsonable(item.key) for item in program_identities if item.key != identity.key]
        if catalog_complete:
            result["status"] = "catalog_match"
            result["catalog_match"] = True
        else:
            result["status"] = "inconclusive"
            result["undetermined"] = [{"reason": "partial_program_catalog", "message": "身份已找到，但程序目录尚未证明找全。"}]
    else:
        result["label_status"] = "label_not_found"
        result["source_only"] = [_jsonable(identity.key)]
        result["program_only"] = [_jsonable(item.key) for item in program_identities]
        result["status"] = "catalog_mismatch" if catalog_complete else "inconclusive"
        if not catalog_complete:
            result["undetermined"] = [{"reason": "partial_program_catalog", "message": "程序目录不完整，来源独有／程序独有只能作为当前差集。"}]
    return result


__all__ = ["match_source_record"]
