"""Link Yu (2022) Tables 5--26 pullbacks to the frozen four-body run04 catalog.

This is an evidence linker.  It never recalculates formulas and it does not
turn a numeric comparison into a pass/fail verdict.  Existing source statuses
and per-row fields are retained; each original-mirror pullback is augmented
with one arithmetic comparison for each of the 143 catalog classes.

Example:
    python examples/link_yu2022_all_tables_catalog.py \
        --report path/to/yu2022_all_tables.json \
        --catalog path/to/run04 \
        --output path/to/yu2022_all_tables_catalog_link.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_CLASS_COUNTS_BY_LOOP = (2, 6, 29, 106)
EXPECTED_TABLES = tuple(range(5, 27))
EXPECTED_MASSES = (1.0, 1.5, 0.8, 1.2)
EXPECTED_TOTAL_MASS = 6.0
EXPECTED_POINTS = tuple(range(8))
EXPECTED_STEPS = (2.0e-5, 1.0e-5)
REFERENCE_VARIABLES = (
    "s12", "s34", "theta0", "phi0", "theta12", "phi12", "theta34", "phi34",
)
PRIMARY_VARIANT = "row_variables"
REFERENCE_MIRROR = 1
EXPECTED_COMPARISONS_PER_VARIANT = (
    len(EXPECTED_TABLES) * sum(EXPECTED_CLASS_COUNTS_BY_LOOP) * len(EXPECTED_POINTS) * 2
)


def _expected_class_ids() -> tuple[str, ...]:
    return tuple(
        f"L{loop_count}-C{index:03d}"
        for loop_count, count in enumerate(EXPECTED_CLASS_COUNTS_BY_LOOP)
        for index in range(1, count + 1)
    )


EXPECTED_CLASS_IDS = _expected_class_ids()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _coordinate_key(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, Mapping):
        return None
    coordinates = tuple(_number(value.get(name)) for name in REFERENCE_VARIABLES)
    if any(item is None for item in coordinates):
        return None
    return tuple(float(item) for item in coordinates)


def _masses(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    parsed = tuple(_number(item) for item in value)
    if len(parsed) != 4 or any(item is None for item in parsed):
        return None
    return tuple(float(item) for item in parsed)


def _loop_for_class_id(class_id: str) -> int | None:
    for loop_count, count in enumerate(EXPECTED_CLASS_COUNTS_BY_LOOP):
        if any(class_id == f"L{loop_count}-C{index:03d}" for index in range(1, count + 1)):
            return loop_count
    return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_catalog_class(
    root: Path,
    index_entry: Mapping[str, Any],
    class_id: str,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_loop = _loop_for_class_id(class_id)
    entry_loop = _integer(index_entry.get("loop_count"))
    if expected_loop is None or entry_loop != expected_loop:
        errors.append("index_loop_count_mismatch")
    if index_entry.get("class_id") != class_id:
        errors.append("index_class_id_mismatch")

    evidence_name = index_entry.get("evidence")
    if not isinstance(evidence_name, str) or not evidence_name:
        return {
            "class_id": class_id,
            "status": "catalog_evidence_unavailable",
            "benchmark_status": index_entry.get("benchmark_status"),
            "identity_errors": errors + ["index_evidence_path_missing"],
            "metadata_errors": [],
            "normalization": None,
            "masses": None,
            "total_mass": None,
            "points": {},
        }

    evidence_path = root / evidence_name
    try:
        evidence = _read_json(evidence_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "class_id": class_id,
            "status": "catalog_evidence_unavailable",
            "benchmark_status": index_entry.get("benchmark_status"),
            "identity_errors": errors + [f"catalog_evidence_read_error:{exc}"],
            "metadata_errors": [],
            "normalization": None,
            "masses": None,
            "total_mass": None,
            "points": {},
        }
    if not isinstance(evidence, Mapping):
        return {
            "class_id": class_id,
            "status": "catalog_evidence_unavailable",
            "benchmark_status": index_entry.get("benchmark_status"),
            "identity_errors": errors + ["catalog_evidence_not_object"],
            "metadata_errors": [],
            "normalization": None,
            "masses": None,
            "total_mass": None,
            "points": {},
        }

    for identity_field in (
        "class_id", "loop_count", "topology_id", "representative_fixed_digest",
    ):
        index_value = index_entry.get(identity_field)
        evidence_value = evidence.get(identity_field)
        if index_value is None or evidence_value is None or index_value != evidence_value:
            errors.append(f"index_evidence_{identity_field}_mismatch")
    topology_id = evidence.get("topology_id")
    if expected_loop is not None and (
        not isinstance(topology_id, str) or not topology_id.startswith(f"L{expected_loop}-T")
    ):
        errors.append("topology_loop_count_mismatch")

    benchmark = evidence.get("benchmark")
    if not isinstance(benchmark, Mapping):
        benchmark = {}
    benchmark_status = benchmark.get("status")
    if index_entry.get("benchmark_status") != benchmark_status:
        errors.append("index_evidence_benchmark_status_mismatch")

    benchmark_evidence = benchmark.get("evidence")
    if not isinstance(benchmark_evidence, Mapping):
        benchmark_evidence = {}
    parsed_masses = _masses(benchmark_evidence.get("masses"))
    total_mass = _number(benchmark_evidence.get("total_mass"))
    normalization = evidence.get("normalization")
    metadata_errors: list[str] = []
    if parsed_masses != EXPECTED_MASSES:
        metadata_errors.append("catalog_masses_mismatch")
    if total_mass != EXPECTED_TOTAL_MASS:
        metadata_errors.append("catalog_total_mass_mismatch")
    if not isinstance(normalization, str) or not normalization:
        metadata_errors.append("catalog_normalization_missing")
    if benchmark_status != "verified_local":
        metadata_errors.append("catalog_benchmark_not_verified_local")

    points: dict[int, dict[str, Any]] = {}
    point_rows = benchmark_evidence.get("point_evidence")
    if not isinstance(point_rows, list):
        point_rows = []
        metadata_errors.append("catalog_point_evidence_missing")
    if len(point_rows) != len(EXPECTED_POINTS):
        metadata_errors.append("catalog_point_count_mismatch")

    for point_row in point_rows:
        if not isinstance(point_row, Mapping):
            metadata_errors.append("catalog_point_entry_not_object")
            continue
        point_index = _integer(point_row.get("point"))
        if point_index not in EXPECTED_POINTS:
            metadata_errors.append("catalog_point_index_invalid")
            continue
        if point_index in points:
            points[point_index] = {
                "coordinates": None,
                "lhs_by_step": {},
                "ambiguous_steps": set(),
                "errors": ["catalog_point_index_duplicated"],
            }
            continue

        point_errors: list[str] = []
        coordinates = _coordinate_key(point_row.get("reference_coordinates"))
        if coordinates is None:
            point_errors.append("catalog_reference_coordinates_missing_or_invalid")

        lhs_by_step: dict[float, float] = {}
        ambiguous_steps: set[float] = set()
        rows = point_row.get("sine_weighted_density_comparison")
        if not isinstance(rows, list):
            rows = []
            point_errors.append("catalog_sine_weighted_comparison_missing")
        for pullback in rows:
            if not isinstance(pullback, Mapping):
                point_errors.append("catalog_pullback_entry_not_object")
                continue
            step = _number(pullback.get("step"))
            lhs = _number(pullback.get("lhs"))
            if step is None or step <= 0.0:
                point_errors.append("catalog_pullback_step_invalid")
                continue
            if lhs is None or lhs <= 0.0:
                point_errors.append("catalog_pullback_lhs_invalid")
                continue
            if step in lhs_by_step:
                ambiguous_steps.add(step)
                lhs_by_step.pop(step, None)
            elif step not in ambiguous_steps:
                lhs_by_step[step] = lhs
        if len(lhs_by_step) + len(ambiguous_steps) != 2:
            point_errors.append("catalog_pullback_step_count_mismatch")
        points[point_index] = {
            "coordinates": coordinates,
            "lhs_by_step": lhs_by_step,
            "ambiguous_steps": ambiguous_steps,
            "errors": point_errors,
        }

    if set(points) != set(EXPECTED_POINTS):
        metadata_errors.append("catalog_point_indices_do_not_cover_0_through_7")
    for point_index, point in points.items():
        if point.get("coordinates") is None:
            metadata_errors.append(f"catalog_point_{point_index}_coordinates_invalid")
        if point.get("errors"):
            metadata_errors.extend(
                f"catalog_point_{point_index}:{error}" for error in point["errors"]
            )
        available_steps = set(point.get("lhs_by_step", {})) | set(point.get("ambiguous_steps", set()))
        if available_steps != set(EXPECTED_STEPS):
            metadata_errors.append(f"catalog_point_{point_index}_steps_do_not_match_run04")

    return {
        "class_id": class_id,
        "status": "verified_local" if not errors and not metadata_errors else "failed_or_incomplete",
        "benchmark_status": benchmark_status,
        "identity_errors": errors,
        "metadata_errors": metadata_errors,
        "normalization": normalization if isinstance(normalization, str) else None,
        "masses": parsed_masses,
        "total_mass": total_mass,
        "points": points,
    }


def _load_catalog(catalog_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = catalog_path if catalog_path.is_dir() else catalog_path.parent
    index_path = catalog_path / "index.json" if catalog_path.is_dir() else catalog_path
    diagnostics: list[str] = []
    try:
        index = _read_json(index_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read catalog index {index_path}: {exc}") from exc
    if not isinstance(index, Mapping):
        raise ValueError("catalog index must contain a JSON object")
    entries = index.get("classes")
    if not isinstance(entries, list):
        raise ValueError("catalog index classes field must be an array")

    by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unexpected_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            diagnostics.append("catalog index contains a non-object class entry")
            continue
        class_id = entry.get("class_id")
        if not isinstance(class_id, str):
            diagnostics.append("catalog index class entry has no string class_id")
            continue
        by_id[class_id].append(entry)
        if class_id not in EXPECTED_CLASS_IDS:
            unexpected_ids.append(class_id)

    classes: dict[str, dict[str, Any]] = {}
    for class_id in EXPECTED_CLASS_IDS:
        matches = by_id.get(class_id, [])
        if len(matches) == 0:
            classes[class_id] = {
                "class_id": class_id,
                "status": "catalog_index_entry_missing",
                "benchmark_status": None,
                "identity_errors": ["catalog_index_entry_missing"],
                "metadata_errors": [],
                "normalization": None,
                "masses": None,
                "total_mass": None,
                "points": {},
            }
        elif len(matches) > 1:
            classes[class_id] = {
                "class_id": class_id,
                "status": "catalog_index_entry_duplicated",
                "benchmark_status": None,
                "identity_errors": ["catalog_index_entry_duplicated"],
                "metadata_errors": [],
                "normalization": None,
                "masses": None,
                "total_mass": None,
                "points": {},
            }
        else:
            classes[class_id] = _load_catalog_class(root, matches[0], class_id)

    index_ids = [entry.get("class_id") for entry in entries if isinstance(entry, Mapping)]
    if len(index_ids) != len(EXPECTED_CLASS_IDS) or set(index_ids) != set(EXPECTED_CLASS_IDS):
        diagnostics.append("catalog class IDs do not match the exact 143-class set")
    if index.get("benchmark_status") != "verified_local":
        diagnostics.append("catalog index benchmark_status is not verified_local")
    if index.get("benchmark_requested") is not True:
        diagnostics.append("catalog index does not record benchmark_requested=true")
    failures = index.get("failures", [])
    if failures:
        diagnostics.append(f"catalog index records {len(failures)} build failure(s)")

    normalizations = sorted({
        info["normalization"]
        for info in classes.values()
        if isinstance(info.get("normalization"), str)
    })
    if len(normalizations) != 1:
        diagnostics.append("catalog class normalization is missing or inconsistent")
    incomplete_classes = [
        class_id for class_id, info in classes.items() if info.get("status") != "verified_local"
    ]
    if incomplete_classes:
        diagnostics.append(
            f"{len(incomplete_classes)} catalog class(es) failed identity, metadata, or benchmark evidence checks"
        )
    common_normalization = normalizations[0] if len(normalizations) == 1 else None

    metadata = {
        "index_path": str(index_path),
        "catalog_root": str(root),
        "schema_version": index.get("schema_version"),
        "index_benchmark_status": index.get("benchmark_status"),
        "class_count": len(entries),
        "expected_class_count": len(EXPECTED_CLASS_IDS),
        "unexpected_class_ids": sorted(set(unexpected_ids)),
        "normalization": common_normalization,
        "diagnostics": diagnostics,
    }
    return classes, metadata


def _row_result(
    row: Mapping[str, Any],
    *,
    classes: Mapping[str, Mapping[str, Any]],
    case_is_target_mass_group: bool,
    case_normalization_matches: bool,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(row))
    mirror = _integer(row.get("mirror"))
    if mirror != REFERENCE_MIRROR:
        result["catalog_crosscheck"] = {
            "status": "excluded_non_reference_mirror",
            "catalog_comparisons": [],
            "note": "Only mirror=1 (the original reference event) is linked to run04.",
        }
        return result

    point_index = _integer(row.get("point_index"))
    step = _number(row.get("step"))
    report_coordinates = _coordinate_key(row.get("reference_coordinates"))
    external = _number(row.get("external_pullback_density"))
    linked: list[dict[str, Any]] = []

    for class_id in EXPECTED_CLASS_IDS:
        info = classes[class_id]
        comparison: dict[str, Any] = {
            "class_id": class_id,
            "catalog_class_status": info.get("status"),
            "catalog_benchmark_status": info.get("benchmark_status"),
            "status": "not_compared",
            "decision": "no_tolerance_verdict",
        }
        if not case_is_target_mass_group:
            comparison["status"] = "case_mass_metadata_mismatch"
            linked.append(comparison)
            continue
        if not case_normalization_matches:
            comparison["status"] = "case_normalization_mismatch"
            linked.append(comparison)
            continue
        if info.get("identity_errors"):
            comparison["status"] = "catalog_identity_invalid"
            comparison["catalog_errors"] = list(info["identity_errors"])
            linked.append(comparison)
            continue
        if info.get("masses") != EXPECTED_MASSES or info.get("total_mass") != EXPECTED_TOTAL_MASS:
            comparison["status"] = "catalog_mass_metadata_mismatch"
            comparison["catalog_errors"] = list(info.get("metadata_errors", ()))
            linked.append(comparison)
            continue
        if info.get("normalization") is None:
            comparison["status"] = "catalog_normalization_missing"
            linked.append(comparison)
            continue
        if info.get("normalization") != row.get("_case_normalization"):
            comparison["status"] = "catalog_normalization_mismatch"
            linked.append(comparison)
            continue
        if point_index not in EXPECTED_POINTS:
            comparison["status"] = "report_point_index_invalid"
            linked.append(comparison)
            continue
        point = info.get("points", {}).get(point_index)
        if not isinstance(point, Mapping):
            comparison["status"] = "catalog_point_missing"
            linked.append(comparison)
            continue
        catalog_coordinates = point.get("coordinates")
        comparison["point_index"] = point_index
        comparison["reference_coordinates_match"] = (
            report_coordinates is not None and report_coordinates == catalog_coordinates
        )
        if not comparison["reference_coordinates_match"]:
            comparison["status"] = "reference_coordinates_mismatch"
            linked.append(comparison)
            continue
        if step is None or step <= 0.0:
            comparison["status"] = "report_step_invalid"
            linked.append(comparison)
            continue
        if step in point.get("ambiguous_steps", set()):
            comparison["status"] = "catalog_step_ambiguous"
            linked.append(comparison)
            continue
        lhs = point.get("lhs_by_step", {}).get(step)
        if lhs is None:
            comparison["status"] = "catalog_step_missing"
            linked.append(comparison)
            continue
        if external is None or external <= 0.0:
            comparison["status"] = "external_pullback_missing_or_invalid"
            comparison["catalog_lhs"] = lhs
            linked.append(comparison)
            continue

        difference = external - float(lhs)
        relative_residual = abs(difference) / max(abs(external), abs(float(lhs)), 1.0e-300)
        comparison.update({
            "catalog_lhs": float(lhs),
            "external_pullback_density": external,
            "difference_external_minus_catalog_lhs": difference,
            "absolute_error_external_vs_catalog_lhs": abs(difference),
            "relative_residual_external_vs_catalog_lhs": relative_residual,
            "ratio_external_over_catalog_lhs": external / float(lhs),
        })
        if info.get("benchmark_status") != "verified_local" or info.get("metadata_errors"):
            comparison["status"] = "compared_catalog_benchmark_incomplete"
            comparison["catalog_errors"] = list(info.get("metadata_errors", ()))
        else:
            comparison["status"] = "compared"
        linked.append(comparison)

    statuses = Counter(str(item.get("status")) for item in linked)
    result["catalog_crosscheck"] = {
        "status": "linked_to_143_classes" if statuses.get("compared", 0) + statuses.get(
            "compared_catalog_benchmark_incomplete", 0
        ) == len(EXPECTED_CLASS_IDS) else "partial_or_ineligible",
        "catalog_comparison_count": len(linked),
        "numeric_comparison_count": sum(
            item.get("relative_residual_external_vs_catalog_lhs") is not None for item in linked
        ),
        "class_status_counts": dict(sorted(statuses.items())),
        "catalog_comparisons": linked,
        "note": "Arithmetic comparison only; no tolerance verdict or pass status is inferred.",
    }
    return result


def _is_target_mass_group(case: Mapping[str, Any]) -> bool:
    return (
        _masses(case.get("masses")) == EXPECTED_MASSES
        and _number(case.get("total_mass")) == EXPECTED_TOTAL_MASS
    )


def _case_steps_match(case: Mapping[str, Any]) -> bool:
    values = case.get("finite_difference_steps")
    if not isinstance(values, (list, tuple)):
        return False
    parsed = tuple(_number(item) for item in values)
    return (
        len(parsed) == len(EXPECTED_STEPS)
        and all(item is not None for item in parsed)
        and set(float(item) for item in parsed) == set(EXPECTED_STEPS)
    )


def _table_key(table: Mapping[str, Any]) -> int | None:
    value = _integer(table.get("table"))
    return value if value in EXPECTED_TABLES else None


def _enrich_table(
    table: Mapping[str, Any],
    *,
    classes: Mapping[str, Mapping[str, Any]],
    case_is_target_mass_group: bool,
    case_normalization_matches: bool,
    case_normalization: Any,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(table))
    variant = table.get("variant")
    raw_rows = table.get("comparisons")
    if not isinstance(raw_rows, list):
        raw_rows = []
    enriched_rows: list[Any] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            enriched_rows.append({
                "source_row": copy.deepcopy(raw),
                "catalog_crosscheck": {
                    "status": "comparison_row_not_object",
                    "catalog_comparisons": [],
                },
            })
            continue
        row_with_context = dict(raw)
        row_with_context["_case_normalization"] = case_normalization
        enriched_rows.append(_row_result(
            row_with_context,
            classes=classes,
            case_is_target_mass_group=case_is_target_mass_group,
            case_normalization_matches=case_normalization_matches,
        ))
        enriched_rows[-1].pop("_case_normalization", None)

    result["comparisons"] = enriched_rows
    comparison_statuses = Counter()
    numeric_comparisons = 0
    excluded_mirrors = 0
    original_rows = 0
    for row in enriched_rows:
        if not isinstance(row, Mapping):
            continue
        original_rows += 1
        crosscheck = row.get("catalog_crosscheck", {})
        if not isinstance(crosscheck, Mapping):
            continue
        comparison_statuses.update(crosscheck.get("class_status_counts", {}))
        numeric_comparisons += int(crosscheck.get("numeric_comparison_count", 0) or 0)
        if crosscheck.get("status") == "excluded_non_reference_mirror":
            excluded_mirrors += 1

    variant_text = variant if isinstance(variant, str) else "(missing variant)"
    result["catalog_crosscheck"] = {
        "status": (
            "complete_numeric_links"
            if numeric_comparisons == len(EXPECTED_CLASS_IDS) * len(EXPECTED_POINTS) * 2
            else "partial_or_ineligible"
        ),
        "variant": variant_text,
        "primary_variant": variant_text == PRIMARY_VARIANT,
        "source_table_status": table.get("status"),
        "table_id_valid": _table_key(table) is not None,
        "table": table.get("table"),
        "input_comparison_row_count": original_rows,
        "non_reference_mirror_rows_excluded": excluded_mirrors,
        "numeric_class_comparison_count": numeric_comparisons,
        "class_comparison_status_counts": dict(sorted(comparison_statuses.items())),
        "expected_class_comparisons_for_complete_table": (
            len(EXPECTED_CLASS_IDS) * len(EXPECTED_POINTS) * 2
        ),
        "note": "Coverage status describes numeric links only; the input table status is retained verbatim and not upgraded.",
    }
    return result


def link_report(report: Mapping[str, Any], classes: Mapping[str, Mapping[str, Any]], catalog_meta: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of the input report augmented with class-level comparisons."""

    output = copy.deepcopy(dict(report))
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("numeric report must contain a cases array")

    catalog_normalization = catalog_meta.get("normalization")
    variant_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "table_entries": 0,
        "valid_table_ids": set(),
        "input_rows": 0,
        "non_reference_mirror_rows_excluded": 0,
        "numeric_class_comparisons": 0,
        "class_comparison_status_counts": Counter(),
    })
    target_case_count = 0
    exact_metadata_case_count = 0
    target_case_step_metadata_mismatch_count = 0
    enriched_cases: list[Any] = []

    for case in cases:
        if not isinstance(case, Mapping):
            enriched_cases.append({
                "source_case": copy.deepcopy(case),
                "catalog_crosscheck": {"status": "case_entry_not_object"},
            })
            continue
        case_copy = copy.deepcopy(dict(case))
        target_mass_group = _is_target_mass_group(case)
        case_steps_match = _case_steps_match(case)
        normalization = case.get("normalization")
        normalization_matches = (
            isinstance(catalog_normalization, str)
            and normalization == catalog_normalization
        )
        if target_mass_group:
            target_case_count += 1
            if normalization_matches:
                exact_metadata_case_count += 1
            if not case_steps_match:
                target_case_step_metadata_mismatch_count += 1

        raw_tables = case.get("tables")
        if not isinstance(raw_tables, list):
            raw_tables = []
        enriched_tables = [
            _enrich_table(
                table,
                classes=classes,
                case_is_target_mass_group=target_mass_group,
                case_normalization_matches=normalization_matches,
                case_normalization=normalization,
            )
            if isinstance(table, Mapping)
            else {
                "source_table": copy.deepcopy(table),
                "catalog_crosscheck": {"status": "table_entry_not_object"},
            }
            for table in raw_tables
        ]
        case_copy["tables"] = enriched_tables
        if not target_mass_group:
            case_status = "not_target_mass_group"
        elif not normalization_matches:
            case_status = "catalog_normalization_mismatch"
        else:
            case_status = "target_mass_group_and_normalization_match"
        case_copy["catalog_crosscheck"] = {
            "status": case_status,
            "target_masses_match": target_mass_group,
            "target_total_mass": EXPECTED_TOTAL_MASS,
            "case_normalization": normalization,
            "catalog_normalization": catalog_normalization,
            "normalization_matches": normalization_matches,
            "finite_difference_steps_match_run04": case_steps_match,
            "source_table_statuses_retained": True,
        }
        enriched_cases.append(case_copy)

        for table in enriched_tables:
            if not isinstance(table, Mapping):
                continue
            crosscheck = table.get("catalog_crosscheck", {})
            if not isinstance(crosscheck, Mapping):
                continue
            variant_value = crosscheck.get("variant")
            variant_name = variant_value if isinstance(variant_value, str) else "(missing variant)"
            stats = variant_stats[variant_name]
            stats["table_entries"] += 1
            table_id = _table_key(table)
            if table_id is not None and target_mass_group and normalization_matches:
                stats["valid_table_ids"].add(table_id)
            stats["input_rows"] += int(crosscheck.get("input_comparison_row_count", 0) or 0)
            stats["non_reference_mirror_rows_excluded"] += int(
                crosscheck.get("non_reference_mirror_rows_excluded", 0) or 0
            )
            stats["numeric_class_comparisons"] += int(
                crosscheck.get("numeric_class_comparison_count", 0) or 0
            )
            stats["class_comparison_status_counts"].update(
                crosscheck.get("class_comparison_status_counts", {})
            )

    output["cases"] = enriched_cases
    finalized_variant_stats: dict[str, Any] = {}
    for variant, stats in sorted(variant_stats.items()):
        is_primary = variant == PRIMARY_VARIANT
        finalized_variant_stats[variant] = {
            "is_primary_specification": is_primary,
            "expected_table_ids": list(EXPECTED_TABLES),
            "observed_valid_table_ids_in_target_case": sorted(stats["valid_table_ids"]),
            "observed_valid_table_count_in_target_case": len(stats["valid_table_ids"]),
            "table_entries_across_report": stats["table_entries"],
            "input_comparison_rows_across_report": stats["input_rows"],
            "non_reference_mirror_rows_excluded": stats["non_reference_mirror_rows_excluded"],
            "numeric_class_comparisons_across_report": stats["numeric_class_comparisons"],
            "expected_comparisons_for_one_complete_variant": EXPECTED_COMPARISONS_PER_VARIANT,
            "class_comparison_status_counts": dict(sorted(stats["class_comparison_status_counts"].items())),
        }

    primary = finalized_variant_stats.get(PRIMARY_VARIANT, {})
    actual_primary = primary.get("numeric_class_comparisons_across_report", 0)
    primary_table_ids = primary.get("observed_valid_table_ids_in_target_case", [])
    diagnostics = list(catalog_meta.get("diagnostics", ()))
    if target_case_count != 1:
        diagnostics.append(f"expected exactly one target mass group, found {target_case_count}")
    if exact_metadata_case_count != 1:
        diagnostics.append(
            f"expected exactly one target case with catalog normalization, found {exact_metadata_case_count}"
        )
    if target_case_step_metadata_mismatch_count:
        diagnostics.append(
            f"{target_case_step_metadata_mismatch_count} target case(s) have finite_difference_steps metadata that does not match run04"
        )
    if primary_table_ids != list(EXPECTED_TABLES):
        diagnostics.append("row_variables does not contain exactly Tables 5--26 in the target case")
    if actual_primary != EXPECTED_COMPARISONS_PER_VARIANT:
        diagnostics.append(
            "row_variables numeric comparison count does not equal the 22*143*8*2 main specification"
        )

    output["catalog_crosscheck"] = {
        "schema_version": "yu2022-all-tables-run04-link-v1",
        "catalog": dict(catalog_meta),
        "target_masses": list(EXPECTED_MASSES),
        "target_total_mass": EXPECTED_TOTAL_MASS,
        "target_normalization": catalog_normalization,
        "reference_mirror": REFERENCE_MIRROR,
        "primary_variant": PRIMARY_VARIANT,
        "expected_classes": len(EXPECTED_CLASS_IDS),
        "expected_points": len(EXPECTED_POINTS),
        "expected_steps_per_point": 2,
        "expected_primary_comparisons": EXPECTED_COMPARISONS_PER_VARIANT,
        "primary_numeric_comparisons": actual_primary,
        "target_case_count": target_case_count,
        "target_case_with_matching_normalization_count": exact_metadata_case_count,
        "target_case_step_metadata_mismatch_count": target_case_step_metadata_mismatch_count,
        "variants": finalized_variant_stats,
        "status": "linked" if not diagnostics else "partial_or_inconsistent",
        "diagnostics": diagnostics,
        "verdict_policy": "Arithmetic evidence only; no tolerance verdict or pass status is inferred.",
    }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Yu2022 all-tables numeric report JSON")
    parser.add_argument("--catalog", required=True, type=Path, help="run04 directory or its index.json")
    parser.add_argument("--output", required=True, type=Path, help="output JSON path")
    args = parser.parse_args(argv)

    report_path = args.report
    output_path = args.output
    try:
        if report_path.resolve() == output_path.resolve():
            raise ValueError("--output must not overwrite --report")
        report = _read_json(report_path)
        if not isinstance(report, Mapping):
            raise ValueError("numeric report must contain a JSON object")
        classes, catalog_meta = _load_catalog(args.catalog)
        linked = link_report(report, classes, catalog_meta)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(linked, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = linked["catalog_crosscheck"]
    print(json.dumps({
        "output": str(output_path),
        "status": summary["status"],
        "expected_primary_comparisons": summary["expected_primary_comparisons"],
        "primary_numeric_comparisons": summary["primary_numeric_comparisons"],
        "diagnostics": summary["diagnostics"],
    }, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "linked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
