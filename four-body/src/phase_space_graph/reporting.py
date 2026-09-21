"""G08 reproducible audit reports for topology, labeling, and coverage runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

from .classification import (
    CombinationSearchReport,
    CoverageIndex,
    build_coverage_index,
    enumerate_invariant_combinations,
    link_labeling_report,
)
from .domain import CountMode, RunStatus, Topology
from .enumeration import EnumerationReport
from .enumeration import EnumerationLimits, enumerate_topologies
from .labeling import LabelingLimits, LabelingSearchReport, enumerate_labelings
from .validation import compare_reference_count, compare_combination_coverage


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class AuditIssue:
    """A report-level completeness or input issue."""

    code: str
    message: str
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("AuditIssue.code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("AuditIssue.message must be a non-empty string")
    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "details": _jsonable(self.details)}


@dataclass(frozen=True, slots=True)
class SourceReference:
    """A source-directory reference kept separate from program truth."""

    name: str
    scope: str
    expected: object
    verified: bool = False
    note: str = ""
    n: int | None = None
    loop_count: int | None = None
    metric: str = 'effective_topologies'

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SourceReference.name must be a non-empty string")
        if not isinstance(self.scope, str) or not self.scope:
            raise ValueError("SourceReference.scope must be a non-empty string")
        if not isinstance(self.verified, bool):
            raise TypeError("SourceReference.verified must be a boolean")
        for name in ('n', 'loop_count'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f'{name} must be a non-negative integer or None')

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scope": self.scope,
            "expected": _jsonable(self.expected),
            "verified": self.verified,
            "note": self.note,
            'n': self.n,
            'loop_count': self.loop_count,
            'metric': self.metric,
        }


@dataclass(frozen=True, slots=True)
class RuntimeMetadata:
    """Environment and repository information needed to rerun a report."""

    rule_version: str
    python_version: str
    platform: str
    package_version: str | None
    git_commit: str | None
    git_dirty: bool | None
    test_command: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_version": self.rule_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "test_command": self.test_command,
        }


def collect_runtime_metadata(
    repository_root: str | Path | None = None,
    *,
    rule_version: str = "g08-2026-09-10",
    test_command: str = "python -m unittest discover -s tests -v",
) -> RuntimeMetadata:
    """Collect rerun metadata without requiring Git or installed package metadata."""

    root = Path(repository_root) if repository_root is not None else Path.cwd()
    commit: str | None = None
    dirty: bool | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            commit = result.stdout.strip() or None
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            if status.returncode == 0:
                dirty = bool(status.stdout.strip())
    except (OSError, ValueError):
        pass
    try:
        package_version = importlib.metadata.version("phase-space-graphic")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    return RuntimeMetadata(
        rule_version=rule_version,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        package_version=package_version,
        git_commit=commit,
        git_dirty=dirty,
        test_command=test_command,
    )


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Complete G08 artifact with explicit status and layered counts."""

    status: RunStatus
    started_at: str | None
    finished_at: str | None
    topology_report: EnumerationReport
    labeling_reports: tuple[LabelingSearchReport, ...]
    combination_report: CombinationSearchReport | None
    coverage: CoverageIndex
    source_references: tuple[SourceReference, ...]
    source_status: str
    runtime: RuntimeMetadata
    topology_count: int
    effective_topology_count: int
    fixed_label_classes: int | None
    particle_relabeling_classes: int | None
    variable_combination_count: int
    dn_count: int
    failure_counts: tuple[tuple[str, int], ...]
    issues: tuple[AuditIssue, ...]
    source_comparisons: tuple[dict[str, object], ...] = ()
    combination_comparison: dict[str, object] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.status is RunStatus.COMPLETE

    @property
    def counts(self) -> dict[str, int | None]:
        return {
            "topologies": self.topology_count,
            "effective_topologies": self.effective_topology_count,
            "fixed_label_classes": self.fixed_label_classes,
            "particle_relabeling_classes": self.particle_relabeling_classes,
            "variable_combination_classes": self.variable_combination_count,
            "dn_classes": self.dn_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "complete": self.complete,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "counts": self.counts,
            "failure_counts": dict(self.failure_counts),
            "issues": [issue.to_dict() for issue in self.issues],
            "source_status": self.source_status,
            "source_references": [reference.to_dict() for reference in self.source_references],
            "runtime": self.runtime.to_dict(),
            "topology_report": self.topology_report.to_dict(),
            "labeling_reports": [report.to_dict() for report in self.labeling_reports],
            "combination_report": self.combination_report.to_dict() if self.combination_report else None,
            "coverage": self.coverage.to_dict(),
            'source_comparisons': list(self.source_comparisons),
            'combination_comparison': _jsonable(self.combination_comparison),
            'acceptance': {
                'enumeration_status': self.status.value,
                'reference_count_status': self.source_status,
                'catalog_status': 'not_checked',
                'note': 'complete means search completion, not scientific or catalog acceptance',
            },
            'topology_screening': [
                {'topology': r.topology.to_dict(),
                 'stage': ('search_' + r.status.value if not r.complete else
                           'accepted' if r.effective else
                           'independence_rejected' if r.momentum_valid_count > 0 else 'momentum_rejected'),
                 'failure_counts': dict(r.failure_counts)} for r in self.labeling_reports
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def to_markdown(self) -> str:
        """Render a compact human-readable summary without dropping JSON evidence."""

        lines = [
            "# 四体图枚举审计报告",
            "",
            f"- 状态：`{self.status.value}`",
            f"- 完整：`{str(self.complete).lower()}`",
            f"- 开始：`{self.started_at or '未记录'}`",
            f"- 结束：`{self.finished_at or '未记录'}`",
            f"- 来源状态：`{self.source_status}`",
            "",
            "## 分层计数",
            "",
            "| 层级 | 数量 |",
            "| --- | ---: |",
        ]
        lines.extend(f"| {name} | {value if value is not None else '未计算'} |" for name, value in self.counts.items())
        lines.extend(["", "## 运行与证据", "", f"- G05 状态：`{self.topology_report.status.value}`"])
        lines.append(f"- G06 报告数：`{len(self.labeling_reports)}`")
        lines.append(
            f"- G07 组合报告：`{self.combination_report.status.value if self.combination_report else 'missing'}`"
        )
        lines.append(f"- 规范链接数：`{len(self.coverage.links)}`")
        lines.append('- complete 仅表示声明范围内计算结束；来源逐图一致性需另行验收。')
        lines.append(f"- 组合集合检查：`{self.combination_comparison.get('status', 'not_checked')}`")
        lines.append(f"- 独立组合中图路线未覆盖的类数：{len(self.combination_comparison.get('combination_only', []))}")
        if self.source_comparisons:
            lines.extend(['', '## 外部计数验收', '', '| 来源 | 指标 | 期望 | 实际 | 状态 |',
                          '| --- | --- | ---: | ---: | --- |'])
            lines.extend(f"| {r['name']} | {r['metric']} | {r.get('expected')} | {r.get('actual')} | {r['status']} |"
                         for r in self.source_comparisons)
            lines.append('计数相同也不代表来源逐图一致；verified 仅指来源记录已核对。')
        if self.failure_counts:
            lines.extend(["", "## 失败原因", "", "| 原因码 | 数量 |", "| --- | ---: |"])
            lines.extend(f"| `{code}` | {count} |" for code, count in self.failure_counts)
        if self.issues:
            lines.extend(["", "## 完整性问题", ""])
            lines.extend(f"- `{issue.code}`：{issue.message}" for issue in self.issues)
        if self.source_references:
            lines.extend(["", "## 来源引用", "", "| 名称 | 范围 | 期望 | 已核验 |", "| --- | --- | --- | --- |"])
            lines.extend(
                f"| {reference.name} | {reference.scope} | {reference.expected} | "
                f"{str(reference.verified).lower()} |"
                for reference in self.source_references
            )
        lines.extend(["", "## 复现环境", ""])
        for key, value in self.runtime.to_dict().items():
            lines.append(f"- {key}：`{value}`")
        return "\n".join(lines) + "\n"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _source_status(comparisons) -> str:
    if not comparisons:
        return "not_provided"
    applicable = [r for r in comparisons if r['status'] != 'out_of_scope']
    if not applicable:
        return 'out_of_scope'
    if any(r['status'] == 'count_mismatch' for r in applicable):
        return 'count_mismatch'
    if all(r['status'] == 'count_match' for r in applicable):
        return 'count_match'
    return 'inconclusive'


def build_audit_report(
    topology_report: EnumerationReport,
    labeling_reports: Iterable[LabelingSearchReport],
    combination_report: CombinationSearchReport | None = None,
    *,
    source_references: Iterable[SourceReference] = (),
    runtime: RuntimeMetadata | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> AuditReport:
    """Combine G05, G06, and G07 artifacts without hiding incomplete layers."""

    if not isinstance(topology_report, EnumerationReport):
        raise TypeError("topology_report must be an EnumerationReport object")
    reports = tuple(labeling_reports)
    if not all(isinstance(report, LabelingSearchReport) for report in reports):
        raise TypeError("labeling_reports must contain LabelingSearchReport objects")
    references = tuple(source_references)
    if not all(isinstance(reference, SourceReference) for reference in references):
        raise TypeError("source_references must contain SourceReference objects")
    issues: list[AuditIssue] = []
    topology_set = set(topology_report.topologies)
    seen: set[Topology] = set()
    links = []
    for report in reports:
        if report.topology not in topology_set:
            issues.append(
                AuditIssue(
                    "labeling_topology_not_in_g05",
                    "A G06 labeling report does not belong to the supplied G05 topology report.",
                    {"topology": report.topology.to_dict()},
                )
            )
        if report.topology in seen:
            issues.append(
                AuditIssue(
                    "duplicate_labeling_report",
                    "More than one G06 report was supplied for the same topology.",
                    {"topology": report.topology.to_dict()},
                )
            )
        seen.add(report.topology)
        try:
            links.extend(link_labeling_report(report.topology, report))
        except (TypeError, ValueError) as exc:
            issues.append(AuditIssue("labeling_link_error", str(exc), {"topology": report.topology.to_dict()}))
    missing = topology_set - seen
    if missing:
        issues.append(
            AuditIssue(
                "missing_labeling_reports",
                "Not every G05 topology has a corresponding G06 labeling report.",
                {"missing_count": len(missing)},
            )
        )
    if topology_report.status is not RunStatus.COMPLETE:
        issues.append(
            AuditIssue(
                "topology_search_incomplete",
                "The G05 topology report is incomplete.",
                {"status": topology_report.status.value, "stop_reason": topology_report.stop_reason},
            )
        )
    for report in reports:
        if report.status is not RunStatus.COMPLETE:
            issues.append(
                AuditIssue(
                    "labeling_search_incomplete",
                    "At least one G06 labeling report is incomplete or failed.",
                    {"status": report.status.value, "stop_reason": report.stop_reason},
                )
            )
    if combination_report is None:
        issues.append(AuditIssue("missing_combination_report", "No G07 combination report was supplied."))
    elif combination_report.status is not RunStatus.COMPLETE:
        issues.append(
            AuditIssue(
                "combination_search_incomplete",
                "The G07 combination report is incomplete or failed.",
                {"status": combination_report.status.value},
            )
        )
    if (any(report.status is RunStatus.FAILED for report in reports)
            or topology_report.status is RunStatus.FAILED
            or (combination_report is not None and combination_report.status is RunStatus.FAILED)):
        status = RunStatus.FAILED
    elif issues:
        status = RunStatus.INCOMPLETE
    else:
        status = RunStatus.COMPLETE
    coverage = build_coverage_index(links)
    combination_comparison = compare_combination_coverage(
        coverage, combination_report, n=topology_report.n,
        size=topology_report.n - 2 + topology_report.loop_count,
        graph_complete=status is RunStatus.COMPLETE)
    if combination_comparison.get('reason') == 'missing_required_combination_scope':
        issues.append(AuditIssue('missing_required_combination_scope', 'G07 does not cover the requested n and d.'))
        if status is RunStatus.COMPLETE:
            status = RunStatus.INCOMPLETE
    elif combination_comparison['status'] == 'inconsistent':
        issues.append(AuditIssue('graph_variables_outside_independent_universe',
                                 'At least one graph variable key is absent from the complete independent universe.'))
        status = RunStatus.FAILED
    failure_counts: Counter[str] = Counter()
    for report in reports:
        failure_counts.update(dict(report.failure_counts))
    modes = {report.count_mode for report in reports}
    fixed = (sum(report.unique_count for report in reports if report.count_mode is CountMode.FIXED_LABELS)
             if CountMode.FIXED_LABELS in modes else None)
    relabeled = (sum(report.unique_count for report in reports if report.count_mode is CountMode.PARTICLE_RELABELING)
                 if CountMode.PARTICLE_RELABELING in modes else None)
    effective = sum(1 for report in reports if report.effective)
    variable_keys = {link.variable_key for link in links}
    dn_keys = {link.combination.dn.label for link in links}
    actual_counts = {'effective_topologies': effective, 'topologies': topology_report.unique_count,
                     'fixed_label_classes': fixed, 'particle_relabeling_classes': relabeled,
                     'variable_combination_classes': len(variable_keys), 'dn_classes': len(dn_keys)}
    comparisons = []
    for reference in references:
        comparable = (isinstance(reference.expected, int) and not isinstance(reference.expected, bool)
                      and reference.expected >= 0 and reference.metric in actual_counts)
        result = (compare_reference_count(
            actual_counts[reference.metric], reference.expected, status.value,
            scope_matches=(reference.n in (None, topology_report.n)
                           and reference.loop_count in (None, topology_report.loop_count)))
            if comparable else {'status': 'not_comparable', 'expected': reference.expected, 'actual': None})
        comparisons.append({'name': reference.name, 'metric': reference.metric,
                            'source_record_verified': reference.verified, **result})
    # Aggregation cannot reconstruct when an earlier computation ran.
    started_value = started_at
    finished_value = finished_at
    return AuditReport(
        status=status,
        started_at=started_value,
        finished_at=finished_value,
        topology_report=topology_report,
        labeling_reports=reports,
        combination_report=combination_report,
        coverage=coverage,
        source_references=references,
        source_status=_source_status(comparisons),
        runtime=runtime or collect_runtime_metadata(),
        topology_count=topology_report.unique_count,
        effective_topology_count=effective,
        fixed_label_classes=fixed,
        particle_relabeling_classes=relabeled,
        variable_combination_count=len(variable_keys),
        dn_count=len(dn_keys),
        failure_counts=tuple(sorted(failure_counts.items())),
        issues=tuple(issues),
        source_comparisons=tuple(comparisons),
        combination_comparison=combination_comparison,
    )


def run_audit(
    n: int = 4,
    loop_count: int = 0,
    *,
    count_mode: CountMode | str = CountMode.FIXED_LABELS,
    topology_limits: EnumerationLimits | None = None,
    labeling_limits: LabelingLimits | None = None,
    combination_sizes: Iterable[int] | None = None,
    use_pruning: bool = True,
    source_references: Iterable[SourceReference] = (),
    runtime: RuntimeMetadata | None = None,
    use_symmetry: bool = True,
) -> AuditReport:
    """Run G05, G06, and G07 in order and return one G08 audit artifact.

    Limits are passed to the corresponding layer.  The returned report keeps
    every layer's status, so a bounded run remains explicitly incomplete.
    """

    started_at = _timestamp()
    runtime = runtime or collect_runtime_metadata()
    topology_report = enumerate_topologies(
        n,
        loop_count,
        limits=topology_limits,
        use_pruning=use_pruning,
    )
    labeling_reports = tuple(
        enumerate_labelings(topology, count_mode=count_mode, limits=labeling_limits, use_symmetry=use_symmetry)
        for topology in topology_report.topologies
    )
    combination_report = enumerate_invariant_combinations(n, sizes=combination_sizes)
    return build_audit_report(
        topology_report,
        labeling_reports,
        combination_report,
        source_references=source_references,
        runtime=runtime,
        started_at=started_at,
        finished_at=_timestamp(),
    )


def write_audit_bundle(
    report: AuditReport,
    output_directory: str | Path,
    *,
    stem: str = "audit_report",
) -> tuple[Path, Path]:
    """Write deterministic JSON and Markdown audit artifacts and return their paths."""

    if not isinstance(report, AuditReport):
        raise TypeError("report must be an AuditReport object")
    if not isinstance(stem, str) or not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty file name without directory separators")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    json_path.write_text(report.to_json() + "\n", encoding="utf-8")
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "AuditIssue",
    "AuditReport",
    "RuntimeMetadata",
    "SourceReference",
    "build_audit_report",
    "collect_runtime_metadata",
    "run_audit",
    "write_audit_bundle",
]
