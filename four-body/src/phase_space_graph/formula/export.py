"""Explicit selection and Markdown export for formula artefacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any
import os
import tempfile

from .records import FormulaArtifact


@dataclass(frozen=True, slots=True)
class FormulaSelection:
    indices: tuple[int, ...] | None = None
    marking: str = "current"

    def __post_init__(self) -> None:
        if self.marking not in {"current", "representative", "all"}:
            raise ValueError("marking must be current, representative, or all")
        if self.indices is not None:
            if isinstance(self.indices, (str, bytes)):
                raise TypeError("formula selection indices must be integers")
            try:
                values = tuple(self.indices)
            except TypeError as exc:
                raise TypeError("formula selection indices must be an iterable of integers") from exc
            if any(isinstance(index, bool) or not isinstance(index, int) for index in values):
                raise TypeError("formula selection indices must be integers, not bool or float")
            object.__setattr__(self, "indices", values)


def _freeze(value: Any) -> Any:
    """Make serialized identity fields usable as a grouping key."""

    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _identity_value(identity: Any, name: str) -> Any:
    if identity is None:
        raise ValueError("representative selection requires identity.mode and identity.topology_key")
    if isinstance(identity, Mapping):
        if name not in identity:
            raise ValueError("representative selection requires identity.mode and identity.topology_key")
        return identity[name]
    if not hasattr(identity, name):
        raise ValueError("representative selection requires identity.mode and identity.topology_key")
    return getattr(identity, name)


def select_artifacts(artifacts: Sequence[FormulaArtifact], selection: FormulaSelection | None = None) -> tuple[FormulaArtifact, ...]:
    selection = selection or FormulaSelection()
    source = tuple(artifacts)
    if selection.indices is None:
        if selection.marking == "current":
            chosen = source[:1]
        else:
            chosen = source
    else:
        if any(index < 0 or index >= len(source) for index in selection.indices):
            raise IndexError("formula selection index out of range")
        if selection.marking == "current" and len(selection.indices) > 1:
            raise ValueError("current marking requires at most one explicit index")
        chosen = tuple(source[index] for index in selection.indices)

    if selection.marking != "representative":
        return chosen

    representatives: dict[tuple[Any, Any], FormulaArtifact] = {}
    for artifact in chosen:
        identity = artifact.identity
        mode = _identity_value(identity, "mode")
        topology_key = _identity_value(identity, "topology_key")
        if mode not in {"fixed_labels", "particle_relabeling"} or not topology_key:
            raise ValueError("representative selection requires a known mode and nonempty topology_key")
        group = (_freeze(mode), _freeze(topology_key))
        representatives.setdefault(group, artifact)
    return tuple(representatives.values())


def _append_steps(lines: list[str], title: str, steps: Sequence[Any]) -> None:
    if not steps:
        return
    lines.extend([f"### {title}", ""])
    for step in steps:
        before = getattr(step, "before", "")
        after = getattr(step, "after", "")
        rule_id = getattr(step, "rule_id", "rule")
        lines.append(f"- `{rule_id}`: `{before}` → `{after}`")
        eliminated = tuple(getattr(step, "eliminated", ()) or ())
        if eliminated:
            lines.append(f"  - 消元：{', '.join(map(str, eliminated))}")
        jacobian = getattr(step, "jacobian", None)
        if jacobian:
            lines.append(f"  - Jacobian：`{jacobian}`")
        conditions = tuple(getattr(step, "conditions", ()) or ())
        if conditions:
            lines.append(f"  - 条件：{'；'.join(map(str, conditions))}")
        evidence = getattr(step, "evidence", None)
        if evidence:
            lines.append(f"  - 证据：`{evidence}`")
    lines.append("")


def _append_identity(lines: list[str], artifact: FormulaArtifact) -> None:
    identity = artifact.identity
    lines.extend(["### 身份", ""])
    if identity is None:
        lines.append("- 未提供身份记录。")
    else:
        for key in ("mode", "topology_key", "key", "digest"):
            try:
                value = _identity_value(identity, key)
            except ValueError:
                continue
            lines.append(f"- {key}：`{value}`")
    lines.append("")


def _append_input(lines: list[str], artifact: FormulaArtifact) -> None:
    card = getattr(artifact, "input_card", None)
    if card is None:
        return
    if hasattr(card, "to_dict"):
        card = card.to_dict()
    if not isinstance(card, Mapping):
        return
    source = card.get("provenance")
    lines.extend(["### 输入来源", ""])
    if source:
        lines.append(f"- 来源：`{source}`")
    else:
        lines.append("- 来源：未提供")
    if artifact.input_digest:
        lines.append(f"- 输入摘要：`{artifact.input_digest}`")
    lines.append("")
    measures = card.get("double_measures") or ()
    particle_measures = card.get("particle_measures") or ()
    if measures or particle_measures:
        lines.extend(["### 输入测度", ""])
        for measure in measures:
            lines.append(f"- 双线测度：`{measure}`")
        for measure in particle_measures:
            lines.append(f"- 粒子测度：`{measure}`")
        lines.append("")


def _render(artifact: FormulaArtifact, index: int) -> str:
    d = artifact.direct_formula
    b = artifact.bridge_formula
    lines = [f"## Formula {index}", "", f"- 状态：`{artifact.statuses}`", f"- 直接主路已验证：{'是' if artifact.direct_verified else '否'}", ""]
    _append_identity(lines, artifact)
    _append_input(lines, artifact)
    if artifact.raw_expression:
        raw = artifact.raw_expression
        lines += ["### F0 原图表达式", "", f"`{raw.expression}`", ""]
        if raw.factors:
            lines += ["- 因子：", *[f"  - `{item}`" for item in raw.factors], ""]
        if raw.double_measures:
            lines += ["- 双线测度：", *[f"  - `{item}`" for item in raw.double_measures], ""]
        if raw.cancellations:
            lines += ["- 抵消：", *[f"  - `{item}`" for item in raw.cancellations], ""]
        for name in ("factor_sources", "measure_sources", "cancellation_sources"):
            values = getattr(raw, name, ())
            if values:
                lines += [f"- {name}：", *[f"  - `{item}`" for item in values], ""]
        _append_steps(lines, "F0 推导过程", raw.steps)
    if artifact.covariant_result:
        cov = artifact.covariant_result
        lines += ["### F_cov 协变中间式", "", f"`{cov.expression}`", ""]
        if cov.remaining_constraints:
            lines += ["- 剩余约束：", *[f"  - `{item}`" for item in cov.remaining_constraints], ""]
        _append_steps(lines, "协变消元过程", cov.steps)
        if cov.diagnostics:
            lines += ["- 诊断：", *[f"  - {item}" for item in cov.diagnostics], ""]
    if d:
        lines += ["### F1_direct", "", f"- profile：`{d.profile}`", f"- 变量：`{', '.join(d.target_variables + d.angle_variables)}`", f"- 密度：`{d.density_expression}`", f"- Jacobian：`{d.jacobian}`", f"- 状态：`{d.status}`"]
        if getattr(d, "normalization", ""):
            lines.append(f"- 归一化：`{d.normalization}`")
        if getattr(d, "differential_measure", ""):
            lines.append(f"- 微分测度：`{d.differential_measure}`")
        if getattr(d, "angle_definitions", ()):
            lines += ["- 角度定义：", *[f"  - `{item}`" for item in d.angle_definitions]]
        lines.append("")
        if d.domain:
            lines += ["- 域：", *[f"  - {x}" for x in d.domain], ""]
        if d.branches:
            lines += ["- 分支：", *[f"  - `{item.key}`：{item.root_condition}" for item in d.branches], ""]
            for branch in d.branches:
                lines += [f"  - 分支 `{branch.key}` 方程：", *[f"    - `{equation}`" for equation in branch.equations]]
                lines += [f"  - 分支 `{branch.key}` 回代：", *[f"    - `{item}`" for item in branch.back_substitution]]
                if branch.domain:
                    lines += [f"  - 分支 `{branch.key}` 域：", *[f"    - {item}" for item in branch.domain]]
                lines.append(f"  - 分支 `{branch.key}` 重数／系数／求和：`{branch.multiplicity}` / `{branch.coefficient}` / `{branch.summed}`")
                lines.append("")
        if d.diagnostics:
            lines += ["- 诊断：", *[f"  - {x}" for x in d.diagnostics], ""]
    if b:
        lines += ["### F_bridge（参考附录）", "", f"- profile：`{b.profile}`", f"- 变量：`{', '.join(b.variables)}`", f"- 密度：`{b.density_expression}`", f"- 状态：`{b.status}`"]
        if getattr(b, "normalization", ""):
            lines.append(f"- 归一化：`{b.normalization}`")
        if getattr(b, "differential_measure", ""):
            lines.append(f"- 微分测度：`{b.differential_measure}`")
        if getattr(b, "angle_definitions", ()):
            lines += ["- 角度定义：", *[f"  - `{item}`" for item in b.angle_definitions]]
        lines.append("")
        if b.constraints:
            lines += ["- 约束：", *[f"  - {x}" for x in b.constraints], ""]
    if artifact.internal_benchmark:
        report = artifact.internal_benchmark
        lines += ["### 内部 benchmark", "", f"- 状态：`{report.status}`", f"- 点数／比较数：{report.points}/{report.compared}", f"- 最大相对残差：`{report.max_relative_residual}`", f"- 坐标最大残差：`{report.coordinate_residual}`", ""]
        if report.failures:
            lines += ["- 失败见证：", *[f"  - `{item}`" for item in report.failures], ""]
        if report.diagnostics:
            lines += ["- 诊断：", *[f"  - {item}" for item in report.diagnostics], ""]
        if getattr(report, "evidence", None):
            lines += ["- 证据：", *[f"  - `{key}`：`{value}`" for key, value in report.evidence.items()], ""]
    if artifact.diagnostics:
        lines += ["### 诊断", "", *[f"- {x}" for x in artifact.diagnostics], ""]
    machine = artifact.to_dict()
    lines += ["### 机器记录", "", "```json", json.dumps(machine, ensure_ascii=False, sort_keys=True, allow_nan=False), "```", ""]
    return "\n".join(lines)


def export_markdown(
    artifacts: Sequence[FormulaArtifact],
    destination: str | os.PathLike[str],
    *,
    selection: FormulaSelection | None = None,
) -> Path:
    chosen = select_artifacts(artifacts, selection)
    if not chosen:
        raise ValueError("empty formula selection cannot be exported as a complete result")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "# 四体相空间公式导出\n\n" + "\n\n".join(_render(item, i) for i, item in enumerate(chosen, 1)) + "\n"
    # Prepare the complete file before reserving the destination.  The final
    # hard-link is exclusive, so an existing export is never replaced.
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.link(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    else:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
    return path


__all__ = ["FormulaSelection", "select_artifacts", "export_markdown"]
