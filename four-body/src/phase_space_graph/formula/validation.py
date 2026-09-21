"""Layered validation reports for formula artefacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from .records import FormulaArtifact


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    check: str
    status: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "status": self.status, "message": self.message}


def validate_artifact(artifact: FormulaArtifact) -> tuple[ValidationFinding, ...]:
    if not isinstance(artifact, FormulaArtifact):
        raise TypeError("validate_artifact expects a FormulaArtifact")
    findings: list[ValidationFinding] = []
    findings.append(ValidationFinding("f0", artifact.statuses.get("f0", "missing"), "raw graph expression is preserved"))
    findings.append(ValidationFinding("covariant", artifact.statuses.get("covariant", "missing"), "covariant intermediate is tracked separately"))
    direct_status = artifact.statuses.get("direct", "missing")
    findings.append(ValidationFinding("direct", direct_status, "direct IKV result"))
    bridge_status = artifact.statuses.get("bridge", "missing")
    findings.append(ValidationFinding("bridge", bridge_status, "recursive reference result"))
    benchmark_status = artifact.statuses.get("benchmark", "missing")
    findings.append(ValidationFinding("benchmark", benchmark_status, "coordinate-aligned internal comparison"))
    if direct_status == "verified" and artifact.direct_formula is None:
        findings.append(ValidationFinding("direct-integrity", "failed", "direct status says verified but no direct formula is present"))
    if direct_status != "verified" and bridge_status == "verified":
        findings.append(ValidationFinding("status-separation", "passed", "bridge status does not upgrade direct status"))
    if artifact.internal_benchmark and artifact.internal_benchmark.status == "failed":
        findings.append(ValidationFinding("benchmark-failure", "failed", "benchmark failure is retained as a diagnostic"))
    objects = {"f0": artifact.raw_expression, "covariant": artifact.covariant_result,
               "direct": artifact.direct_formula, "bridge": artifact.bridge_formula,
               "benchmark": artifact.internal_benchmark}
    for name, value in objects.items():
        status = artifact.statuses.get(name)
        if (value is None and status in {"built", "verified"}) or (
                value is not None and status != value.status):
            findings.append(ValidationFinding(name + "-integrity", "failed", "status disagrees with stored result"))
    if artifact.input_card is None or artifact.input_digest is None:
        findings.append(ValidationFinding("input-provenance", "failed", "input card or fingerprint is missing"))
    else:
        digest = hashlib.sha256(json.dumps(artifact.input_card, ensure_ascii=False,
                               sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if digest != artifact.input_digest or artifact.input_card.get("identity") != artifact.identity:
            findings.append(ValidationFinding("input-provenance", "failed", "input fingerprint or identity mismatch"))
    for name, value in (("direct", artifact.direct_formula), ("bridge", artifact.bridge_formula)):
        if value is not None and value.status in {"built", "verified"}:
            if value.normalization != artifact.options.normalization:
                findings.append(ValidationFinding(name + "-normalization", "failed", "mixed normalization conventions"))
            if not value.differential_measure or not value.angle_definitions:
                findings.append(ValidationFinding(name + "-measure", "failed", "density lacks explicit angular measure"))
    report = artifact.internal_benchmark
    if report is not None and report.status == "verified":
        residuals = (report.max_absolute_residual, report.max_relative_residual, report.coordinate_residual)
        if (report.points <= 0 or report.compared != report.points or report.failures or
                any(r is None or not math.isfinite(r) or r < 0 for r in residuals)):
            findings.append(ValidationFinding("benchmark-evidence", "failed", "empty, incomplete, or nonfinite benchmark"))
    if direct_status == "verified" and not artifact.direct_verified:
        findings.append(ValidationFinding("direct-completion", "failed", "required derivation and benchmark evidence is absent"))
    findings.append(ValidationFinding("source", artifact.statuses.get("source", "pending"),
                                     "external source comparison is separate from local benchmark"))
    return tuple(findings)


def render_validation_markdown(artifact: FormulaArtifact) -> str:
    rows = validate_artifact(artifact)
    lines = ["### 分层验证", "", "| 检查 | 状态 | 说明 |", "| --- | --- | --- |"]
    lines.extend(f"| {r.check} | {r.status} | {r.message} |" for r in rows)
    return "\n".join(lines)


__all__ = ["ValidationFinding", "validate_artifact", "render_validation_markdown"]
