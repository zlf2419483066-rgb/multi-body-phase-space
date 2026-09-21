"""Per-topology catalogs and explicit completeness states (LM-06)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonicalize_topology
from .classification import classify_variable_combination
from .domain import CountMode, Labeling, Topology, ValidationIssue
from .label_identity import MarkedIdentity, canonicalize_physical_labeling
from .physical_labels import NormalizationResult, PhysicalLabelRecord, normalize_physical_labeling, review_physical_record
from .reporting import run_audit


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


def _mode(value: CountMode | str) -> CountMode:
    if isinstance(value, CountMode):
        return value
    try:
        return CountMode(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown count mode: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class SearchEvidence:
    """Evidence supplied by the search/import caller, kept separate from identities."""

    n: int | None = None
    loop_count: int | None = None
    count_mode: CountMode | None = None
    search_status: str = "not_run"
    records_complete: bool = False
    import_status: str = "not_applicable"
    source_rule_version: str | None = None
    raw_counts: tuple[tuple[str, int | None], ...] = ()
    provenance: tuple[tuple[str, Any], ...] = ()

    def __init__(self, data: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        values: dict[str, Any] = dict(data or {})
        values.update(kwargs)
        mode = values.get("count_mode")
        if mode is not None:
            mode = _mode(mode)
        raw_counts = values.get("raw_counts", {})
        if isinstance(raw_counts, Mapping):
            raw_counts = tuple((str(key), value) for key, value in raw_counts.items())
        else:
            raw_counts = tuple(raw_counts)
        provenance = values.get("provenance", {})
        if isinstance(provenance, Mapping):
            provenance = tuple((str(key), value) for key, value in provenance.items())
        else:
            provenance = tuple(provenance)
        object.__setattr__(self, "n", values.get("n"))
        object.__setattr__(self, "loop_count", values.get("loop_count"))
        object.__setattr__(self, "count_mode", mode)
        object.__setattr__(self, "search_status", values.get("search_status", "not_run"))
        object.__setattr__(
            self,
            "records_complete",
            values.get("records_complete", values.get("received_all_records", values.get("complete_records", False))),
        )
        object.__setattr__(self, "import_status", values.get("import_status", "not_applicable"))
        object.__setattr__(self, "source_rule_version", values.get("source_rule_version"))
        object.__setattr__(self, "raw_counts", tuple(raw_counts))
        object.__setattr__(self, "provenance", tuple(provenance))
        if not isinstance(self.records_complete, bool):
            raise TypeError("records_complete must be a boolean")
        if self.search_status not in {"complete", "incomplete", "failed", "not_run"}:
            raise ValueError(f"unknown search_status: {self.search_status!r}")
        if self.import_status not in {"complete", "incomplete", "failed", "not_applicable"}:
            raise ValueError(f"unknown import_status: {self.import_status!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "loop_count": self.loop_count,
            "count_mode": self.count_mode.value if self.count_mode else None,
            "search_status": self.search_status,
            "records_complete": self.records_complete,
            "import_status": self.import_status,
            "source_rule_version": self.source_rule_version,
            "raw_counts": {key: value for key, value in self.raw_counts},
            "provenance": {key: _jsonable(value) for key, value in self.provenance},
        }


@dataclass(frozen=True, slots=True)
class TopologyCatalogLink:
    """Link one identity to its independent variable combination and DN."""

    identity_key: tuple[Any, ...]
    variables: tuple[tuple[int, ...], ...]
    variable_key: tuple[Any, ...]
    dn_label: str

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_key": _jsonable(self.identity_key),
            "variables": [list(variable) for variable in self.variables],
            "variable_key": _jsonable(self.variable_key),
            "dn": self.dn_label,
        }


@dataclass(frozen=True, slots=True)
class TopologyLabelCatalog:
    """All reviewed identities for one topology/mode and their status evidence."""

    topology: Topology
    topology_key: tuple[Any, ...]
    mode: CountMode
    identities: tuple[MarkedIdentity, ...]
    rejected_witnesses: tuple[NormalizationResult, ...]
    links: tuple[TopologyCatalogLink, ...]
    existence: str
    search_status: str
    import_status: str
    enumeration_complete: bool
    default_label_key: tuple[Any, ...] | None
    search_evidence: SearchEvidence
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.existence not in {"yes", "no", "unknown"}:
            raise ValueError("existence must be yes, no, or unknown")
        if not isinstance(self.mode, CountMode):
            raise TypeError("mode must be a CountMode")
        object.__setattr__(self, "identities", tuple(self.identities))
        object.__setattr__(self, "rejected_witnesses", tuple(self.rejected_witnesses))
        object.__setattr__(self, "links", tuple(self.links))
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def identity_count(self) -> int:
        return len(self.identities)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "label_catalog",
            "schema_version": "IM-4-v1",
            "topology": self.topology.to_dict(),
            "topology_key": _jsonable(self.topology_key),
            "mode": self.mode.value,
            "identities": [identity.to_dict() for identity in self.identities],
            "identity_keys": [_jsonable(identity.key) for identity in self.identities],
            "rejected_witnesses": [witness.to_dict() for witness in self.rejected_witnesses],
            "links": [link.to_dict() for link in self.links],
            "existence": self.existence,
            "search_status": self.search_status,
            "import_status": self.import_status,
            "enumeration_complete": self.enumeration_complete,
            "default_label_key": _jsonable(self.default_label_key),
            "search_evidence": self.search_evidence.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class LabelCatalogBundle:
    """A serializable collection of topology catalogs and import diagnostics."""

    catalogs: tuple[TopologyLabelCatalog, ...]
    provenance: tuple[tuple[str, Any], ...] = ()
    issues: tuple[ValidationIssue, ...] = ()
    topology_scope: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "catalogs", tuple(self.catalogs))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "topology_scope", dict(self.topology_scope))

    @property
    def topology_scope_complete(self) -> bool:
        scope = self.topology_scope
        expected = {_restore_tuples(key) for key in scope.get("topology_keys", ())}
        actual = {catalog.topology_key for catalog in self.catalogs}
        return (scope.get("status") == "complete" and scope.get("n") == 4
                and isinstance(scope.get("loop_count"), int)
                and scope.get("mode") in {mode.value for mode in CountMode}
                and "topology_keys" in scope and expected == actual and not self.issues
                and all(c.mode.value == scope["mode"] and c.topology.n == scope["n"]
                        and c.topology.edge_count-len(c.topology.vertices)+1 == scope["loop_count"]
                        for c in self.catalogs))

    @property
    def enumeration_complete(self) -> bool:
        return self.topology_scope_complete and all(
            catalog.enumeration_complete and not catalog.issues for catalog in self.catalogs
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "label_catalog",
            "schema_version": "IM-4-v1",
            "catalogs": [catalog.to_dict() for catalog in self.catalogs],
            "provenance": {key: _jsonable(value) for key, value in self.provenance},
            "issues": [issue.to_dict() for issue in self.issues],
            "topology_scope": _jsonable(self.topology_scope),
        }


@dataclass(frozen=True, slots=True)
class FreshCatalogResult:
    """Fresh G08 audit plus its LM-07/06 converted catalog."""

    legacy_audit: Any
    label_catalog: LabelCatalogBundle

    def to_dict(self) -> dict[str, object]:
        return {
            "legacy_audit": self.legacy_audit.to_dict(),
            "label_catalog": self.label_catalog.to_dict(),
        }


def _as_normalization(value: NormalizationResult | PhysicalLabelRecord) -> NormalizationResult:
    if isinstance(value, NormalizationResult):
        return value
    if isinstance(value, PhysicalLabelRecord):
        stage = "accepted" if value.independence.independent else "dependent"
        return NormalizationResult(stage=stage, record=value)
    raise TypeError("normalization_results must contain NormalizationResult or PhysicalLabelRecord")


def build_topology_catalog(
    topology: Topology,
    normalization_results: Iterable[NormalizationResult | PhysicalLabelRecord],
    *,
    mode: CountMode | str,
    search_evidence: SearchEvidence | Mapping[str, Any] | None = None,
) -> TopologyLabelCatalog:
    """Build a catalog without treating partial review as complete search."""

    if not isinstance(topology, Topology):
        raise TypeError("build_topology_catalog expects a Topology")
    selected_mode = _mode(mode)
    evidence = search_evidence if isinstance(search_evidence, SearchEvidence) else SearchEvidence(search_evidence)
    expected_key = canonicalize_topology(topology).key
    identities: dict[tuple[Any, ...], MarkedIdentity] = {}
    witnesses: list[NormalizationResult] = []
    issues: list[ValidationIssue] = []
    if evidence.count_mode is not None and evidence.count_mode is not selected_mode:
        issues.append(ValidationIssue("evidence_mode_mismatch", "Search evidence belongs to another counting mode."))
    if ((evidence.n is not None and evidence.n != topology.n)
            or (evidence.loop_count is not None and evidence.loop_count != topology.edge_count-len(topology.vertices)+1)):
        issues.append(ValidationIssue("evidence_scope_mismatch", "Search evidence belongs to another n or loop count."))
    for raw in normalization_results:
        result = _as_normalization(raw)
        if result.record is not None:
            try:
                checked = review_physical_record(result.record)
                if checked.stage != result.stage:
                    raise ValueError("normalization stage disagrees with its physical record")
                if canonicalize_topology(result.record.topology).key != expected_key:
                    raise ValueError("record belongs to another topology")
            except (TypeError, ValueError) as exc:
                issues.append(ValidationIssue("record_review_failed", str(exc)))
                continue
        if result.stage != "accepted" or result.record is None:
            witnesses.append(result)
            continue
        try:
            identity = canonicalize_physical_labeling(result.record, selected_mode)
        except (TypeError, ValueError) as exc:
            issues.append(ValidationIssue("identity_construction_failed", str(exc)))
            witnesses.append(result)
            continue
        if identity.topology_key != expected_key:
            issues.append(ValidationIssue("topology_key_mismatch", "An accepted record belongs to another topology key."))
            witnesses.append(result)
            continue
        identities.setdefault(identity.key, identity)

    ordered_identities = tuple(identities[key] for key in sorted(identities))
    links: list[TopologyCatalogLink] = []
    for identity in ordered_identities:
        variables = identity.canonical_record.momentum.variables
        try:
            combination = classify_variable_combination(variables, topology.n)
        except (TypeError, ValueError) as exc:
            issues.append(ValidationIssue("link_construction_failed", str(exc)))
            continue
        links.append(
            TopologyCatalogLink(
                identity_key=identity.key,
                variables=tuple(tuple(variable) for variable in variables),
                variable_key=combination.canonical.key,
                dn_label=combination.dn.label,
            )
        )

    search_complete = evidence.search_status == "complete" and evidence.records_complete
    import_ok = evidence.import_status in {"complete", "not_applicable"}
    enumeration_complete = bool(search_complete and import_ok and not issues)
    if ordered_identities:
        existence = "yes"
    elif enumeration_complete:
        existence = "no"
    else:
        existence = "unknown"
    return TopologyLabelCatalog(
        topology=topology,
        topology_key=expected_key,
        mode=selected_mode,
        identities=ordered_identities,
        rejected_witnesses=tuple(witnesses),
        links=tuple(links),
        existence=existence,
        search_status=evidence.search_status,
        import_status=evidence.import_status,
        enumeration_complete=enumeration_complete,
        default_label_key=ordered_identities[0].key if ordered_identities else None,
        search_evidence=evidence,
        issues=tuple(issues),
    )


def _audit_hash(audit_mapping: Mapping[str, Any]) -> str:
    payload = json.dumps(_jsonable(audit_mapping), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _issue_from_dict(data: Mapping[str, Any]) -> ValidationIssue:
    return ValidationIssue(
        data.get("code", "invalid_import_issue"),
        data.get("message", "invalid import issue"),
        data.get("path", ""),
        data.get("severity", "error"),
    )


def _normalization_from_identity(data: Mapping[str, Any]) -> NormalizationResult:
    record_data = data.get("canonical_record")
    if not isinstance(record_data, Mapping):
        raise ValueError("identity is missing canonical_record")
    topology = Topology.from_dict(record_data["topology"])
    labeling = Labeling.from_dict(record_data["complete_labeling"])
    checked = normalize_physical_labeling(topology, labeling, CountMode.FIXED_LABELS)
    if checked.record is None or checked.stage not in {"accepted", "dependent"}:
        raise ValueError("canonical_record failed LM-02 during catalog read")
    if checked.record.to_dict() != record_data:
        raise ValueError("stored physical certificate differs from recomputed evidence")
    return NormalizationResult(checked.stage, record=checked.record, issues=checked.issues)


def _normalization_from_witness(data: Mapping[str, Any], topology: Topology) -> NormalizationResult:
    record_data = data.get("record")
    if isinstance(record_data, Mapping):
        checked = normalize_physical_labeling(
            Topology.from_dict(record_data["topology"]),
            Labeling.from_dict(record_data["complete_labeling"]),
        )
        if checked.record is None or checked.record.to_dict() != record_data:
            raise ValueError("stored witness certificate differs from recomputed evidence")
        return checked
    evaluation = data.get("legacy_evaluation")
    if isinstance(evaluation, Mapping) and isinstance(evaluation.get("labeling"), Mapping):
        return normalize_physical_labeling(topology, Labeling.from_dict(evaluation["labeling"]))
    issues = tuple(_issue_from_dict(item) for item in data.get("issues", ()))
    return NormalizationResult(data.get("stage", "momentum_rejected"), issues=issues)


def _with_import_issues(catalog: TopologyLabelCatalog, issues: Iterable[ValidationIssue]) -> TopologyLabelCatalog:
    combined = tuple(catalog.issues) + tuple(issues)
    if not combined:
        return catalog
    # Persist the downgrade in evidence as well as the displayed status, so a
    # second save/read cannot upgrade a damaged artifact to complete again.
    evidence = SearchEvidence(catalog.search_evidence.to_dict(), import_status="failed", records_complete=False)
    return replace(catalog, issues=combined, import_status="failed", enumeration_complete=False,
                   existence="yes" if catalog.identities else "unknown", search_evidence=evidence)


def read_label_catalog(path: str | Path) -> LabelCatalogBundle:
    """Recheck stored physical records and preserve all import conflicts."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "label_catalog" or payload.get("schema_version") != "IM-4-v1":
        raise ValueError("unsupported label catalog artifact or schema version")
    catalogs = []
    bundle_issues = [_issue_from_dict(item) for item in payload.get("issues", ())]
    for raw_catalog in payload.get("catalogs", ()):
        topology = Topology.from_dict(raw_catalog["topology"])
        mode = _mode(raw_catalog.get("mode"))
        evidence = SearchEvidence(raw_catalog.get("search_evidence", {}))
        results = []
        issues = [_issue_from_dict(item) for item in raw_catalog.get("issues", ())]
        stored_keys = raw_catalog.get("identity_keys")
        row_keys = [identity.get("key") for identity in raw_catalog.get("identities", ())]
        if stored_keys is None or stored_keys != row_keys:
            issues.append(ValidationIssue("stored_identity_manifest_mismatch", "Stored identity manifest is missing or differs from available rows; completeness is unverified."))
        for identity_data in raw_catalog.get("identities", ()):
            try:
                normalization = _normalization_from_identity(identity_data)
                results.append(normalization)
                parsed = canonicalize_physical_labeling(normalization.record, mode)
                if normalization.stage != "accepted":
                    issues.append(ValidationIssue("stored_identity_not_accepted", "Stored effective identity is not independent."))
                if _restore_tuples(identity_data.get("key")) != parsed.key:
                    issues.append(ValidationIssue("stored_identity_key_mismatch", "stored identity key differs from rebuilt key"))
                if identity_data.get("digest") != parsed.digest:
                    issues.append(ValidationIssue("stored_identity_digest_mismatch", "stored identity digest differs from rebuilt key"))
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(ValidationIssue("stored_identity_rejected", str(exc)))
        for witness in raw_catalog.get("rejected_witnesses", ()):
            try:
                checked = _normalization_from_witness(witness, topology)
                results.append(checked)
                if checked.stage != witness.get("stage"):
                    issues.append(ValidationIssue("stored_witness_stage_mismatch", "Stored rejection changed under physical review."))
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(ValidationIssue("stored_witness_rejected", str(exc)))
        rebuilt = build_topology_catalog(topology, results, mode=mode, search_evidence=evidence)
        if _restore_tuples(raw_catalog.get("topology_key")) != rebuilt.topology_key:
            issues.append(ValidationIssue("stored_topology_key_mismatch", "stored topology_key differs from rebuilt key"))
        stored_default = _restore_tuples(raw_catalog.get("default_label_key"))
        if stored_default != rebuilt.default_label_key:
            issues.append(ValidationIssue("stored_default_key_mismatch", "stored default_label_key differs from rebuilt key"))
        catalogs.append(_with_import_issues(rebuilt, issues))
    provenance = payload.get("provenance", {})
    scope = payload.get("topology_scope", {})
    if "topology_keys" in scope and {_restore_tuples(k) for k in scope["topology_keys"]} != {c.topology_key for c in catalogs}:
        bundle_issues.append(ValidationIssue("stored_scope_mismatch", "Stored topology scope does not match catalog entries."))
    return LabelCatalogBundle(tuple(catalogs), tuple(provenance.items()), tuple(bundle_issues), scope)


def _restore_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_restore_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _restore_tuples(item) for key, item in value.items()}
    return value


def write_label_catalog(bundle: LabelCatalogBundle, path: str | Path) -> Path:
    """Write a catalog with exclusive creation so old evidence cannot be overwritten."""

    if not isinstance(bundle, LabelCatalogBundle):
        raise TypeError("write_label_catalog expects a LabelCatalogBundle")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(bundle.to_dict(), handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return destination


def catalog_from_audit(
    audit_mapping: Mapping[str, Any], *, mode: CountMode | str,
    source_info: Mapping[str, Any] | None = None,
) -> LabelCatalogBundle:
    """Review saved records, retaining search scope separately from labels."""
    if not isinstance(audit_mapping, Mapping):
        raise TypeError("catalog_from_audit expects a decoded audit mapping")
    selected_mode = _mode(mode)
    reports = audit_mapping.get("labeling_reports", ())
    if not isinstance(reports, (list, tuple)):
        raise ValueError("audit_mapping.labeling_reports must be a list")
    topology_report = audit_mapping.get("topology_report") or {}
    info = dict(source_info or {})
    info.setdefault("source_mapping_sha256", _audit_hash(audit_mapping))
    info.setdefault("source_rule_version", (audit_mapping.get("runtime") or {}).get("rule_version", "unknown"))
    info.update({"converted_at": datetime.now(timezone.utc).isoformat(),
                 "source_finished_at": audit_mapping.get("finished_at"),
                 "source_status": audit_mapping.get("status", "unknown")})
    grouped = {}
    global_issues = []
    expected = {}
    for raw in topology_report.get("topologies", ()):
        top = Topology.from_dict(raw)
        expected[canonicalize_topology(top).key] = top
    for report in reports:
        if not isinstance(report, Mapping) or not isinstance(report.get("topology"), Mapping):
            global_issues.append(ValidationIssue("missing_topology_report", "legacy report has no topology witness"))
            continue
        top = Topology.from_dict(report["topology"])
        key = canonicalize_topology(top).key
        entry = grouped.setdefault(key, {"topology":top, "results":[], "status":"complete", "records_complete":True, "issues":[], "reports":[], "witnesses_complete":True})
        entry["reports"].append(report)
        old_status = report.get("status", "not_run")
        if old_status != "complete":
            entry["status"] = "failed" if old_status == "failed" else "incomplete"
            entry["records_complete"] = False
        old_mode = report.get("count_mode")
        if old_mode not in {m.value for m in CountMode}:
            entry["records_complete"] = False
            entry["issues"].append(ValidationIssue("source_mode_unknown", "Missing or unknown source counting mode."))
        elif old_mode == "particle_relabeling" and selected_mode is CountMode.FIXED_LABELS:
            entry["records_complete"] = False
            # Do not silently expand a physical species convention. The saved
            # representatives remain useful, but do not cover fixed labels.
        rows = report.get("valid_results", ())
        declared_count = (report.get("counts") or {}).get("unique")
        if not isinstance(rows, (list, tuple)):
            rows = ()
            entry["records_complete"] = False
        if declared_count is None or declared_count != len(rows):
            entry["records_complete"] = False
            entry["issues"].append(ValidationIssue("legacy_record_count_unverified", "Saved effective records do not have matching declared unique count."))
        for raw in rows:
            try:
                checked = normalize_physical_labeling(top, Labeling.from_dict(raw["labeling"]), selected_mode)
                entry["results"].append(checked)
                if raw.get("valid") is not True or checked.stage != "accepted":
                    entry["issues"].append(ValidationIssue("legacy_validity_conflict", "stored valid result failed current LM-02 review"))
            except (KeyError, TypeError, ValueError) as exc:
                entry["issues"].append(ValidationIssue("legacy_record_rejected", str(exc)))
        witnesses = report.get("failure_witnesses") or {}
        for code, raw in witnesses.items():
            try:
                checked = normalize_physical_labeling(top, Labeling.from_dict(raw["labeling"]), selected_mode)
                entry["results"].append(checked)
                if checked.stage == "accepted" or code not in {issue.code for issue in checked.issues}:
                    entry["issues"].append(ValidationIssue("legacy_failure_conflict", "Failure witness no longer reproduces its rejection code."))
            except (KeyError, TypeError, ValueError) as exc:
                entry["issues"].append(ValidationIssue("legacy_witness_rejected", str(exc)))
        missing_codes = set(report.get("failure_counts") or {}) - set(witnesses)
        if missing_codes:
            entry["witnesses_complete"] = False
            entry["results"].append(NormalizationResult("momentum_rejected", issues=(ValidationIssue("missing_failure_witness", "No replayable labeling for: " + ", ".join(sorted(missing_codes))),)))
    if "topologies" in topology_report:
        for key in set(expected) - set(grouped):
            global_issues.append(ValidationIssue("missing_labeling_report", "A generated topology has no labeling report."))
            grouped[key] = {"topology":expected[key], "results":[], "status":"not_run", "records_complete":False, "issues":[], "reports":[], "witnesses_complete":False}
        if set(grouped) - set(expected):
            global_issues.append(ValidationIssue("topology_outside_scope", "A labeling report is outside the declared topology scope."))
    catalogs = []
    for entry in grouped.values():
        top = entry["topology"]
        provenance = dict(info, legacy_reports=entry["reports"], failure_witnesses_complete=entry["witnesses_complete"])
        evidence = SearchEvidence(
            n=top.n, loop_count=top.edge_count-len(top.vertices)+1, count_mode=selected_mode,
            search_status=entry["status"], records_complete=entry["records_complete"],
            import_status="complete", source_rule_version=info["source_rule_version"],
            raw_counts=audit_mapping.get("counts", {}), provenance=provenance,
        )
        catalog = build_topology_catalog(top, entry["results"], mode=selected_mode, search_evidence=evidence)
        catalogs.append(_with_import_issues(catalog, entry["issues"]))
    scope = {"n":topology_report.get("n"), "loop_count":topology_report.get("loop_count"),
             "mode":selected_mode.value, "status":topology_report.get("status", "unknown")}
    if "topologies" in topology_report:
        scope["topology_keys"] = list(expected)
    return LabelCatalogBundle(tuple(sorted(catalogs, key=lambda c:c.topology_key)), tuple(info.items()), tuple(global_issues), scope)


def run_label_catalog(
    n: int = 4,
    loop_count: int = 0,
    *,
    mode: CountMode | str,
    topology_limits: Any = None,
    labeling_limits: Any = None,
    use_symmetry: bool = True,
) -> FreshCatalogResult:
    """Run the existing G05→G08 pipeline and immediately adapt its records."""

    selected_mode = _mode(mode)
    audit = run_audit(
        n=n,
        loop_count=loop_count,
        count_mode=selected_mode,
        topology_limits=topology_limits,
        labeling_limits=labeling_limits,
        use_symmetry=use_symmetry,
    )
    audit_mapping = audit.to_dict()
    source_info = {
        "source_kind": "fresh_run",
        "source_rule_version": "IM-4-v1",
        "n": n,
        "loop_count": loop_count,
        "operations": ("run_existing_g05_g06_g07_g08", "lm02_recheck", "lm04_lm05_identity", "lm06_catalog"),
        "legacy_status": audit_mapping.get("status", "unknown"),
    }
    bundle = catalog_from_audit(audit_mapping, mode=selected_mode, source_info=source_info)
    return FreshCatalogResult(legacy_audit=audit, label_catalog=bundle)


__all__ = [
    "SearchEvidence",
    "TopologyCatalogLink",
    "TopologyLabelCatalog",
    "LabelCatalogBundle",
    "build_topology_catalog",
    "catalog_from_audit",
    "read_label_catalog",
    "write_label_catalog",
    "FreshCatalogResult",
    "run_label_catalog",
]
