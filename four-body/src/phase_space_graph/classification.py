"""G07 invariant-combination classification and graph-route coverage.

The module keeps the combination route separate from G06's graph search.  A
combination is first checked with the exact G02 rank test, then canonicalized
under one common particle permutation by G04, and finally assigned a DN
frequency signature.  Valid G06 results can be linked to the same combination
records without turning DN into an equivalence key.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import json
from math import comb
from typing import Any, Iterable, Mapping

from .canonical import (
    CanonicalizationResult,
    VariableCombinationResult,
    canonicalize_topology,
    canonicalize_variable_set,
)
from .domain import RunStatus, Topology
from .invariants import (
    IndependenceResult,
    Invariant,
    analyze_invariants,
    candidate_invariants,
    invariant_label,
)
from .labeling import LabelingResult, LabelingSearchReport


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _normalize_variables(variables: Iterable[Iterable[int]], n: int) -> tuple[Invariant, ...]:
    try:
        normalized = tuple(tuple(sorted(variable)) for variable in variables)
    except TypeError as exc:
        raise TypeError("variables must be an iterable of particle subsets") from exc
    candidates = set(candidate_invariants(n))
    if not normalized:
        raise ValueError("a variable combination cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("a variable combination cannot repeat an invariant")
    unknown = tuple(variable for variable in normalized if variable not in candidates)
    if unknown:
        raise ValueError(f"variables are not G02 candidates for n={n}: {unknown!r}")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class DNSignature:
    """DN classification data for one variable combination.

    ``frequency_profile`` includes zero-frequency particles and is sorted in
    descending order, so ``(3;7;2221)`` is represented without losing a
    particle that does not occur in the combination.
    """

    n: int
    variable_count: int
    occurrence_count: int
    frequency_profile: tuple[int, ...]
    particle_frequencies: tuple[tuple[int, int], ...]

    @property
    def key(self) -> tuple[int, int, tuple[int, ...]]:
        return (self.variable_count, self.occurrence_count, self.frequency_profile)

    @property
    def label(self) -> str:
        profile = "".join(str(value) for value in self.frequency_profile)
        return f"({self.variable_count};{self.occurrence_count};{profile})"

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "variable_count": self.variable_count,
            "occurrence_count": self.occurrence_count,
            "frequency_profile": list(self.frequency_profile),
            "particle_frequencies": [
                {"particle": particle, "count": count}
                for particle, count in self.particle_frequencies
            ],
            "key": _jsonable(self.key),
            "label": self.label,
        }


def dn_signature(variables: Iterable[Iterable[int]], n: int) -> DNSignature:
    """Compute a DN signature while retaining all particle frequencies."""

    normalized = _normalize_variables(variables, n)
    counts = {particle: 0 for particle in range(1, n + 1)}
    for variable in normalized:
        for particle in variable:
            counts[particle] += 1
    particle_frequencies = tuple((particle, counts[particle]) for particle in range(1, n + 1))
    profile = tuple(sorted(counts.values(), reverse=True))
    return DNSignature(
        n=n,
        variable_count=len(normalized),
        occurrence_count=sum(profile),
        frequency_profile=profile,
        particle_frequencies=particle_frequencies,
    )


@dataclass(frozen=True, slots=True)
class VariableCombinationRecord:
    """One G02-independent variable combination and its G04/DN classes."""

    n: int
    variables: tuple[Invariant, ...]
    independence: IndependenceResult
    canonical: VariableCombinationResult
    dn: DNSignature

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(invariant_label(variable) for variable in self.variables)

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "variables": [list(variable) for variable in self.variables],
            "labels": list(self.labels),
            "independence": self.independence.to_dict(),
            "canonical": self.canonical.to_dict(),
            "dn": self.dn.to_dict(),
        }


def classify_variable_combination(
    variables: Iterable[Iterable[int]],
    n: int,
) -> VariableCombinationRecord:
    """Return the exact independence, S_n key, and DN record for a combination."""

    normalized = _normalize_variables(variables, n)
    independence = analyze_invariants(normalized, n)
    canonical = canonicalize_variable_set(normalized, n)
    return VariableCombinationRecord(
        n=n,
        variables=normalized,
        independence=independence,
        canonical=canonical,
        dn=dn_signature(normalized, n),
    )


@dataclass(frozen=True, slots=True)
class CombinationSizeSummary:
    """Counts for one requested combination size."""

    size: int
    raw_count: int
    independent_count: int
    unique_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "size": self.size,
            "raw_count": self.raw_count,
            "independent_count": self.independent_count,
            "unique_count": self.unique_count,
        }


@dataclass(frozen=True, slots=True)
class CombinationSearchReport:
    """Complete, serializable result of the G07 combination route."""

    n: int
    sizes: tuple[int, ...]
    status: RunStatus
    candidate_count: int
    summaries: tuple[CombinationSizeSummary, ...]
    independent_records: tuple[VariableCombinationRecord, ...]
    unique_records: tuple[VariableCombinationRecord, ...]

    @property
    def complete(self) -> bool:
        return self.status is RunStatus.COMPLETE

    @property
    def raw_count(self) -> int:
        return sum(summary.raw_count for summary in self.summaries)

    @property
    def independent_count(self) -> int:
        return sum(summary.independent_count for summary in self.summaries)

    @property
    def dependent_count(self) -> int:
        return self.raw_count - self.independent_count

    @property
    def unique_count(self) -> int:
        return len(self.unique_records)

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "sizes": list(self.sizes),
            "status": self.status.value,
            "complete": self.complete,
            "candidate_count": self.candidate_count,
            "counts": {
                "raw": self.raw_count,
                "independent": self.independent_count,
                "dependent": self.dependent_count,
                "unique": self.unique_count,
            },
            "summaries": [summary.to_dict() for summary in self.summaries],
            "independent_records": [record.to_dict() for record in self.independent_records],
            "unique_records": [record.to_dict() for record in self.unique_records],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _default_sizes(n: int) -> tuple[int, ...]:
    if n == 3:
        return (2,)
    if n == 4:
        return (2, 3, 4, 5)
    raise ValueError("G07 currently supports n=3 and n=4 only")


def enumerate_invariant_combinations(
    n: int = 4,
    sizes: Iterable[int] | None = None,
) -> CombinationSearchReport:
    """Enumerate every requested candidate combination and classify it."""

    candidates = candidate_invariants(n)
    if sizes is None:
        requested = _default_sizes(n)
    else:
        raw_sizes = tuple(sizes)
        if any(isinstance(size, bool) or not isinstance(size, int) for size in raw_sizes):
            raise TypeError("sizes must contain integers")
        requested = tuple(sorted(set(raw_sizes)))
    if not requested:
        raise ValueError("sizes cannot be empty")
    if any(size < 2 or size > len(candidates) for size in requested):
        raise ValueError(f"sizes must lie in 2..{len(candidates)} for n={n}")
    all_records: list[VariableCombinationRecord] = []
    unique: dict[tuple[Any, ...], VariableCombinationRecord] = {}
    summaries: list[CombinationSizeSummary] = []
    for size in requested:
        raw_count = comb(len(candidates), size)
        size_records: list[VariableCombinationRecord] = []
        for variables in itertools.combinations(candidates, size):
            record = classify_variable_combination(variables, n)
            if record.independence.independent:
                size_records.append(record)
                unique.setdefault(record.canonical.key, record)
        all_records.extend(size_records)
        summaries.append(
            CombinationSizeSummary(
                size=size,
                raw_count=raw_count,
                independent_count=len(size_records),
                unique_count=len({record.canonical.key for record in size_records}),
            )
        )
    ordered_unique = tuple(unique[key] for key in sorted(unique, key=repr))
    return CombinationSearchReport(
        n=n,
        sizes=requested,
        status=RunStatus.COMPLETE,
        candidate_count=len(candidates),
        summaries=tuple(summaries),
        independent_records=tuple(all_records),
        unique_records=ordered_unique,
    )


@dataclass(frozen=True, slots=True)
class LabelingCombinationLink:
    """Trace one valid G06 label through topology, label, variable, and DN keys."""

    topology: CanonicalizationResult
    labeling: CanonicalizationResult
    combination: VariableCombinationRecord

    @property
    def topology_key(self) -> tuple[Any, ...]:
        return self.topology.key

    @property
    def labeling_key(self) -> tuple[Any, ...]:
        return self.labeling.key

    @property
    def variable_key(self) -> tuple[Any, ...]:
        return self.combination.canonical.key

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology.to_dict(),
            "labeling": self.labeling.to_dict(),
            "combination": self.combination.to_dict(),
        }


def link_labeling_result(topology: Topology, result: LabelingResult) -> LabelingCombinationLink:
    """Link one valid G06 result to its G04 topology and variable keys."""

    if not isinstance(topology, Topology):
        raise TypeError("link_labeling_result expects a Topology object")
    if not isinstance(result, LabelingResult):
        raise TypeError("link_labeling_result expects a LabelingResult object")
    if not result.valid or result.canonical is None or not result.variables:
        raise ValueError("only valid G06 results with a canonical labeling can be linked")
    return LabelingCombinationLink(
        topology=canonicalize_topology(topology),
        labeling=result.canonical,
        combination=classify_variable_combination(result.variables, topology.n),
    )


def link_labeling_report(
    topology: Topology,
    report: LabelingSearchReport,
) -> tuple[LabelingCombinationLink, ...]:
    """Link every valid result retained by one G06 report.

    The report status is deliberately not changed or hidden.  If G06 was
    incomplete, the returned links are only a partial graph-route view and
    callers must inspect ``report.status`` before making completeness claims.
    """

    if not isinstance(report, LabelingSearchReport):
        raise TypeError("link_labeling_report expects a LabelingSearchReport object")
    if report.topology != topology:
        raise ValueError("the report topology does not match the supplied topology")
    return tuple(link_labeling_result(topology, result) for result in report.valid_results)


@dataclass(frozen=True, slots=True)
class CoverageIndex:
    """Many-to-many graph-route coverage index for later G08 reporting."""

    links: tuple[LabelingCombinationLink, ...]
    topology_to_variables: tuple[tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...]], ...]
    variable_to_topologies: tuple[tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...]], ...]
    dn_to_variables: tuple[tuple[str, tuple[tuple[Any, ...], ...]], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "links": [link.to_dict() for link in self.links],
            "topology_to_variables": [
                {"topology_key": _jsonable(topology_key), "variable_keys": _jsonable(variable_keys)}
                for topology_key, variable_keys in self.topology_to_variables
            ],
            "variable_to_topologies": [
                {"variable_key": _jsonable(variable_key), "topology_keys": _jsonable(topology_keys)}
                for variable_key, topology_keys in self.variable_to_topologies
            ],
            "dn_to_variables": [
                {"dn": dn, "variable_keys": _jsonable(variable_keys)}
                for dn, variable_keys in self.dn_to_variables
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_coverage_index(links: Iterable[LabelingCombinationLink]) -> CoverageIndex:
    """Build deterministic many-to-many relations without collapsing DN classes."""

    link_tuple = tuple(links)
    topology_map: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    variable_map: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    dn_map: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for link in link_tuple:
        topology_map[link.topology_key].add(link.variable_key)
        variable_map[link.variable_key].add(link.topology_key)
        dn_map[link.combination.dn.label].add(link.variable_key)
    topology_relations = tuple(
        (key, tuple(sorted(values, key=repr)))
        for key, values in sorted(topology_map.items(), key=lambda item: repr(item[0]))
    )
    variable_relations = tuple(
        (key, tuple(sorted(values, key=repr)))
        for key, values in sorted(variable_map.items(), key=lambda item: repr(item[0]))
    )
    dn_relations = tuple(
        (key, tuple(sorted(values, key=repr)))
        for key, values in sorted(dn_map.items())
    )
    return CoverageIndex(
        links=link_tuple,
        topology_to_variables=topology_relations,
        variable_to_topologies=variable_relations,
        dn_to_variables=dn_relations,
    )


__all__ = [
    "CombinationSearchReport",
    "CombinationSizeSummary",
    "CoverageIndex",
    "DNSignature",
    "LabelingCombinationLink",
    "VariableCombinationRecord",
    "build_coverage_index",
    "classify_variable_combination",
    "dn_signature",
    "enumerate_invariant_combinations",
    "link_labeling_report",
    "link_labeling_result",
]
