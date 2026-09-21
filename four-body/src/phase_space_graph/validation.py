"""External acceptance checks; never imported by graph generation or pruning.

Source counts are expectations, not construction rules. Catalog tracing reads
saved audit evidence and evaluates supplied source structures independently.
"""

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping

from .canonical import canonicalize_topology
from .domain import Labeling, Topology
from .labeling import evaluate_labeling
from .structure import analyze_topology


def compare_reference_count(actual, expected, run_status, *, scope_matches=True):
    """A numerical match never asserts a graph-by-graph catalog match."""
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        raise ValueError('expected count must be a non-negative integer')
    if actual is not None and (isinstance(actual, bool) or not isinstance(actual, int) or actual < 0):
        raise ValueError('actual count must be a non-negative integer or None')
    if not scope_matches:
        status = 'out_of_scope'
    elif run_status != 'complete' or actual is None:
        status = 'inconclusive'
    else:
        status = 'count_match' if actual == expected else 'count_mismatch'
    return {'status': status, 'expected': expected, 'actual': actual,
            'difference': actual - expected if isinstance(actual, int) else None}


@dataclass(frozen=True, slots=True)
class SourceDiagram:
    source: str
    page: int
    figure: str
    topology: Topology | None = None
    transcription_verified: bool = False
    labeling: Labeling | None = None

    def __post_init__(self):
        if not self.source or not self.figure:
            raise ValueError('source and figure are required')
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1:
            raise ValueError('page must be a one-based page number')
        if not isinstance(self.transcription_verified, bool):
            raise TypeError('transcription_verified must be boolean')

    @classmethod
    def from_dict(cls, value):
        return cls(value['source'], value['page'], str(value['figure']),
                   Topology.from_dict(value['topology']) if value.get('topology') is not None else None,
                   value.get('transcription_verified', False),
                   Labeling.from_dict(value['labeling']) if value.get('labeling') is not None else None)


def _key(topology):
    return json.dumps(canonicalize_topology(topology).key, separators=(',', ':'))


def trace_source_catalog(audit: Mapping[str, Any], entries: Iterable[SourceDiagram],
                         *, expected_count: int) -> dict[str, Any]:
    """Locate each source graph in saved G05/G06 evidence without enumerating.

    Unverified transcriptions may be investigated, but cannot establish catalog
    acceptance. Source IDs sharing a key are reported, never assumed to be errors
    or deleted. Supplied particle labels are checked on the SOURCE topology.
    """
    entries = tuple(entries)
    identities = [(e.source, e.page, e.figure) for e in entries]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate source/page/figure identity')
    g05 = audit['topology_report']
    generated = {_key(Topology.from_dict(t)) for t in g05['topologies']}
    labels = {}
    for report in audit.get('labeling_reports', ()):
        key = _key(Topology.from_dict(report['topology']))
        if key in labels:
            raise ValueError('duplicate G06 reports for one canonical topology')
        labels[key] = report
    effective = {key for key, r in labels.items() if r.get('valid_results') and key in generated}
    rows = []
    groups = {}
    accepted = set()
    for entry in entries:
        row = {'source': entry.source, 'page': entry.page, 'figure': entry.figure,
               'transcription_verified': entry.transcription_verified}
        rows.append(row)
        if entry.topology is None:
            row['stage'] = 'not_transcribed'
            continue
        structure = analyze_topology(entry.topology)
        row['structure'] = structure.to_dict()
        if not structure.field_report.valid:
            row['stage'] = 'field_rejected'
            continue
        if not structure.valid:
            row['stage'] = 'structure_rejected'
            continue
        if entry.topology.n != g05['n'] or structure.metrics.l != g05['loop_count']:
            row['stage'] = 'scope_mismatch'
            continue
        key = _key(entry.topology)
        row['canonical_key'] = json.loads(key)
        groups.setdefault(key, []).append({'source': entry.source, 'page': entry.page, 'figure': entry.figure})
        if entry.labeling is not None:
            row['source_label_evaluation'] = evaluate_labeling(entry.topology, entry.labeling).to_dict()
        if key not in generated:
            row['stage'] = ('missing_from_complete_search' if g05['status'] == 'complete'
                            else 'unknown_in_partial_search')
            continue
        report = labels.get(key)
        if report is None:
            row['stage'] = 'label_search_missing'
            continue
        row['label_counts'] = report['counts']
        row['failure_counts'] = report.get('failure_counts', {})
        row['failure_samples'] = report.get('failure_samples', [])
        row['failure_witnesses'] = report.get('failure_witnesses', {})
        row['evidence_note'] = ('Per-reason certificates recorded' if row['failure_witnesses']
                                else 'Legacy audit may lack per-reason certificates; absence is not proof of no failure')
        if report.get('valid_results'):
            row['witness'] = report['valid_results'][0]
            row['witness_topology'] = report['topology']
        if report['status'] != 'complete':
            row['stage'] = 'label_search_' + report['status']
        elif report.get('valid_results'):
            row['stage'] = 'accepted'
            accepted.add(key)
        elif report['counts']['momentum_valid'] > 0:
            row['stage'] = 'independence_rejected'
        else:
            row['stage'] = 'momentum_rejected'
    evidence_complete = (audit['status'] == 'complete' and g05['status'] == 'complete'
                         and set(labels) == generated and all(r['status'] == 'complete' for r in labels.values()))
    count = compare_reference_count(audit['counts'].get('effective_topologies'), expected_count,
                                    'complete' if evidence_complete else 'incomplete')
    full_transcription = (len(entries) == expected_count
                          and all(e.topology is not None and e.transcription_verified for e in entries))
    unresolved = any(row['stage'] != 'accepted' or
                     row.get('source_label_evaluation', {}).get('valid') is False for row in rows)
    duplicate_groups = [value for value in groups.values() if len(value) > 1]
    extra = sorted(effective - accepted)
    if not full_transcription or not evidence_complete:
        status = 'inconclusive'
    elif unresolved or duplicate_groups or extra or count['status'] != 'count_match':
        status = 'catalog_mismatch'
    else:
        status = 'catalog_match'
    return {'status': status, 'count_comparison': count,
            'complete_verified_transcription': full_transcription, 'rows': rows,
            'same_topology_source_groups': duplicate_groups,
            'unmatched_effective_keys': [json.loads(key) for key in extra],
            'note': 'Equal counts alone do not prove catalog coverage; shared keys require review.'}


def compare_combination_coverage(coverage, combinations, *, n, size, graph_complete):
    """Compare actual canonical sets at the required invariant count."""
    if combinations is None or combinations.n != n or size not in combinations.sizes:
        return {'status': 'inconclusive', 'reason': 'missing_required_combination_scope',
                'n': n, 'size': size, 'graph_only': [], 'combination_only': []}
    universe = {r.canonical.key: r for r in combinations.unique_records if len(r.variables) == size}
    actual = {link.variable_key for link in coverage.links}
    graph_only = actual - set(universe)
    combination_only = set(universe) - actual
    status = ('inconclusive' if not graph_complete or not combinations.complete
              else 'inconsistent' if graph_only else 'checked')
    return {'status': status, 'n': n, 'size': size,
            'graph_only': [list(key) for key in sorted(graph_only, key=repr)],
            'combination_only': [universe[key].to_dict() for key in sorted(combination_only, key=repr)],
            'note': 'Independent combinations without graphs are a difference set, not automatically missing graphs.'}
