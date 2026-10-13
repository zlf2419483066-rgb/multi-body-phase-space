"""Profile-independent invariant-occurrence extraction for five-body sources.

The extractor is a structural adapter.  It preserves every double-line
occurrence from a reviewed fixed-label source and exposes a separate view of
unique particle subsets.  It does not infer independence, remove duplicate
occurrences, or move a particle-relabeling source into a fixed-label profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..label_identity import canonicalize_candidate_labeling
from ..physical_labels import review_candidate_record
from .sources import (
    FIXED_MODE,
    SourceError,
    UnsupportedSourceError,
    VerifiedSource,
    _jsonable,
    _key_token,
    _source_digest,
)


INVARIANT_SET_SCHEMA_VERSION = "F5-I-invariant-set-v1"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class InvariantSetOccurrence:
    """One ordered double-line invariant occurrence from the source record."""

    occurrence_index: int
    edge_index: int
    edge_occurrence: int
    particle_subset: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.occurrence_index, bool) or not isinstance(self.occurrence_index, int):
            raise TypeError("occurrence_index must be an integer")
        if self.occurrence_index < 0:
            raise ValueError("occurrence_index must be non-negative")
        for value, name in ((self.edge_index, "edge_index"), (self.edge_occurrence, "edge_occurrence")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        subset = tuple(self.particle_subset)
        if len(subset) not in {2, 3, 4}:
            raise ValueError("invariant particle subsets must have cardinality 2, 3, or 4")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in subset):
            raise TypeError("invariant particle IDs must be integers")
        if tuple(sorted(subset)) != subset or len(set(subset)) != len(subset):
            raise ValueError("invariant particle subsets must be sorted and unique")
        if subset[0] < 1 or subset[-1] > 5:
            raise ValueError("invariant particle IDs must lie in 1..5")
        object.__setattr__(self, "particle_subset", subset)

    @property
    def occurrence(self) -> int:
        """Compatibility alias for the source edge occurrence number."""

        return self.edge_occurrence

    @property
    def edge_ref(self) -> tuple[int, int]:
        return self.edge_index, self.edge_occurrence

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence_index": self.occurrence_index,
            "edge_index": self.edge_index,
            "edge_occurrence": self.edge_occurrence,
            "particle_subset": list(self.particle_subset),
        }


@dataclass(frozen=True, slots=True)
class InvariantSet:
    """Immutable structural invariant occurrence data for one source identity."""

    source_digest: str
    identity_digest: str
    mode: str
    topology_key: tuple[Any, ...]
    identity_key: tuple[Any, ...]
    occurrences: tuple[InvariantSetOccurrence, ...]
    unique_subsets: tuple[tuple[int, ...], ...]
    duplicate_groups: tuple[tuple[int, ...], ...]
    schema_version: str = INVARIANT_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INVARIANT_SET_SCHEMA_VERSION:
            raise ValueError("unsupported invariant set schema")
        if not isinstance(self.source_digest, str) or not self.source_digest:
            raise ValueError("source_digest must be non-empty text")
        if not isinstance(self.identity_digest, str) or not self.identity_digest:
            raise ValueError("identity_digest must be non-empty text")
        if self.mode != FIXED_MODE:
            raise UnsupportedSourceError("only fixed_labels invariant sets are supported")
        occurrences = tuple(self.occurrences)
        if any(not isinstance(item, InvariantSetOccurrence) for item in occurrences):
            raise TypeError("occurrences must contain InvariantSetOccurrence values")
        if tuple(item.occurrence_index for item in occurrences) != tuple(range(len(occurrences))):
            raise ValueError("occurrence indexes must be consecutive and source ordered")
        occurrence_subsets = tuple(item.particle_subset for item in occurrences)
        expected_unique: list[tuple[int, ...]] = []
        expected_groups: dict[str, list[int]] = {}
        for item in occurrences:
            token = _key_token(item.particle_subset)
            if token not in expected_groups:
                expected_groups[token] = []
                expected_unique.append(item.particle_subset)
            expected_groups[token].append(item.occurrence_index)
        unique_subsets = tuple(tuple(item) for item in self.unique_subsets)
        if any(
            any(type(value) is not int for value in subset)
            for subset in unique_subsets
        ):
            raise TypeError("unique_subsets must contain integer particle IDs")
        if unique_subsets != tuple(expected_unique):
            raise ValueError("unique_subsets must exactly follow first occurrence order")
        groups = tuple(tuple(group) for group in self.duplicate_groups)
        seen_indexes: set[int] = set()
        for group in groups:
            if any(type(index) is not int for index in group):
                raise TypeError("duplicate group indexes must be integers")
            if len(group) < 2 or tuple(sorted(group)) != group:
                raise ValueError("duplicate groups must contain sorted repeated indexes")
            if any(index < 0 or index >= len(occurrences) for index in group):
                raise ValueError("duplicate group index is outside occurrences")
            if seen_indexes.intersection(group):
                raise ValueError("duplicate groups may not overlap")
            seen_indexes.update(group)
            if len({occurrences[index].particle_subset for index in group}) != 1:
                raise ValueError("duplicate group occurrences must share one particle subset")
        expected_duplicates = tuple(
            tuple(expected_groups[_key_token(item)])
            for item in expected_unique
            if len(expected_groups[_key_token(item)]) > 1
        )
        if groups != expected_duplicates:
            raise ValueError("duplicate_groups must exactly follow repeated occurrence indexes")
        object.__setattr__(self, "topology_key", tuple(_freeze(item) for item in self.topology_key))
        object.__setattr__(self, "identity_key", tuple(_freeze(item) for item in self.identity_key))
        object.__setattr__(self, "occurrences", occurrences)
        object.__setattr__(self, "unique_subsets", unique_subsets)
        object.__setattr__(self, "duplicate_groups", groups)

    @property
    def full_topology_key(self) -> tuple[Any, ...]:
        return self.topology_key

    @property
    def full_identity_key(self) -> tuple[Any, ...]:
        return self.identity_key

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible view of the immutable result."""

        return {
            "schema_version": self.schema_version,
            "source_digest": self.source_digest,
            "identity_digest": self.identity_digest,
            "mode": self.mode,
            "topology_key": _jsonable(self.topology_key),
            "identity_key": _jsonable(self.identity_key),
            "occurrences": [item.to_dict() for item in self.occurrences],
            "unique_subsets": _jsonable(self.unique_subsets),
            "duplicate_groups": _jsonable(self.duplicate_groups),
        }


def _checked_source(source: VerifiedSource) -> None:
    if not isinstance(source, VerifiedSource):
        raise TypeError("source must be a VerifiedSource")
    if source.mode != FIXED_MODE:
        raise UnsupportedSourceError(
            "particle_relabeling invariant extraction is unsupported"
        )
    review_candidate_record(source.source_record)
    rebuilt = canonicalize_candidate_labeling(source.source_record, FIXED_MODE)
    if rebuilt.to_dict() != source.identity.to_dict():
        raise SourceError("source identity no longer matches rederived fixed-label identity")


def extract_invariant_set(source: VerifiedSource) -> InvariantSet:
    """Extract all ordered double-line occurrences from one reviewed source."""

    _checked_source(source)
    labels = source.source_record.complete_labeling.double_lines
    occurrences = tuple(
        InvariantSetOccurrence(
            occurrence_index=index,
            edge_index=label.edge_index,
            edge_occurrence=label.occurrence,
            particle_subset=tuple(sorted(label.particle_ids)),
        )
        for index, label in enumerate(labels)
    )
    unique: list[tuple[int, ...]] = []
    indexes: dict[str, list[int]] = {}
    for item in occurrences:
        token = _key_token(item.particle_subset)
        if token not in indexes:
            indexes[token] = []
            unique.append(item.particle_subset)
        indexes[token].append(item.occurrence_index)
    duplicate_groups = tuple(
        tuple(indexes[token]) for token in (_key_token(item) for item in unique)
        if len(indexes[token]) > 1
    )
    return InvariantSet(
        source_digest=_source_digest(source.source_record),
        identity_digest=source.digest,
        mode=source.mode,
        topology_key=source.topology_key,
        identity_key=source.key,
        occurrences=occurrences,
        unique_subsets=tuple(unique),
        duplicate_groups=duplicate_groups,
    )


__all__ = [
    "INVARIANT_SET_SCHEMA_VERSION", "InvariantSetOccurrence", "InvariantSet",
    "extract_invariant_set",
]
