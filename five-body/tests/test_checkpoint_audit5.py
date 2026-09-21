"""Audit consistency checks for committed five-body checkpoints."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phase_space_graph5.catalog_checkpoint import (  # noqa: E402
    CatalogCheckpointWriter,
    CheckpointCorruptionError,
    read_checkpoint,
)
from phase_space_graph5.canonical import canonicalize_topology  # noqa: E402
from phase_space_graph5.domain import CountMode  # noqa: E402
from phase_space_graph5.fixtures import fixture_a  # noqa: E402
from phase_space_graph5.label_identity import canonicalize_candidate_labeling  # noqa: E402
from phase_space_graph5.physical_labels import normalize_candidate_labeling  # noqa: E402


def _identity_pair():
    topology, labeling = fixture_a()
    checked = normalize_candidate_labeling(topology, labeling)
    if checked.record is None:
        raise AssertionError(checked.to_dict())
    source = checked.record
    identities = {
        mode.value: canonicalize_candidate_labeling(source, mode)
        for mode in CountMode
    }
    return topology, source, identities


def _append(
    root: Path,
    *,
    identities: dict[str, tuple[object, ...]],
    source: object,
    label_complete: dict[str, bool],
    identity_complete: dict[str, bool],
    topology_complete: bool,
    audit: dict[str, object],
) -> None:
    topology, _, _ = _identity_pair()
    key = canonicalize_topology(topology).key
    writer = CatalogCheckpointWriter(root, scope={"n": 5}, expected_topology_keys=(key,))
    writer.append_topology(
        key,
        identities_by_mode=identities,
        source_records_by_mode={
            mode: {
                identity.digest: source
                for identity in values
            }
            for mode, values in identities.items()
        },
        label_enumeration_complete=label_complete,
        identity_save_complete=identity_complete,
        topology_complete=topology_complete,
        audit=audit,
    )


class CheckpointAudit5Tests(unittest.TestCase):
    def test_declared_identity_counts_must_match_each_mode(self) -> None:
        topology, source, identities = _identity_pair()
        del topology
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            _append(
                root,
                identities={mode: (identity,) for mode, identity in identities.items()},
                source=source,
                label_complete={mode: True for mode in identities},
                identity_complete={mode: True for mode in identities},
                topology_complete=True,
                audit={
                    "identity_counts": {"fixed_labels": 1, "particle_relabeling": 0},
                },
            )
            with self.assertRaises(CheckpointCorruptionError):
                read_checkpoint(root)

    def test_complete_positive_block_cannot_hide_a_mode(self) -> None:
        _, source, identities = _identity_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            _append(
                root,
                identities={"fixed_labels": (identities["fixed_labels"],), "particle_relabeling": ()},
                source=source,
                label_complete={"fixed_labels": True, "particle_relabeling": True},
                identity_complete={"fixed_labels": True, "particle_relabeling": True},
                topology_complete=True,
                audit={
                    "momentum_pass_count": 1,
                    "identity_counts": {"fixed_labels": 1, "particle_relabeling": 0},
                },
            )
            with self.assertRaises(CheckpointCorruptionError):
                read_checkpoint(root)

    def test_complete_zero_p_block_cannot_contain_identities(self) -> None:
        _, source, identities = _identity_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            _append(
                root,
                identities={mode: (identity,) for mode, identity in identities.items()},
                source=source,
                label_complete={mode: True for mode in identities},
                identity_complete={mode: True for mode in identities},
                topology_complete=True,
                audit={
                    "momentum_pass_count": 0,
                    "identity_counts": {"fixed_labels": 1, "particle_relabeling": 1},
                },
            )
            with self.assertRaises(CheckpointCorruptionError):
                read_checkpoint(root)

    def test_partial_mode_may_still_be_empty(self) -> None:
        _, source, identities = _identity_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            _append(
                root,
                identities={"fixed_labels": (identities["fixed_labels"],), "particle_relabeling": ()},
                source=source,
                label_complete={"fixed_labels": True, "particle_relabeling": False},
                identity_complete={"fixed_labels": True, "particle_relabeling": False},
                topology_complete=False,
                audit={
                    "momentum_pass_count": 1,
                    "identity_counts": {"fixed_labels": 1, "particle_relabeling": 0},
                    "stop_reason": "deadline_exceeded",
                },
            )
            checkpoint = read_checkpoint(root)
            self.assertFalse(checkpoint.complete)
            self.assertFalse(checkpoint.records[0].complete)

    def test_zero_p_complete_block_with_empty_modes_is_valid(self) -> None:
        _, source, _ = _identity_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            _append(
                root,
                identities={"fixed_labels": (), "particle_relabeling": ()},
                source=source,
                label_complete={"fixed_labels": True, "particle_relabeling": True},
                identity_complete={"fixed_labels": True, "particle_relabeling": True},
                topology_complete=True,
                audit={
                    "momentum_pass_count": 0,
                    "identity_counts": {"fixed_labels": 0, "particle_relabeling": 0},
                },
            )
            checkpoint = read_checkpoint(root)
            self.assertTrue(checkpoint.complete)
            self.assertEqual(
                {mode: len(values) for mode, values in checkpoint.records[0].identities_by_mode.items()},
                {"fixed_labels": 0, "particle_relabeling": 0},
            )

    def test_legacy_complete_audit_without_optional_counts_remains_readable(self) -> None:
        _, source, identities = _identity_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            _append(
                root,
                identities={mode: (identity,) for mode, identity in identities.items()},
                source=source,
                label_complete={mode: True for mode in identities},
                identity_complete={mode: True for mode in identities},
                topology_complete=True,
                audit={},
            )
            self.assertTrue(read_checkpoint(root).complete)


if __name__ == "__main__":
    unittest.main()
