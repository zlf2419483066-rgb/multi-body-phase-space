"""Recheck formula cards through LM-02/LM-12 without changing their rules."""

from dataclasses import replace
from ..domain import CountMode, Labeling, Topology
from ..formula_inputs import FormulaInputCard, build_formula_input
from ..label_identity import MarkedIdentity, canonicalize_physical_labeling
from ..physical_labels import NormalizationResult, PhysicalLabelRecord, normalize_physical_labeling


def validate_formula_card(card: FormulaInputCard) -> FormulaInputCard:
    if not isinstance(card, FormulaInputCard):
        raise TypeError("expected FormulaInputCard")
    if card.availability != "ready":
        raise ValueError(card.unavailable_reason or "formula input is unavailable")
    if card.n != 4 or not isinstance(card.record, dict):
        raise ValueError("ready formula card requires a complete four-body record")
    stored = card.record
    checked = normalize_physical_labeling(
        Topology.from_dict(stored["topology"]),
        Labeling.from_dict(stored["complete_labeling"]),
    )
    if checked.stage != "accepted" or checked.record is None or checked.record.to_dict() != stored:
        raise ValueError("formula card record disagrees with recomputed physical evidence")
    rebuilt = build_formula_input(checked, measure_convention=card.measure_convention, provenance=card.provenance)
    # Identity maps may refer to the original graph, so compare their physical
    # key and canonical record, not a newly computed identity-to-itself map.
    if card.identity is not None:
        parsed = canonicalize_physical_labeling(checked.record, card.identity["mode"]).to_dict()
        for key in ("schema_version", "mode", "topology_key", "key", "digest", "canonical_record"):
            if card.identity.get(key) != parsed[key]:
                raise ValueError(f"formula identity mismatch: {key}")
        if card.mode != parsed["mode"]:
            raise ValueError("formula identity mode mismatch")
    elif card.mode is not None:
        raise ValueError("formula mode without an identity")
    if replace(card, identity=None, mode=None) != rebuilt:
        raise ValueError("formula card factors/variables/status disagree with LM-12 reconstruction")
    return card


def prepare_formula_input(
    record: PhysicalLabelRecord | NormalizationResult | FormulaInputCard,
    *,
    measure_convention: str,
    identity: MarkedIdentity | None = None,
) -> FormulaInputCard:
    if isinstance(record, FormulaInputCard):
        if record.measure_convention != measure_convention:
            raise ValueError("normalization conflicts with formula input card")
        if identity is not None:
            if record.record is None:
                raise ValueError("an identity cannot supply a missing physical record")
            if record.availability == "ready":
                validate_formula_card(record)
            checked = normalize_physical_labeling(
                Topology.from_dict(record.record["topology"]),
                Labeling.from_dict(record.record["complete_labeling"]),
            )
            return build_formula_input(checked, measure_convention=measure_convention,
                                       identity=identity, provenance=record.provenance)
        card = record
    else:
        card = build_formula_input(record, measure_convention=measure_convention, identity=identity)
    if card.availability != "ready":
        return card
    validate_formula_card(card)
    if card.identity is None:
        checked = normalize_physical_labeling(
            Topology.from_dict(card.record["topology"]),
            Labeling.from_dict(card.record["complete_labeling"]),
        )
        marked = canonicalize_physical_labeling(checked.record, CountMode.FIXED_LABELS)
        card = build_formula_input(checked, measure_convention=measure_convention,
                                   identity=marked, provenance=card.provenance)
    return card
