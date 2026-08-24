from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_conservative_metric_chart_atlas_manifest_wp10c9d6c7c3b5c4f25fic as target  # noqa: E402


def test_diagnosis_authorizes_metric_atlas() -> None:
    lock = target._validate_diagnosis(require_clean=False)
    assert lock["summary"]["classification"] == target.diagnosis.METRIC_CLASSIFICATION
    assert lock["summary"]["atlas_supported"]


def test_contract_preserves_original_physics() -> None:
    contract = target._contract()
    chart = contract["mathematical_chart"]
    assert "remains binding for physics" in chart["original_coordinate"]
    assert "original primitive/original q space" in chart["conservation"]
    assert contract["scope"]["new_exact_free_field_calls"] == 0
    assert contract["scope"]["new_trajectory_segments"] == 0
    assert contract["anchors"]["historical_raw_hard_condition_preserved"] == 2.5e3


def test_overlap_budget_and_gates_are_frozen() -> None:
    contract = target._contract()
    assert contract["anchors"]["accepted_anchor_attempt"] == 82
    assert contract["anchors"]["overlap_candidate_attempt"] == 83
    assert contract["scope"]["new_exact_coordinate_jacobians"] == 2
    assert contract["scope"]["new_retractions"] == 3
    assert contract["gates"]["maximum_metric_jacobian_condition"] == 10.0
    assert contract["gates"]["forward_replay_bitwise"]


def test_canonical_manifest_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(target.CANONICAL_DIRECTORY / "atlas_contract.json")
    assert summary["classification"] == target.CLASSIFICATION
    assert summary["definitions_only"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
    assert contract["authorized_execution"] == target.AUTHORIZED_NEXT
