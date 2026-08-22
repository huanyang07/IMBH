from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_arclength_transport_manifest_wp10c9d6c7c3b5c4f25f2 as target


def test_parent_selects_arclength_and_supersedes_fixed_window_six() -> None:
    lock = target._validate_parent(require_clean=False)
    assert set(lock) == {"diagnosis_hashes", "window_05_hashes"}


def test_contract_freezes_phase_transport_and_no_truth_preflight() -> None:
    contract = target._contract(target._validate_parent(require_clean=False))
    assert contract["mathematical_system"]["coordinate_equation"] == "dy/ds=f_Q(y,t)/nu(y,t)"
    assert contract["collocation"]["node_count"] == 5
    assert contract["moving_exact_chart_transport"]["one_exact_augmented_coordinate_Jacobian_at_anchor"]
    assert contract["preflight"]["new_exact_fixed_Q_rate_calls_equal"] == 0


def test_canonical_manifest_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["transport_preflight_authorized"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
