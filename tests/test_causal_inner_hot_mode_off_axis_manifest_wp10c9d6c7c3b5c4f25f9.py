from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_hot_mode_off_axis_manifest_wp10c9d6c7c3b5c4f25f9 as target


def test_contract_separates_hot_mode_and_forbids_fixed_q_dynamics() -> None:
    locked = target._validate_parent(require_clean=False)
    contract = target._contract(locked)
    assert contract["architecture"]["discrete_mode"] == "hot distinct from cold"
    assert contract["truth_budget"]["new_exact_free_rate_calls"] == 3
    assert contract["truth_budget"]["new_fixed_Q_reaction_calls"] == 0
    assert contract["truth_budget"]["new_nonlinear_roots"] == 0
    assert not contract["complete_cycle_execution_authorized"]


def test_canonical_package_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
