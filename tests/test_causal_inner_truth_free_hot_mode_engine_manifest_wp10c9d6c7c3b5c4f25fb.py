from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_truth_free_hot_mode_engine_manifest_wp10c9d6c7c3b5c4f25fb as target


def test_contract_is_truth_free_and_retains_macro_ledger() -> None:
    locked = target._validate_parent(require_clean=False)
    contract = target._contract(locked)
    assert contract["engine"]["state"].startswith("q82_plus")
    assert contract["engine"]["macro_ledger"].startswith("all_82")
    assert all(value == 0 for value in contract["online_forbidden"].values())
    assert contract["binding_replay"]["oversize_step_must_reject_without_propagation"]
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
