from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_hot_mode_off_axis_manifest_wp10c9d6c7c3b5c4f25f9_v2 as target


def test_superseding_contract_changes_only_execution_mechanics() -> None:
    locked = target._validate_parent(require_clean=False)
    contract = target._contract(locked)
    assert contract["supersedes"] == target.parent.WORK_PACKAGE
    assert not contract["scientific_contract_changed"]
    assert contract["execution_repair"]["per_witness_exact_scratch_required"]
    assert contract["truth_budget"]["new_exact_free_rate_calls"] == 3
    assert contract["gates"] == target.parent._contract(
        locked["underlying_parent"]
    )["gates"]


def test_canonical_package_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    assert summary["passed"]
    assert not summary["scientific_contract_changed"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
