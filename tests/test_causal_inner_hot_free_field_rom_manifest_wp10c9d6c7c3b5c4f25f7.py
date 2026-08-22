from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_hot_free_field_rom_manifest_wp10c9d6c7c3b5c4f25f7 as target


def test_parent_selects_free_field_and_rejects_fixed_q_clock() -> None:
    target._validate_parent(require_clean=False)
    contract = target._contract(target._validate_parent(require_clean=False))
    assert contract["evaluation"]["fixed_Q_reaction_calls"] == 0
    assert contract["evaluation"]["node_count"] == 5
    assert not contract["fixed_Q_arclength_physical_time_authorized"]


def test_holdout_and_cost_gates_are_prospective() -> None:
    contract = target._contract(target._validate_parent(require_clean=False))
    assert contract["evaluation"]["training_indices"] == [0, 2, 4]
    assert contract["evaluation"]["holdout_indices"] == [1, 3]
    assert contract["gates"]["maximum_hidden_rate_holdout_defect"] == 0.05
    assert contract["gates"]["maximum_projected_256_witness_wall_hours"] == 24.0


def test_canonical_package_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    assert summary["passed"] and summary["definitions_only"]
    assert summary["hot_free_field_rom_preflight_authorized"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
