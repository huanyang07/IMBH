import json

import pytest

import run_causal_inner_cycle_wide_inputs_and_global_ap_dry_run_manifest_wp10c9d6c7c3b5c4f25fizzo as runner


def test_decomposition_starts_with_truth_free_production_size_dry_run():
    contract = runner._contract()
    stages = contract["decomposition"]
    dry_run = contract["global_AP_dry_run"]
    assert stages[0]["name"] == "production_size_global_AP_dry_run"
    assert not stages[0]["requires_new_truth"]
    assert dry_run["global_state_dimension"] == 1034
    assert dry_run["gates"]["online_truth_calls"] == 0
    assert contract["legacy_evidence_frozen_observations"]["old_physical_time_advanced_seconds"] == 0.016
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


def test_parent_preexecution_blocker_is_hash_locked():
    hashes = runner._validate_parent()
    assert "summary.json" in hashes


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_decomposition_closes():
    hashes = runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    contract = json.loads((runner.CANONICAL_DIRECTORY / "decomposition_contract.json").read_text())
    assert hashes and summary["passed"] and summary["definitions_only"]
    assert summary["authorized_next"] == runner.AUTHORIZED_NEXT
    assert summary["complete_cycle_steps"] == 0
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]
