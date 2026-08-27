import json

import pytest

import run_causal_inner_production_size_global_ap_dry_run_wp10c9d6c7c3b5c4f25fizzo1 as runner


def test_parent_global_contract_is_hash_locked_and_production_sized():
    hashes, contract = runner._validate_parent()
    assert "summary.json" in hashes
    assert contract["global_AP_dry_run"]["global_state_dimension"] == 1034
    assert contract["global_AP_dry_run"]["gates"]["online_truth_calls"] == 0


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="dry run not executed")
def test_canonical_global_dry_run_closes_without_boundary_or_cycle_claim():
    hashes = runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    metrics = json.loads((runner.CANONICAL_DIRECTORY / "global_dry_run_metrics.json").read_text())
    assert hashes and summary["passed"] and summary["production_size_global_AP_dry_run_certified"]
    assert metrics["online_truth_calls"] == 0
    assert metrics["all_checkpoints_bitwise"] and metrics["all_suffix_replays_bitwise"]
    assert summary["periodic_boundary_only"] and not summary["physical_boundary_ports_certified"]
    assert summary["complete_cycle_steps"] == 0
