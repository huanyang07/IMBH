import json

import pytest

import run_causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzn as runner


def test_preexecution_contract_selects_ap_architecture_and_fails_closed():
    _, metrics = runner._validate_parent()
    contract = runner._contract(metrics)
    architecture = contract["selected_mathematical_architecture"]
    readiness = contract["preexecution_readiness"]
    assert "exponential" in architecture["macrostep"]
    assert architecture["online_prohibitions"]["truth_residual_calls"] == 0
    assert contract["cycle_execution_budget"]["minimum_average_macrostep_seconds"] == 5.7888
    assert readiness["architecture_complete"]
    assert not readiness["inputs_complete"]
    assert not readiness["complete_cycle_execution_ready"]
    assert contract["decision"]["complete_cycle_runner_must_not_exist_yet"]
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


def test_all_missing_cycle_inputs_are_binding():
    _, metrics = runner._validate_parent()
    missing = runner._contract(metrics)["preexecution_readiness"]["missing_binding_inputs"]
    assert missing
    assert all(missing.values())
    assert "hot_exit_guard_and_conservative_reset" in missing
    assert "global_spatial_exponential_action_benchmark" in missing


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_preexecution_manifest_closes_without_cycle_authorization():
    hashes = runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    contract = json.loads((runner.CANONICAL_DIRECTORY / "preexecution_contract.json").read_text())
    assert hashes
    assert summary["passed"] and summary["mathematical_architecture_selected"]
    assert not summary["complete_cycle_execution_ready"]
    assert summary["complete_cycle_steps"] == 0
    assert summary["authorized_next"] is None
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]
