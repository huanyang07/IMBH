import json

import pytest

import run_causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzy as runner


def test_parent_kernel_certificate_is_hash_locked():
    assert "summary.json" in runner._validate_parent()[0]


def test_final_contract_selects_reduced_hybrid_architecture_and_stops():
    _, metrics = runner._validate_parent()
    contract = runner._contract(metrics)
    architecture = contract["selected_mathematical_architecture"]
    assert architecture["online_state"].startswith("y=(Q1,Q2,Q3,Q4")
    assert architecture["online_prohibitions"]["truth_residual_calls"] == 0
    assert contract["single_cycle_execution_contract"]["maximum_wall_days"] == 3.0
    assert "not required to return" in contract["single_cycle_execution_contract"]["important_non_gate"]
    assert not contract["current_readiness"]["complete_cycle_execution_ready"]
    assert not contract["decision"]["complete_cycle_execution_authorized"]
    assert contract["decision"]["complete_cycle_steps"] == 0


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_final_preexecution_manifest_is_blocked_on_external_physics():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert summary["mathematical_architecture_verified"]
    assert not summary["physical_payloads_acquired"]
    assert not summary["complete_cycle_execution_ready"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
