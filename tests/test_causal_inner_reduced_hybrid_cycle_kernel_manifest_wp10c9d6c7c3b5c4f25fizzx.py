import json

import pytest

import run_causal_inner_reduced_hybrid_cycle_kernel_manifest_wp10c9d6c7c3b5c4f25fizzx as runner


def test_parent_sequence_validator_is_hash_locked():
    assert "summary.json" in runner._validate_parent()


def test_contract_requires_real_physics_before_production():
    contract = runner._contract()
    gates = contract["production_fail_closed"]
    assert gates["require_physical_model_complete"]
    assert gates["reject_synthetic_fixture"]
    assert gates["require_heldout_physical_validation_complete"]
    assert not gates["cycle_runner_available_in_certificate"]
    assert contract["prefix_and_cost_certificate"]["complete_cycle_steps"] == 0


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_cycle_kernel_manifest_is_definitions_only():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert not summary["reduced_hybrid_cycle_kernel_certified"]
    assert not summary["production_adapter_certified"]
    assert not summary["physical_payloads_acquired"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
