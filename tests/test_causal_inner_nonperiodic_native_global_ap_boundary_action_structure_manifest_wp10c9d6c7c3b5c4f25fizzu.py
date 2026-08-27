import json

import pytest

import run_causal_inner_nonperiodic_native_global_ap_boundary_action_structure_manifest_wp10c9d6c7c3b5c4f25fizzu as runner


def test_parent_artifacts_are_hash_locked_and_physics_remains_missing():
    inputs, ports = runner._validate_parents()
    assert "summary.json" in inputs and "prefix_port_payloads.npz" in ports


def test_contract_freezes_exact_sbp_sat_identity_and_claim_boundary():
    contract = runner._contract()
    assert contract["native_state"]["global_dimension"] == 1232
    assert contract["certificate"]["inner_incoming_count"] == 0
    assert contract["certificate"]["outer_incoming_count"] == 11
    assert contract["binding_energy_identity"]["no_sample_only_claim"]
    assert not contract["scientific_boundary"]["complete_cycle_execution_authorized"]


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_manifest_closes_and_is_definitions_only():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert not summary["nonperiodic_global_AP_boundary_action_certified"]
    assert not summary["physical_model_complete"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
