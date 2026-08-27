import json

import pytest

import run_causal_inner_heldout_atlas_and_hybrid_sequence_validation_manifest_wp10c9d6c7c3b5c4f25fizzw as runner


def test_parent_atlas_certificate_is_hash_locked():
    assert "summary.json" in runner._validate_parent()


def test_contract_advances_event_time_phase_and_q_together():
    contract = runner._contract()
    extension = contract["event_time_schema_extension"]
    assert "integrated_phase_advance" in extension
    assert extension["reset_order"] == ["localize entry guard", "apply integrated q ledger impulse", "advance elapsed time by duration", "advance unwrapped phase", "switch mode", "require destination guard margin"]
    assert contract["smooth_reduced_flow"]["state"] == "y=(q1,q2,q3,q4,phi_unwrapped)"
    assert contract["structure_certificate"]["complete_cycle_steps"] == 0


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_heldout_sequence_manifest_is_nonexecuting():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert summary["finite_event_phase_advance_frozen"]
    assert not summary["heldout_validator_certified"]
    assert not summary["hybrid_sequence_validator_certified"]
    assert not summary["physical_payloads_acquired"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
