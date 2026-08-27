import json

import pytest

import run_causal_inner_cycle_wide_missing_input_acquisition_and_event_reset_manifest_wp10c9d6c7c3b5c4f25fizzs as runner


def test_negative_event_evidence_is_locked_and_microstepping_is_rejected():
    _, _, old, prognosis = runner._validate_parent()
    contract = runner._contract(old, prognosis)
    negative = contract["binding_negative_evidence"]
    assert negative["old_exact_witnesses"] == 192
    assert negative["old_completed_patches"] == 64
    assert not negative["old_cycle_return_observed"]
    assert not negative["old_hot_exit_observed"]
    assert contract["prohibitions"]["continue_old_hot_exit_microstepping"]


def test_reset_is_ledger_exact_by_construction_but_uncalibrated():
    _, _, old, prognosis = runner._validate_parent()
    contract = runner._contract(old, prognosis)
    reset = contract["conservative_reset_structure"]
    assert "C(z_plus-z_minus)=DeltaQ_event" == reset["identity"]
    assert len(reset["physical_calibration_missing"]) == 4
    assert not contract["required_physical_inputs"]["impact_guard_and_reset_truth"]
    assert not contract["required_physical_inputs"]["hot_exit_guard_and_reset_truth"]


def test_manifest_stops_before_complete_cycle():
    _, _, old, prognosis = runner._validate_parent()
    contract = runner._contract(old, prognosis)
    assert contract["physical_target"]["maximum_online_macrosteps"] == 100000
    assert contract["prohibitions"]["complete_cycle_runner_or_step"]
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]
    assert contract["claim_boundary"]["complete_cycle_steps"] == 0


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_manifest_closes():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert not summary["cycle_wide_physical_inputs_complete"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
