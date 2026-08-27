import json

import numpy as np
import pytest

import run_causal_inner_heldout_atlas_and_hybrid_sequence_validator_wp10c9d6c7c3b5c4f25fizzw1 as runner


def test_parent_manifest_is_hash_locked_and_nonexecuting():
    _, contract = runner._validate_parent()
    assert contract["structure_certificate"]["complete_cycle_steps"] == 0


def test_synthetic_validator_preflight_passes_and_wrong_order_fails():
    metrics, arrays = runner._certificate()
    assert metrics["passed"]
    assert metrics["minimum_smooth_observed_order"] >= 4.5
    assert metrics["hybrid_event_names"] == ["cold_to_hot", "hot_to_recovery"]
    assert metrics["checkpoint_roundtrip_bitwise"]
    assert metrics["restart_suffix_replay_bitwise"]
    assert metrics["heldout_audit_passed"]
    assert metrics["wrong_event_order_rejected"]
    assert np.array_equal(arrays["hybrid_final_state"], arrays["replayed_final_state"])


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(), reason="validator not executed")
def test_canonical_validator_certificate_closes():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    metrics = json.loads((runner.CANONICAL_DIRECTORY / "validator_metrics.json").read_text())
    assert summary["passed"] and summary["heldout_validator_structure_certified"]
    assert summary["hybrid_sequence_validator_structure_certified"]
    assert summary["finite_event_phase_advance_certified"]
    assert metrics["complete_cycle_steps"] == 0
    assert not summary["physical_payloads_acquired"]
    assert not summary["heldout_physical_validation_complete"]
    assert not summary["complete_cycle_execution_authorized"]
