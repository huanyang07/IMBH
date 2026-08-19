from __future__ import annotations

import numpy as np

import run_causal_inner_active8_mixed_parity_database_manifest_wp10c9d6c7c3b5c4f25bj as f25bj


def test_parent_authorizes_only_the_mixed_database_manifest():
    frozen = f25bj._validate_parent()
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == (
        "definitions_only_active8_mixed_direction_parity_database_manifest"
    )


def test_direction_design_is_full_rank_separated_and_deterministic():
    metrics, arrays = f25bj._direction_design()
    repeated_metrics, repeated_arrays = f25bj._direction_design()
    assert metrics == repeated_metrics
    for name in arrays:
        assert np.array_equal(arrays[name], repeated_arrays[name])
    assert arrays["training_directions_active8"].shape == (8, 40)
    assert arrays["tuning_directions_active8"].shape == (8, 8)
    assert arrays["holdout_directions_active8"].shape == (8, 8)
    assert metrics["quadratic_feature_rank"] == 36
    assert metrics["cubic_kernel_rank"] == 40
    assert metrics["quartic_kernel_rank"] == 40
    assert metrics["quadratic_feature_condition_number"] <= 8.0
    assert metrics["cubic_kernel_condition_number"] <= 10.0
    assert metrics["quartic_kernel_condition_number"] <= 10.0
    assert metrics["validation_to_training_projective_separation"] >= 0.30


def test_decoder_diagnosis_selects_cubic_odd_and_quartic_even():
    diagnosis = f25bj._decoder_diagnosis()
    assert diagnosis["post_result_diagnosis"]
    assert not diagnosis["independent_validation_claimed"]
    assert 2.9 <= diagnosis["median_odd_decoder_exponent"] <= 3.1
    assert 3.8 <= diagnosis["median_even_decoder_exponent"] <= 4.2
    assert diagnosis["cubic_decoder_rank4_energy"] >= 0.95
    assert diagnosis["quartic_decoder_rank4_energy"] >= 0.95


def test_contract_preserves_the_470_architecture_and_holds_out_data():
    contract = f25bj._contract()
    assert contract["fixed_online_architecture"]["state"] == (
        "q162_plus_z280_plus_a28_equals_470"
    )
    assert contract["database_design"]["planned_total_candidates"] == 128
    assert contract["database_design"]["untouched_holdout_directions_at_0p01"] == 8
    assert contract["closure_models"]["online_truth_calls"] == 0
    assert contract["closure_models"]["online_Newton_retractions"] == 0
    assert contract["parity_targets"]["hidden_decoder_odd"] == "cubic"
    assert contract["parity_targets"]["hidden_decoder_even"] == "quartic"
    assert contract["decision"]["geometry_pass"]["authorizes_only"] == (
        "WP10c9d6c7c3b5c4f25bl"
    )
    assert not contract["claim_boundary"]["local_closure_validated"]
    assert not contract["claim_boundary"]["predictive_cycle_authorized"]
