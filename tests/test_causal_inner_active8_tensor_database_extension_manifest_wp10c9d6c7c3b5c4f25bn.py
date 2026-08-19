from __future__ import annotations

import numpy as np

import run_causal_inner_active8_tensor_database_extension_manifest_wp10c9d6c7c3b5c4f25bn as f25bn


def test_parent_authorizes_only_the_database_extension_manifest():
    frozen = f25bn._validate_parent(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25bn.parent.AUTHORIZED_NEXT
    assert not frozen["summary"]["independent_validation_claimed"]


def test_frozen_design_is_complete_conditioned_and_separated():
    metrics, arrays = f25bn._design()
    repeated_metrics, repeated_arrays = f25bn._design()
    assert metrics == repeated_metrics
    for name in arrays:
        assert np.array_equal(arrays[name], repeated_arrays[name])
    assert arrays["total_training_directions_active8"].shape == (8, 120)
    assert arrays["new_tuning_directions_active8"].shape == (8, 8)
    assert arrays["new_holdout_directions_active8"].shape == (8, 16)
    assert arrays["rank4_curvature_basis"].shape == (560, 4)
    assert metrics["quadratic_feature_rank"] == 36
    assert metrics["cubic_feature_rank"] == 120
    assert metrics["cubic_feature_condition_number"] <= 25.0
    assert metrics["minimum_validation_to_training_projective_separation"] >= 0.27
    assert metrics["planned_candidate_count"] == 192
    assert all(f25bn._design_checks(metrics).values())


def test_contract_preserves_the_selected_mathematical_architecture():
    contract = f25bn._contract()
    architecture = contract["mathematical_architecture"]
    assert architecture["online_state"] == "q162_plus_z280_plus_a28_equals_470"
    assert architecture["curvature_is_algebraic_not_dynamic"]
    assert architecture["online_truth_calls_per_macrostep"] == 0
    assert architecture["online_Newton_retractions_per_macrostep"] == 0
    assert architecture["stored_nonlinear_coefficients"] == 4_848
    assert contract["database_design"]["planned_total_candidates"] == 192


def test_fit_is_hash_locked_before_validation_is_revealed():
    leakage = f25bn._contract()["leakage_control"]
    order = leakage["rate_evaluation_order"]
    assert order.index("freeze_and_hash_all_three_coefficient_maps") < order.index(
        "new_tuning_high_and_low"
    )
    assert order.index("freeze_and_hash_all_three_coefficient_maps") < order.index(
        "new_holdout_high"
    )
    assert leakage["fit_may_read_new_training_responses_only"]
    assert leakage["holdout_cannot_change_any_coefficient_or_threshold"]


def test_model_gates_bind_online_state_and_rate_errors_not_hidden90_error():
    gates = f25bn._contract()["binding_independent_model_gates"]
    assert gates["holdout_maximum_full_departure_rate_relative_error"] == 0.05
    assert gates["maximum_full_scaled_state_decoder_relative_error"] == 2.5e-3
    assert gates["maximum_reconstructed_C_phys_residual_infinity"] == 2.5e-4
    assert not any("hidden_decoder_relative_error" in name for name in gates)


def test_claim_boundary_still_blocks_trajectory_and_cycle():
    boundary = f25bn._contract()["claim_boundary"]
    assert boundary["old_kernel_model_remains_rejected"]
    assert not boundary["independent_validation_complete"]
    assert not boundary["trajectory_authorized"]
    assert not boundary["predictive_cycle_authorized"]
    assert not boundary["reduced_slow_evolution_authorized"]
