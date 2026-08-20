from __future__ import annotations

import numpy as np

import run_causal_inner_departure28_independent_validation_manifest_wp10c9d6c7c3b5c4f25bv as f25bv


def test_parent_authorizes_only_definitions_only_independent_manifest():
    frozen = f25bv._validate_parent(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["diagnostic_only"]
    assert frozen["summary"]["new_truth_evaluations"] == 0
    assert frozen["summary"]["authorized_next"] == (
        "definitions_only_departure28_dual_polynomial_independent_validation_manifest"
    )


def test_new_directions_are_maximin_and_untouched():
    metrics, arrays = f25bv._design()
    assert arrays["revealed_high_directions_active8"].shape == (8, 160)
    assert arrays["new_holdout_directions_active8"].shape == (8, 16)
    assert arrays["new_radial_directions_active8"].shape == (8, 8)
    assert metrics["minimum_new_holdout_to_revealed_projective_separation"] >= 0.257
    assert metrics["minimum_new_holdout_mutual_projective_separation"] >= 0.28
    assert metrics["maximum_absolute_new_direction_component"] <= 0.75
    assert np.array_equal(
        arrays["new_radial_directions_active8"],
        arrays["new_holdout_directions_active8"][:, :8],
    )


def test_contract_freezes_full_departure_architecture_and_original_gates():
    contract = f25bv._contract()
    architecture = contract["mathematical_architecture"]
    assert architecture["online_state"] == "q162_plus_z280_plus_a28_equals_470"
    assert architecture["departure_rate_input_dimension"] == 28
    assert architecture["even_rate_kernel"] == "dot_squared"
    assert architecture["odd_rate_kernel"] == "dot_cubed"
    assert architecture["even_target_weight_exponent"] == 1.0
    assert architecture["odd_target_weight_exponent"] == 0.0
    assert architecture["dynamic_curvature_augmentation"] is False
    assert architecture["online_truth_calls_per_macrostep"] == 0
    assert architecture["online_Newton_retractions_per_macrostep"] == 0
    assert architecture["stored_total_nonlinear_coefficients"] == 9_440
    gates = contract["binding_independent_model_gates"]
    assert gates["holdout_maximum_nonlinear_departure_rate_relative_error"] == 0.25
    assert gates["holdout_maximum_full_departure_rate_relative_error"] == 0.05
    assert contract["binding_fit_gates"]["even_system_condition_number"] == 1.0e8
    assert contract["binding_fit_gates"]["odd_system_condition_number"] == 1.0e7


def test_manifest_authorizes_geometry_only():
    contract = f25bv._contract()
    assert f25bv.AUTHORIZED_NEXT == "WP10c9d6c7c3b5c4f25bw"
    assert contract["decision"]["predictive_cycle_authorized"] is False
    assert contract["decision"]["reduced_slow_evolution_authorized"] is False
    assert contract["leakage_control"][
        "all_rate_coefficients_frozen_and_hashed_before_new_rate_truth"
    ]
    assert contract["leakage_control"]["certified_rank4_decoder_remains_frozen"]
