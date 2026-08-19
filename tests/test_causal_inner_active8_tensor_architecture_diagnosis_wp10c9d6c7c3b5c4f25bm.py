from __future__ import annotations

import numpy as np

import run_causal_inner_active8_tensor_architecture_diagnosis_wp10c9d6c7c3b5c4f25bm as f25bm


def test_parent_rejection_and_truth_database_are_preserved():
    frozen = f25bm._validate_parent(require_clean=False)
    assert not frozen["summary"]["passed"]
    assert frozen["summary"]["truth_database_passed"]
    assert frozen["summary"]["authorized_next"] is None


def test_cubic_features_have_the_polynomial_kernel_identity():
    rng = np.random.default_rng(23)
    left = rng.normal(size=(7, 8))
    right = rng.normal(size=(11, 8))
    actual = f25bm._cubic_features(left) @ f25bm._cubic_features(right).T
    expected = (left @ right.T) ** 3
    assert f25bm._cubic_features(left).shape == (7, 120)
    assert np.allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


def test_extended_design_is_full_rank_conditioned_and_separated():
    metrics, arrays = f25bm._extended_direction_design()
    assert arrays["total_training_directions_active8"].shape == (8, 120)
    assert arrays["new_tuning_directions_active8"].shape == (8, 8)
    assert arrays["new_holdout_directions_active8"].shape == (8, 16)
    assert metrics["quadratic_feature_rank"] == 36
    assert metrics["cubic_feature_rank"] == 120
    assert metrics["cubic_feature_condition_number"] <= 25.0
    assert metrics["minimum_new_validation_to_training_projective_separation"] >= 0.27
    assert metrics["new_candidate_count"] == 192


def test_rank4_curvature_is_capacity_only_and_not_dynamically_augmented():
    metrics, arrays = f25bm._capacity_analysis()
    assert arrays["rank4_curvature_basis"].shape == (560, 4)
    assert metrics["training_hidden_singular_energy_rank4"] >= 0.99
    assert metrics["rank4_odd_only_capacity"]["revealed_holdout"][
        "maximum_full_scaled_state_relative_error"
    ] <= 1.0e-3
    assert metrics["dynamic_augmentation_unstable_eigenvalue_count"] >= 1
