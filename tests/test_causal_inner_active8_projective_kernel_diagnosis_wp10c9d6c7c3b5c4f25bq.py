from __future__ import annotations

import numpy as np

import run_causal_inner_active8_projective_kernel_diagnosis_wp10c9d6c7c3b5c4f25bq as f25bq


def test_parent_rejection_and_truth_database_are_preserved():
    frozen = f25bq._validate_parent(require_clean=False)
    assert not frozen["summary"]["passed"]
    assert frozen["summary"]["truth_database_passed"]
    assert not frozen["summary"]["independent_model_validation_passed"]
    assert frozen["summary"]["authorized_next"] is None


def test_even_kernel_is_projective_even_and_positive_semidefinite():
    rng = np.random.default_rng(41)
    points = rng.normal(size=(19, 8))
    points /= np.linalg.norm(points, axis=1)[:, None]
    kernel = f25bq._even_kernel(points, points)
    assert np.allclose(f25bq._even_kernel(-points, points), kernel)
    assert np.min(np.linalg.eigvalsh(kernel)) >= -1.0e-12


def test_frozen_architecture_is_small_and_has_no_online_truth_path():
    assert f25bq.EVEN_TARGET_WEIGHT_EXPONENT == 2.0
    assert f25bq.EVEN_TIKHONOV_REGULARIZATION == 1.0 / 64.0
    assert f25bq.EVEN_QUARTIC_KERNEL_WEIGHT == 1.0 / 320.0
    assert f25bq.TOTAL_NONLINEAR_COEFFICIENT_COUNT == 7200


def test_revealed_diagnostic_passes_but_requires_new_holdout():
    diagnosis, arrays = f25bq._diagnose()
    assert all(diagnosis["checks"].values())
    assert diagnosis["metrics"]["diagnostic_only_revealed_validation"]
    assert diagnosis["metrics"][
        "tuning_maximum_nonlinear_departure_rate_relative_error"
    ] <= 0.25
    assert diagnosis["metrics"][
        "holdout_maximum_nonlinear_departure_rate_relative_error"
    ] <= 0.25
    assert diagnosis["metrics"][
        "tuning_maximum_full_departure_rate_relative_error"
    ] <= 0.05
    assert diagnosis["metrics"][
        "holdout_maximum_full_departure_rate_relative_error"
    ] <= 0.05
    assert arrays["even_kernel_coefficients"].shape == (120, 28)
    assert arrays["odd_cubic_coefficients"].shape == (120, 28)
    assert f25bq.AUTHORIZED_NEXT.startswith("definitions_only_")
