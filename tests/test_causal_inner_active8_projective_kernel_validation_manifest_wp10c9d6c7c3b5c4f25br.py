from __future__ import annotations

import numpy as np

import run_causal_inner_active8_projective_kernel_validation_manifest_wp10c9d6c7c3b5c4f25br as f25br


def test_parent_authorizes_only_new_independent_manifest():
    frozen = f25br._validate_parent(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["diagnostic_only"]
    assert not frozen["summary"]["independent_model_validation_passed"]
    assert frozen["summary"]["authorized_next"] == f25br.parent.AUTHORIZED_NEXT


def test_new_holdout_design_is_separated_and_exactly_sized():
    metrics, arrays = f25br._design()
    assert arrays["revealed_high_directions_active8"].shape == (8, 144)
    assert arrays["new_holdout_directions_active8"].shape == (8, 16)
    assert arrays["new_radial_directions_active8"].shape == (8, 8)
    assert metrics["planned_candidate_count"] == 48
    assert metrics[
        "minimum_new_holdout_to_revealed_projective_separation"
    ] >= 0.265
    assert metrics["minimum_new_holdout_mutual_projective_separation"] >= 0.27
    assert metrics["maximum_absolute_new_direction_component"] <= 0.75
    assert np.array_equal(
        arrays["new_radial_directions_active8"],
        arrays["new_holdout_directions_active8"][:, :8],
    )


def test_contract_freezes_architecture_leakage_and_gates():
    contract = f25br._contract()
    architecture = contract["mathematical_architecture"]
    assert architecture["even_target_weight_exponent"] == 2.0
    assert architecture["even_Tikhonov_regularization"] == 1.0 / 64.0
    assert architecture["online_truth_calls_per_macrostep"] == 0
    assert architecture["online_Newton_retractions_per_macrostep"] == 0
    assert architecture["stored_nonlinear_coefficients_after_refit"] == 7872
    assert contract["leakage_control"][
        "all_coefficients_frozen_and_hashed_before_new_rate_truth"
    ]
    assert contract["binding_independent_model_gates"][
        "holdout_maximum_nonlinear_departure_rate_relative_error"
    ] == 0.25
    assert contract["binding_independent_model_gates"][
        "holdout_maximum_full_departure_rate_relative_error"
    ] == 0.05
    assert not contract["decision"]["predictive_cycle_authorized"]


def test_design_checks_pass():
    metrics, _arrays = f25br._design()
    checks = f25br._design_checks(metrics, f25br._contract()["design_gates"])
    assert all(checks.values())
