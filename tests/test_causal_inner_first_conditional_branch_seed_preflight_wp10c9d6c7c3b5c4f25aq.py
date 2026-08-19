from __future__ import annotations

import numpy as np

import run_causal_inner_first_conditional_branch_seed_preflight_wp10c9d6c7c3b5c4f25aq as f25aq


def test_frozen_branch_seed_manifest_is_locked():
    frozen = f25aq._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25aq.WORK_PACKAGE


def test_exact_mapped_only_coordinate_map_is_full_rank_and_differentiable():
    components = f25aq._coordinate_components()
    metrics, arrays = f25aq._coordinate_audit(components)
    assert metrics["coordinate_rank"] == 162
    assert metrics["coordinate_condition_number"] <= 1.0e4
    assert metrics["directional_derivative_relative_defect"] <= 1.0e-6
    assert metrics["Q3_rowspace_relative_defect"] <= 1.0e-10
    assert not metrics["responsive_height_one_form_used"]
    assert arrays["coordinate_jacobian"].shape == (162, 560)


def test_direct_predictor_is_linear_exact_but_outside_trust_region():
    components = f25aq._coordinate_components()
    metrics, arrays = f25aq._direct_predictor_audit(components)
    assert metrics["predictor_relative_linear_residual"] <= 1.0e-10
    assert metrics["predictor_maximum_scaled_component"] > 5.0e-3
    assert metrics["branch_linear_condition_number"] > 1.0e8
    assert metrics["nonbase_physical_truth_calls"] == 0
    assert not metrics["direct_root_attempted"]
    assert arrays["hidden_basis"].shape == (560, 398)
    assert np.isfinite(arrays["direct_branch_predictor"]).all()
