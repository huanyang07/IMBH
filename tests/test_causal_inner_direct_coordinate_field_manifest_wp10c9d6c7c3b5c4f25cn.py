from __future__ import annotations

import numpy as np
import pytest

import run_causal_inner_direct_coordinate_field_manifest_wp10c9d6c7c3b5c4f25cn as f25cn


@pytest.fixture(scope="module")
def closure_fit():
    return f25cn._fit_q_closure()


def test_parent_authorizes_only_a_definitions_only_transition_forecast():
    frozen = f25cn._validate_parent(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["classification"] == f25cn.parent.FULL_CLASSIFICATION
    assert frozen["summary"]["authorized_next"] == (
        "definitions_only_recentered_transition_forecast_manifest"
    )
    assert frozen["summary"]["new_truth_rate_calls"] == 0
    assert not frozen["summary"]["trajectory_authorized"]


def test_q_kernel_is_exactly_anchor_subtracted():
    centers = np.eye(3)
    anchored = f25cn._anchored_kernel(np.zeros((1, 3)), centers)
    assert np.array_equal(anchored, np.zeros((1, 3)))


def test_direct_coordinate_fit_closes_q_rate_and_preserves_other_blocks(
    closure_fit,
):
    arrays, metrics = closure_fit
    assert arrays["q_rate_centers"].shape == (16, 28)
    assert arrays["q_rate_coefficients"].shape == (16, 162)
    assert metrics["kernel_rank"] == 16
    assert metrics["kernel_condition_number"] < 1.0e3
    assert metrics["maximum_training_full_coordinate_rate_relative_error"] < 0.05
    assert metrics["maximum_training_q162_rate_relative_error"] < 1.0e-6
    assert metrics["maximum_training_z280_rate_relative_error"] < 0.15
    assert metrics["maximum_training_a28_rate_relative_error"] < 0.15
    assert metrics["origin_q_rate_correction_norm"] < 1.0e-14
    assert metrics["maximum_direct_field_implementation_relative_defect"] < 1.0e-12
    assert metrics["state_dependent_coordinate_Jacobian_calls"] == 0


def test_online_field_does_not_use_coordinate_jacobian(monkeypatch, closure_fit):
    arrays, _metrics = closure_fit
    model = f25cn.parent.manifest.parent.vector_field.ReducedVectorField()

    def forbidden_coordinate_jacobian(*_args, **_kwargs):
        raise AssertionError("online direct field rebuilt the coordinate Jacobian")

    monkeypatch.setattr(
        f25cn.parent.manifest.parent.vector_field.manifest.parent.geometry.chart_tools,
        "_coordinate_jacobian",
        forbidden_coordinate_jacobian,
    )
    field = f25cn.DirectCoordinateField(arrays, model=model)
    coordinate = arrays["training_online_coordinates"][0]
    rate = field.field(coordinate)
    assert rate.shape == (470,)
    assert np.all(np.isfinite(rate))


def test_manifest_freezes_fresh_exact_holdout_without_trajectory():
    contract = f25cn._contract()
    holdout = contract["independent_exact_rate_holdout"]
    boundaries = contract["authorization_boundaries"]
    assert holdout["count"] == 8
    assert holdout["coefficients_frozen_before_truth"]
    assert holdout["state_may_not_become_chart_center"]
    assert boundaries["new_truth_rate_calls_during_manifest"] == 0
    assert boundaries["new_generator_assemblies"] == 0
    assert boundaries["new_nonlinear_roots"] == 0
    assert boundaries["propagated_states"] == 0
    assert not boundaries["trajectory_authorized"]
    assert not boundaries["predictive_cycle_authorized"]
    assert not boundaries["reduced_slow_evolution_authorized"]
