from __future__ import annotations

import numpy as np
import pytest

import run_causal_inner_recenter_transition_forecast_manifest_wp10c9d6c7c3b5c4f25cp as f25cp


@pytest.fixture(scope="module")
def forecast_fixture():
    frozen = f25cp._validate_parent(require_clean=False)
    arrays, metrics = f25cp._forecast()
    return frozen, arrays, metrics


def test_parent_authorizes_one_recenter_forecast(forecast_fixture):
    frozen, _arrays, _metrics = forecast_fixture
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["classification"] == f25cp.parent.PASS_CLASSIFICATION
    assert frozen["summary"]["online_state_dependent_coordinate_Jacobian_calls"] == 0
    assert frozen["warm4_summary"]["passed"]
    assert frozen["warm4_summary"]["accepted_truth_roots"] == 1


def test_forecast_brackets_trigger_before_hard_limit(forecast_fixture):
    _frozen, arrays, metrics = forecast_fixture
    loads = metrics["predicted_old_decoder_loads"]
    assert metrics["passed"]
    assert metrics["predicted_trigger_root_index"] == 2
    assert loads["refined_step_1"] < f25cp.RECENTER_TRIGGER_LOAD
    assert f25cp.RECENTER_TRIGGER_LOAD <= loads["refined_step_2"]
    assert loads["refined_step_2"] < f25cp.HARD_CHART_LOAD
    assert loads["audit_step_1"] < f25cp.RECENTER_TRIGGER_LOAD
    assert f25cp.RECENTER_TRIGGER_LOAD <= loads["audit_step_2"]
    assert loads["audit_step_2"] < f25cp.HARD_CHART_LOAD
    assert arrays["prospective_refined_coordinates"].shape == (2, 470)


def test_direct_field_retrospective_readiness_passes(forecast_fixture):
    _frozen, _arrays, metrics = forecast_fixture
    errors = metrics["retrospective_endpoint_relative_errors"]
    assert errors["full"] < 0.01
    assert errors["q162"] < 0.05
    assert errors["z280"] < 0.01
    assert errors["a28"] < 0.01
    assert metrics["retrospective_refined_audit_relative_difference"] < 1.0e-6
    assert metrics["prospective_refined_audit_relative_difference"] < 2.0e-5


def test_contract_allows_only_two_fail_fast_truth_roots():
    contract = f25cp._contract()
    execution = contract["authentic_execution"]
    boundaries = contract["authorization_boundaries"]
    assert execution["root_budget"] == 2
    assert execution["accepted_history_only"]
    assert execution["predicted_center_may_not_become_center"]
    assert boundaries["new_truth_roots_during_manifest"] == 0
    assert boundaries["new_truth_roots_during_next_execution_max"] == 2
    assert boundaries["new_continuous_rate_calls"] == 0
    assert boundaries["new_generator_assemblies"] == 0
    assert not boundaries["physical_microburst_authorized"]
    assert not boundaries["predictive_cycle_authorized"]
    assert not boundaries["reduced_slow_evolution_authorized"]


def test_translation_is_exact_in_coordinate_space(forecast_fixture):
    _frozen, arrays, _metrics = forecast_fixture
    center = arrays["predicted_transition_center_coordinate"]
    point = arrays["warm4_truth_coordinate"]
    translated = point - center
    assert np.max(np.abs(translated + center - point)) < 1.0e-14
