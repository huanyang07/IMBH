from __future__ import annotations

import numpy as np
import pytest

import run_causal_inner_compensated_decoder_repair_manifest_wp10c9d6c7c3b5c4f25cl as f25cl


@pytest.fixture(scope="module")
def repair_fit():
    return f25cl._fit_repair(include_physical_audits=False)


def test_parent_failure_is_decoder_only():
    frozen = f25cl._validate_parent(require_clean=False)
    assert frozen["summary"]["truth_passed"]
    assert not frozen["summary"]["independent_model_passed"]
    assert not frozen["metrics"]["model_checks"]["decoder_full_state_error"]
    assert all(
        passed
        for name, passed in frozen["metrics"]["model_checks"].items()
        if name != "decoder_full_state_error"
    )


def test_repair_is_geometry_only_and_field_compensated():
    contract = f25cl._contract()
    repair = contract["decoder_repair"]
    compensation = contract["exact_field_compensation"]
    assert repair["uses_new_rate_truth"] is False
    assert repair["training_geometry_count"] == 16
    assert compensation["independently_validated_full_state_rate_is_unchanged"]
    assert "minus_G_times_C_geometry" in compensation["new_closure"]


def test_gaussian_repair_fit_closes_revealed_geometry_without_rate_change(repair_fit):
    arrays, metrics = repair_fit
    assert arrays["decoder_repair_centers"].shape == (16, 28)
    assert arrays["decoder_repair_coefficients"].shape == (16, 560)
    assert metrics["kernel_rank"] == 16
    assert metrics["kernel_condition_number"] < 1.0e4
    assert metrics["maximum_repaired_decoder_relative_error"] < 1.0e-6
    assert metrics["maximum_compensated_full_rate_invariance_defect"] < 1.0e-12


def test_repair_value_has_full_state_shape(repair_fit):
    arrays, _metrics = repair_fit
    value = f25cl._repair_value(
        arrays["training_departure_coordinates"][0],
        arrays["decoder_repair_centers"],
        arrays["decoder_repair_coefficients"],
    )
    assert value.shape == (560,)
    assert np.all(np.isfinite(value))


def test_holdout_is_new_mixed_corner_geometry():
    directions, labels, metrics = f25cl._holdout_design()
    assert directions.shape == (4, 28)
    assert np.allclose(
        np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=1.0e-14
    )
    assert len(set(labels)) == 4
    assert metrics["minimum_holdout_pair_separation"] > 0.0


def test_manifest_authorizes_no_trajectory():
    boundaries = f25cl._contract()["authorization_boundaries"]
    assert boundaries["new_truth_rate_calls"] == 0
    assert boundaries["new_generator_assemblies"] == 0
    assert boundaries["new_nonlinear_roots"] == 0
    assert boundaries["propagated_states"] == 0
    assert not boundaries["trajectory_authorized"]
    assert not boundaries["predictive_cycle_authorized"]
    assert not boundaries["reduced_slow_evolution_authorized"]
