from __future__ import annotations

import json

import numpy as np
import pytest

import run_causal_inner_direct_coordinate_field_validation_wp10c9d6c7c3b5c4f25co as f25co


@pytest.fixture(scope="module")
def frozen_inputs():
    frozen = f25co._validate_manifest(require_clean=False)
    return frozen, f25co._load_inputs(frozen)


def test_manifest_froze_field_before_holdout_truth(frozen_inputs):
    frozen, inputs = frozen_inputs
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25co.WORK_PACKAGE
    assert frozen["summary"]["new_truth_rate_calls"] == 0
    assert frozen["summary"]["state_dependent_coordinate_Jacobian_online"] is False
    assert inputs["states"].shape == (8, 112, 5)
    assert inputs["coordinates"].shape == (8, 470)


def test_online_prediction_executes_with_jacobian_helper_forbidden(frozen_inputs):
    _frozen, inputs = frozen_inputs
    coordinate_rate, full_rate, wall = (
        f25co._online_prediction_without_coordinate_jacobian(
            inputs["direct"], inputs["coordinates"][0]
        )
    )
    assert coordinate_rate.shape == (470,)
    assert full_rate.shape == (560,)
    assert np.all(np.isfinite(coordinate_rate))
    assert np.all(np.isfinite(full_rate))
    assert wall >= 0.0


def test_field_gate_binds_every_coordinate_block():
    gates = f25co.manifest._contract()["binding_independent_field_gates"]
    metrics = {
        "maximum_full_state_rate_relative_error": 0.15,
        "median_full_state_rate_relative_error": 0.075,
        "maximum_full_coordinate_rate_relative_error": 0.15,
        "median_full_coordinate_rate_relative_error": 0.075,
        "maximum_q162_rate_relative_error": 0.15,
        "median_q162_rate_relative_error": 0.075,
        "maximum_z280_rate_relative_error": 0.15,
        "maximum_a28_rate_relative_error": 0.15,
        "radial_sign_disagreement_count": 0,
        "maximum_decoder_full_state_relative_error": 0.005,
        "maximum_decoder_coordinate_relative_mismatch": 0.005,
        "online_state_dependent_coordinate_Jacobian_calls": 0,
    }
    assert all(f25co._field_checks(metrics, gates).values())
    metrics["maximum_q162_rate_relative_error"] = 0.150001
    assert not f25co._field_checks(metrics, gates)["maximum_q162_rate"]


def test_progress_arrays_cover_truth_and_prediction():
    shapes = f25co._progress_array_shapes()
    assert shapes["total_rates_per_second"] == (560,)
    assert shapes["exact_online_470_coordinate_rates_per_second"] == (470,)
    assert shapes["predicted_online_470_coordinate_rates_per_second"] == (470,)
    assert shapes["repaired_decoded_scaled_deltas"] == (560,)


def test_canonical_result_if_present():
    if not f25co.CANONICAL_DIRECTORY.exists():
        return
    f25co._checksums(f25co.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25co.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25co.CANONICAL_DIRECTORY / "rate_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["classification"] in {
        f25co.PASS_CLASSIFICATION,
        f25co.FAIL_CLASSIFICATION,
    }
    assert summary["passed"] == (
        metrics["truth_passed"] and metrics["field_passed"]
    )
    assert summary["online_state_dependent_coordinate_Jacobian_calls"] == 0
    assert not summary["predictive_cycle_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
