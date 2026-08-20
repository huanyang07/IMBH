from __future__ import annotations

import json

import numpy as np
import pytest

import run_causal_inner_shell_gated_atlas_rate_validation_wp10c9d6c7c3b5c4f25ck as f25ck


@pytest.fixture(scope="module")
def frozen_inputs():
    frozen = f25ck._validate_manifest(require_clean=False)
    return frozen, f25ck._load_inputs(frozen)


def test_manifest_authorizes_only_independent_rate_validation(frozen_inputs):
    frozen, _inputs = frozen_inputs
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25ck.WORK_PACKAGE
    assert frozen["summary"]["coefficients_frozen_before_truth"]
    assert frozen["geometry"]["candidate_primitive_states"].shape == (8, 112, 5)


def test_inputs_are_mixed_holdouts_with_frozen_extension(frozen_inputs):
    _frozen, inputs = frozen_inputs
    assert inputs["states"].shape == (8, 112, 5)
    assert inputs["deltas"].shape == (8, 560)
    assert inputs["departures"].shape == (8, 28)
    assert tuple(inputs["direction_indices"]) == (0, 1, 2, 3, 0, 1, 2, 3)
    assert tuple(inputs["component_bounds"][:4]) == (0.0125,) * 4
    assert tuple(inputs["component_bounds"][4:]) == (0.015,) * 4
    assert inputs["extension"]["extension_center_directions"].shape == (4, 28)


def test_atlas_prediction_is_finite_and_does_not_refit(frozen_inputs):
    _frozen, inputs = frozen_inputs
    prediction = f25ck._predict_atlas(
        inputs["model"],
        inputs["extension"],
        inputs["deltas"][0],
        inputs["departures"][0],
    )
    assert prediction["online_coordinate"].shape == (470,)
    assert prediction["decoded_delta"].shape == (560,)
    assert prediction["predicted_full_state_rate"].shape == (560,)
    assert prediction["predicted_a28_rate"].shape == (28,)
    assert 0.0 <= prediction["shell_weight"] <= 1.0
    assert all(
        np.all(np.isfinite(prediction[name]))
        for name in (
            "decoded_delta",
            "decoded_coordinate",
            "predicted_full_state_rate",
            "predicted_a28_rate",
        )
    )


def test_progress_schema_is_restartable_and_complete():
    progress = f25ck._empty_progress()
    assert progress["evaluations"] == []
    assert progress["failures"] == []
    for name, shape in f25ck._progress_array_shapes().items():
        assert progress[name].shape == (0,) + shape


def test_model_gate_checks_bind_all_prospective_thresholds():
    gates = f25ck.manifest._contract()["binding_independent_model_gates"]
    metrics = {
        "maximum_full_state_rate_relative_error": 0.15,
        "median_full_state_rate_relative_error": 0.075,
        "maximum_a28_rate_relative_error": 0.15,
        "radial_sign_disagreement_count": 0,
        "maximum_decoder_full_state_relative_error": 0.005,
        "maximum_decoder_coordinate_relative_mismatch": 0.005,
    }
    assert all(f25ck._model_checks(metrics, gates).values())
    metrics["maximum_full_state_rate_relative_error"] = 0.1500001
    assert not f25ck._model_checks(metrics, gates)["maximum_full_state_rate_error"]


def test_canonical_validation_if_present():
    if not f25ck.CANONICAL_DIRECTORY.exists():
        return
    f25ck._checksums(f25ck.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25ck.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25ck.CANONICAL_DIRECTORY / "rate_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    arrays = np.load(
        f25ck.CANONICAL_DIRECTORY / "rate_arrays.npz", allow_pickle=False
    )
    assert summary["classification"] in {
        f25ck.PASS_CLASSIFICATION,
        f25ck.FAIL_CLASSIFICATION,
    }
    assert summary["passed"] == (
        all(metrics["truth_checks"].values())
        and all(metrics["model_checks"].values())
    )
    assert arrays["exact_online_470_coordinate_rates_per_second"].shape == (
        8,
        470,
    )
    assert not summary["geometry_candidate_became_atlas_center"]
    assert not summary["predictive_cycle_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
