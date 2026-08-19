from __future__ import annotations

import numpy as np

import run_causal_inner_expanded_departure_rate_screen_wp10c9d6c7c3b5c4f25be as f25be


def test_frozen_manifest_is_locked():
    frozen = f25be._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25be.WORK_PACKAGE
    assert frozen["summary"]["planned_nonbase_continuous_rate_evaluations"] == 16


def test_expanded_inputs_have_sixteen_ordered_states():
    inputs = f25be._load_inputs()
    assert inputs["states"].shape == (16, 112, 5)
    assert inputs["coordinates"].shape == (16, 28)
    assert [item["candidate_index"] for item in inputs["candidates"]] == list(
        range(16)
    )


def test_pair_analysis_recovers_known_cubic_amplification():
    directions = np.eye(2)
    candidates = [
        {"direction_index": 0, "sign": -1},
        {"direction_index": 0, "sign": 1},
        {"direction_index": 1, "sign": -1},
        {"direction_index": 1, "sign": 1},
    ]
    coordinates = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]])
    linear = coordinates.copy()
    rates = np.array([[-10.0, 0.0], [10.0, 0.0], [0.0, -10.0], [0.0, 10.0]])
    prior = {
        "effective_departure_radii": np.column_stack(
            (np.full(2, 0.25), np.full(2, 0.5), np.full(2, 1.0))
        ),
        "central_radial_growth_per_second": np.column_stack(
            (np.full(2, 1.0625), np.full(2, 1.25), np.full(2, 2.0))
        ),
        "central_departure_nonlinear_relative_defects": np.column_stack(
            (np.full(2, 0.0625), np.full(2, 0.25), np.full(2, 1.0))
        ),
    }
    metrics, arrays = f25be._pair_analysis(
        candidates, coordinates, rates, linear, directions, prior
    )
    assert np.allclose(arrays["effective_departure_radii"], 2.0)
    assert np.allclose(arrays["central_radial_growth_per_second"], 5.0)
    assert np.allclose(arrays["central_departure_nonlinear_relative_defects"], 4.0)
    assert np.allclose(arrays["nonlinear_amplification_from_0p005"], 4.0)
    assert np.allclose(arrays["effective_amplitude_exponents"], 2.0)
    assert metrics["median_current_departure_nonlinear_relative_defect"] == 4.0


def test_classification_keeps_unresolved_and_failed_paths_separate():
    classification, next_artifact = f25be._classify(True, 0.2)
    assert classification == f25be.NONLINEAR_CLASSIFICATION
    assert next_artifact == "definitions_only_mixed_direction_adaptive_28D_database_manifest"
    classification, next_artifact = f25be._classify(True, 0.02)
    assert classification == f25be.UNRESOLVED_CLASSIFICATION
    assert next_artifact == "definitions_only_exact_departure_chart_amplitude_0p02_manifest"
    classification, next_artifact = f25be._classify(False, 1.0)
    assert classification == f25be.FAIL_CLASSIFICATION
    assert next_artifact is None
