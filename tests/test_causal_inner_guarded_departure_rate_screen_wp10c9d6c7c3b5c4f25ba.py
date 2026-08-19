from __future__ import annotations

import numpy as np

import run_causal_inner_guarded_departure_rate_screen_wp10c9d6c7c3b5c4f25ba as f25ba


def test_frozen_rate_screen_manifest_is_locked():
    frozen = f25ba._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25ba.WORK_PACKAGE
    assert frozen["summary"]["planned_nonbase_continuous_rate_evaluations"] == 48


def test_hash_locked_rate_screen_inputs_have_expected_dimensions():
    inputs = f25ba._load_inputs()
    assert inputs["states"].shape[0] == 48
    assert inputs["deltas"].shape == (48, 560)
    assert inputs["coordinates"].shape == (48, 28)
    assert inputs["generator"].shape == (560, 560)
    assert inputs["departure_basis"].shape == (560, 28)


def test_radial_analysis_recovers_linear_and_cubic_signal():
    directions = np.eye(2)
    candidates = []
    coordinates = []
    rates = []
    linear = []
    radii = (0.1, 0.2, 0.3)
    for direction_index in range(2):
        direction = directions[:, direction_index]
        for amplitude_index, radius in enumerate(radii):
            for sign in (-1, 1):
                coordinate = sign * radius * direction
                linear_rate = 2.0 * coordinate
                nonlinear_rate = linear_rate - coordinate * radius**2
                candidates.append(
                    {
                        "direction_index": direction_index,
                        "amplitude_index": amplitude_index,
                        "sign": sign,
                    }
                )
                coordinates.append(coordinate)
                rates.append(nonlinear_rate)
                linear.append(linear_rate)
    metrics, arrays = f25ba._radial_analysis(
        candidates,
        np.asarray(coordinates),
        np.asarray(rates),
        np.asarray(linear),
        directions,
    )
    assert metrics["negative_fitted_cubic_count"] == 2
    assert np.all(arrays["central_radial_growth_per_second"][:, -1] < 2.0)


def test_signal_classifier_separates_sampling_paths_without_claiming_closure():
    classification, next_artifact = f25ba._classify(True, 0.2)
    assert classification == f25ba.NONLINEAR_CLASSIFICATION
    assert next_artifact == (
        "definitions_only_mixed_direction_adaptive_28D_database_manifest"
    )
    classification, next_artifact = f25ba._classify(True, 0.01)
    assert classification == f25ba.UNRESOLVED_CLASSIFICATION
    assert next_artifact == "definitions_only_expanded_safe_departure_chart_manifest"
    assert f25ba._classify(False, 1.0) == (f25ba.FAIL_CLASSIFICATION, None)
