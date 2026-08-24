from __future__ import annotations

import numpy as np

import run_causal_inner_adaptive_metric_chart_radius_recovery_execution_wp10c9d6c7c3b5c4f25fin as target


def test_manifest_authorizes_exactly_one_recovery_segment() -> None:
    lock = target._validate_manifest(require_clean=False)
    assert lock["summary"]["classification"] == target.manifest.CLASSIFICATION
    assert lock["contract"]["history"]["recovery_segment_seconds"] == 0.001
    assert lock["contract"]["scope"]["new_exact_free_field_calls"] == 2


def test_checkpoint_advances_only_accepted_endpoint() -> None:
    seed = target._seed()
    endpoint_coordinate = np.ones(470)
    endpoint_state = np.ones((112, 5))
    endpoint_rate = np.full(470, 2.0)
    field_arrays = {
        "metric_transform470x470": np.eye(470),
        "metric_augmented560x560": np.eye(560),
        "gauge_basis560x90": np.zeros((560, 90)),
    }
    checkpoint = target._checkpoint_arrays(
        seed, endpoint_coordinate, endpoint_state, endpoint_rate, field_arrays
    )
    np.testing.assert_array_equal(
        checkpoint["previous_coordinate470"], seed["current_coordinate470"]
    )
    np.testing.assert_array_equal(checkpoint["current_coordinate470"], endpoint_coordinate)
    assert float(checkpoint["elapsed_seconds"]) == target.manifest.ENDPOINT_ELAPSED_SECONDS
    assert int(checkpoint["accepted_segments_total"]) == 92
    assert float(checkpoint["previous_span_seconds"]) == 0.001


def test_history_replay_is_bitwise_for_frozen_arithmetic() -> None:
    seed = target._seed()
    endpoint = seed["candidate_target470"]
    endpoint_rate = seed["current_coordinate_rate470_per_s"]
    midpoint_target, midpoint_rate = target.execution._hermite(
        seed["current_coordinate470"],
        seed["current_coordinate_rate470_per_s"],
        endpoint,
        endpoint_rate,
        target.manifest.SEGMENT_SECONDS,
        0.5,
    )
    assert target._history_replay(
        seed, endpoint, endpoint_rate, midpoint_target, midpoint_rate
    )


def test_physical_status_is_fail_closed() -> None:
    assert target.PHYSICAL_FAILURE_CLASSIFICATION != target.PASS_CLASSIFICATION
    assert target.NUMERICAL_FAILURE_CLASSIFICATION != target.PASS_CLASSIFICATION
