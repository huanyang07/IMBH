from __future__ import annotations

import numpy as np

import run_causal_inner_metric_chart_short_suffix_execution_wp10c9d6c7c3b5c4f25fih as target


def test_manifest_authorizes_fixed_short_suffix() -> None:
    lock = target._validate_manifest(require_clean=False)
    assert lock["summary"]["classification"] == target.manifest.CLASSIFICATION
    assert lock["contract"]["suffix"]["new_segments"] == 4
    assert lock["contract"]["scope"]["new_exact_free_field_calls"] == 5


def test_blind_midpoint_is_only_on_fourth_suffix_segment() -> None:
    assert [
        target._blind_required(value) for value in (73, 74, 75, 76)
    ] == [False, False, False, True]


def test_rejected_record_does_not_advance_progress() -> None:
    progress = target._initial_progress()
    arrays = {
        "endpoint_coordinate470": np.ones(470),
        "endpoint_primitive_state": np.ones((112, 5)),
        "endpoint_coordinate_rate470_per_s": np.ones(470),
        "endpoint_metric_transform470x470": np.eye(470),
        "endpoint_metric_augmented560x560": np.eye(560),
        "endpoint_gauge_basis560x90": np.zeros((560, 90)),
    }
    result = target._advance(
        progress,
        {"accepted": False, "span_seconds": 2.5e-4, "elapsed_seconds_after": 0.0},
        arrays,
    )
    assert result is progress


def test_checkpoint_conversion_is_lossless() -> None:
    progress = target._initial_progress()
    arrays = target._checkpoint_arrays(progress)
    replay = target._progress_from_checkpoint(arrays)
    assert replay["elapsed_seconds"] == progress["elapsed_seconds"]
    assert replay["accepted_segments_total"] == progress["accepted_segments_total"]
    np.testing.assert_array_equal(replay["current_coordinate"], progress["current_coordinate"])
    np.testing.assert_array_equal(replay["metric_transform"], progress["metric_transform"])
