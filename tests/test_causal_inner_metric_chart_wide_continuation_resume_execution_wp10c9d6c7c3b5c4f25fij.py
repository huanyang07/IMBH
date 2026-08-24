from __future__ import annotations

import numpy as np

import run_causal_inner_metric_chart_wide_continuation_resume_execution_wp10c9d6c7c3b5c4f25fij as target


def test_manifest_authorizes_bounded_wide_resume() -> None:
    lock = target._validate_manifest(require_clean=False)
    assert lock["summary"]["classification"] == target.manifest.CLASSIFICATION
    assert lock["contract"]["scope"]["maximum_accepted_segments"] == 64
    assert lock["contract"]["scope"]["maximum_exact_free_field_calls"] == 88


def test_adaptive_growth_requires_fourth_blind_acceptance() -> None:
    unchanged = target._transition_after_attempt(
        span=5.0e-4,
        accepted=True,
        blind_required=False,
        physical_failure=False,
        chart_failure=False,
        accepted_since_growth=3,
    )
    assert unchanged["next_span"] == 5.0e-4
    grown = target._transition_after_attempt(
        span=5.0e-4,
        accepted=True,
        blind_required=True,
        physical_failure=False,
        chart_failure=False,
        accepted_since_growth=3,
    )
    assert grown["next_span"] == 1.0e-3
    assert grown["accepted_since_growth"] == 0


def test_numerical_rejection_halves_but_physical_failure_stops() -> None:
    numerical = target._transition_after_attempt(
        span=2.0e-3,
        accepted=False,
        blind_required=False,
        physical_failure=False,
        chart_failure=False,
        accepted_since_growth=2,
    )
    assert numerical["next_span"] == 1.0e-3
    assert numerical["stop_classification"] is None
    physical = target._transition_after_attempt(
        span=2.0e-3,
        accepted=False,
        blind_required=False,
        physical_failure=True,
        chart_failure=False,
        accepted_since_growth=2,
    )
    assert physical["stop_classification"] == target.PHYSICAL_FAILURE_CLASSIFICATION


def test_rejected_record_never_changes_accepted_history() -> None:
    progress = target._initial_progress()
    before = progress["current_coordinate"].copy()
    metrics = {
        "accepted": False,
        "next_span_seconds": 2.5e-4,
        "accepted_since_growth_after": 0,
        "seen_negative_after": False,
        "section_after": progress["previous_section"],
        "stop_classification": None,
        "nonclosing_event": None,
        "cycle_event": None,
    }
    arrays = {
        "accepted_coordinate470": np.ones(470),
        "accepted_primitive_state": np.ones((112, 5)),
        "accepted_coordinate_rate470_per_s": np.ones(470),
        "accepted_metric_transform470x470": np.eye(470),
        "accepted_metric_augmented560x560": np.eye(560),
        "accepted_gauge_basis560x90": np.zeros((560, 90)),
    }
    result = target._apply_record(progress, metrics, arrays)
    np.testing.assert_array_equal(result["current_coordinate"], before)
    assert result["accepted_segments_new"] == 0
    assert result["attempts"] == 1


def test_checkpoint_conversion_preserves_adaptive_state() -> None:
    progress = target._initial_progress()
    arrays = target._checkpoint_arrays(progress)
    metadata = {"cycle_event": None, "nonclosing_events": [], "stop_classification": None}
    replay = target._progress_from_checkpoint(arrays, metadata)
    assert replay["next_span"] == progress["next_span"]
    assert replay["accepted_since_growth"] == progress["accepted_since_growth"]
    np.testing.assert_array_equal(replay["metric_transform"], progress["metric_transform"])


def test_well_conditioned_retraction_nonclosure_is_not_chart_failure() -> None:
    metrics = {
        "minimum_reconstruction_factor": 1.0,
        "maximum_height_ratio": 0.1,
        "minimum_scattering_optical_depth": 19.0,
        "maximum_metric_augmented_condition_number": 2.0,
    }
    assert target._retraction_physical_passed(metrics)
    assert not target._retraction_chart_failed(metrics)
    metrics["maximum_metric_augmented_condition_number"] = 11.0
    assert target._retraction_chart_failed(metrics)
