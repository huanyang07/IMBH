from __future__ import annotations

import numpy as np

import run_causal_inner_adaptive_metric_chart_continuation_execution_wp10c9d6c7c3b5c4f25fip as target


def test_manifest_authorizes_repeated_adaptive_execution() -> None:
    lock = target._validate_manifest(require_clean=False)
    assert lock["summary"]["classification"] == target.manifest.CLASSIFICATION
    assert lock["contract"]["scope"]["maximum_accepted_segments"] == 8
    assert lock["contract"]["adaptive_policy"][
        "physically_admissible_chart_failure_halves_span"
    ]


def test_policy_grows_only_after_blind_fourth_accept() -> None:
    policy = target._policy()
    transition = target.transition_after_attempt(
        policy=policy,
        span_seconds=1.0e-3,
        tentative_segment_number=96,
        accepted=True,
        physical_failure=False,
        accepted_since_growth=3,
    )
    assert transition["next_span_seconds"] == 2.0e-3


def test_rejected_record_does_not_mutate_accepted_history() -> None:
    progress = target._initial_progress()
    coordinate = progress["current_coordinate"].copy()
    metrics = {
        "accepted": False,
        "next_span_seconds": 5.0e-4,
        "accepted_since_growth_after": 0,
        "stop_reason": None,
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
    np.testing.assert_array_equal(result["current_coordinate"], coordinate)
    assert result["attempts"] == 1
    assert result["accepted_segments_new"] == 0
    assert result["next_span"] == 5.0e-4


def test_checkpoint_roundtrip_preserves_adaptive_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(target, "SCRATCH_DIRECTORY", tmp_path)
    progress = target._initial_progress()
    directory = tmp_path / "attempt_0000"
    directory.mkdir()
    assert target._write_checkpoint(directory, progress)
    replay = target._progress_from_checkpoint(
        target._load_npz(directory / "accepted_checkpoint.npz")
    )
    assert replay["next_span"] == progress["next_span"]
    assert replay["accepted_since_growth"] == progress["accepted_since_growth"]
    np.testing.assert_array_equal(replay["current_state"], progress["current_state"])
