from __future__ import annotations

import numpy as np

import run_causal_inner_tangent_phase_lap_recurrence_stage1_execution_wp10c9d6c7c3b5c4f25fix as target


def test_manifest_authorizes_only_stage1() -> None:
    lock = target._validate_manifest(require_clean=False)
    stage = lock["contract"]["staged_scope"]["stage1"]
    assert lock["summary"]["classification"] == target.manifest.CLASSIFICATION
    assert lock["summary"]["authorized_next"] == target.WORK_PACKAGE
    assert stage["accepted_segments"] == 48
    assert stage["maximum_exact_free_field_calls"] == 54
    assert not lock["summary"]["complete_cycle_execution_authorized"]


def test_initial_progress_replays_manifest_seed_without_loss() -> None:
    seed = target._seed()
    progress = target._initial_progress()
    np.testing.assert_array_equal(
        progress["current_coordinate"], seed["current_coordinate470"]
    )
    np.testing.assert_array_equal(
        progress["current_rate"], seed["current_coordinate_rate470_per_s"]
    )
    assert progress["elapsed_seconds"] == target.INITIAL_ELAPSED_SECONDS
    assert progress["accepted_segments_new"] == 0


def test_initial_prediction_uses_only_frozen_trailing_history() -> None:
    progress = target._initial_progress()
    metrics, arrays, chart = target._prediction(progress)
    assert metrics["history_samples"] == 12
    assert metrics["frozen_before_exact_endpoint"]
    assert metrics["prior_cumulative_phase_advance_radians"] == 0.0
    assert metrics["training_two_plane_energy_fraction"] >= 0.999
    assert arrays["training_raw_rates470_per_s"].shape == (12, 470)
    assert arrays["predicted_unit_tangent470"].shape == (470,)
    assert chart.predicted_phase_increment > 0.0


def test_registered_return_candidate_requires_phase_section_state_and_orientation(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(target, "SCRATCH_DIRECTORY", tmp_path)
    seed = target._seed()
    progress = target._initial_progress()
    reference = seed["phase_lap_reference_coordinate470"]
    rate = seed["current_coordinate_rate470_per_s"]
    progress["current_coordinate"] = (
        reference - progress["next_span"] * rate
    )
    phase = {"phase_increment": target.manifest.PHASE_LAP_RADIANS}
    result = target._recurrence_gate(
        progress=progress,
        endpoint_coordinate=reference,
        endpoint_rate=rate,
        phase_geometry=phase,
    )
    assert result["registered_section_bracket"]
    assert result["crossing_phase_advance_radians"] == target.manifest.PHASE_LAP_RADIANS
    assert result["crossing_return_distance_over_path_length"] < 1e-15
    assert result["endpoint_metric_tangent_cosine"] > 0.999999999999
    assert result["section_derivative_fraction_of_reference_speed"] > 0.999999
    assert result["coarse_recurrence_candidate"]


def test_prediction_is_persisted_before_exact_attempt(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(target, "SCRATCH_DIRECTORY", tmp_path)

    def fake_exact_attempt(*, progress, inputs, exact_chart):
        del progress, inputs, exact_chart
        assert (tmp_path / "attempt_0000" / "phase_prediction.json").exists()
        assert (tmp_path / "attempt_0000" / "phase_prediction.npz").exists()
        return {"accepted": False}, {}

    monkeypatch.setattr(target, "_ORIGINAL_ENGINE_ATTEMPT", fake_exact_attempt)
    with target._engine_context():
        metrics, _arrays = target._phase_attempt(
            progress=target._initial_progress(), inputs=None, exact_chart=None
        )
    assert metrics["phase_geometry"] is None
    assert metrics["recurrence_geometry"] is None


def test_mocked_accepted_endpoint_passes_phase_and_accumulates(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(target, "SCRATCH_DIRECTORY", tmp_path)

    def fake_exact_attempt(*, progress, inputs, exact_chart):
        del inputs, exact_chart
        frozen = target._load_npz(
            tmp_path / "attempt_0000" / "phase_prediction.npz"
        )
        transform = frozen["phase_observer_metric_transform470x470"]
        speed = float(target._seed()["reference_metric_speed_per_s"])
        endpoint_rate = np.linalg.solve(
            transform, frozen["predicted_unit_tangent470"] * speed
        )
        endpoint = (
            progress["current_coordinate"]
            + progress["next_span"] * endpoint_rate
        )
        return (
            {"accepted": True, "numerical_passed": True, "stop_reason": None},
            {
                "endpoint_coordinate470": endpoint,
                "endpoint_coordinate_rate470_per_s": endpoint_rate,
            },
        )

    monkeypatch.setattr(target, "_ORIGINAL_ENGINE_ATTEMPT", fake_exact_attempt)
    with target._engine_context():
        metrics, _arrays = target._phase_attempt(
            progress=target._initial_progress(), inputs=None, exact_chart=None
        )
    assert metrics["accepted"]
    assert metrics["phase_geometry"]["passed"]
    assert metrics["phase_geometry"]["phase_increment"] > 0.0
    assert metrics["recurrence_geometry"][
        "cumulative_phase_advance_radians"
    ] > 0.0
    assert metrics["recurrence_geometry"]["cumulative_metric_path_length"] > 0.0
    assert not metrics["recurrence_geometry"]["phase_lap_observed"]


def test_failed_phase_candidate_does_not_enter_accumulation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(target, "SCRATCH_DIRECTORY", tmp_path)
    assert target._accepted_attempts() == []
    assert target._prior_accumulation() == {
        "cumulative_phase_advance_radians": 0.0,
        "cumulative_metric_path_length": 0.0,
        "registered_section_value": 0.0,
    }


def test_engine_context_is_isolated() -> None:
    original_manifest = target.engine.manifest
    original_attempt = target.engine._attempt
    with target._engine_context():
        assert target.engine.manifest is target
        assert target.engine._attempt is target._phase_attempt
        assert target.engine.SCRATCH_DIRECTORY == target.SCRATCH_DIRECTORY
    assert target.engine.manifest is original_manifest
    assert target.engine._attempt is original_attempt
