from __future__ import annotations

import numpy as np

import run_causal_inner_conservative_tangent_phase_atlas_holdout_execution_wp10c9d6c7c3b5c4f25fiv as target


def test_manifest_authorizes_only_prospective_phase_holdout() -> None:
    lock = target._validate_manifest(require_clean=False)
    assert lock["summary"]["classification"] == target.manifest.CLASSIFICATION
    assert lock["contract"]["scope"]["accepted_segments"] == 16
    assert lock["contract"]["scope"]["maximum_exact_free_field_calls"] == 20
    assert not lock["summary"]["complete_cycle_execution_authorized"]


def test_initial_progress_replays_terminal_seed_without_loss() -> None:
    seed = target._seed()
    progress = target._initial_progress()
    np.testing.assert_array_equal(
        progress["current_coordinate"], seed["current_coordinate470"]
    )
    np.testing.assert_array_equal(
        progress["metric_transform"], seed["metric_transform470x470"]
    )
    assert progress["elapsed_seconds"] == target.INITIAL_ELAPSED_SECONDS
    assert progress["accepted_segments_new"] == 0


def test_initial_prediction_uses_only_frozen_trailing_history() -> None:
    progress = target._initial_progress()
    metrics, arrays, chart = target._prediction(progress)
    assert metrics["history_samples"] == target.manifest.SELECTED_WINDOW
    assert metrics["frozen_before_exact_holdout"]
    assert metrics["training_two_plane_energy_fraction"] >= 0.999
    assert arrays["training_raw_rates470_per_s"].shape == (12, 470)
    assert arrays["predicted_unit_tangent470"].shape == (470,)
    assert chart.predicted_phase_increment > 0.0


def test_phase_gate_accepts_the_retrospective_next_holdout() -> None:
    diagnostic = target._diagnostic_arrays()
    rates = diagnostic["trajectory_raw_rates470_per_s"]
    transform = diagnostic["terminal_metric_transform470x470"]
    training = rates[-13:-1]
    unit = target.normalized_metric_tangents(training, transform)
    chart = target.fit_tangent_phase_chart(unit)
    progress = target._initial_progress()
    prediction, prediction_arrays, _unused = target._prediction(progress)
    # Replace only the chart-specific arrays so this is a pure gate test.
    prediction["training_two_plane_energy_fraction"] = chart.two_plane_energy_fraction
    prediction["training_relative_radial_rms"] = chart.training_relative_radial_rms
    prediction["predicted_phase_increment"] = chart.predicted_phase_increment
    prediction_arrays["terminal_metric_transform470x470"] = transform
    prediction_arrays["predicted_unit_tangent470"] = chart.predicted_unit_tangent()
    result = target._phase_gate(
        prediction, prediction_arrays, chart, rates[-1]
    )
    assert result["passed"]
    assert result["phase_increment"] > 0.0


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


def test_engine_context_is_isolated() -> None:
    original_manifest = target.engine.manifest
    original_attempt = target.engine._attempt
    with target._engine_context():
        assert target.engine.manifest is target
        assert target.engine._attempt is target._phase_attempt
        assert target.engine.SCRATCH_DIRECTORY == target.SCRATCH_DIRECTORY
    assert target.engine.manifest is original_manifest
    assert target.engine._attempt is original_attempt
