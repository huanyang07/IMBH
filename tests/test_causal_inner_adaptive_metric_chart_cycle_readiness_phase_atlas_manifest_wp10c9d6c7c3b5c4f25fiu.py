from __future__ import annotations

import numpy as np

import run_causal_inner_adaptive_metric_chart_cycle_readiness_phase_atlas_manifest_wp10c9d6c7c3b5c4f25fiu as target


def test_parent_is_open_and_fixed_section_forecasts_are_rejected() -> None:
    lock = target._validate_parent(require_clean=False)
    forecast = lock["metrics"]["gate_values"]["updated_forecast_bundle"]
    assert lock["summary"]["classification"] == target.parent.OPEN_CLASSIFICATION
    assert forecast["raw_velocity_forecast_range_seconds"] is None
    assert forecast["orientation_forecast_range_seconds"] is None
    assert len(forecast["raw_velocity_windows_without_forward_zero"]) == 7


def test_trajectory_is_lossless_and_strictly_ordered() -> None:
    trajectory = target._trajectory()
    assert trajectory["rates470_per_s"].shape == (49, 470)
    assert trajectory["times_seconds"].shape == (49,)
    assert np.all(np.diff(trajectory["times_seconds"]) > 0.0)
    assert trajectory["times_seconds"][-1] == 0.16400000000000012


def test_retrospective_phase_evidence_selects_only_a_prospective_holdout() -> None:
    lock = target._validate_parent(require_clean=False)
    metrics, arrays, definitions = target._evaluate(lock)
    selected = metrics["window_audits"][str(target.SELECTED_WINDOW)]
    assert metrics["classification"] == target.CLASSIFICATION
    assert metrics["passed"]
    assert metrics["selection_is_retrospective"]
    assert not metrics["retrospective_cross_validation_is_binding_execution_evidence"]
    assert selected["prediction_count"] == 37
    assert selected["all_phase_increments_positive"]
    assert selected["maximum_direction_prediction_defect_radians"] < 0.005
    assert arrays["terminal_chart__plane_basis470x2"].shape == (470, 2)
    assert definitions["contract"]["scope"]["accepted_segments"] == 16


def test_architecture_requires_recurrence_beyond_phase_lap() -> None:
    lock = target._validate_parent(require_clean=False)
    metrics, _arrays, definitions = target._evaluate(lock)
    architecture = definitions["architecture"]
    assert metrics["authorized_next"] == target.AUTHORIZED_NEXT
    assert architecture["phase_lap_is_not_a_cycle"]
    assert (
        architecture["cycle_certificate_requires"][
            "state_return_distance_fraction_of_path_length_at_most"
        ]
        == 0.10
    )
    assert "multiple shooting" in architecture[
        "offline_periodic_orbit_if_cycle_is_certified"
    ]["method"]
    assert not metrics["complete_cycle_execution_authorized"]
    assert not metrics["reduced_slow_evolution_authorized"]
