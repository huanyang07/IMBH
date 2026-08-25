from __future__ import annotations

import run_causal_inner_adaptive_metric_chart_cycle_readiness_reforecast_manifest_wp10c9d6c7c3b5c4f25fis as target


def test_parent_open_transient_is_binding_and_supported() -> None:
    lock = target._validate_parent(require_clean=False)
    assert lock["summary"]["classification"] == target.parent.OPEN_CLASSIFICATION
    assert lock["metrics"]["gate_values"]["accepted_segments"] == 32
    assert lock["metrics"]["gate_values"]["retryable_chart_failures"] == 7


def test_seed_preserves_terminal_history_and_freezes_quarter_ms_span() -> None:
    seed = target._seed()
    assert float(seed["elapsed_seconds"]) == target.INITIAL_ELAPSED_SECONDS
    assert int(seed["accepted_segments_total"]) == 132
    assert float(seed["previous_span_seconds"]) == 2.5e-4
    assert float(seed["next_span_seconds"]) == 2.5e-4
    assert int(seed["accepted_since_growth"]) == 0


def test_orientation_reforecast_exposes_model_disagreement_without_inference() -> None:
    forecast = target._geometry_reforecast()
    assert forecast["sample_count"] == 40
    assert not forecast["section_turning_point_observed"]
    assert not forecast["section_negative_observed"]
    assert forecast["section_velocity_strictly_decreasing"]
    assert 0.06 < forecast["terminal_orientation_cosine"] < 0.07
    assert 86.0 < forecast["terminal_orientation_angle_degrees"] < 87.0
    assert forecast["consecutive_acceleration_relaxation_intervals"] >= 8
    raw = forecast["raw_velocity_forecast_range_seconds"]
    orientation = forecast["orientation_forecast_range_seconds"]
    assert 0.20 < raw[0] < raw[1] < 0.24
    assert 0.15 < orientation[0] < orientation[1] < 0.18
    assert (
        forecast["orientation_forecast_spread_seconds"]
        < forecast["raw_velocity_forecast_spread_seconds"]
    )


def test_cost_scope_and_authorization_remain_bounded() -> None:
    cost = target._cost_projection()
    contract = target._contract()
    assert cost["cost_gate_passed"]
    assert cost["maximum_projected_wall_hours"] <= 8.0
    assert not cost["complete_cycle_runtime_identifiable"]
    assert contract["scope"]["maximum_accepted_segments"] == 48
    assert contract["scope"]["nominal_new_horizon_seconds"] == 0.012
    assert contract["adaptive_policy"]["maximum_segment_seconds"] == 2.5e-4
    assert contract["adaptive_policy"]["minimum_segment_seconds"] == 1.25e-4
    assert not contract["forecast_discrimination"][
        "orientation_extrapolation_binding"
    ]
    assert "infer a turn from either forecast" in contract["forbidden"]
    assert "authorize reduced slow evolution" in contract["forbidden"]
