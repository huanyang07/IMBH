from __future__ import annotations

import run_causal_inner_adaptive_metric_chart_cycle_readiness_manifest_wp10c9d6c7c3b5c4f25fiq as target


def test_parent_continuation_is_binding_and_supported() -> None:
    lock = target._validate_parent(require_clean=False)
    assert lock["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert lock["metrics"]["gate_values"]["retryable_chart_failures"] == 2


def test_cycle_readiness_seed_is_exact_terminal_history() -> None:
    seed = target._seed()
    assert float(seed["elapsed_seconds"]) == target.INITIAL_ELAPSED_SECONDS
    assert int(seed["accepted_segments_total"]) == 100
    assert int(seed["accepted_since_growth"]) == 2
    assert float(seed["next_span_seconds"]) == 5.0e-4


def test_turning_forecast_is_consistent_but_nonbinding() -> None:
    forecast = target._geometry_forecast()
    assert not forecast["cycle_observed"]
    assert not forecast["section_negative_observed"]
    assert forecast["terminal_section_velocity_per_second"] > 0.0
    zeros = [
        value["forecast_zero_velocity_time_seconds"]
        for value in forecast["linear_zero_velocity_forecasts"].values()
    ]
    assert all(0.23 < value < 0.25 for value in zeros)
    assert forecast["forecast_spread_seconds"] < 5.0e-3


def test_cost_and_scope_are_prospectively_bounded() -> None:
    cost = target._cost_projection()
    contract = target._contract()
    assert cost["cost_gate_passed"]
    assert cost["reserved_projected_wall_hours"] <= 8.0
    assert not cost["complete_cycle_runtime_identifiable"]
    assert contract["scope"]["maximum_accepted_segments"] == 32
    assert contract["scope"]["initial_span_new_horizon_seconds"] == 0.016
    assert contract["adaptive_policy"]["minimum_segment_seconds"] == 1.25e-4
    assert "authorize a complete cycle before a negative section is observed" in contract[
        "forbidden"
    ]
