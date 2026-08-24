from __future__ import annotations

import numpy as np

import run_causal_inner_metric_chart_wide_continuation_resume_manifest_wp10c9d6c7c3b5c4f25fii as target


def test_suffix_authorizes_bounded_resume() -> None:
    lock = target._validate_parent(require_clean=False)
    assert lock["classification"] == target.parent.PASS_CLASSIFICATION
    assert target.parent.AUTHORIZED_NEXT == (
        "WP10c9d6c7c3b5c4f25fii_metric_chart_wide_continuation_resume_manifest"
    )


def test_resume_seed_preserves_complete_original_path_and_terminal_history() -> None:
    seed = target._seed()
    assert seed["trajectory_coordinates"].shape == (141, 470)
    assert seed["trajectory_primitive_states"].shape == (141, 112, 5)
    assert float(seed["elapsed_seconds"]) == target.INITIAL_ELAPSED_SECONDS
    assert int(seed["accepted_segments_total"]) == 76
    assert not bool(seed["seen_negative_section"])


def test_cost_projection_is_bounded_with_rejection_reserve() -> None:
    cost = target._cost_projection()
    assert cost["cost_gate_passed"]
    assert cost["rejection_or_event_call_reserve"] == 8
    assert cost["reserved_projected_wall_hours"] <= 10.0
    assert np.isclose(cost["maximum_no_rejection_horizon_seconds"], 0.118)


def test_contract_keeps_physics_original_and_caps_growth() -> None:
    contract = target._contract()
    assert contract["truth_system"]["all_physics_ledgers_events_and_interpolation_in_original_q"]
    assert contract["adaptive_policy"]["maximum_segment_seconds"] == 2.0e-3
    assert contract["adaptive_policy"]["original_physical_or_metric_chart_failure_stops_immediately"]
    assert contract["scope"]["maximum_accepted_segments"] == 64
    assert contract["scope"]["maximum_exact_free_field_calls"] == 88
