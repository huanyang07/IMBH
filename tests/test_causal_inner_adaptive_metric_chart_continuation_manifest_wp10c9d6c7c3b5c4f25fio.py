from __future__ import annotations

import run_causal_inner_adaptive_metric_chart_continuation_manifest_wp10c9d6c7c3b5c4f25fio as target


def test_parent_is_accepted_radius_recovery() -> None:
    lock = target._validate_parent(require_clean=False)
    assert lock["classification"] == target.parent.PASS_CLASSIFICATION


def test_adaptive_policy_retries_chart_failure_and_stops_physics() -> None:
    policy = target._contract()["adaptive_policy"]
    assert policy["physically_admissible_chart_failure_halves_span"]
    assert policy["endpoint_or_midpoint_numerical_failure_halves_span"]
    assert policy["physical_failure_stops"]
    assert policy["rejected_candidate_is_never_propagated"]


def test_scope_is_short_cost_bounded_tranche() -> None:
    contract = target._contract()
    assert contract["scope"]["maximum_accepted_segments"] == 8
    assert contract["scope"]["maximum_attempted_segments"] == 12
    assert contract["scope"]["maximum_exact_free_field_calls"] == 12
    assert contract["scope"]["maximum_wall_hours"] == 3.0
    assert target._cost_projection()["cost_gate_passed"]


def test_seed_starts_at_recovery_checkpoint() -> None:
    seed = target._seed()
    assert float(seed["elapsed_seconds"]) == target.INITIAL_ELAPSED_SECONDS
    assert int(seed["accepted_segments_total"]) == 92
    assert float(seed["next_span_seconds"]) == 0.001
    assert int(seed["accepted_since_growth"]) == 0


def test_pass_does_not_directly_authorize_complete_cycle() -> None:
    contract = target._contract()
    assert "readiness manifest" in contract["decision"]["pass_authorizes"]
    assert any("complete-cycle" in value for value in contract["forbidden"])
