from __future__ import annotations

import run_causal_inner_amplitude_0p02_departure_chart_manifest_wp10c9d6c7c3b5c4f25bf as f25bf


def test_unresolved_amplitude_0p01_parent_is_locked():
    frozen = f25bf._validate_parent()
    assert frozen["rate_summary"]["passed"]
    assert not frozen["rate_summary"]["nonlinear_signal_resolved"]
    assert frozen["rate_summary"]["component_bound"] == 1.0e-2


def test_rung_doubles_bound_but_preserves_direction_family():
    family = f25bf._contract()["candidate_family"]
    assert family["maximum_scaled_component_bound"] == 2.0e-2
    assert family["prior_maximum_scaled_component_bound"] == 1.0e-2
    assert family["planned_candidates"] == 16
    assert family["prior_states_are_not_propagated_or_extrapolated"]


def test_geometry_preflight_remains_rate_free_and_fail_closed():
    contract = f25bf._contract()
    gates = contract["binding_preflight_gates"]
    assert gates["nonbase_continuous_rate_evaluations_equal"] == 0
    assert gates["propagated_states_equal"] == 0
    assert not contract["exact_geometric_retraction"]["rate_reaction_lift_used"]
    assert contract["decision"]["pass_authorizes_only"] == (
        "definitions_only_amplitude_0p02_sixteen_rate_screen_manifest"
    )
