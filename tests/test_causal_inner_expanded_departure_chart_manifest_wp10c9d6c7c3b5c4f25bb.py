from __future__ import annotations

import run_causal_inner_expanded_departure_chart_manifest_wp10c9d6c7c3b5c4f25bb as f25bb


def test_unresolved_nonlinear_signal_parent_is_locked():
    parent = f25bb._validate_parent()
    assert parent["summary"]["passed"]
    assert not parent["summary"]["nonlinear_signal_resolved"]
    assert parent["summary"][
        "median_largest_departure_nonlinear_relative_defect"
    ] < 0.1


def test_first_expanded_rung_doubles_bound_and_is_geometry_only():
    contract = f25bb._contract()
    family = contract["candidate_family"]
    assert family["maximum_scaled_component_bound"] == 1.0e-2
    assert family["planned_candidates"] == 16
    gates = contract["binding_preflight_gates"]
    assert gates["nonbase_continuous_rate_evaluations_equal"] == 0
    assert gates["propagated_states_equal"] == 0


def test_expanded_rung_retains_exact_geometric_retraction():
    retraction = f25bb._contract()["exact_geometric_retraction"]
    assert retraction["target"] == "C_phys_x_equals_C_phys_x0"
    assert not retraction["rate_reaction_lift_used"]


def test_pass_authorizes_only_sixteen_rate_screen_manifest():
    assert f25bb._contract()["decision"]["pass_authorizes_only"] == (
        "definitions_only_expanded_amplitude_0p01_sixteen_rate_screen_manifest"
    )
