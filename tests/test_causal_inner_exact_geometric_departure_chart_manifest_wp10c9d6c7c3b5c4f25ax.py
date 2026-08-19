from __future__ import annotations

import run_causal_inner_exact_geometric_departure_chart_manifest_wp10c9d6c7c3b5c4f25ax as f25ax


def test_parent_architecture_and_failed_reaction_lift_are_locked():
    parents = f25ax._validate_parent()
    assert parents["parent_summary"]["passed"]
    assert parents["parent_summary"]["online_coordinate_rank"] == 470
    assert (
        parents["failed_metrics"]["fail_fast_coordinate_retraction"][
            "failure_kind"
        ]
        == "rate_reaction_is_not_a_geometric_retraction"
    )


def test_candidate_family_is_bounded_and_rate_free():
    contract = f25ax._contract()
    family = contract["candidate_family"]
    assert family["direction_count"] == 8
    assert family["planned_candidates"] == 48
    assert max(family["maximum_component_bounds"]) == 5.0e-3
    gates = contract["binding_preflight_gates"]
    assert gates["nonbase_continuous_rate_evaluations_equal"] == 0
    assert gates["propagated_states_equal"] == 0


def test_retraction_uses_exact_coordinate_normal_not_reaction_lift():
    retraction = f25ax._contract()["exact_geometric_retraction"]
    assert retraction["target"] == "C_phys_x_equals_C_phys_x0"
    assert retraction["state_local_derivative"] == "exact_descriptor_coordinate_Jacobian"
    assert retraction["rate_reaction_lift_used"] is False


def test_pass_authorizes_only_a_definitions_only_rate_database_manifest():
    contract = f25ax._contract()
    assert contract["decision"]["pass_authorizes_only"] == (
        "definitions_only_guarded_nonlinear_28D_rate_database_manifest"
    )
    assert not contract["claim_boundary"]["nonlinear_rate_database_executed"]
    assert not contract["claim_boundary"]["predictive_cycle_authorized"]
