from __future__ import annotations

import run_causal_inner_expanded_departure_rate_screen_manifest_wp10c9d6c7c3b5c4f25bd as f25bd


def test_expanded_chart_parent_and_prior_screen_are_locked():
    frozen = f25bd._validate_parent()
    assert frozen["chart_summary"]["passed"]
    assert frozen["chart_summary"]["completed_candidate_count"] == 16
    assert frozen["chart_summary"]["maximum_scaled_component_bound"] == 1.0e-2
    assert not frozen["prior_summary"]["nonlinear_signal_resolved"]


def test_screen_uses_exactly_sixteen_rates_without_propagation():
    truth = f25bd._contract()["truth_evaluation"]
    assert truth["nonbase_continuous_rate_evaluations"] == 16
    assert truth["new_complete_generator_assemblies"] == 0
    assert truth["new_nonlinear_roots"] == 0
    assert truth["propagated_states"] == 0
    assert truth["save_physical_reaction_action"]


def test_signal_decision_is_prospective_and_saturation_free():
    contract = f25bd._contract()
    assert contract["nonlinear_signal_classifier"]["resolved_if_at_least"] == 0.10
    assert contract["nonlinear_signal_audit"][
        "radial_saturation_is_diagnostic_not_binding"
    ]
    assert contract["nonlinear_signal_audit"][
        "equilibrium_branch_selection_is_not_an_outcome"
    ]


def test_axial_screen_cannot_claim_a_full_closure():
    claims = f25bd._contract()["claim_boundary"]
    assert not claims["sixteen_axial_samples_are_a_full_28D_closure_database"]
    assert not claims["prior_plus_current_axial_samples_are_a_full_28D_database"]
    assert not claims["predictive_cycle_authorized"]
