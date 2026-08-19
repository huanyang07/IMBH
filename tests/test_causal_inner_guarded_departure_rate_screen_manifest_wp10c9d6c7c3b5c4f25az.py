from __future__ import annotations

import run_causal_inner_guarded_departure_rate_screen_manifest_wp10c9d6c7c3b5c4f25az as f25az


def test_exact_geometric_chart_parent_is_locked():
    parent = f25az._validate_parent()
    assert parent["summary"]["passed"]
    assert parent["summary"]["completed_candidate_count"] == 48
    assert parent["summary"]["nonbase_continuous_rate_evaluations"] == 0


def test_rate_screen_uses_all_certified_states_without_propagation():
    truth = f25az._contract()["truth_evaluation"]
    assert truth["nonbase_continuous_rate_evaluations"] == 48
    assert truth["new_complete_generator_assemblies"] == 0
    assert truth["new_nonlinear_roots"] == 0
    assert truth["propagated_states"] == 0
    assert truth["save_physical_reaction_action"]


def test_small_amplitude_limit_uses_the_complete_saved_generator():
    audit = f25az._contract()["linear_limit_audit"]
    assert audit["smallest_component_bound"] == 2.5e-4
    assert audit["reference_increment"] == (
        "complete_generator_times_actual_scaled_delta"
    )


def test_signal_classifier_does_not_force_branch_or_saturation():
    classifier = f25az._contract()["nonlinear_signal_classifier"]
    assert classifier["radial_saturation_is_diagnostic_not_binding"]
    assert classifier["equilibrium_branch_selection_is_not_an_outcome"]
    claims = f25az._contract()["claim_boundary"]
    assert not claims["48_axial_samples_are_a_full_28D_closure_database"]
    assert not claims["predictive_cycle_authorized"]
