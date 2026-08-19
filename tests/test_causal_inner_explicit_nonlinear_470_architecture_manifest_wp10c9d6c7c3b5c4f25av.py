from __future__ import annotations

import run_causal_inner_explicit_nonlinear_470_architecture_manifest_wp10c9d6c7c3b5c4f25av as f25av


def test_failed_branch_path_and_prior_470_certificate_are_locked():
    parents = f25av._validate_parents()
    assert not parents["parent_summary"]["passed"]
    assert parents["stable_summary"]["passed"]
    assert parents["stable_summary"]["total_architecture_dimension"] == 470


def test_online_partition_retains_departure_coordinates_explicitly():
    contract = f25av._contract()
    partition = contract["state_partition"]
    assert 162 + 280 + 28 == partition["online_continuous_dimension"] == 470
    assert 470 + 90 == partition["full_dimension"] == 560
    assert contract["online_dynamics"]["discrete_branch_label_required"] is False


def test_architecture_uses_exact_geometric_chart_not_reaction_lift():
    contract = f25av._contract()
    chart = contract["coordinates"]["finite_amplitude_unstable_chart"]
    assert "geometric_Newton_retraction_on_exact_C_phys" in chart
    assert "not_the_rate_reaction_lift" in chart


def test_claim_boundary_requires_nonlinear_closure_before_prediction():
    claims = f25av._contract()["claim_boundary"]
    assert not claims["nonlinear_28D_closure_identified"]
    assert not claims["online_integrator_implemented"]
    assert not claims["predictive_cycle_authorized"]
    assert not claims["reduced_slow_evolution_authorized"]
