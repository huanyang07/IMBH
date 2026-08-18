from __future__ import annotations

import run_causal_inner_intrinsic_constraint_geometry_manifest_wp10c9d6c7c3b5c4f25al as f25al


def test_parent_reaction_chart_rejection_is_locked():
    summary, hashes = f25al._validate_parent()
    assert not summary["passed"]
    assert summary["classification"] == f25al.parent.FAIL_CLASSIFICATION
    assert "nonlinear_screen.npz" in hashes


def test_state_geometry_does_not_reuse_rate_reaction():
    geometry = f25al._contract()["coordinate_geometry"]
    assert geometry["minimum_norm_normal"].startswith("N_equals_Q_transpose")
    assert geometry["physical_reaction_lift_used_for_state_retraction"] is False
    assert geometry["physical_reaction_lift_retained_for_rate_constraint"] is True


def test_nonequilibrium_spectrum_is_not_promoted_to_branch_stability():
    diagnostic = f25al._contract()["saved_generator_diagnostics"]
    assert diagnostic["instantaneous_eigenvalues_are_normal_hyperbolicity_certificate"] is False
    assert diagnostic["new_full_generator_assemblies"] == 0


def test_pass_routes_to_equilibrium_centered_architecture_only():
    decision = f25al._contract()["decision"]
    assert "equilibrium_centered" in decision["pass"]
    assert "constrained_equilibrium" in decision["pass_authorizes_only"]
