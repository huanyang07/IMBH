from __future__ import annotations

import run_causal_inner_bordered_branch_homotopy_launch_manifest_wp10c9d6c7c3b5c4f25ar as f25ar


def test_rejected_direct_predictor_parent_is_locked():
    parent = f25ar._validate_parent()
    assert parent["summary"]["passed"]
    assert parent["summary"]["homotopy_required"]
    assert not parent["summary"]["direct_root_attempted"]


def test_bordered_homotopy_has_exact_anchor_and_square_equation_count():
    contract = f25ar._contract()
    homotopy = contract["bordered_homotopy"]
    assert homotopy["unknowns"]["total"] == 722
    assert homotopy["tau_zero_anchor_residual_required_exact"]
    assert homotopy["tau_target"] == 1.0 / 64.0
    assert not homotopy["forward_BDF_history_is_used"]


def test_first_rung_is_trust_limited_and_fail_closed():
    contract = f25ar._contract()
    policy = contract["nonlinear_policy"]
    assert policy["maximum_scaled_anchor_departure"] == 5.0e-3
    assert policy["maximum_new_fixed_Q_rate_evaluations"] == 17
    assert not policy["rejected_candidate_may_define_future_history"]
    assert "Gauss_Newton_KKT_seed" in policy["initial_matrix"]


def test_claim_boundary_does_not_confuse_launch_with_branch_discovery():
    contract = f25ar._contract()
    claims = contract["claim_boundary"]
    assert not claims["tau_one_reached"]
    assert not claims["physical_conditional_branch_found"]
    assert not claims["normal_hyperbolicity_certified"]
    assert not claims["reduced_slow_evolution_authorized"]
