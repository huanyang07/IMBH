from __future__ import annotations

import run_causal_inner_coordinate_hessian_diagnosis_manifest_wp10c9d6c7c3b5c4f25at as f25at


def test_failed_launch_parent_is_locked_without_reclassification():
    parent = f25at._validate_parent()
    assert not parent["summary"]["passed"]
    assert parent["summary"]["rate_evaluations"] == 1


def test_sparse_coloring_matches_the_five_cell_hessian_stencil():
    contract = f25at._contract()
    recovery = contract["sparse_recovery"]
    assert recovery["cell_half_bandwidth"] == 2
    assert recovery["cell_color_count"] == 5
    assert recovery["colored_direction_count"] == 25
    assert recovery["central_coordinate_jacobian_evaluations"] == 50


def test_diagnosis_adds_no_fixed_q_truth_call():
    contract = f25at._contract()
    recovery = contract["sparse_recovery"]
    assert recovery["new_fixed_Q_rate_evaluations"] == 0
    assert recovery["new_complete_fixed_Q_generator_evaluations"] == 0
    assert not contract["claim_boundary"]["failed_homotopy_candidate_reclassified"]


def test_pass_authorizes_only_corrected_launch_manifest():
    contract = f25at._contract()
    assert "corrected_homotopy_launch_manifest_authorized" in contract["decision"]["pass"]
    assert not contract["claim_boundary"]["physical_conditional_branch_found"]
    assert not contract["claim_boundary"]["reduced_slow_evolution_authorized"]
