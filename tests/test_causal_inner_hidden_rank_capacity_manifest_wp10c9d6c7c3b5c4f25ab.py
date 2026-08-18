from __future__ import annotations

import run_causal_inner_hidden_rank_capacity_manifest_wp10c9d6c7c3b5c4f25ab as f25ab


def test_parent_rejection_is_preserved_and_locked():
    summary, hashes = f25ab._validate_parent()
    assert not summary["passed"]
    assert summary["base_architecture_passed"]
    assert summary["authorized_next"] is None
    assert "candidate_errors.npz" in hashes


def test_eckart_young_capacity_bound_and_claim_boundary_are_explicit():
    contract = f25ab._contract()
    bound = contract["exact_lower_bound"]
    assert "rank_of_any_order_r" in bound["pointwise_rank_constraint"]
    assert bound["target_hidden_order"] == 130
    assert bound["interpretation"].startswith("necessary_")
    claims = contract["claim_boundary"]
    assert not claims["coherent_dynamic_realizability_certified"]
    assert not claims["structure_preserving_basis_certified"]


def test_capacity_gate_has_tenfold_margin_and_forbids_truth_work():
    contract = f25ab._contract()
    assert contract["binding_gates"]["capacity_safety_fraction_of_transfer_gate_max"] == 0.10
    budget = contract["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0
    assert budget["allowed_reduced_model_promotions"] == 0
