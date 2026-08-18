from __future__ import annotations

import run_causal_inner_high_order_hermite_manifest_wp10c9d6c7c3b5c4f25af as f25af


def test_rejected_R320_parent_is_preserved():
    summary, hashes = f25af._validate_parent()
    assert not summary["passed"]
    assert summary["best_hidden_order"] == 120
    assert summary["authorized_next"] is None
    assert "candidate_errors.npz" in hashes


def test_unseen_eighth_grid_is_prospective_and_training_only():
    split = f25af._contract()["frequency_split"]
    assert split["training_interval_fractions"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert split["validation_interval_fractions"] == [0.125, 0.375, 0.625, 0.875]
    assert split["training_frequency_count"] == 129
    assert split["validation_frequency_count_including_shared_DC"] == 129
    assert split["non_DC_validation_frequencies_evaluated_before_freeze"] is False
    assert split["validation_information_may_influence_basis"] is False


def test_dimension_and_stability_contract_is_binding():
    contract = f25af._contract()
    assert contract["dimension_budget"]["online_dimensions"] == [470, 478, 486, 490, 494, 510]
    assert contract["dimension_budget"]["maximum_online_dimension"] == 510
    assert contract["exact_architecture"]["test"].endswith("Chat_transpose_Z")
    assert "strictly_negative" in contract["exact_architecture"]["stability_identity"]
    assert contract["binding_gates"]["complete_nonstable_eigenvalue_count_equal"] == 28


def test_truth_work_and_online_implementation_remain_forbidden():
    contract = f25af._contract()
    budget = contract["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0
    assert contract["claim_boundary"]["online_integrator_implementation_authorized"] is False
