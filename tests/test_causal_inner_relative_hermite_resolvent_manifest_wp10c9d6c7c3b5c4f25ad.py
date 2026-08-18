from __future__ import annotations

import run_causal_inner_relative_hermite_resolvent_manifest_wp10c9d6c7c3b5c4f25ad as f25ad


def test_capacity_parent_pass_is_locked():
    summary, hashes = f25ad._validate_parent()
    assert summary["passed"]
    assert summary["target_safety_passed"]
    assert not summary["coherent_dynamic_realizability_certified"]
    assert "pointwise_rank_bounds.npz" in hashes


def test_basis_uses_training_only_relative_Hermite_snapshots():
    design = f25ad._contract()["relative_Hermite_resolvent_POD"]
    assert design["training_frequencies"].startswith("unchanged_33")
    assert "never_used_in_basis" in design["heldout_frequencies"]
    assert design["heldout_information_may_influence_basis"] is False
    assert set(design["snapshot_groups"]) == {"Xw", "local_interval_scale_times_dXw"}


def test_stability_pair_and_R320_budget_are_binding():
    contract = f25ad._contract()
    architecture = contract["exact_architecture"]
    assert architecture["test"].endswith("Chat_transpose_Z")
    assert "strictly_negative" in architecture["stability_identity"]
    assert contract["dimension_budget"]["online_dimensions"] == [302, 310, 314, 318, 320]
    assert contract["binding_gates"]["complete_nonstable_eigenvalue_count_equal"] == 28


def test_truth_work_remains_forbidden():
    budget = f25ad._contract()["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0
