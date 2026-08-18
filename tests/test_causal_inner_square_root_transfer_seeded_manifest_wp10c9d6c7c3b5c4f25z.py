from __future__ import annotations

import run_causal_inner_square_root_transfer_seeded_manifest_wp10c9d6c7c3b5c4f25z as f25z


def test_reassessment_preserves_prior_rejection_and_locks_inputs():
    summary, hashes = f25z._validate_parent()
    assert not summary["passed"]
    assert summary["classification"] == "constrained_lyapunov_reduction_numerical_failure_stop"
    assert "decisive_model.npz" in hashes


def test_square_root_architecture_has_no_raw_P_inverse():
    architecture = f25z._contract()["exact_architecture"]
    assert architecture["square_root"].startswith("P_equals_T_transpose_T")
    assert architecture["trial"].startswith("Vhat_equals")
    assert architecture["test"] == "What_equals_horizontal_concatenation_Chat_transpose_Zhat"
    assert not any("P_inverse" in str(value) for value in architecture.values())


def test_transfer_seed_is_projected_and_candidate_dimensions_respect_R320():
    contract = f25z._contract()
    seed = contract["transfer_seed"]
    assert seed["seed_order"] == 130
    assert "kernel_Chat_projection" in seed["mapping"]
    assert contract["dimension_budget"]["online_dimensions"] == [302, 310, 314, 318, 320]
    assert max(contract["dimension_budget"]["online_dimensions"]) == 320


def test_truth_work_is_forbidden_and_joint_gates_are_frozen():
    contract = f25z._contract()
    budget = contract["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0
    gates = contract["binding_gates"]
    assert gates["projected_seed_effective_rank_min"] == 130
    assert gates["complete_nonstable_eigenvalue_count_equal"] == 28
    assert gates["resolved_self_energy"]["RMS_normalized_dynamic_transfer_relative_error_max"] == 0.10
    assert gates["conservative_face_flux"]["RMS_normalized_dynamic_transfer_relative_error_max"] == 0.10
