from __future__ import annotations

import run_causal_inner_constrained_lyapunov_reduction_manifest_wp10c9d6c7c3b5c4f25x as f25x


def test_parent_pass_and_inputs_are_locked():
    summary, hashes = f25x._validate_parent()
    assert summary["passed"]
    assert summary["effective_real_rank"] == 28
    assert "summary.json" in hashes


def test_architecture_preserves_exact_unstable_and_conservative_rows():
    contract = f25x._contract()
    architecture = contract["exact_architecture"]
    assert architecture["unstable_dynamics"].endswith("without_reduction")
    assert architecture["required_identity"].startswith("first_162")
    assert architecture["hidden_trial_constraint"] == "C_Z_equals_zero"
    assert architecture["face_flux_single_valued_before_conservative_divergence"]


def test_candidate_dimensions_respect_R320_and_truth_work_is_forbidden():
    contract = f25x._contract()
    assert contract["dimension_budget"]["online_dimensions"] == [302, 310, 318, 320]
    assert max(contract["dimension_budget"]["online_dimensions"]) == 320
    budget = contract["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0


def test_stability_conservation_and_transfer_gates_are_jointly_binding():
    gates = f25x._contract()["binding_gates"]
    assert gates["reduced_Lyapunov_identity_relative_defect_max"] == 1.0e-8
    assert gates["conservative_test_identity_defect_max"] == 5.0e-9
    assert gates["complete_nonstable_eigenvalue_count_equal"] == 28
    assert gates["resolved_self_energy"]["RMS_normalized_dynamic_transfer_relative_error_max"] == 0.10
    assert gates["conservative_face_flux"]["RMS_normalized_dynamic_transfer_relative_error_max"] == 0.10
