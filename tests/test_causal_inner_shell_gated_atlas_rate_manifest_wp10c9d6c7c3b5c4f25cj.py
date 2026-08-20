from __future__ import annotations

import run_causal_inner_shell_gated_atlas_rate_manifest_wp10c9d6c7c3b5c4f25cj as f25cj


def test_parent_authorizes_full_mixed_rate_manifest():
    frozen = f25cj._validate_parent(require_clean=False)
    assert frozen["summary"]["classification"] == f25cj.parent.FULL_CLASSIFICATION
    assert frozen["summary"]["authorized_next"] == (
        "definitions_only_full_mixed_holdout_exact_rate_manifest"
    )
    assert frozen["arrays"]["candidate_primitive_states"].shape == (8, 112, 5)


def test_contract_hash_locks_model_before_truth():
    contract = f25cj._contract()
    assert contract["candidate_database"]["count"] == 8
    assert contract["candidate_database"]["independent_of_extension_fit"]
    assert not contract["candidate_database"]["coefficient_refit_after_truth"]
    assert contract["model_contract"]["decoder_and_rate_coefficients_hash_locked_before_truth"]


def test_exact_and_independent_model_gates_are_binding():
    contract = f25cj._contract()
    exact = contract["binding_exact_rate_gates"]
    model = contract["binding_independent_model_gates"]
    assert exact["completed_nonbase_rate_evaluations_equal"] == 8
    assert exact["maximum_raw_Schur_condition_number"] == 1.0e6
    assert model["maximum_full_state_rate_relative_error"] == 0.15
    assert model["median_full_state_rate_relative_error"] == 0.075
    assert model["maximum_a28_rate_relative_error"] == 0.15
    assert model["radial_sign_disagreement_count_equal"] == 0


def test_no_trajectory_or_cycle_is_authorized():
    boundaries = f25cj._contract()["authorization_boundaries"]
    assert boundaries["new_truth_calls_during_manifest"] == 0
    assert boundaries["new_generator_assemblies"] == 0
    assert boundaries["new_nonlinear_roots"] == 0
    assert boundaries["propagated_states"] == 0
    assert not boundaries["trajectory_authorized"]
    assert not boundaries["predictive_cycle_authorized"]
    assert not boundaries["reduced_slow_evolution_authorized"]
