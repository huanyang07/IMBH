from __future__ import annotations

import run_causal_inner_transition_hidden_tangent_manifest_wp10c9d6c7c3b5c4f25dj as f25dj


def test_common_basis_authorizes_only_a_tangent_manifest() -> None:
    frozen = f25dj._validate_parent(require_clean=False)
    assert frozen["parent_hashes"]
    assert "common_transition_hidden_basis" in frozen["parent_classification"]
    assert frozen["saved_generator_complete_JVP_relative_defect"] <= 1.0e-10


def test_contract_uses_complete_coordinate_tangent() -> None:
    contract = f25dj._contract()
    operator = contract["exact_local_operator"]
    audit = contract["coordinate_Hessian_audit"]
    assert "J470_A560_W560" in operator["coordinate_field_tangent"]
    assert operator["coordinate_Hessian_term_required"]
    assert audit["new_fixed_Q_rate_calls"] == 0
    assert audit["new_complete_generator_assemblies"] == 0
    assert len(audit["dimensionless_step_ladder"]) == 3


def test_rejected_old_lifting_is_not_reused() -> None:
    contract = f25dj._contract()
    inputs = contract["input_separation"]
    assert not inputs["old_rejected_resolved_lifting_reused"]
    assert not inputs["old_rejected_transfer_model_reused"]
    assert inputs["full_y470_offline_reference_and_fallback_preserved"]


def test_rank_adaptive_fallback_is_bounded_and_fail_fast() -> None:
    contract = f25dj._contract()
    fallback = contract["rank_adaptive_fallback"]
    budget = contract["execution_budget"]
    assert fallback["candidate_hidden_ranks"][0] == 8
    assert fallback["candidate_hidden_ranks"][-1] == 128
    assert fallback["maximum_rank"] == 128
    assert budget["new_exact_fixed_Q_rate_evaluations_equal"] == 0
    assert budget["new_complete_generator_assemblies_equal"] == 0
    assert budget["propagated_states_equal"] == 0
    assert budget["sealed_16ms_truth_calls_equal"] == 0


def test_tangent_spectrum_is_not_misclassified_as_branch_stability() -> None:
    contract = f25dj._contract()
    boundary = contract["authorization_boundaries"]
    assert (
        contract["exact_local_operator"]["transition_spectrum_role"]
        == "diagnostic_only_not_branch_stability"
    )
    assert not boundary["branch_root_authorized"]
    assert not boundary["transition_trajectory_authorized"]
    assert not boundary["online_solver_authorized"]
    assert not boundary["reduced_slow_evolution_authorized"]


def test_canonical_manifest_if_present() -> None:
    if not f25dj.CANONICAL_DIRECTORY.exists():
        return
    f25dj._checksums(f25dj.CANONICAL_DIRECTORY)
    summary = f25dj._read(f25dj.CANONICAL_DIRECTORY / "summary.json")
    contract = f25dj._read(
        f25dj.CANONICAL_DIRECTORY / "transition_hidden_tangent_contract.json"
    )
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["authorized_next"] == f25dj.AUTHORIZED_NEXT
    assert summary["saved_complete_generator_reuse_frozen"]
    assert not summary["old_rejected_resolved_lifting_reused"]
    assert not summary["transition_tangent_executed"]
    assert summary["new_exact_fixed_Q_rate_calls"] == 0
    assert summary["new_complete_generator_assemblies"] == 0
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert not contract["authorization_boundaries"][
        "this_package_executes_the_tangent_diagnostic"
    ]
