from __future__ import annotations

import numpy as np

import run_causal_inner_primary_hidden_fast_root_manifest_wp10c9d6c7c3b5c4f25df as f25df


def test_derivative_certificate_authorizes_only_this_manifest() -> None:
    frozen = f25df._validate_parent(require_clean=False)
    assert frozen["parent_hashes"]


def test_dual_coordinate_geometry_closes_and_naive_projection_is_invalid() -> None:
    arrays, metrics = f25df._dual_geometry()
    assert all(f25df._geometry_checks(metrics).values())
    R = arrays["macro_restriction_R82"]
    L = arrays["macro_lifting_L82"]
    Z = arrays["hidden_basis_Z388"]
    P = arrays["fiber_projection_P470"]
    Q = arrays["hidden_dual_Q388"]
    assert np.allclose(R @ L, np.eye(f25df.MACRO_DIMENSION))
    assert np.allclose(Q @ L, 0.0)
    assert np.allclose(Q @ Z, np.eye(f25df.HIDDEN_DIMENSION))
    assert np.allclose(L @ R + Z @ Q, np.eye(f25df.COORDINATE_DIMENSION))
    assert np.linalg.norm(Z.T @ L, ord=np.inf) > 0.6
    assert np.allclose(P, np.eye(f25df.COORDINATE_DIMENSION) - L @ R)


def test_hidden_rate_and_post_root_architecture_are_mathematically_complete() -> None:
    _, metrics = f25df._dual_geometry()
    contract = f25df._contract(metrics)
    architecture = contract["mathematical_architecture"]
    tangent = contract["complete_hidden_tangent"]
    post = contract["post_root_mathematics"]
    assert architecture["hidden_rate"] == "H(X,z)=Qz_F(y)"
    assert architecture["naive_residual_Z_transpose_F_forbidden"]
    assert tangent["coordinate_Hessian_term_required"]
    assert tangent["physical_reaction_derivative_required"]
    assert post["root_is_not_automatically_an_invariant_slow_manifold"]
    assert post["critical_graph_derivative"] == "Dh=-Azz_inverse_Azx"


def test_anchor_preflight_stops_before_generator_and_root() -> None:
    _, metrics = f25df._dual_geometry()
    contract = f25df._contract(metrics)
    stages = contract["prospective_execution"]["stage_order"]
    assert stages.index(
        "compute_F0_G0_H0_and_apply_initial_hidden_fraction_gate"
    ) < stages.index(
        "assemble_complete_coordinate_and_hidden_tangent_if_preflight_passes"
    )
    assert "stop_without_generator_or_root" in stages[3]
    assert (
        contract["binding_gates"]["initial_hidden_coordinate_rate_fraction_max"]
        == 0.25
    )


def test_truth_and_claim_boundaries_are_narrow() -> None:
    _, metrics = f25df._dual_geometry()
    contract = f25df._contract(metrics)
    budget = contract["prospective_execution"]["budgets"]
    boundary = contract["authorization_boundaries"]
    assert budget["new_exact_fixed_Q_rate_evaluations_max"] == 12
    assert budget["new_complete_physical_generator_assemblies_max"] == 2
    assert budget["new_intrinsic_hidden_roots_max"] == 1
    assert budget["propagated_states_equal"] == 0
    assert budget["sealed_16ms_truth_calls_equal"] == 0
    assert not boundary["branch_root_in_this_package"]
    assert not boundary["physical_microburst_authorized"]
    assert not boundary["reduced_slow_evolution_authorized"]


def test_canonical_manifest_if_present() -> None:
    if not f25df.CANONICAL_DIRECTORY.exists():
        return
    f25df._checksums(f25df.CANONICAL_DIRECTORY)
    summary = f25df._read(f25df.CANONICAL_DIRECTORY / "summary.json")
    payload = f25df._read(
        f25df.CANONICAL_DIRECTORY / "dual_hidden_geometry_metrics.json"
    )
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["dual_consistent_hidden_residual_frozen"]
    assert summary["naive_Z_transpose_F_residual_rejected"]
    assert summary["authorized_next"] == f25df.AUTHORIZED_NEXT
    assert not summary["branch_root_in_this_package"]
    assert not summary["sealed_16ms_opened"]
    assert all(payload["checks"].values())
