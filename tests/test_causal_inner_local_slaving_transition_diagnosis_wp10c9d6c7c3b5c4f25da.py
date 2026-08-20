from __future__ import annotations

import numpy as np

import run_causal_inner_local_slaving_transition_diagnosis_wp10c9d6c7c3b5c4f25da as f25da


def test_pairwise_q5_restriction_and_lifting_are_conservative() -> None:
    scales = np.arange(1.0, 161.0)
    arrays, metrics = f25da._conservative_operators(scales)
    restriction = arrays["macro_restriction"]
    lifting = arrays["constraint_compatible_piecewise_constant_lifting"]
    np.testing.assert_allclose(
        restriction @ lifting,
        np.eye(f25da.MACRO_DIMENSION),
        atol=2.0e-16,
        rtol=0.0,
    )
    assert metrics["storage_restriction_rank"] == 80
    assert metrics["total_restriction_rank"] == 82
    assert metrics["global_M_J_E_restriction_relative_defect"] < 2.0e-16
    assert metrics["global_M_J_E_lifting_relative_defect"] < 2.0e-16


def test_full_slaving_rejection_is_a_binding_gate() -> None:
    conservative = {
        "storage_restriction_rank": 80,
        "total_restriction_rank": 82,
        "restriction_lifting_identity_infinity_defect": 0.0,
        "global_M_J_E_restriction_relative_defect": 0.0,
        "global_M_J_E_lifting_relative_defect": 0.0,
    }
    tangent = {
        "maximum_Jacobian_step_ladder_relative_defect": 1.0e-6,
        "nonstable_active_dimension": 1,
        "all_active_slaving_is_stable": False,
        "nonstable_only_promotion_gap_ratio": 3.0,
        "selected_fast_gap_ratio": 20.0,
        "selected_fast_block_spectral_abscissa_per_second": -1.0,
        "ordered_Schur_invariance_relative_defect": 0.0,
    }
    seed = {
        "macro_seed_effective_rank_at_1e_8": 0,
        "maximum_seed_macro_rate_relative_error": 0.01,
        "maximum_seed_transition_rate_relative_error": 0.01,
    }
    checks = f25da._checks(conservative, tangent, seed)
    assert all(checks.values())
    tangent["all_active_slaving_is_stable"] = True
    assert not f25da._checks(conservative, tangent, seed)["full_slaving_rejected"]


def test_revised_architecture_separates_branch_and_transition_layers() -> None:
    tangent = {
        "fast_affine_equilibrium_offset_in_normalized_active_coordinates": 2.0,
        "full_slaving_maximum_spectral_abscissa_per_second": 1.0,
        "nonstable_only_promotion_gap_ratio": 3.0,
        "selected_fast_gap_ratio": 20.0,
    }
    seed = {"maximum_validated_active_radius": 1.0}
    contract = f25da._revised_architecture(tangent, seed)
    assert not contract["diagnosis"]["validated_forward_patch_is_slow_graph"]
    assert contract["online_branch_layer"]["online_fast_microsteps"] == 0
    assert (
        contract["offline_transition_layer"]["online_use"]
        == "interpolated_conservative_jump_map_not_transition_ODE"
    )
    assert contract["next_definitions_only_package"]["work_package"] == f25da.AUTHORIZED_NEXT


def test_canonical_diagnosis_if_present() -> None:
    if not f25da.CANONICAL_DIRECTORY.exists():
        return
    f25da._checksums(f25da.CANONICAL_DIRECTORY)
    summary = f25da._read(f25da.CANONICAL_DIRECTORY / "summary.json")
    metrics = f25da._read(f25da.CANONICAL_DIRECTORY / "diagnostic_metrics.json")
    architecture = f25da._read(
        f25da.CANONICAL_DIRECTORY / "revised_architecture.json"
    )
    assert summary["passed"]
    assert summary["classification"] == f25da.CLASSIFICATION
    assert summary["authorized_next"] == f25da.AUTHORIZED_NEXT
    assert summary["conservative_U80_plus_a2_geometry_passed"]
    assert summary["all_active_slaving_rejected"]
    assert summary["forward_patch_is_transition_layer_seed"]
    assert not summary["branch_memory_screen_executed"]
    assert not summary["online_reduced_solver_implementation_authorized"]
    assert all(metrics["checks"].values())
    tangent = metrics["active_tangent"]
    assert tangent["full_slaving_maximum_spectral_abscissa_per_second"] > 0.0
    assert tangent["nonstable_only_promotion_gap_ratio"] < f25da.SPECTRAL_GAP_GATE
    assert tangent["selected_fast_gap_ratio"] >= f25da.SPECTRAL_GAP_GATE
    assert metrics["seed"]["macro_seed_effective_rank_at_1e_8"] == 0
    assert not architecture["diagnosis"]["validated_forward_patch_is_slow_graph"]
