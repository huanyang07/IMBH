from __future__ import annotations

import numpy as np

import run_causal_inner_transition_hidden_basis_screen_wp10c9d6c7c3b5c4f25di as f25di


def test_parent_authorizes_only_the_saved_array_screen() -> None:
    frozen = f25di._validate_parent(require_clean=False)
    assert frozen["parent_hashes"]
    assert "rank_adaptive_hidden_impulse_map" in frozen["parent_classification"]


def test_seed_only_basis_passes_coordinate_and_physical_gates() -> None:
    metrics, arrays, checks = f25di._screen()
    assert all(checks.values())
    assert metrics["classification"] == f25di.COMMON_CLASSIFICATION
    assert metrics["selected_basis_source"] == "prior_seed_only"
    assert not metrics["primary_direction_added_as_atlas_center"]
    assert metrics["selected_hidden_rank"] == 8
    assert metrics["minimum_training_hidden_action_energy_capture"] >= 0.99
    assert metrics["current_primary_hidden_action_energy_capture"] >= 0.95
    assert (
        metrics["current_primary_gauge_fixed_physical_action_energy_capture"]
        >= 0.95
    )
    assert arrays["selected_hidden_basis388"].shape == (388, 8)
    assert arrays["selected_coordinate_action_basis470"].shape == (470, 8)
    assert arrays["selected_gauge_fixed_physical_basis560"].shape == (560, 8)


def test_basis_is_dual_consistent_and_macro_annihilating() -> None:
    metrics, arrays, _ = f25di._screen()
    basis = arrays["selected_hidden_basis388"]
    action_basis = arrays["selected_coordinate_action_basis470"]
    restriction = arrays["macro_restriction_R82"]
    assert np.linalg.norm(basis.T @ basis - np.eye(8), ord=np.inf) <= 5.0e-12
    assert np.linalg.norm(restriction @ action_basis, ord=np.inf) <= 5.0e-12
    assert metrics["dual_identity_infinity_defect"] <= 5.0e-12


def test_leave_family_out_capture_is_reported() -> None:
    metrics, _, _ = f25di._screen()
    leave = metrics["leave_family_out"]
    assert leave["prior_revealed_nine"]["heldout_count"] == 9
    assert leave["prior_validation_four"]["heldout_count"] == 4
    assert (
        leave["prior_revealed_nine"][
            "minimum_heldout_hidden_action_energy_capture"
        ]
        > 0.99
    )
    assert (
        leave["prior_validation_four"][
            "minimum_heldout_hidden_action_energy_capture"
        ]
        > 0.99
    )


def test_selected_contract_preserves_authorization_boundary() -> None:
    metrics, _, checks = f25di._screen()
    contract = f25di._selected_contract(metrics, checks)
    interpretation = contract["interpretation"]
    boundary = contract["authorization_boundaries"]
    assert interpretation["common_transition_hidden_basis_candidate_supported"]
    assert interpretation["full470_offline_transition_reference_still_required"]
    assert not interpretation["basis_is_a_certified_transition_dynamics_model"]
    assert not boundary["complete_tangent_executed"]
    assert not boundary["online_solver_authorized"]
    assert not boundary["reduced_slow_evolution_authorized"]


def test_canonical_screen_if_present() -> None:
    if not f25di.CANONICAL_DIRECTORY.exists():
        return
    f25di._checksums(f25di.CANONICAL_DIRECTORY)
    summary = f25di._read(f25di.CANONICAL_DIRECTORY / "summary.json")
    payload = f25di._read(
        f25di.CANONICAL_DIRECTORY / "hidden_basis_screen_metrics.json"
    )
    assert summary["passed"]
    assert summary["saved_arrays_only"]
    assert summary["classification"] == f25di.COMMON_CLASSIFICATION
    assert summary["selected_hidden_rank"] == 8
    assert summary["new_exact_fixed_Q_rate_calls"] == 0
    assert summary["new_complete_generator_assemblies"] == 0
    assert summary["new_nonlinear_roots"] == 0
    assert summary["new_chart_retractions"] == 0
    assert summary["propagated_states"] == 0
    assert not summary["sealed_16ms_opened"]
    assert all(payload["checks"].values())
