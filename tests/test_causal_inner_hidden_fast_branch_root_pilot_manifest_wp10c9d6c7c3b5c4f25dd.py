from __future__ import annotations

import numpy as np

import run_causal_inner_hidden_fast_branch_root_pilot_manifest_wp10c9d6c7c3b5c4f25dd as f25dd


def test_intrinsic_fiber_geometry_is_square_and_exact() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(f25dd.MACRO_DIMENSION, f25dd.CHART_DIMENSION))
    q, _ = np.linalg.qr(raw.T, mode="complete")
    row_basis = q[:, : f25dd.MACRO_DIMENSION]
    hidden = q[:, f25dd.MACRO_DIMENSION :]
    restriction = row_basis.T
    lifting = row_basis
    macro = rng.normal(size=f25dd.MACRO_DIMENSION)
    z = rng.normal(size=f25dd.HIDDEN_DIMENSION)
    y = lifting @ macro + hidden @ z
    assert restriction.shape == (82, 470)
    assert hidden.shape == (470, 388)
    assert np.allclose(restriction @ y, macro)
    assert np.allclose(hidden.T @ (y - lifting @ macro), z)


def test_contract_separates_candidate_and_binding_truth() -> None:
    contract = f25dd._contract()
    objects = contract["mathematical_objects"]
    hierarchy = contract["truth_hierarchy"]
    execution = contract["prospective_execution"]
    assert objects["no_artificial_82_channel_physical_reaction"]
    assert hierarchy["decoder_normal_component_is_binding"]
    assert not hierarchy["old_forward_patch_may_seed_this_root"]
    assert not hierarchy["raw_decoder_macro_mismatch_may_be_ignored"]
    assert execution["sealed_candidate_truth_calls_equal"] == 0
    assert execution["budgets"]["new_intrinsic_hidden_roots_equal"] == 0
    assert execution["budgets"]["new_exact_fixed_Q_rate_evaluations_equal"] == 0
    assert contract["future_root_contract_frozen_but_not_authorized"][
        "coordinate_Hessian_term_required"
    ]


def test_decision_contract_never_labels_one_root_cold_or_hot() -> None:
    decision = f25dd._contract()["decision"]
    assert not decision["stable_root_pass"]["cold_or_hot_label_assigned"]
    assert not decision["sealed_16ms_opened_by_any_outcome"]
    assert "transition" in decision["converged_but_unstable"]["classification"]


def test_checks_fail_closed() -> None:
    metrics = {
        "restriction_rank": 82,
        "restriction_lifting_identity_infinity": 0.0,
        "hidden_basis_annihilation_infinity": 0.0,
        "hidden_basis_orthogonality_infinity": 0.0,
        "maximum_candidate_fiber_reconstruction_relative_defect": 0.0,
        "primary_decoder_relative_error": 0.03,
        "sealed_decoder_relative_error": 0.04,
        "decoded_physical": {
            "all_selected_decoded_states_physically_admissible": True,
            "raw_decoder_is_exact_coordinate_chart": False,
        },
        "exact_geometric_chart_required": True,
        "primary_forward_patch_weight": 0.0,
        "sealed_forward_patch_weight": 0.0,
        "primary_and_sealed_labels": "unclassified",
    }
    assert all(f25dd._checks(metrics).values())
    metrics["primary_forward_patch_weight"] = 1.0
    assert not f25dd._checks(metrics)["old_invalid_field_not_reused"]


def test_canonical_manifest_if_present() -> None:
    if not f25dd.CANONICAL_DIRECTORY.exists():
        return
    f25dd._checksums(f25dd.CANONICAL_DIRECTORY)
    summary = f25dd._read(f25dd.CANONICAL_DIRECTORY / "summary.json")
    contract = f25dd._read(
        f25dd.CANONICAL_DIRECTORY / "branch_root_pilot_contract.json"
    )
    metrics = f25dd._read(
        f25dd.CANONICAL_DIRECTORY / "fiber_geometry_metrics.json"
    )
    assert summary["passed"]
    assert summary["classification"] == f25dd.CLASSIFICATION
    assert summary["authorized_next"] == f25dd.AUTHORIZED_NEXT
    assert not summary["branch_root_execution_authorized"]
    assert summary["exact_geometric_chart_preflight_authorized"]
    assert not summary["sealed_16ms_execution_authorized"]
    assert all(metrics["checks"].values())
    assert metrics["metrics"]["hidden_dimension"] == 388
    assert contract["authorization_boundaries"]["this_package_definitions_only"]
