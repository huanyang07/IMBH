from __future__ import annotations

import numpy as np

import run_causal_inner_bordered_branch_homotopy_launch_wp10c9d6c7c3b5c4f25as as f25as


def test_frozen_homotopy_launch_manifest_is_locked():
    frozen = f25as._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25as.WORK_PACKAGE


def test_ruiz_equilibrated_solve_preserves_the_linear_solution():
    rng = np.random.default_rng(25)
    matrix = rng.normal(size=(20, 20))
    matrix += np.diag(np.geomspace(1.0e-4, 1.0e4, 20))
    right = rng.normal(size=20)
    solution, metrics = f25as._equilibrated_solve(matrix, right)
    assert np.linalg.norm(matrix @ solution - right) / np.linalg.norm(right) <= 1.0e-10
    assert np.isfinite(metrics["equilibrated_condition_number"])


def test_anchor_is_exact_and_first_tangent_is_inside_frozen_trust_region():
    system = f25as._anchor_system()
    assert np.max(np.abs(system["tau_zero_residual"])) <= 1.0e-12
    assert system["linear_metrics"]["equilibrated_condition_number"] <= 1.0e6
    assert np.max(np.abs(system["predictor_correction"][:560])) <= 5.0e-3
    assert system["linear_metrics"]["relative_linear_residual"] <= 1.0e-10


def test_launch_cannot_claim_a_tau_one_branch():
    contract = f25as.manifest._contract()
    assert contract["bordered_homotopy"]["tau_target"] == 1.0 / 64.0
    assert not contract["claim_boundary"]["tau_one_reached"]
    assert not contract["claim_boundary"]["physical_conditional_branch_found"]
