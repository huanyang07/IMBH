from __future__ import annotations

import numpy as np

import run_causal_inner_unstable_exact_conservative_fiber_audit_wp10c9d6c7c3b5c4f25u as f25u


def test_manifest_is_locked_and_truth_work_is_forbidden(monkeypatch):
    for name, value in f25u.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25u._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0


def test_real_nonnormal_spectral_fiber_is_exact_and_stable_after_deflation():
    diagonal = np.diag((2.0, 0.5, -1.0, -3.0))
    transform = np.asarray(
        ((1.0, 2.0, 0.0, 0.5), (0.0, 1.0, 1.0, 0.0),
         (0.0, 0.0, 1.0, 1.0), (0.2, 0.0, 0.0, 1.0))
    )
    generator = transform @ diagonal @ np.linalg.inv(transform)
    fiber, metrics = f25u._spectral_fiber(generator, -1.0e-8, 2)
    assert metrics["full_nonstable_eigenvalue_count"] == 2
    assert metrics["stable_complement_nonstable_eigenvalue_count"] == 0
    assert metrics["biorthogonality_defect"] < 1.0e-11
    assert metrics["spectral_projector_commutator_relative_defect"] < 1.0e-11
    assert metrics["right_invariance_relative_defect"] < 1.0e-11
    assert metrics["left_invariance_relative_defect"] < 1.0e-11
    assert np.max(np.real(np.linalg.eigvals(fiber["stable_operator"]))) < 0.0


def test_conservative_compatibility_captures_residual_nonstable_content():
    generator = np.diag((2.0, -1.0, -2.0, -3.0))
    fiber, _ = f25u._spectral_fiber(generator, -1.0e-8, 1)
    restriction = np.asarray(((0.0, 1.0, 0.0, 0.0),))
    lifting = restriction.T
    _, metrics = f25u._conservative_compatibility(
        fiber, restriction, lifting
    )
    assert metrics["R32_stable_coordinate_rank"] == 1
    assert metrics["nonstable_residual_rank"] == 1
    assert metrics["exact_unstable_augmented_dimension"] == 2
    assert metrics["augmented_nonstable_capture_relative_defect"] < 1.0e-12


def test_cross_anchor_alignment_preserves_biorthogonality():
    first = np.eye(4)[:, :2]
    angle = 0.1
    rotation = np.asarray(
        ((np.cos(angle), 0.0), (0.0, np.cos(angle)),
         (np.sin(angle), 0.0), (0.0, np.sin(angle)))
    )
    fibers = {
        "primary": {"right_basis": first, "left_dual_transpose": first.T},
        "heldout": {"right_basis": rotation, "left_dual_transpose": rotation.T},
    }
    metrics, _ = f25u._cross_anchor_metrics(fibers)
    assert metrics["right_principal_cosine_min"] > 0.99
    assert metrics["aligned_heldout_biorthogonality_defect"] < 1.0e-12
