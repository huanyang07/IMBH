from __future__ import annotations

import numpy as np

import run_causal_inner_constrained_lyapunov_reduction_audit_wp10c9d6c7c3b5c4f25y as f25y


def test_manifest_is_locked_and_truth_work_is_forbidden(monkeypatch):
    for name, value in f25y.THREAD_ENVIRONMENT.items(): monkeypatch.setenv(name, value)
    frozen = f25y._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0


def test_P_minimum_lift_makes_conservative_test_rows_exact():
    generator = np.diag((-1.0, -2.0, -3.0, -4.0))
    restriction = np.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)))
    output = np.asarray(((0.0, 0.0, 1.0, 1.0),))
    system, metrics = f25y._exact_stable_system(
        generator, output, restriction,
        np.empty((4, 0)), np.empty((0, 4)), np.empty((0, 0)),
    )
    assert metrics["conservative_lift_identity_defect"] < 1.0e-12
    assert metrics["full_conservative_test_identity_defect"] < 1.0e-12
    assert metrics["full_trial_test_biorthogonality_defect"] < 1.0e-12
    assert metrics["full_stable_spectral_abscissa_per_second"] < 0.0
    assert system["hidden_basis"].shape == (4, 2)


def test_constrained_candidate_is_Lyapunov_stable_and_conservative():
    generator = np.diag((-1.0, -2.0, -3.0, -4.0))
    restriction = np.asarray(((1.0, 0.0, 0.0, 0.0),))
    output = np.asarray(((0.0, 1.0, 1.0, 1.0),))
    system, _ = f25y._exact_stable_system(
        generator, output, restriction,
        np.empty((4, 0)), np.empty((0, 4)), np.empty((0, 0)),
    )
    hidden = system["hidden_basis"][:, :2]
    hidden = f25y._P_orthonormalize(hidden, system["certificate"])
    trial = np.hstack((system["conservative_lift"], hidden))
    gram = trial.T @ system["certificate"] @ trial
    test = system["certificate"] @ trial @ np.linalg.inv(gram)
    reduced = test.T @ system["stable_operator"] @ trial
    residual = gram @ reduced + reduced.T @ gram + trial.T @ trial
    assert np.max(np.abs(test[:, :1] - system["conservative_map"].T)) < 1.0e-12
    assert np.linalg.norm(residual) < 1.0e-12
    assert np.max(np.real(np.linalg.eigvals(reduced))) < 0.0


def test_pole_defect_detects_count_or_location_changes():
    reference = np.asarray((1.0 + 2.0j, 1.0 - 2.0j))
    assert f25y._pole_defect(reference, reference.copy()) == 0.0
    assert np.isinf(f25y._pole_defect(reference, reference[:1]))
    assert f25y._pole_defect(reference, reference + 1.0) > 0.0
