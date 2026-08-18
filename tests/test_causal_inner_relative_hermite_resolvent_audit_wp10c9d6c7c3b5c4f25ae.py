from __future__ import annotations

import numpy as np

import run_causal_inner_relative_hermite_resolvent_audit_wp10c9d6c7c3b5c4f25ae as f25ae


def test_manifest_is_locked_and_heldout_midpoints_are_forbidden(monkeypatch):
    for name, value in f25ae.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25ae._validate_manifest(require_clean=False)
    design = frozen["contract"]["relative_Hermite_resolvent_POD"]
    assert design["heldout_information_may_influence_basis"] is False
    assert frozen["contract"]["execution_budget"]["allowed_new_full_560_direction_generator_assemblies"] == 0


def test_frequency_half_widths_use_one_sided_and_nearest_intervals():
    widths = f25ae._frequency_half_widths(np.asarray((0.0, 2.0, 6.0, 14.0)))
    np.testing.assert_allclose(widths, (1.0, 1.0, 2.0, 4.0))


def test_relative_Hermite_basis_is_orthonormal_and_solves_derivatives():
    operator = np.diag((-1.0, -2.0, -3.0, -4.0))
    forcing = np.asarray(((1.0,), (2.0,), (1.0,), (0.5,)))
    observation = np.asarray(((1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 1.0)))
    vectors, eigenvalues, metrics = f25ae._relative_hermite_basis(
        operator, forcing, observation, np.asarray((0.0, 1.0, 3.0)), 1
    )
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(4), atol=1.0e-12)
    assert np.all(np.diff(eigenvalues) <= 0.0)
    assert metrics["maximum_snapshot_solve_relative_residual"] < 1.0e-12
    assert metrics["heldout_midpoint_responses_used"] is False
    assert metrics["shared_DC_training_control_used"] is True


def test_relative_Hermite_basis_rejects_nonincreasing_frequencies():
    with np.testing.assert_raises(ValueError):
        f25ae._frequency_half_widths(np.asarray((0.0, 1.0, 1.0)))
