from __future__ import annotations

import numpy as np

import run_causal_inner_stable_parametric_online_audit_wp10c9d6c7c3b5c4f25ai as f25ai


def test_manifest_is_locked(monkeypatch):
    for name, value in f25ai.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25ai._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25ai.WORK_PACKAGE


def test_orthogonal_procrustes_recovers_a_rotation():
    angle = 0.37
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    source = np.asarray(((1.0, 0.2), (0.3, 1.4), (-0.4, 0.8)))
    target = source @ rotation
    recovered = f25ai._orthogonal_procrustes(source, target)
    np.testing.assert_allclose(recovered, rotation, atol=1.0e-12)
    np.testing.assert_allclose(recovered.T @ recovered, np.eye(2), atol=1.0e-12)


def test_convex_descriptor_grid_preserves_strict_stability():
    G0 = np.diag((1.0, 2.0))
    G1 = np.diag((3.0, 1.5))
    K0 = np.asarray(((-2.0, 0.3), (-0.3, -1.0)))
    K1 = np.asarray(((-1.0, -0.2), (0.2, -3.0)))
    A0 = np.linalg.solve(G0, K0)
    A1 = np.linalg.solve(G1, K1)
    metrics, arrays = f25ai._descriptor_grid(
        A0, G0, A1, G1, np.linspace(0.0, 1.0, 11)
    )
    assert metrics["minimum_metric_eigenvalue"] > 0.0
    assert metrics["maximum_symmetric_dissipation_eigenvalue"] < 0.0
    assert metrics["maximum_spectral_abscissa_per_second"] < 0.0
    assert np.all(arrays["metric_minimum_eigenvalue"] > 0.0)


def test_unstable_grid_records_fast_positive_bundle():
    U0 = np.diag((2.0, 4.0))
    U1 = np.diag((3.0, 5.0))
    metrics, arrays = f25ai._unstable_grid(U0, U1, np.linspace(0.0, 1.0, 5))
    assert metrics["minimum_positive_real_part_count"] == 2
    assert metrics["minimum_real_part_per_second"] == 2.0
    assert metrics["maximum_real_part_per_second"] == 5.0
    assert np.all(arrays["positive_real_part_count"] == 2)
