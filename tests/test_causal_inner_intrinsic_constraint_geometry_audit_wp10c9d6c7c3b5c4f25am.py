from __future__ import annotations

import numpy as np

import run_causal_inner_intrinsic_constraint_geometry_audit_wp10c9d6c7c3b5c4f25am as f25am


def test_manifest_is_locked(monkeypatch):
    for name, value in f25am.manifest.parent.manifest.parent.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25am._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25am.WORK_PACKAGE


def test_orthogonal_geometry_is_a_true_projector():
    rows = np.asarray(((1.0, 2.0, 0.0), (0.0, 1.0, 1.0)))
    geometry = f25am._orthogonal_geometry(rows)
    projector = geometry["projector"]
    tangent = geometry["tangent"]
    np.testing.assert_allclose(projector @ projector, projector, atol=1.0e-14)
    np.testing.assert_allclose(projector.T, projector, atol=1.0e-14)
    np.testing.assert_allclose(rows @ tangent, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(tangent.T @ tangent, np.eye(1), atol=1.0e-14)


def test_minimum_norm_normal_closes_constraint_identity():
    rows = np.asarray(((2.0, 0.0, 1.0), (0.0, 3.0, 1.0)))
    normal = f25am._orthogonal_geometry(rows)["normal"]
    np.testing.assert_allclose(rows @ normal, np.eye(2), atol=1.0e-14)


def test_pass_classification_is_equilibrium_centered():
    assert "equilibrium_centered" in f25am.PASS_CLASSIFICATION
    assert "geometry_failed" in f25am.FAIL_CLASSIFICATION
