from __future__ import annotations

import numpy as np

import run_causal_inner_nonlinear_bundle_screen_wp10c9d6c7c3b5c4f25ak as f25ak


def test_manifest_is_locked(monkeypatch):
    for name, value in f25ak.manifest.parent.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25ak._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25ak.WORK_PACKAGE


def test_energy_directions_select_largest_symmetric_growth():
    operator = np.diag((1.0, 3.0, 2.0))
    directions, values = f25ak._energy_directions(operator, 2)
    np.testing.assert_allclose(values, (3.0, 2.0))
    np.testing.assert_allclose(directions.T @ directions, np.eye(2), atol=1.0e-14)


def test_radial_metrics_detect_a_negative_cubic():
    radii = np.asarray(((0.1, 0.2, 0.3),))
    linear = np.asarray((2.0,))
    rates = np.zeros((1, 3, 2, 1))
    for index, radius in enumerate(radii[0]):
        value = radius * (2.0 - 30.0 * radius**2)
        rates[0, index, 0, 0] = -value
        rates[0, index, 1, 0] = value
    metrics, arrays = f25ak._radial_metrics(linear, radii, rates)
    assert metrics["negative_fitted_cubic_count"] == 1
    assert metrics["nonpositive_largest_amplitude_growth_count"] == 1
    assert arrays["fitted_cubic_growth_coefficients"][0] < 0.0


def test_failure_and_hybrid_classifications_are_distinct():
    assert "evaluator_failed" in f25ak.FAIL_CLASSIFICATION
    assert "hybrid_branch_event" in f25ak.HYBRID_CLASSIFICATION
    assert "normal_form" in f25ak.LOCAL_CLASSIFICATION
