from __future__ import annotations

import numpy as np

import run_causal_inner_high_order_hermite_audit_wp10c9d6c7c3b5c4f25ag as f25ag


def test_manifest_is_locked_and_validation_is_forbidden(monkeypatch):
    for name, value in f25ag.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25ag._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25ag.WORK_PACKAGE
    split = frozen["contract"]["frequency_split"]
    assert split["validation_information_may_influence_basis"] is False


def test_refined_frequency_grids_are_disjoint_except_for_DC():
    parent = np.concatenate(([0.0], np.geomspace(1.0, 8.0, 32)))
    training, validation = f25ag._refined_frequency_grids(parent)
    assert training.size == 129
    assert validation.size == 129
    assert training[0] == validation[0] == 0.0
    assert np.intersect1d(training[1:], validation[1:]).size == 0
    assert np.all(np.diff(training) > 0.0)
    assert np.all(np.diff(validation) > 0.0)


def test_first_interval_is_linear_and_positive_intervals_are_geometric():
    parent = np.concatenate(([0.0], np.geomspace(2.0, 16.0, 32)))
    training, validation = f25ag._refined_frequency_grids(parent)
    np.testing.assert_allclose(training[:5], (0.0, 0.5, 1.0, 1.5, 2.0))
    np.testing.assert_allclose(validation[:5], (0.0, 0.25, 0.75, 1.25, 1.75))
    left, right = parent[1], parent[2]
    np.testing.assert_allclose(training[5], left * (right / left) ** 0.25)
    np.testing.assert_allclose(validation[5], left * (right / left) ** 0.125)


def test_candidate_orders_preserve_the_frozen_online_budget():
    frozen = f25ag._validate_manifest(require_clean=False)
    assert frozen["contract"]["dimension_budget"]["online_dimensions"] == [
        470,
        478,
        486,
        490,
        494,
        510,
    ]
