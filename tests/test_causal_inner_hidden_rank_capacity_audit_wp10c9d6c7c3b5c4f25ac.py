from __future__ import annotations

import numpy as np

import run_causal_inner_hidden_rank_capacity_audit_wp10c9d6c7c3b5c4f25ac as f25ac


def test_manifest_is_locked_and_model_promotion_is_forbidden(monkeypatch):
    for name, value in f25ac.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25ac._validate_manifest(require_clean=False)
    assert frozen["contract"]["execution_budget"]["allowed_reduced_model_promotions"] == 0
    assert frozen["contract"]["exact_lower_bound"]["target_hidden_order"] == 130


def test_eckart_young_tail_is_exact_for_diagonal_singular_values():
    singular = np.asarray(((4.0, 3.0, 2.0, 1.0),))
    dynamic_norm = np.asarray((np.sqrt(30.0),))
    total_norm = np.asarray((2.0 * np.sqrt(30.0),))
    dynamic, total = f25ac._tail_relative_errors(
        singular, dynamic_norm, total_norm, order=2
    )
    np.testing.assert_allclose(dynamic, np.sqrt(5.0 / 30.0))
    np.testing.assert_allclose(total, 0.5 * np.sqrt(5.0 / 30.0))


def test_order_at_or_above_matrix_rank_has_zero_lower_bound():
    singular = np.asarray(((2.0, 1.0), (3.0, 2.0)))
    dynamic, total = f25ac._tail_relative_errors(
        singular, np.ones(2), np.ones(2), order=2
    )
    np.testing.assert_array_equal(dynamic, np.zeros(2))
    np.testing.assert_array_equal(total, np.zeros(2))


def test_error_metrics_match_binding_frequency_aggregation():
    metrics = f25ac._error_metrics(
        np.asarray((0.1, 0.2)), np.asarray((0.05, 0.1))
    )
    assert metrics["maximum_normalized_dynamic_transfer_relative_error"] == 0.2
    assert metrics["DC_normalized_dynamic_transfer_relative_error"] == 0.1
    np.testing.assert_allclose(
        metrics["RMS_normalized_total_transfer_relative_error"],
        np.sqrt((0.05**2 + 0.1**2) / 2.0),
    )
