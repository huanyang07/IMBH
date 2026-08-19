from __future__ import annotations

import numpy as np

import run_causal_inner_parity_low_rank_architecture_audit_wp10c9d6c7c3b5c4f25bi as f25bi


def test_frozen_parity_manifest_is_locked():
    frozen = f25bi._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25bi.WORK_PACKAGE
    assert frozen["summary"]["planned_new_truth_rate_evaluations"] == 0


def test_parity_terms_recover_quadratic_and_cubic_scaling():
    metrics = {
        "evaluations": [
            {"direction_index": 0, "amplitude_index": 0, "sign": -1},
            {"direction_index": 0, "amplitude_index": 0, "sign": 1},
        ]
    }
    arrays = {
        "candidate_departure_coordinates": np.array([[-2.0], [2.0]]),
        "departure_linear_references_per_second": np.array([[-2.0], [2.0]]),
        "departure_rate_increments_per_second": np.array([[2.0], [14.0]]),
    }
    old_dimension = f25bi.manifest.ACTIVE_INPUT_DIMENSION
    f25bi.manifest.ACTIVE_INPUT_DIMENSION = 1
    try:
        terms = f25bi._parity_terms(metrics, arrays, 0)
    finally:
        f25bi.manifest.ACTIVE_INPUT_DIMENSION = old_dimension
    assert np.allclose(terms["quadratic_coefficients"], 2.0)
    assert np.allclose(terms["cubic_coefficients"], 0.5)


def test_row_balanced_basis_reports_exact_rank_energy():
    coefficients = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    basis, singular, energy = f25bi._row_balanced_basis(coefficients, 1)
    assert basis.shape == (3, 1)
    assert singular.shape == (2,)
    assert energy == 1.0


def test_diagnostic_claim_boundary_remains_nonpredictive():
    claims = f25bi.manifest._contract()["claim_boundary"]
    assert not claims["thresholds_were_selected_blind_to_existing_results"]
    assert not claims["mixed_direction_coefficients_identified"]
    assert not claims["predictive_cycle_authorized"]
