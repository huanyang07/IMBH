from __future__ import annotations

import json

import numpy as np

import run_causal_inner_authentic_center_exact_rate_training_manifest_wp10c9d6c7c3b5c4f25ct as f25ct


def test_geometry_certificate_authorizes_rate_training_manifest():
    frozen = f25ct._validate_parent(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["classification"] == f25ct.parent.FULL_CLASSIFICATION
    assert frozen["summary"]["authorized_next"] == (
        "definitions_only_authentic_center_exact_rate_training_manifest"
    )
    assert frozen["geometry"]["candidate_primitive_states"].shape == (8, 112, 5)


def test_weighted_affine_fit_recovers_synthetic_maps():
    rng = np.random.default_rng(2519)
    active = rng.normal(size=(12, 3))
    weights = np.linspace(0.5, 1.5, 12)
    affine = rng.normal(size=(4, 9))
    targets = f25ct._affine_features(active) @ affine
    fitted, metrics = f25ct._weighted_affine_fit(
        active, targets, weights, intercept=True, regularization=0.0
    )
    assert metrics["design_rank"] == 4
    assert np.allclose(fitted, affine, atol=1.0e-12)
    linear = rng.normal(size=(3, 9))
    fitted_linear, metrics_linear = f25ct._weighted_affine_fit(
        active, active @ linear, weights, intercept=False, regularization=0.0
    )
    assert metrics_linear["design_rank"] == 3
    assert np.allclose(fitted_linear, linear, atol=1.0e-12)


def test_contract_separates_training_and_holdout_truth():
    contract = f25ct._contract()
    assert contract["execution_order"][-1] == "fit_and_hash_local_field_coefficients"
    assert contract["cost_budget"] == {
        "new_exact_continuous_rate_calls_equal": 5,
        "new_complete_generator_assemblies_equal": 0,
        "new_nonlinear_fixed_Q_roots_equal": 0,
        "propagated_states_equal": 0,
        "holdout_rate_calls_equal": 0,
    }
    assert contract["coefficient_blind_holdout"][
        "rate_truth_forbidden_during_training_package"
    ]
    assert contract["fit_database"]["coefficients_frozen_before_holdout_truth"]
    assert contract["frozen_local_field"][
        "online_state_dependent_coordinate_Jacobian_calls"
    ] == 0


def test_active_coordinates_use_only_center_local_departure_block():
    local = np.zeros((2, 470))
    local[:, :442] = 100.0
    local[0, -28:] = np.arange(28)
    local[1, -28:] = -np.arange(28)
    basis = np.eye(28, 3)
    active = f25ct._active_coordinates(local, basis)
    assert np.array_equal(active[0], -active[1])
    assert np.array_equal(active[0], np.arange(3) / f25ct.ACTIVE_SCALE)


def test_canonical_manifest_if_present():
    if not f25ct.CANONICAL_DIRECTORY.exists():
        return
    f25ct._checksums(f25ct.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25ct.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25ct.CANONICAL_DIRECTORY / "readiness_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    arrays = f25ct._load_npz(
        f25ct.CANONICAL_DIRECTORY / "frozen_rate_training_design.npz"
    )
    assert summary["classification"] == f25ct.CLASSIFICATION
    assert summary["passed"] and summary["definitions_only"]
    assert summary["authorized_next"] == f25ct.AUTHORIZED_NEXT
    assert summary["planned_new_training_exact_rate_calls"] == 5
    assert summary["planned_future_blind_holdout_exact_rate_calls"] == 4
    assert metrics["passed"] and all(metrics["checks"].values())
    assert arrays["decoder_affine_coefficients"].shape == (3, 560)
    assert arrays["training_primitive_states"].shape == (4, 112, 5)
    assert arrays["holdout_primitive_states"].shape == (4, 112, 5)
