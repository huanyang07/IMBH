from __future__ import annotations

import numpy as np

import run_causal_inner_departure28_dual_polynomial_diagnosis_wp10c9d6c7c3b5c4f25bu as f25bu


def test_parent_rejection_is_rate_model_only():
    frozen = f25bu._validate_parent(require_clean=False)
    assert frozen["summary"]["truth_database_passed"]
    assert not frozen["summary"]["independent_model_validation_passed"]
    assert sorted(
        name
        for name, passed in frozen["metrics"]["model_checks"].items()
        if not passed
    ) == [
        "holdout_maximum_full_departure_rate_relative_error",
        "holdout_maximum_nonlinear_departure_rate_relative_error",
    ]


def test_revealed_database_uses_all_160_high_radius_direction_pairs():
    database = f25bu._revealed_database()
    assert database["directions"].shape == (160, 28)
    assert database["quadratic_targets"].shape == (160, 28)
    assert database["cubic_targets"].shape == (160, 28)
    assert database["signed_departure_coordinates"].shape == (320, 28)
    assert np.array_equal(
        np.unique(database["signed_source_codes"], return_counts=True)[1],
        np.array([112, 176, 32]),
    )


def test_dual_kernels_enforce_required_parity():
    rng = np.random.default_rng(97)
    left = rng.normal(size=(7, 28))
    right = rng.normal(size=(9, 28))
    assert np.array_equal(
        f25bu._kernel(-left, right, f25bu.EVEN_KERNEL_POWER),
        f25bu._kernel(left, right, f25bu.EVEN_KERNEL_POWER),
    )
    assert np.array_equal(
        f25bu._kernel(-left, right, f25bu.ODD_KERNEL_POWER),
        -f25bu._kernel(left, right, f25bu.ODD_KERNEL_POWER),
    )


def test_fast_loo_formula_matches_brute_pair_removal():
    rng = np.random.default_rng(101)
    directions = rng.normal(size=(9, 5))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    coordinates = np.repeat(directions, 2, axis=0)
    coordinates[0::2] *= -0.01
    coordinates[1::2] *= 0.01
    targets = rng.normal(size=(9, 3))
    fast, _metrics = f25bu._loo_predictions(
        directions,
        coordinates,
        targets,
        power=2,
        weight_exponent=1.0,
        regularization=2.0e-3,
    )
    brute = np.empty_like(fast)
    norms = np.linalg.norm(targets, axis=1)
    weights = (np.median(norms) / norms) ** 1.0
    for pair in range(9):
        keep = np.arange(9) != pair
        system = f25bu._kernel(directions[keep], directions[keep], 2)
        system += 2.0e-3 * np.diag(1.0 / weights[keep])
        coefficients = np.linalg.solve(system, targets[keep])
        unit = coordinates[2 * pair : 2 * pair + 2]
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        brute[2 * pair : 2 * pair + 2] = (
            f25bu._kernel(unit, directions[keep], 2) @ coefficients
        )
    assert np.allclose(fast, brute, rtol=1.0e-10, atol=1.0e-10)


def test_selected_architecture_passes_revealed_and_loo_diagnostic_gates():
    diagnosis, arrays = f25bu._diagnose()
    assert all(diagnosis["checks"].values())
    metrics = diagnosis["metrics"]
    retrospective = metrics["retrospective_validation"]
    loo = metrics["leave_one_direction_out"]
    assert retrospective["maximum_nonlinear_departure_rate_relative_error"] < 0.25
    assert retrospective["maximum_full_departure_rate_relative_error"] < 0.05
    assert loo["maximum_nonlinear_departure_rate_relative_error"] < 0.25
    assert loo["maximum_full_departure_rate_relative_error"] < 0.05
    assert loo["even_fit"]["condition_number"] < 1.0e8
    assert loo["odd_fit"]["condition_number"] < 1.0e7
    assert arrays["all_revealed_even_coefficients"].shape == (160, 28)
    assert arrays["all_revealed_odd_coefficients"].shape == (160, 28)
    assert arrays["frozen_rank4_curvature_decoder_coefficients"].shape == (120, 4)
    assert metrics["dynamic_state_dimension"] == 470
    assert not metrics["dynamic_curvature_augmentation"]
