from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_slow_inexact_newton import (
    bounded_trf_inexact_direction,
    good_broyden_matrix_update,
)


def test_bounded_direction_is_descent_even_when_trust_is_active() -> None:
    matrix = np.diag((1.0, 2.0, 4.0))
    residual = np.asarray((2.0, -1.0, 0.5))
    direction = bounded_trf_inexact_direction(
        matrix, residual, maximum_absolute_step=0.25
    )
    assert direction.maximum_absolute_step <= 0.25 * (1.0 + 1.0e-12)
    assert direction.forcing_two_norm < 1.0
    assert direction.normalized_directional_derivative < 0.0
    assert direction.active_bound_count >= 1


def test_good_broyden_update_enforces_the_new_secant() -> None:
    matrix = np.eye(3)
    step = np.asarray((0.5, -0.25, 0.125))
    change = np.asarray((0.75, 0.5, -0.25))
    updated = good_broyden_matrix_update(matrix, step, change)
    np.testing.assert_allclose(updated @ step, change, rtol=0.0, atol=1.0e-15)


def test_good_broyden_rejects_zero_displacement() -> None:
    with np.testing.assert_raises(ValueError):
        good_broyden_matrix_update(np.eye(2), np.zeros(2), np.ones(2))
