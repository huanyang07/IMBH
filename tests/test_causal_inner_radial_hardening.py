from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from imri_qpe.layer3_minidisk_1d import (
    causal_radial_dense_colored_audit,
    causal_radial_jvp_step_sweep,
    causal_radial_partial_dense_central_jacobian,
)


def test_partial_dense_and_colored_audit_close_for_declared_pattern() -> None:
    matrix = np.asarray(
        [
            [2.0, -0.5, 0.0, 0.0],
            [0.25, 1.5, -0.75, 0.0],
            [0.0, -0.2, 1.0, 0.4],
            [0.0, 0.0, 0.3, 0.8],
        ]
    )

    def residual(values: np.ndarray) -> np.ndarray:
        return matrix @ np.asarray(values, dtype=float)

    columns, dense = causal_radial_partial_dense_central_jacobian(
        residual,
        np.zeros(4),
        range(4),
        finite_difference_step=4.0e-5,
    )
    np.testing.assert_array_equal(columns, np.arange(4))
    np.testing.assert_allclose(dense, matrix, rtol=0.0, atol=2.0e-15)
    audit = causal_radial_dense_colored_audit(
        residual,
        np.zeros(4),
        matrix,
        csc_matrix(matrix != 0.0),
        range(4),
        finite_difference_step=4.0e-5,
    )
    assert audit.maximum_relative_defect <= 2.0e-15
    assert audit.maximum_off_pattern_relative_entry == 0.0


def test_dense_audit_detects_an_undeclared_derivative() -> None:
    matrix = np.eye(4)
    matrix[0, 3] = 0.25
    declared = np.eye(4, dtype=bool)
    audit = causal_radial_dense_colored_audit(
        lambda values: matrix @ np.asarray(values, dtype=float),
        np.zeros(4),
        matrix,
        declared,
        range(4),
        finite_difference_step=4.0e-5,
    )
    assert audit.maximum_relative_defect <= 2.0e-15
    assert audit.maximum_off_pattern_relative_entry == 0.25


def test_jvp_step_sweep_reproduces_a_linear_action() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.2, 0.0],
            [-0.1, 0.7, 0.3],
            [0.0, 0.4, 1.2],
        ]
    )
    direction = np.asarray([0.5, -0.25, 1.0])
    sweep = causal_radial_jvp_step_sweep(
        lambda values: matrix @ np.asarray(values, dtype=float),
        np.zeros(3),
        matrix,
        direction,
        (1.0e-5, 2.0e-5, 4.0e-5, 8.0e-5),
        selected_step=4.0e-5,
    )
    assert sweep.selected_matrix_relative_defect <= 2.0e-15
    assert sweep.minimum_adjacent_relative_change <= 2.0e-15
    np.testing.assert_allclose(
        sweep.direct_actions,
        np.broadcast_to(matrix @ direction, sweep.direct_actions.shape),
        rtol=0.0,
        atol=2.0e-15,
    )
