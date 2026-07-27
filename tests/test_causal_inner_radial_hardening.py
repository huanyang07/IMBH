from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from imri_qpe.layer3_minidisk_1d import (
    causal_radial_dense_colored_audit,
    causal_radial_jvp_spatial_attribution,
    causal_radial_jvp_step_sweep,
    causal_radial_one_sided_jvp_sweep,
    causal_radial_partial_dense_central_jacobian,
    causal_radial_project_jvp_actions,
    causal_radial_volume_weighted_scaled_direction,
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


def test_projected_jvp_and_spatial_attribution_localize_one_cell() -> None:
    steps = np.asarray([1.0e-5, 2.0e-5, 4.0e-5])
    matrix_action = np.asarray([1.0, 2.0, 3.0, 4.0])
    direct = np.broadcast_to(matrix_action, (3, 4)).copy()
    direct[1, 2:] += np.asarray([1.0e-4, -2.0e-4])
    direct[2, 2:] += np.asarray([3.0e-4, -6.0e-4])
    projected = causal_radial_project_jvp_actions(
        direct,
        matrix_action,
        steps,
        (0, 1),
        selected_step=4.0e-5,
    )
    assert projected.selected_matrix_relative_defect == 0.0
    np.testing.assert_array_equal(
        projected.adjacent_relative_changes,
        np.zeros(2),
    )
    attribution = causal_radial_jvp_spatial_attribution(
        direct,
        steps,
        n_fields=2,
    )
    np.testing.assert_array_equal(
        attribution.dominant_cells,
        np.asarray([1, 1]),
    )
    np.testing.assert_allclose(
        attribution.cell_squared_fractions[:, 1],
        np.ones(2),
        rtol=0.0,
        atol=0.0,
    )


def test_one_sided_jvp_exposes_quadratic_curvature() -> None:
    sweep = causal_radial_one_sided_jvp_sweep(
        lambda values: np.asarray(
            [
                values[0] + values[0] ** 2,
                2.0 * values[1],
            ]
        ),
        np.zeros(2),
        np.asarray([1.0, 0.5]),
        (1.0e-4, 2.0e-4),
    )
    np.testing.assert_allclose(
        sweep.centered_actions,
        np.broadcast_to(np.asarray([1.0, 1.0]), (2, 2)),
        rtol=0.0,
        atol=2.0e-13,
    )
    assert (
        sweep.one_sided_relative_mismatches[1]
        > sweep.one_sided_relative_mismatches[0]
    )


def test_volume_weighted_direction_has_unit_rms() -> None:
    measures = np.asarray([1.0, 2.0, 3.0])
    direction = causal_radial_volume_weighted_scaled_direction(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [-1.0, 1.0],
            ]
        ),
        measures,
    )
    weights = measures / np.sum(measures)
    assert np.isclose(
        np.sqrt(np.sum(weights[:, None] * direction * direction)),
        1.0,
    )
