from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from imri_qpe.layer3_minidisk_1d.causal_inner_radial_localization import (
    causal_radial_colored_block_jacobians,
    causal_radial_first_consecutive_recovery,
    causal_radial_history_convergence,
    causal_radial_prefix_face_fluxes,
)


def test_colored_block_jacobians_share_residual_evaluations() -> None:
    first = np.asarray(
        [
            [1.0, 0.2, 0.0],
            [-0.1, 0.7, 0.3],
            [0.0, 0.4, 1.2],
        ]
    )
    second = np.asarray(
        [
            [0.5, 0.0, 0.1],
            [0.0, -0.2, 0.0],
            [0.4, 0.0, 0.8],
        ]
    )
    pattern = csc_matrix((first != 0.0) | (second != 0.0))
    blocks = causal_radial_colored_block_jacobians(
        lambda values: {
            "first": first @ np.asarray(values, dtype=float),
            "second": second @ np.asarray(values, dtype=float),
        },
        np.zeros(3),
        pattern,
        finite_difference_step=4.0e-5,
    )
    np.testing.assert_allclose(
        blocks["first"].toarray(),
        first,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        blocks["second"].toarray(),
        second,
        rtol=0.0,
        atol=2.0e-15,
    )


def test_prefix_fluxes_and_three_grid_recovery_metrics() -> None:
    inner = np.asarray([[1.0, 2.0], [1.5, 2.5]])
    transport = np.asarray(
        [
            [[0.1, -0.2], [0.3, 0.4]],
            [[0.2, -0.1], [0.4, 0.5]],
        ]
    )
    faces = causal_radial_prefix_face_fluxes(inner, transport)
    np.testing.assert_allclose(
        faces[:, -1],
        inner + np.sum(transport, axis=1),
        rtol=0.0,
        atol=0.0,
    )
    exact = np.asarray([[1.0, -0.5], [1.2, -0.4]])
    coarse = exact + 4.0e-2
    medium = exact + 1.0e-2
    fine = exact + 2.5e-3
    metrics = causal_radial_history_convergence(
        coarse,
        medium,
        fine,
        minimum_order=0.75,
        maximum_fine_normalized_difference=0.05,
        minimum_fine_signed_cosine=0.90,
    )
    assert metrics.observed_order is not None
    assert metrics.observed_order >= 1.9
    assert metrics.history_cosine > 0.99
    assert metrics.error_cosine > 0.99
    np.testing.assert_allclose(
        metrics.component_history_cosines,
        metrics.component_fine_signed_cosines,
        rtol=0.0,
        atol=0.0,
    )
    assert metrics.history_cosine == metrics.fine_signed_cosine
    assert metrics.passed
    assert causal_radial_first_consecutive_recovery(
        np.asarray([False, True, True, False])
    ) == 1
    assert (
        causal_radial_first_consecutive_recovery(
            np.asarray([False, True, False])
        )
        is None
    )


def test_history_metrics_use_fixed_physical_activity_and_error_cosine() -> None:
    exact = np.asarray([[2.0, 1.0e-14], [2.5, -1.0e-14]])
    coarse = exact + np.asarray([4.0e-2, 4.0e-14])
    medium = exact + np.asarray([1.0e-2, 1.0e-14])
    fine = exact + np.asarray([2.5e-3, 2.5e-15])
    metrics = causal_radial_history_convergence(
        coarse,
        medium,
        fine,
        minimum_order=0.75,
        maximum_fine_normalized_difference=0.05,
        minimum_fine_signed_cosine=0.90,
        minimum_relative_activity=1.0e-8,
        component_reference_scales=np.asarray([2.0, 1.0]),
        minimum_error_cosine=0.90,
    )
    np.testing.assert_array_equal(
        metrics.significant_components,
        np.asarray([True, False]),
    )
    np.testing.assert_allclose(
        metrics.component_normalization_scales,
        np.asarray([2.0, 1.0]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        metrics.component_activity_thresholds,
        np.asarray([2.0e-8, 1.0e-8]),
        rtol=0.0,
        atol=0.0,
    )
    assert metrics.error_cosine > 0.99
    assert metrics.passed
