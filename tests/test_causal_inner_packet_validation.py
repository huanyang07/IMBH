from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (
    causal_exact_semigroup_integral_history,
    causal_packet_history_metrics,
)


def test_packet_history_metrics_recovers_second_order_sequence() -> None:
    times = np.linspace(0.0, 1.0, 33)
    reference = np.column_stack(
        (np.sin(times), np.cos(2.0 * times), 0.5 + times)
    )
    error = np.column_stack(
        (np.cos(times), np.sin(2.0 * times), times**2)
    )
    report = causal_packet_history_metrics(
        reference + error,
        reference + 0.25 * error,
        reference + 0.0625 * error,
        physical_scales=np.ones(3),
        minimum_rms_order=0.75,
        minimum_maximum_order=0.75,
        minimum_significant_component_order=0.75,
        maximum_fine_normalized_difference=0.5,
        minimum_history_cosine=0.90,
        minimum_refinement_error_cosine=0.90,
    )
    assert report.passed
    assert abs(report.observed_rms_order - 2.0) <= 5.0e-15
    assert abs(report.observed_maximum_order - 2.0) <= 5.0e-15
    np.testing.assert_allclose(report.component_orders, 2.0)
    assert report.refinement_error_cosine >= 1.0 - 5.0e-15


def test_exact_semigroup_integral_handles_multiple_packets() -> None:
    generator = np.diag(np.asarray((-2.0, -0.5, 0.75)))
    initial = np.asarray(
        (
            (1.0, -0.5),
            (0.25, 2.0),
            (-1.5, 0.75),
        )
    )
    times = np.linspace(0.0, 0.8, 17)
    history = np.asarray(
        [
            np.exp(np.diag(generator)[:, None] * time) * initial
            for time in times
        ]
    )
    result = causal_exact_semigroup_integral_history(
        generator,
        history,
        initial,
    )
    expected = np.asarray(
        [
            (
                (np.exp(np.diag(generator)[:, None] * time) - 1.0)
                / np.diag(generator)[:, None]
            )
            * initial
            for time in times
        ]
    )
    np.testing.assert_allclose(
        result.integrated_states,
        expected,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert result.maximum_relative_solve_residual <= 2.0e-16
