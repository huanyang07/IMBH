from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (
    causal_align_characteristic_field,
    causal_field_history_norm,
    causal_restrict_proper_cell_averages,
    causal_sine_power_window,
    causal_trapezoid_weights,
    causal_windowed_richardson_reference,
)


def test_sine_power_window_has_declared_support_and_endpoint_order() -> None:
    points = np.linspace(-0.25, 1.25, 61)
    values = causal_sine_power_window(
        points,
        lower_log_radius=0.0,
        upper_log_radius=1.0,
        power=4,
    )
    assert np.all(values[(points < 0.0) | (points > 1.0)] == 0.0)
    assert values[np.argmin(np.abs(points - 0.5))] == 1.0
    assert np.all(values >= 0.0)


def test_characteristic_field_alignment_repairs_only_signs() -> None:
    scales = np.asarray((1.0, 2.0, 3.0, 4.0, 5.0))
    base = np.diag(scales)
    field = np.stack((base, -base, base), axis=0)
    aligned = causal_align_characteristic_field(field, scales)
    expected = np.stack((base, base, base), axis=0)
    np.testing.assert_allclose(
        aligned.physical_right_eigenvectors,
        expected,
    )
    assert aligned.minimum_adjacent_overlap == 1.0
    assert aligned.maximum_dimensionless_norm_defect == 0.0


def test_proper_measure_restriction_preserves_integrals() -> None:
    rng = np.random.default_rng(20260729)
    fine = rng.normal(size=(3, 8, 5))
    measures = np.linspace(1.0, 2.0, 8)
    coarse = causal_restrict_proper_cell_averages(
        fine,
        measures,
        refinement_factor=2,
    )
    grouped = measures.reshape(4, 2).sum(axis=1)
    np.testing.assert_allclose(
        np.einsum("tci,c->ti", coarse, grouped),
        np.einsum("tci,c->ti", fine, measures),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_history_norm_uses_fixed_physical_weights() -> None:
    times = np.asarray((0.0, 0.25, 1.0))
    weights = causal_trapezoid_weights(times)
    assert np.isclose(np.sum(weights), 1.0)
    values = np.ones((3, 4, 5))
    norm = causal_field_history_norm(
        values,
        cell_measures=np.ones(4),
        field_scales=np.ones(5),
        time_weights=weights,
    )
    assert np.isclose(norm, np.sqrt(5.0))


def test_richardson_reference_recovers_second_order_sequence() -> None:
    times = np.linspace(0.0, 1.0, 9)
    cells = 6
    shape = (times.size, cells, 5)
    reference = np.ones(shape)
    pattern = np.zeros(shape)
    pattern[..., 0] = (
        np.sin(np.pi * times)[:, None]
        * np.linspace(0.5, 1.5, cells)[None, :]
    )
    coarse = reference + pattern
    medium = reference + 0.25 * pattern
    fine = reference + 0.0625 * pattern
    report = causal_windowed_richardson_reference(
        coarse,
        medium,
        fine,
        times=times,
        coarse_cell_measures=np.linspace(1.0, 2.0, cells),
        field_scales=np.ones(5),
    )
    assert abs(report.observed_order - 2.0) <= 5.0e-15
    assert report.minimum_significant_component_order >= 2.0 - 5.0e-15
    assert report.refinement_error_cosine >= 1.0 - 5.0e-15
    np.testing.assert_allclose(
        report.observed_reference,
        reference,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert report.reference_choice_to_fine_difference_ratio <= 5.0e-15
