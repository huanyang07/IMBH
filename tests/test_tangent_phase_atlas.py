from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.tangent_phase_atlas import (
    fit_tangent_phase_chart,
    normalized_metric_tangents,
    rolling_tangent_phase_audit,
)


def _circle_tangents(count: int, *, reverse: bool = False) -> np.ndarray:
    angle = np.linspace(0.1, 1.2, count)
    if reverse:
        angle = angle[::-1]
    radius = 0.6
    return np.column_stack(
        (
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.full(count, np.sqrt(1.0 - radius**2)),
        )
    )


def test_exact_circle_chart_predicts_next_tangent() -> None:
    tangents = _circle_tangents(14)
    chart = fit_tangent_phase_chart(tangents[:12])
    result = chart.evaluate_next(tangents[12])
    assert result["phase_increment"] > 0.0
    assert result["relative_radial_defect"] < 1.0e-12
    assert result["out_of_plane_defect"] < 1.0e-12
    assert result["direction_prediction_defect_radians"] < 1.0e-7
    assert chart.two_plane_energy_fraction > 1.0 - 1.0e-12


def test_reverse_traversal_is_oriented_to_increasing_phase() -> None:
    chart = fit_tangent_phase_chart(_circle_tangents(12, reverse=True))
    assert chart.orientation_sign in (-1, 1)
    assert np.all(np.diff(chart.training_phases) > 0.0)


def test_rolling_audit_uses_only_trailing_samples() -> None:
    metrics, arrays = rolling_tangent_phase_audit(
        _circle_tangents(24), window_size=12
    )
    assert metrics["prediction_count"] == 12
    assert metrics["all_phase_increments_positive"]
    assert metrics["maximum_direction_prediction_defect_radians"] < 1.0e-7
    assert arrays["phase_increment"].shape == (12,)


def test_metric_normalization_is_rowwise_and_fail_closed() -> None:
    rates = np.asarray(((3.0, 0.0), (0.0, 2.0)))
    tangents = normalized_metric_tangents(rates, np.diag((2.0, 0.5)))
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)
    np.testing.assert_allclose(tangents, np.eye(2))
    with pytest.raises(ValueError, match="zero"):
        normalized_metric_tangents(np.zeros((2, 2)), np.eye(2))
