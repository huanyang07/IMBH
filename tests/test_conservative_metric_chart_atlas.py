from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.conservative_metric_chart_atlas import (
    ConservativeMetricChart,
    block_whitening_transform,
)


def test_block_whitening_and_chart_roundtrip() -> None:
    jacobian = np.asarray([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
    ])
    transform, metrics = block_whitening_transform(jacobian, (2, 1))
    chart = ConservativeMetricChart(np.asarray([1.0, 2.0, 3.0]), transform, (2, 1))
    coordinate = np.asarray([1.5, 1.0, 4.0])
    rate = np.asarray([2.0, -3.0, 4.0])
    np.testing.assert_allclose(chart.decode(chart.encode(coordinate)), coordinate)
    np.testing.assert_allclose(chart.pull_rate(chart.push_rate(rate)), rate)
    assert metrics["metric_jacobian_condition_number"] == 1.0
    assert chart.inverse_closure_defect == 0.0


def test_transform_must_be_block_diagonal_positive_definite() -> None:
    anchor = np.zeros(2)
    with np.testing.assert_raises(ValueError):
        ConservativeMetricChart(anchor, np.asarray([[1.0, 0.1], [0.1, 1.0]]), (1, 1))
    with np.testing.assert_raises(ValueError):
        ConservativeMetricChart(anchor, np.asarray([[1.0, 0.0], [0.0, -1.0]]), (1, 1))
