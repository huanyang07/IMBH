from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.phase_collocation import (
    PiecewisePhaseCollocation,
    PolynomialPhaseSegment,
    direction_cosine,
    gauss_lobatto_nodes,
    relative_vector_defect,
)


def test_cubic_hermite_recovers_value_and_rate() -> None:
    def value(time: float) -> np.ndarray:
        return np.asarray((1.0 + 2.0 * time - time**2 + 0.5 * time**3, time**3))

    def rate(time: float) -> np.ndarray:
        return np.asarray((2.0 - 2.0 * time + 1.5 * time**2, 3.0 * time**2))

    segment = PolynomialPhaseSegment.from_constraints(
        start_time_seconds=0.0,
        end_time_seconds=2.0,
        value_times_seconds=np.asarray((0.0, 2.0)),
        values=np.stack((value(0.0), value(2.0))),
        rate_times_seconds=np.asarray((0.0, 2.0)),
        rates_per_second=np.stack((rate(0.0), rate(2.0))),
    )
    assert np.allclose(segment.value(0.7), value(0.7), atol=2.0e-14)
    assert np.allclose(segment.rate(0.7), rate(0.7), atol=2.0e-14)


def test_piecewise_interfaces_and_lobatto_nodes() -> None:
    left = PolynomialPhaseSegment.from_constraints(
        start_time_seconds=0.0,
        end_time_seconds=1.0,
        value_times_seconds=np.asarray((0.0, 1.0)),
        values=np.asarray(((0.0,), (1.0,))),
        rate_times_seconds=np.empty(0),
        rates_per_second=np.empty((0, 1)),
    )
    right = PolynomialPhaseSegment.from_constraints(
        start_time_seconds=1.0,
        end_time_seconds=2.0,
        value_times_seconds=np.asarray((1.0, 2.0)),
        values=np.asarray(((1.0,), (2.0,))),
        rate_times_seconds=np.empty(0),
        rates_per_second=np.empty((0, 1)),
    )
    atlas = PiecewisePhaseCollocation((left, right))
    assert np.array_equal(atlas.interface_value_defects(), np.zeros(1))
    nodes = gauss_lobatto_nodes(8)
    assert len(nodes) == 8
    assert nodes[0] == 0.0 and nodes[-1] == 1.0
    assert np.all(np.diff(nodes) > 0.0)


def test_vector_metrics() -> None:
    reference = np.asarray((3.0, 4.0))
    assert relative_vector_defect(reference, reference) == 0.0
    assert direction_cosine(reference, 2.0 * reference) == 1.0
