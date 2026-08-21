from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.phase_collocation import (
    PiecewisePhaseCollocation,
    PolynomialPhaseSegment,
    direction_cosine,
    gauss_lobatto_nodes,
    lagrange_differentiation_matrix,
    lagrange_integration_matrix,
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


def test_lagrange_matrices_exactly_differentiate_and_integrate_polynomials() -> None:
    nodes = gauss_lobatto_nodes(8)
    differentiation = lagrange_differentiation_matrix(nodes)
    integration = lagrange_integration_matrix(nodes)
    values = 2.0 - 3.0 * nodes + 4.0 * nodes**3 - 0.5 * nodes**6
    derivative = -3.0 + 12.0 * nodes**2 - 3.0 * nodes**5
    antiderivative = 2.0 * nodes - 1.5 * nodes**2 + nodes**4 - nodes**7 / 14.0
    np.testing.assert_allclose(differentiation @ values, derivative, atol=2.0e-12)
    np.testing.assert_allclose(integration @ values, antiderivative, atol=2.0e-13)
    np.testing.assert_array_equal(integration[0], np.zeros(len(nodes)))


def test_lagrange_matrices_reject_duplicate_nodes() -> None:
    duplicate = np.asarray((0.0, 0.5, 0.5, 1.0))
    with pytest.raises(ValueError, match="distinct"):
        lagrange_differentiation_matrix(duplicate)
    with pytest.raises(ValueError, match="distinct"):
        lagrange_integration_matrix(duplicate)
