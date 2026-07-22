from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    causal_cumulative_trapezoid,
    causal_refined_spread_upper_bound,
    causal_transport_rank_audit,
)


def test_cumulative_trapezoid_integrates_linear_vector_history() -> None:
    times = np.asarray([0.0, 0.25, 1.0])
    values = np.column_stack((2.0 * times + 1.0, -times))
    result = causal_cumulative_trapezoid(times, values)

    np.testing.assert_allclose(
        result,
        np.column_stack((times**2 + times, -0.5 * times**2)),
        rtol=0.0,
        atol=2.0e-16,
    )


def test_transport_rank_audit_accepts_one_common_hidden_amplitude() -> None:
    amplitudes = np.asarray([1.0, 0.5, -0.25, 0.125])
    direction = np.asarray([0.2, 0.8, -0.3])
    audit = causal_transport_rank_audit(amplitudes[:, None] * direction)

    assert audit.passed
    assert audit.second_to_first_ratio < 1.0e-15
    assert audit.third_to_first_ratio < 1.0e-15
    assert abs(np.dot(audit.dominant_direction, direction)) == pytest.approx(
        np.linalg.norm(direction),
    )


def test_transport_rank_audit_rejects_independent_transport_modes() -> None:
    values = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )
    audit = causal_transport_rank_audit(values)

    assert not audit.passed
    assert audit.second_to_first_ratio == pytest.approx(1.0)


def test_refined_spread_bound_includes_full_coarse_fine_difference() -> None:
    coarse = np.asarray([0.4, 0.08])
    fine = np.asarray([0.3, 0.09])
    uncertainty, upper = causal_refined_spread_upper_bound(coarse, fine)

    np.testing.assert_allclose(uncertainty, [0.1, 0.01])
    np.testing.assert_allclose(upper, [0.4, 0.1])


@pytest.mark.parametrize(
    "times,values",
    (
        ([0.0, 0.0], [1.0, 2.0]),
        ([0.0, 1.0], [1.0]),
    ),
)
def test_cumulative_trapezoid_rejects_invalid_inputs(times, values) -> None:
    with pytest.raises(ValueError):
        causal_cumulative_trapezoid(np.asarray(times), np.asarray(values))
