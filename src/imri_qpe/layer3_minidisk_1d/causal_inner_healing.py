"""Numerical utilities for causal coordinate-fiber healing audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalTransportRankAudit:
    """Rank-one diagnostic for a normalized interface-transport history."""

    singular_values: np.ndarray
    second_to_first_ratio: float
    third_to_first_ratio: float
    dominant_direction: np.ndarray
    passed: bool


def causal_cumulative_trapezoid(
    times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Return cumulative trapezoidal integrals on a strictly increasing grid."""

    time = np.asarray(times, dtype=float)
    samples = np.asarray(values, dtype=float)
    if (
        time.ndim != 1
        or time.size < 1
        or samples.shape[0] != time.size
        or np.any(~np.isfinite(time))
        or np.any(~np.isfinite(samples))
        or np.any(np.diff(time) <= 0.0)
    ):
        raise ValueError("cumulative trapezoid inputs are invalid")
    result = np.zeros_like(samples, dtype=float)
    if time.size == 1:
        return result
    increments = (
        0.5
        * np.diff(time).reshape((-1,) + (1,) * (samples.ndim - 1))
        * (samples[:-1] + samples[1:])
    )
    result[1:] = np.cumsum(increments, axis=0)
    return result


def causal_transport_rank_audit(
    transport_differences: np.ndarray,
    *,
    maximum_secondary_ratio: float = 0.1,
) -> CausalTransportRankAudit:
    """Audit whether three-component transport samples are nearly rank one.

    The input has shape ``(n_samples, 3)``.  Each component must already use
    its frozen physical normalization so the singular values do not depend on
    units.
    """

    values = np.asarray(transport_differences, dtype=float)
    threshold = float(maximum_secondary_ratio)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != 3
        or np.any(~np.isfinite(values))
        or not np.isfinite(threshold)
        or threshold < 0.0
    ):
        raise ValueError("transport rank-audit inputs are invalid")
    _left, singular, right = np.linalg.svd(values, full_matrices=False)
    padded = np.zeros(3, dtype=float)
    padded[: singular.size] = singular
    leading = max(float(padded[0]), np.finfo(float).tiny)
    second_ratio = float(padded[1] / leading)
    third_ratio = float(padded[2] / leading)
    direction = np.asarray(right[0], dtype=float)
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    return CausalTransportRankAudit(
        singular_values=padded,
        second_to_first_ratio=second_ratio,
        third_to_first_ratio=third_ratio,
        dominant_direction=direction,
        passed=bool(
            padded[0] > np.finfo(float).tiny
            and second_ratio <= threshold
            and third_ratio <= threshold
        ),
    )


def causal_refined_spread_upper_bound(
    coarse_spreads: np.ndarray,
    fine_spreads: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative refinement uncertainty and upper spread bounds."""

    coarse = np.asarray(coarse_spreads, dtype=float)
    fine = np.asarray(fine_spreads, dtype=float)
    if (
        coarse.shape != fine.shape
        or np.any(~np.isfinite(coarse))
        or np.any(~np.isfinite(fine))
        or np.any(coarse < 0.0)
        or np.any(fine < 0.0)
    ):
        raise ValueError("refined spread inputs are invalid")
    uncertainty = np.abs(fine - coarse)
    return uncertainty, fine + uncertainty
