"""Numerical utilities for causal coordinate-fiber healing audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CausalTransportRankAudit:
    """Rank-one diagnostic for a normalized interface-transport history."""

    singular_values: np.ndarray
    second_to_first_ratio: float
    third_to_first_ratio: float
    dominant_direction: np.ndarray
    passed: bool


@dataclass(frozen=True)
class CausalPathComponentDecomposition:
    """One path-integrated finite-difference output decomposition."""

    endpoint_difference: np.ndarray
    component_contributions: np.ndarray
    reconstructed_difference: np.ndarray
    maximum_reconstruction_relative_defect: float
    maximum_quadrature_relative_defect: float
    quadrature_order: int
    finite_difference_relative_step: float


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


def causal_internal_face_boundary_rates(
    internal_face_fluxes: np.ndarray,
) -> np.ndarray:
    """Map signed internal-face fluxes to conservative shell boundary rates.

    The input has shape ``(..., n_internal_faces, n_components)``.  The
    returned array has one more shell than internal faces and follows the
    production shell-ledger convention ``F_right - F_left``.  Consequently
    every internal face enters its two neighboring shells with opposite
    signs and the shell sum vanishes exactly up to floating-point summation.
    """

    values = np.asarray(internal_face_fluxes, dtype=float)
    if (
        values.ndim < 2
        or values.shape[-2] < 1
        or values.shape[-1] < 1
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("internal-face fluxes are invalid")
    result = np.zeros(
        values.shape[:-2] + (values.shape[-2] + 1, values.shape[-1]),
        dtype=float,
    )
    result[..., :-1, :] += values
    result[..., 1:, :] -= values
    return result


def _path_component_contributions(
    function: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    final: np.ndarray,
    *,
    quadrature_order: int,
    finite_difference_relative_step: float,
) -> np.ndarray:
    delta = final - initial
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    fractions = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    reference = np.asarray(function(initial), dtype=float)
    if reference.ndim != 1 or np.any(~np.isfinite(reference)):
        raise ValueError("path-decomposition output must be a finite vector")
    contributions = np.zeros((initial.size, reference.size), dtype=float)
    for fraction, weight in zip(fractions, weights, strict=True):
        point = initial + float(fraction) * delta
        for component in range(initial.size):
            if delta[component] == 0.0:
                continue
            step = finite_difference_relative_step * max(
                abs(float(point[component])),
                abs(float(delta[component])),
                1.0,
            )
            plus = point.copy()
            minus = point.copy()
            plus[component] += step
            minus[component] -= step
            derivative = (
                np.asarray(function(plus), dtype=float)
                - np.asarray(function(minus), dtype=float)
            ) / (2.0 * step)
            if derivative.shape != reference.shape or np.any(
                ~np.isfinite(derivative)
            ):
                raise ValueError("path-decomposition derivative is invalid")
            contributions[component] += (
                float(weight) * delta[component] * derivative
            )
    return contributions


def causal_path_integrated_component_decomposition(
    function: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    final: np.ndarray,
    *,
    quadrature_order: int = 16,
    finite_difference_relative_step: float = 1.0e-6,
) -> CausalPathComponentDecomposition:
    """Attribute a finite output change along the symmetric straight path.

    Each input component receives its Aumann--Shapley line-integral
    contribution.  Central finite differences evaluate the path Jacobian;
    half-order Gauss--Legendre quadrature supplies an independent integration
    check.  The endpoint difference remains the binding finite change.
    """

    start = np.asarray(initial, dtype=float)
    stop = np.asarray(final, dtype=float)
    order = int(quadrature_order)
    step = float(finite_difference_relative_step)
    if (
        start.ndim != 1
        or stop.shape != start.shape
        or start.size < 1
        or np.any(~np.isfinite(start))
        or np.any(~np.isfinite(stop))
        or order != quadrature_order
        or order < 4
        or order % 2
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("path-decomposition inputs are invalid")
    start_output = np.asarray(function(start), dtype=float)
    stop_output = np.asarray(function(stop), dtype=float)
    if (
        start_output.ndim != 1
        or stop_output.shape != start_output.shape
        or np.any(~np.isfinite(start_output))
        or np.any(~np.isfinite(stop_output))
    ):
        raise ValueError("path-decomposition endpoint outputs are invalid")
    contributions = _path_component_contributions(
        function,
        start,
        stop,
        quadrature_order=order,
        finite_difference_relative_step=step,
    )
    coarse = _path_component_contributions(
        function,
        start,
        stop,
        quadrature_order=order // 2,
        finite_difference_relative_step=step,
    )
    endpoint = stop_output - start_output
    reconstructed = np.sum(contributions, axis=0)
    scale = np.maximum(
        np.maximum(np.abs(start_output), np.abs(stop_output)),
        np.maximum(np.abs(endpoint), 1.0),
    )
    return CausalPathComponentDecomposition(
        endpoint_difference=endpoint,
        component_contributions=contributions,
        reconstructed_difference=reconstructed,
        maximum_reconstruction_relative_defect=float(
            np.max(np.abs(reconstructed - endpoint) / scale)
        ),
        maximum_quadrature_relative_defect=float(
            np.max(np.abs(contributions - coarse) / scale[None, :])
        ),
        quadrature_order=order,
        finite_difference_relative_step=step,
    )


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
