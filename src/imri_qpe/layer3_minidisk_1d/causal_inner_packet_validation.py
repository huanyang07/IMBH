"""Prospective causal-inner packet propagation audit helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve


@dataclass(frozen=True)
class CausalPacketHistoryMetrics:
    """Three-grid convergence metrics for one physical history."""

    significant_components: np.ndarray
    observed_rms_order: float
    observed_maximum_order: float
    component_orders: np.ndarray
    minimum_significant_component_order: float
    coarse_medium_rms_difference: float
    medium_fine_rms_difference: float
    maximum_fine_normalized_difference: float
    history_cosine: float
    refinement_error_cosine: float
    passed: bool


@dataclass(frozen=True)
class CausalExactSemigroupIntegral:
    """Exact-in-time integral reconstructed from a linear semigroup."""

    integrated_states: np.ndarray
    correction_states: np.ndarray
    relative_solve_residuals: np.ndarray
    maximum_relative_solve_residual: float


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float).ravel()
    second = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.dot(first, second) / denominator)


def causal_packet_history_metrics(
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
    *,
    physical_scales: np.ndarray,
    minimum_rms_order: float,
    minimum_maximum_order: float,
    minimum_significant_component_order: float,
    maximum_fine_normalized_difference: float,
    minimum_history_cosine: float,
    minimum_refinement_error_cosine: float,
    relative_activity: float = 1.0e-8,
) -> CausalPacketHistoryMetrics:
    """Evaluate the frozen prospective three-grid history contract."""

    histories = tuple(
        np.asarray(values, dtype=float)
        for values in (coarse, medium, fine)
    )
    scales = np.asarray(physical_scales, dtype=float).ravel()
    if (
        histories[0].ndim != 2
        or histories[0].shape != histories[1].shape
        or histories[0].shape != histories[2].shape
        or histories[0].shape[1] != scales.size
        or np.any(~np.isfinite(np.asarray(histories)))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or not 0.0 < float(relative_activity) < 1.0
    ):
        raise ValueError("packet-history inputs are invalid")
    normalized = tuple(values / scales[None, :] for values in histories)
    response = np.max(np.abs(np.asarray(normalized)), axis=(0, 1))
    significant = response >= float(relative_activity)
    if not np.any(significant):
        raise ValueError("packet history has no significant component")

    coarse_active, medium_active, fine_active = (
        values[:, significant] for values in normalized
    )
    coarse_medium = medium_active - coarse_active
    medium_fine = fine_active - medium_active
    tiny = np.finfo(float).tiny
    coarse_rms = float(np.sqrt(np.mean(coarse_medium**2)))
    fine_rms = float(np.sqrt(np.mean(medium_fine**2)))
    coarse_maximum = float(np.max(np.abs(coarse_medium)))
    fine_maximum = float(np.max(np.abs(medium_fine)))
    component_coarse = np.sqrt(np.mean(coarse_medium**2, axis=0))
    component_fine = np.sqrt(np.mean(medium_fine**2, axis=0))
    component_orders = np.log2(
        np.maximum(component_coarse, tiny)
        / np.maximum(component_fine, tiny)
    )
    rms_order = float(
        np.log2(max(coarse_rms, tiny) / max(fine_rms, tiny))
    )
    maximum_order = float(
        np.log2(
            max(coarse_maximum, tiny) / max(fine_maximum, tiny)
        )
    )
    history_cosine = _cosine(medium_active, fine_active)
    error_cosine = _cosine(coarse_medium, medium_fine)
    minimum_component = float(np.min(component_orders))
    passed = bool(
        rms_order >= float(minimum_rms_order)
        and maximum_order >= float(minimum_maximum_order)
        and minimum_component
        >= float(minimum_significant_component_order)
        and fine_maximum <= float(maximum_fine_normalized_difference)
        and history_cosine >= float(minimum_history_cosine)
        and error_cosine >= float(minimum_refinement_error_cosine)
    )
    return CausalPacketHistoryMetrics(
        significant_components=np.flatnonzero(significant),
        observed_rms_order=rms_order,
        observed_maximum_order=maximum_order,
        component_orders=np.asarray(component_orders, dtype=float),
        minimum_significant_component_order=minimum_component,
        coarse_medium_rms_difference=coarse_rms,
        medium_fine_rms_difference=fine_rms,
        maximum_fine_normalized_difference=fine_maximum,
        history_cosine=history_cosine,
        refinement_error_cosine=error_cosine,
        passed=passed,
    )


def causal_exact_semigroup_integral_history(
    generator: np.ndarray,
    state_history: np.ndarray,
    initial_states: np.ndarray,
) -> CausalExactSemigroupIntegral:
    """Recover ``integral exp(t G) v dt`` at every supplied time.

    The history has shape ``(time, state, packet)``.  One LU factorization
    solves ``G x(t) = exp(t G) v - v`` for every nonzero time and packet.
    A single iterative-refinement step is applied and separately returned.
    """

    matrix = np.asarray(generator, dtype=float)
    history = np.asarray(state_history, dtype=float)
    initial = np.asarray(initial_states, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or history.ndim != 3
        or history.shape[1] != matrix.shape[0]
        or initial.shape != history.shape[1:]
        or history.shape[0] < 2
        or np.any(~np.isfinite(matrix))
        or np.any(~np.isfinite(history))
        or np.any(~np.isfinite(initial))
    ):
        raise ValueError("semigroup-integral inputs are invalid")
    delta = history - initial[None, :, :]
    right_hand_side = np.transpose(
        delta[1:],
        (1, 0, 2),
    ).reshape(matrix.shape[0], -1)
    factorization = lu_factor(matrix, check_finite=False)
    integrated = lu_solve(
        factorization,
        right_hand_side,
        check_finite=False,
    )
    residual = matrix @ integrated - right_hand_side
    correction = lu_solve(
        factorization,
        residual,
        check_finite=False,
    )
    refined = integrated - correction
    refined_residual = matrix @ refined - right_hand_side
    column_scales = np.maximum(
        np.linalg.norm(right_hand_side, axis=0),
        np.finfo(float).tiny,
    )
    relative = np.linalg.norm(refined_residual, axis=0) / column_scales

    output_shape = (history.shape[0] - 1, history.shape[2])
    integrated_history = np.zeros_like(history)
    correction_history = np.zeros_like(history)
    integrated_history[1:] = np.transpose(
        refined.reshape(matrix.shape[0], *output_shape),
        (1, 0, 2),
    )
    correction_history[1:] = np.transpose(
        (refined - integrated).reshape(matrix.shape[0], *output_shape),
        (1, 0, 2),
    )
    residual_history = np.zeros(
        (history.shape[0], history.shape[2]),
        dtype=float,
    )
    residual_history[1:] = relative.reshape(output_shape)
    return CausalExactSemigroupIntegral(
        integrated_states=integrated_history,
        correction_states=correction_history,
        relative_solve_residuals=residual_history,
        maximum_relative_solve_residual=float(np.max(relative)),
    )
