"""Adaptive arclength phase primitives for offline fast-mode atlases.

The expensive constrained vector field is supplied by a caller.  This module
only performs phase normalization, a projected Picard collocation update,
and deterministic fast-regime classification.  It therefore remains usable
by truth-free tests and by the table-driven online phase engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .phase_collocation import (
    direction_cosine,
    gauss_lobatto_nodes,
    lagrange_differentiation_matrix,
    lagrange_integration_matrix,
)


Array = np.ndarray
RateEvaluator = Callable[[Array, float], Array]


def _finite(value, *, ndim: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def weighted_speed(rate: Array, weights: Array | None = None) -> float:
    """Return the strictly positive norm defining trajectory arclength."""

    vector = _finite(rate, ndim=1, name="rate")
    if weights is None:
        scaled = vector
    else:
        scale = _finite(weights, ndim=1, name="weights")
        if scale.shape != vector.shape or np.any(scale <= 0.0):
            raise ValueError("weights must be positive and match the rate")
        scaled = scale * vector
    speed = float(np.linalg.norm(scaled))
    if not np.isfinite(speed) or speed <= np.finfo(float).tiny:
        raise ValueError("arclength phase speed is zero or nonfinite")
    return speed


def normalized_phase_rate(
    rate: Array, weights: Array | None = None
) -> tuple[Array, float]:
    """Return ``dy/ds`` and ``ds/dt`` for weighted arclength ``s``."""

    vector = _finite(rate, ndim=1, name="rate")
    speed = weighted_speed(vector, weights)
    return vector / speed, speed


def _orthonormal_basis(value: Array, dimension: int) -> Array:
    basis = _finite(value, ndim=2, name="basis")
    if basis.shape[0] != dimension or basis.shape[1] < 1:
        raise ValueError("basis has the wrong coordinate dimension")
    defect = float(np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]), ord=np.inf))
    if defect > 1.0e-10:
        raise ValueError("basis must be column-orthonormal")
    return basis


def arclength_picard_window(
    *,
    start_coordinate: Array,
    start_time_seconds: float,
    arclength_span: float,
    basis: Array,
    evaluator: RateEvaluator,
    node_count: int = 5,
    weights: Array | None = None,
) -> dict[str, Array | float]:
    """Apply one projected Picard update to ``dy/ds=f/||Wf||``.

    Physical time is integrated as a second dependent variable through
    ``dt/ds=1/||Wf||``.  Predictor and final nodal evaluations are retained so
    the caller can bind the final full-vector and time-map defects.
    """

    start = _finite(start_coordinate, ndim=1, name="start_coordinate")
    phase_basis = _orthonormal_basis(basis, len(start))
    span = float(arclength_span)
    start_time = float(start_time_seconds)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("arclength_span must be positive")
    if not np.isfinite(start_time):
        raise ValueError("start_time_seconds must be finite")
    if int(node_count) != node_count or node_count < 3:
        raise ValueError("arclength Picard windows require at least three nodes")
    if weights is not None:
        phase_weights = _finite(weights, ndim=1, name="weights")
        if phase_weights.shape != start.shape or np.any(phase_weights <= 0.0):
            raise ValueError("weights must be positive and match the coordinate")
    else:
        phase_weights = None

    nodes = gauss_lobatto_nodes(int(node_count))
    integration = lagrange_integration_matrix(nodes)
    differentiation = lagrange_differentiation_matrix(nodes)
    projector = phase_basis @ phase_basis.T

    start_rate = _finite(evaluator(start, start_time), ndim=1, name="start_rate")
    if start_rate.shape != start.shape:
        raise ValueError("evaluator returned the wrong coordinate dimension")
    start_direction, start_speed = normalized_phase_rate(start_rate, phase_weights)
    projected_start = projector @ start_direction
    predictor_coordinates = start[None, :] + (
        span * nodes[:, None] * projected_start[None, :]
    )
    predictor_times = start_time + span * nodes / start_speed
    predictor_rates = np.asarray(
        [
            evaluator(coordinate, time_value)
            for coordinate, time_value in zip(
                predictor_coordinates, predictor_times, strict=True
            )
        ],
        dtype=float,
    )
    if predictor_rates.shape != (int(node_count), len(start)):
        raise ValueError("predictor evaluator returned inconsistent rates")
    predictor_speeds = np.asarray(
        [weighted_speed(rate, phase_weights) for rate in predictor_rates]
    )
    predictor_directions = predictor_rates / predictor_speeds[:, None]
    reduced_directions = predictor_directions @ phase_basis
    coordinates = start[None, :] + (
        span * (integration @ reduced_directions) @ phase_basis.T
    )
    times = start_time + span * (integration @ (1.0 / predictor_speeds))

    final_rates = np.asarray(
        [
            evaluator(coordinate, time_value)
            for coordinate, time_value in zip(coordinates, times, strict=True)
        ],
        dtype=float,
    )
    if final_rates.shape != (int(node_count), len(start)):
        raise ValueError("final evaluator returned inconsistent rates")
    final_speeds = np.asarray(
        [weighted_speed(rate, phase_weights) for rate in final_rates]
    )
    final_directions = final_rates / final_speeds[:, None]
    projected_final = (final_directions @ phase_basis) @ phase_basis.T
    coordinate_derivative = differentiation @ coordinates / span
    time_derivative = differentiation @ times / span
    tiny = np.finfo(float).tiny
    projected_defects = np.linalg.norm(
        coordinate_derivative - projected_final, axis=1
    ) / np.maximum(np.linalg.norm(projected_final, axis=1), tiny)
    full_defects = np.linalg.norm(
        coordinate_derivative - final_directions, axis=1
    )
    normal_defects = np.linalg.norm(
        final_directions - projected_final, axis=1
    )
    direction_cosines = np.asarray(
        [
            direction_cosine(left, right)
            for left, right in zip(
                coordinate_derivative, final_directions, strict=True
            )
        ]
    )
    inverse_speeds = 1.0 / final_speeds
    time_mapping_defects = np.abs(time_derivative - inverse_speeds) / inverse_speeds
    return {
        "start_time_seconds": start_time,
        "end_time_seconds": float(times[-1]),
        "physical_duration_seconds": float(times[-1] - start_time),
        "arclength_span": span,
        "nodes": nodes,
        "predictor_coordinates": predictor_coordinates,
        "predictor_times_seconds": predictor_times,
        "predictor_rates_per_second": predictor_rates,
        "predictor_phase_speeds_per_second": predictor_speeds,
        "coordinates": coordinates,
        "times_seconds": times,
        "final_rates_per_second": final_rates,
        "final_phase_speeds_per_second": final_speeds,
        "final_phase_directions": final_directions,
        "coordinate_derivatives_per_arclength": coordinate_derivative,
        "time_derivatives_seconds_per_arclength": time_derivative,
        "projected_defects": projected_defects,
        "full_defects": full_defects,
        "normal_defects": normal_defects,
        "direction_cosines": direction_cosines,
        "time_mapping_defects": time_mapping_defects,
        "endpoint": coordinates[-1],
        "endpoint_time_seconds": float(times[-1]),
    }


@dataclass(frozen=True)
class AdaptiveArclengthPolicy:
    """Fail-closed segment-size policy, independent of truth evaluation."""

    initial_span: float = 2.5e-2
    minimum_span: float = 6.25e-3
    maximum_span: float = 5.0e-2
    shrink_factor: float = 0.5
    growth_factor: float = 1.5
    maximum_retries: int = 2

    def __post_init__(self) -> None:
        values = (
            self.initial_span,
            self.minimum_span,
            self.maximum_span,
            self.shrink_factor,
            self.growth_factor,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("arclength policy values must be positive and finite")
        if not self.minimum_span <= self.initial_span <= self.maximum_span:
            raise ValueError("initial arclength span lies outside policy bounds")
        if not 0.0 < self.shrink_factor < 1.0 or self.growth_factor <= 1.0:
            raise ValueError("invalid shrink or growth factor")
        if int(self.maximum_retries) != self.maximum_retries or self.maximum_retries < 0:
            raise ValueError("maximum_retries must be a nonnegative integer")

    def retry_span(self, span: float, retry_index: int) -> float:
        if retry_index >= self.maximum_retries:
            raise RuntimeError("arclength retry budget exhausted")
        candidate = float(span) * self.shrink_factor
        if candidate < self.minimum_span:
            raise RuntimeError("arclength minimum span would be violated")
        return candidate

    def next_span(self, span: float, *, growth_margin_passed: bool) -> float:
        value = float(span)
        if growth_margin_passed:
            return min(self.maximum_span, self.growth_factor * value)
        return value


@dataclass(frozen=True)
class FastRegimePolicy:
    """Prospective candidate rules; candidates still require refinement."""

    equilibrium_speed_ratio_maximum: float = 1.0e-3
    recurrence_distance_over_local_span_maximum: float = 0.1
    recurrence_direction_cosine_minimum: float = 0.99
    candidate_persistence_segments: int = 2


def classify_fast_regime(
    *,
    legacy_exit_run: int,
    equilibrium_run: int,
    recurrence_run: int,
    terminal_speed_ratio: float,
    closest_return_distance_over_local_span: float,
    closest_return_direction_cosine: float,
    policy: FastRegimePolicy = FastRegimePolicy(),
) -> str:
    """Classify a candidate regime without turning it into a certificate."""

    persistence = int(policy.candidate_persistence_segments)
    if legacy_exit_run >= persistence:
        return "legacy_transverse_exit_candidate"
    if (
        equilibrium_run >= persistence
        and terminal_speed_ratio <= policy.equilibrium_speed_ratio_maximum
    ):
        return "fast_equilibrium_candidate"
    if (
        recurrence_run >= persistence
        and closest_return_distance_over_local_span
        <= policy.recurrence_distance_over_local_span_maximum
        and closest_return_direction_cosine
        >= policy.recurrence_direction_cosine_minimum
    ):
        return "recurrent_orbit_candidate"
    return "continuing_fast_branch"
