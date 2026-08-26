"""Deterministic inexact-Newton algebra for fixed-slow root solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class BoundedInexactDirection:
    step: np.ndarray
    predicted_residual: np.ndarray
    forcing_two_norm: float
    forcing_infinity_norm: float
    normalized_directional_derivative: float
    maximum_absolute_step: float
    active_bound_count: int
    backend_success: bool
    backend_status: int
    backend_iterations: int
    backend_message: str


def bounded_trf_inexact_direction(
    jacobian,
    residual,
    *,
    maximum_absolute_step: float = 0.25,
    maximum_backend_iterations: int = 500,
    backend_tolerance: float = 1.0e-12,
) -> BoundedInexactDirection:
    """Return a bounded TRF direction and its inexact-Newton diagnostics."""

    matrix = np.asarray(jacobian, dtype=float)
    values = np.asarray(residual, dtype=float).ravel()
    trust = float(maximum_absolute_step)
    maximum_iterations = int(maximum_backend_iterations)
    tolerance = float(backend_tolerance)
    if (
        matrix.shape != (values.size, values.size)
        or np.any(~np.isfinite(matrix))
        or np.any(~np.isfinite(values))
        or not np.isfinite(trust)
        or trust <= 0.0
        or maximum_iterations <= 0
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("bounded inexact-direction inputs are invalid")
    result = lsq_linear(
        matrix,
        -values,
        bounds=(-trust, trust),
        method="trf",
        lsq_solver="exact",
        tol=tolerance,
        max_iter=maximum_iterations,
    )
    step = np.asarray(result.x, dtype=float)
    predicted = values + matrix @ step
    two_scale = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    infinity_scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    gradient = matrix.T @ values
    objective_scale = max(float(values @ values), np.finfo(float).tiny)
    active = np.abs(step) >= trust * (1.0 - 1.0e-8)
    return BoundedInexactDirection(
        step=step,
        predicted_residual=predicted,
        forcing_two_norm=float(np.linalg.norm(predicted) / two_scale),
        forcing_infinity_norm=float(np.max(np.abs(predicted)) / infinity_scale),
        normalized_directional_derivative=float((gradient @ step) / objective_scale),
        maximum_absolute_step=float(np.max(np.abs(step))),
        active_bound_count=int(np.sum(active)),
        backend_success=bool(result.success),
        backend_status=int(result.status),
        backend_iterations=int(result.nit),
        backend_message=str(result.message),
    )


def good_broyden_matrix_update(jacobian, step, residual_change) -> np.ndarray:
    """Apply the good-Broyden secant update to a square matrix."""

    matrix = np.asarray(jacobian, dtype=float)
    displacement = np.asarray(step, dtype=float).ravel()
    change = np.asarray(residual_change, dtype=float).ravel()
    if (
        matrix.shape != (displacement.size, displacement.size)
        or change.shape != displacement.shape
        or np.any(~np.isfinite(matrix))
        or np.any(~np.isfinite(displacement))
        or np.any(~np.isfinite(change))
    ):
        raise ValueError("good-Broyden update inputs are invalid")
    denominator = float(displacement @ displacement)
    if denominator <= np.finfo(float).tiny:
        raise ValueError("good-Broyden update displacement is zero")
    return matrix + np.outer(
        change - matrix @ displacement, displacement
    ) / denominator


__all__ = (
    "BoundedInexactDirection",
    "bounded_trf_inexact_direction",
    "good_broyden_matrix_update",
)
