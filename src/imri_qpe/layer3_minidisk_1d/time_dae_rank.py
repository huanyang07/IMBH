"""Count and rank audits for candidate time-dependent DAE formulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeDAECount:
    """Unknown and residual counts for one shared-interface-flux system."""

    inner_nodes: int
    outer_cells: int
    boundary_eliminated: bool
    differential_variables: int
    algebraic_variables: int
    unknowns: int
    residuals: int


@dataclass(frozen=True)
class DescriptorRankAudit:
    """Singular-value audit of one instantaneous descriptor matrix."""

    matrix: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_estimate: float


def shared_flux_time_dae_count(
    inner_nodes: int,
    outer_cells: int,
    *,
    boundary_eliminated: bool,
) -> TimeDAECount:
    """Return the exact count for the lean shared-interface-flux layout."""

    if inner_nodes < 2 or outer_cells < 2:
        raise ValueError("the DAE count requires at least two nodes per domain")
    differential = 3 * outer_cells - int(boundary_eliminated)
    algebraic = 2 * inner_nodes + outer_cells + 5
    unknowns = differential + algebraic
    residuals = 2 * inner_nodes + 4 * outer_cells + (
        4 if boundary_eliminated else 5
    )
    if unknowns != residuals:
        raise RuntimeError("time-dependent DAE count is not square")
    return TimeDAECount(
        inner_nodes=int(inner_nodes),
        outer_cells=int(outer_cells),
        boundary_eliminated=bool(boundary_eliminated),
        differential_variables=int(differential),
        algebraic_variables=int(algebraic),
        unknowns=int(unknowns),
        residuals=int(residuals),
    )


def _rank_audit(matrix, relative_threshold: float) -> DescriptorRankAudit:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or np.any(~np.isfinite(matrix)):
        raise ValueError("rank-audit matrix must be finite and two-dimensional")
    if not 0.0 < relative_threshold < 1.0:
        raise ValueError("relative_threshold must lie in (0,1)")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = relative_threshold * largest
    rank = int(np.sum(singular_values > threshold))
    smallest = float(singular_values[-1]) if singular_values.size else 0.0
    condition = np.inf if smallest == 0.0 else largest / smallest
    return DescriptorRankAudit(
        matrix=matrix,
        singular_values=np.asarray(singular_values, dtype=float),
        rank=rank,
        condition_estimate=float(condition),
    )


def equilibrate_matrix(matrix, *, iterations: int = 6) -> np.ndarray:
    """Return a two-sided max-norm equilibration for numerical rank audits."""

    matrix = np.array(matrix, dtype=float, copy=True)
    if matrix.ndim != 2 or np.any(~np.isfinite(matrix)):
        raise ValueError("matrix must be finite and two-dimensional")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    for _ in range(iterations):
        row_scale = np.max(np.abs(matrix), axis=1)
        row_scale[row_scale == 0.0] = 1.0
        matrix /= row_scale[:, None]
        column_scale = np.max(np.abs(matrix), axis=0)
        column_scale[column_scale == 0.0] = 1.0
        matrix /= column_scale[None, :]
    return matrix


def matrix_rank_audit(
    matrix,
    *,
    relative_threshold: float = 1.0e-10,
    equilibrate: bool = False,
) -> DescriptorRankAudit:
    """Audit any matrix, optionally after declared two-sided equilibration."""

    if equilibrate:
        matrix = equilibrate_matrix(matrix)
    return _rank_audit(matrix, relative_threshold)


def eliminated_descriptor_audit(
    storage,
    tangent,
    flux_jacobian,
    algebraic_jacobian,
    *,
    relative_threshold: float = 1.0e-10,
) -> DescriptorRankAudit:
    """Audit ``K_elim = [[M P, -f_y], [0, g0_y]]``."""

    storage = np.asarray(storage, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    flux_jacobian = np.asarray(flux_jacobian, dtype=float)
    algebraic_jacobian = np.asarray(algebraic_jacobian, dtype=float)
    n = storage.shape[0]
    if storage.shape != (n, n):
        raise ValueError("storage must be square")
    if tangent.shape != (n, n - 1):
        raise ValueError("tangent must have shape (n,n-1)")
    m = flux_jacobian.shape[1]
    if flux_jacobian.shape != (n, m):
        raise ValueError("flux_jacobian has the wrong shape")
    if algebraic_jacobian.shape != (m - 1, m):
        raise ValueError("algebraic_jacobian must have shape (m-1,m)")
    matrix = np.block(
        [
            [storage @ tangent, -flux_jacobian],
            [np.zeros((m - 1, n - 1)), algebraic_jacobian],
        ]
    )
    return _rank_audit(matrix, relative_threshold)


def normal_closure_audit(
    storage,
    tangent,
    flux_jacobian,
    algebraic_jacobian,
    *,
    relative_threshold: float = 1.0e-10,
) -> DescriptorRankAudit:
    """Audit ``[g0_y; w.T f_y]`` using the left null vector of ``M P``."""

    storage = np.asarray(storage, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    flux_jacobian = np.asarray(flux_jacobian, dtype=float)
    algebraic_jacobian = np.asarray(algebraic_jacobian, dtype=float)
    storage_tangent = storage @ tangent
    left_vectors = np.linalg.svd(storage_tangent, full_matrices=True)[0]
    normal = left_vectors[:, -1]
    matrix = np.vstack((algebraic_jacobian, normal @ flux_jacobian))
    return _rank_audit(matrix, relative_threshold)


def constrained_tangency_audit(
    storage,
    constraint_gradient,
    flux_jacobian,
    algebraic_jacobian,
    *,
    relative_threshold: float = 1.0e-10,
) -> DescriptorRankAudit:
    """Audit ``[g0_y; c_q M^-1 f_y]`` for the constrained candidate."""

    storage = np.asarray(storage, dtype=float)
    constraint_gradient = np.asarray(constraint_gradient, dtype=float)
    flux_jacobian = np.asarray(flux_jacobian, dtype=float)
    algebraic_jacobian = np.asarray(algebraic_jacobian, dtype=float)
    n = storage.shape[0]
    if storage.shape != (n, n) or constraint_gradient.shape != (n,):
        raise ValueError("storage or constraint gradient has the wrong shape")
    m = flux_jacobian.shape[1]
    if flux_jacobian.shape != (n, m):
        raise ValueError("flux_jacobian has the wrong shape")
    if algebraic_jacobian.shape != (m - 1, m):
        raise ValueError("algebraic_jacobian must have shape (m-1,m)")
    tangency_jacobian = constraint_gradient @ np.linalg.solve(
        storage, flux_jacobian
    )
    matrix = np.vstack((algebraic_jacobian, tangency_jacobian))
    return _rank_audit(matrix, relative_threshold)
