"""Domain-scoped hardening helpers for frozen radial audits.

The global WP10c9d5a gate deliberately judged every residual row together.
These helpers preserve that result while allowing a follow-up audit to ask a
more limited question: whether a declared inner control volume and its stencil
halo have a stable frozen derivative.  They do not define a new physical
operator, boundary condition, or time integrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


VectorFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CausalRadialProjectedJVPAudit:
    """One stored matrix action and direct JVPs projected onto chosen rows."""

    selected_rows: np.ndarray
    steps: np.ndarray
    direct_actions: np.ndarray
    matrix_action: np.ndarray
    matrix_relative_defects: np.ndarray
    adjacent_relative_changes: np.ndarray
    selected_step: float
    selected_step_index: int
    selected_matrix_relative_defect: float


@dataclass(frozen=True)
class CausalRadialJVPSpatialAttribution:
    """Cellwise attribution of adjacent direct-JVP changes."""

    steps: np.ndarray
    cell_squared_fractions: np.ndarray
    dominant_cells: np.ndarray


@dataclass(frozen=True)
class CausalRadialOneSidedJVPSweep:
    """Forward, backward, and centered actions over one step ladder."""

    steps: np.ndarray
    forward_actions: np.ndarray
    backward_actions: np.ndarray
    centered_actions: np.ndarray
    one_sided_relative_mismatches: np.ndarray


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size < 1 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _steps(values: Iterable[float]) -> np.ndarray:
    steps = np.asarray(tuple(float(value) for value in values), dtype=float)
    if (
        steps.ndim != 1
        or steps.size < 2
        or np.any(~np.isfinite(steps))
        or np.any(steps <= 0.0)
        or np.any(np.diff(steps) <= 0.0)
    ):
        raise ValueError("directional derivative steps are invalid")
    return steps


def causal_radial_project_jvp_actions(
    direct_actions: np.ndarray,
    matrix_action: np.ndarray,
    steps: Iterable[float],
    selected_rows: Iterable[int],
    *,
    selected_step: float,
) -> CausalRadialProjectedJVPAudit:
    """Project a stored JVP step sweep onto a declared residual-row domain."""

    step_values = _steps(steps)
    direct = np.asarray(direct_actions, dtype=float)
    matrix = _finite_vector(matrix_action, name="matrix_action")
    rows = np.asarray(tuple(int(value) for value in selected_rows), dtype=int)
    chosen = float(selected_step)
    if (
        direct.shape != (step_values.size, matrix.size)
        or np.any(~np.isfinite(direct))
        or rows.ndim != 1
        or rows.size < 1
        or np.unique(rows).size != rows.size
        or np.any(rows < 0)
        or np.any(rows >= matrix.size)
        or not np.isfinite(chosen)
        or not np.any(step_values == chosen)
    ):
        raise ValueError("projected JVP inputs are invalid")
    projected_direct = direct[:, rows]
    projected_matrix = matrix[rows]
    matrix_scales = np.maximum(
        np.maximum(
            np.linalg.norm(projected_direct, axis=1),
            float(np.linalg.norm(projected_matrix)),
        ),
        np.finfo(float).tiny,
    )
    matrix_defects = (
        np.linalg.norm(
            projected_direct - projected_matrix[None, :],
            axis=1,
        )
        / matrix_scales
    )
    adjacent_scales = np.maximum(
        np.maximum(
            np.linalg.norm(projected_direct[:-1], axis=1),
            np.linalg.norm(projected_direct[1:], axis=1),
        ),
        np.finfo(float).tiny,
    )
    adjacent_changes = (
        np.linalg.norm(np.diff(projected_direct, axis=0), axis=1)
        / adjacent_scales
    )
    selected_index = int(np.flatnonzero(step_values == chosen)[0])
    return CausalRadialProjectedJVPAudit(
        selected_rows=rows,
        steps=step_values,
        direct_actions=np.asarray(projected_direct, dtype=float),
        matrix_action=np.asarray(projected_matrix, dtype=float),
        matrix_relative_defects=np.asarray(matrix_defects, dtype=float),
        adjacent_relative_changes=np.asarray(
            adjacent_changes,
            dtype=float,
        ),
        selected_step=chosen,
        selected_step_index=selected_index,
        selected_matrix_relative_defect=float(
            matrix_defects[selected_index]
        ),
    )


def causal_radial_jvp_spatial_attribution(
    direct_actions: np.ndarray,
    steps: Iterable[float],
    *,
    n_fields: int,
) -> CausalRadialJVPSpatialAttribution:
    """Attribute every adjacent direct-JVP change to residual cells."""

    step_values = _steps(steps)
    direct = np.asarray(direct_actions, dtype=float)
    fields = int(n_fields)
    if (
        fields < 1
        or direct.ndim != 2
        or direct.shape[0] != step_values.size
        or direct.shape[1] % fields != 0
        or np.any(~np.isfinite(direct))
    ):
        raise ValueError("spatial JVP attribution inputs are invalid")
    differences = np.diff(direct, axis=0).reshape(
        step_values.size - 1,
        -1,
        fields,
    )
    cell_squares = np.sum(differences * differences, axis=2)
    total_squares = np.sum(cell_squares, axis=1)
    if np.any(total_squares <= np.finfo(float).tiny):
        raise ValueError("adjacent JVP changes are degenerate")
    fractions = cell_squares / total_squares[:, None]
    return CausalRadialJVPSpatialAttribution(
        steps=step_values,
        cell_squared_fractions=np.asarray(fractions, dtype=float),
        dominant_cells=np.argmax(fractions, axis=1).astype(int),
    )


def causal_radial_one_sided_jvp_sweep(
    function: VectorFunction,
    base: np.ndarray,
    direction: np.ndarray,
    steps: Iterable[float],
) -> CausalRadialOneSidedJVPSweep:
    """Evaluate forward, backward, and centered actions at fixed branches."""

    point = _finite_vector(base, name="base")
    vector = _finite_vector(direction, name="direction")
    step_values = _steps(steps)
    if vector.shape != point.shape:
        raise ValueError("one-sided JVP direction shape is invalid")
    reference = _finite_vector(function(point), name="function(base)")
    forward = []
    backward = []
    for step in step_values:
        plus = _finite_vector(
            function(point + step * vector),
            name="function(base + step * direction)",
        )
        minus = _finite_vector(
            function(point - step * vector),
            name="function(base - step * direction)",
        )
        if plus.shape != reference.shape or minus.shape != reference.shape:
            raise ValueError("one-sided JVP residual shape changed")
        forward.append((plus - reference) / step)
        backward.append((reference - minus) / step)
    forward_array = np.asarray(forward, dtype=float)
    backward_array = np.asarray(backward, dtype=float)
    centered = 0.5 * (forward_array + backward_array)
    mismatch_scales = np.maximum(
        np.maximum(
            np.linalg.norm(forward_array, axis=1),
            np.linalg.norm(backward_array, axis=1),
        ),
        np.finfo(float).tiny,
    )
    mismatches = (
        np.linalg.norm(forward_array - backward_array, axis=1)
        / mismatch_scales
    )
    return CausalRadialOneSidedJVPSweep(
        steps=step_values,
        forward_actions=forward_array,
        backward_actions=backward_array,
        centered_actions=centered,
        one_sided_relative_mismatches=np.asarray(
            mismatches,
            dtype=float,
        ),
    )


def causal_radial_volume_weighted_scaled_direction(
    values: np.ndarray,
    cell_measures: np.ndarray,
) -> np.ndarray:
    """Normalize one scaled primitive field by its cell-volume RMS."""

    field = np.asarray(values, dtype=float)
    measures = _finite_vector(cell_measures, name="cell_measures")
    if (
        field.ndim != 2
        or field.shape[0] != measures.size
        or np.any(~np.isfinite(field))
        or np.any(measures <= 0.0)
    ):
        raise ValueError("volume-weighted direction inputs are invalid")
    active = np.linalg.norm(field, axis=1) > np.finfo(float).tiny
    if not np.any(active):
        raise ValueError("volume-weighted direction is degenerate")
    normalized_measures = np.zeros_like(measures)
    normalized_measures[active] = (
        measures[active] / float(np.sum(measures[active]))
    )
    norm = float(
        np.sqrt(
            np.sum(
                normalized_measures[:, None] * field * field,
            )
        )
    )
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("volume-weighted direction is degenerate")
    return np.asarray(field / norm, dtype=float)
