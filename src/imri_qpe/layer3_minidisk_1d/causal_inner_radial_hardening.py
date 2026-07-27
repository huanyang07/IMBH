"""Numerical-hardening helpers for frozen radial candidate audits.

These routines are deliberately production neutral.  They compare a stored
colored finite-difference Jacobian with independently evaluated dense columns
and expose directional finite-difference step sensitivity.  They do not
define a new spatial operator or time integrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.sparse import csc_matrix, issparse


VectorFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CausalRadialDenseColoredAudit:
    """Dense-column comparison with one stored colored Jacobian."""

    selected_columns: np.ndarray
    dense_columns: np.ndarray
    colored_columns: np.ndarray
    per_column_relative_defects: np.ndarray
    maximum_relative_defect: float
    maximum_off_pattern_relative_entry: float
    dense_scale: float


@dataclass(frozen=True)
class CausalRadialJVPStepSweep:
    """Directional central-difference actions over a declared step ladder."""

    steps: np.ndarray
    direct_actions: np.ndarray
    matrix_action: np.ndarray
    matrix_relative_defects: np.ndarray
    adjacent_relative_changes: np.ndarray
    selected_step: float
    selected_step_index: int
    selected_matrix_relative_defect: float
    minimum_adjacent_relative_change: float


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size < 1 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def causal_radial_partial_dense_central_jacobian(
    function: VectorFunction,
    base: np.ndarray,
    selected_columns: Iterable[int],
    *,
    finite_difference_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate selected dense central-difference Jacobian columns."""

    point = _finite_vector(base, name="base")
    columns = np.asarray(tuple(int(value) for value in selected_columns))
    step = float(finite_difference_step)
    if (
        columns.ndim != 1
        or columns.size < 1
        or np.unique(columns).size != columns.size
        or np.any(columns < 0)
        or np.any(columns >= point.size)
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("dense-column audit inputs are invalid")
    reference = _finite_vector(function(point), name="function(base)")
    dense = np.empty((reference.size, columns.size), dtype=float)
    for offset, column in enumerate(columns):
        perturbation = np.zeros_like(point)
        perturbation[column] = step
        plus = _finite_vector(
            function(point + perturbation),
            name="function(base + step)",
        )
        minus = _finite_vector(
            function(point - perturbation),
            name="function(base - step)",
        )
        if plus.shape != reference.shape or minus.shape != reference.shape:
            raise ValueError("dense-column residual shape changed")
        dense[:, offset] = (plus - minus) / (2.0 * step)
    return columns, dense


def causal_radial_dense_colored_audit(
    function: VectorFunction,
    base: np.ndarray,
    colored_jacobian: np.ndarray,
    declared_pattern: np.ndarray | csc_matrix,
    selected_columns: Iterable[int],
    *,
    finite_difference_step: float,
) -> CausalRadialDenseColoredAudit:
    """Compare dense columns and audit derivatives outside the declared band."""

    point = _finite_vector(base, name="base")
    colored = np.asarray(colored_jacobian, dtype=float)
    if colored.ndim != 2 or colored.shape[1] != point.size:
        raise ValueError("colored Jacobian shape is invalid")
    columns, dense = causal_radial_partial_dense_central_jacobian(
        function,
        point,
        selected_columns,
        finite_difference_step=finite_difference_step,
    )
    if colored.shape[0] != dense.shape[0] or np.any(~np.isfinite(colored)):
        raise ValueError("colored and dense Jacobian shapes differ")
    colored_selected = colored[:, columns]
    scale = max(
        float(np.max(np.abs(dense))),
        float(np.max(np.abs(colored_selected))),
        np.finfo(float).tiny,
    )
    difference = dense - colored_selected
    per_column_scales = np.maximum(
        np.maximum(
            np.max(np.abs(dense), axis=0),
            np.max(np.abs(colored_selected), axis=0),
        ),
        np.finfo(float).tiny,
    )
    per_column_defects = (
        np.max(np.abs(difference), axis=0) / per_column_scales
    )
    pattern = (
        declared_pattern.toarray()
        if issparse(declared_pattern)
        else np.asarray(declared_pattern)
    )
    if pattern.shape != colored.shape:
        raise ValueError("declared Jacobian pattern shape is invalid")
    selected_pattern = np.asarray(pattern[:, columns], dtype=bool)
    off_pattern = np.abs(dense[~selected_pattern])
    maximum_off_pattern = (
        0.0 if off_pattern.size == 0 else float(np.max(off_pattern) / scale)
    )
    return CausalRadialDenseColoredAudit(
        selected_columns=columns,
        dense_columns=dense,
        colored_columns=np.array(colored_selected, copy=True),
        per_column_relative_defects=per_column_defects,
        maximum_relative_defect=float(
            np.max(np.abs(difference)) / scale
        ),
        maximum_off_pattern_relative_entry=maximum_off_pattern,
        dense_scale=scale,
    )


def causal_radial_jvp_step_sweep(
    function: VectorFunction,
    base: np.ndarray,
    matrix: np.ndarray,
    direction: np.ndarray,
    steps: Iterable[float],
    *,
    selected_step: float,
) -> CausalRadialJVPStepSweep:
    """Compare a stored matrix action with a central-difference step ladder."""

    point = _finite_vector(base, name="base")
    vector = _finite_vector(direction, name="direction")
    operator = np.asarray(matrix, dtype=float)
    step_values = np.asarray(tuple(float(value) for value in steps))
    chosen = float(selected_step)
    if (
        vector.shape != point.shape
        or operator.shape[1] != point.size
        or np.any(~np.isfinite(operator))
        or step_values.ndim != 1
        or step_values.size < 3
        or np.any(~np.isfinite(step_values))
        or np.any(step_values <= 0.0)
        or np.any(np.diff(step_values) <= 0.0)
        or not np.any(step_values == chosen)
    ):
        raise ValueError("JVP step-sweep inputs are invalid")
    matrix_action = operator @ vector
    direct_actions = []
    for step in step_values:
        plus = _finite_vector(
            function(point + step * vector),
            name="function(base + step * direction)",
        )
        minus = _finite_vector(
            function(point - step * vector),
            name="function(base - step * direction)",
        )
        direct_actions.append((plus - minus) / (2.0 * step))
    direct = np.asarray(direct_actions, dtype=float)
    if direct.shape[1:] != matrix_action.shape:
        raise ValueError("JVP residual shape changed")

    matrix_scales = np.maximum(
        np.maximum(
            np.linalg.norm(direct, axis=1),
            float(np.linalg.norm(matrix_action)),
        ),
        np.finfo(float).tiny,
    )
    matrix_defects = (
        np.linalg.norm(direct - matrix_action[None, :], axis=1)
        / matrix_scales
    )
    adjacent_scales = np.maximum(
        np.maximum(
            np.linalg.norm(direct[:-1], axis=1),
            np.linalg.norm(direct[1:], axis=1),
        ),
        np.finfo(float).tiny,
    )
    adjacent_changes = (
        np.linalg.norm(np.diff(direct, axis=0), axis=1)
        / adjacent_scales
    )
    selected_index = int(np.flatnonzero(step_values == chosen)[0])
    return CausalRadialJVPStepSweep(
        steps=step_values,
        direct_actions=direct,
        matrix_action=np.asarray(matrix_action, dtype=float),
        matrix_relative_defects=matrix_defects,
        adjacent_relative_changes=adjacent_changes,
        selected_step=chosen,
        selected_step_index=selected_index,
        selected_matrix_relative_defect=float(
            matrix_defects[selected_index]
        ),
        minimum_adjacent_relative_change=float(
            np.min(adjacent_changes)
        ),
    )
