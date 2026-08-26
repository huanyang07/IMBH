"""Cellwise fixed-slow reduction for the seven-field relaxation system.

The exact mass, angular-momentum, and total-energy states are frozen in
each radial cell.  Radial momentum, causal shear, height, and vertical
momentum remain dynamic.  This is a cellwise slow-manifold construction;
it is deliberately distinct from the older three-global-ledger reaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from imri_qpe.constants import C

from .causal_inner_generalized_maxwell_cattaneo import (
    generalized_maxwell_cattaneo_local_state,
    generalized_maxwell_cattaneo_principal,
)
from .causal_inner_generalized_maxwell_cattaneo_radial import (
    GeneralizedMaxwellCattaneoRadialOperator,
    generalized_maxwell_cattaneo_radial_operator,
)
from .causal_inner_geometry import kerr_schild_column_geometry


SLOW_EXACT_ROWS = np.asarray((0, 2, 3), dtype=int)
FAST_EQUATION_ROWS = np.asarray((1, 4, 5, 6), dtype=int)
SLOW_CHART_INDICES = np.asarray((0, 2, 3), dtype=int)
FAST_CHART_INDICES = np.asarray((1, 4, 5, 6), dtype=int)
SLOW_CHART_SCALES = np.asarray((1.0, 0.1, 1.0), dtype=float)
FAST_CHART_SCALES = np.asarray((0.1, 1.0e-4, 1.0, 0.03), dtype=float)


@dataclass(frozen=True)
class FixedSlowReconstruction:
    primitive_charts: np.ndarray
    slow_targets: np.ndarray
    maximum_constraint_relative_defect: float
    maximum_newton_corrections: int
    maximum_scaled_slow_chart_correction: float


@dataclass(frozen=True)
class FixedSlowEvaluation:
    reconstruction: FixedSlowReconstruction
    radial_operator: GeneralizedMaxwellCattaneoRadialOperator
    projected_primitive_rates_per_second: np.ndarray
    projected_fast_rates_per_second: np.ndarray
    normalized_fast_rates: np.ndarray
    slow_integrated_drift_per_second: np.ndarray
    maximum_temporal_projection_solve_relative_defect: float


def _relative(defect, *references) -> float:
    scale = max(
        *(float(np.max(np.abs(np.asarray(item)))) for item in references),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(np.asarray(defect))) / scale)


def _local_state(context, cell: int, chart: np.ndarray):
    radius = float(context.grid.centers[cell])
    geometry = kerr_schild_column_geometry(
        radius, context.grid.gravitational_radius
    )
    return generalized_maxwell_cattaneo_local_state(
        geometry,
        chart,
        proper_vertical_frequency=float(
            context.vertical_frequency.frequency(radius)
        ),
        alpha=float(context.alpha),
        stress_factor=float(context.stress_factor),
    )


def generalized_maxwell_cattaneo_slow_targets(
    context,
    primitive_charts,
) -> np.ndarray:
    """Return cell-integrated exact ``(M,J,E)`` targets."""

    charts = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if charts.shape != (n_cells, 7) or np.any(~np.isfinite(charts)):
        raise ValueError("seven-field slow-target charts are invalid")
    targets = np.empty((n_cells, 3), dtype=float)
    for cell in range(n_cells):
        state = _local_state(context, cell, charts[cell])
        measure = float(context.grid.cell_measures[cell])
        targets[cell] = measure * state.conservative_state6[SLOW_EXACT_ROWS]
    return targets


def generalized_maxwell_cattaneo_fast_charts(primitive_charts) -> np.ndarray:
    charts = np.asarray(primitive_charts, dtype=float)
    if charts.ndim != 2 or charts.shape[1] != 7 or np.any(~np.isfinite(charts)):
        raise ValueError("seven-field fast-chart input is invalid")
    return np.array(charts[:, FAST_CHART_INDICES], copy=True)


def generalized_maxwell_cattaneo_reconstruct_fixed_slow(
    context,
    slow_targets,
    fast_charts,
    *,
    template_charts,
    constraint_tolerance: float = 1.0e-11,
    maximum_newton_corrections: int = 8,
) -> FixedSlowReconstruction:
    """Reconstruct local slow charts at fixed exact ``(M,J,E)``."""

    targets = np.asarray(slow_targets, dtype=float)
    fast = np.asarray(fast_charts, dtype=float)
    template = np.asarray(template_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if (
        targets.shape != (n_cells, 3)
        or fast.shape != (n_cells, 4)
        or template.shape != (n_cells, 7)
        or np.any(~np.isfinite(targets))
        or np.any(~np.isfinite(fast))
        or np.any(~np.isfinite(template))
    ):
        raise ValueError("fixed-slow reconstruction inputs are invalid")
    tolerance = float(constraint_tolerance)
    maximum = int(maximum_newton_corrections)
    if tolerance <= 0.0 or maximum < 0:
        raise ValueError("fixed-slow reconstruction controls are invalid")
    charts = np.array(template, copy=True)
    charts[:, FAST_CHART_INDICES] = fast
    greatest_defect = 0.0
    greatest_corrections = 0
    greatest_scaled_correction = 0.0
    steps = np.asarray((2.0e-6, 2.0e-7, 2.0e-6), dtype=float)
    for cell in range(n_cells):
        measure = float(context.grid.cell_measures[cell])
        target = targets[cell]
        scale = np.maximum(np.abs(target), np.finfo(float).tiny)

        def residual(slow_values: np.ndarray) -> np.ndarray:
            candidate = np.array(charts[cell], copy=True)
            candidate[SLOW_CHART_INDICES] = slow_values
            state = _local_state(context, cell, candidate)
            value = measure * state.conservative_state6[SLOW_EXACT_ROWS]
            return (value - target) / scale

        slow = np.array(charts[cell, SLOW_CHART_INDICES], copy=True)
        values = residual(slow)
        corrections = 0
        while float(np.max(np.abs(values))) > tolerance:
            if corrections >= maximum:
                raise RuntimeError("fixed-slow local reconstruction did not converge")
            jacobian = np.empty((3, 3), dtype=float)
            for column, step in enumerate(steps):
                direction = np.zeros(3)
                direction[column] = step
                jacobian[:, column] = (
                    residual(slow + direction) - residual(slow - direction)
                ) / (2.0 * step)
            correction = np.linalg.solve(jacobian, -values)
            accepted = False
            old_norm = float(np.max(np.abs(values)))
            for exponent in range(9):
                factor = 2.0 ** (-exponent)
                candidate = slow + factor * correction
                try:
                    candidate_values = residual(candidate)
                except (ValueError, FloatingPointError, OverflowError):
                    continue
                if float(np.max(np.abs(candidate_values))) < old_norm:
                    slow = candidate
                    values = candidate_values
                    greatest_scaled_correction = max(
                        greatest_scaled_correction,
                        float(np.max(np.abs(factor * correction / SLOW_CHART_SCALES))),
                    )
                    accepted = True
                    break
            if not accepted:
                raise RuntimeError("fixed-slow local reconstruction line search failed")
            corrections += 1
        charts[cell, SLOW_CHART_INDICES] = slow
        greatest_defect = max(greatest_defect, float(np.max(np.abs(values))))
        greatest_corrections = max(greatest_corrections, corrections)
    return FixedSlowReconstruction(
        primitive_charts=charts,
        slow_targets=np.array(targets, copy=True),
        maximum_constraint_relative_defect=greatest_defect,
        maximum_newton_corrections=greatest_corrections,
        maximum_scaled_slow_chart_correction=greatest_scaled_correction,
    )


def generalized_maxwell_cattaneo_projected_fast_evaluation(
    context,
    slow_targets,
    fast_charts,
    *,
    template_charts,
    fast_rate_scales_per_second: np.ndarray | None = None,
    quadrature_order: int = 8,
    constraint_tolerance: float = 1.0e-11,
) -> FixedSlowEvaluation:
    """Evaluate the exact fixed-slow projected fast vector field."""

    reconstruction = generalized_maxwell_cattaneo_reconstruct_fixed_slow(
        context,
        slow_targets,
        fast_charts,
        template_charts=template_charts,
        constraint_tolerance=constraint_tolerance,
    )
    charts = reconstruction.primitive_charts
    operator = generalized_maxwell_cattaneo_radial_operator(
        context,
        charts,
        quadrature_order=int(quadrature_order),
    )
    n_cells = charts.shape[0]
    projected = np.empty_like(charts)
    defects = np.empty(n_cells, dtype=float)
    for cell, radius_value in enumerate(np.asarray(context.grid.centers, dtype=float)):
        radius = float(radius_value)
        geometry = kerr_schild_column_geometry(
            radius, context.grid.gravitational_radius
        )
        principal = generalized_maxwell_cattaneo_principal(
            geometry,
            charts[cell],
            proper_vertical_frequency=float(
                context.vertical_frequency.frequency(radius)
            ),
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
        )
        matrix = np.vstack(
            (
                principal.temporal_matrix[SLOW_EXACT_ROWS],
                principal.temporal_matrix[FAST_EQUATION_ROWS],
            )
        )
        right = np.concatenate(
            (
                np.zeros(3),
                operator.equation_right_hand_sides_per_cm[
                    cell, FAST_EQUATION_ROWS
                ],
            )
        )
        rate_per_ct = np.linalg.solve(matrix, right)
        projected[cell] = C * rate_per_ct
        defects[cell] = _relative(matrix @ rate_per_ct - right, matrix @ rate_per_ct, right)
    fast_rates = projected[:, FAST_CHART_INDICES]
    if fast_rate_scales_per_second is None:
        normalized = np.array(fast_rates, copy=True)
    else:
        scales = np.asarray(fast_rate_scales_per_second, dtype=float)
        if scales.shape == (4,):
            scales = np.broadcast_to(scales, fast_rates.shape)
        if scales.shape != fast_rates.shape or np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("fixed-slow fast-rate scales are invalid")
        normalized = fast_rates / scales
    integrated_rhs = (
        operator.equation_right_hand_sides_per_cm
        * np.asarray(context.grid.cell_measures)[:, None]
    )
    slow_drift = C * integrated_rhs[:, SLOW_EXACT_ROWS]
    return FixedSlowEvaluation(
        reconstruction=reconstruction,
        radial_operator=operator,
        projected_primitive_rates_per_second=projected,
        projected_fast_rates_per_second=fast_rates,
        normalized_fast_rates=normalized,
        slow_integrated_drift_per_second=slow_drift,
        maximum_temporal_projection_solve_relative_defect=float(np.max(defects)),
    )


def generalized_maxwell_cattaneo_fast_rate_scales(
    fast_rates_per_second,
) -> np.ndarray:
    rates = np.asarray(fast_rates_per_second, dtype=float)
    if rates.ndim != 2 or rates.shape[1] != 4 or np.any(~np.isfinite(rates)):
        raise ValueError("fixed-slow base fast rates are invalid")
    component = np.max(np.abs(rates), axis=0)
    floor = max(float(np.max(component)) * 1.0e-12, np.finfo(float).tiny)
    return np.maximum(component, floor)


def radius_one_colored_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    coordinates,
    *,
    relative_step: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate a four-field radius-one stencil with 12 evaluations."""

    point = np.asarray(coordinates, dtype=float)
    if point.ndim != 2 or point.shape[1] != 4 or np.any(~np.isfinite(point)):
        raise ValueError("colored-Jacobian coordinates are invalid")
    step = float(relative_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("colored-Jacobian step is invalid")
    base = np.asarray(function(point), dtype=float)
    if base.shape != point.shape or np.any(~np.isfinite(base)):
        raise ValueError("colored-Jacobian function output is invalid")
    n_cells = point.shape[0]
    jacobian = np.zeros((4 * n_cells, 4 * n_cells), dtype=float)
    for field in range(4):
        for color in range(3):
            perturbed = np.array(point, copy=True)
            columns = np.arange(color, n_cells, 3, dtype=int)
            perturbed[columns, field] += step
            difference = (np.asarray(function(perturbed), dtype=float) - base) / step
            for cell in columns:
                for output_cell in range(max(0, cell - 1), min(n_cells, cell + 2)):
                    rows = slice(4 * output_cell, 4 * output_cell + 4)
                    jacobian[rows, 4 * cell + field] = difference[output_cell]
    return base, jacobian


def directional_jacobian_relative_defect(
    function: Callable[[np.ndarray], np.ndarray],
    coordinates,
    jacobian,
    direction,
    *,
    relative_step: float = 2.0e-6,
) -> float:
    point = np.asarray(coordinates, dtype=float)
    vector = np.asarray(direction, dtype=float)
    matrix = np.asarray(jacobian, dtype=float)
    if vector.shape != point.shape or matrix.shape != (point.size, point.size):
        raise ValueError("directional Jacobian audit shapes are invalid")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("directional Jacobian audit direction is invalid")
    unit = vector / norm
    step = float(relative_step)
    finite_difference = (
        np.asarray(function(point + step * unit), dtype=float)
        - np.asarray(function(point - step * unit), dtype=float)
    ).ravel() / (2.0 * step)
    analytic = matrix @ unit.ravel()
    return _relative(analytic - finite_difference, analytic, finite_difference)


__all__ = (
    "FAST_CHART_INDICES",
    "FAST_CHART_SCALES",
    "FAST_EQUATION_ROWS",
    "FixedSlowEvaluation",
    "FixedSlowReconstruction",
    "SLOW_CHART_INDICES",
    "SLOW_EXACT_ROWS",
    "directional_jacobian_relative_defect",
    "generalized_maxwell_cattaneo_fast_charts",
    "generalized_maxwell_cattaneo_fast_rate_scales",
    "generalized_maxwell_cattaneo_projected_fast_evaluation",
    "generalized_maxwell_cattaneo_reconstruct_fixed_slow",
    "generalized_maxwell_cattaneo_slow_targets",
    "radius_one_colored_jacobian",
)
