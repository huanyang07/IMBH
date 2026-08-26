"""Analytic vertical-equilibrium reconstruction for the reduced slow model.

The online macro state keeps exact cellwise mass, angular momentum, and total
energy together with radial drift and causal shear stress.  Only the local
vertical height/momentum pair is eliminated.  This module deliberately does
not solve, or assume the existence of, a global fixed-slow fast root.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_generalized_maxwell_cattaneo import (
    generalized_maxwell_cattaneo_local_state,
)
from .causal_inner_generalized_maxwell_cattaneo_semidiscrete import (
    generalized_maxwell_cattaneo_hydrostatic_embedding,
)
from .causal_inner_generalized_maxwell_cattaneo_slow_manifold import (
    SLOW_EXACT_ROWS,
)
from .causal_inner_geometry import kerr_schild_column_geometry


_LOCAL_UNKNOWN_INDICES = np.asarray((0, 2, 3), dtype=int)
_LOCAL_UNKNOWN_SCALES = np.asarray((1.0, 0.1, 1.0), dtype=float)
_DERIVATIVE_STEPS = np.asarray((2.0e-6, 2.0e-7, 2.0e-6), dtype=float)


@dataclass(frozen=True)
class HydrostaticInvariantReconstruction:
    """Seven-field state reconstructed from ``(M,J,E,beta_r,chi)``."""

    primitive_charts: np.ndarray
    slow_targets: np.ndarray
    radial_velocity_over_c: np.ndarray
    specific_shear_stress: np.ndarray
    maximum_constraint_relative_defect: float
    maximum_newton_corrections: int
    maximum_scaled_local_inverse_condition_number: float
    maximum_scaled_unknown_correction: float


def _hydrostatic_chart(
    values: np.ndarray,
    *,
    radial_velocity_over_c: float,
    specific_shear_stress: float,
    proper_vertical_frequency: float,
) -> np.ndarray:
    chart5 = np.asarray(
        (
            values[0],
            radial_velocity_over_c,
            values[1],
            values[2],
            specific_shear_stress,
        ),
        dtype=float,
    )
    return generalized_maxwell_cattaneo_hydrostatic_embedding(
        chart5,
        proper_vertical_frequency=proper_vertical_frequency,
    )


def reconstruct_hydrostatic_fixed_invariants(
    context,
    slow_targets,
    radial_velocity_over_c,
    specific_shear_stress,
    *,
    template_charts,
    constraint_tolerance: float = 1.0e-10,
    maximum_newton_corrections: int = 8,
) -> HydrostaticInvariantReconstruction:
    """Recover the vertical-equilibrium seven-field state cell by cell.

    Each independent three-by-three solve adjusts ``(lnSigma,beta_phi,lnT)``
    so the exact seven-field mass, angular momentum, and total energy equal
    the supplied cell-integrated targets.  ``beta_r`` and ``chi`` are resolved
    macro auxiliaries and are never set by this reconstruction.
    """

    targets = np.asarray(slow_targets, dtype=float)
    radial = np.asarray(radial_velocity_over_c, dtype=float)
    stress = np.asarray(specific_shear_stress, dtype=float)
    template = np.asarray(template_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    tolerance = float(constraint_tolerance)
    maximum = int(maximum_newton_corrections)
    if (
        targets.shape != (n_cells, 3)
        or radial.shape != (n_cells,)
        or stress.shape != (n_cells,)
        or template.shape != (n_cells, 7)
        or np.any(~np.isfinite(targets))
        or np.any(~np.isfinite(radial))
        or np.any(~np.isfinite(stress))
        or np.any(~np.isfinite(template))
        or np.any(np.abs(radial) >= 1.0)
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
        or maximum < 0
    ):
        raise ValueError("hydrostatic invariant reconstruction inputs are invalid")

    charts = np.empty_like(template)
    greatest_defect = 0.0
    greatest_corrections = 0
    greatest_condition = 0.0
    greatest_scaled_correction = 0.0
    for cell, radius_value in enumerate(np.asarray(context.grid.centers, dtype=float)):
        radius = float(radius_value)
        omega = float(context.vertical_frequency.frequency(radius))
        geometry = kerr_schild_column_geometry(
            radius, context.grid.gravitational_radius
        )
        target = targets[cell]
        target_scale = np.maximum(np.abs(target), np.finfo(float).tiny)
        measure = float(context.grid.cell_measures[cell])

        def chart_from_unknowns(unknowns: np.ndarray) -> np.ndarray:
            return _hydrostatic_chart(
                np.asarray(unknowns, dtype=float),
                radial_velocity_over_c=float(radial[cell]),
                specific_shear_stress=float(stress[cell]),
                proper_vertical_frequency=omega,
            )

        def residual(unknowns: np.ndarray) -> np.ndarray:
            chart = chart_from_unknowns(unknowns)
            state = generalized_maxwell_cattaneo_local_state(
                geometry,
                chart,
                proper_vertical_frequency=omega,
                alpha=float(context.alpha),
                stress_factor=float(context.stress_factor),
            )
            value = measure * state.conservative_state6[SLOW_EXACT_ROWS]
            return (value - target) / target_scale

        unknowns = np.asarray(
            template[cell, _LOCAL_UNKNOWN_INDICES], dtype=float
        ).copy()
        values = residual(unknowns)
        corrections = 0
        local_condition = 0.0
        while float(np.max(np.abs(values))) > tolerance:
            if corrections >= maximum:
                raise RuntimeError(
                    "hydrostatic invariant local reconstruction did not converge"
                )
            jacobian = np.empty((3, 3), dtype=float)
            for column, step in enumerate(_DERIVATIVE_STEPS):
                direction = np.zeros(3, dtype=float)
                direction[column] = step
                jacobian[:, column] = (
                    residual(unknowns + direction)
                    - residual(unknowns - direction)
                ) / (2.0 * step)
            scaled_jacobian = jacobian * _LOCAL_UNKNOWN_SCALES[None, :]
            local_condition = max(
                local_condition, float(np.linalg.cond(scaled_jacobian))
            )
            correction = np.linalg.solve(jacobian, -values)
            old_norm = float(np.max(np.abs(values)))
            accepted = False
            for exponent in range(11):
                factor = 2.0 ** (-exponent)
                candidate = unknowns + factor * correction
                try:
                    candidate_values = residual(candidate)
                except (ValueError, FloatingPointError, OverflowError):
                    continue
                if float(np.max(np.abs(candidate_values))) < old_norm:
                    unknowns = candidate
                    values = candidate_values
                    greatest_scaled_correction = max(
                        greatest_scaled_correction,
                        float(
                            np.max(
                                np.abs(
                                    factor
                                    * correction
                                    / _LOCAL_UNKNOWN_SCALES
                                )
                            )
                        ),
                    )
                    accepted = True
                    break
            if not accepted:
                raise RuntimeError(
                    "hydrostatic invariant local reconstruction line search failed"
                )
            corrections += 1
        charts[cell] = chart_from_unknowns(unknowns)
        greatest_defect = max(greatest_defect, float(np.max(np.abs(values))))
        greatest_corrections = max(greatest_corrections, corrections)
        greatest_condition = max(greatest_condition, local_condition)
    return HydrostaticInvariantReconstruction(
        primitive_charts=charts,
        slow_targets=np.array(targets, copy=True),
        radial_velocity_over_c=np.array(radial, copy=True),
        specific_shear_stress=np.array(stress, copy=True),
        maximum_constraint_relative_defect=greatest_defect,
        maximum_newton_corrections=greatest_corrections,
        maximum_scaled_local_inverse_condition_number=greatest_condition,
        maximum_scaled_unknown_correction=greatest_scaled_correction,
    )


def hydrostatic_invariant_local_scaled_jacobian(
    context,
    cell: int,
    primitive_chart,
) -> np.ndarray:
    """Differentiate local ``(M,J,E)`` with respect to scaled slow charts.

    Rows are normalized by the absolute local invariant values and columns
    correspond to ``(lnSigma,beta_phi,lnT)/(1,0.1,1)``.  The inverse of this
    matrix is the local tangent used by a sparse implicit macro solver.
    """

    index = int(cell)
    chart = np.asarray(primitive_chart, dtype=float)
    n_cells = int(context.grid.centers.size)
    if (
        index < 0
        or index >= n_cells
        or chart.shape != (7,)
        or np.any(~np.isfinite(chart))
    ):
        raise ValueError("local hydrostatic invariant Jacobian inputs are invalid")
    radius = float(context.grid.centers[index])
    omega = float(context.vertical_frequency.frequency(radius))
    geometry = kerr_schild_column_geometry(
        radius, context.grid.gravitational_radius
    )
    measure = float(context.grid.cell_measures[index])
    unknowns = np.asarray(chart[_LOCAL_UNKNOWN_INDICES], dtype=float)

    def values(candidate: np.ndarray) -> np.ndarray:
        candidate_chart = _hydrostatic_chart(
            np.asarray(candidate, dtype=float),
            radial_velocity_over_c=float(chart[1]),
            specific_shear_stress=float(chart[4]),
            proper_vertical_frequency=omega,
        )
        state = generalized_maxwell_cattaneo_local_state(
            geometry,
            candidate_chart,
            proper_vertical_frequency=omega,
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
        )
        return measure * state.conservative_state6[SLOW_EXACT_ROWS]

    base = values(unknowns)
    scale = np.maximum(np.abs(base), np.finfo(float).tiny)
    jacobian = np.empty((3, 3), dtype=float)
    for column, step in enumerate(_DERIVATIVE_STEPS):
        direction = np.zeros(3, dtype=float)
        direction[column] = step
        jacobian[:, column] = (
            values(unknowns + direction) - values(unknowns - direction)
        ) / (2.0 * step)
    return jacobian / scale[:, None] * _LOCAL_UNKNOWN_SCALES[None, :]


__all__ = (
    "HydrostaticInvariantReconstruction",
    "hydrostatic_invariant_local_scaled_jacobian",
    "reconstruct_hydrostatic_fixed_invariants",
)
