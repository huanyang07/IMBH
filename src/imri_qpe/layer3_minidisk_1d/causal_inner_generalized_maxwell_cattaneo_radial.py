"""First-order radial seven-field DLM operator and bounded SSPRK2 step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_generalized_maxwell_cattaneo import (
    generalized_maxwell_cattaneo_local_state,
    generalized_maxwell_cattaneo_principal,
)
from .causal_inner_generalized_maxwell_cattaneo_semidiscrete import (
    generalized_maxwell_cattaneo_hydrostatic_embedding,
    generalized_maxwell_cattaneo_lower_source,
)
from .causal_inner_generalized_maxwell_cattaneo_spatial import (
    generalized_maxwell_cattaneo_signed_fluctuations,
)
from .causal_inner_geometry import kerr_schild_column_geometry


_N_FIELDS = 7
_EXACT_ROWS = np.asarray((0, 1, 2, 3, 5, 6), dtype=int)
_CHART_SCALES = np.asarray((1.0, 0.1, 0.1, 1.0, 1.0e-4, 1.0, 0.03))


@dataclass(frozen=True)
class GeneralizedMaxwellCattaneoRadialOperator:
    """One full first-order radial semidiscrete evaluation."""

    primitive_charts: np.ndarray
    outer_exterior_chart: np.ndarray
    weighted_shared_exact_fluxes_over_c: np.ndarray
    weighted_negative_fluctuations_over_c: np.ndarray
    weighted_positive_fluctuations_over_c: np.ndarray
    weighted_spatial_equation_residuals_over_c: np.ndarray
    weighted_equation_sources_per_ct: np.ndarray
    equation_right_hand_sides_per_cm: np.ndarray
    primitive_rates_per_ct: np.ndarray
    temporal_solve_relative_residuals: np.ndarray
    center_eigenvalues_over_c: np.ndarray
    maximum_imaginary_speed_over_c: float
    maximum_light_cone_excess_over_c: float
    maximum_eigenvector_condition_number: float
    maximum_CFL_for_timestep: float | None
    incoming_inner_characteristics: int
    incoming_outer_characteristics: int
    minimum_height_over_radius: float
    maximum_height_over_radius: float
    minimum_optical_depth: float
    exact_integrated_states: np.ndarray
    exact_global_boundary_source_rate_per_ct: np.ndarray


@dataclass(frozen=True)
class GeneralizedMaxwellCattaneoSSPRK2Step:
    """One audited SSPRK2 chart step and its two stage operators."""

    initial_charts: np.ndarray
    euler_stage_charts: np.ndarray
    accepted_charts: np.ndarray
    timestep_seconds: float
    initial_operator: GeneralizedMaxwellCattaneoRadialOperator
    euler_operator: GeneralizedMaxwellCattaneoRadialOperator
    accepted_operator: GeneralizedMaxwellCattaneoRadialOperator
    maximum_scaled_chart_change: float
    exact_flux_balance_relative_defect: float


def _exact_flux(local_state) -> np.ndarray:
    result = np.zeros(_N_FIELDS, dtype=float)
    result[:4] = local_state.conservative_flux6_over_c[:4]
    result[5:] = local_state.conservative_flux6_over_c[4:]
    return result


def _exact_state(local_state) -> np.ndarray:
    result = np.zeros(_N_FIELDS, dtype=float)
    result[:4] = local_state.conservative_state6[:4]
    result[5:] = local_state.conservative_state6[4:]
    return result


def _relative(defect, *references) -> float:
    scale = max(
        *(float(np.max(np.abs(np.asarray(item)))) for item in references),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(np.asarray(defect))) / scale)


def _outer_chart(context, charts: np.ndarray) -> np.ndarray:
    exterior = getattr(context, "outer_boundary_frozen_exterior_chart", None)
    mode = getattr(context, "outer_boundary_flux_mode", None)
    if mode != "frozen_exterior_rusanov" or exterior is None:
        raise ValueError("radial seven-field audit requires a frozen exterior chart")
    radius = float(context.grid.edges[-1])
    return generalized_maxwell_cattaneo_hydrostatic_embedding(
        np.asarray(exterior, dtype=float),
        proper_vertical_frequency=float(context.vertical_frequency.frequency(radius)),
    )


def generalized_maxwell_cattaneo_radial_operator(
    context,
    primitive_charts,
    *,
    timestep_seconds: float | None = None,
    quadrature_order: int = 8,
) -> GeneralizedMaxwellCattaneoRadialOperator:
    """Evaluate all cells and faces without advancing the supplied state."""

    charts = np.asarray(primitive_charts, dtype=float)
    grid = context.grid
    n_cells = int(np.asarray(grid.centers).size)
    if (
        charts.shape != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(charts))
        or np.any(charts[:, 1] ** 2 + charts[:, 2] ** 2 >= 1.0)
        or np.any(np.abs(charts[:, 6]) >= 1.0)
    ):
        raise ValueError("radial seven-field chart array is inadmissible")
    exterior = _outer_chart(context, charts)
    negative = np.zeros((n_cells + 1, _N_FIELDS), dtype=float)
    positive = np.zeros_like(negative)
    shared = np.zeros((n_cells + 1, len(_EXACT_ROWS)), dtype=float)
    face_max_speeds = np.zeros(n_cells + 1, dtype=float)
    max_imaginary = 0.0
    max_light = 0.0
    max_condition = 0.0

    def update_principal(principal, face: int) -> None:
        nonlocal max_imaginary, max_light, max_condition
        face_max_speeds[face] = float(
            np.max(np.abs(np.real(principal.eigenvalues_over_c)))
        )
        max_imaginary = max(max_imaginary, principal.maximum_imaginary_speed_over_c)
        max_light = max(max_light, principal.maximum_light_cone_excess_over_c)
        max_condition = max(max_condition, principal.eigenvector_condition_number)

    inner_radius = float(grid.edges[0])
    inner_geometry = kerr_schild_column_geometry(inner_radius, grid.gravitational_radius)
    inner_omega = float(context.vertical_frequency.frequency(inner_radius))
    inner_principal = generalized_maxwell_cattaneo_principal(
        inner_geometry,
        charts[0],
        proper_vertical_frequency=inner_omega,
        alpha=float(context.alpha),
        stress_factor=float(context.stress_factor),
    )
    update_principal(inner_principal, 0)
    inner_state = generalized_maxwell_cattaneo_local_state(
        inner_geometry,
        charts[0],
        proper_vertical_frequency=inner_omega,
        alpha=float(context.alpha),
        stress_factor=float(context.stress_factor),
    )
    shared[0] = float(grid.face_measures[0]) * _exact_flux(inner_state)[_EXACT_ROWS]
    incoming_inner = int(np.sum(np.real(inner_principal.eigenvalues_over_c) > 1.0e-12))

    for face in range(1, n_cells):
        radius = float(grid.edges[face])
        geometry = kerr_schild_column_geometry(radius, grid.gravitational_radius)
        split = generalized_maxwell_cattaneo_signed_fluctuations(
            geometry,
            charts[face - 1],
            charts[face],
            proper_vertical_frequency=float(context.vertical_frequency.frequency(radius)),
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
            quadrature_order=int(quadrature_order),
        )
        measure = float(grid.face_measures[face])
        negative[face] = measure * split.negative_fluctuation_over_c
        positive[face] = measure * split.positive_fluctuation_over_c
        shared[face] = measure * split.left_shared_exact_flux_over_c
        update_principal(split.midpoint_principal, face)

    outer_radius = float(grid.edges[-1])
    outer_geometry = kerr_schild_column_geometry(outer_radius, grid.gravitational_radius)
    outer_split = generalized_maxwell_cattaneo_signed_fluctuations(
        outer_geometry,
        charts[-1],
        exterior,
        proper_vertical_frequency=float(context.vertical_frequency.frequency(outer_radius)),
        alpha=float(context.alpha),
        stress_factor=float(context.stress_factor),
        quadrature_order=int(quadrature_order),
    )
    outer_measure = float(grid.face_measures[-1])
    negative[-1] = outer_measure * outer_split.negative_fluctuation_over_c
    positive[-1] = outer_measure * outer_split.positive_fluctuation_over_c
    shared[-1] = outer_measure * outer_split.left_shared_exact_flux_over_c
    update_principal(outer_split.midpoint_principal, n_cells)
    incoming_outer = int(
        np.sum(np.real(outer_split.midpoint_principal.eigenvalues_over_c) < -1.0e-12)
    )

    spatial = np.zeros((n_cells, _N_FIELDS), dtype=float)
    spatial[:, _EXACT_ROWS] = shared[1:] - shared[:-1]
    spatial[:, 4] = positive[:-1, 4] + negative[1:, 4]
    sources = np.zeros_like(spatial)
    rates = np.zeros_like(charts)
    solve_defects = np.zeros(n_cells, dtype=float)
    center_eigenvalues = np.empty((n_cells, _N_FIELDS), dtype=complex)
    exact_states = np.empty((n_cells, _N_FIELDS), dtype=float)
    height_ratios = np.empty(n_cells, dtype=float)
    optical_depths = np.empty(n_cells, dtype=float)
    for cell, radius in enumerate(np.asarray(grid.centers, dtype=float)):
        geometry = kerr_schild_column_geometry(float(radius), grid.gravitational_radius)
        omega = float(context.vertical_frequency.frequency(float(radius)))
        source = generalized_maxwell_cattaneo_lower_source(
            geometry,
            charts[cell],
            proper_vertical_frequency=omega,
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
            kappa=float(context.kappa),
        )
        measure = float(grid.cell_measures[cell])
        sources[cell] = measure * source.source_per_cm
        principal = generalized_maxwell_cattaneo_principal(
            geometry,
            charts[cell],
            proper_vertical_frequency=omega,
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
        )
        center_eigenvalues[cell] = principal.eigenvalues_over_c
        max_imaginary = max(max_imaginary, principal.maximum_imaginary_speed_over_c)
        max_light = max(max_light, principal.maximum_light_cone_excess_over_c)
        max_condition = max(max_condition, principal.eigenvector_condition_number)
        local = principal.local_state
        exact_states[cell] = measure * _exact_state(local)
        height_ratios[cell] = local.proper_half_thickness / float(radius)
        optical_depths[cell] = source.scattering_optical_depth
    stream = getattr(context, "stream_sources", None)
    if stream is not None:
        sources[:, :4] += np.asarray(stream.weighted_killing_source_per_ct, dtype=float)

    integrated_rhs = sources - spatial
    rhs_per_cm = integrated_rhs / np.asarray(grid.cell_measures)[:, None]
    for cell, radius in enumerate(np.asarray(grid.centers, dtype=float)):
        geometry = kerr_schild_column_geometry(float(radius), grid.gravitational_radius)
        principal = generalized_maxwell_cattaneo_principal(
            geometry,
            charts[cell],
            proper_vertical_frequency=float(context.vertical_frequency.frequency(float(radius))),
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
        )
        scaled_rhs = rhs_per_cm[cell] / principal.equation_row_scales
        scaled_rate = np.linalg.solve(principal.scaled_temporal_matrix, scaled_rhs)
        rates[cell] = principal.primitive_column_scales * scaled_rate
        solve_defects[cell] = _relative(
            principal.temporal_matrix @ rates[cell] - rhs_per_cm[cell],
            principal.temporal_matrix @ rates[cell],
            rhs_per_cm[cell],
        )
    cfl = None
    if timestep_seconds is not None:
        timestep = float(timestep_seconds)
        effective_widths = np.asarray(grid.cell_measures, dtype=float) / np.maximum(
            np.asarray(grid.face_measures[:-1], dtype=float),
            np.asarray(grid.face_measures[1:], dtype=float),
        )
        cell_speeds = np.maximum(face_max_speeds[:-1], face_max_speeds[1:])
        cfl = float(np.max(timestep * C * cell_speeds / effective_widths))
    boundary_source_rate = (
        np.sum(sources[:, _EXACT_ROWS], axis=0)
        - (shared[-1] - shared[0])
    )
    return GeneralizedMaxwellCattaneoRadialOperator(
        primitive_charts=np.array(charts, copy=True),
        outer_exterior_chart=exterior,
        weighted_shared_exact_fluxes_over_c=shared,
        weighted_negative_fluctuations_over_c=negative,
        weighted_positive_fluctuations_over_c=positive,
        weighted_spatial_equation_residuals_over_c=spatial,
        weighted_equation_sources_per_ct=sources,
        equation_right_hand_sides_per_cm=rhs_per_cm,
        primitive_rates_per_ct=rates,
        temporal_solve_relative_residuals=solve_defects,
        center_eigenvalues_over_c=center_eigenvalues,
        maximum_imaginary_speed_over_c=float(max_imaginary),
        maximum_light_cone_excess_over_c=float(max_light),
        maximum_eigenvector_condition_number=float(max_condition),
        maximum_CFL_for_timestep=cfl,
        incoming_inner_characteristics=incoming_inner,
        incoming_outer_characteristics=incoming_outer,
        minimum_height_over_radius=float(np.min(height_ratios)),
        maximum_height_over_radius=float(np.max(height_ratios)),
        minimum_optical_depth=float(np.min(optical_depths)),
        exact_integrated_states=exact_states,
        exact_global_boundary_source_rate_per_ct=np.asarray(boundary_source_rate),
    )


def generalized_maxwell_cattaneo_ssprk2_step(
    context,
    primitive_charts,
    *,
    timestep_seconds: float,
    quadrature_order: int = 8,
) -> GeneralizedMaxwellCattaneoSSPRK2Step:
    """Take one bounded explicit step; acceptance remains the caller's duty."""

    initial = np.asarray(primitive_charts, dtype=float)
    timestep = float(timestep_seconds)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("SSPRK2 timestep must be positive")
    first = generalized_maxwell_cattaneo_radial_operator(
        context, initial, timestep_seconds=timestep, quadrature_order=quadrature_order
    )
    euler = initial + timestep * C * first.primitive_rates_per_ct
    second = generalized_maxwell_cattaneo_radial_operator(
        context, euler, timestep_seconds=timestep, quadrature_order=quadrature_order
    )
    accepted = 0.5 * initial + 0.5 * (
        euler + timestep * C * second.primitive_rates_per_ct
    )
    final = generalized_maxwell_cattaneo_radial_operator(
        context, accepted, timestep_seconds=timestep, quadrature_order=quadrature_order
    )
    scaled_change = float(np.max(np.abs((accepted - initial) / _CHART_SCALES)))
    initial_total = np.sum(first.exact_integrated_states[:, _EXACT_ROWS], axis=0)
    final_total = np.sum(final.exact_integrated_states[:, _EXACT_ROWS], axis=0)
    expected_change = 0.5 * timestep * C * (
        first.exact_global_boundary_source_rate_per_ct
        + second.exact_global_boundary_source_rate_per_ct
    )
    balance = _relative(
        final_total - initial_total - expected_change,
        final_total - initial_total,
        expected_change,
    )
    return GeneralizedMaxwellCattaneoSSPRK2Step(
        initial_charts=np.array(initial, copy=True),
        euler_stage_charts=euler,
        accepted_charts=accepted,
        timestep_seconds=timestep,
        initial_operator=first,
        euler_operator=second,
        accepted_operator=final,
        maximum_scaled_chart_change=scaled_change,
        exact_flux_balance_relative_defect=balance,
    )


__all__ = (
    "GeneralizedMaxwellCattaneoRadialOperator",
    "GeneralizedMaxwellCattaneoSSPRK2Step",
    "generalized_maxwell_cattaneo_radial_operator",
    "generalized_maxwell_cattaneo_ssprk2_step",
)
