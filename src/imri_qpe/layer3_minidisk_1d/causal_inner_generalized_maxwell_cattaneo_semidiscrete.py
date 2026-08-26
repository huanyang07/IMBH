"""Local sources and a fixed-geometry periodic seven-field cell operator.

This module is the nonpropagating semidiscrete audit layer selected after the
isolated-interface certificate.  It supplies the lower-order source in the
same seven equation rows as the generalized Maxwell--Cattaneo principal and
assembles a first-order periodic DLM fluctuation operator.  Radial boundary
conditions and trajectory integration intentionally remain outside its API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C, DEFAULT_KAPPA_ES

from .causal_inner_generalized_maxwell_cattaneo import (
    GeneralizedMaxwellCattaneoSourceLedger,
    audit_generalized_maxwell_cattaneo_source_ledger,
    generalized_maxwell_cattaneo_local_state,
    generalized_maxwell_cattaneo_principal,
)
from .causal_inner_generalized_maxwell_cattaneo_spatial import (
    generalized_maxwell_cattaneo_signed_fluctuations,
)
from .causal_inner_geometry import (
    KerrSchildColumnGeometry,
    ValenciaPerfectFluidPrimitive,
    audit_kerr_schild_column_sources,
)
from .causal_inner_recovery import FixedHeightGasRadiationColumnEOS
from .causal_inner_stress import (
    causal_rest_frame_shear_rate,
    causal_stress_column_state,
)
from .causal_inner_thermal import (
    QuasiHydrostaticGasRadiationColumnEOS,
    causal_comoving_energy_source,
    causal_diffusion_cooling_rate,
)


_N_FIELDS = 7
_EXACT_ROWS = np.asarray((0, 1, 2, 3, 5, 6), dtype=int)


@dataclass(frozen=True)
class GeneralizedMaxwellCattaneoLowerSource:
    """One complete local lower-order source in equation-row order."""

    source_per_cm: np.ndarray
    perfect_fluid_geometry_source_per_cm: np.ndarray
    stress_geometry_source_per_cm: np.ndarray
    radiative_cooling_source_per_cm: np.ndarray
    shear_relaxation_source_per_cm: np.ndarray
    height_material_source_per_cm: np.ndarray
    vertical_momentum_source_per_cm: np.ndarray
    connection_shear_rate_per_second: float
    vertical_acceleration_cm_per_s2: float
    hydrostatic_force_acceleration_cm_per_s2: float
    scattering_optical_depth: float
    source_ledger: GeneralizedMaxwellCattaneoSourceLedger


@dataclass(frozen=True)
class GeneralizedMaxwellCattaneoPeriodicOperator:
    """Fixed-geometry first-order periodic DLM semidiscrete evaluation."""

    primitive_charts: np.ndarray
    cell_spacing_cm: float
    equation_sources_per_cm: np.ndarray
    interface_total_jumps_over_c: np.ndarray
    interface_negative_fluctuations_over_c: np.ndarray
    interface_positive_fluctuations_over_c: np.ndarray
    spatial_equation_residuals_per_cm: np.ndarray
    equation_right_hand_sides_per_cm: np.ndarray
    primitive_rates_per_ct: np.ndarray
    temporal_solve_relative_residuals: np.ndarray
    global_exact_flux_ledger_relative_defect: float
    maximum_interface_split_relative_defect: float


def _chart(values) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (_N_FIELDS,) or np.any(~np.isfinite(result)):
        raise ValueError("semidiscrete chart must be finite and length seven")
    if result[1] ** 2 + result[2] ** 2 >= 1.0 or abs(float(result[6])) >= 1.0:
        raise ValueError("semidiscrete chart velocity must be subluminal")
    return result


def generalized_maxwell_cattaneo_hydrostatic_embedding(
    five_field_chart,
    *,
    proper_vertical_frequency: float,
) -> np.ndarray:
    """Embed a five-field chart on the dynamic-height equilibrium manifold."""

    chart5 = np.asarray(five_field_chart, dtype=float)
    if chart5.shape != (5,) or np.any(~np.isfinite(chart5)):
        raise ValueError("five-field chart must be finite and length five")
    eos = QuasiHydrostaticGasRadiationColumnEOS(
        proper_vertical_frequency=float(proper_vertical_frequency)
    )
    thermodynamics = eos.from_surface_density_temperature(
        float(np.exp(chart5[0])), float(np.exp(chart5[3]))
    )
    return np.concatenate(
        (
            chart5,
            [np.log(thermodynamics.proper_half_thickness), 0.0],
        )
    )


def _primitive_and_stress(
    geometry: KerrSchildColumnGeometry,
    chart: np.ndarray,
    *,
    proper_vertical_frequency: float,
):
    sigma = float(np.exp(chart[0]))
    temperature = float(np.exp(chart[3]))
    height = float(np.exp(chart[5]))
    beta_h = float(chart[6])
    eos = FixedHeightGasRadiationColumnEOS(proper_half_thickness=height)
    thermodynamics = eos.from_surface_density_temperature(sigma, temperature)
    vertical_energy = 0.5 * C**2 * (
        beta_h**2 + (float(proper_vertical_frequency) * height / C) ** 2
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=sigma,
        radial_velocity_over_c=float(chart[1]),
        azimuthal_velocity_over_c=float(chart[2]),
        specific_internal_energy=float(
            thermodynamics.specific_internal_energy + vertical_energy
        ),
        integrated_pressure=float(thermodynamics.integrated_pressure),
    )
    stress = causal_stress_column_state(
        geometry, primitive, specific_stress=float(chart[4])
    )
    return thermodynamics, primitive, stress


def generalized_maxwell_cattaneo_lower_source(
    geometry: KerrSchildColumnGeometry,
    chart,
    *,
    proper_vertical_frequency: float,
    alpha: float,
    stress_factor: float = 1.0,
    kappa: float = DEFAULT_KAPPA_ES,
    fast_vertical_multiplier: float = 1.0,
) -> GeneralizedMaxwellCattaneoLowerSource:
    """Return the frozen lower source without principal double counting."""

    values = _chart(chart)
    omega = float(proper_vertical_frequency)
    multiplier = float(fast_vertical_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("fast_vertical_multiplier must be positive")
    local = generalized_maxwell_cattaneo_local_state(
        geometry,
        values,
        proper_vertical_frequency=omega,
        alpha=alpha,
        stress_factor=stress_factor,
    )
    thermodynamics, primitive, stress = _primitive_and_stress(
        geometry, values, proper_vertical_frequency=omega
    )
    if not np.array_equal(
        np.asarray(stress.killing_conserved, dtype=float),
        np.asarray(local.conservative_state6[:4], dtype=float),
    ):
        raise RuntimeError("local source physical state does not match principal state")

    source = np.zeros(_N_FIELDS, dtype=float)
    perfect_component = np.zeros(_N_FIELDS, dtype=float)
    stress_component = np.zeros(_N_FIELDS, dtype=float)
    cooling_component = np.zeros(_N_FIELDS, dtype=float)
    shear_component = np.zeros(_N_FIELDS, dtype=float)
    height_component = np.zeros(_N_FIELDS, dtype=float)
    vertical_component = np.zeros(_N_FIELDS, dtype=float)

    perfect = audit_kerr_schild_column_sources(geometry, primitive)
    perfect_component[1] = float(perfect.radial_momentum_source)
    stress_component[1] = float(stress.radial_geometric_source_increment)
    cooling_rate, optical_depth = causal_diffusion_cooling_rate(
        thermodynamics, kappa=float(kappa)
    )
    cooling = causal_comoving_energy_source(
        geometry, primitive, comoving_energy_rate=-float(cooling_rate)
    )
    cooling_component[:4] = cooling.killing_source_per_ct
    source += perfect_component + stress_component + cooling_component

    connection_shear = causal_rest_frame_shear_rate(
        geometry,
        primitive,
        radial_lower_four_velocity_derivative=np.zeros(3, dtype=float),
    )
    chi = float(values[4])
    shear_component[4] = (
        local.specific_viscosity_seconds
        / local.relaxation_time_seconds
        * connection_shear
        - chi / local.relaxation_time_seconds
    ) / C
    source += shear_component

    sigma = float(local.surface_density)
    height = float(local.proper_half_thickness)
    beta_h = float(values[6])
    u0 = float(local.four_velocity[0])
    rest_mass = float(local.conservative_state6[0])
    pressure_acceleration = float(local.integrated_pressure / (sigma * height))
    gravity_acceleration = float(omega**2 * height)
    hydrostatic_acceleration = pressure_acceleration - gravity_acceleration
    damping_rate = float(alpha) * omega
    acceleration = (
        multiplier**2 * hydrostatic_acceleration
        - multiplier * damping_rate * C * beta_h
    )
    height_component[5] = rest_mass * beta_h / u0
    vertical_component[6] = rest_mass * acceleration / (C * u0)
    source += height_component + vertical_component

    ledger = audit_generalized_maxwell_cattaneo_source_ledger(
        local, alpha=float(alpha)
    )
    return GeneralizedMaxwellCattaneoLowerSource(
        source_per_cm=source,
        perfect_fluid_geometry_source_per_cm=perfect_component,
        stress_geometry_source_per_cm=stress_component,
        radiative_cooling_source_per_cm=cooling_component,
        shear_relaxation_source_per_cm=shear_component,
        height_material_source_per_cm=height_component,
        vertical_momentum_source_per_cm=vertical_component,
        connection_shear_rate_per_second=float(connection_shear),
        vertical_acceleration_cm_per_s2=float(acceleration),
        hydrostatic_force_acceleration_cm_per_s2=float(hydrostatic_acceleration),
        scattering_optical_depth=float(optical_depth),
        source_ledger=ledger,
    )


def _relative_maximum(defect, *references) -> float:
    scale = max(
        *(float(np.max(np.abs(np.asarray(value)))) for value in references),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(np.asarray(defect))) / scale)


def generalized_maxwell_cattaneo_periodic_operator(
    geometry: KerrSchildColumnGeometry,
    primitive_charts,
    *,
    cell_spacing_cm: float,
    proper_vertical_frequency: float,
    alpha: float,
    stress_factor: float = 1.0,
    include_lower_sources: bool = True,
    fast_vertical_multiplier: float = 1.0,
    quadrature_order: int = 8,
) -> GeneralizedMaxwellCattaneoPeriodicOperator:
    """Evaluate the first-order periodic path-conservative cell operator."""

    charts = np.asarray(primitive_charts, dtype=float)
    spacing = float(cell_spacing_cm)
    if (
        charts.ndim != 2
        or charts.shape[0] < 3
        or charts.shape[1:] != (_N_FIELDS,)
        or np.any(~np.isfinite(charts))
        or not np.isfinite(spacing)
        or spacing <= 0.0
    ):
        raise ValueError("periodic operator inputs are invalid")
    n_cells = charts.shape[0]
    common = {
        "proper_vertical_frequency": float(proper_vertical_frequency),
        "alpha": float(alpha),
        "stress_factor": float(stress_factor),
        "quadrature_order": int(quadrature_order),
    }
    jumps = np.empty((n_cells, _N_FIELDS), dtype=float)
    negative = np.empty_like(jumps)
    positive = np.empty_like(jumps)
    maximum_split = 0.0
    for interface in range(n_cells):
        split = generalized_maxwell_cattaneo_signed_fluctuations(
            geometry,
            charts[interface],
            charts[(interface + 1) % n_cells],
            **common,
        )
        jumps[interface] = split.path_jump.total_principal_jump_over_c
        negative[interface] = split.negative_fluctuation_over_c
        positive[interface] = split.positive_fluctuation_over_c
        maximum_split = max(
            maximum_split, float(split.split_closure_relative_defect)
        )
    spatial = np.empty_like(jumps)
    for cell in range(n_cells):
        spatial[cell] = (
            positive[(cell - 1) % n_cells] + negative[cell]
        ) / spacing
    sources = np.zeros_like(spatial)
    if include_lower_sources:
        for cell in range(n_cells):
            sources[cell] = generalized_maxwell_cattaneo_lower_source(
                geometry,
                charts[cell],
                proper_vertical_frequency=proper_vertical_frequency,
                alpha=alpha,
                stress_factor=stress_factor,
                fast_vertical_multiplier=fast_vertical_multiplier,
            ).source_per_cm
    right_hand_sides = sources - spatial
    rates = np.empty_like(charts)
    solve_defects = np.empty(n_cells, dtype=float)
    for cell in range(n_cells):
        principal = generalized_maxwell_cattaneo_principal(
            geometry,
            charts[cell],
            proper_vertical_frequency=proper_vertical_frequency,
            alpha=alpha,
            stress_factor=stress_factor,
        )
        scaled_rhs = right_hand_sides[cell] / principal.equation_row_scales
        scaled_rate = np.linalg.solve(
            principal.scaled_temporal_matrix, scaled_rhs
        )
        rates[cell] = principal.primitive_column_scales * scaled_rate
        solve_defects[cell] = _relative_maximum(
            principal.temporal_matrix @ rates[cell] - right_hand_sides[cell],
            principal.temporal_matrix @ rates[cell],
            right_hand_sides[cell],
        )
    exact_sum = np.sum(spatial[:, _EXACT_ROWS], axis=0)
    exact_ledger = _relative_maximum(
        exact_sum,
        spatial[:, _EXACT_ROWS],
    )
    return GeneralizedMaxwellCattaneoPeriodicOperator(
        primitive_charts=np.array(charts, copy=True),
        cell_spacing_cm=spacing,
        equation_sources_per_cm=sources,
        interface_total_jumps_over_c=jumps,
        interface_negative_fluctuations_over_c=negative,
        interface_positive_fluctuations_over_c=positive,
        spatial_equation_residuals_per_cm=spatial,
        equation_right_hand_sides_per_cm=right_hand_sides,
        primitive_rates_per_ct=rates,
        temporal_solve_relative_residuals=solve_defects,
        global_exact_flux_ledger_relative_defect=exact_ledger,
        maximum_interface_split_relative_defect=maximum_split,
    )


__all__ = (
    "GeneralizedMaxwellCattaneoLowerSource",
    "GeneralizedMaxwellCattaneoPeriodicOperator",
    "generalized_maxwell_cattaneo_hydrostatic_embedding",
    "generalized_maxwell_cattaneo_lower_source",
    "generalized_maxwell_cattaneo_periodic_operator",
)
