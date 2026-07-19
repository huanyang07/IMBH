"""Shared source-compatible regression context for causal evolution."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from imri_qpe.constants import C, G
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    make_causal_five_field_seed,
)
from .causal_inner_geometry import (
    ValenciaPerfectFluidPrimitive,
    kerr_schild_column_geometry,
    make_kerr_schild_column_grid,
)
from .causal_inner_migration import (
    KERR_SCHILD_HILL_ENERGY_ZERO,
    SchwarzschildCurvatureVerticalFrequency,
    exact_kerr_schild_compact_stream_sources,
    kerr_schild_stream_injection,
)
from .hill_roche_nozzle import (
    GasRadiationHillRocheNozzleProvider,
    fiducial_hill_roche_nozzle_geometry,
)


CAUSAL_REGRESSION_STREAM_CENTER_RG = 240.0
CAUSAL_REGRESSION_STREAM_LOG_WIDTH = 0.08
CAUSAL_REGRESSION_STREAM_MDOT_EDD = 5.0
CAUSAL_REGRESSION_STREAM_SURFACE_DENSITY = 1.0e5
CAUSAL_REGRESSION_STREAM_TEMPERATURE = 1.0e6


def make_causal_five_field_regression_context(
    n_cells: int,
    *,
    spatial_reconstruction: str = "piecewise_constant",
) -> CausalFiveFieldDAEContext:
    """Return the exact circularized C2 context used since WP10c5q."""

    if int(n_cells) != n_cells or n_cells < 2:
        raise ValueError("causal regression n_cells must be at least two")
    mass = FiducialParams().M2_g
    gravitational_radius = G * mass / C**2
    grid = make_kerr_schild_column_grid(
        1.8 * gravitational_radius,
        335.0 * gravitational_radius,
        int(n_cells),
        gravitational_radius,
    )
    nozzle_geometry = replace(
        fiducial_hill_roche_nozzle_geometry(),
        energy_zero=KERR_SCHILD_HILL_ENERGY_ZERO,
    )
    context = CausalFiveFieldDAEContext(
        grid=grid,
        vertical_frequency=SchwarzschildCurvatureVerticalFrequency(
            gravitational_radius
        ),
        outer_boundary_provider=GasRadiationHillRocheNozzleProvider(
            nozzle_geometry,
            transverse_quadrature_zones=24,
        ),
        include_radiative_cooling=True,
        spatial_reconstruction=spatial_reconstruction,
    ).validated()
    radius = CAUSAL_REGRESSION_STREAM_CENTER_RG * gravitational_radius
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    thermodynamics = context.vertical_frequency.eos(
        radius
    ).from_surface_density_temperature(
        CAUSAL_REGRESSION_STREAM_SURFACE_DENSITY,
        CAUSAL_REGRESSION_STREAM_TEMPERATURE,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=CAUSAL_REGRESSION_STREAM_SURFACE_DENSITY,
        radial_velocity_over_c=(
            2.0 * gravitational_radius / radius
        ),
        azimuthal_velocity_over_c=float(
            np.sqrt(gravitational_radius / radius)
            / geometry.base.lapse
        ),
        specific_internal_energy=(
            thermodynamics.specific_internal_energy
        ),
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    injection = kerr_schild_stream_injection(
        geometry,
        primitive,
        rest_mass_rate=(
            CAUSAL_REGRESSION_STREAM_MDOT_EDD
            * eddington_mdot(mass)
        ),
    )
    stream_sources = exact_kerr_schild_compact_stream_sources(
        context.grid,
        injection,
        center=radius,
        log_width=CAUSAL_REGRESSION_STREAM_LOG_WIDTH,
        shape="compact_c2",
    )
    return replace(
        context,
        stream_sources=stream_sources,
    ).validated()


def causal_five_field_regression_seed_parameters(
    context: CausalFiveFieldDAEContext,
) -> dict:
    """Return the shared source-compatible C2 continuum seed parameters."""

    context = context.validated()
    if context.stream_sources is None:
        raise ValueError("causal regression seed requires a stream")
    gravitational_radius = context.grid.gravitational_radius
    inner_plateau = 6.0 * gravitational_radius
    outer_plateau = (
        CAUSAL_REGRESSION_STREAM_CENTER_RG * gravitational_radius
    )
    source_rate = float(np.sum(context.stream_sources.rest_mass))
    unit_state = make_causal_five_field_seed(
        context,
        inner_surface_density=1.0,
        inner_temperature=1.0e6,
        profile_inner_plateau_radius=inner_plateau,
        profile_outer_plateau_radius=outer_plateau,
    )
    unit_inner_rate = float(
        C * unit_state.weighted_face_fluxes_over_c[0, 0]
    )
    if unit_inner_rate >= 0.0:
        raise ValueError("causal regression seed requires inner inflow")
    inner_surface_density = source_rate / abs(unit_inner_rate)

    target_h_over_r = 0.1
    inner_radius = float(context.grid.edges[0])
    eos = context.vertical_frequency.eos(inner_radius)

    def thickness(log_temperature: float) -> float:
        column = eos.from_surface_density_temperature(
            inner_surface_density,
            float(np.exp(log_temperature)),
        )
        return column.proper_half_thickness / inner_radius

    lower = float(np.log(1.0e5))
    upper = float(np.log(1.0e7))
    if (
        thickness(lower) >= target_h_over_r
        or thickness(upper) <= target_h_over_r
    ):
        raise ValueError("causal regression thickness target is unbracketed")
    for _iteration in range(80):
        middle = 0.5 * (lower + upper)
        if thickness(middle) < target_h_over_r:
            lower = middle
        else:
            upper = middle
    return {
        "inner_surface_density": inner_surface_density,
        "outer_surface_density": (
            CAUSAL_REGRESSION_STREAM_SURFACE_DENSITY
        ),
        "inner_temperature": float(
            np.exp(0.5 * (lower + upper))
        ),
        "outer_temperature": 8.0e5,
        "inner_radial_velocity_over_c": -0.40,
        "inner_azimuthal_velocity_over_c": 0.60,
        "outer_radial_velocity_margin_over_c": 1.0e-5,
        "profile_inner_plateau_radius": inner_plateau,
        "profile_outer_plateau_radius": outer_plateau,
        "profile_interpolate_log_h_over_r": True,
    }
