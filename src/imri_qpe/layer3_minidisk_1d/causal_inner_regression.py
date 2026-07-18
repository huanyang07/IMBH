"""Shared source-compatible regression context for causal evolution."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from imri_qpe.constants import C, G
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from .causal_inner_dae_system import CausalFiveFieldDAEContext
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
