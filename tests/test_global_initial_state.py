from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    conservatively_map_global_profile,
    construct_global_constant_pressure_startup,
    evaluate_global_rusanov_profile,
    global_conservative_rhs,
    recover_global_primitives,
)
from imri_qpe.parameters import FiducialParams


def test_conservative_global_profile_mapping_preserves_annular_mass() -> None:
    mass = FiducialParams().M2_g
    potential = PaczynskiWiitaPotential(mass)
    radius = np.geomspace(4.5 * potential.r_g, 335.0 * potential.r_g, 256)
    sigma = np.full_like(radius, 125.0)
    velocity = np.full_like(radius, -2.0e6)
    omega = np.asarray(potential.omega_k(radius), dtype=float)
    temperature = np.full_like(radius, 2.0e6)

    grid, state, correction = conservatively_map_global_profile(
        radius,
        sigma,
        velocity,
        omega,
        temperature,
        mass,
        64,
    )
    expected_mass = np.pi * sigma[0] * (
        grid.edges[-1] ** 2 - grid.edges[0] ** 2
    )
    assert np.isclose(np.sum(state.mass), expected_mass, rtol=2.0e-13)
    primitives = recover_global_primitives(
        grid,
        state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    assert np.all(primitives.surface_density > 0.0)
    assert np.all(primitives.temperature > 0.0)
    assert np.all(np.isfinite(correction))


def test_conservative_global_profile_mapping_rejects_extrapolated_bounds() -> None:
    mass = FiducialParams().M2_g
    potential = PaczynskiWiitaPotential(mass)
    radius = np.geomspace(6.0 * potential.r_g, 20.0 * potential.r_g, 8)
    values = np.ones_like(radius)
    with pytest.raises(ValueError):
        conservatively_map_global_profile(
            radius,
            values,
            -values,
            values,
            values,
            mass,
            8,
            inner_radius=5.0 * potential.r_g,
        )


def test_constant_pressure_startup_is_a_discrete_radial_equilibrium() -> None:
    mass = FiducialParams().M2_g
    potential = PaczynskiWiitaPotential(mass)
    grid, state, correction, audit = (
        construct_global_constant_pressure_startup(
            mass,
            32,
            inner_radius=4.5 * potential.r_g,
            outer_radius=335.0 * potential.r_g,
            aspect_ratio=0.05,
            minimum_scattering_optical_depth=10.0,
        )
    )
    primitives = recover_global_primitives(
        grid,
        state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    np.testing.assert_allclose(
        primitives.omega,
        potential.omega_k(grid.centers),
        rtol=2.0e-13,
    )
    np.testing.assert_array_equal(
        primitives.radial_velocity, np.zeros(grid.centers.size)
    )
    assert audit.minimum_scattering_optical_depth >= 10.0
    assert audit.maximum_relative_pressure_defect < 5.0e-11
    assert audit.maximum_relative_aspect_ratio_defect < 3.0e-11

    profile = evaluate_global_rusanov_profile(
        grid,
        state,
        mass,
        reference_state=state,
        specific_mechanical_energy_correction=correction,
    )
    rhs = global_conservative_rhs(
        profile.face_fluxes, profile.cell_sources
    )
    np.testing.assert_array_equal(rhs.mass, np.zeros(grid.centers.size))
    np.testing.assert_array_equal(
        rhs.angular_momentum, np.zeros(grid.centers.size)
    )
    np.testing.assert_array_equal(
        rhs.total_energy, np.zeros(grid.centers.size)
    )
    force_scale = np.max(np.abs(profile.cell_sources.radial_momentum))
    assert np.max(np.abs(rhs.radial_momentum)) <= 2.0e-9 * force_scale
