from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    conservatively_map_global_profile,
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
