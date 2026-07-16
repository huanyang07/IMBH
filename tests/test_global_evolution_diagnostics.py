from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.constants import M_SUN
from imri_qpe.layer3_minidisk_1d.global_evolution_diagnostics import (
    global_fixed_radius_diagnostics,
    global_sonic_resolution_diagnostic,
)
from imri_qpe.layer3_minidisk_1d.global_signed_evolution import (
    GlobalFaceFluxes,
    global_effective_sound_speed,
    recover_global_primitives,
    state_from_thermodynamic_primitives,
)
from imri_qpe.layer3_minidisk_1d.grid import make_log_grid
from imri_qpe.layer3_minidisk_1d.transonic_potential import (
    PaczynskiWiitaPotential,
)


def _crossing_state(n_cells: int = 12):
    mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(4.5 * potential.r_g, 20.0 * potential.r_g, n_cells)
    sigma = 3.0e4 * (grid.centers / grid.centers[0]) ** -0.4
    temperature = 2.0e7 * (grid.centers / grid.centers[0]) ** -0.2
    omega = np.asarray(potential.omega_k(grid.centers), dtype=float)
    resting = state_from_thermodynamic_primitives(
        grid,
        sigma,
        np.zeros(n_cells),
        omega,
        temperature,
        mass,
    )
    resting_primitives = recover_global_primitives(grid, resting, mass)
    sound = global_effective_sound_speed(resting_primitives)
    radial_velocity = -sound * np.linspace(2.0, 0.5, n_cells)
    state = state_from_thermodynamic_primitives(
        grid,
        sigma,
        radial_velocity,
        omega,
        temperature,
        mass,
    )
    return mass, grid, recover_global_primitives(grid, state, mass)


def test_fixed_radius_diagnostics_share_one_physical_location() -> None:
    mass, grid, primitives = _crossing_state()
    face_count = grid.centers.size + 1
    fluxes = GlobalFaceFluxes(
        mass=np.linspace(-4.0, 2.0, face_count),
        radial_momentum=np.linspace(1.0, 3.0, face_count),
        angular_momentum=np.linspace(-8.0, 4.0, face_count),
        total_energy=np.linspace(-12.0, 6.0, face_count),
    )
    radius = float(grid.centers[3])
    diagnostic = global_fixed_radius_diagnostics(
        grid, primitives, fluxes, mass, (radius,)
    )[0]
    assert diagnostic.radius == radius
    assert diagnostic.surface_density == pytest.approx(
        primitives.surface_density[3], rel=2.0e-14
    )
    assert diagnostic.temperature == pytest.approx(
        primitives.temperature[3], rel=2.0e-14
    )
    assert diagnostic.omega_over_omega_k == pytest.approx(1.0, rel=2.0e-14)
    assert diagnostic.radial_mach_number < -1.0
    assert max(diagnostic.characteristic_speeds) < 0.0


def test_sonic_resolution_diagnostic_finds_innermost_crossing() -> None:
    _mass, grid, primitives = _crossing_state()
    diagnostic = global_sonic_resolution_diagnostic(grid, primitives)
    assert diagnostic.sonic_radius is not None
    assert grid.edges[0] < diagnostic.sonic_radius < grid.edges[-1]
    assert diagnostic.cells_between_inner_face_and_sonic_radius is not None
    assert diagnostic.cells_between_inner_face_and_sonic_radius > 0
    assert diagnostic.minimum_velocity_gradient_length_over_cell_width > 0.0
    assert diagnostic.minimum_velocity_gradient_length_over_H > 0.0
    assert 0 <= diagnostic.minimum_over_cell_width_index < grid.centers.size
    assert 0 <= diagnostic.minimum_over_H_index < grid.centers.size
    assert diagnostic.gradient_audit_outer_radius > diagnostic.sonic_radius
