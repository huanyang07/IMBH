from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedFluxBoundary,
    SignedThermalClosure,
    advance_signed_thermal_implicit,
    make_log_grid,
    normalized_stream_cell_rates,
    signed_thermal_profile,
    solve_signed_flux_steady,
    solve_signed_thermal_steady,
    solve_signed_thermoviscous_steady,
)
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


def _supplied_disk(n: int = 48, outer_mode: str = "tidal_wall"):
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(6.1 * potential.r_g, 335.0 * potential.r_g, n)
    viscosity = 1.0e-4 * grid.centers**2 * potential.omega_k(grid.centers)
    source_mass, _ = normalized_stream_cell_rates(
        grid,
        5.0 * eddington_mdot(mass),
        center=240.0 * potential.r_g,
        log_width=0.08,
    )
    stream_l = float(potential.l_k(248.96693 * potential.r_g))
    stream_B = float(
        potential.phi(248.96693 * potential.r_g)
        + 0.5 * (stream_l / (248.96693 * potential.r_g)) ** 2
    )
    transport = solve_signed_flux_steady(
        grid,
        viscosity,
        mass,
        boundary=SignedFluxBoundary(outer_mode=outer_mode),
        source_mass_rate_cells=source_mass,
        source_specific_angular_momentum=np.full(n, stream_l),
    )
    closure = SignedThermalClosure(
        stream_specific_angular_momentum=stream_l,
        stream_specific_total_energy=stream_B,
        temperature_bounds=(1.0e3, 1.0e9),
    )
    return mass, grid, transport, closure


def test_signed_thermal_profile_closes_global_energy_ledger() -> None:
    mass, grid, transport, closure = _supplied_disk()
    profile = signed_thermal_profile(grid, transport, 1.0e6, mass, closure=closure)
    scale = max(
        float(np.sum(np.abs(profile.advective_rate_cells))),
        float(np.sum(profile.viscous_heating_rate_cells)),
        float(np.sum(profile.radiative_cooling_rate_cells)),
        1.0,
    )
    assert abs(profile.energy_budget_defect) / scale < 2.0e-15
    assert np.any(profile.stream_heating_rate_cells != 0.0)
    assert np.all(profile.thermal_energy_cells > 0.0)


def test_implicit_thermal_step_satisfies_cell_energy_equation() -> None:
    mass, grid, transport, closure = _supplied_disk(32)
    result = advance_signed_thermal_implicit(
        grid,
        transport,
        1.0e6,
        mass,
        1.0e3,
        closure=closure,
    )

    assert np.all(result.temperature > 0.0)
    assert result.maximum_energy_residual < 1.0e-11


def test_steady_thermal_roots_exist_for_wall_and_open_boundaries() -> None:
    for outer_mode in ("tidal_wall", "zero_torque"):
        mass, grid, transport, closure = _supplied_disk(48, outer_mode)
        result = solve_signed_thermal_steady(
            grid,
            transport,
            np.full(grid.centers.size, 1.0e6),
            mass,
            closure=closure,
            tolerance=1.0e-6,
            max_nfev=200,
        )

        assert result.accepted
        assert result.maximum_normalized_residual < 1.0e-6
        assert np.all(result.temperature > closure.temperature_bounds[0])
        assert np.all(result.temperature < closure.temperature_bounds[1])
        assert np.max(result.profile.H / grid.centers) < 1.0


def test_thermoviscous_fixed_point_matches_alpha_viscosity() -> None:
    mass, grid, transport, closure = _supplied_disk(32, "tidal_wall")
    result = solve_signed_thermoviscous_steady(
        grid,
        mass,
        alpha=0.01,
        boundary=SignedFluxBoundary(outer_mode="tidal_wall"),
        source_mass_rate_cells=transport.source_mass_rate_cells,
        source_specific_angular_momentum=np.full(
            grid.centers.size, closure.stream_specific_angular_momentum
        ),
        thermal_closure=closure,
        temperature_seed=np.full(grid.centers.size, 1.0e6),
        damping=0.25,
        tolerance=3.0e-3,
        max_iterations=50,
    )

    assert result.converged
    target = 0.01 * result.thermal.profile.H**2 * result.transport.omega
    assert np.max(np.abs(np.log(target / result.viscosity))) < 5.0e-3
    assert result.thermal.maximum_normalized_residual < 1.0e-6
