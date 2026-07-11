from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedFluxBoundary,
    SignedThermalClosure,
    make_log_grid,
    normalized_stream_injection_state,
    signed_total_energy_profile,
    signed_vertical_work_rate_cells,
    solve_signed_flux_steady,
    solve_signed_thermal_steady,
    solve_signed_total_energy_steady,
    solve_signed_total_energy_thermoviscous_steady,
)
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


def _supplied_disk(n: int = 32, outer_mode: str = "tidal_wall"):
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(6.1 * potential.r_g, 335.0 * potential.r_g, n)
    stream_rate = 5.0 * eddington_mdot(mass)
    stream_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(stream_radius))
    stream_B = float(
        potential.phi(stream_radius)
        + 0.5 * (stream_l / stream_radius) ** 2
    )
    stream = normalized_stream_injection_state(
        grid,
        stream_rate,
        center=240.0 * potential.r_g,
        log_width=0.08,
        specific_angular_momentum=stream_l,
        specific_total_energy=stream_B,
    )
    viscosity = 1.0e-4 * grid.centers**2 * potential.omega_k(grid.centers)
    transport = solve_signed_flux_steady(
        grid,
        viscosity,
        mass,
        boundary=SignedFluxBoundary(outer_mode=outer_mode),
        stream_state=stream,
    )
    closure = SignedThermalClosure(temperature_bounds=(1.0e3, 1.0e9))
    internal = solve_signed_thermal_steady(
        grid,
        transport,
        np.full(n, 1.0e6),
        mass,
        closure=closure,
    )
    return mass, potential, grid, stream, transport, closure, internal


def test_total_energy_flux_contains_bernoulli_and_torque_work_once() -> None:
    mass, potential, grid, _stream, transport, closure, internal = _supplied_disk()
    profile = signed_total_energy_profile(
        grid, transport, internal.temperature, mass, closure=closure
    )
    expected_torque_work = (
        -potential.omega_k(grid.edges) * transport.viscous_torque_faces
    )

    assert np.allclose(profile.torque_work_flux_faces, expected_torque_work)
    assert np.allclose(
        profile.total_energy_flux_faces,
        profile.advective_energy_flux_faces + expected_torque_work,
    )
    assert np.allclose(
        profile.stream_energy_rate_cells,
        transport.source_total_energy_rate_cells,
    )


def test_total_energy_telescoping_defect_and_external_power_sign() -> None:
    mass, _potential, grid, _stream, transport, closure, internal = _supplied_disk()
    base = signed_total_energy_profile(
        grid, transport, internal.temperature, mass, closure=closure
    )
    external_power = np.linspace(1.0e30, 2.0e30, grid.centers.size)
    powered = signed_total_energy_profile(
        grid,
        transport,
        internal.temperature,
        mass,
        closure=closure,
        external_power_rate_cells=external_power,
    )
    scale = max(float(np.sum(np.abs(base.net_energy_rate_cells))), 1.0)

    assert abs(base.total_energy_telescoping_defect) / scale < 2.0e-15
    assert np.allclose(
        powered.net_energy_rate_cells,
        base.net_energy_rate_cells + external_power,
    )


@pytest.mark.parametrize("outer_mode", ["tidal_wall", "zero_torque"])
def test_fixed_transport_total_energy_roots_close(outer_mode: str) -> None:
    mass, _potential, grid, _stream, transport, closure, internal = _supplied_disk(
        32, outer_mode
    )
    result = solve_signed_total_energy_steady(
        grid,
        transport,
        internal.temperature,
        mass,
        closure=closure,
        tolerance=1.0e-6,
        max_nfev=500,
    )

    assert result.accepted
    assert result.maximum_normalized_residual < 1.0e-6
    assert np.max(np.abs(result.profile.net_energy_rate_cells)) > 0.0
    global_scale = max(
        float(np.sum(np.abs(result.profile.radiative_loss_rate_cells))), 1.0
    )
    assert abs(np.sum(result.profile.net_energy_rate_cells)) / global_scale < 1.0e-10


def test_total_energy_thermoviscous_fixed_point_is_self_consistent() -> None:
    mass, _potential, grid, stream, _transport, closure, _internal = _supplied_disk(
        24, "tidal_wall"
    )
    result = solve_signed_total_energy_thermoviscous_steady(
        grid,
        mass,
        alpha=0.01,
        boundary=SignedFluxBoundary(outer_mode="tidal_wall"),
        stream_state=stream,
        closure=closure,
        temperature_seed=np.full(grid.centers.size, 1.0e6),
        damping=0.2,
        tolerance=2.0e-3,
        max_iterations=60,
    )
    target = 0.01 * result.energy.profile.H**2 * result.transport.omega

    assert result.converged
    assert result.energy.accepted
    assert result.maximum_log_viscosity_change <= 2.0e-3
    assert np.max(np.abs(np.log(target / result.viscosity))) <= 2.0e-3


def test_vertical_work_cell_integral_converges_for_smooth_manufactured_state() -> None:
    errors = []
    for n in (64, 128, 256, 512):
        grid = make_log_grid(1.0, np.e, n)
        x = np.log(grid.centers)
        x_edges = np.log(grid.edges)
        mdot = np.full(n, 2.0)
        sigma = np.exp(0.3 * x)
        pressure = np.exp(0.6 * x)
        integrated_pressure = np.exp(0.7 * x)
        density = np.exp(0.2 * x)
        numerical = signed_vertical_work_rate_cells(
            grid,
            mdot,
            sigma,
            pressure,
            integrated_pressure,
            density,
        )
        exact = 0.5 * np.diff(np.exp(0.4 * x_edges))
        errors.append(float(np.max(np.abs(numerical - exact))))

    assert all(fine < 0.3 * coarse for coarse, fine in zip(errors, errors[1:]))


def test_distributed_external_torque_requires_named_power() -> None:
    mass, _potential, grid, stream, _transport, closure, _internal = _supplied_disk(
        16, "tidal_wall"
    )
    external_angular = np.zeros(grid.centers.size)
    external_angular[-1] = 1.0e35

    with pytest.raises(ValueError, match="requires named external power"):
        solve_signed_total_energy_thermoviscous_steady(
            grid,
            mass,
            alpha=0.01,
            boundary=SignedFluxBoundary(outer_mode="tidal_wall"),
            stream_state=stream,
            closure=closure,
            temperature_seed=np.full(grid.centers.size, 1.0e6),
            external_angular_rate_cells=external_angular,
            max_iterations=1,
        )
