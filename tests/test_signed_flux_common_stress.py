from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedFluxBoundary,
    SignedThermalClosure,
    common_alpha_stress_torque,
    diffusive_alpha_torque,
    make_log_grid,
    normalized_stream_injection_state,
    solve_common_stress_total_energy_steady,
    solve_nonkeplerian_common_stress_steady,
    solve_signed_flux_steady,
    solve_signed_thermal_steady,
)
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


def _supplied_wall(n: int = 16):
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(10.0 * potential.r_g, 335.0 * potential.r_g, n)
    stream_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(stream_radius))
    stream_B = float(
        potential.phi(stream_radius)
        + 0.5 * (stream_l / stream_radius) ** 2
    )
    stream = normalized_stream_injection_state(
        grid,
        5.0 * eddington_mdot(mass),
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
        boundary=SignedFluxBoundary(outer_mode="tidal_wall"),
        stream_state=stream,
    )
    closure = SignedThermalClosure(temperature_bounds=(1.0e3, 1.0e9))
    thermal = solve_signed_thermal_steady(
        grid,
        transport,
        np.full(n, 1.0e6),
        mass,
        closure=closure,
    )
    return mass, potential, grid, transport, closure, thermal


def test_diffusive_and_common_stress_differ_by_keplerian_shear() -> None:
    mass, potential, grid, transport, closure, thermal = _supplied_wall()
    common = common_alpha_stress_torque(
        grid,
        transport.surface_density,
        thermal.temperature,
        mass,
        alpha=0.01,
        closure=closure,
    )
    diffusive = diffusive_alpha_torque(
        grid,
        transport.surface_density,
        thermal.temperature,
        mass,
        alpha=0.01,
        closure=closure,
    )
    expected = -potential.dln_omega_k_dlnR(grid.centers)

    assert np.allclose(diffusive / common, expected, rtol=2.0e-13)


def test_common_stress_homotopy_closes_torque_and_total_energy() -> None:
    mass, _potential, grid, transport, closure, thermal = _supplied_wall()
    sigma = transport.surface_density
    temperature = thermal.temperature
    result = None

    for stress_fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = solve_common_stress_total_energy_steady(
            grid,
            transport,
            sigma,
            temperature,
            mass,
            alpha=0.01,
            closure=closure,
            stress_fraction=stress_fraction,
            tolerance=2.0e-6,
            max_nfev=1000,
        )
        assert result.accepted, (
            stress_fraction,
            result.maximum_stress_residual,
            result.maximum_energy_residual,
            result.message,
        )
        sigma = result.surface_density
        temperature = result.temperature

    assert result is not None
    common = common_alpha_stress_torque(
        grid,
        result.surface_density,
        result.temperature,
        mass,
        alpha=0.01,
        closure=closure,
    )
    required = transport.viscous_torque_centers
    scale = np.maximum(np.abs(required), 1.0e-8 * np.max(np.abs(required)))

    assert np.max(np.abs(common - required) / scale) < 2.0e-6
    assert np.array_equal(result.transport.mdot_faces, transport.mdot_faces)
    assert np.array_equal(
        result.transport.angular_flux_faces, transport.angular_flux_faces
    )
    assert np.array_equal(
        result.transport.viscous_torque_faces, transport.viscous_torque_faces
    )


def test_simultaneous_nonkeplerian_homotopy_closes_all_equations() -> None:
    mass, potential, grid, transport, closure, thermal = _supplied_wall(12)
    common = solve_common_stress_total_energy_steady(
        grid,
        transport,
        transport.surface_density,
        thermal.temperature,
        mass,
        alpha=0.01,
        closure=closure,
        tolerance=2.0e-6,
    )
    assert common.accepted
    sigma = common.surface_density
    temperature = common.temperature
    omega = potential.omega_k(grid.centers)
    result = None

    for support_fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = solve_nonkeplerian_common_stress_steady(
            grid,
            transport,
            sigma,
            temperature,
            omega,
            mass,
            alpha=0.01,
            closure=closure,
            radial_support_fraction=support_fraction,
            tolerance=2.0e-6,
            max_nfev=2000,
        )
        assert result.accepted, (
            support_fraction,
            result.maximum_stress_residual,
            result.maximum_radial_residual,
            result.maximum_energy_residual,
            result.minimum_dln_l_dln_R,
            result.maximum_dln_omega_dln_R,
            result.message,
        )
        sigma = result.surface_density
        temperature = result.temperature
        omega = result.omega

    assert result is not None
    assert result.maximum_stress_residual < 2.0e-6
    assert result.maximum_radial_residual < 2.0e-6
    assert result.maximum_energy_residual < 2.0e-6
    assert result.minimum_dln_l_dln_R > 0.0
    assert result.maximum_dln_omega_dln_R < 0.0
