from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedThermalClosure,
    common_stress_torque_centers,
    gas_pressure_fraction,
    linear_torque_faces,
    make_log_grid,
    pack_eliminated_boundary_coordinates,
    recover_thermodynamics_from_pi_beta,
    regularized_zero_torque_remap,
    remap_zero_torque_thermodynamics,
    transport_outer_torque,
    unpack_eliminated_boundary_coordinates,
    vertical_state,
)
from imri_qpe.units import solar_masses_to_g


def test_linear_torque_faces_preserve_zero_torque_asymptotic() -> None:
    grid = make_log_grid(10.0, 80.0, 16)
    amplitude = 7.5
    torque = amplitude * (grid.edges[-1] - grid.centers)

    faces = linear_torque_faces(grid, torque)

    np.testing.assert_allclose(
        faces,
        amplitude * (grid.edges[-1] - grid.edges),
        rtol=2.0e-14,
        atol=2.0e-13,
    )
    assert abs(faces[-1]) < 1.0e-12


def test_regularized_remap_preserves_zero_torque_profile() -> None:
    coarse = make_log_grid(10.0, 80.0, 16)
    fine = make_log_grid(10.0, 80.0, 48)
    amplitude = 3.25
    coarse_torque = amplitude * (coarse.edges[-1] - coarse.centers)

    fine_torque = regularized_zero_torque_remap(coarse, fine, coarse_torque)

    np.testing.assert_allclose(
        fine_torque,
        amplitude * (fine.edges[-1] - fine.centers),
        rtol=2.0e-14,
    )
    assert abs(linear_torque_faces(fine, fine_torque)[-1]) < 1.0e-12


def test_transport_outer_torque_retains_flux_dependencies() -> None:
    base = transport_outer_torque(-2.0, 5.0, -11.0)
    mdot_shift = transport_outer_torque(-1.9, 5.0, -11.0) - base
    angular_shift = transport_outer_torque(-2.0, 5.0, -10.9) - base

    assert base == 1.0
    assert np.isclose(mdot_shift, 0.5)
    assert np.isclose(angular_shift, -0.1)


def test_pressure_beta_coordinates_recover_vertical_primitives() -> None:
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    radius = np.asarray([30.0, 80.0, 240.0]) * potential.r_g
    sigma = np.asarray([2.0e4, 8.0e4, 3.0e5])
    temperature = np.asarray([8.0e6, 3.0e6, 9.0e5])
    closure = SignedThermalClosure()
    state = vertical_state(
        sigma,
        temperature,
        radius,
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )

    recovered_sigma, recovered_temperature, residual = (
        recover_thermodynamics_from_pi_beta(
            radius,
            state.Pi,
            gas_pressure_fraction(state),
            mass,
            closure=closure,
            surface_density_seed=1.1 * sigma,
            temperature_seed=0.9 * temperature,
        )
    )

    np.testing.assert_allclose(recovered_sigma, sigma, rtol=2.0e-10)
    np.testing.assert_allclose(
        recovered_temperature, temperature, rtol=2.0e-10
    )
    assert residual < 1.0e-10


def test_zero_torque_thermodynamic_remap_round_trips_one_grid() -> None:
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(20.0 * potential.r_g, 335.0 * potential.r_g, 12)
    coordinate = np.linspace(0.0, 1.0, grid.centers.size)
    sigma = 2.0e5 * np.exp(0.7 * coordinate)
    temperature = 2.0e6 * np.exp(-0.4 * coordinate)
    closure = SignedThermalClosure()

    remap = remap_zero_torque_thermodynamics(
        grid,
        grid,
        sigma,
        temperature,
        mass,
        alpha=0.01,
        closure=closure,
    )
    torque = common_stress_torque_centers(
        grid,
        sigma,
        temperature,
        mass,
        alpha=0.01,
        closure=closure,
    )

    np.testing.assert_allclose(remap.torque_centers, torque, rtol=2.0e-14)
    np.testing.assert_allclose(remap.surface_density, sigma, rtol=3.0e-10)
    np.testing.assert_allclose(remap.temperature, temperature, rtol=3.0e-10)
    assert remap.maximum_inversion_residual < 1.0e-10


def test_eliminated_boundary_chart_satisfies_zero_torque_exactly() -> None:
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(20.0 * potential.r_g, 335.0 * potential.r_g, 10)
    coordinate = np.linspace(0.0, 1.0, grid.centers.size)
    sigma = 3.0e5 * np.exp(0.4 * coordinate)
    temperature = 1.5e6 * np.exp(-0.2 * coordinate)
    omega = potential.omega_k(grid.centers)
    closure = SignedThermalClosure()
    packed = pack_eliminated_boundary_coordinates(
        grid,
        sigma,
        temperature,
        omega,
        mass,
        closure=closure,
    )

    recovered = unpack_eliminated_boundary_coordinates(
        packed,
        grid,
        mass,
        alpha=0.01,
        closure=closure,
    )

    assert packed.shape == (3 * grid.centers.size - 1,)
    assert np.all(recovered.surface_density > 0.0)
    assert np.all(recovered.temperature > 0.0)
    assert abs(linear_torque_faces(grid, recovered.torque_centers)[-1]) < (
        2.0e-13 * np.max(recovered.torque_centers)
    )
    assert recovered.inversion_residual < 1.0e-10
