from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedThermalClosure,
    advance_eliminated_outer_dae_backward_euler,
    advance_flux_primary_outer_dae_backward_euler,
    angular_fluxes_for_common_stress_open_edge,
    audit_outer_dae_backward_euler_ledgers,
    eliminated_boundary_tangent,
    evaluate_outer_dae_profile,
    evaluate_flux_primary_outer_dae_profile,
    linear_torque_faces,
    make_log_grid,
    outer_storage_matrix,
    pack_eliminated_boundary_coordinates,
    pack_outer_primitives,
    solve_eliminated_instantaneous_flux,
    unpack_eliminated_boundary_coordinates,
    unpack_outer_primitives,
)
from imri_qpe.units import solar_masses_to_g


def _prototype_state(n: int = 8):
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(40.0 * potential.r_g, 335.0 * potential.r_g, n)
    coordinate = np.linspace(0.0, 1.0, n)
    sigma = 2.0e5 * np.exp(0.5 * coordinate)
    temperature = 2.0e6 * np.exp(-0.3 * coordinate)
    omega = 0.99 * potential.omega_k(grid.centers)
    closure = SignedThermalClosure()
    return mass, grid, sigma, temperature, omega, closure


def test_outer_dae_flux_ledgers_telescope() -> None:
    mass, grid, sigma, temperature, omega, closure = _prototype_state()
    mdot_faces = np.linspace(2.0e19, -1.0e19, grid.edges.size)
    profile = evaluate_outer_dae_profile(
        grid,
        sigma,
        temperature,
        omega,
        mdot_faces,
        mass,
        alpha=0.01,
        closure=closure,
    )

    assert np.isclose(
        np.sum(profile.mass_rhs), mdot_faces[-1] - mdot_faces[0]
    )
    assert np.isclose(
        np.sum(profile.angular_rhs),
        profile.angular_flux_faces[-1] - profile.angular_flux_faces[0],
    )
    radial_work = profile.energy_rhs - (
        profile.energy_flux_faces[1:] - profile.energy_flux_faces[:-1]
    )
    assert np.isclose(
        np.sum(profile.energy_rhs),
        profile.energy_flux_faces[-1]
        - profile.energy_flux_faces[0]
        + np.sum(radial_work),
    )


def test_outer_low_mach_storage_matrix_has_full_rank() -> None:
    mass, grid, sigma, temperature, omega, closure = _prototype_state()
    state = pack_outer_primitives(sigma, temperature, omega)

    storage, scales = outer_storage_matrix(
        state,
        grid,
        mass,
        closure=closure,
    )

    scaled = storage / scales[:, None]
    assert storage.shape == (3 * grid.centers.size,) * 2
    assert np.linalg.matrix_rank(scaled, tol=1.0e-9) == state.size


def test_eliminated_tangent_has_expected_rank_and_zero_outer_torque() -> None:
    mass, grid, sigma, temperature, omega, closure = _prototype_state()
    coordinates = pack_eliminated_boundary_coordinates(
        grid,
        sigma,
        temperature,
        omega,
        mass,
        closure=closure,
    )

    state, tangent = eliminated_boundary_tangent(
        coordinates,
        grid,
        mass,
        alpha=0.01,
        closure=closure,
    )
    recovered = unpack_eliminated_boundary_coordinates(
        coordinates,
        grid,
        mass,
        alpha=0.01,
        closure=closure,
    )
    unpack_outer_primitives(state, grid)

    assert tangent.shape == (3 * grid.centers.size, 3 * grid.centers.size - 1)
    assert np.linalg.matrix_rank(tangent, tol=1.0e-8) == tangent.shape[1]
    assert abs(linear_torque_faces(grid, recovered.torque_centers)[-1]) < (
        2.0e-13 * np.max(recovered.torque_centers)
    )


def test_eliminated_backward_euler_rejects_unconfigured_outer_inflow() -> None:
    mass, grid, sigma, temperature, omega, closure = _prototype_state(6)
    potential = PaczynskiWiitaPotential(mass)
    alpha = 0.01
    coordinates = pack_eliminated_boundary_coordinates(
        grid,
        sigma,
        temperature,
        omega,
        mass,
        closure=closure,
    )
    recovered = unpack_eliminated_boundary_coordinates(
        coordinates,
        grid,
        mass,
        alpha=alpha,
        closure=closure,
    )
    mdot_seed = -np.linspace(0.2, 0.5, grid.edges.size) * 1.0e22
    provisional = evaluate_outer_dae_profile(
        grid,
        recovered.surface_density,
        recovered.temperature,
        recovered.omega,
        mdot_seed,
        mass,
        alpha=alpha,
        closure=closure,
    )
    omega_k = potential.omega_k(grid.centers)
    balanced_omega = np.sqrt(
        recovered.omega**2 + provisional.radial_residual * omega_k**2
    )
    coordinates = pack_eliminated_boundary_coordinates(
        grid,
        recovered.surface_density,
        recovered.temperature,
        balanced_omega,
        mass,
        closure=closure,
    )
    instantaneous = solve_eliminated_instantaneous_flux(
        coordinates,
        mdot_seed,
        grid,
        mass,
        alpha=alpha,
        closure=closure,
        tolerance=1.0e-8,
    )
    assert instantaneous.maximum_residual < 1.0e-8

    loading_time = float(
        np.sum(instantaneous.profile.mass_cells)
        / np.max(np.abs(instantaneous.mdot_faces))
    )
    step = advance_eliminated_outer_dae_backward_euler(
        coordinates,
        instantaneous.mdot_faces,
        grid,
        mass,
        1.0e-6 * loading_time,
        alpha=alpha,
        closure=closure,
        tolerance=2.0e-7,
    )

    assert not step.accepted
    assert step.maximum_residual < 1.0e-6
    assert step.mdot_faces[-1] > 0.0
    assert abs(step.profile.torque_faces[-1]) < (
        2.0e-12 * np.max(step.profile.torque_centers)
    )
    ledger = audit_outer_dae_backward_euler_ledgers(
        instantaneous.profile,
        step.profile,
        1.0e-6 * loading_time,
    )
    assert ledger.relative_mass_defect < 2.0e-6
    assert ledger.relative_angular_momentum_defect < 2.0e-6
    assert ledger.relative_energy_defect < 2.0e-6


def test_flux_primary_angular_profile_closes_stress_and_open_edge() -> None:
    mass, grid, sigma, temperature, omega, closure = _prototype_state(8)
    alpha = 0.01
    mdot_faces = np.linspace(2.0e22, -5.0e22, grid.edges.size)
    common = evaluate_outer_dae_profile(
        grid,
        sigma,
        temperature,
        omega,
        mdot_faces,
        mass,
        alpha=alpha,
        closure=closure,
    ).torque_centers
    angular_faces = angular_fluxes_for_common_stress_open_edge(
        grid,
        mdot_faces,
        omega,
        common,
    )

    result = evaluate_flux_primary_outer_dae_profile(
        grid,
        sigma,
        temperature,
        omega,
        mdot_faces,
        angular_faces,
        mass,
        alpha=alpha,
        closure=closure,
    )

    assert np.max(np.abs(result.stress_residual)) < 2.0e-13
    assert abs(result.profile.torque_faces[-1]) < (
        2.0e-13 * np.max(np.abs(result.profile.torque_centers))
    )


def test_flux_primary_backward_euler_step_is_conservative() -> None:
    mass, grid, sigma, temperature, omega, closure = _prototype_state(6)
    potential = PaczynskiWiitaPotential(mass)
    alpha = 0.01
    mdot_faces = np.linspace(2.0e22, -5.0e22, grid.edges.size)
    provisional_common = evaluate_outer_dae_profile(
        grid,
        sigma,
        temperature,
        omega,
        mdot_faces,
        mass,
        alpha=alpha,
        closure=closure,
    ).torque_centers
    provisional_angular = angular_fluxes_for_common_stress_open_edge(
        grid, mdot_faces, omega, provisional_common
    )
    provisional = evaluate_flux_primary_outer_dae_profile(
        grid,
        sigma,
        temperature,
        omega,
        mdot_faces,
        provisional_angular,
        mass,
        alpha=alpha,
        closure=closure,
    )
    omega_k = potential.omega_k(grid.centers)
    balanced_omega = np.sqrt(
        omega**2 + provisional.profile.radial_residual * omega_k**2
    )
    common = evaluate_outer_dae_profile(
        grid,
        sigma,
        temperature,
        balanced_omega,
        mdot_faces,
        mass,
        alpha=alpha,
        closure=closure,
    ).torque_centers
    angular_faces = angular_fluxes_for_common_stress_open_edge(
        grid, mdot_faces, balanced_omega, common
    )
    state = pack_outer_primitives(sigma, temperature, balanced_omega)
    old = evaluate_flux_primary_outer_dae_profile(
        grid,
        sigma,
        temperature,
        balanced_omega,
        mdot_faces,
        angular_faces,
        mass,
        alpha=alpha,
        closure=closure,
    )
    loading_time = float(
        np.sum(old.profile.mass_cells) / np.max(np.abs(mdot_faces))
    )
    dt = 1.0e-8 * loading_time

    step = advance_flux_primary_outer_dae_backward_euler(
        state,
        mdot_faces,
        angular_faces,
        grid,
        mass,
        dt,
        alpha=alpha,
        closure=closure,
        tolerance=2.0e-7,
        mass_ledger_tolerance=2.0e-6,
        energy_ledger_tolerance=2.0e-6,
    )

    assert step.accepted, step.maximum_residual
    assert step.maximum_residual < 2.0e-7
    assert step.mdot_faces[0] >= 0.0
    assert step.mdot_faces[-1] <= 0.0
    ledger = audit_outer_dae_backward_euler_ledgers(
        old.profile, step.evaluation.profile, dt
    )
    assert ledger.relative_mass_defect < 2.0e-6
    assert ledger.relative_angular_momentum_defect < 2.0e-6
    assert ledger.relative_energy_defect < 2.0e-6
