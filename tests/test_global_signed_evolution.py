from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq

from imri_qpe.constants import C, M_SUN
from imri_qpe.layer3_minidisk_1d.entropy_advection import total_pressure
from imri_qpe.layer3_minidisk_1d.global_signed_evolution import (
    GlobalCellSources,
    GlobalConservativeState,
    GlobalFaceFluxes,
    GlobalFluxPrimaryLayout,
    add_global_alpha_stress_fluxes,
    apply_global_hill_roche_outer_boundary,
    apply_global_reference_characteristic_inner_flux,
    advance_global_alpha_stress_backward_euler,
    advance_global_backward_euler,
    advance_global_radiative_cooling_backward_euler,
    advance_global_imex,
    advance_global_inviscid_rusanov,
    audit_global_backward_euler_ledgers,
    audit_global_physical_flux_eigensystem,
    global_alpha_stress_torque_faces,
    global_backward_euler_residual,
    global_backward_euler_jacobian_sparsity,
    global_compact_stream_cell_sources,
    global_conservative_rhs,
    global_descriptor_mass_matrix,
    evaluate_global_inviscid_profile,
    evaluate_global_rusanov_profile,
    global_inviscid_cfl_timestep,
    global_inner_characteristic_audit,
    global_outer_characteristic_audit,
    global_physical_bernoulli,
    global_radiative_cooling_rate_cells,
    global_temporal_vertical_work_cells,
    global_vertical_work_rate_cells,
    load_global_mechanical_energy_reference,
    make_global_mechanical_energy_reference,
    manufactured_backward_euler_jacobian,
    pack_global_flux_primary_state,
    recover_global_primitives,
    reconstruct_global_outer_edge_state,
    save_global_mechanical_energy_reference,
    state_from_primitives,
    state_from_thermodynamic_primitives,
    unpack_global_flux_primary_state,
)
from imri_qpe.layer3_minidisk_1d.hill_roche_nozzle import (
    GasRadiationHillRocheNozzleProvider,
    HillRocheNozzleReservoir,
    fiducial_hill_roche_nozzle_geometry,
)
from imri_qpe.layer3_minidisk_1d.grid import make_log_grid
from imri_qpe.layer3_minidisk_1d.signed_flux_common_stress import (
    positive_edge_reconstruction,
)
from imri_qpe.layer3_minidisk_1d.signed_flux_thermal import SignedThermalClosure
from imri_qpe.layer3_minidisk_1d.signed_flux_total_energy import (
    signed_vertical_work_rate_cells,
)
from imri_qpe.layer3_minidisk_1d.time_dae_boundary import (
    recover_thermodynamics_from_pi_beta,
)
from imri_qpe.layer3_minidisk_1d.transonic_potential import PaczynskiWiitaPotential
from imri_qpe.layer3_minidisk_1d.transonic_thermo import (
    integrated_stress,
    radiative_cooling,
    vertical_state,
)
from imri_qpe.parameters import FiducialParams


def _manufactured_case(n_cells: int = 8):
    grid = make_log_grid(1.0, 3.0, n_cells)
    radius = grid.centers
    state = state_from_primitives(
        grid,
        2.0 + 0.1 * radius / radius[-1],
        -0.03 + 0.06 * np.linspace(0.0, 1.0, n_cells),
        radius ** (-1.5),
        -1.0 / radius + 0.02,
    )
    coordinate = np.linspace(-1.0, 1.0, n_cells + 1)
    fluxes = GlobalFaceFluxes(
        mass=coordinate,
        radial_momentum=0.2 + 0.1 * coordinate,
        angular_momentum=2.0 - 0.3 * coordinate,
        total_energy=-0.5 + 0.4 * coordinate,
    )
    sources = GlobalCellSources(
        mass=np.linspace(0.0, 0.2, n_cells),
        radial_momentum=np.linspace(-0.1, 0.1, n_cells),
        angular_momentum=np.linspace(0.3, 0.0, n_cells),
        total_energy=np.linspace(-0.2, 0.2, n_cells),
    )
    return grid, state, fluxes, sources


def _physical_roche_column(temperature: float, n_cells: int = 32):
    params = FiducialParams()
    potential = PaczynskiWiitaPotential(params.M2_g)
    outer_radius = 335.0 * potential.r_g
    grid = make_log_grid(100.0 * potential.r_g, outer_radius, n_cells)
    target_edge_density = 1.0e-8
    edge_pressure = float(total_pressure(target_edge_density, temperature))
    edge_height = np.sqrt(edge_pressure / target_edge_density) / float(
        potential.omega_k(outer_radius)
    )
    surface_density = np.full(
        n_cells, 2.0 * target_edge_density * edge_height
    )
    state = state_from_thermodynamic_primitives(
        grid,
        surface_density,
        np.zeros(n_cells),
        np.asarray(potential.omega_k(grid.centers), dtype=float),
        np.full(n_cells, temperature),
        params.M2_g,
    )
    primitives = recover_global_primitives(grid, state, params.M2_g)
    provider = GasRadiationHillRocheNozzleProvider(
        fiducial_hill_roche_nozzle_geometry(),
        transverse_quadrature_zones=16,
    )
    return params, grid, state, primitives, provider


def test_global_layout_has_four_differential_and_four_face_fields() -> None:
    for n_cells in (1, 8, 16):
        layout = GlobalFluxPrimaryLayout(n_cells)
        assert layout.differential_size == 4 * n_cells
        assert layout.algebraic_size == 4 * (n_cells + 1)
        assert layout.state_size == 8 * n_cells + 4
        assert layout.residual_size == layout.state_size
        assert global_descriptor_mass_matrix(layout).shape == (
            layout.state_size,
            layout.state_size,
        )
        assert np.linalg.matrix_rank(
            global_descriptor_mass_matrix(layout).toarray()
        ) == layout.differential_size
        assert np.linalg.matrix_rank(
            manufactured_backward_euler_jacobian(layout, 0.25).toarray()
        ) == layout.state_size


def test_global_backward_euler_sparsity_is_nearest_neighbor() -> None:
    for n_cells in (1, 8, 16):
        pattern = global_backward_euler_jacobian_sparsity(n_cells)
        assert pattern.shape == (4 * n_cells, 4 * n_cells)
        assert pattern.nnz == 48 * n_cells - 32


def test_global_state_round_trip_preserves_signed_flux_crossing() -> None:
    _grid, state, fluxes, _sources = _manufactured_case()
    layout = GlobalFluxPrimaryLayout(state.n_cells)
    packed = pack_global_flux_primary_state(state, fluxes)
    restored_state, restored_fluxes = unpack_global_flux_primary_state(
        packed, layout
    )
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        np.testing.assert_array_equal(
            getattr(restored_state, name), getattr(state, name)
        )
        np.testing.assert_array_equal(
            getattr(restored_fluxes, name), getattr(fluxes, name)
        )
    assert np.any(restored_fluxes.mass < 0.0)
    assert np.any(restored_fluxes.mass > 0.0)
    assert np.any(np.isclose(restored_fluxes.mass, 0.0))


def test_manufactured_backward_euler_state_closes_all_ledgers() -> None:
    _grid, old_state, fluxes, sources = _manufactured_case()
    dt = 1.0e-4
    rhs = global_conservative_rhs(fluxes, sources)
    new_state = GlobalConservativeState(
        **{
            name: getattr(old_state, name) + dt * getattr(rhs, name)
            for name in (
                "mass",
                "radial_momentum",
                "angular_momentum",
                "total_energy",
            )
        }
    ).validated()
    residual = global_backward_euler_residual(
        new_state, old_state, dt, fluxes, sources
    )
    storage_scale = np.concatenate(
        tuple(
            np.maximum(np.abs(getattr(new_state, name)), 1.0)
            for name in (
                "mass",
                "radial_momentum",
                "angular_momentum",
                "total_energy",
            )
        )
    )
    assert np.max(np.abs(residual) / storage_scale) < 2.0e-16
    audit = audit_global_backward_euler_ledgers(
        new_state, old_state, dt, fluxes, sources
    )
    assert audit.maximum_relative_defect < 1.0e-14


def test_constant_face_flux_has_zero_source_free_rhs() -> None:
    n_cells = 6
    fluxes = GlobalFaceFluxes(
        mass=np.full(n_cells + 1, -2.0),
        radial_momentum=np.full(n_cells + 1, 3.0),
        angular_momentum=np.full(n_cells + 1, -4.0),
        total_energy=np.full(n_cells + 1, 5.0),
    )
    rhs = global_conservative_rhs(fluxes, GlobalCellSources.zeros(n_cells))
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        np.testing.assert_array_equal(getattr(rhs, name), np.zeros(n_cells))


def test_thermodynamic_primitive_recovery_round_trip() -> None:
    mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(6.2 * potential.r_g, 300.0 * potential.r_g, 10)
    sigma = np.geomspace(1.0e4, 3.0e6, grid.centers.size)
    radial_velocity = C * np.linspace(-2.0e-3, 2.0e-3, grid.centers.size)
    omega = 0.82 * potential.omega_k(grid.centers)
    temperature = np.geomspace(2.0e6, 2.0e8, grid.centers.size)
    state = state_from_thermodynamic_primitives(
        grid,
        sigma,
        radial_velocity,
        omega,
        temperature,
        mass,
    )
    recovered = recover_global_primitives(grid, state, mass)
    np.testing.assert_allclose(recovered.surface_density, sigma, rtol=1.0e-14)
    np.testing.assert_allclose(
        recovered.radial_velocity, radial_velocity, rtol=1.0e-14, atol=1.0e-8
    )
    np.testing.assert_allclose(recovered.omega, omega, rtol=1.0e-14)
    np.testing.assert_allclose(recovered.temperature, temperature, rtol=2.0e-11)


def test_mechanical_reference_correction_preserves_thermal_round_trip() -> None:
    mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(6.2 * potential.r_g, 300.0 * potential.r_g, 10)
    sigma = np.geomspace(1.0e4, 3.0e6, grid.centers.size)
    radial_velocity = C * np.linspace(-2.0e-3, 2.0e-3, grid.centers.size)
    omega = 0.82 * potential.omega_k(grid.centers)
    temperature = np.geomspace(2.0e6, 2.0e8, grid.centers.size)
    correction = np.linspace(-3.0e16, 2.0e16, grid.centers.size)
    baseline = state_from_thermodynamic_primitives(
        grid, sigma, radial_velocity, omega, temperature, mass
    )
    corrected = state_from_thermodynamic_primitives(
        grid,
        sigma,
        radial_velocity,
        omega,
        temperature,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    assert not np.array_equal(corrected.total_energy, baseline.total_energy)
    recovered = recover_global_primitives(
        grid,
        corrected,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    np.testing.assert_allclose(recovered.surface_density, sigma, rtol=1.0e-14)
    np.testing.assert_allclose(
        recovered.radial_velocity, radial_velocity, rtol=1.0e-14, atol=1.0e-8
    )
    np.testing.assert_allclose(recovered.omega, omega, rtol=1.0e-14)
    np.testing.assert_allclose(recovered.temperature, temperature, rtol=2.0e-11)


def test_mechanical_reference_checkpoint_round_trip_is_exact(tmp_path) -> None:
    mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(6.2 * potential.r_g, 40.0 * potential.r_g, 6)
    correction = np.linspace(-3.0e16, 2.0e16, grid.centers.size)
    state = state_from_thermodynamic_primitives(
        grid,
        np.geomspace(1.0e5, 1.0e6, grid.centers.size),
        np.linspace(-2.0e6, 1.0e6, grid.centers.size),
        0.9 * potential.omega_k(grid.centers),
        np.geomspace(2.0e6, 8.0e6, grid.centers.size),
        mass,
        specific_mechanical_energy_correction=correction,
    )
    reference = make_global_mechanical_energy_reference(
        grid,
        correction,
        state,
        provenance={"generator": "unit-test", "quadrature_order": 32},
    )
    path = tmp_path / "mechanical_reference.npz"
    save_global_mechanical_energy_reference(path, reference)
    restored = load_global_mechanical_energy_reference(
        path, grid=grid, reference_state=state
    )
    np.testing.assert_array_equal(restored.grid_edges, grid.edges)
    np.testing.assert_array_equal(restored.specific_offset, correction)
    assert restored.offset_sha256 == reference.offset_sha256
    assert restored.reference_state_sha256 == reference.reference_state_sha256
    assert restored.provenance == reference.provenance

    shifted = replace(
        state, total_energy=state.total_energy + 1.0e16 * state.mass
    )
    with pytest.raises(ValueError, match="generating state mismatch"):
        load_global_mechanical_energy_reference(
            path, grid=grid, reference_state=shifted
        )


def test_primitive_recovery_rejects_nonphysical_internal_energy() -> None:
    mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(6.2 * potential.r_g, 30.0 * potential.r_g, 4)
    state = state_from_thermodynamic_primitives(
        grid,
        np.full(4, 1.0e5),
        np.zeros(4),
        potential.omega_k(grid.centers),
        np.full(4, 1.0e7),
        mass,
    )
    invalid = GlobalConservativeState(
        mass=state.mass,
        radial_momentum=state.radial_momentum,
        angular_momentum=state.angular_momentum,
        total_energy=state.total_energy - 1.0e25 * state.mass,
    )
    with np.testing.assert_raises_regex(ValueError, "outside temperature bounds"):
        recover_global_primitives(grid, invalid, mass)


def _manufactured_rotating_equilibrium(n_cells: int, pressure_slope: float):
    mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(12.0 * potential.r_g, 120.0 * potential.r_g, n_cells)
    closure = SignedThermalClosure()
    seed_sigma = np.full(n_cells, 1.0e8)
    seed_temperature = np.full(n_cells, 1.0e6)
    reference_radius = np.sqrt(grid.edges[0] * grid.edges[-1])
    reference = vertical_state(
        1.0e8,
        1.0e6,
        reference_radius,
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )
    reference_pi = float(reference.Pi)
    reference_beta = float(reference.P_gas / reference.P_tot)
    target_pi = reference_pi * (grid.centers / reference_radius) ** pressure_slope
    sigma, temperature, inversion_residual = recover_thermodynamics_from_pi_beta(
        grid.centers,
        target_pi,
        np.full(n_cells, reference_beta),
        mass,
        closure=closure,
        surface_density_seed=seed_sigma,
        temperature_seed=seed_temperature,
    )
    omega_k = potential.omega_k(grid.centers)
    omega_squared = omega_k**2 + pressure_slope * target_pi / (
        sigma * grid.centers**2
    )
    assert np.all(omega_squared > 0.0)
    state = state_from_thermodynamic_primitives(
        grid,
        sigma,
        np.zeros(n_cells),
        np.sqrt(omega_squared),
        temperature,
        mass,
    )
    return grid, state, mass, inversion_residual


def _maximum_radial_balance_error(n_cells: int, pressure_slope: float) -> float:
    grid, state, mass, inversion_residual = _manufactured_rotating_equilibrium(
        n_cells, pressure_slope
    )
    assert inversion_residual < 1.0e-9
    profile = evaluate_global_inviscid_profile(grid, state, mass)
    rhs = global_conservative_rhs(profile.face_fluxes, profile.cell_sources)
    radial_scale = np.maximum(
        np.abs(profile.face_fluxes.radial_momentum[:-1])
        + np.abs(profile.face_fluxes.radial_momentum[1:])
        + np.abs(profile.cell_sources.radial_momentum),
        1.0,
    )
    return float(np.max(np.abs(rhs.radial_momentum) / radial_scale))


def test_inviscid_pair_preserves_constant_pi_keplerian_equilibrium() -> None:
    error = _maximum_radial_balance_error(48, 0.0)
    assert error < 2.0e-9


def test_inviscid_pressure_supported_equilibrium_converges() -> None:
    errors = [
        _maximum_radial_balance_error(n_cells, -0.25)
        for n_cells in (24, 48, 96)
    ]
    assert errors[1] < errors[0] / 3.0
    assert errors[2] < errors[1] / 3.0


def test_inviscid_flux_remains_finite_through_stagnation() -> None:
    mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(8.0 * potential.r_g, 80.0 * potential.r_g, 8)
    log_radius = np.log(grid.centers)
    stagnation = np.log(grid.edges[4])
    radial_velocity = (
        2.0e-3
        * C
        * (log_radius - stagnation)
        / (log_radius[-1] - log_radius[0])
    )
    state = state_from_thermodynamic_primitives(
        grid,
        np.geomspace(1.0e5, 3.0e5, 8),
        radial_velocity,
        0.9 * potential.omega_k(grid.centers),
        np.geomspace(5.0e6, 2.0e7, 8),
        mass,
    )
    profile = evaluate_global_inviscid_profile(grid, state, mass)
    assert np.all(np.isfinite(profile.face_fluxes.mass))
    assert np.any(profile.face_fluxes.mass < 0.0)
    assert np.any(profile.face_fluxes.mass > 0.0)
    assert abs(profile.face_fluxes.mass[4]) < max(
        abs(profile.face_fluxes.mass[0]), abs(profile.face_fluxes.mass[-1])
    ) * 1.0e-12


def test_equilibrium_corrected_rusanov_recovers_smooth_reference_flux() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(32, -0.25)
    smooth = evaluate_global_inviscid_profile(grid, state, mass)
    corrected = evaluate_global_rusanov_profile(
        grid, state, mass, reference_state=state
    )
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        np.testing.assert_allclose(
            getattr(corrected.face_fluxes, name),
            getattr(smooth.face_fluxes, name),
            rtol=2.0e-15,
            atol=0.0,
        )


def test_conserved_donor_outer_face_uses_one_shared_mass_flux() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(16, -0.25)
    base = recover_global_primitives(grid, state, mass)
    radial_velocity = np.geomspace(1.0e5, 2.0e5, grid.centers.size)
    state = state_from_thermodynamic_primitives(
        grid,
        base.surface_density,
        radial_velocity,
        base.omega,
        base.temperature,
        mass,
    )
    primitives = recover_global_primitives(grid, state, mass)
    legacy = evaluate_global_rusanov_profile(
        grid,
        state,
        mass,
        reference_state=state,
        boundary_mode="open_no_inflow",
        alpha=0.1,
        stress_boundary_mode="outer_zero_torque",
    )
    donor = evaluate_global_rusanov_profile(
        grid,
        state,
        mass,
        reference_state=state,
        boundary_mode="open_no_inflow",
        alpha=0.1,
        stress_boundary_mode="outer_zero_torque",
        open_face_reconstruction="conserved_donor",
    )
    expected_mass = (
        2.0
        * np.pi
        * grid.centers[-1]
        * primitives.surface_density[-1]
        * primitives.radial_velocity[-1]
    )
    expected_l = grid.centers[-1] ** 2 * primitives.omega[-1]
    expected_bernoulli = (
        primitives.specific_total_energy[-1]
        + primitives.vertical.Pi[-1] / primitives.surface_density[-1]
    )
    expected_radial = (
        expected_mass * primitives.radial_velocity[-1]
        + 2.0 * np.pi * grid.edges[-1] * primitives.vertical.Pi[-1]
    )
    assert donor.viscous_torque_faces[-1] == 0.0
    assert donor.face_fluxes.mass[-1] == expected_mass
    assert donor.face_fluxes.radial_momentum[-1] == expected_radial
    assert donor.face_fluxes.angular_momentum[-1] == expected_mass * expected_l
    assert donor.face_fluxes.total_energy[-1] == expected_mass * expected_bernoulli
    assert legacy.face_fluxes.mass[-1] != donor.face_fluxes.mass[-1]


def test_physical_face_bernoulli_excludes_cell_mechanical_offset() -> None:
    grid, baseline, mass, _residual = _manufactured_rotating_equilibrium(
        16, -0.25
    )
    base = recover_global_primitives(grid, baseline, mass)
    correction = np.linspace(-3.0e16, 2.0e16, grid.centers.size)
    radial_velocity = np.geomspace(1.0e5, 2.0e5, grid.centers.size)
    state = state_from_thermodynamic_primitives(
        grid,
        base.surface_density,
        radial_velocity,
        base.omega,
        base.temperature,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    primitives = recover_global_primitives(
        grid,
        state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    physical_bernoulli = global_physical_bernoulli(primitives, correction)
    np.testing.assert_allclose(
        physical_bernoulli,
        primitives.specific_total_energy
        - correction
        + primitives.vertical.Pi / primitives.surface_density,
        rtol=2.0e-15,
    )
    donor = evaluate_global_rusanov_profile(
        grid,
        state,
        mass,
        reference_state=state,
        boundary_mode="open_no_inflow",
        open_face_reconstruction="conserved_donor",
        specific_mechanical_energy_correction=correction,
    )
    expected_mass = (
        2.0
        * np.pi
        * grid.centers[-1]
        * primitives.surface_density[-1]
        * primitives.radial_velocity[-1]
    )
    assert donor.face_fluxes.total_energy[-1] == pytest.approx(
        expected_mass * physical_bernoulli[-1], rel=2.0e-15
    )

    smooth = evaluate_global_inviscid_profile(
        grid,
        state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    balanced = evaluate_global_rusanov_profile(
        grid,
        state,
        mass,
        reference_state=state,
        specific_mechanical_energy_correction=correction,
    )
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        np.testing.assert_allclose(
            getattr(balanced.face_fluxes, name),
            getattr(smooth.face_fluxes, name),
            rtol=2.0e-15,
        )


def test_outer_characteristic_audit_counts_subsonic_exterior_condition() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(16, -0.25)
    primitives = recover_global_primitives(grid, state, mass)
    sound_speed = global_outer_characteristic_audit(
        primitives
    ).effective_sound_speed
    subsonic = state_from_thermodynamic_primitives(
        grid,
        primitives.surface_density,
        np.full(grid.centers.size, 0.1 * sound_speed),
        primitives.omega,
        primitives.temperature,
        mass,
    )
    audit = global_outer_characteristic_audit(
        recover_global_primitives(grid, subsonic, mass)
    )
    assert 0.0 < audit.radial_mach_number < 1.0
    assert audit.incoming_characteristics == 1
    assert audit.eigenvalues[0] < 0.0
    assert all(value > 0.0 for value in audit.eigenvalues[1:])


def test_outer_characteristic_audit_counts_supersonic_outflow() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(16, -0.25)
    primitives = recover_global_primitives(grid, state, mass)
    sound_speed = global_outer_characteristic_audit(
        primitives
    ).effective_sound_speed
    supersonic = state_from_thermodynamic_primitives(
        grid,
        primitives.surface_density,
        np.full(grid.centers.size, 1.1 * sound_speed),
        primitives.omega,
        primitives.temperature,
        mass,
    )
    audit = global_outer_characteristic_audit(
        recover_global_primitives(grid, supersonic, mass)
    )
    assert audit.radial_mach_number > 1.0
    assert audit.incoming_characteristics == 0
    assert all(value > 0.0 for value in audit.eigenvalues)


def test_roche_edge_reconstruction_uses_one_exact_physical_column() -> None:
    params, grid, _state, primitives, _provider = _physical_roche_column(
        8.0e5
    )
    edge = reconstruct_global_outer_edge_state(
        grid, primitives, params.M2_g
    )
    assert edge.radius == grid.edges[-1]
    assert edge.density == pytest.approx(1.0e-8, rel=3.0e-12)
    assert edge.pressure == pytest.approx(
        total_pressure(edge.density, edge.temperature), rel=3.0e-12
    )
    assert edge.integrated_pressure / edge.surface_density == pytest.approx(
        edge.pressure / edge.density, rel=3.0e-12
    )
    assert edge.bernoulli == pytest.approx(
        edge.specific_total_energy
        + edge.integrated_pressure / edge.surface_density,
        rel=2.0e-15,
    )


def test_roche_boundary_closed_branch_retains_only_pressure_traction() -> None:
    params, grid, state, primitives, provider = _physical_roche_column(1.0e5)
    base = evaluate_global_rusanov_profile(grid, state, params.M2_g)
    fluxes, audit = apply_global_hill_roche_outer_boundary(
        grid,
        base.face_fluxes,
        primitives,
        params.M2_g,
        provider,
    )
    assert not audit.gate.choked
    assert audit.applied_mass_flux == 0.0
    assert audit.no_inward_mass
    assert audit.incoming_acoustic_conditions == 1
    assert fluxes.angular_momentum[-1] == 0.0
    assert fluxes.total_energy[-1] == 0.0
    assert fluxes.radial_momentum[-1] == pytest.approx(
        audit.pressure_traction, rel=2.0e-15
    )


def test_roche_boundary_choked_branch_uses_one_conservative_nozzle_state() -> None:
    params, grid, state, _primitives, provider = _physical_roche_column(8.0e5)
    profile = evaluate_global_rusanov_profile(
        grid,
        state,
        params.M2_g,
        boundary_mode="roche_outer",
        stress_boundary_mode="outer_zero_torque",
        outer_overflow_provider=provider,
    )
    audit = profile.outer_roche_boundary
    assert audit is not None
    assert audit.gate.choked
    assert audit.gate.solution is not None
    assert audit.applied_mass_flux > 0.0
    assert audit.no_inward_mass
    assert audit.incoming_acoustic_conditions == 1
    assert abs(audit.angular_flux_relative_mismatch) < 2.0e-12
    assert abs(audit.energy_flux_relative_mismatch) < 2.0e-12
    assert abs(audit.binary_pattern_power_relative_mismatch) < 2.0e-12
    assert profile.viscous_torque_faces[-1] == 0.0
    assert profile.face_fluxes.mass[-1] == audit.applied_mass_flux
    assert (
        profile.face_fluxes.radial_momentum[-1]
        > audit.pressure_traction
    )


def test_roche_boundary_flux_is_continuous_across_opening_threshold() -> None:
    params = FiducialParams()
    potential = PaczynskiWiitaPotential(params.M2_g)
    geometry = fiducial_hill_roche_nozzle_geometry()
    provider = GasRadiationHillRocheNozzleProvider(
        geometry, transverse_quadrature_zones=16
    )
    radius = 335.0 * potential.r_g
    density = 1.0e-8
    specific_l = float(potential.l_k(radius))

    def availability(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        reservoir = HillRocheNozzleReservoir(
            radius=radius,
            density=density,
            pressure=total_pressure(density, temperature),
            radial_velocity=0.0,
            specific_angular_momentum=specific_l,
            temperature=temperature,
        )
        return provider.available_specific_energy(reservoir)

    threshold = float(
        np.exp(brentq(availability, np.log(1.0e4), np.log(1.0e6)))
    )
    below = _physical_roche_column(0.999 * threshold)
    above = _physical_roche_column(1.001 * threshold)
    below_profile = evaluate_global_rusanov_profile(
        below[1],
        below[2],
        below[0].M2_g,
        boundary_mode="roche_outer",
        stress_boundary_mode="outer_zero_torque",
        outer_overflow_provider=below[4],
    )
    above_profile = evaluate_global_rusanov_profile(
        above[1],
        above[2],
        above[0].M2_g,
        boundary_mode="roche_outer",
        stress_boundary_mode="outer_zero_torque",
        outer_overflow_provider=above[4],
    )
    below_audit = below_profile.outer_roche_boundary
    above_audit = above_profile.outer_roche_boundary
    assert below_audit is not None and above_audit is not None
    assert not below_audit.gate.choked
    assert above_audit.gate.choked
    assert above_audit.applied_mass_flux > 0.0
    assert (
        above_audit.applied_radial_momentum_flux
        - above_audit.pressure_traction
    ) / above_audit.pressure_traction < 2.0e-2
    assert above_audit.applied_mass_flux < 1.0e-2 * (
        2.0
        * np.pi
        * above_audit.edge_state.radius
        * above_audit.edge_state.surface_density
        * above_audit.edge_state.adiabatic_sound_speed
    )


def test_inner_characteristic_audit_counts_subsonic_acoustic_input() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(16, -0.25)
    primitives = recover_global_primitives(grid, state, mass)
    sound_speed = global_inner_characteristic_audit(
        primitives
    ).effective_sound_speed
    inflow = state_from_thermodynamic_primitives(
        grid,
        primitives.surface_density,
        np.full(grid.centers.size, -0.8 * sound_speed),
        primitives.omega,
        primitives.temperature,
        mass,
    )
    audit = global_inner_characteristic_audit(
        recover_global_primitives(grid, inflow, mass)
    )
    assert -1.0 < audit.radial_mach_number < 0.0
    assert audit.incoming_characteristics == 1
    assert all(value < 0.0 for value in audit.eigenvalues[:3])
    assert audit.eigenvalues[3] > 0.0


def test_reference_characteristic_inner_flux_preserves_reference_exactly() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(16, -0.25)
    primitives = recover_global_primitives(grid, state, mass)
    base = evaluate_global_rusanov_profile(
        grid,
        state,
        mass,
        reference_state=state,
        boundary_mode="open_no_inflow",
    )
    selected = evaluate_global_rusanov_profile(
        grid,
        state,
        mass,
        reference_state=state,
        boundary_mode="characteristic_inner_open_outer",
    )
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        np.testing.assert_array_equal(
            getattr(selected.face_fluxes, name), getattr(base.face_fluxes, name)
        )
    assert selected.inner_characteristic_projection.incoming_amplitude_after == 0.0
    assert primitives.radial_velocity[0] == 0.0


def test_reference_characteristic_inner_flux_removes_only_incoming_acoustic_mode() -> None:
    grid, reference_state, mass, _residual = _manufactured_rotating_equilibrium(
        16, -0.25
    )
    reference = recover_global_primitives(grid, reference_state, mass)
    perturbed = state_from_thermodynamic_primitives(
        grid,
        reference.surface_density,
        np.concatenate(([-2.0e5], np.zeros(grid.centers.size - 1))),
        reference.omega,
        np.concatenate(
            (([1.03 * reference.temperature[0]]), reference.temperature[1:])
        ),
        mass,
    )
    primitives = recover_global_primitives(grid, perturbed, mass)
    raw = evaluate_global_rusanov_profile(
        grid,
        perturbed,
        mass,
        boundary_mode="transmissive",
    )
    corrected, audit = apply_global_reference_characteristic_inner_flux(
        grid,
        raw.face_fluxes,
        primitives,
        reference,
        mass,
    )
    scale = max(abs(audit.incoming_amplitude_before), 1.0)
    assert abs(audit.incoming_amplitude_after) < 5.0e-10 * scale
    assert audit.outgoing_amplitude_after == pytest.approx(
        audit.outgoing_amplitude_before, rel=2.0e-10
    )
    assert corrected.mass[0] < 0.0
    projected_l = grid.centers[0] ** 2 * primitives.omega[0]
    assert corrected.angular_momentum[0] / corrected.mass[0] == pytest.approx(
        projected_l, rel=2.0e-10
    )
    potential = PaczynskiWiitaPotential(mass)
    projected_vertical = vertical_state(
        primitives.surface_density[0],
        audit.projected_temperature,
        grid.centers[0],
        potential,
    )
    projected_bernoulli = (
        potential.phi(grid.centers[0])
        + 0.5 * audit.projected_radial_velocity**2
        + 0.5 * (grid.centers[0] * primitives.omega[0]) ** 2
        + projected_vertical.e
        + projected_vertical.Pi / primitives.surface_density[0]
    )
    assert corrected.total_energy[0] / corrected.mass[0] == pytest.approx(
        projected_bernoulli, rel=2.0e-10
    )
    expected_radial = (
        corrected.mass[0] * audit.projected_radial_velocity
        + 2.0 * np.pi * grid.edges[0] * projected_vertical.Pi
    )
    assert corrected.radial_momentum[0] == pytest.approx(
        expected_radial, rel=2.0e-10
    )


def test_reference_characteristic_inner_flux_does_not_reflect_outgoing_mode() -> None:
    grid, reference_state, mass, _residual = _manufactured_rotating_equilibrium(
        16, -0.25
    )
    reference = recover_global_primitives(grid, reference_state, mass)
    temperature = np.array(reference.temperature, copy=True)
    temperature[0] *= 1.01
    thermal_trial = state_from_thermodynamic_primitives(
        grid,
        reference.surface_density,
        reference.radial_velocity,
        reference.omega,
        temperature,
        mass,
    )
    thermal = recover_global_primitives(grid, thermal_trial, mass)
    sound_speed = global_inner_characteristic_audit(
        reference
    ).effective_sound_speed
    pressure_velocity = (
        thermal.vertical.Pi[0] - reference.vertical.Pi[0]
    ) / (reference.surface_density[0] * sound_speed)
    velocity = np.array(reference.radial_velocity, copy=True)
    velocity[0] -= pressure_velocity
    outgoing_state = state_from_thermodynamic_primitives(
        grid,
        reference.surface_density,
        velocity,
        reference.omega,
        temperature,
        mass,
    )
    outgoing = recover_global_primitives(grid, outgoing_state, mass)
    raw = evaluate_global_rusanov_profile(
        grid, outgoing_state, mass, boundary_mode="transmissive"
    )
    corrected, audit = apply_global_reference_characteristic_inner_flux(
        grid, raw.face_fluxes, outgoing, reference, mass
    )
    scale = max(abs(audit.outgoing_amplitude_before), 1.0)
    assert abs(audit.incoming_amplitude_before) < 2.0e-8 * scale
    assert abs(audit.incoming_amplitude_after) < 2.0e-8 * scale
    assert abs(
        audit.outgoing_amplitude_after - audit.outgoing_amplitude_before
    ) < 2.0e-8 * scale
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        np.testing.assert_allclose(
            getattr(corrected, name)[0],
            getattr(raw.face_fluxes, name)[0],
            rtol=2.0e-8,
            atol=0.0,
        )


def test_characteristic_energy_correction_is_continuous_with_nonzero_offset() -> None:
    grid, baseline, mass, _residual = _manufactured_rotating_equilibrium(
        16, -0.25
    )
    base = recover_global_primitives(grid, baseline, mass)
    correction = np.linspace(-3.0e16, 2.0e16, grid.centers.size)
    reference_state = state_from_thermodynamic_primitives(
        grid,
        base.surface_density,
        -0.5
        * global_inner_characteristic_audit(base).effective_sound_speed
        * np.ones(grid.centers.size),
        base.omega,
        base.temperature,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    reference = recover_global_primitives(
        grid,
        reference_state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    exact = evaluate_global_rusanov_profile(
        grid,
        reference_state,
        mass,
        reference_state=reference_state,
        boundary_mode="characteristic_inner_open_outer",
        specific_mechanical_energy_correction=correction,
    )
    unprojected = evaluate_global_rusanov_profile(
        grid,
        reference_state,
        mass,
        reference_state=reference_state,
        boundary_mode="open_no_inflow",
        specific_mechanical_energy_correction=correction,
    )
    np.testing.assert_array_equal(
        exact.face_fluxes.total_energy, unprojected.face_fluxes.total_energy
    )

    energy_corrections = []
    amplitudes = np.asarray([1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6])
    sound_speed = global_inner_characteristic_audit(
        reference
    ).effective_sound_speed
    for amplitude in amplitudes:
        velocity = np.array(reference.radial_velocity, copy=True)
        velocity[0] += amplitude * sound_speed
        state = state_from_thermodynamic_primitives(
            grid,
            reference.surface_density,
            velocity,
            reference.omega,
            reference.temperature,
            mass,
            specific_mechanical_energy_correction=correction,
        )
        raw = evaluate_global_rusanov_profile(
            grid,
            state,
            mass,
            reference_state=reference_state,
            boundary_mode="open_no_inflow",
            specific_mechanical_energy_correction=correction,
        )
        selected = evaluate_global_rusanov_profile(
            grid,
            state,
            mass,
            reference_state=reference_state,
            boundary_mode="characteristic_inner_open_outer",
            specific_mechanical_energy_correction=correction,
        )
        energy_corrections.append(
            selected.face_fluxes.total_energy[0]
            - raw.face_fluxes.total_energy[0]
        )
        scale = max(
            abs(selected.inner_characteristic_projection.incoming_amplitude_before),
            1.0,
        )
        assert abs(
            selected.inner_characteristic_projection.incoming_amplitude_after
        ) < 5.0e-10 * scale
    energy_corrections = np.asarray(energy_corrections)
    assert np.all(np.abs(energy_corrections[1:]) < np.abs(energy_corrections[:-1]))
    normalized = energy_corrections / amplitudes
    assert normalized[-1] == pytest.approx(normalized[-2], rel=2.0e-4)

    rhs = global_conservative_rhs(selected.face_fluxes, selected.cell_sources)
    mass_timescale = float(
        np.min(
            state.mass
            / np.maximum(np.abs(rhs.mass), np.finfo(float).tiny)
        )
    )
    dt = 1.0e-5 * mass_timescale
    updated = GlobalConservativeState(
        **{
            name: getattr(state, name) + dt * getattr(rhs, name)
            for name in (
                "mass",
                "radial_momentum",
                "angular_momentum",
                "total_energy",
            )
        }
    ).validated()
    ledger = audit_global_backward_euler_ledgers(
        updated, state, dt, selected.face_fluxes, selected.cell_sources
    )
    assert ledger.maximum_relative_defect < 1.0e-9


def test_physical_flux_eigensystem_audits_analytic_acoustic_projection() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(16, -0.25)
    base = recover_global_primitives(grid, state, mass)
    sound_speed = global_inner_characteristic_audit(base).effective_sound_speed
    correction = np.linspace(-3.0e16, 2.0e16, grid.centers.size)
    reference_state = state_from_thermodynamic_primitives(
        grid,
        base.surface_density,
        np.full(grid.centers.size, -0.5 * sound_speed),
        base.omega,
        base.temperature,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    reference = recover_global_primitives(
        grid,
        reference_state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    audit = audit_global_physical_flux_eigensystem(
        grid,
        reference,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    assert audit.numerical_eigenvalues[0] < 0.0
    assert audit.numerical_eigenvalues[-1] > 0.0
    assert audit.finite_difference_refinement_defect < 2.0e-4
    assert audit.maximum_analytic_eigenvalue_defect_over_sound_speed < 1.0e-3
    assert audit.incoming_acoustic_left_alignment > 0.999
    assert audit.maximum_biorthogonality_defect < 1.0e-9
    assert audit.maximum_eigenpair_residual < 1.0e-10
    assert 0.0 <= audit.incoming_acoustic_left_alignment <= 1.0


def test_global_radial_column_work_matches_certified_inward_flux_form() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(32, -0.25)
    primitives = recover_global_primitives(grid, state, mass)
    inward_faces = np.geomspace(2.0e18, 3.0e18, grid.edges.size)
    outward_faces = -inward_faces
    global_work = global_vertical_work_rate_cells(
        grid, outward_faces, primitives
    )
    signed_work = signed_vertical_work_rate_cells(
        grid,
        0.5 * (inward_faces[:-1] + inward_faces[1:]),
        primitives.surface_density,
        primitives.vertical.P_tot,
        primitives.vertical.Pi,
        primitives.vertical.rho,
    )
    np.testing.assert_allclose(global_work, signed_work, rtol=2.0e-15)


def test_temporal_column_work_integrates_manufactured_linear_path() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(8, -0.25)
    primitives = recover_global_primitives(grid, state, mass)
    old_x = np.linspace(-0.2, 0.1, grid.centers.size)
    new_x = old_x + np.linspace(0.03, 0.08, grid.centers.size)
    old_q = np.linspace(2.0e30, 3.0e30, grid.centers.size)
    new_q = old_q + np.linspace(0.4e30, 0.7e30, grid.centers.size)
    sigma = primitives.surface_density
    old_vertical = replace(
        primitives.vertical,
        H=np.exp(old_x),
        Pi=old_q * sigma / state.mass,
    )
    new_vertical = replace(
        primitives.vertical,
        H=np.exp(new_x),
        Pi=new_q * sigma / state.mass,
    )
    old_primitives = replace(primitives, vertical=old_vertical)
    new_primitives = replace(primitives, vertical=new_vertical)
    work = global_temporal_vertical_work_cells(
        state, old_primitives, state, new_primitives
    )
    expected = 0.5 * (old_q + new_q) * (new_x - old_x)
    np.testing.assert_allclose(work, expected, rtol=5.0e-15)


def test_column_work_backward_euler_residual_and_ledger_close() -> None:
    mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(12.0 * potential.r_g, 120.0 * potential.r_g, 8)
    sigma = np.geomspace(1.0e5, 3.0e5, grid.centers.size)
    velocity = np.linspace(-2.0e5, 3.0e5, grid.centers.size)
    omega = 0.9 * potential.omega_k(grid.centers)
    old = state_from_thermodynamic_primitives(
        grid, sigma, velocity, omega, np.full(8, 2.0e7), mass
    )
    new = state_from_thermodynamic_primitives(
        grid,
        1.01 * sigma,
        0.98 * velocity,
        1.002 * omega,
        np.full(8, 2.1e7),
        mass,
    )
    old_primitives = recover_global_primitives(grid, old, mass)
    new_primitives = recover_global_primitives(grid, new, mass)
    temporal = global_temporal_vertical_work_cells(
        old, old_primitives, new, new_primitives
    )
    dt = 0.25
    zero_faces = np.zeros(grid.edges.size)
    fluxes = GlobalFaceFluxes(
        mass=zero_faces,
        radial_momentum=zero_faces,
        angular_momentum=zero_faces,
        total_energy=zero_faces,
    )
    sources = GlobalCellSources(
        mass=(new.mass - old.mass) / dt,
        radial_momentum=(new.radial_momentum - old.radial_momentum) / dt,
        angular_momentum=(new.angular_momentum - old.angular_momentum) / dt,
        total_energy=(new.total_energy - old.total_energy + temporal) / dt,
    )
    residual = global_backward_euler_residual(
        new,
        old,
        dt,
        fluxes,
        sources,
        energy_storage_correction=temporal,
    )
    assert np.max(np.abs(residual)) < 1.0e-9 * np.max(np.abs(new.total_energy))
    audit = audit_global_backward_euler_ledgers(
        new,
        old,
        dt,
        fluxes,
        sources,
        energy_storage_correction=temporal,
    )
    assert audit.maximum_relative_defect < 3.0e-16


def test_column_work_does_not_duplicate_torque_flux_or_other_sources() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(16, -0.25)
    primitives = recover_global_primitives(grid, state, mass)
    moving = state_from_thermodynamic_primitives(
        grid,
        primitives.surface_density,
        np.linspace(-3.0e5, 4.0e5, grid.centers.size),
        primitives.omega,
        primitives.temperature,
        mass,
    )
    legacy = evaluate_global_rusanov_profile(
        grid,
        moving,
        mass,
        reference_state=moving,
        alpha=0.1,
        include_vertical_column_work=False,
    )
    selected = evaluate_global_rusanov_profile(
        grid,
        moving,
        mass,
        reference_state=moving,
        alpha=0.1,
        include_vertical_column_work=True,
    )
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        np.testing.assert_array_equal(
            getattr(selected.face_fluxes, name), getattr(legacy.face_fluxes, name)
        )
    for name in ("mass", "radial_momentum", "angular_momentum"):
        np.testing.assert_array_equal(
            getattr(selected.cell_sources, name), getattr(legacy.cell_sources, name)
        )
    np.testing.assert_allclose(
        selected.cell_sources.total_energy - legacy.cell_sources.total_energy,
        selected.vertical_work_rate_cells,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_common_alpha_stress_adds_paired_outward_fluxes_once() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(32, -0.25)
    profile = evaluate_global_inviscid_profile(grid, state, mass)
    stressed, torque = add_global_alpha_stress_fluxes(
        grid,
        profile.face_fluxes,
        profile.primitives,
        alpha=0.1,
    )
    expected_centers = (
        2.0
        * np.pi
        * grid.centers**2
        * np.asarray(
            integrated_stress(profile.primitives.vertical, alpha=0.1),
            dtype=float,
        )
    )
    expected_torque = positive_edge_reconstruction(grid, expected_centers)
    omega_faces = positive_edge_reconstruction(
        grid, profile.primitives.omega
    )
    np.testing.assert_allclose(torque, expected_torque, rtol=2.0e-15)
    np.testing.assert_array_equal(stressed.mass, profile.face_fluxes.mass)
    np.testing.assert_array_equal(
        stressed.radial_momentum, profile.face_fluxes.radial_momentum
    )
    np.testing.assert_allclose(
        stressed.angular_momentum - profile.face_fluxes.angular_momentum,
        torque,
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        stressed.total_energy - profile.face_fluxes.total_energy,
        omega_faces * torque,
        rtol=2.0e-15,
    )


def test_common_alpha_stress_zero_torque_boundary_is_explicit() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(16, -0.25)
    primitives = recover_global_primitives(grid, state, mass)
    extrapolated = global_alpha_stress_torque_faces(
        grid, primitives, alpha=0.1
    )
    zero_torque = global_alpha_stress_torque_faces(
        grid, primitives, alpha=0.1, boundary_mode="zero_torque"
    )
    outer_zero = global_alpha_stress_torque_faces(
        grid, primitives, alpha=0.1, boundary_mode="outer_zero_torque"
    )
    assert extrapolated[0] > 0.0
    assert extrapolated[-1] > 0.0
    assert zero_torque[0] == 0.0
    assert zero_torque[-1] == 0.0
    assert outer_zero[0] == extrapolated[0]
    assert outer_zero[-1] == 0.0
    np.testing.assert_array_equal(zero_torque[1:-1], extrapolated[1:-1])


def test_radiative_cooling_is_only_a_total_energy_sink() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(8, 0.0)
    adiabatic = evaluate_global_inviscid_profile(grid, state, mass)
    cooling = evaluate_global_inviscid_profile(
        grid, state, mass, include_radiative_cooling=True
    )
    expected = (
        np.asarray(radiative_cooling(cooling.primitives.vertical), dtype=float)
        * grid.area
    )
    np.testing.assert_allclose(
        global_radiative_cooling_rate_cells(grid, cooling.primitives),
        expected,
        rtol=2.0e-15,
    )
    np.testing.assert_array_equal(
        cooling.cell_sources.mass, adiabatic.cell_sources.mass
    )
    np.testing.assert_array_equal(
        cooling.cell_sources.radial_momentum,
        adiabatic.cell_sources.radial_momentum,
    )
    np.testing.assert_array_equal(
        cooling.cell_sources.angular_momentum,
        adiabatic.cell_sources.angular_momentum,
    )
    np.testing.assert_allclose(
        cooling.cell_sources.total_energy,
        -expected,
        rtol=2.0e-15,
    )


def test_compact_stream_moments_are_exact_across_meshes() -> None:
    total_mass_rate = 7.5e20
    radial_velocity = -2.0e7
    specific_l = 3.0e19
    specific_energy = -4.0e18
    active_cells = []
    for n_cells in (16, 32, 64):
        grid = make_log_grid(100.0, 300.0, n_cells)
        source = global_compact_stream_cell_sources(
            grid,
            total_mass_rate,
            center=200.0,
            log_width=0.15,
            specific_radial_velocity=radial_velocity,
            specific_angular_momentum=specific_l,
            specific_total_energy=specific_energy,
        )
        np.testing.assert_allclose(
            np.sum(source.mass), total_mass_rate, rtol=2.0e-15
        )
        np.testing.assert_allclose(
            np.sum(source.radial_momentum),
            total_mass_rate * radial_velocity,
            rtol=2.0e-15,
        )
        np.testing.assert_allclose(
            np.sum(source.angular_momentum),
            total_mass_rate * specific_l,
            rtol=2.0e-15,
        )
        np.testing.assert_allclose(
            np.sum(source.total_energy),
            total_mass_rate * specific_energy,
            rtol=2.0e-15,
        )
        active_cells.append(int(np.count_nonzero(source.mass)))
    assert active_cells[0] < active_cells[1] < active_cells[2]


def test_monolithic_backward_euler_accepts_exact_stream_moments() -> None:
    grid, initial, mass, _residual = _manufactured_rotating_equilibrium(8, 0.0)
    primitives = recover_global_primitives(grid, initial, mass)
    dt = 1.0
    center = float(np.sqrt(grid.edges[0] * grid.edges[-1]))
    source_rate = float(np.sum(initial.mass) * 1.0e-8 / dt)
    source = global_compact_stream_cell_sources(
        grid,
        source_rate,
        center=center,
        log_width=0.15,
        specific_radial_velocity=0.0,
        specific_angular_momentum=float(
            center**2
            * np.interp(
                np.log(center), np.log(grid.centers), primitives.omega
            )
        ),
        specific_total_energy=float(
            np.median(primitives.specific_total_energy)
        ),
    )
    result = advance_global_backward_euler(
        grid,
        initial,
        mass,
        dt,
        reference_state=initial,
        boundary_mode="transmissive",
        external_sources=source,
    )
    assert result.accepted, result.message
    assert result.maximum_scaled_residual < 1.0e-8
    assert result.maximum_storage_scaled_ledger_defect < 1.0e-8
    np.testing.assert_allclose(np.sum(source.mass), source_rate, rtol=2.0e-15)
    recovered = recover_global_primitives(grid, result.state, mass)
    assert np.all(recovered.surface_density > 0.0)
    assert np.all(recovered.temperature > 0.0)


def test_local_implicit_cooling_is_conservative_and_first_order() -> None:
    grid, initial, mass, _residual = _manufactured_rotating_equilibrium(8, 0.0)
    primitives = recover_global_primitives(grid, initial, mass)
    cooling_rate = global_radiative_cooling_rate_cells(grid, primitives)
    internal_energy = initial.mass * primitives.specific_internal_energy
    total_time = 0.02 * float(np.min(internal_energy / cooling_rate))
    states = []
    for n_steps in (1, 2, 4):
        current = initial
        for _ in range(n_steps):
            result = advance_global_radiative_cooling_backward_euler(
                grid, current, mass, total_time / n_steps
            )
            assert result.accepted, result.message
            assert result.maximum_storage_scaled_ledger_defect < 1.0e-12
            current = result.state
        states.append(current)
    for state in states:
        np.testing.assert_array_equal(state.mass, initial.mass)
        np.testing.assert_array_equal(
            state.radial_momentum, initial.radial_momentum
        )
        np.testing.assert_array_equal(
            state.angular_momentum, initial.angular_momentum
        )
    temperatures = [
        recover_global_primitives(grid, state, mass).temperature
        for state in states
    ]
    assert np.all(temperatures[-1] < primitives.temperature)
    coarse_error = float(
        np.max(np.abs(temperatures[0] / temperatures[1] - 1.0))
    )
    fine_error = float(
        np.max(np.abs(temperatures[1] / temperatures[2] - 1.0))
    )
    assert coarse_error / fine_error > 1.5


def test_monolithic_backward_euler_resolves_radiative_cooling() -> None:
    grid, initial, mass, _residual = _manufactured_rotating_equilibrium(8, 0.0)
    common = dict(
        alpha=0.0,
        reference_state=initial,
        boundary_mode="open_no_inflow",
    )
    adiabatic = advance_global_backward_euler(
        grid, initial, mass, 10.0, **common
    )
    cooled = advance_global_backward_euler(
        grid,
        initial,
        mass,
        10.0,
        include_radiative_cooling=True,
        **common,
    )
    assert adiabatic.accepted, adiabatic.message
    assert cooled.accepted, cooled.message
    adiabatic_temperature = recover_global_primitives(
        grid, adiabatic.state, mass
    ).temperature
    cooled_temperature = recover_global_primitives(
        grid, cooled.state, mass
    ).temperature
    assert np.all(cooled_temperature < adiabatic_temperature)
    assert np.max(adiabatic_temperature - cooled_temperature) > 1.0e-4


def test_implicit_common_stress_step_closes_angular_and_energy_ledgers() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(8, -0.25)
    primitives = recover_global_primitives(grid, state, mass)
    dt = global_inviscid_cfl_timestep(grid, primitives, cfl=0.01)
    result = advance_global_alpha_stress_backward_euler(
        grid,
        state,
        mass,
        dt,
        alpha=0.1,
        stress_boundary_mode="zero_torque",
    )
    assert result.accepted, result.message
    assert result.maximum_scaled_residual < 1.0e-9
    assert result.ledger.maximum_relative_defect < 1.0e-8
    np.testing.assert_array_equal(result.state.mass, state.mass)
    np.testing.assert_array_equal(
        result.state.radial_momentum, state.radial_momentum
    )
    assert np.any(result.state.angular_momentum != state.angular_momentum)
    assert np.any(result.state.total_energy != state.total_energy)
    recovered = recover_global_primitives(grid, result.state, mass)
    assert np.all(recovered.omega > 0.0)
    assert np.all(recovered.temperature > 0.0)


def test_rejected_implicit_stress_step_returns_original_state() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(8, -0.25)
    primitives = recover_global_primitives(grid, state, mass)
    dt = global_inviscid_cfl_timestep(grid, primitives, cfl=0.01)
    result = advance_global_alpha_stress_backward_euler(
        grid,
        state,
        mass,
        dt,
        alpha=0.1,
        stress_boundary_mode="zero_torque",
        residual_tolerance=1.0e-30,
        ledger_tolerance=1.0e-30,
        max_nfev=1,
    )
    assert not result.accepted
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        np.testing.assert_array_equal(
            getattr(result.state, name), getattr(state, name)
        )


def test_split_imex_rejects_before_returning_a_nonphysical_state() -> None:
    grid, initial, mass, _residual = _manufactured_rotating_equilibrium(8, -0.25)
    primitives = recover_global_primitives(grid, initial, mass)
    dt = global_inviscid_cfl_timestep(grid, primitives, cfl=0.002)
    current = initial
    accepted_steps = 0
    for _ in range(8):
        old = current
        result = advance_global_imex(
            grid,
            current,
            mass,
            dt,
            alpha=0.1,
            reference_state=initial,
            boundary_mode="open_no_inflow",
            stress_boundary_mode="zero_torque",
        )
        if not result.accepted:
            for name in (
                "mass",
                "radial_momentum",
                "angular_momentum",
                "total_energy",
            ):
                np.testing.assert_array_equal(
                    getattr(result.state, name), getattr(old, name)
                )
            break
        assert result.stress is not None
        assert result.maximum_storage_scaled_ledger_defect < 1.0e-12
        current = result.state
        accepted_steps += 1
    assert 0 < accepted_steps < 8


def test_rejected_imex_step_returns_original_state() -> None:
    grid, state, mass = _discontinuous_global_state()
    primitives = recover_global_primitives(grid, state, mass)
    dt = 1.0e6 * global_inviscid_cfl_timestep(grid, primitives, cfl=0.5)
    result = advance_global_imex(grid, state, mass, dt, alpha=0.1)
    assert not result.accepted
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        np.testing.assert_array_equal(
            getattr(result.state, name), getattr(state, name)
        )


def test_monolithic_backward_euler_advances_source_free_stress_state() -> None:
    grid, initial, mass, _residual = _manufactured_rotating_equilibrium(8, -0.25)
    primitives = recover_global_primitives(grid, initial, mass)
    dt = global_inviscid_cfl_timestep(grid, primitives, cfl=0.002)
    current = initial
    for _ in range(4):
        result = advance_global_backward_euler(
            grid,
            current,
            mass,
            dt,
            alpha=0.1,
            reference_state=initial,
            boundary_mode="open_no_inflow",
            stress_boundary_mode="zero_torque",
        )
        assert result.accepted, result.message
        assert result.maximum_scaled_residual < 1.0e-8
        assert result.maximum_storage_scaled_ledger_defect < 1.0e-8
        current = result.state
    recovered = recover_global_primitives(grid, current, mass)
    assert np.all(recovered.surface_density > 0.0)
    assert np.all(recovered.omega > 0.0)
    assert np.all(recovered.temperature > 0.0)


def test_colored_jacobian_is_rejected_under_thermal_energy_scaling() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(8, -0.25)
    primitives = recover_global_primitives(grid, state, mass)
    dt = global_inviscid_cfl_timestep(grid, primitives, cfl=0.002)
    common = dict(
        alpha=0.1,
        reference_state=state,
        boundary_mode="open_no_inflow",
        stress_boundary_mode="zero_torque",
    )
    sparse = advance_global_backward_euler(
        grid, state, mass, dt, use_sparse_jacobian=True, **common
    )
    dense = advance_global_backward_euler(
        grid, state, mass, dt, use_sparse_jacobian=False, **common
    )
    default = advance_global_backward_euler(grid, state, mass, dt, **common)
    assert not sparse.accepted
    assert dense.accepted, dense.message
    assert default.accepted, default.message
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        np.testing.assert_array_equal(
            getattr(sparse.state, name), getattr(state, name)
        )
        np.testing.assert_array_equal(
            getattr(default.state, name), getattr(dense.state, name)
        )


def test_sparse_forward_jacobian_is_directionally_certified() -> None:
    grid, state, mass, _residual = _manufactured_rotating_equilibrium(8, -0.25)
    base = recover_global_primitives(grid, state, mass)
    state = state_from_thermodynamic_primitives(
        grid,
        base.surface_density,
        np.full(grid.centers.size, -1.0e-5 * C),
        base.omega,
        base.temperature,
        mass,
    )
    primitives = recover_global_primitives(grid, state, mass)
    dt = global_inviscid_cfl_timestep(grid, primitives, cfl=0.002)
    common = dict(
        alpha=0.1,
        reference_state=state,
        boundary_mode="transmissive",
        stress_boundary_mode="zero_torque",
    )
    sparse = advance_global_backward_euler(
        grid,
        state,
        mass,
        dt,
        jacobian_mode="sparse_forward",
        **common,
    )
    dense = advance_global_backward_euler(
        grid, state, mass, dt, jacobian_mode="dense", **common
    )
    assert sparse.jacobian_audit is not None
    assert sparse.jacobian_audit.accepted
    assert sparse.jacobian_audit.directions == 4 * grid.centers.size
    assert sparse.jacobian_audit.maximum_relative_defect < 1.0e-4
    assert sparse.accepted, sparse.message
    assert dense.accepted, dense.message
    sparse_primitives = recover_global_primitives(grid, sparse.state, mass)
    dense_primitives = recover_global_primitives(grid, dense.state, mass)
    np.testing.assert_allclose(
        sparse_primitives.surface_density,
        dense_primitives.surface_density,
        rtol=1.0e-8,
    )
    np.testing.assert_allclose(
        sparse_primitives.radial_velocity,
        dense_primitives.radial_velocity,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        sparse_primitives.omega,
        dense_primitives.omega,
        rtol=1.0e-8,
    )
    np.testing.assert_allclose(
        sparse_primitives.temperature,
        dense_primitives.temperature,
        rtol=1.0e-6,
    )


def _discontinuous_global_state(n_cells: int = 32):
    mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(10.0 * potential.r_g, 40.0 * potential.r_g, n_cells)
    left = np.arange(n_cells) < n_cells // 2
    sigma = np.where(left, 8.0e6, 2.0e6)
    temperature = np.where(left, 2.0e7, 8.0e6)
    radial_velocity = np.where(left, 2.0e-4 * C, -1.0e-4 * C)
    omega = 0.85 * potential.omega_k(grid.centers)
    state = state_from_thermodynamic_primitives(
        grid, sigma, radial_velocity, omega, temperature, mass
    )
    return grid, state, mass


def test_rusanov_step_preserves_positive_thermal_state_and_ledgers() -> None:
    grid, state, mass = _discontinuous_global_state()
    primitives = recover_global_primitives(grid, state, mass)
    dt = global_inviscid_cfl_timestep(grid, primitives, cfl=0.005)
    result = advance_global_inviscid_rusanov(
        grid, state, mass, dt, boundary_mode="open_no_inflow"
    )
    assert result.accepted, result.message
    assert result.ledger.maximum_relative_defect < 1.0e-8
    recovered = recover_global_primitives(grid, result.state, mass)
    assert np.all(recovered.surface_density > 0.0)
    assert np.all(recovered.temperature > 0.0)


def test_rusanov_rejects_nonphysical_step_without_clipping() -> None:
    grid, state, mass = _discontinuous_global_state()
    primitives = recover_global_primitives(grid, state, mass)
    dt = global_inviscid_cfl_timestep(grid, primitives, cfl=0.5)
    result = advance_global_inviscid_rusanov(grid, state, mass, 1.0e6 * dt)
    assert not result.accepted
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        np.testing.assert_array_equal(
            getattr(result.state, name), getattr(state, name)
        )


def test_open_boundaries_block_unconfigured_advective_inflow() -> None:
    grid, state, mass = _discontinuous_global_state()
    profile = evaluate_global_rusanov_profile(
        grid, state, mass, boundary_mode="open_no_inflow"
    )
    assert profile.face_fluxes.mass[0] == 0.0
    assert profile.face_fluxes.mass[-1] == 0.0
    assert profile.face_fluxes.angular_momentum[0] == 0.0
    assert profile.face_fluxes.angular_momentum[-1] == 0.0
    assert profile.face_fluxes.total_energy[0] == 0.0
    assert profile.face_fluxes.total_energy[-1] == 0.0
    assert profile.face_fluxes.radial_momentum[0] > 0.0
    assert profile.face_fluxes.radial_momentum[-1] > 0.0


def test_open_boundaries_retain_outgoing_fluxes() -> None:
    grid, state, mass = _discontinuous_global_state()
    primitives = recover_global_primitives(grid, state, mass)
    outward_state = state_from_thermodynamic_primitives(
        grid,
        primitives.surface_density,
        -primitives.radial_velocity,
        primitives.omega,
        primitives.temperature,
        mass,
    )
    profile = evaluate_global_rusanov_profile(
        grid, outward_state, mass, boundary_mode="open_no_inflow"
    )
    assert profile.face_fluxes.mass[0] < 0.0
    assert profile.face_fluxes.mass[-1] > 0.0


def _evolve_inviscid_fixed_steps(
    grid,
    state,
    mass,
    reference_state,
    total_time: float,
    n_steps: int,
):
    current = state
    for _ in range(n_steps):
        result = advance_global_inviscid_rusanov(
            grid,
            current,
            mass,
            total_time / n_steps,
            reference_state=reference_state,
            boundary_mode="open_no_inflow",
        )
        assert result.accepted, result.message
        current = result.state
    return current


def _flatten_conserved_state(state: GlobalConservativeState) -> np.ndarray:
    return np.concatenate(
        tuple(
            getattr(state, name)
            for name in (
                "mass",
                "radial_momentum",
                "angular_momentum",
                "total_energy",
            )
        )
    )


def test_source_free_rusanov_timestep_convergence() -> None:
    grid, reference, mass, _residual = _manufactured_rotating_equilibrium(
        32, -0.25
    )
    initial = reference
    initial_primitives = recover_global_primitives(grid, initial, mass)
    total_time = global_inviscid_cfl_timestep(
        grid, initial_primitives, cfl=0.02
    )
    states = [
        _evolve_inviscid_fixed_steps(
            grid, initial, mass, reference, total_time, n_steps
        )
        for n_steps in (1, 2, 4)
    ]
    vectors = [_flatten_conserved_state(state) for state in states]
    radial_scale = float(
        np.max(initial.mass)
        * np.max(
            np.abs(initial_primitives.radial_velocity)
            + np.sqrt(
                (
                    (5.0 / 3.0)
                    * np.asarray(initial_primitives.vertical.P_gas)
                    + (4.0 / 3.0)
                    * np.asarray(initial_primitives.vertical.P_rad)
                )
                / np.asarray(initial_primitives.vertical.rho)
            )
        )
    )
    scale = np.concatenate(
        (
            np.full(initial.n_cells, np.max(initial.mass)),
            np.full(initial.n_cells, radial_scale),
            np.full(initial.n_cells, np.max(np.abs(initial.angular_momentum))),
            np.full(initial.n_cells, np.max(np.abs(initial.total_energy))),
        )
    )
    coarse_error = float(np.max(np.abs(vectors[0] - vectors[1]) / scale))
    fine_error = float(np.max(np.abs(vectors[1] - vectors[2]) / scale))
    assert fine_error > 0.0
    assert coarse_error / fine_error > 1.7


def test_source_free_equilibrium_interior_drift_converges_with_mesh() -> None:
    fine_grid, fine_reference, mass, _residual = (
        _manufactured_rotating_equilibrium(64, -0.25)
    )
    fine_primitives = recover_global_primitives(fine_grid, fine_reference, mass)
    total_time = global_inviscid_cfl_timestep(
        fine_grid, fine_primitives, cfl=0.005
    )
    maximum_drifts = []
    interior_drifts = []
    for n_cells in (16, 32, 64):
        grid, reference, local_mass, _local_residual = (
            _manufactured_rotating_equilibrium(n_cells, -0.25)
        )
        evolved = _evolve_inviscid_fixed_steps(
            grid, reference, local_mass, reference, total_time, 4
        )
        primitives = recover_global_primitives(grid, evolved, local_mass)
        maximum_drifts.append(
            float(np.max(np.abs(primitives.radial_velocity)) / C)
        )
        interior_drifts.append(
            float(np.max(np.abs(primitives.radial_velocity[2:-2])) / C)
        )
    assert interior_drifts[1] < interior_drifts[0] / 2.5
    assert interior_drifts[2] < interior_drifts[1] / 2.5
    assert max(maximum_drifts) < 4.0e-10
    assert maximum_drifts[2] < maximum_drifts[0]
