"""Map the canonical physical open control into the one-domain solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalConservativeState,
    PaczynskiWiitaPotential,
    advance_global_backward_euler,
    evaluate_global_rusanov_profile,
    evaluate_coupled_open_overflow_residual,
    fiducial_hill_tidal_geometry,
    global_compact_stream_cell_sources,
    global_inner_characteristic_audit,
    global_outer_characteristic_audit,
    global_radiative_cooling_rate_cells,
    make_log_grid,
    recover_global_primitives,
    state_from_thermodynamic_primitives,
    vertical_state,
)
from imri_qpe.scales import eddington_luminosity, eddington_mdot

from run_coupled_inner_outer_mesh_certification import _load_source
from run_coupled_open_overflow_continuation import _open_context, _target_mesh


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical/coupled_open_overflow_eigenvalue"
OUTPUT = ROOT / "outputs/tables/global_physical_open_preflight.json"
DT_LOADING_FRACTION = 1.0e-9


def _canonical_open_evaluation():
    base, _wall_state = _load_source()
    context_96 = _open_context(base, 1.0)
    with np.load(CANONICAL / "Ninner96_Nouter64.npz") as data:
        state_96 = np.asarray(data["state"], dtype=float)
    context_144, _seed = _target_mesh(context_96, state_96, 144, 96)
    with np.load(CANONICAL / "Ninner144_Nouter96.npz") as data:
        state_144 = np.asarray(data["state"], dtype=float)
    return context_144, evaluate_coupled_open_overflow_residual(
        state_144, context_144
    )


def _mapped_global_state(context, evaluation, n_cells: int):
    inner = evaluation.base.inner_profile
    outer = evaluation.base.outer_transport
    energy = evaluation.base.outer_energy_profile
    outer_grid = context.base.outer_grid
    radius = np.concatenate((inner.R, outer_grid.centers))
    sigma = np.concatenate((inner.Sigma, outer.surface_density))
    radial_velocity = np.concatenate((-inner.u, energy.radial_velocity))
    omega = np.concatenate((inner.Omega, outer.omega))
    temperature = np.concatenate((inner.T, energy.temperature))
    grid = make_log_grid(inner.R[0], outer_grid.edges[-1], n_cells)
    log_radius = np.log(radius)
    mapped_sigma = np.exp(
        np.interp(np.log(grid.centers), log_radius, np.log(sigma))
    )
    mapped_temperature = np.exp(
        np.interp(np.log(grid.centers), log_radius, np.log(temperature))
    )
    mapped_omega = np.exp(
        np.interp(np.log(grid.centers), log_radius, np.log(omega))
    )
    mapped_velocity = np.interp(
        np.log(grid.centers), log_radius, radial_velocity
    )
    state = state_from_thermodynamic_primitives(
        grid,
        mapped_sigma,
        mapped_velocity,
        mapped_omega,
        mapped_temperature,
        context.base.inner_params.M2_g,
    )
    return grid, state, np.zeros(n_cells, dtype=float)


def _extrapolating_interpolation(query, nodes, values, *, positive: bool):
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    query = np.asarray(query, dtype=float)
    work = np.log(values) if positive else values
    result = np.interp(query, nodes, work)
    left = query < nodes[0]
    right = query > nodes[-1]
    if np.any(left):
        slope = (work[1] - work[0]) / (nodes[1] - nodes[0])
        result[left] = work[0] + slope * (query[left] - nodes[0])
    if np.any(right):
        slope = (work[-1] - work[-2]) / (nodes[-1] - nodes[-2])
        result[right] = work[-1] + slope * (query[right] - nodes[-1])
    return np.exp(result) if positive else result


def _conservatively_mapped_global_state(
    context, evaluation, n_cells: int, *, quadrature_order: int = 32
):
    inner = evaluation.base.inner_profile
    outer = evaluation.base.outer_transport
    energy = evaluation.base.outer_energy_profile
    outer_grid = context.base.outer_grid
    radius = np.concatenate((inner.R, outer_grid.centers))
    log_radius = np.log(radius)
    sigma = np.concatenate((inner.Sigma, outer.surface_density))
    radial_velocity = np.concatenate((-inner.u, energy.radial_velocity))
    omega = np.concatenate((inner.Omega, outer.omega))
    temperature = np.concatenate((inner.T, energy.temperature))
    grid = make_log_grid(inner.R[0], outer_grid.edges[-1], n_cells)
    mass = context.base.inner_params.M2_g
    potential = PaczynskiWiitaPotential(mass)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    components = [np.empty(n_cells, dtype=float) for _ in range(4)]
    mechanical_energy = np.empty(n_cells, dtype=float)
    internal_energy = np.empty(n_cells, dtype=float)
    for index, (left, right) in enumerate(
        zip(np.log(grid.edges[:-1]), np.log(grid.edges[1:]))
    ):
        x = 0.5 * (left + right) + 0.5 * (right - left) * nodes
        r = np.exp(x)
        sigma_q = _extrapolating_interpolation(
            x, log_radius, sigma, positive=True
        )
        velocity_q = _extrapolating_interpolation(
            x, log_radius, radial_velocity, positive=False
        )
        omega_q = _extrapolating_interpolation(
            x, log_radius, omega, positive=True
        )
        temperature_q = _extrapolating_interpolation(
            x, log_radius, temperature, positive=True
        )
        vertical = vertical_state(
            sigma_q, temperature_q, r, potential
        )
        specific_total = (
            np.asarray(potential.phi(r), dtype=float)
            + 0.5 * velocity_q**2
            + 0.5 * (r * omega_q) ** 2
            + np.asarray(vertical.e, dtype=float)
        )
        specific_mechanical = (
            np.asarray(potential.phi(r), dtype=float)
            + 0.5 * velocity_q**2
            + 0.5 * (r * omega_q) ** 2
        )
        measure = 2.0 * np.pi * r**2 * sigma_q
        integrands = (
            measure,
            measure * velocity_q,
            measure * r**2 * omega_q,
            measure * specific_total,
        )
        for component, integrand in zip(components, integrands):
            component[index] = (
                0.5 * (right - left) * np.sum(weights * integrand)
            )
        mechanical_energy[index] = (
            0.5
            * (right - left)
            * np.sum(weights * measure * specific_mechanical)
        )
        internal_energy[index] = (
            0.5
            * (right - left)
            * np.sum(weights * measure * np.asarray(vertical.e, dtype=float))
        )
    state = GlobalConservativeState(*components).validated()
    cell_velocity = state.radial_momentum / state.mass
    cell_omega = state.angular_momentum / (state.mass * grid.centers**2)
    center_mechanical = (
        np.asarray(potential.phi(grid.centers), dtype=float)
        + 0.5 * cell_velocity**2
        + 0.5 * (grid.centers * cell_omega) ** 2
    )
    averaged_mechanical = mechanical_energy / state.mass
    correction = np.asarray(averaged_mechanical - center_mechanical, dtype=float)
    averaged_internal = internal_energy / state.mass
    if np.any(~np.isfinite(correction)) or np.any(averaged_internal <= 0.0):
        raise ValueError("finite-volume energy reference is not physical")
    return grid, state, correction


def _initial_case(
    context,
    evaluation,
    n_cells: int,
    *,
    mapping_mode: str = "conservative",
    open_face_reconstruction: str = "primitive_product",
    include_vertical_column_work: bool = False,
    boundary_mode: str = "open_no_inflow",
):
    if mapping_mode == "conservative":
        grid, initial, mechanical_correction = _conservatively_mapped_global_state(
            context, evaluation, n_cells
        )
    elif mapping_mode == "pointwise":
        grid, initial, mechanical_correction = _mapped_global_state(
            context, evaluation, n_cells
        )
    else:
        raise ValueError("mapping_mode must be conservative or pointwise")
    mass = context.base.inner_params.M2_g
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = 5.0 * eddington_mdot(mass)
    circularization_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(circularization_radius))
    stream_energy = float(
        potential.phi(circularization_radius)
        + 0.5 * (stream_l / circularization_radius) ** 2
    )
    stream = global_compact_stream_cell_sources(
        grid,
        stream_rate,
        center=240.0 * potential.r_g,
        log_width=0.08,
        specific_radial_velocity=0.0,
        specific_angular_momentum=stream_l,
        specific_total_energy=stream_energy,
    )
    initial_primitives = recover_global_primitives(
        grid,
        initial,
        mass,
        specific_mechanical_energy_correction=mechanical_correction,
    )
    characteristic = global_outer_characteristic_audit(initial_primitives)
    inner_characteristic = global_inner_characteristic_audit(
        initial_primitives
    )
    hill_geometry = fiducial_hill_tidal_geometry()
    profile_without_stream = evaluate_global_rusanov_profile(
        grid,
        initial,
        mass,
        reference_state=initial,
        boundary_mode=boundary_mode,
        alpha=context.base.alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=include_vertical_column_work,
        open_face_reconstruction=open_face_reconstruction,
        specific_mechanical_energy_correction=mechanical_correction,
    )
    profile_with_stream = evaluate_global_rusanov_profile(
        grid,
        initial,
        mass,
        reference_state=initial,
        boundary_mode=boundary_mode,
        alpha=context.base.alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=include_vertical_column_work,
        external_sources=stream,
        open_face_reconstruction=open_face_reconstruction,
        specific_mechanical_energy_correction=mechanical_correction,
    )
    donor_l = float(grid.centers[-1] ** 2 * initial_primitives.omega[-1])
    donor_bernoulli = float(
        initial_primitives.specific_total_energy[-1]
        + initial_primitives.vertical.Pi[-1]
        / initial_primitives.surface_density[-1]
    )
    outer_mass_flux = float(profile_with_stream.face_fluxes.mass[-1])
    expected_outer_radial = float(
        outer_mass_flux * initial_primitives.radial_velocity[-1]
        + 2.0
        * np.pi
        * grid.edges[-1]
        * initial_primitives.vertical.Pi[-1]
    )
    expected_outer_angular = outer_mass_flux * donor_l
    expected_outer_energy = outer_mass_flux * donor_bernoulli

    def relative_defect(actual: float, expected: float) -> float:
        return float(
            abs(actual - expected) / max(abs(actual), abs(expected), 1.0)
        )

    mapping = {
        "mapping_mode": mapping_mode,
        "open_face_reconstruction": open_face_reconstruction,
        "include_vertical_column_work": include_vertical_column_work,
        "boundary_mode": boundary_mode,
        "n_cells": n_cells,
        "active_source_cells": int(np.count_nonzero(stream.mass)),
        "inner_mass_flux_over_supply": float(
            profile_with_stream.face_fluxes.mass[0] / stream_rate
        ),
        "outer_mass_flux_over_supply": float(
            outer_mass_flux / stream_rate
        ),
        "outer_radial_flux_donor_consistency": relative_defect(
            float(profile_with_stream.face_fluxes.radial_momentum[-1]),
            expected_outer_radial,
        ),
        "outer_angular_flux_donor_consistency": relative_defect(
            float(profile_with_stream.face_fluxes.angular_momentum[-1]),
            expected_outer_angular,
        ),
        "outer_energy_flux_donor_consistency": relative_defect(
            float(profile_with_stream.face_fluxes.total_energy[-1]),
            expected_outer_energy,
        ),
        "maximum_face_mass_flux_change_from_stream_source": float(
            np.max(
                np.abs(
                    profile_with_stream.face_fluxes.mass
                    - profile_without_stream.face_fluxes.mass
                )
            )
            / stream_rate
        ),
        "inner_cell_radial_velocity_c": float(
            initial_primitives.radial_velocity[0] / 2.99792458e10
        ),
        "outer_cell_radial_velocity_c": float(
            initial_primitives.radial_velocity[-1] / 2.99792458e10
        ),
        "outer_characteristic_audit": {
            "radial_velocity": characteristic.radial_velocity,
            "effective_sound_speed": characteristic.effective_sound_speed,
            "radial_mach_number": characteristic.radial_mach_number,
            "eigenvalues": list(characteristic.eigenvalues),
            "incoming_characteristics": (
                characteristic.incoming_characteristics
            ),
        },
        "inner_characteristic_audit": {
            "radial_velocity": inner_characteristic.radial_velocity,
            "effective_sound_speed": (
                inner_characteristic.effective_sound_speed
            ),
            "radial_mach_number": inner_characteristic.radial_mach_number,
            "eigenvalues": list(inner_characteristic.eigenvalues),
            "incoming_characteristics": (
                inner_characteristic.incoming_characteristics
            ),
        },
        "outer_boundary_geometry": {
            "outer_radius_over_hill_radius": float(
                grid.edges[-1] / hill_geometry.hill_radius
            ),
            "outer_radius_over_fiducial_truncation_radius": float(
                grid.edges[-1] / hill_geometry.truncation_radius
            ),
            "is_roche_saddle": False,
            "exterior_thermodynamic_state_declared": False,
            "characteristic_contract_closed": False,
        },
        "integrated_vertical_work_over_eddington_luminosity": float(
            np.sum(profile_with_stream.vertical_work_rate_cells)
            / eddington_luminosity(mass)
        ),
        "maximum_absolute_specific_mechanical_energy_correction": float(
            np.max(np.abs(mechanical_correction))
        ),
        "minimum_recovered_specific_internal_energy": float(
            np.min(initial_primitives.specific_internal_energy)
        ),
    }
    return (
        grid,
        initial,
        mass,
        potential,
        stream_rate,
        stream,
        initial_primitives,
        mechanical_correction,
        mapping,
    )


def _mapping_attempt(
    context,
    evaluation,
    n_cells: int,
    *,
    mapping_mode: str,
    open_face_reconstruction: str = "primitive_product",
    include_vertical_column_work: bool = False,
    boundary_mode: str = "open_no_inflow",
):
    try:
        return {
            "accepted": True,
            **_initial_case(
                context,
                evaluation,
                n_cells,
                mapping_mode=mapping_mode,
                open_face_reconstruction=open_face_reconstruction,
                include_vertical_column_work=include_vertical_column_work,
                boundary_mode=boundary_mode,
            )[-1],
        }
    except ValueError as error:
        return {
            "accepted": False,
            "mapping_mode": mapping_mode,
            "open_face_reconstruction": open_face_reconstruction,
            "boundary_mode": boundary_mode,
            "n_cells": n_cells,
            "message": str(error),
        }


def _mechanical_reference_quadrature_audit(
    context, evaluation, n_cells: int
) -> dict[str, float | int]:
    grid_32, state_32, correction_32 = _conservatively_mapped_global_state(
        context, evaluation, n_cells, quadrature_order=32
    )
    grid_64, state_64, correction_64 = _conservatively_mapped_global_state(
        context, evaluation, n_cells, quadrature_order=64
    )
    if not np.array_equal(grid_32.edges, grid_64.edges):
        raise RuntimeError("quadrature audit grids differ")
    mass = context.base.inner_params.M2_g
    primitives_32 = recover_global_primitives(
        grid_32,
        state_32,
        mass,
        specific_mechanical_energy_correction=correction_32,
    )
    primitives_64 = recover_global_primitives(
        grid_64,
        state_64,
        mass,
        specific_mechanical_energy_correction=correction_64,
    )

    def maximum_relative(left, right) -> float:
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        return float(
            np.max(
                np.abs(left - right)
                / np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
            )
        )

    return {
        "n_cells": n_cells,
        "maximum_mass_relative_difference": maximum_relative(
            state_32.mass, state_64.mass
        ),
        "maximum_radial_momentum_relative_difference": maximum_relative(
            state_32.radial_momentum, state_64.radial_momentum
        ),
        "maximum_angular_momentum_relative_difference": maximum_relative(
            state_32.angular_momentum, state_64.angular_momentum
        ),
        "maximum_total_energy_relative_difference": maximum_relative(
            state_32.total_energy, state_64.total_energy
        ),
        "maximum_correction_relative_difference": maximum_relative(
            correction_32, correction_64
        ),
        "maximum_temperature_relative_difference": maximum_relative(
            primitives_32.temperature, primitives_64.temperature
        ),
    }


def _run_case(
    context,
    evaluation,
    n_cells: int,
    n_steps: int,
    *,
    mapping_mode: str = "pointwise",
    jacobian_mode: str = "dense",
    open_face_reconstruction: str = "primitive_product",
    include_vertical_column_work: bool = False,
    boundary_mode: str = "open_no_inflow",
):
    (
        grid,
        initial,
        mass,
        potential,
        stream_rate,
        stream,
        initial_primitives,
        mechanical_correction,
        mapping,
    ) = _initial_case(
        context,
        evaluation,
        n_cells,
        mapping_mode=mapping_mode,
        open_face_reconstruction=open_face_reconstruction,
        include_vertical_column_work=include_vertical_column_work,
        boundary_mode=boundary_mode,
    )
    loading_time = float(np.sum(initial.mass) / stream_rate)
    dt = DT_LOADING_FRACTION * loading_time / n_steps
    current = initial
    results = []
    for _ in range(n_steps):
        result = advance_global_backward_euler(
            grid,
            current,
            mass,
            dt,
            alpha=context.base.alpha,
            reference_state=initial,
            boundary_mode=boundary_mode,
            stress_boundary_mode="outer_zero_torque",
            include_radiative_cooling=True,
            include_vertical_column_work=include_vertical_column_work,
            external_sources=stream,
            jacobian_mode=jacobian_mode,
            open_face_reconstruction=open_face_reconstruction,
            specific_mechanical_energy_correction=mechanical_correction,
            max_nfev=300,
        )
        results.append(result)
        if not result.accepted:
            break
        current = result.state
    final_result = results[-1]
    final = recover_global_primitives(
        grid,
        final_result.state,
        mass,
        specific_mechanical_energy_correction=mechanical_correction,
    )
    return {
        "initial": {
            **mapping,
            "inner_radius_rg": float(grid.edges[0] / potential.r_g),
            "loading_time_seconds": loading_time,
            "maximum_H_over_R": float(
                np.max(np.asarray(initial_primitives.vertical.H) / grid.centers)
            ),
            "radiative_luminosity_over_eddington": float(
                np.sum(
                    global_radiative_cooling_rate_cells(
                        grid, initial_primitives
                    )
                )
                / eddington_luminosity(mass)
            ),
        },
        "step": {
            "dt_over_loading_time": DT_LOADING_FRACTION,
            "dt_seconds": dt,
            "steps_attempted": n_steps,
            "steps_accepted": sum(result.accepted for result in results),
            "all_steps_accepted": len(results) == n_steps
            and all(result.accepted for result in results),
            "maximum_nfev": max(result.nfev for result in results),
            "messages": [result.message for result in results],
            "jacobian_audits": [
                None
                if result.jacobian_audit is None
                else {
                    "directions": result.jacobian_audit.directions,
                    "pattern_nonzeros": result.jacobian_audit.pattern_nonzeros,
                    "maximum_absolute_defect": (
                        result.jacobian_audit.maximum_absolute_defect
                    ),
                    "maximum_relative_defect": (
                        result.jacobian_audit.maximum_relative_defect
                    ),
                    "accepted": result.jacobian_audit.accepted,
                }
                for result in results
            ],
            "inner_characteristic_projections": [
                None
                if result.profile.inner_characteristic_projection is None
                else {
                    "incoming_amplitude_before": (
                        result.profile.inner_characteristic_projection
                        .incoming_amplitude_before
                    ),
                    "incoming_amplitude_after": (
                        result.profile.inner_characteristic_projection
                        .incoming_amplitude_after
                    ),
                    "outgoing_amplitude_before": (
                        result.profile.inner_characteristic_projection
                        .outgoing_amplitude_before
                    ),
                    "outgoing_amplitude_after": (
                        result.profile.inner_characteristic_projection
                        .outgoing_amplitude_after
                    ),
                    "projected_radial_velocity": (
                        result.profile.inner_characteristic_projection
                        .projected_radial_velocity
                    ),
                }
                for result in results
            ],
            "maximum_scaled_residual": max(
                result.maximum_scaled_residual for result in results
            ),
            "maximum_storage_scaled_ledger_defect": max(
                result.maximum_storage_scaled_ledger_defect
                for result in results
            ),
            "minimum_surface_density": float(
                np.min(final.surface_density)
            ),
            "minimum_temperature": float(np.min(final.temperature)),
            "maximum_H_over_R": float(
                np.max(np.asarray(final.vertical.H) / grid.centers)
            ),
            "inner_mass_flux_over_supply": float(
                final_result.profile.face_fluxes.mass[0] / stream_rate
            ),
            "outer_mass_flux_over_supply": float(
                final_result.profile.face_fluxes.mass[-1] / stream_rate
            ),
        },
    }


def main() -> None:
    context, evaluation = _canonical_open_evaluation()
    potential = PaczynskiWiitaPotential(context.base.inner_params.M2_g)
    circularization_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(circularization_radius))
    stream_energy = float(
        potential.phi(circularization_radius)
        + 0.5 * (stream_l / circularization_radius) ** 2
    )
    coarse = _run_case(context, evaluation, 16, 1)
    temporal = _run_case(context, evaluation, 16, 2)
    medium = _run_case(context, evaluation, 24, 2)
    refined = _run_case(context, evaluation, 32, 2)
    pointwise_mapping_only = [
        _mapping_attempt(
            context, evaluation, n_cells, mapping_mode="pointwise"
        )
        for n_cells in (16, 24, 32, 48)
    ]
    conservative_mapping_only = [
        _mapping_attempt(
            context, evaluation, n_cells, mapping_mode="conservative"
        )
        for n_cells in (16, 24, 32, 48, 64, 96, 128)
    ]
    mechanical_reference_quadrature = [
        _mechanical_reference_quadrature_audit(
            context, evaluation, n_cells
        )
        for n_cells in (64, 96, 128)
    ]
    donor_mapping_only = [
        _mapping_attempt(
            context,
            evaluation,
            n_cells,
            mapping_mode="conservative",
            open_face_reconstruction="conserved_donor",
        )
        for n_cells in (64, 96, 128)
    ]
    conservative_step = _run_case(
        context,
        evaluation,
        64,
        1,
        mapping_mode="conservative",
    )
    conservative_temporal = _run_case(
        context,
        evaluation,
        64,
        2,
        mapping_mode="conservative",
    )
    sparse_64 = _run_case(
        context,
        evaluation,
        64,
        1,
        mapping_mode="conservative",
        jacobian_mode="sparse_forward",
    )
    sparse_96 = _run_case(
        context,
        evaluation,
        96,
        1,
        mapping_mode="conservative",
        jacobian_mode="sparse_forward",
    )
    donor_64 = _run_case(
        context,
        evaluation,
        64,
        1,
        mapping_mode="conservative",
        jacobian_mode="sparse_forward",
        open_face_reconstruction="conserved_donor",
    )
    donor_96 = _run_case(
        context,
        evaluation,
        96,
        1,
        mapping_mode="conservative",
        jacobian_mode="sparse_forward",
        open_face_reconstruction="conserved_donor",
    )
    column_energy_64 = _run_case(
        context,
        evaluation,
        64,
        1,
        mapping_mode="conservative",
        jacobian_mode="sparse_forward",
        open_face_reconstruction="conserved_donor",
        include_vertical_column_work=True,
    )
    column_energy_96 = _run_case(
        context,
        evaluation,
        96,
        1,
        mapping_mode="conservative",
        jacobian_mode="sparse_forward",
        open_face_reconstruction="conserved_donor",
        include_vertical_column_work=True,
    )
    characteristic_inner_64 = _run_case(
        context,
        evaluation,
        64,
        1,
        mapping_mode="conservative",
        jacobian_mode="sparse_forward",
        open_face_reconstruction="conserved_donor",
        include_vertical_column_work=True,
        boundary_mode="characteristic_inner_open_outer",
    )
    characteristic_inner_96 = _run_case(
        context,
        evaluation,
        96,
        1,
        mapping_mode="conservative",
        jacobian_mode="sparse_forward",
        open_face_reconstruction="conserved_donor",
        include_vertical_column_work=True,
        boundary_mode="characteristic_inner_open_outer",
    )
    target_inner = float(
        -evaluation.base.outer_transport.mdot_faces[0]
        / (5.0 * eddington_mdot(context.base.inner_params.M2_g))
    )
    target_outer = float(
        -evaluation.base.outer_transport.mdot_faces[-1]
        / (5.0 * eddington_mdot(context.base.inner_params.M2_g))
    )
    report = {
        "input_closure": {
            "M2_g": context.base.inner_params.M2_g,
            "stream_rate_over_eddington": 5.0,
            "source_center_rg": 240.0,
            "source_log_width": 0.08,
            "circularization_radius_rg": 248.96693,
            "specific_radial_velocity": 0.0,
            "specific_angular_momentum": stream_l,
            "specific_total_energy": stream_energy,
            "alpha": context.base.alpha,
            "outer_radius_rg": 335.0,
            "target_inner_outward_mass_flux_over_supply": target_inner,
            "target_outer_outward_mass_flux_over_supply": target_outer,
        },
        "runs": [coarse, temporal, medium, refined],
        "pointwise_mapping_only_meshes": pointwise_mapping_only,
        "conservative_mapping_only_meshes": conservative_mapping_only,
        "mechanical_reference_quadrature_audit": (
            mechanical_reference_quadrature
        ),
        "conserved_donor_mapping_only_meshes": donor_mapping_only,
        "conservative_N64_step": conservative_step,
        "conservative_N64_two_half_steps": conservative_temporal,
        "conservative_N64_temporal_comparison": {
            "inner_flux_difference": float(
                conservative_temporal["step"][
                    "inner_mass_flux_over_supply"
                ]
                - conservative_step["step"][
                    "inner_mass_flux_over_supply"
                ]
            ),
            "outer_flux_difference": float(
                conservative_temporal["step"][
                    "outer_mass_flux_over_supply"
                ]
                - conservative_step["step"][
                    "outer_mass_flux_over_supply"
                ]
            ),
            "maximum_H_over_R_relative_difference": float(
                conservative_temporal["step"]["maximum_H_over_R"]
                / conservative_step["step"]["maximum_H_over_R"]
                - 1.0
            ),
        },
        "sparse_evolved_mesh_runs": [sparse_64, sparse_96],
        "sparse_evolved_N64_N96_comparison": {
            "inner_flux_difference_over_supply": float(
                sparse_96["step"]["inner_mass_flux_over_supply"]
                - sparse_64["step"]["inner_mass_flux_over_supply"]
            ),
            "outer_flux_difference_over_supply": float(
                sparse_96["step"]["outer_mass_flux_over_supply"]
                - sparse_64["step"]["outer_mass_flux_over_supply"]
            ),
            "outer_flux_relative_difference": float(
                sparse_96["step"]["outer_mass_flux_over_supply"]
                / sparse_64["step"]["outer_mass_flux_over_supply"]
                - 1.0
            ),
            "maximum_H_over_R_relative_difference": float(
                sparse_96["step"]["maximum_H_over_R"]
                / sparse_64["step"]["maximum_H_over_R"]
                - 1.0
            ),
            "flux_mesh_gate_pass": bool(
                abs(
                    sparse_96["step"]["outer_mass_flux_over_supply"]
                    - sparse_64["step"]["outer_mass_flux_over_supply"]
                )
                <= 0.01
            ),
        },
        "conserved_donor_evolved_mesh_runs": [donor_64, donor_96],
        "column_energy_evolved_mesh_runs": [
            column_energy_64,
            column_energy_96,
        ],
        "column_energy_N64_N96_comparison": {
            "inner_flux_difference_over_supply": float(
                column_energy_96["step"]["inner_mass_flux_over_supply"]
                - column_energy_64["step"]["inner_mass_flux_over_supply"]
            ),
            "outer_flux_difference_over_supply": float(
                column_energy_96["step"]["outer_mass_flux_over_supply"]
                - column_energy_64["step"]["outer_mass_flux_over_supply"]
            ),
            "maximum_H_over_R_relative_difference": float(
                column_energy_96["step"]["maximum_H_over_R"]
                / column_energy_64["step"]["maximum_H_over_R"]
                - 1.0
            ),
            "flux_mesh_gate_pass": bool(
                abs(
                    column_energy_96["step"]["outer_mass_flux_over_supply"]
                    - column_energy_64["step"]["outer_mass_flux_over_supply"]
                )
                <= 0.01
            ),
        },
        "characteristic_inner_evolved_mesh_runs": [
            characteristic_inner_64,
            characteristic_inner_96,
        ],
        "characteristic_inner_N64_N96_comparison": {
            "inner_flux_difference_over_supply": float(
                characteristic_inner_96["step"][
                    "inner_mass_flux_over_supply"
                ]
                - characteristic_inner_64["step"][
                    "inner_mass_flux_over_supply"
                ]
            ),
            "outer_flux_difference_over_supply": float(
                characteristic_inner_96["step"][
                    "outer_mass_flux_over_supply"
                ]
                - characteristic_inner_64["step"][
                    "outer_mass_flux_over_supply"
                ]
            ),
            "maximum_H_over_R_relative_difference": float(
                characteristic_inner_96["step"]["maximum_H_over_R"]
                / characteristic_inner_64["step"]["maximum_H_over_R"]
                - 1.0
            ),
        },
        "conserved_donor_N64_N96_comparison": {
            "inner_flux_difference_over_supply": float(
                donor_96["step"]["inner_mass_flux_over_supply"]
                - donor_64["step"]["inner_mass_flux_over_supply"]
            ),
            "outer_flux_difference_over_supply": float(
                donor_96["step"]["outer_mass_flux_over_supply"]
                - donor_64["step"]["outer_mass_flux_over_supply"]
            ),
            "outer_flux_relative_difference": float(
                donor_96["step"]["outer_mass_flux_over_supply"]
                / donor_64["step"]["outer_mass_flux_over_supply"]
                - 1.0
            ),
            "maximum_H_over_R_relative_difference": float(
                donor_96["step"]["maximum_H_over_R"]
                / donor_64["step"]["maximum_H_over_R"]
                - 1.0
            ),
            "flux_mesh_gate_pass": bool(
                abs(
                    donor_96["step"]["outer_mass_flux_over_supply"]
                    - donor_64["step"]["outer_mass_flux_over_supply"]
                )
                <= 0.01
            ),
        },
        "mapping_only_conclusion": (
            "Pointwise coarse mapping creates the boundary-flux sensitivity. "
            "Strict conservative remapping rejects N<=48 on thermal primitive "
            "recovery, while N64 and N96 are admissible and approach the "
            "canonical coupled boundary fluxes."
        ),
        "temporal_comparison_N16": {
            "inner_flux_difference": float(
                temporal["step"]["inner_mass_flux_over_supply"]
                - coarse["step"]["inner_mass_flux_over_supply"]
            ),
            "outer_flux_difference": float(
                temporal["step"]["outer_mass_flux_over_supply"]
                - coarse["step"]["outer_mass_flux_over_supply"]
            ),
            "maximum_H_over_R_relative_difference": float(
                temporal["step"]["maximum_H_over_R"]
                / coarse["step"]["maximum_H_over_R"]
                - 1.0
            ),
        },
        "mesh_comparison_N16_N24_two_steps": {
            "inner_flux_difference": float(
                medium["step"]["inner_mass_flux_over_supply"]
                - temporal["step"]["inner_mass_flux_over_supply"]
            ),
            "outer_flux_difference": float(
                medium["step"]["outer_mass_flux_over_supply"]
                - temporal["step"]["outer_mass_flux_over_supply"]
            ),
            "maximum_H_over_R_relative_difference": float(
                medium["step"]["maximum_H_over_R"]
                / temporal["step"]["maximum_H_over_R"]
                - 1.0
            ),
        },
        "mesh_comparison_N24_N32_two_steps": {
            "inner_flux_difference": float(
                refined["step"]["inner_mass_flux_over_supply"]
                - medium["step"]["inner_mass_flux_over_supply"]
            ),
            "outer_flux_difference": float(
                refined["step"]["outer_mass_flux_over_supply"]
                - medium["step"]["outer_mass_flux_over_supply"]
            ),
            "maximum_H_over_R_relative_difference": float(
                refined["step"]["maximum_H_over_R"]
                / medium["step"]["maximum_H_over_R"]
                - 1.0
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
