"""Audit the first one-domain signed conservative descriptor layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C, M_SUN
from imri_qpe.layer3_minidisk_1d import (
    GlobalCellSources,
    GlobalConservativeState,
    GlobalFaceFluxes,
    GlobalFluxPrimaryLayout,
    advance_global_alpha_stress_backward_euler,
    advance_global_backward_euler,
    advance_global_imex,
    advance_global_inviscid_rusanov,
    advance_global_radiative_cooling_backward_euler,
    audit_global_backward_euler_ledgers,
    global_backward_euler_residual,
    global_backward_euler_jacobian_sparsity,
    global_conservative_rhs,
    global_descriptor_mass_matrix,
    global_effective_sound_speed,
    evaluate_global_inviscid_profile,
    evaluate_global_rusanov_profile,
    global_inviscid_cfl_timestep,
    global_radiative_cooling_rate_cells,
    make_log_grid,
    manufactured_backward_euler_jacobian,
    PaczynskiWiitaPotential,
    positive_edge_reconstruction,
    recover_global_primitives,
    recover_thermodynamics_from_pi_beta,
    SignedThermalClosure,
    state_from_primitives,
    state_from_thermodynamic_primitives,
    vertical_state,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/global_signed_descriptor_rank.json"


def _rusanov_audit() -> dict[str, object]:
    n_cells = 32
    central_mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(central_mass)
    grid = make_log_grid(
        10.0 * potential.r_g, 40.0 * potential.r_g, n_cells
    )
    left = np.arange(n_cells) < n_cells // 2
    state = state_from_thermodynamic_primitives(
        grid,
        np.where(left, 8.0e6, 2.0e6),
        np.where(left, 2.0e-4 * C, -1.0e-4 * C),
        0.85 * potential.omega_k(grid.centers),
        np.where(left, 2.0e7, 8.0e6),
        central_mass,
    )
    primitives = recover_global_primitives(grid, state, central_mass)
    base_dt = global_inviscid_cfl_timestep(grid, primitives, cfl=0.005)
    accepted = advance_global_inviscid_rusanov(
        grid,
        state,
        central_mass,
        base_dt,
        boundary_mode="open_no_inflow",
    )
    rejected = advance_global_inviscid_rusanov(
        grid,
        state,
        central_mass,
        10.0 * base_dt,
        boundary_mode="open_no_inflow",
    )
    smooth = evaluate_global_inviscid_profile(grid, state, central_mass)
    corrected = evaluate_global_rusanov_profile(
        grid, state, central_mass, reference_state=state
    )
    open_profile = evaluate_global_rusanov_profile(
        grid, state, central_mass, boundary_mode="open_no_inflow"
    )
    differences = []
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        reference = getattr(smooth.face_fluxes, name)
        difference = getattr(corrected.face_fluxes, name) - reference
        differences.append(
            float(
                np.max(np.abs(difference))
                / max(float(np.max(np.abs(reference))), 1.0)
            )
        )
    return {
        "accepted_cfl": 0.005,
        "accepted_step": accepted.accepted,
        "accepted_maximum_relative_ledger_defect": (
            accepted.ledger.maximum_relative_defect
        ),
        "ten_times_larger_step_rejected": not rejected.accepted,
        "rejected_state_is_unchanged": all(
            np.array_equal(getattr(rejected.state, name), getattr(state, name))
            for name in (
                "mass",
                "radial_momentum",
                "angular_momentum",
                "total_energy",
            )
        ),
        "maximum_equilibrium_corrected_flux_difference": max(differences),
        "inner_unconfigured_inflow_blocked": bool(
            open_profile.face_fluxes.mass[0] == 0.0
        ),
        "outer_unconfigured_inflow_blocked": bool(
            open_profile.face_fluxes.mass[-1] == 0.0
        ),
    }


def _radial_equilibrium_state(n_cells: int, pressure_slope: float):
    central_mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(central_mass)
    grid = make_log_grid(
        12.0 * potential.r_g, 120.0 * potential.r_g, n_cells
    )
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
    target_pi = reference_pi * (
        grid.centers / reference_radius
    ) ** pressure_slope
    sigma, temperature, inversion_residual = recover_thermodynamics_from_pi_beta(
        grid.centers,
        target_pi,
        np.full(n_cells, reference_beta),
        central_mass,
        closure=closure,
        surface_density_seed=seed_sigma,
        temperature_seed=seed_temperature,
    )
    if inversion_residual >= 1.0e-9:
        raise RuntimeError("manufactured thermodynamic inversion did not close")
    omega_k = potential.omega_k(grid.centers)
    omega = np.sqrt(
        omega_k**2
        + pressure_slope * target_pi / (sigma * grid.centers**2)
    )
    state = state_from_thermodynamic_primitives(
        grid,
        sigma,
        np.zeros(n_cells),
        omega,
        temperature,
        central_mass,
    )
    return grid, state, central_mass


def _radial_equilibrium_error(n_cells: int, pressure_slope: float) -> float:
    grid, state, central_mass = _radial_equilibrium_state(
        n_cells, pressure_slope
    )
    profile = evaluate_global_inviscid_profile(grid, state, central_mass)
    rhs = global_conservative_rhs(profile.face_fluxes, profile.cell_sources)
    scale = np.maximum(
        np.abs(profile.face_fluxes.radial_momentum[:-1])
        + np.abs(profile.face_fluxes.radial_momentum[1:])
        + np.abs(profile.cell_sources.radial_momentum),
        1.0,
    )
    return float(np.max(np.abs(rhs.radial_momentum) / scale))


def _flatten_state(state: GlobalConservativeState) -> np.ndarray:
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


def _source_free_temporal_audit() -> dict[str, object]:
    grid, initial, central_mass = _radial_equilibrium_state(32, -0.25)
    primitives = recover_global_primitives(grid, initial, central_mass)
    total_time = global_inviscid_cfl_timestep(grid, primitives, cfl=0.02)
    storage_scales = {
        "mass": float(np.sum(np.abs(initial.mass))),
        "radial_momentum": float(np.sum(initial.mass)) * C,
        "angular_momentum": float(np.sum(np.abs(initial.angular_momentum))),
        "total_energy": float(np.sum(np.abs(initial.total_energy))),
    }
    states = []
    maximum_ledger_defect = 0.0
    maximum_storage_scaled_ledger_defect = 0.0
    all_steps_accepted = True
    for n_steps in (1, 2, 4):
        current = initial
        for _ in range(n_steps):
            result = advance_global_inviscid_rusanov(
                grid,
                current,
                central_mass,
                total_time / n_steps,
                reference_state=initial,
                boundary_mode="open_no_inflow",
            )
            all_steps_accepted = all_steps_accepted and result.accepted
            maximum_ledger_defect = max(
                maximum_ledger_defect,
                result.ledger.maximum_relative_defect,
            )
            maximum_storage_scaled_ledger_defect = max(
                maximum_storage_scaled_ledger_defect,
                max(
                    abs(result.ledger.defects[name])
                    / max(storage_scales[name], 1.0)
                    for name in (
                        "mass",
                        "radial_momentum",
                        "angular_momentum",
                        "total_energy",
                    )
                ),
            )
            if not result.accepted:
                break
            current = result.state
        states.append(current)
    mass_scale = float(np.max(initial.mass))
    radial_scale = mass_scale * float(
        np.max(
            np.abs(primitives.radial_velocity)
            + global_effective_sound_speed(primitives)
        )
    )
    angular_scale = float(np.max(np.abs(initial.angular_momentum)))
    energy_scale = float(np.max(np.abs(initial.total_energy)))
    scale = np.concatenate(
        (
            np.full(initial.n_cells, mass_scale),
            np.full(initial.n_cells, radial_scale),
            np.full(initial.n_cells, angular_scale),
            np.full(initial.n_cells, energy_scale),
        )
    )
    vectors = [_flatten_state(state) for state in states]
    coarse_error = float(np.max(np.abs(vectors[0] - vectors[1]) / scale))
    fine_error = float(np.max(np.abs(vectors[1] - vectors[2]) / scale))
    return {
        "mesh_cells": 32,
        "total_time_seconds": total_time,
        "full_step_cfl": 0.02,
        "all_steps_accepted": all_steps_accepted,
        "one_vs_two_step_error": coarse_error,
        "two_vs_four_step_error": fine_error,
        "error_ratio": float(coarse_error / fine_error),
        "maximum_relative_ledger_defect": maximum_ledger_defect,
        "maximum_storage_scaled_ledger_defect": (
            maximum_storage_scaled_ledger_defect
        ),
    }


def _source_free_mesh_audit() -> dict[str, object]:
    fine_grid, fine_initial, central_mass = _radial_equilibrium_state(64, -0.25)
    fine_primitives = recover_global_primitives(
        fine_grid, fine_initial, central_mass
    )
    total_time = global_inviscid_cfl_timestep(
        fine_grid, fine_primitives, cfl=0.005
    )
    meshes = []
    for n_cells in (16, 32, 64):
        grid, initial, local_mass = _radial_equilibrium_state(n_cells, -0.25)
        current = initial
        all_steps_accepted = True
        for _ in range(4):
            result = advance_global_inviscid_rusanov(
                grid,
                current,
                local_mass,
                total_time / 4.0,
                reference_state=initial,
                boundary_mode="open_no_inflow",
            )
            all_steps_accepted = all_steps_accepted and result.accepted
            if not result.accepted:
                break
            current = result.state
        primitives = recover_global_primitives(grid, current, local_mass)
        meshes.append(
            {
                "n_cells": n_cells,
                "all_steps_accepted": all_steps_accepted,
                "maximum_radial_velocity_c": float(
                    np.max(np.abs(primitives.radial_velocity)) / C
                ),
                "interior_radial_velocity_c": float(
                    np.max(np.abs(primitives.radial_velocity[2:-2])) / C
                ),
                "maximum_drift_cell": int(
                    np.argmax(np.abs(primitives.radial_velocity))
                ),
            }
        )
    return {
        "total_time_seconds": total_time,
        "steps_per_mesh": 4,
        "meshes": meshes,
        "interior_drift_ratios": [
            float(
                meshes[index]["interior_radial_velocity_c"]
                / meshes[index + 1]["interior_radial_velocity_c"]
            )
            for index in range(len(meshes) - 1)
        ],
        "all_interior_mesh_gates_pass": bool(
            all(mesh["all_steps_accepted"] for mesh in meshes)
            and all(
                meshes[index + 1]["interior_radial_velocity_c"]
                < meshes[index]["interior_radial_velocity_c"] / 2.5
                for index in range(len(meshes) - 1)
            )
        ),
        "open_edge_response_is_bounded": bool(
            max(mesh["maximum_radial_velocity_c"] for mesh in meshes)
            < 4.0e-10
            and meshes[-1]["maximum_radial_velocity_c"]
            < meshes[0]["maximum_radial_velocity_c"]
        ),
    }


def _common_stress_flux_audit() -> dict[str, object]:
    grid, state, central_mass = _radial_equilibrium_state(32, -0.25)
    base = evaluate_global_rusanov_profile(
        grid,
        state,
        central_mass,
        reference_state=state,
    )
    stressed = evaluate_global_rusanov_profile(
        grid,
        state,
        central_mass,
        reference_state=state,
        alpha=0.1,
    )
    zero_boundary = evaluate_global_rusanov_profile(
        grid,
        state,
        central_mass,
        reference_state=state,
        alpha=0.1,
        stress_boundary_mode="zero_torque",
    )
    torque = np.asarray(stressed.viscous_torque_faces, dtype=float)
    omega_faces = positive_edge_reconstruction(
        grid, stressed.primitives.omega
    )
    angular_difference = (
        stressed.face_fluxes.angular_momentum
        - base.face_fluxes.angular_momentum
    )
    energy_difference = (
        stressed.face_fluxes.total_energy - base.face_fluxes.total_energy
    )
    angular_scale = max(float(np.max(np.abs(torque))), 1.0)
    energy_scale = max(float(np.max(np.abs(omega_faces * torque))), 1.0)
    return {
        "alpha": 0.1,
        "outward_orientation": True,
        "minimum_viscous_torque": float(np.min(torque)),
        "maximum_mass_flux_change": float(
            np.max(np.abs(stressed.face_fluxes.mass - base.face_fluxes.mass))
        ),
        "maximum_radial_momentum_flux_change": float(
            np.max(
                np.abs(
                    stressed.face_fluxes.radial_momentum
                    - base.face_fluxes.radial_momentum
                )
            )
        ),
        "maximum_scaled_angular_pair_mismatch": float(
            np.max(np.abs(angular_difference - torque)) / angular_scale
        ),
        "maximum_scaled_energy_pair_mismatch": float(
            np.max(np.abs(energy_difference - omega_faces * torque))
            / energy_scale
        ),
        "zero_torque_boundary_is_exact": bool(
            zero_boundary.viscous_torque_faces[0] == 0.0
            and zero_boundary.viscous_torque_faces[-1] == 0.0
        ),
    }


def _implicit_stress_audit() -> dict[str, object]:
    grid, initial, central_mass = _radial_equilibrium_state(8, -0.25)
    primitives = recover_global_primitives(grid, initial, central_mass)
    total_time = global_inviscid_cfl_timestep(grid, primitives, cfl=0.01)
    states = []
    all_steps_accepted = True
    maximum_ledger_defect = 0.0
    maximum_storage_ledger_defect = 0.0
    maximum_residual = 0.0
    maximum_nfev = 0
    for n_steps in (1, 2, 4):
        current = initial
        for _ in range(n_steps):
            result = advance_global_alpha_stress_backward_euler(
                grid,
                current,
                central_mass,
                total_time / n_steps,
                alpha=0.1,
                stress_boundary_mode="zero_torque",
            )
            all_steps_accepted = all_steps_accepted and result.accepted
            maximum_ledger_defect = max(
                maximum_ledger_defect,
                result.ledger.maximum_relative_defect,
            )
            maximum_storage_ledger_defect = max(
                maximum_storage_ledger_defect,
                result.maximum_storage_scaled_ledger_defect,
            )
            maximum_residual = max(
                maximum_residual, result.maximum_scaled_residual
            )
            maximum_nfev = max(maximum_nfev, result.nfev)
            if not result.accepted:
                break
            current = result.state
        states.append(current)
    vectors = [_flatten_state(state) for state in states]
    scale = np.concatenate(
        tuple(
            np.full(
                initial.n_cells,
                max(float(np.max(np.abs(getattr(initial, name)))), 1.0),
            )
            for name in (
                "mass",
                "radial_momentum",
                "angular_momentum",
                "total_energy",
            )
        )
    )
    coarse_error = float(np.max(np.abs(vectors[0] - vectors[1]) / scale))
    fine_error = float(np.max(np.abs(vectors[1] - vectors[2]) / scale))
    rejected = advance_global_alpha_stress_backward_euler(
        grid,
        initial,
        central_mass,
        total_time,
        alpha=0.1,
        stress_boundary_mode="zero_torque",
        residual_tolerance=1.0e-30,
        ledger_tolerance=1.0e-30,
        max_nfev=1,
    )
    return {
        "mesh_cells": 8,
        "total_time_seconds": total_time,
        "alpha": 0.1,
        "all_steps_accepted": all_steps_accepted,
        "one_vs_two_step_error": coarse_error,
        "two_vs_four_step_error": fine_error,
        "error_ratio": float(coarse_error / fine_error),
        "maximum_scaled_residual": maximum_residual,
        "maximum_relative_ledger_defect": maximum_ledger_defect,
        "maximum_storage_scaled_ledger_defect": (
            maximum_storage_ledger_defect
        ),
        "maximum_nfev": maximum_nfev,
        "forced_rejection_returns_original_state": bool(
            not rejected.accepted
            and all(
                np.array_equal(
                    getattr(rejected.state, name), getattr(initial, name)
                )
                for name in (
                    "mass",
                    "radial_momentum",
                    "angular_momentum",
                    "total_energy",
                )
            )
        ),
    }


def _monolithic_backward_euler_audit() -> dict[str, object]:
    coarse_grid, coarse_initial, central_mass = _radial_equilibrium_state(
        8, -0.25
    )
    coarse_primitives = recover_global_primitives(
        coarse_grid, coarse_initial, central_mass
    )
    total_time = 4.0 * global_inviscid_cfl_timestep(
        coarse_grid, coarse_primitives, cfl=0.002
    )

    def evolve(n_cells: int, n_steps: int):
        grid, initial, local_mass = _radial_equilibrium_state(
            n_cells, -0.25
        )
        current = initial
        first_state = None
        maximum_residual = 0.0
        maximum_storage_ledger_defect = 0.0
        maximum_nfev = 0
        accepted_steps = 0
        for _ in range(n_steps):
            result = advance_global_backward_euler(
                grid,
                current,
                local_mass,
                total_time / n_steps,
                alpha=0.1,
                reference_state=initial,
                boundary_mode="open_no_inflow",
                stress_boundary_mode="zero_torque",
            )
            maximum_residual = max(
                maximum_residual, result.maximum_scaled_residual
            )
            maximum_storage_ledger_defect = max(
                maximum_storage_ledger_defect,
                result.maximum_storage_scaled_ledger_defect,
            )
            maximum_nfev = max(maximum_nfev, result.nfev)
            if not result.accepted:
                break
            current = result.state
            accepted_steps += 1
            if first_state is None:
                first_state = current
        final = recover_global_primitives(grid, current, local_mass)
        summary = {
            "n_cells": n_cells,
            "steps_attempted": n_steps,
            "steps_accepted": accepted_steps,
            "step_seconds": total_time / n_steps,
            "maximum_scaled_residual": maximum_residual,
            "maximum_storage_scaled_ledger_defect": (
                maximum_storage_ledger_defect
            ),
            "maximum_nfev": maximum_nfev,
            "minimum_surface_density": float(
                np.min(final.surface_density)
            ),
            "maximum_surface_density": float(
                np.max(final.surface_density)
            ),
            "minimum_omega": float(np.min(final.omega)),
            "minimum_temperature": float(np.min(final.temperature)),
            "maximum_radial_velocity_c": float(
                np.max(np.abs(final.radial_velocity)) / C
            ),
            "maximum_H_over_R": float(
                np.max(np.asarray(final.vertical.H) / grid.centers)
            ),
            "remaining_mass_fraction": float(
                np.sum(current.mass) / np.sum(initial.mass)
            ),
        }
        return grid, initial, current, first_state, final, summary

    coarse = evolve(8, 4)
    fine_time = evolve(8, 8)
    fine_mesh = evolve(16, 8)
    coarse_final = coarse[4]
    fine_time_final = fine_time[4]
    temporal_differences = {
        "maximum_surface_density_fraction": float(
            np.max(
                np.abs(
                    coarse_final.surface_density
                    / fine_time_final.surface_density
                    - 1.0
                )
            )
        ),
        "maximum_temperature_fraction": float(
            np.max(
                np.abs(
                    coarse_final.temperature / fine_time_final.temperature
                    - 1.0
                )
            )
        ),
        "maximum_omega_fraction": float(
            np.max(
                np.abs(coarse_final.omega / fine_time_final.omega - 1.0)
            )
        ),
        "maximum_radial_velocity_difference_c": float(
            np.max(
                np.abs(
                    coarse_final.radial_velocity
                    - fine_time_final.radial_velocity
                )
            )
            / C
        ),
    }

    colored = advance_global_backward_euler(
        coarse[0],
        coarse[1],
        central_mass,
        total_time / 4.0,
        alpha=0.1,
        reference_state=coarse[1],
        boundary_mode="open_no_inflow",
        stress_boundary_mode="zero_torque",
        use_sparse_jacobian=True,
    )
    colored_primitives = recover_global_primitives(
        coarse[0], colored.state, central_mass
    )
    dense_primitives = recover_global_primitives(
        coarse[0], coarse[3], central_mass
    )
    sparsity = global_backward_euler_jacobian_sparsity(8)
    colored_audit = {
        "accepted": colored.accepted,
        "pattern_shape": list(sparsity.shape),
        "pattern_nonzeros": int(sparsity.nnz),
        "dense_nfev": coarse[5]["maximum_nfev"],
        "colored_nfev": colored.nfev,
        "maximum_surface_density_fraction": float(
            np.max(
                np.abs(
                    colored_primitives.surface_density
                    / dense_primitives.surface_density
                    - 1.0
                )
            )
        ),
        "maximum_temperature_fraction": float(
            np.max(
                np.abs(
                    colored_primitives.temperature
                    / dense_primitives.temperature
                    - 1.0
                )
            )
        ),
        "maximum_omega_fraction": float(
            np.max(
                np.abs(
                    colored_primitives.omega / dense_primitives.omega - 1.0
                )
            )
        ),
        "maximum_radial_velocity_difference_c": float(
            np.max(
                np.abs(
                    colored_primitives.radial_velocity
                    - dense_primitives.radial_velocity
                )
            )
            / C
        ),
    }

    split_current = coarse[1]
    split_accepted_steps = 0
    split_rejected_state_unchanged = False
    for _ in range(8):
        old = split_current
        split = advance_global_imex(
            coarse[0],
            split_current,
            central_mass,
            total_time / 4.0,
            alpha=0.1,
            reference_state=coarse[1],
            boundary_mode="open_no_inflow",
            stress_boundary_mode="zero_torque",
        )
        if not split.accepted:
            split_rejected_state_unchanged = all(
                np.array_equal(getattr(split.state, name), getattr(old, name))
                for name in (
                    "mass",
                    "radial_momentum",
                    "angular_momentum",
                    "total_energy",
                )
            )
            break
        split_current = split.state
        split_accepted_steps += 1
    return {
        "alpha": 0.1,
        "total_time_seconds": total_time,
        "runs": [coarse[5], fine_time[5], fine_mesh[5]],
        "temporal_differences_4_vs_8": temporal_differences,
        "colored_jacobian_audit": colored_audit,
        "split_imex_steps_before_rejection": split_accepted_steps,
        "split_imex_rejected_state_unchanged": split_rejected_state_unchanged,
    }


def _radiative_cooling_audit() -> dict[str, object]:
    grid, initial, central_mass = _radial_equilibrium_state(8, 0.0)
    primitives = recover_global_primitives(grid, initial, central_mass)
    cooling_rate = global_radiative_cooling_rate_cells(grid, primitives)
    internal_energy = initial.mass * primitives.specific_internal_energy
    total_time = 0.02 * float(np.min(internal_energy / cooling_rate))
    states = []
    maximum_storage_ledger_defect = 0.0
    all_local_steps_accepted = True
    for n_steps in (1, 2, 4):
        current = initial
        for _ in range(n_steps):
            result = advance_global_radiative_cooling_backward_euler(
                grid,
                current,
                central_mass,
                total_time / n_steps,
            )
            all_local_steps_accepted = (
                all_local_steps_accepted and result.accepted
            )
            maximum_storage_ledger_defect = max(
                maximum_storage_ledger_defect,
                result.maximum_storage_scaled_ledger_defect,
            )
            if not result.accepted:
                break
            current = result.state
        states.append(current)
    temperatures = [
        recover_global_primitives(grid, state, central_mass).temperature
        for state in states
    ]
    coarse_error = float(
        np.max(np.abs(temperatures[0] / temperatures[1] - 1.0))
    )
    fine_error = float(
        np.max(np.abs(temperatures[1] / temperatures[2] - 1.0))
    )

    common = {
        "alpha": 0.0,
        "reference_state": initial,
        "boundary_mode": "open_no_inflow",
    }
    adiabatic = advance_global_backward_euler(
        grid, initial, central_mass, 10.0, **common
    )
    cooled = advance_global_backward_euler(
        grid,
        initial,
        central_mass,
        10.0,
        include_radiative_cooling=True,
        **common,
    )
    adiabatic_temperature = recover_global_primitives(
        grid, adiabatic.state, central_mass
    ).temperature
    cooled_temperature = recover_global_primitives(
        grid, cooled.state, central_mass
    ).temperature
    temperature_drop = adiabatic_temperature - cooled_temperature
    return {
        "all_local_steps_accepted": all_local_steps_accepted,
        "local_total_time_seconds": total_time,
        "local_one_vs_two_error": coarse_error,
        "local_two_vs_four_error": fine_error,
        "local_error_ratio": float(coarse_error / fine_error),
        "maximum_local_storage_scaled_ledger_defect": (
            maximum_storage_ledger_defect
        ),
        "monolithic_adiabatic_accepted": adiabatic.accepted,
        "monolithic_cooled_accepted": cooled.accepted,
        "monolithic_step_seconds": 10.0,
        "monolithic_cooled_maximum_residual": (
            cooled.maximum_scaled_residual
        ),
        "monolithic_cooled_storage_scaled_ledger_defect": (
            cooled.maximum_storage_scaled_ledger_defect
        ),
        "minimum_monolithic_temperature_drop": float(
            np.min(temperature_drop)
        ),
        "maximum_monolithic_temperature_drop": float(
            np.max(temperature_drop)
        ),
    }


def _case(n_cells: int) -> dict[str, object]:
    grid = make_log_grid(1.0, 3.0, n_cells)
    radius = grid.centers
    old_state = state_from_primitives(
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
    ledger = audit_global_backward_euler_ledgers(
        new_state, old_state, dt, fluxes, sources
    )
    layout = GlobalFluxPrimaryLayout(n_cells)
    descriptor = global_descriptor_mass_matrix(layout).toarray()
    backward_euler = manufactured_backward_euler_jacobian(layout, dt).toarray()
    singular_values = np.linalg.svd(backward_euler, compute_uv=False)
    central_mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(central_mass)
    physical_grid = make_log_grid(
        6.2 * potential.r_g, 300.0 * potential.r_g, n_cells
    )
    physical_sigma = np.geomspace(1.0e4, 3.0e6, n_cells)
    physical_velocity = C * np.linspace(-2.0e-3, 2.0e-3, n_cells)
    physical_omega = 0.82 * potential.omega_k(physical_grid.centers)
    physical_temperature = np.geomspace(2.0e6, 2.0e8, n_cells)
    physical_state = state_from_thermodynamic_primitives(
        physical_grid,
        physical_sigma,
        physical_velocity,
        physical_omega,
        physical_temperature,
        central_mass,
    )
    recovered = recover_global_primitives(
        physical_grid, physical_state, central_mass
    )
    return {
        "n_cells": n_cells,
        "differential_variables": layout.differential_size,
        "algebraic_face_flux_variables": layout.algebraic_size,
        "unknowns": layout.state_size,
        "residuals": layout.residual_size,
        "descriptor_rank": int(np.linalg.matrix_rank(descriptor)),
        "backward_euler_rank": int(np.linalg.matrix_rank(backward_euler)),
        "smallest_singular_value": float(singular_values[-1]),
        "largest_singular_value": float(singular_values[0]),
        "condition_estimate": float(singular_values[0] / singular_values[-1]),
        "maximum_scaled_manufactured_residual": float(
            np.max(np.abs(residual) / storage_scale)
        ),
        "maximum_relative_ledger_defect": ledger.maximum_relative_defect,
        "mass_flux_has_both_signs": bool(
            np.any(fluxes.mass < 0.0) and np.any(fluxes.mass > 0.0)
        ),
        "mass_flux_has_zero_crossing": bool(np.any(np.isclose(fluxes.mass, 0.0))),
        "maximum_temperature_round_trip_error": float(
            np.max(np.abs(recovered.temperature / physical_temperature - 1.0))
        ),
        "maximum_surface_density_round_trip_error": float(
            np.max(np.abs(recovered.surface_density / physical_sigma - 1.0))
        ),
        "maximum_radial_velocity_round_trip_error_c": float(
            np.max(np.abs(recovered.radial_velocity - physical_velocity)) / C
        ),
        "constant_pi_keplerian_balance_error": _radial_equilibrium_error(
            n_cells, 0.0
        ),
        "pressure_supported_balance_error": _radial_equilibrium_error(
            n_cells, -0.25
        ),
    }


def main() -> None:
    meshes = [_case(n_cells) for n_cells in (8, 16, 32)]
    report = {
        "architecture": "one-domain four-conservation-law flux-primary DAE",
        "state_count": "8*N + 4",
        "differential_count": "4*N",
        "algebraic_count": "4*(N+1)",
        "face_orientation": "outward positive",
        "sonic_treatment": "emerges dynamically from radial momentum",
        "physics_status": "manufactured conservation and rank prototype only",
        "meshes": meshes,
        "all_rank_gates_pass": all(
            mesh["descriptor_rank"] == mesh["differential_variables"]
            and mesh["backward_euler_rank"] == mesh["unknowns"]
            for mesh in meshes
        ),
        "all_ledger_gates_pass": all(
            mesh["maximum_relative_ledger_defect"] < 1.0e-14 for mesh in meshes
        ),
        "all_primitive_recovery_gates_pass": all(
            mesh["maximum_temperature_round_trip_error"] < 1.0e-9
            and mesh["maximum_surface_density_round_trip_error"] < 1.0e-12
            and mesh["maximum_radial_velocity_round_trip_error_c"] < 1.0e-12
            for mesh in meshes
        ),
        "radial_balance_error_ratios": [
            float(
                meshes[index]["pressure_supported_balance_error"]
                / meshes[index + 1]["pressure_supported_balance_error"]
            )
            for index in range(len(meshes) - 1)
        ],
        "all_radial_balance_gates_pass": bool(
            all(mesh["constant_pi_keplerian_balance_error"] < 2.0e-9 for mesh in meshes)
            and all(
                meshes[index + 1]["pressure_supported_balance_error"]
                < meshes[index]["pressure_supported_balance_error"] / 3.0
                for index in range(len(meshes) - 1)
            )
        ),
        "rusanov_audit": _rusanov_audit(),
        "source_free_temporal_audit": _source_free_temporal_audit(),
        "source_free_mesh_audit": _source_free_mesh_audit(),
        "common_stress_flux_audit": _common_stress_flux_audit(),
        "implicit_stress_audit": _implicit_stress_audit(),
        "monolithic_backward_euler_audit": (
            _monolithic_backward_euler_audit()
        ),
        "radiative_cooling_audit": _radiative_cooling_audit(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
