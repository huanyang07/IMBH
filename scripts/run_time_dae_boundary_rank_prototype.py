"""Compare eliminated and constrained outer DAE rank at small physical meshes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedThermalClosure,
    constrained_tangency_audit,
    eliminated_boundary_tangent,
    eliminated_descriptor_audit,
    evaluate_outer_dae_profile,
    linear_torque_faces,
    make_log_grid,
    matrix_rank_audit,
    normal_closure_audit,
    outer_storage_matrix,
    pack_eliminated_boundary_coordinates,
    pack_outer_primitives,
    unpack_outer_primitives,
)
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/time_dae_boundary_rank_prototype.json"
MESHES = (8, 12, 16)


def _difference_jacobian(function, state, relative_step: float = 1.0e-6):
    state = np.asarray(state, dtype=float)
    base = np.asarray(function(state), dtype=float)
    jacobian = np.empty((base.size, state.size), dtype=float)
    for column in range(state.size):
        step = relative_step * max(1.0, abs(state[column]))
        plus = state.copy()
        minus = state.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (function(plus) - function(minus)) / (
            2.0 * step
        )
    return jacobian


def _rank(matrix, relative: float = 1.0e-9):
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    return int(np.sum(singular > relative * singular[0])), singular


def _one_mesh(n: int):
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(40.0 * potential.r_g, 335.0 * potential.r_g, n)
    coordinate = np.linspace(0.0, 1.0, n)
    sigma_seed = 2.0e5 * np.exp(0.6 * coordinate)
    temperature_seed = 2.0e6 * np.exp(-0.25 * coordinate)
    omega_seed = 0.99 * potential.omega_k(grid.centers)
    closure = SignedThermalClosure()
    alpha = 0.01
    coordinates = pack_eliminated_boundary_coordinates(
        grid,
        sigma_seed,
        temperature_seed,
        omega_seed,
        mass,
        closure=closure,
    )
    state, tangent = eliminated_boundary_tangent(
        coordinates,
        grid,
        mass,
        alpha=alpha,
        closure=closure,
    )
    sigma, temperature, omega = unpack_outer_primitives(state, grid)
    mdot_scale = 5.0 * eddington_mdot(mass)
    mdot_scaled = -np.linspace(0.2, 0.8, n + 1)
    profile = evaluate_outer_dae_profile(
        grid,
        sigma,
        temperature,
        omega,
        mdot_scale * mdot_scaled,
        mass,
        alpha=alpha,
        closure=closure,
    )
    storage, row_scales = outer_storage_matrix(
        state,
        grid,
        mass,
        closure=closure,
    )
    time_scale = float(np.sum(profile.mass_cells) / mdot_scale)
    scaled_storage = storage / row_scales[:, None]

    def differential_rhs(face_state):
        trial = evaluate_outer_dae_profile(
            grid,
            sigma,
            temperature,
            omega,
            mdot_scale * face_state,
            mass,
            alpha=alpha,
            closure=closure,
        )
        rhs = np.concatenate(
            (trial.mass_rhs, trial.angular_rhs, trial.energy_rhs)
        )
        return time_scale * rhs / row_scales

    def radial_rows(face_state):
        return evaluate_outer_dae_profile(
            grid,
            sigma,
            temperature,
            omega,
            mdot_scale * face_state,
            mass,
            alpha=alpha,
            closure=closure,
        ).radial_residual

    flux_jacobian = _difference_jacobian(differential_rhs, mdot_scaled)
    algebraic_jacobian = _difference_jacobian(radial_rows, mdot_scaled)
    eliminated = eliminated_descriptor_audit(
        scaled_storage,
        tangent,
        flux_jacobian,
        algebraic_jacobian,
        relative_threshold=1.0e-9,
    )
    normal = normal_closure_audit(
        scaled_storage,
        tangent,
        flux_jacobian,
        algebraic_jacobian,
        relative_threshold=1.0e-9,
    )

    torque_scale = float(np.max(np.abs(profile.torque_centers)))

    def constraint(full_state):
        local_sigma, local_temperature, local_omega = unpack_outer_primitives(
            full_state, grid
        )
        local = evaluate_outer_dae_profile(
            grid,
            local_sigma,
            local_temperature,
            local_omega,
            mdot_scale * mdot_scaled,
            mass,
            alpha=alpha,
            closure=closure,
        )
        return np.asarray([local.torque_faces[-1] / torque_scale])

    constraint_gradient = _difference_jacobian(constraint, state)[0]
    constrained = constrained_tangency_audit(
        scaled_storage,
        constraint_gradient,
        flux_jacobian,
        algebraic_jacobian,
        relative_threshold=1.0e-9,
    )
    storage_rank, storage_singular = _rank(scaled_storage)
    tangent_rank, tangent_singular = _rank(tangent)
    eliminated_equilibrated = matrix_rank_audit(
        eliminated.matrix,
        relative_threshold=1.0e-9,
        equilibrate=True,
    )
    normal_equilibrated = matrix_rank_audit(
        normal.matrix,
        relative_threshold=1.0e-9,
        equilibrate=True,
    )
    constrained_equilibrated = matrix_rank_audit(
        constrained.matrix,
        relative_threshold=1.0e-9,
        equilibrate=True,
    )
    omega_k = potential.omega_k(grid.centers)
    sound_speed = profile.H * omega_k
    radial_mach = np.abs(profile.radial_velocity) / sound_speed
    return {
        "outer_cells": n,
        "outer_torque_relative": float(
            linear_torque_faces(grid, profile.torque_centers)[-1]
            / torque_scale
        ),
        "maximum_radial_mach": float(np.max(radial_mach)),
        "storage_rank": storage_rank,
        "storage_dimension": int(scaled_storage.shape[0]),
        "storage_smallest_singular": float(storage_singular[-1]),
        "tangent_rank": tangent_rank,
        "tangent_columns": int(tangent.shape[1]),
        "tangent_smallest_singular": float(tangent_singular[-1]),
        "eliminated_rank": eliminated.rank,
        "eliminated_dimension": int(eliminated.matrix.shape[0]),
        "eliminated_condition": eliminated.condition_estimate,
        "eliminated_smallest_singular": float(eliminated.singular_values[-1]),
        "eliminated_equilibrated_rank": eliminated_equilibrated.rank,
        "eliminated_equilibrated_condition": (
            eliminated_equilibrated.condition_estimate
        ),
        "normal_rank": normal.rank,
        "normal_dimension": int(normal.matrix.shape[0]),
        "normal_condition": normal.condition_estimate,
        "normal_equilibrated_rank": normal_equilibrated.rank,
        "normal_equilibrated_condition": normal_equilibrated.condition_estimate,
        "constrained_rank": constrained.rank,
        "constrained_dimension": int(constrained.matrix.shape[0]),
        "constrained_condition": constrained.condition_estimate,
        "constrained_equilibrated_rank": constrained_equilibrated.rank,
        "constrained_equilibrated_condition": (
            constrained_equilibrated.condition_estimate
        ),
        "algebraic_rank": _rank(algebraic_jacobian)[0],
        "algebraic_rows": int(algebraic_jacobian.shape[0]),
    }


def main() -> None:
    report = {"meshes": [_one_mesh(n) for n in MESHES]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
