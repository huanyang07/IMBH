"""Audit the repository-compatible flux-primary time DAE at small meshes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedThermalClosure,
    angular_fluxes_for_common_stress_open_edge,
    common_stress_torque_centers,
    evaluate_flux_primary_outer_dae_profile,
    make_log_grid,
    matrix_rank_audit,
    outer_storage_matrix,
    pack_outer_primitives,
)
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/time_dae_flux_primary_rank_prototype.json"
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


def _one_mesh(n: int):
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(40.0 * potential.r_g, 335.0 * potential.r_g, n)
    coordinate = np.linspace(0.0, 1.0, n)
    sigma = 2.0e5 * np.exp(0.6 * coordinate)
    temperature = 2.0e6 * np.exp(-0.25 * coordinate)
    omega = 0.99 * potential.omega_k(grid.centers)
    closure = SignedThermalClosure()
    alpha = 0.01
    state = pack_outer_primitives(sigma, temperature, omega)
    mdot_scale = 5.0 * eddington_mdot(mass)
    mdot_scaled = np.linspace(0.2, -0.8, n + 1)
    mdot_faces = mdot_scale * mdot_scaled
    common_torque = common_stress_torque_centers(
        grid,
        sigma,
        temperature,
        mass,
        alpha=alpha,
        closure=closure,
    )
    angular_faces = angular_fluxes_for_common_stress_open_edge(
        grid,
        mdot_faces,
        omega,
        common_torque,
    )
    angular_scale = max(float(np.max(np.abs(angular_faces))), 1.0)
    algebraic_state = np.concatenate(
        (mdot_scaled, angular_faces / angular_scale)
    )
    profile = evaluate_flux_primary_outer_dae_profile(
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
    storage, row_scales = outer_storage_matrix(
        state,
        grid,
        mass,
        closure=closure,
    )
    scaled_storage = storage / row_scales[:, None]
    time_scale = float(np.sum(profile.profile.mass_cells) / mdot_scale)
    inner_angular_target = float(angular_faces[0])
    torque_scale = max(float(np.max(np.abs(common_torque))), 1.0)

    def evaluate_y(values):
        local_mdot = mdot_scale * values[: n + 1]
        local_angular = angular_scale * values[n + 1 :]
        return evaluate_flux_primary_outer_dae_profile(
            grid,
            sigma,
            temperature,
            omega,
            local_mdot,
            local_angular,
            mass,
            alpha=alpha,
            closure=closure,
        )

    def differential_rhs(values):
        local = evaluate_y(values).profile
        rhs = np.concatenate(
            (local.mass_rhs, local.angular_rhs, local.energy_rhs)
        )
        return time_scale * rhs / row_scales

    def algebraic_rows(values):
        local = evaluate_y(values)
        edge = local.profile.torque_faces[-1] / torque_scale
        inner = (
            local.profile.angular_flux_faces[0] - inner_angular_target
        ) / angular_scale
        return np.concatenate(
            (local.stress_residual, local.profile.radial_residual, [inner, edge])
        )

    flux_jacobian = _difference_jacobian(differential_rhs, algebraic_state)
    algebraic_jacobian = _difference_jacobian(
        algebraic_rows, algebraic_state
    )
    descriptor = np.block(
        [
            [scaled_storage, -flux_jacobian],
            [
                np.zeros(
                    (algebraic_jacobian.shape[0], scaled_storage.shape[1])
                ),
                algebraic_jacobian,
            ],
        ]
    )
    raw_algebraic = matrix_rank_audit(
        algebraic_jacobian, relative_threshold=1.0e-9
    )
    scaled_algebraic = matrix_rank_audit(
        algebraic_jacobian,
        relative_threshold=1.0e-9,
        equilibrate=True,
    )
    raw_descriptor = matrix_rank_audit(
        descriptor, relative_threshold=1.0e-9
    )
    scaled_descriptor = matrix_rank_audit(
        descriptor,
        relative_threshold=1.0e-9,
        equilibrate=True,
    )
    sound_speed = profile.profile.H * potential.omega_k(grid.centers)
    radial_mach = np.abs(profile.profile.radial_velocity) / sound_speed
    return {
        "outer_cells": n,
        "differential_dimension": 3 * n,
        "algebraic_dimension": 2 * n + 2,
        "total_dimension": 5 * n + 2,
        "maximum_radial_mach": float(np.max(radial_mach)),
        "maximum_stress_residual": float(
            np.max(np.abs(profile.stress_residual))
        ),
        "outer_torque_relative": float(
            profile.profile.torque_faces[-1] / torque_scale
        ),
        "algebraic_raw_rank": raw_algebraic.rank,
        "algebraic_equilibrated_rank": scaled_algebraic.rank,
        "algebraic_equilibrated_condition": (
            scaled_algebraic.condition_estimate
        ),
        "descriptor_raw_rank": raw_descriptor.rank,
        "descriptor_equilibrated_rank": scaled_descriptor.rank,
        "descriptor_equilibrated_condition": (
            scaled_descriptor.condition_estimate
        ),
    }


def main() -> None:
    report = {"meshes": [_one_mesh(n) for n in MESHES]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
