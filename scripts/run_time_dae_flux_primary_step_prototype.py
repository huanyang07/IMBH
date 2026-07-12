"""Run conservative flux-primary backward-Euler steps from the open control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    advance_flux_primary_outer_dae_backward_euler,
    angular_fluxes_for_common_stress_open_edge,
    audit_outer_dae_backward_euler_ledgers,
    common_stress_torque_centers,
    evaluate_coupled_open_overflow_residual,
    evaluate_flux_primary_outer_dae_profile,
    make_log_grid,
    pack_outer_primitives,
)

from run_coupled_inner_outer_mesh_certification import _load_source
from run_coupled_open_overflow_continuation import _open_context, _target_mesh


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical/coupled_open_overflow_eigenvalue"
OUTPUT = ROOT / "outputs/tables/time_dae_flux_primary_step_prototype.json"
TARGET_MESHES = (16, 32)


def _load_state(name: str) -> np.ndarray:
    with np.load(CANONICAL / name) as data:
        return np.asarray(data["state"], dtype=float)


def _positive_interpolate(target, source, values):
    return np.exp(np.interp(np.log(target), np.log(source), np.log(values)))


def _one_mesh(source_evaluation, context, n: int):
    source_grid = context.base.outer_grid
    grid = make_log_grid(source_grid.edges[0], source_grid.edges[-1], n)
    transport = source_evaluation.base.outer_transport
    energy = source_evaluation.base.outer_energy_profile
    sigma = _positive_interpolate(
        grid.centers, source_grid.centers, transport.surface_density
    )
    temperature = _positive_interpolate(
        grid.centers, source_grid.centers, energy.temperature
    )
    omega = _positive_interpolate(
        grid.centers, source_grid.centers, transport.omega
    )
    mdot_faces = np.interp(
        np.log(grid.edges),
        np.log(source_grid.edges),
        transport.mdot_faces,
    )
    params = context.base.inner_params
    closure = context.base.outer_closure
    alpha = context.base.alpha
    common = common_stress_torque_centers(
        grid,
        sigma,
        temperature,
        params.M2_g,
        alpha=alpha,
        closure=closure,
        stress_factor=context.base.stress_factor,
    )
    angular_faces = angular_fluxes_for_common_stress_open_edge(
        grid, mdot_faces, omega, common
    )
    potential = PaczynskiWiitaPotential(params.M2_g)
    provisional = evaluate_flux_primary_outer_dae_profile(
        grid,
        sigma,
        temperature,
        omega,
        mdot_faces,
        angular_faces,
        params.M2_g,
        alpha=alpha,
        closure=closure,
        stress_factor=context.base.stress_factor,
    )
    omega_k = potential.omega_k(grid.centers)
    omega = np.sqrt(
        omega**2 + provisional.profile.radial_residual * omega_k**2
    )
    common = common_stress_torque_centers(
        grid,
        sigma,
        temperature,
        params.M2_g,
        alpha=alpha,
        closure=closure,
        stress_factor=context.base.stress_factor,
    )
    angular_faces = angular_fluxes_for_common_stress_open_edge(
        grid, mdot_faces, omega, common
    )
    state = pack_outer_primitives(sigma, temperature, omega)
    old = evaluate_flux_primary_outer_dae_profile(
        grid,
        sigma,
        temperature,
        omega,
        mdot_faces,
        angular_faces,
        params.M2_g,
        alpha=alpha,
        closure=closure,
        stress_factor=context.base.stress_factor,
    )
    loading_time = float(
        np.sum(old.profile.mass_cells) / np.max(np.abs(mdot_faces))
    )
    rows = []
    for fraction in (1.0e-8, 1.0e-7, 1.0e-6):
        dt = fraction * loading_time
        result = advance_flux_primary_outer_dae_backward_euler(
            state,
            mdot_faces,
            angular_faces,
            grid,
            params.M2_g,
            dt,
            alpha=alpha,
            closure=closure,
            stress_factor=context.base.stress_factor,
            tolerance=1.0e-7,
            max_nfev=400,
        )
        ledger = audit_outer_dae_backward_euler_ledgers(
            old.profile, result.evaluation.profile, dt
        )
        rows.append(
            {
                "dt_over_loading_time": fraction,
                "accepted": result.accepted,
                "maximum_residual": result.maximum_residual,
                "nfev": result.nfev,
                "mdot_inner": float(result.mdot_faces[0]),
                "mdot_outer": float(result.mdot_faces[-1]),
                "mass_defect": ledger.relative_mass_defect,
                "angular_defect": ledger.relative_angular_momentum_defect,
                "energy_defect": ledger.relative_energy_defect,
            }
        )
    return {"outer_cells": n, "steps": rows}


def main() -> None:
    base, _wall_state = _load_source()
    context_96 = _open_context(base, 1.0)
    state_96 = _load_state("Ninner96_Nouter64.npz")
    context_144, _seed = _target_mesh(context_96, state_96, 144, 96)
    state_144 = _load_state("Ninner144_Nouter96.npz")
    evaluation = evaluate_coupled_open_overflow_residual(
        state_144, context_144
    )
    report = {
        "meshes": [_one_mesh(evaluation, context_144, n) for n in TARGET_MESHES]
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
