"""Audit exact global stream moments and one monolithic source-bearing step."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import M_SUN
from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedThermalClosure,
    advance_global_backward_euler,
    global_compact_stream_cell_sources,
    make_log_grid,
    recover_global_primitives,
    recover_thermodynamics_from_pi_beta,
    state_from_thermodynamic_primitives,
    vertical_state,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/global_signed_stream_preflight.json"


def _equilibrium(n_cells: int):
    central_mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(central_mass)
    grid = make_log_grid(
        12.0 * potential.r_g, 120.0 * potential.r_g, n_cells
    )
    closure = SignedThermalClosure()
    reference_radius = float(np.sqrt(grid.edges[0] * grid.edges[-1]))
    reference = vertical_state(
        1.0e8,
        1.0e6,
        reference_radius,
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )
    sigma, temperature, inversion_residual = recover_thermodynamics_from_pi_beta(
        grid.centers,
        np.full(n_cells, float(reference.Pi)),
        np.full(n_cells, float(reference.P_gas / reference.P_tot)),
        central_mass,
        closure=closure,
        surface_density_seed=np.full(n_cells, 1.0e8),
        temperature_seed=np.full(n_cells, 1.0e6),
    )
    if inversion_residual >= 1.0e-9:
        raise RuntimeError("stream preflight thermodynamic inversion failed")
    state = state_from_thermodynamic_primitives(
        grid,
        sigma,
        np.zeros(n_cells),
        potential.omega_k(grid.centers),
        temperature,
        central_mass,
    )
    return grid, state, central_mass


def main() -> None:
    exact_meshes = []
    total_mass_rate = 7.5e20
    source_radial_velocity = -2.0e7
    source_l = 3.0e19
    source_energy = -4.0e18
    for n_cells in (16, 32, 64):
        grid = make_log_grid(100.0, 300.0, n_cells)
        source = global_compact_stream_cell_sources(
            grid,
            total_mass_rate,
            center=200.0,
            log_width=0.15,
            specific_radial_velocity=source_radial_velocity,
            specific_angular_momentum=source_l,
            specific_total_energy=source_energy,
        )
        exact_meshes.append(
            {
                "n_cells": n_cells,
                "active_source_cells": int(np.count_nonzero(source.mass)),
                "mass_relative_error": float(
                    abs(np.sum(source.mass) / total_mass_rate - 1.0)
                ),
                "radial_momentum_relative_error": float(
                    abs(
                        np.sum(source.radial_momentum)
                        / (total_mass_rate * source_radial_velocity)
                        - 1.0
                    )
                ),
                "angular_momentum_relative_error": float(
                    abs(
                        np.sum(source.angular_momentum)
                        / (total_mass_rate * source_l)
                        - 1.0
                    )
                ),
                "total_energy_relative_error": float(
                    abs(
                        np.sum(source.total_energy)
                        / (total_mass_rate * source_energy)
                        - 1.0
                    )
                ),
            }
        )

    grid, initial, central_mass = _equilibrium(8)
    primitives = recover_global_primitives(grid, initial, central_mass)
    center = float(np.sqrt(grid.edges[0] * grid.edges[-1]))
    dt = 1.0
    step_mass_rate = float(np.sum(initial.mass) * 1.0e-8 / dt)
    step_source = global_compact_stream_cell_sources(
        grid,
        step_mass_rate,
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
    step = advance_global_backward_euler(
        grid,
        initial,
        central_mass,
        dt,
        reference_state=initial,
        boundary_mode="open_no_inflow",
        external_sources=step_source,
    )
    final = recover_global_primitives(grid, step.state, central_mass)
    report = {
        "source_shape": "compact_c2",
        "constant_injected_state": True,
        "exact_moment_meshes": exact_meshes,
        "all_exact_moment_gates_pass": all(
            max(
                mesh["mass_relative_error"],
                mesh["radial_momentum_relative_error"],
                mesh["angular_momentum_relative_error"],
                mesh["total_energy_relative_error"],
            )
            < 3.0e-15
            for mesh in exact_meshes
        ),
        "monolithic_step": {
            "accepted": step.accepted,
            "dt_seconds": dt,
            "source_mass_rate": step_mass_rate,
            "maximum_scaled_residual": step.maximum_scaled_residual,
            "maximum_storage_scaled_ledger_defect": (
                step.maximum_storage_scaled_ledger_defect
            ),
            "nfev": step.nfev,
            "minimum_surface_density": float(
                np.min(final.surface_density)
            ),
            "minimum_temperature": float(np.min(final.temperature)),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
