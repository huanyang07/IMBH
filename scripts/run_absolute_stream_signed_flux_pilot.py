"""Absolute stream-supply pilot with tidal-wall and open outer boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedFluxBoundary,
    make_log_grid,
    normalized_stream_cell_rates,
    solve_signed_flux_steady,
)
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/absolute_stream_signed_flux_pilot.json"
N_VALUES = (128, 256, 512)


def _stagnation_radius(grid, mdot_faces) -> float | None:
    sign_change = np.flatnonzero(mdot_faces[:-1] * mdot_faces[1:] < 0.0)
    if sign_change.size == 0:
        zeros = np.flatnonzero(mdot_faces == 0.0)
        return None if zeros.size == 0 else float(grid.edges[int(zeros[0])])
    index = int(sign_change[0])
    left = float(mdot_faces[index])
    right = float(mdot_faces[index + 1])
    fraction = -left / (right - left)
    return float(np.exp((1.0 - fraction) * np.log(grid.edges[index]) + fraction * np.log(grid.edges[index + 1])))


def run() -> list[dict[str, object]]:
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = 5.0 * eddington_mdot(mass)
    stream_l = float(potential.l_k(248.96693 * potential.r_g))
    rows = []
    for n in N_VALUES:
        grid = make_log_grid(6.1 * potential.r_g, 335.0 * potential.r_g, n)
        omega = np.asarray(potential.omega_k(grid.centers), dtype=float)
        viscosity = 0.01 * 0.1**2 * grid.centers**2 * omega
        source_mass, _ = normalized_stream_cell_rates(
            grid,
            stream_rate,
            center=240.0 * potential.r_g,
            log_width=0.08,
            specific_angular_momentum=stream_l,
        )
        source_l = np.full(n, stream_l, dtype=float)
        for outer_mode in ("tidal_wall", "zero_torque"):
            result = solve_signed_flux_steady(
                grid,
                viscosity,
                mass,
                boundary=SignedFluxBoundary(outer_mode=outer_mode),
                source_mass_rate_cells=source_mass,
                source_specific_angular_momentum=source_l,
            )
            stagnation = _stagnation_radius(grid, result.mdot_faces)
            angular_scale = stream_rate * stream_l
            row = {
                "N": n,
                "outer_mode": outer_mode,
                "stream_rate_over_edd": 5.0,
                "inner_mdot_over_stream": float(result.mdot_faces[0] / stream_rate),
                "outer_mdot_over_stream": float(result.mdot_faces[-1] / stream_rate),
                "mass_residual_max_over_stream": float(
                    np.max(np.abs(result.mass_rate_cells)) / stream_rate
                ),
                "minimum_sigma": float(np.min(result.surface_density)),
                "outer_sigma": float(result.surface_density[-1]),
                "stagnation_radius_rg": (
                    None if stagnation is None else stagnation / potential.r_g
                ),
                "outer_torque_over_stream_J": float(
                    result.viscous_torque_faces[-1] / angular_scale
                ),
                "required_mixing_torque_over_stream_J": float(
                    result.angular_momentum_budget_defect / angular_scale
                ),
                "angular_budget_rate_over_stream_J": float(
                    result.angular_momentum_budget_rate / angular_scale
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
