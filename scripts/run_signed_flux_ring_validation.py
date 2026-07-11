"""Resolution validation for the independent-Sigma signed-flux disk core."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedFluxBoundary,
    advance_signed_flux_implicit,
    make_log_grid,
    signed_flux_transport,
)
from imri_qpe.units import solar_masses_to_g


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/signed_flux_ring_validation.json"
N_VALUES = (64, 128, 256)


def _moments(grid, sigma) -> tuple[float, float, float]:
    mass_cells = sigma * grid.area
    total = float(np.sum(mass_cells))
    log_r = np.log(grid.centers)
    mean = float(np.sum(mass_cells * log_r) / total)
    variance = float(np.sum(mass_cells * (log_r - mean) ** 2) / total)
    return total, mean, variance


def run() -> list[dict[str, object]]:
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    radius0 = 80.0 * potential.r_g
    viscosity = 1.0e14
    target_time = 2.0e-4 * radius0**2 / viscosity
    boundary = SignedFluxBoundary(outer_mode="tidal_wall")
    rows = []

    for n in N_VALUES:
        grid = make_log_grid(6.1 * potential.r_g, 300.0 * potential.r_g, n)
        sigma = 1.0 + 100.0 * np.exp(
            -0.5 * (np.log(grid.centers / radius0) / 0.15) ** 2
        )
        initial_mass, initial_mean, initial_variance = _moments(grid, sigma)
        initial_angular = float(
            np.sum(sigma * grid.area * potential.l_k(grid.centers))
        )
        time = 0.0
        expected_mass = initial_mass
        expected_angular = initial_angular
        steps = 0
        minimum_sigma = float(np.min(sigma))
        requested_steps = 20
        while steps < requested_steps:
            dt = target_time / requested_steps
            result = advance_signed_flux_implicit(
                grid, sigma, viscosity, mass, dt, boundary=boundary
            )
            expected_mass += dt * result.transport.mass_budget_rate
            expected_angular += dt * result.transport.angular_momentum_budget_rate
            sigma = result.surface_density
            minimum_sigma = min(minimum_sigma, float(np.min(sigma)))
            time += dt
            steps += 1

        final_mass, final_mean, final_variance = _moments(grid, sigma)
        final_angular = float(
            np.sum(sigma * grid.area * potential.l_k(grid.centers))
        )
        final_transport = signed_flux_transport(
            grid, sigma, viscosity, mass, boundary=boundary
        )
        interior = final_transport.mdot_faces[1:-1]
        rows.append(
            {
                "N": n,
                "steps": steps,
                "time_over_tvisc_R0": time / (radius0**2 / viscosity),
                "minimum_sigma": minimum_sigma,
                "has_inflow": bool(np.any(interior > 0.0)),
                "has_decretion": bool(np.any(interior < 0.0)),
                "mass_budget_relative_error": float(
                    abs(final_mass - expected_mass) / initial_mass
                ),
                "angular_budget_relative_error": float(
                    abs(final_angular - expected_angular) / initial_angular
                ),
                "log_radius_variance_initial": initial_variance,
                "log_radius_variance_final": final_variance,
                "mean_log_radius_change": final_mean - initial_mean,
                "instantaneous_angular_defect_relative": float(
                    abs(final_transport.angular_momentum_budget_defect)
                    / max(abs(final_transport.angular_momentum_budget_rate), 1.0)
                ),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
