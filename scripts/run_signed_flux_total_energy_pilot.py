"""Resolution ladder for the angularly closed total-energy reservoir."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedFluxBoundary,
    SignedThermalClosure,
    make_log_grid,
    normalized_stream_injection_state,
    signed_thermal_fixed_radius_diagnostics,
    solve_signed_total_energy_thermoviscous_steady,
)
from imri_qpe.scales import eddington_luminosity, eddington_mdot
from imri_qpe.units import solar_masses_to_g


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/signed_flux_total_energy_pilot.json"
N_VALUES = (64, 128, 256, 512)
INNER_RADII_RG = (6.1, 10.0)


def run() -> list[dict[str, object]]:
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = 5.0 * eddington_mdot(mass)
    stream_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(stream_radius))
    stream_B = float(
        potential.phi(stream_radius)
        + 0.5 * (stream_l / stream_radius) ** 2
    )
    closure = SignedThermalClosure(temperature_bounds=(1.0e3, 1.0e9))
    rows = []
    for inner_radius_rg in INNER_RADII_RG:
        for n in N_VALUES:
            grid = make_log_grid(
                inner_radius_rg * potential.r_g, 335.0 * potential.r_g, n
            )
            stream = normalized_stream_injection_state(
                grid,
                stream_rate,
                center=240.0 * potential.r_g,
                log_width=0.08,
                specific_angular_momentum=stream_l,
                specific_total_energy=stream_B,
            )
            for outer_mode in ("tidal_wall", "zero_torque"):
                solved = solve_signed_total_energy_thermoviscous_steady(
                    grid,
                    mass,
                    alpha=0.01,
                    boundary=SignedFluxBoundary(outer_mode=outer_mode),
                    stream_state=stream,
                    closure=closure,
                    temperature_seed=np.full(n, 1.0e6),
                    damping=0.2,
                    tolerance=2.0e-3,
                    max_iterations=60,
                    energy_tolerance=1.0e-6,
                    energy_max_nfev=1000,
                )
                transport = solved.transport
                profile = solved.energy.profile
                radiative = float(np.sum(profile.radiative_loss_rate_cells))
                total_scale = max(
                    radiative,
                    float(np.sum(np.abs(profile.stream_energy_rate_cells))),
                    float(np.sum(np.abs(profile.vertical_work_rate_cells))),
                    1.0,
                )
                row = {
                    "inner_radius_rg": inner_radius_rg,
                    "N": n,
                    "outer_mode": outer_mode,
                    "converged": solved.converged,
                    "iterations": solved.iterations,
                    "maximum_log_viscosity_change": (
                        solved.maximum_log_viscosity_change
                    ),
                    "total_energy_residual": (
                        solved.energy.maximum_normalized_residual
                    ),
                    "inner_mdot_over_stream": float(
                        transport.mdot_faces[0] / stream_rate
                    ),
                    "outer_mdot_over_stream": float(
                        transport.mdot_faces[-1] / stream_rate
                    ),
                    "outer_torque_over_stream_J": float(
                        transport.viscous_torque_faces[-1]
                        / (stream_rate * stream_l)
                    ),
                    "unmodeled_angular_defect_relative": float(
                        transport.angular_momentum_budget_defect
                        / (stream_rate * stream_l)
                    ),
                    "temperature_min": float(np.min(solved.energy.temperature)),
                    "temperature_max": float(np.max(solved.energy.temperature)),
                    "max_H_over_R": float(np.max(profile.H / grid.centers)),
                    "minimum_tau_scattering": float(np.min(profile.tau)),
                    "Lrad_over_LEdd": radiative / eddington_luminosity(mass),
                    "vertical_work_over_Lrad": float(
                        np.sum(profile.vertical_work_rate_cells) / radiative
                    ),
                    "outer_torque_work_over_LEdd": float(
                        profile.torque_work_flux_faces[-1]
                        / eddington_luminosity(mass)
                    ),
                    "total_energy_telescoping_defect_relative": float(
                        profile.total_energy_telescoping_defect / total_scale
                    ),
                    "maximum_radial_pressure_force_fraction": float(
                        np.max(profile.radial_pressure_force_fraction)
                    ),
                    "fixed_radius_diagnostics": (
                        signed_thermal_fixed_radius_diagnostics(
                            grid,
                            profile,
                            potential.r_g,
                            radii_rg=(12.0, 15.0, 20.0, 30.0),
                        )
                    ),
                    "effective_optical_depth_available": False,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
