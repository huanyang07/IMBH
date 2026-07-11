"""Thermal relaxation of the absolute-stream signed-flux pilot."""

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
    solve_signed_flux_steady,
    solve_signed_thermal_steady,
)
from imri_qpe.scales import eddington_luminosity, eddington_mdot
from imri_qpe.units import solar_masses_to_g


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/signed_flux_thermal_pilot.json"
N_VALUES = (64, 128, 256, 512)


def run() -> list[dict[str, object]]:
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = 5.0 * eddington_mdot(mass)
    stream_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(stream_radius))
    stream_B = float(
        potential.phi(stream_radius) + 0.5 * (stream_l / stream_radius) ** 2
    )
    rows = []
    for n in N_VALUES:
        grid = make_log_grid(6.1 * potential.r_g, 335.0 * potential.r_g, n)
        omega = potential.omega_k(grid.centers)
        viscosity = 0.01 * 0.1**2 * grid.centers**2 * omega
        stream_state = normalized_stream_injection_state(
            grid,
            stream_rate,
            center=240.0 * potential.r_g,
            log_width=0.08,
            specific_angular_momentum=stream_l,
            specific_total_energy=stream_B,
        )
        closure = SignedThermalClosure(
            temperature_bounds=(1.0e3, 1.0e9),
        )
        for outer_mode in ("tidal_wall", "zero_torque"):
            transport = solve_signed_flux_steady(
                grid,
                viscosity,
                mass,
                boundary=SignedFluxBoundary(outer_mode=outer_mode),
                stream_state=stream_state,
            )
            thermal = solve_signed_thermal_steady(
                grid,
                transport,
                np.full(n, 1.0e6),
                mass,
                closure=closure,
                tolerance=1.0e-6,
                max_nfev=500,
            )
            profile = thermal.profile
            viscous = float(np.sum(profile.viscous_heating_rate_cells))
            stream = float(np.sum(profile.stream_heating_rate_cells))
            radiative = float(np.sum(profile.radiative_cooling_rate_cells))
            advective_sink = float(-np.sum(profile.advective_rate_cells))
            input_power = viscous + stream
            thin = profile.tau < 1.0
            row = {
                "N": n,
                "outer_mode": outer_mode,
                "accepted": thermal.accepted,
                "nfev": thermal.nfev,
                "maximum_normalized_residual": thermal.maximum_normalized_residual,
                "inner_mdot_over_stream": float(transport.mdot_faces[0] / stream_rate),
                "outer_mdot_over_stream": float(transport.mdot_faces[-1] / stream_rate),
                "temperature_min": float(np.min(thermal.temperature)),
                "temperature_max": float(np.max(thermal.temperature)),
                "max_H_over_R": float(np.max(profile.H / grid.centers)),
                "minimum_tau": float(np.min(profile.tau)),
                "minimum_tau_radius_rg": float(
                    grid.centers[int(np.argmin(profile.tau))] / potential.r_g
                ),
                "cells_tau_lt1": int(np.count_nonzero(thin)),
                "luminosity_fraction_tau_lt1": float(
                    np.sum(profile.radiative_cooling_rate_cells[thin])
                    / max(radiative, 1.0)
                ),
                "Lrad_over_LEdd": radiative / eddington_luminosity(mass),
                "stream_heating_over_viscous": stream / viscous,
                "internal_energy_export_fraction": advective_sink / input_power,
                "maximum_radial_pressure_force_fraction": float(
                    np.max(profile.radial_pressure_force_fraction)
                ),
                "minimum_dln_l_k_dln_R": float(np.min(profile.dln_l_k_dln_R)),
                "effective_optical_depth_available": False,
                "fixed_radius_diagnostics": signed_thermal_fixed_radius_diagnostics(
                    grid, profile, potential.r_g
                ),
                "energy_closure_relative": float(
                    (input_power - radiative - advective_sink)
                    / max(abs(input_power), abs(radiative), abs(advective_sink), 1.0)
                ),
                "internal_energy_ledger_defect_relative": float(
                    profile.internal_energy_ledger_defect
                    / max(abs(input_power), abs(radiative), 1.0)
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
