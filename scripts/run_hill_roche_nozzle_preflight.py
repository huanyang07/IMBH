"""Evaluate the standalone Hill/Roche nozzle on mapped physical edge states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    HillRocheNozzleProvider,
    HillRocheNozzleReservoir,
    fiducial_hill_roche_nozzle_geometry,
)
from imri_qpe.layer3_minidisk_1d.signed_flux_common_stress import (
    positive_edge_reconstruction,
)

from run_global_physical_open_preflight import (
    _canonical_open_evaluation,
    _initial_case,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/hill_roche_nozzle_preflight.json"


def _mapped_edge_gate(context, evaluation, n_cells: int) -> dict:
    (
        grid,
        _state,
        _mass,
        _potential,
        stream_rate,
        _stream,
        primitives,
        _correction,
        _mapping,
    ) = _initial_case(
        context,
        evaluation,
        n_cells,
        mapping_mode="conservative",
        open_face_reconstruction="conserved_donor",
        include_vertical_column_work=True,
        boundary_mode="characteristic_inner_open_outer",
    )
    gas_pressure = float(
        positive_edge_reconstruction(grid, primitives.vertical.P_gas)[-1]
    )
    radiation_pressure = float(
        positive_edge_reconstruction(grid, primitives.vertical.P_rad)[-1]
    )
    gamma = (
        (5.0 / 3.0) * gas_pressure
        + (4.0 / 3.0) * radiation_pressure
    ) / (gas_pressure + radiation_pressure)
    log_centers = np.log(grid.centers)
    log_outer = float(np.log(grid.edges[-1]))
    velocity_slope = (
        primitives.radial_velocity[-1] - primitives.radial_velocity[-2]
    ) / (log_centers[-1] - log_centers[-2])
    outer_velocity = float(
        primitives.radial_velocity[-1]
        + velocity_slope * (log_outer - log_centers[-1])
    )
    outer_omega = float(
        positive_edge_reconstruction(grid, primitives.omega)[-1]
    )
    reservoir = HillRocheNozzleReservoir(
        radius=float(grid.edges[-1]),
        density=float(
            positive_edge_reconstruction(grid, primitives.vertical.rho)[-1]
        ),
        pressure=gas_pressure + radiation_pressure,
        radial_velocity=outer_velocity,
        specific_angular_momentum=float(grid.edges[-1] ** 2 * outer_omega),
    )
    geometry = fiducial_hill_roche_nozzle_geometry(
        channel_count=2, filling_factor=1.0
    )
    gate = HillRocheNozzleProvider(geometry, gamma=gamma).evaluate(reservoir)
    result = {
        "n_cells": n_cells,
        "gamma": gamma,
        "reservoir_radius_over_hill_radius": (
            reservoir.radius / geometry.nominal_hill_radius
        ),
        "saddle_radius_over_hill_radius": (
            geometry.saddle_radius / geometry.nominal_hill_radius
        ),
        "density": reservoir.density,
        "pressure": reservoir.pressure,
        "radial_velocity": reservoir.radial_velocity,
        "specific_angular_momentum": reservoir.specific_angular_momentum,
        "choked": gate.choked,
        "available_specific_energy": gate.available_specific_energy,
        "reservoir_enthalpy": gate.reservoir_enthalpy,
        "required_enthalpy_multiplier": gate.required_enthalpy_multiplier,
    }
    if gate.solution is not None:
        result.update(
            {
                "overflow_over_stream_supply": (
                    gate.solution.saddle_flux.mass / stream_rate
                ),
                "sonic_residual": gate.solution.sonic_residual,
                "jacobi_residual": gate.solution.jacobi_residual,
                "energy_pairing_residual": (
                    gate.solution.energy_pairing_residual
                ),
            }
        )
    return result


def main() -> None:
    context, evaluation = _canonical_open_evaluation()
    report = {
        "model": {
            "potential": "PW secondary plus local Hill tide",
            "thermal_process": "adiabatic fixed local gamma",
            "channel_count": 2,
            "filling_factor": 1.0,
            "production_boundary": False,
        },
        "mapped_edge_gates": [
            _mapped_edge_gate(context, evaluation, n_cells)
            for n_cells in (64, 96)
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
