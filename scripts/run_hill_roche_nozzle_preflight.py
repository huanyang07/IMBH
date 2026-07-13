"""Evaluate the standalone Hill/Roche nozzle on mapped physical edge states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GasRadiationHillRocheNozzleProvider,
    HillRocheNozzleReservoir,
    fiducial_hill_roche_nozzle_geometry,
    gas_radiation_adiabatic_sound_speed_squared,
    evaluate_global_rusanov_profile,
    reconstruct_global_outer_edge_state,
)

from run_global_physical_open_preflight import (
    _canonical_open_evaluation,
    _initial_case,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/hill_roche_nozzle_preflight.json"


def _mapped_edge_gate(
    context, evaluation, n_cells: int, filling_factor: float
) -> dict:
    (
        grid,
        state,
        _mass,
        _potential,
        stream_rate,
        _stream,
        primitives,
        correction,
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
    edge = reconstruct_global_outer_edge_state(
        grid, primitives, context.base.inner_params.M2_g
    )
    gamma_one = (
        gas_radiation_adiabatic_sound_speed_squared(
            edge.density, edge.temperature
        )
        * edge.density
        / edge.pressure
    )
    reservoir = HillRocheNozzleReservoir(
        radius=edge.radius,
        density=edge.density,
        pressure=edge.pressure,
        radial_velocity=edge.radial_velocity,
        specific_angular_momentum=edge.specific_angular_momentum,
        temperature=edge.temperature,
    )
    geometry = fiducial_hill_roche_nozzle_geometry(
        channel_count=2, filling_factor=filling_factor
    )
    provider = GasRadiationHillRocheNozzleProvider(
        geometry, transverse_quadrature_zones=48
    )
    profile = evaluate_global_rusanov_profile(
        grid,
        state,
        context.base.inner_params.M2_g,
        boundary_mode="roche_outer",
        stress_boundary_mode="outer_zero_torque",
        primitives=primitives,
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=correction,
    )
    boundary = profile.outer_roche_boundary
    if boundary is None:
        raise RuntimeError("production Roche boundary audit is missing")
    gate = boundary.gate
    result = {
        "n_cells": n_cells,
        "filling_factor": filling_factor,
        "gamma_one": gamma_one,
        "reservoir_radius_over_hill_radius": (
            reservoir.radius / geometry.nominal_hill_radius
        ),
        "saddle_radius_over_hill_radius": (
            geometry.saddle_radius / geometry.nominal_hill_radius
        ),
        "surface_density": edge.surface_density,
        "density": reservoir.density,
        "temperature": reservoir.temperature,
        "pressure": reservoir.pressure,
        "radial_velocity": reservoir.radial_velocity,
        "specific_angular_momentum": reservoir.specific_angular_momentum,
        "disk_bernoulli": edge.bernoulli,
        "choked": gate.choked,
        "available_specific_energy": gate.available_specific_energy,
        "reservoir_enthalpy": gate.reservoir_enthalpy,
        "required_enthalpy_multiplier": gate.required_enthalpy_multiplier,
        "incoming_acoustic_conditions": (
            boundary.incoming_acoustic_conditions
        ),
        "no_inward_mass": boundary.no_inward_mass,
        "applied_mass_flux": boundary.applied_mass_flux,
        "pressure_traction": boundary.pressure_traction,
        "angular_flux_relative_mismatch": (
            boundary.angular_flux_relative_mismatch
        ),
        "energy_flux_relative_mismatch": (
            boundary.energy_flux_relative_mismatch
        ),
        "binary_pattern_power_relative_mismatch": (
            boundary.binary_pattern_power_relative_mismatch
        ),
    }
    if gate.solution is not None:
        result.update(
            {
                "overflow_over_stream_supply": (
                    gate.solution.saddle_flux.mass / stream_rate
                ),
                "sonic_residual": gate.solution.sonic_residual,
                "entropy_residual": gate.solution.entropy_residual,
                "jacobi_residual": gate.solution.jacobi_residual,
                "energy_pairing_residual": (
                    gate.solution.energy_pairing_residual
                ),
                "disk_energy_relative_mismatch": (
                    gate.solution.edge_total_energy_flux
                    / (gate.solution.saddle_flux.mass * edge.bernoulli)
                    - 1.0
                ),
            }
        )
    return result


def main() -> None:
    context, evaluation = _canonical_open_evaluation()
    report = {
        "model": {
            "potential": "PW secondary plus local Hill tide",
            "thermal_process": "adiabatic exact shared gas+radiation EOS",
            "channel_count": 2,
            "filling_factors": [0.25, 0.5, 1.0],
            "production_boundary": True,
        },
        "mapped_edge_gates": [
            _mapped_edge_gate(context, evaluation, n_cells, filling_factor)
            for n_cells in (64, 96, 128)
            for filling_factor in (0.25, 0.5, 1.0)
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
