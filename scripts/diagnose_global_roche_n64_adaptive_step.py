"""Locate the changes limiting the current N64 adaptive Roche-loading step."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalAdaptiveStepConfig,
    PaczynskiWiitaPotential,
    advance_global_adaptive_backward_euler,
    global_inner_characteristic_audit,
    load_global_adaptive_restart,
    recover_global_primitives,
)

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
RESTART = ROOT / "outputs/checkpoints/global_roche_adaptive_N64.npz"
OUTPUT = ROOT / "outputs/tables/global_roche_n64_adaptive_step_diagnostic.json"


def _maximum_location(values: np.ndarray, radii: np.ndarray, r_g: float) -> dict:
    index = int(np.nanargmax(values))
    return {
        "value": float(values[index]),
        "cell_index": index,
        "radius_over_r_g": float(radii[index] / r_g),
    }


def main() -> None:
    context, evaluation = _canonical_open_evaluation()
    grid, _canonical, _correction, stream, stream_rate, provider = _prepared_case(
        context, evaluation, 64
    )
    mass = context.base.inner_params.M2_g
    potential = PaczynskiWiitaPotential(mass)
    _loaded_grid, restart = load_global_adaptive_restart(RESTART, grid=grid)
    current = restart.state
    reference = restart.reference_state
    correction = restart.mechanical_reference.specific_offset
    loading_time = float(np.sum(reference.mass) / stream_rate)
    config = GlobalAdaptiveStepConfig(
        minimum_dt=1.0e-9 * loading_time,
        maximum_dt=5.0e-7 * loading_time,
        shrink_factor=0.5,
        growth_factor=1.5,
        maximum_retries=6,
        easy_nfev=20,
        maximum_log_surface_density_change=0.02,
        maximum_log_temperature_change=0.02,
        maximum_relative_thickness_change=0.02,
    )
    step_options = {
        "alpha": context.base.alpha,
        "reference_state": reference,
        "boundary_mode": "characteristic_inner_roche_outer",
        "stress_boundary_mode": "outer_zero_torque",
        "include_radiative_cooling": True,
        "include_vertical_column_work": True,
        "external_sources": stream,
        "jacobian_mode": "sparse_forward",
        "outer_overflow_provider": provider,
        "max_nfev": 300,
    }
    result = advance_global_adaptive_backward_euler(
        grid,
        current,
        mass,
        restart.dt_next,
        config,
        specific_mechanical_energy_correction=correction,
        step_options=step_options,
    )
    old = recover_global_primitives(
        grid,
        current,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    initial = recover_global_primitives(
        grid,
        reference,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    current_characteristic = global_inner_characteristic_audit(old)
    initial_characteristic = global_inner_characteristic_audit(initial)
    report = {
        "elapsed_loading_fraction": restart.elapsed_time / loading_time,
        "requested_dt_over_loading_time": restart.dt_next / loading_time,
        "accepted": result.accepted,
        "attempts": [
            {
                "dt_over_loading_time": attempt.dt / loading_time,
                "nonlinear_accepted": attempt.nonlinear_accepted,
                "physical_change_accepted": attempt.physical_change_accepted,
                "nfev": attempt.nfev,
                "maximum_scaled_residual": attempt.maximum_scaled_residual,
                "maximum_log_surface_density_change": (
                    attempt.maximum_log_surface_density_change
                ),
                "maximum_log_temperature_change": (
                    attempt.maximum_log_temperature_change
                ),
                "maximum_relative_thickness_change": (
                    attempt.maximum_relative_thickness_change
                ),
                "message": attempt.message,
            }
            for attempt in result.attempts
        ],
        "current_state": {
            "maximum_H_over_R": _maximum_location(
                np.asarray(old.vertical.H) / grid.centers,
                grid.centers,
                potential.r_g,
            ),
            "maximum_log_surface_density_change_from_reference": (
                _maximum_location(
                    np.abs(np.log(old.surface_density / initial.surface_density)),
                    grid.centers,
                    potential.r_g,
                )
            ),
            "maximum_log_temperature_change_from_reference": (
                _maximum_location(
                    np.abs(np.log(old.temperature / initial.temperature)),
                    grid.centers,
                    potential.r_g,
                )
            ),
            "minimum_temperature": float(np.min(old.temperature)),
            "inner_characteristic": {
                "radial_velocity": current_characteristic.radial_velocity,
                "effective_sound_speed": (
                    current_characteristic.effective_sound_speed
                ),
                "radial_mach_number": current_characteristic.radial_mach_number,
                "eigenvalues": list(current_characteristic.eigenvalues),
                "incoming_characteristics": (
                    current_characteristic.incoming_characteristics
                ),
            },
            "initial_inner_characteristic": {
                "radial_velocity": initial_characteristic.radial_velocity,
                "effective_sound_speed": (
                    initial_characteristic.effective_sound_speed
                ),
                "radial_mach_number": initial_characteristic.radial_mach_number,
                "eigenvalues": list(initial_characteristic.eigenvalues),
                "incoming_characteristics": (
                    initial_characteristic.incoming_characteristics
                ),
            },
        },
    }
    if result.accepted:
        new = recover_global_primitives(
            grid,
            result.state,
            mass,
            specific_mechanical_energy_correction=correction,
        )
        old_thickness = np.asarray(old.vertical.H) / grid.centers
        new_thickness = np.asarray(new.vertical.H) / grid.centers
        report["accepted_step_locations"] = {
            "absolute_log_surface_density_change": _maximum_location(
                np.abs(np.log(new.surface_density / old.surface_density)),
                grid.centers,
                potential.r_g,
            ),
            "absolute_log_temperature_change": _maximum_location(
                np.abs(np.log(new.temperature / old.temperature)),
                grid.centers,
                potential.r_g,
            ),
            "relative_thickness_change": _maximum_location(
                np.abs(new_thickness - old_thickness)
                / np.maximum(old_thickness, 1.0e-300),
                grid.centers,
                potential.r_g,
            ),
        }
        report["accepted_step"] = {
            "dt_used_over_loading_time": result.dt_used / loading_time,
            "dt_next_over_loading_time": result.dt_next / loading_time,
            "maximum_scaled_residual": result.step.maximum_scaled_residual,
            "maximum_storage_scaled_ledger_defect": (
                result.step.maximum_storage_scaled_ledger_defect
            ),
            "inner_mass_flux_over_supply": (
                result.step.profile.face_fluxes.mass[0] / stream_rate
            ),
            "outer_mass_flux_over_supply": (
                result.step.profile.face_fluxes.mass[-1] / stream_rate
            ),
        }
        projection = result.step.profile.inner_characteristic_projection
        if projection is not None:
            report["accepted_step"]["inner_characteristic_projection"] = {
                "incoming_amplitude_before": (
                    projection.incoming_amplitude_before
                ),
                "incoming_amplitude_after": projection.incoming_amplitude_after,
                "outgoing_amplitude_before": (
                    projection.outgoing_amplitude_before
                ),
                "outgoing_amplitude_after": (
                    projection.outgoing_amplitude_after
                ),
                "projected_radial_velocity": (
                    projection.projected_radial_velocity
                ),
                "projected_integrated_pressure": (
                    projection.projected_integrated_pressure
                ),
                "projected_temperature": projection.projected_temperature,
            }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
