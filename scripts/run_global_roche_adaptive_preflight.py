"""Exercise adaptive physical Roche loading with restart after every step."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalAdaptiveRestart,
    GlobalAdaptiveStepConfig,
    advance_global_adaptive_backward_euler,
    load_global_adaptive_restart,
    make_global_mechanical_energy_reference,
    save_global_adaptive_restart,
)

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/global_roche_adaptive_preflight.json"
RESTART = ROOT / "outputs/checkpoints/global_roche_adaptive_N64.npz"
TARGET_LOADING_FRACTION = 5.0e-7
INITIAL_DT_LOADING_FRACTION = 1.0e-7


def _finite_or_none(value: float):
    return float(value) if np.isfinite(value) else None


def _attempt_record(attempt) -> dict:
    return {
        "dt": attempt.dt,
        "nonlinear_accepted": attempt.nonlinear_accepted,
        "physical_change_accepted": attempt.physical_change_accepted,
        "nfev": attempt.nfev,
        "maximum_scaled_residual": attempt.maximum_scaled_residual,
        "maximum_log_surface_density_change": _finite_or_none(
            attempt.maximum_log_surface_density_change
        ),
        "maximum_log_temperature_change": _finite_or_none(
            attempt.maximum_log_temperature_change
        ),
        "maximum_relative_thickness_change": _finite_or_none(
            attempt.maximum_relative_thickness_change
        ),
        "message": attempt.message,
    }


def main() -> None:
    context, evaluation = _canonical_open_evaluation()
    grid, initial, correction, stream, stream_rate, provider = _prepared_case(
        context, evaluation, 64
    )
    mass = context.base.inner_params.M2_g
    loading_time = float(np.sum(initial.mass) / stream_rate)
    mechanical = make_global_mechanical_energy_reference(
        grid,
        correction,
        initial,
        provenance={
            "case": "global-roche-adaptive-N64",
            "source": "canonical coupled open control remapped conservatively",
        },
    )
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
        "reference_state": initial,
        "boundary_mode": "characteristic_inner_roche_outer",
        "stress_boundary_mode": "outer_zero_torque",
        "include_radiative_cooling": True,
        "include_vertical_column_work": True,
        "external_sources": stream,
        "jacobian_mode": "sparse_forward",
        "outer_overflow_provider": provider,
        "max_nfev": 300,
    }
    current = initial
    elapsed = 0.0
    dt_next = INITIAL_DT_LOADING_FRACTION * loading_time
    accepted_steps = 0
    rejected_attempts = 0
    records = []
    target_time = TARGET_LOADING_FRACTION * loading_time
    while elapsed < target_time and accepted_steps < 20:
        requested_dt = min(dt_next, target_time - elapsed)
        result = advance_global_adaptive_backward_euler(
            grid,
            current,
            mass,
            requested_dt,
            config,
            specific_mechanical_energy_correction=correction,
            step_options=step_options,
        )
        rejected_attempts += sum(
            not (
                attempt.nonlinear_accepted
                and attempt.physical_change_accepted
            )
            for attempt in result.attempts
        )
        records.append(
            {
                "accepted": result.accepted,
                "requested_dt_over_loading_time": requested_dt / loading_time,
                "dt_used_over_loading_time": result.dt_used / loading_time,
                "dt_next_over_loading_time": result.dt_next / loading_time,
                "attempts": [
                    _attempt_record(attempt) for attempt in result.attempts
                ],
                "roche_choked": result.step.profile.outer_roche_boundary.gate.choked,
                "roche_available_specific_energy": (
                    result.step.profile.outer_roche_boundary
                    .gate.available_specific_energy
                ),
            }
        )
        if not result.accepted:
            break
        current = result.state
        elapsed += result.dt_used
        dt_next = result.dt_next
        accepted_steps += 1
        restart = GlobalAdaptiveRestart(
            state=current,
            reference_state=initial,
            mechanical_reference=mechanical,
            elapsed_time=elapsed,
            dt_next=dt_next,
            accepted_steps=accepted_steps,
            rejected_attempts=rejected_attempts,
            provenance={
                "case": "global-roche-adaptive-N64",
                "target_loading_fraction": TARGET_LOADING_FRACTION,
            },
        )
        save_global_adaptive_restart(RESTART, grid, restart)
        loaded_grid, loaded = load_global_adaptive_restart(RESTART, grid=grid)
        if not np.array_equal(loaded_grid.edges, grid.edges):
            raise RuntimeError("adaptive restart changed the grid")
        for name in (
            "mass",
            "radial_momentum",
            "angular_momentum",
            "total_energy",
        ):
            if not np.array_equal(
                getattr(loaded.state, name), getattr(current, name)
            ):
                raise RuntimeError("adaptive restart changed the accepted state")
        current = loaded.state
        elapsed = loaded.elapsed_time
        dt_next = loaded.dt_next
        correction = loaded.mechanical_reference.specific_offset
    report = {
        "n_cells": 64,
        "target_loading_fraction": TARGET_LOADING_FRACTION,
        "elapsed_loading_fraction": elapsed / loading_time,
        "accepted_steps": accepted_steps,
        "rejected_attempts": rejected_attempts,
        "target_reached": elapsed >= target_time,
        "restart_after_every_accepted_step": True,
        "disk_mass_relative_change": float(
            np.sum(current.mass) / np.sum(initial.mass) - 1.0
        ),
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
