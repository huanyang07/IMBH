"""Exercise adaptive physical Roche loading with restart after every step."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalAdaptiveRestart,
    GlobalAdaptiveStepConfig,
    GlobalCellSources,
    advance_global_adaptive_backward_euler,
    evaluate_global_rusanov_profile,
    global_fixed_radius_diagnostics,
    global_inner_characteristic_audit,
    global_outer_characteristic_audit,
    global_roche_closure_diagnostic,
    global_sonic_resolution_diagnostic,
    load_global_adaptive_restart,
    make_global_mechanical_energy_reference,
    recover_global_primitives,
    save_global_adaptive_milestone,
    save_global_adaptive_restart,
)

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/global_roche_adaptive_preflight.json"
RESTART = ROOT / "outputs/checkpoints/global_roche_adaptive_N64.npz"
TARGET_LOADING_FRACTION = 5.0e-7
INITIAL_DT_LOADING_FRACTION = 1.0e-7
FIXED_DIAGNOSTIC_RADII_RG = (4.65, 4.75, 5.00)


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
        "controller": (
            None if attempt.controller is None else asdict(attempt.controller)
        ),
        "nonlinear_solve_audit": (
            None
            if attempt.nonlinear_solve_audit is None
            else asdict(attempt.nonlinear_solve_audit)
        ),
        "message": attempt.message,
    }


def _git_metadata() -> dict:
    def command(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", *arguments),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None

    return {
        "full_sha": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
    }


def _conserved_flux_record(face_fluxes, face_index: int, stream) -> dict:
    source_mass = float(np.sum(stream.mass))
    source_angular = float(np.sum(stream.angular_momentum))
    source_energy = float(np.sum(stream.total_energy))

    def ratio(value: float, scale: float):
        return None if scale == 0.0 else float(value / scale)

    return {
        "mass": float(face_fluxes.mass[face_index]),
        "angular_momentum": float(
            face_fluxes.angular_momentum[face_index]
        ),
        "total_energy": float(face_fluxes.total_energy[face_index]),
        "mass_over_source": ratio(
            float(face_fluxes.mass[face_index]), source_mass
        ),
        "angular_momentum_over_source": ratio(
            float(face_fluxes.angular_momentum[face_index]), source_angular
        ),
        "total_energy_over_source": ratio(
            float(face_fluxes.total_energy[face_index]), source_energy
        ),
    }


def _maximum_accepted_ledger_defect(records: list[dict]) -> float | None:
    values = [
        record["maximum_storage_scaled_ledger_defect"]
        for record in records
        if record["accepted"]
    ]
    return max(values) if values else None


def _target_time_tolerance(target_time: float) -> float:
    """Return a roundoff-only tolerance for an accumulated target time."""

    return 64.0 * np.finfo(float).eps * max(abs(float(target_time)), 1.0)


def _final_step_config(
    config: GlobalAdaptiveStepConfig, requested_dt: float
) -> GlobalAdaptiveStepConfig:
    """Allow an exact final landing below the ordinary controller minimum."""

    if requested_dt <= 0.0:
        raise ValueError("requested final timestep must be positive")
    if requested_dt >= config.minimum_dt:
        return config
    return replace(config, minimum_dt=float(requested_dt))


def run_adaptive_campaign(
    context,
    evaluation,
    *,
    n_cells: int,
    target_loading_fraction: float,
    initial_dt_loading_fraction: float,
    restart_path: Path,
    resume: bool = False,
    maximum_accepted_steps: int = 20,
    inner_radius_rg: float | None = None,
    maximum_nfev: int = 300,
    minimum_dt_loading_fraction: float = 1.0e-9,
    resume_dt_cap_loading_fraction: float | None = None,
    reference_loading_time_seconds: float | None = None,
    milestone_directory: Path | None = None,
    milestone_case: str | None = None,
    source_enabled: bool = True,
    prepared_case=None,
    inner_boundary_mode: str | None = None,
) -> dict:
    """Run or resume one mesh while checkpointing every accepted state."""

    if int(maximum_nfev) != maximum_nfev or maximum_nfev < 1:
        raise ValueError("maximum_nfev must be a positive integer")
    if minimum_dt_loading_fraction <= 0.0:
        raise ValueError("minimum_dt_loading_fraction must be positive")
    if (
        resume_dt_cap_loading_fraction is not None
        and resume_dt_cap_loading_fraction <= 0.0
    ):
        raise ValueError("resume_dt_cap_loading_fraction must be positive")

    if prepared_case is None:
        grid, initial, correction, stream, stream_rate, provider = (
            _prepared_case(
                context,
                evaluation,
                n_cells,
                inner_radius_rg=inner_radius_rg,
            )
        )
    else:
        grid, initial, correction, stream, stream_rate, provider = prepared_case
        if grid.centers.size != int(n_cells):
            raise ValueError("prepared adaptive case has the wrong mesh size")
    evolution_sources = (
        stream
        if source_enabled
        else GlobalCellSources.zeros(grid.centers.size)
    )
    mass = context.base.inner_params.M2_g
    loading_time = float(np.sum(initial.mass) / stream_rate)
    if reference_loading_time_seconds is None:
        reference_loading_time_seconds = loading_time
    if (
        not np.isfinite(reference_loading_time_seconds)
        or reference_loading_time_seconds <= 0.0
    ):
        raise ValueError("reference loading time must be positive and finite")
    git_metadata = _git_metadata()
    mechanical = make_global_mechanical_energy_reference(
        grid,
        correction,
        initial,
        provenance={
            "case": f"global-roche-adaptive-N{n_cells}",
            "source": "canonical coupled open control remapped conservatively",
        },
    )
    config = GlobalAdaptiveStepConfig(
        minimum_dt=minimum_dt_loading_fraction * loading_time,
        maximum_dt=5.0e-7 * loading_time,
        shrink_factor=0.5,
        growth_factor=1.5,
        maximum_retries=6,
        easy_nfev=20,
        maximum_log_surface_density_change=0.02,
        maximum_log_temperature_change=0.02,
        maximum_relative_thickness_change=0.02,
    )
    boundary_mode = (
        inner_boundary_mode
        if inner_boundary_mode is not None
        else (
            "characteristic_inner_roche_outer"
            if inner_radius_rg is None
            else "roche_outer"
        )
    )
    step_options = {
        "alpha": context.base.alpha,
        "reference_state": initial,
        "boundary_mode": boundary_mode,
        "stress_boundary_mode": "outer_zero_torque",
        "include_radiative_cooling": True,
        "include_vertical_column_work": True,
        "external_sources": evolution_sources,
        "jacobian_mode": "sparse_forward",
        "outer_overflow_provider": provider,
        "max_nfev": int(maximum_nfev),
    }
    restart_provenance = {
        "case": f"global-roche-adaptive-N{n_cells}",
        "n_cells": n_cells,
        "target_loading_fraction": target_loading_fraction,
        "inner_radius_rg": inner_radius_rg,
        "git": git_metadata,
        "mesh_loading_time_seconds": loading_time,
        "reference_loading_time_seconds": reference_loading_time_seconds,
        "jacobian_backend": step_options["jacobian_mode"],
        "maximum_nfev": int(maximum_nfev),
        "minimum_dt_loading_fraction": minimum_dt_loading_fraction,
        "boundary_mode": boundary_mode,
        "stress_boundary_mode": step_options["stress_boundary_mode"],
        "include_radiative_cooling": True,
        "include_vertical_column_work": True,
        "source_enabled": bool(source_enabled),
        "adaptive_physical_change_limits": {
            "log_surface_density": (
                config.maximum_log_surface_density_change
            ),
            "log_temperature": config.maximum_log_temperature_change,
            "relative_thickness": (
                config.maximum_relative_thickness_change
            ),
        },
    }
    if resume:
        if not restart_path.exists():
            raise ValueError("requested adaptive restart does not exist")
        _loaded_grid, loaded = load_global_adaptive_restart(
            restart_path, grid=grid
        )
        for name in (
            "mass",
            "radial_momentum",
            "angular_momentum",
            "total_energy",
        ):
            if not np.array_equal(
                getattr(loaded.reference_state, name), getattr(initial, name)
            ):
                raise ValueError(
                    "adaptive restart reference differs from canonical mapping"
                )
        current = loaded.state
        initial = loaded.reference_state
        correction = loaded.mechanical_reference.specific_offset
        mechanical = loaded.mechanical_reference
        if bool(loaded.provenance.get("source_enabled", True)) != bool(
            source_enabled
        ):
            raise ValueError("adaptive restart source mode differs from request")
        elapsed = loaded.elapsed_time
        dt_next = loaded.dt_next
        if resume_dt_cap_loading_fraction is not None:
            dt_next = min(
                dt_next,
                resume_dt_cap_loading_fraction * loading_time,
            )
        accepted_steps = loaded.accepted_steps
        rejected_attempts = loaded.rejected_attempts
        prior_attempt_history = list(
            loaded.provenance.get("attempt_history", [])
        )
    else:
        current = initial
        elapsed = 0.0
        dt_next = initial_dt_loading_fraction * loading_time
        accepted_steps = 0
        rejected_attempts = 0
        prior_attempt_history = []
    starting_steps = accepted_steps
    records = []
    target_time = target_loading_fraction * loading_time
    target_tolerance = _target_time_tolerance(target_time)
    while (
        elapsed < target_time - target_tolerance
        and accepted_steps - starting_steps < maximum_accepted_steps
    ):
        requested_dt = min(dt_next, target_time - elapsed)
        step_config = _final_step_config(config, requested_dt)
        result = advance_global_adaptive_backward_euler(
            grid,
            current,
            mass,
            requested_dt,
            step_config,
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
                "roche_closure": asdict(
                    global_roche_closure_diagnostic(
                        result.step.profile.outer_roche_boundary,
                        provider,
                        mass_flux_scale=stream_rate,
                    )
                ),
                "inner_mass_flux_over_supply": (
                    result.step.profile.face_fluxes.mass[0] / stream_rate
                ),
                "inner_conserved_fluxes": _conserved_flux_record(
                    result.step.profile.face_fluxes, 0, stream
                ),
                "outer_mass_flux_over_supply": (
                    result.step.profile.face_fluxes.mass[-1] / stream_rate
                ),
                "outer_conserved_fluxes": _conserved_flux_record(
                    result.step.profile.face_fluxes, -1, stream
                ),
                "maximum_storage_scaled_ledger_defect": (
                    result.step.maximum_storage_scaled_ledger_defect
                ),
                "ledger_defects": result.step.ledger.defects,
                "ledger_relative_defects": (
                    result.step.ledger.relative_defects
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
                **restart_provenance,
                "attempt_history": prior_attempt_history + records,
            },
        )
        save_global_adaptive_restart(restart_path, grid, restart)
        loaded_grid, loaded = load_global_adaptive_restart(
            restart_path, grid=grid
        )
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
    target_time_roundoff_snapped = bool(
        abs(elapsed - target_time) <= target_tolerance
    )
    if target_time_roundoff_snapped:
        elapsed = target_time
    final = recover_global_primitives(
        grid,
        current,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    final_profile = evaluate_global_rusanov_profile(
        grid,
        current,
        mass,
        reference_state=initial,
        boundary_mode=boundary_mode,
        alpha=context.base.alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=True,
        external_sources=evolution_sources,
        primitives=final,
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=correction,
    )
    final_boundary = final_profile.outer_roche_boundary
    if final_boundary is None:
        raise RuntimeError("final adaptive state lacks a Roche boundary audit")
    final_inner_characteristic = global_inner_characteristic_audit(final)
    final_outer_characteristic = global_outer_characteristic_audit(final)
    reference_primitives = recover_global_primitives(
        grid,
        initial,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    reference_inner_sound = global_inner_characteristic_audit(
        reference_primitives
    ).effective_sound_speed
    inner_projection = final_profile.inner_characteristic_projection
    incoming_amplitude = (
        None
        if inner_projection is None
        else inner_projection.incoming_amplitude_before
    )
    fixed_radii = tuple(
        radius_rg * context.base.inner_params.r_g
        for radius_rg in FIXED_DIAGNOSTIC_RADII_RG
        if grid.edges[0]
        <= radius_rg * context.base.inner_params.r_g
        <= grid.edges[-1]
    )
    fixed_radius_diagnostics = global_fixed_radius_diagnostics(
        grid,
        final,
        final_profile.face_fluxes,
        mass,
        fixed_radii,
    )
    sonic_diagnostic = global_sonic_resolution_diagnostic(grid, final)
    roche_diagnostic = global_roche_closure_diagnostic(
        final_boundary,
        provider,
        mass_flux_scale=stream_rate,
    )
    final_inner_fluxes = _conserved_flux_record(
        final_profile.face_fluxes, 0, stream
    )
    final_outer_fluxes = _conserved_flux_record(
        final_profile.face_fluxes, -1, stream
    )
    full_attempt_history = prior_attempt_history + records
    report = {
        "git": git_metadata,
        "n_cells": n_cells,
        "source_enabled": bool(source_enabled),
        "target_loading_fraction": target_loading_fraction,
        "target_time_seconds": target_time,
        "elapsed_time_seconds": elapsed,
        "mesh_loading_time_seconds": loading_time,
        "reference_loading_time_seconds": reference_loading_time_seconds,
        "minimum_dt_loading_fraction": minimum_dt_loading_fraction,
        "resume_dt_cap_loading_fraction": resume_dt_cap_loading_fraction,
        "elapsed_loading_fraction": elapsed / loading_time,
        "elapsed_reference_loading_fraction": (
            elapsed / reference_loading_time_seconds
        ),
        "accepted_steps": accepted_steps,
        "accepted_steps_this_run": accepted_steps - starting_steps,
        "rejected_attempts": rejected_attempts,
        "target_reached": bool(
            elapsed >= target_time - target_tolerance
        ),
        "target_time_roundoff_snapped": target_time_roundoff_snapped,
        "restart_after_every_accepted_step": True,
        "disk_mass_relative_change": float(
            np.sum(current.mass) / np.sum(initial.mass) - 1.0
        ),
        "maximum_H_over_R": float(
            np.max(np.asarray(final.vertical.H) / grid.centers)
        ),
        "minimum_temperature": float(np.min(final.temperature)),
        "final_inner_mass_flux_over_supply": float(
            final_profile.face_fluxes.mass[0] / stream_rate
        ),
        "final_inner_conserved_fluxes": final_inner_fluxes,
        "final_outer_mass_flux_over_supply": float(
            final_profile.face_fluxes.mass[-1] / stream_rate
        ),
        "final_outer_conserved_fluxes": final_outer_fluxes,
        "final_roche_choked": final_boundary.gate.choked,
        "final_roche_available_specific_energy": (
            final_boundary.gate.available_specific_energy
        ),
        "final_inner_characteristic_incoming_amplitude": incoming_amplitude,
        "final_inner_characteristic_amplitude_over_reference_sound": (
            None
            if incoming_amplitude is None
            else abs(incoming_amplitude) / reference_inner_sound
        ),
        "final_inner_radial_mach_number": (
            final_inner_characteristic.radial_mach_number
        ),
        "final_inner_incoming_characteristics": (
            final_inner_characteristic.incoming_characteristics
        ),
        "final_outer_incoming_characteristics": (
            final_outer_characteristic.incoming_characteristics
        ),
        "fixed_radius_diagnostics": [
            asdict(diagnostic) for diagnostic in fixed_radius_diagnostics
        ],
        "sonic_resolution": asdict(sonic_diagnostic),
        "roche_closure": asdict(roche_diagnostic),
        "diagnostic_definitions": {
            "flux_orientation": "positive toward increasing radius",
            "sonic_crossing": "innermost abs(radial Mach)=1 crossing",
            "radial_gradient_length": "abs(d ln abs(v_R) / dR)^-1",
            "reference_loading_time": (
                "shared campaign reference; mesh loading time remains "
                "separately reported"
            ),
            "fixed_radius_interpolation": (
                "linear in log radius; positive primitives logarithmic and "
                "signed values linear"
            ),
        },
        "maximum_accepted_storage_scaled_ledger_defect_this_run": (
            _maximum_accepted_ledger_defect(records)
        ),
        "maximum_accepted_storage_scaled_ledger_defect_full_history": (
            _maximum_accepted_ledger_defect(full_attempt_history)
        ),
        "records": records,
    }
    if milestone_directory is not None and elapsed > 0.0:
        final_restart = GlobalAdaptiveRestart(
            state=current,
            reference_state=initial,
            mechanical_reference=mechanical,
            elapsed_time=elapsed,
            dt_next=dt_next,
            accepted_steps=accepted_steps,
            rejected_attempts=rejected_attempts,
            provenance={
                **restart_provenance,
                "attempt_history": prior_attempt_history + records,
            },
        )
        report["milestone_checkpoint"] = save_global_adaptive_milestone(
            milestone_directory,
            (
                milestone_case
                if milestone_case is not None
                else f"global-roche-adaptive-N{n_cells}"
            ),
            grid,
            final_restart,
            metadata={
                "git": git_metadata,
                "target_reached": report["target_reached"],
                "mesh_loading_time_seconds": loading_time,
                "reference_loading_time_seconds": (
                    reference_loading_time_seconds
                ),
                "elapsed_loading_fraction": report["elapsed_loading_fraction"],
                "elapsed_reference_loading_fraction": (
                    report["elapsed_reference_loading_fraction"]
                ),
                "inner_conserved_fluxes": final_inner_fluxes,
                "outer_conserved_fluxes": final_outer_fluxes,
                "sonic_resolution": report["sonic_resolution"],
                "roche_closure": report["roche_closure"],
                "maximum_accepted_storage_scaled_ledger_defect_this_run": (
                    report[
                        "maximum_accepted_storage_scaled_ledger_defect_this_run"
                    ]
                ),
                "maximum_accepted_storage_scaled_ledger_defect_full_history": (
                    report[
                        "maximum_accepted_storage_scaled_ledger_defect_full_history"
                    ]
                ),
                "last_step_ledgers": (
                    None
                    if not full_attempt_history
                    else {
                        "defects": full_attempt_history[-1]["ledger_defects"],
                        "relative_defects": full_attempt_history[-1][
                            "ledger_relative_defects"
                        ],
                    }
                ),
            },
        )
    return report


def main() -> None:
    context, evaluation = _canonical_open_evaluation()
    report = run_adaptive_campaign(
        context,
        evaluation,
        n_cells=64,
        target_loading_fraction=TARGET_LOADING_FRACTION,
        initial_dt_loading_fraction=INITIAL_DT_LOADING_FRACTION,
        restart_path=RESTART,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
