"""Build and gate a fresh low-mass global finite-volume startup state."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GasRadiationHillRocheNozzleProvider,
    GlobalAdaptiveRestart,
    GlobalCellSources,
    PaczynskiWiitaPotential,
    advance_global_backward_euler,
    construct_global_constant_pressure_startup,
    evaluate_global_rusanov_profile,
    fiducial_hill_roche_nozzle_geometry,
    global_compact_stream_cell_sources,
    global_conservative_rhs,
    global_inner_characteristic_audit,
    global_roche_closure_diagnostic,
    make_global_mechanical_energy_reference,
    predict_global_explicit_euler_state,
    recover_global_primitives,
    save_global_adaptive_restart,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_global_roche_adaptive_preflight import _git_metadata


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/global_fresh_low_mass_startup.json"
CHECKPOINTS = ROOT / "outputs/checkpoints/global_fresh_low_mass_startup"
MESHES = (64, 96)
INNER_RADIUS_RG = 4.5
OUTER_RADIUS_RG = 335.0
ASPECT_RATIO = 0.05
MINIMUM_SCATTERING_DEPTH = 10.0
STREAM_OVER_EDDINGTON = 5.0
FIRST_DT_LOADING_FRACTION = 1.0e-8
MAXIMUM_NFEV = 200
CHARACTERISTIC_CACHE_SIZE = 32
HOLD_TARGET_LOADING_FRACTION = 2.0e-7
HOLD_STEPS = 20
MAXIMUM_PHYSICAL_CHANGE = 0.02


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint-directory", type=Path, default=CHECKPOINTS)
    parser.add_argument("--maximum-nfev", type=int, default=MAXIMUM_NFEV)
    parser.add_argument(
        "--skip-first-step",
        action="store_true",
        help="construct and audit both meshes without running the N64 gate",
    )
    parser.add_argument(
        "--run-matched-holds",
        action="store_true",
        help="run fixed-history source-on/source-off holds after the N64 gate",
    )
    return parser.parse_args()


def _physical_source_and_roche(grid, mass: float):
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = STREAM_OVER_EDDINGTON * eddington_mdot(mass)
    circularization_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(circularization_radius))
    stream_energy = float(
        potential.phi(circularization_radius)
        + 0.5 * (stream_l / circularization_radius) ** 2
    )
    stream = global_compact_stream_cell_sources(
        grid,
        stream_rate,
        center=240.0 * potential.r_g,
        log_width=0.08,
        specific_radial_velocity=0.0,
        specific_angular_momentum=stream_l,
        specific_total_energy=stream_energy,
    )
    provider = GasRadiationHillRocheNozzleProvider(
        fiducial_hill_roche_nozzle_geometry(),
        transverse_quadrature_zones=32,
    )
    return stream, stream_rate, provider


def _relative_radial_balance(profile, rhs) -> float:
    scale = max(
        float(np.max(np.abs(profile.cell_sources.radial_momentum))),
        1.0,
    )
    return float(np.max(np.abs(rhs.radial_momentum)) / scale)


def _state_changes(old, new, grid) -> dict[str, float]:
    old_h = np.asarray(old.vertical.H, dtype=float) / grid.centers
    new_h = np.asarray(new.vertical.H, dtype=float) / grid.centers
    return {
        "maximum_log_surface_density_change": float(
            np.max(np.abs(np.log(new.surface_density / old.surface_density)))
        ),
        "maximum_log_temperature_change": float(
            np.max(np.abs(np.log(new.temperature / old.temperature)))
        ),
        "maximum_relative_thickness_change": float(
            np.max(np.abs(new_h - old_h) / old_h)
        ),
    }


def _save_initial_restart(
    path: Path,
    grid,
    state,
    correction,
    dt: float,
    *,
    n_cells: int,
) -> None:
    mechanical = make_global_mechanical_energy_reference(
        grid,
        correction,
        state,
        provenance={
            "case": f"global-fresh-low-mass-startup-N{n_cells}",
            "reference": "constant-Pi static Keplerian initial datum",
        },
    )
    restart = GlobalAdaptiveRestart(
        state=state,
        reference_state=state,
        mechanical_reference=mechanical,
        elapsed_time=0.0,
        dt_next=dt,
        accepted_steps=0,
        rejected_attempts=0,
        provenance={
            "case": f"global-fresh-low-mass-startup-N{n_cells}",
            "n_cells": n_cells,
            "inner_radius_rg": INNER_RADIUS_RG,
            "outer_radius_rg": OUTER_RADIUS_RG,
            "aspect_ratio": ASPECT_RATIO,
            "minimum_scattering_depth": MINIMUM_SCATTERING_DEPTH,
            "source_enabled": True,
            "git": _git_metadata(),
        },
    )
    save_global_adaptive_restart(path, grid, restart)


def _profile(
    grid,
    state,
    mass: float,
    reference_state,
    correction,
    sources,
    provider,
    *,
    alpha: float,
):
    return evaluate_global_rusanov_profile(
        grid,
        state,
        mass,
        reference_state=reference_state,
        boundary_mode="characteristic_inner_roche_outer",
        alpha=alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=True,
        external_sources=sources,
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=correction,
    )


def _run_fixed_hold(
    prepared_case,
    *,
    n_cells: int,
    label: str,
    sources,
    target_time: float,
    maximum_nfev: int,
    alpha: float,
    checkpoint_directory: Path,
):
    (
        grid,
        initial,
        correction,
        _stream,
        stream_rate,
        provider,
        _predictor,
    ) = prepared_case
    dt = target_time / HOLD_STEPS
    state = initial
    step_rows = []
    accepted = True
    final_profile = None
    for step_index in range(HOLD_STEPS):
        old_primitives = recover_global_primitives(
            grid,
            state,
            FiducialParams().M2_g,
            specific_mechanical_energy_correction=correction,
        )
        old_profile = _profile(
            grid,
            state,
            FiducialParams().M2_g,
            initial,
            correction,
            sources,
            provider,
            alpha=alpha,
        )
        predictor = predict_global_explicit_euler_state(
            state, dt, old_profile
        )
        step = advance_global_backward_euler(
            grid,
            state,
            FiducialParams().M2_g,
            dt,
            alpha=alpha,
            reference_state=initial,
            boundary_mode="characteristic_inner_roche_outer",
            stress_boundary_mode="outer_zero_torque",
            include_radiative_cooling=True,
            include_vertical_column_work=True,
            external_sources=sources,
            jacobian_mode="sparse_forward",
            outer_overflow_provider=provider,
            specific_mechanical_energy_correction=correction,
            inner_characteristic_cache_size=CHARACTERISTIC_CACHE_SIZE,
            initial_guess_state=predictor,
            max_nfev=maximum_nfev,
        )
        row = {
            "step": step_index + 1,
            "accepted_by_equations": step.accepted,
            "nfev": step.nfev,
            "maximum_scaled_residual": step.maximum_scaled_residual,
            "maximum_storage_scaled_ledger_defect": (
                step.maximum_storage_scaled_ledger_defect
            ),
            "message": step.message,
        }
        if not step.accepted:
            accepted = False
            step_rows.append(row)
            final_profile = step.profile
            break
        new_primitives = recover_global_primitives(
            grid,
            step.state,
            FiducialParams().M2_g,
            specific_mechanical_energy_correction=correction,
        )
        changes = _state_changes(old_primitives, new_primitives, grid)
        physical_accepted = bool(
            max(changes.values()) <= MAXIMUM_PHYSICAL_CHANGE
        )
        row.update(changes)
        row["accepted_by_physical_change"] = physical_accepted
        step_rows.append(row)
        final_profile = step.profile
        if not physical_accepted:
            accepted = False
            break
        state = step.state

    elapsed_time = dt * sum(
        bool(row.get("accepted_by_equations"))
        and bool(row.get("accepted_by_physical_change"))
        for row in step_rows
    )
    if final_profile is None:
        raise RuntimeError("fixed hold produced no profile")
    final_primitives = recover_global_primitives(
        grid,
        state,
        FiducialParams().M2_g,
        specific_mechanical_energy_correction=correction,
    )
    characteristic = global_inner_characteristic_audit(final_primitives)
    if final_profile.outer_roche_boundary is None:
        raise RuntimeError("fixed hold lacks a Roche audit")
    roche = global_roche_closure_diagnostic(
        final_profile.outer_roche_boundary,
        provider,
        mass_flux_scale=stream_rate,
    )
    h_over_r = np.asarray(final_primitives.vertical.H) / grid.centers
    accepted = bool(
        accepted
        and len(step_rows) == HOLD_STEPS
        and characteristic.incoming_characteristics == 1
        and roche.channel_state == "closed"
    )
    checkpoint_path = checkpoint_directory / f"{label}_N{n_cells}.npz"
    mechanical = make_global_mechanical_energy_reference(
        grid,
        correction,
        initial,
        provenance={
            "case": f"global-fresh-startup-{label}-N{n_cells}",
            "reference": "constant-Pi static Keplerian initial datum",
        },
    )
    save_global_adaptive_restart(
        checkpoint_path,
        grid,
        GlobalAdaptiveRestart(
            state=state,
            reference_state=initial,
            mechanical_reference=mechanical,
            elapsed_time=elapsed_time,
            dt_next=dt,
            accepted_steps=sum(
                bool(row.get("accepted_by_equations"))
                and bool(row.get("accepted_by_physical_change"))
                for row in step_rows
            ),
            rejected_attempts=int(not accepted),
            provenance={
                "case": f"global-fresh-startup-{label}-N{n_cells}",
                "source_enabled": label == "source_on",
                "fixed_timestep_seconds": dt,
                "target_time_seconds": target_time,
                "git": _git_metadata(),
            },
        ),
    )
    return {
        "label": label,
        "n_cells": n_cells,
        "accepted": accepted,
        "target_time_seconds": target_time,
        "elapsed_time_seconds": elapsed_time,
        "fixed_timestep_seconds": dt,
        "accepted_steps": sum(
            bool(row.get("accepted_by_equations"))
            and bool(row.get("accepted_by_physical_change"))
            for row in step_rows
        ),
        "maximum_nfev": max(row["nfev"] for row in step_rows),
        "maximum_scaled_residual": max(
            row["maximum_scaled_residual"] for row in step_rows
        ),
        "maximum_storage_scaled_ledger_defect": max(
            row["maximum_storage_scaled_ledger_defect"]
            for row in step_rows
        ),
        "maximum_H_over_R": float(np.max(h_over_r)),
        "inner_mass_flux_over_stream": float(
            final_profile.face_fluxes.mass[0] / stream_rate
        ),
        "inner_angular_momentum_flux": float(
            final_profile.face_fluxes.angular_momentum[0]
        ),
        "inner_total_energy_flux": float(
            final_profile.face_fluxes.total_energy[0]
        ),
        "disk_mass_change_over_injected_increment": float(
            (np.sum(state.mass) - np.sum(initial.mass))
            / (stream_rate * target_time)
        ),
        "inner_characteristic": asdict(characteristic),
        "roche_closure": asdict(roche),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "step_rows": step_rows,
    }


def main() -> None:
    arguments = _arguments()
    if arguments.maximum_nfev < 1:
        raise ValueError("maximum nfev must be positive")
    output_path = arguments.output
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    checkpoint_directory = arguments.checkpoint_directory
    if not checkpoint_directory.is_absolute():
        checkpoint_directory = ROOT / checkpoint_directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    fiducial = FiducialParams()
    mass = fiducial.M2_g
    potential = PaczynskiWiitaPotential(mass)
    prepared = {}
    mesh_rows = []
    for n_cells in MESHES:
        grid, state, correction, construction = (
            construct_global_constant_pressure_startup(
                mass,
                n_cells,
                inner_radius=INNER_RADIUS_RG * potential.r_g,
                outer_radius=OUTER_RADIUS_RG * potential.r_g,
                aspect_ratio=ASPECT_RATIO,
                minimum_scattering_optical_depth=(
                    MINIMUM_SCATTERING_DEPTH
                ),
                viscous_drift_alpha=fiducial.alpha_cool,
            )
        )
        stream, stream_rate, provider = _physical_source_and_roche(grid, mass)
        loading_time = float(np.sum(state.mass) / stream_rate)
        dt = FIRST_DT_LOADING_FRACTION * loading_time
        primitives = recover_global_primitives(
            grid,
            state,
            mass,
            specific_mechanical_energy_correction=correction,
        )
        equilibrium = evaluate_global_rusanov_profile(
            grid,
            state,
            mass,
            reference_state=state,
            specific_mechanical_energy_correction=correction,
        )
        equilibrium_rhs = global_conservative_rhs(
            equilibrium.face_fluxes, equilibrium.cell_sources
        )
        physical = evaluate_global_rusanov_profile(
            grid,
            state,
            mass,
            reference_state=state,
            boundary_mode="characteristic_inner_roche_outer",
            alpha=fiducial.alpha_cool,
            stress_boundary_mode="outer_zero_torque",
            include_radiative_cooling=True,
            include_vertical_column_work=True,
            external_sources=stream,
            outer_overflow_provider=provider,
            specific_mechanical_energy_correction=correction,
        )
        predictor = predict_global_explicit_euler_state(state, dt, physical)
        predictor_primitives = recover_global_primitives(
            grid,
            predictor,
            mass,
            specific_mechanical_energy_correction=correction,
        )
        if physical.outer_roche_boundary is None:
            raise RuntimeError("fresh startup lacks a Roche boundary audit")
        roche = global_roche_closure_diagnostic(
            physical.outer_roche_boundary,
            provider,
            mass_flux_scale=stream_rate,
        )
        characteristic = global_inner_characteristic_audit(primitives)
        row = {
            "n_cells": n_cells,
            "construction": asdict(construction),
            "loading_time_seconds": loading_time,
            "first_dt_seconds": dt,
            "inviscid_radial_balance_relative_defect": (
                _relative_radial_balance(equilibrium, equilibrium_rhs)
            ),
            "inner_characteristic": asdict(characteristic),
            "roche_closure": asdict(roche),
            "predictor_changes": _state_changes(
                primitives, predictor_primitives, grid
            ),
            "source_mass_normalization_relative_defect": float(
                abs(np.sum(stream.mass) / stream_rate - 1.0)
            ),
            "initial_checkpoint": str(
                (
                    checkpoint_directory / f"initial_N{n_cells}.npz"
                ).relative_to(ROOT)
            ),
        }
        _save_initial_restart(
            checkpoint_directory / f"initial_N{n_cells}.npz",
            grid,
            state,
            correction,
            dt,
            n_cells=n_cells,
        )
        prepared[n_cells] = (
            grid,
            state,
            correction,
            stream,
            stream_rate,
            provider,
            predictor,
        )
        mesh_rows.append(row)

    mass_spread = abs(
        mesh_rows[1]["construction"]["total_mass"]
        / mesh_rows[0]["construction"]["total_mass"]
        - 1.0
    )
    preflight_accepted = bool(
        all(
            # Primitive recovery uses the production 1e-12 scalar root
            # tolerance; its pressure reconstruction sets this audit floor.
            row["inviscid_radial_balance_relative_defect"] <= 2.0e-9
            and row["inner_characteristic"]["incoming_characteristics"] == 1
            and row["roche_closure"]["channel_state"] == "closed"
            and row["source_mass_normalization_relative_defect"] <= 2.0e-13
            for row in mesh_rows
        )
        and mass_spread <= 0.01
    )
    first_step = None
    if preflight_accepted and not arguments.skip_first_step:
        (
            grid,
            state,
            correction,
            stream,
            stream_rate,
            provider,
            predictor,
        ) = prepared[64]
        dt = FIRST_DT_LOADING_FRACTION * float(np.sum(state.mass) / stream_rate)
        result = advance_global_backward_euler(
            grid,
            state,
            mass,
            dt,
            alpha=fiducial.alpha_cool,
            reference_state=state,
            boundary_mode="characteristic_inner_roche_outer",
            stress_boundary_mode="outer_zero_torque",
            include_radiative_cooling=True,
            include_vertical_column_work=True,
            external_sources=stream,
            jacobian_mode="sparse_forward",
            outer_overflow_provider=provider,
            specific_mechanical_energy_correction=correction,
            inner_characteristic_cache_size=CHARACTERISTIC_CACHE_SIZE,
            initial_guess_state=predictor,
            max_nfev=int(arguments.maximum_nfev),
        )
        first_step = {
            "accepted": result.accepted,
            "message": result.message,
            "nfev": result.nfev,
            "maximum_scaled_residual": result.maximum_scaled_residual,
            "maximum_storage_scaled_ledger_defect": (
                result.maximum_storage_scaled_ledger_defect
            ),
            "jacobian_audit": (
                None
                if result.jacobian_audit is None
                else asdict(result.jacobian_audit)
            ),
            "nonlinear_solve_audit": (
                None
                if result.nonlinear_solve_audit is None
                else asdict(result.nonlinear_solve_audit)
            ),
        }

    hold_rows = []
    hold_mesh_gate = None
    if (
        arguments.run_matched_holds
        and preflight_accepted
        and first_step is not None
        and first_step["accepted"]
    ):
        target_time = (
            HOLD_TARGET_LOADING_FRACTION
            * mesh_rows[0]["loading_time_seconds"]
        )
        for label in ("source_on", "source_off"):
            hold_rows.append(
                _run_fixed_hold(
                    prepared[64],
                    n_cells=64,
                    label=label,
                    sources=(
                        prepared[64][3]
                        if label == "source_on"
                        else GlobalCellSources.zeros(64)
                    ),
                    target_time=target_time,
                    maximum_nfev=int(arguments.maximum_nfev),
                    alpha=fiducial.alpha_cool,
                    checkpoint_directory=checkpoint_directory,
                )
            )
        n64_passed = all(row["accepted"] for row in hold_rows)
        if n64_passed:
            for label in ("source_on", "source_off"):
                hold_rows.append(
                    _run_fixed_hold(
                        prepared[96],
                        n_cells=96,
                        label=label,
                        sources=(
                            prepared[96][3]
                            if label == "source_on"
                            else GlobalCellSources.zeros(96)
                        ),
                        target_time=target_time,
                        maximum_nfev=int(arguments.maximum_nfev),
                        alpha=fiducial.alpha_cool,
                        checkpoint_directory=checkpoint_directory,
                    )
                )
        if len(hold_rows) == 4:
            source_on = {
                row["n_cells"]: row
                for row in hold_rows
                if row["label"] == "source_on"
            }
            mass_flux_spread = abs(
                source_on[96]["inner_mass_flux_over_stream"]
                - source_on[64]["inner_mass_flux_over_stream"]
            )
            thickness_spread = abs(
                source_on[96]["maximum_H_over_R"]
                / source_on[64]["maximum_H_over_R"]
                - 1.0
            )
            hold_mesh_gate = {
                "inner_mass_flux_spread_over_stream": float(
                    mass_flux_spread
                ),
                "maximum_H_over_R_relative_spread": float(
                    thickness_spread
                ),
                "mass_flux_limit": 0.01,
                "thickness_limit": 0.02,
                "accepted": bool(
                    all(row["accepted"] for row in hold_rows)
                    and mass_flux_spread <= 0.01
                    and thickness_spread <= 0.02
                ),
            }

    accepted = bool(
        preflight_accepted
        and (
            arguments.skip_first_step
            or (first_step is not None and first_step["accepted"])
        )
        and (
            not arguments.run_matched_holds
            or (
                hold_mesh_gate is not None
                and hold_mesh_gate["accepted"]
            )
        )
    )
    output = {
        "git": _git_metadata(),
        "configuration": {
            "meshes": list(MESHES),
            "inner_radius_rg": INNER_RADIUS_RG,
            "outer_radius_rg": OUTER_RADIUS_RG,
            "aspect_ratio": ASPECT_RATIO,
            "minimum_scattering_depth": MINIMUM_SCATTERING_DEPTH,
            "viscous_drift_alpha": fiducial.alpha_cool,
            "stream_over_eddington": STREAM_OVER_EDDINGTON,
            "first_dt_loading_fraction": FIRST_DT_LOADING_FRACTION,
            "maximum_nfev": int(arguments.maximum_nfev),
            "characteristic_cache_size": CHARACTERISTIC_CACHE_SIZE,
            "predictor_role": "initial guess only",
            "hold_target_loading_fraction": (
                HOLD_TARGET_LOADING_FRACTION
            ),
            "hold_steps": HOLD_STEPS,
            "maximum_physical_change_per_step": MAXIMUM_PHYSICAL_CHANGE,
        },
        "mesh_rows": mesh_rows,
        "cross_mesh_total_mass_relative_spread": float(mass_spread),
        "preflight_accepted": preflight_accepted,
        "first_step": first_step,
        "hold_rows": hold_rows,
        "hold_mesh_gate": hold_mesh_gate,
        "accepted": accepted,
    }
    output_path.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(output_path)
    print(json.dumps(output, indent=2))
    if not accepted:
        raise RuntimeError("fresh global startup failed its bounded gate")


if __name__ == "__main__":
    main()
