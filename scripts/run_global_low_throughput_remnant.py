"""Build a fresh low-throughput transonic remnant on global FV meshes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GasRadiationHillRocheNozzleProvider,
    GlobalAdaptiveRestart,
    PaczynskiWiitaPotential,
    TransonicSlimParams,
    conservatively_map_global_profile,
    continue_transonic_supersonic_plunge,
    evaluate_global_rusanov_profile,
    fiducial_hill_roche_nozzle_geometry,
    global_compact_stream_cell_sources,
    global_effective_sound_speed,
    global_roche_closure_diagnostic,
    make_global_mechanical_energy_reference,
    remap_profile_to_new_sonic_grid,
    recover_global_primitives,
    save_global_adaptive_restart,
    solve_global_inner_steady_projection,
    solve_low_mdot_transonic_homotopy,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_global_roche_adaptive_preflight import _git_metadata


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/global_low_throughput_remnant.json"
CHECKPOINTS = ROOT / "outputs/checkpoints/global_low_throughput_remnant"
TRANSONIC_RATIOS = (1.0e-3, 3.0e-3, 1.0e-2, 2.5e-2)
TRANSONIC_NODES = 64
OUTER_RADIUS_RG = 335.0
INNER_RADIUS_RG = 4.5
GLOBAL_MESHES = (64, 96)
PHYSICAL_STREAM_OVER_EDDINGTON = 5.0
INNER_THROUGHPUT_LIMIT = 1.0e-2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint-directory", type=Path, default=CHECKPOINTS)
    parser.add_argument("--maximum-nfev-per-stage", type=int, default=900)
    parser.add_argument("--final-maximum-nfev", type=int, default=1600)
    parser.add_argument("--reuse-transonic-profile", action="store_true")
    return parser.parse_args()


def _usable(status) -> bool:
    return bool(
        status.physically_valid
        or (
            status.optimizer_acceptable
            and status.equations_converged
            and status.sonic_regular
            and status.active_bounds_clear
            and status.outer_thin
        )
    )


def solve_low_throughput_transonic(
    *,
    maximum_nfev_per_stage: int,
    final_maximum_nfev: int,
):
    """Solve the fixed low-rate sequence and return the target profile."""

    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    previous = None
    rows = []
    target = None
    target_params = None
    for ratio in TRANSONIC_RATIOS:
        params = TransonicSlimParams(
            M2_g=fiducial.M2_g,
            Mdot_g_s=ratio * mdot_edd,
            alpha=fiducial.alpha_cool,
            n_nodes=TRANSONIC_NODES,
            R_out_rg=OUTER_RADIUS_RG,
            max_nfev=final_maximum_nfev,
            residual_tol=3.0e-4,
        )
        guess = (
            None
            if previous is None
            else remap_profile_to_new_sonic_grid(previous, params)
        )
        result = solve_low_mdot_transonic_homotopy(
            params,
            initial_guess=guess,
            max_nfev_per_stage=maximum_nfev_per_stage,
            final_max_nfev=final_maximum_nfev,
        )
        final = result.final_result
        status = final.status
        usable = _usable(status)
        rows.append(
            {
                "mdot_over_eddington": ratio,
                "usable": usable,
                "physically_valid": status.physically_valid,
                "equations_converged": status.equations_converged,
                "sonic_regular": status.sonic_regular,
                "active_bounds_clear": status.active_bounds_clear,
                "outer_thin": status.outer_thin,
                "maximum_residual": final.max_residual,
                "sonic_radius_rg": (
                    final.profile.sonic_radius / params.r_g
                ),
                "maximum_H_over_R": float(
                    np.max(final.profile.H_over_R)
                ),
                "nfev": final.nfev,
                "message": final.message,
            }
        )
        if not usable:
            break
        previous = final.profile
        target = final.profile
        target_params = params
    accepted = bool(
        len(rows) == len(TRANSONIC_RATIOS)
        and rows[-1]["usable"]
        and target is not None
    )
    return target, target_params, rows, accepted


def _physical_source_and_roche(grid, mass: float):
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = PHYSICAL_STREAM_OVER_EDDINGTON * eddington_mdot(mass)
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


def main() -> None:
    arguments = _arguments()
    checkpoint_directory = arguments.checkpoint_directory
    if not checkpoint_directory.is_absolute():
        checkpoint_directory = ROOT / checkpoint_directory
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    profile_path = checkpoint_directory / "transonic_profile.npz"
    if arguments.reuse_transonic_profile:
        fiducial = FiducialParams()
        params = TransonicSlimParams(
            M2_g=fiducial.M2_g,
            Mdot_g_s=(
                TRANSONIC_RATIOS[-1] * eddington_mdot(fiducial.M2_g)
            ),
            alpha=fiducial.alpha_cool,
            n_nodes=TRANSONIC_NODES,
            R_out_rg=OUTER_RADIUS_RG,
        )
        with np.load(profile_path) as data:
            radius = np.asarray(data["radius"], dtype=float)
            sigma = np.asarray(data["surface_density"], dtype=float)
            velocity = np.asarray(data["radial_velocity"], dtype=float)
            omega = np.asarray(data["omega"], dtype=float)
            temperature = np.asarray(data["temperature"], dtype=float)
        transonic_rows = [{"reused_profile": str(profile_path.relative_to(ROOT))}]
        transonic_accepted = True
    else:
        target, params, transonic_rows, transonic_accepted = (
            solve_low_throughput_transonic(
                maximum_nfev_per_stage=arguments.maximum_nfev_per_stage,
                final_maximum_nfev=arguments.final_maximum_nfev,
            )
        )
        if not transonic_accepted or target is None or params is None:
            raise RuntimeError("low-throughput transonic sequence failed")
        plunge = continue_transonic_supersonic_plunge(
            target,
            params,
            INNER_RADIUS_RG * params.r_g,
        )
        radius = np.concatenate((plunge.R[:-1], target.R))
        sigma = np.concatenate((plunge.Sigma[:-1], target.Sigma))
        velocity = -np.concatenate((plunge.u[:-1], target.u))
        omega = np.concatenate((plunge.Omega[:-1], target.Omega))
        temperature = np.concatenate((plunge.T[:-1], target.T))
        np.savez_compressed(
            profile_path,
            radius=radius,
            surface_density=sigma,
            radial_velocity=velocity,
            omega=omega,
            temperature=temperature,
            mdot_g_s=params.Mdot_g_s,
            mdot_over_eddington=TRANSONIC_RATIOS[-1],
        )

    mesh_rows = []
    accepted = transonic_accepted
    for n_cells in GLOBAL_MESHES:
        grid, mapped, correction = conservatively_map_global_profile(
            radius,
            sigma,
            velocity,
            omega,
            temperature,
            params.M2_g,
            n_cells,
            quadrature_order=32,
        )
        stream, stream_rate, provider = _physical_source_and_roche(
            grid, params.M2_g
        )
        initial_profile = evaluate_global_rusanov_profile(
            grid,
            mapped,
            params.M2_g,
            reference_state=mapped,
            boundary_mode="roche_outer",
            alpha=params.alpha,
            stress_boundary_mode="outer_zero_torque",
            include_radiative_cooling=True,
            include_vertical_column_work=True,
            external_sources=stream,
            outer_overflow_provider=provider,
            specific_mechanical_energy_correction=correction,
        )
        mapped_primitives = recover_global_primitives(
            grid,
            mapped,
            params.M2_g,
            specific_mechanical_energy_correction=correction,
        )
        mapped_mach = (
            mapped_primitives.radial_velocity
            / global_effective_sound_speed(mapped_primitives)
        )
        projection_applicable = bool(mapped_mach[0] < -1.0)
        if projection_applicable:
            projection = solve_global_inner_steady_projection(
                grid,
                mapped,
                params.M2_g,
                alpha=params.alpha,
                reference_state=mapped,
                external_sources=stream,
                outer_overflow_provider=provider,
                specific_mechanical_energy_correction=correction,
                maximum_nfev=100,
            )
            final_state = projection.state
            final_profile = projection.profile
            projection_accepted = projection.audit.accepted
            projection_audit = asdict(projection.audit)
        else:
            final_state = mapped
            final_profile = initial_profile
            projection_accepted = True
            projection_audit = {
                "applicable": False,
                "reason": "first global cell is subsonic",
                "first_cell_radial_mach_number": float(mapped_mach[0]),
            }
        roche = final_profile.outer_roche_boundary
        if roche is None:
            raise RuntimeError("low-throughput remnant lacks Roche audit")
        inner_fraction = float(
            final_profile.face_fluxes.mass[0] / stream_rate
        )
        throughput_gate = abs(inner_fraction) <= INNER_THROUGHPUT_LIMIT
        mesh_accepted = bool(projection_accepted and throughput_gate)
        accepted = accepted and mesh_accepted
        loading_time = float(np.sum(mapped.mass) / stream_rate)
        mechanical = make_global_mechanical_energy_reference(
            grid,
            correction,
            mapped,
            provenance={
                "case": f"global-low-throughput-remnant-N{n_cells}",
                "reference": "mapped low-Mdot transonic remnant",
            },
        )
        restart = GlobalAdaptiveRestart(
            state=final_state,
            reference_state=mapped,
            mechanical_reference=mechanical,
            elapsed_time=0.0,
            dt_next=1.0e-8 * loading_time,
            accepted_steps=0,
            rejected_attempts=0,
            provenance={
                "case": f"global-low-throughput-remnant-N{n_cells}",
                "n_cells": n_cells,
                "inner_radius_rg": INNER_RADIUS_RG,
                "source_enabled": True,
                "git": _git_metadata(),
                "attempt_history": [],
                "projection_audit": projection_audit,
            },
        )
        restart_path = checkpoint_directory / f"projected_N{n_cells}.npz"
        save_global_adaptive_restart(restart_path, grid, restart)
        mesh_rows.append(
            {
                "n_cells": n_cells,
                "accepted": mesh_accepted,
                "mapped_inner_mdot_over_physical_stream": float(
                    initial_profile.face_fluxes.mass[0] / stream_rate
                ),
                "projected_inner_mdot_over_physical_stream": inner_fraction,
                "projection_applicable": projection_applicable,
                "projection_audit": projection_audit,
                "mapped_first_cell_radial_mach_number": float(mapped_mach[0]),
                "roche_closure": asdict(
                    global_roche_closure_diagnostic(
                        roche,
                        provider,
                        mass_flux_scale=stream_rate,
                    )
                ),
                "disk_mass": float(np.sum(mapped.mass)),
                "loading_time_seconds": loading_time,
                "restart": str(restart_path.relative_to(ROOT)),
            }
        )
    output = {
        "transonic_ratios": list(TRANSONIC_RATIOS),
        "transonic_nodes": TRANSONIC_NODES,
        "outer_radius_rg": OUTER_RADIUS_RG,
        "inner_radius_rg": INNER_RADIUS_RG,
        "physical_stream_over_eddington": PHYSICAL_STREAM_OVER_EDDINGTON,
        "inner_throughput_limit": INNER_THROUGHPUT_LIMIT,
        "transonic_stages": transonic_rows,
        "mesh_rows": mesh_rows,
        "accepted": bool(accepted),
    }
    report = arguments.output
    if not report.is_absolute():
        report = ROOT / report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(report)
    if not accepted:
        raise RuntimeError("low-throughput global remnant failed its gates")


if __name__ == "__main__":
    main()
