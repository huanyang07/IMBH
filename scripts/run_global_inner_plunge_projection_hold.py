"""Hold-test a projected plunge under the source-on production evolution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalAdaptiveRestart,
    evaluate_global_rusanov_profile,
    global_fixed_radius_diagnostics,
    load_global_adaptive_restart,
    recover_global_primitives,
    save_global_adaptive_restart,
)

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import (
    FIXED_DIAGNOSTIC_RADII_RG,
    _conserved_flux_record,
    run_adaptive_campaign,
)
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
INNER_RADIUS_RG = 4.5
REFERENCE_N_CELLS = 64
HOLD_LIMITS = {
    "inner_mass_flux_over_supply": 1.0e-2,
    "inner_angular_flux_relative": 2.0e-2,
    "inner_total_energy_flux_relative": 2.0e-2,
    "fixed_radius_mach": 1.0e-1,
    "fixed_radius_log_surface_density": 2.0e-2,
    "fixed_radius_log_temperature": 2.0e-2,
    "maximum_H_over_R_relative": 1.0e-2,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, required=True)
    parser.add_argument("--projection-restart", type=Path, required=True)
    parser.add_argument("--hold-restart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-reference-loading-fraction", type=float, default=2.0e-7
    )
    parser.add_argument("--maximum-nfev", type=int, default=600)
    parser.add_argument("--maximum-accepted-steps", type=int, default=40)
    return parser.parse_args()


def _relative_difference(new: float, old: float) -> float:
    return float(abs(new / old - 1.0)) if old != 0.0 else float("inf")


def _comparison(initial: dict, final: dict) -> dict:
    initial_fixed = {
        item["radius"]: item for item in initial["fixed_radius_diagnostics"]
    }
    final_fixed = {
        item["radius"]: item for item in final["fixed_radius_diagnostics"]
    }
    fixed_rows = []
    for radius in sorted(set(initial_fixed) & set(final_fixed)):
        old = initial_fixed[radius]
        new = final_fixed[radius]
        fixed_rows.append(
            {
                "radius": radius,
                "radial_mach_difference": float(
                    new["radial_mach_number"] - old["radial_mach_number"]
                ),
                "log_surface_density_difference": float(
                    np.log(new["surface_density"] / old["surface_density"])
                ),
                "log_temperature_difference": float(
                    np.log(new["temperature"] / old["temperature"])
                ),
            }
        )
    old_flux = initial["inner_conserved_fluxes"]
    new_flux = final["final_inner_conserved_fluxes"]
    metrics = {
        "inner_mass_flux_over_supply": abs(
            new_flux["mass_over_source"] - old_flux["mass_over_source"]
        ),
        "inner_angular_flux_relative": _relative_difference(
            new_flux["angular_momentum"], old_flux["angular_momentum"]
        ),
        "inner_total_energy_flux_relative": _relative_difference(
            new_flux["total_energy"], old_flux["total_energy"]
        ),
        "fixed_radius_mach": max(
            (abs(item["radial_mach_difference"]) for item in fixed_rows),
            default=0.0,
        ),
        "fixed_radius_log_surface_density": max(
            (
                abs(item["log_surface_density_difference"])
                for item in fixed_rows
            ),
            default=0.0,
        ),
        "fixed_radius_log_temperature": max(
            (abs(item["log_temperature_difference"]) for item in fixed_rows),
            default=0.0,
        ),
        "maximum_H_over_R_relative": _relative_difference(
            final["maximum_H_over_R"], initial["maximum_H_over_R"]
        ),
    }
    return {
        "metrics": metrics,
        "limits": HOLD_LIMITS,
        "passes": all(
            metrics[name] <= limit for name, limit in HOLD_LIMITS.items()
        ),
        "fixed_radius_differences": fixed_rows,
    }


def main() -> None:
    arguments = _arguments()
    if arguments.n_cells not in {64, 96}:
        raise ValueError("projection hold supports N64 or N96")
    projection_path = arguments.projection_restart
    hold_path = arguments.hold_restart
    output = arguments.output
    for name, path in (
        ("projection restart", projection_path),
        ("hold restart", hold_path),
        ("output", output),
    ):
        if not path.is_absolute():
            if name == "projection restart":
                projection_path = ROOT / path
            elif name == "hold restart":
                hold_path = ROOT / path
            else:
                output = ROOT / path
    context, evaluation = _canonical_open_evaluation()
    grid, initial, correction, stream, stream_rate, provider = _prepared_case(
        context,
        evaluation,
        arguments.n_cells,
        inner_radius_rg=INNER_RADIUS_RG,
    )
    loaded_grid, projected = load_global_adaptive_restart(
        projection_path, grid=grid
    )
    if not np.array_equal(loaded_grid.edges, grid.edges):
        raise RuntimeError("projection restart grid mismatch")
    hold_restart = GlobalAdaptiveRestart(
        state=projected.state,
        reference_state=projected.reference_state,
        mechanical_reference=projected.mechanical_reference,
        elapsed_time=0.0,
        dt_next=projected.dt_next,
        accepted_steps=0,
        rejected_attempts=0,
        provenance=dict(projected.provenance),
    )
    save_global_adaptive_restart(hold_path, grid, hold_restart)
    mass = context.base.inner_params.M2_g
    primitives = recover_global_primitives(
        grid,
        projected.state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    profile = evaluate_global_rusanov_profile(
        grid,
        projected.state,
        mass,
        reference_state=projected.reference_state,
        boundary_mode="roche_outer",
        alpha=context.base.alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=True,
        external_sources=stream,
        primitives=primitives,
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=correction,
    )
    fixed_radii = tuple(
        value * context.base.inner_params.r_g
        for value in FIXED_DIAGNOSTIC_RADII_RG
    )
    initial_record = {
        "maximum_H_over_R": float(
            np.max(np.asarray(primitives.vertical.H) / grid.centers)
        ),
        "inner_conserved_fluxes": _conserved_flux_record(
            profile.face_fluxes, 0, stream
        ),
        "fixed_radius_diagnostics": [
            asdict(item)
            for item in global_fixed_radius_diagnostics(
                grid,
                primitives,
                profile.face_fluxes,
                mass,
                fixed_radii,
            )
        ],
    }
    (
        _reference_grid,
        reference_initial,
        _reference_correction,
        _reference_stream,
        reference_rate,
        _reference_provider,
    ) = _prepared_case(
        context,
        evaluation,
        REFERENCE_N_CELLS,
        inner_radius_rg=INNER_RADIUS_RG,
    )
    reference_loading_time = float(
        np.sum(reference_initial.mass) / reference_rate
    )
    mesh_loading_time = float(np.sum(initial.mass) / stream_rate)
    target_time = (
        arguments.target_reference_loading_fraction * reference_loading_time
    )
    run = run_adaptive_campaign(
        context,
        evaluation,
        n_cells=arguments.n_cells,
        target_loading_fraction=target_time / mesh_loading_time,
        initial_dt_loading_fraction=1.0e-8,
        restart_path=hold_path,
        resume=True,
        maximum_accepted_steps=arguments.maximum_accepted_steps,
        inner_radius_rg=INNER_RADIUS_RG,
        maximum_nfev=arguments.maximum_nfev,
        minimum_dt_loading_fraction=1.0e-10,
        reference_loading_time_seconds=reference_loading_time,
        milestone_directory=(
            ROOT
            / "outputs/checkpoints/milestones"
            / f"global_inner_plunge_projection_hold_N{arguments.n_cells}"
        ),
        milestone_case=(
            f"global-inner-plunge-projection-hold-N{arguments.n_cells}"
        ),
        source_enabled=True,
    )
    comparison = _comparison(initial_record, run)
    report = {
        "n_cells": arguments.n_cells,
        "target_reference_loading_fraction": (
            arguments.target_reference_loading_fraction
        ),
        "initial_projected_state": initial_record,
        "hold": run,
        "comparison": comparison,
        "hold_gate_passed": bool(run["target_reached"] and comparison["passes"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
