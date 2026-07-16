"""Matched source-off/source-on holds for the low-throughput remnant."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalAdaptiveRestart,
    PaczynskiWiitaPotential,
    evaluate_global_rusanov_profile,
    global_fixed_radius_diagnostics,
    load_global_adaptive_restart,
    recover_global_primitives,
    save_global_adaptive_restart,
)

from run_global_low_throughput_remnant import (
    INNER_RADIUS_RG,
    _physical_source_and_roche,
)
from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import (
    FIXED_DIAGNOSTIC_RADII_RG,
    _conserved_flux_record,
    run_adaptive_campaign,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/checkpoints/global_low_throughput_remnant"
DEFAULT_RESTARTS = (
    ROOT / "outputs/checkpoints/global_low_throughput_remnant_hold"
)
HOLD_LIMITS = {
    "inner_mass_flux_over_supply": 1.0e-3,
    "fixed_radius_mach": 1.0e-1,
    "fixed_radius_log_surface_density": 2.0e-2,
    "fixed_radius_log_temperature": 2.0e-2,
    "maximum_H_over_R_relative": 1.0e-2,
}
SOURCE_DIFFERENCE_LIMIT = 1.0e-5


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, required=True)
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--restart-directory", type=Path, default=DEFAULT_RESTARTS
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-reference-loading-fraction", type=float, default=2.0e-7
    )
    parser.add_argument("--maximum-nfev", type=int, default=600)
    parser.add_argument("--maximum-accepted-steps", type=int, default=40)
    return parser.parse_args()


def _relative_difference(new: float, old: float) -> float:
    return float(abs(new / old - 1.0)) if old != 0.0 else float("inf")


def _initial_record(grid, restart, mass, alpha, stream, provider) -> dict:
    primitives = recover_global_primitives(
        grid,
        restart.state,
        mass,
        specific_mechanical_energy_correction=(
            restart.mechanical_reference.specific_offset
        ),
    )
    profile = evaluate_global_rusanov_profile(
        grid,
        restart.state,
        mass,
        reference_state=restart.reference_state,
        boundary_mode="characteristic_inner_roche_outer",
        alpha=alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=True,
        external_sources=stream,
        primitives=primitives,
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=(
            restart.mechanical_reference.specific_offset
        ),
    )
    r_g = PaczynskiWiitaPotential(mass).r_g
    diagnostic_radii = tuple(
        value * r_g
        for value in FIXED_DIAGNOSTIC_RADII_RG
        if grid.edges[0]
        <= value * r_g
        <= grid.edges[-1]
    )
    return {
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
                diagnostic_radii,
            )
        ],
    }


def _drift(initial: dict, final: dict) -> dict:
    initial_fixed = {
        item["radius"]: item for item in initial["fixed_radius_diagnostics"]
    }
    final_fixed = {
        item["radius"]: item for item in final["fixed_radius_diagnostics"]
    }
    rows = []
    for radius in sorted(set(initial_fixed) & set(final_fixed)):
        old = initial_fixed[radius]
        new = final_fixed[radius]
        rows.append(
            {
                "radius": radius,
                "mach": abs(
                    new["radial_mach_number"]
                    - old["radial_mach_number"]
                ),
                "log_surface_density": abs(
                    np.log(new["surface_density"] / old["surface_density"])
                ),
                "log_temperature": abs(
                    np.log(new["temperature"] / old["temperature"])
                ),
            }
        )
    metrics = {
        "inner_mass_flux_over_supply": abs(
            final["final_inner_conserved_fluxes"]["mass_over_source"]
            - initial["inner_conserved_fluxes"]["mass_over_source"]
        ),
        "fixed_radius_mach": max((row["mach"] for row in rows), default=0.0),
        "fixed_radius_log_surface_density": max(
            (row["log_surface_density"] for row in rows), default=0.0
        ),
        "fixed_radius_log_temperature": max(
            (row["log_temperature"] for row in rows), default=0.0
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
        "fixed_radius_differences": rows,
    }


def main() -> None:
    arguments = _arguments()
    if arguments.n_cells not in {64, 96}:
        raise ValueError("remnant hold supports N64 or N96")
    input_directory = arguments.input_directory
    restart_directory = arguments.restart_directory
    output = arguments.output
    if not input_directory.is_absolute():
        input_directory = ROOT / input_directory
    if not restart_directory.is_absolute():
        restart_directory = ROOT / restart_directory
    if not output.is_absolute():
        output = ROOT / output
    grid, base = load_global_adaptive_restart(
        input_directory / f"projected_N{arguments.n_cells}.npz"
    )
    context, evaluation = _canonical_open_evaluation()
    mass = context.base.inner_params.M2_g
    stream, stream_rate, provider = _physical_source_and_roche(grid, mass)
    prepared = (
        grid,
        base.reference_state,
        base.mechanical_reference.specific_offset,
        stream,
        stream_rate,
        provider,
    )
    reference_grid, reference = load_global_adaptive_restart(
        input_directory / "projected_N64.npz"
    )
    _reference_stream, reference_rate, _reference_provider = (
        _physical_source_and_roche(reference_grid, mass)
    )
    reference_loading_time = float(
        np.sum(reference.reference_state.mass) / reference_rate
    )
    initial = _initial_record(
        grid, base, mass, context.base.alpha, stream, provider
    )
    restart_directory.mkdir(parents=True, exist_ok=True)
    runs = {}
    final_states = {}
    target_time = (
        arguments.target_reference_loading_fraction * reference_loading_time
    )
    mesh_loading_time = float(np.sum(base.reference_state.mass) / stream_rate)
    for source_enabled in (False, True):
        label = "source_on" if source_enabled else "source_off"
        path = restart_directory / f"{label}_N{arguments.n_cells}.npz"
        restart = GlobalAdaptiveRestart(
            state=base.state,
            reference_state=base.reference_state,
            mechanical_reference=base.mechanical_reference,
            elapsed_time=0.0,
            dt_next=1.0e-8 * mesh_loading_time,
            accepted_steps=0,
            rejected_attempts=0,
            provenance={
                **base.provenance,
                "case": (
                    f"global-low-throughput-remnant-{label}"
                    f"-N{arguments.n_cells}"
                ),
                "source_enabled": source_enabled,
                "attempt_history": [],
            },
        )
        save_global_adaptive_restart(path, grid, restart)
        runs[label] = run_adaptive_campaign(
            context,
            evaluation,
            n_cells=arguments.n_cells,
            target_loading_fraction=target_time / mesh_loading_time,
            initial_dt_loading_fraction=1.0e-8,
            restart_path=path,
            resume=True,
            maximum_accepted_steps=arguments.maximum_accepted_steps,
            inner_radius_rg=INNER_RADIUS_RG,
            maximum_nfev=arguments.maximum_nfev,
            minimum_dt_loading_fraction=1.0e-10,
            reference_loading_time_seconds=reference_loading_time,
            milestone_directory=(
                restart_directory
                / "milestones"
                / f"{label}_N{arguments.n_cells}"
            ),
            milestone_case=(
                f"global-low-throughput-remnant-{label}-N{arguments.n_cells}"
            ),
            source_enabled=source_enabled,
            prepared_case=prepared,
            inner_boundary_mode="characteristic_inner_roche_outer",
        )
        _final_grid, final_states[label] = load_global_adaptive_restart(
            path, grid=grid
        )

    drifts = {label: _drift(initial, run) for label, run in runs.items()}
    elapsed = runs["source_on"]["elapsed_time_seconds"]
    source_differences = {}
    for name in ("mass", "angular_momentum", "total_energy"):
        actual = float(
            np.sum(
                getattr(final_states["source_on"].state, name)
                - getattr(final_states["source_off"].state, name)
            )
        )
        expected = float(np.sum(getattr(stream, name)) * elapsed)
        source_differences[name] = {
            "actual": actual,
            "expected": expected,
            "relative_defect": abs(actual - expected)
            / max(abs(expected), 1.0),
        }
    source_difference_pass = all(
        row["relative_defect"] <= SOURCE_DIFFERENCE_LIMIT
        for row in source_differences.values()
    )
    hold_gate = bool(
        all(run["target_reached"] for run in runs.values())
        and all(drift["passes"] for drift in drifts.values())
        and all(not run["final_roche_choked"] for run in runs.values())
        and source_difference_pass
    )
    report = {
        "n_cells": arguments.n_cells,
        "target_reference_loading_fraction": (
            arguments.target_reference_loading_fraction
        ),
        "initial": initial,
        "runs": runs,
        "drifts": drifts,
        "source_on_minus_source_off": source_differences,
        "source_difference_limit": SOURCE_DIFFERENCE_LIMIT,
        "source_difference_pass": source_difference_pass,
        "hold_gate_passed": hold_gate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
