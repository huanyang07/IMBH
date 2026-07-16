"""Regenerate an exact-common-time supersonic/Roche diagnostic snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import run_adaptive_campaign
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
INNER_RADIUS_RG = 4.5


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-loading-fraction", type=float, default=1.0e-7
    )
    parser.add_argument("--maximum-nfev", type=int, default=600)
    parser.add_argument("--maximum-accepted-steps", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--milestone-directory", type=Path)
    return parser.parse_args()


def _fixed_radius_comparisons(runs: list[dict], reference: dict) -> list[dict]:
    reference_by_radius = {
        item["radius"]: item for item in reference["fixed_radius_diagnostics"]
    }
    comparisons = []
    for run in runs:
        if run is reference:
            continue
        for item in run["fixed_radius_diagnostics"]:
            reference_item = reference_by_radius[item["radius"]]
            comparisons.append(
                {
                    "n_cells": run["n_cells"],
                    "reference_n_cells": reference["n_cells"],
                    "radius": item["radius"],
                    "radial_mach_difference": (
                        item["radial_mach_number"]
                        - reference_item["radial_mach_number"]
                    ),
                    "log_surface_density_difference": float(
                        np.log(
                            item["surface_density"]
                            / reference_item["surface_density"]
                        )
                    ),
                    "log_temperature_difference": float(
                        np.log(
                            item["temperature"]
                            / reference_item["temperature"]
                        )
                    ),
                    "omega_over_omega_k_difference": (
                        item["omega_over_omega_k"]
                        - reference_item["omega_over_omega_k"]
                    ),
                    "H_over_R_difference": (
                        item["H_over_R"] - reference_item["H_over_R"]
                    ),
                    "mass_flux_over_reference_supply_difference": (
                        item["mass_flux"]
                        - reference_item["mass_flux"]
                    )
                    / reference["stream_mass_rate"],
                }
            )
    return comparisons


def main() -> None:
    arguments = _arguments()
    if arguments.reference_loading_fraction <= 0.0:
        raise ValueError("reference loading fraction must be positive")
    if arguments.maximum_nfev < 1:
        raise ValueError("maximum nfev must be positive")
    context, evaluation = _canonical_open_evaluation()
    (
        _reference_grid,
        reference_state,
        _correction,
        _stream,
        stream_rate,
        _provider,
    ) = _prepared_case(
        context,
        evaluation,
        128,
        inner_radius_rg=INNER_RADIUS_RG,
    )
    reference_loading_time = float(np.sum(reference_state.mass) / stream_rate)
    target_time = arguments.reference_loading_fraction * reference_loading_time
    milestone_directory = arguments.milestone_directory
    if milestone_directory is None:
        milestone_directory = (
            ROOT / "outputs/checkpoints/milestones/global_exact_common_time"
        )
    elif not milestone_directory.is_absolute():
        milestone_directory = ROOT / milestone_directory
    runs = []
    for n_cells in (64, 96, 128):
        _grid, initial, _correction, _stream, local_stream_rate, _provider = (
            _prepared_case(
                context,
                evaluation,
                n_cells,
                inner_radius_rg=INNER_RADIUS_RG,
            )
        )
        mesh_loading_time = float(np.sum(initial.mass) / local_stream_rate)
        target_mesh_fraction = target_time / mesh_loading_time
        run = run_adaptive_campaign(
            context,
            evaluation,
            n_cells=n_cells,
            target_loading_fraction=target_mesh_fraction,
            initial_dt_loading_fraction=target_mesh_fraction,
            restart_path=(
                ROOT
                / "outputs/checkpoints"
                / f"global_exact_common_time_N{n_cells}.npz"
            ),
            resume=arguments.resume,
            maximum_accepted_steps=arguments.maximum_accepted_steps,
            inner_radius_rg=INNER_RADIUS_RG,
            maximum_nfev=arguments.maximum_nfev,
            minimum_dt_loading_fraction=1.0e-9,
            reference_loading_time_seconds=reference_loading_time,
            milestone_directory=milestone_directory,
            milestone_case="global-exact-common-time",
        )
        run["stream_mass_rate"] = float(local_stream_rate)
        runs.append(run)
    if not all(run["target_reached"] for run in runs):
        raise RuntimeError("at least one mesh failed the exact common-time target")
    if len({run["elapsed_time_seconds"] for run in runs}) != 1:
        raise RuntimeError("mesh snapshots do not have exactly equal physical time")
    reference = runs[-1]
    comparisons = []
    for run in runs[:-1]:
        comparisons.append(
            {
                "n_cells": run["n_cells"],
                "reference_n_cells": reference["n_cells"],
                "exact_common_physical_time": bool(
                    run["elapsed_time_seconds"]
                    == reference["elapsed_time_seconds"]
                ),
                "disk_mass_change_difference": (
                    run["disk_mass_relative_change"]
                    - reference["disk_mass_relative_change"]
                ),
                "maximum_H_over_R_relative_difference": (
                    run["maximum_H_over_R"] / reference["maximum_H_over_R"]
                    - 1.0
                ),
                "inner_mass_flux_over_supply_difference": (
                    run["final_inner_mass_flux_over_supply"]
                    - reference["final_inner_mass_flux_over_supply"]
                ),
                "inner_angular_momentum_flux_relative_difference": (
                    run["final_inner_conserved_fluxes"]["angular_momentum"]
                    / reference["final_inner_conserved_fluxes"][
                        "angular_momentum"
                    ]
                    - 1.0
                ),
                "inner_total_energy_flux_relative_difference": (
                    run["final_inner_conserved_fluxes"]["total_energy"]
                    / reference["final_inner_conserved_fluxes"]["total_energy"]
                    - 1.0
                ),
                "sonic_radius_difference": (
                    run["sonic_resolution"]["sonic_radius"]
                    - reference["sonic_resolution"]["sonic_radius"]
                ),
            }
        )
    report = {
        "reference_loading_fraction": arguments.reference_loading_fraction,
        "reference_loading_time_seconds": reference_loading_time,
        "exact_common_physical_time_seconds": target_time,
        "maximum_nfev": arguments.maximum_nfev,
        "maximum_accepted_steps": arguments.maximum_accepted_steps,
        "resume": arguments.resume,
        "milestone_directory": str(milestone_directory),
        "runs": runs,
        "comparisons_to_N128": comparisons,
        "fixed_radius_comparisons_to_N128": _fixed_radius_comparisons(
            runs, reference
        ),
    }
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
