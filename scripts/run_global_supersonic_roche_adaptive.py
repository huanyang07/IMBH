"""Run shared-time adaptive Roche loading with a supersonic inner face."""

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
    parser.add_argument("--target-loading-fraction", required=True, type=float)
    parser.add_argument("--maximum-accepted-steps", default=40, type=int)
    parser.add_argument("--maximum-nfev", default=300, type=int)
    parser.add_argument(
        "--minimum-dt-loading-fraction", default=1.0e-9, type=float
    )
    parser.add_argument("--resume-dt-cap-loading-fraction", type=float)
    parser.add_argument(
        "--meshes",
        nargs="+",
        type=int,
        choices=(64, 96, 128),
        default=(64, 96, 128),
        help="Mesh sizes to run; defaults to the full shared-time ladder.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--milestone-directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.target_loading_fraction <= 0.0:
        raise ValueError("target loading fraction must be positive")
    if arguments.maximum_nfev < 1:
        raise ValueError("maximum nfev must be positive")
    if arguments.minimum_dt_loading_fraction <= 0.0:
        raise ValueError("minimum dt loading fraction must be positive")
    if (
        arguments.resume_dt_cap_loading_fraction is not None
        and arguments.resume_dt_cap_loading_fraction <= 0.0
    ):
        raise ValueError("resume dt cap loading fraction must be positive")
    meshes = tuple(dict.fromkeys(arguments.meshes))
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
    reference_loading_time_seconds = float(
        np.sum(reference_state.mass) / stream_rate
    )
    milestone_directory = arguments.milestone_directory
    if milestone_directory is None:
        milestone_directory = (
            ROOT / "outputs/checkpoints/milestones/global_supersonic_roche"
        )
    elif not milestone_directory.is_absolute():
        milestone_directory = ROOT / milestone_directory
    runs = [
        run_adaptive_campaign(
            context,
            evaluation,
            n_cells=n_cells,
            target_loading_fraction=arguments.target_loading_fraction,
            initial_dt_loading_fraction=1.0e-7,
            restart_path=(
                ROOT
                / "outputs/checkpoints"
                / f"global_supersonic_roche_N{n_cells}.npz"
            ),
            resume=arguments.resume,
            maximum_accepted_steps=arguments.maximum_accepted_steps,
            inner_radius_rg=INNER_RADIUS_RG,
            maximum_nfev=arguments.maximum_nfev,
            minimum_dt_loading_fraction=(
                arguments.minimum_dt_loading_fraction
            ),
            resume_dt_cap_loading_fraction=(
                arguments.resume_dt_cap_loading_fraction
            ),
            reference_loading_time_seconds=reference_loading_time_seconds,
            milestone_directory=milestone_directory,
            milestone_case="global-supersonic-roche",
        )
        for n_cells in meshes
    ]
    reference = max(runs, key=lambda run: run["n_cells"])
    comparisons = []
    for run in runs:
        if run is reference:
            continue
        elapsed_difference = (
            run["elapsed_loading_fraction"]
            - reference["elapsed_loading_fraction"]
        )
        elapsed_scale = max(
            abs(run["elapsed_loading_fraction"]),
            abs(reference["elapsed_loading_fraction"]),
            1.0e-300,
        )
        comparisons.append({
            "n_cells": run["n_cells"],
            "reference_n_cells": reference["n_cells"],
            "elapsed_loading_fraction_difference": elapsed_difference,
            "elapsed_time_seconds_difference": (
                run["elapsed_time_seconds"]
                - reference["elapsed_time_seconds"]
            ),
            "exact_common_physical_time": bool(
                run["elapsed_time_seconds"]
                == reference["elapsed_time_seconds"]
            ),
            "comparable_at_shared_time": (
                abs(elapsed_difference) <= 5.0e-4 * elapsed_scale
            ),
            "disk_mass_change_difference": (
                run["disk_mass_relative_change"]
                - reference["disk_mass_relative_change"]
            ),
            "maximum_H_over_R_relative_difference": (
                run["maximum_H_over_R"]
                / reference["maximum_H_over_R"]
                - 1.0
            ),
            "inner_flux_difference_over_supply": (
                run["final_inner_mass_flux_over_supply"]
                - reference["final_inner_mass_flux_over_supply"]
            ),
            "inner_angular_momentum_flux_difference": (
                run["final_inner_conserved_fluxes"]["angular_momentum"]
                - reference["final_inner_conserved_fluxes"][
                    "angular_momentum"
                ]
            ),
            "inner_total_energy_flux_difference": (
                run["final_inner_conserved_fluxes"]["total_energy"]
                - reference["final_inner_conserved_fluxes"]["total_energy"]
            ),
        })
    report = {
        "inner_radius_rg": INNER_RADIUS_RG,
        "target_loading_fraction": arguments.target_loading_fraction,
        "reference_loading_time_seconds": reference_loading_time_seconds,
        "reference_loading_time_definition": (
            "canonical N128 conservatively mapped initial mass divided by stream supply"
        ),
        "milestone_directory": str(milestone_directory),
        "selected_meshes": list(meshes),
        "maximum_nfev": arguments.maximum_nfev,
        "minimum_dt_loading_fraction": (
            arguments.minimum_dt_loading_fraction
        ),
        "resume_dt_cap_loading_fraction": (
            arguments.resume_dt_cap_loading_fraction
        ),
        "resume": arguments.resume,
        "runs": runs,
        "comparisons_to_N128": comparisons,
    }
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
