"""Run shared-time adaptive Roche loading with a supersonic inner face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import run_adaptive_campaign


ROOT = Path(__file__).resolve().parents[1]
INNER_RADIUS_RG = 4.5


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-loading-fraction", required=True, type=float)
    parser.add_argument("--maximum-accepted-steps", default=40, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.target_loading_fraction <= 0.0:
        raise ValueError("target loading fraction must be positive")
    context, evaluation = _canonical_open_evaluation()
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
        )
        for n_cells in (64, 96, 128)
    ]
    reference = runs[-1]
    comparisons = []
    for run in runs[:-1]:
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
        })
    report = {
        "inner_radius_rg": INNER_RADIUS_RG,
        "target_loading_fraction": arguments.target_loading_fraction,
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
