"""Certify a shared physical time and extend the accepted N64 restart."""

from __future__ import annotations

import json
from pathlib import Path

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import run_adaptive_campaign


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/global_roche_mesh_time_certification.json"
SHARED_LOADING_FRACTION = 1.0e-7
EXTENDED_N64_LOADING_FRACTION = 1.0e-6


def _last_flux(run: dict, name: str) -> float:
    if not run["records"]:
        raise ValueError("mesh certification run has no accepted records")
    return float(run["records"][-1][name])


def main() -> None:
    context, evaluation = _canonical_open_evaluation()
    shared_runs = [
        run_adaptive_campaign(
            context,
            evaluation,
            n_cells=n_cells,
            target_loading_fraction=SHARED_LOADING_FRACTION,
            initial_dt_loading_fraction=SHARED_LOADING_FRACTION,
            restart_path=(
                ROOT
                / f"outputs/checkpoints/global_roche_shared_N{n_cells}.npz"
            ),
        )
        for n_cells in (64, 96, 128)
    ]
    extended_n64 = run_adaptive_campaign(
        context,
        evaluation,
        n_cells=64,
        target_loading_fraction=EXTENDED_N64_LOADING_FRACTION,
        initial_dt_loading_fraction=1.0e-7,
        restart_path=(
            ROOT / "outputs/checkpoints/global_roche_adaptive_N64.npz"
        ),
        resume=True,
        maximum_accepted_steps=30,
    )
    reference = shared_runs[-1]
    comparisons = []
    for run in shared_runs[:-1]:
        comparisons.append(
            {
                "n_cells": run["n_cells"],
                "reference_n_cells": reference["n_cells"],
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
                    _last_flux(run, "inner_mass_flux_over_supply")
                    - _last_flux(reference, "inner_mass_flux_over_supply")
                ),
                "roche_energy_difference": (
                    run["records"][-1]["roche_available_specific_energy"]
                    - reference["records"][-1][
                        "roche_available_specific_energy"
                    ]
                ),
            }
        )
    report = {
        "shared_loading_fraction": SHARED_LOADING_FRACTION,
        "shared_time_runs": shared_runs,
        "shared_time_comparisons_to_N128": comparisons,
        "extended_N64": extended_n64,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
