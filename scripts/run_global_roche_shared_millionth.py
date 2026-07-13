"""Resume all meshes to the shared 1e-6 loading-time checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import run_adaptive_campaign


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/global_roche_shared_millionth.json"
TARGET_LOADING_FRACTION = 1.0e-6


def main() -> None:
    context, evaluation = _canonical_open_evaluation()
    restart_paths = {
        64: ROOT / "outputs/checkpoints/global_roche_adaptive_N64.npz",
        96: ROOT / "outputs/checkpoints/global_roche_shared_N96.npz",
        128: ROOT / "outputs/checkpoints/global_roche_shared_N128.npz",
    }
    runs = [
        run_adaptive_campaign(
            context,
            evaluation,
            n_cells=n_cells,
            target_loading_fraction=TARGET_LOADING_FRACTION,
            initial_dt_loading_fraction=5.0e-8,
            restart_path=restart_paths[n_cells],
            resume=True,
            maximum_accepted_steps=40,
        )
        for n_cells in (64, 96, 128)
    ]
    reference = runs[-1]
    comparisons = [
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
                run["final_inner_mass_flux_over_supply"]
                - reference["final_inner_mass_flux_over_supply"]
            ),
            "roche_energy_difference": (
                run["final_roche_available_specific_energy"]
                - reference["final_roche_available_specific_energy"]
            ),
        }
        for run in runs[:-1]
    ]
    report = {
        "target_loading_fraction": TARGET_LOADING_FRACTION,
        "runs": runs,
        "comparisons_to_N128": comparisons,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
