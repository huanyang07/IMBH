"""Continue the source-free N64 control through a bounded relaxation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import load_global_adaptive_restart

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import run_adaptive_campaign
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
INNER_RADIUS_RG = 4.5
REFERENCE_N_CELLS = 64
DEFAULT_TARGETS = (2.0e-7, 5.0e-7, 1.0e-6, 2.0e-6)
MINIMUM_GRADIENT_LENGTH_OVER_CELL = 1.0
RELAXATION_LIMITS = {
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-cells", type=int, default=64)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--maximum-nfev", type=int, default=600)
    parser.add_argument("--maximum-accepted-steps", type=int, default=80)
    parser.add_argument(
        "--maximum-reference-loading-fraction",
        type=float,
        default=max(DEFAULT_TARGETS),
    )
    return parser.parse_args()


def _relative_difference(left: float, right: float) -> float:
    return float(abs(left / right - 1.0)) if right != 0.0 else float("inf")


def _milestone_comparison(left: dict, right: dict) -> dict:
    left_fixed = {
        item["radius"]: item for item in left["fixed_radius_diagnostics"]
    }
    right_fixed = {
        item["radius"]: item for item in right["fixed_radius_diagnostics"]
    }
    fixed_rows = []
    for radius in sorted(set(left_fixed) & set(right_fixed)):
        old = left_fixed[radius]
        new = right_fixed[radius]
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
    left_flux = left["final_inner_conserved_fluxes"]
    right_flux = right["final_inner_conserved_fluxes"]
    metrics = {
        "inner_mass_flux_over_supply": abs(
            right_flux["mass_over_source"] - left_flux["mass_over_source"]
        ),
        "inner_angular_flux_relative": _relative_difference(
            right_flux["angular_momentum"], left_flux["angular_momentum"]
        ),
        "inner_total_energy_flux_relative": _relative_difference(
            right_flux["total_energy"], left_flux["total_energy"]
        ),
        "fixed_radius_mach": max(
            (abs(row["radial_mach_difference"]) for row in fixed_rows),
            default=0.0,
        ),
        "fixed_radius_log_surface_density": max(
            (
                abs(row["log_surface_density_difference"])
                for row in fixed_rows
            ),
            default=0.0,
        ),
        "fixed_radius_log_temperature": max(
            (abs(row["log_temperature_difference"]) for row in fixed_rows),
            default=0.0,
        ),
        "maximum_H_over_R_relative": _relative_difference(
            right["maximum_H_over_R"], left["maximum_H_over_R"]
        ),
    }
    return {
        "left_reference_loading_fraction": (
            left["elapsed_reference_loading_fraction"]
        ),
        "right_reference_loading_fraction": (
            right["elapsed_reference_loading_fraction"]
        ),
        "metrics": metrics,
        "limits": RELAXATION_LIMITS,
        "passes": all(
            metrics[name] <= limit
            for name, limit in RELAXATION_LIMITS.items()
        ),
        "fixed_radius_differences": fixed_rows,
    }


def _last_controller(run: dict) -> dict | None:
    for record in reversed(run["records"]):
        if not record["accepted"]:
            continue
        for attempt in reversed(record["attempts"]):
            if (
                attempt["nonlinear_accepted"]
                and attempt["physical_change_accepted"]
            ):
                return attempt["controller"]
    return None


def _last_controller_from_restart(path: Path) -> dict | None:
    _grid, restart = load_global_adaptive_restart(path)
    history = restart.provenance.get("attempt_history", [])
    for record in reversed(history):
        if not record.get("accepted", False):
            continue
        for attempt in reversed(record.get("attempts", [])):
            if (
                attempt.get("nonlinear_accepted", False)
                and attempt.get("physical_change_accepted", False)
            ):
                return attempt.get("controller")
    return None


def _milestone_directory(n_cells: int) -> Path:
    suffix = "" if n_cells == 64 else f"_N{n_cells}"
    return (
        ROOT
        / "outputs/checkpoints/milestones"
        / f"global_source_free_relaxation{suffix}"
    )


def _existing_milestone(
    target_reference_fraction: float, n_cells: int
) -> Path | None:
    directory = _milestone_directory(n_cells)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.get("checkpoints", []):
        fraction = entry.get("metadata", {}).get(
            "elapsed_reference_loading_fraction"
        )
        if fraction is not None and np.isclose(
            float(fraction),
            target_reference_fraction,
            rtol=0.0,
            atol=1.0e-16,
        ):
            return directory / entry["path"]
    return None


def main() -> None:
    arguments = _arguments()
    if arguments.n_cells not in {64, 96, 128}:
        raise ValueError("relaxation audit supports N64, N96, or N128")
    restart_path = arguments.restart
    if restart_path is None:
        if arguments.n_cells == 64:
            restart_path = Path(
                "outputs/checkpoints/global_source_on_off/source_off_N64.npz"
            )
        else:
            restart_path = Path(
                "outputs/checkpoints/global_source_free_relaxation"
                f"/source_off_N{arguments.n_cells}.npz"
            )
    if not restart_path.is_absolute():
        restart_path = ROOT / restart_path
    context, evaluation = _canonical_open_evaluation()
    (
        _reference_grid,
        reference_initial,
        _reference_correction,
        _reference_stream,
        reference_stream_rate,
        _reference_provider,
    ) = _prepared_case(
        context,
        evaluation,
        REFERENCE_N_CELLS,
        inner_radius_rg=INNER_RADIUS_RG,
    )
    reference_loading_time = float(
        np.sum(reference_initial.mass) / reference_stream_rate
    )
    _grid, initial, _correction, _stream, stream_rate, _provider = (
        _prepared_case(
            context,
            evaluation,
            arguments.n_cells,
            inner_radius_rg=INNER_RADIUS_RG,
        )
    )
    mesh_loading_time = float(np.sum(initial.mass) / stream_rate)
    stages = []
    comparisons = []
    termination = "bounded_target_exhausted_without_relaxation"
    requested_targets = tuple(
        target
        for target in DEFAULT_TARGETS
        if target <= arguments.maximum_reference_loading_fraction
    )
    if not requested_targets:
        raise ValueError(
            "maximum reference loading fraction excludes every target"
        )
    for target_reference_fraction in requested_targets:
        target_time = target_reference_fraction * reference_loading_time
        target_mesh_fraction = target_time / mesh_loading_time
        existing = _existing_milestone(
            target_reference_fraction, arguments.n_cells
        )
        stage_restart = restart_path if existing is None else existing
        run = run_adaptive_campaign(
            context,
            evaluation,
            n_cells=arguments.n_cells,
            target_loading_fraction=target_mesh_fraction,
            initial_dt_loading_fraction=target_mesh_fraction,
            restart_path=stage_restart,
            resume=stage_restart.exists(),
            maximum_accepted_steps=arguments.maximum_accepted_steps,
            inner_radius_rg=INNER_RADIUS_RG,
            maximum_nfev=arguments.maximum_nfev,
            minimum_dt_loading_fraction=1.0e-10,
            reference_loading_time_seconds=reference_loading_time,
            milestone_directory=(
                None
                if existing is not None
                else _milestone_directory(arguments.n_cells)
            ),
            milestone_case=(
                f"global-source-free-relaxation-N{arguments.n_cells}"
            ),
            source_enabled=False,
        )
        run["last_accepted_controller"] = (
            _last_controller(run)
            or _last_controller_from_restart(stage_restart)
        )
        stages.append(run)
        if not run["target_reached"]:
            termination = "named_numerical_stop_before_target"
            break
        if len(stages) >= 2:
            comparison = _milestone_comparison(stages[-2], stages[-1])
            comparisons.append(comparison)
        if (
            run["sonic_resolution"][
                "minimum_velocity_gradient_length_over_cell_width"
            ]
            < MINIMUM_GRADIENT_LENGTH_OVER_CELL
        ):
            termination = "named_plunge_resolution_stop"
            break
        if len(stages) >= 2:
            if comparison["passes"]:
                termination = "inner_mapping_relaxation_gate_passed"
                break
    report = {
        "n_cells": arguments.n_cells,
        "inner_radius_rg": INNER_RADIUS_RG,
        "source_enabled": False,
        "mesh_loading_time_seconds": mesh_loading_time,
        "reference_loading_time_seconds": reference_loading_time,
        "targets_requested_in_reference_loading_fraction": list(
            requested_targets
        ),
        "minimum_gradient_length_over_cell_gate": (
            MINIMUM_GRADIENT_LENGTH_OVER_CELL
        ),
        "relaxation_limits": RELAXATION_LIMITS,
        "termination": termination,
        "relaxation_gate_passed": (
            termination == "inner_mapping_relaxation_gate_passed"
        ),
        "stages": stages,
        "comparisons": comparisons,
    }
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
