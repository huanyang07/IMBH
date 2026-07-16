"""Benchmark one immutable supersonic-Roche restart step without advancing it."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    advance_global_backward_euler,
    load_global_adaptive_restart,
    recover_global_primitives,
)

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
INNER_RADIUS_RG = 4.5


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meshes", nargs="+", type=int, choices=(64, 96, 128), required=True
    )
    parser.add_argument("--maximum-nfev", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _primitive_change_record(grid, old, new) -> dict:
    old_h_over_r = np.asarray(old.vertical.H, dtype=float) / grid.centers
    new_h_over_r = np.asarray(new.vertical.H, dtype=float) / grid.centers
    component_changes = {
        "log_surface_density": np.abs(
            np.log(new.surface_density / old.surface_density)
        ),
        "radial_velocity_over_c": np.abs(
            (new.radial_velocity - old.radial_velocity) / C
        ),
        "log_omega": np.abs(np.log(new.omega / old.omega)),
        "log_temperature": np.abs(np.log(new.temperature / old.temperature)),
        "relative_thickness": np.abs(new_h_over_r / old_h_over_r - 1.0),
    }
    maxima = {
        name: float(np.max(values)) for name, values in component_changes.items()
    }
    controlling_name = max(maxima, key=maxima.get)
    controlling_values = component_changes[controlling_name]
    controlling_index = int(np.argmax(controlling_values))
    return {
        "maxima": maxima,
        "maximum_packed_primitive_update": float(
            max(
                maxima["log_surface_density"],
                maxima["radial_velocity_over_c"],
                maxima["log_omega"],
                maxima["log_temperature"],
            )
        ),
        "controller": {
            "variable": controlling_name,
            "cell_index": controlling_index,
            "radius": float(grid.centers[controlling_index]),
            "change": float(controlling_values[controlling_index]),
        },
    }


def _run_mesh(
    context,
    evaluation,
    n_cells: int,
    maximum_nfev: int,
) -> dict:
    grid, initial, correction, stream, stream_rate, provider = _prepared_case(
        context,
        evaluation,
        n_cells,
        inner_radius_rg=INNER_RADIUS_RG,
    )
    restart_path = (
        ROOT / "outputs/checkpoints" / f"global_supersonic_roche_N{n_cells}.npz"
    )
    loaded_grid, restart = load_global_adaptive_restart(restart_path, grid=grid)
    if not np.array_equal(loaded_grid.edges, grid.edges):
        raise RuntimeError("restart grid changed while loading benchmark state")
    for name in ("mass", "radial_momentum", "angular_momentum", "total_energy"):
        if not np.array_equal(
            getattr(restart.reference_state, name), getattr(initial, name)
        ):
            raise RuntimeError("restart reference differs from canonical mapping")
    mass = context.base.inner_params.M2_g
    old = recover_global_primitives(
        grid,
        restart.state,
        mass,
        specific_mechanical_energy_correction=(
            restart.mechanical_reference.specific_offset
        ),
    )
    wall_start = perf_counter()
    result = advance_global_backward_euler(
        grid,
        restart.state,
        mass,
        restart.dt_next,
        alpha=context.base.alpha,
        reference_state=restart.reference_state,
        boundary_mode="roche_outer",
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=True,
        external_sources=stream,
        jacobian_mode="sparse_forward",
        outer_overflow_provider=provider,
        max_nfev=maximum_nfev,
        specific_mechanical_energy_correction=(
            restart.mechanical_reference.specific_offset
        ),
    )
    measured_wall_seconds = perf_counter() - wall_start
    new = recover_global_primitives(
        grid,
        result.state if result.accepted else restart.state,
        mass,
        specific_mechanical_energy_correction=(
            restart.mechanical_reference.specific_offset
        ),
    )
    return {
        "n_cells": n_cells,
        "restart_path": str(restart_path),
        "restart_elapsed_time_seconds": restart.elapsed_time,
        "requested_dt_seconds": restart.dt_next,
        "requested_dt_over_mesh_loading_time": float(
            restart.dt_next / (np.sum(initial.mass) / stream_rate)
        ),
        "accepted": result.accepted,
        "message": result.message,
        "nfev": result.nfev,
        "maximum_scaled_residual": result.maximum_scaled_residual,
        "maximum_storage_scaled_ledger_defect": (
            result.maximum_storage_scaled_ledger_defect
        ),
        "measured_wall_seconds": measured_wall_seconds,
        "jacobian_audit": (
            None if result.jacobian_audit is None else asdict(result.jacobian_audit)
        ),
        "nonlinear_solve_audit": (
            None
            if result.nonlinear_solve_audit is None
            else asdict(result.nonlinear_solve_audit)
        ),
        "primitive_changes": _primitive_change_record(grid, old, new),
        "input_state_unchanged": all(
            np.array_equal(
                getattr(restart.state, name),
                getattr(
                    load_global_adaptive_restart(restart_path, grid=grid)[1].state,
                    name,
                ),
            )
            for name in (
                "mass",
                "radial_momentum",
                "angular_momentum",
                "total_energy",
            )
        ),
    }


def main() -> None:
    arguments = _arguments()
    if arguments.maximum_nfev < 1:
        raise ValueError("maximum nfev must be positive")
    context, evaluation = _canonical_open_evaluation()
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": "read-only exact-next-step solver-efficiency baseline",
        "inner_radius_rg": INNER_RADIUS_RG,
        "maximum_nfev": arguments.maximum_nfev,
        "jacobian_mode": "sparse_forward",
        "runs": [],
    }
    for n_cells in dict.fromkeys(arguments.meshes):
        report["runs"].append(
            _run_mesh(
                context,
                evaluation,
                n_cells,
                arguments.maximum_nfev,
            )
        )
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
