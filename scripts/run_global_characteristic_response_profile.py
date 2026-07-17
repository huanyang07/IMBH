"""Profile one immutable low-throughput characteristic-boundary Jacobian."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalCellSources,
    advance_global_backward_euler,
    load_global_adaptive_restart,
)

from run_global_low_throughput_remnant import _physical_source_and_roche
from run_global_physical_open_preflight import _canonical_open_evaluation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "outputs/checkpoints/global_low_throughput_remnant/projected_N64.npz"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dt-loading-fraction", type=float, default=1.0e-8
    )
    parser.add_argument("--maximum-nfev", type=int, default=1)
    parser.add_argument(
        "--inner-characteristic-cache-size", type=int, default=0
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    input_path = arguments.input
    output_path = arguments.output
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if arguments.dt_loading_fraction <= 0.0:
        raise ValueError("dt loading fraction must be positive")
    if arguments.maximum_nfev < 1:
        raise ValueError("maximum nfev must be positive")
    if arguments.inner_characteristic_cache_size < 0:
        raise ValueError("inner characteristic cache size must be non-negative")

    grid, restart = load_global_adaptive_restart(input_path)
    context, _evaluation = _canonical_open_evaluation()
    mass = context.base.inner_params.M2_g
    stream, stream_rate, provider = _physical_source_and_roche(grid, mass)
    loading_time = float(np.sum(restart.reference_state.mass) / stream_rate)
    dt = float(arguments.dt_loading_fraction * loading_time)
    state_before = {
        name: np.array(getattr(restart.state, name), copy=True)
        for name in (
            "mass",
            "radial_momentum",
            "angular_momentum",
            "total_energy",
        )
    }

    wall_start = perf_counter()
    result = advance_global_backward_euler(
        grid,
        restart.state,
        mass,
        dt,
        alpha=context.base.alpha,
        reference_state=restart.reference_state,
        boundary_mode="characteristic_inner_roche_outer",
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=True,
        external_sources=GlobalCellSources.zeros(grid.centers.size),
        jacobian_mode="sparse_forward",
        outer_overflow_provider=provider,
        max_nfev=arguments.maximum_nfev,
        specific_mechanical_energy_correction=(
            restart.mechanical_reference.specific_offset
        ),
        inner_characteristic_cache_size=(
            arguments.inner_characteristic_cache_size
        ),
    )
    measured_wall_seconds = float(perf_counter() - wall_start)
    input_unchanged = all(
        np.array_equal(getattr(restart.state, name), values)
        for name, values in state_before.items()
    )
    report = {
        "input": str(input_path),
        "n_cells": int(grid.centers.size),
        "source_enabled": False,
        "dt_seconds": dt,
        "dt_over_loading_time": arguments.dt_loading_fraction,
        "maximum_nfev": arguments.maximum_nfev,
        "inner_characteristic_cache_size": (
            arguments.inner_characteristic_cache_size
        ),
        "accepted": bool(result.accepted),
        "message": result.message,
        "maximum_scaled_residual": result.maximum_scaled_residual,
        "maximum_storage_scaled_ledger_defect": (
            result.maximum_storage_scaled_ledger_defect
        ),
        "measured_wall_seconds": measured_wall_seconds,
        "jacobian_audit": (
            None
            if result.jacobian_audit is None
            else asdict(result.jacobian_audit)
        ),
        "nonlinear_solve_audit": (
            None
            if result.nonlinear_solve_audit is None
            else asdict(result.nonlinear_solve_audit)
        ),
        "input_state_unchanged": input_unchanged,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(output_path)


if __name__ == "__main__":
    main()
