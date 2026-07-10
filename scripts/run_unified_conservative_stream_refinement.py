"""Refine the physical stream-fed conservative checkpoint."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PhysicalTransportClosure,
    conservative_residual_audit,
    conservative_seed_from_legacy,
    remap_conservative_state,
    remap_profile_to_new_sonic_grid,
    solve_conservative_disk,
    transonic_profile_from_state_vector,
)
from run_unified_conservative_stream_regression import (
    load_anchor,
    physical_circularization_radius,
    residual_aware_grid,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_stream_regression"
OUTPUT = ROOT / "outputs/tables/unified_conservative_stream_refinement.json"
START_N = int(os.environ.get("IMBH_CONSERVATIVE_STREAM_REFINE_START_N", "128"))
TARGETS = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_CONSERVATIVE_STREAM_REFINE_N", "192,256").split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_CONSERVATIVE_STREAM_REFINE_MAX_NFEV", "220"))
PASSES = int(os.environ.get("IMBH_CONSERVATIVE_STREAM_REFINE_PASSES", "2"))
ENERGY_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_STREAM_REFINE_ENERGY_WEIGHT", "5"))


def start_state():
    anchor, old_params = load_anchor()
    profile = transonic_profile_from_state_vector(anchor, old_params)
    disk = replace(
        old_params,
        n_nodes=START_N,
        custom_grid_xi=residual_aware_grid(old_params, START_N),
        grid_power=0.6,
        max_nfev=1,
    )
    legacy = remap_profile_to_new_sonic_grid(profile, disk, method="pchip")
    closure = PhysicalTransportClosure(
        stream_circularization_radius=physical_circularization_radius(),
    )
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    params = replace(
        params,
        disk=replace(params.disk, stream_torque_delta_l_fraction=0.0),
        residual_tolerance=1.0e-5,
        max_nfev=MAX_NFEV,
        energy_balance_weight=ENERGY_WEIGHT,
    )
    checkpoint = CHECKPOINT_DIR / f"mdot2_fs080_N{START_N}.npz"
    with np.load(checkpoint) as data:
        state = np.asarray(data["x"], dtype=float)
    if state.shape != seed.shape:
        raise ValueError("starting stream checkpoint has the wrong shape")
    return state, params, old_params


def run() -> list[dict[str, object]]:
    state, params, mesh_reference = start_state()
    rows: list[dict[str, object]] = []
    for target_n in TARGETS:
        disk = replace(
            params.disk,
            n_nodes=target_n,
            custom_grid_xi=residual_aware_grid(mesh_reference, target_n),
            grid_power=0.6,
            max_nfev=1,
        )
        state, params = remap_conservative_state(state, params, disk, method="pchip")
        params = replace(
            params,
            max_nfev=MAX_NFEV,
            residual_tolerance=1.0e-5,
            energy_balance_weight=ENERGY_WEIGHT,
        )
        initial = conservative_residual_audit(state, params)
        pass_rows = []
        solved = None
        for pass_index in range(1, max(PASSES, 1) + 1):
            solved = solve_conservative_disk(state, params)
            state = solved.x
            pass_rows.append(
                {
                    "pass": pass_index,
                    "nfev": solved.nfev,
                    "accepted": solved.accepted,
                    "final": asdict(solved.final_audit),
                }
            )
            if solved.accepted:
                break
        assert solved is not None
        row = {
            "N": target_n,
            "initial": asdict(initial),
            "final": asdict(solved.final_audit),
            "accepted": solved.accepted,
            "passes": pass_rows,
        }
        rows.append(row)
        np.savez_compressed(
            CHECKPOINT_DIR / f"mdot2_fs080_N{target_n}.npz",
            x=state,
            row_json=np.asarray(json.dumps(row, sort_keys=True)),
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        if not solved.accepted:
            break
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
