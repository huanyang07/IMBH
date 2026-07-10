"""Refine an accepted unified conservative no-wind checkpoint in stages."""

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
from run_unified_conservative_no_wind_regression import load_anchor


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_no_wind_regression"
OUTPUT = ROOT / "outputs/tables/unified_conservative_grid_refinement.json"
START_N = int(os.environ.get("IMBH_CONSERVATIVE_REFINE_START_N", "64"))
TARGETS = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_CONSERVATIVE_REFINE_N", "96,128").split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_CONSERVATIVE_REFINE_MAX_NFEV", "220"))
PASSES = int(os.environ.get("IMBH_CONSERVATIVE_REFINE_PASSES", "3"))
ENERGY_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_REFINE_ENERGY_WEIGHT", "5"))
SONIC_MODE = os.environ.get("IMBH_CONSERVATIVE_REFINE_SONIC_MODE", "legacy").strip().lower()
SONIC_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_REFINE_SONIC_WEIGHT", "1"))
JACOBIAN_STEP_RAW = os.environ.get("IMBH_CONSERVATIVE_REFINE_JACOBIAN_STEP", "").strip()


def _start_state():
    anchor, anchor_params = load_anchor()
    profile = transonic_profile_from_state_vector(anchor, anchor_params)
    disk = replace(
        anchor_params,
        n_nodes=START_N,
        custom_grid_xi=None,
        grid_power=0.6,
        max_nfev=1,
    )
    legacy = remap_profile_to_new_sonic_grid(profile, disk, method="pchip")
    closure = PhysicalTransportClosure(stream_circularization_radius=0.8 * disk.R_out)
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    params = replace(
        params,
        energy_balance_weight=ENERGY_WEIGHT,
        sonic_mode=SONIC_MODE,
        sonic_weight=SONIC_WEIGHT,
        jacobian_rel_step=float(JACOBIAN_STEP_RAW) if JACOBIAN_STEP_RAW else None,
    )
    checkpoint = CHECKPOINT_DIR / f"mdot5_N{START_N}.npz"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    with np.load(checkpoint) as data:
        candidate = np.asarray(data["x"], dtype=float)
    if candidate.shape != seed.shape:
        raise ValueError("starting conservative checkpoint has the wrong shape")
    return candidate, params


def run() -> list[dict[str, object]]:
    state, params = _start_state()
    rows: list[dict[str, object]] = []
    for target_n in TARGETS:
        disk = replace(
            params.disk,
            n_nodes=target_n,
            custom_grid_xi=None,
            grid_power=0.6,
            max_nfev=1,
        )
        state, params = remap_conservative_state(state, params, disk, method="pchip")
        params = replace(params, max_nfev=MAX_NFEV, residual_tolerance=1.0e-5)
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
            "success": solved.success,
            "passes": pass_rows,
        }
        rows.append(row)
        np.savez_compressed(
            CHECKPOINT_DIR / f"mdot5_N{target_n}.npz",
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
