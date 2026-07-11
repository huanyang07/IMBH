"""Nested N426 -> N512 -> N640 validation of the exact-source eta=8 anchor."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    conservative_residual_audit,
    nested_refined_conservative_grid,
    remap_conservative_state,
    solve_conservative_disk_block_jacobian,
)
import run_unified_conservative_block_eta_continuation as continuation


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_block_eta"
OUTPUT = ROOT / "outputs/tables/unified_conservative_eta8_mesh_validation.json"
TARGETS = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_ETA8_MESH_TARGETS", "512,640").split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_ETA8_MESH_MAX_NFEV", "40"))


def run() -> list[dict[str, object]]:
    source = CHECKPOINT_DIR / "mdot5_eta8_block_mass5_N426.npz"
    with np.load(source) as data:
        state = np.asarray(data["x"], dtype=float)
        grid = np.asarray(data["custom_grid_xi"], dtype=float)
    params = continuation._problem(grid, 8.0)
    rows: list[dict[str, object]] = []

    for target_n in TARGETS:
        target_grid = nested_refined_conservative_grid(
            state, params, target_n=target_n
        )
        target_disk = replace(
            params.disk,
            n_nodes=target_n,
            custom_grid_xi=target_grid,
        )
        seed, target_params = remap_conservative_state(
            state, params, target_disk, method="pchip"
        )
        target_params = replace(
            target_params,
            mass_weight=continuation.MASS_WEIGHT,
            closure=replace(
                target_params.closure,
                wind_launch_energy_multiplier=8.0,
            ),
            residual_tolerance=3.0e-5,
        )
        initial = conservative_residual_audit(seed, target_params)
        solved = solve_conservative_disk_block_jacobian(
            seed,
            target_params,
            max_nfev=MAX_NFEV,
        )
        state = solved.x
        params = target_params
        row = {
            "N": target_n,
            "initial": asdict(initial),
            "final": asdict(solved.final_audit),
            "accepted": solved.final_audit.maximum <= 3.0e-5,
            "nfev": solved.nfev,
            "message": solved.message,
            "summary": continuation._summary(state, params),
        }
        rows.append(row)
        np.savez_compressed(
            CHECKPOINT_DIR / f"mdot5_eta8_block_mass5_N{target_n}.npz",
            x=state,
            eta_E=np.asarray(8.0),
            custom_grid_xi=np.asarray(target_grid, dtype=float),
            row_json=np.asarray(json.dumps(row, sort_keys=True)),
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        if not row["accepted"]:
            break

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
