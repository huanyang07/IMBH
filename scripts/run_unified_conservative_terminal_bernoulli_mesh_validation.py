"""Nested-grid validation of the B_infinity=0.02 c^2 terminal-wind root."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    conservative_residual_audit,
    conservative_wind_escape_profile,
    nested_refined_conservative_grid,
    remap_conservative_state,
    solve_conservative_disk_block_jacobian,
)
import run_unified_conservative_terminal_bernoulli_ladder as ladder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / (
    "outputs/checkpoints/unified_conservative_terminal_bernoulli/"
    "mdot5_Binf_0p02c2_cap0.3_N426.npz"
)
SOURCE = Path(os.environ.get("IMBH_TERMINAL_BERNOULLI_MESH_SOURCE", str(DEFAULT_SOURCE)))
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_terminal_bernoulli"
OUTPUT = ROOT / "outputs/tables/unified_conservative_terminal_bernoulli_mesh_validation.json"
TARGETS = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_TERMINAL_BERNOULLI_MESH_TARGETS", "512").split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_TERMINAL_BERNOULLI_MESH_MAX_NFEV", "100"))


def run() -> list[dict[str, object]]:
    with np.load(SOURCE) as data:
        state = np.asarray(data["x"], dtype=float)
        grid = np.asarray(data["custom_grid_xi"], dtype=float)
        source_row = (
            json.loads(str(np.asarray(data["row_json"]).item()))
            if "row_json" in data.files
            else None
        )
    params = ladder._params(grid, 0.02)
    rows: list[dict[str, object]] = []
    if OUTPUT.exists():
        rows.extend(json.loads(OUTPUT.read_text()))
    if isinstance(source_row, dict) and "N" in source_row:
        rows.append(source_row)
    for target_n in TARGETS:
        target_grid = nested_refined_conservative_grid(state, params, target_n=target_n)
        target_disk = replace(
            params.disk,
            n_nodes=target_n,
            custom_grid_xi=target_grid,
        )
        seed, target_params = remap_conservative_state(
            state, params, target_disk, method="pchip"
        )
        initial = conservative_residual_audit(seed, target_params)
        solved = solve_conservative_disk_block_jacobian(
            seed,
            target_params,
            max_nfev=MAX_NFEV,
        )
        state = solved.x
        params = target_params
        wind = conservative_wind_escape_profile(state, params)
        row = {
            "N": target_n,
            "initial": asdict(initial),
            "final": asdict(solved.final_audit),
            "accepted": bool(solved.final_audit.maximum <= 3.0e-5),
            "nfev": solved.nfev,
            "message": solved.message,
            "wind_over_mdot_inner": float(
                np.sum(wind["wind_mass"]) / params.flux_scales.mdot
            ),
            "cap_active_intervals": int(np.count_nonzero(wind["wind_cap_active"])),
        }
        rows = [old for old in rows if int(old["N"]) != target_n]
        rows.append(row)
        np.savez_compressed(
            CHECKPOINT_DIR / f"mdot5_Binf_0p02c2_cap0.3_N{target_n}.npz",
            x=np.asarray(state, dtype=float),
            custom_grid_xi=np.asarray(target_grid, dtype=float),
            target_terminal_bernoulli_over_c2=np.asarray(0.02),
            row_json=np.asarray(json.dumps(row, sort_keys=True)),
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        if not row["accepted"]:
            break
    rows.sort(key=lambda row: int(row["N"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
