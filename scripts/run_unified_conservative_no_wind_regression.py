"""Staged no-wind regression for the unified conservative solver."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PhysicalTransportClosure,
    TransonicSlimParams,
    conservative_residual_audit,
    conservative_seed_from_legacy,
    remap_profile_to_new_sonic_grid,
    solve_conservative_disk,
    transonic_profile_from_state_vector,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results/canonical/no_wind_mdot5/state.npz"
OUTPUT = ROOT / "outputs/tables/unified_conservative_no_wind_regression.json"
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_no_wind_regression"
N_VALUES = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_CONSERVATIVE_NO_WIND_N", "64,96,128").split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_CONSERVATIVE_NO_WIND_MAX_NFEV", "120"))
PASSES = int(os.environ.get("IMBH_CONSERVATIVE_NO_WIND_PASSES", "1"))
SONIC_MODE = os.environ.get("IMBH_CONSERVATIVE_NO_WIND_SONIC_MODE", "legacy").strip().lower()
SONIC_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_NO_WIND_SONIC_WEIGHT", "1"))
JACOBIAN_STEP_RAW = os.environ.get("IMBH_CONSERVATIVE_NO_WIND_JACOBIAN_STEP", "").strip()
RESUME = os.environ.get("IMBH_CONSERVATIVE_NO_WIND_RESUME", "0").strip().lower() in {
    "1", "true", "yes", "on"
}


def load_anchor() -> tuple[np.ndarray, TransonicSlimParams]:
    fiducial = FiducialParams()
    with np.load(CHECKPOINT) as data:
        grid = np.asarray(data["custom_grid_xi"], dtype=float)
        slopes = np.asarray(data["outer_match_log_slopes"], dtype=float)
        params = TransonicSlimParams(
            M2_g=fiducial.M2_g,
            Mdot_g_s=float(data["ratio"]) * eddington_mdot(fiducial.M2_g),
            alpha=0.01,
            mu_stress=0.0,
            stress_factor=1.0,
            R_out_rg=float(data["R_out_rg"]),
            n_nodes=int(data["n_nodes"]),
            grid_power=float(data["grid_power"]),
            custom_grid_xi=tuple(float(value) for value in grid),
            outer_closure=str(np.asarray(data["outer_closure"]).item()),
            outer_match_log_slopes=(float(slopes[0]), float(slopes[1])),
            residual_tol=1.0e-8,
            max_nfev=1,
        )
        z = np.asarray(data["z"], dtype=float)
    return z, params


def run() -> list[dict[str, object]]:
    anchor, anchor_params = load_anchor()
    anchor_profile = transonic_profile_from_state_vector(anchor, anchor_params)
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for n_nodes in N_VALUES:
        disk = replace(
            anchor_params,
            n_nodes=int(n_nodes),
            custom_grid_xi=None,
            grid_power=0.6,
            max_nfev=1,
        )
        legacy_seed = remap_profile_to_new_sonic_grid(anchor_profile, disk, method="pchip")
        closure = PhysicalTransportClosure(
            stream_circularization_radius=0.8 * disk.R_out,
        )
        seed, solver_params = conservative_seed_from_legacy(legacy_seed, disk, closure)
        solver_params = replace(
            solver_params,
            residual_tolerance=1.0e-5,
            max_nfev=MAX_NFEV,
            sonic_mode=SONIC_MODE,
            sonic_weight=SONIC_WEIGHT,
            jacobian_rel_step=float(JACOBIAN_STEP_RAW) if JACOBIAN_STEP_RAW else None,
        )
        checkpoint_path = CHECKPOINT_DIR / f"mdot5_N{n_nodes}.npz"
        if RESUME and checkpoint_path.exists():
            with np.load(checkpoint_path) as checkpoint:
                candidate = np.asarray(checkpoint["x"], dtype=float)
            if candidate.shape == seed.shape:
                seed = candidate
        initial = conservative_residual_audit(seed, solver_params)
        solved = solve_conservative_disk(seed, solver_params)
        pass_rows = [
            {
                "pass": 1,
                "nfev": int(solved.nfev),
                "accepted": bool(solved.accepted),
                "final": asdict(solved.final_audit),
            }
        ]
        for pass_index in range(2, max(PASSES, 1) + 1):
            if solved.accepted:
                break
            solved = solve_conservative_disk(solved.x, solver_params)
            pass_rows.append(
                {
                    "pass": pass_index,
                    "nfev": int(solved.nfev),
                    "accepted": bool(solved.accepted),
                    "final": asdict(solved.final_audit),
                }
            )
        row: dict[str, object] = {
            "N": int(n_nodes),
            "initial": asdict(initial),
            "final": asdict(solved.final_audit),
            "success": bool(solved.success),
            "accepted": bool(solved.accepted),
            "nfev": int(solved.nfev),
            "cost": float(solved.cost),
            "optimality": float(solved.optimality),
            "message": solved.message,
            "passes": pass_rows,
        }
        rows.append(row)
        np.savez_compressed(
            checkpoint_path,
            x=solved.x,
            row_json=np.asarray(json.dumps(row, sort_keys=True)),
        )
        print(json.dumps(row, sort_keys=True), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
