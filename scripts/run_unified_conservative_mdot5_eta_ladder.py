"""Lower the wind launch energy on the certified Mdot=5 wind branch."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    conservative_residual_audit,
    conservative_residual_profile,
    solve_conservative_disk,
    unpack_conservative_state,
)
import run_unified_conservative_mdot5_wind_ladder as wind_ladder


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs/checkpoints/unified_conservative_mdot5_wind_ladder"
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_mdot5_eta_ladder"
OUTPUT = ROOT / "outputs/tables/unified_conservative_mdot5_eta_ladder.json"
N_NODES = int(os.environ.get("IMBH_CONSERVATIVE_M5_ETA_N", "384"))
EPSILON_W = float(os.environ.get("IMBH_CONSERVATIVE_M5_ETA_EPSILON", "0.2"))
ETA_VALUES = tuple(
    float(piece)
    for piece in os.environ.get(
        "IMBH_CONSERVATIVE_M5_ETA_VALUES",
        "98.125,90,80,70,60,50,40,30,20,10,7,5,3,2,1",
    ).split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_CONSERVATIVE_M5_ETA_MAX_NFEV", "1600"))
PASSES = int(os.environ.get("IMBH_CONSERVATIVE_M5_ETA_PASSES", "4"))
SCOUT_TOLERANCE = float(os.environ.get("IMBH_CONSERVATIVE_M5_ETA_SCOUT_TOLERANCE", "1e-4"))
START_ETA_RAW = os.environ.get("IMBH_CONSERVATIVE_M5_ETA_START", "").strip()


def _summary(state, params) -> dict[str, float]:
    _logu, _logT, F, _j, _epsilon, logR_son, _logR = unpack_conservative_state(state, params)
    profile = conservative_residual_profile(state, params)
    score = np.max(
        np.vstack(
            [
                np.abs(profile[name])
                for name in ("radial", "mass", "angular_momentum", "energy", "energy_compatibility")
            ]
        ),
        axis=0,
    )
    peak = int(np.argmax(score))
    return {
        "F_inner": float(F[0]),
        "F_outer": float(F[-1]),
        "Rson_rg": float(np.exp(logR_son) / params.disk.r_g),
        "peak_interval_residual": float(score[peak]),
        "peak_interval_R_rg": float(profile["R_mid_rg"][peak]),
    }


def run() -> list[dict[str, object]]:
    wind_ladder.N_NODES = N_NODES
    state, params = wind_ladder._starting_problem()
    safe_eps = str(EPSILON_W).replace(".", "p")
    source = SOURCE_DIR / f"mdot5_fs0p3_eps{safe_eps}_eta98.125_N{N_NODES}.npz"
    with np.load(source) as data:
        state = np.asarray(data["x"], dtype=float)
        grid = tuple(float(value) for value in np.asarray(data["custom_grid_xi"], dtype=float))
    params = replace(
        params,
        disk=replace(
            params.disk,
            custom_grid_xi=grid,
            wind_energy_limited_epsilon=EPSILON_W,
        ),
        max_nfev=MAX_NFEV,
    )

    previous_eta = 98.125
    if START_ETA_RAW:
        previous_eta = float(START_ETA_RAW)
        safe_eta = str(previous_eta).replace(".", "p")
        restart = CHECKPOINT_DIR / f"mdot5_fs0p3_eps{safe_eps}_eta{safe_eta}_N{N_NODES}.npz"
        with np.load(restart) as data:
            state = np.asarray(data["x"], dtype=float)
            params = replace(
                params,
                disk=replace(
                    params.disk,
                    custom_grid_xi=tuple(float(value) for value in data["custom_grid_xi"]),
                ),
                closure=replace(params.closure, wind_launch_energy_multiplier=previous_eta),
            )

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for eta_E in ETA_VALUES:
        params = replace(
            params,
            closure=replace(params.closure, wind_launch_energy_multiplier=float(eta_E)),
        )
        initial = conservative_residual_audit(state, params)
        final = initial
        passes: list[dict[str, object]] = []
        changed = not np.isclose(eta_E, previous_eta, rtol=0.0, atol=1.0e-14)
        if changed or initial.maximum > 3.0e-5:
            for pass_index in range(1, PASSES + 1):
                solved = solve_conservative_disk(state, params)
                state = solved.x
                final = solved.final_audit
                passes.append(
                    {"pass": pass_index, "nfev": solved.nfev, "final": asdict(final)}
                )
                if final.maximum <= 3.0e-5:
                    break
        row: dict[str, object] = {
            "eta_E": float(eta_E),
            "epsilon_w": EPSILON_W,
            "initial": asdict(initial),
            "final": asdict(final),
            "accepted_exploratory": final.maximum <= 3.0e-5,
            "accepted_preferred": final.maximum <= 1.0e-5,
            "continued_as_scout": final.maximum <= SCOUT_TOLERANCE,
            "summary": _summary(state, params),
            "passes": passes,
        }
        rows.append(row)
        safe_eta = str(eta_E).replace(".", "p")
        np.savez_compressed(
            CHECKPOINT_DIR / f"mdot5_fs0p3_eps{safe_eps}_eta{safe_eta}_N{N_NODES}.npz",
            x=state,
            custom_grid_xi=np.asarray(params.disk.custom_grid_xi, dtype=float),
            row_json=np.asarray(json.dumps(row, sort_keys=True)),
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        if not row["continued_as_scout"]:
            break
        previous_eta = float(eta_E)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
