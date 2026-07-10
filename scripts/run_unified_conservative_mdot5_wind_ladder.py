"""Add an energy-limited wind to a certified Mdot=5 compact-stream root."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PhysicalTransportClosure,
    conservative_residual_audit,
    conservative_residual_profile,
    solve_conservative_disk,
    unpack_conservative_state,
)
import run_unified_conservative_mdot5_stream_ladder as stream_ladder


ROOT = Path(__file__).resolve().parents[1]
STREAM_CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_mdot5_stream_ladder"
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_mdot5_wind_ladder"
OUTPUT = ROOT / "outputs/tables/unified_conservative_mdot5_wind_ladder.json"
N_NODES = int(os.environ.get("IMBH_CONSERVATIVE_M5_WIND_N", "384"))
STREAM_FRACTION = float(os.environ.get("IMBH_CONSERVATIVE_M5_WIND_STREAM_FRACTION", "0.3"))
ETA_E = float(os.environ.get("IMBH_CONSERVATIVE_M5_WIND_ETA_E", "98.125"))
EPSILON_VALUES = tuple(
    float(piece)
    for piece in os.environ.get(
        "IMBH_CONSERVATIVE_M5_WIND_EPSILON", "0,0.02,0.05,0.1,0.2,0.3,0.5"
    ).split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_CONSERVATIVE_M5_WIND_MAX_NFEV", "1300"))
PASSES = int(os.environ.get("IMBH_CONSERVATIVE_M5_WIND_PASSES", "4"))
SCOUT_TOLERANCE = float(os.environ.get("IMBH_CONSERVATIVE_M5_WIND_SCOUT_TOLERANCE", "5e-5"))
START_EPSILON_RAW = os.environ.get("IMBH_CONSERVATIVE_M5_WIND_START_EPSILON", "").strip()


def _starting_problem():
    stream_ladder.TARGET_N = N_NODES
    state, params = stream_ladder._truncate_to_minidisk(*stream_ladder._base_problem())
    safe_fs = str(STREAM_FRACTION).replace(".", "p")
    path = STREAM_CHECKPOINT_DIR / f"mdot5_fs{safe_fs}_Rout335_N{N_NODES}.npz"
    with np.load(path) as data:
        state = np.asarray(data["x"], dtype=float)
        grid = tuple(float(value) for value in np.asarray(data["custom_grid_xi"], dtype=float))
    params = replace(
        params,
        disk=replace(
            params.disk,
            custom_grid_xi=grid,
            stream_source_fraction=STREAM_FRACTION,
            wind_energy_limited_epsilon=0.0,
            wind_eddington_chi=0.99,
            wind_activation_width_fraction=0.005,
        ),
        closure=PhysicalTransportClosure(
            stream_circularization_radius=stream_ladder._physical_circularization_radius(),
            wind_angular_momentum_factor=1.0,
            wind_launch_energy_multiplier=ETA_E,
        ),
        max_nfev=MAX_NFEV,
    )
    return state, params


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
    state, params = _starting_problem()
    previous_epsilon = 0.0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if START_EPSILON_RAW:
        previous_epsilon = float(START_EPSILON_RAW)
        safe = str(previous_epsilon).replace(".", "p")
        safe_fs = str(STREAM_FRACTION).replace(".", "p")
        path = CHECKPOINT_DIR / f"mdot5_fs{safe_fs}_eps{safe}_eta{ETA_E:g}_N{N_NODES}.npz"
        with np.load(path) as data:
            state = np.asarray(data["x"], dtype=float)
            params = replace(
                params,
                disk=replace(
                    params.disk,
                    custom_grid_xi=tuple(float(value) for value in data["custom_grid_xi"]),
                    wind_energy_limited_epsilon=previous_epsilon,
                ),
            )

    rows: list[dict[str, object]] = []
    for epsilon_w in EPSILON_VALUES:
        params = replace(
            params,
            disk=replace(params.disk, wind_energy_limited_epsilon=float(epsilon_w)),
        )
        initial = conservative_residual_audit(state, params)
        final = initial
        passes: list[dict[str, object]] = []
        changed = not np.isclose(epsilon_w, previous_epsilon, rtol=0.0, atol=1.0e-14)
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
            "stream_fraction": STREAM_FRACTION,
            "epsilon_w": float(epsilon_w),
            "eta_E": ETA_E,
            "initial": asdict(initial),
            "final": asdict(final),
            "accepted_exploratory": final.maximum <= 3.0e-5,
            "accepted_preferred": final.maximum <= 1.0e-5,
            "continued_as_scout": final.maximum <= SCOUT_TOLERANCE,
            "summary": _summary(state, params),
            "passes": passes,
        }
        rows.append(row)
        safe = str(epsilon_w).replace(".", "p")
        safe_fs = str(STREAM_FRACTION).replace(".", "p")
        np.savez_compressed(
            CHECKPOINT_DIR / f"mdot5_fs{safe_fs}_eps{safe}_eta{ETA_E:g}_N{N_NODES}.npz",
            x=state,
            custom_grid_xi=np.asarray(params.disk.custom_grid_xi, dtype=float),
            row_json=np.asarray(json.dumps(row, sort_keys=True)),
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        if not row["continued_as_scout"]:
            break
        previous_epsilon = float(epsilon_w)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
