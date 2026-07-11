"""Continue the eta=8 anchor into a target-terminal-Bernoulli wind closure."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    conservative_residual_audit,
    conservative_wind_escape_profile,
    solve_conservative_disk_block_jacobian,
)
import run_unified_conservative_block_eta_continuation as continuation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / (
    "outputs/checkpoints/unified_conservative_block_eta/"
    "mdot5_eta8_block_mass5_N426.npz"
)
SOURCE = Path(os.environ.get("IMBH_TERMINAL_BERNOULLI_SOURCE", str(DEFAULT_SOURCE)))
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_terminal_bernoulli"
OUTPUT_LABEL = os.environ.get("IMBH_TERMINAL_BERNOULLI_OUTPUT_LABEL", "ladder")
OUTPUT = ROOT / f"outputs/tables/unified_conservative_terminal_bernoulli_{OUTPUT_LABEL}.json"
TARGETS_C2 = tuple(
    float(piece)
    for piece in os.environ.get(
        "IMBH_TERMINAL_BERNOULLI_TARGETS_C2", "0.10,0.08,0.06,0.04,0.02,0.01,0.005,0"
    ).split(",")
    if piece.strip()
)
MASS_CAP = float(os.environ.get("IMBH_TERMINAL_BERNOULLI_MASS_CAP", "0.3"))
MAX_NFEV = int(os.environ.get("IMBH_TERMINAL_BERNOULLI_MAX_NFEV", "60"))


def _params(grid: np.ndarray, target_c2: float):
    base = continuation._problem(grid, 8.0)
    return replace(
        base,
        closure=replace(
            base.closure,
            wind_launch_mode="terminal_bernoulli",
            wind_terminal_bernoulli=float(target_c2 * C**2),
            wind_mass_loading_cap_per_log_radius=MASS_CAP,
        ),
    )


def _wind_summary(state: np.ndarray, params, target_c2: float) -> dict[str, float | int]:
    profile = conservative_wind_escape_profile(
        state,
        params,
        target_terminal_bernoulli=float(target_c2 * C**2),
    )
    wind_mass = np.asarray(profile["wind_mass"], dtype=float)
    total = float(np.sum(wind_mass))
    raw = np.asarray(profile["wind_raw_prime"], dtype=float)
    effective = np.asarray(profile["wind_prime"], dtype=float)
    return {
        "wind_over_mdot_inner": total / params.flux_scales.mdot,
        "cap_active_intervals": int(np.count_nonzero(profile["wind_cap_active"])),
        "maximum_raw_over_effective": float(
            np.max(raw / np.maximum(effective, np.finfo(float).tiny))
        ),
        "maximum_terminal_margin_over_c2": float(
            np.max(np.abs(profile["terminal_margin"])) / C**2
        ),
    }


def run() -> list[dict[str, object]]:
    with np.load(SOURCE) as data:
        state = np.asarray(data["x"], dtype=float)
        grid = np.asarray(data["custom_grid_xi"], dtype=float)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for target_c2 in TARGETS_C2:
        params = _params(grid, target_c2)
        initial = conservative_residual_audit(state, params)
        solved = solve_conservative_disk_block_jacobian(
            state,
            params,
            max_nfev=MAX_NFEV,
        )
        state = solved.x
        accepted = bool(solved.final_audit.maximum <= 3.0e-5)
        row = {
            "target_terminal_bernoulli_over_c2": target_c2,
            "mass_cap_per_log_radius": MASS_CAP,
            "initial": asdict(initial),
            "final": asdict(solved.final_audit),
            "accepted": accepted,
            "nfev": solved.nfev,
            "message": solved.message,
            "wind": _wind_summary(state, params, target_c2),
        }
        rows.append(row)
        label = str(target_c2).replace(".", "p")
        np.savez_compressed(
            CHECKPOINT_DIR / f"mdot5_Binf_{label}c2_cap{MASS_CAP:g}_N426.npz",
            x=np.asarray(state, dtype=float),
            custom_grid_xi=np.asarray(grid, dtype=float),
            target_terminal_bernoulli_over_c2=np.asarray(target_c2),
            row_json=np.asarray(json.dumps(row, sort_keys=True)),
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        if not accepted:
            break

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
