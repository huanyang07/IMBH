"""Add an energy-limited wind to the compact physical stream branch."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PhysicalTransportClosure,
    conservative_residual_audit,
    remap_conservative_state,
    solve_conservative_disk,
    unpack_conservative_state,
)
from run_unified_conservative_compact_source_homotopy import start_state


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_wind_continuation"
SOURCE_CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_stream_regression"
OUTPUT = ROOT / "outputs/tables/unified_conservative_wind_continuation.json"
N_NODES = int(os.environ.get("IMBH_CONSERVATIVE_WIND_N", "128"))
EPSILON_VALUES = tuple(
    float(piece)
    for piece in os.environ.get(
        "IMBH_CONSERVATIVE_WIND_EPSILON", "0,0.1,0.3,0.5,0.7,0.9,0.99,0.998"
    ).split(",")
    if piece.strip()
)
LAUNCH_MULTIPLIER = float(os.environ.get("IMBH_CONSERVATIVE_WIND_ETA_E", "98.125"))
MAX_NFEV = int(os.environ.get("IMBH_CONSERVATIVE_WIND_MAX_NFEV", "180"))
PASSES = int(os.environ.get("IMBH_CONSERVATIVE_WIND_PASSES", "2"))
ENERGY_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_WIND_ENERGY_WEIGHT", "5"))
MASS_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_WIND_MASS_WEIGHT", "1"))
ANGULAR_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_WIND_ANGULAR_WEIGHT", "1"))
ENERGY_FLUX_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_WIND_ENERGY_FLUX_WEIGHT", "1"))
INNER_MASS_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_WIND_INNER_MASS_WEIGHT", "1"))
FORCE_CORRECT = os.environ.get("IMBH_CONSERVATIVE_WIND_FORCE_CORRECT", "1") != "0"
SCOUT_TOLERANCE = float(os.environ.get("IMBH_CONSERVATIVE_WIND_SCOUT_TOLERANCE", "3e-5"))
START_EPSILON_RAW = os.environ.get("IMBH_CONSERVATIVE_WIND_START_EPSILON", "").strip()
SONIC_MODE = os.environ.get("IMBH_CONSERVATIVE_WIND_SONIC_MODE", "legacy").strip().lower()
SONIC_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_WIND_SONIC_WEIGHT", "1"))
JACOBIAN_STEP_RAW = os.environ.get("IMBH_CONSERVATIVE_WIND_JACOBIAN_STEP", "").strip()


def _starting_problem():
    state, params = start_state()
    source_n = int(params.disk.n_nodes)
    endpoint = SOURCE_CHECKPOINT_DIR / f"mdot2_fs080_compact_blend1p0_N{source_n}.npz"
    with np.load(endpoint) as data:
        state = np.asarray(data["x"], dtype=float)
    if N_NODES != source_n:
        old_grid = np.asarray(params.disk.custom_grid_xi, dtype=float)
        new_grid = np.interp(
            np.linspace(0.0, 1.0, N_NODES),
            np.linspace(0.0, 1.0, source_n),
            old_grid,
        )
        state, params = remap_conservative_state(
            state,
            params,
            replace(
                params.disk,
                n_nodes=N_NODES,
                custom_grid_xi=tuple(float(value) for value in new_grid),
            ),
        )
    params = replace(
        params,
        closure=PhysicalTransportClosure(
            stream_circularization_radius=params.closure.stream_circularization_radius,
            stream_specific_angular_momentum=params.closure.stream_specific_angular_momentum,
            stream_specific_energy=params.closure.stream_specific_energy,
            wind_angular_momentum_factor=1.0,
            wind_launch_energy_multiplier=LAUNCH_MULTIPLIER,
        ),
        disk=replace(
            params.disk,
            stream_source_shape="compact_c2",
            stream_source_shape_blend=1.0,
            wind_energy_limited_epsilon=0.0,
            wind_eddington_chi=0.99,
            wind_activation_width_fraction=0.005,
        ),
        residual_tolerance=3.0e-5,
        max_nfev=MAX_NFEV,
        mass_weight=MASS_WEIGHT,
        angular_momentum_weight=ANGULAR_WEIGHT,
        energy_flux_weight=ENERGY_FLUX_WEIGHT,
        energy_balance_weight=ENERGY_WEIGHT,
        inner_mass_weight=INNER_MASS_WEIGHT,
        sonic_mode=SONIC_MODE,
        sonic_weight=SONIC_WEIGHT,
        jacobian_rel_step=float(JACOBIAN_STEP_RAW) if JACOBIAN_STEP_RAW else None,
    )
    return state, params


def _flux_summary(state: np.ndarray, params) -> dict[str, float]:
    _logu, _logT, F, j, epsilon, logR_son, _logR = unpack_conservative_state(state, params)
    return {
        "F_inner": float(F[0]),
        "F_outer": float(F[-1]),
        "integrated_wind_minus_stream": float(F[-1] - F[0]),
        "j_inner": float(j[0]),
        "j_outer": float(j[-1]),
        "epsilon_inner": float(epsilon[0]),
        "epsilon_outer": float(epsilon[-1]),
        "Rson_rg": float(np.exp(logR_son) / params.disk.r_g),
    }


def run() -> list[dict[str, object]]:
    state, params = _starting_problem()
    previous_epsilon = 0.0
    if START_EPSILON_RAW:
        start_epsilon = float(START_EPSILON_RAW)
        previous_epsilon = start_epsilon
        safe_start = str(start_epsilon).replace(".", "p")
        start_path = CHECKPOINT_DIR / f"mdot2_fs080_eps{safe_start}_eta{LAUNCH_MULTIPLIER:g}_N{N_NODES}.npz"
        with np.load(start_path) as data:
            candidate = np.asarray(data["x"], dtype=float)
            checkpoint_grid = (
                np.asarray(data["custom_grid_xi"], dtype=float)
                if "custom_grid_xi" in data.files
                else None
            )
        if candidate.shape != state.shape:
            raise ValueError("wind continuation restart checkpoint has the wrong shape")
        state = candidate
        if checkpoint_grid is not None:
            params = replace(
                params,
                disk=replace(
                    params.disk,
                    custom_grid_xi=tuple(float(value) for value in checkpoint_grid),
                ),
            )
        params = replace(
            params,
            disk=replace(params.disk, wind_energy_limited_epsilon=start_epsilon),
        )
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for epsilon_w in EPSILON_VALUES:
        params = replace(
            params,
            disk=replace(params.disk, wind_energy_limited_epsilon=float(epsilon_w)),
        )
        initial = conservative_residual_audit(state, params)
        final = initial
        pass_rows = []
        parameter_changed = not np.isclose(
            float(epsilon_w), previous_epsilon, rtol=0.0, atol=1.0e-14
        )
        if initial.maximum > 3.0e-5 or (FORCE_CORRECT and parameter_changed):
            for pass_index in range(1, max(PASSES, 1) + 1):
                solved = solve_conservative_disk(state, params)
                state = solved.x
                final = solved.final_audit
                pass_rows.append(
                    {
                        "pass": pass_index,
                        "nfev": solved.nfev,
                        "accepted_exploratory": final.maximum <= 3.0e-5,
                        "final": asdict(final),
                    }
                )
                if final.maximum <= 3.0e-5:
                    break
        row: dict[str, object] = {
            "epsilon_w": epsilon_w,
            "eta_E": LAUNCH_MULTIPLIER,
            "initial": asdict(initial),
            "final": asdict(final),
            "accepted_exploratory": final.maximum <= 3.0e-5,
            "accepted_preferred": final.maximum <= 1.0e-5,
            "continued_as_scout": final.maximum <= SCOUT_TOLERANCE,
            "flux": _flux_summary(state, params),
            "passes": pass_rows,
        }
        rows.append(row)
        safe = str(epsilon_w).replace(".", "p")
        np.savez_compressed(
            CHECKPOINT_DIR / f"mdot2_fs080_eps{safe}_eta{LAUNCH_MULTIPLIER:g}_N{N_NODES}.npz",
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
