"""Block-Jacobian and bordered continuation for the exact-source Mdot=5 branch."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    conservative_eta_pseudo_arclength_step,
    conservative_residual_audit,
    conservative_transport_profile,
    solve_conservative_disk_block_jacobian,
    unpack_conservative_state,
)
import run_unified_conservative_mdot5_wind_ladder as wind_ladder


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / (
    "outputs/checkpoints/unified_conservative_source_band_certification/"
    "mdot5_eta10p0_source64_N426.npz"
)
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_block_eta"
OUTPUT = ROOT / "outputs/tables/unified_conservative_block_eta_continuation.json"
ANCHOR_MAX_NFEV = int(os.environ.get("IMBH_BLOCK_ETA_ANCHOR_MAX_NFEV", "40"))
ARC_MAX_NFEV = int(os.environ.get("IMBH_BLOCK_ETA_ARC_MAX_NFEV", "25"))
MAX_ARC_STEPS = int(os.environ.get("IMBH_BLOCK_ETA_MAX_STEPS", "4"))
MIN_ETA = float(os.environ.get("IMBH_BLOCK_ETA_MIN", "9.0"))
REUSE = os.environ.get("IMBH_BLOCK_ETA_REUSE", "1").strip() != "0"
MASS_WEIGHT = float(os.environ.get("IMBH_BLOCK_ETA_MASS_WEIGHT", "5.0"))
DIRECT_ETA_VALUES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_BLOCK_ETA_DIRECT_VALUES", "9,8.5,8").split(",")
    if piece.strip()
)


def _problem(grid: np.ndarray, eta_e: float):
    wind_ladder.N_NODES = 384
    _unused, params = wind_ladder._starting_problem()
    return replace(
        params,
        disk=replace(
            params.disk,
            n_nodes=int(grid.size),
            custom_grid_xi=tuple(float(value) for value in grid),
            wind_energy_limited_epsilon=0.2,
        ),
        closure=replace(params.closure, wind_launch_energy_multiplier=float(eta_e)),
        mass_weight=MASS_WEIGHT,
        residual_tolerance=3.0e-5,
    )


def _checkpoint_path(label: str) -> Path:
    return CHECKPOINT_DIR / f"mdot5_{label}_N426.npz"


def _save(label: str, state, eta_e: float, grid: np.ndarray, row: dict[str, object]) -> None:
    np.savez_compressed(
        _checkpoint_path(label),
        x=np.asarray(state, dtype=float),
        eta_E=np.asarray(float(eta_e)),
        custom_grid_xi=np.asarray(grid, dtype=float),
        row_json=np.asarray(json.dumps(row, sort_keys=True)),
    )


def _load(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    with np.load(path) as data:
        state = np.asarray(data["x"], dtype=float)
        eta_e = float(data["eta_E"]) if "eta_E" in data.files else 10.0
        grid = np.asarray(data["custom_grid_xi"], dtype=float)
    return state, eta_e, grid


def _summary(state, params) -> dict[str, float]:
    _logu, _logT, F, _j, _epsilon, log_r_son, _log_r = unpack_conservative_state(
        state, params
    )
    transport = conservative_transport_profile(state, params)
    return {
        "F_inner": float(F[0]),
        "F_outer": float(F[-1]),
        "Rson_rg": float(np.exp(log_r_son) / params.disk.r_g),
        "wind_over_mdot_inner": float(
            np.sum(transport["wind_mass"]) / params.flux_scales.mdot
        ),
        "stream_over_mdot_inner": float(
            np.sum(transport["stream_mass"]) / params.flux_scales.mdot
        ),
    }


def _polish_anchor(state, eta_e: float, grid: np.ndarray, passes: int):
    params = _problem(grid, eta_e)
    rows = []
    for pass_index in range(1, passes + 1):
        result = solve_conservative_disk_block_jacobian(
            state,
            params,
            max_nfev=ANCHOR_MAX_NFEV,
        )
        state = result.x
        rows.append(
            {
                "pass": pass_index,
                "nfev": result.nfev,
                "message": result.message,
                "audit": asdict(result.final_audit),
            }
        )
        if result.final_audit.maximum <= 3.0e-5:
            break
    row = {
        "kind": "fixed_eta_anchor",
        "eta_E": eta_e,
        "accepted": conservative_residual_audit(state, params).maximum <= 3.0e-5,
        "passes": rows,
        "summary": _summary(state, params),
    }
    return state, row


def run() -> list[dict[str, object]]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    eta10_path = _checkpoint_path("eta10")
    eta11_path = _checkpoint_path("eta11")

    if REUSE and eta10_path.exists():
        eta10_state, _eta, grid = _load(eta10_path)
    else:
        with np.load(SOURCE_CHECKPOINT) as data:
            eta10_state = np.asarray(data["x"], dtype=float)
            grid = np.asarray(data["custom_grid_xi"], dtype=float)
        eta10_state, row = _polish_anchor(eta10_state, 10.0, grid, passes=2)
        rows.append(row)
        _save("eta10", eta10_state, 10.0, grid, row)

    if REUSE and eta11_path.exists():
        eta11_state, _eta, _grid = _load(eta11_path)
    else:
        eta11_state, row = _polish_anchor(eta10_state, 11.0, grid, passes=1)
        rows.append(row)
        _save("eta11", eta11_state, 11.0, grid, row)

    previous_state, previous_eta = eta11_state, 11.0
    current_state, current_eta = eta10_state, 10.0
    factor = 0.25
    accepted_steps = 0
    attempts = 0
    while accepted_steps < MAX_ARC_STEPS and current_eta > MIN_ETA:
        attempts += 1
        params = _problem(grid, current_eta)
        result = conservative_eta_pseudo_arclength_step(
            previous_state,
            previous_eta,
            current_state,
            current_eta,
            params,
            step_factor=factor,
            max_nfev=ARC_MAX_NFEV,
        )
        accepted = bool(
            result.final_audit.maximum <= 3.0e-5
            and abs(result.arc_residual) <= 1.0e-6
        )
        final_params = _problem(grid, result.eta_E)
        row = {
            "kind": "bordered_eta_step",
            "attempt": attempts,
            "step_factor": factor,
            "eta_E": result.eta_E,
            "accepted": accepted,
            "nfev": result.nfev,
            "arc_residual": result.arc_residual,
            "tangent_mu": result.tangent_mu,
            "message": result.message,
            "audit": asdict(result.final_audit),
            "summary": _summary(result.x, final_params),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if accepted:
            label = f"eta_arc_{result.eta_E:.8f}"
            _save(label, result.x, result.eta_E, grid, row)
            previous_state, previous_eta = current_state, current_eta
            current_state, current_eta = result.x, result.eta_E
            accepted_steps += 1
            factor = min(0.5, 2.0 * factor)
        else:
            factor *= 0.5
            if factor < 0.0625:
                break

    for eta_e in DIRECT_ETA_VALUES:
        if eta_e >= current_eta:
            continue
        params = _problem(grid, eta_e)
        initial = conservative_residual_audit(current_state, params)
        if initial.maximum <= 3.0e-5:
            result_state = current_state
            final = initial
            nfev = 0
            message = "accepted without correction"
        else:
            solved = solve_conservative_disk_block_jacobian(
                current_state,
                params,
                max_nfev=ANCHOR_MAX_NFEV,
            )
            result_state = solved.x
            final = solved.final_audit
            nfev = solved.nfev
            message = solved.message
        accepted = bool(final.maximum <= 3.0e-5)
        row = {
            "kind": "direct_eta_corrector",
            "eta_E": eta_e,
            "accepted": accepted,
            "nfev": nfev,
            "message": message,
            "audit": asdict(final),
            "summary": _summary(result_state, params),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if not accepted:
            break
        label = f"eta{eta_e:g}_block_mass{MASS_WEIGHT:g}"
        _save(label, result_state, eta_e, grid, row)
        previous_state, previous_eta = current_state, current_eta
        current_state, current_eta = result_state, eta_e

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
