"""Experimental local-Mdot BVP pilot for the Mdot=5 mass-coupled wind branch."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_standard_slim_stream_mass_annulus_scan as scan  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_state,
    entropy_gradient_log,
    pack_state,
    sonic_residual_pair,
    state_partials,
    stream_heating_rate,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    transonic_profile_from_state_vector,
    wind_energy_loss_rate,
    wind_energy_per_mass,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (  # noqa: E402
    _interval_geometry,
    _interval_residual_from_unpacked,
    _outer_residual_block,
    computational_grid,
)
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402


DEFAULT_ANCHOR = (
    ROOT
    / "outputs/checkpoints/m5_energy_wind_powerlaw_mass_coupled_adaptive_0p05_to_0p10/"
    "zeta_0p1_N896.npz"
)
ANCHOR = Path(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ANCHOR", str(DEFAULT_ANCHOR))).expanduser()
if not ANCHOR.is_absolute():
    ANCHOR = ROOT / ANCHOR
OUTPUT_STEM = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_OUTPUT_STEM", "m5_local_mdot_bvp_pilot").strip()
N_NODES = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_N_NODES", "128"))
REMAP_METHOD = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_REMAP_METHOD", "linear").strip().lower()
MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_MAX_NFEV", "300"))
RESIDUAL_TOL = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_RESIDUAL_TOL", "1e-6"))
PIVOT = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_PIVOT", "C2").strip()
MASS_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_MASS_WEIGHT", "1.0"))
WIND_ENERGY_MULTIPLIER = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_WIND_ENERGY_MULTIPLIER", "1.0"))
JSON_OUTPUT = ROOT / f"outputs/tables/{OUTPUT_STEM}.json"
MD_OUTPUT = ROOT / f"outputs/tables/{OUTPUT_STEM}.md"
CHECKPOINT_DIR = ROOT / f"outputs/checkpoints/{OUTPUT_STEM}"


def _format(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "nan"
    if number == 0.0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e4:
        return f"{number:.3e}"
    return f"{number:.6g}"


def _local_params(params, logR: np.ndarray, logMdot: np.ndarray):
    return replace(
        params,
        wind_sink_fraction=0.0,
        mdot_profile_mode="tabulated",
        mdot_profile_logR=tuple(float(x) for x in logR),
        mdot_profile_logMdot=tuple(float(x) for x in logMdot),
    )


def _pack(logu, logT, logMdot, logR_son: float, lambda0: float) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(logu, dtype=float),
            np.asarray(logT, dtype=float),
            np.asarray(logMdot, dtype=float),
            np.array([float(logR_son), float(lambda0)]),
        ]
    )


def _unpack(x: np.ndarray, params) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    n = int(params.n_nodes)
    logu = np.asarray(x[:n], dtype=float)
    logT = np.asarray(x[n : 2 * n], dtype=float)
    logMdot = np.asarray(x[2 * n : 3 * n], dtype=float)
    logR_son = float(x[-2])
    lambda0 = float(x[-1])
    logR = computational_grid(params, logR_son)
    return logu, logT, logMdot, logR_son, lambda0, logR


def _state_vector(logu: np.ndarray, logT: np.ndarray, logR_son: float, lambda0: float) -> np.ndarray:
    return pack_state(logu, logT, logR_son, lambda0)


def _wind_mass_prime(xm: float, ym: np.ndarray, gm: np.ndarray, lambda0: float, params) -> float:
    state = algebraic_state(xm, float(ym[0]), float(ym[1]), lambda0, params)
    partials = state_partials(xm, ym, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)
    dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], gm))
    Tdsdx = entropy_gradient_log(xm, ym, gm, lambda0, params)
    Q_visc = -state.W * dOmega_dx
    Q_adv = -(state.Sigma * state.u / state.R) * Tdsdx
    Q_stream = stream_heating_rate(xm, params)
    Q_wind = wind_energy_loss_rate(state, Q_visc, Q_stream, Q_adv, params)
    E_w = float(WIND_ENERGY_MULTIPLIER * wind_energy_per_mass(params.M2_g, state.R))
    if E_w <= 0.0:
        raise ValueError("wind energy multiplier produced non-positive E_w")
    return float(2.0 * np.pi * state.R**2 * Q_wind / E_w)


def residual(x: np.ndarray, params) -> np.ndarray:
    try:
        logu, logT, logMdot, logR_son, lambda0, logR = _unpack(x, params)
        if np.any(np.diff(logR) <= 0.0):
            raise ValueError("mapped radius must increase")
        local_params = _local_params(params, logR, logMdot)
        rows: list[float] = []
        for idx in range(params.n_nodes - 1):
            rows.extend(_interval_residual_from_unpacked(logu, logT, logR, lambda0, local_params, idx))
        z = _state_vector(logu, logT, logR_son, lambda0)
        rows.extend(_outer_residual_block(z, local_params))
        rows.extend(sonic_residual_pair(z, local_params, pivot=PIVOT))
        rows.append(float(logMdot[0] - np.log(params.Mdot_g_s)))
        for idx in range(params.n_nodes - 1):
            dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
            ym = 0.5 * (y_left + y_right)
            gm = (y_right - y_left) / dx
            logMdot_mid = 0.5 * (logMdot[idx] + logMdot[idx + 1])
            mdot_mid = float(np.exp(logMdot_mid))
            dlogMdot_dx = float((logMdot[idx + 1] - logMdot[idx]) / dx)
            source_prime = stream_source_prime(xm, local_params)
            wind_prime = _wind_mass_prime(xm, ym, gm, lambda0, local_params)
            target = float((wind_prime - source_prime) / mdot_mid)
            rows.append(float(MASS_WEIGHT * (dlogMdot_dx - target)))
        return np.asarray(rows, dtype=float)
    except Exception:
        return np.full(3 * params.n_nodes + 2, 1.0e6)


def _bounds(params) -> tuple[np.ndarray, np.ndarray]:
    state_lower, state_upper = scan.state_bounds(params)
    n = int(params.n_nodes)
    log_mdot_inner = np.log(params.Mdot_g_s)
    lower = np.concatenate(
        [
            state_lower[:n],
            state_lower[n : 2 * n],
            np.full(n, log_mdot_inner + np.log(1.0e-3)),
            state_lower[-2:],
        ]
    )
    upper = np.concatenate(
        [
            state_upper[:n],
            state_upper[n : 2 * n],
            np.full(n, log_mdot_inner + np.log(1.0e3)),
            state_upper[-2:],
        ]
    )
    return lower, upper


def _sparsity(params):
    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None
    n = int(params.n_nodes)
    size = 3 * n + 2
    pattern = lil_matrix((size, size), dtype=int)
    row = 0
    logR_col = size - 2
    lambda_col = size - 1
    for idx in range(n - 1):
        cols = (idx, idx + 1, n + idx, n + idx + 1, 2 * n + idx, 2 * n + idx + 1, logR_col, lambda_col)
        for col in cols:
            pattern[row : row + 2, col] = 1
        row += 2
    for col in (n - 1, 2 * n - 1, 3 * n - 1, logR_col, lambda_col):
        pattern[row : row + 2, col] = 1
    row += 2
    for col in (0, n, 2 * n, logR_col, lambda_col):
        pattern[row : row + 2, col] = 1
    row += 2
    pattern[row, 2 * n] = 1
    row += 1
    for idx in range(n - 1):
        cols = (idx, idx + 1, n + idx, n + idx + 1, 2 * n + idx, 2 * n + idx + 1, logR_col, lambda_col)
        for col in cols:
            pattern[row, col] = 1
        row += 1
    return pattern.tocsr()


def _row(label: str, x: np.ndarray, params, initial_full: float, result=None) -> dict[str, Any]:
    logu, logT, logMdot, logR_son, lambda0, logR = _unpack(x, params)
    local_params = _local_params(params, logR, logMdot)
    z = _state_vector(logu, logT, logR_son, lambda0)
    r = residual(x, params)
    adv = scan.advection_diagnostic(z, local_params)
    stream = scan.stream_diagnostic(z, local_params)
    audit = scan.residual_audit_from_state_vector(z, local_params)
    return {
        "label": label,
        "N": int(params.n_nodes),
        "initial_full": float(initial_full),
        "final_full": float(np.linalg.norm(r, ord=np.inf)),
        "mass_residual_max": float(np.linalg.norm(r[-params.n_nodes :], ord=np.inf)),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "Mdot_outer_over_inner": float(np.exp(logMdot[-1]) / params.Mdot_g_s),
        "Mdot_tabulated_outer_over_inner": float(stream["Mdot_outer_over_inner"]),
        "f_adv_global": float(adv["f_adv_global"]),
        "Lrad_LEdd": float(adv["Lrad_LEdd"]),
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "nfev": np.nan if result is None else int(result.nfev),
        "success": False if result is None else bool(result.success),
        "message": "" if result is None else str(result.message),
        "wind_energy_multiplier": float(WIND_ENERGY_MULTIPLIER),
    }


def _write_outputs(rows: list[dict[str, Any]]) -> None:
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(scan.json_safe(rows), indent=2, sort_keys=True) + "\n")
    cols = [
        "label",
        "N",
        "initial_full",
        "final_full",
        "mass_residual_max",
        "Mdot_outer_over_inner",
        "f_adv_global",
        "Lrad_LEdd",
        "Rson_rg",
        "nfev",
        "success",
        "wind_energy_multiplier",
    ]
    lines = [
        "# Mdot=5 Local-Mdot BVP Pilot",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(col, "")) for col in cols) + " |")
    MD_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not ANCHOR.exists():
        raise FileNotFoundError(ANCHOR)
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = scan.load_anchor(ANCHOR, fiducial, mdot_edd)
    if int(anchor_params.n_nodes) != N_NODES:
        target_params = replace(anchor_params, n_nodes=N_NODES, custom_grid_xi=None)
        profile = transonic_profile_from_state_vector(anchor_z, anchor_params)
        anchor_z = scan.remap_profile_to_new_sonic_grid(
            profile,
            target_params,
            temperature_mdot_power=0.0,
            method=REMAP_METHOD,
        )
        anchor_params = scan.apply_outer_slopes_from_state(anchor_z, target_params)
    logu, logT, logR_son, lambda0, logR = scan.unpack_state(anchor_z, anchor_params)
    mdot_seed = np.asarray([stream_mass_rate_and_derivative(float(x), anchor_params)[0] for x in logR], dtype=float)
    local_params = replace(anchor_params, wind_sink_fraction=0.0, mdot_profile_mode="source_sink")
    x0 = _pack(logu, logT, np.log(mdot_seed), logR_son, lambda0)
    lower, upper = _bounds(local_params)
    x0 = np.clip(x0, lower + 1.0e-12, upper - 1.0e-12)
    initial_full = float(np.linalg.norm(residual(x0, local_params), ord=np.inf))
    print(
        f"anchor={scan.relative_root_path(ANCHOR)} N={N_NODES} initial_full={initial_full:.3e} "
        f"Mout/Min={mdot_seed[-1] / local_params.Mdot_g_s:.6g} "
        f"Ew_multiplier={WIND_ENERGY_MULTIPLIER:.6g}",
        flush=True,
    )

    from scipy.optimize import least_squares

    result = least_squares(
        lambda trial: residual(trial, local_params),
        x0,
        bounds=(lower, upper),
        jac_sparsity=_sparsity(local_params),
        x_scale="jac",
        loss="linear",
        ftol=RESIDUAL_TOL,
        xtol=RESIDUAL_TOL,
        gtol=RESIDUAL_TOL,
        max_nfev=MAX_NFEV,
        verbose=0,
    )
    rows = [_row("initial", x0, local_params, initial_full), _row("polished", result.x, local_params, initial_full, result)]
    _write_outputs(rows)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHECKPOINT_DIR / f"local_mdot_N{N_NODES}.npz",
        x=np.asarray(result.x, dtype=float),
        row_json=np.array(json.dumps(scan.json_safe(rows[-1]), sort_keys=True)),
    )
    print(
        f"final_full={rows[-1]['final_full']:.3e} mass={rows[-1]['mass_residual_max']:.3e} "
        f"success={rows[-1]['success']} nfev={rows[-1]['nfev']}",
        flush=True,
    )
    print(f"wrote {JSON_OUTPUT.relative_to(ROOT)}", flush=True)
    print(f"wrote {MD_OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
