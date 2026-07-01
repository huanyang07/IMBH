"""Residual-based outer mesh adaptation for high-rate standard slim checkpoints."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    pressure_supported_omega_target,
    remap_profile_to_new_sonic_grid,
    residual_audit_from_state_vector,
    solve_square_transonic_polish,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_ANCHOR",
    "outputs/checkpoints/slim_benchmark_mesh_closure_validation_pressure_N640_grid060_0p93_1_certify/m1000_N640_pressure_one_sided_mdot_1.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_TABLE",
    "outputs/tables/slim_benchmark_adaptive_outer_mesh_mdot1.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_FIGURE",
    "outputs/figures/slim_benchmark_adaptive_outer_mesh_mdot1.png",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_adaptive_outer_mesh_mdot1",
)

N_NODES = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_N", "640"))
STRENGTHS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_STRENGTHS", "6,12,24")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
BLEND = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_BLEND", "0.75"))
POWER = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_POWER", "0.5"))
SMOOTH_PASSES = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_SMOOTH_PASSES", "2"))
MONITOR_FLOOR = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_MONITOR_FLOOR", "1.0"))
OUTER_TAPER_START = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_OUTER_TAPER_START", "0.0"))
NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_NEWTON_MAX_ITER", "28"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_NEWTON_MAX_NFEV", "2600"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_NEWTON_MAX_STEP_NORM", "0.16"))
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_NEWTON_LINEAR_SOLVER", "regularized_lsmr")
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_ANCHOR_TOL", "3e-6"))
REFRESH_REPOLISH = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_REFRESH_REPOLISH", "1") != "0"
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_OUTER_MESH_PIVOTS", "C2,C1").replace(":", ",").split(",")
    if piece.strip()
)


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    *,
    ratio: float,
    R_out_rg: float,
    n_nodes: int,
    grid_power: float,
    custom_grid_xi: tuple[float, ...] | None = None,
    outer_closure: str = "thin_value",
    outer_match_log_slopes: tuple[float, float] | None = None,
    outer_temperature_logT: float | None = None,
    outer_entropy_logK: float | None = None,
    outer_omega_log_offset: float = 0.0,
) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(ratio) * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=float(R_out_rg),
        n_nodes=int(n_nodes),
        grid_power=float(grid_power),
        custom_grid_xi=custom_grid_xi,
        max_nfev=NEWTON_MAX_NFEV,
        residual_tol=1.0e-8,
        outer_closure=outer_closure,
        outer_match_log_slopes=outer_match_log_slopes,
        outer_temperature_logT=outer_temperature_logT,
        outer_entropy_logK=outer_entropy_logK,
        outer_omega_log_offset=float(outer_omega_log_offset),
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


def one_sided_outer_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    dx = float(logR[-1] - logR[-2])
    return float((logu[-1] - logu[-2]) / dx), float((logT[-1] - logT[-2]) / dx)


def refresh_outer_slopes_from_state(z: np.ndarray, params: TransonicSlimParams) -> TransonicSlimParams:
    if params.outer_closure in {
        "pressure_supported_thin_energy",
        "pressure_supported_temperature",
        "pressure_supported_entropy",
    }:
        return replace(params, outer_match_log_slopes=one_sided_outer_slopes(z, params))
    return params


def load_checkpoint(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    data = np.load(path, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    custom_grid_xi = None
    if "custom_grid_xi" in data:
        candidate_grid = np.asarray(data["custom_grid_xi"], dtype=float)
        if candidate_grid.shape == (int(data["n_nodes"]),):
            custom_grid_xi = tuple(float(value) for value in candidate_grid)
    outer_closure = "thin_value"
    if "outer_closure" in data:
        outer_closure = str(np.asarray(data["outer_closure"]).item())
    slopes = None
    if "outer_match_log_slopes" in data:
        candidate = np.asarray(data["outer_match_log_slopes"], dtype=float)
        if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
            slopes = (float(candidate[0]), float(candidate[1]))
    outer_temperature_logT = None
    if "outer_temperature_logT" in data and np.isfinite(float(data["outer_temperature_logT"])):
        outer_temperature_logT = float(data["outer_temperature_logT"])
    outer_entropy_logK = None
    if "outer_entropy_logK" in data and np.isfinite(float(data["outer_entropy_logK"])):
        outer_entropy_logK = float(data["outer_entropy_logK"])
    outer_omega_log_offset = 0.0
    if "outer_omega_log_offset" in data and np.isfinite(float(data["outer_omega_log_offset"])):
        outer_omega_log_offset = float(data["outer_omega_log_offset"])
    params = params_for(
        fiducial,
        mdot_edd,
        ratio=float(data["ratio"]),
        R_out_rg=float(data["R_out_rg"]),
        n_nodes=int(data["n_nodes"]),
        grid_power=float(data["grid_power"]) if "grid_power" in data else 1.0,
        custom_grid_xi=custom_grid_xi,
        outer_closure=outer_closure,
        outer_match_log_slopes=slopes,
        outer_temperature_logT=outer_temperature_logT,
        outer_entropy_logK=outer_entropy_logK,
        outer_omega_log_offset=outer_omega_log_offset,
    )
    return z, refresh_outer_slopes_from_state(z, params)


def interval_residuals(z: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    return np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            for idx in range(len(logR) - 1)
        ],
        dtype=float,
    )


def smooth_score(score: np.ndarray, passes: int) -> np.ndarray:
    smoothed = np.asarray(score, dtype=float)
    for _ in range(max(int(passes), 0)):
        if smoothed.size <= 2:
            break
        padded = np.pad(smoothed, (1, 1), mode="edge")
        smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return smoothed


def enforce_min_spacing(xi: np.ndarray, min_spacing: float = 1.0e-10) -> np.ndarray:
    adjusted = np.asarray(xi, dtype=float).copy()
    adjusted[0] = 0.0
    adjusted[-1] = 1.0
    for idx in range(1, adjusted.size):
        adjusted[idx] = max(adjusted[idx], adjusted[idx - 1] + min_spacing)
    if adjusted[-1] > 1.0:
        scale = 1.0 / adjusted[-1]
        adjusted *= scale
    adjusted[-1] = 1.0
    for idx in range(adjusted.size - 2, -1, -1):
        adjusted[idx] = min(adjusted[idx], adjusted[idx + 1] - min_spacing)
    adjusted[0] = 0.0
    adjusted[-1] = 1.0
    if np.any(np.diff(adjusted) <= 0.0):
        raise RuntimeError("adaptive grid spacing collapsed")
    return adjusted


def adaptive_grid_xi(source_z: np.ndarray, source_params: TransonicSlimParams, *, strength: float) -> tuple[tuple[float, ...], dict[str, float]]:
    _logu, _logT, logR_son, _lambda0, logR = unpack_state(source_z, source_params)
    span = float(np.log(source_params.R_out) - logR_son)
    source_xi = (logR - logR_son) / span
    intervals = interval_residuals(source_z, source_params)
    score = np.maximum(np.abs(intervals[:, 1]), 1.0e-300)
    score = smooth_score(score, SMOOTH_PASSES)
    if OUTER_TAPER_START > 0.0:
        mids = 0.5 * (source_xi[:-1] + source_xi[1:])
        taper = np.clip((mids - OUTER_TAPER_START) / max(1.0 - OUTER_TAPER_START, 1.0e-12), 0.0, 1.0)
        score *= taper
    score_norm = score / max(float(np.max(score)), 1.0e-300)
    monitor = MONITOR_FLOOR + float(strength) * score_norm**POWER
    ds = np.diff(source_xi)
    cumulative = np.concatenate([[0.0], np.cumsum(monitor * ds)])
    cumulative /= cumulative[-1]
    target = np.linspace(0.0, 1.0, N_NODES)
    adapted = np.interp(target, cumulative, source_xi)
    source_reference = np.interp(target, np.linspace(0.0, 1.0, source_xi.size), source_xi)
    blended = (1.0 - BLEND) * source_reference + BLEND * adapted
    blended = enforce_min_spacing(blended)
    info = {
        "monitor_strength": float(strength),
        "monitor_power": float(POWER),
        "monitor_max": float(np.max(monitor)),
        "monitor_p90": float(np.percentile(monitor, 90.0)),
        "source_outer_dx": float(logR[-1] - logR[-2]),
        "target_outer_dxi": float(blended[-1] - blended[-2]),
        "target_inner_dxi": float(blended[1] - blended[0]),
        "target_outer_1pct_nodes": int(np.count_nonzero(blended >= 0.99)),
        "target_outer_5pct_nodes": int(np.count_nonzero(blended >= 0.95)),
    }
    return tuple(float(value) for value in blended), info


def remap_seed(source_z: np.ndarray, source_params: TransonicSlimParams, target_params: TransonicSlimParams) -> np.ndarray:
    profile = transonic_profile_from_state_vector(source_z, source_params)
    return remap_profile_to_new_sonic_grid(profile, target_params, temperature_mdot_power=0.0)


def polish_best(z0: np.ndarray, params: TransonicSlimParams):
    best = None
    best_full = np.inf
    for pivot in PIVOTS:
        result = solve_square_transonic_polish(
            params,
            z0,
            pivot=pivot,
            method="newton",
            max_iter=NEWTON_MAX_ITER,
            max_nfev=NEWTON_MAX_NFEV,
            residual_tol=1.0e-8,
            use_block_jacobian=True,
            linear_solver=NEWTON_LINEAR_SOLVER,
            max_step_norm=NEWTON_MAX_STEP_NORM,
        )
        full = max_residual(result.z, params)
        if full < best_full:
            best = result
            best_full = full
        if full <= ANCHOR_TOL:
            break
    if best is None:
        raise RuntimeError("no polish pivots configured")
    return best


def pressure_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    slopes = params.outer_match_log_slopes
    if slopes is None:
        return {"outer_pressure_target": np.nan, "outer_pressure_residual": np.nan}
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    ln_omega = float(np.log(profile.Omega[-1] / profile.Omega_K[-1]))
    target = pressure_supported_omega_target(
        float(logR[-1]),
        np.array([logu[-1], logT[-1]], dtype=float),
        np.asarray(slopes, dtype=float),
        lambda0,
        params,
    ) + float(params.outer_omega_log_offset)
    return {"outer_pressure_target": float(target), "outer_pressure_residual": float(ln_omega - target)}


def peak_interval(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float, float]:
    intervals = interval_residuals(z, params)
    _logu, _logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    idx = int(np.argmax(np.abs(intervals[:, 1])))
    return (
        float(np.exp(0.5 * (logR[idx] + logR[idx + 1])) / params.r_g),
        float(intervals[idx, 0]),
        float(intervals[idx, 1]),
    )


def row_for_result(
    *,
    label: str,
    seed: np.ndarray,
    z: np.ndarray,
    params: TransonicSlimParams,
    polish,
    elapsed_s: float,
    grid_info: dict[str, float],
) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    peak_R, peak_R_value, peak_E_value = peak_interval(z, params)
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    full = max_residual(z, params)
    return {
        "label": label,
        "ratio": float(params.mdot_edd_ratio),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "grid_power": float(params.grid_power),
        "custom_grid": bool(params.custom_grid_xi is not None),
        "initial_full": max_residual(seed, params),
        "final_full": full,
        "accepted": bool(full <= ACCEPTANCE_TOL),
        "anchor_eligible": bool(full <= ANCHOR_TOL),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "outer_pressure_residual": float(pressure_diagnostic(z, params)["outer_pressure_residual"]),
        "peak_interval_R_rg": float(peak_R),
        "peak_interval_R": float(peak_R_value),
        "peak_interval_E": float(peak_E_value),
        "outer_dx": float(logR[-1] - logR[-2]),
        "outer_dx_ratio_to_uniform": float((logR[-1] - logR[-2]) / ((logR[-1] - logR[0]) / (params.n_nodes - 1))),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0_over_lK_isco": float(audit.lambda0_over_lK_isco),
        "max_H_R": float(np.max(profile.H_over_R)),
        "integrated_adv": float(profile.integrated_advective_fraction),
        "pivot": str(polish.pivot),
        "method": str(polish.method),
        "nfev": int(polish.result.nfev),
        "iterations": int(polish.iterations),
        "elapsed_s": float(elapsed_s),
        "message": str(polish.result.message),
        **grid_info,
        "z": np.asarray(z, dtype=float),
        "custom_grid_xi": np.asarray(params.custom_grid_xi, dtype=float)
        if params.custom_grid_xi is not None
        else np.asarray([], dtype=float),
    }


def save_checkpoint(row: dict[str, Any], params: TransonicSlimParams) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe = str(row["label"]).replace(".", "p").replace("-", "m")
    stem = f"{safe}_mdot_{float(row['ratio']):.8g}_N{int(row['N'])}".replace(".", "p")
    path = CHECKPOINT_DIR / f"{stem}.npz"
    slopes = params.outer_match_log_slopes
    payload = {key: value for key, value in row.items() if key not in {"z", "custom_grid_xi"}}
    np.savez_compressed(
        path,
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(row["R_out_rg"]),
        n_nodes=np.array(row["N"]),
        grid_power=np.array(params.grid_power),
        custom_grid_xi=np.asarray(row["custom_grid_xi"], dtype=float),
        outer_closure=np.array(params.outer_closure),
        outer_match_log_slopes=np.asarray([np.nan, np.nan] if slopes is None else slopes, dtype=float),
        outer_temperature_logT=np.array(np.nan if params.outer_temperature_logT is None else params.outer_temperature_logT),
        outer_entropy_logK=np.array(np.nan if params.outer_entropy_logK is None else params.outer_entropy_logK),
        outer_omega_log_offset=np.array(params.outer_omega_log_offset),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted"]),
        row_json=np.array(json.dumps(json_safe(payload), sort_keys=True)),
    )


def write_table(rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Adaptive Outer Mesh",
        "",
        "Generated by `scripts/run_standard_slim_adaptive_outer_mesh.py`.",
        "",
        f"Anchor `{ANCHOR_CHECKPOINT.relative_to(ROOT)}`, N `{N_NODES}`, strengths `{','.join(f'{value:g}' for value in STRENGTHS)}`, "
        f"blend `{BLEND:g}`, power `{POWER:g}`, smooth passes `{SMOOTH_PASSES}`, refresh repolish `{REFRESH_REPOLISH}`.",
        "",
        "| label | Mdot/Edd | N | initial full | final full | accepted | anchor | dominant | int E | peak R/rg | peak E | outer dx | outer 1% nodes | outer 5% nodes | max H/R | int adv | pivot | nfev | elapsed s | message |",
        "|---|---:|---:|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {label} | {ratio} | {N} | {initial_full} | {final_full} | {accepted} | {anchor_eligible} | "
            "{dominant} | {interval_E} | {peak_interval_R_rg} | {peak_interval_E} | {outer_dx} | "
            "{target_outer_1pct_nodes} | {target_outer_5pct_nodes} | {max_H_R} | {integrated_adv} | "
            "{pivot} | {nfev} | {elapsed_s} | {message} |".format(**formatted).replace("\n", " ")
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(
        json.dumps(json_safe([{key: value for key, value in row.items() if key not in {"z", "custom_grid_xi"}} for row in rows]), indent=2, sort_keys=True)
        + "\n"
    )


def write_figure(rows: list[dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    if not rows:
        return
    width, height = 1050, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = 90, 80, 970, 540
    draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
    x_vals = np.arange(len(rows), dtype=float)
    y_vals = np.log10(np.maximum(np.asarray([float(row["final_full"]) for row in rows], dtype=float), 1.0e-16))
    y_min, y_max = float(np.floor(np.min(y_vals))), float(np.ceil(np.max(y_vals)))
    if y_max <= y_min:
        y_max = y_min + 1.0
    for idx, row in enumerate(rows):
        xx = x_vals[idx] / max(len(rows) - 1, 1)
        yy = np.log10(max(float(row["final_full"]), 1.0e-16))
        px = x0 + int(xx * (x1 - x0))
        py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
        color = (31, 119, 180) if row["accepted"] else (214, 39, 40)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
        draw.text((px - 30, py + 10), str(row["label"]), fill=(20, 20, 20), font=font)
    for tol, label in ((ACCEPTANCE_TOL, "accept"), (ANCHOR_TOL, "anchor")):
        yy = np.log10(tol)
        py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
        draw.line((x0, py, x1, py), fill=(120, 120, 120), width=1)
        draw.text((x1 - 72, py - 14), label, fill=(80, 80, 80), font=font)
    draw.text((90, 25), "Residual-based adaptive outer mesh: final residual", fill=(20, 20, 20), font=font)
    draw.text((x0 + 4, y0 + 4), f"1e{int(y_max)}", fill=(80, 80, 80), font=font)
    draw.text((x0 + 4, y1 - 18), f"1e{int(y_min)}", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_z, source_params = load_checkpoint(ANCHOR_CHECKPOINT, fiducial, mdot_edd)
    rows: list[dict[str, Any]] = []
    for strength in STRENGTHS:
        custom_grid_xi, grid_info = adaptive_grid_xi(source_z, source_params, strength=strength)
        base_params = params_for(
            fiducial,
            mdot_edd,
            ratio=source_params.mdot_edd_ratio,
            R_out_rg=source_params.R_out_rg,
            n_nodes=N_NODES,
            grid_power=source_params.grid_power,
            custom_grid_xi=custom_grid_xi,
            outer_closure=source_params.outer_closure,
            outer_temperature_logT=source_params.outer_temperature_logT,
            outer_entropy_logK=source_params.outer_entropy_logK,
            outer_omega_log_offset=source_params.outer_omega_log_offset,
        )
        seed = remap_seed(source_z, source_params, base_params)
        target_params = refresh_outer_slopes_from_state(seed, base_params)
        label = f"s{strength:g}".replace(".", "p")
        print(
            f"{label} initial={max_residual(seed, target_params):.3e} "
            f"outer1={grid_info['target_outer_1pct_nodes']} outer5={grid_info['target_outer_5pct_nodes']}",
            flush=True,
        )
        t0 = time.perf_counter()
        polish = polish_best(seed, target_params)
        final_params = refresh_outer_slopes_from_state(polish.z, target_params)
        if REFRESH_REPOLISH:
            polish = polish_best(polish.z, final_params)
            final_params = refresh_outer_slopes_from_state(polish.z, final_params)
        elapsed = time.perf_counter() - t0
        row = row_for_result(
            label=label,
            seed=seed,
            z=polish.z,
            params=final_params,
            polish=polish,
            elapsed_s=elapsed,
            grid_info=grid_info,
        )
        rows.append(row)
        save_checkpoint(row, final_params)
        write_table(rows)
        write_figure(rows)
        print(
            f"  final={row['final_full']:.3e} dom={row['dominant']} "
            f"anchor={row['anchor_eligible']} peak_R={row['peak_interval_R_rg']:.4g}",
            flush=True,
        )
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
