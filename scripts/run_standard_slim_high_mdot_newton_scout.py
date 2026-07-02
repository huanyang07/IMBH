"""Lean Newton-only scout for the standard no-wind high-Mdot branch."""

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
    residual_audit_from_state_vector,
    solve_square_transonic_polish,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_luminosity, eddington_mdot
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant
from run_standard_slim_mdot_predictor_audit import clip_state, tangent_predictor, tangent_vector


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_ANCHOR",
    "outputs/checkpoints/slim_benchmark_adaptive_outer_mesh_mdot1_scan/s8_mdot_1_N640.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_TABLE",
    "outputs/tables/slim_benchmark_high_mdot_no_wind_newton_scout.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_FIGURE",
    "outputs/figures/slim_benchmark_high_mdot_no_wind_newton_scout.png",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_high_mdot_no_wind_newton_scout",
)
TARGETS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_TARGETS", "1.05,1.1,1.2,1.35,1.5,1.75,2.0")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
STRESS_FACTOR = float(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_STRESS", "1.0"))
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_ANCHOR_TOL", "3e-6"))
NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_NEWTON_MAX_ITER", "16"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_NEWTON_MAX_NFEV", "1600"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_NEWTON_MAX_STEP_NORM", "0.18"))
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_LINEAR_SOLVER", "regularized_lsmr")
SLOPE_PICARD_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_SLOPE_PICARD", "2"))
INNER_RADIUS_RG = float(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_INNER_RG", "20.0"))
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_SCOUT_PIVOTS", "C2,C1").replace(":", ",").split(",")
    if piece.strip()
)


def custom_grid_from_data(data) -> tuple[float, ...] | None:
    if "custom_grid_xi" not in data:
        return None
    candidate = np.asarray(data["custom_grid_xi"], dtype=float)
    if candidate.shape == (int(data["n_nodes"]),) and candidate.size > 0:
        return tuple(float(value) for value in candidate)
    return None


def one_sided_outer_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    dx = float(logR[-1] - logR[-2])
    return float((logu[-1] - logu[-2]) / dx), float((logT[-1] - logT[-2]) / dx)


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    ratio: float,
    *,
    R_out_rg: float,
    n_nodes: int,
    grid_power: float,
    custom_grid_xi: tuple[float, ...] | None,
    slopes: tuple[float, float] | None,
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
        outer_closure="pressure_supported_thin_energy",
        outer_match_log_slopes=slopes,
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def load_anchor(fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    data = np.load(ANCHOR_CHECKPOINT, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    params = params_for(
        fiducial,
        mdot_edd,
        float(data["ratio"]),
        R_out_rg=float(data["R_out_rg"]),
        n_nodes=int(data["n_nodes"]),
        grid_power=float(data["grid_power"]) if "grid_power" in data else 1.0,
        custom_grid_xi=custom_grid_from_data(data),
        slopes=None,
    )
    return z, replace(params, outer_match_log_slopes=one_sided_outer_slopes(z, params))


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


def best_seed(
    *,
    previous_z: np.ndarray | None,
    previous_params: TransonicSlimParams | None,
    current_z: np.ndarray,
    current_params: TransonicSlimParams,
    target_params: TransonicSlimParams,
) -> tuple[str, np.ndarray, float]:
    candidates: list[tuple[str, np.ndarray, float]] = []
    copied = clip_state(np.asarray(current_z, dtype=float), target_params)
    candidates.append(("copy", copied, max_residual(copied, target_params)))
    try:
        dz_dmu, _meta = tangent_vector(current_z, current_params, pivot="C2")
        tangent, _ = tangent_predictor(current_z, current_params, target_params, dz_dmu)
        tangent = clip_state(tangent, target_params)
        candidates.append(("tangent", tangent, max_residual(tangent, target_params)))
    except Exception:
        pass
    if previous_z is not None and previous_params is not None and previous_z.shape == current_z.shape:
        dmu = float(np.log(current_params.Mdot_g_s) - np.log(previous_params.Mdot_g_s))
        if abs(dmu) > 1.0e-14:
            scale = (float(np.log(target_params.Mdot_g_s) - np.log(current_params.Mdot_g_s))) / dmu
            secant = clip_state(current_z + scale * (current_z - previous_z), target_params)
            candidates.append(("secant", secant, max_residual(secant, target_params)))
    return min(candidates, key=lambda item: item[2])


def polish_newton(z0: np.ndarray, params: TransonicSlimParams):
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
        raise RuntimeError("no Newton pivots configured")
    return best


def slope_picard_polish(z0: np.ndarray, params: TransonicSlimParams):
    current_z = np.asarray(z0, dtype=float)
    current_params = params
    history: list[dict[str, float | int | str]] = []
    polish = None
    for index in range(max(1, SLOPE_PICARD_MAX_ITER)):
        t0 = time.perf_counter()
        polish = polish_newton(current_z, current_params)
        elapsed = time.perf_counter() - t0
        solved_full = max_residual(polish.z, current_params)
        new_slopes = one_sided_outer_slopes(polish.z, current_params)
        refreshed_params = replace(current_params, outer_match_log_slopes=new_slopes)
        refreshed_full = max_residual(polish.z, refreshed_params)
        old_slopes = np.asarray(current_params.outer_match_log_slopes, dtype=float)
        slope_delta = float(np.max(np.abs(np.asarray(new_slopes, dtype=float) - old_slopes)))
        history.append(
            {
                "index": int(index),
                "solved_full": float(solved_full),
                "refreshed_full": float(refreshed_full),
                "slope_delta": float(slope_delta),
                "elapsed_s": float(elapsed),
                "pivot": str(polish.pivot),
            }
        )
        current_z = np.asarray(polish.z, dtype=float)
        current_params = refreshed_params
        if refreshed_full <= ANCHOR_TOL and slope_delta <= 1.0e-3:
            break
    if polish is None:
        raise RuntimeError("slope Picard polish produced no result")
    return polish, current_params, history


def trapz_log(values: np.ndarray, R: np.ndarray) -> float:
    logR = np.log(np.asarray(R, dtype=float))
    weights = 2.0 * np.pi * np.asarray(R, dtype=float) ** 2
    return float(np.trapezoid(np.asarray(values, dtype=float) * weights, logR))


def masked_trapz_log(values: np.ndarray, R: np.ndarray, mask: np.ndarray) -> float:
    if int(np.count_nonzero(mask)) < 2:
        return np.nan
    return trapz_log(np.asarray(values)[mask], np.asarray(R)[mask])


def diagnostics(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    profile = transonic_profile_from_state_vector(z, params)
    R = np.asarray(profile.R, dtype=float)
    R_rg = R / params.r_g
    qv = np.asarray(profile.Q_visc, dtype=float)
    qr = np.asarray(profile.Q_rad, dtype=float)
    qa = np.asarray(profile.Q_adv, dtype=float)
    visc = trapz_log(np.abs(qv), R) + 1.0e-300
    inner = R_rg <= INNER_RADIUS_RG
    inner_visc = masked_trapz_log(np.abs(qv), R, inner)
    inner_adv = masked_trapz_log(qa, R, inner)
    ledd = eddington_luminosity(params.M2_g, kappa=params.kappa)
    return {
        "f_adv_global": float(trapz_log(qa, R) / visc),
        "f_adv_pos": float(trapz_log(np.maximum(qa, 0.0), R) / visc),
        "f_adv_inner": float(inner_adv / (inner_visc + 1.0e-300)) if np.isfinite(inner_visc) else np.nan,
        "Lrad_LEdd": float(trapz_log(qr, R) / ledd),
        "Lvisc_LEdd": float(trapz_log(qv, R) / ledd),
        "max_H_R": float(np.max(profile.H_over_R)),
        "min_tau": float(np.min(profile.tau)),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0": float(profile.lambda0),
    }


def row_for(
    label: str,
    z: np.ndarray,
    params: TransonicSlimParams,
    *,
    seed_kind: str,
    seed_full: float,
    initial_full: float,
    elapsed_s: float,
    polish,
    picard_history: list[dict[str, Any]],
) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(z, params)
    full = max_residual(z, params)
    return {
        "label": label,
        "ratio": float(params.mdot_edd_ratio),
        "N": int(params.n_nodes),
        "full": float(full),
        "accepted": bool(full <= ACCEPTANCE_TOL),
        "anchor": bool(full <= ANCHOR_TOL),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "seed_kind": seed_kind,
        "seed_full": float(seed_full),
        "initial_full": float(initial_full),
        "pivot": str(polish.pivot) if polish is not None else "-",
        "nfev": int(polish.result.nfev) if polish is not None else 0,
        "picard_iterations": int(len(picard_history)),
        "picard_slope_delta": float(picard_history[-1]["slope_delta"]) if picard_history else 0.0,
        "elapsed_s": float(elapsed_s),
        "picard_history": picard_history,
        **diagnostics(z, params),
    }


def save_checkpoint(label: str, z: np.ndarray, params: TransonicSlimParams, row: dict[str, Any]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe = label.replace(".", "p").replace("-", "m")
    np.savez_compressed(
        CHECKPOINT_DIR / f"{safe}.npz",
        z=np.asarray(z, dtype=float),
        ratio=np.array(params.mdot_edd_ratio),
        R_out_rg=np.array(params.R_out_rg),
        n_nodes=np.array(params.n_nodes),
        grid_power=np.array(params.grid_power),
        custom_grid_xi=np.asarray(params.custom_grid_xi, dtype=float)
        if params.custom_grid_xi is not None
        else np.asarray([], dtype=float),
        outer_closure=np.array(params.outer_closure),
        outer_match_log_slopes=np.asarray(params.outer_match_log_slopes, dtype=float),
        full=np.array(row["full"]),
        accepted=np.array(row["accepted"]),
        row_json=np.array(json.dumps(json_safe({k: v for k, v in row.items() if k != "picard_history"}), sort_keys=True)),
    )


def write_table(rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim High-Mdot Newton Scout",
        "",
        "Generated by `scripts/run_standard_slim_high_mdot_newton_scout.py`.",
        "",
        f"Targets `{TARGETS}`, acceptance `{ACCEPTANCE_TOL:g}`, anchor `{ANCHOR_TOL:g}`, slope Picard `{SLOPE_PICARD_MAX_ITER}`.",
        "",
        "| label | Mdot/Edd | full | accepted | anchor | dominant | seed | seed full | int R | int E | outer omega | f_adv global | f_adv inner | f_adv pos | Lrad/LEdd | Lvisc/LEdd | max H/R | min tau | Rson/rg | Picard | slope delta | pivot | nfev | elapsed s |",
        "|---|---:|---:|:---:|:---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {label} | {ratio} | {full} | {accepted} | {anchor} | {dominant} | {seed_kind} | {seed_full} | "
            "{interval_R} | {interval_E} | {outer_omega} | {f_adv_global} | {f_adv_inner} | {f_adv_pos} | "
            "{Lrad_LEdd} | {Lvisc_LEdd} | {max_H_R} | {min_tau} | {Rson_rg} | {picard_iterations} | "
            "{picard_slope_delta} | {pivot} | {nfev} | {elapsed_s} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps(json_safe(rows), indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    if not rows:
        return
    image = Image.new("RGB", (1100, 680), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    panels = [
        (70, 60, 510, 300, "advection", ("f_adv_global", "f_adv_inner", "f_adv_pos"), False),
        (590, 60, 1030, 300, "luminosity", ("Lrad_LEdd", "Lvisc_LEdd"), False),
        (70, 370, 510, 620, "max H/R", ("max_H_R",), False),
        (590, 370, 1030, 620, "residual", ("full",), True),
    ]
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44)]
    ratios = np.asarray([float(row["ratio"]) for row in rows], dtype=float)
    xvals = np.log10(ratios)
    x_min, x_max = float(np.min(xvals)), float(np.max(xvals))
    if x_max <= x_min:
        x_min -= 0.05
        x_max += 0.05
    for x0, y0, x1, y1, title, keys, logy in panels:
        draw.rectangle((x0, y0, x1, y1), outline=(80, 80, 80), width=1)
        draw.text((x0 + 8, y0 + 6), title, fill=(20, 20, 20), font=font)
        ydata = []
        for key in keys:
            ydata.extend([float(row[key]) for row in rows if np.isfinite(float(row[key]))])
        if not ydata:
            continue
        if logy:
            mapped = np.log10(np.maximum(np.abs(ydata), 1.0e-16))
            y_min, y_max = float(np.floor(np.min(mapped))), float(np.ceil(np.max(mapped)))
        else:
            y_min, y_max = float(np.min(ydata)), float(np.max(ydata))
            pad = 0.1 * max(y_max - y_min, 1.0e-8)
            y_min -= pad
            y_max += pad
        if y_max <= y_min:
            y_max = y_min + 1.0

        def px(ratio: float) -> int:
            return x0 + 42 + int((np.log10(ratio) - x_min) / (x_max - x_min) * (x1 - x0 - 65))

        def py(value: float) -> int:
            yy = np.log10(max(abs(value), 1.0e-16)) if logy else value
            return y1 - 28 - int((yy - y_min) / (y_max - y_min) * (y1 - y0 - 62))

        for idx, key in enumerate(keys):
            color = colors[idx % len(colors)]
            pts = [(px(float(row["ratio"])), py(float(row[key]))) for row in rows if np.isfinite(float(row[key]))]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)
            for point in pts:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
            draw.text((x1 - 130, y0 + 22 + 13 * idx), key, fill=color, font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    current_z, current_params = load_anchor(fiducial, mdot_edd)
    rows: list[dict[str, Any]] = []
    anchor_row = row_for(
        "anchor",
        current_z,
        current_params,
        seed_kind="-",
        seed_full=np.nan,
        initial_full=max_residual(current_z, current_params),
        elapsed_s=0.0,
        polish=None,
        picard_history=[],
    )
    rows.append(anchor_row)
    write_table(rows)
    write_figure(rows)
    previous_z = None
    previous_params = None
    base = {
        "R_out_rg": current_params.R_out_rg,
        "n_nodes": current_params.n_nodes,
        "grid_power": current_params.grid_power,
        "custom_grid_xi": current_params.custom_grid_xi,
    }
    for target in TARGETS:
        if target <= current_params.mdot_edd_ratio * (1.0 + 1.0e-12):
            continue
        target_params = params_for(
            fiducial,
            mdot_edd,
            target,
            slopes=one_sided_outer_slopes(current_z, current_params),
            **base,
        )
        seed_kind, seed, seed_full = best_seed(
            previous_z=previous_z,
            previous_params=previous_params,
            current_z=current_z,
            current_params=current_params,
            target_params=target_params,
        )
        print(f"target={target:g} seed={seed_kind} seed_full={seed_full:.3e}", flush=True)
        t0 = time.perf_counter()
        polish, solved_params, history = slope_picard_polish(seed, target_params)
        elapsed = time.perf_counter() - t0
        row = row_for(
            f"mdot_{target:g}",
            polish.z,
            solved_params,
            seed_kind=seed_kind,
            seed_full=seed_full,
            initial_full=seed_full,
            elapsed_s=elapsed,
            polish=polish,
            picard_history=history,
        )
        rows.append(row)
        write_table(rows)
        write_figure(rows)
        save_checkpoint(f"mdot_{target:g}", polish.z, solved_params, row)
        print(
            f"  final={row['full']:.3e} accepted={row['accepted']} dom={row['dominant']} "
            f"f_adv_inner={row['f_adv_inner']:.3e} Lrad={row['Lrad_LEdd']:.3e}",
            flush=True,
        )
        if not row["accepted"]:
            print("  stopping at first non-accepted target", flush=True)
            break
        previous_z = current_z
        previous_params = current_params
        current_z = np.asarray(polish.z, dtype=float)
        current_params = solved_params
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
