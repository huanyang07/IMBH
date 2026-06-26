"""Scan true soft-prior weights for outer-slope unknowns."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _differential_interval_residual_from_unpacked,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    state_bounds,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_full_slope_tuned_sonic_audit import residual_vector
from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_slope_unknown_root import (
    SOURCE_CHECKPOINT,
    active_physical_max,
    branch_metrics,
    dominant_block,
    extended_residual,
    extended_sparsity,
    fit_outer_slopes,
    load_source,
    params_for,
    remap_state_pchip,
    unpack_unknown,
)


ROOT = Path(__file__).resolve().parents[1]
N65_SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_slope_unknown_root" / "n8_medium_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_slope_prior_scan.md"
FIGURE_OUTPUT = ROOT / "outputs" / "figures" / "transonic_slope_prior_scan.png"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_slope_prior_scan"

SIGMAS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2)
SLOPE_BOUND_HALF_WIDTH = 8.0e-2
PRIOR_MODE = "n_fit"
PRIOR_WINDOW_FRACTION = 1.0
PRIOR_MIN_POINTS = 8
MAX_NFEV_N65 = 600
MAX_NFEV_LOCKED = 220
MAX_NFEV_RELEASE = 850
SONIC_MODE = "symmetric"
SONIC_PIVOT = "C1"
SONIC_COMPONENTS = ("D", "C1", "C2")

LOCK_LOGU_HALF_WIDTH = 6.0e-2
LOCK_LOGT_HALF_WIDTH = 4.0e-2
LOCK_RSON_RG_HALF_WIDTH = 5.0e-2
LOCK_LAMBDA_HALF_WIDTH = 1.0e-3


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def load_checkpoint(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), json.loads(str(data["row_json"].item()))


def source_from_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "ratio": float(row["ratio"]),
        "R_out_rg": float(row["R_out_rg"]),
        "interval_form": "differential",
        "integrated_weighting": "none",
        "g_u": float(row["g_u_solved"]),
        "g_T": float(row["g_T_solved"]),
    }


def full_bounds(params: TransonicSlimParams, prior_slopes: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = state_bounds(params)
    slope_lower = np.asarray(prior_slopes, dtype=float) - SLOPE_BOUND_HALF_WIDTH
    slope_upper = np.asarray(prior_slopes, dtype=float) + SLOPE_BOUND_HALF_WIDTH
    return np.concatenate([lower, slope_lower]), np.concatenate([upper, slope_upper])


def locked_bounds(
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
    z_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = full_bounds(params, prior_slopes)
    n = params.n_nodes
    logu_seed = z_seed[:n]
    logT_seed = z_seed[n : 2 * n]
    logR_seed = float(z_seed[2 * n])
    lambda_seed = float(z_seed[2 * n + 1])
    rson_seed_rg = float(np.exp(logR_seed) / params.r_g)

    lower[:n] = np.maximum(lower[:n], logu_seed - LOCK_LOGU_HALF_WIDTH)
    upper[:n] = np.minimum(upper[:n], logu_seed + LOCK_LOGU_HALF_WIDTH)
    lower[n : 2 * n] = np.maximum(lower[n : 2 * n], logT_seed - LOCK_LOGT_HALF_WIDTH)
    upper[n : 2 * n] = np.minimum(upper[n : 2 * n], logT_seed + LOCK_LOGT_HALF_WIDTH)

    rson_low = max(params.R_son_bounds_rg[0] + 1.0e-6, rson_seed_rg - LOCK_RSON_RG_HALF_WIDTH)
    rson_high = min(params.R_son_bounds_rg[1] - 1.0e-6, rson_seed_rg + LOCK_RSON_RG_HALF_WIDTH)
    lower[2 * n] = max(lower[2 * n], np.log(rson_low * params.r_g))
    upper[2 * n] = min(upper[2 * n], np.log(rson_high * params.r_g))
    lower[2 * n + 1] = max(lower[2 * n + 1], lambda_seed - LOCK_LAMBDA_HALF_WIDTH)
    upper[2 * n + 1] = min(upper[2 * n + 1], lambda_seed + LOCK_LAMBDA_HALF_WIDTH)
    return lower, upper


def solve_extended(
    x0: np.ndarray,
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
    sigma_slopes: tuple[float, float],
    *,
    bounds: tuple[np.ndarray, np.ndarray],
    max_nfev: int,
):
    lower, upper = bounds
    x_start = np.clip(np.asarray(x0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: extended_residual(trial, params, prior_slopes, sigma_slopes),
        x_start,
        jac_sparsity=extended_sparsity(params),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=max_nfev,
    )


def residual_limit(n_nodes: int) -> float:
    return 2.0e-6 if n_nodes >= 129 else 5.0e-6


def branch_pass(metrics: dict[str, float]) -> bool:
    return bool(
        abs(metrics["delta_Rson_rg"]) <= 5.0e-2
        and abs(metrics["delta_lambda0"]) <= 1.0e-3
        and abs(metrics["delta_int_adv"]) <= 1.0e-2
        and metrics["branch_distance"] <= 4.0
    )


def interval_localization(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float | int | str]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    values = []
    for idx in range(params.n_nodes - 1):
        residual = _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
        R_mid_rg = float(np.exp(0.5 * (logR[idx] + logR[idx + 1])) / params.r_g)
        values.append(
            {
                "idx": idx,
                "R_mid_rg": R_mid_rg,
                "interval_R": float(residual[0]),
                "interval_E": float(residual[1]),
                "abs_max": float(np.max(np.abs(residual))),
            }
        )
    worst = max(values, key=lambda row: row["abs_max"])
    outer = values[-1]
    return {
        "worst_i": int(worst["idx"]),
        "worst_R_mid_rg": float(worst["R_mid_rg"]),
        "worst_interval_abs": float(worst["abs_max"]),
        "worst_block": "interval_R" if abs(float(worst["interval_R"])) >= abs(float(worst["interval_E"])) else "interval_E",
        "outer_i": int(outer["idx"]),
        "outer_R_mid_rg": float(outer["R_mid_rg"]),
        "outer_interval_abs": float(outer["abs_max"]),
        "outer_interval_R": float(outer["interval_R"]),
        "outer_interval_E": float(outer["interval_E"]),
    }


def row_from_result(
    *,
    case: str,
    phase: str,
    sigma: float,
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
    effective_points: int,
    effective_inner_fraction: float,
    z_seed: np.ndarray,
    source_z: np.ndarray,
    source_params: TransonicSlimParams,
    result,
) -> dict[str, object]:
    z, solved_slopes = unpack_unknown(np.asarray(result.x, dtype=float), params)
    solved_params = replace(params, outer_match_log_slopes=solved_slopes)
    base = residual_vector(z, solved_params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS)
    extended = extended_residual(result.x, params, prior_slopes, (sigma, sigma))
    audit = residual_audit_from_state_vector(z, solved_params)
    profile = profile_from_state_vector(z, solved_params)
    prior_residual = np.asarray(
        [
            (solved_slopes[0] - prior_slopes[0]) / sigma,
            (solved_slopes[1] - prior_slopes[1]) / sigma,
        ],
        dtype=float,
    )
    branch = branch_metrics(z, solved_params, source_z, source_params)
    physical = active_physical_max(audit)
    residual_pass = bool(physical <= residual_limit(params.n_nodes))
    branch_ok = branch_pass(branch)
    localization = interval_localization(z, solved_params)
    return {
        "case": case,
        "phase": phase,
        "sigma_g": float(sigma),
        "n_nodes": solved_params.n_nodes,
        "ratio": solved_params.mdot_edd_ratio,
        "R_out_rg": solved_params.R_out_rg,
        "g_u_prior": float(prior_slopes[0]),
        "g_T_prior": float(prior_slopes[1]),
        "g_u_solved": float(solved_slopes[0]),
        "g_T_solved": float(solved_slopes[1]),
        "delta_g_u": float(solved_slopes[0] - prior_slopes[0]),
        "delta_g_T": float(solved_slopes[1] - prior_slopes[1]),
        "prior_max": float(np.max(np.abs(prior_residual))),
        "prior_l2": float(np.sqrt(np.mean(prior_residual**2))),
        "prior_chi2": float(np.dot(prior_residual, prior_residual)),
        "physical_active": physical,
        "base_max": float(np.max(np.abs(base))),
        "extended_max": float(np.max(np.abs(extended))),
        "dominant": dominant_block(audit),
        "residual_pass": residual_pass,
        "branch_pass": branch_ok,
        "science_pass": bool(residual_pass and branch_ok),
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_1": audit.outer_omega,
        "outer_2": audit.outer_energy,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "K": audit.sonic_K,
        "compat_max": max(abs(audit.sonic_C1), abs(audit.sonic_C2), abs(audit.sonic_K)),
        "Rson_rg": float(profile.sonic_radius / solved_params.r_g),
        "lambda0": float(profile.lambda0),
        "int_adv": float(profile.integrated_advective_fraction),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": audit.outer_H_over_R,
        "seed_delta_norm": float(np.linalg.norm(z - z_seed) / np.sqrt(len(z))),
        "effective_points": effective_points,
        "effective_inner_fraction": effective_inner_fraction,
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        **branch,
        **localization,
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    sigma_label = f"{float(row['sigma_g']):.0e}".replace("-", "m").replace("+", "").replace(".", "p")
    stem = f"{row['case']}_{row['phase']}_sigma{sigma_label}_{float(row['ratio']):.8f}".replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        n_nodes=np.array(row["n_nodes"]),
        R_out_rg=np.array(row["R_out_rg"]),
        g_u_solved=np.array(row["g_u_solved"]),
        g_T_solved=np.array(row["g_T_solved"]),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Slope Prior Softness Scan",
        "",
        "Generated by `scripts/run_transonic_slope_prior_scan.py`.",
        "",
        "Scans isotropic soft-prior widths for `g_u_out,g_T_out` while reporting disk physical residuals separately from the prior penalty. `physical` and block columns exclude slope-prior rows; `prior max` and `extended max` show the objective penalty scale.",
        "",
        "| case | phase | N | sigma_g | physical | base max | extended max | prior max | prior chi2 | dominant | residual pass | branch pass | science pass | dg_u | dg_T | g_u solved | g_T solved | Rson/rg | dRson | lambda0 | dlambda0 | int adv | branch dist | worst i | worst R/rg | worst block | worst interval | outer interval | interval R | interval E | outer 1 | D | C1 | C2 | K | nfev | success | message |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {phase} | {n_nodes} | {sigma_g} | {physical_active} | {base_max} | {extended_max} | "
            "{prior_max} | {prior_chi2} | {dominant} | {residual_pass} | {branch_pass} | {science_pass} | "
            "{delta_g_u} | {delta_g_T} | {g_u_solved} | {g_T_solved} | {Rson_rg} | {delta_Rson_rg} | "
            "{lambda0} | {delta_lambda0} | {int_adv} | {branch_distance} | {worst_i} | {worst_R_mid_rg} | "
            "{worst_block} | {worst_interval_abs} | {outer_interval_abs} | {interval_R} | {interval_E} | "
            "{outer_1} | {D} | {C1} | {C2} | {K} | {nfev} | {success} | {message} |".format(
                case=row["case"],
                phase=row["phase"],
                n_nodes=row["n_nodes"],
                sigma_g=fmt(float(row["sigma_g"])),
                physical_active=fmt(float(row["physical_active"])),
                base_max=fmt(float(row["base_max"])),
                extended_max=fmt(float(row["extended_max"])),
                prior_max=fmt(float(row["prior_max"])),
                prior_chi2=fmt(float(row["prior_chi2"])),
                dominant=row["dominant"],
                residual_pass="yes" if row["residual_pass"] else "no",
                branch_pass="yes" if row["branch_pass"] else "no",
                science_pass="yes" if row["science_pass"] else "no",
                delta_g_u=fmt(float(row["delta_g_u"])),
                delta_g_T=fmt(float(row["delta_g_T"])),
                g_u_solved=fmt(float(row["g_u_solved"])),
                g_T_solved=fmt(float(row["g_T_solved"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row["delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row["delta_lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                branch_distance=fmt(float(row["branch_distance"])),
                worst_i=row["worst_i"],
                worst_R_mid_rg=fmt(float(row["worst_R_mid_rg"])),
                worst_block=row["worst_block"],
                worst_interval_abs=fmt(float(row["worst_interval_abs"])),
                outer_interval_abs=fmt(float(row["outer_interval_abs"])),
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                outer_1=fmt(float(row["outer_1"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def load_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_plot(rows: list[dict[str, object]]) -> None:
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scale = 2
    width, height = 1300 * scale, 850 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = load_font(28 * scale, bold=True)
    font_axis = load_font(17 * scale, bold=True)
    font_tick = load_font(14 * scale)
    font_note = load_font(15 * scale)
    text = (35, 39, 47)
    axis = (88, 94, 105)
    grid = (222, 226, 232)
    colors = {"N65_local": (35, 111, 176), "N67_refine": (192, 57, 43)}

    draw.text((62 * scale, 36 * scale), "Soft Outer-Slope Prior Scan", font=font_title, fill=text)
    draw.text(
        (62 * scale, 78 * scale),
        "Physical residual excludes prior rows; scan tests whether freeing slopes removes the outer spike.",
        font=font_note,
        fill=(78, 82, 88),
    )

    panels = [
        ("physical residual", "physical_active", (90 * scale, 140 * scale, 780 * scale, 385 * scale)),
        ("|delta slope|", "delta", (90 * scale, 500 * scale, 780 * scale, 745 * scale)),
    ]
    sigma_min = np.log10(min(float(row["sigma_g"]) for row in rows))
    sigma_max = np.log10(max(float(row["sigma_g"]) for row in rows))

    def x_to_px(sigma: float, box) -> float:
        return box[0] + (np.log10(sigma) - sigma_min) / (sigma_max - sigma_min) * (box[2] - box[0])

    def y_to_px(value: float, box, y_min: float, y_max: float) -> float:
        clipped = min(max(abs(value), 10.0**y_min), 10.0**y_max)
        return box[3] - (np.log10(clipped) - y_min) / (y_max - y_min) * (box[3] - box[1])

    y_ranges = {
        "physical_active": (-7.0, -2.0),
        "delta": (-14.0, -1.0),
    }
    for title, key, box in panels:
        y_min, y_max = y_ranges[key]
        draw.rectangle(box, outline=axis, width=2 * scale)
        draw.text((box[0], box[1] - 32 * scale), title, font=font_axis, fill=text)
        for sigma in SIGMAS:
            x = x_to_px(sigma, box)
            draw.line((x, box[1], x, box[3]), fill=grid, width=1 * scale)
            label = f"{sigma:.0e}".replace("e-0", "e-")
            tw = draw.textlength(label, font=font_tick)
            draw.text((x - tw / 2, box[3] + 10 * scale), label, font=font_tick, fill=axis)
        for power in range(int(y_min), int(y_max) + 1):
            y = y_to_px(10.0**power, box, y_min, y_max)
            draw.line((box[0], y, box[2], y), fill=grid, width=1 * scale)
            label = f"1e{power}"
            tw = draw.textlength(label, font=font_tick)
            draw.text((box[0] - tw - 10 * scale, y - 8 * scale), label, font=font_tick, fill=axis)
        for case in ("N65_local", "N67_refine"):
            case_rows = sorted([row for row in rows if row["case"] == case], key=lambda row: float(row["sigma_g"]))
            points = []
            for row in case_rows:
                value = max(abs(float(row["delta_g_u"])), abs(float(row["delta_g_T"]))) if key == "delta" else float(row[key])
                points.append((x_to_px(float(row["sigma_g"]), box), y_to_px(value, box, y_min, y_max)))
            if len(points) >= 2:
                draw.line(points, fill=colors[case], width=4 * scale)
                for x, y in points:
                    draw.ellipse((x - 4 * scale, y - 4 * scale, x + 4 * scale, y + 4 * scale), fill=colors[case])
        draw.text(((box[0] + box[2]) / 2 - 55 * scale, box[3] + 45 * scale), "sigma_g", font=font_axis, fill=text)

    legend_x = 850 * scale
    legend_y = 165 * scale
    draw.text((legend_x, legend_y - 42 * scale), "Cases", font=font_axis, fill=text)
    for case, color in colors.items():
        draw.line((legend_x, legend_y, legend_x + 42 * scale, legend_y), fill=color, width=5 * scale)
        draw.text((legend_x + 54 * scale, legend_y - 12 * scale), case, font=font_note, fill=text)
        best = min([row for row in rows if row["case"] == case], key=lambda row: float(row["physical_active"]))
        legend_y += 30 * scale
        draw.text(
            (legend_x + 54 * scale, legend_y - 10 * scale),
            f"best physical {fmt(float(best['physical_active']))} at sigma={fmt(float(best['sigma_g']))}",
            font=font_tick,
            fill=(78, 82, 88),
        )
        legend_y += 54 * scale

    note_y = 395 * scale
    draw.text((legend_x, note_y), "Reading", font=font_axis, fill=text)
    for line in [
        "Slope drift stays tiny across the scan.",
        "Residual trends do not reveal a loose-prior cure.",
        "Outer-edge spike is therefore not just hard priors.",
    ]:
        note_y += 30 * scale
        draw.text((legend_x, note_y), line, font=font_note, fill=(78, 82, 88))

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(FIGURE_OUTPUT)


def build_n65_case(fiducial: FiducialParams, mdot_edd: float):
    source_z, source = load_source()
    source_params = params_for(
        fiducial,
        mdot_edd,
        source,
        n_nodes=64,
        slopes=(float(source["g_u"]), float(source["g_T"])),
    )
    seed_params = params_for(
        fiducial,
        mdot_edd,
        source,
        n_nodes=65,
        slopes=(float(source["g_u"]), float(source["g_T"])),
    )
    seed_z = remap_state_pchip(source_z, source_params, seed_params)
    return seed_z, seed_params, source_z, source_params


def build_n67_case(fiducial: FiducialParams, mdot_edd: float):
    n65_z, n65_row = load_checkpoint(N65_SOURCE_CHECKPOINT)
    n65_source = source_from_row(n65_row)
    n65_params = params_for(
        fiducial,
        mdot_edd,
        n65_source,
        n_nodes=65,
        slopes=(float(n65_row["g_u_solved"]), float(n65_row["g_T_solved"])),
    )
    seed_params = params_for(
        fiducial,
        mdot_edd,
        n65_source,
        n_nodes=67,
        slopes=n65_params.outer_match_log_slopes,
    )
    seed_z = remap_state_pchip(n65_z, n65_params, seed_params)
    return seed_z, seed_params, n65_z, n65_params


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    cases = {
        "N65_local": (*build_n65_case(fiducial, mdot_edd), MAX_NFEV_N65),
        "N67_refine": (*build_n67_case(fiducial, mdot_edd), MAX_NFEV_RELEASE),
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    rows: list[dict[str, object]] = []
    for case, (seed_z, seed_params, source_z, source_params, max_nfev) in cases.items():
        prior_gu, prior_gT, effective_points, effective_inner_fraction = fit_outer_slopes(
            seed_z,
            seed_params,
            mode=PRIOR_MODE,
            window_fraction=PRIOR_WINDOW_FRACTION,
            min_points=PRIOR_MIN_POINTS,
        )
        prior_slopes = (prior_gu, prior_gT)
        params = replace(seed_params, outer_match_log_slopes=prior_slopes)
        x_seed = np.concatenate([seed_z, np.asarray(prior_slopes, dtype=float)])
        for sigma in SIGMAS:
            if case == "N67_refine":
                locked = solve_extended(
                    x_seed,
                    params,
                    prior_slopes,
                    (sigma, sigma),
                    bounds=locked_bounds(params, prior_slopes, seed_z),
                    max_nfev=MAX_NFEV_LOCKED,
                )
                result = solve_extended(
                    locked.x,
                    params,
                    prior_slopes,
                    (sigma, sigma),
                    bounds=full_bounds(params, prior_slopes),
                    max_nfev=int(max_nfev),
                )
                phase = "locked_release"
            else:
                result = solve_extended(
                    x_seed,
                    params,
                    prior_slopes,
                    (sigma, sigma),
                    bounds=full_bounds(params, prior_slopes),
                    max_nfev=int(max_nfev),
                )
                phase = "direct"
            row = row_from_result(
                case=case,
                phase=phase,
                sigma=float(sigma),
                params=params,
                prior_slopes=prior_slopes,
                effective_points=effective_points,
                effective_inner_fraction=effective_inner_fraction,
                z_seed=seed_z,
                source_z=source_z,
                source_params=source_params,
                result=result,
            )
            rows.append(row)
            save_checkpoint(row)
            write_table(rows)
            print(
                f"{case} sigma={sigma:.0e} physical={row['physical_active']:.3e} "
                f"prior_max={row['prior_max']:.3e} dg=({row['delta_g_u']:.3e},{row['delta_g_T']:.3e}) "
                f"worst={row['worst_interval_abs']:.3e} R={row['worst_R_mid_rg']:.1f} nfev={row['nfev']}",
                flush=True,
            )

    draw_plot(rows)
    print(f"wrote {TABLE_OUTPUT}")
    print(f"wrote {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
