"""Certification attempts for selected high-Mdot no-wind slim checkpoints."""

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
    remap_profile_to_new_sonic_grid,
    residual_audit_from_state_vector,
    solve_square_transonic_polish,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_luminosity, eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant


ROOT = Path(__file__).resolve().parents[1]
CASE_SPECS = tuple(
    piece.strip()
    for piece in os.environ.get(
        "IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_CASES",
        "m2:outputs/checkpoints/slim_benchmark_high_mdot_no_wind_newton_scout_N512_1p45_to2_loose/mdot_2.npz;"
        "m3:outputs/checkpoints/slim_benchmark_high_mdot_no_wind_newton_scout_N512_2_to3_loose/mdot_3.npz",
    )
    .replace(",", ";")
    .split(";")
    if piece.strip()
)
N_VALUES = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_N_VALUES", "512").replace(":", ",").split(",")
    if piece.strip()
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_TABLE",
    "outputs/tables/slim_benchmark_high_mdot_no_wind_certification.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_FIGURE",
    "outputs/figures/slim_benchmark_high_mdot_no_wind_certification.png",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_high_mdot_no_wind_certification",
)
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_ANCHOR_TOL", "3e-6"))
NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_NEWTON_MAX_ITER", "24"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_NEWTON_MAX_NFEV", "3200"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_NEWTON_MAX_STEP_NORM", "0.10"))
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_LINEAR_SOLVER", "regularized_lsmr")
SLOPE_PICARD_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_SLOPE_PICARD", "4"))
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_PIVOTS", "C2,C1").replace(":", ",").split(",")
    if piece.strip()
)
INNER_RADIUS_RG = float(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_CERTIFY_INNER_RG", "20.0"))


def parse_case_specs() -> list[tuple[str, Path]]:
    cases = []
    for spec in CASE_SPECS:
        if ":" not in spec:
            raise ValueError(f"case spec must be label:path, got {spec!r}")
        label, path = spec.split(":", 1)
        cases.append((label.strip(), ROOT / path.strip()))
    return cases


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


def load_checkpoint(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    data = np.load(path, allow_pickle=True)
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
    z = np.asarray(data["z"], dtype=float)
    return z, replace(params, outer_match_log_slopes=one_sided_outer_slopes(z, params))


def remap_to_n(z: np.ndarray, params: TransonicSlimParams, target_n: int, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    if int(target_n) == int(params.n_nodes):
        target_params = replace(params, outer_match_log_slopes=one_sided_outer_slopes(z, params))
        return np.asarray(z, dtype=float), target_params
    target_params = params_for(
        fiducial,
        mdot_edd,
        params.mdot_edd_ratio,
        R_out_rg=params.R_out_rg,
        n_nodes=int(target_n),
        grid_power=params.grid_power,
        custom_grid_xi=None,
        slopes=None,
    )
    profile = transonic_profile_from_state_vector(z, params)
    seed = remap_profile_to_new_sonic_grid(profile, target_params, temperature_mdot_power=0.0)
    target_params = replace(target_params, outer_match_log_slopes=one_sided_outer_slopes(seed, target_params))
    return seed, target_params


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


def polish_once(z0: np.ndarray, params: TransonicSlimParams):
    best = None
    best_full = np.inf
    for pivot in PIVOTS:
        pivot_initial = max_residual(z0, params)
        print(f"    pivot={pivot} initial_full={pivot_initial:.3e}", flush=True)
        t0 = time.perf_counter()
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
        elapsed = time.perf_counter() - t0
        full = max_residual(result.z, params)
        print(
            f"    pivot={result.pivot} full={full:.3e} square={result.final_square_max_residual:.3e} "
            f"unused={result.unused_compatibility:.3e} it={result.iterations} nfev={result.result.nfev} "
            f"elapsed={elapsed:.1f}s msg={result.result.message}",
            flush=True,
        )
        if full < best_full:
            best = result
            best_full = full
        if full <= ANCHOR_TOL:
            break
    if best is None:
        raise RuntimeError("no certification pivots configured")
    return best


def slope_picard_polish(z0: np.ndarray, params: TransonicSlimParams) -> tuple[Any, TransonicSlimParams, list[dict[str, Any]]]:
    z = np.asarray(z0, dtype=float)
    current_params = params
    history: list[dict[str, Any]] = []
    polish = None
    for index in range(max(1, SLOPE_PICARD_MAX_ITER)):
        print(f"  picard={index} start_full={max_residual(z, current_params):.3e}", flush=True)
        t0 = time.perf_counter()
        polish = polish_once(z, current_params)
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
                "nfev": int(polish.result.nfev),
                "message": str(polish.result.message),
            }
        )
        print(
            f"  picard={index} solved_full={solved_full:.3e} refreshed_full={refreshed_full:.3e} "
            f"slope_delta={slope_delta:.3e} elapsed={elapsed:.1f}s",
            flush=True,
        )
        z = np.asarray(polish.z, dtype=float)
        current_params = refreshed_params
        if refreshed_full <= ANCHOR_TOL and slope_delta <= 1.0e-3:
            break
    if polish is None:
        raise RuntimeError("certification polish produced no result")
    return polish, current_params, history


def trapz_log(values: np.ndarray, R: np.ndarray) -> float:
    logR = np.log(np.asarray(R, dtype=float))
    weights = 2.0 * np.pi * np.asarray(R, dtype=float) ** 2
    return float(np.trapezoid(np.asarray(values, dtype=float) * weights, logR))


def diagnostics(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    profile = transonic_profile_from_state_vector(z, params)
    R = np.asarray(profile.R, dtype=float)
    R_rg = R / params.r_g
    qv = np.asarray(profile.Q_visc, dtype=float)
    qr = np.asarray(profile.Q_rad, dtype=float)
    qa = np.asarray(profile.Q_adv, dtype=float)
    visc = trapz_log(np.abs(qv), R) + 1.0e-300
    inner = R_rg <= INNER_RADIUS_RG
    if int(np.count_nonzero(inner)) >= 2:
        inner_visc = trapz_log(np.abs(qv[inner]), R[inner]) + 1.0e-300
        inner_adv = trapz_log(qa[inner], R[inner])
    else:
        inner_adv = np.nan
        inner_visc = np.nan
    ledd = eddington_luminosity(params.M2_g, kappa=params.kappa)
    return {
        "f_adv_global": float(trapz_log(qa, R) / visc),
        "f_adv_pos": float(trapz_log(np.maximum(qa, 0.0), R) / visc),
        "f_adv_inner": float(inner_adv / inner_visc) if np.isfinite(inner_visc) else np.nan,
        "Lrad_LEdd": float(trapz_log(qr, R) / ledd),
        "Lvisc_LEdd": float(trapz_log(qv, R) / ledd),
        "max_H_R": float(np.max(profile.H_over_R)),
        "min_tau": float(np.min(profile.tau)),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0": float(profile.lambda0),
    }


def row_for(
    *,
    label: str,
    source_path: Path,
    source_n: int,
    seed: np.ndarray,
    seed_params: TransonicSlimParams,
    polish,
    params: TransonicSlimParams,
    history: list[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(polish.z, params)
    full = max_residual(polish.z, params)
    return {
        "label": label,
        "source_checkpoint": str(source_path.relative_to(ROOT)),
        "source_N": int(source_n),
        "N": int(params.n_nodes),
        "ratio": float(params.mdot_edd_ratio),
        "seed_full": max_residual(seed, seed_params),
        "full": float(full),
        "accepted": bool(full <= ACCEPTANCE_TOL),
        "anchor": bool(full <= ANCHOR_TOL),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "sonic_D": float(audit.sonic_D),
        "sonic_K": float(audit.sonic_K),
        "smin_over_smax": float(audit.sonic_smin_over_smax),
        "picard_iterations": int(len(history)),
        "picard_slope_delta": float(history[-1]["slope_delta"]) if history else np.nan,
        "last_solved_full": float(history[-1]["solved_full"]) if history else np.nan,
        "pivot": str(polish.pivot),
        "nfev": int(polish.result.nfev),
        "iterations": int(polish.iterations),
        "elapsed_s": float(elapsed_s),
        "message": str(polish.result.message),
        "picard_history": history,
        **diagnostics(polish.z, params),
    }


def save_checkpoint(row: dict[str, Any], z: np.ndarray, params: TransonicSlimParams) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{row['label']}_N{int(params.n_nodes)}".replace(".", "p").replace("-", "m")
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
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
        "# Standard Slim High-Mdot Checkpoint Certification",
        "",
        "Generated by `scripts/run_standard_slim_high_mdot_checkpoint_certification.py`.",
        "",
        f"N values `{N_VALUES}`, acceptance `{ACCEPTANCE_TOL:g}`, anchor `{ANCHOR_TOL:g}`, slope Picard `{SLOPE_PICARD_MAX_ITER}`.",
        "",
        "| label | Mdot/Edd | source N | N | seed full | full | accepted | anchor | dominant | int R | int E | outer omega | f_adv inner | f_adv pos | Lrad/LEdd | max H/R | Rson/rg | Picard | slope delta | pivot | nfev | elapsed s | message |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {label} | {ratio} | {source_N} | {N} | {seed_full} | {full} | {accepted} | {anchor} | {dominant} | "
            "{interval_R} | {interval_E} | {outer_omega} | {f_adv_inner} | {f_adv_pos} | {Lrad_LEdd} | "
            "{max_H_R} | {Rson_rg} | {picard_iterations} | {picard_slope_delta} | {pivot} | {nfev} | {elapsed_s} | {message} |".format(
                **formatted
            )
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
    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = 90, 70, 930, 530
    draw.rectangle((x0, y0, x1, y1), outline=(70, 70, 70), width=1)
    ratios = np.asarray([float(row["ratio"]) for row in rows], dtype=float)
    residuals = np.asarray([float(row["full"]) for row in rows], dtype=float)
    xvals = np.log10(ratios)
    yvals = np.log10(np.maximum(residuals, 1.0e-16))
    x_min, x_max = float(np.min(xvals)), float(np.max(xvals))
    if x_max <= x_min:
        x_min -= 0.05
        x_max += 0.05
    y_min, y_max = float(np.floor(np.min(yvals))), float(np.ceil(np.max(yvals)))
    if y_max <= y_min:
        y_max = y_min + 1.0
    colors = {512: (31, 119, 180), 640: (214, 39, 40), 768: (44, 160, 44)}
    for row in rows:
        px = x0 + int((np.log10(float(row["ratio"])) - x_min) / (x_max - x_min) * (x1 - x0))
        py = y1 - int((np.log10(max(float(row["full"]), 1.0e-16)) - y_min) / (y_max - y_min) * (y1 - y0))
        color = colors.get(int(row["N"]), (80, 80, 80))
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
        draw.text((px + 7, py - 6), f"{row['label']} N{row['N']}", fill=color, font=font)
    for tol, label in ((ACCEPTANCE_TOL, "accept"), (ANCHOR_TOL, "anchor")):
        py = y1 - int((np.log10(tol) - y_min) / (y_max - y_min) * (y1 - y0))
        draw.line((x0, py, x1, py), fill=(120, 120, 120), width=1)
        draw.text((x1 - 75, py - 13), label, fill=(80, 80, 80), font=font)
    draw.text((90, 25), "High-Mdot checkpoint certification residuals", fill=(20, 20, 20), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows: list[dict[str, Any]] = []
    for label, path in parse_case_specs():
        source_z, source_params = load_checkpoint(path, fiducial, mdot_edd)
        source_n = int(source_params.n_nodes)
        current_z = source_z
        current_params = source_params
        for target_n in N_VALUES:
            seed, seed_params = remap_to_n(current_z, current_params, int(target_n), fiducial, mdot_edd)
            print(f"{label} N{target_n} seed_full={max_residual(seed, seed_params):.3e}", flush=True)
            t0 = time.perf_counter()
            polish, solved_params, history = slope_picard_polish(seed, seed_params)
            elapsed = time.perf_counter() - t0
            row = row_for(
                label=label,
                source_path=path,
                source_n=source_n,
                seed=seed,
                seed_params=seed_params,
                polish=polish,
                params=solved_params,
                history=history,
                elapsed_s=elapsed,
            )
            rows.append(row)
            save_checkpoint(row, polish.z, solved_params)
            write_table(rows)
            write_figure(rows)
            print(
                f"  full={row['full']:.3e} accepted={row['accepted']} anchor={row['anchor']} "
                f"dom={row['dominant']} f_adv_inner={row['f_adv_inner']:.3e}",
                flush=True,
            )
            current_z = np.asarray(polish.z, dtype=float)
            current_params = solved_params
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
