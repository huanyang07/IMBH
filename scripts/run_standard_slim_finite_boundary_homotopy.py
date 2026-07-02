"""Finite-Rout homotopy from a high-rate standard slim-disk checkpoint."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

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
from imri_qpe.scales import eddington_luminosity, eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_FINITE_BOUNDARY_ANCHOR",
    "outputs/checkpoints/slim_benchmark_mesh_closure_validation_pressure_N640_grid060_0p93_1_certify/m1000_N640_pressure_one_sided_mdot_1.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_FINITE_BOUNDARY_TABLE",
    "outputs/tables/slim_benchmark_finite_boundary_homotopy.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_FINITE_BOUNDARY_FIGURE",
    "outputs/figures/slim_benchmark_finite_boundary_homotopy.png",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_FINITE_BOUNDARY_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_finite_boundary_homotopy",
)

R_OUT_LADDER = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_ROUT_LADDER", "10000,7000,5000,3000,1000")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
N_VALUES_RAW = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_N_VALUES", "")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
GRID_POWER = float(os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_GRID_POWER", "0.6"))
OUTER_CLOSURE_MODE = os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_OUTER_CLOSURE", "pressure_one_sided")
NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_NEWTON_MAX_ITER", "24"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_NEWTON_MAX_NFEV", "1800"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_NEWTON_MAX_STEP_NORM", "0.16"))
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_NEWTON_LINEAR_SOLVER", "regularized_lsmr")
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_ANCHOR_TOL", "3e-6"))
REFRESH_REPOLISH = os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_REFRESH_REPOLISH", "0") != "0"
INNER_RADIUS_RG = float(os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_INNER_RG", "20.0"))
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_PIVOTS", "C2,C1").replace(":", ",").split(",")
    if piece.strip()
)

if OUTER_CLOSURE_MODE not in {"thin_value", "pressure_one_sided"}:
    raise ValueError("IMBH_STANDARD_SLIM_FINITE_BOUNDARY_OUTER_CLOSURE must be thin_value or pressure_one_sided")
if len(N_VALUES_RAW) not in {0, 1, len(R_OUT_LADDER)}:
    raise ValueError("N_VALUES must be empty, one integer, or match the Rout ladder length")


def n_for_index(index: int, fallback: int) -> int:
    if not N_VALUES_RAW:
        return int(fallback)
    if len(N_VALUES_RAW) == 1:
        return int(N_VALUES_RAW[0])
    return int(N_VALUES_RAW[index])


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    *,
    ratio: float,
    R_out_rg: float,
    n_nodes: int,
    custom_grid_xi: tuple[float, ...] | None = None,
) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(ratio) * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=float(R_out_rg),
        n_nodes=int(n_nodes),
        grid_power=GRID_POWER,
        custom_grid_xi=custom_grid_xi,
        max_nfev=NEWTON_MAX_NFEV,
        residual_tol=1.0e-8,
        outer_closure="thin_value",
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


def one_sided_outer_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    dx = float(logR[-1] - logR[-2])
    return float((logu[-1] - logu[-2]) / dx), float((logT[-1] - logT[-2]) / dx)


def apply_outer_closure_from_state(z: np.ndarray, params: TransonicSlimParams) -> TransonicSlimParams:
    if OUTER_CLOSURE_MODE == "thin_value":
        return replace(params, outer_closure="thin_value", outer_match_log_slopes=None)
    return replace(
        params,
        outer_closure="pressure_supported_thin_energy",
        outer_match_log_slopes=one_sided_outer_slopes(z, params),
    )


def remap_seed(source_z: np.ndarray, source_params: TransonicSlimParams, target_params: TransonicSlimParams) -> np.ndarray:
    if (
        source_params.n_nodes == target_params.n_nodes
        and np.isclose(source_params.R_out_rg, target_params.R_out_rg)
        and np.isclose(source_params.grid_power, target_params.grid_power)
    ):
        return np.array(source_z, copy=True)
    profile = transonic_profile_from_state_vector(source_z, source_params)
    return remap_profile_to_new_sonic_grid(profile, target_params, temperature_mdot_power=0.0)


def load_checkpoint(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    data = np.load(path, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    ratio = float(data["ratio"])
    custom_grid_xi = None
    if "custom_grid_xi" in data:
        candidate_grid = np.asarray(data["custom_grid_xi"], dtype=float)
        if candidate_grid.shape == (int(data["n_nodes"]),):
            custom_grid_xi = tuple(float(value) for value in candidate_grid)
    params = params_for(
        fiducial,
        mdot_edd,
        ratio=ratio,
        R_out_rg=float(data["R_out_rg"]),
        n_nodes=int(data["n_nodes"]),
        custom_grid_xi=custom_grid_xi,
    )
    if "grid_power" in data:
        params = replace(params, grid_power=float(data["grid_power"]))
    return z, apply_outer_closure_from_state(z, params)


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
        raise RuntimeError("no pivots configured")
    return best


def pressure_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    if params.outer_match_log_slopes is None:
        return {"outer_pressure_target": np.nan, "outer_pressure_residual": np.nan}
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    ln_omega = float(np.log(profile.Omega[-1] / profile.Omega_K[-1]))
    target = pressure_supported_omega_target(
        float(logR[-1]),
        np.array([logu[-1], logT[-1]], dtype=float),
        np.asarray(params.outer_match_log_slopes, dtype=float),
        lambda0,
        params,
    ) + float(params.outer_omega_log_offset)
    return {"outer_pressure_target": float(target), "outer_pressure_residual": float(ln_omega - target)}


def trapz_log(values: np.ndarray, R: np.ndarray) -> float:
    logR = np.log(np.asarray(R, dtype=float))
    weights = 2.0 * np.pi * np.asarray(R, dtype=float) ** 2
    return float(np.trapezoid(np.asarray(values, dtype=float) * weights, logR))


def masked_trapz_log(values: np.ndarray, R: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    R = np.asarray(R, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if int(np.count_nonzero(mask)) < 2:
        return np.nan
    return trapz_log(values[mask], R[mask])


def advection_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    profile = transonic_profile_from_state_vector(z, params)
    R = np.asarray(profile.R, dtype=float)
    R_rg = R / params.r_g
    qv = np.asarray(profile.Q_visc, dtype=float)
    qr = np.asarray(profile.Q_rad, dtype=float)
    qa = np.asarray(profile.Q_adv, dtype=float)
    visc = trapz_log(np.abs(qv), R) + 1.0e-300
    rad = trapz_log(qr, R)
    adv = trapz_log(qa, R)
    adv_pos = trapz_log(np.maximum(qa, 0.0), R)
    inner = R_rg <= INNER_RADIUS_RG
    inner_visc = masked_trapz_log(np.abs(qv), R, inner)
    inner_adv = masked_trapz_log(qa, R, inner)
    inner_adv_pos = masked_trapz_log(np.maximum(qa, 0.0), R, inner)
    ledd = eddington_luminosity(params.M2_g, kappa=params.kappa)
    return {
        "f_adv_global": float(adv / visc),
        "f_adv_pos": float(adv_pos / visc),
        "f_adv_inner": float(inner_adv / (inner_visc + 1.0e-300)) if np.isfinite(inner_visc) else np.nan,
        "f_adv_inner_pos": float(inner_adv_pos / (inner_visc + 1.0e-300)) if np.isfinite(inner_visc) else np.nan,
        "Lrad_LEdd": float(rad / ledd),
    }


def interval_peak_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    intervals = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            for idx in range(len(logR) - 1)
        ],
        dtype=float,
    )
    R_mid = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    peak_R = int(np.argmax(np.abs(intervals[:, 0])))
    peak_E = int(np.argmax(np.abs(intervals[:, 1])))
    return {
        "peak_interval_R_rg": float(R_mid[peak_R]),
        "peak_interval_R_value": float(intervals[peak_R, 0]),
        "peak_interval_E_rg": float(R_mid[peak_E]),
        "peak_interval_E_value": float(intervals[peak_E, 1]),
        "median_abs_interval_E": float(np.median(np.abs(intervals[:, 1]))),
        "p90_abs_interval_E": float(np.quantile(np.abs(intervals[:, 1]), 0.9)),
    }


def row_for_result(
    *,
    stage: str,
    initial_z: np.ndarray,
    z: np.ndarray,
    params: TransonicSlimParams,
    polish,
    elapsed_s: float,
) -> dict[str, object]:
    audit = residual_audit_from_state_vector(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    full = max_residual(z, params)
    slopes = params.outer_match_log_slopes
    return {
        "stage": stage,
        "ratio": float(params.mdot_edd_ratio),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "grid_power": float(params.grid_power),
        "outer_closure": params.outer_closure,
        "g_u": np.nan if slopes is None else float(slopes[0]),
        "g_T": np.nan if slopes is None else float(slopes[1]),
        "initial_full": max_residual(initial_z, params),
        "final_full": full,
        "accepted": bool(full <= ACCEPTANCE_TOL),
        "anchor_eligible": bool(full <= ANCHOR_TOL),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "outer_H_R": float(audit.outer_H_over_R),
        "outer_Qadv_Qvisc": float(audit.outer_Qadv_over_Qvisc),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0_over_lK_isco": float(audit.lambda0_over_lK_isco),
        "max_H_R": float(np.max(profile.H_over_R)),
        "integrated_adv": float(profile.integrated_advective_fraction),
        **advection_diagnostic(z, params),
        **interval_peak_diagnostic(z, params),
        "pivot": str(polish.pivot),
        "method": str(polish.method),
        "nfev": int(polish.result.nfev),
        "iterations": int(polish.iterations),
        "elapsed_s": float(elapsed_s),
        "message": str(polish.result.message),
        **pressure_diagnostic(z, params),
        "z": np.asarray(z, dtype=float),
    }


def save_checkpoint(row: dict[str, object], params: TransonicSlimParams) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"Rout_{float(row['R_out_rg']):.8g}_mdot_{float(row['ratio']):.8g}_N{int(row['N'])}".replace(".", "p")
    slopes = params.outer_match_log_slopes
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(row["R_out_rg"]),
        n_nodes=np.array(row["N"]),
        grid_power=np.array(params.grid_power),
        custom_grid_xi=np.asarray(params.custom_grid_xi, dtype=float)
        if params.custom_grid_xi is not None
        else np.asarray([], dtype=float),
        outer_closure=np.array(params.outer_closure),
        outer_match_log_slopes=np.asarray([np.nan, np.nan] if slopes is None else slopes, dtype=float),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted"]),
        row_json=np.array(json.dumps(json_safe({key: value for key, value in row.items() if key != "z"}), sort_keys=True)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Finite-Boundary Homotopy",
        "",
        "Generated by `scripts/run_standard_slim_finite_boundary_homotopy.py`.",
        "",
        f"Anchor `{ANCHOR_CHECKPOINT.relative_to(ROOT)}`, Rout ladder `{','.join(f'{value:g}' for value in R_OUT_LADDER)}`, "
        f"N values `{','.join(str(value) for value in N_VALUES_RAW) if N_VALUES_RAW else 'anchor N'}`, "
        f"grid power `{GRID_POWER:g}`, closure `{OUTER_CLOSURE_MODE}`, refresh repolish `{REFRESH_REPOLISH}`.",
        "",
        "| stage | Mdot/Edd | Rout/rg | N | grid power | initial full | final full | accepted | anchor | dominant | int R | int E | peak E R/rg | median abs E | outer omega | pressure mismatch | f_adv global | f_adv inner | f_adv pos | Lrad/LEdd | max H/R | int adv | Rson/rg | lambda/lK | pivot | nfev | elapsed s | message |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {stage} | {ratio} | {R_out_rg} | {N} | {grid_power} | {initial_full} | {final_full} | "
            "{accepted} | {anchor_eligible} | {dominant} | {interval_R} | {interval_E} | "
            "{peak_interval_E_rg} | {median_abs_interval_E} | {outer_omega} | {outer_pressure_residual} | "
            "{f_adv_global} | {f_adv_inner} | {f_adv_pos} | {Lrad_LEdd} | "
            "{max_H_R} | {integrated_adv} | {Rson_rg} | {lambda0_over_lK_isco} | "
            "{pivot} | {nfev} | {elapsed_s} | {message} |".format(**formatted).replace("\n", " ")
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(
        json.dumps(json_safe([{key: value for key, value in row.items() if key != "z"} for row in rows]), indent=2, sort_keys=True) + "\n"
    )


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    if not rows:
        return
    width, height = 1000, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = 90, 80, 930, 540
    draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
    x_vals = np.log10(np.asarray([float(row["R_out_rg"]) for row in rows], dtype=float))
    y_vals = np.log10(np.maximum(np.asarray([float(row["final_full"]) for row in rows], dtype=float), 1.0e-16))
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    y_min, y_max = float(np.floor(np.min(y_vals))), float(np.ceil(np.max(y_vals)))
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    points = []
    for row in rows:
        xx = np.log10(float(row["R_out_rg"]))
        yy = np.log10(max(float(row["final_full"]), 1.0e-16))
        px = x0 + int((xx - x_min) / (x_max - x_min) * (x1 - x0))
        py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
        points.append((px, py, row))
    if len(points) >= 2:
        draw.line([(px, py) for px, py, _row in points], fill=(31, 119, 180), width=3)
    for px, py, row in points:
        color = (31, 119, 180) if row["accepted"] else (214, 39, 40)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
        draw.text((px + 6, py - 12), fmt(row["R_out_rg"]), fill=(20, 20, 20), font=font)
    for tol, label in ((ACCEPTANCE_TOL, "accept"), (ANCHOR_TOL, "anchor")):
        yy = np.log10(tol)
        py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
        draw.line((x0, py, x1, py), fill=(120, 120, 120), width=1)
        draw.text((x1 - 72, py - 14), label, fill=(80, 80, 80), font=font)
    draw.text((90, 25), "Finite-boundary homotopy: final residual vs Rout", fill=(20, 20, 20), font=font)
    draw.text((x0 + 4, y0 + 4), f"1e{int(y_max)}", fill=(80, 80, 80), font=font)
    draw.text((x0 + 4, y1 - 18), f"1e{int(y_min)}", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_z, source_params = load_checkpoint(ANCHOR_CHECKPOINT, fiducial, mdot_edd)
    rows: list[dict[str, object]] = []
    for index, R_out_rg in enumerate(R_OUT_LADDER):
        n_nodes = n_for_index(index, source_params.n_nodes)
        target_base = params_for(
            fiducial,
            mdot_edd,
            ratio=source_params.mdot_edd_ratio,
            R_out_rg=R_out_rg,
            n_nodes=n_nodes,
            custom_grid_xi=source_params.custom_grid_xi if int(n_nodes) == int(source_params.n_nodes) else None,
        )
        seed = remap_seed(source_z, source_params, target_base)
        target_params = apply_outer_closure_from_state(seed, target_base)
        print(
            f"Rout={R_out_rg:g} N={n_nodes} initial={max_residual(seed, target_params):.3e}",
            flush=True,
        )
        t0 = time.perf_counter()
        polish = polish_best(seed, target_params)
        final_params = apply_outer_closure_from_state(polish.z, target_params)
        if REFRESH_REPOLISH:
            polish = polish_best(polish.z, final_params)
            final_params = apply_outer_closure_from_state(polish.z, final_params)
        elapsed = time.perf_counter() - t0
        row = row_for_result(
            stage="finite_boundary",
            initial_z=seed,
            z=polish.z,
            params=final_params,
            polish=polish,
            elapsed_s=elapsed,
        )
        rows.append(row)
        save_checkpoint(row, final_params)
        write_table(rows)
        write_figure(rows)
        print(
            f"  final={row['final_full']:.3e} dom={row['dominant']} "
            f"Rson={row['Rson_rg']:.4g} accepted={row['accepted']} anchor={row['anchor_eligible']}",
            flush=True,
        )
        source_z = np.asarray(polish.z, dtype=float)
        source_params = final_params
        if not row["accepted"]:
            print("  stopping at first non-accepted finite-boundary step", flush=True)
            break
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
