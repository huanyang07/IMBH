"""Outer reservoir thermal/entropy homotopy for the standard no-wind slim disk."""

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
    residual_audit_from_state_vector,
    solve_square_transonic_polish,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_OUTER_THERMAL_ANCHOR",
    "outputs/checkpoints/slim_benchmark_finite_boundary_homotopy_mdot1_adaptive_mesh_3000_1000/Rout_1000_mdot_1_N640.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_OUTER_THERMAL_TABLE",
    "outputs/tables/slim_benchmark_outer_thermal_homotopy_mdot1_rout1000.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_OUTER_THERMAL_FIGURE",
    "outputs/figures/slim_benchmark_outer_thermal_homotopy_mdot1_rout1000.png",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_OUTER_THERMAL_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_outer_thermal_homotopy_mdot1_rout1000",
)

MODE = os.environ.get("IMBH_STANDARD_SLIM_OUTER_THERMAL_MODE", "temperature")
BRANCH_SPECS = tuple(
    piece.strip()
    for piece in os.environ.get(
        "IMBH_STANDARD_SLIM_OUTER_THERMAL_BRANCHES",
        "cool:0,-0.25,-0.5,-1;hot:0,0.25,0.5,1",
    ).split(";")
    if piece.strip()
)
NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_OUTER_THERMAL_NEWTON_MAX_ITER", "28"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_OUTER_THERMAL_NEWTON_MAX_NFEV", "2600"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_OUTER_THERMAL_NEWTON_MAX_STEP_NORM", "0.16"))
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_OUTER_THERMAL_NEWTON_LINEAR_SOLVER", "regularized_lsmr")
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_OUTER_THERMAL_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_OUTER_THERMAL_ANCHOR_TOL", "3e-6"))
REFRESH_REPOLISH = os.environ.get("IMBH_STANDARD_SLIM_OUTER_THERMAL_REFRESH_REPOLISH", "0") != "0"
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_OUTER_THERMAL_PIVOTS", "C2,C1").replace(":", ",").split(",")
    if piece.strip()
)

if MODE not in {"temperature", "entropy"}:
    raise ValueError("IMBH_STANDARD_SLIM_OUTER_THERMAL_MODE must be temperature or entropy")


def parse_branch_specs() -> list[tuple[str, list[float]]]:
    branches: list[tuple[str, list[float]]] = []
    for spec in BRANCH_SPECS:
        if ":" not in spec:
            raise ValueError(f"branch spec must be label:offsets, got {spec!r}")
        label, offsets = spec.split(":", 1)
        values = [float(piece) for piece in offsets.replace(",", ":").split(":") if piece.strip()]
        if not values:
            raise ValueError(f"branch {label!r} has no offsets")
        branches.append((label.strip(), values))
    return branches


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    *,
    ratio: float,
    R_out_rg: float,
    n_nodes: int,
    grid_power: float,
    custom_grid_xi: tuple[float, ...] | None,
    mode: str,
    target_value: float,
    outer_omega_log_offset: float = 0.0,
) -> TransonicSlimParams:
    closure = "pressure_supported_temperature" if mode == "temperature" else "pressure_supported_entropy"
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
        outer_closure=closure,
        outer_temperature_logT=float(target_value) if mode == "temperature" else None,
        outer_entropy_logK=float(target_value) if mode == "entropy" else None,
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


def apply_outer_slopes_from_state(z: np.ndarray, params: TransonicSlimParams) -> TransonicSlimParams:
    return replace(params, outer_match_log_slopes=one_sided_outer_slopes(z, params))


def custom_grid_from_data(data) -> tuple[float, ...] | None:
    if "custom_grid_xi" not in data:
        return None
    candidate_grid = np.asarray(data["custom_grid_xi"], dtype=float)
    if candidate_grid.shape == (int(data["n_nodes"]),):
        return tuple(float(value) for value in candidate_grid)
    return None


def load_anchor(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    data = np.load(path, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    target = 0.0
    anchor_mode = MODE
    if anchor_mode == "temperature" and "outer_temperature_logT" in data:
        candidate = float(data["outer_temperature_logT"])
        if np.isfinite(candidate):
            target = candidate
    if anchor_mode == "entropy" and "outer_entropy_logK" in data:
        candidate = float(data["outer_entropy_logK"])
        if np.isfinite(candidate):
            target = candidate
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
        custom_grid_xi=custom_grid_from_data(data),
        mode=anchor_mode,
        target_value=target,
        outer_omega_log_offset=outer_omega_log_offset,
    )
    return z, apply_outer_slopes_from_state(z, params)


def entropy_proxy(z: np.ndarray, params: TransonicSlimParams) -> float:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    state = algebraic_state(logR[-1], logu[-1], logT[-1], lambda0, params)
    return float(np.log(state.P + 1.0e-300) - params.gamma_gas * np.log(state.rho + 1.0e-300))


def outer_temperature_logT(z: np.ndarray, params: TransonicSlimParams) -> float:
    _logu, logT, _logR_son, _lambda0, _logR = unpack_state(z, params)
    return float(logT[-1])


def target_base_value(z: np.ndarray, params: TransonicSlimParams, mode: str) -> float:
    return outer_temperature_logT(z, params) if mode == "temperature" else entropy_proxy(z, params)


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


def row_for_result(
    *,
    branch: str,
    offset: float,
    base_value: float,
    target_value: float,
    seed: np.ndarray,
    z: np.ndarray,
    params: TransonicSlimParams,
    polish,
    elapsed_s: float,
) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    achieved_value = target_base_value(z, params, MODE)
    full = max_residual(z, params)
    pressure = pressure_diagnostic(z, params)
    return {
        "branch": branch,
        "mode": MODE,
        "offset": float(offset),
        "base_value": float(base_value),
        "target_value": float(target_value),
        "achieved_value": float(achieved_value),
        "target_residual": float(achieved_value - target_value),
        "temperature_factor": float(np.exp(offset)) if MODE == "temperature" else np.nan,
        "ratio": float(params.mdot_edd_ratio),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "initial_full": max_residual(seed, params),
        "final_full": full,
        "accepted": bool(full <= ACCEPTANCE_TOL),
        "anchor_eligible": bool(full <= ANCHOR_TOL),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        **pressure,
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0_over_lK_isco": float(audit.lambda0_over_lK_isco),
        "max_H_R": float(np.max(profile.H_over_R)),
        "integrated_adv": float(profile.integrated_advective_fraction),
        "outer_H_R": float(audit.outer_H_over_R),
        "outer_Qadv_Qvisc": float(audit.outer_Qadv_over_Qvisc),
        "pivot": str(polish.pivot),
        "method": str(polish.method),
        "nfev": int(polish.result.nfev),
        "iterations": int(polish.iterations),
        "elapsed_s": float(elapsed_s),
        "message": str(polish.result.message),
        "z": np.asarray(z, dtype=float),
        "custom_grid_xi": np.asarray(params.custom_grid_xi, dtype=float)
        if params.custom_grid_xi is not None
        else np.asarray([], dtype=float),
    }


def save_checkpoint(row: dict[str, Any], params: TransonicSlimParams) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_branch = str(row["branch"]).replace(".", "p").replace("-", "m")
    safe_offset = f"{float(row['offset']):.4g}".replace(".", "p").replace("-", "m")
    stem = f"{safe_branch}_{MODE}_offset_{safe_offset}_mdot_{float(row['ratio']):.8g}_N{int(row['N'])}".replace(".", "p")
    slopes = params.outer_match_log_slopes
    payload = {key: value for key, value in row.items() if key not in {"z", "custom_grid_xi"}}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
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
        branch=np.array(row["branch"]),
        mode=np.array(row["mode"]),
        offset=np.array(row["offset"]),
        row_json=np.array(json.dumps(json_safe(payload), sort_keys=True)),
    )


def write_table(rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Outer Thermal Homotopy",
        "",
        "Generated by `scripts/run_standard_slim_outer_thermal_homotopy.py`.",
        "",
        f"Anchor `{ANCHOR_CHECKPOINT.relative_to(ROOT)}`, mode `{MODE}`, branches `{';'.join(BRANCH_SPECS)}`, "
        f"refresh repolish `{REFRESH_REPOLISH}`.",
        "",
        "| branch | mode | offset | temp factor | Mdot/Edd | Rout/rg | initial full | final full | accepted | anchor | dominant | target residual | int R | int E | outer omega | outer thermal | pressure mismatch | max H/R | int adv | Rson/rg | pivot | nfev | elapsed s | message |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {branch} | {mode} | {offset} | {temperature_factor} | {ratio} | {R_out_rg} | {initial_full} | {final_full} | "
            "{accepted} | {anchor_eligible} | {dominant} | {target_residual} | {interval_R} | {interval_E} | "
            "{outer_omega} | {outer_energy} | {outer_pressure_residual} | {max_H_R} | {integrated_adv} | "
            "{Rson_rg} | {pivot} | {nfev} | {elapsed_s} | {message} |".format(**formatted).replace("\n", " ")
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
    width, height = 1000, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = 90, 80, 930, 540
    draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
    offsets = np.asarray([float(row["offset"]) for row in rows], dtype=float)
    residuals = np.log10(np.maximum(np.asarray([float(row["final_full"]) for row in rows], dtype=float), 1.0e-16))
    x_min, x_max = float(np.min(offsets)), float(np.max(offsets))
    if x_max <= x_min:
        x_min -= 0.5
        x_max += 0.5
    y_min, y_max = float(np.floor(np.min(residuals))), float(np.ceil(np.max(residuals)))
    if y_max <= y_min:
        y_max = y_min + 1.0
    colors = {"cool": (31, 119, 180), "hot": (214, 39, 40)}
    for branch in sorted(set(str(row["branch"]) for row in rows)):
        selected = sorted([row for row in rows if row["branch"] == branch], key=lambda row: float(row["offset"]))
        points = []
        for row in selected:
            xx = float(row["offset"])
            yy = np.log10(max(float(row["final_full"]), 1.0e-16))
            px = x0 + int((xx - x_min) / (x_max - x_min) * (x1 - x0))
            py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
            points.append((px, py))
        color = colors.get(branch, (44, 160, 44))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
    for tol, label in ((ACCEPTANCE_TOL, "accept"), (ANCHOR_TOL, "anchor")):
        yy = np.log10(tol)
        py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
        draw.line((x0, py, x1, py), fill=(120, 120, 120), width=1)
        draw.text((x1 - 72, py - 14), label, fill=(80, 80, 80), font=font)
    draw.text((90, 25), f"Outer {MODE} homotopy: residual vs offset", fill=(20, 20, 20), font=font)
    draw.text((x0 + 4, y0 + 4), f"1e{int(y_max)}", fill=(80, 80, 80), font=font)
    draw.text((x0 + 4, y1 - 18), f"1e{int(y_min)}", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def run_branch(
    *,
    label: str,
    offsets: list[float],
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    base_value: float,
    fiducial: FiducialParams,
    mdot_edd: float,
    rows: list[dict[str, Any]],
) -> None:
    current_z = np.asarray(anchor_z, dtype=float)
    current_params = anchor_params
    for offset in offsets:
        target_value = base_value + float(offset)
        params = params_for(
            fiducial,
            mdot_edd,
            ratio=current_params.mdot_edd_ratio,
            R_out_rg=current_params.R_out_rg,
            n_nodes=current_params.n_nodes,
            grid_power=current_params.grid_power,
            custom_grid_xi=current_params.custom_grid_xi,
            mode=MODE,
            target_value=target_value,
            outer_omega_log_offset=current_params.outer_omega_log_offset,
        )
        params = apply_outer_slopes_from_state(current_z, params)
        seed = np.asarray(current_z, dtype=float)
        print(f"{label} offset={offset:g} initial={max_residual(seed, params):.3e}", flush=True)
        t0 = time.perf_counter()
        polish = polish_best(seed, params)
        final_params = apply_outer_slopes_from_state(polish.z, params)
        if REFRESH_REPOLISH:
            polish = polish_best(polish.z, final_params)
            final_params = apply_outer_slopes_from_state(polish.z, final_params)
        elapsed = time.perf_counter() - t0
        row = row_for_result(
            branch=label,
            offset=float(offset),
            base_value=base_value,
            target_value=target_value,
            seed=seed,
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
            f"target_res={row['target_residual']:.3e} accepted={row['accepted']} anchor={row['anchor_eligible']}",
            flush=True,
        )
        if row["accepted"]:
            current_z = np.asarray(polish.z, dtype=float)
            current_params = final_params
        else:
            print(f"  stopping branch {label} at first non-accepted offset", flush=True)
            break


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = load_anchor(ANCHOR_CHECKPOINT, fiducial, mdot_edd)
    base_value = target_base_value(anchor_z, anchor_params, MODE)
    rows: list[dict[str, Any]] = []
    for label, offsets in parse_branch_specs():
        run_branch(
            label=label,
            offsets=offsets,
            anchor_z=anchor_z,
            anchor_params=anchor_params,
            base_value=base_value,
            fiducial=fiducial,
            mdot_edd=mdot_edd,
            rows=rows,
        )
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
