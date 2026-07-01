"""Torque-only stream annulus scan for the standard slim disk.

This replaces a hard outer angular-momentum Dirichlet offset with a smooth
cumulative angular-momentum torque deposited near the outer disk.
"""

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
from imri_qpe.layer3_minidisk_1d.transonic_local import stream_torque_specific_l_and_derivative
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_TORQUE_ANCHOR",
    "outputs/checkpoints/slim_benchmark_physical_rout_homotopy_mdot1_1000_300/Rout_300_mdot_1_N640.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_TORQUE_TABLE",
    "outputs/tables/slim_benchmark_stream_torque_annulus_mdot1_rout300.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_TORQUE_FIGURE",
    "outputs/figures/slim_benchmark_stream_torque_annulus_mdot1_rout300.png",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_TORQUE_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_stream_torque_annulus_mdot1_rout300",
)

BRANCH_SPECS = tuple(
    piece.strip()
    for piece in os.environ.get(
        "IMBH_STANDARD_SLIM_STREAM_TORQUE_BRANCHES",
        "subK:0,-1e-5,-3e-5,-1e-4,-3e-4,-1e-3;superK:0,1e-5,3e-5,1e-4,3e-4,1e-3",
    ).split(";")
    if piece.strip()
)
CENTER_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TORQUE_CENTER_FRACTION", "0.8"))
LOG_WIDTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TORQUE_LOG_WIDTH", "0.08"))
NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TORQUE_NEWTON_MAX_ITER", "28"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TORQUE_NEWTON_MAX_NFEV", "2600"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TORQUE_NEWTON_MAX_STEP_NORM", "0.16"))
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_STREAM_TORQUE_NEWTON_LINEAR_SOLVER", "regularized_lsmr")
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TORQUE_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TORQUE_ANCHOR_TOL", "3e-6"))
REFRESH_REPOLISH = os.environ.get("IMBH_STANDARD_SLIM_STREAM_TORQUE_REFRESH_REPOLISH", "0") != "0"
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_TORQUE_PIVOTS", "C2,C1").replace(":", ",").split(",")
    if piece.strip()
)


def parse_branch_specs() -> list[tuple[str, list[float]]]:
    branches: list[tuple[str, list[float]]] = []
    for spec in BRANCH_SPECS:
        if ":" not in spec:
            raise ValueError(f"branch spec must be label:amplitudes, got {spec!r}")
        label, values = spec.split(":", 1)
        amplitudes = [float(piece) for piece in values.replace(",", ":").split(":") if piece.strip()]
        if not amplitudes:
            raise ValueError(f"branch {label!r} has no amplitudes")
        branches.append((label.strip(), amplitudes))
    return branches


def custom_grid_from_data(data) -> tuple[float, ...] | None:
    if "custom_grid_xi" not in data:
        return None
    candidate = np.asarray(data["custom_grid_xi"], dtype=float)
    if candidate.shape == (int(data["n_nodes"]),):
        return tuple(float(value) for value in candidate)
    return None


def one_sided_outer_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    dx = float(logR[-1] - logR[-2])
    return float((logu[-1] - logu[-2]) / dx), float((logT[-1] - logT[-2]) / dx)


def apply_outer_slopes_from_state(z: np.ndarray, params: TransonicSlimParams) -> TransonicSlimParams:
    return replace(params, outer_match_log_slopes=one_sided_outer_slopes(z, params))


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    *,
    ratio: float,
    R_out_rg: float,
    n_nodes: int,
    grid_power: float,
    custom_grid_xi: tuple[float, ...] | None,
    torque_fraction: float,
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
        outer_omega_log_offset=0.0,
        stream_torque_delta_l_fraction=float(torque_fraction),
        stream_torque_center_fraction=CENTER_FRACTION,
        stream_torque_log_width=LOG_WIDTH,
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def load_anchor(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    data = np.load(path, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    params = params_for(
        fiducial,
        mdot_edd,
        ratio=float(data["ratio"]),
        R_out_rg=float(data["R_out_rg"]),
        n_nodes=int(data["n_nodes"]),
        grid_power=float(data["grid_power"]) if "grid_power" in data else 1.0,
        custom_grid_xi=custom_grid_from_data(data),
        torque_fraction=0.0,
    )
    return z, apply_outer_slopes_from_state(z, params)


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


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


def angular_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    slopes = params.outer_match_log_slopes
    if slopes is None:
        return {
            "pressure_target": np.nan,
            "achieved_omega_log_offset": np.nan,
            "omega_target_residual": np.nan,
        }
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    ln_omega = float(np.log(profile.Omega[-1] / profile.Omega_K[-1]))
    pressure_target = pressure_supported_omega_target(
        float(logR[-1]),
        np.array([logu[-1], logT[-1]], dtype=float),
        np.asarray(slopes, dtype=float),
        lambda0,
        params,
    )
    return {
        "pressure_target": float(pressure_target),
        "achieved_omega_log_offset": float(ln_omega - pressure_target),
        "omega_target_residual": float(ln_omega - pressure_target),
    }


def stream_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    _logu, _logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    R_center = float(params.stream_torque_center_fraction * params.R_out)
    l_ref = float(params.potential.l_k(R_center))
    stream_outer, stream_outer_deriv = stream_torque_specific_l_and_derivative(float(logR[-1]), params)
    stream_center, stream_center_deriv = stream_torque_specific_l_and_derivative(float(np.log(R_center)), params)
    return {
        "Rinj_rg": float(R_center / params.r_g),
        "stream_l_outer_over_lKinj": float(stream_outer / l_ref),
        "stream_l_center_over_lKinj": float(stream_center / l_ref),
        "dstream_l_dlnR_outer_over_lKinj": float(stream_outer_deriv / l_ref),
        "dstream_l_dlnR_center_over_lKinj": float(stream_center_deriv / l_ref),
    }


def row_for_result(
    *,
    branch: str,
    torque_fraction: float,
    seed: np.ndarray,
    z: np.ndarray,
    params: TransonicSlimParams,
    polish,
    elapsed_s: float,
) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    full = max_residual(z, params)
    return {
        "branch": branch,
        "torque_fraction": float(torque_fraction),
        "ratio": float(params.mdot_edd_ratio),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "grid_power": float(params.grid_power),
        "center_fraction": float(params.stream_torque_center_fraction),
        "log_width": float(params.stream_torque_log_width),
        "initial_full": max_residual(seed, params),
        "final_full": full,
        "accepted": bool(full <= ACCEPTANCE_TOL),
        "anchor_eligible": bool(full <= ANCHOR_TOL),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        **angular_diagnostic(z, params),
        **stream_diagnostic(z, params),
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
    safe_torque = f"{float(row['torque_fraction']):.4g}".replace(".", "p").replace("-", "m")
    stem = f"{safe_branch}_torque_{safe_torque}_mdot_{float(row['ratio']):.8g}_N{int(row['N'])}".replace(".", "p")
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
        stream_torque_delta_l_fraction=np.array(params.stream_torque_delta_l_fraction),
        stream_torque_center_fraction=np.array(params.stream_torque_center_fraction),
        stream_torque_log_width=np.array(params.stream_torque_log_width),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted"]),
        branch=np.array(row["branch"]),
        row_json=np.array(json.dumps(json_safe(payload), sort_keys=True)),
    )


def write_table(rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Stream-Torque Annulus Scan",
        "",
        "Generated by `scripts/run_standard_slim_stream_torque_annulus_scan.py`.",
        "",
        f"Anchor `{ANCHOR_CHECKPOINT.relative_to(ROOT)}`, branches `{';'.join(BRANCH_SPECS)}`, "
        f"Rinj/Rout `{CENTER_FRACTION:g}`, log width `{LOG_WIDTH:g}`, refresh repolish `{REFRESH_REPOLISH}`.",
        "",
        "| branch | torque Delta l/lKinj | Mdot/Edd | Rout/rg | Rinj/rg | initial full | final full | accepted | anchor | dominant | outer omega | achieved offset | int R | int E | outer energy | stream outer | max H/R | int adv | Rson/rg | pivot | nfev | elapsed s | message |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {branch} | {torque_fraction} | {ratio} | {R_out_rg} | {Rinj_rg} | {initial_full} | {final_full} | "
            "{accepted} | {anchor_eligible} | {dominant} | {outer_omega} | {achieved_omega_log_offset} | "
            "{interval_R} | {interval_E} | {outer_energy} | {stream_l_outer_over_lKinj} | {max_H_R} | "
            "{integrated_adv} | {Rson_rg} | {pivot} | {nfev} | {elapsed_s} | {message} |".format(**formatted).replace("\n", " ")
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
    amplitudes = np.asarray([float(row["torque_fraction"]) for row in rows], dtype=float)
    residuals = np.log10(np.maximum(np.asarray([float(row["final_full"]) for row in rows], dtype=float), 1.0e-16))
    x_min, x_max = float(np.min(amplitudes)), float(np.max(amplitudes))
    if x_max <= x_min:
        x_min -= 0.5
        x_max += 0.5
    y_min, y_max = float(np.floor(np.min(residuals))), float(np.ceil(np.max(residuals)))
    if y_max <= y_min:
        y_max = y_min + 1.0
    colors = {"subK": (31, 119, 180), "superK": (214, 39, 40)}
    for branch in sorted(set(str(row["branch"]) for row in rows)):
        selected = sorted([row for row in rows if row["branch"] == branch], key=lambda row: float(row["torque_fraction"]))
        points = []
        for row in selected:
            xx = float(row["torque_fraction"])
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
    draw.text((90, 25), "Stream torque annulus: residual vs Delta l/lK(Rinj)", fill=(20, 20, 20), font=font)
    draw.text((x0 + 4, y0 + 4), f"1e{int(y_max)}", fill=(80, 80, 80), font=font)
    draw.text((x0 + 4, y1 - 18), f"1e{int(y_min)}", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def run_branch(
    *,
    label: str,
    amplitudes: list[float],
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    fiducial: FiducialParams,
    mdot_edd: float,
    rows: list[dict[str, Any]],
) -> None:
    current_z = np.asarray(anchor_z, dtype=float)
    current_params = anchor_params
    for torque_fraction in amplitudes:
        params = params_for(
            fiducial,
            mdot_edd,
            ratio=current_params.mdot_edd_ratio,
            R_out_rg=current_params.R_out_rg,
            n_nodes=current_params.n_nodes,
            grid_power=current_params.grid_power,
            custom_grid_xi=current_params.custom_grid_xi,
            torque_fraction=float(torque_fraction),
        )
        params = apply_outer_slopes_from_state(current_z, params)
        seed = np.asarray(current_z, dtype=float)
        print(f"{label} torque={torque_fraction:g} initial={max_residual(seed, params):.3e}", flush=True)
        t0 = time.perf_counter()
        polish = polish_best(seed, params)
        final_params = apply_outer_slopes_from_state(polish.z, params)
        if REFRESH_REPOLISH:
            polish = polish_best(polish.z, final_params)
            final_params = apply_outer_slopes_from_state(polish.z, final_params)
        elapsed = time.perf_counter() - t0
        row = row_for_result(
            branch=label,
            torque_fraction=float(torque_fraction),
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
            f"outer_omega={row['outer_omega']:.3e} accepted={row['accepted']} anchor={row['anchor_eligible']}",
            flush=True,
        )
        if row["accepted"]:
            current_z = np.asarray(polish.z, dtype=float)
            current_params = final_params
        else:
            print(f"  stopping branch {label} at first non-accepted torque", flush=True)
            break


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = load_anchor(ANCHOR_CHECKPOINT, fiducial, mdot_edd)
    rows: list[dict[str, Any]] = []
    for label, amplitudes in parse_branch_specs():
        run_branch(
            label=label,
            amplitudes=amplitudes,
            anchor_z=anchor_z,
            anchor_params=anchor_params,
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
