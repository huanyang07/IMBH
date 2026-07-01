"""Small stream-heating geometry scan for the standard slim disk."""

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
    stream_heating_rate,
    stream_mass_rate_and_derivative,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_ANCHOR",
    "outputs/checkpoints/slim_benchmark_stream_mass_plus_torque_annulus_mdot1_rout300/load_mass_0p03_torque_0p01_mdot_1_N640.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_TABLE",
    "outputs/tables/slim_benchmark_stream_heating_geometry_scan_mdot1_rout300.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_FIGURE",
    "outputs/figures/slim_benchmark_stream_heating_geometry_scan_mdot1_rout300.png",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_stream_heating_geometry_scan_mdot1_rout300",
)

CENTER_FRACTIONS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_CENTERS", "0.8,0.6,0.4").replace(":", ",").split(",")
    if piece.strip()
)
LOG_WIDTHS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_WIDTHS", "0.08,0.30").replace(":", ",").split(",")
    if piece.strip()
)
HEATING_EFFICIENCY = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_ETA", "10.0"))
STREAM_MASS_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_MASS_FRACTION", "0.03"))
STREAM_TORQUE_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_TORQUE_FRACTION", "0.01"))
NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_NEWTON_MAX_ITER", "40"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_NEWTON_MAX_NFEV", "4800"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_NEWTON_MAX_STEP_NORM", "0.14"))
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_NEWTON_LINEAR_SOLVER", "regularized_lsmr")
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_ANCHOR_TOL", "3e-6"))
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_HEATING_GEOMETRY_PIVOTS", "C2,C1").replace(":", ",").split(",")
    if piece.strip()
)


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


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    *,
    ratio: float,
    R_out_rg: float,
    n_nodes: int,
    grid_power: float,
    custom_grid_xi: tuple[float, ...] | None,
    center_fraction: float,
    log_width: float,
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
        stream_torque_delta_l_fraction=STREAM_TORQUE_FRACTION,
        stream_torque_center_fraction=float(center_fraction),
        stream_torque_log_width=float(log_width),
        stream_source_fraction=STREAM_MASS_FRACTION,
        stream_source_center_fraction=float(center_fraction),
        stream_source_log_width=float(log_width),
        stream_heating_efficiency=HEATING_EFFICIENCY,
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def load_anchor(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["z"], dtype=float), {
        "ratio": float(data["ratio"]),
        "R_out_rg": float(data["R_out_rg"]),
        "n_nodes": int(data["n_nodes"]),
        "grid_power": float(data["grid_power"]) if "grid_power" in data else 1.0,
        "custom_grid_xi": custom_grid_from_data(data),
    }


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


def source_diagnostics(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    profile = transonic_profile_from_state_vector(z, params)
    _logu, _logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    q_stream = np.asarray([stream_heating_rate(float(x), params) for x in logR], dtype=float)
    qv = np.asarray(profile.Q_visc, dtype=float)
    qr = np.asarray(profile.Q_rad, dtype=float)
    weights = 2.0 * np.pi * profile.R**2
    int_stream = float(np.trapezoid(q_stream * weights, logR))
    int_visc = float(np.trapezoid(np.abs(qv) * weights, logR) + 1.0e-300)
    peak = int(np.argmax(q_stream))
    mdot_outer, _ = stream_mass_rate_and_derivative(float(logR[-1]), params)
    return {
        "Rinj_rg": float(params.stream_source_center_fraction * params.R_out_rg),
        "Mdot_outer_over_inner": float(mdot_outer / params.Mdot_g_s),
        "max_Qstream_Qvisc": float(np.max(q_stream / (np.abs(qv) + 1.0e-300))),
        "max_Qstream_Qrad": float(np.max(q_stream / (np.abs(qr) + 1.0e-300))),
        "integrated_Qstream_Qvisc": float(int_stream / int_visc),
        "peak_Qstream_R_rg": float(profile.R[peak] / params.r_g),
        "max_H_R": float(np.max(profile.H_over_R)),
        "integrated_adv": float(profile.integrated_advective_fraction),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0": float(profile.lambda0),
    }


def row_for_result(
    *,
    center_fraction: float,
    log_width: float,
    seed: np.ndarray,
    z: np.ndarray,
    params: TransonicSlimParams,
    polish,
    elapsed_s: float,
) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(z, params)
    full = max_residual(z, params)
    return {
        "center_fraction": float(center_fraction),
        "log_width": float(log_width),
        "eta_heat": HEATING_EFFICIENCY,
        "mass_fraction": STREAM_MASS_FRACTION,
        "torque_fraction": STREAM_TORQUE_FRACTION,
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
        **source_diagnostics(z, params),
        "pivot": str(polish.pivot),
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
    stem = (
        f"center_{float(row['center_fraction']):.3g}_width_{float(row['log_width']):.3g}_eta_{HEATING_EFFICIENCY:.3g}"
        f"_mass_{STREAM_MASS_FRACTION:.3g}_torque_{STREAM_TORQUE_FRACTION:.3g}_N{int(row['N'])}"
    ).replace(".", "p").replace("-", "m")
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
        stream_source_fraction=np.array(params.stream_source_fraction),
        stream_source_center_fraction=np.array(params.stream_source_center_fraction),
        stream_source_log_width=np.array(params.stream_source_log_width),
        stream_mass_fraction=np.array(params.stream_mass_fraction),
        stream_mass_center_fraction=np.array(params.stream_mass_center_fraction),
        stream_mass_log_width=np.array(params.stream_mass_log_width),
        stream_heating_efficiency=np.array(params.stream_heating_efficiency),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted"]),
        row_json=np.array(json.dumps(json_safe(payload), sort_keys=True)),
    )


def write_table(rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Stream-Heating Geometry Scan",
        "",
        "Generated by `scripts/run_standard_slim_stream_heating_geometry_scan.py`.",
        "",
        f"Anchor `{ANCHOR_CHECKPOINT.relative_to(ROOT)}`, centers `{','.join(f'{value:g}' for value in CENTER_FRACTIONS)}`, "
        f"widths `{','.join(f'{value:g}' for value in LOG_WIDTHS)}`, eta `{HEATING_EFFICIENCY:g}`, "
        f"mass fraction `{STREAM_MASS_FRACTION:g}`, torque fraction `{STREAM_TORQUE_FRACTION:g}`.",
        "",
        "| Rinj/Rout | log width | final full | accepted | anchor | dominant | outer omega | int R | int E | max Qs/Qv | int Qs/Qv | peak R/rg | max H/R | int adv | Rson/rg | pivot | nfev | elapsed s | message |",
        "|---:|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {center_fraction} | {log_width} | {final_full} | {accepted} | {anchor_eligible} | {dominant} | {outer_omega} | "
            "{interval_R} | {interval_E} | {max_Qstream_Qvisc} | {integrated_Qstream_Qvisc} | {peak_Qstream_R_rg} | "
            "{max_H_R} | {integrated_adv} | {Rson_rg} | {pivot} | {nfev} | {elapsed_s} | {message} |".format(**formatted).replace("\n", " ")
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
    image = Image.new("RGB", (1000, 650), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = 90, 80, 930, 560
    draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
    centers = sorted(set(float(row["center_fraction"]) for row in rows))
    widths = sorted(set(float(row["log_width"]) for row in rows))
    residuals = np.asarray([float(row["final_full"]) for row in rows])
    qints = np.asarray([float(row["integrated_Qstream_Qvisc"]) for row in rows])
    y_values = np.log10(np.maximum(residuals, 1.0e-16))
    y_min, y_max = float(np.floor(np.min(y_values))), float(np.ceil(np.max(y_values)))
    if y_max <= y_min:
        y_max = y_min + 1.0
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    for i, width in enumerate(widths):
        selected = sorted([row for row in rows if np.isclose(float(row["log_width"]), width)], key=lambda row: float(row["center_fraction"]))
        points = []
        for row in selected:
            xx = centers.index(float(row["center_fraction"])) / max(1, len(centers) - 1)
            yy = np.log10(max(float(row["final_full"]), 1.0e-16))
            px = x0 + int(xx * (x1 - x0))
            py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
            radius = max(4, int(40.0 * float(row["integrated_Qstream_Qvisc"]) / (float(np.max(qints)) + 1.0e-300)))
            color = colors[i % len(colors)]
            points.append((px, py))
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
        if len(points) >= 2:
            draw.line(points, fill=colors[i % len(colors)], width=2)
        draw.text((x1 - 110, y0 + 18 + 14 * i), f"width {width:g}", fill=colors[i % len(colors)], font=font)
    for idx, center in enumerate(centers):
        px = x0 + int(idx / max(1, len(centers) - 1) * (x1 - x0))
        draw.text((px - 18, y1 + 10), f"{center:g}", fill=(80, 80, 80), font=font)
    for tol, label in ((ACCEPTANCE_TOL, "accept"), (ANCHOR_TOL, "anchor")):
        yy = np.log10(tol)
        py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
        draw.line((x0, py, x1, py), fill=(120, 120, 120), width=1)
        draw.text((x1 - 72, py - 14), label, fill=(80, 80, 80), font=font)
    draw.text((90, 25), "Geometry scan: residual vs Rinj/Rout; marker size = integrated Qstream/Qvisc", fill=(20, 20, 20), font=font)
    draw.text((450, 610), "Rinj/Rout", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, base = load_anchor(ANCHOR_CHECKPOINT, fiducial, mdot_edd)
    rows: list[dict[str, Any]] = []
    for width in LOG_WIDTHS:
        for center in CENTER_FRACTIONS:
            params = params_for(fiducial, mdot_edd, center_fraction=center, log_width=width, **base)
            params = replace(params, outer_match_log_slopes=one_sided_outer_slopes(anchor_z, params))
            seed = np.asarray(anchor_z, dtype=float)
            print(f"center={center:g} width={width:g} initial={max_residual(seed, params):.3e}", flush=True)
            t0 = time.perf_counter()
            polish = polish_best(seed, params)
            final_params = replace(params, outer_match_log_slopes=one_sided_outer_slopes(polish.z, params))
            elapsed = time.perf_counter() - t0
            row = row_for_result(center_fraction=center, log_width=width, seed=seed, z=polish.z, params=final_params, polish=polish, elapsed_s=elapsed)
            rows.append(row)
            save_checkpoint(row, final_params)
            write_table(rows)
            write_figure(rows)
            print(
                f"  final={row['final_full']:.3e} dom={row['dominant']} "
                f"int_Qs/Qv={row['integrated_Qstream_Qvisc']:.3e} accepted={row['accepted']} anchor={row['anchor_eligible']}",
                flush=True,
            )
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
