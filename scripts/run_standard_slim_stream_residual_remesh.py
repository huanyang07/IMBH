"""Residual-based remeshing audit for stream-fed high-Mdot slim-disk checkpoints."""

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
    remap_profile_to_new_sonic_grid,
    solve_square_transonic_polish,
    stream_heating_rate,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    transonic_profile_from_state_vector,
    unpack_state,
    wind_sink_prime,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import fmt, json_safe
from run_standard_slim_stream_anchor_regression import (
    ROOT,
    advection_diagnostic,
    dominant,
    interval_peak_diagnostic,
    max_residual,
    params_from_checkpoint,
    refresh_outer_slopes_from_state,
    row_for_anchor,
    stream_diagnostic,
)


ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_REMESH_ANCHOR",
    "outputs/checkpoints/high_mdot_stream_source_bridge_m2_adaptive_df_strong_outer_0p808_to0p81/"
    "adaptive_mass_0p8086_torque_0p005_mdot_2_N640.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_REMESH_TABLE",
    "outputs/tables/standard_slim_stream_residual_remesh.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
PROFILE_OUTPUT = TABLE_OUTPUT.with_name(TABLE_OUTPUT.stem + "_profiles.json")
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_REMESH_CHECKPOINTS",
    "outputs/checkpoints/standard_slim_stream_residual_remesh",
)

N_NODES_LIST = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_N", "640").replace(":", ",").split(",")
    if piece.strip()
)
STRENGTHS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_STRENGTHS", "6,12,24").replace(":", ",").split(",")
    if piece.strip()
)
BLEND = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_BLEND", "0.75"))
POWER = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_POWER", "0.5"))
SMOOTH_PASSES = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_SMOOTH_PASSES", "2"))
MONITOR_FLOOR = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_MONITOR_FLOOR", "1.0"))
REFERENCE_GRID = os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_REFERENCE", "current").strip().lower()
DENSE_FACTOR = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_DENSE_FACTOR", "32"))

W_INTERVAL_E = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_W_INTERVAL_E", "1.0"))
W_SOURCE = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_W_SOURCE", "0.8"))
W_MDOT = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_W_MDOT", "0.8"))
W_QSTREAM = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_W_QSTREAM", "0.5"))
W_OUTER = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_W_OUTER", "1.0"))
OUTER_WIDTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_OUTER_WIDTH", "0.018"))

NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_NEWTON_MAX_ITER", "32"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_NEWTON_MAX_NFEV", "3600"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_NEWTON_MAX_STEP_NORM", "0.16"))
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_NEWTON_LINEAR_SOLVER", "regularized_lsmr")
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_ANCHOR_TOL", "3e-6"))
REFRESH_REPOLISH = os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_REFRESH_REPOLISH", "1") != "0"
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_REMESH_PIVOTS", "C2,C1").replace(":", ",").split(",")
    if piece.strip()
)


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


def normalize_component(values: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.abs(np.asarray(values, dtype=float)), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.max(clean))
    if scale <= 0.0:
        return np.zeros_like(clean)
    return clean / scale


def enforce_min_spacing(xi: np.ndarray, min_spacing: float = 1.0e-10) -> np.ndarray:
    adjusted = np.asarray(xi, dtype=float).copy()
    adjusted[0] = 0.0
    adjusted[-1] = 1.0
    for idx in range(1, adjusted.size):
        adjusted[idx] = max(adjusted[idx], adjusted[idx - 1] + min_spacing)
    if adjusted[-1] > 1.0:
        adjusted *= 1.0 / adjusted[-1]
    adjusted[-1] = 1.0
    for idx in range(adjusted.size - 2, -1, -1):
        adjusted[idx] = min(adjusted[idx], adjusted[idx + 1] - min_spacing)
    adjusted[0] = 0.0
    adjusted[-1] = 1.0
    if np.any(np.diff(adjusted) <= 0.0):
        raise RuntimeError("residual-remeshed grid spacing collapsed")
    return adjusted


def source_integrals(logR: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    source_prime = np.asarray([stream_source_prime(float(x), params) for x in logR], dtype=float)
    wind_prime = np.asarray([wind_sink_prime(float(x), params) for x in logR], dtype=float)
    signed_prime = wind_prime - source_prime
    mdot_inner, _ = stream_mass_rate_and_derivative(float(logR[0]), params)
    mdot_outer, _ = stream_mass_rate_and_derivative(float(logR[-1]), params)
    signed_integral = float(np.trapezoid(signed_prime, logR))
    budget_error = float((mdot_outer - mdot_inner) - signed_integral)
    return {
        "source_integral_over_inner": float(np.trapezoid(source_prime, logR) / params.Mdot_g_s),
        "wind_integral_over_inner": float(np.trapezoid(wind_prime, logR) / params.Mdot_g_s),
        "signed_integral_over_inner": float(signed_integral / params.Mdot_g_s),
        "budget_error_over_inner": float(budget_error / params.Mdot_g_s),
    }


def residual_remesh_grid_xi(
    source_z: np.ndarray,
    source_params: TransonicSlimParams,
    *,
    n_nodes: int,
    strength: float,
) -> tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]:
    _logu, _logT, logR_son, _lambda0, logR = unpack_state(source_z, source_params)
    logR_out = float(np.log(source_params.R_out))
    span = max(logR_out - float(logR_son), 1.0e-12)
    source_xi = (logR - float(logR_son)) / span
    dense_count = max(4096, int(DENSE_FACTOR) * max(int(n_nodes), int(source_params.n_nodes)))
    dense_xi = np.linspace(0.0, 1.0, dense_count)
    dense_logR = float(logR_son) + dense_xi * span

    intervals = interval_residuals(source_z, source_params)
    interval_mid_xi = 0.5 * (source_xi[:-1] + source_xi[1:])
    interval_E = normalize_component(intervals[:, 1])
    interval_dense = np.interp(dense_xi, interval_mid_xi, interval_E, left=interval_E[0], right=interval_E[-1])

    source_dense = np.asarray([stream_source_prime(float(x), source_params) for x in dense_logR], dtype=float)
    wind_dense = np.asarray([wind_sink_prime(float(x), source_params) for x in dense_logR], dtype=float)
    mdot_prime_dense = np.asarray(
        [stream_mass_rate_and_derivative(float(x), source_params)[1] for x in dense_logR],
        dtype=float,
    )
    qstream_dense = np.asarray([stream_heating_rate(float(x), source_params) for x in dense_logR], dtype=float)
    if dense_count >= 3:
        dqstream_dense = np.gradient(qstream_dense, dense_logR)
    else:
        dqstream_dense = np.zeros_like(qstream_dense)
    outer_width = max(float(OUTER_WIDTH), 1.0e-5)
    outer_dense = np.exp(-0.5 * ((dense_xi - 1.0) / outer_width) ** 2)

    composite = (
        W_INTERVAL_E * interval_dense
        + W_SOURCE * normalize_component(source_dense)
        + W_SOURCE * normalize_component(wind_dense)
        + W_MDOT * normalize_component(mdot_prime_dense)
        + W_QSTREAM * normalize_component(dqstream_dense)
        + W_OUTER * normalize_component(outer_dense)
    )
    composite = smooth_score(composite, SMOOTH_PASSES)
    monitor = MONITOR_FLOOR + float(strength) * normalize_component(composite) ** float(POWER)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (monitor[:-1] + monitor[1:]) * np.diff(dense_xi))])
    cumulative /= cumulative[-1]
    target = np.linspace(0.0, 1.0, int(n_nodes))
    adapted = np.interp(target, cumulative, dense_xi)

    if REFERENCE_GRID in {"current", "source"}:
        reference = np.interp(target, np.linspace(0.0, 1.0, source_xi.size), source_xi)
    elif REFERENCE_GRID in {"power", "baseline"}:
        reference = target ** float(source_params.grid_power)
    elif REFERENCE_GRID in {"uniform", "linear"}:
        reference = target
    else:
        raise ValueError(f"unknown residual-remesh reference grid {REFERENCE_GRID!r}")
    blended = enforce_min_spacing((1.0 - BLEND) * reference + BLEND * adapted)
    new_logR = float(logR_son) + blended * span

    old_integrals = source_integrals(logR, source_params)
    new_integrals = source_integrals(new_logR, source_params)
    peak_monitor = int(np.argmax(monitor))
    grid_info: dict[str, Any] = {
        "monitor_strength": float(strength),
        "monitor_power": float(POWER),
        "monitor_blend": float(BLEND),
        "monitor_floor": float(MONITOR_FLOOR),
        "monitor_reference": REFERENCE_GRID,
        "monitor_max": float(np.max(monitor)),
        "monitor_p90": float(np.percentile(monitor, 90.0)),
        "peak_monitor_xi": float(dense_xi[peak_monitor]),
        "peak_monitor_R_rg": float(np.exp(dense_logR[peak_monitor]) / source_params.r_g),
        "source_outer_dx": float(logR[-1] - logR[-2]),
        "target_outer_dx": float(new_logR[-1] - new_logR[-2]),
        "target_outer_dxi": float(blended[-1] - blended[-2]),
        "target_inner_dxi": float(blended[1] - blended[0]),
        "target_outer_1pct_nodes": int(np.count_nonzero(blended >= 0.99)),
        "target_outer_5pct_nodes": int(np.count_nonzero(blended >= 0.95)),
        "old_source_integral_over_inner": old_integrals["source_integral_over_inner"],
        "new_source_integral_over_inner": new_integrals["source_integral_over_inner"],
        "source_integral_delta_over_inner": float(
            new_integrals["source_integral_over_inner"] - old_integrals["source_integral_over_inner"]
        ),
        "old_budget_error_over_inner": old_integrals["budget_error_over_inner"],
        "new_budget_error_over_inner": new_integrals["budget_error_over_inner"],
    }
    profile_info = {
        "dense_xi": dense_xi,
        "dense_R_rg": np.exp(dense_logR) / source_params.r_g,
        "monitor": monitor,
        "component_interval_E": interval_dense,
        "component_source": normalize_component(source_dense),
        "component_wind": normalize_component(wind_dense),
        "component_mdot_prime": normalize_component(mdot_prime_dense),
        "component_qstream_prime": normalize_component(dqstream_dense),
        "component_outer": normalize_component(outer_dense),
        "old_xi": source_xi,
        "new_xi": blended,
    }
    return tuple(float(value) for value in blended), grid_info, profile_info


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


def row_for_result(
    *,
    label: str,
    source_z: np.ndarray,
    seed: np.ndarray,
    z: np.ndarray,
    params: TransonicSlimParams,
    polish,
    elapsed_s: float,
    grid_info: dict[str, Any],
) -> dict[str, Any]:
    base = row_for_anchor(label, ANCHOR_CHECKPOINT, z, params)
    base.update(
        {
            "initial_full": max_residual(seed, params),
            "source_full_on_target_params": max_residual(source_z, params) if source_z.shape == seed.shape else np.nan,
            "final_full": base.pop("full"),
            "accepted": bool(base["accepted"]),
            "anchor_eligible": bool(base["anchor_eligible"]),
            "pivot": str(polish.pivot),
            "method": str(polish.method),
            "nfev": int(polish.result.nfev),
            "iterations": int(polish.iterations),
            "elapsed_s": float(elapsed_s),
            "message": str(polish.result.message),
            **advection_diagnostic(z, params),
            **stream_diagnostic(z, params),
            **interval_peak_diagnostic(z, params),
            **grid_info,
            "z": np.asarray(z, dtype=float),
            "custom_grid_xi": np.asarray(params.custom_grid_xi, dtype=float)
            if params.custom_grid_xi is not None
            else np.asarray([], dtype=float),
        }
    )
    return base


def checkpoint_stem(row: dict[str, Any]) -> str:
    safe = str(row["label"]).replace(".", "p").replace("-", "m")
    mass = f"{float(row['source_fraction']):.6g}".replace(".", "p").replace("-", "m")
    torque = f"{float(row['torque_fraction']):.4g}".replace(".", "p").replace("-", "m")
    return f"{safe}_mass_{mass}_torque_{torque}_mdot_{float(row['ratio']):.8g}_N{int(row['N'])}".replace(".", "p")


def save_checkpoint(row: dict[str, Any], params: TransonicSlimParams) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    slopes = params.outer_match_log_slopes
    payload = {key: value for key, value in row.items() if key not in {"z", "custom_grid_xi"}}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{checkpoint_stem(row)}.npz",
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
        outer_robin_chi=np.array(params.outer_robin_chi),
        outer_robin_slope_target=np.array(params.outer_robin_slope_target),
        outer_robin_slope_scale=np.array(params.outer_robin_slope_scale),
        stream_torque_delta_l_fraction=np.array(params.stream_torque_delta_l_fraction),
        stream_torque_center_fraction=np.array(params.stream_torque_center_fraction),
        stream_torque_log_width=np.array(params.stream_torque_log_width),
        stream_source_fraction=np.array(params.stream_source_fraction),
        stream_source_center_fraction=np.array(params.stream_source_center_fraction),
        stream_source_log_width=np.array(params.stream_source_log_width),
        stream_source_shape=np.array(params.stream_source_shape),
        stream_source_shape_blend=np.array(params.stream_source_shape_blend),
        wind_sink_fraction=np.array(params.wind_sink_fraction),
        wind_sink_center_fraction=np.array(params.wind_sink_center_fraction),
        wind_sink_log_width=np.array(params.wind_sink_log_width),
        stream_heating_efficiency=np.array(params.stream_heating_efficiency),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted"]),
        row_json=np.array(json.dumps(json_safe(payload), sort_keys=True)),
    )


def serializable_profile(label: str, profile_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        **{
            key: np.asarray(value, dtype=float).tolist()
            for key, value in profile_info.items()
            if key
            in {
                "dense_xi",
                "dense_R_rg",
                "monitor",
                "component_interval_E",
                "component_source",
                "component_wind",
                "component_mdot_prime",
                "component_qstream_prime",
                "component_outer",
                "old_xi",
                "new_xi",
            }
        },
    }


def write_outputs(rows: list[dict[str, Any]], profiles: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Stream Residual Remesh",
        "",
        "Generated by `scripts/run_standard_slim_stream_residual_remesh.py`.",
        "",
        f"Anchor `{ANCHOR_CHECKPOINT.relative_to(ROOT)}`, strengths `{','.join(f'{value:g}' for value in STRENGTHS)}`, "
        f"N `{','.join(str(value) for value in N_NODES_LIST)}`, blend `{BLEND:g}`, power `{POWER:g}`, "
        f"reference `{REFERENCE_GRID}`.",
        "",
        "| label | N | strength | source frac | torque frac | initial full | final full | accepted | strict | dominant | int E | outer omega | peak E R/rg | old source int | new source int | source int delta | outer dx | outer 1% nodes | outer 5% nodes | f_adv global | f_adv inner | Lrad/LEdd | max H/R | Rson/rg | pivot | nfev | elapsed s |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        formatted = {
            key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value
            for key, value in row.items()
        }
        for key in ("source_fraction", "torque_fraction"):
            formatted[key] = f"{float(row[key]):.6g}"
        lines.append(
            "| {label} | {N} | {monitor_strength} | {source_fraction} | {torque_fraction} | "
            "{initial_full} | {final_full} | {accepted} | {anchor_eligible} | {dominant} | "
            "{interval_E} | {outer_omega} | {peak_interval_E_rg} | {old_source_integral_over_inner} | "
            "{new_source_integral_over_inner} | {source_integral_delta_over_inner} | {target_outer_dx} | "
            "{target_outer_1pct_nodes} | {target_outer_5pct_nodes} | {f_adv_global} | {f_adv_inner} | "
            "{Lrad_LEdd} | {max_H_R} | {Rson_rg} | {pivot} | {nfev} | {elapsed_s} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(
        json.dumps(json_safe([{key: value for key, value in row.items() if key not in {"z", "custom_grid_xi"}} for row in rows]), indent=2, sort_keys=True)
        + "\n"
    )
    PROFILE_OUTPUT.write_text(json.dumps(json_safe(profiles), indent=2, sort_keys=True) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_z, source_params = params_from_checkpoint(ANCHOR_CHECKPOINT, fiducial, mdot_edd)
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for n_nodes in N_NODES_LIST:
        for strength in STRENGTHS:
            custom_grid_xi, grid_info, profile_info = residual_remesh_grid_xi(
                source_z,
                source_params,
                n_nodes=int(n_nodes),
                strength=float(strength),
            )
            base_params = replace(
                source_params,
                n_nodes=int(n_nodes),
                custom_grid_xi=custom_grid_xi,
                max_nfev=NEWTON_MAX_NFEV,
                residual_tol=1.0e-8,
            )
            seed = remap_seed(source_z, source_params, base_params)
            target_params = refresh_outer_slopes_from_state(seed, base_params)
            label = f"N{int(n_nodes)}_s{float(strength):g}".replace(".", "p")
            print(
                f"{label} initial={max_residual(seed, target_params):.3e} "
                f"outer1={grid_info['target_outer_1pct_nodes']} outer5={grid_info['target_outer_5pct_nodes']} "
                f"source_delta={grid_info['source_integral_delta_over_inner']:.3e}",
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
                source_z=source_z,
                seed=seed,
                z=polish.z,
                params=final_params,
                polish=polish,
                elapsed_s=elapsed,
                grid_info=grid_info,
            )
            rows.append(row)
            profiles.append(serializable_profile(label, profile_info))
            save_checkpoint(row, final_params)
            write_outputs(rows, profiles)
            print(
                f"  final={row['final_full']:.3e} dom={row['dominant']} "
                f"anchor={row['anchor_eligible']} peakE={row['peak_interval_E_rg']:.4g} nfev={row['nfev']}",
                flush=True,
            )
    write_outputs(rows, profiles)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {PROFILE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
