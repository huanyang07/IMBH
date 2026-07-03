"""Fixed-physics grid homotopy for near-wall stream-fed slim-disk checkpoints."""

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
    residual_audit_from_state_vector,
    residual_partition_audit_from_state_vector,
    solve_square_transonic_polish,
    transonic_profile_from_state_vector,
    unpack_state,
)
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
from run_standard_slim_stream_residual_remesh import residual_remesh_grid_xi


ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_ANCHOR",
    "outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_target_remesh_0898_to0898125/"
    "phys_gate_target_remesh_0898_to0898125_mass_0p8980625_torque_0p005_mdot_2_N896.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_TABLE",
    "outputs/tables/high_mdot_stream_grid_homotopy.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_CHECKPOINTS",
    "outputs/checkpoints/high_mdot_stream_grid_homotopy",
)

TARGET_ETA = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_TARGET_ETA", "1.0"))
INITIAL_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_INITIAL_STEP", "0.05"))
MIN_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_MIN_STEP", "0.00625"))
MAX_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_MAX_STEP", "0.1"))
GROWTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_GROWTH", "1.25"))
SHRINK = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_SHRINK", "0.5"))
MAX_INITIAL_FULL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_MAX_INITIAL_FULL", "5e-4"))
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_ANCHOR_TOL", "3e-6"))
PHYSICAL_E_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_PHYSICAL_E_TOL", "3e-5"))

TARGET_STRENGTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_TARGET_STRENGTH", "8"))
REMAP_METHOD = os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_REMAP_METHOD", "linear").strip().lower()

NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_NEWTON_MAX_ITER", "8"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_NEWTON_MAX_NFEV", "280"))
NEWTON_RESIDUAL_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_NEWTON_RESIDUAL_TOL", "1e-8"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_NEWTON_MAX_STEP_NORM", "0.16"))
NEWTON_JACOBIAN_REL_STEP = float(
    os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_NEWTON_JACOBIAN_REL_STEP", "3e-5")
)
NEWTON_LINEAR_SOLVER = os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_NEWTON_LINEAR_SOLVER",
    "regularized_lsmr",
)
NEWTON_LINEAR_DAMPINGS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_NEWTON_LINEAR_DAMPINGS", "1e-2,1e-1,1")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_GRID_HOMOTOPY_PIVOTS", "C2").replace(":", ",").split(",")
    if piece.strip()
)


def current_grid_xi(params: TransonicSlimParams) -> np.ndarray:
    if params.custom_grid_xi is not None:
        return np.asarray(params.custom_grid_xi, dtype=float)
    return np.linspace(0.0, 1.0, int(params.n_nodes)) ** float(params.grid_power)


def enforce_min_spacing(xi: np.ndarray, min_spacing: float = 1.0e-10) -> np.ndarray:
    adjusted = np.asarray(xi, dtype=float).copy()
    adjusted[0] = 0.0
    adjusted[-1] = 1.0
    for idx in range(1, adjusted.size):
        adjusted[idx] = max(adjusted[idx], adjusted[idx - 1] + min_spacing)
    if adjusted[-1] > 1.0:
        adjusted /= adjusted[-1]
    adjusted[-1] = 1.0
    for idx in range(adjusted.size - 2, -1, -1):
        adjusted[idx] = min(adjusted[idx], adjusted[idx + 1] - min_spacing)
    adjusted[0] = 0.0
    adjusted[-1] = 1.0
    if np.any(np.diff(adjusted) <= 0.0):
        raise RuntimeError("grid homotopy spacing collapsed")
    return adjusted


def blended_grid(source_xi: np.ndarray, target_xi: np.ndarray, eta: float) -> tuple[float, ...]:
    eta = min(max(float(eta), 0.0), 1.0)
    xi = (1.0 - eta) * np.asarray(source_xi, dtype=float) + eta * np.asarray(target_xi, dtype=float)
    return tuple(float(value) for value in enforce_min_spacing(xi))


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
            residual_tol=NEWTON_RESIDUAL_TOL,
            use_block_jacobian=True,
            jacobian_rel_step=NEWTON_JACOBIAN_REL_STEP,
            linear_solver=NEWTON_LINEAR_SOLVER,
            linear_dampings=NEWTON_LINEAR_DAMPINGS,
            max_step_norm=NEWTON_MAX_STEP_NORM,
        )
        full = max_residual(result.z, params)
        if full < best_full:
            best = result
            best_full = full
        if full <= ANCHOR_TOL:
            break
    if best is None:
        raise RuntimeError("no grid-homotopy pivots configured")
    return best


def physical_energy(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(residual_partition_audit_from_state_vector(z, params).physical_energy_max)


def row_for_state(
    *,
    label: str,
    eta: float,
    step: float,
    seed: np.ndarray,
    z: np.ndarray,
    params: TransonicSlimParams,
    polish,
    elapsed_s: float,
    target_grid_info: dict[str, Any],
    action: str,
) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(z, params)
    partition = residual_partition_audit_from_state_vector(z, params)
    full = max_residual(z, params)
    base = row_for_anchor(label, ANCHOR_CHECKPOINT, z, params)
    source_xi = current_grid_xi(params)
    base.update(
        {
            "eta": float(eta),
            "attempt_step": float(step),
            "initial_full": max_residual(seed, params),
            "final_full": float(full),
            "accepted": bool(full <= ACCEPTANCE_TOL and partition.physical_energy_max <= PHYSICAL_E_TOL),
            "solver_accepted": bool(full <= ACCEPTANCE_TOL),
            "anchor_eligible": bool(full <= ANCHOR_TOL and partition.physical_energy_max <= PHYSICAL_E_TOL),
            "physical_E_gate_eligible": bool(partition.physical_energy_max <= PHYSICAL_E_TOL),
            "physical_E_tol": float(PHYSICAL_E_TOL),
            "dominant": dominant(audit),
            "interval_E": float(audit.interval_energy_max),
            "interval_R": float(audit.interval_radial_max),
            "outer_omega": float(audit.outer_omega),
            "partition_physical_E": float(partition.physical_energy_max),
            "partition_peak_physical_E_rg": float(partition.peak_physical_energy_rg),
            "partition_buffer_E": float(partition.buffer_energy_max),
            "partition_peak_buffer_E_rg": float(partition.peak_buffer_energy_rg),
            "grid_dx_min": float(np.min(np.diff(source_xi))),
            "grid_dx_max": float(np.max(np.diff(source_xi))),
            "grid_nodes_250_270": int(np.count_nonzero((node_radii_rg(z, params) >= 250.0) & (node_radii_rg(z, params) <= 270.0))),
            "pivot": str(polish.pivot),
            "method": str(polish.method),
            "nfev": int(polish.result.nfev),
            "iterations": int(polish.iterations),
            "elapsed_s": float(elapsed_s),
            "message": str(polish.result.message),
            "action": str(action),
            **stream_diagnostic(z, params),
            **advection_diagnostic(z, params),
            **interval_peak_diagnostic(z, params),
            **{f"target_{key}": value for key, value in target_grid_info.items()},
            "z": np.asarray(z, dtype=float),
            "custom_grid_xi": np.asarray(params.custom_grid_xi, dtype=float)
            if params.custom_grid_xi is not None
            else np.asarray([], dtype=float),
        }
    )
    return base


def node_radii_rg(z: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    _logu, _logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    return np.exp(logR) / params.r_g


def checkpoint_stem(row: dict[str, Any]) -> str:
    safe = str(row["label"]).replace(".", "p").replace("-", "m")
    eta = f"{float(row['eta']):.9g}".replace(".", "p").replace("-", "m")
    source = f"{float(row['source_fraction']):.9g}".replace(".", "p").replace("-", "m")
    return f"{safe}_eta_{eta}_mass_{source}_N{int(row['N'])}".replace(".", "p")


def save_checkpoint(row: dict[str, Any], params: TransonicSlimParams) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in row.items() if key not in {"z", "custom_grid_xi"}}
    slopes = params.outer_match_log_slopes
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
        outer_buffer_inner_rg=np.array(np.nan if params.outer_buffer_inner_rg is None else params.outer_buffer_inner_rg),
        outer_buffer_radial_weight=np.array(params.outer_buffer_radial_weight),
        outer_buffer_energy_weight=np.array(params.outer_buffer_energy_weight),
        outer_buffer_boundary_weight=np.array(params.outer_buffer_boundary_weight),
        outer_buffer_taper_log_width=np.array(params.outer_buffer_taper_log_width),
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
        interval_residual_form=np.array(params.interval_residual_form),
        integrated_residual_weighting=np.array(params.integrated_residual_weighting),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted"]),
        row_json=np.array(json.dumps(json_safe(payload), sort_keys=True)),
    )


def write_outputs(rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Stream Grid Homotopy",
        "",
        "Generated by `scripts/run_standard_slim_stream_grid_homotopy.py`.",
        "",
        f"Anchor `{ANCHOR_CHECKPOINT.relative_to(ROOT)}`, target eta `{TARGET_ETA:g}`, "
        f"initial step `{INITIAL_STEP:g}`, min step `{MIN_STEP:g}`, target strength `{TARGET_STRENGTH:g}`, "
        f"physical E tol `{PHYSICAL_E_TOL:g}`.",
        "",
        "| label | eta | step | action | source frac | initial full | final full | accepted | solver accepted | phys ok | phys E | peak phys E R/rg | buffer E | int E | outer omega | nodes 250-270 | min dxi | max dxi | target peak R/rg | target monitor R/rg | f_adv global | f_adv inner | Lrad/LEdd | max H/R | Rson/rg | pivot | nfev | elapsed s |",
        "|---|---:|---:|---|---:|---:|---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        formatted = {
            key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value
            for key, value in row.items()
        }
        lines.append(
            "| {label} | {eta} | {attempt_step} | {action} | {source_fraction} | {initial_full} | "
            "{final_full} | {accepted} | {solver_accepted} | {physical_E_gate_eligible} | "
            "{partition_physical_E} | {partition_peak_physical_E_rg} | {partition_buffer_E} | "
            "{interval_E} | {outer_omega} | {grid_nodes_250_270} | {grid_dx_min} | {grid_dx_max} | "
            "{target_monitor_peak_physical_E_R_rg} | {target_peak_monitor_R_rg} | {f_adv_global} | "
            "{f_adv_inner} | {Lrad_LEdd} | {max_H_R} | {Rson_rg} | {pivot} | {nfev} | {elapsed_s} |".format(
                **formatted
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(
        json.dumps(
            json_safe([{key: value for key, value in row.items() if key not in {"z", "custom_grid_xi"}} for row in rows]),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = params_from_checkpoint(ANCHOR_CHECKPOINT, fiducial, mdot_edd)
    anchor_params = refresh_outer_slopes_from_state(anchor_z, anchor_params)
    source_xi = current_grid_xi(anchor_params)
    target_xi, target_grid_info, _profile_info = residual_remesh_grid_xi(
        anchor_z,
        anchor_params,
        n_nodes=int(anchor_params.n_nodes),
        strength=TARGET_STRENGTH,
    )
    target_xi_array = np.asarray(target_xi, dtype=float)

    rows: list[dict[str, Any]] = []
    current_eta = 0.0
    current_z = np.asarray(anchor_z, dtype=float)
    current_params = replace(anchor_params, custom_grid_xi=tuple(float(value) for value in source_xi))
    current_params = refresh_outer_slopes_from_state(current_z, current_params)
    anchor_partition = residual_partition_audit_from_state_vector(current_z, current_params)
    anchor_full = max_residual(current_z, current_params)
    print(
        f"anchor eta=0 full={anchor_full:.3e} physE={anchor_partition.physical_energy_max:.3e} "
        f"targetPeak={target_grid_info['monitor_peak_physical_E_R_rg']:.4g}",
        flush=True,
    )

    step = min(abs(INITIAL_STEP), abs(TARGET_ETA - current_eta))
    direction = 1.0 if TARGET_ETA >= current_eta else -1.0
    attempt = 0
    while direction * (TARGET_ETA - current_eta) > 1.0e-12:
        trial_step = min(step, direction * (TARGET_ETA - current_eta))
        eta = current_eta + direction * trial_step
        custom_grid_xi = blended_grid(source_xi, target_xi_array, eta)
        trial_params = replace(
            current_params,
            custom_grid_xi=custom_grid_xi,
            max_nfev=NEWTON_MAX_NFEV,
            residual_tol=NEWTON_RESIDUAL_TOL,
        )
        profile = transonic_profile_from_state_vector(current_z, current_params)
        seed = remap_profile_to_new_sonic_grid(
            profile,
            trial_params,
            temperature_mdot_power=0.0,
            method=REMAP_METHOD,
        )
        trial_params = refresh_outer_slopes_from_state(seed, trial_params)
        initial_full = max_residual(seed, trial_params)
        attempt += 1
        print(
            f"eta attempt={attempt} current={current_eta:.6g} target={eta:.6g} "
            f"step={direction * trial_step:.6g} initial={initial_full:.3e}",
            flush=True,
        )
        if initial_full > MAX_INITIAL_FULL and trial_step > MIN_STEP * (1.0 + 1.0e-12):
            step = max(MIN_STEP, trial_step * SHRINK)
            print(f"  pre-reject initial residual; reducing eta step to {step:.6g}", flush=True)
            continue

        t0 = time.perf_counter()
        polish = polish_best(seed, trial_params)
        final_params = refresh_outer_slopes_from_state(polish.z, trial_params)
        elapsed = time.perf_counter() - t0
        row = row_for_state(
            label="grid_homotopy",
            eta=eta,
            step=direction * trial_step,
            seed=seed,
            z=polish.z,
            params=final_params,
            polish=polish,
            elapsed_s=elapsed,
            target_grid_info=target_grid_info,
            action="pending",
        )
        if row["accepted"]:
            current_eta = float(eta)
            current_z = np.asarray(polish.z, dtype=float)
            current_params = final_params
            if row["anchor_eligible"] and int(row["nfev"]) <= 8:
                step = min(MAX_STEP, max(MIN_STEP, trial_step * GROWTH))
                row["action"] = "accepted_grow"
            else:
                step = max(MIN_STEP, trial_step)
                row["action"] = "accepted_hold"
        else:
            if trial_step <= MIN_STEP * (1.0 + 1.0e-12):
                row["action"] = "stop_min_step_failed"
                rows.append(row)
                save_checkpoint(row, final_params)
                write_outputs(rows)
                print(
                    f"  stop eta={eta:.6g} full={row['final_full']:.3e} "
                    f"physE={row['partition_physical_E']:.3e}",
                    flush=True,
                )
                break
            step = max(MIN_STEP, trial_step * SHRINK)
            row["action"] = "reject_shrink"
            print(f"  rejected; reducing eta step to {step:.6g}", flush=True)
        rows.append(row)
        save_checkpoint(row, final_params)
        write_outputs(rows)
        print(
            f"  final={row['final_full']:.3e} accepted={row['accepted']} "
            f"physE={row['partition_physical_E']:.3e} next_step={direction * step:.6g} "
            f"action={row['action']}",
            flush=True,
        )

    write_outputs(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
