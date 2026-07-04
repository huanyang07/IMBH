"""Defect-preserving nested refinement for high-source stream checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    pack_state,
    residual_partition_audit_from_state_vector,
    solve_square_transonic_polish,
    state_bounds,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _differential_interval_residual_from_unpacked,
    _interval_residual_from_unpacked,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import json_safe
from run_standard_slim_stream_mass_annulus_scan import (
    NEWTON_LINEAR_DAMPINGS,
    advection_diagnostic,
    apply_outer_slopes_from_state,
    load_anchor,
    max_residual,
    relative_root_path,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STREAM_NESTED_REFINEMENT_CHECKPOINT",
    "outputs/checkpoints/high_mdot_stream_outer_buffer_energy_merit_next_diag4_0898125_to089825/"
    "energy_merit_next_diag4_mass_0p89825_torque_0p005_mdot_2_N896.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STREAM_NESTED_REFINEMENT_TABLE",
    "outputs/tables/high_mdot_stream_nested_refinement.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STREAM_NESTED_REFINEMENT_CHECKPOINT_DIR",
    "outputs/checkpoints/high_mdot_stream_nested_refinement",
)
TOP_K = int(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_TOP_K", "12"))
WINDOW_SPECS = os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_WINDOWS", "259.2:20,298:10,333.8:30")
MAX_INSERT = int(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_MAX_INSERT", "24"))
POLISH = os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_POLISH", "1") != "0"
MAX_ITER = int(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_MAX_ITER", "3"))
MAX_NFEV = int(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_MAX_NFEV", "4000"))
RESIDUAL_TOL = float(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_RESIDUAL_TOL", "1e-8"))
JACOBIAN_REL_STEP = float(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_JACOBIAN_REL_STEP", "3e-5"))
ENERGY_JACOBIAN_REL_STEP_RAW = os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_ENERGY_JACOBIAN_REL_STEP", "").strip()
ENERGY_JACOBIAN_REL_STEP = None if not ENERGY_JACOBIAN_REL_STEP_RAW else float(ENERGY_JACOBIAN_REL_STEP_RAW)
ENERGY_MERIT = os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_ENERGY_MERIT", "physical_max").strip().lower()
ENERGY_MERIT_TOL = float(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_ENERGY_MERIT_TOL", "1e-5"))
ENERGY_ROW_PRIORITY = float(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_ENERGY_ROW_PRIORITY", "5"))
MAX_STEP_NORM = float(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_MAX_STEP_NORM", "0.16"))
LINE_SEARCH_MAX_REDUCTIONS = int(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_LINE_SEARCH_MAX_REDUCTIONS", "12"))
LOCAL_INIT = os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_LOCAL_INIT", "1") != "0"
LOCAL_INIT_MAX_NFEV = int(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_LOCAL_INIT_MAX_NFEV", "120"))
LOCAL_INIT_ENERGY_WEIGHT = float(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_LOCAL_INIT_ENERGY_WEIGHT", "5"))
LOCAL_INIT_PRIOR_WEIGHT = float(os.environ.get("IMBH_STREAM_NESTED_REFINEMENT_LOCAL_INIT_PRIOR_WEIGHT", "1e-5"))


def parse_windows(spec: str) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for piece in spec.replace(";", ",").split(","):
        if not piece.strip():
            continue
        if ":" not in piece:
            raise ValueError(f"window must be center_rg:half_width_rg, got {piece!r}")
        center, half_width = piece.split(":", 1)
        windows.append((float(center), float(half_width)))
    return windows


def interval_residuals(z: np.ndarray, params) -> tuple[np.ndarray, np.ndarray]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    residuals = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            for idx in range(len(logR) - 1)
        ],
        dtype=float,
    )
    R_mid_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    return R_mid_rg, residuals


def split_interval_indices(z: np.ndarray, params) -> np.ndarray:
    R_mid_rg, residuals = interval_residuals(z, params)
    selected: set[int] = set()
    if params.outer_buffer_inner_rg is None:
        physical_indices = np.arange(R_mid_rg.size, dtype=int)
    else:
        physical_indices = np.nonzero(R_mid_rg < float(params.outer_buffer_inner_rg))[0]
    if physical_indices.size == 0:
        physical_indices = np.arange(R_mid_rg.size, dtype=int)
    if TOP_K > 0:
        order = physical_indices[np.argsort(np.abs(residuals[physical_indices, 1]))[::-1]]
        selected.update(int(idx) for idx in order[:TOP_K])
    for center, half_width in parse_windows(WINDOW_SPECS):
        window_indices = np.nonzero((np.abs(R_mid_rg - center) <= half_width) & np.isin(np.arange(R_mid_rg.size), physical_indices))[0]
        selected.update(int(idx) for idx in window_indices)
    if not selected:
        selected.add(int(physical_indices[np.argmax(np.abs(residuals[physical_indices, 1]))]))
    selected = set(sorted(selected)[: max(1, MAX_INSERT)])
    return np.asarray(sorted(selected), dtype=int)


def local_initialize_inserted_nodes(
    seed: np.ndarray,
    params,
    inserted_nodes: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not LOCAL_INIT or inserted_nodes.size == 0:
        return seed, {"enabled": bool(LOCAL_INIT), "nfev": 0, "local_max_before": np.nan, "local_max_after": np.nan}
    from scipy.optimize import least_squares

    logu, logT, logR_son, lambda0, logR = unpack_state(seed, params)
    lower, upper = state_bounds(params)
    active_columns = np.concatenate([inserted_nodes, params.n_nodes + inserted_nodes])
    x0 = np.concatenate([logu[inserted_nodes], logT[inserted_nodes]])
    local_lower = lower[active_columns]
    local_upper = upper[active_columns]
    touched: set[int] = set()
    for node in inserted_nodes:
        if node > 0:
            touched.add(int(node) - 1)
        if node < params.n_nodes - 1:
            touched.add(int(node))
    solve_intervals = np.asarray(sorted(touched), dtype=int)
    base_x = np.array(x0, copy=True)

    def unpack_trial(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        trial_logu = np.array(logu, copy=True)
        trial_logT = np.array(logT, copy=True)
        trial_logu[inserted_nodes] = x[: inserted_nodes.size]
        trial_logT[inserted_nodes] = x[inserted_nodes.size :]
        return trial_logu, trial_logT

    def residual(x: np.ndarray) -> np.ndarray:
        trial_logu, trial_logT = unpack_trial(x)
        pieces: list[float] = []
        for idx in solve_intervals:
            row = _interval_residual_from_unpacked(trial_logu, trial_logT, logR, lambda0, params, int(idx))
            pieces.extend([float(row[0]), LOCAL_INIT_ENERGY_WEIGHT * float(row[1])])
        if LOCAL_INIT_PRIOR_WEIGHT > 0.0:
            pieces.extend((LOCAL_INIT_PRIOR_WEIGHT * (np.asarray(x, dtype=float) - base_x)).tolist())
        return np.asarray(pieces, dtype=float)

    before = residual(x0)
    lsq = least_squares(
        residual,
        x0,
        bounds=(local_lower, local_upper),
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=LOCAL_INIT_MAX_NFEV,
    )
    final_logu, final_logT = unpack_trial(np.asarray(lsq.x, dtype=float))
    initialized = pack_state(final_logu, final_logT, logR_son, lambda0)
    after = residual(np.asarray(lsq.x, dtype=float))
    return initialized, {
        "enabled": True,
        "inserted_nodes": inserted_nodes.tolist(),
        "solve_intervals": solve_intervals.tolist(),
        "local_max_before": float(np.max(np.abs(before))),
        "local_max_after": float(np.max(np.abs(after))),
        "nfev": int(lsq.nfev),
        "cost": float(lsq.cost),
        "optimality": float(lsq.optimality),
        "success": bool(lsq.success),
        "message": str(lsq.message),
    }


def nested_seed(z: np.ndarray, params) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray, dict[str, Any]]:
    logu, logT, logR_son, lambda0, _logR = unpack_state(z, params)
    old_xi = np.asarray(params.custom_grid_xi if params.custom_grid_xi is not None else np.linspace(0.0, 1.0, params.n_nodes), dtype=float)
    split_indices = split_interval_indices(z, params)
    midpoint_xi = 0.5 * (old_xi[split_indices] + old_xi[split_indices + 1])
    new_xi = np.asarray(sorted(set(old_xi.tolist() + midpoint_xi.tolist())), dtype=float)
    new_xi[0] = 0.0
    new_xi[-1] = 1.0
    if np.any(np.diff(new_xi) <= 0.0):
        raise RuntimeError("nested refinement produced a non-monotonic grid")
    new_params = replace(
        params,
        n_nodes=int(new_xi.size),
        custom_grid_xi=tuple(float(value) for value in new_xi),
        max_nfev=MAX_NFEV,
        residual_tol=RESIDUAL_TOL,
    )
    new_logu = np.interp(new_xi, old_xi, logu)
    new_logT = np.interp(new_xi, old_xi, logT)
    for idx, xi_value in enumerate(old_xi):
        new_idx = int(np.argmin(np.abs(new_xi - xi_value)))
        if abs(float(new_xi[new_idx] - xi_value)) <= 1.0e-14:
            new_logu[new_idx] = logu[idx]
            new_logT[new_idx] = logT[idx]
    inserted_nodes = np.asarray([int(np.argmin(np.abs(new_xi - xi_value))) for xi_value in midpoint_xi], dtype=int)
    seed = pack_state(new_logu, new_logT, logR_son, lambda0)
    new_params = apply_outer_slopes_from_state(seed, new_params)
    seed, local_info = local_initialize_inserted_nodes(seed, new_params, inserted_nodes)
    new_params = apply_outer_slopes_from_state(seed, new_params)
    return seed, new_params, split_indices, inserted_nodes, local_info


def diagnostics(z: np.ndarray, params) -> dict[str, Any]:
    partition = residual_partition_audit_from_state_vector(z, params)
    R_mid_rg, residuals = interval_residuals(z, params)
    peak_E = int(np.argmax(np.abs(residuals[:, 1])))
    profile = transonic_profile_from_state_vector(z, params)
    row = {
        "full": max_residual(z, params),
        "physical_E": float(partition.physical_energy_max),
        "physical_E_l2": float(partition.physical_energy_l2),
        "buffer_E": float(partition.buffer_energy_max),
        "terminal_omega": float(partition.terminal_omega),
        "peak_E_rg": float(R_mid_rg[peak_E]),
        "peak_E_value": float(residuals[peak_E, 1]),
        "N": int(params.n_nodes),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "max_H_R": float(np.max(profile.H_over_R)),
    }
    row.update(advection_diagnostic(z, params))
    return row


def save_checkpoint(z: np.ndarray, params, row: dict[str, Any]) -> str:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_mass = f"{params.stream_source_fraction:.9g}".replace(".", "p").replace("-", "m")
    path = CHECKPOINT_DIR / f"nested_refined_mass_{safe_mass}_N{params.n_nodes}.npz"
    slopes = params.outer_match_log_slopes
    np.savez_compressed(
        path,
        z=np.asarray(z, dtype=float),
        ratio=np.array(params.mdot_edd_ratio),
        R_out_rg=np.array(params.R_out_rg),
        n_nodes=np.array(params.n_nodes),
        grid_power=np.array(params.grid_power),
        custom_grid_xi=np.asarray(params.custom_grid_xi, dtype=float),
        outer_closure=np.array(params.outer_closure),
        outer_match_log_slopes=np.asarray([np.nan, np.nan] if slopes is None else slopes, dtype=float),
        outer_robin_chi=np.array(params.outer_robin_chi),
        outer_robin_slope_target=np.array(params.outer_robin_slope_target),
        outer_robin_slope_scale=np.array(params.outer_robin_slope_scale),
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
        stream_mass_fraction=np.array(params.stream_mass_fraction),
        stream_mass_center_fraction=np.array(params.stream_mass_center_fraction),
        stream_mass_log_width=np.array(params.stream_mass_log_width),
        wind_sink_fraction=np.array(params.wind_sink_fraction),
        wind_sink_center_fraction=np.array(params.wind_sink_center_fraction),
        wind_sink_log_width=np.array(params.wind_sink_log_width),
        stream_heating_efficiency=np.array(params.stream_heating_efficiency),
        interval_residual_form=np.array(params.interval_residual_form),
        integrated_residual_weighting=np.array(params.integrated_residual_weighting),
        row_json=np.array(json.dumps(json_safe(row), sort_keys=True)),
    )
    return relative_root_path(path)


def write_outputs(row: dict[str, Any]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(json_safe(row), indent=2, sort_keys=True) + "\n")
    lines = [
        "# High-Source Nested Refinement",
        "",
        f"Checkpoint `{relative_root_path(CHECKPOINT)}`.",
        "",
        f"Top-K `{TOP_K}`, windows `{WINDOW_SPECS}`, inserted `{row['inserted_nodes']}` nodes.",
        f"Local inserted-node init `{row['local_init'].get('local_max_before', np.nan):.6e}` -> "
        f"`{row['local_init'].get('local_max_after', np.nan):.6e}` in "
        f"`{row['local_init'].get('nfev', 0)}` evaluations.",
        "",
        "| metric | before | seed | final |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in (
        "N",
        "full",
        "physical_E",
        "physical_E_l2",
        "buffer_E",
        "terminal_omega",
        "peak_E_rg",
        "peak_E_value",
        "f_adv_global",
        "f_adv_inner",
        "Lrad_LEdd",
        "Rson_rg",
        "max_H_R",
    ):
        lines.append(f"| `{key}` | {row['before'][key]:.6e} | {row['seed'][key]:.6e} | {row['final'][key]:.6e} |")
    if row.get("checkpoint"):
        lines.extend(["", f"Final checkpoint `{row['checkpoint']}`."])
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    z, params = load_anchor(CHECKPOINT, fiducial, mdot_edd)
    before = diagnostics(z, params)
    seed, refined_params, split_indices, inserted_nodes, local_info = nested_seed(z, params)
    seed_diag = diagnostics(seed, refined_params)
    final_z = seed
    final_params = refined_params
    polish_row: dict[str, Any] = {"enabled": bool(POLISH)}
    if POLISH:
        polish = solve_square_transonic_polish(
            refined_params,
            seed,
            pivot="auto",
            method="newton",
            max_iter=MAX_ITER,
            max_nfev=MAX_NFEV,
            residual_tol=RESIDUAL_TOL,
            jacobian_rel_step=JACOBIAN_REL_STEP,
            energy_jacobian_rel_step=ENERGY_JACOBIAN_REL_STEP,
            use_block_jacobian=True,
            line_search_max_reductions=LINE_SEARCH_MAX_REDUCTIONS,
            linear_solver="regularized_lsmr",
            linear_dampings=NEWTON_LINEAR_DAMPINGS,
            max_step_norm=MAX_STEP_NORM,
            energy_merit=ENERGY_MERIT,
            energy_merit_tol=ENERGY_MERIT_TOL,
            energy_merit_l2_tol=ENERGY_MERIT_TOL,
            energy_merit_global_tol=ENERGY_MERIT_TOL,
            energy_merit_require_decrease=True,
            energy_row_priority=ENERGY_ROW_PRIORITY,
        )
        final_z = polish.z
        final_params = refined_params
        polish_row.update(
            {
                "success": bool(polish.result.optimizer_success),
                "message": str(polish.result.message),
                "nfev": int(polish.result.nfev),
                "njev": int(polish.result.njev),
                "iterations": int(polish.iterations),
                "pivot": str(polish.pivot),
                "final_square": float(polish.final_square_max_residual),
                "final_step_norm": float(polish.final_step_norm),
            }
        )
    final = diagnostics(final_z, final_params)
    row = {
        "checkpoint": "",
        "source_checkpoint": relative_root_path(CHECKPOINT),
        "inserted_nodes": int(refined_params.n_nodes - params.n_nodes),
        "split_indices": split_indices.tolist(),
        "inserted_node_indices": inserted_nodes.tolist(),
        "local_init": local_info,
        "params": asdict(final_params),
        "before": before,
        "seed": seed_diag,
        "final": final,
        "polish": polish_row,
    }
    row["checkpoint"] = save_checkpoint(final_z, final_params, row)
    write_outputs(row)
    print(f"wrote {relative_root_path(TABLE_OUTPUT)}")
    print(f"final checkpoint {row['checkpoint']}")


if __name__ == "__main__":
    main()
