"""Local energy-row patch solve for high-source stream checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import pack_state, residual_partition_audit_from_state_vector, state_bounds, unpack_state
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _differential_interval_residual_from_unpacked,
    _interval_residual_from_unpacked,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import json_safe
from run_standard_slim_stream_mass_annulus_scan import (
    advection_diagnostic,
    load_anchor,
    max_residual,
    relative_root_path,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STREAM_ENERGY_PATCH_CHECKPOINT",
    "outputs/checkpoints/high_mdot_stream_outer_buffer_energy_merit_next_diag4_089825_to08985/"
    "energy_merit_next_diag4_mass_0p8985_torque_0p005_mdot_2_N896.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STREAM_ENERGY_PATCH_TABLE",
    "outputs/tables/high_mdot_stream_energy_patch_solve.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STREAM_ENERGY_PATCH_CHECKPOINT_DIR",
    "outputs/checkpoints/high_mdot_stream_energy_patch_solve",
)
WINDOW_SPECS = os.environ.get("IMBH_STREAM_ENERGY_PATCH_WINDOWS", "259.2:20,298:10,333.8:30")
TOP_K = int(os.environ.get("IMBH_STREAM_ENERGY_PATCH_TOP_K", "8"))
NODE_PAD = int(os.environ.get("IMBH_STREAM_ENERGY_PATCH_NODE_PAD", "1"))
ENERGY_WEIGHT = float(os.environ.get("IMBH_STREAM_ENERGY_PATCH_ENERGY_WEIGHT", "5"))
PRIOR_WEIGHT = float(os.environ.get("IMBH_STREAM_ENERGY_PATCH_PRIOR_WEIGHT", "1e-4"))
MAX_NFEV = int(os.environ.get("IMBH_STREAM_ENERGY_PATCH_MAX_NFEV", "250"))


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


def selected_intervals(z: np.ndarray, params) -> np.ndarray:
    R_mid_rg, residuals = interval_residuals(z, params)
    selected: set[int] = set()
    for center, half_width in parse_windows(WINDOW_SPECS):
        mask = np.abs(R_mid_rg - center) <= half_width
        selected.update(int(idx) for idx in np.nonzero(mask)[0])
    if TOP_K > 0:
        order = np.argsort(np.abs(residuals[:, 1]))[::-1]
        selected.update(int(idx) for idx in order[:TOP_K])
    if not selected:
        selected.add(int(np.argmax(np.abs(residuals[:, 1]))))
    return np.asarray(sorted(selected), dtype=int)


def active_nodes_from_intervals(intervals: np.ndarray, n_nodes: int) -> np.ndarray:
    nodes: set[int] = set()
    for idx in intervals:
        start = max(1, int(idx) - NODE_PAD)
        stop = min(n_nodes - 2, int(idx) + 1 + NODE_PAD)
        nodes.update(range(start, stop + 1))
    return np.asarray(sorted(nodes), dtype=int)


def touched_intervals(nodes: np.ndarray, n_nodes: int) -> np.ndarray:
    intervals: set[int] = set()
    for node in nodes:
        if node > 0:
            intervals.add(int(node) - 1)
        if node < n_nodes - 1:
            intervals.add(int(node))
    return np.asarray(sorted(intervals), dtype=int)


def local_patch_solve(z: np.ndarray, params) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import least_squares

    logu, logT, logR_son, lambda0, logR = unpack_state(z, params)
    target_intervals = selected_intervals(z, params)
    active_nodes = active_nodes_from_intervals(target_intervals, params.n_nodes)
    solve_intervals = touched_intervals(active_nodes, params.n_nodes)
    lower, upper = state_bounds(params)
    active_columns = np.concatenate([active_nodes, params.n_nodes + active_nodes])
    x0 = np.concatenate([logu[active_nodes], logT[active_nodes]])
    local_lower = lower[active_columns]
    local_upper = upper[active_columns]
    base_x = np.array(x0, copy=True)

    def unpack_trial(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        trial_logu = np.array(logu, copy=True)
        trial_logT = np.array(logT, copy=True)
        trial_logu[active_nodes] = x[: active_nodes.size]
        trial_logT[active_nodes] = x[active_nodes.size :]
        return trial_logu, trial_logT

    def residual(x: np.ndarray) -> np.ndarray:
        trial_logu, trial_logT = unpack_trial(x)
        pieces: list[float] = []
        for idx in solve_intervals:
            row = _interval_residual_from_unpacked(trial_logu, trial_logT, logR, lambda0, params, int(idx))
            pieces.extend([float(row[0]), ENERGY_WEIGHT * float(row[1])])
        if PRIOR_WEIGHT > 0.0:
            pieces.extend((PRIOR_WEIGHT * (np.asarray(x, dtype=float) - base_x)).tolist())
        return np.asarray(pieces, dtype=float)

    before_local = residual(x0)
    lsq = least_squares(
        residual,
        x0,
        bounds=(local_lower, local_upper),
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV,
    )
    patched_logu, patched_logT = unpack_trial(np.asarray(lsq.x, dtype=float))
    patched_z = pack_state(patched_logu, patched_logT, logR_son, lambda0)
    after_local = residual(np.asarray(lsq.x, dtype=float))
    info = {
        "target_intervals": target_intervals.tolist(),
        "active_nodes": active_nodes.tolist(),
        "solve_intervals": solve_intervals.tolist(),
        "local_max_before": float(np.max(np.abs(before_local))),
        "local_max_after": float(np.max(np.abs(after_local))),
        "nfev": int(lsq.nfev),
        "cost": float(lsq.cost),
        "optimality": float(lsq.optimality),
        "success": bool(lsq.success),
        "message": str(lsq.message),
    }
    return patched_z, info


def checkpoint_payload(z: np.ndarray, params) -> dict[str, Any]:
    slopes = params.outer_match_log_slopes
    return {
        "z": np.asarray(z, dtype=float),
        "ratio": np.array(params.mdot_edd_ratio),
        "R_out_rg": np.array(params.R_out_rg),
        "n_nodes": np.array(params.n_nodes),
        "grid_power": np.array(params.grid_power),
        "custom_grid_xi": np.asarray(params.custom_grid_xi or np.linspace(0.0, 1.0, params.n_nodes), dtype=float),
        "outer_closure": np.array(params.outer_closure),
        "outer_match_log_slopes": np.asarray([np.nan, np.nan] if slopes is None else slopes, dtype=float),
        "outer_robin_chi": np.array(params.outer_robin_chi),
        "outer_robin_slope_target": np.array(params.outer_robin_slope_target),
        "outer_robin_slope_scale": np.array(params.outer_robin_slope_scale),
        "outer_buffer_inner_rg": np.array(np.nan if params.outer_buffer_inner_rg is None else params.outer_buffer_inner_rg),
        "outer_buffer_radial_weight": np.array(params.outer_buffer_radial_weight),
        "outer_buffer_energy_weight": np.array(params.outer_buffer_energy_weight),
        "outer_buffer_boundary_weight": np.array(params.outer_buffer_boundary_weight),
        "outer_buffer_taper_log_width": np.array(params.outer_buffer_taper_log_width),
        "stream_torque_delta_l_fraction": np.array(params.stream_torque_delta_l_fraction),
        "stream_torque_center_fraction": np.array(params.stream_torque_center_fraction),
        "stream_torque_log_width": np.array(params.stream_torque_log_width),
        "stream_source_fraction": np.array(params.stream_source_fraction),
        "stream_source_center_fraction": np.array(params.stream_source_center_fraction),
        "stream_source_log_width": np.array(params.stream_source_log_width),
        "stream_source_shape": np.array(params.stream_source_shape),
        "stream_source_shape_blend": np.array(params.stream_source_shape_blend),
        "stream_mass_fraction": np.array(params.stream_mass_fraction),
        "stream_mass_center_fraction": np.array(params.stream_mass_center_fraction),
        "stream_mass_log_width": np.array(params.stream_mass_log_width),
        "wind_sink_fraction": np.array(params.wind_sink_fraction),
        "wind_sink_center_fraction": np.array(params.wind_sink_center_fraction),
        "wind_sink_log_width": np.array(params.wind_sink_log_width),
        "stream_heating_efficiency": np.array(params.stream_heating_efficiency),
        "interval_residual_form": np.array(params.interval_residual_form),
        "integrated_residual_weighting": np.array(params.integrated_residual_weighting),
    }


def diagnostics(z: np.ndarray, params) -> dict[str, Any]:
    partition = residual_partition_audit_from_state_vector(z, params)
    R_mid_rg, residuals = interval_residuals(z, params)
    peak_E = int(np.argmax(np.abs(residuals[:, 1])))
    row = {
        "full": max_residual(z, params),
        "physical_E": float(partition.physical_energy_max),
        "physical_E_l2": float(partition.physical_energy_l2),
        "buffer_E": float(partition.buffer_energy_max),
        "terminal_omega": float(partition.terminal_omega),
        "peak_E_rg": float(R_mid_rg[peak_E]),
        "peak_E_value": float(residuals[peak_E, 1]),
    }
    row.update(advection_diagnostic(z, params))
    return row


def write_outputs(row: dict[str, Any]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(json_safe(row), indent=2, sort_keys=True) + "\n")
    lines = [
        "# High-Source Energy Patch Solve",
        "",
        f"Checkpoint `{relative_root_path(CHECKPOINT)}`.",
        "",
        f"Windows `{WINDOW_SPECS}`, top-K `{TOP_K}`, node pad `{NODE_PAD}`, energy weight `{ENERGY_WEIGHT:g}`, "
        f"prior weight `{PRIOR_WEIGHT:g}`.",
        "",
        "| metric | before | after |",
        "| --- | ---: | ---: |",
    ]
    for key in (
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
    ):
        lines.append(f"| `{key}` | {row['before'][key]:.6e} | {row['after'][key]:.6e} |")
    lines.extend(
        [
            "",
            f"Local patch max residual: `{row['patch']['local_max_before']:.6e}` -> "
            f"`{row['patch']['local_max_after']:.6e}` in `{row['patch']['nfev']}` function evaluations.",
            "",
            f"Patched checkpoint `{row['patched_checkpoint']}`.",
        ]
    )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    z, params = load_anchor(CHECKPOINT, fiducial, mdot_edd)
    before = diagnostics(z, params)
    patched_z, patch_info = local_patch_solve(z, params)
    after = diagnostics(patched_z, params)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_mass = f"{params.stream_source_fraction:.9g}".replace(".", "p").replace("-", "m")
    checkpoint_path = CHECKPOINT_DIR / f"energy_patch_mass_{safe_mass}_N{params.n_nodes}.npz"
    payload = checkpoint_payload(patched_z, params)
    payload["row_json"] = np.array(json.dumps(json_safe({"before": before, "after": after, "patch": patch_info}), sort_keys=True))
    np.savez_compressed(checkpoint_path, **payload)
    row = {
        "checkpoint": relative_root_path(CHECKPOINT),
        "patched_checkpoint": relative_root_path(checkpoint_path),
        "params": asdict(params),
        "before": before,
        "after": after,
        "patch": patch_info,
    }
    write_outputs(row)
    print(f"wrote {relative_root_path(TABLE_OUTPUT)}")
    print(f"patched checkpoint {relative_root_path(checkpoint_path)}")


if __name__ == "__main__":
    main()
