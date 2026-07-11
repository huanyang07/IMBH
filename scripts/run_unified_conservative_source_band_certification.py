"""Certify low-eta conservative roots on source-only refined grids."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import (
    conservative_jacobian_sparsity,
    conservative_residual,
    conservative_residual_audit,
    conservative_residual_profile,
    conservative_state_bounds,
    conservative_transport_profile,
    multidomain_conservative_grid,
    remap_conservative_state,
    solve_conservative_disk,
    source_block_refined_conservative_grid,
    unpack_conservative_state,
)
import run_unified_conservative_mdot5_wind_ladder as wind_ladder


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs/checkpoints/unified_conservative_mdot5_eta_ladder"
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_source_band_certification"
OUTPUT = ROOT / "outputs/tables/unified_conservative_source_band_certification.json"
ETA_VALUES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_SOURCE_CERT_ETA_VALUES", "10,8,7").split(",")
    if piece.strip()
)
SOURCE_NODE_VALUES = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_SOURCE_CERT_NODE_VALUES", "24,32,48,64").split(",")
    if piece.strip()
)
BASE_N = int(os.environ.get("IMBH_SOURCE_CERT_BASE_N", "384"))
LOCAL_MAX_NFEV = int(os.environ.get("IMBH_SOURCE_CERT_LOCAL_MAX_NFEV", "400"))
GLOBAL_MAX_NFEV = int(os.environ.get("IMBH_SOURCE_CERT_GLOBAL_MAX_NFEV", "500"))
HALO_NODES = int(os.environ.get("IMBH_SOURCE_CERT_HALO_NODES", "2"))
INNER_NODES = int(os.environ.get("IMBH_SOURCE_CERT_INNER_NODES", "12"))
TARGET_TOTAL_N = int(os.environ.get("IMBH_SOURCE_CERT_TARGET_N", "0"))


def _safe(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _load_eta(eta_e: float):
    wind_ladder.N_NODES = BASE_N
    _unused, params = wind_ladder._starting_problem()
    path = SOURCE_DIR / f"mdot5_fs0p3_eps0p2_eta{_safe(eta_e)}_N{BASE_N}.npz"
    with np.load(path) as data:
        state = np.asarray(data["x"], dtype=float)
        grid = tuple(float(value) for value in data["custom_grid_xi"])
    params = replace(
        params,
        disk=replace(
            params.disk,
            custom_grid_xi=grid,
            wind_energy_limited_epsilon=0.2,
        ),
        closure=replace(params.closure, wind_launch_energy_multiplier=float(eta_e)),
        max_nfev=GLOBAL_MAX_NFEV,
        jacobian_rel_step=1.0e-4,
    )
    return state, params


def _source_indices(state, params) -> np.ndarray:
    *_fields, log_r = unpack_conservative_state(state, params)
    center = np.log(
        params.disk.stream_source_center_fraction * params.disk.R_out
    )
    width = float(params.disk.stream_source_log_width)
    tolerance = 1.0e-12 * max(abs(center), 1.0)
    return np.flatnonzero(
        (log_r >= center - width - tolerance)
        & (log_r <= center + width + tolerance)
    )


def _source_local_correct(state, params):
    """Relax source+halo fields while freezing the sonic and outer blocks."""

    inside = _source_indices(state, params)
    n = int(params.disk.n_nodes)
    first = max(1, int(inside[0]) - HALO_NODES)
    last = min(n - 2, int(inside[-1]) + HALO_NODES)
    active_nodes = np.arange(first, last + 1)
    active_columns = np.concatenate(
        [field * n + active_nodes for field in range(5)]
    )
    interval_indices = np.arange(first - 1, last + 1)
    active_rows = np.concatenate(
        [np.arange(5 * idx, 5 * idx + 5) for idx in interval_indices]
    )
    lower, upper = conservative_state_bounds(params)
    sparsity = conservative_jacobian_sparsity(params)[active_rows, :][
        :, active_columns
    ]
    reference = np.asarray(state, dtype=float).copy()

    def residual(values):
        trial = reference.copy()
        trial[active_columns] = values
        return conservative_residual(trial, params)[active_rows]

    result = least_squares(
        residual,
        reference[active_columns],
        bounds=(lower[active_columns], upper[active_columns]),
        jac_sparsity=sparsity,
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=LOCAL_MAX_NFEV,
        diff_step=params.jacobian_rel_step,
        verbose=0,
    )
    corrected = reference.copy()
    corrected[active_columns] = result.x
    return corrected, {
        "nfev": int(result.nfev),
        "message": str(result.message),
        "active_node_first": first,
        "active_node_last": last,
        "active_nodes": int(active_nodes.size),
    }


def _inner_local_correct(state, params):
    """Relax the sonic block while freezing the certified source region."""

    n = int(params.disk.n_nodes)
    active_nodes = np.arange(min(max(INNER_NODES, 3), n - 2))
    active_columns = np.concatenate(
        [field * n + active_nodes for field in range(5)]
        + [np.asarray([5 * n], dtype=int)]
    )
    interval_indices = np.arange(active_nodes.size)
    interval_rows = np.concatenate(
        [np.arange(5 * idx, 5 * idx + 5) for idx in interval_indices]
    )
    tail = 5 * (n - 1)
    active_rows = np.concatenate(
        [interval_rows, np.arange(tail + 2, tail + 6)]
    )
    lower, upper = conservative_state_bounds(params)
    sparsity = conservative_jacobian_sparsity(params)[active_rows, :][
        :, active_columns
    ]
    reference = np.asarray(state, dtype=float).copy()

    def residual(values):
        trial = reference.copy()
        trial[active_columns] = values
        return conservative_residual(trial, params)[active_rows]

    result = least_squares(
        residual,
        reference[active_columns],
        bounds=(lower[active_columns], upper[active_columns]),
        jac_sparsity=sparsity,
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=LOCAL_MAX_NFEV,
        diff_step=params.jacobian_rel_step,
        verbose=0,
    )
    corrected = reference.copy()
    corrected[active_columns] = result.x
    return corrected, {
        "nfev": int(result.nfev),
        "message": str(result.message),
        "active_nodes": int(active_nodes.size),
    }


def _profile_summary(state, params) -> dict[str, object]:
    residual = conservative_residual_profile(state, params)
    transport = conservative_transport_profile(state, params)
    fields = (
        "radial",
        "mass",
        "angular_momentum",
        "energy",
        "energy_compatibility",
    )
    values = np.vstack([np.abs(residual[name]) for name in fields])
    score = np.max(values, axis=0)
    peak = int(np.argmax(score))
    peak_family = fields[int(np.argmax(values[:, peak]))]
    center = params.disk.stream_source_center_fraction * params.disk.R_out_rg
    width = params.disk.stream_source_log_width
    left = center * np.exp(-width)
    right = center * np.exp(width)
    source_mask = (residual["R_mid_rg"] >= left) & (residual["R_mid_rg"] <= right)
    outside_mask = ~source_mask
    exact_total = float(np.sum(transport["stream_mass"]))
    quadrature_total = float(np.sum(transport["stream_mass_quadrature"]))
    target = float(params.disk.stream_source_fraction * params.flux_scales.mdot)
    return {
        "peak_residual": float(score[peak]),
        "peak_family": peak_family,
        "peak_radius_rg": float(residual["R_mid_rg"][peak]),
        "source_max": float(np.max(score[source_mask])),
        "outside_max": float(np.max(score[outside_mask])),
        "first_interval": float(score[0]),
        "exact_stream_total_over_target": exact_total / target,
        "simpson_stream_total_over_target": quadrature_total / target,
        "simpson_stream_error_over_mdot_inner": float(
            (quadrature_total - exact_total) / params.flux_scales.mdot
        ),
        "source_nodes": int(_source_indices(state, params).size),
        "total_nodes": int(params.disk.n_nodes),
    }


def run() -> list[dict[str, object]]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for eta_e in ETA_VALUES:
        base_state, base_params = _load_eta(eta_e)
        for source_nodes in SOURCE_NODE_VALUES:
            if TARGET_TOTAL_N > 0:
                grid = multidomain_conservative_grid(
                    base_state,
                    base_params,
                    target_n=TARGET_TOTAL_N,
                    source_nodes=source_nodes,
                    frozen_inner_nodes=INNER_NODES,
                )
            else:
                grid = source_block_refined_conservative_grid(
                    base_state, base_params, source_nodes=source_nodes
                )
            target_disk = replace(
                base_params.disk,
                n_nodes=len(grid),
                custom_grid_xi=grid,
            )
            remapped, params = remap_conservative_state(
                base_state, base_params, target_disk, method="pchip"
            )
            remap_audit = conservative_residual_audit(remapped, params)
            local_state, local_info = _source_local_correct(remapped, params)
            local_audit = conservative_residual_audit(local_state, params)
            inner_state, inner_info = _inner_local_correct(local_state, params)
            inner_audit = conservative_residual_audit(inner_state, params)
            solved = solve_conservative_disk(inner_state, params)
            row = {
                "eta_E": eta_e,
                "requested_source_nodes": source_nodes,
                "remap": asdict(remap_audit),
                "local": asdict(local_audit),
                "local_solver": local_info,
                "inner": asdict(inner_audit),
                "inner_solver": inner_info,
                "global": asdict(solved.final_audit),
                "global_nfev": int(solved.nfev),
                "global_message": solved.message,
                "accepted_exploratory": solved.final_audit.maximum <= 3.0e-5,
                "accepted_preferred": solved.final_audit.maximum <= 1.0e-5,
                "profile": _profile_summary(solved.x, params),
            }
            rows.append(row)
            path = CHECKPOINT_DIR / (
                f"mdot5_eta{_safe(eta_e)}_source{source_nodes}_N{params.disk.n_nodes}.npz"
            )
            np.savez_compressed(
                path,
                x=solved.x,
                custom_grid_xi=np.asarray(params.disk.custom_grid_xi, dtype=float),
                row_json=np.asarray(json.dumps(row, sort_keys=True)),
            )
            print(json.dumps(row, sort_keys=True), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
