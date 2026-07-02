"""Mass-loading stream annulus scan for the standard slim disk."""

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
    remap_profile_to_new_sonic_grid,
    residual_audit_from_state_vector,
    solve_square_transonic_polish,
    square_collocation_jacobian,
    square_collocation_residual,
    state_bounds,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    stream_torque_specific_l_and_derivative,
    transonic_profile_from_state_vector,
    unpack_state,
    wind_sink_prime,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_luminosity, eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_MASS_ANCHOR",
    "outputs/checkpoints/slim_benchmark_physical_rout_homotopy_mdot1_1000_300/Rout_300_mdot_1_N640.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_MASS_TABLE",
    "outputs/tables/slim_benchmark_stream_mass_annulus_mdot1_rout300.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_MASS_FIGURE",
    "outputs/figures/slim_benchmark_stream_mass_annulus_mdot1_rout300.png",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_MASS_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_stream_mass_annulus_mdot1_rout300",
)

BRANCH_SPECS = tuple(
    piece.strip()
    for piece in os.environ.get(
        "IMBH_STANDARD_SLIM_STREAM_MASS_BRANCHES",
        "load:0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2",
    ).split(";")
    if piece.strip()
)
MASS_CENTER_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_CENTER_FRACTION", "0.8"))
MASS_LOG_WIDTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOG_WIDTH", "0.08"))
MASS_SOURCE_SHAPE_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_SHAPE", "").strip().lower()
MASS_SOURCE_SHAPE = MASS_SOURCE_SHAPE_OVERRIDE or "tanh"
MASS_SOURCE_SHAPE_BLEND_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_SHAPE_BLEND", "").strip()
TORQUE_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TORQUE_FRACTION", "0.0"))
TORQUE_CENTER_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TORQUE_CENTER_FRACTION", str(MASS_CENTER_FRACTION)))
TORQUE_LOG_WIDTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TORQUE_LOG_WIDTH", str(MASS_LOG_WIDTH)))
OUTER_CLOSURE_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_CLOSURE", "").strip()
OUTER_ROBIN_CHI_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_ROBIN_CHI", "").strip()
OUTER_ROBIN_SLOPE_TARGET_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_ROBIN_SLOPE_TARGET", "").strip()
OUTER_ROBIN_SLOPE_SCALE_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_ROBIN_SLOPE_SCALE", "").strip()
N_NODES_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_N_NODES", "").strip()
GRID_POWER_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_GRID_POWER", "").strip()
GRID_TRANSFER_MODE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_GRID_TRANSFER", "power").strip().lower()
REMAP_METHOD = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_REMAP_METHOD", "linear").strip().lower()
SOURCE_GRID_MODE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID", "none").strip().lower()
SOURCE_GRID_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_FRACTION", "0.35"))
SOURCE_GRID_HALF_WIDTHS = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_HALF_WIDTHS", "4.0"))
SOURCE_GRID_OUTER_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_OUTER_FRACTION", "0.0"))
SOURCE_GRID_OUTER_WIDTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_OUTER_WIDTH", "0.04"))
USE_SECANT_PREDICTOR = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_USE_SECANT_PREDICTOR", "0") != "0"
USE_TANGENT_PREDICTOR = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_USE_TANGENT_PREDICTOR", "0") != "0"
SECANT_DAMPING_VALUES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SECANT_DAMPINGS", "1,0.5,0.25,0.1")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
TANGENT_DAMPING_VALUES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_DAMPINGS", "1,0.5,0.25,0.1")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
TANGENT_FD_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_FD_STEP", "1e-5"))
TANGENT_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_SOLVER", "equilibrated_lsmr")
TANGENT_LINEAR_DAMPING = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_LINEAR_DAMPING", "0.0"))
TANGENT_MAXITER = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_MAXITER", "3000"))
ADAPTIVE_TARGET_RAW = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_TARGET", "").strip()
ADAPTIVE_INITIAL_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_INITIAL_STEP", "0.001"))
ADAPTIVE_MIN_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_MIN_STEP", "0.00025"))
ADAPTIVE_MAX_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_MAX_STEP", "0.005"))
ADAPTIVE_MAX_INITIAL_FULL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_MAX_INITIAL_FULL", "0.08"))
ADAPTIVE_GROWTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_GROWTH", "1.5"))
ADAPTIVE_SHRINK = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_SHRINK", "0.5"))
ADAPTIVE_COST_SHRINK_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_SHRINK_NFEV", "20"))
ADAPTIVE_COST_HARD_SHRINK_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_HARD_SHRINK_NFEV", "60"))
ADAPTIVE_COST_GROW_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_GROW_NFEV", "8"))
ADAPTIVE_COST_SHRINK = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_SHRINK", "0.5"))
ADAPTIVE_COST_HARD_SHRINK = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_HARD_SHRINK", "0.25"))
NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_MAX_ITER", "30"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_MAX_NFEV", "3000"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_MAX_STEP_NORM", "0.16"))
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_LINEAR_SOLVER", "regularized_lsmr")
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ANCHOR_TOL", "3e-6"))
REFRESH_REPOLISH = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_REFRESH_REPOLISH", "0") != "0"
INNER_RADIUS_RG = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_INNER_RG", "20.0"))
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_PIVOTS", "C2,C1").replace(":", ",").split(",")
    if piece.strip()
)


def parse_branch_specs() -> list[tuple[str, list[float]]]:
    branches: list[tuple[str, list[float]]] = []
    for spec in BRANCH_SPECS:
        if ":" not in spec:
            raise ValueError(f"branch spec must be label:mass_fractions, got {spec!r}")
        label, values = spec.split(":", 1)
        fractions = [float(piece) for piece in values.replace(",", ":").split(":") if piece.strip()]
        if not fractions:
            raise ValueError(f"branch {label!r} has no mass fractions")
        branches.append((label.strip(), fractions))
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


def resample_custom_grid_xi(custom_grid_xi: tuple[float, ...] | None, n_nodes: int) -> tuple[float, ...] | None:
    if custom_grid_xi is None:
        return None
    old = np.asarray(custom_grid_xi, dtype=float)
    if old.size < 2:
        return None
    source_index = np.linspace(0.0, 1.0, old.size)
    target_index = np.linspace(0.0, 1.0, int(n_nodes))
    new = np.interp(target_index, source_index, old)
    new[0] = 0.0
    new[-1] = 1.0
    if np.any(np.diff(new) <= 0.0):
        return None
    return tuple(float(value) for value in new)


def source_annulus_grid_xi(
    *,
    logR_son: float,
    R_out: float,
    n_nodes: int,
    grid_power: float,
    center_fraction: float,
    log_width: float,
) -> tuple[float, ...]:
    if SOURCE_GRID_MODE in {"", "none", "off", "0"}:
        return tuple(float(value) for value in np.linspace(0.0, 1.0, int(n_nodes)) ** float(grid_power))
    if SOURCE_GRID_MODE not in {"annulus", "source", "focused", "outer", "tail", "annulus_outer", "residual"}:
        raise ValueError(f"unknown source grid mode {SOURCE_GRID_MODE!r}")
    n_nodes = int(n_nodes)
    logR_out = float(np.log(R_out))
    denominator = max(logR_out - float(logR_son), 1.0e-12)
    logR_center = float(np.log(float(center_fraction) * R_out))
    xi_dense = np.linspace(0.0, 1.0, max(4096, 24 * n_nodes))
    logR_dense = float(logR_son) + xi_dense * denominator
    width = max(float(log_width) * max(float(SOURCE_GRID_HALF_WIDTHS), 0.5) / 2.0, 1.0e-6)
    source_density = np.exp(-0.5 * ((logR_dense - logR_center) / width) ** 2)
    outer_width = max(float(SOURCE_GRID_OUTER_WIDTH), 1.0e-4)
    outer_density = np.exp(-0.5 * ((xi_dense - 1.0) / outer_width) ** 2)
    source_weight = 0.0 if SOURCE_GRID_MODE in {"outer", "tail"} else 4.0 * float(SOURCE_GRID_FRACTION)
    outer_weight = 0.5 * max(1.0 - float(grid_power), 0.0) + 8.0 * float(SOURCE_GRID_OUTER_FRACTION)
    if SOURCE_GRID_MODE in {"outer", "tail", "annulus_outer", "residual"} and SOURCE_GRID_OUTER_FRACTION <= 0.0:
        outer_weight += 4.0
    density = 1.0 + source_weight * source_density + outer_weight * outer_density
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[:-1] + density[1:]) * np.diff(xi_dense))])
    cdf /= cdf[-1]
    grid = np.interp(np.linspace(0.0, 1.0, n_nodes), cdf, xi_dense)
    grid[0] = 0.0
    grid[-1] = 1.0
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("source-focused grid generation produced a non-monotonic grid")
    return tuple(float(value) for value in grid)


def prepare_anchor_grid(
    z: np.ndarray,
    params: TransonicSlimParams,
    fiducial: FiducialParams,
    mdot_edd: float,
) -> tuple[np.ndarray, TransonicSlimParams]:
    n_nodes = int(N_NODES_OVERRIDE) if N_NODES_OVERRIDE else int(params.n_nodes)
    grid_power = float(GRID_POWER_OVERRIDE) if GRID_POWER_OVERRIDE else float(params.grid_power)
    logu, _logT, logR_son, _lambda0, _logR = unpack_state(z, params)
    custom_grid_xi = params.custom_grid_xi
    if SOURCE_GRID_MODE not in {"", "none", "off", "0"}:
        custom_grid_xi = source_annulus_grid_xi(
            logR_son=float(logR_son),
            R_out=float(params.R_out),
            n_nodes=n_nodes,
            grid_power=grid_power,
            center_fraction=params.stream_source_center_fraction,
            log_width=params.stream_source_log_width,
        )
    elif GRID_TRANSFER_MODE in {"resample", "resample_current", "current", "preserve"} and n_nodes != int(params.n_nodes):
        custom_grid_xi = resample_custom_grid_xi(params.custom_grid_xi, n_nodes)
    elif n_nodes != int(params.n_nodes):
        custom_grid_xi = None
    target_params = params_for(
        fiducial,
        mdot_edd,
        ratio=params.mdot_edd_ratio,
        R_out_rg=params.R_out_rg,
        n_nodes=n_nodes,
        grid_power=grid_power,
        custom_grid_xi=custom_grid_xi if n_nodes == len(custom_grid_xi or ()) else custom_grid_xi,
        mass_fraction=float(params.stream_source_fraction),
        source_center_fraction=params.stream_source_center_fraction,
        source_log_width=params.stream_source_log_width,
        source_shape=params.stream_source_shape,
        source_shape_blend=params.stream_source_shape_blend,
        torque_fraction=params.stream_torque_delta_l_fraction,
        torque_center_fraction=params.stream_torque_center_fraction,
        torque_log_width=params.stream_torque_log_width,
        wind_sink_fraction=params.wind_sink_fraction,
        wind_sink_center_fraction=params.wind_sink_center_fraction,
        wind_sink_log_width=params.wind_sink_log_width,
        stream_heating_efficiency=params.stream_heating_efficiency,
        outer_closure=params.outer_closure,
        outer_robin_chi=params.outer_robin_chi,
        outer_robin_slope_target=params.outer_robin_slope_target,
        outer_robin_slope_scale=params.outer_robin_slope_scale,
    )
    if (
        n_nodes == int(params.n_nodes)
        and np.isclose(grid_power, params.grid_power)
        and (
            (custom_grid_xi is None and params.custom_grid_xi is None)
            or (
                custom_grid_xi is not None
                and params.custom_grid_xi is not None
                and np.allclose(np.asarray(custom_grid_xi), np.asarray(params.custom_grid_xi))
            )
        )
    ):
        return z, apply_outer_slopes_from_state(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    remapped_z = remap_profile_to_new_sonic_grid(profile, target_params, temperature_mdot_power=0.0, method=REMAP_METHOD)
    return remapped_z, apply_outer_slopes_from_state(remapped_z, target_params)


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    *,
    ratio: float,
    R_out_rg: float,
    n_nodes: int,
    grid_power: float,
    custom_grid_xi: tuple[float, ...] | None,
    mass_fraction: float,
    source_center_fraction: float | None = None,
    source_log_width: float | None = None,
    source_shape: str | None = None,
    source_shape_blend: float | None = None,
    torque_fraction: float | None = None,
    torque_center_fraction: float | None = None,
    torque_log_width: float | None = None,
    wind_sink_fraction: float = 0.0,
    wind_sink_center_fraction: float = 0.8,
    wind_sink_log_width: float = 0.08,
    stream_heating_efficiency: float = 0.0,
    outer_closure: str | None = None,
    outer_robin_chi: float = 0.0,
    outer_robin_slope_target: float = 0.0,
    outer_robin_slope_scale: float = 1.0,
) -> TransonicSlimParams:
    source_center = MASS_CENTER_FRACTION if source_center_fraction is None else float(source_center_fraction)
    source_width = MASS_LOG_WIDTH if source_log_width is None else float(source_log_width)
    selected_source_shape = MASS_SOURCE_SHAPE if source_shape is None else str(source_shape).strip().lower()
    selected_source_shape_blend = (
        float(MASS_SOURCE_SHAPE_BLEND_OVERRIDE) if MASS_SOURCE_SHAPE_BLEND_OVERRIDE else (1.0 if source_shape_blend is None else float(source_shape_blend))
    )
    torque_delta = TORQUE_FRACTION if torque_fraction is None else float(torque_fraction)
    torque_center = TORQUE_CENTER_FRACTION if torque_center_fraction is None else float(torque_center_fraction)
    torque_width = TORQUE_LOG_WIDTH if torque_log_width is None else float(torque_log_width)
    closure = OUTER_CLOSURE_OVERRIDE if OUTER_CLOSURE_OVERRIDE else (outer_closure or "pressure_supported_thin_energy")
    robin_chi = float(OUTER_ROBIN_CHI_OVERRIDE) if OUTER_ROBIN_CHI_OVERRIDE else float(outer_robin_chi)
    robin_slope_target = (
        float(OUTER_ROBIN_SLOPE_TARGET_OVERRIDE) if OUTER_ROBIN_SLOPE_TARGET_OVERRIDE else float(outer_robin_slope_target)
    )
    robin_slope_scale = (
        float(OUTER_ROBIN_SLOPE_SCALE_OVERRIDE) if OUTER_ROBIN_SLOPE_SCALE_OVERRIDE else float(outer_robin_slope_scale)
    )
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
        outer_omega_log_offset=0.0,
        outer_robin_chi=robin_chi,
        outer_robin_slope_target=robin_slope_target,
        outer_robin_slope_scale=robin_slope_scale,
        stream_torque_delta_l_fraction=torque_delta,
        stream_torque_center_fraction=torque_center,
        stream_torque_log_width=torque_width,
        stream_source_fraction=float(mass_fraction),
        stream_source_center_fraction=source_center,
        stream_source_log_width=source_width,
        stream_source_shape=selected_source_shape,
        stream_source_shape_blend=selected_source_shape_blend,
        wind_sink_fraction=float(wind_sink_fraction),
        wind_sink_center_fraction=float(wind_sink_center_fraction),
        wind_sink_log_width=float(wind_sink_log_width),
        stream_heating_efficiency=float(stream_heating_efficiency),
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def scalar_from_data(data, key: str, default):
    if key not in data:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value


def load_anchor(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    data = np.load(path, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    mass_fraction = float(scalar_from_data(data, "stream_source_fraction", scalar_from_data(data, "stream_mass_fraction", 0.0)))
    params = params_for(
        fiducial,
        mdot_edd,
        ratio=float(scalar_from_data(data, "ratio", 1.0)),
        R_out_rg=float(scalar_from_data(data, "R_out_rg", 1000.0)),
        n_nodes=int(scalar_from_data(data, "n_nodes", (len(z) - 2) // 2)),
        grid_power=float(scalar_from_data(data, "grid_power", 1.0)),
        custom_grid_xi=custom_grid_from_data(data),
        mass_fraction=mass_fraction,
        source_center_fraction=float(scalar_from_data(data, "stream_source_center_fraction", MASS_CENTER_FRACTION)),
        source_log_width=float(scalar_from_data(data, "stream_source_log_width", MASS_LOG_WIDTH)),
        source_shape=MASS_SOURCE_SHAPE_OVERRIDE or str(scalar_from_data(data, "stream_source_shape", MASS_SOURCE_SHAPE)),
        source_shape_blend=float(
            MASS_SOURCE_SHAPE_BLEND_OVERRIDE or scalar_from_data(data, "stream_source_shape_blend", 1.0)
        ),
        torque_fraction=float(scalar_from_data(data, "stream_torque_delta_l_fraction", TORQUE_FRACTION)),
        torque_center_fraction=float(scalar_from_data(data, "stream_torque_center_fraction", TORQUE_CENTER_FRACTION)),
        torque_log_width=float(scalar_from_data(data, "stream_torque_log_width", TORQUE_LOG_WIDTH)),
        wind_sink_fraction=float(scalar_from_data(data, "wind_sink_fraction", 0.0)),
        wind_sink_center_fraction=float(scalar_from_data(data, "wind_sink_center_fraction", 0.8)),
        wind_sink_log_width=float(scalar_from_data(data, "wind_sink_log_width", 0.08)),
        stream_heating_efficiency=float(scalar_from_data(data, "stream_heating_efficiency", 0.0)),
        outer_closure=str(scalar_from_data(data, "outer_closure", "pressure_supported_thin_energy")),
        outer_robin_chi=float(scalar_from_data(data, "outer_robin_chi", 0.0)),
        outer_robin_slope_target=float(scalar_from_data(data, "outer_robin_slope_target", 0.0)),
        outer_robin_slope_scale=float(scalar_from_data(data, "outer_robin_slope_scale", 1.0)),
    )
    return z, apply_outer_slopes_from_state(z, params)


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


def clip_state(z: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    lower, upper = state_bounds(params)
    return np.clip(np.asarray(z, dtype=float), lower + 1.0e-12, upper - 1.0e-12)


def finite_difference_source_column(anchor_z: np.ndarray, anchor_params: TransonicSlimParams, *, pivot: str) -> tuple[np.ndarray, float]:
    f0 = float(anchor_params.stream_source_fraction)
    step = min(abs(float(TANGENT_FD_STEP)), 0.25 * max(f0, 1.0e-3), 0.25 * max(1.0 - f0, 1.0e-3))
    if step <= 0.0:
        raise ValueError("source finite-difference step collapsed")
    if f0 - step >= 0.0 and f0 + step < 1.0 + anchor_params.wind_sink_fraction:
        plus = replace(anchor_params, stream_source_fraction=f0 + step, stream_mass_fraction=0.0)
        minus = replace(anchor_params, stream_source_fraction=f0 - step, stream_mass_fraction=0.0)
        f_plus = square_collocation_residual(anchor_z, plus, pivot=pivot)
        f_minus = square_collocation_residual(anchor_z, minus, pivot=pivot)
        return (f_plus - f_minus) / (2.0 * step), step
    plus = replace(anchor_params, stream_source_fraction=f0 + step, stream_mass_fraction=0.0)
    f_base = square_collocation_residual(anchor_z, anchor_params, pivot=pivot)
    f_plus = square_collocation_residual(anchor_z, plus, pivot=pivot)
    return (f_plus - f_base) / step, step


def equilibrated_tangent_solve(jac, rhs: np.ndarray) -> np.ndarray:
    try:
        from scipy.sparse import diags
        from scipy.sparse.linalg import lsmr, splu
    except Exception as exc:
        raise RuntimeError("scipy is required for source-fraction tangent prediction") from exc

    if TANGENT_SOLVER == "splu":
        return np.asarray(splu(jac.tocsc(), permc_spec="COLAMD").solve(rhs), dtype=float)
    if TANGENT_SOLVER == "lsmr":
        result = lsmr(
            jac.tocsr(),
            rhs,
            damp=TANGENT_LINEAR_DAMPING,
            atol=1.0e-10,
            btol=1.0e-10,
            maxiter=max(TANGENT_MAXITER, 5 * jac.shape[1]),
        )
        return np.asarray(result[0], dtype=float)
    if TANGENT_SOLVER not in {"equilibrated_lsmr", "equilibrated_direct"}:
        raise ValueError("unknown tangent solver")

    jac_csr = jac.tocsr()
    row_norm = np.sqrt(np.asarray(jac_csr.multiply(jac_csr).sum(axis=1)).ravel())
    row_scale = 1.0 / np.maximum(row_norm, 1.0e-12)
    row_scaled = diags(row_scale) @ jac_csr
    col_norm = np.sqrt(np.asarray(row_scaled.multiply(row_scaled).sum(axis=0)).ravel())
    col_scale = 1.0 / np.maximum(col_norm, 1.0e-12)
    balanced = (row_scaled @ diags(col_scale)).tocsc()
    scaled_rhs = row_scale * np.asarray(rhs, dtype=float)
    if TANGENT_SOLVER == "equilibrated_direct" and TANGENT_LINEAR_DAMPING == 0.0:
        try:
            y = splu(balanced, permc_spec="COLAMD").solve(scaled_rhs)
            return col_scale * np.asarray(y, dtype=float)
        except Exception:
            pass
    result = lsmr(
        balanced,
        scaled_rhs,
        damp=TANGENT_LINEAR_DAMPING,
        atol=1.0e-12,
        btol=1.0e-12,
        maxiter=max(TANGENT_MAXITER, 10 * balanced.shape[1]),
    )
    return col_scale * np.asarray(result[0], dtype=float)


def source_fraction_tangent(anchor_z: np.ndarray, anchor_params: TransonicSlimParams, *, pivot: str) -> np.ndarray:
    jac = square_collocation_jacobian(anchor_z, anchor_params, pivot=pivot)
    f_source, _fd_step = finite_difference_source_column(anchor_z, anchor_params, pivot=pivot)
    return equilibrated_tangent_solve(jac, -f_source)


def source_fraction_seed(
    *,
    target_fraction: float,
    current_fraction: float,
    current_z: np.ndarray,
    prev_fraction: float | None,
    prev_z: np.ndarray | None,
    params: TransonicSlimParams,
) -> tuple[np.ndarray, str, float]:
    current_seed = np.asarray(current_z, dtype=float)
    current_full = max_residual(current_seed, params)
    best_seed = current_seed
    best_label = "current"
    best_full = current_full
    if USE_SECANT_PREDICTOR and prev_z is not None and prev_fraction is not None and abs(current_fraction - prev_fraction) > 1.0e-12:
        step_factor = (float(target_fraction) - current_fraction) / (current_fraction - prev_fraction)
        for damping in SECANT_DAMPING_VALUES:
            trial_seed = clip_state(current_z + float(damping) * step_factor * (current_z - prev_z), params)
            trial_full = max_residual(trial_seed, params)
            if trial_full < best_full:
                best_seed = trial_seed
                best_label = f"secant:{float(damping):g}"
                best_full = trial_full
    if USE_TANGENT_PREDICTOR and abs(float(target_fraction) - current_fraction) > 1.0e-14:
        try:
            anchor_params = replace(params, stream_source_fraction=float(current_fraction), stream_mass_fraction=0.0)
            anchor_params = apply_outer_slopes_from_state(current_z, anchor_params)
            pivot = PIVOTS[0] if PIVOTS else "C2"
            dz_df = source_fraction_tangent(current_z, anchor_params, pivot=pivot)
            df = float(target_fraction) - current_fraction
            for damping in TANGENT_DAMPING_VALUES:
                trial_seed = clip_state(current_z + float(damping) * df * dz_df, params)
                trial_full = max_residual(trial_seed, params)
                if trial_full < best_full:
                    best_seed = trial_seed
                    best_label = f"tangent:{float(damping):g}"
                    best_full = trial_full
        except Exception as exc:
            print(f"  tangent predictor unavailable: {exc}", flush=True)
    return best_seed, best_label, best_full


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
    R_mass = float(params.stream_source_center_fraction * params.R_out)
    R_torque = float(params.stream_torque_center_fraction * params.R_out)
    mdot_inner, _dmdot_inner = stream_mass_rate_and_derivative(float(logR[0]), params)
    mdot_outer, dmdot_outer = stream_mass_rate_and_derivative(float(logR[-1]), params)
    mdot_center, dmdot_center = stream_mass_rate_and_derivative(float(np.log(R_mass)), params)
    source_prime = np.asarray([stream_source_prime(float(x), params) for x in logR], dtype=float)
    wind_prime = np.asarray([wind_sink_prime(float(x), params) for x in logR], dtype=float)
    budget_integral = float(np.trapezoid(wind_prime - source_prime, logR))
    budget_error = float((mdot_outer - mdot_inner) - budget_integral)
    budget_scale = max(abs(mdot_outer - mdot_inner), abs(budget_integral), abs(params.Mdot_g_s), 1.0)
    l_ref = float(params.potential.l_k(R_torque))
    stream_l_outer, _stream_l_outer_deriv = stream_torque_specific_l_and_derivative(float(logR[-1]), params)
    return {
        "Rinj_mass_rg": float(R_mass / params.r_g),
        "Rinj_torque_rg": float(R_torque / params.r_g),
        "Mdot_inner_over_param": float(mdot_inner / params.Mdot_g_s),
        "Mdot_outer_over_inner": float(mdot_outer / params.Mdot_g_s),
        "Mdot_center_over_inner": float(mdot_center / params.Mdot_g_s),
        "dMdot_dlnR_outer_over_inner": float(dmdot_outer / params.Mdot_g_s),
        "dMdot_dlnR_center_over_inner": float(dmdot_center / params.Mdot_g_s),
        "stream_source_integral_over_inner": float(np.trapezoid(source_prime, logR) / params.Mdot_g_s),
        "wind_sink_integral_over_inner": float(np.trapezoid(wind_prime, logR) / params.Mdot_g_s),
        "mass_budget_error_over_inner": float(budget_error / params.Mdot_g_s),
        "relative_mass_budget_error": float(abs(budget_error) / budget_scale),
        "stream_l_outer_over_lKinj": float(stream_l_outer / l_ref) if l_ref > 0.0 else np.nan,
    }


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
    branch: str,
    mass_fraction: float,
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
        "mass_fraction": float(mass_fraction),
        "torque_fraction": float(params.stream_torque_delta_l_fraction),
        "ratio": float(params.mdot_edd_ratio),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "grid_power": float(params.grid_power),
        "mass_center_fraction": float(params.stream_source_center_fraction),
        "mass_log_width": float(params.stream_source_log_width),
        "mass_source_shape": str(params.stream_source_shape),
        "mass_source_shape_blend": float(params.stream_source_shape_blend),
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
        **advection_diagnostic(z, params),
        **interval_peak_diagnostic(z, params),
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
    safe_mass = f"{float(row['mass_fraction']):.4g}".replace(".", "p").replace("-", "m")
    stem = f"{safe_branch}_mass_{safe_mass}_torque_{float(row['torque_fraction']):.4g}_mdot_{float(row['ratio']):.8g}_N{int(row['N'])}".replace(
        ".", "p"
    ).replace("-", "m")
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
        stream_mass_fraction=np.array(params.stream_mass_fraction),
        stream_mass_center_fraction=np.array(params.stream_mass_center_fraction),
        stream_mass_log_width=np.array(params.stream_mass_log_width),
        wind_sink_fraction=np.array(params.wind_sink_fraction),
        wind_sink_center_fraction=np.array(params.wind_sink_center_fraction),
        wind_sink_log_width=np.array(params.wind_sink_log_width),
        stream_heating_efficiency=np.array(params.stream_heating_efficiency),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted"]),
        branch=np.array(row["branch"]),
        row_json=np.array(json.dumps(json_safe(payload), sort_keys=True)),
    )


def write_table(rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Stream-Mass Annulus Scan",
        "",
        "Generated by `scripts/run_standard_slim_stream_mass_annulus_scan.py`.",
        "",
        f"Anchor `{ANCHOR_CHECKPOINT.relative_to(ROOT)}`, branches `{';'.join(BRANCH_SPECS)}`, "
        f"Rinj/Rout `{MASS_CENTER_FRACTION:g}`, log width `{MASS_LOG_WIDTH:g}`, source shape `{MASS_SOURCE_SHAPE}`, "
        f"source blend `{MASS_SOURCE_SHAPE_BLEND_OVERRIDE or 'checkpoint/default'}`, "
        f"torque fraction `{TORQUE_FRACTION:g}`, refresh repolish `{REFRESH_REPOLISH}`.",
        "",
        "| branch | source fraction | source shape | source blend | torque fraction | predictor | step | next step | cost action | Mdot outer/inner | Mdot center/inner | source integral | rel budget err | Rout/rg | Rinj/rg | initial full | final full | accepted | anchor | dominant | int R | int E | peak E R/rg | median abs E | outer omega | f_adv global | f_adv inner | f_adv pos | Lrad/LEdd | max H/R | int adv | Rson/rg | pivot | nfev | elapsed s | message |",
        "|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        display_row = {
            "predictor": "-",
            "attempt_step": np.nan,
            "next_step": np.nan,
            "cost_action": "-",
            **row,
        }
        formatted = {
            key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value
            for key, value in display_row.items()
        }
        for key in ("Mdot_outer_over_inner", "Mdot_center_over_inner"):
            formatted[key] = f"{float(display_row[key]):.6g}"
        lines.append(
            "| {branch} | {mass_fraction} | {mass_source_shape} | {mass_source_shape_blend} | {torque_fraction} | {predictor} | {attempt_step} | {next_step} | {cost_action} | "
            "{Mdot_outer_over_inner} | {Mdot_center_over_inner} | "
            "{stream_source_integral_over_inner} | {relative_mass_budget_error} | {R_out_rg} | {Rinj_mass_rg} | "
            "{initial_full} | {final_full} | {accepted} | {anchor_eligible} | "
            "{dominant} | {interval_R} | {interval_E} | {peak_interval_E_rg} | {median_abs_interval_E} | "
            "{outer_omega} | {f_adv_global} | {f_adv_inner} | {f_adv_pos} | {Lrad_LEdd} | {max_H_R} | "
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
    fractions = np.asarray([float(row["mass_fraction"]) for row in rows], dtype=float)
    residuals = np.log10(np.maximum(np.asarray([float(row["final_full"]) for row in rows], dtype=float), 1.0e-16))
    x_min, x_max = float(np.min(fractions)), float(np.max(fractions))
    if x_max <= x_min:
        x_min -= 0.5
        x_max += 0.5
    y_min, y_max = float(np.floor(np.min(residuals))), float(np.ceil(np.max(residuals)))
    if y_max <= y_min:
        y_max = y_min + 1.0
    for branch in sorted(set(str(row["branch"]) for row in rows)):
        selected = sorted([row for row in rows if row["branch"] == branch], key=lambda row: float(row["mass_fraction"]))
        points = []
        for row in selected:
            xx = float(row["mass_fraction"])
            yy = np.log10(max(float(row["final_full"]), 1.0e-16))
            px = x0 + int((xx - x_min) / (x_max - x_min) * (x1 - x0))
            py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
            points.append((px, py))
        color = (31, 119, 180)
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
    for tol, label in ((ACCEPTANCE_TOL, "accept"), (ANCHOR_TOL, "anchor")):
        yy = np.log10(tol)
        py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
        draw.line((x0, py, x1, py), fill=(120, 120, 120), width=1)
        draw.text((x1 - 72, py - 14), label, fill=(80, 80, 80), font=font)
    draw.text((90, 25), "Stream mass annulus: residual vs deposited mass fraction", fill=(20, 20, 20), font=font)
    draw.text((x0 + 4, y0 + 4), f"1e{int(y_max)}", fill=(80, 80, 80), font=font)
    draw.text((x0 + 4, y1 - 18), f"1e{int(y_min)}", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def run_branch(
    *,
    label: str,
    mass_fractions: list[float],
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    fiducial: FiducialParams,
    mdot_edd: float,
    rows: list[dict[str, Any]],
) -> None:
    current_z = np.asarray(anchor_z, dtype=float)
    current_params = anchor_params
    current_fraction = float(anchor_params.stream_source_fraction)
    prev_z: np.ndarray | None = None
    prev_fraction: float | None = None
    for mass_fraction in mass_fractions:
        params = params_for(
            fiducial,
            mdot_edd,
            ratio=current_params.mdot_edd_ratio,
            R_out_rg=current_params.R_out_rg,
            n_nodes=current_params.n_nodes,
            grid_power=current_params.grid_power,
            custom_grid_xi=current_params.custom_grid_xi,
            mass_fraction=float(mass_fraction),
            source_center_fraction=current_params.stream_source_center_fraction,
            source_log_width=current_params.stream_source_log_width,
            source_shape=current_params.stream_source_shape,
            source_shape_blend=current_params.stream_source_shape_blend,
            torque_fraction=current_params.stream_torque_delta_l_fraction,
            torque_center_fraction=current_params.stream_torque_center_fraction,
            torque_log_width=current_params.stream_torque_log_width,
            wind_sink_fraction=current_params.wind_sink_fraction,
            wind_sink_center_fraction=current_params.wind_sink_center_fraction,
            wind_sink_log_width=current_params.wind_sink_log_width,
            stream_heating_efficiency=current_params.stream_heating_efficiency,
            outer_closure=current_params.outer_closure,
            outer_robin_chi=current_params.outer_robin_chi,
            outer_robin_slope_target=current_params.outer_robin_slope_target,
            outer_robin_slope_scale=current_params.outer_robin_slope_scale,
        )
        params = apply_outer_slopes_from_state(current_z, params)
        seed, predictor, initial_full = source_fraction_seed(
            target_fraction=float(mass_fraction),
            current_fraction=current_fraction,
            current_z=current_z,
            prev_fraction=prev_fraction,
            prev_z=prev_z,
            params=params,
        )
        print(
            f"{label} mass_fraction={mass_fraction:g} predictor={predictor} "
            f"initial={initial_full:.3e}",
            flush=True,
        )
        t0 = time.perf_counter()
        polish = polish_best(seed, params)
        final_params = apply_outer_slopes_from_state(polish.z, params)
        if REFRESH_REPOLISH:
            polish = polish_best(polish.z, final_params)
            final_params = apply_outer_slopes_from_state(polish.z, final_params)
        elapsed = time.perf_counter() - t0
        row = row_for_result(
            branch=label,
            mass_fraction=float(mass_fraction),
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
            f"Mdot_outer/inner={row['Mdot_outer_over_inner']:.5g} accepted={row['accepted']} anchor={row['anchor_eligible']}",
            flush=True,
        )
        if row["accepted"]:
            prev_z = np.asarray(current_z, dtype=float)
            prev_fraction = current_fraction
            current_z = np.asarray(polish.z, dtype=float)
            current_params = final_params
            current_fraction = float(mass_fraction)
        else:
            print(f"  stopping branch {label} at first non-accepted mass fraction", flush=True)
            break


def run_adaptive_branch(
    *,
    label: str,
    target_fraction: float,
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    fiducial: FiducialParams,
    mdot_edd: float,
    rows: list[dict[str, Any]],
) -> None:
    current_z = np.asarray(anchor_z, dtype=float)
    current_params = anchor_params
    current_fraction = float(anchor_params.stream_source_fraction)
    prev_z: np.ndarray | None = None
    prev_fraction: float | None = None
    step = min(abs(float(ADAPTIVE_INITIAL_STEP)), abs(float(target_fraction) - current_fraction))
    direction = 1.0 if float(target_fraction) >= current_fraction else -1.0
    attempt = 0

    if step <= 0.0:
        return

    while direction * (float(target_fraction) - current_fraction) > 1.0e-12:
        remaining = direction * (float(target_fraction) - current_fraction)
        trial_step = min(step, remaining)
        mass_fraction = current_fraction + direction * trial_step
        params = params_for(
            fiducial,
            mdot_edd,
            ratio=current_params.mdot_edd_ratio,
            R_out_rg=current_params.R_out_rg,
            n_nodes=current_params.n_nodes,
            grid_power=current_params.grid_power,
            custom_grid_xi=current_params.custom_grid_xi,
            mass_fraction=float(mass_fraction),
            source_center_fraction=current_params.stream_source_center_fraction,
            source_log_width=current_params.stream_source_log_width,
            source_shape=current_params.stream_source_shape,
            source_shape_blend=current_params.stream_source_shape_blend,
            torque_fraction=current_params.stream_torque_delta_l_fraction,
            torque_center_fraction=current_params.stream_torque_center_fraction,
            torque_log_width=current_params.stream_torque_log_width,
            wind_sink_fraction=current_params.wind_sink_fraction,
            wind_sink_center_fraction=current_params.wind_sink_center_fraction,
            wind_sink_log_width=current_params.wind_sink_log_width,
            stream_heating_efficiency=current_params.stream_heating_efficiency,
            outer_closure=current_params.outer_closure,
            outer_robin_chi=current_params.outer_robin_chi,
            outer_robin_slope_target=current_params.outer_robin_slope_target,
            outer_robin_slope_scale=current_params.outer_robin_slope_scale,
        )
        params = apply_outer_slopes_from_state(current_z, params)
        seed, predictor, initial_full = source_fraction_seed(
            target_fraction=float(mass_fraction),
            current_fraction=current_fraction,
            current_z=current_z,
            prev_fraction=prev_fraction,
            prev_z=prev_z,
            params=params,
        )
        attempt += 1
        print(
            f"{label} attempt={attempt} current={current_fraction:.6g} target={mass_fraction:.6g} "
            f"step={direction * trial_step:.6g} predictor={predictor} initial={initial_full:.3e}",
            flush=True,
        )
        if initial_full > ADAPTIVE_MAX_INITIAL_FULL and trial_step > ADAPTIVE_MIN_STEP * (1.0 + 1.0e-12):
            step = max(ADAPTIVE_MIN_STEP, trial_step * ADAPTIVE_SHRINK)
            print(f"  pre-reject initial residual; reducing step to {step:.6g}", flush=True)
            continue

        t0 = time.perf_counter()
        polish = polish_best(seed, params)
        final_params = apply_outer_slopes_from_state(polish.z, params)
        if REFRESH_REPOLISH:
            polish = polish_best(polish.z, final_params)
            final_params = apply_outer_slopes_from_state(polish.z, final_params)
        elapsed = time.perf_counter() - t0
        row = row_for_result(
            branch=label,
            mass_fraction=float(mass_fraction),
            seed=seed,
            z=polish.z,
            params=final_params,
            polish=polish,
            elapsed_s=elapsed,
        )
        row["predictor"] = predictor
        row["attempt_step"] = float(direction * trial_step)
        row["cost_action"] = "pending"
        should_break = False
        if row["accepted"]:
            prev_z = np.asarray(current_z, dtype=float)
            prev_fraction = current_fraction
            current_z = np.asarray(polish.z, dtype=float)
            current_params = final_params
            current_fraction = float(mass_fraction)
            if row["nfev"] >= ADAPTIVE_COST_HARD_SHRINK_NFEV:
                step = max(ADAPTIVE_MIN_STEP, trial_step * ADAPTIVE_COST_HARD_SHRINK)
                row["cost_action"] = f"hard_shrink_nfev>={ADAPTIVE_COST_HARD_SHRINK_NFEV}"
            elif row["nfev"] >= ADAPTIVE_COST_SHRINK_NFEV:
                step = max(ADAPTIVE_MIN_STEP, trial_step * ADAPTIVE_COST_SHRINK)
                row["cost_action"] = f"shrink_nfev>={ADAPTIVE_COST_SHRINK_NFEV}"
            elif row["anchor_eligible"] and initial_full < 0.5 * ADAPTIVE_MAX_INITIAL_FULL and row["nfev"] <= ADAPTIVE_COST_GROW_NFEV:
                step = min(ADAPTIVE_MAX_STEP, max(ADAPTIVE_MIN_STEP, trial_step * ADAPTIVE_GROWTH))
                row["cost_action"] = f"grow_nfev<={ADAPTIVE_COST_GROW_NFEV}"
            else:
                step = max(ADAPTIVE_MIN_STEP, trial_step)
                row["cost_action"] = "hold"
        else:
            if trial_step <= ADAPTIVE_MIN_STEP * (1.0 + 1.0e-12):
                print(f"  stopping adaptive branch {label}: minimum step failed", flush=True)
                step = max(ADAPTIVE_MIN_STEP, trial_step)
                row["cost_action"] = "stop_min_step_failed"
                should_break = True
            else:
                step = max(ADAPTIVE_MIN_STEP, trial_step * ADAPTIVE_SHRINK)
                row["cost_action"] = "reject_shrink"
                print(f"  rejected; reducing step to {step:.6g}", flush=True)
        row["next_step"] = float(direction * step)
        rows.append(row)
        save_checkpoint(row, final_params)
        write_table(rows)
        write_figure(rows)
        print(
            f"  final={row['final_full']:.3e} dom={row['dominant']} "
            f"Mdot_outer/inner={row['Mdot_outer_over_inner']:.5g} accepted={row['accepted']} "
            f"anchor={row['anchor_eligible']} nfev={row['nfev']} next_step={direction * step:.6g} "
            f"action={row['cost_action']}",
            flush=True,
        )
        if should_break:
            break


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = load_anchor(ANCHOR_CHECKPOINT, fiducial, mdot_edd)
    anchor_z, anchor_params = prepare_anchor_grid(anchor_z, anchor_params, fiducial, mdot_edd)
    rows: list[dict[str, Any]] = []
    if ADAPTIVE_TARGET_RAW:
        branches = parse_branch_specs()
        label = branches[0][0] if branches else "adaptive"
        run_adaptive_branch(
            label=label,
            target_fraction=float(ADAPTIVE_TARGET_RAW),
            anchor_z=anchor_z,
            anchor_params=anchor_params,
            fiducial=fiducial,
            mdot_edd=mdot_edd,
            rows=rows,
        )
    else:
        for label, fractions in parse_branch_specs():
            run_branch(
                label=label,
                mass_fractions=fractions,
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
