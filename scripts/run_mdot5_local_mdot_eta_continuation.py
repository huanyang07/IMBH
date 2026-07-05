"""Staged launch-energy continuation for the local-Mdot Mdot=5 wind BVP."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mdot5_local_mdot_bvp_pilot as pilot  # noqa: E402
import run_standard_slim_stream_mass_annulus_scan as scan  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_state,
    entropy_gradient_log,
    state_partials,
    stream_heating_rate,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    transonic_profile_from_state_vector,
    wind_energy_loss_rate,
    wind_energy_per_mass,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _interval_geometry  # noqa: E402
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402


DEFAULT_ANCHOR = (
    ROOT
    / "outputs/checkpoints/m5_energy_wind_powerlaw_mass_coupled_adaptive_0p015_to_0p03/"
    "zeta_0p03_N896.npz"
)
ANCHOR = Path(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_ANCHOR", str(DEFAULT_ANCHOR))).expanduser()
if not ANCHOR.is_absolute():
    ANCHOR = ROOT / ANCHOR
START_X_CHECKPOINT_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_START_X_CHECKPOINT", "").strip()
START_X_CHECKPOINT = Path(START_X_CHECKPOINT_RAW).expanduser() if START_X_CHECKPOINT_RAW else None
if START_X_CHECKPOINT is not None and not START_X_CHECKPOINT.is_absolute():
    START_X_CHECKPOINT = ROOT / START_X_CHECKPOINT

OUTPUT_STEM = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTPUT_STEM", "m5_local_mdot_eta_continuation_zeta0p03_N96")
N_NODES = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_N_NODES", "96"))
ETA_VALUES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_VALUES", "100,60,40,33.3333333333").split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_MAX_NFEV", "220"))
RESIDUAL_TOL = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_TOL", "1e-7"))
MASS_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_MASS_WEIGHT", "1.0"))
INNER_MDOT_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_MDOT_WEIGHT", "1.0"))
REMAP_METHOD = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMAP_METHOD", "linear").strip().lower()
ACCEPT_TOL = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_ACCEPT_TOL", "1e-5"))
OUTER_BUFFER_WEIGHT_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_BUFFER_WEIGHT", "").strip()
OUTER_BUFFER_INNER_RG_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_BUFFER_INNER_RG", "").strip()
OUTER_CLOSURE_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_CLOSURE", "").strip()
OUTER_ROBIN_CHI_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_ROBIN_CHI", "").strip()
OUTER_ROBIN_SLOPE_TARGET_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_ROBIN_SLOPE_TARGET", "").strip()
OUTER_ROBIN_SLOPE_SCALE_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_ROBIN_SLOPE_SCALE", "").strip()
OUTER_OMEGA_LOG_OFFSET_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_OMEGA_LOG_OFFSET", "").strip()
INTERVAL_RESIDUAL_FORM_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INTERVAL_FORM", "").strip()
INTEGRATED_WEIGHTING_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INTEGRATED_WEIGHTING", "").strip()
OUTER_SLOPE_PICARD_ITERS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_SLOPE_PICARD_ITERS", "0"))
SEED_ONLY = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SEED_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
DEFECT_REMAP_SWEEPS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_DEFECT_REMAP_SWEEPS", "4"))
INNER_RELAX_OUTER_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_OUTER_RG", "0.0"))
INNER_RELAX_MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_MAX_NFEV", "80"))
INNER_RELAX_INCLUDE_MDOT = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_INCLUDE_MDOT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
INNER_RELAX_INCLUDE_GLOBALS = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_INCLUDE_GLOBALS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
INNER_RELAX_ANCHOR_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_ANCHOR_WEIGHT", "1e-4"))
OUTER_RELAX_MIN_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_MIN_RG", "0.0"))
OUTER_RELAX_MAX_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_MAX_RG", "0.0"))
OUTER_RELAX_MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_MAX_NFEV", "80"))
OUTER_RELAX_INCLUDE_ENERGY = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_INCLUDE_ENERGY", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OUTER_RELAX_INCLUDE_MDOT = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_INCLUDE_MDOT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OUTER_RELAX_INCLUDE_GLOBALS = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_INCLUDE_GLOBALS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OUTER_RELAX_ANCHOR_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_ANCHOR_WEIGHT", "1.0"))
NESTED_REFINE_MIN_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_NESTED_REFINE_MIN_RG", "0.0"))
NESTED_REFINE_MAX_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_NESTED_REFINE_MAX_RG", "inf"))
RESIDUAL_REMESH_STRENGTH = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_STRENGTH", "0.0"))
RESIDUAL_REMESH_BLEND = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_BLEND", "0.7"))
RESIDUAL_REMESH_POWER = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_POWER", "0.5"))
RESIDUAL_REMESH_SMOOTH_PASSES = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_SMOOTH_PASSES", "2"))
RESIDUAL_REMESH_FLOOR = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_FLOOR", "1.0"))
RESIDUAL_REMESH_DENSE_FACTOR = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_DENSE_FACTOR", "32"))
W_REMESH_INTERVAL_R = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_INTERVAL_R", "1.0"))
W_REMESH_INTERVAL_E = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_INTERVAL_E", "1.0"))
W_REMESH_MASS = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_MASS", "0.5"))
W_REMESH_SOURCE = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_SOURCE", "0.8"))
W_REMESH_WIND = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_WIND", "0.8"))
W_REMESH_MDOT_GRAD = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_MDOT_GRAD", "0.8"))
W_REMESH_OUTER = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_OUTER", "1.0"))
REMESH_OUTER_WIDTH = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_OUTER_WIDTH", "0.04"))

JSON_OUTPUT = ROOT / f"outputs/tables/{OUTPUT_STEM}.json"
MD_OUTPUT = ROOT / f"outputs/tables/{OUTPUT_STEM}.md"
PROFILE_OUTPUT = ROOT / f"outputs/tables/{OUTPUT_STEM}_profiles.json"
CHECKPOINT_DIR = ROOT / f"outputs/checkpoints/{OUTPUT_STEM}"


def _format(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "nan"
    if number == 0.0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e4:
        return f"{number:.3e}"
    return f"{number:.6g}"


def _safe_eta_label(value: float) -> str:
    return f"{float(value):.8g}".replace(".", "p").replace("-", "m")


def _cumtrapz(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(values, dtype=float))
    if out.size < 2:
        return out
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(np.asarray(x, dtype=float)))
    return out


def _normalize_component(values: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.abs(np.asarray(values, dtype=float)), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.max(clean)) if clean.size else 0.0
    if scale <= 0.0:
        return np.zeros_like(clean)
    return clean / scale


def _smooth_score(score: np.ndarray, passes: int) -> np.ndarray:
    smoothed = np.asarray(score, dtype=float)
    for _ in range(max(int(passes), 0)):
        if smoothed.size <= 2:
            break
        padded = np.pad(smoothed, (1, 1), mode="edge")
        smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return smoothed


def _enforce_min_spacing(xi: np.ndarray, min_spacing: float = 1.0e-10) -> np.ndarray:
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


def _set_eta(eta_E: float) -> None:
    pilot.WIND_ENERGY_MULTIPLIER = float(eta_E)
    pilot.MASS_WEIGHT = float(MASS_WEIGHT)


def _inner_mdot_row_index(params) -> int:
    return 2 * (int(params.n_nodes) - 1) + 2 + 2


def _residual(x: np.ndarray, params) -> np.ndarray:
    rows = np.asarray(pilot.residual(x, params), dtype=float).copy()
    if INNER_MDOT_WEIGHT != 1.0:
        rows[_inner_mdot_row_index(params)] *= float(INNER_MDOT_WEIGHT)
    return rows


def _apply_local_overrides(params):
    kwargs: dict[str, Any] = {}
    if OUTER_BUFFER_WEIGHT_RAW:
        weight = float(OUTER_BUFFER_WEIGHT_RAW)
        kwargs.update(
            outer_buffer_radial_weight=weight,
            outer_buffer_energy_weight=weight,
            outer_buffer_boundary_weight=weight,
        )
    if OUTER_BUFFER_INNER_RG_RAW:
        if OUTER_BUFFER_INNER_RG_RAW.lower() in {"none", "off", "null"}:
            kwargs["outer_buffer_inner_rg"] = None
        else:
            kwargs["outer_buffer_inner_rg"] = float(OUTER_BUFFER_INNER_RG_RAW)
    if OUTER_CLOSURE_RAW:
        kwargs["outer_closure"] = OUTER_CLOSURE_RAW
    if OUTER_ROBIN_CHI_RAW:
        kwargs["outer_robin_chi"] = float(OUTER_ROBIN_CHI_RAW)
    if OUTER_ROBIN_SLOPE_TARGET_RAW:
        kwargs["outer_robin_slope_target"] = float(OUTER_ROBIN_SLOPE_TARGET_RAW)
    if OUTER_ROBIN_SLOPE_SCALE_RAW:
        kwargs["outer_robin_slope_scale"] = float(OUTER_ROBIN_SLOPE_SCALE_RAW)
    if OUTER_OMEGA_LOG_OFFSET_RAW:
        kwargs["outer_omega_log_offset"] = float(OUTER_OMEGA_LOG_OFFSET_RAW)
    if INTERVAL_RESIDUAL_FORM_RAW:
        kwargs["interval_residual_form"] = INTERVAL_RESIDUAL_FORM_RAW
    if INTEGRATED_WEIGHTING_RAW:
        kwargs["integrated_residual_weighting"] = INTEGRATED_WEIGHTING_RAW
    return replace(params, **kwargs) if kwargs else params


def _restore_checkpoint_params(params, data) -> Any:
    kwargs: dict[str, Any] = {}
    if "custom_grid_xi" in data:
        custom_grid = np.asarray(data["custom_grid_xi"], dtype=float)
        if custom_grid.size == int(params.n_nodes):
            kwargs["custom_grid_xi"] = tuple(float(value) for value in custom_grid)
    if "outer_match_log_slopes" in data:
        slopes = np.asarray(data["outer_match_log_slopes"], dtype=float)
        if slopes.shape == (2,) and np.all(np.isfinite(slopes)):
            kwargs["outer_match_log_slopes"] = (float(slopes[0]), float(slopes[1]))
    return replace(params, **kwargs) if kwargs else params


def _state_and_params_for_n(anchor_z: np.ndarray, anchor_params, n_nodes: int) -> tuple[np.ndarray, Any]:
    params = anchor_params
    z = anchor_z
    if int(params.n_nodes) != int(n_nodes):
        target_params = replace(params, n_nodes=int(n_nodes), custom_grid_xi=None)
        profile = transonic_profile_from_state_vector(z, params)
        state_remap_method = (
            "pchip"
            if REMAP_METHOD
            in {
                "mass_ode",
                "defect_preserving",
                "pchip_mass_ode",
                "nested_mass_ode",
                "nested_defect_preserving",
            }
            else REMAP_METHOD
        )
        z = scan.remap_profile_to_new_sonic_grid(
            profile,
            target_params,
            temperature_mdot_power=0.0,
            method=state_remap_method,
        )
        params = scan.apply_outer_slopes_from_state(z, target_params)
    local_params = replace(params, wind_sink_fraction=0.0, mdot_profile_mode="source_sink")
    return z, _apply_local_overrides(local_params)


def _make_seed(anchor_z: np.ndarray, anchor_params) -> tuple[np.ndarray, Any]:
    z, local_params = _state_and_params_for_n(anchor_z, anchor_params, N_NODES)
    logu, logT, logR_son, lambda0, logR = scan.unpack_state(z, local_params)
    mdot_seed = np.asarray([stream_mass_rate_and_derivative(float(x), local_params)[0] for x in logR], dtype=float)
    x0 = pilot._pack(logu, logT, np.log(mdot_seed), logR_son, lambda0)
    lower, upper = pilot._bounds(local_params)
    return np.clip(x0, lower + 1.0e-12, upper - 1.0e-12), local_params


def _remap_local_x_to_params(x_old: np.ndarray, old_params, new_params) -> np.ndarray:
    logu_old, logT_old, logMdot_old, logR_son, lambda0, logR_old = pilot._unpack(x_old, old_params)
    logR_new = pilot.computational_grid(new_params, logR_son)
    def interp(values: np.ndarray) -> np.ndarray:
        if REMAP_METHOD in {
            "pchip",
            "monotone",
            "shape_preserving",
            "mass_ode",
            "defect_preserving",
            "pchip_mass_ode",
            "nested_mass_ode",
            "nested_defect_preserving",
        }:
            try:
                from scipy.interpolate import PchipInterpolator

                return np.asarray(PchipInterpolator(logR_old, values, extrapolate=True)(logR_new), dtype=float)
            except Exception:
                return np.interp(logR_new, logR_old, values)
        return np.interp(logR_new, logR_old, values)

    logu_new = interp(logu_old)
    logT_new = interp(logT_old)
    logMdot_new = interp(logMdot_old)
    if REMAP_METHOD in {"mass_ode", "pchip_mass_ode", "nested_mass_ode"}:
        logMdot_new = _mdot_ode_logmdot_seed(logu_new, logT_new, logMdot_new, logR_new, lambda0, new_params)
    elif REMAP_METHOD in {"log_mass_ode", "log_mdot_ode"}:
        logMdot_new = _mass_ode_logmdot_seed(logu_new, logT_new, logMdot_new, logR_new, lambda0, new_params)
    elif REMAP_METHOD in {"defect_preserving", "mass_defect", "defect_preserving_mass", "nested_defect_preserving"}:
        logMdot_new = _defect_preserving_logmdot_seed(
            x_old,
            old_params,
            logu_new,
            logT_new,
            logMdot_new,
            logR_new,
            lambda0,
            new_params,
        )
    x_new = pilot._pack(logu_new, logT_new, logMdot_new, logR_son, lambda0)
    lower, upper = pilot._bounds(new_params)
    return np.clip(x_new, lower + 1.0e-12, upper - 1.0e-12)


def _node_preserving_refined_params(x_old: np.ndarray, old_params, target_params):
    old_n = int(old_params.n_nodes)
    target_n = int(target_params.n_nodes)
    if target_n <= old_n:
        return target_params
    _logu_old, _logT_old, _logMdot_old, logR_son, _lambda0, logR_old = pilot._unpack(x_old, old_params)
    span = max(float(np.log(old_params.R_out) - logR_son), 1.0e-300)
    xi_values = [float(value) for value in (logR_old - logR_son) / span]
    xi_values[0] = 0.0
    xi_values[-1] = 1.0
    while len(xi_values) < target_n:
        xi_array = np.asarray(xi_values, dtype=float)
        gaps = np.diff(xi_array)
        mid_xi = 0.5 * (xi_array[:-1] + xi_array[1:])
        mid_R_rg = np.exp(float(logR_son) + mid_xi * span) / old_params.r_g
        allowed = (mid_R_rg >= float(NESTED_REFINE_MIN_RG)) & (mid_R_rg <= float(NESTED_REFINE_MAX_RG))
        if np.any(allowed):
            scores = np.where(allowed, gaps, -np.inf)
            idx = int(np.argmax(scores))
        else:
            idx = int(np.argmax(gaps))
        xi_values.insert(idx + 1, 0.5 * (xi_values[idx] + xi_values[idx + 1]))
    xi = _enforce_min_spacing(np.asarray(xi_values[:target_n], dtype=float))
    return replace(target_params, custom_grid_xi=tuple(float(value) for value in xi))


def _mdot_ode_logmdot_seed(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot_guess: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    params,
) -> np.ndarray:
    local_params = pilot._local_params(params, logR, logMdot_guess)
    mdot_nodes = np.empty_like(logMdot_guess)
    mdot_nodes[0] = float(params.Mdot_g_s)
    for idx in range(len(logR) - 1):
        dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
        ym = 0.5 * (y_left + y_right)
        gm = (y_right - y_left) / dx
        wind_prime = pilot._wind_mass_prime(xm, ym, gm, lambda0, local_params)
        source_prime = stream_source_prime(xm, local_params)
        mdot_nodes[idx + 1] = mdot_nodes[idx] + float(wind_prime - source_prime) * dx
        mdot_nodes[idx + 1] = max(mdot_nodes[idx + 1], 1.0e-6 * float(params.Mdot_g_s))
    return np.log(mdot_nodes)


def _mass_target_profile(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    params,
) -> np.ndarray:
    local_params = pilot._local_params(params, logR, logMdot)
    target = np.empty(len(logR) - 1, dtype=float)
    for idx in range(len(logR) - 1):
        dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
        ym = 0.5 * (y_left + y_right)
        gm = (y_right - y_left) / dx
        logMdot_mid = 0.5 * (logMdot[idx] + logMdot[idx + 1])
        mdot_mid = float(np.exp(logMdot_mid))
        wind_prime = pilot._wind_mass_prime(xm, ym, gm, lambda0, local_params)
        source_prime = stream_source_prime(xm, local_params)
        target[idx] = float((wind_prime - source_prime) / mdot_mid)
    return target


def _mass_ode_logmdot_seed(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot_guess: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    params,
) -> np.ndarray:
    logMdot = np.asarray(logMdot_guess, dtype=float).copy()
    logMdot[0] = float(np.log(params.Mdot_g_s))
    for _ in range(max(1, DEFECT_REMAP_SWEEPS)):
        target = _mass_target_profile(logu, logT, logMdot, logR, lambda0, params)
        next_logMdot = np.empty_like(logMdot)
        next_logMdot[0] = float(np.log(params.Mdot_g_s))
        for idx, dx in enumerate(np.diff(logR)):
            next_logMdot[idx + 1] = next_logMdot[idx] + float(target[idx]) * float(dx)
        logMdot = next_logMdot
    return logMdot


def _defect_preserving_logmdot_seed(
    x_old: np.ndarray,
    old_params,
    logu_new: np.ndarray,
    logT_new: np.ndarray,
    logMdot_guess: np.ndarray,
    logR_new: np.ndarray,
    lambda0: float,
    new_params,
) -> np.ndarray:
    logu_old, logT_old, logMdot_old, _logR_son_old, _lambda0_old, logR_old = pilot._unpack(x_old, old_params)
    old_residual = np.asarray(pilot.residual(x_old, old_params), dtype=float)
    old_n = int(old_params.n_nodes)
    old_mass_start = _inner_mdot_row_index(old_params) + 1
    old_mass_defect = old_residual[old_mass_start : old_mass_start + old_n - 1] / max(float(MASS_WEIGHT), 1.0e-300)
    old_mid = 0.5 * (logR_old[:-1] + logR_old[1:])
    new_mid = 0.5 * (logR_new[:-1] + logR_new[1:])
    defect = np.interp(new_mid, old_mid, old_mass_defect, left=old_mass_defect[0], right=old_mass_defect[-1])
    logMdot_outer_target = float(logMdot_old[-1])

    logMdot = np.asarray(logMdot_guess, dtype=float).copy()
    logMdot[0] = float(np.log(new_params.Mdot_g_s))
    for _ in range(max(1, DEFECT_REMAP_SWEEPS)):
        target = _mass_target_profile(logu_new, logT_new, logMdot, logR_new, lambda0, new_params) + defect
        span = max(float(logR_new[-1] - logR_new[0]), 1.0e-300)
        predicted_outer = float(logMdot[0] + np.sum(target * np.diff(logR_new)))
        budget_correction = (logMdot_outer_target - predicted_outer) / span
        target = target + budget_correction
        next_logMdot = np.empty_like(logMdot)
        next_logMdot[0] = float(np.log(new_params.Mdot_g_s))
        for idx, dx in enumerate(np.diff(logR_new)):
            next_logMdot[idx + 1] = next_logMdot[idx] + float(target[idx]) * float(dx)
        logMdot = next_logMdot
    return logMdot


def _residual_remesh(
    x: np.ndarray,
    params,
    eta_E: float,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    if RESIDUAL_REMESH_STRENGTH <= 0.0:
        return x, params, {}

    source_profile = _profile("residual_remesh_source", x, params, eta_E)
    logu, logT, logMdot, logR_son, _lambda0, logR = pilot._unpack(x, params)
    span = max(float(logR[-1] - logR[0]), 1.0e-12)
    source_xi = (np.asarray(logR, dtype=float) - float(logR[0])) / span
    interval_mid_xi = 0.5 * (source_xi[:-1] + source_xi[1:])
    interval_mid_R_rg = np.asarray(source_profile["R_mid_rg"], dtype=float)
    dense_count = max(4096, int(RESIDUAL_REMESH_DENSE_FACTOR) * int(params.n_nodes))
    dense_xi = np.linspace(0.0, 1.0, dense_count)

    def dense_from_interval(values: np.ndarray) -> np.ndarray:
        arr = _normalize_component(np.asarray(values, dtype=float))
        if arr.size == 0:
            return np.zeros_like(dense_xi)
        return np.interp(dense_xi, interval_mid_xi, arr, left=arr[0], right=arr[-1])

    outer_width = max(float(REMESH_OUTER_WIDTH), 1.0e-5)
    outer_dense = np.exp(-0.5 * ((dense_xi - 1.0) / outer_width) ** 2)
    composite = (
        W_REMESH_INTERVAL_R * dense_from_interval(np.asarray(source_profile["interval_R"], dtype=float))
        + W_REMESH_INTERVAL_E * dense_from_interval(np.asarray(source_profile["interval_E"], dtype=float))
        + W_REMESH_MASS * dense_from_interval(np.asarray(source_profile["local_mass_residual"], dtype=float))
        + W_REMESH_SOURCE * dense_from_interval(np.asarray(source_profile["Mstream_prime_over_Mdot"], dtype=float))
        + W_REMESH_WIND * dense_from_interval(np.asarray(source_profile["Mwind_prime_over_Mdot"], dtype=float))
        + W_REMESH_MDOT_GRAD * dense_from_interval(np.asarray(source_profile["dlogMdot_dlogR"], dtype=float))
        + W_REMESH_OUTER * _normalize_component(outer_dense)
    )
    composite = _smooth_score(composite, RESIDUAL_REMESH_SMOOTH_PASSES)
    monitor = RESIDUAL_REMESH_FLOOR + float(RESIDUAL_REMESH_STRENGTH) * _normalize_component(composite) ** float(
        RESIDUAL_REMESH_POWER
    )
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (monitor[:-1] + monitor[1:]) * np.diff(dense_xi))])
    cumulative /= cumulative[-1]
    target = np.linspace(0.0, 1.0, int(params.n_nodes))
    adapted = np.interp(target, cumulative, dense_xi)
    reference = np.interp(target, np.linspace(0.0, 1.0, source_xi.size), source_xi)
    blended = _enforce_min_spacing((1.0 - float(RESIDUAL_REMESH_BLEND)) * reference + float(RESIDUAL_REMESH_BLEND) * adapted)
    remeshed_params = replace(params, custom_grid_xi=tuple(float(value) for value in blended))
    remeshed_x = _remap_local_x_to_params(x, params, remeshed_params)
    remeshed_params = scan.apply_outer_slopes_from_state(_z_from_x(remeshed_x, remeshed_params), remeshed_params)

    initial_full = float(np.linalg.norm(pilot.residual(x, params), ord=np.inf))
    remeshed_full = float(np.linalg.norm(pilot.residual(remeshed_x, remeshed_params), ord=np.inf))
    peak_monitor = int(np.argmax(monitor))
    peak_R_rg = float(interval_mid_R_rg[int(np.argmax(np.abs(source_profile["interval_R"])))]) if interval_mid_R_rg.size else math.nan
    info = {
        "residual_remesh_strength": float(RESIDUAL_REMESH_STRENGTH),
        "residual_remesh_blend": float(RESIDUAL_REMESH_BLEND),
        "residual_remesh_power": float(RESIDUAL_REMESH_POWER),
        "residual_remesh_initial_full": initial_full,
        "residual_remesh_seed_full": remeshed_full,
        "residual_remesh_peak_monitor_rg": float(np.exp(float(logR[0]) + dense_xi[peak_monitor] * span) / params.r_g),
        "residual_remesh_peak_interval_R_rg": peak_R_rg,
        "residual_remesh_outer_1pct_nodes": int(np.count_nonzero(blended >= 0.99)),
        "residual_remesh_outer_5pct_nodes": int(np.count_nonzero(blended >= 0.95)),
        "residual_remesh_source_dx_outer": float(source_xi[-1] - source_xi[-2]),
        "residual_remesh_target_dx_outer": float(blended[-1] - blended[-2]),
        "residual_remesh_source_min_dxi": float(np.min(np.diff(source_xi))),
        "residual_remesh_target_min_dxi": float(np.min(np.diff(blended))),
    }
    return remeshed_x, remeshed_params, info



def _jac_norms(result, params) -> dict[str, Any]:
    jac = getattr(result, "jac", None)
    if jac is None:
        return {}
    if hasattr(jac, "multiply"):
        row_norm = np.sqrt(np.asarray(jac.multiply(jac).sum(axis=1)).ravel())
        col_norm = np.sqrt(np.asarray(jac.multiply(jac).sum(axis=0)).ravel())
    else:
        array = np.asarray(jac, dtype=float)
        row_norm = np.linalg.norm(array, axis=1)
        col_norm = np.linalg.norm(array, axis=0)
    n = int(params.n_nodes)
    mass_start = 2 * (n - 1) + 2 + 2 + 1
    out: dict[str, Any] = {
        "jac_row_norm_min": float(np.nanmin(row_norm)) if row_norm.size else math.nan,
        "jac_row_norm_median": float(np.nanmedian(row_norm)) if row_norm.size else math.nan,
        "jac_row_norm_max": float(np.nanmax(row_norm)) if row_norm.size else math.nan,
        "jac_col_norm_min": float(np.nanmin(col_norm)) if col_norm.size else math.nan,
        "jac_col_norm_median": float(np.nanmedian(col_norm)) if col_norm.size else math.nan,
        "jac_col_norm_max": float(np.nanmax(col_norm)) if col_norm.size else math.nan,
        "jac_row_norm_interval_R": row_norm[0 : 2 * (n - 1) : 2].tolist(),
        "jac_row_norm_interval_E": row_norm[1 : 2 * (n - 1) : 2].tolist(),
        "jac_row_norm_mass": row_norm[mass_start : mass_start + n - 1].tolist(),
        "jac_col_norm_logu": col_norm[:n].tolist(),
        "jac_col_norm_logT": col_norm[n : 2 * n].tolist(),
        "jac_col_norm_logMdot": col_norm[2 * n : 3 * n].tolist(),
    }
    if out["jac_row_norm_mass"]:
        mass_norm = np.asarray(out["jac_row_norm_mass"], dtype=float)
        out["jac_row_norm_mass_median"] = float(np.nanmedian(mass_norm))
        out["jac_row_norm_mass_max"] = float(np.nanmax(mass_norm))
    return out


def _profile(label: str, x: np.ndarray, params, eta_E: float, jac_norms: dict[str, Any] | None = None) -> dict[str, Any]:
    _set_eta(eta_E)
    logu, logT, logMdot, logR_son, lambda0, logR = pilot._unpack(x, params)
    local_params = pilot._local_params(params, logR, logMdot)
    residual = pilot.residual(x, params)
    n = int(params.n_nodes)
    mass_start = 2 * (n - 1) + 2 + 2 + 1
    inner_mdot_residual = float(residual[_inner_mdot_row_index(params)])
    mass_rows = np.asarray(residual[mass_start : mass_start + n - 1], dtype=float)
    interval_R = np.asarray(residual[0 : 2 * (n - 1) : 2], dtype=float)
    interval_E = np.asarray(residual[1 : 2 * (n - 1) : 2], dtype=float)

    R_mid: list[float] = []
    Qwind: list[float] = []
    Qvisc: list[float] = []
    Qadv: list[float] = []
    H_over_R: list[float] = []
    wind_prime: list[float] = []
    source_prime: list[float] = []
    dlogMdot_dx: list[float] = []
    mass_target: list[float] = []
    Mdot_mid: list[float] = []

    for idx in range(n - 1):
        dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
        ym = 0.5 * (y_left + y_right)
        gm = (y_right - y_left) / dx
        logMdot_mid = 0.5 * (logMdot[idx] + logMdot[idx + 1])
        mdot_mid = float(np.exp(logMdot_mid))
        state = algebraic_state(xm, float(ym[0]), float(ym[1]), lambda0, local_params)
        partials = state_partials(xm, ym, lambda0, local_params, eps_x=local_params.partial_eps, eps_y=local_params.partial_eps)
        dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], gm))
        Tdsdx = entropy_gradient_log(xm, ym, gm, lambda0, local_params)
        q_visc = -state.W * dOmega_dx
        q_adv = -(state.Sigma * state.u / state.R) * Tdsdx
        q_stream = stream_heating_rate(xm, local_params)
        q_wind = wind_energy_loss_rate(state, q_visc, q_stream, q_adv, local_params)
        E_w = float(eta_E * wind_energy_per_mass(local_params.M2_g, state.R))
        wprime = float(2.0 * np.pi * state.R**2 * q_wind / max(E_w, 1.0e-300))
        sprime = float(stream_source_prime(xm, local_params))
        target = float((wprime - sprime) / mdot_mid)

        R_mid.append(float(state.R))
        Qwind.append(float(q_wind))
        Qvisc.append(float(q_visc))
        Qadv.append(float(q_adv))
        H_over_R.append(float(state.H_over_R))
        wind_prime.append(wprime)
        source_prime.append(sprime)
        dlogMdot_dx.append(float((logMdot[idx + 1] - logMdot[idx]) / dx))
        mass_target.append(target)
        Mdot_mid.append(mdot_mid)

    R = np.asarray(R_mid, dtype=float)
    logR_mid = np.log(R)
    source = np.asarray(source_prime, dtype=float)
    wind = np.asarray(wind_prime, dtype=float)
    mdot = np.asarray(Mdot_mid, dtype=float)
    mdot_tilde = mdot + _cumtrapz(source, logR_mid)
    s_eff_tilde = wind / np.maximum(mdot_tilde, 1.0e-300)
    raw_mass = mass_rows / max(float(MASS_WEIGHT), 1.0e-300)
    peak_mass_idx = int(np.argmax(np.abs(raw_mass))) if raw_mass.size else 0
    peak_E_idx = int(np.argmax(np.abs(interval_E))) if interval_E.size else 0

    row: dict[str, Any] = {
        "label": label,
        "eta_E": float(eta_E),
        "N": n,
        "R_mid_rg": (R / local_params.r_g).tolist(),
        "interval_R": interval_R.tolist(),
        "interval_E": interval_E.tolist(),
        "local_mass_residual": raw_mass.tolist(),
        "local_mass_residual_weighted": mass_rows.tolist(),
        "Qwind_Qvisc": (np.asarray(Qwind) / np.maximum(np.abs(np.asarray(Qvisc)), 1.0e-300)).tolist(),
        "Qadv_Qvisc": (np.asarray(Qadv) / np.maximum(np.abs(np.asarray(Qvisc)), 1.0e-300)).tolist(),
        "Mwind_prime_over_Mdot": (wind / np.maximum(mdot, 1.0e-300)).tolist(),
        "Mstream_prime_over_Mdot": (source / np.maximum(mdot, 1.0e-300)).tolist(),
        "dlogMdot_dlogR": np.asarray(dlogMdot_dx, dtype=float).tolist(),
        "mass_target": np.asarray(mass_target, dtype=float).tolist(),
        "Mdot_over_inner": (mdot / local_params.Mdot_g_s).tolist(),
        "Mdot_tilde_over_inner": (mdot_tilde / local_params.Mdot_g_s).tolist(),
        "s_eff_tilde": s_eff_tilde.tolist(),
        "H_over_R": np.asarray(H_over_R, dtype=float).tolist(),
        "peak_mass_residual_rg": float(R[peak_mass_idx] / local_params.r_g) if raw_mass.size else math.nan,
        "peak_mass_residual": float(raw_mass[peak_mass_idx]) if raw_mass.size else math.nan,
        "inner_logMdot_residual": inner_mdot_residual,
        "peak_interval_E_rg": float(R[peak_E_idx] / local_params.r_g) if interval_E.size else math.nan,
        "peak_interval_E": float(interval_E[peak_E_idx]) if interval_E.size else math.nan,
        "mass_residual_p90_abs": float(np.quantile(np.abs(raw_mass), 0.90)) if raw_mass.size else math.nan,
        "s_eff_tilde_p50": float(np.nanmedian(s_eff_tilde)) if s_eff_tilde.size else math.nan,
        "s_eff_tilde_p90": float(np.nanquantile(s_eff_tilde, 0.90)) if s_eff_tilde.size else math.nan,
    }
    if jac_norms:
        row.update(jac_norms)
    return row


def _solve_stage(x0: np.ndarray, params, eta_E: float):
    _set_eta(eta_E)
    lower, upper = pilot._bounds(params)
    x0 = np.clip(x0, lower + 1.0e-12, upper - 1.0e-12)
    from scipy.optimize import least_squares

    return least_squares(
        lambda trial: _residual(trial, params),
        x0,
        bounds=(lower, upper),
        jac_sparsity=pilot._sparsity(params),
        x_scale="jac",
        loss="linear",
        ftol=RESIDUAL_TOL,
        xtol=RESIDUAL_TOL,
        gtol=RESIDUAL_TOL,
        max_nfev=MAX_NFEV,
        verbose=0,
    )


def _inner_window_relax(x0: np.ndarray, params, eta_E: float) -> tuple[np.ndarray, dict[str, Any]]:
    if INNER_RELAX_OUTER_RG <= 0.0:
        return x0, {}
    _set_eta(eta_E)
    logu, logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x0, params)
    n = int(params.n_nodes)
    R_rg = np.exp(logR) / params.r_g
    interval_mid_R_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    interval_mask = interval_mid_R_rg <= float(INNER_RELAX_OUTER_RG)
    if not np.any(interval_mask):
        return x0, {"inner_relax_enabled": True, "inner_relax_applied": False}

    last_interval = int(np.max(np.nonzero(interval_mask)[0]))
    last_node = min(n - 1, last_interval + 1)
    node_indices = np.arange(last_node + 1, dtype=int)
    variable_cols: list[int] = []
    variable_cols.extend(int(idx) for idx in node_indices)
    variable_cols.extend(int(n + idx) for idx in node_indices)
    if INNER_RELAX_INCLUDE_MDOT:
        variable_cols.extend(int(2 * n + idx) for idx in node_indices)
    if INNER_RELAX_INCLUDE_GLOBALS:
        variable_cols.extend([3 * n, 3 * n + 1])
    variable_cols_array = np.asarray(sorted(set(variable_cols)), dtype=int)

    interval_rows: list[int] = []
    for idx in range(last_interval + 1):
        interval_rows.extend([2 * idx, 2 * idx + 1])
    sonic_start = 2 * (n - 1) + 2
    inner_mdot_row = _inner_mdot_row_index(params)
    mass_start = inner_mdot_row + 1
    row_indices: list[int] = interval_rows + [sonic_start, sonic_start + 1]
    if INNER_RELAX_INCLUDE_MDOT:
        row_indices.append(inner_mdot_row)
        row_indices.extend(mass_start + idx for idx in range(last_interval + 1))
    row_indices_array = np.asarray(sorted(set(row_indices)), dtype=int)

    lower, upper = pilot._bounds(params)
    x_ref = np.asarray(x0, dtype=float)
    start = x_ref[variable_cols_array].copy()
    lb = lower[variable_cols_array]
    ub = upper[variable_cols_array]
    initial_residual = _residual(x_ref, params)
    initial_selected = float(np.linalg.norm(initial_residual[row_indices_array], ord=np.inf))
    initial_full = float(np.linalg.norm(initial_residual, ord=np.inf))

    def local_residual(trial: np.ndarray) -> np.ndarray:
        full = x_ref.copy()
        full[variable_cols_array] = trial
        rows = _residual(full, params)[row_indices_array]
        if INNER_RELAX_ANCHOR_WEIGHT > 0.0:
            rows = np.concatenate([rows, float(INNER_RELAX_ANCHOR_WEIGHT) * (trial - start)])
        return rows

    from scipy.optimize import least_squares

    result = least_squares(
        local_residual,
        np.clip(start, lb + 1.0e-12, ub - 1.0e-12),
        bounds=(lb, ub),
        x_scale="jac",
        loss="linear",
        ftol=RESIDUAL_TOL,
        xtol=RESIDUAL_TOL,
        gtol=RESIDUAL_TOL,
        max_nfev=INNER_RELAX_MAX_NFEV,
        verbose=0,
    )
    relaxed = x_ref.copy()
    relaxed[variable_cols_array] = result.x
    final_residual = _residual(relaxed, params)
    final_selected = float(np.linalg.norm(final_residual[row_indices_array], ord=np.inf))
    final_full = float(np.linalg.norm(final_residual, ord=np.inf))
    info = {
        "inner_relax_enabled": True,
        "inner_relax_applied": True,
        "inner_relax_outer_rg": float(INNER_RELAX_OUTER_RG),
        "inner_relax_last_node": int(last_node),
        "inner_relax_last_node_rg": float(R_rg[last_node]),
        "inner_relax_last_interval_rg": float(interval_mid_R_rg[last_interval]),
        "inner_relax_n_variables": int(variable_cols_array.size),
        "inner_relax_n_rows": int(row_indices_array.size),
        "inner_relax_initial_selected": initial_selected,
        "inner_relax_final_selected": final_selected,
        "inner_relax_initial_full": initial_full,
        "inner_relax_final_full": final_full,
        "inner_relax_nfev": int(result.nfev),
        "inner_relax_success": bool(result.success),
        "inner_relax_message": str(result.message),
    }
    return relaxed, info


def _outer_band_relax(x0: np.ndarray, params, eta_E: float) -> tuple[np.ndarray, dict[str, Any]]:
    if OUTER_RELAX_MIN_RG <= 0.0 or OUTER_RELAX_MAX_RG <= OUTER_RELAX_MIN_RG:
        return x0, {}
    _set_eta(eta_E)
    _logu, _logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x0, params)
    n = int(params.n_nodes)
    R_rg = np.exp(logR) / params.r_g
    interval_mid_R_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    interval_mask = (interval_mid_R_rg >= float(OUTER_RELAX_MIN_RG)) & (
        interval_mid_R_rg <= float(OUTER_RELAX_MAX_RG)
    )
    if not np.any(interval_mask):
        return x0, {"outer_relax_enabled": True, "outer_relax_applied": False}

    interval_indices = np.nonzero(interval_mask)[0].astype(int)
    node_indices = np.unique(np.concatenate([interval_indices, interval_indices + 1])).astype(int)
    variable_cols: list[int] = []
    variable_cols.extend(int(idx) for idx in node_indices)
    variable_cols.extend(int(n + idx) for idx in node_indices)
    if OUTER_RELAX_INCLUDE_MDOT:
        variable_cols.extend(int(2 * n + idx) for idx in node_indices)
    if OUTER_RELAX_INCLUDE_GLOBALS:
        variable_cols.extend([3 * n, 3 * n + 1])
    variable_cols_array = np.asarray(sorted(set(variable_cols)), dtype=int)

    row_indices: list[int] = []
    for idx in interval_indices:
        row_indices.append(2 * int(idx))
        if OUTER_RELAX_INCLUDE_ENERGY:
            row_indices.append(2 * int(idx) + 1)
    if OUTER_RELAX_INCLUDE_MDOT:
        mass_start = _inner_mdot_row_index(params) + 1
        row_indices.extend(mass_start + int(idx) for idx in interval_indices)
    row_indices_array = np.asarray(sorted(set(row_indices)), dtype=int)

    lower, upper = pilot._bounds(params)
    x_ref = np.asarray(x0, dtype=float)
    start = x_ref[variable_cols_array].copy()
    lb = lower[variable_cols_array]
    ub = upper[variable_cols_array]
    initial_residual = _residual(x_ref, params)
    initial_selected = float(np.linalg.norm(initial_residual[row_indices_array], ord=np.inf))
    initial_full = float(np.linalg.norm(initial_residual, ord=np.inf))

    def local_residual(trial: np.ndarray) -> np.ndarray:
        full = x_ref.copy()
        full[variable_cols_array] = trial
        rows = _residual(full, params)[row_indices_array]
        if OUTER_RELAX_ANCHOR_WEIGHT > 0.0:
            rows = np.concatenate([rows, float(OUTER_RELAX_ANCHOR_WEIGHT) * (trial - start)])
        return rows

    from scipy.optimize import least_squares

    result = least_squares(
        local_residual,
        np.clip(start, lb + 1.0e-12, ub - 1.0e-12),
        bounds=(lb, ub),
        x_scale="jac",
        loss="linear",
        ftol=RESIDUAL_TOL,
        xtol=RESIDUAL_TOL,
        gtol=RESIDUAL_TOL,
        max_nfev=OUTER_RELAX_MAX_NFEV,
        verbose=0,
    )
    relaxed = x_ref.copy()
    relaxed[variable_cols_array] = result.x
    final_residual = _residual(relaxed, params)
    final_selected = float(np.linalg.norm(final_residual[row_indices_array], ord=np.inf))
    final_full = float(np.linalg.norm(final_residual, ord=np.inf))
    info = {
        "outer_relax_enabled": True,
        "outer_relax_applied": True,
        "outer_relax_min_rg": float(OUTER_RELAX_MIN_RG),
        "outer_relax_max_rg": float(OUTER_RELAX_MAX_RG),
        "outer_relax_first_interval_rg": float(interval_mid_R_rg[interval_indices[0]]),
        "outer_relax_last_interval_rg": float(interval_mid_R_rg[interval_indices[-1]]),
        "outer_relax_n_variables": int(variable_cols_array.size),
        "outer_relax_n_rows": int(row_indices_array.size),
        "outer_relax_initial_selected": initial_selected,
        "outer_relax_final_selected": final_selected,
        "outer_relax_initial_full": initial_full,
        "outer_relax_final_full": final_full,
        "outer_relax_nfev": int(result.nfev),
        "outer_relax_success": bool(result.success),
        "outer_relax_message": str(result.message),
        "outer_relax_peak_node_min_rg": float(np.min(R_rg[node_indices])),
        "outer_relax_peak_node_max_rg": float(np.max(R_rg[node_indices])),
    }
    return relaxed, info


def _z_from_x(x: np.ndarray, params) -> np.ndarray:
    logu, logT, _logMdot, logR_son, lambda0, _logR = pilot._unpack(x, params)
    return pilot._state_vector(logu, logT, logR_son, lambda0)


def _solve_with_picard(x0: np.ndarray, params, eta_E: float):
    result = _solve_stage(x0, params, eta_E)
    current_params = params
    total_nfev = int(result.nfev)
    picard_used = 0
    for _idx in range(max(0, OUTER_SLOPE_PICARD_ITERS)):
        refreshed = scan.apply_outer_slopes_from_state(_z_from_x(result.x, current_params), current_params)
        trial = _solve_stage(result.x, refreshed, eta_E)
        total_nfev += int(trial.nfev)
        picard_used += 1
        if float(np.linalg.norm(pilot.residual(trial.x, refreshed), ord=np.inf)) <= float(
            np.linalg.norm(pilot.residual(result.x, current_params), ord=np.inf)
        ):
            result = trial
            current_params = refreshed
        else:
            break
    return result, current_params, total_nfev, picard_used


def _stage_row(label: str, x0: np.ndarray, result, params, eta_E: float, initial_full: float, profile: dict[str, Any]) -> dict[str, Any]:
    _set_eta(eta_E)
    row = pilot._row(label, result.x, params, initial_full, result)
    unweighted = np.asarray(pilot.residual(result.x, params), dtype=float)
    inner_idx = _inner_mdot_row_index(params)
    n = int(params.n_nodes)
    mass_start = inner_idx + 1
    interval_mass = unweighted[mass_start : mass_start + n - 1]
    row.update(
        {
            "eta_E": float(eta_E),
            "accepted_exploratory": bool(row["final_full"] <= ACCEPT_TOL),
            "inner_logMdot_residual": float(unweighted[inner_idx]),
            "interval_mass_residual_max": float(np.max(np.abs(interval_mass))) if interval_mass.size else math.nan,
            "peak_mass_residual_rg": profile["peak_mass_residual_rg"],
            "peak_mass_residual": profile["peak_mass_residual"],
            "mass_residual_p90_abs": profile["mass_residual_p90_abs"],
            "peak_interval_E_rg": profile["peak_interval_E_rg"],
            "peak_interval_E": profile["peak_interval_E"],
            "seed_initial_full": float(initial_full),
            "seed_initial_weighted_full": float(np.linalg.norm(_residual(x0, params), ord=np.inf)),
            "seed_previous_final_full": float(np.linalg.norm(pilot.residual(x0, params), ord=np.inf)),
            "inner_mdot_weight": float(INNER_MDOT_WEIGHT),
            "interval_residual_form": str(params.interval_residual_form),
            "integrated_residual_weighting": str(params.integrated_residual_weighting),
            "outer_closure": str(params.outer_closure),
            "outer_robin_chi": float(params.outer_robin_chi),
            "outer_buffer_inner_rg": np.nan if params.outer_buffer_inner_rg is None else float(params.outer_buffer_inner_rg),
            "outer_buffer_radial_weight": float(params.outer_buffer_radial_weight),
            "outer_buffer_energy_weight": float(params.outer_buffer_energy_weight),
            "outer_buffer_boundary_weight": float(params.outer_buffer_boundary_weight),
            "jac_row_norm_median": profile.get("jac_row_norm_median", math.nan),
            "jac_row_norm_max": profile.get("jac_row_norm_max", math.nan),
            "jac_col_norm_median": profile.get("jac_col_norm_median", math.nan),
            "jac_col_norm_max": profile.get("jac_col_norm_max", math.nan),
        }
    )
    return row


def _seed_stage_row(label: str, x0: np.ndarray, params, eta_E: float, initial_full: float, profile: dict[str, Any]) -> dict[str, Any]:
    _set_eta(eta_E)
    row = pilot._row(label, x0, params, initial_full, None)
    unweighted = np.asarray(pilot.residual(x0, params), dtype=float)
    inner_idx = _inner_mdot_row_index(params)
    n = int(params.n_nodes)
    mass_start = inner_idx + 1
    interval_mass = unweighted[mass_start : mass_start + n - 1]
    row.update(
        {
            "eta_E": float(eta_E),
            "accepted_exploratory": bool(row["final_full"] <= ACCEPT_TOL),
            "inner_logMdot_residual": float(unweighted[inner_idx]),
            "interval_mass_residual_max": float(np.max(np.abs(interval_mass))) if interval_mass.size else math.nan,
            "peak_mass_residual_rg": profile["peak_mass_residual_rg"],
            "peak_mass_residual": profile["peak_mass_residual"],
            "mass_residual_p90_abs": profile["mass_residual_p90_abs"],
            "peak_interval_E_rg": profile["peak_interval_E_rg"],
            "peak_interval_E": profile["peak_interval_E"],
            "seed_initial_full": float(initial_full),
            "seed_initial_weighted_full": float(np.linalg.norm(_residual(x0, params), ord=np.inf)),
            "seed_previous_final_full": float(np.linalg.norm(pilot.residual(x0, params), ord=np.inf)),
            "inner_mdot_weight": float(INNER_MDOT_WEIGHT),
            "interval_residual_form": str(params.interval_residual_form),
            "integrated_residual_weighting": str(params.integrated_residual_weighting),
            "outer_closure": str(params.outer_closure),
            "outer_robin_chi": float(params.outer_robin_chi),
            "outer_buffer_inner_rg": np.nan if params.outer_buffer_inner_rg is None else float(params.outer_buffer_inner_rg),
            "outer_buffer_radial_weight": float(params.outer_buffer_radial_weight),
            "outer_buffer_energy_weight": float(params.outer_buffer_energy_weight),
            "outer_buffer_boundary_weight": float(params.outer_buffer_boundary_weight),
            "jac_row_norm_median": math.nan,
            "jac_row_norm_max": math.nan,
            "jac_col_norm_median": math.nan,
            "jac_col_norm_max": math.nan,
            "seed_only": True,
            "picard_iters": 0,
            "nfev_total_with_picard": 0,
        }
    )
    return row


def _write_checkpoint(label: str, x: np.ndarray, params, row: dict[str, Any]) -> str:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logu, logT, logMdot, logR_son, lambda0, logR = pilot._unpack(x, params)
    local_params = pilot._local_params(params, logR, logMdot)
    z = pilot._state_vector(logu, logT, logR_son, lambda0)
    safe = _safe_eta_label(float(row["eta_E"]))
    path = CHECKPOINT_DIR / f"{label}_etaE_{safe}_N{int(params.n_nodes)}.npz"
    slopes = local_params.outer_match_log_slopes
    np.savez_compressed(
        path,
        x=np.asarray(x, dtype=float),
        z=np.asarray(z, dtype=float),
        ratio=np.array(local_params.Mdot_g_s / eddington_mdot(local_params.M2_g)),
        R_out_rg=np.array(local_params.R_out_rg),
        n_nodes=np.array(local_params.n_nodes),
        grid_power=np.array(local_params.grid_power),
        custom_grid_xi=np.asarray([] if local_params.custom_grid_xi is None else local_params.custom_grid_xi, dtype=float),
        outer_match_log_slopes=np.asarray([np.nan, np.nan] if slopes is None else slopes, dtype=float),
        wind_energy_multiplier=np.array(row["eta_E"]),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted_exploratory"]),
        row_json=np.array(json.dumps(scan.json_safe(row), sort_keys=True)),
    )
    return scan.relative_root_path(path)


def _write_outputs(rows: list[dict[str, Any]], profiles: list[dict[str, Any]]) -> None:
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(scan.json_safe(rows), indent=2, sort_keys=True) + "\n")
    PROFILE_OUTPUT.write_text(json.dumps(scan.json_safe(profiles), indent=2, sort_keys=True) + "\n")
    cols = [
        "label",
        "eta_E",
        "N",
        "seed_initial_full",
        "final_full",
        "mass_residual_max",
        "inner_logMdot_residual",
        "interval_mass_residual_max",
        "mass_residual_p90_abs",
        "peak_mass_residual_rg",
        "peak_interval_E_rg",
        "interval_E",
        "Mdot_outer_over_inner",
        "f_adv_global",
        "Lrad_LEdd",
        "Rson_rg",
        "nfev",
        "picard_iters",
        "nfev_total_with_picard",
        "seed_only",
        "interval_residual_form",
        "integrated_residual_weighting",
        "outer_closure",
        "outer_robin_chi",
        "residual_remesh_strength",
        "residual_remesh_seed_full",
        "residual_remesh_peak_monitor_rg",
        "residual_remesh_outer_5pct_nodes",
        "inner_relax_outer_rg",
        "inner_relax_initial_full",
        "inner_relax_final_full",
        "inner_relax_initial_selected",
        "inner_relax_final_selected",
        "inner_relax_nfev",
        "outer_relax_min_rg",
        "outer_relax_max_rg",
        "outer_relax_initial_full",
        "outer_relax_final_full",
        "outer_relax_initial_selected",
        "outer_relax_final_selected",
        "outer_relax_nfev",
        "success",
        "accepted_exploratory",
        "checkpoint",
    ]
    lines = [
        "# Mdot=5 Local-Mdot Eta Continuation",
        "",
        "Generated by `scripts/run_mdot5_local_mdot_eta_continuation.py`.",
        "",
        f"Anchor: `{scan.relative_root_path(ANCHOR)}`",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(col, "")) for col in cols) + " |")
    lines.extend(
        [
            "",
            "Profiles include interval-local mass residuals, interval energy residuals,",
            "`Qwind/Qvisc`, `Mwind_prime/Mdot`, `Mstream_prime/Mdot`, `Mdot_tilde`,",
            "`s_eff_tilde`, and final Jacobian row/column norm arrays.",
            "",
            f"Profile JSON: `{scan.relative_root_path(PROFILE_OUTPUT)}`",
        ]
    )
    MD_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not ETA_VALUES:
        raise ValueError("at least one eta_E stage is required")
    if not ANCHOR.exists():
        raise FileNotFoundError(ANCHOR)
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = scan.load_anchor(ANCHOR, fiducial, mdot_edd)
    x, params = _make_seed(anchor_z, anchor_params)
    _set_eta(ETA_VALUES[0])
    if START_X_CHECKPOINT is not None:
        if not START_X_CHECKPOINT.exists():
            raise FileNotFoundError(START_X_CHECKPOINT)
        data = np.load(START_X_CHECKPOINT)
        if "x" not in data:
            raise ValueError(f"{START_X_CHECKPOINT} does not contain a local-Mdot x vector")
        x = np.asarray(data["x"], dtype=float)
        expected = 3 * int(params.n_nodes) + 2
        if x.size != expected:
            if (x.size - 2) % 3 != 0:
                raise ValueError(f"start x has incompatible size {x.size}")
            old_n = int((x.size - 2) // 3)
            _old_z, old_params = _state_and_params_for_n(anchor_z, anchor_params, old_n)
            old_params = _restore_checkpoint_params(old_params, data)
            params = _restore_checkpoint_params(params, data)
            if REMAP_METHOD in {"nested_mass_ode", "nested_defect_preserving"}:
                params = _node_preserving_refined_params(x, old_params, params)
            x = _remap_local_x_to_params(x, old_params, params)
        else:
            params = _restore_checkpoint_params(params, data)
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []

    for stage_index, eta_E in enumerate(ETA_VALUES):
        label = f"stage_{stage_index:02d}"
        _set_eta(eta_E)
        remesh_info: dict[str, Any] = {}
        if RESIDUAL_REMESH_STRENGTH > 0.0:
            x, params, remesh_info = _residual_remesh(x, params, eta_E)
        inner_relax_info: dict[str, Any] = {}
        if INNER_RELAX_OUTER_RG > 0.0:
            x, inner_relax_info = _inner_window_relax(x, params, eta_E)
        outer_relax_info: dict[str, Any] = {}
        if OUTER_RELAX_MIN_RG > 0.0 and OUTER_RELAX_MAX_RG > OUTER_RELAX_MIN_RG:
            x, outer_relax_info = _outer_band_relax(x, params, eta_E)
        initial_full = float(np.linalg.norm(pilot.residual(x, params), ord=np.inf))
        initial_profile = _profile(f"{label}_initial", x, params, eta_E)
        print(f"{label} eta_E={eta_E:.8g} initial_full={initial_full:.3e}", flush=True)
        if SEED_ONLY:
            seed_profile = _profile(f"{label}_seed", x, params, eta_E)
            row = _seed_stage_row(label, x, params, eta_E, initial_full, seed_profile)
            row.update(remesh_info)
            row.update(inner_relax_info)
            row.update(outer_relax_info)
            row["checkpoint"] = _write_checkpoint(label, x, params, row)
            rows.append(row)
            profiles.extend([initial_profile, seed_profile])
            _write_outputs(rows, profiles)
            print(
                f"{label} seed_only final={row['final_full']:.3e} mass={row['mass_residual_max']:.3e} "
                f"peakM={row['peak_mass_residual_rg']:.3f}rg accepted={row['accepted_exploratory']}",
                flush=True,
            )
            continue
        result, stage_params, nfev_total, picard_used = _solve_with_picard(x, params, eta_E)
        final_jac_norms = _jac_norms(result, stage_params)
        final_profile = _profile(f"{label}_final", result.x, stage_params, eta_E, final_jac_norms)
        row = _stage_row(label, x, result, stage_params, eta_E, initial_full, final_profile)
        row.update(remesh_info)
        row.update(inner_relax_info)
        row.update(outer_relax_info)
        row["picard_iters"] = int(picard_used)
        row["nfev_total_with_picard"] = int(nfev_total)
        row["checkpoint"] = _write_checkpoint(label, result.x, stage_params, row)
        rows.append(row)
        profiles.extend([initial_profile, final_profile])
        _write_outputs(rows, profiles)
        print(
            f"{label} final={row['final_full']:.3e} mass={row['mass_residual_max']:.3e} "
            f"peakM={row['peak_mass_residual_rg']:.3f}rg nfev={row['nfev']} "
            f"accepted={row['accepted_exploratory']}",
            flush=True,
        )
        x = np.asarray(result.x, dtype=float)
        params = stage_params

    print(f"wrote {scan.relative_root_path(JSON_OUTPUT)}", flush=True)
    print(f"wrote {scan.relative_root_path(MD_OUTPUT)}", flush=True)
    print(f"wrote {scan.relative_root_path(PROFILE_OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
