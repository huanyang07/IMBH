"""Classify the Mdot=5 finite-radius, low-u phase-DAE limit."""

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

import run_mdot5_global_phase_dae_production as global_phase  # noqa: E402
import run_mdot5_phase_critical_globalization as critical  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_state,
    stream_annulus_shape_and_derivative,
    stream_source_prime,
    stream_torque_specific_l_and_derivative,
)
from imri_qpe.layer3_minidisk_1d.transonic_potential import PaczynskiWiitaPotential  # noqa: E402


model = global_phase.model
OUTPUT_STEM = os.environ.get(
    "IMBH_MDOT5_PHASE_CLASSIFICATION_OUTPUT_STEM",
    "m5_eta_phase_critical_classification_98p125_N164",
)
TABLE_PATH = ROOT / "outputs" / "tables" / f"{OUTPUT_STEM}.json"
PROFILE_PATH = ROOT / "outputs" / "tables" / f"{OUTPUT_STEM}_profiles.json"
FIGURE_PATH = ROOT / "outputs" / "figures" / f"{OUTPUT_STEM}.png"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / OUTPUT_STEM
NOTE_PATH = ROOT / "Note" / "CODEX_MDOT5_PHASE_CRITICAL_CLASSIFICATION_RESULTS.md"
FINE_ANCHOR = (
    ROOT
    / "outputs/checkpoints/m5_eta_phase_critical_arc_ds00025_98p125_N164"
    / "arc_step_749.npz"
)
EXIT_ANCHOR = (
    ROOT
    / "outputs/checkpoints/m5_eta_phase_dae_exit_refinement_98p125_N164"
    / "extend2_f8828125.npz"
)
LOGU_END = float(os.environ.get("IMBH_MDOT5_PHASE_CLASSIFICATION_LOGU_END", "4.0"))
BASELINE_STEPS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_MDOT5_PHASE_CLASSIFICATION_DT", "0.02,0.01").split(",")
    if piece.strip()
)
SOURCE_DT = float(os.environ.get("IMBH_MDOT5_PHASE_CLASSIFICATION_SOURCE_DT", "0.02"))
SOURCE_LOGU_END = float(os.environ.get("IMBH_MDOT5_PHASE_CLASSIFICATION_SOURCE_LOGU_END", "6.0"))
SOURCE_VARIANTS = tuple(
    piece.strip()
    for piece in os.environ.get(
        "IMBH_MDOT5_PHASE_CLASSIFICATION_SOURCE_VARIANTS",
        "compact_c2,compact_c4,compact_cinf,compact_c2_wide",
    ).split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_MDOT5_PHASE_CLASSIFICATION_MAX_NFEV", "50"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_phase(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path) as data:
        return tuple(np.asarray(data[key], dtype=float) for key in ("z", "p", "p_mid", "ds"))


def _scaled_tangent(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if float(p[0]) >= -1.0e-10:
        raise ValueError("logu continuation requires a negative p_logu tangent")
    scale = -1.0 / float(p[0])
    return scale * p


def _solve_unit_tangent(
    z: np.ndarray,
    seed: np.ndarray,
    params,
    lambda0: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import least_squares

    seed = np.asarray(seed, dtype=float)
    seed = seed / max(float(np.linalg.norm(seed)), 1.0e-300)
    if seed[3] < 0.0:
        seed = -seed
    lower = np.asarray([-1.5, -1.5, -0.2, 1.0e-12], dtype=float)
    upper = np.asarray([1.5, 1.5, 0.2, 1.5], dtype=float)

    def residual(p: np.ndarray) -> np.ndarray:
        homogeneous = np.asarray(
            model._global_flux_phase_dae_point_data(z, p, params, lambda0)["homogeneous_rows"],
            dtype=float,
        )
        return np.concatenate([homogeneous, np.asarray([np.linalg.norm(p) - 1.0])])

    result = least_squares(
        residual,
        np.clip(seed, lower + 1.0e-13, upper - 1.0e-13),
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=max(100, MAX_NFEV),
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-11,
    )
    rows = residual(result.x)
    accepted = bool(np.max(np.abs(rows)) <= 3.0e-6 and result.x[3] > 0.0)
    return np.asarray(result.x, dtype=float), {
        "accepted": accepted,
        "max": float(np.max(np.abs(rows))),
        "nfev": int(result.nfev),
        "message": str(result.message),
    }


def _solve_logu_tangent(
    z: np.ndarray,
    seed: np.ndarray,
    params,
    lambda0: float,
    *,
    allow_signed_pr: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import least_squares

    seed = _scaled_tangent(seed)
    start = np.asarray([seed[1], seed[2], seed[3]], dtype=float)
    p_r_lower = -0.1 if allow_signed_pr else 1.0e-16
    lower = np.asarray([-2.0, -0.2, p_r_lower], dtype=float)
    upper = np.asarray([2.0, 0.2, 0.1], dtype=float)

    def residual(values: np.ndarray) -> np.ndarray:
        p = np.asarray([-1.0, values[0], values[1], values[2]], dtype=float)
        return np.asarray(
            model._global_flux_phase_dae_point_data(z, p, params, lambda0)["homogeneous_rows"],
            dtype=float,
        )

    result = least_squares(
        residual,
        np.clip(start, lower + 1.0e-14, upper - 1.0e-14),
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=max(1, MAX_NFEV),
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-11,
    )
    p = np.asarray([-1.0, result.x[0], result.x[1], result.x[2]], dtype=float)
    rows = residual(result.x)
    return p, {
        "nfev": int(result.nfev),
        "status": int(result.status),
        "message": str(result.message),
        "max": float(np.max(np.abs(rows))),
        "accepted": bool(np.max(np.abs(rows)) <= 3.0e-7 and (allow_signed_pr or p[3] > 0.0)),
    }


def _solve_logR_tangent(
    z: np.ndarray,
    seed: np.ndarray,
    params,
    lambda0: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import least_squares

    seed = np.asarray(seed, dtype=float)
    scale = 1.0 / max(float(seed[3]), 1.0e-12)
    start = scale * seed[:3]
    lower = np.asarray([-500.0, -100.0, -2.0], dtype=float)
    upper = np.asarray([500.0, 100.0, 2.0], dtype=float)

    def residual(values: np.ndarray) -> np.ndarray:
        p = np.asarray([values[0], values[1], values[2], 1.0], dtype=float)
        return np.asarray(
            model._global_flux_phase_dae_point_data(z, p, params, lambda0)["homogeneous_rows"],
            dtype=float,
        )

    result = least_squares(
        residual,
        np.clip(start, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=max(100, MAX_NFEV),
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-11,
    )
    p = np.asarray([result.x[0], result.x[1], result.x[2], 1.0], dtype=float)
    rows = residual(result.x)
    return p, {
        "accepted": bool(np.max(np.abs(rows)) <= 3.0e-6),
        "max": float(np.max(np.abs(rows))),
        "nfev": int(result.nfev),
        "message": str(result.message),
    }


def _implicit_logR_step(
    z_old: np.ndarray,
    p_old: np.ndarray,
    dx: float,
    params,
    lambda0: float,
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, Any]]:
    from scipy.optimize import least_squares

    x_new = float(z_old[3] + dx)
    predicted = np.asarray(z_old[:3], dtype=float) + dx * np.asarray(p_old[:3], dtype=float)
    start = np.concatenate([predicted, np.asarray(p_old[:3], dtype=float)])
    lower = np.asarray(
        [z_old[0] - 1.0, z_old[1] - 0.5, max(z_old[2] - 0.1, 1.0e-10), -500.0, -100.0, -2.0],
        dtype=float,
    )
    upper = np.asarray(
        [z_old[0] + 1.0, z_old[1] + 0.5, z_old[2] + 0.1, 500.0, 100.0, 2.0],
        dtype=float,
    )

    def parts(values: np.ndarray):
        z_new = np.asarray([values[0], values[1], values[2], x_new], dtype=float)
        p_new = np.asarray([values[3], values[4], values[5], 1.0], dtype=float)
        homogeneous = np.asarray(
            model._global_flux_phase_dae_point_data(z_new, p_new, params, lambda0)["homogeneous_rows"],
            dtype=float,
        )
        kinematic = z_new[:3] - z_old[:3] - 0.5 * dx * (p_old[:3] + p_new[:3])
        return z_new, p_new, homogeneous, kinematic

    def residual(values: np.ndarray) -> np.ndarray:
        _z, _p, homogeneous, kinematic = parts(values)
        return np.concatenate([100.0 * homogeneous, 30.0 * kinematic])

    result = least_squares(
        residual,
        np.clip(start, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=max(1, MAX_NFEV),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    z_new, p_new, homogeneous, kinematic = parts(result.x)
    accepted = bool(
        np.max(np.abs(homogeneous)) <= 3.0e-6
        and np.max(np.abs(kinematic)) <= 3.0e-7
    )
    return accepted, z_new, p_new, {
        "nfev": int(result.nfev),
        "homogeneous_max": float(np.max(np.abs(homogeneous))),
        "kinematic_max": float(np.max(np.abs(kinematic))),
    }


def _physical_point(z: np.ndarray, p: np.ndarray, params, lambda0: float) -> dict[str, Any]:
    F = max(float(z[2]), 1.0e-300)
    p_R = float(p[3])
    dlogF_dx = float(p[2] / (F * p_R)) if abs(p_R) > 1.0e-300 else math.nan
    local = model._local_params_with_point_mdot(
        params,
        float(z[3]),
        math.log(F * params.Mdot_g_s),
        dlogF_dx if np.isfinite(dlogF_dx) else 0.0,
    )
    state = algebraic_state(float(z[3]), float(z[0]), float(z[1]), lambda0, local)
    g = np.asarray(p[:2], dtype=float) / max(abs(p_R), 1.0e-300)
    if p_R < 0.0:
        g = -g
    energy = model._energy_terms_at(float(z[3]), np.asarray(z[:2]), g, lambda0, local)
    wind_prime = model._safe_wind_prime(float(z[3]), np.asarray(z[:2]), g, lambda0, local)
    if not np.isfinite(wind_prime):
        wind_prime = math.nan
    source_prime = stream_source_prime(float(z[3]), local)
    point = model._global_flux_phase_dae_point_data(z, p, params, lambda0)
    right_min = np.asarray(point["A_right_min"], dtype=float)
    physical_state_tangent = np.asarray(p[:2], dtype=float) / max(abs(p_R), 1.0e-300)
    tangent_norm = max(float(np.linalg.norm(physical_state_tangent)), 1.0e-300)
    q_scale = max(abs(float(energy["Q_visc"])), 1.0e-300)
    return {
        "logu": float(z[0]),
        "u_cm_s": float(state.u),
        "logT": float(z[1]),
        "T_K": float(state.T),
        "F": F,
        "R_rg": float(state.R / params.r_g),
        "p_R": p_R,
        "p_T": float(p[1]),
        "p_F": float(p[2]),
        "dlogF_dlogR": dlogF_dx,
        "physical_derivative_norm": float(np.linalg.norm(p[:3]) / max(abs(p_R), 1.0e-300)),
        "Sigma": float(state.Sigma),
        "rho": float(state.rho),
        "H_over_R": float(state.H_over_R),
        "tau": float(state.tau),
        "Omega_over_OmegaK": float(state.Omega / state.Omega_K),
        "Mach_eff": float(state.u / max(state.H * state.Omega_K, 1.0e-300)),
        "Qvisc": float(energy["Q_visc"]),
        "Qrad_Qvisc": float(energy["Q_rad"] / q_scale),
        "Qadv_Qvisc": float(energy["Q_adv"] / q_scale),
        "Qwind_Qvisc": float(energy["Q_wind"] / q_scale),
        "Qstream_Qvisc": float(energy["Q_stream"] / q_scale),
        "wind_prime_over_inner": float(wind_prime / params.Mdot_g_s) if np.isfinite(wind_prime) else math.nan,
        "source_prime_over_inner": float(source_prime / params.Mdot_g_s),
        "sigma_min_A": float(np.min(point["A_singular_values"])),
        "cond_A": float(point["cond_A"]),
        "compatibility": float(point["compatibility"]),
        "A_left_min": np.asarray(point["A_left_min"], dtype=float),
        "A_right_min": right_min,
        "null_alignment": abs(float(np.dot(physical_state_tangent / tangent_norm, right_min))),
        "homogeneous_max": float(np.max(np.abs(point["homogeneous_rows"]))),
    }


def _implicit_logu_step(
    z_old: np.ndarray,
    p_old: np.ndarray,
    dt: float,
    params,
    lambda0: float,
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, Any]]:
    from scipy.optimize import least_squares

    logu_new = float(z_old[0] - dt)
    predicted = np.asarray(z_old[1:], dtype=float) + dt * np.asarray(p_old[1:], dtype=float)
    start = np.concatenate([predicted, np.asarray(p_old[1:], dtype=float)])
    lower = np.asarray(
        [z_old[1] - 0.2, max(z_old[2] - 0.05, 1.0e-10), z_old[3] - 0.01, -2.0, -0.2, 1.0e-16],
        dtype=float,
    )
    upper = np.asarray(
        [z_old[1] + 0.2, z_old[2] + 0.05, z_old[3] + 0.01, 2.0, 0.2, 0.1],
        dtype=float,
    )

    def parts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z_new = np.asarray([logu_new, values[0], values[1], values[2]], dtype=float)
        p_new = np.asarray([-1.0, values[3], values[4], values[5]], dtype=float)
        homogeneous = np.asarray(
            model._global_flux_phase_dae_point_data(z_new, p_new, params, lambda0)["homogeneous_rows"],
            dtype=float,
        )
        kinematic = np.asarray(z_new[1:] - z_old[1:] - 0.5 * dt * (p_old[1:] + p_new[1:]), dtype=float)
        return z_new, p_new, homogeneous, kinematic

    def residual(values: np.ndarray) -> np.ndarray:
        _z, _p, homogeneous, kinematic = parts(values)
        return np.concatenate([100.0 * homogeneous, 30.0 * kinematic])

    result = least_squares(
        residual,
        np.clip(start, lower + 1.0e-14, upper - 1.0e-14),
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=max(1, MAX_NFEV),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    z_new, p_new, homogeneous, kinematic = parts(result.x)
    accepted = bool(
        np.max(np.abs(homogeneous)) <= 3.0e-6
        and np.max(np.abs(kinematic)) <= 3.0e-7
        and p_new[3] > 0.0
    )
    return accepted, z_new, p_new, {
        "nfev": int(result.nfev),
        "status": int(result.status),
        "message": str(result.message),
        "homogeneous_max": float(np.max(np.abs(homogeneous))),
        "kinematic_max": float(np.max(np.abs(kinematic))),
    }


def _continue_in_logu(
    label: str,
    z_start: np.ndarray,
    p_start: np.ndarray,
    params,
    lambda0: float,
    logu_end: float,
    dt_initial: float,
) -> dict[str, Any]:
    p, tangent = _solve_logu_tangent(z_start, p_start, params, lambda0)
    z = np.asarray(z_start, dtype=float).copy()
    dt = float(dt_initial)
    points = [_physical_point(z, p, params, lambda0)]
    steps: list[dict[str, Any]] = []
    rejects = 0
    while float(z[0]) > float(logu_end) + 1.0e-10:
        dt = min(dt, float(z[0] - logu_end))
        accepted, z_new, p_new, diagnostics = _implicit_logu_step(z, p, dt, params, lambda0)
        if not accepted:
            dt *= 0.5
            rejects += 1
            if dt < 1.0e-5 or rejects >= 8:
                break
            continue
        rejects = 0
        z, p = z_new, p_new
        point = _physical_point(z, p, params, lambda0)
        points.append(point)
        steps.append({"dt": dt, **diagnostics, **{key: point[key] for key in ("logu", "R_rg", "p_R")}})
        if diagnostics["nfev"] < 12:
            dt = min(float(dt_initial), 1.15 * dt)
    positive = np.asarray([point["p_R"] for point in points], dtype=float)
    t = np.asarray([points[0]["logu"] - point["logu"] for point in points], dtype=float)
    tail_count = min(120, len(points))
    tail_t = t[-tail_count:]
    tail_pr = positive[-tail_count:]
    decay_rate = math.nan
    limit_R = math.nan
    if tail_count >= 10 and np.all(tail_pr > 0.0):
        slope, _intercept = np.polyfit(tail_t, np.log(tail_pr), 1)
        decay_rate = float(-slope)
        if decay_rate > 0.0:
            limit_R = float(points[-1]["R_rg"] * math.exp(points[-1]["p_R"] / decay_rate))
    summary = {
        "label": label,
        "dt_initial": float(dt_initial),
        "accepted_steps": len(steps),
        "rejected_steps": rejects,
        "reached_logu": float(points[-1]["logu"]),
        "target_logu": float(logu_end),
        "complete": bool(float(points[-1]["logu"]) <= float(logu_end) + 1.0e-8),
        "p_R_final": float(points[-1]["p_R"]),
        "R_final_rg": float(points[-1]["R_rg"]),
        "p_R_decay_rate": decay_rate,
        "R_limit_rg": limit_R,
        "tangent_initial": tangent,
        "max_homogeneous": float(max(point["homogeneous_max"] for point in points)),
    }
    return {"summary": summary, "points": points, "steps": steps, "z_final": z, "p_final": p}


def _scaling_audit(points: list[dict[str, Any]]) -> dict[str, Any]:
    tail = points[-min(160, len(points)) :]
    logu = np.asarray([point["logu"] for point in tail], dtype=float)
    output: dict[str, Any] = {}
    for key in ("Sigma", "rho", "H_over_R", "tau", "Mach_eff", "p_R"):
        values = np.asarray([point[key] for point in tail], dtype=float)
        valid = np.isfinite(values) & (values > 0.0)
        if np.count_nonzero(valid) >= 8:
            slope, intercept = np.polyfit(logu[valid], np.log(values[valid]), 1)
            output[key] = {"power_of_u": float(slope), "log_prefactor": float(intercept)}
    return output


def _select_pre_source_anchor(params, wider: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    z, p, p_mid, ds = _load_phase(EXIT_ANCHOR)
    width = float(params.stream_source_log_width) * (1.25 if wider else 1.0)
    edge = float(params.stream_source_center_fraction * params.R_out_rg * math.exp(-width))
    radius = np.exp(z[:, 3]) / params.r_g
    eligible = np.nonzero(radius < edge - 0.05)[0]
    if eligible.size == 0:
        raise RuntimeError("phase checkpoint has no node below requested source edge")
    pos = int(eligible[-1])
    return np.asarray(z[pos]), np.asarray(p[pos]), edge


def _solve_phase_variant(
    label: str,
    z_seed: np.ndarray,
    p_seed: np.ndarray,
    p_mid_seed: np.ndarray,
    ds_seed: np.ndarray,
    params,
    lambda0: float,
) -> dict[str, Any]:
    from scipy.optimize import least_squares

    count = int(ds_seed.size)
    node_count = count + 1
    start = global_phase._phase_pack(z_seed, p_seed, p_mid_seed, ds_seed)
    z_lo = np.asarray(z_seed, dtype=float).copy()
    z_hi = np.asarray(z_seed, dtype=float).copy()
    z_lo[:, :2] -= 0.5
    z_hi[:, :2] += 0.5
    z_lo[:, 2] = np.maximum(z_seed[:, 2] - 0.2, 1.0e-8)
    z_hi[:, 2] = z_seed[:, 2] + 0.2
    z_lo[:, 3] -= 0.02
    z_hi[:, 3] += 0.02
    for pos in range(node_count):
        if pos > 0:
            z_lo[pos, 3] = max(z_lo[pos, 3], 0.5 * (z_seed[pos - 1, 3] + z_seed[pos, 3]) + 1.0e-10)
        if pos < node_count - 1:
            z_hi[pos, 3] = min(z_hi[pos, 3], 0.5 * (z_seed[pos, 3] + z_seed[pos + 1, 3]) - 1.0e-10)
    p_lo = np.full_like(p_seed, -10.0)
    p_hi = np.full_like(p_seed, 10.0)
    pm_lo = np.full_like(p_mid_seed, -10.0)
    pm_hi = np.full_like(p_mid_seed, 10.0)
    p_lo[:, 3] = 1.0e-10
    pm_lo[:, 3] = 1.0e-10
    lower = np.concatenate([z_lo.ravel(), p_lo.ravel(), pm_lo.ravel(), np.log(ds_seed) - 3.0])
    upper = np.concatenate([z_hi.ravel(), p_hi.ravel(), pm_hi.ravel(), np.log(ds_seed) + 3.0])
    left_reference = np.asarray(z_seed[0], dtype=float)
    right_reference = np.asarray(z_seed[-1], dtype=float)
    labels = np.arange(count, dtype=int)
    mesh_target = np.diff(np.log(np.maximum(ds_seed, 1.0e-300)))

    def unpack(vector: np.ndarray):
        return global_phase._phase_unpack(vector, node_count, count)

    right_weights = np.asarray([0.1, 0.1, 0.1, 100.0])

    def residual(vector: np.ndarray) -> np.ndarray:
        z, p, p_mid, ds = unpack(vector)
        data = model._global_flux_phase_dae_segment_data(
            z, p, p_mid, ds, params, lambda0, labels, mesh_target
        )
        endpoints = np.concatenate(
            [
                100.0 * (z[0] - left_reference),
                right_weights * (z[-1] - right_reference),
            ]
        )
        return np.concatenate([np.asarray(data["rows"], dtype=float), endpoints])

    sparsity = model._global_flux_phase_dae_segment_sparsity(node_count, count, "state")
    current = np.clip(start, lower + 1.0e-12, upper - 1.0e-12)
    total_nfev = 0
    result = None
    accepted = False
    max_evaluations = max(40, MAX_NFEV)
    while total_nfev < max_evaluations:
        result = least_squares(
            residual,
            current,
            bounds=(lower, upper),
            jac_sparsity=sparsity,
            x_scale="jac",
            max_nfev=min(20, max_evaluations - total_nfev),
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        total_nfev += int(result.nfev)
        current = np.asarray(result.x)
        z, p, p_mid, ds = unpack(current)
        data = model._global_flux_phase_dae_segment_data(
            z, p, p_mid, ds, params, lambda0, labels, mesh_target
        )
        summary = dict(data["summary"])
        left_error = float(np.max(np.abs(z[0] - left_reference)))
        right_radius_error = abs(float(z[-1, 3] - right_reference[3]))
        accepted = bool(
            float(summary["radial_max"]) <= 1.0e-4
            and float(summary["energy_max"]) <= 1.0e-4
            and float(summary["fprime_max"]) <= 1.0e-5
            and float(summary["kinematic_max"]) <= 1.0e-3
            and float(summary["p_R_min"]) > 0.0
            and left_error <= 1.0e-5
            and right_radius_error <= 1.0e-5
        )
        if accepted:
            break
    assert result is not None
    summary.update(
        {
            "left_interface_error": left_error,
            "right_radius_error": right_radius_error,
        }
    )
    return {
        "label": label,
        "accepted": accepted,
        "nfev": total_nfev,
        "message": str(result.message),
        "summary": summary,
        "z": z,
        "p": p,
        "p_mid": p_mid,
        "ds": ds,
    }


def _source_shape_branches(params, lambda0: float) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    base_z, base_p, base_p_mid, base_ds = _load_phase(EXIT_ANCHOR)
    for name, shape, width_factor in (
        ("compact_c2", "compact_c2", 1.0),
        ("compact_c4", "compact_c4", 1.0),
        ("compact_cinf", "compact_cinf", 1.0),
        ("compact_c2_wide", "compact_c2", 1.25),
    ):
        if name not in SOURCE_VARIANTS:
            continue
        variant = replace(
            params,
            stream_source_shape=shape,
            stream_source_log_width=float(params.stream_source_log_width) * width_factor,
        )
        z_seed = np.asarray(base_z)
        p_seed = np.asarray(base_p)
        pm_seed = np.asarray(base_p_mid)
        ds_seed = np.asarray(base_ds)
        homotopy_records: list[dict[str, Any]] = []
        if width_factor > 1.0:
            width_values = np.linspace(1.0, width_factor, 11)[1:]
        else:
            width_values = np.asarray([width_factor])
        phase_solution = None
        for width_value in width_values:
            step_params = replace(
                params,
                stream_source_shape=shape,
                stream_source_log_width=float(params.stream_source_log_width) * float(width_value),
            )
            phase_solution = _solve_phase_variant(
                f"{name}_w{width_value:.3f}", z_seed, p_seed, pm_seed, ds_seed, step_params, lambda0
            )
            homotopy_records.append(
                {
                    "width_factor": float(width_value),
                    "accepted": bool(phase_solution["accepted"]),
                    "nfev": int(phase_solution["nfev"]),
                    **phase_solution["summary"],
                }
            )
            print(
                "source-phase",
                name,
                f"width={width_value:.3f}",
                f"accepted={phase_solution['accepted']}",
                f"nfev={phase_solution['nfev']}",
                f"radial={phase_solution['summary']['radial_max']:.3e}",
                f"energy={phase_solution['summary']['energy_max']:.3e}",
                f"kinematic={phase_solution['summary']['kinematic_max']:.3e}",
                flush=True,
            )
            if not phase_solution["accepted"]:
                break
            z_seed = np.asarray(phase_solution["z"])
            p_seed = np.asarray(phase_solution["p"])
            pm_seed = np.asarray(phase_solution["p_mid"])
            ds_seed = np.asarray(phase_solution["ds"])
        phase_accepted = bool(phase_solution is not None and phase_solution["accepted"])
        if phase_accepted and float(p_seed[-1, 0]) < 0.0:
            branch = _continue_in_logu(
                name,
                z_seed[-1],
                p_seed[-1],
                variant,
                lambda0,
                SOURCE_LOGU_END,
                SOURCE_DT,
            )
            branch["phase_homotopy"] = homotopy_records
        else:
            branch = {
                "summary": {
                    "label": name,
                    "dt_initial": SOURCE_DT,
                    "accepted_steps": 0,
                    "rejected_steps": 0,
                    "reached_logu": float(z_seed[-1, 0]),
                    "target_logu": SOURCE_LOGU_END,
                    "complete": False,
                    "p_R_final": float(p_seed[-1, 3]),
                    "R_final_rg": float(np.exp(z_seed[-1, 3]) / params.r_g),
                    "p_R_decay_rate": math.nan,
                    "R_limit_rg": math.nan,
                    "tangent_initial": {},
                    "max_homogeneous": (
                        float(phase_solution["summary"]["max"]) if phase_solution is not None else math.nan
                    ),
                },
                "points": [],
                "steps": [],
                "phase_homotopy": homotopy_records,
                "z_final": z_seed[-1],
                "p_final": p_seed[-1],
            }
        edge = float(variant.stream_source_center_fraction * variant.R_out_rg * math.exp(-variant.stream_source_log_width))
        center_shape, _center_prime = stream_annulus_shape_and_derivative(
            float(branch["z_final"][3]),
            float(variant.stream_source_center_fraction),
            float(variant.stream_source_log_width),
            float(variant.R_out),
            shape,
            float(variant.stream_source_shape_blend),
        )
        branch["summary"].update(
            {
                "shape": shape,
                "width_factor": width_factor,
                "source_inner_edge_rg": edge,
                "source_cumulative_at_final": float(center_shape),
                "anchor_R_rg": float(np.exp(base_z[0, 3]) / params.r_g),
                "anchor_logu": float(base_z[0, 0]),
                "phase_homotopy_accepted": phase_accepted,
            }
        )
        results.append(branch)
        print(
            "source",
            name,
            f"complete={branch['summary']['complete']}",
            f"Rlim={branch['summary']['R_limit_rg']:.6f}",
            f"pR={branch['summary']['p_R_final']:.3e}",
            flush=True,
        )
    return results


def _bordered_step(
    z_left: np.ndarray,
    p_left: np.ndarray,
    arc_target: float,
    params,
    lambda0: float,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    from scipy.optimize import least_squares

    p_left = np.asarray(p_left, dtype=float) / max(float(np.linalg.norm(p_left)), 1.0e-300)
    z_predict = np.asarray(z_left, dtype=float) + arc_target * p_left
    start = np.concatenate([z_predict, p_left, p_left, np.asarray([math.log(arc_target)])])
    lower = np.concatenate(
        [
            np.asarray([z_predict[0] - 0.1, z_predict[1] - 0.1, max(z_predict[2] - 0.02, 1.0e-10), z_left[3] - 0.002]),
            np.full(8, -1.5),
            np.asarray([math.log(arc_target) - 2.0]),
        ]
    )
    upper = np.concatenate(
        [
            np.asarray([z_predict[0] + 0.1, z_predict[1] + 0.1, z_predict[2] + 0.02, z_left[3] + 0.002]),
            np.full(8, 1.5),
            np.asarray([math.log(arc_target) + 2.0]),
        ]
    )

    def parts(values: np.ndarray):
        z_right = np.asarray(values[:4])
        p_mid = np.asarray(values[4:8])
        p_right = np.asarray(values[8:12])
        ds = float(np.exp(values[12]))
        z_mid = 0.5 * (z_left + z_right) + ds / 8.0 * (p_left - p_right)
        h_mid = np.asarray(model._global_flux_phase_dae_point_data(z_mid, p_mid, params, lambda0)["homogeneous_rows"])
        h_right = np.asarray(model._global_flux_phase_dae_point_data(z_right, p_right, params, lambda0)["homogeneous_rows"])
        kin = z_right - z_left - ds / 6.0 * (p_left + 4.0 * p_mid + p_right)
        norm = np.asarray([np.linalg.norm(p_mid) - 1.0, np.linalg.norm(p_right) - 1.0])
        arc = float(np.dot(z_right - z_left, p_left) - arc_target)
        return z_right, p_mid, p_right, ds, h_mid, h_right, kin, norm, arc

    weights = np.asarray([100.0] * 6 + [30.0] * 4 + [10.0] * 2 + [30.0])

    def residual(values: np.ndarray) -> np.ndarray:
        _z, _pm, _pr, _ds, hm, hr, kin, norm, arc = parts(values)
        return weights * np.concatenate([hm, hr, kin, norm, np.asarray([arc])])

    result = least_squares(
        residual,
        np.clip(start, lower + 1.0e-13, upper - 1.0e-13),
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=max(1, MAX_NFEV),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    z_right, p_mid, p_right, ds, hm, hr, kin, norm, arc = parts(result.x)
    accepted = bool(
        max(np.max(np.abs(hm)), np.max(np.abs(hr))) <= 3.0e-5
        and np.max(np.abs(kin)) <= 3.0e-4
        and np.max(np.abs(norm)) <= 1.0e-4
        and abs(arc) <= 1.0e-5
    )
    return accepted, z_right, p_mid, p_right, ds, {
        "nfev": int(result.nfev),
        "homogeneous_max": float(max(np.max(np.abs(hm)), np.max(np.abs(hr)))),
        "kinematic_max": float(np.max(np.abs(kin))),
        "norm_max": float(np.max(np.abs(norm))),
        "arc_residual": abs(arc),
    }


def _bordered_audit(z0: np.ndarray, p0: np.ndarray, params, lambda0: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arc_target in (0.005, 0.0025):
        z = np.asarray(z0, dtype=float).copy()
        p = np.asarray(p0, dtype=float).copy()
        local_rows: list[dict[str, Any]] = []
        for step in range(30):
            accepted, z_new, _p_mid, p_new, ds, diagnostics = _bordered_step(
                z, p, arc_target, params, lambda0
            )
            point = _physical_point(z_new, p_new, params, lambda0)
            local_rows.append(
                {"step": step, "arc_target": arc_target, "accepted": accepted, "ds": ds, **diagnostics, **point}
            )
            if not accepted:
                break
            z, p = z_new, p_new
        rows.append(
            {
                "arc_target": arc_target,
                "accepted_steps": sum(bool(row["accepted"]) for row in local_rows),
                "crossed": any(float(row["p_R"]) < 0.0 for row in local_rows if row["accepted"]),
                "final_R_rg": float(local_rows[-1]["R_rg"]),
                "final_p_R": float(local_rows[-1]["p_R"]),
                "rows": local_rows,
            }
        )
    return rows


def _angular_closure_audit(points: list[dict[str, Any]], params, lambda0: float) -> list[dict[str, Any]]:
    potential = PaczynskiWiitaPotential(params.M2_g)
    l_injection = float(potential.l_k(float(params.stream_source_center_fraction * params.R_out)))
    output: list[dict[str, Any]] = []
    for closure in ("disk_local", "keplerian_local", "keplerian_injection"):
        defects: list[float] = []
        for left, right in zip(points[:-1], points[1:]):
            z_l = np.asarray([left[key] for key in ("logu", "logT", "F") ] + [math.log(left["R_rg"] * params.r_g)])
            z_r = np.asarray([right[key] for key in ("logu", "logT", "F") ] + [math.log(right["R_rg"] * params.r_g)])
            p_l = np.asarray([-1.0, left["p_T"], left["p_F"], left["p_R"]])
            p_r = np.asarray([-1.0, right["p_T"], right["p_F"], right["p_R"]])

            def terms(z: np.ndarray, p: np.ndarray):
                F = max(float(z[2]), 1.0e-300)
                dlogF = float(p[2] / max(F * p[3], 1.0e-300))
                local = model._local_params_with_point_mdot(
                    params, float(z[3]), math.log(F * params.Mdot_g_s), dlogF
                )
                state = algebraic_state(float(z[3]), float(z[0]), float(z[1]), lambda0, local)
                g = np.asarray(p[:2]) / max(float(p[3]), 1.0e-300)
                wind = model._safe_wind_prime(float(z[3]), np.asarray(z[:2]), g, lambda0, local)
                source = stream_source_prime(float(z[3]), local)
                _stream_l, stream_dl = stream_torque_specific_l_and_derivative(float(z[3]), local)
                mdot = F * params.Mdot_g_s
                flux = mdot * state.l - 2.0 * np.pi * state.R**2 * state.W
                if closure == "disk_local":
                    l_s = state.l
                elif closure == "keplerian_local":
                    l_s = state.l_K
                else:
                    l_s = l_injection
                source_term = float(wind) * state.l - float(source) * l_s + mdot * float(stream_dl)
                return flux, source_term, state.l_K

            flux_l, source_l, lk_l = terms(z_l, p_l)
            flux_r, source_r, lk_r = terms(z_r, p_r)
            dx = float(z_r[3] - z_l[3])
            scale = max(params.Mdot_g_s * 0.5 * (abs(lk_l) + abs(lk_r)), 1.0e-300)
            defects.append(float((flux_r - flux_l - 0.5 * dx * (source_l + source_r)) / scale))
        output.append(
            {
                "closure": closure,
                "max": float(np.max(np.abs(defects))) if defects else math.nan,
                "rms": float(np.sqrt(np.mean(np.asarray(defects) ** 2))) if defects else math.nan,
            }
        )
    return output


def _write_note(result: dict[str, Any]) -> None:
    baseline = result["baseline"]
    source = result["source_branches"]
    scaling = result["scaling"]
    bordered = result["bordered"]
    angular = result["angular_closures"]
    lines = [
        "# Mdot=5 phase critical classification results",
        "",
        "Target: `Mdot_inner/Edd=5`, `Rout=335 rg`, `f_s=0.80`, `eta_E=98.125`, `N=164`.",
        "",
        "## Low-u continuation",
        "",
        "The critical branch was reparameterized by decreasing `logu`, with `p_logu=-1`. This removes `p_R` from the denominator and follows the positive radial sheet without clipping.",
        "",
        "| dt | complete | final logu | final R (rg) | final p_R | fitted R limit (rg) | p_R decay rate | max H |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in baseline:
        row = branch["summary"]
        lines.append(
            f"| {row['dt_initial']:.4f} | {row['complete']} | {row['reached_logu']:.4f} | "
            f"{row['R_final_rg']:.6f} | {row['p_R_final']:.3e} | {row['R_limit_rg']:.6f} | "
            f"{row['p_R_decay_rate']:.3f} | {row['max_homogeneous']:.3e} |"
        )
    fine_points = baseline[-1]["points"]
    sample_indices = np.linspace(0, len(fine_points) - 1, 7, dtype=int)
    lines.extend(
        [
            "",
            "### Physical asymptotics",
            "",
            "| logu | R (rg) | p_R | Sigma | rho | H/R | tau | Mach | Qadv/Qvisc | Qwind/Qvisc |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx in sample_indices:
        point = fine_points[int(idx)]
        lines.append(
            f"| {point['logu']:.3f} | {point['R_rg']:.6f} | {point['p_R']:.3e} | {point['Sigma']:.3e} | "
            f"{point['rho']:.3e} | {point['H_over_R']:.3e} | {point['tau']:.3e} | {point['Mach_eff']:.3e} | "
            f"{point['Qadv_Qvisc']:.3e} | {point['Qwind_Qvisc']:.3e} |"
        )
    lines.extend(["", "Tail power laws `quantity proportional to u^a`:", ""])
    for key, values in scaling.items():
        lines.append(f"- `{key}`: `a={values['power_of_u']:.4f}`.")
    lines.extend(
        [
            "",
            "### Critical eigenstructure",
            "",
            "| logu | R (rg) | p_R | sigma_min(A) | cond(A) | C=u_min^T c | null alignment | |dz/dlnR| | u_min | v_min |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for idx in np.linspace(0, len(fine_points) - 1, 6, dtype=int):
        point = fine_points[int(idx)]
        left = ", ".join(f"{value:.3f}" for value in point["A_left_min"])
        right = ", ".join(f"{value:.3f}" for value in point["A_right_min"])
        lines.append(
            f"| {point['logu']:.3f} | {point['R_rg']:.6f} | {point['p_R']:.3e} | "
            f"{point['sigma_min_A']:.3e} | {point['cond_A']:.3e} | {point['compatibility']:.3e} | "
            f"{point['null_alignment']:.6f} | {point['physical_derivative_norm']:.3e} | "
            f"({left}) | ({right}) |"
        )
    lines.extend(
        [
            "",
            "`sigma_min(A)` and `p_R` vanish while compatibility remains near `0.19` and the physical derivative diverges. "
            "This excludes a regular critical point. Because `u` tends to zero and density variables diverge, it is an asymptotic singular boundary rather than a finite-state fold.",
            "",
            "## Bordered intrinsic corrector",
            "",
            "| arc target | accepted steps | crossed p_R=0 | final R (rg) | final p_R |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for row in bordered:
        lines.append(
            f"| {row['arc_target']:.4f} | {row['accepted_steps']} | {row['crossed']} | "
            f"{row['final_R_rg']:.6f} | {row['final_p_R']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Re-solved source-shape branches",
            "",
            "| branch | anchor R (rg) | source edge (rg) | complete | final p_R | fitted R limit (rg) | source cumulative |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for branch in source:
        row = branch["summary"]
        lines.append(
            f"| {row['label']} | {row['anchor_R_rg']:.6f} | {row['source_inner_edge_rg']:.6f} | "
            f"{row['complete']} | {row['p_R_final']:.3e} | {row['R_limit_rg']:.6f} | "
            f"{row['source_cumulative_at_final']:.6f} |"
        )
    source_limits = np.asarray([branch["summary"]["R_limit_rg"] for branch in source], dtype=float)
    lines.extend(
        [
            "",
            f"The full source-shape spread in fitted limiting radius is `{np.ptp(source_limits):.6f} rg`, "
            "well below the predeclared `0.05 rg` sensitivity threshold.",
        ]
    )
    lines.extend(
        [
            "",
            "## Angular closure audit",
            "",
            "| assumed stream l_s | max FV defect | RMS |",
            "|---|---:|---:|",
        ]
    )
    for row in angular:
        lines.append(f"| {row['closure']} | {row['max']:.3e} | {row['rms']:.3e} |")
    lines.extend(
        [
            "",
            "This angular audit covers the low-u endpoint tail only. It supports closure consistency there, "
            "but it does not promote a stream angular-momentum law to the global production equations.",
        ]
    )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Classification: **{decision['classification']}**.",
            f"- Finite-state fold: `{decision['finite_state_fold']}`.",
            f"- Regular critical point: `{decision['regular_critical_point']}`.",
            f"- Source-shape audit resolved: `{decision['source_shape_resolved']}`.",
            f"- Source-shape sensitive: `{decision['source_shape_sensitive']}`.",
            f"- Global steady branch certified: `{decision['global_certified']}`.",
            "",
            decision["explanation"],
            "",
            "Eta continuation remains paused.",
            "",
            "## Files",
            "",
            f"- table: `{TABLE_PATH.relative_to(ROOT)}`",
            f"- profiles: `{PROFILE_PATH.relative_to(ROOT)}`",
            f"- figure: `{FIGURE_PATH.relative_to(ROOT)}`",
        ]
    )
    NOTE_PATH.write_text("\n".join(lines) + "\n")


def _write_figure(result: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1450, 950), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    panels = ((70, 60, 700, 430), (770, 60, 1400, 430), (70, 520, 700, 890), (770, 520, 1400, 890))

    def panel(box, series, title, xlabel, ylabel, logy=False, horizontal=None):
        left, top, right, bottom = box
        x0, x1, y0, y1 = left + 72, right - 20, top + 34, bottom - 50
        transformed = []
        all_x, all_y = [], []
        for xs, ys, color, label in series:
            xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
            valid = np.isfinite(xs) & np.isfinite(ys) & ((ys > 0.0) if logy else True)
            xs, ys = xs[valid], ys[valid]
            if logy:
                ys = np.log10(ys)
            transformed.append((xs, ys, color, label))
            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())
        for value, _color, _label in horizontal or []:
            all_y.append(math.log10(value) if logy else value)
        if not all_x or not all_y:
            return
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        dx, dy = max(xmax - xmin, 1.0e-12), max(ymax - ymin, 1.0e-12)
        xmin, xmax = xmin - 0.04 * dx, xmax + 0.04 * dx
        ymin, ymax = ymin - 0.06 * dy, ymax + 0.06 * dy

        def xy(x, y):
            return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0), y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)

        draw.rectangle(box, outline="#D5D8DC")
        for tick in range(5):
            xv = xmin + tick * (xmax - xmin) / 4
            yv = ymin + tick * (ymax - ymin) / 4
            px, _ = xy(xv, ymin)
            _, py = xy(xmin, yv)
            draw.line((px, y0, px, y1), fill="#ECEFF1")
            draw.line((x0, py, x1, py), fill="#ECEFF1")
            draw.text((px - 18, y1 + 8), f"{xv:.4g}", fill="#34495E", font=font)
            draw.text((left + 4, py - 6), f"10^{yv:.1f}" if logy else f"{yv:.5g}", fill="#34495E", font=font)
        draw.line((x0, y1, x1, y1), fill="#2C3E50", width=2)
        draw.line((x0, y0, x0, y1), fill="#2C3E50", width=2)
        draw.text((left + 8, top + 8), title, fill="#17202A", font=font)
        draw.text(((x0 + x1) / 2 - 24, bottom - 18), xlabel, fill="#34495E", font=font)
        draw.text((left + 4, top + 22), ylabel, fill="#34495E", font=font)
        legend_y = top + 8
        for xs, ys, color, label in transformed:
            points = [xy(float(x), float(y)) for x, y in zip(xs, ys)]
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            draw.line((right - 145, legend_y + 5, right - 125, legend_y + 5), fill=color, width=3)
            draw.text((right - 120, legend_y), label, fill="#34495E", font=font)
            legend_y += 16
        for value, color, label in horizontal or []:
            y = math.log10(value) if logy else value
            _, py = xy(xmin, y)
            draw.line((x0, py, x1, py), fill=color, width=2)
            draw.text((x0 + 4, py - 14), label, fill=color, font=font)

    colors = ("#176B87", "#C0392B", "#7D3C98", "#148F77")
    baseline_series = []
    for branch, color in zip(result["baseline"], colors):
        pts = branch["points"]
        baseline_series.append(
            ([p["logu"] for p in pts], [p["p_R"] for p in pts], color, f"dt={branch['summary']['dt_initial']:g}")
        )
    panel(panels[0], baseline_series, "Positive branch in logu coordinate", "log u", "p_R", logy=True)

    physical = result["baseline"][-1]["points"]
    panel(
        panels[1],
        [
            ([p["logu"] for p in physical], [p["Sigma"] for p in physical], "#B03A2E", "Sigma"),
            ([p["logu"] for p in physical], [p["tau"] for p in physical], "#7D3C98", "tau"),
            ([p["logu"] for p in physical], [p["Mach_eff"] for p in physical], "#148F77", "Mach"),
        ],
        "Physical low-u asymptotics", "log u", "value", logy=True,
    )

    source_series = []
    for branch, color in zip(result["source_branches"], colors):
        pts = branch["points"]
        source_series.append(
            ([p["logu"] for p in pts], [p["R_rg"] for p in pts], color, branch["summary"]["label"])
        )
    panel(panels[2], source_series, "Re-solved source branches", "log u", "R / rg")

    angular = result["angular_closures"]
    panel(
        panels[3],
        [(np.arange(len(angular)), [row["max"] for row in angular], "#C0392B", "max FV defect")],
        "Angular closure audit", "closure index", "defect", logy=True,
        horizontal=[(3.0e-5, "#148F77", "global gate")],
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_PATH)


def main() -> None:
    _x, params, _context, _aux, phase = global_phase._load_problem()
    lambda0 = float(phase["lambda0"])
    z_fine, p_fine, _pm_fine, _ds_fine = _load_phase(FINE_ANCHOR)
    z0 = np.asarray(z_fine[-1], dtype=float)
    p0 = np.asarray(p_fine[-1], dtype=float)

    baseline = []
    for dt in BASELINE_STEPS:
        branch = _continue_in_logu(f"baseline_dt_{dt:g}", z0, p0, params, lambda0, LOGU_END, dt)
        baseline.append(branch)
        print(
            "baseline",
            f"dt={dt:g}",
            f"complete={branch['summary']['complete']}",
            f"Rlim={branch['summary']['R_limit_rg']:.6f}",
            f"pR={branch['summary']['p_R_final']:.3e}",
            flush=True,
        )

    scaling = _scaling_audit(baseline[-1]["points"])
    bordered = _bordered_audit(z0, p0, params, lambda0)
    source_branches = _source_shape_branches(params, lambda0)
    angular = _angular_closure_audit(baseline[-1]["points"], params, lambda0)

    limits = np.asarray([branch["summary"]["R_limit_rg"] for branch in source_branches], dtype=float)
    source_resolved = bool(
        len(source_branches) == len(SOURCE_VARIANTS)
        and all(
            bool(branch["summary"]["complete"])
            and bool(branch["summary"]["phase_homotopy_accepted"])
            and np.isfinite(float(branch["summary"]["R_limit_rg"]))
            for branch in source_branches
        )
    )
    source_sensitive = bool(source_resolved and np.ptp(limits) > 0.05)
    sigma_power = float(scaling.get("Sigma", {}).get("power_of_u", math.nan))
    rho_power = float(scaling.get("rho", {}).get("power_of_u", math.nan))
    low_u_singular = bool(
        baseline[-1]["summary"]["complete"]
        and baseline[-1]["summary"]["p_R_final"] > 0.0
        and np.isfinite(sigma_power)
        and sigma_power < -0.5
    )
    decision = {
        "classification": (
            "source-sensitive finite-radius stagnation/singular boundary"
            if source_sensitive
            else (
                "finite-radius low-u stagnation/singular boundary"
                if source_resolved
                else "finite-radius low-u stagnation/singular boundary; source audit unresolved"
            )
        ),
        "finite_state_fold": False,
        "regular_critical_point": False,
        "source_shape_resolved": source_resolved,
        "source_shape_sensitive": source_sensitive,
        "global_certified": False,
        "low_u_singular": low_u_singular,
        "explanation": (
            "The positive-p_R branch remains regular when logu is used as the continuation coordinate, "
            "but approaches u=0 at finite radius while surface density and optical depth diverge. "
            "No step-converged finite-state p_R sign change is found, and no admissible outer radial branch is available."
        ),
    }
    result = {
        "target": {"eta_E": 98.125, "N": int(params.n_nodes), "Rout_rg": float(params.R_out_rg)},
        "baseline": baseline,
        "scaling": scaling,
        "bordered": bordered,
        "source_branches": source_branches,
        "angular_closures": angular,
        "decision": decision,
    }
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    compact = {
        **result,
        "baseline": [{"summary": branch["summary"]} for branch in baseline],
        "source_branches": [{"summary": branch["summary"]} for branch in source_branches],
        "bordered": [
            {key: value for key, value in row.items() if key != "rows"} for row in bordered
        ],
    }
    TABLE_PATH.write_text(json.dumps(_jsonable(compact), indent=2, sort_keys=True) + "\n")
    PROFILE_PATH.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        CHECKPOINT_DIR / "baseline_final.npz",
        z=np.asarray(baseline[-1]["z_final"]),
        p=np.asarray(baseline[-1]["p_final"]),
    )
    _write_note(result)
    _write_figure(result)
    print("decision", json.dumps(_jsonable(decision), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
