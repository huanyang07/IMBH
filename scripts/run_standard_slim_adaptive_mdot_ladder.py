"""Adaptive Mdot continuation with tangent prediction and sonic-root injection."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    pressure_supported_omega_target,
    remap_profile_to_new_sonic_grid,
    residual_audit_from_state_vector,
    solve_square_transonic_polish,
    square_collocation_residual,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant
from run_standard_slim_mdot_predictor_audit import clip_state, tangent_predictor, tangent_vector
from run_standard_slim_sonic_compatibility_probe import component_vector, solve_local, sonic_components
from run_standard_slim_sonic_root_injection import injected_state, local_from_z


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_TABLE",
    "outputs/tables/slim_benchmark_adaptive_mdot_ladder.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_FIGURE",
    "outputs/figures/slim_benchmark_adaptive_mdot_ladder.png",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder",
)
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ANCHOR",
    "outputs/checkpoints/slim_benchmark_rout_injection_ladder/Rout_10000.npz",
)

DOWN_TARGET = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_DOWN_TARGET", "9e-4"))
UP_TARGET = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_UP_TARGET", "1.1e-3"))
START_STEP_MU = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_START_STEP", "0.01"))
MAX_STEP_MU = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_MAX_STEP", "0.08"))
MIN_STEP_MU = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_MIN_STEP", "0.0025"))
GROW_FACTOR = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_GROW", "1.35"))
SHRINK_FACTOR = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_SHRINK", "0.5"))
MAX_ATTEMPTS_PER_BRANCH = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_MAX_ATTEMPTS", "18"))
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ANCHOR_TOL", "3e-6"))
CURRENT_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_CURRENT_TOL", "5e-6"))
SCOUT_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_SCOUT_TOL", str(ACCEPTANCE_TOL)))
SOURCE_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_SOURCE_NFEV", "900"))
POLISH_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_POLISH_NFEV", "1800"))
FALLBACK_LSQ_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_FALLBACK_LSQ_NFEV", "260"))
USE_INTEGRATED_PREPOLISH = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_INTEGRATED_PREPOLISH", "0") != "0"
INTEGRATED_WEIGHTING = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_INTEGRATED_WEIGHTING", "inverse_sqrt_dx")
INTEGRATED_PRE_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_INTEGRATED_PRE_NFEV", "500"))
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_PIVOTS", "C2,C1,K").replace(":", ",").split(",")
    if piece.strip()
)
TANGENT_PIVOT = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_TANGENT_PIVOT", "C2")
USE_SCOUT_AS_CURRENT = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_USE_SCOUT_AS_CURRENT", "0") != "0"
STRESS_FACTOR = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_STRESS", "1.0"))
GRID_POWER = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_GRID_POWER", "1.0"))
VERBOSE_PHASES = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_VERBOSE_PHASES", "0") != "0"
NEWTON_USE_BLOCK_JACOBIAN = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_NEWTON_BLOCK_JACOBIAN", "1") != "0"
LSQ_USE_BLOCK_JACOBIAN = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_LSQ_BLOCK_JACOBIAN", "0") != "0"
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_NEWTON_LINEAR_SOLVER", "direct")
NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_NEWTON_MAX_ITER", "12"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_NEWTON_MAX_STEP_NORM", "0.5"))
NEWTON_LINE_SEARCH_REDUCTIONS = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_NEWTON_LINE_SEARCH_REDUCTIONS", "12"))
SONIC_INJECTION_POLICY = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_INJECTION_POLICY", "if_better")
SONIC_INJECTION_ACCEPT_FACTOR = float(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_INJECTION_ACCEPT_FACTOR", "1.0"))
USE_SECANT_PREDICTOR = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_USE_SECANT", "1") != "0"
SKIP_FINAL_IF_CURRENT = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_SKIP_FINAL_IF_CURRENT", "1") != "0"
OUTER_CLOSURE_MODE = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_OUTER_CLOSURE", "thin_value")
OUTER_SLOPE_REFRESHES = int(
    os.environ.get(
        "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_OUTER_SLOPE_REFRESHES",
        "0" if OUTER_CLOSURE_MODE == "thin_value" else "1",
    )
)
OUTER_SLOPE_REFRESH_REPOLISH = os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_OUTER_SLOPE_REFRESH_REPOLISH", "1") != "0"
OUTER_SLOPE_FIT_N = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_OUTER_SLOPE_FIT_N", "8"))
ADAPTIVE_N_VALUES = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ADAPTIVE_N_VALUES", "")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
ADAPTIVE_N_TRIGGER_TOL = float(
    os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ADAPTIVE_N_TRIGGER_TOL", str(ACCEPTANCE_TOL))
)
ADAPTIVE_N_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ADAPTIVE_N_NFEV", str(POLISH_NFEV)))

if OUTER_CLOSURE_MODE not in {"thin_value", "pressure_one_sided", "pressure_polyfit"}:
    raise ValueError(
        "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_OUTER_CLOSURE must be "
        "thin_value, pressure_one_sided, or pressure_polyfit"
    )
if OUTER_SLOPE_REFRESHES < 0:
    raise ValueError("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_OUTER_SLOPE_REFRESHES must be non-negative")
if any(n <= 2 for n in ADAPTIVE_N_VALUES):
    raise ValueError("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ADAPTIVE_N_VALUES entries must exceed 2")
if ADAPTIVE_N_TRIGGER_TOL <= 0.0:
    raise ValueError("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ADAPTIVE_N_TRIGGER_TOL must be positive")


def phase_log(message: str) -> None:
    if VERBOSE_PHASES:
        print(f"    {message}", flush=True)


def pressure_outer_enabled() -> bool:
    return OUTER_CLOSURE_MODE != "thin_value"


def one_sided_outer_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    dx = float(logR[-1] - logR[-2])
    return float((logu[-1] - logu[-2]) / dx), float((logT[-1] - logT[-2]) / dx)


def polyfit_outer_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    count = min(max(int(OUTER_SLOPE_FIT_N), 2), len(logR))
    x = logR[-count:] - logR[-1]
    gu = float(np.polyder(np.poly1d(np.polyfit(x, logu[-count:], min(2, count - 1))))(0.0))
    gT = float(np.polyder(np.poly1d(np.polyfit(x, logT[-count:], min(2, count - 1))))(0.0))
    return gu, gT


def outer_slopes_from_state(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float] | None:
    if OUTER_CLOSURE_MODE == "thin_value":
        return None
    if OUTER_CLOSURE_MODE == "pressure_one_sided":
        return one_sided_outer_slopes(z, params)
    if OUTER_CLOSURE_MODE == "pressure_polyfit":
        return polyfit_outer_slopes(z, params)
    raise ValueError(f"unknown outer closure mode {OUTER_CLOSURE_MODE!r}")


def apply_outer_closure_from_state(z: np.ndarray, params: TransonicSlimParams) -> TransonicSlimParams:
    if not pressure_outer_enabled():
        return replace(params, outer_closure="thin_value", outer_match_log_slopes=None)
    slopes = outer_slopes_from_state(z, params)
    return replace(params, outer_closure="pressure_supported_thin_energy", outer_match_log_slopes=slopes)


def outer_pressure_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    y = np.array([logu[-1], logT[-1]], dtype=float)
    slopes = np.asarray(one_sided_outer_slopes(z, params), dtype=float)
    profile = transonic_profile_from_state_vector(z, params)
    ln_omega = float(np.log(profile.Omega[-1] / profile.Omega_K[-1]))
    target = pressure_supported_omega_target(float(logR[-1]), y, slopes, lambda0, params)
    return {
        "outer_lnOmega_OmegaK": ln_omega,
        "outer_pressure_target": float(target),
        "outer_pressure_residual": float(ln_omega - target),
    }


def outer_closure_row_fields(z: np.ndarray, params: TransonicSlimParams) -> dict[str, object]:
    slopes = params.outer_match_log_slopes
    pressure = outer_pressure_diagnostic(z, params)
    return {
        "outer_closure_mode": OUTER_CLOSURE_MODE,
        "outer_closure": params.outer_closure,
        "outer_slope_source": "-" if OUTER_CLOSURE_MODE == "thin_value" else OUTER_CLOSURE_MODE.replace("pressure_", ""),
        "outer_g_u": np.nan if slopes is None else float(slopes[0]),
        "outer_g_T": np.nan if slopes is None else float(slopes[1]),
        **pressure,
    }


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    ratio: float,
    R_out_rg: float,
    n_nodes: int,
    *,
    grid_power: float | None = None,
    custom_grid_xi: tuple[float, ...] | None = None,
) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(ratio) * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=float(R_out_rg),
        n_nodes=int(n_nodes),
        grid_power=GRID_POWER if grid_power is None else float(grid_power),
        custom_grid_xi=custom_grid_xi,
        max_nfev=max(SOURCE_NFEV, POLISH_NFEV, FALLBACK_LSQ_NFEV),
        residual_tol=1.0e-8,
        outer_closure="thin_value",
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def load_anchor(fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    data = np.load(ANCHOR_CHECKPOINT, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    source_grid_power = float(data["grid_power"]) if "grid_power" in data else 1.0
    custom_grid_xi = None
    if "custom_grid_xi" in data:
        candidate_grid = np.asarray(data["custom_grid_xi"], dtype=float)
        if candidate_grid.shape == (int(data["n_nodes"]),):
            custom_grid_xi = tuple(float(value) for value in candidate_grid)
    params = params_for(
        fiducial,
        mdot_edd,
        float(data["ratio"]),
        float(data["R_out_rg"]),
        int(data["n_nodes"]),
        grid_power=source_grid_power,
        custom_grid_xi=custom_grid_xi,
    )
    return z, apply_outer_closure_from_state(z, params)


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


def square_max(z: np.ndarray, params: TransonicSlimParams, pivot: str) -> float:
    return float(np.max(np.abs(square_collocation_residual(z, params, pivot=pivot))))


def secant_predictor(
    previous_z: np.ndarray | None,
    previous_params: TransonicSlimParams | None,
    source_z: np.ndarray,
    source_params: TransonicSlimParams,
    target_params: TransonicSlimParams,
) -> tuple[np.ndarray | None, float]:
    if not USE_SECANT_PREDICTOR or previous_z is None or previous_params is None:
        return None, np.inf
    if previous_z.shape != source_z.shape:
        return None, np.inf
    previous_mu = float(np.log(previous_params.Mdot_g_s))
    source_mu = float(np.log(source_params.Mdot_g_s))
    target_mu = float(np.log(target_params.Mdot_g_s))
    denominator = source_mu - previous_mu
    if not np.isfinite(denominator) or abs(denominator) <= 1.0e-14:
        return None, np.inf
    scale = (target_mu - source_mu) / denominator
    candidate = clip_state(source_z + scale * (source_z - previous_z), target_params)
    return candidate, max_residual(candidate, target_params)


def square_polish(z0: np.ndarray, params: TransonicSlimParams, *, pivot: str, max_nfev: int, fallback_lsq: bool):
    newton = solve_square_transonic_polish(
        params,
        z0,
        pivot=pivot,
        method="newton",
        max_iter=NEWTON_MAX_ITER,
        max_nfev=max_nfev,
        residual_tol=1.0e-8,
        use_block_jacobian=NEWTON_USE_BLOCK_JACOBIAN,
        linear_solver=NEWTON_LINEAR_SOLVER,
        line_search_max_reductions=NEWTON_LINE_SEARCH_REDUCTIONS,
        max_step_norm=NEWTON_MAX_STEP_NORM,
    )
    if not fallback_lsq or FALLBACK_LSQ_NFEV <= 0 or max_residual(newton.z, params) <= ACCEPTANCE_TOL:
        return newton
    return solve_square_transonic_polish(
        params,
        newton.z,
        pivot=pivot,
        method="least_squares",
        max_nfev=FALLBACK_LSQ_NFEV,
        residual_tol=1.0e-8,
        use_block_jacobian=LSQ_USE_BLOCK_JACOBIAN,
    )


def integrated_prepolish(z0: np.ndarray, params: TransonicSlimParams):
    """Use integrated defects as a conditioning step, then audit physically."""

    if not USE_INTEGRATED_PREPOLISH:
        return z0, np.nan, np.nan, "-"
    integrated_params = replace(
        params,
        interval_residual_form="integrated",
        integrated_residual_weighting=INTEGRATED_WEIGHTING,
        max_nfev=INTEGRATED_PRE_NFEV,
    )
    best = None
    best_integrated = np.inf
    for pivot in PIVOTS:
        polish = square_polish(
            z0,
            integrated_params,
            pivot=pivot,
            max_nfev=INTEGRATED_PRE_NFEV,
            fallback_lsq=False,
        )
        integrated_square = square_max(polish.z, integrated_params, polish.pivot)
        if integrated_square < best_integrated:
            best = polish
            best_integrated = integrated_square
        if integrated_square <= ACCEPTANCE_TOL:
            break
    if best is None:
        raise RuntimeError("integrated pre-polish had no pivot candidates")
    physical_full = max_residual(best.z, params)
    return best.z, float(best_integrated), float(physical_full), str(best.pivot)


def local_root_result(source_z: np.ndarray, params: TransonicSlimParams):
    fixed_lambda0 = float(params.potential.l_k(params.potential.r_isco) / (params.r_g * C))
    seed_local = local_from_z(source_z, params)
    return solve_local(seed_local, params, names=("D", "C1", "C2", "K"), free_lambda=True, fixed_lambda0=fixed_lambda0)


def sonic_injection_polish(source_z: np.ndarray, params: TransonicSlimParams, *, source_full: float):
    if SONIC_INJECTION_POLICY not in {"always", "if_better", "off"}:
        raise ValueError("IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_INJECTION_POLICY must be always, if_better, or off")
    local_result, local_root = local_root_result(source_z, params)
    z_injected = injected_state(source_z, local_root, params)
    injected_full = max_residual(z_injected, params)
    if SONIC_INJECTION_POLICY == "off":
        seed_z = source_z
        seed_label = "source"
    elif SONIC_INJECTION_POLICY == "if_better" and injected_full > SONIC_INJECTION_ACCEPT_FACTOR * source_full:
        seed_z = source_z
        seed_label = "source"
    else:
        seed_z = z_injected
        seed_label = "injected"
    best = None
    best_full = np.inf
    for pivot in PIVOTS:
        polish = square_polish(seed_z, params, pivot=pivot, max_nfev=POLISH_NFEV, fallback_lsq=True)
        full = max_residual(polish.z, params)
        if full < best_full:
            best = polish
            best_full = full
        if full <= ACCEPTANCE_TOL:
            break
    if best is None:
        raise RuntimeError("no pivot candidates configured")
    return z_injected, injected_full, seed_label, local_result, best


def best_square_polish(z0: np.ndarray, params: TransonicSlimParams, *, max_nfev: int, fallback_lsq: bool):
    best = None
    best_full = np.inf
    for pivot in PIVOTS:
        polish = square_polish(z0, params, pivot=pivot, max_nfev=max_nfev, fallback_lsq=fallback_lsq)
        full = max_residual(polish.z, params)
        if full < best_full:
            best = polish
            best_full = full
        if full <= ANCHOR_TOL:
            break
    if best is None:
        raise RuntimeError("no pivot candidates configured")
    return best


def refresh_outer_slopes(polish, params: TransonicSlimParams) -> tuple[Any, TransonicSlimParams, float]:
    if not pressure_outer_enabled() or OUTER_SLOPE_REFRESHES == 0:
        return polish, params, 0.0
    elapsed = 0.0
    current_polish = polish
    current_params = params
    for _refresh in range(OUTER_SLOPE_REFRESHES):
        refreshed_params = apply_outer_closure_from_state(current_polish.z, current_params)
        old_slopes = np.asarray(
            [np.nan, np.nan] if current_params.outer_match_log_slopes is None else current_params.outer_match_log_slopes,
            dtype=float,
        )
        new_slopes = np.asarray(refreshed_params.outer_match_log_slopes, dtype=float)
        if np.all(np.isfinite(old_slopes)) and float(np.max(np.abs(new_slopes - old_slopes))) <= 1.0e-10:
            current_params = refreshed_params
            break
        if not OUTER_SLOPE_REFRESH_REPOLISH:
            current_params = refreshed_params
            break
        t0 = time.perf_counter()
        current_polish = best_square_polish(
            current_polish.z,
            refreshed_params,
            max_nfev=POLISH_NFEV,
            fallback_lsq=True,
        )
        elapsed += time.perf_counter() - t0
        current_params = refreshed_params
    return current_polish, current_params, elapsed


def classify_step(full: float) -> tuple[str, str]:
    if full <= ANCHOR_TOL:
        return "anchor", "accepted as strong anchor"
    if full <= CURRENT_TOL:
        return "current_scout", "accepted as current-quality scout"
    if full <= ACCEPTANCE_TOL:
        return "accepted_scout", "accepted but above anchor tolerance"
    if full <= SCOUT_TOL:
        return "scout", "scout residual only"
    return "reject", "residual above acceptance tolerance"


def polish_anchor_if_needed(anchor_z: np.ndarray, anchor_params: TransonicSlimParams) -> tuple[np.ndarray, TransonicSlimParams]:
    if not pressure_outer_enabled():
        return anchor_z, anchor_params
    initial_full = max_residual(anchor_z, anchor_params)
    if initial_full <= ANCHOR_TOL:
        return anchor_z, anchor_params
    print(f"pressure outer anchor polish initial={initial_full:.3e}", flush=True)
    polish = best_square_polish(anchor_z, anchor_params, max_nfev=POLISH_NFEV, fallback_lsq=True)
    params = apply_outer_closure_from_state(polish.z, anchor_params)
    polish, params, _elapsed = refresh_outer_slopes(polish, params)
    print(f"pressure outer anchor polish final={max_residual(polish.z, params):.3e}", flush=True)
    return np.asarray(polish.z, dtype=float), params


def save_checkpoint(label: str, z: np.ndarray, params: TransonicSlimParams, row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe = label.replace(".", "p").replace("-", "m")
    slopes = params.outer_match_log_slopes
    np.savez(
        CHECKPOINT_DIR / f"{safe}.npz",
        z=np.asarray(z, dtype=float),
        R_out_rg=np.array(params.R_out_rg),
        n_nodes=np.array(params.n_nodes),
        grid_power=np.array(params.grid_power),
        custom_grid_xi=np.asarray(params.custom_grid_xi, dtype=float)
        if params.custom_grid_xi is not None
        else np.asarray([], dtype=float),
        ratio=np.array(params.mdot_edd_ratio),
        outer_closure=np.array(params.outer_closure),
        outer_closure_mode=np.array(OUTER_CLOSURE_MODE),
        outer_match_log_slopes=np.asarray([np.nan, np.nan] if slopes is None else slopes, dtype=float),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted"]),
        branch=np.array(row["branch"]),
    )


def row_for_attempt(
    *,
    branch: str,
    attempt: int,
    source_ratio: float,
    target_ratio: float,
    step_mu: float,
    tangent_full: float,
    secant_full: float,
    predictor: str,
    predictor_full: float,
    source_full: float,
    integrated_full: float,
    integrated_physical_full: float,
    integrated_pivot: str,
    final_seed: str,
    tangent_s: float,
    integrated_s: float,
    source_s: float,
    inject_polish_s: float,
    injected_full: float,
    z_final: np.ndarray,
    params: TransonicSlimParams,
    polish,
    local_result,
    action: str,
    message: str,
) -> dict[str, object]:
    audit = residual_audit_from_state_vector(z_final, params)
    local = local_from_z(z_final, params)
    sonic = sonic_components(local, params)
    local_all = float(np.max(np.abs(component_vector(local, params, ("D", "C1", "C2", "K")))))
    full = max_residual(z_final, params)
    outer_fields = outer_closure_row_fields(z_final, params)
    return {
        "branch": branch,
        "attempt": int(attempt),
        "action": action,
        "accepted": bool(full <= ACCEPTANCE_TOL),
        "anchor_eligible": bool(full <= ANCHOR_TOL),
        "current_eligible": bool(full <= CURRENT_TOL),
        "source_ratio": float(source_ratio),
        "target_ratio": float(target_ratio),
        "step_mu": float(step_mu),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "grid_power": float(params.grid_power),
        "custom_grid": bool(params.custom_grid_xi is not None),
        "tangent_full": float(tangent_full),
        "secant_full": float(secant_full),
        "predictor": str(predictor),
        "predictor_full": float(predictor_full),
        "source_full": float(source_full),
        "integrated_full": float(integrated_full),
        "integrated_physical_full": float(integrated_physical_full),
        "integrated_pivot": str(integrated_pivot),
        "final_seed": str(final_seed),
        "tangent_s": float(tangent_s),
        "integrated_s": float(integrated_s),
        "source_s": float(source_s),
        "inject_polish_s": float(inject_polish_s),
        "elapsed_s": float(tangent_s + integrated_s + source_s + inject_polish_s),
        "injected_full": float(injected_full),
        "final_full": float(full),
        "square_C1": square_max(z_final, params, "C1"),
        "square_C2": square_max(z_final, params, "C2"),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        **outer_fields,
        "D": float(audit.sonic_D),
        "C1": float(audit.sonic_C1),
        "C2": float(audit.sonic_C2),
        "K": float(audit.sonic_K),
        "local_all": local_all,
        "Rson_rg": float(sonic["Rson_rg"]),
        "lambda0_over_lK_isco": float(sonic["lambda0_over_lK_isco"]),
        "M_eff": float(sonic["M_eff"]),
        "H_R": float(sonic["H_R"]),
        "pivot": str(polish.pivot),
        "polish_method": str(polish.method),
        "nfev": int(polish.result.nfev),
        "optimizer_success": bool(polish.result.optimizer_success),
        "local_success": bool(local_result.success),
        "message": message,
    }


def remap_state_to_n(
    z: np.ndarray,
    params: TransonicSlimParams,
    n_nodes: int,
    fiducial: FiducialParams,
    mdot_edd: float,
) -> tuple[np.ndarray, TransonicSlimParams]:
    target_params = params_for(
        fiducial,
        mdot_edd,
        params.mdot_edd_ratio,
        params.R_out_rg,
        n_nodes,
        grid_power=params.grid_power,
        custom_grid_xi=params.custom_grid_xi if int(n_nodes) == int(params.n_nodes) else None,
    )
    if int(n_nodes) == int(params.n_nodes):
        seed = np.array(z, copy=True)
    else:
        profile = transonic_profile_from_state_vector(z, params)
        seed = remap_profile_to_new_sonic_grid(profile, target_params, temperature_mdot_power=0.0)
    target_params = apply_outer_closure_from_state(seed, target_params)
    return seed, target_params


def adaptive_n_row(
    *,
    base_row: dict[str, object],
    source_n: int,
    target_n: int,
    seed_full: float,
    remap_s: float,
    polish_s: float,
    z_final: np.ndarray,
    params: TransonicSlimParams,
    polish,
) -> dict[str, object]:
    full = max_residual(z_final, params)
    base_action, base_message = classify_step(full)
    action = f"adaptive_n_{base_action}" if base_action != "reject" else "adaptive_n_reject"
    return row_for_attempt(
        branch=str(base_row["branch"]),
        attempt=int(base_row["attempt"]),
        source_ratio=float(base_row["target_ratio"]),
        target_ratio=float(base_row["target_ratio"]),
        step_mu=0.0,
        tangent_full=np.nan,
        secant_full=np.nan,
        predictor=f"adaptive_N{source_n}_to_N{target_n}",
        predictor_full=seed_full,
        source_full=seed_full,
        integrated_full=np.nan,
        integrated_physical_full=np.nan,
        integrated_pivot="-",
        final_seed=f"N{source_n}->N{target_n}",
        tangent_s=0.0,
        integrated_s=0.0,
        source_s=remap_s,
        inject_polish_s=polish_s,
        injected_full=np.nan,
        z_final=z_final,
        params=params,
        polish=polish,
        local_result=SimpleNamespace(success=True),
        action=action,
        message=f"{base_message}; adaptive N retry from N{source_n} to N{target_n}",
    )


def adaptive_n_refinements(
    *,
    base_row: dict[str, object],
    base_z: np.ndarray,
    base_params: TransonicSlimParams,
    fiducial: FiducialParams,
    mdot_edd: float,
) -> tuple[list[tuple[dict[str, object], np.ndarray, TransonicSlimParams]], dict[str, object], np.ndarray, TransonicSlimParams]:
    refinements: list[tuple[dict[str, object], np.ndarray, TransonicSlimParams]] = []
    if not ADAPTIVE_N_VALUES:
        return refinements, base_row, base_z, base_params
    if bool(base_row["anchor_eligible"]) or float(base_row["final_full"]) > ADAPTIVE_N_TRIGGER_TOL:
        return refinements, base_row, base_z, base_params

    best_row = base_row
    best_z = np.asarray(base_z, dtype=float)
    best_params = base_params
    source_z = np.asarray(base_z, dtype=float)
    source_params = base_params
    for n_nodes in ADAPTIVE_N_VALUES:
        if int(n_nodes) <= int(source_params.n_nodes):
            continue
        t0 = time.perf_counter()
        seed, target_params = remap_state_to_n(source_z, source_params, int(n_nodes), fiducial, mdot_edd)
        seed_full = max_residual(seed, target_params)
        remap_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        polish = best_square_polish(seed, target_params, max_nfev=ADAPTIVE_N_NFEV, fallback_lsq=True)
        polish, target_params, _refresh_s = refresh_outer_slopes(polish, target_params)
        polish_s = time.perf_counter() - t1
        row = adaptive_n_row(
            base_row=base_row,
            source_n=int(source_params.n_nodes),
            target_n=int(n_nodes),
            seed_full=seed_full,
            remap_s=remap_s,
            polish_s=polish_s,
            z_final=polish.z,
            params=target_params,
            polish=polish,
        )
        refinements.append((row, np.asarray(polish.z, dtype=float), target_params))
        if bool(row["accepted"]) and float(row["final_full"]) < float(best_row["final_full"]):
            best_row = row
            best_z = np.asarray(polish.z, dtype=float)
            best_params = target_params
        source_z = np.asarray(polish.z, dtype=float)
        source_params = target_params
        if bool(row["anchor_eligible"]):
            break
    return refinements, best_row, best_z, best_params


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Adaptive Mdot Ladder",
        "",
        "Generated by `scripts/run_standard_slim_adaptive_mdot_ladder.py`.",
        "",
        "Each attempt uses an LSMR tangent predictor, C1 source polish, local free-lambda sonic-root injection, and square polish.",
        "",
        f"Config: start step `{START_STEP_MU:g}`, max step `{MAX_STEP_MU:g}`, min step `{MIN_STEP_MU:g}`, acceptance `{ACCEPTANCE_TOL:g}`, anchor `{ANCHOR_TOL:g}`, current `{CURRENT_TOL:g}`, source nfev `{SOURCE_NFEV}`, polish nfev `{POLISH_NFEV}`, LSQ fallback `{FALLBACK_LSQ_NFEV}`, integrated pre-polish `{USE_INTEGRATED_PREPOLISH}`, integrated weighting `{INTEGRATED_WEIGHTING}`, integrated nfev `{INTEGRATED_PRE_NFEV}`, Newton block Jacobian `{NEWTON_USE_BLOCK_JACOBIAN}`, Newton linear solver `{NEWTON_LINEAR_SOLVER}`, Newton max iter `{NEWTON_MAX_ITER}`, Newton max step `{NEWTON_MAX_STEP_NORM}`, LSQ block Jacobian `{LSQ_USE_BLOCK_JACOBIAN}`, sonic injection policy `{SONIC_INJECTION_POLICY}`, secant predictor `{USE_SECANT_PREDICTOR}`, skip final if current `{SKIP_FINAL_IF_CURRENT}`, outer closure `{OUTER_CLOSURE_MODE}`, outer slope refreshes `{OUTER_SLOPE_REFRESHES}`, outer refresh repolish `{OUTER_SLOPE_REFRESH_REPOLISH}`, default grid power `{GRID_POWER:g}`, adaptive N `{ADAPTIVE_N_VALUES}`, adaptive N trigger `{ADAPTIVE_N_TRIGGER_TOL:g}`, adaptive N nfev `{ADAPTIVE_N_NFEV}`.",
        "",
        "| branch | attempt | action | accepted | anchor | current | source ratio | target ratio | step mu | N | grid power | closure | slope | g_u | g_T | pressure mismatch | predictor | pred full | tangent full | secant full | int-pre physical | source full | injected full | final full | dominant | int R | int E | outer omega | Rson/rg | M_eff | H/R | final seed | int pivot | final pivot | method | nfev | elapsed s | message |",
        "|---|---:|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {branch} | {attempt} | {action} | {accepted} | {anchor_eligible} | {current_eligible} | {source_ratio} | {target_ratio} | {step_mu} | "
            "{N} | {grid_power} | {outer_closure_mode} | {outer_slope_source} | {outer_g_u} | {outer_g_T} | {outer_pressure_residual} | "
            "{predictor} | {predictor_full} | {tangent_full} | {secant_full} | {integrated_physical_full} | {source_full} | {injected_full} | {final_full} | {dominant} | {interval_R} | {interval_E} | {outer_omega} | "
            "{Rson_rg} | {M_eff} | {H_R} | {final_seed} | {integrated_pivot} | {pivot} | {polish_method} | {nfev} | {elapsed_s} | {message} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps([{key: json_safe(value) for key, value in row.items()} for row in rows], indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, object]]) -> None:
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
    accepted = [row for row in rows if row["accepted"]]
    all_ratios = [float(row["target_ratio"]) for row in rows]
    all_res = [float(row["final_full"]) for row in rows if float(row["final_full"]) > 0.0]
    x_vals = np.log10(np.asarray(all_ratios, dtype=float))
    y_vals = np.log10(np.maximum(np.asarray(all_res, dtype=float), 1.0e-16))
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    if x_max <= x_min:
        x_min -= 0.05
        x_max += 0.05
    y_min, y_max = float(np.floor(np.min(y_vals))), float(np.ceil(np.max(y_vals)))
    if y_max <= y_min:
        y_max = y_min + 1.0
    colors = {"up": (31, 119, 180), "down": (214, 39, 40)}
    for branch in ("down", "up"):
        selected = [row for row in rows if row["branch"] == branch]
        for row in selected:
            xx = np.log10(float(row["target_ratio"]))
            yy = np.log10(max(float(row["final_full"]), 1.0e-16))
            px = x0 + int((xx - x_min) / (x_max - x_min) * (x1 - x0))
            py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
            color = colors[branch]
            radius = 5 if row["accepted"] else 4
            if row["accepted"]:
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
            else:
                draw.rectangle((px - radius, py - radius, px + radius, py + radius), outline=color, width=2)
    for tol, label in ((ACCEPTANCE_TOL, "accept"), (ANCHOR_TOL, "anchor")):
        yy = np.log10(tol)
        py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
        draw.line((x0, py, x1, py), fill=(120, 120, 120), width=1)
        draw.text((x1 - 70, py - 14), label, fill=(80, 80, 80), font=font)
    draw.text((90, 25), "Adaptive Mdot ladder: final residual vs Mdot/Edd", fill=(20, 20, 20), font=font)
    draw.text((x0 + 4, y0 + 4), f"1e{int(y_max)}", fill=(80, 80, 80), font=font)
    draw.text((x0 + 4, y1 - 18), f"1e{int(y_min)}", fill=(80, 80, 80), font=font)
    draw.text((90, 560), f"accepted points: {len(accepted)} / {len(rows)}", fill=(20, 20, 20), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def attempt_step(
    *,
    branch: str,
    attempt: int,
    source_z: np.ndarray,
    source_params: TransonicSlimParams,
    previous_z: np.ndarray | None,
    previous_params: TransonicSlimParams | None,
    target_params: TransonicSlimParams,
    fiducial: FiducialParams,
    mdot_edd: float,
    step_mu: float,
) -> tuple[dict[str, object], np.ndarray, TransonicSlimParams]:
    phase_log("tangent predictor")
    t0 = time.perf_counter()
    dz_dmu, _tangent_meta = tangent_vector(source_z, source_params, pivot=TANGENT_PIVOT)
    z_tangent, _meta = tangent_predictor(source_z, source_params, target_params, dz_dmu)
    z_tangent = clip_state(z_tangent, target_params)
    tangent_full = max_residual(z_tangent, target_params)
    z_secant, secant_full = secant_predictor(previous_z, previous_params, source_z, source_params, target_params)
    if z_secant is not None and secant_full < tangent_full:
        z_predictor = z_secant
        predictor = "secant"
        predictor_full = secant_full
    else:
        z_predictor = z_tangent
        predictor = "tangent"
        predictor_full = tangent_full
    tangent_s = time.perf_counter() - t0
    phase_log(f"predictor done kind={predictor} full={predictor_full:.3e} t={tangent_s:.2f}s")
    phase_log("integrated pre-polish")
    t0 = time.perf_counter()
    z_pre, integrated_full, integrated_physical_full, integrated_pivot = integrated_prepolish(z_predictor, target_params)
    integrated_s = time.perf_counter() - t0
    phase_log(f"integrated done physical={integrated_physical_full:.3e} t={integrated_s:.2f}s")
    phase_log("C1 source polish")
    t0 = time.perf_counter()
    source_polish = square_polish(z_pre, target_params, pivot="C1", max_nfev=SOURCE_NFEV, fallback_lsq=False)
    source_full = max_residual(source_polish.z, target_params)
    source_s = time.perf_counter() - t0
    phase_log(f"source done full={source_full:.3e} t={source_s:.2f}s")
    if SKIP_FINAL_IF_CURRENT and source_full <= CURRENT_TOL:
        z_injected = source_polish.z
        injected_full = np.nan
        final_seed = "source_skip"
        local_result = SimpleNamespace(success=True)
        final_polish = source_polish
        final_full = source_full
        inject_polish_s = 0.0
        phase_log("final skipped because source is current-quality")
    else:
        phase_log("sonic injection + final polish")
        t0 = time.perf_counter()
        z_injected, injected_full, final_seed, local_result, final_polish = sonic_injection_polish(
            source_polish.z,
            target_params,
            source_full=source_full,
        )
        final_full = max_residual(final_polish.z, target_params)
        inject_polish_s = time.perf_counter() - t0
        phase_log(f"final done injected={injected_full:.3e} final={final_full:.3e} t={inject_polish_s:.2f}s")
    phase_log("outer slope refresh")
    final_polish, target_params, refresh_s = refresh_outer_slopes(final_polish, target_params)
    inject_polish_s += refresh_s
    final_full = max_residual(final_polish.z, target_params)
    if refresh_s > 0.0:
        phase_log(f"outer slope refresh done final={final_full:.3e} t={refresh_s:.2f}s")
    action, message = classify_step(final_full)
    row = row_for_attempt(
        branch=branch,
        attempt=attempt,
        source_ratio=source_params.mdot_edd_ratio,
        target_ratio=target_params.mdot_edd_ratio,
        step_mu=step_mu,
        tangent_full=tangent_full,
        secant_full=secant_full,
        predictor=predictor,
        predictor_full=predictor_full,
        source_full=source_full,
        integrated_full=integrated_full,
        integrated_physical_full=integrated_physical_full,
        integrated_pivot=integrated_pivot,
        final_seed=final_seed,
        tangent_s=tangent_s,
        integrated_s=integrated_s,
        source_s=source_s,
        inject_polish_s=inject_polish_s,
        injected_full=injected_full,
        z_final=final_polish.z,
        params=target_params,
        polish=final_polish,
        local_result=local_result,
        action=action,
        message=message,
    )
    _ = fiducial, mdot_edd, z_injected
    return row, final_polish.z, target_params


def next_ratio(current: float, target: float, step_mu: float, direction: int) -> float:
    current_mu = float(np.log(current))
    target_mu = float(np.log(target))
    trial_mu = current_mu + direction * step_mu
    if direction > 0:
        trial_mu = min(trial_mu, target_mu)
    else:
        trial_mu = max(trial_mu, target_mu)
    return float(np.exp(trial_mu))


def run_branch(
    *,
    branch: str,
    target_ratio: float,
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    fiducial: FiducialParams,
    mdot_edd: float,
    rows: list[dict[str, object]],
) -> None:
    direction = 1 if target_ratio > anchor_params.mdot_edd_ratio else -1
    current_z = np.asarray(anchor_z, dtype=float)
    current_params = apply_outer_closure_from_state(current_z, anchor_params)
    history: list[tuple[np.ndarray, TransonicSlimParams]] = [(np.asarray(anchor_z, dtype=float), current_params)]
    step_mu = min(abs(float(np.log(target_ratio / anchor_params.mdot_edd_ratio))), START_STEP_MU)
    attempt = 0
    while attempt < MAX_ATTEMPTS_PER_BRANCH:
        current_ratio = current_params.mdot_edd_ratio
        if (direction > 0 and current_ratio >= target_ratio * (1.0 - 1.0e-12)) or (
            direction < 0 and current_ratio <= target_ratio * (1.0 + 1.0e-12)
        ):
            break
        if step_mu < MIN_STEP_MU:
            print(f"{branch}: stopping, step_mu={step_mu:g} below minimum", flush=True)
            break
        ratio = next_ratio(current_ratio, target_ratio, step_mu, direction)
        target_base = params_for(
            fiducial,
            mdot_edd,
            ratio,
            current_params.R_out_rg,
            current_params.n_nodes,
            grid_power=current_params.grid_power,
            custom_grid_xi=current_params.custom_grid_xi,
        )
        target_params = apply_outer_closure_from_state(current_z, target_base)
        attempt += 1
        print(f"{branch} attempt={attempt} {current_ratio:.7g}->{ratio:.7g} step={step_mu:.4g}", flush=True)
        row, z_candidate, candidate_params = attempt_step(
            branch=branch,
            attempt=attempt,
            source_z=current_z,
            source_params=current_params,
            previous_z=history[-2][0] if len(history) >= 2 else None,
            previous_params=history[-2][1] if len(history) >= 2 else None,
            target_params=target_params,
            fiducial=fiducial,
            mdot_edd=mdot_edd,
            step_mu=step_mu,
        )
        rows.append(row)
        write_table(rows)
        write_figure(rows)
        print(
            f"  {row['action']} final={row['final_full']:.3e} dom={row['dominant']} "
            f"tangent={row['tangent_full']:.3e} source={row['source_full']:.3e}",
            flush=True,
        )
        refinement_rows, selected_row, selected_z, selected_params = adaptive_n_refinements(
            base_row=row,
            base_z=z_candidate,
            base_params=candidate_params,
            fiducial=fiducial,
            mdot_edd=mdot_edd,
        )
        for refinement_row, _refinement_z, _refinement_params in refinement_rows:
            rows.append(refinement_row)
            write_table(rows)
            write_figure(rows)
            print(
                f"  {refinement_row['action']} final={refinement_row['final_full']:.3e} "
                f"N={refinement_row['N']} dom={refinement_row['dominant']} "
                f"seed={refinement_row['source_full']:.3e}",
                flush=True,
            )
        row = selected_row
        z_candidate = selected_z
        candidate_params = selected_params
        reached_target = abs(float(np.log(ratio / target_ratio))) < 1.0e-12
        if row["accepted"]:
            candidate_params = apply_outer_closure_from_state(z_candidate, candidate_params)
            save_checkpoint(f"{branch}_mdot_{ratio:g}", z_candidate, candidate_params, row)
            if reached_target:
                break
            if row["anchor_eligible"]:
                step_mu = min(MAX_STEP_MU, step_mu * GROW_FACTOR)
            elif row["current_eligible"]:
                step_mu = min(MAX_STEP_MU, max(MIN_STEP_MU, step_mu * 1.05))
            else:
                step_mu = max(MIN_STEP_MU, step_mu * 0.8)
            if row["anchor_eligible"] or row["current_eligible"] or USE_SCOUT_AS_CURRENT:
                current_z = z_candidate
                current_params = candidate_params
                history.append((np.asarray(z_candidate, dtype=float), candidate_params))
            else:
                step_mu *= SHRINK_FACTOR
        else:
            step_mu *= SHRINK_FACTOR


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = load_anchor(fiducial, mdot_edd)
    anchor_z, anchor_params = polish_anchor_if_needed(anchor_z, anchor_params)
    anchor_audit = residual_audit_from_state_vector(anchor_z, anchor_params)
    anchor_full = max_residual(anchor_z, anchor_params)
    anchor_outer = outer_closure_row_fields(anchor_z, anchor_params)
    rows: list[dict[str, object]] = [
        {
            "branch": "anchor",
            "attempt": 0,
            "action": "anchor",
            "accepted": bool(anchor_full <= ACCEPTANCE_TOL),
            "anchor_eligible": bool(anchor_full <= ANCHOR_TOL),
            "current_eligible": bool(anchor_full <= CURRENT_TOL),
            "source_ratio": float(anchor_params.mdot_edd_ratio),
            "target_ratio": float(anchor_params.mdot_edd_ratio),
            "step_mu": 0.0,
            "R_out_rg": float(anchor_params.R_out_rg),
            "N": int(anchor_params.n_nodes),
            "grid_power": float(anchor_params.grid_power),
            "tangent_full": np.nan,
            "secant_full": np.nan,
            "predictor": "-",
            "predictor_full": np.nan,
            "source_full": np.nan,
            "integrated_full": np.nan,
            "integrated_physical_full": np.nan,
            "integrated_pivot": "-",
            "final_seed": "-",
            "tangent_s": 0.0,
            "integrated_s": 0.0,
            "source_s": 0.0,
            "inject_polish_s": 0.0,
            "elapsed_s": 0.0,
            "injected_full": np.nan,
            "final_full": anchor_full,
            "square_C1": square_max(anchor_z, anchor_params, "C1"),
            "square_C2": square_max(anchor_z, anchor_params, "C2"),
            "dominant": dominant(anchor_audit),
            "interval_R": float(anchor_audit.interval_radial_max),
            "interval_E": float(anchor_audit.interval_energy_max),
            "outer_omega": float(anchor_audit.outer_omega),
            "outer_energy": float(anchor_audit.outer_energy),
            **anchor_outer,
            "D": float(anchor_audit.sonic_D),
            "C1": float(anchor_audit.sonic_C1),
            "C2": float(anchor_audit.sonic_C2),
            "K": float(anchor_audit.sonic_K),
            "local_all": np.nan,
            "Rson_rg": float(np.exp(local_from_z(anchor_z, anchor_params)[2]) / anchor_params.r_g),
            "lambda0_over_lK_isco": np.nan,
            "M_eff": np.nan,
            "H_R": np.nan,
            "pivot": "-",
            "polish_method": "-",
            "nfev": 0,
            "optimizer_success": True,
            "local_success": True,
            "message": "input anchor checkpoint",
        }
    ]
    write_table(rows)
    run_branch(
        branch="down",
        target_ratio=DOWN_TARGET,
        anchor_z=anchor_z,
        anchor_params=anchor_params,
        fiducial=fiducial,
        mdot_edd=mdot_edd,
        rows=rows,
    )
    run_branch(
        branch="up",
        target_ratio=UP_TARGET,
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
