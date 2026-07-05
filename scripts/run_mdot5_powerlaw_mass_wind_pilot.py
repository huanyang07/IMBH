"""Pilot mass-coupled power-law wind homotopy from the Mdot=5 energy-wind branch."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_standard_slim_stream_mass_annulus_scan as scan  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402


DEFAULT_ANCHOR = (
    ROOT
    / "outputs/checkpoints/m5_energy_wind_eta_adaptive_manual_6425_N896/"
    "eta_adaptive_manual_6425_mass_0p8_wind_0_heat_0_ewind_0p998379467_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz"
)
ANCHOR = Path(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_ANCHOR", str(DEFAULT_ANCHOR))).expanduser()
if not ANCHOR.is_absolute():
    ANCHOR = ROOT / ANCHOR
SLOPE_DIAGNOSTIC = ROOT / "outputs/tables/m5_energy_wind_powerlaw_slope_diagnostics.json"
OUTPUT_STEM = os.environ.get("IMBH_MDOT5_POWERLAW_WIND_OUTPUT_STEM", "m5_energy_wind_powerlaw_mass_coupled_pilot").strip()
JSON_OUTPUT = ROOT / f"outputs/tables/{OUTPUT_STEM}.json"
MD_OUTPUT = ROOT / f"outputs/tables/{OUTPUT_STEM}.md"
CHECKPOINT_DIR = ROOT / f"outputs/checkpoints/{OUTPUT_STEM}"

COUPLING_STRENGTHS_RAW = os.environ.get(
    "IMBH_MDOT5_POWERLAW_WIND_COUPLING_STRENGTHS",
    "0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.35,0.5,0.75,1.0",
)
N_NODES_OVERRIDE_RAW = os.environ.get("IMBH_MDOT5_POWERLAW_WIND_N_NODES", "").strip()
REMAP_METHOD = os.environ.get("IMBH_MDOT5_POWERLAW_WIND_REMAP_METHOD", "linear").strip().lower()
USE_SECANT_PREDICTOR = os.environ.get("IMBH_MDOT5_POWERLAW_WIND_USE_SECANT_PREDICTOR", "1") != "0"
USE_TANGENT_PREDICTOR = os.environ.get("IMBH_MDOT5_POWERLAW_WIND_USE_TANGENT_PREDICTOR", "0") != "0"
TANGENT_TRIGGER_INITIAL_FULL = float(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_TANGENT_TRIGGER_INITIAL_FULL", "0.02"))
TANGENT_FD_ZETA_STEP = float(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_TANGENT_FD_ZETA_STEP", "1e-5"))
PREDICTOR_DAMPING_VALUES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_MDOT5_POWERLAW_WIND_PREDICTOR_DAMPINGS", "1,0.5,0.25,0.1,0.05").split(",")
    if piece.strip()
)
INCLUDE_OUTER_OMEGA_SEED = os.environ.get("IMBH_MDOT5_POWERLAW_WIND_INCLUDE_OUTER_OMEGA_SEED", "0") != "0"
ADAPTIVE_TARGET_RAW = os.environ.get("IMBH_MDOT5_POWERLAW_WIND_ADAPTIVE_TARGET", "").strip()
ADAPTIVE_INITIAL_STEP = float(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_ADAPTIVE_INITIAL_STEP", "0.002"))
ADAPTIVE_MIN_STEP = float(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_ADAPTIVE_MIN_STEP", "0.0005"))
ADAPTIVE_MAX_STEP = float(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_ADAPTIVE_MAX_STEP", "0.005"))
ADAPTIVE_MAX_INITIAL_FULL = float(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_ADAPTIVE_MAX_INITIAL_FULL", "0.02"))
ADAPTIVE_GROWTH = float(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_ADAPTIVE_GROWTH", "1.35"))
ADAPTIVE_SHRINK = float(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_ADAPTIVE_SHRINK", "0.5"))
ADAPTIVE_COST_GROW_NFEV = int(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_ADAPTIVE_COST_GROW_NFEV", "4"))
ADAPTIVE_COST_SHRINK_NFEV = int(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_ADAPTIVE_COST_SHRINK_NFEV", "8"))
ADAPTIVE_MAX_REJECTIONS = int(os.environ.get("IMBH_MDOT5_POWERLAW_WIND_ADAPTIVE_MAX_REJECTIONS", "16"))


def _parse_coupling_strengths() -> tuple[float, ...]:
    values = tuple(float(piece) for piece in COUPLING_STRENGTHS_RAW.replace(":", ",").split(",") if piece.strip())
    if not values:
        raise ValueError("at least one coupling strength is required")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("coupling strengths must lie in [0, 1]")
    return values


def _adaptive_target() -> float | None:
    if not ADAPTIVE_TARGET_RAW:
        return None
    target = float(ADAPTIVE_TARGET_RAW)
    if not 0.0 <= target <= 1.0:
        raise ValueError("adaptive target must lie in [0, 1]")
    return target


def _next_adaptive_step(step: float, accepted: bool, initial_full: float, nfev: int) -> float:
    if accepted and nfev <= ADAPTIVE_COST_GROW_NFEV and initial_full <= 0.5 * ADAPTIVE_MAX_INITIAL_FULL:
        return min(ADAPTIVE_MAX_STEP, max(ADAPTIVE_MIN_STEP, step * ADAPTIVE_GROWTH))
    if (not accepted) or nfev >= ADAPTIVE_COST_SHRINK_NFEV or initial_full > ADAPTIVE_MAX_INITIAL_FULL:
        return max(ADAPTIVE_MIN_STEP, step * ADAPTIVE_SHRINK)
    return min(ADAPTIVE_MAX_STEP, max(ADAPTIVE_MIN_STEP, step))


def _format(value: Any) -> str:
    if isinstance(value, str):
        return value
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


def _load_reference() -> dict[str, float]:
    rows = json.loads(SLOPE_DIAGNOSTIC.read_text())
    for row in rows:
        if row.get("label") == "eta_6p425":
            return {
                "full_implied_mwind_over_inner": float(row["implied_Mwind_over_inner_etaesc1"]),
                "active_inner_rg": float(row["active_R_min_rg"]),
                "reference_epsilon_w": float(row["epsilon_w"]),
                "reference_Qwind_Qvisc": float(row["Qwind_Qvisc"]),
            }
    raise ValueError(f"missing eta_6p425 in {SLOPE_DIAGNOSTIC}")


def _powerlaw_s_for_fraction(wind_fraction: float, inner_rg: float, outer_rg: float) -> float:
    if wind_fraction < 0.0:
        raise ValueError("wind_fraction must be non-negative")
    if inner_rg <= 0.0 or outer_rg <= inner_rg:
        raise ValueError("power-law radii must be positive and increasing")
    if wind_fraction == 0.0:
        return 0.0
    return float(np.log1p(wind_fraction) / np.log(outer_rg / inner_rg))


def _params_for_zeta(base_params, zeta: float, full_mwind: float, inner_rg: float):
    wind_fraction = float(zeta * full_mwind)
    powerlaw_s = _powerlaw_s_for_fraction(wind_fraction, inner_rg, base_params.R_out_rg)
    return (
        replace(
            base_params,
            wind_sink_fraction=wind_fraction,
            wind_sink_shape="powerlaw",
            wind_sink_powerlaw_inner_rg=inner_rg,
            wind_sink_powerlaw_s=powerlaw_s,
        ),
        wind_fraction,
        powerlaw_s,
    )


def _write_checkpoint(row: dict[str, Any], params) -> str:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_zeta = f"{float(row['wind_mass_coupling_strength']):.5g}".replace(".", "p").replace("-", "m")
    path = CHECKPOINT_DIR / f"zeta_{safe_zeta}_N{int(params.n_nodes)}.npz"
    slopes = params.outer_match_log_slopes
    np.savez_compressed(
        path,
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
        wind_sink_shape=np.array(params.wind_sink_shape),
        wind_sink_powerlaw_inner_rg=np.array(params.wind_sink_powerlaw_inner_rg),
        wind_sink_powerlaw_s=np.array(params.wind_sink_powerlaw_s),
        wind_energy_limited_epsilon=np.array(params.wind_energy_limited_epsilon),
        wind_eddington_chi=np.array(params.wind_eddington_chi),
        wind_activation_width_fraction=np.array(params.wind_activation_width_fraction),
        stream_heating_efficiency=np.array(params.stream_heating_efficiency),
        interval_residual_form=np.array(params.interval_residual_form),
        integrated_residual_weighting=np.array(params.integrated_residual_weighting),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted"]),
        branch=np.array(row["branch"]),
        row_json=np.array(json.dumps(scan.json_safe({key: value for key, value in row.items() if key != "z"}), sort_keys=True)),
    )
    return str(path.relative_to(ROOT))


def _write_outputs(rows: list[dict[str, Any]]) -> None:
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(scan.json_safe(rows), indent=2, sort_keys=True) + "\n")
    columns = [
        "wind_mass_coupling_strength",
        "wind_sink_fraction",
        "wind_sink_powerlaw_s",
        "Mdot_outer_over_inner",
        "wind_sink_integral_over_inner",
        "integrated_Qwind_Qvisc",
        "initial_full",
        "initial_full_current_seed",
        "initial_full_mass_compensated_seed",
        "initial_full_stress_ratio_seed",
        "initial_full_mass_comp_omega_seed",
        "seed_strategy",
        "final_full",
        "accepted",
        "physical_E_gate_eligible",
        "dominant",
        "partition_physical_E",
        "outer_omega",
        "f_adv_global",
        "Lrad_LEdd",
        "Rson_rg",
        "polish_nfev_total",
        "checkpoint",
    ]
    lines = [
        "# Mdot=5 Power-Law Mass-Coupled Wind Pilot",
        "",
        "Generated by `scripts/run_mdot5_powerlaw_mass_wind_pilot.py`.",
        "",
        "This is a prescribed power-law wind-mass homotopy, not yet a fully local",
        "`Qwind -> dMdot/dlnR` solved field. The total wind mass is scaled from the",
        "`eta=6.425` energy-wind post-processing estimate, while the radial distribution",
        "is normalized as a power law.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |")
    MD_OUTPUT.write_text("\n".join(lines) + "\n")


def _mass_compensated_seed(z: np.ndarray, old_params, new_params) -> np.ndarray:
    """Scale ``u`` so the initial surface-density profile is approximately preserved."""

    logu, logT, logR_son, lambda0, logR = scan.unpack_state(z, old_params)
    mdot_old = np.asarray([scan.stream_mass_rate_and_derivative(float(x), old_params)[0] for x in logR], dtype=float)
    mdot_new = np.asarray([scan.stream_mass_rate_and_derivative(float(x), new_params)[0] for x in logR], dtype=float)
    if np.any(mdot_old <= 0.0) or np.any(mdot_new <= 0.0):
        raise ValueError("mass-compensated seed requires positive Mdot profiles")
    return scan.pack_state(logu + np.log(mdot_new / mdot_old), logT, logR_son, lambda0)


def _stress_ratio_compensated_seed(z: np.ndarray, old_params, new_params) -> np.ndarray:
    """Adjust ``u`` so ``W/Mdot`` starts close to the old angular profile."""

    logu, logT, logR_son, lambda0, logR = scan.unpack_state(z, old_params)
    repaired_logu = np.asarray(logu, dtype=float).copy()
    fallback = _mass_compensated_seed(z, old_params, new_params)
    fallback_logu = scan.unpack_state(fallback, new_params)[0]
    lower, upper = map(float, new_params.logu_bounds)
    for idx, x in enumerate(logR):
        old_state = scan.algebraic_state(float(x), float(logu[idx]), float(logT[idx]), lambda0, old_params)
        old_mdot = scan.stream_mass_rate_and_derivative(float(x), old_params)[0]
        new_mdot = scan.stream_mass_rate_and_derivative(float(x), new_params)[0]
        target = old_state.W / old_mdot

        def residual(logu_trial: float) -> float:
            state = scan.algebraic_state(float(x), float(logu_trial), float(logT[idx]), lambda0, new_params)
            return float(state.W / new_mdot - target)

        lo = lower
        hi = upper
        try:
            f_lo = residual(lo)
            f_hi = residual(hi)
        except Exception:
            repaired_logu[idx] = fallback_logu[idx]
            continue
        if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0.0:
            repaired_logu[idx] = fallback_logu[idx]
            continue
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = residual(mid)
            if not np.isfinite(f_mid):
                break
            if abs(f_mid) <= 1.0e-10 * max(abs(target), 1.0):
                lo = hi = mid
                break
            if f_lo * f_mid <= 0.0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        repaired_logu[idx] = 0.5 * (lo + hi)
    return scan.pack_state(repaired_logu, logT, logR_son, lambda0)


def _outer_omega_corrected_seed(z: np.ndarray, params, iterations: int = 2) -> tuple[np.ndarray, Any]:
    """Adjust ``lambda0`` to reduce the outer angular residual for a seed."""

    repaired = np.asarray(z, dtype=float)
    repaired_params = params
    for _ in range(max(0, int(iterations))):
        audit = scan.residual_audit_from_state_vector(repaired, repaired_params)
        omega_residual = float(audit.outer_omega)
        if not np.isfinite(omega_residual) or abs(omega_residual) < 1.0e-12:
            break
        logu, logT, logR_son, lambda0, logR = scan.unpack_state(repaired, repaired_params)
        state = scan.algebraic_state(float(logR[-1]), float(logu[-1]), float(logT[-1]), lambda0, repaired_params)
        delta_omega = state.Omega * (np.exp(-omega_residual) - 1.0)
        delta_lambda0 = state.R**2 * delta_omega / (repaired_params.r_g * C)
        repaired = scan.pack_state(logu, logT, logR_son, lambda0 + float(delta_lambda0))
        repaired_params = scan.apply_outer_slopes_from_state(repaired, repaired_params)
    return repaired, repaired_params


def _candidate_row(seed_kind: str, z: np.ndarray, params) -> tuple[str, np.ndarray, Any, float]:
    candidate_params = scan.apply_outer_slopes_from_state(z, params)
    return seed_kind, np.asarray(z, dtype=float), candidate_params, scan.max_residual(z, candidate_params)


def _finite_difference_zeta_column(
    anchor_z: np.ndarray,
    anchor_params,
    *,
    zeta0: float,
    full_mwind: float,
    inner_rg: float,
    pivot: str,
) -> tuple[np.ndarray, float]:
    step = min(abs(float(TANGENT_FD_ZETA_STEP)), 0.25 * max(float(zeta0), 1.0e-5), 0.25 * max(1.0 - float(zeta0), 1.0e-3))
    if step <= 0.0:
        raise ValueError("zeta finite-difference step collapsed")
    if zeta0 - step >= 0.0 and zeta0 + step <= 1.0:
        plus, _wf_plus, _s_plus = _params_for_zeta(anchor_params, zeta0 + step, full_mwind, inner_rg)
        minus, _wf_minus, _s_minus = _params_for_zeta(anchor_params, zeta0 - step, full_mwind, inner_rg)
        f_plus = scan.square_collocation_residual(anchor_z, plus, pivot=pivot)
        f_minus = scan.square_collocation_residual(anchor_z, minus, pivot=pivot)
        return (f_plus - f_minus) / (2.0 * step), step
    shifted_zeta = zeta0 + step if zeta0 + step <= 1.0 else zeta0 - step
    sign = 1.0 if shifted_zeta > zeta0 else -1.0
    shifted, _wf_shifted, _s_shifted = _params_for_zeta(anchor_params, shifted_zeta, full_mwind, inner_rg)
    f_base = scan.square_collocation_residual(anchor_z, anchor_params, pivot=pivot)
    f_shifted = scan.square_collocation_residual(anchor_z, shifted, pivot=pivot)
    return sign * (f_shifted - f_base) / step, step


def _zeta_tangent(anchor_z: np.ndarray, anchor_params, *, zeta0: float, full_mwind: float, inner_rg: float):
    pivot = scan.PIVOTS[0] if scan.PIVOTS else "C2"
    jac = scan.square_collocation_jacobian(anchor_z, anchor_params, pivot=pivot)
    f_zeta, fd_step = _finite_difference_zeta_column(
        anchor_z,
        anchor_params,
        zeta0=zeta0,
        full_mwind=full_mwind,
        inner_rg=inner_rg,
        pivot=pivot,
    )
    dz_dzeta = scan.equilibrated_tangent_solve(jac, -f_zeta)
    linear_residual = np.asarray(jac @ dz_dzeta + f_zeta, dtype=float)
    return dz_dzeta, {
        "predictor_tangent_fd_zeta_step": float(fd_step),
        "predictor_tangent_solver": str(scan.TANGENT_SOLVER),
        "predictor_tangent_linear_damping": float(scan.TANGENT_LINEAR_DAMPING),
        "predictor_tangent_norm_inf": float(np.linalg.norm(dz_dzeta, ord=np.inf)),
        "predictor_tangent_norm_l2": float(np.linalg.norm(dz_dzeta)),
        "predictor_tangent_linear_residual_norm": float(np.linalg.norm(linear_residual)),
        "predictor_tangent_linear_residual_inf": float(np.linalg.norm(linear_residual, ord=np.inf)),
    }


def _infer_zeta(params, full_mwind: float) -> float:
    if full_mwind <= 0.0:
        return 0.0
    shape = str(getattr(params, "wind_sink_shape", "")).strip().lower()
    if shape != "powerlaw":
        return 0.0
    return max(0.0, min(1.0, float(params.wind_sink_fraction) / float(full_mwind)))


def _seed_candidates(
    *,
    current_z: np.ndarray,
    current_params,
    target_params,
    target_zeta: float,
    current_zeta: float,
    prev_zeta: float | None,
    prev_z: np.ndarray | None,
    full_mwind: float,
    inner_rg: float,
) -> tuple[list[tuple[str, np.ndarray, Any, float]], dict[str, Any]]:
    candidates = [
        _candidate_row("current", current_z, target_params),
    ]
    compensated_seed = _mass_compensated_seed(current_z, current_params, target_params)
    candidates.append(_candidate_row("mass_compensated_u", compensated_seed, target_params))
    stress_seed = _stress_ratio_compensated_seed(current_z, current_params, target_params)
    candidates.append(_candidate_row("stress_ratio_compensated_u", stress_seed, target_params))

    omega_seed, omega_params = _outer_omega_corrected_seed(compensated_seed, scan.apply_outer_slopes_from_state(compensated_seed, target_params))
    omega_row = _candidate_row("mass_compensated_outer_omega", omega_seed, omega_params)
    if INCLUDE_OUTER_OMEGA_SEED:
        candidates.append(omega_row)

    diagnostics: dict[str, Any] = {
        "predictor_initial_full_current_seed": float(candidates[0][3]),
        "predictor_initial_full_mass_compensated_seed": float(candidates[1][3]),
        "predictor_initial_full_stress_ratio_seed": float(candidates[2][3]),
        "predictor_initial_full_mass_comp_omega_seed": float(omega_row[3]),
        "predictor_initial_full_secant_best": np.nan,
        "predictor_secant_damping_chosen": np.nan,
        "predictor_secant_clip_count_best": 0,
        "predictor_initial_full_tangent_best": np.nan,
        "predictor_tangent_damping_chosen": np.nan,
        "predictor_tangent_clip_count_best": 0,
        "predictor_tangent_error": "",
        "predictor_tangent_secant_cosine": np.nan,
    }

    secant_direction: np.ndarray | None = None
    if (
        USE_SECANT_PREDICTOR
        and prev_z is not None
        and prev_zeta is not None
        and abs(float(current_zeta) - float(prev_zeta)) > 1.0e-14
    ):
        step_factor = (float(target_zeta) - float(current_zeta)) / (float(current_zeta) - float(prev_zeta))
        secant_direction = step_factor * (np.asarray(current_z, dtype=float) - np.asarray(prev_z, dtype=float))
        for damping in PREDICTOR_DAMPING_VALUES:
            trial_seed, clip_count = scan.clip_state_with_count(current_z + float(damping) * secant_direction, target_params)
            row = _candidate_row(f"secant:{float(damping):g}", trial_seed, target_params)
            candidates.append(row)
            if not np.isfinite(diagnostics["predictor_initial_full_secant_best"]) or row[3] < diagnostics[
                "predictor_initial_full_secant_best"
            ]:
                diagnostics["predictor_initial_full_secant_best"] = float(row[3])
                diagnostics["predictor_secant_damping_chosen"] = float(damping)
                diagnostics["predictor_secant_clip_count_best"] = int(clip_count)

    best_full = min(row[3] for row in candidates)
    if USE_TANGENT_PREDICTOR and best_full > TANGENT_TRIGGER_INITIAL_FULL:
        try:
            anchor_params = scan.apply_outer_slopes_from_state(current_z, current_params)
            dz_dzeta, tangent_info = _zeta_tangent(
                current_z,
                anchor_params,
                zeta0=current_zeta,
                full_mwind=full_mwind,
                inner_rg=inner_rg,
            )
            diagnostics.update(tangent_info)
            dzeta = float(target_zeta) - float(current_zeta)
            if secant_direction is not None:
                tangent_direction = dzeta * dz_dzeta
                denom = float(np.linalg.norm(secant_direction) * np.linalg.norm(tangent_direction))
                diagnostics["predictor_tangent_secant_cosine"] = (
                    float(np.dot(secant_direction, tangent_direction) / denom) if denom > 0.0 else np.nan
                )
            for damping in PREDICTOR_DAMPING_VALUES:
                trial_seed, clip_count = scan.clip_state_with_count(current_z + float(damping) * dzeta * dz_dzeta, target_params)
                row = _candidate_row(f"tangent:{float(damping):g}", trial_seed, target_params)
                candidates.append(row)
                if not np.isfinite(diagnostics["predictor_initial_full_tangent_best"]) or row[3] < diagnostics[
                    "predictor_initial_full_tangent_best"
                ]:
                    diagnostics["predictor_initial_full_tangent_best"] = float(row[3])
                    diagnostics["predictor_tangent_damping_chosen"] = float(damping)
                    diagnostics["predictor_tangent_clip_count_best"] = int(clip_count)
        except Exception as exc:
            diagnostics["predictor_tangent_error"] = str(exc)
            print(f"  zeta tangent predictor unavailable: {exc}", flush=True)
    return candidates, diagnostics


def main() -> None:
    if not ANCHOR.exists():
        raise FileNotFoundError(ANCHOR)
    reference = _load_reference()
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = scan.load_anchor(ANCHOR, fiducial, mdot_edd)
    if N_NODES_OVERRIDE_RAW:
        n_nodes = int(N_NODES_OVERRIDE_RAW)
        target_params = replace(anchor_params, n_nodes=n_nodes, custom_grid_xi=None)
        profile = scan.transonic_profile_from_state_vector(anchor_z, anchor_params)
        anchor_z = scan.remap_profile_to_new_sonic_grid(
            profile,
            target_params,
            temperature_mdot_power=0.0,
            method=REMAP_METHOD,
        )
        anchor_params = scan.apply_outer_slopes_from_state(anchor_z, target_params)
        print(
            f"remapped anchor to N={n_nodes} method={REMAP_METHOD} "
            f"initial_full={scan.max_residual(anchor_z, anchor_params):.3e}",
            flush=True,
        )
    full_mwind = float(reference["full_implied_mwind_over_inner"])
    inner_rg = max(2.05, float(reference["active_inner_rg"]))
    current_z = np.asarray(anchor_z, dtype=float)
    current_params = anchor_params
    current_zeta = _infer_zeta(current_params, full_mwind=full_mwind)
    prev_zeta: float | None = None
    prev_z: np.ndarray | None = None
    rows: list[dict[str, Any]] = []

    print(
        f"anchor={scan.relative_root_path(ANCHOR)} current_zeta={current_zeta:.6g} "
        f"full_implied_Mwind/Min={full_mwind:.6g} powerlaw_inner_rg={inner_rg:.6g}",
        flush=True,
    )

    adaptive_target = _adaptive_target()
    adaptive_step = min(ADAPTIVE_MAX_STEP, max(ADAPTIVE_MIN_STEP, ADAPTIVE_INITIAL_STEP))
    adaptive_rejections = 0
    fixed_targets = () if adaptive_target is not None else _parse_coupling_strengths()
    fixed_index = 0
    if adaptive_target is not None:
        if adaptive_target < current_zeta:
            raise ValueError("adaptive target must be greater than or equal to the anchor zeta")
        print(
            f"adaptive target={adaptive_target:.6g} step0={adaptive_step:.6g} "
            f"step_min={ADAPTIVE_MIN_STEP:.6g} step_max={ADAPTIVE_MAX_STEP:.6g} "
            f"max_initial={ADAPTIVE_MAX_INITIAL_FULL:.3e}",
            flush=True,
        )

    while True:
        if adaptive_target is None:
            if fixed_index >= len(fixed_targets):
                break
            zeta = float(fixed_targets[fixed_index])
            fixed_index += 1
        else:
            remaining = float(adaptive_target) - float(current_zeta)
            if remaining <= 1.0e-12:
                break
            zeta = float(current_zeta) + min(float(adaptive_step), remaining)
        trial_params, wind_fraction, powerlaw_s = _params_for_zeta(current_params, zeta, full_mwind, inner_rg)
        trial_params = scan.apply_outer_slopes_from_state(current_z, trial_params)
        candidates, predictor_diagnostics = _seed_candidates(
            current_z=current_z,
            current_params=current_params,
            target_params=trial_params,
            target_zeta=float(zeta),
            current_zeta=float(current_zeta),
            prev_zeta=prev_zeta,
            prev_z=prev_z,
            full_mwind=full_mwind,
            inner_rg=inner_rg,
        )
        seed_strategy, seed0, solve_params, initial_full = min(candidates, key=lambda item: item[3])
        print(
            f"zeta={zeta:g} wind_fraction={wind_fraction:.6g} s={powerlaw_s:.6g} "
            f"dzeta={float(zeta) - float(current_zeta):.6g} "
            f"Mout/Min~{1.0 + wind_fraction - trial_params.stream_source_fraction:.6g} "
            f"initial_current={predictor_diagnostics['predictor_initial_full_current_seed']:.3e} "
            f"initial_comp={predictor_diagnostics['predictor_initial_full_mass_compensated_seed']:.3e} "
            f"initial_stress={predictor_diagnostics['predictor_initial_full_stress_ratio_seed']:.3e} "
            f"initial_omega={predictor_diagnostics['predictor_initial_full_mass_comp_omega_seed']:.3e} "
            f"secant_best={predictor_diagnostics['predictor_initial_full_secant_best']:.3e} "
            f"tangent_best={predictor_diagnostics['predictor_initial_full_tangent_best']:.3e} "
            f"seed={seed_strategy}",
            flush=True,
        )
        if (
            adaptive_target is not None
            and initial_full > ADAPTIVE_MAX_INITIAL_FULL
            and adaptive_step > ADAPTIVE_MIN_STEP * (1.0 + 1.0e-12)
        ):
            adaptive_rejections += 1
            new_step = max(ADAPTIVE_MIN_STEP, adaptive_step * ADAPTIVE_SHRINK)
            print(
                f"  adaptive pre-shrink: initial={initial_full:.3e} "
                f"> {ADAPTIVE_MAX_INITIAL_FULL:.3e}; step {adaptive_step:.6g}->{new_step:.6g}",
                flush=True,
            )
            adaptive_step = new_step
            if adaptive_rejections >= ADAPTIVE_MAX_REJECTIONS:
                print("stopping: adaptive predictor rejected too many times", flush=True)
                break
            continue
        old_z = np.asarray(current_z, dtype=float)
        old_zeta = float(current_zeta)
        t0 = time.perf_counter()
        seed, polish, final_params, elapsed, meta = scan.polish_with_optional_residual_remesh(
            seed=seed0,
            params=solve_params,
            remesh_after_accept=False,
            remesh_on_reject=False,
        )
        elapsed = time.perf_counter() - t0 if elapsed <= 0.0 else elapsed
        row = scan.row_for_result(
            branch="powerlaw_mass_wind_pilot",
            mass_fraction=float(final_params.stream_source_fraction),
            seed=seed,
            z=polish.z,
            params=final_params,
            polish=polish,
            elapsed_s=elapsed,
            extra={
                **meta,
                **predictor_diagnostics,
                "wind_mass_coupling_strength": float(zeta),
                "full_implied_Mwind_over_inner_etaesc1": float(full_mwind),
                "powerlaw_inner_rg": float(inner_rg),
                "predictor_initial_full_current": float(initial_full),
                "initial_full_current_seed": float(predictor_diagnostics["predictor_initial_full_current_seed"]),
                "initial_full_mass_compensated_seed": float(predictor_diagnostics["predictor_initial_full_mass_compensated_seed"]),
                "initial_full_stress_ratio_seed": float(predictor_diagnostics["predictor_initial_full_stress_ratio_seed"]),
                "initial_full_mass_comp_omega_seed": float(predictor_diagnostics["predictor_initial_full_mass_comp_omega_seed"]),
                "seed_strategy": str(seed_strategy),
                "anchor_checkpoint": scan.relative_root_path(ANCHOR),
                "reference_Qwind_Qvisc": float(reference["reference_Qwind_Qvisc"]),
                "reference_epsilon_w": float(reference["reference_epsilon_w"]),
                "adaptive_enabled": bool(adaptive_target is not None),
                "adaptive_target": np.nan if adaptive_target is None else float(adaptive_target),
                "adaptive_step": np.nan if adaptive_target is None else float(zeta - old_zeta),
                "adaptive_step_before": np.nan if adaptive_target is None else float(adaptive_step),
                "adaptive_max_initial_full": float(ADAPTIVE_MAX_INITIAL_FULL),
            },
            lean_diagnostics=False,
        )
        row["predictor"] = "current"
        row["predictor_initial_full"] = float(initial_full)
        scan.apply_physical_gate(row)
        row["checkpoint"] = _write_checkpoint(row, final_params)
        rows.append(row)
        _write_outputs(rows)
        print(
            f"  final={row['final_full']:.3e} accepted={row['accepted']} "
            f"phys_ok={row.get('physical_E_gate_eligible', True)} dom={row['dominant']} "
            f"physE={row.get('partition_physical_E', np.nan):.3e} "
            f"nfev={row.get('polish_nfev_total', row.get('nfev', np.nan))}",
            flush=True,
        )
        accepted = bool(row["accepted"]) and bool(row.get("physical_E_gate_eligible", True))
        if bool(row["accepted"]) and bool(row.get("physical_E_gate_eligible", True)):
            prev_z = old_z
            prev_zeta = old_zeta
            current_z = np.asarray(polish.z, dtype=float)
            current_params = final_params
            current_zeta = float(zeta)
            if adaptive_target is not None:
                adaptive_rejections = 0
                nfev = int(row.get("polish_nfev_total", row.get("nfev", 999999)))
                new_step = _next_adaptive_step(adaptive_step, accepted=True, initial_full=float(initial_full), nfev=nfev)
                print(f"  adaptive accepted: next_step {adaptive_step:.6g}->{new_step:.6g}", flush=True)
                adaptive_step = new_step
        else:
            if adaptive_target is not None and adaptive_step > ADAPTIVE_MIN_STEP * (1.0 + 1.0e-12):
                adaptive_rejections += 1
                new_step = _next_adaptive_step(adaptive_step, accepted=False, initial_full=float(initial_full), nfev=999999)
                print(f"  adaptive rejected: retry step {adaptive_step:.6g}->{new_step:.6g}", flush=True)
                adaptive_step = new_step
                if adaptive_rejections < ADAPTIVE_MAX_REJECTIONS:
                    continue
            print("stopping at first failed mass-coupled power-law step", flush=True)
            break

    print(f"wrote {JSON_OUTPUT.relative_to(ROOT)}", flush=True)
    print(f"wrote {MD_OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
