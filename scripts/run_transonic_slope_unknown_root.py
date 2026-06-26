"""N65 fixed-Mdot root with outer slopes as weakly constrained unknowns."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    computational_grid,
    pack_state,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    state_bounds,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_full_slope_tuned_sonic_audit import residual_vector, sparsity_pattern
from run_transonic_outer_slope_calibration_audit import fmt


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "transonic_integrated_defect_slope_law_audit"
    / "differential_symmetric_D_C1_C2_0p90277664.npz"
)
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_slope_unknown_root.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_slope_unknown_root"

N_NODES = 65
MAX_NFEV = 500
SONIC_MODE = "symmetric"
SONIC_PIVOT = "C1"
SONIC_COMPONENTS = ("D", "C1", "C2")
SLOPE_BOUND_HALF_WIDTH = 8.0e-2

PRIOR_SPECS = (
    ("n8_tight", "n_fit", 1.0, 8, 3.0e-3, 1.0e-3),
    ("n8_medium", "n_fit", 1.0, 8, 1.0e-2, 3.0e-3),
    ("n8_loose", "n_fit", 1.0, 8, 3.0e-2, 1.0e-2),
    ("win092_medium", "window", 0.92, 8, 1.0e-2, 3.0e-3),
    ("win085_medium", "window", 0.85, 8, 1.0e-2, 3.0e-3),
    ("win065_medium", "window", 0.65, 8, 1.0e-2, 3.0e-3),
)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def load_source() -> tuple[np.ndarray, dict[str, object]]:
    with np.load(SOURCE_CHECKPOINT, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), json.loads(str(data["row_json"].item()))


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    source: dict[str, object],
    n_nodes: int,
    slopes: tuple[float, float],
) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(source["ratio"]) * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=n_nodes,
        R_out_rg=float(source["R_out_rg"]),
        residual_tol=1.0e-6,
        max_nfev=MAX_NFEV,
        outer_closure="full_slope_match",
        outer_match_log_slopes=slopes,
        interval_residual_form=str(source["interval_form"]),
        integrated_residual_weighting=str(source["integrated_weighting"]),
    )


def remap_state_pchip(z: np.ndarray, old_params: TransonicSlimParams, new_params: TransonicSlimParams) -> np.ndarray:
    logu_old, logT_old, logR_son, lambda0, logR_old = unpack_state(z, old_params)
    logR_new = computational_grid(new_params, logR_son)
    logu_new = PchipInterpolator(logR_old, logu_old)(logR_new)
    logT_new = PchipInterpolator(logR_old, logT_old)(logR_new)
    return pack_state(logu_new, logT_new, logR_son, lambda0)


def fit_outer_slopes(
    z: np.ndarray,
    params: TransonicSlimParams,
    *,
    mode: str,
    window_fraction: float,
    min_points: int,
    degree: int = 2,
) -> tuple[float, float, int, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    if mode == "n_fit":
        start = max(0, len(logR) - int(min_points))
    elif mode == "window":
        threshold = np.log(params.R_out * float(window_fraction))
        candidates = np.flatnonzero(logR >= threshold)
        if len(candidates) >= int(min_points):
            start = int(candidates[0])
        else:
            start = max(0, len(logR) - int(min_points))
    else:
        raise ValueError(f"unknown slope prior mode {mode!r}")
    count = len(logR) - start
    fit_degree = min(int(degree), count - 1)
    x = logR[start:] - logR[-1]
    gu_poly = np.poly1d(np.polyfit(x, logu[start:], fit_degree))
    gT_poly = np.poly1d(np.polyfit(x, logT[start:], fit_degree))
    effective_inner_fraction = float(np.exp(logR[start] - logR[-1]))
    return float(np.polyder(gu_poly)(0.0)), float(np.polyder(gT_poly)(0.0)), int(count), effective_inner_fraction


def unpack_unknown(x: np.ndarray, params: TransonicSlimParams) -> tuple[np.ndarray, tuple[float, float]]:
    state_size = 2 * params.n_nodes + 2
    return np.asarray(x[:state_size], dtype=float), (float(x[state_size]), float(x[state_size + 1]))


def extended_residual(
    x: np.ndarray,
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
    sigma_slopes: tuple[float, float],
) -> np.ndarray:
    z, slopes = unpack_unknown(x, params)
    slope_params = replace(params, outer_match_log_slopes=slopes)
    base = residual_vector(z, slope_params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS)
    prior = np.asarray(
        [
            (slopes[0] - prior_slopes[0]) / sigma_slopes[0],
            (slopes[1] - prior_slopes[1]) / sigma_slopes[1],
        ],
        dtype=float,
    )
    return np.concatenate([base, prior])


def extended_sparsity(params: TransonicSlimParams):
    base = sparsity_pattern(params, SONIC_MODE, SONIC_COMPONENTS)
    state_size = 2 * params.n_nodes + 2
    pattern = lil_matrix((base.shape[0] + 2, state_size + 2), dtype=int)
    pattern[: base.shape[0], :state_size] = base
    outer_row = 2 * (params.n_nodes - 1)
    pattern[outer_row : outer_row + 2, state_size : state_size + 2] = 1
    pattern[base.shape[0], state_size] = 1
    pattern[base.shape[0] + 1, state_size + 1] = 1
    return pattern.tocsr()


def extended_bounds(params: TransonicSlimParams, prior_slopes: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = state_bounds(params)
    slope_lower = np.asarray(prior_slopes, dtype=float) - SLOPE_BOUND_HALF_WIDTH
    slope_upper = np.asarray(prior_slopes, dtype=float) + SLOPE_BOUND_HALF_WIDTH
    return np.concatenate([lower, slope_lower]), np.concatenate([upper, slope_upper])


def solve_variant(
    z0: np.ndarray,
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
    sigma_slopes: tuple[float, float],
):
    lower, upper = extended_bounds(params, prior_slopes)
    x0 = np.concatenate([np.asarray(z0, dtype=float), np.asarray(prior_slopes, dtype=float)])
    x_start = np.clip(x0, lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: extended_residual(trial, params, prior_slopes, sigma_slopes),
        x_start,
        jac_sparsity=extended_sparsity(params),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV,
    )


def active_physical_max(audit) -> float:
    return max(
        abs(audit.interval_radial_max),
        abs(audit.interval_energy_max),
        abs(audit.outer_omega),
        abs(audit.outer_energy),
        abs(audit.sonic_D),
        abs(audit.sonic_C1),
        abs(audit.sonic_C2),
        abs(audit.sonic_K),
    )


def dominant_block(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_1": abs(audit.outer_omega),
        "outer_2": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
        "K": abs(audit.sonic_K),
    }
    return max(values, key=values.get)


def branch_metrics(z: np.ndarray, params: TransonicSlimParams, source_z: np.ndarray, source_params: TransonicSlimParams) -> dict[str, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    source_logu, source_logT, _source_logR_son, _source_lambda0, source_logR = unpack_state(source_z, source_params)
    source_logu_interp = np.interp(logR, source_logR, source_logu)
    source_logT_interp = np.interp(logR, source_logR, source_logT)
    profile = profile_from_state_vector(z, params)
    source_profile = profile_from_state_vector(source_z, source_params)
    delta_rson = float(profile.sonic_radius / params.r_g - source_profile.sonic_radius / source_params.r_g)
    delta_lambda = float(profile.lambda0 - source_profile.lambda0)
    delta_int_adv = float(profile.integrated_advective_fraction - source_profile.integrated_advective_fraction)
    rms_dlogu = float(np.sqrt(np.mean((logu - source_logu_interp) ** 2)))
    rms_dlogT = float(np.sqrt(np.mean((logT - source_logT_interp) ** 2)))
    distance = float(
        np.sqrt(
            (delta_rson / 0.05) ** 2
            + (delta_lambda / 1.0e-3) ** 2
            + (rms_dlogu / 0.02) ** 2
            + (rms_dlogT / 0.01) ** 2
        )
    )
    return {
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "delta_Rson_rg": delta_rson,
        "lambda0": float(profile.lambda0),
        "delta_lambda0": delta_lambda,
        "int_adv": float(profile.integrated_advective_fraction),
        "delta_int_adv": delta_int_adv,
        "rms_dlogu_vs_N64": rms_dlogu,
        "rms_dlogT_vs_N64": rms_dlogT,
        "max_dlogu_vs_N64": float(np.max(np.abs(logu - source_logu_interp))),
        "max_dlogT_vs_N64": float(np.max(np.abs(logT - source_logT_interp))),
        "branch_distance": distance,
    }


def row_from_result(
    *,
    label: str,
    prior_mode: str,
    window_fraction: float,
    n_fit: int,
    effective_points: int,
    effective_inner_fraction: float,
    prior_slopes: tuple[float, float],
    sigma_slopes: tuple[float, float],
    params: TransonicSlimParams,
    z0: np.ndarray,
    result,
    source_z: np.ndarray,
    source_params: TransonicSlimParams,
) -> dict[str, object]:
    z, solved_slopes = unpack_unknown(np.asarray(result.x, dtype=float), params)
    solved_params = replace(params, outer_match_log_slopes=solved_slopes)
    initial_extended = extended_residual(np.concatenate([z0, prior_slopes]), params, prior_slopes, sigma_slopes)
    final_extended = extended_residual(result.x, params, prior_slopes, sigma_slopes)
    base_final = residual_vector(z, solved_params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS)
    audit = residual_audit_from_state_vector(z, solved_params)
    profile = profile_from_state_vector(z, solved_params)
    prior_residual = np.asarray(
        [
            (solved_slopes[0] - prior_slopes[0]) / sigma_slopes[0],
            (solved_slopes[1] - prior_slopes[1]) / sigma_slopes[1],
        ],
        dtype=float,
    )
    branch = branch_metrics(z, solved_params, source_z, source_params)
    residual_pass = bool(active_physical_max(audit) <= 2.0e-6)
    branch_pass = bool(
        abs(branch["delta_Rson_rg"]) <= 0.05
        and abs(branch["delta_lambda0"]) <= 1.0e-3
        and abs(branch["delta_int_adv"]) <= 1.0e-2
    )
    return {
        "label": label,
        "prior_mode": prior_mode,
        "target_window_fraction": float(window_fraction),
        "effective_points": effective_points,
        "effective_inner_fraction": effective_inner_fraction,
        "n_fit": n_fit,
        "ratio": solved_params.mdot_edd_ratio,
        "R_out_rg": solved_params.R_out_rg,
        "n_nodes": solved_params.n_nodes,
        "g_u_prior": float(prior_slopes[0]),
        "g_T_prior": float(prior_slopes[1]),
        "sigma_g_u": float(sigma_slopes[0]),
        "sigma_g_T": float(sigma_slopes[1]),
        "g_u_solved": float(solved_slopes[0]),
        "g_T_solved": float(solved_slopes[1]),
        "delta_g_u": float(solved_slopes[0] - prior_slopes[0]),
        "delta_g_T": float(solved_slopes[1] - prior_slopes[1]),
        "prior_max": float(np.max(np.abs(prior_residual))),
        "prior_chi2": float(np.dot(prior_residual, prior_residual)),
        "initial_extended_max": float(np.max(np.abs(initial_extended))),
        "final_extended_max": float(np.max(np.abs(final_extended))),
        "base_final_max": float(np.max(np.abs(base_final))),
        "physical_active": active_physical_max(audit),
        "dominant": dominant_block(audit),
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_1": audit.outer_omega,
        "outer_2": audit.outer_energy,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "K": audit.sonic_K,
        "compat_max": max(abs(audit.sonic_C1), abs(audit.sonic_C2), abs(audit.sonic_K)),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": audit.outer_H_over_R,
        "residual_pass": residual_pass,
        "branch_pass": branch_pass,
        "science_pass": bool(residual_pass and branch_pass),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        **branch,
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{row['label']}_{float(row['ratio']):.8f}".replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        n_nodes=np.array(row["n_nodes"]),
        R_out_rg=np.array(row["R_out_rg"]),
        g_u_solved=np.array(row["g_u_solved"]),
        g_T_solved=np.array(row["g_T_solved"]),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Slope-Unknown Fixed-Mdot Root Audit",
        "",
        "Generated by `scripts/run_transonic_slope_unknown_root.py`.",
        "",
        "N65 fixed-Mdot solve at `R_out=6500 rg` with `g_u_out,g_T_out` appended as weakly constrained unknowns. The prior slopes are fitted from the remapped N64 source profile; the solved profile is judged by the usual differential physical audit.",
        "",
        "| label | physical | base max | extended max | dominant | residual pass | branch pass | science pass | g_u prior | g_u solved | dg_u | g_T prior | g_T solved | dg_T | prior max | prior chi2 | eff window | points | Rson/rg | dRson | lambda0 | dlambda0 | int adv | dint adv | branch dist | interval R | outer 1 | D | C1 | C2 | K | nfev | success | message |",
        "|---|---:|---:|---:|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {physical_active} | {base_final_max} | {final_extended_max} | {dominant} | "
            "{residual_pass} | {branch_pass} | {science_pass} | {g_u_prior} | {g_u_solved} | {delta_g_u} | "
            "{g_T_prior} | {g_T_solved} | {delta_g_T} | {prior_max} | {prior_chi2} | {effective_inner_fraction} | "
            "{effective_points} | {Rson_rg} | {delta_Rson_rg} | {lambda0} | {delta_lambda0} | {int_adv} | "
            "{delta_int_adv} | {branch_distance} | {interval_R} | {outer_1} | {D} | {C1} | {C2} | {K} | "
            "{nfev} | {success} | {message} |".format(
                label=row["label"],
                physical_active=fmt(float(row["physical_active"])),
                base_final_max=fmt(float(row["base_final_max"])),
                final_extended_max=fmt(float(row["final_extended_max"])),
                dominant=row["dominant"],
                residual_pass="yes" if row["residual_pass"] else "no",
                branch_pass="yes" if row["branch_pass"] else "no",
                science_pass="yes" if row["science_pass"] else "no",
                g_u_prior=fmt(float(row["g_u_prior"])),
                g_u_solved=fmt(float(row["g_u_solved"])),
                delta_g_u=fmt(float(row["delta_g_u"])),
                g_T_prior=fmt(float(row["g_T_prior"])),
                g_T_solved=fmt(float(row["g_T_solved"])),
                delta_g_T=fmt(float(row["delta_g_T"])),
                prior_max=fmt(float(row["prior_max"])),
                prior_chi2=fmt(float(row["prior_chi2"])),
                effective_inner_fraction=fmt(float(row["effective_inner_fraction"])),
                effective_points=row["effective_points"],
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row["delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row["delta_lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                delta_int_adv=fmt(float(row["delta_int_adv"])),
                branch_distance=fmt(float(row["branch_distance"])),
                interval_R=fmt(float(row["interval_R"])),
                outer_1=fmt(float(row["outer_1"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_z, source = load_source()
    source_params = params_for(
        fiducial,
        mdot_edd,
        source,
        n_nodes=64,
        slopes=(float(source["g_u"]), float(source["g_T"])),
    )
    seed_params = params_for(
        fiducial,
        mdot_edd,
        source,
        n_nodes=N_NODES,
        slopes=(float(source["g_u"]), float(source["g_T"])),
    )
    seed_z = remap_state_pchip(source_z, source_params, seed_params)

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for label, prior_mode, window_fraction, n_fit, sigma_gu, sigma_gT in PRIOR_SPECS:
        prior_gu, prior_gT, effective_points, effective_inner_fraction = fit_outer_slopes(
            seed_z,
            seed_params,
            mode=prior_mode,
            window_fraction=window_fraction,
            min_points=n_fit,
        )
        prior_slopes = (prior_gu, prior_gT)
        sigma_slopes = (sigma_gu, sigma_gT)
        params = replace(seed_params, outer_match_log_slopes=prior_slopes)
        result = solve_variant(seed_z, params, prior_slopes, sigma_slopes)
        row = row_from_result(
            label=label,
            prior_mode=prior_mode,
            window_fraction=window_fraction,
            n_fit=n_fit,
            effective_points=effective_points,
            effective_inner_fraction=effective_inner_fraction,
            prior_slopes=prior_slopes,
            sigma_slopes=sigma_slopes,
            params=params,
            z0=seed_z,
            result=result,
            source_z=source_z,
            source_params=source_params,
        )
        rows.append(row)
        save_checkpoint(row)
        write_table(rows)
        print(
            f"{label} physical={row['physical_active']:.3e} base={row['base_final_max']:.3e} "
            f"dg=({row['delta_g_u']:.3e},{row['delta_g_T']:.3e}) "
            f"Rson={row['Rson_rg']:.4f} pass={row['science_pass']} nfev={row['nfev']}",
            flush=True,
        )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
