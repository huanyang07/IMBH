"""Staged N-continuation for the slope-unknown fixed-Mdot root."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    profile_from_state_vector,
    residual_audit_from_state_vector,
    state_bounds,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_full_slope_tuned_sonic_audit import residual_vector
from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_slope_unknown_root import (
    SOURCE_CHECKPOINT as N64_SOURCE_CHECKPOINT,
    active_physical_max,
    branch_metrics,
    dominant_block,
    extended_residual,
    extended_sparsity,
    fit_outer_slopes,
    load_source,
    params_for,
    remap_state_pchip,
    row_json,
    unpack_unknown,
)


ROOT = Path(__file__).resolve().parents[1]
N65_SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_slope_unknown_root" / "n8_medium_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_staged_resolution_continuation.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_staged_resolution_continuation"

N_SEQUENCE = (65, 67, 69, 73, 77, 81, 89, 97, 105, 113, 121, 129)
TARGET_NODES = {81, 97, 113, 129}
MAX_NFEV_PRECONDITION = 260
MAX_NFEV_LOCKED = 450
MAX_NFEV_RELEASE = 900
MAX_NFEV_POLISH = 650
SLOPE_BOUND_HALF_WIDTH = 8.0e-2
BRANCH_DISTANCE_LIMIT = 4.0
BRIDGE_RESIDUAL_FACTOR = 5.0

PRIOR_LABEL = "n8_medium"
PRIOR_MODE = "n_fit"
PRIOR_WINDOW_FRACTION = 1.0
PRIOR_MIN_POINTS = 8
SIGMA_SLOPES = (1.0e-2, 3.0e-3)

LOCK_LOGU_HALF_WIDTH = 6.0e-2
LOCK_LOGT_HALF_WIDTH = 4.0e-2
LOCK_RSON_RG_HALF_WIDTH = 5.0e-2
LOCK_LAMBDA_HALF_WIDTH = 1.0e-3

POLISH_LOGU_HALF_WIDTH = 3.0e-2
POLISH_LOGT_HALF_WIDTH = 2.0e-2
POLISH_RSON_RG_HALF_WIDTH = 2.5e-2
POLISH_LAMBDA_HALF_WIDTH = 5.0e-4


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def local_row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def load_checkpoint(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), json.loads(str(data["row_json"].item()))


def base_source_from_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "ratio": float(row["ratio"]),
        "R_out_rg": float(row["R_out_rg"]),
        "interval_form": "differential",
        "integrated_weighting": "none",
        "g_u": float(row["g_u_solved"]),
        "g_T": float(row["g_T_solved"]),
    }


def full_extended_bounds(
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = state_bounds(params)
    slope_lower = np.asarray(prior_slopes, dtype=float) - SLOPE_BOUND_HALF_WIDTH
    slope_upper = np.asarray(prior_slopes, dtype=float) + SLOPE_BOUND_HALF_WIDTH
    return np.concatenate([lower, slope_lower]), np.concatenate([upper, slope_upper])


def locked_extended_bounds(
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
    z_seed: np.ndarray,
    *,
    logu_half_width: float = LOCK_LOGU_HALF_WIDTH,
    logT_half_width: float = LOCK_LOGT_HALF_WIDTH,
    rson_rg_half_width: float = LOCK_RSON_RG_HALF_WIDTH,
    lambda_half_width: float = LOCK_LAMBDA_HALF_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = full_extended_bounds(params, prior_slopes)
    n = params.n_nodes
    logu_seed = z_seed[:n]
    logT_seed = z_seed[n : 2 * n]
    logR_seed = float(z_seed[2 * n])
    lambda_seed = float(z_seed[2 * n + 1])
    rson_seed_rg = float(np.exp(logR_seed) / params.r_g)

    lower[:n] = np.maximum(lower[:n], logu_seed - logu_half_width)
    upper[:n] = np.minimum(upper[:n], logu_seed + logu_half_width)
    lower[n : 2 * n] = np.maximum(lower[n : 2 * n], logT_seed - logT_half_width)
    upper[n : 2 * n] = np.minimum(upper[n : 2 * n], logT_seed + logT_half_width)

    rson_low = max(params.R_son_bounds_rg[0] + 1.0e-6, rson_seed_rg - rson_rg_half_width)
    rson_high = min(params.R_son_bounds_rg[1] - 1.0e-6, rson_seed_rg + rson_rg_half_width)
    lower[2 * n] = max(lower[2 * n], np.log(rson_low * params.r_g))
    upper[2 * n] = min(upper[2 * n], np.log(rson_high * params.r_g))
    lower[2 * n + 1] = max(lower[2 * n + 1], lambda_seed - lambda_half_width)
    upper[2 * n + 1] = min(upper[2 * n + 1], lambda_seed + lambda_half_width)
    return lower, upper


def polish_extended_bounds(
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
    z_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return locked_extended_bounds(
        params,
        prior_slopes,
        z_center,
        logu_half_width=POLISH_LOGU_HALF_WIDTH,
        logT_half_width=POLISH_LOGT_HALF_WIDTH,
        rson_rg_half_width=POLISH_RSON_RG_HALF_WIDTH,
        lambda_half_width=POLISH_LAMBDA_HALF_WIDTH,
    )


def solve_extended(
    x0: np.ndarray,
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
    sigma_slopes: tuple[float, float],
    bounds: tuple[np.ndarray, np.ndarray],
    max_nfev: int,
):
    lower, upper = bounds
    x_start = np.clip(np.asarray(x0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
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
        max_nfev=max_nfev,
    )


def residual_limit(n_nodes: int) -> float:
    return 2.0e-6 if n_nodes >= 129 else 5.0e-6


def branch_pass(metrics: dict[str, float]) -> bool:
    return bool(
        abs(metrics["delta_Rson_rg"]) <= 5.0e-2
        and abs(metrics["delta_lambda0"]) <= 1.0e-3
        and abs(metrics["delta_int_adv"]) <= 1.0e-2
        and metrics["branch_distance"] <= BRANCH_DISTANCE_LIMIT
    )


def bridge_residual_limit(n_nodes: int) -> float:
    return BRIDGE_RESIDUAL_FACTOR * residual_limit(n_nodes)


def row_from_state(
    *,
    n_nodes: int,
    phase: str,
    prior_slopes: tuple[float, float],
    sigma_slopes: tuple[float, float],
    effective_points: int,
    effective_inner_fraction: float,
    params: TransonicSlimParams,
    x: np.ndarray,
    z_seed: np.ndarray,
    result,
    previous_z: np.ndarray,
    previous_params: TransonicSlimParams,
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
) -> dict[str, object]:
    z, solved_slopes = unpack_unknown(np.asarray(x, dtype=float), params)
    solved_params = replace(params, outer_match_log_slopes=solved_slopes)
    base = residual_vector(z, solved_params, "symmetric", "C1", ("D", "C1", "C2"))
    final_extended = extended_residual(x, params, prior_slopes, sigma_slopes)
    audit = residual_audit_from_state_vector(z, solved_params)
    profile = profile_from_state_vector(z, solved_params)
    previous = branch_metrics(z, solved_params, previous_z, previous_params)
    anchor = branch_metrics(z, solved_params, anchor_z, anchor_params)
    prior_residual = np.asarray(
        [
            (solved_slopes[0] - prior_slopes[0]) / sigma_slopes[0],
            (solved_slopes[1] - prior_slopes[1]) / sigma_slopes[1],
        ],
        dtype=float,
    )
    physical = active_physical_max(audit)
    residual_pass = bool(physical <= residual_limit(n_nodes))
    previous_branch_pass = branch_pass(previous)
    anchor_branch_pass = branch_pass(anchor)
    science_pass = bool(residual_pass and previous_branch_pass and anchor_branch_pass)
    bridge_pass = bool(physical <= bridge_residual_limit(n_nodes) and previous_branch_pass)
    return {
        "n_nodes": n_nodes,
        "phase": phase,
        "ratio": solved_params.mdot_edd_ratio,
        "R_out_rg": solved_params.R_out_rg,
        "prior_label": PRIOR_LABEL,
        "effective_points": effective_points,
        "effective_inner_fraction": effective_inner_fraction,
        "g_u_prior": float(prior_slopes[0]),
        "g_T_prior": float(prior_slopes[1]),
        "g_u_solved": float(solved_slopes[0]),
        "g_T_solved": float(solved_slopes[1]),
        "delta_g_u": float(solved_slopes[0] - prior_slopes[0]),
        "delta_g_T": float(solved_slopes[1] - prior_slopes[1]),
        "prior_max": float(np.max(np.abs(prior_residual))),
        "prior_chi2": float(np.dot(prior_residual, prior_residual)),
        "physical_active": physical,
        "base_final_max": float(np.max(np.abs(base))),
        "final_extended_max": float(np.max(np.abs(final_extended))),
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
        "Rson_rg": float(profile.sonic_radius / solved_params.r_g),
        "lambda0": float(profile.lambda0),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": audit.outer_H_over_R,
        "int_adv": float(profile.integrated_advective_fraction),
        "residual_limit": residual_limit(n_nodes),
        "bridge_residual_limit": bridge_residual_limit(n_nodes),
        "residual_pass": residual_pass,
        "bridge_pass": bridge_pass,
        "previous_branch_pass": previous_branch_pass,
        "anchor_branch_pass": anchor_branch_pass,
        "science_pass": science_pass,
        "previous_delta_Rson_rg": previous["delta_Rson_rg"],
        "previous_delta_lambda0": previous["delta_lambda0"],
        "previous_delta_int_adv": previous["delta_int_adv"],
        "previous_branch_distance": previous["branch_distance"],
        "previous_rms_dlogu": previous["rms_dlogu_vs_N64"],
        "previous_rms_dlogT": previous["rms_dlogT_vs_N64"],
        "previous_max_dlogu": previous["max_dlogu_vs_N64"],
        "previous_max_dlogT": previous["max_dlogT_vs_N64"],
        "anchor_delta_Rson_rg": anchor["delta_Rson_rg"],
        "anchor_delta_lambda0": anchor["delta_lambda0"],
        "anchor_delta_int_adv": anchor["delta_int_adv"],
        "anchor_branch_distance": anchor["branch_distance"],
        "anchor_rms_dlogu": anchor["rms_dlogu_vs_N64"],
        "anchor_rms_dlogT": anchor["rms_dlogT_vs_N64"],
        "anchor_max_dlogu": anchor["max_dlogu_vs_N64"],
        "anchor_max_dlogT": anchor["max_dlogT_vs_N64"],
        "seed_delta_norm": float(np.linalg.norm(z - z_seed) / np.sqrt(len(z))),
        "nfev": int(result.nfev) if result is not None else 0,
        "success": bool(result.success) if result is not None else True,
        "message": str(result.message) if result is not None else "anchor row",
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"N{int(row['n_nodes'])}_{row['phase']}_{float(row['ratio']):.8f}".replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        n_nodes=np.array(row["n_nodes"]),
        R_out_rg=np.array(row["R_out_rg"]),
        g_u_solved=np.array(row["g_u_solved"]),
        g_T_solved=np.array(row["g_T_solved"]),
        row_json=np.array(local_row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Staged Resolution Continuation",
        "",
        "Generated by `scripts/run_transonic_staged_resolution_continuation.py`.",
        "",
        "Starts from the passing N65 slope-unknown root and stages through higher resolution with PCHIP remap, local N-fit outer-slope priors, an integrated-defect preconditioner, a locked differential polish, and a released/pinned differential polish. Branch checks are reported against both the previous accepted stage and the N65 anchor. `bridge pass` is a near-tolerance local handoff criterion using the previous-stage branch gate; science-grade rows still require the stricter anchor check.",
        "",
        "| N | phase | target | physical | limit | bridge limit | dominant | residual pass | bridge pass | prev branch | anchor branch | science pass | g_u prior | g_u solved | dg_u | g_T prior | g_T solved | dg_T | prior max | eff window | Rson/rg | dRson prev | dRson anchor | lambda0 | dlambda prev | int adv | dint prev | branch prev | branch anchor | rms u prev | rms T prev | interval R | outer 1 | D | C1 | C2 | K | nfev | success | message |",
        "|---:|---|:---:|---:|---:|---:|---|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {n_nodes} | {phase} | {target_node} | {physical_active} | {residual_limit} | {bridge_residual_limit} | {dominant} | "
            "{residual_pass} | {bridge_pass} | {previous_branch_pass} | {anchor_branch_pass} | {science_pass} | "
            "{g_u_prior} | {g_u_solved} | {delta_g_u} | {g_T_prior} | {g_T_solved} | {delta_g_T} | "
            "{prior_max} | {effective_inner_fraction} | {Rson_rg} | {previous_delta_Rson_rg} | "
            "{anchor_delta_Rson_rg} | {lambda0} | {previous_delta_lambda0} | {int_adv} | "
            "{previous_delta_int_adv} | {previous_branch_distance} | {anchor_branch_distance} | "
            "{previous_rms_dlogu} | {previous_rms_dlogT} | {interval_R} | {outer_1} | "
            "{D} | {C1} | {C2} | {K} | {nfev} | {success} | {message} |".format(
                n_nodes=row["n_nodes"],
                phase=row["phase"],
                target_node="yes" if row["n_nodes"] in TARGET_NODES else "no",
                physical_active=fmt(float(row["physical_active"])),
                residual_limit=fmt(float(row["residual_limit"])),
                bridge_residual_limit=fmt(float(row["bridge_residual_limit"])),
                dominant=row["dominant"],
                residual_pass="yes" if row["residual_pass"] else "no",
                bridge_pass="yes" if row["bridge_pass"] else "no",
                previous_branch_pass="yes" if row["previous_branch_pass"] else "no",
                anchor_branch_pass="yes" if row["anchor_branch_pass"] else "no",
                science_pass="yes" if row["science_pass"] else "no",
                g_u_prior=fmt(float(row["g_u_prior"])),
                g_u_solved=fmt(float(row["g_u_solved"])),
                delta_g_u=fmt(float(row["delta_g_u"])),
                g_T_prior=fmt(float(row["g_T_prior"])),
                g_T_solved=fmt(float(row["g_T_solved"])),
                delta_g_T=fmt(float(row["delta_g_T"])),
                prior_max=fmt(float(row["prior_max"])),
                effective_inner_fraction=fmt(float(row["effective_inner_fraction"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                previous_delta_Rson_rg=fmt(float(row["previous_delta_Rson_rg"])),
                anchor_delta_Rson_rg=fmt(float(row["anchor_delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                previous_delta_lambda0=fmt(float(row["previous_delta_lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                previous_delta_int_adv=fmt(float(row["previous_delta_int_adv"])),
                previous_branch_distance=fmt(float(row["previous_branch_distance"])),
                anchor_branch_distance=fmt(float(row["anchor_branch_distance"])),
                previous_rms_dlogu=fmt(float(row.get("previous_rms_dlogu", 0.0))),
                previous_rms_dlogT=fmt(float(row.get("previous_rms_dlogT", 0.0))),
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


def accepted_candidate(rows: list[dict[str, object]]) -> dict[str, object] | None:
    science = [row for row in rows if row["science_pass"]]
    if science:
        return min(science, key=lambda row: float(row["physical_active"]))
    bridge = [row for row in rows if row["bridge_pass"]]
    if bridge:
        return min(bridge, key=lambda row: float(row["physical_active"]))
    branch = [
        row
        for row in rows
        if row["previous_branch_pass"]
        and float(row["physical_active"]) <= 10.0 * bridge_residual_limit(int(row["n_nodes"]))
    ]
    if branch:
        return min(branch, key=lambda row: float(row["physical_active"]))
    return None


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)

    n64_z, n64_source = load_source()
    n64_params = params_for(
        fiducial,
        mdot_edd,
        n64_source,
        n_nodes=64,
        slopes=(float(n64_source["g_u"]), float(n64_source["g_T"])),
    )
    n65_z, n65_row = load_checkpoint(N65_SOURCE_CHECKPOINT)
    n65_source = base_source_from_row(n65_row)
    n65_params = params_for(
        fiducial,
        mdot_edd,
        n65_source,
        n_nodes=65,
        slopes=(float(n65_row["g_u_solved"]), float(n65_row["g_T_solved"])),
    )

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    anchor_row = {
        **{key: n65_row[key] for key in n65_row if key != "z"},
        "phase": "anchor",
        "previous_branch_pass": True,
        "anchor_branch_pass": True,
        "bridge_pass": True,
        "science_pass": True,
        "previous_delta_Rson_rg": 0.0,
        "previous_delta_lambda0": 0.0,
        "previous_delta_int_adv": 0.0,
        "previous_branch_distance": 0.0,
        "previous_rms_dlogu": 0.0,
        "previous_rms_dlogT": 0.0,
        "previous_max_dlogu": 0.0,
        "previous_max_dlogT": 0.0,
        "anchor_delta_Rson_rg": 0.0,
        "anchor_delta_lambda0": 0.0,
        "anchor_delta_int_adv": 0.0,
        "anchor_branch_distance": 0.0,
        "anchor_rms_dlogu": 0.0,
        "anchor_rms_dlogT": 0.0,
        "anchor_max_dlogu": 0.0,
        "anchor_max_dlogT": 0.0,
        "residual_limit": residual_limit(65),
        "bridge_residual_limit": bridge_residual_limit(65),
        "seed_delta_norm": 0.0,
        "z": n65_z,
    }
    rows.append(anchor_row)
    save_checkpoint(anchor_row)
    write_table(rows)

    previous_z = n65_z
    previous_params = n65_params
    anchor_z = n65_z
    anchor_params = n65_params

    for n_nodes in N_SEQUENCE[1:]:
        seed_source = base_source_from_row(
            {
                "ratio": previous_params.mdot_edd_ratio,
                "R_out_rg": previous_params.R_out_rg,
                "g_u_solved": previous_params.outer_match_log_slopes[0],
                "g_T_solved": previous_params.outer_match_log_slopes[1],
            }
        )
        seed_params = params_for(
            fiducial,
            mdot_edd,
            seed_source,
            n_nodes=n_nodes,
            slopes=previous_params.outer_match_log_slopes,
        )
        z_seed = remap_state_pchip(previous_z, previous_params, seed_params)
        prior_slopes = fit_outer_slopes(
            z_seed,
            seed_params,
            mode=PRIOR_MODE,
            window_fraction=PRIOR_WINDOW_FRACTION,
            min_points=PRIOR_MIN_POINTS,
        )
        prior_gu, prior_gT, effective_points, effective_inner_fraction = prior_slopes
        prior_pair = (prior_gu, prior_gT)
        params = replace(seed_params, outer_match_log_slopes=prior_pair)
        x_seed = np.concatenate([z_seed, np.asarray(prior_pair, dtype=float)])

        integrated_params = replace(
            params,
            interval_residual_form="integrated",
            integrated_residual_weighting="inverse_sqrt_dx",
        )
        precondition_result = solve_extended(
            x_seed,
            integrated_params,
            prior_pair,
            SIGMA_SLOPES,
            locked_extended_bounds(params, prior_pair, z_seed),
            MAX_NFEV_PRECONDITION,
        )
        precondition_row = row_from_state(
            n_nodes=n_nodes,
            phase="integrated_pre",
            prior_slopes=prior_pair,
            sigma_slopes=SIGMA_SLOPES,
            effective_points=effective_points,
            effective_inner_fraction=effective_inner_fraction,
            params=params,
            x=precondition_result.x,
            z_seed=z_seed,
            result=precondition_result,
            previous_z=previous_z,
            previous_params=previous_params,
            anchor_z=anchor_z,
            anchor_params=anchor_params,
        )
        rows.append(precondition_row)
        save_checkpoint(precondition_row)
        write_table(rows)
        print(
            f"N={n_nodes} integrated_pre physical={precondition_row['physical_active']:.3e} "
            f"Rson={precondition_row['Rson_rg']:.4f} bridge={precondition_row['bridge_pass']} "
            f"science={precondition_row['science_pass']} nfev={precondition_row['nfev']}",
            flush=True,
        )

        locked_seed = precondition_result.x if precondition_row["previous_branch_pass"] else x_seed
        locked_result = solve_extended(
            locked_seed,
            params,
            prior_pair,
            SIGMA_SLOPES,
            locked_extended_bounds(params, prior_pair, z_seed),
            MAX_NFEV_LOCKED,
        )
        locked_row = row_from_state(
            n_nodes=n_nodes,
            phase="locked",
            prior_slopes=prior_pair,
            sigma_slopes=SIGMA_SLOPES,
            effective_points=effective_points,
            effective_inner_fraction=effective_inner_fraction,
            params=params,
            x=locked_result.x,
            z_seed=z_seed,
            result=locked_result,
            previous_z=previous_z,
            previous_params=previous_params,
            anchor_z=anchor_z,
            anchor_params=anchor_params,
        )
        rows.append(locked_row)
        save_checkpoint(locked_row)
        write_table(rows)
        print(
            f"N={n_nodes} locked physical={locked_row['physical_active']:.3e} "
            f"Rson={locked_row['Rson_rg']:.4f} bridge={locked_row['bridge_pass']} "
            f"science={locked_row['science_pass']} nfev={locked_row['nfev']}",
            flush=True,
        )

        release_result = solve_extended(
            locked_result.x,
            params,
            prior_pair,
            SIGMA_SLOPES,
            full_extended_bounds(params, prior_pair),
            MAX_NFEV_RELEASE,
        )
        release_row = row_from_state(
            n_nodes=n_nodes,
            phase="release",
            prior_slopes=prior_pair,
            sigma_slopes=SIGMA_SLOPES,
            effective_points=effective_points,
            effective_inner_fraction=effective_inner_fraction,
            params=params,
            x=release_result.x,
            z_seed=z_seed,
            result=release_result,
            previous_z=previous_z,
            previous_params=previous_params,
            anchor_z=anchor_z,
            anchor_params=anchor_params,
        )
        rows.append(release_row)
        save_checkpoint(release_row)
        write_table(rows)
        print(
            f"N={n_nodes} release physical={release_row['physical_active']:.3e} "
            f"Rson={release_row['Rson_rg']:.4f} bridge={release_row['bridge_pass']} "
            f"science={release_row['science_pass']} nfev={release_row['nfev']}",
            flush=True,
        )

        stage_rows = [precondition_row, locked_row, release_row]
        if release_row["previous_branch_pass"]:
            polish_result = solve_extended(
                release_result.x,
                params,
                prior_pair,
                SIGMA_SLOPES,
                polish_extended_bounds(params, prior_pair, np.asarray(release_row["z"], dtype=float)),
                MAX_NFEV_POLISH,
            )
            polish_row = row_from_state(
                n_nodes=n_nodes,
                phase="branch_polish",
                prior_slopes=prior_pair,
                sigma_slopes=SIGMA_SLOPES,
                effective_points=effective_points,
                effective_inner_fraction=effective_inner_fraction,
                params=params,
                x=polish_result.x,
                z_seed=z_seed,
                result=polish_result,
                previous_z=previous_z,
                previous_params=previous_params,
                anchor_z=anchor_z,
                anchor_params=anchor_params,
            )
            rows.append(polish_row)
            stage_rows.append(polish_row)
            save_checkpoint(polish_row)
            write_table(rows)
            print(
                f"N={n_nodes} branch_polish physical={polish_row['physical_active']:.3e} "
                f"Rson={polish_row['Rson_rg']:.4f} bridge={polish_row['bridge_pass']} "
                f"science={polish_row['science_pass']} nfev={polish_row['nfev']}",
                flush=True,
            )

        chosen = accepted_candidate(stage_rows)
        if chosen is None:
            print(
                f"N={n_nodes} no branch-compatible bridge; stopped before poisoning the next stage.",
                flush=True,
            )
            break
        previous_z = np.asarray(chosen["z"], dtype=float)
        previous_params = replace(params, outer_match_log_slopes=(float(chosen["g_u_solved"]), float(chosen["g_T_solved"])))
        print(
            f"N={n_nodes} accepted phase={chosen['phase']} physical={chosen['physical_active']:.3e} "
            f"bridge={chosen['bridge_pass']} science={chosen['science_pass']}",
            flush=True,
        )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
