"""Compare midpoint and mixed trapezoidal collocation for the fixed-Mdot root."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _differential_interval_residual_from_unpacked,
    _outer_boundary_residual,
    _sonic_component_values,
    pack_state,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    state_bounds,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import scaled_differential_matrix
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_full_slope_tuned_sonic_audit import residual_vector, sparsity_pattern
from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_slope_unknown_root import (
    active_physical_max,
    branch_metrics,
    dominant_block,
    fit_outer_slopes,
    remap_state_pchip,
    unpack_unknown,
)
from run_transonic_staged_resolution_continuation import (
    N65_SOURCE_CHECKPOINT,
    CHECKPOINT_DIR as STAGED_CHECKPOINT_DIR,
    base_source_from_row,
    branch_pass,
    full_extended_bounds,
    load_checkpoint,
    params_for,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_trapezoid_collocation_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_trapezoid_collocation_audit"

CASES = (
    ("N65_from_anchor", 65, N65_SOURCE_CHECKPOINT),
    ("N73_from_midpoint_bridge", 73, STAGED_CHECKPOINT_DIR / "N73_branch_polish_0p90277664.npz"),
    ("N81_from_N73_bridge", 81, STAGED_CHECKPOINT_DIR / "N73_branch_polish_0p90277664.npz"),
)
SCHEMES = ("midpoint", "trapezoid_mixed")
MAX_NFEV = 650
SONIC_MODE = "symmetric"
SONIC_PIVOT = "C1"
SONIC_COMPONENTS = ("D", "C1", "C2")
SLOPE_BOUND_HALF_WIDTH = 8.0e-2
SIGMA_SLOPES = (1.0e-2, 3.0e-3)
PRIOR_MODE = "n_fit"
PRIOR_WINDOW_FRACTION = 1.0
PRIOR_MIN_POINTS = 8


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


def local_gradient_scaled(logR: float, y: np.ndarray, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    """Return the local derivative from the scaled matrix equation."""

    A, c, _radial_scale, _energy_scale = scaled_differential_matrix(logR, y, lambda0, params)
    return np.linalg.solve(A, -c)


def trapezoid_interval_residual_from_unpacked(
    logu: np.ndarray,
    logT: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    params: TransonicSlimParams,
    idx: int,
) -> np.ndarray:
    """Return a mixed trapezoid interval residual.

    The first interval touches the sonic point, where the local ODE matrix is
    near singular by construction. Keep that interval on the existing midpoint
    residual and use trapezoid only for the regular outer intervals.
    """

    if idx == 0:
        return _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
    dx = float(logR[idx + 1] - logR[idx])
    y_left = np.array([logu[idx], logT[idx]], dtype=float)
    y_right = np.array([logu[idx + 1], logT[idx + 1]], dtype=float)
    g_left = local_gradient_scaled(float(logR[idx]), y_left, lambda0, params)
    g_right = local_gradient_scaled(float(logR[idx + 1]), y_right, lambda0, params)
    return (y_right - y_left - 0.5 * dx * (g_left + g_right)) / np.sqrt(dx)


def trapezoid_selected_residual(z: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    """Return selected residual rows for the mixed trapezoid scheme."""

    residual = np.zeros(2 * (params.n_nodes - 1) + 2 + len(SONIC_COMPONENTS), dtype=float)
    try:
        logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
        if np.any(np.diff(logR) <= 0.0):
            raise ValueError("mapped radius must increase")
        row = 0
        for idx in range(params.n_nodes - 1):
            residual[row : row + 2] = trapezoid_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            row += 2
        residual[row : row + 2] = _outer_boundary_residual(logR[-1], np.array([logu[-1], logT[-1]]), lambda0, params)
        row += 2
        residual[row : row + len(SONIC_COMPONENTS)] = _sonic_component_values(z, params, SONIC_COMPONENTS)
    except Exception:
        residual.fill(1.0e6)
    return residual


def selected_residual(z: np.ndarray, params: TransonicSlimParams, scheme: str) -> np.ndarray:
    if scheme == "midpoint":
        return residual_vector(z, params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS)
    if scheme == "trapezoid_mixed":
        return trapezoid_selected_residual(z, params)
    raise ValueError(f"unknown collocation scheme {scheme!r}")


def extended_residual(
    x: np.ndarray,
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
    sigma_slopes: tuple[float, float],
    scheme: str,
) -> np.ndarray:
    z, slopes = unpack_unknown(x, params)
    slope_params = replace(params, outer_match_log_slopes=slopes)
    base = selected_residual(z, slope_params, scheme)
    prior = np.asarray(
        [
            (slopes[0] - prior_slopes[0]) / sigma_slopes[0],
            (slopes[1] - prior_slopes[1]) / sigma_slopes[1],
        ],
        dtype=float,
    )
    return np.concatenate([base, prior])


def extended_sparsity(params: TransonicSlimParams, scheme: str):
    if scheme == "midpoint":
        base = sparsity_pattern(params, SONIC_MODE, SONIC_COMPONENTS)
        state_size = 2 * params.n_nodes + 2
        pattern = lil_matrix((base.shape[0] + 2, state_size + 2), dtype=int)
        pattern[: base.shape[0], :state_size] = base
    else:
        state_size = 2 * params.n_nodes + 2
        n_rows = 2 * (params.n_nodes - 1) + 2 + len(SONIC_COMPONENTS) + 2
        pattern = lil_matrix((n_rows, state_size + 2), dtype=int)
        row = 0
        for idx in range(params.n_nodes - 1):
            for col in (idx, idx + 1, params.n_nodes + idx, params.n_nodes + idx + 1, state_size - 2, state_size - 1):
                pattern[row : row + 2, col] = 1
            row += 2
        for col in (params.n_nodes - 1, 2 * params.n_nodes - 1, state_size - 2, state_size - 1):
            pattern[row : row + 2, col] = 1
        row += 2
        for col in (0, params.n_nodes, state_size - 2, state_size - 1):
            pattern[row : row + len(SONIC_COMPONENTS), col] = 1
    outer_row = 2 * (params.n_nodes - 1)
    pattern[outer_row : outer_row + 2, state_size : state_size + 2] = 1
    pattern[-2, state_size] = 1
    pattern[-1, state_size + 1] = 1
    return pattern.tocsr()


def solve_scheme(
    x0: np.ndarray,
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
    scheme: str,
):
    lower, upper = full_extended_bounds(params, prior_slopes)
    slope_lower = np.asarray(prior_slopes, dtype=float) - SLOPE_BOUND_HALF_WIDTH
    slope_upper = np.asarray(prior_slopes, dtype=float) + SLOPE_BOUND_HALF_WIDTH
    lower[-2:] = slope_lower
    upper[-2:] = slope_upper
    x_start = np.clip(np.asarray(x0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: extended_residual(trial, params, prior_slopes, SIGMA_SLOPES, scheme),
        x_start,
        jac_sparsity=extended_sparsity(params, scheme),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV,
    )


def remap_checkpoint_to_case(
    *,
    fiducial: FiducialParams,
    mdot_edd: float,
    checkpoint: Path,
    n_nodes: int,
) -> tuple[np.ndarray, TransonicSlimParams, dict[str, object], np.ndarray, TransonicSlimParams]:
    z_source, row = load_checkpoint(checkpoint)
    source_params = params_for(
        fiducial,
        mdot_edd,
        base_source_from_row(row),
        n_nodes=int(row["n_nodes"]),
        slopes=(float(row["g_u_solved"]), float(row["g_T_solved"])),
    )
    target_source = base_source_from_row(row)
    target_params = params_for(
        fiducial,
        mdot_edd,
        target_source,
        n_nodes=n_nodes,
        slopes=source_params.outer_match_log_slopes,
    )
    if n_nodes == source_params.n_nodes:
        z_seed = z_source
    else:
        z_seed = remap_state_pchip(z_source, source_params, target_params)
    return z_seed, target_params, row, z_source, source_params


def residual_limit(n_nodes: int) -> float:
    return 2.0e-6 if n_nodes >= 129 else 5.0e-6


def row_from_result(
    *,
    label: str,
    scheme: str,
    n_nodes: int,
    params: TransonicSlimParams,
    prior_slopes: tuple[float, float],
    effective_points: int,
    effective_inner_fraction: float,
    z_seed: np.ndarray,
    source_z: np.ndarray,
    source_params: TransonicSlimParams,
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    result,
) -> dict[str, object]:
    z, solved_slopes = unpack_unknown(np.asarray(result.x, dtype=float), params)
    solved_params = replace(params, outer_match_log_slopes=solved_slopes)
    selected = selected_residual(z, solved_params, scheme)
    extended = extended_residual(result.x, params, prior_slopes, SIGMA_SLOPES, scheme)
    audit = residual_audit_from_state_vector(z, solved_params)
    profile = profile_from_state_vector(z, solved_params)
    source_branch = branch_metrics(z, solved_params, source_z, source_params)
    anchor_branch = branch_metrics(z, solved_params, anchor_z, anchor_params)
    prior_residual = np.asarray(
        [
            (solved_slopes[0] - prior_slopes[0]) / SIGMA_SLOPES[0],
            (solved_slopes[1] - prior_slopes[1]) / SIGMA_SLOPES[1],
        ],
        dtype=float,
    )
    physical = active_physical_max(audit)
    residual_pass = bool(physical <= residual_limit(n_nodes))
    source_branch_pass = branch_pass(source_branch)
    anchor_branch_pass = branch_pass(anchor_branch)
    return {
        "label": label,
        "scheme": scheme,
        "n_nodes": n_nodes,
        "ratio": solved_params.mdot_edd_ratio,
        "R_out_rg": solved_params.R_out_rg,
        "g_u_prior": float(prior_slopes[0]),
        "g_T_prior": float(prior_slopes[1]),
        "g_u_solved": float(solved_slopes[0]),
        "g_T_solved": float(solved_slopes[1]),
        "delta_g_u": float(solved_slopes[0] - prior_slopes[0]),
        "delta_g_T": float(solved_slopes[1] - prior_slopes[1]),
        "prior_max": float(np.max(np.abs(prior_residual))),
        "prior_chi2": float(np.dot(prior_residual, prior_residual)),
        "effective_points": effective_points,
        "effective_inner_fraction": effective_inner_fraction,
        "selected_max": float(np.max(np.abs(selected))),
        "extended_max": float(np.max(np.abs(extended))),
        "physical_active": physical,
        "dominant": dominant_block(audit),
        "residual_limit": residual_limit(n_nodes),
        "residual_pass": residual_pass,
        "source_branch_pass": source_branch_pass,
        "anchor_branch_pass": anchor_branch_pass,
        "science_pass": bool(residual_pass and source_branch_pass and anchor_branch_pass),
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
        "int_adv": float(profile.integrated_advective_fraction),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": audit.outer_H_over_R,
        "source_delta_Rson_rg": source_branch["delta_Rson_rg"],
        "source_delta_lambda0": source_branch["delta_lambda0"],
        "source_delta_int_adv": source_branch["delta_int_adv"],
        "source_branch_distance": source_branch["branch_distance"],
        "anchor_delta_Rson_rg": anchor_branch["delta_Rson_rg"],
        "anchor_delta_lambda0": anchor_branch["delta_lambda0"],
        "anchor_delta_int_adv": anchor_branch["delta_int_adv"],
        "anchor_branch_distance": anchor_branch["branch_distance"],
        "seed_delta_norm": float(np.linalg.norm(z - z_seed) / np.sqrt(len(z))),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{row['label']}_{row['scheme']}_{float(row['ratio']):.8f}".replace(".", "p")
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
        "# Trapezoid Collocation Audit",
        "",
        "Generated by `scripts/run_transonic_trapezoid_collocation_audit.py`.",
        "",
        "Compares the existing midpoint interval residual against a mixed trapezoidal residual. The first sonic-adjacent interval remains midpoint; all outer intervals use trapezoid. All rows are judged by the existing differential physical audit, not by the selected collocation residual alone.",
        "",
        "| label | scheme | N | selected | physical | limit | dominant | residual pass | source branch | anchor branch | science pass | g_u prior | g_u solved | dg_u | g_T prior | g_T solved | dg_T | prior max | Rson/rg | dRson source | dRson anchor | lambda0 | dlambda source | int adv | dint source | branch source | branch anchor | interval R | interval E | outer 1 | D | C1 | C2 | K | seed norm | nfev | success | message |",
        "|---|---|---:|---:|---:|---:|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {scheme} | {n_nodes} | {selected_max} | {physical_active} | {residual_limit} | "
            "{dominant} | {residual_pass} | {source_branch_pass} | {anchor_branch_pass} | {science_pass} | "
            "{g_u_prior} | {g_u_solved} | {delta_g_u} | {g_T_prior} | {g_T_solved} | {delta_g_T} | "
            "{prior_max} | {Rson_rg} | {source_delta_Rson_rg} | {anchor_delta_Rson_rg} | {lambda0} | "
            "{source_delta_lambda0} | {int_adv} | {source_delta_int_adv} | {source_branch_distance} | "
            "{anchor_branch_distance} | {interval_R} | {interval_E} | {outer_1} | {D} | {C1} | {C2} | "
            "{K} | {seed_delta_norm} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                scheme=row["scheme"],
                n_nodes=row["n_nodes"],
                selected_max=fmt(float(row["selected_max"])),
                physical_active=fmt(float(row["physical_active"])),
                residual_limit=fmt(float(row["residual_limit"])),
                dominant=row["dominant"],
                residual_pass="yes" if row["residual_pass"] else "no",
                source_branch_pass="yes" if row["source_branch_pass"] else "no",
                anchor_branch_pass="yes" if row["anchor_branch_pass"] else "no",
                science_pass="yes" if row["science_pass"] else "no",
                g_u_prior=fmt(float(row["g_u_prior"])),
                g_u_solved=fmt(float(row["g_u_solved"])),
                delta_g_u=fmt(float(row["delta_g_u"])),
                g_T_prior=fmt(float(row["g_T_prior"])),
                g_T_solved=fmt(float(row["g_T_solved"])),
                delta_g_T=fmt(float(row["delta_g_T"])),
                prior_max=fmt(float(row["prior_max"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                source_delta_Rson_rg=fmt(float(row["source_delta_Rson_rg"])),
                anchor_delta_Rson_rg=fmt(float(row["anchor_delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                source_delta_lambda0=fmt(float(row["source_delta_lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                source_delta_int_adv=fmt(float(row["source_delta_int_adv"])),
                source_branch_distance=fmt(float(row["source_branch_distance"])),
                anchor_branch_distance=fmt(float(row["anchor_branch_distance"])),
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                outer_1=fmt(float(row["outer_1"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                seed_delta_norm=fmt(float(row["seed_delta_norm"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_row = load_checkpoint(N65_SOURCE_CHECKPOINT)
    anchor_params = params_for(
        fiducial,
        mdot_edd,
        base_source_from_row(anchor_row),
        n_nodes=65,
        slopes=(float(anchor_row["g_u_solved"]), float(anchor_row["g_T_solved"])),
    )

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for label, n_nodes, checkpoint in CASES:
        if not checkpoint.exists():
            print(f"skip {label}: missing {checkpoint}", flush=True)
            continue
        z_seed, seed_params, _source_row, source_z, source_params = remap_checkpoint_to_case(
            fiducial=fiducial,
            mdot_edd=mdot_edd,
            checkpoint=checkpoint,
            n_nodes=n_nodes,
        )
        prior_gu, prior_gT, effective_points, effective_inner_fraction = fit_outer_slopes(
            z_seed,
            seed_params,
            mode=PRIOR_MODE,
            window_fraction=PRIOR_WINDOW_FRACTION,
            min_points=PRIOR_MIN_POINTS,
        )
        prior_slopes = (prior_gu, prior_gT)
        params = replace(seed_params, outer_match_log_slopes=prior_slopes)
        x_seed = np.concatenate([z_seed, np.asarray(prior_slopes, dtype=float)])

        for scheme in SCHEMES:
            result = solve_scheme(x_seed, params, prior_slopes, scheme)
            row = row_from_result(
                label=label,
                scheme=scheme,
                n_nodes=n_nodes,
                params=params,
                prior_slopes=prior_slopes,
                effective_points=effective_points,
                effective_inner_fraction=effective_inner_fraction,
                z_seed=z_seed,
                source_z=source_z,
                source_params=source_params,
                anchor_z=anchor_z,
                anchor_params=anchor_params,
                result=result,
            )
            rows.append(row)
            save_checkpoint(row)
            write_table(rows)
            print(
                f"{label} {scheme} physical={row['physical_active']:.3e} "
                f"selected={row['selected_max']:.3e} Rson={row['Rson_rg']:.4f} "
                f"science={row['science_pass']} nfev={row['nfev']}",
                flush=True,
            )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
