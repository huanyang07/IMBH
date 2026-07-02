"""Source-fraction tangent predictor audit for stream-fed slim-disk branches."""

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
    remap_profile_to_new_sonic_grid,
    residual_audit_from_state_vector,
    solve_square_transonic_polish,
    square_collocation_jacobian,
    square_collocation_residual,
    state_bounds,
    transonic_profile_from_state_vector,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import fmt, json_safe
from run_standard_slim_stream_anchor_regression import (
    ROOT,
    advection_diagnostic,
    dominant,
    interval_peak_diagnostic,
    max_residual,
    params_from_checkpoint,
    refresh_outer_slopes_from_state,
    stream_diagnostic,
)


TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_TANGENT_TABLE",
    "outputs/tables/standard_slim_stream_source_tangent_audit.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")

DEFAULT_CASES = (
    "fs080_to0805="
    "outputs/checkpoints/standard_slim_stream_residual_remesh_fs070_N640_N768_N896_s12/"
    "N768_s12_mass_0p7_torque_0p005_mdot_2_N768.npz,"
    "outputs/checkpoints/standard_slim_stream_residual_remesh_fs080_N640_N768_N896_s12/"
    "N768_s12_mass_0p8_torque_0p005_mdot_2_N768.npz,0.805;"
    "fs0805_to0808585="
    "outputs/checkpoints/standard_slim_stream_residual_remesh_fs080_N640_N768_N896_s12/"
    "N768_s12_mass_0p8_torque_0p005_mdot_2_N768.npz,"
    "outputs/checkpoints/standard_slim_stream_residual_remesh_fs0805_N640_N768_N896_s12/"
    "N768_s12_mass_0p805_torque_0p005_mdot_2_N768.npz,0.808585;"
    "fs0808585_to081="
    "outputs/checkpoints/standard_slim_stream_residual_remesh_fs0805_N640_N768_N896_s12/"
    "N768_s12_mass_0p805_torque_0p005_mdot_2_N768.npz,"
    "outputs/checkpoints/standard_slim_stream_residual_remesh_fs0808585_N768_N896_s12/"
    "N768_s12_mass_0p808585_torque_0p005_mdot_2_N768.npz,0.81"
)

CASE_SPECS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_CASES", DEFAULT_CASES).split(";")
    if piece.strip()
)
PREDICTORS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_PREDICTORS", "current,secant,tangent").replace(":", ",").split(",")
    if piece.strip()
)
PIVOT = os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_PIVOT", "C2")
SOURCE_FD_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_FD_STEP", "1e-5"))
TANGENT_DAMPINGS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_DAMPINGS", "1,0.5,0.25,0.1").replace(":", ",").split(",")
    if piece.strip()
)
SECANT_DAMPINGS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_SECANT_DAMPINGS", "1,0.5,0.25,0.1").replace(":", ",").split(",")
    if piece.strip()
)
TANGENT_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_SOLVER", "equilibrated_lsmr")
TANGENT_LINEAR_DAMPING = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_LINEAR_DAMPING", "0.0"))
TANGENT_MAXITER = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_MAXITER", "3000"))
POLISH = os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_POLISH", "1") != "0"
POLISH_PREDICTORS = {
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_POLISH_PREDICTORS", ",".join(PREDICTORS)).replace(":", ",").split(",")
    if piece.strip()
}
POLISH_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_POLISH_MAX_ITER", "32"))
POLISH_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_POLISH_NFEV", "3600"))
POLISH_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_POLISH_MAX_STEP_NORM", "0.16"))
POLISH_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_POLISH_LINEAR_SOLVER", "regularized_lsmr")
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_TANGENT_ANCHOR_TOL", "3e-6"))


def parse_case_specs() -> list[tuple[str, Path, Path, float]]:
    cases: list[tuple[str, Path, Path, float]] = []
    for spec in CASE_SPECS:
        if "=" not in spec:
            raise ValueError("case specs must use label=prev,current,target_fraction")
        label, payload = spec.split("=", 1)
        pieces = [piece.strip() for piece in payload.split(",") if piece.strip()]
        if len(pieces) != 3:
            raise ValueError(f"case {label!r} must contain prev,current,target_fraction")
        cases.append((label.strip(), ROOT / pieces[0], ROOT / pieces[1], float(pieces[2])))
    return cases


def clip_state(z: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    lower, upper = state_bounds(params)
    return np.clip(np.asarray(z, dtype=float), lower + 1.0e-12, upper - 1.0e-12)


def source_fraction(params: TransonicSlimParams) -> float:
    return float(params.stream_source_fraction if params.stream_source_fraction != 0.0 else params.stream_mass_fraction)


def params_with_source_fraction(params: TransonicSlimParams, fraction: float) -> TransonicSlimParams:
    return replace(
        params,
        stream_source_fraction=float(fraction),
        stream_mass_fraction=0.0,
        max_nfev=POLISH_NFEV,
        residual_tol=1.0e-8,
    )


def remap_to_params(z: np.ndarray, params: TransonicSlimParams, target_params: TransonicSlimParams) -> np.ndarray:
    profile = transonic_profile_from_state_vector(z, params)
    return remap_profile_to_new_sonic_grid(profile, target_params, temperature_mdot_power=0.0)


def previous_on_current_grid(
    prev_z: np.ndarray,
    prev_params: TransonicSlimParams,
    current_params: TransonicSlimParams,
) -> tuple[np.ndarray, TransonicSlimParams]:
    target_params = params_with_source_fraction(current_params, source_fraction(prev_params))
    remapped = remap_to_params(prev_z, prev_params, target_params)
    target_params = refresh_outer_slopes_from_state(remapped, target_params)
    return clip_state(remapped, target_params), target_params


def finite_difference_source_column(anchor_z: np.ndarray, anchor_params: TransonicSlimParams, *, pivot: str) -> tuple[np.ndarray, float]:
    f0 = source_fraction(anchor_params)
    step = min(abs(float(SOURCE_FD_STEP)), 0.25 * max(f0, 1.0e-3), 0.25 * max(1.0 - f0, 1.0e-3))
    if step <= 0.0:
        raise ValueError("source finite-difference step collapsed")
    if f0 - step >= 0.0 and f0 + step < 1.0 + anchor_params.wind_sink_fraction:
        plus = params_with_source_fraction(anchor_params, f0 + step)
        minus = params_with_source_fraction(anchor_params, f0 - step)
        f_plus = square_collocation_residual(anchor_z, plus, pivot=pivot)
        f_minus = square_collocation_residual(anchor_z, minus, pivot=pivot)
        return (f_plus - f_minus) / (2.0 * step), step
    plus = params_with_source_fraction(anchor_params, f0 + step)
    f_base = square_collocation_residual(anchor_z, anchor_params, pivot=pivot)
    f_plus = square_collocation_residual(anchor_z, plus, pivot=pivot)
    return (f_plus - f_base) / step, step


def equilibrated_tangent_solve(jac, rhs: np.ndarray, *, damping: float, use_direct: bool) -> tuple[np.ndarray, float]:
    try:
        from scipy.sparse import diags
        from scipy.sparse.linalg import lsmr, splu
    except Exception as exc:
        raise RuntimeError("scipy is required for tangent predictor") from exc

    jac_csr = jac.tocsr()
    row_norm = np.sqrt(np.asarray(jac_csr.multiply(jac_csr).sum(axis=1)).ravel())
    row_scale = 1.0 / np.maximum(row_norm, 1.0e-12)
    row_scaled = diags(row_scale) @ jac_csr
    col_norm = np.sqrt(np.asarray(row_scaled.multiply(row_scaled).sum(axis=0)).ravel())
    col_scale = 1.0 / np.maximum(col_norm, 1.0e-12)
    balanced = (row_scaled @ diags(col_scale)).tocsc()
    scaled_rhs = row_scale * np.asarray(rhs, dtype=float)
    if use_direct and damping == 0.0:
        try:
            y = splu(balanced, permc_spec="COLAMD").solve(scaled_rhs)
            residual = balanced @ y - scaled_rhs
            return col_scale * np.asarray(y, dtype=float), float(np.max(np.abs(residual)))
        except Exception:
            pass
    result = lsmr(
        balanced,
        scaled_rhs,
        damp=float(damping),
        atol=1.0e-12,
        btol=1.0e-12,
        maxiter=max(TANGENT_MAXITER, 10 * balanced.shape[1]),
    )
    y = np.asarray(result[0], dtype=float)
    residual = balanced @ y - scaled_rhs
    return col_scale * y, float(np.max(np.abs(residual)))


def source_tangent_vector(anchor_z: np.ndarray, anchor_params: TransonicSlimParams, *, pivot: str) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from scipy.sparse.linalg import lsmr, splu
    except Exception as exc:
        raise RuntimeError("scipy is required for tangent predictor") from exc

    t0 = time.perf_counter()
    jac = square_collocation_jacobian(anchor_z, anchor_params, pivot=pivot)
    build_elapsed = time.perf_counter() - t0
    f_source, fd_step = finite_difference_source_column(anchor_z, anchor_params, pivot=pivot)
    linear_start = time.perf_counter()
    if TANGENT_SOLVER == "splu":
        method = "splu"
        dz_df = splu(jac.tocsc(), permc_spec="COLAMD").solve(-f_source)
        scaled_residual = np.nan
    elif TANGENT_SOLVER == "lsmr":
        method = "lsmr"
        result = lsmr(
            jac.tocsr(),
            -f_source,
            damp=TANGENT_LINEAR_DAMPING,
            atol=1.0e-10,
            btol=1.0e-10,
            maxiter=max(TANGENT_MAXITER, 5 * jac.shape[1]),
        )
        dz_df = np.asarray(result[0], dtype=float)
        scaled_residual = np.nan
    elif TANGENT_SOLVER == "equilibrated_lsmr":
        method = "equilibrated_lsmr"
        dz_df, scaled_residual = equilibrated_tangent_solve(jac, -f_source, damping=TANGENT_LINEAR_DAMPING, use_direct=False)
    elif TANGENT_SOLVER == "equilibrated_direct":
        method = "equilibrated_direct"
        dz_df, scaled_residual = equilibrated_tangent_solve(jac, -f_source, damping=TANGENT_LINEAR_DAMPING, use_direct=True)
    else:
        raise ValueError("unknown tangent solver")
    linear_elapsed = time.perf_counter() - linear_start
    residual = jac @ dz_df + f_source
    meta = {
        "tangent_method": method,
        "source_fd_step": float(fd_step),
        "linear_damping": float(TANGENT_LINEAR_DAMPING),
        "linear_residual_max": float(np.max(np.abs(residual))),
        "linear_scaled_residual_max": float(scaled_residual),
        "dz_df_inf": float(np.linalg.norm(dz_df, ord=np.inf)),
        "dz_df_rms": float(np.sqrt(np.mean(dz_df**2))),
        "jacobian_shape": list(jac.shape),
        "jacobian_nnz": int(jac.nnz),
        "jacobian_build_elapsed_s": float(build_elapsed),
        "linear_solve_elapsed_s": float(linear_elapsed),
    }
    return np.asarray(dz_df, dtype=float), meta


def current_seed(current_z: np.ndarray, target_params: TransonicSlimParams) -> tuple[np.ndarray, dict[str, Any]]:
    return clip_state(current_z, target_params), {"seed_detail": "current"}


def secant_seed(
    prev_z_on_grid: np.ndarray,
    current_z: np.ndarray,
    current_fraction: float,
    prev_fraction: float,
    target_fraction: float,
    target_params: TransonicSlimParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    if abs(current_fraction - prev_fraction) <= 1.0e-14:
        return clip_state(current_z, target_params), {"seed_detail": "secant_unavailable", "seed_damping": 0.0}
    step_factor = (float(target_fraction) - current_fraction) / (current_fraction - prev_fraction)
    best_z = clip_state(current_z, target_params)
    best_full = max_residual(best_z, target_params)
    best_damping = 0.0
    for damping in SECANT_DAMPINGS:
        candidate = clip_state(current_z + float(damping) * step_factor * (current_z - prev_z_on_grid), target_params)
        full = max_residual(candidate, target_params)
        if full < best_full:
            best_z = candidate
            best_full = full
            best_damping = float(damping)
    return best_z, {"seed_detail": f"secant:{best_damping:g}", "seed_damping": float(best_damping)}


def tangent_seed(
    current_z: np.ndarray,
    current_fraction: float,
    target_fraction: float,
    target_params: TransonicSlimParams,
    dz_df: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    df = float(target_fraction) - float(current_fraction)
    best_z = clip_state(current_z, target_params)
    best_full = max_residual(best_z, target_params)
    best_damping = 0.0
    for damping in TANGENT_DAMPINGS:
        candidate = clip_state(current_z + float(damping) * df * dz_df, target_params)
        full = max_residual(candidate, target_params)
        if full < best_full:
            best_z = candidate
            best_full = full
            best_damping = float(damping)
    return best_z, {"seed_detail": f"tangent:{best_damping:g}", "seed_damping": float(best_damping)}


def polish_best(z0: np.ndarray, params: TransonicSlimParams):
    best = None
    best_full = np.inf
    for pivot in (PIVOT, "C1" if PIVOT != "C1" else "C2"):
        result = solve_square_transonic_polish(
            params,
            z0,
            pivot=pivot,
            method="newton",
            max_iter=POLISH_MAX_ITER,
            max_nfev=POLISH_NFEV,
            residual_tol=1.0e-8,
            use_block_jacobian=True,
            linear_solver=POLISH_LINEAR_SOLVER,
            max_step_norm=POLISH_MAX_STEP_NORM,
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


def row_for_seed(
    *,
    case: str,
    predictor: str,
    z: np.ndarray,
    target_params: TransonicSlimParams,
    meta: dict[str, Any],
    tangent_meta: dict[str, Any],
) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(z, target_params)
    square = square_collocation_residual(z, target_params, pivot=PIVOT)
    row: dict[str, Any] = {
        "case": case,
        "predictor": predictor,
        "seed_detail": str(meta.get("seed_detail", predictor)),
        "seed_damping": float(meta.get("seed_damping", np.nan)),
        "target_fraction": source_fraction(target_params),
        "ratio": float(target_params.mdot_edd_ratio),
        "R_out_rg": float(target_params.R_out_rg),
        "N": int(target_params.n_nodes),
        "initial_full": max_residual(z, target_params),
        "initial_square": float(np.max(np.abs(square))),
        "initial_dominant": dominant(audit),
        "initial_interval_R": float(audit.interval_radial_max),
        "initial_interval_E": float(audit.interval_energy_max),
        "initial_outer_omega": float(audit.outer_omega),
        "initial_peak_E_rg": float(interval_peak_diagnostic(z, target_params)["peak_interval_E_rg"]),
        "final_full": np.nan,
        "final_square": np.nan,
        "final_dominant": "-",
        "final_interval_E": np.nan,
        "final_outer_omega": np.nan,
        "final_peak_E_rg": np.nan,
        "final_f_adv_global": np.nan,
        "final_f_adv_inner": np.nan,
        "final_Lrad_LEdd": np.nan,
        "final_max_H_R": np.nan,
        "final_Rson_rg": np.nan,
        "polish_nfev": 0,
        "polish_iterations": 0,
        "polish_success": False,
        "polish_pivot": "-",
        "polish_message": "-",
    }
    if predictor == "tangent":
        row.update(tangent_meta)
    if POLISH and predictor in POLISH_PREDICTORS:
        t0 = time.perf_counter()
        polish = polish_best(z, target_params)
        elapsed = time.perf_counter() - t0
        final_params = refresh_outer_slopes_from_state(polish.z, target_params)
        final_audit = residual_audit_from_state_vector(polish.z, final_params)
        profile = transonic_profile_from_state_vector(polish.z, final_params)
        row.update(
            {
                "final_full": max_residual(polish.z, final_params),
                "final_square": float(np.max(np.abs(square_collocation_residual(polish.z, final_params, pivot=polish.pivot)))),
                "final_dominant": dominant(final_audit),
                "final_interval_E": float(final_audit.interval_energy_max),
                "final_outer_omega": float(final_audit.outer_omega),
                "final_peak_E_rg": float(interval_peak_diagnostic(polish.z, final_params)["peak_interval_E_rg"]),
                "final_max_H_R": float(np.max(profile.H_over_R)),
                "final_Rson_rg": float(profile.sonic_radius / final_params.r_g),
                "polish_nfev": int(polish.result.nfev),
                "polish_iterations": int(polish.iterations),
                "polish_success": bool(polish.result.optimizer_success),
                "polish_pivot": str(polish.pivot),
                "polish_elapsed_s": float(elapsed),
                "polish_message": str(polish.result.message),
                **{f"final_{key}": value for key, value in advection_diagnostic(polish.z, final_params).items()},
                **{f"final_{key}": value for key, value in stream_diagnostic(polish.z, final_params).items()},
            }
        )
    return row


def write_table(rows: list[dict[str, Any]], cases_meta: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Stream Source-Fraction Tangent Audit",
        "",
        "Generated by `scripts/run_standard_slim_stream_source_tangent_audit.py`.",
        "",
        f"Pivot `{PIVOT}`, finite-difference df `{SOURCE_FD_STEP:g}`, predictors `{','.join(PREDICTORS)}`, polish `{POLISH}`.",
        "",
        "Cases:",
        "",
    ]
    for meta in cases_meta:
        lines.append(
            f"- `{meta['case']}`: previous f_s `{meta['previous_fraction']:.8g}`, "
            f"current f_s `{meta['current_fraction']:.8g}`, target f_s `{meta['target_fraction']:.8g}`, "
            f"N `{meta['N']}`"
        )
    lines.extend(
        [
            "",
            "| case | predictor | seed | target f_s | initial full | initial square | dominant | int E | outer omega | peak E R/rg | final full | final dominant | final int E | final outer omega | final f_adv global | final f_adv inner | final Lrad/LEdd | final max H/R | final Rson/rg | nfev | success |",
            "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        formatted = {
            key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value
            for key, value in row.items()
        }
        formatted["target_fraction"] = f"{float(row['target_fraction']):.8g}"
        lines.append(
            "| {case} | {predictor} | {seed_detail} | {target_fraction} | {initial_full} | {initial_square} | "
            "{initial_dominant} | {initial_interval_E} | {initial_outer_omega} | {initial_peak_E_rg} | "
            "{final_full} | {final_dominant} | {final_interval_E} | {final_outer_omega} | "
            "{final_f_adv_global} | {final_f_adv_inner} | {final_Lrad_LEdd} | {final_max_H_R} | "
            "{final_Rson_rg} | {polish_nfev} | {polish_success} |".format(**formatted)
        )
    JSON_OUTPUT.write_text(json.dumps(json_safe({"cases": cases_meta, "rows": rows}), indent=2, sort_keys=True) + "\n")
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def run_case(label: str, prev_path: Path, current_path: Path, target_fraction: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    prev_z, prev_params = params_from_checkpoint(prev_path, fiducial, mdot_edd)
    current_z, current_params = params_from_checkpoint(current_path, fiducial, mdot_edd)
    current_fraction = source_fraction(current_params)
    prev_fraction = source_fraction(prev_params)
    target_params = params_with_source_fraction(current_params, float(target_fraction))
    target_params = refresh_outer_slopes_from_state(current_z, target_params)
    prev_on_grid, _prev_on_grid_params = previous_on_current_grid(prev_z, prev_params, current_params)
    tangent_meta: dict[str, Any] = {}
    dz_df = None
    if "tangent" in PREDICTORS:
        print(f"{label}: building source-fraction tangent at f_s={current_fraction:.8g}", flush=True)
        dz_df, tangent_meta = source_tangent_vector(current_z, current_params, pivot=PIVOT)
    case_meta = {
        "case": label,
        "previous_path": str(prev_path.relative_to(ROOT)),
        "current_path": str(current_path.relative_to(ROOT)),
        "previous_fraction": float(prev_fraction),
        "current_fraction": float(current_fraction),
        "target_fraction": float(target_fraction),
        "N": int(current_params.n_nodes),
        "current_full": max_residual(current_z, current_params),
        "previous_on_current_grid_full": max_residual(prev_on_grid, params_with_source_fraction(current_params, prev_fraction)),
        **{f"tangent_{key}": value for key, value in tangent_meta.items()},
    }
    rows: list[dict[str, Any]] = []
    for predictor in PREDICTORS:
        if predictor == "current":
            z_seed, meta = current_seed(current_z, target_params)
        elif predictor == "secant":
            z_seed, meta = secant_seed(prev_on_grid, current_z, current_fraction, prev_fraction, target_fraction, target_params)
        elif predictor == "tangent":
            if dz_df is None:
                raise RuntimeError("tangent predictor requested without tangent vector")
            z_seed, meta = tangent_seed(current_z, current_fraction, target_fraction, target_params, dz_df)
        else:
            raise ValueError(f"unknown predictor {predictor!r}")
        row = row_for_seed(case=label, predictor=predictor, z=z_seed, target_params=target_params, meta=meta, tangent_meta=tangent_meta)
        rows.append(row)
        print(
            f"  {label} {predictor} seed={row['seed_detail']} initial={row['initial_full']:.3e} "
            f"final={row['final_full']:.3e} nfev={row['polish_nfev']} success={row['polish_success']}",
            flush=True,
        )
    return rows, case_meta


def main() -> None:
    rows: list[dict[str, Any]] = []
    cases_meta: list[dict[str, Any]] = []
    for label, prev_path, current_path, target_fraction in parse_case_specs():
        case_rows, case_meta = run_case(label, prev_path, current_path, target_fraction)
        rows.extend(case_rows)
        cases_meta.append(case_meta)
        write_table(rows, cases_meta)
    write_table(rows, cases_meta)
    print(f"wrote {TABLE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
