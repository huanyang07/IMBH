"""Predictor audit for standard slim-disk Mdot continuation."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    pack_state,
    remap_profile_to_new_sonic_grid,
    residual_audit_from_state_vector,
    solve_square_transonic_polish,
    square_collocation_jacobian,
    square_collocation_residual,
    state_bounds,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe, local_thin_targets, solve_local_thin_state
from run_standard_slim_mdot_injection_ladder import ACCEPTANCE_TOL, STRESS_FACTOR, dominant


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_TABLE",
    "outputs/tables/slim_benchmark_mdot_predictor_audit.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_FIGURE",
    "outputs/figures/slim_benchmark_mdot_predictor_audit.png",
)
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_ANCHOR",
    "outputs/checkpoints/slim_benchmark_rout_injection_ladder/Rout_10000.npz",
)

TARGET_RATIOS = tuple(
    float(piece)
    for piece in os.environ.get(
        "IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_TARGETS",
        "0.99e-3,0.98e-3,0.95e-3,0.90e-3,1.01e-3,1.02e-3,1.05e-3,1.10e-3",
    )
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
PREDICTORS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_PREDICTORS", "current_remap,thin_algebraic,tangent").replace(":", ",").split(",")
    if piece.strip()
)
PIVOT = os.environ.get("IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_PIVOT", "C2")
MU_FD_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_MU_FD_STEP", "3e-5"))
TANGENT_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_TANGENT_SOLVER", "equilibrated_lsmr")
TANGENT_DAMPING = float(os.environ.get("IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_TANGENT_DAMPING", "0.0"))
TANGENT_MAXITER = int(os.environ.get("IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_TANGENT_MAXITER", "2000"))
POLISH = os.environ.get("IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_POLISH", "0") != "0"
POLISH_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_POLISH_NFEV", "350"))
FALLBACK_LSQ_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_MDOT_PREDICTOR_AUDIT_FALLBACK_LSQ_NFEV", "80"))


def params_for(fiducial: FiducialParams, mdot_edd: float, ratio: float, R_out_rg: float, n_nodes: int) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(ratio) * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=float(R_out_rg),
        n_nodes=int(n_nodes),
        residual_tol=1.0e-8,
        max_nfev=max(POLISH_NFEV, FALLBACK_LSQ_NFEV),
        outer_closure="thin_value",
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def load_anchor(fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    if not ANCHOR_CHECKPOINT.exists():
        raise FileNotFoundError(f"anchor checkpoint not found: {ANCHOR_CHECKPOINT}")
    data = np.load(ANCHOR_CHECKPOINT, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    params = params_for(fiducial, mdot_edd, float(data["ratio"]), float(data["R_out_rg"]), int(data["n_nodes"]))
    return z, params


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


def clip_state(z: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    lower, upper = state_bounds(params)
    return np.clip(np.asarray(z, dtype=float), lower + 1.0e-12, upper - 1.0e-12)


def current_remap_predictor(anchor_z: np.ndarray, anchor_params: TransonicSlimParams, target_params: TransonicSlimParams) -> tuple[np.ndarray, dict[str, object]]:
    profile = transonic_profile_from_state_vector(anchor_z, anchor_params)
    return remap_profile_to_new_sonic_grid(profile, target_params), {"valid_fraction": 1.0, "local_norm_max": np.nan}


def thin_algebraic_predictor(anchor_z: np.ndarray, anchor_params: TransonicSlimParams, target_params: TransonicSlimParams) -> tuple[np.ndarray, dict[str, object]]:
    base_z, _base_meta = current_remap_predictor(anchor_z, anchor_params, target_params)
    base_profile = transonic_profile_from_state_vector(base_z, target_params)
    base_logu, base_logT, logR_son, lambda0, logR = unpack_state(base_z, target_params)
    Sigma = np.empty_like(logR)
    T = np.empty_like(logR)
    local_norm = np.empty_like(logR)
    valid = np.zeros_like(logR, dtype=bool)
    previous = None
    for idx, x in enumerate(logR):
        try:
            target = local_thin_targets(float(x), target_params, lambda0)
            if target["W_req"] <= 0.0 or target["Q_visc_thin"] <= 0.0:
                raise ValueError("invalid thin target")
            seed = previous
            if seed is None:
                seed = np.log(np.array([base_profile.Sigma[idx], base_profile.T[idx]], dtype=float))
            sigma, temp, norm = solve_local_thin_state(float(x), target_params, lambda0, seed)
            if not np.isfinite(norm) or norm > 1.0e-5:
                raise RuntimeError("local thin solve did not converge tightly")
            Sigma[idx] = sigma
            T[idx] = temp
            local_norm[idx] = norm
            valid[idx] = True
            previous = np.log(np.array([sigma, temp], dtype=float))
        except Exception:
            Sigma[idx] = float(base_profile.Sigma[idx])
            T[idx] = float(np.exp(base_logT[idx]))
            local_norm[idx] = np.nan
            previous = np.log(np.array([Sigma[idx], T[idx]], dtype=float))
    R = np.exp(logR)
    u = target_params.Mdot_g_s / (2.0 * np.pi * R * Sigma)
    z = pack_state(np.log(u), np.log(T), logR_son, lambda0)
    return clip_state(z, target_params), {
        "valid_fraction": float(np.mean(valid)),
        "local_norm_max": float(np.nanmax(local_norm)) if np.any(np.isfinite(local_norm)) else np.nan,
    }


def finite_difference_mdot_column(anchor_z: np.ndarray, anchor_params: TransonicSlimParams, *, pivot: str) -> np.ndarray:
    plus = replace(anchor_params, Mdot_g_s=anchor_params.Mdot_g_s * float(np.exp(MU_FD_STEP)))
    minus = replace(anchor_params, Mdot_g_s=anchor_params.Mdot_g_s * float(np.exp(-MU_FD_STEP)))
    f_plus = square_collocation_residual(anchor_z, plus, pivot=pivot)
    f_minus = square_collocation_residual(anchor_z, minus, pivot=pivot)
    return (f_plus - f_minus) / (2.0 * MU_FD_STEP)


def _equilibrated_tangent_solve(jac, rhs: np.ndarray, *, damping: float, use_direct: bool) -> tuple[np.ndarray, float]:
    try:
        from scipy.sparse import diags
        from scipy.sparse.linalg import lsmr, splu
    except Exception as exc:
        raise RuntimeError("scipy is required for equilibrated tangent predictor") from exc

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


def tangent_vector(anchor_z: np.ndarray, anchor_params: TransonicSlimParams, *, pivot: str) -> tuple[np.ndarray, dict[str, object]]:
    try:
        from scipy.sparse.linalg import lsmr, splu
    except Exception as exc:
        raise RuntimeError("scipy is required for tangent predictor") from exc
    jac = square_collocation_jacobian(anchor_z, anchor_params, pivot=pivot).tocsc()
    f_mu = finite_difference_mdot_column(anchor_z, anchor_params, pivot=pivot)
    scaled_residual = np.nan
    if TANGENT_SOLVER == "splu":
        method = "splu"
        dz_dmu = splu(jac, permc_spec="COLAMD").solve(-f_mu)
        residual = jac @ dz_dmu + f_mu
    elif TANGENT_SOLVER == "lsmr":
        method = "lsmr"
        result = lsmr(
            jac,
            -f_mu,
            damp=TANGENT_DAMPING,
            atol=1.0e-10,
            btol=1.0e-10,
            maxiter=max(TANGENT_MAXITER, 5 * jac.shape[1]),
        )
        dz_dmu = np.asarray(result[0], dtype=float)
        residual = jac @ dz_dmu + f_mu
    elif TANGENT_SOLVER == "equilibrated_lsmr":
        method = "equilibrated_lsmr"
        dz_dmu, scaled_residual = _equilibrated_tangent_solve(jac, -f_mu, damping=TANGENT_DAMPING, use_direct=False)
        residual = jac @ dz_dmu + f_mu
    elif TANGENT_SOLVER == "equilibrated_direct":
        method = "equilibrated_direct"
        dz_dmu, scaled_residual = _equilibrated_tangent_solve(jac, -f_mu, damping=TANGENT_DAMPING, use_direct=True)
        residual = jac @ dz_dmu + f_mu
    else:
        raise ValueError("TANGENT_SOLVER must be 'lsmr', 'splu', 'equilibrated_lsmr', or 'equilibrated_direct'")
    try:
        condition = float(np.linalg.cond(jac.toarray()))
    except Exception:
        condition = np.nan
    test_params = replace(anchor_params, Mdot_g_s=anchor_params.Mdot_g_s * float(np.exp(MU_FD_STEP)))
    test_z = clip_state(anchor_z + MU_FD_STEP * dz_dmu, test_params)
    test_residual = float(np.max(np.abs(square_collocation_residual(test_z, test_params, pivot=pivot))))
    return np.asarray(dz_dmu, dtype=float), {
        "tangent_method": method,
        "tangent_damping": float(TANGENT_DAMPING),
        "tangent_maxiter": int(TANGENT_MAXITER),
        "tangent_linear_residual": float(np.max(np.abs(residual))),
        "tangent_scaled_residual": float(scaled_residual),
        "tangent_test_square": test_residual,
        "tangent_condition": condition,
    }


def tangent_predictor(
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    target_params: TransonicSlimParams,
    dz_dmu: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    dmu = float(np.log(target_params.Mdot_g_s / anchor_params.Mdot_g_s))
    return clip_state(anchor_z + dmu * dz_dmu, target_params), {"valid_fraction": 1.0, "local_norm_max": np.nan}


def quick_polish(z0: np.ndarray, params: TransonicSlimParams, *, pivot: str):
    first = solve_square_transonic_polish(
        params,
        z0,
        pivot=pivot,
        method="newton",
        max_nfev=POLISH_NFEV,
        residual_tol=1.0e-8,
        use_block_jacobian=True,
        linear_solver="direct",
        max_step_norm=0.5,
    )
    if max_residual(first.z, params) <= ACCEPTANCE_TOL or FALLBACK_LSQ_NFEV <= 0:
        return first
    return solve_square_transonic_polish(
        params,
        first.z,
        pivot=pivot,
        method="least_squares",
        max_nfev=FALLBACK_LSQ_NFEV,
        residual_tol=1.0e-8,
        use_block_jacobian=True,
    )


def predictor_state(
    predictor: str,
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    target_params: TransonicSlimParams,
    dz_dmu: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, object]]:
    if predictor == "current_remap":
        return current_remap_predictor(anchor_z, anchor_params, target_params)
    if predictor == "thin_algebraic":
        return thin_algebraic_predictor(anchor_z, anchor_params, target_params)
    if predictor == "tangent":
        if dz_dmu is None:
            raise RuntimeError("tangent predictor requested but tangent vector is unavailable")
        return tangent_predictor(anchor_z, anchor_params, target_params, dz_dmu)
    raise ValueError(f"unknown predictor {predictor!r}")


def row_for_prediction(
    *,
    predictor: str,
    ratio: float,
    z: np.ndarray,
    params: TransonicSlimParams,
    meta: dict[str, object],
    tangent_meta: dict[str, object],
) -> dict[str, object]:
    audit = residual_audit_from_state_vector(z, params)
    square = square_collocation_residual(z, params, pivot=PIVOT)
    row: dict[str, Any] = {
        "predictor": predictor,
        "ratio": float(ratio),
        "dmu": float(np.log(ratio / ANCHOR_RATIO_FOR_OUTPUT)),
        "initial_full": max_residual(z, params),
        "initial_square": float(np.max(np.abs(square))),
        "initial_dominant": dominant(audit),
        "initial_interval_R": float(audit.interval_radial_max),
        "initial_interval_E": float(audit.interval_energy_max),
        "valid_fraction": float(meta.get("valid_fraction", np.nan)),
        "local_norm_max": float(meta.get("local_norm_max", np.nan)),
        "final_full": np.nan,
        "final_square": np.nan,
        "final_dominant": "-",
        "polish_nfev": 0,
        "polish_success": False,
        "polish_message": "-",
    }
    row.update(tangent_meta if predictor == "tangent" else {})
    if POLISH:
        polish = quick_polish(z, params, pivot=PIVOT)
        final_audit = residual_audit_from_state_vector(polish.z, params)
        row.update(
            {
                "final_full": max_residual(polish.z, params),
                "final_square": float(np.max(np.abs(square_collocation_residual(polish.z, params, pivot=PIVOT)))),
                "final_dominant": dominant(final_audit),
                "polish_nfev": int(polish.result.nfev),
                "polish_success": bool(polish.result.optimizer_success),
                "polish_message": str(polish.result.message),
            }
        )
    return row


def write_table(rows: list[dict[str, object]], tangent_meta: dict[str, object], anchor_ratio: float) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Mdot Predictor Audit",
        "",
        "Generated by `scripts/run_standard_slim_mdot_predictor_audit.py`.",
        "",
        f"Anchor `{anchor_ratio:g}`, pivot `{PIVOT}`, finite-difference dmu `{MU_FD_STEP:g}`, polish `{POLISH}`.",
        "",
        "Tangent metadata:",
        "",
    ]
    for key, value in tangent_meta.items():
        lines.append(f"- `{key}`: `{fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value}`")
    lines.extend(
        [
            "",
            "| predictor | Mdot/Edd | dmu | initial full | initial square | dominant | int R | int E | valid frac | local norm | final full | final square | final dominant | nfev | success | message |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|:---:|---|",
        ]
    )
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {predictor} | {ratio} | {dmu} | {initial_full} | {initial_square} | {initial_dominant} | "
            "{initial_interval_R} | {initial_interval_E} | {valid_fraction} | {local_norm_max} | "
            "{final_full} | {final_square} | {final_dominant} | {polish_nfev} | {polish_success} | {polish_message} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps(json_safe({"tangent": tangent_meta, "rows": rows}), indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    width, height = 1150, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = 90, 80, 1080, 600
    draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
    predictors = list(dict.fromkeys(str(row["predictor"]) for row in rows))
    colors = {
        "current_remap": (31, 119, 180),
        "thin_algebraic": (44, 160, 44),
        "tangent": (214, 39, 40),
    }
    ratios = sorted(set(float(row["ratio"]) for row in rows))
    x_values = np.asarray([np.log10(ratio) for ratio in ratios], dtype=float)
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    if x_max <= x_min:
        x_max = x_min + 1.0
    y_values = [float(row["initial_full"]) for row in rows if np.isfinite(float(row["initial_full"])) and float(row["initial_full"]) > 0.0]
    y_log = np.log10(np.maximum(np.asarray(y_values, dtype=float), 1.0e-16))
    y_min, y_max = float(np.floor(np.min(y_log))), float(np.ceil(np.max(y_log)))
    if y_max <= y_min:
        y_max = y_min + 1.0
    for predictor in predictors:
        selected = sorted([row for row in rows if row["predictor"] == predictor], key=lambda row: float(row["ratio"]))
        points = []
        for row in selected:
            xx = np.log10(float(row["ratio"]))
            yy = np.log10(max(float(row["initial_full"]), 1.0e-16))
            px = x0 + int((xx - x_min) / (x_max - x_min) * (x1 - x0))
            py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
            points.append((px, py))
        color = colors.get(predictor, (20, 20, 20))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
    for idx, predictor in enumerate(predictors):
        y = 25 + 18 * idx
        color = colors.get(predictor, (20, 20, 20))
        draw.line((90, y + 6, 120, y + 6), fill=color, width=3)
        draw.text((130, y), predictor, fill=(20, 20, 20), font=font)
    draw.text((700, 25), "Initial full residual vs log10(Mdot/Edd)", fill=(20, 20, 20), font=font)
    draw.text((x0 + 4, y0 + 4), f"1e{int(y_max)}", fill=(80, 80, 80), font=font)
    draw.text((x0 + 4, y1 - 18), f"1e{int(y_min)}", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


ANCHOR_RATIO_FOR_OUTPUT = 1.0e-3


def main() -> None:
    global ANCHOR_RATIO_FOR_OUTPUT
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = load_anchor(fiducial, mdot_edd)
    ANCHOR_RATIO_FOR_OUTPUT = float(anchor_params.mdot_edd_ratio)
    dz_dmu = None
    tangent_meta: dict[str, object] = {}
    if "tangent" in PREDICTORS:
        print("building tangent predictor", flush=True)
        dz_dmu, tangent_meta = tangent_vector(anchor_z, anchor_params, pivot=PIVOT)
        tangent_meta["dz_dmu_inf"] = float(np.linalg.norm(dz_dmu, ord=np.inf))
        tangent_meta["dz_dmu_rms"] = float(np.sqrt(np.mean(dz_dmu**2)))
    rows: list[dict[str, object]] = []
    for ratio in TARGET_RATIOS:
        target_params = params_for(fiducial, mdot_edd, ratio, anchor_params.R_out_rg, anchor_params.n_nodes)
        print(f"target ratio={ratio:g}", flush=True)
        for predictor in PREDICTORS:
            z_pred, meta = predictor_state(predictor, anchor_z, anchor_params, target_params, dz_dmu)
            row = row_for_prediction(
                predictor=predictor,
                ratio=ratio,
                z=z_pred,
                params=target_params,
                meta=meta,
                tangent_meta=tangent_meta,
            )
            rows.append(row)
            print(
                f"  {predictor}: initial={row['initial_full']:.3e} dom={row['initial_dominant']} "
                f"final={row['final_full'] if np.isfinite(row['final_full']) else np.nan:.3e}",
                flush=True,
            )
            write_table(rows, tangent_meta, ANCHOR_RATIO_FOR_OUTPUT)
            write_figure(rows)
    write_table(rows, tangent_meta, ANCHOR_RATIO_FOR_OUTPUT)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
