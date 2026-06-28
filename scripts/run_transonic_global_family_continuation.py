"""Global-family lambda/theta continuation diagnostic for the transonic root.

This is the first Track-A experiment from the post-fold handoff.  Instead of
perturbing lambda while holding the sonic buffer fixed, the full dynamic
two-domain BVP is re-solved with a relaxed far thermal condition:

    theta_out = B_far,T^(0)

The solver replaces the far thermal residual with a lambda-target row.  A
physical member of this relaxed family must return theta_out ~= 0 while keeping
all other residual blocks small.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    B_rank_minors,
    algebraic_state,
    phase_space_null_tangent,
    phase_space_tangent_derivative,
    sonic_diagnostics,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_lambda_family_fold_map import refine_bracket
from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_dynamic_sonic_patch import (
    CHECKPOINT_DIR as DYNAMIC_CHECKPOINT_DIR,
    dynamic_audit,
    dynamic_patch_residual,
    load_row,
)
from run_transonic_two_domain_outer_extension import far_boundary_residual, integrated_advective_fraction, outer_grid, state_bounds_two_domain
from run_transonic_two_domain_sonic_refinement_sprint import buffer_inner_grid, make_buffer_params, unpack_buffer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = Path(
    os.environ.get(
        "IMBH_GLOBAL_FAMILY_SOURCE",
        str(DYNAMIC_CHECKPOINT_DIR / "Nreg64_0p90277664.npz"),
    )
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_GLOBAL_FAMILY_TABLE",
    "outputs/tables/transonic_global_family_continuation.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_GLOBAL_FAMILY_FIGURE",
    "outputs/figures/transonic_global_family_continuation.png",
)
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_global_family_continuation"

DELTA_LAMBDAS = tuple(
    float(piece)
    for piece in os.environ.get(
        "IMBH_GLOBAL_FAMILY_DELTA_LAMBDAS",
        "0,0.00025,0.0005,0.00075,0.001,0.0015,0.002,-0.00025,-0.0005,-0.00075,-0.001,-0.0015,-0.002",
    )
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
LAMBDA_SCALE = float(os.environ.get("IMBH_GLOBAL_FAMILY_LAMBDA_SCALE", "1e-4"))
MAX_NFEV = int(os.environ.get("IMBH_GLOBAL_FAMILY_MAX_NFEV", "450"))
POLISH_NFEV = int(os.environ.get("IMBH_GLOBAL_FAMILY_POLISH_NFEV", "120"))
SCIENCE_LIMIT = float(os.environ.get("IMBH_GLOBAL_FAMILY_SCIENCE_LIMIT", "5e-6"))
FOLD_PROBE_R_RG = float(os.environ.get("IMBH_GLOBAL_FAMILY_FOLD_PROBE_R_RG", "6.0"))
FOLD_TARGET_R_RG = float(os.environ.get("IMBH_GLOBAL_FAMILY_FOLD_TARGET_R_RG", "8.0"))
FOLD_DS = float(os.environ.get("IMBH_GLOBAL_FAMILY_FOLD_DS", "1e-3"))
FOLD_S_MAX = float(os.environ.get("IMBH_GLOBAL_FAMILY_FOLD_S_MAX", "5.0"))
FOLD_MAX_STEPS = int(os.environ.get("IMBH_GLOBAL_FAMILY_FOLD_MAX_STEPS", "6000"))
TANGENT_EPS = float(os.environ.get("IMBH_GLOBAL_FAMILY_TANGENT_EPS", "2e-6"))


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items() if key != "x"}, sort_keys=True)


def continuation_residual(x: np.ndarray, params, lambda_target: float) -> np.ndarray:
    """Dynamic two-domain residual with far thermal balance replaced by lambda target."""

    rows = []
    try:
        logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_buffer(x, params)
        logR_i = buffer_inner_grid(logR_son, params)
        logR_o = outer_grid(params)  # type: ignore[arg-type]
        if np.any(np.diff(logR_i) <= 0.0) or np.any(np.diff(logR_o) <= 0.0):
            raise ValueError("mapped radii must increase")

        rows.append(dynamic_patch_residual(logu_i, logT_i, logR_son, lambda0, params))
        for idx in range(1, params.n_inner - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx))
        for idx in range(params.n_outer - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx))
        rows.append(np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float))
        far = far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params)  # type: ignore[arg-type]
        rows.append(np.array([far[0], (lambda0 - lambda_target) / LAMBDA_SCALE], dtype=float))
        sonic = sonic_diagnostics(logR_son, np.array([logu_i[0], logT_i[0]], dtype=float), lambda0, params.physics)
        rows.append(np.array([sonic.D, sonic.C1, sonic.C2, sonic.compatibility], dtype=float))
        return np.concatenate(rows)
    except Exception:
        return np.full(2 * params.n_inner + 2 * params.n_outer + 4, 1.0e6)


def continuation_sparsity(params):
    """Sparse pattern for ``continuation_residual``."""

    n_unknown = 2 * params.n_inner + 2 * params.n_outer + 2
    n_rows = 2 * params.n_inner + 2 * params.n_outer + 4
    pattern = lil_matrix((n_rows, n_unknown), dtype=int)
    ni = params.n_inner
    no = params.n_outer
    iu = 0
    iT = ni
    ou = 2 * ni
    oT = 2 * ni + no
    logR_col = 2 * ni + 2 * no
    lambda_col = logR_col + 1
    row = 0

    for col in (iu, iu + 1, iT, iT + 1, logR_col, lambda_col):
        pattern[row : row + 2, col] = 1
    row += 2

    for idx in range(1, ni - 1):
        for col in (iu + idx, iu + idx + 1, iT + idx, iT + idx + 1, logR_col, lambda_col):
            pattern[row : row + 2, col] = 1
        row += 2

    for idx in range(no - 1):
        for col in (ou + idx, ou + idx + 1, oT + idx, oT + idx + 1, lambda_col):
            pattern[row : row + 2, col] = 1
        row += 2

    for col in (iu + ni - 1, iT + ni - 1, ou, oT):
        pattern[row : row + 2, col] = 1
    row += 2

    for col in (ou + no - 2, ou + no - 1, oT + no - 2, oT + no - 1, lambda_col):
        pattern[row : row + 2, col] = 1
    row += 2

    for col in (iu, iT, logR_col, lambda_col):
        pattern[row : row + 4, col] = 1
    return pattern.tocsr()


def solve_continuation(seed: np.ndarray, params, lambda_target: float, max_nfev: int):
    lower, upper = state_bounds_two_domain(params)  # type: ignore[arg-type]
    x0 = np.clip(np.asarray(seed, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: continuation_residual(trial, params, lambda_target),
        x0,
        jac_sparsity=continuation_sparsity(params),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=max_nfev,
    )


def interp_profile_at(logR_target: float, logR: np.ndarray, logu: np.ndarray, logT: np.ndarray) -> np.ndarray:
    if not float(logR[0]) <= logR_target <= float(logR[-1]):
        raise ValueError("fold probe radius is outside the inner profile")
    return np.array(
        [
            np.interp(logR_target, logR, logu),
            np.interp(logR_target, logR, logT),
        ],
        dtype=float,
    )


def orient_tangent(logR: float, y: np.ndarray, lambda0: float, params, reference: np.ndarray | None = None) -> np.ndarray:
    tangent = phase_space_null_tangent(logR, y, lambda0, params.physics, previous=reference).tangent
    if reference is None and tangent[0] < 0.0:
        tangent = -tangent
    return tangent


def heun_step(z: np.ndarray, lambda0: float, params, ds: float, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = orient_tangent(float(z[0]), z[1:], lambda0, params, previous)
    z_pred = z + ds * p0
    p1 = orient_tangent(float(z_pred[0]), z_pred[1:], lambda0, params, p0)
    z_new = z + 0.5 * ds * (p0 + p1)
    p_new = orient_tangent(float(z_new[0]), z_new[1:], lambda0, params, p1)
    return z_new, p_new


def fold_probe(x: np.ndarray, params) -> dict[str, object]:
    """Trace the local desingularized curve from the global profile near R=6 rg."""

    logu_i, logT_i, _logu_o, _logT_o, logR_son, lambda0 = unpack_buffer(x, params)
    logR_i = buffer_inner_grid(logR_son, params)
    logR_probe = np.log(FOLD_PROBE_R_RG * params.r_g)
    try:
        y_probe = interp_profile_at(logR_probe, logR_i, logu_i, logT_i)
        z = np.array([logR_probe, y_probe[0], y_probe[1]], dtype=float)
        p = orient_tangent(float(z[0]), z[1:], lambda0, params)
    except Exception as exc:
        return {"fold_status": "probe_failed", "fold_message": str(exc)}

    R_values = [float(np.exp(z[0]) / params.r_g)]
    px_values = [float(p[0])]
    smin_B_values = []
    H_R_values = []
    omega_values = []
    s_value = 0.0
    px_sign = float(np.sign(p[0]))
    for _idx in range(FOLD_MAX_STEPS):
        if s_value >= FOLD_S_MAX:
            break
        z_prev = z.copy()
        p_prev = p.copy()
        s_prev = s_value
        try:
            z, p = heun_step(z, lambda0, params, FOLD_DS, p)
        except Exception as exc:
            return {
                "fold_status": "trace_failed",
                "fold_message": str(exc),
                "R_trace_end_rg": float(R_values[-1]),
                "R_trace_max_rg": float(np.max(R_values)),
                "p_x_min": float(np.min(px_values)),
                "p_x_sign_changes": int(np.sum(np.sign(px_values[1:]) != np.sign(px_values[:-1]))) if len(px_values) > 1 else 0,
            }
        s_value += FOLD_DS
        R_rg = float(np.exp(z[0]) / params.r_g)
        R_values.append(R_rg)
        px_values.append(float(p[0]))
        try:
            tangent = phase_space_null_tangent(float(z[0]), z[1:], lambda0, params.physics, previous=p)
            state = algebraic_state(float(z[0]), float(z[1]), float(z[2]), lambda0, params.physics)
            smin_B_values.append(float(tangent.smin_over_smax_B))
            H_R_values.append(float(state.H_over_R))
            omega_values.append(float(state.Omega / state.Omega_K))
        except Exception:
            pass
        next_sign = float(np.sign(p[0]))
        if px_sign > 0.0 and next_sign < 0.0:
            try:
                s_fold, z_fold, p_fold, bracket = refine_bracket((s_prev, z_prev, p_prev), (s_value, z.copy(), p.copy()), lambda0, params)
                tangent = phase_space_null_tangent(float(z_fold[0]), z_fold[1:], lambda0, params.physics, previous=p_fold)
                dp_ds = phase_space_tangent_derivative(float(z_fold[0]), z_fold[1:], lambda0, params.physics, tangent.tangent, eps=TANGENT_EPS)
                state = algebraic_state(float(z_fold[0]), float(z_fold[1]), float(z_fold[2]), lambda0, params.physics)
                minors = B_rank_minors(float(z_fold[0]), z_fold[1:], lambda0, params.physics)
                return {
                    "fold_status": "fold_found",
                    "fold_message": "p_x sign change bracketed",
                    "R_fold_rg": float(np.exp(z_fold[0]) / params.r_g),
                    "s_fold": float(s_fold),
                    "p_x_fold": float(tangent.tangent[0]),
                    "dpx_ds_fold": float(dp_ds[0]),
                    "smin_over_smax_B_fold": float(tangent.smin_over_smax_B),
                    "B_minors_norm_fold": float(np.linalg.norm(minors)),
                    "H_R_fold": float(state.H_over_R),
                    "Omega_over_K_fold": float(state.Omega / state.Omega_K),
                    "R_trace_end_rg": R_rg,
                    "R_trace_max_rg": float(np.max(R_values)),
                    "p_x_min": float(np.min(px_values)),
                    "p_x_sign_changes": 1,
                    "min_smin_over_smax_B_track": float(np.nanmin(smin_B_values)) if smin_B_values else np.nan,
                    "max_H_R_track": float(np.nanmax(H_R_values)) if H_R_values else np.nan,
                    "max_abs_Omega_over_K_track": float(np.nanmax(np.abs(omega_values))) if omega_values else np.nan,
                    "bracket_width_s": float(bracket),
                }
            except Exception as exc:
                return {"fold_status": "fold_refine_failed", "fold_message": str(exc)}
        if R_rg >= FOLD_TARGET_R_RG:
            return {
                "fold_status": "target_reached",
                "fold_message": f"reached R >= {FOLD_TARGET_R_RG:g} rg before a fold",
                "R_fold_rg": np.nan,
                "R_trace_end_rg": R_rg,
                "R_trace_max_rg": float(np.max(R_values)),
                "p_x_min": float(np.min(px_values)),
                "p_x_sign_changes": 0,
                "min_smin_over_smax_B_track": float(np.nanmin(smin_B_values)) if smin_B_values else np.nan,
                "max_H_R_track": float(np.nanmax(H_R_values)) if H_R_values else np.nan,
                "max_abs_Omega_over_K_track": float(np.nanmax(np.abs(omega_values))) if omega_values else np.nan,
            }
        if next_sign != 0.0:
            px_sign = next_sign
    return {
        "fold_status": "no_fold_within_smax",
        "fold_message": "completed fold probe arclength limit",
        "R_fold_rg": np.nan,
        "R_trace_end_rg": float(R_values[-1]),
        "R_trace_max_rg": float(np.max(R_values)),
        "p_x_min": float(np.min(px_values)),
        "p_x_sign_changes": int(np.sum(np.sign(px_values[1:]) != np.sign(px_values[:-1]))) if len(px_values) > 1 else 0,
        "min_smin_over_smax_B_track": float(np.nanmin(smin_B_values)) if smin_B_values else np.nan,
        "max_H_R_track": float(np.nanmax(H_R_values)) if H_R_values else np.nan,
        "max_abs_Omega_over_K_track": float(np.nanmax(np.abs(omega_values))) if omega_values else np.nan,
    }


def audit_continuation(label: str, x: np.ndarray, params, lambda_target: float, result=None) -> dict[str, object]:
    row = dynamic_audit(label, x, params, result)
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_buffer(x, params)
    logR_i = buffer_inner_grid(logR_son, params)
    logR_o = outer_grid(params)  # type: ignore[arg-type]
    far = far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params)  # type: ignore[arg-type]
    residual = continuation_residual(x, params, lambda_target)
    blocks = {
        "patch": abs(float(row["patch"])),
        "regular_R": abs(float(row["regular_R"])),
        "regular_E": abs(float(row["regular_E"])),
        "outer_R": abs(float(row["outer_R"])),
        "outer_E": abs(float(row["outer_E"])),
        "interface": abs(float(row["interface"])),
        "far_omega": abs(float(far[0])),
        "D": abs(float(row["D"])),
        "C1": abs(float(row["C1"])),
        "C2": abs(float(row["C2"])),
        "K": abs(float(row["K"])),
    }
    physical_except_theta = max(blocks.values())
    row.update(
        {
            "lambda_target": float(lambda_target),
            "delta_lambda_target": float(lambda_target - params.lambda_ref) if hasattr(params, "lambda_ref") else np.nan,
            "lambda_error": float(lambda0 - lambda_target),
            "theta_out": float(far[1]),
            "selected_max": float(np.max(np.abs(residual))),
            "physical_except_theta": float(physical_except_theta),
            "dominant_except_theta": max(blocks, key=blocks.get),
            "passes_except_theta": bool(physical_except_theta <= SCIENCE_LIMIT),
            "passes_physical_theta": bool(physical_except_theta <= SCIENCE_LIMIT and abs(float(far[1])) <= SCIENCE_LIMIT),
        }
    )
    combined_logR = np.concatenate([logR_i, logR_o[1:]])
    combined_logu = np.concatenate([logu_i, logu_o[1:]])
    combined_logT = np.concatenate([logT_i, logT_o[1:]])
    row["int_adv"] = integrated_advective_fraction(combined_logu, combined_logT, combined_logR, lambda0, params)  # type: ignore[arg-type]
    row.update(fold_probe(x, params))
    row["x"] = np.asarray(x, dtype=float)
    return row


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = str(row["label"]).replace(".", "p").replace("-", "m")
    payload = {key: value for key, value in row.items() if key != "x"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}_0p90277664.npz",
        x=np.asarray(row["x"], dtype=float),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Transonic Global-Family Continuation",
        "",
        "Generated by `scripts/run_transonic_global_family_continuation.py`.",
        "",
        "Track-A diagnostic: the dynamic two-domain BVP is re-solved while replacing the far thermal row with a lambda-target row. The omitted physical far-energy residual is reported as `theta_out`. A viable physical branch requires `theta_out -> 0`, small residuals in all other blocks, and no local projection fold before the target radius.",
        "",
        f"Config: `source={SOURCE_CHECKPOINT}`, `lambda_scale={LAMBDA_SCALE:g}`, `max_nfev={MAX_NFEV}`, `fold_probe_R={FOLD_PROBE_R_RG:g} rg`, `fold_target_R={FOLD_TARGET_R_RG:g} rg`.",
        "",
        "| label | lambda target | lambda0 | lambda error | theta_out | selected | phys excl theta | pass excl | pass theta | dominant excl | fold status | R fold/rg | R trace max/rg | p_x min | min sB | max H/R track | Rson/rg | int adv | far omega | patch | regular R | regular E | outer R | outer E | D | C1 | C2 | K | nfev | success | message |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {lambda_target} | {lambda0} | {lambda_error} | {theta_out} | {selected_max} | "
            "{physical_except_theta} | {passes_except_theta} | {passes_physical_theta} | {dominant_except_theta} | "
            "{fold_status} | {R_fold_rg} | {R_trace_max_rg} | {p_x_min} | {min_smin_over_smax_B_track} | "
            "{max_H_R_track} | {Rson_rg} | {int_adv} | {far_omega} | {patch} | {regular_R} | {regular_E} | "
            "{outer_R} | {outer_E} | {D} | {C1} | {C2} | {K} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                lambda_target=fmt(float(row["lambda_target"])),
                lambda0=fmt(float(row["lambda0"])),
                lambda_error=fmt(float(row["lambda_error"])),
                theta_out=fmt(float(row["theta_out"])),
                selected_max=fmt(float(row["selected_max"])),
                physical_except_theta=fmt(float(row["physical_except_theta"])),
                passes_except_theta="yes" if row["passes_except_theta"] else "no",
                passes_physical_theta="yes" if row["passes_physical_theta"] else "no",
                dominant_except_theta=row["dominant_except_theta"],
                fold_status=row.get("fold_status", "n/a"),
                R_fold_rg=fmt(float(row.get("R_fold_rg", np.nan))),
                R_trace_max_rg=fmt(float(row.get("R_trace_max_rg", np.nan))),
                p_x_min=fmt(float(row.get("p_x_min", np.nan))),
                min_smin_over_smax_B_track=fmt(float(row.get("min_smin_over_smax_B_track", np.nan))),
                max_H_R_track=fmt(float(row.get("max_H_R_track", np.nan))),
                Rson_rg=fmt(float(row["Rson_rg"])),
                int_adv=fmt(float(row["int_adv"])),
                far_omega=fmt(float(row["far_omega"])),
                patch=fmt(float(row["patch"])),
                regular_R=fmt(float(row["regular_R"])),
                regular_E=fmt(float(row["regular_E"])),
                outer_R=fmt(float(row["outer_R"])),
                outer_E=fmt(float(row["outer_E"])),
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
    JSON_OUTPUT.write_text(json.dumps([{key: json_safe(value) for key, value in row.items() if key != "x"} for row in rows], indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_fallback_figure(rows, str(exc))
        return
    if not rows:
        return
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda row: float(row["lambda_target"]))
    lambdas = np.asarray([float(row["lambda0"]) for row in rows_sorted], dtype=float)
    theta = np.asarray([float(row["theta_out"]) for row in rows_sorted], dtype=float)
    phys = np.asarray([float(row["physical_except_theta"]) for row in rows_sorted], dtype=float)
    rfold = np.asarray([float(row.get("R_fold_rg", np.nan)) for row in rows_sorted], dtype=float)
    rmax = np.asarray([float(row.get("R_trace_max_rg", np.nan)) for row in rows_sorted], dtype=float)
    ok = np.asarray([bool(row["passes_except_theta"]) for row in rows_sorted], dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.5), sharex=True)
    axes[0].axhline(0.0, color="0.25", lw=1.0)
    axes[0].plot(lambdas, theta, color="tab:blue", lw=1.4)
    axes[0].scatter(lambdas[ok], theta[ok], color="tab:green", s=28, label="nonthermal residual ok")
    axes[0].scatter(lambdas[~ok], theta[~ok], color="tab:red", s=22, label="residual high")
    axes[0].set_ylabel(r"$\theta_{\rm out}$")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(lambdas, phys, color="tab:purple", marker="o", ms=3)
    axes[1].axhline(SCIENCE_LIMIT, color="0.35", lw=1.0, ls="--")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("physical excl. theta")

    axes[2].plot(lambdas, rmax, color="tab:orange", marker="o", ms=3, label="trace max")
    axes[2].plot(lambdas, rfold, color="tab:brown", marker="x", ms=4, label="fold")
    axes[2].axhline(FOLD_TARGET_R_RG, color="0.35", lw=1.0, ls="--")
    axes[2].set_ylabel(r"$R/r_g$")
    axes[2].set_xlabel(r"$\lambda_0$")
    axes[2].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)


def write_fallback_figure(rows: list[dict[str, object]], reason: str) -> None:
    """Write a minimal PNG diagnostic when matplotlib is unavailable."""

    if not rows:
        return
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: matplotlib unavailable ({reason}); PIL unavailable ({exc})", flush=True)
        return
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda row: float(row["lambda_target"]))
    x_values = np.asarray([float(row["lambda0"]) for row in rows_sorted], dtype=float)
    theta = np.asarray([float(row["theta_out"]) for row in rows_sorted], dtype=float)
    phys = np.asarray([max(float(row["physical_except_theta"]), 1.0e-16) for row in rows_sorted], dtype=float)
    rfold = np.asarray([float(row.get("R_fold_rg", np.nan)) for row in rows_sorted], dtype=float)
    rmax = np.asarray([float(row.get("R_trace_max_rg", np.nan)) for row in rows_sorted], dtype=float)

    width, height = 1200, 1350
    margin_l, margin_r = 135, 45
    panel_h = 330
    panel_gap = 85
    top = 85
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def finite_limits(values: np.ndarray, *, log: bool = False, include_zero: bool = False) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (0.0, 1.0)
        if log:
            finite = np.log10(np.maximum(finite, 1.0e-16))
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
        if include_zero:
            ymin = min(ymin, 0.0)
            ymax = max(ymax, 0.0)
        if ymax <= ymin:
            pad = max(abs(ymax), 1.0) * 0.05
            return ymin - pad, ymax + pad
        pad = 0.08 * (ymax - ymin)
        return ymin - pad, ymax + pad

    xmin, xmax = float(np.min(x_values)), float(np.max(x_values))
    if xmax <= xmin:
        xmin -= 5.0e-4
        xmax += 5.0e-4

    def project(x: float, y: float, y_limits: tuple[float, float], panel_top: int) -> tuple[int, int]:
        y0, y1 = y_limits
        px = margin_l + int((x - xmin) / (xmax - xmin) * (width - margin_l - margin_r))
        py = panel_top + panel_h - int((y - y0) / (y1 - y0) * panel_h)
        return px, py

    def draw_panel(
        panel_idx: int,
        title: str,
        ylabel: str,
        series: list[tuple[np.ndarray, str]],
        y_limits: tuple[float, float],
        *,
        log: bool = False,
        hlines: tuple[float, ...] = (),
    ) -> None:
        panel_top = top + panel_idx * (panel_h + panel_gap)
        x0, x1 = margin_l, width - margin_r
        y0, y1 = panel_top, panel_top + panel_h
        draw.rectangle((x0, y0, x1, y1), outline=(80, 80, 80), width=2)
        draw.text((margin_l, panel_top - 35), title, fill=(20, 20, 20), font=font)
        draw.text((10, panel_top + panel_h // 2 - 10), ylabel, fill=(20, 20, 20), font=font)
        for value in hlines:
            value_plot = np.log10(max(value, 1.0e-16)) if log else value
            if y_limits[0] <= value_plot <= y_limits[1]:
                x_a, y_a = project(xmin, value_plot, y_limits, panel_top)
                x_b, y_b = project(xmax, value_plot, y_limits, panel_top)
                draw.line((x_a, y_a, x_b, y_b), fill=(120, 120, 120), width=1)
        colors = {
            "theta": (31, 119, 180),
            "phys": (117, 44, 143),
            "rmax": (255, 127, 14),
            "rfold": (140, 86, 75),
        }
        for values, name in series:
            plot_values = np.log10(np.maximum(values, 1.0e-16)) if log else values
            points = [
                project(float(x), float(y), y_limits, panel_top)
                for x, y in zip(x_values, plot_values)
                if np.isfinite(x) and np.isfinite(y)
            ]
            if len(points) >= 2:
                draw.line(points, fill=colors.get(name, (0, 0, 0)), width=3)
            for point in points:
                draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=colors.get(name, (0, 0, 0)))
            if points:
                draw.text((points[-1][0] + 8, points[-1][1] - 7), name, fill=colors.get(name, (0, 0, 0)), font=font)
        for tick in (xmin, xmax):
            px, _ = project(tick, y_limits[0], y_limits, panel_top)
            draw.text((px - 35, y1 + 8), f"{tick:.6f}", fill=(20, 20, 20), font=font)
        draw.text((x0 + 6, y0 + 6), f"{y_limits[1]:.2e}", fill=(80, 80, 80), font=font)
        draw.text((x0 + 6, y1 - 18), f"{y_limits[0]:.2e}", fill=(80, 80, 80), font=font)

    theta_limits = finite_limits(theta, include_zero=True)
    phys_limits = finite_limits(phys, log=True)
    radius_values = np.concatenate([rfold[np.isfinite(rfold)], rmax[np.isfinite(rmax)]])
    radius_limits = finite_limits(radius_values if radius_values.size else np.array([FOLD_TARGET_R_RG]), include_zero=False)

    draw_panel(0, "Outer thermal homotopy", "theta_out", [(theta, "theta")], theta_limits, hlines=(0.0,))
    draw_panel(1, "Residual excluding theta", "log10 residual", [(phys, "phys")], phys_limits, log=True, hlines=(SCIENCE_LIMIT,))
    draw_panel(2, "Local phase-space fold probe", "R / rg", [(rmax, "rmax"), (rfold, "rfold")], radius_limits, hlines=(FOLD_TARGET_R_RG,))
    draw.text((margin_l, height - 55), "lambda0", fill=(20, 20, 20), font=font)
    draw.text((margin_l, 20), f"Fallback renderer: matplotlib unavailable ({reason})", fill=(100, 100, 100), font=font)
    image.save(FIGURE_OUTPUT)


def target_sequence(lambda_ref: float) -> list[tuple[str, list[float]]]:
    positive = [lambda_ref + delta for delta in DELTA_LAMBDAS if delta >= 0.0]
    negative = [lambda_ref + delta for delta in DELTA_LAMBDAS if delta < 0.0]
    positive = sorted(dict.fromkeys(positive))
    negative = sorted(dict.fromkeys(negative), reverse=True)
    return [("plus", positive), ("minus", negative)]


def main() -> None:
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(f"missing source checkpoint: {SOURCE_CHECKPOINT}")
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_x, source_meta = load_row(SOURCE_CHECKPOINT)
    ratio = float(source_meta["ratio"])
    params = make_buffer_params(
        fiducial,
        ratio,
        mdot_edd,
        int(source_meta["n_regular"]),
        int(source_meta["n_outer"]),
        float(source_meta["R_far_rg"]),
        float(source_meta["delta_s"]),
    )
    # Attach the reference lambda as a lightweight script-side value for audits.
    object.__setattr__(params, "lambda_ref", float(source_meta["lambda0"]))
    lambda_ref = float(source_meta["lambda0"])
    rows: list[dict[str, object]] = []
    write_table(rows)

    for direction, lambdas in target_sequence(lambda_ref):
        if not lambdas:
            continue
        current_x = np.asarray(source_x, dtype=float).copy()
        for lambda_target in lambdas:
            label = f"{direction}_dlambda{lambda_target - lambda_ref:+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")
            print(f"{label}: solving lambda_target={lambda_target:.9f}", flush=True)
            release = solve_continuation(current_x, params, lambda_target, MAX_NFEV)
            polish = solve_continuation(release.x, params, lambda_target, POLISH_NFEV)
            row = audit_continuation(label, polish.x, params, lambda_target, polish)
            rows.append(row)
            save_checkpoint(row)
            write_table(rows)
            write_figure(rows)
            print(
                f"{label}: theta={row['theta_out']:.3e} phys_ex_theta={row['physical_except_theta']:.3e} "
                f"fold={row.get('fold_status')} Rmax={row.get('R_trace_max_rg', np.nan):.4g} "
                f"lambda={row['lambda0']:.9f}",
                flush=True,
            )
            current_x = np.asarray(polish.x, dtype=float)

    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {JSON_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
