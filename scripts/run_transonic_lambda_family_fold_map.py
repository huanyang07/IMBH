"""Map how the desingularized radial fold responds to lambda0 perturbations."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d.transonic_local import (
    B_rank_minors,
    algebraic_state,
    phase_space_null_tangent,
    phase_space_tangent_derivative,
    sonic_derivative_branches,
    sonic_diagnostics,
)

from run_transonic_branch_connection_scan import solve_fixed_dlambda
from run_transonic_desingularized_barrier_flow import incoming_state_at, json_safe
from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_sonic_reverse_fit import FLOW_EPS0, load_context, solve_fit
from run_transonic_two_domain_sonic_flowmap import LHOPITAL_EPS


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_LAMBDA_FOLD_MAP_TABLE",
    "outputs/tables/transonic_lambda_family_fold_map.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_name(TABLE_OUTPUT.stem + ".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_LAMBDA_FOLD_MAP_FIGURE",
    "outputs/figures/transonic_lambda_family_fold_map.png",
)

DELTAS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_LAMBDA_FOLD_DELTAS", "-0.004,-0.003,-0.002,-0.001,0,0.001,0.002,0.003,0.004")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
LOCAL_START_R_RG = float(os.environ.get("IMBH_LAMBDA_FOLD_LOCAL_START_R_RG", "6.0"))
DS = float(os.environ.get("IMBH_LAMBDA_FOLD_DS", "1e-3"))
S_MAX = float(os.environ.get("IMBH_LAMBDA_FOLD_S_MAX", "12.0"))
TARGET_R_RG = float(os.environ.get("IMBH_LAMBDA_FOLD_TARGET_R_RG", "8.0"))
MAX_STEPS = int(os.environ.get("IMBH_LAMBDA_FOLD_MAX_STEPS", "12000"))
BISECTION_STEPS = int(os.environ.get("IMBH_LAMBDA_FOLD_BISECTION_STEPS", "36"))
TANGENT_EPS = float(os.environ.get("IMBH_LAMBDA_FOLD_TANGENT_EPS", "2e-6"))
CRITK_ACCEPT = float(os.environ.get("IMBH_LAMBDA_FOLD_CRITK_ACCEPT", "1e-6"))
BRANCH_INDEX = int(os.environ.get("IMBH_LAMBDA_FOLD_BRANCH_INDEX", "1"))
BRANCH_HALF_WIDTH = float(os.environ.get("IMBH_LAMBDA_FOLD_BRANCH_HALF_WIDTH", "1000"))
BRANCH_POINTS = int(os.environ.get("IMBH_LAMBDA_FOLD_BRANCH_POINTS", "2001"))


def fixed(value: object, digits: int) -> str:
    number = float(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "nan"


def sci(value: object, digits: int = 3) -> str:
    number = float(value)
    return f"{number:.{digits}e}" if np.isfinite(number) else "nan"


def tangent_at(z: np.ndarray, lambda0: float, params, previous=None):
    return phase_space_null_tangent(float(z[0]), z[1:], lambda0, params.physics, previous=previous).tangent


def heun_step(z: np.ndarray, lambda0: float, params, ds: float, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = tangent_at(z, lambda0, params, previous=previous)
    z_pred = z + ds * p0
    p1 = tangent_at(z_pred, lambda0, params, previous=p0)
    z_new = z + 0.5 * ds * (p0 + p1)
    p_new = tangent_at(z_new, lambda0, params, previous=p1)
    return z_new, p_new


def fold_diagnostics(
    row: dict[str, object],
    s_fold: float,
    z_fold: np.ndarray,
    p_fold: np.ndarray,
    lambda0: float,
    params,
    bracket_width_s: float,
) -> dict[str, object]:
    tangent_diag = phase_space_null_tangent(float(z_fold[0]), z_fold[1:], lambda0, params.physics, previous=p_fold)
    p = tangent_diag.tangent
    dp_ds = phase_space_tangent_derivative(
        float(z_fold[0]),
        z_fold[1:],
        lambda0,
        params.physics,
        p,
        eps=TANGENT_EPS,
    )
    sonic = sonic_diagnostics(float(z_fold[0]), z_fold[1:], lambda0, params.physics)
    state = algebraic_state(float(z_fold[0]), float(z_fold[1]), float(z_fold[2]), lambda0, params.physics)
    minors = B_rank_minors(float(z_fold[0]), z_fold[1:], lambda0, params.physics)
    row.update(
        {
            "status": "fold_found",
            "s_fold": float(s_fold),
            "R_fold_rg": float(np.exp(z_fold[0]) / params.r_g),
            "logR_fold": float(z_fold[0]),
            "logu_fold": float(z_fold[1]),
            "logT_fold": float(z_fold[2]),
            "p_x_fold": float(p[0]),
            "p_u_fold": float(p[1]),
            "p_T_fold": float(p[2]),
            "dpx_ds_fold": float(dp_ds[0]),
            "smin_over_smax_A_fold": float(tangent_diag.smin_over_smax_A),
            "smin_over_smax_B_fold": float(tangent_diag.smin_over_smax_B),
            "condB_fold": float(1.0 / (tangent_diag.smin_over_smax_B + 1.0e-300)),
            "Bp_max_fold": float(np.max(np.abs(tangent_diag.residual))),
            "B_minors_norm_fold": float(np.linalg.norm(minors)),
            "D_fold": float(sonic.D),
            "C1_fold": float(sonic.C1),
            "C2_fold": float(sonic.C2),
            "K_fold": float(sonic.compatibility),
            "H_R_fold": float(state.H_over_R),
            "Omega_over_K_fold": float(state.Omega / state.Omega_K),
            "bracket_width_s": float(bracket_width_s),
        }
    )
    return row


def refine_bracket(
    left: tuple[float, np.ndarray, np.ndarray],
    right: tuple[float, np.ndarray, np.ndarray],
    lambda0: float,
    params,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    s_left, z_left, p_left = left
    s_right, z_right, p_right = right
    for _idx in range(BISECTION_STEPS):
        ds_mid = 0.5 * (s_right - s_left)
        z_mid, p_mid = heun_step(z_left, lambda0, params, ds_mid, p_left)
        s_mid = s_left + ds_mid
        if float(p_mid[0]) > 0.0:
            s_left, z_left, p_left = s_mid, z_mid, p_mid
        else:
            s_right, z_right, p_right = s_mid, z_mid, p_mid
    z_fold = 0.5 * (z_left + z_right)
    p_fold = tangent_at(z_fold, lambda0, params, previous=p_left)
    return 0.5 * (s_left + s_right), z_fold, p_fold, s_right - s_left


def locate_fold_from_seed(
    z0: np.ndarray,
    p0: np.ndarray,
    lambda0: float,
    params,
) -> tuple[dict[str, object], tuple[float, np.ndarray, np.ndarray] | None]:
    z = np.asarray(z0, dtype=float).copy()
    p = tangent_at(z, lambda0, params, previous=p0)
    s_value = 0.0
    px_sign = float(np.sign(p[0]))
    R_values = [float(np.exp(z[0]) / params.r_g)]
    smin_B_values = []
    H_R_values = []
    omega_values = []
    last_message = "completed s_max"
    for _step in range(MAX_STEPS):
        if s_value >= S_MAX:
            break
        z_prev = z.copy()
        p_prev = p.copy()
        s_prev = s_value
        try:
            z, p = heun_step(z, lambda0, params, DS, p)
        except Exception as exc:
            last_message = str(exc)
            break
        s_value += DS
        R_rg = float(np.exp(z[0]) / params.r_g)
        R_values.append(R_rg)
        try:
            tangent_diag = phase_space_null_tangent(float(z[0]), z[1:], lambda0, params.physics, previous=p)
            state = algebraic_state(float(z[0]), float(z[1]), float(z[2]), lambda0, params.physics)
            smin_B_values.append(float(tangent_diag.smin_over_smax_B))
            H_R_values.append(float(state.H_over_R))
            omega_values.append(float(state.Omega / state.Omega_K))
        except Exception:
            pass
        next_sign = float(np.sign(p[0]))
        if px_sign > 0.0 and next_sign < 0.0:
            s_fold, z_fold, p_fold, bracket = refine_bracket((s_prev, z_prev, p_prev), (s_value, z.copy(), p.copy()), lambda0, params)
            summary = {
                "s_end": float(s_value),
                "R_end_rg": R_rg,
                "R_max_rg": float(np.max(R_values)),
                "target_reached": False,
                "message": "fold bracketed",
                "min_smin_over_smax_B_track": float(np.nanmin(smin_B_values)) if smin_B_values else np.nan,
                "max_H_R_track": float(np.nanmax(H_R_values)) if H_R_values else np.nan,
                "max_abs_Omega_over_K_track": float(np.nanmax(np.abs(omega_values))) if omega_values else np.nan,
            }
            return summary, (s_fold, z_fold, p_fold, bracket)
        if next_sign != 0.0:
            px_sign = next_sign
        if R_rg >= TARGET_R_RG:
            last_message = f"reached R >= {TARGET_R_RG:g} rg"
            return (
                {
                    "status": "target_reached_without_fold",
                    "s_end": float(s_value),
                    "R_end_rg": R_rg,
                    "R_max_rg": float(np.max(R_values)),
                    "target_reached": True,
                    "message": last_message,
                    "final_p_x": float(p[0]),
                    "min_smin_over_smax_B_track": float(np.nanmin(smin_B_values)) if smin_B_values else np.nan,
                    "max_H_R_track": float(np.nanmax(H_R_values)) if H_R_values else np.nan,
                    "max_abs_Omega_over_K_track": float(np.nanmax(np.abs(omega_values))) if omega_values else np.nan,
                },
                None,
            )
    return (
        {
            "status": "no_fold_within_smax",
            "s_end": float(s_value),
            "R_end_rg": float(R_values[-1]),
            "R_max_rg": float(np.max(R_values)),
            "target_reached": False,
            "message": last_message,
            "final_p_x": float(p[0]),
            "min_smin_over_smax_B_track": float(np.nanmin(smin_B_values)) if smin_B_values else np.nan,
            "max_H_R_track": float(np.nanmax(H_R_values)) if H_R_values else np.nan,
            "max_abs_Omega_over_K_track": float(np.nanmax(np.abs(omega_values))) if omega_values else np.nan,
        },
        None,
    )


def reference_solution(ctx):
    result, info = solve_fit(ctx, "fixed_buffer", None)
    return result, info, float(info["lambda0"]), float(info["lambda0"] - ctx.lambda0_old)


def strict_critical_seed_row(ctx, lambda_ref: float, dlambda_ref: float, delta_lambda: float) -> dict[str, object]:
    total_dlambda = dlambda_ref + delta_lambda
    case = solve_fixed_dlambda(ctx, total_dlambda)
    row: dict[str, object] = {
        "mode": "strict_fixed_buffer_critical",
        "delta_lambda_ref": float(delta_lambda),
        "lambda0": float(case.lambda0),
        "critK_seed": float(case.critK),
        "critical_success": bool(case.success),
        "critical_message": case.message,
        "Rson_rg": float(np.exp(case.logR_son) / ctx.params.r_g),
        "D_seed": float(case.D),
        "C1_seed": float(case.C1),
        "C2_seed": float(case.C2),
        "K_seed": float(case.K),
    }
    if (not case.success) or (not np.isfinite(case.critK)) or case.critK > CRITK_ACCEPT or not np.all(np.isfinite(case.y_s)):
        row.update({"status": "critical_seed_rejected", "message": f"critK>{CRITK_ACCEPT:g} or failed"})
        return row
    try:
        branches = sonic_derivative_branches(
            case.logR_son,
            case.y_s,
            case.lambda0,
            ctx.params.physics,
            eps=LHOPITAL_EPS,
            form="scaled",
            half_width=BRANCH_HALF_WIDTH,
            scan_points=BRANCH_POINTS,
        )
        if BRANCH_INDEX >= len(branches):
            row.update({"status": "missing_sonic_branch", "message": f"found {len(branches)} branches"})
            return row
        branch = branches[BRANCH_INDEX]
        z0 = np.array([case.logR_son + FLOW_EPS0, *(case.y_s + FLOW_EPS0 * branch.gradient)], dtype=float)
        p0 = np.array([1.0, float(branch.gradient[0]), float(branch.gradient[1])], dtype=float)
        flow_summary, fold = locate_fold_from_seed(z0, p0, case.lambda0, ctx.params)
        row.update(flow_summary)
        if fold is not None:
            s_fold, z_fold, p_fold, bracket = fold
            row = fold_diagnostics(row, s_fold, z_fold, p_fold, case.lambda0, ctx.params, bracket)
        return row
    except Exception as exc:
        row.update({"status": "strict_flow_failed", "message": str(exc)})
        return row


def local_R6_seed_row(ctx, lambda_ref: float, delta_lambda: float) -> dict[str, object]:
    logR0, y0, _lambda0 = incoming_state_at(LOCAL_START_R_RG)
    lambda0 = lambda_ref + delta_lambda
    z0 = np.array([logR0, float(y0[0]), float(y0[1])], dtype=float)
    p0 = phase_space_null_tangent(float(z0[0]), z0[1:], lambda0, ctx.params.physics, prefer_positive_x=True).tangent
    row: dict[str, object] = {
        "mode": "local_R6_lambda_sensitivity",
        "delta_lambda_ref": float(delta_lambda),
        "lambda0": float(lambda0),
        "R_start_rg": float(LOCAL_START_R_RG),
        "critK_seed": np.nan,
        "critical_success": False,
        "critical_message": "not a critical seed; local sensitivity only",
    }
    flow_summary, fold = locate_fold_from_seed(z0, p0, lambda0, ctx.params)
    row.update(flow_summary)
    if fold is not None:
        s_fold, z_fold, p_fold, bracket = fold
        row = fold_diagnostics(row, s_fold, z_fold, p_fold, lambda0, ctx.params, bracket)
    return row


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lambda-Family Fold Map",
        "",
        "Generated by `scripts/run_transonic_lambda_family_fold_map.py`.",
        "",
        f"Config: `ds={DS:g}`, `s_max={S_MAX:g}`, `target_R={TARGET_R_RG:g} rg`, `critK_accept={CRITK_ACCEPT:g}`.",
        "",
        "| mode | delta lambda | lambda0 | status | Rson | critK seed | R fold | R max | R end | target | dp_x/ds | sB fold | H/R fold | Omega/K fold | final p_x | message |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {delta} | {lambda0} | {status} | {Rson} | {critK} | {Rfold} | {Rmax} | {Rend} | {target} | "
            "{dpx} | {sB} | {HR} | {omega} | {final_px} | {message} |".format(
                mode=row.get("mode", ""),
                delta=sci(row.get("delta_lambda_ref", np.nan)),
                lambda0=fixed(row.get("lambda0", np.nan), 9),
                status=row.get("status", ""),
                Rson=fixed(row.get("Rson_rg", np.nan), 6),
                critK=sci(row.get("critK_seed", np.nan)),
                Rfold=fixed(row.get("R_fold_rg", np.nan), 9),
                Rmax=fixed(row.get("R_max_rg", np.nan), 6),
                Rend=fixed(row.get("R_end_rg", np.nan), 6),
                target="yes" if row.get("target_reached", False) else "no",
                dpx=sci(row.get("dpx_ds_fold", np.nan)),
                sB=sci(row.get("smin_over_smax_B_fold", np.nan)),
                HR=sci(row.get("H_R_fold", np.nan)),
                omega=fixed(row.get("Omega_over_K_fold", np.nan), 6),
                final_px=sci(row.get("final_p_x", np.nan)),
                message=str(row.get("message", row.get("critical_message", ""))).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def write_json(rows: list[dict[str, object]]) -> None:
    JSON_OUTPUT.write_text(json.dumps([{key: json_safe(value) for key, value in row.items()} for row in rows], indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"figure skipped: {exc}", flush=True)
        return
    width, height = 1080, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 24), "Lambda-family fold map", fill=(20, 20, 20))
    panels = [
        ("fold/max R", "R_display", False),
        ("critK seed", "critK_seed", True),
        ("smin/smax B at fold", "smin_over_smax_B_fold", True),
    ]
    colors = {
        "strict_fixed_buffer_critical": (31, 119, 180),
        "local_R6_lambda_sensitivity": (214, 39, 40),
    }
    plot_l, plot_r = 100, width - 60
    panel_h = 130
    rows_for_plot = []
    for row in rows:
        row = dict(row)
        row["R_display"] = row.get("R_fold_rg", row.get("R_max_rg", np.nan))
        rows_for_plot.append(row)
    for pidx, (title, key, logy) in enumerate(panels):
        top = 70 + 165 * pidx
        bottom = top + panel_h
        draw.rectangle((plot_l, top, plot_r, bottom), outline=(60, 60, 60), width=1)
        xs = np.asarray([float(row["delta_lambda_ref"]) for row in rows_for_plot], dtype=float)
        values = np.asarray([float(row.get(key, np.nan)) for row in rows_for_plot], dtype=float)
        finite_values = values[np.isfinite(values) & ((values > 0.0) if logy else np.ones_like(values, dtype=bool))]
        if finite_values.size == 0:
            continue
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        ymin, ymax = float(np.min(finite_values)), float(np.max(finite_values))
        if logy:
            ymin = max(ymin * 0.7, 1.0e-18)
            ymax = max(ymax * 1.4, ymin * 10.0)
        else:
            pad = max(0.1 * (ymax - ymin), 1.0e-6)
            ymin -= pad
            ymax += pad

        def sx(value: float) -> float:
            return plot_l + (value - xmin) / (xmax - xmin + 1.0e-300) * (plot_r - plot_l)

        def sy(value: float) -> float:
            if logy:
                return bottom - (np.log10(value) - np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin) + 1.0e-300) * (bottom - top)
            return bottom - (value - ymin) / (ymax - ymin + 1.0e-300) * (bottom - top)

        draw.text((plot_l + 8, top + 8), title, fill=(20, 20, 20))
        for midx, mode in enumerate(colors):
            series = sorted([row for row in rows_for_plot if row["mode"] == mode and np.isfinite(float(row.get(key, np.nan)))], key=lambda row: float(row["delta_lambda_ref"]))
            points = []
            for row in series:
                value = float(row.get(key, np.nan))
                if np.isfinite(value) and ((value > 0.0) if logy else True):
                    points.append((sx(float(row["delta_lambda_ref"])), sy(value)))
            color = colors[mode]
            if len(points) >= 2:
                draw.line(points, fill=color, width=2)
            for point in points:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
            if pidx == 0:
                draw.text((plot_l + 12, top + 24 + 16 * midx), mode, fill=color)
    draw.text((plot_l, height - 26), "delta lambda relative to reference critical fit", fill=(20, 20, 20))
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    ctx = load_context()
    _result, _info, lambda_ref, dlambda_ref = reference_solution(ctx)
    rows: list[dict[str, object]] = []
    print(f"lambda_ref={lambda_ref:.12g} dlambda_ref={dlambda_ref:.6g}", flush=True)
    for delta in DELTAS:
        strict = strict_critical_seed_row(ctx, lambda_ref, dlambda_ref, delta)
        rows.append(strict)
        print(
            f"strict delta={delta:+.4g}: status={strict.get('status')} "
            f"critK={float(strict.get('critK_seed', np.nan)):.3e} "
            f"Rfold={float(strict.get('R_fold_rg', np.nan)):.6g}",
            flush=True,
        )
    for delta in DELTAS:
        local = local_R6_seed_row(ctx, lambda_ref, delta)
        rows.append(local)
        print(
            f"local delta={delta:+.4g}: status={local.get('status')} "
            f"Rfold={float(local.get('R_fold_rg', np.nan)):.6g} "
            f"Rmax={float(local.get('R_max_rg', np.nan)):.6g} target={local.get('target_reached')}",
            flush=True,
        )
    write_table(rows)
    write_json(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {JSON_OUTPUT}", flush=True)
    if FIGURE_OUTPUT.exists():
        print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
