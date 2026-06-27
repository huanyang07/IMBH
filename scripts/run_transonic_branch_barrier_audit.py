"""Localize the outward exact-branch ODE barrier near the sonic point."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    algebraic_state,
    local_ode_rhs,
    sonic_derivative_branches,
    sonic_diagnostics,
)

from run_transonic_branch_connection_scan import solve_fixed_dlambda
from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_sonic_reverse_fit import load_context, solve_fit
from run_transonic_sonic_reverse_fit import INITIAL_DLAMBDA
from run_transonic_two_domain_sonic_flowmap import LHOPITAL_EPS


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_BRANCH_BARRIER_TABLE",
    "outputs/tables/transonic_branch_barrier_audit.md",
)
POINTS_OUTPUT = TABLE_OUTPUT.with_name(TABLE_OUTPUT.stem + "_points.json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_BRANCH_BARRIER_FIGURE",
    "outputs/figures/transonic_branch_barrier_audit.png",
)

R_END_RG = float(os.environ.get("IMBH_BRANCH_BARRIER_R_END_RG", "6.5"))
FLOW_EPS0 = float(os.environ.get("IMBH_BRANCH_BARRIER_EPS0", "1e-6"))
FLOW_MAX_STEP = float(os.environ.get("IMBH_BRANCH_BARRIER_MAX_STEP", "2e-5"))
FLOW_RTOL = float(os.environ.get("IMBH_BRANCH_BARRIER_RTOL", "1e-9"))
FLOW_ATOL = float(os.environ.get("IMBH_BRANCH_BARRIER_ATOL", "1e-11"))
BRANCH_SEQUENCE = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_BRANCH_BARRIER_BRANCHES", "0,1").replace(":", ",").split(",")
    if piece.strip()
)
FIXED_DLAMBDA = float(os.environ.get("IMBH_BRANCH_BARRIER_FIXED_DLAMBDA", "-0.00153"))
BRANCH_HALF_WIDTH = float(os.environ.get("IMBH_BRANCH_BARRIER_BRANCH_HALF_WIDTH", "1000"))
BRANCH_POINTS = int(os.environ.get("IMBH_BRANCH_BARRIER_BRANCH_POINTS", "2001"))


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def critical_case_from_free_fit(ctx):
    from run_transonic_branch_connection_scan import critical_case_from_info

    result, info = solve_fit(ctx, "fixed_buffer", None)
    return critical_case_from_info("free_reverse_fit", "free_dlambda", INITIAL_DLAMBDA, result, info, ctx)


def heating_metrics(logR: float, y: np.ndarray, g: np.ndarray, lambda0: float, params) -> tuple[float, float, float]:
    q_visc, q_rad, q_adv, _energy = _heating_terms_from_gradient(logR, y, g, lambda0, params.physics)
    return (
        float(q_adv / (q_visc + 1.0e-300)),
        float(q_rad / (q_visc + 1.0e-300)),
        float(np.sign(q_visc)),
    )


def diagnostic_row(case, branch_index: int, branch, logR: float, y: np.ndarray, params) -> dict[str, object]:
    row: dict[str, object] = {
        "case": case.label,
        "mode": case.mode,
        "branch": int(branch_index),
        "a": float(branch.a),
        "R_rg": float(np.exp(logR) / params.r_g),
        "dx": float(logR - case.logR_son),
        "logR": float(logR),
        "logu": float(y[0]),
        "logT": float(y[1]),
        "lambda0": float(case.lambda0),
        "Rson_rg": float(np.exp(case.logR_son) / params.r_g),
    }
    try:
        sonic = sonic_diagnostics(logR, y, case.lambda0, params.physics)
        row.update(
            {
                "D": float(sonic.D),
                "C1": float(sonic.C1),
                "C2": float(sonic.C2),
                "K": float(sonic.compatibility),
                "N": float(sonic.N),
                "smin_over_smax": float(sonic.smin_over_smax),
                "condA": float(1.0 / (sonic.smin_over_smax + 1.0e-300)),
                "M_eff": float(sonic.M_eff),
            }
        )
    except Exception as exc:
        row.update(
            {
                "D": np.nan,
                "C1": np.nan,
                "C2": np.nan,
                "K": np.nan,
                "N": np.nan,
                "smin_over_smax": np.nan,
                "condA": np.nan,
                "M_eff": np.nan,
                "diagnostic_error": str(exc),
            }
        )
    try:
        g = local_ode_rhs(logR, y, case.lambda0, params.physics)
        row.update({"g_u": float(g[0]), "g_T": float(g[1]), "g_abs_max": float(np.max(np.abs(g)))})
    except Exception as exc:
        g = np.array([np.nan, np.nan], dtype=float)
        row.update({"g_u": np.nan, "g_T": np.nan, "g_abs_max": np.nan, "rhs_error": str(exc)})
    try:
        state = algebraic_state(logR, float(y[0]), float(y[1]), case.lambda0, params.physics)
        row.update(
            {
                "u_c": float(state.u),
                "T_K": float(state.T),
                "Sigma": float(state.Sigma),
                "rho": float(state.rho),
                "P": float(state.P),
                "H_R": float(state.H_over_R),
                "Omega_over_K": float(state.Omega / state.Omega_K),
                "tau": float(state.tau),
                "entropy_proxy": float(np.log(state.P) - (5.0 / 3.0) * np.log(state.rho)),
            }
        )
        if np.all(np.isfinite(g)):
            qadv_qvisc, qrad_qvisc, qvisc_sign = heating_metrics(logR, y, g, case.lambda0, params)
            row.update(
                {
                    "qadv_qvisc": qadv_qvisc,
                    "qrad_qvisc": qrad_qvisc,
                    "qvisc_sign": qvisc_sign,
                }
            )
    except Exception as exc:
        row["state_error"] = str(exc)
    return row


def integrate_track(case, branch_index: int, branch, params) -> tuple[dict[str, object], list[dict[str, object]]]:
    x0 = float(case.logR_son + FLOW_EPS0)
    x1 = float(np.log(R_END_RG * params.r_g))
    y0 = np.asarray(case.y_s, dtype=float) + FLOW_EPS0 * np.asarray(branch.gradient, dtype=float)

    def rhs(x_value: float, y_value: np.ndarray) -> np.ndarray:
        return local_ode_rhs(float(x_value), y_value, case.lambda0, params.physics)

    try:
        sol = solve_ivp(
            rhs,
            (x0, x1),
            y0,
            method="Radau",
            max_step=FLOW_MAX_STEP,
            rtol=FLOW_RTOL,
            atol=FLOW_ATOL,
            dense_output=False,
        )
        success = bool(sol.success)
        message = str(sol.message)
        nfev = int(sol.nfev)
        logR_values = np.asarray(sol.t, dtype=float)
        y_values = np.asarray(sol.y.T, dtype=float)
    except Exception as exc:
        success = False
        message = str(exc)
        nfev = 0
        logR_values = np.array([x0], dtype=float)
        y_values = np.array([y0], dtype=float)

    rows = [diagnostic_row(case, branch_index, branch, float(logR), np.asarray(y, dtype=float), params) for logR, y in zip(logR_values, y_values)]
    finite = [row for row in rows if np.isfinite(float(row.get("R_rg", np.nan)))]
    last = finite[-1] if finite else {}
    tail = finite[max(0, len(finite) - 50) :]
    abs_D = np.asarray([abs(float(row.get("D", np.nan))) for row in tail], dtype=float)
    abs_K = np.asarray([abs(float(row.get("K", np.nan))) for row in tail], dtype=float)
    smin = np.asarray([float(row.get("smin_over_smax", np.nan)) for row in tail], dtype=float)
    g_abs = np.asarray([float(row.get("g_abs_max", np.nan)) for row in tail], dtype=float)
    H_R = np.asarray([float(row.get("H_R", np.nan)) for row in tail], dtype=float)
    omega = np.asarray([float(row.get("Omega_over_K", np.nan)) for row in tail], dtype=float)
    qadv = np.asarray([abs(float(row.get("qadv_qvisc", np.nan))) for row in tail], dtype=float)
    summary = {
        "case": case.label,
        "mode": case.mode,
        "branch": int(branch_index),
        "a": float(branch.a),
        "g_s_u": float(branch.gradient[0]),
        "g_s_T": float(branch.gradient[1]),
        "success": success,
        "message": message,
        "nfev": nfev,
        "n_points": len(rows),
        "Rson_rg": float(np.exp(case.logR_son) / params.r_g),
        "lambda0": float(case.lambda0),
        "critK_sonic": float(case.critK),
        "reached_R_rg": float(last.get("R_rg", np.nan)),
        "reached_dx": float(last.get("dx", np.nan)),
        "last_D": float(last.get("D", np.nan)),
        "last_C1": float(last.get("C1", np.nan)),
        "last_C2": float(last.get("C2", np.nan)),
        "last_K": float(last.get("K", np.nan)),
        "last_smin_over_smax": float(last.get("smin_over_smax", np.nan)),
        "last_condA": float(last.get("condA", np.nan)),
        "last_g_abs_max": float(last.get("g_abs_max", np.nan)),
        "last_H_R": float(last.get("H_R", np.nan)),
        "last_Omega_over_K": float(last.get("Omega_over_K", np.nan)),
        "last_qadv_qvisc": float(last.get("qadv_qvisc", np.nan)),
        "tail_min_abs_D": float(np.nanmin(abs_D)) if np.any(np.isfinite(abs_D)) else np.nan,
        "tail_min_abs_K": float(np.nanmin(abs_K)) if np.any(np.isfinite(abs_K)) else np.nan,
        "tail_min_smin_over_smax": float(np.nanmin(smin)) if np.any(np.isfinite(smin)) else np.nan,
        "tail_max_condA": float(1.0 / (np.nanmin(smin) + 1.0e-300)) if np.any(np.isfinite(smin)) else np.nan,
        "tail_max_g_abs": float(np.nanmax(g_abs)) if np.any(np.isfinite(g_abs)) else np.nan,
        "tail_max_H_R": float(np.nanmax(H_R)) if np.any(np.isfinite(H_R)) else np.nan,
        "tail_min_H_R": float(np.nanmin(H_R)) if np.any(np.isfinite(H_R)) else np.nan,
        "tail_max_abs_omega_over_K": float(np.nanmax(np.abs(omega))) if np.any(np.isfinite(omega)) else np.nan,
        "tail_max_abs_qadv_qvisc": float(np.nanmax(qadv)) if np.any(np.isfinite(qadv)) else np.nan,
    }
    summary["classification"] = classify_track(summary)
    return summary, rows


def classify_track(summary: dict[str, object]) -> str:
    min_smin = float(summary.get("tail_min_smin_over_smax", np.nan))
    max_g = float(summary.get("tail_max_g_abs", np.nan))
    max_H_R = float(summary.get("tail_max_H_R", np.nan))
    max_omega = float(summary.get("tail_max_abs_omega_over_K", np.nan))
    max_qadv = float(summary.get("tail_max_abs_qadv_qvisc", np.nan))
    if np.isfinite(min_smin) and min_smin < 1.0e-4 and np.isfinite(max_g) and max_g > 1.0e3:
        return "singular_matrix_or_second_critical"
    if (np.isfinite(max_H_R) and max_H_R > 0.5) or (np.isfinite(max_omega) and max_omega > 2.0) or (np.isfinite(max_qadv) and max_qadv > 1.0e3):
        return "physical_or_closure_breakdown"
    if not bool(summary.get("success", False)):
        return "numerical_branch_turn_or_stiff_barrier"
    return "regular_to_R_end"


def run_audit() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ctx = load_context()
    cases = [critical_case_from_free_fit(ctx), solve_fixed_dlambda(ctx, FIXED_DLAMBDA)]
    summaries: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for case in cases:
        if not np.isfinite(float(case.critK)) or not np.all(np.isfinite(case.y_s)):
            summaries.append(
                {
                    "case": case.label,
                    "mode": case.mode,
                    "success": False,
                    "message": case.message,
                    "classification": "critical_fit_failed",
                    "Rson_rg": float(np.exp(case.logR_son) / ctx.params.r_g),
                    "lambda0": float(case.lambda0),
                    "critK_sonic": float(case.critK),
                }
            )
            continue
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
        for branch_index in BRANCH_SEQUENCE:
            if branch_index >= len(branches):
                summaries.append(
                    {
                        "case": case.label,
                        "mode": case.mode,
                        "branch": branch_index,
                        "success": False,
                        "message": f"found only {len(branches)} branches",
                        "classification": "missing_branch",
                        "Rson_rg": float(np.exp(case.logR_son) / ctx.params.r_g),
                        "lambda0": float(case.lambda0),
                        "critK_sonic": float(case.critK),
                    }
                )
                continue
            summary, rows = integrate_track(case, branch_index, branches[branch_index], ctx.params)
            summaries.append(summary)
            point_rows.extend(rows)
    return summaries, point_rows


def write_outputs(summaries: list[dict[str, object]], point_rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Transonic Branch Barrier Audit",
        "",
        "Generated by `scripts/run_transonic_branch_barrier_audit.py`.",
        "",
        f"Target integration end: `{fmt(R_END_RG)} rg`; `eps0={FLOW_EPS0:g}`, `max_step={FLOW_MAX_STEP:g}`.",
        "",
        "| case | branch | class | reached R/rg | reached dx | min smin/smax | max condA | max |g| | last D | last K | last H/R | last Omega/K | last Qadv/Qvisc | Rson/rg | lambda0 | critK | nfev | message |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {case} | {branch} | {classification} | {reached_R_rg} | {reached_dx} | "
            "{tail_min_smin_over_smax} | {tail_max_condA} | {tail_max_g_abs} | {last_D} | "
            "{last_K} | {last_H_R} | {last_Omega_over_K} | {last_qadv_qvisc} | "
            "{Rson_rg} | {lambda0} | {critK_sonic} | {nfev} | {message} |".format(
                case=row.get("case", ""),
                branch=row.get("branch", ""),
                classification=row.get("classification", ""),
                reached_R_rg=fmt(float(row.get("reached_R_rg", np.nan))),
                reached_dx=fmt(float(row.get("reached_dx", np.nan))),
                tail_min_smin_over_smax=fmt(float(row.get("tail_min_smin_over_smax", np.nan))),
                tail_max_condA=fmt(float(row.get("tail_max_condA", np.nan))),
                tail_max_g_abs=fmt(float(row.get("tail_max_g_abs", np.nan))),
                last_D=fmt(float(row.get("last_D", np.nan))),
                last_K=fmt(float(row.get("last_K", np.nan))),
                last_H_R=fmt(float(row.get("last_H_R", np.nan))),
                last_Omega_over_K=fmt(float(row.get("last_Omega_over_K", np.nan))),
                last_qadv_qvisc=fmt(float(row.get("last_qadv_qvisc", np.nan))),
                Rson_rg=fmt(float(row.get("Rson_rg", np.nan))),
                lambda0=fmt(float(row.get("lambda0", np.nan))),
                critK_sonic=fmt(float(row.get("critK_sonic", np.nan))),
                nfev=row.get("nfev", ""),
                message=str(row.get("message", "")).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    POINTS_OUTPUT.write_text(json.dumps([{key: json_safe(value) for key, value in row.items()} for row in point_rows], indent=2, sort_keys=True) + "\n")
    write_figure(point_rows)


def write_figure(point_rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"figure skipped: {exc}", flush=True)
        return

    rows = [
        row
        for row in point_rows
        if row.get("case") == "free_reverse_fit"
        and int(row.get("branch", -1)) == 1
        and np.isfinite(float(row.get("R_rg", np.nan)))
    ]
    if not rows:
        return
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 22), "Barrier audit: free reverse fit, branch 1", fill=(20, 20, 20))
    panels = [
        ("abs(D)", lambda row: abs(float(row.get("D", np.nan))), (31, 119, 180)),
        ("smin/smax", lambda row: float(row.get("smin_over_smax", np.nan)), (214, 39, 40)),
        ("max |g|", lambda row: float(row.get("g_abs_max", np.nan)), (44, 160, 44)),
        ("|Qadv/Qvisc|", lambda row: abs(float(row.get("qadv_qvisc", np.nan))), (148, 103, 189)),
    ]
    plot_l, plot_r = 90, width - 50
    panel_h = 130
    x_values = np.asarray([float(row["R_rg"]) for row in rows], dtype=float)
    xmin, xmax = float(np.nanmin(x_values)), float(np.nanmax(x_values))
    for idx, (label, getter, color) in enumerate(panels):
        top = 60 + idx * 155
        bottom = top + panel_h
        draw.rectangle((plot_l, top, plot_r, bottom), outline=(50, 50, 50), width=1)
        y_values = np.asarray([getter(row) for row in rows], dtype=float)
        finite = np.isfinite(y_values) & (y_values > 0.0)
        if not np.any(finite):
            continue
        ymin = max(float(np.nanmin(y_values[finite])) * 0.7, 1.0e-12)
        ymax = max(float(np.nanmax(y_values[finite])) * 1.4, ymin * 10.0)

        def sx(value: float) -> float:
            return plot_l + (value - xmin) / (xmax - xmin + 1.0e-300) * (plot_r - plot_l)

        def sy(value: float) -> float:
            return bottom - (np.log10(value) - np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin) + 1.0e-300) * (bottom - top)

        points = [(sx(float(x)), sy(float(y))) for x, y in zip(x_values[finite], y_values[finite])]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        for xpix, ypix in points[:: max(1, len(points) // 80)]:
            draw.ellipse((xpix - 2, ypix - 2, xpix + 2, ypix + 2), fill=color)
        draw.text((plot_l + 8, top + 8), label, fill=color)
        draw.text((plot_l - 75, top + 8), f"{ymax:.2g}", fill=(20, 20, 20))
        draw.text((plot_l - 75, bottom - 15), f"{ymin:.2g}", fill=(20, 20, 20))
    draw.text((plot_l, height - 35), f"R/rg from {xmin:.4g} to {xmax:.4g}", fill=(20, 20, 20))
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    summaries, point_rows = run_audit()
    write_outputs(summaries, point_rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {POINTS_OUTPUT}", flush=True)
    if FIGURE_OUTPUT.exists():
        print(f"wrote {FIGURE_OUTPUT}", flush=True)
    for row in summaries:
        print(
            f"{row.get('case')} branch={row.get('branch')} class={row.get('classification')} "
            f"R={float(row.get('reached_R_rg', np.nan)):.6g} "
            f"min_s={float(row.get('tail_min_smin_over_smax', np.nan)):.3e} "
            f"max_g={float(row.get('tail_max_g_abs', np.nan)):.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
