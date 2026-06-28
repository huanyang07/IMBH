"""Desingularized phase-space flow through the second-critical barrier."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    algebraic_state,
    local_ode_rhs,
    phase_space_null_tangent,
    sonic_diagnostics,
)

from run_transonic_barrier_critical_probe import load_seed_rows, solve_probe
from run_transonic_branch_barrier_audit import POINTS_OUTPUT
from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_sonic_reverse_fit import load_context


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_DESINGULARIZED_FLOW_TABLE",
    "outputs/tables/transonic_desingularized_barrier_flow.md",
)
TRACE_OUTPUT = TABLE_OUTPUT.with_name(TABLE_OUTPUT.stem + "_trace.json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_DESINGULARIZED_FLOW_FIGURE",
    "outputs/figures/transonic_desingularized_barrier_flow.png",
)

START_R_RG = float(os.environ.get("IMBH_DESINGULARIZED_FLOW_START_R_RG", "6.0"))
TARGET_R_RG = float(os.environ.get("IMBH_DESINGULARIZED_FLOW_TARGET_R_RG", "7.0"))
S_MAX = float(os.environ.get("IMBH_DESINGULARIZED_FLOW_S_MAX", "1.2"))
DS = float(os.environ.get("IMBH_DESINGULARIZED_FLOW_DS", "2e-4"))
PX_FLOOR_FOR_G = float(os.environ.get("IMBH_DESINGULARIZED_FLOW_PX_FLOOR", "1e-8"))
MAX_STEPS = int(os.environ.get("IMBH_DESINGULARIZED_FLOW_MAX_STEPS", "12000"))


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def incoming_trace_rows() -> list[dict[str, object]]:
    if not POINTS_OUTPUT.exists():
        raise RuntimeError(f"missing barrier-audit trace {POINTS_OUTPUT}")
    rows = json.loads(POINTS_OUTPUT.read_text())
    return [
        row
        for row in rows
        if row.get("case") == "free_reverse_fit"
        and int(row.get("branch", -1)) == 1
        and np.isfinite(float(row.get("logR", np.nan)))
    ]


def incoming_state_at(R_rg: float) -> tuple[float, np.ndarray, float]:
    rows = incoming_trace_rows()
    if not rows:
        raise RuntimeError("missing incoming free_reverse_fit branch=1 rows")
    order = np.argsort([float(row["logR"]) for row in rows])
    logR = np.asarray([float(rows[idx]["logR"]) for idx in order], dtype=float)
    logu = np.asarray([float(rows[idx]["logu"]) for idx in order], dtype=float)
    logT = np.asarray([float(rows[idx]["logT"]) for idx in order], dtype=float)
    lambda0 = float(rows[int(order[0])]["lambda0"])
    ctx = load_context()
    target_logR = float(np.log(R_rg * ctx.params.r_g))
    if not logR[0] <= target_logR <= logR[-1]:
        raise ValueError(f"R={R_rg:g} rg is outside incoming trace range")
    return (
        target_logR,
        np.array(
            [
                float(PchipInterpolator(logR, logu)(target_logR)),
                float(PchipInterpolator(logR, logT)(target_logR)),
            ],
            dtype=float,
        ),
        lambda0,
    )


def fixed_lambda_critical_state() -> tuple[float, np.ndarray, float]:
    ctx = load_context()
    seeds = [
        seed
        for seed in load_seed_rows()
        if seed.get("case") == "free_reverse_fit" and int(seed.get("branch", -1)) == 1
    ]
    if not seeds:
        raise RuntimeError("missing free_reverse_fit branch=1 seed")
    candidate = solve_probe(ctx, seeds[0], free_lambda=False)
    return (
        float(np.log(float(candidate["R_rg"]) * ctx.params.r_g)),
        np.array([float(candidate["logu"]), float(candidate["logT"])], dtype=float),
        float(candidate["lambda0"]),
    )


def diagnostics(label: str, s_value: float, z: np.ndarray, tangent: np.ndarray, lambda0: float, params) -> dict[str, object]:
    logR = float(z[0])
    y = np.asarray(z[1:], dtype=float)
    tangent_diag = phase_space_null_tangent(logR, y, lambda0, params.physics, previous=tangent)
    p = np.asarray(tangent_diag.tangent, dtype=float)
    sonic = sonic_diagnostics(logR, y, lambda0, params.physics)
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params.physics)
    if abs(float(p[0])) > PX_FLOOR_FOR_G:
        g_phase = p[1:] / p[0]
        try:
            g_ode = local_ode_rhs(logR, y, lambda0, params.physics)
            g_error = float(np.max(np.abs(g_phase - g_ode)))
        except Exception:
            g_ode = np.array([np.nan, np.nan], dtype=float)
            g_error = np.nan
        try:
            q_visc, q_rad, q_adv, _energy = _heating_terms_from_gradient(logR, y, g_phase, lambda0, params.physics)
            qadv_qvisc = float(q_adv / (q_visc + 1.0e-300))
            qrad_qvisc = float(q_rad / (q_visc + 1.0e-300))
            qvisc_sign = float(np.sign(q_visc))
        except Exception:
            qadv_qvisc = np.nan
            qrad_qvisc = np.nan
            qvisc_sign = np.nan
    else:
        g_phase = np.array([np.nan, np.nan], dtype=float)
        g_ode = np.array([np.nan, np.nan], dtype=float)
        g_error = np.nan
        qadv_qvisc = np.nan
        qrad_qvisc = np.nan
        qvisc_sign = np.nan
    return {
        "label": label,
        "s": float(s_value),
        "logR": logR,
        "R_rg": float(np.exp(logR) / params.r_g),
        "logu": float(y[0]),
        "logT": float(y[1]),
        "lambda0": float(lambda0),
        "p_x": float(p[0]),
        "p_u": float(p[1]),
        "p_T": float(p[2]),
        "Bp_max": float(np.max(np.abs(tangent_diag.residual))),
        "smin_over_smax_A": float(tangent_diag.smin_over_smax_A),
        "smin_over_smax_B": float(tangent_diag.smin_over_smax_B),
        "condA": float(1.0 / (tangent_diag.smin_over_smax_A + 1.0e-300)),
        "condB": float(1.0 / (tangent_diag.smin_over_smax_B + 1.0e-300)),
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "M_eff": float(sonic.M_eff),
        "H_R": float(state.H_over_R),
        "Omega_over_K": float(state.Omega / state.Omega_K),
        "qadv_qvisc": qadv_qvisc,
        "qrad_qvisc": qrad_qvisc,
        "qvisc_sign": qvisc_sign,
        "g_phase_u": float(g_phase[0]),
        "g_phase_T": float(g_phase[1]),
        "g_ode_u": float(g_ode[0]),
        "g_ode_T": float(g_ode[1]),
        "g_error": g_error,
    }


def heun_step(z: np.ndarray, lambda0: float, params, ds: float, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = phase_space_null_tangent(float(z[0]), z[1:], lambda0, params.physics, previous=previous).tangent
    z_pred = z + ds * p0
    p1 = phase_space_null_tangent(float(z_pred[0]), z_pred[1:], lambda0, params.physics, previous=p0).tangent
    z_new = z + 0.5 * ds * (p0 + p1)
    p_new = phase_space_null_tangent(float(z_new[0]), z_new[1:], lambda0, params.physics, previous=p1).tangent
    return z_new, p_new


def run_flow(label: str, logR0: float, y0: np.ndarray, lambda0: float, initial_sign: float) -> tuple[dict[str, object], list[dict[str, object]]]:
    ctx = load_context()
    z = np.array([logR0, float(y0[0]), float(y0[1])], dtype=float)
    p0 = phase_space_null_tangent(float(z[0]), z[1:], lambda0, ctx.params.physics, prefer_positive_x=True).tangent
    if initial_sign < 0.0:
        p0 = -p0
    rows = [diagnostics(label, 0.0, z, p0, lambda0, ctx.params)]
    p = p0
    s_value = 0.0
    target_crossed = False
    px_sign_changes = 0
    previous_px_sign = float(np.sign(p[0]))
    message = "completed s_max"
    for _step in range(MAX_STEPS):
        if s_value >= S_MAX:
            break
        try:
            z, p = heun_step(z, lambda0, ctx.params, DS, p)
        except Exception as exc:
            message = str(exc)
            break
        s_value += DS
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(p)):
            message = "non-finite z or p"
            break
        row = diagnostics(label, s_value, z, p, lambda0, ctx.params)
        rows.append(row)
        px_sign = float(np.sign(row["p_x"]))
        if px_sign != 0.0 and previous_px_sign != 0.0 and px_sign != previous_px_sign:
            px_sign_changes += 1
        if px_sign != 0.0:
            previous_px_sign = px_sign
        if float(row["R_rg"]) >= TARGET_R_RG:
            target_crossed = True
            message = f"reached R >= {TARGET_R_RG:g} rg"
            break
        if float(row["R_rg"]) < 2.05:
            message = "fell near inner horizon"
            break
    R_values = np.asarray([float(row["R_rg"]) for row in rows], dtype=float)
    px_values = np.asarray([float(row["p_x"]) for row in rows], dtype=float)
    smin_A = np.asarray([float(row["smin_over_smax_A"]) for row in rows], dtype=float)
    smin_B = np.asarray([float(row["smin_over_smax_B"]) for row in rows], dtype=float)
    H_R = np.asarray([float(row["H_R"]) for row in rows], dtype=float)
    omega = np.asarray([float(row["Omega_over_K"]) for row in rows], dtype=float)
    summary = {
        "label": label,
        "initial_sign": float(initial_sign),
        "n_steps": len(rows) - 1,
        "s_end": float(s_value),
        "message": message,
        "target_crossed": bool(target_crossed),
        "px_sign_changes": int(px_sign_changes),
        "R_start_rg": float(R_values[0]),
        "R_end_rg": float(R_values[-1]),
        "R_min_rg": float(np.nanmin(R_values)),
        "R_max_rg": float(np.nanmax(R_values)),
        "min_abs_px": float(np.nanmin(np.abs(px_values))),
        "final_px": float(px_values[-1]),
        "min_smin_A": float(np.nanmin(smin_A)),
        "min_smin_B": float(np.nanmin(smin_B)),
        "max_condA": float(1.0 / (np.nanmin(smin_A) + 1.0e-300)),
        "max_condB": float(1.0 / (np.nanmin(smin_B) + 1.0e-300)),
        "max_H_R": float(np.nanmax(H_R)),
        "min_H_R": float(np.nanmin(H_R)),
        "max_abs_Omega_over_K": float(np.nanmax(np.abs(omega))),
        "classification": classify_flow(R_values, px_values, smin_B, H_R, omega, target_crossed),
    }
    return summary, rows


def classify_flow(
    R_values: np.ndarray,
    px_values: np.ndarray,
    smin_B: np.ndarray,
    H_R: np.ndarray,
    omega: np.ndarray,
    target_crossed: bool,
) -> str:
    if np.any(np.isfinite(smin_B)) and float(np.nanmin(smin_B)) < 1.0e-8:
        return "B_singular_or_degenerate"
    if np.any(np.isfinite(H_R)) and float(np.nanmax(H_R)) > 0.5:
        return "physical_thickness_breakdown"
    if np.any(np.isfinite(omega)) and float(np.nanmax(np.abs(omega))) > 2.0:
        return "physical_rotation_breakdown"
    if target_crossed:
        return "crosses_to_target_R"
    if float(np.nanmax(R_values)) - float(R_values[0]) > 0.0 and float(R_values[-1]) < float(np.nanmax(R_values)) - 1.0e-4:
        return "R_turnaround"
    if np.nanmin(np.abs(px_values)) < 1.0e-3:
        return "near_projected_fold"
    return "no_crossing_within_smax"


def write_table(summaries: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Desingularized Barrier Flow",
        "",
        "Generated by `scripts/run_transonic_desingularized_barrier_flow.py`.",
        "",
        f"Config: `ds={DS:g}`, `s_max={S_MAX:g}`, `start_R={START_R_RG:g} rg`, `target_R={TARGET_R_RG:g} rg`.",
        "",
        "| label | class | R start | R end | R min | R max | target | px flips | min |p_x| | min sA | min sB | max condA | max condB | max H/R | max |Omega/K| | steps | message |",
        "|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {label} | {classification} | {R_start_rg} | {R_end_rg} | {R_min_rg} | {R_max_rg} | "
            "{target_crossed} | {px_sign_changes} | {min_abs_px} | {min_smin_A} | {min_smin_B} | "
            "{max_condA} | {max_condB} | {max_H_R} | {max_abs_Omega_over_K} | {n_steps} | {message} |".format(
                label=row["label"],
                classification=row["classification"],
                R_start_rg=fmt(float(row["R_start_rg"])),
                R_end_rg=fmt(float(row["R_end_rg"])),
                R_min_rg=fmt(float(row["R_min_rg"])),
                R_max_rg=fmt(float(row["R_max_rg"])),
                target_crossed="yes" if row["target_crossed"] else "no",
                px_sign_changes=row["px_sign_changes"],
                min_abs_px=fmt(float(row["min_abs_px"])),
                min_smin_A=fmt(float(row["min_smin_A"])),
                min_smin_B=fmt(float(row["min_smin_B"])),
                max_condA=fmt(float(row["max_condA"])),
                max_condB=fmt(float(row["max_condB"])),
                max_H_R=fmt(float(row["max_H_R"])),
                max_abs_Omega_over_K=fmt(float(row["max_abs_Omega_over_K"])),
                n_steps=row["n_steps"],
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def write_trace(rows: list[dict[str, object]]) -> None:
    TRACE_OUTPUT.write_text(json.dumps([{key: json_safe(value) for key, value in row.items()} for row in rows], indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"figure skipped: {exc}", flush=True)
        return
    labels = list(dict.fromkeys(str(row["label"]) for row in rows))
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 20), "Desingularized phase-space barrier flow", fill=(20, 20, 20))
    panels = [
        ("R/rg", lambda row: float(row["R_rg"]), False),
        ("signed p_x", lambda row: float(row["p_x"]), False),
        ("smin/smax A", lambda row: float(row["smin_over_smax_A"]), True),
        ("smin/smax B", lambda row: float(row["smin_over_smax_B"]), True),
    ]
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    plot_l, plot_r = 90, width - 50
    panel_h = 135
    for pidx, (title, getter, logy) in enumerate(panels):
        top = 60 + 165 * pidx
        bottom = top + panel_h
        draw.rectangle((plot_l, top, plot_r, bottom), outline=(60, 60, 60), width=1)
        all_s = np.asarray([float(row["s"]) for row in rows], dtype=float)
        finite_s = all_s[np.isfinite(all_s)]
        if finite_s.size == 0:
            continue
        xmin, xmax = float(np.min(finite_s)), float(np.max(finite_s))
        values = np.asarray([getter(row) for row in rows], dtype=float)
        finite_values = values[np.isfinite(values) & ((values > 0.0) if logy else np.ones_like(values, dtype=bool))]
        if finite_values.size == 0:
            continue
        ymin = float(np.min(finite_values))
        ymax = float(np.max(finite_values))
        if logy:
            ymin = max(ymin * 0.7, 1.0e-16)
            ymax = max(ymax * 1.4, ymin * 10.0)
        else:
            pad = max((ymax - ymin) * 0.1, 1.0e-3)
            ymin -= pad
            ymax += pad

        def sx(value: float) -> float:
            return plot_l + (value - xmin) / (xmax - xmin + 1.0e-300) * (plot_r - plot_l)

        def sy(value: float) -> float:
            if logy:
                return bottom - (np.log10(value) - np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin) + 1.0e-300) * (bottom - top)
            return bottom - (value - ymin) / (ymax - ymin + 1.0e-300) * (bottom - top)

        draw.text((plot_l + 8, top + 8), title, fill=(20, 20, 20))
        if title == "signed p_x" and ymin < 0.0 < ymax:
            y_zero = sy(0.0)
            draw.line((plot_l, y_zero, plot_r, y_zero), fill=(150, 150, 150), width=1)
        for lidx, label in enumerate(labels):
            series = [row for row in rows if row["label"] == label]
            points = []
            for row in series:
                value = getter(row)
                if np.isfinite(value) and ((value > 0.0) if logy else True):
                    points.append((sx(float(row["s"])), sy(float(value))))
            color = colors[lidx % len(colors)]
            if len(points) >= 2:
                draw.line(points, fill=color, width=2)
            if pidx == 0:
                draw.text((plot_l + 12, top + 22 + 16 * lidx), label, fill=color)
    draw.text((plot_l, height - 25), "s", fill=(20, 20, 20))
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    incoming_logR, incoming_y, incoming_lambda0 = incoming_state_at(START_R_RG)
    critical_logR, critical_y, critical_lambda0 = fixed_lambda_critical_state()
    cases = [
        ("incoming_R6_plus", incoming_logR, incoming_y, incoming_lambda0, 1.0),
        ("fixedcrit_plus", critical_logR, critical_y, critical_lambda0, 1.0),
        ("fixedcrit_minus", critical_logR, critical_y, critical_lambda0, -1.0),
    ]
    summaries: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    for label, logR, y, lambda0, sign in cases:
        summary, rows = run_flow(label, logR, y, lambda0, sign)
        summaries.append(summary)
        all_rows.extend(rows)
        print(
            f"{label}: class={summary['classification']} Rmax={summary['R_max_rg']:.6g} "
            f"Rend={summary['R_end_rg']:.6g} px_flips={summary['px_sign_changes']} "
            f"min_sB={summary['min_smin_B']:.3e}",
            flush=True,
        )
    write_table(summaries)
    write_trace(all_rows)
    write_figure(all_rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {TRACE_OUTPUT}", flush=True)
    if FIGURE_OUTPUT.exists():
        print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
