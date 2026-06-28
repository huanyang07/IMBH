"""Event-refined radial fold diagnostics for the desingularized transonic flow."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    B_rank_minors,
    algebraic_state,
    phase_space_null_tangent,
    phase_space_tangent_derivative,
    sonic_diagnostics,
)

from run_transonic_desingularized_barrier_flow import incoming_state_at, json_safe
from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_sonic_reverse_fit import load_context


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_FOLD_REFINEMENT_TABLE",
    "outputs/tables/transonic_desingularized_fold_refinement.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_name(TABLE_OUTPUT.stem + ".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_FOLD_REFINEMENT_FIGURE",
    "outputs/figures/transonic_desingularized_fold_refinement.png",
)

START_R_RG = float(os.environ.get("IMBH_FOLD_REFINEMENT_START_R_RG", "6.0"))
S_MAX = float(os.environ.get("IMBH_FOLD_REFINEMENT_S_MAX", "3.2"))
MAX_STEPS = int(os.environ.get("IMBH_FOLD_REFINEMENT_MAX_STEPS", "100000"))
BISECTION_STEPS = int(os.environ.get("IMBH_FOLD_REFINEMENT_BISECTION_STEPS", "44"))
TANGENT_EPS = float(os.environ.get("IMBH_FOLD_REFINEMENT_TANGENT_EPS", "2e-6"))
DS_VALUES = tuple(
    float(value)
    for value in os.environ.get("IMBH_FOLD_REFINEMENT_DS", "5e-4,2e-4,1e-4,5e-5").split(",")
    if value.strip()
)


METRICS: dict[str, np.ndarray | None] = {
    "euclidean": None,
    "x4": np.array([4.0, 1.0, 1.0], dtype=float),
}


def tangent_at(z: np.ndarray, lambda0: float, params, metric, previous=None):
    return phase_space_null_tangent(float(z[0]), z[1:], lambda0, params.physics, metric=metric, previous=previous).tangent


def heun_step(z: np.ndarray, lambda0: float, params, ds: float, previous: np.ndarray, metric) -> tuple[np.ndarray, np.ndarray]:
    p0 = tangent_at(z, lambda0, params, metric, previous=previous)
    z_pred = z + ds * p0
    p1 = tangent_at(z_pred, lambda0, params, metric, previous=p0)
    z_new = z + 0.5 * ds * (p0 + p1)
    p_new = tangent_at(z_new, lambda0, params, metric, previous=p1)
    return z_new, p_new


def q_ratios(logR: float, y: np.ndarray, p: np.ndarray, lambda0: float, params) -> tuple[float, float, float]:
    if abs(float(p[0])) < 1.0e-12:
        return np.nan, np.nan, np.nan
    try:
        g = p[1:] / p[0]
        q_visc, q_rad, q_adv, _energy = _heating_terms_from_gradient(logR, y, g, lambda0, params.physics)
    except Exception:
        return np.nan, np.nan, np.nan
    return (
        float(q_adv / (q_visc + 1.0e-300)),
        float(q_rad / (q_visc + 1.0e-300)),
        float(np.sign(q_visc)),
    )


def row_diagnostics(
    metric_name: str,
    ds: float,
    status: str,
    s_fold: float,
    z_fold: np.ndarray,
    p_fold: np.ndarray,
    lambda0: float,
    params,
    bracket_width_s: float,
    q_left: tuple[float, float, float],
    q_right: tuple[float, float, float],
) -> dict[str, object]:
    tangent_diag = phase_space_null_tangent(float(z_fold[0]), z_fold[1:], lambda0, params.physics, metric=METRICS[metric_name], previous=p_fold)
    p = np.asarray(tangent_diag.tangent, dtype=float)
    dp_ds = phase_space_tangent_derivative(
        float(z_fold[0]),
        z_fold[1:],
        lambda0,
        params.physics,
        p,
        metric=METRICS[metric_name],
        eps=TANGENT_EPS,
    )
    sonic = sonic_diagnostics(float(z_fold[0]), z_fold[1:], lambda0, params.physics)
    state = algebraic_state(float(z_fold[0]), float(z_fold[1]), float(z_fold[2]), lambda0, params.physics)
    minors = B_rank_minors(float(z_fold[0]), z_fold[1:], lambda0, params.physics)
    return {
        "metric": metric_name,
        "ds": float(ds),
        "status": status,
        "s_fold": float(s_fold),
        "R_fold_rg": float(np.exp(z_fold[0]) / params.r_g),
        "logR_fold": float(z_fold[0]),
        "logu_fold": float(z_fold[1]),
        "logT_fold": float(z_fold[2]),
        "lambda0": float(lambda0),
        "p_x": float(p[0]),
        "p_u": float(p[1]),
        "p_T": float(p[2]),
        "dpx_ds": float(dp_ds[0]),
        "dpu_ds": float(dp_ds[1]),
        "dpT_ds": float(dp_ds[2]),
        "Bp_max": float(np.max(np.abs(tangent_diag.residual))),
        "smin_over_smax_A": float(tangent_diag.smin_over_smax_A),
        "smin_over_smax_B": float(tangent_diag.smin_over_smax_B),
        "condA": float(1.0 / (tangent_diag.smin_over_smax_A + 1.0e-300)),
        "condB": float(1.0 / (tangent_diag.smin_over_smax_B + 1.0e-300)),
        "B_minors_norm": float(np.linalg.norm(minors)),
        "B_minors_max": float(np.max(np.abs(minors))),
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "M_eff": float(sonic.M_eff),
        "H_R": float(state.H_over_R),
        "Omega_over_K": float(state.Omega / state.Omega_K),
        "qadv_qvisc_left": float(q_left[0]),
        "qrad_qvisc_left": float(q_left[1]),
        "qvisc_sign_left": float(q_left[2]),
        "qadv_qvisc_right": float(q_right[0]),
        "qrad_qvisc_right": float(q_right[1]),
        "qvisc_sign_right": float(q_right[2]),
        "bracket_width_s": float(bracket_width_s),
    }


def refine_bracket(
    left: tuple[float, np.ndarray, np.ndarray],
    right: tuple[float, np.ndarray, np.ndarray],
    lambda0: float,
    params,
    metric,
) -> tuple[float, np.ndarray, np.ndarray, float, tuple[float, float, float], tuple[float, float, float]]:
    s_left, z_left, p_left = left
    s_right, z_right, p_right = right
    for _idx in range(BISECTION_STEPS):
        ds_mid = 0.5 * (s_right - s_left)
        z_mid, p_mid = heun_step(z_left, lambda0, params, ds_mid, p_left, metric)
        s_mid = s_left + ds_mid
        if float(p_mid[0]) > 0.0:
            s_left, z_left, p_left = s_mid, z_mid, p_mid
        else:
            s_right, z_right, p_right = s_mid, z_mid, p_mid
    s_fold = 0.5 * (s_left + s_right)
    z_fold = 0.5 * (z_left + z_right)
    p_fold = tangent_at(z_fold, lambda0, params, metric, previous=p_left)
    q_left = q_ratios(float(z_left[0]), z_left[1:], p_left, lambda0, params)
    q_right = q_ratios(float(z_right[0]), z_right[1:], p_right, lambda0, params)
    return s_fold, z_fold, p_fold, s_right - s_left, q_left, q_right


def locate_fold(metric_name: str, metric, ds: float) -> dict[str, object]:
    ctx = load_context()
    logR0, y0, lambda0 = incoming_state_at(START_R_RG)
    z = np.array([logR0, float(y0[0]), float(y0[1])], dtype=float)
    p = tangent_at(z, lambda0, ctx.params, metric, previous=None)
    s_value = 0.0
    left: tuple[float, np.ndarray, np.ndarray] | None = None
    previous_sign = float(np.sign(p[0]))
    for _step in range(MAX_STEPS):
        if s_value >= S_MAX:
            break
        z_prev = z.copy()
        p_prev = p.copy()
        s_prev = s_value
        try:
            z, p = heun_step(z, lambda0, ctx.params, ds, p, metric)
        except Exception as exc:
            return {
                "metric": metric_name,
                "ds": float(ds),
                "status": f"step_failed:{exc}",
                "s_fold": np.nan,
                "R_fold_rg": np.nan,
            }
        s_value += ds
        current_sign = float(np.sign(p[0]))
        if previous_sign > 0.0 and current_sign < 0.0:
            left = (s_prev, z_prev, p_prev)
            right = (s_value, z.copy(), p.copy())
            s_fold, z_fold, p_fold, bracket_width_s, q_left, q_right = refine_bracket(left, right, lambda0, ctx.params, metric)
            return row_diagnostics(
                metric_name,
                ds,
                "fold_found",
                s_fold,
                z_fold,
                p_fold,
                lambda0,
                ctx.params,
                bracket_width_s,
                q_left,
                q_right,
            )
        if current_sign != 0.0:
            previous_sign = current_sign
    return {
        "metric": metric_name,
        "ds": float(ds),
        "status": "fold_not_found",
        "s_fold": float(s_value),
        "R_fold_rg": float(np.exp(z[0]) / ctx.params.r_g),
    }


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    def fixed(value: object, digits: int) -> str:
        number = float(value)
        return f"{number:.{digits}f}" if np.isfinite(number) else "nan"

    def sci(value: object, digits: int = 3) -> str:
        number = float(value)
        return f"{number:.{digits}e}" if np.isfinite(number) else "nan"

    def ratio(value: object) -> str:
        number = float(value)
        return f"{number:.6g}" if np.isfinite(number) else "undef"

    lines = [
        "# Desingularized Fold Refinement",
        "",
        "Generated by `scripts/run_transonic_desingularized_fold_refinement.py`.",
        "",
        f"Config: `start_R={START_R_RG:g} rg`, `s_max={S_MAX:g}`, `tangent_eps={TANGENT_EPS:g}`.",
        "",
        "| metric | ds | status | R_fold | s_fold | p_x | dp_x/ds | sA | sB | condB | D | K | H/R | Omega/K | qadv/qvisc L | qadv/qvisc R | bracket ds |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {ds} | {status} | {R_fold} | {s_fold} | {p_x} | {dpx_ds} | "
            "{sA} | {sB} | {condB} | {D} | {K} | {H_R} | {Omega} | {qL} | {qR} | {bracket} |".format(
                metric=row.get("metric", ""),
                ds=sci(row.get("ds", np.nan), digits=1),
                status=str(row.get("status", "")),
                R_fold=fixed(row.get("R_fold_rg", np.nan), 9),
                s_fold=fixed(row.get("s_fold", np.nan), 9),
                p_x=sci(row.get("p_x", np.nan)),
                dpx_ds=sci(row.get("dpx_ds", np.nan)),
                sA=sci(row.get("smin_over_smax_A", np.nan)),
                sB=sci(row.get("smin_over_smax_B", np.nan)),
                condB=sci(row.get("condB", np.nan)),
                D=sci(row.get("D", np.nan)),
                K=sci(row.get("K", np.nan)),
                H_R=sci(row.get("H_R", np.nan)),
                Omega=fixed(row.get("Omega_over_K", np.nan), 6),
                qL=ratio(row.get("qadv_qvisc_left", np.nan)),
                qR=ratio(row.get("qadv_qvisc_right", np.nan)),
                bracket=sci(row.get("bracket_width_s", np.nan)),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def write_json(rows: list[dict[str, object]]) -> None:
    JSON_OUTPUT.write_text(
        __import__("json").dumps([{key: json_safe(value) for key, value in row.items()} for row in rows], indent=2, sort_keys=True)
        + "\n"
    )


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"figure skipped: {exc}", flush=True)
        return
    valid = [row for row in rows if row.get("status") == "fold_found"]
    if not valid:
        return
    width, height = 980, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 24), "Desingularized radial-fold refinement", fill=(20, 20, 20))
    panels = [
        ("R_fold/rg", "R_fold_rg", False),
        ("dp_x/ds", "dpx_ds", False),
        ("smin/smax B", "smin_over_smax_B", True),
    ]
    colors = {"euclidean": (31, 119, 180), "x4": (214, 39, 40)}
    plot_l, plot_r = 90, width - 50
    panel_h = 130
    for pidx, (title, key, logy) in enumerate(panels):
        top = 70 + 165 * pidx
        bottom = top + panel_h
        draw.rectangle((plot_l, top, plot_r, bottom), outline=(60, 60, 60), width=1)
        ds_values = np.asarray([float(row["ds"]) for row in valid], dtype=float)
        values = np.asarray([float(row[key]) for row in valid], dtype=float)
        xmin, xmax = float(np.min(ds_values)), float(np.max(ds_values))
        finite = values[np.isfinite(values) & ((values > 0.0) if logy else np.ones_like(values, dtype=bool))]
        if finite.size == 0:
            continue
        ymin, ymax = float(np.min(finite)), float(np.max(finite))
        if logy:
            ymin = max(ymin * 0.7, 1.0e-16)
            ymax = max(ymax * 1.4, ymin * 10.0)
        else:
            pad = max(0.1 * (ymax - ymin), 1.0e-12)
            ymin -= pad
            ymax += pad

        def sx(value: float) -> float:
            return plot_r - (value - xmin) / (xmax - xmin + 1.0e-300) * (plot_r - plot_l)

        def sy(value: float) -> float:
            if logy:
                return bottom - (np.log10(value) - np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin) + 1.0e-300) * (bottom - top)
            return bottom - (value - ymin) / (ymax - ymin + 1.0e-300) * (bottom - top)

        draw.text((plot_l + 8, top + 8), title, fill=(20, 20, 20))
        for metric_name in METRICS:
            series = sorted([row for row in valid if row["metric"] == metric_name], key=lambda row: float(row["ds"]), reverse=True)
            points = [(sx(float(row["ds"])), sy(float(row[key]))) for row in series if np.isfinite(float(row[key]))]
            color = colors.get(metric_name, (44, 160, 44))
            if len(points) >= 2:
                draw.line(points, fill=color, width=2)
            for point in points:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
            if pidx == 0:
                draw.text((plot_l + 12, top + 24 + 16 * list(METRICS).index(metric_name)), metric_name, fill=color)
    draw.text((plot_l, height - 28), "ds decreases to the right", fill=(20, 20, 20))
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    rows: list[dict[str, object]] = []
    for metric_name, metric in METRICS.items():
        for ds in DS_VALUES:
            row = locate_fold(metric_name, metric, ds)
            rows.append(row)
            print(
                f"{metric_name} ds={ds:g}: status={row.get('status')} "
                f"R_fold={float(row.get('R_fold_rg', np.nan)):.9g} "
                f"dpx_ds={float(row.get('dpx_ds', np.nan)):.3e} "
                f"sB={float(row.get('smin_over_smax_B', np.nan)):.3e}",
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
