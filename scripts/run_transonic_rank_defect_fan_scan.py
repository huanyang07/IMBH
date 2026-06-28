"""Fan scan of outgoing phase-space branches from a B-rank-defect point."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    algebraic_state,
    extended_phase_space_matrix,
    phase_space_null_tangent,
    sonic_diagnostics,
)

from run_transonic_barrier_critical_probe import load_seed_rows, solve_probe
from run_transonic_desingularized_barrier_flow import json_safe
from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_sonic_reverse_fit import load_context


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_RANK_DEFECT_FAN_TABLE",
    "outputs/tables/transonic_rank_defect_fan_scan.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_name(TABLE_OUTPUT.stem + ".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_RANK_DEFECT_FAN_FIGURE",
    "outputs/figures/transonic_rank_defect_fan_scan.png",
)

CANDIDATE_MODE = os.environ.get("IMBH_RANK_DEFECT_FAN_CANDIDATE", "free_lambda")
NUM_ALPHA = int(os.environ.get("IMBH_RANK_DEFECT_FAN_NUM_ALPHA", "73"))
EPS0 = float(os.environ.get("IMBH_RANK_DEFECT_FAN_EPS0", "1e-5"))
DS = float(os.environ.get("IMBH_RANK_DEFECT_FAN_DS", "2e-3"))
S_MAX = float(os.environ.get("IMBH_RANK_DEFECT_FAN_S_MAX", "6.0"))
MAX_STEPS = int(os.environ.get("IMBH_RANK_DEFECT_FAN_MAX_STEPS", "4000"))
TARGET_R_RG = float(os.environ.get("IMBH_RANK_DEFECT_FAN_TARGET_R_RG", "8.0"))
PX_MIN = float(os.environ.get("IMBH_RANK_DEFECT_FAN_PX_MIN", "1e-6"))
PHYSICAL_HR_MAX = float(os.environ.get("IMBH_RANK_DEFECT_FAN_HR_MAX", "0.5"))
PHYSICAL_OMEGA_MAX = float(os.environ.get("IMBH_RANK_DEFECT_FAN_OMEGA_MAX", "2.0"))


def sci(value: object, digits: int = 3) -> str:
    number = float(value)
    return f"{number:.{digits}e}" if np.isfinite(number) else "nan"


def fixed(value: object, digits: int) -> str:
    number = float(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "nan"


def load_rank_defect_candidate(ctx) -> dict[str, object]:
    seeds = [
        seed
        for seed in load_seed_rows()
        if seed.get("case") == "free_reverse_fit" and int(seed.get("branch", -1)) == 1
    ]
    if not seeds:
        raise RuntimeError("missing free_reverse_fit branch=1 seed")
    free_lambda = CANDIDATE_MODE == "free_lambda"
    if CANDIDATE_MODE not in {"free_lambda", "fixed_lambda"}:
        raise ValueError("IMBH_RANK_DEFECT_FAN_CANDIDATE must be free_lambda or fixed_lambda")
    row = solve_probe(ctx, seeds[0], free_lambda=free_lambda)
    return {
        "label": f"{CANDIDATE_MODE}_rank_defect",
        "R_rg": float(row["R_rg"]),
        "logR": float(np.log(float(row["R_rg"]) * ctx.params.r_g)),
        "logu": float(row["logu"]),
        "logT": float(row["logT"]),
        "lambda0": float(row["lambda0"]),
        "critK": float(row["critK"]),
    }


def null_basis(logR: float, y: np.ndarray, lambda0: float, params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, _A, _c = extended_phase_space_matrix(logR, y, lambda0, params)
    _U, singular_values, Vt = np.linalg.svd(B, full_matrices=True)
    basis = np.asarray(Vt[-2:, :].T, dtype=float)
    return basis, np.asarray(singular_values, dtype=float), B


def q_metrics(logR: float, y: np.ndarray, p: np.ndarray, lambda0: float, params) -> dict[str, float]:
    if abs(float(p[0])) <= PX_MIN:
        return {"qadv_qvisc": np.nan, "qrad_qvisc": np.nan, "qvisc_sign": np.nan, "g_abs": np.nan}
    g = np.asarray(p[1:], dtype=float) / float(p[0])
    try:
        q_visc, q_rad, q_adv, _energy = _heating_terms_from_gradient(logR, y, g, lambda0, params)
        return {
            "qadv_qvisc": float(q_adv / (q_visc + 1.0e-300)),
            "qrad_qvisc": float(q_rad / (q_visc + 1.0e-300)),
            "qvisc_sign": float(np.sign(q_visc)),
            "g_abs": float(np.max(np.abs(g))),
        }
    except Exception:
        return {"qadv_qvisc": np.nan, "qrad_qvisc": np.nan, "qvisc_sign": np.nan, "g_abs": np.nan}


def local_physical_metrics(logR: float, y: np.ndarray, p: np.ndarray, lambda0: float, params) -> dict[str, float]:
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    out = {
        "H_R": float(state.H_over_R),
        "Omega_over_K": float(state.Omega / state.Omega_K),
    }
    out.update(q_metrics(logR, y, p, lambda0, params))
    return out


def tangent_at(z: np.ndarray, lambda0: float, params, previous: np.ndarray) -> np.ndarray:
    return phase_space_null_tangent(float(z[0]), z[1:], lambda0, params, previous=previous).tangent


def heun_step(z: np.ndarray, lambda0: float, params, ds: float, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = tangent_at(z, lambda0, params, previous=previous)
    z_pred = z + ds * p0
    p1 = tangent_at(z_pred, lambda0, params, previous=p0)
    z_new = z + 0.5 * ds * (p0 + p1)
    p_new = tangent_at(z_new, lambda0, params, previous=p1)
    return z_new, p_new


def classify(summary: dict[str, object]) -> str:
    if float(summary.get("p_x_rank", np.nan)) <= PX_MIN:
        return "rank_inward"
    if float(summary.get("p_x_initial", np.nan)) <= PX_MIN:
        return "immediate_turnback"
    if bool(summary.get("target_reached", False)):
        return "target_reached"
    if bool(summary.get("step_failed", False)):
        return "step_failed"
    if float(summary.get("max_H_R", np.nan)) > PHYSICAL_HR_MAX:
        return "physical_thickness_breakdown"
    if float(summary.get("max_abs_Omega_over_K", np.nan)) > PHYSICAL_OMEGA_MAX:
        return "physical_rotation_breakdown"
    R_start = float(summary.get("R_start_rg", np.nan))
    R_max = float(summary.get("R_max_rg", np.nan))
    R_end = float(summary.get("R_end_rg", np.nan))
    if np.isfinite(R_start) and np.isfinite(R_max) and R_max > R_start + 0.25:
        if np.isfinite(R_end) and R_end < R_max - 1.0e-3:
            return "outward_then_turns"
        return "outward_progress"
    return "stagnant_or_inward"


def integrate_direction(
    alpha: float,
    p_rank: np.ndarray,
    candidate: dict[str, object],
    ctx,
) -> dict[str, object]:
    logR_b = float(candidate["logR"])
    y_b = np.array([float(candidate["logu"]), float(candidate["logT"])], dtype=float)
    lambda0 = float(candidate["lambda0"])
    z0 = np.array([logR_b, y_b[0], y_b[1]], dtype=float) + EPS0 * p_rank
    p = phase_space_null_tangent(float(z0[0]), z0[1:], lambda0, ctx.params.physics, previous=p_rank).tangent
    start_physical = local_physical_metrics(float(z0[0]), z0[1:], p, lambda0, ctx.params.physics)
    rows_R = [float(np.exp(z0[0]) / ctx.params.r_g)]
    px_values = [float(p[0])]
    smin_B = []
    H_R = [float(start_physical["H_R"])]
    omega = [float(start_physical["Omega_over_K"])]
    qvisc_sign = [float(start_physical["qvisc_sign"])]
    z = z0.copy()
    s_value = 0.0
    message = "completed s_max"
    step_failed = False
    target_reached = False
    for _step in range(MAX_STEPS):
        if s_value >= S_MAX:
            break
        try:
            z, p = heun_step(z, lambda0, ctx.params.physics, DS, p)
        except Exception as exc:
            message = str(exc)
            step_failed = True
            break
        s_value += DS
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(p)):
            message = "non-finite z or p"
            step_failed = True
            break
        R_rg = float(np.exp(z[0]) / ctx.params.r_g)
        rows_R.append(R_rg)
        px_values.append(float(p[0]))
        try:
            tangent_diag = phase_space_null_tangent(float(z[0]), z[1:], lambda0, ctx.params.physics, previous=p)
            state_metrics = local_physical_metrics(float(z[0]), z[1:], p, lambda0, ctx.params.physics)
            smin_B.append(float(tangent_diag.smin_over_smax_B))
            H_R.append(float(state_metrics["H_R"]))
            omega.append(float(state_metrics["Omega_over_K"]))
            qvisc_sign.append(float(state_metrics["qvisc_sign"]))
        except Exception:
            pass
        if R_rg >= TARGET_R_RG:
            target_reached = True
            message = f"reached R >= {TARGET_R_RG:g} rg"
            break
        if np.isfinite(H_R[-1]) and H_R[-1] > PHYSICAL_HR_MAX:
            message = "H/R exceeded physical threshold"
            break
        if np.isfinite(omega[-1]) and abs(omega[-1]) > PHYSICAL_OMEGA_MAX:
            message = "Omega/OmegaK exceeded physical threshold"
            break
    R_values = np.asarray(rows_R, dtype=float)
    px_array = np.asarray(px_values, dtype=float)
    summary: dict[str, object] = {
        "candidate": candidate["label"],
        "alpha": float(alpha),
        "alpha_deg": float(alpha * 180.0 / np.pi),
        "p_x_rank": float(p_rank[0]),
        "p_u_rank": float(p_rank[1]),
        "p_T_rank": float(p_rank[2]),
        "p_x_initial": float(px_values[0]),
        "R_start_rg": float(R_values[0]),
        "R_end_rg": float(R_values[-1]),
        "R_max_rg": float(np.nanmax(R_values)),
        "R_min_rg": float(np.nanmin(R_values)),
        "delta_R_max_rg": float(np.nanmax(R_values) - R_values[0]),
        "s_end": float(s_value),
        "n_steps": int(len(R_values) - 1),
        "target_reached": bool(target_reached),
        "step_failed": bool(step_failed),
        "message": message,
        "min_p_x": float(np.nanmin(px_array)),
        "max_p_x": float(np.nanmax(px_array)),
        "p_x_sign_changes": int(np.count_nonzero(np.diff(np.sign(px_array[np.nonzero(px_array)]))) if np.any(px_array != 0.0) else 0),
        "min_smin_over_smax_B": float(np.nanmin(smin_B)) if smin_B else np.nan,
        "max_H_R": float(np.nanmax(H_R)) if H_R else np.nan,
        "max_abs_Omega_over_K": float(np.nanmax(np.abs(omega))) if omega else np.nan,
        "min_qvisc_sign": float(np.nanmin(qvisc_sign)) if qvisc_sign else np.nan,
        "initial_H_R": float(start_physical["H_R"]),
        "initial_Omega_over_K": float(start_physical["Omega_over_K"]),
        "initial_qadv_qvisc": float(start_physical["qadv_qvisc"]),
        "initial_qrad_qvisc": float(start_physical["qrad_qvisc"]),
        "initial_qvisc_sign": float(start_physical["qvisc_sign"]),
        "initial_g_abs": float(start_physical["g_abs"]),
    }
    summary["classification"] = classify(summary)
    return summary


def scan_rows(candidate: dict[str, object], ctx) -> tuple[dict[str, object], list[dict[str, object]]]:
    logR_b = float(candidate["logR"])
    y_b = np.array([float(candidate["logu"]), float(candidate["logT"])], dtype=float)
    lambda0 = float(candidate["lambda0"])
    basis, singular_values, B = null_basis(logR_b, y_b, lambda0, ctx.params.physics)
    sonic = sonic_diagnostics(logR_b, y_b, lambda0, ctx.params.physics)
    state = algebraic_state(logR_b, float(y_b[0]), float(y_b[1]), lambda0, ctx.params.physics)
    center = {
        **candidate,
        "smin_over_smax_B": float(singular_values[-1] / (singular_values[0] + 1.0e-300)),
        "B_rank": int(np.linalg.matrix_rank(B, tol=1.0e-10)),
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "H_R": float(state.H_over_R),
        "Omega_over_K": float(state.Omega / state.Omega_K),
    }
    rows: list[dict[str, object]] = []
    for idx, alpha in enumerate(np.linspace(0.0, 2.0 * np.pi, NUM_ALPHA, endpoint=False)):
        p = np.cos(alpha) * basis[:, 0] + np.sin(alpha) * basis[:, 1]
        p = p / (np.linalg.norm(p) + 1.0e-300)
        row = integrate_direction(float(alpha), p, candidate, ctx)
        row["index"] = idx
        rows.append(row)
        if idx % max(1, NUM_ALPHA // 12) == 0:
            print(
                f"alpha={row['alpha_deg']:.1f} class={row['classification']} "
                f"Rmax={row['R_max_rg']:.4g} px0={row['p_x_initial']:.3g}",
                flush=True,
            )
    return center, rows


def write_table(center: dict[str, object], rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    promising = [
        row
        for row in rows
        if row["classification"] in {"target_reached", "outward_progress", "outward_then_turns"}
        and float(row["p_x_rank"]) > PX_MIN
        and float(row["p_x_initial"]) > PX_MIN
    ]
    lines = [
        "# Rank-Defect Fan Scan",
        "",
        "Generated by `scripts/run_transonic_rank_defect_fan_scan.py`.",
        "",
        f"Candidate: `{center['label']}`, `R={fixed(center['R_rg'], 9)} rg`, "
        f"`lambda0={fixed(center['lambda0'], 9)}`, `sB={sci(center['smin_over_smax_B'])}`.",
        "",
        f"Config: `N_alpha={NUM_ALPHA}`, `eps0={EPS0:g}`, `ds={DS:g}`, `s_max={S_MAX:g}`, `target_R={TARGET_R_RG:g} rg`.",
        "",
        f"Promising outward rows: `{len(promising)}`.",
        "",
        "| idx | alpha deg | class | p_x rank | p_x start | R start | R max | R end | dR max | target | min p_x | px flips | min sB | max H/R | max |Omega/K| | qvisc sign | g abs start | steps | message |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {idx} | {alpha} | {classification} | {px_rank} | {px_start} | {R_start} | {R_max} | {R_end} | {dR} | {target} | "
            "{min_px} | {flips} | {sB} | {HR} | {omega} | {qsign} | {gabs} | {steps} | {message} |".format(
                idx=row["index"],
                alpha=fixed(row["alpha_deg"], 3),
                classification=row["classification"],
                px_rank=sci(row["p_x_rank"]),
                px_start=sci(row["p_x_initial"]),
                R_start=fixed(row["R_start_rg"], 6),
                R_max=fixed(row["R_max_rg"], 6),
                R_end=fixed(row["R_end_rg"], 6),
                dR=sci(row["delta_R_max_rg"]),
                target="yes" if row["target_reached"] else "no",
                min_px=sci(row["min_p_x"]),
                flips=row["p_x_sign_changes"],
                sB=sci(row["min_smin_over_smax_B"]),
                HR=sci(row["max_H_R"]),
                omega=fixed(row["max_abs_Omega_over_K"], 6),
                qsign=sci(row["min_qvisc_sign"]),
                gabs=sci(row["initial_g_abs"]),
                steps=row["n_steps"],
                message=str(row["message"]).replace("|", "/")[:100],
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def write_json(center: dict[str, object], rows: list[dict[str, object]]) -> None:
    JSON_OUTPUT.write_text(
        json.dumps(
            {
                "center": {key: json_safe(value) for key, value in center.items()},
                "rows": [{key: json_safe(value) for key, value in row.items()} for row in rows],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"figure skipped: {exc}", flush=True)
        return
    width, height = 1120, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 24), "Rank-defect fan scan", fill=(20, 20, 20))
    panels = [
        ("R_max/rg", "R_max_rg", False),
        ("p_x start", "p_x_initial", False),
        ("min smin/smax B", "min_smin_over_smax_B", True),
        ("initial |g|", "initial_g_abs", True),
    ]
    xs = np.asarray([float(row["alpha_deg"]) for row in rows], dtype=float)
    plot_l, plot_r = 90, width - 50
    panel_h = 125
    colors = {
        "target_reached": (44, 160, 44),
        "outward_progress": (31, 119, 180),
        "outward_then_turns": (148, 103, 189),
        "stagnant_or_inward": (127, 127, 127),
        "rank_inward": (200, 200, 200),
        "immediate_turnback": (180, 180, 180),
        "step_failed": (214, 39, 40),
    }
    for pidx, (title, key, logy) in enumerate(panels):
        top = 65 + 155 * pidx
        bottom = top + panel_h
        draw.rectangle((plot_l, top, plot_r, bottom), outline=(60, 60, 60), width=1)
        values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=float)
        finite = values[np.isfinite(values) & ((values > 0.0) if logy else np.ones_like(values, dtype=bool))]
        if finite.size == 0:
            continue
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
        if logy:
            ymin = max(ymin * 0.6, 1.0e-20)
            ymax = max(ymax * 1.5, ymin * 10.0)
        else:
            pad = max(0.1 * (ymax - ymin), 1.0e-9)
            ymin -= pad
            ymax += pad

        def sx(value: float) -> float:
            return plot_l + (value - xmin) / (xmax - xmin + 1.0e-300) * (plot_r - plot_l)

        def sy(value: float) -> float:
            if logy:
                return bottom - (np.log10(value) - np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin) + 1.0e-300) * (bottom - top)
            return bottom - (value - ymin) / (ymax - ymin + 1.0e-300) * (bottom - top)

        draw.text((plot_l + 8, top + 8), title, fill=(20, 20, 20))
        points = []
        for row in rows:
            x = float(row["alpha_deg"])
            value = float(row.get(key, np.nan))
            if np.isfinite(value) and ((value > 0.0) if logy else True):
                points.append((sx(x), sy(value), colors.get(str(row["classification"]), (0, 0, 0))))
        for xpix, ypix, color in points:
            draw.ellipse((xpix - 3, ypix - 3, xpix + 3, ypix + 3), fill=color)
    draw.text((plot_l, height - 28), "alpha [deg]", fill=(20, 20, 20))
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    ctx = load_context()
    candidate = load_rank_defect_candidate(ctx)
    center, rows = scan_rows(candidate, ctx)
    write_table(center, rows)
    write_json(center, rows)
    write_figure(rows)
    counts = {name: sum(1 for row in rows if row["classification"] == name) for name in sorted({str(row["classification"]) for row in rows})}
    print(f"classification counts: {counts}", flush=True)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {JSON_OUTPUT}", flush=True)
    if FIGURE_OUTPUT.exists():
        print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
