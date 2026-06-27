"""Backward shooting audit for the dynamic sonic-patch branch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient
from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state, local_ode_rhs, sonic_diagnostics
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_dynamic_sonic_patch import CHECKPOINT_DIR as DYNAMIC_CHECKPOINT_DIR, load_row
from run_transonic_two_domain_sonic_refinement_sprint import buffer_inner_grid, make_buffer_params, unpack_buffer


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_sonic_backward_shooting_audit.md"
FIGURE_OUTPUT = ROOT / "outputs" / "figures" / "transonic_sonic_backward_shooting_D_C.png"
SOURCE_CHECKPOINT = DYNAMIC_CHECKPOINT_DIR / os.environ.get("IMBH_BACKWARD_SOURCE", "Nreg64_0p90277664.npz")
BUFFER_INDEX = int(os.environ.get("IMBH_BACKWARD_BUFFER_INDEX", "1"))
X_BELOW_OLD_SONIC = float(os.environ.get("IMBH_BACKWARD_X_BELOW", "0.12"))
MAX_STEP = float(os.environ.get("IMBH_BACKWARD_MAX_STEP", "5e-4"))
RTOL = float(os.environ.get("IMBH_BACKWARD_RTOL", "1e-8"))
ATOL = float(os.environ.get("IMBH_BACKWARD_ATOL", "1e-10"))
DEFAULT_FOCUS_LAMBDAS = (-1.25e-3, -1.5e-3, -1.53e-3, -1.55e-3, -1.57e-3, -1.58e-3, -1.6e-3, -1.75e-3, -2.0e-3)


def parse_float_sequence(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return tuple(float(piece) for piece in raw.replace(":", ",").split(",") if piece.strip())


def tag_float(value: float) -> str:
    return f"{value:+.5g}".replace("+", "p").replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class ShootCase:
    label: str
    dlambda: float = 0.0
    dlogu_b: float = 0.0
    dlogT_b: float = 0.0


def cases() -> tuple[ShootCase, ...]:
    deltas = (1.0e-3, 3.0e-3)
    rows = [ShootCase("base")]
    for delta in deltas:
        rows.append(ShootCase(f"lambda_p{delta:g}", dlambda=delta))
        rows.append(ShootCase(f"lambda_m{delta:g}", dlambda=-delta))
    for delta in parse_float_sequence("IMBH_BACKWARD_FOCUS_LAMBDAS", DEFAULT_FOCUS_LAMBDAS):
        rows.append(ShootCase(f"lambda_focus_{tag_float(delta)}", dlambda=delta))
    for delta in deltas:
        rows.append(ShootCase(f"logu_p{delta:g}", dlogu_b=delta))
        rows.append(ShootCase(f"logu_m{delta:g}", dlogu_b=-delta))
    for delta in deltas:
        rows.append(ShootCase(f"logT_p{delta:g}", dlogT_b=delta))
        rows.append(ShootCase(f"logT_m{delta:g}", dlogT_b=-delta))
    return tuple(rows)


def safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def evaluate_profile(logR: np.ndarray, y: np.ndarray, lambda0: float, params) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for x_value, y_value in zip(logR, y.T):
        try:
            sonic = sonic_diagnostics(float(x_value), y_value, lambda0, params.physics)
            state = algebraic_state(float(x_value), float(y_value[0]), float(y_value[1]), lambda0, params.physics)
            g = local_ode_rhs(float(x_value), y_value, lambda0, params.physics)
            q_visc, q_rad, q_adv, _energy = _heating_terms_from_gradient(float(x_value), y_value, g, lambda0, params.physics)
            qadv_qvisc = q_adv / (q_visc + 1.0e-300)
            qrad_qvisc = q_rad / (q_visc + 1.0e-300)
            qvisc_sign = np.sign(q_visc)
            cond = 1.0 / (sonic.smin_over_smax + 1.0e-300)
            rows.append(
                {
                    "logR": float(x_value),
                    "R_rg": float(np.exp(x_value) / params.r_g),
                    "D": float(sonic.D),
                    "C1": float(sonic.C1),
                    "C2": float(sonic.C2),
                    "K": float(sonic.compatibility),
                    "crit": max(abs(float(sonic.D)), abs(float(sonic.C1)), abs(float(sonic.C2))),
                    "critK": max(abs(float(sonic.D)), abs(float(sonic.C1)), abs(float(sonic.C2)), abs(float(sonic.compatibility))),
                    "smin_over_smax": float(sonic.smin_over_smax),
                    "condA": float(cond),
                    "M_eff": float(sonic.M_eff),
                    "H_R": float(state.H_over_R),
                    "qvisc_sign": float(qvisc_sign),
                    "qadv_qvisc": safe_float(float(qadv_qvisc)),
                    "qrad_qvisc": safe_float(float(qrad_qvisc)),
                }
            )
        except Exception:
            rows.append(
                {
                    "logR": float(x_value),
                    "R_rg": np.nan,
                    "D": np.nan,
                    "C1": np.nan,
                    "C2": np.nan,
                    "K": np.nan,
                    "crit": np.inf,
                    "critK": np.inf,
                    "smin_over_smax": np.nan,
                    "condA": np.nan,
                    "M_eff": np.nan,
                    "H_R": np.nan,
                    "qvisc_sign": np.nan,
                    "qadv_qvisc": np.nan,
                    "qrad_qvisc": np.nan,
                }
            )
    return rows


def shoot_case(case: ShootCase, x_source: np.ndarray, params, logR_i: np.ndarray, logu_i: np.ndarray, logT_i: np.ndarray, logR_son_old: float, lambda0_old: float) -> tuple[dict[str, object], list[dict[str, float]]]:
    if BUFFER_INDEX <= 0 or BUFFER_INDEX >= len(logR_i):
        raise ValueError("BUFFER_INDEX must choose a regular inner buffer node")
    x_b = float(logR_i[BUFFER_INDEX])
    y_b = np.array([logu_i[BUFFER_INDEX] + case.dlogu_b, logT_i[BUFFER_INDEX] + case.dlogT_b], dtype=float)
    lambda0 = float(lambda0_old + case.dlambda)
    x_end = float(logR_son_old - X_BELOW_OLD_SONIC)

    def rhs(x_value: float, y_value: np.ndarray) -> np.ndarray:
        return local_ode_rhs(float(x_value), y_value, lambda0, params.physics)

    success = False
    message = ""
    nfev = 0
    try:
        sol = solve_ivp(
            rhs,
            (x_b, x_end),
            y_b,
            method="Radau",
            max_step=MAX_STEP,
            rtol=RTOL,
            atol=ATOL,
        )
        success = bool(sol.success)
        message = str(sol.message)
        nfev = int(sol.nfev)
        if sol.y.shape[1] == 0:
            raise RuntimeError(message)
        profile = evaluate_profile(np.asarray(sol.t, dtype=float), np.asarray(sol.y, dtype=float), lambda0, params)
    except Exception as exc:
        message = str(exc)
        profile = []

    finite = [row for row in profile if np.isfinite(row["critK"])]
    if finite:
        best = min(finite, key=lambda row: row["critK"])
        min_d = min(finite, key=lambda row: abs(row["D"]))
    else:
        best = {
            "logR": np.nan,
            "R_rg": np.nan,
            "D": np.nan,
            "C1": np.nan,
            "C2": np.nan,
            "K": np.nan,
            "crit": np.inf,
            "critK": np.inf,
            "smin_over_smax": np.nan,
            "condA": np.nan,
            "M_eff": np.nan,
            "H_R": np.nan,
            "qvisc_sign": np.nan,
            "qadv_qvisc": np.nan,
            "qrad_qvisc": np.nan,
        }
        min_d = best

    row = {
        "case": case.label,
        "dlambda": case.dlambda,
        "dlogu_b": case.dlogu_b,
        "dlogT_b": case.dlogT_b,
        "success": success,
        "nfev": nfev,
        "n_points": len(profile),
        "message": message,
        "start_R_rg": float(np.exp(x_b) / params.r_g),
        "end_R_rg": float(np.exp(x_end) / params.r_g),
        "reached_R_rg": float(profile[-1]["R_rg"]) if profile and np.isfinite(profile[-1]["R_rg"]) else np.nan,
        "reached_x_minus_old_xs": float(profile[-1]["logR"] - logR_son_old) if profile and np.isfinite(profile[-1]["logR"]) else np.nan,
        "best_R_rg": best["R_rg"],
        "best_x_minus_old_xs": float(best["logR"] - logR_son_old) if np.isfinite(best["logR"]) else np.nan,
        "best_crit": best["crit"],
        "best_critK": best["critK"],
        "best_D": best["D"],
        "best_C1": best["C1"],
        "best_C2": best["C2"],
        "best_K": best["K"],
        "best_smin_over_smax": best["smin_over_smax"],
        "best_condA": best["condA"],
        "best_M_eff": best["M_eff"],
        "best_H_R": best["H_R"],
        "best_qvisc_sign": best["qvisc_sign"],
        "best_qadv_qvisc": best["qadv_qvisc"],
        "best_qrad_qvisc": best["qrad_qvisc"],
        "min_abs_D": abs(float(min_d["D"])) if np.isfinite(min_d["D"]) else np.nan,
        "minD_R_rg": min_d["R_rg"],
        "minD_C1": min_d["C1"],
        "minD_C2": min_d["C2"],
        "minD_K": min_d["K"],
    }
    return row, profile


def write_table(rows: list[dict[str, object]], source_meta: dict[str, object]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sonic Backward-Shooting Audit",
        "",
        "Generated by `scripts/run_transonic_sonic_backward_shooting_audit.py`.",
        "",
        f"Source checkpoint: `{SOURCE_CHECKPOINT}`.",
        f"Old dynamic branch: `Rson={float(source_meta['Rson_rg']):.8g} rg`, `lambda0={float(source_meta['lambda0']):.8g}`, `delta_s={float(source_meta['delta_s']):g}`.",
        f"Backward integration: buffer index `{BUFFER_INDEX}`, `x_b -> x_s - {X_BELOW_OLD_SONIC:g}`, `max_step={MAX_STEP:g}`.",
        "",
        "| case | dlambda | dlogu_b | dlogT_b | success | points | reached R/rg | reached x-xs | best crit D/C | best crit D/C/K | best R/rg | x-xs | D | C1 | C2 | K | smin/smax | cond(A) | M_eff | H/R | Qvisc sign | Qadv/Qvisc | min |D| | minD R/rg | minD C1 | minD C2 | minD K | nfev | message |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {dlambda} | {dlogu_b} | {dlogT_b} | {success} | {n_points} | {reached_R_rg} | {reached_x_minus_old_xs} | {best_crit} | {best_critK} | "
            "{best_R_rg} | {best_x_minus_old_xs} | {best_D} | {best_C1} | {best_C2} | {best_K} | "
            "{best_smin_over_smax} | {best_condA} | {best_M_eff} | {best_H_R} | {best_qvisc_sign} | {best_qadv_qvisc} | "
            "{min_abs_D} | {minD_R_rg} | {minD_C1} | {minD_C2} | {minD_K} | {nfev} | {message} |".format(
                case=row["case"],
                dlambda=fmt(float(row["dlambda"])),
                dlogu_b=fmt(float(row["dlogu_b"])),
                dlogT_b=fmt(float(row["dlogT_b"])),
                success="yes" if row["success"] else "no",
                n_points=row["n_points"],
                reached_R_rg=fmt(float(row["reached_R_rg"])),
                reached_x_minus_old_xs=fmt(float(row["reached_x_minus_old_xs"])),
                best_crit=fmt(float(row["best_crit"])),
                best_critK=fmt(float(row["best_critK"])),
                best_R_rg=fmt(float(row["best_R_rg"])),
                best_x_minus_old_xs=fmt(float(row["best_x_minus_old_xs"])),
                best_D=fmt(float(row["best_D"])),
                best_C1=fmt(float(row["best_C1"])),
                best_C2=fmt(float(row["best_C2"])),
                best_K=fmt(float(row["best_K"])),
                best_smin_over_smax=fmt(float(row["best_smin_over_smax"])),
                best_condA=fmt(float(row["best_condA"])),
                best_M_eff=fmt(float(row["best_M_eff"])),
                best_H_R=fmt(float(row["best_H_R"])),
                best_qvisc_sign=fmt(float(row["best_qvisc_sign"])),
                best_qadv_qvisc=fmt(float(row["best_qadv_qvisc"])),
                min_abs_D=fmt(float(row["min_abs_D"])),
                minD_R_rg=fmt(float(row["minD_R_rg"])),
                minD_C1=fmt(float(row["minD_C1"])),
                minD_C2=fmt(float(row["minD_C2"])),
                minD_K=fmt(float(row["minD_K"])),
                nfev=row["nfev"],
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def write_figure(profiles: dict[str, list[dict[str, float]]], rows: list[dict[str, object]], logR_son_old: float) -> None:
    finite_rows = [row for row in rows if np.isfinite(float(row["best_critK"]))]
    best_label = str(min(finite_rows, key=lambda row: float(row["best_critK"]))["case"]) if finite_rows else "base"
    labels = ["base"] if best_label == "base" else ["base", best_label]
    success_rows = [row for row in finite_rows if row["success"]]
    if success_rows:
        best_success = str(min(success_rows, key=lambda row: float(row["best_critK"]))["case"])
        if best_success not in labels:
            labels.append(best_success)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; using PIL figure fallback: {exc}", flush=True)
        write_figure_pil(profiles, labels, logR_son_old)
        return

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(labels), 1, figsize=(7.0, 3.0 * len(labels)), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        profile = profiles.get(label, [])
        x = np.array([row["logR"] - logR_son_old for row in profile], dtype=float)
        for key, color in (("D", "black"), ("C1", "#1f77b4"), ("C2", "#ff7f0e"), ("K", "#2ca02c")):
            y = np.array([abs(row[key]) for row in profile], dtype=float)
            ax.semilogy(x, np.maximum(y, 1.0e-16), label=key, color=color)
        ax.axvline(0.0, color="0.5", lw=1.0, ls="--")
        ax.set_ylabel("|diagnostic|")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("logR - old logR_sonic")
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)


def write_figure_pil(profiles: dict[str, list[dict[str, float]]], labels: list[str], logR_son_old: float) -> None:
    from PIL import Image, ImageDraw

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    width = 1200
    panel_h = 430
    left = 90
    right = 40
    top_margin = 55
    bottom_margin = 65
    colors = {"D": (0, 0, 0), "C1": (31, 119, 180), "C2": (255, 127, 14), "K": (44, 160, 44)}
    image = Image.new("RGB", (width, panel_h * len(labels)), "white")
    draw = ImageDraw.Draw(image)

    for panel, label in enumerate(labels):
        y0 = panel * panel_h
        profile = profiles.get(label, [])
        finite_profile = [row for row in profile if np.isfinite(row["logR"])]
        if not finite_profile:
            continue
        x_values = np.array([row["logR"] - logR_son_old for row in finite_profile], dtype=float)
        y_values = []
        for key in colors:
            y_values.extend([abs(row[key]) for row in finite_profile if np.isfinite(row[key]) and abs(row[key]) > 0.0])
        y_min = max(min(y_values), 1.0e-16) if y_values else 1.0e-16
        y_max = max(y_values) if y_values else 1.0
        log_y_min = np.floor(np.log10(y_min))
        log_y_max = np.ceil(np.log10(y_max))
        if log_y_max <= log_y_min:
            log_y_max = log_y_min + 1.0
        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        if abs(x_max - x_min) < 1.0e-12:
            x_min -= 1.0e-3
            x_max += 1.0e-3

        plot_left = left
        plot_right = width - right
        plot_top = y0 + top_margin
        plot_bottom = y0 + panel_h - bottom_margin

        def px(x_value: float) -> int:
            return int(plot_left + (x_value - x_min) / (x_max - x_min) * (plot_right - plot_left))

        def py(y_value: float) -> int:
            ly = np.log10(max(float(y_value), 1.0e-16))
            return int(plot_bottom - (ly - log_y_min) / (log_y_max - log_y_min) * (plot_bottom - plot_top))

        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(0, 0, 0))
        draw.text((plot_left, y0 + 18), label, fill=(0, 0, 0))
        draw.text((plot_left, plot_bottom + 30), "logR - old logR_sonic", fill=(0, 0, 0))
        draw.text((10, plot_top), "|diagnostic|", fill=(0, 0, 0))
        for frac in np.linspace(0.0, 1.0, 6):
            x_tick = x_min + frac * (x_max - x_min)
            x_pix = px(x_tick)
            draw.line((x_pix, plot_bottom, x_pix, plot_bottom + 5), fill=(0, 0, 0))
            draw.text((x_pix - 25, plot_bottom + 8), f"{x_tick:.3g}", fill=(0, 0, 0))
        for exponent in range(int(log_y_min), int(log_y_max) + 1):
            y_pix = py(10.0**exponent)
            draw.line((plot_left - 5, y_pix, plot_left, y_pix), fill=(0, 0, 0))
            draw.text((plot_left - 75, y_pix - 7), f"1e{exponent}", fill=(0, 0, 0))
        if x_min <= 0.0 <= x_max:
            x_zero = px(0.0)
            draw.line((x_zero, plot_top, x_zero, plot_bottom), fill=(130, 130, 130))

        for offset, (key, color) in enumerate(colors.items()):
            points = []
            for row in finite_profile:
                value = abs(row[key])
                if np.isfinite(value) and value > 0.0:
                    points.append((px(row["logR"] - logR_son_old), py(value)))
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            draw.line((plot_right - 140, plot_top + 18 + 20 * offset, plot_right - 112, plot_top + 18 + 20 * offset), fill=color, width=3)
            draw.text((plot_right - 105, plot_top + 10 + 20 * offset), key, fill=color)

    image.save(FIGURE_OUTPUT)


def main() -> None:
    x_source, meta = load_row(SOURCE_CHECKPOINT)
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    params = make_buffer_params(
        fiducial,
        float(meta["ratio"]),
        mdot_edd,
        int(meta["n_regular"]),
        int(meta["n_outer"]),
        float(meta["R_far_rg"]),
        float(meta["delta_s"]),
    )
    logu_i, logT_i, _logu_o, _logT_o, logR_son, lambda0 = unpack_buffer(x_source, params)
    logR_i = buffer_inner_grid(logR_son, params)
    print(
        f"backward shooting source={SOURCE_CHECKPOINT.name} Rson={np.exp(logR_son)/params.r_g:.6g} "
        f"lambda0={lambda0:.6g} buffer_index={BUFFER_INDEX} cases={len(cases())}",
        flush=True,
    )
    rows: list[dict[str, object]] = []
    profiles: dict[str, list[dict[str, float]]] = {}
    for case in cases():
        row, profile = shoot_case(case, x_source, params, logR_i, logu_i, logT_i, logR_son, lambda0)
        rows.append(row)
        profiles[case.label] = profile
        print(
            f"{case.label}: success={row['success']} best={float(row['best_critK']):.3e} "
            f"R={float(row['best_R_rg']):.5g} D={float(row['best_D']):.3e} "
            f"C=({float(row['best_C1']):.3e},{float(row['best_C2']):.3e}) K={float(row['best_K']):.3e}",
            flush=True,
        )
    write_table(rows, meta)
    write_figure(profiles, rows, logR_son)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    if FIGURE_OUTPUT.exists():
        print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
