"""Generate the hardened global slim/wind audit requested for Sprint A."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imri_qpe.constants import G
from imri_qpe.layer1_hill_flow import hill_radius
from imri_qpe.layer2_scurve import ThermalEquilibriumParams, compute_scurve
from imri_qpe.layer3_minidisk_1d import (
    GlobalSlimParams,
    energy_limited_wind,
    energy_residual_metrics,
    entropy_gradient_consistency_error,
    entropy_gradient_log_formula,
    entropy_temperature_gradient,
    evaluate_global_slim_profile,
    make_log_grid,
    normalized_mdot_continuity_residual,
    pointwise_energy_residual,
    q_advective,
    q_available,
    relax_temperature_energy_balance,
    wind_energy_per_mass,
    xi_eff,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.units import g_per_s_to_msun_per_year


ROOT = Path(__file__).resolve().parents[1]
FIGURE_OUTPUT = ROOT / "outputs" / "figures" / "global_slim_audit_hardened.png"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "global_slim_audit_hardened.md"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def centered_log_grid(R_first: float, R_last: float, n: int):
    ratio = (R_last / R_first) ** (1.0 / (n - 1))
    return make_log_grid(R_first / math.sqrt(ratio), R_last * math.sqrt(ratio), n)


def map_linear(value: float, lo: float, hi: float, pixel_lo: float, pixel_hi: float) -> float:
    if hi == lo:
        return 0.5 * (pixel_lo + pixel_hi)
    return pixel_lo + (value - lo) / (hi - lo) * (pixel_hi - pixel_lo)


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=(35, 35, 35), anchor=None) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_axes(draw, box, x_ticks, y_ticks, x_min, x_max, y_min, y_max, font_tick, axis_color, grid_color):
    left, top, right, bottom = box
    draw.rectangle(box, outline=axis_color, width=3)
    for tick in x_ticks:
        x = map_linear(tick, x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=grid_color, width=1)
        draw.line((x, bottom, x, bottom + 8), fill=axis_color, width=2)
        draw_text(draw, (x, bottom + 18), f"{tick:.2g}", font_tick, axis_color, "ma")
    for tick in y_ticks:
        y = map_linear(tick, y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=grid_color, width=1)
        draw.line((left - 8, y, left, y), fill=axis_color, width=2)
        draw_text(draw, (left - 16, y), f"{tick:g}", font_tick, axis_color, "rm")


def line_points(x, y, x_min, x_max, y_min, y_max, box):
    left, top, right, bottom = box
    points = []
    for xi, yi in zip(x, y):
        if not np.isfinite(xi) or not np.isfinite(yi):
            continue
        yi = min(max(float(yi), y_min), y_max)
        points.append((map_linear(float(xi), x_min, x_max, left, right), map_linear(yi, y_min, y_max, bottom, top)))
    return points


def draw_line(draw, x, y, x_min, x_max, y_min, y_max, box, color, width):
    points = line_points(x, y, x_min, x_max, y_min, y_max, box)
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_reference(draw, box, value, y_min, y_max, color, width=2):
    if y_min <= value <= y_max:
        y = map_linear(value, y_min, y_max, box[3], box[1])
        draw.line((box[0], y, box[2], y), fill=color, width=width)


def interpolated_hot_branch_profile(grid, params: FiducialParams, target_mdot_msun_yr: float):
    thermal_params = ThermalEquilibriumParams(
        alpha=0.01,
        mu_stress=0.0,
        advective_entropy_gradient=0.3,
    )
    sigma_scan = np.geomspace(1.0e3, 1.0e8, 80)
    Sigma = []
    T = []
    local_mdot = []
    used_interpolation = []

    for R in grid.centers:
        result = compute_scurve(
            R,
            sigma_scan,
            params.M2_g,
            thermal_params,
            T_bounds=(1.0e3, 1.0e10),
            n_T_brackets=180,
        )
        mdot = g_per_s_to_msun_per_year(result.mdot)
        hot = result.stable & (result.T_c >= 1.5e6) & (mdot > 0.0)
        if not np.any(hot):
            hot = result.stable & (mdot > 0.0)
        idx = np.where(hot)[0]
        order = np.argsort(mdot[idx])
        idx = idx[order]
        log_mdot = np.log(mdot[idx])
        keep = np.r_[True, np.diff(log_mdot) > 1.0e-5]
        idx = idx[keep]
        log_mdot = log_mdot[keep]
        target = math.log(target_mdot_msun_yr)

        if len(idx) > 1 and float(np.min(log_mdot)) <= target <= float(np.max(log_mdot)):
            log_sigma = np.interp(target, log_mdot, np.log(result.Sigma[idx]))
            log_T = np.interp(target, log_mdot, np.log(result.T_c[idx]))
            log_local_mdot = target
            used_interpolation.append(True)
        else:
            nearest = idx[int(np.argmin(np.abs(log_mdot - target)))]
            log_sigma = math.log(float(result.Sigma[nearest]))
            log_T = math.log(float(result.T_c[nearest]))
            log_local_mdot = math.log(float(mdot[nearest]))
            used_interpolation.append(False)

        Sigma.append(math.exp(log_sigma))
        T.append(math.exp(log_T))
        local_mdot.append(math.exp(log_local_mdot))

    return np.asarray(Sigma), np.asarray(T), np.asarray(local_mdot), np.asarray(used_interpolation, dtype=bool)


def interior_slice(n: int, boundary_exclude: int):
    if boundary_exclude == 0 or 2 * boundary_exclude >= n:
        return slice(None)
    return slice(boundary_exclude, -boundary_exclude)


def integrated_fraction(numerator, denominator, area) -> float:
    den = float(np.sum(denominator * area))
    if den == 0.0:
        return float("nan")
    return float(np.sum(numerator * area) / den)


def profile_with_stencil(profile, params: GlobalSlimParams, gradient_method: str, boundary_exclude: int):
    first_law = entropy_temperature_gradient(
        profile.R,
        profile.rho,
        profile.T,
        P=profile.P,
        e=profile.e,
        gradient_method=gradient_method,
    )
    log_formula = entropy_gradient_log_formula(
        profile.R,
        profile.rho,
        profile.T,
        mu_mol=params.mu_mol,
        gamma_gas=params.gamma_gas,
        gradient_method=gradient_method,
    )
    consistency = entropy_gradient_consistency_error(first_law, log_formula)
    xi = np.asarray(xi_eff(profile.R, profile.rho, profile.P, first_law), dtype=float)
    Q_adv = np.asarray(q_advective(profile.Sigma, profile.v_R, first_law), dtype=float)
    v_esc = np.sqrt(2.0 * G * params.M2_g / profile.R)
    E_w = wind_energy_per_mass(params.M2_g, profile.R, v_inf=params.v_inf_factor * v_esc)
    Q_avail = q_available(profile.Q_visc, Q_adv=Q_adv)
    Q_wind, Q_rad, dotSigma_w = energy_limited_wind(Q_avail, profile.Q_edd, E_w, params.epsilon_wind)
    Q_wind = np.asarray(Q_wind, dtype=float)
    Q_rad = np.asarray(Q_rad, dtype=float)
    dotSigma_w = np.asarray(dotSigma_w, dtype=float)
    metrics = energy_residual_metrics(
        profile.area,
        profile.Q_visc,
        Q_rad,
        Q_adv,
        Q_wind,
        R=profile.R,
        boundary_exclude=boundary_exclude,
    )
    point_residual = pointwise_energy_residual(profile.Q_visc, Q_rad, Q_adv, Q_wind)
    continuity = normalized_mdot_continuity_residual(profile.R, profile.Mdot, dotSigma_w=dotSigma_w)
    idx = interior_slice(len(profile.R), boundary_exclude)
    return {
        "TdsdR_first_law": first_law,
        "TdsdR_log": log_formula,
        "consistency": consistency,
        "xi_eff": xi,
        "Q_adv": Q_adv,
        "Q_rad": Q_rad,
        "Q_wind": Q_wind,
        "dotSigma_w": dotSigma_w,
        "metrics": metrics,
        "point_residual": point_residual,
        "continuity": continuity,
        "median_xi_interior": float(np.nanmedian(xi[idx])),
        "max_xi_interior": float(np.nanmax(xi[idx])),
        "integrated_Qadv_Qvisc": integrated_fraction(Q_adv, profile.Q_visc, profile.area),
        "median_entropy_error_interior": float(np.nanmedian(consistency[idx])),
        "max_entropy_error_interior": float(np.nanmax(consistency[idx])),
        "max_continuity_interior": float(np.nanmax(np.abs(continuity[idx]))),
    }


def run_case(n: int, params: FiducialParams, global_params: GlobalSlimParams, target_mdot_msun_yr: float):
    R_H = hill_radius(params.a_cm, params.q)
    grid = centered_log_grid(0.06 * R_H, params.tidal_truncation_fraction * R_H, n)
    Sigma, T_hot, local_mdot, interpolated = interpolated_hot_branch_profile(grid, params, target_mdot_msun_yr)
    profile = evaluate_global_slim_profile(grid, Sigma, T_hot, global_params)
    boundary_exclude = min(3, max(1, n // 8))
    variants = {
        "limited": profile_with_stencil(profile, global_params, "limited", boundary_exclude),
        "centered": profile_with_stencil(profile, global_params, "centered", boundary_exclude),
    }
    return {
        "n": n,
        "grid": grid,
        "profile": profile,
        "variants": variants,
        "local_mdot": local_mdot,
        "interpolated": interpolated,
        "boundary_exclude": boundary_exclude,
        "R_H": R_H,
    }


def format_float(value: float, precision: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-2 or abs(value) >= 1.0e3:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}g}"


def write_markdown(cases, relaxed_summary, target_mdot_msun_yr: float) -> None:
    lines = [
        "# Hardened Global Slim Audit",
        "",
        "This table is generated by `scripts/plot_global_slim_audit_hardened.py`.",
        "",
        "The audit evaluates the stitched local hot-branch candidate with two entropy-gradient stencils.",
        "The headline residual is the L1 residual; it cannot be hidden by signed cancellation.",
        "",
        "## Resolution And Stencil Summary",
        "",
        "| N | stencil | signed | L1 | L2 | max interior | max at boundary? | median xi_eff | max xi_eff | int Qadv/Qvisc | median entropy err | max entropy err | max continuity err |",
        "|---:|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        for stencil in ("limited", "centered"):
            result = case["variants"][stencil]
            metrics = result["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(case["n"]),
                        stencil,
                        format_float(metrics.signed),
                        format_float(metrics.L1),
                        format_float(metrics.L2),
                        format_float(metrics.max_abs_interior),
                        "yes" if metrics.max_abs_is_boundary else "no",
                        format_float(result["median_xi_interior"]),
                        format_float(result["max_xi_interior"]),
                        format_float(result["integrated_Qadv_Qvisc"]),
                        format_float(result["median_entropy_error_interior"]),
                        format_float(result["max_entropy_error_interior"]),
                        format_float(result["max_continuity_interior"]),
                    ]
                )
                + " |"
            )

    finest = cases[-1]["variants"]["limited"]
    lines.extend(
        [
            "",
            "## Fixed-Sigma Relaxation Cross-Check",
            "",
            "| quantity | value |",
            "|---|---:|",
        ]
    )
    for key, value in relaxed_summary.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The local-root candidate remains energetically inconsistent in an L1 sense; the signed residual alone is insufficient.",
            "- Limited and centered stencils give the same qualitative result, so the failure is not caused by a single limiter choice.",
            "- The independent entropy-gradient formula agrees best in the smooth interior and exposes larger disagreement near boundary/slope-change regions.",
            "- The `Mdot` continuity residual is large because this candidate was stitched from local roots rather than solved from global mass conservation.",
            "- Sprint B should therefore build the isolated constant-`Mdot` no-wind solver rather than add more wind freedom.",
            "",
            f"Target local hot-branch rate: `{target_mdot_msun_yr:.5g} Msun/yr`.",
        ]
    )
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_hardened_figure(cases, target_mdot_msun_yr: float) -> None:
    scale = 2
    width, height = 1720 * scale, 1240 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_title = load_font(38 * scale, bold=True)
    font_sub = load_font(21 * scale)
    font_panel = load_font(23 * scale, bold=True)
    font_axis = load_font(19 * scale, bold=True)
    font_tick = load_font(16 * scale)
    font_note = load_font(18 * scale)
    font_note_bold = load_font(18 * scale, bold=True)

    axis_color = (42, 46, 52)
    grid_color = (224, 228, 232)
    ref_color = (110, 116, 124)
    colors = {12: (180, 73, 63), 24: (40, 116, 168), 48: (68, 140, 86)}
    centered_color = (112, 86, 150)

    draw_text(draw, (90 * scale, 44 * scale), "Hardened global slim audit", font_title, axis_color)
    draw_text(
        draw,
        (90 * scale, 94 * scale),
        "Residual norms, entropy-formula cross-checks, stencil comparison, and boundary-aware diagnostics.",
        font_sub,
        (78, 82, 88),
    )

    left = 125 * scale
    mid = 825 * scale
    top = 175 * scale
    row_gap = 112 * scale
    panel_w = 560 * scale
    panel_h = 330 * scale
    boxes = [
        (left, top, left + panel_w, top + panel_h),
        (mid, top, mid + panel_w, top + panel_h),
        (left, top + panel_h + row_gap, left + panel_w, top + 2 * panel_h + row_gap),
        (mid, top + panel_h + row_gap, mid + panel_w, top + 2 * panel_h + row_gap),
    ]
    x_ticks = [0.06, 0.1, 0.2, 0.3, 0.5]
    x_min = float(np.min(cases[-1]["grid"].centers / cases[-1]["R_H"]))
    x_max = float(np.max(cases[-1]["grid"].centers / cases[-1]["R_H"]))

    # Panel 1: xi convergence.
    box = boxes[0]
    draw_axes(draw, box, x_ticks, [-2, 0, 0.3, 2, 4, 6, 8], x_min, x_max, -2, 8, font_tick, axis_color, grid_color)
    draw_reference(draw, box, 0.3, -2, 8, ref_color, width=2 * scale)
    for case in cases:
        x = case["grid"].centers / case["R_H"]
        draw_line(draw, x, case["variants"]["limited"]["xi_eff"], x_min, x_max, -2, 8, box, colors[case["n"]], 4 * scale)
    draw_text(draw, (box[0], box[1] - 42 * scale), "xi_eff convergence", font_panel, axis_color)

    # Panel 2: pointwise energy residual.
    finest = cases[-1]
    x = finest["grid"].centers / finest["R_H"]
    box = boxes[1]
    draw_axes(draw, box, x_ticks, [-1, -0.75, -0.5, -0.25, 0], x_min, x_max, -1, 0, font_tick, axis_color, grid_color)
    draw_reference(draw, box, 0.0, -1, 0, ref_color, width=2 * scale)
    draw_line(draw, x, finest["variants"]["limited"]["point_residual"], x_min, x_max, -1, 0, box, colors[48], 4 * scale)
    draw_line(draw, x, finest["variants"]["centered"]["point_residual"], x_min, x_max, -1, 0, box, centered_color, 4 * scale)
    draw_text(draw, (box[0], box[1] - 42 * scale), "Pointwise energy residual", font_panel, axis_color)

    # Panel 3: entropy consistency.
    box = boxes[2]
    draw_axes(draw, box, x_ticks, [-8, -6, -4, -2, 0], x_min, x_max, -8, 0, font_tick, axis_color, grid_color)
    for stencil, color in [("limited", colors[48]), ("centered", centered_color)]:
        err = np.log10(np.maximum(finest["variants"][stencil]["consistency"], 1.0e-12))
        draw_line(draw, x, err, x_min, x_max, -8, 0, box, color, 4 * scale)
    draw_text(draw, (box[0], box[1] - 42 * scale), "Entropy formula disagreement", font_panel, axis_color)

    # Panel 4: L1 convergence.
    box = boxes[3]
    draw_axes(draw, box, [12, 24, 48], [0, 2, 4, 6, 8], 10, 50, 0, 8, font_tick, axis_color, grid_color)
    for stencil, color in [("limited", colors[48]), ("centered", centered_color)]:
        ns = np.asarray([case["n"] for case in cases], dtype=float)
        L1 = np.asarray([case["variants"][stencil]["metrics"].L1 for case in cases], dtype=float)
        draw_line(draw, ns, L1, 10, 50, 0, 8, box, color, 5 * scale)
    draw_text(draw, (box[0], box[1] - 42 * scale), "L1 residual vs resolution", font_panel, axis_color)

    for box, ylabel in [
        (boxes[0], "xi_eff"),
        (boxes[1], "r_E"),
        (boxes[2], "log10 error"),
        (boxes[3], "L1 residual"),
    ]:
        draw_text(draw, ((box[0] + box[2]) / 2, box[3] + 50 * scale), "R/R_H" if box != boxes[3] else "N_R", font_axis, axis_color, "ma")
        layer = Image.new("RGBA", (260 * scale, 42 * scale), (255, 255, 255, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.text((0, 0), ylabel, font=font_axis, fill=axis_color)
        rotated = layer.rotate(90, expand=True)
        image.paste(rotated, (box[0] - 90 * scale, int((box[1] + box[3]) / 2 - rotated.height / 2)), rotated)

    legend_x = 1420 * scale
    legend_y = 188 * scale
    draw_text(draw, (legend_x, legend_y), "Curves", font_note_bold, axis_color)
    legend_y += 38 * scale
    for n in [12, 24, 48]:
        draw.line((legend_x, legend_y + 12 * scale, legend_x + 58 * scale, legend_y + 12 * scale), fill=colors[n], width=6 * scale)
        draw_text(draw, (legend_x + 75 * scale, legend_y), f"limited N={n}", font_note, axis_color)
        legend_y += 35 * scale
    draw.line((legend_x, legend_y + 12 * scale, legend_x + 58 * scale, legend_y + 12 * scale), fill=centered_color, width=6 * scale)
    draw_text(draw, (legend_x + 75 * scale, legend_y), "centered N=48", font_note, axis_color)

    legend_y += 58 * scale
    draw_text(draw, (legend_x, legend_y), "Finest-grid facts", font_note_bold, axis_color)
    legend_y += 34 * scale
    limited = finest["variants"]["limited"]
    centered = finest["variants"]["centered"]
    facts = [
        f"limited L1 = {limited['metrics'].L1:.2g}",
        f"centered L1 = {centered['metrics'].L1:.2g}",
        f"limited int Qadv/Qvisc = {limited['integrated_Qadv_Qvisc']:.2g}",
        f"centered int Qadv/Qvisc = {centered['integrated_Qadv_Qvisc']:.2g}",
        f"median xi = {limited['median_xi_interior']:.2g}",
        f"max continuity err = {limited['max_continuity_interior']:.2g}",
        f"target Mdot = {target_mdot_msun_yr:.4g}",
    ]
    for fact in facts:
        draw_text(draw, (legend_x, legend_y), fact, font_note, (78, 82, 88))
        legend_y += 29 * scale
    legend_y += 26 * scale
    draw_text(draw, (legend_x, legend_y), "Conclusion", font_note_bold, axis_color)
    legend_y += 34 * scale
    for line in [
        "The failure persists",
        "under resolution and",
        "stencil checks. It is",
        "not just signed",
        "cancellation or one",
        "boundary cell.",
    ]:
        draw_text(draw, (legend_x, legend_y), line, font_note, (78, 82, 88))
        legend_y += 29 * scale

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    params = FiducialParams()
    target_mdot_msun_yr = 0.02445
    global_params = GlobalSlimParams(params.M2_g, alpha=0.01, epsilon_wind=0.3, zero_inner_torque=False)

    cases = [run_case(n, params, global_params, target_mdot_msun_yr) for n in (12, 24, 48)]
    baseline = cases[1]
    relaxation = relax_temperature_energy_balance(
        baseline["grid"],
        baseline["profile"].Sigma,
        baseline["profile"].T,
        global_params,
        max_iter=120,
        tol=1.0e-3,
        damping=0.25,
        max_log_step=0.2,
    )
    relaxed_variant = profile_with_stencil(
        relaxation.profile,
        global_params,
        "limited",
        baseline["boundary_exclude"],
    )
    relaxed_summary = {
        "converged": str(relaxation.converged),
        "iterations": str(relaxation.iterations),
        "signed residual": format_float(relaxed_variant["metrics"].signed),
        "L1 residual": format_float(relaxed_variant["metrics"].L1),
        "median xi_eff": format_float(relaxed_variant["median_xi_interior"]),
        "int Qadv/Qvisc": format_float(relaxed_variant["integrated_Qadv_Qvisc"]),
        "Mdot range [Msun/yr]": (
            f"{format_float(float(np.min(g_per_s_to_msun_per_year(relaxation.profile.Mdot))))} to "
            f"{format_float(float(np.max(g_per_s_to_msun_per_year(relaxation.profile.Mdot))))}"
        ),
        "inward cells": f"{100.0 * np.mean(relaxation.profile.Mdot > 0.0):.0f}%",
        "max H/R": format_float(float(np.nanmax(relaxation.profile.H_over_R))),
    }

    write_markdown(cases, relaxed_summary, target_mdot_msun_yr)
    draw_hardened_figure(cases, target_mdot_msun_yr)

    print(f"wrote {FIGURE_OUTPUT}")
    print(f"wrote {TABLE_OUTPUT}")
    for case in cases:
        limited = case["variants"]["limited"]
        centered = case["variants"]["centered"]
        print(
            f"N={case['n']}: limited L1={limited['metrics'].L1:.4g}, "
            f"centered L1={centered['metrics'].L1:.4g}, "
            f"limited int Qadv/Qvisc={limited['integrated_Qadv_Qvisc']:.4g}, "
            f"median xi={limited['median_xi_interior']:.4g}"
        )
    print("relaxed summary")
    for key, value in relaxed_summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
