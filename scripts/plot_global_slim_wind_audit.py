"""Generate a global slim/wind audit figure from real model outputs."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imri_qpe.layer1_hill_flow import hill_radius
from imri_qpe.layer2_scurve import ThermalEquilibriumParams, compute_scurve
from imri_qpe.layer3_minidisk_1d import (
    GlobalSlimParams,
    evaluate_global_slim_profile,
    integrated_energy_residual,
    make_log_grid,
    relax_temperature_energy_balance,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.units import g_per_s_to_msun_per_year


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "figures" / "global_slim_wind_audit.png"


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


def map_linear(value: float, lo: float, hi: float, pixel_lo: float, pixel_hi: float) -> float:
    if hi == lo:
        return 0.5 * (pixel_lo + pixel_hi)
    return pixel_lo + (value - lo) / (hi - lo) * (pixel_hi - pixel_lo)


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=(35, 35, 35), anchor=None) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def line_points(x, y, x_min, x_max, y_min, y_max, box):
    left, top, right, bottom = box
    points = []
    for xi, yi in zip(x, y):
        if not np.isfinite(xi) or not np.isfinite(yi):
            continue
        yi = min(max(float(yi), y_min), y_max)
        points.append(
            (
                map_linear(float(xi), x_min, x_max, left, right),
                map_linear(yi, y_min, y_max, bottom, top),
            )
        )
    return points


def draw_line(draw: ImageDraw.ImageDraw, x, y, x_min, x_max, y_min, y_max, box, color, width: int) -> None:
    points = line_points(x, y, x_min, x_max, y_min, y_max, box)
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_axes(
    draw: ImageDraw.ImageDraw,
    box,
    x_ticks,
    y_ticks,
    x_min,
    x_max,
    y_min,
    y_max,
    font_tick,
    axis_color,
    grid_color,
) -> None:
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


def draw_reference(draw: ImageDraw.ImageDraw, box, value, y_min, y_max, color, width: int = 2) -> None:
    left, top, right, bottom = box
    if y_min <= value <= y_max:
        y = map_linear(value, y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=color, width=width)


def centered_log_grid(R_first: float, R_last: float, n: int):
    ratio = (R_last / R_first) ** (1.0 / (n - 1))
    return make_log_grid(R_first / math.sqrt(ratio), R_last * math.sqrt(ratio), n)


def interpolated_hot_branch_profile(grid, params: FiducialParams, target_mdot_msun_yr: float):
    thermal_params = ThermalEquilibriumParams(
        alpha=0.01,
        mu_stress=0.0,
        advective_entropy_gradient=0.3,
    )
    sigma_scan = np.geomspace(1.0e3, 1.0e8, 100)
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
            n_T_brackets=240,
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


def integrated_fraction(numerator, denominator, area) -> float:
    den = float(np.sum(denominator * area))
    if den == 0.0:
        return float("nan")
    return float(np.sum(numerator * area) / den)


def summarize(label: str, profile, target_mdot_msun_yr: float) -> list[str]:
    mdot = g_per_s_to_msun_per_year(profile.Mdot)
    return [
        f"{label}:",
        f"global residual = {integrated_energy_residual(profile):.2g}",
        f"median xi_eff = {np.nanmedian(profile.xi_eff):.2g}",
        f"int Qadv/Qvisc = {integrated_fraction(profile.Q_adv, profile.Q_visc, profile.area):.2g}",
        f"Mdot range = {np.nanmin(mdot):.3g} to {np.nanmax(mdot):.3g}",
        f"inward cells = {100.0 * np.mean(profile.Mdot > 0.0):.0f}%",
        f"max H/R = {np.nanmax(profile.H_over_R):.2g}",
        f"target local Mdot = {target_mdot_msun_yr:.4g}",
    ]


def main() -> None:
    params = FiducialParams()
    R_H = hill_radius(params.a_cm, params.q)
    target_mdot_msun_yr = 0.02445
    grid = centered_log_grid(0.06 * R_H, params.tidal_truncation_fraction * R_H, 18)
    x = grid.centers / R_H

    Sigma, T_hot, local_mdot, interpolated = interpolated_hot_branch_profile(grid, params, target_mdot_msun_yr)
    global_params = GlobalSlimParams(
        params.M2_g,
        alpha=0.01,
        epsilon_wind=0.3,
        zero_inner_torque=False,
    )
    candidate = evaluate_global_slim_profile(grid, Sigma, T_hot, global_params)
    relaxation = relax_temperature_energy_balance(
        grid,
        Sigma,
        T_hot,
        global_params,
        max_iter=120,
        tol=1.0e-3,
        damping=0.25,
        max_log_step=0.2,
    )
    relaxed = relaxation.profile

    scale = 2
    width, height = 1700 * scale, 1220 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_title = load_font(38 * scale, bold=True)
    font_sub = load_font(21 * scale)
    font_panel = load_font(24 * scale, bold=True)
    font_axis = load_font(20 * scale, bold=True)
    font_tick = load_font(17 * scale)
    font_note = load_font(18 * scale)
    font_note_bold = load_font(18 * scale, bold=True)

    axis_color = (42, 46, 52)
    grid_color = (224, 228, 232)
    candidate_color = (190, 62, 52)
    relaxed_color = (33, 108, 168)
    ref_color = (110, 116, 124)
    warning_color = (150, 79, 34)

    draw_text(draw, (90 * scale, 44 * scale), "Global slim/wind audit of the local hot branch", font_title, axis_color)
    subtitle = (
        "Hot local roots are sampled across 0.06-0.5 R_H, then evaluated with v_R from angular momentum "
        "and Q_adv = Sigma v_R T ds/dR."
    )
    draw_text(draw, (90 * scale, 94 * scale), subtitle, font_sub, (78, 82, 88))

    left = 125 * scale
    mid = 830 * scale
    top = 175 * scale
    row_gap = 110 * scale
    panel_w = 560 * scale
    panel_h = 330 * scale
    boxes = [
        (left, top, left + panel_w, top + panel_h),
        (mid, top, mid + panel_w, top + panel_h),
        (left, top + panel_h + row_gap, left + panel_w, top + 2 * panel_h + row_gap),
        (mid, top + panel_h + row_gap, mid + panel_w, top + 2 * panel_h + row_gap),
    ]
    x_min, x_max = float(x.min()), float(x.max())
    x_ticks = [0.06, 0.1, 0.2, 0.3, 0.5]

    panels = [
        (
            boxes[0],
            "Entropy-gradient coefficient",
            "xi_eff",
            candidate.xi_eff,
            relaxed.xi_eff,
            (-2.0, 8.0),
            [-2, 0, 0.3, 2, 4, 6, 8],
            [0.3],
        ),
        (
            boxes[1],
            "Energy residual",
            "(Qvisc-Qrad-Qadv-Qwind)/Qvisc",
            candidate.energy_residual / candidate.Q_visc,
            relaxed.energy_residual / relaxed.Q_visc,
            (-8.0, 1.0),
            [-8, -6, -4, -2, 0, 1],
            [0.0],
        ),
        (
            boxes[2],
            "Advective fraction",
            "Qadv/Qvisc",
            candidate.advective_fraction,
            relaxed.advective_fraction,
            (-10.0, 8.0),
            [-10, -5, 0, 1, 4, 8],
            [0.0, 1.0],
        ),
        (
            boxes[3],
            "Radial mass flux",
            "Mdot(R) [Msun/yr]",
            g_per_s_to_msun_per_year(candidate.Mdot),
            g_per_s_to_msun_per_year(relaxed.Mdot),
            (-0.04, 0.14),
            [-0.04, 0.0, 0.02445, 0.06, 0.10, 0.14],
            [0.0, target_mdot_msun_yr],
        ),
    ]

    for box, title, ylabel, y_candidate, y_relaxed, yrange, yticks, refs in panels:
        y_min, y_max = yrange
        draw_axes(draw, box, x_ticks, yticks, x_min, x_max, y_min, y_max, font_tick, axis_color, grid_color)
        for ref in refs:
            draw_reference(draw, box, ref, y_min, y_max, ref_color, width=2 * scale)
        draw_line(draw, x, y_candidate, x_min, x_max, y_min, y_max, box, candidate_color, 5 * scale)
        draw_line(draw, x, y_relaxed, x_min, x_max, y_min, y_max, box, relaxed_color, 5 * scale)
        draw_text(draw, (box[0], box[1] - 42 * scale), title, font_panel, axis_color)
        draw_text(draw, ((box[0] + box[2]) / 2, box[3] + 50 * scale), "R/R_H", font_axis, axis_color, "ma")
        y_label_layer = Image.new("RGBA", (360 * scale, 42 * scale), (255, 255, 255, 0))
        y_draw = ImageDraw.Draw(y_label_layer)
        y_draw.text((0, 0), ylabel, font=font_axis, fill=axis_color)
        rotated = y_label_layer.rotate(90, expand=True)
        image.paste(rotated, (box[0] - 105 * scale, int((box[1] + box[3]) / 2 - rotated.height / 2)), rotated)

    legend_x = 1415 * scale
    legend_y = 190 * scale
    draw_text(draw, (legend_x, legend_y), "Curves", font_note_bold, axis_color)
    legend_y += 40 * scale
    draw.line((legend_x, legend_y + 12 * scale, legend_x + 58 * scale, legend_y + 12 * scale), fill=candidate_color, width=6 * scale)
    draw_text(draw, (legend_x + 75 * scale, legend_y), "local hot roots", font_note, axis_color)
    legend_y += 40 * scale
    draw.line((legend_x, legend_y + 12 * scale, legend_x + 58 * scale, legend_y + 12 * scale), fill=relaxed_color, width=6 * scale)
    draw_text(draw, (legend_x + 75 * scale, legend_y), "energy-relaxed", font_note, axis_color)
    legend_y += 56 * scale
    draw_text(draw, (legend_x, legend_y), "Numerical result", font_note_bold, axis_color)
    legend_y += 38 * scale
    notes = [
        "The local hot roots do",
        "not reproduce the",
        "imposed xi = 0.3 once",
        "radial gradients are",
        "computed directly.",
        "",
        "The relaxed profile",
        "closes energy, but it",
        "is not a clean steady",
        "hot branch: xi_eff is",
        "broad and one radial",
        "zone flows outward.",
    ]
    for line in notes:
        draw_text(draw, (legend_x, legend_y), line, font_note, warning_color if line.startswith("not") else (78, 82, 88))
        legend_y += 28 * scale

    legend_y += 18 * scale
    draw_text(draw, (legend_x, legend_y), "Key metrics", font_note_bold, axis_color)
    legend_y += 34 * scale
    metrics = [
        "candidate:",
        f"residual {integrated_energy_residual(candidate):.2g}",
        f"int Qadv/Qvisc {integrated_fraction(candidate.Q_adv, candidate.Q_visc, candidate.area):.2g}",
        f"max H/R {np.nanmax(candidate.H_over_R):.2g}",
        "",
        "relaxed:",
        f"residual {integrated_energy_residual(relaxed):.1g}",
        f"inward cells {100.0 * np.mean(relaxed.Mdot > 0.0):.0f}%",
        f"max H/R {np.nanmax(relaxed.H_over_R):.2g}",
        f"iterations {relaxation.iterations}",
        f"interp cells {100.0 * np.mean(interpolated):.0f}%",
        f"local Mdot max {local_mdot.max():.4g}",
    ]
    for line in metrics:
        draw_text(draw, (legend_x, legend_y), line, font_note_bold if line.endswith(":") else font_note, axis_color)
        legend_y += 27 * scale

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(OUTPUT)

    print(f"wrote {OUTPUT}")
    print("local-root candidate")
    for line in summarize("candidate", candidate, target_mdot_msun_yr)[1:]:
        print(f"  {line}")
    print("energy-relaxed")
    for line in summarize("relaxed", relaxed, target_mdot_msun_yr)[1:]:
        print(f"  {line}")
    print(f"relaxation converged = {relaxation.converged} in {relaxation.iterations} iterations")


if __name__ == "__main__":
    main()
