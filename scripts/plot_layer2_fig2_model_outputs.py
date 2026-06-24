"""Generate a Fig. 2-style S-curve plot from actual Layer-2 model roots."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imri_qpe.layer1_hill_flow import hill_radius
from imri_qpe.layer2_scurve import ThermalEquilibriumParams, compute_scurve
from imri_qpe.parameters import FiducialParams
from imri_qpe.units import g_per_s_to_msun_per_year


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "figures" / "layer2_physical_advective_scurve.png"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def log_map(value: float, lo: float, hi: float, pixel_lo: float, pixel_hi: float) -> float:
    return pixel_lo + (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (
        pixel_hi - pixel_lo
    )


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=(35, 35, 35), anchor=None) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def curve_points(result, mdot_msun_yr, mask, x_min, x_max, y_min, y_max, box):
    left, top, right, bottom = box
    order = np.argsort(result.Sigma[mask])
    sigma = result.Sigma[mask][order]
    mdot = mdot_msun_yr[mask][order]
    return [
        (
            log_map(float(s), x_min, x_max, left, right),
            log_map(float(m), y_min, y_max, bottom, top),
        )
        for s, m in zip(sigma, mdot)
    ]


def draw_dashed_line(draw: ImageDraw.ImageDraw, points, color, width: int) -> None:
    dash = 14.0
    gap = 9.0
    for p0, p1 in zip(points[:-1], points[1:]):
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0.0:
            continue
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        distance = 0.0
        while distance < length:
            end = min(distance + dash, length)
            draw.line((x0 + ux * distance, y0 + uy * distance, x0 + ux * end, y0 + uy * end), fill=color, width=width)
            distance += dash + gap


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color, width: int = 4) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 16.0
    spread = 0.55
    p1 = (end[0] - head * math.cos(angle - spread), end[1] - head * math.sin(angle - spread))
    p2 = (end[0] - head * math.cos(angle + spread), end[1] - head * math.sin(angle + spread))
    draw.polygon([end, p1, p2], fill=color)


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, color, stable: bool, radius: int = 5) -> None:
    if stable:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255), width=2)
    else:
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=3)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=3)


def main() -> None:
    params = FiducialParams()
    R_u = params.unstable_radius_hill_fraction * hill_radius(params.a_cm, params.q)
    thermal_params = ThermalEquilibriumParams(
        alpha=0.01,
        mu_stress=0.0,
        advective_entropy_gradient=0.3,
    )
    sigma_grid = np.geomspace(1.0e3, 1.0e8, 120)
    result = compute_scurve(
        R_u,
        sigma_grid,
        params.M2_g,
        thermal_params,
        T_bounds=(1.0e3, 1.0e10),
        n_T_brackets=600,
    )
    mdot_msun_yr = g_per_s_to_msun_per_year(result.mdot)

    cool_mask = result.stable & (result.T_c < 1.5e6)
    unstable_mask = ~result.stable
    hot_mask = result.stable & (result.T_c >= 1.5e6)

    x_min = 1.0e3
    x_max = 1.0e8
    y_min = 1.0e-9
    y_max = 1.0e3

    scale = 2
    width, height = 1500 * scale, 1050 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_title = load_font(38 * scale, bold=True)
    font_sub = load_font(21 * scale)
    font_axis = load_font(25 * scale, bold=True)
    font_tick = load_font(19 * scale)
    font_label = load_font(21 * scale, bold=True)
    font_note = load_font(18 * scale)

    axis_color = (42, 46, 52)
    grid_color = (224, 228, 232)
    cool_color = (31, 91, 181)
    unstable_color = (220, 130, 36)
    hot_color = (190, 47, 47)
    arrow_color = (92, 97, 104)

    margin_left = 180 * scale
    margin_right = 390 * scale
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = 175 * scale
    plot_bottom = 865 * scale
    box = (plot_left, plot_top, plot_right, plot_bottom)

    draw_text(draw, (plot_left, 52 * scale), "Computed local S-curve with slim-disk advective cooling", font_title, axis_color)
    subtitle = "Actual Layer-2 roots at R_u = 0.3 R_H; mu_stress = 0; Q_adv = xi Mdot P / (2 pi R^2 rho), xi = 0.3"
    draw_text(draw, (plot_left, 103 * scale), subtitle, font_sub, (78, 82, 88))

    draw.rectangle(box, outline=axis_color, width=3 * scale)
    x_ticks = [1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8]
    y_ticks = [1.0e-9, 1.0e-6, 1.0e-3, 1.0e0, 1.0e3]
    for tick in x_ticks:
        x = log_map(tick, x_min, x_max, plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill=grid_color, width=1 * scale)
        draw.line((x, plot_bottom, x, plot_bottom + 10 * scale), fill=axis_color, width=2 * scale)
        draw_text(draw, (x, plot_bottom + 23 * scale), f"{tick:.0e}", font_tick, axis_color, "ma")
    for tick in y_ticks:
        y = log_map(tick, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1 * scale)
        draw.line((plot_left - 10 * scale, y, plot_left, y), fill=axis_color, width=2 * scale)
        draw_text(draw, (plot_left - 22 * scale, y), f"{tick:.0e}", font_tick, axis_color, "rm")

    for mask, color, dashed in [(cool_mask, cool_color, False), (unstable_mask, unstable_color, True), (hot_mask, hot_color, False)]:
        points = curve_points(result, mdot_msun_yr, mask, x_min, x_max, y_min, y_max, box)
        if dashed:
            draw_dashed_line(draw, points, color, width=5 * scale)
        else:
            draw.line(points, fill=color, width=5 * scale, joint="curve")
        branch_stable = result.stable[mask][np.argsort(result.Sigma[mask])]
        for idx in np.linspace(0, len(points) - 1, 12, dtype=int):
            draw_marker(draw, points[idx][0], points[idx][1], color, bool(branch_stable[idx]), radius=5 * scale)

    cool_points = curve_points(result, mdot_msun_yr, cool_mask, x_min, x_max, y_min, y_max, box)
    unstable_points = curve_points(result, mdot_msun_yr, unstable_mask, x_min, x_max, y_min, y_max, box)
    hot_points = curve_points(result, mdot_msun_yr, hot_mask, x_min, x_max, y_min, y_max, box)

    draw_arrow(draw, cool_points[15], cool_points[-8], arrow_color, width=2 * scale)
    draw_arrow(
        draw,
        cool_points[-1],
        hot_points[int(np.argmin(np.abs(result.Sigma[hot_mask] - result.Sigma[cool_mask].max())))],
        arrow_color,
        width=2 * scale,
    )
    draw_arrow(draw, hot_points[-8], hot_points[20], arrow_color, width=2 * scale)

    draw_text(draw, (cool_points[min(35, len(cool_points) - 1)][0] + 10 * scale, cool_points[min(35, len(cool_points) - 1)][1] - 36 * scale), "cool stable", font_label, cool_color)
    draw_text(draw, (unstable_points[len(unstable_points) // 2][0] - 16 * scale, unstable_points[len(unstable_points) // 2][1] - 42 * scale), "unstable", font_label, unstable_color)
    draw_text(draw, (hot_points[len(hot_points) // 2][0] - 40 * scale, hot_points[len(hot_points) // 2][1] - 40 * scale), "hot advective", font_label, hot_color)
    draw_text(draw, (plot_left + 415 * scale, plot_top + 390 * scale), "load -> heat -> drain -> cool", font_label, arrow_color)

    draw_text(draw, ((plot_left + plot_right) / 2, height - 70 * scale), "Surface density Sigma (g cm^-2)", font_axis, axis_color, "ma")
    y_layer = Image.new("RGBA", (670 * scale, 60 * scale), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_layer)
    y_draw.text((0, 0), "Accretion-rate proxy Mdot (Msun yr^-1)", font=font_axis, fill=axis_color)
    rotated = y_layer.rotate(90, expand=True)
    image.paste(rotated, (42 * scale, int((plot_top + plot_bottom) / 2 - rotated.height / 2)), rotated)

    legend_x = plot_right + 52 * scale
    legend_y = plot_top + 10 * scale
    draw_text(draw, (legend_x, legend_y), "Branches", font_axis, axis_color)
    legend_y += 52 * scale
    for label, color, stable, dashed in [
        ("cool stable", cool_color, True, False),
        ("unstable", unstable_color, False, True),
        ("hot stable", hot_color, True, False),
    ]:
        pts = [(legend_x, legend_y + 13 * scale), (legend_x + 60 * scale, legend_y + 13 * scale)]
        if dashed:
            draw_dashed_line(draw, pts, color, width=5 * scale)
        else:
            draw.line((pts[0], pts[1]), fill=color, width=5 * scale)
        draw_marker(draw, legend_x + 30 * scale, legend_y + 13 * scale, color, stable, radius=6 * scale)
        draw_text(draw, (legend_x + 80 * scale, legend_y), label, font_sub, axis_color)
        legend_y += 46 * scale

    legend_y += 25 * scale
    draw_text(draw, (legend_x, legend_y), "Important caveat", font_axis, axis_color)
    legend_y += 48 * scale
    notes = [
        "The red branch appears",
        "from the explicit local",
        "slim-disk advection",
        "term, not the toy",
        "temperature switch.",
        "",
        "Without advection, the",
        "thin-disk closure gives",
        "only cool stable +",
        "unstable roots.",
    ]
    for line in notes:
        draw_text(draw, (legend_x, legend_y), line, font_note, (78, 82, 88))
        legend_y += 29 * scale

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
