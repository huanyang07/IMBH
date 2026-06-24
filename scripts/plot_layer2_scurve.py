"""Generate a fiducial Layer-2 S-curve diagnostic figure.

This script intentionally avoids Matplotlib because the bundled runtime for
this project currently has NumPy and Pillow, but not Matplotlib.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imri_qpe.layer1_hill_flow.hill_geometry import hill_radius
from imri_qpe.layer2_scurve import ThermalEquilibriumParams, compute_scurve
from imri_qpe.parameters import FiducialParams
from imri_qpe.units import g_per_s_to_msun_per_year


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "figures" / "layer2_fiducial_scurve.png"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a common sans font if present, otherwise use Pillow's default."""

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def log_map(value: float, lo: float, hi: float, pixel_lo: float, pixel_hi: float) -> float:
    """Map a positive value to a pixel coordinate on a log axis."""

    return pixel_lo + (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (
        pixel_hi - pixel_lo
    )


def draw_dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color, width: int = 4) -> None:
    """Draw a dashed polyline through points."""

    dash = 14.0
    gap = 9.0
    for start, end in zip(points[:-1], points[1:]):
        x0, y0 = start
        x1, y1 = end
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0.0:
            continue
        ux = (x1 - x0) / length
        uy = (y1 - y0) / length
        distance = 0.0
        while distance < length:
            segment_end = min(distance + dash, length)
            xa = x0 + ux * distance
            ya = y0 + uy * distance
            xb = x0 + ux * segment_end
            yb = y0 + uy * segment_end
            draw.line([(xa, ya), (xb, yb)], fill=color, width=width)
            distance += dash + gap


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, color, stable: bool, radius: int = 6) -> None:
    """Draw filled circles for stable points and x markers for unstable points."""

    if stable:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255), width=2)
    else:
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=3)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=3)


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font,
    fill=(35, 35, 35),
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def main() -> None:
    params = FiducialParams()
    R_hill = hill_radius(params.a_cm, params.q)
    R_u = params.unstable_radius_hill_fraction * R_hill
    sigma_grid = np.geomspace(1.0e4, 1.0e7, 36)

    scans = [
        ("mu_stress = 0", (37, 99, 180), ThermalEquilibriumParams(alpha=0.01, mu_stress=0.0)),
        ("mu_stress = 4/7", (35, 139, 79), ThermalEquilibriumParams(alpha=0.01, mu_stress=4.0 / 7.0)),
        ("mu_stress = 1", (128, 70, 155), ThermalEquilibriumParams(alpha=0.01, mu_stress=1.0)),
    ]

    results = []
    for label, color, thermal_params in scans:
        result = compute_scurve(R_u, sigma_grid, params.M2_g, thermal_params, n_T_brackets=320)
        mdot_msun_yr = g_per_s_to_msun_per_year(result.mdot)
        results.append((label, color, result, mdot_msun_yr))

    all_sigma = np.concatenate([result.Sigma for _, _, result, _ in results if len(result.Sigma)])
    all_mdot = np.concatenate([mdot for _, _, _, mdot in results if len(mdot)])
    x_min = 10.0 ** math.floor(math.log10(float(all_sigma.min())))
    x_max = 10.0 ** math.ceil(math.log10(float(all_sigma.max())))
    y_min = 10.0 ** math.floor(math.log10(float(all_mdot.min())))
    y_max = 10.0 ** math.ceil(math.log10(float(all_mdot.max())))

    scale = 2
    width, height = 1500 * scale, 1050 * scale
    margin_left = 170 * scale
    margin_right = 370 * scale
    margin_top = 150 * scale
    margin_bottom = 170 * scale
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_title = load_font(38 * scale, bold=True)
    font_subtitle = load_font(22 * scale)
    font_axis = load_font(25 * scale, bold=True)
    font_tick = load_font(20 * scale)
    font_legend = load_font(22 * scale)
    font_note = load_font(19 * scale)

    grid_color = (224, 228, 232)
    axis_color = (45, 49, 54)
    text_color = (32, 34, 37)

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=axis_color, width=3 * scale)

    x_ticks = [10.0**p for p in range(int(math.log10(x_min)), int(math.log10(x_max)) + 1)]
    y_ticks = [10.0**p for p in range(int(math.log10(y_min)), int(math.log10(y_max)) + 1)]

    for tick in x_ticks:
        x = log_map(tick, x_min, x_max, plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill=grid_color, width=1 * scale)
        draw.line((x, plot_bottom, x, plot_bottom + 12 * scale), fill=axis_color, width=3 * scale)
        draw_text(draw, (x, plot_bottom + 24 * scale), f"1e{int(math.log10(tick))}", font_tick, text_color, "ma")

    for tick in y_ticks:
        y = log_map(tick, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1 * scale)
        draw.line((plot_left - 12 * scale, y, plot_left, y), fill=axis_color, width=3 * scale)
        draw_text(draw, (plot_left - 24 * scale, y), f"1e{int(math.log10(tick))}", font_tick, text_color, "rm")

    draw_text(draw, (plot_left, 48 * scale), "Layer 2 fiducial local S-curve", font_title, text_color)
    subtitle = "R_u = 0.3 R_H, M2 = 1e4 Msun, alpha = 0.01; filled circles are thermally stable roots"
    draw_text(draw, (plot_left, 98 * scale), subtitle, font_subtitle, (78, 82, 88))
    draw_text(draw, ((plot_left + plot_right) / 2, height - 78 * scale), "Surface density Sigma (g cm^-2)", font_axis, text_color, "ma")

    y_label = "Accretion-rate proxy Mdot (Msun yr^-1)"
    label_layer = Image.new("RGBA", (560 * scale, 55 * scale), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text((0, 0), y_label, font=font_axis, fill=text_color)
    rotated = label_layer.rotate(90, expand=True)
    image.paste(rotated, (38 * scale, int((plot_top + plot_bottom) / 2 - rotated.height / 2)), rotated)

    for label, color, result, mdot_msun_yr in results:
        for branch in sorted(set(result.branch_index.tolist())):
            mask = result.branch_index == branch
            branch_sigma = result.Sigma[mask]
            branch_mdot = mdot_msun_yr[mask]
            branch_stable = result.stable[mask]
            order = np.argsort(branch_sigma)
            points = [
                (
                    log_map(float(branch_sigma[i]), x_min, x_max, plot_left, plot_right),
                    log_map(float(branch_mdot[i]), y_min, y_max, plot_bottom, plot_top),
                )
                for i in order
            ]
            if len(points) >= 2:
                if np.all(branch_stable):
                    draw.line(points, fill=color, width=5 * scale, joint="curve")
                elif not np.any(branch_stable):
                    draw_dashed_line(draw, points, color, width=5 * scale)
                else:
                    draw.line(points, fill=color, width=4 * scale, joint="curve")
            for i, point in zip(order, points):
                draw_marker(draw, point[0], point[1], color, bool(branch_stable[i]), radius=6 * scale)

    legend_x = plot_right + 55 * scale
    legend_y = plot_top + 25 * scale
    draw_text(draw, (legend_x, legend_y), "Stress law", font_axis, text_color)
    legend_y += 52 * scale
    for label, color, result, _ in results:
        draw.line((legend_x, legend_y + 12 * scale, legend_x + 58 * scale, legend_y + 12 * scale), fill=color, width=6 * scale)
        draw_marker(draw, legend_x + 29 * scale, legend_y + 12 * scale, color, True, radius=6 * scale)
        draw_text(draw, (legend_x + 76 * scale, legend_y), label, font_legend, text_color)
        legend_y += 46 * scale

    legend_y += 26 * scale
    draw_text(draw, (legend_x, legend_y), "Stability", font_axis, text_color)
    legend_y += 50 * scale
    draw_marker(draw, legend_x + 26 * scale, legend_y + 10 * scale, (37, 99, 180), True, radius=7 * scale)
    draw_text(draw, (legend_x + 76 * scale, legend_y), "stable", font_legend, text_color)
    legend_y += 46 * scale
    draw_marker(draw, legend_x + 26 * scale, legend_y + 10 * scale, (37, 99, 180), False, radius=7 * scale)
    draw_text(draw, (legend_x + 76 * scale, legend_y), "unstable", font_legend, text_color)

    note_y = plot_bottom - 185 * scale
    notes = [
        "Result:",
        "mu_stress = 0 gives paired",
        "stable/unstable roots.",
        "mu_stress >= 4/7 is stable",
        "in this simplified scan.",
    ]
    for line in notes:
        draw_text(draw, (legend_x, note_y), line, font_note, (78, 82, 88))
        note_y += 31 * scale

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()

