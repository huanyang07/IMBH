"""Generate a fiducial Layer-3 one-zone cycle diagnostic figure."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imri_qpe.constants import DAY, M_SUN
from imri_qpe.layer1_hill_flow import hill_radius
from imri_qpe.layer3_minidisk_1d import one_zone_cycle
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import omega_k
from imri_qpe.units import g_per_s_to_msun_per_year, msun_per_year_to_g_per_s, solar_masses_to_g


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "figures" / "layer3_one_zone_cycle.png"


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


def map_linear(value: float, lo: float, hi: float, pixel_lo: float, pixel_hi: float) -> float:
    return pixel_lo + (value - lo) / (hi - lo) * (pixel_hi - pixel_lo)


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=(35, 35, 35), anchor=None) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def polyline(draw: ImageDraw.ImageDraw, x, y, x_min, x_max, y_min, y_max, box, color, width: int) -> None:
    left, top, right, bottom = box
    points = [
        (
            map_linear(float(xi), x_min, x_max, left, right),
            map_linear(float(yi), y_min, y_max, bottom, top),
        )
        for xi, yi in zip(x, y)
    ]
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_panel_axes(
    draw: ImageDraw.ImageDraw,
    box,
    x_min: float,
    x_max: float,
    y_ticks: list[float],
    y_labels: list[str],
    x_ticks: list[float],
    font_tick,
    axis_color,
    grid_color,
) -> None:
    left, top, right, bottom = box
    draw.rectangle((left, top, right, bottom), outline=axis_color, width=3)
    for tick in x_ticks:
        x = map_linear(tick, x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=grid_color, width=1)
        draw.line((x, bottom, x, bottom + 8), fill=axis_color, width=2)
        draw_text(draw, (x, bottom + 18), f"{tick:g}", font_tick, axis_color, "ma")
    y_min, y_max = y_ticks[0], y_ticks[-1]
    for tick, label in zip(y_ticks, y_labels):
        y = map_linear(tick, y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=grid_color, width=1)
        draw.line((left - 8, y, left, y), fill=axis_color, width=2)
        draw_text(draw, (left - 18, y), label, font_tick, axis_color, "rm")


def build_cycle_series(cycle, n_cycles: int = 4, points_per_phase: int = 80):
    times = []
    masses = []
    mdots = []

    for cycle_index in range(n_cycles):
        t0 = cycle_index * cycle.P_QPE

        load_t = np.linspace(t0, t0 + cycle.t_load, points_per_phase, endpoint=False)
        load_phase = (load_t - t0) / cycle.t_load
        load_m = cycle.M_min + load_phase * cycle.delta_M
        load_mdot = np.zeros_like(load_t)

        high_t = np.linspace(t0 + cycle.t_load, t0 + cycle.P_QPE, points_per_phase)
        high_phase = (high_t - (t0 + cycle.t_load)) / cycle.t_high
        high_m = cycle.M_min + (1.0 - high_phase) * cycle.delta_M
        high_mdot = np.full_like(high_t, cycle.mdot_burst)

        times.extend(load_t.tolist())
        times.extend(high_t.tolist())
        masses.extend(load_m.tolist())
        masses.extend(high_m.tolist())
        mdots.extend(load_mdot.tolist())
        mdots.extend(high_mdot.tolist())

    return np.array(times), np.array(masses), np.array(mdots)


def main() -> None:
    params = FiducialParams()
    R_u = params.unstable_radius_hill_fraction * hill_radius(params.a_cm, params.q)
    Omega_K = omega_k(params.M2_g, R_u)

    cycle = one_zone_cycle(
        Mmax=solar_masses_to_g(1.6e-4),
        zeta=0.2,
        mdot_cap=msun_per_year_to_g_per_s(1.0e-2),
        mdot_low=0.0,
        alpha_hot=0.1,
        H_over_R_hot=0.1,
        Omega_K=Omega_K,
    )

    time_s, mass_g, mdot_g_s = build_cycle_series(cycle)
    time_d = time_s / DAY
    mass_msun = mass_g / M_SUN
    mdot_msun_yr = g_per_s_to_msun_per_year(mdot_g_s)
    mdot_cap_msun_yr = 1.0e-2

    scale = 2
    width, height = 1500 * scale, 1050 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_title = load_font(38 * scale, bold=True)
    font_sub = load_font(22 * scale)
    font_axis = load_font(25 * scale, bold=True)
    font_tick = load_font(20 * scale)
    font_note = load_font(21 * scale)
    font_note_bold = load_font(22 * scale, bold=True)

    axis_color = (42, 46, 52)
    grid_color = (224, 228, 232)
    mass_color = (37, 99, 180)
    mdot_color = (190, 80, 52)
    supply_color = (70, 135, 88)
    fill_hot = (253, 233, 224)

    margin_left = 220 * scale
    margin_right = 350 * scale
    plot_right = width - margin_right
    top_box = (margin_left, 190 * scale, plot_right, 510 * scale)
    bottom_box = (margin_left, 650 * scale, plot_right, 915 * scale)

    x_min = 0.0
    x_max = math.ceil(float(time_d.max()))
    x_ticks = [0, 5, 10, 15, 20, 25]

    draw_text(draw, (margin_left, 50 * scale), "Layer 3 fiducial one-zone cycle", font_title, axis_color)
    subtitle = "Mmax = 1.6e-4 Msun, zeta = 0.2, Mdot_cap = 1e-2 Msun/yr, alpha_h(H/R)^2 = 1e-3"
    draw_text(draw, (margin_left, 100 * scale), subtitle, font_sub, (78, 82, 88))

    for cycle_index in range(4):
        hot_start = (cycle_index * cycle.P_QPE + cycle.t_load) / DAY
        hot_end = ((cycle_index + 1) * cycle.P_QPE) / DAY
        for box in (top_box, bottom_box):
            x0 = map_linear(hot_start, x_min, x_max, box[0], box[2])
            x1 = map_linear(hot_end, x_min, x_max, box[0], box[2])
            draw.rectangle((x0, box[1], x1, box[3]), fill=fill_hot)

    mass_ticks = [0.0, 0.5e-4, 1.0e-4, 1.5e-4]
    draw_panel_axes(
        draw,
        top_box,
        x_min,
        x_max,
        mass_ticks,
        ["0", "0.5e-4", "1.0e-4", "1.5e-4"],
        x_ticks,
        font_tick,
        axis_color,
        grid_color,
    )
    polyline(draw, time_d, mass_msun, x_min, x_max, mass_ticks[0], mass_ticks[-1], top_box, mass_color, 5 * scale)

    mdot_ticks = [0.0, 0.01, 0.02, 0.03]
    draw_panel_axes(
        draw,
        bottom_box,
        x_min,
        x_max,
        mdot_ticks,
        ["0", "0.01", "0.02", "0.03"],
        x_ticks,
        font_tick,
        axis_color,
        grid_color,
    )
    polyline(draw, time_d, mdot_msun_yr, x_min, x_max, mdot_ticks[0], mdot_ticks[-1], bottom_box, mdot_color, 5 * scale)
    supply_y = map_linear(mdot_cap_msun_yr, mdot_ticks[0], mdot_ticks[-1], bottom_box[3], bottom_box[1])
    draw.line((bottom_box[0], supply_y, bottom_box[2], supply_y), fill=supply_color, width=3 * scale)

    draw_text(
        draw,
        ((top_box[0] + top_box[2]) / 2, top_box[1] - 28 * scale),
        "Minidisk mass M_d(t) [Msun]",
        font_axis,
        axis_color,
        "ma",
    )
    draw_text(
        draw,
        ((bottom_box[0] + bottom_box[2]) / 2, bottom_box[1] - 28 * scale),
        "Burst accretion-rate proxy [Msun yr^-1]",
        font_axis,
        axis_color,
        "ma",
    )
    draw_text(draw, ((bottom_box[0] + bottom_box[2]) / 2, height - 70 * scale), "Time (days)", font_axis, axis_color, "ma")

    legend_x = plot_right + 52 * scale
    legend_y = top_box[1] + 10 * scale
    draw_text(draw, (legend_x, legend_y), "Curves", font_note_bold, axis_color)
    legend_y += 48 * scale
    draw.line((legend_x, legend_y + 12 * scale, legend_x + 58 * scale, legend_y + 12 * scale), fill=mass_color, width=6 * scale)
    draw_text(draw, (legend_x + 76 * scale, legend_y), "disk mass", font_note, axis_color)
    legend_y += 46 * scale
    draw.line((legend_x, legend_y + 12 * scale, legend_x + 58 * scale, legend_y + 12 * scale), fill=mdot_color, width=6 * scale)
    draw_text(draw, (legend_x + 76 * scale, legend_y), "burst proxy", font_note, axis_color)
    legend_y += 46 * scale
    draw.line((legend_x, legend_y + 12 * scale, legend_x + 58 * scale, legend_y + 12 * scale), fill=supply_color, width=4 * scale)
    draw_text(draw, (legend_x + 76 * scale, legend_y), "captured supply", font_note, axis_color)
    legend_y += 46 * scale
    draw.rectangle((legend_x, legend_y, legend_x + 58 * scale, legend_y + 24 * scale), fill=fill_hot, outline=(231, 196, 179))
    draw_text(draw, (legend_x + 76 * scale, legend_y - 2 * scale), "hot state", font_note, axis_color)

    summary_y = bottom_box[1] + 10 * scale
    draw_text(draw, (legend_x, summary_y), "Fiducial result", font_note_bold, axis_color)
    summary_y += 48 * scale
    summary = [
        f"t_load = {cycle.t_load / DAY:.2f} d",
        f"t_high = {cycle.t_high / DAY:.2f} d",
        f"P_QPE = {cycle.P_QPE / DAY:.2f} d",
        f"D = {cycle.duty_cycle:.2f}",
        f"Mdot_burst = {g_per_s_to_msun_per_year(cycle.mdot_burst):.3f} Msun/yr",
    ]
    for line in summary:
        draw_text(draw, (legend_x, summary_y), line, font_note, (78, 82, 88))
        summary_y += 35 * scale

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
