"""Generate Sprint-B isolated no-wind slim-disk continuation diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imri_qpe.layer1_hill_flow import hill_radius
from imri_qpe.layer1_hill_flow.hill_geometry import schwarzschild_isco_radius
from imri_qpe.layer3_minidisk_1d import IsolatedSlimParams, continue_isolated_slim_branch, make_log_grid
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import g_per_s_to_msun_per_year


ROOT = Path(__file__).resolve().parents[1]
FIGURE_OUTPUT = ROOT / "outputs" / "figures" / "isolated_slim_branch_continuation.png"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "isolated_slim_branch_summary.md"


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


def map_log(value: float, lo: float, hi: float, pixel_lo: float, pixel_hi: float) -> float:
    return map_linear(math.log10(value), math.log10(lo), math.log10(hi), pixel_lo, pixel_hi)


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=(35, 35, 35), anchor=None) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_axes(draw, box, x_ticks, y_ticks, x_min, x_max, y_min, y_max, font_tick, axis_color, grid_color, x_log=True):
    left, top, right, bottom = box
    draw.rectangle(box, outline=axis_color, width=3)
    for tick in x_ticks:
        x = map_log(tick, x_min, x_max, left, right) if x_log else map_linear(tick, x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=grid_color, width=1)
        draw.line((x, bottom, x, bottom + 8), fill=axis_color, width=2)
        label = f"{tick:g}" if tick < 100 else f"{tick:.0f}"
        draw_text(draw, (x, bottom + 18), label, font_tick, axis_color, "ma")
    for tick in y_ticks:
        y = map_linear(tick, y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=grid_color, width=1)
        draw.line((left - 8, y, left, y), fill=axis_color, width=2)
        draw_text(draw, (left - 16, y), f"{tick:g}", font_tick, axis_color, "rm")


def draw_line(draw, x, y, box, x_min, x_max, y_min, y_max, color, width=4, x_log=True):
    left, top, right, bottom = box
    points = []
    for xi, yi in zip(x, y):
        if not np.isfinite(xi) or not np.isfinite(yi) or xi <= 0.0:
            continue
        yi = min(max(float(yi), y_min), y_max)
        xp = map_log(float(xi), x_min, x_max, left, right) if x_log else map_linear(float(xi), x_min, x_max, left, right)
        yp = map_linear(yi, y_min, y_max, bottom, top)
        points.append((xp, yp))
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_markers(draw, x, y, converged, box, x_min, x_max, y_min, y_max, color, x_log=True):
    left, top, right, bottom = box
    for xi, yi, ok in zip(x, y, converged):
        if not np.isfinite(xi) or not np.isfinite(yi) or xi <= 0.0:
            continue
        yi = min(max(float(yi), y_min), y_max)
        xp = map_log(float(xi), x_min, x_max, left, right) if x_log else map_linear(float(xi), x_min, x_max, left, right)
        yp = map_linear(yi, y_min, y_max, bottom, top)
        r = 6
        if ok:
            draw.ellipse((xp - r, yp - r, xp + r, yp + r), fill=color, outline=(255, 255, 255), width=2)
        else:
            draw.line((xp - r, yp - r, xp + r, yp + r), fill=color, width=3)
            draw.line((xp - r, yp + r, xp + r, yp - r), fill=color, width=3)


def integrated_fraction(numerator, denominator, area) -> float:
    den = float(np.sum(denominator * area))
    if den == 0.0:
        return float("nan")
    return float(np.sum(numerator * area) / den)


def metric_rows(continuation, mdot_edd):
    rows = []
    for Mdot, result in zip(continuation.mdot_values, continuation.results):
        ratio = float(Mdot / mdot_edd)
        if result.profile is None:
            rows.append(
                {
                    "ratio": ratio,
                    "mdot_msun_yr": g_per_s_to_msun_per_year(Mdot),
                    "converged": False,
                    "profile": False,
                    "L1": np.nan,
                    "max_residual": np.nan,
                    "adv_frac": np.nan,
                    "max_HR": np.nan,
                    "median_xi": np.nan,
                    "max_ang": np.nan,
                    "message": result.message,
                }
            )
            continue
        profile = result.profile
        rows.append(
            {
                "ratio": ratio,
                "mdot_msun_yr": g_per_s_to_msun_per_year(Mdot),
                "converged": result.converged,
                "profile": True,
                "L1": result.L1_residual,
                "max_residual": result.max_abs_residual,
                "adv_frac": integrated_fraction(profile.Q_adv, profile.Q_visc, profile.area),
                "max_HR": float(np.nanmax(profile.H_over_R)),
                "median_xi": float(np.nanmedian(profile.xi_eff)),
                "max_ang": float(np.nanmax(np.abs(profile.angular_momentum_residual))),
                "message": result.message,
            }
        )
    return rows


def format_float(value: float, precision: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-2 or abs(value) >= 1.0e3:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}g}"


def write_table(rows, target_ratio: float) -> None:
    lines = [
        "# Isolated No-Wind Slim-Disk Continuation",
        "",
        "Generated by `scripts/plot_isolated_slim_branch.py`.",
        "",
        "This is the Sprint-B isolated benchmark: no stream source, no tidal torque, no wind, and constant imposed `Mdot`.",
        "",
        "| Mdot/Mdot_Edd | Mdot [Msun/yr] | converged | L1 | max residual | int Qadv/Qvisc | max H/R | median xi_eff | max angular residual | message |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_float(row["ratio"]),
                    format_float(row["mdot_msun_yr"]),
                    "yes" if row["converged"] else "no",
                    format_float(row["L1"]),
                    format_float(row["max_residual"]),
                    format_float(row["adv_frac"]),
                    format_float(row["max_HR"]),
                    format_float(row["median_xi"]),
                    format_float(row["max_ang"]),
                    row["message"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The QPE burst-rate target is `Mdot/Mdot_Edd ~= {target_ratio:.1f}`.",
            "",
            "- The repaired solver recovers the low-rate thin disk and a smooth advective branch through moderate super-Eddington rates.",
            "- The branch reaches `H/R > 0.4` by a few `Mdot_Edd`, so the Keplerian reduction is no longer physically trustworthy there.",
            "- The QPE target remains outside the validated reduced-solver regime and has order-unity residuals.",
            "- The next physical step is a transonic slim-disk solver with radial momentum and sonic regularity, not stream/tide/wind.",
        ]
    )
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_figure(rows, selected_profiles, target_ratio: float) -> None:
    scale = 2
    width, height = 1700 * scale, 1220 * scale
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
    curve_color = (41, 114, 170)
    fail_color = (190, 75, 58)
    ref_color = (110, 116, 124)
    profile_color = (68, 140, 86)
    target_color = (145, 82, 45)

    draw_text(draw, (90 * scale, 44 * scale), "Isolated no-wind slim-disk continuation", font_title, axis_color)
    draw_text(
        draw,
        (90 * scale, 94 * scale),
        "Constant Mdot, no stream/tide/wind; Sigma is set by angular momentum and T is relaxed against Qvisc = Qrad + Qadv.",
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

    x = np.asarray([row["ratio"] for row in rows], dtype=float)
    converged = np.asarray([row["converged"] for row in rows], dtype=bool)
    L1 = np.asarray([row["L1"] for row in rows], dtype=float)
    HR = np.asarray([row["max_HR"] for row in rows], dtype=float)
    adv = np.asarray([row["adv_frac"] for row in rows], dtype=float)
    max_res = np.asarray([row["max_residual"] for row in rows], dtype=float)
    x_min, x_max = 1.0e-3, 120.0
    x_ticks = [1.0e-3, 1.0e-2, 0.1, 1, 10, 100]

    panels = [
        (boxes[0], "Energy L1 residual", L1, (0.0, 8.0), [0, 0.01, 1, 2, 4, 8], [0.01]),
        (boxes[1], "Max H/R", HR, (0.0, 6.5), [0, 0.3, 0.5, 1, 3, 6], [0.3, 0.5]),
        (boxes[2], "Integrated advection", adv, (-0.5, 8.0), [-0.5, 0, 1, 2, 4, 8], [0, 1]),
        (boxes[3], "Max point residual", max_res, (0.0, 10.0), [0, 0.01, 1, 2, 5, 10], [0.01]),
    ]
    for box, title, y, yrange, yticks, refs in panels:
        draw_axes(draw, box, x_ticks, yticks, x_min, x_max, yrange[0], yrange[1], font_tick, axis_color, grid_color)
        for ref in refs:
            yref = map_linear(ref, yrange[0], yrange[1], box[3], box[1])
            draw.line((box[0], yref, box[2], yref), fill=ref_color, width=2 * scale)
        xtarget = map_log(target_ratio, x_min, x_max, box[0], box[2])
        draw.line((xtarget, box[1], xtarget, box[3]), fill=target_color, width=2 * scale)
        draw_line(draw, x, y, box, x_min, x_max, yrange[0], yrange[1], curve_color, width=4 * scale)
        draw_markers(draw, x, y, converged, box, x_min, x_max, yrange[0], yrange[1], curve_color)
        draw_text(draw, (box[0], box[1] - 42 * scale), title, font_panel, axis_color)
        draw_text(draw, ((box[0] + box[2]) / 2, box[3] + 50 * scale), "Mdot/Mdot_Edd", font_axis, axis_color, "ma")

    legend_x = 1420 * scale
    legend_y = 190 * scale
    draw_text(draw, (legend_x, legend_y), "Result", font_note_bold, axis_color)
    legend_y += 40 * scale
    notes = [
        "The repaired",
        "reduced solver",
        "passes thin and",
        "moderate slim",
        "checks.",
        "",
        "The target rate is",
        f"{target_ratio:.1f} Mdot_Edd.",
        "",
        "At the target,",
        "H/R is far too",
        "large and the",
        "residual is",
        "order unity.",
    ]
    for line in notes:
        draw_text(draw, (legend_x, legend_y), line, font_note, target_color if "target" in line else (78, 82, 88))
        legend_y += 29 * scale

    legend_y += 30 * scale
    draw_text(draw, (legend_x, legend_y), "Markers", font_note_bold, axis_color)
    legend_y += 38 * scale
    draw.ellipse((legend_x, legend_y, legend_x + 14 * scale, legend_y + 14 * scale), fill=curve_color)
    draw_text(draw, (legend_x + 32 * scale, legend_y - 4 * scale), "converged", font_note, axis_color)
    legend_y += 35 * scale
    draw.line((legend_x, legend_y, legend_x + 16 * scale, legend_y + 16 * scale), fill=curve_color, width=3 * scale)
    draw.line((legend_x, legend_y + 16 * scale, legend_x + 16 * scale, legend_y), fill=curve_color, width=3 * scale)
    draw_text(draw, (legend_x + 32 * scale, legend_y - 4 * scale), "not converged", font_note, axis_color)
    legend_y += 35 * scale
    draw.line((legend_x, legend_y + 8 * scale, legend_x + 55 * scale, legend_y + 8 * scale), fill=target_color, width=3 * scale)
    draw_text(draw, (legend_x + 72 * scale, legend_y - 4 * scale), "QPE target", font_note, axis_color)

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    params = FiducialParams()
    R_H = hill_radius(params.a_cm, params.q)
    R_in = 1.2 * schwarzschild_isco_radius(params.M2_g)
    grid = make_log_grid(R_in, params.tidal_truncation_fraction * R_H, 10)
    mdot_edd = eddington_mdot(params.M2_g)
    mdot_ratios = np.array([1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 0.1, 0.3, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0])
    target_mdot_msun_yr = 0.02445
    target_ratio = target_mdot_msun_yr / g_per_s_to_msun_per_year(mdot_edd)

    base_params = IsolatedSlimParams(
        params.M2_g,
        mdot_ratios[0] * mdot_edd,
        R_in,
        alpha=0.01,
        sigma_brackets=100,
        T_bounds=(1.0e3, 1.0e10),
    )
    continuation = continue_isolated_slim_branch(
        grid,
        base_params,
        mdot_ratios * mdot_edd,
        max_iter=80,
        tol=1.0e-3,
        damping=1.0,
        boundary_exclude=2,
    )
    rows = metric_rows(continuation, mdot_edd)
    write_table(rows, target_ratio)
    draw_figure(rows, continuation.results, target_ratio)

    print(f"wrote {FIGURE_OUTPUT}")
    print(f"wrote {TABLE_OUTPUT}")
    for row in rows:
        print(
            f"Mdot/Medd={row['ratio']:.3g} converged={row['converged']} "
            f"L1={format_float(row['L1'])} adv={format_float(row['adv_frac'])} "
            f"H/R={format_float(row['max_HR'])} message={row['message']}"
        )


if __name__ == "__main__":
    main()
