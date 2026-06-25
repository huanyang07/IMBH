"""Plot residual components from the transonic coarse-to-fine audit table."""

from __future__ import annotations

from pathlib import Path
from math import log10

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TABLE_INPUT = ROOT / "outputs" / "tables" / "transonic_coarse_to_fine.md"
FIGURE_OUTPUT = ROOT / "outputs" / "figures" / "transonic_coarse_to_fine_residuals.png"
RESIDUAL_TOL = 3.0e-4


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "yes"


def parse_float(value: str) -> float:
    try:
        return float(value.strip())
    except ValueError:
        return np.nan


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("| refine_target"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        rows.append(
            {
                "stage": parts[0],
                "N": int(parts[1]),
                "R_out_rg": parse_float(parts[2]),
                "mdot": parse_float(parts[3]),
                "usable": parse_bool(parts[4]),
                "physical": parse_bool(parts[5]),
                "equations": parse_bool(parts[6]),
                "sonic": parse_bool(parts[7]),
                "max_residual": parse_float(parts[11]),
                "interval_radial": parse_float(parts[12]),
                "outer_omega": abs(parse_float(parts[13])),
                "D": abs(parse_float(parts[14])),
                "C1": abs(parse_float(parts[15])),
                "C2": abs(parse_float(parts[16])),
                "B2_max": parse_float(parts[17]),
            }
        )
    return rows


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def y_position(value: float, plot_top: int, plot_bottom: int, y_min: float, y_max: float) -> float:
    clipped = min(max(float(value), y_min), y_max)
    frac = (log10(clipped) - log10(y_min)) / (log10(y_max) - log10(y_min))
    return plot_bottom - frac * (plot_bottom - plot_top)


def draw_text_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill, fnt) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - 0.5 * (bbox[2] - bbox[0]), xy[1] - 0.5 * (bbox[3] - bbox[1])), text, fill=fill, font=fnt)


def main() -> None:
    rows = read_rows(TABLE_INPUT)
    if not rows:
        raise RuntimeError(f"no refine_target rows found in {TABLE_INPUT}")

    width_px, height_px = 1400, 900
    margin_left, margin_right = 115, 55
    margin_top, margin_bottom = 125, 230
    plot_left, plot_right = margin_left, width_px - margin_right
    plot_top, plot_bottom = margin_top, height_px - margin_bottom
    y_min, y_max = 3.0e-5, 8.0e-4

    image = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(34, bold=True)
    label_font = font(22)
    small_font = font(18)
    tick_font = font(17)

    draw.text((margin_left, 22), "Transonic coarse-to-fine residual audit", fill="#1f1f1f", font=title_font)
    draw.text((margin_left, 66), "B2 determinant ramp: target refinements use weights 0.3, 1, 3", fill="#4a4a4a", font=small_font)

    for tick in [3.0e-5, 5.0e-5, 1.0e-4, 2.0e-4, 3.0e-4, 5.0e-4, 8.0e-4]:
        y = y_position(tick, plot_top, plot_bottom, y_min, y_max)
        color = "#dadde3" if tick != RESIDUAL_TOL else "#1f1f1f"
        line_width = 1 if tick != RESIDUAL_TOL else 2
        draw.line((plot_left, y, plot_right, y), fill=color, width=line_width)
        draw.text((25, y - 10), f"{tick:.0e}", fill="#424242", font=tick_font)
    draw.text((plot_right - 185, y_position(RESIDUAL_TOL, plot_top, plot_bottom, y_min, y_max) - 28), "tol=3e-4", fill="#1f1f1f", font=small_font)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#222222", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#222222", width=2)

    components = [
        ("max residual", "max_residual", "#1f77b4"),
        ("|D|", "D", "#d62728"),
        ("|outer Omega|", "outer_omega", "#2ca02c"),
        ("interval radial", "interval_radial", "#9467bd"),
        ("B2 max", "B2_max", "#ff7f0e"),
    ]

    group_width = (plot_right - plot_left) / len(rows)
    bar_width = min(34.0, group_width / (len(components) + 2.0))
    for idx, row in enumerate(rows):
        center = plot_left + (idx + 0.5) * group_width
        for comp_idx, (_label, key, color) in enumerate(components):
            x_center = center + (comp_idx - 0.5 * (len(components) - 1)) * bar_width
            value = float(row[key])
            y = y_position(value, plot_top, plot_bottom, y_min, y_max)
            draw.rectangle((x_center - 0.45 * bar_width, y, x_center + 0.45 * bar_width, plot_bottom), fill=color)
        if not bool(row["usable"]):
            draw_text_centered(draw, (center, y_position(4.4e-4, plot_top, plot_bottom, y_min, y_max) - 12), "not usable", "#6f1d1b", small_font)
        draw_text_centered(draw, (center, plot_bottom + 28), f"N={row['N']}", "#222222", label_font)
        draw_text_centered(draw, (center, plot_bottom + 58), f"Mdot={row['mdot']:.2g}", "#4a4a4a", small_font)

    draw.text((margin_left, plot_top - 35), "Absolute scaled residual", fill="#222222", font=label_font)
    draw_text_centered(draw, ((plot_left + plot_right) / 2.0, height_px - 35), "Target solve", "#222222", label_font)

    legend_x = margin_left
    legend_y = height_px - 105
    for label, _key, color in components:
        draw.rectangle((legend_x, legend_y, legend_x + 24, legend_y + 16), fill=color)
        draw.text((legend_x + 32, legend_y - 3), label, fill="#222222", font=small_font)
        legend_x += 230

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)
    print(f"wrote {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
