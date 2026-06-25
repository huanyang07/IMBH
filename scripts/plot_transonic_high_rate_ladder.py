"""Plot the high-rate transonic continuation ladder."""

from __future__ import annotations

from math import log10
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TABLE_INPUT = ROOT / "outputs" / "tables" / "transonic_high_rate_ladder.md"
FIGURE_OUTPUT = ROOT / "outputs" / "figures" / "transonic_high_rate_ladder.png"
RESIDUAL_TOL = 3.0e-4


def parse_float(value: str) -> float:
    try:
        return float(value.strip())
    except ValueError:
        return np.nan


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "yes"


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not (line.startswith("| scout |") or line.startswith("| confirm |")):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        rows.append(
            {
                "label": parts[0],
                "seed": parts[1],
                "N": int(parts[2]),
                "ratio": parse_float(parts[4]),
                "accepted": parse_bool(parts[5]),
                "dominant": parts[11],
                "max_HR": parse_float(parts[14]),
                "max_Qadv_Qvisc": parse_float(parts[17]),
                "int_adv": parse_float(parts[18]),
                "max_residual": parse_float(parts[20]),
                "outer_omega": abs(parse_float(parts[23])),
                "D": abs(parse_float(parts[25])),
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


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill, fnt) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - 0.5 * (bbox[2] - bbox[0]), xy[1] - 0.5 * (bbox[3] - bbox[1])), text, fill=fill, font=fnt)


def map_logx(value: float, left: int, right: int, x_min: float, x_max: float) -> float:
    frac = (log10(value) - log10(x_min)) / (log10(x_max) - log10(x_min))
    return left + frac * (right - left)


def map_logy(value: float, top: int, bottom: int, y_min: float, y_max: float) -> float:
    clipped = min(max(abs(value), y_min), y_max)
    frac = (log10(clipped) - log10(y_min)) / (log10(y_max) - log10(y_min))
    return bottom - frac * (bottom - top)


def map_liny(value: float, top: int, bottom: int, y_min: float, y_max: float) -> float:
    clipped = min(max(value, y_min), y_max)
    frac = (clipped - y_min) / (y_max - y_min)
    return bottom - frac * (bottom - top)


def draw_axes(draw, bounds, x_ticks, y_ticks, x_min, x_max, y_min, y_max, title, y_label, log_y, title_font, tick_font, label_font):
    left, top, right, bottom = bounds
    draw.text((left, top - 45), title, fill="#222222", font=title_font)
    draw.line((left, bottom, right, bottom), fill="#222222", width=2)
    draw.line((left, top, left, bottom), fill="#222222", width=2)
    for tick in x_ticks:
        x = map_logx(tick, left, right, x_min, x_max)
        draw.line((x, bottom, x, bottom + 7), fill="#222222", width=1)
        text_center(draw, (x, bottom + 25), f"{tick:g}", "#333333", tick_font)
    for tick in y_ticks:
        y = map_logy(tick, top, bottom, y_min, y_max) if log_y else map_liny(tick, top, bottom, y_min, y_max)
        draw.line((left, y, right, y), fill="#e1e5ea", width=1)
        draw.text((left - 78, y - 10), f"{tick:g}", fill="#333333", font=tick_font)
    draw.text((left, bottom + 50), "Mdot/MdotEdd", fill="#222222", font=label_font)
    draw.text((left - 82, top - 30), y_label, fill="#222222", font=label_font)


def draw_series(draw, rows, key, bounds, x_min, x_max, y_min, y_max, log_y, color, line_width=3):
    points = []
    for row in rows:
        x = map_logx(float(row["ratio"]), bounds[0], bounds[2], x_min, x_max)
        y_value = float(row[key])
        y = map_logy(y_value, bounds[1], bounds[3], y_min, y_max) if log_y else map_liny(y_value, bounds[1], bounds[3], y_min, y_max)
        points.append((x, y, row))
    for p0, p1 in zip(points, points[1:]):
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=line_width)
    for x, y, row in points:
        fill = "#1b9e77" if bool(row["accepted"]) else "#d95f02"
        r = 6 if row["label"] == "scout" else 8
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline="#111111", width=1)


def main() -> None:
    rows = read_rows(TABLE_INPUT)
    if not rows:
        raise RuntimeError(f"no scout/confirm rows found in {TABLE_INPUT}")
    scout = [row for row in rows if row["label"] == "scout"]
    confirm = [row for row in rows if row["label"] == "confirm"]
    scout.sort(key=lambda row: float(row["ratio"]))
    confirm.sort(key=lambda row: float(row["ratio"]))

    width, height = 1350, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28, bold=True)
    header_font = font(36, bold=True)
    label_font = font(20)
    tick_font = font(17)
    small_font = font(16)

    draw.text((75, 25), "Transonic high-rate continuation ladder", fill="#1f1f1f", font=header_font)
    draw.text((75, 72), "Green points are accepted; orange points fail the 3e-4 residual acceptance threshold.", fill="#4a4a4a", font=label_font)

    x_min, x_max = 0.045, 1.08
    x_ticks = [0.05, 0.1, 0.2, 0.5, 0.7, 1.0]
    top_bounds = (130, 165, 1250, 430)
    bottom_bounds = (130, 555, 1250, 805)

    draw_axes(
        draw,
        top_bounds,
        x_ticks,
        [1.0e-4, 2.0e-4, 3.0e-4, 5.0e-4],
        x_min,
        x_max,
        8.0e-5,
        6.0e-4,
        "Residual frontier",
        "residual",
        True,
        title_font,
        tick_font,
        label_font,
    )
    y_tol = map_logy(RESIDUAL_TOL, top_bounds[1], top_bounds[3], 8.0e-5, 6.0e-4)
    draw.line((top_bounds[0], y_tol, top_bounds[2], y_tol), fill="#111111", width=2)
    draw.text((top_bounds[2] - 120, y_tol - 28), "tol=3e-4", fill="#111111", font=small_font)
    draw_series(draw, scout, "max_residual", top_bounds, x_min, x_max, 8.0e-5, 6.0e-4, True, "#377eb8")
    draw_series(draw, confirm, "max_residual", top_bounds, x_min, x_max, 8.0e-5, 6.0e-4, True, "#984ea3", line_width=2)

    draw_axes(
        draw,
        bottom_bounds,
        x_ticks,
        [0.0, 0.05, 0.1, 0.15, 0.2],
        x_min,
        x_max,
        0.0,
        0.18,
        "Thickness and advection",
        "value",
        False,
        title_font,
        tick_font,
        label_font,
    )
    draw_series(draw, scout, "max_HR", bottom_bounds, x_min, x_max, 0.0, 0.18, False, "#4daf4a")
    adv_rows = [dict(row, int_adv=abs(float(row["int_adv"]))) for row in scout]
    draw_series(draw, adv_rows, "int_adv", bottom_bounds, x_min, x_max, 0.0, 0.18, False, "#e41a1c", line_width=2)
    draw.rectangle((890, 575, 910, 591), fill="#4daf4a")
    draw.text((920, 570), "max H/R", fill="#222222", font=small_font)
    draw.rectangle((1030, 575, 1050, 591), fill="#e41a1c")
    draw.text((1060, 570), "|integrated advective fraction|", fill="#222222", font=small_font)

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)
    print(f"wrote {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
