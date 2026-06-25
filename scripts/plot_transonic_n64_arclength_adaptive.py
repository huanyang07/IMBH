"""Plot adaptive pseudo-arclength continuation rows for the N=64 branch."""

from __future__ import annotations

from math import log10
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TABLE_INPUT = ROOT / "outputs" / "tables" / "transonic_n64_arclength_adaptive.md"
FIGURE_OUTPUT = ROOT / "outputs" / "figures" / "transonic_n64_arclength_adaptive.png"
RESIDUAL_TOL = 3.0e-4


def parse_float(value: str) -> float:
    try:
        return float(value.strip())
    except ValueError:
        return float("nan")


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "yes"


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not (line.startswith("| seed |") or line.startswith("| arc_")):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        rows.append(
            {
                "label": parts[0],
                "ratio": parse_float(parts[2]),
                "predicted": parse_float(parts[3]),
                "accepted": parse_bool(parts[4]),
                "dominant": parts[8],
                "max_HR": parse_float(parts[9]),
                "max_residual": parse_float(parts[11]),
                "outer_omega": abs(parse_float(parts[13])),
                "D": abs(parse_float(parts[14])),
                "step": parse_float(parts[18]),
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


def draw_axes(draw, bounds, x_ticks, y_ticks, x_min, x_max, y_min, y_max, title, title_font, tick_font, label_font):
    left, top, right, bottom = bounds
    draw.text((left, top - 48), title, fill="#222222", font=title_font)
    draw.line((left, bottom, right, bottom), fill="#222222", width=2)
    draw.line((left, top, left, bottom), fill="#222222", width=2)
    for tick in x_ticks:
        if tick < x_min or tick > x_max:
            continue
        x = map_logx(tick, left, right, x_min, x_max)
        draw.line((x, bottom, x, bottom + 7), fill="#222222", width=1)
        text_center(draw, (x, bottom + 25), f"{tick:g}", "#333333", tick_font)
    for tick in y_ticks:
        y = map_logy(tick, top, bottom, y_min, y_max)
        draw.line((left, y, right, y), fill="#e1e5ea", width=1)
        draw.text((left - 80, y - 10), f"{tick:g}", fill="#333333", font=tick_font)
    draw.text((left, bottom + 50), "Mdot/MdotEdd", fill="#222222", font=label_font)


def draw_series(draw, rows, key, bounds, x_min, x_max, y_min, y_max, color, width=3):
    points = []
    for row in rows:
        x = map_logx(float(row["ratio"]), bounds[0], bounds[2], x_min, x_max)
        y = map_logy(float(row[key]), bounds[1], bounds[3], y_min, y_max)
        points.append((x, y, row))
    for p0, p1 in zip(points, points[1:]):
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)
    for x, y, row in points:
        fill = "#1b9e77" if bool(row["accepted"]) else "#d95f02"
        radius = 6 if row["label"] == "seed" else 8
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline="#111111", width=1)


def main() -> None:
    rows = read_rows(TABLE_INPUT)
    if not rows:
        raise RuntimeError(f"no seed/arclength rows found in {TABLE_INPUT}")
    rows.sort(key=lambda row: float(row["ratio"]))

    frontier = max(float(row["ratio"]) for row in rows if bool(row["accepted"]))
    width, height = 1350, 790
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28, bold=True)
    header_font = font(36, bold=True)
    label_font = font(20)
    tick_font = font(17)
    small_font = font(16)

    draw.text((75, 28), "N=64 adaptive pseudo-arclength continuation", fill="#1f1f1f", font=header_font)
    draw.text(
        (75, 75),
        f"Accepted frontier: Mdot/MdotEdd = {frontier:.4g}; tolerance fixed at 3e-4.",
        fill="#4a4a4a",
        font=label_font,
    )

    bounds = (135, 170, 1250, 600)
    x_min = 0.28
    x_max = max(1.05, 1.08 * max(float(row["ratio"]) for row in rows))
    y_min, y_max = 7.0e-5, 5.0e-4
    draw_axes(
        draw,
        bounds,
        [0.3, 0.35, 0.4, 0.5, 0.7, 1.0],
        [1.0e-4, 1.5e-4, 2.0e-4, 3.0e-4, 5.0e-4],
        x_min,
        x_max,
        y_min,
        y_max,
        "Residual blocks",
        title_font,
        tick_font,
        label_font,
    )
    y_tol = map_logy(RESIDUAL_TOL, bounds[1], bounds[3], y_min, y_max)
    draw.line((bounds[0], y_tol, bounds[2], y_tol), fill="#111111", width=2)
    draw.text((bounds[2] - 120, y_tol - 28), "tol=3e-4", fill="#111111", font=small_font)
    draw_series(draw, rows, "max_residual", bounds, x_min, x_max, y_min, y_max, "#4daf4a", width=3)
    draw_series(draw, rows, "D", bounds, x_min, x_max, y_min, y_max, "#e41a1c", width=2)
    draw_series(draw, rows, "outer_omega", bounds, x_min, x_max, y_min, y_max, "#984ea3", width=2)

    legend_y = 705
    for x, label, color in [
        (135, "max residual", "#4daf4a"),
        (345, "|D|", "#e41a1c"),
        (465, "|outer Omega|", "#984ea3"),
        (665, "accepted point", "#1b9e77"),
        (865, "failed point", "#d95f02"),
    ]:
        if "point" in label:
            draw.ellipse((x, legend_y - 2, x + 18, legend_y + 16), fill=color, outline="#111111")
        else:
            draw.rectangle((x, legend_y, x + 22, legend_y + 15), fill=color)
        draw.text((x + 32, legend_y - 3), label, fill="#222222", font=small_font)

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)
    print(f"wrote {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
