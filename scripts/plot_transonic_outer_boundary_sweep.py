"""Plot target rows from the transonic outer-boundary sweep."""

from __future__ import annotations

from math import log10
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TABLE_INPUT = ROOT / "outputs" / "tables" / "transonic_outer_boundary_sweep.md"
FIGURE_OUTPUT = ROOT / "outputs" / "figures" / "transonic_outer_boundary_sweep.png"
RESIDUAL_TOL = 3.0e-4


def parse_float(value: str) -> float:
    try:
        return float(value.strip())
    except ValueError:
        return np.nan


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "yes"


def read_target_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("| target |"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        rows.append(
            {
                "N": int(parts[1]),
                "R_out": parse_float(parts[2]),
                "usable": parse_bool(parts[4]),
                "dominant": parts[9],
                "max_residual": parse_float(parts[15]),
                "interval_radial": parse_float(parts[16]),
                "outer_omega": abs(parse_float(parts[18])),
                "D": abs(parse_float(parts[20])),
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


def residual_color(value: float, vmin: float = 1.0e-5, vmax: float = 6.0e-4) -> tuple[int, int, int]:
    frac = (log10(max(value, vmin)) - log10(vmin)) / (log10(vmax) - log10(vmin))
    frac = min(max(frac, 0.0), 1.0)
    low = np.array([237, 248, 251], dtype=float)
    mid = np.array([116, 169, 207], dtype=float)
    high = np.array([202, 0, 32], dtype=float)
    if frac < 0.6:
        t = frac / 0.6
        color = (1.0 - t) * low + t * mid
    else:
        t = (frac - 0.6) / 0.4
        color = (1.0 - t) * mid + t * high
    return tuple(int(round(c)) for c in color)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill, fnt) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - 0.5 * (bbox[2] - bbox[0]), xy[1] - 0.5 * (bbox[3] - bbox[1])), text, fill=fill, font=fnt)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, object]],
    key: str,
    title: str,
    origin: tuple[int, int],
    cell_w: int,
    cell_h: int,
    ns: list[int],
    r_outs: list[float],
    title_font,
    label_font,
    small_font,
) -> None:
    x0, y0 = origin
    draw.text((x0, y0 - 52), title, fill="#222222", font=title_font)
    for col, n_nodes in enumerate(ns):
        text_center(draw, (x0 + (col + 0.5) * cell_w, y0 - 18), f"N={n_nodes}", "#222222", label_font)
    for row_idx, r_out in enumerate(r_outs):
        text_center(draw, (x0 - 58, y0 + (row_idx + 0.5) * cell_h), f"{r_out:g}", "#222222", label_font)
        for col, n_nodes in enumerate(ns):
            match = next(item for item in rows if item["N"] == n_nodes and item["R_out"] == r_out)
            value = float(match[key])
            left = x0 + col * cell_w
            top = y0 + row_idx * cell_h
            fill = residual_color(value)
            draw.rectangle((left, top, left + cell_w - 4, top + cell_h - 4), fill=fill, outline="#ffffff", width=2)
            text_fill = "#111111" if value < 2.0e-4 else "#ffffff"
            text_center(draw, (left + 0.5 * cell_w, top + 0.42 * cell_h), f"{value:.2e}", text_fill, small_font)
            text_center(draw, (left + 0.5 * cell_w, top + 0.68 * cell_h), str(match["dominant"]), text_fill, small_font)
            if not bool(match["usable"]):
                draw.rectangle((left + 5, top + 5, left + cell_w - 9, top + cell_h - 9), outline="#111111", width=4)
                text_center(draw, (left + 0.5 * cell_w, top + 0.18 * cell_h), "not usable", text_fill, small_font)
    draw.text((x0 - 98, y0 - 18), "R_out", fill="#222222", font=label_font)


def main() -> None:
    rows = read_target_rows(TABLE_INPUT)
    if not rows:
        raise RuntimeError(f"no target rows found in {TABLE_INPUT}")

    ns = sorted({int(row["N"]) for row in rows})
    r_outs = sorted({float(row["R_out"]) for row in rows})
    width, height = 1250, 740
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28, bold=True)
    header_font = font(36, bold=True)
    label_font = font(20)
    small_font = font(16)

    draw.text((75, 28), "Outer-boundary sweep at Mdot/MdotEdd=0.1", fill="#1f1f1f", font=header_font)
    draw.text((75, 75), "Cells show residual value and dominant residual block; outlined cells are not usable.", fill="#4a4a4a", font=label_font)
    draw.text((75, 105), f"Reference tolerance: {RESIDUAL_TOL:.0e}", fill="#4a4a4a", font=label_font)

    draw_panel(
        draw,
        rows,
        "max_residual",
        "Max residual",
        (145, 190),
        155,
        105,
        ns,
        r_outs,
        title_font,
        label_font,
        small_font,
    )
    draw_panel(
        draw,
        rows,
        "outer_omega",
        "|outer Omega|",
        (710, 190),
        155,
        105,
        ns,
        r_outs,
        title_font,
        label_font,
        small_font,
    )

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)
    print(f"wrote {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
