"""Generate first transonic slim-disk Milestone-T1 diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams, remap_profile_to_new_sonic_grid, solve_transonic_outer_branch
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
FIGURE_OUTPUT = ROOT / "outputs" / "figures" / "transonic_branch_summary.png"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_solver_audit.md"


def load_font(size: int, bold: bool = False):
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


def draw_text(draw, xy, text: str, font, fill=(35, 39, 47), anchor=None) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_axes(draw, box, x_ticks, y_ticks, x_min, x_max, y_min, y_max, font_tick, axis_color, grid_color, y_log=False):
    left, top, right, bottom = box
    draw.rectangle(box, outline=axis_color, width=3)
    for tick in x_ticks:
        x = map_log(tick, x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=grid_color, width=1)
        draw.line((x, bottom, x, bottom + 8), fill=axis_color, width=2)
        draw_text(draw, (x, bottom + 18), f"{tick:g}", font_tick, axis_color, "ma")
    for tick in y_ticks:
        if y_log:
            y = map_log(tick, y_min, y_max, bottom, top)
        else:
            y = map_linear(tick, y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=grid_color, width=1)
        draw.line((left - 8, y, left, y), fill=axis_color, width=2)
        draw_text(draw, (left - 14, y), f"{tick:g}", font_tick, axis_color, "rm")


def draw_series(draw, x, y, ok, box, x_min, x_max, y_min, y_max, color, y_log=False):
    left, top, right, bottom = box
    points = []
    for xi, yi in zip(x, y):
        if not np.isfinite(xi) or not np.isfinite(yi) or xi <= 0.0:
            continue
        yi = min(max(float(yi), y_min), y_max)
        xp = map_log(float(xi), x_min, x_max, left, right)
        yp = map_log(yi, y_min, y_max, bottom, top) if y_log else map_linear(yi, y_min, y_max, bottom, top)
        points.append((xp, yp))
    if len(points) >= 2:
        draw.line(points, fill=color, width=4, joint="curve")
    for xi, yi, is_ok in zip(x, y, ok):
        if not np.isfinite(xi) or not np.isfinite(yi) or xi <= 0.0:
            continue
        yi = min(max(float(yi), y_min), y_max)
        xp = map_log(float(xi), x_min, x_max, left, right)
        yp = map_log(yi, y_min, y_max, bottom, top) if y_log else map_linear(yi, y_min, y_max, bottom, top)
        r = 7
        if is_ok:
            draw.ellipse((xp - r, yp - r, xp + r, yp + r), fill=color, outline=(255, 255, 255), width=2)
        else:
            draw.line((xp - r, yp - r, xp + r, yp + r), fill=color, width=3)
            draw.line((xp - r, yp + r, xp + r, yp - r), fill=color, width=3)


def solve_rows():
    fiducial = FiducialParams()
    M2_g = fiducial.M2_g
    mdot_edd = eddington_mdot(M2_g)
    ratios = np.array([1.0e-3, 3.0e-3, 1.0e-2, 2.0e-2, 3.0e-2, 5.0e-2, 0.1, 1.0])
    rows = []
    previous_profile = None
    for ratio in ratios:
        params = TransonicSlimParams(
            M2_g=M2_g,
            Mdot_g_s=float(ratio * mdot_edd),
            alpha=fiducial.alpha_cool,
            n_nodes=18,
            R_out_rg=300.0,
            max_nfev=220,
            residual_tol=1.0e-3,
        )
        guess = remap_profile_to_new_sonic_grid(previous_profile, params) if previous_profile is not None else None
        result = solve_transonic_outer_branch(params, guess)
        profile = result.profile
        audit = result.residual_audit
        status = result.status
        # This is only a smoke-audit seed. Scientific continuation still uses
        # the stricter physical-validity flag reported in the table.
        if status.equations_converged and profile is not None:
            previous_profile = profile
        rows.append(
            {
                "ratio": float(ratio),
                "physically_valid": bool(status.physically_valid),
                "optimizer_ok": bool(status.optimizer_acceptable),
                "equations_converged": bool(status.equations_converged),
                "sonic_regular": bool(status.sonic_regular),
                "optimizer_success": bool(result.optimizer_success),
                "max_residual": float(result.max_residual),
                "interval_radial": float(audit.interval_radial_max),
                "interval_energy": float(audit.interval_energy_max),
                "outer_omega": float(audit.outer_omega),
                "outer_energy": float(audit.outer_energy),
                "sonic_D": float(audit.sonic_D),
                "sonic_C1": float(audit.sonic_C1),
                "sonic_C2": float(audit.sonic_C2),
                "sonic_smin": float(audit.sonic_smin_over_smax),
                "null_radial": float(audit.sonic_null_radial_fraction),
                "M_eff": float(audit.sonic_M_eff),
                "lambda_ratio": float(audit.lambda0_over_lK_isco),
                "outer_HR": float(audit.outer_H_over_R),
                "active_bounds": ",".join(audit.active_bounds) if audit.active_bounds else "none",
                "nfev": int(result.nfev),
                "message": result.message,
                "Rson_rg": np.nan if profile is None else float(profile.sonic_radius / params.r_g),
                "lambda0": np.nan if profile is None else float(profile.lambda0),
                "max_HR": np.nan if profile is None else float(np.max(profile.H_over_R)),
                "adv_frac": np.nan if profile is None else float(profile.integrated_advective_fraction),
                "energy_L1": np.nan if profile is None else float(profile.energy_L1),
                "sonic_crossings": -1 if profile is None else int(profile.sonic_crossings),
            }
        )
        print(
            f"Mdot/Mdot_Edd={ratio:g} physical={status.physically_valid} "
            f"equations={status.equations_converged} sonic={status.sonic_regular} "
            f"lambda/lK={audit.lambda0_over_lK_isco:.3g} max_residual={result.max_residual:.3g} nfev={result.nfev}",
            flush=True,
        )
    return rows


def write_table(rows) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Transonic Slim-Disk Milestone T1 Audit",
        "",
        "Generated by `scripts/plot_transonic_slim_branch.py`.",
        "",
        "This is an isolated, no-wind, pseudo-Newtonian transonic collocation smoke test with analytic local partials, a block-local sparse Jacobian, and a scaled algebraic sonic residual. It is not yet a production branch calculation.",
        "",
        "`physical` is deliberately stricter than a small global residual: it also requires optimizer convergence, separated residual blocks, sonic regularity, no active bounds, outer thinness, and the low-Mdot angular-momentum sanity check.",
        "",
        "For speed, this smoke script may reuse equation-converged rows as initial guesses for later rows. Those rows remain marked nonphysical unless the stricter checks pass.",
        "",
        "| Mdot/Mdot_Edd | physical | equations | sonic | optimizer ok | optimizer raw | max residual | interval R | interval E | outer Omega | outer E | D | C1 | C2 | smin/smax | lambda/lK_ISCO | active bounds | R_son/r_g | l0/(r_g c) | max H/R | sonic crossings | nfev | message |",
        "|---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {ratio:g} | {physical} | {equations} | {sonic} | {optimizer_ok} | {optimizer} | {max_residual:.3g} | "
            "{interval_radial:.3g} | {interval_energy:.3g} | {outer_omega:.3g} | {outer_energy:.3g} | "
            "{sonic_D:.3g} | {sonic_C1:.3g} | {sonic_C2:.3g} | {sonic_smin:.3g} | {lambda_ratio:.3g} | "
            "{active_bounds} | {Rson_rg:.3g} | {lambda0:.3g} | {max_HR:.3g} | {sonic_crossings} | {nfev} | {message} |".format(
                ratio=row["ratio"],
                physical="yes" if row["physically_valid"] else "no",
                equations="yes" if row["equations_converged"] else "no",
                sonic="yes" if row["sonic_regular"] else "no",
                optimizer_ok="yes" if row["optimizer_ok"] else "no",
                optimizer="yes" if row["optimizer_success"] else "no",
                max_residual=row["max_residual"],
                interval_radial=row["interval_radial"],
                interval_energy=row["interval_energy"],
                outer_omega=row["outer_omega"],
                outer_energy=row["outer_energy"],
                sonic_D=row["sonic_D"],
                sonic_C1=row["sonic_C1"],
                sonic_C2=row["sonic_C2"],
                sonic_smin=row["sonic_smin"],
                lambda_ratio=row["lambda_ratio"],
                active_bounds=row["active_bounds"].replace("|", "/"),
                Rson_rg=row["Rson_rg"],
                lambda0=row["lambda0"],
                max_HR=row["max_HR"],
                sonic_crossings=row["sonic_crossings"],
                nfev=row["nfev"],
                message=row["message"].replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The table separates optimizer success, equation residuals, sonic regularity, and physical validity.",
            "- The sonic residual uses smoothly scaled equations and algebraic determinant/adjugate compatibility conditions.",
            "- A low-Mdot row with small residual is still not physically valid if `lambda/lK_ISCO` is far from unity or if the sonic crossing is absent.",
            "- Higher-Mdot failures should not be interpreted physically until the low-Mdot eigenvalue and sonic regularity pass this audit.",
        ]
    )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def draw_figure(rows) -> None:
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1680, 1180
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(42, bold=True)
    subtitle_font = load_font(24)
    label_font = load_font(26, bold=True)
    tick_font = load_font(18)
    note_font = load_font(23)
    axis_color = (42, 47, 56)
    grid_color = (230, 234, 240)
    blue = (36, 112, 177)
    rust = (151, 80, 42)

    draw_text(draw, (80, 52), "Transonic slim-disk solver-hardening audit", title_font)
    draw_text(draw, (80, 104), "Scaled sonic residual; markers show physical validity, not residual-only convergence.", subtitle_font, (78, 84, 96))

    ratios = np.asarray([row["ratio"] for row in rows])
    ok = np.asarray([row["physically_valid"] for row in rows])
    residual = np.asarray([row["max_residual"] for row in rows])
    HR = np.asarray([row["max_HR"] for row in rows])
    adv = np.asarray([row["adv_frac"] for row in rows])
    Rson = np.asarray([row["Rson_rg"] for row in rows])
    x_min, x_max = 1.0e-3, 100.0
    x_ticks = [1.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0]
    boxes = [
        (120, 190, 710, 505),
        (840, 190, 1430, 505),
        (120, 640, 710, 955),
        (840, 640, 1430, 955),
    ]
    panels = [
        ("Max collocation residual", residual, [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 0.1, 1.0], 1.0e-5, 1.0, True),
        ("Max H/R", HR, [0.01, 0.1, 1.0, 10.0], 0.005, 10.0, True),
        ("Integrated advection", adv, [-0.01, 0.0, 0.05, 0.1, 0.2], -0.02, 0.22, False),
        ("Sonic radius / r_g", Rson, [3, 5, 7, 9, 11], 3.0, 12.0, False),
    ]
    for box, (title, values, yticks, ymin, ymax, ylog) in zip(boxes, panels):
        draw_text(draw, (box[0], box[1] - 44), title, label_font)
        draw_axes(draw, box, x_ticks, yticks, x_min, x_max, ymin, ymax, tick_font, axis_color, grid_color, y_log=ylog)
        draw_series(draw, ratios, values, ok, box, x_min, x_max, ymin, ymax, blue, y_log=ylog)
        target_x = map_log(94.2, x_min, x_max, box[0], box[2])
        draw.line((target_x, box[1], target_x, box[3]), fill=rust, width=3)

    legend_x = 1480
    draw_text(draw, (legend_x, 210), "Result", label_font)
    notes = [
        "Physical",
        "validity is",
        "stricter than",
        "max residual.",
        "",
        "Low-Mdot",
        "lambda0 must",
        "approach the",
        "thin-disk",
        "ISCO value.",
    ]
    y = 265
    for line in notes:
        draw_text(draw, (legend_x, y), line, note_font, rust if "not robust" in line else (64, 70, 82))
        y += 32
    draw.ellipse((legend_x, 610, legend_x + 16, 626), fill=blue)
    draw_text(draw, (legend_x + 34, 606), "physical", note_font)
    draw.line((legend_x, 660, legend_x + 18, 678), fill=blue, width=3)
    draw.line((legend_x, 678, legend_x + 18, 660), fill=blue, width=3)
    draw_text(draw, (legend_x + 34, 652), "not physical", note_font)
    draw.line((legend_x, 715, legend_x + 58, 715), fill=rust, width=4)
    draw_text(draw, (legend_x + 70, 702), "QPE target", note_font)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    rows = solve_rows()
    write_table(rows)
    draw_figure(rows)
    print(f"wrote {TABLE_OUTPUT}")
    print(f"wrote {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
