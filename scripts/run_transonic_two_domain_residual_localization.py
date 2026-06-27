"""Residual localization diagnostics for two-domain inner-grid refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _differential_interval_residual_from_unpacked,
    _heating_terms_from_gradient,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    algebraic_state,
    scaled_differential_matrix,
    sonic_diagnostics,
    xi_eff_from_gradient,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_mesh_validation import load_checkpoint, make_params
from run_transonic_two_domain_outer_extension import audit_row, inner_grid, unpack_two_domain


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
COMBINED_PROFILE_TABLE = TABLE_DIR / "transonic_two_domain_inner_residual_profile.md"
SONIC_TABLE = TABLE_DIR / "transonic_two_domain_sonic_scaling.md"
PROFILE_FIGURE = FIGURE_DIR / "transonic_two_domain_inner_residual_profile.png"
SONIC_FIGURE = FIGURE_DIR / "transonic_two_domain_sonic_scaling.png"


@dataclass(frozen=True)
class CaseSpec:
    label: str
    checkpoint: Path
    include_profile: bool
    note: str


CASES = (
    CaseSpec(
        "N65",
        ROOT / "outputs" / "checkpoints" / "transonic_two_domain_mesh_validation" / "outer_N65_O54_R1e5_0p90277664.npz",
        True,
        "pressure-supported two-domain source",
    ),
    CaseSpec(
        "N66",
        ROOT / "outputs" / "checkpoints" / "transonic_two_domain_inner_refinement" / "N66_0p90277664.npz",
        False,
        "first one-node inner refinement",
    ),
    CaseSpec(
        "N67",
        ROOT / "outputs" / "checkpoints" / "transonic_two_domain_inner_refinement" / "N67_0p90277664.npz",
        True,
        "second one-node inner refinement",
    ),
    CaseSpec(
        "N69",
        ROOT / "outputs" / "checkpoints" / "transonic_two_domain_inner_refinement" / "N69_0p90277664.npz",
        False,
        "two-node continuation step",
    ),
    CaseSpec(
        "N73",
        ROOT / "outputs" / "checkpoints" / "transonic_two_domain_inner_refinement" / "N73_0p90277664.npz",
        False,
        "four-node continuation step",
    ),
    CaseSpec(
        "N77",
        ROOT / "outputs" / "checkpoints" / "transonic_two_domain_inner_refinement" / "N77_0p90277664.npz",
        True,
        "late pre-failure staged refinement",
    ),
    CaseSpec(
        "N81",
        ROOT / "outputs" / "checkpoints" / "transonic_two_domain_inner_refinement" / "N81_0p90277664.npz",
        True,
        "first clearly failed staged refinement",
    ),
)


def load_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def line_label(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1.0e-2 and abs(value) < 1.0e4:
        return f"{value:.4g}"
    return f"{value:.3e}"


def case_params(fiducial: FiducialParams, mdot_edd: float, x: np.ndarray, meta: dict[str, object]):
    n_inner = int(meta.get("n_inner", (len(x) - 2) // 4))
    n_outer = int(meta["n_outer"])
    return make_params(
        fiducial,
        float(meta["ratio"]),
        mdot_edd,
        n_inner,
        n_outer,
        float(meta["R_far_rg"]),
    )


def active_physical(row: dict[str, object]) -> float:
    return max(
        float(abs(row["inner_R"])),
        float(abs(row["inner_E"])),
        float(abs(row["outer_R"])),
        float(abs(row["outer_E"])),
        float(abs(row["interface"])),
        float(abs(row["far_omega"])),
        float(abs(row["far_energy"])),
        float(abs(row["D"])),
        float(abs(row["C1"])),
        float(abs(row["C2"])),
        float(abs(row["K"])),
    )


def interval_rows(label: str, x: np.ndarray, params) -> list[dict[str, object]]:
    logu_i, logT_i, _logu_o, _logT_o, logR_son, lambda0 = unpack_two_domain(x, params)
    logR_i = inner_grid(logR_son, params)
    rows: list[dict[str, object]] = []
    for idx in range(params.n_inner - 1):
        dx = float(logR_i[idx + 1] - logR_i[idx])
        x_mid = 0.5 * float(logR_i[idx] + logR_i[idx + 1])
        y_mid = np.array(
            [
                0.5 * (float(logu_i[idx]) + float(logu_i[idx + 1])),
                0.5 * (float(logT_i[idx]) + float(logT_i[idx + 1])),
            ],
            dtype=float,
        )
        g_mid = np.array(
            [
                (float(logu_i[idx + 1]) - float(logu_i[idx])) / dx,
                (float(logT_i[idx + 1]) - float(logT_i[idx])) / dx,
            ],
            dtype=float,
        )
        residual = _differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx)
        state = algebraic_state(x_mid, y_mid[0], y_mid[1], lambda0, params.physics)
        q_visc, q_rad, q_adv, energy = _heating_terms_from_gradient(x_mid, y_mid, g_mid, lambda0, params.physics)
        matrix, _rhs, _radial_scale, _energy_scale = scaled_differential_matrix(x_mid, y_mid, lambda0, params.physics)
        try:
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            smax = float(np.max(singular_values))
            smin = float(np.min(singular_values))
            smin_over_smax = smin / (smax + 1.0e-300)
            cond_A = smax / (smin + 1.0e-300)
        except Exception:
            smin_over_smax = np.nan
            cond_A = np.inf
        xi_eff = float(xi_eff_from_gradient(x_mid, y_mid, g_mid, lambda0, params.physics))
        dominant = "interval_R" if abs(float(residual[0])) >= abs(float(residual[1])) else "interval_E"
        rows.append(
            {
                "case": label,
                "i": idx,
                "R_left_rg": float(np.exp(logR_i[idx]) / params.r_g),
                "R_mid_rg": float(np.exp(x_mid) / params.r_g),
                "R_right_rg": float(np.exp(logR_i[idx + 1]) / params.r_g),
                "distance_from_sonic_dx": float(x_mid - logR_son),
                "dx": dx,
                "interval_R": float(residual[0]),
                "interval_E": float(residual[1]),
                "interval_abs_max": float(np.max(np.abs(residual))),
                "dominant_interval": dominant,
                "cond_A_mid": cond_A,
                "smin_over_smax_A_mid": smin_over_smax,
                "Qadv_Qvisc": float(q_adv / (q_visc + 1.0e-300)),
                "Qrad_Qvisc": float(q_rad / (q_visc + 1.0e-300)),
                "energy_residual_raw": float(energy),
                "H_over_R": float(state.H_over_R),
                "xi_eff": xi_eff,
            }
        )
    return rows


def profile_summary(label: str, note: str, x: np.ndarray, params, rows: list[dict[str, object]]) -> dict[str, object]:
    audit = audit_row(label, x, params)
    radial = max(rows, key=lambda row: abs(float(row["interval_R"])))
    energy = max(rows, key=lambda row: abs(float(row["interval_E"])))
    worst = max(rows, key=lambda row: float(row["interval_abs_max"]))
    finite_cond = [float(row["cond_A_mid"]) for row in rows if np.isfinite(float(row["cond_A_mid"]))]
    audit.update(
        {
            "case": label,
            "note": note,
            "physical_recomputed": active_physical(audit),
            "worst_i": int(worst["i"]),
            "worst_R_mid_rg": float(worst["R_mid_rg"]),
            "worst_interval_abs": float(worst["interval_abs_max"]),
            "worst_interval_block": str(worst["dominant_interval"]),
            "radial_i": int(radial["i"]),
            "radial_R_mid_rg": float(radial["R_mid_rg"]),
            "radial_abs": abs(float(radial["interval_R"])),
            "energy_i": int(energy["i"]),
            "energy_R_mid_rg": float(energy["R_mid_rg"]),
            "energy_abs": abs(float(energy["interval_E"])),
            "max_cond_A_mid": float(max(finite_cond)) if finite_cond else np.nan,
            "median_cond_A_mid": float(np.median(finite_cond)) if finite_cond else np.nan,
        }
    )
    return audit


def sonic_row(label: str, note: str, x: np.ndarray, params) -> dict[str, object]:
    logu_i, logT_i, _logu_o, _logT_o, logR_son, lambda0 = unpack_two_domain(x, params)
    logR_i = inner_grid(logR_son, params)
    audit = audit_row(label, x, params)
    sonic = sonic_diagnostics(logR_son, np.array([logu_i[0], logT_i[0]], dtype=float), lambda0, params.physics)
    first_dx = float(logR_i[1] - logR_i[0]) if len(logR_i) > 1 else np.nan
    second_dx = float(logR_i[2] - logR_i[1]) if len(logR_i) > 2 else np.nan
    row = {
        "case": label,
        "note": note,
        "N_inner": params.n_inner,
        "N_outer": params.n_outer,
        "physical": active_physical(audit),
        "dominant": audit["dominant"],
        "inner_R": audit["inner_R"],
        "inner_E": audit["inner_E"],
        "outer_R": audit["outer_R"],
        "far_omega": audit["far_omega"],
        "far_energy": audit["far_energy"],
        "D": audit["D"],
        "C1": audit["C1"],
        "C2": audit["C2"],
        "K": audit["K"],
        "N_sonic": sonic.N,
        "smin_over_smax": sonic.smin_over_smax,
        "null_radial_fraction": sonic.null_radial_fraction,
        "M_eff": sonic.M_eff,
        "radial_scale": sonic.radial_scale,
        "energy_scale": sonic.energy_scale,
        "first_dx": first_dx,
        "second_dx": second_dx,
        "Rson_rg": audit["Rson_rg"],
        "lambda0": audit["lambda0"],
        "int_adv": audit["int_adv"],
        "max_HR": audit["max_HR"],
    }
    return row


def table_line_for_summary(row: dict[str, object]) -> str:
    return (
        "| {case} | {n_inner} | {n_outer} | {physical} | {dominant} | {inner_R} | {inner_E} | {D} | {C1} | "
        "{C2} | {K} | {worst_i} | {worst_R_mid_rg} | {worst_interval_block} | {worst_interval_abs} | "
        "{radial_i} | {radial_R_mid_rg} | {radial_abs} | {energy_i} | {energy_R_mid_rg} | {energy_abs} | "
        "{max_cond_A_mid} | {median_cond_A_mid} | {Rson_rg} | {lambda0} | {int_adv} | {note} |"
    ).format(
        case=row["case"],
        n_inner=row["n_inner"],
        n_outer=row["n_outer"],
        physical=fmt(float(row["physical_recomputed"])),
        dominant=row["dominant"],
        inner_R=fmt(float(row["inner_R"])),
        inner_E=fmt(float(row["inner_E"])),
        D=fmt(float(row["D"])),
        C1=fmt(float(row["C1"])),
        C2=fmt(float(row["C2"])),
        K=fmt(float(row["K"])),
        worst_i=row["worst_i"],
        worst_R_mid_rg=fmt(float(row["worst_R_mid_rg"])),
        worst_interval_block=row["worst_interval_block"],
        worst_interval_abs=fmt(float(row["worst_interval_abs"])),
        radial_i=row["radial_i"],
        radial_R_mid_rg=fmt(float(row["radial_R_mid_rg"])),
        radial_abs=fmt(float(row["radial_abs"])),
        energy_i=row["energy_i"],
        energy_R_mid_rg=fmt(float(row["energy_R_mid_rg"])),
        energy_abs=fmt(float(row["energy_abs"])),
        max_cond_A_mid=fmt(float(row["max_cond_A_mid"])),
        median_cond_A_mid=fmt(float(row["median_cond_A_mid"])),
        Rson_rg=fmt(float(row["Rson_rg"])),
        lambda0=fmt(float(row["lambda0"])),
        int_adv=fmt(float(row["int_adv"])),
        note=str(row["note"]),
    )


def interval_table_lines(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| case | i | R_left/rg | R_mid/rg | R_right/rg | dx | distance from sonic | interval_R | interval_E | abs max | dominant | condition(scaled A_mid) | smin/smax(scaled A_mid) | Qadv/Qvisc | Qrad/Qvisc | H/R | xi_eff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {i} | {R_left_rg} | {R_mid_rg} | {R_right_rg} | {dx} | {distance_from_sonic_dx} | "
            "{interval_R} | {interval_E} | {interval_abs_max} | {dominant_interval} | {cond_A_mid} | "
            "{smin_over_smax_A_mid} | {Qadv_Qvisc} | {Qrad_Qvisc} | {H_over_R} | {xi_eff} |".format(
                case=row["case"],
                i=row["i"],
                R_left_rg=fmt(float(row["R_left_rg"])),
                R_mid_rg=fmt(float(row["R_mid_rg"])),
                R_right_rg=fmt(float(row["R_right_rg"])),
                dx=fmt(float(row["dx"])),
                distance_from_sonic_dx=fmt(float(row["distance_from_sonic_dx"])),
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                interval_abs_max=fmt(float(row["interval_abs_max"])),
                dominant_interval=row["dominant_interval"],
                cond_A_mid=fmt(float(row["cond_A_mid"])),
                smin_over_smax_A_mid=fmt(float(row["smin_over_smax_A_mid"])),
                Qadv_Qvisc=fmt(float(row["Qadv_Qvisc"])),
                Qrad_Qvisc=fmt(float(row["Qrad_Qvisc"])),
                H_over_R=fmt(float(row["H_over_R"])),
                xi_eff=fmt(float(row["xi_eff"])),
            )
        )
    return lines


def write_profile_tables(summaries: list[dict[str, object]], rows_by_case: dict[str, list[dict[str, object]]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Inner Residual Localization",
        "",
        "Generated by `scripts/run_transonic_two_domain_residual_localization.py`.",
        "",
        "These diagnostics use the existing pressure-supported two-domain checkpoints without re-solving. The residuals are the same midpoint differential residuals used by the two-domain solver, restricted here to the inner domain `R_son -> R_match`.",
        "",
        "## Case Summary",
        "",
        "| case | N inner | N outer | physical | dominant | inner R | inner E | D | C1 | C2 | K | worst i | worst R/rg | worst block | worst interval | radial max i | radial max R/rg | radial abs | energy max i | energy max R/rg | energy abs | max cond(A) | median cond(A) | Rson/rg | lambda0 | int adv | note |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        lines.append(table_line_for_summary(summary))
    lines.extend(["", "## Full Inner Interval Table", ""])
    all_rows: list[dict[str, object]] = []
    for summary in summaries:
        all_rows.extend(rows_by_case[str(summary["case"])])
    lines.extend(interval_table_lines(all_rows))
    COMBINED_PROFILE_TABLE.write_text("\n".join(lines) + "\n")

    for summary in summaries:
        case = str(summary["case"])
        if not any(spec.label == case and spec.include_profile for spec in CASES):
            continue
        per_case = [
            f"# Two-Domain Inner Residual Profile: {case}",
            "",
            "Generated by `scripts/run_transonic_two_domain_residual_localization.py`.",
            "",
            "## Summary",
            "",
            "| case | N inner | N outer | physical | dominant | inner R | inner E | D | C1 | C2 | K | worst i | worst R/rg | worst block | worst interval | radial max i | radial max R/rg | radial abs | energy max i | energy max R/rg | energy abs | max cond(A) | median cond(A) | Rson/rg | lambda0 | int adv | note |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            table_line_for_summary(summary),
            "",
            "## Inner Intervals",
            "",
        ]
        per_case.extend(interval_table_lines(rows_by_case[case]))
        output = TABLE_DIR / f"transonic_two_domain_inner_residual_profile_{case}.md"
        output.write_text("\n".join(per_case) + "\n")


def write_sonic_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Sonic Residual Scaling",
        "",
        "Generated by `scripts/run_transonic_two_domain_residual_localization.py`.",
        "",
        "This table tracks the sonic block as the staged inner grid is refined. The outer configuration is frozen at the pressure-supported two-domain baseline (`R_match=6500 rg`, `R_far=1e5 rg`, `N_outer=54`).",
        "",
        "| case | N_inner | physical | dominant | D | C1 | C2 | K | smin/smax | null radial fraction | first dx | second dx | Rson/rg | lambda0 | int adv | inner R | inner E | outer R | far omega | far energy | M_eff | radial scale | energy scale | note |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {N_inner} | {physical} | {dominant} | {D} | {C1} | {C2} | {K} | {smin_over_smax} | "
            "{null_radial_fraction} | {first_dx} | {second_dx} | {Rson_rg} | {lambda0} | {int_adv} | "
            "{inner_R} | {inner_E} | {outer_R} | {far_omega} | {far_energy} | {M_eff} | {radial_scale} | "
            "{energy_scale} | {note} |".format(
                case=row["case"],
                N_inner=row["N_inner"],
                physical=fmt(float(row["physical"])),
                dominant=row["dominant"],
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                smin_over_smax=fmt(float(row["smin_over_smax"])),
                null_radial_fraction=fmt(float(row["null_radial_fraction"])),
                first_dx=fmt(float(row["first_dx"])),
                second_dx=fmt(float(row["second_dx"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                inner_R=fmt(float(row["inner_R"])),
                inner_E=fmt(float(row["inner_E"])),
                outer_R=fmt(float(row["outer_R"])),
                far_omega=fmt(float(row["far_omega"])),
                far_energy=fmt(float(row["far_energy"])),
                M_eff=fmt(float(row["M_eff"])),
                radial_scale=fmt(float(row["radial_scale"])),
                energy_scale=fmt(float(row["energy_scale"])),
                note=str(row["note"]),
            )
        )
    SONIC_TABLE.write_text("\n".join(lines) + "\n")


def data_range(values: list[float], default: tuple[float, float]) -> tuple[float, float]:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return default
    lo = min(finite)
    hi = max(finite)
    if lo == hi:
        lo *= 0.8
        hi *= 1.2
    return lo, hi


def draw_dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill, width: int, dash: int) -> None:
    for start, end in zip(points[:-1], points[1:]):
        x1, y1 = start
        x2, y2 = end
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length <= 0.0:
            continue
        ux = (x2 - x1) / length
        uy = (y2 - y1) / length
        pos = 0.0
        while pos < length:
            segment_end = min(pos + dash, length)
            draw.line(
                (
                    x1 + ux * pos,
                    y1 + uy * pos,
                    x1 + ux * segment_end,
                    y1 + uy * segment_end,
                ),
                fill=fill,
                width=width,
            )
            pos += 2.0 * dash


def draw_axes(draw, box, x_min, x_max, y_min, y_max, font_tick, font_axis, x_label: str, y_label: str, *, log_x=True, log_y=True):
    text = (35, 39, 47)
    grid = (222, 226, 232)
    axis = (88, 94, 105)
    draw.rectangle(box, outline=axis, width=2)

    def x_to_px(value: float) -> float:
        transformed = np.log10(max(value, 1.0e-300)) if log_x else value
        return box[0] + (transformed - x_min) / (x_max - x_min) * (box[2] - box[0])

    def y_to_px(value: float) -> float:
        clipped = max(abs(value), 10.0**y_min) if log_y else value
        transformed = np.log10(clipped) if log_y else clipped
        return box[3] - (transformed - y_min) / (y_max - y_min) * (box[3] - box[1])

    if log_x:
        powers = range(int(np.floor(x_min)), int(np.ceil(x_max)) + 1)
        x_ticks = [10.0**power for power in powers]
    else:
        x_ticks = np.linspace(x_min, x_max, 5)
    for tick in x_ticks:
        t = np.log10(max(float(tick), 1.0e-300)) if log_x else float(tick)
        if t < x_min or t > x_max:
            continue
        x = x_to_px(float(tick))
        draw.line((x, box[1], x, box[3]), fill=grid, width=1)
        label = f"{int(tick)}" if abs(float(tick) - round(float(tick))) < 1.0e-8 else line_label(float(tick))
        tw = draw.textlength(label, font=font_tick)
        draw.text((x - tw / 2, box[3] + 9), label, font=font_tick, fill=axis)

    if log_y:
        y_ticks = [10.0**power for power in range(int(y_min), int(y_max) + 1)]
    else:
        y_ticks = np.linspace(y_min, y_max, 5)
    for tick in y_ticks:
        y = y_to_px(float(tick))
        draw.line((box[0], y, box[2], y), fill=grid, width=1)
        label = f"1e{int(np.log10(tick))}" if log_y else line_label(float(tick))
        tw = draw.textlength(label, font=font_tick)
        draw.text((box[0] - tw - 8, y - 7), label, font=font_tick, fill=axis)

    tw = draw.textlength(x_label, font=font_axis)
    draw.text(((box[0] + box[2]) / 2 - tw / 2, box[3] + 40), x_label, font=font_axis, fill=text)
    draw.text((box[0], box[1] - 34), y_label, font=font_axis, fill=text)
    return x_to_px, y_to_px


def draw_profile_plot(rows_by_case: dict[str, list[dict[str, object]]], summaries: list[dict[str, object]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    scale = 2
    width, height = 1500 * scale, 980 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = load_font(30 * scale, bold=True)
    font_axis = load_font(18 * scale, bold=True)
    font_tick = load_font(15 * scale)
    font_note = load_font(16 * scale)
    font_legend = load_font(16 * scale)
    text = (35, 39, 47)
    colors = {
        "N65": (35, 111, 176),
        "N67": (38, 166, 91),
        "N77": (192, 57, 43),
        "N81": (142, 68, 173),
    }
    profile_cases = [spec.label for spec in CASES if spec.include_profile and spec.label in rows_by_case]
    all_profile_rows = [row for case in profile_cases for row in rows_by_case[case]]
    all_x = [float(row["R_mid_rg"]) for row in all_profile_rows]
    all_y = [max(abs(float(row["interval_R"])), abs(float(row["interval_E"])), 1.0e-16) for row in all_profile_rows]
    x_min = float(np.floor(np.log10(min(all_x))))
    x_max = float(np.ceil(np.log10(max(all_x))))
    y_min = min(-12.0, float(np.floor(np.log10(min(all_y)))))
    y_max = max(-4.0, float(np.ceil(np.log10(max(all_y)))))

    draw.text((70 * scale, 35 * scale), "Two-Domain Inner Residual Localization", font=font_title, fill=text)
    draw.text(
        (70 * scale, 82 * scale),
        "Solid lines: |interval_R|. Dashed lines: |interval_E|. Same two-domain checkpoints, no re-solve.",
        font=font_note,
        fill=(78, 82, 88),
    )
    box = (105 * scale, 150 * scale, 1090 * scale, 820 * scale)
    x_to_px, y_to_px = draw_axes(
        draw,
        box,
        x_min,
        x_max,
        y_min,
        y_max,
        font_tick,
        font_axis,
        "R_mid / r_g",
        "scaled inner interval residual",
    )
    for case in profile_cases:
        case_rows = rows_by_case[case]
        points_R = [(x_to_px(float(row["R_mid_rg"])), y_to_px(abs(float(row["interval_R"])))) for row in case_rows]
        points_E = [(x_to_px(float(row["R_mid_rg"])), y_to_px(abs(float(row["interval_E"])))) for row in case_rows]
        color = colors.get(case, (0, 0, 0))
        draw.line(points_R, fill=color, width=4 * scale)
        draw_dashed_line(draw, points_E, fill=color, width=3 * scale, dash=7 * scale)

    legend_x = 1140 * scale
    legend_y = 165 * scale
    draw.text((legend_x, legend_y - 46 * scale), "Cases", font=font_axis, fill=text)
    summary_by_case = {str(summary["case"]): summary for summary in summaries}
    for case in profile_cases:
        summary = summary_by_case[case]
        color = colors.get(case, (0, 0, 0))
        draw.line((legend_x, legend_y, legend_x + 44 * scale, legend_y), fill=color, width=5 * scale)
        draw.text((legend_x + 56 * scale, legend_y - 11 * scale), case, font=font_legend, fill=text)
        detail = (
            f"phys {line_label(float(summary['physical_recomputed']))}, "
            f"worst i={summary['worst_i']}"
        )
        draw.text((legend_x + 56 * scale, legend_y + 14 * scale), detail, font=font_tick, fill=(78, 82, 88))
        legend_y += 66 * scale

    note_y = 520 * scale
    draw.text((legend_x, note_y), "Use With", font=font_axis, fill=text)
    notes = [
        "Compare this plot to the sonic table.",
        "If intervals stay tiny while C1 grows,",
        "the refinement failure is endpoint-led.",
    ]
    for note in notes:
        note_y += 30 * scale
        draw.text((legend_x, note_y), note, font=font_note, fill=(78, 82, 88))

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(PROFILE_FIGURE)


def draw_sonic_plot(rows: list[dict[str, object]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    scale = 2
    width, height = 1500 * scale, 980 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = load_font(30 * scale, bold=True)
    font_axis = load_font(18 * scale, bold=True)
    font_tick = load_font(15 * scale)
    font_note = load_font(16 * scale)
    font_legend = load_font(16 * scale)
    text = (35, 39, 47)
    colors = {
        "D": (35, 111, 176),
        "C1": (192, 57, 43),
        "C2": (38, 166, 91),
        "K": (142, 68, 173),
        "inner_R": (243, 156, 18),
    }
    n_values = [float(row["N_inner"]) for row in rows]
    residual_values = [abs(float(row[key])) for row in rows for key in ("D", "C1", "C2", "K", "inner_R")]
    dx_values = [float(row["first_dx"]) for row in rows]
    x_min, x_max = data_range(n_values, (60.0, 85.0))
    x_pad = max(1.0, 0.04 * (x_max - x_min))
    x_min -= x_pad
    x_max += x_pad
    y_min = min(-8.0, float(np.floor(np.log10(max(min(value for value in residual_values if value > 0.0), 1.0e-16)))))
    y_max = max(-3.0, float(np.ceil(np.log10(max(residual_values)))))
    dx_min, dx_max = data_range(dx_values, (0.1, 0.2))
    dx_min *= 0.92
    dx_max *= 1.08

    draw.text((70 * scale, 35 * scale), "Sonic Residual Scaling Under Inner Refinement", font=font_title, fill=text)
    draw.text(
        (70 * scale, 82 * scale),
        "Outer domain frozen at R_match=6500 rg, R_far=1e5 rg, N_outer=54.",
        font=font_note,
        fill=(78, 82, 88),
    )

    top_box = (105 * scale, 150 * scale, 1090 * scale, 500 * scale)
    x_to_px, y_to_px = draw_axes(
        draw,
        top_box,
        x_min,
        x_max,
        y_min,
        y_max,
        font_tick,
        font_axis,
        "N_inner",
        "absolute residual",
        log_x=False,
        log_y=True,
    )
    for key in ("D", "C1", "C2", "K", "inner_R"):
        points = [(x_to_px(float(row["N_inner"])), y_to_px(abs(float(row[key])))) for row in rows]
        draw.line(points, fill=colors[key], width=4 * scale)
        for point in points:
            radius = 4 * scale
            draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=colors[key])

    bottom_box = (105 * scale, 610 * scale, 1090 * scale, 820 * scale)
    x2_to_px, y2_to_px = draw_axes(
        draw,
        bottom_box,
        x_min,
        x_max,
        dx_min,
        dx_max,
        font_tick,
        font_axis,
        "N_inner",
        "first interval dx",
        log_x=False,
        log_y=False,
    )
    dx_points = [(x2_to_px(float(row["N_inner"])), y2_to_px(float(row["first_dx"]))) for row in rows]
    draw.line(dx_points, fill=(88, 94, 105), width=4 * scale)
    for point in dx_points:
        radius = 4 * scale
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=(88, 94, 105))

    legend_x = 1140 * scale
    legend_y = 170 * scale
    draw.text((legend_x, legend_y - 46 * scale), "Sonic Block", font=font_axis, fill=text)
    for key in ("D", "C1", "C2", "K", "inner_R"):
        draw.line((legend_x, legend_y, legend_x + 44 * scale, legend_y), fill=colors[key], width=5 * scale)
        draw.text((legend_x + 56 * scale, legend_y - 11 * scale), key, font=font_legend, fill=text)
        legend_y += 44 * scale

    note_y = 500 * scale
    draw.text((legend_x, note_y), "Reading", font=font_axis, fill=text)
    notes = [
        "C1/C2 grow as dx shrinks.",
        "Inner interval residuals remain",
        "below the sonic block until N81.",
        "This points to endpoint regularity.",
    ]
    for note in notes:
        note_y += 30 * scale
        draw.text((legend_x, note_y), note, font=font_note, fill=(78, 82, 88))

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(SONIC_FIGURE)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows_by_case: dict[str, list[dict[str, object]]] = {}
    profile_summaries: list[dict[str, object]] = []
    sonic_rows: list[dict[str, object]] = []

    for spec in CASES:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"missing checkpoint for {spec.label}: {spec.checkpoint}")
        x, meta = load_checkpoint(spec.checkpoint)
        params = case_params(fiducial, mdot_edd, x, meta)
        rows = interval_rows(spec.label, x, params)
        rows_by_case[spec.label] = rows
        profile_summaries.append(profile_summary(spec.label, spec.note, x, params, rows))
        sonic_rows.append(sonic_row(spec.label, spec.note, x, params))
        print(
            f"{spec.label} N={params.n_inner} physical={sonic_rows[-1]['physical']:.3e} "
            f"dominant={sonic_rows[-1]['dominant']} worst_interval={profile_summaries[-1]['worst_interval_abs']:.3e}",
            flush=True,
        )

    write_profile_tables(profile_summaries, rows_by_case)
    write_sonic_table(sonic_rows)
    draw_profile_plot(rows_by_case, profile_summaries)
    draw_sonic_plot(sonic_rows)
    print(f"wrote {COMBINED_PROFILE_TABLE}")
    print(f"wrote {SONIC_TABLE}")
    print(f"wrote {PROFILE_FIGURE}")
    print(f"wrote {SONIC_FIGURE}")


if __name__ == "__main__":
    main()
