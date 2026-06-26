"""Localize transonic interval residuals for fixed-Mdot checkpoint states."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _differential_interval_residual_from_unpacked,
    _heating_terms_from_gradient,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    algebraic_state,
    scaled_differential_matrix,
    xi_eff_from_gradient,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_interval_residual_profile.md"
FIGURE_OUTPUT = ROOT / "outputs" / "figures" / "transonic_interval_residual_profile.png"


@dataclass(frozen=True)
class CaseSpec:
    label: str
    checkpoint: Path
    note: str


CASES = (
    CaseSpec(
        "N65_root",
        ROOT / "outputs" / "checkpoints" / "transonic_slope_unknown_root" / "n8_medium_0p90277664.npz",
        "good local slope-unknown root",
    ),
    CaseSpec(
        "N67_bridge",
        ROOT / "outputs" / "checkpoints" / "transonic_staged_resolution_continuation" / "N67_branch_polish_0p90277664.npz",
        "first staged-resolution bridge",
    ),
    CaseSpec(
        "N73_bridge",
        ROOT / "outputs" / "checkpoints" / "transonic_staged_resolution_continuation" / "N73_branch_polish_0p90277664.npz",
        "later staged-resolution bridge",
    ),
    CaseSpec(
        "N77_failed_bridge",
        ROOT / "outputs" / "checkpoints" / "transonic_staged_resolution_continuation" / "N77_release_0p90277664.npz",
        "failed staged-resolution bridge",
    ),
    CaseSpec(
        "N81_midpoint_bad",
        ROOT
        / "outputs"
        / "checkpoints"
        / "transonic_trapezoid_collocation_audit"
        / "N81_from_N73_bridge_midpoint_0p90277664.npz",
        "N81 midpoint remap from N73 bridge",
    ),
    CaseSpec(
        "N81_trapezoid_bad",
        ROOT
        / "outputs"
        / "checkpoints"
        / "transonic_trapezoid_collocation_audit"
        / "N81_from_N73_bridge_trapezoid_mixed_0p90277664.npz",
        "N81 mixed-trapezoid audit state",
    ),
    CaseSpec(
        "N129_branch_jump",
        ROOT / "outputs" / "checkpoints" / "transonic_slope_law_nested_grid_audit" / "N129_0p90277664.npz",
        "old nested-grid branch jump",
    ),
)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def load_checkpoint(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), json.loads(str(data["row_json"].item()))


def slopes_from_row(row: dict[str, object]) -> tuple[float, float]:
    g_u = row.get("g_u_solved", row.get("g_u", row.get("g_u_prior")))
    g_T = row.get("g_T_solved", row.get("g_T", row.get("g_T_prior")))
    if g_u is None or g_T is None:
        raise ValueError("checkpoint row has no usable outer slopes")
    return float(g_u), float(g_T)


def params_from_row(
    fiducial: FiducialParams,
    mdot_edd: float,
    z: np.ndarray,
    row: dict[str, object],
) -> TransonicSlimParams:
    n_nodes = int(row.get("n_nodes", (len(z) - 2) // 2))
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(row["ratio"]) * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=n_nodes,
        R_out_rg=float(row["R_out_rg"]),
        residual_tol=1.0e-6,
        max_nfev=650,
        outer_closure="full_slope_match",
        outer_match_log_slopes=slopes_from_row(row),
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def active_physical_max(audit) -> float:
    return max(
        abs(audit.interval_radial_max),
        abs(audit.interval_energy_max),
        abs(audit.outer_omega),
        abs(audit.outer_energy),
        abs(audit.sonic_D),
        abs(audit.sonic_C1),
        abs(audit.sonic_C2),
        abs(audit.sonic_K),
    )


def dominant_interval(row: dict[str, object]) -> str:
    return "interval_R" if abs(float(row["interval_R"])) >= abs(float(row["interval_E"])) else "interval_E"


def interval_rows(label: str, z: np.ndarray, params: TransonicSlimParams) -> list[dict[str, object]]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    rows: list[dict[str, object]] = []
    for idx in range(params.n_nodes - 1):
        dx = float(logR[idx + 1] - logR[idx])
        x_mid = 0.5 * float(logR[idx] + logR[idx + 1])
        y_mid = np.array(
            [
                0.5 * (float(logu[idx]) + float(logu[idx + 1])),
                0.5 * (float(logT[idx]) + float(logT[idx + 1])),
            ],
            dtype=float,
        )
        g_mid = np.array(
            [
                (float(logu[idx + 1]) - float(logu[idx])) / dx,
                (float(logT[idx + 1]) - float(logT[idx])) / dx,
            ],
            dtype=float,
        )
        residual = _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
        local_state = algebraic_state(x_mid, y_mid[0], y_mid[1], lambda0, params)
        q_visc, q_rad, q_adv, energy = _heating_terms_from_gradient(x_mid, y_mid, g_mid, lambda0, params)
        matrix, _rhs, _radial_scale, _energy_scale = scaled_differential_matrix(x_mid, y_mid, lambda0, params)
        try:
            cond_A = float(np.linalg.cond(matrix))
        except Exception:
            cond_A = np.inf
        xi_eff = float(xi_eff_from_gradient(x_mid, y_mid, g_mid, lambda0, params))
        row = {
            "case": label,
            "i": idx,
            "R_mid_rg": float(np.exp(x_mid) / params.r_g),
            "dx": dx,
            "interval_R": float(residual[0]),
            "interval_E": float(residual[1]),
            "interval_abs_max": float(np.max(np.abs(residual))),
            "H_over_R": float(local_state.H_over_R),
            "Qadv_Qvisc": float(q_adv / (q_visc + 1.0e-300)),
            "Qrad_Qvisc": float(q_rad / (q_visc + 1.0e-300)),
            "energy_residual_raw": float(energy),
            "xi_eff": xi_eff,
            "cond_A": cond_A,
        }
        row["dominant_interval"] = dominant_interval(row)
        rows.append(row)
    return rows


def summarize_case(label: str, note: str, z: np.ndarray, params: TransonicSlimParams, rows: list[dict[str, object]]) -> dict[str, object]:
    audit = residual_audit_from_state_vector(z, params)
    profile = profile_from_state_vector(z, params)
    interval_max = max(rows, key=lambda row: float(row["interval_abs_max"]))
    radial_max = max(rows, key=lambda row: abs(float(row["interval_R"])))
    energy_max = max(rows, key=lambda row: abs(float(row["interval_E"])))
    return {
        "case": label,
        "note": note,
        "N": params.n_nodes,
        "physical": active_physical_max(audit),
        "interval_R_max": audit.interval_radial_max,
        "interval_E_max": audit.interval_energy_max,
        "outer_1": audit.outer_omega,
        "outer_2": audit.outer_energy,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "K": audit.sonic_K,
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0": float(profile.lambda0),
        "int_adv": float(profile.integrated_advective_fraction),
        "max_HR": float(np.max(profile.H_over_R)),
        "worst_i": int(interval_max["i"]),
        "worst_R_mid_rg": float(interval_max["R_mid_rg"]),
        "worst_block": str(interval_max["dominant_interval"]),
        "worst_interval_abs": float(interval_max["interval_abs_max"]),
        "radial_i": int(radial_max["i"]),
        "radial_R_mid_rg": float(radial_max["R_mid_rg"]),
        "energy_i": int(energy_max["i"]),
        "energy_R_mid_rg": float(energy_max["R_mid_rg"]),
        "max_cond_A": float(max(float(row["cond_A"]) for row in rows if np.isfinite(float(row["cond_A"])))),
        "median_cond_A": float(np.median([float(row["cond_A"]) for row in rows if np.isfinite(float(row["cond_A"]))])),
        "outer_HR": audit.outer_H_over_R,
    }


def write_table(summaries: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Transonic Interval Residual Profile",
        "",
        "Generated by `scripts/run_transonic_interval_residual_profile.py`.",
        "",
        "Per-interval differential residual localization for the fixed-Mdot `Mdot/Edd ~= 0.9028`, `R_out=6500 rg` family. The table reports midpoint interval residuals and local diagnostics; the figure overlays `|interval_R|` and `|interval_E|` versus radius.",
        "",
        "## Case Summary",
        "",
        "| case | N | physical | interval R | interval E | worst i | worst R/rg | worst block | worst interval | Rson/rg | lambda0 | int adv | max H/R | outer H/R | radial max i | radial max R/rg | energy max i | energy max R/rg | max cond(A) | median cond(A) | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        lines.append(
            "| {case} | {N} | {physical} | {interval_R_max} | {interval_E_max} | {worst_i} | {worst_R_mid_rg} | "
            "{worst_block} | {worst_interval_abs} | {Rson_rg} | {lambda0} | {int_adv} | {max_HR} | {outer_HR} | "
            "{radial_i} | {radial_R_mid_rg} | {energy_i} | {energy_R_mid_rg} | {max_cond_A} | {median_cond_A} | {note} |".format(
                case=summary["case"],
                N=summary["N"],
                physical=fmt(float(summary["physical"])),
                interval_R_max=fmt(float(summary["interval_R_max"])),
                interval_E_max=fmt(float(summary["interval_E_max"])),
                worst_i=summary["worst_i"],
                worst_R_mid_rg=fmt(float(summary["worst_R_mid_rg"])),
                worst_block=summary["worst_block"],
                worst_interval_abs=fmt(float(summary["worst_interval_abs"])),
                Rson_rg=fmt(float(summary["Rson_rg"])),
                lambda0=fmt(float(summary["lambda0"])),
                int_adv=fmt(float(summary["int_adv"])),
                max_HR=fmt(float(summary["max_HR"])),
                outer_HR=fmt(float(summary["outer_HR"])),
                radial_i=summary["radial_i"],
                radial_R_mid_rg=fmt(float(summary["radial_R_mid_rg"])),
                energy_i=summary["energy_i"],
                energy_R_mid_rg=fmt(float(summary["energy_R_mid_rg"])),
                max_cond_A=fmt(float(summary["max_cond_A"])),
                median_cond_A=fmt(float(summary["median_cond_A"])),
                note=summary["note"],
            )
        )
    lines.extend(
        [
            "",
            "## Full Interval Table",
            "",
            "| case | i | R_mid/rg | dx | interval_R | interval_E | abs max | dominant | H/R | Qadv/Qvisc | Qrad/Qvisc | xi_eff | cond(A) |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {i} | {R_mid_rg} | {dx} | {interval_R} | {interval_E} | {interval_abs_max} | "
            "{dominant_interval} | {H_over_R} | {Qadv_Qvisc} | {Qrad_Qvisc} | {xi_eff} | {cond_A} |".format(
                case=row["case"],
                i=row["i"],
                R_mid_rg=fmt(float(row["R_mid_rg"])),
                dx=fmt(float(row["dx"])),
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                interval_abs_max=fmt(float(row["interval_abs_max"])),
                dominant_interval=row["dominant_interval"],
                H_over_R=fmt(float(row["H_over_R"])),
                Qadv_Qvisc=fmt(float(row["Qadv_Qvisc"])),
                Qrad_Qvisc=fmt(float(row["Qrad_Qvisc"])),
                xi_eff=fmt(float(row["xi_eff"])),
                cond_A=fmt(float(row["cond_A"])),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


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


def draw_plot(summaries: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scale = 2
    width, height = 1500 * scale, 1050 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = load_font(30 * scale, bold=True)
    font_axis = load_font(18 * scale, bold=True)
    font_tick = load_font(15 * scale)
    font_note = load_font(16 * scale)
    font_legend = load_font(16 * scale)
    text = (35, 39, 47)
    grid = (222, 226, 232)
    axis = (88, 94, 105)
    colors = {
        "N65_root": (35, 111, 176),
        "N67_bridge": (38, 166, 91),
        "N73_bridge": (243, 156, 18),
        "N77_failed_bridge": (192, 57, 43),
        "N81_midpoint_bad": (142, 68, 173),
        "N81_trapezoid_bad": (22, 160, 133),
        "N129_branch_jump": (87, 96, 111),
    }
    case_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        case_rows.setdefault(str(row["case"]), []).append(row)

    draw.text((70 * scale, 36 * scale), "Interval Residual Localization", font=font_title, fill=text)
    subtitle = "Fixed Mdot/Edd ~= 0.9028, R_out=6500 rg; physical audit uses midpoint differential residuals"
    draw.text((70 * scale, 82 * scale), subtitle, font=font_note, fill=(78, 82, 88))

    panels = [
        ("|interval_R|", "interval_R", (95 * scale, 145 * scale, 1080 * scale, 495 * scale)),
        ("|interval_E|", "interval_E", (95 * scale, 585 * scale, 1080 * scale, 935 * scale)),
    ]
    all_x = np.asarray([float(row["R_mid_rg"]) for row in rows], dtype=float)
    all_y = np.asarray(
        [max(abs(float(row["interval_R"])), abs(float(row["interval_E"]))) for row in rows],
        dtype=float,
    )
    x_min = float(np.floor(np.log10(np.min(all_x[all_x > 0.0]))))
    x_max = float(np.ceil(np.log10(np.max(all_x))))
    y_min = min(-7.0, float(np.floor(np.log10(max(np.min(all_y[all_y > 0.0]), 1.0e-12)))))
    y_max = max(-1.0, float(np.ceil(np.log10(np.max(all_y)))))

    def x_to_px(x_rg: float, box) -> float:
        return box[0] + (np.log10(max(x_rg, 1.0e-300)) - x_min) / (x_max - x_min) * (box[2] - box[0])

    def y_to_px(y_value: float, box) -> float:
        clipped = min(max(abs(y_value), 10.0**y_min), 10.0**y_max)
        return box[3] - (np.log10(clipped) - y_min) / (y_max - y_min) * (box[3] - box[1])

    x_ticks = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    y_ticks = [10.0**power for power in range(int(y_min), int(y_max) + 1)]
    for title, key, box in panels:
        draw.rectangle(box, outline=axis, width=2 * scale)
        draw.text((box[0], box[1] - 36 * scale), title, font=font_axis, fill=text)
        for tick in x_ticks:
            if np.log10(tick) < x_min or np.log10(tick) > x_max:
                continue
            x = x_to_px(tick, box)
            draw.line((x, box[1], x, box[3]), fill=grid, width=1 * scale)
            label = f"{int(tick)}"
            tw = draw.textlength(label, font=font_tick)
            draw.text((x - tw / 2, box[3] + 10 * scale), label, font=font_tick, fill=axis)
        for tick in y_ticks:
            y = y_to_px(tick, box)
            draw.line((box[0], y, box[2], y), fill=grid, width=1 * scale)
            label = f"1e{int(np.log10(tick))}"
            tw = draw.textlength(label, font=font_tick)
            draw.text((box[0] - tw - 10 * scale, y - 8 * scale), label, font=font_tick, fill=axis)
        for case, case_data in case_rows.items():
            points = [
                (
                    x_to_px(float(row["R_mid_rg"]), box),
                    y_to_px(abs(float(row[key])), box),
                )
                for row in case_data
            ]
            if len(points) >= 2:
                draw.line(points, fill=colors.get(case, (0, 0, 0)), width=3 * scale)
        draw.text(((box[0] + box[2]) / 2 - 45 * scale, box[3] + 48 * scale), "R_mid / r_g", font=font_axis, fill=text)

    legend_x = 1130 * scale
    legend_y = 150 * scale
    draw.text((legend_x, legend_y - 42 * scale), "Cases", font=font_axis, fill=text)
    for case, summary in [(str(summary["case"]), summary) for summary in summaries]:
        color = colors.get(case, (0, 0, 0))
        draw.line((legend_x, legend_y, legend_x + 42 * scale, legend_y), fill=color, width=5 * scale)
        draw.text((legend_x + 54 * scale, legend_y - 11 * scale), case, font=font_legend, fill=text)
        y2 = legend_y + 23 * scale
        fact = f"max {fmt(float(summary['worst_interval_abs']))} at R={fmt(float(summary['worst_R_mid_rg']))}"
        draw.text((legend_x + 54 * scale, y2 - 4 * scale), fact, font=font_tick, fill=(78, 82, 88))
        legend_y += 66 * scale

    note_y = 680 * scale
    draw.text((legend_x, note_y), "Reading", font=font_axis, fill=text)
    for line in [
        "N65-N77 interval floors peak near R_out.",
        "N77 physical max is sonic C2, not interval.",
        "N81/N129 failures are inner localized.",
        "N81 trapezoid has a broad energy defect.",
    ]:
        note_y += 30 * scale
        draw.text((legend_x, note_y), line, font=font_note, fill=(78, 82, 88))

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for case in CASES:
        if not case.checkpoint.exists():
            print(f"skip {case.label}: missing {case.checkpoint}", flush=True)
            continue
        z, row = load_checkpoint(case.checkpoint)
        params = params_from_row(fiducial, mdot_edd, z, row)
        rows = interval_rows(case.label, z, params)
        summaries.append(summarize_case(case.label, case.note, z, params, rows))
        all_rows.extend(rows)
        print(
            f"{case.label} N={params.n_nodes} physical={summaries[-1]['physical']:.3e} "
            f"worst={summaries[-1]['worst_interval_abs']:.3e} "
            f"at R={summaries[-1]['worst_R_mid_rg']:.3f} rg",
            flush=True,
        )

    write_table(summaries, all_rows)
    draw_plot(summaries, all_rows)
    print(f"wrote {TABLE_OUTPUT}")
    print(f"wrote {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
