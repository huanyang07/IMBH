"""Localize interval residuals for high-source compact stream checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imri_qpe.layer3_minidisk_1d import (
    collocation_residual,
    residual_audit_from_state_vector,
    residual_partition_audit_from_state_vector,
    stream_heating_rate,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    transonic_profile_from_state_vector,
    unpack_state,
    wind_sink_prime,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import fmt, json_safe
from run_standard_slim_stream_anchor_regression import ROOT, dominant, params_from_checkpoint


DEFAULT_CASES = ";".join(
    [
        "fs08338125=outputs/checkpoints/high_mdot_stream_compact_fs08278_to0835_tangent_guarded_N896/"
        "compact_c2_restart_mass_0p8338_torque_0p005_mdot_2_N896.npz",
        "fs0835=outputs/checkpoints/high_mdot_stream_compact_fs08278_to0835_tangent_guarded_N896/"
        "compact_c2_restart_mass_0p835_torque_0p005_mdot_2_N896.npz",
        "fs0836=outputs/checkpoints/high_mdot_stream_compact_fs0835_to0850_tangent_guarded_N896/"
        "compact_c2_to085_mass_0p836_torque_0p005_mdot_2_N896.npz",
        "fs08365=outputs/checkpoints/high_mdot_stream_compact_fs0835_to0850_tangent_guarded_N896/"
        "compact_c2_to085_mass_0p8365_torque_0p005_mdot_2_N896.npz",
    ]
)

CASE_SPECS = tuple(piece.strip() for piece in os.environ.get("IMBH_STREAM_INTERVAL_PROFILE_CASES", DEFAULT_CASES).split(";") if piece.strip())
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STREAM_INTERVAL_PROFILE_TABLE",
    "outputs/tables/high_mdot_stream_compact_interval_profile_fs08338_to08365.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STREAM_INTERVAL_PROFILE_FIGURE",
    "outputs/figures/high_mdot_stream_compact_interval_profile_fs08338_to08365.png",
)
FOCUS_R_MIN_RG = float(os.environ.get("IMBH_STREAM_INTERVAL_PROFILE_RMIN_RG", "220"))
FOCUS_R_MAX_RG = float(os.environ.get("IMBH_STREAM_INTERVAL_PROFILE_RMAX_RG", "300"))


def parse_case_specs() -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    for spec in CASE_SPECS:
        if "=" not in spec:
            raise ValueError("case specs must use label=path")
        label, path = spec.split("=", 1)
        cases.append((label.strip(), ROOT / path.strip()))
    return cases


def interval_rows(label: str, z: np.ndarray, params) -> list[dict[str, Any]]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    rows: list[dict[str, Any]] = []
    qstream_values: list[float] = []
    for idx in range(len(logR) - 1):
        dx = float(logR[idx + 1] - logR[idx])
        x_mid = 0.5 * float(logR[idx] + logR[idx + 1])
        residual = _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
        mdot_mid, dmdot_mid = stream_mass_rate_and_derivative(x_mid, params)
        source_mid = stream_source_prime(x_mid, params)
        wind_mid = wind_sink_prime(x_mid, params)
        qstream = stream_heating_rate(x_mid, params)
        qstream_values.append(float(qstream))
        rows.append(
            {
                "case": label,
                "i": idx,
                "R_mid_rg": float(np.exp(x_mid) / params.r_g),
                "dx": dx,
                "interval_R": float(residual[0]),
                "interval_E": float(residual[1]),
                "abs_interval_E": float(abs(residual[1])),
                "abs_interval_R": float(abs(residual[0])),
                "Mdot_over_inner": float(mdot_mid / params.Mdot_g_s),
                "dMdot_dlnR_over_inner": float(dmdot_mid / params.Mdot_g_s),
                "source_prime_over_inner": float(source_mid / params.Mdot_g_s),
                "wind_prime_over_inner": float(wind_mid / params.Mdot_g_s),
                "Qstream": float(qstream),
            }
        )
    if len(rows) >= 2:
        x_midpoints = np.log(np.asarray([row["R_mid_rg"] * params.r_g for row in rows], dtype=float))
        dqstream = np.gradient(np.asarray(qstream_values, dtype=float), x_midpoints)
    else:
        dqstream = np.zeros(len(rows), dtype=float)
    qscale = max(float(np.max(np.abs(qstream_values))) if qstream_values else 0.0, 1.0e-300)
    for row, derivative in zip(rows, dqstream):
        row["dQstream_dlnR_normalized"] = float(derivative / qscale)
    return rows


def summarize_case(label: str, path: Path, z: np.ndarray, params, rows: list[dict[str, Any]]) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(z, params)
    partition = residual_partition_audit_from_state_vector(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    peak_E = max(rows, key=lambda row: float(row["abs_interval_E"]))
    source_peak = max(rows, key=lambda row: abs(float(row["source_prime_over_inner"])))
    focused = [row for row in rows if FOCUS_R_MIN_RG <= float(row["R_mid_rg"]) <= FOCUS_R_MAX_RG]
    focused_peak = max(focused or rows, key=lambda row: float(row["abs_interval_E"]))
    return {
        "case": label,
        "checkpoint": str(path.relative_to(ROOT)),
        "source_fraction": float(params.stream_source_fraction),
        "source_shape": str(params.stream_source_shape),
        "source_blend": float(params.stream_source_shape_blend),
        "torque_fraction": float(params.stream_torque_delta_l_fraction),
        "N": int(params.n_nodes),
        "full": float(np.max(np.abs(collocation_residual(z, params)))),
        "dominant": dominant(audit),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "partition_buffer_inner_rg": float(partition.buffer_inner_rg),
        "partition_physical_E": float(partition.physical_energy_max),
        "partition_buffer_E": float(partition.buffer_energy_max),
        "partition_terminal_omega": float(partition.terminal_omega),
        "partition_peak_physical_E_rg": float(partition.peak_physical_energy_rg),
        "partition_peak_buffer_E_rg": float(partition.peak_buffer_energy_rg),
        "peak_E_i": int(peak_E["i"]),
        "peak_E_R_rg": float(peak_E["R_mid_rg"]),
        "peak_E": float(peak_E["interval_E"]),
        "focus_peak_E_i": int(focused_peak["i"]),
        "focus_peak_E_R_rg": float(focused_peak["R_mid_rg"]),
        "focus_peak_E": float(focused_peak["interval_E"]),
        "source_peak_R_rg": float(source_peak["R_mid_rg"]),
        "source_peak_value": float(source_peak["source_prime_over_inner"]),
        "source_center_R_rg": float(params.stream_source_center_fraction * params.R_out_rg),
        "Rout_rg": float(params.R_out_rg),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "max_H_R": float(np.max(profile.H_over_R)),
        "f_adv_global": float(profile.integrated_advective_fraction),
    }


def write_table(summaries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Compact Stream Interval Residual Profile",
        "",
        "Generated by `scripts/run_standard_slim_stream_interval_profile.py`.",
        "",
        f"Focused annulus window: `{FOCUS_R_MIN_RG:g}` to `{FOCUS_R_MAX_RG:g}` rg.",
        "",
        "## Summary",
        "",
        "| case | f_s | full | dominant | interval_E | physical E | buffer E | outer_omega | peak E R/rg | peak physical E R/rg | peak buffer E R/rg | source peak R/rg | source center R/rg | N | checkpoint |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        lines.append(
            "| {case} | {source_fraction} | {full} | {dominant} | {interval_E} | {partition_physical_E} | "
            "{partition_buffer_E} | {outer_omega} | {peak_E_R_rg} | {partition_peak_physical_E_rg} | "
            "{partition_peak_buffer_E_rg} | {source_peak_R_rg} | {source_center_R_rg} | {N} | `{checkpoint}` |".format(
                case=summary["case"],
                source_fraction=f"{float(summary['source_fraction']):.9g}",
                full=fmt(float(summary["full"])),
                dominant=summary["dominant"],
                interval_E=fmt(float(summary["interval_E"])),
                partition_physical_E=fmt(float(summary["partition_physical_E"])),
                partition_buffer_E=fmt(float(summary["partition_buffer_E"])),
                outer_omega=fmt(float(summary["outer_omega"])),
                peak_E_R_rg=fmt(float(summary["peak_E_R_rg"])),
                partition_peak_physical_E_rg=fmt(float(summary["partition_peak_physical_E_rg"])),
                partition_peak_buffer_E_rg=fmt(float(summary["partition_peak_buffer_E_rg"])),
                source_peak_R_rg=fmt(float(summary["source_peak_R_rg"])),
                source_center_R_rg=fmt(float(summary["source_center_R_rg"])),
                N=summary["N"],
                checkpoint=summary["checkpoint"],
            )
        )
    lines.extend(
        [
            "",
            "## Focused Interval Rows",
            "",
            "| case | i | R/rg | dx | interval_R | interval_E | source'/Mdot_in | dMdot/dlnR/Mdot_in | dQstream norm | Mdot/Mdot_in |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if not FOCUS_R_MIN_RG <= float(row["R_mid_rg"]) <= FOCUS_R_MAX_RG:
            continue
        lines.append(
            "| {case} | {i} | {R_mid_rg} | {dx} | {interval_R} | {interval_E} | {source_prime_over_inner} | "
            "{dMdot_dlnR_over_inner} | {dQstream_dlnR_normalized} | {Mdot_over_inner} |".format(
                case=row["case"],
                i=row["i"],
                R_mid_rg=fmt(float(row["R_mid_rg"])),
                dx=fmt(float(row["dx"])),
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                source_prime_over_inner=fmt(float(row["source_prime_over_inner"])),
                dMdot_dlnR_over_inner=fmt(float(row["dMdot_dlnR_over_inner"])),
                dQstream_dlnR_normalized=fmt(float(row["dQstream_dlnR_normalized"])),
                Mdot_over_inner=fmt(float(row["Mdot_over_inner"])),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps(json_safe({"summaries": summaries, "rows": rows}), indent=2, sort_keys=True) + "\n")


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


def draw_profile(summaries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scale = 2
    width, height = 1500 * scale, 1100 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = load_font(30 * scale, bold=True)
    font_axis = load_font(17 * scale, bold=True)
    font_tick = load_font(14 * scale)
    font_note = load_font(15 * scale)
    text = (35, 39, 47)
    axis = (86, 94, 105)
    grid = (224, 229, 236)
    colors = [(35, 111, 176), (38, 166, 91), (243, 156, 18), (192, 57, 43), (142, 68, 173)]
    case_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if FOCUS_R_MIN_RG <= float(row["R_mid_rg"]) <= FOCUS_R_MAX_RG:
            case_rows.setdefault(str(row["case"]), []).append(row)

    draw.text((70 * scale, 36 * scale), "Compact Stream Source-Annulus Residual Wall", font=font_title, fill=text)
    draw.text(
        (70 * scale, 82 * scale),
        f"N896 checkpoints, focused on {FOCUS_R_MIN_RG:g}-{FOCUS_R_MAX_RG:g} rg",
        font=font_note,
        fill=(82, 88, 96),
    )

    panels = [
        ("|interval_E|", "abs_interval_E", True, (95 * scale, 145 * scale, 1040 * scale, 390 * scale)),
        ("source prime / Mdot_in", "source_prime_over_inner", False, (95 * scale, 475 * scale, 1040 * scale, 720 * scale)),
        ("grid spacing dx", "dx", False, (95 * scale, 805 * scale, 1040 * scale, 1050 * scale)),
    ]
    x_min, x_max = FOCUS_R_MIN_RG, FOCUS_R_MAX_RG

    def x_px(x: float, box) -> float:
        return box[0] + (x - x_min) / (x_max - x_min) * (box[2] - box[0])

    for title, key, log_y, box in panels:
        values = [abs(float(row[key])) for row in rows if FOCUS_R_MIN_RG <= float(row["R_mid_rg"]) <= FOCUS_R_MAX_RG]
        ymin = min(values) if values else 0.0
        ymax = max(values) if values else 1.0
        if log_y:
            ymin = max(min(value for value in values if value > 0.0), 1.0e-14)
            ymax = max(ymax, ymin * 10.0)
            ylo = float(np.floor(np.log10(ymin)))
            yhi = float(np.ceil(np.log10(ymax)))
            y_ticks = [10.0**power for power in range(int(ylo), int(yhi) + 1)]

            def y_px(value: float, box=box, ylo=ylo, yhi=yhi) -> float:
                return box[3] - (np.log10(max(abs(value), 10.0**ylo)) - ylo) / (yhi - ylo) * (box[3] - box[1])

        else:
            pad = 0.08 * (ymax - ymin if ymax > ymin else max(abs(ymax), 1.0))
            ylo, yhi = ymin - pad, ymax + pad
            y_ticks = np.linspace(ylo, yhi, 5)

            def y_px(value: float, box=box, ylo=ylo, yhi=yhi) -> float:
                return box[3] - (float(value) - ylo) / (yhi - ylo) * (box[3] - box[1])

        draw.rectangle(box, outline=axis, width=2 * scale)
        draw.text((box[0], box[1] - 30 * scale), title, font=font_axis, fill=text)
        for tick in np.linspace(x_min, x_max, 5):
            x = x_px(float(tick), box)
            draw.line((x, box[1], x, box[3]), fill=grid, width=1 * scale)
            label = f"{tick:.0f}"
            tw = draw.textlength(label, font=font_tick)
            draw.text((x - tw / 2, box[3] + 8 * scale), label, font=font_tick, fill=axis)
        for tick in y_ticks:
            y = y_px(float(tick))
            draw.line((box[0], y, box[2], y), fill=grid, width=1 * scale)
            label = f"1e{int(np.log10(tick))}" if log_y else fmt(float(tick))
            tw = draw.textlength(label, font=font_tick)
            draw.text((box[0] - tw - 8 * scale, y - 8 * scale), label, font=font_tick, fill=axis)
        for idx, (case, case_data) in enumerate(case_rows.items()):
            points = [(x_px(float(row["R_mid_rg"]), box), y_px(abs(float(row[key])) if log_y else float(row[key]))) for row in case_data]
            if len(points) >= 2:
                draw.line(points, fill=colors[idx % len(colors)], width=3 * scale)
        draw.text(((box[0] + box[2]) / 2 - 35 * scale, box[3] + 42 * scale), "R / r_g", font=font_axis, fill=text)

    legend_x = 1085 * scale
    legend_y = 160 * scale
    draw.text((legend_x, legend_y - 42 * scale), "Cases", font=font_axis, fill=text)
    for idx, summary in enumerate(summaries):
        color = colors[idx % len(colors)]
        draw.line((legend_x, legend_y, legend_x + 44 * scale, legend_y), fill=color, width=5 * scale)
        draw.text((legend_x + 56 * scale, legend_y - 11 * scale), str(summary["case"]), font=font_note, fill=text)
        detail = f"f_s={float(summary['source_fraction']):.6f}, peak={fmt(float(summary['peak_E_R_rg']))} rg"
        draw.text((legend_x + 56 * scale, legend_y + 14 * scale), detail, font=font_tick, fill=(82, 88, 96))
        legend_y += 62 * scale
    note_y = 510 * scale
    draw.text((legend_x, note_y), "Takeaway", font=font_axis, fill=text)
    if summaries:
        latest = summaries[-1]
        peak_rg = float(latest["peak_E_R_rg"])
        source_peak_rg = float(latest["source_peak_R_rg"])
        source_center_rg = float(latest["source_center_R_rg"])
        outer_rg = float(latest["Rout_rg"])
        if peak_rg > 0.95 * outer_rg:
            takeaway_lines = [
                f"Latest interval_E peak is near",
                f"the outer edge at {peak_rg:.1f} rg.",
                f"Source derivative peaks near",
                f"{source_peak_rg:.1f} rg, center {source_center_rg:.1f} rg.",
                "This points to outer-boundary",
                "energy/source-tail control.",
            ]
        else:
            takeaway_lines = [
                f"Latest interval_E peak is inside",
                f"the source annulus at {peak_rg:.1f} rg.",
                f"Source derivative peaks near",
                f"{source_peak_rg:.1f} rg, center {source_center_rg:.1f} rg.",
                "This points to source-annulus",
                "energy-discretization/mesh control.",
            ]
    else:
        takeaway_lines = ["No checkpoint summaries available."]
    for line in takeaway_lines:
        note_y += 28 * scale
        draw.text((legend_x, note_y), line, font=font_note, fill=(82, 88, 96))

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for label, path in parse_case_specs():
        if not path.exists():
            print(f"skip {label}: missing {path}", flush=True)
            continue
        z, params = params_from_checkpoint(path, fiducial, mdot_edd)
        rows = interval_rows(label, z, params)
        summary = summarize_case(label, path, z, params, rows)
        summaries.append(summary)
        all_rows.extend(rows)
        print(
            f"{label}: f_s={summary['source_fraction']:.7g} full={summary['full']:.3e} "
            f"dom={summary['dominant']} peakE={summary['peak_E_R_rg']:.3f} rg",
            flush=True,
        )
    write_table(summaries, all_rows)
    draw_profile(summaries, all_rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
