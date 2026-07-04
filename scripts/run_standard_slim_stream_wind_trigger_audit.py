"""Audit the local activation bracket for energy-limited wind on stream anchors."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import algebraic_state, stream_heating_rate, unpack_state
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient
from imri_qpe.layer3_minidisk_1d.winds import q_edd_vertical
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import fmt, json_safe
from run_standard_slim_stream_anchor_regression import params_from_checkpoint


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_WIND_TRIGGER_TABLE",
    "outputs/tables/high_mdot_stream_m5_wind_trigger_audit.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_WIND_TRIGGER_FIGURE",
    "outputs/figures/high_mdot_stream_m5_wind_trigger_audit.png",
)

DEFAULT_ANCHORS: tuple[tuple[str, str], ...] = (
    (
        "m5_fs080_heat0",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_heating_scout_eta0_to1e3_N896/"
        "m5heat_scout_mass_0p8_heat_0_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "m5_fs080_heat0p1",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_heating_eta3e2_to1e1_N896/"
        "m5heat_aggressive_mass_0p8_heat_0p1_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "m5_fs080_heat1",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_heating_eta01_to1_N896/"
        "m5heat_eta01_to1_mass_0p8_heat_1_torque_0p005_mdot_5_N896.npz",
    ),
)


def configured_anchors() -> tuple[tuple[str, Path], ...]:
    raw = os.environ.get("IMBH_STANDARD_SLIM_STREAM_WIND_TRIGGER_ANCHORS", "").strip()
    if not raw:
        return tuple((label, ROOT / path) for label, path in DEFAULT_ANCHORS)
    anchors: list[tuple[str, Path]] = []
    for piece in raw.split(";"):
        if not piece.strip():
            continue
        if "=" not in piece:
            raise ValueError("anchor specs must use label=path")
        label, path = piece.split("=", 1)
        anchors.append((label.strip(), ROOT / path.strip()))
    return tuple(anchors)


def interval_profile(z: np.ndarray, params) -> dict[str, np.ndarray]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    rows: dict[str, list[float]] = {
        "R_rg": [],
        "Q_visc": [],
        "Q_stream": [],
        "Q_rad": [],
        "Q_adv": [],
        "Q_avail": [],
        "Q_edd_z": [],
        "trigger": [],
        "trigger_over_Qvisc_abs": [],
        "Qavail_over_Qedd": [],
        "Qrad_over_Qedd": [],
        "Qvisc_over_Qedd": [],
    }
    for idx in range(len(logR) - 1):
        dx = float(logR[idx + 1] - logR[idx])
        xm = float(0.5 * (logR[idx] + logR[idx + 1]))
        ym = np.array([0.5 * (logu[idx] + logu[idx + 1]), 0.5 * (logT[idx] + logT[idx + 1])], dtype=float)
        gm = np.array([(logu[idx + 1] - logu[idx]) / dx, (logT[idx + 1] - logT[idx]) / dx], dtype=float)
        qv, qr, qa, _qe = _heating_terms_from_gradient(xm, ym, gm, lambda0, params)
        state = algebraic_state(xm, float(ym[0]), float(ym[1]), lambda0, params)
        qs = stream_heating_rate(xm, params)
        qedd = float(q_edd_vertical(state.Omega_K, state.H, kappa=params.kappa))
        qavail = float(qv + qs - qa)
        trigger = float(qavail - qedd)
        rows["R_rg"].append(float(np.exp(xm) / params.r_g))
        rows["Q_visc"].append(float(qv))
        rows["Q_stream"].append(float(qs))
        rows["Q_rad"].append(float(qr))
        rows["Q_adv"].append(float(qa))
        rows["Q_avail"].append(float(qavail))
        rows["Q_edd_z"].append(float(qedd))
        rows["trigger"].append(trigger)
        rows["trigger_over_Qvisc_abs"].append(float(trigger / (abs(qv) + 1.0e-300)))
        rows["Qavail_over_Qedd"].append(float(qavail / (qedd + 1.0e-300)))
        rows["Qrad_over_Qedd"].append(float(qr / (qedd + 1.0e-300)))
        rows["Qvisc_over_Qedd"].append(float(qv / (qedd + 1.0e-300)))
    return {key: np.asarray(value, dtype=float) for key, value in rows.items()}


def trapz_interval(values: np.ndarray, R_rg: np.ndarray, params) -> float:
    R = np.asarray(R_rg, dtype=float) * params.r_g
    weights = 2.0 * np.pi * R**2
    return float(np.trapezoid(weights * np.asarray(values, dtype=float), np.log(R)))


def row_for_anchor(label: str, path: Path, z: np.ndarray, params) -> dict[str, Any]:
    profile = interval_profile(z, params)
    trigger = profile["trigger"]
    qavail = profile["Q_avail"]
    qedd = profile["Q_edd_z"]
    qv = profile["Q_visc"]
    qr = profile["Q_rad"]
    active = trigger > 0.0
    peak_trigger_idx = int(np.argmax(trigger))
    peak_ratio_idx = int(np.argmax(profile["Qavail_over_Qedd"]))
    int_avail = trapz_interval(qavail, profile["R_rg"], params)
    int_edd = trapz_interval(qedd, profile["R_rg"], params)
    int_trigger_positive = trapz_interval(np.maximum(trigger, 0.0), profile["R_rg"], params)
    int_visc_abs = trapz_interval(np.abs(qv), profile["R_rg"], params) + 1.0e-300
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "ratio": float(params.mdot_edd_ratio),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "source_fraction": float(params.stream_source_fraction),
        "heat_eta": float(params.stream_heating_efficiency),
        "wind_energy_limited_epsilon": float(params.wind_energy_limited_epsilon),
        "max_trigger": float(np.max(trigger)),
        "max_trigger_over_Qvisc_abs": float(np.max(profile["trigger_over_Qvisc_abs"])),
        "max_Qavail_over_Qedd": float(np.max(profile["Qavail_over_Qedd"])),
        "median_Qavail_over_Qedd": float(np.median(profile["Qavail_over_Qedd"])),
        "integrated_Qavail_over_Qedd": float(int_avail / (int_edd + 1.0e-300)),
        "integrated_positive_trigger_Qvisc": float(int_trigger_positive / int_visc_abs),
        "active_interval_fraction": float(np.count_nonzero(active) / max(trigger.size, 1)),
        "peak_trigger_R_rg": float(profile["R_rg"][peak_trigger_idx]),
        "peak_ratio_R_rg": float(profile["R_rg"][peak_ratio_idx]),
        "peak_Qvisc_over_Qedd": float(profile["Qvisc_over_Qedd"][peak_ratio_idx]),
        "peak_Qrad_over_Qedd": float(profile["Qrad_over_Qedd"][peak_ratio_idx]),
        "peak_Qadv_over_Qvisc": float(profile["Q_adv"][peak_ratio_idx] / (abs(qv[peak_ratio_idx]) + 1.0e-300)),
        "peak_Qstream_over_Qvisc": float(profile["Q_stream"][peak_ratio_idx] / (abs(qv[peak_ratio_idx]) + 1.0e-300)),
        "min_Qavail_over_Qedd": float(np.min(profile["Qavail_over_Qedd"])),
        "max_Qrad_over_Qedd": float(np.max(qr / (qedd + 1.0e-300))),
        "profile": profile,
    }


def write_table(rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Stream Wind-Trigger Audit",
        "",
        "Generated by `scripts/run_standard_slim_stream_wind_trigger_audit.py`.",
        "",
        "Trigger quantity:",
        "",
        "```text",
        "Q_trigger = Q_visc + Q_stream - Q_adv - Q_Edd,z",
        "Q_Edd,z = 2 c Omega_K^2 H / kappa",
        "```",
        "",
        "| anchor | Mdot/Edd | Rout/rg | N | f_s | eta_heat | max Qavail/Qedd | median Qavail/Qedd | int Qavail/Qedd | max trigger/Qvisc | positive trigger/Qvisc | active frac | peak ratio R/rg | Qvisc/Qedd at peak | Qrad/Qedd at peak | Qadv/Qvisc at peak | Qstream/Qvisc at peak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        display = {
            key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value
            for key, value in row.items()
            if key != "profile"
        }
        lines.append(
            "| {label} | {ratio} | {R_out_rg} | {N} | {source_fraction} | {heat_eta} | "
            "{max_Qavail_over_Qedd} | {median_Qavail_over_Qedd} | {integrated_Qavail_over_Qedd} | "
            "{max_trigger_over_Qvisc_abs} | {integrated_positive_trigger_Qvisc} | {active_interval_fraction} | "
            "{peak_ratio_R_rg} | {peak_Qvisc_over_Qedd} | {peak_Qrad_over_Qedd} | "
            "{peak_Qadv_over_Qvisc} | {peak_Qstream_over_Qvisc} |".format(**display)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(
        json.dumps(json_safe([{key: value for key, value in row.items() if key != "profile"} for row in rows]), indent=2, sort_keys=True)
        + "\n"
    )


def _polyline(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in points]


def write_figure(rows: list[dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    if not rows:
        return
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1100, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = 95, 80, 1015, 595
    draw.rectangle((x0, y0, x1, y1), outline=(70, 70, 70), width=2)

    all_R = np.concatenate([row["profile"]["R_rg"] for row in rows])
    all_y = np.concatenate([np.maximum(row["profile"]["Qavail_over_Qedd"], 1.0e-12) for row in rows])
    xmin, xmax = float(np.log10(np.min(all_R))), float(np.log10(np.max(all_R)))
    ymin, ymax = float(np.floor(np.log10(np.min(all_y)))), float(np.ceil(np.log10(max(np.max(all_y), 1.0))))
    ymin = min(ymin, -8.0)
    ymax = max(ymax, 0.0)
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]

    def map_x(R_rg: np.ndarray) -> np.ndarray:
        return x0 + (np.log10(R_rg) - xmin) / (xmax - xmin) * (x1 - x0)

    def map_y(values: np.ndarray) -> np.ndarray:
        ylog = np.log10(np.maximum(values, 1.0e-12))
        return y1 - (ylog - ymin) / (ymax - ymin) * (y1 - y0)

    for tick in range(int(np.ceil(xmin)), int(np.floor(xmax)) + 1):
        x = x0 + (tick - xmin) / (xmax - xmin) * (x1 - x0)
        draw.line((x, y0, x, y1), fill=(235, 235, 235))
        draw.text((x - 12, y1 + 8), f"1e{tick}", fill=(40, 40, 40), font=font)
    for tick in range(int(ymin), int(ymax) + 1):
        y = y1 - (tick - ymin) / (ymax - ymin) * (y1 - y0)
        draw.line((x0, y, x1, y), fill=(235, 235, 235))
        draw.text((25, y - 5), f"1e{tick}", fill=(40, 40, 40), font=font)
    y_one = map_y(np.asarray([1.0]))[0]
    draw.line((x0, y_one, x1, y_one), fill=(0, 0, 0), width=2)
    draw.text((x1 - 110, y_one - 18), "activation", fill=(0, 0, 0), font=font)

    for idx, row in enumerate(rows):
        profile = row["profile"]
        color = colors[idx % len(colors)]
        points = list(zip(map_x(profile["R_rg"]), map_y(profile["Qavail_over_Qedd"])))
        if len(points) > 1:
            draw.line(_polyline(points), fill=color, width=3)
        draw.rectangle((x0 + 15, y0 + 18 + 22 * idx, x0 + 30, y0 + 30 + 22 * idx), fill=color)
        draw.text((x0 + 38, y0 + 16 + 22 * idx), str(row["label"]), fill=(30, 30, 30), font=font)

    draw.text((x0, 28), "Energy-limited wind trigger: Qavail / Qedd,z", fill=(20, 20, 20), font=font)
    draw.text((430, height - 45), "R / rg", fill=(20, 20, 20), font=font)
    draw.text((15, 40), "Qavail/Qedd,z", fill=(20, 20, 20), font=font)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows: list[dict[str, Any]] = []
    for label, path in configured_anchors():
        z, params = params_from_checkpoint(path, fiducial, mdot_edd)
        row = row_for_anchor(label, path, z, params)
        rows.append(row)
        print(
            f"{label}: max_Qavail/Qedd={row['max_Qavail_over_Qedd']:.3e} "
            f"active={row['active_interval_fraction']:.3g} peak_R={row['peak_ratio_R_rg']:.3g}",
            flush=True,
        )
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
