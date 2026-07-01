"""Radial diagnostics for stream-heated standard slim-disk checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    stream_heating_rate,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe


ROOT = Path(__file__).resolve().parents[1]
CASE_SPECS = tuple(
    piece.strip()
    for piece in os.environ.get(
        "IMBH_STANDARD_SLIM_STREAM_HEATING_PROFILE_CASES",
        "eta0:outputs/checkpoints/slim_benchmark_stream_heating_annulus_mdot1_rout300/heat_0_mass_0p03_torque_0p01_mdot_1_N640.npz;"
        "eta10:outputs/checkpoints/slim_benchmark_stream_heating_annulus_mdot1_rout300_high_eta/heat_10_mass_0p03_torque_0p01_mdot_1_N640.npz;"
        "eta30:outputs/checkpoints/slim_benchmark_stream_heating_annulus_mdot1_rout300_high_eta/heat_30_mass_0p03_torque_0p01_mdot_1_N640.npz",
    ).split(";")
    if piece.strip()
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_HEATING_PROFILE_TABLE",
    "outputs/tables/slim_benchmark_stream_heating_profile_diagnostics.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_HEATING_PROFILE_FIGURE",
    "outputs/figures/slim_benchmark_stream_heating_profile_diagnostics.png",
)


def parse_case_specs() -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    for spec in CASE_SPECS:
        if ":" not in spec:
            raise ValueError(f"case spec must be label:path, got {spec!r}")
        label, path = spec.split(":", 1)
        cases.append((label.strip(), ROOT / path.strip()))
    return cases


def optional_float(data, key: str, default: float) -> float:
    if key in data and np.isfinite(float(data[key])):
        return float(data[key])
    return float(default)


def custom_grid_from_data(data) -> tuple[float, ...] | None:
    if "custom_grid_xi" not in data:
        return None
    candidate = np.asarray(data["custom_grid_xi"], dtype=float)
    if candidate.shape == (int(data["n_nodes"]),):
        return tuple(float(value) for value in candidate)
    return None


def params_from_checkpoint(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    data = np.load(path, allow_pickle=True)
    slopes = None
    if "outer_match_log_slopes" in data:
        candidate = np.asarray(data["outer_match_log_slopes"], dtype=float)
        if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
            slopes = (float(candidate[0]), float(candidate[1]))
    params = TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(data["ratio"]) * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=float(data["R_out_rg"]),
        n_nodes=int(data["n_nodes"]),
        grid_power=float(data["grid_power"]) if "grid_power" in data else 1.0,
        custom_grid_xi=custom_grid_from_data(data),
        outer_closure=str(data["outer_closure"]) if "outer_closure" in data else "pressure_supported_thin_energy",
        outer_match_log_slopes=slopes,
        stream_torque_delta_l_fraction=optional_float(data, "stream_torque_delta_l_fraction", 0.0),
        stream_torque_center_fraction=optional_float(data, "stream_torque_center_fraction", 0.8),
        stream_torque_log_width=optional_float(data, "stream_torque_log_width", 0.08),
        stream_mass_fraction=optional_float(data, "stream_mass_fraction", 0.0),
        stream_mass_center_fraction=optional_float(data, "stream_mass_center_fraction", 0.8),
        stream_mass_log_width=optional_float(data, "stream_mass_log_width", 0.08),
        stream_heating_efficiency=optional_float(data, "stream_heating_efficiency", 0.0),
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )
    return np.asarray(data["z"], dtype=float), params


def case_arrays(label: str, path: Path, fiducial: FiducialParams, mdot_edd: float) -> dict[str, Any]:
    z, params = params_from_checkpoint(path, fiducial, mdot_edd)
    profile = transonic_profile_from_state_vector(z, params)
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    q_stream = np.asarray([stream_heating_rate(float(x), params) for x in logR], dtype=float)
    qv = np.asarray(profile.Q_visc, dtype=float)
    qr = np.asarray(profile.Q_rad, dtype=float)
    qa = np.asarray(profile.Q_adv, dtype=float)
    weights = 2.0 * np.pi * profile.R**2
    int_stream = float(np.trapezoid(q_stream * weights, logR))
    int_visc = float(np.trapezoid(np.abs(qv) * weights, logR) + 1.0e-300)
    interval_r = []
    interval_e = []
    interval_R = []
    for idx in range(len(logR) - 1):
        residual = _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
        interval_r.append(float(abs(residual[0])))
        interval_e.append(float(abs(residual[1])))
        interval_R.append(float(np.exp(0.5 * (logR[idx] + logR[idx + 1])) / params.r_g))
    peak_index = int(np.argmax(q_stream))
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "params": params,
        "R_rg": profile.R / params.r_g,
        "T": profile.T,
        "H_R": profile.H_over_R,
        "xi_eff": profile.xi_eff,
        "Qstream_Qvisc": q_stream / (np.abs(qv) + 1.0e-300),
        "Qstream_Qrad": q_stream / (np.abs(qr) + 1.0e-300),
        "Qadv_Qvisc": qa / (np.abs(qv) + 1.0e-300),
        "interval_R_rg": np.asarray(interval_R, dtype=float),
        "interval_radial": np.asarray(interval_r, dtype=float),
        "interval_energy": np.asarray(interval_e, dtype=float),
        "summary": {
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "eta_heat": float(params.stream_heating_efficiency),
            "mass_fraction": float(params.stream_mass_fraction),
            "torque_fraction": float(params.stream_torque_delta_l_fraction),
            "max_Qstream_Qvisc": float(np.max(q_stream / (np.abs(qv) + 1.0e-300))),
            "max_Qstream_Qrad": float(np.max(q_stream / (np.abs(qr) + 1.0e-300))),
            "integrated_Qstream_Qvisc": float(int_stream / int_visc),
            "peak_Qstream_R_rg": float(profile.R[peak_index] / params.r_g),
            "max_H_R": float(np.max(profile.H_over_R)),
            "max_T": float(np.max(profile.T)),
            "integrated_adv": float(profile.integrated_advective_fraction),
            "max_abs_Qadv_Qvisc": float(np.max(np.abs(qa / (np.abs(qv) + 1.0e-300)))),
            "max_interval_R": float(np.max(interval_r)),
            "max_interval_E": float(np.max(interval_e)),
        },
    }


def log_bounds(values: np.ndarray, floor: float = 1.0e-30) -> tuple[float, float]:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > floor)]
    if positive.size == 0:
        return -12.0, 0.0
    lo = float(np.floor(np.log10(np.min(positive))))
    hi = float(np.ceil(np.log10(np.max(positive))))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def lin_bounds(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def write_panel(draw, box, title: str, cases: list[dict[str, Any]], key: str, *, logy: bool = False, interval: bool = False) -> None:
    from PIL import ImageFont

    font = ImageFont.load_default()
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(80, 80, 80), width=1)
    draw.text((x0 + 6, y0 + 5), title, fill=(20, 20, 20), font=font)
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    all_x = np.concatenate([case["interval_R_rg"] if interval else case["R_rg"] for case in cases])
    if interval:
        all_y = np.concatenate([case["interval_radial"] if key == "interval_radial" else case["interval_energy"] for case in cases])
    else:
        all_y = np.concatenate([np.abs(case[key]) if logy else case[key] for case in cases])
    x_lo, x_hi = log_bounds(all_x)
    y_lo, y_hi = log_bounds(all_y) if logy else lin_bounds(all_y)

    def map_x(x):
        return x0 + 45 + int((np.log10(x) - x_lo) / (x_hi - x_lo) * (x1 - x0 - 60))

    def map_y(y):
        yy = np.log10(max(abs(float(y)), 1.0e-30)) if logy else float(y)
        return y1 - 25 - int((yy - y_lo) / (y_hi - y_lo) * (y1 - y0 - 55))

    draw.text((x0 + 5, y1 - 18), f"R 1e{x_lo:.0f}-1e{x_hi:.0f}", fill=(90, 90, 90), font=font)
    draw.text((x0 + 5, y0 + 20), f"y {y_lo:.2g}..{y_hi:.2g}", fill=(90, 90, 90), font=font)
    for idx, case in enumerate(cases):
        x_values = case["interval_R_rg"] if interval else case["R_rg"]
        y_values = case["interval_radial"] if key == "interval_radial" else case["interval_energy"] if interval else case[key]
        points = [(map_x(float(x)), map_y(float(y))) for x, y in zip(x_values, y_values) if np.isfinite(x) and np.isfinite(y)]
        color = colors[idx % len(colors)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        if points:
            draw.ellipse((points[-1][0] - 3, points[-1][1] - 3, points[-1][0] + 3, points[-1][1] + 3), fill=color)
        draw.text((x1 - 90, y0 + 18 + 13 * idx), case["label"], fill=color, font=font)


def write_figure(cases: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    width, height = 1300, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    panels = [
        (50, 50, 630, 310, "Qstream/Qvisc", "Qstream_Qvisc", True, False),
        (690, 50, 1270, 310, "Qadv/Qvisc", "Qadv_Qvisc", False, False),
        (50, 340, 630, 600, "H/R", "H_R", False, False),
        (690, 340, 1270, 600, "Temperature", "T", True, False),
        (50, 630, 630, 880, "Interval radial residual", "interval_radial", True, True),
        (690, 630, 1270, 880, "Interval energy residual", "interval_energy", True, True),
    ]
    for x0, y0, x1, y1, title, key, logy, interval in panels:
        write_panel(draw, (x0, y0, x1, y1), title, cases, key, logy=logy, interval=interval)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def write_table(cases: list[dict[str, Any]]) -> None:
    rows = [case["summary"] for case in cases]
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Stream-Heating Profile Diagnostics",
        "",
        "Generated by `scripts/run_standard_slim_stream_heating_profile_diagnostics.py`.",
        "",
        "| case | eta heat | max Qs/Qv | max Qs/Qrad | int Qs/Qv | peak R/rg | max H/R | max T | int adv | max R res | max E res | checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {label} | {eta_heat} | {max_Qstream_Qvisc} | {max_Qstream_Qrad} | {integrated_Qstream_Qvisc} | "
            "{peak_Qstream_R_rg} | {max_H_R} | {max_T} | {integrated_adv} | {max_interval_R} | {max_interval_E} | {path} |".format(
                **formatted
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps(json_safe(rows), indent=2, sort_keys=True) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    cases = [case_arrays(label, path, fiducial, mdot_edd) for label, path in parse_case_specs()]
    write_table(cases)
    write_figure(cases)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
