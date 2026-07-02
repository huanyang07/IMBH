"""Advection and luminosity diagnostics for high-Mdot no-wind slim disks."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    residual_audit_from_state_vector,
    transonic_profile_from_state_vector,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_luminosity, eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_HIGH_MDOT_DIAG_ANCHOR",
    "outputs/checkpoints/slim_benchmark_adaptive_outer_mesh_mdot1_scan/s8_mdot_1_N640.npz",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_HIGH_MDOT_DIAG_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_high_mdot_no_wind_ladder",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_HIGH_MDOT_DIAG_TABLE",
    "outputs/tables/slim_benchmark_high_mdot_advection_diagnostics.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_HIGH_MDOT_DIAG_FIGURE",
    "outputs/figures/slim_benchmark_high_mdot_advection_diagnostics.png",
)
INNER_RADIUS_RG = float(os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_DIAG_INNER_RG", "20.0"))
CASE_SPECS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_HIGH_MDOT_DIAG_CASES", "").replace(";", ",").split(",")
    if piece.strip()
)


def custom_grid_from_data(data) -> tuple[float, ...] | None:
    if "custom_grid_xi" not in data:
        return None
    candidate = np.asarray(data["custom_grid_xi"], dtype=float)
    if candidate.shape == (int(data["n_nodes"]),) and candidate.size > 0:
        return tuple(float(value) for value in candidate)
    return None


def optional_float(data, key: str, default: float) -> float:
    if key in data and np.isfinite(float(data[key])):
        return float(data[key])
    return float(default)


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
        outer_closure=str(np.asarray(data["outer_closure"]).item()) if "outer_closure" in data else "thin_value",
        outer_match_log_slopes=slopes,
        residual_tol=1.0e-8,
        max_nfev=1,
        outer_omega_log_offset=optional_float(data, "outer_omega_log_offset", 0.0),
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )
    return np.asarray(data["z"], dtype=float), params


def parse_case_specs() -> list[tuple[str, Path]]:
    if CASE_SPECS:
        cases: list[tuple[str, Path]] = []
        for spec in CASE_SPECS:
            if ":" not in spec:
                raise ValueError(f"case spec must be label:path, got {spec!r}")
            label, path = spec.split(":", 1)
            cases.append((label.strip(), ROOT / path.strip()))
        return cases

    cases = [("mdot1_anchor", ANCHOR_CHECKPOINT)]
    if CHECKPOINT_DIR.exists():
        candidates: list[tuple[float, Path]] = []
        for path in CHECKPOINT_DIR.glob("*.npz"):
            try:
                data = np.load(path, allow_pickle=True)
                ratio = float(data["ratio"])
            except Exception:
                continue
            candidates.append((ratio, path))
        seen = {1.0}
        for ratio, path in sorted(candidates):
            rounded = round(ratio, 10)
            if rounded in seen:
                continue
            seen.add(rounded)
            cases.append((f"mdot{ratio:.6g}", path))
    return cases


def trapz_log(values: np.ndarray, R: np.ndarray) -> float:
    logR = np.log(np.asarray(R, dtype=float))
    weights = 2.0 * np.pi * np.asarray(R, dtype=float) ** 2
    return float(np.trapezoid(np.asarray(values, dtype=float) * weights, logR))


def masked_trapz_log(values: np.ndarray, R: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    R = np.asarray(R, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if int(np.count_nonzero(mask)) < 2:
        return np.nan
    return trapz_log(values[mask], R[mask])


def case_row(label: str, path: Path, fiducial: FiducialParams, mdot_edd: float) -> dict[str, Any]:
    z, params = params_from_checkpoint(path, fiducial, mdot_edd)
    profile = transonic_profile_from_state_vector(z, params)
    audit = residual_audit_from_state_vector(z, params)
    full = float(np.max(np.abs(collocation_residual(z, params))))
    R = np.asarray(profile.R, dtype=float)
    R_rg = R / params.r_g
    qv = np.asarray(profile.Q_visc, dtype=float)
    qr = np.asarray(profile.Q_rad, dtype=float)
    qa = np.asarray(profile.Q_adv, dtype=float)
    visc = trapz_log(np.abs(qv), R) + 1.0e-300
    visc_signed = trapz_log(qv, R)
    rad = trapz_log(qr, R)
    adv = trapz_log(qa, R)
    adv_pos = trapz_log(np.maximum(qa, 0.0), R)
    adv_abs = trapz_log(np.abs(qa), R)
    inner = R_rg <= INNER_RADIUS_RG
    inner_visc = masked_trapz_log(np.abs(qv), R, inner)
    inner_adv = masked_trapz_log(qa, R, inner)
    inner_adv_pos = masked_trapz_log(np.maximum(qa, 0.0), R, inner)
    ledd = eddington_luminosity(params.M2_g, kappa=params.kappa)
    max_hr_index = int(np.argmax(profile.H_over_R))
    max_qadv_index = int(np.argmax(np.abs(qa / (np.abs(qv) + 1.0e-300))))
    return {
        "label": label,
        "checkpoint": str(path.relative_to(ROOT)),
        "ratio": float(params.mdot_edd_ratio),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "full": full,
        "accepted": bool(full <= 1.0e-5),
        "anchor": bool(full <= 3.0e-6),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0": float(profile.lambda0),
        "lambda0_over_lK_isco": float(audit.lambda0_over_lK_isco),
        "max_H_R": float(np.max(profile.H_over_R)),
        "R_max_H_R_rg": float(R_rg[max_hr_index]),
        "min_tau": float(np.min(profile.tau)),
        "sonic_crossings": int(profile.sonic_crossings),
        "f_adv_global": float(adv / visc),
        "f_adv_pos": float(adv_pos / visc),
        "f_adv_abs": float(adv_abs / visc),
        "f_adv_inner": float(inner_adv / (inner_visc + 1.0e-300)) if np.isfinite(inner_visc) else np.nan,
        "f_adv_inner_pos": float(inner_adv_pos / (inner_visc + 1.0e-300)) if np.isfinite(inner_visc) else np.nan,
        "Lrad_LEdd": float(rad / ledd),
        "Lvisc_LEdd": float(visc_signed / ledd),
        "Lvisc_abs_LEdd": float(visc / ledd),
        "Ladv_LEdd": float(adv / ledd),
        "max_abs_Qadv_Qvisc": float(np.max(np.abs(qa / (np.abs(qv) + 1.0e-300)))),
        "R_max_abs_Qadv_Qvisc_rg": float(R_rg[max_qadv_index]),
        "integrated_adv_profile": float(profile.integrated_advective_fraction),
    }


def write_table(rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim High-Mdot Advection Diagnostics",
        "",
        "Generated by `scripts/run_standard_slim_high_mdot_advection_diagnostics.py`.",
        "",
        f"Inner diagnostic radius: `{INNER_RADIUS_RG:g} rg`.",
        "",
        "| label | Mdot/Edd | full | accepted | anchor | dominant | f_adv global | f_adv inner | f_adv pos | Lrad/LEdd | Lvisc/LEdd | max H/R | min tau | Rson/rg | lambda0/lK | max |Qadv/Qvisc| | R max adv/rg | checkpoint |",
        "|---|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {label} | {ratio} | {full} | {accepted} | {anchor} | {dominant} | {f_adv_global} | {f_adv_inner} | "
            "{f_adv_pos} | {Lrad_LEdd} | {Lvisc_LEdd} | {max_H_R} | {min_tau} | {Rson_rg} | "
            "{lambda0_over_lK_isco} | {max_abs_Qadv_Qvisc} | {R_max_abs_Qadv_Qvisc_rg} | {checkpoint} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps(json_safe(rows), indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    if not rows:
        return

    image = Image.new("RGB", (1200, 760), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    panels = [
        (70, 60, 560, 350, "Advective fractions", ("f_adv_global", "f_adv_inner", "f_adv_pos"), False),
        (650, 60, 1140, 350, "Luminosities / L_Edd", ("Lrad_LEdd", "Lvisc_LEdd"), True),
        (70, 420, 560, 700, "Max H/R", ("max_H_R",), False),
        (650, 420, 1140, 700, "Residual", ("full",), True),
    ]
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    x = np.asarray([float(row["ratio"]) for row in rows], dtype=float)
    xlog = np.log10(x)
    x_min = float(np.min(xlog))
    x_max = float(np.max(xlog))
    if x_max <= x_min:
        x_min -= 0.05
        x_max += 0.05

    for x0, y0, x1, y1, title, keys, logy in panels:
        draw.rectangle((x0, y0, x1, y1), outline=(70, 70, 70), width=1)
        draw.text((x0 + 8, y0 + 6), title, fill=(20, 20, 20), font=font)
        values = []
        for key in keys:
            values.extend([abs(float(row[key])) for row in rows if np.isfinite(float(row[key]))])
        if not values:
            continue
        if logy:
            y_values = np.log10(np.maximum(np.asarray(values, dtype=float), 1.0e-16))
            y_min = float(np.floor(np.min(y_values)))
            y_max = float(np.ceil(np.max(y_values)))
        else:
            y_min = float(min(0.0, np.min(values)))
            y_max = float(np.max(values))
            pad = 0.1 * max(y_max - y_min, 1.0e-8)
            y_min -= pad
            y_max += pad
        if y_max <= y_min:
            y_max = y_min + 1.0

        def map_x(value: float) -> int:
            return x0 + 45 + int((np.log10(value) - x_min) / (x_max - x_min) * (x1 - x0 - 70))

        def map_y(value: float) -> int:
            yy = np.log10(max(abs(value), 1.0e-16)) if logy else value
            return y1 - 30 - int((yy - y_min) / (y_max - y_min) * (y1 - y0 - 70))

        for idx, key in enumerate(keys):
            color = colors[idx % len(colors)]
            pts = [(map_x(float(row["ratio"])), map_y(float(row[key]))) for row in rows if np.isfinite(float(row[key]))]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)
            for px, py in pts:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
            draw.text((x1 - 140, y0 + 22 + 14 * idx), key, fill=color, font=font)
        draw.text((x0 + 5, y1 - 18), "log Mdot/Edd", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows = [case_row(label, path, fiducial, mdot_edd) for label, path in parse_case_specs()]
    rows.sort(key=lambda row: float(row["ratio"]))
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
