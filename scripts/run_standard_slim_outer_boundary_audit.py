"""Audit outer-boundary diagnostics along a standard-slim Mdot ladder."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    pressure_supported_omega_target,
    residual_audit_from_state_vector,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import max_residual, params_for
from run_standard_slim_analytic_seed_audit import fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_OUTER_AUDIT_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_secant_3e4_3e3",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_OUTER_AUDIT_TABLE",
    "outputs/tables/slim_benchmark_outer_boundary_audit.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_OUTER_AUDIT_FIGURE",
    "outputs/figures/slim_benchmark_outer_boundary_audit.png",
)
BRANCH_FILTER = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_OUTER_AUDIT_BRANCHES", "up,down").replace(":", ",").split(",")
    if piece.strip()
)


def checkpoint_paths() -> list[Path]:
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {CHECKPOINT_DIR}")
    return sorted(CHECKPOINT_DIR.glob("*.npz"))


def load_row(path: Path, fiducial: FiducialParams, mdot_edd: float) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    ratio = float(data["ratio"])
    branch = str(np.asarray(data["branch"]).item()) if "branch" in data else "unknown"
    params = params_for(fiducial, mdot_edd, ratio, float(data["R_out_rg"]), int(data["n_nodes"]))
    if "outer_closure" in data:
        outer_closure = str(np.asarray(data["outer_closure"]).item())
        slopes = None
        if "outer_match_log_slopes" in data:
            candidate = np.asarray(data["outer_match_log_slopes"], dtype=float)
            if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                slopes = (float(candidate[0]), float(candidate[1]))
        params = replace(params, outer_closure=outer_closure, outer_match_log_slopes=slopes)
    audit = residual_audit_from_state_vector(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    dx_outer = float(logR[-1] - logR[-2])
    g_outer = np.array([(logu[-1] - logu[-2]) / dx_outer, (logT[-1] - logT[-2]) / dx_outer], dtype=float)
    ln_omega = float(np.log(profile.Omega[-1] / profile.Omega_K[-1]))
    pressure_target = pressure_supported_omega_target(
        float(logR[-1]),
        np.array([logu[-1], logT[-1]], dtype=float),
        g_outer,
        lambda0,
        params,
    )
    qadv_qvisc = np.divide(
        profile.Q_adv,
        profile.Q_visc,
        out=np.full_like(profile.Q_adv, np.nan, dtype=float),
        where=profile.Q_visc != 0.0,
    )
    full = max_residual(z, params)
    row = {
        "path": str(path.relative_to(ROOT)),
        "branch": branch,
        "outer_closure": params.outer_closure,
        "ratio": ratio,
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "full": full,
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "outer_H_R": float(audit.outer_H_over_R),
        "outer_Qadv_Qvisc": float(audit.outer_Qadv_over_Qvisc),
        "outer_lnOmega_OmegaK": ln_omega,
        "outer_pressure_target": float(pressure_target),
        "outer_pressure_residual": float(ln_omega - pressure_target),
        "outer_logu_slope": float(g_outer[0]),
        "outer_logT_slope": float(g_outer[1]),
        "max_H_R": float(np.max(profile.H_over_R)),
        "max_abs_Qadv_Qvisc": float(np.nanmax(np.abs(qadv_qvisc))),
        "integrated_adv": float(profile.integrated_advective_fraction),
        "energy_L1": float(profile.energy_L1),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0_over_lK_isco": float(audit.lambda0_over_lK_isco),
    }
    return row


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Outer-Boundary Audit",
        "",
        f"Checkpoint directory: `{CHECKPOINT_DIR.relative_to(ROOT)}`.",
        "",
        "| branch | closure | Mdot/Edd | full | dominant | outer omega | pressure target | pressure residual | outer energy | outer H/R | outer Qadv/Qvisc | max H/R | max abs Qadv/Qvisc | int adv | Rson/rg | lambda/lK | g_u(out) | g_T(out) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {branch} | {outer_closure} | {ratio} | {full} | {dominant} | {outer_omega} | {outer_pressure_target} | "
            "{outer_pressure_residual} | {outer_energy} | {outer_H_R} | {outer_Qadv_Qvisc} | "
            "{max_H_R} | {max_abs_Qadv_Qvisc} | {integrated_adv} | {Rson_rg} | {lambda0_over_lK_isco} | "
            "{outer_logu_slope} | {outer_logT_slope} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps(json_safe(rows), indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    if not rows:
        return
    width, height = 1100, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = 90, 80, 1040, 560
    draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
    series = [
        ("|outer omega|", "outer_omega", (31, 119, 180)),
        ("|pressure residual|", "outer_pressure_residual", (214, 39, 40)),
        ("outer H/R", "outer_H_R", (44, 160, 44)),
        ("|max Qadv/Qvisc|", "max_abs_Qadv_Qvisc", (148, 103, 189)),
    ]
    ratios = np.asarray([float(row["ratio"]) for row in rows], dtype=float)
    x_values = np.log10(ratios)
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    if x_max <= x_min:
        x_min -= 0.1
        x_max += 0.1
    y_values: list[float] = []
    for _label, key, _color in series:
        for row in rows:
            value = abs(float(row[key])) if "residual" in key or "omega" in key else float(row[key])
            if np.isfinite(value) and value > 0.0:
                y_values.append(value)
    y_log = np.log10(np.maximum(np.asarray(y_values, dtype=float), 1.0e-16))
    y_min, y_max = float(np.floor(np.min(y_log))), float(np.ceil(np.max(y_log)))
    if y_max <= y_min:
        y_max = y_min + 1.0
    for label, key, color in series:
        points = []
        for row in sorted(rows, key=lambda item: float(item["ratio"])):
            value = abs(float(row[key])) if "residual" in key or "omega" in key else float(row[key])
            if not np.isfinite(value) or value <= 0.0:
                continue
            xx = np.log10(float(row["ratio"]))
            yy = np.log10(max(value, 1.0e-16))
            px = x0 + int((xx - x_min) / (x_max - x_min) * (x1 - x0))
            py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
    for idx, (label, _key, color) in enumerate(series):
        y = 25 + 18 * idx
        draw.line((90, y + 6, 130, y + 6), fill=color, width=3)
        draw.text((140, y), label, fill=(20, 20, 20), font=font)
    draw.text((650, 25), "Outer-boundary diagnostics vs log10(Mdot/Edd)", fill=(20, 20, 20), font=font)
    draw.text((x0 + 4, y0 + 4), f"1e{int(y_max)}", fill=(80, 80, 80), font=font)
    draw.text((x0 + 4, y1 - 18), f"1e{int(y_min)}", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows = []
    for path in checkpoint_paths():
        row = load_row(path, fiducial, mdot_edd)
        if BRANCH_FILTER and str(row["branch"]) not in BRANCH_FILTER:
            continue
        rows.append(row)
    rows.sort(key=lambda row: (str(row["branch"]), float(row["ratio"])))
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
