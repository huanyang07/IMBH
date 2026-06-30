"""Fixed-eigenvalue profile solve for the standard no-wind slim benchmark."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    residual_audit_from_state_vector,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _pack_profile_unknowns_to_state,
    _profile_jac_sparsity_pattern,
    _profile_jacobian_from_unknowns,
    _profile_residual_from_unknowns,
    _profile_unknown_bounds,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import (
    ALPHA,
    MDOT_RATIO,
    PRIMARY_STRESS_FACTOR,
    R_SON_RG,
    analytic_seed_state,
    fmt,
    json_safe,
    n_nodes_for,
    pressure_support_diagnostics,
)
from run_standard_slim_benchmark_thin_limit import thin_metrics


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_FIXED_EIGEN_TABLE",
    "outputs/tables/slim_benchmark_fixed_eigen_profile.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_FIXED_EIGEN_FIGURE",
    "outputs/figures/slim_benchmark_fixed_eigen_profile.png",
)

R_OUT_CASES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_FIXED_EIGEN_ROUTS", "300,1000").replace(":", ",").split(",")
    if piece.strip()
)
STRESS_FACTORS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_FIXED_EIGEN_STRESS_FACTORS", "1.0,1.5").replace(":", ",").split(",")
    if piece.strip()
)
SOLVE_FORM = os.environ.get("IMBH_STANDARD_SLIM_FIXED_EIGEN_SOLVE_FORM", "integrated")
SOLVE_WEIGHTING = os.environ.get("IMBH_STANDARD_SLIM_FIXED_EIGEN_SOLVE_WEIGHTING", "inverse_sqrt_dx")
RUN_DIFFERENTIAL_POLISH = os.environ.get("IMBH_STANDARD_SLIM_FIXED_EIGEN_DIFFERENTIAL_POLISH", "0") != "0"
MAX_NFEV_INTEGRATED = int(os.environ.get("IMBH_STANDARD_SLIM_FIXED_EIGEN_MAX_NFEV_INTEGRATED", "100"))
MAX_NFEV_DIFFERENTIAL = int(os.environ.get("IMBH_STANDARD_SLIM_FIXED_EIGEN_MAX_NFEV_DIFFERENTIAL", "100"))
RESIDUAL_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_FIXED_EIGEN_RESIDUAL_TOL", "1e-7"))
USE_BLOCK_JACOBIAN = os.environ.get("IMBH_STANDARD_SLIM_FIXED_EIGEN_BLOCK_JACOBIAN", "0") != "0"


def make_params(fiducial: FiducialParams, mdot_edd: float, R_out_rg: float, stress_factor: float, *, form: str, weighting: str) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=MDOT_RATIO * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=float(stress_factor),
        R_out_rg=float(R_out_rg),
        n_nodes=n_nodes_for(float(R_out_rg)),
        max_nfev=max(MAX_NFEV_INTEGRATED, MAX_NFEV_DIFFERENTIAL),
        residual_tol=RESIDUAL_TOL,
        outer_closure="thin_value",
        interval_residual_form=form,
        integrated_residual_weighting=weighting,
    )


def profile_unknowns_from_state(z: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    logu, logT, _logR_son, _lambda0, _logR = unpack_state(z, params)
    return np.concatenate([logu, logT])


def solve_profile_stage(
    x0: np.ndarray,
    params: TransonicSlimParams,
    fixed_logR_son: float,
    fixed_lambda0: float,
    *,
    max_nfev: int,
):
    lower, upper = _profile_unknown_bounds(params)
    clipped = np.clip(np.asarray(x0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    kwargs = (
        {"jac": lambda trial: _profile_jacobian_from_unknowns(trial, params, fixed_logR_son, fixed_lambda0)}
        if USE_BLOCK_JACOBIAN
        else {"jac_sparsity": _profile_jac_sparsity_pattern(params)}
    )
    return least_squares(
        lambda trial: _profile_residual_from_unknowns(trial, params, fixed_logR_son, fixed_lambda0),
        clipped,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=max_nfev,
        **kwargs,
    )


def dominant_block(row: dict[str, object]) -> str:
    blocks = {
        "profile_solve": abs(float(row["solve_profile_max"])),
        "profile_diff": abs(float(row["differential_profile_max"])),
        "interval_R": abs(float(row["interval_R_max"])),
        "interval_E": abs(float(row["interval_E_max"])),
        "outer_omega": abs(float(row["outer_omega"])),
        "outer_energy": abs(float(row["outer_energy"])),
        "D": abs(float(row["D"])),
        "C1": abs(float(row["C1"])),
        "C2": abs(float(row["C2"])),
        "K": abs(float(row["K"])),
    }
    return max(blocks, key=blocks.get)


def row_for_solution(
    *,
    label: str,
    solve_params: TransonicSlimParams,
    audit_params: TransonicSlimParams,
    x: np.ndarray,
    result,
    fixed_logR_son: float,
    fixed_lambda0: float,
    stress_factor: float,
) -> dict[str, object]:
    z = _pack_profile_unknowns_to_state(x, audit_params, fixed_logR_son, fixed_lambda0)
    solve_residual = _profile_residual_from_unknowns(x, solve_params, fixed_logR_son, fixed_lambda0)
    diff_residual = _profile_residual_from_unknowns(x, audit_params, fixed_logR_son, fixed_lambda0)
    audit = residual_audit_from_state_vector(z, audit_params)
    profile = transonic_profile_from_state_vector(z, audit_params)
    thin = thin_metrics(profile, audit_params)
    diag = pressure_support_diagnostics(z, audit_params)
    mask = np.ones(len(profile.R), dtype=bool)
    if len(mask) > 8:
        mask[:2] = False
        mask[-2:] = False
    row = {
        "label": label,
        "ratio": float(audit_params.mdot_edd_ratio),
        "stress_factor": float(stress_factor),
        "R_out_rg": float(audit_params.R_out_rg),
        "N": int(audit_params.n_nodes),
        "solve_form": solve_params.interval_residual_form,
        "solve_weighting": solve_params.integrated_residual_weighting,
        "solve_profile_max": float(np.max(np.abs(solve_residual))),
        "differential_profile_max": float(np.max(np.abs(diff_residual))),
        "full_overdetermined_max": float(np.max(np.abs(collocation_residual(z, audit_params)))),
        "interval_R_max": float(audit.interval_radial_max),
        "interval_E_max": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "D": float(audit.sonic_D),
        "C1": float(audit.sonic_C1),
        "C2": float(audit.sonic_C2),
        "K": float(audit.sonic_K),
        "sonic_smin": float(audit.sonic_smin_over_smax),
        "Rson_rg": float(profile.sonic_radius / audit_params.r_g),
        "lambda0_over_lK_isco": float(audit.lambda0_over_lK_isco),
        "max_abs_omega": float(thin["max_abs_omega_frac"]),
        "max_abs_qadv_qvisc": float(thin["max_abs_qadv_qvisc"]),
        "max_abs_qbalance": float(thin["max_abs_qbalance_thin"]),
        "max_HR": float(thin["max_HR"]),
        "median_pw_nusigma_error": float(thin["median_pw_nusigma_error"]),
        "pressure_frac": float(np.max(np.abs(diag["pressure_frac"][mask]))),
        "inertia_frac": float(np.max(np.abs(diag["inertia_frac"][mask]))),
        "optimizer_success": bool(result.success),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "message": str(result.message),
    }
    row["dominant"] = dominant_block(row)
    return row


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Fixed-Eigenvalue Profile Solve",
        "",
        "Generated by `scripts/run_standard_slim_fixed_eigen_profile.py`.",
        "",
        "This Stage B benchmark holds `Rson=5.9 rg` and `lambda0=lK_ISCO/(r_g c)`, solves only the nodal profile, and audits the full transonic residual afterward.",
        "",
        f"Default diagnostic settings: solve form `{SOLVE_FORM}`, weighting `{SOLVE_WEIGHTING}`, integrated max_nfev `{MAX_NFEV_INTEGRATED}`, differential polish `{'on' if RUN_DIFFERENTIAL_POLISH else 'off'}`, block Jacobian `{'on' if USE_BLOCK_JACOBIAN else 'off'}`. Override with `IMBH_STANDARD_SLIM_FIXED_EIGEN_*` environment variables for production runs.",
        "",
        "| label | stress | R_out/rg | N | solve form | weighting | solve max | diff profile | full overdet | dominant | int R | int E | outer omega | outer E | D | C1 | C2 | K | Rson/rg | lambda/lK | Omega err | Qadv/Qvisc | qbalance | H/R | PW med err | p support | inertia | nfev | success | optimality | message |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {label} | {stress_factor} | {R_out_rg} | {N} | {solve_form} | {solve_weighting} | "
            "{solve_profile_max} | {differential_profile_max} | {full_overdetermined_max} | {dominant} | "
            "{interval_R_max} | {interval_E_max} | {outer_omega} | {outer_energy} | {D} | {C1} | {C2} | {K} | "
            "{Rson_rg} | {lambda0_over_lK_isco} | {max_abs_omega} | {max_abs_qadv_qvisc} | {max_abs_qbalance} | "
            "{max_HR} | {median_pw_nusigma_error} | {pressure_frac} | {inertia_frac} | {nfev} | {optimizer_success} | "
            "{optimality} | {message} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps([{key: json_safe(value) for key, value in row.items()} for row in rows], indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    primary = [
        row
        for row in rows
        if row["label"] == "differential_polish" and abs(float(row["stress_factor"]) - PRIMARY_STRESS_FACTOR) < 1.0e-12
    ]
    if not primary:
        primary = [row for row in rows if row["label"] == "integrated_solve" and abs(float(row["stress_factor"]) - PRIMARY_STRESS_FACTOR) < 1.0e-12]
    if not primary:
        return
    primary = sorted(primary, key=lambda row: float(row["R_out_rg"]))
    width, height = 900, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    panels = [
        (70, 70, 410, 440, "log10 diff profile residual", "differential_profile_max"),
        (500, 70, 840, 440, "log10 full overdetermined residual", "full_overdetermined_max"),
    ]
    x_values = np.asarray([float(row["R_out_rg"]) for row in primary], dtype=float)
    x_plot = np.log10(x_values)
    x_min, x_max = float(np.min(x_plot)), float(np.max(x_plot))
    if x_max <= x_min:
        x_max = x_min + 1.0
    for x0, y0, x1, y1, title, key in panels:
        values = np.asarray([abs(float(row[key])) for row in primary], dtype=float)
        y_plot = np.log10(np.maximum(values, 1.0e-16))
        y_min, y_max = float(np.min(y_plot)), float(np.max(y_plot))
        if y_max <= y_min:
            y_min -= 1.0
            y_max += 1.0
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad
        draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
        draw.text((x0, y0 - 25), title, fill=(20, 20, 20), font=font)
        points = []
        for xx, yy, label in zip(x_plot, y_plot, x_values):
            px = x0 + int((xx - x_min) / (x_max - x_min) * (x1 - x0))
            py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
            points.append((px, py))
            draw.text((px - 15, y1 + 8), f"{label:g}", fill=(50, 50, 50), font=font)
        if len(points) >= 2:
            draw.line(points, fill=(31, 119, 180), width=3)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=(31, 119, 180))
        draw.text((x0 + 5, y0 + 5), f"{y_max:.2e}", fill=(80, 80, 80), font=font)
        draw.text((x0 + 5, y1 - 18), f"{y_min:.2e}", fill=(80, 80, 80), font=font)
    draw.text((70, 25), "Fixed-eigen profile benchmark at Mdot/Edd=1e-3", fill=(20, 20, 20), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows: list[dict[str, object]] = []
    for stress_factor in STRESS_FACTORS:
        for R_out_rg in R_OUT_CASES:
            solve_params = make_params(fiducial, mdot_edd, R_out_rg, stress_factor, form=SOLVE_FORM, weighting=SOLVE_WEIGHTING)
            audit_params = replace(solve_params, interval_residual_form="differential", integrated_residual_weighting="none")
            lambda0 = float(audit_params.potential.l_k(audit_params.potential.r_isco) / (audit_params.r_g * C))
            fixed_logR_son = float(np.log(R_SON_RG * audit_params.r_g))
            print(
                f"fixed-eigen ratio={MDOT_RATIO:g} stress={stress_factor:g} R_out={R_out_rg:g} "
                f"N={audit_params.n_nodes} solve={SOLVE_FORM}/{SOLVE_WEIGHTING}",
                flush=True,
            )
            z_seed, _arrays = analytic_seed_state(audit_params, fixed_logR_son, lambda0)
            x0 = profile_unknowns_from_state(z_seed, audit_params)
            integrated = solve_profile_stage(
                x0,
                solve_params,
                fixed_logR_son,
                lambda0,
                max_nfev=MAX_NFEV_INTEGRATED,
            )
            rows.append(
                row_for_solution(
                    label="integrated_solve",
                    solve_params=solve_params,
                    audit_params=audit_params,
                    x=integrated.x,
                    result=integrated,
                    fixed_logR_son=fixed_logR_son,
                    fixed_lambda0=lambda0,
                    stress_factor=stress_factor,
                )
            )
            print(
                f"  integrated solve={rows[-1]['solve_profile_max']:.3e} diff={rows[-1]['differential_profile_max']:.3e} "
                f"full={rows[-1]['full_overdetermined_max']:.3e} dom={rows[-1]['dominant']}",
                flush=True,
            )
            if RUN_DIFFERENTIAL_POLISH:
                differential = solve_profile_stage(
                    integrated.x,
                    audit_params,
                    fixed_logR_son,
                    lambda0,
                    max_nfev=MAX_NFEV_DIFFERENTIAL,
                )
                rows.append(
                    row_for_solution(
                        label="differential_polish",
                        solve_params=audit_params,
                        audit_params=audit_params,
                        x=differential.x,
                        result=differential,
                        fixed_logR_son=fixed_logR_son,
                        fixed_lambda0=lambda0,
                        stress_factor=stress_factor,
                    )
                )
                print(
                    f"  differential diff={rows[-1]['differential_profile_max']:.3e} "
                    f"full={rows[-1]['full_overdetermined_max']:.3e} dom={rows[-1]['dominant']}",
                    flush=True,
                )
            write_table(rows)
            write_figure(rows)
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
