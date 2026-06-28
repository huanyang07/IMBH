"""Low-Mdot standard no-wind slim-disk benchmark.

This is the first control experiment from
``Note/CODEX_STANDARD_NO_WIND_SLIM_DISK_BENCHMARK_PLAN.md``.  It runs the
existing full transonic no-wind equations in a single-BH Paczynski-Wiita setup
and checks whether the result approaches a standard thin disk at low Mdot.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    TransonicSlimParams,
    collocation_residual,
    differential_residual_scales,
    make_log_grid,
    phase_space_null_tangent,
    remap_profile_to_new_sonic_grid,
    solve_low_mdot_transonic_homotopy,
    solve_isolated_slim_disk,
    solve_transonic_outer_branch,
    transonic_profile_from_state_vector,
)
from imri_qpe.layer3_minidisk_1d.isolated_slim_solver import IsolatedSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import pack_state
from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_THIN_TABLE",
    "outputs/tables/slim_benchmark_thin_limit.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_THIN_FIGURE",
    "outputs/figures/slim_benchmark_thin_limit_profiles.png",
)
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "standard_slim_benchmark_thin_limit"

RATIOS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_THIN_RATIOS", "1e-4,1e-3,1e-2").replace(":", ",").split(",")
    if piece.strip()
)
N_NODES = int(os.environ.get("IMBH_STANDARD_SLIM_THIN_N", "24"))
R_OUT_RG = float(os.environ.get("IMBH_STANDARD_SLIM_THIN_ROUT_RG", "1e4"))
R_OUT_LADDER = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_THIN_ROUT_LADDER", f"300,1000,{R_OUT_RG:g}").replace(":", ",").split(",")
    if piece.strip()
)
ALPHA = float(os.environ.get("IMBH_STANDARD_SLIM_THIN_ALPHA", "0.01"))
MAX_NFEV_STAGE = int(os.environ.get("IMBH_STANDARD_SLIM_THIN_MAX_NFEV_STAGE", "450"))
MAX_NFEV_FINAL = int(os.environ.get("IMBH_STANDARD_SLIM_THIN_MAX_NFEV_FINAL", "700"))
RESIDUAL_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_THIN_RESIDUAL_TOL", "1e-5"))
STRESS_FACTOR = float(os.environ.get("IMBH_STANDARD_SLIM_THIN_STRESS_FACTOR", "1.0"))
PROFILE_FIT_EXCLUDE = int(os.environ.get("IMBH_STANDARD_SLIM_THIN_PROFILE_EXCLUDE", "4"))
FOLD_DS = float(os.environ.get("IMBH_STANDARD_SLIM_THIN_FOLD_DS", "0.01"))
FOLD_S_MAX = float(os.environ.get("IMBH_STANDARD_SLIM_THIN_FOLD_S_MAX", "12"))
FOLD_MAX_STEPS = int(os.environ.get("IMBH_STANDARD_SLIM_THIN_FOLD_MAX_STEPS", "2400"))


def fmt(value: float) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.4g}"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items() if key != "z"}, sort_keys=True)


def make_params(fiducial: FiducialParams, ratio: float, mdot_edd: float, R_out_rg: float | None = None) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=float(R_OUT_RG if R_out_rg is None else R_out_rg),
        n_nodes=N_NODES,
        grid_power=1.0,
        max_nfev=MAX_NFEV_FINAL,
        residual_tol=RESIDUAL_TOL,
        outer_closure="thin_value",
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def reduced_seed(params: TransonicSlimParams) -> tuple[np.ndarray | None, dict[str, object]]:
    """Return a reduced nearly-Keplerian seed for the full transonic solve."""

    R_in = 1.08 * params.potential.r_isco
    grid = make_log_grid(R_in, params.R_out, max(params.n_nodes, 24))
    reduced_params = IsolatedSlimParams(
        M2_g=params.M2_g,
        Mdot_g_s=params.Mdot_g_s,
        R_in=R_in,
        alpha=params.alpha,
        mu_mol=params.mu_mol,
        kappa=params.kappa,
        gamma_gas=params.gamma_gas,
        l_in=float(params.potential.l_k(params.potential.r_isco)),
        T_bounds=(np.exp(params.logT_bounds[0]), np.exp(params.logT_bounds[1])),
        sigma_brackets=160,
    )
    result = solve_isolated_slim_disk(grid, reduced_params, max_iter=90, tol=2.0e-3, damping=0.55)
    meta = {
        "reduced_converged": bool(result.converged),
        "reduced_failed": bool(result.failed),
        "reduced_max_residual": float(result.max_abs_residual),
        "reduced_L1": float(result.L1_residual),
        "reduced_message": result.message,
    }
    if result.profile is None:
        return None, meta
    from imri_qpe.layer3_minidisk_1d import initial_guess_from_reduced_solver

    return initial_guess_from_reduced_solver(result.profile, params), meta


def state_from_profile(profile) -> np.ndarray:
    return pack_state(np.log(profile.u), np.log(profile.T), np.log(profile.sonic_radius), profile.lambda0)


def profile_mask(profile) -> np.ndarray:
    mask = np.ones_like(profile.R, dtype=bool)
    n = len(mask)
    exclude = min(PROFILE_FIT_EXCLUDE, max(0, n // 4 - 1))
    if exclude > 0 and 2 * exclude < n:
        mask[:exclude] = False
        mask[-exclude:] = False
    return mask


def effective_nu_sigma(profile) -> np.ndarray:
    dOmega_dlnR = np.gradient(profile.Omega_K, np.log(profile.R), edge_order=1)
    return profile.W / (-dOmega_dlnR + 1.0e-300)


def pw_thin_nu_sigma(profile, params: TransonicSlimParams) -> np.ndarray:
    potential = params.potential
    shear = -potential.dln_omega_k_dlnR(profile.R)
    l_k = potential.l_k(profile.R)
    l0 = profile.lambda0 * params.r_g * C
    numerator = params.Mdot_g_s * (l_k - l0)
    denominator = 2.0 * np.pi * profile.R**2 * shear * profile.Omega_K + 1.0e-300
    return numerator / denominator


def newtonian_thin_nu_sigma(profile, params: TransonicSlimParams) -> np.ndarray:
    l0 = profile.lambda0 * params.r_g * C
    R0 = max(l0**2 / (G * params.M2_g), 0.0)
    factor = np.maximum(1.0 - np.sqrt(R0 / profile.R), 0.0)
    return params.Mdot_g_s * factor / (3.0 * np.pi)


def luminosities(profile) -> tuple[float, float, float]:
    R = np.asarray(profile.R, dtype=float)
    weights = 2.0 * np.pi * R[:-1] * np.diff(R)
    qrad = 0.5 * (profile.Q_rad[:-1] + profile.Q_rad[1:])
    qvisc = 0.5 * (profile.Q_visc[:-1] + profile.Q_visc[1:])
    qadv = 0.5 * (profile.Q_adv[:-1] + profile.Q_adv[1:])
    return (
        float(np.sum(weights * qrad)),
        float(np.sum(weights * qvisc)),
        float(np.sum(weights * qadv)),
    )


def thin_metrics(profile, params: TransonicSlimParams) -> dict[str, float]:
    mask = profile_mask(profile)
    nu_sigma_eff = effective_nu_sigma(profile)
    nu_sigma_pw = pw_thin_nu_sigma(profile, params)
    nu_sigma_newton = newtonian_thin_nu_sigma(profile, params)
    pw_error = np.abs(nu_sigma_eff / (nu_sigma_pw + 1.0e-300) - 1.0)
    newton_error = np.full_like(nu_sigma_eff, np.nan)
    valid_newton = nu_sigma_newton > 0.0
    newton_error[valid_newton] = np.abs(nu_sigma_eff[valid_newton] / nu_sigma_newton[valid_newton] - 1.0)
    qadv_qvisc = np.divide(profile.Q_adv, profile.Q_visc, out=np.full_like(profile.Q_adv, np.nan), where=profile.Q_visc != 0.0)
    qbal = np.abs(profile.Q_visc - profile.Q_rad) / (np.abs(profile.Q_visc) + np.abs(profile.Q_rad) + 1.0e-300)
    l_rad, l_visc, l_adv = luminosities(profile)
    return {
        "max_abs_omega_frac": float(np.nanmax(np.abs(profile.Omega[mask] / profile.Omega_K[mask] - 1.0))),
        "max_abs_qadv_qvisc": float(np.nanmax(np.abs(qadv_qvisc[mask]))),
        "max_abs_qbalance_thin": float(np.nanmax(qbal[mask])),
        "max_HR": float(np.nanmax(profile.H_over_R[mask])),
        "max_pw_nusigma_error": float(np.nanmax(pw_error[mask])),
        "median_pw_nusigma_error": float(np.nanmedian(pw_error[mask])),
        "max_newton_nusigma_error": float(np.nanmax(newton_error[mask & valid_newton])),
        "median_newton_nusigma_error": float(np.nanmedian(newton_error[mask & valid_newton])),
        "min_tau": float(np.nanmin(profile.tau[mask])),
        "L_rad_over_L_edd": float(l_rad / (0.1 * C**2 * eddington_mdot(params.M2_g, kappa=params.kappa))),
        "L_visc_over_L_edd": float(l_visc / (0.1 * C**2 * eddington_mdot(params.M2_g, kappa=params.kappa))),
        "integrated_adv_over_visc": float(l_adv / (abs(l_visc) + 1.0e-300)),
    }


def quick_fold_audit(profile, params: TransonicSlimParams) -> dict[str, object]:
    """Trace the local phase-space curve and flag radial projection folds."""

    R = np.asarray(profile.R, dtype=float)
    logR = np.log(R)
    start = min(2, len(R) - 1)
    z = np.array([logR[start], np.log(profile.u[start]), np.log(profile.T[start])], dtype=float)
    lambda0 = float(profile.lambda0)
    try:
        p = phase_space_null_tangent(float(z[0]), z[1:], lambda0, params, previous=None).tangent
        if p[0] < 0.0:
            p = -p
    except Exception as exc:
        return {"fold_status": "start_failed", "fold_message": str(exc)}

    px_values = [float(p[0])]
    R_values = [float(np.exp(z[0]) / params.r_g)]
    smin_values = []
    previous = p
    px_sign = float(np.sign(p[0]))
    s_value = 0.0
    for _idx in range(FOLD_MAX_STEPS):
        if s_value >= FOLD_S_MAX:
            break
        try:
            p0 = phase_space_null_tangent(float(z[0]), z[1:], lambda0, params, previous=previous).tangent
            z_pred = z + FOLD_DS * p0
            p1 = phase_space_null_tangent(float(z_pred[0]), z_pred[1:], lambda0, params, previous=p0).tangent
            z = z + 0.5 * FOLD_DS * (p0 + p1)
            diag = phase_space_null_tangent(float(z[0]), z[1:], lambda0, params, previous=p1)
            p = diag.tangent
            previous = p
        except Exception as exc:
            return {
                "fold_status": "trace_failed",
                "fold_message": str(exc),
                "R_trace_max_rg": float(np.max(R_values)),
                "p_x_min": float(np.min(px_values)),
                "min_smin_over_smax_B": float(np.nanmin(smin_values)) if smin_values else np.nan,
            }
        s_value += FOLD_DS
        R_rg = float(np.exp(z[0]) / params.r_g)
        R_values.append(R_rg)
        px_values.append(float(p[0]))
        smin_values.append(float(diag.smin_over_smax_B))
        next_sign = float(np.sign(p[0]))
        if px_sign > 0.0 and next_sign < 0.0:
            return {
                "fold_status": "fold_found",
                "fold_message": "p_x sign change",
                "R_fold_rg": R_rg,
                "R_trace_max_rg": float(np.max(R_values)),
                "p_x_min": float(np.min(px_values)),
                "min_smin_over_smax_B": float(np.nanmin(smin_values)) if smin_values else np.nan,
            }
        if R_rg >= params.R_out_rg:
            return {
                "fold_status": "target_reached",
                "fold_message": "trace reached R_out",
                "R_fold_rg": np.nan,
                "R_trace_max_rg": R_rg,
                "p_x_min": float(np.min(px_values)),
                "min_smin_over_smax_B": float(np.nanmin(smin_values)) if smin_values else np.nan,
            }
        if next_sign != 0.0:
            px_sign = next_sign
    return {
        "fold_status": "no_fold_within_smax",
        "fold_message": "arclength limit reached",
        "R_fold_rg": np.nan,
        "R_trace_max_rg": float(np.max(R_values)),
        "p_x_min": float(np.min(px_values)),
        "min_smin_over_smax_B": float(np.nanmin(smin_values)) if smin_values else np.nan,
    }


def sanity_rows(fiducial: FiducialParams) -> list[dict[str, object]]:
    potential = PaczynskiWiitaPotential(fiducial.M2_g)
    R = np.geomspace(6.2 * potential.r_g, 1.0e3 * potential.r_g, 64)
    numeric = np.gradient(np.log(potential.omega_k(R)), np.log(R), edge_order=2)
    analytic = potential.dln_omega_k_dlnR(R)
    l_values = potential.l_k(R)
    min_idx = int(np.argmin(l_values))
    params = TransonicSlimParams(M2_g=fiducial.M2_g, Mdot_g_s=1.0e-4 * eddington_mdot(fiducial.M2_g), alpha=ALPHA, stress_factor=STRESS_FACTOR)
    state = algebraic_state(np.log(20.0 * params.r_g), np.log(1.0e5), np.log(1.0e6), float(potential.l_k(potential.r_isco) / (params.r_g * C)), params)
    radial_scale, energy_scale = differential_residual_scales(np.log(20.0 * params.r_g), np.array([np.log(1.0e5), np.log(1.0e6)]), float(potential.l_k(potential.r_isco) / (params.r_g * C)), params)
    return [
        {
            "test": "PW dlnOmegaK/dlnR",
            "value": float(np.max(np.abs(numeric[2:-2] - analytic[2:-2]))),
            "pass": bool(np.max(np.abs(numeric[2:-2] - analytic[2:-2])) < 2.0e-3),
            "note": "finite-difference check over 6.2-1e3 rg",
        },
        {
            "test": "PW lK minimum",
            "value": float(R[min_idx] / potential.r_g),
            "pass": bool(abs(float(R[min_idx] / potential.r_g) - 6.0) < 0.5),
            "note": "coarse grid minimum should land near 6 rg",
        },
        {
            "test": "vertical state positive",
            "value": float(min(state.H, state.rho, state.P, state.Pi, state.tau)),
            "pass": bool(min(state.H, state.rho, state.P, state.Pi, state.tau) > 0.0),
            "note": "sample algebraic transonic state",
        },
        {
            "test": "residual scales finite",
            "value": float(min(radial_scale, energy_scale)),
            "pass": bool(np.isfinite(radial_scale) and np.isfinite(energy_scale) and radial_scale > 0.0 and energy_scale > 0.0),
            "note": "sample radial/energy scales",
        },
    ]


def write_table(rows: list[dict[str, object]], sanity: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim-Disk Low-Mdot Thin-Limit Benchmark",
        "",
        "Generated by `scripts/run_standard_slim_benchmark_thin_limit.py`.",
        "",
        "This is a single-BH no-wind control experiment using the existing full transonic Paczynski-Wiita equations. The reduced nearly-Keplerian solver is used only as an initial seed/reference.",
        "",
        f"Config: final `R_out={R_OUT_RG:g} rg`, ladder `{','.join(f'{value:g}' for value in R_OUT_LADDER)} rg`, `N={N_NODES}`, `alpha={ALPHA:g}`, `stress_factor={STRESS_FACTOR:g}`, `Mdot_Edd=L_Edd/(0.1 c^2)`. This is a first-pass low-resolution control benchmark; accepted rows should later be repeated at higher `N`.",
        "",
        "## Sanity Tests",
        "",
        "| test | value | pass | note |",
        "|---|---:|:---:|---|",
    ]
    for row in sanity:
        lines.append(f"| {row['test']} | {fmt(float(row['value']))} | {'yes' if row['pass'] else 'no'} | {row['note']} |")
    lines.extend(
        [
            "",
            "## Thin-Limit Runs",
            "",
            "| Mdot/Edd | accepted | physical | equations | sonic | outer thin | residual | dominant | Omega err | Qadv/Qvisc | Q balance | H/R max | PW nuSigma max | PW nuSigma med | Newton nuSigma med | fold status | R trace max/rg | R fold/rg | Rson/rg | lambda0 | lambda/lK_ISCO | int adv | Lrad/Ledd | min tau | nfev | message |",
            "|---:|:---:|:---:|:---:|:---:|:---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {ratio} | {accepted} | {physical} | {equations} | {sonic} | {outer_thin} | {max_residual} | {dominant} | "
            "{omega_err} | {qadv} | {qbalance} | {max_HR} | {pw_max} | {pw_med} | {newton_med} | "
            "{fold_status} | {R_trace_max_rg} | {R_fold_rg} | {Rson_rg} | {lambda0} | {lambda_ratio} | "
            "{int_adv} | {Lrad} | {min_tau} | {nfev} | {message} |".format(
                ratio=fmt(float(row["ratio"])),
                accepted="yes" if row["accepted"] else "no",
                physical="yes" if row["physical"] else "no",
                equations="yes" if row["equations"] else "no",
                sonic="yes" if row["sonic"] else "no",
                outer_thin="yes" if row["outer_thin"] else "no",
                max_residual=fmt(float(row["max_residual"])),
                dominant=row["dominant"],
                omega_err=fmt(float(row["max_abs_omega_frac"])),
                qadv=fmt(float(row["max_abs_qadv_qvisc"])),
                qbalance=fmt(float(row["max_abs_qbalance_thin"])),
                max_HR=fmt(float(row["max_HR"])),
                pw_max=fmt(float(row["max_pw_nusigma_error"])),
                pw_med=fmt(float(row["median_pw_nusigma_error"])),
                newton_med=fmt(float(row["median_newton_nusigma_error"])),
                fold_status=row["fold_status"],
                R_trace_max_rg=fmt(float(row.get("R_trace_max_rg", np.nan))),
                R_fold_rg=fmt(float(row.get("R_fold_rg", np.nan))),
                Rson_rg=fmt(float(row["Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                lambda_ratio=fmt(float(row["lambda0_over_lK_isco"])),
                int_adv=fmt(float(row["integrated_advective_fraction"])),
                Lrad=fmt(float(row["L_rad_over_L_edd"])),
                min_tau=fmt(float(row["min_tau"])),
                nfev=int(row["nfev"]),
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(
        json.dumps(
            {
                "sanity": [{key: json_safe(value) for key, value in row.items()} for row in sanity],
                "runs": [{key: json_safe(value) for key, value in row.items() if key != "z"} for row in rows],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def draw_polyline(draw, points, color, width=2):
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def write_figure(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    profiles = [row["profile_arrays"] for row in rows if "profile_arrays" in row]
    if not profiles:
        return
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1400, 950
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    panels = [
        (90, 80, 650, 390, "Omega/OmegaK - 1", "omega"),
        (780, 80, 1340, 390, "Qadv/Qvisc", "qadv"),
        (90, 520, 650, 830, "H/R", "hr"),
        (780, 520, 1340, 830, "PW nuSigma fractional error", "nusigma"),
    ]
    colors = [(31, 119, 180), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
    x_all = np.concatenate([np.log10(np.asarray(profile["R_rg"])) for profile in profiles])
    x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))

    def y_limits(key: str) -> tuple[float, float]:
        values = []
        for profile in profiles:
            arr = np.asarray(profile[key], dtype=float)
            if key in {"qadv", "nusigma"}:
                arr = np.log10(np.maximum(np.abs(arr), 1.0e-12))
            values.append(arr[np.isfinite(arr)])
        merged = np.concatenate(values)
        if merged.size == 0:
            return 0.0, 1.0
        lo, hi = float(np.nanmin(merged)), float(np.nanmax(merged))
        if hi <= lo:
            pad = max(abs(hi), 1.0) * 0.05
            return lo - pad, hi + pad
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    limits = {key: y_limits(key) for *_rest, key in panels}

    for x0, y0, x1, y1, title, key in panels:
        draw.rectangle((x0, y0, x1, y1), outline=(70, 70, 70), width=2)
        draw.text((x0, y0 - 28), title, fill=(20, 20, 20), font=font)
        lo, hi = limits[key]
        draw.text((x0 + 5, y0 + 5), f"{hi:.2e}", fill=(80, 80, 80), font=font)
        draw.text((x0 + 5, y1 - 18), f"{lo:.2e}", fill=(80, 80, 80), font=font)
        for profile, color, row in zip(profiles, colors, rows):
            x = np.log10(np.asarray(profile["R_rg"], dtype=float))
            y = np.asarray(profile[key], dtype=float)
            if key in {"qadv", "nusigma"}:
                y = np.log10(np.maximum(np.abs(y), 1.0e-12))
            points = []
            for xx, yy in zip(x, y):
                if not np.isfinite(xx) or not np.isfinite(yy):
                    continue
                px = x0 + int((xx - x_min) / (x_max - x_min) * (x1 - x0))
                py = y1 - int((yy - lo) / (hi - lo) * (y1 - y0))
                points.append((px, py))
            draw_polyline(draw, points, color, width=2)
            if points:
                draw.text((points[-1][0] + 5, points[-1][1] - 8), fmt(float(row["ratio"])), fill=color, font=font)
        draw.text((x0, y1 + 12), "log10 R/rg", fill=(20, 20, 20), font=font)
    image.save(FIGURE_OUTPUT)


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"mdot{float(row['ratio']):.6g}".replace(".", "p").replace("-", "m")
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        row_json=np.array(row_json(row)),
    )


def row_from_solution(
    ratio: float,
    params: TransonicSlimParams,
    final,
    stages: list[dict[str, object]],
    seed_meta: dict[str, object],
) -> dict[str, object]:
    profile = final.profile
    audit = final.residual_audit
    status = final.status
    metrics = thin_metrics(profile, params)
    fold = quick_fold_audit(profile, params)
    block_values = {
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_omega": abs(audit.outer_omega),
        "outer_energy": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
        "K": abs(audit.sonic_K),
    }
    accepted = bool(
        status.equations_converged
        and status.sonic_regular
        and status.outer_thin
        and metrics["max_abs_omega_frac"] < 2.0e-2
        and metrics["max_abs_qadv_qvisc"] < 5.0e-2
        and metrics["max_HR"] < 0.05
    )
    z = state_from_profile(profile)
    residual = collocation_residual(z, params)
    row: dict[str, object] = {
        "ratio": float(ratio),
        "R_out_rg": float(params.R_out_rg),
        "accepted": accepted,
        "physical": bool(status.physically_valid),
        "equations": bool(status.equations_converged),
        "sonic": bool(status.sonic_regular),
        "outer_thin": bool(status.outer_thin),
        "max_residual": float(np.max(np.abs(residual))),
        "dominant": max(block_values, key=block_values.get),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0": float(profile.lambda0),
        "lambda0_over_lK_isco": float(audit.lambda0_over_lK_isco),
        "integrated_advective_fraction": float(profile.integrated_advective_fraction),
        "energy_L1": float(profile.energy_L1),
        "nfev": int(final.nfev),
        "message": final.message,
        "optimizer_success": bool(final.optimizer_success),
        "stages": stages,
        "seed_meta": seed_meta,
        "z": z,
    }
    row.update(metrics)
    row.update(fold)
    qadv_qvisc = np.divide(profile.Q_adv, profile.Q_visc, out=np.full_like(profile.Q_adv, np.nan), where=profile.Q_visc != 0.0)
    nu_sigma_eff = effective_nu_sigma(profile)
    nu_sigma_pw = pw_thin_nu_sigma(profile, params)
    row["profile_arrays"] = {
        "R_rg": profile.R / params.r_g,
        "omega": profile.Omega / profile.Omega_K - 1.0,
        "qadv": qadv_qvisc,
        "hr": profile.H_over_R,
        "nusigma": nu_sigma_eff / (nu_sigma_pw + 1.0e-300) - 1.0,
    }
    return row


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    sanity = sanity_rows(fiducial)
    rows: list[dict[str, object]] = []
    previous_profile = None
    for ratio in RATIOS:
        current_profile = previous_profile
        final = None
        params = None
        seed_meta: dict[str, object] = {}
        stage_rows: list[dict[str, object]] = []
        for stage_idx, R_out_rg in enumerate(R_OUT_LADDER):
            params = make_params(fiducial, ratio, mdot_edd, R_out_rg)
            initial_guess = remap_profile_to_new_sonic_grid(current_profile, params) if current_profile is not None else None
            seed_meta = {"seed": "remap" if initial_guess is not None else "reduced", "R_out_ladder": R_OUT_LADDER}
            if initial_guess is None:
                initial_guess, seed_meta = reduced_seed(params)
                seed_meta["seed"] = "reduced"
                seed_meta["R_out_ladder"] = R_OUT_LADDER
            print(f"thin benchmark ratio={ratio:g} R_out={R_out_rg:g} seed={seed_meta.get('seed', 'reduced')}", flush=True)
            if stage_idx == 0:
                result = solve_low_mdot_transonic_homotopy(
                    params,
                    initial_guess=initial_guess,
                    max_nfev_per_stage=MAX_NFEV_STAGE,
                    final_max_nfev=MAX_NFEV_FINAL,
                    sonic_weight_sequence=(1.0,),
                    outer_weight_sequence=(0.3, 1.0),
                    use_stage_block_jacobian=False,
                )
                final = result.final_result
                stage_rows.extend(
                    [
                        {
                            "R_out_rg": float(R_out_rg),
                            "name": stage.name,
                            "max_residual": float(stage.max_residual),
                            "nfev": int(stage.nfev),
                            "success": bool(stage.optimizer_success),
                            "message": stage.message,
                        }
                        for stage in result.stages
                    ]
                )
            else:
                final = solve_transonic_outer_branch(
                    replace(params, max_nfev=MAX_NFEV_FINAL),
                    initial_guess=initial_guess,
                    outer_residual_weight=1.0,
                    sonic_residual_weight=1.0,
                )
                stage_rows.append(
                    {
                        "R_out_rg": float(R_out_rg),
                        "name": "direct_outer_extension",
                        "max_residual": float(final.max_residual),
                        "nfev": int(final.nfev),
                        "success": bool(final.optimizer_success),
                        "message": final.message,
                    }
                )
            current_profile = final.profile if final is not None else current_profile
            print(
                f"  stage R_out={R_out_rg:g} max_res={final.max_residual:.3e} "
                f"sonic={final.status.sonic_regular} outer={final.status.outer_thin}",
                flush=True,
            )
        if final is None or params is None:
            raise RuntimeError("no benchmark stage completed")
        row = row_from_solution(float(ratio), params, final, stage_rows, seed_meta)
        rows.append(row)
        save_checkpoint(row)
        write_table(rows, sanity)
        write_figure(rows)
        print(
            f"ratio={ratio:g} accepted={row['accepted']} residual={row['max_residual']:.3e} "
            f"Omega_err={row['max_abs_omega_frac']:.3e} Qadv/Qvisc={row['max_abs_qadv_qvisc']:.3e} "
            f"fold={row['fold_status']}",
            flush=True,
        )
        if row["accepted"]:
            previous_profile = transonic_profile_from_state_vector(np.asarray(row["z"], dtype=float), params)
        else:
            previous_profile = final.profile if final.status.equations_converged and final.status.sonic_regular else previous_profile
    write_table(rows, sanity)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {JSON_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
