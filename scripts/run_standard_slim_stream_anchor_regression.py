"""Regression audit for high-Mdot no-wind and stream-fed slim-disk anchors."""

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
    pressure_supported_omega_target,
    residual_audit_from_state_vector,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    stream_torque_specific_l_and_derivative,
    transonic_profile_from_state_vector,
    unpack_state,
    wind_sink_prime,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_luminosity, eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_ANCHOR_TABLE",
    "outputs/tables/standard_slim_stream_regression_anchors.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
INNER_RADIUS_RG = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_ANCHOR_INNER_RG", "20.0"))
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_ANCHOR_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_ANCHOR_TOL", "3e-6"))

DEFAULT_ANCHORS: tuple[tuple[str, str], ...] = (
    (
        "m5_nowind_largeR",
        "outputs/checkpoints/slim_benchmark_high_mdot_no_wind_m5_adaptive_outer_mesh_N768_spot/s32_mdot_5_N768.npz",
    ),
    (
        "m2_R300_nowind",
        "outputs/checkpoints/high_mdot_finite_Rout_nowind_bridge_m2/Rout_300_mdot_2_N640.npz",
    ),
    (
        "m2_R300_fs050_notorque",
        "outputs/checkpoints/high_mdot_stream_source_bridge_m2_narrow_adaptive_source_grid_0p3_0p5/"
        "load_mass_0p5_torque_0_mdot_2_N640.npz",
    ),
    (
        "m2_R300_fs080_torquep005",
        "outputs/checkpoints/high_mdot_stream_source_bridge_m2_residual_outer_grid_secant_0p75_0p80_fine/"
        "load_mass_0p8_torque_0p005_mdot_2_N640.npz",
    ),
    (
        "m2_R300_fs0808585_torquep005",
        "outputs/checkpoints/high_mdot_stream_source_bridge_m2_adaptive_df_strong_outer_0p808_to0p81/"
        "adaptive_mass_0p8086_torque_0p005_mdot_2_N640.npz",
    ),
)


def configured_anchors() -> tuple[tuple[str, Path], ...]:
    raw = os.environ.get("IMBH_STANDARD_SLIM_STREAM_ANCHORS", "").strip()
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


def scalar(data: np.lib.npyio.NpzFile, key: str, default: Any) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value


def finite_optional(data: np.lib.npyio.NpzFile, key: str) -> float | None:
    if key not in data:
        return None
    value = float(np.asarray(data[key]).item())
    return value if np.isfinite(value) else None


def custom_grid_from_data(data: np.lib.npyio.NpzFile) -> tuple[float, ...] | None:
    if "custom_grid_xi" not in data:
        return None
    candidate = np.asarray(data["custom_grid_xi"], dtype=float)
    n_nodes = int(scalar(data, "n_nodes", candidate.size))
    if candidate.shape != (n_nodes,):
        return None
    return tuple(float(value) for value in candidate)


def slopes_from_data(data: np.lib.npyio.NpzFile) -> tuple[float, float] | None:
    if "outer_match_log_slopes" not in data:
        return None
    candidate = np.asarray(data["outer_match_log_slopes"], dtype=float)
    if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
        return (float(candidate[0]), float(candidate[1]))
    return None


def params_from_checkpoint(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    data = np.load(path, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    ratio = float(scalar(data, "ratio", 1.0))
    n_nodes = int(scalar(data, "n_nodes", (len(z) - 2) // 2))
    params = TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=float(scalar(data, "R_out_rg", 1000.0)),
        n_nodes=n_nodes,
        grid_power=float(scalar(data, "grid_power", 1.0)),
        custom_grid_xi=custom_grid_from_data(data),
        max_nfev=int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_ANCHOR_MAX_NFEV", "3000")),
        residual_tol=1.0e-8,
        outer_closure=str(scalar(data, "outer_closure", "pressure_supported_thin_energy")),
        outer_match_log_slopes=slopes_from_data(data),
        outer_temperature_logT=finite_optional(data, "outer_temperature_logT"),
        outer_entropy_logK=finite_optional(data, "outer_entropy_logK"),
        outer_omega_log_offset=float(scalar(data, "outer_omega_log_offset", 0.0)),
        outer_robin_chi=float(scalar(data, "outer_robin_chi", 0.0)),
        outer_robin_slope_target=float(scalar(data, "outer_robin_slope_target", 0.0)),
        outer_robin_slope_scale=float(scalar(data, "outer_robin_slope_scale", 1.0)),
        stream_torque_delta_l_fraction=float(scalar(data, "stream_torque_delta_l_fraction", 0.0)),
        stream_torque_center_fraction=float(scalar(data, "stream_torque_center_fraction", 0.8)),
        stream_torque_log_width=float(scalar(data, "stream_torque_log_width", 0.08)),
        stream_source_fraction=float(scalar(data, "stream_source_fraction", 0.0)),
        stream_source_center_fraction=float(scalar(data, "stream_source_center_fraction", 0.8)),
        stream_source_log_width=float(scalar(data, "stream_source_log_width", 0.08)),
        stream_source_shape=str(scalar(data, "stream_source_shape", "tanh")),
        stream_source_shape_blend=float(scalar(data, "stream_source_shape_blend", 1.0)),
        stream_mass_fraction=float(scalar(data, "stream_mass_fraction", 0.0)),
        stream_mass_center_fraction=float(scalar(data, "stream_mass_center_fraction", 0.8)),
        stream_mass_log_width=float(scalar(data, "stream_mass_log_width", 0.08)),
        wind_sink_fraction=float(scalar(data, "wind_sink_fraction", 0.0)),
        wind_sink_center_fraction=float(scalar(data, "wind_sink_center_fraction", 0.8)),
        wind_sink_log_width=float(scalar(data, "wind_sink_log_width", 0.08)),
        stream_heating_efficiency=float(scalar(data, "stream_heating_efficiency", 0.0)),
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )
    return z, refresh_outer_slopes_from_state(z, params)


def one_sided_outer_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    dx = float(logR[-1] - logR[-2])
    return float((logu[-1] - logu[-2]) / dx), float((logT[-1] - logT[-2]) / dx)


def refresh_outer_slopes_from_state(z: np.ndarray, params: TransonicSlimParams) -> TransonicSlimParams:
    if params.outer_closure in {
        "pressure_supported_thin_energy",
        "pressure_supported_temperature",
        "pressure_supported_entropy",
    }:
        return replace(params, outer_match_log_slopes=one_sided_outer_slopes(z, params))
    return params


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


def dominant(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_omega": abs(audit.outer_omega),
        "outer_energy": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
        "K": abs(audit.sonic_K),
    }
    return max(values, key=values.get)


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


def advection_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    profile = transonic_profile_from_state_vector(z, params)
    R = np.asarray(profile.R, dtype=float)
    R_rg = R / params.r_g
    qv = np.asarray(profile.Q_visc, dtype=float)
    qr = np.asarray(profile.Q_rad, dtype=float)
    qa = np.asarray(profile.Q_adv, dtype=float)
    visc = trapz_log(np.abs(qv), R) + 1.0e-300
    rad = trapz_log(qr, R)
    adv = trapz_log(qa, R)
    adv_pos = trapz_log(np.maximum(qa, 0.0), R)
    inner = R_rg <= INNER_RADIUS_RG
    inner_visc = masked_trapz_log(np.abs(qv), R, inner)
    inner_adv = masked_trapz_log(qa, R, inner)
    inner_adv_pos = masked_trapz_log(np.maximum(qa, 0.0), R, inner)
    ledd = eddington_luminosity(params.M2_g, kappa=params.kappa)
    return {
        "f_adv_global": float(adv / visc),
        "f_adv_pos": float(adv_pos / visc),
        "f_adv_inner": float(inner_adv / (inner_visc + 1.0e-300)) if np.isfinite(inner_visc) else np.nan,
        "f_adv_inner_pos": float(inner_adv_pos / (inner_visc + 1.0e-300)) if np.isfinite(inner_visc) else np.nan,
        "Lrad_LEdd": float(rad / ledd),
    }


def stream_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    _logu, _logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    R_mass = float(params.stream_source_center_fraction * params.R_out)
    R_torque = float(params.stream_torque_center_fraction * params.R_out)
    mdot_inner, _dmdot_inner = stream_mass_rate_and_derivative(float(logR[0]), params)
    mdot_outer, dmdot_outer = stream_mass_rate_and_derivative(float(logR[-1]), params)
    mdot_center, dmdot_center = stream_mass_rate_and_derivative(float(np.log(R_mass)), params)
    source_prime = np.asarray([stream_source_prime(float(x), params) for x in logR], dtype=float)
    wind_prime = np.asarray([wind_sink_prime(float(x), params) for x in logR], dtype=float)
    budget_integral = float(np.trapezoid(wind_prime - source_prime, logR))
    budget_error = float((mdot_outer - mdot_inner) - budget_integral)
    budget_scale = max(abs(mdot_outer - mdot_inner), abs(budget_integral), abs(params.Mdot_g_s), 1.0)
    l_ref = float(params.potential.l_k(R_torque))
    stream_l_outer, _stream_l_outer_deriv = stream_torque_specific_l_and_derivative(float(logR[-1]), params)
    return {
        "source_fraction": float(params.stream_source_fraction if params.stream_source_fraction != 0.0 else params.stream_mass_fraction),
        "source_shape": str(params.stream_source_shape),
        "source_shape_blend": float(params.stream_source_shape_blend),
        "torque_fraction": float(params.stream_torque_delta_l_fraction),
        "Rinj_mass_rg": float(R_mass / params.r_g),
        "Rinj_torque_rg": float(R_torque / params.r_g),
        "Mdot_inner_over_param": float(mdot_inner / params.Mdot_g_s),
        "Mdot_outer_over_inner": float(mdot_outer / params.Mdot_g_s),
        "Mdot_center_over_inner": float(mdot_center / params.Mdot_g_s),
        "dMdot_dlnR_outer_over_inner": float(dmdot_outer / params.Mdot_g_s),
        "dMdot_dlnR_center_over_inner": float(dmdot_center / params.Mdot_g_s),
        "stream_source_integral_over_inner": float(np.trapezoid(source_prime, logR) / params.Mdot_g_s),
        "wind_sink_integral_over_inner": float(np.trapezoid(wind_prime, logR) / params.Mdot_g_s),
        "mass_budget_error_over_inner": float(budget_error / params.Mdot_g_s),
        "relative_mass_budget_error": float(abs(budget_error) / budget_scale),
        "stream_l_outer_over_lKinj": float(stream_l_outer / l_ref) if l_ref > 0.0 else np.nan,
    }


def pressure_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    slopes = params.outer_match_log_slopes
    if slopes is None:
        return {"outer_pressure_target": np.nan, "outer_pressure_residual": np.nan}
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    ln_omega = float(np.log(profile.Omega[-1] / profile.Omega_K[-1]))
    target = pressure_supported_omega_target(
        float(logR[-1]),
        np.array([logu[-1], logT[-1]], dtype=float),
        np.asarray(slopes, dtype=float),
        lambda0,
        params,
    ) + float(params.outer_omega_log_offset)
    return {"outer_pressure_target": float(target), "outer_pressure_residual": float(ln_omega - target)}


def interval_peak_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    intervals = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            for idx in range(len(logR) - 1)
        ],
        dtype=float,
    )
    R_mid = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    peak_R = int(np.argmax(np.abs(intervals[:, 0])))
    peak_E = int(np.argmax(np.abs(intervals[:, 1])))
    return {
        "peak_interval_R_rg": float(R_mid[peak_R]),
        "peak_interval_R_value": float(intervals[peak_R, 0]),
        "peak_interval_E_rg": float(R_mid[peak_E]),
        "peak_interval_E_value": float(intervals[peak_E, 1]),
        "median_abs_interval_E": float(np.median(np.abs(intervals[:, 1]))),
        "p90_abs_interval_E": float(np.quantile(np.abs(intervals[:, 1]), 0.9)),
    }


def row_for_anchor(label: str, path: Path, z: np.ndarray, params: TransonicSlimParams) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    logu, _logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    full = max_residual(z, params)
    outer_dx = float(logR[-1] - logR[-2])
    uniform_dx = float((logR[-1] - logR[0]) / max(params.n_nodes - 1, 1))
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "ratio": float(params.mdot_edd_ratio),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "grid_power": float(params.grid_power),
        "custom_grid": bool(params.custom_grid_xi is not None),
        "outer_dx": outer_dx,
        "outer_dx_ratio_to_uniform": float(outer_dx / uniform_dx) if uniform_dx > 0.0 else np.nan,
        "full": full,
        "accepted": bool(full <= ACCEPTANCE_TOL),
        "anchor_eligible": bool(full <= ANCHOR_TOL),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        **pressure_diagnostic(z, params),
        **stream_diagnostic(z, params),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0_over_lK_isco": float(audit.lambda0_over_lK_isco),
        "max_H_R": float(np.max(profile.H_over_R)),
        "integrated_adv": float(profile.integrated_advective_fraction),
        **advection_diagnostic(z, params),
        **interval_peak_diagnostic(z, params),
        "outer_H_R": float(audit.outer_H_over_R),
        "outer_Qadv_Qvisc": float(audit.outer_Qadv_over_Qvisc),
        "logu_outer": float(logu[-1]),
    }


def write_table(rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Stream Regression Anchors",
        "",
        "Generated by `scripts/run_standard_slim_stream_anchor_regression.py`.",
        "",
        f"Acceptance tolerance `{ACCEPTANCE_TOL:g}`; strict anchor tolerance `{ANCHOR_TOL:g}`.",
        "",
        "| anchor | Mdot/Edd | Rout/rg | N | source frac | source shape | source blend | torque frac | Mdot outer/inner | source integral | rel budget err | full | accepted | strict | dominant | int E | outer omega | peak E R/rg | outer dx/uniform | f_adv global | f_adv inner | Lrad/LEdd | max H/R | Rson/rg |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        formatted = {
            key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value
            for key, value in row.items()
        }
        for key in ("Mdot_outer_over_inner", "source_fraction", "torque_fraction"):
            formatted[key] = f"{float(row[key]):.6g}"
        lines.append(
            "| {label} | {ratio} | {R_out_rg} | {N} | {source_fraction} | {source_shape} | {source_shape_blend} | {torque_fraction} | "
            "{Mdot_outer_over_inner} | {stream_source_integral_over_inner} | {relative_mass_budget_error} | "
            "{full} | {accepted} | {anchor_eligible} | {dominant} | {interval_E} | {outer_omega} | "
            "{peak_interval_E_rg} | {outer_dx_ratio_to_uniform} | {f_adv_global} | {f_adv_inner} | "
            "{Lrad_LEdd} | {max_H_R} | {Rson_rg} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps(json_safe(rows), indent=2, sort_keys=True) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows: list[dict[str, Any]] = []
    for label, path in configured_anchors():
        if not path.exists():
            raise FileNotFoundError(path)
        z, params = params_from_checkpoint(path, fiducial, mdot_edd)
        row = row_for_anchor(label, path, z, params)
        rows.append(row)
        print(
            f"{label}: full={row['full']:.3e} dom={row['dominant']} "
            f"Mout/Min={row['Mdot_outer_over_inner']:.6g} accepted={row['accepted']} strict={row['anchor_eligible']}",
            flush=True,
        )
    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
