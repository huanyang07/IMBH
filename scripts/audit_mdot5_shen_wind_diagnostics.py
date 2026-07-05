"""Consolidated Mdot=5 wind budget and Shen-style slope diagnostics."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_standard_slim_stream_mass_annulus_scan as scan  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_state,
    collocation_residual,
    residual_partition_audit_from_state_vector,
    stream_heating_rate,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    transonic_profile_from_state_vector,
    unpack_state,
    wind_energy_loss_rate,
    wind_energy_per_mass,
    wind_mass_loss_prime_from_energy,
    wind_sink_prime,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient  # noqa: E402
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402


TARGET_S = (0.05, 0.10, 0.20, 0.30, 0.50)
JSON_OUTPUT = ROOT / "outputs/tables/m5_wind_shen_budget_diagnostics.json"
MD_OUTPUT = ROOT / "outputs/tables/m5_wind_shen_budget_diagnostics.md"
PROFILE_OUTPUT = ROOT / "outputs/tables/m5_wind_shen_slope_profiles.json"


@dataclass(frozen=True)
class StandardAnchor:
    label: str
    family: str
    rel_path: str


STANDARD_ANCHORS: tuple[StandardAnchor, ...] = (
    StandardAnchor(
        "no_wind_m5_fs080",
        "no_wind_high_mdot",
        "outputs/checkpoints/high_mdot_stream_m5_compact_N896_050_to080_no_energy_merit/"
        "m5n896fast2_mass_0p8_torque_0p005_mdot_5_N896.npz",
    ),
    StandardAnchor(
        "ewind_0p98",
        "energy_only_wind",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_098_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac098_mass_0p8_wind_0_heat_0_ewind_0p98_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    StandardAnchor(
        "ewind_0p997",
        "energy_only_wind",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_0997_0999_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac0997_0999_mass_0p8_wind_0_heat_0_ewind_0p997_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    StandardAnchor(
        "eta_6p20",
        "energy_only_wind",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_590_620_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta590_620_mass_0p8_wind_0_heat_0_ewind_0p997970569_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    StandardAnchor(
        "eta_6p35",
        "energy_only_wind",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_635_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta635_mass_0p8_wind_0_heat_0_ewind_0p998253253_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    StandardAnchor(
        "eta_6p425",
        "energy_only_wind",
        "outputs/checkpoints/m5_energy_wind_eta_adaptive_manual_6425_N896/"
        "eta_adaptive_manual_6425_mass_0p8_wind_0_heat_0_ewind_0p998379467_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    StandardAnchor(
        "powerlaw_zeta_0p03",
        "prescribed_powerlaw_bridge",
        "outputs/checkpoints/m5_energy_wind_powerlaw_mass_coupled_adaptive_0p015_to_0p03/zeta_0p03_N896.npz",
    ),
    StandardAnchor(
        "powerlaw_zeta_0p05",
        "prescribed_powerlaw_bridge",
        "outputs/checkpoints/m5_energy_wind_powerlaw_mass_coupled_adaptive_0p03_to_0p05/zeta_0p05_N896.npz",
    ),
    StandardAnchor(
        "powerlaw_zeta_0p10",
        "prescribed_powerlaw_bridge",
        "outputs/checkpoints/m5_energy_wind_powerlaw_mass_coupled_adaptive_0p05_to_0p10/zeta_0p1_N896.npz",
    ),
)

LOCAL_BVP_TABLES: tuple[tuple[str, str], ...] = (
    ("local_bvp_zeta_0p03_etaE33_N96", "outputs/tables/m5_local_mdot_bvp_zeta0p03_N96_etaE33_pilot.json"),
    ("local_bvp_zeta_0p03_etaE33_N128", "outputs/tables/m5_local_mdot_bvp_zeta0p03_N128_etaE33_pilot.json"),
)


def _format(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "nan"
    if number == 0.0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e4:
        return f"{number:.3e}"
    return f"{number:.6g}"


def _target_s_key(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _cumtrapz(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(values)
    if values.size < 2:
        return out
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(x))
    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(np.count_nonzero(mask)) == 0:
        return math.nan
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w)
    if cdf[-1] <= 0.0:
        return math.nan
    return float(np.interp(float(q) * cdf[-1], cdf, v))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(np.count_nonzero(mask)) == 0:
        return math.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def _fit_slope(logR: np.ndarray, log_mdot_ratio: np.ndarray, mask: np.ndarray) -> float:
    logR = np.asarray(logR, dtype=float)
    log_mdot_ratio = np.asarray(log_mdot_ratio, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(logR) & np.isfinite(log_mdot_ratio)
    if int(np.count_nonzero(mask)) < 4:
        return math.nan
    x = logR[mask] - float(np.mean(logR[mask]))
    y = log_mdot_ratio[mask] - float(np.mean(log_mdot_ratio[mask]))
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return math.nan
    return float(np.dot(x, y) / denom)


def _smoothness_and_spikes(values: np.ndarray, logR: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values, dtype=float)
    logR = np.asarray(logR, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values) & (values > 0.0)
    if int(np.count_nonzero(mask)) < 4:
        return math.nan, 0
    selected = np.log(values[mask])
    selected_x = logR[mask]
    slope = np.abs(np.diff(selected) / np.maximum(np.diff(selected_x), 1.0e-300))
    ratio = np.maximum(values[1:], 1.0e-300) / np.maximum(values[:-1], 1.0e-300)
    ratio_mask = mask[1:] & mask[:-1]
    spikes = int(np.count_nonzero(ratio_mask & ((ratio > 5.0) | (ratio < 0.2))))
    return float(np.nanmedian(slope)) if slope.size else math.nan, spikes


def _interval_profiles(z: np.ndarray, params) -> dict[str, np.ndarray]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    R_values: list[float] = []
    qvisc_values: list[float] = []
    qrad_values: list[float] = []
    qadv_values: list[float] = []
    qwind_values: list[float] = []
    stream_heat_values: list[float] = []
    mdot_values: list[float] = []
    source_prime_values: list[float] = []
    wind_sink_prime_values: list[float] = []
    l_values: list[float] = []
    h_over_r_values: list[float] = []

    for idx in range(len(logR) - 1):
        dx = float(logR[idx + 1] - logR[idx])
        xm = float(0.5 * (logR[idx] + logR[idx + 1]))
        ym = np.array([0.5 * (logu[idx] + logu[idx + 1]), 0.5 * (logT[idx] + logT[idx + 1])], dtype=float)
        gm = np.array([(logu[idx + 1] - logu[idx]) / dx, (logT[idx + 1] - logT[idx]) / dx], dtype=float)
        qv, qr, qa, _qe = _heating_terms_from_gradient(xm, ym, gm, lambda0, params)
        qs = stream_heating_rate(xm, params)
        state = algebraic_state(xm, float(ym[0]), float(ym[1]), lambda0, params)
        qw = wind_energy_loss_rate(state, qv, qs, qa, params)
        mdot, _dmdot = stream_mass_rate_and_derivative(xm, params)

        R_values.append(float(state.R))
        qvisc_values.append(float(qv))
        qrad_values.append(float(qr))
        qadv_values.append(float(qa))
        qwind_values.append(float(qw))
        stream_heat_values.append(float(qs))
        mdot_values.append(float(mdot))
        source_prime_values.append(float(stream_source_prime(xm, params)))
        wind_sink_prime_values.append(float(wind_sink_prime(xm, params)))
        l_values.append(float(state.l))
        h_over_r_values.append(float(state.H_over_R))

    R = np.asarray(R_values, dtype=float)
    logR_mid = np.log(R)
    source_prime = np.asarray(source_prime_values, dtype=float)
    mdot = np.asarray(mdot_values, dtype=float)
    source_cumulative = _cumtrapz(source_prime, logR_mid)
    mdot_tilde = mdot + source_cumulative
    e_bind = wind_energy_per_mass(params.M2_g, R)
    qwind = np.asarray(qwind_values, dtype=float)
    implied_wind_prime = wind_mass_loss_prime_from_energy(qwind, R, e_bind)

    return {
        "R": R,
        "logR": logR_mid,
        "qvisc": np.asarray(qvisc_values, dtype=float),
        "qrad": np.asarray(qrad_values, dtype=float),
        "qadv": np.asarray(qadv_values, dtype=float),
        "qwind": qwind,
        "qstream": np.asarray(stream_heat_values, dtype=float),
        "mdot": mdot,
        "source_prime": source_prime,
        "wind_sink_prime": np.asarray(wind_sink_prime_values, dtype=float),
        "mdot_tilde": mdot_tilde,
        "implied_wind_prime_etaesc1": np.asarray(implied_wind_prime, dtype=float),
        "e_bind": np.asarray(e_bind, dtype=float),
        "l": np.asarray(l_values, dtype=float),
        "H_over_R": np.asarray(h_over_r_values, dtype=float),
    }


def _clean_masks(profile: dict[str, np.ndarray], params) -> dict[str, np.ndarray]:
    R_rg = profile["R"] / params.r_g
    qwind = profile["qwind"]
    source_prime = np.abs(profile["source_prime"])
    current_wind = np.abs(profile["wind_sink_prime"])
    inner_limit = max(float(R_rg[0]) * 1.03, 6.0)
    boundary_ok = (R_rg > inner_limit) & (R_rg < 0.94 * float(params.R_out_rg))
    source_ok = source_prime <= max(float(np.nanmax(source_prime)) * 1.0e-4, params.Mdot_g_s * 1.0e-12)
    qwind_active = qwind > max(float(np.nanmax(qwind)) * 1.0e-8, 0.0) if qwind.size else np.zeros(0, dtype=bool)
    current_wind_active = (
        current_wind > max(float(np.nanmax(current_wind)) * 1.0e-8, 0.0) if current_wind.size else np.zeros(0, dtype=bool)
    )
    return {
        "source_free": source_ok,
        "wind_energy_clean": boundary_ok & source_ok & qwind_active,
        "current_wind_clean": boundary_ok & source_ok & current_wind_active,
        "boundary_ok": boundary_ok,
    }


def _budget_row(anchor: StandardAnchor, z: np.ndarray, params, profiles: dict[str, np.ndarray]) -> dict[str, Any]:
    stream_info = scan.stream_diagnostic(z, params)
    wind_info = scan.wind_energy_diagnostic(z, params)
    adv_info = scan.advection_diagnostic(z, params)
    audit = scan.residual_audit_from_state_vector(z, params)
    partition = residual_partition_audit_from_state_vector(z, params)
    profile = transonic_profile_from_state_vector(z, params)

    R = profiles["R"]
    logR = profiles["logR"]
    weights = 2.0 * np.pi * R**2
    qplus = profiles["qvisc"] + profiles["qstream"]
    qloss = profiles["qrad"] + profiles["qadv"] + profiles["qwind"]
    energy_signed = float(np.trapezoid(weights * (qplus - qloss), logR))
    energy_scale = float(np.trapezoid(weights * (np.abs(qplus) + np.abs(qloss)), logR) + 1.0e-300)
    current_mwind = float(np.trapezoid(profiles["wind_sink_prime"], logR))
    implied_mwind = float(np.trapezoid(profiles["implied_wind_prime_etaesc1"], logR))
    l_ref = float(np.nanmedian(profiles["l"])) if profiles["l"].size else math.nan
    current_jwind = float(np.trapezoid(profiles["wind_sink_prime"] * profiles["l"], logR))
    implied_jwind = float(np.trapezoid(profiles["implied_wind_prime_etaesc1"] * profiles["l"], logR))
    full = float(np.max(np.abs(collocation_residual(z, params))))

    return {
        "label": anchor.label,
        "family": anchor.family,
        "checkpoint": anchor.rel_path,
        "N": int(params.n_nodes),
        "final_full_current_code": full,
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "physical_E": float(partition.physical_energy_max),
        "buffer_E": float(partition.buffer_energy_max),
        "peak_physical_E_rg": float(partition.peak_physical_energy_rg),
        "peak_buffer_E_rg": float(partition.peak_buffer_energy_rg),
        "Mdot_outer_over_inner": float(stream_info["Mdot_outer_over_inner"]),
        "stream_source_integral_over_inner": float(stream_info["stream_source_integral_over_inner"]),
        "current_wind_sink_integral_over_inner": float(stream_info["wind_sink_integral_over_inner"]),
        "current_wind_sink_integral_midpoint_over_inner": float(current_mwind / params.Mdot_g_s),
        "current_mass_budget_relerr": float(stream_info["relative_mass_budget_error"]),
        "implied_etaesc1_Mwind_over_inner": float(implied_mwind / params.Mdot_g_s),
        "required_Mdot_outer_etaesc1_over_inner": float(stream_info["Mdot_outer_over_inner"] + implied_mwind / params.Mdot_g_s),
        "energy_budget_signed_over_scale": float(energy_signed / energy_scale),
        "wind_AM_mode": "diagnostic_lw_equals_l_not_equation_coupled",
        "current_Jwind_over_Mdot_lmedian": float(current_jwind / (params.Mdot_g_s * l_ref)) if l_ref > 0.0 else math.nan,
        "implied_Jwind_etaesc1_over_Mdot_lmedian": float(implied_jwind / (params.Mdot_g_s * l_ref)) if l_ref > 0.0 else math.nan,
        "epsilon_w": float(params.wind_energy_limited_epsilon),
        "wind_chi": float(params.wind_eddington_chi),
        "wind_activation_width_fraction": float(params.wind_activation_width_fraction),
        "Qwind_Qvisc": float(wind_info["integrated_Qwind_Qvisc"]),
        "peak_Qwind_R_rg": float(wind_info["peak_Qwind_R_rg"]),
        "f_adv_global": float(adv_info["f_adv_global"]),
        "f_adv_inner": float(adv_info["f_adv_inner"]),
        "f_adv_pos": float(adv_info["f_adv_pos"]),
        "Lrad_LEdd": float(adv_info["Lrad_LEdd"]),
        "max_H_R": float(np.nanmax(profile.H_over_R)),
        "Rson_rg": float(profile.R[0] / params.r_g),
        "wind_sink_shape": str(params.wind_sink_shape),
        "wind_sink_fraction": float(params.wind_sink_fraction),
        "wind_sink_powerlaw_s": float(params.wind_sink_powerlaw_s),
    }


def _shen_row(anchor: StandardAnchor, params, profiles: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    R = profiles["R"]
    R_rg = R / params.r_g
    logR = profiles["logR"]
    mdot_tilde = np.maximum(profiles["mdot_tilde"], 1.0e-300)
    current_wind_prime = profiles["wind_sink_prime"]
    implied_wind_prime = profiles["implied_wind_prime_etaesc1"]
    qwind = profiles["qwind"]
    e_bind = profiles["e_bind"]
    masks = _clean_masks(profiles, params)
    current_s_eff_tilde = current_wind_prime / mdot_tilde
    implied_s_eff_tilde = implied_wind_prime / mdot_tilde
    current_weights = np.maximum(current_wind_prime, 0.0)
    implied_weights = np.maximum(implied_wind_prime, 0.0)
    current_clean = masks["current_wind_clean"]
    implied_clean = masks["wind_energy_clean"]

    log_ratio = np.log(mdot_tilde / max(float(params.Mdot_g_s), 1.0e-300))
    s_fit_current = _fit_slope(logR, log_ratio, current_clean)
    s_fit_energy_region = _fit_slope(logR, log_ratio, implied_clean)
    current_mwind = float(np.trapezoid(current_wind_prime, logR))
    implied_mwind = float(np.trapezoid(implied_wind_prime, logR))
    inner_ref_rg = float(getattr(params, "wind_sink_powerlaw_inner_rg", R_rg[0]))
    if not np.isfinite(inner_ref_rg) or inner_ref_rg <= 0.0:
        inner_ref_rg = float(R_rg[0])
    lever = max(float(params.R_out_rg) / max(inner_ref_rg, 1.0e-300), 1.0 + 1.0e-12)
    s_equiv_current = float(np.log1p(max(current_mwind / params.Mdot_g_s, 0.0)) / np.log(lever))
    s_equiv_implied = float(np.log1p(max(implied_mwind / params.Mdot_g_s, 0.0)) / np.log(lever))

    row: dict[str, Any] = {
        "label": anchor.label,
        "family": anchor.family,
        "source_corrected_fit_mask_count_current": int(np.count_nonzero(current_clean)),
        "source_corrected_fit_mask_count_energy": int(np.count_nonzero(implied_clean)),
        "s_fit_tilde_current_mdot_profile": s_fit_current,
        "s_fit_tilde_in_energy_active_region": s_fit_energy_region,
        "s_equiv_current_mass_profile": s_equiv_current,
        "s_equiv_implied_etaesc1": s_equiv_implied,
        "s_eff_tilde_current_p50": _weighted_quantile(current_s_eff_tilde, current_weights, 0.50),
        "s_eff_tilde_current_p10": _weighted_quantile(current_s_eff_tilde, current_weights, 0.10),
        "s_eff_tilde_current_p90": _weighted_quantile(current_s_eff_tilde, current_weights, 0.90),
        "s_eff_tilde_etaesc1_p50": _weighted_quantile(implied_s_eff_tilde, implied_weights, 0.50),
        "s_eff_tilde_etaesc1_p10": _weighted_quantile(implied_s_eff_tilde, implied_weights, 0.10),
        "s_eff_tilde_etaesc1_p90": _weighted_quantile(implied_s_eff_tilde, implied_weights, 0.90),
        "s_eff_tilde_etaesc1_weighted_mean": _weighted_mean(implied_s_eff_tilde, implied_weights),
    }

    profile_out: dict[str, Any] = {
        "label": anchor.label,
        "family": anchor.family,
        "R_rg": R_rg.tolist(),
        "Mdot_over_inner": (profiles["mdot"] / params.Mdot_g_s).tolist(),
        "Mdot_tilde_over_inner": (mdot_tilde / params.Mdot_g_s).tolist(),
        "stream_prime_over_inner": (profiles["source_prime"] / params.Mdot_g_s).tolist(),
        "current_wind_prime_over_inner": (current_wind_prime / params.Mdot_g_s).tolist(),
        "implied_wind_prime_etaesc1_over_inner": (implied_wind_prime / params.Mdot_g_s).tolist(),
        "s_eff_tilde_current": current_s_eff_tilde.tolist(),
        "s_eff_tilde_etaesc1": implied_s_eff_tilde.tolist(),
        "Qwind_Qvisc_local": (qwind / np.maximum(np.abs(profiles["qvisc"]), 1.0e-300)).tolist(),
        "fit_mask_current": current_clean.tolist(),
        "fit_mask_energy": implied_clean.tolist(),
    }

    for target_s in TARGET_S:
        denom = np.maximum(target_s * mdot_tilde, 1.0e-300)
        e_req = 2.0 * np.pi * R**2 * qwind / denom
        eta_req = e_req / np.maximum(e_bind, 1.0e-300)
        v_inf_over_c = np.sqrt(np.maximum(2.0 * e_req, 0.0)) / C
        smooth, spikes = _smoothness_and_spikes(eta_req, logR, implied_clean)
        safe = _target_s_key(target_s)
        row[f"etaE_req_p50_s{safe}"] = _weighted_quantile(eta_req, implied_weights, 0.50)
        row[f"etaE_req_p10_s{safe}"] = _weighted_quantile(eta_req, implied_weights, 0.10)
        row[f"etaE_req_p90_s{safe}"] = _weighted_quantile(eta_req, implied_weights, 0.90)
        row[f"etaE_req_frac_lt1_s{safe}"] = float(np.mean(eta_req[implied_clean] < 1.0)) if np.any(implied_clean) else math.nan
        row[f"etaE_req_frac_gt50_s{safe}"] = float(np.mean(eta_req[implied_clean] > 50.0)) if np.any(implied_clean) else math.nan
        row[f"etaE_req_smoothness_s{safe}"] = smooth
        row[f"etaE_req_spike_count_s{safe}"] = spikes
        row[f"vinf_over_c_p50_s{safe}"] = _weighted_quantile(v_inf_over_c, implied_weights, 0.50)
        profile_out[f"etaE_req_s{safe}"] = eta_req.tolist()
        profile_out[f"vinf_over_c_req_s{safe}"] = v_inf_over_c.tolist()

    return row, profile_out


def _load_local_bvp_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, rel_path in LOCAL_BVP_TABLES:
        path = ROOT / rel_path
        if not path.exists():
            rows.append({"label": label, "family": "local_mdot_bvp_prototype", "checkpoint": rel_path, "missing": True})
            continue
        data = json.loads(path.read_text())
        polished = next((item for item in data if item.get("label") == "polished"), data[-1] if data else {})
        rows.append(
            {
                "label": label,
                "family": "local_mdot_bvp_prototype",
                "checkpoint": rel_path,
                "missing": False,
                "N": polished.get("N", math.nan),
                "final_full": polished.get("final_full", math.nan),
                "mass_residual_max": polished.get("mass_residual_max", math.nan),
                "interval_E": polished.get("interval_E", math.nan),
                "interval_R": polished.get("interval_R", math.nan),
                "outer_omega": polished.get("outer_omega", math.nan),
                "Mdot_outer_over_inner": polished.get("Mdot_outer_over_inner", math.nan),
                "f_adv_global": polished.get("f_adv_global", math.nan),
                "Lrad_LEdd": polished.get("Lrad_LEdd", math.nan),
                "Rson_rg": polished.get("Rson_rg", math.nan),
                "wind_energy_multiplier": polished.get("wind_energy_multiplier", math.nan),
                "success": polished.get("success", False),
                "interpretation": "prototype_only_not_strict",
            }
        )
    return rows


def _write_markdown(summary: dict[str, Any]) -> None:
    rows = summary["standard_anchor_rows"]
    shen_rows = summary["shen_slope_rows"]
    local_rows = summary["local_bvp_rows"]

    budget_cols = [
        "label",
        "family",
        "N",
        "final_full_current_code",
        "physical_E",
        "interval_E",
        "Mdot_outer_over_inner",
        "stream_source_integral_over_inner",
        "current_wind_sink_integral_over_inner",
        "implied_etaesc1_Mwind_over_inner",
        "required_Mdot_outer_etaesc1_over_inner",
        "energy_budget_signed_over_scale",
        "Qwind_Qvisc",
        "f_adv_global",
        "f_adv_inner",
        "Lrad_LEdd",
        "max_H_R",
        "Rson_rg",
    ]
    shen_cols = [
        "label",
        "family",
        "s_equiv_current_mass_profile",
        "s_fit_tilde_current_mdot_profile",
        "s_equiv_implied_etaesc1",
        "s_eff_tilde_etaesc1_p50",
        "etaE_req_p50_s0p10",
        "etaE_req_p10_s0p10",
        "etaE_req_p90_s0p10",
        "vinf_over_c_p50_s0p10",
        "etaE_req_p50_s0p30",
        "etaE_req_p10_s0p30",
        "etaE_req_p90_s0p30",
        "vinf_over_c_p50_s0p30",
        "source_corrected_fit_mask_count_current",
        "source_corrected_fit_mask_count_energy",
    ]
    local_cols = [
        "label",
        "N",
        "final_full",
        "mass_residual_max",
        "interval_E",
        "Mdot_outer_over_inner",
        "f_adv_global",
        "Lrad_LEdd",
        "Rson_rg",
        "wind_energy_multiplier",
        "success",
        "interpretation",
    ]

    lines = [
        "# Mdot=5 Wind Budget and Shen-Slope Diagnostics",
        "",
        "Generated by `scripts/audit_mdot5_shen_wind_diagnostics.py`.",
        "",
        "The power-law bridge rows are calibration diagnostics. They are not labeled as physical local wind solutions.",
        "Angular-momentum columns report an implied `l_w=l` sink only; wind AM is not yet coupled into the BVP equations.",
        "",
        "## Anchor and Budget Summary",
        "",
        "| " + " | ".join(budget_cols) + " |",
        "| " + " | ".join("---" for _ in budget_cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(col, "")) for col in budget_cols) + " |")

    lines.extend(
        [
            "",
            "## Shen-Style Source-Corrected Slopes and Launch Energy",
            "",
            "`s_fit_tilde_current_mdot_profile` is meaningful mainly for prescribed mass-profile rows.",
            "`etaE_req` uses the actual `Qwind(R)` profile and target `Mdot_tilde \\propto R^s`.",
            "",
            "| " + " | ".join(shen_cols) + " |",
            "| " + " | ".join("---" for _ in shen_cols) + " |",
        ]
    )
    for row in shen_rows:
        lines.append("| " + " | ".join(_format(row.get(col, "")) for col in shen_cols) + " |")

    lines.extend(
        [
            "",
            "## Local Mdot(R) BVP Prototype Rows",
            "",
            "| " + " | ".join(local_cols) + " |",
            "| " + " | ".join("---" for _ in local_cols) + " |",
        ]
    )
    for row in local_rows:
        lines.append("| " + " | ".join(_format(row.get(col, "")) for col in local_cols) + " |")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- summary JSON: `{scan.relative_root_path(JSON_OUTPUT)}`",
            f"- profile JSON: `{scan.relative_root_path(PROFILE_OUTPUT)}`",
        ]
    )
    MD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    budget_rows: list[dict[str, Any]] = []
    shen_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for anchor in STANDARD_ANCHORS:
        path = ROOT / anchor.rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        z, params = scan.load_anchor(path, fiducial, mdot_edd)
        profiles = _interval_profiles(z, params)
        budget_rows.append(_budget_row(anchor, z, params, profiles))
        shen_row, profile_row = _shen_row(anchor, params, profiles)
        shen_rows.append(shen_row)
        profile_rows.append(profile_row)
        print(f"audited {anchor.label}: {anchor.rel_path}", flush=True)

    summary = {
        "standard_anchor_rows": scan.json_safe(budget_rows),
        "shen_slope_rows": scan.json_safe(shen_rows),
        "local_bvp_rows": scan.json_safe(_load_local_bvp_rows()),
        "target_s_values": TARGET_S,
        "notes": {
            "powerlaw_bridge_role": "calibration/debug target, not physical local wind solution",
            "am_budget_status": "implied l_w=l sink diagnostic only; wind AM is not equation-coupled yet",
        },
    }
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(scan.json_safe(summary), indent=2, sort_keys=True) + "\n")
    PROFILE_OUTPUT.write_text(json.dumps(scan.json_safe(profile_rows), indent=2, sort_keys=True) + "\n")
    _write_markdown(summary)
    print(f"wrote {scan.relative_root_path(JSON_OUTPUT)}", flush=True)
    print(f"wrote {scan.relative_root_path(MD_OUTPUT)}", flush=True)
    print(f"wrote {scan.relative_root_path(PROFILE_OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
