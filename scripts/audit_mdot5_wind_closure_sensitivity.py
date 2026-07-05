"""Frozen-profile closure sensitivity for the Mdot=5 energy-wind branch."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_standard_slim_stream_mass_annulus_scan as scan  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_state,
    stream_heating_rate,
    unpack_state,
    wind_energy_loss_rate,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient  # noqa: E402
from imri_qpe.layer3_minidisk_1d.winds import energy_limited_wind_derivatives, q_edd_vertical  # noqa: E402
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402


TARGET_PROFILES: tuple[tuple[str, float, str], ...] = (
    (
        "target_0p2_from_eps098",
        0.2,
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_098_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac098_mass_0p8_wind_0_heat_0_ewind_0p98_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "target_0p5_from_eps0995",
        0.5,
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_099_10_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac099_10_mass_0p8_wind_0_heat_0_ewind_0p995_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "target_0p8_from_eta635",
        0.8,
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_635_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta635_mass_0p8_wind_0_heat_0_ewind_0p998253253_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
)

CHI_VALUES = (0.995, 0.99, 0.985)
WIDTH_VALUES = (0.001, 0.0025, 0.005, 0.01)

JSON_OUTPUT = ROOT / "outputs/tables/m5_energy_wind_closure_sensitivity_frozen.json"
MD_OUTPUT = ROOT / "outputs/tables/m5_energy_wind_closure_sensitivity_frozen.md"


def _format(value: Any) -> str:
    if isinstance(value, str):
        return value
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


def _wind_integrals(z: np.ndarray, params) -> dict[str, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    qwind: list[float] = []
    qvisc: list[float] = []
    qrad: list[float] = []
    qadv: list[float] = []
    dqa: list[float] = []
    dqe: list[float] = []
    activation: list[float] = []
    qedd_values: list[float] = []
    radii: list[float] = []
    for idx in range(len(logR) - 1):
        dx = float(logR[idx + 1] - logR[idx])
        xm = float(0.5 * (logR[idx] + logR[idx + 1]))
        ym = np.array([0.5 * (logu[idx] + logu[idx + 1]), 0.5 * (logT[idx] + logT[idx + 1])], dtype=float)
        gm = np.array([(logu[idx + 1] - logu[idx]) / dx, (logT[idx + 1] - logT[idx]) / dx], dtype=float)
        qv, qr, qa, _qe = _heating_terms_from_gradient(xm, ym, gm, lambda0, params)
        qs = stream_heating_rate(xm, params)
        state = algebraic_state(xm, float(ym[0]), float(ym[1]), lambda0, params)
        qw = wind_energy_loss_rate(state, qv, qs, qa, params)
        qavail = float(qv + qs - qa)
        qedd = float(q_edd_vertical(state.Omega_K, state.H, kappa=params.kappa))
        width_fraction = float(params.wind_activation_width_fraction)
        da, de = energy_limited_wind_derivatives(
            qavail,
            qedd,
            float(params.wind_energy_limited_epsilon),
            chi_edd=float(params.wind_eddington_chi),
            activation_width=width_fraction * qedd,
            activation_width_dQedd=width_fraction,
        )
        qwind.append(float(qw))
        qvisc.append(float(qv))
        qrad.append(float(qr))
        qadv.append(float(qa))
        dqa.append(float(da))
        dqe.append(float(de))
        activation.append(float(qavail - float(params.wind_eddington_chi) * qedd))
        qedd_values.append(float(qedd))
        radii.append(float(np.exp(xm)))
    R = np.asarray(radii, dtype=float)
    weights = 2.0 * np.pi * R**2
    logR_mid = np.log(R)
    qw = np.asarray(qwind, dtype=float)
    qv = np.asarray(qvisc, dtype=float)
    qr = np.asarray(qrad, dtype=float)
    qa = np.asarray(qadv, dtype=float)
    activation_arr = np.asarray(activation, dtype=float)
    qedd_arr = np.asarray(qedd_values, dtype=float)
    int_wind = float(np.trapezoid(weights * qw, logR_mid))
    int_visc = float(np.trapezoid(weights * np.abs(qv), logR_mid) + 1.0e-300)
    int_rad = float(np.trapezoid(weights * np.abs(qr), logR_mid) + 1.0e-300)
    int_adv = float(np.trapezoid(weights * np.abs(qa), logR_mid) + 1.0e-300)
    active = qw > 0.0
    transition = np.abs(activation_arr) <= np.maximum(0.01 * np.maximum(qedd_arr, 1.0e-300), 1.0e-300)
    peak_idx = int(np.argmax(qw)) if qw.size else 0
    return {
        "integrated_Qwind_Qvisc": float(int_wind / int_visc),
        "integrated_Qwind_Qrad": float(int_wind / int_rad),
        "integrated_Qwind_Qadv_abs": float(int_wind / int_adv),
        "active_interval_fraction": float(np.count_nonzero(active) / max(qw.size, 1)),
        "transition_interval_fraction": float(np.count_nonzero(transition) / max(qw.size, 1)),
        "peak_Qwind_R_rg": float(R[peak_idx] / params.r_g) if qw.size and qw[peak_idx] > 0.0 else math.nan,
        "max_dQwind_dQavail": float(np.max(dqa)) if dqa else math.nan,
        "min_dQwind_dQedd": float(np.min(dqe)) if dqe else math.nan,
    }


def audit_profile(label: str, target_qwind_qvisc: float, rel_path: str, fiducial: FiducialParams, mdot_edd: float) -> list[dict[str, Any]]:
    z, base_params = scan.load_anchor(ROOT / rel_path, fiducial, mdot_edd)
    base_diag = scan.wind_energy_diagnostic(z, base_params)
    rows: list[dict[str, Any]] = []
    for chi in CHI_VALUES:
        for width in WIDTH_VALUES:
            unit_params = replace(
                base_params,
                wind_energy_limited_epsilon=1.0,
                wind_eddington_chi=float(chi),
                wind_activation_width_fraction=float(width),
            )
            unit_diag = _wind_integrals(z, unit_params)
            unit_ratio = float(unit_diag["integrated_Qwind_Qvisc"])
            epsilon_required = float(target_qwind_qvisc / unit_ratio) if unit_ratio > 0.0 else math.inf
            achievable = bool(np.isfinite(epsilon_required) and 0.0 <= epsilon_required <= 1.0)
            trial_epsilon = min(max(epsilon_required, 0.0), 1.0) if np.isfinite(epsilon_required) else 1.0
            trial_params = replace(
                base_params,
                wind_energy_limited_epsilon=float(trial_epsilon),
                wind_eddington_chi=float(chi),
                wind_activation_width_fraction=float(width),
            )
            trial_diag = _wind_integrals(z, trial_params)
            rows.append(
                {
                    "profile_label": label,
                    "target_Qwind_Qvisc": float(target_qwind_qvisc),
                    "profile_checkpoint": rel_path,
                    "base_epsilon": float(base_params.wind_energy_limited_epsilon),
                    "base_Qwind_Qvisc": float(base_diag["integrated_Qwind_Qvisc"]),
                    "chi_edd": float(chi),
                    "width_fraction": float(width),
                    "unit_epsilon_Qwind_Qvisc": float(unit_ratio),
                    "epsilon_required_frozen": float(epsilon_required),
                    "target_achievable_frozen": achievable,
                    **trial_diag,
                }
            )
    return rows


def write_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "profile_label",
        "target_Qwind_Qvisc",
        "chi_edd",
        "width_fraction",
        "epsilon_required_frozen",
        "target_achievable_frozen",
        "integrated_Qwind_Qvisc",
        "active_interval_fraction",
        "transition_interval_fraction",
        "peak_Qwind_R_rg",
        "max_dQwind_dQavail",
        "min_dQwind_dQedd",
    ]
    lines = [
        "# Mdot=5 Energy-Wind Closure Sensitivity, Frozen Profiles",
        "",
        "Generated by `scripts/audit_mdot5_wind_closure_sensitivity.py`.",
        "",
        "This is a frozen-profile audit, not a repolished solution family.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |")
    MD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows: list[dict[str, Any]] = []
    for label, target, rel_path in TARGET_PROFILES:
        rows.extend(audit_profile(label, target, rel_path, fiducial, mdot_edd))
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(scan.json_safe(rows), indent=2, sort_keys=True) + "\n")
    write_markdown(rows)
    print(f"wrote {scan.relative_root_path(JSON_OUTPUT)}")
    print(f"wrote {scan.relative_root_path(MD_OUTPUT)}")


if __name__ == "__main__":
    main()
