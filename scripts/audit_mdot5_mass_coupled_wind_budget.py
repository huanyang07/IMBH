"""Implied mass and angular-momentum budget for energy-only wind anchors."""

from __future__ import annotations

import json
import math
import sys
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
    wind_energy_per_mass,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient  # noqa: E402
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402


ANCHORS: tuple[tuple[str, str], ...] = (
    (
        "ewind_0p98",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_098_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac098_mass_0p8_wind_0_heat_0_ewind_0p98_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "ewind_0p997",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_0997_0999_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac0997_0999_mass_0p8_wind_0_heat_0_ewind_0p997_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "eta_6p20",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_590_620_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta590_620_mass_0p8_wind_0_heat_0_ewind_0p997970569_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "eta_6p35",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_635_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta635_mass_0p8_wind_0_heat_0_ewind_0p998253253_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
)

JSON_OUTPUT = ROOT / "outputs/tables/m5_energy_wind_implied_mass_coupled_budget.json"
MD_OUTPUT = ROOT / "outputs/tables/m5_energy_wind_implied_mass_coupled_budget.md"


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


def implied_budget(label: str, rel_path: str, fiducial: FiducialParams, mdot_edd: float) -> dict[str, Any]:
    z, params = scan.load_anchor(ROOT / rel_path, fiducial, mdot_edd)
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    R_values: list[float] = []
    qwind_values: list[float] = []
    mdot_wind_prime_values: list[float] = []
    jwind_prime_values: list[float] = []
    l_values: list[float] = []
    for idx in range(len(logR) - 1):
        dx = float(logR[idx + 1] - logR[idx])
        xm = float(0.5 * (logR[idx] + logR[idx + 1]))
        ym = np.array([0.5 * (logu[idx] + logu[idx + 1]), 0.5 * (logT[idx] + logT[idx + 1])], dtype=float)
        gm = np.array([(logu[idx + 1] - logu[idx]) / dx, (logT[idx + 1] - logT[idx]) / dx], dtype=float)
        qv, _qr, qa, _qe = _heating_terms_from_gradient(xm, ym, gm, lambda0, params)
        qs = stream_heating_rate(xm, params)
        state = algebraic_state(xm, float(ym[0]), float(ym[1]), lambda0, params)
        qw = wind_energy_loss_rate(state, qv, qs, qa, params)
        R = float(state.R)
        E_w = float(wind_energy_per_mass(params.M2_g, R))
        mdot_prime = float(2.0 * np.pi * R**2 * qw / max(E_w, 1.0e-300))
        R_values.append(R)
        qwind_values.append(float(qw))
        mdot_wind_prime_values.append(mdot_prime)
        l_values.append(float(state.l))
        jwind_prime_values.append(mdot_prime * float(state.l))

    R_arr = np.asarray(R_values, dtype=float)
    logR_mid = np.log(R_arr)
    mdot_prime_arr = np.asarray(mdot_wind_prime_values, dtype=float)
    jwind_prime_arr = np.asarray(jwind_prime_values, dtype=float)
    qwind_arr = np.asarray(qwind_values, dtype=float)
    l_arr = np.asarray(l_values, dtype=float)
    mdot_wind = float(np.trapezoid(mdot_prime_arr, logR_mid))
    jwind = float(np.trapezoid(jwind_prime_arr, logR_mid))
    stream_info = scan.stream_diagnostic(z, params)
    wind_info = scan.wind_energy_diagnostic(z, params)
    adv_info = scan.advection_diagnostic(z, params)
    peak_idx = int(np.argmax(qwind_arr)) if qwind_arr.size else 0
    l_ref = float(np.nanmedian(l_arr)) if l_arr.size else math.nan
    return {
        "label": label,
        "checkpoint": rel_path,
        "epsilon_w": float(params.wind_energy_limited_epsilon),
        "Qwind_Qvisc": float(wind_info["integrated_Qwind_Qvisc"]),
        "f_adv_global": float(adv_info["f_adv_global"]),
        "Lrad_LEdd": float(adv_info["Lrad_LEdd"]),
        "Mdot_outer_current_over_inner": float(stream_info["Mdot_outer_over_inner"]),
        "implied_Mwind_over_inner_etaesc1": float(mdot_wind / params.Mdot_g_s),
        "required_Mdot_outer_over_inner_zeta1": float(stream_info["Mdot_outer_over_inner"] + mdot_wind / params.Mdot_g_s),
        "required_Mdot_outer_over_inner_zeta025": float(stream_info["Mdot_outer_over_inner"] + 0.25 * mdot_wind / params.Mdot_g_s),
        "required_Mdot_outer_over_inner_zeta05": float(stream_info["Mdot_outer_over_inner"] + 0.5 * mdot_wind / params.Mdot_g_s),
        "required_Mdot_outer_over_inner_zeta075": float(stream_info["Mdot_outer_over_inner"] + 0.75 * mdot_wind / params.Mdot_g_s),
        "implied_Jwind_over_Mdot_lmedian": float(jwind / (params.Mdot_g_s * l_ref)) if l_ref > 0.0 else math.nan,
        "peak_Qwind_R_rg": float(R_arr[peak_idx] / params.r_g) if qwind_arr.size and qwind_arr[peak_idx] > 0.0 else math.nan,
        "peak_Mwind_prime_R_rg": float(R_arr[int(np.argmax(mdot_prime_arr))] / params.r_g)
        if mdot_prime_arr.size and np.max(mdot_prime_arr) > 0.0
        else math.nan,
    }


def write_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "label",
        "epsilon_w",
        "Qwind_Qvisc",
        "f_adv_global",
        "Lrad_LEdd",
        "Mdot_outer_current_over_inner",
        "implied_Mwind_over_inner_etaesc1",
        "required_Mdot_outer_over_inner_zeta025",
        "required_Mdot_outer_over_inner_zeta05",
        "required_Mdot_outer_over_inner_zeta075",
        "required_Mdot_outer_over_inner_zeta1",
        "implied_Jwind_over_Mdot_lmedian",
        "peak_Qwind_R_rg",
        "peak_Mwind_prime_R_rg",
    ]
    lines = [
        "# Mdot=5 Energy-Wind Implied Mass-Coupled Budget",
        "",
        "Generated by `scripts/audit_mdot5_mass_coupled_wind_budget.py`.",
        "",
        "This is a post-processing diagnostic of the energy-only solutions.",
        "It assumes `E_w = GM/(2R)` and `l_w = l` for the displayed implied mass and angular-momentum loss.",
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
    rows = [implied_budget(label, rel_path, fiducial, mdot_edd) for label, rel_path in ANCHORS]
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(scan.json_safe(rows), indent=2, sort_keys=True) + "\n")
    write_markdown(rows)
    print(f"wrote {scan.relative_root_path(JSON_OUTPUT)}")
    print(f"wrote {scan.relative_root_path(MD_OUTPUT)}")


if __name__ == "__main__":
    main()
