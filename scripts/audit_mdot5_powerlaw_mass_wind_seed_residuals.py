"""Residual localization for power-law mass-coupled wind seeds."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_standard_slim_stream_mass_annulus_scan as scan  # noqa: E402
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402
from run_mdot5_powerlaw_mass_wind_pilot import (  # noqa: E402
    ANCHOR,
    _load_reference,
    _mass_compensated_seed,
    _outer_omega_corrected_seed,
    _parse_coupling_strengths,
    _powerlaw_s_for_fraction,
    _stress_ratio_compensated_seed,
)


N_VALUES_RAW = os.environ.get("IMBH_MDOT5_POWERLAW_WIND_AUDIT_N_VALUES", "256,896")
OUTPUT_JSON = ROOT / "outputs/tables/m5_energy_wind_powerlaw_mass_seed_residuals.json"
OUTPUT_MD = ROOT / "outputs/tables/m5_energy_wind_powerlaw_mass_seed_residuals.md"


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


def _dominant_block(audit) -> tuple[str, float]:
    blocks = {
        "interval_R": abs(float(audit.interval_radial_max)),
        "interval_E": abs(float(audit.interval_energy_max)),
        "outer_omega": abs(float(audit.outer_omega)),
        "outer_energy": abs(float(audit.outer_energy)),
        "sonic_D": abs(float(audit.sonic_D)),
        "sonic_C1": abs(float(audit.sonic_C1)),
        "sonic_C2": abs(float(audit.sonic_C2)),
        "sonic_K": abs(float(audit.sonic_K)),
    }
    key = max(blocks, key=blocks.__getitem__)
    return key, float(blocks[key])


def _anchor_at_n(anchor_z: np.ndarray, anchor_params, n_nodes: int):
    if int(anchor_params.n_nodes) == int(n_nodes):
        return np.asarray(anchor_z, dtype=float), anchor_params
    target_params = replace(anchor_params, n_nodes=int(n_nodes), custom_grid_xi=None)
    profile = scan.transonic_profile_from_state_vector(anchor_z, anchor_params)
    seed = scan.remap_profile_to_new_sonic_grid(profile, target_params, temperature_mdot_power=0.0, method="linear")
    return seed, scan.apply_outer_slopes_from_state(seed, target_params)


def _row_for_seed(
    *,
    z,
    params,
    seed_kind: str,
    n_nodes: int,
    zeta: float,
    wind_fraction: float,
    powerlaw_s: float,
) -> dict[str, Any]:
    audit = scan.residual_audit_from_state_vector(z, params)
    partition = scan.residual_partition_audit_from_state_vector(z, params)
    interval = scan.interval_peak_diagnostic(z, params)
    stream = scan.stream_diagnostic(z, params)
    wind = scan.wind_energy_diagnostic(z, params)
    adv = scan.advection_diagnostic(z, params)
    dom, dom_value = _dominant_block(audit)
    return {
        "N": int(n_nodes),
        "zeta": float(zeta),
        "seed_kind": seed_kind,
        "wind_sink_fraction": float(wind_fraction),
        "wind_sink_powerlaw_s": float(powerlaw_s),
        "full": float(scan.max_residual(z, params)),
        "dominant": dom,
        "dominant_value": float(dom_value),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "sonic_D": float(audit.sonic_D),
        "sonic_C1": float(audit.sonic_C1),
        "sonic_C2": float(audit.sonic_C2),
        "partition_physical_E": float(partition.physical_energy_max),
        "partition_buffer_E": float(partition.buffer_energy_max),
        "peak_interval_E_rg": float(interval["peak_interval_E_rg"]),
        "median_abs_interval_E": float(interval["median_abs_interval_E"]),
        "Mdot_outer_over_inner": float(stream["Mdot_outer_over_inner"]),
        "wind_sink_integral_over_inner": float(stream["wind_sink_integral_over_inner"]),
        "mass_budget_error_over_inner": float(stream["mass_budget_error_over_inner"]),
        "integrated_Qwind_Qvisc": float(wind["integrated_Qwind_Qvisc"]),
        "f_adv_global": float(adv["f_adv_global"]),
        "Lrad_LEdd": float(adv["Lrad_LEdd"]),
        "Rson_rg": float(np.exp(scan.unpack_state(z, params)[2]) / params.r_g),
    }


def main() -> None:
    reference = _load_reference()
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = scan.load_anchor(ANCHOR, fiducial, mdot_edd)
    full_mwind = float(reference["full_implied_mwind_over_inner"])
    inner_rg = max(2.05, float(reference["active_inner_rg"]))
    n_values = tuple(int(piece) for piece in N_VALUES_RAW.replace(":", ",").split(",") if piece.strip())
    rows: list[dict[str, Any]] = []
    for n_nodes in n_values:
        base_z, base_params = _anchor_at_n(anchor_z, anchor_params, n_nodes)
        for zeta in _parse_coupling_strengths():
            wind_fraction = float(zeta * full_mwind)
            powerlaw_s = _powerlaw_s_for_fraction(wind_fraction, inner_rg, base_params.R_out_rg)
            trial_params = replace(
                base_params,
                wind_sink_fraction=wind_fraction,
                wind_sink_shape="powerlaw",
                wind_sink_powerlaw_inner_rg=inner_rg,
                wind_sink_powerlaw_s=powerlaw_s,
            )
            raw_params = scan.apply_outer_slopes_from_state(base_z, trial_params)
            rows.append(
                _row_for_seed(
                    z=base_z,
                    params=raw_params,
                    seed_kind="current",
                    n_nodes=n_nodes,
                    zeta=zeta,
                    wind_fraction=wind_fraction,
                    powerlaw_s=powerlaw_s,
                )
            )
            comp_z = _mass_compensated_seed(base_z, base_params, trial_params)
            comp_params = scan.apply_outer_slopes_from_state(comp_z, trial_params)
            rows.append(
                _row_for_seed(
                    z=comp_z,
                    params=comp_params,
                    seed_kind="mass_compensated_u",
                    n_nodes=n_nodes,
                    zeta=zeta,
                    wind_fraction=wind_fraction,
                    powerlaw_s=powerlaw_s,
                )
            )
            stress_z = _stress_ratio_compensated_seed(base_z, base_params, trial_params)
            stress_params = scan.apply_outer_slopes_from_state(stress_z, trial_params)
            rows.append(
                _row_for_seed(
                    z=stress_z,
                    params=stress_params,
                    seed_kind="stress_ratio_compensated_u",
                    n_nodes=n_nodes,
                    zeta=zeta,
                    wind_fraction=wind_fraction,
                    powerlaw_s=powerlaw_s,
                )
            )
            omega_z, omega_params = _outer_omega_corrected_seed(comp_z, comp_params)
            rows.append(
                _row_for_seed(
                    z=omega_z,
                    params=omega_params,
                    seed_kind="mass_compensated_outer_omega",
                    n_nodes=n_nodes,
                    zeta=zeta,
                    wind_fraction=wind_fraction,
                    powerlaw_s=powerlaw_s,
                )
            )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(scan.json_safe(rows), indent=2, sort_keys=True) + "\n")
    columns = [
        "N",
        "zeta",
        "seed_kind",
        "wind_sink_fraction",
        "wind_sink_powerlaw_s",
        "full",
        "dominant",
        "interval_R",
        "interval_E",
        "outer_omega",
        "outer_energy",
        "partition_physical_E",
        "partition_buffer_E",
        "peak_interval_E_rg",
        "Mdot_outer_over_inner",
        "wind_sink_integral_over_inner",
        "integrated_Qwind_Qvisc",
        "f_adv_global",
        "Lrad_LEdd",
    ]
    lines = [
        "# Mdot=5 Power-Law Mass-Wind Seed Residuals",
        "",
        "Generated by `scripts/audit_mdot5_powerlaw_mass_wind_seed_residuals.py`.",
        "",
        "This is a no-Newton residual localization audit for the prescribed power-law",
        "mass-coupled wind seeds.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |")
    OUTPUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUTPUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
