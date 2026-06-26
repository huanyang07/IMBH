"""Audit whether the fixed-Mdot residual floor is an outer thermal-match issue."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _outer_thin_boundary_residual,
    computational_grid,
    matched_outer_state,
    pack_state,
    pressure_supported_omega_target,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    square_collocation_residual,
    square_jac_sparsity_pattern,
    state_bounds,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    differential_residual,
    differential_residual_scales,
    local_gradient,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_n64_arclength_adaptive" / "030_arc_x1_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_outer_thermal_match_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_outer_thermal_match_audit"

N_NODES = 64
SOURCE_R_OUT_RG = 3000.0
RESIDUAL_TOL = 1.0e-6
MAX_NFEV = 50
PIVOTS = ("C1", "C2")
CORE_R_OUT_RG = 3000.0
R_OUT_PROBE_RG = (1000.0, 10000.0)


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def load_source() -> tuple[np.ndarray, float]:
    with np.load(SOURCE_CHECKPOINT, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), float(data["ratio"])


def params_for(fiducial: FiducialParams, mdot_edd: float, ratio: float, R_out_rg: float) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=N_NODES,
        R_out_rg=R_out_rg,
        residual_tol=RESIDUAL_TOL,
        max_nfev=MAX_NFEV,
    )


def remap_state_with_outer_extrapolation(z: np.ndarray, old_params: TransonicSlimParams, new_params: TransonicSlimParams) -> np.ndarray:
    logu_old, logT_old, logR_son, lambda0, logR_old = unpack_state(z, old_params)
    logR_new = computational_grid(new_params, logR_son)
    g_u, g_T = polyfit_outer_log_slopes(z, old_params)
    logu_new = np.interp(logR_new, logR_old, logu_old)
    logT_new = np.interp(logR_new, logR_old, logT_old)
    high = logR_new > logR_old[-1]
    if np.any(high):
        logu_new[high] = logu_old[-1] + g_u * (logR_new[high] - logR_old[-1])
        logT_new[high] = logT_old[-1] + g_T * (logR_new[high] - logR_old[-1])
    return pack_state(logu_new, logT_new, logR_son, lambda0)


def one_sided_outer_log_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    dx = logR[-1] - logR[-2]
    return float((logu[-1] - logu[-2]) / dx), float((logT[-1] - logT[-2]) / dx)


def polyfit_outer_log_slopes(z: np.ndarray, params: TransonicSlimParams, n_fit: int = 8) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    count = min(int(n_fit), len(logR))
    x = logR[-count:] - logR[-1]
    gu = float(np.polyder(np.poly1d(np.polyfit(x, logu[-count:], min(2, count - 1))))(0.0))
    gT = float(np.polyder(np.poly1d(np.polyfit(x, logT[-count:], min(2, count - 1))))(0.0))
    return gu, gT


def local_ode_outer_log_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    gradient = local_gradient(logR[-1], np.array([logu[-1], logT[-1]], dtype=float), lambda0, params)
    return float(gradient[0]), float(gradient[1])


def slope_pair(z: np.ndarray, params: TransonicSlimParams, source: str) -> tuple[float, float]:
    if source == "one_sided":
        return one_sided_outer_log_slopes(z, params)
    if source == "polyfit":
        return polyfit_outer_log_slopes(z, params)
    if source == "local_ode":
        return local_ode_outer_log_slopes(z, params)
    raise ValueError(f"unknown slope source {source!r}")


def pressure_thin_residual(logR: float, y: np.ndarray, lambda0: float, params: TransonicSlimParams, slopes) -> np.ndarray:
    thin = _outer_thin_boundary_residual(logR, y, lambda0, params)
    target = pressure_supported_omega_target(logR, y, slopes, lambda0, params)
    return np.asarray([thin[0] - target, thin[1]], dtype=float)


def scaled_full_local_residual(logR: float, y: np.ndarray, slopes, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    raw = differential_residual(logR, y, np.asarray(slopes, dtype=float), lambda0, params)
    scales = np.asarray(differential_residual_scales(logR, y, lambda0, params), dtype=float)
    return raw / scales


def matched_pressure_thin_state(
    logR: float,
    lambda0: float,
    params: TransonicSlimParams,
    slopes,
    *,
    initial_y: np.ndarray,
) -> np.ndarray:
    lower = np.array([params.logu_bounds[0], params.logT_bounds[0]], dtype=float)
    upper = np.array([params.logu_bounds[1], params.logT_bounds[1]], dtype=float)
    y0 = np.clip(np.asarray(initial_y, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    result = least_squares(
        lambda trial: pressure_thin_residual(logR, trial, lambda0, params, slopes),
        y0,
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=80,
    )
    return np.asarray(result.x, dtype=float)


def outer_residual(z: np.ndarray, params: TransonicSlimParams, closure: str, slopes) -> np.ndarray:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    y = np.array([logu[-1], logT[-1]], dtype=float)
    if closure == "pressure_thin_direct":
        return pressure_thin_residual(logR[-1], y, lambda0, params, slopes)
    if closure == "matched_pressure_thin":
        y_match = matched_pressure_thin_state(logR[-1], lambda0, params, slopes, initial_y=y)
        return y - y_match
    if closure == "matched_full":
        y_match = matched_outer_state(logR[-1], lambda0, params, g_match=slopes, initial_y=y)
        return y - y_match
    raise ValueError(f"unknown closure {closure!r}")


def custom_square_residual(z: np.ndarray, params: TransonicSlimParams, pivot: str, closure: str, slopes) -> np.ndarray:
    residual = square_collocation_residual(z, params, pivot=pivot)
    outer_row = 2 * (params.n_nodes - 1)
    residual[outer_row : outer_row + 2] = outer_residual(z, params, closure, slopes)
    return residual


def local_match_diagnostic(z: np.ndarray, params: TransonicSlimParams, closure: str, slopes) -> dict[str, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    y = np.array([logu[-1], logT[-1]], dtype=float)
    direct = pressure_thin_residual(logR[-1], y, lambda0, params, slopes)
    full = scaled_full_local_residual(logR[-1], y, slopes, lambda0, params)
    result = {
        "initial_pressure_omega": float(direct[0]),
        "initial_thin_energy": float(direct[1]),
        "initial_full_radial": float(full[0]),
        "initial_full_energy": float(full[1]),
        "target_dlogu": np.nan,
        "target_dlogT": np.nan,
        "target_pressure_omega": np.nan,
        "target_thin_energy": np.nan,
        "target_full_radial": np.nan,
        "target_full_energy": np.nan,
    }
    if closure == "pressure_thin_direct":
        return result
    if closure == "matched_pressure_thin":
        y_match = matched_pressure_thin_state(logR[-1], lambda0, params, slopes, initial_y=y)
    elif closure == "matched_full":
        y_match = matched_outer_state(logR[-1], lambda0, params, g_match=slopes, initial_y=y)
    else:
        raise ValueError(f"unknown closure {closure!r}")
    target_direct = pressure_thin_residual(logR[-1], y_match, lambda0, params, slopes)
    target_full = scaled_full_local_residual(logR[-1], y_match, slopes, lambda0, params)
    result.update(
        {
            "target_dlogu": float(y[0] - y_match[0]),
            "target_dlogT": float(y[1] - y_match[1]),
            "target_pressure_omega": float(target_direct[0]),
            "target_thin_energy": float(target_direct[1]),
            "target_full_radial": float(target_full[0]),
            "target_full_energy": float(target_full[1]),
        }
    )
    return result


def solve_variant(z0: np.ndarray, params: TransonicSlimParams, pivot: str, closure: str, slopes):
    lower, upper = state_bounds(params)
    z_start = np.clip(z0, lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: custom_square_residual(trial, params, pivot, closure, slopes),
        z_start,
        jac_sparsity=square_jac_sparsity_pattern(params),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV,
    )


def dominant_block(values: dict[str, float]) -> str:
    return max(values, key=lambda key: abs(values[key]))


def row_from_result(
    label: str,
    ratio: float,
    R_out_rg: float,
    slope_source: str,
    slopes: tuple[float, float],
    closure: str,
    pivot: str,
    params: TransonicSlimParams,
    z0: np.ndarray,
    result,
) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    selected_initial = custom_square_residual(z0, params, pivot, closure, slopes)
    selected_final = custom_square_residual(z, params, pivot, closure, slopes)
    audit = residual_audit_from_state_vector(z, params)
    outer = outer_residual(z, params, closure, slopes)
    thin_audit = residual_audit_from_state_vector(z, replace(params, outer_closure="thin_value", outer_match_log_slopes=None))
    profile = profile_from_state_vector(z, params)
    physical_values = {
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_1": float(outer[0]),
        "outer_2": float(outer[1]),
        "D": float(audit.sonic_D),
        "K": float(audit.sonic_K),
    }
    return {
        "label": label,
        "ratio": ratio,
        "R_out_rg": R_out_rg,
        "slope_source": slope_source,
        "g_u": float(slopes[0]),
        "g_T": float(slopes[1]),
        "closure": closure,
        "pivot": pivot,
        "selected_initial": float(np.max(np.abs(selected_initial))),
        "selected_final": float(np.max(np.abs(selected_final))),
        "physical_active": float(max(abs(value) for value in physical_values.values())),
        "dominant": dominant_block(physical_values),
        "interval_R": physical_values["interval_R"],
        "interval_E": physical_values["interval_E"],
        "outer_1": float(outer[0]),
        "outer_2": float(outer[1]),
        "legacy_outer_Omega": float(thin_audit.outer_omega),
        "legacy_outer_E": float(thin_audit.outer_energy),
        "D": float(audit.sonic_D),
        "C1": float(audit.sonic_C1),
        "C2": float(audit.sonic_C2),
        "K": float(audit.sonic_K),
        "compat_max": max(abs(audit.sonic_C1), abs(audit.sonic_C2), abs(audit.sonic_K)),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": float(audit.outer_H_over_R),
        "outer_Qadv_Qvisc": float(audit.outer_Qadv_over_Qvisc),
        "int_adv": float(profile.integrated_advective_fraction),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{row['label']}_R{float(row['R_out_rg']):.0f}_{row['slope_source']}_"
        f"{row['closure']}_{row['pivot']}_{float(row['ratio']):.8f}"
    ).replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(row["R_out_rg"]),
        pivot=np.array(row["pivot"]),
        row_json=np.array(row_json(payload)),
    )


def write_table(diagnostics: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Outer Thermal Match Audit",
        "",
        "Generated by `scripts/run_transonic_outer_thermal_match_audit.py`.",
        "",
        "This audit separates the remaining finite-Mdot residual floor into slope-source, outer thermal-condition, and finite-`R_out` pieces. The `R_out != 3000` rows are one-shot remap probes, not independent continuation branches.",
        "",
        "## Initial Outer Diagnostics",
        "",
        "| label | R_out/rg | slope | closure | g_u | g_T | initial pressure Omega | initial thin E | initial full R | initial full E | target dlogu | target dlogT | target pressure Omega | target thin E | target full R | target full E |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics:
        lines.append(
            "| {label} | {R_out_rg} | {slope_source} | {closure} | {g_u} | {g_T} | "
            "{initial_pressure_omega} | {initial_thin_energy} | {initial_full_radial} | {initial_full_energy} | "
            "{target_dlogu} | {target_dlogT} | {target_pressure_omega} | {target_thin_energy} | {target_full_radial} | {target_full_energy} |".format(
                label=row["label"],
                R_out_rg=fmt(float(row["R_out_rg"])),
                slope_source=row["slope_source"],
                closure=row["closure"],
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                initial_pressure_omega=fmt(float(row["initial_pressure_omega"])),
                initial_thin_energy=fmt(float(row["initial_thin_energy"])),
                initial_full_radial=fmt(float(row["initial_full_radial"])),
                initial_full_energy=fmt(float(row["initial_full_energy"])),
                target_dlogu=fmt(float(row["target_dlogu"])),
                target_dlogT=fmt(float(row["target_dlogT"])),
                target_pressure_omega=fmt(float(row["target_pressure_omega"])),
                target_thin_energy=fmt(float(row["target_thin_energy"])),
                target_full_radial=fmt(float(row["target_full_radial"])),
                target_full_energy=fmt(float(row["target_full_energy"])),
            )
        )
    lines.extend(
        [
            "",
            "## Fixed-Mdot Solves",
            "",
            "| label | R_out/rg | slope | closure | pivot | selected initial | selected final | physical active | dominant | interval R | interval E | outer 1 | outer 2 | legacy Omega | legacy E | D | C1 | C2 | K | compat max | max H/R | outer H/R | outer Qadv/Qvisc | int adv | nfev | success | message |",
            "|---|---:|---|---|:---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | {R_out_rg} | {slope_source} | {closure} | {pivot} | {selected_initial} | {selected_final} | "
            "{physical_active} | {dominant} | {interval_R} | {interval_E} | {outer_1} | {outer_2} | {legacy_outer_Omega} | "
            "{legacy_outer_E} | {D} | {C1} | {C2} | {K} | {compat_max} | {max_HR} | {outer_HR} | {outer_Qadv_Qvisc} | "
            "{int_adv} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                R_out_rg=fmt(float(row["R_out_rg"])),
                slope_source=row["slope_source"],
                closure=row["closure"],
                pivot=row["pivot"],
                selected_initial=fmt(float(row["selected_initial"])),
                selected_final=fmt(float(row["selected_final"])),
                physical_active=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                outer_1=fmt(float(row["outer_1"])),
                outer_2=fmt(float(row["outer_2"])),
                legacy_outer_Omega=fmt(float(row["legacy_outer_Omega"])),
                legacy_outer_E=fmt(float(row["legacy_outer_E"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                compat_max=fmt(float(row["compat_max"])),
                max_HR=fmt(float(row["max_HR"])),
                outer_HR=fmt(float(row["outer_HR"])),
                outer_Qadv_Qvisc=fmt(float(row["outer_Qadv_Qvisc"])),
                int_adv=fmt(float(row["int_adv"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def specs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slope_source in ("polyfit", "local_ode"):
        for closure in ("pressure_thin_direct", "matched_pressure_thin", "matched_full"):
            for pivot in PIVOTS:
                rows.append(
                    {
                        "label": "core",
                        "R_out_rg": CORE_R_OUT_RG,
                        "slope_source": slope_source,
                        "closure": closure,
                        "pivot": pivot,
                    }
                )
    for R_out_rg in R_OUT_PROBE_RG:
        for closure in ("pressure_thin_direct", "matched_pressure_thin", "matched_full"):
            rows.append(
                {
                    "label": "Rout_probe",
                    "R_out_rg": R_out_rg,
                    "slope_source": "local_ode",
                    "closure": closure,
                    "pivot": "C1",
                }
            )
    return rows


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_z, ratio = load_source()
    source_params = params_for(fiducial, mdot_edd, ratio, SOURCE_R_OUT_RG)

    diagnostics: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for spec in specs():
        params = params_for(fiducial, mdot_edd, ratio, float(spec["R_out_rg"]))
        z0 = (
            source_z
            if np.isclose(float(spec["R_out_rg"]), SOURCE_R_OUT_RG)
            else remap_state_with_outer_extrapolation(source_z, source_params, params)
        )
        slopes = slope_pair(z0, params, str(spec["slope_source"]))
        diagnostic = {
            **spec,
            "g_u": slopes[0],
            "g_T": slopes[1],
            **local_match_diagnostic(z0, params, str(spec["closure"]), slopes),
        }
        diagnostics.append(diagnostic)
        write_table(diagnostics, rows)

        result = solve_variant(z0, params, str(spec["pivot"]), str(spec["closure"]), slopes)
        row = row_from_result(
            str(spec["label"]),
            ratio,
            float(spec["R_out_rg"]),
            str(spec["slope_source"]),
            slopes,
            str(spec["closure"]),
            str(spec["pivot"]),
            params,
            z0,
            result,
        )
        rows.append(row)
        save_checkpoint(row)
        write_table(diagnostics, rows)
        print(
            f"{row['label']} R={row['R_out_rg']:.0f} {row['slope_source']} {row['closure']} "
            f"pivot={row['pivot']} selected {row['selected_initial']:.3e}->{row['selected_final']:.3e} "
            f"physical={row['physical_active']:.3e} dom={row['dominant']}",
            flush=True,
        )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
