"""Calibrate smooth outer slope estimates for the direct full-slope closure."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    profile_from_state_vector,
    reduced_outer_log_slopes,
    residual_audit_from_state_vector,
    square_collocation_residual,
    state_bounds,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_full_slope_sensitivity_audit import load_checkpoint
from run_transonic_full_slope_tuned_sonic_audit import residual_vector, sparsity_pattern


ROOT = Path(__file__).resolve().parents[1]
SLOPE_SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_rout_continuation_audit" / "R5000_refresh2_0p90277664.npz"
BEST_POLISH_CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "transonic_full_slope_best_polish"
    / "g_T_plus_1e-3_symmetric_D_C1_C2_0p90277664.npz"
)
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_outer_slope_calibration_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_outer_slope_calibration_audit"

R_OUT_RG = 5000.0
N_NODES = 64
MAX_NFEV = 300
SONIC_MODE = "symmetric"
SONIC_PIVOT = "C1"
SONIC_COMPONENTS = ("D", "C1", "C2")


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def params_for(fiducial: FiducialParams, mdot_edd: float, ratio: float, slopes: tuple[float, float]) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=N_NODES,
        R_out_rg=R_OUT_RG,
        residual_tol=1.0e-6,
        max_nfev=MAX_NFEV,
        outer_closure="full_slope_match",
        outer_match_log_slopes=slopes,
        interval_residual_form="differential",
    )


def polyfit_outer_log_slopes(
    z: np.ndarray,
    params: TransonicSlimParams,
    *,
    n_fit: int = 8,
    degree: int = 2,
    skip_outer: int = 0,
) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    stop = len(logR) - int(skip_outer)
    if stop < 3:
        raise ValueError("skip_outer leaves too few points for an outer fit")
    count = min(int(n_fit), stop)
    start = stop - count
    x = logR[start:stop] - logR[-1]
    fit_degree = min(int(degree), count - 1)
    gu_poly = np.poly1d(np.polyfit(x, logu[start:stop], fit_degree))
    gT_poly = np.poly1d(np.polyfit(x, logT[start:stop], fit_degree))
    return float(np.polyder(gu_poly)(0.0)), float(np.polyder(gT_poly)(0.0))


def slope_from_spec(spec: dict[str, object], profiles: dict[str, tuple[np.ndarray, TransonicSlimParams]]) -> tuple[float, float]:
    source = str(spec["profile_source"])
    z, params = profiles[source]
    if spec["estimator"] == "reduced":
        _logu, _logT, _logR_son, lambda0, _logR = unpack_state(z, params)
        slopes = reduced_outer_log_slopes(params, lambda0)
    else:
        slopes = polyfit_outer_log_slopes(
            z,
            params,
            n_fit=int(spec["n_fit"]),
            degree=int(spec["degree"]),
            skip_outer=int(spec["skip_outer"]),
        )
    correction = np.asarray(spec["correction"], dtype=float)
    return float(slopes[0] + correction[0]), float(slopes[1] + correction[1])


def slope_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for n_fit in (5, 6, 8, 10, 12, 16):
        for degree in (1, 2, 3):
            specs.append(
                {
                    "label": f"rout_poly_o{degree}_n{n_fit}",
                    "profile_source": "rout_matched",
                    "estimator": "polyfit",
                    "n_fit": n_fit,
                    "degree": degree,
                    "skip_outer": 0,
                    "correction": (0.0, 0.0),
                }
            )
    for n_fit in (8, 12):
        for degree in (1, 2):
            specs.append(
                {
                    "label": f"best_poly_o{degree}_n{n_fit}",
                    "profile_source": "best_polish",
                    "estimator": "polyfit",
                    "n_fit": n_fit,
                    "degree": degree,
                    "skip_outer": 0,
                    "correction": (0.0, 0.0),
                }
            )
    for skip_outer in (1, 2):
        specs.append(
            {
                "label": f"rout_poly_o2_n8_skip{skip_outer}",
                "profile_source": "rout_matched",
                "estimator": "polyfit",
                "n_fit": 8,
                "degree": 2,
                "skip_outer": skip_outer,
                "correction": (0.0, 0.0),
            }
        )
    specs.extend(
        [
            {
                "label": "rout_reduced",
                "profile_source": "rout_matched",
                "estimator": "reduced",
                "n_fit": 0,
                "degree": 0,
                "skip_outer": 0,
                "correction": (0.0, 0.0),
            },
            {
                "label": "rout_poly_o2_n8_gT_plus_1e-3",
                "profile_source": "rout_matched",
                "estimator": "polyfit",
                "n_fit": 8,
                "degree": 2,
                "skip_outer": 0,
                "correction": (0.0, 1.0e-3),
            },
            {
                "label": "rout_poly_o2_n8_gu_minus_3e-3",
                "profile_source": "rout_matched",
                "estimator": "polyfit",
                "n_fit": 8,
                "degree": 2,
                "skip_outer": 0,
                "correction": (-3.0e-3, 0.0),
            },
        ]
    )
    return specs


def solve_variant(z0: np.ndarray, params: TransonicSlimParams):
    lower, upper = state_bounds(params)
    z_start = np.clip(np.asarray(z0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: residual_vector(trial, params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS),
        z_start,
        jac_sparsity=sparsity_pattern(params, SONIC_MODE, SONIC_COMPONENTS),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV,
    )


def active_physical_max(audit) -> float:
    return max(
        abs(audit.interval_radial_max),
        abs(audit.interval_energy_max),
        abs(audit.outer_omega),
        abs(audit.outer_energy),
        abs(audit.sonic_D),
        abs(audit.sonic_C1),
        abs(audit.sonic_C2),
        abs(audit.sonic_K),
    )


def dominant_block(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_1": abs(audit.outer_omega),
        "outer_2": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
        "K": abs(audit.sonic_K),
    }
    return max(values, key=values.get)


def row_from_result(
    *,
    spec: dict[str, object],
    ratio: float,
    params: TransonicSlimParams,
    z0: np.ndarray,
    result,
) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    initial = residual_vector(z0, params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS)
    final = residual_vector(z, params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS)
    initial_audit = residual_audit_from_state_vector(z0, params)
    audit = residual_audit_from_state_vector(z, params)
    legacy_audit = residual_audit_from_state_vector(z, replace(params, outer_closure="thin_value", outer_match_log_slopes=None))
    output_slopes = polyfit_outer_log_slopes(z, params, n_fit=8, degree=2)
    profile = profile_from_state_vector(z, params)
    slopes = params.outer_match_log_slopes or (np.nan, np.nan)
    correction = tuple(float(value) for value in spec["correction"])
    return {
        "label": spec["label"],
        "profile_source": spec["profile_source"],
        "estimator": spec["estimator"],
        "n_fit": int(spec["n_fit"]),
        "degree": int(spec["degree"]),
        "skip_outer": int(spec["skip_outer"]),
        "dg_u": correction[0],
        "dg_T": correction[1],
        "ratio": ratio,
        "sonic_mode": SONIC_MODE,
        "sonic_components": SONIC_COMPONENTS,
        "g_u": float(slopes[0]),
        "g_T": float(slopes[1]),
        "g_u_out": float(output_slopes[0]),
        "g_T_out": float(output_slopes[1]),
        "slope_delta_out": float(np.max(np.abs(np.asarray(output_slopes) - np.asarray(slopes, dtype=float)))),
        "initial_selected": float(np.max(np.abs(initial))),
        "initial_physical": active_physical_max(initial_audit),
        "initial_dominant": dominant_block(initial_audit),
        "final_selected": float(np.max(np.abs(final))),
        "final_physical": active_physical_max(audit),
        "dominant": dominant_block(audit),
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_1": audit.outer_omega,
        "outer_2": audit.outer_energy,
        "legacy_outer_Omega": legacy_audit.outer_omega,
        "legacy_outer_E": legacy_audit.outer_energy,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "K": audit.sonic_K,
        "compat_max": max(abs(audit.sonic_C1), abs(audit.sonic_C2), abs(audit.sonic_K)),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0": float(profile.lambda0),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": audit.outer_H_over_R,
        "outer_Qadv_Qvisc": audit.outer_Qadv_over_Qvisc,
        "int_adv": profile.integrated_advective_fraction,
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{row['label']}_{float(row['ratio']):.8f}".replace("+", "p").replace("-", "m").replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        g_u=np.array(row["g_u"]),
        g_T=np.array(row["g_T"]),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Outer Slope Calibration Audit",
        "",
        "Generated by `scripts/run_transonic_outer_slope_calibration_audit.py`.",
        "",
        "Fixed `Mdot/Edd ~= 0.9028`, `N=64`, `R_out=5000 rg`, direct `full_slope_match` closure, symmetric `D,C1,C2` sonic rows. This tests how the final fixed-Mdot residual depends on the slope estimator itself.",
        "",
        "| label | source | estimator | n_fit | degree | skip | dg_u | dg_T | g_u | g_T | initial physical | final physical | dominant | slope delta out | g_u out | g_T out | interval R | interval E | outer 1 | outer 2 | D | C1 | C2 | K | compat max | Rson/rg | lambda0 | max H/R | outer H/R | int adv | nfev | success | message |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {profile_source} | {estimator} | {n_fit} | {degree} | {skip_outer} | {dg_u} | {dg_T} | "
            "{g_u} | {g_T} | {initial_physical} | {final_physical} | {dominant} | {slope_delta_out} | "
            "{g_u_out} | {g_T_out} | {interval_R} | {interval_E} | {outer_1} | {outer_2} | {D} | {C1} | "
            "{C2} | {K} | {compat_max} | {Rson_rg} | {lambda0} | {max_HR} | {outer_HR} | {int_adv} | "
            "{nfev} | {success} | {message} |".format(
                label=row["label"],
                profile_source=row["profile_source"],
                estimator=row["estimator"],
                n_fit=row["n_fit"],
                degree=row["degree"],
                skip_outer=row["skip_outer"],
                dg_u=fmt(float(row["dg_u"])),
                dg_T=fmt(float(row["dg_T"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                initial_physical=fmt(float(row["initial_physical"])),
                final_physical=fmt(float(row["final_physical"])),
                dominant=row["dominant"],
                slope_delta_out=fmt(float(row["slope_delta_out"])),
                g_u_out=fmt(float(row["g_u_out"])),
                g_T_out=fmt(float(row["g_T_out"])),
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                outer_1=fmt(float(row["outer_1"])),
                outer_2=fmt(float(row["outer_2"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                compat_max=fmt(float(row["compat_max"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                max_HR=fmt(float(row["max_HR"])),
                outer_HR=fmt(float(row["outer_HR"])),
                int_adv=fmt(float(row["int_adv"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rout_z, ratio = load_checkpoint(SLOPE_SOURCE_CHECKPOINT)
    best_z, best_ratio = load_checkpoint(BEST_POLISH_CHECKPOINT)
    if not np.isclose(ratio, best_ratio):
        raise RuntimeError("slope source and best-polish checkpoints have different accretion rates")

    base_params = params_for(fiducial, mdot_edd, ratio, slopes=(0.0, 0.0))
    profiles = {
        "rout_matched": (rout_z, base_params),
        "best_polish": (best_z, base_params),
    }

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for spec in slope_specs():
        slopes = slope_from_spec(spec, profiles)
        params = params_for(fiducial, mdot_edd, ratio, slopes=slopes)
        result = solve_variant(best_z, params)
        row = row_from_result(spec=spec, ratio=ratio, params=params, z0=best_z, result=result)
        rows.append(row)
        save_checkpoint(row)
        write_table(rows)
        print(
            f"{row['label']} g=({row['g_u']:.5f},{row['g_T']:.5f}) "
            f"final={row['final_physical']:.3e} dom={row['dominant']} nfev={row['nfev']}",
            flush=True,
        )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
