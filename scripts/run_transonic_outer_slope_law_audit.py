"""Fit a radius-dependent outer-slope correction for full-slope matching."""

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
    residual_audit_from_state_vector,
    state_bounds,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_full_slope_sensitivity_audit import load_checkpoint
from run_transonic_full_slope_tuned_sonic_audit import residual_vector, sparsity_pattern
from run_transonic_outer_slope_calibration_audit import fmt, polyfit_outer_log_slopes, row_json


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_outer_slope_law_audit.md"
LAW_OUTPUT = ROOT / "outputs" / "tables" / "transonic_outer_slope_law_coefficients.json"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_outer_slope_law_audit"

SOURCE_CHECKPOINTS = {
    5000.0: ROOT
    / "outputs"
    / "checkpoints"
    / "transonic_outer_slope_calibration_audit"
    / "rout_poly_o2_n8_gT_plus_1em3_0p90277664.npz",
    5500.0: ROOT
    / "outputs"
    / "checkpoints"
    / "transonic_controlled_slope_rout_continuation"
    / "raw_calibrated_R5500_0p90277664.npz",
    6000.0: ROOT
    / "outputs"
    / "checkpoints"
    / "transonic_controlled_slope_rout_continuation"
    / "raw_calibrated_R6000_0p90277664.npz",
    6500.0: ROOT
    / "outputs"
    / "checkpoints"
    / "transonic_controlled_slope_rout_continuation"
    / "raw_calibrated_R6500_0p90277664.npz",
}

N_NODES = 64
MAX_NFEV = 220
SONIC_MODE = "symmetric"
SONIC_PIVOT = "C1"
SONIC_COMPONENTS = ("D", "C1", "C2")
REFERENCE_R_OUT_RG = 5000.0


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def local_row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    ratio: float,
    R_out_rg: float,
    slopes: tuple[float, float],
    *,
    n_nodes: int = N_NODES,
) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=n_nodes,
        R_out_rg=R_out_rg,
        residual_tol=1.0e-6,
        max_nfev=MAX_NFEV,
        outer_closure="full_slope_match",
        outer_match_log_slopes=slopes,
        interval_residual_form="differential",
    )


def correction_specs() -> tuple[dict[str, object], ...]:
    specs: list[dict[str, object]] = [{"label": "baseline", "dg_u": 0.0, "dg_T": 0.0}]
    for delta in (-3.0e-3, -1.0e-3, 1.0e-3, 3.0e-3, 6.0e-3):
        specs.append({"label": f"gT_{delta:+.0e}", "dg_u": 0.0, "dg_T": delta})
    for delta in (-1.0e-2, -6.0e-3, -3.0e-3, 3.0e-3):
        specs.append({"label": f"gu_{delta:+.0e}", "dg_u": delta, "dg_T": 0.0})
    for dg_u, dg_T in ((-1.0e-2, -3.0e-3), (-6.0e-3, -3.0e-3), (-3.0e-3, 1.0e-3), (-6.0e-3, 1.0e-3)):
        specs.append({"label": f"gu_{dg_u:+.0e}_gT_{dg_T:+.0e}", "dg_u": dg_u, "dg_T": dg_T})
    return tuple(specs)


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
    R_out_rg: float,
    base_slopes: tuple[float, float],
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
    output_slopes = polyfit_outer_log_slopes(z, params, n_fit=8, degree=2, skip_outer=0)
    profile = profile_from_state_vector(z, params)
    slopes = params.outer_match_log_slopes or (np.nan, np.nan)
    return {
        "label": spec["label"],
        "ratio": ratio,
        "R_out_rg": R_out_rg,
        "dg_u": float(spec["dg_u"]),
        "dg_T": float(spec["dg_T"]),
        "g_u_base": float(base_slopes[0]),
        "g_T_base": float(base_slopes[1]),
        "g_u": float(slopes[0]),
        "g_T": float(slopes[1]),
        "g_u_out": float(output_slopes[0]),
        "g_T_out": float(output_slopes[1]),
        "slope_delta_out": float(np.max(np.abs(np.asarray(output_slopes) - np.asarray(slopes, dtype=float)))),
        "initial_selected": float(np.max(np.abs(initial))),
        "initial_physical": active_physical_max(initial_audit),
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
    stem = f"R{float(row['R_out_rg']):.0f}_{row['label']}_{float(row['ratio']):.8f}"
    stem = stem.replace("+", "p").replace("-", "m").replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(row["R_out_rg"]),
        g_u=np.array(row["g_u"]),
        g_T=np.array(row["g_T"]),
        dg_u=np.array(row["dg_u"]),
        dg_T=np.array(row["dg_T"]),
        row_json=np.array(local_row_json(payload)),
    )


def best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for radius in sorted({float(row["R_out_rg"]) for row in rows}):
        radius_rows = [row for row in rows if np.isclose(float(row["R_out_rg"]), radius)]
        result.append(min(radius_rows, key=lambda row: float(row["final_physical"])))
    return result


def fit_law(best: list[dict[str, object]]) -> dict[str, object]:
    radii = np.asarray([float(row["R_out_rg"]) for row in best], dtype=float)
    x = np.log(radii / REFERENCE_R_OUT_RG)
    degree = min(2, len(best) - 1)
    gu_coeff = np.polyfit(x, np.asarray([float(row["dg_u"]) for row in best], dtype=float), degree)
    gT_coeff = np.polyfit(x, np.asarray([float(row["dg_T"]) for row in best], dtype=float), degree)
    return {
        "reference_R_out_rg": REFERENCE_R_OUT_RG,
        "x_definition": "x = ln(R_out_rg / reference_R_out_rg)",
        "degree": int(degree),
        "dg_u_poly_coeff_descending": [float(value) for value in gu_coeff],
        "dg_T_poly_coeff_descending": [float(value) for value in gT_coeff],
        "best_rows": [
            {
                "R_out_rg": float(row["R_out_rg"]),
                "label": str(row["label"]),
                "dg_u": float(row["dg_u"]),
                "dg_T": float(row["dg_T"]),
                "g_u": float(row["g_u"]),
                "g_T": float(row["g_T"]),
                "final_physical": float(row["final_physical"]),
                "dominant": str(row["dominant"]),
            }
            for row in best
        ],
    }


def write_table(rows: list[dict[str, object]]) -> None:
    best = best_rows(rows)
    law = fit_law(best)
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Outer Slope Law Audit",
        "",
        "Generated by `scripts/run_transonic_outer_slope_law_audit.py`.",
        "",
        "Each radius uses a local polished seed, computes the local quadratic `n_fit=8` polyfit slope, applies finite corrections, and solves with direct `full_slope_match` plus symmetric `D,C1,C2` sonic rows.",
        "",
        "## Best By Radius",
        "",
        "| R_out/rg | label | dg_u | dg_T | g_u | g_T | final physical | dominant | slope delta out | interval R | C1 | C2 | K | nfev | success |",
        "|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in best:
        lines.append(
            "| {R_out_rg} | {label} | {dg_u} | {dg_T} | {g_u} | {g_T} | {final_physical} | {dominant} | "
            "{slope_delta_out} | {interval_R} | {C1} | {C2} | {K} | {nfev} | {success} |".format(
                R_out_rg=fmt(float(row["R_out_rg"])),
                label=row["label"],
                dg_u=fmt(float(row["dg_u"])),
                dg_T=fmt(float(row["dg_T"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                final_physical=fmt(float(row["final_physical"])),
                dominant=row["dominant"],
                slope_delta_out=fmt(float(row["slope_delta_out"])),
                interval_R=fmt(float(row["interval_R"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Fitted Correction Law",
            "",
            f"`{law['x_definition']}`",
            "",
            f"`dg_u(x)` coefficients, descending polynomial order: `{law['dg_u_poly_coeff_descending']}`",
            "",
            f"`dg_T(x)` coefficients, descending polynomial order: `{law['dg_T_poly_coeff_descending']}`",
            "",
            "## All Rows",
            "",
            "| R_out/rg | label | dg_u | dg_T | g_u base | g_T base | g_u | g_T | final physical | dominant | slope delta out | interval R | interval E | outer 1 | outer 2 | D | C1 | C2 | K | compat max | Rson/rg | lambda0 | max H/R | outer H/R | int adv | nfev | success | message |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {R_out_rg} | {label} | {dg_u} | {dg_T} | {g_u_base} | {g_T_base} | {g_u} | {g_T} | "
            "{final_physical} | {dominant} | {slope_delta_out} | {interval_R} | {interval_E} | {outer_1} | "
            "{outer_2} | {D} | {C1} | {C2} | {K} | {compat_max} | {Rson_rg} | {lambda0} | {max_HR} | "
            "{outer_HR} | {int_adv} | {nfev} | {success} | {message} |".format(
                R_out_rg=fmt(float(row["R_out_rg"])),
                label=row["label"],
                dg_u=fmt(float(row["dg_u"])),
                dg_T=fmt(float(row["dg_T"])),
                g_u_base=fmt(float(row["g_u_base"])),
                g_T_base=fmt(float(row["g_T_base"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                final_physical=fmt(float(row["final_physical"])),
                dominant=row["dominant"],
                slope_delta_out=fmt(float(row["slope_delta_out"])),
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
    LAW_OUTPUT.write_text(json.dumps(law, indent=2, sort_keys=True) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    ratio: float | None = None
    for R_out_rg, checkpoint in SOURCE_CHECKPOINTS.items():
        seed_z, seed_ratio = load_checkpoint(checkpoint)
        ratio = seed_ratio if ratio is None else ratio
        if not np.isclose(seed_ratio, ratio):
            raise RuntimeError("source checkpoints have different accretion rates")
        base_params = params_for(fiducial, mdot_edd, ratio, R_out_rg, slopes=(0.0, 0.0))
        base_slopes = polyfit_outer_log_slopes(seed_z, base_params, n_fit=8, degree=2, skip_outer=0)
        for spec in correction_specs():
            slopes = (base_slopes[0] + float(spec["dg_u"]), base_slopes[1] + float(spec["dg_T"]))
            params = params_for(fiducial, mdot_edd, ratio, R_out_rg, slopes=slopes)
            result = solve_variant(seed_z, params)
            row = row_from_result(
                spec=spec,
                ratio=ratio,
                R_out_rg=R_out_rg,
                base_slopes=base_slopes,
                params=params,
                z0=seed_z,
                result=result,
            )
            rows.append(row)
            save_checkpoint(row)
            write_table(rows)
            print(
                f"R={R_out_rg:.0f} {row['label']} final={row['final_physical']:.3e} "
                f"dom={row['dominant']} g=({row['g_u']:.5f},{row['g_T']:.5f})",
                flush=True,
            )

    print(f"wrote {TABLE_OUTPUT}")
    print(f"wrote {LAW_OUTPUT}")


if __name__ == "__main__":
    main()
