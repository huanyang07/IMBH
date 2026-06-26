"""Nested-grid validation for the best slope-law fixed-Mdot root."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    computational_grid,
    pack_state,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    state_bounds,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_full_slope_tuned_sonic_audit import residual_vector, sparsity_pattern
from run_transonic_outer_slope_calibration_audit import fmt


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "transonic_integrated_defect_slope_law_audit"
    / "differential_symmetric_D_C1_C2_0p90277664.npz"
)
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_slope_law_nested_grid_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_slope_law_nested_grid_audit"

NODES = (33, 65, 129)
MAX_NFEV = 450
SONIC_MODE = "symmetric"
SONIC_PIVOT = "C1"
SONIC_COMPONENTS = ("D", "C1", "C2")


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


def load_source() -> tuple[np.ndarray, dict[str, object]]:
    with np.load(SOURCE_CHECKPOINT, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), json.loads(str(data["row_json"].item()))


def params_for(fiducial: FiducialParams, mdot_edd: float, source: dict[str, object], n_nodes: int) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(source["ratio"]) * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=n_nodes,
        R_out_rg=float(source["R_out_rg"]),
        residual_tol=1.0e-6,
        max_nfev=MAX_NFEV,
        outer_closure="full_slope_match",
        outer_match_log_slopes=(float(source["g_u"]), float(source["g_T"])),
        interval_residual_form=str(source["interval_form"]),
        integrated_residual_weighting=str(source["integrated_weighting"]),
    )


def remap_state(z: np.ndarray, old_params: TransonicSlimParams, new_params: TransonicSlimParams) -> np.ndarray:
    logu_old, logT_old, logR_son, lambda0, logR_old = unpack_state(z, old_params)
    logR_new = computational_grid(new_params, logR_son)
    logu_new = np.interp(logR_new, logR_old, logu_old)
    logT_new = np.interp(logR_new, logR_old, logT_old)
    return pack_state(logu_new, logT_new, logR_son, lambda0)


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


def max_profile_difference(z: np.ndarray, params: TransonicSlimParams, source_z: np.ndarray, source_params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    source_logu, source_logT, _source_logR_son, _source_lambda0, source_logR = unpack_state(source_z, source_params)
    source_logu_interp = np.interp(logR, source_logR, source_logu)
    source_logT_interp = np.interp(logR, source_logR, source_logT)
    return float(np.max(np.abs(logu - source_logu_interp))), float(np.max(np.abs(logT - source_logT_interp)))


def row_from_result(
    *,
    params: TransonicSlimParams,
    z0: np.ndarray,
    result,
    source_z: np.ndarray,
    source_params: TransonicSlimParams,
) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    initial = residual_vector(z0, params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS)
    final = residual_vector(z, params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS)
    audit = residual_audit_from_state_vector(z, params)
    source_audit = residual_audit_from_state_vector(source_z, source_params)
    profile = profile_from_state_vector(z, params)
    source_profile = profile_from_state_vector(source_z, source_params)
    max_dlogu, max_dlogT = max_profile_difference(z, params, source_z, source_params)
    slopes = params.outer_match_log_slopes or (np.nan, np.nan)
    return {
        "n_nodes": params.n_nodes,
        "ratio": params.mdot_edd_ratio,
        "R_out_rg": params.R_out_rg,
        "interval_form": params.interval_residual_form,
        "integrated_weighting": params.integrated_residual_weighting,
        "g_u": float(slopes[0]),
        "g_T": float(slopes[1]),
        "initial_selected": float(np.max(np.abs(initial))),
        "final_selected": float(np.max(np.abs(final))),
        "final_physical": active_physical_max(audit),
        "dominant": dominant_block(audit),
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_1": audit.outer_omega,
        "outer_2": audit.outer_energy,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "K": audit.sonic_K,
        "compat_max": max(abs(audit.sonic_C1), abs(audit.sonic_C2), abs(audit.sonic_K)),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "delta_Rson_rg": float(profile.sonic_radius / params.r_g - source_profile.sonic_radius / source_params.r_g),
        "lambda0": float(profile.lambda0),
        "delta_lambda0": float(profile.lambda0 - source_profile.lambda0),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": audit.outer_H_over_R,
        "int_adv": profile.integrated_advective_fraction,
        "delta_int_adv": float(profile.integrated_advective_fraction - source_profile.integrated_advective_fraction),
        "max_dlogu_vs_N64": max_dlogu,
        "max_dlogT_vs_N64": max_dlogT,
        "source_physical": active_physical_max(source_audit),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"N{int(row['n_nodes'])}_{float(row['ratio']):.8f}".replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(row["R_out_rg"]),
        n_nodes=np.array(row["n_nodes"]),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Slope-Law Nested Grid Audit",
        "",
        "Generated by `scripts/run_transonic_slope_law_nested_grid_audit.py`.",
        "",
        "Uses the best integrated-defect result, fixed `R_out=6500 rg`, fixed full-slope boundary slopes, differential interval residuals, and symmetric `D,C1,C2` sonic rows. Each grid is seeded by remapping the N64 source, not chained from another grid.",
        "",
        "| N | selected final | physical final | dominant | interval R | outer 1 | D | C1 | C2 | K | Rson/rg | dRson/rg | lambda0 | dlambda0 | max H/R | outer H/R | int adv | dint adv | max dlogu vs N64 | max dlogT vs N64 | nfev | success | message |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {n_nodes} | {final_selected} | {final_physical} | {dominant} | {interval_R} | {outer_1} | "
            "{D} | {C1} | {C2} | {K} | {Rson_rg} | {delta_Rson_rg} | {lambda0} | {delta_lambda0} | "
            "{max_HR} | {outer_HR} | {int_adv} | {delta_int_adv} | {max_dlogu_vs_N64} | {max_dlogT_vs_N64} | "
            "{nfev} | {success} | {message} |".format(
                n_nodes=row["n_nodes"],
                final_selected=fmt(float(row["final_selected"])),
                final_physical=fmt(float(row["final_physical"])),
                dominant=row["dominant"],
                interval_R=fmt(float(row["interval_R"])),
                outer_1=fmt(float(row["outer_1"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row["delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row["delta_lambda0"])),
                max_HR=fmt(float(row["max_HR"])),
                outer_HR=fmt(float(row["outer_HR"])),
                int_adv=fmt(float(row["int_adv"])),
                delta_int_adv=fmt(float(row["delta_int_adv"])),
                max_dlogu_vs_N64=fmt(float(row["max_dlogu_vs_N64"])),
                max_dlogT_vs_N64=fmt(float(row["max_dlogT_vs_N64"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_z, source = load_source()
    source_params = params_for(fiducial, mdot_edd, source, n_nodes=64)

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for n_nodes in NODES:
        params = params_for(fiducial, mdot_edd, source, n_nodes=n_nodes)
        z_seed = remap_state(source_z, source_params, params)
        result = solve_variant(z_seed, params)
        row = row_from_result(params=params, z0=z_seed, result=result, source_z=source_z, source_params=source_params)
        rows.append(row)
        save_checkpoint(row)
        write_table(rows)
        print(
            f"N={n_nodes} physical={row['final_physical']:.3e} dom={row['dominant']} "
            f"Rson={row['Rson_rg']:.4f} nfev={row['nfev']}",
            flush=True,
        )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
