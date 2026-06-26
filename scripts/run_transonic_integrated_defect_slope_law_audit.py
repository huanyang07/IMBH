"""Test integrated collocation defects from the best slope-law fixed-Mdot root."""

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

from run_transonic_full_slope_tuned_sonic_audit import residual_vector, sparsity_pattern
from run_transonic_outer_slope_calibration_audit import fmt


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "transonic_outer_slope_law_audit"
    / "R6500_baseline_0p90277664.npz"
)
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_integrated_defect_slope_law_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_integrated_defect_slope_law_audit"

N_NODES = 64
MAX_NFEV = 350
INTERVAL_SPECS = (
    ("differential", "differential", "none"),
    ("integrated_raw", "integrated", "none"),
    ("integrated_l2", "integrated", "inverse_sqrt_dx"),
)
SONIC_SPECS = (
    ("pivot_C1", "C1", ("D", "C1")),
    ("pivot_C2", "C2", ("D", "C2")),
    ("symmetric_D_C1_C2", "symmetric", ("D", "C1", "C2")),
    ("symmetric_D_C1_C2_K", "symmetric", ("D", "C1", "C2", "K")),
)


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


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    source: dict[str, object],
    interval_form: str,
    integrated_weighting: str,
) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(source["ratio"]) * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=N_NODES,
        R_out_rg=float(source["R_out_rg"]),
        residual_tol=1.0e-6,
        max_nfev=MAX_NFEV,
        outer_closure="full_slope_match",
        outer_match_log_slopes=(float(source["g_u"]), float(source["g_T"])),
        interval_residual_form=interval_form,
        integrated_residual_weighting=integrated_weighting,
    )


def solve_variant(
    z0: np.ndarray,
    params: TransonicSlimParams,
    mode: str,
    pivot: str,
    sonic_components: tuple[str, ...],
):
    lower, upper = state_bounds(params)
    z_start = np.clip(np.asarray(z0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: residual_vector(trial, params, mode, pivot, sonic_components),
        z_start,
        jac_sparsity=sparsity_pattern(params, mode, sonic_components),
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
    interval_label: str,
    sonic_label: str,
    mode: str,
    pivot: str,
    sonic_components: tuple[str, ...],
    params: TransonicSlimParams,
    z0: np.ndarray,
    result,
) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    initial = residual_vector(z0, params, mode, pivot, sonic_components)
    final = residual_vector(z, params, mode, pivot, sonic_components)
    audit = residual_audit_from_state_vector(z, params)
    differential_audit = residual_audit_from_state_vector(z, replace(params, interval_residual_form="differential", integrated_residual_weighting="none"))
    profile = profile_from_state_vector(z, params)
    slopes = params.outer_match_log_slopes or (np.nan, np.nan)
    return {
        "interval_label": interval_label,
        "sonic_label": sonic_label,
        "mode": mode,
        "sonic_components": sonic_components,
        "ratio": params.mdot_edd_ratio,
        "R_out_rg": params.R_out_rg,
        "interval_form": params.interval_residual_form,
        "integrated_weighting": params.integrated_residual_weighting,
        "g_u": float(slopes[0]),
        "g_T": float(slopes[1]),
        "initial_selected": float(np.max(np.abs(initial))),
        "final_selected": float(np.max(np.abs(final))),
        "final_physical": active_physical_max(audit),
        "differential_physical": active_physical_max(differential_audit),
        "dominant": dominant_block(audit),
        "differential_dominant": dominant_block(differential_audit),
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "diff_interval_R": differential_audit.interval_radial_max,
        "diff_interval_E": differential_audit.interval_energy_max,
        "outer_1": audit.outer_omega,
        "outer_2": audit.outer_energy,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "K": audit.sonic_K,
        "compat_max": max(abs(audit.sonic_C1), abs(audit.sonic_C2), abs(audit.sonic_K)),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0": float(profile.lambda0),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": audit.outer_H_over_R,
        "int_adv": profile.integrated_advective_fraction,
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{row['interval_label']}_{row['sonic_label']}_{float(row['ratio']):.8f}"
    stem = stem.replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(row["R_out_rg"]),
        interval_label=np.array(row["interval_label"]),
        sonic_label=np.array(row["sonic_label"]),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Integrated Defect Slope-Law Audit",
        "",
        "Generated by `scripts/run_transonic_integrated_defect_slope_law_audit.py`.",
        "",
        "Starts from the best slope-law fixed-Mdot root, `R_out=6500 rg`, and compares differential versus integrated interval residuals while always reporting the unweighted differential physical audit.",
        "",
        "| interval | sonic | selected final | physical final | differential physical | dominant | differential dominant | interval R | diff interval R | outer 1 | outer 2 | D | C1 | C2 | K | compat max | Rson/rg | lambda0 | max H/R | outer H/R | int adv | nfev | success | message |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {interval_label} | {sonic_label} | {final_selected} | {final_physical} | {differential_physical} | "
            "{dominant} | {differential_dominant} | {interval_R} | {diff_interval_R} | {outer_1} | {outer_2} | "
            "{D} | {C1} | {C2} | {K} | {compat_max} | {Rson_rg} | {lambda0} | {max_HR} | {outer_HR} | "
            "{int_adv} | {nfev} | {success} | {message} |".format(
                interval_label=row["interval_label"],
                sonic_label=row["sonic_label"],
                final_selected=fmt(float(row["final_selected"])),
                final_physical=fmt(float(row["final_physical"])),
                differential_physical=fmt(float(row["differential_physical"])),
                dominant=row["dominant"],
                differential_dominant=row["differential_dominant"],
                interval_R=fmt(float(row["interval_R"])),
                diff_interval_R=fmt(float(row["diff_interval_R"])),
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
    seed_z, source = load_source()

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for interval_label, interval_form, integrated_weighting in INTERVAL_SPECS:
        params = params_for(fiducial, mdot_edd, source, interval_form, integrated_weighting)
        for sonic_label, mode, sonic_components in SONIC_SPECS:
            pivot = "C1" if mode == "symmetric" else mode
            result = solve_variant(seed_z, params, mode, pivot, sonic_components)
            row = row_from_result(
                interval_label=interval_label,
                sonic_label=sonic_label,
                mode=mode,
                pivot=pivot,
                sonic_components=sonic_components,
                params=params,
                z0=seed_z,
                result=result,
            )
            rows.append(row)
            save_checkpoint(row)
            write_table(rows)
            print(
                f"{interval_label} {sonic_label} physical={row['final_physical']:.3e} "
                f"diff={row['differential_physical']:.3e} dom={row['dominant']} nfev={row['nfev']}",
                flush=True,
            )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
