"""Compare nested matched outer state against direct full-slope matching."""

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
    square_collocation_jacobian,
    square_collocation_residual,
    square_jac_sparsity_pattern,
    state_bounds,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_rout_continuation_audit" / "R5000_refresh2_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_full_slope_match_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_full_slope_match_audit"

R_OUT_RG = 5000.0
N_NODES = 64
RESIDUAL_TOL = 1.0e-6
MAX_NFEV = 500
PIVOTS = ("C1", "C2")
SPECS = (
    ("matched_state_differential", "matched_outer_state", "differential", "none"),
    ("matched_state_integrated_raw", "matched_outer_state", "integrated", "none"),
    ("matched_state_integrated_l2", "matched_outer_state", "integrated", "inverse_sqrt_dx"),
    ("full_slope_differential", "full_slope_match", "differential", "none"),
    ("full_slope_integrated_raw", "full_slope_match", "integrated", "none"),
    ("full_slope_integrated_l2", "full_slope_match", "integrated", "inverse_sqrt_dx"),
)


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


def ruiz_equilibrate_dense(matrix: np.ndarray, n_iter: int = 5, floor: float = 1.0e-300) -> np.ndarray:
    equilibrated = np.asarray(matrix, dtype=float).copy()
    for _ in range(n_iter):
        row_norm = np.linalg.norm(equilibrated, axis=1)
        equilibrated = (1.0 / np.maximum(row_norm, floor))[:, None] * equilibrated
        col_norm = np.linalg.norm(equilibrated, axis=0)
        equilibrated = equilibrated * (1.0 / np.maximum(col_norm, floor))[None, :]
    return equilibrated


def condition_metrics(z: np.ndarray, params: TransonicSlimParams, pivot: str) -> dict[str, float]:
    jac = square_collocation_jacobian(z, params, pivot=pivot, rel_step=3.0e-5).toarray()
    raw_s = np.linalg.svd(jac, compute_uv=False)
    eq_s = np.linalg.svd(ruiz_equilibrate_dense(jac), compute_uv=False)
    return {
        "raw_cond": float(raw_s[0] / max(raw_s[-1], 1.0e-300)),
        "raw_smin": float(raw_s[-1]),
        "eq_cond": float(eq_s[0] / max(eq_s[-1], 1.0e-300)),
        "eq_smin": float(eq_s[-1]),
    }


def load_source() -> tuple[np.ndarray, float]:
    with np.load(SOURCE_CHECKPOINT, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), float(data["ratio"])


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    ratio: float,
    *,
    outer_closure: str,
    interval_form: str,
    integrated_weighting: str,
    slopes: tuple[float, float],
) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=N_NODES,
        R_out_rg=R_OUT_RG,
        residual_tol=RESIDUAL_TOL,
        max_nfev=MAX_NFEV,
        outer_closure=outer_closure,
        outer_match_log_slopes=slopes,
        interval_residual_form=interval_form,
        integrated_residual_weighting=integrated_weighting,
    )


def polyfit_outer_log_slopes(z: np.ndarray, params: TransonicSlimParams, n_fit: int = 8) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    count = min(int(n_fit), len(logR))
    x = logR[-count:] - logR[-1]
    gu = float(np.polyder(np.poly1d(np.polyfit(x, logu[-count:], min(2, count - 1))))(0.0))
    gT = float(np.polyder(np.poly1d(np.polyfit(x, logT[-count:], min(2, count - 1))))(0.0))
    return gu, gT


def solve_variant(z0: np.ndarray, params: TransonicSlimParams, pivot: str):
    lower, upper = state_bounds(params)
    z_start = np.clip(np.asarray(z0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: square_collocation_residual(trial, params, pivot=pivot),
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
    label: str,
    ratio: float,
    params: TransonicSlimParams,
    pivot: str,
    z0: np.ndarray,
    result,
    conditioning: dict[str, float],
) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    initial = square_collocation_residual(z0, params, pivot=pivot)
    final = square_collocation_residual(z, params, pivot=pivot)
    audit = residual_audit_from_state_vector(z, params)
    profile = profile_from_state_vector(z, params)
    output_slopes = polyfit_outer_log_slopes(z, params)
    input_slopes = params.outer_match_log_slopes or (np.nan, np.nan)
    return {
        "label": label,
        "ratio": ratio,
        "outer_closure": params.outer_closure,
        "interval_form": params.interval_residual_form,
        "integrated_weighting": params.integrated_residual_weighting,
        "pivot": pivot,
        "g_u_in": float(input_slopes[0]),
        "g_T_in": float(input_slopes[1]),
        "g_u_out": float(output_slopes[0]),
        "g_T_out": float(output_slopes[1]),
        "slope_delta": float(np.max(np.abs(np.asarray(output_slopes) - np.asarray(input_slopes, dtype=float)))),
        "selected_initial": float(np.max(np.abs(initial))),
        "selected_final": float(np.max(np.abs(final))),
        "physical_active": active_physical_max(audit),
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
        "lambda0": float(profile.lambda0),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": audit.outer_H_over_R,
        "outer_Qadv_Qvisc": audit.outer_Qadv_over_Qvisc,
        "int_adv": profile.integrated_advective_fraction,
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        **conditioning,
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{row['label']}_{row['pivot']}_{float(row['ratio']):.8f}".replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        pivot=np.array(row["pivot"]),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full Slope Match Audit",
        "",
        "Generated by `scripts/run_transonic_full_slope_match_audit.py`.",
        "",
        "Starts from the best `R_out=5000 rg`, `Mdot/Edd ~= 0.9028` checkpoint and compares the old nested `matched_outer_state` closure with the direct `full_slope_match` closure.",
        "",
        "| label | closure | interval | weighting | pivot | selected initial | selected final | physical active | dominant | raw cond | eq cond | raw smin | eq smin | slope delta | g_u in | g_T in | g_u out | g_T out | interval R | interval E | outer 1 | outer 2 | D | C1 | C2 | K | compat max | Rson/rg | lambda0 | max H/R | outer H/R | outer Qadv/Qvisc | int adv | nfev | success | message |",
        "|---|---|---|---|:---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {outer_closure} | {interval_form} | {integrated_weighting} | {pivot} | {selected_initial} | "
            "{selected_final} | {physical_active} | {dominant} | {raw_cond} | {eq_cond} | {raw_smin} | {eq_smin} | "
            "{slope_delta} | {g_u_in} | {g_T_in} | {g_u_out} | {g_T_out} | {interval_R} | {interval_E} | "
            "{outer_1} | {outer_2} | {D} | {C1} | {C2} | {K} | {compat_max} | {Rson_rg} | {lambda0} | "
            "{max_HR} | {outer_HR} | {outer_Qadv_Qvisc} | {int_adv} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                outer_closure=row["outer_closure"],
                interval_form=row["interval_form"],
                integrated_weighting=row["integrated_weighting"],
                pivot=row["pivot"],
                selected_initial=fmt(float(row["selected_initial"])),
                selected_final=fmt(float(row["selected_final"])),
                physical_active=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                raw_cond=fmt(float(row["raw_cond"])),
                eq_cond=fmt(float(row["eq_cond"])),
                raw_smin=fmt(float(row["raw_smin"])),
                eq_smin=fmt(float(row["eq_smin"])),
                slope_delta=fmt(float(row["slope_delta"])),
                g_u_in=fmt(float(row["g_u_in"])),
                g_T_in=fmt(float(row["g_T_in"])),
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
                outer_Qadv_Qvisc=fmt(float(row["outer_Qadv_Qvisc"])),
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
    z0, ratio = load_source()
    slope_params = params_for(
        fiducial,
        mdot_edd,
        ratio,
        outer_closure="thin_value",
        interval_form="differential",
        integrated_weighting="none",
        slopes=(0.0, 0.0),
    )
    slopes = polyfit_outer_log_slopes(z0, slope_params)

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for label, outer_closure, interval_form, integrated_weighting in SPECS:
        for pivot in PIVOTS:
            params = params_for(
                fiducial,
                mdot_edd,
                ratio,
                outer_closure=outer_closure,
                interval_form=interval_form,
                integrated_weighting=integrated_weighting,
                slopes=slopes,
            )
            conditioning = condition_metrics(z0, params, pivot)
            result = solve_variant(z0, params, pivot)
            row = row_from_result(label, ratio, params, pivot, z0, result, conditioning)
            rows.append(row)
            save_checkpoint(row)
            write_table(rows)
            print(
                f"{label} pivot={pivot} selected {row['selected_initial']:.3e}->{row['selected_final']:.3e} "
                f"physical={row['physical_active']:.3e} dom={row['dominant']} nfev={row['nfev']}",
                flush=True,
            )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
