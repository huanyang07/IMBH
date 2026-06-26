"""Compare differential and integrated interval defects for fixed-Mdot solves."""

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
from imri_qpe.layer3_minidisk_1d.transonic_local import local_gradient
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_n64_arclength_adaptive" / "030_arc_x1_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_integrated_defect_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_integrated_defect_audit"

N_NODES = 64
R_OUT_RG = 3000.0
RESIDUAL_TOL = 1.0e-6
MAX_NFEV = 60
PIVOT = "C1"


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


def local_ode_outer_log_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    gradient = local_gradient(logR[-1], np.array([logu[-1], logT[-1]], dtype=float), lambda0, params)
    return float(gradient[0]), float(gradient[1])


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    ratio: float,
    *,
    outer_closure: str,
    interval_form: str,
    integrated_weighting: str = "none",
    slopes=None,
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


def active_physical_max(audit) -> float:
    return max(
        abs(audit.interval_radial_max),
        abs(audit.interval_energy_max),
        abs(audit.outer_omega),
        abs(audit.outer_energy),
        abs(audit.sonic_D),
        abs(audit.sonic_K),
    )


def dominant_block(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_1": abs(audit.outer_omega),
        "outer_2": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "K": abs(audit.sonic_K),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
    }
    return max(values, key=values.get)


def solve_variant(z0: np.ndarray, params: TransonicSlimParams, pivot: str):
    lower, upper = state_bounds(params)
    z_start = np.clip(z0, lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda z: square_collocation_residual(z, params, pivot=pivot),
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


def row_from_result(label: str, ratio: float, params: TransonicSlimParams, z0: np.ndarray, result, conditioning: dict[str, float]) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    initial = square_collocation_residual(z0, params, pivot=PIVOT)
    final = square_collocation_residual(z, params, pivot=PIVOT)
    audit = residual_audit_from_state_vector(z, params)
    legacy_params = replace(params, outer_closure="thin_value", outer_match_log_slopes=None, interval_residual_form="differential")
    legacy_audit = residual_audit_from_state_vector(z, legacy_params)
    profile = profile_from_state_vector(z, params)
    return {
        "label": label,
        "ratio": ratio,
        "outer_closure": params.outer_closure,
        "interval_form": params.interval_residual_form,
        "integrated_weighting": params.integrated_residual_weighting,
        "selected_initial": float(np.max(np.abs(initial))),
        "selected_final": float(np.max(np.abs(final))),
        "physical_active": active_physical_max(audit),
        "dominant": dominant_block(audit),
        "legacy_outer_Omega": legacy_audit.outer_omega,
        "legacy_outer_E": legacy_audit.outer_energy,
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_1": audit.outer_omega,
        "outer_2": audit.outer_energy,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "K": audit.sonic_K,
        "compat_max": max(abs(audit.sonic_C1), abs(audit.sonic_C2), abs(audit.sonic_K)),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": audit.outer_H_over_R,
        "int_adv": profile.integrated_advective_fraction,
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        **conditioning,
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{row['label']}_{float(row['ratio']):.8f}".replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Integrated Defect Audit",
        "",
        "Generated by `scripts/run_transonic_integrated_defect_audit.py`.",
        "",
        "Rows compare differential and integrated midpoint interval residuals at the native `Mdot/Edd ~= 0.9028`, `N=64` checkpoint. Physical interval residual columns are always reconstructed in differential form.",
        "",
        "| label | outer closure | interval form | weighting | selected initial | selected final | physical active | dominant | raw cond | eq cond | raw smin | eq smin | interval R | interval E | outer 1 | outer 2 | legacy Omega | legacy E | D | C1 | C2 | K | compat max | max H/R | outer H/R | int adv | nfev | success | message |",
        "|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {outer_closure} | {interval_form} | {integrated_weighting} | {selected_initial} | {selected_final} | {physical_active} | "
            "{dominant} | {raw_cond} | {eq_cond} | {raw_smin} | {eq_smin} | {interval_R} | {interval_E} | "
            "{outer_1} | {outer_2} | {legacy_outer_Omega} | {legacy_outer_E} | {D} | {C1} | {C2} | {K} | "
            "{compat_max} | {max_HR} | {outer_HR} | {int_adv} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                outer_closure=row["outer_closure"],
                interval_form=row["interval_form"],
                integrated_weighting=row["integrated_weighting"],
                selected_initial=fmt(float(row["selected_initial"])),
                selected_final=fmt(float(row["selected_final"])),
                physical_active=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                raw_cond=fmt(float(row["raw_cond"])),
                eq_cond=fmt(float(row["eq_cond"])),
                raw_smin=fmt(float(row["raw_smin"])),
                eq_smin=fmt(float(row["eq_smin"])),
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
    base_params = params_for(fiducial, mdot_edd, ratio, outer_closure="thin_value", interval_form="differential")
    slopes = local_ode_outer_log_slopes(z0, base_params)
    specs = [
        ("thin_differential", "thin_value", "differential", "none", None),
        ("thin_integrated_raw", "thin_value", "integrated", "none", None),
        ("thin_integrated_l2_weighted", "thin_value", "integrated", "inverse_sqrt_dx", None),
        ("thin_integrated_dx_weighted", "thin_value", "integrated", "inverse_dx", None),
        ("matched_local_differential", "matched_outer_state", "differential", "none", slopes),
        ("matched_local_integrated_raw", "matched_outer_state", "integrated", "none", slopes),
        ("matched_local_integrated_l2_weighted", "matched_outer_state", "integrated", "inverse_sqrt_dx", slopes),
        ("matched_local_integrated_dx_weighted", "matched_outer_state", "integrated", "inverse_dx", slopes),
    ]

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for label, outer_closure, interval_form, integrated_weighting, variant_slopes in specs:
        params = params_for(
            fiducial,
            mdot_edd,
            ratio,
            outer_closure=outer_closure,
            interval_form=interval_form,
            integrated_weighting=integrated_weighting,
            slopes=variant_slopes,
        )
        conditioning = condition_metrics(z0, params, PIVOT)
        result = solve_variant(z0, params, PIVOT)
        row = row_from_result(label, ratio, params, z0, result, conditioning)
        rows.append(row)
        save_checkpoint(row)
        write_table(rows)
        print(
            f"{label} selected {row['selected_initial']:.3e}->{row['selected_final']:.3e} "
            f"physical={row['physical_active']:.3e} eq_cond={row['eq_cond']:.3e}",
            flush=True,
        )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
