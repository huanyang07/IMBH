"""Audit local matched-state outer closure for fixed-Mdot roots."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    matched_outer_state,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    square_collocation_residual,
    square_jac_sparsity_pattern,
    state_bounds,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import differential_residual, local_gradient
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_n64_arclength_adaptive" / "030_arc_x1_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_matched_outer_state_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_matched_outer_state_audit"

N_NODES = 64
R_OUT_RG = 3000.0
RESIDUAL_TOL = 1.0e-6
MAX_NFEV = 60
PIVOTS = ("C1", "C2")


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


def params_for(fiducial: FiducialParams, mdot_edd: float, ratio: float, slopes=None) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=N_NODES,
        R_out_rg=R_OUT_RG,
        residual_tol=RESIDUAL_TOL,
        max_nfev=MAX_NFEV,
        outer_closure="matched_outer_state",
        outer_match_log_slopes=slopes,
    )


def load_source() -> tuple[np.ndarray, float]:
    with np.load(SOURCE_CHECKPOINT, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), float(data["ratio"])


def one_sided_outer_log_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    dx = logR[-1] - logR[-2]
    return (
        float((logu[-1] - logu[-2]) / dx),
        float((logT[-1] - logT[-2]) / dx),
    )


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


def slope_specs(z: np.ndarray, params: TransonicSlimParams) -> list[tuple[str, tuple[float, float]]]:
    return [
        ("native_one_sided", one_sided_outer_log_slopes(z, params)),
        ("native_outer_polyfit", polyfit_outer_log_slopes(z, params)),
        ("local_ode_gradient", local_ode_outer_log_slopes(z, params)),
    ]


def local_match_diagnostic(z: np.ndarray, params: TransonicSlimParams, slopes: tuple[float, float]) -> dict[str, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    y = np.array([logu[-1], logT[-1]], dtype=float)
    y_match = matched_outer_state(logR[-1], lambda0, params, g_match=slopes, initial_y=y)
    local_residual = differential_residual(logR[-1], y_match, np.asarray(slopes, dtype=float), lambda0, params)
    return {
        "target_logu": float(y_match[0]),
        "target_logT": float(y_match[1]),
        "initial_dlogu": float(y[0] - y_match[0]),
        "initial_dlogT": float(y[1] - y_match[1]),
        "local_radial_raw": float(local_residual[0]),
        "local_energy_raw": float(local_residual[1]),
    }


def dominant_block(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_u": abs(audit.outer_omega),
        "outer_T": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
        "K": abs(audit.sonic_K),
    }
    return max(values, key=values.get)


def solve_matched(z0: np.ndarray, params: TransonicSlimParams, pivot: str):
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


def row_from_result(label: str, pivot: str, ratio: float, params: TransonicSlimParams, result) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    residual = square_collocation_residual(z, params, pivot=pivot)
    legacy_params = replace(params, outer_closure="thin_value", outer_match_log_slopes=None)
    legacy = square_collocation_residual(z, legacy_params, pivot=pivot)
    audit = residual_audit_from_state_vector(z, params)
    legacy_audit = residual_audit_from_state_vector(z, legacy_params)
    profile = profile_from_state_vector(z, params)
    return {
        "label": label,
        "ratio": ratio,
        "pivot": pivot,
        "selected_initial": float(np.max(np.abs(square_collocation_residual(load_source()[0], params, pivot=pivot)))),
        "selected_final": float(np.max(np.abs(residual))),
        "legacy_square": float(np.max(np.abs(legacy))),
        "dominant": dominant_block(audit),
        "legacy_dominant": dominant_block(legacy_audit),
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_u": audit.outer_omega,
        "outer_T": audit.outer_energy,
        "legacy_outer_Omega": legacy_audit.outer_omega,
        "legacy_outer_E": legacy_audit.outer_energy,
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


def write_table(diagnostics: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Matched Outer-State Closure Audit",
        "",
        "Generated by `scripts/run_transonic_matched_outer_state_audit.py`.",
        "",
        "The matched closure solves local full radial/energy equations at `R_out` for a target `(logu, logT)` using a supplied outer slope pair, then imposes value matching in the global BVP.",
        "",
        "## Initial Local Match Diagnostics",
        "",
        "| slope source | g_u | g_T | target logu | target logT | initial dlogu | initial dlogT | local radial raw | local energy raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics:
        lines.append(
            "| {label} | {g_u} | {g_T} | {target_logu} | {target_logT} | {initial_dlogu} | {initial_dlogT} | {local_radial_raw} | {local_energy_raw} |".format(
                label=row["label"],
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                target_logu=fmt(float(row["target_logu"])),
                target_logT=fmt(float(row["target_logT"])),
                initial_dlogu=fmt(float(row["initial_dlogu"])),
                initial_dlogT=fmt(float(row["initial_dlogT"])),
                local_radial_raw=fmt(float(row["local_radial_raw"])),
                local_energy_raw=fmt(float(row["local_energy_raw"])),
            )
        )
    lines.extend(
        [
            "",
            "## Fixed-Mdot Solves",
            "",
            "| label | pivot | selected initial | selected final | legacy square | dominant | legacy dominant | interval R | interval E | outer u | outer T | legacy Omega | legacy E | D | C1 | C2 | K | compat max | max H/R | outer H/R | int adv | nfev | success | message |",
            "|---|:---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | {pivot} | {selected_initial} | {selected_final} | {legacy_square} | {dominant} | {legacy_dominant} | "
            "{interval_R} | {interval_E} | {outer_u} | {outer_T} | {legacy_outer_Omega} | {legacy_outer_E} | "
            "{D} | {C1} | {C2} | {K} | {compat_max} | {max_HR} | {outer_HR} | {int_adv} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                pivot=row["pivot"],
                selected_initial=fmt(float(row["selected_initial"])),
                selected_final=fmt(float(row["selected_final"])),
                legacy_square=fmt(float(row["legacy_square"])),
                dominant=row["dominant"],
                legacy_dominant=row["legacy_dominant"],
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                outer_u=fmt(float(row["outer_u"])),
                outer_T=fmt(float(row["outer_T"])),
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
    base_params = params_for(fiducial, mdot_edd, ratio, slopes=(-1.0, -1.0))

    diagnostics: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for label, slopes in slope_specs(z0, base_params):
        params = params_for(fiducial, mdot_edd, ratio, slopes=slopes)
        diag = local_match_diagnostic(z0, params, slopes)
        diagnostics.append({"label": label, "g_u": slopes[0], "g_T": slopes[1], **diag})
        write_table(diagnostics, rows)
        for pivot in PIVOTS:
            result = solve_matched(z0, params, pivot)
            row = row_from_result(label, pivot, ratio, params, result)
            rows.append(row)
            save_checkpoint(row)
            write_table(diagnostics, rows)
            print(
                f"{label} pivot={pivot} selected {row['selected_initial']:.3e}->{row['selected_final']:.3e} "
                f"legacy={row['legacy_square']:.3e} dom={row['dominant']}",
                flush=True,
            )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
