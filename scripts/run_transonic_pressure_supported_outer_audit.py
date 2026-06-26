"""Audit pressure-supported outer rotation closure for fixed-Mdot roots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _outer_thin_boundary_residual,
    pressure_supported_omega_target,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    square_jac_sparsity_pattern,
    square_collocation_residual,
    state_bounds,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state, differential_residual, local_gradient, state_partials
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_n64_arclength_adaptive" / "030_arc_x1_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_pressure_supported_outer_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_pressure_supported_outer_audit"

N_NODES = 64
R_OUT_RG = 3000.0
RESIDUAL_TOL = 1.0e-6
MAX_NFEV = 80
PIVOTS = ("C1", "C2")
VARIANTS = (
    ("pressure_supported_correct_sign", -1.0),
    ("wrong_sign_cancellation_check", 1.0),
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
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def params_for(fiducial: FiducialParams, mdot_edd: float, ratio: float, outer_closure: str, slopes=None) -> TransonicSlimParams:
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
    )


def load_source() -> tuple[np.ndarray, float]:
    with np.load(SOURCE_CHECKPOINT, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), float(data["ratio"])


def checkpoint_outer_log_slopes(z: np.ndarray, params: TransonicSlimParams, n_fit: int = 8) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    count = min(int(n_fit), len(logR))
    x = logR[-count:] - logR[-1]
    gu = float(np.polyder(np.poly1d(np.polyfit(x, logu[-count:], min(2, count - 1))))(0.0))
    gT = float(np.polyder(np.poly1d(np.polyfit(x, logT[-count:], min(2, count - 1))))(0.0))
    return gu, gT


def one_sided_outer_log_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    dx = logR[-1] - logR[-2]
    return (
        float((logu[-1] - logu[-2]) / dx),
        float((logT[-1] - logT[-2]) / dx),
    )


def local_ode_outer_log_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    gradient = local_gradient(logR[-1], np.array([logu[-1], logT[-1]], dtype=float), lambda0, params)
    return float(gradient[0]), float(gradient[1])


def pressure_diagnostic(z: np.ndarray, params: TransonicSlimParams, slopes: tuple[float, float], slope_source: str) -> dict[str, object]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    y = np.array([logu[-1], logT[-1]], dtype=float)
    state = algebraic_state(logR[-1], y[0], y[1], lambda0, params)
    partials = state_partials(logR[-1], y, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)
    dPi_dx = partials.x["Pi"] + float(np.dot(partials.y["Pi"], np.asarray(slopes, dtype=float)))
    g_Pi = float(dPi_dx / (state.Pi + 1.0e-300))
    thin = _outer_thin_boundary_residual(logR[-1], y, lambda0, params)
    target = pressure_supported_omega_target(logR[-1], y, slopes, lambda0, params)
    radial = differential_residual(logR[-1], y, np.asarray(slopes, dtype=float), lambda0, params)[0]
    radial_scale = state.R**2 * state.Omega_K**2 + 1.0e-300
    return {
        "g_u": float(slopes[0]),
        "g_T": float(slopes[1]),
        "g_Pi": g_Pi,
        "outer_HR": float(state.H_over_R),
        "outer_HR_sq": float(state.H_over_R**2),
        "actual_lnOmega": float(thin[0]),
        "predicted_delta": float(target),
        "legacy_pressure_residual": float(thin[0] - target),
        "wrong_sign_residual": float(thin[0] + target),
        "endpoint_radial_scaled": float(radial / radial_scale),
        "thin_energy": float(thin[1]),
        "lambda0": float(lambda0),
        "slope_source": slope_source,
    }


def dominant_block(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_Omega": abs(audit.outer_omega),
        "outer_E": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
        "K": abs(audit.sonic_K),
    }
    return max(values, key=values.get)


def pressure_signed_square_residual(z: np.ndarray, params: TransonicSlimParams, pivot: str, slopes: tuple[float, float], sign: float) -> np.ndarray:
    residual = square_collocation_residual(z, params, pivot=pivot)
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    y = np.array([logu[-1], logT[-1]], dtype=float)
    target = pressure_supported_omega_target(logR[-1], y, slopes, lambda0, params)
    outer_row = 2 * (params.n_nodes - 1)
    residual[outer_row] = residual[outer_row] + sign * target
    return residual


def solve_pressure_supported(z0: np.ndarray, params: TransonicSlimParams, pivot: str, slopes: tuple[float, float], sign: float):
    lower, upper = state_bounds(params)
    z_start = np.clip(z0, lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda z: pressure_signed_square_residual(z, params, pivot, slopes, sign),
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


def row_from_result(label: str, pivot: str, ratio: float, params: TransonicSlimParams, slopes: tuple[float, float], sign: float, result) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    residual = pressure_signed_square_residual(z, params, pivot, slopes, sign)
    legacy = square_collocation_residual(z, params, pivot=pivot)
    legacy_audit = residual_audit_from_state_vector(z, params)
    profile = profile_from_state_vector(z, params)
    outer_row = 2 * (params.n_nodes - 1)
    custom_values = {
        "interval_R": abs(legacy_audit.interval_radial_max),
        "interval_E": abs(legacy_audit.interval_energy_max),
        "outer_Omega": abs(float(residual[outer_row])),
        "outer_E": abs(legacy_audit.outer_energy),
        "D": abs(legacy_audit.sonic_D),
        "C1": abs(legacy_audit.sonic_C1),
        "C2": abs(legacy_audit.sonic_C2),
        "K": abs(legacy_audit.sonic_K),
    }
    return {
        "label": label,
        "ratio": ratio,
        "pivot": pivot,
        "selected_max": float(np.max(np.abs(residual))),
        "legacy_square": float(np.max(np.abs(legacy))),
        "dominant": max(custom_values, key=custom_values.get),
        "legacy_dominant": dominant_block(legacy_audit),
        "interval_R": legacy_audit.interval_radial_max,
        "interval_E": legacy_audit.interval_energy_max,
        "outer_pressure_Omega": float(residual[outer_row]),
        "outer_legacy_Omega": legacy_audit.outer_omega,
        "outer_E": legacy_audit.outer_energy,
        "D": legacy_audit.sonic_D,
        "C1": legacy_audit.sonic_C1,
        "C2": legacy_audit.sonic_C2,
        "K": legacy_audit.sonic_K,
        "compat_max": max(abs(legacy_audit.sonic_C1), abs(legacy_audit.sonic_C2), abs(legacy_audit.sonic_K)),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": legacy_audit.outer_H_over_R,
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
        "# Pressure-Supported Outer Closure Audit",
        "",
        "Generated by `scripts/run_transonic_pressure_supported_outer_audit.py`.",
        "",
        "## Pressure-Support Diagnostic",
        "",
        "| slope source | g_u | g_T | g_Pi | outer H/R | (H/R)^2 | actual ln(Omega/OmegaK) | predicted delta | actual-minus-target | wrong-sign residual | endpoint radial residual | thin energy | lambda0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for diagnostic in diagnostics:
        lines.append(
            "| {slope_source} | {g_u} | {g_T} | {g_Pi} | {outer_HR} | {outer_HR_sq} | {actual_lnOmega} | "
            "{predicted_delta} | {legacy_pressure_residual} | {wrong_sign_residual} | {endpoint_radial_scaled} | "
            "{thin_energy} | {lambda0} |".format(
                slope_source=diagnostic["slope_source"],
                g_u=fmt(float(diagnostic["g_u"])),
                g_T=fmt(float(diagnostic["g_T"])),
                g_Pi=fmt(float(diagnostic["g_Pi"])),
                outer_HR=fmt(float(diagnostic["outer_HR"])),
                outer_HR_sq=fmt(float(diagnostic["outer_HR_sq"])),
                actual_lnOmega=fmt(float(diagnostic["actual_lnOmega"])),
                predicted_delta=fmt(float(diagnostic["predicted_delta"])),
                legacy_pressure_residual=fmt(float(diagnostic["legacy_pressure_residual"])),
                wrong_sign_residual=fmt(float(diagnostic["wrong_sign_residual"])),
                endpoint_radial_scaled=fmt(float(diagnostic["endpoint_radial_scaled"])),
                thin_energy=fmt(float(diagnostic["thin_energy"])),
                lambda0=fmt(float(diagnostic["lambda0"])),
            )
        )
    lines.extend(
        [
            "",
            "The `local_ode_gradient` row uses the gradient required by the implemented local radial/energy ODE at the endpoint. Its target reproduces the actual angular-velocity offset, fixing the sign convention. Smooth outer-annulus slopes instead predict a sub-Keplerian correction and expose the endpoint mismatch.",
            "",
            "## Fixed-Mdot Solves",
            "",
            "| label | pivot | selected max | legacy square | dominant | legacy dominant | interval R | interval E | pressure Omega | legacy Omega | outer E | D | C1 | C2 | K | compat max | max H/R | outer H/R | int adv | nfev | success | message |",
            "|---|:---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | {pivot} | {selected_max} | {legacy_square} | {dominant} | {legacy_dominant} | "
            "{interval_R} | {interval_E} | {outer_pressure_Omega} | {outer_legacy_Omega} | {outer_E} | "
            "{D} | {C1} | {C2} | {K} | {compat_max} | {max_HR} | {outer_HR} | {int_adv} | {nfev} | "
            "{success} | {message} |".format(
                label=row["label"],
                pivot=row["pivot"],
                selected_max=fmt(float(row["selected_max"])),
                legacy_square=fmt(float(row["legacy_square"])),
                dominant=row["dominant"],
                legacy_dominant=row["legacy_dominant"],
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                outer_pressure_Omega=fmt(float(row["outer_pressure_Omega"])),
                outer_legacy_Omega=fmt(float(row["outer_legacy_Omega"])),
                outer_E=fmt(float(row["outer_E"])),
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
    legacy_params = params_for(fiducial, mdot_edd, ratio, "thin_value")
    slopes = checkpoint_outer_log_slopes(z0, legacy_params)
    diagnostics = [
        pressure_diagnostic(z0, legacy_params, one_sided_outer_log_slopes(z0, legacy_params), "native_checkpoint_one_sided"),
        pressure_diagnostic(z0, legacy_params, slopes, "native_checkpoint_outer_polyfit"),
        pressure_diagnostic(z0, legacy_params, local_ode_outer_log_slopes(z0, legacy_params), "local_ode_gradient"),
    ]

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for label, sign in VARIANTS:
        for pivot in PIVOTS:
            result = solve_pressure_supported(z0, legacy_params, pivot, slopes, sign)
            row = row_from_result(label, pivot, ratio, legacy_params, slopes, sign, result)
            rows.append(row)
            save_checkpoint(row)
            write_table(diagnostics, rows)
            print(
                f"{label} pivot={pivot} selected={row['selected_max']:.3e} "
                f"legacy={row['legacy_square']:.3e} dom={row['dominant']}",
                flush=True,
            )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
