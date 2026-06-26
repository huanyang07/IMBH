"""Audit alternative outer closures for the high-rate fixed-Mdot floor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _heating_terms_from_gradient,
    _interval_residual_from_unpacked,
    _outer_boundary_residual,
    _residual_scales,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    sonic_residual_pair,
    square_collocation_residual,
    state_bounds,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state, state_partials
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_n64_arclength_adaptive" / "030_arc_x1_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_outer_closure_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_outer_closure_audit"

N_NODES = 64
R_OUT_RG = 3000.0
RESIDUAL_TOL = 1.0e-6
MAX_NFEV = 160
PIVOTS = ("C1", "C2")
CLOSURES = (
    "thin_value",
    "value_full_energy",
    "value_zero_adv",
    "shear_thin_energy",
    "shear_full_energy",
)


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def params_for(fiducial: FiducialParams, mdot_edd: float, ratio: float) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=N_NODES,
        R_out_rg=R_OUT_RG,
        residual_tol=RESIDUAL_TOL,
        max_nfev=MAX_NFEV,
    )


def load_source() -> tuple[np.ndarray, float]:
    with np.load(SOURCE_CHECKPOINT, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), float(data["ratio"])


def outer_gradient(logu: np.ndarray, logT: np.ndarray, logR: np.ndarray) -> np.ndarray:
    dx = logR[-1] - logR[-2]
    return np.array([(logu[-1] - logu[-2]) / dx, (logT[-1] - logT[-2]) / dx], dtype=float)


def outer_closure_residual(logu: np.ndarray, logT: np.ndarray, logR: np.ndarray, lambda0: float, params: TransonicSlimParams, closure: str) -> np.ndarray:
    y = np.array([logu[-1], logT[-1]], dtype=float)
    thin = _outer_boundary_residual(logR[-1], y, lambda0, params)
    state = algebraic_state(logR[-1], y[0], y[1], lambda0, params)
    g = outer_gradient(logu, logT, logR)
    partials = state_partials(logR[-1], y, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)
    dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], g))
    dlnOmega_dx = dOmega_dx / (state.Omega + 1.0e-300)
    dlnOmegaK_dx = float(params.potential.dln_omega_k_dlnR(state.R))
    omega_shear = (dlnOmega_dx - dlnOmegaK_dx) / np.sqrt(dlnOmegaK_dx**2 + 1.0)
    _q_visc, _q_rad, q_adv, energy = _heating_terms_from_gradient(logR[-1], y, g, lambda0, params)
    _radial_scale, energy_scale = _residual_scales(logR[-1], y, params, lambda0)
    full_energy = energy / energy_scale
    zero_adv = q_adv / (abs(_q_visc) + abs(_q_rad) + abs(q_adv) + 1.0e-300)

    if closure == "thin_value":
        return thin
    if closure == "value_full_energy":
        return np.array([thin[0], full_energy], dtype=float)
    if closure == "value_zero_adv":
        return np.array([thin[0], zero_adv], dtype=float)
    if closure == "shear_thin_energy":
        return np.array([omega_shear, thin[1]], dtype=float)
    if closure == "shear_full_energy":
        return np.array([omega_shear, full_energy], dtype=float)
    raise ValueError(f"unknown closure {closure!r}")


def closure_square_residual(z: np.ndarray, params: TransonicSlimParams, pivot: str, closure: str) -> np.ndarray:
    residual = np.zeros(2 * params.n_nodes + 2, dtype=float)
    try:
        logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
        if np.any(np.diff(logR) <= 0.0):
            raise ValueError("mapped radius must increase")
        row = 0
        for idx in range(params.n_nodes - 1):
            residual[row : row + 2] = _interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            row += 2
        residual[row : row + 2] = outer_closure_residual(logu, logT, logR, lambda0, params, closure)
        row += 2
        residual[row : row + 2] = sonic_residual_pair(z, params, pivot=pivot)
    except Exception:
        residual.fill(1.0e6)
    return residual


def closure_jac_sparsity(params: TransonicSlimParams, closure: str):
    unknown_size = 2 * params.n_nodes + 2
    pattern = lil_matrix((2 * params.n_nodes + 2, unknown_size), dtype=int)
    row = 0
    for idx in range(params.n_nodes - 1):
        columns = (idx, idx + 1, params.n_nodes + idx, params.n_nodes + idx + 1, unknown_size - 2, unknown_size - 1)
        for col in columns:
            pattern[row : row + 2, col] = 1
        row += 2

    if closure in {"thin_value"}:
        outer_columns = (params.n_nodes - 1, 2 * params.n_nodes - 1, unknown_size - 2, unknown_size - 1)
    else:
        outer_columns = (
            params.n_nodes - 2,
            params.n_nodes - 1,
            2 * params.n_nodes - 2,
            2 * params.n_nodes - 1,
            unknown_size - 2,
            unknown_size - 1,
        )
    for col in outer_columns:
        pattern[row : row + 2, col] = 1
    row += 2
    for col in (0, params.n_nodes, unknown_size - 2, unknown_size - 1):
        pattern[row : row + 2, col] = 1
    return pattern.tocsr()


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


def solve_closure(z0: np.ndarray, params: TransonicSlimParams, pivot: str, closure: str):
    lower, upper = state_bounds(params)
    z_start = np.clip(z0, lower + 1.0e-12, upper - 1.0e-12)
    result = least_squares(
        lambda z: closure_square_residual(z, params, pivot, closure),
        z_start,
        jac_sparsity=closure_jac_sparsity(params, closure),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV,
    )
    return result


def row_from_result(closure: str, pivot: str, ratio: float, params: TransonicSlimParams, result) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    selected = closure_square_residual(z, params, pivot, closure)
    legacy = square_collocation_residual(z, params, pivot=pivot)
    audit = residual_audit_from_state_vector(z, params)
    profile = profile_from_state_vector(z, params)
    return {
        "closure": closure,
        "pivot": pivot,
        "ratio": ratio,
        "selected_max": float(np.max(np.abs(selected))),
        "legacy_square": float(np.max(np.abs(legacy))),
        "dominant_legacy": dominant_block(audit),
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_Omega": audit.outer_omega,
        "outer_E": audit.outer_energy,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "K": audit.sonic_K,
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": audit.outer_H_over_R,
        "outer_Qadv_over_Qvisc": audit.outer_Qadv_over_Qvisc,
        "int_adv": profile.integrated_advective_fraction,
        "Rson_rg": profile.sonic_radius / params.r_g,
        "lambda_ratio": audit.lambda0_over_lK_isco,
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "z": z,
    }


def save_checkpoint(row: dict[str, object], params: TransonicSlimParams) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{row['closure']}_{row['pivot']}_{float(row['ratio']):.8f}".replace(".", "p")
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(params.R_out_rg),
        closure=np.array(row["closure"]),
        pivot=np.array(row["pivot"]),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Transonic Outer Closure Audit",
        "",
        "Generated by `scripts/run_transonic_outer_closure_audit.py`.",
        "",
        "Rows solve the native `Mdot/Edd ~= 0.9028`, `N=64`, `R_out=3000 r_g` square system with alternative outer endpoint closures.",
        "",
        "| closure | pivot | selected max | legacy square | legacy dominant | interval R | interval E | outer Omega | outer E | D | C1 | C2 | K | max H/R | outer H/R | outer Qadv/Qvisc | int adv | Rson/rg | lambda/lK_ISCO | nfev | success | message |",
        "|---|:---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {closure} | {pivot} | {selected_max} | {legacy_square} | {dominant_legacy} | "
            "{interval_R} | {interval_E} | {outer_Omega} | {outer_E} | {D} | {C1} | {C2} | {K} | "
            "{max_HR} | {outer_HR} | {outer_Qadv_over_Qvisc} | {int_adv} | {Rson_rg} | {lambda_ratio} | "
            "{nfev} | {success} | {message} |".format(
                closure=row["closure"],
                pivot=row["pivot"],
                selected_max=fmt(float(row["selected_max"])),
                legacy_square=fmt(float(row["legacy_square"])),
                dominant_legacy=row["dominant_legacy"],
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                outer_Omega=fmt(float(row["outer_Omega"])),
                outer_E=fmt(float(row["outer_E"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                max_HR=fmt(float(row["max_HR"])),
                outer_HR=fmt(float(row["outer_HR"])),
                outer_Qadv_over_Qvisc=fmt(float(row["outer_Qadv_over_Qvisc"])),
                int_adv=fmt(float(row["int_adv"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                lambda_ratio=fmt(float(row["lambda_ratio"])),
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
    params = params_for(fiducial, mdot_edd, ratio)
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for closure in CLOSURES:
        for pivot in PIVOTS:
            result = solve_closure(z0, params, pivot, closure)
            row = row_from_result(closure, pivot, ratio, params, result)
            rows.append(row)
            save_checkpoint(row, params)
            write_table(rows)
            print(
                f"{closure} pivot={pivot} selected={row['selected_max']:.3e} "
                f"legacy={row['legacy_square']:.3e} dom={row['dominant_legacy']}",
                flush=True,
            )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
