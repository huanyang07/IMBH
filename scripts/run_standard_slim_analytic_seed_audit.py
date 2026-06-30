"""Analytic thin-seed audit for the standard no-wind slim benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    computational_grid,
    differential_residual_scales,
    pack_state,
    residual_audit_from_state_vector,
    sonic_diagnostics,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state, differential_residual, state_partials
from imri_qpe.layer3_minidisk_1d.transonic_thermo import integrated_stress, radiative_cooling, vertical_state
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ANALYTIC_SEED_TABLE",
    "outputs/tables/slim_benchmark_analytic_seed_residual_audit.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FORMULATION_TABLE = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_FORMULATION_TABLE",
    "outputs/tables/slim_benchmark_formulation_checks.md",
)
FORMULATION_JSON = FORMULATION_TABLE.with_suffix(".json")
PROFILE_TABLE = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_RESIDUAL_PROFILE_TABLE",
    "outputs/tables/slim_benchmark_residual_profile_mdot1e-3.md",
)
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ANALYTIC_SEED_FIGURE",
    "outputs/figures/slim_benchmark_analytic_seed_residuals.png",
)

MDOT_RATIO = float(os.environ.get("IMBH_STANDARD_SLIM_ANALYTIC_RATIO", "1e-3"))
ALPHA = float(os.environ.get("IMBH_STANDARD_SLIM_ANALYTIC_ALPHA", "0.01"))
STRESS_FACTORS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_ANALYTIC_STRESS_FACTORS", "1.0,1.5").replace(":", ",").split(",")
    if piece.strip()
)
R_OUT_CASES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_ANALYTIC_ROUTS", "300,1000,3000,10000").replace(":", ",").split(",")
    if piece.strip()
)
R_SON_RG = float(os.environ.get("IMBH_STANDARD_SLIM_ANALYTIC_RSON_RG", "5.9"))
NODES_BY_ROUT = {
    300.0: 64,
    1000.0: 80,
    3000.0: 96,
    10000.0: 128,
}
DEFAULT_N_PER_LOG_SPAN = float(os.environ.get("IMBH_STANDARD_SLIM_ANALYTIC_N_PER_LOG_SPAN", "16"))
PRIMARY_STRESS_FACTOR = float(os.environ.get("IMBH_STANDARD_SLIM_ANALYTIC_PRIMARY_STRESS_FACTOR", "1.0"))


def fmt(value: float) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.4g}"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    return value


def n_nodes_for(R_out_rg: float) -> int:
    if float(R_out_rg) in NODES_BY_ROUT:
        return NODES_BY_ROUT[float(R_out_rg)]
    span = np.log(float(R_out_rg) / R_SON_RG)
    return int(max(64, np.ceil(DEFAULT_N_PER_LOG_SPAN * span)))


def make_params(fiducial: FiducialParams, mdot_edd: float, R_out_rg: float, stress_factor: float) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=MDOT_RATIO * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=float(stress_factor),
        R_out_rg=float(R_out_rg),
        n_nodes=n_nodes_for(float(R_out_rg)),
        residual_tol=1.0e-6,
        max_nfev=1,
        outer_closure="thin_value",
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def local_thin_targets(logR: float, params: TransonicSlimParams, lambda0: float) -> dict[str, float]:
    R = float(np.exp(logR))
    potential = params.potential
    l0 = float(lambda0 * params.r_g * C)
    l_k = float(potential.l_k(R))
    W_req = params.Mdot_g_s * (l_k - l0) / (2.0 * np.pi * R**2)
    shear = float(potential.dln_omega_k_dlnR(R))
    Q_visc_thin = -W_req * float(potential.omega_k(R)) * shear
    return {
        "R": R,
        "l0": l0,
        "l_k": l_k,
        "W_req": float(W_req),
        "Q_visc_thin": float(Q_visc_thin),
        "Omega_K": float(potential.omega_k(R)),
        "shear": shear,
    }


def thin_local_residual(log_sigma_T: np.ndarray, logR: float, params: TransonicSlimParams, lambda0: float) -> np.ndarray:
    logSigma, logT = np.asarray(log_sigma_T, dtype=float)
    Sigma = float(np.exp(logSigma))
    T = float(np.exp(logT))
    target = local_thin_targets(logR, params, lambda0)
    if target["W_req"] <= 0.0 or target["Q_visc_thin"] <= 0.0:
        return np.array([1.0e6, 1.0e6], dtype=float)
    vertical = vertical_state(Sigma, T, target["R"], params.potential, params.mu_mol, params.kappa, params.gamma_gas)
    W_model = float(integrated_stress(vertical, params.alpha, params.mu_stress, params.stress_factor))
    Q_rad = float(radiative_cooling(vertical, params.kappa))
    return np.array(
        [
            np.log(W_model / target["W_req"]),
            np.log(Q_rad / target["Q_visc_thin"]),
        ],
        dtype=float,
    )


def solve_local_thin_state(
    logR: float,
    params: TransonicSlimParams,
    lambda0: float,
    previous: np.ndarray | None,
) -> tuple[float, float, float]:
    """Return ``Sigma,T,local_residual`` for one analytic thin-disk node."""

    lower = np.array([np.log(1.0e-8), np.log(1.0e3)], dtype=float)
    upper = np.array([np.log(1.0e14), np.log(1.0e10)], dtype=float)
    seeds: list[np.ndarray] = []
    if previous is not None and np.all(np.isfinite(previous)):
        seeds.append(np.asarray(previous, dtype=float))
    for Sigma0 in (1.0e1, 1.0e3, 1.0e5, 1.0e7, 1.0e9):
        for T0 in (1.0e4, 1.0e5, 1.0e6, 1.0e7):
            seeds.append(np.log(np.array([Sigma0, T0], dtype=float)))
    best_x = None
    best_norm = np.inf
    for seed in seeds:
        x0 = np.clip(seed, lower + 1.0e-12, upper - 1.0e-12)
        try:
            result = least_squares(
                lambda trial: thin_local_residual(trial, logR, params, lambda0),
                x0,
                bounds=(lower, upper),
                x_scale="jac",
                ftol=1.0e-12,
                xtol=1.0e-12,
                gtol=1.0e-12,
                max_nfev=120,
            )
        except Exception:
            continue
        norm = float(np.max(np.abs(thin_local_residual(result.x, logR, params, lambda0))))
        if norm < best_norm:
            best_norm = norm
            best_x = np.asarray(result.x, dtype=float)
        if norm < 1.0e-10:
            break
    if best_x is None:
        raise RuntimeError(f"local analytic seed solve failed at R={np.exp(logR) / params.r_g:g} rg")
    return float(np.exp(best_x[0])), float(np.exp(best_x[1])), best_norm


def analytic_seed_state(params: TransonicSlimParams, logR_son: float, lambda0: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    logR = computational_grid(params, logR_son)
    Sigma = np.empty_like(logR)
    T = np.empty_like(logR)
    local_norm = np.empty_like(logR)
    previous = None
    for idx, x in enumerate(logR):
        sigma, temp, norm = solve_local_thin_state(float(x), params, lambda0, previous)
        Sigma[idx] = sigma
        T[idx] = temp
        local_norm[idx] = norm
        previous = np.log(np.array([sigma, temp], dtype=float))
    R = np.exp(logR)
    u = params.Mdot_g_s / (2.0 * np.pi * R * Sigma)
    z = pack_state(np.log(u), np.log(T), logR_son, lambda0)
    return z, {"logR": logR, "R": R, "Sigma": Sigma, "T": T, "u": u, "local_norm": local_norm}


def interval_residuals(z: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    logu, logT, _logR_son, lambda0, logR = params_unpack(z, params)
    return np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            for idx in range(len(logR) - 1)
        ],
        dtype=float,
    )


def params_unpack(z: np.ndarray, params: TransonicSlimParams):
    from imri_qpe.layer3_minidisk_1d import unpack_state

    return unpack_state(z, params)


def pressure_support_diagnostics(z: np.ndarray, params: TransonicSlimParams) -> dict[str, np.ndarray]:
    logu, logT, _logR_son, lambda0, logR = params_unpack(z, params)
    measured = []
    pressure = []
    inertia = []
    qadv_qvisc = []
    qbalance = []
    angular_residual = []
    stress_residual = []
    omega = []
    hr = []
    smin = []
    condA = []
    for idx, x in enumerate(logR):
        y = np.array([logu[idx], logT[idx]], dtype=float)
        if idx == 0:
            g = np.array(
                [
                    (logu[1] - logu[0]) / (logR[1] - logR[0]),
                    (logT[1] - logT[0]) / (logR[1] - logR[0]),
                ],
                dtype=float,
            )
        elif idx == len(logR) - 1:
            g = np.array(
                [
                    (logu[-1] - logu[-2]) / (logR[-1] - logR[-2]),
                    (logT[-1] - logT[-2]) / (logR[-1] - logR[-2]),
                ],
                dtype=float,
            )
        else:
            g = np.array(
                [
                    (logu[idx + 1] - logu[idx - 1]) / (logR[idx + 1] - logR[idx - 1]),
                    (logT[idx + 1] - logT[idx - 1]) / (logR[idx + 1] - logR[idx - 1]),
                ],
                dtype=float,
            )
        state = algebraic_state(float(x), y[0], y[1], lambda0, params)
        partials = state_partials(float(x), y, lambda0, params)
        dPi_dx = partials.x["Pi"] + float(np.dot(partials.y["Pi"], g))
        dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], g))
        Tdsdx_residual = differential_residual(float(x), y, g, lambda0, params)
        _radial_scale, energy_scale = differential_residual_scales(float(x), y, lambda0, params)
        Q_visc = -state.W * dOmega_dx
        Q_adv = Q_visc - state.Q_rad - Tdsdx_residual[1]
        measured.append(state.Omega**2 / (state.Omega_K**2 + 1.0e-300) - 1.0)
        pressure.append((dPi_dx / state.Sigma) / (state.R**2 * state.Omega_K**2 + 1.0e-300))
        inertia.append((state.u**2 * g[0]) / (state.R**2 * state.Omega_K**2 + 1.0e-300))
        qadv_qvisc.append(Q_adv / (Q_visc + 1.0e-300))
        qbalance.append((Q_visc - state.Q_rad) / (abs(Q_visc) + abs(state.Q_rad) + 1.0e-300))
        target = local_thin_targets(float(x), params, lambda0)
        angular_residual.append((state.l - target["l_k"]) / (target["l_k"] + 1.0e-300))
        stress_residual.append(state.W / (target["W_req"] + 1.0e-300) - 1.0)
        omega.append(state.Omega / state.Omega_K - 1.0)
        hr.append(state.H_over_R)
        sonic = sonic_diagnostics(float(x), y, lambda0, params)
        smin.append(sonic.smin_over_smax)
        condA.append(1.0 / (sonic.smin_over_smax + 1.0e-300))
        _ = energy_scale
    return {
        "measured_omega2_frac": np.asarray(measured, dtype=float),
        "pressure_frac": np.asarray(pressure, dtype=float),
        "inertia_frac": np.asarray(inertia, dtype=float),
        "qadv_qvisc": np.asarray(qadv_qvisc, dtype=float),
        "qbalance": np.asarray(qbalance, dtype=float),
        "angular_residual": np.asarray(angular_residual, dtype=float),
        "stress_residual": np.asarray(stress_residual, dtype=float),
        "omega_frac": np.asarray(omega, dtype=float),
        "H_over_R": np.asarray(hr, dtype=float),
        "smin_over_smax_A": np.asarray(smin, dtype=float),
        "condA": np.asarray(condA, dtype=float),
    }


def row_for_seed(params: TransonicSlimParams, z: np.ndarray, arrays: dict[str, np.ndarray], stress_factor: float) -> dict[str, object]:
    logu, logT, logR_son, lambda0, logR = params_unpack(z, params)
    residual = collocation_residual(z, params)
    intervals = interval_residuals(z, params)
    audit = residual_audit_from_state_vector(z, params)
    diag = pressure_support_diagnostics(z, params)
    mask = np.ones(len(logR), dtype=bool)
    if len(mask) > 8:
        mask[:2] = False
        mask[-2:] = False
    row = {
        "stress_factor": float(stress_factor),
        "ratio": float(params.mdot_edd_ratio),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "lambda0": float(lambda0),
        "lambda0_over_lK_isco": float(lambda0 * params.r_g * C / params.potential.l_k(params.potential.r_isco)),
        "analytic_local_max": float(np.max(arrays["local_norm"])),
        "selected_max": float(np.max(np.abs(residual))),
        "interval_R_max": float(np.max(np.abs(intervals[:, 0]))),
        "interval_E_max": float(np.max(np.abs(intervals[:, 1]))),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "D": float(audit.sonic_D),
        "C1": float(audit.sonic_C1),
        "C2": float(audit.sonic_C2),
        "K": float(audit.sonic_K),
        "sonic_smin": float(audit.sonic_smin_over_smax),
        "max_abs_omega": float(np.max(np.abs(diag["omega_frac"][mask]))),
        "max_abs_measured_omega2": float(np.max(np.abs(diag["measured_omega2_frac"][mask]))),
        "max_abs_pressure_support": float(np.max(np.abs(diag["pressure_frac"][mask]))),
        "max_abs_inertia_support": float(np.max(np.abs(diag["inertia_frac"][mask]))),
        "max_abs_qadv_qvisc": float(np.max(np.abs(diag["qadv_qvisc"][mask]))),
        "max_abs_qbalance": float(np.max(np.abs(diag["qbalance"][mask]))),
        "max_HR": float(np.max(diag["H_over_R"][mask])),
        "max_abs_angular_residual": float(np.max(np.abs(diag["angular_residual"][mask]))),
        "max_abs_stress_residual": float(np.max(np.abs(diag["stress_residual"][mask]))),
        "min_smin_A": float(np.min(diag["smin_over_smax_A"][mask])),
        "max_condA": float(np.max(diag["condA"][mask])),
    }
    row["dominant"] = max(
        {
            "interval_R": abs(row["interval_R_max"]),
            "interval_E": abs(row["interval_E_max"]),
            "outer_omega": abs(row["outer_omega"]),
            "outer_energy": abs(row["outer_energy"]),
            "D": abs(row["D"]),
            "C1": abs(row["C1"]),
            "C2": abs(row["C2"]),
            "K": abs(row["K"]),
        },
        key=lambda key: {
            "interval_R": abs(row["interval_R_max"]),
            "interval_E": abs(row["interval_E_max"]),
            "outer_omega": abs(row["outer_omega"]),
            "outer_energy": abs(row["outer_energy"]),
            "D": abs(row["D"]),
            "C1": abs(row["C1"]),
            "C2": abs(row["C2"]),
            "K": abs(row["K"]),
        }[key],
    )
    return row


def residual_profile_rows(params: TransonicSlimParams, z: np.ndarray, arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    logu, logT, _logR_son, lambda0, logR = params_unpack(z, params)
    intervals = interval_residuals(z, params)
    diag = pressure_support_diagnostics(z, params)
    rows = []
    for idx, x in enumerate(logR):
        sonic = sonic_diagnostics(float(x), np.array([logu[idx], logT[idx]], dtype=float), lambda0, params)
        rows.append(
            {
                "node": idx,
                "R_rg": float(np.exp(x) / params.r_g),
                "interval_R_left": float(intervals[idx - 1, 0]) if idx > 0 else np.nan,
                "interval_E_left": float(intervals[idx - 1, 1]) if idx > 0 else np.nan,
                "interval_R_right": float(intervals[idx, 0]) if idx < len(intervals) else np.nan,
                "interval_E_right": float(intervals[idx, 1]) if idx < len(intervals) else np.nan,
                "D": float(sonic.D),
                "C1": float(sonic.C1),
                "C2": float(sonic.C2),
                "K": float(sonic.compatibility),
                "smin_over_smax_A": float(sonic.smin_over_smax),
                "condA": float(1.0 / (sonic.smin_over_smax + 1.0e-300)),
                "Omega_frac": float(diag["omega_frac"][idx]),
                "measured_omega2_frac": float(diag["measured_omega2_frac"][idx]),
                "pressure_frac": float(diag["pressure_frac"][idx]),
                "inertia_frac": float(diag["inertia_frac"][idx]),
                "Qadv_Qvisc": float(diag["qadv_qvisc"][idx]),
                "qbalance": float(diag["qbalance"][idx]),
                "H_R": float(diag["H_over_R"][idx]),
                "angular_residual": float(diag["angular_residual"][idx]),
                "stress_residual": float(diag["stress_residual"][idx]),
                "local_solve_norm": float(arrays["local_norm"][idx]),
            }
        )
    return rows


def write_main_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Analytic Seed Residual Audit",
        "",
        "Generated by `scripts/run_standard_slim_analytic_seed_audit.py`.",
        "",
        "Analytic seed solves `W_model=W_req` and `Q_rad=Q_visc_thin`, then evaluates the full transonic residuals without optimizer steps.",
        "",
        "| stress | R_out/rg | N | local seed | selected | dominant | int R | int E | outer omega | outer E | D | C1 | C2 | K | Rson/rg | lambda/lK | Omega err | pressure frac | inertia frac | Qadv/Qvisc | qbalance | H/R | ang resid | stress resid | min sA | max condA |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {stress_factor} | {R_out_rg} | {N} | {analytic_local_max} | {selected_max} | {dominant} | "
            "{interval_R_max} | {interval_E_max} | {outer_omega} | {outer_energy} | {D} | {C1} | {C2} | {K} | "
            "{Rson_rg} | {lambda0_over_lK_isco} | {max_abs_omega} | {max_abs_pressure_support} | "
            "{max_abs_inertia_support} | {max_abs_qadv_qvisc} | {max_abs_qbalance} | {max_HR} | "
            "{max_abs_angular_residual} | {max_abs_stress_residual} | {min_smin_A} | {max_condA} |".format(
                **{key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps([{key: json_safe(value) for key, value in row.items()} for row in rows], indent=2, sort_keys=True) + "\n")


def write_formulation_table(rows: list[dict[str, object]]) -> None:
    FORMULATION_TABLE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Formulation Checks",
        "",
        "Generated by `scripts/run_standard_slim_analytic_seed_audit.py`.",
        "",
        "| stress | R_out/rg | N | angular residual | stress residual | qbalance | Qadv/Qvisc | measured omega2 | pressure support | inertia support | support mismatch | selected | dominant |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    payload = []
    for row in rows:
        support_mismatch = float(row["max_abs_measured_omega2"] - row["max_abs_pressure_support"] - row["max_abs_inertia_support"])
        check = {
            "stress_factor": row["stress_factor"],
            "R_out_rg": row["R_out_rg"],
            "N": row["N"],
            "max_abs_angular_residual": row["max_abs_angular_residual"],
            "max_abs_stress_residual": row["max_abs_stress_residual"],
            "max_abs_qbalance": row["max_abs_qbalance"],
            "max_abs_qadv_qvisc": row["max_abs_qadv_qvisc"],
            "max_abs_measured_omega2": row["max_abs_measured_omega2"],
            "max_abs_pressure_support": row["max_abs_pressure_support"],
            "max_abs_inertia_support": row["max_abs_inertia_support"],
            "support_mismatch": support_mismatch,
            "selected_max": row["selected_max"],
            "dominant": row["dominant"],
        }
        payload.append(check)
        lines.append(
            "| {stress_factor} | {R_out_rg} | {N} | {max_abs_angular_residual} | {max_abs_stress_residual} | "
            "{max_abs_qbalance} | {max_abs_qadv_qvisc} | {max_abs_measured_omega2} | {max_abs_pressure_support} | "
            "{max_abs_inertia_support} | {support_mismatch} | {selected_max} | {dominant} |".format(
                **{key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in check.items()}
            )
        )
    FORMULATION_TABLE.write_text("\n".join(lines) + "\n")
    FORMULATION_JSON.write_text(json.dumps([{key: json_safe(value) for key, value in row.items()} for row in payload], indent=2, sort_keys=True) + "\n")


def write_profile_table(rows: list[dict[str, object]]) -> None:
    PROFILE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Analytic Seed Residual Profile",
        "",
        "Generated by `scripts/run_standard_slim_analytic_seed_audit.py` for the primary stress factor and largest R_out.",
        "",
        "| node | R/rg | int R left | int E left | int R right | int E right | D | C1 | C2 | K | sA | condA | Omega frac | omega2 frac | pressure frac | inertia frac | Qadv/Qvisc | qbalance | H/R | angular resid | stress resid | local norm |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {node} | {R_rg} | {interval_R_left} | {interval_E_left} | {interval_R_right} | {interval_E_right} | "
            "{D} | {C1} | {C2} | {K} | {smin_over_smax_A} | {condA} | {Omega_frac} | {measured_omega2_frac} | "
            "{pressure_frac} | {inertia_frac} | {Qadv_Qvisc} | {qbalance} | {H_R} | {angular_residual} | "
            "{stress_residual} | {local_solve_norm} |".format(
                **{key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
            )
        )
    PROFILE_TABLE.write_text("\n".join(lines) + "\n")


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    primary = [row for row in rows if abs(float(row["stress_factor"]) - PRIMARY_STRESS_FACTOR) < 1.0e-12]
    if not primary:
        return
    primary = sorted(primary, key=lambda row: float(row["R_out_rg"]))
    width, height = 1150, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    panels = [
        (95, 80, 520, 360, "log10 selected max residual", "selected_max", True),
        (650, 80, 1075, 360, "log10 sonic |C2|", "C2", True),
        (95, 500, 520, 780, "log10 max |Omega/OmegaK - 1|", "max_abs_omega", True),
        (650, 500, 1075, 780, "log10 max |thermal imbalance|", "max_abs_qbalance", True),
    ]
    x_values = np.asarray([float(row["R_out_rg"]) for row in primary], dtype=float)
    x_plot = np.log10(x_values)
    x_min, x_max = float(np.min(x_plot)), float(np.max(x_plot))
    if x_max <= x_min:
        x_max = x_min + 1.0
    for x0, y0, x1, y1, title, key, logy in panels:
        values = np.asarray([abs(float(row[key])) for row in primary], dtype=float)
        y_plot = np.log10(np.maximum(values, 1.0e-16)) if logy else values
        y_min, y_max = float(np.min(y_plot)), float(np.max(y_plot))
        if y_max <= y_min:
            y_min -= 1.0
            y_max += 1.0
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad
        draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
        draw.text((x0, y0 - 28), title, fill=(20, 20, 20), font=font)
        points = []
        for xx, yy, label in zip(x_plot, y_plot, x_values):
            px = x0 + int((xx - x_min) / (x_max - x_min) * (x1 - x0))
            py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
            points.append((px, py))
            draw.text((px - 15, y1 + 8), f"{label:g}", fill=(50, 50, 50), font=font)
        if len(points) >= 2:
            draw.line(points, fill=(31, 119, 180), width=3)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=(31, 119, 180))
        draw.text((x0 + 5, y0 + 5), f"{y_max:.2e}", fill=(80, 80, 80), font=font)
        draw.text((x0 + 5, y1 - 18), f"{y_min:.2e}", fill=(80, 80, 80), font=font)
    draw.text((95, 25), "Analytic thin seed audit at Mdot/Edd=1e-3, x-axis R_out/rg", fill=(20, 20, 20), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    all_rows: list[dict[str, object]] = []
    primary_profile_rows: list[dict[str, object]] = []
    for stress_factor in STRESS_FACTORS:
        for R_out_rg in R_OUT_CASES:
            params = make_params(fiducial, mdot_edd, R_out_rg, stress_factor)
            lambda0 = float(params.potential.l_k(params.potential.r_isco) / (params.r_g * C))
            logR_son = float(np.log(R_SON_RG * params.r_g))
            print(
                f"analytic seed ratio={MDOT_RATIO:g} stress={stress_factor:g} "
                f"R_out={R_out_rg:g} N={params.n_nodes}",
                flush=True,
            )
            z, arrays = analytic_seed_state(params, logR_son, lambda0)
            row = row_for_seed(params, z, arrays, stress_factor)
            all_rows.append(row)
            if abs(stress_factor - PRIMARY_STRESS_FACTOR) < 1.0e-12 and R_out_rg == max(R_OUT_CASES):
                primary_profile_rows = residual_profile_rows(params, z, arrays)
            write_main_table(all_rows)
            write_formulation_table(all_rows)
            if primary_profile_rows:
                write_profile_table(primary_profile_rows)
            write_figure(all_rows)
            print(
                f"  selected={row['selected_max']:.3e} dominant={row['dominant']} "
                f"omega={row['max_abs_omega']:.3e} qbal={row['max_abs_qbalance']:.3e}",
                flush=True,
            )
    write_main_table(all_rows)
    write_formulation_table(all_rows)
    if primary_profile_rows:
        write_profile_table(primary_profile_rows)
    write_figure(all_rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FORMULATION_TABLE}", flush=True)
    print(f"wrote {PROFILE_TABLE}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
