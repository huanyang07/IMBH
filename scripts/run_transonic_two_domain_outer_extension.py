"""Prototype two-domain outer extension for the fixed-Mdot transonic root."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _differential_interval_residual_from_unpacked,
    _heating_terms_from_gradient,
    _outer_thin_boundary_residual,
    pressure_supported_omega_target,
    state_bounds,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    algebraic_state,
    differential_residual,
    differential_residual_scales,
    sonic_diagnostics,
    state_partials,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_slope_unknown_root import unpack_state
from run_transonic_staged_resolution_continuation import N65_SOURCE_CHECKPOINT, load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_two_domain_outer_extension.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_two_domain_outer_extension"

R_MATCH_RG = 6500.0
R_FAR_RG = 5.0e4
N_INNER = 65
N_OUTER = 32
CASES = (
    ("Rfar5e4", 5.0e4, 32),
    ("Rfar1e5", 1.0e5, 36),
)
FAR_CLOSURES = ("thin", "pressure_supported")
MAX_NFEV_LOCKED = 450
MAX_NFEV_RELEASE = 900
MAX_NFEV_POLISH = 300

LOCK_INNER_LOGU_HALF_WIDTH = 5.0e-2
LOCK_INNER_LOGT_HALF_WIDTH = 4.0e-2
LOCK_OUTER_LOGU_HALF_WIDTH = 5.0e-1
LOCK_OUTER_LOGT_HALF_WIDTH = 5.0e-1
LOCK_RSON_RG_HALF_WIDTH = 5.0e-2
LOCK_LAMBDA_HALF_WIDTH = 1.0e-3


@dataclass(frozen=True)
class TwoDomainParams:
    physics: TransonicSlimParams
    n_inner: int
    n_outer: int
    R_match_rg: float
    R_far_rg: float
    far_closure: str = "thin"
    grid_power_inner: float = 1.0
    grid_power_outer: float = 1.0

    @property
    def r_g(self) -> float:
        return self.physics.r_g

    @property
    def R_match(self) -> float:
        return self.R_match_rg * self.r_g

    @property
    def R_far(self) -> float:
        return self.R_far_rg * self.r_g


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


def two_domain_size(params: TwoDomainParams) -> int:
    return 2 * params.n_inner + 2 * params.n_outer + 2


def unpack_two_domain(x: np.ndarray, params: TwoDomainParams):
    x = np.asarray(x, dtype=float)
    expected = two_domain_size(params)
    if x.shape != (expected,):
        raise ValueError(f"x must have shape ({expected},)")
    ni = params.n_inner
    no = params.n_outer
    offset = 0
    logu_inner = x[offset : offset + ni]
    offset += ni
    logT_inner = x[offset : offset + ni]
    offset += ni
    logu_outer = x[offset : offset + no]
    offset += no
    logT_outer = x[offset : offset + no]
    offset += no
    logR_son = float(x[offset])
    lambda0 = float(x[offset + 1])
    return logu_inner, logT_inner, logu_outer, logT_outer, logR_son, lambda0


def pack_two_domain(
    logu_inner: np.ndarray,
    logT_inner: np.ndarray,
    logu_outer: np.ndarray,
    logT_outer: np.ndarray,
    logR_son: float,
    lambda0: float,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(logu_inner, dtype=float),
            np.asarray(logT_inner, dtype=float),
            np.asarray(logu_outer, dtype=float),
            np.asarray(logT_outer, dtype=float),
            np.array([logR_son, lambda0], dtype=float),
        ]
    )


def inner_grid(logR_son: float, params: TwoDomainParams) -> np.ndarray:
    xi = np.linspace(0.0, 1.0, params.n_inner)
    mapped = xi**params.grid_power_inner
    return logR_son + mapped * (np.log(params.R_match) - logR_son)


def outer_grid(params: TwoDomainParams) -> np.ndarray:
    eta = np.linspace(0.0, 1.0, params.n_outer)
    mapped = eta**params.grid_power_outer
    return np.log(params.R_match) + mapped * (np.log(params.R_far) - np.log(params.R_match))


def sonic_values(logR_son: float, logu0: float, logT0: float, lambda0: float, params: TwoDomainParams) -> tuple[float, float, float, float]:
    sonic = sonic_diagnostics(logR_son, np.array([logu0, logT0], dtype=float), lambda0, params.physics)
    return float(sonic.D), float(sonic.C1), float(sonic.C2), float(sonic.compatibility)


def far_slope(logu_o: np.ndarray, logT_o: np.ndarray, logR_o: np.ndarray) -> np.ndarray:
    dx = float(logR_o[-1] - logR_o[-2])
    return np.array(
        [
            (float(logu_o[-1]) - float(logu_o[-2])) / dx,
            (float(logT_o[-1]) - float(logT_o[-2])) / dx,
        ],
        dtype=float,
    )


def far_force_diagnostic(logu_o: np.ndarray, logT_o: np.ndarray, logR_o: np.ndarray, lambda0: float, params: TwoDomainParams) -> dict[str, float]:
    y = np.array([logu_o[-1], logT_o[-1]], dtype=float)
    g = far_slope(logu_o, logT_o, logR_o)
    state = algebraic_state(float(logR_o[-1]), float(y[0]), float(y[1]), lambda0, params.physics)
    partials = state_partials(float(logR_o[-1]), y, lambda0, params.physics, eps_x=params.physics.partial_eps, eps_y=params.physics.partial_eps)
    dPi_dx = partials.x["Pi"] + float(np.dot(partials.y["Pi"], g))
    dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], g))
    denom = state.R**2 * state.Omega_K**2 + 1.0e-300
    inertia_fraction = float(state.u**2 * g[0] / denom)
    pressure_fraction = float((dPi_dx / state.Sigma) / denom)
    actual_omega2_fraction = float((state.Omega**2 - state.Omega_K**2) / (state.Omega_K**2 + 1.0e-300))
    target = pressure_supported_omega_target(float(logR_o[-1]), y, g, lambda0, params.physics)
    thin = _outer_thin_boundary_residual(float(logR_o[-1]), y, lambda0, params.physics)
    raw = differential_residual(float(logR_o[-1]), y, g, lambda0, params.physics)
    radial_scale, energy_scale = differential_residual_scales(float(logR_o[-1]), y, lambda0, params.physics)
    Q_visc, Q_rad, Q_adv, energy = _heating_terms_from_gradient(float(logR_o[-1]), y, g, lambda0, params.physics)
    shear = float(params.physics.potential.dln_omega_k_dlnR(state.R))
    Q_visc_thin = -state.W * state.Omega_K * shear
    return {
        "g_u_far": float(g[0]),
        "g_T_far": float(g[1]),
        "g_Pi_far": float(dPi_dx / (state.Pi + 1.0e-300)),
        "dlnOmega_far": float(dOmega_dx / (state.Omega + 1.0e-300)),
        "thin_omega": float(thin[0]),
        "pressure_target": float(target),
        "pressure_residual": float(thin[0] - target),
        "inertia_fraction": inertia_fraction,
        "pressure_fraction": pressure_fraction,
        "supported_omega2_fraction": inertia_fraction + pressure_fraction,
        "actual_omega2_fraction": actual_omega2_fraction,
        "radial_scaled": float(raw[0] / radial_scale),
        "thin_energy": float(thin[1]),
        "local_energy_scaled": float(raw[1] / energy_scale),
        "Qvisc_over_Qvisc_thin": float(Q_visc / (Q_visc_thin + 1.0e-300)),
        "Qadv_over_Qvisc": float(Q_adv / (Q_visc + 1.0e-300)),
        "Qrad_over_Qvisc": float(Q_rad / (Q_visc + 1.0e-300)),
        "energy_scaled_check": float(energy / energy_scale),
    }


def far_boundary_residual(logu_o: np.ndarray, logT_o: np.ndarray, logR_o: np.ndarray, lambda0: float, params: TwoDomainParams) -> np.ndarray:
    y = np.array([logu_o[-1], logT_o[-1]], dtype=float)
    thin = _outer_thin_boundary_residual(float(logR_o[-1]), y, lambda0, params.physics)
    if params.far_closure == "thin":
        return thin
    if params.far_closure == "pressure_supported":
        target = pressure_supported_omega_target(float(logR_o[-1]), y, far_slope(logu_o, logT_o, logR_o), lambda0, params.physics)
        return np.array([thin[0] - target, thin[1]], dtype=float)
    raise ValueError(f"unknown far_closure {params.far_closure!r}")


def two_domain_residual(x: np.ndarray, params: TwoDomainParams) -> np.ndarray:
    rows = []
    try:
        logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_two_domain(x, params)
        logR_i = inner_grid(logR_son, params)
        logR_o = outer_grid(params)
        if np.any(np.diff(logR_i) <= 0.0) or np.any(np.diff(logR_o) <= 0.0):
            raise ValueError("mapped radii must increase")

        for idx in range(params.n_inner - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx))
        for idx in range(params.n_outer - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx))

        rows.append(np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float))
        rows.append(far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params))

        D, C1, C2, _K = sonic_values(logR_son, float(logu_i[0]), float(logT_i[0]), lambda0, params)
        rows.append(np.array([D, C1, C2], dtype=float))
        return np.concatenate(rows)
    except Exception:
        return np.full(2 * (params.n_inner - 1) + 2 * (params.n_outer - 1) + 7, 1.0e6)


def two_domain_sparsity(params: TwoDomainParams):
    n_unknown = two_domain_size(params)
    n_rows = 2 * (params.n_inner - 1) + 2 * (params.n_outer - 1) + 2 + 2 + 3
    pattern = lil_matrix((n_rows, n_unknown), dtype=int)
    ni = params.n_inner
    no = params.n_outer
    iu = 0
    iT = ni
    ou = 2 * ni
    oT = 2 * ni + no
    logR_col = 2 * ni + 2 * no
    lambda_col = logR_col + 1

    row = 0
    for idx in range(ni - 1):
        columns = (iu + idx, iu + idx + 1, iT + idx, iT + idx + 1, logR_col, lambda_col)
        for col in columns:
            pattern[row : row + 2, col] = 1
        row += 2
    for idx in range(no - 1):
        columns = (ou + idx, ou + idx + 1, oT + idx, oT + idx + 1, lambda_col)
        for col in columns:
            pattern[row : row + 2, col] = 1
        row += 2

    for col in (iu + ni - 1, iT + ni - 1, ou, oT):
        pattern[row : row + 2, col] = 1
    row += 2

    for col in (ou + no - 2, ou + no - 1, oT + no - 2, oT + no - 1, lambda_col):
        pattern[row : row + 2, col] = 1
    row += 2

    for col in (iu, iT, logR_col, lambda_col):
        pattern[row : row + 3, col] = 1
    return pattern.tocsr()


def state_bounds_two_domain(params: TwoDomainParams) -> tuple[np.ndarray, np.ndarray]:
    lower_one, upper_one = state_bounds(params.physics)
    lower_logu, upper_logu = lower_one[0], upper_one[0]
    lower_logT, upper_logT = lower_one[params.physics.n_nodes], upper_one[params.physics.n_nodes]
    lower = np.concatenate(
        [
            np.full(params.n_inner, lower_logu),
            np.full(params.n_inner, lower_logT),
            np.full(params.n_outer, lower_logu),
            np.full(params.n_outer, lower_logT),
            np.array([lower_one[-2], lower_one[-1]], dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.full(params.n_inner, upper_logu),
            np.full(params.n_inner, upper_logT),
            np.full(params.n_outer, upper_logu),
            np.full(params.n_outer, upper_logT),
            np.array([upper_one[-2], upper_one[-1]], dtype=float),
        ]
    )
    return lower, upper


def locked_bounds(params: TwoDomainParams, seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = state_bounds_two_domain(params)
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_two_domain(seed, params)
    ni = params.n_inner
    no = params.n_outer
    iu = 0
    iT = ni
    ou = 2 * ni
    oT = 2 * ni + no
    logR_col = 2 * ni + 2 * no
    lambda_col = logR_col + 1
    lower[iu : iu + ni] = np.maximum(lower[iu : iu + ni], logu_i - LOCK_INNER_LOGU_HALF_WIDTH)
    upper[iu : iu + ni] = np.minimum(upper[iu : iu + ni], logu_i + LOCK_INNER_LOGU_HALF_WIDTH)
    lower[iT : iT + ni] = np.maximum(lower[iT : iT + ni], logT_i - LOCK_INNER_LOGT_HALF_WIDTH)
    upper[iT : iT + ni] = np.minimum(upper[iT : iT + ni], logT_i + LOCK_INNER_LOGT_HALF_WIDTH)
    lower[ou : ou + no] = np.maximum(lower[ou : ou + no], logu_o - LOCK_OUTER_LOGU_HALF_WIDTH)
    upper[ou : ou + no] = np.minimum(upper[ou : ou + no], logu_o + LOCK_OUTER_LOGU_HALF_WIDTH)
    lower[oT : oT + no] = np.maximum(lower[oT : oT + no], logT_o - LOCK_OUTER_LOGT_HALF_WIDTH)
    upper[oT : oT + no] = np.minimum(upper[oT : oT + no], logT_o + LOCK_OUTER_LOGT_HALF_WIDTH)

    rson_seed_rg = float(np.exp(logR_son) / params.r_g)
    lower[logR_col] = max(lower[logR_col], np.log((rson_seed_rg - LOCK_RSON_RG_HALF_WIDTH) * params.r_g))
    upper[logR_col] = min(upper[logR_col], np.log((rson_seed_rg + LOCK_RSON_RG_HALF_WIDTH) * params.r_g))
    lower[lambda_col] = max(lower[lambda_col], lambda0 - LOCK_LAMBDA_HALF_WIDTH)
    upper[lambda_col] = min(upper[lambda_col], lambda0 + LOCK_LAMBDA_HALF_WIDTH)
    return lower, upper


def solve_two_domain(seed: np.ndarray, params: TwoDomainParams, bounds: tuple[np.ndarray, np.ndarray], max_nfev: int):
    lower, upper = bounds
    x0 = np.clip(np.asarray(seed, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: two_domain_residual(trial, params),
        x0,
        jac_sparsity=two_domain_sparsity(params),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=max_nfev,
    )


def interval_blocks(logu: np.ndarray, logT: np.ndarray, logR: np.ndarray, lambda0: float, params: TwoDomainParams) -> np.ndarray:
    return np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params.physics, idx)
            for idx in range(len(logR) - 1)
        ],
        dtype=float,
    )


def integrated_advective_fraction(logu: np.ndarray, logT: np.ndarray, logR: np.ndarray, lambda0: float, params: TwoDomainParams) -> float:
    weights = []
    qvisc_values = []
    qadv_values = []
    R = np.exp(logR)
    for idx in range(len(logR) - 1):
        dx = float(logR[idx + 1] - logR[idx])
        xm = 0.5 * float(logR[idx] + logR[idx + 1])
        ym = np.array([0.5 * (logu[idx] + logu[idx + 1]), 0.5 * (logT[idx] + logT[idx + 1])], dtype=float)
        gm = np.array([(logu[idx + 1] - logu[idx]) / dx, (logT[idx + 1] - logT[idx]) / dx], dtype=float)
        qv, _qr, qa, _qe = _heating_terms_from_gradient(xm, ym, gm, lambda0, params.physics)
        weights.append(2.0 * np.pi * float(np.exp(xm)) * float(R[idx + 1] - R[idx]))
        qvisc_values.append(qv)
        qadv_values.append(qa)
    weights = np.asarray(weights, dtype=float)
    qvisc_values = np.asarray(qvisc_values, dtype=float)
    qadv_values = np.asarray(qadv_values, dtype=float)
    return float(np.sum(weights * qadv_values) / (np.sum(weights * np.abs(qvisc_values)) + 1.0e-300))


def audit_row(label: str, x: np.ndarray, params: TwoDomainParams, result=None) -> dict[str, object]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_two_domain(x, params)
    logR_i = inner_grid(logR_son, params)
    logR_o = outer_grid(params)
    inner = interval_blocks(logu_i, logT_i, logR_i, lambda0, params)
    outer = interval_blocks(logu_o, logT_o, logR_o, lambda0, params)
    interface = np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float)
    far = far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params)
    far_diag = far_force_diagnostic(logu_o, logT_o, logR_o, lambda0, params)
    D, C1, C2, K = sonic_values(logR_son, float(logu_i[0]), float(logT_i[0]), lambda0, params)
    residual = two_domain_residual(x, params)

    combined_logR = np.concatenate([logR_i, logR_o[1:]])
    combined_logu = np.concatenate([logu_i, logu_o[1:]])
    combined_logT = np.concatenate([logT_i, logT_o[1:]])
    H_over_R = np.asarray(
        [
            algebraic_state(float(lr), float(lu), float(lt), lambda0, params.physics).H_over_R
            for lr, lu, lt in zip(combined_logR, combined_logu, combined_logT)
        ],
        dtype=float,
    )
    g_inner_match = np.array(
        [
            (logu_i[-1] - logu_i[-2]) / (logR_i[-1] - logR_i[-2]),
            (logT_i[-1] - logT_i[-2]) / (logR_i[-1] - logR_i[-2]),
        ],
        dtype=float,
    )
    g_outer_match = np.array(
        [
            (logu_o[1] - logu_o[0]) / (logR_o[1] - logR_o[0]),
            (logT_o[1] - logT_o[0]) / (logR_o[1] - logR_o[0]),
        ],
        dtype=float,
    )
    block_values = {
        "inner_R": float(np.max(np.abs(inner[:, 0]))),
        "inner_E": float(np.max(np.abs(inner[:, 1]))),
        "outer_R": float(np.max(np.abs(outer[:, 0]))),
        "outer_E": float(np.max(np.abs(outer[:, 1]))),
        "interface": float(np.max(np.abs(interface))),
        "far_omega": abs(float(far[0])),
        "far_energy": abs(float(far[1])),
        "D": abs(D),
        "C1": abs(C1),
        "C2": abs(C2),
        "K": abs(K),
    }
    physical = max(block_values.values())
    dominant = max(block_values, key=block_values.get)
    return {
        "label": label,
        "far_closure": params.far_closure,
        "ratio": params.physics.mdot_edd_ratio,
        "R_match_rg": params.R_match_rg,
        "R_far_rg": params.R_far_rg,
        "n_inner": params.n_inner,
        "n_outer": params.n_outer,
        "selected_max": float(np.max(np.abs(residual))),
        "physical_active": physical,
        "dominant": dominant,
        "inner_R": block_values["inner_R"],
        "inner_E": block_values["inner_E"],
        "outer_R": block_values["outer_R"],
        "outer_E": block_values["outer_E"],
        "interface": block_values["interface"],
        "far_omega": float(far[0]),
        "far_energy": float(far[1]),
        "D": D,
        "C1": C1,
        "C2": C2,
        "K": K,
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "lambda0": float(lambda0),
        "int_adv": integrated_advective_fraction(combined_logu, combined_logT, combined_logR, lambda0, params),
        "max_HR": float(np.max(H_over_R)),
        "match_HR_inner": float(H_over_R[params.n_inner - 1]),
        "far_HR": float(H_over_R[-1]),
        "g_u_inner_match": float(g_inner_match[0]),
        "g_T_inner_match": float(g_inner_match[1]),
        "g_u_outer_match": float(g_outer_match[0]),
        "g_T_outer_match": float(g_outer_match[1]),
        "g_u_far": far_diag["g_u_far"],
        "g_T_far": far_diag["g_T_far"],
        "g_Pi_far": far_diag["g_Pi_far"],
        "dlnOmega_far": far_diag["dlnOmega_far"],
        "far_thin_omega": far_diag["thin_omega"],
        "far_pressure_target": far_diag["pressure_target"],
        "far_pressure_residual": far_diag["pressure_residual"],
        "far_inertia_fraction": far_diag["inertia_fraction"],
        "far_pressure_fraction": far_diag["pressure_fraction"],
        "far_supported_omega2_fraction": far_diag["supported_omega2_fraction"],
        "far_actual_omega2_fraction": far_diag["actual_omega2_fraction"],
        "far_radial_scaled": far_diag["radial_scaled"],
        "far_thin_energy": far_diag["thin_energy"],
        "far_local_energy_scaled": far_diag["local_energy_scaled"],
        "far_Qvisc_over_Qvisc_thin": far_diag["Qvisc_over_Qvisc_thin"],
        "far_Qadv_over_Qvisc": far_diag["Qadv_over_Qvisc"],
        "far_Qrad_over_Qvisc": far_diag["Qrad_over_Qvisc"],
        "nfev": int(result.nfev) if result is not None else 0,
        "success": bool(result.success) if result is not None else True,
        "message": str(result.message) if result is not None else "seed evaluation",
        "x": np.asarray(x, dtype=float),
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{row['label']}_{float(row['ratio']):.8f}".replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "x"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        x=np.asarray(row["x"], dtype=float),
        ratio=np.array(row["ratio"]),
        far_closure=np.array(row["far_closure"]),
        R_match_rg=np.array(row["R_match_rg"]),
        R_far_rg=np.array(row["R_far_rg"]),
        n_inner=np.array(row["n_inner"]),
        n_outer=np.array(row["n_outer"]),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    case_text = ", ".join(f"`{label}`: `{R_MATCH_RG:g} -> {r_far:g} rg` with `N_outer={n_outer}`" for label, r_far, n_outer in CASES)
    lines = [
        "# Two-Domain Outer Extension Prototype",
        "",
        "Generated by `scripts/run_transonic_two_domain_outer_extension.py`.",
        "",
        f"Prototype midpoint collocation with inner domain `R_son -> {R_MATCH_RG:g} rg`, outer-domain cases {case_text}, interface continuity, and selectable far boundary. The `thin` closure imposes exact `Omega=Omega_K`; the `pressure_supported` closure imposes radial force balance using the last outer interval slope while retaining the thin thermal balance.",
        "",
        "| label | closure | selected | physical | dominant | inner R | inner E | outer R | outer E | interface | far closure omega | far energy | thin omega | pressure target | pressure residual | inertia frac | pressure frac | radial scaled | local E scaled | D | C1 | C2 | K | Rson/rg | lambda0 | int adv | max H/R | match H/R | far H/R | g_u match in | g_T match in | g_u match out | g_T match out | g_u far | g_T far | g_Pi far | dlnOmega far | Qvisc/thin | Qadv/Qvisc | Qrad/Qvisc | nfev | success | message |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {far_closure} | {selected_max} | {physical_active} | {dominant} | {inner_R} | {inner_E} | "
            "{outer_R} | {outer_E} | {interface} | {far_omega} | {far_energy} | {far_thin_omega} | "
            "{far_pressure_target} | {far_pressure_residual} | {far_inertia_fraction} | {far_pressure_fraction} | "
            "{far_radial_scaled} | {far_local_energy_scaled} | {D} | {C1} | {C2} | {K} | {Rson_rg} | {lambda0} | "
            "{int_adv} | {max_HR} | {match_HR_inner} | {far_HR} | {g_u_inner_match} | {g_T_inner_match} | "
            "{g_u_outer_match} | {g_T_outer_match} | {g_u_far} | {g_T_far} | {g_Pi_far} | {dlnOmega_far} | "
            "{far_Qvisc_over_Qvisc_thin} | {far_Qadv_over_Qvisc} | {far_Qrad_over_Qvisc} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                far_closure=row["far_closure"],
                selected_max=fmt(float(row["selected_max"])),
                physical_active=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                inner_R=fmt(float(row["inner_R"])),
                inner_E=fmt(float(row["inner_E"])),
                outer_R=fmt(float(row["outer_R"])),
                outer_E=fmt(float(row["outer_E"])),
                interface=fmt(float(row["interface"])),
                far_omega=fmt(float(row["far_omega"])),
                far_energy=fmt(float(row["far_energy"])),
                far_thin_omega=fmt(float(row["far_thin_omega"])),
                far_pressure_target=fmt(float(row["far_pressure_target"])),
                far_pressure_residual=fmt(float(row["far_pressure_residual"])),
                far_inertia_fraction=fmt(float(row["far_inertia_fraction"])),
                far_pressure_fraction=fmt(float(row["far_pressure_fraction"])),
                far_radial_scaled=fmt(float(row["far_radial_scaled"])),
                far_local_energy_scaled=fmt(float(row["far_local_energy_scaled"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                max_HR=fmt(float(row["max_HR"])),
                match_HR_inner=fmt(float(row["match_HR_inner"])),
                far_HR=fmt(float(row["far_HR"])),
                g_u_inner_match=fmt(float(row["g_u_inner_match"])),
                g_T_inner_match=fmt(float(row["g_T_inner_match"])),
                g_u_outer_match=fmt(float(row["g_u_outer_match"])),
                g_T_outer_match=fmt(float(row["g_T_outer_match"])),
                g_u_far=fmt(float(row["g_u_far"])),
                g_T_far=fmt(float(row["g_T_far"])),
                g_Pi_far=fmt(float(row["g_Pi_far"])),
                dlnOmega_far=fmt(float(row["dlnOmega_far"])),
                far_Qvisc_over_Qvisc_thin=fmt(float(row["far_Qvisc_over_Qvisc_thin"])),
                far_Qadv_over_Qvisc=fmt(float(row["far_Qadv_over_Qvisc"])),
                far_Qrad_over_Qvisc=fmt(float(row["far_Qrad_over_Qvisc"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def make_params(fiducial: FiducialParams, ratio: float, mdot_edd: float, R_far_rg: float, n_outer: int, far_closure: str) -> TwoDomainParams:
    if far_closure not in FAR_CLOSURES:
        raise ValueError(f"far_closure must be one of {FAR_CLOSURES}")
    physics_outer_closure = "pressure_supported_thin_energy" if far_closure == "pressure_supported" else "thin_value"
    physics = TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=N_INNER,
        R_out_rg=R_far_rg,
        residual_tol=1.0e-6,
        max_nfev=MAX_NFEV_RELEASE,
        outer_closure=physics_outer_closure,
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )
    return TwoDomainParams(physics=physics, n_inner=N_INNER, n_outer=n_outer, R_match_rg=R_MATCH_RG, R_far_rg=R_far_rg, far_closure=far_closure)


def make_seed(params: TwoDomainParams) -> tuple[np.ndarray, dict[str, object]]:
    z65, row65 = load_checkpoint(N65_SOURCE_CHECKPOINT)
    logu_i, logT_i, logR_son, lambda0, _logR_i = unpack_state(z65, params.physics)
    g_u = float(row65["g_u_solved"])
    g_T = float(row65["g_T_solved"])
    logR_o = outer_grid(params)
    delta = logR_o - logR_o[0]
    logu_o = logu_i[-1] + g_u * delta
    logT_o = logT_i[-1] + g_T * delta
    logu_o[0] = logu_i[-1]
    logT_o[0] = logT_i[-1]
    return pack_two_domain(logu_i, logT_i, logu_o, logT_o, logR_son, lambda0), row65


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    _z65, row65 = load_checkpoint(N65_SOURCE_CHECKPOINT)
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for case_label, R_far_rg, n_outer in CASES:
        thin_params = make_params(fiducial, float(row65["ratio"]), mdot_edd, R_far_rg, n_outer, "thin")
        seed, _seed_row = make_seed(thin_params)

        seed_row = audit_row(f"{case_label}_thin_seed", seed, thin_params)
        rows.append(seed_row)
        save_checkpoint(seed_row)
        write_table(rows)
        print(
            f"{case_label} thin seed physical={seed_row['physical_active']:.3e} dominant={seed_row['dominant']} "
            f"Rson={seed_row['Rson_rg']:.4f} far_HR={seed_row['far_HR']:.3e}",
            flush=True,
        )

        locked_result = solve_two_domain(seed, thin_params, locked_bounds(thin_params, seed), MAX_NFEV_LOCKED)
        locked_row = audit_row(f"{case_label}_thin_locked", locked_result.x, thin_params, locked_result)
        rows.append(locked_row)
        save_checkpoint(locked_row)
        write_table(rows)
        print(
            f"{case_label} thin locked physical={locked_row['physical_active']:.3e} dominant={locked_row['dominant']} "
            f"Rson={locked_row['Rson_rg']:.4f} nfev={locked_row['nfev']}",
            flush=True,
        )

        release_result = solve_two_domain(locked_result.x, thin_params, state_bounds_two_domain(thin_params), MAX_NFEV_RELEASE)
        release_row = audit_row(f"{case_label}_thin_release", release_result.x, thin_params, release_result)
        rows.append(release_row)
        save_checkpoint(release_row)
        write_table(rows)
        print(
            f"{case_label} thin release physical={release_row['physical_active']:.3e} dominant={release_row['dominant']} "
            f"Rson={release_row['Rson_rg']:.4f} nfev={release_row['nfev']}",
            flush=True,
        )

        polish_result = solve_two_domain(release_result.x, thin_params, state_bounds_two_domain(thin_params), MAX_NFEV_POLISH)
        polish_row = audit_row(f"{case_label}_thin_polish", polish_result.x, thin_params, polish_result)
        rows.append(polish_row)
        save_checkpoint(polish_row)
        write_table(rows)
        print(
            f"{case_label} thin polish physical={polish_row['physical_active']:.3e} dominant={polish_row['dominant']} "
            f"Rson={polish_row['Rson_rg']:.4f} nfev={polish_row['nfev']}",
            flush=True,
        )

        pressure_params = make_params(fiducial, float(row65["ratio"]), mdot_edd, R_far_rg, n_outer, "pressure_supported")
        pressure_start = polish_result.x
        pressure_start_row = audit_row(f"{case_label}_pressure_supported_start", pressure_start, pressure_params)
        rows.append(pressure_start_row)
        save_checkpoint(pressure_start_row)
        write_table(rows)
        print(
            f"{case_label} pressure start physical={pressure_start_row['physical_active']:.3e} "
            f"dominant={pressure_start_row['dominant']} thin_omega={pressure_start_row['far_thin_omega']:.3e} "
            f"target={pressure_start_row['far_pressure_target']:.3e}",
            flush=True,
        )

        pressure_locked = solve_two_domain(pressure_start, pressure_params, locked_bounds(pressure_params, pressure_start), MAX_NFEV_LOCKED)
        pressure_locked_row = audit_row(f"{case_label}_pressure_supported_locked", pressure_locked.x, pressure_params, pressure_locked)
        rows.append(pressure_locked_row)
        save_checkpoint(pressure_locked_row)
        write_table(rows)
        print(
            f"{case_label} pressure locked physical={pressure_locked_row['physical_active']:.3e} "
            f"dominant={pressure_locked_row['dominant']} Rson={pressure_locked_row['Rson_rg']:.4f} nfev={pressure_locked_row['nfev']}",
            flush=True,
        )

        pressure_release = solve_two_domain(pressure_locked.x, pressure_params, state_bounds_two_domain(pressure_params), MAX_NFEV_RELEASE)
        pressure_release_row = audit_row(f"{case_label}_pressure_supported_release", pressure_release.x, pressure_params, pressure_release)
        rows.append(pressure_release_row)
        save_checkpoint(pressure_release_row)
        write_table(rows)
        print(
            f"{case_label} pressure release physical={pressure_release_row['physical_active']:.3e} "
            f"dominant={pressure_release_row['dominant']} Rson={pressure_release_row['Rson_rg']:.4f} nfev={pressure_release_row['nfev']}",
            flush=True,
        )

        pressure_polish = solve_two_domain(pressure_release.x, pressure_params, state_bounds_two_domain(pressure_params), MAX_NFEV_POLISH)
        pressure_polish_row = audit_row(f"{case_label}_pressure_supported_polish", pressure_polish.x, pressure_params, pressure_polish)
        rows.append(pressure_polish_row)
        save_checkpoint(pressure_polish_row)
        write_table(rows)
        print(
            f"{case_label} pressure polish physical={pressure_polish_row['physical_active']:.3e} "
            f"dominant={pressure_polish_row['dominant']} Rson={pressure_polish_row['Rson_rg']:.4f} nfev={pressure_polish_row['nfev']}",
            flush=True,
        )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
