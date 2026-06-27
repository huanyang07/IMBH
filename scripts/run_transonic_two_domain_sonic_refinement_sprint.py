"""Sonic-focused two-domain inner-refinement experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _differential_interval_residual_from_unpacked,
    _heating_terms_from_gradient,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state, sonic_diagnostics
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_mesh_validation import (
    combined_profile_arrays,
    compare_to_reference,
    load_checkpoint,
    make_params,
)
from run_transonic_two_domain_outer_extension import (
    R_MATCH_RG,
    TwoDomainParams,
    audit_row,
    far_boundary_residual,
    inner_grid,
    integrated_advective_fraction,
    outer_grid,
    pack_two_domain,
    state_bounds_two_domain,
    unpack_two_domain,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_two_domain_mesh_validation" / "outer_N65_O54_R1e5_0p90277664.npz"
TABLE_DIR = ROOT / "outputs" / "tables"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_two_domain_sonic_refinement_sprint"
SONIC_POLISH_TABLE = TABLE_DIR / "transonic_two_domain_sonic_focused_polish.md"
BUFFER_TABLE = TABLE_DIR / "transonic_two_domain_sonic_buffer_refinement.md"
DEFECT_TABLE = TABLE_DIR / "transonic_two_domain_defect_preserving_refinement.md"
NESTED_TABLE = TABLE_DIR / "transonic_two_domain_nested_regular_refinement.md"

MAX_NFEV_SONIC_POLISH = 600
MAX_NFEV_BUFFER_RELEASE = 300
MAX_NFEV_BUFFER_POLISH = 150
MAX_NFEV_LOCAL_SPLIT = 80
SCIENCE_LIMIT = 5.0e-6
BUFFER_GATE_LIMIT = 2.0e-5
NESTED_GATE_LIMIT = 5.0e-5
DELTAS = (0.02, 0.03, 0.05)
N_REGULAR_BUFFER = 64
N_REGULAR_NESTED = (32, 64, 128)


@dataclass(frozen=True)
class BufferGridParams:
    physics: object
    n_regular: int
    n_outer: int
    R_match_rg: float
    R_far_rg: float
    delta_s: float
    far_closure: str = "pressure_supported"
    grid_power_outer: float = 1.0

    @property
    def n_inner(self) -> int:
        return self.n_regular + 2

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


def save_checkpoint(label: str, x: np.ndarray, row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in row.items() if key != "x"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{label}_0p90277664.npz",
        x=np.asarray(x, dtype=float),
        row_json=np.array(row_json(payload)),
    )


def make_buffer_params(
    fiducial: FiducialParams,
    ratio: float,
    mdot_edd: float,
    n_regular: int,
    n_outer: int,
    R_far_rg: float,
    delta_s: float,
) -> BufferGridParams:
    base = make_params(fiducial, ratio, mdot_edd, n_regular + 2, n_outer, R_far_rg)
    return BufferGridParams(
        physics=base.physics,
        n_regular=int(n_regular),
        n_outer=int(n_outer),
        R_match_rg=R_MATCH_RG,
        R_far_rg=float(R_far_rg),
        delta_s=float(delta_s),
        far_closure="pressure_supported",
    )


def buffer_inner_grid(logR_son: float, params: BufferGridParams) -> np.ndarray:
    logR_buffer = logR_son + params.delta_s
    logR_match = np.log(params.R_match)
    if logR_buffer >= logR_match:
        raise ValueError("sonic buffer exceeds match radius")
    regular = np.linspace(logR_buffer, logR_match, params.n_regular + 1)
    return np.concatenate([np.array([logR_son], dtype=float), regular])


def interval_residual_between(
    x_left: float,
    y_left: np.ndarray,
    x_right: float,
    y_right: np.ndarray,
    lambda0: float,
    physics,
) -> np.ndarray:
    return _differential_interval_residual_from_unpacked(
        np.array([float(y_left[0]), float(y_right[0])], dtype=float),
        np.array([float(y_left[1]), float(y_right[1])], dtype=float),
        np.array([float(x_left), float(x_right)], dtype=float),
        lambda0,
        physics,
        0,
    )


def source_first_slope(x_source: np.ndarray, source_params: TwoDomainParams) -> np.ndarray:
    logu_i, logT_i, _logu_o, _logT_o, logR_son, _lambda0 = unpack_two_domain(x_source, source_params)
    logR_i = inner_grid(logR_son, source_params)
    dx = float(logR_i[1] - logR_i[0])
    return np.array(
        [
            (float(logu_i[1]) - float(logu_i[0])) / dx,
            (float(logT_i[1]) - float(logT_i[0])) / dx,
        ],
        dtype=float,
    )


def unpack_buffer(x: np.ndarray, params: BufferGridParams):
    return unpack_two_domain(x, params)  # type: ignore[arg-type]


def patch_residual(logu_i: np.ndarray, logT_i: np.ndarray, params: BufferGridParams, g_s: np.ndarray) -> np.ndarray:
    slope = np.array(
        [
            (float(logu_i[1]) - float(logu_i[0])) / params.delta_s,
            (float(logT_i[1]) - float(logT_i[0])) / params.delta_s,
        ],
        dtype=float,
    )
    scale = np.maximum(1.0, np.abs(g_s))
    return (slope - g_s) / scale


def buffer_residual(x: np.ndarray, params: BufferGridParams, g_s: np.ndarray) -> np.ndarray:
    rows = []
    try:
        logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_buffer(x, params)
        logR_i = buffer_inner_grid(logR_son, params)
        logR_o = outer_grid(params)  # type: ignore[arg-type]
        if np.any(np.diff(logR_i) <= 0.0) or np.any(np.diff(logR_o) <= 0.0):
            raise ValueError("mapped radii must increase")

        rows.append(patch_residual(logu_i, logT_i, params, g_s))
        for idx in range(1, params.n_inner - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx))
        for idx in range(params.n_outer - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx))
        rows.append(np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float))
        rows.append(far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params))  # type: ignore[arg-type]
        sonic = sonic_diagnostics(logR_son, np.array([logu_i[0], logT_i[0]], dtype=float), lambda0, params.physics)
        rows.append(np.array([sonic.D, sonic.C1, sonic.C2], dtype=float))
        return np.concatenate(rows)
    except Exception:
        return np.full(2 * params.n_inner + 2 * params.n_outer + 3, 1.0e6)


def buffer_sparsity(params: BufferGridParams):
    n_unknown = 2 * params.n_inner + 2 * params.n_outer + 2
    n_rows = 2 * params.n_inner + 2 * params.n_outer + 3
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

    for col in (iu, iu + 1, iT, iT + 1):
        pattern[row : row + 2, col] = 1
    row += 2

    for idx in range(1, ni - 1):
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


def solve_buffer(seed: np.ndarray, params: BufferGridParams, g_s: np.ndarray, max_nfev: int):
    lower, upper = state_bounds_two_domain(params)  # type: ignore[arg-type]
    x0 = np.clip(np.asarray(seed, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: buffer_residual(trial, params, g_s),
        x0,
        jac_sparsity=buffer_sparsity(params),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=max_nfev,
    )


def buffer_audit(label: str, x: np.ndarray, params: BufferGridParams, g_s: np.ndarray, result=None) -> dict[str, object]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_buffer(x, params)
    logR_i = buffer_inner_grid(logR_son, params)
    logR_o = outer_grid(params)  # type: ignore[arg-type]
    patch = patch_residual(logu_i, logT_i, params, g_s)
    regular = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx)
            for idx in range(1, params.n_inner - 1)
        ],
        dtype=float,
    )
    ordinary_first = _differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, 0)
    outer = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx)
            for idx in range(params.n_outer - 1)
        ],
        dtype=float,
    )
    interface = np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float)
    far = far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params)  # type: ignore[arg-type]
    sonic = sonic_diagnostics(logR_son, np.array([logu_i[0], logT_i[0]], dtype=float), lambda0, params.physics)
    residual = buffer_residual(x, params, g_s)
    combined_logR = np.concatenate([logR_i, logR_o[1:]])
    combined_logu = np.concatenate([logu_i, logu_o[1:]])
    combined_logT = np.concatenate([logT_i, logT_o[1:]])
    H_over_R = [
        algebraic_state(float(lr), float(lu), float(lt), lambda0, params.physics).H_over_R
        for lr, lu, lt in zip(combined_logR, combined_logu, combined_logT)
    ]
    blocks = {
        "patch": float(np.max(np.abs(patch))),
        "regular_R": float(np.max(np.abs(regular[:, 0]))) if len(regular) else 0.0,
        "regular_E": float(np.max(np.abs(regular[:, 1]))) if len(regular) else 0.0,
        "outer_R": float(np.max(np.abs(outer[:, 0]))),
        "outer_E": float(np.max(np.abs(outer[:, 1]))),
        "interface": float(np.max(np.abs(interface))),
        "far_omega": abs(float(far[0])),
        "far_energy": abs(float(far[1])),
        "D": abs(float(sonic.D)),
        "C1": abs(float(sonic.C1)),
        "C2": abs(float(sonic.C2)),
    }
    physical = max(blocks.values())
    dominant = max(blocks, key=blocks.get)
    return {
        "label": label,
        "ratio": params.physics.mdot_edd_ratio,
        "delta_s": params.delta_s,
        "n_regular": params.n_regular,
        "n_inner": params.n_inner,
        "n_outer": params.n_outer,
        "R_far_rg": params.R_far_rg,
        "selected_max": float(np.max(np.abs(residual))),
        "physical_active": physical,
        "dominant": dominant,
        "patch": blocks["patch"],
        "regular_R": blocks["regular_R"],
        "regular_E": blocks["regular_E"],
        "ordinary_first_R": float(ordinary_first[0]),
        "ordinary_first_E": float(ordinary_first[1]),
        "outer_R": blocks["outer_R"],
        "outer_E": blocks["outer_E"],
        "interface": blocks["interface"],
        "far_omega": float(far[0]),
        "far_energy": float(far[1]),
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "smin_over_smax": float(sonic.smin_over_smax),
        "first_dx": float(logR_i[1] - logR_i[0]),
        "second_dx": float(logR_i[2] - logR_i[1]) if len(logR_i) > 2 else np.nan,
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "lambda0": float(lambda0),
        "int_adv": integrated_advective_fraction(combined_logu, combined_logT, combined_logR, lambda0, params),  # type: ignore[arg-type]
        "max_HR": float(np.max(H_over_R)),
        "nfev": int(result.nfev) if result is not None else 0,
        "success": bool(result.success) if result is not None else True,
        "message": str(result.message) if result is not None else "seed evaluation",
        "x": np.asarray(x, dtype=float),
    }


def pchip_extrap(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.asarray(PchipInterpolator(x_old, y_old, extrapolate=True)(x_new), dtype=float)


def local_defect_split(
    x_left: float,
    y_left: np.ndarray,
    x_mid: float,
    y_interp: np.ndarray,
    x_right: float,
    y_right: np.ndarray,
    lambda0: float,
    physics,
    prior_weight: float,
) -> tuple[np.ndarray, float, bool]:
    lower = np.array([physics.logu_bounds[0], physics.logT_bounds[0]], dtype=float)
    upper = np.array([physics.logu_bounds[1], physics.logT_bounds[1]], dtype=float)
    scale = np.array([max(abs(float(y_interp[0])), 1.0), max(abs(float(y_interp[1])), 1.0)], dtype=float)

    def residual(y_mid: np.ndarray) -> np.ndarray:
        left = interval_residual_between(x_left, y_left, x_mid, y_mid, lambda0, physics)
        right = interval_residual_between(x_mid, y_mid, x_right, y_right, lambda0, physics)
        prior = prior_weight * (np.asarray(y_mid, dtype=float) - y_interp) / scale
        return np.concatenate([left, right, prior])

    result = least_squares(
        residual,
        np.clip(y_interp, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=MAX_NFEV_LOCAL_SPLIT,
    )
    return np.asarray(result.x, dtype=float), float(np.max(np.abs(residual(result.x)))), bool(result.success)


def defect_preserving_values(
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
    lambda0: float,
    physics,
    prior_weight: float = 0.02,
) -> tuple[np.ndarray, dict[str, object]]:
    x_old = np.asarray(x_old, dtype=float)
    y_old = np.asarray(y_old, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    interp_u = pchip_extrap(x_old, y_old[:, 0], x_new)
    interp_T = pchip_extrap(x_old, y_old[:, 1], x_new)
    y_new = np.column_stack([interp_u, interp_T])
    local_defects = []
    local_success = []
    copied = 0
    for idx, x_value in enumerate(x_new):
        exact = np.where(np.isclose(x_old, x_value, rtol=0.0, atol=1.0e-11))[0]
        if len(exact):
            y_new[idx] = y_old[int(exact[0])]
            copied += 1
            continue
        right = int(np.searchsorted(x_old, x_value))
        if right <= 0 or right >= len(x_old):
            continue
        left = right - 1
        y_split, defect, success = local_defect_split(
            float(x_old[left]),
            y_old[left],
            float(x_value),
            y_new[idx],
            float(x_old[right]),
            y_old[right],
            lambda0,
            physics,
            prior_weight,
        )
        y_new[idx] = y_split
        local_defects.append(defect)
        local_success.append(success)
    stats = {
        "local_splits": len(local_defects),
        "copied": copied,
        "local_defect_max": float(max(local_defects)) if local_defects else 0.0,
        "local_defect_median": float(np.median(local_defects)) if local_defects else 0.0,
        "local_success_fraction": float(np.mean(local_success)) if local_success else 1.0,
    }
    return y_new, stats


def source_to_buffer_seed(
    x_source: np.ndarray,
    source_params: TwoDomainParams,
    target_params: BufferGridParams,
    g_s: np.ndarray,
    method: str,
) -> tuple[np.ndarray, dict[str, object]]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_two_domain(x_source, source_params)
    old_logR_i = inner_grid(logR_son, source_params)
    old_logR_o = outer_grid(source_params)
    new_logR_i = buffer_inner_grid(logR_son, target_params)
    new_logR_o = outer_grid(target_params)  # type: ignore[arg-type]

    y_buffer = np.array([logu_i[0], logT_i[0]], dtype=float) + g_s * target_params.delta_s
    keep = old_logR_i > new_logR_i[1] + 1.0e-11
    old_aug_x = np.concatenate([new_logR_i[:2], old_logR_i[keep]])
    old_aug_y = np.vstack(
        [
            np.array([logu_i[0], logT_i[0]], dtype=float),
            y_buffer,
            np.column_stack([logu_i[keep], logT_i[keep]]),
        ]
    )
    order = np.argsort(old_aug_x)
    old_aug_x = old_aug_x[order]
    old_aug_y = old_aug_y[order]

    stats: dict[str, object] = {"local_splits": 0, "copied": 0, "local_defect_max": 0.0, "local_defect_median": 0.0, "local_success_fraction": 1.0}
    if method == "pchip_patch":
        new_y = np.column_stack(
            [
                pchip_extrap(old_aug_x, old_aug_y[:, 0], new_logR_i),
                pchip_extrap(old_aug_x, old_aug_y[:, 1], new_logR_i),
            ]
        )
    elif method == "defect_preserving":
        new_y, stats = defect_preserving_values(old_aug_x, old_aug_y, new_logR_i, lambda0, target_params.physics)
    else:
        raise ValueError(f"unknown remap method {method!r}")
    new_y[0] = old_aug_y[0]
    new_y[1] = y_buffer

    logu_o_new = pchip_extrap(old_logR_o, logu_o, new_logR_o)
    logT_o_new = pchip_extrap(old_logR_o, logT_o, new_logR_o)
    logu_o_new[0] = float(new_y[-1, 0])
    logT_o_new[0] = float(new_y[-1, 1])
    seed = pack_two_domain(new_y[:, 0], new_y[:, 1], logu_o_new, logT_o_new, logR_son, lambda0)
    return seed, stats


def buffer_to_buffer_seed(
    x_old: np.ndarray,
    old_params: BufferGridParams,
    new_params: BufferGridParams,
    method: str,
) -> tuple[np.ndarray, dict[str, object]]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_buffer(x_old, old_params)
    old_logR_i = buffer_inner_grid(logR_son, old_params)
    old_logR_o = outer_grid(old_params)  # type: ignore[arg-type]
    new_logR_i = buffer_inner_grid(logR_son, new_params)
    new_logR_o = outer_grid(new_params)  # type: ignore[arg-type]
    old_y = np.column_stack([logu_i, logT_i])
    stats: dict[str, object] = {"local_splits": 0, "copied": 0, "local_defect_max": 0.0, "local_defect_median": 0.0, "local_success_fraction": 1.0}
    if method == "pchip_patch":
        new_y = np.column_stack(
            [
                pchip_extrap(old_logR_i, logu_i, new_logR_i),
                pchip_extrap(old_logR_i, logT_i, new_logR_i),
            ]
        )
    elif method == "defect_preserving":
        new_y, stats = defect_preserving_values(old_logR_i, old_y, new_logR_i, lambda0, new_params.physics)
    else:
        raise ValueError(f"unknown remap method {method!r}")
    new_y[0] = old_y[0]
    new_y[1] = old_y[1]
    logu_o_new = pchip_extrap(old_logR_o, logu_o, new_logR_o)
    logT_o_new = pchip_extrap(old_logR_o, logT_o, new_logR_o)
    logu_o_new[0] = float(new_y[-1, 0])
    logT_o_new[0] = float(new_y[-1, 1])
    seed = pack_two_domain(new_y[:, 0], new_y[:, 1], logu_o_new, logT_o_new, logR_son, lambda0)
    return seed, stats


def focused_state_from_vars(anchor: np.ndarray, params: TwoDomainParams, vars_free: np.ndarray, n_first_nodes: int) -> np.ndarray:
    trial = np.asarray(anchor, dtype=float).copy()
    ni = params.n_inner
    trial[:n_first_nodes] = vars_free[:n_first_nodes]
    trial[ni : ni + n_first_nodes] = vars_free[n_first_nodes : 2 * n_first_nodes]
    trial[-2:] = vars_free[2 * n_first_nodes : 2 * n_first_nodes + 2]
    return trial


def sonic_focused_polish(anchor: np.ndarray, params: TwoDomainParams, n_first_nodes: int = 5):
    logu_i, logT_i, _logu_o, _logT_o, logR_son, lambda0 = unpack_two_domain(anchor, params)
    y0 = np.concatenate([logu_i[:n_first_nodes], logT_i[:n_first_nodes], np.array([logR_son, lambda0])])
    lower_state, upper_state = state_bounds_two_domain(params)
    lower = np.concatenate(
        [
            np.maximum(lower_state[:n_first_nodes], logu_i[:n_first_nodes] - 0.08),
            np.maximum(lower_state[params.n_inner : params.n_inner + n_first_nodes], logT_i[:n_first_nodes] - 0.08),
            np.array([np.log((np.exp(logR_son) / params.r_g - 0.04) * params.r_g), lambda0 - 8.0e-4], dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.minimum(upper_state[:n_first_nodes], logu_i[:n_first_nodes] + 0.08),
            np.minimum(upper_state[params.n_inner : params.n_inner + n_first_nodes], logT_i[:n_first_nodes] + 0.08),
            np.array([np.log((np.exp(logR_son) / params.r_g + 0.04) * params.r_g), lambda0 + 8.0e-4], dtype=float),
        ]
    )
    scale = np.maximum(1.0, np.abs(y0))

    def residual(vars_free: np.ndarray) -> np.ndarray:
        trial = focused_state_from_vars(anchor, params, vars_free, n_first_nodes)
        logu_t, logT_t, _logu_o, _logT_o, logR_t, lambda_t = unpack_two_domain(trial, params)
        logR_i = inner_grid(logR_t, params)
        rows = []
        sonic = sonic_diagnostics(logR_t, np.array([logu_t[0], logT_t[0]], dtype=float), lambda_t, params.physics)
        rows.append(np.array([sonic.D, sonic.C1, sonic.C2, sonic.compatibility], dtype=float))
        for idx in range(min(n_first_nodes - 1, 4)):
            rows.append(_differential_interval_residual_from_unpacked(logu_t, logT_t, logR_i, lambda_t, params.physics, idx))
        prior = 2.0e-3 * (vars_free - y0) / scale
        rows.append(prior)
        return np.concatenate(rows)

    result = least_squares(
        residual,
        np.clip(y0, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=MAX_NFEV_SONIC_POLISH,
    )
    polished = focused_state_from_vars(anchor, params, result.x, n_first_nodes)
    return polished, result


def row_with_reference(row: dict[str, object], ref_row: dict[str, float], ref_arrays: dict[str, np.ndarray], params) -> dict[str, object]:
    comparison = compare_to_reference(np.asarray(row["x"], dtype=float), params, ref_arrays)
    row.update(
        {
            "delta_Rson_rg": float(row["Rson_rg"] - ref_row["Rson_rg"]),
            "delta_lambda0": float(row["lambda0"] - ref_row["lambda0"]),
            "delta_int_adv": float(row["int_adv"] - ref_row["int_adv"]),
            "max_dlogu_ref": comparison["max_dlogu"],
            "rms_dlogu_ref": comparison["rms_dlogu"],
            "max_dlogT_ref": comparison["max_dlogT"],
            "rms_dlogT_ref": comparison["rms_dlogT"],
        }
    )
    return row


def buffer_row_with_reference(row: dict[str, object], ref_row: dict[str, float], ref_arrays: dict[str, np.ndarray], params: BufferGridParams) -> dict[str, object]:
    comparison = compare_buffer_to_reference(np.asarray(row["x"], dtype=float), params, ref_arrays)
    row.update(
        {
            "delta_Rson_rg": float(row["Rson_rg"] - ref_row["Rson_rg"]),
            "delta_lambda0": float(row["lambda0"] - ref_row["lambda0"]),
            "delta_int_adv": float(row["int_adv"] - ref_row["int_adv"]),
            "max_dlogu_ref": comparison["max_dlogu"],
            "rms_dlogu_ref": comparison["rms_dlogu"],
            "max_dlogT_ref": comparison["max_dlogT"],
            "rms_dlogT_ref": comparison["rms_dlogT"],
        }
    )
    return row


def buffer_profile_arrays(x: np.ndarray, params: BufferGridParams) -> dict[str, np.ndarray]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_buffer(x, params)
    logR_i = buffer_inner_grid(logR_son, params)
    logR_o = outer_grid(params)  # type: ignore[arg-type]
    logR = np.concatenate([logR_i, logR_o[1:]])
    logu = np.concatenate([logu_i, logu_o[1:]])
    logT = np.concatenate([logT_i, logT_o[1:]])
    H_over_R = []
    lnOmega = []
    for lr, lu, lt in zip(logR, logu, logT):
        state = algebraic_state(float(lr), float(lu), float(lt), lambda0, params.physics)
        H_over_R.append(float(state.H_over_R))
        lnOmega.append(float(np.log(state.Omega / state.Omega_K)))
    return {
        "logR": logR,
        "logu": logu,
        "logT": logT,
        "H_over_R": np.asarray(H_over_R, dtype=float),
        "lnOmega": np.asarray(lnOmega, dtype=float),
    }


def compare_buffer_to_reference(x: np.ndarray, params: BufferGridParams, ref_arrays: dict[str, np.ndarray]) -> dict[str, float]:
    arrays = buffer_profile_arrays(x, params)
    logR = arrays["logR"]
    mask = (logR >= ref_arrays["logR"][0]) & (logR <= ref_arrays["logR"][-1])
    if not np.any(mask):
        return {key: np.nan for key in ("max_dlogu", "rms_dlogu", "max_dlogT", "rms_dlogT")}
    x_eval = logR[mask]
    dlogu = arrays["logu"][mask] - np.interp(x_eval, ref_arrays["logR"], ref_arrays["logu"])
    dlogT = arrays["logT"][mask] - np.interp(x_eval, ref_arrays["logR"], ref_arrays["logT"])
    return {
        "max_dlogu": float(np.max(np.abs(dlogu))),
        "rms_dlogu": float(np.sqrt(np.mean(dlogu**2))),
        "max_dlogT": float(np.max(np.abs(dlogT))),
        "rms_dlogT": float(np.sqrt(np.mean(dlogT**2))),
    }


def write_sonic_polish_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Sonic-Focused N65 Polish",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_refinement_sprint.py`.",
        "",
        "| label | physical | dominant | inner R | inner E | outer R | far omega | D | C1 | C2 | K | Rson/rg | dRson | lambda0 | dlambda | int adv | dint adv | nfev | success | message |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {physical_active} | {dominant} | {inner_R} | {inner_E} | {outer_R} | {far_omega} | "
            "{D} | {C1} | {C2} | {K} | {Rson_rg} | {delta_Rson_rg} | {lambda0} | {delta_lambda0} | "
            "{int_adv} | {delta_int_adv} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                physical_active=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                inner_R=fmt(float(row["inner_R"])),
                inner_E=fmt(float(row["inner_E"])),
                outer_R=fmt(float(row["outer_R"])),
                far_omega=fmt(float(row["far_omega"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row.get("delta_Rson_rg", np.nan))),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row.get("delta_lambda0", np.nan))),
                int_adv=fmt(float(row["int_adv"])),
                delta_int_adv=fmt(float(row.get("delta_int_adv", np.nan))),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    SONIC_POLISH_TABLE.write_text("\n".join(lines) + "\n")


def write_buffer_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Sonic Buffer Refinement",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_refinement_sprint.py`.",
        "",
        "The first interval is a fixed sonic patch with residual `(y_buffer-y_s)/Delta_s - g_s`; ordinary differential collocation starts at the buffer node.",
        "",
        "| label | stage | method | delta | N regular | N inner | physical | dominant | patch | regular R | regular E | ordinary first R | ordinary first E | outer R | far omega | D | C1 | C2 | K | first dx | Rson/rg | dRson | lambda0 | dlambda | int adv | dint adv | seed local max | nfev | success | message |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {stage} | {method} | {delta_s} | {n_regular} | {n_inner} | {physical_active} | {dominant} | "
            "{patch} | {regular_R} | {regular_E} | {ordinary_first_R} | {ordinary_first_E} | {outer_R} | "
            "{far_omega} | {D} | {C1} | {C2} | {K} | {first_dx} | {Rson_rg} | {delta_Rson_rg} | "
            "{lambda0} | {delta_lambda0} | {int_adv} | {delta_int_adv} | {seed_local_defect_max} | "
            "{nfev} | {success} | {message} |".format(
                label=row["label"],
                stage=row["stage"],
                method=row["method"],
                delta_s=fmt(float(row["delta_s"])),
                n_regular=row["n_regular"],
                n_inner=row["n_inner"],
                physical_active=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                patch=fmt(float(row["patch"])),
                regular_R=fmt(float(row["regular_R"])),
                regular_E=fmt(float(row["regular_E"])),
                ordinary_first_R=fmt(float(row["ordinary_first_R"])),
                ordinary_first_E=fmt(float(row["ordinary_first_E"])),
                outer_R=fmt(float(row["outer_R"])),
                far_omega=fmt(float(row["far_omega"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                first_dx=fmt(float(row["first_dx"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row.get("delta_Rson_rg", np.nan))),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row.get("delta_lambda0", np.nan))),
                int_adv=fmt(float(row["int_adv"])),
                delta_int_adv=fmt(float(row.get("delta_int_adv", np.nan))),
                seed_local_defect_max=fmt(float(row.get("seed_local_defect_max", np.nan))),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    BUFFER_TABLE.write_text("\n".join(lines) + "\n")


def write_defect_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Defect-Preserving Seed Audit",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_refinement_sprint.py`.",
        "",
        "| label | delta | method | stage | N regular | physical | dominant | regular R | regular E | patch | D | C1 | C2 | local splits | copied | local max | local median | local success frac |",
        "|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {delta_s} | {method} | {stage} | {n_regular} | {physical_active} | {dominant} | "
            "{regular_R} | {regular_E} | {patch} | {D} | {C1} | {C2} | {local_splits} | {copied} | "
            "{local_defect_max} | {local_defect_median} | {local_success_fraction} |".format(
                label=row["label"],
                delta_s=fmt(float(row["delta_s"])),
                method=row["method"],
                stage=row["stage"],
                n_regular=row["n_regular"],
                physical_active=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                regular_R=fmt(float(row["regular_R"])),
                regular_E=fmt(float(row["regular_E"])),
                patch=fmt(float(row["patch"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                local_splits=row.get("local_splits", 0),
                copied=row.get("copied", 0),
                local_defect_max=fmt(float(row.get("local_defect_max", np.nan))),
                local_defect_median=fmt(float(row.get("local_defect_median", np.nan))),
                local_success_fraction=fmt(float(row.get("local_success_fraction", np.nan))),
            )
        )
    DEFECT_TABLE.write_text("\n".join(lines) + "\n")


def write_nested_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Nested Regular-Domain Refinement",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_refinement_sprint.py`.",
        "",
        "| label | stage | method | delta | N regular | N inner | physical | dominant | patch | regular R | regular E | outer R | far omega | D | C1 | C2 | K | Rson/rg | dRson | lambda0 | dlambda | int adv | dint adv | nfev | success | message |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {stage} | {method} | {delta_s} | {n_regular} | {n_inner} | {physical_active} | {dominant} | "
            "{patch} | {regular_R} | {regular_E} | {outer_R} | {far_omega} | {D} | {C1} | {C2} | {K} | "
            "{Rson_rg} | {delta_Rson_rg} | {lambda0} | {delta_lambda0} | {int_adv} | {delta_int_adv} | "
            "{nfev} | {success} | {message} |".format(
                label=row["label"],
                stage=row["stage"],
                method=row["method"],
                delta_s=fmt(float(row["delta_s"])),
                n_regular=row["n_regular"],
                n_inner=row["n_inner"],
                physical_active=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                patch=fmt(float(row["patch"])),
                regular_R=fmt(float(row["regular_R"])),
                regular_E=fmt(float(row["regular_E"])),
                outer_R=fmt(float(row["outer_R"])),
                far_omega=fmt(float(row["far_omega"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row.get("delta_Rson_rg", np.nan))),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row.get("delta_lambda0", np.nan))),
                int_adv=fmt(float(row["int_adv"])),
                delta_int_adv=fmt(float(row.get("delta_int_adv", np.nan))),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    NESTED_TABLE.write_text("\n".join(lines) + "\n")


def attach_stage(row: dict[str, object], label: str, stage: str, method: str, stats: dict[str, object]) -> dict[str, object]:
    row["label"] = label
    row["stage"] = stage
    row["method"] = method
    row.update({f"seed_{key}": value for key, value in stats.items()})
    row.update(stats)
    return row


def run_buffer_case(
    label: str,
    anchor_x: np.ndarray,
    anchor_params: TwoDomainParams,
    target_params: BufferGridParams,
    g_s: np.ndarray,
    method: str,
    ref_row: dict[str, float],
    ref_arrays: dict[str, np.ndarray],
    *,
    solve_case: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    seed, stats = source_to_buffer_seed(anchor_x, anchor_params, target_params, g_s, method)
    seed_row = buffer_row_with_reference(buffer_audit(label, seed, target_params, g_s), ref_row, ref_arrays, target_params)
    attach_stage(seed_row, label, "seed", method, stats)
    print(
        f"{label} {method} seed physical={seed_row['physical_active']:.3e} "
        f"dominant={seed_row['dominant']} local={stats['local_defect_max']:.3e}",
        flush=True,
    )
    if not solve_case:
        return seed_row, [seed_row]
    release = solve_buffer(seed, target_params, g_s, MAX_NFEV_BUFFER_RELEASE)
    release_row = buffer_row_with_reference(buffer_audit(label, release.x, target_params, g_s, release), ref_row, ref_arrays, target_params)
    attach_stage(release_row, label, "release", method, stats)
    print(
        f"{label} {method} release physical={release_row['physical_active']:.3e} "
        f"dominant={release_row['dominant']} nfev={release.nfev}",
        flush=True,
    )
    polish = solve_buffer(release.x, target_params, g_s, MAX_NFEV_BUFFER_POLISH)
    polish_row = buffer_row_with_reference(buffer_audit(label, polish.x, target_params, g_s, polish), ref_row, ref_arrays, target_params)
    attach_stage(polish_row, label, "polish", method, stats)
    print(
        f"{label} {method} polish physical={polish_row['physical_active']:.3e} "
        f"dominant={polish_row['dominant']} nfev={polish.nfev}",
        flush=True,
    )
    save_checkpoint(f"{label}_{method}", np.asarray(polish_row["x"], dtype=float), polish_row)
    return polish_row, [seed_row, release_row, polish_row]


def run_nested(
    start_x: np.ndarray,
    start_params: BufferGridParams,
    g_s: np.ndarray,
    ratio: float,
    fiducial: FiducialParams,
    mdot_edd: float,
    ref_row: dict[str, float],
    ref_arrays: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current_x = start_x
    current_params = start_params
    for n_regular in N_REGULAR_NESTED:
        if n_regular == current_params.n_regular:
            row = buffer_row_with_reference(buffer_audit(f"Nreg{n_regular}", current_x, current_params, g_s), ref_row, ref_arrays, current_params)
            attach_stage(row, f"Nreg{n_regular}", "loaded", "defect_preserving", {})
            rows.append(row)
            continue
        target_params = make_buffer_params(fiducial, ratio, mdot_edd, n_regular, current_params.n_outer, current_params.R_far_rg, current_params.delta_s)
        seed, stats = buffer_to_buffer_seed(current_x, current_params, target_params, "defect_preserving")
        seed_row = buffer_row_with_reference(buffer_audit(f"Nreg{n_regular}", seed, target_params, g_s), ref_row, ref_arrays, target_params)
        attach_stage(seed_row, f"Nreg{n_regular}", "seed", "defect_preserving", stats)
        rows.append(seed_row)
        print(
            f"nested Nreg{n_regular} seed physical={seed_row['physical_active']:.3e} "
            f"dominant={seed_row['dominant']} local={stats['local_defect_max']:.3e}",
            flush=True,
        )
        release = solve_buffer(seed, target_params, g_s, MAX_NFEV_BUFFER_RELEASE)
        release_row = buffer_row_with_reference(buffer_audit(f"Nreg{n_regular}", release.x, target_params, g_s, release), ref_row, ref_arrays, target_params)
        attach_stage(release_row, f"Nreg{n_regular}", "release", "defect_preserving", stats)
        rows.append(release_row)
        print(
            f"nested Nreg{n_regular} release physical={release_row['physical_active']:.3e} "
            f"dominant={release_row['dominant']} nfev={release.nfev}",
            flush=True,
        )
        polish = solve_buffer(release.x, target_params, g_s, MAX_NFEV_BUFFER_POLISH)
        polish_row = buffer_row_with_reference(buffer_audit(f"Nreg{n_regular}", polish.x, target_params, g_s, polish), ref_row, ref_arrays, target_params)
        attach_stage(polish_row, f"Nreg{n_regular}", "polish", "defect_preserving", stats)
        rows.append(polish_row)
        print(
            f"nested Nreg{n_regular} polish physical={polish_row['physical_active']:.3e} "
            f"dominant={polish_row['dominant']} nfev={polish.nfev}",
            flush=True,
        )
        save_checkpoint(f"nested_Nreg{n_regular}", np.asarray(polish_row["x"], dtype=float), polish_row)
        current_x = np.asarray(polish_row["x"], dtype=float)
        current_params = target_params
        if float(polish_row["physical_active"]) > NESTED_GATE_LIMIT:
            break
    return rows


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_x, source_meta = load_checkpoint(SOURCE_CHECKPOINT)
    ratio = float(source_meta["ratio"])
    source_params = make_params(
        fiducial,
        ratio,
        mdot_edd,
        int(source_meta["n_inner"]),
        int(source_meta["n_outer"]),
        float(source_meta["R_far_rg"]),
    )
    ref_arrays = combined_profile_arrays(source_x, source_params)
    source_row = audit_row("source", source_x, source_params)
    ref_row = {
        "Rson_rg": float(source_row["Rson_rg"]),
        "lambda0": float(source_row["lambda0"]),
        "int_adv": float(source_row["int_adv"]),
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    source_row = row_with_reference(source_row, ref_row, ref_arrays, source_params)
    source_row["stage"] = "loaded"
    source_row["method"] = "baseline"
    sonic_rows = [source_row]
    polished_x, polish_result = sonic_focused_polish(source_x, source_params)
    polished_row = audit_row("sonic_focused", polished_x, source_params, polish_result)
    polished_row = row_with_reference(polished_row, ref_row, ref_arrays, source_params)
    polished_row["stage"] = "focused"
    polished_row["method"] = "first_nodes"
    sonic_rows.append(polished_row)
    write_sonic_polish_table(sonic_rows)
    print(
        f"sonic focus source={source_row['physical_active']:.3e}/{source_row['dominant']} "
        f"polished={polished_row['physical_active']:.3e}/{polished_row['dominant']} nfev={polish_result.nfev}",
        flush=True,
    )
    save_checkpoint("sonic_focused", polished_x, polished_row)

    source_sonic_max = max(abs(float(source_row["D"])), abs(float(source_row["C1"])), abs(float(source_row["C2"])), abs(float(source_row["K"])))
    polished_sonic_max = max(abs(float(polished_row["D"])), abs(float(polished_row["C1"])), abs(float(polished_row["C2"])), abs(float(polished_row["K"])))
    if float(polished_row["physical_active"]) <= 1.5 * float(source_row["physical_active"]) and polished_sonic_max < source_sonic_max:
        anchor_x = polished_x
        anchor_label = "sonic_focused"
    else:
        anchor_x = source_x
        anchor_label = "source"
    print(f"using {anchor_label} as buffer anchor", flush=True)

    g_s = source_first_slope(anchor_x, source_params)
    buffer_rows: list[dict[str, object]] = []
    defect_rows: list[dict[str, object]] = []
    finals: list[tuple[dict[str, object], BufferGridParams]] = []
    for delta_s in DELTAS:
        target_params = make_buffer_params(
            fiducial,
            ratio,
            mdot_edd,
            N_REGULAR_BUFFER,
            int(source_meta["n_outer"]),
            float(source_meta["R_far_rg"]),
            delta_s,
        )
        for method in ("pchip_patch", "defect_preserving"):
            label = f"delta{str(delta_s).replace('.', 'p')}_Nreg{N_REGULAR_BUFFER}"
            final, stages = run_buffer_case(
                label,
                anchor_x,
                source_params,
                target_params,
                g_s,
                method,
                ref_row,
                ref_arrays,
                solve_case=(method == "defect_preserving"),
            )
            buffer_rows.extend(stages)
            defect_rows.append(stages[0])
            if method == "defect_preserving":
                finals.append((final, target_params))
            write_buffer_table(buffer_rows)
            write_defect_table(defect_rows)

    best_final, best_params = min(finals, key=lambda item: float(item[0]["physical_active"]))
    nested_rows: list[dict[str, object]] = []
    if (
        str(best_final.get("method")) == "defect_preserving"
        and float(best_final["physical_active"]) <= BUFFER_GATE_LIMIT
    ) or float(best_final["physical_active"]) <= SCIENCE_LIMIT:
        print(
            f"buffer gate passed with {best_final['label']} {best_final['method']} "
            f"physical={best_final['physical_active']:.3e}; running nested refinement",
            flush=True,
        )
        nested_rows = run_nested(
            np.asarray(best_final["x"], dtype=float),
            best_params,
            g_s,
            ratio,
            fiducial,
            mdot_edd,
            ref_row,
            ref_arrays,
        )
    else:
        print(
            f"buffer gate not passed: best={best_final['label']} {best_final['method']} "
            f"physical={best_final['physical_active']:.3e}",
            flush=True,
        )
    write_buffer_table(buffer_rows)
    write_defect_table(defect_rows)
    write_nested_table(nested_rows)
    print(f"wrote {SONIC_POLISH_TABLE}")
    print(f"wrote {BUFFER_TABLE}")
    print(f"wrote {DEFECT_TABLE}")
    print(f"wrote {NESTED_TABLE}")


if __name__ == "__main__":
    main()
