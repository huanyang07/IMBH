"""Two-domain sonic Taylor patch experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.sparse import lil_matrix

from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    algebraic_state,
    local_scaled_residual,
    scaled_differential_matrix,
    sonic_directional_B,
    sonic_frozen_scaled_directional_B,
    sonic_diagnostics,
    sonic_lhopital_residual,
    sonic_lhopital_residual_form,
    sonic_null_vectors,
    sonic_unscaled_directional_B,
    sonic_unscaled_null_vectors,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_dynamic_sonic_patch import (
    CHECKPOINT_DIR as DYNAMIC_CHECKPOINT_DIR,
    FIXED_BUFFER_CHECKPOINT,
    load_row,
)
from run_transonic_two_domain_mesh_validation import combined_profile_arrays, load_checkpoint, make_params
from run_transonic_two_domain_outer_extension import (
    far_boundary_residual,
    integrated_advective_fraction,
    outer_grid,
    pack_two_domain,
    state_bounds_two_domain,
    unpack_two_domain,
)
from run_transonic_two_domain_sonic_refinement_sprint import (
    SOURCE_CHECKPOINT,
    BufferGridParams,
    buffer_inner_grid,
    buffer_to_buffer_seed,
    make_buffer_params,
    row_json,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_two_domain_sonic_taylor_patch.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_two_domain_sonic_taylor_patch"
DYNAMIC_N64_CHECKPOINT = DYNAMIC_CHECKPOINT_DIR / "Nreg64_0p90277664.npz"
DEFAULT_N_SEQUENCE = (64,)
MAX_NFEV_RELEASE = int(os.environ.get("IMBH_TAYLOR_MAX_NFEV_RELEASE", "900"))
MAX_NFEV_POLISH = int(os.environ.get("IMBH_TAYLOR_MAX_NFEV_POLISH", "450"))
SCIENCE_LIMIT = 5.0e-6
STOP_LIMIT = 2.0e-3
LHOPITAL_EPS = float(os.environ.get("IMBH_TAYLOR_LHOPITAL_EPS", "1e-5"))
LHOPITAL_FORM = os.environ.get("IMBH_TAYLOR_LHOPITAL_FORM", "scaled")
COMPAT_WEIGHT = float(os.environ.get("IMBH_TAYLOR_COMPAT_WEIGHT", "0.2"))
TAYLOR_SCALE = float(os.environ.get("IMBH_TAYLOR_RELATION_SCALE", "1e-2"))
G_BOUND = float(os.environ.get("IMBH_TAYLOR_G_BOUND", "500"))
H_BOUND = float(os.environ.get("IMBH_TAYLOR_H_BOUND", "100000"))
SEED_MODE = os.environ.get("IMBH_TAYLOR_SEED", "buffer")
ROOT_SCAN_HALF_WIDTH = float(os.environ.get("IMBH_TAYLOR_ROOT_HALF_WIDTH", "1000"))
ROOT_SCAN_POINTS = int(os.environ.get("IMBH_TAYLOR_ROOT_POINTS", "2001"))


def parse_n_sequence() -> tuple[int, ...]:
    raw = os.environ.get("IMBH_TAYLOR_N_SEQUENCE")
    if raw is None or not raw.strip():
        return DEFAULT_N_SEQUENCE
    return tuple(int(piece) for piece in raw.replace(":", ",").split(",") if piece.strip())


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def taylor_row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items() if key != "x"}, sort_keys=True)


def save_checkpoint(label: str, x: np.ndarray, row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in row.items() if key != "x"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{label}_0p90277664.npz",
        x=np.asarray(x, dtype=float),
        row_json=np.array(row_json({key: json_safe(value) for key, value in payload.items()})),
    )


def base_size(params: BufferGridParams) -> int:
    return 2 * params.n_inner + 2 * params.n_outer + 2


def extra_size(mode: str) -> int:
    if mode == "taylor1":
        return 2
    if mode == "taylor2":
        return 4
    raise ValueError(f"unknown Taylor mode {mode!r}")


def patch_row_count(mode: str) -> int:
    regularity_rows = 3 if LHOPITAL_FORM == "soft_compat" else 1
    midpoint_rows = 2 if mode == "taylor2" else 0
    return 1 + 2 + regularity_rows + 2 + midpoint_rows


def pack_taylor(base: np.ndarray, g_s: np.ndarray, h_s: np.ndarray | None, mode: str) -> np.ndarray:
    if mode == "taylor1":
        return np.concatenate([np.asarray(base, dtype=float), np.asarray(g_s, dtype=float)])
    if h_s is None:
        h_s = np.zeros(2, dtype=float)
    return np.concatenate([np.asarray(base, dtype=float), np.asarray(g_s, dtype=float), np.asarray(h_s, dtype=float)])


def unpack_taylor(x: np.ndarray, params: BufferGridParams, mode: str):
    size = base_size(params)
    base = np.asarray(x[:size], dtype=float)
    g_s = np.asarray(x[size : size + 2], dtype=float)
    if mode == "taylor2":
        h_s = np.asarray(x[size + 2 : size + 4], dtype=float)
    else:
        h_s = np.zeros(2, dtype=float)
    return base, g_s, h_s


def initial_taylor_state(base: np.ndarray, params: BufferGridParams, mode: str) -> np.ndarray:
    logu_i, logT_i, _logu_o, _logT_o, logR_son, _lambda0 = unpack_two_domain(base, params)  # type: ignore[arg-type]
    logR_i = buffer_inner_grid(logR_son, params)
    dx = float(logR_i[1] - logR_i[0])
    g_s = np.array([(float(logu_i[1]) - float(logu_i[0])) / dx, (float(logT_i[1]) - float(logT_i[0])) / dx], dtype=float)
    if SEED_MODE == "root":
        g_s = lhopital_root_seed(base, params, g_s)
    if mode == "taylor2":
        y_s = np.array([float(logu_i[0]), float(logT_i[0])], dtype=float)
        y_b = np.array([float(logu_i[1]), float(logT_i[1])], dtype=float)
        h_s = 2.0 * (y_b - y_s - params.delta_s * g_s) / params.delta_s**2
        h_s = np.clip(h_s, -0.8 * H_BOUND, 0.8 * H_BOUND)
    else:
        h_s = np.zeros(2, dtype=float)
    return pack_taylor(base, g_s, h_s, mode)


def raw_lhopital_value(logR: float, y: np.ndarray, g: np.ndarray, lambda0: float, params: BufferGridParams) -> float:
    form = "scaled" if LHOPITAL_FORM == "soft_compat" else LHOPITAL_FORM
    if form == "raw":
        nulls = sonic_unscaled_null_vectors(logR, y, lambda0, params.physics)
        B = sonic_unscaled_directional_B(logR, y, g, lambda0, params.physics, eps=LHOPITAL_EPS)
    else:
        nulls = sonic_null_vectors(logR, y, lambda0, params.physics)
        if form == "scaled":
            B = sonic_directional_B(logR, y, g, lambda0, params.physics, eps=LHOPITAL_EPS)
        elif form == "frozen_scaled":
            B = sonic_frozen_scaled_directional_B(logR, y, g, lambda0, params.physics, eps=LHOPITAL_EPS)
        else:
            raise ValueError(f"unknown L'Hopital form {LHOPITAL_FORM!r}")
    return float(np.dot(nulls.left_null, B))


def lhopital_root_seed(base: np.ndarray, params: BufferGridParams, g_ref: np.ndarray) -> np.ndarray:
    logu_i, logT_i, _logu_o, _logT_o, logR_son, lambda0 = unpack_two_domain(base, params)  # type: ignore[arg-type]
    y_s = np.array([float(logu_i[0]), float(logT_i[0])], dtype=float)
    if LHOPITAL_FORM == "raw":
        nulls = sonic_unscaled_null_vectors(logR_son, y_s, lambda0, params.physics)
    else:
        nulls = sonic_null_vectors(logR_son, y_s, lambda0, params.physics)
    g_p = np.linalg.lstsq(nulls.matrix, -nulls.rhs, rcond=None)[0]
    r = nulls.right_null / (np.linalg.norm(nulls.right_null) + 1.0e-300)
    a_ref = float(np.dot(g_ref - g_p, r))
    a_values = np.linspace(a_ref - ROOT_SCAN_HALF_WIDTH, a_ref + ROOT_SCAN_HALF_WIDTH, ROOT_SCAN_POINTS)

    def raw_from_a(a: float) -> float:
        return raw_lhopital_value(logR_son, y_s, g_p + float(a) * r, lambda0, params)

    raw_values = np.asarray([raw_from_a(float(a)) for a in a_values], dtype=float)
    roots: list[np.ndarray] = []
    for idx in range(len(a_values) - 1):
        left_f = float(raw_values[idx])
        right_f = float(raw_values[idx + 1])
        if not np.isfinite(left_f) or not np.isfinite(right_f) or left_f * right_f > 0.0:
            continue
        try:
            root_a = float(brentq(raw_from_a, float(a_values[idx]), float(a_values[idx + 1]), xtol=1.0e-10, rtol=1.0e-10, maxiter=80))
        except ValueError:
            continue
        roots.append(g_p + root_a * r)
    if not roots:
        return g_ref
    return min(roots, key=lambda g: float(np.linalg.norm(g - g_ref)))


def taylor_patch_residual(logu_i: np.ndarray, logT_i: np.ndarray, logR_son: float, lambda0: float, g_s: np.ndarray, h_s: np.ndarray, params: BufferGridParams, mode: str) -> np.ndarray:
    y_s = np.array([float(logu_i[0]), float(logT_i[0])], dtype=float)
    y_b = np.array([float(logu_i[1]), float(logT_i[1])], dtype=float)
    matrix, rhs, _radial_scale, _energy_scale = scaled_differential_matrix(logR_son, y_s, lambda0, params.physics)
    sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
    rows = [np.array([sonic.D], dtype=float), matrix @ g_s + rhs]
    if LHOPITAL_FORM == "soft_compat":
        rows.append(COMPAT_WEIGHT * np.array([sonic.C1, sonic.C2, sonic.compatibility], dtype=float))
    else:
        rows.append(np.array([sonic_lhopital_residual_form(logR_son, y_s, g_s, lambda0, params.physics, eps=LHOPITAL_EPS, form=LHOPITAL_FORM)], dtype=float))
    if mode == "taylor1":
        rows.append((y_b - y_s - params.delta_s * g_s) / TAYLOR_SCALE)
    else:
        rows.append((y_b - y_s - params.delta_s * g_s - 0.5 * params.delta_s**2 * h_s) / TAYLOR_SCALE)
        x_mid = logR_son + 0.5 * params.delta_s
        y_mid = y_s + 0.5 * params.delta_s * g_s + 0.125 * params.delta_s**2 * h_s
        g_mid = g_s + 0.5 * params.delta_s * h_s
        rows.append(local_scaled_residual(x_mid, y_mid, g_mid, lambda0, params.physics))
    return np.concatenate(rows)


def taylor_residual(x: np.ndarray, params: BufferGridParams, mode: str) -> np.ndarray:
    rows = []
    try:
        base, g_s, h_s = unpack_taylor(x, params, mode)
        logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_two_domain(base, params)  # type: ignore[arg-type]
        logR_i = buffer_inner_grid(logR_son, params)
        logR_o = outer_grid(params)  # type: ignore[arg-type]
        if np.any(np.diff(logR_i) <= 0.0) or np.any(np.diff(logR_o) <= 0.0):
            raise ValueError("mapped radii must increase")

        rows.append(taylor_patch_residual(logu_i, logT_i, logR_son, lambda0, g_s, h_s, params, mode))
        for idx in range(1, params.n_inner - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx))
        for idx in range(params.n_outer - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx))
        rows.append(np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float))
        rows.append(far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params))  # type: ignore[arg-type]
        return np.concatenate(rows)
    except Exception:
        return np.full(2 * params.n_inner + 2 * params.n_outer + patch_row_count(mode) - 2, 1.0e6)


def taylor_sparsity(params: BufferGridParams, mode: str):
    ni = params.n_inner
    no = params.n_outer
    base_cols = base_size(params)
    n_unknown = base_cols + extra_size(mode)
    n_rows = 2 * params.n_inner + 2 * params.n_outer + patch_row_count(mode) - 2
    pattern = lil_matrix((n_rows, n_unknown), dtype=int)
    iu = 0
    iT = ni
    ou = 2 * ni
    oT = 2 * ni + no
    logR_col = 2 * ni + 2 * no
    lambda_col = logR_col + 1
    g_cols = (base_cols, base_cols + 1)
    h_cols = (base_cols + 2, base_cols + 3)
    patch_rows = patch_row_count(mode)
    row = 0

    patch_cols = [iu, iu + 1, iT, iT + 1, logR_col, lambda_col, *g_cols]
    if mode == "taylor2":
        patch_cols.extend(h_cols)
    for col in patch_cols:
        pattern[row : row + patch_rows, col] = 1
    row += patch_rows

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
    return pattern.tocsr()


def taylor_bounds(params: BufferGridParams, mode: str) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = state_bounds_two_domain(params)  # type: ignore[arg-type]
    lower = np.concatenate([lower, np.full(extra_size(mode), -G_BOUND)])
    upper = np.concatenate([upper, np.full(extra_size(mode), G_BOUND)])
    if mode == "taylor2":
        lower[-2:] = -H_BOUND
        upper[-2:] = H_BOUND
    return lower, upper


def solve_taylor(seed: np.ndarray, params: BufferGridParams, mode: str, max_nfev: int):
    lower, upper = taylor_bounds(params, mode)
    x0 = np.clip(np.asarray(seed, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: taylor_residual(trial, params, mode),
        x0,
        jac_sparsity=taylor_sparsity(params, mode),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=max_nfev,
    )


def taylor_audit(label: str, x: np.ndarray, params: BufferGridParams, mode: str, result=None) -> dict[str, object]:
    base, g_s, h_s = unpack_taylor(x, params, mode)
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_two_domain(base, params)  # type: ignore[arg-type]
    logR_i = buffer_inner_grid(logR_son, params)
    logR_o = outer_grid(params)  # type: ignore[arg-type]
    y_s = np.array([logu_i[0], logT_i[0]], dtype=float)
    y_b = np.array([logu_i[1], logT_i[1]], dtype=float)
    matrix, rhs, _radial_scale, _energy_scale = scaled_differential_matrix(logR_son, y_s, lambda0, params.physics)
    sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
    lhop_form = "scaled" if LHOPITAL_FORM == "soft_compat" else LHOPITAL_FORM
    lhop = sonic_lhopital_residual_form(logR_son, y_s, g_s, lambda0, params.physics, eps=LHOPITAL_EPS, form=lhop_form)
    if mode == "taylor1":
        taylor_relation = (y_b - y_s - params.delta_s * g_s) / TAYLOR_SCALE
        midpoint = np.zeros(2, dtype=float)
    else:
        taylor_relation = (y_b - y_s - params.delta_s * g_s - 0.5 * params.delta_s**2 * h_s) / TAYLOR_SCALE
        midpoint = local_scaled_residual(
            logR_son + 0.5 * params.delta_s,
            y_s + 0.5 * params.delta_s * g_s + 0.125 * params.delta_s**2 * h_s,
            g_s + 0.5 * params.delta_s * h_s,
            lambda0,
            params.physics,
        )
    regular = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx)
            for idx in range(1, params.n_inner - 1)
        ],
        dtype=float,
    )
    outer = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx)
            for idx in range(params.n_outer - 1)
        ],
        dtype=float,
    )
    interface = np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float)
    far = far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params)  # type: ignore[arg-type]
    combined_logR = np.concatenate([logR_i, logR_o[1:]])
    combined_logu = np.concatenate([logu_i, logu_o[1:]])
    combined_logT = np.concatenate([logT_i, logT_o[1:]])
    H_over_R = [
        algebraic_state(float(lr), float(lu), float(lt), lambda0, params.physics).H_over_R
        for lr, lu, lt in zip(combined_logR, combined_logu, combined_logT)
    ]
    blocks = {
        "D": abs(float(sonic.D)),
        "local_F": float(np.max(np.abs(matrix @ g_s + rhs))),
        "L": abs(float(lhop)),
        "taylor": float(np.max(np.abs(taylor_relation))),
        "midpoint": float(np.max(np.abs(midpoint))),
        "regular_R": float(np.max(np.abs(regular[:, 0]))) if len(regular) else 0.0,
        "regular_E": float(np.max(np.abs(regular[:, 1]))) if len(regular) else 0.0,
        "outer_R": float(np.max(np.abs(outer[:, 0]))),
        "outer_E": float(np.max(np.abs(outer[:, 1]))),
        "interface": float(np.max(np.abs(interface))),
        "far_omega": abs(float(far[0])),
        "far_energy": abs(float(far[1])),
        "C1": abs(float(sonic.C1)),
        "C2": abs(float(sonic.C2)),
        "K": abs(float(sonic.compatibility)),
    }
    active_blocks = dict(blocks)
    if LHOPITAL_FORM == "soft_compat":
        active_blocks.pop("L")
    physical = max(active_blocks.values())
    dominant = max(active_blocks, key=active_blocks.get)
    return {
        "label": label,
        "mode": mode,
        "lhopital_form": LHOPITAL_FORM,
        "compat_weight": COMPAT_WEIGHT,
        "ratio": params.physics.mdot_edd_ratio,
        "delta_s": params.delta_s,
        "n_regular": params.n_regular,
        "n_inner": params.n_inner,
        "n_outer": params.n_outer,
        "R_far_rg": params.R_far_rg,
        "selected_max": float(np.max(np.abs(taylor_residual(x, params, mode)))),
        "physical_active": physical,
        "passes_science": bool(physical <= SCIENCE_LIMIT),
        "dominant": dominant,
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "local_F": blocks["local_F"],
        "L": float(lhop),
        "taylor": blocks["taylor"],
        "midpoint": blocks["midpoint"],
        "regular_R": blocks["regular_R"],
        "regular_E": blocks["regular_E"],
        "outer_R": blocks["outer_R"],
        "outer_E": blocks["outer_E"],
        "interface": blocks["interface"],
        "far_omega": float(far[0]),
        "far_energy": float(far[1]),
        "smin_over_smax": float(sonic.smin_over_smax),
        "g_u": float(g_s[0]),
        "g_T": float(g_s[1]),
        "h_u": float(h_s[0]),
        "h_T": float(h_s[1]),
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "lambda0": float(lambda0),
        "int_adv": integrated_advective_fraction(combined_logu, combined_logT, combined_logR, lambda0, params),  # type: ignore[arg-type]
        "max_HR": float(np.max(H_over_R)),
        "nfev": int(result.nfev) if result is not None else 0,
        "success": bool(result.success) if result is not None else True,
        "message": str(result.message) if result is not None else "seed evaluation",
        "x": np.asarray(x, dtype=float),
    }


def add_reference(row: dict[str, object], ref_row: dict[str, float]) -> dict[str, object]:
    row["delta_Rson_rg"] = float(row["Rson_rg"] - ref_row["Rson_rg"])
    row["delta_lambda0"] = float(row["lambda0"] - ref_row["lambda0"])
    row["delta_int_adv"] = float(row["int_adv"] - ref_row["int_adv"])
    return row


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Sonic Taylor Patch",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_taylor_patch.py`.",
        "",
        "The sonic derivative is an independent unknown. `taylor1` solves `D`, `A_s g_s+c_s`, L'Hopital regularity, and the first-order Taylor buffer relation. `taylor2` adds curvature and a midpoint local residual.",
        "",
        "| label | stage | mode | L form | N regular | delta | physical | selected | pass | dominant | D | C1 | C2 | K | local F | L | taylor | midpoint | regular R | regular E | far omega | Rson/rg | dRson | lambda0 | dlambda | int adv | dint adv | g_u | g_T | h_u | h_T | nfev | success | message |",
        "|---|---|---|---|---:|---:|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {stage} | {mode} | {lhopital_form} | {n_regular} | {delta_s} | {physical_active} | {selected_max} | {passes_science} | "
            "{dominant} | {D} | {C1} | {C2} | {K} | {local_F} | {L} | {taylor} | {midpoint} | {regular_R} | "
            "{regular_E} | {far_omega} | {Rson_rg} | {delta_Rson_rg} | {lambda0} | {delta_lambda0} | {int_adv} | "
            "{delta_int_adv} | {g_u} | {g_T} | {h_u} | {h_T} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                stage=row["stage"],
                mode=row["mode"],
                lhopital_form=row["lhopital_form"],
                n_regular=row["n_regular"],
                delta_s=fmt(float(row["delta_s"])),
                physical_active=fmt(float(row["physical_active"])),
                selected_max=fmt(float(row["selected_max"])),
                passes_science="yes" if row["passes_science"] else "no",
                dominant=row["dominant"],
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                local_F=fmt(float(row["local_F"])),
                L=fmt(float(row["L"])),
                taylor=fmt(float(row["taylor"])),
                midpoint=fmt(float(row["midpoint"])),
                regular_R=fmt(float(row["regular_R"])),
                regular_E=fmt(float(row["regular_E"])),
                far_omega=fmt(float(row["far_omega"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row["delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row["delta_lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                delta_int_adv=fmt(float(row["delta_int_adv"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                h_u=fmt(float(row["h_u"])),
                h_T=fmt(float(row["h_T"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def load_buffer_source(fiducial: FiducialParams, ratio: float, mdot_edd: float) -> tuple[np.ndarray, BufferGridParams, str]:
    path = DYNAMIC_N64_CHECKPOINT if DYNAMIC_N64_CHECKPOINT.exists() else FIXED_BUFFER_CHECKPOINT
    source_x, source_meta = load_row(path)
    params = make_buffer_params(
        fiducial,
        ratio,
        mdot_edd,
        int(source_meta["n_regular"]),
        int(source_meta["n_outer"]),
        float(source_meta["R_far_rg"]),
        float(os.environ.get("IMBH_TAYLOR_DELTA_S", str(source_meta["delta_s"]))),
    )
    return source_x, params, path.name


def load_taylor_checkpoint(path: Path, fiducial: FiducialParams, ratio: float, mdot_edd: float) -> tuple[np.ndarray, BufferGridParams, str]:
    x, meta = load_row(path)
    params = make_buffer_params(
        fiducial,
        ratio,
        mdot_edd,
        int(meta["n_regular"]),
        int(meta["n_outer"]),
        float(meta["R_far_rg"]),
        float(meta["delta_s"]),
    )
    return x, params, path.name


def main() -> None:
    mode = os.environ.get("IMBH_TAYLOR_MODE", "taylor1")
    n_sequence = parse_n_sequence()
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
    _ref_arrays = combined_profile_arrays(source_x, source_params)
    ref_row = {
        "Rson_rg": float(source_meta["Rson_rg"]),
        "lambda0": float(source_meta["lambda0"]),
        "int_adv": float(source_meta["int_adv"]),
    }
    resume_path = os.environ.get("IMBH_TAYLOR_START_CHECKPOINT")
    if resume_path:
        current_x, current_params, source_name = load_taylor_checkpoint(Path(resume_path), fiducial, ratio, mdot_edd)
    else:
        base_x, current_params, source_name = load_buffer_source(fiducial, ratio, mdot_edd)
        current_x = initial_taylor_state(base_x, current_params, mode)
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"source={source_name} mode={mode} lhopital_form={LHOPITAL_FORM} delta_s={current_params.delta_s:g} sequence={n_sequence}", flush=True)

    for n_regular in n_sequence:
        if n_regular == current_params.n_regular:
            seed = current_x
            params = current_params
        else:
            params = make_buffer_params(fiducial, ratio, mdot_edd, n_regular, current_params.n_outer, current_params.R_far_rg, current_params.delta_s)
            old_base, old_g, old_h = unpack_taylor(current_x, current_params, mode)
            new_base, _stats = buffer_to_buffer_seed(old_base, current_params, params, "defect_preserving")
            seed = pack_taylor(new_base, old_g, old_h, mode)
        seed_row = add_reference(taylor_audit(f"Nreg{n_regular}", seed, params, mode), ref_row)
        seed_row["stage"] = "seed"
        rows.append(seed_row)
        print(
            f"Nreg{n_regular} {mode} seed physical={seed_row['physical_active']:.3e} "
            f"dominant={seed_row['dominant']} g=({seed_row['g_u']:.3g},{seed_row['g_T']:.3g})",
            flush=True,
        )

        release = solve_taylor(seed, params, mode, MAX_NFEV_RELEASE)
        release_row = add_reference(taylor_audit(f"Nreg{n_regular}", release.x, params, mode, release), ref_row)
        release_row["stage"] = "release"
        rows.append(release_row)
        write_table(rows)
        print(
            f"Nreg{n_regular} {mode} release physical={release_row['physical_active']:.3e} "
            f"dominant={release_row['dominant']} nfev={release.nfev}",
            flush=True,
        )

        polish = solve_taylor(release.x, params, mode, MAX_NFEV_POLISH)
        polish_row = add_reference(taylor_audit(f"Nreg{n_regular}", polish.x, params, mode, polish), ref_row)
        polish_row["stage"] = "polish"
        rows.append(polish_row)
        write_table(rows)
        print(
            f"Nreg{n_regular} {mode} polish physical={polish_row['physical_active']:.3e} "
            f"dominant={polish_row['dominant']} nfev={polish.nfev}",
            flush=True,
        )
        save_checkpoint(f"{mode}_Nreg{n_regular}", np.asarray(polish_row["x"], dtype=float), polish_row)
        current_x = np.asarray(polish_row["x"], dtype=float)
        current_params = params
        if float(polish_row["physical_active"]) > STOP_LIMIT:
            print(
                f"stopping after Nreg{n_regular}: physical={polish_row['physical_active']:.3e} exceeds {STOP_LIMIT:.1e}",
                flush=True,
            )
            break

    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
