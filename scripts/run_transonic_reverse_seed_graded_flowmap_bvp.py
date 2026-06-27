"""Reverse-fitted offset flow-map BVP with a graded near-sonic grid."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _differential_interval_residual_from_unpacked,
    _integrated_interval_residual_from_unpacked,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    algebraic_state,
    local_ode_rhs,
    sonic_derivative_branches,
    sonic_diagnostics,
    sonic_lhopital_residual_form,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_sonic_reverse_fit import SOURCE_CHECKPOINT, load_context, solve_fit
from run_transonic_two_domain_dynamic_sonic_patch import load_row
from run_transonic_two_domain_sonic_refinement_sprint import buffer_inner_grid, unpack_buffer
from run_transonic_two_domain_sonic_flowmap import (
    CHECKPOINT_DIR,
    LHOPITAL_EPS,
    branch_from_a,
    branch_gradient_from_a,
    compatibility_value,
    integrate_flow_map_nodes,
    make_flowmap_params,
    outer_grid,
    pack_flowmap,
    pack_smooth_flowmap,
    pchip_extrap,
    save_checkpoint,
    smooth_flowmap_bounds,
    smooth_flowmap_size,
    smooth_flowmap_sparsity,
    unpack_smooth_flowmap,
)
from run_transonic_two_domain_outer_extension import far_boundary_residual, integrated_advective_fraction


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get("IMBH_GRADED_FLOWMAP_TABLE", "outputs/tables/transonic_reverse_seed_graded_flowmap_bvp.md")

EPSILON0 = float(os.environ.get("IMBH_GRADED_FLOWMAP_EPSILON0", "1e-6"))
EPSILON_BUFS = tuple(float(piece) for piece in os.environ.get("IMBH_GRADED_FLOWMAP_EPS_BUFS", "0.001,0.002,0.005").replace(":", ",").split(",") if piece.strip())
BRANCH_SEQUENCE = tuple(int(piece) for piece in os.environ.get("IMBH_GRADED_FLOWMAP_BRANCHES", "0,1").replace(":", ",").split(",") if piece.strip())
N_SEQUENCE = tuple(int(piece) for piece in os.environ.get("IMBH_GRADED_FLOWMAP_N_SEQUENCE", "48").replace(":", ",").split(",") if piece.strip())
SEED_MODES = tuple(piece.strip() for piece in os.environ.get("IMBH_GRADED_FLOWMAP_SEED_MODES", "micro_ode_blend").replace(":", ",").split(",") if piece.strip())
MAX_NFEV_RELEASE = int(os.environ.get("IMBH_GRADED_FLOWMAP_MAX_NFEV_RELEASE", "160"))
MAX_NFEV_POLISH = int(os.environ.get("IMBH_GRADED_FLOWMAP_MAX_NFEV_POLISH", "80"))
BRANCH_HALF_WIDTH = float(os.environ.get("IMBH_GRADED_FLOWMAP_BRANCH_HALF_WIDTH", "1000"))
BRANCH_POINTS = int(os.environ.get("IMBH_GRADED_FLOWMAP_BRANCH_POINTS", "2001"))
MICRO_FIRST_DX = float(os.environ.get("IMBH_GRADED_FLOWMAP_MICRO_FIRST_DX", "0.001"))
MICRO_GROWTH = float(os.environ.get("IMBH_GRADED_FLOWMAP_MICRO_GROWTH", "1.2"))
MICRO_EXTENT = float(os.environ.get("IMBH_GRADED_FLOWMAP_MICRO_EXTENT", "0.08"))
TRANSITION_EXTENT = float(os.environ.get("IMBH_GRADED_FLOWMAP_TRANSITION_EXTENT", "0.0"))
TRANSITION_FIRST_DX = float(os.environ.get("IMBH_GRADED_FLOWMAP_TRANSITION_FIRST_DX", "0.001"))
TRANSITION_GROWTH = float(os.environ.get("IMBH_GRADED_FLOWMAP_TRANSITION_GROWTH", "1.2"))
ODE_SEED_EXTENT = float(os.environ.get("IMBH_GRADED_FLOWMAP_ODE_SEED_EXTENT", "0.0"))
BLEND_END_EXTENT = float(os.environ.get("IMBH_GRADED_FLOWMAP_BLEND_END_EXTENT", "0.0"))
MICRO_SOLVE_FORM = os.environ.get("IMBH_GRADED_FLOWMAP_MICRO_SOLVE_FORM", "integrated").strip().lower()
FLOW_SOLVE_WEIGHT = float(os.environ.get("IMBH_GRADED_FLOWMAP_FLOW_WEIGHT", "1.0"))
MICRO_SOLVE_WEIGHT = float(os.environ.get("IMBH_GRADED_FLOWMAP_MICRO_WEIGHT", "1.0"))
TRANSITION_SOLVE_WEIGHT = float(os.environ.get("IMBH_GRADED_FLOWMAP_TRANSITION_WEIGHT", "1.0"))
TAIL_SOLVE_WEIGHT = float(os.environ.get("IMBH_GRADED_FLOWMAP_TAIL_WEIGHT", "1.0"))
TAIL_WEIGHT_COUNT = int(os.environ.get("IMBH_GRADED_FLOWMAP_TAIL_WEIGHT_COUNT", "0"))
BLEND_WIDTH = float(os.environ.get("IMBH_GRADED_FLOWMAP_BLEND_WIDTH", "0.6"))
ACCEPTANCE_LIMIT = float(os.environ.get("IMBH_GRADED_FLOWMAP_ACCEPTANCE_LIMIT", "1e-4"))
POST_ACCEPTANCE_LIMIT = float(os.environ.get("IMBH_GRADED_FLOWMAP_POST_ACCEPTANCE_LIMIT", "1e-3"))
SCIENCE_LIMIT = float(os.environ.get("IMBH_GRADED_FLOWMAP_SCIENCE_LIMIT", "5e-6"))
SEED_ONLY = os.environ.get("IMBH_GRADED_FLOWMAP_SEED_ONLY", "0") != "0"
STOP_ON_ACCEPTANCE = os.environ.get("IMBH_GRADED_FLOWMAP_STOP_ON_ACCEPTANCE", "1") != "0"
START_CHECKPOINT = os.environ.get("IMBH_GRADED_FLOWMAP_START_CHECKPOINT", "").strip()
TRANSITION_EXTENT_SEQUENCE = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_GRADED_FLOWMAP_TRANSITION_EXTENTS", "").replace(":", ",").split(",")
    if piece.strip()
)


@dataclass(frozen=True)
class GridConfig:
    micro_first_dx: float
    micro_growth: float
    micro_extent: float
    transition_extent: float
    transition_first_dx: float
    transition_growth: float


GRID_CONFIGS: dict[int, GridConfig] = {}


@dataclass(frozen=True)
class GradedGrid:
    logR: np.ndarray
    n_micro: int
    n_transition: int
    micro_extent: float
    transition_extent: float
    first_dx: float
    last_micro_dx: float
    first_transition_dx: float
    first_regular_dx: float


def default_grid_config() -> GridConfig:
    return GridConfig(
        micro_first_dx=MICRO_FIRST_DX,
        micro_growth=MICRO_GROWTH,
        micro_extent=MICRO_EXTENT,
        transition_extent=TRANSITION_EXTENT,
        transition_first_dx=TRANSITION_FIRST_DX,
        transition_growth=TRANSITION_GROWTH,
    )


def row_grid_config(row: dict[str, object]) -> GridConfig:
    return GridConfig(
        micro_first_dx=float(row.get("first_dx", MICRO_FIRST_DX)),
        micro_growth=MICRO_GROWTH,
        micro_extent=float(row.get("micro_extent", MICRO_EXTENT)),
        transition_extent=float(row.get("transition_extent", TRANSITION_EXTENT)),
        transition_first_dx=float(row.get("first_transition_dx", TRANSITION_FIRST_DX)),
        transition_growth=TRANSITION_GROWTH,
    )


def grid_config(params) -> GridConfig:
    return GRID_CONFIGS.get(id(params), default_grid_config())


def register_grid_config(params, config: GridConfig):
    GRID_CONFIGS[id(params)] = config
    return params


def micro_offsets(epsilon_buf: float, config: GridConfig) -> np.ndarray:
    if config.micro_first_dx <= 0.0:
        raise ValueError("MICRO_FIRST_DX must be positive")
    if config.micro_growth < 1.0:
        raise ValueError("MICRO_GROWTH must be >= 1")
    if config.micro_extent <= epsilon_buf:
        raise ValueError("MICRO_EXTENT must exceed epsilon_buf")
    offsets = [float(epsilon_buf)]
    current = float(epsilon_buf)
    dx = config.micro_first_dx
    while current < config.micro_extent - 1.0e-14:
        step = min(dx, config.micro_extent - current)
        if step <= 0.0:
            break
        current += step
        offsets.append(current)
        dx *= config.micro_growth
        if len(offsets) > 10000:
            raise RuntimeError("graded microgrid construction did not terminate")
    return np.asarray(offsets, dtype=float)


def transition_tail_offsets(start: float, config: GridConfig) -> np.ndarray:
    if config.transition_extent <= start:
        return np.empty(0, dtype=float)
    if config.transition_first_dx <= 0.0:
        raise ValueError("TRANSITION_FIRST_DX must be positive")
    if config.transition_growth < 1.0:
        raise ValueError("TRANSITION_GROWTH must be >= 1")
    offsets = []
    current = float(start)
    dx = config.transition_first_dx
    while current < config.transition_extent - 1.0e-14:
        step = min(dx, config.transition_extent - current)
        if step <= 0.0:
            break
        current += step
        offsets.append(current)
        dx *= config.transition_growth
        if len(offsets) > 10000:
            raise RuntimeError("graded transition grid construction did not terminate")
    return np.asarray(offsets, dtype=float)


def graded_inner_grid(logR_son: float, params) -> GradedGrid:
    logR_match = np.log(params.R_match)
    config = grid_config(params)
    offsets = micro_offsets(params.epsilon_buf, config)
    n_micro = len(offsets) - 1
    transition_offsets = transition_tail_offsets(float(offsets[-1]), config)
    n_transition = len(transition_offsets)
    shaped_offsets = np.concatenate([offsets, transition_offsets])
    n_shaped = n_micro + n_transition
    if n_shaped >= params.n_regular:
        raise ValueError("graded micro/transition grid consumes all regular intervals")
    shaped = logR_son + shaped_offsets
    if shaped[-1] >= logR_match:
        raise ValueError("graded micro/transition grid exceeds match radius")
    n_tail = params.n_regular - n_shaped
    regular = np.linspace(float(shaped[-1]), logR_match, n_tail + 1)
    logR = np.concatenate([shaped, regular[1:]])
    dx = np.diff(logR)
    return GradedGrid(
        logR=logR,
        n_micro=n_micro,
        n_transition=n_transition,
        micro_extent=float(offsets[-1]),
        transition_extent=float(shaped_offsets[-1]),
        first_dx=float(dx[0]) if len(dx) else np.nan,
        last_micro_dx=float(dx[n_micro - 1]) if n_micro > 0 else np.nan,
        first_transition_dx=float(dx[n_micro]) if n_transition > 0 and n_micro < len(dx) else np.nan,
        first_regular_dx=float(dx[n_shaped]) if n_shaped < len(dx) else np.nan,
    )


def solver_interval_residual(logu: np.ndarray, logT: np.ndarray, logR: np.ndarray, lambda0: float, params, idx: int, n_micro: int) -> np.ndarray:
    if idx >= n_micro or MICRO_SOLVE_FORM == "differential":
        return _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params.physics, idx)
    residual = _integrated_interval_residual_from_unpacked(logu, logT, logR, lambda0, params.physics, idx)
    dx = float(logR[idx + 1] - logR[idx])
    if MICRO_SOLVE_FORM == "integrated":
        return residual
    if MICRO_SOLVE_FORM == "integrated_sqrt_dx":
        return residual / np.sqrt(dx)
    if MICRO_SOLVE_FORM == "integrated_dx":
        return residual / dx
    raise ValueError(f"unknown MICRO_SOLVE_FORM {MICRO_SOLVE_FORM!r}")


def graded_residual(x: np.ndarray, params) -> np.ndarray:
    rows = []
    try:
        y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0, a_value = unpack_smooth_flowmap(x, params)
        grid = graded_inner_grid(logR_son, params)
        logR_i = grid.logR
        logR_o = outer_grid(params)  # type: ignore[arg-type]
        if np.any(np.diff(logR_i) <= 0.0) or np.any(np.diff(logR_o) <= 0.0):
            raise ValueError("mapped radii must increase")
        sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
        g = branch_gradient_from_a(logR_son, y_s, lambda0, params.physics, a_value, form=params.lhopital_form)
        lhopital = sonic_lhopital_residual_form(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form=params.lhopital_form)
        rows.append(np.array([sonic.D, compatibility_value(sonic, params.compatibility_pivot), lhopital], dtype=float))
        branch = branch_from_a(logR_son, y_s, lambda0, params, a_value, kind="graded")
        flow_values = integrate_flow_map_nodes(logR_son, y_s, lambda0, params, branch, np.array([logR_i[0]], dtype=float))
        rows.append(FLOW_SOLVE_WEIGHT * (np.array([logu_i[0], logT_i[0]], dtype=float) - flow_values[0]))
        for idx in range(params.n_inner - 1):
            residual = solver_interval_residual(logu_i, logT_i, logR_i, lambda0, params, idx, grid.n_micro)
            if idx < grid.n_micro:
                rows.append(MICRO_SOLVE_WEIGHT * residual)
            elif idx < grid.n_micro + grid.n_transition:
                rows.append(TRANSITION_SOLVE_WEIGHT * residual)
            elif TAIL_WEIGHT_COUNT > 0 and idx < grid.n_micro + grid.n_transition + TAIL_WEIGHT_COUNT:
                rows.append(TAIL_SOLVE_WEIGHT * residual)
            else:
                rows.append(residual)
        for idx in range(params.n_outer - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx))
        rows.append(np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float))
        rows.append(far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params))  # type: ignore[arg-type]
        return np.concatenate(rows)
    except Exception:
        return np.full(smooth_flowmap_size(params), 1.0e6)


def solve_graded(seed: np.ndarray, params, max_nfev: int):
    lower, upper = smooth_flowmap_bounds(params)
    x0 = np.clip(np.asarray(seed, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: graded_residual(trial, params),
        x0,
        jac_sparsity=smooth_flowmap_sparsity(params),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=max_nfev,
    )


def seed_profile(params, branch_a: float, reverse_info: dict[str, object], ctx, source_x: np.ndarray, seed_mode: str) -> np.ndarray:
    logu_i_old, logT_i_old, logu_o_old, logT_o_old, logR_son_old, _lambda_old = unpack_buffer(source_x, ctx.params)
    logR_i_old = buffer_inner_grid(logR_son_old, ctx.params)
    logR_o_old = outer_grid(ctx.params)  # type: ignore[arg-type]
    logR_son = float(reverse_info["logR_son"])
    lambda0 = float(reverse_info["lambda0"])
    y_s = np.asarray(reverse_info["y_s"], dtype=float)
    grid = graded_inner_grid(logR_son, params)
    logR_i_new = grid.logR
    logR_o_new = outer_grid(params)  # type: ignore[arg-type]
    branch = branch_from_a(logR_son, y_s, lambda0, params, branch_a, kind="graded-seed")

    old_logu = pchip_extrap(logR_i_old, logu_i_old, logR_i_new)
    old_logT = pchip_extrap(logR_i_old, logT_i_old, logR_i_new)
    try:
        if seed_mode == "inner_ode":
            values = integrate_flow_map_nodes(logR_son, y_s, lambda0, params, branch, logR_i_new)
            logu_i_new = values[:, 0]
            logT_i_new = values[:, 1]
        elif seed_mode in {"micro_ode_blend", "transition_ode_blend", "capped_ode_blend", "capped_hermite_blend"}:
            ode_count = grid.n_micro + 1
            if seed_mode == "transition_ode_blend":
                ode_count = grid.n_micro + grid.n_transition + 1
            elif seed_mode in {"capped_ode_blend", "capped_hermite_blend"}:
                seed_extent = ODE_SEED_EXTENT if ODE_SEED_EXTENT > grid.micro_extent else grid.micro_extent
                ode_count = int(np.searchsorted(logR_i_new - logR_son, seed_extent, side="right"))
                ode_count = max(grid.n_micro + 1, min(ode_count, len(logR_i_new)))
            values = integrate_flow_map_nodes(logR_son, y_s, lambda0, params, branch, logR_i_new[:ode_count])
            logu_i_new = old_logu.copy()
            logT_i_new = old_logT.copy()
            logu_i_new[:ode_count] = values[:, 0]
            logT_i_new[:ode_count] = values[:, 1]
            anchor = ode_count - 1
            if seed_mode == "capped_hermite_blend" and anchor + 1 < len(logR_i_new):
                blend_end_extent = BLEND_END_EXTENT if BLEND_END_EXTENT > 0.0 else grid.transition_extent
                end = int(np.searchsorted(logR_i_new - logR_son, blend_end_extent, side="right")) - 1
                end = max(anchor + 1, min(end, len(logR_i_new) - 1))
                anchor_x = float(logR_i_new[anchor])
                end_x = float(logR_i_new[end])
                if end_x > anchor_x:
                    g_anchor = local_ode_rhs(anchor_x, values[-1], lambda0, params.physics)
                    if 0 < end < len(logR_i_new) - 1:
                        dx_old = float(logR_i_new[end + 1] - logR_i_new[end - 1])
                        old_slope = np.array(
                            [
                                (old_logu[end + 1] - old_logu[end - 1]) / dx_old,
                                (old_logT[end + 1] - old_logT[end - 1]) / dx_old,
                            ],
                            dtype=float,
                        )
                    else:
                        dx_old = float(logR_i_new[end] - logR_i_new[end - 1])
                        old_slope = np.array(
                            [
                                (old_logu[end] - old_logu[end - 1]) / dx_old,
                                (old_logT[end] - old_logT[end - 1]) / dx_old,
                            ],
                            dtype=float,
                        )
                    bridge_x = logR_i_new[anchor : end + 1]
                    logu_i_new[anchor : end + 1] = CubicHermiteSpline(
                        [anchor_x, end_x],
                        [float(values[-1, 0]), float(old_logu[end])],
                        [float(g_anchor[0]), float(old_slope[0])],
                    )(bridge_x)
                    logT_i_new[anchor : end + 1] = CubicHermiteSpline(
                        [anchor_x, end_x],
                        [float(values[-1, 1]), float(old_logT[end])],
                        [float(g_anchor[1]), float(old_slope[1])],
                    )(bridge_x)
            else:
                delta = values[-1] - np.array([old_logu[anchor], old_logT[anchor]], dtype=float)
                if BLEND_WIDTH > 0.0:
                    distance = np.maximum(logR_i_new[anchor + 1 :] - logR_i_new[anchor], 0.0)
                    weight = np.exp(-distance / BLEND_WIDTH)
                    logu_i_new[anchor + 1 :] += delta[0] * weight
                    logT_i_new[anchor + 1 :] += delta[1] * weight
        else:
            raise ValueError(f"unknown seed_mode {seed_mode!r}")
    except Exception as exc:
        print(f"{seed_mode} ODE seed fallback to old branch: {exc}", flush=True)
        logu_i_new = old_logu
        logT_i_new = old_logT

    logu_o_new = pchip_extrap(logR_o_old, logu_o_old, logR_o_new)
    logT_o_new = pchip_extrap(logR_o_old, logT_o_old, logR_o_new)
    logu_o_new[0] = float(logu_i_new[-1])
    logT_o_new[0] = float(logT_i_new[-1])
    base = pack_flowmap(y_s, logu_i_new, logT_i_new, logu_o_new, logT_o_new, logR_son, lambda0)
    return pack_smooth_flowmap(base, branch_a)


def audit(label: str, x: np.ndarray, params, result=None, *, seed_mode: str, epsilon_buf: float, branch_index: int) -> dict[str, object]:
    y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0, a_value = unpack_smooth_flowmap(x, params)
    grid = graded_inner_grid(logR_son, params)
    logR_i = grid.logR
    logR_o = outer_grid(params)  # type: ignore[arg-type]
    sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
    g = branch_gradient_from_a(logR_son, y_s, lambda0, params.physics, a_value, form=params.lhopital_form)
    lhopital = sonic_lhopital_residual_form(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form=params.lhopital_form)
    branch = branch_from_a(logR_son, y_s, lambda0, params, a_value, kind="graded-audit")
    flow_values = integrate_flow_map_nodes(logR_son, y_s, lambda0, params, branch, np.array([logR_i[0]], dtype=float))
    flow_residual = np.array([logu_i[0], logT_i[0]], dtype=float) - flow_values[0]
    inner = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx)
            for idx in range(params.n_inner - 1)
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
    n_shaped = grid.n_micro + grid.n_transition
    micro = inner[: grid.n_micro]
    transition = inner[grid.n_micro : n_shaped]
    tail = inner[n_shaped:]
    post = inner[grid.n_micro :]
    interface = np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float)
    far = far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params)  # type: ignore[arg-type]
    combined_logR = np.concatenate([np.array([logR_son]), logR_i, logR_o[1:]])
    combined_logu = np.concatenate([np.array([y_s[0]]), logu_i, logu_o[1:]])
    combined_logT = np.concatenate([np.array([y_s[1]]), logT_i, logT_o[1:]])
    H_over_R = [
        algebraic_state(float(lr), float(lu), float(lt), lambda0, params.physics).H_over_R
        for lr, lu, lt in zip(combined_logR, combined_logu, combined_logT)
    ]
    blocks = {
        "D": abs(float(sonic.D)),
        "C1": abs(float(sonic.C1)),
        "C2": abs(float(sonic.C2)),
        "K": abs(float(sonic.compatibility)),
        "L": abs(float(lhopital)),
        "flow": float(np.max(np.abs(flow_residual))),
        "micro_R": float(np.max(np.abs(micro[:, 0]))) if len(micro) else 0.0,
        "micro_E": float(np.max(np.abs(micro[:, 1]))) if len(micro) else 0.0,
        "transition_R": float(np.max(np.abs(transition[:, 0]))) if len(transition) else 0.0,
        "transition_E": float(np.max(np.abs(transition[:, 1]))) if len(transition) else 0.0,
        "tail_R": float(np.max(np.abs(tail[:, 0]))) if len(tail) else 0.0,
        "tail_E": float(np.max(np.abs(tail[:, 1]))) if len(tail) else 0.0,
        "post_R": float(np.max(np.abs(post[:, 0]))) if len(post) else 0.0,
        "post_E": float(np.max(np.abs(post[:, 1]))) if len(post) else 0.0,
        "outer_R": float(np.max(np.abs(outer[:, 0]))),
        "outer_E": float(np.max(np.abs(outer[:, 1]))),
        "interface": float(np.max(np.abs(interface))),
        "far_omega": abs(float(far[0])),
        "far_energy": abs(float(far[1])),
    }
    physical = max(blocks.values())
    dominant = max(blocks, key=blocks.get)
    selected = float(np.max(np.abs(graded_residual(x, params))))
    if len(post):
        post_abs = np.max(np.abs(post), axis=1)
        worst_post_offset = int(np.argmax(post_abs))
        worst_post_i = grid.n_micro + worst_post_offset
        worst_post_component = "R" if abs(float(inner[worst_post_i, 0])) >= abs(float(inner[worst_post_i, 1])) else "E"
        worst_post_mid = 0.5 * float(logR_i[worst_post_i] + logR_i[worst_post_i + 1])
        worst_post_dx = float(logR_i[worst_post_i + 1] - logR_i[worst_post_i])
    else:
        worst_post_i = -1
        worst_post_component = "none"
        worst_post_mid = np.nan
        worst_post_dx = np.nan
    return {
        "label": label,
        "stage": "seed" if result is None else "solve",
        "seed_mode": seed_mode,
        "epsilon_buf": epsilon_buf,
        "branch": branch_index,
        "n_regular": params.n_regular,
        "n_micro": grid.n_micro,
        "n_transition": grid.n_transition,
        "micro_form": MICRO_SOLVE_FORM,
        "micro_extent": grid.micro_extent,
        "transition_extent": grid.transition_extent,
        "first_dx": grid.first_dx,
        "last_micro_dx": grid.last_micro_dx,
        "first_transition_dx": grid.first_transition_dx,
        "first_regular_dx": grid.first_regular_dx,
        "selected_max": selected,
        "physical_active": physical,
        "near_sonic_physical": max(blocks["flow"], blocks["micro_R"], blocks["micro_E"]),
        "passes_acceptance": bool(max(blocks["flow"], blocks["micro_R"], blocks["micro_E"]) <= ACCEPTANCE_LIMIT),
        "passes_science": bool(physical <= SCIENCE_LIMIT),
        "dominant": dominant,
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "L": float(lhopital),
        "flow": blocks["flow"],
        "micro_R": blocks["micro_R"],
        "micro_E": blocks["micro_E"],
        "transition_R": blocks["transition_R"],
        "transition_E": blocks["transition_E"],
        "tail_R": blocks["tail_R"],
        "tail_E": blocks["tail_E"],
        "post_R": blocks["post_R"],
        "post_E": blocks["post_E"],
        "outer_R": blocks["outer_R"],
        "outer_E": blocks["outer_E"],
        "interface": blocks["interface"],
        "far_omega": float(far[0]),
        "far_energy": float(far[1]),
        "first_micro_R": float(micro[0, 0]) if len(micro) else np.nan,
        "first_micro_E": float(micro[0, 1]) if len(micro) else np.nan,
        "first_post_R": float(post[0, 0]) if len(post) else np.nan,
        "first_post_E": float(post[0, 1]) if len(post) else np.nan,
        "worst_post_i": worst_post_i,
        "worst_post_component": worst_post_component,
        "worst_post_R_mid_rg": float(np.exp(worst_post_mid) / params.r_g) if np.isfinite(worst_post_mid) else np.nan,
        "worst_post_dx": worst_post_dx,
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "lambda0": float(lambda0),
        "int_adv": integrated_advective_fraction(combined_logu, combined_logT, combined_logR, lambda0, params),  # type: ignore[arg-type]
        "max_HR": float(np.max(H_over_R)),
        "a": float(a_value),
        "g_u": float(g[0]),
        "g_T": float(g[1]),
        "nfev": int(result.nfev) if result is not None else 0,
        "success": bool(result.success) if result is not None else True,
        "message": str(result.message) if result is not None else "seed evaluation",
        "x": np.asarray(x, dtype=float),
    }


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reverse-Seeded Graded Flow-Map BVP",
        "",
        "Generated by `scripts/run_transonic_reverse_seed_graded_flowmap_bvp.py`.",
        "",
        f"Config: `eps0={EPSILON0:g}`, `first_dx={MICRO_FIRST_DX:g}`, `growth={MICRO_GROWTH:g}`, "
        f"`micro_extent={MICRO_EXTENT:g}`, `transition_extent={TRANSITION_EXTENT:g}`, "
        f"`transition_first_dx={TRANSITION_FIRST_DX:g}`, `transition_growth={TRANSITION_GROWTH:g}`, "
        f"`micro_form={MICRO_SOLVE_FORM}`, `flow_weight={FLOW_SOLVE_WEIGHT:g}`, "
        f"`micro_weight={MICRO_SOLVE_WEIGHT:g}`, `transition_weight={TRANSITION_SOLVE_WEIGHT:g}`, "
        f"`acceptance={ACCEPTANCE_LIMIT:g}`.",
        "",
        "| label | stage | seed | eps_buf | branch | N regular | N micro | N trans | physical | near sonic | accepted | selected | dominant | flow | micro R | micro E | trans R | trans E | tail R | tail E | post R | post E | worst post | worst R/rg | worst dx | outer R | far omega | D | C1 | C2 | K | L | Rson/rg | lambda0 | int adv | max H/R | first dx | first trans dx | first tail dx | a | g_u | g_T | nfev | success | message |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {stage} | {seed_mode} | {epsilon_buf} | {branch} | {n_regular} | {n_micro} | {n_transition} | "
            "{physical_active} | {near_sonic_physical} | {passes_acceptance} | {selected_max} | {dominant} | "
            "{flow} | {micro_R} | {micro_E} | {transition_R} | {transition_E} | {tail_R} | {tail_E} | "
            "{post_R} | {post_E} | {worst_post} | {worst_post_R_mid_rg} | {worst_post_dx} | "
            "{outer_R} | {far_omega} | {D} | {C1} | {C2} | {K} | {L} | {Rson_rg} | {lambda0} | "
            "{int_adv} | {max_HR} | {first_dx} | {first_transition_dx} | {first_regular_dx} | "
            "{a} | {g_u} | {g_T} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                stage=row["stage"],
                seed_mode=row["seed_mode"],
                epsilon_buf=fmt(float(row["epsilon_buf"])),
                branch=row["branch"],
                n_regular=row["n_regular"],
                n_micro=row["n_micro"],
                n_transition=row["n_transition"],
                physical_active=fmt(float(row["physical_active"])),
                near_sonic_physical=fmt(float(row["near_sonic_physical"])),
                passes_acceptance="yes" if row["passes_acceptance"] else "no",
                selected_max=fmt(float(row["selected_max"])),
                dominant=row["dominant"],
                flow=fmt(float(row["flow"])),
                micro_R=fmt(float(row["micro_R"])),
                micro_E=fmt(float(row["micro_E"])),
                transition_R=fmt(float(row["transition_R"])),
                transition_E=fmt(float(row["transition_E"])),
                tail_R=fmt(float(row["tail_R"])),
                tail_E=fmt(float(row["tail_E"])),
                post_R=fmt(float(row["post_R"])),
                post_E=fmt(float(row["post_E"])),
                worst_post=f"{row['worst_post_i']}:{row['worst_post_component']}",
                worst_post_R_mid_rg=fmt(float(row["worst_post_R_mid_rg"])),
                worst_post_dx=fmt(float(row["worst_post_dx"])),
                outer_R=fmt(float(row["outer_R"])),
                far_omega=fmt(float(row["far_omega"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                L=fmt(float(row["L"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                max_HR=fmt(float(row["max_HR"])),
                first_dx=fmt(float(row["first_dx"])),
                first_transition_dx=fmt(float(row["first_transition_dx"])),
                first_regular_dx=fmt(float(row["first_regular_dx"])),
                a=fmt(float(row["a"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def make_params(
    fiducial: FiducialParams,
    source_meta: dict[str, object],
    n_regular: int,
    epsilon_buf: float,
    branch_index: int,
    branch_a: float,
    mdot_edd: float,
    config: GridConfig | None = None,
):
    params = make_flowmap_params(
        fiducial,
        float(source_meta["ratio"]),
        mdot_edd,
        int(n_regular),
        int(source_meta["n_outer"]),
        float(source_meta["R_far_rg"]),
        float(epsilon_buf),
        int(branch_index),
        float(branch_a),
    )
    return register_grid_config(replace(params, epsilon0=EPSILON0), config or default_grid_config())


def load_start_checkpoint(path: Path, fiducial: FiducialParams, source_meta: dict[str, object], mdot_edd: float):
    data = np.load(path, allow_pickle=True)
    row = json.loads(str(data["row_json"].item()))
    params = make_params(
        fiducial,
        source_meta,
        int(row["n_regular"]),
        float(row["epsilon_buf"]),
        int(row["branch"]),
        float(row["a"]),
        mdot_edd,
        row_grid_config(row),
    )
    return np.asarray(data["x"], dtype=float), params, row


def remap_smooth_seed(old_x: np.ndarray, old_params, new_params) -> np.ndarray:
    y_old, logu_old, logT_old, logu_o_old, logT_o_old, logR_old, lambda_old, a_old = unpack_smooth_flowmap(old_x, old_params)
    old_grid = graded_inner_grid(logR_old, old_params)
    new_grid = graded_inner_grid(logR_old, new_params)
    old_outer = outer_grid(old_params)  # type: ignore[arg-type]
    new_outer = outer_grid(new_params)  # type: ignore[arg-type]
    logu_new = pchip_extrap(old_grid.logR, logu_old, new_grid.logR)
    logT_new = pchip_extrap(old_grid.logR, logT_old, new_grid.logR)
    old_shaped = old_grid.n_micro + old_grid.n_transition
    new_shaped = new_grid.n_micro + new_grid.n_transition
    if new_shaped > old_shaped and old_shaped < len(old_grid.logR) and new_shaped < len(new_grid.logR):
        x_start = float(old_grid.logR[old_shaped])
        x_end = float(new_grid.logR[new_shaped])
        new_nodes = new_grid.logR[(new_grid.logR > x_start + 1.0e-12) & (new_grid.logR <= x_end + 1.0e-12)]
        if len(new_nodes) > 0 and x_end > x_start:
            y_start = np.array([float(logu_old[old_shaped]), float(logT_old[old_shaped])], dtype=float)

            def rhs(x_value: float, y_value: np.ndarray) -> np.ndarray:
                return local_ode_rhs(float(x_value), y_value, float(lambda_old), new_params.physics)

            try:
                sol = solve_ivp(
                    rhs,
                    (x_start, x_end),
                    y_start,
                    method="Radau",
                    dense_output=True,
                    max_step=0.001,
                    rtol=1.0e-8,
                    atol=1.0e-10,
                )
                if sol.success and sol.sol is not None:
                    ode_values = np.asarray(sol.sol(new_nodes).T, dtype=float)
                    node_indices = np.searchsorted(new_grid.logR, new_nodes)
                    endpoint_pchip = np.array([logu_new[node_indices[-1]], logT_new[node_indices[-1]]], dtype=float)
                    logu_new[node_indices] = ode_values[:, 0]
                    logT_new[node_indices] = ode_values[:, 1]
                    if len(node_indices) > 0 and BLEND_WIDTH > 0.0:
                        anchor = int(node_indices[-1])
                        delta = ode_values[-1] - endpoint_pchip
                        distance = np.maximum(new_grid.logR[anchor + 1 :] - new_grid.logR[anchor], 0.0)
                        weight = np.exp(-distance / BLEND_WIDTH)
                        logu_new[anchor + 1 :] += delta[0] * weight
                        logT_new[anchor + 1 :] += delta[1] * weight
            except Exception as exc:
                print(f"transition ODE remap fallback to PCHIP: {exc}", flush=True)
    logu_o_new = pchip_extrap(old_outer, logu_o_old, new_outer)
    logT_o_new = pchip_extrap(old_outer, logT_o_old, new_outer)
    logu_o_new[0] = float(logu_new[-1])
    logT_o_new[0] = float(logT_new[-1])
    return pack_smooth_flowmap(pack_flowmap(y_old, logu_new, logT_new, logu_o_new, logT_o_new, logR_old, lambda_old), a_old)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    ctx = load_context()
    source_x, source_meta = load_row(SOURCE_CHECKPOINT)
    _fit_result, reverse_info = solve_fit(ctx, "fixed_buffer", None)
    logR_son = float(reverse_info["logR_son"])
    y_s = np.asarray(reverse_info["y_s"], dtype=float)
    lambda0 = float(reverse_info["lambda0"])
    branches = sonic_derivative_branches(
        logR_son,
        y_s,
        lambda0,
        ctx.params.physics,
        eps=LHOPITAL_EPS,
        form="scaled",
        half_width=BRANCH_HALF_WIDTH,
        scan_points=BRANCH_POINTS,
    )
    if not branches:
        raise RuntimeError("reverse-fitted point has no L'Hopital branches")
    print(
        f"graded flow-map BVP Rson={float(reverse_info['Rson_rg']):.6g} lambda={lambda0:.6g} "
        f"branches={len(branches)} eps={EPSILON_BUFS} N={N_SEQUENCE} modes={SEED_MODES}",
        flush=True,
    )
    transition_extents = TRANSITION_EXTENT_SEQUENCE or (TRANSITION_EXTENT,)
    rows: list[dict[str, object]] = []
    accepted = False
    for seed_mode in SEED_MODES:
        for epsilon_buf in EPSILON_BUFS:
            for branch_index in BRANCH_SEQUENCE:
                if branch_index >= len(branches):
                    continue
                branch = branches[branch_index]
                current_x = None
                current_params = None
                if START_CHECKPOINT:
                    current_x, current_params, start_row = load_start_checkpoint(Path(START_CHECKPOINT), fiducial, source_meta, mdot_edd)
                    print(
                        f"loaded start checkpoint {START_CHECKPOINT} "
                        f"N={start_row.get('n_regular')} eps={start_row.get('epsilon_buf')} "
                        f"branch={start_row.get('branch')} transition={start_row.get('transition_extent')}",
                        flush=True,
                    )
                for transition_extent in transition_extents:
                    stage_config = replace(default_grid_config(), transition_extent=float(transition_extent))
                    for n_regular in N_SEQUENCE:
                        params = make_params(fiducial, source_meta, n_regular, epsilon_buf, branch_index, float(branch.a), mdot_edd, stage_config)
                        if current_x is None or current_params is None:
                            seed = seed_profile(params, float(branch.a), reverse_info, ctx, source_x, seed_mode)
                        else:
                            seed = remap_smooth_seed(current_x, current_params, params)
                        label = (
                            f"graded_{seed_mode}_eps{epsilon_buf:g}_branch{branch_index}_"
                            f"te{transition_extent:g}_N{n_regular}"
                        ).replace(".", "p")
                        seed_row = audit(label, seed, params, seed_mode=seed_mode, epsilon_buf=epsilon_buf, branch_index=branch_index)
                        seed_row["stage"] = "seed"
                        rows.append(seed_row)
                        write_table(rows)
                        print(
                            f"{label} seed near={seed_row['near_sonic_physical']:.3e} physical={seed_row['physical_active']:.3e} "
                            f"dominant={seed_row['dominant']} n_micro={seed_row['n_micro']} n_trans={seed_row['n_transition']} "
                            f"tail={max(seed_row['tail_R'], seed_row['tail_E']):.3e}",
                            flush=True,
                        )
                        if SEED_ONLY:
                            current_x = seed
                            current_params = params
                            continue

                        release = solve_graded(seed, params, MAX_NFEV_RELEASE)
                        release_row = audit(label, release.x, params, release, seed_mode=seed_mode, epsilon_buf=epsilon_buf, branch_index=branch_index)
                        release_row["stage"] = "release"
                        rows.append(release_row)
                        write_table(rows)
                        print(
                            f"{label} release near={release_row['near_sonic_physical']:.3e} physical={release_row['physical_active']:.3e} "
                            f"dominant={release_row['dominant']} tail={max(release_row['tail_R'], release_row['tail_E']):.3e} "
                            f"nfev={release.nfev}",
                            flush=True,
                        )

                        polish = solve_graded(release.x, params, MAX_NFEV_POLISH)
                        polish_row = audit(label, polish.x, params, polish, seed_mode=seed_mode, epsilon_buf=epsilon_buf, branch_index=branch_index)
                        polish_row["stage"] = "polish"
                        rows.append(polish_row)
                        write_table(rows)
                        save_checkpoint(f"graded_{label}", np.asarray(polish_row["x"], dtype=float), polish_row)
                        print(
                            f"{label} polish near={polish_row['near_sonic_physical']:.3e} physical={polish_row['physical_active']:.3e} "
                            f"dominant={polish_row['dominant']} tail={max(polish_row['tail_R'], polish_row['tail_E']):.3e} "
                            f"nfev={polish.nfev}",
                            flush=True,
                        )
                        current_x = np.asarray(polish_row["x"], dtype=float)
                        current_params = params
                        near_ok = bool(polish_row["passes_acceptance"])
                        transition_ok = max(float(polish_row["transition_R"]), float(polish_row["transition_E"])) <= ACCEPTANCE_LIMIT
                        post_ok = max(float(polish_row["post_R"]), float(polish_row["post_E"])) <= POST_ACCEPTANCE_LIMIT
                        if near_ok and transition_ok and post_ok:
                            accepted = True
                if accepted:
                    if STOP_ON_ACCEPTANCE:
                        print("full acceptance criterion met; stopping scan", flush=True)
                        write_table(rows)
                        return
                    accepted = False
    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"checkpoints in {CHECKPOINT_DIR}", flush=True)


if __name__ == "__main__":
    main()
