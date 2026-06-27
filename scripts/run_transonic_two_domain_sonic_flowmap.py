"""Offset sonic flow-map experiment for the two-domain transonic root."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.sparse import lil_matrix

from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    TransonicSlimParams,
    _differential_interval_residual_from_unpacked,
    state_bounds,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    SonicDerivativeBranch,
    algebraic_state,
    differential_matrix,
    local_ode_rhs,
    scaled_differential_matrix,
    sonic_derivative_branches,
    sonic_diagnostics,
    sonic_lhopital_residual_form,
    sonic_null_vectors,
    sonic_unscaled_null_vectors,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_dynamic_sonic_patch import CHECKPOINT_DIR as DYNAMIC_CHECKPOINT_DIR, load_row
from run_transonic_two_domain_mesh_validation import load_checkpoint
from run_transonic_two_domain_outer_extension import (
    R_MATCH_RG,
    far_boundary_residual,
    integrated_advective_fraction,
    outer_grid,
)
from run_transonic_two_domain_sonic_refinement_sprint import (
    SOURCE_CHECKPOINT,
    buffer_inner_grid,
    make_buffer_params,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_two_domain_sonic_flowmap"
BRANCH_TABLE = TABLE_DIR / "transonic_sonic_flowmap_branch_audit.md"
ADJUGATE_TABLE = TABLE_DIR / "transonic_sonic_adjugate_lhopital_audit.md"
FIT_TABLE = TABLE_DIR / "transonic_sonic_flowmap_fit.md"
SMOOTH_FIT_TABLE = TABLE_DIR / "transonic_sonic_flowmap_smooth_fit.md"
CONSTRAINED_FIT_TABLE = TABLE_DIR / "transonic_sonic_flowmap_constrained_fit.md"
BVP_TABLE = TABLE_DIR / "transonic_two_domain_sonic_flowmap_bvp.md"
SMOOTH_BVP_TABLE = TABLE_DIR / "transonic_two_domain_sonic_flowmap_smooth_bvp.md"
DYNAMIC_N64_CHECKPOINT = DYNAMIC_CHECKPOINT_DIR / "Nreg64_0p90277664.npz"

DEFAULT_EPSILON_BUFS = (0.02, 0.015, 0.01, 0.0075, 0.005)
DEFAULT_BVP_EPSILON_BUFS = (0.02,)
DEFAULT_N_SEQUENCE = (64,)
SCIENCE_LIMIT = 5.0e-6
MAX_NFEV_FIT = int(os.environ.get("IMBH_FLOWMAP_MAX_NFEV_FIT", "160"))
MAX_NFEV_SMOOTH_FIT = int(os.environ.get("IMBH_FLOWMAP_MAX_NFEV_SMOOTH_FIT", "120"))
MAX_NFEV_CONSTRAINED_FIT = int(os.environ.get("IMBH_FLOWMAP_MAX_NFEV_CONSTRAINED_FIT", "80"))
MAX_NFEV_RELEASE = int(os.environ.get("IMBH_FLOWMAP_MAX_NFEV_RELEASE", "220"))
MAX_NFEV_POLISH = int(os.environ.get("IMBH_FLOWMAP_MAX_NFEV_POLISH", "120"))
MAX_NFEV_SMOOTH_BVP_RELEASE = int(os.environ.get("IMBH_FLOWMAP_MAX_NFEV_SMOOTH_BVP_RELEASE", "80"))
MAX_NFEV_SMOOTH_BVP_POLISH = int(os.environ.get("IMBH_FLOWMAP_MAX_NFEV_SMOOTH_BVP_POLISH", "40"))
EPSILON0 = float(os.environ.get("IMBH_FLOWMAP_EPSILON0", "1e-5"))
FLOW_RTOL = float(os.environ.get("IMBH_FLOWMAP_RTOL", "1e-8"))
FLOW_ATOL = float(os.environ.get("IMBH_FLOWMAP_ATOL", "1e-10"))
FLOW_METHOD = os.environ.get("IMBH_FLOWMAP_METHOD", "Radau")
FLOW_MATCH_SCALE = float(os.environ.get("IMBH_FLOWMAP_MATCH_SCALE", "1.0"))
INNER_GRID_POWER = float(os.environ.get("IMBH_FLOWMAP_INNER_GRID_POWER", "1.0"))
REMAP_METHOD = os.environ.get("IMBH_FLOWMAP_REMAP_METHOD", "pchip").strip().lower()
MICRO_R_RG = float(os.environ.get("IMBH_FLOWMAP_MICRO_R_RG", "0.0"))
N_MICRO = int(os.environ.get("IMBH_FLOWMAP_N_MICRO", "0"))
MICRO_SEED_FLOWMAP = os.environ.get("IMBH_FLOWMAP_MICRO_SEED_FLOWMAP", "0") != "0"
LHOPITAL_FORM = os.environ.get("IMBH_FLOWMAP_LHOPITAL_FORM", "scaled")
LHOPITAL_EPS = float(os.environ.get("IMBH_FLOWMAP_LHOPITAL_EPS", "1e-5"))
ROOT_SCAN_HALF_WIDTH = float(os.environ.get("IMBH_FLOWMAP_ROOT_HALF_WIDTH", "1000"))
ROOT_SCAN_POINTS = int(os.environ.get("IMBH_FLOWMAP_ROOT_POINTS", "1201"))
BVP_BRANCH_HALF_WIDTH = float(os.environ.get("IMBH_FLOWMAP_BVP_BRANCH_HALF_WIDTH", "80"))
BVP_BRANCH_SCAN_POINTS = int(os.environ.get("IMBH_FLOWMAP_BVP_BRANCH_POINTS", "121"))
SMOOTH_A_HALF_WIDTH = float(os.environ.get("IMBH_FLOWMAP_SMOOTH_A_HALF_WIDTH", "500"))
CONSTRAINED_A_HALF_WIDTH = float(os.environ.get("IMBH_FLOWMAP_CONSTRAINED_A_HALF_WIDTH", "120"))
CONSTRAINED_RSON_HALF_WIDTH_RG = float(os.environ.get("IMBH_FLOWMAP_CONSTRAINED_RSON_HALF_WIDTH_RG", "0.25"))
CONSTRAINED_Y_HALF_WIDTH = float(os.environ.get("IMBH_FLOWMAP_CONSTRAINED_Y_HALF_WIDTH", "0.6"))
CONSTRAINED_LAMBDA_HALF_WIDTH = float(os.environ.get("IMBH_FLOWMAP_CONSTRAINED_LAMBDA_HALF_WIDTH", "0.03"))
CONSTRAINED_COMPAT_WEIGHT = float(os.environ.get("IMBH_FLOWMAP_CONSTRAINED_COMPAT_WEIGHT", "80"))
CONSTRAINED_REG_WEIGHT = float(os.environ.get("IMBH_FLOWMAP_CONSTRAINED_REG_WEIGHT", "10"))
CONSTRAINED_FLOW_WEIGHT = float(os.environ.get("IMBH_FLOWMAP_CONSTRAINED_FLOW_WEIGHT", "1"))
CONSTRAINED_G_LIMIT = float(os.environ.get("IMBH_FLOWMAP_CONSTRAINED_G_LIMIT", "1000"))
CONSTRAINED_G_WEIGHT = float(os.environ.get("IMBH_FLOWMAP_CONSTRAINED_G_WEIGHT", "20"))
SMOOTH_BVP_A_HALF_WIDTH = float(os.environ.get("IMBH_FLOWMAP_SMOOTH_BVP_A_HALF_WIDTH", "500"))
ADJUGATE_SCAN_HALF_WIDTH = float(os.environ.get("IMBH_FLOWMAP_ADJ_HALF_WIDTH", "1000"))
ADJUGATE_SCAN_POINTS = int(os.environ.get("IMBH_FLOWMAP_ADJ_POINTS", "1201"))


@dataclass(frozen=True)
class FlowmapParams:
    physics: TransonicSlimParams
    n_regular: int
    n_outer: int
    R_match_rg: float
    R_far_rg: float
    epsilon_buf: float
    epsilon0: float
    branch_index: int
    branch_a_center: float | None
    lhopital_form: str = "scaled"
    compatibility_pivot: str = "C1"
    far_closure: str = "pressure_supported"
    inner_grid_power: float = 1.0
    micro_R_rg: float = 0.0
    n_micro: int = 0
    grid_power_outer: float = 1.0
    flow_match_scale: float = 1.0
    flow_method: str = "Radau"
    flow_rtol: float = 1.0e-8
    flow_atol: float = 1.0e-10

    @property
    def n_inner(self) -> int:
        return self.n_regular + 1

    @property
    def r_g(self) -> float:
        return self.physics.r_g

    @property
    def R_match(self) -> float:
        return self.R_match_rg * self.r_g

    @property
    def R_far(self) -> float:
        return self.R_far_rg * self.r_g


@dataclass(frozen=True)
class FlowMapResult:
    y_buffer: np.ndarray
    branch: SonicDerivativeBranch
    success: bool
    message: str
    nfev: int
    n_steps: int
    max_HR: float


def parse_float_sequence(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return tuple(float(piece) for piece in raw.replace(":", ",").split(",") if piece.strip())


def parse_int_sequence(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return tuple(int(piece) for piece in raw.replace(":", ",").split(",") if piece.strip())


def parse_branch_sequence() -> tuple[int, ...]:
    raw = os.environ.get("IMBH_FLOWMAP_BRANCHES")
    if raw is None or not raw.strip():
        return (0, 1)
    return tuple(int(piece) for piece in raw.replace(":", ",").split(",") if piece.strip())


def parse_string_sequence(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return tuple(piece.strip() for piece in raw.replace(":", ",").split(",") if piece.strip())


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items() if key != "x"}, sort_keys=True)


def inner_grid_label_suffix(params: FlowmapParams) -> str:
    pieces: list[str] = []
    if params.micro_R_rg > 0.0 and params.n_micro > 0:
        micro_tag = f"mr{params.micro_R_rg:g}".replace(".", "p").replace("-", "m")
        pieces.append(f"{micro_tag}_nm{params.n_micro}")
    elif abs(params.inner_grid_power - 1.0) < 1.0e-12:
        pass
    else:
        pieces.append(f"ip{params.inner_grid_power:g}".replace(".", "p").replace("-", "m"))
    if REMAP_METHOD != "pchip":
        pieces.append(REMAP_METHOD.replace(".", "p").replace("-", "m"))
    return "" if not pieces else "_" + "_".join(pieces)


def pchip_extrap(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.asarray(PchipInterpolator(x_old, y_old, extrapolate=True)(x_new), dtype=float)


def ode_slope_remap(
    logR_old: np.ndarray,
    logu_old: np.ndarray,
    logT_old: np.ndarray,
    logR_new: np.ndarray,
    lambda0: float,
    params: FlowmapParams,
) -> tuple[np.ndarray, np.ndarray]:
    slopes = []
    for logR, logu, logT in zip(logR_old, logu_old, logT_old):
        slopes.append(local_ode_rhs(float(logR), np.array([logu, logT], dtype=float), lambda0, params.physics))
    slope_array = np.asarray(slopes, dtype=float)
    if slope_array.shape != (len(logR_old), 2) or not np.all(np.isfinite(slope_array)):
        raise ValueError("ODE remap slopes are not finite")
    logu_new = CubicHermiteSpline(logR_old, logu_old, slope_array[:, 0], extrapolate=True)(logR_new)
    logT_new = CubicHermiteSpline(logR_old, logT_old, slope_array[:, 1], extrapolate=True)(logR_new)
    return np.asarray(logu_new, dtype=float), np.asarray(logT_new, dtype=float)


def remap_profile(
    logR_old: np.ndarray,
    logu_old: np.ndarray,
    logT_old: np.ndarray,
    logR_new: np.ndarray,
    lambda0: float,
    params: FlowmapParams,
) -> tuple[np.ndarray, np.ndarray]:
    if REMAP_METHOD == "hermite":
        try:
            return ode_slope_remap(logR_old, logu_old, logT_old, logR_new, lambda0, params)
        except Exception as exc:
            print(f"Hermite remap fallback to PCHIP: {exc}", flush=True)
    elif REMAP_METHOD != "pchip":
        raise ValueError(f"unknown remap method {REMAP_METHOD!r}")
    return (
        pchip_extrap(logR_old, logu_old, logR_new),
        pchip_extrap(logR_old, logT_old, logR_new),
    )


def make_flowmap_params(
    fiducial: FiducialParams,
    ratio: float,
    mdot_edd: float,
    n_regular: int,
    n_outer: int,
    R_far_rg: float,
    epsilon_buf: float,
    branch_index: int,
    branch_a_center: float | None,
) -> FlowmapParams:
    physics = TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=n_regular + 1,
        R_out_rg=R_far_rg,
        residual_tol=1.0e-6,
        max_nfev=MAX_NFEV_RELEASE,
        outer_closure="pressure_supported_thin_energy",
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )
    return FlowmapParams(
        physics=physics,
        n_regular=int(n_regular),
        n_outer=int(n_outer),
        R_match_rg=R_MATCH_RG,
        R_far_rg=float(R_far_rg),
        epsilon_buf=float(epsilon_buf),
        epsilon0=EPSILON0,
        branch_index=int(branch_index),
        branch_a_center=branch_a_center,
        lhopital_form=LHOPITAL_FORM,
        inner_grid_power=INNER_GRID_POWER,
        micro_R_rg=MICRO_R_RG,
        n_micro=N_MICRO,
        flow_match_scale=FLOW_MATCH_SCALE,
        flow_method=FLOW_METHOD,
        flow_rtol=FLOW_RTOL,
        flow_atol=FLOW_ATOL,
    )


def flow_inner_grid(logR_son: float, params: FlowmapParams) -> np.ndarray:
    logR_buffer = logR_son + params.epsilon_buf
    logR_match = np.log(params.R_match)
    if logR_buffer >= logR_match:
        raise ValueError("sonic buffer exceeds match radius")
    if params.micro_R_rg > 0.0 or params.n_micro > 0:
        if params.micro_R_rg <= 0.0 or params.n_micro <= 0:
            raise ValueError("both micro_R_rg and n_micro must be positive for the micro-domain grid")
        if params.n_micro >= params.n_regular:
            raise ValueError("n_micro must be smaller than n_regular")
        logR_micro = np.log(params.micro_R_rg * params.r_g)
        if not logR_buffer < logR_micro < logR_match:
            raise ValueError("micro-domain radius must lie between sonic buffer and match radius")
        n_regular_outer = params.n_regular - params.n_micro
        micro = np.linspace(logR_buffer, logR_micro, params.n_micro + 1)
        regular = np.linspace(logR_micro, logR_match, n_regular_outer + 1)
        return np.concatenate([micro, regular[1:]])
    if params.inner_grid_power <= 0.0:
        raise ValueError("inner grid power must be positive")
    unit = np.linspace(0.0, 1.0, params.n_inner)
    stretched = unit**params.inner_grid_power
    return logR_buffer + (logR_match - logR_buffer) * stretched


def flowmap_size(params: FlowmapParams) -> int:
    return 2 + 2 * params.n_inner + 2 * params.n_outer + 2


def unpack_flowmap(x: np.ndarray, params: FlowmapParams):
    x = np.asarray(x, dtype=float)
    expected = flowmap_size(params)
    if x.shape != (expected,):
        raise ValueError(f"x must have shape ({expected},)")
    ni = params.n_inner
    no = params.n_outer
    offset = 0
    y_s = x[offset : offset + 2]
    offset += 2
    logu_i = x[offset : offset + ni]
    offset += ni
    logT_i = x[offset : offset + ni]
    offset += ni
    logu_o = x[offset : offset + no]
    offset += no
    logT_o = x[offset : offset + no]
    offset += no
    logR_son = float(x[offset])
    lambda0 = float(x[offset + 1])
    return y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0


def pack_flowmap(
    y_s: np.ndarray,
    logu_i: np.ndarray,
    logT_i: np.ndarray,
    logu_o: np.ndarray,
    logT_o: np.ndarray,
    logR_son: float,
    lambda0: float,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(y_s, dtype=float),
            np.asarray(logu_i, dtype=float),
            np.asarray(logT_i, dtype=float),
            np.asarray(logu_o, dtype=float),
            np.asarray(logT_o, dtype=float),
            np.array([logR_son, lambda0], dtype=float),
        ]
    )


def smooth_flowmap_size(params: FlowmapParams) -> int:
    return flowmap_size(params) + 1


def pack_smooth_flowmap(base: np.ndarray, a_value: float) -> np.ndarray:
    return np.concatenate([np.asarray(base, dtype=float), np.array([float(a_value)], dtype=float)])


def unpack_smooth_flowmap(x: np.ndarray, params: FlowmapParams):
    x = np.asarray(x, dtype=float)
    expected = smooth_flowmap_size(params)
    if x.shape != (expected,):
        raise ValueError(f"x must have shape ({expected},)")
    y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_flowmap(x[:-1], params)
    a_value = float(x[-1])
    return y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0, a_value


def load_dynamic_source(fiducial: FiducialParams, ratio: float, mdot_edd: float):
    if not DYNAMIC_N64_CHECKPOINT.exists():
        raise FileNotFoundError(f"missing dynamic sonic-patch checkpoint: {DYNAMIC_N64_CHECKPOINT}")
    x, meta = load_row(DYNAMIC_N64_CHECKPOINT)
    params = make_buffer_params(
        fiducial,
        ratio,
        mdot_edd,
        int(meta["n_regular"]),
        int(meta["n_outer"]),
        float(meta["R_far_rg"]),
        float(meta["delta_s"]),
    )
    return x, params, meta


def select_branch(logR_son: float, y_s: np.ndarray, lambda0: float, params: FlowmapParams, *, wide: bool = False) -> SonicDerivativeBranch:
    half_width = ROOT_SCAN_HALF_WIDTH if wide or params.branch_a_center is None else BVP_BRANCH_HALF_WIDTH
    scan_points = ROOT_SCAN_POINTS if wide or params.branch_a_center is None else BVP_BRANCH_SCAN_POINTS
    branches = sonic_derivative_branches(
        logR_son,
        y_s,
        lambda0,
        params.physics,
        eps=LHOPITAL_EPS,
        form=params.lhopital_form,
        a_center=params.branch_a_center,
        half_width=half_width,
        scan_points=scan_points,
    )
    if not branches and not wide:
        branches = sonic_derivative_branches(
            logR_son,
            y_s,
            lambda0,
            params.physics,
            eps=LHOPITAL_EPS,
            form=params.lhopital_form,
            a_center=params.branch_a_center,
            half_width=ROOT_SCAN_HALF_WIDTH,
            scan_points=max(ROOT_SCAN_POINTS // 2, BVP_BRANCH_SCAN_POINTS),
        )
    if not branches:
        raise RuntimeError("no sonic derivative branch found")
    if params.branch_a_center is not None:
        return min(branches, key=lambda branch: abs(branch.a - float(params.branch_a_center)))
    if params.branch_index >= len(branches):
        raise RuntimeError(f"branch {params.branch_index} unavailable; found {len(branches)} branches")
    return branches[params.branch_index]


def branch_gradient_from_a(logR_son: float, y_s: np.ndarray, lambda0: float, physics, a_value: float, form: str = "scaled") -> np.ndarray:
    if form == "raw":
        nulls = sonic_unscaled_null_vectors(logR_son, y_s, lambda0, physics)
    else:
        nulls = sonic_null_vectors(logR_son, y_s, lambda0, physics)
    g_p = np.linalg.lstsq(nulls.matrix, -nulls.rhs, rcond=None)[0]
    r = nulls.right_null / (np.linalg.norm(nulls.right_null) + 1.0e-300)
    return np.asarray(g_p + float(a_value) * r, dtype=float)


def branch_from_a(logR_son: float, y_s: np.ndarray, lambda0: float, params: FlowmapParams, a_value: float, kind: str = "smooth") -> SonicDerivativeBranch:
    g = branch_gradient_from_a(logR_son, y_s, lambda0, params.physics, a_value, form=params.lhopital_form)
    l_norm = sonic_lhopital_residual_form(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form=params.lhopital_form)
    return SonicDerivativeBranch(
        kind=kind,
        form=params.lhopital_form,
        a=float(a_value),
        gradient=g,
        lhopital_raw=float(l_norm),
        lhopital_normalized=float(l_norm),
    )


def integrate_flow_map(logR_son: float, y_s: np.ndarray, lambda0: float, params: FlowmapParams, branch: SonicDerivativeBranch) -> FlowMapResult:
    if not 0.0 < params.epsilon0 < params.epsilon_buf:
        raise ValueError("epsilon0 must be positive and smaller than epsilon_buf")
    x0 = logR_son + params.epsilon0
    x1 = logR_son + params.epsilon_buf
    y0 = np.asarray(y_s, dtype=float) + params.epsilon0 * branch.gradient

    def rhs(x_value: float, y_value: np.ndarray) -> np.ndarray:
        return local_ode_rhs(float(x_value), y_value, lambda0, params.physics)

    sol = solve_ivp(
        rhs,
        (x0, x1),
        y0,
        method=params.flow_method,
        rtol=params.flow_rtol,
        atol=params.flow_atol,
    )
    if (not sol.success) or sol.y.shape[1] == 0 or not np.all(np.isfinite(sol.y[:, -1])):
        raise RuntimeError(sol.message)
    h_over_r = [
        algebraic_state(float(x_val), float(y_val[0]), float(y_val[1]), lambda0, params.physics).H_over_R
        for x_val, y_val in zip(sol.t, sol.y.T)
    ]
    return FlowMapResult(
        y_buffer=np.asarray(sol.y[:, -1], dtype=float),
        branch=branch,
        success=True,
        message=str(sol.message),
        nfev=int(sol.nfev),
        n_steps=int(sol.y.shape[1]),
        max_HR=float(np.max(h_over_r)) if h_over_r else np.nan,
    )


def integrate_flow_map_nodes(
    logR_son: float,
    y_s: np.ndarray,
    lambda0: float,
    params: FlowmapParams,
    branch: SonicDerivativeBranch,
    logR_nodes: np.ndarray,
) -> np.ndarray:
    nodes = np.asarray(logR_nodes, dtype=float)
    if nodes.size == 0:
        return np.empty((0, 2), dtype=float)
    if not 0.0 < params.epsilon0 < params.epsilon_buf:
        raise ValueError("epsilon0 must be positive and smaller than epsilon_buf")
    x0 = logR_son + params.epsilon0
    if np.any(nodes <= x0):
        raise ValueError("flow-map target nodes must be beyond epsilon0")
    x1 = float(np.max(nodes))
    y0 = np.asarray(y_s, dtype=float) + params.epsilon0 * branch.gradient

    def rhs(x_value: float, y_value: np.ndarray) -> np.ndarray:
        return local_ode_rhs(float(x_value), y_value, lambda0, params.physics)

    sol = solve_ivp(
        rhs,
        (x0, x1),
        y0,
        method=params.flow_method,
        rtol=params.flow_rtol,
        atol=params.flow_atol,
        dense_output=True,
    )
    if (not sol.success) or sol.sol is None:
        raise RuntimeError(sol.message)
    values = np.asarray(sol.sol(nodes).T, dtype=float)
    if values.shape != (len(nodes), 2) or not np.all(np.isfinite(values)):
        raise RuntimeError("flow-map node integration produced non-finite values")
    return values


def sonic_flow_map(logR_son: float, y_s: np.ndarray, lambda0: float, params: FlowmapParams) -> FlowMapResult:
    branch = select_branch(logR_son, np.asarray(y_s, dtype=float), lambda0, params)
    return integrate_flow_map(logR_son, y_s, lambda0, params, branch)


def sonic_flow_map_from_a(logR_son: float, y_s: np.ndarray, lambda0: float, params: FlowmapParams, a_value: float, kind: str = "smooth") -> FlowMapResult:
    return integrate_flow_map(logR_son, y_s, lambda0, params, branch_from_a(logR_son, y_s, lambda0, params, a_value, kind=kind))


def dynamic_target_at_buffer(dynamic_x: np.ndarray, dynamic_params, logR_son: float, epsilon_buf: float) -> np.ndarray:
    logu_i, logT_i, _logu_o, _logT_o, old_logR_son, _lambda0 = dynamic_unpack(dynamic_x, dynamic_params)
    old_logR_i = buffer_inner_grid(old_logR_son, dynamic_params)
    target_logR = logR_son + epsilon_buf
    return np.array(
        [
            pchip_extrap(old_logR_i, logu_i, np.array([target_logR], dtype=float))[0],
            pchip_extrap(old_logR_i, logT_i, np.array([target_logR], dtype=float))[0],
        ],
        dtype=float,
    )


def dynamic_unpack(x: np.ndarray, params):
    from run_transonic_two_domain_outer_extension import unpack_two_domain

    return unpack_two_domain(x, params)  # type: ignore[arg-type]


def seed_from_dynamic(dynamic_x: np.ndarray, dynamic_params, target_params: FlowmapParams) -> np.ndarray:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = dynamic_unpack(dynamic_x, dynamic_params)
    old_logR_i = buffer_inner_grid(logR_son, dynamic_params)
    old_logR_o = outer_grid(dynamic_params)  # type: ignore[arg-type]
    new_logR_i = flow_inner_grid(logR_son, target_params)
    new_logR_o = outer_grid(target_params)  # type: ignore[arg-type]
    new_logu_i = pchip_extrap(old_logR_i, logu_i, new_logR_i)
    new_logT_i = pchip_extrap(old_logR_i, logT_i, new_logR_i)
    new_logu_o = pchip_extrap(old_logR_o, logu_o, new_logR_o)
    new_logT_o = pchip_extrap(old_logR_o, logT_o, new_logR_o)
    new_logu_o[0] = float(new_logu_i[-1])
    new_logT_o[0] = float(new_logT_i[-1])
    y_s = np.array([logu_i[0], logT_i[0]], dtype=float)
    return pack_flowmap(y_s, new_logu_i, new_logT_i, new_logu_o, new_logT_o, logR_son, lambda0)


def seed_from_flowmap_boundary(dynamic_x: np.ndarray, dynamic_params, target_params: FlowmapParams) -> np.ndarray:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = dynamic_unpack(dynamic_x, dynamic_params)
    old_logR_i = buffer_inner_grid(logR_son, dynamic_params)
    old_logR_o = outer_grid(dynamic_params)  # type: ignore[arg-type]
    new_logR_i = flow_inner_grid(logR_son, target_params)
    new_logR_o = outer_grid(target_params)  # type: ignore[arg-type]
    y_s = np.array([logu_i[0], logT_i[0]], dtype=float)
    flow = sonic_flow_map(logR_son, y_s, lambda0, target_params)

    keep_inner = old_logR_i > new_logR_i[0] + 1.0e-11
    inner_x = np.concatenate([np.array([new_logR_i[0]], dtype=float), old_logR_i[keep_inner]])
    inner_y = np.vstack([flow.y_buffer, np.column_stack([logu_i[keep_inner], logT_i[keep_inner]])])
    order = np.argsort(inner_x)
    inner_x = inner_x[order]
    inner_y = inner_y[order]
    new_logu_i = pchip_extrap(inner_x, inner_y[:, 0], new_logR_i)
    new_logT_i = pchip_extrap(inner_x, inner_y[:, 1], new_logR_i)
    new_logu_i[0] = float(flow.y_buffer[0])
    new_logT_i[0] = float(flow.y_buffer[1])
    if MICRO_SEED_FLOWMAP and target_params.micro_R_rg > 0.0 and target_params.n_micro > 0:
        micro_nodes = new_logR_i[: target_params.n_micro + 1]
        try:
            micro_values = integrate_flow_map_nodes(logR_son, y_s, lambda0, target_params, flow.branch, micro_nodes)
            new_logu_i[: target_params.n_micro + 1] = micro_values[:, 0]
            new_logT_i[: target_params.n_micro + 1] = micro_values[:, 1]
        except Exception as exc:
            print(f"micro flow-map seed fallback to PCHIP: {exc}", flush=True)

    new_logu_o = pchip_extrap(old_logR_o, logu_o, new_logR_o)
    new_logT_o = pchip_extrap(old_logR_o, logT_o, new_logR_o)
    new_logu_o[0] = float(new_logu_i[-1])
    new_logT_o[0] = float(new_logT_i[-1])
    return pack_flowmap(y_s, new_logu_i, new_logT_i, new_logu_o, new_logT_o, logR_son, lambda0)


def seed_from_flowmap(old_x: np.ndarray, old_params: FlowmapParams, new_params: FlowmapParams) -> np.ndarray:
    y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_flowmap(old_x, old_params)
    old_logR_i = flow_inner_grid(logR_son, old_params)
    old_logR_o = outer_grid(old_params)  # type: ignore[arg-type]
    new_logR_i = flow_inner_grid(logR_son, new_params)
    new_logR_o = outer_grid(new_params)  # type: ignore[arg-type]
    new_logu_i, new_logT_i = remap_profile(old_logR_i, logu_i, logT_i, new_logR_i, lambda0, old_params)
    new_logu_o, new_logT_o = remap_profile(old_logR_o, logu_o, logT_o, new_logR_o, lambda0, old_params)
    new_logu_o[0] = float(new_logu_i[-1])
    new_logT_o[0] = float(new_logT_i[-1])
    return pack_flowmap(y_s, new_logu_i, new_logT_i, new_logu_o, new_logT_o, logR_son, lambda0)


def smooth_seed_from_dynamic(dynamic_x: np.ndarray, dynamic_params, target_params: FlowmapParams, seed_mode: str, a_value: float) -> np.ndarray:
    if seed_mode == "dynamic":
        base = seed_from_dynamic(dynamic_x, dynamic_params, target_params)
    elif seed_mode == "flowmap":
        base = seed_from_flowmap_boundary(dynamic_x, dynamic_params, target_params)
    else:
        raise ValueError(f"unknown smooth BVP seed mode {seed_mode!r}")
    return pack_smooth_flowmap(base, a_value)


def smooth_seed_from_smooth(old_x: np.ndarray, old_params: FlowmapParams, new_params: FlowmapParams) -> np.ndarray:
    y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0, a_value = unpack_smooth_flowmap(old_x, old_params)
    base_old = pack_flowmap(y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0)
    base_new = seed_from_flowmap(base_old, old_params, new_params)
    if MICRO_SEED_FLOWMAP and new_params.micro_R_rg > 0.0 and new_params.n_micro > 0:
        y_new, logu_i_new, logT_i_new, logu_o_new, logT_o_new, logR_son_new, lambda0_new = unpack_flowmap(base_new, new_params)
        logR_i_new = flow_inner_grid(logR_son_new, new_params)
        micro_nodes = logR_i_new[: new_params.n_micro + 1]
        branch = branch_from_a(logR_son_new, y_new, lambda0_new, new_params, a_value, kind="smooth")
        try:
            micro_values = integrate_flow_map_nodes(logR_son_new, y_new, lambda0_new, new_params, branch, micro_nodes)
            logu_i_new[: new_params.n_micro + 1] = micro_values[:, 0]
            logT_i_new[: new_params.n_micro + 1] = micro_values[:, 1]
            base_new = pack_flowmap(y_new, logu_i_new, logT_i_new, logu_o_new, logT_o_new, logR_son_new, lambda0_new)
        except Exception as exc:
            print(f"smooth micro flow-map seed fallback to remap: {exc}", flush=True)
    return pack_smooth_flowmap(base_new, a_value)


def apply_fit_seed(seed: np.ndarray, params: FlowmapParams, fit_row: dict[str, object]) -> np.ndarray:
    y_s, logu_i, logT_i, logu_o, logT_o, _logR_son, _lambda0 = unpack_flowmap(seed, params)
    y_s = np.array([float(fit_row["logu_s"]), float(fit_row["logT_s"])], dtype=float)
    logR_son = float(fit_row["logR_son"])
    lambda0 = float(fit_row["lambda0"])
    flow = sonic_flow_map(logR_son, y_s, lambda0, params)
    logu_i = np.asarray(logu_i, dtype=float).copy()
    logT_i = np.asarray(logT_i, dtype=float).copy()
    logu_i[0] = flow.y_buffer[0]
    logT_i[0] = flow.y_buffer[1]
    logu_o = np.asarray(logu_o, dtype=float).copy()
    logT_o = np.asarray(logT_o, dtype=float).copy()
    logu_o[0] = float(logu_i[-1])
    logT_o[0] = float(logT_i[-1])
    return pack_flowmap(y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0)


def compatibility_value(sonic, pivot: str) -> float:
    if pivot == "C1":
        return float(sonic.C1)
    if pivot == "C2":
        return float(sonic.C2)
    if pivot == "K":
        return float(sonic.compatibility)
    raise ValueError(f"unknown compatibility pivot {pivot!r}")


def adjugate_components(logR: float, y: np.ndarray, lambda0: float, physics, form: str = "scaled") -> tuple[float, np.ndarray]:
    if form == "raw":
        matrix, rhs = differential_matrix(logR, y, lambda0, physics)
    elif form == "scaled":
        matrix, rhs, _radial_scale, _energy_scale = scaled_differential_matrix(logR, y, lambda0, physics)
    else:
        raise ValueError(f"unknown adjugate form {form!r}")
    a11, a12 = float(matrix[0, 0]), float(matrix[0, 1])
    a21, a22 = float(matrix[1, 0]), float(matrix[1, 1])
    c1, c2 = float(rhs[0]), float(rhs[1])
    det = a11 * a22 - a12 * a21
    numerator = np.array([a22 * c1 - a12 * c2, a11 * c2 - a21 * c1], dtype=float)
    return float(det), numerator


def adjugate_lhopital_vector(
    logR: float,
    y: np.ndarray,
    g: np.ndarray,
    lambda0: float,
    physics,
    *,
    eps: float = 1.0e-5,
    form: str = "scaled",
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    det_plus, num_plus = adjugate_components(logR + eps, y + eps * g, lambda0, physics, form=form)
    det_minus, num_minus = adjugate_components(logR - eps, y - eps * g, lambda0, physics, form=form)
    ddet = (det_plus - det_minus) / (2.0 * eps)
    dnum = (num_plus - num_minus) / (2.0 * eps)
    raw = g * ddet + dnum
    scale = np.maximum(np.maximum(np.abs(g * ddet), np.abs(dnum)), 1.0e-300)
    return raw, raw / scale


def flowmap_residual(x: np.ndarray, params: FlowmapParams) -> np.ndarray:
    rows = []
    try:
        y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_flowmap(x, params)
        logR_i = flow_inner_grid(logR_son, params)
        logR_o = outer_grid(params)  # type: ignore[arg-type]
        if np.any(np.diff(logR_i) <= 0.0) or np.any(np.diff(logR_o) <= 0.0):
            raise ValueError("mapped radii must increase")
        sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
        rows.append(np.array([sonic.D, compatibility_value(sonic, params.compatibility_pivot)], dtype=float))
        flow = sonic_flow_map(logR_son, y_s, lambda0, params)
        rows.append((np.array([logu_i[0], logT_i[0]], dtype=float) - flow.y_buffer) / params.flow_match_scale)
        for idx in range(params.n_inner - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx))
        for idx in range(params.n_outer - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx))
        rows.append(np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float))
        rows.append(far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params))  # type: ignore[arg-type]
        return np.concatenate(rows)
    except Exception:
        return np.full(flowmap_size(params), 1.0e6)


def flowmap_sparsity(params: FlowmapParams):
    n_unknown = flowmap_size(params)
    n_rows = n_unknown
    pattern = lil_matrix((n_rows, n_unknown), dtype=int)
    ni = params.n_inner
    no = params.n_outer
    sonic_cols = (0, 1)
    iu = 2
    iT = iu + ni
    ou = iT + ni
    oT = ou + no
    logR_col = oT + no
    lambda_col = logR_col + 1
    row = 0

    for col in (*sonic_cols, logR_col, lambda_col):
        pattern[row : row + 2, col] = 1
    row += 2

    for col in (*sonic_cols, iu, iT, logR_col, lambda_col):
        pattern[row : row + 2, col] = 1
    row += 2

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
    return pattern.tocsr()


def flowmap_bounds(params: FlowmapParams) -> tuple[np.ndarray, np.ndarray]:
    lower_one, upper_one = state_bounds(params.physics)
    lower_logu = float(lower_one[0])
    upper_logu = float(upper_one[0])
    lower_logT = float(lower_one[params.physics.n_nodes])
    upper_logT = float(upper_one[params.physics.n_nodes])
    lower = np.concatenate(
        [
            np.array([lower_logu, lower_logT], dtype=float),
            np.full(params.n_inner, lower_logu),
            np.full(params.n_inner, lower_logT),
            np.full(params.n_outer, lower_logu),
            np.full(params.n_outer, lower_logT),
            np.array([lower_one[-2], lower_one[-1]], dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.array([upper_logu, upper_logT], dtype=float),
            np.full(params.n_inner, upper_logu),
            np.full(params.n_inner, upper_logT),
            np.full(params.n_outer, upper_logu),
            np.full(params.n_outer, upper_logT),
            np.array([upper_one[-2], upper_one[-1]], dtype=float),
        ]
    )
    return lower, upper


def smooth_flowmap_residual(x: np.ndarray, params: FlowmapParams) -> np.ndarray:
    rows = []
    try:
        y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0, a_value = unpack_smooth_flowmap(x, params)
        logR_i = flow_inner_grid(logR_son, params)
        logR_o = outer_grid(params)  # type: ignore[arg-type]
        if np.any(np.diff(logR_i) <= 0.0) or np.any(np.diff(logR_o) <= 0.0):
            raise ValueError("mapped radii must increase")
        sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
        g = branch_gradient_from_a(logR_son, y_s, lambda0, params.physics, a_value, form=params.lhopital_form)
        lhopital = sonic_lhopital_residual_form(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form=params.lhopital_form)
        rows.append(np.array([sonic.D, compatibility_value(sonic, params.compatibility_pivot), lhopital], dtype=float))
        flow = sonic_flow_map_from_a(logR_son, y_s, lambda0, params, a_value)
        rows.append((np.array([logu_i[0], logT_i[0]], dtype=float) - flow.y_buffer) / params.flow_match_scale)
        for idx in range(params.n_inner - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx))
        for idx in range(params.n_outer - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx))
        rows.append(np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float))
        rows.append(far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params))  # type: ignore[arg-type]
        return np.concatenate(rows)
    except Exception:
        return np.full(smooth_flowmap_size(params), 1.0e6)


def smooth_flowmap_sparsity(params: FlowmapParams):
    n_unknown = smooth_flowmap_size(params)
    n_rows = n_unknown
    pattern = lil_matrix((n_rows, n_unknown), dtype=int)
    ni = params.n_inner
    no = params.n_outer
    sonic_cols = (0, 1)
    iu = 2
    iT = iu + ni
    ou = iT + ni
    oT = ou + no
    logR_col = oT + no
    lambda_col = logR_col + 1
    a_col = lambda_col + 1
    row = 0

    for col in (*sonic_cols, logR_col, lambda_col, a_col):
        pattern[row : row + 3, col] = 1
    row += 3

    for col in (*sonic_cols, iu, iT, logR_col, lambda_col, a_col):
        pattern[row : row + 2, col] = 1
    row += 2

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
    return pattern.tocsr()


def smooth_flowmap_bounds(params: FlowmapParams) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = flowmap_bounds(params)
    center = 0.0 if params.branch_a_center is None else float(params.branch_a_center)
    lower = np.concatenate([lower, np.array([center - SMOOTH_BVP_A_HALF_WIDTH], dtype=float)])
    upper = np.concatenate([upper, np.array([center + SMOOTH_BVP_A_HALF_WIDTH], dtype=float)])
    return lower, upper


def solve_smooth_flowmap(seed: np.ndarray, params: FlowmapParams, max_nfev: int):
    lower, upper = smooth_flowmap_bounds(params)
    x0 = np.clip(np.asarray(seed, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: smooth_flowmap_residual(trial, params),
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


def solve_flowmap(seed: np.ndarray, params: FlowmapParams, max_nfev: int):
    lower, upper = flowmap_bounds(params)
    x0 = np.clip(np.asarray(seed, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: flowmap_residual(trial, params),
        x0,
        jac_sparsity=flowmap_sparsity(params),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=max_nfev,
    )


def flowmap_audit(label: str, x: np.ndarray, params: FlowmapParams, result=None) -> dict[str, object]:
    y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_flowmap(x, params)
    logR_i = flow_inner_grid(logR_son, params)
    logR_o = outer_grid(params)  # type: ignore[arg-type]
    sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
    flow = sonic_flow_map(logR_son, y_s, lambda0, params)
    flow_residual = np.array([logu_i[0], logT_i[0]], dtype=float) - flow.y_buffer
    regular = np.asarray(
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
    micro_count = params.n_micro if params.micro_R_rg > 0.0 and params.n_micro > 0 else 0
    micro = regular[:micro_count]
    post_micro = regular[micro_count:]
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
        "flow": float(np.max(np.abs(flow_residual))),
        "micro_R": float(np.max(np.abs(micro[:, 0]))) if len(micro) else 0.0,
        "micro_E": float(np.max(np.abs(micro[:, 1]))) if len(micro) else 0.0,
        "post_micro_R": float(np.max(np.abs(post_micro[:, 0]))) if len(post_micro) else 0.0,
        "post_micro_E": float(np.max(np.abs(post_micro[:, 1]))) if len(post_micro) else 0.0,
        "regular_R": float(np.max(np.abs(regular[:, 0]))) if len(regular) else 0.0,
        "regular_E": float(np.max(np.abs(regular[:, 1]))) if len(regular) else 0.0,
        "outer_R": float(np.max(np.abs(outer[:, 0]))),
        "outer_E": float(np.max(np.abs(outer[:, 1]))),
        "interface": float(np.max(np.abs(interface))),
        "far_omega": abs(float(far[0])),
        "far_energy": abs(float(far[1])),
    }
    physical = max(blocks.values())
    dominant = max(blocks, key=blocks.get)
    residual = flowmap_residual(x, params)
    return {
        "label": label,
        "stage": "seed" if result is None else "solve",
        "ratio": params.physics.mdot_edd_ratio,
        "epsilon0": params.epsilon0,
        "epsilon_buf": params.epsilon_buf,
        "branch": params.branch_index,
        "branch_a_center": params.branch_a_center if params.branch_a_center is not None else np.nan,
        "branch_a": float(flow.branch.a),
        "branch_kind": flow.branch.kind,
        "n_regular": params.n_regular,
        "n_inner": params.n_inner,
        "n_outer": params.n_outer,
        "R_far_rg": params.R_far_rg,
        "inner_grid_power": params.inner_grid_power,
        "micro_R_rg": params.micro_R_rg,
        "n_micro": params.n_micro,
        "micro_seed_flowmap": MICRO_SEED_FLOWMAP,
        "remap_method": REMAP_METHOD,
        "selected_max": float(np.max(np.abs(residual))),
        "physical_active": physical,
        "passes_science": bool(physical <= SCIENCE_LIMIT),
        "dominant": dominant,
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "flow_u": float(flow_residual[0]),
        "flow_T": float(flow_residual[1]),
        "flow": blocks["flow"],
        "micro_R": blocks["micro_R"],
        "micro_E": blocks["micro_E"],
        "post_micro_R": blocks["post_micro_R"],
        "post_micro_E": blocks["post_micro_E"],
        "regular_R": blocks["regular_R"],
        "regular_E": blocks["regular_E"],
        "outer_R": blocks["outer_R"],
        "outer_E": blocks["outer_E"],
        "interface": blocks["interface"],
        "far_omega": float(far[0]),
        "far_energy": float(far[1]),
        "smin_over_smax": float(sonic.smin_over_smax),
        "g_u": float(flow.branch.gradient[0]),
        "g_T": float(flow.branch.gradient[1]),
        "L_raw": float(flow.branch.lhopital_raw),
        "L_norm": float(flow.branch.lhopital_normalized),
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "logR_son": float(logR_son),
        "logu_s": float(y_s[0]),
        "logT_s": float(y_s[1]),
        "lambda0": float(lambda0),
        "int_adv": integrated_advective_fraction(combined_logu, combined_logT, combined_logR, lambda0, params),  # type: ignore[arg-type]
        "max_HR": float(np.max(H_over_R)),
        "flow_max_HR": float(flow.max_HR),
        "flow_nfev": int(flow.nfev),
        "flow_steps": int(flow.n_steps),
        "nfev": int(result.nfev) if result is not None else 0,
        "success": bool(result.success) if result is not None else True,
        "message": str(result.message) if result is not None else "seed evaluation",
        "x": np.asarray(x, dtype=float),
    }


def smooth_flowmap_audit(label: str, x: np.ndarray, params: FlowmapParams, result=None) -> dict[str, object]:
    y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0, a_value = unpack_smooth_flowmap(x, params)
    logR_i = flow_inner_grid(logR_son, params)
    logR_o = outer_grid(params)  # type: ignore[arg-type]
    sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
    g = branch_gradient_from_a(logR_son, y_s, lambda0, params.physics, a_value, form=params.lhopital_form)
    lhopital = sonic_lhopital_residual_form(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form=params.lhopital_form)
    adj_raw, adj_norm = adjugate_lhopital_vector(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form="scaled")
    flow = sonic_flow_map_from_a(logR_son, y_s, lambda0, params, a_value)
    flow_residual = np.array([logu_i[0], logT_i[0]], dtype=float) - flow.y_buffer
    regular = np.asarray(
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
    micro_count = params.n_micro if params.micro_R_rg > 0.0 and params.n_micro > 0 else 0
    micro = regular[:micro_count]
    post_micro = regular[micro_count:]
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
        "post_micro_R": float(np.max(np.abs(post_micro[:, 0]))) if len(post_micro) else 0.0,
        "post_micro_E": float(np.max(np.abs(post_micro[:, 1]))) if len(post_micro) else 0.0,
        "regular_R": float(np.max(np.abs(regular[:, 0]))) if len(regular) else 0.0,
        "regular_E": float(np.max(np.abs(regular[:, 1]))) if len(regular) else 0.0,
        "outer_R": float(np.max(np.abs(outer[:, 0]))),
        "outer_E": float(np.max(np.abs(outer[:, 1]))),
        "interface": float(np.max(np.abs(interface))),
        "far_omega": abs(float(far[0])),
        "far_energy": abs(float(far[1])),
    }
    physical = max(blocks.values())
    dominant = max(blocks, key=blocks.get)
    residual = smooth_flowmap_residual(x, params)
    return {
        "label": label,
        "stage": "seed" if result is None else "solve",
        "ratio": params.physics.mdot_edd_ratio,
        "epsilon0": params.epsilon0,
        "epsilon_buf": params.epsilon_buf,
        "branch": params.branch_index,
        "n_regular": params.n_regular,
        "n_inner": params.n_inner,
        "n_outer": params.n_outer,
        "R_far_rg": params.R_far_rg,
        "inner_grid_power": params.inner_grid_power,
        "micro_R_rg": params.micro_R_rg,
        "n_micro": params.n_micro,
        "micro_seed_flowmap": MICRO_SEED_FLOWMAP,
        "remap_method": REMAP_METHOD,
        "selected_max": float(np.max(np.abs(residual))),
        "physical_active": physical,
        "passes_science": bool(physical <= SCIENCE_LIMIT),
        "dominant": dominant,
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "L": float(lhopital),
        "adj_u": float(adj_norm[0]),
        "adj_T": float(adj_norm[1]),
        "adj_u_raw": float(adj_raw[0]),
        "adj_T_raw": float(adj_raw[1]),
        "flow_u": float(flow_residual[0]),
        "flow_T": float(flow_residual[1]),
        "flow": blocks["flow"],
        "micro_R": blocks["micro_R"],
        "micro_E": blocks["micro_E"],
        "post_micro_R": blocks["post_micro_R"],
        "post_micro_E": blocks["post_micro_E"],
        "regular_R": blocks["regular_R"],
        "regular_E": blocks["regular_E"],
        "outer_R": blocks["outer_R"],
        "outer_E": blocks["outer_E"],
        "interface": blocks["interface"],
        "far_omega": float(far[0]),
        "far_energy": float(far[1]),
        "smin_over_smax": float(sonic.smin_over_smax),
        "a": float(a_value),
        "g_u": float(g[0]),
        "g_T": float(g[1]),
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "logR_son": float(logR_son),
        "logu_s": float(y_s[0]),
        "logT_s": float(y_s[1]),
        "lambda0": float(lambda0),
        "int_adv": integrated_advective_fraction(combined_logu, combined_logT, combined_logR, lambda0, params),  # type: ignore[arg-type]
        "max_HR": float(np.max(H_over_R)),
        "flow_max_HR": float(flow.max_HR),
        "flow_nfev": int(flow.nfev),
        "flow_steps": int(flow.n_steps),
        "nfev": int(result.nfev) if result is not None else 0,
        "success": bool(result.success) if result is not None else True,
        "message": str(result.message) if result is not None else "seed evaluation",
        "x": np.asarray(x, dtype=float),
    }


def save_checkpoint(label: str, x: np.ndarray, row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in row.items() if key != "x"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{label}_0p90277664.npz",
        x=np.asarray(x, dtype=float),
        row_json=np.array(row_json(payload)),
    )


def load_smooth_checkpoint(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, FlowmapParams, dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    row = json.loads(str(data["row_json"]))
    params = make_flowmap_params(
        fiducial,
        float(row["ratio"]),
        mdot_edd,
        int(row["n_regular"]),
        int(row["n_outer"]),
        float(row["R_far_rg"]),
        float(row["epsilon_buf"]),
        int(row["branch"]),
        float(row.get("a", 0.0)),
    )
    params = replace(params, inner_grid_power=float(row.get("inner_grid_power", params.inner_grid_power)))
    params = replace(
        params,
        micro_R_rg=float(row.get("micro_R_rg", params.micro_R_rg)),
        n_micro=int(row.get("n_micro", params.n_micro)),
    )
    return np.asarray(data["x"], dtype=float), params, row


def initial_branches(dynamic_x: np.ndarray, dynamic_params, physics) -> tuple[SonicDerivativeBranch, ...]:
    logu_i, logT_i, _logu_o, _logT_o, logR_son, lambda0 = dynamic_unpack(dynamic_x, dynamic_params)
    y_s = np.array([logu_i[0], logT_i[0]], dtype=float)
    return sonic_derivative_branches(
        logR_son,
        y_s,
        lambda0,
        physics,
        eps=LHOPITAL_EPS,
        form=LHOPITAL_FORM,
        half_width=ROOT_SCAN_HALF_WIDTH,
        scan_points=ROOT_SCAN_POINTS,
    )


def branch_audit_rows(dynamic_x: np.ndarray, dynamic_params, ratio: float, fiducial: FiducialParams, mdot_edd: float, epsilon_bufs: tuple[float, ...]) -> list[dict[str, object]]:
    logu_i, logT_i, _logu_o, _logT_o, logR_son, lambda0 = dynamic_unpack(dynamic_x, dynamic_params)
    y_s = np.array([logu_i[0], logT_i[0]], dtype=float)
    rows = []
    for epsilon_buf in epsilon_bufs:
        probe_params = make_flowmap_params(
            fiducial,
            ratio,
            mdot_edd,
            int(dynamic_params.n_regular),
            int(dynamic_params.n_outer),
            float(dynamic_params.R_far_rg),
            epsilon_buf,
            0,
            None,
        )
        branches = initial_branches(dynamic_x, dynamic_params, probe_params.physics)
        target = dynamic_target_at_buffer(dynamic_x, dynamic_params, logR_son, epsilon_buf)
        for idx, branch in enumerate(branches):
            params = make_flowmap_params(
                fiducial,
                ratio,
                mdot_edd,
                int(dynamic_params.n_regular),
                int(dynamic_params.n_outer),
                float(dynamic_params.R_far_rg),
                epsilon_buf,
                idx,
                float(branch.a),
            )
            try:
                flow = sonic_flow_map(logR_son, y_s, lambda0, params)
                delta = flow.y_buffer - target
                rows.append(
                    {
                        "epsilon0": EPSILON0,
                        "epsilon_buf": epsilon_buf,
                        "branch": idx,
                        "branch_a": float(flow.branch.a),
                        "kind": flow.branch.kind,
                        "success": True,
                        "message": flow.message,
                        "g_u": float(flow.branch.gradient[0]),
                        "g_T": float(flow.branch.gradient[1]),
                        "L_raw": float(flow.branch.lhopital_raw),
                        "L_norm": float(flow.branch.lhopital_normalized),
                        "flow_logu": float(flow.y_buffer[0]),
                        "flow_logT": float(flow.y_buffer[1]),
                        "target_logu": float(target[0]),
                        "target_logT": float(target[1]),
                        "dlogu_target": float(delta[0]),
                        "dlogT_target": float(delta[1]),
                        "dist_target": float(np.linalg.norm(delta)),
                        "flow_nfev": int(flow.nfev),
                        "flow_steps": int(flow.n_steps),
                        "flow_max_HR": float(flow.max_HR),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "epsilon0": EPSILON0,
                        "epsilon_buf": epsilon_buf,
                        "branch": idx,
                        "branch_a": float(branch.a),
                        "kind": branch.kind,
                        "success": False,
                        "message": str(exc),
                        "g_u": float(branch.gradient[0]),
                        "g_T": float(branch.gradient[1]),
                        "L_raw": float(branch.lhopital_raw),
                        "L_norm": float(branch.lhopital_normalized),
                        "flow_logu": np.nan,
                        "flow_logT": np.nan,
                        "target_logu": float(target[0]),
                        "target_logT": float(target[1]),
                        "dlogu_target": np.nan,
                        "dlogT_target": np.nan,
                        "dist_target": np.inf,
                        "flow_nfev": 0,
                        "flow_steps": 0,
                        "flow_max_HR": np.nan,
                    }
                )
    return rows


def adjugate_audit_rows(dynamic_x: np.ndarray, dynamic_params, ratio: float, fiducial: FiducialParams, mdot_edd: float, epsilon_buf: float) -> list[dict[str, object]]:
    logu_i, logT_i, _logu_o, _logT_o, logR_son, lambda0 = dynamic_unpack(dynamic_x, dynamic_params)
    y_s = np.array([logu_i[0], logT_i[0]], dtype=float)
    params = make_flowmap_params(
        fiducial,
        ratio,
        mdot_edd,
        int(dynamic_params.n_regular),
        int(dynamic_params.n_outer),
        float(dynamic_params.R_far_rg),
        epsilon_buf,
        0,
        None,
    )
    target = dynamic_target_at_buffer(dynamic_x, dynamic_params, logR_son, epsilon_buf)
    a_values = np.linspace(-ADJUGATE_SCAN_HALF_WIDTH, ADJUGATE_SCAN_HALF_WIDTH, ADJUGATE_SCAN_POINTS)

    def adj_norm_from_a(a_value: float) -> np.ndarray:
        g = branch_gradient_from_a(logR_son, y_s, lambda0, params.physics, a_value, form=params.lhopital_form)
        _raw, normalized = adjugate_lhopital_vector(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form="scaled")
        return normalized

    values = np.full((len(a_values), 2), np.nan, dtype=float)
    for idx, a_value in enumerate(a_values):
        try:
            values[idx] = adj_norm_from_a(float(a_value))
        except Exception:
            values[idx] = np.nan

    candidates: list[tuple[str, float]] = []
    for component, name in enumerate(("adj_u", "adj_T")):
        for idx in range(len(a_values) - 1):
            left_f = float(values[idx, component])
            right_f = float(values[idx + 1, component])
            if not np.isfinite(left_f) or not np.isfinite(right_f):
                continue
            if left_f == 0.0:
                candidates.append((f"{name}_root", float(a_values[idx])))
            elif left_f * right_f <= 0.0:
                try:
                    root_a = float(brentq(lambda a: float(adj_norm_from_a(float(a))[component]), float(a_values[idx]), float(a_values[idx + 1]), xtol=1.0e-9, rtol=1.0e-9, maxiter=80))
                except ValueError:
                    continue
                candidates.append((f"{name}_root", root_a))

    finite = np.all(np.isfinite(values), axis=1)
    if np.any(finite):
        scores = np.max(np.abs(values[finite]), axis=1)
        finite_indices = np.flatnonzero(finite)
        best_idx = int(finite_indices[int(np.argmin(scores))])
        left = float(a_values[max(0, best_idx - 5)])
        right = float(a_values[min(len(a_values) - 1, best_idx + 5)])
        if right > left:
            try:
                minimum = minimize_scalar(
                    lambda a: float(np.max(np.abs(adj_norm_from_a(float(a))))),
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": 1.0e-8},
                )
                if minimum.success:
                    candidates.append(("adj_norm_min", float(minimum.x)))
            except ValueError:
                pass

    unique: list[tuple[str, float]] = []
    for kind, a_value in sorted(candidates, key=lambda item: item[1]):
        if unique and abs(a_value - unique[-1][1]) < 1.0e-5:
            if kind == "adj_norm_min":
                unique[-1] = (kind, a_value)
            continue
        unique.append((kind, a_value))

    rows: list[dict[str, object]] = []
    for kind, a_value in unique:
        g = branch_gradient_from_a(logR_son, y_s, lambda0, params.physics, a_value, form=params.lhopital_form)
        adj_raw, adj_norm = adjugate_lhopital_vector(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form="scaled")
        svd_L = sonic_lhopital_residual_form(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form=params.lhopital_form)
        try:
            flow = sonic_flow_map_from_a(logR_son, y_s, lambda0, params, a_value, kind=kind)
            delta = flow.y_buffer - target
            success = True
            message = flow.message
            flow_nfev = flow.nfev
            flow_steps = flow.n_steps
            flow_max_HR = flow.max_HR
            dist_target = float(np.linalg.norm(delta))
            dlogu_target = float(delta[0])
            dlogT_target = float(delta[1])
        except Exception as exc:
            success = False
            message = str(exc)
            flow_nfev = 0
            flow_steps = 0
            flow_max_HR = np.nan
            dist_target = np.inf
            dlogu_target = np.nan
            dlogT_target = np.nan
        rows.append(
            {
                "kind": kind,
                "epsilon_buf": epsilon_buf,
                "a": float(a_value),
                "g_u": float(g[0]),
                "g_T": float(g[1]),
                "svd_L": float(svd_L),
                "adj_u": float(adj_norm[0]),
                "adj_T": float(adj_norm[1]),
                "adj_u_raw": float(adj_raw[0]),
                "adj_T_raw": float(adj_raw[1]),
                "adj_norm": float(np.max(np.abs(adj_norm))),
                "success": success,
                "message": message,
                "dist_target": dist_target,
                "dlogu_target": dlogu_target,
                "dlogT_target": dlogT_target,
                "flow_nfev": flow_nfev,
                "flow_steps": flow_steps,
                "flow_max_HR": flow_max_HR,
            }
        )
    return rows


def fit_local_flowmap(
    dynamic_x: np.ndarray,
    dynamic_params,
    params: FlowmapParams,
    branch_a: float,
):
    logu_i, logT_i, _logu_o, _logT_o, logR_seed, lambda_seed = dynamic_unpack(dynamic_x, dynamic_params)
    y_seed = np.array([logu_i[0], logT_i[0]], dtype=float)
    lower_one, upper_one = state_bounds(params.physics)
    rson_seed_rg = float(np.exp(logR_seed) / params.r_g)
    rson_low_rg = max(params.physics.R_son_bounds_rg[0], rson_seed_rg - 0.6)
    rson_high_rg = min(params.physics.R_son_bounds_rg[1], rson_seed_rg + 0.6)
    lower = np.array(
        [
            np.log(rson_low_rg * params.r_g),
            max(float(lower_one[0]), float(y_seed[0] - 2.0)),
            max(float(lower_one[params.physics.n_nodes]), float(y_seed[1] - 2.0)),
            max(float(lower_one[-1]), float(lambda_seed - 0.08)),
        ],
        dtype=float,
    )
    upper = np.array(
        [
            np.log(rson_high_rg * params.r_g),
            min(float(upper_one[0]), float(y_seed[0] + 2.0)),
            min(float(upper_one[params.physics.n_nodes]), float(y_seed[1] + 2.0)),
            min(float(upper_one[-1]), float(lambda_seed + 0.08)),
        ],
        dtype=float,
    )
    x0 = np.clip(np.array([logR_seed, y_seed[0], y_seed[1], lambda_seed], dtype=float), lower + 1.0e-12, upper - 1.0e-12)

    def residual(vars_free: np.ndarray) -> np.ndarray:
        logR_son = float(vars_free[0])
        y_s = np.array([vars_free[1], vars_free[2]], dtype=float)
        lambda0 = float(vars_free[3])
        target_current = dynamic_target_at_buffer(dynamic_x, dynamic_params, logR_son, params.epsilon_buf)
        trial_params = FlowmapParams(**{**params.__dict__, "branch_a_center": float(branch_a)})
        try:
            sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
            flow = sonic_flow_map(logR_son, y_s, lambda0, trial_params)
            return np.array(
                [
                    sonic.D,
                    compatibility_value(sonic, params.compatibility_pivot),
                    flow.y_buffer[0] - target_current[0],
                    flow.y_buffer[1] - target_current[1],
                ],
                dtype=float,
            )
        except Exception:
            return np.full(4, 1.0e6)

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV_FIT,
    )
    target = dynamic_target_at_buffer(dynamic_x, dynamic_params, float(result.x[0]), params.epsilon_buf)
    return result, target, residual(result.x)


def fit_audit_row(label: str, dynamic_x: np.ndarray, dynamic_params, params: FlowmapParams, branch_a: float, result, target: np.ndarray, residual: np.ndarray) -> dict[str, object]:
    logR_son = float(result.x[0])
    y_s = np.array([result.x[1], result.x[2]], dtype=float)
    lambda0 = float(result.x[3])
    trial_params = FlowmapParams(**{**params.__dict__, "branch_a_center": float(branch_a)})
    sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
    flow = sonic_flow_map(logR_son, y_s, lambda0, trial_params)
    dynamic_logu_i, dynamic_logT_i, _logu_o, _logT_o, dynamic_logR, dynamic_lambda = dynamic_unpack(dynamic_x, dynamic_params)
    flow_delta = flow.y_buffer - target
    blocks = {
        "D": abs(float(sonic.D)),
        params.compatibility_pivot: abs(compatibility_value(sonic, params.compatibility_pivot)),
        "flow": float(np.max(np.abs(flow_delta))),
    }
    return {
        "label": label,
        "epsilon_buf": params.epsilon_buf,
        "branch": params.branch_index,
        "branch_a_center": float(branch_a),
        "branch_a": float(flow.branch.a),
        "physical_active": max(blocks.values()),
        "selected_max": float(np.max(np.abs(residual))),
        "dominant": max(blocks, key=blocks.get),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "flow_u": float(flow_delta[0]),
        "flow_T": float(flow_delta[1]),
        "flow": float(np.max(np.abs(flow_delta))),
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "delta_Rson_rg": float(np.exp(logR_son) / params.r_g - np.exp(dynamic_logR) / params.r_g),
        "logR_son": logR_son,
        "logu_s": float(y_s[0]),
        "logT_s": float(y_s[1]),
        "delta_logu_s": float(y_s[0] - dynamic_logu_i[0]),
        "delta_logT_s": float(y_s[1] - dynamic_logT_i[0]),
        "lambda0": lambda0,
        "delta_lambda0": float(lambda0 - dynamic_lambda),
        "target_logu": float(target[0]),
        "target_logT": float(target[1]),
        "g_u": float(flow.branch.gradient[0]),
        "g_T": float(flow.branch.gradient[1]),
        "flow_nfev": int(flow.nfev),
        "flow_steps": int(flow.n_steps),
    }


def fit_local_flowmap_smooth(
    dynamic_x: np.ndarray,
    dynamic_params,
    params: FlowmapParams,
    branch_a: float,
    regularity: str,
):
    logu_i, logT_i, _logu_o, _logT_o, logR_seed, lambda_seed = dynamic_unpack(dynamic_x, dynamic_params)
    y_seed = np.array([logu_i[0], logT_i[0]], dtype=float)
    lower_one, upper_one = state_bounds(params.physics)
    rson_seed_rg = float(np.exp(logR_seed) / params.r_g)
    rson_low_rg = max(params.physics.R_son_bounds_rg[0], rson_seed_rg - 0.8)
    rson_high_rg = min(params.physics.R_son_bounds_rg[1], rson_seed_rg + 0.8)
    lower = np.array(
        [
            np.log(rson_low_rg * params.r_g),
            max(float(lower_one[0]), float(y_seed[0] - 2.5)),
            max(float(lower_one[params.physics.n_nodes]), float(y_seed[1] - 2.5)),
            max(float(lower_one[-1]), float(lambda_seed - 0.12)),
            branch_a - SMOOTH_A_HALF_WIDTH,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            np.log(rson_high_rg * params.r_g),
            min(float(upper_one[0]), float(y_seed[0] + 2.5)),
            min(float(upper_one[params.physics.n_nodes]), float(y_seed[1] + 2.5)),
            min(float(upper_one[-1]), float(lambda_seed + 0.12)),
            branch_a + SMOOTH_A_HALF_WIDTH,
        ],
        dtype=float,
    )
    x0 = np.clip(np.array([logR_seed, y_seed[0], y_seed[1], lambda_seed, branch_a], dtype=float), lower + 1.0e-12, upper - 1.0e-12)

    def regularity_residual(logR_son: float, y_s: np.ndarray, lambda0: float, a_value: float) -> float:
        g = branch_gradient_from_a(logR_son, y_s, lambda0, params.physics, a_value, form=params.lhopital_form)
        if regularity == "svd":
            return sonic_lhopital_residual_form(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form=params.lhopital_form)
        raw, normalized = adjugate_lhopital_vector(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form="scaled")
        _ = raw
        if regularity == "adj_u":
            return float(normalized[0])
        if regularity == "adj_T":
            return float(normalized[1])
        raise ValueError(f"unknown smooth regularity {regularity!r}")

    def residual(vars_free: np.ndarray) -> np.ndarray:
        logR_son = float(vars_free[0])
        y_s = np.array([vars_free[1], vars_free[2]], dtype=float)
        lambda0 = float(vars_free[3])
        a_value = float(vars_free[4])
        target_current = dynamic_target_at_buffer(dynamic_x, dynamic_params, logR_son, params.epsilon_buf)
        try:
            sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
            flow = sonic_flow_map_from_a(logR_son, y_s, lambda0, params, a_value)
            return np.array(
                [
                    sonic.D,
                    compatibility_value(sonic, params.compatibility_pivot),
                    regularity_residual(logR_son, y_s, lambda0, a_value),
                    flow.y_buffer[0] - target_current[0],
                    flow.y_buffer[1] - target_current[1],
                ],
                dtype=float,
            )
        except Exception:
            return np.full(5, 1.0e6)

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV_SMOOTH_FIT,
    )
    target = dynamic_target_at_buffer(dynamic_x, dynamic_params, float(result.x[0]), params.epsilon_buf)
    return result, target, residual(result.x)


def smooth_fit_audit_row(
    label: str,
    dynamic_x: np.ndarray,
    dynamic_params,
    params: FlowmapParams,
    regularity: str,
    result,
    target: np.ndarray,
    residual: np.ndarray,
) -> dict[str, object]:
    logR_son = float(result.x[0])
    y_s = np.array([result.x[1], result.x[2]], dtype=float)
    lambda0 = float(result.x[3])
    a_value = float(result.x[4])
    sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
    g = branch_gradient_from_a(logR_son, y_s, lambda0, params.physics, a_value, form=params.lhopital_form)
    flow = sonic_flow_map_from_a(logR_son, y_s, lambda0, params, a_value)
    adj_raw, adj_norm = adjugate_lhopital_vector(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form="scaled")
    svd_L = sonic_lhopital_residual_form(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form=params.lhopital_form)
    dynamic_logu_i, dynamic_logT_i, _logu_o, _logT_o, dynamic_logR, dynamic_lambda = dynamic_unpack(dynamic_x, dynamic_params)
    flow_delta = flow.y_buffer - target
    blocks = {
        "D": abs(float(sonic.D)),
        params.compatibility_pivot: abs(compatibility_value(sonic, params.compatibility_pivot)),
        "regularity": abs(float(residual[2])),
        "flow": float(np.max(np.abs(flow_delta))),
    }
    return {
        "label": label,
        "regularity": regularity,
        "epsilon_buf": params.epsilon_buf,
        "branch": params.branch_index,
        "physical_active": max(blocks.values()),
        "selected_max": float(np.max(np.abs(residual))),
        "dominant": max(blocks, key=blocks.get),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "regularity_residual": float(residual[2]),
        "svd_L": float(svd_L),
        "adj_u": float(adj_norm[0]),
        "adj_T": float(adj_norm[1]),
        "adj_u_raw": float(adj_raw[0]),
        "adj_T_raw": float(adj_raw[1]),
        "flow_u": float(flow_delta[0]),
        "flow_T": float(flow_delta[1]),
        "flow": float(np.max(np.abs(flow_delta))),
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "delta_Rson_rg": float(np.exp(logR_son) / params.r_g - np.exp(dynamic_logR) / params.r_g),
        "logR_son": logR_son,
        "logu_s": float(y_s[0]),
        "logT_s": float(y_s[1]),
        "delta_logu_s": float(y_s[0] - dynamic_logu_i[0]),
        "delta_logT_s": float(y_s[1] - dynamic_logT_i[0]),
        "lambda0": lambda0,
        "delta_lambda0": float(lambda0 - dynamic_lambda),
        "a": a_value,
        "g_u": float(g[0]),
        "g_T": float(g[1]),
        "flow_nfev": int(flow.nfev),
        "flow_steps": int(flow.n_steps),
    }


def fit_local_flowmap_constrained(
    dynamic_x: np.ndarray,
    dynamic_params,
    params: FlowmapParams,
    branch_a: float,
    regularity: str,
):
    logu_i, logT_i, _logu_o, _logT_o, logR_seed, lambda_seed = dynamic_unpack(dynamic_x, dynamic_params)
    y_seed = np.array([logu_i[0], logT_i[0]], dtype=float)
    lower_one, upper_one = state_bounds(params.physics)
    rson_seed_rg = float(np.exp(logR_seed) / params.r_g)
    rson_low_rg = max(params.physics.R_son_bounds_rg[0], rson_seed_rg - CONSTRAINED_RSON_HALF_WIDTH_RG)
    rson_high_rg = min(params.physics.R_son_bounds_rg[1], rson_seed_rg + CONSTRAINED_RSON_HALF_WIDTH_RG)
    lower = np.array(
        [
            np.log(rson_low_rg * params.r_g),
            max(float(lower_one[0]), float(y_seed[0] - CONSTRAINED_Y_HALF_WIDTH)),
            max(float(lower_one[params.physics.n_nodes]), float(y_seed[1] - CONSTRAINED_Y_HALF_WIDTH)),
            max(float(lower_one[-1]), float(lambda_seed - CONSTRAINED_LAMBDA_HALF_WIDTH)),
            branch_a - CONSTRAINED_A_HALF_WIDTH,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            np.log(rson_high_rg * params.r_g),
            min(float(upper_one[0]), float(y_seed[0] + CONSTRAINED_Y_HALF_WIDTH)),
            min(float(upper_one[params.physics.n_nodes]), float(y_seed[1] + CONSTRAINED_Y_HALF_WIDTH)),
            min(float(upper_one[-1]), float(lambda_seed + CONSTRAINED_LAMBDA_HALF_WIDTH)),
            branch_a + CONSTRAINED_A_HALF_WIDTH,
        ],
        dtype=float,
    )
    x0 = np.clip(np.array([logR_seed, y_seed[0], y_seed[1], lambda_seed, branch_a], dtype=float), lower + 1.0e-12, upper - 1.0e-12)

    def regularity_residual(logR_son: float, y_s: np.ndarray, lambda0: float, a_value: float) -> float:
        g = branch_gradient_from_a(logR_son, y_s, lambda0, params.physics, a_value, form=params.lhopital_form)
        if regularity == "svd":
            return sonic_lhopital_residual_form(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form=params.lhopital_form)
        _raw, normalized = adjugate_lhopital_vector(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form="scaled")
        if regularity == "adj_u":
            return float(normalized[0])
        if regularity == "adj_T":
            return float(normalized[1])
        raise ValueError(f"unknown constrained regularity {regularity!r}")

    def unweighted(vars_free: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        logR_son = float(vars_free[0])
        y_s = np.array([vars_free[1], vars_free[2]], dtype=float)
        lambda0 = float(vars_free[3])
        a_value = float(vars_free[4])
        target_current = dynamic_target_at_buffer(dynamic_x, dynamic_params, logR_son, params.epsilon_buf)
        sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
        g = branch_gradient_from_a(logR_son, y_s, lambda0, params.physics, a_value, form=params.lhopital_form)
        flow = sonic_flow_map_from_a(logR_son, y_s, lambda0, params, a_value)
        flow_delta = flow.y_buffer - target_current
        reg = regularity_residual(logR_son, y_s, lambda0, a_value)
        g_hinge = np.maximum(0.0, np.abs(g) - CONSTRAINED_G_LIMIT) / CONSTRAINED_G_LIMIT
        rows = np.concatenate(
            [
                np.array([sonic.D, sonic.C1, sonic.C2, sonic.compatibility, reg], dtype=float),
                flow_delta,
                g_hinge,
            ]
        )
        aux = {"sonic": sonic, "g": g, "flow": flow, "flow_delta": flow_delta, "regularity": reg, "g_hinge": g_hinge}
        return rows, aux

    def residual(vars_free: np.ndarray) -> np.ndarray:
        try:
            rows, _aux = unweighted(vars_free)
            return np.concatenate(
                [
                    CONSTRAINED_COMPAT_WEIGHT * rows[:4],
                    CONSTRAINED_REG_WEIGHT * rows[4:5],
                    CONSTRAINED_FLOW_WEIGHT * rows[5:7],
                    CONSTRAINED_G_WEIGHT * rows[7:9],
                ]
            )
        except Exception:
            return np.full(9, 1.0e6)

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV_CONSTRAINED_FIT,
    )
    target = dynamic_target_at_buffer(dynamic_x, dynamic_params, float(result.x[0]), params.epsilon_buf)
    return result, target, residual(result.x), unweighted(result.x)


def constrained_fit_audit_row(
    label: str,
    dynamic_x: np.ndarray,
    dynamic_params,
    params: FlowmapParams,
    regularity: str,
    result,
    target: np.ndarray,
    weighted_residual: np.ndarray,
    unweighted_result: tuple[np.ndarray, dict[str, object]],
) -> dict[str, object]:
    rows, aux = unweighted_result
    logR_son = float(result.x[0])
    y_s = np.array([result.x[1], result.x[2]], dtype=float)
    lambda0 = float(result.x[3])
    a_value = float(result.x[4])
    sonic = aux["sonic"]
    g = np.asarray(aux["g"], dtype=float)
    flow = aux["flow"]
    flow_delta = np.asarray(aux["flow_delta"], dtype=float)
    g_hinge = np.asarray(aux["g_hinge"], dtype=float)
    adj_raw, adj_norm = adjugate_lhopital_vector(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form="scaled")
    svd_L = sonic_lhopital_residual_form(logR_son, y_s, g, lambda0, params.physics, eps=LHOPITAL_EPS, form=params.lhopital_form)
    dynamic_logu_i, dynamic_logT_i, _logu_o, _logT_o, dynamic_logR, dynamic_lambda = dynamic_unpack(dynamic_x, dynamic_params)
    blocks = {
        "D": abs(float(sonic.D)),
        "C1": abs(float(sonic.C1)),
        "C2": abs(float(sonic.C2)),
        "K": abs(float(sonic.compatibility)),
        "regularity": abs(float(rows[4])),
        "flow": float(np.max(np.abs(flow_delta))),
        "g_hinge": float(np.max(np.abs(g_hinge))),
    }
    return {
        "label": label,
        "regularity": regularity,
        "epsilon_buf": params.epsilon_buf,
        "branch": params.branch_index,
        "physical_active": max(blocks.values()),
        "weighted_max": float(np.max(np.abs(weighted_residual))),
        "dominant": max(blocks, key=blocks.get),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "regularity_residual": float(rows[4]),
        "svd_L": float(svd_L),
        "adj_u": float(adj_norm[0]),
        "adj_T": float(adj_norm[1]),
        "adj_u_raw": float(adj_raw[0]),
        "adj_T_raw": float(adj_raw[1]),
        "flow_u": float(flow_delta[0]),
        "flow_T": float(flow_delta[1]),
        "flow": float(np.max(np.abs(flow_delta))),
        "g_hinge": float(np.max(g_hinge)),
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "delta_Rson_rg": float(np.exp(logR_son) / params.r_g - np.exp(dynamic_logR) / params.r_g),
        "logR_son": logR_son,
        "logu_s": float(y_s[0]),
        "logT_s": float(y_s[1]),
        "delta_logu_s": float(y_s[0] - dynamic_logu_i[0]),
        "delta_logT_s": float(y_s[1] - dynamic_logT_i[0]),
        "lambda0": lambda0,
        "delta_lambda0": float(lambda0 - dynamic_lambda),
        "a": a_value,
        "g_u": float(g[0]),
        "g_T": float(g[1]),
        "flow_nfev": int(flow.nfev),
        "flow_steps": int(flow.n_steps),
        "target_logu": float(target[0]),
        "target_logT": float(target[1]),
    }


def write_branch_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sonic Flow-Map Branch Audit",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_flowmap.py`.",
        "",
        "| eps0 | eps buf | branch | a | kind | success | dist target | dlogu | dlogT | g_u | g_T | L raw | L norm | flow max H/R | flow nfev | message |",
        "|---:|---:|---:|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {epsilon0} | {epsilon_buf} | {branch} | {branch_a} | {kind} | {success} | {dist_target} | {dlogu_target} | "
            "{dlogT_target} | {g_u} | {g_T} | {L_raw} | {L_norm} | {flow_max_HR} | {flow_nfev} | {message} |".format(
                epsilon0=fmt(float(row["epsilon0"])),
                epsilon_buf=fmt(float(row["epsilon_buf"])),
                branch=row["branch"],
                branch_a=fmt(float(row["branch_a"])),
                kind=row["kind"],
                success="yes" if row["success"] else "no",
                dist_target=fmt(float(row["dist_target"])),
                dlogu_target=fmt(float(row["dlogu_target"])),
                dlogT_target=fmt(float(row["dlogT_target"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                L_raw=fmt(float(row["L_raw"])),
                L_norm=fmt(float(row["L_norm"])),
                flow_max_HR=fmt(float(row["flow_max_HR"])),
                flow_nfev=row["flow_nfev"],
                message=str(row["message"]).replace("|", "/"),
            )
        )
    BRANCH_TABLE.write_text("\n".join(lines) + "\n")


def write_adjugate_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sonic Adjugate L'Hopital Audit",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_flowmap.py`.",
        "",
        "The adjugate check uses `g = -adj(A)c / det(A)` and audits the quotient condition `g det' + N' = 0` along `g = g_p + a r`.",
        "",
        "| kind | eps buf | a | g_u | g_T | adj norm | adj u | adj T | SVD L | success | dist target | dlogu | dlogT | flow max H/R | flow nfev | message |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {kind} | {epsilon_buf} | {a} | {g_u} | {g_T} | {adj_norm} | {adj_u} | {adj_T} | {svd_L} | {success} | "
            "{dist_target} | {dlogu_target} | {dlogT_target} | {flow_max_HR} | {flow_nfev} | {message} |".format(
                kind=row["kind"],
                epsilon_buf=fmt(float(row["epsilon_buf"])),
                a=fmt(float(row["a"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                adj_norm=fmt(float(row["adj_norm"])),
                adj_u=fmt(float(row["adj_u"])),
                adj_T=fmt(float(row["adj_T"])),
                svd_L=fmt(float(row["svd_L"])),
                success="yes" if row["success"] else "no",
                dist_target=fmt(float(row["dist_target"])),
                dlogu_target=fmt(float(row["dlogu_target"])),
                dlogT_target=fmt(float(row["dlogT_target"])),
                flow_max_HR=fmt(float(row["flow_max_HR"])),
                flow_nfev=row["flow_nfev"],
                message=str(row["message"]).replace("|", "/"),
            )
        )
    ADJUGATE_TABLE.write_text("\n".join(lines) + "\n")


def write_fit_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sonic Flow-Map Local Fit",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_flowmap.py`.",
        "",
        "| label | eps buf | branch | physical | selected | dominant | D | C1 | C2 | K | flow | Rson/rg | dRson | lambda0 | dlambda | dlogu_s | dlogT_s | g_u | g_T | nfev | success | message |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {epsilon_buf} | {branch} | {physical_active} | {selected_max} | {dominant} | {D} | {C1} | {C2} | {K} | "
            "{flow} | {Rson_rg} | {delta_Rson_rg} | {lambda0} | {delta_lambda0} | {delta_logu_s} | {delta_logT_s} | "
            "{g_u} | {g_T} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                epsilon_buf=fmt(float(row["epsilon_buf"])),
                branch=row["branch"],
                physical_active=fmt(float(row["physical_active"])),
                selected_max=fmt(float(row["selected_max"])),
                dominant=row["dominant"],
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                flow=fmt(float(row["flow"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row["delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row["delta_lambda0"])),
                delta_logu_s=fmt(float(row["delta_logu_s"])),
                delta_logT_s=fmt(float(row["delta_logT_s"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    FIT_TABLE.write_text("\n".join(lines) + "\n")


def write_smooth_fit_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sonic Flow-Map Smooth Local Fit",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_flowmap.py`.",
        "",
        "This fit promotes the branch coordinate `a` to a continuous unknown instead of rescanning roots inside the residual.",
        "",
        "| label | regularity | eps buf | branch | physical | selected | dominant | D | C1 | C2 | K | reg | SVD L | adj u | adj T | flow | Rson/rg | dRson | lambda0 | dlambda | a | g_u | g_T | nfev | success | message |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {regularity} | {epsilon_buf} | {branch} | {physical_active} | {selected_max} | {dominant} | "
            "{D} | {C1} | {C2} | {K} | {regularity_residual} | {svd_L} | {adj_u} | {adj_T} | {flow} | "
            "{Rson_rg} | {delta_Rson_rg} | {lambda0} | {delta_lambda0} | {a} | {g_u} | {g_T} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                regularity=row["regularity"],
                epsilon_buf=fmt(float(row["epsilon_buf"])),
                branch=row["branch"],
                physical_active=fmt(float(row["physical_active"])),
                selected_max=fmt(float(row["selected_max"])),
                dominant=row["dominant"],
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                regularity_residual=fmt(float(row["regularity_residual"])),
                svd_L=fmt(float(row["svd_L"])),
                adj_u=fmt(float(row["adj_u"])),
                adj_T=fmt(float(row["adj_T"])),
                flow=fmt(float(row["flow"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row["delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row["delta_lambda0"])),
                a=fmt(float(row["a"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    SMOOTH_FIT_TABLE.write_text("\n".join(lines) + "\n")


def write_constrained_fit_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sonic Flow-Map Constrained Local Fit",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_flowmap.py`.",
        "",
        "This fit promotes `a` to a continuous unknown, weights all sonic compatibility rows, and penalizes gradients above the configured limit.",
        "",
        f"Config: `a_half_width={CONSTRAINED_A_HALF_WIDTH:g}`, `g_limit={CONSTRAINED_G_LIMIT:g}`, `compat_weight={CONSTRAINED_COMPAT_WEIGHT:g}`, `reg_weight={CONSTRAINED_REG_WEIGHT:g}`, `flow_weight={CONSTRAINED_FLOW_WEIGHT:g}`, `g_weight={CONSTRAINED_G_WEIGHT:g}`.",
        "",
        "| label | regularity | eps buf | branch | physical | weighted | dominant | D | C1 | C2 | K | reg | SVD L | adj u | adj T | flow | g hinge | Rson/rg | dRson | lambda0 | dlambda | a | g_u | g_T | nfev | success | message |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {regularity} | {epsilon_buf} | {branch} | {physical_active} | {weighted_max} | {dominant} | "
            "{D} | {C1} | {C2} | {K} | {regularity_residual} | {svd_L} | {adj_u} | {adj_T} | {flow} | "
            "{g_hinge} | {Rson_rg} | {delta_Rson_rg} | {lambda0} | {delta_lambda0} | {a} | {g_u} | {g_T} | "
            "{nfev} | {success} | {message} |".format(
                label=row["label"],
                regularity=row["regularity"],
                epsilon_buf=fmt(float(row["epsilon_buf"])),
                branch=row["branch"],
                physical_active=fmt(float(row["physical_active"])),
                weighted_max=fmt(float(row["weighted_max"])),
                dominant=row["dominant"],
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                regularity_residual=fmt(float(row["regularity_residual"])),
                svd_L=fmt(float(row["svd_L"])),
                adj_u=fmt(float(row["adj_u"])),
                adj_T=fmt(float(row["adj_T"])),
                flow=fmt(float(row["flow"])),
                g_hinge=fmt(float(row["g_hinge"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row["delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row["delta_lambda0"])),
                a=fmt(float(row["a"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    CONSTRAINED_FIT_TABLE.write_text("\n".join(lines) + "\n")


def write_bvp_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Sonic Flow-Map BVP",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_flowmap.py`.",
        "",
        "| label | stage | eps buf | branch | N regular | physical | selected | pass | dominant | D | C1 | C2 | K | flow | regular R | regular E | outer R | outer E | far omega | Rson/rg | lambda0 | int adv | max H/R | g_u | g_T | branch a | nfev | success | message |",
        "|---|---|---:|---:|---:|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {stage} | {epsilon_buf} | {branch} | {n_regular} | {physical_active} | {selected_max} | {passes_science} | "
            "{dominant} | {D} | {C1} | {C2} | {K} | {flow} | {regular_R} | {regular_E} | {outer_R} | {outer_E} | "
            "{far_omega} | {Rson_rg} | {lambda0} | {int_adv} | {max_HR} | {g_u} | {g_T} | {branch_a} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                stage=row["stage"],
                epsilon_buf=fmt(float(row["epsilon_buf"])),
                branch=row["branch"],
                n_regular=row["n_regular"],
                physical_active=fmt(float(row["physical_active"])),
                selected_max=fmt(float(row["selected_max"])),
                passes_science="yes" if row["passes_science"] else "no",
                dominant=row["dominant"],
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                flow=fmt(float(row["flow"])),
                regular_R=fmt(float(row["regular_R"])),
                regular_E=fmt(float(row["regular_E"])),
                outer_R=fmt(float(row["outer_R"])),
                outer_E=fmt(float(row["outer_E"])),
                far_omega=fmt(float(row["far_omega"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                max_HR=fmt(float(row["max_HR"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                branch_a=fmt(float(row["branch_a"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    BVP_TABLE.write_text("\n".join(lines) + "\n")


def write_smooth_bvp_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Sonic Flow-Map Smooth-a BVP",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_flowmap.py`.",
        "",
        "This BVP promotes the branch coordinate `a` to an unknown and adds the L'Hopital row `L(a)` to keep the residual smooth.",
        "",
        f"Config: `a_half_width={SMOOTH_BVP_A_HALF_WIDTH:g}`, `epsilon0={EPSILON0:g}`, "
        f"`method={FLOW_METHOD}`, `inner_grid_power={INNER_GRID_POWER:g}`, "
        f"`micro_R_rg={MICRO_R_RG:g}`, `n_micro={N_MICRO}`, "
        f"`micro_seed_flowmap={MICRO_SEED_FLOWMAP}`, `remap={REMAP_METHOD}`.",
        "",
        "| label | stage | eps buf | branch | N regular | physical | selected | pass | dominant | D | C1 | C2 | K | L | flow | regular R | regular E | outer R | outer E | far omega | Rson/rg | lambda0 | int adv | max H/R | a | g_u | g_T | nfev | success | message |",
        "|---|---|---:|---:|---:|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {stage} | {epsilon_buf} | {branch} | {n_regular} | {physical_active} | {selected_max} | {passes_science} | "
            "{dominant} | {D} | {C1} | {C2} | {K} | {L} | {flow} | {regular_R} | {regular_E} | {outer_R} | {outer_E} | "
            "{far_omega} | {Rson_rg} | {lambda0} | {int_adv} | {max_HR} | {a} | {g_u} | {g_T} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                stage=row["stage"],
                epsilon_buf=fmt(float(row["epsilon_buf"])),
                branch=row["branch"],
                n_regular=row["n_regular"],
                physical_active=fmt(float(row["physical_active"])),
                selected_max=fmt(float(row["selected_max"])),
                passes_science="yes" if row["passes_science"] else "no",
                dominant=row["dominant"],
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                L=fmt(float(row["L"])),
                flow=fmt(float(row["flow"])),
                regular_R=fmt(float(row["regular_R"])),
                regular_E=fmt(float(row["regular_E"])),
                outer_R=fmt(float(row["outer_R"])),
                outer_E=fmt(float(row["outer_E"])),
                far_omega=fmt(float(row["far_omega"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                max_HR=fmt(float(row["max_HR"])),
                a=fmt(float(row["a"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    SMOOTH_BVP_TABLE.write_text("\n".join(lines) + "\n")


def main() -> None:
    epsilon_bufs = parse_float_sequence("IMBH_FLOWMAP_BRANCH_EPS_BUFS", DEFAULT_EPSILON_BUFS)
    bvp_epsilon_bufs = parse_float_sequence("IMBH_FLOWMAP_BVP_EPS_BUFS", DEFAULT_BVP_EPSILON_BUFS)
    n_sequence = parse_int_sequence("IMBH_FLOWMAP_N_SEQUENCE", DEFAULT_N_SEQUENCE)
    branch_sequence = parse_branch_sequence()
    smooth_regularities = parse_string_sequence("IMBH_FLOWMAP_SMOOTH_REGULARITIES", ("svd",))
    constrained_regularities = parse_string_sequence("IMBH_FLOWMAP_CONSTRAINED_REGULARITIES", ("svd",))
    bvp_seed_modes = parse_string_sequence("IMBH_FLOWMAP_BVP_SEED_MODES", ("dynamic",))
    run_bvp = os.environ.get("IMBH_FLOWMAP_RUN_BVP", "1") != "0"
    solve_bvp = os.environ.get("IMBH_FLOWMAP_SOLVE_BVP", "1") != "0"
    run_polish = os.environ.get("IMBH_FLOWMAP_RUN_POLISH", "1") != "0"
    run_smooth_bvp = os.environ.get("IMBH_FLOWMAP_RUN_SMOOTH_BVP", "0") != "0"
    solve_smooth_bvp = os.environ.get("IMBH_FLOWMAP_SOLVE_SMOOTH_BVP", "1") != "0"
    run_smooth_bvp_polish = os.environ.get("IMBH_FLOWMAP_RUN_SMOOTH_BVP_POLISH", "0") != "0"
    smooth_start_checkpoint = os.environ.get("IMBH_FLOWMAP_SMOOTH_START_CHECKPOINT")
    run_fit = os.environ.get("IMBH_FLOWMAP_RUN_FIT", "1") != "0"
    run_smooth_fit = os.environ.get("IMBH_FLOWMAP_RUN_SMOOTH_FIT", "1") != "0"
    run_constrained_fit = os.environ.get("IMBH_FLOWMAP_RUN_CONSTRAINED_FIT", "1") != "0"
    run_adjugate = os.environ.get("IMBH_FLOWMAP_RUN_ADJUGATE", "1") != "0"

    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    _source_x, source_meta = load_checkpoint(SOURCE_CHECKPOINT)
    ratio = float(source_meta["ratio"])
    dynamic_x, dynamic_params, dynamic_meta = load_dynamic_source(fiducial, ratio, mdot_edd)
    smooth_start: tuple[np.ndarray, FlowmapParams, dict[str, Any]] | None = None
    if smooth_start_checkpoint:
        start_path = Path(smooth_start_checkpoint)
        if not start_path.is_absolute():
            start_path = ROOT / start_path
        smooth_start = load_smooth_checkpoint(start_path, fiducial, mdot_edd)
        _start_x, start_params, start_row = smooth_start
        print(
            f"loaded smooth start checkpoint {start_path} "
            f"N={start_params.n_regular} eps={start_params.epsilon_buf:g} branch={start_params.branch_index} "
            f"p={start_params.inner_grid_power:g} physical={float(start_row['physical_active']):.3e}",
            flush=True,
        )
    probe_params = make_flowmap_params(
        fiducial,
        ratio,
        mdot_edd,
        int(dynamic_meta["n_regular"]),
        int(dynamic_meta["n_outer"]),
        float(dynamic_meta["R_far_rg"]),
        bvp_epsilon_bufs[0],
        0,
        None,
    )
    branches = initial_branches(dynamic_x, dynamic_params, probe_params.physics)
    if not branches:
        raise RuntimeError("initial dynamic state has no L'Hopital branches")
    print(
        f"flowmap ratio={ratio:.8f} branches={len(branches)} "
        f"eps_scan={epsilon_bufs} bvp_eps={bvp_epsilon_bufs} N={n_sequence}",
        flush=True,
    )

    branch_rows = branch_audit_rows(dynamic_x, dynamic_params, ratio, fiducial, mdot_edd, epsilon_bufs)
    write_branch_table(branch_rows)
    print(f"wrote {BRANCH_TABLE}", flush=True)

    if run_adjugate:
        adjugate_rows = adjugate_audit_rows(dynamic_x, dynamic_params, ratio, fiducial, mdot_edd, bvp_epsilon_bufs[0])
        write_adjugate_table(adjugate_rows)
        print(f"wrote {ADJUGATE_TABLE}", flush=True)

    smooth_fit_rows: list[dict[str, object]] = []
    if run_smooth_fit:
        for epsilon_buf in bvp_epsilon_bufs:
            for branch_index in branch_sequence:
                if branch_index >= len(branches):
                    continue
                branch_a = float(branches[branch_index].a)
                params = make_flowmap_params(
                    fiducial,
                    ratio,
                    mdot_edd,
                    int(dynamic_meta["n_regular"]),
                    int(dynamic_meta["n_outer"]),
                    float(dynamic_meta["R_far_rg"]),
                    epsilon_buf,
                    branch_index,
                    branch_a,
                )
                for regularity in smooth_regularities:
                    result, target, residual = fit_local_flowmap_smooth(dynamic_x, dynamic_params, params, branch_a, regularity)
                    row = smooth_fit_audit_row(
                        f"eps{epsilon_buf:g}_branch{branch_index}_{regularity}",
                        dynamic_x,
                        dynamic_params,
                        params,
                        regularity,
                        result,
                        target,
                        residual,
                    )
                    smooth_fit_rows.append(row)
                    print(
                        f"smooth fit eps={epsilon_buf:g} branch={branch_index} reg={regularity} "
                        f"physical={row['physical_active']:.3e} dominant={row['dominant']} nfev={row['nfev']}",
                        flush=True,
                    )
        write_smooth_fit_table(smooth_fit_rows)
        print(f"wrote {SMOOTH_FIT_TABLE}", flush=True)

    constrained_fit_rows: list[dict[str, object]] = []
    if run_constrained_fit:
        for epsilon_buf in bvp_epsilon_bufs:
            for branch_index in branch_sequence:
                if branch_index >= len(branches):
                    continue
                branch_a = float(branches[branch_index].a)
                params = make_flowmap_params(
                    fiducial,
                    ratio,
                    mdot_edd,
                    int(dynamic_meta["n_regular"]),
                    int(dynamic_meta["n_outer"]),
                    float(dynamic_meta["R_far_rg"]),
                    epsilon_buf,
                    branch_index,
                    branch_a,
                )
                for regularity in constrained_regularities:
                    result, target, weighted, unweighted_result = fit_local_flowmap_constrained(dynamic_x, dynamic_params, params, branch_a, regularity)
                    row = constrained_fit_audit_row(
                        f"eps{epsilon_buf:g}_branch{branch_index}_{regularity}",
                        dynamic_x,
                        dynamic_params,
                        params,
                        regularity,
                        result,
                        target,
                        weighted,
                        unweighted_result,
                    )
                    constrained_fit_rows.append(row)
                    print(
                        f"constrained fit eps={epsilon_buf:g} branch={branch_index} reg={regularity} "
                        f"physical={row['physical_active']:.3e} dominant={row['dominant']} weighted={row['weighted_max']:.3e} nfev={row['nfev']}",
                        flush=True,
                    )
        write_constrained_fit_table(constrained_fit_rows)
        print(f"wrote {CONSTRAINED_FIT_TABLE}", flush=True)

    fit_rows: list[dict[str, object]] = []
    best_fit_by_key: dict[tuple[float, int], dict[str, object]] = {}
    if run_fit:
        for epsilon_buf in bvp_epsilon_bufs:
            for branch_index in branch_sequence:
                if branch_index >= len(branches):
                    continue
                branch_a = float(branches[branch_index].a)
                params = make_flowmap_params(
                    fiducial,
                    ratio,
                    mdot_edd,
                    int(dynamic_meta["n_regular"]),
                    int(dynamic_meta["n_outer"]),
                    float(dynamic_meta["R_far_rg"]),
                    epsilon_buf,
                    branch_index,
                    branch_a,
                )
                result, target, residual = fit_local_flowmap(dynamic_x, dynamic_params, params, branch_a)
                row = fit_audit_row(f"eps{epsilon_buf:g}_branch{branch_index}", dynamic_x, dynamic_params, params, branch_a, result, target, residual)
                fit_rows.append(row)
                best_fit_by_key[(epsilon_buf, branch_index)] = row
                print(
                    f"fit eps={epsilon_buf:g} branch={branch_index} physical={row['physical_active']:.3e} "
                    f"dominant={row['dominant']} nfev={row['nfev']}",
                    flush=True,
                )
        write_fit_table(fit_rows)
        print(f"wrote {FIT_TABLE}", flush=True)

    bvp_rows: list[dict[str, object]] = []
    if run_bvp:
        for seed_mode in bvp_seed_modes:
            for epsilon_buf in bvp_epsilon_bufs:
                for branch_index in branch_sequence:
                    if branch_index >= len(branches):
                        continue
                    branch_a = float(branches[branch_index].a)
                    current_x: np.ndarray | None = None
                    current_params: FlowmapParams | None = None
                    fit_row = best_fit_by_key.get((epsilon_buf, branch_index))
                    for n_regular in n_sequence:
                        params = make_flowmap_params(
                            fiducial,
                            ratio,
                            mdot_edd,
                            int(n_regular),
                            int(dynamic_meta["n_outer"]),
                            float(dynamic_meta["R_far_rg"]),
                            epsilon_buf,
                            branch_index,
                            branch_a,
                        )
                        if current_x is None or current_params is None:
                            if seed_mode == "dynamic":
                                seed = seed_from_dynamic(dynamic_x, dynamic_params, params)
                                if fit_row is not None and float(fit_row["physical_active"]) < 5.0e-3:
                                    try:
                                        seed = apply_fit_seed(seed, params, fit_row)
                                    except Exception as exc:
                                        print(f"fit seed failed eps={epsilon_buf:g} branch={branch_index}: {exc}", flush=True)
                            elif seed_mode == "flowmap":
                                seed = seed_from_flowmap_boundary(dynamic_x, dynamic_params, params)
                            else:
                                raise ValueError(f"unknown BVP seed mode {seed_mode!r}")
                        else:
                            seed = seed_from_flowmap(current_x, current_params, params)
                        label = f"eps{epsilon_buf:g}_{seed_mode}_branch{branch_index}_N{n_regular}{inner_grid_label_suffix(params)}"
                        seed_row = flowmap_audit(label, seed, params)
                        seed_row["stage"] = "seed"
                        bvp_rows.append(seed_row)
                        write_bvp_table(bvp_rows)
                        print(
                            f"BVP seed mode={seed_mode} eps={epsilon_buf:g} branch={branch_index} N={n_regular} "
                            f"physical={seed_row['physical_active']:.3e} dominant={seed_row['dominant']}",
                            flush=True,
                        )
                        if not solve_bvp:
                            current_x = seed
                            current_params = params
                            continue
                        release = solve_flowmap(seed, params, MAX_NFEV_RELEASE)
                        release_row = flowmap_audit(label, release.x, params, release)
                        release_row["stage"] = "release"
                        bvp_rows.append(release_row)
                        write_bvp_table(bvp_rows)
                        print(
                            f"BVP release mode={seed_mode} eps={epsilon_buf:g} branch={branch_index} N={n_regular} "
                            f"physical={release_row['physical_active']:.3e} dominant={release_row['dominant']} nfev={release.nfev}",
                            flush=True,
                        )
                        if not run_polish:
                            current_x = np.asarray(release_row["x"], dtype=float)
                            current_params = params
                            continue
                        polish = solve_flowmap(release.x, params, MAX_NFEV_POLISH)
                        polish_row = flowmap_audit(label, polish.x, params, polish)
                        polish_row["stage"] = "polish"
                        bvp_rows.append(polish_row)
                        write_bvp_table(bvp_rows)
                        save_checkpoint(label, np.asarray(polish_row["x"], dtype=float), polish_row)
                        print(
                            f"BVP polish mode={seed_mode} eps={epsilon_buf:g} branch={branch_index} N={n_regular} "
                            f"physical={polish_row['physical_active']:.3e} dominant={polish_row['dominant']} nfev={polish.nfev}",
                            flush=True,
                        )
                        current_x = np.asarray(polish_row["x"], dtype=float)
                        current_params = params
        write_bvp_table(bvp_rows)
        print(f"wrote {BVP_TABLE}", flush=True)

    smooth_bvp_rows: list[dict[str, object]] = []
    if run_smooth_bvp:
        for seed_mode in bvp_seed_modes:
            for epsilon_buf in bvp_epsilon_bufs:
                for branch_index in branch_sequence:
                    if branch_index >= len(branches):
                        continue
                    branch_a = float(branches[branch_index].a)
                    current_x: np.ndarray | None = None
                    current_params: FlowmapParams | None = None
                    if smooth_start is not None:
                        start_x, start_params, _start_row = smooth_start
                        if start_params.branch_index == branch_index and abs(start_params.epsilon_buf - epsilon_buf) < 1.0e-12:
                            current_x = start_x
                            current_params = start_params
                            print(
                                f"smooth BVP starting from checkpoint branch={branch_index} eps={epsilon_buf:g} "
                                f"N={start_params.n_regular} p={start_params.inner_grid_power:g}",
                                flush=True,
                            )
                    for n_regular in n_sequence:
                        params = make_flowmap_params(
                            fiducial,
                            ratio,
                            mdot_edd,
                            int(n_regular),
                            int(dynamic_meta["n_outer"]),
                            float(dynamic_meta["R_far_rg"]),
                            epsilon_buf,
                            branch_index,
                            branch_a,
                        )
                        if current_x is None or current_params is None:
                            seed = smooth_seed_from_dynamic(dynamic_x, dynamic_params, params, seed_mode, branch_a)
                        else:
                            seed = smooth_seed_from_smooth(current_x, current_params, params)
                        label = f"eps{epsilon_buf:g}_{seed_mode}_branch{branch_index}_N{n_regular}{inner_grid_label_suffix(params)}"
                        seed_row = smooth_flowmap_audit(label, seed, params)
                        seed_row["stage"] = "seed"
                        smooth_bvp_rows.append(seed_row)
                        write_smooth_bvp_table(smooth_bvp_rows)
                        print(
                            f"smooth BVP seed mode={seed_mode} eps={epsilon_buf:g} branch={branch_index} N={n_regular} "
                            f"physical={seed_row['physical_active']:.3e} dominant={seed_row['dominant']} a={seed_row['a']:.3g}",
                            flush=True,
                        )
                        if not solve_smooth_bvp:
                            current_x = seed
                            current_params = params
                            continue
                        release = solve_smooth_flowmap(seed, params, MAX_NFEV_SMOOTH_BVP_RELEASE)
                        release_row = smooth_flowmap_audit(label, release.x, params, release)
                        release_row["stage"] = "release"
                        smooth_bvp_rows.append(release_row)
                        write_smooth_bvp_table(smooth_bvp_rows)
                        print(
                            f"smooth BVP release mode={seed_mode} eps={epsilon_buf:g} branch={branch_index} N={n_regular} "
                            f"physical={release_row['physical_active']:.3e} dominant={release_row['dominant']} "
                            f"a={release_row['a']:.3g} nfev={release.nfev}",
                            flush=True,
                        )
                        if not run_smooth_bvp_polish:
                            current_x = np.asarray(release_row["x"], dtype=float)
                            current_params = params
                            continue
                        polish = solve_smooth_flowmap(release.x, params, MAX_NFEV_SMOOTH_BVP_POLISH)
                        polish_row = smooth_flowmap_audit(label, polish.x, params, polish)
                        polish_row["stage"] = "polish"
                        smooth_bvp_rows.append(polish_row)
                        write_smooth_bvp_table(smooth_bvp_rows)
                        save_checkpoint(f"smooth_{label}", np.asarray(polish_row["x"], dtype=float), polish_row)
                        print(
                            f"smooth BVP polish mode={seed_mode} eps={epsilon_buf:g} branch={branch_index} N={n_regular} "
                            f"physical={polish_row['physical_active']:.3e} dominant={polish_row['dominant']} "
                            f"a={polish_row['a']:.3g} nfev={polish.nfev}",
                            flush=True,
                        )
                        current_x = np.asarray(polish_row["x"], dtype=float)
                        current_params = params
        write_smooth_bvp_table(smooth_bvp_rows)
        print(f"wrote {SMOOTH_BVP_TABLE}", flush=True)


if __name__ == "__main__":
    main()
