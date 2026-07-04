"""Mass-loading stream annulus scan for the standard slim disk."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    pressure_supported_omega_target,
    remap_profile_to_new_sonic_grid,
    residual_audit_from_state_vector,
    residual_partition_audit_from_state_vector,
    select_sonic_compatibility_pivot,
    solve_square_transonic_polish,
    square_collocation_jacobian,
    square_collocation_residual,
    pack_state,
    state_bounds,
    stream_heating_rate,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    stream_torque_specific_l_and_derivative,
    transonic_profile_from_state_vector,
    unused_sonic_compatibility,
    unpack_state,
    wind_sink_prime,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _differential_interval_residual_from_unpacked,
    _heating_terms_from_gradient,
    _interval_residual_from_unpacked,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_luminosity, eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant
from run_standard_slim_stream_residual_remesh import residual_remesh_grid_xi


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_MASS_ANCHOR",
    "outputs/checkpoints/slim_benchmark_physical_rout_homotopy_mdot1_1000_300/Rout_300_mdot_1_N640.npz",
)
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_MASS_TABLE",
    "outputs/tables/slim_benchmark_stream_mass_annulus_mdot1_rout300.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_MASS_FIGURE",
    "outputs/figures/slim_benchmark_stream_mass_annulus_mdot1_rout300.png",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_MASS_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_stream_mass_annulus_mdot1_rout300",
)
NEWTON_AUDIT_DIR_RAW = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_AUDIT_DIR", "").strip()
NEWTON_AUDIT_DIR = (ROOT / NEWTON_AUDIT_DIR_RAW) if NEWTON_AUDIT_DIR_RAW else None

BRANCH_SPECS = tuple(
    piece.strip()
    for piece in os.environ.get(
        "IMBH_STANDARD_SLIM_STREAM_MASS_BRANCHES",
        "load:0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2",
    ).split(";")
    if piece.strip()
)
R_OUT_RG_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_R_OUT_RG", "").strip()
FIXED_RINJ_RG_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_FIXED_RINJ_RG", "").strip()
FIXED_TORQUE_RINJ_RG_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_FIXED_TORQUE_RINJ_RG", "").strip()
MASS_CENTER_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_CENTER_FRACTION", "0.8"))
MASS_LOG_WIDTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOG_WIDTH", "0.08"))
MASS_SOURCE_SHAPE_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_SHAPE", "").strip().lower()
MASS_SOURCE_SHAPE = MASS_SOURCE_SHAPE_OVERRIDE or "tanh"
MASS_SOURCE_SHAPE_BLEND_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_SHAPE_BLEND", "").strip()
TORQUE_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TORQUE_FRACTION", "0.0"))
TORQUE_CENTER_FRACTION_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TORQUE_CENTER_FRACTION", "").strip()
TORQUE_CENTER_FRACTION = float(TORQUE_CENTER_FRACTION_OVERRIDE or MASS_CENTER_FRACTION)
TORQUE_LOG_WIDTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TORQUE_LOG_WIDTH", str(MASS_LOG_WIDTH)))
OUTER_CLOSURE_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_CLOSURE", "").strip()
OUTER_ROBIN_CHI_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_ROBIN_CHI", "").strip()
OUTER_ROBIN_SLOPE_TARGET_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_ROBIN_SLOPE_TARGET", "").strip()
OUTER_ROBIN_SLOPE_SCALE_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_ROBIN_SLOPE_SCALE", "").strip()
OUTER_BUFFER_INNER_RG_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_INNER_RG", "").strip()
OUTER_BUFFER_RADIAL_WEIGHT = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_RADIAL_WEIGHT", "1.0"))
OUTER_BUFFER_ENERGY_WEIGHT = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_ENERGY_WEIGHT", "1.0"))
OUTER_BUFFER_BOUNDARY_WEIGHT = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_BOUNDARY_WEIGHT", "1.0"))
OUTER_BUFFER_TAPER_LOG_WIDTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_TAPER_LOG_WIDTH", "0.0"))
N_NODES_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_N_NODES", "").strip()
GRID_POWER_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_GRID_POWER", "").strip()
GRID_TRANSFER_MODE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_GRID_TRANSFER", "power").strip().lower()
REMAP_METHOD = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_REMAP_METHOD", "linear").strip().lower()
SOURCE_GRID_MODE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID", "none").strip().lower()
SOURCE_GRID_BLEND_WITH_CURRENT = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_BLEND_WITH_CURRENT", "1.0"))
SOURCE_GRID_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_FRACTION", "0.35"))
SOURCE_GRID_HALF_WIDTHS = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_HALF_WIDTHS", "4.0"))
SOURCE_GRID_OUTER_FRACTION = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_OUTER_FRACTION", "0.0"))
SOURCE_GRID_OUTER_WIDTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_OUTER_WIDTH", "0.04"))
SOURCE_GRID_TARGET_FRACTION_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_TARGET_FRACTION", "").strip()
SOURCE_GRID_TARGET_WEIGHT = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_TARGET_WEIGHT", "8.0"))
SOURCE_GRID_TARGET_HALF_WIDTHS = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_TARGET_HALF_WIDTHS", "0.75"))
USE_SECANT_PREDICTOR = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_USE_SECANT_PREDICTOR", "0") != "0"
USE_TANGENT_PREDICTOR = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_USE_TANGENT_PREDICTOR", "0") != "0"
SECANT_DAMPING_VALUES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_SECANT_DAMPINGS", "1,0.5,0.25,0.1")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
TANGENT_DAMPING_VALUES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_DAMPINGS", "1,0.5,0.25,0.1")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
TANGENT_FD_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_FD_STEP", "1e-5"))
TANGENT_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_SOLVER", "equilibrated_lsmr")
TANGENT_LINEAR_DAMPING = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_LINEAR_DAMPING", "0.0"))
TANGENT_MAXITER = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_MAXITER", "3000"))
TANGENT_TRIGGER_INITIAL_FULL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_TRIGGER_INITIAL_FULL", "0"))
ADAPTIVE_TARGET_RAW = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_TARGET", "").strip()
HEATING_EFFICIENCIES_RAW = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_HEATING_EFFICIENCIES", "").strip()
HEATING_LABEL = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_HEATING_LABEL", "heating").strip() or "heating"
ADAPTIVE_INITIAL_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_INITIAL_STEP", "0.001"))
ADAPTIVE_MIN_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_MIN_STEP", "0.00025"))
ADAPTIVE_MAX_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_MAX_STEP", "0.005"))
ADAPTIVE_MAX_INITIAL_FULL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_MAX_INITIAL_FULL", "0.08"))
ADAPTIVE_GROWTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_GROWTH", "1.5"))
ADAPTIVE_SHRINK = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_SHRINK", "0.5"))
ADAPTIVE_COST_SHRINK_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_SHRINK_NFEV", "20"))
ADAPTIVE_COST_HARD_SHRINK_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_HARD_SHRINK_NFEV", "60"))
ADAPTIVE_COST_GROW_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_GROW_NFEV", "8"))
ADAPTIVE_COST_SHRINK = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_SHRINK", "0.5"))
ADAPTIVE_COST_HARD_SHRINK = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_HARD_SHRINK", "0.25"))
NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_MAX_ITER", "30"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_MAX_NFEV", "3000"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_MAX_STEP_NORM", "0.16"))
POLISH_METHOD = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_POLISH_METHOD", "newton").strip().lower()
NEWTON_JACOBIAN_REL_STEP = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_JACOBIAN_REL_STEP", "3e-5"))
NEWTON_ENERGY_JACOBIAN_REL_STEP_RAW = os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_JACOBIAN_REL_STEP",
    "",
).strip()
NEWTON_ENERGY_JACOBIAN_REL_STEP = (
    None if not NEWTON_ENERGY_JACOBIAN_REL_STEP_RAW else float(NEWTON_ENERGY_JACOBIAN_REL_STEP_RAW)
)
NEWTON_LINE_SEARCH_MIN_ALPHA = float(
    os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_LINE_SEARCH_MIN_ALPHA", "1e-6")
)
NEWTON_LINE_SEARCH_MAX_REDUCTIONS = int(
    os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_LINE_SEARCH_MAX_REDUCTIONS", "12")
)
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_LINEAR_SOLVER", "regularized_lsmr")
NEWTON_LINEAR_DAMPINGS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_LINEAR_DAMPINGS", "0,1e-4,1e-3,1e-2,1e-1,1")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
NEWTON_ENERGY_MERIT = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_MERIT", "off").strip().lower()
NEWTON_ENERGY_MERIT_TOL_RAW = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_MERIT_TOL", "").strip()
NEWTON_ENERGY_MERIT_L2_TOL_RAW = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_MERIT_L2_TOL", "").strip()
NEWTON_ENERGY_MERIT_GLOBAL_TOL_RAW = os.environ.get(
    "IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_MERIT_GLOBAL_TOL", ""
).strip()
NEWTON_ENERGY_MERIT_REQUIRE_DECREASE = (
    os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_MERIT_REQUIRE_DECREASE", "1") != "0"
)
NEWTON_ENERGY_ROW_PRIORITY = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_ROW_PRIORITY", "1.0"))
LOCAL_PATCH_ON_REJECT = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_ON_REJECT", "0") != "0"
LOCAL_PATCH_HALF_WIDTH_RG = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_HALF_WIDTH_RG", "3.0"))
LOCAL_PATCH_TOP_K = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_TOP_K", "0"))
LOCAL_PATCH_NODE_PAD = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_NODE_PAD", "0"))
LOCAL_PATCH_MAX_ACTIVE_NODES = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_MAX_ACTIVE_NODES", "80"))
LOCAL_PATCH_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_MAX_NFEV", "80"))
LOCAL_PATCH_MAX_PASSES = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_MAX_PASSES", "1"))
LOCAL_PATCH_GLOBAL_AFTER_PHYSICAL = (
    os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_GLOBAL_AFTER_PHYSICAL", "1") != "0"
)
LOCAL_PATCH_ENERGY_WEIGHT = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_ENERGY_WEIGHT", "5.0"))
LOCAL_PATCH_PRIOR_WEIGHT = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_PRIOR_WEIGHT", "1e-4"))
NEWTON_RESIDUAL_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_RESIDUAL_TOL", "1e-8"))
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ANCHOR_TOL", "3e-6"))
ACCEPT_SEED_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_ACCEPT_SEED_TOL", "0"))
PHYSICAL_E_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_PHYSICAL_E_TOL", "inf"))
REQUIRE_PHYSICAL_E_GATE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_REQUIRE_PHYSICAL_E_GATE", "0") != "0"
CLEANUP_REPOLISH_PASSES = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_CLEANUP_REPOLISH_PASSES", "0"))
CLEANUP_REPOLISH_ONLY_ACCEPTED = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_CLEANUP_ONLY_ACCEPTED", "1") != "0"
CLEANUP_REPOLISH_MAX_BASE_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_CLEANUP_MAX_BASE_NFEV", "1000000000"))
CLEANUP_REPOLISH_MAX_BASE_PHYSICAL_E = float(
    os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_CLEANUP_MAX_BASE_PHYSICAL_E", "inf")
)
CLEANUP_POLISH_SPECS_RAW = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_CLEANUP_POLISH_SPECS", "same").strip()
LEAN_REJECT_DIAGNOSTICS = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_LEAN_REJECT_DIAGNOSTICS", "0") != "0"
INTERVAL_RESIDUAL_FORM = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_INTERVAL_FORM", "differential").strip().lower()
INTEGRATED_RESIDUAL_WEIGHTING = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_INTEGRATED_WEIGHTING", "none").strip().lower()
FORCE_INTERVAL_RESIDUAL_FORM = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_FORCE_INTERVAL_FORM", "").strip().lower()
FORCE_INTEGRATED_RESIDUAL_WEIGHTING = (
    os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_FORCE_INTEGRATED_WEIGHTING", "").strip().lower()
)
REFRESH_REPOLISH = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_REFRESH_REPOLISH", "0") != "0"
RESIDUAL_REMESH_EVERY_STEP = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_EVERY_STEP", "0") != "0"
RESIDUAL_REMESH_ON_REJECT = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_ON_REJECT", "0") != "0"
RESIDUAL_REMESH_STRENGTH = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_STRENGTH", "12"))
RESIDUAL_REMESH_N_NODES_OVERRIDE = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_N_NODES", "").strip()
RESIDUAL_REMESH_MAX_INITIAL_FULL = float(
    os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_MAX_INITIAL_FULL", "inf")
)
OUTER_SLOPE_PICARD = os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_SLOPE_PICARD", "0") != "0"
OUTER_SLOPE_PICARD_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_SLOPE_PICARD_MAX_ITER", "3"))
OUTER_SLOPE_PICARD_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_SLOPE_PICARD_TOL", str(ACCEPTANCE_TOL)))
OUTER_SLOPE_PICARD_SLOPE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_SLOPE_PICARD_SLOPE_TOL", "1e-8"))
OUTER_SLOPE_PICARD_DAMPINGS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_SLOPE_PICARD_DAMPINGS", "0.3,0.5,1.0")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
INNER_RADIUS_RG = float(os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_INNER_RG", "20.0"))
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_STREAM_MASS_PIVOTS", "C2,C1").replace(":", ",").split(",")
    if piece.strip()
)


def parse_branch_specs() -> list[tuple[str, list[float]]]:
    branches: list[tuple[str, list[float]]] = []
    for spec in BRANCH_SPECS:
        if ":" not in spec:
            raise ValueError(f"branch spec must be label:mass_fractions, got {spec!r}")
        label, values = spec.split(":", 1)
        fractions = [float(piece) for piece in values.replace(",", ":").split(":") if piece.strip()]
        if not fractions:
            raise ValueError(f"branch {label!r} has no mass fractions")
        branches.append((label.strip(), fractions))
    return branches


def parse_heating_efficiencies() -> list[float]:
    if not HEATING_EFFICIENCIES_RAW:
        return []
    return [float(piece) for piece in HEATING_EFFICIENCIES_RAW.replace(",", ":").split(":") if piece.strip()]


def parse_cleanup_polish_specs() -> tuple[str, ...]:
    if not CLEANUP_POLISH_SPECS_RAW:
        return ("same",)
    separator = ";" if ";" in CLEANUP_POLISH_SPECS_RAW else ","
    specs = tuple(piece.strip().lower() for piece in CLEANUP_POLISH_SPECS_RAW.split(separator) if piece.strip())
    return specs or ("same",)


CLEANUP_POLISH_SPECS = parse_cleanup_polish_specs()


def energy_merit_tol_from_env(raw: str, fallback: float) -> float:
    if raw:
        return float(raw)
    if np.isfinite(PHYSICAL_E_TOL):
        return float(PHYSICAL_E_TOL)
    return float(fallback)


def target_r_out_rg(default_r_out_rg: float) -> float:
    return float(R_OUT_RG_OVERRIDE) if R_OUT_RG_OVERRIDE else float(default_r_out_rg)


def center_fraction_for_radius(
    fallback_fraction: float,
    *,
    R_out_rg: float,
    fixed_radius_rg: str,
    label: str,
) -> float:
    if fixed_radius_rg:
        fraction = float(fixed_radius_rg) / float(R_out_rg)
    else:
        fraction = float(fallback_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            f"{label} center fraction must be between 0 and 1; got {fraction:.6g} "
            f"for R_out={float(R_out_rg):.6g} rg"
        )
    return fraction


def source_center_for_rout(fallback_fraction: float, R_out_rg: float) -> float:
    return center_fraction_for_radius(
        fallback_fraction,
        R_out_rg=float(R_out_rg),
        fixed_radius_rg=FIXED_RINJ_RG_OVERRIDE,
        label="stream source",
    )


def torque_center_for_rout(fallback_fraction: float, R_out_rg: float) -> float:
    fixed_radius = FIXED_TORQUE_RINJ_RG_OVERRIDE
    if not fixed_radius and FIXED_RINJ_RG_OVERRIDE and not TORQUE_CENTER_FRACTION_OVERRIDE:
        fixed_radius = FIXED_RINJ_RG_OVERRIDE
    return center_fraction_for_radius(
        fallback_fraction,
        R_out_rg=float(R_out_rg),
        fixed_radius_rg=fixed_radius,
        label="stream torque",
    )


def custom_grid_from_data(data) -> tuple[float, ...] | None:
    if "custom_grid_xi" not in data:
        return None
    candidate = np.asarray(data["custom_grid_xi"], dtype=float)
    if candidate.shape == (int(data["n_nodes"]),):
        return tuple(float(value) for value in candidate)
    return None


def one_sided_outer_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    dx = float(logR[-1] - logR[-2])
    return float((logu[-1] - logu[-2]) / dx), float((logT[-1] - logT[-2]) / dx)


def apply_outer_slopes_from_state(z: np.ndarray, params: TransonicSlimParams) -> TransonicSlimParams:
    return replace(params, outer_match_log_slopes=one_sided_outer_slopes(z, params))


def resample_custom_grid_xi(custom_grid_xi: tuple[float, ...] | None, n_nodes: int) -> tuple[float, ...] | None:
    if custom_grid_xi is None:
        return None
    old = np.asarray(custom_grid_xi, dtype=float)
    if old.size < 2:
        return None
    source_index = np.linspace(0.0, 1.0, old.size)
    target_index = np.linspace(0.0, 1.0, int(n_nodes))
    new = np.interp(target_index, source_index, old)
    new[0] = 0.0
    new[-1] = 1.0
    if np.any(np.diff(new) <= 0.0):
        return None
    return tuple(float(value) for value in new)


def blend_grid_with_reference(
    target_grid_xi: tuple[float, ...],
    reference_grid_xi: tuple[float, ...] | None,
    *,
    n_nodes: int,
    grid_power: float,
    blend: float,
) -> tuple[float, ...]:
    blend = min(max(float(blend), 0.0), 1.0)
    target = np.asarray(target_grid_xi, dtype=float)
    reference = resample_custom_grid_xi(reference_grid_xi, n_nodes)
    if reference is None:
        reference_array = np.linspace(0.0, 1.0, int(n_nodes)) ** float(grid_power)
    else:
        reference_array = np.asarray(reference, dtype=float)
    mixed = (1.0 - blend) * reference_array + blend * target
    mixed[0] = 0.0
    mixed[-1] = 1.0
    if np.any(np.diff(mixed) <= 0.0):
        raise ValueError("source-grid blend produced a non-monotonic grid")
    return tuple(float(value) for value in mixed)


def source_annulus_grid_xi(
    *,
    logR_son: float,
    R_out: float,
    n_nodes: int,
    grid_power: float,
    center_fraction: float,
    log_width: float,
) -> tuple[float, ...]:
    if SOURCE_GRID_MODE in {"", "none", "off", "0"}:
        return tuple(float(value) for value in np.linspace(0.0, 1.0, int(n_nodes)) ** float(grid_power))
    if SOURCE_GRID_MODE not in {
        "annulus",
        "source",
        "focused",
        "outer",
        "tail",
        "annulus_outer",
        "residual",
        "annulus_peak",
        "source_peak",
        "focused_peak",
    }:
        raise ValueError(f"unknown source grid mode {SOURCE_GRID_MODE!r}")
    n_nodes = int(n_nodes)
    logR_out = float(np.log(R_out))
    denominator = max(logR_out - float(logR_son), 1.0e-12)
    logR_center = float(np.log(float(center_fraction) * R_out))
    xi_dense = np.linspace(0.0, 1.0, max(4096, 24 * n_nodes))
    logR_dense = float(logR_son) + xi_dense * denominator
    width = max(float(log_width) * max(float(SOURCE_GRID_HALF_WIDTHS), 0.5) / 2.0, 1.0e-6)
    source_density = np.exp(-0.5 * ((logR_dense - logR_center) / width) ** 2)
    target_density = np.zeros_like(source_density)
    target_weight = 0.0
    if SOURCE_GRID_MODE in {"annulus_peak", "source_peak", "focused_peak"} or SOURCE_GRID_TARGET_FRACTION_OVERRIDE:
        target_fraction = (
            float(SOURCE_GRID_TARGET_FRACTION_OVERRIDE)
            if SOURCE_GRID_TARGET_FRACTION_OVERRIDE
            else float(center_fraction)
        )
        target_fraction = min(max(target_fraction, float(np.exp(float(logR_son)) / R_out) * 1.001), 0.999)
        logR_target = float(np.log(target_fraction * R_out))
        target_width = max(float(log_width) * max(float(SOURCE_GRID_TARGET_HALF_WIDTHS), 0.05) / 2.0, 1.0e-6)
        target_density = np.exp(-0.5 * ((logR_dense - logR_target) / target_width) ** 2)
        target_weight = max(float(SOURCE_GRID_TARGET_WEIGHT), 0.0)
    outer_width = max(float(SOURCE_GRID_OUTER_WIDTH), 1.0e-4)
    outer_density = np.exp(-0.5 * ((xi_dense - 1.0) / outer_width) ** 2)
    source_weight = 0.0 if SOURCE_GRID_MODE in {"outer", "tail"} else 4.0 * float(SOURCE_GRID_FRACTION)
    outer_weight = 0.5 * max(1.0 - float(grid_power), 0.0) + 8.0 * float(SOURCE_GRID_OUTER_FRACTION)
    if SOURCE_GRID_MODE in {"outer", "tail", "annulus_outer", "residual"} and SOURCE_GRID_OUTER_FRACTION <= 0.0:
        outer_weight += 4.0
    density = 1.0 + source_weight * source_density + target_weight * target_density + outer_weight * outer_density
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[:-1] + density[1:]) * np.diff(xi_dense))])
    cdf /= cdf[-1]
    grid = np.interp(np.linspace(0.0, 1.0, n_nodes), cdf, xi_dense)
    grid[0] = 0.0
    grid[-1] = 1.0
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("source-focused grid generation produced a non-monotonic grid")
    return tuple(float(value) for value in grid)


def prepare_anchor_grid(
    z: np.ndarray,
    params: TransonicSlimParams,
    fiducial: FiducialParams,
    mdot_edd: float,
) -> tuple[np.ndarray, TransonicSlimParams]:
    n_nodes = int(N_NODES_OVERRIDE) if N_NODES_OVERRIDE else int(params.n_nodes)
    grid_power = float(GRID_POWER_OVERRIDE) if GRID_POWER_OVERRIDE else float(params.grid_power)
    target_R_out_rg = target_r_out_rg(params.R_out_rg)
    target_source_center = source_center_for_rout(params.stream_source_center_fraction, target_R_out_rg)
    target_torque_center = torque_center_for_rout(params.stream_torque_center_fraction, target_R_out_rg)
    logu, _logT, logR_son, _lambda0, _logR = unpack_state(z, params)
    custom_grid_xi = params.custom_grid_xi
    if SOURCE_GRID_MODE not in {"", "none", "off", "0"}:
        generated_grid_xi = source_annulus_grid_xi(
            logR_son=float(logR_son),
            R_out=float(target_R_out_rg * params.r_g),
            n_nodes=n_nodes,
            grid_power=grid_power,
            center_fraction=target_source_center,
            log_width=params.stream_source_log_width,
        )
        custom_grid_xi = blend_grid_with_reference(
            generated_grid_xi,
            params.custom_grid_xi,
            n_nodes=n_nodes,
            grid_power=grid_power,
            blend=SOURCE_GRID_BLEND_WITH_CURRENT,
        )
    elif GRID_TRANSFER_MODE in {"resample", "resample_current", "current", "preserve"} and n_nodes != int(params.n_nodes):
        custom_grid_xi = resample_custom_grid_xi(params.custom_grid_xi, n_nodes)
    elif n_nodes != int(params.n_nodes):
        custom_grid_xi = None
    target_params = params_for(
        fiducial,
        mdot_edd,
        ratio=params.mdot_edd_ratio,
        R_out_rg=target_R_out_rg,
        n_nodes=n_nodes,
        grid_power=grid_power,
        custom_grid_xi=custom_grid_xi if n_nodes == len(custom_grid_xi or ()) else custom_grid_xi,
        mass_fraction=float(params.stream_source_fraction),
        source_center_fraction=target_source_center,
        source_log_width=params.stream_source_log_width,
        source_shape=params.stream_source_shape,
        source_shape_blend=params.stream_source_shape_blend,
        torque_fraction=params.stream_torque_delta_l_fraction,
        torque_center_fraction=target_torque_center,
        torque_log_width=params.stream_torque_log_width,
        wind_sink_fraction=params.wind_sink_fraction,
        wind_sink_center_fraction=params.wind_sink_center_fraction,
        wind_sink_log_width=params.wind_sink_log_width,
        stream_heating_efficiency=params.stream_heating_efficiency,
        outer_closure=params.outer_closure,
        outer_robin_chi=params.outer_robin_chi,
        outer_robin_slope_target=params.outer_robin_slope_target,
        outer_robin_slope_scale=params.outer_robin_slope_scale,
        outer_buffer_inner_rg=params.outer_buffer_inner_rg,
        outer_buffer_radial_weight=params.outer_buffer_radial_weight,
        outer_buffer_energy_weight=params.outer_buffer_energy_weight,
        outer_buffer_boundary_weight=params.outer_buffer_boundary_weight,
        outer_buffer_taper_log_width=params.outer_buffer_taper_log_width,
        interval_residual_form=params.interval_residual_form,
        integrated_residual_weighting=params.integrated_residual_weighting,
    )
    if (
        n_nodes == int(params.n_nodes)
        and np.isclose(grid_power, params.grid_power)
        and np.isclose(target_R_out_rg, params.R_out_rg)
        and np.isclose(target_source_center, params.stream_source_center_fraction)
        and np.isclose(target_torque_center, params.stream_torque_center_fraction)
        and (
            (custom_grid_xi is None and params.custom_grid_xi is None)
            or (
                custom_grid_xi is not None
                and params.custom_grid_xi is not None
                and np.allclose(np.asarray(custom_grid_xi), np.asarray(params.custom_grid_xi))
            )
        )
    ):
        return z, apply_outer_slopes_from_state(z, target_params)
    profile = transonic_profile_from_state_vector(z, params)
    remapped_z = remap_profile_to_new_sonic_grid(profile, target_params, temperature_mdot_power=0.0, method=REMAP_METHOD)
    return remapped_z, apply_outer_slopes_from_state(remapped_z, target_params)


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    *,
    ratio: float,
    R_out_rg: float,
    n_nodes: int,
    grid_power: float,
    custom_grid_xi: tuple[float, ...] | None,
    mass_fraction: float,
    source_center_fraction: float | None = None,
    source_log_width: float | None = None,
    source_shape: str | None = None,
    source_shape_blend: float | None = None,
    torque_fraction: float | None = None,
    torque_center_fraction: float | None = None,
    torque_log_width: float | None = None,
    wind_sink_fraction: float = 0.0,
    wind_sink_center_fraction: float = 0.8,
    wind_sink_log_width: float = 0.08,
    stream_heating_efficiency: float = 0.0,
    outer_closure: str | None = None,
    outer_robin_chi: float = 0.0,
    outer_robin_slope_target: float = 0.0,
    outer_robin_slope_scale: float = 1.0,
    outer_buffer_inner_rg: float | None = None,
    outer_buffer_radial_weight: float = 1.0,
    outer_buffer_energy_weight: float = 1.0,
    outer_buffer_boundary_weight: float = 1.0,
    outer_buffer_taper_log_width: float = 0.0,
    interval_residual_form: str | None = None,
    integrated_residual_weighting: str | None = None,
) -> TransonicSlimParams:
    R_out_value = float(R_out_rg)
    source_center = source_center_for_rout(
        MASS_CENTER_FRACTION if source_center_fraction is None else float(source_center_fraction),
        R_out_value,
    )
    source_width = MASS_LOG_WIDTH if source_log_width is None else float(source_log_width)
    selected_source_shape = MASS_SOURCE_SHAPE if source_shape is None else str(source_shape).strip().lower()
    selected_source_shape_blend = (
        float(MASS_SOURCE_SHAPE_BLEND_OVERRIDE) if MASS_SOURCE_SHAPE_BLEND_OVERRIDE else (1.0 if source_shape_blend is None else float(source_shape_blend))
    )
    torque_delta = TORQUE_FRACTION if torque_fraction is None else float(torque_fraction)
    torque_center = torque_center_for_rout(
        TORQUE_CENTER_FRACTION if torque_center_fraction is None else float(torque_center_fraction),
        R_out_value,
    )
    torque_width = TORQUE_LOG_WIDTH if torque_log_width is None else float(torque_log_width)
    closure = OUTER_CLOSURE_OVERRIDE if OUTER_CLOSURE_OVERRIDE else (outer_closure or "pressure_supported_thin_energy")
    robin_chi = float(OUTER_ROBIN_CHI_OVERRIDE) if OUTER_ROBIN_CHI_OVERRIDE else float(outer_robin_chi)
    robin_slope_target = (
        float(OUTER_ROBIN_SLOPE_TARGET_OVERRIDE) if OUTER_ROBIN_SLOPE_TARGET_OVERRIDE else float(outer_robin_slope_target)
    )
    robin_slope_scale = (
        float(OUTER_ROBIN_SLOPE_SCALE_OVERRIDE) if OUTER_ROBIN_SLOPE_SCALE_OVERRIDE else float(outer_robin_slope_scale)
    )
    buffer_inner = None if outer_buffer_inner_rg is None else float(outer_buffer_inner_rg)
    if OUTER_BUFFER_INNER_RG_OVERRIDE:
        override_buffer_inner = float(OUTER_BUFFER_INNER_RG_OVERRIDE)
        if override_buffer_inner < R_out_value * (1.0 - 1.0e-12):
            buffer_inner = override_buffer_inner
    interval_form = INTERVAL_RESIDUAL_FORM if interval_residual_form is None else str(interval_residual_form).strip().lower()
    integrated_weighting = (
        INTEGRATED_RESIDUAL_WEIGHTING
        if integrated_residual_weighting is None
        else str(integrated_residual_weighting).strip().lower()
    )
    if FORCE_INTERVAL_RESIDUAL_FORM:
        interval_form = FORCE_INTERVAL_RESIDUAL_FORM
    if FORCE_INTEGRATED_RESIDUAL_WEIGHTING:
        integrated_weighting = FORCE_INTEGRATED_RESIDUAL_WEIGHTING
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(ratio) * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=R_out_value,
        n_nodes=int(n_nodes),
        grid_power=float(grid_power),
        custom_grid_xi=custom_grid_xi,
        max_nfev=NEWTON_MAX_NFEV,
        residual_tol=NEWTON_RESIDUAL_TOL,
        outer_closure=closure,
        outer_omega_log_offset=0.0,
        outer_robin_chi=robin_chi,
        outer_robin_slope_target=robin_slope_target,
        outer_robin_slope_scale=robin_slope_scale,
        outer_buffer_inner_rg=buffer_inner,
        outer_buffer_radial_weight=float(OUTER_BUFFER_RADIAL_WEIGHT if OUTER_BUFFER_INNER_RG_OVERRIDE else outer_buffer_radial_weight),
        outer_buffer_energy_weight=float(OUTER_BUFFER_ENERGY_WEIGHT if OUTER_BUFFER_INNER_RG_OVERRIDE else outer_buffer_energy_weight),
        outer_buffer_boundary_weight=float(
            OUTER_BUFFER_BOUNDARY_WEIGHT if OUTER_BUFFER_INNER_RG_OVERRIDE else outer_buffer_boundary_weight
        ),
        outer_buffer_taper_log_width=float(
            OUTER_BUFFER_TAPER_LOG_WIDTH if OUTER_BUFFER_INNER_RG_OVERRIDE else outer_buffer_taper_log_width
        ),
        stream_torque_delta_l_fraction=torque_delta,
        stream_torque_center_fraction=torque_center,
        stream_torque_log_width=torque_width,
        stream_source_fraction=float(mass_fraction),
        stream_source_center_fraction=source_center,
        stream_source_log_width=source_width,
        stream_source_shape=selected_source_shape,
        stream_source_shape_blend=selected_source_shape_blend,
        wind_sink_fraction=float(wind_sink_fraction),
        wind_sink_center_fraction=float(wind_sink_center_fraction),
        wind_sink_log_width=float(wind_sink_log_width),
        stream_heating_efficiency=float(stream_heating_efficiency),
        interval_residual_form=interval_form,
        integrated_residual_weighting=integrated_weighting,
    )


def scalar_from_data(data, key: str, default):
    if key not in data:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value


def load_anchor(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    data = np.load(path, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    mass_fraction = float(scalar_from_data(data, "stream_source_fraction", scalar_from_data(data, "stream_mass_fraction", 0.0)))
    params = params_for(
        fiducial,
        mdot_edd,
        ratio=float(scalar_from_data(data, "ratio", 1.0)),
        R_out_rg=float(scalar_from_data(data, "R_out_rg", 1000.0)),
        n_nodes=int(scalar_from_data(data, "n_nodes", (len(z) - 2) // 2)),
        grid_power=float(scalar_from_data(data, "grid_power", 1.0)),
        custom_grid_xi=custom_grid_from_data(data),
        mass_fraction=mass_fraction,
        source_center_fraction=float(scalar_from_data(data, "stream_source_center_fraction", MASS_CENTER_FRACTION)),
        source_log_width=float(scalar_from_data(data, "stream_source_log_width", MASS_LOG_WIDTH)),
        source_shape=MASS_SOURCE_SHAPE_OVERRIDE or str(scalar_from_data(data, "stream_source_shape", MASS_SOURCE_SHAPE)),
        source_shape_blend=float(
            MASS_SOURCE_SHAPE_BLEND_OVERRIDE or scalar_from_data(data, "stream_source_shape_blend", 1.0)
        ),
        torque_fraction=float(scalar_from_data(data, "stream_torque_delta_l_fraction", TORQUE_FRACTION)),
        torque_center_fraction=float(scalar_from_data(data, "stream_torque_center_fraction", TORQUE_CENTER_FRACTION)),
        torque_log_width=float(scalar_from_data(data, "stream_torque_log_width", TORQUE_LOG_WIDTH)),
        wind_sink_fraction=float(scalar_from_data(data, "wind_sink_fraction", 0.0)),
        wind_sink_center_fraction=float(scalar_from_data(data, "wind_sink_center_fraction", 0.8)),
        wind_sink_log_width=float(scalar_from_data(data, "wind_sink_log_width", 0.08)),
        stream_heating_efficiency=float(scalar_from_data(data, "stream_heating_efficiency", 0.0)),
        outer_closure=str(scalar_from_data(data, "outer_closure", "pressure_supported_thin_energy")),
        outer_robin_chi=float(scalar_from_data(data, "outer_robin_chi", 0.0)),
        outer_robin_slope_target=float(scalar_from_data(data, "outer_robin_slope_target", 0.0)),
        outer_robin_slope_scale=float(scalar_from_data(data, "outer_robin_slope_scale", 1.0)),
        outer_buffer_inner_rg=(
            None
            if not np.isfinite(float(scalar_from_data(data, "outer_buffer_inner_rg", np.nan)))
            else float(scalar_from_data(data, "outer_buffer_inner_rg", np.nan))
        ),
        outer_buffer_radial_weight=float(scalar_from_data(data, "outer_buffer_radial_weight", 1.0)),
        outer_buffer_energy_weight=float(scalar_from_data(data, "outer_buffer_energy_weight", 1.0)),
        outer_buffer_boundary_weight=float(scalar_from_data(data, "outer_buffer_boundary_weight", 1.0)),
        outer_buffer_taper_log_width=float(scalar_from_data(data, "outer_buffer_taper_log_width", 0.0)),
        interval_residual_form=str(scalar_from_data(data, "interval_residual_form", INTERVAL_RESIDUAL_FORM)),
        integrated_residual_weighting=str(
            scalar_from_data(data, "integrated_residual_weighting", INTEGRATED_RESIDUAL_WEIGHTING)
        ),
    )
    return z, apply_outer_slopes_from_state(z, params)


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


def relative_root_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def clip_state(z: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    clipped, _count = clip_state_with_count(z, params)
    return clipped


def clip_state_with_count(z: np.ndarray, params: TransonicSlimParams) -> tuple[np.ndarray, int]:
    lower, upper = state_bounds(params)
    array = np.asarray(z, dtype=float)
    clipped = np.clip(array, lower + 1.0e-12, upper - 1.0e-12)
    return clipped, int(np.count_nonzero(clipped != array))


def finite_difference_source_column(anchor_z: np.ndarray, anchor_params: TransonicSlimParams, *, pivot: str) -> tuple[np.ndarray, float]:
    f0 = float(anchor_params.stream_source_fraction)
    step = min(abs(float(TANGENT_FD_STEP)), 0.25 * max(f0, 1.0e-3), 0.25 * max(1.0 - f0, 1.0e-3))
    if step <= 0.0:
        raise ValueError("source finite-difference step collapsed")
    if f0 - step >= 0.0 and f0 + step < 1.0 + anchor_params.wind_sink_fraction:
        plus = replace(anchor_params, stream_source_fraction=f0 + step, stream_mass_fraction=0.0)
        minus = replace(anchor_params, stream_source_fraction=f0 - step, stream_mass_fraction=0.0)
        f_plus = square_collocation_residual(anchor_z, plus, pivot=pivot)
        f_minus = square_collocation_residual(anchor_z, minus, pivot=pivot)
        return (f_plus - f_minus) / (2.0 * step), step
    plus = replace(anchor_params, stream_source_fraction=f0 + step, stream_mass_fraction=0.0)
    f_base = square_collocation_residual(anchor_z, anchor_params, pivot=pivot)
    f_plus = square_collocation_residual(anchor_z, plus, pivot=pivot)
    return (f_plus - f_base) / step, step


def equilibrated_tangent_solve(jac, rhs: np.ndarray) -> np.ndarray:
    try:
        from scipy.sparse import diags
        from scipy.sparse.linalg import lsmr, splu
    except Exception as exc:
        raise RuntimeError("scipy is required for source-fraction tangent prediction") from exc

    if TANGENT_SOLVER == "splu":
        return np.asarray(splu(jac.tocsc(), permc_spec="COLAMD").solve(rhs), dtype=float)
    if TANGENT_SOLVER == "lsmr":
        result = lsmr(
            jac.tocsr(),
            rhs,
            damp=TANGENT_LINEAR_DAMPING,
            atol=1.0e-10,
            btol=1.0e-10,
            maxiter=max(TANGENT_MAXITER, 5 * jac.shape[1]),
        )
        return np.asarray(result[0], dtype=float)
    if TANGENT_SOLVER not in {"equilibrated_lsmr", "equilibrated_direct"}:
        raise ValueError("unknown tangent solver")

    jac_csr = jac.tocsr()
    row_norm = np.sqrt(np.asarray(jac_csr.multiply(jac_csr).sum(axis=1)).ravel())
    row_scale = 1.0 / np.maximum(row_norm, 1.0e-12)
    row_scaled = diags(row_scale) @ jac_csr
    col_norm = np.sqrt(np.asarray(row_scaled.multiply(row_scaled).sum(axis=0)).ravel())
    col_scale = 1.0 / np.maximum(col_norm, 1.0e-12)
    balanced = (row_scaled @ diags(col_scale)).tocsc()
    scaled_rhs = row_scale * np.asarray(rhs, dtype=float)
    if TANGENT_SOLVER == "equilibrated_direct" and TANGENT_LINEAR_DAMPING == 0.0:
        try:
            y = splu(balanced, permc_spec="COLAMD").solve(scaled_rhs)
            return col_scale * np.asarray(y, dtype=float)
        except Exception:
            pass
    result = lsmr(
        balanced,
        scaled_rhs,
        damp=TANGENT_LINEAR_DAMPING,
        atol=1.0e-12,
        btol=1.0e-12,
        maxiter=max(TANGENT_MAXITER, 10 * balanced.shape[1]),
    )
    return col_scale * np.asarray(result[0], dtype=float)


def source_fraction_tangent(anchor_z: np.ndarray, anchor_params: TransonicSlimParams, *, pivot: str) -> tuple[np.ndarray, dict[str, Any]]:
    jac = square_collocation_jacobian(anchor_z, anchor_params, pivot=pivot)
    f_source, fd_step = finite_difference_source_column(anchor_z, anchor_params, pivot=pivot)
    dz_df = equilibrated_tangent_solve(jac, -f_source)
    linear_residual = np.asarray(jac @ dz_df + f_source, dtype=float)
    return dz_df, {
        "predictor_tangent_fd_step": float(fd_step),
        "predictor_tangent_solver": str(TANGENT_SOLVER),
        "predictor_tangent_linear_damping": float(TANGENT_LINEAR_DAMPING),
        "predictor_tangent_norm_inf": float(np.linalg.norm(dz_df, ord=np.inf)),
        "predictor_tangent_norm_l2": float(np.linalg.norm(dz_df)),
        "predictor_tangent_linear_residual_norm": float(np.linalg.norm(linear_residual)),
        "predictor_tangent_linear_residual_inf": float(np.linalg.norm(linear_residual, ord=np.inf)),
    }


def source_fraction_seed(
    *,
    target_fraction: float,
    current_fraction: float,
    current_z: np.ndarray,
    prev_fraction: float | None,
    prev_z: np.ndarray | None,
    params: TransonicSlimParams,
) -> tuple[np.ndarray, str, float, dict[str, Any]]:
    current_seed = np.asarray(current_z, dtype=float)
    current_full = max_residual(current_seed, params)
    best_seed = current_seed
    best_label = "current"
    best_full = current_full
    diagnostics: dict[str, Any] = {
        "predictor_initial_full_current": float(current_full),
        "predictor_initial_full_secant_best": np.nan,
        "predictor_initial_full_tangent_best": np.nan,
        "predictor_chosen": "current",
        "predictor_secant_damping_chosen": np.nan,
        "predictor_tangent_damping_chosen": np.nan,
        "predictor_tangent_fd_step": np.nan,
        "predictor_tangent_solver": str(TANGENT_SOLVER),
        "predictor_tangent_linear_damping": float(TANGENT_LINEAR_DAMPING),
        "predictor_tangent_norm_inf": np.nan,
        "predictor_tangent_norm_l2": np.nan,
        "predictor_tangent_linear_residual_norm": np.nan,
        "predictor_tangent_linear_residual_inf": np.nan,
        "predictor_tangent_error": "",
        "predictor_tangent_secant_cosine": np.nan,
        "predictor_state_clip_count": 0,
        "predictor_secant_clip_count_best": 0,
        "predictor_tangent_clip_count_best": 0,
    }
    secant_direction: np.ndarray | None = None
    if USE_SECANT_PREDICTOR and prev_z is not None and prev_fraction is not None and abs(current_fraction - prev_fraction) > 1.0e-12:
        step_factor = (float(target_fraction) - current_fraction) / (current_fraction - prev_fraction)
        secant_direction = step_factor * (current_z - prev_z)
        for damping in SECANT_DAMPING_VALUES:
            trial_seed, clip_count = clip_state_with_count(current_z + float(damping) * secant_direction, params)
            trial_full = max_residual(trial_seed, params)
            if not np.isfinite(diagnostics["predictor_initial_full_secant_best"]) or trial_full < diagnostics[
                "predictor_initial_full_secant_best"
            ]:
                diagnostics["predictor_initial_full_secant_best"] = float(trial_full)
                diagnostics["predictor_secant_damping_chosen"] = float(damping)
                diagnostics["predictor_secant_clip_count_best"] = int(clip_count)
            if trial_full < best_full:
                best_seed = trial_seed
                best_label = f"secant:{float(damping):g}"
                best_full = trial_full
                diagnostics["predictor_state_clip_count"] = int(clip_count)
    if (
        USE_TANGENT_PREDICTOR
        and best_full > TANGENT_TRIGGER_INITIAL_FULL
        and abs(float(target_fraction) - current_fraction) > 1.0e-14
    ):
        try:
            anchor_params = replace(params, stream_source_fraction=float(current_fraction), stream_mass_fraction=0.0)
            anchor_params = apply_outer_slopes_from_state(current_z, anchor_params)
            pivot = PIVOTS[0] if PIVOTS else "C2"
            dz_df, tangent_info = source_fraction_tangent(current_z, anchor_params, pivot=pivot)
            diagnostics.update(tangent_info)
            if secant_direction is not None:
                tangent_direction = (float(target_fraction) - current_fraction) * dz_df
                denom = float(np.linalg.norm(secant_direction) * np.linalg.norm(tangent_direction))
                diagnostics["predictor_tangent_secant_cosine"] = (
                    float(np.dot(secant_direction, tangent_direction) / denom) if denom > 0.0 else np.nan
                )
            df = float(target_fraction) - current_fraction
            for damping in TANGENT_DAMPING_VALUES:
                trial_seed, clip_count = clip_state_with_count(current_z + float(damping) * df * dz_df, params)
                trial_full = max_residual(trial_seed, params)
                if not np.isfinite(diagnostics["predictor_initial_full_tangent_best"]) or trial_full < diagnostics[
                    "predictor_initial_full_tangent_best"
                ]:
                    diagnostics["predictor_initial_full_tangent_best"] = float(trial_full)
                    diagnostics["predictor_tangent_damping_chosen"] = float(damping)
                    diagnostics["predictor_tangent_clip_count_best"] = int(clip_count)
                if trial_full < best_full:
                    best_seed = trial_seed
                    best_label = f"tangent:{float(damping):g}"
                    best_full = trial_full
                    diagnostics["predictor_state_clip_count"] = int(clip_count)
        except Exception as exc:
            diagnostics["predictor_tangent_error"] = str(exc)
            print(f"  tangent predictor unavailable: {exc}", flush=True)
    diagnostics["predictor_chosen"] = str(best_label)
    diagnostics["predictor_initial_full_best"] = float(best_full)
    return best_seed, best_label, best_full, diagnostics


def polish_best(z0: np.ndarray, params: TransonicSlimParams):
    best = None
    best_full = np.inf
    energy_tol = energy_merit_tol_from_env(NEWTON_ENERGY_MERIT_TOL_RAW, NEWTON_RESIDUAL_TOL)
    energy_l2_tol = energy_merit_tol_from_env(NEWTON_ENERGY_MERIT_L2_TOL_RAW, energy_tol)
    energy_global_tol = energy_merit_tol_from_env(NEWTON_ENERGY_MERIT_GLOBAL_TOL_RAW, energy_tol)
    for pivot in PIVOTS:
        result = solve_square_transonic_polish(
            params,
            z0,
            pivot=pivot,
            method=POLISH_METHOD,
            max_iter=NEWTON_MAX_ITER,
            max_nfev=NEWTON_MAX_NFEV,
            residual_tol=NEWTON_RESIDUAL_TOL,
            use_block_jacobian=True,
            jacobian_rel_step=NEWTON_JACOBIAN_REL_STEP,
            energy_jacobian_rel_step=NEWTON_ENERGY_JACOBIAN_REL_STEP,
            line_search_min_alpha=NEWTON_LINE_SEARCH_MIN_ALPHA,
            line_search_max_reductions=NEWTON_LINE_SEARCH_MAX_REDUCTIONS,
            linear_solver=NEWTON_LINEAR_SOLVER,
            linear_dampings=NEWTON_LINEAR_DAMPINGS,
            max_step_norm=NEWTON_MAX_STEP_NORM,
            energy_merit=NEWTON_ENERGY_MERIT,
            energy_merit_tol=energy_tol,
            energy_merit_l2_tol=energy_l2_tol,
            energy_merit_global_tol=energy_global_tol,
            energy_merit_require_decrease=NEWTON_ENERGY_MERIT_REQUIRE_DECREASE,
            energy_row_priority=NEWTON_ENERGY_ROW_PRIORITY,
        )
        full = max_residual(result.z, params)
        if full < best_full:
            best = result
            best_full = full
        if full <= ANCHOR_TOL:
            break
    if best is None:
        raise RuntimeError("no polish pivots configured")
    return best


def slope_delta(old: tuple[float, float] | None, new: tuple[float, float] | None) -> float:
    if old is None or new is None:
        return np.nan
    return float(np.max(np.abs(np.asarray(new, dtype=float) - np.asarray(old, dtype=float))))


def damped_slopes(
    old: tuple[float, float] | None,
    new: tuple[float, float],
    damping: float,
) -> tuple[float, float]:
    if old is None:
        return new
    old_array = np.asarray(old, dtype=float)
    new_array = np.asarray(new, dtype=float)
    damped = old_array + float(damping) * (new_array - old_array)
    return (float(damped[0]), float(damped[1]))


def polish_with_outer_slope_control(
    z0: np.ndarray,
    params: TransonicSlimParams,
) -> tuple[Any, TransonicSlimParams, dict[str, Any]]:
    polish = polish_best(z0, params)
    total_nfev = int(polish.result.nfev)
    total_iterations = int(polish.iterations)
    solver_params = params
    final_params = apply_outer_slopes_from_state(polish.z, solver_params)
    final_full = max_residual(polish.z, final_params)
    initial_refresh_delta = slope_delta(solver_params.outer_match_log_slopes, final_params.outer_match_log_slopes)
    meta: dict[str, Any] = {
        "outer_picard_enabled": bool(OUTER_SLOPE_PICARD),
        "outer_picard_iterations": 0,
        "outer_picard_damping": np.nan,
        "outer_picard_slope_delta": initial_refresh_delta,
        "outer_picard_final_full": float(final_full),
        "outer_picard_final_outer_omega": float(residual_audit_from_state_vector(polish.z, final_params).outer_omega),
        "polish_nfev_total": int(total_nfev),
        "polish_iterations_total": int(total_iterations),
    }
    if REFRESH_REPOLISH and not OUTER_SLOPE_PICARD:
        polish = polish_best(polish.z, final_params)
        total_nfev += int(polish.result.nfev)
        total_iterations += int(polish.iterations)
        solver_params = final_params
        final_params = apply_outer_slopes_from_state(polish.z, solver_params)
        final_full = max_residual(polish.z, final_params)
        meta.update(
            {
                "outer_picard_iterations": 1,
                "outer_picard_damping": 1.0,
                "outer_picard_slope_delta": slope_delta(
                    solver_params.outer_match_log_slopes,
                    final_params.outer_match_log_slopes,
                ),
                "outer_picard_final_full": float(final_full),
                "outer_picard_final_outer_omega": float(
                    residual_audit_from_state_vector(polish.z, final_params).outer_omega
                ),
                "polish_nfev_total": int(total_nfev),
                "polish_iterations_total": int(total_iterations),
            }
        )
        return polish, final_params, meta
    if not OUTER_SLOPE_PICARD:
        return polish, final_params, meta

    best_polish = polish
    best_solver_params = solver_params
    best_final_params = final_params
    best_full = float(final_full)
    last_damping = np.nan
    for iteration in range(1, max(OUTER_SLOPE_PICARD_MAX_ITER, 0) + 1):
        target_slopes = one_sided_outer_slopes(best_polish.z, best_final_params)
        previous_slopes = best_solver_params.outer_match_log_slopes
        delta = slope_delta(previous_slopes, target_slopes)
        if np.isfinite(delta) and delta <= OUTER_SLOPE_PICARD_SLOPE_TOL and best_full <= OUTER_SLOPE_PICARD_TOL:
            break

        iteration_best: tuple[Any, TransonicSlimParams, TransonicSlimParams, float, float] | None = None
        for damping in OUTER_SLOPE_PICARD_DAMPINGS or (1.0,):
            trial_slopes = damped_slopes(previous_slopes, target_slopes, float(damping))
            trial_solver_params = replace(best_solver_params, outer_match_log_slopes=trial_slopes)
            trial_polish = polish_best(best_polish.z, trial_solver_params)
            total_nfev += int(trial_polish.result.nfev)
            total_iterations += int(trial_polish.iterations)
            trial_final_params = apply_outer_slopes_from_state(trial_polish.z, trial_solver_params)
            trial_full = max_residual(trial_polish.z, trial_final_params)
            if iteration_best is None or trial_full < iteration_best[3]:
                iteration_best = (
                    trial_polish,
                    trial_solver_params,
                    trial_final_params,
                    float(trial_full),
                    float(damping),
                )

        if iteration_best is None:
            break
        improved = iteration_best[3] <= best_full or (best_full <= ACCEPTANCE_TOL and iteration_best[3] <= ACCEPTANCE_TOL)
        if not improved:
            break
        best_polish, best_solver_params, best_final_params, best_full, last_damping = iteration_best
        best_delta = slope_delta(best_solver_params.outer_match_log_slopes, best_final_params.outer_match_log_slopes)
        meta.update(
            {
                "outer_picard_iterations": int(iteration),
                "outer_picard_damping": float(last_damping),
                "outer_picard_slope_delta": best_delta,
                "outer_picard_final_full": float(best_full),
                "outer_picard_final_outer_omega": float(
                    residual_audit_from_state_vector(best_polish.z, best_final_params).outer_omega
                ),
                "polish_nfev_total": int(total_nfev),
                "polish_iterations_total": int(total_iterations),
            }
        )
        if (
            best_full <= OUTER_SLOPE_PICARD_TOL
            and np.isfinite(best_delta)
            and best_delta <= OUTER_SLOPE_PICARD_SLOPE_TOL
        ):
            break
    return best_polish, best_final_params, meta


def physical_energy_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(residual_partition_audit_from_state_vector(z, params).physical_energy_max)


def physical_interval_residuals(z: np.ndarray, params: TransonicSlimParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    residuals = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            for idx in range(len(logR) - 1)
        ],
        dtype=float,
    )
    R_mid_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    if params.outer_buffer_inner_rg is None:
        physical_mask = np.ones(len(R_mid_rg), dtype=bool)
    else:
        physical_mask = R_mid_rg < float(params.outer_buffer_inner_rg)
    return R_mid_rg, residuals, physical_mask


def configured_interval_residuals(z: np.ndarray, params: TransonicSlimParams) -> tuple[np.ndarray, np.ndarray]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    residuals = np.asarray(
        [_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx) for idx in range(len(logR) - 1)],
        dtype=float,
    )
    R_mid_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    return R_mid_rg, residuals


def local_patch_interval_selection(z: np.ndarray, params: TransonicSlimParams, mode: str = "physical") -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode not in {"physical", "global"}:
        raise ValueError("local patch mode must be 'physical' or 'global'")
    if mode == "global":
        R_mid_rg, residuals = configured_interval_residuals(z, params)
        energy = np.abs(residuals[:, 1])
        peak_interval = int(np.argmax(energy))
        selectable = np.ones(len(R_mid_rg), dtype=bool)
    else:
        R_mid_rg, residuals, physical_mask = physical_interval_residuals(z, params)
        physical_indices = np.nonzero(physical_mask)[0]
        if physical_indices.size == 0:
            physical_indices = np.arange(len(R_mid_rg), dtype=int)
        energy = np.abs(residuals[physical_indices, 1])
        peak_interval = int(physical_indices[int(np.argmax(energy))])
        selectable = physical_mask
    selected: set[int] = {peak_interval}
    if LOCAL_PATCH_HALF_WIDTH_RG > 0.0:
        peak_R = float(R_mid_rg[peak_interval])
        window = np.nonzero((np.abs(R_mid_rg - peak_R) <= LOCAL_PATCH_HALF_WIDTH_RG) & selectable)[0]
        selected.update(int(idx) for idx in window)
    if LOCAL_PATCH_TOP_K > 0:
        if mode == "global":
            order = np.argsort(np.abs(residuals[:, 1]))[::-1]
        else:
            order = physical_indices[np.argsort(energy)[::-1]]
        selected.update(int(idx) for idx in order[:LOCAL_PATCH_TOP_K])
    return np.asarray(sorted(selected), dtype=int)


def active_nodes_for_patch(intervals: np.ndarray, n_nodes: int) -> np.ndarray:
    nodes: set[int] = set()
    for idx in intervals:
        start = max(1, int(idx) - LOCAL_PATCH_NODE_PAD)
        stop = min(n_nodes - 2, int(idx) + 1 + LOCAL_PATCH_NODE_PAD)
        nodes.update(range(start, stop + 1))
    active = np.asarray(sorted(nodes), dtype=int)
    if LOCAL_PATCH_MAX_ACTIVE_NODES > 0 and active.size > LOCAL_PATCH_MAX_ACTIVE_NODES:
        center = int(active[active.size // 2])
        order = sorted(active.tolist(), key=lambda node: (abs(node - center), node))
        active = np.asarray(sorted(order[:LOCAL_PATCH_MAX_ACTIVE_NODES]), dtype=int)
    return active


def touched_intervals_for_nodes(nodes: np.ndarray, n_nodes: int) -> np.ndarray:
    intervals: set[int] = set()
    for node in nodes:
        if node > 0:
            intervals.add(int(node) - 1)
        if node < n_nodes - 1:
            intervals.add(int(node))
    return np.asarray(sorted(intervals), dtype=int)


def square_residual_audit(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, str, float]:
    pivot = select_sonic_compatibility_pivot(z, params)
    square = square_collocation_residual(z, params, pivot=pivot)
    unused = unused_sonic_compatibility(z, params, pivot=pivot)
    return float(np.max(np.abs(square))), str(pivot), float(unused)


def local_physical_energy_patch(
    z: np.ndarray,
    params: TransonicSlimParams,
    *,
    mode: str = "physical",
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        raise RuntimeError("scipy is required for local physical-energy patch") from exc

    intervals = local_patch_interval_selection(z, params, mode=mode)
    active_nodes = active_nodes_for_patch(intervals, params.n_nodes)
    if active_nodes.size == 0:
        raise RuntimeError("local patch selected no active nodes")
    solve_intervals = touched_intervals_for_nodes(active_nodes, params.n_nodes)
    logu, logT, logR_son, lambda0, logR = unpack_state(z, params)
    lower, upper = state_bounds(params)
    active_columns = np.concatenate([active_nodes, params.n_nodes + active_nodes])
    x0 = np.concatenate([logu[active_nodes], logT[active_nodes]])
    local_lower = lower[active_columns]
    local_upper = upper[active_columns]
    base_x = np.array(x0, copy=True)

    def unpack_trial(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        trial_logu = np.array(logu, copy=True)
        trial_logT = np.array(logT, copy=True)
        trial_logu[active_nodes] = x[: active_nodes.size]
        trial_logT[active_nodes] = x[active_nodes.size :]
        return trial_logu, trial_logT

    def residual(x: np.ndarray) -> np.ndarray:
        trial_logu, trial_logT = unpack_trial(x)
        pieces: list[float] = []
        for idx in solve_intervals:
            row = _interval_residual_from_unpacked(trial_logu, trial_logT, logR, lambda0, params, int(idx))
            pieces.extend([float(row[0]), LOCAL_PATCH_ENERGY_WEIGHT * float(row[1])])
        if LOCAL_PATCH_PRIOR_WEIGHT > 0.0:
            pieces.extend((LOCAL_PATCH_PRIOR_WEIGHT * (np.asarray(x, dtype=float) - base_x)).tolist())
        return np.asarray(pieces, dtype=float)

    before_local = residual(x0)
    lsq = least_squares(
        residual,
        x0,
        bounds=(local_lower, local_upper),
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=LOCAL_PATCH_MAX_NFEV,
    )
    patched_logu, patched_logT = unpack_trial(np.asarray(lsq.x, dtype=float))
    patched = np.asarray(pack_state(patched_logu, patched_logT, logR_son, lambda0), dtype=float)
    after_local = residual(np.asarray(lsq.x, dtype=float))
    before_square, before_pivot, before_unused = square_residual_audit(z, params)
    after_square, after_pivot, after_unused = square_residual_audit(patched, params)
    info = {
        "local_patch_enabled": True,
        "local_patch_mode": str(mode),
        "local_patch_target_intervals": intervals.tolist(),
        "local_patch_active_nodes": active_nodes.tolist(),
        "local_patch_solve_intervals": solve_intervals.tolist(),
        "local_patch_local_max_before": float(np.max(np.abs(before_local))),
        "local_patch_local_max_after": float(np.max(np.abs(after_local))),
        "local_patch_nfev": int(lsq.nfev),
        "local_patch_cost": float(lsq.cost),
        "local_patch_optimality": float(lsq.optimality),
        "local_patch_success": bool(lsq.success),
        "local_patch_message": str(lsq.message),
        "local_patch_square_before": float(before_square),
        "local_patch_square_after": float(after_square),
        "local_patch_pivot_before": str(before_pivot),
        "local_patch_pivot_after": str(after_pivot),
        "local_patch_unused_before": float(before_unused),
        "local_patch_unused_after": float(after_unused),
    }
    return patched, info


def local_patch_polish(polish, patched_z: np.ndarray, info: dict[str, Any]):
    nfev = int(getattr(polish.result, "nfev", 0)) + int(info.get("local_patch_nfev", 0))
    message = f"{polish.result.message}; local physical-energy patch adopted"
    return SimpleNamespace(
        z=np.asarray(patched_z, dtype=float),
        pivot=str(info.get("local_patch_pivot_after", polish.pivot)),
        method=f"{polish.method}+local_patch",
        result=SimpleNamespace(nfev=nfev, message=message),
        iterations=int(polish.iterations),
        line_search_reductions=int(getattr(polish, "line_search_reductions", 0)),
        final_step_norm=float(getattr(polish, "final_step_norm", 0.0)),
        final_linear_damping=float(getattr(polish, "final_linear_damping", 0.0)),
        newton_audit=tuple(getattr(polish, "newton_audit", ())),
    )


def maybe_local_physical_energy_patch(
    polish,
    params: TransonicSlimParams,
    elapsed: float,
    meta: dict[str, Any],
) -> tuple[Any, TransonicSlimParams, float, dict[str, Any]]:
    meta = dict(meta)
    meta.setdefault("local_patch_enabled", bool(LOCAL_PATCH_ON_REJECT))
    meta.setdefault("local_patch_attempted", False)
    meta.setdefault("local_patch_adopted", False)
    if not LOCAL_PATCH_ON_REJECT:
        return polish, params, elapsed, meta
    if not gate_would_reject(polish.z, params):
        meta["local_patch_skip_reason"] = "base_already_accepted"
        return polish, params, elapsed, meta
    current_z = np.asarray(polish.z, dtype=float)
    current_params = params
    base_full = max_residual(current_z, current_params)
    base_physical = physical_energy_residual(current_z, current_params)
    total_patch_nfev = 0
    adopted_any = False
    last_info: dict[str, Any] = {}
    adopted_info: dict[str, Any] = {}
    max_passes = max(1, int(LOCAL_PATCH_MAX_PASSES))
    for patch_pass in range(max_passes):
        before_full = max_residual(current_z, current_params)
        before_physical = physical_energy_residual(current_z, current_params)
        acceptance_tol, _anchor_tol = acceptance_tolerances_for_params(current_params)
        before_physical_ok = (
            bool(np.isfinite(before_physical) and before_physical <= PHYSICAL_E_TOL) if np.isfinite(PHYSICAL_E_TOL) else True
        )
        mode = "global" if LOCAL_PATCH_GLOBAL_AFTER_PHYSICAL and before_physical_ok and before_full > acceptance_tol else "physical"
        try:
            t0 = time.perf_counter()
            patched_z, patch_info = local_physical_energy_patch(current_z, current_params, mode=mode)
            elapsed += time.perf_counter() - t0
        except Exception as exc:
            meta["local_patch_attempted"] = True
            meta["local_patch_error"] = str(exc)
            print(f"  local patch failed: {exc}", flush=True)
            break
        patched_params = apply_outer_slopes_from_state(patched_z, current_params)
        after_full = max_residual(patched_z, patched_params)
        after_physical = physical_energy_residual(patched_z, patched_params)
        acceptance_tol, _anchor_tol = acceptance_tolerances_for_params(patched_params)
        physical_ok = bool(np.isfinite(after_physical) and after_physical <= PHYSICAL_E_TOL) if np.isfinite(PHYSICAL_E_TOL) else True
        accepted_by_gate = bool(after_full <= acceptance_tol and (physical_ok or not REQUIRE_PHYSICAL_E_GATE))
        improves = bool(after_full <= before_full and after_physical <= before_physical)
        no_full_regression = bool(after_full <= before_full * (1.0 + 1.0e-8) + 1.0e-14)
        physical_improves = bool(after_physical < before_physical)
        adopt = bool((improves or (no_full_regression and physical_improves)) and (accepted_by_gate or after_full <= before_full * (1.0 + 1.0e-8) + 1.0e-14))
        total_patch_nfev += int(patch_info.get("local_patch_nfev", 0))
        patch_info.update(
            {
                "local_patch_attempted": True,
                "local_patch_pass": int(patch_pass + 1),
                "local_patch_passes": int(patch_pass + 1),
                "local_patch_full_before": float(base_full),
                "local_patch_full_after": float(after_full),
                "local_patch_pass_full_before": float(before_full),
                "local_patch_pass_full_after": float(after_full),
                "local_patch_physical_E_before": float(base_physical),
                "local_patch_physical_E_after": float(after_physical),
                "local_patch_pass_physical_E_before": float(before_physical),
                "local_patch_pass_physical_E_after": float(after_physical),
                "local_patch_gate_accepted": bool(accepted_by_gate),
                "local_patch_adopted": bool(adopt),
                "local_patch_acceptance_tol": float(acceptance_tol),
                "local_patch_total_nfev": int(total_patch_nfev),
            }
        )
        last_info = patch_info
        print(
            f"  local patch pass {patch_pass + 1} ({mode}) full={before_full:.3e}->{after_full:.3e} "
            f"physE={before_physical:.3e}->{after_physical:.3e} adopted={adopt}",
            flush=True,
        )
        if not adopt:
            break
        adopted_any = True
        adopted_info = dict(patch_info)
        current_z = np.asarray(patched_z, dtype=float)
        current_params = patched_params
        if accepted_by_gate:
            break
    final_info = adopted_info if adopted_any else last_info
    if final_info:
        final_info["local_patch_passes"] = int(final_info.get("local_patch_pass", 0))
        final_info["local_patch_total_nfev"] = int(total_patch_nfev)
        final_info["local_patch_adopted"] = bool(adopted_any)
        meta.update(final_info)
    if not adopted_any:
        return polish, params, elapsed, meta
    patched_polish = local_patch_polish(polish, current_z, {**final_info, "local_patch_nfev": total_patch_nfev})
    meta["polish_nfev_total"] = int(meta.get("polish_nfev_total", getattr(polish.result, "nfev", 0))) + int(total_patch_nfev)
    return patched_polish, current_params, elapsed, meta


def choose_better_physical_state(
    best_polish,
    best_params: TransonicSlimParams,
    candidate_polish,
    candidate_params: TransonicSlimParams,
) -> bool:
    """Return true when candidate has a better physical audit without losing acceptance."""

    best_physical = physical_energy_residual(best_polish.z, best_params)
    candidate_physical = physical_energy_residual(candidate_polish.z, candidate_params)
    best_full = max_residual(best_polish.z, best_params)
    candidate_full = max_residual(candidate_polish.z, candidate_params)
    if candidate_physical < best_physical and candidate_full <= max(ACCEPTANCE_TOL, 2.0 * best_full):
        return True
    if candidate_physical <= 1.05 * best_physical and candidate_full < best_full:
        return True
    return False


def cleanup_params_for_spec(params: TransonicSlimParams, spec: str) -> tuple[TransonicSlimParams, str]:
    spec = str(spec).strip().lower()
    if spec in {"", "same", "current"}:
        return params, "same"
    pieces = spec.split(":")
    interval_form = pieces[0].strip()
    if interval_form not in {"differential", "integrated", "integrated_physical_energy", "conservative_physical_energy"}:
        raise ValueError(f"unknown cleanup polish interval form {interval_form!r}")
    if len(pieces) > 2:
        raise ValueError(f"cleanup polish spec must be form[:weighting], got {spec!r}")
    weighting = pieces[1].strip() if len(pieces) == 2 and pieces[1].strip() else params.integrated_residual_weighting
    if interval_form in {"differential", "integrated_physical_energy"}:
        weighting = "none"
    if weighting not in {"none", "inverse_sqrt_dx", "inverse_dx"}:
        raise ValueError(f"unknown cleanup polish weighting {weighting!r}")
    return replace(params, interval_residual_form=interval_form, integrated_residual_weighting=weighting), f"{interval_form}:{weighting}"


def maybe_cleanup_repolish(
    polish,
    final_params: TransonicSlimParams,
    elapsed: float,
    meta: dict[str, Any],
) -> tuple[Any, TransonicSlimParams, float, dict[str, Any]]:
    if CLEANUP_REPOLISH_PASSES <= 0:
        return polish, final_params, elapsed, meta
    if CLEANUP_REPOLISH_ONLY_ACCEPTED and max_residual(polish.z, final_params) > ACCEPTANCE_TOL:
        meta.update(
            {
                "cleanup_repolish_enabled": True,
                "cleanup_repolish_specs": ",".join(CLEANUP_POLISH_SPECS),
                "cleanup_repolish_attempted": 0,
                "cleanup_repolish_adopted": 0,
                "cleanup_repolish_adopted_specs": "",
                "cleanup_repolish_skipped_reason": "base_not_accepted",
                "cleanup_repolish_best_physical_E": physical_energy_residual(polish.z, final_params),
            }
        )
        return polish, final_params, elapsed, meta

    base_nfev = int(meta.get("polish_nfev_total", polish.result.nfev))
    base_physical = physical_energy_residual(polish.z, final_params)
    if base_nfev > CLEANUP_REPOLISH_MAX_BASE_NFEV:
        meta.update(
            {
                "cleanup_repolish_enabled": True,
                "cleanup_repolish_specs": ",".join(CLEANUP_POLISH_SPECS),
                "cleanup_repolish_attempted": 0,
                "cleanup_repolish_adopted": 0,
                "cleanup_repolish_adopted_specs": "",
                "cleanup_repolish_skipped_reason": "base_nfev_too_high",
                "cleanup_repolish_best_physical_E": float(base_physical),
            }
        )
        return polish, final_params, elapsed, meta
    if base_physical > CLEANUP_REPOLISH_MAX_BASE_PHYSICAL_E:
        meta.update(
            {
                "cleanup_repolish_enabled": True,
                "cleanup_repolish_specs": ",".join(CLEANUP_POLISH_SPECS),
                "cleanup_repolish_attempted": 0,
                "cleanup_repolish_adopted": 0,
                "cleanup_repolish_adopted_specs": "",
                "cleanup_repolish_skipped_reason": "base_physical_E_too_high",
                "cleanup_repolish_best_physical_E": float(base_physical),
            }
        )
        return polish, final_params, elapsed, meta

    best_polish = polish
    best_params = final_params
    best_physical = base_physical
    cleanup_nfev = 0
    cleanup_iterations = 0
    attempted = 0
    adopted = 0
    cleanup_elapsed = 0.0
    adopted_specs: list[str] = []
    for _pass in range(max(CLEANUP_REPOLISH_PASSES, 0)):
        pass_adopted = False
        for spec in CLEANUP_POLISH_SPECS:
            candidate_input_params, normalized_spec = cleanup_params_for_spec(best_params, spec)
            t0 = time.perf_counter()
            candidate_polish, candidate_params, candidate_meta = polish_with_outer_slope_control(
                best_polish.z,
                candidate_input_params,
            )
            cleanup_elapsed += time.perf_counter() - t0
            attempted += 1
            cleanup_nfev += int(candidate_meta.get("polish_nfev_total", candidate_polish.result.nfev))
            cleanup_iterations += int(candidate_meta.get("polish_iterations_total", candidate_polish.iterations))
            if choose_better_physical_state(best_polish, best_params, candidate_polish, candidate_params):
                best_polish = candidate_polish
                best_params = candidate_params
                best_physical = physical_energy_residual(best_polish.z, best_params)
                adopted += 1
                pass_adopted = True
                adopted_specs.append(normalized_spec)
                if best_physical <= PHYSICAL_E_TOL:
                    break
        if not pass_adopted:
            break

    combined = {
        **meta,
        "cleanup_repolish_enabled": True,
        "cleanup_repolish_specs": ",".join(CLEANUP_POLISH_SPECS),
        "cleanup_repolish_attempted": int(attempted),
        "cleanup_repolish_adopted": int(adopted),
        "cleanup_repolish_adopted_specs": ",".join(adopted_specs),
        "cleanup_repolish_skipped_reason": "",
        "cleanup_repolish_nfev": int(cleanup_nfev),
        "cleanup_repolish_iterations": int(cleanup_iterations),
        "cleanup_repolish_elapsed_s": float(cleanup_elapsed),
        "cleanup_repolish_best_physical_E": float(best_physical),
        "polish_nfev_total": int(meta.get("polish_nfev_total", polish.result.nfev)) + int(cleanup_nfev),
        "polish_iterations_total": int(meta.get("polish_iterations_total", polish.iterations)) + int(cleanup_iterations),
    }
    return best_polish, best_params, elapsed + cleanup_elapsed, combined


def residual_remesh_seed(
    source_z: np.ndarray,
    source_params: TransonicSlimParams,
) -> tuple[np.ndarray, TransonicSlimParams, dict[str, Any]]:
    n_nodes = int(RESIDUAL_REMESH_N_NODES_OVERRIDE) if RESIDUAL_REMESH_N_NODES_OVERRIDE else int(source_params.n_nodes)
    custom_grid_xi, grid_info, _profile_info = residual_remesh_grid_xi(
        source_z,
        source_params,
        n_nodes=n_nodes,
        strength=RESIDUAL_REMESH_STRENGTH,
    )
    target_params = replace(
        source_params,
        n_nodes=n_nodes,
        custom_grid_xi=custom_grid_xi,
        max_nfev=NEWTON_MAX_NFEV,
        residual_tol=NEWTON_RESIDUAL_TOL,
    )
    profile = transonic_profile_from_state_vector(source_z, source_params)
    seed = remap_profile_to_new_sonic_grid(profile, target_params, temperature_mdot_power=0.0, method=REMAP_METHOD)
    return seed, apply_outer_slopes_from_state(seed, target_params), grid_info


def seed_accept_polish(seed: np.ndarray, pivot: str) -> Any:
    return SimpleNamespace(
        z=np.asarray(seed, dtype=float),
        pivot=str(pivot),
        method="seed_accept",
        result=SimpleNamespace(nfev=0, message="predictor seed accepted without Newton polish"),
        iterations=0,
        line_search_reductions=0,
        final_step_norm=0.0,
        final_linear_damping=0.0,
        newton_audit=(),
    )


def polish_with_optional_residual_remesh(
    *,
    seed: np.ndarray,
    params: TransonicSlimParams,
    remesh_after_accept: bool,
    remesh_on_reject: bool,
) -> tuple[np.ndarray, Any, TransonicSlimParams, float, dict[str, Any]]:
    t0 = time.perf_counter()
    if ACCEPT_SEED_TOL > 0.0:
        final_params = apply_outer_slopes_from_state(seed, params)
        seed_final_full = max_residual(seed, final_params)
        if seed_final_full <= ACCEPT_SEED_TOL:
            elapsed = time.perf_counter() - t0
            polish = seed_accept_polish(seed, PIVOTS[0] if PIVOTS else "seed")
            meta = {
                "seed_accept_enabled": True,
                "seed_accept_tol": float(ACCEPT_SEED_TOL),
                "seed_accept_final_full": float(seed_final_full),
                "outer_picard_enabled": bool(OUTER_SLOPE_PICARD),
                "outer_picard_iterations": 0,
                "outer_picard_damping": np.nan,
                "outer_picard_slope_delta": slope_delta(params.outer_match_log_slopes, final_params.outer_match_log_slopes),
                "outer_picard_final_full": float(seed_final_full),
                "outer_picard_final_outer_omega": float(residual_audit_from_state_vector(seed, final_params).outer_omega),
                "polish_nfev_total": 0,
                "polish_iterations_total": 0,
                "residual_remesh_action": "seed_accept",
                "residual_remesh_adopted": False,
                "residual_remesh_initial_full": np.nan,
                "residual_remesh_final_full": np.nan,
            }
            return seed, polish, final_params, elapsed, meta

    polish, final_params, meta = polish_with_outer_slope_control(seed, params)
    elapsed = time.perf_counter() - t0
    final_full = max_residual(polish.z, final_params)
    meta.update(
        {
            "residual_remesh_action": "none",
            "residual_remesh_adopted": False,
            "residual_remesh_initial_full": np.nan,
            "residual_remesh_final_full": np.nan,
        }
    )
    should_try_remesh = (remesh_after_accept and final_full <= ACCEPTANCE_TOL) or (
        remesh_on_reject and final_full > ACCEPTANCE_TOL
    )
    if not should_try_remesh:
        polish, final_params, elapsed, meta = maybe_cleanup_repolish(polish, final_params, elapsed, meta)
        polish, final_params, elapsed, meta = maybe_local_physical_energy_patch(polish, final_params, elapsed, meta)
        return seed, polish, final_params, elapsed, meta

    remesh_seed, remesh_params, grid_info = residual_remesh_seed(polish.z, final_params)
    remesh_initial_full = max_residual(remesh_seed, remesh_params)
    print(
        f"  residual-remesh initial={remesh_initial_full:.3e} "
        f"outer1={grid_info['target_outer_1pct_nodes']} outer5={grid_info['target_outer_5pct_nodes']} "
        f"source_delta={grid_info['source_integral_delta_over_inner']:.3e}",
        flush=True,
    )
    combined_meta = {
        **meta,
        **{f"remesh_{key}": value for key, value in grid_info.items()},
        "residual_remesh_action": "after_accept" if final_full <= ACCEPTANCE_TOL else "after_reject",
        "residual_remesh_adopted": False,
        "residual_remesh_initial_full": float(remesh_initial_full),
        "residual_remesh_final_full": np.nan,
        "residual_remesh_source_integral_delta_over_inner": float(
            grid_info["source_integral_delta_over_inner"]
        ),
        "residual_remesh_target_outer_1pct_nodes": int(grid_info["target_outer_1pct_nodes"]),
        "residual_remesh_target_outer_5pct_nodes": int(grid_info["target_outer_5pct_nodes"]),
        "residual_remesh_target_outer_dx": float(grid_info["target_outer_dx"]),
    }
    if remesh_initial_full > RESIDUAL_REMESH_MAX_INITIAL_FULL:
        combined_meta["residual_remesh_action"] = (
            "after_accept_skip_initial" if final_full <= ACCEPTANCE_TOL else "after_reject_skip_initial"
        )
        print(
            f"  residual-remesh polish skipped: initial {remesh_initial_full:.3e} "
            f"> limit {RESIDUAL_REMESH_MAX_INITIAL_FULL:.3e}",
            flush=True,
        )
        polish, final_params, elapsed, combined_meta = maybe_local_physical_energy_patch(polish, final_params, elapsed, combined_meta)
        return seed, polish, final_params, elapsed, combined_meta

    t1 = time.perf_counter()
    remesh_polish, remesh_final_params, remesh_meta = polish_with_outer_slope_control(remesh_seed, remesh_params)
    elapsed += time.perf_counter() - t1
    remesh_final_full = max_residual(remesh_polish.z, remesh_final_params)
    adopt = remesh_final_full <= final_full or (
        final_full <= ACCEPTANCE_TOL and remesh_final_full <= ACCEPTANCE_TOL
    ) or (final_full > ACCEPTANCE_TOL and remesh_final_full <= ACCEPTANCE_TOL)
    if REQUIRE_PHYSICAL_E_GATE:
        original_physical_E = physical_energy_residual(polish.z, final_params)
        remesh_physical_E = physical_energy_residual(remesh_polish.z, remesh_final_params)
        original_physical_ok = bool(np.isfinite(original_physical_E) and original_physical_E <= PHYSICAL_E_TOL)
        remesh_physical_ok = bool(np.isfinite(remesh_physical_E) and remesh_physical_E <= PHYSICAL_E_TOL)
        if original_physical_ok and not remesh_physical_ok:
            adopt = False
        elif not original_physical_ok and remesh_physical_ok and remesh_final_full <= ACCEPTANCE_TOL:
            adopt = True
        elif not original_physical_ok and not remesh_physical_ok:
            adopt = bool(remesh_physical_E < original_physical_E and remesh_final_full <= max(final_full, ACCEPTANCE_TOL))
        elif original_physical_ok and remesh_physical_ok and remesh_final_full <= ACCEPTANCE_TOL:
            adopt = True
        combined_meta["residual_remesh_original_physical_E"] = float(original_physical_E)
        combined_meta["residual_remesh_physical_E"] = float(remesh_physical_E)
    combined_meta["residual_remesh_adopted"] = bool(adopt)
    combined_meta["residual_remesh_final_full"] = float(remesh_final_full)
    if adopt:
        remesh_meta["polish_nfev_total"] = int(meta["polish_nfev_total"]) + int(remesh_meta["polish_nfev_total"])
        remesh_meta["polish_iterations_total"] = int(meta["polish_iterations_total"]) + int(
            remesh_meta["polish_iterations_total"]
        )
        combined_meta.update(remesh_meta)
        combined_meta["residual_remesh_action"] = "after_accept" if final_full <= ACCEPTANCE_TOL else "after_reject"
        combined_meta["residual_remesh_adopted"] = True
        combined_meta["residual_remesh_initial_full"] = float(remesh_initial_full)
        combined_meta["residual_remesh_final_full"] = float(remesh_final_full)
        remesh_polish, remesh_final_params, elapsed, combined_meta = maybe_cleanup_repolish(
            remesh_polish,
            remesh_final_params,
            elapsed,
            combined_meta,
        )
        remesh_polish, remesh_final_params, elapsed, combined_meta = maybe_local_physical_energy_patch(
            remesh_polish,
            remesh_final_params,
            elapsed,
            combined_meta,
        )
        return remesh_seed, remesh_polish, remesh_final_params, elapsed, combined_meta

    combined_meta["polish_nfev_total"] = int(meta["polish_nfev_total"]) + int(remesh_meta["polish_nfev_total"])
    combined_meta["polish_iterations_total"] = int(meta["polish_iterations_total"]) + int(
        remesh_meta["polish_iterations_total"]
    )
    polish, final_params, elapsed, combined_meta = maybe_cleanup_repolish(polish, final_params, elapsed, combined_meta)
    polish, final_params, elapsed, combined_meta = maybe_local_physical_energy_patch(polish, final_params, elapsed, combined_meta)
    return seed, polish, final_params, elapsed, combined_meta


def angular_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    slopes = params.outer_match_log_slopes
    if slopes is None:
        return {
            "pressure_target": np.nan,
            "achieved_omega_log_offset": np.nan,
            "omega_target_residual": np.nan,
        }
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    ln_omega = float(np.log(profile.Omega[-1] / profile.Omega_K[-1]))
    pressure_target = pressure_supported_omega_target(
        float(logR[-1]),
        np.array([logu[-1], logT[-1]], dtype=float),
        np.asarray(slopes, dtype=float),
        lambda0,
        params,
    )
    return {
        "pressure_target": float(pressure_target),
        "achieved_omega_log_offset": float(ln_omega - pressure_target),
        "omega_target_residual": float(ln_omega - pressure_target),
    }


def stream_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    _logu, _logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    R_mass = float(params.stream_source_center_fraction * params.R_out)
    R_torque = float(params.stream_torque_center_fraction * params.R_out)
    mdot_inner, _dmdot_inner = stream_mass_rate_and_derivative(float(logR[0]), params)
    mdot_outer, dmdot_outer = stream_mass_rate_and_derivative(float(logR[-1]), params)
    mdot_center, dmdot_center = stream_mass_rate_and_derivative(float(np.log(R_mass)), params)
    source_prime = np.asarray([stream_source_prime(float(x), params) for x in logR], dtype=float)
    wind_prime = np.asarray([wind_sink_prime(float(x), params) for x in logR], dtype=float)
    budget_integral = float(np.trapezoid(wind_prime - source_prime, logR))
    budget_error = float((mdot_outer - mdot_inner) - budget_integral)
    budget_scale = max(abs(mdot_outer - mdot_inner), abs(budget_integral), abs(params.Mdot_g_s), 1.0)
    l_ref = float(params.potential.l_k(R_torque))
    stream_l_outer, _stream_l_outer_deriv = stream_torque_specific_l_and_derivative(float(logR[-1]), params)
    return {
        "Rinj_mass_rg": float(R_mass / params.r_g),
        "Rinj_torque_rg": float(R_torque / params.r_g),
        "Mdot_inner_over_param": float(mdot_inner / params.Mdot_g_s),
        "Mdot_outer_over_inner": float(mdot_outer / params.Mdot_g_s),
        "Mdot_center_over_inner": float(mdot_center / params.Mdot_g_s),
        "dMdot_dlnR_outer_over_inner": float(dmdot_outer / params.Mdot_g_s),
        "dMdot_dlnR_center_over_inner": float(dmdot_center / params.Mdot_g_s),
        "stream_source_integral_over_inner": float(np.trapezoid(source_prime, logR) / params.Mdot_g_s),
        "wind_sink_integral_over_inner": float(np.trapezoid(wind_prime, logR) / params.Mdot_g_s),
        "mass_budget_error_over_inner": float(budget_error / params.Mdot_g_s),
        "relative_mass_budget_error": float(abs(budget_error) / budget_scale),
        "stream_l_outer_over_lKinj": float(stream_l_outer / l_ref) if l_ref > 0.0 else np.nan,
    }


def heating_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    q_stream = np.asarray([stream_heating_rate(float(x), params) for x in logR], dtype=float)
    q_visc = np.empty_like(q_stream)
    q_rad = np.empty_like(q_stream)
    q_adv = np.empty_like(q_stream)
    for idx, x in enumerate(logR):
        if idx == 0:
            dx = float(logR[1] - logR[0])
            g = np.array([(logu[1] - logu[0]) / dx, (logT[1] - logT[0]) / dx], dtype=float)
        elif idx == len(logR) - 1:
            dx = float(logR[-1] - logR[-2])
            g = np.array([(logu[-1] - logu[-2]) / dx, (logT[-1] - logT[-2]) / dx], dtype=float)
        else:
            dx = float(logR[idx + 1] - logR[idx - 1])
            g = np.array(
                [(logu[idx + 1] - logu[idx - 1]) / dx, (logT[idx + 1] - logT[idx - 1]) / dx],
                dtype=float,
            )
        qv, qr, qa, _qe = _heating_terms_from_gradient(
            float(x),
            np.array([logu[idx], logT[idx]], dtype=float),
            g,
            lambda0,
            params,
        )
        q_visc[idx] = qv
        q_rad[idx] = qr
        q_adv[idx] = qa

    peak_idx = int(np.argmax(q_stream)) if q_stream.size else 0
    weights = 2.0 * np.pi * np.exp(logR) ** 2
    int_stream = float(np.trapezoid(q_stream * weights, logR))
    int_visc = float(np.trapezoid(np.abs(q_visc) * weights, logR) + 1.0e-300)
    int_rad = float(np.trapezoid(np.abs(q_rad) * weights, logR) + 1.0e-300)
    int_adv = float(np.trapezoid(np.abs(q_adv) * weights, logR) + 1.0e-300)
    return {
        "stream_heating_efficiency": float(params.stream_heating_efficiency),
        "max_Qstream_Qvisc": float(np.max(q_stream / (np.abs(q_visc) + 1.0e-300))),
        "max_Qstream_Qrad": float(np.max(q_stream / (np.abs(q_rad) + 1.0e-300))),
        "max_Qstream_Qadv_abs": float(np.max(q_stream / (np.abs(q_adv) + 1.0e-300))),
        "integrated_Qstream_Qvisc": float(int_stream / int_visc),
        "integrated_Qstream_Qrad": float(int_stream / int_rad),
        "integrated_Qstream_Qadv_abs": float(int_stream / int_adv),
        "peak_Qstream_R_rg": float(np.exp(logR[peak_idx]) / params.r_g),
    }


def trapz_log(values: np.ndarray, R: np.ndarray) -> float:
    logR = np.log(np.asarray(R, dtype=float))
    weights = 2.0 * np.pi * np.asarray(R, dtype=float) ** 2
    return float(np.trapezoid(np.asarray(values, dtype=float) * weights, logR))


def masked_trapz_log(values: np.ndarray, R: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    R = np.asarray(R, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if int(np.count_nonzero(mask)) < 2:
        return np.nan
    return trapz_log(values[mask], R[mask])


def advection_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    profile = transonic_profile_from_state_vector(z, params)
    R = np.asarray(profile.R, dtype=float)
    R_rg = R / params.r_g
    qv = np.asarray(profile.Q_visc, dtype=float)
    qr = np.asarray(profile.Q_rad, dtype=float)
    qa = np.asarray(profile.Q_adv, dtype=float)
    visc = trapz_log(np.abs(qv), R) + 1.0e-300
    rad = trapz_log(qr, R)
    adv = trapz_log(qa, R)
    adv_pos = trapz_log(np.maximum(qa, 0.0), R)
    inner = R_rg <= INNER_RADIUS_RG
    inner_visc = masked_trapz_log(np.abs(qv), R, inner)
    inner_adv = masked_trapz_log(qa, R, inner)
    inner_adv_pos = masked_trapz_log(np.maximum(qa, 0.0), R, inner)
    ledd = eddington_luminosity(params.M2_g, kappa=params.kappa)
    return {
        "f_adv_global": float(adv / visc),
        "f_adv_pos": float(adv_pos / visc),
        "f_adv_inner": float(inner_adv / (inner_visc + 1.0e-300)) if np.isfinite(inner_visc) else np.nan,
        "f_adv_inner_pos": float(inner_adv_pos / (inner_visc + 1.0e-300)) if np.isfinite(inner_visc) else np.nan,
        "Lrad_LEdd": float(rad / ledd),
    }


def interval_peak_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    intervals = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            for idx in range(len(logR) - 1)
        ],
        dtype=float,
    )
    R_mid = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    peak_R = int(np.argmax(np.abs(intervals[:, 0])))
    peak_E = int(np.argmax(np.abs(intervals[:, 1])))
    return {
        "peak_interval_R_rg": float(R_mid[peak_R]),
        "peak_interval_R_value": float(intervals[peak_R, 0]),
        "peak_interval_E_rg": float(R_mid[peak_E]),
        "peak_interval_E_value": float(intervals[peak_E, 1]),
        "median_abs_interval_E": float(np.median(np.abs(intervals[:, 1]))),
        "p90_abs_interval_E": float(np.quantile(np.abs(intervals[:, 1]), 0.9)),
    }


def partition_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float | int]:
    partition = residual_partition_audit_from_state_vector(z, params)
    return {
        "partition_buffer_inner_rg": float(partition.buffer_inner_rg),
        "partition_physical_interval_count": int(partition.physical_interval_count),
        "partition_buffer_interval_count": int(partition.buffer_interval_count),
        "partition_physical_R": float(partition.physical_radial_max),
        "partition_physical_E": float(partition.physical_energy_max),
        "partition_physical_R_l2": float(partition.physical_radial_l2),
        "partition_physical_E_l2": float(partition.physical_energy_l2),
        "partition_buffer_R": float(partition.buffer_radial_max),
        "partition_buffer_E": float(partition.buffer_energy_max),
        "partition_buffer_R_l2": float(partition.buffer_radial_l2),
        "partition_buffer_E_l2": float(partition.buffer_energy_l2),
        "partition_terminal_omega": float(partition.terminal_omega),
        "partition_terminal_energy": float(partition.terminal_energy),
        "partition_peak_physical_E_rg": float(partition.peak_physical_energy_rg),
        "partition_peak_buffer_E_rg": float(partition.peak_buffer_energy_rg),
    }


def newton_audit_rows(polish) -> list[dict[str, Any]]:
    return [asdict(item) for item in getattr(polish, "newton_audit", ())]


def newton_audit_diagnostic(polish) -> dict[str, Any]:
    rows = newton_audit_rows(polish)
    if not rows:
        return {
            "newton_audit_rows": 0,
            "newton_audit_accepted_trials": 0,
            "newton_audit_total_jacobian_s": np.nan,
            "newton_audit_max_jacobian_s": np.nan,
            "newton_audit_total_linear_iterations": 0,
            "newton_audit_max_linear_iterations": 0,
            "newton_audit_max_linear_conda": np.nan,
            "newton_audit_total_linear_s": np.nan,
            "newton_audit_total_line_search_s": np.nan,
            "newton_audit_total_line_search_residual_s": np.nan,
            "newton_audit_total_line_search_energy_s": np.nan,
            "newton_audit_min_physical_energy_after": np.nan,
            "newton_audit_final_physical_energy_after": np.nan,
            "newton_audit_min_energy_merit_after": np.nan,
            "newton_audit_path": "",
        }
    accepted = [row for row in rows if bool(row["accepted"])]
    linear_conda = [float(row["linear_conda"]) for row in rows if np.isfinite(float(row["linear_conda"]))]
    physical_after = [
        float(row.get("physical_energy_after", np.nan))
        for row in rows
        if np.isfinite(float(row.get("physical_energy_after", np.nan)))
    ]
    merit_after = [
        float(row.get("energy_merit_after", np.nan))
        for row in rows
        if np.isfinite(float(row.get("energy_merit_after", np.nan)))
    ]
    linear_s = [float(row.get("linear_solve_s", np.nan)) for row in rows if np.isfinite(float(row.get("linear_solve_s", np.nan)))]
    line_search_s = [
        float(row.get("line_search_s", np.nan)) for row in rows if np.isfinite(float(row.get("line_search_s", np.nan)))
    ]
    line_search_residual_s = [
        float(row.get("line_search_residual_s", np.nan))
        for row in rows
        if np.isfinite(float(row.get("line_search_residual_s", np.nan)))
    ]
    line_search_energy_s = [
        float(row.get("line_search_energy_s", np.nan))
        for row in rows
        if np.isfinite(float(row.get("line_search_energy_s", np.nan)))
    ]
    return {
        "newton_audit_rows": int(len(rows)),
        "newton_audit_accepted_trials": int(len(accepted)),
        "newton_audit_total_jacobian_s": float(sum(float(row["jacobian_build_s"]) for row in rows)),
        "newton_audit_max_jacobian_s": float(max(float(row["jacobian_build_s"]) for row in rows)),
        "newton_audit_total_linear_iterations": int(sum(int(row["linear_iterations"]) for row in rows)),
        "newton_audit_max_linear_iterations": int(max(int(row["linear_iterations"]) for row in rows)),
        "newton_audit_max_linear_conda": float(max(linear_conda)) if linear_conda else np.nan,
        "newton_audit_total_linear_s": float(sum(linear_s)) if linear_s else np.nan,
        "newton_audit_total_line_search_s": float(sum(line_search_s)) if line_search_s else np.nan,
        "newton_audit_total_line_search_residual_s": float(sum(line_search_residual_s)) if line_search_residual_s else np.nan,
        "newton_audit_total_line_search_energy_s": float(sum(line_search_energy_s)) if line_search_energy_s else np.nan,
        "newton_audit_min_physical_energy_after": float(min(physical_after)) if physical_after else np.nan,
        "newton_audit_final_physical_energy_after": float(physical_after[-1]) if physical_after else np.nan,
        "newton_audit_min_energy_merit_after": float(min(merit_after)) if merit_after else np.nan,
        "newton_audit_path": "",
    }


def write_newton_audit(row: dict[str, Any], polish) -> str:
    rows = newton_audit_rows(polish)
    if NEWTON_AUDIT_DIR is None or not rows:
        return ""
    NEWTON_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    safe_branch = str(row["branch"]).replace(".", "p").replace("-", "m")
    safe_mass = f"{float(row['mass_fraction']):.9g}".replace(".", "p").replace("-", "m")
    path = NEWTON_AUDIT_DIR / f"{safe_branch}_mass_{safe_mass}_newton_audit.json"
    payload = {
        "branch": row["branch"],
        "mass_fraction": float(row["mass_fraction"]),
        "predictor": row.get("predictor", ""),
        "predictor_initial_full": row.get("predictor_initial_full", np.nan),
        "final_full": float(row["final_full"]),
        "accepted": bool(row["accepted"]),
        "anchor_eligible": bool(row["anchor_eligible"]),
        "nfev": int(row["nfev"]),
        "polish_nfev_total": int(row.get("polish_nfev_total", row["nfev"])),
        "pivot": str(row["pivot"]),
        "iterations": int(row["iterations"]),
        "line_search_reductions": int(getattr(polish, "line_search_reductions", 0)),
        "final_step_norm": float(getattr(polish, "final_step_norm", np.nan)),
        "final_linear_damping": float(getattr(polish, "final_linear_damping", np.nan)),
        "newton_audit": rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    return relative_root_path(path)


def gate_would_reject(z: np.ndarray, params: TransonicSlimParams) -> bool:
    full = max_residual(z, params)
    acceptance_tol, _anchor_tol = acceptance_tolerances_for_params(params)
    solver_accepted = bool(full <= acceptance_tol)
    if not REQUIRE_PHYSICAL_E_GATE:
        return not solver_accepted
    physical_E = physical_energy_residual(z, params)
    physical_ok = bool(np.isfinite(physical_E) and physical_E <= PHYSICAL_E_TOL)
    return not (solver_accepted and physical_ok)


def acceptance_tolerances_for_params(params: TransonicSlimParams) -> tuple[float, float]:
    if REQUIRE_PHYSICAL_E_GATE and params.interval_residual_form == "integrated_physical_energy":
        hybrid_tol = float(PHYSICAL_E_TOL)
        return max(float(ACCEPTANCE_TOL), hybrid_tol), max(float(ANCHOR_TOL), hybrid_tol)
    return float(ACCEPTANCE_TOL), float(ANCHOR_TOL)


def row_for_result(
    *,
    branch: str,
    mass_fraction: float,
    seed: np.ndarray,
    z: np.ndarray,
    params: TransonicSlimParams,
    polish,
    elapsed_s: float,
    extra: dict[str, Any] | None = None,
    lean_diagnostics: bool = False,
) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(z, params)
    stream_info = stream_diagnostic(z, params)
    full = max_residual(z, params)
    acceptance_tol, anchor_tol = acceptance_tolerances_for_params(params)
    if lean_diagnostics:
        _logu, _logT, logR_son, _lambda0, _logR = unpack_state(z, params)
        profile_info: dict[str, Any] = {
            "Rson_rg": float(np.exp(logR_son) / params.r_g),
            "max_H_R": np.nan,
            "integrated_adv": np.nan,
        }
        advection_info: dict[str, Any] = {
            "f_adv_global": np.nan,
            "f_adv_pos": np.nan,
            "f_adv_inner": np.nan,
            "f_adv_inner_pos": np.nan,
            "Lrad_LEdd": np.nan,
        }
        interval_info: dict[str, Any] = {
            "peak_interval_R_rg": np.nan,
            "peak_interval_R_value": np.nan,
            "peak_interval_E_rg": np.nan,
            "peak_interval_E_value": np.nan,
            "median_abs_interval_E": np.nan,
            "p90_abs_interval_E": np.nan,
        }
        angular_info: dict[str, Any] = {
            "pressure_target": np.nan,
            "achieved_omega_log_offset": np.nan,
            "omega_target_residual": np.nan,
        }
        heating_info: dict[str, Any] = {
            "stream_heating_efficiency": float(params.stream_heating_efficiency),
            "max_Qstream_Qvisc": np.nan,
            "max_Qstream_Qrad": np.nan,
            "max_Qstream_Qadv_abs": np.nan,
            "integrated_Qstream_Qvisc": np.nan,
            "integrated_Qstream_Qrad": np.nan,
            "integrated_Qstream_Qadv_abs": np.nan,
            "peak_Qstream_R_rg": np.nan,
        }
    else:
        profile = transonic_profile_from_state_vector(z, params)
        profile_info = {
            "Rson_rg": float(profile.sonic_radius / params.r_g),
            "max_H_R": float(np.max(profile.H_over_R)),
            "integrated_adv": float(profile.integrated_advective_fraction),
        }
        advection_info = advection_diagnostic(z, params)
        interval_info = interval_peak_diagnostic(z, params)
        angular_info = angular_diagnostic(z, params)
        heating_info = heating_diagnostic(z, params)
    return {
        "branch": branch,
        "mass_fraction": float(mass_fraction),
        "torque_fraction": float(params.stream_torque_delta_l_fraction),
        "ratio": float(params.mdot_edd_ratio),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "grid_power": float(params.grid_power),
        "mass_center_fraction": float(params.stream_source_center_fraction),
        "mass_log_width": float(params.stream_source_log_width),
        "mass_source_shape": str(params.stream_source_shape),
        "mass_source_shape_blend": float(params.stream_source_shape_blend),
        "interval_residual_form": str(params.interval_residual_form),
        "integrated_residual_weighting": str(params.integrated_residual_weighting),
        "newton_energy_merit": str(NEWTON_ENERGY_MERIT),
        "newton_energy_merit_tol": float(energy_merit_tol_from_env(NEWTON_ENERGY_MERIT_TOL_RAW, NEWTON_RESIDUAL_TOL)),
        "newton_energy_merit_l2_tol": float(
            energy_merit_tol_from_env(NEWTON_ENERGY_MERIT_L2_TOL_RAW, NEWTON_RESIDUAL_TOL)
        ),
        "newton_energy_merit_global_tol": float(
            energy_merit_tol_from_env(NEWTON_ENERGY_MERIT_GLOBAL_TOL_RAW, NEWTON_RESIDUAL_TOL)
        ),
        "newton_energy_merit_require_decrease": bool(NEWTON_ENERGY_MERIT_REQUIRE_DECREASE),
        "newton_energy_row_priority": float(NEWTON_ENERGY_ROW_PRIORITY),
        **heating_info,
        "newton_jacobian_rel_step": float(NEWTON_JACOBIAN_REL_STEP),
        "newton_energy_jacobian_rel_step": np.nan
        if NEWTON_ENERGY_JACOBIAN_REL_STEP is None
        else float(NEWTON_ENERGY_JACOBIAN_REL_STEP),
        "outer_buffer_inner_rg": np.nan if params.outer_buffer_inner_rg is None else float(params.outer_buffer_inner_rg),
        "outer_buffer_radial_weight": float(params.outer_buffer_radial_weight),
        "outer_buffer_energy_weight": float(params.outer_buffer_energy_weight),
        "outer_buffer_boundary_weight": float(params.outer_buffer_boundary_weight),
        "outer_buffer_taper_log_width": float(params.outer_buffer_taper_log_width),
        "effective_source_shape": str(params.stream_source_shape),
        "effective_source_shape_blend": float(params.stream_source_shape_blend),
        "effective_torque_fraction": float(params.stream_torque_delta_l_fraction),
        "effective_Rinj_rg": float(stream_info["Rinj_mass_rg"]),
        "effective_torque_Rinj_rg": float(stream_info["Rinj_torque_rg"]),
        "effective_outer_buffer_inner_rg": np.nan if params.outer_buffer_inner_rg is None else float(params.outer_buffer_inner_rg),
        "effective_outer_buffer_radial_weight": float(params.outer_buffer_radial_weight),
        "effective_outer_buffer_energy_weight": float(params.outer_buffer_energy_weight),
        "effective_outer_buffer_boundary_weight": float(params.outer_buffer_boundary_weight),
        "effective_outer_closure": str(params.outer_closure),
        "anchor_checkpoint": relative_root_path(ANCHOR_CHECKPOINT),
        "initial_full": max_residual(seed, params),
        "final_full": full,
        "accepted": bool(full <= acceptance_tol),
        "anchor_eligible": bool(full <= anchor_tol),
        "effective_acceptance_tol": float(acceptance_tol),
        "effective_anchor_tol": float(anchor_tol),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        **angular_info,
        **stream_info,
        "lambda0_over_lK_isco": float(audit.lambda0_over_lK_isco),
        **profile_info,
        **advection_info,
        **interval_info,
        **partition_diagnostic(z, params),
        "outer_H_R": float(audit.outer_H_over_R),
        "outer_Qadv_Qvisc": float(audit.outer_Qadv_over_Qvisc),
        "pivot": str(polish.pivot),
        "method": str(polish.method),
        "nfev": int(polish.result.nfev),
        "iterations": int(polish.iterations),
        **newton_audit_diagnostic(polish),
        "elapsed_s": float(elapsed_s),
        "message": str(polish.result.message),
        "lean_diagnostics": bool(lean_diagnostics),
        **(extra or {}),
        "z": np.asarray(z, dtype=float),
        "custom_grid_xi": np.asarray(params.custom_grid_xi, dtype=float)
        if params.custom_grid_xi is not None
        else np.asarray([], dtype=float),
    }


def apply_physical_gate(row: dict[str, Any]) -> dict[str, Any]:
    physical_E = float(row.get("partition_physical_E", np.nan))
    physical_ok = bool(np.isfinite(physical_E) and physical_E <= PHYSICAL_E_TOL) if np.isfinite(PHYSICAL_E_TOL) else True
    solver_accepted = bool(row["accepted"])
    solver_anchor = bool(row["anchor_eligible"])
    row["solver_accepted"] = solver_accepted
    row["solver_anchor_eligible"] = solver_anchor
    row["physical_E_gate_enabled"] = bool(REQUIRE_PHYSICAL_E_GATE)
    row["physical_E_tol"] = float(PHYSICAL_E_TOL)
    row["physical_E_gate_eligible"] = bool(physical_ok)
    if REQUIRE_PHYSICAL_E_GATE:
        row["accepted"] = bool(solver_accepted and physical_ok)
        row["anchor_eligible"] = bool(solver_anchor and physical_ok)
    return row


def save_checkpoint(row: dict[str, Any], params: TransonicSlimParams) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_branch = str(row["branch"]).replace(".", "p").replace("-", "m")
    safe_mass = f"{float(row['mass_fraction']):.9g}".replace(".", "p").replace("-", "m")
    safe_heat = f"{float(row.get('stream_heating_efficiency', params.stream_heating_efficiency)):.9g}".replace(".", "p").replace("-", "m")
    stem = f"{safe_branch}_mass_{safe_mass}_heat_{safe_heat}_torque_{float(row['torque_fraction']):.4g}_mdot_{float(row['ratio']):.8g}_N{int(row['N'])}".replace(
        ".", "p"
    ).replace("-", "m")
    slopes = params.outer_match_log_slopes
    payload = {key: value for key, value in row.items() if key not in {"z", "custom_grid_xi"}}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(row["R_out_rg"]),
        n_nodes=np.array(row["N"]),
        grid_power=np.array(params.grid_power),
        custom_grid_xi=np.asarray(row["custom_grid_xi"], dtype=float),
        outer_closure=np.array(params.outer_closure),
        outer_match_log_slopes=np.asarray([np.nan, np.nan] if slopes is None else slopes, dtype=float),
        outer_robin_chi=np.array(params.outer_robin_chi),
        outer_robin_slope_target=np.array(params.outer_robin_slope_target),
        outer_robin_slope_scale=np.array(params.outer_robin_slope_scale),
        outer_buffer_inner_rg=np.array(np.nan if params.outer_buffer_inner_rg is None else params.outer_buffer_inner_rg),
        outer_buffer_radial_weight=np.array(params.outer_buffer_radial_weight),
        outer_buffer_energy_weight=np.array(params.outer_buffer_energy_weight),
        outer_buffer_boundary_weight=np.array(params.outer_buffer_boundary_weight),
        outer_buffer_taper_log_width=np.array(params.outer_buffer_taper_log_width),
        stream_torque_delta_l_fraction=np.array(params.stream_torque_delta_l_fraction),
        stream_torque_center_fraction=np.array(params.stream_torque_center_fraction),
        stream_torque_log_width=np.array(params.stream_torque_log_width),
        stream_source_fraction=np.array(params.stream_source_fraction),
        stream_source_center_fraction=np.array(params.stream_source_center_fraction),
        stream_source_log_width=np.array(params.stream_source_log_width),
        stream_source_shape=np.array(params.stream_source_shape),
        stream_source_shape_blend=np.array(params.stream_source_shape_blend),
        stream_mass_fraction=np.array(params.stream_mass_fraction),
        stream_mass_center_fraction=np.array(params.stream_mass_center_fraction),
        stream_mass_log_width=np.array(params.stream_mass_log_width),
        wind_sink_fraction=np.array(params.wind_sink_fraction),
        wind_sink_center_fraction=np.array(params.wind_sink_center_fraction),
        wind_sink_log_width=np.array(params.wind_sink_log_width),
        stream_heating_efficiency=np.array(params.stream_heating_efficiency),
        interval_residual_form=np.array(params.interval_residual_form),
        integrated_residual_weighting=np.array(params.integrated_residual_weighting),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted"]),
        branch=np.array(row["branch"]),
        row_json=np.array(json.dumps(json_safe(payload), sort_keys=True)),
    )


def write_table(rows: list[dict[str, Any]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    meta_row = rows[0] if rows else {}
    lines = [
        "# Standard Slim Stream-Mass Annulus Scan",
        "",
        "Generated by `scripts/run_standard_slim_stream_mass_annulus_scan.py`.",
        "",
        f"Anchor `{relative_root_path(ANCHOR_CHECKPOINT)}`, branches `{';'.join(BRANCH_SPECS)}`, "
        f"heating efficiencies `{HEATING_EFFICIENCIES_RAW or 'off'}`, "
        f"Rinj/Rout `{MASS_CENTER_FRACTION:g}`, log width `{MASS_LOG_WIDTH:g}`, source shape `{MASS_SOURCE_SHAPE}`, "
        f"source blend `{MASS_SOURCE_SHAPE_BLEND_OVERRIDE or 'checkpoint/default'}`, "
        f"torque fraction `{TORQUE_FRACTION:g}`, polish method `{POLISH_METHOD}`, "
        f"Jacobian rel step `{NEWTON_JACOBIAN_REL_STEP:g}`, energy Jacobian rel step "
        f"`{NEWTON_ENERGY_JACOBIAN_REL_STEP if NEWTON_ENERGY_JACOBIAN_REL_STEP is not None else 'same'}`, "
        f"refresh repolish `{REFRESH_REPOLISH}`, "
        f"energy merit `{NEWTON_ENERGY_MERIT}`, energy merit tol "
        f"`{energy_merit_tol_from_env(NEWTON_ENERGY_MERIT_TOL_RAW, NEWTON_RESIDUAL_TOL):g}`, "
        f"energy row priority `{NEWTON_ENERGY_ROW_PRIORITY:g}`, "
        f"energy decrease guard `{NEWTON_ENERGY_MERIT_REQUIRE_DECREASE}`, "
        f"cleanup passes `{CLEANUP_REPOLISH_PASSES}`, cleanup specs `{','.join(CLEANUP_POLISH_SPECS)}`, "
        f"lean rejects `{LEAN_REJECT_DIAGNOSTICS}`, physical gate `{REQUIRE_PHYSICAL_E_GATE}` "
        f"at `{PHYSICAL_E_TOL:g}`, "
        f"residual remesh every step `{RESIDUAL_REMESH_EVERY_STEP}`, remesh on reject `{RESIDUAL_REMESH_ON_REJECT}`, "
        f"outer-slope Picard `{OUTER_SLOPE_PICARD}`, interval form `{INTERVAL_RESIDUAL_FORM}`, "
        f"integrated weighting `{INTEGRATED_RESIDUAL_WEIGHTING}`, "
        f"forced interval `{FORCE_INTERVAL_RESIDUAL_FORM or 'off'}`, "
        f"forced weighting `{FORCE_INTEGRATED_RESIDUAL_WEIGHTING or 'off'}`, "
        f"Rout override `{R_OUT_RG_OVERRIDE or 'checkpoint'}`, fixed Rinj `{FIXED_RINJ_RG_OVERRIDE or 'fraction'}`, "
        f"fixed torque Rinj `{FIXED_TORQUE_RINJ_RG_OVERRIDE or ('source' if FIXED_RINJ_RG_OVERRIDE and not TORQUE_CENTER_FRACTION_OVERRIDE else 'fraction')}`, "
        f"outer buffer inner `{OUTER_BUFFER_INNER_RG_OVERRIDE or 'off'}`, "
        f"buffer weights `(R,E,B)=({OUTER_BUFFER_RADIAL_WEIGHT:g},{OUTER_BUFFER_ENERGY_WEIGHT:g},{OUTER_BUFFER_BOUNDARY_WEIGHT:g})`, "
        f"buffer taper `{OUTER_BUFFER_TAPER_LOG_WIDTH:g}`.",
        "",
        "Effective inherited metadata:",
        "",
        f"- source shape `{meta_row.get('effective_source_shape', 'n/a')}`, "
        f"source blend `{fmt(meta_row.get('effective_source_shape_blend', np.nan))}`, "
        f"torque fraction `{fmt(meta_row.get('effective_torque_fraction', np.nan))}`",
        f"- Rout `{fmt(meta_row.get('R_out_rg', np.nan))} rg`, "
        f"Rinj `{fmt(meta_row.get('effective_Rinj_rg', np.nan))} rg`, "
        f"torque Rinj `{fmt(meta_row.get('effective_torque_Rinj_rg', np.nan))} rg`",
        f"- outer closure `{meta_row.get('effective_outer_closure', 'n/a')}`, "
        f"R_buffer `{fmt(meta_row.get('effective_outer_buffer_inner_rg', np.nan))} rg`, "
        f"buffer weights `(R,E,B)=({fmt(meta_row.get('effective_outer_buffer_radial_weight', np.nan))},"
        f"{fmt(meta_row.get('effective_outer_buffer_energy_weight', np.nan))},"
        f"{fmt(meta_row.get('effective_outer_buffer_boundary_weight', np.nan))})`",
        f"- anchor checkpoint `{meta_row.get('anchor_checkpoint', relative_root_path(ANCHOR_CHECKPOINT))}`, "
        f"anchor source fraction `{fmt(meta_row.get('anchor_source_fraction', np.nan))}`, "
        f"anchor Rout `{fmt(meta_row.get('anchor_Rout_rg', np.nan))} rg`",
        "",
        "| branch | source fraction | source shape | source blend | torque fraction | heat eta | max Qs/Qv | int Qs/Qv | peak Qs R/rg | predictor | init current | init secant | init tangent | tan damp | tan norm inf | tan linres | clip | step | next step | cost action | remesh | Picard iters | nfev total | Mdot outer/inner | Mdot center/inner | source integral | rel budget err | Rout/rg | Rinj/rg | initial full | final full | accepted | anchor | solver accepted | phys ok | dominant | int R | int E | peak E R/rg | median abs E | phys E | phys tol | buffer E | peak phys E R/rg | peak buffer E R/rg | outer omega | f_adv global | f_adv inner | f_adv pos | Lrad/LEdd | max H/R | int adv | Rson/rg | pivot | nfev | elapsed s | message |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        display_row = {
            "predictor": "-",
            "predictor_initial_full_current": np.nan,
            "predictor_initial_full_secant_best": np.nan,
            "predictor_initial_full_tangent_best": np.nan,
            "predictor_tangent_damping_chosen": np.nan,
            "predictor_tangent_norm_inf": np.nan,
            "predictor_tangent_linear_residual_norm": np.nan,
            "predictor_state_clip_count": 0,
            "attempt_step": np.nan,
            "next_step": np.nan,
            "cost_action": "-",
            "residual_remesh_action": "-",
            "residual_remesh_adopted": False,
            "outer_picard_iterations": 0,
            "polish_nfev_total": row.get("nfev", np.nan),
            "solver_accepted": row.get("accepted", False),
            "physical_E_gate_eligible": True,
            "physical_E_tol": PHYSICAL_E_TOL,
            "stream_heating_efficiency": 0.0,
            "max_Qstream_Qvisc": 0.0,
            "integrated_Qstream_Qvisc": 0.0,
            "peak_Qstream_R_rg": np.nan,
            "partition_physical_E": np.nan,
            "partition_buffer_E": np.nan,
            "partition_peak_physical_E_rg": np.nan,
            "partition_peak_buffer_E_rg": np.nan,
            **row,
        }
        formatted = {
            key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value
            for key, value in display_row.items()
        }
        for key in ("Mdot_outer_over_inner", "Mdot_center_over_inner"):
            formatted[key] = f"{float(display_row[key]):.6g}"
        lines.append(
            "| {branch} | {mass_fraction} | {mass_source_shape} | {mass_source_shape_blend} | {torque_fraction} | "
            "{stream_heating_efficiency} | {max_Qstream_Qvisc} | {integrated_Qstream_Qvisc} | {peak_Qstream_R_rg} | {predictor} | "
            "{predictor_initial_full_current} | {predictor_initial_full_secant_best} | {predictor_initial_full_tangent_best} | "
            "{predictor_tangent_damping_chosen} | {predictor_tangent_norm_inf} | {predictor_tangent_linear_residual_norm} | "
            "{predictor_state_clip_count} | {attempt_step} | {next_step} | {cost_action} | "
            "{residual_remesh_action}:{residual_remesh_adopted} | {outer_picard_iterations} | {polish_nfev_total} | "
            "{Mdot_outer_over_inner} | {Mdot_center_over_inner} | "
            "{stream_source_integral_over_inner} | {relative_mass_budget_error} | {R_out_rg} | {Rinj_mass_rg} | "
            "{initial_full} | {final_full} | {accepted} | {anchor_eligible} | "
            "{solver_accepted} | {physical_E_gate_eligible} | "
            "{dominant} | {interval_R} | {interval_E} | {peak_interval_E_rg} | {median_abs_interval_E} | "
            "{partition_physical_E} | {physical_E_tol} | {partition_buffer_E} | {partition_peak_physical_E_rg} | {partition_peak_buffer_E_rg} | "
            "{outer_omega} | {f_adv_global} | {f_adv_inner} | {f_adv_pos} | {Lrad_LEdd} | {max_H_R} | "
            "{integrated_adv} | {Rson_rg} | {pivot} | {nfev} | {elapsed_s} | {message} |".format(**formatted).replace("\n", " ")
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(
        json.dumps(json_safe([{key: value for key, value in row.items() if key not in {"z", "custom_grid_xi"}} for row in rows]), indent=2, sort_keys=True)
        + "\n"
    )


def write_figure(rows: list[dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    if not rows:
        return
    width, height = 1000, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = 90, 80, 930, 540
    draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
    fractions = np.asarray([float(row["mass_fraction"]) for row in rows], dtype=float)
    heating = np.asarray([float(row.get("stream_heating_efficiency", 0.0)) for row in rows], dtype=float)
    use_heating_axis = bool(np.max(fractions) - np.min(fractions) < 1.0e-12 and np.max(heating) - np.min(heating) > 0.0)
    x_values = heating if use_heating_axis else fractions
    residuals = np.log10(np.maximum(np.asarray([float(row["final_full"]) for row in rows], dtype=float), 1.0e-16))
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    if x_max <= x_min:
        x_min -= 0.5
        x_max += 0.5
    y_min, y_max = float(np.floor(np.min(residuals))), float(np.ceil(np.max(residuals)))
    if y_max <= y_min:
        y_max = y_min + 1.0
    for branch in sorted(set(str(row["branch"]) for row in rows)):
        selected = sorted(
            [row for row in rows if row["branch"] == branch],
            key=lambda row: float(row.get("stream_heating_efficiency", 0.0)) if use_heating_axis else float(row["mass_fraction"]),
        )
        points = []
        for row in selected:
            xx = float(row.get("stream_heating_efficiency", 0.0)) if use_heating_axis else float(row["mass_fraction"])
            yy = np.log10(max(float(row["final_full"]), 1.0e-16))
            px = x0 + int((xx - x_min) / (x_max - x_min) * (x1 - x0))
            py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
            points.append((px, py))
        color = (31, 119, 180)
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
    for tol, label in ((ACCEPTANCE_TOL, "accept"), (ANCHOR_TOL, "anchor")):
        yy = np.log10(tol)
        py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
        draw.line((x0, py, x1, py), fill=(120, 120, 120), width=1)
        draw.text((x1 - 72, py - 14), label, fill=(80, 80, 80), font=font)
    title = "Stream heating annulus: residual vs heating efficiency" if use_heating_axis else "Stream mass annulus: residual vs deposited mass fraction"
    draw.text((90, 25), title, fill=(20, 20, 20), font=font)
    draw.text((x0 + 4, y0 + 4), f"1e{int(y_max)}", fill=(80, 80, 80), font=font)
    draw.text((x0 + 4, y1 - 18), f"1e{int(y_min)}", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def run_branch(
    *,
    label: str,
    mass_fractions: list[float],
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    fiducial: FiducialParams,
    mdot_edd: float,
    rows: list[dict[str, Any]],
) -> None:
    current_z = np.asarray(anchor_z, dtype=float)
    current_params = anchor_params
    current_fraction = float(anchor_params.stream_source_fraction)
    prev_z: np.ndarray | None = None
    prev_fraction: float | None = None
    for mass_fraction in mass_fractions:
        params = params_for(
            fiducial,
            mdot_edd,
            ratio=current_params.mdot_edd_ratio,
            R_out_rg=current_params.R_out_rg,
            n_nodes=current_params.n_nodes,
            grid_power=current_params.grid_power,
            custom_grid_xi=current_params.custom_grid_xi,
            mass_fraction=float(mass_fraction),
            source_center_fraction=current_params.stream_source_center_fraction,
            source_log_width=current_params.stream_source_log_width,
            source_shape=current_params.stream_source_shape,
            source_shape_blend=current_params.stream_source_shape_blend,
            torque_fraction=current_params.stream_torque_delta_l_fraction,
            torque_center_fraction=current_params.stream_torque_center_fraction,
            torque_log_width=current_params.stream_torque_log_width,
            wind_sink_fraction=current_params.wind_sink_fraction,
            wind_sink_center_fraction=current_params.wind_sink_center_fraction,
            wind_sink_log_width=current_params.wind_sink_log_width,
            stream_heating_efficiency=current_params.stream_heating_efficiency,
            outer_closure=current_params.outer_closure,
            outer_robin_chi=current_params.outer_robin_chi,
            outer_robin_slope_target=current_params.outer_robin_slope_target,
            outer_robin_slope_scale=current_params.outer_robin_slope_scale,
            outer_buffer_inner_rg=current_params.outer_buffer_inner_rg,
            outer_buffer_radial_weight=current_params.outer_buffer_radial_weight,
            outer_buffer_energy_weight=current_params.outer_buffer_energy_weight,
            outer_buffer_boundary_weight=current_params.outer_buffer_boundary_weight,
            outer_buffer_taper_log_width=current_params.outer_buffer_taper_log_width,
            interval_residual_form=current_params.interval_residual_form,
            integrated_residual_weighting=current_params.integrated_residual_weighting,
        )
        params = apply_outer_slopes_from_state(current_z, params)
        seed, predictor, initial_full, predictor_meta = source_fraction_seed(
            target_fraction=float(mass_fraction),
            current_fraction=current_fraction,
            current_z=current_z,
            prev_fraction=prev_fraction,
            prev_z=prev_z,
            params=params,
        )
        print(
            f"{label} mass_fraction={mass_fraction:g} predictor={predictor} "
            f"initial={initial_full:.3e}",
            flush=True,
        )
        seed, polish, final_params, elapsed, polish_meta = polish_with_optional_residual_remesh(
            seed=seed,
            params=params,
            remesh_after_accept=RESIDUAL_REMESH_EVERY_STEP,
            remesh_on_reject=RESIDUAL_REMESH_ON_REJECT,
        )
        row = row_for_result(
            branch=label,
            mass_fraction=float(mass_fraction),
            seed=seed,
            z=polish.z,
            params=final_params,
            polish=polish,
            elapsed_s=elapsed,
            extra={
                **predictor_meta,
                **polish_meta,
                "anchor_source_fraction": float(anchor_params.stream_source_fraction),
                "anchor_Rout_rg": float(anchor_params.R_out_rg),
            },
            lean_diagnostics=bool(LEAN_REJECT_DIAGNOSTICS and gate_would_reject(polish.z, final_params)),
        )
        row["predictor"] = predictor
        row["predictor_initial_full"] = float(initial_full)
        apply_physical_gate(row)
        row["newton_audit_path"] = "" if row.get("lean_diagnostics", False) else write_newton_audit(row, polish)
        rows.append(row)
        save_checkpoint(row, final_params)
        write_table(rows)
        write_figure(rows)
        print(
            f"  final={row['final_full']:.3e} dom={row['dominant']} "
            f"Mdot_outer/inner={row['Mdot_outer_over_inner']:.5g} accepted={row['accepted']} "
            f"anchor={row['anchor_eligible']} physE={row.get('partition_physical_E', np.nan):.3e} "
            f"phys_ok={row.get('physical_E_gate_eligible', True)} remesh={row.get('residual_remesh_action', 'none')}:"
            f"{row.get('residual_remesh_adopted', False)} patch={row.get('local_patch_adopted', False)} "
            f"picard={row.get('outer_picard_iterations', 0)}",
            flush=True,
        )
        if row["accepted"]:
            prev_z = np.asarray(current_z, dtype=float)
            prev_fraction = current_fraction
            current_z = np.asarray(polish.z, dtype=float)
            current_params = final_params
            current_fraction = float(mass_fraction)
        else:
            print(f"  stopping branch {label} at first non-accepted mass fraction", flush=True)
            break


def run_adaptive_branch(
    *,
    label: str,
    target_fraction: float,
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    fiducial: FiducialParams,
    mdot_edd: float,
    rows: list[dict[str, Any]],
) -> None:
    current_z = np.asarray(anchor_z, dtype=float)
    current_params = anchor_params
    current_fraction = float(anchor_params.stream_source_fraction)
    prev_z: np.ndarray | None = None
    prev_fraction: float | None = None
    step = min(abs(float(ADAPTIVE_INITIAL_STEP)), abs(float(target_fraction) - current_fraction))
    direction = 1.0 if float(target_fraction) >= current_fraction else -1.0
    attempt = 0

    if step <= 0.0:
        return

    while direction * (float(target_fraction) - current_fraction) > 1.0e-12:
        remaining = direction * (float(target_fraction) - current_fraction)
        trial_step = min(step, remaining)
        mass_fraction = current_fraction + direction * trial_step
        params = params_for(
            fiducial,
            mdot_edd,
            ratio=current_params.mdot_edd_ratio,
            R_out_rg=current_params.R_out_rg,
            n_nodes=current_params.n_nodes,
            grid_power=current_params.grid_power,
            custom_grid_xi=current_params.custom_grid_xi,
            mass_fraction=float(mass_fraction),
            source_center_fraction=current_params.stream_source_center_fraction,
            source_log_width=current_params.stream_source_log_width,
            source_shape=current_params.stream_source_shape,
            source_shape_blend=current_params.stream_source_shape_blend,
            torque_fraction=current_params.stream_torque_delta_l_fraction,
            torque_center_fraction=current_params.stream_torque_center_fraction,
            torque_log_width=current_params.stream_torque_log_width,
            wind_sink_fraction=current_params.wind_sink_fraction,
            wind_sink_center_fraction=current_params.wind_sink_center_fraction,
            wind_sink_log_width=current_params.wind_sink_log_width,
            stream_heating_efficiency=current_params.stream_heating_efficiency,
            outer_closure=current_params.outer_closure,
            outer_robin_chi=current_params.outer_robin_chi,
            outer_robin_slope_target=current_params.outer_robin_slope_target,
            outer_robin_slope_scale=current_params.outer_robin_slope_scale,
            outer_buffer_inner_rg=current_params.outer_buffer_inner_rg,
            outer_buffer_radial_weight=current_params.outer_buffer_radial_weight,
            outer_buffer_energy_weight=current_params.outer_buffer_energy_weight,
            outer_buffer_boundary_weight=current_params.outer_buffer_boundary_weight,
            outer_buffer_taper_log_width=current_params.outer_buffer_taper_log_width,
            interval_residual_form=current_params.interval_residual_form,
            integrated_residual_weighting=current_params.integrated_residual_weighting,
        )
        params = apply_outer_slopes_from_state(current_z, params)
        seed, predictor, initial_full, predictor_meta = source_fraction_seed(
            target_fraction=float(mass_fraction),
            current_fraction=current_fraction,
            current_z=current_z,
            prev_fraction=prev_fraction,
            prev_z=prev_z,
            params=params,
        )
        attempt += 1
        print(
            f"{label} attempt={attempt} current={current_fraction:.6g} target={mass_fraction:.6g} "
            f"step={direction * trial_step:.6g} predictor={predictor} initial={initial_full:.3e}",
            flush=True,
        )
        if initial_full > ADAPTIVE_MAX_INITIAL_FULL and trial_step > ADAPTIVE_MIN_STEP * (1.0 + 1.0e-12):
            step = max(ADAPTIVE_MIN_STEP, trial_step * ADAPTIVE_SHRINK)
            print(f"  pre-reject initial residual; reducing step to {step:.6g}", flush=True)
            continue

        seed, polish, final_params, elapsed, polish_meta = polish_with_optional_residual_remesh(
            seed=seed,
            params=params,
            remesh_after_accept=RESIDUAL_REMESH_EVERY_STEP,
            remesh_on_reject=RESIDUAL_REMESH_ON_REJECT,
        )
        row = row_for_result(
            branch=label,
            mass_fraction=float(mass_fraction),
            seed=seed,
            z=polish.z,
            params=final_params,
            polish=polish,
            elapsed_s=elapsed,
            extra={
                **predictor_meta,
                **polish_meta,
                "anchor_source_fraction": float(anchor_params.stream_source_fraction),
                "anchor_Rout_rg": float(anchor_params.R_out_rg),
            },
            lean_diagnostics=bool(LEAN_REJECT_DIAGNOSTICS and gate_would_reject(polish.z, final_params)),
        )
        row["predictor"] = predictor
        row["predictor_initial_full"] = float(initial_full)
        row["attempt_step"] = float(direction * trial_step)
        row["cost_action"] = "pending"
        apply_physical_gate(row)
        row["newton_audit_path"] = "" if row.get("lean_diagnostics", False) else write_newton_audit(row, polish)
        should_break = False
        if row["accepted"]:
            prev_z = np.asarray(current_z, dtype=float)
            prev_fraction = current_fraction
            current_z = np.asarray(polish.z, dtype=float)
            current_params = final_params
            current_fraction = float(mass_fraction)
            effective_nfev = int(row.get("polish_nfev_total", row["nfev"]))
            if effective_nfev >= ADAPTIVE_COST_HARD_SHRINK_NFEV:
                step = max(ADAPTIVE_MIN_STEP, trial_step * ADAPTIVE_COST_HARD_SHRINK)
                row["cost_action"] = f"hard_shrink_nfev>={ADAPTIVE_COST_HARD_SHRINK_NFEV}"
            elif effective_nfev >= ADAPTIVE_COST_SHRINK_NFEV:
                step = max(ADAPTIVE_MIN_STEP, trial_step * ADAPTIVE_COST_SHRINK)
                row["cost_action"] = f"shrink_nfev>={ADAPTIVE_COST_SHRINK_NFEV}"
            elif row["anchor_eligible"] and initial_full < 0.5 * ADAPTIVE_MAX_INITIAL_FULL and effective_nfev <= ADAPTIVE_COST_GROW_NFEV:
                step = min(ADAPTIVE_MAX_STEP, max(ADAPTIVE_MIN_STEP, trial_step * ADAPTIVE_GROWTH))
                row["cost_action"] = f"grow_nfev<={ADAPTIVE_COST_GROW_NFEV}"
            else:
                step = max(ADAPTIVE_MIN_STEP, trial_step)
                row["cost_action"] = "hold"
        else:
            if trial_step <= ADAPTIVE_MIN_STEP * (1.0 + 1.0e-12):
                print(f"  stopping adaptive branch {label}: minimum step failed", flush=True)
                step = max(ADAPTIVE_MIN_STEP, trial_step)
                row["cost_action"] = "stop_min_step_failed"
                should_break = True
            else:
                step = max(ADAPTIVE_MIN_STEP, trial_step * ADAPTIVE_SHRINK)
                row["cost_action"] = "reject_shrink"
                print(f"  rejected; reducing step to {step:.6g}", flush=True)
        row["next_step"] = float(direction * step)
        rows.append(row)
        save_checkpoint(row, final_params)
        write_table(rows)
        write_figure(rows)
        print(
            f"  final={row['final_full']:.3e} dom={row['dominant']} "
            f"Mdot_outer/inner={row['Mdot_outer_over_inner']:.5g} accepted={row['accepted']} "
            f"anchor={row['anchor_eligible']} physE={row.get('partition_physical_E', np.nan):.3e} "
            f"phys_ok={row.get('physical_E_gate_eligible', True)} nfev={row['nfev']} "
            f"nfev_total={int(row.get('polish_nfev_total', row['nfev']))} next_step={direction * step:.6g} "
            f"action={row['cost_action']} patch={row.get('local_patch_adopted', False)}",
            flush=True,
        )
        if should_break:
            break


def run_heating_branch(
    *,
    label: str,
    heating_efficiencies: list[float],
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    fiducial: FiducialParams,
    mdot_edd: float,
    rows: list[dict[str, Any]],
) -> None:
    current_z = np.asarray(anchor_z, dtype=float)
    current_params = anchor_params
    current_eta = float(anchor_params.stream_heating_efficiency)
    for eta in heating_efficiencies:
        params = params_for(
            fiducial,
            mdot_edd,
            ratio=current_params.mdot_edd_ratio,
            R_out_rg=current_params.R_out_rg,
            n_nodes=current_params.n_nodes,
            grid_power=current_params.grid_power,
            custom_grid_xi=current_params.custom_grid_xi,
            mass_fraction=current_params.stream_source_fraction,
            source_center_fraction=current_params.stream_source_center_fraction,
            source_log_width=current_params.stream_source_log_width,
            source_shape=current_params.stream_source_shape,
            source_shape_blend=current_params.stream_source_shape_blend,
            torque_fraction=current_params.stream_torque_delta_l_fraction,
            torque_center_fraction=current_params.stream_torque_center_fraction,
            torque_log_width=current_params.stream_torque_log_width,
            wind_sink_fraction=current_params.wind_sink_fraction,
            wind_sink_center_fraction=current_params.wind_sink_center_fraction,
            wind_sink_log_width=current_params.wind_sink_log_width,
            stream_heating_efficiency=float(eta),
            outer_closure=current_params.outer_closure,
            outer_robin_chi=current_params.outer_robin_chi,
            outer_robin_slope_target=current_params.outer_robin_slope_target,
            outer_robin_slope_scale=current_params.outer_robin_slope_scale,
            outer_buffer_inner_rg=current_params.outer_buffer_inner_rg,
            outer_buffer_radial_weight=current_params.outer_buffer_radial_weight,
            outer_buffer_energy_weight=current_params.outer_buffer_energy_weight,
            outer_buffer_boundary_weight=current_params.outer_buffer_boundary_weight,
            outer_buffer_taper_log_width=current_params.outer_buffer_taper_log_width,
            interval_residual_form=current_params.interval_residual_form,
            integrated_residual_weighting=current_params.integrated_residual_weighting,
        )
        params = apply_outer_slopes_from_state(current_z, params)
        seed = np.asarray(current_z, dtype=float)
        initial_full = max_residual(seed, params)
        print(
            f"{label} heating_eta={eta:g} current_eta={current_eta:g} initial={initial_full:.3e}",
            flush=True,
        )
        seed, polish, final_params, elapsed, polish_meta = polish_with_optional_residual_remesh(
            seed=seed,
            params=params,
            remesh_after_accept=RESIDUAL_REMESH_EVERY_STEP,
            remesh_on_reject=RESIDUAL_REMESH_ON_REJECT,
        )
        row = row_for_result(
            branch=label,
            mass_fraction=float(params.stream_source_fraction),
            seed=seed,
            z=polish.z,
            params=final_params,
            polish=polish,
            elapsed_s=elapsed,
            extra={
                **polish_meta,
                "predictor_initial_full_current": float(initial_full),
                "predictor_initial_full_secant_best": np.nan,
                "predictor_initial_full_tangent_best": np.nan,
                "anchor_source_fraction": float(anchor_params.stream_source_fraction),
                "anchor_heating_efficiency": float(anchor_params.stream_heating_efficiency),
                "anchor_Rout_rg": float(anchor_params.R_out_rg),
                "heating_step": float(eta - current_eta),
            },
            lean_diagnostics=bool(LEAN_REJECT_DIAGNOSTICS and gate_would_reject(polish.z, final_params)),
        )
        row["predictor"] = "current"
        row["predictor_initial_full"] = float(initial_full)
        apply_physical_gate(row)
        row["newton_audit_path"] = "" if row.get("lean_diagnostics", False) else write_newton_audit(row, polish)
        rows.append(row)
        save_checkpoint(row, final_params)
        write_table(rows)
        write_figure(rows)
        print(
            f"  final={row['final_full']:.3e} dom={row['dominant']} heat={row['stream_heating_efficiency']:.5g} "
            f"maxQsQv={row['max_Qstream_Qvisc']:.3e} intQsQv={row['integrated_Qstream_Qvisc']:.3e} "
            f"accepted={row['accepted']} anchor={row['anchor_eligible']} physE={row.get('partition_physical_E', np.nan):.3e} "
            f"phys_ok={row.get('physical_E_gate_eligible', True)} remesh={row.get('residual_remesh_action', 'none')}:"
            f"{row.get('residual_remesh_adopted', False)}",
            flush=True,
        )
        if row["accepted"]:
            current_z = np.asarray(polish.z, dtype=float)
            current_params = final_params
            current_eta = float(eta)
        else:
            print(f"  stopping heating branch {label} at first non-accepted eta", flush=True)
            break


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = load_anchor(ANCHOR_CHECKPOINT, fiducial, mdot_edd)
    anchor_z, anchor_params = prepare_anchor_grid(anchor_z, anchor_params, fiducial, mdot_edd)
    rows: list[dict[str, Any]] = []
    heating_efficiencies = parse_heating_efficiencies()
    if heating_efficiencies:
        run_heating_branch(
            label=HEATING_LABEL,
            heating_efficiencies=heating_efficiencies,
            anchor_z=anchor_z,
            anchor_params=anchor_params,
            fiducial=fiducial,
            mdot_edd=mdot_edd,
            rows=rows,
        )
    elif ADAPTIVE_TARGET_RAW:
        branches = parse_branch_specs()
        label = branches[0][0] if branches else "adaptive"
        run_adaptive_branch(
            label=label,
            target_fraction=float(ADAPTIVE_TARGET_RAW),
            anchor_z=anchor_z,
            anchor_params=anchor_params,
            fiducial=fiducial,
            mdot_edd=mdot_edd,
            rows=rows,
        )
    else:
        for label, fractions in parse_branch_specs():
            run_branch(
                label=label,
                mass_fractions=fractions,
                anchor_z=anchor_z,
                anchor_params=anchor_params,
                fiducial=fiducial,
                mdot_edd=mdot_edd,
                rows=rows,
            )
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
