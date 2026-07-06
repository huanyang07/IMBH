"""Staged launch-energy continuation for the local-Mdot Mdot=5 wind BVP."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mdot5_local_mdot_bvp_pilot as pilot  # noqa: E402
import run_standard_slim_stream_mass_annulus_scan as scan  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_state,
    differential_residual,
    differential_residual_scales,
    entropy_gradient_log,
    scaled_differential_matrix,
    state_partials,
    stream_annulus_shape_and_derivative,
    stream_heating_rate,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    transonic_profile_from_state_vector,
    wind_energy_loss_rate,
    wind_energy_per_mass,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (  # noqa: E402
    _interval_geometry,
    _outer_buffer_interval_weights,
)
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402


DEFAULT_ANCHOR = (
    ROOT
    / "outputs/checkpoints/m5_energy_wind_powerlaw_mass_coupled_adaptive_0p015_to_0p03/"
    "zeta_0p03_N896.npz"
)
ANCHOR = Path(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_ANCHOR", str(DEFAULT_ANCHOR))).expanduser()
if not ANCHOR.is_absolute():
    ANCHOR = ROOT / ANCHOR
START_X_CHECKPOINT_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_START_X_CHECKPOINT", "").strip()
START_X_CHECKPOINT = Path(START_X_CHECKPOINT_RAW).expanduser() if START_X_CHECKPOINT_RAW else None
if START_X_CHECKPOINT is not None and not START_X_CHECKPOINT.is_absolute():
    START_X_CHECKPOINT = ROOT / START_X_CHECKPOINT

OUTPUT_STEM = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTPUT_STEM", "m5_local_mdot_eta_continuation_zeta0p03_N96")
N_NODES = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_N_NODES", "96"))
ETA_VALUES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_VALUES", "100,60,40,33.3333333333").split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_MAX_NFEV", "220"))
RESIDUAL_TOL = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_TOL", "1e-7"))
MASS_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_MASS_WEIGHT", "1.0"))
INNER_MDOT_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_MDOT_WEIGHT", "1.0"))
REMAP_METHOD = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMAP_METHOD", "linear").strip().lower()
USE_LOCAL_JACOBIAN = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_USE_LOCAL_JACOBIAN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LOCAL_JACOBIAN_STEP = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_LOCAL_JACOBIAN_STEP", "1e-6"))
SOURCE_BAND_EXTRA_ROWS = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_EXTRA_ROWS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SOURCE_BAND_EXTRA_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_EXTRA_WEIGHT", "1.0"))
SOURCE_BAND_MIN_RG_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MIN_RG", "").strip()
SOURCE_BAND_MAX_RG_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MAX_RG", "").strip()
SOURCE_BAND_TAPER_LOG_WIDTH = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_TAPER_LOG_WIDTH", "0.0"))
SOURCE_MICRO_DOMAIN = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_DOMAIN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SOURCE_MICRO_NODES = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_NODES", "32"))
SOURCE_MICRO_LOCAL_CORRECT = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_LOCAL_CORRECT", "1"
).strip().lower() in {"1", "true", "yes", "on"}
SOURCE_MICRO_FREEZE_EDGES = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_FREEZE_EDGES", "1"
).strip().lower() in {"1", "true", "yes", "on"}
SOURCE_MICRO_EDGE_ANCHOR_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_EDGE_ANCHOR_WEIGHT", "0.1"))
SOURCE_MICRO_ALL_ANCHOR_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_ALL_ANCHOR_WEIGHT", "0.0"))
SOURCE_MICRO_INCLUDE_GLOBALS = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_INCLUDE_GLOBALS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
SOURCE_MICRO_STATE_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_STATE_WEIGHT", "1.0"))
SOURCE_MICRO_HERMITE_OVERSHOOT_LIMIT = float(
    os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_HERMITE_OVERSHOOT_LIMIT", "0.5")
)
SOURCE_MICRO_MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_MAX_NFEV", "120"))
SOURCE_DOMAIN_CORRECT = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_DOMAIN_CORRECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SOURCE_DOMAIN_FRACTIONS_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_DOMAIN_FRACTIONS", "0.25,0.5,0.75")
SOURCE_DOMAIN_MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_DOMAIN_MAX_NFEV", "120"))
SOURCE_DOMAIN_FREEZE_EDGES = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_DOMAIN_FREEZE_EDGES", "1"
).strip().lower() in {"1", "true", "yes", "on"}
SOURCE_DOMAIN_EDGE_ANCHOR_WEIGHT = float(
    os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_DOMAIN_EDGE_ANCHOR_WEIGHT", "0.1")
)
SOURCE_DOMAIN_ALL_ANCHOR_WEIGHT = float(
    os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_DOMAIN_ALL_ANCHOR_WEIGHT", "0.0")
)
SOURCE_DOMAIN_INCLUDE_GLOBALS = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_DOMAIN_INCLUDE_GLOBALS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
SOURCE_DOMAIN_LINE_SEARCH_STEPS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_DOMAIN_LINE_SEARCH_STEPS", "12"))
SOURCE_DOMAIN_HALO_INTERVALS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_DOMAIN_HALO_INTERVALS", "0"))
SOURCE_BUFFER_CORRECT = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_CORRECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SOURCE_BUFFER_FRACTIONS_RAW = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_FRACTIONS", SOURCE_DOMAIN_FRACTIONS_RAW
)
SOURCE_BUFFER_HALO_INTERVALS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_HALO_INTERVALS", "4"))
SOURCE_BUFFER_MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_MAX_NFEV", "160"))
SOURCE_BUFFER_FREEZE_EDGES = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_FREEZE_EDGES", "1"
).strip().lower() in {"1", "true", "yes", "on"}
SOURCE_BUFFER_EDGE_ANCHOR_WEIGHT = float(
    os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_EDGE_ANCHOR_WEIGHT", "0.1")
)
SOURCE_BUFFER_ALL_ANCHOR_WEIGHT = float(
    os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_ALL_ANCHOR_WEIGHT", "0.0")
)
SOURCE_BUFFER_INCLUDE_GLOBALS = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_INCLUDE_GLOBALS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
SOURCE_BUFFER_LINE_SEARCH_STEPS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_LINE_SEARCH_STEPS", "12"))
SOURCE_BUFFER_STATE_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_STATE_WEIGHT", "1.0"))
SOURCE_BUFFER_INTEGRAL_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_INTEGRAL_WEIGHT", "1.0"))
SOURCE_BUFFER_JUMP_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_JUMP_WEIGHT", "1.0"))
SOURCE_BUFFER_MASS_QUADRATURE = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_MASS_QUADRATURE", "midpoint"
).strip().lower()
SOURCE_ELEMENT_REFINE = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_REFINE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SOURCE_ELEMENT_SUBDIVISIONS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_SUBDIVISIONS", "2"))
SOURCE_ELEMENT_HALO_INTERVALS = int(
    os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_HALO_INTERVALS", str(SOURCE_BUFFER_HALO_INTERVALS))
)
SOURCE_ELEMENT_REMAP_METHOD = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_REMAP_METHOD", "linear").strip().lower()
SOURCE_ELEMENT_MASS_SEED = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_MASS_SEED", "none").strip().lower()
SOURCE_ELEMENT_MASS_SEED_SWEEPS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_MASS_SEED_SWEEPS", "2"))
SOURCE_BAND_FINITE_VOLUME_MASS_RAW = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_FINITE_VOLUME_MASS", ""
).strip().lower()
SOURCE_BAND_FINITE_VOLUME_MASS = (
    SOURCE_BAND_FINITE_VOLUME_MASS_RAW in {"1", "true", "yes", "on"}
    if SOURCE_BAND_FINITE_VOLUME_MASS_RAW
    else SOURCE_MICRO_DOMAIN
)
SOURCE_BAND_EXTRA_AUDIT_ONLY_RAW = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_EXTRA_AUDIT_ONLY", ""
).strip().lower()
SOURCE_BAND_EXTRA_AUDIT_ONLY = (
    SOURCE_BAND_EXTRA_AUDIT_ONLY_RAW in {"1", "true", "yes", "on"}
    if SOURCE_BAND_EXTRA_AUDIT_ONLY_RAW
    else SOURCE_MICRO_DOMAIN
)
ACCEPT_TOL = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_ACCEPT_TOL", "1e-5"))
OUTER_BUFFER_WEIGHT_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_BUFFER_WEIGHT", "").strip()
OUTER_BUFFER_INNER_RG_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_BUFFER_INNER_RG", "").strip()
OUTER_CLOSURE_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_CLOSURE", "").strip()
OUTER_ROBIN_CHI_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_ROBIN_CHI", "").strip()
OUTER_ROBIN_SLOPE_TARGET_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_ROBIN_SLOPE_TARGET", "").strip()
OUTER_ROBIN_SLOPE_SCALE_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_ROBIN_SLOPE_SCALE", "").strip()
OUTER_OMEGA_LOG_OFFSET_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_OMEGA_LOG_OFFSET", "").strip()
INTERVAL_RESIDUAL_FORM_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INTERVAL_FORM", "").strip()
INTEGRATED_WEIGHTING_RAW = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INTEGRATED_WEIGHTING", "").strip()
OUTER_SLOPE_PICARD_ITERS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_SLOPE_PICARD_ITERS", "0"))
SEED_ONLY = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_SEED_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
DEFECT_REMAP_SWEEPS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_DEFECT_REMAP_SWEEPS", "4"))
STATE_DEFECT_REMAP_SWEEPS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_STATE_DEFECT_REMAP_SWEEPS", "5"))
STATE_DEFECT_REMAP_DAMPING = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_STATE_DEFECT_REMAP_DAMPING", "0.2"))
STATE_DEFECT_REMAP_MATCH_OUTER = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_STATE_DEFECT_REMAP_MATCH_OUTER", "1"
).strip().lower() in {"1", "true", "yes", "on"}
STATE_DEFECT_REMAP_MAX_DY = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_STATE_DEFECT_REMAP_MAX_DY", "0.05"))
INNER_RELAX_OUTER_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_OUTER_RG", "0.0"))
INNER_RELAX_MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_MAX_NFEV", "80"))
INNER_RELAX_INCLUDE_MDOT = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_INCLUDE_MDOT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
INNER_RELAX_INCLUDE_GLOBALS = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_INCLUDE_GLOBALS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
INNER_RELAX_ANCHOR_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_ANCHOR_WEIGHT", "1e-4"))
OUTER_RELAX_MIN_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_MIN_RG", "0.0"))
OUTER_RELAX_MAX_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_MAX_RG", "0.0"))
OUTER_RELAX_MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_MAX_NFEV", "80"))
OUTER_RELAX_INCLUDE_ENERGY = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_INCLUDE_ENERGY", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OUTER_RELAX_INCLUDE_MDOT = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_INCLUDE_MDOT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OUTER_RELAX_INCLUDE_GLOBALS = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_INCLUDE_GLOBALS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OUTER_RELAX_ANCHOR_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_ANCHOR_WEIGHT", "1.0"))
BLOCK_CORRECT = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_CORRECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BLOCK_HALF_WIDTH = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_HALF_WIDTH", "3"))
BLOCK_MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_MAX_NFEV", "80"))
BLOCK_EDGE_ANCHOR_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_EDGE_ANCHOR_WEIGHT", "1e-2"))
BLOCK_ALL_ANCHOR_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_ALL_ANCHOR_WEIGHT", "0.0"))
BLOCK_INCLUDE_OUTER = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_INCLUDE_OUTER", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BLOCK_INCLUDE_GLOBALS = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_INCLUDE_GLOBALS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BLOCK_PEAK_KIND = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_PEAK_KIND", "radial").strip().lower()
BLOCK_LINE_SEARCH_STEPS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_LINE_SEARCH_STEPS", "12"))
BLOCK_ACCEPT_STRICT_GUARDS = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_ACCEPT_STRICT_GUARDS", "1"
).strip().lower() in {"1", "true", "yes", "on"}
BLOCK_FAST_LOCAL_RESIDUAL = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_FAST_LOCAL_RESIDUAL", "1"
).strip().lower() in {"1", "true", "yes", "on"}
BAND_CORRECT = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_CORRECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BAND_MIN_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_MIN_RG", "0.0"))
BAND_MAX_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_MAX_RG", "0.0"))
BAND_MAX_NFEV = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_MAX_NFEV", "100"))
BAND_EDGE_ANCHOR_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_EDGE_ANCHOR_WEIGHT", "0.1"))
BAND_ALL_ANCHOR_WEIGHT = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_ALL_ANCHOR_WEIGHT", "0.0"))
BAND_INCLUDE_GLOBALS = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_INCLUDE_GLOBALS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BAND_LINE_SEARCH_STEPS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_LINE_SEARCH_STEPS", "12"))
BAND_ACCEPT_STRICT_GUARDS = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_ACCEPT_STRICT_GUARDS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
NESTED_REFINE_MIN_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_NESTED_REFINE_MIN_RG", "0.0"))
NESTED_REFINE_MAX_RG = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_NESTED_REFINE_MAX_RG", "inf"))
GRID_HOMOTOPY_STEPS = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_GRID_HOMOTOPY_STEPS", "0"))
GRID_HOMOTOPY_COLLAPSE_FRACTION = float(
    os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_GRID_HOMOTOPY_COLLAPSE_FRACTION", "0.08")
)
GRID_HOMOTOPY_BLOCK_CORRECT = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_GRID_HOMOTOPY_BLOCK_CORRECT", "0"
).strip().lower() in {"1", "true", "yes", "on"}
RADIAL_AUDIT_FORMS = tuple(
    piece.strip().lower()
    for piece in os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RADIAL_AUDIT_FORMS", "").split(",")
    if piece.strip()
)
RADIAL_AUDIT_TOP_N = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RADIAL_AUDIT_TOP_N", "20"))
JACOBIAN_AUDIT = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_JACOBIAN_AUDIT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
JACOBIAN_AUDIT_HALF_WIDTH = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_JACOBIAN_AUDIT_HALF_WIDTH", "3"))
JACOBIAN_AUDIT_INCLUDE_GLOBALS = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_JACOBIAN_AUDIT_INCLUDE_GLOBALS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
TRANSITION_GRID_AUDIT = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_GRID_AUDIT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TRANSITION_ALIGN_NODES = os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_ALIGN_NODES", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TRANSITION_ALIGN_INCLUDE_SOURCE = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_ALIGN_INCLUDE_SOURCE", "1"
).strip().lower() in {"1", "true", "yes", "on"}
TRANSITION_ALIGN_INCLUDE_BUFFER = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_ALIGN_INCLUDE_BUFFER", "1"
).strip().lower() in {"1", "true", "yes", "on"}
TRANSITION_ALIGN_INCLUDE_PEAK = os.environ.get(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_ALIGN_INCLUDE_PEAK", "1"
).strip().lower() in {"1", "true", "yes", "on"}
TRANSITION_SIDE_FRACTION = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_SIDE_FRACTION", "0.02"))
RESIDUAL_REMESH_STRENGTH = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_STRENGTH", "0.0"))
RESIDUAL_REMESH_BLEND = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_BLEND", "0.7"))
RESIDUAL_REMESH_POWER = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_POWER", "0.5"))
RESIDUAL_REMESH_SMOOTH_PASSES = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_SMOOTH_PASSES", "2"))
RESIDUAL_REMESH_FLOOR = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_FLOOR", "1.0"))
RESIDUAL_REMESH_DENSE_FACTOR = int(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_REMESH_DENSE_FACTOR", "32"))
W_REMESH_INTERVAL_R = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_INTERVAL_R", "1.0"))
W_REMESH_INTERVAL_E = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_INTERVAL_E", "1.0"))
W_REMESH_MASS = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_MASS", "0.5"))
W_REMESH_SOURCE = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_SOURCE", "0.8"))
W_REMESH_WIND = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_WIND", "0.8"))
W_REMESH_MDOT_GRAD = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_MDOT_GRAD", "0.8"))
W_REMESH_OUTER = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_W_OUTER", "1.0"))
REMESH_OUTER_WIDTH = float(os.environ.get("IMBH_MDOT5_LOCAL_MDOT_ETA_REMESH_OUTER_WIDTH", "0.04"))

JSON_OUTPUT = ROOT / f"outputs/tables/{OUTPUT_STEM}.json"
MD_OUTPUT = ROOT / f"outputs/tables/{OUTPUT_STEM}.md"
PROFILE_OUTPUT = ROOT / f"outputs/tables/{OUTPUT_STEM}_profiles.json"
CHECKPOINT_DIR = ROOT / f"outputs/checkpoints/{OUTPUT_STEM}"

PCHIP_REMAP_METHODS = {
    "pchip",
    "monotone",
    "shape_preserving",
    "mass_ode",
    "defect_preserving",
    "pchip_mass_ode",
    "nested_mass_ode",
    "nested_defect_preserving",
    "state_defect_preserving",
    "defect_preserving_state",
    "state_mass_defect",
    "nested_state_defect_preserving",
    "nested_state_mass_defect",
}
MASS_DEFECT_REMAP_METHODS = {
    "defect_preserving",
    "mass_defect",
    "defect_preserving_mass",
    "nested_defect_preserving",
    "state_defect_preserving",
    "defect_preserving_state",
    "state_mass_defect",
    "nested_state_defect_preserving",
    "nested_state_mass_defect",
}
STATE_DEFECT_REMAP_METHODS = {
    "state_defect_preserving",
    "defect_preserving_state",
    "state_mass_defect",
    "nested_state_defect_preserving",
    "nested_state_mass_defect",
}
NESTED_REMAP_METHODS = {
    "nested_mass_ode",
    "nested_defect_preserving",
    "nested_state_defect_preserving",
    "nested_state_mass_defect",
}


def _format(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "nan"
    if number == 0.0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e4:
        return f"{number:.3e}"
    return f"{number:.6g}"


def _safe_eta_label(value: float) -> str:
    return f"{float(value):.8g}".replace(".", "p").replace("-", "m")


def _cumtrapz(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(values, dtype=float))
    if out.size < 2:
        return out
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(np.asarray(x, dtype=float)))
    return out


def _normalize_component(values: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.abs(np.asarray(values, dtype=float)), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.max(clean)) if clean.size else 0.0
    if scale <= 0.0:
        return np.zeros_like(clean)
    return clean / scale


def _smooth_score(score: np.ndarray, passes: int) -> np.ndarray:
    smoothed = np.asarray(score, dtype=float)
    for _ in range(max(int(passes), 0)):
        if smoothed.size <= 2:
            break
        padded = np.pad(smoothed, (1, 1), mode="edge")
        smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return smoothed


def _enforce_min_spacing(xi: np.ndarray, min_spacing: float = 1.0e-10) -> np.ndarray:
    adjusted = np.asarray(xi, dtype=float).copy()
    adjusted[0] = 0.0
    adjusted[-1] = 1.0
    for idx in range(1, adjusted.size):
        adjusted[idx] = max(adjusted[idx], adjusted[idx - 1] + min_spacing)
    if adjusted[-1] > 1.0:
        adjusted *= 1.0 / adjusted[-1]
    adjusted[-1] = 1.0
    for idx in range(adjusted.size - 2, -1, -1):
        adjusted[idx] = min(adjusted[idx], adjusted[idx + 1] - min_spacing)
    adjusted[0] = 0.0
    adjusted[-1] = 1.0
    if np.any(np.diff(adjusted) <= 0.0):
        raise RuntimeError("residual-remeshed grid spacing collapsed")
    return adjusted


def _set_eta(eta_E: float) -> None:
    pilot.WIND_ENERGY_MULTIPLIER = float(eta_E)
    pilot.MASS_WEIGHT = float(MASS_WEIGHT)


def _inner_mdot_row_index(params) -> int:
    return 2 * (int(params.n_nodes) - 1) + 2 + 2


def _residual(x: np.ndarray, params) -> np.ndarray:
    rows = _production_residual_base(x, params)
    if SOURCE_BAND_EXTRA_ROWS and not SOURCE_BAND_EXTRA_AUDIT_ONLY:
        rows = np.concatenate([rows, _source_band_extra_residual_rows(x, params)])
    return rows


def _radial_terms(logR: float, y: np.ndarray, g: np.ndarray, lambda0: float, params) -> dict[str, float]:
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    partials = state_partials(logR, y, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)
    dPi_dx_explicit = float(partials.x["Pi"])
    dPi_dx_gradient = float(np.dot(partials.y["Pi"], g))
    pressure_explicit = dPi_dx_explicit / state.Sigma
    pressure_gradient = dPi_dx_gradient / state.Sigma
    inertial = state.u**2 * float(g[0])
    gravity_centrifugal = -state.R**2 * (state.Omega**2 - state.Omega_K**2)
    raw = float(inertial + gravity_centrifugal + pressure_explicit + pressure_gradient)
    radial_scale, _energy_scale = differential_residual_scales(logR, y, lambda0, params)
    scale = max(float(radial_scale), 1.0e-300)
    return {
        "inertial": float(inertial),
        "gravity_centrifugal": float(gravity_centrifugal),
        "pressure_explicit": float(pressure_explicit),
        "pressure_gradient": float(pressure_gradient),
        "raw_sum": raw,
        "scale": scale,
        "inertial_scaled": float(inertial / scale),
        "gravity_centrifugal_scaled": float(gravity_centrifugal / scale),
        "pressure_explicit_scaled": float(pressure_explicit / scale),
        "pressure_gradient_scaled": float(pressure_gradient / scale),
        "raw_sum_scaled": float(raw / scale),
    }


def _scaled_residual_at(logR: float, y: np.ndarray, g: np.ndarray, lambda0: float, params) -> np.ndarray:
    raw = differential_residual(logR, y, g, lambda0, params)
    radial_scale, energy_scale = differential_residual_scales(logR, y, lambda0, params)
    return np.asarray(raw, dtype=float) / np.asarray([radial_scale, energy_scale], dtype=float)


def _source_band_default_bounds_rg(params) -> tuple[float, float]:
    if SOURCE_BAND_MIN_RG_RAW and SOURCE_BAND_MAX_RG_RAW:
        return float(SOURCE_BAND_MIN_RG_RAW), float(SOURCE_BAND_MAX_RG_RAW)
    source_fraction = float(getattr(params, "stream_source_fraction", 0.0))
    legacy_fraction = float(getattr(params, "stream_mass_fraction", 0.0))
    if source_fraction == 0.0 and legacy_fraction == 0.0:
        return math.inf, -math.inf
    if source_fraction != 0.0:
        center_fraction = float(getattr(params, "stream_source_center_fraction", 0.8))
        log_width = float(getattr(params, "stream_source_log_width", 0.08))
    else:
        center_fraction = float(getattr(params, "stream_mass_center_fraction", 0.8))
        log_width = float(getattr(params, "stream_mass_log_width", 0.08))
    center_rg = center_fraction * float(params.R_out_rg)
    default_min = center_rg * math.exp(-log_width)
    default_max = center_rg * math.exp(log_width)
    band_min = float(SOURCE_BAND_MIN_RG_RAW) if SOURCE_BAND_MIN_RG_RAW else default_min
    band_max = float(SOURCE_BAND_MAX_RG_RAW) if SOURCE_BAND_MAX_RG_RAW else default_max
    return band_min, band_max


def _source_band_default_bounds_logR(params) -> tuple[float, float]:
    band_min_rg, band_max_rg = _source_band_default_bounds_rg(params)
    if not (np.isfinite(band_min_rg) and np.isfinite(band_max_rg)) or band_max_rg <= band_min_rg:
        return math.inf, -math.inf
    return float(np.log(band_min_rg * params.r_g)), float(np.log(band_max_rg * params.r_g))


def _source_center_logR(params) -> float:
    source_fraction = float(getattr(params, "stream_source_fraction", 0.0))
    legacy_fraction = float(getattr(params, "stream_mass_fraction", 0.0))
    if source_fraction == 0.0 and legacy_fraction == 0.0:
        return math.nan
    if source_fraction != 0.0:
        center_fraction = float(getattr(params, "stream_source_center_fraction", 0.8))
    else:
        center_fraction = float(getattr(params, "stream_mass_center_fraction", 0.8))
    center_rg = center_fraction * float(params.R_out_rg)
    return float(np.log(center_rg * params.r_g)) if center_rg > 0.0 else math.nan


def _interval_overlaps_source_band(logR: np.ndarray, idx: int, params) -> bool:
    band_min, band_max = _source_band_default_bounds_logR(params)
    if not (np.isfinite(band_min) and np.isfinite(band_max)) or band_max <= band_min:
        return False
    left = float(logR[idx])
    right = float(logR[idx + 1])
    return bool(right >= band_min and left <= band_max)


def _stream_source_integral(logR_left: float, logR_right: float, params) -> float:
    """Analytic integral of the compact/tanh stream source over ``dlnR``."""

    source_fraction = float(getattr(params, "stream_source_fraction", 0.0))
    legacy_fraction = float(getattr(params, "stream_mass_fraction", 0.0))
    if source_fraction == 0.0 and legacy_fraction == 0.0:
        return 0.0
    if source_fraction != 0.0:
        fraction = source_fraction
        center_fraction = float(getattr(params, "stream_source_center_fraction", 0.8))
        log_width = float(getattr(params, "stream_source_log_width", 0.08))
        shape = str(getattr(params, "stream_source_shape", "tanh"))
        blend = float(getattr(params, "stream_source_shape_blend", 1.0))
    else:
        fraction = legacy_fraction
        center_fraction = float(getattr(params, "stream_mass_center_fraction", 0.8))
        log_width = float(getattr(params, "stream_mass_log_width", 0.08))
        shape = "tanh"
        blend = 1.0
    shape_left, _ = stream_annulus_shape_and_derivative(
        float(logR_left), center_fraction, log_width, float(params.R_out), shape, blend
    )
    shape_right, _ = stream_annulus_shape_and_derivative(
        float(logR_right), center_fraction, log_width, float(params.R_out), shape, blend
    )
    return float(params.Mdot_g_s * fraction * (shape_right - shape_left))


def _finite_volume_mass_residual_from_unpacked(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    local_params,
    idx: int,
) -> float:
    """Return a dimensionless finite-volume mass residual for one interval."""

    wind_integral, source_integral, mdot_scale, mdot_left, mdot_right = _finite_volume_mass_terms_from_unpacked(
        logu, logT, logMdot, logR, lambda0, local_params, idx
    )
    return float(MASS_WEIGHT * (mdot_right - mdot_left - (wind_integral - source_integral)) / mdot_scale)


def _finite_volume_mass_terms_from_unpacked(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    local_params,
    idx: int,
) -> tuple[float, float, float, float, float]:
    """Return wind/source integrals, scale, and endpoint Mdot values."""

    dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
    F_left = _ode_slope(float(logR[idx]), y_left, lambda0, local_params)
    F_right = _ode_slope(float(logR[idx + 1]), y_right, lambda0, local_params)
    if not (np.all(np.isfinite(F_left)) and np.all(np.isfinite(F_right))):
        F_left = F_right = (y_right - y_left) / dx
    y_mid, _used_hermite_midpoint = _bounded_hermite_midpoint(y_left, y_right, F_left, F_right, dx, local_params)
    F_mid = _ode_slope(xm, y_mid, lambda0, local_params)
    if not np.all(np.isfinite(F_mid)):
        F_mid = (y_right - y_left) / dx
    wind_left = _safe_wind_prime(float(logR[idx]), y_left, F_left, lambda0, local_params)
    wind_mid = _safe_wind_prime(xm, y_mid, F_mid, lambda0, local_params)
    wind_right = _safe_wind_prime(float(logR[idx + 1]), y_right, F_right, lambda0, local_params)
    if not (np.isfinite(wind_left) and np.isfinite(wind_mid) and np.isfinite(wind_right)):
        gm = (y_right - y_left) / dx
        linear_mid = 0.5 * (y_left + y_right)
        wind_mid = _safe_wind_prime(xm, linear_mid, gm, lambda0, local_params)
        if not np.isfinite(wind_mid):
            wind_mid = 0.0
        wind_left = wind_mid if not np.isfinite(wind_left) else wind_left
        wind_right = wind_mid if not np.isfinite(wind_right) else wind_right
    wind_integral = (dx / 6.0) * (wind_left + 4.0 * wind_mid + wind_right)
    source_integral = _stream_source_integral(float(logR[idx]), float(logR[idx + 1]), local_params)
    mdot_left = float(np.exp(logMdot[idx]))
    mdot_right = float(np.exp(logMdot[idx + 1]))
    mdot_scale = max(math.sqrt(max(mdot_left, 1.0e-300) * max(mdot_right, 1.0e-300)), 1.0e-300)
    return float(wind_integral), float(source_integral), float(mdot_scale), mdot_left, mdot_right


def _source_buffer_mass_terms_from_unpacked(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    local_params,
    idx: int,
) -> tuple[float, float, float, float, float]:
    """Mass terms for the reduced source-buffer solve.

    The production residual keeps the more expensive Simpson/end-point ODE
    finite-volume row. The reduced corrector can use midpoint wind quadrature
    to make local Jacobian evaluations affordable while keeping the analytic
    stream-source integral exact.
    """

    if SOURCE_BUFFER_MASS_QUADRATURE in {"simpson", "fv", "finite_volume"}:
        return _finite_volume_mass_terms_from_unpacked(logu, logT, logMdot, logR, lambda0, local_params, idx)
    dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
    ym = 0.5 * (y_left + y_right)
    gm = (y_right - y_left) / dx
    wind_prime = _safe_wind_prime(xm, ym, gm, lambda0, local_params)
    if not np.isfinite(wind_prime):
        wind_prime = 0.0
    wind_integral = float(wind_prime * dx)
    source_integral = _stream_source_integral(float(logR[idx]), float(logR[idx + 1]), local_params)
    mdot_left = float(np.exp(logMdot[idx]))
    mdot_right = float(np.exp(logMdot[idx + 1]))
    mdot_scale = max(math.sqrt(max(mdot_left, 1.0e-300) * max(mdot_right, 1.0e-300)), 1.0e-300)
    return wind_integral, float(source_integral), float(mdot_scale), mdot_left, mdot_right


def _bounded_hermite_midpoint(
    y_left: np.ndarray,
    y_right: np.ndarray,
    F_left: np.ndarray,
    F_right: np.ndarray,
    dx: float,
    params,
) -> tuple[np.ndarray, bool]:
    linear = 0.5 * (y_left + y_right)
    if not (np.all(np.isfinite(F_left)) and np.all(np.isfinite(F_right))):
        return linear, False
    candidate = linear + (float(dx) / 8.0) * (F_left - F_right)
    lower = np.asarray([params.logu_bounds[0], params.logT_bounds[0]], dtype=float)
    upper = np.asarray([params.logu_bounds[1], params.logT_bounds[1]], dtype=float)
    lo = np.minimum(y_left, y_right) - abs(float(SOURCE_MICRO_HERMITE_OVERSHOOT_LIMIT))
    hi = np.maximum(y_left, y_right) + abs(float(SOURCE_MICRO_HERMITE_OVERSHOOT_LIMIT))
    lo = np.maximum(lo, lower)
    hi = np.minimum(hi, upper)
    if not np.all(np.isfinite(candidate)) or np.any(candidate < lo) or np.any(candidate > hi):
        return linear, False
    return candidate, True


def _safe_wind_prime(logR: float, y: np.ndarray, g: np.ndarray, lambda0: float, params) -> float:
    try:
        value = pilot._wind_mass_prime(float(logR), np.asarray(y, dtype=float), np.asarray(g, dtype=float), lambda0, params)
        return float(value) if np.isfinite(value) else math.nan
    except Exception:
        return math.nan


def _production_residual_base(x: np.ndarray, params) -> np.ndarray:
    """Return local-Mdot residual rows with optional finite-volume source-band mass rows."""

    rows = np.asarray(pilot.residual(x, params), dtype=float).copy()
    if INNER_MDOT_WEIGHT != 1.0:
        rows[_inner_mdot_row_index(params)] *= float(INNER_MDOT_WEIGHT)
    if not SOURCE_BAND_FINITE_VOLUME_MASS:
        return rows
    try:
        logu, logT, logMdot, _logR_son, lambda0, logR = pilot._unpack(x, params)
        local_params = pilot._local_params(params, logR, logMdot)
        mass_start = _inner_mdot_row_index(params) + 1
        for idx in range(int(params.n_nodes) - 1):
            if _interval_overlaps_source_band(logR, idx, local_params):
                rows[mass_start + idx] = _finite_volume_mass_residual_from_unpacked(
                    logu, logT, logMdot, logR, lambda0, local_params, idx
                )
    except Exception:
        return np.full(3 * int(params.n_nodes) + 2, 1.0e6, dtype=float)
    return rows


def _source_band_row_weight(R_rg: float, band_min_rg: float, band_max_rg: float) -> float:
    if not (np.isfinite(R_rg) and np.isfinite(band_min_rg) and np.isfinite(band_max_rg)):
        return 0.0
    if band_max_rg <= band_min_rg:
        return 0.0
    if band_min_rg <= R_rg <= band_max_rg:
        return float(SOURCE_BAND_EXTRA_WEIGHT)
    width = float(SOURCE_BAND_TAPER_LOG_WIDTH)
    if width <= 0.0:
        return 0.0
    logR = math.log(max(float(R_rg), 1.0e-300))
    log_min = math.log(float(band_min_rg))
    log_max = math.log(float(band_max_rg))
    if log_min - width <= logR < log_min:
        s = (logR - (log_min - width)) / width
    elif log_max < logR <= log_max + width:
        s = ((log_max + width) - logR) / width
    else:
        return 0.0
    smooth = s * s * (3.0 - 2.0 * s)
    return float(SOURCE_BAND_EXTRA_WEIGHT * smooth)


def _source_band_extra_residual_data(x: np.ndarray, params) -> dict[str, Any]:
    n = int(params.n_nodes)
    rows: list[float] = []
    row_R_rg: list[float] = []
    row_interval: list[int] = []
    row_component: list[str] = []
    row_weight: list[float] = []
    try:
        logu, logT, logMdot, _logR_son, lambda0, logR = pilot._unpack(x, params)
        local_params = pilot._local_params(params, logR, logMdot)
        for idx in range(n - 1):
            for local_row in range(4):
                value, R_rg, weight, component = _source_band_extra_row_from_unpacked(
                    logu, logT, logR, lambda0, local_params, 4 * idx + local_row
                )
                rows.append(value)
                row_R_rg.append(R_rg)
                row_interval.append(idx)
                row_component.append(component)
                row_weight.append(weight)
    except Exception:
        size = 4 * max(n - 1, 0)
        rows = [1.0e6] * size
        row_R_rg = [math.nan] * size
        row_interval = [-1] * size
        row_component = [""] * size
        row_weight = [math.nan] * size
    return {
        "rows": np.asarray(rows, dtype=float),
        "R_rg": np.asarray(row_R_rg, dtype=float),
        "interval": np.asarray(row_interval, dtype=int),
        "component": row_component,
        "weight": np.asarray(row_weight, dtype=float),
    }


def _source_band_extra_row_count(params) -> int:
    return 4 * max(int(params.n_nodes) - 1, 0)


def _source_band_extra_row_from_unpacked(
    logu: np.ndarray,
    logT: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    local_params,
    extra_row: int,
) -> tuple[float, float, float, str]:
    interval_idx = int(extra_row) // 4
    slot = int(extra_row) % 4
    fraction = 0.25 if slot < 2 else 0.75
    component_index = slot % 2
    component = "R" if component_index == 0 else "E"
    dx, y_left, y_right, _xm = _interval_geometry(logu, logT, logR, interval_idx)
    g = (y_right - y_left) / dx
    xq = float(logR[interval_idx] + fraction * dx)
    yq = (1.0 - fraction) * y_left + fraction * y_right
    R_rg = float(np.exp(xq) / local_params.r_g)
    band_min_rg, band_max_rg = _source_band_default_bounds_rg(local_params)
    weight = _source_band_row_weight(R_rg, band_min_rg, band_max_rg)
    residual = weight * _scaled_residual_at(xq, yq, g, lambda0, local_params)
    return float(residual[component_index]), R_rg, weight, component


def _source_band_extra_residual_rows(x: np.ndarray, params) -> np.ndarray:
    return np.asarray(_source_band_extra_residual_data(x, params)["rows"], dtype=float)


def _source_band_extra_profile(x: np.ndarray, params) -> dict[str, Any]:
    data = _source_band_extra_residual_data(x, params)
    rows = np.asarray(data["rows"], dtype=float)
    weights = np.asarray(data["weight"], dtype=float)
    active = weights > 0.0
    if rows.size == 0:
        return {
            "source_band_extra_rows_enabled": bool(SOURCE_BAND_EXTRA_ROWS),
            "source_band_extra_row_count": 0,
            "source_band_extra_active_row_count": 0,
            "source_band_extra_max": math.nan,
            "source_band_extra_radial_max": math.nan,
            "source_band_extra_energy_max": math.nan,
            "source_band_extra_peak_R_rg": math.nan,
        }
    active_rows = rows[active] if np.any(active) else rows
    active_R = np.asarray(data["R_rg"], dtype=float)[active] if np.any(active) else np.asarray(data["R_rg"], dtype=float)
    active_components = np.asarray(data["component"], dtype=object)[active] if np.any(active) else np.asarray(data["component"], dtype=object)
    peak = int(np.argmax(np.abs(active_rows))) if active_rows.size else 0
    radial = active_rows[active_components == "R"]
    energy = active_rows[active_components == "E"]
    return {
        "source_band_extra_rows_enabled": bool(SOURCE_BAND_EXTRA_ROWS),
        "source_band_extra_weight": float(SOURCE_BAND_EXTRA_WEIGHT),
        "source_band_extra_min_rg": float(_source_band_default_bounds_rg(params)[0]),
        "source_band_extra_max_rg": float(_source_band_default_bounds_rg(params)[1]),
        "source_band_extra_taper_log_width": float(SOURCE_BAND_TAPER_LOG_WIDTH),
        "source_band_extra_row_count": int(rows.size),
        "source_band_extra_active_row_count": int(np.count_nonzero(active)),
        "source_band_extra_max": float(np.max(np.abs(active_rows))) if active_rows.size else math.nan,
        "source_band_extra_radial_max": float(np.max(np.abs(radial))) if radial.size else math.nan,
        "source_band_extra_energy_max": float(np.max(np.abs(energy))) if energy.size else math.nan,
        "source_band_extra_peak_R_rg": float(active_R[peak]) if active_R.size else math.nan,
    }


def _ode_slope(logR: float, y: np.ndarray, lambda0: float, params) -> np.ndarray:
    try:
        A, c, _radial_scale, _energy_scale = scaled_differential_matrix(logR, y, lambda0, params)
        return np.linalg.solve(A, -c)
    except Exception:
        return np.full(2, np.nan)


def _split_interval_radial_residual(
    logR_left: float,
    y_left: np.ndarray,
    logR_right: float,
    y_right: np.ndarray,
    y_mid: np.ndarray,
    lambda0: float,
    params,
) -> dict[str, float]:
    logR_mid = 0.5 * (float(logR_left) + float(logR_right))
    dx_left = logR_mid - float(logR_left)
    dx_right = float(logR_right) - logR_mid
    if dx_left <= 0.0 or dx_right <= 0.0:
        return {"left": math.nan, "right": math.nan, "max_abs": math.nan}
    g_left = (np.asarray(y_mid, dtype=float) - np.asarray(y_left, dtype=float)) / dx_left
    g_right = (np.asarray(y_right, dtype=float) - np.asarray(y_mid, dtype=float)) / dx_right
    y_left_mid = 0.5 * (np.asarray(y_left, dtype=float) + np.asarray(y_mid, dtype=float))
    y_right_mid = 0.5 * (np.asarray(y_mid, dtype=float) + np.asarray(y_right, dtype=float))
    r_left = float(_scaled_residual_at(0.5 * (float(logR_left) + logR_mid), y_left_mid, g_left, lambda0, params)[0])
    r_right = float(_scaled_residual_at(0.5 * (logR_mid + float(logR_right)), y_right_mid, g_right, lambda0, params)[0])
    return {"left": r_left, "right": r_right, "max_abs": float(max(abs(r_left), abs(r_right)))}


def _source_transition_radii_rg(params) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    source_fraction = float(getattr(params, "stream_source_fraction", 0.0))
    legacy_fraction = float(getattr(params, "stream_mass_fraction", 0.0))
    active_fraction = source_fraction if source_fraction != 0.0 else legacy_fraction
    if active_fraction > 0.0:
        center_fraction = float(getattr(params, "stream_source_center_fraction", getattr(params, "stream_mass_center_fraction", 0.8)))
        width = float(getattr(params, "stream_source_log_width", getattr(params, "stream_mass_log_width", 0.08)))
        center_rg = float(center_fraction * params.R_out_rg)
        shape = str(getattr(params, "stream_source_shape", "tanh")).strip().lower()
        blend = float(getattr(params, "stream_source_shape_blend", 1.0))
        compact = shape in {"compact", "compact_c2", "c2"} and blend >= 1.0 - 1.0e-12
        transitions.extend(
            [
                {
                    "name": "source_support_inner" if compact else "source_width_inner",
                    "R_rg": float(center_rg * math.exp(-width)),
                    "exact_zero_derivative_edge": bool(compact),
                },
                {
                    "name": "source_peak",
                    "R_rg": center_rg,
                    "exact_zero_derivative_edge": False,
                },
                {
                    "name": "source_support_outer" if compact else "source_width_outer",
                    "R_rg": float(center_rg * math.exp(width)),
                    "exact_zero_derivative_edge": bool(compact),
                },
            ]
        )
    if params.outer_buffer_inner_rg is not None:
        transitions.append(
            {
                "name": "outer_buffer_inner",
                "R_rg": float(params.outer_buffer_inner_rg),
                "exact_zero_derivative_edge": False,
            }
        )
    return transitions


def _transition_grid_audit(x: np.ndarray, params, peak_R_rg: float | None = None) -> dict[str, Any]:
    _logu, _logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x, params)
    node_R_rg = np.exp(logR) / params.r_g
    interval_mid_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    transitions = _source_transition_radii_rg(params)
    if peak_R_rg is not None and np.isfinite(float(peak_R_rg)):
        transitions.append({"name": "peak_interval_R", "R_rg": float(peak_R_rg), "exact_zero_derivative_edge": False})
    entries: list[dict[str, Any]] = []
    for transition in transitions:
        R_rg = float(transition["R_rg"])
        if not np.isfinite(R_rg) or R_rg <= 0.0:
            continue
        log_value = math.log(R_rg * params.r_g)
        nearest_node = int(np.argmin(np.abs(logR - log_value)))
        nearest_interval = int(np.argmin(np.abs(interval_mid_rg - R_rg))) if interval_mid_rg.size else -1
        right_index = int(np.searchsorted(logR, log_value, side="right"))
        straddle = -1
        if 0 < right_index < len(logR):
            straddle = right_index - 1
        entries.append(
            {
                "name": transition["name"],
                "R_rg": R_rg,
                "exact_zero_derivative_edge": bool(transition.get("exact_zero_derivative_edge", False)),
                "nearest_node": nearest_node,
                "nearest_node_R_rg": float(node_R_rg[nearest_node]),
                "nearest_node_dlnR": float(logR[nearest_node] - log_value),
                "nearest_interval": nearest_interval,
                "nearest_interval_R_mid_rg": float(interval_mid_rg[nearest_interval]) if nearest_interval >= 0 else math.nan,
                "straddling_interval": straddle,
                "straddling_interval_R_left_rg": float(node_R_rg[straddle]) if straddle >= 0 else math.nan,
                "straddling_interval_R_right_rg": float(node_R_rg[straddle + 1]) if straddle >= 0 else math.nan,
                "is_existing_node": bool(abs(float(logR[nearest_node] - log_value)) < 1.0e-10),
            }
        )
    source_dx = np.diff(logR)
    return {
        "n_nodes": int(params.n_nodes),
        "R_out_rg": float(params.R_out_rg),
        "node_min_dlnR": float(np.min(source_dx)) if source_dx.size else math.nan,
        "node_max_dlnR": float(np.max(source_dx)) if source_dx.size else math.nan,
        "transitions": entries,
    }


def _mandatory_transition_xi_values(x_old: np.ndarray, old_params, target_params) -> list[tuple[int, float, str]]:
    if not TRANSITION_ALIGN_NODES:
        return []
    _logu, _logT, _logMdot, logR_son, _lambda0, logR_old = pilot._unpack(x_old, old_params)
    span = max(float(np.log(target_params.R_out) - logR_son), 1.0e-300)
    residual = np.asarray(pilot.residual(x_old, old_params), dtype=float)
    n_old = int(old_params.n_nodes)
    interval_R = residual[0 : 2 * (n_old - 1) : 2]
    peak_idx = int(np.argmax(np.abs(interval_R))) if interval_R.size else -1
    peak_R_rg = float(np.exp(0.5 * (logR_old[peak_idx] + logR_old[peak_idx + 1])) / old_params.r_g) if peak_idx >= 0 else math.nan

    candidates: list[tuple[int, float, str]] = []
    transitions = []
    if TRANSITION_ALIGN_INCLUDE_SOURCE or TRANSITION_ALIGN_INCLUDE_BUFFER:
        for transition in _source_transition_radii_rg(old_params):
            name = str(transition.get("name", ""))
            is_source = name.startswith("source_")
            is_buffer = name.startswith("outer_buffer")
            if (is_source and TRANSITION_ALIGN_INCLUDE_SOURCE) or (is_buffer and TRANSITION_ALIGN_INCLUDE_BUFFER):
                transitions.append(transition)
    if TRANSITION_ALIGN_INCLUDE_PEAK and np.isfinite(peak_R_rg):
        transitions.append({"name": "peak_interval_R", "R_rg": peak_R_rg})
    for transition in transitions:
        R_rg = float(transition["R_rg"])
        if not np.isfinite(R_rg) or R_rg <= 0.0:
            continue
        primary_xi = (math.log(R_rg * old_params.r_g) - float(logR_son)) / span
        candidates.append((0, float(primary_xi), str(transition["name"])))
        if TRANSITION_SIDE_FRACTION > 0.0:
            candidates.append((1, (math.log(R_rg * (1.0 - TRANSITION_SIDE_FRACTION) * old_params.r_g) - float(logR_son)) / span, f"{transition['name']}_minus"))
            candidates.append((1, (math.log(R_rg * (1.0 + TRANSITION_SIDE_FRACTION) * old_params.r_g) - float(logR_son)) / span, f"{transition['name']}_plus"))
    return [(priority, xi, name) for priority, xi, name in candidates if np.isfinite(xi) and 0.0 < xi < 1.0]


def _insert_xi_if_new(xi_values: list[float], xi: float, min_spacing: float = 1.0e-10) -> bool:
    if any(abs(float(value) - float(xi)) <= min_spacing for value in xi_values):
        return False
    xi_values.append(float(xi))
    xi_values.sort()
    return True


def _apply_local_overrides(params):
    kwargs: dict[str, Any] = {}
    if OUTER_BUFFER_WEIGHT_RAW:
        weight = float(OUTER_BUFFER_WEIGHT_RAW)
        kwargs.update(
            outer_buffer_radial_weight=weight,
            outer_buffer_energy_weight=weight,
            outer_buffer_boundary_weight=weight,
        )
    if OUTER_BUFFER_INNER_RG_RAW:
        if OUTER_BUFFER_INNER_RG_RAW.lower() in {"none", "off", "null"}:
            kwargs["outer_buffer_inner_rg"] = None
        else:
            kwargs["outer_buffer_inner_rg"] = float(OUTER_BUFFER_INNER_RG_RAW)
    if OUTER_CLOSURE_RAW:
        kwargs["outer_closure"] = OUTER_CLOSURE_RAW
    if OUTER_ROBIN_CHI_RAW:
        kwargs["outer_robin_chi"] = float(OUTER_ROBIN_CHI_RAW)
    if OUTER_ROBIN_SLOPE_TARGET_RAW:
        kwargs["outer_robin_slope_target"] = float(OUTER_ROBIN_SLOPE_TARGET_RAW)
    if OUTER_ROBIN_SLOPE_SCALE_RAW:
        kwargs["outer_robin_slope_scale"] = float(OUTER_ROBIN_SLOPE_SCALE_RAW)
    if OUTER_OMEGA_LOG_OFFSET_RAW:
        kwargs["outer_omega_log_offset"] = float(OUTER_OMEGA_LOG_OFFSET_RAW)
    if INTERVAL_RESIDUAL_FORM_RAW:
        kwargs["interval_residual_form"] = INTERVAL_RESIDUAL_FORM_RAW
    if INTEGRATED_WEIGHTING_RAW:
        kwargs["integrated_residual_weighting"] = INTEGRATED_WEIGHTING_RAW
    return replace(params, **kwargs) if kwargs else params


def _restore_checkpoint_params(params, data) -> Any:
    kwargs: dict[str, Any] = {}
    if "custom_grid_xi" in data:
        custom_grid = np.asarray(data["custom_grid_xi"], dtype=float)
        if custom_grid.size == int(params.n_nodes):
            kwargs["custom_grid_xi"] = tuple(float(value) for value in custom_grid)
    if "outer_match_log_slopes" in data:
        slopes = np.asarray(data["outer_match_log_slopes"], dtype=float)
        if slopes.shape == (2,) and np.all(np.isfinite(slopes)):
            kwargs["outer_match_log_slopes"] = (float(slopes[0]), float(slopes[1]))
    return replace(params, **kwargs) if kwargs else params


def _state_and_params_for_n(anchor_z: np.ndarray, anchor_params, n_nodes: int) -> tuple[np.ndarray, Any]:
    params = anchor_params
    z = anchor_z
    if int(params.n_nodes) != int(n_nodes):
        target_params = replace(params, n_nodes=int(n_nodes), custom_grid_xi=None)
        profile = transonic_profile_from_state_vector(z, params)
        state_remap_method = "pchip" if REMAP_METHOD in PCHIP_REMAP_METHODS else REMAP_METHOD
        z = scan.remap_profile_to_new_sonic_grid(
            profile,
            target_params,
            temperature_mdot_power=0.0,
            method=state_remap_method,
        )
        params = scan.apply_outer_slopes_from_state(z, target_params)
    local_params = replace(params, wind_sink_fraction=0.0, mdot_profile_mode="source_sink")
    return z, _apply_local_overrides(local_params)


def _make_seed(anchor_z: np.ndarray, anchor_params) -> tuple[np.ndarray, Any]:
    z, local_params = _state_and_params_for_n(anchor_z, anchor_params, N_NODES)
    logu, logT, logR_son, lambda0, logR = scan.unpack_state(z, local_params)
    mdot_seed = np.asarray([stream_mass_rate_and_derivative(float(x), local_params)[0] for x in logR], dtype=float)
    x0 = pilot._pack(logu, logT, np.log(mdot_seed), logR_son, lambda0)
    lower, upper = pilot._bounds(local_params)
    return np.clip(x0, lower + 1.0e-12, upper - 1.0e-12), local_params


def _remap_local_x_to_params(x_old: np.ndarray, old_params, new_params) -> np.ndarray:
    logu_old, logT_old, logMdot_old, logR_son, lambda0, logR_old = pilot._unpack(x_old, old_params)
    logR_new = pilot.computational_grid(new_params, logR_son)
    def interp(values: np.ndarray) -> np.ndarray:
        if REMAP_METHOD in PCHIP_REMAP_METHODS:
            try:
                from scipy.interpolate import PchipInterpolator

                return np.asarray(PchipInterpolator(logR_old, values, extrapolate=True)(logR_new), dtype=float)
            except Exception:
                return np.interp(logR_new, logR_old, values)
        return np.interp(logR_new, logR_old, values)

    logu_new = interp(logu_old)
    logT_new = interp(logT_old)
    logMdot_new = interp(logMdot_old)
    if REMAP_METHOD in {"mass_ode", "pchip_mass_ode", "nested_mass_ode"}:
        logMdot_new = _mdot_ode_logmdot_seed(logu_new, logT_new, logMdot_new, logR_new, lambda0, new_params)
    elif REMAP_METHOD in {"log_mass_ode", "log_mdot_ode"}:
        logMdot_new = _mass_ode_logmdot_seed(logu_new, logT_new, logMdot_new, logR_new, lambda0, new_params)
    elif REMAP_METHOD in MASS_DEFECT_REMAP_METHODS:
        logMdot_new = _defect_preserving_logmdot_seed(
            x_old,
            old_params,
            logu_new,
            logT_new,
            logMdot_new,
            logR_new,
            lambda0,
            new_params,
        )
    if REMAP_METHOD in STATE_DEFECT_REMAP_METHODS:
        logu_new, logT_new = _defect_preserving_state_seed(
            x_old,
            old_params,
            logu_new,
            logT_new,
            logMdot_new,
            logR_new,
            lambda0,
            new_params,
        )
        logMdot_new = _defect_preserving_logmdot_seed(
            x_old,
            old_params,
            logu_new,
            logT_new,
            logMdot_new,
            logR_new,
            lambda0,
            new_params,
        )
    x_new = pilot._pack(logu_new, logT_new, logMdot_new, logR_son, lambda0)
    lower, upper = pilot._bounds(new_params)
    return np.clip(x_new, lower + 1.0e-12, upper - 1.0e-12)


def _hermite_interpolate(logR_old: np.ndarray, values: np.ndarray, slopes: np.ndarray, logR_new: np.ndarray) -> np.ndarray:
    old = np.asarray(logR_old, dtype=float)
    vals = np.asarray(values, dtype=float)
    deriv = np.asarray(slopes, dtype=float)
    new = np.asarray(logR_new, dtype=float)
    out = np.empty_like(new)
    for pos, x_value in enumerate(new):
        x = float(x_value)
        if x <= old[0]:
            idx = 0
        elif x >= old[-1]:
            idx = old.size - 2
        else:
            idx = int(np.searchsorted(old, x, side="right") - 1)
        dx = float(old[idx + 1] - old[idx])
        if dx <= 0.0:
            out[pos] = vals[idx]
            continue
        t = float((x - old[idx]) / dx)
        if (
            np.isfinite(deriv[idx])
            and np.isfinite(deriv[idx + 1])
            and np.isfinite(vals[idx])
            and np.isfinite(vals[idx + 1])
        ):
            h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
            h10 = t**3 - 2.0 * t**2 + t
            h01 = -2.0 * t**3 + 3.0 * t**2
            h11 = t**3 - t**2
            candidate = h00 * vals[idx] + h10 * dx * deriv[idx] + h01 * vals[idx + 1] + h11 * dx * deriv[idx + 1]
            limit = abs(float(SOURCE_MICRO_HERMITE_OVERSHOOT_LIMIT))
            lo = min(float(vals[idx]), float(vals[idx + 1])) - limit
            hi = max(float(vals[idx]), float(vals[idx + 1])) + limit
            if np.isfinite(candidate) and lo <= candidate <= hi:
                out[pos] = candidate
            else:
                out[pos] = (1.0 - t) * vals[idx] + t * vals[idx + 1]
        else:
            out[pos] = (1.0 - t) * vals[idx] + t * vals[idx + 1]
    return out


def _state_and_mdot_hermite_slopes(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_params = pilot._local_params(params, logR, logMdot)
    y_slopes = np.empty((len(logR), 2), dtype=float)
    mdot_slopes = np.empty(len(logR), dtype=float)
    fallback_state = np.gradient(np.vstack([logu, logT]), logR, axis=1, edge_order=1)
    fallback_mdot = np.gradient(logMdot, logR, edge_order=1)
    for idx, x_value in enumerate(logR):
        y = np.asarray([logu[idx], logT[idx]], dtype=float)
        slope = _ode_slope(float(x_value), y, lambda0, local_params)
        if not np.all(np.isfinite(slope)):
            slope = np.asarray([fallback_state[0, idx], fallback_state[1, idx]], dtype=float)
        y_slopes[idx] = slope
        try:
            mdot = max(float(np.exp(logMdot[idx])), 1.0e-300)
            source_prime = stream_source_prime(float(x_value), local_params)
            wind_prime = pilot._wind_mass_prime(float(x_value), y, slope, lambda0, local_params)
            mdot_slopes[idx] = float((wind_prime - source_prime) / mdot)
        except Exception:
            mdot_slopes[idx] = float(fallback_mdot[idx])
    return y_slopes[:, 0], y_slopes[:, 1], mdot_slopes


def _hermite_remap_local_x_to_params(x_old: np.ndarray, old_params, new_params) -> np.ndarray:
    logu_old, logT_old, logMdot_old, logR_son, lambda0, logR_old = pilot._unpack(x_old, old_params)
    logR_new = pilot.computational_grid(new_params, logR_son)
    slope_u, slope_T, slope_mdot = _state_and_mdot_hermite_slopes(
        logu_old, logT_old, logMdot_old, logR_old, lambda0, old_params
    )
    logu_new = _hermite_interpolate(logR_old, logu_old, slope_u, logR_new)
    logT_new = _hermite_interpolate(logR_old, logT_old, slope_T, logR_new)
    logMdot_new = _hermite_interpolate(logR_old, logMdot_old, slope_mdot, logR_new)
    x_new = pilot._pack(logu_new, logT_new, logMdot_new, logR_son, lambda0)
    lower, upper = pilot._bounds(new_params)
    return np.clip(x_new, lower + 1.0e-12, upper - 1.0e-12)


def _source_microdomain_params_from_x(x_old: np.ndarray, old_params):
    logu, _logT, _logMdot, logR_son, _lambda0, logR_old = pilot._unpack(x_old, old_params)
    _ = logu
    band_min, band_max = _source_band_default_bounds_logR(old_params)
    center = _source_center_logR(old_params)
    if not (np.isfinite(band_min) and np.isfinite(band_max) and band_max > band_min and np.isfinite(center)):
        return old_params, {
            "source_microdomain_enabled": True,
            "source_microdomain_applied": False,
            "source_microdomain_reason": "invalid source support",
        }
    n_band = max(3, int(SOURCE_MICRO_NODES))
    band_nodes = np.linspace(band_min, band_max, n_band)
    absolute_nodes = np.concatenate([logR_old, band_nodes, np.asarray([band_min, center, band_max], dtype=float)])
    absolute_nodes = absolute_nodes[(absolute_nodes >= logR_old[0] - 1.0e-12) & (absolute_nodes <= logR_old[-1] + 1.0e-12)]
    absolute_nodes = np.asarray(sorted({round(float(value), 14) for value in absolute_nodes}), dtype=float)
    absolute_nodes[0] = float(logR_old[0])
    absolute_nodes[-1] = float(logR_old[-1])
    span = max(float(logR_old[-1] - logR_son), 1.0e-300)
    xi = _enforce_min_spacing((absolute_nodes - float(logR_son)) / span)
    new_params = replace(old_params, n_nodes=int(xi.size), custom_grid_xi=tuple(float(value) for value in xi))
    return new_params, {
        "source_microdomain_enabled": True,
        "source_microdomain_applied": True,
        "source_microdomain_old_N": int(old_params.n_nodes),
        "source_microdomain_new_N": int(new_params.n_nodes),
        "source_microdomain_requested_band_nodes": int(SOURCE_MICRO_NODES),
        "source_microdomain_actual_band_nodes": int(np.count_nonzero((absolute_nodes >= band_min) & (absolute_nodes <= band_max))),
        "source_microdomain_min_rg": float(np.exp(band_min) / old_params.r_g),
        "source_microdomain_center_rg": float(np.exp(center) / old_params.r_g),
        "source_microdomain_max_rg": float(np.exp(band_max) / old_params.r_g),
    }


def _node_preserving_refined_params(x_old: np.ndarray, old_params, target_params):
    old_n = int(old_params.n_nodes)
    target_n = int(target_params.n_nodes)
    if target_n <= old_n:
        return target_params
    _logu_old, _logT_old, _logMdot_old, logR_son, _lambda0, logR_old = pilot._unpack(x_old, old_params)
    span = max(float(np.log(old_params.R_out) - logR_son), 1.0e-300)
    xi_values = [float(value) for value in (logR_old - logR_son) / span]
    xi_values[0] = 0.0
    xi_values[-1] = 1.0
    transition_candidates = sorted(_mandatory_transition_xi_values(x_old, old_params, target_params), key=lambda item: item[0])
    inserted_transitions: list[str] = []
    for _priority, candidate_xi, name in transition_candidates:
        if len(xi_values) >= target_n:
            break
        if _insert_xi_if_new(xi_values, candidate_xi):
            inserted_transitions.append(name)
    while len(xi_values) < target_n:
        xi_array = np.asarray(xi_values, dtype=float)
        gaps = np.diff(xi_array)
        mid_xi = 0.5 * (xi_array[:-1] + xi_array[1:])
        mid_R_rg = np.exp(float(logR_son) + mid_xi * span) / old_params.r_g
        allowed = (mid_R_rg >= float(NESTED_REFINE_MIN_RG)) & (mid_R_rg <= float(NESTED_REFINE_MAX_RG))
        if np.any(allowed):
            scores = np.where(allowed, gaps, -np.inf)
            idx = int(np.argmax(scores))
        else:
            idx = int(np.argmax(gaps))
        xi_values.insert(idx + 1, 0.5 * (xi_values[idx] + xi_values[idx + 1]))
    xi = _enforce_min_spacing(np.asarray(xi_values[:target_n], dtype=float))
    return replace(target_params, custom_grid_xi=tuple(float(value) for value in xi))


def _mdot_ode_logmdot_seed(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot_guess: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    params,
) -> np.ndarray:
    local_params = pilot._local_params(params, logR, logMdot_guess)
    mdot_nodes = np.empty_like(logMdot_guess)
    mdot_nodes[0] = float(params.Mdot_g_s)
    for idx in range(len(logR) - 1):
        dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
        ym = 0.5 * (y_left + y_right)
        gm = (y_right - y_left) / dx
        wind_prime = pilot._wind_mass_prime(xm, ym, gm, lambda0, local_params)
        source_prime = stream_source_prime(xm, local_params)
        mdot_nodes[idx + 1] = mdot_nodes[idx] + float(wind_prime - source_prime) * dx
        mdot_nodes[idx + 1] = max(mdot_nodes[idx + 1], 1.0e-6 * float(params.Mdot_g_s))
    return np.log(mdot_nodes)


def _mass_target_profile(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    params,
) -> np.ndarray:
    local_params = pilot._local_params(params, logR, logMdot)
    target = np.empty(len(logR) - 1, dtype=float)
    for idx in range(len(logR) - 1):
        dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
        ym = 0.5 * (y_left + y_right)
        gm = (y_right - y_left) / dx
        logMdot_mid = 0.5 * (logMdot[idx] + logMdot[idx + 1])
        mdot_mid = float(np.exp(logMdot_mid))
        wind_prime = pilot._wind_mass_prime(xm, ym, gm, lambda0, local_params)
        source_prime = stream_source_prime(xm, local_params)
        target[idx] = float((wind_prime - source_prime) / mdot_mid)
    return target


def _mass_ode_logmdot_seed(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot_guess: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    params,
) -> np.ndarray:
    logMdot = np.asarray(logMdot_guess, dtype=float).copy()
    logMdot[0] = float(np.log(params.Mdot_g_s))
    for _ in range(max(1, DEFECT_REMAP_SWEEPS)):
        target = _mass_target_profile(logu, logT, logMdot, logR, lambda0, params)
        next_logMdot = np.empty_like(logMdot)
        next_logMdot[0] = float(np.log(params.Mdot_g_s))
        for idx, dx in enumerate(np.diff(logR)):
            next_logMdot[idx + 1] = next_logMdot[idx] + float(target[idx]) * float(dx)
        logMdot = next_logMdot
    return logMdot


def _defect_preserving_logmdot_seed(
    x_old: np.ndarray,
    old_params,
    logu_new: np.ndarray,
    logT_new: np.ndarray,
    logMdot_guess: np.ndarray,
    logR_new: np.ndarray,
    lambda0: float,
    new_params,
) -> np.ndarray:
    logu_old, logT_old, logMdot_old, _logR_son_old, _lambda0_old, logR_old = pilot._unpack(x_old, old_params)
    old_residual = np.asarray(pilot.residual(x_old, old_params), dtype=float)
    old_n = int(old_params.n_nodes)
    old_mass_start = _inner_mdot_row_index(old_params) + 1
    old_mass_defect = old_residual[old_mass_start : old_mass_start + old_n - 1] / max(float(MASS_WEIGHT), 1.0e-300)
    old_mid = 0.5 * (logR_old[:-1] + logR_old[1:])
    new_mid = 0.5 * (logR_new[:-1] + logR_new[1:])
    defect = np.interp(new_mid, old_mid, old_mass_defect, left=old_mass_defect[0], right=old_mass_defect[-1])
    logMdot_outer_target = float(logMdot_old[-1])

    logMdot = np.asarray(logMdot_guess, dtype=float).copy()
    logMdot[0] = float(np.log(new_params.Mdot_g_s))
    for _ in range(max(1, DEFECT_REMAP_SWEEPS)):
        target = _mass_target_profile(logu_new, logT_new, logMdot, logR_new, lambda0, new_params) + defect
        span = max(float(logR_new[-1] - logR_new[0]), 1.0e-300)
        predicted_outer = float(logMdot[0] + np.sum(target * np.diff(logR_new)))
        budget_correction = (logMdot_outer_target - predicted_outer) / span
        target = target + budget_correction
        next_logMdot = np.empty_like(logMdot)
        next_logMdot[0] = float(np.log(new_params.Mdot_g_s))
        for idx, dx in enumerate(np.diff(logR_new)):
            next_logMdot[idx + 1] = next_logMdot[idx] + float(target[idx]) * float(dx)
        logMdot = next_logMdot
    return logMdot


def _defect_preserving_state_seed(
    x_old: np.ndarray,
    old_params,
    logu_guess: np.ndarray,
    logT_guess: np.ndarray,
    logMdot_new: np.ndarray,
    logR_new: np.ndarray,
    lambda0: float,
    new_params,
) -> tuple[np.ndarray, np.ndarray]:
    logu_old, logT_old, _logMdot_old, _logR_son_old, _lambda0_old, logR_old = pilot._unpack(x_old, old_params)
    old_residual = np.asarray(pilot.residual(x_old, old_params), dtype=float)
    old_n = int(old_params.n_nodes)
    old_state_defect = old_residual[: 2 * (old_n - 1)].reshape(old_n - 1, 2)
    old_mid = 0.5 * (logR_old[:-1] + logR_old[1:])
    new_mid = 0.5 * (logR_new[:-1] + logR_new[1:])
    defect = np.column_stack(
        [
            np.interp(new_mid, old_mid, old_state_defect[:, 0], left=old_state_defect[0, 0], right=old_state_defect[-1, 0]),
            np.interp(new_mid, old_mid, old_state_defect[:, 1], left=old_state_defect[0, 1], right=old_state_defect[-1, 1]),
        ]
    )

    y_guess = np.column_stack([np.asarray(logu_guess, dtype=float), np.asarray(logT_guess, dtype=float)])
    inner_target = y_guess[0].copy()
    outer_target = np.array([float(logu_old[-1]), float(logT_old[-1])], dtype=float)
    lower, upper = pilot._bounds(new_params)
    n = int(new_params.n_nodes)
    lower_y = np.column_stack([lower[:n], lower[n : 2 * n]]) + 1.0e-12
    upper_y = np.column_stack([upper[:n], upper[n : 2 * n]]) - 1.0e-12
    damping = float(np.clip(STATE_DEFECT_REMAP_DAMPING, 0.0, 1.0))
    max_dy = abs(float(STATE_DEFECT_REMAP_MAX_DY))
    local_params = pilot._local_params(new_params, logR_new, logMdot_new)

    for _ in range(max(1, STATE_DEFECT_REMAP_SWEEPS)):
        next_y = np.empty_like(y_guess)
        next_y[0] = inner_target
        for idx, dx in enumerate(np.diff(logR_new)):
            dx = float(dx)
            y_left = next_y[idx]
            y_right_guess = y_guess[idx + 1]
            y_mid = 0.5 * (y_left + y_right_guess)
            candidate = y_right_guess.copy()
            try:
                A, c, _radial_scale, _energy_scale = scaled_differential_matrix(
                    float(0.5 * (logR_new[idx] + logR_new[idx + 1])),
                    y_mid,
                    lambda0,
                    local_params,
                )
                current_g = (y_right_guess - y_left) / dx
                current_residual = np.asarray(A @ current_g + c, dtype=float)
                delta_g = np.linalg.solve(A, defect[idx] - current_residual)
                step = dx * np.asarray(delta_g, dtype=float)
                if max_dy > 0.0:
                    step_norm = float(np.max(np.abs(step)))
                    if np.isfinite(step_norm) and step_norm > max_dy:
                        step *= max_dy / step_norm
                proposed = y_right_guess + step
                if np.all(np.isfinite(proposed)):
                    candidate = (1.0 - damping) * y_right_guess + damping * proposed
            except Exception:
                candidate = y_right_guess.copy()
            next_y[idx + 1] = np.clip(candidate, lower_y[idx + 1], upper_y[idx + 1])

        if STATE_DEFECT_REMAP_MATCH_OUTER:
            span = max(float(logR_new[-1] - logR_new[0]), 1.0e-300)
            xi = (logR_new - logR_new[0]) / span
            correction = outer_target - next_y[-1]
            next_y = next_y + xi[:, None] * correction
            next_y[0] = inner_target
            next_y = np.clip(next_y, lower_y, upper_y)
        y_guess = next_y
    return y_guess[:, 0].copy(), y_guess[:, 1].copy()


def _residual_remesh(
    x: np.ndarray,
    params,
    eta_E: float,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    if RESIDUAL_REMESH_STRENGTH <= 0.0:
        return x, params, {}

    source_profile = _profile("residual_remesh_source", x, params, eta_E)
    logu, logT, logMdot, logR_son, _lambda0, logR = pilot._unpack(x, params)
    span = max(float(logR[-1] - logR[0]), 1.0e-12)
    source_xi = (np.asarray(logR, dtype=float) - float(logR[0])) / span
    interval_mid_xi = 0.5 * (source_xi[:-1] + source_xi[1:])
    interval_mid_R_rg = np.asarray(source_profile["R_mid_rg"], dtype=float)
    dense_count = max(4096, int(RESIDUAL_REMESH_DENSE_FACTOR) * int(params.n_nodes))
    dense_xi = np.linspace(0.0, 1.0, dense_count)

    def dense_from_interval(values: np.ndarray) -> np.ndarray:
        arr = _normalize_component(np.asarray(values, dtype=float))
        if arr.size == 0:
            return np.zeros_like(dense_xi)
        return np.interp(dense_xi, interval_mid_xi, arr, left=arr[0], right=arr[-1])

    outer_width = max(float(REMESH_OUTER_WIDTH), 1.0e-5)
    outer_dense = np.exp(-0.5 * ((dense_xi - 1.0) / outer_width) ** 2)
    composite = (
        W_REMESH_INTERVAL_R * dense_from_interval(np.asarray(source_profile["interval_R"], dtype=float))
        + W_REMESH_INTERVAL_E * dense_from_interval(np.asarray(source_profile["interval_E"], dtype=float))
        + W_REMESH_MASS * dense_from_interval(np.asarray(source_profile["local_mass_residual"], dtype=float))
        + W_REMESH_SOURCE * dense_from_interval(np.asarray(source_profile["Mstream_prime_over_Mdot"], dtype=float))
        + W_REMESH_WIND * dense_from_interval(np.asarray(source_profile["Mwind_prime_over_Mdot"], dtype=float))
        + W_REMESH_MDOT_GRAD * dense_from_interval(np.asarray(source_profile["dlogMdot_dlogR"], dtype=float))
        + W_REMESH_OUTER * _normalize_component(outer_dense)
    )
    composite = _smooth_score(composite, RESIDUAL_REMESH_SMOOTH_PASSES)
    monitor = RESIDUAL_REMESH_FLOOR + float(RESIDUAL_REMESH_STRENGTH) * _normalize_component(composite) ** float(
        RESIDUAL_REMESH_POWER
    )
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (monitor[:-1] + monitor[1:]) * np.diff(dense_xi))])
    cumulative /= cumulative[-1]
    target = np.linspace(0.0, 1.0, int(params.n_nodes))
    adapted = np.interp(target, cumulative, dense_xi)
    reference = np.interp(target, np.linspace(0.0, 1.0, source_xi.size), source_xi)
    blended = _enforce_min_spacing((1.0 - float(RESIDUAL_REMESH_BLEND)) * reference + float(RESIDUAL_REMESH_BLEND) * adapted)
    remeshed_params = replace(params, custom_grid_xi=tuple(float(value) for value in blended))
    remeshed_x = _remap_local_x_to_params(x, params, remeshed_params)
    remeshed_params = scan.apply_outer_slopes_from_state(_z_from_x(remeshed_x, remeshed_params), remeshed_params)

    initial_full = float(np.linalg.norm(pilot.residual(x, params), ord=np.inf))
    remeshed_full = float(np.linalg.norm(pilot.residual(remeshed_x, remeshed_params), ord=np.inf))
    peak_monitor = int(np.argmax(monitor))
    peak_R_rg = float(interval_mid_R_rg[int(np.argmax(np.abs(source_profile["interval_R"])))]) if interval_mid_R_rg.size else math.nan
    info = {
        "residual_remesh_strength": float(RESIDUAL_REMESH_STRENGTH),
        "residual_remesh_blend": float(RESIDUAL_REMESH_BLEND),
        "residual_remesh_power": float(RESIDUAL_REMESH_POWER),
        "residual_remesh_initial_full": initial_full,
        "residual_remesh_seed_full": remeshed_full,
        "residual_remesh_peak_monitor_rg": float(np.exp(float(logR[0]) + dense_xi[peak_monitor] * span) / params.r_g),
        "residual_remesh_peak_interval_R_rg": peak_R_rg,
        "residual_remesh_outer_1pct_nodes": int(np.count_nonzero(blended >= 0.99)),
        "residual_remesh_outer_5pct_nodes": int(np.count_nonzero(blended >= 0.95)),
        "residual_remesh_source_dx_outer": float(source_xi[-1] - source_xi[-2]),
        "residual_remesh_target_dx_outer": float(blended[-1] - blended[-2]),
        "residual_remesh_source_min_dxi": float(np.min(np.diff(source_xi))),
        "residual_remesh_target_min_dxi": float(np.min(np.diff(blended))),
    }
    return remeshed_x, remeshed_params, info



def _jac_norms(result, params) -> dict[str, Any]:
    jac = getattr(result, "jac", None)
    if jac is None:
        return {}
    if hasattr(jac, "multiply"):
        row_norm = np.sqrt(np.asarray(jac.multiply(jac).sum(axis=1)).ravel())
        col_norm = np.sqrt(np.asarray(jac.multiply(jac).sum(axis=0)).ravel())
    else:
        array = np.asarray(jac, dtype=float)
        row_norm = np.linalg.norm(array, axis=1)
        col_norm = np.linalg.norm(array, axis=0)
    n = int(params.n_nodes)
    mass_start = 2 * (n - 1) + 2 + 2 + 1
    out: dict[str, Any] = {
        "jac_row_norm_min": float(np.nanmin(row_norm)) if row_norm.size else math.nan,
        "jac_row_norm_median": float(np.nanmedian(row_norm)) if row_norm.size else math.nan,
        "jac_row_norm_max": float(np.nanmax(row_norm)) if row_norm.size else math.nan,
        "jac_col_norm_min": float(np.nanmin(col_norm)) if col_norm.size else math.nan,
        "jac_col_norm_median": float(np.nanmedian(col_norm)) if col_norm.size else math.nan,
        "jac_col_norm_max": float(np.nanmax(col_norm)) if col_norm.size else math.nan,
        "jac_row_norm_interval_R": row_norm[0 : 2 * (n - 1) : 2].tolist(),
        "jac_row_norm_interval_E": row_norm[1 : 2 * (n - 1) : 2].tolist(),
        "jac_row_norm_mass": row_norm[mass_start : mass_start + n - 1].tolist(),
        "jac_col_norm_logu": col_norm[:n].tolist(),
        "jac_col_norm_logT": col_norm[n : 2 * n].tolist(),
        "jac_col_norm_logMdot": col_norm[2 * n : 3 * n].tolist(),
    }
    if out["jac_row_norm_mass"]:
        mass_norm = np.asarray(out["jac_row_norm_mass"], dtype=float)
        out["jac_row_norm_mass_median"] = float(np.nanmedian(mass_norm))
        out["jac_row_norm_mass_max"] = float(np.nanmax(mass_norm))
    return out


def _radial_residual_representation_audit(x: np.ndarray, params, eta_E: float, top_n: int | None = None) -> dict[str, Any]:
    if not RADIAL_AUDIT_FORMS:
        return {}
    _set_eta(eta_E)
    logu, logT, logMdot, _logR_son, lambda0, logR = pilot._unpack(x, params)
    local_params = pilot._local_params(params, logR, logMdot)
    residual = np.asarray(pilot.residual(x, params), dtype=float)
    n = int(params.n_nodes)
    interval_R = residual[0 : 2 * (n - 1) : 2]
    interval_E = residual[1 : 2 * (n - 1) : 2]
    if interval_R.size == 0:
        return {}
    count = min(int(RADIAL_AUDIT_TOP_N if top_n is None else top_n), interval_R.size)
    top_indices = np.argsort(-np.abs(interval_R))[:count]
    rows: list[dict[str, Any]] = []
    for idx in top_indices:
        idx = int(idx)
        dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
        ym_linear = 0.5 * (y_left + y_right)
        g_secant = (y_right - y_left) / dx
        r_mid = _scaled_residual_at(xm, ym_linear, g_secant, lambda0, local_params)
        r_left = _scaled_residual_at(logR[idx], y_left, g_secant, lambda0, local_params)
        r_right = _scaled_residual_at(logR[idx + 1], y_right, g_secant, lambda0, local_params)
        trap = 0.5 * (r_left + r_right)
        simpson = (r_left + 4.0 * r_mid + r_right) / 6.0
        split_linear = _split_interval_radial_residual(logR[idx], y_left, logR[idx + 1], y_right, ym_linear, lambda0, local_params)

        g_left_ode = _ode_slope(logR[idx], y_left, lambda0, local_params)
        g_right_ode = _ode_slope(logR[idx + 1], y_right, lambda0, local_params)
        if np.all(np.isfinite(g_left_ode)) and np.all(np.isfinite(g_right_ode)):
            y_mid_hermite = ym_linear - 0.125 * dx * (g_right_ode - g_left_ode)
            split_hermite = _split_interval_radial_residual(
                logR[idx], y_left, logR[idx + 1], y_right, y_mid_hermite, lambda0, local_params
            )
            hermite_mid_offset_norm = float(np.linalg.norm(y_mid_hermite - ym_linear))
        else:
            split_hermite = {"left": math.nan, "right": math.nan, "max_abs": math.nan}
            hermite_mid_offset_norm = math.nan

        logMdot_mid = 0.5 * (logMdot[idx] + logMdot[idx + 1])
        mdot_mid = float(np.exp(logMdot_mid))
        wind_prime = pilot._wind_mass_prime(xm, ym_linear, g_secant, lambda0, local_params)
        source_prime = stream_source_prime(xm, local_params)
        buffer_weights = _outer_buffer_interval_weights(xm, local_params)
        terms = _radial_terms(xm, ym_linear, g_secant, lambda0, local_params)
        tau = float(
            max(
                abs(float(interval_R[idx]) - float(trap[0])),
                abs(float(interval_R[idx]) - float(simpson[0])),
                abs(float(trap[0]) - float(simpson[0])),
                abs(float(interval_R[idx]) - float(split_linear["max_abs"])),
            )
        )
        rows.append(
            {
                "interval_index": idx,
                "R_mid_rg": float(np.exp(xm) / local_params.r_g),
                "R_left_rg": float(np.exp(logR[idx]) / local_params.r_g),
                "R_right_rg": float(np.exp(logR[idx + 1]) / local_params.r_g),
                "h_dlnR": float(dx),
                "R_diff_reported": float(interval_R[idx]),
                "E_diff_reported": float(interval_E[idx]),
                "R_midpoint_scaled": float(r_mid[0]),
                "R_trapezoid_equiv": float(trap[0]),
                "R_simpson_equiv": float(simpson[0]),
                "R_split_linear_left": float(split_linear["left"]),
                "R_split_linear_right": float(split_linear["right"]),
                "R_split_linear_max_abs": float(split_linear["max_abs"]),
                "R_split_hermite_left": float(split_hermite["left"]),
                "R_split_hermite_right": float(split_hermite["right"]),
                "R_split_hermite_max_abs": float(split_hermite["max_abs"]),
                "hermite_mid_offset_norm": hermite_mid_offset_norm,
                "representation_tau": tau,
                "source_prime_over_Mdot": float(source_prime / max(mdot_mid, 1.0e-300)),
                "wind_prime_over_Mdot": float(wind_prime / max(mdot_mid, 1.0e-300)),
                "outer_buffer_radial_weight": float(buffer_weights[0]),
                "outer_buffer_energy_weight": float(buffer_weights[1]),
                "radial_terms": terms,
            }
        )
    peak_idx = int(top_indices[0])
    return {
        "forms": list(RADIAL_AUDIT_FORMS),
        "top_n": count,
        "peak_interval_index": peak_idx,
        "peak_interval_R_rg": float(np.exp(0.5 * (logR[peak_idx] + logR[peak_idx + 1])) / local_params.r_g),
        "rows": rows,
    }


def _local_block_jacobian_audit(x: np.ndarray, params, eta_E: float, peak_interval_index: int | None = None) -> dict[str, Any]:
    if not JACOBIAN_AUDIT:
        return {}
    _set_eta(eta_E)
    residual0 = np.asarray(pilot.residual(x, params), dtype=float)
    n = int(params.n_nodes)
    if peak_interval_index is None:
        interval_R = residual0[0 : 2 * (n - 1) : 2]
        peak_interval_index = int(np.argmax(np.abs(interval_R))) if interval_R.size else 0
    half_width = max(1, int(JACOBIAN_AUDIT_HALF_WIDTH))
    first_interval = max(0, int(peak_interval_index) - half_width)
    last_interval = min(n - 2, int(peak_interval_index) + half_width)
    interval_indices = np.arange(first_interval, last_interval + 1, dtype=int)
    node_indices = np.arange(first_interval, last_interval + 2, dtype=int)
    mass_start = _inner_mdot_row_index(params) + 1
    row_indices: list[int] = []
    row_kind: list[str] = []
    for idx in interval_indices:
        row_indices.extend([2 * int(idx), 2 * int(idx) + 1, mass_start + int(idx)])
        row_kind.extend(["radial", "energy", "mass"])
    if JACOBIAN_AUDIT_INCLUDE_GLOBALS:
        row_indices.extend([2 * (n - 1), 2 * (n - 1) + 1])
        row_kind.extend(["outer_omega", "outer_energy"])
    row_array = np.asarray(row_indices, dtype=int)

    variable_cols: list[int] = []
    variable_kind: list[str] = []
    for idx in node_indices:
        variable_cols.append(int(idx))
        variable_kind.append("logu")
    for idx in node_indices:
        variable_cols.append(int(n + idx))
        variable_kind.append("logT")
    for idx in node_indices:
        variable_cols.append(int(2 * n + idx))
        variable_kind.append("logMdot")
    if JACOBIAN_AUDIT_INCLUDE_GLOBALS:
        variable_cols.extend([3 * n, 3 * n + 1])
        variable_kind.extend(["logR_son", "lambda0"])
    col_array = np.asarray(variable_cols, dtype=int)

    lower, upper = pilot._bounds(params)
    jac = np.empty((row_array.size, col_array.size), dtype=float)
    x_ref = np.asarray(x, dtype=float)
    for j, col in enumerate(col_array):
        scale = max(abs(float(x_ref[col])), 1.0)
        step = 1.0e-6 * scale
        plus = x_ref.copy()
        minus = x_ref.copy()
        if plus[col] + step > upper[col]:
            step = min(step, max(float(upper[col] - x_ref[col]) * 0.5, 1.0e-10))
        if minus[col] - step < lower[col]:
            step = min(step, max(float(x_ref[col] - lower[col]) * 0.5, 1.0e-10))
        plus[col] += step
        minus[col] -= step
        r_plus = np.asarray(pilot.residual(plus, params), dtype=float)[row_array]
        r_minus = np.asarray(pilot.residual(minus, params), dtype=float)[row_array]
        jac[:, j] = (r_plus - r_minus) / (2.0 * step)

    row_norm = np.linalg.norm(jac, axis=1)
    col_norm = np.linalg.norm(jac, axis=0)
    singular_values = np.linalg.svd(jac, compute_uv=False) if jac.size else np.asarray([], dtype=float)
    kind_norms: dict[str, float] = {}
    for kind in sorted(set(row_kind)):
        mask = np.asarray([value == kind for value in row_kind], dtype=bool)
        kind_norms[f"row_norm_{kind}_max"] = float(np.max(row_norm[mask])) if np.any(mask) else math.nan
        kind_norms[f"row_norm_{kind}_median"] = float(np.median(row_norm[mask])) if np.any(mask) else math.nan
    col_kind_norms: dict[str, float] = {}
    for kind in sorted(set(variable_kind)):
        mask = np.asarray([value == kind for value in variable_kind], dtype=bool)
        col_kind_norms[f"col_norm_{kind}_max"] = float(np.max(col_norm[mask])) if np.any(mask) else math.nan
        col_kind_norms[f"col_norm_{kind}_median"] = float(np.median(col_norm[mask])) if np.any(mask) else math.nan

    _logu, _logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x, params)
    interval_mid_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    smax = float(np.max(singular_values)) if singular_values.size else math.nan
    smin = float(np.min(singular_values)) if singular_values.size else math.nan
    return {
        "enabled": True,
        "peak_interval_index": int(peak_interval_index),
        "peak_interval_R_rg": float(interval_mid_rg[int(peak_interval_index)]),
        "first_interval": int(first_interval),
        "last_interval": int(last_interval),
        "first_interval_R_rg": float(interval_mid_rg[first_interval]),
        "last_interval_R_rg": float(interval_mid_rg[last_interval]),
        "n_rows": int(row_array.size),
        "n_cols": int(col_array.size),
        "row_indices": row_array.tolist(),
        "row_kind": row_kind,
        "variable_cols": col_array.tolist(),
        "variable_kind": variable_kind,
        "row_norm_min": float(np.min(row_norm)) if row_norm.size else math.nan,
        "row_norm_median": float(np.median(row_norm)) if row_norm.size else math.nan,
        "row_norm_max": float(np.max(row_norm)) if row_norm.size else math.nan,
        "col_norm_min": float(np.min(col_norm)) if col_norm.size else math.nan,
        "col_norm_median": float(np.median(col_norm)) if col_norm.size else math.nan,
        "col_norm_max": float(np.max(col_norm)) if col_norm.size else math.nan,
        "singular_values": singular_values.tolist(),
        "singular_value_min": smin,
        "singular_value_max": smax,
        "condition_estimate": float(smax / smin) if np.isfinite(smax) and np.isfinite(smin) and smin > 0.0 else math.inf,
        **kind_norms,
        **col_kind_norms,
    }


def _profile(label: str, x: np.ndarray, params, eta_E: float, jac_norms: dict[str, Any] | None = None) -> dict[str, Any]:
    _set_eta(eta_E)
    logu, logT, logMdot, logR_son, lambda0, logR = pilot._unpack(x, params)
    local_params = pilot._local_params(params, logR, logMdot)
    residual = _production_residual_base(x, params)
    n = int(params.n_nodes)
    mass_start = 2 * (n - 1) + 2 + 2 + 1
    inner_mdot_residual = float(residual[_inner_mdot_row_index(params)])
    mass_rows = np.asarray(residual[mass_start : mass_start + n - 1], dtype=float)
    interval_R = np.asarray(residual[0 : 2 * (n - 1) : 2], dtype=float)
    interval_E = np.asarray(residual[1 : 2 * (n - 1) : 2], dtype=float)

    R_mid: list[float] = []
    Qwind: list[float] = []
    Qvisc: list[float] = []
    Qadv: list[float] = []
    H_over_R: list[float] = []
    wind_prime: list[float] = []
    source_prime: list[float] = []
    dlogMdot_dx: list[float] = []
    mass_target: list[float] = []
    Mdot_mid: list[float] = []

    for idx in range(n - 1):
        dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
        ym = 0.5 * (y_left + y_right)
        gm = (y_right - y_left) / dx
        logMdot_mid = 0.5 * (logMdot[idx] + logMdot[idx + 1])
        mdot_mid = float(np.exp(logMdot_mid))
        state = algebraic_state(xm, float(ym[0]), float(ym[1]), lambda0, local_params)
        partials = state_partials(xm, ym, lambda0, local_params, eps_x=local_params.partial_eps, eps_y=local_params.partial_eps)
        dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], gm))
        Tdsdx = entropy_gradient_log(xm, ym, gm, lambda0, local_params)
        q_visc = -state.W * dOmega_dx
        q_adv = -(state.Sigma * state.u / state.R) * Tdsdx
        q_stream = stream_heating_rate(xm, local_params)
        q_wind = wind_energy_loss_rate(state, q_visc, q_stream, q_adv, local_params)
        E_w = float(eta_E * wind_energy_per_mass(local_params.M2_g, state.R))
        wprime = float(2.0 * np.pi * state.R**2 * q_wind / max(E_w, 1.0e-300))
        sprime = float(stream_source_prime(xm, local_params))
        target = float((wprime - sprime) / mdot_mid)

        R_mid.append(float(state.R))
        Qwind.append(float(q_wind))
        Qvisc.append(float(q_visc))
        Qadv.append(float(q_adv))
        H_over_R.append(float(state.H_over_R))
        wind_prime.append(wprime)
        source_prime.append(sprime)
        dlogMdot_dx.append(float((logMdot[idx + 1] - logMdot[idx]) / dx))
        mass_target.append(target)
        Mdot_mid.append(mdot_mid)

    R = np.asarray(R_mid, dtype=float)
    logR_mid = np.log(R)
    source = np.asarray(source_prime, dtype=float)
    wind = np.asarray(wind_prime, dtype=float)
    mdot = np.asarray(Mdot_mid, dtype=float)
    mdot_tilde = mdot + _cumtrapz(source, logR_mid)
    s_eff_tilde = wind / np.maximum(mdot_tilde, 1.0e-300)
    raw_mass = mass_rows / max(float(MASS_WEIGHT), 1.0e-300)
    peak_R_idx = int(np.argmax(np.abs(interval_R))) if interval_R.size else 0
    peak_mass_idx = int(np.argmax(np.abs(raw_mass))) if raw_mass.size else 0
    peak_E_idx = int(np.argmax(np.abs(interval_E))) if interval_E.size else 0

    row: dict[str, Any] = {
        "label": label,
        "eta_E": float(eta_E),
        "N": n,
        "R_mid_rg": (R / local_params.r_g).tolist(),
        "interval_R": interval_R.tolist(),
        "interval_E": interval_E.tolist(),
        "local_mass_residual": raw_mass.tolist(),
        "local_mass_residual_weighted": mass_rows.tolist(),
        "local_interval_R_max": float(np.max(np.abs(interval_R))) if interval_R.size else math.nan,
        "local_interval_E_max": float(np.max(np.abs(interval_E))) if interval_E.size else math.nan,
        "peak_interval_R_rg": float(R[peak_R_idx] / local_params.r_g) if interval_R.size else math.nan,
        "peak_interval_R": float(interval_R[peak_R_idx]) if interval_R.size else math.nan,
        "Qwind_Qvisc": (np.asarray(Qwind) / np.maximum(np.abs(np.asarray(Qvisc)), 1.0e-300)).tolist(),
        "Qadv_Qvisc": (np.asarray(Qadv) / np.maximum(np.abs(np.asarray(Qvisc)), 1.0e-300)).tolist(),
        "Mwind_prime_over_Mdot": (wind / np.maximum(mdot, 1.0e-300)).tolist(),
        "Mstream_prime_over_Mdot": (source / np.maximum(mdot, 1.0e-300)).tolist(),
        "dlogMdot_dlogR": np.asarray(dlogMdot_dx, dtype=float).tolist(),
        "mass_target": np.asarray(mass_target, dtype=float).tolist(),
        "Mdot_over_inner": (mdot / local_params.Mdot_g_s).tolist(),
        "Mdot_tilde_over_inner": (mdot_tilde / local_params.Mdot_g_s).tolist(),
        "s_eff_tilde": s_eff_tilde.tolist(),
        "H_over_R": np.asarray(H_over_R, dtype=float).tolist(),
        "peak_mass_residual_rg": float(R[peak_mass_idx] / local_params.r_g) if raw_mass.size else math.nan,
        "peak_mass_residual": float(raw_mass[peak_mass_idx]) if raw_mass.size else math.nan,
        "inner_logMdot_residual": inner_mdot_residual,
        "peak_interval_E_rg": float(R[peak_E_idx] / local_params.r_g) if interval_E.size else math.nan,
        "peak_interval_E": float(interval_E[peak_E_idx]) if interval_E.size else math.nan,
        "mass_residual_p90_abs": float(np.quantile(np.abs(raw_mass), 0.90)) if raw_mass.size else math.nan,
        "s_eff_tilde_p50": float(np.nanmedian(s_eff_tilde)) if s_eff_tilde.size else math.nan,
        "s_eff_tilde_p90": float(np.nanquantile(s_eff_tilde, 0.90)) if s_eff_tilde.size else math.nan,
    }
    if jac_norms:
        row.update(jac_norms)
    if RADIAL_AUDIT_FORMS:
        radial_audit = _radial_residual_representation_audit(x, params, eta_E)
        row["radial_residual_representation_audit"] = radial_audit
        peak_R = radial_audit.get("peak_interval_R_rg") if radial_audit else None
    else:
        peak_R = float(R[int(np.argmax(np.abs(interval_R)))] / local_params.r_g) if interval_R.size else None
    if TRANSITION_GRID_AUDIT or TRANSITION_ALIGN_NODES or RADIAL_AUDIT_FORMS:
        row["transition_grid_audit"] = _transition_grid_audit(x, local_params, peak_R)
    if JACOBIAN_AUDIT:
        peak_idx = int(np.argmax(np.abs(interval_R))) if interval_R.size else 0
        row["local_block_jacobian_audit"] = _local_block_jacobian_audit(x, params, eta_E, peak_idx)
    if SOURCE_BAND_EXTRA_ROWS:
        row.update(_source_band_extra_profile(x, local_params))
    row["source_band_finite_volume_mass"] = bool(SOURCE_BAND_FINITE_VOLUME_MASS)
    row["source_band_extra_audit_only"] = bool(SOURCE_BAND_EXTRA_AUDIT_ONLY)
    return row


def _solve_stage(x0: np.ndarray, params, eta_E: float):
    _set_eta(eta_E)
    lower, upper = pilot._bounds(params)
    x0 = np.clip(x0, lower + 1.0e-12, upper - 1.0e-12)
    from scipy.optimize import least_squares

    kwargs: dict[str, Any] = {
        "bounds": (lower, upper),
        "x_scale": "jac",
        "loss": "linear",
        "ftol": RESIDUAL_TOL,
        "xtol": RESIDUAL_TOL,
        "gtol": RESIDUAL_TOL,
        "max_nfev": MAX_NFEV,
        "verbose": 0,
    }
    if USE_LOCAL_JACOBIAN:
        kwargs["jac"] = lambda trial: _local_finite_difference_jacobian(trial, params)
    elif not (SOURCE_BAND_EXTRA_ROWS and not SOURCE_BAND_EXTRA_AUDIT_ONLY):
        kwargs["jac_sparsity"] = pilot._sparsity(params)
    return least_squares(
        lambda trial: _residual(trial, params),
        x0,
        **kwargs,
    )


def _inner_window_relax(x0: np.ndarray, params, eta_E: float) -> tuple[np.ndarray, dict[str, Any]]:
    if INNER_RELAX_OUTER_RG <= 0.0:
        return x0, {}
    _set_eta(eta_E)
    logu, logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x0, params)
    n = int(params.n_nodes)
    R_rg = np.exp(logR) / params.r_g
    interval_mid_R_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    interval_mask = interval_mid_R_rg <= float(INNER_RELAX_OUTER_RG)
    if not np.any(interval_mask):
        return x0, {"inner_relax_enabled": True, "inner_relax_applied": False}

    last_interval = int(np.max(np.nonzero(interval_mask)[0]))
    last_node = min(n - 1, last_interval + 1)
    node_indices = np.arange(last_node + 1, dtype=int)
    variable_cols: list[int] = []
    variable_cols.extend(int(idx) for idx in node_indices)
    variable_cols.extend(int(n + idx) for idx in node_indices)
    if INNER_RELAX_INCLUDE_MDOT:
        variable_cols.extend(int(2 * n + idx) for idx in node_indices)
    if INNER_RELAX_INCLUDE_GLOBALS:
        variable_cols.extend([3 * n, 3 * n + 1])
    variable_cols_array = np.asarray(sorted(set(variable_cols)), dtype=int)

    interval_rows: list[int] = []
    for idx in range(last_interval + 1):
        interval_rows.extend([2 * idx, 2 * idx + 1])
    sonic_start = 2 * (n - 1) + 2
    inner_mdot_row = _inner_mdot_row_index(params)
    mass_start = inner_mdot_row + 1
    row_indices: list[int] = interval_rows + [sonic_start, sonic_start + 1]
    if INNER_RELAX_INCLUDE_MDOT:
        row_indices.append(inner_mdot_row)
        row_indices.extend(mass_start + idx for idx in range(last_interval + 1))
    row_indices_array = np.asarray(sorted(set(row_indices)), dtype=int)

    lower, upper = pilot._bounds(params)
    x_ref = np.asarray(x0, dtype=float)
    start = x_ref[variable_cols_array].copy()
    lb = lower[variable_cols_array]
    ub = upper[variable_cols_array]
    initial_residual = _residual(x_ref, params)
    initial_selected = float(np.linalg.norm(initial_residual[row_indices_array], ord=np.inf))
    initial_full = float(np.linalg.norm(initial_residual, ord=np.inf))

    def local_residual(trial: np.ndarray) -> np.ndarray:
        full = x_ref.copy()
        full[variable_cols_array] = trial
        rows = _residual(full, params)[row_indices_array]
        if INNER_RELAX_ANCHOR_WEIGHT > 0.0:
            rows = np.concatenate([rows, float(INNER_RELAX_ANCHOR_WEIGHT) * (trial - start)])
        return rows

    from scipy.optimize import least_squares

    result = least_squares(
        local_residual,
        np.clip(start, lb + 1.0e-12, ub - 1.0e-12),
        bounds=(lb, ub),
        x_scale="jac",
        loss="linear",
        ftol=RESIDUAL_TOL,
        xtol=RESIDUAL_TOL,
        gtol=RESIDUAL_TOL,
        max_nfev=INNER_RELAX_MAX_NFEV,
        verbose=0,
    )
    relaxed = x_ref.copy()
    relaxed[variable_cols_array] = result.x
    final_residual = _residual(relaxed, params)
    final_selected = float(np.linalg.norm(final_residual[row_indices_array], ord=np.inf))
    final_full = float(np.linalg.norm(final_residual, ord=np.inf))
    info = {
        "inner_relax_enabled": True,
        "inner_relax_applied": True,
        "inner_relax_outer_rg": float(INNER_RELAX_OUTER_RG),
        "inner_relax_last_node": int(last_node),
        "inner_relax_last_node_rg": float(R_rg[last_node]),
        "inner_relax_last_interval_rg": float(interval_mid_R_rg[last_interval]),
        "inner_relax_n_variables": int(variable_cols_array.size),
        "inner_relax_n_rows": int(row_indices_array.size),
        "inner_relax_initial_selected": initial_selected,
        "inner_relax_final_selected": final_selected,
        "inner_relax_initial_full": initial_full,
        "inner_relax_final_full": final_full,
        "inner_relax_nfev": int(result.nfev),
        "inner_relax_success": bool(result.success),
        "inner_relax_message": str(result.message),
    }
    return relaxed, info


def _outer_band_relax(x0: np.ndarray, params, eta_E: float) -> tuple[np.ndarray, dict[str, Any]]:
    if OUTER_RELAX_MIN_RG <= 0.0 or OUTER_RELAX_MAX_RG <= OUTER_RELAX_MIN_RG:
        return x0, {}
    _set_eta(eta_E)
    _logu, _logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x0, params)
    n = int(params.n_nodes)
    R_rg = np.exp(logR) / params.r_g
    interval_mid_R_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    interval_mask = (interval_mid_R_rg >= float(OUTER_RELAX_MIN_RG)) & (
        interval_mid_R_rg <= float(OUTER_RELAX_MAX_RG)
    )
    if not np.any(interval_mask):
        return x0, {"outer_relax_enabled": True, "outer_relax_applied": False}

    interval_indices = np.nonzero(interval_mask)[0].astype(int)
    node_indices = np.unique(np.concatenate([interval_indices, interval_indices + 1])).astype(int)
    variable_cols: list[int] = []
    variable_cols.extend(int(idx) for idx in node_indices)
    variable_cols.extend(int(n + idx) for idx in node_indices)
    if OUTER_RELAX_INCLUDE_MDOT:
        variable_cols.extend(int(2 * n + idx) for idx in node_indices)
    if OUTER_RELAX_INCLUDE_GLOBALS:
        variable_cols.extend([3 * n, 3 * n + 1])
    variable_cols_array = np.asarray(sorted(set(variable_cols)), dtype=int)

    row_indices: list[int] = []
    for idx in interval_indices:
        row_indices.append(2 * int(idx))
        if OUTER_RELAX_INCLUDE_ENERGY:
            row_indices.append(2 * int(idx) + 1)
    if OUTER_RELAX_INCLUDE_MDOT:
        mass_start = _inner_mdot_row_index(params) + 1
        row_indices.extend(mass_start + int(idx) for idx in interval_indices)
    row_indices_array = np.asarray(sorted(set(row_indices)), dtype=int)

    lower, upper = pilot._bounds(params)
    x_ref = np.asarray(x0, dtype=float)
    start = x_ref[variable_cols_array].copy()
    lb = lower[variable_cols_array]
    ub = upper[variable_cols_array]
    initial_residual = _residual(x_ref, params)
    initial_selected = float(np.linalg.norm(initial_residual[row_indices_array], ord=np.inf))
    initial_full = float(np.linalg.norm(initial_residual, ord=np.inf))

    def local_residual(trial: np.ndarray) -> np.ndarray:
        full = x_ref.copy()
        full[variable_cols_array] = trial
        rows = _residual(full, params)[row_indices_array]
        if OUTER_RELAX_ANCHOR_WEIGHT > 0.0:
            rows = np.concatenate([rows, float(OUTER_RELAX_ANCHOR_WEIGHT) * (trial - start)])
        return rows

    from scipy.optimize import least_squares

    result = least_squares(
        local_residual,
        np.clip(start, lb + 1.0e-12, ub - 1.0e-12),
        bounds=(lb, ub),
        x_scale="jac",
        loss="linear",
        ftol=RESIDUAL_TOL,
        xtol=RESIDUAL_TOL,
        gtol=RESIDUAL_TOL,
        max_nfev=OUTER_RELAX_MAX_NFEV,
        verbose=0,
    )
    relaxed = x_ref.copy()
    relaxed[variable_cols_array] = result.x
    final_residual = _residual(relaxed, params)
    final_selected = float(np.linalg.norm(final_residual[row_indices_array], ord=np.inf))
    final_full = float(np.linalg.norm(final_residual, ord=np.inf))
    info = {
        "outer_relax_enabled": True,
        "outer_relax_applied": True,
        "outer_relax_min_rg": float(OUTER_RELAX_MIN_RG),
        "outer_relax_max_rg": float(OUTER_RELAX_MAX_RG),
        "outer_relax_first_interval_rg": float(interval_mid_R_rg[interval_indices[0]]),
        "outer_relax_last_interval_rg": float(interval_mid_R_rg[interval_indices[-1]]),
        "outer_relax_n_variables": int(variable_cols_array.size),
        "outer_relax_n_rows": int(row_indices_array.size),
        "outer_relax_initial_selected": initial_selected,
        "outer_relax_final_selected": final_selected,
        "outer_relax_initial_full": initial_full,
        "outer_relax_final_full": final_full,
        "outer_relax_nfev": int(result.nfev),
        "outer_relax_success": bool(result.success),
        "outer_relax_message": str(result.message),
        "outer_relax_peak_node_min_rg": float(np.min(R_rg[node_indices])),
        "outer_relax_peak_node_max_rg": float(np.max(R_rg[node_indices])),
    }
    return relaxed, info


def _residual_metrics_for_x(x: np.ndarray, params) -> dict[str, float]:
    residual = np.asarray(_production_residual_base(x, params), dtype=float)
    n = int(params.n_nodes)
    _logu, _logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x, params)
    interval_R = residual[0 : 2 * (n - 1) : 2]
    interval_E = residual[1 : 2 * (n - 1) : 2]
    mass_start = _inner_mdot_row_index(params) + 1
    mass_rows = residual[mass_start : mass_start + n - 1] / max(float(MASS_WEIGHT), 1.0e-300)
    peak_R_idx = int(np.argmax(np.abs(interval_R))) if interval_R.size else 0
    peak_E_idx = int(np.argmax(np.abs(interval_E))) if interval_E.size else 0
    peak_M_idx = int(np.argmax(np.abs(mass_rows))) if mass_rows.size else 0
    interval_mid_R_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    outer_start = 2 * (n - 1)
    return {
        "full": float(np.linalg.norm(residual, ord=np.inf)),
        "interval_R": float(np.max(np.abs(interval_R))) if interval_R.size else math.nan,
        "interval_E": float(np.max(np.abs(interval_E))) if interval_E.size else math.nan,
        "mass": float(np.max(np.abs(mass_rows))) if mass_rows.size else math.nan,
        "outer_omega": float(residual[outer_start]) if residual.size > outer_start else math.nan,
        "outer_energy": float(residual[outer_start + 1]) if residual.size > outer_start + 1 else math.nan,
        "peak_interval_R_index": float(peak_R_idx),
        "peak_interval_R_rg": float(interval_mid_R_rg[peak_R_idx]) if interval_mid_R_rg.size else math.nan,
        "peak_interval_E_index": float(peak_E_idx),
        "peak_interval_E_rg": float(interval_mid_R_rg[peak_E_idx]) if interval_mid_R_rg.size else math.nan,
        "peak_mass_index": float(peak_M_idx),
        "peak_mass_rg": float(interval_mid_R_rg[peak_M_idx]) if interval_mid_R_rg.size else math.nan,
    }


def _fast_block_rows(
    x: np.ndarray,
    params,
    interval_indices: np.ndarray,
    include_outer: bool,
) -> np.ndarray:
    expected = 3 * int(interval_indices.size) + (2 if include_outer else 0)
    try:
        logu, logT, logMdot, logR_son, lambda0, logR = pilot._unpack(x, params)
        if np.any(np.diff(logR) <= 0.0):
            raise ValueError("mapped radius must increase")
        local_params = pilot._local_params(params, logR, logMdot)
        rows: list[float] = []
        for idx_value in interval_indices:
            idx = int(idx_value)
            rows.extend(pilot._interval_residual_from_unpacked(logu, logT, logR, lambda0, local_params, idx))
            dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
            if SOURCE_BAND_FINITE_VOLUME_MASS and _interval_overlaps_source_band(logR, idx, local_params):
                rows.append(
                    _finite_volume_mass_residual_from_unpacked(logu, logT, logMdot, logR, lambda0, local_params, idx)
                )
            else:
                ym = 0.5 * (y_left + y_right)
                gm = (y_right - y_left) / dx
                logMdot_mid = 0.5 * (logMdot[idx] + logMdot[idx + 1])
                mdot_mid = float(np.exp(logMdot_mid))
                dlogMdot_dx = float((logMdot[idx + 1] - logMdot[idx]) / dx)
                source_prime = stream_source_prime(xm, local_params)
                wind_prime = pilot._wind_mass_prime(xm, ym, gm, lambda0, local_params)
                target = float((wind_prime - source_prime) / mdot_mid)
                rows.append(float(MASS_WEIGHT * (dlogMdot_dx - target)))
        if include_outer:
            z = pilot._state_vector(logu, logT, logR_son, lambda0)
            rows.extend(pilot._outer_residual_block(z, local_params))
        return np.asarray(rows, dtype=float)
    except Exception:
        return np.full(expected, 1.0e6, dtype=float)


def _fast_eval_rows(x: np.ndarray, params, row_indices: np.ndarray) -> np.ndarray:
    row_array = np.asarray(row_indices, dtype=int)
    try:
        logu, logT, logMdot, logR_son, lambda0, logR = pilot._unpack(x, params)
        if np.any(np.diff(logR) <= 0.0):
            raise ValueError("mapped radius must increase")
        local_params = pilot._local_params(params, logR, logMdot)
        n = int(params.n_nodes)
        outer_start = 2 * (n - 1)
        sonic_start = outer_start + 2
        inner_mdot_row = _inner_mdot_row_index(params)
        mass_start = inner_mdot_row + 1
        extra_start = 3 * n + 2
        extra_stop = extra_start + _source_band_extra_row_count(params)
        interval_cache: dict[int, np.ndarray] = {}
        outer_cache: np.ndarray | None = None
        sonic_cache: np.ndarray | None = None
        extra_cache: dict[int, float] = {}
        values: list[float] = []
        for row_value in row_array:
            row = int(row_value)
            if 0 <= row < outer_start:
                interval_idx = row // 2
                component = row % 2
                cached = interval_cache.get(interval_idx)
                if cached is None:
                    cached = np.asarray(
                        pilot._interval_residual_from_unpacked(logu, logT, logR, lambda0, local_params, interval_idx),
                        dtype=float,
                    )
                    interval_cache[interval_idx] = cached
                values.append(float(cached[component]))
            elif outer_start <= row < outer_start + 2:
                if outer_cache is None:
                    z = pilot._state_vector(logu, logT, logR_son, lambda0)
                    outer_cache = np.asarray(pilot._outer_residual_block(z, local_params), dtype=float)
                values.append(float(outer_cache[row - outer_start]))
            elif sonic_start <= row < sonic_start + 2:
                if sonic_cache is None:
                    z = pilot._state_vector(logu, logT, logR_son, lambda0)
                    sonic_cache = np.asarray(pilot.sonic_residual_pair(z, local_params, pivot=pilot.PIVOT), dtype=float)
                values.append(float(sonic_cache[row - sonic_start]))
            elif row == inner_mdot_row:
                values.append(float(INNER_MDOT_WEIGHT * (logMdot[0] - np.log(params.Mdot_g_s))))
            elif mass_start <= row < mass_start + n - 1:
                interval_idx = row - mass_start
                if SOURCE_BAND_FINITE_VOLUME_MASS and _interval_overlaps_source_band(logR, interval_idx, local_params):
                    values.append(
                        _finite_volume_mass_residual_from_unpacked(
                            logu, logT, logMdot, logR, lambda0, local_params, interval_idx
                        )
                    )
                else:
                    dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, interval_idx)
                    ym = 0.5 * (y_left + y_right)
                    gm = (y_right - y_left) / dx
                    logMdot_mid = 0.5 * (logMdot[interval_idx] + logMdot[interval_idx + 1])
                    mdot_mid = float(np.exp(logMdot_mid))
                    dlogMdot_dx = float((logMdot[interval_idx + 1] - logMdot[interval_idx]) / dx)
                    source_prime = stream_source_prime(xm, local_params)
                    wind_prime = pilot._wind_mass_prime(xm, ym, gm, lambda0, local_params)
                    target = float((wind_prime - source_prime) / mdot_mid)
                    values.append(float(MASS_WEIGHT * (dlogMdot_dx - target)))
            elif SOURCE_BAND_EXTRA_ROWS and not SOURCE_BAND_EXTRA_AUDIT_ONLY and extra_start <= row < extra_stop:
                extra_row = row - extra_start
                cached_value = extra_cache.get(extra_row)
                if cached_value is None:
                    cached_value, _R_rg, _weight, _component = _source_band_extra_row_from_unpacked(
                        logu, logT, logR, lambda0, local_params, extra_row
                    )
                    extra_cache[extra_row] = cached_value
                values.append(float(cached_value))
            else:
                raise IndexError(f"row {row} is outside residual range")
        return np.asarray(values, dtype=float)
    except Exception:
        return np.full(row_array.size, 1.0e6, dtype=float)


def _jacobian_rows_for_column(col: int, n: int) -> np.ndarray | None:
    outer_start = 2 * (n - 1)
    sonic_start = outer_start + 2
    inner_mdot_row = sonic_start + 2
    mass_start = inner_mdot_row + 1
    extra_start = 3 * n + 2
    if col >= 3 * n:
        return None
    if col < n:
        node = int(col)
        kind = "state"
    elif col < 2 * n:
        node = int(col - n)
        kind = "state"
    else:
        node = int(col - 2 * n)
        kind = "mdot"
    rows: set[int] = set()
    for interval_idx in (node - 1, node):
        if 0 <= interval_idx < n - 1:
            rows.update({2 * interval_idx, 2 * interval_idx + 1, mass_start + interval_idx})
            if SOURCE_BAND_EXTRA_ROWS and not SOURCE_BAND_EXTRA_AUDIT_ONLY:
                rows.update(range(extra_start + 4 * interval_idx, extra_start + 4 * interval_idx + 4))
    if node == 0:
        rows.update({sonic_start, sonic_start + 1})
        if kind == "mdot":
            rows.add(inner_mdot_row)
    if kind == "mdot" and node == 1:
        rows.update({sonic_start, sonic_start + 1})
    if node == n - 1:
        rows.update({outer_start, outer_start + 1})
    if kind == "mdot" and node == n - 2:
        rows.update({outer_start, outer_start + 1})
    return np.asarray(sorted(rows), dtype=int)


def _local_finite_difference_jacobian(x: np.ndarray, params):
    from scipy.sparse import coo_matrix

    x_ref = np.asarray(x, dtype=float)
    n = int(params.n_nodes)
    size = 3 * n + 2
    lower, upper = pilot._bounds(params)
    residual_size = int(_residual(x_ref, params).size)
    all_rows = np.arange(residual_size, dtype=int)
    base_full: np.ndarray | None = None
    row_out: list[int] = []
    col_out: list[int] = []
    data_out: list[float] = []
    base_step = max(abs(float(LOCAL_JACOBIAN_STEP)), 1.0e-12)
    for col in range(size):
        rows = _jacobian_rows_for_column(col, n)
        use_full = rows is None
        if use_full:
            rows = all_rows
        scale = max(abs(float(x_ref[col])), 1.0)
        step = base_step * scale
        can_plus = x_ref[col] + step < upper[col]
        can_minus = x_ref[col] - step > lower[col]
        if not can_plus and not can_minus:
            continue

        def eval_rows(trial_x: np.ndarray) -> np.ndarray:
            if use_full:
                return _residual(trial_x, params)
            return _fast_eval_rows(trial_x, params, rows)

        if can_plus and can_minus:
            plus = x_ref.copy()
            minus = x_ref.copy()
            plus[col] += step
            minus[col] -= step
            deriv = (eval_rows(plus) - eval_rows(minus)) / (2.0 * step)
        else:
            if base_full is None:
                base_full = _residual(x_ref, params)
            base_rows = base_full if use_full else base_full[rows]
            trial = x_ref.copy()
            if can_plus:
                trial[col] += step
                deriv = (eval_rows(trial) - base_rows) / step
            else:
                trial[col] -= step
                deriv = (base_rows - eval_rows(trial)) / step
        finite = np.isfinite(deriv) & (deriv != 0.0)
        if np.any(finite):
            finite_rows = rows[finite]
            row_out.extend(int(row) for row in finite_rows)
            col_out.extend([int(col)] * int(finite_rows.size))
            data_out.extend(float(value) for value in deriv[finite])
    return coo_matrix((data_out, (row_out, col_out)), shape=(residual_size, size)).tocsr()


def _block_guard_pass(initial: dict[str, float], trial: dict[str, float]) -> bool:
    if not BLOCK_ACCEPT_STRICT_GUARDS:
        return True
    if not np.isfinite(trial.get("full", math.inf)):
        return False
    if trial["full"] >= initial["full"]:
        return False
    if trial["interval_E"] > max(1.0e-5, 1.5 * initial["interval_E"]):
        return False
    if trial["mass"] > 3.0e-6:
        return False
    if abs(trial["outer_omega"]) > 2.0e-5:
        return False
    return True


def _coupled_block_correct(x0: np.ndarray, params, eta_E: float) -> tuple[np.ndarray, dict[str, Any]]:
    if not (BLOCK_CORRECT or GRID_HOMOTOPY_BLOCK_CORRECT):
        return x0, {}
    _set_eta(eta_E)
    n = int(params.n_nodes)
    initial_metrics = _residual_metrics_for_x(x0, params)
    selector = BLOCK_PEAK_KIND
    if selector not in {"radial", "energy", "mass", "auto"}:
        selector = "radial"
    if selector == "auto":
        candidates = {
            "radial": float(initial_metrics["interval_R"]),
            "energy": float(initial_metrics["interval_E"]),
            "mass": float(initial_metrics["mass"]),
        }
        selector = max(candidates, key=lambda key: abs(candidates[key]))
    if selector == "mass":
        peak_idx = int(initial_metrics["peak_mass_index"])
    elif selector == "energy":
        peak_idx = int(initial_metrics["peak_interval_E_index"])
    else:
        peak_idx = int(initial_metrics["peak_interval_R_index"])
    half_width = max(1, int(BLOCK_HALF_WIDTH))
    first_interval = max(0, peak_idx - half_width)
    last_interval = min(n - 2, peak_idx + half_width)
    interval_indices = np.arange(first_interval, last_interval + 1, dtype=int)
    node_indices = np.arange(first_interval, last_interval + 2, dtype=int)

    mass_start = _inner_mdot_row_index(params) + 1
    row_indices: list[int] = []
    row_kinds: list[str] = []
    for idx in interval_indices:
        row_indices.extend([2 * int(idx), 2 * int(idx) + 1, mass_start + int(idx)])
        row_kinds.extend(["radial", "energy", "mass"])
    if BLOCK_INCLUDE_OUTER:
        outer_start = 2 * (n - 1)
        row_indices.extend([outer_start, outer_start + 1])
        row_kinds.extend(["outer_omega", "outer_energy"])
    row_array = np.asarray(row_indices, dtype=int)

    variable_cols: list[int] = []
    variable_kinds: list[str] = []
    for idx in node_indices:
        variable_cols.append(int(idx))
        variable_kinds.append("logu")
    for idx in node_indices:
        variable_cols.append(int(n + idx))
        variable_kinds.append("logT")
    for idx in node_indices:
        variable_cols.append(int(2 * n + idx))
        variable_kinds.append("logMdot")
    if BLOCK_INCLUDE_GLOBALS:
        variable_cols.extend([3 * n, 3 * n + 1])
        variable_kinds.extend(["logR_son", "lambda0"])
    variable_cols_array = np.asarray(variable_cols, dtype=int)

    lower, upper = pilot._bounds(params)
    x_ref = np.asarray(x0, dtype=float)
    start = x_ref[variable_cols_array].copy()
    lb = lower[variable_cols_array]
    ub = upper[variable_cols_array]
    edge_nodes = {int(node_indices[0]), int(node_indices[-1])}
    edge_cols: set[int] = set()
    for idx in edge_nodes:
        edge_cols.update({idx, n + idx, 2 * n + idx})
    edge_anchor_positions = np.asarray(
        [pos for pos, col in enumerate(variable_cols_array) if int(col) in edge_cols],
        dtype=int,
    )

    if BLOCK_FAST_LOCAL_RESIDUAL:
        initial_rows = _fast_block_rows(x_ref, params, interval_indices, BLOCK_INCLUDE_OUTER)
    else:
        initial_rows = np.asarray(pilot.residual(x_ref, params), dtype=float)[row_array]
    initial_selected = float(np.linalg.norm(initial_rows, ord=np.inf))

    def local_residual(trial: np.ndarray) -> np.ndarray:
        full = x_ref.copy()
        full[variable_cols_array] = trial
        if BLOCK_FAST_LOCAL_RESIDUAL:
            rows = _fast_block_rows(full, params, interval_indices, BLOCK_INCLUDE_OUTER)
        else:
            rows = np.asarray(pilot.residual(full, params), dtype=float)[row_array]
        anchors: list[np.ndarray] = []
        if BLOCK_EDGE_ANCHOR_WEIGHT > 0.0 and edge_anchor_positions.size:
            anchors.append(float(BLOCK_EDGE_ANCHOR_WEIGHT) * (trial[edge_anchor_positions] - start[edge_anchor_positions]))
        if BLOCK_ALL_ANCHOR_WEIGHT > 0.0:
            anchors.append(float(BLOCK_ALL_ANCHOR_WEIGHT) * (trial - start))
        if anchors:
            rows = np.concatenate([rows, *anchors])
        return rows

    from scipy.optimize import least_squares

    result = least_squares(
        local_residual,
        np.clip(start, lb + 1.0e-12, ub - 1.0e-12),
        bounds=(lb, ub),
        x_scale="jac",
        loss="linear",
        ftol=RESIDUAL_TOL,
        xtol=RESIDUAL_TOL,
        gtol=RESIDUAL_TOL,
        max_nfev=BLOCK_MAX_NFEV,
        verbose=0,
    )

    candidate = x_ref.copy()
    candidate[variable_cols_array] = result.x
    if BLOCK_FAST_LOCAL_RESIDUAL:
        candidate_rows = _fast_block_rows(candidate, params, interval_indices, BLOCK_INCLUDE_OUTER)
    else:
        candidate_rows = np.asarray(pilot.residual(candidate, params), dtype=float)[row_array]
    candidate_metrics = _residual_metrics_for_x(candidate, params)
    candidate_selected = float(np.linalg.norm(candidate_rows, ord=np.inf))

    best_x = x_ref
    best_metrics = initial_metrics
    best_alpha = 0.0
    trials: list[dict[str, Any]] = []
    step_delta = candidate - x_ref
    for exponent in range(max(1, int(BLOCK_LINE_SEARCH_STEPS))):
        alpha = 0.5**exponent
        trial_x = np.clip(x_ref + alpha * step_delta, lower + 1.0e-12, upper - 1.0e-12)
        metrics = _residual_metrics_for_x(trial_x, params)
        guard = _block_guard_pass(initial_metrics, metrics)
        trials.append(
            {
                "alpha": float(alpha),
                "full": metrics["full"],
                "interval_R": metrics["interval_R"],
                "interval_E": metrics["interval_E"],
                "mass": metrics["mass"],
                "outer_omega": metrics["outer_omega"],
                "guard_pass": bool(guard),
            }
        )
        if guard and metrics["full"] < best_metrics["full"]:
            best_x = trial_x
            best_metrics = metrics
            best_alpha = float(alpha)

    applied = bool(best_alpha > 0.0)
    _logu, _logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x_ref, params)
    interval_mid_R_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    info = {
        "block_correct_enabled": True,
        "block_correct_applied": applied,
        "block_correct_peak_kind": str(selector),
        "block_correct_half_width": int(half_width),
        "block_correct_first_interval": int(first_interval),
        "block_correct_last_interval": int(last_interval),
        "block_correct_first_interval_R_rg": float(interval_mid_R_rg[first_interval]),
        "block_correct_last_interval_R_rg": float(interval_mid_R_rg[last_interval]),
        "block_correct_peak_interval_R_rg": float(initial_metrics["peak_interval_R_rg"]),
        "block_correct_n_variables": int(variable_cols_array.size),
        "block_correct_n_rows": int(row_array.size),
        "block_correct_edge_anchor_weight": float(BLOCK_EDGE_ANCHOR_WEIGHT),
        "block_correct_all_anchor_weight": float(BLOCK_ALL_ANCHOR_WEIGHT),
        "block_correct_include_outer": bool(BLOCK_INCLUDE_OUTER),
        "block_correct_include_globals": bool(BLOCK_INCLUDE_GLOBALS),
        "block_correct_fast_local_residual": bool(BLOCK_FAST_LOCAL_RESIDUAL),
        "block_correct_initial_selected": initial_selected,
        "block_correct_candidate_selected": candidate_selected,
        "block_correct_initial_full": initial_metrics["full"],
        "block_correct_candidate_full": candidate_metrics["full"],
        "block_correct_final_full": best_metrics["full"],
        "block_correct_initial_interval_R": initial_metrics["interval_R"],
        "block_correct_final_interval_R": best_metrics["interval_R"],
        "block_correct_initial_interval_E": initial_metrics["interval_E"],
        "block_correct_final_interval_E": best_metrics["interval_E"],
        "block_correct_initial_mass": initial_metrics["mass"],
        "block_correct_final_mass": best_metrics["mass"],
        "block_correct_initial_outer_omega": initial_metrics["outer_omega"],
        "block_correct_final_outer_omega": best_metrics["outer_omega"],
        "block_correct_alpha": best_alpha,
        "block_correct_nfev": int(result.nfev),
        "block_correct_success": bool(result.success),
        "block_correct_message": str(result.message),
        "block_correct_row_indices": row_array.tolist(),
        "block_correct_row_kinds": row_kinds,
        "block_correct_variable_cols": variable_cols_array.tolist(),
        "block_correct_variable_kinds": variable_kinds,
        "block_correct_trials": trials,
    }
    return best_x, info


def _band_guard_pass(initial: dict[str, float], trial: dict[str, float]) -> bool:
    if not BAND_ACCEPT_STRICT_GUARDS:
        return bool(np.isfinite(trial.get("full", math.inf)))
    if not np.isfinite(trial.get("full", math.inf)):
        return False
    if trial["full"] >= initial["full"]:
        return False
    if trial["interval_E"] > max(1.0e-5, 1.5 * initial["interval_E"]):
        return False
    if trial["mass"] > max(1.0e-5, 1.5 * initial["mass"]):
        return False
    if abs(trial["outer_omega"]) > 2.0e-5:
        return False
    return True


def _reduced_band_sparsity(
    variable_cols: np.ndarray,
    interval_indices: np.ndarray,
    n_nodes: int,
    include_globals: bool,
    edge_anchor_positions: np.ndarray,
    all_anchor_weight: float,
):
    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None
    n_interval_rows = 3 * int(interval_indices.size)
    n_anchor_rows = int(edge_anchor_positions.size) + (int(variable_cols.size) if all_anchor_weight > 0.0 else 0)
    pattern = lil_matrix((n_interval_rows + n_anchor_rows, int(variable_cols.size)), dtype=int)
    col_to_local = {int(col): int(pos) for pos, col in enumerate(variable_cols)}
    global_cols = [3 * int(n_nodes), 3 * int(n_nodes) + 1] if include_globals else []
    for local_interval_idx, interval_idx_value in enumerate(interval_indices):
        interval_idx = int(interval_idx_value)
        full_cols = [
            interval_idx,
            interval_idx + 1,
            n_nodes + interval_idx,
            n_nodes + interval_idx + 1,
            2 * n_nodes + interval_idx,
            2 * n_nodes + interval_idx + 1,
            *global_cols,
        ]
        for row in range(3 * local_interval_idx, 3 * local_interval_idx + 3):
            for full_col in full_cols:
                local_col = col_to_local.get(int(full_col))
                if local_col is not None:
                    pattern[row, local_col] = 1
    row = n_interval_rows
    for pos in edge_anchor_positions:
        pattern[row, int(pos)] = 1
        row += 1
    if all_anchor_weight > 0.0:
        for pos in range(int(variable_cols.size)):
            pattern[row, pos] = 1
            row += 1
    return pattern.tocsr()


def _reduced_band_correct(x0: np.ndarray, params, eta_E: float) -> tuple[np.ndarray, dict[str, Any]]:
    if not BAND_CORRECT:
        return x0, {}
    if BAND_MIN_RG <= 0.0 or BAND_MAX_RG <= BAND_MIN_RG:
        return x0, {"band_correct_enabled": True, "band_correct_applied": False, "band_correct_reason": "invalid band"}
    _set_eta(eta_E)
    n = int(params.n_nodes)
    x_ref = np.asarray(x0, dtype=float)
    _logu, _logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x_ref, params)
    interval_mid_R_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    interval_mask = (interval_mid_R_rg >= float(BAND_MIN_RG)) & (interval_mid_R_rg <= float(BAND_MAX_RG))
    if not np.any(interval_mask):
        return x0, {
            "band_correct_enabled": True,
            "band_correct_applied": False,
            "band_correct_reason": "no intervals in band",
        }
    interval_indices = np.nonzero(interval_mask)[0].astype(int)
    first_interval = int(interval_indices[0])
    last_interval = int(interval_indices[-1])
    node_indices = np.arange(first_interval, last_interval + 2, dtype=int)

    variable_cols: list[int] = []
    variable_kinds: list[str] = []
    for idx in node_indices:
        variable_cols.append(int(idx))
        variable_kinds.append("logu")
    for idx in node_indices:
        variable_cols.append(int(n + idx))
        variable_kinds.append("logT")
    for idx in node_indices:
        variable_cols.append(int(2 * n + idx))
        variable_kinds.append("logMdot")
    if BAND_INCLUDE_GLOBALS:
        variable_cols.extend([3 * n, 3 * n + 1])
        variable_kinds.extend(["logR_son", "lambda0"])
    variable_cols_array = np.asarray(variable_cols, dtype=int)

    lower, upper = pilot._bounds(params)
    start = x_ref[variable_cols_array].copy()
    lb = lower[variable_cols_array]
    ub = upper[variable_cols_array]
    edge_nodes = {int(node_indices[0]), int(node_indices[-1])}
    edge_cols: set[int] = set()
    for idx in edge_nodes:
        edge_cols.update({idx, n + idx, 2 * n + idx})
    edge_anchor_positions = np.asarray(
        [pos for pos, col in enumerate(variable_cols_array) if int(col) in edge_cols],
        dtype=int,
    )

    initial_metrics = _residual_metrics_for_x(x_ref, params)
    initial_rows = _fast_block_rows(x_ref, params, interval_indices, False)
    initial_selected = float(np.linalg.norm(initial_rows, ord=np.inf))

    def local_residual(trial: np.ndarray) -> np.ndarray:
        full = x_ref.copy()
        full[variable_cols_array] = trial
        rows = _fast_block_rows(full, params, interval_indices, False)
        anchors: list[np.ndarray] = []
        if BAND_EDGE_ANCHOR_WEIGHT > 0.0 and edge_anchor_positions.size:
            anchors.append(float(BAND_EDGE_ANCHOR_WEIGHT) * (trial[edge_anchor_positions] - start[edge_anchor_positions]))
        if BAND_ALL_ANCHOR_WEIGHT > 0.0:
            anchors.append(float(BAND_ALL_ANCHOR_WEIGHT) * (trial - start))
        if anchors:
            rows = np.concatenate([rows, *anchors])
        return rows

    sparsity = _reduced_band_sparsity(
        variable_cols_array,
        interval_indices,
        n,
        BAND_INCLUDE_GLOBALS,
        edge_anchor_positions if BAND_EDGE_ANCHOR_WEIGHT > 0.0 else np.asarray([], dtype=int),
        BAND_ALL_ANCHOR_WEIGHT,
    )

    from scipy.optimize import least_squares

    result = least_squares(
        local_residual,
        np.clip(start, lb + 1.0e-12, ub - 1.0e-12),
        bounds=(lb, ub),
        jac_sparsity=sparsity,
        x_scale="jac",
        loss="linear",
        ftol=RESIDUAL_TOL,
        xtol=RESIDUAL_TOL,
        gtol=RESIDUAL_TOL,
        max_nfev=BAND_MAX_NFEV,
        verbose=0,
    )

    candidate = x_ref.copy()
    candidate[variable_cols_array] = result.x
    candidate_rows = _fast_block_rows(candidate, params, interval_indices, False)
    candidate_metrics = _residual_metrics_for_x(candidate, params)
    candidate_selected = float(np.linalg.norm(candidate_rows, ord=np.inf))

    best_x = x_ref
    best_metrics = initial_metrics
    best_alpha = 0.0
    trials: list[dict[str, Any]] = []
    step_delta = candidate - x_ref
    for exponent in range(max(1, int(BAND_LINE_SEARCH_STEPS))):
        alpha = 0.5**exponent
        trial_x = np.clip(x_ref + alpha * step_delta, lower + 1.0e-12, upper - 1.0e-12)
        metrics = _residual_metrics_for_x(trial_x, params)
        guard = _band_guard_pass(initial_metrics, metrics)
        trials.append(
            {
                "alpha": float(alpha),
                "full": metrics["full"],
                "interval_R": metrics["interval_R"],
                "interval_E": metrics["interval_E"],
                "mass": metrics["mass"],
                "outer_omega": metrics["outer_omega"],
                "guard_pass": bool(guard),
            }
        )
        if guard and metrics["full"] < best_metrics["full"]:
            best_x = trial_x
            best_metrics = metrics
            best_alpha = float(alpha)

    return best_x, {
        "band_correct_enabled": True,
        "band_correct_applied": bool(best_alpha > 0.0),
        "band_correct_min_rg": float(BAND_MIN_RG),
        "band_correct_max_rg": float(BAND_MAX_RG),
        "band_correct_first_interval": first_interval,
        "band_correct_last_interval": last_interval,
        "band_correct_first_interval_R_rg": float(interval_mid_R_rg[first_interval]),
        "band_correct_last_interval_R_rg": float(interval_mid_R_rg[last_interval]),
        "band_correct_n_intervals": int(interval_indices.size),
        "band_correct_n_variables": int(variable_cols_array.size),
        "band_correct_n_rows": int(3 * interval_indices.size),
        "band_correct_edge_anchor_weight": float(BAND_EDGE_ANCHOR_WEIGHT),
        "band_correct_all_anchor_weight": float(BAND_ALL_ANCHOR_WEIGHT),
        "band_correct_include_globals": bool(BAND_INCLUDE_GLOBALS),
        "band_correct_initial_selected": initial_selected,
        "band_correct_candidate_selected": candidate_selected,
        "band_correct_initial_full": initial_metrics["full"],
        "band_correct_candidate_full": candidate_metrics["full"],
        "band_correct_final_full": best_metrics["full"],
        "band_correct_initial_interval_R": initial_metrics["interval_R"],
        "band_correct_final_interval_R": best_metrics["interval_R"],
        "band_correct_initial_interval_E": initial_metrics["interval_E"],
        "band_correct_final_interval_E": best_metrics["interval_E"],
        "band_correct_initial_mass": initial_metrics["mass"],
        "band_correct_final_mass": best_metrics["mass"],
        "band_correct_initial_outer_omega": initial_metrics["outer_omega"],
        "band_correct_final_outer_omega": best_metrics["outer_omega"],
        "band_correct_alpha": best_alpha,
        "band_correct_nfev": int(result.nfev),
        "band_correct_success": bool(result.success),
        "band_correct_message": str(result.message),
        "band_correct_variable_kinds": variable_kinds,
        "band_correct_trials": trials,
    }


def _source_band_interval_indices(x: np.ndarray, params) -> tuple[np.ndarray, np.ndarray]:
    _logu, _logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x, params)
    local_params = params
    mask = np.asarray([_interval_overlaps_source_band(logR, idx, local_params) for idx in range(int(params.n_nodes) - 1)])
    if not np.any(mask):
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    intervals = np.nonzero(mask)[0].astype(int)
    nodes = np.arange(int(intervals[0]), int(intervals[-1]) + 2, dtype=int)
    return intervals, nodes


def _hermite_simpson_source_interval_rows(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    local_params,
    idx: int,
) -> np.ndarray:
    dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
    F_left = _ode_slope(float(logR[idx]), y_left, lambda0, local_params)
    F_right = _ode_slope(float(logR[idx + 1]), y_right, lambda0, local_params)
    if not (np.all(np.isfinite(F_left)) and np.all(np.isfinite(F_right))):
        F_left = F_right = (y_right - y_left) / dx
    y_mid, _used_hermite_midpoint = _bounded_hermite_midpoint(y_left, y_right, F_left, F_right, dx, local_params)
    F_mid = _ode_slope(xm, y_mid, lambda0, local_params)
    if not np.all(np.isfinite(F_mid)):
        F_mid = (y_right - y_left) / dx
    state_rows = SOURCE_MICRO_STATE_WEIGHT * (y_right - y_left - (dx / 6.0) * (F_left + 4.0 * F_mid + F_right)) / max(dx, 1.0e-12)
    mass_row = _finite_volume_mass_residual_from_unpacked(logu, logT, logMdot, logR, lambda0, local_params, idx)
    return np.asarray([state_rows[0], state_rows[1], mass_row], dtype=float)


def _source_micro_residual_rows(x: np.ndarray, params, interval_indices: np.ndarray) -> np.ndarray:
    try:
        logu, logT, logMdot, _logR_son, lambda0, logR = pilot._unpack(x, params)
        local_params = pilot._local_params(params, logR, logMdot)
        rows: list[float] = []
        for idx_value in interval_indices:
            rows.extend(
                _hermite_simpson_source_interval_rows(
                    logu, logT, logMdot, logR, lambda0, local_params, int(idx_value)
                )
            )
        return np.asarray(rows, dtype=float)
    except Exception:
        return np.full(3 * int(interval_indices.size), 1.0e6, dtype=float)


def _source_micro_sparsity(
    variable_cols: np.ndarray,
    interval_indices: np.ndarray,
    n_nodes: int,
    include_globals: bool,
    anchor_positions: np.ndarray,
    all_anchor_weight: float,
):
    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None
    n_interval_rows = 3 * int(interval_indices.size)
    n_anchor_rows = int(anchor_positions.size) + (int(variable_cols.size) if all_anchor_weight > 0.0 else 0)
    pattern = lil_matrix((n_interval_rows + n_anchor_rows, int(variable_cols.size)), dtype=int)
    col_to_local = {int(col): int(pos) for pos, col in enumerate(variable_cols)}
    global_cols = [3 * int(n_nodes), 3 * int(n_nodes) + 1] if include_globals else []
    for local_interval_idx, interval_idx_value in enumerate(interval_indices):
        interval_idx = int(interval_idx_value)
        neighbor_nodes = range(max(0, interval_idx - 1), min(n_nodes - 1, interval_idx + 2) + 1)
        full_cols: list[int] = []
        for node in neighbor_nodes:
            full_cols.extend([node, n_nodes + node, 2 * n_nodes + node])
        full_cols.extend(global_cols)
        for row in range(3 * local_interval_idx, 3 * local_interval_idx + 3):
            for full_col in full_cols:
                local_col = col_to_local.get(int(full_col))
                if local_col is not None:
                    pattern[row, local_col] = 1
    row = n_interval_rows
    for pos in anchor_positions:
        pattern[row, int(pos)] = 1
        row += 1
    if all_anchor_weight > 0.0:
        for pos in range(int(variable_cols.size)):
            pattern[row, pos] = 1
            row += 1
    return pattern.tocsr()


def _source_extra_max_for_x(x: np.ndarray, params) -> float:
    try:
        logu, logT, logMdot, _logR_son, _lambda0, logR = pilot._unpack(x, params)
        local_params = pilot._local_params(params, logR, logMdot)
        return float(_source_band_extra_profile(x, local_params).get("source_band_extra_max", math.nan))
    except Exception:
        return math.inf


def _source_microdomain_correct(x0: np.ndarray, params, eta_E: float) -> tuple[np.ndarray, dict[str, Any]]:
    if not SOURCE_MICRO_LOCAL_CORRECT:
        return x0, {"source_micro_correct_enabled": False}
    _set_eta(eta_E)
    n = int(params.n_nodes)
    x_ref = np.asarray(x0, dtype=float)
    interval_indices, node_indices = _source_band_interval_indices(x_ref, params)
    if interval_indices.size == 0 or node_indices.size < 3:
        return x0, {
            "source_micro_correct_enabled": True,
            "source_micro_correct_applied": False,
            "source_micro_correct_reason": "no source-band intervals",
        }
    active_nodes = node_indices[1:-1] if SOURCE_MICRO_FREEZE_EDGES else node_indices
    variable_cols: list[int] = []
    for idx in active_nodes:
        variable_cols.append(int(idx))
    for idx in active_nodes:
        variable_cols.append(int(n + idx))
    for idx in active_nodes:
        variable_cols.append(int(2 * n + idx))
    if SOURCE_MICRO_INCLUDE_GLOBALS:
        variable_cols.extend([3 * n, 3 * n + 1])
    variable_cols_array = np.asarray(variable_cols, dtype=int)
    if variable_cols_array.size == 0:
        return x0, {
            "source_micro_correct_enabled": True,
            "source_micro_correct_applied": False,
            "source_micro_correct_reason": "no variable columns",
        }
    lower, upper = pilot._bounds(params)
    start = x_ref[variable_cols_array].copy()
    lb = lower[variable_cols_array]
    ub = upper[variable_cols_array]
    edge_nodes = {int(node_indices[0]), int(node_indices[-1])}
    edge_cols: set[int] = set()
    if not SOURCE_MICRO_FREEZE_EDGES:
        for idx in edge_nodes:
            edge_cols.update({idx, n + idx, 2 * n + idx})
    edge_anchor_positions = np.asarray(
        [pos for pos, col in enumerate(variable_cols_array) if int(col) in edge_cols],
        dtype=int,
    )

    initial_rows = _source_micro_residual_rows(x_ref, params, interval_indices)
    initial_metrics = _residual_metrics_for_x(x_ref, params)
    initial_extra = _source_extra_max_for_x(x_ref, params)
    initial_score = float(max(initial_metrics["full"], initial_extra if np.isfinite(initial_extra) else 0.0))

    def local_residual(trial: np.ndarray) -> np.ndarray:
        full = x_ref.copy()
        full[variable_cols_array] = trial
        rows = _source_micro_residual_rows(full, params, interval_indices)
        anchors: list[np.ndarray] = []
        if SOURCE_MICRO_EDGE_ANCHOR_WEIGHT > 0.0 and edge_anchor_positions.size:
            anchors.append(float(SOURCE_MICRO_EDGE_ANCHOR_WEIGHT) * (trial[edge_anchor_positions] - start[edge_anchor_positions]))
        if SOURCE_MICRO_ALL_ANCHOR_WEIGHT > 0.0:
            anchors.append(float(SOURCE_MICRO_ALL_ANCHOR_WEIGHT) * (trial - start))
        if anchors:
            rows = np.concatenate([rows, *anchors])
        return rows

    sparsity = _source_micro_sparsity(
        variable_cols_array,
        interval_indices,
        n,
        SOURCE_MICRO_INCLUDE_GLOBALS,
        edge_anchor_positions if SOURCE_MICRO_EDGE_ANCHOR_WEIGHT > 0.0 else np.asarray([], dtype=int),
        SOURCE_MICRO_ALL_ANCHOR_WEIGHT,
    )

    from scipy.optimize import least_squares

    result = least_squares(
        local_residual,
        np.clip(start, lb + 1.0e-12, ub - 1.0e-12),
        bounds=(lb, ub),
        jac_sparsity=sparsity,
        x_scale="jac",
        loss="linear",
        ftol=RESIDUAL_TOL,
        xtol=RESIDUAL_TOL,
        gtol=RESIDUAL_TOL,
        max_nfev=SOURCE_MICRO_MAX_NFEV,
        verbose=0,
    )

    candidate = x_ref.copy()
    candidate[variable_cols_array] = result.x
    lower_full, upper_full = pilot._bounds(params)
    step_delta = candidate - x_ref
    best_x = x_ref
    best_metrics = initial_metrics
    best_extra = initial_extra
    best_score = initial_score
    best_alpha = 0.0
    trials: list[dict[str, Any]] = []
    for exponent in range(12):
        alpha = 0.5**exponent
        trial_x = np.clip(x_ref + alpha * step_delta, lower_full + 1.0e-12, upper_full - 1.0e-12)
        metrics = _residual_metrics_for_x(trial_x, params)
        extra = _source_extra_max_for_x(trial_x, params)
        score = float(max(metrics["full"], extra if np.isfinite(extra) else 0.0))
        guard = bool(np.isfinite(score) and metrics["mass"] <= max(3.0e-6, 3.0 * initial_metrics["mass"], 1.0e-12))
        trials.append(
            {
                "alpha": float(alpha),
                "score": score,
                "full": metrics["full"],
                "mass": metrics["mass"],
                "source_band_extra": extra,
                "guard_pass": guard,
            }
        )
        if guard and score < best_score:
            best_x = trial_x
            best_metrics = metrics
            best_extra = extra
            best_score = score
            best_alpha = float(alpha)

    final_rows = _source_micro_residual_rows(best_x, params, interval_indices)
    return best_x, {
        "source_micro_correct_enabled": True,
        "source_micro_correct_applied": bool(best_alpha > 0.0),
        "source_micro_correct_freeze_edges": bool(SOURCE_MICRO_FREEZE_EDGES),
        "source_micro_correct_include_globals": bool(SOURCE_MICRO_INCLUDE_GLOBALS),
        "source_micro_correct_n_intervals": int(interval_indices.size),
        "source_micro_correct_n_nodes": int(node_indices.size),
        "source_micro_correct_n_variables": int(variable_cols_array.size),
        "source_micro_correct_initial_selected": float(np.linalg.norm(initial_rows, ord=np.inf)) if initial_rows.size else math.nan,
        "source_micro_correct_final_selected": float(np.linalg.norm(final_rows, ord=np.inf)) if final_rows.size else math.nan,
        "source_micro_correct_initial_score": initial_score,
        "source_micro_correct_final_score": best_score,
        "source_micro_correct_initial_full": initial_metrics["full"],
        "source_micro_correct_final_full": best_metrics["full"],
        "source_micro_correct_initial_mass": initial_metrics["mass"],
        "source_micro_correct_final_mass": best_metrics["mass"],
        "source_micro_correct_initial_extra": initial_extra,
        "source_micro_correct_final_extra": best_extra,
        "source_micro_correct_alpha": best_alpha,
        "source_micro_correct_nfev": int(result.nfev),
        "source_micro_correct_success": bool(result.success),
        "source_micro_correct_message": str(result.message),
        "source_micro_correct_trials": trials,
    }


def _source_domain_sample_fractions() -> np.ndarray:
    values: list[float] = []
    for piece in SOURCE_DOMAIN_FRACTIONS_RAW.split(","):
        text = piece.strip()
        if not text:
            continue
        value = float(text)
        if not 0.0 < value < 1.0:
            raise ValueError("source-domain sample fractions must lie strictly between 0 and 1")
        values.append(value)
    if not values:
        values = [0.5]
    return np.asarray(sorted(set(float(value) for value in values)), dtype=float)


def _source_domain_interval_rows(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    local_params,
    idx: int,
    fractions: np.ndarray,
) -> np.ndarray:
    dx, y_left, y_right, _xm = _interval_geometry(logu, logT, logR, idx)
    if dx <= 0.0:
        return np.full(2 * int(fractions.size) + 1, 1.0e6, dtype=float)
    g = (y_right - y_left) / dx
    rows: list[float] = []
    for frac in fractions:
        t = float(frac)
        xq = float(logR[idx] + t * dx)
        yq = (1.0 - t) * y_left + t * y_right
        rows.extend(_scaled_residual_at(xq, yq, g, lambda0, local_params))
    rows.append(_finite_volume_mass_residual_from_unpacked(logu, logT, logMdot, logR, lambda0, local_params, idx))
    return np.asarray(rows, dtype=float)


def _source_domain_residual_rows(x: np.ndarray, params, interval_indices: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    expected = (2 * int(fractions.size) + 1) * int(interval_indices.size)
    try:
        logu, logT, logMdot, _logR_son, lambda0, logR = pilot._unpack(x, params)
        local_params = pilot._local_params(params, logR, logMdot)
        rows: list[float] = []
        for idx_value in interval_indices:
            rows.extend(
                _source_domain_interval_rows(
                    logu, logT, logMdot, logR, lambda0, local_params, int(idx_value), fractions
                )
            )
        return np.asarray(rows, dtype=float)
    except Exception:
        return np.full(expected, 1.0e6, dtype=float)


def _source_domain_sparsity(
    variable_cols: np.ndarray,
    interval_indices: np.ndarray,
    n_nodes: int,
    fractions: np.ndarray,
    include_globals: bool,
    anchor_positions: np.ndarray,
    all_anchor_weight: float,
):
    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None
    rows_per_interval = 2 * int(fractions.size) + 1
    n_interval_rows = rows_per_interval * int(interval_indices.size)
    n_anchor_rows = int(anchor_positions.size) + (int(variable_cols.size) if all_anchor_weight > 0.0 else 0)
    pattern = lil_matrix((n_interval_rows + n_anchor_rows, int(variable_cols.size)), dtype=int)
    col_to_local = {int(col): int(pos) for pos, col in enumerate(variable_cols)}
    global_cols = [3 * int(n_nodes), 3 * int(n_nodes) + 1] if include_globals else []
    for local_interval_idx, interval_idx_value in enumerate(interval_indices):
        interval_idx = int(interval_idx_value)
        full_cols = [
            interval_idx,
            interval_idx + 1,
            n_nodes + interval_idx,
            n_nodes + interval_idx + 1,
            2 * n_nodes + interval_idx,
            2 * n_nodes + interval_idx + 1,
            *global_cols,
        ]
        row_start = rows_per_interval * local_interval_idx
        for row in range(row_start, row_start + rows_per_interval):
            for full_col in full_cols:
                local_col = col_to_local.get(int(full_col))
                if local_col is not None:
                    pattern[row, local_col] = 1
    row = n_interval_rows
    for pos in anchor_positions:
        pattern[row, int(pos)] = 1
        row += 1
    if all_anchor_weight > 0.0:
        for pos in range(int(variable_cols.size)):
            pattern[row, pos] = 1
            row += 1
    return pattern.tocsr()


def _source_domain_correct(x0: np.ndarray, params, eta_E: float) -> tuple[np.ndarray, dict[str, Any]]:
    if not SOURCE_DOMAIN_CORRECT:
        return x0, {}
    _set_eta(eta_E)
    n = int(params.n_nodes)
    x_ref = np.asarray(x0, dtype=float)
    fractions = _source_domain_sample_fractions()
    interval_indices, node_indices = _source_band_interval_indices(x_ref, params)
    if interval_indices.size == 0 or node_indices.size < 2:
        return x0, {
            "source_domain_correct_enabled": True,
            "source_domain_correct_applied": False,
            "source_domain_correct_reason": "no source-domain intervals",
        }
    halo = max(0, int(SOURCE_DOMAIN_HALO_INTERVALS))
    if halo > 0:
        first = max(0, int(interval_indices[0]) - halo)
        last = min(n - 2, int(interval_indices[-1]) + halo)
        interval_indices = np.arange(first, last + 1, dtype=int)
        node_indices = np.arange(first, last + 2, dtype=int)
    active_nodes = node_indices[1:-1] if SOURCE_DOMAIN_FREEZE_EDGES and node_indices.size > 2 else node_indices
    variable_cols: list[int] = []
    variable_kinds: list[str] = []
    for idx in active_nodes:
        variable_cols.append(int(idx))
        variable_kinds.append("logu")
    for idx in active_nodes:
        variable_cols.append(int(n + idx))
        variable_kinds.append("logT")
    for idx in active_nodes:
        variable_cols.append(int(2 * n + idx))
        variable_kinds.append("logMdot")
    if SOURCE_DOMAIN_INCLUDE_GLOBALS:
        variable_cols.extend([3 * n, 3 * n + 1])
        variable_kinds.extend(["logR_son", "lambda0"])
    variable_cols_array = np.asarray(variable_cols, dtype=int)
    if variable_cols_array.size == 0:
        return x0, {
            "source_domain_correct_enabled": True,
            "source_domain_correct_applied": False,
            "source_domain_correct_reason": "no variable columns",
        }

    lower, upper = pilot._bounds(params)
    start = x_ref[variable_cols_array].copy()
    lb = lower[variable_cols_array]
    ub = upper[variable_cols_array]
    edge_nodes = {int(node_indices[0]), int(node_indices[-1])}
    edge_cols: set[int] = set()
    if not SOURCE_DOMAIN_FREEZE_EDGES:
        for idx in edge_nodes:
            edge_cols.update({idx, n + idx, 2 * n + idx})
    edge_anchor_positions = np.asarray(
        [pos for pos, col in enumerate(variable_cols_array) if int(col) in edge_cols],
        dtype=int,
    )

    initial_rows = _source_domain_residual_rows(x_ref, params, interval_indices, fractions)
    initial_metrics = _residual_metrics_for_x(x_ref, params)
    initial_extra = _source_extra_max_for_x(x_ref, params)
    initial_score = float(
        max(
            float(np.linalg.norm(initial_rows, ord=np.inf)) if initial_rows.size else 0.0,
            initial_metrics["full"],
            initial_extra if np.isfinite(initial_extra) else 0.0,
        )
    )

    def local_residual(trial: np.ndarray) -> np.ndarray:
        full = x_ref.copy()
        full[variable_cols_array] = trial
        rows = _source_domain_residual_rows(full, params, interval_indices, fractions)
        anchors: list[np.ndarray] = []
        if SOURCE_DOMAIN_EDGE_ANCHOR_WEIGHT > 0.0 and edge_anchor_positions.size:
            anchors.append(float(SOURCE_DOMAIN_EDGE_ANCHOR_WEIGHT) * (trial[edge_anchor_positions] - start[edge_anchor_positions]))
        if SOURCE_DOMAIN_ALL_ANCHOR_WEIGHT > 0.0:
            anchors.append(float(SOURCE_DOMAIN_ALL_ANCHOR_WEIGHT) * (trial - start))
        if anchors:
            rows = np.concatenate([rows, *anchors])
        return rows

    sparsity = _source_domain_sparsity(
        variable_cols_array,
        interval_indices,
        n,
        fractions,
        SOURCE_DOMAIN_INCLUDE_GLOBALS,
        edge_anchor_positions if SOURCE_DOMAIN_EDGE_ANCHOR_WEIGHT > 0.0 else np.asarray([], dtype=int),
        SOURCE_DOMAIN_ALL_ANCHOR_WEIGHT,
    )

    from scipy.optimize import least_squares

    result = least_squares(
        local_residual,
        np.clip(start, lb + 1.0e-12, ub - 1.0e-12),
        bounds=(lb, ub),
        jac_sparsity=sparsity,
        x_scale="jac",
        loss="linear",
        ftol=RESIDUAL_TOL,
        xtol=RESIDUAL_TOL,
        gtol=RESIDUAL_TOL,
        max_nfev=SOURCE_DOMAIN_MAX_NFEV,
        verbose=0,
    )

    candidate = x_ref.copy()
    candidate[variable_cols_array] = result.x
    candidate_rows = _source_domain_residual_rows(candidate, params, interval_indices, fractions)
    candidate_metrics = _residual_metrics_for_x(candidate, params)
    candidate_extra = _source_extra_max_for_x(candidate, params)
    candidate_score = float(
        max(
            float(np.linalg.norm(candidate_rows, ord=np.inf)) if candidate_rows.size else 0.0,
            candidate_metrics["full"],
            candidate_extra if np.isfinite(candidate_extra) else 0.0,
        )
    )

    best_x = x_ref
    best_metrics = initial_metrics
    best_rows = initial_rows
    best_extra = initial_extra
    best_score = initial_score
    best_alpha = 0.0
    lower_full, upper_full = pilot._bounds(params)
    step_delta = candidate - x_ref
    trials: list[dict[str, Any]] = []
    for exponent in range(max(1, int(SOURCE_DOMAIN_LINE_SEARCH_STEPS))):
        alpha = 0.5**exponent
        trial_x = np.clip(x_ref + alpha * step_delta, lower_full + 1.0e-12, upper_full - 1.0e-12)
        trial_rows = _source_domain_residual_rows(trial_x, params, interval_indices, fractions)
        metrics = _residual_metrics_for_x(trial_x, params)
        extra = _source_extra_max_for_x(trial_x, params)
        selected = float(np.linalg.norm(trial_rows, ord=np.inf)) if trial_rows.size else math.nan
        score = float(max(selected, metrics["full"], extra if np.isfinite(extra) else 0.0))
        guard = bool(np.isfinite(score) and metrics["full"] <= max(initial_metrics["full"], initial_score))
        trials.append(
            {
                "alpha": float(alpha),
                "score": score,
                "selected": selected,
                "full": metrics["full"],
                "mass": metrics["mass"],
                "source_band_extra": extra,
                "guard_pass": guard,
            }
        )
        if guard and score < best_score:
            best_x = trial_x
            best_metrics = metrics
            best_rows = trial_rows
            best_extra = extra
            best_score = score
            best_alpha = float(alpha)

    _logu, _logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x_ref, params)
    interval_mid_R_rg = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g
    return best_x, {
        "source_domain_correct_enabled": True,
        "source_domain_correct_applied": bool(best_alpha > 0.0),
        "source_domain_correct_fractions": fractions.tolist(),
        "source_domain_correct_halo_intervals": int(halo),
        "source_domain_correct_freeze_edges": bool(SOURCE_DOMAIN_FREEZE_EDGES),
        "source_domain_correct_include_globals": bool(SOURCE_DOMAIN_INCLUDE_GLOBALS),
        "source_domain_correct_first_interval": int(interval_indices[0]),
        "source_domain_correct_last_interval": int(interval_indices[-1]),
        "source_domain_correct_first_interval_R_rg": float(interval_mid_R_rg[int(interval_indices[0])]),
        "source_domain_correct_last_interval_R_rg": float(interval_mid_R_rg[int(interval_indices[-1])]),
        "source_domain_correct_n_intervals": int(interval_indices.size),
        "source_domain_correct_n_nodes": int(node_indices.size),
        "source_domain_correct_n_variables": int(variable_cols_array.size),
        "source_domain_correct_n_rows": int((2 * fractions.size + 1) * interval_indices.size),
        "source_domain_correct_initial_selected": float(np.linalg.norm(initial_rows, ord=np.inf)) if initial_rows.size else math.nan,
        "source_domain_correct_candidate_selected": float(np.linalg.norm(candidate_rows, ord=np.inf)) if candidate_rows.size else math.nan,
        "source_domain_correct_final_selected": float(np.linalg.norm(best_rows, ord=np.inf)) if best_rows.size else math.nan,
        "source_domain_correct_initial_score": initial_score,
        "source_domain_correct_candidate_score": candidate_score,
        "source_domain_correct_final_score": best_score,
        "source_domain_correct_initial_full": initial_metrics["full"],
        "source_domain_correct_candidate_full": candidate_metrics["full"],
        "source_domain_correct_final_full": best_metrics["full"],
        "source_domain_correct_initial_mass": initial_metrics["mass"],
        "source_domain_correct_final_mass": best_metrics["mass"],
        "source_domain_correct_initial_extra": initial_extra,
        "source_domain_correct_candidate_extra": candidate_extra,
        "source_domain_correct_final_extra": best_extra,
        "source_domain_correct_alpha": best_alpha,
        "source_domain_correct_nfev": int(result.nfev),
        "source_domain_correct_success": bool(result.success),
        "source_domain_correct_message": str(result.message),
        "source_domain_correct_variable_kinds": variable_kinds,
        "source_domain_correct_trials": trials,
    }


def _source_buffer_sample_fractions() -> np.ndarray:
    values: list[float] = []
    for piece in SOURCE_BUFFER_FRACTIONS_RAW.split(","):
        text = piece.strip()
        if not text:
            continue
        value = float(text)
        if not 0.0 < value < 1.0:
            raise ValueError("source-buffer sample fractions must lie strictly between 0 and 1")
        values.append(value)
    if not values:
        values = [0.5]
    return np.asarray(sorted(set(float(value) for value in values)), dtype=float)


def _source_buffer_interval_indices(x: np.ndarray, params) -> tuple[np.ndarray, np.ndarray]:
    interval_indices, _node_indices = _source_band_interval_indices(x, params)
    if interval_indices.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    n = int(params.n_nodes)
    halo = max(0, int(SOURCE_BUFFER_HALO_INTERVALS))
    first = max(0, int(interval_indices[0]) - halo)
    last = min(n - 2, int(interval_indices[-1]) + halo)
    intervals = np.arange(first, last + 1, dtype=int)
    nodes = np.arange(first, last + 2, dtype=int)
    return intervals, nodes


def _source_buffer_initial_delta(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    local_params,
    interval_indices: np.ndarray,
) -> np.ndarray:
    deltas = np.empty(int(interval_indices.size), dtype=float)
    for pos, idx_value in enumerate(interval_indices):
        idx = int(idx_value)
        try:
            wind_integral, source_integral, _scale, _left, _right = _source_buffer_mass_terms_from_unpacked(
                logu, logT, logMdot, logR, lambda0, local_params, idx
            )
            deltas[pos] = float(wind_integral - source_integral)
        except Exception:
            deltas[pos] = 0.0
    return deltas


def _source_buffer_interval_rows(
    logu: np.ndarray,
    logT: np.ndarray,
    logMdot: np.ndarray,
    logR: np.ndarray,
    lambda0: float,
    local_params,
    idx: int,
    delta_m: float,
    fractions: np.ndarray,
) -> np.ndarray:
    rows: list[float] = []
    dx, y_left, y_right, _xm = _interval_geometry(logu, logT, logR, idx)
    if dx <= 0.0:
        return np.full(2 * int(fractions.size) + 2, 1.0e6, dtype=float)
    g = (y_right - y_left) / dx
    for frac in fractions:
        t = float(frac)
        xq = float(logR[idx] + t * dx)
        yq = (1.0 - t) * y_left + t * y_right
        rows.extend(float(SOURCE_BUFFER_STATE_WEIGHT) * _scaled_residual_at(xq, yq, g, lambda0, local_params))
    wind_integral, source_integral, mdot_scale, mdot_left, mdot_right = _source_buffer_mass_terms_from_unpacked(
        logu, logT, logMdot, logR, lambda0, local_params, idx
    )
    net_integral = float(wind_integral - source_integral)
    rows.append(float(SOURCE_BUFFER_INTEGRAL_WEIGHT) * (float(delta_m) - net_integral) / mdot_scale)
    rows.append(float(SOURCE_BUFFER_JUMP_WEIGHT) * (mdot_right - mdot_left - float(delta_m)) / mdot_scale)
    return np.asarray(rows, dtype=float)


def _source_buffer_residual_rows(
    x: np.ndarray,
    params,
    interval_indices: np.ndarray,
    fractions: np.ndarray,
    delta_m: np.ndarray,
) -> np.ndarray:
    rows_per_interval = 2 * int(fractions.size) + 2
    expected = rows_per_interval * int(interval_indices.size)
    try:
        logu, logT, logMdot, _logR_son, lambda0, logR = pilot._unpack(x, params)
        local_params = pilot._local_params(params, logR, logMdot)
        rows: list[float] = []
        for pos, idx_value in enumerate(interval_indices):
            rows.extend(
                _source_buffer_interval_rows(
                    logu,
                    logT,
                    logMdot,
                    logR,
                    lambda0,
                    local_params,
                    int(idx_value),
                    float(delta_m[pos]),
                    fractions,
                )
            )
        return np.asarray(rows, dtype=float)
    except Exception:
        return np.full(expected, 1.0e6, dtype=float)


def _source_buffer_row_summary(
    rows: np.ndarray,
    interval_indices: np.ndarray,
    fractions: np.ndarray,
    logR: np.ndarray,
    params,
) -> dict[str, Any]:
    rows_per_interval = 2 * int(fractions.size) + 2
    if rows.size != rows_per_interval * int(interval_indices.size) or rows.size == 0:
        return {
            "selected": math.nan,
            "state": math.nan,
            "integral": math.nan,
            "jump": math.nan,
            "peak_jump_R_rg": math.nan,
            "peak_integral_R_rg": math.nan,
        }
    matrix = np.asarray(rows, dtype=float).reshape(int(interval_indices.size), rows_per_interval)
    state_cols = 2 * int(fractions.size)
    integral = matrix[:, state_cols]
    jump = matrix[:, state_cols + 1]
    mids = np.exp(0.5 * (logR[interval_indices] + logR[interval_indices + 1])) / params.r_g
    peak_jump = int(np.argmax(np.abs(jump))) if jump.size else 0
    peak_integral = int(np.argmax(np.abs(integral))) if integral.size else 0
    return {
        "selected": float(np.linalg.norm(rows, ord=np.inf)),
        "state": float(np.linalg.norm(matrix[:, :state_cols], ord=np.inf)) if state_cols > 0 else math.nan,
        "integral": float(np.linalg.norm(integral, ord=np.inf)) if integral.size else math.nan,
        "jump": float(np.linalg.norm(jump, ord=np.inf)) if jump.size else math.nan,
        "peak_jump_R_rg": float(mids[peak_jump]) if mids.size else math.nan,
        "peak_integral_R_rg": float(mids[peak_integral]) if mids.size else math.nan,
    }


def _source_buffer_sparsity(
    variable_cols: np.ndarray,
    n_state_vars: int,
    interval_indices: np.ndarray,
    n_nodes: int,
    fractions: np.ndarray,
    include_globals: bool,
    anchor_positions: np.ndarray,
    all_anchor_weight: float,
):
    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None
    rows_per_interval = 2 * int(fractions.size) + 2
    n_interval_rows = rows_per_interval * int(interval_indices.size)
    n_anchor_rows = int(anchor_positions.size) + (int(n_state_vars) if all_anchor_weight > 0.0 else 0)
    pattern = lil_matrix((n_interval_rows + n_anchor_rows, int(variable_cols.size)), dtype=int)
    col_to_local = {int(col): int(pos) for pos, col in enumerate(variable_cols[:n_state_vars])}
    global_cols = [3 * int(n_nodes), 3 * int(n_nodes) + 1] if include_globals else []
    delta_start = int(n_state_vars)
    for local_interval_idx, interval_idx_value in enumerate(interval_indices):
        interval_idx = int(interval_idx_value)
        state_cols = [
            interval_idx,
            interval_idx + 1,
            n_nodes + interval_idx,
            n_nodes + interval_idx + 1,
            2 * n_nodes + interval_idx,
            2 * n_nodes + interval_idx + 1,
            *global_cols,
        ]
        row_start = rows_per_interval * local_interval_idx
        for row in range(row_start, row_start + rows_per_interval):
            for full_col in state_cols:
                local_col = col_to_local.get(int(full_col))
                if local_col is not None:
                    pattern[row, local_col] = 1
            pattern[row, delta_start + local_interval_idx] = 1
    row = n_interval_rows
    for pos in anchor_positions:
        pattern[row, int(pos)] = 1
        row += 1
    if all_anchor_weight > 0.0:
        for pos in range(int(n_state_vars)):
            pattern[row, pos] = 1
            row += 1
    return pattern.tocsr()


def _source_buffer_correct(x0: np.ndarray, params, eta_E: float) -> tuple[np.ndarray, dict[str, Any]]:
    if not SOURCE_BUFFER_CORRECT:
        return x0, {}
    _set_eta(eta_E)
    n = int(params.n_nodes)
    x_ref = np.asarray(x0, dtype=float)
    fractions = _source_buffer_sample_fractions()
    interval_indices, node_indices = _source_buffer_interval_indices(x_ref, params)
    if interval_indices.size == 0 or node_indices.size < 2:
        return x0, {
            "source_buffer_correct_enabled": True,
            "source_buffer_correct_applied": False,
            "source_buffer_correct_reason": "no source-buffer intervals",
        }
    active_nodes = node_indices[1:-1] if SOURCE_BUFFER_FREEZE_EDGES and node_indices.size > 2 else node_indices
    state_cols: list[int] = []
    variable_kinds: list[str] = []
    for idx in active_nodes:
        state_cols.append(int(idx))
        variable_kinds.append("logu")
    for idx in active_nodes:
        state_cols.append(int(n + idx))
        variable_kinds.append("logT")
    for idx in active_nodes:
        state_cols.append(int(2 * n + idx))
        variable_kinds.append("logMdot")
    if SOURCE_BUFFER_INCLUDE_GLOBALS:
        state_cols.extend([3 * n, 3 * n + 1])
        variable_kinds.extend(["logR_son", "lambda0"])
    state_cols_array = np.asarray(state_cols, dtype=int)
    if state_cols_array.size == 0:
        return x0, {
            "source_buffer_correct_enabled": True,
            "source_buffer_correct_applied": False,
            "source_buffer_correct_reason": "no state variable columns",
        }

    logu, logT, logMdot, _logR_son, lambda0, logR = pilot._unpack(x_ref, params)
    local_params = pilot._local_params(params, logR, logMdot)
    delta0 = _source_buffer_initial_delta(logu, logT, logMdot, logR, lambda0, local_params, interval_indices)
    start_state = x_ref[state_cols_array].copy()
    start = np.concatenate([start_state, delta0])
    lower, upper = pilot._bounds(params)
    state_lb = lower[state_cols_array]
    state_ub = upper[state_cols_array]
    mdot_scale = max(float(params.Mdot_g_s), 1.0e-300)
    delta_limit = 5.0 * mdot_scale
    lb = np.concatenate([state_lb, np.full(delta0.size, -delta_limit, dtype=float)])
    ub = np.concatenate([state_ub, np.full(delta0.size, delta_limit, dtype=float)])

    edge_nodes = {int(node_indices[0]), int(node_indices[-1])}
    edge_cols: set[int] = set()
    if not SOURCE_BUFFER_FREEZE_EDGES:
        for idx in edge_nodes:
            edge_cols.update({idx, n + idx, 2 * n + idx})
    edge_anchor_positions = np.asarray(
        [pos for pos, col in enumerate(state_cols_array) if int(col) in edge_cols],
        dtype=int,
    )

    initial_rows = _source_buffer_residual_rows(x_ref, params, interval_indices, fractions, delta0)
    initial_metrics = _residual_metrics_for_x(x_ref, params)
    initial_extra = _source_extra_max_for_x(x_ref, params)
    initial_summary = _source_buffer_row_summary(initial_rows, interval_indices, fractions, logR, params)
    initial_score = float(
        max(
            initial_summary["selected"] if np.isfinite(initial_summary["selected"]) else 0.0,
            initial_metrics["full"],
            initial_extra if np.isfinite(initial_extra) else 0.0,
        )
    )

    def unpack_trial(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        full = x_ref.copy()
        full[state_cols_array] = trial[: state_cols_array.size]
        delta = np.asarray(trial[state_cols_array.size :], dtype=float)
        return full, delta

    def local_residual(trial: np.ndarray) -> np.ndarray:
        full, delta = unpack_trial(trial)
        rows = _source_buffer_residual_rows(full, params, interval_indices, fractions, delta)
        anchors: list[np.ndarray] = []
        if SOURCE_BUFFER_EDGE_ANCHOR_WEIGHT > 0.0 and edge_anchor_positions.size:
            anchors.append(float(SOURCE_BUFFER_EDGE_ANCHOR_WEIGHT) * (trial[edge_anchor_positions] - start[edge_anchor_positions]))
        if SOURCE_BUFFER_ALL_ANCHOR_WEIGHT > 0.0:
            anchors.append(float(SOURCE_BUFFER_ALL_ANCHOR_WEIGHT) * (trial[: state_cols_array.size] - start[: state_cols_array.size]))
        if anchors:
            rows = np.concatenate([rows, *anchors])
        return rows

    variable_cols = np.concatenate([state_cols_array, -1 - np.arange(delta0.size, dtype=int)])
    sparsity = _source_buffer_sparsity(
        variable_cols,
        state_cols_array.size,
        interval_indices,
        n,
        fractions,
        SOURCE_BUFFER_INCLUDE_GLOBALS,
        edge_anchor_positions if SOURCE_BUFFER_EDGE_ANCHOR_WEIGHT > 0.0 else np.asarray([], dtype=int),
        SOURCE_BUFFER_ALL_ANCHOR_WEIGHT,
    )

    from scipy.optimize import least_squares

    result = least_squares(
        local_residual,
        np.clip(start, lb + 1.0e-12, ub - 1.0e-12),
        bounds=(lb, ub),
        jac_sparsity=sparsity,
        x_scale="jac",
        loss="linear",
        ftol=RESIDUAL_TOL,
        xtol=RESIDUAL_TOL,
        gtol=RESIDUAL_TOL,
        max_nfev=SOURCE_BUFFER_MAX_NFEV,
        verbose=0,
    )

    candidate, candidate_delta = unpack_trial(result.x)
    candidate_rows = _source_buffer_residual_rows(candidate, params, interval_indices, fractions, candidate_delta)
    candidate_metrics = _residual_metrics_for_x(candidate, params)
    candidate_extra = _source_extra_max_for_x(candidate, params)
    _cu, _cT, _cM, _cson, _clambda, candidate_logR = pilot._unpack(candidate, params)
    _ = _cu, _cT, _cM, _cson, _clambda
    candidate_summary = _source_buffer_row_summary(candidate_rows, interval_indices, fractions, candidate_logR, params)
    candidate_score = float(
        max(
            candidate_summary["selected"] if np.isfinite(candidate_summary["selected"]) else 0.0,
            candidate_metrics["full"],
            candidate_extra if np.isfinite(candidate_extra) else 0.0,
        )
    )

    best_x = x_ref
    best_delta = delta0
    best_rows = initial_rows
    best_metrics = initial_metrics
    best_extra = initial_extra
    best_summary = initial_summary
    best_score = initial_score
    best_alpha = 0.0
    lower_full, upper_full = pilot._bounds(params)
    step_delta_x = candidate - x_ref
    step_delta_aux = candidate_delta - delta0
    trials: list[dict[str, Any]] = []
    for exponent in range(max(1, int(SOURCE_BUFFER_LINE_SEARCH_STEPS))):
        alpha = 0.5**exponent
        trial_x = np.clip(x_ref + alpha * step_delta_x, lower_full + 1.0e-12, upper_full - 1.0e-12)
        trial_delta = delta0 + alpha * step_delta_aux
        trial_rows = _source_buffer_residual_rows(trial_x, params, interval_indices, fractions, trial_delta)
        trial_logu, trial_logT, trial_logMdot, _trial_logR_son, _trial_lambda0, trial_logR = pilot._unpack(trial_x, params)
        _ = trial_logu, trial_logT, trial_logMdot, _trial_logR_son, _trial_lambda0
        summary = _source_buffer_row_summary(trial_rows, interval_indices, fractions, trial_logR, params)
        metrics = _residual_metrics_for_x(trial_x, params)
        extra = _source_extra_max_for_x(trial_x, params)
        selected = summary["selected"]
        score = float(max(selected, metrics["full"], extra if np.isfinite(extra) else 0.0))
        guard = bool(np.isfinite(score) and metrics["full"] <= max(initial_metrics["full"], initial_score))
        trials.append(
            {
                "alpha": float(alpha),
                "score": score,
                "selected": selected,
                "state": summary["state"],
                "integral": summary["integral"],
                "jump": summary["jump"],
                "full": metrics["full"],
                "mass": metrics["mass"],
                "source_band_extra": extra,
                "guard_pass": guard,
            }
        )
        secondary_improved = (
            np.isfinite(extra)
            and np.isfinite(best_extra)
            and extra < best_extra
            and metrics["full"] <= best_metrics["full"] + 1.0e-14
        )
        if guard and (score < best_score or (score <= best_score * (1.0 + 1.0e-12) and secondary_improved)):
            best_x = trial_x
            best_delta = trial_delta
            best_rows = trial_rows
            best_metrics = metrics
            best_extra = extra
            best_summary = summary
            best_score = score
            best_alpha = float(alpha)

    _logu, _logT, _logMdot, _logR_son, _lambda0, final_logR = pilot._unpack(best_x, params)
    final_summary = _source_buffer_row_summary(best_rows, interval_indices, fractions, final_logR, params)
    delta_scale = max(float(params.Mdot_g_s), 1.0e-300)
    return best_x, {
        "source_buffer_correct_enabled": True,
        "source_buffer_correct_applied": bool(best_alpha > 0.0),
        "source_buffer_correct_fractions": fractions.tolist(),
        "source_buffer_correct_halo_intervals": int(SOURCE_BUFFER_HALO_INTERVALS),
        "source_buffer_correct_freeze_edges": bool(SOURCE_BUFFER_FREEZE_EDGES),
        "source_buffer_correct_include_globals": bool(SOURCE_BUFFER_INCLUDE_GLOBALS),
        "source_buffer_correct_local_jacobian": "sparse_fd",
        "source_buffer_correct_mass_quadrature": str(SOURCE_BUFFER_MASS_QUADRATURE),
        "source_buffer_correct_first_interval": int(interval_indices[0]),
        "source_buffer_correct_last_interval": int(interval_indices[-1]),
        "source_buffer_correct_n_intervals": int(interval_indices.size),
        "source_buffer_correct_n_nodes": int(node_indices.size),
        "source_buffer_correct_n_state_variables": int(state_cols_array.size),
        "source_buffer_correct_n_delta_variables": int(delta0.size),
        "source_buffer_correct_n_rows": int((2 * fractions.size + 2) * interval_indices.size),
        "source_buffer_correct_initial_selected": initial_summary["selected"],
        "source_buffer_correct_candidate_selected": candidate_summary["selected"],
        "source_buffer_correct_final_selected": final_summary["selected"],
        "source_buffer_correct_initial_state": initial_summary["state"],
        "source_buffer_correct_final_state": final_summary["state"],
        "source_buffer_correct_initial_integral": initial_summary["integral"],
        "source_buffer_correct_final_integral": final_summary["integral"],
        "source_buffer_correct_initial_jump": initial_summary["jump"],
        "source_buffer_correct_final_jump": final_summary["jump"],
        "source_buffer_correct_final_peak_jump_R_rg": final_summary["peak_jump_R_rg"],
        "source_buffer_correct_final_peak_integral_R_rg": final_summary["peak_integral_R_rg"],
        "source_buffer_correct_initial_score": initial_score,
        "source_buffer_correct_candidate_score": candidate_score,
        "source_buffer_correct_final_score": best_score,
        "source_buffer_correct_initial_full": initial_metrics["full"],
        "source_buffer_correct_candidate_full": candidate_metrics["full"],
        "source_buffer_correct_final_full": best_metrics["full"],
        "source_buffer_correct_initial_mass": initial_metrics["mass"],
        "source_buffer_correct_final_mass": best_metrics["mass"],
        "source_buffer_correct_initial_extra": initial_extra,
        "source_buffer_correct_candidate_extra": candidate_extra,
        "source_buffer_correct_final_extra": best_extra,
        "source_buffer_correct_delta_min_over_inner": float(np.min(best_delta) / delta_scale) if best_delta.size else math.nan,
        "source_buffer_correct_delta_max_over_inner": float(np.max(best_delta) / delta_scale) if best_delta.size else math.nan,
        "source_buffer_correct_delta_sum_over_inner": float(np.sum(best_delta) / delta_scale) if best_delta.size else math.nan,
        "source_buffer_correct_alpha": best_alpha,
        "source_buffer_correct_nfev": int(result.nfev),
        "source_buffer_correct_success": bool(result.success),
        "source_buffer_correct_message": str(result.message),
        "source_buffer_correct_variable_kinds": variable_kinds,
        "source_buffer_correct_trials": trials,
    }


def _source_element_interval_indices(x: np.ndarray, params) -> tuple[np.ndarray, np.ndarray]:
    interval_indices, _node_indices = _source_band_interval_indices(x, params)
    if interval_indices.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    n = int(params.n_nodes)
    halo = max(0, int(SOURCE_ELEMENT_HALO_INTERVALS))
    first = max(0, int(interval_indices[0]) - halo)
    last = min(n - 2, int(interval_indices[-1]) + halo)
    intervals = np.arange(first, last + 1, dtype=int)
    nodes = np.arange(first, last + 2, dtype=int)
    return intervals, nodes


def _source_element_refined_params_from_x(x_old: np.ndarray, old_params):
    _logu, _logT, _logMdot, logR_son, _lambda0, logR_old = pilot._unpack(x_old, old_params)
    interval_indices, node_indices = _source_element_interval_indices(x_old, old_params)
    subdivisions = max(2, int(SOURCE_ELEMENT_SUBDIVISIONS))
    if interval_indices.size == 0 or node_indices.size < 2:
        return old_params, {
            "source_element_refine_enabled": True,
            "source_element_refine_applied": False,
            "source_element_refine_reason": "no source-element intervals",
        }
    inserted: list[float] = []
    for idx_value in interval_indices:
        idx = int(idx_value)
        left = float(logR_old[idx])
        right = float(logR_old[idx + 1])
        if right <= left:
            continue
        for sub_idx in range(1, subdivisions):
            inserted.append(left + (right - left) * float(sub_idx) / float(subdivisions))
    if not inserted:
        return old_params, {
            "source_element_refine_enabled": True,
            "source_element_refine_applied": False,
            "source_element_refine_reason": "no new internal nodes",
        }
    absolute_nodes = np.concatenate([logR_old, np.asarray(inserted, dtype=float)])
    absolute_nodes = absolute_nodes[(absolute_nodes >= logR_old[0] - 1.0e-12) & (absolute_nodes <= logR_old[-1] + 1.0e-12)]
    absolute_nodes = np.asarray(sorted({round(float(value), 14) for value in absolute_nodes}), dtype=float)
    absolute_nodes[0] = float(logR_old[0])
    absolute_nodes[-1] = float(logR_old[-1])
    if absolute_nodes.size <= logR_old.size:
        return old_params, {
            "source_element_refine_enabled": True,
            "source_element_refine_applied": False,
            "source_element_refine_reason": "refined grid did not add nodes",
        }
    span = max(float(logR_old[-1] - logR_son), 1.0e-300)
    xi = _enforce_min_spacing((absolute_nodes - float(logR_son)) / span)
    new_params = replace(old_params, n_nodes=int(xi.size), custom_grid_xi=tuple(float(value) for value in xi))
    interval_mid_rg = np.exp(0.5 * (logR_old[interval_indices] + logR_old[interval_indices + 1])) / old_params.r_g
    return new_params, {
        "source_element_refine_enabled": True,
        "source_element_refine_applied": True,
        "source_element_refine_old_N": int(old_params.n_nodes),
        "source_element_refine_new_N": int(new_params.n_nodes),
        "source_element_refine_subdivisions": int(subdivisions),
        "source_element_refine_halo_intervals": int(SOURCE_ELEMENT_HALO_INTERVALS),
        "source_element_refine_split_intervals": int(interval_indices.size),
        "source_element_refine_inserted_nodes": int(new_params.n_nodes - old_params.n_nodes),
        "source_element_refine_first_R_rg": float(interval_mid_rg[0]) if interval_mid_rg.size else math.nan,
        "source_element_refine_last_R_rg": float(interval_mid_rg[-1]) if interval_mid_rg.size else math.nan,
    }


def _source_element_remap_x_to_params(x_old: np.ndarray, old_params, new_params) -> np.ndarray:
    method = SOURCE_ELEMENT_REMAP_METHOD
    if method in {"hermite", "ode_hermite", "ode-slope-hermite"}:
        x_new = _hermite_remap_local_x_to_params(x_old, old_params, new_params)
        return _source_element_apply_mass_seed(x_old, old_params, x_new, new_params)
    logu_old, logT_old, logMdot_old, logR_son, lambda0, logR_old = pilot._unpack(x_old, old_params)
    logR_new = pilot.computational_grid(new_params, logR_son)

    def interp(values: np.ndarray) -> np.ndarray:
        if method in {"pchip", "monotone", "shape_preserving"}:
            try:
                from scipy.interpolate import PchipInterpolator

                return np.asarray(PchipInterpolator(logR_old, values, extrapolate=True)(logR_new), dtype=float)
            except Exception:
                return np.interp(logR_new, logR_old, values)
        return np.interp(logR_new, logR_old, values)

    logu_new = interp(logu_old)
    logT_new = interp(logT_old)
    logMdot_new = interp(logMdot_old)
    x_new = pilot._pack(logu_new, logT_new, logMdot_new, logR_son, lambda0)
    lower, upper = pilot._bounds(new_params)
    x_new = np.clip(x_new, lower + 1.0e-12, upper - 1.0e-12)
    return _source_element_apply_mass_seed(x_old, old_params, x_new, new_params)


def _source_element_apply_mass_seed(x_old: np.ndarray, old_params, x_new: np.ndarray, new_params) -> np.ndarray:
    if SOURCE_ELEMENT_MASS_SEED not in {"fv", "fv_budget", "mass_budget", "conservative"}:
        return x_new
    logu_new, logT_new, logMdot_new, logR_son, lambda0, logR_new = pilot._unpack(x_new, new_params)
    _logu_old, _logT_old, logMdot_old, _logR_son_old, _lambda0_old, logR_old = pilot._unpack(x_old, old_params)
    interval_indices, _node_indices = _source_element_interval_indices(x_old, old_params)
    if interval_indices.size == 0:
        return x_new
    old_to_new: list[int] = []
    for value in logR_old:
        old_to_new.append(int(np.argmin(np.abs(logR_new - float(value)))))
    logMdot = np.asarray(logMdot_new, dtype=float).copy()
    lower, upper = pilot._bounds(new_params)
    logMdot_lower = lower[2 * int(new_params.n_nodes) : 3 * int(new_params.n_nodes)] + 1.0e-12
    logMdot_upper = upper[2 * int(new_params.n_nodes) : 3 * int(new_params.n_nodes)] - 1.0e-12
    for _sweep in range(max(1, int(SOURCE_ELEMENT_MASS_SEED_SWEEPS))):
        local_params = pilot._local_params(new_params, logR_new, logMdot)
        for old_idx_value in interval_indices:
            old_idx = int(old_idx_value)
            left_pos = int(old_to_new[old_idx])
            right_pos = int(old_to_new[old_idx + 1])
            if right_pos <= left_pos + 1:
                continue
            sub_indices = np.arange(left_pos, right_pos, dtype=int)
            net = np.empty(sub_indices.size, dtype=float)
            for pos, sub_idx in enumerate(sub_indices):
                try:
                    wind_integral, source_integral, _scale, _left, _right = _source_buffer_mass_terms_from_unpacked(
                        logu_new, logT_new, logMdot, logR_new, lambda0, local_params, int(sub_idx)
                    )
                    net[pos] = float(wind_integral - source_integral)
                except Exception:
                    net[pos] = 0.0
            M_left = float(np.exp(logMdot_old[old_idx]))
            M_right = float(np.exp(logMdot_old[old_idx + 1]))
            correction = (M_right - M_left - float(np.sum(net))) / float(max(net.size, 1))
            current = M_left
            for pos, sub_idx in enumerate(sub_indices):
                current += float(net[pos]) + correction
                node = int(sub_idx + 1)
                if node < right_pos:
                    logMdot[node] = float(np.log(max(current, 1.0e-8 * float(new_params.Mdot_g_s))))
            logMdot[left_pos] = float(logMdot_old[old_idx])
            logMdot[right_pos] = float(logMdot_old[old_idx + 1])
        logMdot = np.clip(logMdot, logMdot_lower, logMdot_upper)
    x_seeded = pilot._pack(logu_new, logT_new, logMdot, logR_son, lambda0)
    lower_full, upper_full = pilot._bounds(new_params)
    return np.clip(x_seeded, lower_full + 1.0e-12, upper_full - 1.0e-12)


def _source_element_refine_seed(x0: np.ndarray, params, eta_E: float) -> tuple[np.ndarray, Any, dict[str, Any]]:
    if not SOURCE_ELEMENT_REFINE:
        return x0, params, {}
    _set_eta(eta_E)
    new_params, info = _source_element_refined_params_from_x(x0, params)
    if not info.get("source_element_refine_applied", False):
        return x0, params, info
    before = _residual_metrics_for_x(x0, params)
    before_extra = _source_extra_max_for_x(x0, params)
    x_refined = _source_element_remap_x_to_params(x0, params, new_params)
    after = _residual_metrics_for_x(x_refined, new_params)
    after_extra = _source_extra_max_for_x(x_refined, new_params)
    info.update(
        {
            "source_element_refine_remap_method": str(SOURCE_ELEMENT_REMAP_METHOD),
            "source_element_refine_mass_seed": str(SOURCE_ELEMENT_MASS_SEED),
            "source_element_refine_mass_seed_sweeps": int(SOURCE_ELEMENT_MASS_SEED_SWEEPS),
            "source_element_refine_before_full": before["full"],
            "source_element_refine_before_mass": before["mass"],
            "source_element_refine_before_extra": before_extra,
            "source_element_refine_seed_full": after["full"],
            "source_element_refine_seed_mass": after["mass"],
            "source_element_refine_seed_extra": after_extra,
        }
    )
    return x_refined, new_params, info


def _source_microdomain_seed(x0: np.ndarray, params, eta_E: float) -> tuple[np.ndarray, Any, dict[str, Any]]:
    if not SOURCE_MICRO_DOMAIN:
        return x0, params, {}
    new_params, info = _source_microdomain_params_from_x(x0, params)
    if not info.get("source_microdomain_applied", False):
        return x0, params, info
    x_micro = _hermite_remap_local_x_to_params(x0, params, new_params)
    before = _residual_metrics_for_x(x0, params)
    after_seed = _residual_metrics_for_x(x_micro, new_params)
    info.update(
        {
            "source_microdomain_remap_method": "ode_slope_hermite",
            "source_microdomain_before_full": before["full"],
            "source_microdomain_seed_full": after_seed["full"],
            "source_microdomain_seed_mass": after_seed["mass"],
            "source_microdomain_seed_extra": _source_extra_max_for_x(x_micro, new_params),
        }
    )
    x_corrected, correct_info = _source_microdomain_correct(x_micro, new_params, eta_E)
    info.update(correct_info)
    final = _residual_metrics_for_x(x_corrected, new_params)
    info.update(
        {
            "source_microdomain_final_full": final["full"],
            "source_microdomain_final_mass": final["mass"],
            "source_microdomain_final_extra": _source_extra_max_for_x(x_corrected, new_params),
        }
    )
    return x_corrected, new_params, info


def _xi_from_x(x: np.ndarray, params) -> np.ndarray:
    _logu, _logT, _logMdot, _logR_son, _lambda0, logR = pilot._unpack(x, params)
    span = max(float(logR[-1] - logR[0]), 1.0e-300)
    xi = (np.asarray(logR, dtype=float) - float(logR[0])) / span
    xi[0] = 0.0
    xi[-1] = 1.0
    return _enforce_min_spacing(xi)


def _target_xi_for_params(params, logR_son: float) -> np.ndarray:
    logR = pilot.computational_grid(params, logR_son)
    span = max(float(logR[-1] - logR[0]), 1.0e-300)
    xi = (np.asarray(logR, dtype=float) - float(logR[0])) / span
    xi[0] = 0.0
    xi[-1] = 1.0
    return _enforce_min_spacing(xi)


def _collapsed_refinement_xi(old_xi: np.ndarray, final_xi: np.ndarray) -> np.ndarray:
    old = np.asarray(old_xi, dtype=float)
    final = np.asarray(final_xi, dtype=float)
    collapsed = final.copy()
    tol = 5.0e-10
    fraction = float(np.clip(GRID_HOMOTOPY_COLLAPSE_FRACTION, 1.0e-5, 0.45))
    matched = np.zeros(final.size, dtype=bool)
    for idx, value in enumerate(final):
        nearest = int(np.argmin(np.abs(old - value)))
        if abs(float(old[nearest] - value)) <= tol:
            collapsed[idx] = float(old[nearest])
            matched[idx] = True
    for left_idx in range(old.size - 1):
        left = float(old[left_idx])
        right = float(old[left_idx + 1])
        if right <= left:
            continue
        mask = (~matched) & (final > left + tol) & (final < right - tol)
        insert_indices = np.nonzero(mask)[0]
        if insert_indices.size == 0:
            continue
        ordered = insert_indices[np.argsort(final[insert_indices])]
        for order, final_idx in enumerate(ordered):
            collapsed[int(final_idx)] = left + fraction * (order + 1) / (ordered.size + 1) * (right - left)
    collapsed[0] = 0.0
    collapsed[-1] = 1.0
    return _enforce_min_spacing(collapsed)


def _grid_homotopy_seed(
    x_old: np.ndarray,
    old_params,
    final_params,
    eta_E: float,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    if GRID_HOMOTOPY_STEPS <= 0:
        return _remap_local_x_to_params(x_old, old_params, final_params), final_params, {}
    _set_eta(eta_E)
    _logu, _logT, _logMdot, logR_son, _lambda0, _logR = pilot._unpack(x_old, old_params)
    old_xi = _xi_from_x(x_old, old_params)
    final_xi = _target_xi_for_params(final_params, logR_son)
    if final_xi.size <= old_xi.size:
        return _remap_local_x_to_params(x_old, old_params, final_params), final_params, {
            "grid_homotopy_enabled": True,
            "grid_homotopy_applied": False,
            "grid_homotopy_reason": "target grid is not a refinement",
        }
    collapsed_xi = _collapsed_refinement_xi(old_xi, final_xi)
    current_params = replace(final_params, custom_grid_xi=tuple(float(value) for value in collapsed_xi))
    current_x = _remap_local_x_to_params(x_old, old_params, current_params)
    stages: list[dict[str, Any]] = []
    metrics = _residual_metrics_for_x(current_x, current_params)
    stages.append(
        {
            "step": 0,
            "gamma": 0.0,
            "full": metrics["full"],
            "interval_R": metrics["interval_R"],
            "interval_E": metrics["interval_E"],
            "mass": metrics["mass"],
            "peak_interval_R_rg": metrics["peak_interval_R_rg"],
            "peak_mass_rg": metrics["peak_mass_rg"],
        }
    )
    print(
        "grid_homotopy step=0 "
        f"full={metrics['full']:.3e} R={metrics['interval_R']:.3e} "
        f"E={metrics['interval_E']:.3e} M={metrics['mass']:.3e}",
        flush=True,
    )
    for step in range(1, max(1, int(GRID_HOMOTOPY_STEPS)) + 1):
        gamma = float(step) / float(max(1, int(GRID_HOMOTOPY_STEPS)))
        xi = _enforce_min_spacing((1.0 - gamma) * collapsed_xi + gamma * final_xi)
        next_params = replace(final_params, custom_grid_xi=tuple(float(value) for value in xi))
        current_x = _remap_local_x_to_params(current_x, current_params, next_params)
        current_params = next_params
        before = _residual_metrics_for_x(current_x, current_params)
        block_info: dict[str, Any] = {}
        if GRID_HOMOTOPY_BLOCK_CORRECT:
            current_x, block_info = _coupled_block_correct(current_x, current_params, eta_E)
        after = _residual_metrics_for_x(current_x, current_params)
        stages.append(
            {
                "step": int(step),
                "gamma": gamma,
                "before_full": before["full"],
                "full": after["full"],
                "interval_R": after["interval_R"],
                "interval_E": after["interval_E"],
                "mass": after["mass"],
                "peak_interval_R_rg": after["peak_interval_R_rg"],
                "peak_mass_rg": after["peak_mass_rg"],
                "block_applied": bool(block_info.get("block_correct_applied", False)),
                "block_nfev": block_info.get("block_correct_nfev"),
                "block_alpha": block_info.get("block_correct_alpha"),
            }
        )
        print(
            f"grid_homotopy step={step} gamma={gamma:.3f} "
            f"before={before['full']:.3e} full={after['full']:.3e} "
            f"R={after['interval_R']:.3e} E={after['interval_E']:.3e} M={after['mass']:.3e}",
            flush=True,
        )
    final_metrics = _residual_metrics_for_x(current_x, current_params)
    return current_x, current_params, {
        "grid_homotopy_enabled": True,
        "grid_homotopy_applied": True,
        "grid_homotopy_steps": int(GRID_HOMOTOPY_STEPS),
        "grid_homotopy_collapse_fraction": float(GRID_HOMOTOPY_COLLAPSE_FRACTION),
        "grid_homotopy_block_correct": bool(GRID_HOMOTOPY_BLOCK_CORRECT),
        "grid_homotopy_initial_full": float(stages[0]["full"]),
        "grid_homotopy_final_full": final_metrics["full"],
        "grid_homotopy_final_interval_R": final_metrics["interval_R"],
        "grid_homotopy_final_interval_E": final_metrics["interval_E"],
        "grid_homotopy_final_mass": final_metrics["mass"],
        "grid_homotopy_stages": stages,
    }


def _z_from_x(x: np.ndarray, params) -> np.ndarray:
    logu, logT, _logMdot, logR_son, lambda0, _logR = pilot._unpack(x, params)
    return pilot._state_vector(logu, logT, logR_son, lambda0)


def _solve_with_picard(x0: np.ndarray, params, eta_E: float):
    result = _solve_stage(x0, params, eta_E)
    current_params = params
    total_nfev = int(result.nfev)
    picard_used = 0
    for _idx in range(max(0, OUTER_SLOPE_PICARD_ITERS)):
        refreshed = scan.apply_outer_slopes_from_state(_z_from_x(result.x, current_params), current_params)
        trial = _solve_stage(result.x, refreshed, eta_E)
        total_nfev += int(trial.nfev)
        picard_used += 1
        if float(np.linalg.norm(_residual(trial.x, refreshed), ord=np.inf)) <= float(
            np.linalg.norm(_residual(result.x, current_params), ord=np.inf)
        ):
            result = trial
            current_params = refreshed
        else:
            break
    return result, current_params, total_nfev, picard_used


def _stage_row(label: str, x0: np.ndarray, result, params, eta_E: float, initial_full: float, profile: dict[str, Any]) -> dict[str, Any]:
    _set_eta(eta_E)
    row = pilot._row(label, result.x, params, initial_full, result)
    unweighted = np.asarray(_production_residual_base(result.x, params), dtype=float)
    base_final_full = float(row["final_full"])
    augmented_final_full = float(np.linalg.norm(_residual(result.x, params), ord=np.inf))
    reported_final_full = augmented_final_full
    inner_idx = _inner_mdot_row_index(params)
    n = int(params.n_nodes)
    mass_start = inner_idx + 1
    interval_mass = unweighted[mass_start : mass_start + n - 1]
    row.update(
        {
            "eta_E": float(eta_E),
            "base_final_full": base_final_full,
            "augmented_final_full": augmented_final_full,
            "final_full": reported_final_full,
            "accepted_exploratory": bool(reported_final_full <= ACCEPT_TOL),
            "inner_logMdot_residual": float(unweighted[inner_idx]),
            "interval_mass_residual_max": float(np.max(np.abs(interval_mass))) if interval_mass.size else math.nan,
            "peak_mass_residual_rg": profile["peak_mass_residual_rg"],
            "peak_mass_residual": profile["peak_mass_residual"],
            "mass_residual_p90_abs": profile["mass_residual_p90_abs"],
            "source_band_extra_rows_enabled": bool(SOURCE_BAND_EXTRA_ROWS),
            "source_band_extra_audit_only": bool(SOURCE_BAND_EXTRA_AUDIT_ONLY),
            "source_band_finite_volume_mass": bool(SOURCE_BAND_FINITE_VOLUME_MASS),
            "source_band_extra_max": profile.get("source_band_extra_max", math.nan),
            "source_band_extra_radial_max": profile.get("source_band_extra_radial_max", math.nan),
            "source_band_extra_energy_max": profile.get("source_band_extra_energy_max", math.nan),
            "source_band_extra_peak_R_rg": profile.get("source_band_extra_peak_R_rg", math.nan),
            "source_band_extra_active_row_count": profile.get("source_band_extra_active_row_count", 0),
            "local_interval_R": profile["local_interval_R_max"],
            "local_interval_E": profile["local_interval_E_max"],
            "peak_interval_R_rg": profile["peak_interval_R_rg"],
            "peak_interval_R": profile["peak_interval_R"],
            "peak_interval_E_rg": profile["peak_interval_E_rg"],
            "peak_interval_E": profile["peak_interval_E"],
            "seed_initial_full": float(initial_full),
            "seed_initial_weighted_full": float(np.linalg.norm(_residual(x0, params), ord=np.inf)),
            "seed_previous_final_full": float(np.linalg.norm(pilot.residual(x0, params), ord=np.inf)),
            "inner_mdot_weight": float(INNER_MDOT_WEIGHT),
            "use_local_jacobian": bool(USE_LOCAL_JACOBIAN),
            "local_jacobian_step": float(LOCAL_JACOBIAN_STEP),
            "interval_residual_form": str(params.interval_residual_form),
            "integrated_residual_weighting": str(params.integrated_residual_weighting),
            "outer_closure": str(params.outer_closure),
            "outer_robin_chi": float(params.outer_robin_chi),
            "outer_buffer_inner_rg": np.nan if params.outer_buffer_inner_rg is None else float(params.outer_buffer_inner_rg),
            "outer_buffer_radial_weight": float(params.outer_buffer_radial_weight),
            "outer_buffer_energy_weight": float(params.outer_buffer_energy_weight),
            "outer_buffer_boundary_weight": float(params.outer_buffer_boundary_weight),
            "jac_row_norm_median": profile.get("jac_row_norm_median", math.nan),
            "jac_row_norm_max": profile.get("jac_row_norm_max", math.nan),
            "jac_col_norm_median": profile.get("jac_col_norm_median", math.nan),
            "jac_col_norm_max": profile.get("jac_col_norm_max", math.nan),
        }
    )
    return row


def _seed_stage_row(label: str, x0: np.ndarray, params, eta_E: float, initial_full: float, profile: dict[str, Any]) -> dict[str, Any]:
    _set_eta(eta_E)
    row = pilot._row(label, x0, params, initial_full, None)
    unweighted = np.asarray(_production_residual_base(x0, params), dtype=float)
    base_final_full = float(row["final_full"])
    augmented_final_full = float(np.linalg.norm(_residual(x0, params), ord=np.inf))
    reported_final_full = augmented_final_full
    inner_idx = _inner_mdot_row_index(params)
    n = int(params.n_nodes)
    mass_start = inner_idx + 1
    interval_mass = unweighted[mass_start : mass_start + n - 1]
    row.update(
        {
            "eta_E": float(eta_E),
            "base_final_full": base_final_full,
            "augmented_final_full": augmented_final_full,
            "final_full": reported_final_full,
            "accepted_exploratory": bool(reported_final_full <= ACCEPT_TOL),
            "inner_logMdot_residual": float(unweighted[inner_idx]),
            "interval_mass_residual_max": float(np.max(np.abs(interval_mass))) if interval_mass.size else math.nan,
            "peak_mass_residual_rg": profile["peak_mass_residual_rg"],
            "peak_mass_residual": profile["peak_mass_residual"],
            "mass_residual_p90_abs": profile["mass_residual_p90_abs"],
            "source_band_extra_rows_enabled": bool(SOURCE_BAND_EXTRA_ROWS),
            "source_band_extra_audit_only": bool(SOURCE_BAND_EXTRA_AUDIT_ONLY),
            "source_band_finite_volume_mass": bool(SOURCE_BAND_FINITE_VOLUME_MASS),
            "source_band_extra_max": profile.get("source_band_extra_max", math.nan),
            "source_band_extra_radial_max": profile.get("source_band_extra_radial_max", math.nan),
            "source_band_extra_energy_max": profile.get("source_band_extra_energy_max", math.nan),
            "source_band_extra_peak_R_rg": profile.get("source_band_extra_peak_R_rg", math.nan),
            "source_band_extra_active_row_count": profile.get("source_band_extra_active_row_count", 0),
            "local_interval_R": profile["local_interval_R_max"],
            "local_interval_E": profile["local_interval_E_max"],
            "peak_interval_R_rg": profile["peak_interval_R_rg"],
            "peak_interval_R": profile["peak_interval_R"],
            "peak_interval_E_rg": profile["peak_interval_E_rg"],
            "peak_interval_E": profile["peak_interval_E"],
            "seed_initial_full": float(initial_full),
            "seed_initial_weighted_full": float(np.linalg.norm(_residual(x0, params), ord=np.inf)),
            "seed_previous_final_full": float(np.linalg.norm(pilot.residual(x0, params), ord=np.inf)),
            "inner_mdot_weight": float(INNER_MDOT_WEIGHT),
            "use_local_jacobian": bool(USE_LOCAL_JACOBIAN),
            "local_jacobian_step": float(LOCAL_JACOBIAN_STEP),
            "interval_residual_form": str(params.interval_residual_form),
            "integrated_residual_weighting": str(params.integrated_residual_weighting),
            "outer_closure": str(params.outer_closure),
            "outer_robin_chi": float(params.outer_robin_chi),
            "outer_buffer_inner_rg": np.nan if params.outer_buffer_inner_rg is None else float(params.outer_buffer_inner_rg),
            "outer_buffer_radial_weight": float(params.outer_buffer_radial_weight),
            "outer_buffer_energy_weight": float(params.outer_buffer_energy_weight),
            "outer_buffer_boundary_weight": float(params.outer_buffer_boundary_weight),
            "jac_row_norm_median": math.nan,
            "jac_row_norm_max": math.nan,
            "jac_col_norm_median": math.nan,
            "jac_col_norm_max": math.nan,
            "seed_only": True,
            "picard_iters": 0,
            "nfev_total_with_picard": 0,
        }
    )
    return row


def _write_checkpoint(label: str, x: np.ndarray, params, row: dict[str, Any]) -> str:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logu, logT, logMdot, logR_son, lambda0, logR = pilot._unpack(x, params)
    local_params = pilot._local_params(params, logR, logMdot)
    z = pilot._state_vector(logu, logT, logR_son, lambda0)
    safe = _safe_eta_label(float(row["eta_E"]))
    path = CHECKPOINT_DIR / f"{label}_etaE_{safe}_N{int(params.n_nodes)}.npz"
    slopes = local_params.outer_match_log_slopes
    np.savez_compressed(
        path,
        x=np.asarray(x, dtype=float),
        z=np.asarray(z, dtype=float),
        ratio=np.array(local_params.Mdot_g_s / eddington_mdot(local_params.M2_g)),
        R_out_rg=np.array(local_params.R_out_rg),
        n_nodes=np.array(local_params.n_nodes),
        grid_power=np.array(local_params.grid_power),
        custom_grid_xi=np.asarray([] if local_params.custom_grid_xi is None else local_params.custom_grid_xi, dtype=float),
        outer_match_log_slopes=np.asarray([np.nan, np.nan] if slopes is None else slopes, dtype=float),
        wind_energy_multiplier=np.array(row["eta_E"]),
        full=np.array(row["final_full"]),
        accepted=np.array(row["accepted_exploratory"]),
        row_json=np.array(json.dumps(scan.json_safe(row), sort_keys=True)),
    )
    return scan.relative_root_path(path)


def _write_outputs(rows: list[dict[str, Any]], profiles: list[dict[str, Any]]) -> None:
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(scan.json_safe(rows), indent=2, sort_keys=True) + "\n")
    PROFILE_OUTPUT.write_text(json.dumps(scan.json_safe(profiles), indent=2, sort_keys=True) + "\n")
    cols = [
        "label",
        "eta_E",
        "N",
        "seed_initial_full",
        "final_full",
        "base_final_full",
        "augmented_final_full",
        "mass_residual_max",
        "inner_logMdot_residual",
        "interval_mass_residual_max",
        "mass_residual_p90_abs",
        "peak_mass_residual_rg",
        "source_band_extra_rows_enabled",
        "source_band_extra_audit_only",
        "source_band_finite_volume_mass",
        "source_band_extra_max",
        "source_band_extra_radial_max",
        "source_band_extra_energy_max",
        "source_band_extra_peak_R_rg",
        "source_band_extra_active_row_count",
        "local_interval_R",
        "local_interval_E",
        "peak_interval_R_rg",
        "peak_interval_E_rg",
        "interval_E",
        "Mdot_outer_over_inner",
        "f_adv_global",
        "Lrad_LEdd",
        "Rson_rg",
        "nfev",
        "picard_iters",
        "nfev_total_with_picard",
        "seed_only",
        "use_local_jacobian",
        "local_jacobian_step",
        "interval_residual_form",
        "integrated_residual_weighting",
        "outer_closure",
        "outer_robin_chi",
        "residual_remesh_strength",
        "residual_remesh_seed_full",
        "residual_remesh_peak_monitor_rg",
        "residual_remesh_outer_5pct_nodes",
        "inner_relax_outer_rg",
        "inner_relax_initial_full",
        "inner_relax_final_full",
        "inner_relax_initial_selected",
        "inner_relax_final_selected",
        "inner_relax_nfev",
        "outer_relax_min_rg",
        "outer_relax_max_rg",
        "outer_relax_initial_full",
        "outer_relax_final_full",
        "outer_relax_initial_selected",
        "outer_relax_final_selected",
        "outer_relax_nfev",
        "grid_homotopy_applied",
        "grid_homotopy_steps",
        "grid_homotopy_initial_full",
        "grid_homotopy_final_full",
        "grid_homotopy_final_interval_R",
        "grid_homotopy_final_interval_E",
        "grid_homotopy_final_mass",
        "source_microdomain_applied",
        "source_microdomain_old_N",
        "source_microdomain_new_N",
        "source_microdomain_actual_band_nodes",
        "source_microdomain_seed_full",
        "source_microdomain_seed_extra",
        "source_microdomain_final_full",
        "source_microdomain_final_mass",
        "source_microdomain_final_extra",
        "source_micro_correct_applied",
        "source_micro_correct_freeze_edges",
        "source_micro_correct_n_intervals",
        "source_micro_correct_initial_selected",
        "source_micro_correct_final_selected",
        "source_micro_correct_initial_extra",
        "source_micro_correct_final_extra",
        "source_micro_correct_alpha",
        "source_micro_correct_nfev",
        "source_element_refine_applied",
        "source_element_refine_old_N",
        "source_element_refine_new_N",
        "source_element_refine_subdivisions",
        "source_element_refine_halo_intervals",
        "source_element_refine_split_intervals",
        "source_element_refine_inserted_nodes",
        "source_element_refine_remap_method",
        "source_element_refine_mass_seed",
        "source_element_refine_mass_seed_sweeps",
        "source_element_refine_before_full",
        "source_element_refine_before_mass",
        "source_element_refine_before_extra",
        "source_element_refine_seed_full",
        "source_element_refine_seed_mass",
        "source_element_refine_seed_extra",
        "source_domain_correct_applied",
        "source_domain_correct_n_intervals",
        "source_domain_correct_n_variables",
        "source_domain_correct_n_rows",
        "source_domain_correct_initial_selected",
        "source_domain_correct_final_selected",
        "source_domain_correct_initial_full",
        "source_domain_correct_final_full",
        "source_domain_correct_initial_mass",
        "source_domain_correct_final_mass",
        "source_domain_correct_initial_extra",
        "source_domain_correct_final_extra",
        "source_domain_correct_alpha",
        "source_domain_correct_nfev",
        "source_buffer_correct_applied",
        "source_buffer_correct_halo_intervals",
        "source_buffer_correct_n_intervals",
        "source_buffer_correct_n_state_variables",
        "source_buffer_correct_n_delta_variables",
        "source_buffer_correct_n_rows",
        "source_buffer_correct_mass_quadrature",
        "source_buffer_correct_initial_selected",
        "source_buffer_correct_final_selected",
        "source_buffer_correct_initial_state",
        "source_buffer_correct_final_state",
        "source_buffer_correct_initial_integral",
        "source_buffer_correct_final_integral",
        "source_buffer_correct_initial_jump",
        "source_buffer_correct_final_jump",
        "source_buffer_correct_final_peak_jump_R_rg",
        "source_buffer_correct_initial_full",
        "source_buffer_correct_final_full",
        "source_buffer_correct_initial_mass",
        "source_buffer_correct_final_mass",
        "source_buffer_correct_initial_extra",
        "source_buffer_correct_final_extra",
        "source_buffer_correct_delta_min_over_inner",
        "source_buffer_correct_delta_max_over_inner",
        "source_buffer_correct_delta_sum_over_inner",
        "source_buffer_correct_alpha",
        "source_buffer_correct_nfev",
        "band_correct_applied",
        "band_correct_min_rg",
        "band_correct_max_rg",
        "band_correct_initial_full",
        "band_correct_candidate_full",
        "band_correct_final_full",
        "band_correct_final_interval_R",
        "band_correct_final_interval_E",
        "band_correct_final_mass",
        "band_correct_alpha",
        "band_correct_nfev",
        "block_correct_applied",
        "block_correct_peak_kind",
        "block_correct_half_width",
        "block_correct_edge_anchor_weight",
        "block_correct_all_anchor_weight",
        "block_correct_include_outer",
        "block_correct_initial_full",
        "block_correct_candidate_full",
        "block_correct_final_full",
        "block_correct_initial_interval_R",
        "block_correct_final_interval_R",
        "block_correct_final_interval_E",
        "block_correct_final_mass",
        "block_correct_final_outer_omega",
        "block_correct_alpha",
        "block_correct_nfev",
        "success",
        "accepted_exploratory",
        "checkpoint",
    ]
    lines = [
        "# Mdot=5 Local-Mdot Eta Continuation",
        "",
        "Generated by `scripts/run_mdot5_local_mdot_eta_continuation.py`.",
        "",
        f"Anchor: `{scan.relative_root_path(ANCHOR)}`",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(col, "")) for col in cols) + " |")
    lines.extend(
        [
            "",
            "Profiles include interval-local mass residuals, interval energy residuals,",
            "`Qwind/Qvisc`, `Mwind_prime/Mdot`, `Mstream_prime/Mdot`, `Mdot_tilde`,",
            "`s_eff_tilde`, final Jacobian row/column norm arrays, and optional",
            "radial representation / transition-grid / local-block Jacobian audits.",
            "",
            f"Profile JSON: `{scan.relative_root_path(PROFILE_OUTPUT)}`",
        ]
    )
    MD_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not ETA_VALUES:
        raise ValueError("at least one eta_E stage is required")
    if not ANCHOR.exists():
        raise FileNotFoundError(ANCHOR)
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = scan.load_anchor(ANCHOR, fiducial, mdot_edd)
    x, params = _make_seed(anchor_z, anchor_params)
    _set_eta(ETA_VALUES[0])
    startup_info: dict[str, Any] = {}
    if START_X_CHECKPOINT is not None:
        if not START_X_CHECKPOINT.exists():
            raise FileNotFoundError(START_X_CHECKPOINT)
        data = np.load(START_X_CHECKPOINT)
        if "x" not in data:
            raise ValueError(f"{START_X_CHECKPOINT} does not contain a local-Mdot x vector")
        x = np.asarray(data["x"], dtype=float)
        expected = 3 * int(params.n_nodes) + 2
        if x.size != expected:
            if (x.size - 2) % 3 != 0:
                raise ValueError(f"start x has incompatible size {x.size}")
            old_n = int((x.size - 2) // 3)
            _old_z, old_params = _state_and_params_for_n(anchor_z, anchor_params, old_n)
            old_params = _restore_checkpoint_params(old_params, data)
            params = _restore_checkpoint_params(params, data)
            if REMAP_METHOD in NESTED_REMAP_METHODS:
                params = _node_preserving_refined_params(x, old_params, params)
            if GRID_HOMOTOPY_STEPS > 0:
                x, params, startup_info = _grid_homotopy_seed(x, old_params, params, ETA_VALUES[0])
            else:
                x = _remap_local_x_to_params(x, old_params, params)
        else:
            params = _restore_checkpoint_params(params, data)
    if SOURCE_MICRO_DOMAIN:
        x, params, micro_info = _source_microdomain_seed(x, params, ETA_VALUES[0])
        startup_info.update(micro_info)
    if SOURCE_ELEMENT_REFINE:
        x, params, element_info = _source_element_refine_seed(x, params, ETA_VALUES[0])
        startup_info.update(element_info)
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []

    for stage_index, eta_E in enumerate(ETA_VALUES):
        label = f"stage_{stage_index:02d}"
        _set_eta(eta_E)
        remesh_info: dict[str, Any] = {}
        if RESIDUAL_REMESH_STRENGTH > 0.0:
            x, params, remesh_info = _residual_remesh(x, params, eta_E)
        inner_relax_info: dict[str, Any] = {}
        if INNER_RELAX_OUTER_RG > 0.0:
            x, inner_relax_info = _inner_window_relax(x, params, eta_E)
        outer_relax_info: dict[str, Any] = {}
        if OUTER_RELAX_MIN_RG > 0.0 and OUTER_RELAX_MAX_RG > OUTER_RELAX_MIN_RG:
            x, outer_relax_info = _outer_band_relax(x, params, eta_E)
        band_correct_info: dict[str, Any] = {}
        if BAND_CORRECT:
            x, band_correct_info = _reduced_band_correct(x, params, eta_E)
        source_domain_info: dict[str, Any] = {}
        if SOURCE_DOMAIN_CORRECT:
            x, source_domain_info = _source_domain_correct(x, params, eta_E)
        source_buffer_info: dict[str, Any] = {}
        if SOURCE_BUFFER_CORRECT:
            x, source_buffer_info = _source_buffer_correct(x, params, eta_E)
        block_correct_info: dict[str, Any] = {}
        if BLOCK_CORRECT:
            x, block_correct_info = _coupled_block_correct(x, params, eta_E)
        initial_full = float(np.linalg.norm(_residual(x, params), ord=np.inf))
        initial_profile = _profile(f"{label}_initial", x, params, eta_E)
        print(f"{label} eta_E={eta_E:.8g} initial_full={initial_full:.3e}", flush=True)
        if SEED_ONLY:
            seed_profile = _profile(f"{label}_seed", x, params, eta_E)
            row = _seed_stage_row(label, x, params, eta_E, initial_full, seed_profile)
            row.update(remesh_info)
            row.update(inner_relax_info)
            row.update(outer_relax_info)
            row.update(band_correct_info)
            row.update(source_domain_info)
            row.update(source_buffer_info)
            row.update(block_correct_info)
            row.update(startup_info)
            row["checkpoint"] = _write_checkpoint(label, x, params, row)
            rows.append(row)
            profiles.extend([initial_profile, seed_profile])
            _write_outputs(rows, profiles)
            print(
                f"{label} seed_only final={row['final_full']:.3e} mass={row['mass_residual_max']:.3e} "
                f"peakM={row['peak_mass_residual_rg']:.3f}rg accepted={row['accepted_exploratory']}",
                flush=True,
            )
            continue
        result, stage_params, nfev_total, picard_used = _solve_with_picard(x, params, eta_E)
        final_jac_norms = _jac_norms(result, stage_params)
        final_profile = _profile(f"{label}_final", result.x, stage_params, eta_E, final_jac_norms)
        row = _stage_row(label, x, result, stage_params, eta_E, initial_full, final_profile)
        row.update(remesh_info)
        row.update(inner_relax_info)
        row.update(outer_relax_info)
        row.update(band_correct_info)
        row.update(source_domain_info)
        row.update(source_buffer_info)
        row.update(block_correct_info)
        row.update(startup_info)
        row["picard_iters"] = int(picard_used)
        row["nfev_total_with_picard"] = int(nfev_total)
        row["checkpoint"] = _write_checkpoint(label, result.x, stage_params, row)
        rows.append(row)
        profiles.extend([initial_profile, final_profile])
        _write_outputs(rows, profiles)
        print(
            f"{label} final={row['final_full']:.3e} mass={row['mass_residual_max']:.3e} "
            f"peakM={row['peak_mass_residual_rg']:.3f}rg nfev={row['nfev']} "
            f"accepted={row['accepted_exploratory']}",
            flush=True,
        )
        x = np.asarray(result.x, dtype=float)
        params = stage_params

    print(f"wrote {scan.relative_root_path(JSON_OUTPUT)}", flush=True)
    print(f"wrote {scan.relative_root_path(MD_OUTPUT)}", flush=True)
    print(f"wrote {scan.relative_root_path(PROFILE_OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
