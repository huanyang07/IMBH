"""Globalize the certified Mdot=5 K13 phase-space DAE segment.

This driver builds a composite BVP. Ordinary global/source rows remain active
outside the phase interval, while the old lnR rows and their inactive interior
global states are removed inside the phase segment.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_CHECKPOINT = (
    ROOT
    / "results/canonical/phase_dae_entry_N164/state.npz"
)

# Reproduce the exact source/phase context used by the certified K13 run.
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_N_NODES", "164")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_ETA_VALUES", "98.125")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_START_X_CHECKPOINT", str(DEFAULT_CHECKPOINT))
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_CORE_ONLY", "1")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_RELEASE_HALO", "12")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION", "1")
os.environ.setdefault(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SOURCE_MODE",
    "lobatto_source_element",
)
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_HSFV_PRODUCTION_EVALUATE_ONLY", "1")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT", "1")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_INTERVALS", "34")
os.environ.setdefault(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_TANGENT_CONSISTENCY_WEIGHT",
    "0",
)
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT", "1")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT_INTERVALS", "13")
os.environ.setdefault(
    "IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT_KINEMATIC_SCHEME",
    "simpson",
)
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT_MODE", "evaluate")

import run_mdot5_local_mdot_eta_continuation as model  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_state,
    stream_source_prime,
    stream_torque_specific_l_and_derivative,
)
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402


ETA_E = float(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_ETA_E", "98.125"))
STAGES = tuple(
    piece.strip().lower()
    for piece in os.environ.get(
        "IMBH_MDOT5_GLOBAL_PHASE_STAGES",
        "evaluate,exterior,local,coupled",
    ).split(",")
    if piece.strip()
)
MAX_NFEV_EXTERIOR = int(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_MAX_NFEV_EXTERIOR", "24"))
MAX_NFEV_LOCAL = int(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_MAX_NFEV_LOCAL", "40"))
MAX_NFEV_COUPLED = int(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_MAX_NFEV_COUPLED", "60"))
INTERFACE_WEIGHT = float(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_INTERFACE_WEIGHT", "100"))
INTERFACE_ENERGY_WEIGHT = float(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_INTERFACE_ENERGY_WEIGHT", "10"))
SONIC_WEIGHT = float(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_SONIC_WEIGHT", "30"))
OUTER_WEIGHT = float(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_OUTER_WEIGHT", "10"))
GLOBAL_MASS_WEIGHT = float(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_MASS_WEIGHT", "10"))
GLOBAL_STATE_TRUST = float(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_GLOBAL_STATE_TRUST", "0.02"))
GLOBAL_F_TRUST = float(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_GLOBAL_F_TRUST", "0.02"))
INTERFACE_HALO = int(os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_INTERFACE_HALO", "3"))
RANK_AUDIT = os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_RANK_AUDIT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OUTPUT_STEM = os.environ.get(
    "IMBH_MDOT5_GLOBAL_PHASE_OUTPUT_STEM",
    "m5_eta_global_phase_dae_k13_98p125_N164",
)
RESUME_CHECKPOINT_RAW = os.environ.get("IMBH_MDOT5_GLOBAL_PHASE_RESUME_CHECKPOINT", "").strip()
TABLE_PATH = ROOT / "outputs" / "tables" / f"{OUTPUT_STEM}.json"
PROFILE_PATH = ROOT / "outputs" / "tables" / f"{OUTPUT_STEM}_profiles.json"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / OUTPUT_STEM
NOTE_PATH = ROOT / "docs/reports/current/CODEX_MDOT5_GLOBAL_PHASE_DAE_PRODUCTION_RESULTS.md"


def _load_problem() -> tuple[np.ndarray, Any, dict[str, Any], np.ndarray, dict[str, Any]]:
    checkpoint = Path(os.environ["IMBH_MDOT5_LOCAL_MDOT_ETA_START_X_CHECKPOINT"]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    fiducial = FiducialParams()
    anchor_z, anchor_params = model.scan.load_anchor(
        model.ANCHOR,
        fiducial,
        eddington_mdot(fiducial.M2_g),
    )
    _x_seed, params = model._make_seed(anchor_z, anchor_params)
    with np.load(checkpoint) as data:
        x_log = np.asarray(data["x"], dtype=float)
        params = model._restore_checkpoint_params(params, data)
    expected = 3 * int(params.n_nodes) + 2
    if x_log.size != expected:
        raise ValueError(f"checkpoint x has size {x_log.size}, expected {expected}")
    model.START_X_CHECKPOINT = checkpoint
    model._set_eta(ETA_E)
    context = model._global_flux_hsfv_context(x_log, params)
    if not context.get("ok", False):
        raise RuntimeError(str(context.get("reason", "invalid source context")))
    aux = np.asarray(context["aux0"], dtype=float)
    source_data = model._global_flux_hsfv_source_data(x_log, params, context, aux)
    context["source_row_count"] = int(np.asarray(source_data.get("rows", []), dtype=float).size)
    fv_data = model._global_flux_hsfv_fv_control_data(x_log, params, context)
    context["fv_control_row_count"] = int(np.asarray(fv_data.get("rows", []), dtype=float).size)
    context["fv_control_interval_indices"] = np.asarray(fv_data.get("interval", []), dtype=int)
    phase = model._global_flux_phase_dae_seed(x_log, params, context, aux)
    if not phase.get("ok", False):
        raise RuntimeError(str(phase.get("reason", "invalid phase seed")))
    if int(np.asarray(phase["interval_indices"]).size) != 13:
        raise RuntimeError("production seed is not the certified K13 segment")
    return x_log, params, context, aux, phase


def _phase_pack(z: np.ndarray, p: np.ndarray, p_mid: np.ndarray, ds: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(z, dtype=float).ravel(),
            np.asarray(p, dtype=float).ravel(),
            np.asarray(p_mid, dtype=float).ravel(),
            np.log(np.maximum(np.asarray(ds, dtype=float), 1.0e-12)),
        ]
    )


def _phase_unpack(vector: np.ndarray, node_count: int, interval_count: int) -> tuple[np.ndarray, ...]:
    vector = np.asarray(vector, dtype=float)
    z_size = 4 * node_count
    p_size = 4 * node_count
    p_mid_size = 4 * interval_count
    z = vector[:z_size].reshape(node_count, 4)
    p = vector[z_size : z_size + p_size].reshape(node_count, 4)
    cursor = z_size + p_size
    p_mid = vector[cursor : cursor + p_mid_size].reshape(interval_count, 4)
    cursor += p_mid_size
    ds = np.exp(vector[cursor : cursor + interval_count])
    return z, p, p_mid, ds


def _phase_bounds(phase: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(phase["z_seed"], dtype=float)
    p = np.asarray(phase["p_seed"], dtype=float)
    p_mid = np.asarray(phase["p_mid_seed"], dtype=float)
    ds = np.maximum(np.asarray(phase["ds_seed"], dtype=float), 1.0e-12)
    z_lo = z.copy()
    z_hi = z.copy()
    z_lo[:, :2] -= abs(float(model.GLOBAL_FLUX_PHASE_DAE_SEGMENT_STATE_TRUST))
    z_hi[:, :2] += abs(float(model.GLOBAL_FLUX_PHASE_DAE_SEGMENT_STATE_TRUST))
    z_lo[:, 2] = np.maximum(z[:, 2] - abs(float(model.GLOBAL_FLUX_PHASE_DAE_SEGMENT_F_TRUST)), 1.0e-12)
    z_hi[:, 2] = z[:, 2] + abs(float(model.GLOBAL_FLUX_PHASE_DAE_SEGMENT_F_TRUST))
    log_r_trust = abs(float(model.GLOBAL_FLUX_PHASE_DAE_SEGMENT_LOGR_TRUST))
    z_lo[:, 3] -= log_r_trust
    z_hi[:, 3] += log_r_trust
    seed_log_r = np.asarray(z[:, 3], dtype=float)
    for pos in range(z.shape[0]):
        if pos > 0:
            z_lo[pos, 3] = max(z_lo[pos, 3], 0.5 * (seed_log_r[pos - 1] + seed_log_r[pos]) + 1.0e-10)
        if pos < z.shape[0] - 1:
            z_hi[pos, 3] = min(z_hi[pos, 3], 0.5 * (seed_log_r[pos] + seed_log_r[pos + 1]) - 1.0e-10)
    lower = np.concatenate(
        [
            z_lo.ravel(),
            np.full(p.size, -10.0),
            np.full(p_mid.size, -10.0),
            np.log(ds) - 5.0,
        ]
    )
    upper = np.concatenate(
        [
            z_hi.ravel(),
            np.full(p.size, 10.0),
            np.full(p_mid.size, 10.0),
            np.log(ds) + 5.0,
        ]
    )
    return lower, upper


def _source_row_metadata(
    source_data: dict[str, Any],
    context: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(source_data.get("rows", []), dtype=float)
    intervals = np.asarray(source_data.get("interval", []), dtype=int)
    if intervals.size != rows.size:
        context_intervals = np.asarray(context.get("interval_indices", []), dtype=int)
        if context_intervals.size and rows.size % context_intervals.size == 0:
            intervals = np.repeat(context_intervals, rows.size // context_intervals.size)
        else:
            intervals = np.full(rows.size, -1, dtype=int)
    components = np.asarray(source_data.get("component", []), dtype=object).astype(str)
    if components.size != rows.size:
        groups = np.asarray(source_data.get("groups", []), dtype=object).astype(str)
        inferred = []
        for group in groups:
            if "radial" in group:
                inferred.append("radial")
            elif "energy" in group:
                inferred.append("energy")
            elif "mass" in group or "Fprime" in group:
                inferred.append("mass")
            else:
                inferred.append("compatibility")
        components = np.asarray(inferred, dtype=object)
    return intervals, components


def _ordinary_keep_masks(
    params,
    context: dict[str, Any],
    source_data: dict[str, Any],
    phase_intervals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(params.n_nodes)
    base_keep = np.ones(3 * n + 2, dtype=bool)
    source_intervals = np.asarray(context["interval_indices"], dtype=int)
    for idx in source_intervals:
        base_keep[2 * int(idx)] = False
        base_keep[2 * int(idx) + 1] = False
    mass_start = model._inner_mdot_row_index(params) + 1
    for idx in np.asarray(phase_intervals, dtype=int):
        base_keep[mass_start + int(idx)] = False
    row_intervals, _components = _source_row_metadata(source_data, context)
    phase_set = set(int(value) for value in np.asarray(phase_intervals, dtype=int))
    source_keep = np.asarray([int(value) not in phase_set for value in row_intervals], dtype=bool)
    return base_keep, source_keep


def _phase_interval_energy(
    z_l: np.ndarray,
    z_r: np.ndarray,
    p_l: np.ndarray,
    p_m: np.ndarray,
    p_r: np.ndarray,
    ds: float,
    params,
    lambda0: float,
) -> dict[str, float]:
    ds = max(float(ds), 1.0e-12)
    z_mid = 0.5 * (np.asarray(z_l, dtype=float) + np.asarray(z_r, dtype=float))
    z_mid += (ds / 8.0) * (np.asarray(p_l, dtype=float) - np.asarray(p_r, dtype=float))
    numerator = 0.0
    denominator = 0.0
    for z_q, p_q, coefficient in (
        (z_l, p_l, 1.0),
        (z_mid, p_m, 4.0),
        (z_r, p_r, 1.0),
    ):
        p_r_q = float(p_q[3])
        if abs(p_r_q) <= 1.0e-12:
            return {"numerator": math.nan, "denominator": math.nan, "residual": 1.0e6}
        F_q = max(float(z_q[2]), 1.0e-300)
        dlogF_dx = float(p_q[2]) / (F_q * p_r_q)
        local = model._local_params_with_point_mdot(
            params,
            float(z_q[3]),
            math.log(F_q * max(float(params.Mdot_g_s), 1.0e-300)),
            dlogF_dx,
        )
        terms = model._energy_terms_at(
            float(z_q[3]),
            np.asarray(z_q[:2], dtype=float),
            np.asarray(p_q[:2], dtype=float) / p_r_q,
            float(lambda0),
            local,
        )
        weight = ds * float(coefficient) / 6.0 * p_r_q
        numerator += weight * float(terms["area"]) * float(terms["raw"])
        denominator += abs(weight) * float(terms["area"]) * float(terms["denom"])
    return {
        "numerator": float(numerator),
        "denominator": float(denominator),
        "residual": float(numerator / max(abs(denominator), 1.0e-300)),
    }


def _interface_energy_data(
    x_log: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    p_mid: np.ndarray,
    ds: np.ndarray,
    phase_intervals: np.ndarray,
    params,
) -> dict[str, Any]:
    logu, logT, log_mdot, _log_r_son, lambda0, log_r = model.pilot._unpack(x_log, params)
    local = model.pilot._local_params(params, log_r, log_mdot)
    first = int(phase_intervals[0])
    last = int(phase_intervals[-1])
    left_global = model._source_interface_energy_terms_from_unpacked(
        logu, logT, log_mdot, log_r, float(lambda0), local, first - 1
    )
    right_global = model._source_interface_energy_terms_from_unpacked(
        logu, logT, log_mdot, log_r, float(lambda0), local, last + 1
    )
    left_phase = _phase_interval_energy(z[0], z[1], p[0], p_mid[0], p[1], ds[0], params, float(lambda0))
    right_phase = _phase_interval_energy(
        z[-2], z[-1], p[-2], p_mid[-1], p[-1], ds[-1], params, float(lambda0)
    )

    def combine(global_part: dict[str, float], phase_part: dict[str, float]) -> float:
        numerator = float(global_part.get("numerator", math.nan)) + float(phase_part.get("numerator", math.nan))
        denominator = abs(float(global_part.get("denominator", math.nan))) + abs(
            float(phase_part.get("denominator", math.nan))
        )
        return float(numerator / max(denominator, 1.0e-300))

    rows = np.asarray([combine(left_global, left_phase), combine(right_global, right_phase)], dtype=float)
    return {
        "rows": rows,
        "left_global": left_global,
        "left_phase": left_phase,
        "right_phase": right_phase,
        "right_global": right_global,
        "max": float(np.nanmax(np.abs(rows))),
    }


def _phase_mass_profile(
    z: np.ndarray,
    p: np.ndarray,
    p_mid: np.ndarray,
    ds: np.ndarray,
    phase_intervals: np.ndarray,
    params,
    lambda0: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for pos, interval in enumerate(np.asarray(phase_intervals, dtype=int)):
        ds_i = max(float(ds[pos]), 1.0e-12)
        z_mid = 0.5 * (z[pos] + z[pos + 1]) + (ds_i / 8.0) * (p[pos] - p[pos + 1])
        targets: list[float] = []
        for z_q, p_q in ((z[pos], p[pos]), (z_mid, p_mid[pos]), (z[pos + 1], p[pos + 1])):
            point = model._global_flux_phase_dae_point_data(z_q, p_q, params, float(lambda0))
            targets.append(float(p_q[2]) - float(point["fprime"]))
        integral = ds_i / 6.0 * (targets[0] + 4.0 * targets[1] + targets[2])
        defect = float(z[pos + 1, 2] - z[pos, 2] - integral)
        rows.append(
            {
                "interval": int(interval),
                "R_mid_rg": float(np.exp(0.5 * (z[pos, 3] + z[pos + 1, 3])) / params.r_g),
                "FV_mass": defect,
            }
        )
    return rows


def _phase_angular_profile(
    z: np.ndarray,
    p: np.ndarray,
    p_mid: np.ndarray,
    ds: np.ndarray,
    phase_intervals: np.ndarray,
    params,
    lambda0: float,
) -> list[dict[str, float]]:
    inner_scale = max(float(params.Mdot_g_s), 1.0e-300)
    output: list[dict[str, float]] = []

    def point(z_q: np.ndarray, p_q: np.ndarray) -> tuple[float, float, float]:
        p_r = float(p_q[3])
        F_q = max(float(z_q[2]), 1.0e-300)
        dlogF_dx = float(p_q[2]) / max(F_q * p_r, 1.0e-300)
        local = model._local_params_with_point_mdot(
            params,
            float(z_q[3]),
            math.log(F_q * inner_scale),
            dlogF_dx,
        )
        y = np.asarray(z_q[:2], dtype=float)
        g = np.asarray(p_q[:2], dtype=float) / p_r
        state = algebraic_state(float(z_q[3]), float(y[0]), float(y[1]), float(lambda0), local)
        wind_prime = model._safe_wind_prime(float(z_q[3]), y, g, float(lambda0), local)
        if not np.isfinite(wind_prime):
            wind_prime = 0.0
        source_prime = stream_source_prime(float(z_q[3]), local)
        _stream_l, stream_dl_dx = stream_torque_specific_l_and_derivative(float(z_q[3]), local)
        mdot = F_q * inner_scale
        torque = 2.0 * np.pi * state.R**2 * state.W
        flux = mdot * state.l - torque
        source = float(wind_prime) * state.l - float(source_prime) * state.l + mdot * float(stream_dl_dx)
        return float(flux), float(source), float(state.l_K)

    for pos, interval in enumerate(np.asarray(phase_intervals, dtype=int)):
        ds_i = max(float(ds[pos]), 1.0e-12)
        z_mid = 0.5 * (z[pos] + z[pos + 1]) + (ds_i / 8.0) * (p[pos] - p[pos + 1])
        left = point(z[pos], p[pos])
        middle = point(z_mid, p_mid[pos])
        right = point(z[pos + 1], p[pos + 1])
        source_integral = ds_i / 6.0 * (
            p[pos, 3] * left[1] + 4.0 * p_mid[pos, 3] * middle[1] + p[pos + 1, 3] * right[1]
        )
        scale = max(inner_scale * abs(float(middle[2])), 1.0e-300)
        defect = float((right[0] - left[0] - source_integral) / scale)
        output.append(
            {
                "interval": int(interval),
                "R_mid_rg": float(np.exp(float(z_mid[3])) / params.r_g),
                "angular_FV": defect,
            }
        )
    return output


def _base_group(row: int, params) -> str:
    n = int(params.n_nodes)
    interval_rows = 2 * (n - 1)
    if row < interval_rows:
        return "outside_radial" if row % 2 == 0 else "outside_energy"
    if row < interval_rows + 2:
        return "outer"
    if row < interval_rows + 4:
        return "sonic"
    if row == model._inner_mdot_row_index(params):
        return "inner_mass"
    return "outside_fv_mass"


def _ordinary_production_sparsity(
    params,
    context: dict[str, Any],
    base_keep: np.ndarray,
    source_keep: np.ndarray,
    source_row_count: int,
    fv_row_count: int,
):
    from scipy.sparse import hstack, lil_matrix, vstack

    n = int(params.n_nodes)
    base_size = 3 * n + 2
    aux_size = int(np.asarray(context["aux0"], dtype=float).size)
    base = model.pilot._sparsity(params).tocsr()[np.asarray(base_keep, dtype=bool), :]
    base = hstack([base, lil_matrix((base.shape[0], aux_size), dtype=int)], format="csr")

    intervals = np.asarray(context["interval_indices"], dtype=int)
    interval_count = int(intervals.size)
    source = lil_matrix((source_row_count, base_size + aux_size), dtype=int)
    if interval_count and source_row_count % interval_count == 0:
        rows_per_interval = source_row_count // interval_count
        midpoint_offset = base_size
        f_mid_offset = midpoint_offset + 2 * interval_count
        for interval_pos, idx_value in enumerate(intervals):
            idx = int(idx_value)
            columns = [
                idx,
                idx + 1,
                n + idx,
                n + idx + 1,
                2 * n + idx,
                2 * n + idx + 1,
                3 * n + 1,
                midpoint_offset + 2 * interval_pos,
                midpoint_offset + 2 * interval_pos + 1,
                f_mid_offset + interval_pos,
            ]
            row_slice = slice(rows_per_interval * interval_pos, rows_per_interval * (interval_pos + 1))
            source[row_slice, columns] = 1
    else:
        source[:, :] = 1
    source = source.tocsr()[np.asarray(source_keep, dtype=bool), :]

    blocks = [base, source]
    if fv_row_count > 0:
        fv = model._global_flux_hsfv_fv_control_sparsity(params, context, aux_size)
        blocks.append(fv.tocsr())
    return vstack(blocks, format="csr")


def _composite_setup(
    x_log: np.ndarray,
    params,
    context: dict[str, Any],
    aux: np.ndarray,
    phase: dict[str, Any],
) -> dict[str, Any]:
    from scipy.sparse import hstack

    x_flux = model._log_x_to_flux_x(x_log, params)
    z = np.asarray(phase["z_seed"], dtype=float)
    p = np.asarray(phase["p_seed"], dtype=float)
    p_mid = np.asarray(phase["p_mid_seed"], dtype=float)
    ds = np.asarray(phase["ds_seed"], dtype=float)
    phase_vector = _phase_pack(z, p, p_mid, ds)
    full0 = np.concatenate([x_flux, np.asarray(aux, dtype=float), phase_vector])
    base_size = int(x_flux.size)
    aux_size = int(np.asarray(aux).size)
    phase_offset = base_size + aux_size
    phase_intervals = np.asarray(phase["interval_indices"], dtype=int)
    phase_nodes = np.asarray(phase["node_indices"], dtype=int)
    source0 = model._global_flux_hsfv_source_data(x_log, params, context, aux)
    base_keep, source_keep = _ordinary_keep_masks(params, context, source0, phase_intervals)
    fv0 = model._global_flux_hsfv_fv_control_data(x_log, params, context)
    fv_count = int(np.asarray(fv0.get("rows", []), dtype=float).size)
    ordinary_keep = np.concatenate([base_keep, source_keep, np.ones(fv_count, dtype=bool)])
    ordinary_sparsity = _ordinary_production_sparsity(
        params,
        context,
        base_keep,
        source_keep,
        int(np.asarray(source0.get("rows", []), dtype=float).size),
        fv_count,
    )
    aux_used = np.asarray(ordinary_sparsity[:, base_size:].getnnz(axis=0) > 0).reshape(-1)
    global_used = np.asarray(ordinary_sparsity[:, :base_size].getnnz(axis=0) > 0).reshape(-1)

    lower_flux, upper_flux = model._flux_production_bounds(params)
    lower_flux = np.maximum(lower_flux, x_flux - GLOBAL_STATE_TRUST)
    upper_flux = np.minimum(upper_flux, x_flux + GLOBAL_STATE_TRUST)
    n = int(params.n_nodes)
    lower_flux[2 * n : 3 * n] = np.maximum(
        lower_flux[2 * n : 3 * n], x_flux[2 * n : 3 * n] - GLOBAL_F_TRUST
    )
    upper_flux[2 * n : 3 * n] = np.minimum(
        upper_flux[2 * n : 3 * n], x_flux[2 * n : 3 * n] + GLOBAL_F_TRUST
    )
    phase_lower, phase_upper = _phase_bounds(phase)
    lower = np.concatenate([lower_flux, np.asarray(context["aux_lower"], dtype=float), phase_lower])
    upper = np.concatenate([upper_flux, np.asarray(context["aux_upper"], dtype=float), phase_upper])
    if np.any(lower >= upper):
        raise RuntimeError("composite bounds collapsed")

    # The ordinary global states strictly inside the phase block are retained
    # only as immutable output placeholders; they are never solver variables.
    interior_nodes = phase_nodes[1:-1]
    for node in interior_nodes:
        for offset in (0, n, 2 * n):
            global_used[offset + int(node)] = False
    endpoint_columns: list[int] = []
    for node in (int(phase_nodes[0]), int(phase_nodes[-1])):
        endpoint_columns.extend([node, n + node, 2 * n + node])
    endpoint_columns.extend([3 * n, 3 * n + 1])
    global_used[np.asarray(endpoint_columns, dtype=int)] = True

    extended_ordinary_sparsity = hstack(
        [ordinary_sparsity, np.zeros((ordinary_sparsity.shape[0], phase_vector.size), dtype=int)],
        format="csr",
    )
    return {
        "params": params,
        "context": context,
        "base_size": base_size,
        "aux_size": aux_size,
        "phase_offset": phase_offset,
        "phase_size": int(phase_vector.size),
        "phase_node_count": int(z.shape[0]),
        "phase_interval_count": int(ds.size),
        "phase_intervals": phase_intervals,
        "phase_nodes": phase_nodes,
        "mesh_target": np.diff(np.log(np.maximum(ds, 1.0e-300))),
        "base_keep": base_keep,
        "source_keep": source_keep,
        "ordinary_keep": ordinary_keep,
        "ordinary_sparsity": extended_ordinary_sparsity,
        "global_used": global_used,
        "aux_used": aux_used,
        "full0": full0,
        "lower": lower,
        "upper": upper,
        "immutable_x_flux": np.asarray(x_flux, dtype=float),
    }


def _unpack_full(full: np.ndarray, setup: dict[str, Any]) -> tuple[np.ndarray, ...]:
    base_size = int(setup["base_size"])
    aux_size = int(setup["aux_size"])
    phase_offset = int(setup["phase_offset"])
    x_flux = np.asarray(full[:base_size], dtype=float)
    aux = np.asarray(full[base_size:phase_offset], dtype=float)
    phase_vector = np.asarray(full[phase_offset:], dtype=float)
    z, p, p_mid, ds = _phase_unpack(
        phase_vector,
        int(setup["phase_node_count"]),
        int(setup["phase_interval_count"]),
    )
    return x_flux, aux, z, p, p_mid, ds


def _composite_data(full: np.ndarray, setup: dict[str, Any], *, audit: bool = True) -> dict[str, Any]:
    params = setup["params"]
    context = setup["context"]
    phase_intervals = np.asarray(setup["phase_intervals"], dtype=int)
    phase_nodes = np.asarray(setup["phase_nodes"], dtype=int)
    x_flux, aux, z, p, p_mid, ds = _unpack_full(full, setup)
    x_log = model._flux_x_to_log_x(x_flux, params)
    logu, logT, _log_mdot, _log_r_son, lambda0, log_r = model.pilot._unpack(x_log, params)

    base_all = model._production_residual_flux_base(
        x_flux,
        params,
        source_guard_context=None,
        include_source_guards=False,
        skip_source_dynamics_override=True,
    )
    source_data = model._global_flux_hsfv_source_data(x_log, params, context, aux)
    source_all = np.asarray(source_data.get("rows", []), dtype=float)
    source_raw_all = np.asarray(source_data.get("raw_rows", source_all), dtype=float)
    fv_data = model._global_flux_hsfv_fv_control_data(x_log, params, context)
    fv_all = np.asarray(fv_data.get("rows", []), dtype=float)
    base_keep = np.asarray(setup["base_keep"], dtype=bool)
    source_keep = np.asarray(setup["source_keep"], dtype=bool)

    base_weighted = np.asarray(base_all, dtype=float).copy()
    interval_row_count = 2 * (int(params.n_nodes) - 1)
    base_weighted[interval_row_count : interval_row_count + 2] *= OUTER_WEIGHT
    base_weighted[interval_row_count + 2 : interval_row_count + 4] *= SONIC_WEIGHT
    mass_start_weight = model._inner_mdot_row_index(params) + 1
    base_weighted[mass_start_weight : mass_start_weight + int(params.n_nodes) - 1] *= GLOBAL_MASS_WEIGHT
    ordinary_rows = np.concatenate([base_weighted[base_keep], source_all[source_keep], fv_all])
    ordinary_raw = np.concatenate([base_all[base_keep], source_raw_all[source_keep], fv_all])
    ordinary_groups = [
        _base_group(int(row), params) for row in np.nonzero(base_keep)[0]
    ]
    ordinary_groups.extend(
        f"source_{group}" for group in np.asarray(source_data.get("groups", []), dtype=object)[source_keep]
    )
    ordinary_groups.extend(["fv_control"] * int(fv_all.size))

    phase_data = model._global_flux_phase_dae_segment_data(
        z,
        p,
        p_mid,
        ds,
        params,
        float(lambda0),
        phase_intervals,
        np.asarray(setup["mesh_target"], dtype=float),
    )
    phase_rows = np.asarray(phase_data.get("rows", []), dtype=float)
    phase_raw = np.asarray(phase_data.get("raw_rows", []), dtype=float)
    phase_groups = [str(value) for value in np.asarray(phase_data.get("groups", []), dtype=object)]

    endpoint_raw: list[float] = []
    endpoint_groups: list[str] = []
    for local_pos, global_node, side in (
        (0, int(phase_nodes[0]), "left"),
        (-1, int(phase_nodes[-1]), "right"),
    ):
        global_state = np.asarray(
            [logu[global_node], logT[global_node], x_flux[2 * int(params.n_nodes) + global_node], log_r[global_node]],
            dtype=float,
        )
        for component, value in zip(("logu", "logT", "F", "logR"), z[local_pos] - global_state):
            endpoint_raw.append(float(value))
            endpoint_groups.append(f"interface_{side}_{component}")
    endpoint_raw_array = np.asarray(endpoint_raw, dtype=float)
    endpoint_rows = INTERFACE_WEIGHT * endpoint_raw_array

    interface_energy = _interface_energy_data(x_log, z, p, p_mid, ds, phase_intervals, params)
    interface_energy_raw = np.asarray(interface_energy["rows"], dtype=float)
    interface_energy_rows = INTERFACE_ENERGY_WEIGHT * interface_energy_raw

    rows = np.concatenate([ordinary_rows, phase_rows, endpoint_rows, interface_energy_rows])
    raw = np.concatenate([ordinary_raw, phase_raw, endpoint_raw_array, interface_energy_raw])
    groups = [
        *ordinary_groups,
        *phase_groups,
        *endpoint_groups,
        "interface_energy_left",
        "interface_energy_right",
    ]
    if rows.size != len(groups) or raw.size != rows.size:
        raise RuntimeError("composite residual metadata size mismatch")
    if not audit:
        return {"rows": rows}

    phase_summary = dict(phase_data.get("summary", {}))
    source_intervals, source_components = _source_row_metadata(source_data, context)
    kept_source_raw = source_raw_all[source_keep]
    kept_components = source_components[source_keep] if source_components.size == source_keep.size else np.asarray([])
    outside_radial_values = [
        abs(float(base_all[2 * idx]))
        for idx in range(int(params.n_nodes) - 1)
        if base_keep[2 * idx]
    ]
    outside_energy_values = [
        abs(float(base_all[2 * idx + 1]))
        for idx in range(int(params.n_nodes) - 1)
        if base_keep[2 * idx + 1]
    ]
    if kept_components.size:
        outside_radial_values.extend(np.abs(kept_source_raw[kept_components == "radial"]).tolist())
        outside_energy_values.extend(np.abs(kept_source_raw[kept_components == "energy"]).tolist())

    mass_start = model._inner_mdot_row_index(params) + 1
    phase_set = set(int(value) for value in phase_intervals)
    global_mass_profile = []
    for idx in range(int(params.n_nodes) - 1):
        if idx in phase_set:
            continue
        global_mass_profile.append(
            {
                "interval": idx,
                "R_mid_rg": float(np.exp(0.5 * (log_r[idx] + log_r[idx + 1])) / params.r_g),
                "FV_mass": float(base_all[mass_start + idx]) / max(float(model.GLOBAL_FLUX_PRODUCTION_MASS_WEIGHT), 1.0e-300),
            }
        )
    phase_mass_profile = _phase_mass_profile(z, p, p_mid, ds, phase_intervals, params, float(lambda0))
    combined_mass_profile = sorted(global_mass_profile + phase_mass_profile, key=lambda item: int(item["interval"]))
    mass_values = np.asarray([item["FV_mass"] for item in combined_mass_profile], dtype=float)
    phase_angular_profile = _phase_angular_profile(z, p, p_mid, ds, phase_intervals, params, float(lambda0))
    angular_values = np.asarray([item["angular_FV"] for item in phase_angular_profile], dtype=float)

    interval_rows = 2 * (int(params.n_nodes) - 1)
    outer = np.asarray(base_all[interval_rows : interval_rows + 2], dtype=float)
    sonic = np.asarray(base_all[interval_rows + 2 : interval_rows + 4], dtype=float)
    endpoint_max = float(np.nanmax(np.abs(endpoint_raw_array)))
    global_fv_max = float(np.nanmax(np.abs(mass_values))) if mass_values.size else math.nan
    outside_radial = float(np.nanmax(np.asarray(outside_radial_values))) if outside_radial_values else math.nan
    outside_energy = float(np.nanmax(np.asarray(outside_energy_values))) if outside_energy_values else math.nan
    p_r_values = np.concatenate([np.asarray(p[:, 3]), np.asarray(p_mid[:, 3])])
    exploratory = bool(
        float(phase_summary.get("radial_max", math.inf)) <= 1.0e-4
        and float(phase_summary.get("energy_max", math.inf)) <= 1.0e-4
        and float(phase_summary.get("fprime_max", math.inf)) <= 1.0e-5
        and float(phase_summary.get("kinematic_max", math.inf)) <= 1.0e-3
        and endpoint_max <= 1.0e-3
        and global_fv_max <= 3.0e-5
        and outside_radial <= 3.0e-5
        and outside_energy <= 3.0e-5
        and float(np.nanmax(np.abs(outer))) <= 5.0e-5
        and float(np.nanmax(np.abs(sonic))) <= 5.0e-5
        and float(np.nanmin(p_r_values)) > 0.0
    )
    summary = {
        "weighted_max": float(np.nanmax(np.abs(rows))),
        "raw_max": float(np.nanmax(np.abs(raw))),
        "phase_radial": float(phase_summary.get("radial_max", math.nan)),
        "phase_energy": float(phase_summary.get("energy_max", math.nan)),
        "phase_fprime": float(phase_summary.get("fprime_max", math.nan)),
        "phase_kinematic": float(phase_summary.get("kinematic_max", math.nan)),
        "phase_mesh": float(phase_summary.get("mesh_max", math.nan)),
        "phase_norm": float(phase_summary.get("norm_max", math.nan)),
        "endpoint_mismatch": endpoint_max,
        "interface_energy": float(interface_energy["max"]),
        "global_fv_mass": global_fv_max,
        "outside_radial": outside_radial,
        "outside_energy": outside_energy,
        "outer": float(np.nanmax(np.abs(outer))),
        "sonic": float(np.nanmax(np.abs(sonic))),
        "p_R_min": float(np.nanmin(p_r_values)),
        "p_R_max": float(np.nanmax(p_r_values)),
        "phase_angular_fv": float(np.nanmax(np.abs(angular_values))) if angular_values.size else math.nan,
        "accepted_exploratory": exploratory,
    }
    old_phase_mask = np.asarray([int(value) in phase_set for value in source_intervals], dtype=bool)
    old_phase_raw = source_raw_all[old_phase_mask] if old_phase_mask.size == source_raw_all.size else np.asarray([])
    summary["old_phase_source_audit"] = (
        float(np.nanmax(np.abs(old_phase_raw))) if old_phase_raw.size else math.nan
    )
    return {
        "rows": rows,
        "raw_rows": raw,
        "groups": groups,
        "summary": summary,
        "phase_data": phase_data,
        "interface_energy": interface_energy,
        "mass_profile": combined_mass_profile,
        "angular_profile": phase_angular_profile,
        "source_data": source_data,
        "x_log": x_log,
        "x_flux": x_flux,
        "aux": aux,
        "z": z,
        "p": p,
        "p_mid": p_mid,
        "ds": ds,
    }


def _composite_sparsity(setup: dict[str, Any]):
    from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack

    total_cols = int(setup["full0"].size)
    base_size = int(setup["base_size"])
    phase_offset = int(setup["phase_offset"])
    node_count = int(setup["phase_node_count"])
    interval_count = int(setup["phase_interval_count"])
    params = setup["params"]
    phase_nodes = np.asarray(setup["phase_nodes"], dtype=int)
    ordinary = setup["ordinary_sparsity"].tocsr()

    phase_local = model._global_flux_phase_dae_segment_sparsity(node_count, interval_count, "state")
    # The local state-mode helper appends eight endpoint anchor rows. Composite
    # production replaces them with exact phase/global interface rows below.
    phase_local = phase_local[:-8, :].tocsr()
    phase_left = lil_matrix((phase_local.shape[0], phase_offset), dtype=int)
    phase_left[:, 3 * int(params.n_nodes) + 1] = 1
    phase_block = hstack([phase_left.tocsr(), phase_local], format="csr")

    interface = lil_matrix((8, total_cols), dtype=int)
    n = int(params.n_nodes)
    phase_z_offset = phase_offset
    for row_base, phase_pos, global_node in (
        (0, 0, int(phase_nodes[0])),
        (4, node_count - 1, int(phase_nodes[-1])),
    ):
        for component, global_col in enumerate(
            (global_node, n + global_node, 2 * n + global_node, 3 * n)
        ):
            interface[row_base + component, global_col] = 1
            interface[row_base + component, phase_z_offset + 4 * phase_pos + component] = 1

    energy = lil_matrix((2, total_cols), dtype=int)
    first = int(setup["phase_intervals"][0])
    last = int(setup["phase_intervals"][-1])
    for row, nodes in (
        (0, range(max(0, first - 2), min(n, first + 2))),
        (1, range(max(0, last), min(n, last + 4))),
    ):
        for node in nodes:
            energy[row, node] = 1
            energy[row, n + node] = 1
            energy[row, 2 * n + node] = 1
        energy[row, 3 * n] = 1
        energy[row, 3 * n + 1] = 1
    p_offset = phase_offset + 4 * node_count
    p_mid_offset = p_offset + 4 * node_count
    ds_offset = p_mid_offset + 4 * interval_count
    for row, interval_pos in ((0, 0), (1, interval_count - 1)):
        for phase_pos in (interval_pos, interval_pos + 1):
            energy[row, phase_offset + 4 * phase_pos : phase_offset + 4 * phase_pos + 4] = 1
            energy[row, p_offset + 4 * phase_pos : p_offset + 4 * phase_pos + 4] = 1
        energy[row, p_mid_offset + 4 * interval_pos : p_mid_offset + 4 * interval_pos + 4] = 1
        energy[row, ds_offset + interval_pos] = 1

    if ordinary.shape[1] != total_cols:
        raise RuntimeError(f"ordinary sparsity has {ordinary.shape[1]} columns, expected {total_cols}")
    pattern = vstack([ordinary, phase_block, interface.tocsr(), energy.tocsr()], format="csr")
    expected_rows = int(_composite_data(setup["full0"], setup, audit=False)["rows"].size)
    if pattern.shape != (expected_rows, total_cols):
        raise RuntimeError(f"composite sparsity shape {pattern.shape}, expected {(expected_rows, total_cols)}")
    return pattern


def _release_mask(stage: str, setup: dict[str, Any]) -> np.ndarray:
    total = int(setup["full0"].size)
    base_size = int(setup["base_size"])
    aux_size = int(setup["aux_size"])
    phase_offset = int(setup["phase_offset"])
    params = setup["params"]
    n = int(params.n_nodes)
    release = np.zeros(total, dtype=bool)
    global_used = np.asarray(setup["global_used"], dtype=bool)
    aux_used = np.asarray(setup["aux_used"], dtype=bool)
    stage = str(stage).strip().lower()
    if stage in {"exterior", "coupled", "full"}:
        release[:base_size] = global_used
        release[base_size:phase_offset] = aux_used
    if stage in {"local", "interface"}:
        nodes: set[int] = set()
        first = int(setup["phase_nodes"][0])
        last = int(setup["phase_nodes"][-1])
        for center in (first, last):
            nodes.update(range(max(0, center - INTERFACE_HALO), min(n, center + INTERFACE_HALO + 1)))
        interior = set(int(value) for value in np.asarray(setup["phase_nodes"])[1:-1])
        nodes.difference_update(interior)
        for node in nodes:
            release[node] = True
            release[n + node] = True
            release[2 * n + node] = True
        release[3 * n : 3 * n + 2] = True
        release[:base_size] &= global_used
        # Retained source auxiliary variables enter the local guard rows; they
        # are inexpensive compared with releasing the far global state.
        release[base_size:phase_offset] = aux_used
    if stage in {"source", "source_band"}:
        first = max(0, int(setup["context"]["interval_indices"][0]) - INTERFACE_HALO)
        last = min(n - 1, int(setup["context"]["interval_indices"][-1]) + INTERFACE_HALO + 1)
        interior = set(int(value) for value in np.asarray(setup["phase_nodes"])[1:-1])
        for node in range(first, last + 1):
            if node in interior:
                continue
            release[node] = bool(global_used[node])
            release[n + node] = bool(global_used[n + node])
            release[2 * n + node] = bool(global_used[2 * n + node])
        release[base_size:phase_offset] = aux_used
    if stage in {"local", "interface", "source", "source_band", "coupled", "full"}:
        release[phase_offset:] = True
    return release


def _certification_score(summary: dict[str, Any]) -> float:
    scales = {
        "phase_radial": 1.0e-4,
        "phase_energy": 1.0e-4,
        "phase_fprime": 1.0e-5,
        "phase_kinematic": 1.0e-3,
        "endpoint_mismatch": 1.0e-3,
        "interface_energy": 1.0e-4,
        "global_fv_mass": 3.0e-5,
        "outside_radial": 3.0e-5,
        "outside_energy": 3.0e-5,
        "outer": 5.0e-5,
        "sonic": 5.0e-5,
    }
    values = []
    for key, scale in scales.items():
        value = abs(float(summary.get(key, math.inf)))
        values.append(value / scale if np.isfinite(value) else math.inf)
    p_r = float(summary.get("p_R_min", -math.inf))
    if not np.isfinite(p_r) or p_r <= 0.0:
        values.append(math.inf)
    return float(max(values))


def _run_stage(
    name: str,
    full_start: np.ndarray,
    setup: dict[str, Any],
    sparsity,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, Any], Any | None]:
    from scipy.optimize import least_squares

    release = _release_mask(name, setup)
    initial = _composite_data(full_start, setup)
    initial_summary = dict(initial["summary"])
    record: dict[str, Any] = {
        "stage": str(name),
        "released_variables": int(np.count_nonzero(release)),
        "active_rows": int(initial["rows"].size),
        "initial": initial_summary,
        "initial_l2": float(np.linalg.norm(initial["rows"])),
    }
    if not np.any(release) or max_nfev <= 0:
        record.update({"accepted": True, "nfev": 0, "status": 0, "message": "evaluate_only", "final": initial_summary})
        return np.asarray(full_start, dtype=float), record, None

    fixed = np.asarray(full_start, dtype=float).copy()
    x0 = fixed[release]
    lower = np.asarray(setup["lower"], dtype=float)[release]
    upper = np.asarray(setup["upper"], dtype=float)[release]
    x0 = np.clip(x0, lower + 1.0e-12, upper - 1.0e-12)

    def expand(reduced: np.ndarray) -> np.ndarray:
        full = fixed.copy()
        full[release] = reduced
        return full

    def residual(reduced: np.ndarray) -> np.ndarray:
        return np.asarray(_composite_data(expand(reduced), setup, audit=False)["rows"], dtype=float)

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        jac_sparsity=sparsity[:, release],
        x_scale="jac",
        loss="linear",
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
        max_nfev=max(1, int(max_nfev)),
        verbose=0,
    )
    candidate = expand(result.x)
    final = _composite_data(candidate, setup)
    final_summary = dict(final["summary"])
    initial_l2 = float(np.linalg.norm(initial["rows"]))
    final_l2 = float(np.linalg.norm(final["rows"]))
    phase_guard = max(
        float(final_summary.get("phase_radial", math.inf))
        / max(float(initial_summary.get("phase_radial", 0.0)), 1.0e-4),
        float(final_summary.get("phase_energy", math.inf))
        / max(float(initial_summary.get("phase_energy", 0.0)), 1.0e-4),
        float(final_summary.get("phase_kinematic", math.inf))
        / max(float(initial_summary.get("phase_kinematic", 0.0)), 1.0e-3),
    )
    accepted = bool(
        np.isfinite(final_l2)
        and final_l2 <= initial_l2 * (1.0 + 1.0e-8)
        and phase_guard <= 2.0
        and float(final_summary.get("p_R_min", -math.inf)) > 0.0
        and float(final_summary.get("endpoint_mismatch", math.inf))
        <= max(1.0e-3, 2.0 * float(initial_summary.get("endpoint_mismatch", 0.0)))
        and float(final_summary.get("sonic", math.inf))
        <= max(5.0e-5, 2.0 * float(initial_summary.get("sonic", 0.0)))
    )
    record.update(
        {
            "accepted": accepted,
            "nfev": int(result.nfev),
            "njev": int(result.njev) if result.njev is not None else 0,
            "status": int(result.status),
            "message": str(result.message),
            "final": final_summary,
            "final_l2": final_l2,
            "initial_score": _certification_score(initial_summary),
            "final_score": _certification_score(final_summary),
            "phase_guard": phase_guard,
        }
    )
    return (candidate if accepted else np.asarray(full_start, dtype=float)), record, result


def _column_families(setup: dict[str, Any]) -> np.ndarray:
    total = int(setup["full0"].size)
    names = np.full(total, "", dtype=object)
    n = int(setup["params"].n_nodes)
    base_size = int(setup["base_size"])
    phase_offset = int(setup["phase_offset"])
    node_count = int(setup["phase_node_count"])
    interval_count = int(setup["phase_interval_count"])
    names[:n] = "global_logu"
    names[n : 2 * n] = "global_logT"
    names[2 * n : 3 * n] = "global_F"
    names[3 * n : base_size] = "global_scalar"
    names[base_size:phase_offset] = "source_aux"
    cursor = phase_offset
    names[cursor : cursor + 4 * node_count] = "phase_z"
    cursor += 4 * node_count
    names[cursor : cursor + 4 * node_count] = "phase_p"
    cursor += 4 * node_count
    names[cursor : cursor + 4 * interval_count] = "phase_p_mid"
    cursor += 4 * interval_count
    names[cursor:] = "phase_ds"
    return names


def _rank_audit(jacobian, release: np.ndarray, setup: dict[str, Any]) -> dict[str, Any]:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import structural_rank
    from scipy.sparse.linalg import svds

    if jacobian is None:
        return {"available": False}
    J = csr_matrix(jacobian)
    out: dict[str, Any] = {
        "available": True,
        "rows": int(J.shape[0]),
        "variables": int(J.shape[1]),
        "structural_rank": int(structural_rank(J)),
    }
    try:
        if J.shape[1] <= 400:
            _u, singular_all, vt_all = np.linalg.svd(J.toarray(), full_matrices=False)
            tolerance = max(J.shape) * np.finfo(float).eps * max(float(singular_all[0]), 1.0)
            out["numerical_rank"] = int(np.count_nonzero(singular_all > tolerance))
            out["smallest_singular_values"] = np.asarray(singular_all[-4:], dtype=float)[::-1].tolist()
            out["condition"] = float(singular_all[0] / max(float(singular_all[-1]), 1.0e-300))
            weakest = np.asarray(vt_all[-1], dtype=float)
            families = _column_families(setup)[release]
            localization: dict[str, float] = {}
            for family in sorted(set(str(value) for value in families)):
                values = weakest[families == family]
                localization[family] = float(np.sqrt(np.mean(values**2))) if values.size else 0.0
            out["weakest_right_rms"] = localization
            return out
        k = min(4, min(J.shape) - 1)
        if k <= 0:
            return out
        _u, singular, vt = svds(J, k=k, which="SM", return_singular_vectors=True)
        order = np.argsort(singular)
        singular = np.asarray(singular, dtype=float)[order]
        vt = np.asarray(vt, dtype=float)[order]
        out["smallest_singular_values"] = singular.tolist()
        out["condition_lower_bound"] = float(1.0 / max(float(singular[0]), 1.0e-300))
        weakest = np.asarray(vt[0], dtype=float)
        families = _column_families(setup)[release]
        localization: dict[str, float] = {}
        for family in sorted(set(str(value) for value in families)):
            values = weakest[families == family]
            localization[family] = float(np.sqrt(np.mean(values**2))) if values.size else 0.0
        out["weakest_right_rms"] = localization
    except Exception as exc:
        out["numerical_error"] = str(exc)
    return out


def _diagnostic_x(data: dict[str, Any], setup: dict[str, Any]) -> np.ndarray:
    x_flux = np.asarray(data["x_flux"], dtype=float).copy()
    z = np.asarray(data["z"], dtype=float)
    n = int(setup["params"].n_nodes)
    for pos, node in enumerate(np.asarray(setup["phase_nodes"], dtype=int)):
        x_flux[int(node)] = float(z[pos, 0])
        x_flux[n + int(node)] = float(z[pos, 1])
        x_flux[2 * n + int(node)] = float(z[pos, 2])
    return model._flux_x_to_log_x(x_flux, setup["params"])


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _physical_diagnostics(x_log: np.ndarray, params) -> dict[str, Any]:
    try:
        logu, logT, log_mdot, log_r_son, lambda0, log_r = model.pilot._unpack(x_log, params)
        local = model.pilot._local_params(params, log_r, log_mdot)
        z_state = model.pilot._state_vector(logu, logT, float(log_r_son), float(lambda0))
        profile = model.transonic_profile_from_state_vector(z_state, local)
        advection = model.scan.advection_diagnostic(z_state, local)
        mdot = np.exp(np.asarray(log_mdot, dtype=float))
        return _jsonable(
            {
                **advection,
                "Rson_rg": float(profile.sonic_radius / params.r_g),
                "max_H_R": float(np.max(profile.H_over_R)),
                "integrated_adv": float(profile.integrated_advective_fraction),
                "Mdot_outer_over_inner": float(mdot[-1] / max(float(mdot[0]), 1.0e-300)),
                "wind_sink_fraction_net": float(1.0 - mdot[-1] / max(float(mdot[0]), 1.0e-300)),
            }
        )
    except Exception as exc:
        return {"error": str(exc)}


def _write_checkpoint(data: dict[str, Any], setup: dict[str, Any], full: np.ndarray) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / "stage_00_etaE_98p125_N164.npz"
    diagnostic_x = _diagnostic_x(data, setup)
    np.savez_compressed(
        path,
        x=np.asarray(diagnostic_x, dtype=float),
        global_phase_composite_full=np.asarray(full, dtype=float),
        global_phase_x_flux=np.asarray(data["x_flux"], dtype=float),
        global_phase_source_aux=np.asarray(data["aux"], dtype=float),
        global_flux_phase_dae_segment_aux_interval_indices=np.asarray(setup["phase_intervals"], dtype=int),
        global_flux_phase_dae_segment_aux_node_indices=np.asarray(setup["phase_nodes"], dtype=int),
        global_flux_phase_dae_segment_aux_z=np.asarray(data["z"], dtype=float),
        global_flux_phase_dae_segment_aux_p=np.asarray(data["p"], dtype=float),
        global_flux_phase_dae_segment_aux_p_mid=np.asarray(data["p_mid"], dtype=float),
        global_flux_phase_dae_segment_aux_ds=np.asarray(data["ds"], dtype=float),
        wind_energy_multiplier=np.asarray(ETA_E),
        n_nodes=np.asarray(int(setup["params"].n_nodes)),
        R_out_rg=np.asarray(float(setup["params"].R_out_rg)),
        accepted=np.asarray(bool(data["summary"].get("accepted_exploratory", False))),
    )
    return path


def _write_note(
    stages: list[dict[str, Any]],
    final_data: dict[str, Any],
    rank: dict[str, Any],
    checkpoint: Path,
    physical: dict[str, Any],
) -> None:
    summary = final_data["summary"]
    lines = [
        "# Mdot=5 Global Phase-Space DAE Production Results",
        "",
        "## Target",
        "",
        "- `Mdot_inner/Edd = 5`",
        "- `Rout = 335 rg`",
        "- `Rinj = 240 rg`",
        "- `f_s = 0.80`",
        "- compact source, local-Mdot wind",
        "- `eta_E = 98.125`, `N = 164`",
        "- phase replacement intervals `129--141`",
        "",
        "## Formulation",
        "",
        "The K13 phase trajectory is now part of a composite production residual.",
        "Old global radial, energy, mass, and source-element rows are removed on",
        "the phase intervals. Global interior states displaced by the phase mesh",
        "are excluded from the active variable vector. Interface state continuity",
        "and adjacent finite-volume energy balances are active; derivative",
        "continuity is not imposed. Angular momentum flux remains an audit.",
        "",
        "## Staged Coupling",
        "",
        "| stage | accepted | nfev | variables | initial score | final score | final weighted max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in stages:
        final = stage.get("final", stage.get("initial", {}))
        lines.append(
            "| {stage} | {accepted} | {nfev} | {variables} | {initial:.3e} | {final_score:.3e} | {weighted:.3e} |".format(
                stage=stage.get("stage", ""),
                accepted=stage.get("accepted", False),
                nfev=int(stage.get("nfev", 0)),
                variables=int(stage.get("released_variables", 0)),
                initial=float(stage.get("initial_score", _certification_score(stage.get("initial", {})))),
                final_score=float(stage.get("final_score", _certification_score(final))),
                weighted=float(final.get("weighted_max", math.nan)),
            )
        )
    lines.extend(
        [
            "",
            "## Final Unified Residuals",
            "",
            "| diagnostic | value | exploratory limit |",
            "| --- | ---: | ---: |",
            f"| phase radial | `{float(summary['phase_radial']):.6e}` | `1e-4` |",
            f"| physical phase energy | `{float(summary['phase_energy']):.6e}` | `1e-4` |",
            f"| phase F-prime | `{float(summary['phase_fprime']):.6e}` | `1e-5` |",
            f"| phase kinematic | `{float(summary['phase_kinematic']):.6e}` | `1e-3` |",
            f"| interface state mismatch | `{float(summary['endpoint_mismatch']):.6e}` | `1e-3` |",
            f"| interface FV energy | `{float(summary['interface_energy']):.6e}` | `1e-4` |",
            f"| global FV mass | `{float(summary['global_fv_mass']):.6e}` | `3e-5` |",
            f"| outside radial | `{float(summary['outside_radial']):.6e}` | `3e-5` |",
            f"| outside energy | `{float(summary['outside_energy']):.6e}` | `3e-5` |",
            f"| sonic | `{float(summary['sonic']):.6e}` | `5e-5` |",
            f"| outer | `{float(summary['outer']):.6e}` | `5e-5` |",
            f"| p_R min | `{float(summary['p_R_min']):.6e}` | `>0` |",
            f"| angular FV audit | `{float(summary['phase_angular_fv']):.6e}` | audit only |",
            f"| removed old phase-row audit | `{float(summary['old_phase_source_audit']):.6e}` | audit only |",
            "",
            f"Unified exploratory acceptance: `{bool(summary['accepted_exploratory'])}`.",
            "",
            "## Gauge And Rank Audit",
            "",
            f"- active rows: `{rank.get('rows', 'n/a')}`",
            f"- active variables: `{rank.get('variables', 'n/a')}`",
            f"- structural rank: `{rank.get('structural_rank', 'n/a')}`",
            f"- smallest singular values: `{rank.get('smallest_singular_values', [])}`",
            f"- weakest right-vector RMS by family: `{rank.get('weakest_right_rms', {})}`",
            "",
            "## Physical Diagnostics",
            "",
            "```json",
            json.dumps(_jsonable(physical), indent=2, sort_keys=True),
            "```",
            "",
            "## Files",
            "",
            f"- checkpoint: `{checkpoint.relative_to(ROOT)}`",
            f"- table: `{TABLE_PATH.relative_to(ROOT)}`",
            f"- profiles: `{PROFILE_PATH.relative_to(ROOT)}`",
            "",
            "Eta continuation remains paused unless every unified exploratory gate passes.",
        ]
    )
    NOTE_PATH.write_text("\n".join(lines) + "\n")


def _persist(
    stages: list[dict[str, Any]],
    final_data: dict[str, Any],
    setup: dict[str, Any],
    full: np.ndarray,
    rank: dict[str, Any],
) -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = _write_checkpoint(final_data, setup, full)
    diagnostic_x = _diagnostic_x(final_data, setup)
    physical = _physical_diagnostics(diagnostic_x, setup["params"])
    table = {
        "target": {
            "mdot_inner_edd": 5.0,
            "Rout_rg": 335.0,
            "Rinj_rg": 240.0,
            "stream_fraction": 0.80,
            "eta_E": ETA_E,
            "n_nodes": int(setup["params"].n_nodes),
        },
        "phase_intervals": np.asarray(setup["phase_intervals"], dtype=int),
        "phase_nodes": np.asarray(setup["phase_nodes"], dtype=int),
        "stages": stages,
        "final": final_data["summary"],
        "rank_audit": rank,
        "physical": physical,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
    }
    TABLE_PATH.write_text(json.dumps(_jsonable(table), indent=2, sort_keys=True) + "\n")
    profile = {
        "phase_profile": final_data["phase_data"].get("profile", []),
        "phase_kinematic_profile": final_data["phase_data"].get("kinematic_profile", []),
        "global_fv_mass_profile": final_data["mass_profile"],
        "phase_angular_profile": final_data["angular_profile"],
        "interface_energy": final_data["interface_energy"],
    }
    PROFILE_PATH.write_text(json.dumps(_jsonable(profile), indent=2, sort_keys=True) + "\n")
    _write_note(stages, final_data, rank, checkpoint, physical)


def main() -> None:
    x_log, params, context, aux, phase = _load_problem()
    setup = _composite_setup(x_log, params, context, aux, phase)
    sparsity = _composite_sparsity(setup)
    full = np.asarray(setup["full0"], dtype=float).copy()
    stages: list[dict[str, Any]] = []
    if RESUME_CHECKPOINT_RAW:
        resume = Path(RESUME_CHECKPOINT_RAW).expanduser()
        if not resume.is_absolute():
            resume = ROOT / resume
        with np.load(resume) as data:
            saved = np.asarray(data["global_phase_composite_full"], dtype=float)
        if saved.shape != full.shape:
            raise ValueError(f"resume composite shape {saved.shape} does not match {full.shape}")
        full = saved
        if TABLE_PATH.exists():
            try:
                prior = json.loads(TABLE_PATH.read_text())
                stages = list(prior.get("stages", []))
            except Exception:
                stages = []
    last_jacobian = None
    last_release = np.zeros(full.size, dtype=bool)

    initial_data = _composite_data(full, setup)
    evaluate_record = {
        "stage": "evaluate",
        "accepted": True,
        "nfev": 0,
        "released_variables": 0,
        "active_rows": int(initial_data["rows"].size),
        "initial": dict(initial_data["summary"]),
        "final": dict(initial_data["summary"]),
        "initial_score": _certification_score(initial_data["summary"]),
        "final_score": _certification_score(initial_data["summary"]),
        "initial_l2": float(np.linalg.norm(initial_data["rows"])),
        "final_l2": float(np.linalg.norm(initial_data["rows"])),
    }
    if "evaluate" in STAGES:
        stages.append(evaluate_record)
    print(
        "evaluate",
        f"score={evaluate_record['final_score']:.3e}",
        json.dumps(_jsonable(initial_data["summary"]), sort_keys=True),
        flush=True,
    )

    stage_limits = {
        "exterior": MAX_NFEV_EXTERIOR,
        "local": MAX_NFEV_LOCAL,
        "interface": MAX_NFEV_LOCAL,
        "source": MAX_NFEV_LOCAL,
        "source_band": MAX_NFEV_LOCAL,
        "coupled": MAX_NFEV_COUPLED,
        "full": MAX_NFEV_COUPLED,
    }
    for stage in STAGES:
        if stage == "evaluate":
            continue
        full_next, record, result = _run_stage(stage, full, setup, sparsity, stage_limits.get(stage, 0))
        stages.append(record)
        if record.get("accepted", False):
            full = full_next
            if result is not None:
                last_jacobian = result.jac
                last_release = _release_mask(stage, setup)
        print(
            stage,
            f"accepted={record.get('accepted', False)}",
            f"nfev={record.get('nfev', 0)}",
            f"score={float(record.get('final_score', math.nan)):.3e}",
            json.dumps(_jsonable(record.get("final", {})), sort_keys=True),
            flush=True,
        )
        final_data = _composite_data(full, setup)
        _persist(stages, final_data, setup, full, {"available": False, "pending": True})
        if final_data["summary"].get("accepted_exploratory", False):
            break

    final_data = _composite_data(full, setup)
    rank = (
        _rank_audit(last_jacobian, last_release, setup)
        if RANK_AUDIT and last_jacobian is not None
        else {"available": False}
    )
    _persist(stages, final_data, setup, full, rank)
    print("final", json.dumps(_jsonable(final_data["summary"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
