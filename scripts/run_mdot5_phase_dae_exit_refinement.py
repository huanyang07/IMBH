"""Refine the K13 phase tail and audit a monotone interval-14 extension."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT_RADIAL_WEIGHT", "100")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT_ENERGY_WEIGHT", "100")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT_FPRIME_WEIGHT", "100")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT_KINEMATIC_WEIGHT", "30")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT_NORM_WEIGHT", "10")
os.environ.setdefault("IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT_MESH_WEIGHT", "10")

import run_mdot5_global_phase_dae_production as global_phase  # noqa: E402


model = global_phase.model
INPUT_CHECKPOINT = Path(
    os.environ.get(
        "IMBH_MDOT5_PHASE_EXIT_INPUT",
        str(
            ROOT
            / "outputs/checkpoints/m5_eta_global_phase_dae_k13_98p125_N164"
            / "stage_00_etaE_98p125_N164.npz"
        ),
    )
).expanduser()
if not INPUT_CHECKPOINT.is_absolute():
    INPUT_CHECKPOINT = ROOT / INPUT_CHECKPOINT
MAX_NFEV = int(os.environ.get("IMBH_MDOT5_PHASE_EXIT_MAX_NFEV", "40"))
ENDPOINT_WEIGHT = float(os.environ.get("IMBH_MDOT5_PHASE_EXIT_ENDPOINT_WEIGHT", "100"))
P_R_FLOOR = float(os.environ.get("IMBH_MDOT5_PHASE_EXIT_P_R_FLOOR", "1e-4"))
CASES = tuple(
    piece.strip().lower()
    for piece in os.environ.get(
        "IMBH_MDOT5_PHASE_EXIT_CASES",
        "refine2,extend2,extend3",
    ).split(",")
    if piece.strip()
)
OUTPUT_STEM = os.environ.get("IMBH_MDOT5_PHASE_EXIT_OUTPUT_STEM", "m5_eta_phase_dae_exit_refinement_98p125_N164")
TABLE_PATH = ROOT / "outputs" / "tables" / f"{OUTPUT_STEM}.json"
PROFILE_PATH = ROOT / "outputs" / "tables" / f"{OUTPUT_STEM}_profiles.json"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / OUTPUT_STEM
NOTE_PATH = ROOT / "Note" / "CODEX_MDOT5_PHASE_DAE_EXIT_REFINEMENT_RESULTS.md"


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0.0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    out = value / norm
    if out[3] < 0.0:
        out = -out
    return out


def _refine_tail(
    z: np.ndarray,
    p: np.ndarray,
    p_mid: np.ndarray,
    ds: np.ndarray,
    tail_count: int,
) -> tuple[np.ndarray, ...]:
    split_start = max(0, int(ds.size) - max(int(tail_count), 0))
    z_out = [np.asarray(z[0], dtype=float)]
    p_out = [np.asarray(p[0], dtype=float)]
    p_mid_out: list[np.ndarray] = []
    ds_out: list[float] = []
    for pos in range(int(ds.size)):
        if pos < split_start:
            p_mid_out.append(np.asarray(p_mid[pos], dtype=float))
            ds_out.append(float(ds[pos]))
            z_out.append(np.asarray(z[pos + 1], dtype=float))
            p_out.append(np.asarray(p[pos + 1], dtype=float))
            continue
        ds_half = 0.5 * float(ds[pos])
        z_middle = 0.5 * (np.asarray(z[pos]) + np.asarray(z[pos + 1]))
        z_middle += float(ds[pos]) / 8.0 * (np.asarray(p[pos]) - np.asarray(p[pos + 1]))
        p_middle = _normalize(np.asarray(p_mid[pos], dtype=float))
        p_mid_left = _normalize(0.5 * (np.asarray(p[pos]) + p_middle))
        p_mid_right = _normalize(0.5 * (p_middle + np.asarray(p[pos + 1])))
        p_mid_out.extend([p_mid_left, p_mid_right])
        ds_out.extend([ds_half, ds_half])
        z_out.extend([z_middle, np.asarray(z[pos + 1], dtype=float)])
        p_out.extend([p_middle, np.asarray(p[pos + 1], dtype=float)])
    return (
        np.asarray(z_out, dtype=float),
        np.asarray(p_out, dtype=float),
        np.asarray(p_mid_out, dtype=float),
        np.asarray(ds_out, dtype=float),
    )


def _append_extension(
    z: np.ndarray,
    p: np.ndarray,
    p_mid: np.ndarray,
    ds: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, ...]:
    target = np.asarray(target, dtype=float)
    dx = float(target[3] - z[-1, 3])
    if dx <= 0.0:
        raise ValueError("extension target must be outside the phase endpoint")
    finite_tangent = np.asarray(
        [
            (target[0] - z[-1, 0]) / dx,
            (target[1] - z[-1, 1]) / dx,
            (target[2] - z[-1, 2]) / dx,
            1.0,
        ],
        dtype=float,
    )
    p_right = _normalize(0.7 * np.asarray(p[-1]) + 0.3 * _normalize(finite_tangent))
    p_middle = _normalize(0.5 * (np.asarray(p[-1]) + p_right))
    p_r_average = max((float(p[-1, 3]) + 4.0 * float(p_middle[3]) + float(p_right[3])) / 6.0, 1.0e-5)
    ds_new = max(dx / p_r_average, 1.0e-8)
    return (
        np.vstack([z, target]),
        np.vstack([p, p_right]),
        np.vstack([p_mid, p_middle]),
        np.concatenate([ds, np.asarray([ds_new])]),
    )


def _bounds(z: np.ndarray, p: np.ndarray, p_mid: np.ndarray, ds: np.ndarray, monotone: bool) -> tuple[np.ndarray, np.ndarray]:
    z_lo = np.asarray(z, dtype=float).copy()
    z_hi = np.asarray(z, dtype=float).copy()
    z_lo[:, :2] -= 0.03
    z_hi[:, :2] += 0.03
    z_lo[:, 2] = np.maximum(z[:, 2] - 0.03, 1.0e-12)
    z_hi[:, 2] = z[:, 2] + 0.03
    z_lo[:, 3] -= 0.03
    z_hi[:, 3] += 0.03
    for pos in range(z.shape[0]):
        if pos > 0:
            z_lo[pos, 3] = max(z_lo[pos, 3], 0.5 * (z[pos - 1, 3] + z[pos, 3]) + 1.0e-10)
        if pos < z.shape[0] - 1:
            z_hi[pos, 3] = min(z_hi[pos, 3], 0.5 * (z[pos, 3] + z[pos + 1, 3]) - 1.0e-10)
    p_lo = np.full(p.size, -10.0).reshape(p.shape)
    p_hi = np.full(p.size, 10.0).reshape(p.shape)
    pm_lo = np.full(p_mid.size, -10.0).reshape(p_mid.shape)
    pm_hi = np.full(p_mid.size, 10.0).reshape(p_mid.shape)
    if monotone:
        p_lo[:, 3] = P_R_FLOOR
        pm_lo[:, 3] = P_R_FLOOR
    lower = np.concatenate([z_lo.ravel(), p_lo.ravel(), pm_lo.ravel(), np.log(ds) - 3.0])
    upper = np.concatenate([z_hi.ravel(), p_hi.ravel(), pm_hi.ravel(), np.log(ds) + 3.0])
    return lower, upper


def _solve_case(
    label: str,
    z_seed: np.ndarray,
    p_seed: np.ndarray,
    p_mid_seed: np.ndarray,
    ds_seed: np.ndarray,
    params,
    lambda0: float,
    left_reference: np.ndarray,
    right_reference: np.ndarray,
    monotone: bool,
    free_right_state: bool,
) -> tuple[dict[str, Any], tuple[np.ndarray, ...]]:
    from scipy.optimize import least_squares

    interval_count = int(ds_seed.size)
    node_count = interval_count + 1
    labels = np.arange(interval_count, dtype=int)
    mesh_target = np.diff(np.log(np.maximum(ds_seed, 1.0e-300)))
    start = global_phase._phase_pack(z_seed, p_seed, p_mid_seed, ds_seed)
    lower, upper = _bounds(z_seed, p_seed, p_mid_seed, ds_seed, monotone)

    def unpack(vector: np.ndarray) -> tuple[np.ndarray, ...]:
        return global_phase._phase_unpack(vector, node_count, interval_count)

    def residual(vector: np.ndarray) -> np.ndarray:
        z, p, p_mid, ds = unpack(vector)
        data = model._global_flux_phase_dae_segment_data(
            z, p, p_mid, ds, params, float(lambda0), labels, mesh_target
        )
        left_rows = ENDPOINT_WEIGHT * (z[0] - left_reference)
        right_weights = np.asarray(
            [1.0, 1.0, 1.0, ENDPOINT_WEIGHT] if free_right_state else [ENDPOINT_WEIGHT] * 4,
            dtype=float,
        )
        endpoints = np.concatenate([left_rows, right_weights * (z[-1] - right_reference)])
        return np.concatenate([np.asarray(data["rows"], dtype=float), endpoints])

    sparsity = model._global_flux_phase_dae_segment_sparsity(node_count, interval_count, "state")
    result = least_squares(
        residual,
        np.clip(start, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        jac_sparsity=sparsity,
        x_scale="jac",
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
        max_nfev=max(1, MAX_NFEV),
        verbose=0,
    )
    jac_smin = math.nan
    jac_condition = math.nan
    try:
        jacobian = result.jac.toarray() if hasattr(result.jac, "toarray") else np.asarray(result.jac, dtype=float)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        if singular.size:
            jac_smin = float(singular[-1])
            jac_condition = float(singular[0] / max(jac_smin, 1.0e-300))
    except Exception:
        pass
    z, p, p_mid, ds = unpack(result.x)
    data = model._global_flux_phase_dae_segment_data(
        z, p, p_mid, ds, params, float(lambda0), labels, mesh_target
    )
    summary = dict(data["summary"])
    endpoint = float(np.max(np.abs(np.concatenate([z[0] - left_reference, z[-1] - right_reference]))))
    right_state_shift = float(np.max(np.abs(z[-1, :3] - right_reference[:3])))
    right_logr_mismatch = abs(float(z[-1, 3] - right_reference[3]))
    endpoint_gate = max(float(np.max(np.abs(z[0] - left_reference))), right_logr_mismatch)
    accepted = bool(
        float(summary.get("radial_max", math.inf)) <= 1.0e-4
        and float(summary.get("energy_max", math.inf)) <= 1.0e-4
        and float(summary.get("fprime_max", math.inf)) <= 1.0e-5
        and float(summary.get("kinematic_max", math.inf)) <= 1.0e-3
        and endpoint_gate <= 1.0e-3
        and float(summary.get("p_R_min", -math.inf)) > 0.0
    )
    row = {
        "label": label,
        "intervals": interval_count,
        "monotone_bound": bool(monotone),
        "nfev": int(result.nfev),
        "status": int(result.status),
        "message": str(result.message),
        "endpoint_mismatch": endpoint,
        "endpoint_gate": endpoint_gate,
        "right_state_shift": right_state_shift,
        "right_logR_mismatch": right_logr_mismatch,
        "jacobian_smin": jac_smin,
        "jacobian_condition": jac_condition,
        "accepted": accepted,
        **summary,
        "profile": data.get("profile", []),
        "kinematic_profile": data.get("kinematic_profile", []),
    }
    return row, (z, p, p_mid, ds)


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


def _case_fraction(case: str) -> float:
    if "threequarter" in case:
        return 0.75
    if "quarter" in case:
        return 0.25
    if "half" in case:
        return 0.5
    match = re.search(r"_f(\d+)$", case)
    if match:
        digits = match.group(1)
        return float(int(digits)) / float(10 ** len(digits))
    return 1.0


def _prior_extension_checkpoint(tail: int, fraction: float) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for label, value in (("quarter", 0.25), ("half", 0.5), ("threequarter", 0.75)):
        path = CHECKPOINT_DIR / f"extend{tail}_{label}.npz"
        if value < fraction and path.exists():
            candidates.append((value, path))
    for path in CHECKPOINT_DIR.glob(f"extend{tail}_f*.npz"):
        value = _case_fraction(path.stem)
        if value < fraction:
            candidates.append((value, path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def main() -> None:
    x_log, params, context, aux, phase = global_phase._load_problem()
    setup = global_phase._composite_setup(x_log, params, context, aux, phase)
    with np.load(INPUT_CHECKPOINT) as data:
        full = np.asarray(data["global_phase_composite_full"], dtype=float)
    composite = global_phase._composite_data(full, setup)
    z0 = np.asarray(composite["z"], dtype=float)
    p0 = np.asarray(composite["p"], dtype=float)
    p_mid0 = np.asarray(composite["p_mid"], dtype=float)
    ds0 = np.asarray(composite["ds"], dtype=float)
    x_flux = np.asarray(composite["x_flux"], dtype=float)
    n = int(params.n_nodes)
    _u, _t, _m, _rs, lambda0, log_r = model.pilot._unpack(composite["x_log"], params)
    target_node = int(setup["phase_nodes"][-1]) + 1
    target = np.asarray(
        [
            x_flux[target_node],
            x_flux[n + target_node],
            x_flux[2 * n + target_node],
            log_r[target_node],
        ],
        dtype=float,
    )
    results: list[dict[str, Any]] = []
    saved: dict[str, tuple[np.ndarray, ...]] = {}
    for case in CASES:
        tail = 3 if case.startswith(("refine3", "extend3")) else 2
        extend = case.startswith("extend")
        refined = saved.get(f"refine{tail}")
        if refined is None:
            prior_label = f"refine{tail}"
            prior_path = _prior_extension_checkpoint(tail, _case_fraction(case)) if extend else None
            if prior_path is None:
                prior_path = CHECKPOINT_DIR / f"{prior_label}.npz"
            if extend and prior_path.exists():
                with np.load(prior_path) as prior:
                    refined = (
                        np.asarray(prior["z"], dtype=float),
                        np.asarray(prior["p"], dtype=float),
                        np.asarray(prior["p_mid"], dtype=float),
                        np.asarray(prior["ds"], dtype=float),
                    )
            else:
                refined = _refine_tail(z0, p0, p_mid0, ds0, tail)
        case_target = np.asarray(target, dtype=float).copy()
        if extend:
            fraction = _case_fraction(case)
            z_last = np.asarray(refined[0][-1], dtype=float)
            p_last = np.asarray(refined[1][-1], dtype=float)
            target_log_r = float(z0[-1, 3] + fraction * (target[3] - z0[-1, 3]))
            dx = float(target_log_r - z_last[3])
            case_target[3] = target_log_r
            case_target[:3] = z_last[:3] + dx * p_last[:3] / max(float(p_last[3]), 1.0e-6)
            case_target[2] = max(float(case_target[2]), 1.0e-12)
        seed = _append_extension(*refined, case_target) if extend else refined
        row, solution = _solve_case(
            case,
            *seed,
            params,
            float(lambda0),
            np.asarray(z0[0], dtype=float),
            np.asarray(case_target if extend else z0[-1], dtype=float),
            not case.startswith("free"),
            extend,
        )
        results.append(row)
        saved[case] = solution
        print(case, json.dumps(_jsonable({key: value for key, value in row.items() if key not in {"profile", "kinematic_profile"}}), sort_keys=True), flush=True)

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(json.dumps(_jsonable(results), indent=2, sort_keys=True) + "\n")
    PROFILE_PATH.write_text(
        json.dumps(
            _jsonable(
                {
                    row["label"]: {
                        "profile": row.pop("profile"),
                        "kinematic_profile": row.pop("kinematic_profile"),
                    }
                    for row in results
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for label, (z, p, p_mid, ds) in saved.items():
        np.savez_compressed(
            CHECKPOINT_DIR / f"{label}.npz",
            z=z,
            p=p,
            p_mid=p_mid,
            ds=ds,
            accepted=np.asarray(next(row["accepted"] for row in results if row["label"] == label)),
        )
    lines = [
        "# Mdot=5 Phase-Space DAE Exit Refinement Results",
        "",
        "The unified K13 solve localizes its remaining radial/FV defect in the",
        "ordinary source elements immediately outside the right phase interface.",
        "This audit h-refines the K13 tail and tests a positive-p_R extension to",
        "the next global node.",
        "",
        "| case | intervals | nfev | radial | energy | F-prime | kinematic | p_R min | endpoint | accepted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['label']} | {row['intervals']} | {row['nfev']} | {row['radial_max']:.3e} | "
            f"{row['energy_max']:.3e} | {row['fprime_max']:.3e} | {row['kinematic_max']:.3e} | "
            f"{row['p_R_min']:.3e} | {row['endpoint_mismatch']:.3e} | {row['accepted']} |"
        )
    lines.extend(
        [
            "",
            "A monotone interval-14 extension is accepted only if all direct physical",
            "phase residual gates pass. Eta continuation remains paused.",
        ]
    )
    NOTE_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
