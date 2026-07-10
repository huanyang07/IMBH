"""Refine a unified conservative wind checkpoint without changing its grid shape."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    conservative_residual_audit,
    conservative_residual_profile,
    remap_conservative_state,
    residual_adapted_conservative_grid,
    solve_conservative_disk,
)
from run_unified_conservative_wind_continuation import CHECKPOINT_DIR, _starting_problem


ROOT = Path(__file__).resolve().parents[1]
EPSILON_W = float(os.environ.get("IMBH_CONSERVATIVE_WIND_REFINE_EPSILON", "0.32"))
TARGET_N = int(os.environ.get("IMBH_CONSERVATIVE_WIND_REFINE_N", "192"))
MAX_NFEV = int(os.environ.get("IMBH_CONSERVATIVE_WIND_REFINE_MAX_NFEV", "500"))
PASSES = int(os.environ.get("IMBH_CONSERVATIVE_WIND_REFINE_PASSES", "3"))
FREEZE_PREFIX_NODES = int(os.environ.get("IMBH_CONSERVATIVE_WIND_REFINE_FREEZE_PREFIX_NODES", "0"))
ADAPT_GAIN = float(os.environ.get("IMBH_CONSERVATIVE_WIND_REFINE_ADAPT_GAIN", "0"))
ADAPT_BLEND = float(os.environ.get("IMBH_CONSERVATIVE_WIND_REFINE_ADAPT_BLEND", "0.35"))
ADAPT_MAX_RADIUS_RAW = os.environ.get("IMBH_CONSERVATIVE_WIND_REFINE_ADAPT_MAX_RADIUS_RG", "").strip()


def _resample_grid(
    custom_grid_xi: tuple[float, ...] | None,
    target_n: int,
    *,
    freeze_prefix_nodes: int = 0,
) -> tuple[float, ...] | None:
    if custom_grid_xi is None:
        return None
    old = np.asarray(custom_grid_xi, dtype=float)
    frozen = int(freeze_prefix_nodes)
    if frozen < 0 or frozen >= old.size:
        raise ValueError("freeze_prefix_nodes must lie in [0, source_n)")
    if frozen and target_n < old.size:
        raise ValueError("prefix-frozen remapping only supports refinement")
    if frozen:
        tail_old = old[frozen - 1 :]
        tail_coordinate = np.linspace(0.0, 1.0, tail_old.size)
        tail_target = np.linspace(0.0, 1.0, target_n - frozen + 1)
        tail = np.interp(tail_target, tail_coordinate, tail_old)[1:]
        return tuple(float(value) for value in np.concatenate([old[:frozen], tail]))
    old_index = np.linspace(0.0, 1.0, old.size)
    new_index = np.linspace(0.0, 1.0, target_n)
    return tuple(float(value) for value in np.interp(new_index, old_index, old))


def run() -> dict[str, object]:
    _unused, params = _starting_problem()
    source_n = int(params.disk.n_nodes)
    params = replace(
        params,
        disk=replace(params.disk, wind_energy_limited_epsilon=EPSILON_W),
    )
    safe = str(EPSILON_W).replace(".", "p")
    source = CHECKPOINT_DIR / f"mdot2_fs080_eps{safe}_eta{params.closure.wind_launch_energy_multiplier:g}_N{params.disk.n_nodes}.npz"
    with np.load(source) as data:
        state = np.asarray(data["x"], dtype=float)
        if "custom_grid_xi" in data.files:
            params = replace(
                params,
                disk=replace(
                    params.disk,
                    custom_grid_xi=tuple(
                        float(value) for value in np.asarray(data["custom_grid_xi"], dtype=float)
                    ),
                ),
            )

    target_grid = (
        residual_adapted_conservative_grid(
            state,
            params,
            target_n=TARGET_N,
            gain=ADAPT_GAIN,
            blend=ADAPT_BLEND,
            max_radius_rg=float(ADAPT_MAX_RADIUS_RAW) if ADAPT_MAX_RADIUS_RAW else None,
        )
        if ADAPT_GAIN > 0.0
        else _resample_grid(
            params.disk.custom_grid_xi,
            TARGET_N,
            freeze_prefix_nodes=FREEZE_PREFIX_NODES,
        )
    )
    target_disk = replace(
        params.disk,
        n_nodes=TARGET_N,
        custom_grid_xi=target_grid,
    )
    state, params = remap_conservative_state(state, params, target_disk)
    params = replace(params, max_nfev=MAX_NFEV)
    initial = conservative_residual_audit(state, params)
    passes: list[dict[str, object]] = []
    for pass_index in range(1, PASSES + 1):
        solved = solve_conservative_disk(state, params)
        state = solved.x
        passes.append({"pass": pass_index, "nfev": solved.nfev, "audit": asdict(solved.final_audit)})
        if solved.final_audit.maximum <= 1.0e-5:
            break
    final = conservative_residual_audit(state, params)
    profile = conservative_residual_profile(state, params)
    peaks = {}
    for name in ("radial", "mass", "angular_momentum", "energy", "energy_compatibility"):
        index = int(np.argmax(np.abs(profile[name])))
        peaks[name] = {
            "maximum": float(abs(profile[name][index])),
            "R_rg": float(profile["R_mid_rg"][index]),
            "interval": index,
        }
    row: dict[str, object] = {
        "epsilon_w": EPSILON_W,
        "source_n": source_n,
        "target_n": TARGET_N,
        "adapt_gain": ADAPT_GAIN,
        "adapt_blend": ADAPT_BLEND,
        "adapt_max_radius_rg": float(ADAPT_MAX_RADIUS_RAW) if ADAPT_MAX_RADIUS_RAW else None,
        "freeze_prefix_nodes": FREEZE_PREFIX_NODES,
        "initial": asdict(initial),
        "final": asdict(final),
        "accepted_exploratory": final.maximum <= 3.0e-5,
        "accepted_preferred": final.maximum <= 1.0e-5,
        "peaks": peaks,
        "passes": passes,
    }
    target = CHECKPOINT_DIR / f"mdot2_fs080_eps{safe}_eta{params.closure.wind_launch_energy_multiplier:g}_N{TARGET_N}.npz"
    archive_suffix = f"_adapt{ADAPT_GAIN:g}" if ADAPT_GAIN > 0.0 else "_refined"
    archive = target.with_name(target.stem + archive_suffix + target.suffix)
    checkpoint_payload = {
        "x": state,
        "custom_grid_xi": np.asarray(params.disk.custom_grid_xi, dtype=float),
        "row_json": np.asarray(json.dumps(row, sort_keys=True)),
    }
    np.savez_compressed(archive, **checkpoint_payload)
    if row["accepted_exploratory"]:
        np.savez_compressed(target, **checkpoint_payload)
    output = ROOT / "outputs/tables" / f"unified_conservative_wind_refinement_eps{safe}_N{TARGET_N}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


if __name__ == "__main__":
    run()
