"""Continue the physical stream branch from tanh to compact C2 injection."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    conservative_residual_audit,
    conservative_seed_from_legacy,
    remap_profile_to_new_sonic_grid,
    solve_conservative_disk,
    transonic_profile_from_state_vector,
)
from run_unified_conservative_stream_regression import (
    load_anchor,
    physical_circularization_radius,
    residual_aware_grid,
)
from imri_qpe.layer3_minidisk_1d import PhysicalTransportClosure


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_stream_regression"
OUTPUT = ROOT / "outputs/tables/unified_conservative_compact_source_homotopy.json"
N_NODES = int(os.environ.get("IMBH_CONSERVATIVE_COMPACT_N", "128"))
BLENDS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_CONSERVATIVE_COMPACT_BLENDS", "0,0.1,0.25,0.5,0.75,1").split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_CONSERVATIVE_COMPACT_MAX_NFEV", "180"))
PASSES = int(os.environ.get("IMBH_CONSERVATIVE_COMPACT_PASSES", "2"))
ENERGY_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_COMPACT_ENERGY_WEIGHT", "5"))
RESUME_ENDPOINT = os.environ.get("IMBH_CONSERVATIVE_COMPACT_RESUME_ENDPOINT", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
FORCE_POLISH = os.environ.get("IMBH_CONSERVATIVE_COMPACT_FORCE_POLISH", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
SONIC_MODE = os.environ.get("IMBH_CONSERVATIVE_COMPACT_SONIC_MODE", "legacy").strip().lower()
SONIC_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_COMPACT_SONIC_WEIGHT", "1"))
JACOBIAN_STEP_RAW = os.environ.get("IMBH_CONSERVATIVE_COMPACT_JACOBIAN_STEP", "").strip()


def start_state():
    anchor, old_params = load_anchor()
    profile = transonic_profile_from_state_vector(anchor, old_params)
    disk = replace(
        old_params,
        n_nodes=N_NODES,
        custom_grid_xi=residual_aware_grid(old_params, N_NODES),
        grid_power=0.6,
        max_nfev=1,
    )
    legacy = remap_profile_to_new_sonic_grid(profile, disk, method="pchip")
    closure = PhysicalTransportClosure(
        stream_circularization_radius=physical_circularization_radius(),
    )
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    params = replace(
        params,
        disk=replace(params.disk, stream_torque_delta_l_fraction=0.0),
        residual_tolerance=3.0e-5,
        max_nfev=MAX_NFEV,
        energy_balance_weight=ENERGY_WEIGHT,
        sonic_mode=SONIC_MODE,
        sonic_weight=SONIC_WEIGHT,
        jacobian_rel_step=float(JACOBIAN_STEP_RAW) if JACOBIAN_STEP_RAW else None,
    )
    checkpoint = CHECKPOINT_DIR / f"mdot2_fs080_N{N_NODES}.npz"
    with np.load(checkpoint) as data:
        state = np.asarray(data["x"], dtype=float)
    if state.shape != seed.shape:
        raise ValueError("physical tanh checkpoint has the wrong shape")
    return state, params


def run() -> list[dict[str, object]]:
    state, params = start_state()
    endpoint = CHECKPOINT_DIR / f"mdot2_fs080_compact_blend1p0_N{N_NODES}.npz"
    if RESUME_ENDPOINT and endpoint.exists():
        with np.load(endpoint) as data:
            candidate = np.asarray(data["x"], dtype=float)
        if candidate.shape == state.shape:
            state = candidate
    rows: list[dict[str, object]] = []
    for blend in BLENDS:
        params = replace(
            params,
            disk=replace(
                params.disk,
                stream_source_shape="compact_c2",
                stream_source_shape_blend=float(blend),
            ),
        )
        initial = conservative_residual_audit(state, params)
        pass_rows = []
        solved = None
        final = initial
        if FORCE_POLISH or initial.maximum > 3.0e-5:
            for pass_index in range(1, max(PASSES, 1) + 1):
                solved = solve_conservative_disk(state, params)
                state = solved.x
                final = solved.final_audit
                pass_rows.append(
                    {
                        "pass": pass_index,
                        "nfev": solved.nfev,
                        "accepted_exploratory": final.maximum <= 3.0e-5,
                        "accepted_preferred": final.maximum <= 1.0e-5,
                        "final": asdict(final),
                    }
                )
                if final.maximum <= 3.0e-5:
                    break
        row = {
            "blend": blend,
            "initial": asdict(initial),
            "final": asdict(final),
            "accepted_exploratory": final.maximum <= 3.0e-5,
            "accepted_preferred": final.maximum <= 1.0e-5,
            "passes": pass_rows,
        }
        rows.append(row)
        safe = str(blend).replace(".", "p")
        np.savez_compressed(
            CHECKPOINT_DIR / f"mdot2_fs080_compact_blend{safe}_N{N_NODES}.npz",
            x=state,
            row_json=np.asarray(json.dumps(row, sort_keys=True)),
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        if not row["accepted_exploratory"]:
            break
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
