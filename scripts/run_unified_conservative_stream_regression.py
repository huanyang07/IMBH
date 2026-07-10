"""Physical stream-angular-momentum regression for the conservative solver."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer1_hill_flow import circularization_radius, hill_radius
from imri_qpe.layer3_minidisk_1d import (
    PhysicalTransportClosure,
    TransonicSlimParams,
    conservative_residual_audit,
    conservative_seed_from_legacy,
    remap_profile_to_new_sonic_grid,
    solve_conservative_disk,
    transonic_profile_from_state_vector,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "results/canonical/stream_no_wind_mdot2_fs080/state.npz"
OUTPUT = ROOT / "outputs/tables/unified_conservative_stream_regression.json"
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_stream_regression"
N_NODES = int(os.environ.get("IMBH_CONSERVATIVE_STREAM_N", "128"))
MAX_NFEV = int(os.environ.get("IMBH_CONSERVATIVE_STREAM_MAX_NFEV", "260"))
PASSES = int(os.environ.get("IMBH_CONSERVATIVE_STREAM_PASSES", "3"))
ENERGY_BALANCE_WEIGHT = float(os.environ.get("IMBH_CONSERVATIVE_STREAM_ENERGY_WEIGHT", "20"))
EVALUATE_ONLY = os.environ.get("IMBH_CONSERVATIVE_STREAM_EVALUATE_ONLY", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
RESUME = os.environ.get("IMBH_CONSERVATIVE_STREAM_RESUME", "0").strip().lower() in {
    "1", "true", "yes", "on"
}


def load_anchor() -> tuple[np.ndarray, TransonicSlimParams]:
    fiducial = FiducialParams()
    with np.load(ANCHOR) as data:
        grid = np.asarray(data["custom_grid_xi"], dtype=float)
        slopes = np.asarray(data["outer_match_log_slopes"], dtype=float)
        params = TransonicSlimParams(
            M2_g=fiducial.M2_g,
            Mdot_g_s=float(data["ratio"]) * eddington_mdot(fiducial.M2_g),
            alpha=0.01,
            mu_stress=0.0,
            stress_factor=1.0,
            R_out_rg=float(data["R_out_rg"]),
            n_nodes=int(data["n_nodes"]),
            grid_power=float(data["grid_power"]),
            custom_grid_xi=tuple(float(value) for value in grid),
            outer_closure=str(np.asarray(data["outer_closure"]).item()),
            outer_match_log_slopes=(float(slopes[0]), float(slopes[1])),
            stream_source_fraction=float(data["stream_source_fraction"]),
            stream_source_center_fraction=float(data["stream_source_center_fraction"]),
            stream_source_log_width=float(data["stream_source_log_width"]),
            # The retained checkpoint predates shape metadata.  Direct
            # residual replay proves that its production shape was tanh;
            # compact_c2 gives order-unity radial/energy defects.
            stream_source_shape="tanh",
            stream_torque_delta_l_fraction=float(data["stream_torque_delta_l_fraction"]),
            stream_torque_center_fraction=float(data["stream_torque_center_fraction"]),
            stream_torque_log_width=float(data["stream_torque_log_width"]),
            stream_heating_efficiency=0.0,
            wind_energy_limited_epsilon=0.0,
            residual_tol=1.0e-8,
            max_nfev=1,
        )
        z = np.asarray(data["z"], dtype=float)
    return z, params


def physical_circularization_radius() -> float:
    fiducial = FiducialParams()
    return float(circularization_radius(hill_radius(fiducial.a_cm, fiducial.q), fiducial.lambda_j))


def residual_aware_grid(params: TransonicSlimParams, n_nodes: int) -> tuple[float, ...]:
    """Downsample the accepted residual-aware coordinate density."""

    if params.custom_grid_xi is None:
        return tuple(float(value) for value in np.linspace(0.0, 1.0, n_nodes))
    source = np.asarray(params.custom_grid_xi, dtype=float)
    positions = np.linspace(0.0, float(source.size - 1), n_nodes)
    values = np.interp(positions, np.arange(source.size, dtype=float), source)
    values[0] = 0.0
    values[-1] = 1.0
    return tuple(float(value) for value in values)


def run() -> dict[str, object]:
    anchor, old_params = load_anchor()
    profile = transonic_profile_from_state_vector(anchor, old_params)
    remap_disk = replace(
        old_params,
        n_nodes=N_NODES,
        custom_grid_xi=residual_aware_grid(old_params, N_NODES),
        grid_power=0.6,
        max_nfev=1,
    )
    legacy_seed = remap_profile_to_new_sonic_grid(profile, remap_disk, method="pchip")
    closure = PhysicalTransportClosure(
        stream_circularization_radius=physical_circularization_radius(),
    )
    seed, params = conservative_seed_from_legacy(legacy_seed, remap_disk, closure)
    # The cumulative legacy offset is seed information only.  Production
    # angular transport is entirely in the explicit stream-carried term.
    params = replace(
        params,
        disk=replace(params.disk, stream_torque_delta_l_fraction=0.0),
        residual_tolerance=1.0e-5,
        max_nfev=MAX_NFEV,
        energy_balance_weight=ENERGY_BALANCE_WEIGHT,
    )
    checkpoint_path = CHECKPOINT_DIR / f"mdot2_fs080_N{N_NODES}.npz"
    if RESUME and checkpoint_path.exists():
        with np.load(checkpoint_path) as checkpoint:
            candidate = np.asarray(checkpoint["x"], dtype=float)
        if candidate.shape == seed.shape:
            seed = candidate
    initial = conservative_residual_audit(seed, params)
    pass_rows = []
    state = seed
    solved = None
    if not EVALUATE_ONLY:
        for pass_index in range(1, max(PASSES, 1) + 1):
            solved = solve_conservative_disk(state, params)
            state = solved.x
            pass_rows.append(
                {
                    "pass": pass_index,
                    "nfev": solved.nfev,
                    "accepted": solved.accepted,
                    "final": asdict(solved.final_audit),
                }
            )
            if solved.accepted:
                break
    final = initial if solved is None else solved.final_audit
    row: dict[str, object] = {
        "N": N_NODES,
        "stream_fraction": float(params.disk.stream_source_fraction),
        "R_circ_rg": float(physical_circularization_radius() / params.disk.r_g),
        "legacy_torque_fraction": float(old_params.stream_torque_delta_l_fraction),
        "production_torque_fraction": float(params.disk.stream_torque_delta_l_fraction),
        "initial": asdict(initial),
        "final": asdict(final),
        "accepted": bool(solved.accepted) if solved is not None else False,
        "passes": pass_rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint_path,
        x=state,
        row_json=np.asarray(json.dumps(row, sort_keys=True)),
    )
    print(json.dumps(row, sort_keys=True))
    return row


if __name__ == "__main__":
    run()
