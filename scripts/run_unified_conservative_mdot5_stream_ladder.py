"""Continue the unified conservative Mdot=5 disk onto a compact stream source."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from imri_qpe.layer1_hill_flow import circularization_radius, hill_radius
from imri_qpe.layer3_minidisk_1d import (
    ConservativeBoundary,
    PhysicalTransportClosure,
    conservative_residual_audit,
    conservative_residual_profile,
    conservative_seed_from_legacy,
    reconstruct_conservative_state,
    remap_conservative_state,
    remap_profile_to_new_sonic_grid,
    solve_conservative_disk,
    transonic_profile_from_state_vector,
    unpack_conservative_state,
)
from imri_qpe.parameters import FiducialParams
from run_unified_conservative_no_wind_regression import (
    CHECKPOINT_DIR as NO_WIND_CHECKPOINT_DIR,
    load_anchor,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_mdot5_stream_ladder"
OUTPUT = ROOT / "outputs/tables/unified_conservative_mdot5_stream_ladder.json"
TARGET_ROUT_RG = float(os.environ.get("IMBH_CONSERVATIVE_M5_STREAM_ROUT_RG", "335"))
TARGET_RINJ_RG = float(os.environ.get("IMBH_CONSERVATIVE_M5_STREAM_RINJ_RG", "240"))
TARGET_N = int(os.environ.get("IMBH_CONSERVATIVE_M5_STREAM_N", "192"))
SOURCE_FRACTIONS = tuple(
    float(piece)
    for piece in os.environ.get(
        "IMBH_CONSERVATIVE_M5_STREAM_FRACTIONS",
        "0,0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3",
    ).split(",")
    if piece.strip()
)
MAX_NFEV = int(os.environ.get("IMBH_CONSERVATIVE_M5_STREAM_MAX_NFEV", "700"))
PASSES = int(os.environ.get("IMBH_CONSERVATIVE_M5_STREAM_PASSES", "4"))
SCOUT_TOLERANCE = float(os.environ.get("IMBH_CONSERVATIVE_M5_STREAM_SCOUT_TOLERANCE", "5e-5"))
RESUME_FRACTION_RAW = os.environ.get("IMBH_CONSERVATIVE_M5_STREAM_RESUME_FRACTION", "").strip()
RESUME_N = int(os.environ.get("IMBH_CONSERVATIVE_M5_STREAM_RESUME_N", str(TARGET_N)))


def _physical_circularization_radius() -> float:
    fiducial = FiducialParams()
    return float(circularization_radius(hill_radius(fiducial.a_cm, fiducial.q), fiducial.lambda_j))


def _source_grid(n_nodes: int, logR_son: float, r_g: float) -> tuple[float, ...]:
    """Return an outer-resolved grid with exact injection and support landmarks."""

    logR_out = np.log(TARGET_ROUT_RG * r_g)
    mapped = np.linspace(0.0, 1.0, n_nodes) ** 0.6
    landmarks_rg = [
        TARGET_RINJ_RG * np.exp(-0.08),
        TARGET_RINJ_RG,
        TARGET_RINJ_RG * np.exp(0.08),
        TARGET_ROUT_RG,
    ]
    landmarks = [
        (np.log(radius * r_g) - logR_son) / (logR_out - logR_son)
        for radius in landmarks_rg
        if np.exp(logR_son) / r_g < radius <= TARGET_ROUT_RG
    ]
    for value in landmarks:
        index = int(np.argmin(np.abs(mapped - value)))
        mapped[index] = value
    mapped = np.sort(np.unique(mapped))
    if mapped.size != n_nodes:
        raise ValueError("source-grid landmark insertion produced duplicate nodes")
    mapped[0] = 0.0
    mapped[-1] = 1.0
    return tuple(float(value) for value in mapped)


def _base_problem():
    legacy, disk = load_anchor()
    legacy_profile = transonic_profile_from_state_vector(legacy, disk)
    base_disk = replace(disk, n_nodes=128, custom_grid_xi=None, grid_power=0.6, max_nfev=1)
    legacy_seed = remap_profile_to_new_sonic_grid(legacy_profile, base_disk, method="pchip")
    closure = PhysicalTransportClosure(stream_circularization_radius=0.8 * base_disk.R_out)
    state, params = conservative_seed_from_legacy(legacy_seed, base_disk, closure)
    with np.load(NO_WIND_CHECKPOINT_DIR / "mdot5_N128.npz") as data:
        state = np.asarray(data["x"], dtype=float)
    params = replace(
        params,
        residual_tolerance=3.0e-5,
        max_nfev=MAX_NFEV,
        mass_weight=3.0,
        angular_momentum_weight=3.0,
        energy_flux_weight=1.0,
        energy_balance_weight=5.0,
        inner_mass_weight=5.0,
        sonic_mode="conservative",
        sonic_weight=30.0,
        jacobian_rel_step=1.0e-4,
    )
    return state, params


def _truncate_to_minidisk(state, params):
    *_fields, logR_son, _logR = unpack_conservative_state(state, params)
    target_disk = replace(
        params.disk,
        R_out_rg=TARGET_ROUT_RG,
        n_nodes=TARGET_N,
        custom_grid_xi=_source_grid(TARGET_N, logR_son, params.disk.r_g),
        grid_power=0.6,
        stream_source_fraction=0.0,
        stream_source_center_fraction=TARGET_RINJ_RG / TARGET_ROUT_RG,
        stream_source_log_width=0.08,
        stream_source_shape="compact_c2",
        stream_source_shape_blend=1.0,
        stream_torque_delta_l_fraction=0.0,
        stream_heating_efficiency=0.0,
        wind_energy_limited_epsilon=0.0,
    )
    state, params = remap_conservative_state(state, params, target_disk, method="pchip")
    logu, logT, F, j, _epsilon, _logR_son, logR = unpack_conservative_state(state, params)
    outer = reconstruct_conservative_state(
        float(logR[-1]),
        float(logu[-1]),
        float(logT[-1]),
        float(F[-1]),
        float(j[-1]),
        params.disk,
        params.flux_scales,
    )
    boundary = ConservativeBoundary(
        outer_log_temperature=float(logT[-1]),
        outer_log_omega_ratio=float(np.log(outer.Omega / outer.Omega_K)),
    )
    return state, replace(
        params,
        boundary=boundary,
        closure=PhysicalTransportClosure(
            stream_circularization_radius=_physical_circularization_radius(),
        ),
    )


def _summary(state, params) -> dict[str, float]:
    _logu, _logT, F, _j, _epsilon, logR_son, _logR = unpack_conservative_state(state, params)
    profile = conservative_residual_profile(state, params)
    score = np.max(
        np.vstack(
            [
                np.abs(profile[name])
                for name in ("radial", "mass", "angular_momentum", "energy", "energy_compatibility")
            ]
        ),
        axis=0,
    )
    peak = int(np.argmax(score))
    return {
        "F_inner": float(F[0]),
        "F_outer": float(F[-1]),
        "Rson_rg": float(np.exp(logR_son) / params.disk.r_g),
        "peak_interval_residual": float(score[peak]),
        "peak_interval_R_rg": float(profile["R_mid_rg"][peak]),
    }


def run() -> list[dict[str, object]]:
    state, params = _truncate_to_minidisk(*_base_problem())
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    previous_fraction = 0.0
    if RESUME_FRACTION_RAW:
        previous_fraction = float(RESUME_FRACTION_RAW)
        safe = str(previous_fraction).replace(".", "p")
        path = CHECKPOINT_DIR / f"mdot5_fs{safe}_Rout{TARGET_ROUT_RG:g}_N{RESUME_N}.npz"
        with np.load(path) as data:
            candidate = np.asarray(data["x"], dtype=float)
            if "custom_grid_xi" in data.files:
                source_params = replace(
                    params,
                    disk=replace(
                        params.disk,
                        n_nodes=RESUME_N,
                        custom_grid_xi=tuple(float(value) for value in data["custom_grid_xi"]),
                    ),
                )
            else:
                raise ValueError("Mdot=5 stream restart checkpoint is missing its custom grid")
        if RESUME_N == TARGET_N:
            state = candidate
            params = source_params
        else:
            state, params = remap_conservative_state(candidate, source_params, params.disk)

    rows: list[dict[str, object]] = []
    for fraction in SOURCE_FRACTIONS:
        params = replace(
            params,
            disk=replace(params.disk, stream_source_fraction=float(fraction)),
        )
        initial = conservative_residual_audit(state, params)
        final = initial
        passes: list[dict[str, object]] = []
        changed = not np.isclose(fraction, previous_fraction, rtol=0.0, atol=1.0e-14)
        if changed or initial.maximum > 3.0e-5:
            for pass_index in range(1, PASSES + 1):
                solved = solve_conservative_disk(state, params)
                state = solved.x
                final = solved.final_audit
                passes.append(
                    {"pass": pass_index, "nfev": solved.nfev, "final": asdict(final)}
                )
                if final.maximum <= 3.0e-5:
                    break
        row: dict[str, object] = {
            "stream_fraction": float(fraction),
            "Rout_rg": TARGET_ROUT_RG,
            "Rinj_rg": TARGET_RINJ_RG,
            "R_circ_rg": _physical_circularization_radius() / params.disk.r_g,
            "initial": asdict(initial),
            "final": asdict(final),
            "accepted_exploratory": final.maximum <= 3.0e-5,
            "accepted_preferred": final.maximum <= 1.0e-5,
            "continued_as_scout": final.maximum <= SCOUT_TOLERANCE,
            "summary": _summary(state, params),
            "passes": passes,
        }
        rows.append(row)
        safe = str(fraction).replace(".", "p")
        np.savez_compressed(
            CHECKPOINT_DIR / f"mdot5_fs{safe}_Rout{TARGET_ROUT_RG:g}_N{TARGET_N}.npz",
            x=state,
            custom_grid_xi=np.asarray(params.disk.custom_grid_xi, dtype=float),
            row_json=np.asarray(json.dumps(row, sort_keys=True)),
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        if not row["continued_as_scout"]:
            break
        previous_fraction = float(fraction)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
