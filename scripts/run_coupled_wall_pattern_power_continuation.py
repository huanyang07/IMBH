"""Continue the coupled ideal wall to binary pattern-speed power."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    audit_coupled_rank,
    fiducial_hill_tidal_geometry,
    hill_outer_torque_weights,
    solve_coupled_inner_outer_steady,
)
from imri_qpe.scales import eddington_luminosity

from run_coupled_inner_outer_interface_continuation import _load_finest_mesh
from run_coupled_inner_outer_rank_prototype import (
    _jsonable,
    _rank_summary,
    _stage_summary,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/coupled_wall_pattern_power_continuation.json"
CHECKPOINT = ROOT / "outputs/checkpoints/coupled_wall_pattern_power_full.npz"
POWER_STAGES = (0.0, 0.25, 0.50, 0.75, 1.0)


def _stage_metrics(result, context, geometry, weights):
    stage = _stage_summary(result, context)
    inner = result.evaluation.inner_profile
    outer = result.evaluation.outer_energy_profile
    transport = result.evaluation.outer_transport
    luminosity = float(
        (
            np.trapezoid(2.0 * np.pi * inner.R * inner.Q_rad, inner.R)
            + np.sum(outer.radiative_loss_rate_cells)
        )
        / eddington_luminosity(context.inner_params.M2_g)
    )
    torque = float(transport.viscous_torque_faces[-1])
    omega_out = float(transport.omega_faces[-1])
    added_heat = float(np.sum(outer.external_power_rate_cells))
    effective_external_power = -omega_out * torque + added_heat
    expected_external_power = -(
        (1.0 - context.wall_pattern_power_fraction) * omega_out
        + context.wall_pattern_power_fraction * geometry.pattern_omega
    ) * torque
    power_scale = max(abs(expected_external_power), abs(omega_out * torque), 1.0)
    stream_angular = float(np.sum(transport.source_angular_rate_cells))
    H_over_R = outer.H / context.outer_grid.centers
    active = weights > 0.0
    return {
        "pattern_power_fraction": context.wall_pattern_power_fraction,
        "accepted": result.accepted,
        "maximum_residual": result.maximum_residual,
        "nfev": result.nfev,
        "lambda0": stage["lambda0"],
        "sonic_radius_rg": stage["sonic_radius_rg"],
        "primitive_audits": stage["primitive_audits"],
        "composite_Lrad_over_LEdd": luminosity,
        "max_outer_H_over_R": float(np.max(H_over_R)),
        "max_tidal_band_H_over_R": float(np.max(H_over_R[active])),
        "weighted_tidal_band_H_over_R": float(
            np.sum(weights * H_over_R)
        ),
        "outer_wall_torque": torque,
        "wall_torque_over_stream_angular_flux": torque / stream_angular,
        "disk_outer_omega": omega_out,
        "binary_pattern_omega": geometry.pattern_omega,
        "disk_to_pattern_omega_ratio": omega_out / geometry.pattern_omega,
        "added_tidal_heat": added_heat,
        "effective_external_power": effective_external_power,
        "expected_external_power": expected_external_power,
        "relative_power_identity_mismatch": abs(
            effective_external_power - expected_external_power
        )
        / power_scale,
    }


def run():
    context, state = _load_finest_mesh()
    geometry = fiducial_hill_tidal_geometry()
    weights = hill_outer_torque_weights(
        context.outer_grid,
        geometry.hill_radius,
    )
    rows = []
    result = None
    last_accepted_state = None
    last_accepted_context = None
    last_accepted_result = None
    for fraction in POWER_STAGES:
        context = replace(
            context,
            wall_pattern_omega=geometry.pattern_omega,
            wall_pattern_power_fraction=fraction,
            wall_power_weights=weights,
        )
        result = solve_coupled_inner_outer_steady(
            state,
            context,
            tolerance=1.0e-7,
            max_nfev=100,
        )
        rows.append(_stage_metrics(result, context, geometry, weights))
        if not result.accepted:
            break
        state = result.state
        last_accepted_state = result.state
        last_accepted_context = context
        last_accepted_result = result

    assert result is not None
    rank = audit_coupled_rank(result.state, context) if result.accepted else None
    full_rank = bool(
        rank is not None
        and rank.ranks_by_relative_threshold["1e-10"] == result.state.size
        and rank.preboundary_nullity == 2
        and rank.interface_response_rank == 2
        and rank.sonic_rank == 2
    )
    reached_full = bool(
        len(rows) == len(POWER_STAGES)
        and rows[-1]["pattern_power_fraction"] == 1.0
        and rows[-1]["accepted"]
    )
    luminosity_change = (
        rows[-1]["composite_Lrad_over_LEdd"]
        / rows[0]["composite_Lrad_over_LEdd"]
        - 1.0
    )
    thickness_change = (
        rows[-1]["max_outer_H_over_R"] / rows[0]["max_outer_H_over_R"]
        - 1.0
    )
    validity_failures = [
        row
        for row in rows
        if row["max_tidal_band_H_over_R"] >= 0.3
    ]
    tidal_band_thick = bool(validity_failures)
    model_valid_rows = [
        row
        for row in rows
        if row["accepted"] and row["max_tidal_band_H_over_R"] < 0.3
    ]
    output = {
        "mesh": [context.inner_params.n_nodes, context.outer_grid.centers.size],
        "interface_rg": context.inner_params.R_out_rg,
        "power_stages_requested": list(POWER_STAGES),
        "hill_radius_secondary_rg": (
            geometry.hill_radius / context.inner_params.r_g
        ),
        "configured_truncation_radius_secondary_rg": (
            geometry.truncation_radius / context.inner_params.r_g
        ),
        "tidal_kernel_onset_hill_fraction": 0.35,
        "rows": rows,
        "reached_full_pattern_power": reached_full,
        "full_rank_gate": full_rank,
        "rank_audit": None if rank is None else _rank_summary(rank),
        "relative_luminosity_change": luminosity_change,
        "relative_max_outer_H_over_R_change": thickness_change,
        "tidal_band_reaches_H_over_R_0p3": tidal_band_thick,
        "first_tidal_band_validity_failure_fraction": (
            None
            if not validity_failures
            else validity_failures[0]["pattern_power_fraction"]
        ),
        "last_numerically_accepted_fraction": (
            None
            if last_accepted_context is None
            else last_accepted_context.wall_pattern_power_fraction
        ),
        "last_model_valid_fraction": (
            None
            if not model_valid_rows
            else model_valid_rows[-1]["pattern_power_fraction"]
        ),
        "pattern_power_gate": bool(
            reached_full
            and full_rank
            and rows[-1]["relative_power_identity_mismatch"] <= 1.0e-10
        ),
        "next_stage": (
            "promote_inner_mdot_and_test_open_overflow"
            if tidal_band_thick
            else "steady_stability_audit"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(_jsonable(output), indent=2, sort_keys=True) + "\n")
    if (
        last_accepted_state is not None
        and last_accepted_context is not None
        and last_accepted_result is not None
    ):
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CHECKPOINT,
            state=last_accepted_state,
            inner_radius=last_accepted_result.evaluation.inner_profile.R,
            outer_radius=last_accepted_context.outer_grid.centers,
            wall_power_weights=weights,
            pattern_omega=geometry.pattern_omega,
            pattern_power_fraction=(
                last_accepted_context.wall_pattern_power_fraction
            ),
        )
    return output


def main() -> None:
    print(json.dumps(_jsonable(run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
