"""Certify a causally outgoing inner plunge for global Roche evolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    advance_global_backward_euler,
    continue_transonic_supersonic_plunge,
    evaluate_global_rusanov_profile,
    global_inner_characteristic_audit,
    recover_global_primitives,
)

from run_global_physical_open_preflight import (
    _canonical_open_evaluation,
    _conservatively_mapped_global_state,
)
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/global_supersonic_plunge_preflight.json"
INNER_RADIUS_RG = 4.5
DT_LOADING_FRACTION = 1.0e-9


def _state_relative_difference(left, right) -> float:
    return max(
        float(
            np.max(
                np.abs(getattr(left, name) - getattr(right, name))
                / np.maximum(np.abs(getattr(right, name)), 1.0e-300)
            )
        )
        for name in (
            "mass",
            "radial_momentum",
            "angular_momentum",
            "total_energy",
        )
    )


def _advance(
    context,
    grid,
    initial,
    correction,
    stream,
    provider,
    dt: float,
    n_steps: int,
):
    current = initial
    records = []
    for _index in range(n_steps):
        result = advance_global_backward_euler(
            grid,
            current,
            context.base.inner_params.M2_g,
            dt,
            alpha=context.base.alpha,
            reference_state=initial,
            boundary_mode="roche_outer",
            stress_boundary_mode="outer_zero_torque",
            include_radiative_cooling=True,
            include_vertical_column_work=True,
            external_sources=stream,
            jacobian_mode="sparse_forward",
            specific_mechanical_energy_correction=correction,
            outer_overflow_provider=provider,
            max_nfev=300,
        )
        records.append(
            {
                "accepted": result.accepted,
                "message": result.message,
                "nfev": result.nfev,
                "maximum_scaled_residual": result.maximum_scaled_residual,
                "maximum_storage_scaled_ledger_defect": (
                    result.maximum_storage_scaled_ledger_defect
                ),
            }
        )
        if not result.accepted:
            break
        current = result.state
    return current, records


def _mesh_run(context, evaluation, n_cells: int) -> dict:
    grid, initial, correction, stream, stream_rate, provider = _prepared_case(
        context,
        evaluation,
        n_cells,
        inner_radius_rg=INNER_RADIUS_RG,
    )
    mass = context.base.inner_params.M2_g
    loading_time = float(np.sum(initial.mass) / stream_rate)
    primitives = recover_global_primitives(
        grid,
        initial,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    characteristic = global_inner_characteristic_audit(primitives)
    profile = evaluate_global_rusanov_profile(
        grid,
        initial,
        mass,
        reference_state=initial,
        boundary_mode="roche_outer",
        alpha=context.base.alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=True,
        external_sources=stream,
        primitives=primitives,
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=correction,
    )
    full_state, full_records = _advance(
        context,
        grid,
        initial,
        correction,
        stream,
        provider,
        DT_LOADING_FRACTION * loading_time,
        1,
    )
    half_state, half_records = _advance(
        context,
        grid,
        initial,
        correction,
        stream,
        provider,
        0.5 * DT_LOADING_FRACTION * loading_time,
        2,
    )
    final = recover_global_primitives(
        grid,
        half_state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    final_profile = evaluate_global_rusanov_profile(
        grid,
        half_state,
        mass,
        reference_state=initial,
        boundary_mode="roche_outer",
        alpha=context.base.alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=True,
        external_sources=stream,
        primitives=final,
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=correction,
    )
    return {
        "n_cells": n_cells,
        "inner_edge_radius_rg": grid.edges[0] / context.base.inner_params.r_g,
        "inner_center_radius_rg": grid.centers[0] / context.base.inner_params.r_g,
        "initial_inner_mach": characteristic.radial_mach_number,
        "initial_incoming_characteristics": characteristic.incoming_characteristics,
        "initial_inner_mass_flux_over_supply": (
            profile.face_fluxes.mass[0] / stream_rate
        ),
        "initial_outer_mass_flux_over_supply": (
            profile.face_fluxes.mass[-1] / stream_rate
        ),
        "inner_characteristic_projection_active": (
            profile.inner_characteristic_projection is not None
        ),
        "minimum_specific_internal_energy": float(
            np.min(primitives.specific_internal_energy)
        ),
        "maximum_H_over_R": float(
            np.max(np.asarray(primitives.vertical.H) / grid.centers)
        ),
        "full_step": full_records,
        "half_steps": half_records,
        "full_half_maximum_state_relative_difference": (
            _state_relative_difference(full_state, half_state)
        ),
        "final_inner_mass_flux_over_supply": (
            final_profile.face_fluxes.mass[0] / stream_rate
        ),
        "final_outer_mass_flux_over_supply": (
            final_profile.face_fluxes.mass[-1] / stream_rate
        ),
    }


def main() -> None:
    context, evaluation = _canonical_open_evaluation()
    params = context.base.inner_params
    plunge = continue_transonic_supersonic_plunge(
        evaluation.base.inner_profile,
        params,
        INNER_RADIUS_RG * params.r_g,
    )
    quadrature = []
    for n_cells in (64, 96, 128):
        grid_32, state_32, correction_32 = _conservatively_mapped_global_state(
            context,
            evaluation,
            n_cells,
            quadrature_order=32,
            inner_radius_rg=INNER_RADIUS_RG,
        )
        grid_64, state_64, correction_64 = _conservatively_mapped_global_state(
            context,
            evaluation,
            n_cells,
            quadrature_order=64,
            inner_radius_rg=INNER_RADIUS_RG,
        )
        quadrature.append(
            {
                "n_cells": n_cells,
                "grid_identical": bool(np.array_equal(grid_32.edges, grid_64.edges)),
                "maximum_state_relative_difference": (
                    _state_relative_difference(state_32, state_64)
                ),
                "maximum_mechanical_offset_relative_difference": float(
                    np.max(np.abs(correction_32 - correction_64))
                    / max(np.max(np.abs(correction_64)), 1.0e-300)
                ),
            }
        )
    report = {
        "inner_radius_rg": INNER_RADIUS_RG,
        "stationary_plunge": {
            "sonic_radius_rg": plunge.R[-1] / params.r_g,
            "inner_radial_mach_number": plunge.radial_mach_number[0],
            "inner_incoming_characteristics": int(
                plunge.incoming_characteristics[0]
            ),
            "inner_radial_velocity_over_c": float(
                -plunge.u[0] / 2.99792458e10
            ),
            "sonic_gradient_mismatch": plunge.sonic_gradient_mismatch,
            "maximum_scaled_differential_residual": (
                plunge.maximum_scaled_differential_residual
            ),
        },
        "quadrature_comparisons": quadrature,
        "mesh_runs": [
            _mesh_run(context, evaluation, n_cells)
            for n_cells in (64, 96, 128)
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
