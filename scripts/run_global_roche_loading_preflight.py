"""Run the bounded no-tide loading preflight with the physical Roche edge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GasRadiationHillRocheNozzleProvider,
    PaczynskiWiitaPotential,
    advance_global_backward_euler,
    fiducial_hill_roche_nozzle_geometry,
    global_compact_stream_cell_sources,
    recover_global_primitives,
)
from imri_qpe.scales import eddington_mdot

from run_global_physical_open_preflight import (
    _canonical_open_evaluation,
    _conservatively_mapped_global_state,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/global_roche_loading_preflight.json"
TOTAL_LOADING_FRACTION = 1.0e-9


def _prepared_case(
    context,
    evaluation,
    n_cells: int,
    *,
    inner_radius_rg: float | None = None,
):
    grid, state, correction = _conservatively_mapped_global_state(
        context,
        evaluation,
        n_cells,
        quadrature_order=32,
        inner_radius_rg=inner_radius_rg,
    )
    mass = context.base.inner_params.M2_g
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = 5.0 * eddington_mdot(mass)
    circularization_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(circularization_radius))
    stream_energy = float(
        potential.phi(circularization_radius)
        + 0.5 * (stream_l / circularization_radius) ** 2
    )
    stream = global_compact_stream_cell_sources(
        grid,
        stream_rate,
        center=240.0 * potential.r_g,
        log_width=0.08,
        specific_radial_velocity=0.0,
        specific_angular_momentum=stream_l,
        specific_total_energy=stream_energy,
    )
    provider = GasRadiationHillRocheNozzleProvider(
        fiducial_hill_roche_nozzle_geometry(),
        transverse_quadrature_zones=32,
    )
    return grid, state, correction, stream, stream_rate, provider


def _run(context, evaluation, n_cells: int, n_steps: int) -> dict:
    grid, initial, correction, stream, stream_rate, provider = _prepared_case(
        context, evaluation, n_cells
    )
    mass = context.base.inner_params.M2_g
    loading_time = float(np.sum(initial.mass) / stream_rate)
    dt = TOTAL_LOADING_FRACTION * loading_time / n_steps
    current = initial
    steps = []
    for _index in range(n_steps):
        result = advance_global_backward_euler(
            grid,
            current,
            mass,
            dt,
            alpha=context.base.alpha,
            reference_state=initial,
            boundary_mode="characteristic_inner_roche_outer",
            stress_boundary_mode="outer_zero_torque",
            include_radiative_cooling=True,
            include_vertical_column_work=True,
            external_sources=stream,
            jacobian_mode="sparse_forward",
            specific_mechanical_energy_correction=correction,
            outer_overflow_provider=provider,
            max_nfev=300,
        )
        boundary = result.profile.outer_roche_boundary
        steps.append(
            {
                "accepted": result.accepted,
                "message": result.message,
                "nfev": result.nfev,
                "maximum_scaled_residual": result.maximum_scaled_residual,
                "maximum_storage_scaled_ledger_defect": (
                    result.maximum_storage_scaled_ledger_defect
                ),
                "roche_choked": None if boundary is None else boundary.gate.choked,
                "roche_available_specific_energy": (
                    None
                    if boundary is None
                    else boundary.gate.available_specific_energy
                ),
                "outer_mass_flux_over_supply": (
                    result.profile.face_fluxes.mass[-1] / stream_rate
                ),
                "inner_mass_flux_over_supply": (
                    result.profile.face_fluxes.mass[0] / stream_rate
                ),
            }
        )
        if not result.accepted:
            break
        current = result.state
    final = recover_global_primitives(
        grid,
        current,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    return {
        "n_cells": n_cells,
        "n_steps": n_steps,
        "dt_seconds": dt,
        "total_time_over_loading_time": (
            len(steps) * dt / loading_time
        ),
        "all_steps_accepted": (
            len(steps) == n_steps and all(step["accepted"] for step in steps)
        ),
        "disk_mass_relative_change": float(
            np.sum(current.mass) / np.sum(initial.mass) - 1.0
        ),
        "maximum_H_over_R": float(
            np.max(np.asarray(final.vertical.H) / grid.centers)
        ),
        "minimum_temperature": float(np.min(final.temperature)),
        "steps": steps,
    }


def main() -> None:
    context, evaluation = _canonical_open_evaluation()
    runs = [
        _run(context, evaluation, n_cells, n_steps)
        for n_cells in (64, 96, 128)
        for n_steps in (1, 2)
    ]
    by_key = {(run["n_cells"], run["n_steps"]): run for run in runs}
    report = {
        "model": {
            "stream_supply_over_eddington": 5.0,
            "distributed_tide": False,
            "wind": False,
            "outer_boundary": "exact-EOS closed-to-choked Roche channel",
            "total_loading_fraction": TOTAL_LOADING_FRACTION,
        },
        "runs": runs,
        "temporal_comparisons": [
            {
                "n_cells": n_cells,
                "mass_change_difference": (
                    by_key[(n_cells, 2)]["disk_mass_relative_change"]
                    - by_key[(n_cells, 1)]["disk_mass_relative_change"]
                ),
                "maximum_H_over_R_relative_difference": (
                    by_key[(n_cells, 2)]["maximum_H_over_R"]
                    / by_key[(n_cells, 1)]["maximum_H_over_R"]
                    - 1.0
                ),
            }
            for n_cells in (64, 96, 128)
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
