"""Classify an evolved source-free state with one conservative mesh remap."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalCellSources,
    evaluate_global_rusanov_profile,
    global_fixed_radius_diagnostics,
    global_roche_closure_diagnostic,
    global_sonic_resolution_diagnostic,
    load_global_adaptive_restart,
    recover_global_primitives,
    remap_global_cell_integrals,
    remap_global_conservative_state,
)

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import FIXED_DIAGNOSTIC_RADII_RG
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
INNER_RADIUS_RG = 4.5


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-n-cells", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _component_totals(state) -> dict[str, float]:
    return {
        name: float(np.sum(getattr(state, name)))
        for name in (
            "mass",
            "radial_momentum",
            "angular_momentum",
            "total_energy",
        )
    }


def _relative_total_defects(source, target) -> dict[str, float]:
    source_totals = _component_totals(source)
    target_totals = _component_totals(target)
    return {
        name: float(
            abs(target_totals[name] - value) / max(abs(value), 1.0)
        )
        for name, value in source_totals.items()
    }


def _profile_record(
    *,
    grid,
    state,
    reference_state,
    correction,
    mass: float,
    alpha: float,
    provider,
    stream_rate: float,
    fixed_radii,
) -> dict:
    primitives = recover_global_primitives(
        grid,
        state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    profile = evaluate_global_rusanov_profile(
        grid,
        state,
        mass,
        reference_state=reference_state,
        boundary_mode="roche_outer",
        alpha=alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=True,
        external_sources=GlobalCellSources.zeros(grid.centers.size),
        primitives=primitives,
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=correction,
    )
    boundary = profile.outer_roche_boundary
    if boundary is None:
        raise RuntimeError("source-free snapshot lacks a Roche audit")
    return {
        "n_cells": int(grid.centers.size),
        "maximum_H_over_R": float(
            np.max(np.asarray(primitives.vertical.H) / grid.centers)
        ),
        "minimum_temperature": float(np.min(primitives.temperature)),
        "minimum_surface_density": float(
            np.min(primitives.surface_density)
        ),
        "inner_fluxes_over_source": {
            "mass": float(profile.face_fluxes.mass[0] / stream_rate),
            "angular_momentum": float(
                profile.face_fluxes.angular_momentum[0] / stream_rate
            ),
            "total_energy": float(
                profile.face_fluxes.total_energy[0] / stream_rate
            ),
        },
        "sonic_resolution": asdict(
            global_sonic_resolution_diagnostic(grid, primitives)
        ),
        "roche_closure": asdict(
            global_roche_closure_diagnostic(
                boundary, provider, mass_flux_scale=stream_rate
            )
        ),
        "fixed_radius_diagnostics": [
            asdict(item)
            for item in global_fixed_radius_diagnostics(
                grid,
                primitives,
                profile.face_fluxes,
                mass,
                fixed_radii,
            )
        ],
    }


def _fixed_radius_differences(source: dict, target: dict) -> list[dict]:
    source_rows = {
        row["radius"]: row for row in source["fixed_radius_diagnostics"]
    }
    target_rows = {
        row["radius"]: row for row in target["fixed_radius_diagnostics"]
    }
    rows = []
    for radius in sorted(set(source_rows) & set(target_rows)):
        old = source_rows[radius]
        new = target_rows[radius]
        rows.append(
            {
                "radius": radius,
                "radial_mach_difference": float(
                    new["radial_mach_number"] - old["radial_mach_number"]
                ),
                "log_surface_density_difference": float(
                    np.log(
                        new["surface_density"] / old["surface_density"]
                    )
                ),
                "log_temperature_difference": float(
                    np.log(new["temperature"] / old["temperature"])
                ),
            }
        )
    return rows


def main() -> None:
    arguments = _arguments()
    source_path = arguments.source
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source_grid, restart = load_global_adaptive_restart(source_path)
    context, evaluation = _canonical_open_evaluation()
    (
        target_grid,
        _target_initial,
        _target_correction,
        _target_stream,
        stream_rate,
        target_provider,
    ) = _prepared_case(
        context,
        evaluation,
        arguments.target_n_cells,
        inner_radius_rg=INNER_RADIUS_RG,
    )
    (
        _source_prepared_grid,
        _source_initial,
        _source_correction,
        _source_stream,
        _source_stream_rate,
        source_provider,
    ) = _prepared_case(
        context,
        evaluation,
        source_grid.centers.size,
        inner_radius_rg=INNER_RADIUS_RG,
    )
    target_state = remap_global_conservative_state(
        source_grid, restart.state, target_grid
    )
    target_reference = remap_global_conservative_state(
        source_grid, restart.reference_state, target_grid
    )
    correction_numerator = remap_global_cell_integrals(
        source_grid,
        restart.reference_state.mass
        * restart.mechanical_reference.specific_offset,
        target_grid,
    )
    target_correction = correction_numerator / target_reference.mass
    mass = context.base.inner_params.M2_g
    fixed_radii = tuple(
        radius_rg * context.base.inner_params.r_g
        for radius_rg in FIXED_DIAGNOSTIC_RADII_RG
        if target_grid.edges[0]
        <= radius_rg * context.base.inner_params.r_g
        <= target_grid.edges[-1]
    )
    source_record = _profile_record(
        grid=source_grid,
        state=restart.state,
        reference_state=restart.reference_state,
        correction=restart.mechanical_reference.specific_offset,
        mass=mass,
        alpha=context.base.alpha,
        provider=source_provider,
        stream_rate=stream_rate,
        fixed_radii=fixed_radii,
    )
    target_record = _profile_record(
        grid=target_grid,
        state=target_state,
        reference_state=target_reference,
        correction=target_correction,
        mass=mass,
        alpha=context.base.alpha,
        provider=target_provider,
        stream_rate=stream_rate,
        fixed_radii=fixed_radii,
    )
    report = {
        "classification": "conservative_remap_only_no_time_advance",
        "source_checkpoint": str(source_path.relative_to(ROOT)),
        "elapsed_time_seconds": restart.elapsed_time,
        "source_n_cells": int(source_grid.centers.size),
        "target_n_cells": int(target_grid.centers.size),
        "component_total_relative_defects": _relative_total_defects(
            restart.state, target_state
        ),
        "reference_total_relative_defects": _relative_total_defects(
            restart.reference_state, target_reference
        ),
        "source": source_record,
        "target": target_record,
        "fixed_radius_target_minus_source": _fixed_radius_differences(
            source_record, target_record
        ),
    }
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
