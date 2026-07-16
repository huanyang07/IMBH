"""Project the causally outgoing plunge onto the production steady operator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalAdaptiveRestart,
    evaluate_global_rusanov_profile,
    global_conservative_rhs,
    global_roche_closure_diagnostic,
    load_global_adaptive_restart,
    make_global_mechanical_energy_reference,
    save_global_adaptive_restart,
    solve_global_inner_steady_projection,
)

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import _git_metadata
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
INNER_RADIUS_RG = 4.5


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart-output", type=Path, required=True)
    parser.add_argument("--maximum-nfev", type=int, default=100)
    return parser.parse_args()


def _normalized_global_balance(profile) -> dict[str, float]:
    rhs = global_conservative_rhs(profile.face_fluxes, profile.cell_sources)
    result = {}
    for name in (
        "mass",
        "radial_momentum",
        "angular_momentum",
        "total_energy",
    ):
        flux = np.asarray(getattr(profile.face_fluxes, name), dtype=float)
        source = np.asarray(getattr(profile.cell_sources, name), dtype=float)
        scale = max(
            abs(float(flux[0])),
            abs(float(flux[-1])),
            float(np.sum(np.abs(source))),
            1.0,
        )
        result[name] = float(np.sum(getattr(rhs, name)) / scale)
    return result


def main() -> None:
    arguments = _arguments()
    if arguments.n_cells not in {64, 96}:
        raise ValueError("plunge projection adoption supports N64 or N96")
    context, evaluation = _canonical_open_evaluation()
    grid, initial, correction, stream, stream_rate, provider = _prepared_case(
        context,
        evaluation,
        arguments.n_cells,
        inner_radius_rg=INNER_RADIUS_RG,
    )
    mass = context.base.inner_params.M2_g
    result = solve_global_inner_steady_projection(
        grid,
        initial,
        mass,
        alpha=context.base.alpha,
        reference_state=initial,
        external_sources=stream,
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=correction,
        maximum_nfev=arguments.maximum_nfev,
    )
    initial_profile = evaluate_global_rusanov_profile(
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
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=correction,
    )
    if not result.audit.accepted:
        raise RuntimeError(
            "inner plunge projection failed its declared acceptance gates"
        )
    loading_time = float(np.sum(initial.mass) / stream_rate)
    mechanical = make_global_mechanical_energy_reference(
        grid,
        correction,
        initial,
        provenance={
            "case": f"global-inner-plunge-projection-N{arguments.n_cells}",
            "reference": "canonical coupled open control",
        },
    )
    restart_path = arguments.restart_output
    if not restart_path.is_absolute():
        restart_path = ROOT / restart_path
    restart = GlobalAdaptiveRestart(
        state=result.state,
        reference_state=initial,
        mechanical_reference=mechanical,
        elapsed_time=0.0,
        dt_next=1.0e-8 * loading_time,
        accepted_steps=0,
        rejected_attempts=0,
        provenance={
            "case": f"global-inner-plunge-projection-N{arguments.n_cells}",
            "n_cells": arguments.n_cells,
            "inner_radius_rg": INNER_RADIUS_RG,
            "source_enabled": True,
            "git": _git_metadata(),
            "attempt_history": [],
            "projection_audit": asdict(result.audit),
        },
    )
    save_global_adaptive_restart(restart_path, grid, restart)
    loaded_grid, loaded = load_global_adaptive_restart(restart_path, grid=grid)
    if not np.array_equal(loaded_grid.edges, grid.edges):
        raise RuntimeError("projection restart changed the grid")
    for name in (
        "mass",
        "radial_momentum",
        "angular_momentum",
        "total_energy",
    ):
        if not np.array_equal(
            getattr(loaded.state, name), getattr(result.state, name)
        ):
            raise RuntimeError("projection restart changed the state")
    roche = result.profile.outer_roche_boundary
    if roche is None:
        raise RuntimeError("projected state lacks a Roche boundary audit")
    report = {
        "n_cells": arguments.n_cells,
        "inner_radius_rg": INNER_RADIUS_RG,
        "source_enabled": True,
        "classification": "local_causally_outgoing_projection",
        "global_steady_projection_allowed": False,
        "global_steady_incompatibility": (
            "closed Roche edge, zero outer torque, no tide, and no wind "
            "cannot dispose of the unaccreted stream mass and angular momentum"
        ),
        "initial_global_balance": _normalized_global_balance(initial_profile),
        "projected_global_balance": _normalized_global_balance(result.profile),
        "initial_inner_mass_flux_over_supply": float(
            initial_profile.face_fluxes.mass[0] / stream_rate
        ),
        "projected_inner_mass_flux_over_supply": float(
            result.profile.face_fluxes.mass[0] / stream_rate
        ),
        "projection_audit": asdict(result.audit),
        "roche_closure": asdict(
            global_roche_closure_diagnostic(
                roche, provider, mass_flux_scale=stream_rate
            )
        ),
        "restart": str(restart_path.relative_to(ROOT)),
        "restart_round_trip_exact": True,
    }
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
