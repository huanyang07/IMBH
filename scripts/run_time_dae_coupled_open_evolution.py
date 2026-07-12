"""Run the first directly coupled no-tide open backward-Euler controls."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CoupledTimeDAEContext,
    advance_coupled_time_dae_backward_euler,
    coupled_time_dae_state_size,
    evaluate_coupled_open_overflow_residual,
    pack_coupled_time_dae_state,
    solve_coupled_open_overflow_steady,
    unpack_coupled_open_state,
    unpack_coupled_state,
    unpack_coupled_time_dae_state,
)

from run_coupled_inner_outer_mesh_certification import _load_source
from run_coupled_open_overflow_continuation import _open_context, _target_mesh


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "results/canonical/coupled_open_overflow_eigenvalue/"
    "Ninner96_Nouter64.npz"
)
OUTPUT = ROOT / "outputs/tables/time_dae_coupled_open_evolution.json"


def _build_case(
    n_inner: int,
    n_outer: int,
    *,
    interface_stencil_fraction: float = 0.0,
):
    base, _wall_state = _load_source()
    source_context = _open_context(base, 1.0)
    with np.load(SOURCE) as data:
        source_state = np.asarray(data["state"], dtype=float)
    open_context, seed = _target_mesh(
        source_context, source_state, n_inner, n_outer
    )
    steady = solve_coupled_open_overflow_steady(
        seed, open_context, tolerance=1.0e-7, max_nfev=100
    )
    if not steady.accepted:
        raise RuntimeError(
            f"small-mesh open seed failed: {steady.maximum_residual:.3e}"
        )
    if interface_stencil_fraction > 0.0:
        stages = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
        stages = [
            stage for stage in stages if stage < interface_stencil_fraction
        ] + [interface_stencil_fraction]
        for fraction in stages:
            open_context = replace(
                open_context,
                base=replace(
                    open_context.base,
                    interface_stencil_fraction=float(fraction),
                ),
            )
            steady = solve_coupled_open_overflow_steady(
                steady.state,
                open_context,
                tolerance=1.0e-7,
                max_nfev=100,
            )
            if not steady.accepted:
                raise RuntimeError(
                    "interface-stencil steady homotopy failed at "
                    f"{fraction:.3f}: {steady.maximum_residual:.3e}"
                )
    evaluation = evaluate_coupled_open_overflow_residual(
        steady.state, open_context
    )
    base_state, _mdot = unpack_coupled_open_state(
        steady.state, open_context
    )
    inner, sigma, temperature, omega, _angular, energy = unpack_coupled_state(
        base_state, evaluation.trial_context
    )
    context = CoupledTimeDAEContext(
        base=evaluation.trial_context,
        mass_flux_scale=open_context.mass_flux_scale,
        angular_flux_scale=evaluation.trial_context.angular_flux_scale,
        energy_flux_scale=evaluation.trial_context.energy_flux_scale,
    )
    state = pack_coupled_time_dae_state(
        inner,
        sigma,
        temperature,
        omega,
        evaluation.base.outer_transport.mdot_faces,
        evaluation.base.outer_transport.angular_flux_faces,
        energy,
        context,
    )
    loading_time = float(
        np.sum(sigma * context.base.outer_grid.area)
        / context.mass_flux_scale
    )
    return context, state, loading_time


def _step_summary(result, context):
    _inner, _outer, mdot, _angular, _energy = unpack_coupled_time_dae_state(
        result.state, context
    )
    return {
        "accepted": result.accepted,
        "maximum_residual": result.maximum_residual,
        "nfev": result.nfev,
        "mdot_inner_over_supply": float(mdot[0] / context.mass_flux_scale),
        "mdot_outer_over_supply": float(mdot[-1] / context.mass_flux_scale),
        "relative_mass_defect": result.ledger.relative_mass_defect,
        "relative_angular_momentum_defect": (
            result.ledger.relative_angular_momentum_defect
        ),
        "relative_energy_defect": result.ledger.relative_energy_defect,
        "maximum_interface_continuity": float(
            np.max(np.abs(result.evaluation.interface_continuity))
        ),
        "maximum_interface_flux_extraction": float(
            np.max(np.abs(result.evaluation.interface_flux_extraction))
        ),
        "maximum_inner_core": float(
            np.max(np.abs(result.evaluation.inner_core))
        ),
        "maximum_outer_mass": float(
            np.max(np.abs(result.evaluation.outer_mass))
        ),
        "maximum_outer_angular_momentum": float(
            np.max(np.abs(result.evaluation.outer_angular_momentum))
        ),
        "maximum_outer_energy": float(
            np.max(np.abs(result.evaluation.outer_energy))
        ),
        "maximum_outer_stress": float(
            np.max(np.abs(result.evaluation.outer_stress))
        ),
        "maximum_outer_radial": float(
            np.max(np.abs(result.evaluation.outer_radial))
        ),
        "maximum_outer_radial_index": int(
            np.argmax(np.abs(result.evaluation.outer_radial))
        ),
        "maximum_outer_radial_radius_rg": float(
            context.base.outer_grid.centers[
                np.argmax(np.abs(result.evaluation.outer_radial))
            ]
            / context.base.inner_params.r_g
        ),
        "open_edge_residual": abs(result.evaluation.open_edge),
    }


def _advance(state, fraction, loading_time, context):
    return advance_coupled_time_dae_backward_euler(
        state,
        fraction * loading_time,
        context,
        tolerance=1.0e-7,
        ledger_tolerance=1.0e-7,
        max_nfev=80,
    )


def _resolved_temporal_sequence(state, loading_time, context):
    fractions = (1.0e-7, 5.0e-8, 2.5e-8, 1.25e-8)
    single = {
        fraction: _advance(state, fraction, loading_time, context)
        for fraction in fractions
    }
    doubled = {}
    for fraction in fractions[1:]:
        doubled[fraction] = _advance(
            single[fraction].state, fraction, loading_time, context
        )
    ni = context.base.inner_params.n_nodes
    no = context.base.outer_grid.centers.size
    outer_slice = slice(2 * ni + 2, 2 * ni + 2 + 3 * no)
    errors = []
    for coarse, half in zip(fractions[:-1], fractions[1:]):
        full_difference = np.asarray(
            single[coarse].state - doubled[half].state, dtype=float
        )
        errors.append(
            {
                "dt_over_loading_time": coarse,
                "maximum_full_state_difference": float(
                    np.max(np.abs(full_difference))
                ),
                "maximum_outer_q_difference": float(
                    np.max(np.abs(full_difference[outer_slice]))
                ),
            }
        )
    for index in range(len(errors) - 1):
        errors[index]["full_state_error_ratio_to_next"] = float(
            errors[index]["maximum_full_state_difference"]
            / errors[index + 1]["maximum_full_state_difference"]
        )
        errors[index]["outer_q_error_ratio_to_next"] = float(
            errors[index]["maximum_outer_q_difference"]
            / errors[index + 1]["maximum_outer_q_difference"]
        )
    state_change = np.asarray(single[fractions[0]].state - state, dtype=float)
    return {
        "all_steps_accepted": bool(
            all(result.accepted for result in single.values())
            and all(result.accepted for result in doubled.values())
        ),
        "maximum_resolved_state_change": float(np.max(np.abs(state_change))),
        "maximum_resolved_outer_q_change": float(
            np.max(np.abs(state_change[outer_slice]))
        ),
        "errors": errors,
        "single_steps": {
            str(fraction): _step_summary(result, context)
            for fraction, result in single.items()
        },
        "second_half_steps": {
            str(fraction): _step_summary(result, context)
            for fraction, result in doubled.items()
        },
    }


def main() -> None:
    meshes = []
    cases = {}
    for n_inner, n_outer in ((16, 8), (24, 16)):
        context, state, loading_time = _build_case(
            n_inner, n_outer, interface_stencil_fraction=1.0
        )
        cases[(n_inner, n_outer)] = (context, state, loading_time)
        step = _advance(state, 1.0e-9, loading_time, context)
        meshes.append(
            {
                "inner_nodes": n_inner,
                "outer_cells": n_outer,
                "unknowns": coupled_time_dae_state_size(context),
                "dt_over_loading_time": 1.0e-9,
                "step": _step_summary(step, context),
            }
        )
    context, state, loading_time = cases[(16, 8)]
    temporal = _resolved_temporal_sequence(state, loading_time, context)
    report = {
        "architecture": "flux-primary direct inner coupling",
        "physics": "absolute stream source, radiative cooling, no tide, no wind",
        "meshes": meshes,
        "temporal_comparison": temporal,
        "temporal_order_status": (
            "resolved three-level full-step/two-half-step comparison"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
