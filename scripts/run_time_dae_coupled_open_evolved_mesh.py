"""Compare a bounded evolved interval on two coupled time-DAE meshes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    advance_coupled_time_dae_backward_euler,
    unpack_coupled_time_dae_state,
)
from imri_qpe.scales import eddington_luminosity

from run_time_dae_coupled_open_evolution import _build_case, _step_summary


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/tables/time_dae_coupled_open_repeated.json"
OUTPUT = ROOT / "outputs/tables/time_dae_coupled_open_evolved_mesh.json"
DT_FRACTION = 1.25e-8
STEPS = 4
FINE_DT_FRACTION = 6.25e-9
FINE_STEPS = 8


def _advance_mesh(n_inner: int, n_outer: int, dt_fraction: float, steps: int):
    context, state, loading_time = _build_case(
        n_inner, n_outer, interface_stencil_fraction=1.0
    )
    initial = np.array(state, copy=True)
    results = []
    for _step in range(steps):
        result = advance_coupled_time_dae_backward_euler(
            state,
            dt_fraction * loading_time,
            context,
            tolerance=1.0e-7,
            ledger_tolerance=1.0e-7,
            max_nfev=100,
        )
        results.append(result)
        if not result.accepted:
            break
        state = result.state
    final = results[-1]
    _inner, _outer, mdot, _angular, _energy = unpack_coupled_time_dae_state(
        final.state, context
    )
    ni = context.base.inner_params.n_nodes
    no = context.base.outer_grid.centers.size
    outer_slice = slice(2 * ni + 2, 2 * ni + 2 + 3 * no)
    profile = final.evaluation.outer.profile
    return {
        "inner_nodes": n_inner,
        "outer_cells": n_outer,
        "accepted_steps": sum(result.accepted for result in results),
        "all_steps_accepted": len(results) == steps
        and all(result.accepted for result in results),
        "dt_over_loading_time": dt_fraction,
        "requested_steps": steps,
        "time_over_loading_time": len(results) * dt_fraction,
        "elapsed_time_seconds": len(results) * dt_fraction * loading_time,
        "maximum_outer_q_change": float(
            np.max(np.abs(final.state[outer_slice] - initial[outer_slice]))
        ),
        "mdot_inner_over_supply": float(mdot[0] / context.mass_flux_scale),
        "mdot_outer_over_supply": float(mdot[-1] / context.mass_flux_scale),
        "maximum_H_over_R": float(
            np.max(profile.H / context.base.outer_grid.centers)
        ),
        "radiative_luminosity_over_eddington": float(
            np.sum(profile.radiative_loss_rate_cells)
            / eddington_luminosity(context.base.inner_params.M2_g)
        ),
        "final_step": _step_summary(final, context),
    }


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError("run the repeated-step control first")
    repeated = json.loads(SOURCE.read_text())
    coarse = repeated["steps"][STEPS - 1]
    fine = _advance_mesh(24, 16, FINE_DT_FRACTION, FINE_STEPS)
    comparison = {
        "inner_fraction_difference": float(
            fine["mdot_inner_over_supply"] - coarse["mdot_inner_over_supply"]
        ),
        "maximum_H_over_R_relative_difference": float(
            fine["maximum_H_over_R"] / coarse["maximum_H_over_R"] - 1.0
        ),
        "luminosity_relative_difference": float(
            fine["radiative_luminosity_over_eddington"]
            / coarse["radiative_luminosity_over_eddington"]
            - 1.0
        ),
    }
    report = {
        "dt_over_loading_time": DT_FRACTION,
        "steps": STEPS,
        "coarse_16_8": coarse,
        "fine_24_16": fine,
        "comparison": comparison,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
