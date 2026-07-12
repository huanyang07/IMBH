"""Run a bounded repeated-step and restart audit of the coupled open DAE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    advance_coupled_time_dae_backward_euler,
    load_coupled_time_dae_restart,
    save_coupled_time_dae_restart,
    unpack_coupled_time_dae_state,
    unpack_outer_primitives,
)
from imri_qpe.scales import eddington_luminosity

from run_time_dae_coupled_open_evolution import _build_case, _step_summary


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/time_dae_coupled_open_repeated.json"
RESTART = ROOT / "outputs/checkpoints/time_dae_coupled_open_restart.npz"
INNER_NODES = 16
OUTER_CELLS = 8
DT_FRACTION = 1.25e-8
STEPS = 8
RESTART_AFTER = 4


def _diagnostics(
    result, context, initial_state, initial_mass, loading_time, step_number
):
    ni = context.base.inner_params.n_nodes
    no = context.base.outer_grid.centers.size
    outer_slice = slice(2 * ni + 2, 2 * ni + 2 + 3 * no)
    _inner, _outer, mdot, _angular, _energy = unpack_coupled_time_dae_state(
        result.state, context
    )
    profile = result.evaluation.outer.profile
    return {
        "step": step_number,
        "time_over_loading_time": step_number * DT_FRACTION,
        "elapsed_time_seconds": step_number * DT_FRACTION * loading_time,
        "maximum_outer_q_change": float(
            np.max(np.abs(result.state[outer_slice] - initial_state[outer_slice]))
        ),
        "disk_mass_relative_change": float(
            np.sum(profile.mass_cells) / initial_mass - 1.0
        ),
        "maximum_H_over_R": float(
            np.max(profile.H / context.base.outer_grid.centers)
        ),
        "radiative_luminosity_over_eddington": float(
            np.sum(profile.radiative_loss_rate_cells)
            / eddington_luminosity(context.base.inner_params.M2_g)
        ),
        "mdot_inner_over_supply": float(mdot[0] / context.mass_flux_scale),
        "mdot_outer_over_supply": float(mdot[-1] / context.mass_flux_scale),
        "step_result": _step_summary(result, context),
    }


def main() -> None:
    context, initial_state, loading_time = _build_case(
        INNER_NODES, OUTER_CELLS, interface_stencil_fraction=1.0
    )
    _inner, initial_outer, _mdot, _angular, _energy = (
        unpack_coupled_time_dae_state(initial_state, context)
    )
    initial_sigma, _temperature, _omega = unpack_outer_primitives(
        initial_outer, context.base.outer_grid
    )
    initial_mass = float(np.sum(initial_sigma * context.base.outer_grid.area))
    state = np.array(initial_state, copy=True)
    rows = []
    restart_state_difference = None
    restart_continuation_difference = None
    prefetched_result = None
    for step_number in range(1, STEPS + 1):
        if prefetched_result is None:
            result = advance_coupled_time_dae_backward_euler(
                state,
                DT_FRACTION * loading_time,
                context,
                tolerance=1.0e-7,
                ledger_tolerance=1.0e-7,
                max_nfev=100,
            )
        else:
            result = prefetched_result
            prefetched_result = None
        rows.append(
            _diagnostics(
                result,
                context,
                initial_state,
                initial_mass,
                loading_time,
                step_number,
            )
        )
        if not result.accepted:
            break
        state = result.state
        if step_number == RESTART_AFTER:
            save_coupled_time_dae_restart(
                RESTART,
                state,
                context,
                elapsed_time=step_number * DT_FRACTION * loading_time,
                step_number=step_number,
            )
            restored = load_coupled_time_dae_restart(RESTART, context)
            restart_state_difference = float(
                np.max(np.abs(restored.state - state))
            )
            direct = advance_coupled_time_dae_backward_euler(
                state,
                DT_FRACTION * loading_time,
                context,
                tolerance=1.0e-7,
                ledger_tolerance=1.0e-7,
                max_nfev=100,
            )
            restarted = advance_coupled_time_dae_backward_euler(
                restored.state,
                DT_FRACTION * loading_time,
                context,
                tolerance=1.0e-7,
                ledger_tolerance=1.0e-7,
                max_nfev=100,
            )
            restart_continuation_difference = float(
                np.max(np.abs(direct.state - restarted.state))
            )
            prefetched_result = restarted
    report = {
        "inner_nodes": INNER_NODES,
        "outer_cells": OUTER_CELLS,
        "dt_over_loading_time": DT_FRACTION,
        "requested_steps": STEPS,
        "accepted_steps": sum(row["step_result"]["accepted"] for row in rows),
        "all_requested_steps_accepted": len(rows) == STEPS
        and all(row["step_result"]["accepted"] for row in rows),
        "restart_after_step": RESTART_AFTER,
        "restart_state_difference": restart_state_difference,
        "restart_continuation_difference": restart_continuation_difference,
        "steps": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
