"""Apply the eliminated-boundary DAE prototype to the canonical open state."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    advance_eliminated_outer_dae_backward_euler,
    audit_outer_dae_backward_euler_ledgers,
    evaluate_coupled_open_overflow_residual,
    pack_eliminated_boundary_coordinates,
    solve_eliminated_instantaneous_flux,
)

from run_coupled_inner_outer_mesh_certification import _load_source
from run_coupled_open_overflow_continuation import _open_context, _target_mesh


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical/coupled_open_overflow_eigenvalue"
OUTPUT = ROOT / "outputs/tables/time_dae_open_step_prototype.json"


def _load_state(name: str) -> np.ndarray:
    with np.load(CANONICAL / name) as data:
        return np.asarray(data["state"], dtype=float)


def main() -> None:
    base, _wall_state = _load_source()
    context_96 = _open_context(base, 1.0)
    state_96 = _load_state("Ninner96_Nouter64.npz")
    context_144, _seed = _target_mesh(context_96, state_96, 144, 96)
    state_144 = _load_state("Ninner144_Nouter96.npz")
    evaluation = evaluate_coupled_open_overflow_residual(
        state_144,
        context_144,
    )
    outer = evaluation.base.outer_transport
    energy = evaluation.base.outer_energy_profile
    grid = context_144.base.outer_grid
    params = context_144.base.inner_params
    coordinates = pack_eliminated_boundary_coordinates(
        grid,
        outer.surface_density,
        energy.temperature,
        outer.omega,
        params.M2_g,
        closure=context_144.base.outer_closure,
    )
    instantaneous = solve_eliminated_instantaneous_flux(
        coordinates,
        outer.mdot_faces,
        grid,
        params.M2_g,
        alpha=context_144.base.alpha,
        closure=context_144.base.outer_closure,
        stress_factor=context_144.base.stress_factor,
        tolerance=1.0e-7,
        max_nfev=300,
    )
    loading_time = float(
        np.sum(instantaneous.profile.mass_cells)
        / max(np.max(np.abs(instantaneous.mdot_faces)), 1.0)
    )
    steps = []
    for fraction in (1.0e-8, 1.0e-7, 1.0e-6):
        result = advance_eliminated_outer_dae_backward_euler(
            coordinates,
            instantaneous.mdot_faces,
            grid,
            params.M2_g,
            fraction * loading_time,
            alpha=context_144.base.alpha,
            closure=context_144.base.outer_closure,
            stress_factor=context_144.base.stress_factor,
            tolerance=1.0e-7,
            max_nfev=400,
        )
        ledger = audit_outer_dae_backward_euler_ledgers(
            instantaneous.profile,
            result.profile,
            fraction * loading_time,
        )
        steps.append(
            {
                "dt_over_loading_time": fraction,
                "accepted": result.accepted,
                "maximum_residual": result.maximum_residual,
                "nfev": result.nfev,
                "mdot_inner": float(result.mdot_faces[0]),
                "mdot_outer": float(result.mdot_faces[-1]),
                "maximum_radial_residual": float(
                    np.max(np.abs(result.profile.radial_residual))
                ),
                "relative_mass_defect": ledger.relative_mass_defect,
                "relative_angular_defect": (
                    ledger.relative_angular_momentum_defect
                ),
                "relative_energy_defect": ledger.relative_energy_defect,
            }
        )
    report = {
        "instantaneous": {
            "accepted": instantaneous.accepted,
            "maximum_residual": instantaneous.maximum_residual,
            "nfev": instantaneous.nfev,
            "mdot_inner": float(instantaneous.mdot_faces[0]),
            "mdot_outer": float(instantaneous.mdot_faces[-1]),
        },
        "steps": steps,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
