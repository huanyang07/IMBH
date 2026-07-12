"""Run the single authorized zero-torque endpoint remap and refinement audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    evaluate_coupled_open_overflow_residual,
    solve_coupled_open_overflow_steady,
)

from run_coupled_inner_outer_mesh_certification import _load_source
from run_coupled_open_overflow_continuation import (
    _open_context,
    _target_mesh,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical/coupled_open_overflow_eigenvalue"
OUTPUT = ROOT / "outputs/tables/coupled_open_edge_asymptotic_audit.json"
CHECKPOINTS = ROOT / "outputs/checkpoints/coupled_open_edge_asymptotic_audit"


def _load_state(name: str) -> np.ndarray:
    with np.load(CANONICAL / name) as data:
        return np.asarray(data["state"], dtype=float)


def _residual_summary(state, context):
    evaluation = evaluate_coupled_open_overflow_residual(
        state,
        context,
        include_inner_profile=False,
    )
    base = evaluation.base
    return {
        "maximum": float(np.max(np.abs(evaluation.residual))),
        "stress_maximum": float(np.max(np.abs(base.outer_stress))),
        "stress_last": float(base.outer_stress[-1]),
        "radial_maximum": float(np.max(np.abs(base.outer_radial))),
        "radial_last": float(base.outer_radial[-1]),
        "energy_maximum": float(np.max(np.abs(base.outer_energy))),
        "energy_last": float(base.outer_energy[-1]),
        "inner_maximum": float(np.max(np.abs(base.inner_core))),
        "interface_maximum": float(
            np.max(np.abs(base.interface_boundary))
        ),
        "edge_boundary": float(evaluation.edge_boundary),
    }


def _result_summary(result, context):
    values = _residual_summary(result.state, context)
    values.update(
        {
            "accepted": bool(result.accepted),
            "nfev": int(result.nfev),
            "message": str(result.message),
            "mdot_inner": float(result.evaluation.mdot_inner),
            "mdot_outer": float(result.evaluation.mdot_outer),
        }
    )
    return values


def main() -> None:
    base, _wall_state = _load_source()
    context_96 = _open_context(base, 1.0)
    state_96 = _load_state("Ninner96_Nouter64.npz")
    context_144, _seed_144 = _target_mesh(
        context_96,
        state_96,
        144,
        96,
    )
    state_144 = _load_state("Ninner144_Nouter96.npz")

    _old_context, old_seed = _target_mesh(
        context_144,
        state_144,
        168,
        112,
        outer_remap="log_primitives",
    )
    context_168, asymptotic_seed = _target_mesh(
        context_144,
        state_144,
        168,
        112,
        outer_remap="zero_torque",
    )
    old_summary = _residual_summary(old_seed, context_168)
    asymptotic_summary = _residual_summary(asymptotic_seed, context_168)
    result_168 = solve_coupled_open_overflow_steady(
        asymptotic_seed,
        context_168,
        tolerance=1.0e-7,
        max_nfev=100,
    )
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHECKPOINTS / "Ninner168_Nouter112.npz",
        state=result_168.state,
        accepted=result_168.accepted,
        maximum_residual=result_168.maximum_residual,
    )
    report = {
        "old_seed": old_summary,
        "asymptotic_seed": asymptotic_summary,
        "Ninner168_Nouter112": _result_summary(result_168, context_168),
        "Ninner192_Nouter128": None,
    }
    if result_168.accepted:
        context_192, seed_192 = _target_mesh(
            context_168,
            result_168.state,
            192,
            128,
            outer_remap="zero_torque",
        )
        result_192 = solve_coupled_open_overflow_steady(
            seed_192,
            context_192,
            tolerance=1.0e-7,
            max_nfev=100,
        )
        np.savez_compressed(
            CHECKPOINTS / "Ninner192_Nouter128.npz",
            state=result_192.state,
            accepted=result_192.accepted,
            maximum_residual=result_192.maximum_residual,
        )
        report["Ninner192_Nouter128"] = _result_summary(
            result_192, context_192
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
