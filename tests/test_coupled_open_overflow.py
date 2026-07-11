from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_augmented_wall_exactly_recovers_base_coupled_root() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_coupled_inner_outer_mesh_certification import _load_source

    from imri_qpe.layer3_minidisk_1d import (
        CoupledOpenOverflowContext,
        evaluate_coupled_open_overflow_residual,
        pack_coupled_open_state,
    )

    base, base_state = _load_source()
    mass_scale = float(np.sum(base.outer_template.source_mass_rate_cells))
    torque_scale = max(
        abs(float(base.outer_template.viscous_torque_faces[-1])),
        1.0,
    )
    context = CoupledOpenOverflowContext(
        base=base,
        boundary_fraction=0.0,
        mass_flux_scale=mass_scale,
        torque_scale=torque_scale,
    )
    state = pack_coupled_open_state(
        base_state,
        float(base.outer_template.mdot_faces[0]),
        context,
    )
    evaluation = evaluate_coupled_open_overflow_residual(state, context)

    assert state.size == base_state.size + 1
    assert evaluation.residual.size == state.size
    assert evaluation.edge_boundary == 0.0
    assert evaluation.mdot_outer == 0.0
    np.testing.assert_allclose(
        evaluation.residual[:-1],
        evaluation.base.residual,
        rtol=0.0,
        atol=0.0,
    )
