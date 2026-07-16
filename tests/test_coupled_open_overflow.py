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


def test_supply_rescaling_preserves_all_stream_moments_and_flux_ratios() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_coupled_inner_outer_mesh_certification import _load_source

    from imri_qpe.layer3_minidisk_1d import (
        CoupledOpenOverflowContext,
        evaluate_coupled_open_overflow_residual,
        pack_coupled_open_state,
        rescale_coupled_open_supply,
        unpack_coupled_open_state,
        unpack_coupled_state,
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
    factor = 0.25
    scaled_context, scaled_state = rescale_coupled_open_supply(
        state,
        context,
        factor,
    )
    old_base_state, old_mdot = unpack_coupled_open_state(state, context)
    new_base_state, new_mdot = unpack_coupled_open_state(
        scaled_state,
        scaled_context,
    )
    old_components = unpack_coupled_state(old_base_state, context.base)
    new_components = unpack_coupled_state(
        new_base_state,
        scaled_context.base,
    )

    assert scaled_context.mass_flux_scale == factor * context.mass_flux_scale
    assert scaled_context.torque_scale == factor * context.torque_scale
    assert new_mdot == factor * old_mdot
    np.testing.assert_allclose(new_components[-2], factor * old_components[-2])
    np.testing.assert_allclose(new_components[-1], factor * old_components[-1])
    for name in (
        "mdot_faces",
        "angular_flux_faces",
        "source_mass_rate_cells",
        "source_angular_rate_cells",
        "source_total_energy_rate_cells",
        "mass_rate_cells",
    ):
        np.testing.assert_allclose(
            getattr(scaled_context.base.outer_template, name),
            factor * getattr(context.base.outer_template, name),
            rtol=0.0,
            atol=0.0,
        )
    assert scaled_state[-1] == state[-1]
    scaled_evaluation = evaluate_coupled_open_overflow_residual(
        scaled_state,
        scaled_context,
        include_inner_profile=False,
    )
    assert scaled_evaluation.residual.shape == scaled_state.shape
    assert np.all(np.isfinite(scaled_evaluation.residual))
