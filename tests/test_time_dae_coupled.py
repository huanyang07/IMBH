from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def coupled_time_case():
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_coupled_inner_outer_mesh_certification import _load_source
    from run_coupled_open_overflow_continuation import _open_context, _target_mesh

    from imri_qpe.layer3_minidisk_1d import (
        CoupledTimeDAEContext,
        evaluate_coupled_open_overflow_residual,
        pack_coupled_time_dae_state,
        solve_coupled_open_overflow_steady,
        unpack_coupled_open_state,
        unpack_coupled_state,
    )

    base, _wall_state = _load_source()
    source_context = _open_context(base, 1.0)
    with np.load(
        ROOT
        / "results/canonical/coupled_open_overflow_eigenvalue/"
        "Ninner96_Nouter64.npz"
    ) as data:
        source_state = np.asarray(data["state"], dtype=float)
    open_context, seed = _target_mesh(source_context, source_state, 16, 8)
    steady = solve_coupled_open_overflow_steady(
        seed, open_context, tolerance=1.0e-7, max_nfev=50
    )
    assert steady.accepted
    evaluation = evaluate_coupled_open_overflow_residual(
        steady.state, open_context
    )
    base_state, _mdot_inner = unpack_coupled_open_state(
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
        / open_context.mass_flux_scale
    )
    return context, state, loading_time


def test_coupled_time_dae_has_exact_flux_primary_count(coupled_time_case) -> None:
    from imri_qpe.layer3_minidisk_1d import (
        coupled_time_dae_jacobian_sparsity,
        coupled_time_dae_row_slices,
        coupled_time_dae_state_size,
        evaluate_coupled_time_dae_backward_euler_residual,
        unpack_coupled_time_dae_state,
    )

    context, state, loading_time = coupled_time_case
    ni = context.base.inner_params.n_nodes
    no = context.base.outer_grid.centers.size
    assert state.size == 2 * ni + 5 * no + 5
    assert state.size == coupled_time_dae_state_size(context)
    rows = coupled_time_dae_row_slices(context)
    assert rows["open_edge"].stop == state.size
    sparsity = coupled_time_dae_jacobian_sparsity(context)
    assert sparsity.shape == (state.size, state.size)
    assert sparsity.nnz < state.size**2 // 3
    assert sparsity[rows["open_edge"].start, -2] == 1

    evaluation = evaluate_coupled_time_dae_backward_euler_residual(
        state, state, 1.0e-9 * loading_time, context
    )
    _inner, _outer, mdot, _angular, energy = unpack_coupled_time_dae_state(
        state, context
    )
    assert evaluation.residual.shape == state.shape
    assert evaluation.extracted_inner_flux.mdot == mdot[0]
    assert evaluation.outer.profile.energy_flux_faces[0] == energy
    assert np.sum(context.base.outer_template.source_mass_rate_cells) > 0.0
    assert np.sum(evaluation.outer.profile.radiative_loss_rate_cells) > 0.0


def test_coupled_time_dae_accepts_one_small_open_step(coupled_time_case) -> None:
    from imri_qpe.layer3_minidisk_1d import (
        advance_coupled_time_dae_backward_euler,
        unpack_coupled_time_dae_state,
    )

    context, state, loading_time = coupled_time_case
    result = advance_coupled_time_dae_backward_euler(
        state,
        1.0e-9 * loading_time,
        context,
        tolerance=1.0e-7,
        ledger_tolerance=1.0e-7,
        max_nfev=40,
    )
    _inner, _outer, mdot, _angular, _energy = unpack_coupled_time_dae_state(
        result.state, context
    )
    assert result.accepted
    assert result.linear_solver == "colored_central_sparse_trust_region_lsmr"
    assert result.maximum_residual < 1.0e-7
    assert mdot[0] > 0.0
    assert mdot[-1] < 0.0
    assert result.ledger.relative_mass_defect < 1.0e-7
    assert result.ledger.relative_angular_momentum_defect < 1.0e-7
    assert result.ledger.relative_energy_defect < 1.0e-7


def test_coupled_time_dae_sparsity_covers_numeric_jacobian(
    coupled_time_case,
) -> None:
    from imri_qpe.layer3_minidisk_1d import (
        coupled_time_dae_jacobian_sparsity,
        evaluate_coupled_time_dae_backward_euler_residual,
    )

    source_context, state, loading_time = coupled_time_case
    dt = 1.0e-9 * loading_time
    for fraction in (0.0, 1.0):
        context = replace(
            source_context,
            base=replace(
                source_context.base,
                interface_stencil_fraction=fraction,
            ),
        )

        def residual(values):
            return evaluate_coupled_time_dae_backward_euler_residual(
                values, state, dt, context
            ).residual

        jacobian = np.empty((state.size, state.size), dtype=float)
        for column in range(state.size):
            step = 1.0e-6 * max(1.0, abs(state[column]))
            plus = np.array(state, copy=True)
            minus = np.array(state, copy=True)
            plus[column] += step
            minus[column] -= step
            jacobian[:, column] = (residual(plus) - residual(minus)) / (
                2.0 * step
            )
        pattern = coupled_time_dae_jacobian_sparsity(context).toarray().astype(bool)
        row_scale = np.maximum(np.max(np.abs(jacobian), axis=1), 1.0e-14)
        relative = np.abs(jacobian) / row_scale[:, None]
        assert not np.any((~pattern) & (relative > 1.0e-10))


def test_coupled_time_dae_restart_round_trip(
    coupled_time_case, tmp_path: Path
) -> None:
    from imri_qpe.layer3_minidisk_1d import (
        load_coupled_time_dae_restart,
        save_coupled_time_dae_restart,
    )

    context, state, loading_time = coupled_time_case
    path = tmp_path / "restart.npz"
    save_coupled_time_dae_restart(
        path,
        state,
        context,
        elapsed_time=1.0e-7 * loading_time,
        step_number=3,
    )
    restored = load_coupled_time_dae_restart(path, context)
    np.testing.assert_array_equal(restored.state, state)
    assert restored.elapsed_time == 1.0e-7 * loading_time
    assert restored.step_number == 3
    incompatible = replace(
        context,
        base=replace(context.base, interface_stencil_fraction=1.0),
    )
    with pytest.raises(ValueError, match="interface stencil fraction"):
        load_coupled_time_dae_restart(path, incompatible)
