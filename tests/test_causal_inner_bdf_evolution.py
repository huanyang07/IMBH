from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldBDFRestart,
    advance_causal_five_field_increment_backward_euler,
    advance_causal_five_field_increment_bdf,
    causal_five_field_bdf_restarts_equal,
    causal_five_field_bdf_step_ledgers,
    evaluate_causal_five_field_increment_bdf,
    evolve_causal_five_field_fixed_bdf2,
    load_causal_five_field_bdf_restart,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
    save_causal_five_field_bdf_restart,
    unpack_causal_five_field_state,
)


def _step_config() -> CausalFiveFieldAdaptiveStepConfig:
    return CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=1.0e-10,
        maximum_dt=1.0e-5,
        maximum_scaled_primitive_change=0.2,
        maximum_scaled_total_change=0.25,
        residual_tolerance=1.0e-10,
        algebraic_residual_tolerance=1.0e-10,
        conservation_tolerance=1.0e-9,
        maximum_newton_iterations=12,
    ).validated()


def _seed():
    context = make_causal_five_field_regression_context(4)
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    return context, vector


def test_bdf1_sparse_step_matches_backward_euler() -> None:
    context, vector = _seed()
    predictor = np.zeros_like(vector)
    timestep = 1.0e-8
    config = _step_config()

    legacy = advance_causal_five_field_increment_backward_euler(
        context,
        vector,
        timestep,
        predictor,
        config,
    )
    generic = advance_causal_five_field_increment_bdf(
        context,
        vector,
        timestep,
        predictor,
        config,
        order=1,
    )

    assert legacy.accepted
    assert generic.accepted
    np.testing.assert_array_equal(generic.state_vector, legacy.state_vector)
    np.testing.assert_array_equal(
        generic.physical_increment,
        legacy.physical_increment,
    )
    assert generic.order == 1
    assert generic.history is not None
    assert (
        generic.maximum_discrete_ledger_relative_defect
        <= config.conservation_tolerance
    )


def test_bdf_step_can_reuse_one_jacobian_across_newton_iterations() -> None:
    context, vector = _seed()
    predictor = np.zeros_like(vector)
    timestep = 1.0e-8
    config = _step_config()

    rebuilt = advance_causal_five_field_increment_bdf(
        context,
        vector,
        timestep,
        predictor,
        config,
        order=1,
    )
    reused = advance_causal_five_field_increment_bdf(
        context,
        vector,
        timestep,
        predictor,
        replace(config, jacobian_reuse_iterations=12),
        order=1,
    )

    assert rebuilt.accepted
    assert reused.accepted
    assert rebuilt.jacobian_evaluations > 1
    assert reused.jacobian_evaluations == 1
    assert reused.jacobian_evaluations < rebuilt.jacobian_evaluations
    assert reused.maximum_scaled_residual <= config.residual_tolerance
    assert (
        reused.maximum_discrete_ledger_relative_defect
        <= config.conservation_tolerance
    )
    scale = np.maximum(np.abs(rebuilt.physical_increment), 1.0)
    assert (
        np.max(
            np.abs(
                reused.physical_increment - rebuilt.physical_increment
            )
            / scale
        )
        <= 1.0e-6
    )


def test_bdf_discrete_ledger_matches_summed_residual() -> None:
    context, vector = _seed()
    increment = np.zeros_like(vector)
    timestep = 2.5e-8
    discrete, physical = causal_five_field_bdf_step_ledgers(
        context,
        vector,
        increment,
        timestep,
        order=1,
        history=None,
    )
    evaluation = evaluate_causal_five_field_increment_bdf(
        increment,
        context,
        old_vector=vector,
        timestep_seconds=timestep,
        order=1,
    )

    np.testing.assert_allclose(
        discrete.closure_defect,
        C * timestep * np.sum(evaluation.conservation_rows, axis=0),
        rtol=2.0e-14,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        physical.actual_conserved_storage,
        0.0,
        rtol=0.0,
        atol=0.0,
    )


def test_fixed_bdf2_uses_one_startup_step_and_reaches_target() -> None:
    context, vector = _seed()
    progress: list[tuple[int, int]] = []
    result = evolve_causal_five_field_fixed_bdf2(
        context,
        vector,
        np.zeros_like(vector),
        1.0e-8,
        4.0e-8,
        4,
        _step_config(),
        progress=lambda completed, total, _state, _history: (
            progress.append((completed, total))
        ),
    )

    assert result.passed
    assert result.completed_steps == 4
    assert result.bdf1_steps == 1
    assert result.bdf2_steps == 3
    assert result.history is not None
    assert result.state_gates["passed"]
    assert result.maximum_discrete_ledger_relative_defect <= 1.0e-9
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]

    initial = unpack_causal_five_field_state(vector, 4)
    final = unpack_causal_five_field_state(result.state_vector, 4)
    expected_storage = np.sum(
        context.grid.cell_measures[:, None]
        * (final.conserved - initial.conserved),
        axis=0,
    )
    np.testing.assert_allclose(
        result.cumulative_physical_ledger.actual_conserved_storage,
        expected_storage,
        # Direct endpoint subtraction loses digits that the increment sum keeps.
        rtol=2.0e-8,
        atol=1.0e-8,
    )


def test_fixed_bdf2_accepts_wp10c7i_spatial_operator() -> None:
    context = make_causal_five_field_regression_context(
        4,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_rate_scheme="arithmetic_face",
        cell_source_quadrature="gauss_legendre_4_local_rates",
        cell_storage_quadrature="gauss_legendre_4",
    )
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    snapshots: dict[int, np.ndarray] = {}
    result = evolve_causal_five_field_fixed_bdf2(
        context,
        vector,
        np.zeros_like(vector),
        1.0e-8,
        4.0e-8,
        4,
        _step_config(),
        progress=lambda completed, _total, state, _history: (
            snapshots.__setitem__(completed, np.array(state, copy=True))
        ),
    )

    assert result.passed
    assert result.bdf1_steps == 1
    assert result.bdf2_steps == 3
    assert tuple(snapshots) == (1, 2, 3, 4)
    np.testing.assert_array_equal(snapshots[4], result.state_vector)
    assert result.maximum_discrete_ledger_relative_defect <= 1.0e-9
    assert result.cumulative_physical_ledger_relative_defect <= 1.0e-3


def test_fixed_bdf2_complete_restart_replays_bitwise(tmp_path) -> None:
    context, vector = _seed()
    split: dict[str, object] = {}

    def capture(
        completed: int,
        _total: int,
        state: np.ndarray,
        history,
    ) -> None:
        if completed == 2:
            split["state"] = np.array(state, copy=True)
            split["history"] = history

    full = evolve_causal_five_field_fixed_bdf2(
        context,
        vector,
        np.zeros_like(vector),
        1.0e-8,
        4.0e-8,
        4,
        _step_config(),
        progress=capture,
    )
    assert full.passed
    state = np.asarray(split["state"], dtype=float)
    history = split["history"]
    restart = CausalFiveFieldBDFRestart(
        state_vector=state,
        history=history,
        elapsed_time=2.0e-8,
        dt_next=1.0e-8,
        next_order=2,
        accepted_steps=2,
        rejected_attempts=0,
        provenance={"work_package": "WP10c7b", "case": "split"},
    )
    path = tmp_path / "bdf_split.npz"
    save_causal_five_field_bdf_restart(path, context, restart)
    restored = load_causal_five_field_bdf_restart(path, context)
    assert causal_five_field_bdf_restarts_equal(restart, restored)
    with pytest.raises(ValueError, match="spatial reconstruction"):
        load_causal_five_field_bdf_restart(
            path,
            replace(context, spatial_reconstruction="plm_smooth"),
        )
    for name, value in (
        ("boundary_trace_reconstruction", "plm_one_sided"),
        ("cell_rate_scheme", "quadratic_log_radius"),
        ("cell_source_quadrature", "gauss_legendre_4"),
        ("cell_storage_quadrature", "gauss_legendre_4"),
    ):
        with pytest.raises(ValueError, match=name.replace("_", " ")):
            load_causal_five_field_bdf_restart(
                path,
                replace(
                    context,
                    **{name: value},
                ),
            )

    replay = evolve_causal_five_field_fixed_bdf2(
        context,
        restored.state_vector,
        restored.history.previous_physical_increment,
        restored.history.previous_timestep_seconds,
        2.0e-8,
        2,
        _step_config(),
        startup_with_bdf1=False,
        initial_history=restored.history,
    )

    assert replay.passed
    assert replay.bdf1_steps == 0
    assert replay.bdf2_steps == 2
    np.testing.assert_array_equal(
        replay.state_vector,
        full.state_vector,
    )
    assert replay.history is not None
    np.testing.assert_array_equal(
        replay.history.previous_physical_increment,
        full.history.previous_physical_increment,
    )
    np.testing.assert_array_equal(
        replay.history.previous_vertical_killing_increment,
        full.history.previous_vertical_killing_increment,
    )


def test_fixed_bdf2_continuation_requires_matching_history() -> None:
    context, vector = _seed()
    with pytest.raises(ValueError, match="requires history"):
        evolve_causal_five_field_fixed_bdf2(
            context,
            vector,
            np.zeros_like(vector),
            1.0e-8,
            2.0e-8,
            2,
            _step_config(),
            startup_with_bdf1=False,
        )
