from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_BDF2_MAXIMUM_STEP_RATIO,
    CausalFiveFieldBDFRestart,
    audit_causal_five_field_dae_jacobian,
    causal_bdf_coefficients,
    causal_bdf_discrete_ledger,
    causal_bdf_increment_rate,
    causal_bdf_quadratic_history_predictor,
    causal_five_field_bdf_history,
    causal_five_field_bdf_restarts_equal,
    causal_five_field_dae_scaling,
    causal_five_field_path_temporal_storage_increment,
    causal_five_field_state_from_primitives,
    causal_trapezoidal_physical_interval_ledger,
    evaluate_causal_five_field_increment_bdf,
    evaluate_causal_five_field_increment_backward_euler,
    evaluate_causal_five_field_dae,
    load_causal_five_field_bdf_restart,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
    save_causal_five_field_bdf_restart,
    unpack_causal_five_field_state,
)


def test_bdf_coefficients_cover_order_one_and_equal_step_order_two() -> None:
    first = causal_bdf_coefficients(1, 0.25)
    second = causal_bdf_coefficients(2, 0.25, 0.25)

    assert first.order == 1
    assert first.current_increment_coefficient == 1.0
    assert first.previous_increment_coefficient == 0.0
    assert first.step_ratio is None
    assert second.order == 2
    assert second.step_ratio == 1.0
    assert second.current_increment_coefficient == 1.5
    assert second.previous_increment_coefficient == -0.5

    with pytest.raises(ValueError, match="order"):
        causal_bdf_coefficients(3, 0.25, 0.25)
    with pytest.raises(ValueError, match="previous timestep"):
        causal_bdf_coefficients(2, 0.25)
    with pytest.raises(ValueError, match="stability bound"):
        causal_bdf_coefficients(
            2,
            CAUSAL_BDF2_MAXIMUM_STEP_RATIO + 1.0e-6,
            1.0,
        )


def test_variable_step_bdf2_is_exact_for_a_quadratic() -> None:
    previous_time = 0.3
    old_time = 1.0
    new_time = 1.4
    coefficients = causal_bdf_coefficients(
        2,
        new_time - old_time,
        old_time - previous_time,
    )
    quadratic = lambda value: value**2
    current_increment = quadratic(new_time) - quadratic(old_time)
    previous_increment = quadratic(old_time) - quadratic(previous_time)
    derivative = causal_bdf_increment_rate(
        current_increment,
        previous_increment,
        coefficients,
    )

    assert float(derivative) == pytest.approx(
        2.0 * new_time,
        rel=2.0e-15,
    )


def test_quadratic_history_predictor_is_exact_on_nonuniform_steps() -> None:
    older_time = 0.2
    previous_time = 0.7
    current_time = 1.4
    requested_timestep = 0.3
    polynomial = lambda value: 2.0 - 0.5 * value + 1.2 * value**2
    predicted = causal_bdf_quadratic_history_predictor(
        polynomial(current_time),
        polynomial(current_time) - polynomial(previous_time),
        current_time - previous_time,
        polynomial(previous_time) - polynomial(older_time),
        previous_time - older_time,
        requested_timestep,
    )

    assert float(predicted) == pytest.approx(
        polynomial(current_time + requested_timestep),
        rel=2.0e-15,
    )


def _integrate_scalar_relaxation(
    subdivisions: int,
    *,
    rate: float,
) -> float:
    timestep = 1.0 / subdivisions
    previous = 1.0
    current = previous / (1.0 + rate * timestep)
    coefficients = causal_bdf_coefficients(2, timestep, timestep)
    for _index in range(1, subdivisions):
        current_increment_coefficient = (
            coefficients.current_increment_coefficient
        )
        previous_increment_coefficient = (
            coefficients.previous_increment_coefficient
        )
        numerator = (
            (
                current_increment_coefficient
                - previous_increment_coefficient
            )
            * current
            + previous_increment_coefficient * previous
        )
        new = numerator / (
            current_increment_coefficient + rate * timestep
        )
        previous, current = current, new
    return current


def test_bdf2_stiff_scalar_relaxation_converges_at_second_order() -> None:
    rate = 4.0
    exact = np.exp(-rate)
    errors = [
        abs(
            _integrate_scalar_relaxation(subdivisions, rate=rate)
            - exact
        )
        for subdivisions in (40, 80, 160)
    ]
    orders = [
        np.log2(errors[index] / errors[index + 1])
        for index in range(2)
    ]

    assert min(orders) > 1.8


def _integrate_index_one_dae(subdivisions: int) -> tuple[float, float]:
    timestep = 1.0 / subdivisions
    previous_x = 1.0
    current_x = previous_x / (1.0 + 0.5 * timestep)
    current_z = 0.5 * current_x
    coefficients = causal_bdf_coefficients(2, timestep, timestep)
    for _index in range(1, subdivisions):
        a0 = coefficients.current_increment_coefficient
        a_previous = coefficients.previous_increment_coefficient
        matrix = np.asarray(
            [
                [a0 / timestep + 1.0, -1.0],
                [-0.5, 1.0],
            ]
        )
        right = np.asarray(
            [
                (
                    (a0 - a_previous) * current_x
                    + a_previous * previous_x
                )
                / timestep,
                0.0,
            ]
        )
        new_x, new_z = np.linalg.solve(matrix, right)
        assert new_z == pytest.approx(0.5 * new_x, abs=2.0e-15)
        previous_x, current_x = current_x, new_x
        current_z = new_z
    return current_x, current_z


def test_bdf2_index_one_dae_keeps_algebraic_constraint_and_order() -> None:
    exact = np.exp(-0.5)
    solutions = [
        _integrate_index_one_dae(subdivisions)
        for subdivisions in (20, 40, 80)
    ]
    errors = [abs(solution[0] - exact) for solution in solutions]
    orders = [
        np.log2(errors[index] / errors[index + 1])
        for index in range(2)
    ]

    assert min(orders) > 1.8
    for differential, algebraic in solutions:
        assert algebraic == pytest.approx(
            0.5 * differential,
            abs=2.0e-15,
        )


def test_manufactured_vertical_storage_derivative_is_second_order() -> None:
    endpoint = 1.0
    errors = []
    for timestep in (0.1, 0.05, 0.025):
        coefficients = causal_bdf_coefficients(
            2,
            timestep,
            timestep,
        )
        storage = np.sin
        current_increment = (
            storage(endpoint) - storage(endpoint - timestep)
        )
        previous_increment = (
            storage(endpoint - timestep)
            - storage(endpoint - 2.0 * timestep)
        )
        derivative = causal_bdf_increment_rate(
            current_increment,
            previous_increment,
            coefficients,
        )
        errors.append(abs(float(derivative) - np.cos(endpoint)))
    orders = [
        np.log2(errors[index] / errors[index + 1])
        for index in range(2)
    ]

    assert min(orders) > 1.9


def test_discrete_and_physical_ledgers_have_distinct_contracts() -> None:
    rate = 2.0
    timestep = 0.1
    coefficients = causal_bdf_coefficients(2, timestep, timestep)
    previous_previous = 1.0
    previous = np.exp(-rate * timestep)
    weighted_old_terms = (
        (
            coefficients.current_increment_coefficient
            - coefficients.previous_increment_coefficient
        )
        * previous
        + coefficients.previous_increment_coefficient
        * previous_previous
    )
    current = weighted_old_terms / (
        coefficients.current_increment_coefficient + rate * timestep
    )
    discrete = causal_bdf_discrete_ledger(
        current - previous,
        previous - previous_previous,
        rate * current,
        coefficients,
    )

    assert float(discrete.closure_defect) == pytest.approx(
        0.0,
        abs=3.0e-16,
    )

    defects = []
    for local_timestep in (0.1, 0.05, 0.025):
        old = 1.0
        new = np.exp(-rate * local_timestep)
        physical = causal_trapezoidal_physical_interval_ledger(
            new - old,
            rate * old,
            rate * new,
            local_timestep,
        )
        defects.append(abs(float(physical.closure_defect)))
    orders = [
        np.log2(defects[index] / defects[index + 1])
        for index in range(2)
    ]

    assert min(orders) > 2.9


def _three_level_five_field_states():
    context = make_causal_five_field_regression_context(4)
    current = make_causal_five_field_seed(context)
    previous_primitives = np.array(current.primitives, copy=True)
    new_primitives = np.array(current.primitives, copy=True)
    interior = slice(None, -1)
    previous_primitives[interior, 0] -= 1.0e-4
    previous_primitives[interior, 1] -= 2.0e-5
    previous_primitives[interior, 2] += 1.0e-5
    previous_primitives[interior, 3] -= 1.5e-4
    previous_primitives[interior, 4] *= 0.9998
    new_primitives[interior, 0] += 1.2e-4
    new_primitives[interior, 1] += 1.0e-5
    new_primitives[interior, 2] -= 1.5e-5
    new_primitives[interior, 3] += 2.0e-4
    new_primitives[interior, 4] *= 1.0003
    previous = causal_five_field_state_from_primitives(
        context,
        previous_primitives,
    )
    new = causal_five_field_state_from_primitives(
        context,
        new_primitives,
    )
    return (
        context,
        pack_causal_five_field_state(previous),
        pack_causal_five_field_state(current),
        pack_causal_five_field_state(new),
    )


def test_five_field_bdf1_evaluator_is_backward_euler_parity() -> None:
    context, _previous, current, new = _three_level_five_field_states()
    increment = new - current
    generic = evaluate_causal_five_field_increment_bdf(
        increment,
        context,
        old_vector=current,
        timestep_seconds=2.0e-4,
        order=1,
    )
    legacy = evaluate_causal_five_field_increment_backward_euler(
        increment,
        context,
        old_vector=current,
        timestep_seconds=2.0e-4,
    )

    np.testing.assert_array_equal(generic.residual, legacy.residual)
    np.testing.assert_array_equal(
        generic.temporal_conserved_storage,
        legacy.temporal_conserved_storage,
    )
    np.testing.assert_array_equal(
        generic.temporal_vertical_storage,
        legacy.temporal_vertical_storage,
    )


def test_five_field_bdf2_weights_declared_and_vertical_history() -> None:
    context, previous, current, new = _three_level_five_field_states()
    previous_increment = current - previous
    current_increment = new - current
    previous_dt = 3.0e-4
    current_dt = 2.0e-4
    history = causal_five_field_bdf_history(
        context,
        current,
        previous_increment,
        previous_dt,
    )
    evaluation = evaluate_causal_five_field_increment_bdf(
        current_increment,
        context,
        old_vector=current,
        timestep_seconds=current_dt,
        order=2,
        history=history,
    )
    coefficients = causal_bdf_coefficients(
        2,
        current_dt,
        previous_dt,
    )
    n_cells = context.grid.centers.size
    current_conserved = current_increment[: 5 * n_cells].reshape(
        n_cells,
        5,
    )
    previous_conserved = previous_increment[: 5 * n_cells].reshape(
        n_cells,
        5,
    )
    expected_conserved = (
        context.grid.cell_measures[:, None]
        * causal_bdf_increment_rate(
            current_conserved,
            previous_conserved,
            coefficients,
        )
        / C
    )
    current_state = unpack_causal_five_field_state(current, n_cells)
    new_state = unpack_causal_five_field_state(new, n_cells)
    current_vertical = causal_five_field_path_temporal_storage_increment(
        context,
        current_state.primitives,
        new_state.primitives,
    ).vertical_killing_increment
    expected_vertical = (
        context.grid.cell_measures[:, None]
        * causal_bdf_increment_rate(
            current_vertical,
            history.previous_vertical_killing_increment,
            coefficients,
        )
        / C
    )

    np.testing.assert_allclose(
        evaluation.temporal_conserved_storage,
        expected_conserved,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        evaluation.temporal_vertical_storage,
        expected_vertical,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_complete_bdf_restart_round_trips_bitwise(tmp_path) -> None:
    context, previous, current, _new = _three_level_five_field_states()
    history = causal_five_field_bdf_history(
        context,
        current,
        current - previous,
        2.0e-4,
    )
    restart = CausalFiveFieldBDFRestart(
        state_vector=current,
        history=history,
        elapsed_time=0.25,
        dt_next=2.5e-4,
        next_order=2,
        accepted_steps=8,
        rejected_attempts=1,
        provenance={"work_package": "WP10c7a", "case": "roundtrip"},
    )
    path = tmp_path / "causal_bdf_restart.npz"
    save_causal_five_field_bdf_restart(path, context, restart)
    restored = load_causal_five_field_bdf_restart(path, context)

    assert causal_five_field_bdf_restarts_equal(restart, restored)


def test_five_field_bdf2_increment_jacobian_remains_full_rank() -> None:
    context, previous, current, _new = _three_level_five_field_states()
    n_cells = context.grid.centers.size
    current_state = unpack_causal_five_field_state(current, n_cells)
    scaling = causal_five_field_dae_scaling(
        current_state,
        evaluate_causal_five_field_dae(current, context),
    )
    history = causal_five_field_bdf_history(
        context,
        current,
        current - previous,
        0.1,
    )
    audit = audit_causal_five_field_dae_jacobian(
        lambda increment: evaluate_causal_five_field_increment_bdf(
            increment,
            context,
            old_vector=current,
            timestep_seconds=0.1,
            order=2,
            history=history,
        ).residual,
        np.zeros_like(current),
        scaling,
        finite_difference_step=2.0e-6,
        rank_relative_threshold=1.0e-11,
    )

    assert audit.dimensions == (65, 65)
    assert audit.full_rank
