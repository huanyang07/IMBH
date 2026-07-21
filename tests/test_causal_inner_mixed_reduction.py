from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    causal_conservation_constrained_balanced_rom,
    causal_descriptor_explicit_matrices,
    causal_finite_time_output_operator,
    causal_five_field_reduced_descriptor_matrices,
    causal_five_field_reduced_stationary_residual,
    causal_gate_normalized_finite_time_null_gain,
    causal_linear_initial_response,
    causal_linear_transfer_response,
    causal_log_time_quadrature,
    causal_lyapunov_metric_audit,
    causal_projective_ab2_prediction,
    causal_projective_euler_prediction,
    causal_rom_initial_response,
    causal_rom_memory_kernel_actions,
    causal_stable_rational_krylov_rom,
    causal_stable_rom_initial_response,
    causal_stable_rom_transfer_response,
    causal_stream_descriptor_inputs,
    causal_truncate_mixed_mode_rom,
    causal_weighted_constraint_null_basis,
    causal_weighted_constraint_null_projection,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_migration import (
    KerrSchildCellSourceRates,
)


def _stable_system() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dynamic = np.asarray(
        [
            [-0.2, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, -0.1, 0.0, 0.4, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, -2.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.0, -4.0, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, -8.0],
        ]
    )
    inputs = np.column_stack(
        (
            np.asarray((1.0, 0.5, 0.2, 0.1, 0.05, 0.025)),
            np.asarray((0.1, 0.2, 0.4, 0.8, 0.4, 0.2)),
        )
    )
    outputs = np.asarray(
        (
            (1.0, 0.0, 0.5, 0.0, 0.2, 0.0),
            (0.0, 1.0, 0.0, 0.5, 0.0, 0.2),
            (0.2, 0.1, 0.3, 0.4, 0.5, 0.6),
        )
    )
    protected = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        )
    )
    return dynamic, inputs, outputs, protected


def test_descriptor_explicit_matrices_solve_both_right_hand_sides() -> None:
    descriptor = np.diag((2.0, 3.0))
    stationary = np.asarray(((4.0, 1.0), (0.0, 6.0)))
    inputs = np.asarray(((2.0,), (3.0,)))

    dynamic, explicit_inputs, defect = causal_descriptor_explicit_matrices(
        descriptor,
        stationary,
        inputs,
    )

    assert np.allclose(descriptor @ dynamic, -stationary)
    assert np.allclose(descriptor @ explicit_inputs, inputs)
    assert defect < 1.0e-15


def test_stream_descriptor_inputs_preserve_linked_physical_column() -> None:
    stream = np.asarray(
        ((2.0, -1.0, 4.0, 8.0), (3.0, -1.5, 6.0, 12.0))
    )
    scales = np.arange(1.0, 11.0)

    inputs, names = causal_stream_descriptor_inputs(stream, scales)

    expected = np.zeros((2, 5))
    expected[:, :4] = stream
    assert names[0] == "physical_stream_amplitude"
    assert np.allclose(inputs[:, 0], expected.ravel() / scales)
    assert np.all(inputs.reshape(2, 5, 4)[:, 4, :] == 0.0)


def test_stream_descriptor_input_sign_matches_reduced_residual() -> None:
    context = make_causal_five_field_regression_context(2)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
    )
    source = context.stream_sources
    assert source is not None
    inputs, _names = causal_stream_descriptor_inputs(
        source.weighted_killing_source_per_ct,
        reduced["conservation_row_scales"],
    )

    step = 1.0e-5

    def scaled_context(factor: float):
        return replace(
            context,
            stream_sources=KerrSchildCellSourceRates(
                rest_mass=factor * source.rest_mass,
                radial_momentum_over_c=(
                    factor * source.radial_momentum_over_c
                ),
                angular_momentum_over_c=(
                    factor * source.angular_momentum_over_c
                ),
                killing_energy_over_c2=(
                    factor * source.killing_energy_over_c2
                ),
            ),
        )

    plus = causal_five_field_reduced_stationary_residual(
        state.primitives.ravel(),
        scaled_context(1.0 + step),
    )
    minus = causal_five_field_reduced_stationary_residual(
        state.primitives.ravel(),
        scaled_context(1.0 - step),
    )
    residual_derivative = (
        (plus - minus)
        / (2.0 * step)
        / reduced["conservation_row_scales"]
    )
    assert np.allclose(residual_derivative, -inputs[:, 0], rtol=2.0e-10)


def test_log_time_quadrature_is_positive_and_integrates_constant() -> None:
    times, weights = causal_log_time_quadrature(12.0, sample_count=20)

    assert times[0] == 0.0
    assert times[-1] == 12.0
    assert np.all(np.diff(times) > 0.0)
    assert np.all(weights > 0.0)
    assert np.sum(weights) == pytest.approx(12.0)


def test_weighted_constraint_projection_preserves_null_directions() -> None:
    constraints = np.asarray(((1.0, 1.0, 0.0), (0.0, 1.0, 1.0)))
    directions = np.column_stack(
        (
            np.asarray((2.0, -1.0, 3.0)),
            np.asarray((1.0, -1.0, 1.0)),
        )
    )
    projected, defects = causal_weighted_constraint_null_projection(
        directions,
        constraints,
        state_weights=np.asarray((1.0, 2.0, 4.0)),
    )

    assert np.max(np.abs(constraints @ projected)) < 1.0e-14
    assert np.max(defects) < 1.0e-14
    assert np.allclose(projected[:, 1], directions[:, 1])

    vector, defect = causal_weighted_constraint_null_projection(
        directions[:, 0],
        constraints,
    )
    assert vector.shape == (3,)
    assert defect < 1.0e-14
    assert np.max(np.abs(constraints @ vector)) < 1.0e-14


def test_weighted_constraint_null_basis_whitens_dependent_rows() -> None:
    constraints = np.asarray(
        (
            (1.0, 1.0, 0.0, 0.0),
            (1.0e12, 1.0e12, 0.0, 0.0),
            (0.0, 0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        )
    )
    weights = np.asarray((1.0, 4.0, 9.0, 16.0))

    audit = causal_weighted_constraint_null_basis(
        constraints,
        state_weights=weights,
        relative_rank_tolerance=1.0e-12,
    )

    assert audit.constraint_rank == 2
    assert audit.active_row_count == 3
    assert audit.nullity == 2
    assert audit.rank_threshold > 0.0
    assert np.isfinite(audit.condition_estimate)
    assert audit.whitened_constraint_defect < 1.0e-14
    assert audit.weighted_orthogonality_defect < 1.0e-14
    assert np.max(np.abs(constraints @ audit.basis)) < 1.0e-3
    assert np.allclose(
        audit.basis.T @ np.diag(weights) @ audit.basis,
        np.eye(2),
        atol=1.0e-14,
    )


def test_weighted_projection_preserves_dependent_row_rejection() -> None:
    constraints = np.asarray(
        (
            (1.0, 1.0, 0.0),
            (2.0, 2.0, 0.0),
        )
    )
    with pytest.raises(ValueError, match="linearly independent"):
        causal_weighted_constraint_null_projection(
            np.ones(3),
            constraints,
        )


def test_finite_time_output_operator_distinguishes_endpoint_increment() -> None:
    dynamic = np.diag((-1.0, -2.0))
    outputs = np.asarray(((1.0, 2.0), (3.0, 4.0)))
    horizon = np.log(2.0)

    endpoint = causal_finite_time_output_operator(
        dynamic,
        outputs,
        horizon,
        response_kind="endpoint",
    )
    increment = causal_finite_time_output_operator(
        dynamic,
        outputs,
        horizon,
        response_kind="increment",
    )

    expected_endpoint = outputs @ np.diag((0.5, 0.25))
    assert np.allclose(endpoint, expected_endpoint, atol=2.0e-14)
    assert np.allclose(increment, expected_endpoint - outputs, atol=2.0e-14)
    assert np.allclose(
        causal_finite_time_output_operator(
            dynamic,
            outputs,
            0.0,
            response_kind="increment",
        ),
        0.0,
    )


def test_gate_normalized_null_gain_reports_near_degenerate_subspace() -> None:
    constraints = np.asarray(((1.0, 0.0, 0.0),))
    weights = np.asarray((4.0, 9.0, 16.0))
    response = np.asarray(
        (
            (0.0, 6.0, 0.0),
            (0.0, 0.0, 11.64),
        )
    )
    gates = np.asarray((2.0, 3.0))

    audit = causal_gate_normalized_finite_time_null_gain(
        response,
        constraints,
        gates,
        state_weights=weights,
        leading_subspace_ratio=0.95,
    )

    assert np.allclose(audit.singular_values, (1.0, 0.97))
    assert audit.maximum_gain == pytest.approx(1.0)
    assert np.allclose(audit.per_output_maximum_gains, (1.0, 0.97))
    assert audit.maximum_per_output_gain == pytest.approx(1.0)
    assert audit.maximum_admissible_lower_gain == pytest.approx(1.0)
    assert audit.maximum_admissible_upper_gain == pytest.approx(1.0)
    assert audit.controlling_output_index == 0
    assert np.max(
        np.abs(constraints @ audit.controlling_state_direction)
    ) < 1.0e-14
    assert np.max(
        np.abs(audit.controlling_gate_normalized_output_response)
    ) == pytest.approx(audit.maximum_per_output_gain)
    assert audit.leading_subspace_dimension == 2
    assert audit.admissible_leading_subspace_dimension == 2
    assert np.max(
        np.abs(constraints @ audit.leading_state_subspace)
    ) < 1.0e-14
    assert np.allclose(
        audit.leading_state_subspace.T
        @ np.diag(weights)
        @ audit.leading_state_subspace,
        np.eye(2),
        atol=1.0e-14,
    )
    assert np.linalg.norm(
        audit.leading_gate_normalized_output_response
    ) == pytest.approx(audit.maximum_gain)
    assert np.allclose(
        audit.leading_raw_output_response / gates,
        audit.leading_gate_normalized_output_response,
    )


def test_per_output_null_gain_is_invariant_to_duplicate_output_rows() -> None:
    constraints = np.asarray(((1.0, 0.0),))
    response = np.asarray(((0.0, 2.0),))
    duplicated = np.vstack((response, response))

    single = causal_gate_normalized_finite_time_null_gain(
        response,
        constraints,
        np.ones(1),
    )
    repeated = causal_gate_normalized_finite_time_null_gain(
        duplicated,
        constraints,
        np.ones(2),
    )

    assert single.maximum_per_output_gain == pytest.approx(2.0)
    assert repeated.maximum_per_output_gain == pytest.approx(2.0)
    assert repeated.maximum_gain == pytest.approx(np.sqrt(2.0) * 2.0)


def test_pointwise_amplitude_contract_bounds_continuum_l2_gain() -> None:
    audit = causal_gate_normalized_finite_time_null_gain(
        np.asarray(((0.0, 10.0),)),
        np.asarray(((1.0, 0.0),)),
        np.ones(1),
        state_weights=np.ones(2),
        state_amplitudes_scaled=np.asarray((1.0, 0.1)),
    )

    assert audit.maximum_per_output_gain == pytest.approx(10.0)
    assert audit.per_output_l2_maximum_pointwise_ratios[0] == pytest.approx(
        10.0
    )
    assert audit.maximum_admissible_lower_gain == pytest.approx(1.0)
    assert audit.maximum_admissible_upper_gain == pytest.approx(1.0)
    assert np.max(
        np.abs(
            audit.controlling_admissible_state_direction
            / np.asarray((1.0, 0.1))
        )
    ) == pytest.approx(1.0)


def test_equal_window_projective_predictions_use_declared_secants() -> None:
    linear = np.asarray((0.0, 2.0, 4.0))
    assert causal_projective_euler_prediction(
        linear[:1],
        linear[1:2],
    )[0] == pytest.approx(linear[2])

    quadratic = np.asarray((0.0, 1.0, 4.0, 9.0))
    predicted = causal_projective_ab2_prediction(
        quadratic[:1],
        quadratic[1:2],
        quadratic[2:3],
    )
    assert predicted[0] == pytest.approx(8.0)


def test_balanced_rom_preserves_protected_values_and_dynamics() -> None:
    dynamic, inputs, outputs, protected = _stable_system()
    rom = causal_conservation_constrained_balanced_rom(
        dynamic,
        inputs,
        outputs,
        protected,
        order=5,
        horizon_seconds=5.0,
        state_weights=np.arange(1.0, 7.0),
        initial_directions=np.eye(6)[:, 2:],
        sample_count=10,
    )

    assert rom.order == 5
    assert np.allclose(rom.test_basis.T @ rom.trial_basis, np.eye(5))
    assert np.allclose(
        protected @ rom.trial_basis,
        np.column_stack((np.eye(2), np.zeros((2, 3)))),
    )
    assert np.allclose(
        rom.dynamic_matrix[:2],
        protected @ dynamic @ rom.trial_basis,
    )
    assert rom.biorthogonality_defect < 1.0e-10
    assert rom.protected_value_defect < 1.0e-10
    assert rom.protected_dynamics_defect < 1.0e-10


def test_full_order_balanced_rom_reproduces_linear_response() -> None:
    dynamic, inputs, outputs, protected = _stable_system()
    rom = causal_conservation_constrained_balanced_rom(
        dynamic,
        inputs,
        outputs,
        protected,
        order=6,
        horizon_seconds=3.0,
        initial_directions=np.eye(6),
        sample_count=10,
    )
    directions = np.column_stack((inputs, np.eye(6)[:, 4]))
    times = np.asarray((0.0, 0.1, 1.0, 3.0))
    full = causal_linear_initial_response(dynamic, directions, times)
    reduced = causal_rom_initial_response(rom, directions, times)

    assert np.allclose(reduced, full, rtol=2.0e-10, atol=2.0e-11)


def test_memory_kernel_vanishes_for_full_order_projection() -> None:
    dynamic, inputs, outputs, protected = _stable_system()
    rom = causal_conservation_constrained_balanced_rom(
        dynamic,
        inputs,
        outputs,
        protected,
        order=6,
        horizon_seconds=3.0,
        initial_directions=np.eye(6),
        sample_count=10,
    )

    kernel = causal_rom_memory_kernel_actions(
        dynamic,
        rom,
        np.asarray((0.0, 0.5, 2.0)),
    )

    assert np.max(np.abs(kernel)) < 1.0e-10


def test_truncated_rom_reuses_the_ordered_bpod_ladder() -> None:
    dynamic, inputs, outputs, protected = _stable_system()
    full = causal_conservation_constrained_balanced_rom(
        dynamic,
        inputs,
        outputs,
        protected,
        order=6,
        horizon_seconds=3.0,
        initial_directions=np.eye(6),
        sample_count=10,
    )

    truncated = causal_truncate_mixed_mode_rom(
        full,
        dynamic,
        inputs,
        outputs,
        order=4,
    )

    assert truncated.order == 4
    assert np.array_equal(truncated.trial_basis, full.trial_basis[:, :4])
    assert np.array_equal(truncated.test_basis, full.test_basis[:, :4])
    assert np.allclose(
        truncated.dynamic_matrix,
        truncated.test_basis.T @ dynamic @ truncated.trial_basis,
    )


def test_balanced_rom_can_report_numerically_resolved_rank() -> None:
    dynamic, inputs, outputs, protected = _stable_system()
    rank_limited_inputs = inputs[:, :1]
    rank_limited_outputs = outputs[:1]

    rom = causal_conservation_constrained_balanced_rom(
        dynamic,
        rank_limited_inputs,
        rank_limited_outputs,
        protected,
        order=6,
        horizon_seconds=1.0,
        sample_count=3,
        allow_rank_truncation=True,
    )

    assert 2 <= rom.order < 6


def test_lyapunov_metric_certifies_a_well_conditioned_stable_system() -> None:
    dynamic, _inputs, _outputs, _protected = _stable_system()

    audit = causal_lyapunov_metric_audit(
        dynamic,
        state_weights=np.arange(1.0, 7.0),
    )

    assert audit.accepted
    assert audit.positive_definite
    assert audit.residual_passed
    assert audit.minimum_eigenvalue > 0.0
    assert audit.relative_residual < 1.0e-10


def test_full_order_stable_rational_rom_reproduces_responses() -> None:
    dynamic, inputs, outputs, protected = _stable_system()
    timescales = np.asarray((0.05, 0.2, 1.0, 5.0))
    rom = causal_stable_rational_krylov_rom(
        dynamic,
        inputs,
        outputs,
        protected,
        order=6,
        timescales_seconds=timescales,
        state_weights=np.arange(1.0, 7.0),
        initial_directions=np.eye(6),
    )

    assert rom.stabilization_succeeded
    assert rom.maximum_real_eigenvalue < 0.0
    assert rom.stabilization_correction_fraction == 0.0
    assert rom.biorthogonality_defect < 1.0e-12
    assert rom.protected_value_defect < 1.0e-12
    assert rom.protected_dynamics_defect == 0.0

    directions = np.column_stack((inputs, np.eye(6)[:, -1]))
    times = np.asarray((0.0, 0.1, 1.0, 3.0))
    full_initial = causal_linear_initial_response(
        dynamic,
        directions,
        times,
    )
    reduced_initial = causal_stable_rom_initial_response(
        rom,
        directions,
        times,
    )
    assert np.allclose(
        reduced_initial,
        full_initial,
        rtol=2.0e-10,
        atol=2.0e-11,
    )

    full_transfer = causal_linear_transfer_response(
        dynamic,
        directions,
        outputs,
        timescales,
    )
    reduced_transfer = causal_stable_rom_transfer_response(
        rom,
        directions,
        timescales,
    )
    assert np.allclose(
        reduced_transfer,
        full_transfer,
        rtol=2.0e-10,
        atol=2.0e-11,
    )


def test_stabilization_preserves_protected_reduced_dynamics() -> None:
    dynamic = np.asarray(
        (
            (-1.0, 40.0, 0.0, 0.0),
            (0.0, -2.0, 40.0, 0.0),
            (0.0, 0.0, -3.0, 40.0),
            (0.0, 0.0, 0.0, -4.0),
        )
    )
    inputs = np.eye(4)[:, :2]
    outputs = np.asarray(((1.0, 0.0, 0.0, 1.0),))
    protected = np.asarray(((1.0, 0.0, 0.0, 0.0),))
    rom = causal_stable_rational_krylov_rom(
        dynamic,
        inputs,
        outputs,
        protected,
        order=3,
        timescales_seconds=np.asarray((0.01, 0.1, 1.0)),
        initial_directions=np.eye(4)[:, 2:],
    )

    assert rom.stabilization_succeeded
    assert rom.maximum_real_eigenvalue <= 1.0e-8
    assert rom.protected_value_defect < 1.0e-11
    assert rom.protected_dynamics_defect < 1.0e-11
