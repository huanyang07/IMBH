from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    causal_conservation_constrained_balanced_rom,
    causal_descriptor_explicit_matrices,
    causal_five_field_reduced_descriptor_matrices,
    causal_five_field_reduced_stationary_residual,
    causal_linear_initial_response,
    causal_linear_transfer_response,
    causal_log_time_quadrature,
    causal_lyapunov_metric_audit,
    causal_rom_initial_response,
    causal_rom_memory_kernel_actions,
    causal_stable_rational_krylov_rom,
    causal_stable_rom_initial_response,
    causal_stable_rom_transfer_response,
    causal_stream_descriptor_inputs,
    causal_truncate_mixed_mode_rom,
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
