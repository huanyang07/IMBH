from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldTemporalControllerConfig,
    advance_causal_five_field_step_doubling_backward_euler,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


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


def _controller_config(
    context,
) -> CausalFiveFieldTemporalControllerConfig:
    return CausalFiveFieldTemporalControllerConfig(
        step_config=_step_config(),
        cooling_inner_cutoff=(
            6.0 * context.grid.gravitational_radius
        ),
        minimum_dt=1.0e-10,
        maximum_dt=1.0e-5,
        physical_ledger_tolerance=1.0e-9,
    ).validated()


def test_causal_regression_context_has_exact_stream_supply() -> None:
    context = make_causal_five_field_regression_context(4)

    assert context.grid.centers.size == 4
    assert context.stream_sources is not None
    assert np.isclose(
        np.sum(context.stream_sources.rest_mass),
        5.0 * eddington_mdot(FiducialParams().M2_g),
        rtol=2.0e-16,
        atol=0.0,
    )


def test_causal_temporal_controller_rejects_gate_schema_drift() -> None:
    context = make_causal_five_field_regression_context(4)
    gates = dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1)
    gates.pop("maximum_log_h_over_r_profile")

    with pytest.raises(ValueError, match="complete v1 schema"):
        CausalFiveFieldTemporalControllerConfig(
            step_config=_step_config(),
            cooling_inner_cutoff=(
                6.0 * context.grid.gravitational_radius
            ),
            minimum_dt=1.0e-10,
            maximum_dt=1.0e-5,
            temporal_accuracy_gates=gates,
        ).validated()

    with pytest.raises(ValueError, match="bracket one"):
        CausalFiveFieldTemporalControllerConfig(
            step_config=_step_config(),
            cooling_inner_cutoff=(
                6.0 * context.grid.gravitational_radius
            ),
            minimum_dt=1.0e-10,
            maximum_dt=1.0e-5,
            minimum_factor=1.0,
        ).validated()


def test_causal_temporal_controller_accepts_two_half_step_state() -> None:
    context = make_causal_five_field_regression_context(4)
    old_vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    result = advance_causal_five_field_step_doubling_backward_euler(
        context,
        old_vector,
        1.0e-8,
        np.zeros_like(old_vector),
        1.0e-8,
        _controller_config(context),
    )

    assert result.accepted
    assert result.dt_used == 1.0e-8
    assert result.dt_next == 2.0e-8
    assert result.normalized_error is not None
    assert result.normalized_error < 1.0
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.accepted
    assert attempt.failure_class == "none"
    assert attempt.full_contract.passed
    assert attempt.first_half_contract.passed
    assert attempt.second_half_contract is not None
    assert attempt.second_half_contract.passed
    assert attempt.second_half_step is not None
    np.testing.assert_array_equal(
        result.state_vector,
        attempt.second_half_step.state_vector,
    )
    np.testing.assert_array_equal(
        result.physical_increment,
        result.state_vector - old_vector,
    )
