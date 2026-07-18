from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldTemporalControllerConfig,
    advance_causal_five_field_step_doubling_backward_euler,
    audit_causal_five_field_endpoint_with_reference_uncertainty,
    audit_causal_five_field_reference_convergence,
    causal_backward_euler_horizon_budget_factor,
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

    with pytest.raises(ValueError, match="must not exceed output horizon"):
        CausalFiveFieldTemporalControllerConfig(
            step_config=_step_config(),
            cooling_inner_cutoff=(
                6.0 * context.grid.gravitational_radius
            ),
            minimum_dt=1.0e-10,
            maximum_dt=1.0e-5,
            output_horizon_seconds=5.0e-6,
        ).validated()


def test_causal_horizon_budget_factor_uses_linear_error_scaling() -> None:
    assert causal_backward_euler_horizon_budget_factor(0.0) == 2.0
    assert causal_backward_euler_horizon_budget_factor(0.4) == 2.0
    assert causal_backward_euler_horizon_budget_factor(2.0) == 0.4


def test_causal_reference_and_combined_endpoint_audits() -> None:
    gates = dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1)
    coarse = {name: 0.2 * gate for name, gate in gates.items()}
    fine = {name: 0.1 * gate for name, gate in gates.items()}
    convergence = audit_causal_five_field_reference_convergence(
        coarse,
        fine,
        gates,
        maximum_reference_uncertainty_fraction=0.25,
        minimum_observed_order=0.75,
        order_floor_fraction=1.0e-3,
    )
    assert convergence["passed"]
    assert all(
        row["observed_order"] == 1.0
        for row in convergence["observables"].values()
    )

    endpoint = {name: 0.8 * gate for name, gate in gates.items()}
    combined = (
        audit_causal_five_field_endpoint_with_reference_uncertainty(
            endpoint,
            fine,
            gates,
        )
    )
    assert combined["passed"]
    assert combined["maximum_combined_normalized_error"] == pytest.approx(
        0.9
    )

    endpoint["maximum_log_h_over_r_profile"] = (
        0.95 * gates["maximum_log_h_over_r_profile"]
    )
    combined = (
        audit_causal_five_field_endpoint_with_reference_uncertainty(
            endpoint,
            fine,
            gates,
        )
    )
    assert not combined["passed"]
    assert combined["violated_observables"] == [
        "maximum_log_h_over_r_profile"
    ]


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
    assert attempt.temporal_budget_fraction == 1.0
    assert attempt.effective_temporal_accuracy_gates == dict(
        CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
    )
    assert attempt.second_half_step is not None
    np.testing.assert_array_equal(
        result.state_vector,
        attempt.second_half_step.state_vector,
    )
    np.testing.assert_array_equal(
        result.physical_increment,
        result.state_vector - old_vector,
    )


def test_causal_temporal_controller_applies_horizon_budget() -> None:
    context = make_causal_five_field_regression_context(4)
    old_vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    config = CausalFiveFieldTemporalControllerConfig(
        step_config=_step_config(),
        cooling_inner_cutoff=(
            6.0 * context.grid.gravitational_radius
        ),
        minimum_dt=1.0e-10,
        maximum_dt=1.0e-5,
        physical_ledger_tolerance=1.0e-9,
        output_horizon_seconds=1.0e-4,
    ).validated()
    result = advance_causal_five_field_step_doubling_backward_euler(
        context,
        old_vector,
        1.0e-8,
        np.zeros_like(old_vector),
        1.0e-8,
        config,
    )

    assert result.accepted
    attempt = result.attempts[0]
    assert attempt.temporal_budget_fraction == pytest.approx(1.0e-4)
    for name, gate in CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1.items():
        assert attempt.effective_temporal_accuracy_gates[
            name
        ] == pytest.approx(1.0e-4 * gate)
    assert result.normalized_error is not None
    assert result.normalized_error < 1.0
    assert result.dt_next == 2.0e-8
