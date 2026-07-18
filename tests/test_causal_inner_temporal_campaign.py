from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldTemporalCampaignState,
    CausalFiveFieldTemporalControllerConfig,
    causal_five_field_temporal_campaign_states_equal,
    evolve_causal_five_field_horizon_budget,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)


def _controller_config(
    context,
    *,
    horizon: float | None,
) -> CausalFiveFieldTemporalControllerConfig:
    step_config = CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=1.0e-10,
        maximum_dt=1.0e-8,
        maximum_scaled_primitive_change=0.2,
        maximum_scaled_total_change=0.25,
        residual_tolerance=1.0e-10,
        algebraic_residual_tolerance=1.0e-10,
        conservation_tolerance=1.0e-9,
        maximum_newton_iterations=12,
    ).validated()
    return CausalFiveFieldTemporalControllerConfig(
        step_config=step_config,
        cooling_inner_cutoff=(
            6.0 * context.grid.gravitational_radius
        ),
        minimum_dt=1.0e-10,
        maximum_dt=1.0e-8,
        temporal_accuracy_gates=dict(
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
        ),
        physical_ledger_tolerance=1.0e-9,
        output_horizon_seconds=horizon,
    )


def _campaign_state(vector: np.ndarray) -> CausalFiveFieldTemporalCampaignState:
    return CausalFiveFieldTemporalCampaignState(
        state_vector=vector,
        previous_physical_increment=np.zeros_like(vector),
        elapsed_time=0.0,
        dt_next=1.0e-8,
        previous_dt=1.0e-8,
    )


def test_horizon_campaign_requires_horizon_budget_config() -> None:
    context = make_causal_five_field_regression_context(4)
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )

    with pytest.raises(
        ValueError,
        match="horizon-campaign contract is invalid",
    ):
        evolve_causal_five_field_horizon_budget(
            context,
            _campaign_state(vector),
            1.0e-8,
            _controller_config(context, horizon=None),
        )


def test_horizon_campaign_reaches_target_with_bitwise_adapter() -> None:
    context = make_causal_five_field_regression_context(4)
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    adapted: list[CausalFiveFieldTemporalCampaignState] = []
    progress = []

    def adapter(
        state: CausalFiveFieldTemporalCampaignState,
    ) -> CausalFiveFieldTemporalCampaignState:
        restored = CausalFiveFieldTemporalCampaignState(
            state_vector=np.array(state.state_vector, copy=True),
            previous_physical_increment=np.array(
                state.previous_physical_increment,
                copy=True,
            ),
            elapsed_time=state.elapsed_time,
            dt_next=state.dt_next,
            previous_dt=state.previous_dt,
            accepted_steps=state.accepted_steps,
            rejected_trials=state.rejected_trials,
            cumulative_budget_fraction=(
                state.cumulative_budget_fraction
            ),
        )
        adapted.append(restored)
        return restored

    result = evolve_causal_five_field_horizon_budget(
        context,
        _campaign_state(vector),
        1.0e-8,
        _controller_config(context, horizon=1.0e-8),
        state_adapter_after_accepted_steps=1,
        state_adapter=adapter,
        progress=progress.append,
    )

    assert result.passed
    assert result.target_reached
    assert result.budget_sum_passed
    assert result.state_adapter_requested
    assert result.state_adapter_performed
    assert result.final_state.elapsed_time == 1.0e-8
    assert result.final_state.accepted_steps == 1
    assert result.final_state.cumulative_budget_fraction == 1.0
    assert len(result.records) == 1
    assert len(progress) == 1
    assert len(adapted) == 1
    assert causal_five_field_temporal_campaign_states_equal(
        result.final_state,
        adapted[0],
    )


def test_horizon_campaign_rejects_state_changing_adapter() -> None:
    context = make_causal_five_field_regression_context(4)
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )

    def adapter(
        state: CausalFiveFieldTemporalCampaignState,
    ) -> CausalFiveFieldTemporalCampaignState:
        changed = np.array(state.state_vector, copy=True)
        changed[0] = np.nextafter(changed[0], np.inf)
        return CausalFiveFieldTemporalCampaignState(
            state_vector=changed,
            previous_physical_increment=np.array(
                state.previous_physical_increment,
                copy=True,
            ),
            elapsed_time=state.elapsed_time,
            dt_next=state.dt_next,
            previous_dt=state.previous_dt,
            accepted_steps=state.accepted_steps,
            rejected_trials=state.rejected_trials,
            cumulative_budget_fraction=(
                state.cumulative_budget_fraction
            ),
        )

    with pytest.raises(
        ValueError,
        match="state adapter did not preserve",
    ):
        evolve_causal_five_field_horizon_budget(
            context,
            _campaign_state(vector),
            1.0e-8,
            _controller_config(context, horizon=1.0e-8),
            state_adapter_after_accepted_steps=1,
            state_adapter=adapter,
        )
