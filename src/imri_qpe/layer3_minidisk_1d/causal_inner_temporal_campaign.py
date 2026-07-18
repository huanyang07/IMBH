"""Bounded temporal campaigns for the causal five-field DAE."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .causal_inner_dae import causal_five_field_dae_count
from .causal_inner_dae_system import CausalFiveFieldDAEContext
from .causal_inner_diagnostics import (
    audit_causal_five_field_state_gates,
)
from .causal_inner_temporal_controller import (
    CausalFiveFieldStepDoublingResult,
    CausalFiveFieldTemporalControllerConfig,
    advance_causal_five_field_step_doubling_backward_euler,
)


@dataclass(frozen=True)
class CausalFiveFieldTemporalCampaignState:
    """Complete in-memory state for one bounded temporal campaign."""

    state_vector: np.ndarray
    previous_physical_increment: np.ndarray
    elapsed_time: float
    dt_next: float
    previous_dt: float
    accepted_steps: int = 0
    rejected_trials: int = 0
    cumulative_budget_fraction: float = 0.0


@dataclass(frozen=True)
class CausalFiveFieldTemporalCampaignRecord:
    """One requested controller step and all of its internal attempts."""

    elapsed_time_before: float
    requested_timestep: float
    result: CausalFiveFieldStepDoublingResult
    rejected_trials: int


@dataclass(frozen=True)
class CausalFiveFieldTemporalCampaignResult:
    """One exact-horizon controller trajectory."""

    final_state: CausalFiveFieldTemporalCampaignState
    target_elapsed_time: float
    records: tuple[CausalFiveFieldTemporalCampaignRecord, ...]
    state_gates: dict[str, object]
    target_reached: bool
    budget_sum_passed: bool
    state_adapter_requested: bool
    state_adapter_performed: bool
    passed: bool
    message: str


def causal_five_field_temporal_campaign_states_equal(
    left: CausalFiveFieldTemporalCampaignState,
    right: CausalFiveFieldTemporalCampaignState,
) -> bool:
    """Return whether two campaign states are bitwise identical."""

    return bool(
        np.array_equal(left.state_vector, right.state_vector)
        and np.array_equal(
            left.previous_physical_increment,
            right.previous_physical_increment,
        )
        and left.elapsed_time == right.elapsed_time
        and left.dt_next == right.dt_next
        and left.previous_dt == right.previous_dt
        and left.accepted_steps == right.accepted_steps
        and left.rejected_trials == right.rejected_trials
        and left.cumulative_budget_fraction
        == right.cumulative_budget_fraction
    )


def _validated_campaign_state(
    state: CausalFiveFieldTemporalCampaignState,
    *,
    expected_size: int,
) -> CausalFiveFieldTemporalCampaignState:
    values = np.asarray(state.state_vector, dtype=float)
    increment = np.asarray(
        state.previous_physical_increment,
        dtype=float,
    )
    scalars = (
        state.elapsed_time,
        state.dt_next,
        state.previous_dt,
        state.cumulative_budget_fraction,
    )
    if (
        values.shape != (expected_size,)
        or increment.shape != values.shape
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(increment))
        or any(not np.isfinite(value) for value in scalars)
        or state.elapsed_time < 0.0
        or state.dt_next <= 0.0
        or state.previous_dt <= 0.0
        or state.accepted_steps < 0
        or state.rejected_trials < 0
        or state.cumulative_budget_fraction < 0.0
    ):
        raise ValueError("causal temporal-campaign state is invalid")
    return CausalFiveFieldTemporalCampaignState(
        state_vector=values,
        previous_physical_increment=increment,
        elapsed_time=float(state.elapsed_time),
        dt_next=float(state.dt_next),
        previous_dt=float(state.previous_dt),
        accepted_steps=int(state.accepted_steps),
        rejected_trials=int(state.rejected_trials),
        cumulative_budget_fraction=float(
            state.cumulative_budget_fraction
        ),
    )


def _landing_config(
    config: CausalFiveFieldTemporalControllerConfig,
    requested_timestep: float,
) -> CausalFiveFieldTemporalControllerConfig:
    if requested_timestep >= config.minimum_dt:
        return config
    step_config = replace(
        config.step_config,
        minimum_dt=min(
            config.step_config.minimum_dt,
            0.5 * requested_timestep,
        ),
    ).validated()
    return replace(
        config,
        step_config=step_config,
        minimum_dt=requested_timestep,
    ).validated()


def evolve_causal_five_field_horizon_budget(
    context: CausalFiveFieldDAEContext,
    initial_state: CausalFiveFieldTemporalCampaignState,
    target_elapsed_time: float,
    config: CausalFiveFieldTemporalControllerConfig,
    *,
    target_time_relative_tolerance: float = 5.0e-14,
    state_adapter_after_accepted_steps: int | None = None,
    state_adapter: Callable[
        [CausalFiveFieldTemporalCampaignState],
        CausalFiveFieldTemporalCampaignState,
    ]
    | None = None,
    progress: Callable[
        [CausalFiveFieldTemporalCampaignRecord],
        None,
    ]
    | None = None,
) -> CausalFiveFieldTemporalCampaignResult:
    """Evolve one exact output horizon under the locked budget rule."""

    context = context.validated()
    config = config.validated()
    count = causal_five_field_dae_count(context.grid.centers.size)
    state = _validated_campaign_state(
        initial_state,
        expected_size=count.total_unknowns,
    )
    target = float(target_elapsed_time)
    relative_tolerance = float(target_time_relative_tolerance)
    if (
        not np.isfinite(target)
        or target <= state.elapsed_time
        or not np.isfinite(relative_tolerance)
        or relative_tolerance <= 0.0
        or config.output_horizon_seconds is None
    ):
        raise ValueError("causal horizon-campaign contract is invalid")
    horizon = target - state.elapsed_time
    horizon_tolerance = max(
        1.0e-20,
        relative_tolerance * target,
    )
    if not np.isclose(
        horizon,
        config.output_horizon_seconds,
        rtol=0.0,
        atol=horizon_tolerance,
    ):
        raise ValueError(
            "controller output horizon does not match target duration"
        )
    adapter_requested = state_adapter_after_accepted_steps is not None
    if adapter_requested != (state_adapter is not None):
        raise ValueError(
            "state adapter and accepted-step trigger must be paired"
        )
    if (
        state_adapter_after_accepted_steps is not None
        and (
            int(state_adapter_after_accepted_steps)
            != state_adapter_after_accepted_steps
            or state_adapter_after_accepted_steps < 1
        )
    ):
        raise ValueError("state-adapter trigger must be a positive integer")

    records: list[CausalFiveFieldTemporalCampaignRecord] = []
    adapter_performed = False
    message = "target reached"

    while True:
        remaining = target - state.elapsed_time
        if abs(remaining) <= horizon_tolerance:
            state = replace(state, elapsed_time=target)
            break
        if remaining <= 0.0:
            message = "target overshot"
            break
        requested_timestep = min(state.dt_next, remaining)
        result = advance_causal_five_field_step_doubling_backward_euler(
            context,
            state.state_vector,
            requested_timestep,
            state.previous_physical_increment,
            state.previous_dt,
            _landing_config(config, requested_timestep),
        )
        rejected = sum(
            not attempt.accepted for attempt in result.attempts
        )
        record = CausalFiveFieldTemporalCampaignRecord(
            elapsed_time_before=state.elapsed_time,
            requested_timestep=requested_timestep,
            result=result,
            rejected_trials=rejected,
        )
        records.append(record)
        if progress is not None:
            progress(record)
        if not result.accepted:
            message = result.message
            break

        new_elapsed = state.elapsed_time + result.dt_used
        if result.dt_used == remaining:
            new_elapsed = target
        state = CausalFiveFieldTemporalCampaignState(
            state_vector=np.asarray(result.state_vector, dtype=float),
            previous_physical_increment=np.asarray(
                result.physical_increment,
                dtype=float,
            ),
            elapsed_time=new_elapsed,
            dt_next=float(result.dt_next),
            previous_dt=float(result.dt_used),
            accepted_steps=state.accepted_steps + 1,
            rejected_trials=state.rejected_trials + rejected,
            cumulative_budget_fraction=(
                state.cumulative_budget_fraction
                + result.dt_used / config.output_horizon_seconds
            ),
        )

        if (
            adapter_requested
            and not adapter_performed
            and state.accepted_steps
            == state_adapter_after_accepted_steps
        ):
            assert state_adapter is not None
            adapted = _validated_campaign_state(
                state_adapter(state),
                expected_size=count.total_unknowns,
            )
            if not causal_five_field_temporal_campaign_states_equal(
                state,
                adapted,
            ):
                raise ValueError(
                    "state adapter did not preserve the campaign state"
                )
            state = adapted
            adapter_performed = True

    target_reached = bool(
        abs(state.elapsed_time - target) <= horizon_tolerance
    )
    budget_sum_passed = bool(
        target_reached
        and np.isclose(
            state.cumulative_budget_fraction,
            1.0,
            rtol=0.0,
            atol=2.0e-13,
        )
    )
    state_gates = audit_causal_five_field_state_gates(
        context,
        state.state_vector,
    )
    passed = bool(
        target_reached
        and budget_sum_passed
        and state_gates["passed"]
        and (not adapter_requested or adapter_performed)
    )
    return CausalFiveFieldTemporalCampaignResult(
        final_state=state,
        target_elapsed_time=target,
        records=tuple(records),
        state_gates=state_gates,
        target_reached=target_reached,
        budget_sum_passed=budget_sum_passed,
        state_adapter_requested=adapter_requested,
        state_adapter_performed=adapter_performed,
        passed=passed,
        message=message,
    )
