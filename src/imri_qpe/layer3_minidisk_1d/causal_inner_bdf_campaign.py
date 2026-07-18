"""Repeated adaptive BDF2 evolution for the causal five-field DAE."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .causal_inner_bdf_controller import (
    CausalFiveFieldAdaptiveBDF2Config,
    CausalFiveFieldAdaptiveBDF2StepResult,
    advance_causal_five_field_adaptive_bdf2,
)
from .causal_inner_bdf_evolution import (
    CausalFiveFieldBDFPhysicalIntervalLedger,
)
from .causal_inner_bdf_restart import (
    CausalFiveFieldAdaptiveBDF2Restart,
)
from .causal_inner_dae_system import CausalFiveFieldDAEContext
from .causal_inner_diagnostics import (
    audit_causal_five_field_state_gates,
)


@dataclass(frozen=True)
class CausalFiveFieldAdaptiveBDF2CampaignResult:
    """One bounded adaptive trajectory and all of its trial records."""

    restart: CausalFiveFieldAdaptiveBDF2Restart
    steps: tuple[CausalFiveFieldAdaptiveBDF2StepResult, ...]
    passed: bool
    message: str


def causal_five_field_bdf_zero_physical_ledger(
) -> CausalFiveFieldBDFPhysicalIntervalLedger:
    """Return an independent zero-valued five-field physical ledger."""

    zero = np.zeros(5, dtype=float)
    return CausalFiveFieldBDFPhysicalIntervalLedger(
        actual_conserved_storage=np.array(zero, copy=True),
        actual_vertical_storage=np.array(zero, copy=True),
        trapezoidal_boundary_transport=np.array(zero, copy=True),
        trapezoidal_endogenous_source=np.array(zero, copy=True),
        exact_prescribed_stream_source=np.array(zero, copy=True),
        closure_defect=np.array(zero, copy=True),
    )


def causal_five_field_bdf_physical_ledger_from_restart(
    restart: CausalFiveFieldAdaptiveBDF2Restart,
) -> CausalFiveFieldBDFPhysicalIntervalLedger:
    """Recover the cumulative physical ledger from a complete restart."""

    return CausalFiveFieldBDFPhysicalIntervalLedger(
        actual_conserved_storage=np.asarray(
            restart.cumulative_actual_conserved_storage,
            dtype=float,
        ),
        actual_vertical_storage=np.asarray(
            restart.cumulative_actual_vertical_storage,
            dtype=float,
        ),
        trapezoidal_boundary_transport=np.asarray(
            restart.cumulative_boundary_transport,
            dtype=float,
        ),
        trapezoidal_endogenous_source=np.asarray(
            restart.cumulative_endogenous_source,
            dtype=float,
        ),
        exact_prescribed_stream_source=np.asarray(
            restart.cumulative_stream_source,
            dtype=float,
        ),
        closure_defect=np.asarray(
            restart.cumulative_closure_defect,
            dtype=float,
        ),
    )


def add_causal_five_field_bdf_physical_ledgers(
    left: CausalFiveFieldBDFPhysicalIntervalLedger,
    right: CausalFiveFieldBDFPhysicalIntervalLedger,
) -> CausalFiveFieldBDFPhysicalIntervalLedger:
    """Add two physical horizon ledgers component by component."""

    return CausalFiveFieldBDFPhysicalIntervalLedger(
        actual_conserved_storage=(
            left.actual_conserved_storage
            + right.actual_conserved_storage
        ),
        actual_vertical_storage=(
            left.actual_vertical_storage + right.actual_vertical_storage
        ),
        trapezoidal_boundary_transport=(
            left.trapezoidal_boundary_transport
            + right.trapezoidal_boundary_transport
        ),
        trapezoidal_endogenous_source=(
            left.trapezoidal_endogenous_source
            + right.trapezoidal_endogenous_source
        ),
        exact_prescribed_stream_source=(
            left.exact_prescribed_stream_source
            + right.exact_prescribed_stream_source
        ),
        closure_defect=left.closure_defect + right.closure_defect,
    )


def causal_five_field_bdf_physical_ledger_relative_defects(
    ledger: CausalFiveFieldBDFPhysicalIntervalLedger,
) -> np.ndarray:
    """Return the five componentwise physical-ledger relative defects."""

    scale = (
        np.abs(ledger.actual_conserved_storage)
        + np.abs(ledger.actual_vertical_storage)
        + np.abs(ledger.trapezoidal_boundary_transport)
        + np.abs(ledger.trapezoidal_endogenous_source)
        + np.abs(ledger.exact_prescribed_stream_source)
    )
    return np.abs(ledger.closure_defect) / np.maximum(
        scale,
        np.finfo(float).tiny,
    )


def _restart_after_attempt(
    template: CausalFiveFieldAdaptiveBDF2Restart,
    *,
    state_vector: np.ndarray,
    history,
    older_physical_increment: np.ndarray,
    older_timestep_seconds: float,
    cumulative: CausalFiveFieldBDFPhysicalIntervalLedger,
    elapsed_time: float,
    dt_next: float,
    next_order: int,
    accepted_steps: int,
    accepted_bdf2_steps: int,
    rejected_attempts: int,
    audit_count: int,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    return CausalFiveFieldAdaptiveBDF2Restart(
        state_vector=np.asarray(state_vector, dtype=float),
        history=history,
        older_physical_increment=np.asarray(
            older_physical_increment,
            dtype=float,
        ),
        older_timestep_seconds=float(older_timestep_seconds),
        cumulative_actual_conserved_storage=np.asarray(
            cumulative.actual_conserved_storage,
            dtype=float,
        ),
        cumulative_actual_vertical_storage=np.asarray(
            cumulative.actual_vertical_storage,
            dtype=float,
        ),
        cumulative_boundary_transport=np.asarray(
            cumulative.trapezoidal_boundary_transport,
            dtype=float,
        ),
        cumulative_endogenous_source=np.asarray(
            cumulative.trapezoidal_endogenous_source,
            dtype=float,
        ),
        cumulative_stream_source=np.asarray(
            cumulative.exact_prescribed_stream_source,
            dtype=float,
        ),
        cumulative_closure_defect=np.asarray(
            cumulative.closure_defect,
            dtype=float,
        ),
        elapsed_time=float(elapsed_time),
        dt_next=float(dt_next),
        next_order=int(next_order),
        accepted_steps=int(accepted_steps),
        accepted_bdf2_steps=int(accepted_bdf2_steps),
        rejected_attempts=int(rejected_attempts),
        audit_count=int(audit_count),
        provenance=dict(template.provenance),
    )


def evolve_causal_five_field_adaptive_bdf2_campaign(
    context: CausalFiveFieldDAEContext,
    initial: CausalFiveFieldAdaptiveBDF2Restart,
    target_elapsed_time: float,
    config: CausalFiveFieldAdaptiveBDF2Config,
    *,
    target_time_relative_tolerance: float = 5.0e-14,
    progress: Callable[
        [
            int,
            CausalFiveFieldAdaptiveBDF2Restart,
            CausalFiveFieldAdaptiveBDF2StepResult,
        ],
        None,
    ]
    | None = None,
) -> CausalFiveFieldAdaptiveBDF2CampaignResult:
    """Advance to one exact target with BDF1 recovery when needed."""

    context = context.validated()
    config = config.validated()
    target = float(target_elapsed_time)
    relative_tolerance = float(target_time_relative_tolerance)
    if (
        not np.isfinite(target)
        or target <= initial.elapsed_time
        or not np.isfinite(relative_tolerance)
        or relative_tolerance <= 0.0
    ):
        raise ValueError("adaptive BDF2 campaign target is invalid")
    tolerance = max(1.0e-20, relative_tolerance * target)
    state = initial
    cumulative = causal_five_field_bdf_physical_ledger_from_restart(
        initial
    )
    steps: list[CausalFiveFieldAdaptiveBDF2StepResult] = []
    relative_accepted = 0
    passed = True
    message = "adaptive BDF2 campaign reached target"

    while state.elapsed_time < target - tolerance:
        remaining = target - state.elapsed_time
        requested = min(state.dt_next, remaining)
        step_config = config
        if requested < config.minimum_dt:
            if requested < config.step_config.minimum_dt:
                passed = False
                message = "adaptive BDF2 exact landing is below step minimum"
                break
            step_config = replace(config, minimum_dt=requested).validated()
        result = advance_causal_five_field_adaptive_bdf2(
            context,
            state.state_vector,
            state.history,
            state.older_physical_increment,
            state.older_timestep_seconds,
            requested,
            step_config,
            next_order=state.next_order,
            accepted_bdf2_steps=state.accepted_bdf2_steps,
        )
        steps.append(result)
        rejected = state.rejected_attempts + sum(
            1 for attempt in result.attempts if not attempt.accepted
        )
        audits = state.audit_count + sum(
            1
            for attempt in result.attempts
            if attempt.independent_audit is not None
        )
        if not result.accepted:
            if state.next_order == 2:
                state = _restart_after_attempt(
                    state,
                    state_vector=state.state_vector,
                    history=state.history,
                    older_physical_increment=(
                        state.older_physical_increment
                    ),
                    older_timestep_seconds=(
                        state.older_timestep_seconds
                    ),
                    cumulative=cumulative,
                    elapsed_time=state.elapsed_time,
                    dt_next=result.dt_next,
                    next_order=1,
                    accepted_steps=state.accepted_steps,
                    accepted_bdf2_steps=state.accepted_bdf2_steps,
                    rejected_attempts=rejected,
                    audit_count=audits,
                )
                continue
            passed = False
            message = "adaptive BDF2 campaign exhausted BDF1 fallback"
            break

        cumulative = add_causal_five_field_bdf_physical_ledgers(
            cumulative,
            result.physical_interval_ledger,
        )
        state = _restart_after_attempt(
            state,
            state_vector=result.state_vector,
            history=result.history,
            older_physical_increment=(
                result.older_physical_increment
            ),
            older_timestep_seconds=result.older_timestep_seconds,
            cumulative=cumulative,
            elapsed_time=state.elapsed_time + result.dt_used,
            dt_next=result.dt_next,
            next_order=2,
            accepted_steps=state.accepted_steps + 1,
            accepted_bdf2_steps=result.accepted_bdf2_steps,
            rejected_attempts=rejected,
            audit_count=audits,
        )
        relative_accepted += 1
        if progress is not None:
            progress(relative_accepted, state, result)

    if abs(state.elapsed_time - target) > tolerance:
        passed = False
        if message == "adaptive BDF2 campaign reached target":
            message = "adaptive BDF2 campaign missed target time"
    if not audit_causal_five_field_state_gates(
        context,
        state.state_vector,
    )["passed"]:
        passed = False
        message = "adaptive BDF2 campaign failed final state gates"
    return CausalFiveFieldAdaptiveBDF2CampaignResult(
        restart=state,
        steps=tuple(steps),
        passed=passed,
        message=message,
    )
