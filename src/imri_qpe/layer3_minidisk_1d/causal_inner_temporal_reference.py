"""Fixed-step temporal references for the causal five-field DAE."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .causal_inner_dae import causal_five_field_dae_count
from .causal_inner_dae_system import CausalFiveFieldDAEContext
from .causal_inner_evolution import (
    CausalFiveFieldAdaptiveStepConfig,
    advance_causal_five_field_increment_backward_euler,
)
from .causal_inner_temporal_controller import (
    audit_causal_five_field_temporal_step_contract,
)
from .causal_inner_diagnostics import (
    audit_causal_five_field_state_gates,
)


@dataclass(frozen=True)
class CausalFiveFieldFixedReferenceResult:
    """One complete fixed-step backward-Euler reference trajectory."""

    state_vector: np.ndarray
    previous_physical_increment: np.ndarray
    subdivisions: int
    timestep_seconds: float
    completed_steps: int
    state_gates: dict[str, object]
    maximum_scaled_residual: float
    maximum_scaled_algebraic_residual: float
    maximum_scaled_primitive_change: float
    maximum_scaled_total_change: float
    maximum_conservation_telescoping_relative_defect: float
    maximum_physical_ledger_relative_defect: float
    maximum_linear_residual: float
    maximum_newton_iterations: int
    function_evaluations: int
    jacobian_evaluations: int
    newton_iterations: int
    passed: bool
    message: str


def evolve_causal_five_field_fixed_reference(
    context: CausalFiveFieldDAEContext,
    initial_vector: np.ndarray,
    previous_physical_increment: np.ndarray,
    previous_dt: float,
    duration_seconds: float,
    subdivisions: int,
    step_config: CausalFiveFieldAdaptiveStepConfig,
    *,
    physical_ledger_tolerance: float,
    progress: Callable[[int, int], None] | None = None,
) -> CausalFiveFieldFixedReferenceResult:
    """Evolve one exact-duration fixed backward-Euler trajectory."""

    context = context.validated()
    step_config = step_config.validated()
    count = causal_five_field_dae_count(context.grid.centers.size)
    state = np.asarray(initial_vector, dtype=float)
    previous_increment = np.asarray(
        previous_physical_increment,
        dtype=float,
    )
    if (
        state.shape != (count.total_unknowns,)
        or previous_increment.shape != state.shape
        or np.any(~np.isfinite(state))
        or np.any(~np.isfinite(previous_increment))
    ):
        raise ValueError("fixed-reference vectors are invalid")
    prior_dt = float(previous_dt)
    duration = float(duration_seconds)
    ledger_tolerance = float(physical_ledger_tolerance)
    if (
        not np.isfinite(prior_dt)
        or prior_dt <= 0.0
        or not np.isfinite(duration)
        or duration <= 0.0
        or not np.isfinite(ledger_tolerance)
        or ledger_tolerance <= 0.0
    ):
        raise ValueError("fixed-reference times and gates must be positive")
    if int(subdivisions) != subdivisions or subdivisions < 1:
        raise ValueError("fixed-reference subdivisions must be positive")
    n_steps = int(subdivisions)
    timestep = duration / n_steps
    if (
        timestep < step_config.minimum_dt
        or timestep > step_config.maximum_dt
    ):
        raise ValueError("fixed-reference timestep lies outside step bounds")

    maximum_scaled_residual = 0.0
    maximum_scaled_algebraic_residual = 0.0
    maximum_scaled_primitive_change = 0.0
    maximum_scaled_total_change = 0.0
    maximum_conservation_defect = 0.0
    maximum_physical_ledger_defect = 0.0
    maximum_linear_residual = 0.0
    maximum_newton_iterations = 0
    function_evaluations = 0
    jacobian_evaluations = 0
    newton_iterations = 0
    completed = 0
    passed = True
    message = "fixed reference completed"

    for index in range(n_steps):
        predictor = previous_increment * (timestep / prior_dt)
        step = advance_causal_five_field_increment_backward_euler(
            context,
            state,
            timestep,
            predictor,
            step_config,
        )
        contract = audit_causal_five_field_temporal_step_contract(
            context,
            state,
            step,
            physical_ledger_tolerance=ledger_tolerance,
        )
        maximum_scaled_residual = max(
            maximum_scaled_residual,
            float(step.maximum_scaled_residual),
        )
        maximum_scaled_algebraic_residual = max(
            maximum_scaled_algebraic_residual,
            float(step.maximum_scaled_algebraic_residual),
        )
        maximum_scaled_primitive_change = max(
            maximum_scaled_primitive_change,
            float(step.maximum_scaled_primitive_change),
        )
        maximum_scaled_total_change = max(
            maximum_scaled_total_change,
            float(step.maximum_scaled_total_change),
        )
        maximum_conservation_defect = max(
            maximum_conservation_defect,
            float(step.conservation_telescoping_relative_defect),
        )
        if contract.maximum_relative_ledger_defect is not None:
            maximum_physical_ledger_defect = max(
                maximum_physical_ledger_defect,
                float(contract.maximum_relative_ledger_defect),
            )
        maximum_linear_residual = max(
            maximum_linear_residual,
            float(step.maximum_linear_residual),
        )
        maximum_newton_iterations = max(
            maximum_newton_iterations,
            int(step.iterations),
        )
        function_evaluations += int(step.function_evaluations)
        jacobian_evaluations += int(step.jacobian_evaluations)
        newton_iterations += int(step.iterations)
        if not contract.passed:
            passed = False
            message = "fixed reference failed a step contract"
            break
        state = np.asarray(step.state_vector, dtype=float)
        previous_increment = np.asarray(
            step.physical_increment,
            dtype=float,
        )
        prior_dt = timestep
        completed = index + 1
        if progress is not None:
            progress(completed, n_steps)

    state_gates = audit_causal_five_field_state_gates(context, state)
    passed = bool(
        passed and completed == n_steps and state_gates["passed"]
    )
    if not state_gates["passed"] and message == "fixed reference completed":
        message = "fixed reference failed final state gates"
    return CausalFiveFieldFixedReferenceResult(
        state_vector=state,
        previous_physical_increment=previous_increment,
        subdivisions=n_steps,
        timestep_seconds=timestep,
        completed_steps=completed,
        state_gates=state_gates,
        maximum_scaled_residual=maximum_scaled_residual,
        maximum_scaled_algebraic_residual=(
            maximum_scaled_algebraic_residual
        ),
        maximum_scaled_primitive_change=(
            maximum_scaled_primitive_change
        ),
        maximum_scaled_total_change=maximum_scaled_total_change,
        maximum_conservation_telescoping_relative_defect=(
            maximum_conservation_defect
        ),
        maximum_physical_ledger_relative_defect=(
            maximum_physical_ledger_defect
        ),
        maximum_linear_residual=maximum_linear_residual,
        maximum_newton_iterations=maximum_newton_iterations,
        function_evaluations=function_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        newton_iterations=newton_iterations,
        passed=passed,
        message=message,
    )
