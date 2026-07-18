"""Adaptive variable-step BDF2 control for the causal five-field DAE."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from .causal_inner_bdf import (
    CausalFiveFieldBDFHistory,
    causal_bdf_quadratic_history_predictor,
)
from .causal_inner_bdf_evolution import (
    CausalFiveFieldBDFPhysicalIntervalLedger,
    CausalFiveFieldBDFStepResult,
    advance_causal_five_field_increment_bdf,
)
from .causal_inner_dae import causal_five_field_dae_count
from .causal_inner_dae_system import CausalFiveFieldDAEContext
from .causal_inner_diagnostics import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    audit_causal_five_field_state_gates,
    causal_five_field_temporal_error_ratio,
    compare_causal_five_field_endpoint_vectors,
)
from .causal_inner_evolution import CausalFiveFieldAdaptiveStepConfig


@dataclass(frozen=True)
class CausalFiveFieldAdaptiveBDF2Config:
    """Predictor-corrector and periodic-audit settings."""

    step_config: CausalFiveFieldAdaptiveStepConfig
    cooling_inner_cutoff: float
    minimum_dt: float
    maximum_dt: float
    temporal_accuracy_gates: Mapping[str, float] = field(
        default_factory=lambda: dict(
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
        )
    )
    local_error_gate_fraction: float = 0.25
    predictor_error_scale: float = 0.2
    safety_factor: float = 0.8
    minimum_factor: float = 0.5
    maximum_factor: float = 2.0
    maximum_retries: int = 6
    audit_interval: int = 4

    def validated(self) -> CausalFiveFieldAdaptiveBDF2Config:
        step_config = self.step_config.validated()
        positive = (
            self.cooling_inner_cutoff,
            self.minimum_dt,
            self.maximum_dt,
            self.local_error_gate_fraction,
            self.predictor_error_scale,
            self.safety_factor,
            self.minimum_factor,
            self.maximum_factor,
        )
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in positive
        ):
            raise ValueError("adaptive BDF2 values must be positive")
        if self.minimum_dt > self.maximum_dt:
            raise ValueError("adaptive BDF2 timestep bounds are reversed")
        if (
            self.minimum_dt < step_config.minimum_dt
            or self.maximum_dt > step_config.maximum_dt
        ):
            raise ValueError(
                "adaptive BDF2 bounds must lie within the step bounds"
            )
        if self.local_error_gate_fraction > 1.0:
            raise ValueError(
                "adaptive BDF2 local gate fraction must not exceed one"
            )
        if (
            self.minimum_factor >= 1.0
            or self.maximum_factor < 1.0
            or self.maximum_factor > 2.0
        ):
            raise ValueError(
                "adaptive BDF2 factors must conservatively bracket one"
            )
        if (
            int(self.maximum_retries) != self.maximum_retries
            or self.maximum_retries < 0
            or int(self.audit_interval) != self.audit_interval
            or self.audit_interval < 1
        ):
            raise ValueError(
                "adaptive BDF2 retry and audit counts are invalid"
            )
        gates = {
            str(name): float(value)
            for name, value in self.temporal_accuracy_gates.items()
        }
        if set(gates) != set(
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
        ):
            raise ValueError(
                "adaptive BDF2 gates must use the complete v1 schema"
            )
        causal_five_field_temporal_error_ratio(
            {name: 0.0 for name in gates},
            gates,
        )
        return replace(
            self,
            step_config=step_config,
            temporal_accuracy_gates=gates,
        )


@dataclass(frozen=True)
class CausalFiveFieldBDF2Audit:
    """One full-versus-two-half independent BDF2 audit."""

    first_half_step: CausalFiveFieldBDFStepResult
    second_half_step: CausalFiveFieldBDFStepResult | None
    temporal_errors: dict[str, float | list[float]] | None
    temporal_gate_audit: dict[str, object] | None
    passed: bool


@dataclass(frozen=True)
class CausalFiveFieldAdaptiveBDF2Attempt:
    """One adaptive BDF1/BDF2 trial."""

    timestep_seconds: float
    order: int
    step: CausalFiveFieldBDFStepResult
    predictor_errors: dict[str, float | list[float]] | None
    estimated_local_errors: dict[str, float] | None
    local_gate_audit: dict[str, object] | None
    independent_audit: CausalFiveFieldBDF2Audit | None
    accepted: bool
    failure_class: str
    proposed_factor: float
    implicit_solves: int


@dataclass(frozen=True)
class CausalFiveFieldAdaptiveBDF2StepResult:
    """One accepted adaptive state with every rejected trial."""

    state_vector: np.ndarray
    history: CausalFiveFieldBDFHistory
    older_physical_increment: np.ndarray
    older_timestep_seconds: float
    accepted: bool
    order: int
    dt_used: float
    dt_next: float
    accepted_bdf2_steps: int
    attempts: tuple[CausalFiveFieldAdaptiveBDF2Attempt, ...]
    physical_interval_ledger: CausalFiveFieldBDFPhysicalIntervalLedger
    message: str


def _empty_physical_ledger() -> CausalFiveFieldBDFPhysicalIntervalLedger:
    zero = np.zeros(5, dtype=float)
    return CausalFiveFieldBDFPhysicalIntervalLedger(
        actual_conserved_storage=np.array(zero, copy=True),
        actual_vertical_storage=np.array(zero, copy=True),
        trapezoidal_boundary_transport=np.array(zero, copy=True),
        trapezoidal_endogenous_source=np.array(zero, copy=True),
        exact_prescribed_stream_source=np.array(zero, copy=True),
        closure_defect=np.array(zero, copy=True),
    )


def _factor(
    normalized_error: float,
    config: CausalFiveFieldAdaptiveBDF2Config,
) -> float:
    error = float(normalized_error)
    if not np.isfinite(error) or error < 0.0:
        raise ValueError("adaptive BDF2 error must be non-negative")
    if error == 0.0:
        return float(config.maximum_factor)
    return float(
        np.clip(
            config.safety_factor * error ** (-1.0 / 3.0),
            config.minimum_factor,
            config.maximum_factor,
        )
    )


def _quadratic_predictor(
    state_vector: np.ndarray,
    history: CausalFiveFieldBDFHistory,
    older_physical_increment: np.ndarray,
    older_timestep_seconds: float,
    timestep_seconds: float,
) -> np.ndarray:
    return causal_bdf_quadratic_history_predictor(
        state_vector,
        history.previous_physical_increment,
        history.previous_timestep_seconds,
        older_physical_increment,
        older_timestep_seconds,
        timestep_seconds,
    )


def _estimated_predictor_errors(
    context: CausalFiveFieldDAEContext,
    baseline: np.ndarray,
    corrected: np.ndarray,
    predicted: np.ndarray,
    config: CausalFiveFieldAdaptiveBDF2Config,
) -> tuple[
    dict[str, float | list[float]],
    dict[str, float],
    dict[str, object],
]:
    raw = compare_causal_five_field_endpoint_vectors(
        context,
        baseline,
        corrected,
        predicted,
        cooling_inner_cutoff=config.cooling_inner_cutoff,
    )
    estimated = {
        name: config.predictor_error_scale * float(raw[name])
        for name in config.temporal_accuracy_gates
    }
    local_gates = {
        name: config.local_error_gate_fraction * float(limit)
        for name, limit in config.temporal_accuracy_gates.items()
    }
    audit = causal_five_field_temporal_error_ratio(
        estimated,
        local_gates,
    )
    return raw, estimated, audit


def _independent_two_half_audit(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    full_step: CausalFiveFieldBDFStepResult,
    timestep_seconds: float,
    history: CausalFiveFieldBDFHistory,
    older_physical_increment: np.ndarray,
    older_timestep_seconds: float,
    config: CausalFiveFieldAdaptiveBDF2Config,
) -> CausalFiveFieldBDF2Audit:
    half_timestep = 0.5 * timestep_seconds
    first_predictor_state = _quadratic_predictor(
        old_vector,
        history,
        older_physical_increment,
        older_timestep_seconds,
        half_timestep,
    )
    first = advance_causal_five_field_increment_bdf(
        context,
        old_vector,
        half_timestep,
        first_predictor_state - old_vector,
        config.step_config,
        order=2,
        history=history,
    )
    second = None
    if first.accepted and first.history is not None:
        second_predictor_state = _quadratic_predictor(
            first.state_vector,
            first.history,
            history.previous_physical_increment,
            history.previous_timestep_seconds,
            half_timestep,
        )
        second = advance_causal_five_field_increment_bdf(
            context,
            first.state_vector,
            half_timestep,
            second_predictor_state - first.state_vector,
            config.step_config,
            order=2,
            history=first.history,
        )
    errors = None
    gate_audit = None
    state_passed = False
    if second is not None and second.accepted:
        errors = compare_causal_five_field_endpoint_vectors(
            context,
            old_vector,
            full_step.state_vector,
            second.state_vector,
            cooling_inner_cutoff=config.cooling_inner_cutoff,
        )
        gate_audit = causal_five_field_temporal_error_ratio(
            {
                name: float(errors[name])
                for name in config.temporal_accuracy_gates
            },
            dict(config.temporal_accuracy_gates),
        )
        state_passed = audit_causal_five_field_state_gates(
            context,
            second.state_vector,
        )["passed"]
    return CausalFiveFieldBDF2Audit(
        first_half_step=first,
        second_half_step=second,
        temporal_errors=errors,
        temporal_gate_audit=gate_audit,
        passed=bool(
            first.accepted
            and second is not None
            and second.accepted
            and state_passed
            and gate_audit is not None
            and gate_audit["passed"]
        ),
    )


def advance_causal_five_field_adaptive_bdf2(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    history: CausalFiveFieldBDFHistory,
    older_physical_increment: np.ndarray,
    older_timestep_seconds: float,
    requested_timestep_seconds: float,
    config: CausalFiveFieldAdaptiveBDF2Config,
    *,
    next_order: int,
    accepted_bdf2_steps: int,
) -> CausalFiveFieldAdaptiveBDF2StepResult:
    """Advance one adaptive BDF1 recovery or BDF2 step."""

    context = context.validated()
    config = config.validated()
    n_cells = int(context.grid.centers.size)
    count = causal_five_field_dae_count(n_cells)
    old_values = np.asarray(old_vector, dtype=float)
    validated_history = history.validated(
        total_unknowns=count.total_unknowns,
        n_cells=n_cells,
    )
    older_increment = np.asarray(
        older_physical_increment,
        dtype=float,
    )
    older_dt = float(older_timestep_seconds)
    requested_dt = float(requested_timestep_seconds)
    if (
        old_values.shape != (count.total_unknowns,)
        or older_increment.shape != old_values.shape
        or np.any(~np.isfinite(old_values))
        or np.any(~np.isfinite(older_increment))
        or not np.isfinite(older_dt)
        or older_dt <= 0.0
        or not np.isfinite(requested_dt)
        or requested_dt <= 0.0
        or next_order not in (1, 2)
        or accepted_bdf2_steps < 0
    ):
        raise ValueError("adaptive BDF2 state is invalid")
    maximum_ratio_dt = (
        config.maximum_factor
        * validated_history.previous_timestep_seconds
    )
    trial_dt = float(
        np.clip(
            min(requested_dt, maximum_ratio_dt),
            config.minimum_dt,
            config.maximum_dt,
        )
    )
    attempts: list[CausalFiveFieldAdaptiveBDF2Attempt] = []

    for _retry in range(config.maximum_retries + 1):
        order = int(next_order)
        if order == 1:
            predictor_increment = (
                validated_history.previous_physical_increment
                * (
                    trial_dt
                    / validated_history.previous_timestep_seconds
                )
            )
        else:
            predictor_state = _quadratic_predictor(
                old_values,
                validated_history,
                older_increment,
                older_dt,
                trial_dt,
            )
            predictor_increment = predictor_state - old_values
        step = advance_causal_five_field_increment_bdf(
            context,
            old_values,
            trial_dt,
            predictor_increment,
            config.step_config,
            order=order,
            history=validated_history if order == 2 else None,
        )
        predictor_errors = None
        estimated_errors = None
        local_audit = None
        independent_audit = None
        state_passed = bool(
            step.accepted
            and audit_causal_five_field_state_gates(
                context,
                step.state_vector,
            )["passed"]
        )
        if order == 2 and state_passed:
            predictor_state = old_values + predictor_increment
            try:
                (
                    predictor_errors,
                    estimated_errors,
                    local_audit,
                ) = _estimated_predictor_errors(
                    context,
                    old_values,
                    step.state_vector,
                    predictor_state,
                    config,
                )
            except ValueError:
                local_audit = None
            audit_due = bool(
                accepted_bdf2_steps == 0
                or (accepted_bdf2_steps + 1) % config.audit_interval == 0
            )
            if audit_due:
                independent_audit = _independent_two_half_audit(
                    context,
                    old_values,
                    step,
                    trial_dt,
                    validated_history,
                    older_increment,
                    older_dt,
                    config,
                )

        if not step.accepted:
            failure_class = "nonlinear_or_step_contract"
            normalized_error = np.inf
        elif not state_passed:
            failure_class = "physical_state_gate"
            normalized_error = np.inf
        elif order == 2 and local_audit is None:
            failure_class = "predictor_error_unavailable"
            normalized_error = np.inf
        elif order == 2 and not local_audit["passed"]:
            failure_class = "predictor_temporal_accuracy"
            normalized_error = float(
                local_audit["maximum_normalized_error"]
            )
        elif (
            independent_audit is not None
            and not independent_audit.passed
        ):
            failure_class = "independent_temporal_audit"
            normalized_error = (
                float(
                    independent_audit.temporal_gate_audit[
                        "maximum_normalized_error"
                    ]
                )
                if independent_audit.temporal_gate_audit is not None
                else np.inf
            )
        else:
            failure_class = "none"
            normalized_error = (
                float(local_audit["maximum_normalized_error"])
                if local_audit is not None
                else 0.25
            )
        factor = (
            _factor(normalized_error, config)
            if np.isfinite(normalized_error)
            else config.minimum_factor
        )
        attempt = CausalFiveFieldAdaptiveBDF2Attempt(
            timestep_seconds=trial_dt,
            order=order,
            step=step,
            predictor_errors=predictor_errors,
            estimated_local_errors=estimated_errors,
            local_gate_audit=local_audit,
            independent_audit=independent_audit,
            accepted=failure_class == "none",
            failure_class=failure_class,
            proposed_factor=float(factor),
            implicit_solves=(
                1
                + (
                    2
                    if independent_audit is not None
                    and independent_audit.second_half_step is not None
                    else (
                        1 if independent_audit is not None else 0
                    )
                )
            ),
        )
        attempts.append(attempt)
        if failure_class == "none":
            assert step.history is not None
            next_dt = float(
                np.clip(
                    trial_dt * factor,
                    config.minimum_dt,
                    min(
                        config.maximum_dt,
                        config.maximum_factor * trial_dt,
                    ),
                )
            )
            return CausalFiveFieldAdaptiveBDF2StepResult(
                state_vector=step.state_vector,
                history=step.history,
                older_physical_increment=np.asarray(
                    validated_history.previous_physical_increment,
                    dtype=float,
                ),
                older_timestep_seconds=(
                    validated_history.previous_timestep_seconds
                ),
                accepted=True,
                order=order,
                dt_used=trial_dt,
                dt_next=next_dt,
                accepted_bdf2_steps=(
                    accepted_bdf2_steps + (1 if order == 2 else 0)
                ),
                attempts=tuple(attempts),
                physical_interval_ledger=step.physical_interval_ledger,
                message="accepted adaptive BDF step",
            )
        next_dt = trial_dt * config.minimum_factor
        if next_dt < config.minimum_dt:
            break
        trial_dt = next_dt

    return CausalFiveFieldAdaptiveBDF2StepResult(
        state_vector=old_values,
        history=validated_history,
        older_physical_increment=older_increment,
        older_timestep_seconds=older_dt,
        accepted=False,
        order=int(next_order),
        dt_used=0.0,
        dt_next=float(max(config.minimum_dt, trial_dt)),
        accepted_bdf2_steps=accepted_bdf2_steps,
        attempts=tuple(attempts),
        physical_interval_ledger=_empty_physical_ledger(),
        message="adaptive BDF2 retries exhausted",
    )
