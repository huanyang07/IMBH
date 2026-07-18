"""Observable-controlled step doubling for the causal five-field DAE."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from .causal_inner_dae import causal_five_field_dae_count
from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    causal_five_field_dae_scaling,
    evaluate_causal_five_field_dae,
    unpack_causal_five_field_state,
)
from .causal_inner_diagnostics import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    audit_causal_five_field_state_gates,
    causal_backward_euler_horizon_budget_factor,
    causal_backward_euler_step_doubling_factor,
    causal_five_field_observable_snapshot,
    causal_five_field_temporal_error_ratio,
    compare_causal_five_field_observables,
)
from .causal_inner_evolution import (
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldPhysicalStepLedger,
    CausalFiveFieldStepResult,
    advance_causal_five_field_increment_backward_euler,
    causal_five_field_physical_step_ledger,
)


@dataclass(frozen=True)
class CausalFiveFieldTemporalControllerConfig:
    """Validated local-error and physical-contract controller settings."""

    step_config: CausalFiveFieldAdaptiveStepConfig
    cooling_inner_cutoff: float
    minimum_dt: float
    maximum_dt: float
    temporal_accuracy_gates: Mapping[str, float] = field(
        default_factory=lambda: dict(
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
        )
    )
    physical_ledger_tolerance: float = 1.0e-10
    safety_factor: float = 0.8
    minimum_factor: float = 0.25
    maximum_factor: float = 2.0
    maximum_retries: int = 6
    output_horizon_seconds: float | None = None
    horizon_budget_fraction: float = 1.0

    def validated(self) -> CausalFiveFieldTemporalControllerConfig:
        step_config = self.step_config.validated()
        positive = (
            self.cooling_inner_cutoff,
            self.minimum_dt,
            self.maximum_dt,
            self.physical_ledger_tolerance,
            self.safety_factor,
            self.minimum_factor,
            self.maximum_factor,
            self.horizon_budget_fraction,
        )
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in positive
        ):
            raise ValueError(
                "causal temporal-controller values must be positive"
            )
        if self.minimum_dt > self.maximum_dt:
            raise ValueError(
                "temporal minimum_dt must not exceed maximum_dt"
            )
        if (
            self.minimum_dt < step_config.minimum_dt
            or self.maximum_dt > step_config.maximum_dt
        ):
            raise ValueError(
                "temporal timestep bounds must lie inside step bounds"
            )
        if (
            self.minimum_factor >= 1.0
            or self.maximum_factor < 1.0
            or self.minimum_factor > self.maximum_factor
        ):
            raise ValueError(
                "timestep factors must bracket one with a shrinking minimum"
            )
        if (
            int(self.maximum_retries) != self.maximum_retries
            or self.maximum_retries < 0
        ):
            raise ValueError(
                "maximum_retries must be a non-negative integer"
            )
        if self.horizon_budget_fraction > 1.0:
            raise ValueError(
                "horizon_budget_fraction must not exceed one"
            )
        if self.output_horizon_seconds is not None:
            horizon = float(self.output_horizon_seconds)
            if not np.isfinite(horizon) or horizon <= 0.0:
                raise ValueError(
                    "output_horizon_seconds must be positive"
                )
            if self.maximum_dt > horizon:
                raise ValueError(
                    "temporal maximum_dt must not exceed output horizon"
                )
        gates = {
            str(name): float(value)
            for name, value in self.temporal_accuracy_gates.items()
        }
        if set(gates) != set(
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
        ):
            raise ValueError(
                "temporal gates must use the complete v1 schema"
            )
        causal_five_field_temporal_error_ratio(
            {name: 0.0 for name in gates},
            gates,
        )
        causal_backward_euler_step_doubling_factor(
            1.0,
            safety_factor=self.safety_factor,
            minimum_factor=self.minimum_factor,
            maximum_factor=self.maximum_factor,
        )
        if self.output_horizon_seconds is not None:
            causal_backward_euler_horizon_budget_factor(
                1.0,
                safety_factor=self.safety_factor,
                minimum_factor=self.minimum_factor,
                maximum_factor=self.maximum_factor,
            )
        return replace(
            self,
            step_config=step_config,
            temporal_accuracy_gates=gates,
        )


@dataclass(frozen=True)
class CausalFiveFieldTemporalStepContract:
    """Independent physical acceptance audit for one implicit step."""

    passed: bool
    state_gates: dict[str, object] | None
    physical_ledger: CausalFiveFieldPhysicalStepLedger | None
    component_relative_ledger_defects: tuple[float, ...] | None
    maximum_relative_ledger_defect: float | None


@dataclass(frozen=True)
class CausalFiveFieldStepDoublingAttempt:
    """One full/two-half trial, including all rejected diagnostics."""

    timestep_seconds: float
    full_step: CausalFiveFieldStepResult
    first_half_step: CausalFiveFieldStepResult
    second_half_step: CausalFiveFieldStepResult | None
    full_contract: CausalFiveFieldTemporalStepContract
    first_half_contract: CausalFiveFieldTemporalStepContract
    second_half_contract: CausalFiveFieldTemporalStepContract | None
    temporal_errors: dict[str, float | list[float]] | None
    temporal_gate_audit: dict[str, object] | None
    temporal_budget_fraction: float
    effective_temporal_accuracy_gates: dict[str, float]
    accepted: bool
    failure_class: str
    proposed_factor: float


@dataclass(frozen=True)
class CausalFiveFieldStepDoublingResult:
    """One accepted temporal state, including every rejected trial."""

    state_vector: np.ndarray
    physical_increment: np.ndarray
    accepted: bool
    dt_used: float
    dt_next: float
    normalized_error: float | None
    attempts: tuple[CausalFiveFieldStepDoublingAttempt, ...]
    message: str


def audit_causal_five_field_temporal_step_contract(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    step: CausalFiveFieldStepResult,
    *,
    physical_ledger_tolerance: float,
) -> CausalFiveFieldTemporalStepContract:
    if not step.accepted:
        return CausalFiveFieldTemporalStepContract(
            passed=False,
            state_gates=None,
            physical_ledger=None,
            component_relative_ledger_defects=None,
            maximum_relative_ledger_defect=None,
        )
    ledger = causal_five_field_physical_step_ledger(
        context,
        old_vector,
        step.physical_increment,
        step.timestep_seconds,
    )
    scale = (
        np.abs(ledger.conserved_storage_change)
        + np.abs(ledger.vertical_storage_change)
        + np.abs(ledger.boundary_transport)
        + np.abs(ledger.endogenous_source)
        + np.abs(ledger.prescribed_stream_source)
    )
    relative = np.abs(ledger.closure_defect) / np.maximum(
        scale,
        np.finfo(float).tiny,
    )
    maximum = float(np.max(relative))
    state_gates = audit_causal_five_field_state_gates(
        context,
        step.state_vector,
    )
    return CausalFiveFieldTemporalStepContract(
        passed=bool(
            state_gates["passed"]
            and maximum <= physical_ledger_tolerance
        ),
        state_gates=state_gates,
        physical_ledger=ledger,
        component_relative_ledger_defects=tuple(
            float(value) for value in relative
        ),
        maximum_relative_ledger_defect=maximum,
    )


def _failure_class(
    contracts: tuple[
        CausalFiveFieldTemporalStepContract,
        CausalFiveFieldTemporalStepContract,
        CausalFiveFieldTemporalStepContract | None,
    ],
    *,
    temporal_gate_audit: dict[str, object] | None,
) -> str:
    available = tuple(
        contract for contract in contracts if contract is not None
    )
    if len(available) != 3:
        return "nonlinear_or_step_acceptance"
    if any(contract.state_gates is None for contract in available):
        return "nonlinear_or_step_acceptance"
    if any(
        not bool(contract.state_gates["passed"])
        for contract in available
    ):
        return "physical_state_gate"
    if any(
        contract.maximum_relative_ledger_defect is None
        or not contract.passed
        for contract in available
    ):
        return "physical_conservation_gate"
    if temporal_gate_audit is None:
        return "temporal_error_unavailable"
    if not bool(temporal_gate_audit["passed"]):
        return "temporal_accuracy_gate"
    return "none"


def advance_causal_five_field_step_doubling_backward_euler(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    dt: float,
    previous_physical_increment: np.ndarray,
    previous_dt: float,
    config: CausalFiveFieldTemporalControllerConfig,
) -> CausalFiveFieldStepDoublingResult:
    """Advance one observable-controlled full/two-half BE state."""

    context = context.validated()
    config = config.validated()
    n_cells = int(context.grid.centers.size)
    count = causal_five_field_dae_count(n_cells)
    old_values = np.asarray(old_vector, dtype=float)
    previous_increment = np.asarray(
        previous_physical_increment,
        dtype=float,
    )
    if (
        old_values.shape != (count.total_unknowns,)
        or previous_increment.shape != old_values.shape
        or np.any(~np.isfinite(old_values))
        or np.any(~np.isfinite(previous_increment))
    ):
        raise ValueError("causal step-doubling vectors are invalid")
    previous_timestep = float(previous_dt)
    requested_timestep = float(dt)
    if (
        not np.isfinite(previous_timestep)
        or previous_timestep <= 0.0
        or not np.isfinite(requested_timestep)
        or requested_timestep <= 0.0
    ):
        raise ValueError(
            "causal step-doubling timesteps must be positive"
        )
    baseline_state = unpack_causal_five_field_state(
        old_values,
        n_cells,
    )
    baseline_evaluation = evaluate_causal_five_field_dae(
        old_values,
        context,
    )
    baseline_scales = causal_five_field_dae_scaling(
        baseline_state,
        baseline_evaluation,
    ).column_scales
    trial_dt = float(
        np.clip(requested_timestep, config.minimum_dt, config.maximum_dt)
    )
    attempts: list[CausalFiveFieldStepDoublingAttempt] = []
    last_next_dt = trial_dt

    for _retry in range(config.maximum_retries + 1):
        half_dt = 0.5 * trial_dt
        full_predictor = previous_increment * (
            trial_dt / previous_timestep
        )
        half_predictor = previous_increment * (
            half_dt / previous_timestep
        )
        full_step = advance_causal_five_field_increment_backward_euler(
            context,
            old_values,
            trial_dt,
            full_predictor,
            config.step_config,
        )
        first_half = advance_causal_five_field_increment_backward_euler(
            context,
            old_values,
            half_dt,
            half_predictor,
            config.step_config,
        )
        second_half = None
        if first_half.accepted:
            second_half = (
                advance_causal_five_field_increment_backward_euler(
                    context,
                    first_half.state_vector,
                    half_dt,
                    first_half.physical_increment,
                    config.step_config,
                )
            )
        full_contract = audit_causal_five_field_temporal_step_contract(
            context,
            old_values,
            full_step,
            physical_ledger_tolerance=config.physical_ledger_tolerance,
        )
        first_contract = audit_causal_five_field_temporal_step_contract(
            context,
            old_values,
            first_half,
            physical_ledger_tolerance=config.physical_ledger_tolerance,
        )
        second_contract = (
            audit_causal_five_field_temporal_step_contract(
                context,
                first_half.state_vector,
                second_half,
                physical_ledger_tolerance=(
                    config.physical_ledger_tolerance
                ),
            )
            if second_half is not None
            else None
        )
        contracts_passed = bool(
            full_contract.passed
            and first_contract.passed
            and second_contract is not None
            and second_contract.passed
        )
        temporal_errors = None
        temporal_gate_audit = None
        temporal_budget_fraction = 1.0
        if config.output_horizon_seconds is not None:
            temporal_budget_fraction = (
                config.horizon_budget_fraction
                * trial_dt
                / config.output_horizon_seconds
            )
        effective_temporal_accuracy_gates = {
            name: float(limit) * temporal_budget_fraction
            for name, limit in config.temporal_accuracy_gates.items()
        }
        if contracts_passed:
            assert second_half is not None
            full_observables = causal_five_field_observable_snapshot(
                context,
                full_step.state_vector,
                cooling_inner_cutoff=config.cooling_inner_cutoff,
            )
            half_observables = causal_five_field_observable_snapshot(
                context,
                second_half.state_vector,
                cooling_inner_cutoff=config.cooling_inner_cutoff,
            )
            temporal_errors = compare_causal_five_field_observables(
                full_observables,
                half_observables,
            )
            temporal_errors[
                "maximum_baseline_scaled_state_difference"
            ] = float(
                np.max(
                    np.abs(
                        (
                            full_step.state_vector
                            - second_half.state_vector
                        )
                        / baseline_scales
                    )
                )
            )
            temporal_gate_audit = (
                causal_five_field_temporal_error_ratio(
                    temporal_errors,
                    effective_temporal_accuracy_gates,
                )
            )
        if temporal_gate_audit is None:
            factor = config.minimum_factor
        else:
            normalized_error = float(
                temporal_gate_audit["maximum_normalized_error"]
            )
            if config.output_horizon_seconds is None:
                factor = causal_backward_euler_step_doubling_factor(
                    normalized_error,
                    safety_factor=config.safety_factor,
                    minimum_factor=config.minimum_factor,
                    maximum_factor=config.maximum_factor,
                )
            else:
                factor = causal_backward_euler_horizon_budget_factor(
                    normalized_error,
                    safety_factor=config.safety_factor,
                    minimum_factor=config.minimum_factor,
                    maximum_factor=config.maximum_factor,
                )
        failure_class = _failure_class(
            (full_contract, first_contract, second_contract),
            temporal_gate_audit=temporal_gate_audit,
        )
        accepted = failure_class == "none"
        attempt = CausalFiveFieldStepDoublingAttempt(
            timestep_seconds=trial_dt,
            full_step=full_step,
            first_half_step=first_half,
            second_half_step=second_half,
            full_contract=full_contract,
            first_half_contract=first_contract,
            second_half_contract=second_contract,
            temporal_errors=temporal_errors,
            temporal_gate_audit=temporal_gate_audit,
            temporal_budget_fraction=float(temporal_budget_fraction),
            effective_temporal_accuracy_gates=(
                effective_temporal_accuracy_gates
            ),
            accepted=accepted,
            failure_class=failure_class,
            proposed_factor=float(factor),
        )
        attempts.append(attempt)
        next_dt = float(
            np.clip(
                trial_dt * factor,
                config.minimum_dt,
                config.maximum_dt,
            )
        )
        last_next_dt = next_dt
        if accepted:
            assert second_half is not None
            accepted_increment = (
                second_half.state_vector - old_values
            )
            return CausalFiveFieldStepDoublingResult(
                state_vector=second_half.state_vector,
                physical_increment=accepted_increment,
                accepted=True,
                dt_used=trial_dt,
                dt_next=next_dt,
                normalized_error=float(
                    temporal_gate_audit["maximum_normalized_error"]
                ),
                attempts=tuple(attempts),
                message="accepted two-half-step state",
            )
        if next_dt >= trial_dt or np.isclose(
            next_dt,
            trial_dt,
            rtol=0.0,
            atol=0.0,
        ):
            break
        trial_dt = next_dt

    return CausalFiveFieldStepDoublingResult(
        state_vector=old_values,
        physical_increment=np.zeros_like(old_values),
        accepted=False,
        dt_used=0.0,
        dt_next=last_next_dt,
        normalized_error=None,
        attempts=tuple(attempts),
        message=(
            "step-doubling retries exhausted without an accepted state"
        ),
    )
