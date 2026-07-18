"""Certify an N16 horizon-budget controller against converged references."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION,
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    CausalFiveFieldAdaptiveRestart,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldDAEContext,
    CausalFiveFieldStepDoublingAttempt,
    CausalFiveFieldTemporalControllerConfig,
    advance_causal_five_field_increment_backward_euler,
    advance_causal_five_field_step_doubling_backward_euler,
    audit_causal_five_field_endpoint_with_reference_uncertainty,
    audit_causal_five_field_reference_convergence,
    audit_causal_five_field_state_gates,
    audit_causal_five_field_temporal_step_contract,
    compare_causal_five_field_endpoint_vectors,
    load_causal_five_field_adaptive_restart,
    make_causal_five_field_regression_context,
    save_causal_five_field_adaptive_restart,
)


ROOT = Path(__file__).resolve().parents[1]
N_CELLS = 16
INITIAL_CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c5k"
    / "causal_wp10c5q_N016_final.npz"
)
LOCAL_CONTROLLER_CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c6c"
    / "causal_wp10c6c_N016_final.npz"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c6d"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables"
    / "causal_horizon_budget_wp10c6d_N016.json"
)
SHARED_PASSING_CEILING_SECONDS = 1.9218219974586337e-3
FIRST_FAILING_TIMESTEP_SECONDS = 3.8436439949172674e-3
INITIAL_TIMESTEP_SECONDS = 0.5 * SHARED_PASSING_CEILING_SECONDS
TARGET_DURATION_SECONDS = 8.0 * SHARED_PASSING_CEILING_SECONDS
REFERENCE_SUBDIVISIONS = (32, 64, 128)
MAXIMUM_REFERENCE_UNCERTAINTY_FRACTION = 0.25
MINIMUM_OBSERVED_ORDER = 0.75
ORDER_FLOOR_FRACTION = 1.0e-3
COOLING_INNER_CUTOFF_RG = 6.0
FINITE_DIFFERENCE_STEP = 2.0e-6
RESTART_SPLIT_ACCEPTED_STEPS = 3
TARGET_TIME_RELATIVE_TOLERANCE = 5.0e-14


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Stop after the mandatory 32/64/128 reference gate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restart_is_bitwise(
    left: CausalFiveFieldAdaptiveRestart,
    right: CausalFiveFieldAdaptiveRestart,
) -> bool:
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
        and left.rejected_attempts == right.rejected_attempts
        and left.provenance == right.provenance
        and left.schema_version == right.schema_version
    )


def _step_config() -> CausalFiveFieldAdaptiveStepConfig:
    minimum_dt = INITIAL_TIMESTEP_SECONDS / 4096.0
    return CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=0.5 * minimum_dt,
        maximum_dt=FIRST_FAILING_TIMESTEP_SECONDS,
        maximum_scaled_primitive_change=0.2,
        maximum_scaled_total_change=0.25,
        shrink_factor=0.5,
        growth_factor=1.5,
        maximum_retries=0,
        easy_iterations=3,
        residual_tolerance=1.0e-11,
        algebraic_residual_tolerance=1.0e-11,
        conservation_tolerance=1.0e-10,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        maximum_newton_iterations=12,
    ).validated()


def _controller_config(
    context: CausalFiveFieldDAEContext,
) -> CausalFiveFieldTemporalControllerConfig:
    minimum_dt = INITIAL_TIMESTEP_SECONDS / 4096.0
    return CausalFiveFieldTemporalControllerConfig(
        step_config=_step_config(),
        cooling_inner_cutoff=(
            COOLING_INNER_CUTOFF_RG
            * context.grid.gravitational_radius
        ),
        minimum_dt=minimum_dt,
        maximum_dt=FIRST_FAILING_TIMESTEP_SECONDS,
        temporal_accuracy_gates=dict(
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
        ),
        physical_ledger_tolerance=1.0e-10,
        safety_factor=0.8,
        minimum_factor=0.25,
        maximum_factor=2.0,
        maximum_retries=6,
        output_horizon_seconds=TARGET_DURATION_SECONDS,
        horizon_budget_fraction=1.0,
    ).validated()


def _landing_config(
    config: CausalFiveFieldTemporalControllerConfig,
    requested_dt: float,
) -> CausalFiveFieldTemporalControllerConfig:
    if requested_dt >= config.minimum_dt:
        return config
    step_config = replace(
        config.step_config,
        minimum_dt=min(
            config.step_config.minimum_dt,
            0.5 * requested_dt,
        ),
    ).validated()
    return replace(
        config,
        step_config=step_config,
        minimum_dt=requested_dt,
    ).validated()


def _step_summary(step) -> dict:
    return {
        "accepted": bool(step.accepted),
        "timestep_seconds": float(step.timestep_seconds),
        "maximum_scaled_residual": float(
            step.maximum_scaled_residual
        ),
        "maximum_scaled_algebraic_residual": float(
            step.maximum_scaled_algebraic_residual
        ),
        "maximum_scaled_primitive_change": float(
            step.maximum_scaled_primitive_change
        ),
        "maximum_scaled_total_change": float(
            step.maximum_scaled_total_change
        ),
        "conservation_telescoping_relative_defect": float(
            step.conservation_telescoping_relative_defect
        ),
        "iterations": int(step.iterations),
        "function_evaluations": int(step.function_evaluations),
        "jacobian_evaluations": int(step.jacobian_evaluations),
        "maximum_linear_residual": float(
            step.maximum_linear_residual
        ),
        "message": str(step.message),
    }


def _contract_summary(contract) -> dict:
    return {
        "passed": bool(contract.passed),
        "maximum_relative_ledger_defect": (
            contract.maximum_relative_ledger_defect
        ),
        "state_gates_passed": bool(
            contract.state_gates is not None
            and contract.state_gates["passed"]
        ),
    }


def _attempt_summary(
    attempt: CausalFiveFieldStepDoublingAttempt,
) -> dict:
    return {
        "timestep_seconds": float(attempt.timestep_seconds),
        "accepted": bool(attempt.accepted),
        "failure_class": str(attempt.failure_class),
        "proposed_factor": float(attempt.proposed_factor),
        "temporal_budget_fraction": float(
            attempt.temporal_budget_fraction
        ),
        "effective_temporal_accuracy_gates": dict(
            attempt.effective_temporal_accuracy_gates
        ),
        "temporal_errors": attempt.temporal_errors,
        "temporal_gate_audit": attempt.temporal_gate_audit,
        "full_step": _step_summary(attempt.full_step),
        "first_half_step": _step_summary(
            attempt.first_half_step
        ),
        "second_half_step": (
            None
            if attempt.second_half_step is None
            else _step_summary(attempt.second_half_step)
        ),
        "full_contract": _contract_summary(attempt.full_contract),
        "first_half_contract": _contract_summary(
            attempt.first_half_contract
        ),
        "second_half_contract": (
            None
            if attempt.second_half_contract is None
            else _contract_summary(attempt.second_half_contract)
        ),
    }


def _attempt_work(
    attempts: tuple[CausalFiveFieldStepDoublingAttempt, ...],
) -> dict[str, int]:
    steps = []
    for attempt in attempts:
        steps.extend((attempt.full_step, attempt.first_half_step))
        if attempt.second_half_step is not None:
            steps.append(attempt.second_half_step)
    return {
        "implicit_solves": len(steps),
        "function_evaluations": int(
            sum(step.function_evaluations for step in steps)
        ),
        "jacobian_evaluations": int(
            sum(step.jacobian_evaluations for step in steps)
        ),
        "newton_iterations": int(
            sum(step.iterations for step in steps)
        ),
    }


def _sum_work(rows: list[dict]) -> dict[str, int]:
    names = (
        "implicit_solves",
        "function_evaluations",
        "jacobian_evaluations",
        "newton_iterations",
    )
    return {
        name: int(sum(row["work"][name] for row in rows))
        for name in names
    }


def _make_restart(
    initial: CausalFiveFieldAdaptiveRestart,
    *,
    state_vector: np.ndarray,
    previous_increment: np.ndarray,
    elapsed_time: float,
    dt_next: float,
    previous_dt: float,
    extension_accepted_steps: int,
    extension_rejected_trials: int,
    role: str,
) -> CausalFiveFieldAdaptiveRestart:
    provenance = {
        "work_package": "WP10c6d",
        "n_cells": N_CELLS,
        "role": role,
        "source": "exact circularized regression stream",
        "output_horizon_seconds": TARGET_DURATION_SECONDS,
        "observable_schema": (
            CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
        ),
        "controller": (
            "dt-weighted global observable budget with accepted "
            "two-half-step state"
        ),
    }
    return CausalFiveFieldAdaptiveRestart(
        state_vector=np.asarray(state_vector, dtype=float),
        previous_physical_increment=np.asarray(
            previous_increment,
            dtype=float,
        ),
        elapsed_time=float(elapsed_time),
        dt_next=float(dt_next),
        previous_dt=float(previous_dt),
        accepted_steps=(
            initial.accepted_steps + extension_accepted_steps
        ),
        rejected_attempts=(
            initial.rejected_attempts + extension_rejected_trials
        ),
        provenance=provenance,
    )


def _run_fixed_reference(
    context: CausalFiveFieldDAEContext,
    initial: CausalFiveFieldAdaptiveRestart,
    subdivisions: int,
) -> dict:
    timestep = TARGET_DURATION_SECONDS / subdivisions
    state = np.asarray(initial.state_vector, dtype=float)
    previous_increment = np.asarray(
        initial.previous_physical_increment,
        dtype=float,
    )
    previous_dt = float(initial.previous_dt)
    rows: list[dict] = []
    passed = True
    terminal_message = "fixed reference completed"
    step_config = _step_config()

    for index in range(subdivisions):
        predictor = previous_increment * (timestep / previous_dt)
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
            physical_ledger_tolerance=1.0e-10,
        )
        rows.append(
            {
                "step": index + 1,
                "step_summary": _step_summary(step),
                "contract": _contract_summary(contract),
            }
        )
        if not contract.passed:
            passed = False
            terminal_message = "fixed reference failed a step contract"
            break
        state = np.asarray(step.state_vector, dtype=float)
        previous_increment = np.asarray(
            step.physical_increment,
            dtype=float,
        )
        previous_dt = timestep
        if (index + 1) % max(16, subdivisions // 4) == 0:
            print(
                json.dumps(
                    {
                        "mode": f"reference_{subdivisions}",
                        "completed_steps": index + 1,
                        "total_steps": subdivisions,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    state_gates = audit_causal_five_field_state_gates(context, state)
    work = {
        "implicit_solves": len(rows),
        "function_evaluations": int(
            sum(
                row["step_summary"]["function_evaluations"]
                for row in rows
            )
        ),
        "jacobian_evaluations": int(
            sum(
                row["step_summary"]["jacobian_evaluations"]
                for row in rows
            )
        ),
        "newton_iterations": int(
            sum(
                row["step_summary"]["iterations"] for row in rows
            )
        ),
    }
    return {
        "subdivisions": subdivisions,
        "timestep_seconds": timestep,
        "state_vector": state,
        "previous_increment": previous_increment,
        "completed_steps": len(rows),
        "state_gates": state_gates,
        "rows": rows,
        "work": work,
        "terminal_message": terminal_message,
        "passed": bool(
            passed
            and len(rows) == subdivisions
            and state_gates["passed"]
        ),
    }


def _run_budgeted(
    context: CausalFiveFieldDAEContext,
    initial: CausalFiveFieldAdaptiveRestart,
    *,
    restart_split: bool,
) -> dict:
    config = _controller_config(context)
    state = np.asarray(initial.state_vector, dtype=float)
    previous_increment = np.asarray(
        initial.previous_physical_increment,
        dtype=float,
    )
    previous_dt = float(initial.previous_dt)
    elapsed = float(initial.elapsed_time)
    target = elapsed + TARGET_DURATION_SECONDS
    dt_next = INITIAL_TIMESTEP_SECONDS
    extension_accepted = 0
    extension_rejected = 0
    cumulative_budget_fraction = 0.0
    rows: list[dict] = []
    restart_roundtrip_bitwise = not restart_split
    restart_performed = False
    terminal_message = "target reached"
    tolerance = max(
        1.0e-20,
        TARGET_TIME_RELATIVE_TOLERANCE * target,
    )

    while True:
        remaining = target - elapsed
        if abs(remaining) <= tolerance:
            elapsed = target
            break
        if remaining <= 0.0:
            terminal_message = "target overshot"
            break
        requested_dt = min(dt_next, remaining)
        local_config = _landing_config(config, requested_dt)
        result = (
            advance_causal_five_field_step_doubling_backward_euler(
                context,
                state,
                requested_dt,
                previous_increment,
                previous_dt,
                local_config,
            )
        )
        rejected = sum(
            not attempt.accepted for attempt in result.attempts
        )
        extension_rejected += rejected
        row = {
            "accepted_step": (
                extension_accepted + 1 if result.accepted else None
            ),
            "elapsed_time_before_seconds": elapsed,
            "requested_timestep_seconds": requested_dt,
            "accepted": bool(result.accepted),
            "dt_used_seconds": float(result.dt_used),
            "dt_next_seconds": float(result.dt_next),
            "normalized_error": result.normalized_error,
            "message": str(result.message),
            "attempts": [
                _attempt_summary(attempt)
                for attempt in result.attempts
            ],
            "work": _attempt_work(result.attempts),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "mode": (
                        "budget_restart_replay"
                        if restart_split
                        else "horizon_budget"
                    ),
                    "accepted_step": row["accepted_step"],
                    "requested_dt": requested_dt,
                    "dt_used": result.dt_used,
                    "dt_next": result.dt_next,
                    "normalized_error": result.normalized_error,
                    "rejected_trials": rejected,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not result.accepted:
            terminal_message = result.message
            break
        state = np.asarray(result.state_vector, dtype=float)
        previous_increment = np.asarray(
            result.physical_increment,
            dtype=float,
        )
        previous_dt = float(result.dt_used)
        dt_next = float(result.dt_next)
        cumulative_budget_fraction += (
            result.dt_used / TARGET_DURATION_SECONDS
        )
        new_elapsed = elapsed + result.dt_used
        if result.dt_used == remaining:
            new_elapsed = target
        elapsed = new_elapsed
        extension_accepted += 1

        if (
            restart_split
            and not restart_performed
            and extension_accepted
            == RESTART_SPLIT_ACCEPTED_STEPS
        ):
            split_restart = _make_restart(
                initial,
                state_vector=state,
                previous_increment=previous_increment,
                elapsed_time=elapsed,
                dt_next=dt_next,
                previous_dt=previous_dt,
                extension_accepted_steps=extension_accepted,
                extension_rejected_trials=extension_rejected,
                role="interrupted_horizon_budget_replay",
            )
            split_path = (
                OUTPUT_CHECKPOINT_DIRECTORY
                / "causal_wp10c6d_N016_split.npz"
            )
            save_causal_five_field_adaptive_restart(
                split_path,
                context,
                split_restart,
            )
            restored = load_causal_five_field_adaptive_restart(
                split_path,
                context,
            )
            restart_roundtrip_bitwise = _restart_is_bitwise(
                split_restart,
                restored,
            )
            state = np.asarray(restored.state_vector, dtype=float)
            previous_increment = np.asarray(
                restored.previous_physical_increment,
                dtype=float,
            )
            elapsed = float(restored.elapsed_time)
            dt_next = float(restored.dt_next)
            previous_dt = float(restored.previous_dt)
            restart_performed = True

    target_reached = bool(abs(elapsed - target) <= tolerance)
    state_gates = audit_causal_five_field_state_gates(context, state)
    budget_sum_passed = bool(
        target_reached
        and np.isclose(
            cumulative_budget_fraction,
            1.0,
            rtol=0.0,
            atol=2.0e-13,
        )
    )
    return {
        "state_vector": state,
        "previous_increment": previous_increment,
        "elapsed_time": elapsed,
        "target_elapsed_time_seconds": target,
        "dt_next": dt_next,
        "previous_dt": previous_dt,
        "extension_accepted_steps": extension_accepted,
        "extension_rejected_trials": extension_rejected,
        "cumulative_budget_fraction": cumulative_budget_fraction,
        "budget_sum_passed": budget_sum_passed,
        "restart_split_requested": restart_split,
        "restart_split_performed": restart_performed,
        "restart_roundtrip_bitwise": restart_roundtrip_bitwise,
        "state_gates": state_gates,
        "rows": rows,
        "work": _sum_work(rows),
        "terminal_message": terminal_message,
        "passed": bool(
            target_reached
            and budget_sum_passed
            and state_gates["passed"]
            and (
                not restart_split
                or (
                    restart_performed
                    and restart_roundtrip_bitwise
                )
            )
        ),
    }


def _public_run(run: dict) -> dict:
    return {
        key: value
        for key, value in run.items()
        if key not in {"state_vector", "previous_increment"}
    }


def _endpoint_errors(
    context: CausalFiveFieldDAEContext,
    baseline: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float | list[float]]:
    return compare_causal_five_field_endpoint_vectors(
        context,
        baseline,
        left,
        right,
        cooling_inner_cutoff=(
            COOLING_INNER_CUTOFF_RG
            * context.grid.gravitational_radius
        ),
    )


def _build_output(reference_only: bool) -> dict:
    context = make_causal_five_field_regression_context(N_CELLS)
    if not INITIAL_CHECKPOINT.exists():
        raise FileNotFoundError(INITIAL_CHECKPOINT)
    initial = load_causal_five_field_adaptive_restart(
        INITIAL_CHECKPOINT,
        context,
    )
    provenance_passed = bool(
        initial.provenance.get("work_package") == "WP10c5q"
        and initial.provenance.get("n_cells") == N_CELLS
        and "exact circularized regression stream"
        in str(initial.provenance.get("source", ""))
    )
    initial_state_gates = audit_causal_five_field_state_gates(
        context,
        initial.state_vector,
    )
    if not provenance_passed or not initial_state_gates["passed"]:
        raise RuntimeError("WP10c5q restart prerequisite failed")

    references = {
        subdivisions: _run_fixed_reference(
            context,
            initial,
            subdivisions,
        )
        for subdivisions in REFERENCE_SUBDIVISIONS
    }
    coarse_errors = _endpoint_errors(
        context,
        initial.state_vector,
        references[32]["state_vector"],
        references[64]["state_vector"],
    )
    fine_errors = _endpoint_errors(
        context,
        initial.state_vector,
        references[64]["state_vector"],
        references[128]["state_vector"],
    )
    reference_convergence = (
        audit_causal_five_field_reference_convergence(
            coarse_errors,
            fine_errors,
            dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1),
            maximum_reference_uncertainty_fraction=(
                MAXIMUM_REFERENCE_UNCERTAINTY_FRACTION
            ),
            minimum_observed_order=MINIMUM_OBSERVED_ORDER,
            order_floor_fraction=ORDER_FLOOR_FRACTION,
        )
    )
    reference_passed = bool(
        all(run["passed"] for run in references.values())
        and reference_convergence["passed"]
    )

    output = {
        "work_package": "WP10c6d",
        "scope": (
            "bounded N16 fixed-reference convergence and dt-weighted "
            "global observable-budget controller"
        ),
        "construction": {
            "n_cells": N_CELLS,
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "temporal_accuracy_gates": dict(
                CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
            ),
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "target_duration_in_shared_ceilings": 8.0,
            "reference_subdivisions": list(
                REFERENCE_SUBDIVISIONS
            ),
            "maximum_reference_uncertainty_fraction": (
                MAXIMUM_REFERENCE_UNCERTAINTY_FRACTION
            ),
            "minimum_observed_order": MINIMUM_OBSERVED_ORDER,
            "order_floor_fraction": ORDER_FLOOR_FRACTION,
            "initial_timestep_seconds": INITIAL_TIMESTEP_SECONDS,
            "accepted_state": "two_half_step_state",
            "local_budget_rule": (
                "global_gate * dt / output_horizon"
            ),
            "factor_rule": (
                "clip(0.8 / normalized_budget_error, 0.25, 2.0)"
            ),
            "reference_only_requested": reference_only,
            "no_n32_n64_n128_production_run": True,
            "no_long_timescale_or_physics_run": True,
        },
        "checkpoint": {
            "path": str(INITIAL_CHECKPOINT.relative_to(ROOT)),
            "sha256": _sha256(INITIAL_CHECKPOINT),
            "elapsed_time_seconds": initial.elapsed_time,
            "provenance": initial.provenance,
            "provenance_passed": provenance_passed,
            "state_gates": initial_state_gates,
        },
        "fixed_references": {
            str(subdivisions): _public_run(run)
            for subdivisions, run in references.items()
        },
        "reference_errors": {
            "32_to_64": coarse_errors,
            "64_to_128": fine_errors,
        },
        "reference_convergence": reference_convergence,
        "reference_gate_passed": reference_passed,
        "local_controller_endpoint_audit": None,
        "horizon_budget": None,
        "restart_replay": None,
        "horizon_budget_endpoint_audit": None,
        "work_audit": None,
        "final_checkpoint": None,
        "authorization": {
            "n16_reference_certified": reference_passed,
            "n16_horizon_budget_controller_certified": False,
            "n32_controller_run_authorized": False,
            "n64_n128_production_run_authorized": False,
            "long_evolution_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_hot_state_or_cycle_certified": False,
        },
        "decision": (
            "reference_gate_passed_controller_not_requested"
            if reference_only and reference_passed
            else (
                "stop_reference_gate_failed"
                if not reference_passed
                else "run_horizon_budget_controller"
            )
        ),
        "passed": False,
    }
    if reference_only or not reference_passed:
        return output

    if not LOCAL_CONTROLLER_CHECKPOINT.exists():
        raise FileNotFoundError(LOCAL_CONTROLLER_CHECKPOINT)
    local_controller = load_causal_five_field_adaptive_restart(
        LOCAL_CONTROLLER_CHECKPOINT,
        context,
    )
    expected_target = initial.elapsed_time + TARGET_DURATION_SECONDS
    local_target_passed = bool(
        np.isclose(
            local_controller.elapsed_time,
            expected_target,
            rtol=0.0,
            atol=max(
                1.0e-20,
                TARGET_TIME_RELATIVE_TOLERANCE * expected_target,
            ),
        )
    )
    if not local_target_passed:
        raise RuntimeError("WP10c6c endpoint has the wrong target time")
    local_errors = _endpoint_errors(
        context,
        initial.state_vector,
        local_controller.state_vector,
        references[128]["state_vector"],
    )
    local_audit = (
        audit_causal_five_field_endpoint_with_reference_uncertainty(
            local_errors,
            fine_errors,
            dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1),
        )
    )
    output["local_controller_endpoint_audit"] = {
        "checkpoint": {
            "path": str(
                LOCAL_CONTROLLER_CHECKPOINT.relative_to(ROOT)
            ),
            "sha256": _sha256(LOCAL_CONTROLLER_CHECKPOINT),
            "target_time_passed": local_target_passed,
        },
        "errors_against_128_step_reference": local_errors,
        "combined_audit": local_audit,
    }

    budget = _run_budgeted(
        context,
        initial,
        restart_split=False,
    )
    replay = _run_budgeted(
        context,
        initial,
        restart_split=True,
    )
    replay_final_bitwise = bool(
        np.array_equal(
            budget["state_vector"],
            replay["state_vector"],
        )
        and np.array_equal(
            budget["previous_increment"],
            replay["previous_increment"],
        )
        and budget["elapsed_time"] == replay["elapsed_time"]
        and budget["dt_next"] == replay["dt_next"]
        and budget["previous_dt"] == replay["previous_dt"]
        and budget["extension_accepted_steps"]
        == replay["extension_accepted_steps"]
        and budget["extension_rejected_trials"]
        == replay["extension_rejected_trials"]
        and budget["cumulative_budget_fraction"]
        == replay["cumulative_budget_fraction"]
    )
    budget_errors = _endpoint_errors(
        context,
        initial.state_vector,
        budget["state_vector"],
        references[128]["state_vector"],
    )
    budget_audit = (
        audit_causal_five_field_endpoint_with_reference_uncertainty(
            budget_errors,
            fine_errors,
            dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1),
        )
    )
    final_restart = _make_restart(
        initial,
        state_vector=budget["state_vector"],
        previous_increment=budget["previous_increment"],
        elapsed_time=budget["elapsed_time"],
        dt_next=budget["dt_next"],
        previous_dt=budget["previous_dt"],
        extension_accepted_steps=budget[
            "extension_accepted_steps"
        ],
        extension_rejected_trials=budget[
            "extension_rejected_trials"
        ],
        role="bounded_horizon_budget_validation",
    )
    final_path = (
        OUTPUT_CHECKPOINT_DIRECTORY
        / "causal_wp10c6d_N016_final.npz"
    )
    save_causal_five_field_adaptive_restart(
        final_path,
        context,
        final_restart,
    )
    restored = load_causal_five_field_adaptive_restart(
        final_path,
        context,
    )
    final_roundtrip_bitwise = _restart_is_bitwise(
        final_restart,
        restored,
    )
    reference_work = references[128]["work"]
    adaptive_work = budget["work"]
    work_ratio = {
        name: (
            float(reference_work[name] / adaptive_work[name])
            if adaptive_work[name] > 0
            else None
        )
        for name in reference_work
    }
    passed = bool(
        budget["passed"]
        and replay["passed"]
        and replay_final_bitwise
        and final_roundtrip_bitwise
        and budget_audit["passed"]
    )
    output["horizon_budget"] = _public_run(budget)
    output["restart_replay"] = {
        **_public_run(replay),
        "final_bitwise_equal_to_uninterrupted": (
            replay_final_bitwise
        ),
    }
    output["horizon_budget_endpoint_audit"] = {
        "errors_against_128_step_reference": budget_errors,
        "combined_audit": budget_audit,
    }
    output["work_audit"] = {
        "horizon_budget": adaptive_work,
        "reference_128": reference_work,
        "reference_over_horizon_budget": work_ratio,
    }
    output["final_checkpoint"] = {
        "path": str(final_path.relative_to(ROOT)),
        "sha256": _sha256(final_path),
        "roundtrip_bitwise": final_roundtrip_bitwise,
    }
    output["authorization"][
        "n16_horizon_budget_controller_certified"
    ] = passed
    output["authorization"][
        "n32_controller_run_authorized"
    ] = passed
    output["decision"] = (
        "n16_passed_n32_bounded_confirmation_authorized"
        if passed
        else "stop_after_n16_horizon_budget_gate"
    )
    output["passed"] = passed
    return output


def main() -> None:
    args = _arguments()
    output = _build_output(args.reference_only)
    output_path = _absolute(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "reference_gate_passed": output[
                    "reference_gate_passed"
                ],
                "decision": output["decision"],
                "passed": output["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
