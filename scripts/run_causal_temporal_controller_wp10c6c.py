"""Validate the causal step-doubling controller at N16 and N32."""

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
    audit_causal_five_field_state_gates,
    audit_causal_five_field_temporal_step_contract,
    causal_five_field_dae_scaling,
    causal_five_field_observable_snapshot,
    causal_five_field_temporal_error_ratio,
    compare_causal_five_field_observables,
    evaluate_causal_five_field_dae,
    load_causal_five_field_adaptive_restart,
    make_causal_five_field_regression_context,
    save_causal_five_field_adaptive_restart,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
RESTART_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c5k"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c6c"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables"
    / "causal_temporal_controller_wp10c6c.json"
)
SUPPORTED_CELL_COUNTS = (16, 32)
SHARED_PASSING_CEILING_SECONDS = 1.9218219974586337e-3
FIRST_FAILING_TIMESTEP_SECONDS = 3.8436439949172674e-3
INITIAL_TIMESTEP_SECONDS = 0.5 * SHARED_PASSING_CEILING_SECONDS
TARGET_DURATION_SECONDS = 8.0 * SHARED_PASSING_CEILING_SECONDS
REFERENCE_STEPS = 64
REFERENCE_TIMESTEP_SECONDS = TARGET_DURATION_SECONDS / REFERENCE_STEPS
COOLING_INNER_CUTOFF_RG = 6.0
FINITE_DIFFERENCE_STEP = 2.0e-6
RESTART_SPLIT_ACCEPTED_STEPS = 3
TARGET_TIME_RELATIVE_TOLERANCE = 5.0e-14


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-cells",
        type=int,
        action="append",
        choices=SUPPORTED_CELL_COUNTS,
        default=None,
        help="Repeat to select meshes; default runs N16 and N32.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _checkpoint_path(n_cells: int) -> Path:
    return (
        RESTART_DIRECTORY
        / f"causal_wp10c5q_N{n_cells:03d}_final.npz"
    )


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


def _controller_config(
    context: CausalFiveFieldDAEContext,
) -> CausalFiveFieldTemporalControllerConfig:
    gravitational_radius = context.grid.gravitational_radius
    minimum_dt = INITIAL_TIMESTEP_SECONDS / 4096.0
    step_config = CausalFiveFieldAdaptiveStepConfig(
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
    return CausalFiveFieldTemporalControllerConfig(
        step_config=step_config,
        cooling_inner_cutoff=(
            COOLING_INNER_CUTOFF_RG * gravitational_radius
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


def _contract_summary(contract) -> dict | None:
    if contract is None:
        return None
    return {
        "passed": bool(contract.passed),
        "state_gates": contract.state_gates,
        "component_relative_ledger_defects": (
            None
            if contract.component_relative_ledger_defects is None
            else list(contract.component_relative_ledger_defects)
        ),
        "maximum_relative_ledger_defect": (
            contract.maximum_relative_ledger_defect
        ),
    }


def _attempt_summary(
    attempt: CausalFiveFieldStepDoublingAttempt,
) -> dict:
    return {
        "timestep_seconds": float(attempt.timestep_seconds),
        "accepted": bool(attempt.accepted),
        "failure_class": attempt.failure_class,
        "proposed_factor": float(attempt.proposed_factor),
        "full_step": _step_summary(attempt.full_step),
        "first_half_step": _step_summary(attempt.first_half_step),
        "second_half_step": (
            None
            if attempt.second_half_step is None
            else _step_summary(attempt.second_half_step)
        ),
        "full_contract": _contract_summary(attempt.full_contract),
        "first_half_contract": _contract_summary(
            attempt.first_half_contract
        ),
        "second_half_contract": _contract_summary(
            attempt.second_half_contract
        ),
        "temporal_errors": attempt.temporal_errors,
        "temporal_gate_audit": attempt.temporal_gate_audit,
        "temporal_budget_fraction": (
            attempt.temporal_budget_fraction
        ),
        "effective_temporal_accuracy_gates": (
            attempt.effective_temporal_accuracy_gates
        ),
    }


def _attempt_work(
    attempts: tuple[CausalFiveFieldStepDoublingAttempt, ...],
) -> dict[str, int]:
    steps = []
    for attempt in attempts:
        steps.extend(
            (attempt.full_step, attempt.first_half_step)
        )
        if attempt.second_half_step is not None:
            steps.append(attempt.second_half_step)
    return {
        "implicit_solves": len(steps),
        "function_evaluations": sum(
            step.function_evaluations for step in steps
        ),
        "jacobian_evaluations": sum(
            step.jacobian_evaluations for step in steps
        ),
        "newton_iterations": sum(step.iterations for step in steps),
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
    *,
    n_cells: int,
    state_vector: np.ndarray,
    previous_increment: np.ndarray,
    elapsed_time: float,
    dt_next: float,
    previous_dt: float,
    accepted_steps: int,
    rejected_attempts: int,
    role: str,
) -> CausalFiveFieldAdaptiveRestart:
    return CausalFiveFieldAdaptiveRestart(
        state_vector=np.asarray(state_vector, dtype=float),
        previous_physical_increment=np.asarray(
            previous_increment,
            dtype=float,
        ),
        elapsed_time=float(elapsed_time),
        dt_next=float(dt_next),
        previous_dt=float(previous_dt),
        accepted_steps=int(accepted_steps),
        rejected_attempts=int(rejected_attempts),
        provenance={
            "work_package": "WP10c6c",
            "parent_work_package": "WP10c5q",
            "n_cells": int(n_cells),
            "role": role,
            "source": (
                "exact circularized regression stream; not ballistic "
                "Layer-1 calibration"
            ),
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
        },
    )


def _run_adaptive(
    context: CausalFiveFieldDAEContext,
    initial: CausalFiveFieldAdaptiveRestart,
    *,
    n_cells: int,
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
            "message": result.message,
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
                    "mesh": n_cells,
                    "mode": (
                        "restart_replay"
                        if restart_split
                        else "adaptive"
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
                n_cells=n_cells,
                state_vector=state,
                previous_increment=previous_increment,
                elapsed_time=elapsed,
                dt_next=dt_next,
                previous_dt=previous_dt,
                accepted_steps=(
                    initial.accepted_steps + extension_accepted
                ),
                rejected_attempts=(
                    initial.rejected_attempts + extension_rejected
                ),
                role="interrupted_controller_replay",
            )
            split_path = (
                OUTPUT_CHECKPOINT_DIRECTORY
                / f"causal_wp10c6c_N{n_cells:03d}_split.npz"
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
    final_state_gates = audit_causal_five_field_state_gates(
        context,
        state,
    )
    return {
        "state_vector": state,
        "previous_increment": previous_increment,
        "elapsed_time": elapsed,
        "dt_next": dt_next,
        "previous_dt": previous_dt,
        "extension_accepted_steps": extension_accepted,
        "extension_rejected_trials": extension_rejected,
        "target_reached": target_reached,
        "target_elapsed_time_seconds": target,
        "restart_split_requested": restart_split,
        "restart_split_performed": restart_performed,
        "restart_roundtrip_bitwise": restart_roundtrip_bitwise,
        "state_gates": final_state_gates,
        "rows": rows,
        "work": _sum_work(rows),
        "terminal_message": terminal_message,
        "passed": bool(
            target_reached
            and final_state_gates["passed"]
            and (
                not restart_split
                or (
                    restart_performed
                    and restart_roundtrip_bitwise
                )
            )
        ),
    }


def _run_fixed_reference(
    context: CausalFiveFieldDAEContext,
    initial: CausalFiveFieldAdaptiveRestart,
    *,
    n_cells: int,
) -> dict:
    config = _controller_config(context)
    state = np.asarray(initial.state_vector, dtype=float)
    previous_increment = np.asarray(
        initial.previous_physical_increment,
        dtype=float,
    )
    previous_dt = float(initial.previous_dt)
    rows: list[dict] = []
    passed = True
    terminal_message = "fixed reference completed"

    for index in range(REFERENCE_STEPS):
        predictor = previous_increment * (
            REFERENCE_TIMESTEP_SECONDS / previous_dt
        )
        step = advance_causal_five_field_increment_backward_euler(
            context,
            state,
            REFERENCE_TIMESTEP_SECONDS,
            predictor,
            config.step_config,
        )
        contract = audit_causal_five_field_temporal_step_contract(
            context,
            state,
            step,
            physical_ledger_tolerance=(
                config.physical_ledger_tolerance
            ),
        )
        row = {
            "step": index + 1,
            "accepted": bool(step.accepted),
            "step_summary": _step_summary(step),
            "contract": _contract_summary(contract),
        }
        rows.append(row)
        if not contract.passed:
            passed = False
            terminal_message = (
                "fixed reference failed a step contract"
            )
            break
        state = np.asarray(step.state_vector, dtype=float)
        previous_increment = np.asarray(
            step.physical_increment,
            dtype=float,
        )
        previous_dt = REFERENCE_TIMESTEP_SECONDS
        if (index + 1) % 16 == 0:
            print(
                json.dumps(
                    {
                        "mesh": n_cells,
                        "mode": "fixed_reference",
                        "completed_steps": index + 1,
                        "total_steps": REFERENCE_STEPS,
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
        "state_vector": state,
        "previous_increment": previous_increment,
        "completed_steps": len(rows),
        "timestep_seconds": REFERENCE_TIMESTEP_SECONDS,
        "target_duration_seconds": TARGET_DURATION_SECONDS,
        "state_gates": state_gates,
        "rows": rows,
        "work": work,
        "terminal_message": terminal_message,
        "passed": bool(
            passed
            and len(rows) == REFERENCE_STEPS
            and state_gates["passed"]
        ),
    }


def _final_accuracy(
    context: CausalFiveFieldDAEContext,
    baseline_vector: np.ndarray,
    adaptive_vector: np.ndarray,
    reference_vector: np.ndarray,
) -> dict:
    cutoff = (
        COOLING_INNER_CUTOFF_RG
        * context.grid.gravitational_radius
    )
    adaptive = causal_five_field_observable_snapshot(
        context,
        adaptive_vector,
        cooling_inner_cutoff=cutoff,
    )
    reference = causal_five_field_observable_snapshot(
        context,
        reference_vector,
        cooling_inner_cutoff=cutoff,
    )
    errors = compare_causal_five_field_observables(
        adaptive,
        reference,
    )
    baseline_state = unpack_causal_five_field_state(
        baseline_vector,
        context.grid.centers.size,
    )
    baseline_evaluation = evaluate_causal_five_field_dae(
        baseline_vector,
        context,
    )
    scales = causal_five_field_dae_scaling(
        baseline_state,
        baseline_evaluation,
    ).column_scales
    errors["maximum_baseline_scaled_state_difference"] = float(
        np.max(
            np.abs(
                (adaptive_vector - reference_vector) / scales
            )
        )
    )
    audit = causal_five_field_temporal_error_ratio(
        errors,
        dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1),
    )
    return {
        "errors": errors,
        "gate_audit": audit,
        "passed": bool(audit["passed"]),
    }


def _public_run(run: dict) -> dict:
    return {
        key: value
        for key, value in run.items()
        if key not in {"state_vector", "previous_increment"}
    }


def _run_resolution(n_cells: int) -> dict:
    context = make_causal_five_field_regression_context(n_cells)
    source_path = _checkpoint_path(n_cells)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    initial = load_causal_five_field_adaptive_restart(
        source_path,
        context,
    )
    provenance_passed = bool(
        initial.provenance.get("work_package") == "WP10c5q"
        and initial.provenance.get("n_cells") == n_cells
        and "exact circularized regression stream"
        in str(initial.provenance.get("source", ""))
    )
    initial_state_gates = audit_causal_five_field_state_gates(
        context,
        initial.state_vector,
    )
    if not provenance_passed or not initial_state_gates["passed"]:
        raise RuntimeError("WP10c5q restart prerequisite failed")

    adaptive = _run_adaptive(
        context,
        initial,
        n_cells=n_cells,
        restart_split=False,
    )
    replay = _run_adaptive(
        context,
        initial,
        n_cells=n_cells,
        restart_split=True,
    )
    reference = _run_fixed_reference(
        context,
        initial,
        n_cells=n_cells,
    )
    replay_final_bitwise = bool(
        np.array_equal(
            adaptive["state_vector"],
            replay["state_vector"],
        )
        and np.array_equal(
            adaptive["previous_increment"],
            replay["previous_increment"],
        )
        and adaptive["elapsed_time"] == replay["elapsed_time"]
        and adaptive["dt_next"] == replay["dt_next"]
        and adaptive["previous_dt"] == replay["previous_dt"]
        and adaptive["extension_accepted_steps"]
        == replay["extension_accepted_steps"]
        and adaptive["extension_rejected_trials"]
        == replay["extension_rejected_trials"]
    )
    accuracy = _final_accuracy(
        context,
        initial.state_vector,
        adaptive["state_vector"],
        reference["state_vector"],
    )
    final_restart = _make_restart(
        n_cells=n_cells,
        state_vector=adaptive["state_vector"],
        previous_increment=adaptive["previous_increment"],
        elapsed_time=adaptive["elapsed_time"],
        dt_next=adaptive["dt_next"],
        previous_dt=adaptive["previous_dt"],
        accepted_steps=(
            initial.accepted_steps
            + adaptive["extension_accepted_steps"]
        ),
        rejected_attempts=(
            initial.rejected_attempts
            + adaptive["extension_rejected_trials"]
        ),
        role="bounded_temporal_controller_validation",
    )
    final_path = (
        OUTPUT_CHECKPOINT_DIRECTORY
        / f"causal_wp10c6c_N{n_cells:03d}_final.npz"
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
    adaptive_work = adaptive["work"]
    reference_work = reference["work"]
    work_ratio = {
        name: (
            float(reference_work[name] / adaptive_work[name])
            if adaptive_work[name] > 0
            else None
        )
        for name in adaptive_work
    }
    passed = bool(
        adaptive["passed"]
        and replay["passed"]
        and reference["passed"]
        and replay_final_bitwise
        and final_roundtrip_bitwise
        and accuracy["passed"]
    )
    return {
        "n_cells": n_cells,
        "checkpoint": {
            "path": str(source_path.relative_to(ROOT)),
            "sha256": _sha256(source_path),
            "elapsed_time_seconds": initial.elapsed_time,
            "accepted_steps": initial.accepted_steps,
            "rejected_attempts": initial.rejected_attempts,
            "provenance": initial.provenance,
            "provenance_passed": provenance_passed,
            "state_gates": initial_state_gates,
        },
        "adaptive": _public_run(adaptive),
        "restart_replay": _public_run(replay),
        "fixed_reference": _public_run(reference),
        "final_accuracy_against_reference": accuracy,
        "restart_replay_final_bitwise": replay_final_bitwise,
        "final_checkpoint": {
            "path": str(final_path.relative_to(ROOT)),
            "sha256": _sha256(final_path),
            "roundtrip_bitwise": final_roundtrip_bitwise,
        },
        "reference_over_adaptive_work_ratio": work_ratio,
        "passed": passed,
    }


def main() -> None:
    args = _arguments()
    selected = (
        list(SUPPORTED_CELL_COUNTS)
        if args.n_cells is None
        else list(dict.fromkeys(args.n_cells))
    )
    runs = [_run_resolution(n_cells) for n_cells in selected]
    complete_pair = set(selected) == set(SUPPORTED_CELL_COUNTS)
    pair_passed = bool(
        complete_pair
        and all(run["passed"] for run in runs)
        and len(
            {
                run["adaptive"]["target_elapsed_time_seconds"]
                for run in runs
            }
        )
        == 1
    )
    output = {
        "work_package": "WP10c6c",
        "scope": (
            "bounded N16/N32 observable-controlled backward-Euler "
            "step-doubling implementation and matched-duration reference"
        ),
        "construction": {
            "selected_cell_counts": selected,
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "temporal_accuracy_gates": dict(
                CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
            ),
            "shared_passing_ceiling_seconds": (
                SHARED_PASSING_CEILING_SECONDS
            ),
            "first_failing_timestep_seconds": (
                FIRST_FAILING_TIMESTEP_SECONDS
            ),
            "initial_timestep_seconds": INITIAL_TIMESTEP_SECONDS,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "target_duration_in_shared_ceilings": 8.0,
            "fixed_reference_steps": REFERENCE_STEPS,
            "fixed_reference_timestep_seconds": (
                REFERENCE_TIMESTEP_SECONDS
            ),
            "restart_split_after_accepted_steps": (
                RESTART_SPLIT_ACCEPTED_STEPS
            ),
            "accepted_state": "two_half_step_state",
            "factor_rule": (
                "clip(0.8/sqrt(normalized_error), 0.25, 2.0)"
            ),
            "no_n64_n128_run": True,
            "no_long_timescale_run": True,
            "no_physics_change": True,
        },
        "runs": runs,
        "mesh_pair": {
            "complete": complete_pair,
            "same_exact_target_time": (
                pair_passed
                or (
                    complete_pair
                    and len(
                        {
                            run["adaptive"][
                                "target_elapsed_time_seconds"
                            ]
                            for run in runs
                        }
                    )
                    == 1
                )
            ),
            "passed": pair_passed,
        },
        "authorization": {
            "production_temporal_controller_implemented": pair_passed,
            "bounded_n16_n32_controller_validation_certified": (
                pair_passed
            ),
            "bounded_n64_controller_confirmation_authorized": (
                pair_passed
            ),
            "n128_controller_run_authorized": False,
            "long_evolution_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_certified": False,
            "hot_state_certified": False,
            "limit_cycle_certified": False,
        },
        "decision": (
            "bounded_temporal_controller_implemented"
            if pair_passed
            else "stop_after_bounded_controller_validation"
        ),
        "passed": pair_passed,
    }
    output_path = _absolute(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        output,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "selected_cell_counts": selected,
                "run_passed": [
                    {
                        "n_cells": run["n_cells"],
                        "passed": run["passed"],
                    }
                    for run in runs
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
