"""Run the bounded adaptive N16 BDF2 certification campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION,
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    CausalFiveFieldAdaptiveBDF2Config,
    CausalFiveFieldAdaptiveBDF2Restart,
    CausalFiveFieldAdaptiveRestart,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldBDFPhysicalIntervalLedger,
    audit_causal_five_field_endpoint_with_reference_uncertainty,
    audit_causal_five_field_state_gates,
    advance_causal_five_field_adaptive_bdf2,
    causal_five_field_adaptive_bdf2_restarts_equal,
    causal_five_field_bdf_history,
    compare_causal_five_field_endpoint_vectors,
    load_causal_five_field_adaptive_bdf2_restart,
    load_causal_five_field_adaptive_restart,
    load_causal_five_field_bdf_restart,
    make_causal_five_field_regression_context,
    save_causal_five_field_adaptive_bdf2_restart,
)


ROOT = Path(__file__).resolve().parents[1]
N_CELLS = 16
BASE_COMMIT = "7567e725d9fa3cb7781414493fd3b9a49a85f63d"
REFERENCE_BASE_COMMIT = "e4c32bcf04cf1ebe62c46261d41a84bc9377bebb"
FIXED_BDF_BASE_COMMIT = "6a298c69c3c398239e6198c1f07472697929aa2e"
INITIAL_CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c5k"
    / "causal_wp10c5q_N016_final.npz"
)
REFERENCE_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c6e"
)
FIXED_BDF_CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c7b"
    / "causal_wp10c7b_N016_bdf2_S0064.npz"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c7c"
)
FINAL_CHECKPOINT = (
    OUTPUT_CHECKPOINT_DIRECTORY / "causal_wp10c7c_N016_final.npz"
)
SPLIT_CHECKPOINT = (
    OUTPUT_CHECKPOINT_DIRECTORY / "causal_wp10c7c_N016_split.npz"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_adaptive_bdf2_wp10c7c_N016.json"
)
SHARED_PASSING_CEILING_SECONDS = 1.9218219974586337e-3
TARGET_DURATION_SECONDS = 8.0 * SHARED_PASSING_CEILING_SECONDS
INITIAL_TIMESTEP_SECONDS = TARGET_DURATION_SECONDS / 64.0
REFERENCE_SUBDIVISIONS = (256, 512)
COOLING_INNER_CUTOFF_RG = 6.0
TARGET_TIME_RELATIVE_TOLERANCE = 5.0e-14
LOCAL_ERROR_GATE_FRACTION = 0.25
PREDICTOR_ERROR_SCALE = 0.2
AUDIT_INTERVAL = 4
SPLIT_ACCEPTED_STEP = 3
MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT = 1.0e-3
MAXIMUM_ADAPTIVE_TO_FIXED_S64_JACOBIAN_FRACTION = 0.75


@dataclass(frozen=True)
class CampaignResult:
    restart: CausalFiveFieldAdaptiveBDF2Restart
    records: tuple[dict, ...]
    passed: bool
    message: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_path(subdivisions: int) -> Path:
    return (
        REFERENCE_DIRECTORY
        / f"causal_wp10c6e_N016_reference_S{subdivisions:04d}.npz"
    )


def _step_config() -> CausalFiveFieldAdaptiveStepConfig:
    return CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=1.0e-9,
        maximum_dt=3.8436439949172674e-3,
        maximum_scaled_primitive_change=0.2,
        maximum_scaled_total_change=0.25,
        shrink_factor=0.5,
        growth_factor=1.5,
        maximum_retries=0,
        easy_iterations=3,
        residual_tolerance=1.0e-11,
        algebraic_residual_tolerance=1.0e-11,
        conservation_tolerance=1.0e-10,
        finite_difference_step=2.0e-6,
        maximum_newton_iterations=12,
    ).validated()


def _controller_config(context) -> CausalFiveFieldAdaptiveBDF2Config:
    return CausalFiveFieldAdaptiveBDF2Config(
        step_config=_step_config(),
        cooling_inner_cutoff=(
            COOLING_INNER_CUTOFF_RG
            * context.grid.gravitational_radius
        ),
        minimum_dt=1.0e-8,
        maximum_dt=SHARED_PASSING_CEILING_SECONDS,
        local_error_gate_fraction=LOCAL_ERROR_GATE_FRACTION,
        predictor_error_scale=PREDICTOR_ERROR_SCALE,
        safety_factor=0.8,
        minimum_factor=0.5,
        maximum_factor=2.0,
        maximum_retries=6,
        audit_interval=AUDIT_INTERVAL,
    ).validated()


def _zero_ledger() -> CausalFiveFieldBDFPhysicalIntervalLedger:
    zero = np.zeros(5, dtype=float)
    return CausalFiveFieldBDFPhysicalIntervalLedger(
        actual_conserved_storage=np.array(zero, copy=True),
        actual_vertical_storage=np.array(zero, copy=True),
        trapezoidal_boundary_transport=np.array(zero, copy=True),
        trapezoidal_endogenous_source=np.array(zero, copy=True),
        exact_prescribed_stream_source=np.array(zero, copy=True),
        closure_defect=np.array(zero, copy=True),
    )


def _ledger_from_restart(
    restart: CausalFiveFieldAdaptiveBDF2Restart,
) -> CausalFiveFieldBDFPhysicalIntervalLedger:
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


def _add_ledgers(
    left: CausalFiveFieldBDFPhysicalIntervalLedger,
    right: CausalFiveFieldBDFPhysicalIntervalLedger,
) -> CausalFiveFieldBDFPhysicalIntervalLedger:
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


def _ledger_relative(
    ledger: CausalFiveFieldBDFPhysicalIntervalLedger,
) -> tuple[float, list[float]]:
    components = (
        ledger.actual_conserved_storage,
        ledger.actual_vertical_storage,
        ledger.trapezoidal_boundary_transport,
        ledger.trapezoidal_endogenous_source,
        ledger.exact_prescribed_stream_source,
    )
    scale = np.sum(
        np.asarray([np.abs(values) for values in components]),
        axis=0,
    )
    relative = np.abs(ledger.closure_defect) / np.maximum(
        scale,
        np.finfo(float).tiny,
    )
    return float(np.max(relative)), [
        float(value) for value in relative
    ]


def _ledger_json(
    ledger: CausalFiveFieldBDFPhysicalIntervalLedger,
) -> dict:
    maximum, components = _ledger_relative(ledger)
    return {
        "actual_conserved_storage": [
            float(value) for value in ledger.actual_conserved_storage
        ],
        "actual_vertical_storage": [
            float(value) for value in ledger.actual_vertical_storage
        ],
        "trapezoidal_boundary_transport": [
            float(value)
            for value in ledger.trapezoidal_boundary_transport
        ],
        "trapezoidal_endogenous_source": [
            float(value)
            for value in ledger.trapezoidal_endogenous_source
        ],
        "exact_prescribed_stream_source": [
            float(value)
            for value in ledger.exact_prescribed_stream_source
        ],
        "closure_defect": [
            float(value) for value in ledger.closure_defect
        ],
        "component_relative_defects": components,
        "maximum_relative_defect": maximum,
    }


def _restart_with(
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
        cumulative_actual_conserved_storage=(
            cumulative.actual_conserved_storage
        ),
        cumulative_actual_vertical_storage=(
            cumulative.actual_vertical_storage
        ),
        cumulative_boundary_transport=(
            cumulative.trapezoidal_boundary_transport
        ),
        cumulative_endogenous_source=(
            cumulative.trapezoidal_endogenous_source
        ),
        cumulative_stream_source=(
            cumulative.exact_prescribed_stream_source
        ),
        cumulative_closure_defect=cumulative.closure_defect,
        elapsed_time=float(elapsed_time),
        dt_next=float(dt_next),
        next_order=int(next_order),
        accepted_steps=int(accepted_steps),
        accepted_bdf2_steps=int(accepted_bdf2_steps),
        rejected_attempts=int(rejected_attempts),
        audit_count=int(audit_count),
        provenance=dict(template.provenance),
    )


def _attempt_work(attempt) -> dict[str, int]:
    steps = [attempt.step]
    if attempt.independent_audit is not None:
        steps.append(attempt.independent_audit.first_half_step)
        if attempt.independent_audit.second_half_step is not None:
            steps.append(attempt.independent_audit.second_half_step)
    return {
        "implicit_solves": len(steps),
        "function_evaluations": sum(
            int(step.function_evaluations) for step in steps
        ),
        "jacobian_evaluations": sum(
            int(step.jacobian_evaluations) for step in steps
        ),
        "newton_iterations": sum(
            int(step.iterations) for step in steps
        ),
    }


def _record(result, requested_dt: float) -> dict:
    attempt_rows = []
    for attempt in result.attempts:
        work = _attempt_work(attempt)
        attempt_rows.append(
            {
                "timestep_seconds": attempt.timestep_seconds,
                "order": attempt.order,
                "accepted": attempt.accepted,
                "failure_class": attempt.failure_class,
                "proposed_factor": attempt.proposed_factor,
                "maximum_scaled_residual": (
                    attempt.step.maximum_scaled_residual
                ),
                "maximum_discrete_ledger_relative_defect": (
                    attempt.step.maximum_discrete_ledger_relative_defect
                ),
                "local_gate_audit": attempt.local_gate_audit,
                "independent_audit": (
                    {
                        "passed": attempt.independent_audit.passed,
                        "temporal_gate_audit": (
                            attempt.independent_audit.temporal_gate_audit
                        ),
                    }
                    if attempt.independent_audit is not None
                    else None
                ),
                "work": work,
            }
        )
    return {
        "requested_timestep_seconds": float(requested_dt),
        "accepted": bool(result.accepted),
        "order": int(result.order),
        "dt_used": float(result.dt_used),
        "dt_next": float(result.dt_next),
        "accepted_bdf2_steps": int(result.accepted_bdf2_steps),
        "attempts": attempt_rows,
    }


def _campaign_work(records: tuple[dict, ...]) -> dict[str, int]:
    totals = {
        "implicit_solves": 0,
        "function_evaluations": 0,
        "jacobian_evaluations": 0,
        "newton_iterations": 0,
    }
    for record in records:
        for attempt in record["attempts"]:
            for name in totals:
                totals[name] += int(attempt["work"][name])
    return totals


def _evolve(
    context,
    initial: CausalFiveFieldAdaptiveBDF2Restart,
    target_elapsed_time: float,
    config: CausalFiveFieldAdaptiveBDF2Config,
    *,
    split_path: Path | None = None,
    split_relative_step: int | None = None,
) -> CampaignResult:
    target = float(target_elapsed_time)
    tolerance = max(
        1.0e-20,
        TARGET_TIME_RELATIVE_TOLERANCE * target,
    )
    state = initial
    cumulative = _ledger_from_restart(initial)
    records: list[dict] = []
    relative_accepted = 0
    message = "adaptive BDF2 campaign reached target"
    passed = True

    while state.elapsed_time < target - tolerance:
        remaining = target - state.elapsed_time
        requested = min(state.dt_next, remaining)
        result = advance_causal_five_field_adaptive_bdf2(
            context,
            state.state_vector,
            state.history,
            state.older_physical_increment,
            state.older_timestep_seconds,
            requested,
            config,
            next_order=state.next_order,
            accepted_bdf2_steps=state.accepted_bdf2_steps,
        )
        records.append(_record(result, requested))
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
                state = _restart_with(
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
        cumulative = _add_ledgers(
            cumulative,
            result.physical_interval_ledger,
        )
        elapsed = state.elapsed_time + result.dt_used
        relative_accepted += 1
        state = _restart_with(
            state,
            state_vector=result.state_vector,
            history=result.history,
            older_physical_increment=(
                result.older_physical_increment
            ),
            older_timestep_seconds=(
                result.older_timestep_seconds
            ),
            cumulative=cumulative,
            elapsed_time=elapsed,
            dt_next=result.dt_next,
            next_order=2,
            accepted_steps=state.accepted_steps + 1,
            accepted_bdf2_steps=result.accepted_bdf2_steps,
            rejected_attempts=rejected,
            audit_count=audits,
        )
        print(
            json.dumps(
                {
                    "accepted_step": relative_accepted,
                    "order": result.order,
                    "dt_used": result.dt_used,
                    "dt_next": result.dt_next,
                    "elapsed_time": elapsed,
                    "remaining": max(0.0, target - elapsed),
                    "audits": audits,
                    "rejected_attempts": rejected,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (
            split_path is not None
            and split_relative_step is not None
            and relative_accepted == split_relative_step
        ):
            save_causal_five_field_adaptive_bdf2_restart(
                split_path,
                context,
                state,
            )
            restored = load_causal_five_field_adaptive_bdf2_restart(
                split_path,
                context,
            )
            if not causal_five_field_adaptive_bdf2_restarts_equal(
                state,
                restored,
            ):
                raise RuntimeError(
                    "adaptive BDF2 split restart is not bitwise"
                )

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
    return CampaignResult(
        restart=state,
        records=tuple(records),
        passed=passed,
        message=message,
    )


def _load_initial(context) -> tuple[
    CausalFiveFieldAdaptiveRestart,
    CausalFiveFieldAdaptiveBDF2Restart,
    str,
]:
    initial_sha = _sha256(INITIAL_CHECKPOINT)
    source = load_causal_five_field_adaptive_restart(
        INITIAL_CHECKPOINT,
        context,
    )
    if not (
        source.provenance.get("work_package") == "WP10c5q"
        and source.provenance.get("n_cells") == N_CELLS
        and "exact circularized regression stream"
        in str(source.provenance.get("source", ""))
        and audit_causal_five_field_state_gates(
            context,
            source.state_vector,
        )["passed"]
    ):
        raise RuntimeError("WP10c5q restart prerequisite failed")
    history = causal_five_field_bdf_history(
        context,
        source.state_vector,
        source.previous_physical_increment,
        source.previous_dt,
    )
    zero = _zero_ledger()
    adaptive = CausalFiveFieldAdaptiveBDF2Restart(
        state_vector=source.state_vector,
        history=history,
        older_physical_increment=source.previous_physical_increment,
        older_timestep_seconds=source.previous_dt,
        cumulative_actual_conserved_storage=(
            zero.actual_conserved_storage
        ),
        cumulative_actual_vertical_storage=(
            zero.actual_vertical_storage
        ),
        cumulative_boundary_transport=(
            zero.trapezoidal_boundary_transport
        ),
        cumulative_endogenous_source=(
            zero.trapezoidal_endogenous_source
        ),
        cumulative_stream_source=(
            zero.exact_prescribed_stream_source
        ),
        cumulative_closure_defect=zero.closure_defect,
        elapsed_time=source.elapsed_time,
        dt_next=INITIAL_TIMESTEP_SECONDS,
        next_order=1,
        accepted_steps=source.accepted_steps,
        accepted_bdf2_steps=0,
        rejected_attempts=source.rejected_attempts,
        audit_count=0,
        provenance={
            "work_package": "WP10c7c",
            "role": "adaptive_bdf2_campaign",
            "base_commit": BASE_COMMIT,
            "n_cells": N_CELLS,
            "source": "exact circularized regression stream",
            "initial_checkpoint_sha256": initial_sha,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
        },
    )
    return source, adaptive, initial_sha


def _load_reference(
    context,
    source: CausalFiveFieldAdaptiveRestart,
    initial_sha: str,
    subdivisions: int,
) -> CausalFiveFieldAdaptiveRestart:
    path = _reference_path(subdivisions)
    restart = load_causal_five_field_adaptive_restart(path, context)
    provenance = restart.provenance
    expected_elapsed = source.elapsed_time + TARGET_DURATION_SECONDS
    tolerance = max(
        1.0e-20,
        TARGET_TIME_RELATIVE_TOLERANCE * expected_elapsed,
    )
    summary = provenance.get("reference_summary")
    if not (
        provenance.get("work_package") == "WP10c6e"
        and provenance.get("role")
        == "direct_fixed_backward_euler_reference"
        and provenance.get("base_commit") == REFERENCE_BASE_COMMIT
        and provenance.get("n_cells") == N_CELLS
        and provenance.get("subdivisions") == subdivisions
        and provenance.get("initial_checkpoint_sha256") == initial_sha
        and isinstance(summary, dict)
        and summary.get("passed", False)
        and abs(restart.elapsed_time - expected_elapsed) <= tolerance
    ):
        raise RuntimeError(
            f"S{subdivisions} reference provenance failed"
        )
    return restart


def _load_fixed_bdf(context, initial_sha: str):
    restart = load_causal_five_field_bdf_restart(
        FIXED_BDF_CHECKPOINT,
        context,
    )
    provenance = restart.provenance
    summary = provenance.get("result_summary")
    if not (
        provenance.get("work_package") == "WP10c7b"
        and provenance.get("role") == "fixed_bdf2_reference"
        and provenance.get("base_commit") == FIXED_BDF_BASE_COMMIT
        and provenance.get("subdivisions") == 64
        and provenance.get("initial_checkpoint_sha256") == initial_sha
        and isinstance(summary, dict)
        and summary.get("passed", False)
    ):
        raise RuntimeError("fixed S64 BDF2 provenance failed")
    return restart, summary


def _endpoint_errors(context, baseline, left, right):
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


def _restart_state_equal(left, right) -> bool:
    arrays = (
        (left.state_vector, right.state_vector),
        (
            left.history.previous_physical_increment,
            right.history.previous_physical_increment,
        ),
        (
            left.history.previous_vertical_killing_increment,
            right.history.previous_vertical_killing_increment,
        ),
        (
            left.older_physical_increment,
            right.older_physical_increment,
        ),
        (
            left.cumulative_actual_conserved_storage,
            right.cumulative_actual_conserved_storage,
        ),
        (
            left.cumulative_actual_vertical_storage,
            right.cumulative_actual_vertical_storage,
        ),
        (
            left.cumulative_boundary_transport,
            right.cumulative_boundary_transport,
        ),
        (
            left.cumulative_endogenous_source,
            right.cumulative_endogenous_source,
        ),
        (
            left.cumulative_stream_source,
            right.cumulative_stream_source,
        ),
        (
            left.cumulative_closure_defect,
            right.cumulative_closure_defect,
        ),
    )
    return bool(
        all(np.array_equal(one, two) for one, two in arrays)
        and left.history.previous_timestep_seconds
        == right.history.previous_timestep_seconds
        and left.older_timestep_seconds == right.older_timestep_seconds
        and left.elapsed_time == right.elapsed_time
        and left.dt_next == right.dt_next
        and left.next_order == right.next_order
        and left.accepted_steps == right.accepted_steps
        and left.accepted_bdf2_steps == right.accepted_bdf2_steps
        and left.rejected_attempts == right.rejected_attempts
        and left.audit_count == right.audit_count
    )


def _write(path: Path, output: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    context = make_causal_five_field_regression_context(N_CELLS)
    source, initial, initial_sha = _load_initial(context)
    references = {
        subdivisions: _load_reference(
            context,
            source,
            initial_sha,
            subdivisions,
        )
        for subdivisions in REFERENCE_SUBDIVISIONS
    }
    fixed_s64, fixed_summary = _load_fixed_bdf(
        context,
        initial_sha,
    )
    config = _controller_config(context)
    target_elapsed = source.elapsed_time + TARGET_DURATION_SECONDS
    uninterrupted = _evolve(
        context,
        initial,
        target_elapsed,
        config,
        split_path=SPLIT_CHECKPOINT,
        split_relative_step=SPLIT_ACCEPTED_STEP,
    )
    if not SPLIT_CHECKPOINT.exists():
        raise RuntimeError("adaptive BDF2 split checkpoint was not written")
    split = load_causal_five_field_adaptive_bdf2_restart(
        SPLIT_CHECKPOINT,
        context,
    )
    split_roundtrip = causal_five_field_adaptive_bdf2_restarts_equal(
        split,
        load_causal_five_field_adaptive_bdf2_restart(
            SPLIT_CHECKPOINT,
            context,
        ),
    )
    replay = _evolve(
        context,
        split,
        target_elapsed,
        config,
    )
    replay_bitwise = bool(
        replay.passed
        and _restart_state_equal(
            uninterrupted.restart,
            replay.restart,
        )
    )
    save_causal_five_field_adaptive_bdf2_restart(
        FINAL_CHECKPOINT,
        context,
        uninterrupted.restart,
    )
    final_restored = load_causal_five_field_adaptive_bdf2_restart(
        FINAL_CHECKPOINT,
        context,
    )
    final_roundtrip = causal_five_field_adaptive_bdf2_restarts_equal(
        uninterrupted.restart,
        final_restored,
    )

    reference_uncertainty = _endpoint_errors(
        context,
        source.state_vector,
        references[256].state_vector,
        references[512].state_vector,
    )
    adaptive_to_reference = _endpoint_errors(
        context,
        source.state_vector,
        uninterrupted.restart.state_vector,
        references[512].state_vector,
    )
    endpoint_audit = (
        audit_causal_five_field_endpoint_with_reference_uncertainty(
            adaptive_to_reference,
            reference_uncertainty,
            dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1),
        )
    )
    adaptive_to_fixed_s64 = _endpoint_errors(
        context,
        source.state_vector,
        uninterrupted.restart.state_vector,
        fixed_s64.state_vector,
    )
    ledger = _ledger_from_restart(uninterrupted.restart)
    ledger_json = _ledger_json(ledger)
    work = _campaign_work(uninterrupted.records)
    fixed_jacobians = int(
        fixed_summary["work"]["jacobian_evaluations"]
    )
    work_fraction = (
        work["jacobian_evaluations"] / fixed_jacobians
    )
    audits = [
        attempt["independent_audit"]
        for record in uninterrupted.records
        for attempt in record["attempts"]
        if attempt["independent_audit"] is not None
    ]
    all_audits_passed = bool(
        audits and all(audit["passed"] for audit in audits)
    )
    local_estimator_passed = all(
        (
            attempt["local_gate_audit"] is None
            or attempt["local_gate_audit"]["passed"]
        )
        for record in uninterrupted.records
        for attempt in record["attempts"]
        if attempt["accepted"]
    )
    work_passed = bool(
        work_fraction
        <= MAXIMUM_ADAPTIVE_TO_FIXED_S64_JACOBIAN_FRACTION
    )
    physical_ledger_passed = bool(
        ledger_json["maximum_relative_defect"]
        <= MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
    )
    passed = bool(
        uninterrupted.passed
        and replay.passed
        and split_roundtrip
        and replay_bitwise
        and final_roundtrip
        and endpoint_audit["passed"]
        and all_audits_passed
        and local_estimator_passed
        and physical_ledger_passed
        and work_passed
    )
    output = {
        "work_package": "WP10c7c",
        "scope": (
            "bounded adaptive N16 variable-step BDF2 certification"
        ),
        "base_commit": BASE_COMMIT,
        "construction": {
            "n_cells": N_CELLS,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "initial_timestep_seconds": INITIAL_TIMESTEP_SECONDS,
            "maximum_timestep_seconds": (
                SHARED_PASSING_CEILING_SECONDS
            ),
            "local_error_gate_fraction": LOCAL_ERROR_GATE_FRACTION,
            "predictor_error_scale": PREDICTOR_ERROR_SCALE,
            "audit_interval": AUDIT_INTERVAL,
            "predictor": "quadratic three-state history",
            "ordinary_step": "one implicit BDF2 corrector",
            "timestep_factor": "clip(0.8 * E^(-1/3), 0.5, 2.0)",
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "no_n32_n64_n128_production_run": True,
            "no_long_timescale_or_physics_run": True,
        },
        "initial_checkpoint": {
            "path": str(INITIAL_CHECKPOINT.relative_to(ROOT)),
            "sha256": initial_sha,
            "elapsed_time_seconds": source.elapsed_time,
        },
        "reference_checkpoints": {
            str(subdivisions): {
                "path": str(
                    _reference_path(subdivisions).relative_to(ROOT)
                ),
                "sha256": _sha256(_reference_path(subdivisions)),
            }
            for subdivisions in REFERENCE_SUBDIVISIONS
        },
        "fixed_s64_checkpoint": {
            "path": str(FIXED_BDF_CHECKPOINT.relative_to(ROOT)),
            "sha256": _sha256(FIXED_BDF_CHECKPOINT),
            "work": fixed_summary["work"],
        },
        "campaign": {
            "passed": uninterrupted.passed,
            "message": uninterrupted.message,
            "accepted_steps": (
                uninterrupted.restart.accepted_steps
                - source.accepted_steps
            ),
            "accepted_bdf2_steps": (
                uninterrupted.restart.accepted_bdf2_steps
            ),
            "rejected_attempts": (
                uninterrupted.restart.rejected_attempts
                - source.rejected_attempts
            ),
            "audit_count": uninterrupted.restart.audit_count,
            "minimum_dt_used": min(
                record["dt_used"]
                for record in uninterrupted.records
                if record["accepted"]
            ),
            "maximum_dt_used": max(
                record["dt_used"]
                for record in uninterrupted.records
                if record["accepted"]
            ),
            "records": list(uninterrupted.records),
            "work": work,
        },
        "endpoint": {
            "adaptive_to_s512": adaptive_to_reference,
            "reference_uncertainty_s256_to_s512": (
                reference_uncertainty
            ),
            "combined_audit": endpoint_audit,
            "adaptive_to_fixed_s64": adaptive_to_fixed_s64,
        },
        "physical_ledger": {
            **ledger_json,
            "gate": MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT,
            "passed": physical_ledger_passed,
        },
        "independent_audits": {
            "count": len(audits),
            "rows": audits,
            "passed": all_audits_passed,
        },
        "local_estimator_passed": local_estimator_passed,
        "restart_replay": {
            "split_checkpoint": {
                "path": str(SPLIT_CHECKPOINT.relative_to(ROOT)),
                "sha256": _sha256(SPLIT_CHECKPOINT),
            },
            "final_checkpoint": {
                "path": str(FINAL_CHECKPOINT.relative_to(ROOT)),
                "sha256": _sha256(FINAL_CHECKPOINT),
            },
            "split_roundtrip_bitwise": split_roundtrip,
            "endpoint_replay_bitwise": replay_bitwise,
            "final_roundtrip_bitwise": final_roundtrip,
            "passed": bool(
                split_roundtrip and replay_bitwise and final_roundtrip
            ),
        },
        "work_audit": {
            "adaptive": work,
            "fixed_s64": fixed_summary["work"],
            "adaptive_to_fixed_s64_jacobian_fraction": work_fraction,
            "maximum_fraction": (
                MAXIMUM_ADAPTIVE_TO_FIXED_S64_JACOBIAN_FRACTION
            ),
            "passed": work_passed,
        },
        "authorization": {
            "wp10c7d_matched_n32_bdf2_authorized": passed,
            "n64_n128_production_authorized": False,
            "long_evolution_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_hot_state_or_cycle_certified": False,
        },
        "decision": (
            "authorize_wp10c7d_matched_n32_bdf2"
            if passed
            else "stop_wp10c7c_gate_failed"
        ),
        "passed": passed,
    }
    _write(DEFAULT_OUTPUT, output)
    print(
        json.dumps(
            {
                "output": str(DEFAULT_OUTPUT),
                "decision": output["decision"],
                "passed": output["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
