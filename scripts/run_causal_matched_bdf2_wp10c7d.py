"""Certify matched N32 adaptive BDF2 against an N32 time reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION,
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    CausalFiveFieldAdaptiveBDF2Config,
    CausalFiveFieldAdaptiveBDF2Restart,
    CausalFiveFieldAdaptiveRestart,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldBDFRestart,
    audit_causal_five_field_endpoint_with_reference_uncertainty,
    audit_causal_five_field_state_gates,
    causal_five_field_adaptive_bdf2_restarts_equal,
    causal_five_field_bdf_history,
    causal_five_field_bdf_physical_ledger_from_restart,
    causal_five_field_bdf_physical_ledger_relative_defects,
    causal_five_field_bdf_restarts_equal,
    causal_five_field_bdf_zero_physical_ledger,
    causal_five_field_h_over_r_profile,
    causal_five_field_temporal_error_ratio,
    compare_causal_five_field_endpoint_vectors,
    evolve_causal_five_field_adaptive_bdf2_campaign,
    evolve_causal_five_field_fixed_bdf2,
    load_causal_five_field_adaptive_bdf2_restart,
    load_causal_five_field_adaptive_restart,
    load_causal_five_field_bdf_restart,
    make_causal_five_field_regression_context,
    save_causal_five_field_adaptive_bdf2_restart,
    save_causal_five_field_bdf_restart,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "1ab69be5573e569234a08b6b0ccf106a5852475c"
N_CELLS = 32
INITIAL_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c5k"
)
INITIAL_CHECKPOINT = (
    INITIAL_DIRECTORY / "causal_wp10c5q_N032_final.npz"
)
N16_INITIAL_CHECKPOINT = (
    INITIAL_DIRECTORY / "causal_wp10c5q_N016_final.npz"
)
N16_FINAL_CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c7c"
    / "causal_wp10c7c_N016_final.npz"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c7d"
)
FINAL_CHECKPOINT = (
    OUTPUT_CHECKPOINT_DIRECTORY / "causal_wp10c7d_N032_final.npz"
)
SPLIT_CHECKPOINT = (
    OUTPUT_CHECKPOINT_DIRECTORY / "causal_wp10c7d_N032_split.npz"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_matched_bdf2_wp10c7d_N032.json"
)
SHARED_PASSING_CEILING_SECONDS = 1.9218219974586337e-3
TARGET_DURATION_SECONDS = 8.0 * SHARED_PASSING_CEILING_SECONDS
INITIAL_TIMESTEP_SECONDS = TARGET_DURATION_SECONDS / 64.0
FIXED_SUBDIVISIONS = (16, 32, 64)
COOLING_INNER_CUTOFF_RG = 6.0
TARGET_TIME_RELATIVE_TOLERANCE = 5.0e-14
MINIMUM_OBSERVED_ORDER = 1.7
MAXIMUM_OBSERVED_ORDER = 2.3
ORDER_FLOOR_GATE_FRACTION = 1.0e-3
MAXIMUM_REFERENCE_UNCERTAINTY_FRACTION = 0.25
MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT = 1.0e-3
PHYSICAL_LEDGER_ORDER_FLOOR = 1.0e-10
MAXIMUM_ADAPTIVE_TO_FIXED_S64_JACOBIAN_FRACTION = 0.75
LOCAL_ERROR_GATE_FRACTION = 0.25
PREDICTOR_ERROR_SCALE = 0.2
AUDIT_INTERVAL = 4
SPLIT_ACCEPTED_STEP = 3
SPATIAL_RESPONSE_GATE = 5.0e-3


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-reference",
        action="store_true",
        help="Recompute the N32 fixed BDF2 reference ladder.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_path(subdivisions: int) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / f"causal_wp10c7d_N032_bdf2_S{subdivisions:04d}.npz"
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


def _ledger_json(ledger) -> dict:
    relative = causal_five_field_bdf_physical_ledger_relative_defects(
        ledger
    )
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
        "component_relative_defects": [
            float(value) for value in relative
        ],
        "maximum_relative_defect": float(np.max(relative)),
    }


def _fixed_summary(result) -> dict:
    return {
        "subdivisions": int(result.subdivisions),
        "timestep_seconds": float(result.timestep_seconds),
        "completed_steps": int(result.completed_steps),
        "bdf1_steps": int(result.bdf1_steps),
        "bdf2_steps": int(result.bdf2_steps),
        "state_gates": result.state_gates,
        "maximum_scaled_residual": float(
            result.maximum_scaled_residual
        ),
        "maximum_scaled_algebraic_residual": float(
            result.maximum_scaled_algebraic_residual
        ),
        "maximum_scaled_primitive_change": float(
            result.maximum_scaled_primitive_change
        ),
        "maximum_scaled_total_change": float(
            result.maximum_scaled_total_change
        ),
        "maximum_discrete_ledger_relative_defect": float(
            result.maximum_discrete_ledger_relative_defect
        ),
        "cumulative_physical_ledger": _ledger_json(
            result.cumulative_physical_ledger
        ),
        "maximum_linear_residual": float(
            result.maximum_linear_residual
        ),
        "maximum_newton_iterations": int(
            result.maximum_newton_iterations
        ),
        "work": {
            "implicit_solves": int(result.completed_steps),
            "function_evaluations": int(result.function_evaluations),
            "jacobian_evaluations": int(result.jacobian_evaluations),
            "newton_iterations": int(result.newton_iterations),
        },
        "passed": bool(result.passed),
        "message": str(result.message),
    }


def _load_initial(context) -> tuple[CausalFiveFieldAdaptiveRestart, str]:
    checksum = _sha256(INITIAL_CHECKPOINT)
    initial = load_causal_five_field_adaptive_restart(
        INITIAL_CHECKPOINT,
        context,
    )
    if not (
        initial.provenance.get("work_package") == "WP10c5q"
        and initial.provenance.get("n_cells") == N_CELLS
        and "exact circularized regression stream"
        in str(initial.provenance.get("source", ""))
        and audit_causal_five_field_state_gates(
            context,
            initial.state_vector,
        )["passed"]
    ):
        raise RuntimeError("N32 WP10c5q restart prerequisite failed")
    return initial, checksum


def _make_fixed_restart(
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha: str,
    result,
) -> CausalFiveFieldBDFRestart:
    if result.history is None:
        raise RuntimeError("fixed N32 BDF2 result lacks history")
    return CausalFiveFieldBDFRestart(
        state_vector=np.asarray(result.state_vector, dtype=float),
        history=result.history,
        elapsed_time=initial.elapsed_time + TARGET_DURATION_SECONDS,
        dt_next=result.timestep_seconds,
        next_order=2,
        accepted_steps=initial.accepted_steps + result.completed_steps,
        rejected_attempts=initial.rejected_attempts,
        provenance={
            "work_package": "WP10c7d",
            "role": "n32_fixed_bdf2_temporal_reference",
            "base_commit": BASE_COMMIT,
            "n_cells": N_CELLS,
            "subdivisions": int(result.subdivisions),
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "source": "exact circularized regression stream",
            "initial_checkpoint_sha256": initial_sha,
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "temporal_method": (
                "one BDF1 startup step then fixed equal-step BDF2"
            ),
            "result_summary": _fixed_summary(result),
        },
    )


def _load_fixed_result(
    context,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha: str,
    subdivisions: int,
) -> dict:
    path = _fixed_path(subdivisions)
    restart = load_causal_five_field_bdf_restart(path, context)
    provenance = restart.provenance
    summary = provenance.get("result_summary")
    timestep = TARGET_DURATION_SECONDS / subdivisions
    tolerance = max(
        1.0e-20,
        TARGET_TIME_RELATIVE_TOLERANCE
        * (initial.elapsed_time + TARGET_DURATION_SECONDS),
    )
    if not (
        provenance.get("work_package") == "WP10c7d"
        and provenance.get("role")
        == "n32_fixed_bdf2_temporal_reference"
        and provenance.get("base_commit") == BASE_COMMIT
        and provenance.get("n_cells") == N_CELLS
        and provenance.get("subdivisions") == subdivisions
        and provenance.get("initial_checkpoint_sha256") == initial_sha
        and provenance.get("observable_schema")
        == CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
        and isinstance(summary, dict)
        and summary.get("passed", False)
        and abs(
            restart.elapsed_time
            - initial.elapsed_time
            - TARGET_DURATION_SECONDS
        )
        <= tolerance
        and restart.history.previous_timestep_seconds == timestep
        and restart.dt_next == timestep
        and audit_causal_five_field_state_gates(
            context,
            restart.state_vector,
        )["passed"]
    ):
        raise RuntimeError(
            f"N32 fixed S{subdivisions} provenance failed"
        )
    return {
        "restart": restart,
        "summary": summary,
        "checkpoint": {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "roundtrip_bitwise": True,
            "reused": True,
        },
    }


def _fixed_progress(subdivisions: int):
    interval = max(2, subdivisions // 4)

    def progress(completed, total, _state, _history) -> None:
        if completed % interval == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "mode": f"n32_fixed_bdf2_s{subdivisions}",
                        "completed_steps": completed,
                        "total_steps": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return progress


def _run_or_load_fixed(
    context,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha: str,
    subdivisions: int,
    *,
    force: bool,
) -> dict:
    path = _fixed_path(subdivisions)
    if path.exists() and not force:
        return _load_fixed_result(
            context,
            initial,
            initial_sha,
            subdivisions,
        )
    result = evolve_causal_five_field_fixed_bdf2(
        context,
        initial.state_vector,
        initial.previous_physical_increment,
        initial.previous_dt,
        TARGET_DURATION_SECONDS,
        subdivisions,
        _step_config(),
        progress=_fixed_progress(subdivisions),
    )
    summary = _fixed_summary(result)
    if not result.passed:
        return {
            "restart": None,
            "summary": summary,
            "checkpoint": None,
        }
    restart = _make_fixed_restart(initial, initial_sha, result)
    save_causal_five_field_bdf_restart(path, context, restart)
    restored = load_causal_five_field_bdf_restart(path, context)
    if not causal_five_field_bdf_restarts_equal(restart, restored):
        raise RuntimeError("fixed N32 BDF2 restart is not bitwise")
    loaded = _load_fixed_result(
        context,
        initial,
        initial_sha,
        subdivisions,
    )
    loaded["checkpoint"]["reused"] = False
    return loaded


def _fixed_reference_audit(context, initial, runs: dict[int, dict]) -> dict:
    adjacent = {
        "16_to_32": _endpoint_errors(
            context,
            initial.state_vector,
            runs[16]["restart"].state_vector,
            runs[32]["restart"].state_vector,
        ),
        "32_to_64": _endpoint_errors(
            context,
            initial.state_vector,
            runs[32]["restart"].state_vector,
            runs[64]["restart"].state_vector,
        ),
    }
    rows = {}
    for name, gate in CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1.items():
        coarse = float(adjacent["16_to_32"][name])
        fine = float(adjacent["32_to_64"][name])
        order = (
            float(np.log2(coarse / fine))
            if coarse > 0.0 and fine > 0.0
            else None
        )
        below_floor = fine <= ORDER_FLOOR_GATE_FRACTION * gate
        order_passed = bool(
            below_floor
            or (
                order is not None
                and MINIMUM_OBSERVED_ORDER
                <= order
                <= MAXIMUM_OBSERVED_ORDER
            )
        )
        rows[name] = {
            "16_to_32_error": coarse,
            "32_to_64_error": fine,
            "fine_observed_order": order,
            "below_order_floor": below_floor,
            "order_passed": order_passed,
        }
    uncertainty_audit = causal_five_field_temporal_error_ratio(
        {
            name: float(adjacent["32_to_64"][name])
            for name in CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
        },
        {
            name: (
                MAXIMUM_REFERENCE_UNCERTAINTY_FRACTION * float(gate)
            )
            for name, gate in (
                CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1.items()
            )
        },
    )
    order_passed = all(row["order_passed"] for row in rows.values())

    ledger_rows = {}
    ledger_order_passed = True
    for component in range(5):
        values = [
            float(
                runs[subdivisions]["summary"][
                    "cumulative_physical_ledger"
                ]["component_relative_defects"][component]
            )
            for subdivisions in FIXED_SUBDIVISIONS
        ]
        order = (
            float(np.log2(values[1] / values[2]))
            if values[1] > 0.0 and values[2] > 0.0
            else None
        )
        below_floor = values[2] <= PHYSICAL_LEDGER_ORDER_FLOOR
        component_passed = bool(
            below_floor
            or (order is not None and order >= MINIMUM_OBSERVED_ORDER)
        )
        ledger_order_passed = ledger_order_passed and component_passed
        ledger_rows[str(component)] = {
            "relative_defects": values,
            "fine_observed_order": order,
            "below_order_floor": below_floor,
            "passed": component_passed,
        }
    fine_ledger = float(
        runs[64]["summary"]["cumulative_physical_ledger"][
            "maximum_relative_defect"
        ]
    )
    physical_passed = bool(
        ledger_order_passed
        and fine_ledger <= MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
    )
    passed = bool(
        all(runs[n]["summary"]["passed"] for n in FIXED_SUBDIVISIONS)
        and order_passed
        and uncertainty_audit["passed"]
        and physical_passed
    )
    return {
        "adjacent_errors": adjacent,
        "observable_order": {
            "minimum": MINIMUM_OBSERVED_ORDER,
            "maximum": MAXIMUM_OBSERVED_ORDER,
            "rows": rows,
            "passed": order_passed,
        },
        "reference_uncertainty": {
            "fraction": MAXIMUM_REFERENCE_UNCERTAINTY_FRACTION,
            "audit": uncertainty_audit,
        },
        "physical_ledger": {
            "rows": ledger_rows,
            "fine_maximum_relative_defect": fine_ledger,
            "maximum_relative_defect": (
                MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
            ),
            "passed": physical_passed,
        },
        "passed": passed,
    }


def _initial_adaptive_restart(
    context,
    source: CausalFiveFieldAdaptiveRestart,
    source_sha: str,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    history = causal_five_field_bdf_history(
        context,
        source.state_vector,
        source.previous_physical_increment,
        source.previous_dt,
    )
    zero = causal_five_field_bdf_zero_physical_ledger()
    return CausalFiveFieldAdaptiveBDF2Restart(
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
        cumulative_stream_source=zero.exact_prescribed_stream_source,
        cumulative_closure_defect=zero.closure_defect,
        elapsed_time=source.elapsed_time,
        dt_next=INITIAL_TIMESTEP_SECONDS,
        next_order=1,
        accepted_steps=source.accepted_steps,
        accepted_bdf2_steps=0,
        rejected_attempts=source.rejected_attempts,
        audit_count=0,
        provenance={
            "work_package": "WP10c7d",
            "role": "matched_n32_adaptive_bdf2",
            "base_commit": BASE_COMMIT,
            "n_cells": N_CELLS,
            "source": "exact circularized regression stream",
            "initial_checkpoint_sha256": source_sha,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
        },
    )


def _attempt_work(attempt) -> dict[str, int]:
    solved = [attempt.step]
    if attempt.independent_audit is not None:
        solved.append(attempt.independent_audit.first_half_step)
        if attempt.independent_audit.second_half_step is not None:
            solved.append(attempt.independent_audit.second_half_step)
    return {
        "implicit_solves": len(solved),
        "function_evaluations": sum(
            int(step.function_evaluations) for step in solved
        ),
        "jacobian_evaluations": sum(
            int(step.jacobian_evaluations) for step in solved
        ),
        "newton_iterations": sum(
            int(step.iterations) for step in solved
        ),
    }


def _campaign_work(campaign) -> dict[str, int]:
    total = {
        "implicit_solves": 0,
        "function_evaluations": 0,
        "jacobian_evaluations": 0,
        "newton_iterations": 0,
    }
    for result in campaign.steps:
        for attempt in result.attempts:
            work = _attempt_work(attempt)
            for name in total:
                total[name] += work[name]
    return total


def _attempt_json(attempt) -> dict:
    return {
        "timestep_seconds": float(attempt.timestep_seconds),
        "order": int(attempt.order),
        "accepted": bool(attempt.accepted),
        "failure_class": str(attempt.failure_class),
        "proposed_factor": float(attempt.proposed_factor),
        "maximum_scaled_residual": float(
            attempt.step.maximum_scaled_residual
        ),
        "maximum_discrete_ledger_relative_defect": float(
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
        "work": _attempt_work(attempt),
    }


def _campaign_json(campaign, initial) -> dict:
    records = []
    for result in campaign.steps:
        records.append(
            {
                "accepted": bool(result.accepted),
                "order": int(result.order),
                "dt_used": float(result.dt_used),
                "dt_next": float(result.dt_next),
                "accepted_bdf2_steps": int(
                    result.accepted_bdf2_steps
                ),
                "attempts": [
                    _attempt_json(attempt)
                    for attempt in result.attempts
                ],
            }
        )
    accepted = [row for row in records if row["accepted"]]
    return {
        "passed": bool(campaign.passed),
        "message": str(campaign.message),
        "accepted_steps": int(
            campaign.restart.accepted_steps - initial.accepted_steps
        ),
        "accepted_bdf2_steps": int(
            campaign.restart.accepted_bdf2_steps
        ),
        "rejected_attempts": int(
            campaign.restart.rejected_attempts
            - initial.rejected_attempts
        ),
        "audit_count": int(campaign.restart.audit_count),
        "minimum_dt_used": min(row["dt_used"] for row in accepted),
        "maximum_dt_used": max(row["dt_used"] for row in accepted),
        "records": records,
        "work": _campaign_work(campaign),
    }


def _adaptive_progress(split_holder, context):
    def progress(relative_step, restart, result) -> None:
        print(
            json.dumps(
                {
                    "mode": "n32_adaptive_bdf2",
                    "accepted_step": relative_step,
                    "order": result.order,
                    "dt_used": result.dt_used,
                    "dt_next": result.dt_next,
                    "elapsed_time": restart.elapsed_time,
                    "audits": restart.audit_count,
                    "rejected_attempts": restart.rejected_attempts,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if relative_step == SPLIT_ACCEPTED_STEP:
            save_causal_five_field_adaptive_bdf2_restart(
                SPLIT_CHECKPOINT,
                context,
                restart,
            )
            restored = load_causal_five_field_adaptive_bdf2_restart(
                SPLIT_CHECKPOINT,
                context,
            )
            if not causal_five_field_adaptive_bdf2_restarts_equal(
                restart,
                restored,
            ):
                raise RuntimeError(
                    "N32 adaptive split restart is not bitwise"
                )
            split_holder["restart"] = restored

    return progress


def _reconstructed_log_h_over_r(context, vector, sample_log_radius):
    centers = np.log(np.asarray(context.grid.centers, dtype=float))
    values = np.log(
        causal_five_field_h_over_r_profile(context, vector)
    )
    sample = np.asarray(sample_log_radius, dtype=float)
    reconstructed = np.interp(sample, centers, values)
    left = sample < centers[0]
    right = sample > centers[-1]
    reconstructed[left] = values[0] + (
        (values[1] - values[0]) / (centers[1] - centers[0])
    ) * (sample[left] - centers[0])
    reconstructed[right] = values[-1] + (
        (values[-1] - values[-2])
        / (centers[-1] - centers[-2])
    ) * (sample[right] - centers[-1])
    return reconstructed


def _cross_mesh_response(n32_context, n32_initial, n32_final) -> dict:
    n16_context = make_causal_five_field_regression_context(16)
    n16_initial = load_causal_five_field_adaptive_restart(
        N16_INITIAL_CHECKPOINT,
        n16_context,
    )
    n16_final = load_causal_five_field_adaptive_bdf2_restart(
        N16_FINAL_CHECKPOINT,
        n16_context,
    )
    if not np.array_equal(
        n16_context.grid.edges[[0, -1]],
        n32_context.grid.edges[[0, -1]],
    ):
        raise RuntimeError("N16/N32 physical domains differ")
    common_time = bool(
        n16_initial.elapsed_time == n32_initial.elapsed_time
        and n16_final.elapsed_time == n32_final.elapsed_time
    )
    sample = np.linspace(
        np.log(float(n32_context.grid.edges[0])),
        np.log(float(n32_context.grid.edges[-1])),
        129,
    )
    n16_response = (
        _reconstructed_log_h_over_r(
            n16_context,
            n16_final.state_vector,
            sample,
        )
        - _reconstructed_log_h_over_r(
            n16_context,
            n16_initial.state_vector,
            sample,
        )
    )
    n32_response = (
        _reconstructed_log_h_over_r(
            n32_context,
            n32_final.state_vector,
            sample,
        )
        - _reconstructed_log_h_over_r(
            n32_context,
            n32_initial.state_vector,
            sample,
        )
    )
    difference = n16_response - n32_response
    maximum = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference**2)))
    return {
        "method": (
            "baseline-subtracted Delta log(H/R), log-linear "
            "cell-center reconstruction with one-cell edge "
            "extrapolation, shared 129-point log-radius grid"
        ),
        "exact_common_time": common_time,
        "elapsed_time_seconds": float(n32_final.elapsed_time),
        "sample_radius_rg": [
            float(
                np.exp(value)
                / n32_context.grid.gravitational_radius
            )
            for value in sample
        ],
        "n16_delta_log_h_over_r": [
            float(value) for value in n16_response
        ],
        "n32_delta_log_h_over_r": [
            float(value) for value in n32_response
        ],
        "maximum_response_difference": maximum,
        "rms_response_difference": rms,
        "gate": SPATIAL_RESPONSE_GATE,
        "passed": bool(common_time and maximum <= SPATIAL_RESPONSE_GATE),
    }


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
    args = _arguments()
    output_path = _absolute(args.output)
    context = make_causal_five_field_regression_context(N_CELLS)
    initial, initial_sha = _load_initial(context)
    fixed_runs = {
        subdivisions: _run_or_load_fixed(
            context,
            initial,
            initial_sha,
            subdivisions,
            force=args.force_reference,
        )
        for subdivisions in FIXED_SUBDIVISIONS
    }
    fixed_complete = bool(
        all(
            fixed_runs[n]["restart"] is not None
            and fixed_runs[n]["summary"]["passed"]
            for n in FIXED_SUBDIVISIONS
        )
    )
    fixed_audit = (
        _fixed_reference_audit(context, initial, fixed_runs)
        if fixed_complete
        else {"passed": False}
    )
    base_output = {
        "work_package": "WP10c7d",
        "scope": (
            "matched N32 fixed-reference and adaptive BDF2 "
            "temporal confirmation"
        ),
        "base_commit": BASE_COMMIT,
        "construction": {
            "n_cells": N_CELLS,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "fixed_subdivisions": list(FIXED_SUBDIVISIONS),
            "selected_reference_subdivisions": 64,
            "reference_uncertainty_pair": [32, 64],
            "maximum_reference_uncertainty_fraction": (
                MAXIMUM_REFERENCE_UNCERTAINTY_FRACTION
            ),
            "initial_timestep_seconds": INITIAL_TIMESTEP_SECONDS,
            "maximum_timestep_seconds": (
                SHARED_PASSING_CEILING_SECONDS
            ),
            "controller_unchanged_from_wp10c7c": True,
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "no_n64_n128_or_long_physics_run": True,
        },
        "initial_checkpoint": {
            "path": str(INITIAL_CHECKPOINT.relative_to(ROOT)),
            "sha256": initial_sha,
            "elapsed_time_seconds": initial.elapsed_time,
        },
        "fixed_references": {
            str(n): {
                "summary": fixed_runs[n]["summary"],
                "checkpoint": fixed_runs[n]["checkpoint"],
            }
            for n in FIXED_SUBDIVISIONS
        },
        "fixed_reference_audit": fixed_audit,
    }
    if not fixed_audit["passed"]:
        output = {
            **base_output,
            "temporal_controller_passed": False,
            "spatial_qualification": None,
            "authorization": {
                "wp10c7d_matched_n32_temporal_certified": False,
                "n64_n128_production_authorized": False,
                "long_evolution_certified": False,
                "tide_authorized": False,
                "wind_authorized": False,
            },
            "decision": "stop_n32_fixed_reference_gate_failed",
            "passed": False,
        }
        _write(output_path, output)
        print(json.dumps({"decision": output["decision"], "passed": False}))
        return

    adaptive_initial = _initial_adaptive_restart(
        context,
        initial,
        initial_sha,
    )
    target_elapsed = initial.elapsed_time + TARGET_DURATION_SECONDS
    split_holder: dict[str, CausalFiveFieldAdaptiveBDF2Restart] = {}
    campaign = evolve_causal_five_field_adaptive_bdf2_campaign(
        context,
        adaptive_initial,
        target_elapsed,
        _controller_config(context),
        target_time_relative_tolerance=TARGET_TIME_RELATIVE_TOLERANCE,
        progress=_adaptive_progress(split_holder, context),
    )
    if "restart" not in split_holder:
        raise RuntimeError("N32 adaptive split checkpoint was not written")
    split = split_holder["restart"]
    replay = evolve_causal_five_field_adaptive_bdf2_campaign(
        context,
        split,
        target_elapsed,
        _controller_config(context),
        target_time_relative_tolerance=TARGET_TIME_RELATIVE_TOLERANCE,
    )
    replay_bitwise = bool(
        replay.passed
        and causal_five_field_adaptive_bdf2_restarts_equal(
            campaign.restart,
            replay.restart,
        )
    )
    save_causal_five_field_adaptive_bdf2_restart(
        FINAL_CHECKPOINT,
        context,
        campaign.restart,
    )
    final_restored = load_causal_five_field_adaptive_bdf2_restart(
        FINAL_CHECKPOINT,
        context,
    )
    final_roundtrip = causal_five_field_adaptive_bdf2_restarts_equal(
        campaign.restart,
        final_restored,
    )
    split_roundtrip = causal_five_field_adaptive_bdf2_restarts_equal(
        split,
        load_causal_five_field_adaptive_bdf2_restart(
            SPLIT_CHECKPOINT,
            context,
        ),
    )

    reference_uncertainty = fixed_audit["adjacent_errors"][
        "32_to_64"
    ]
    adaptive_to_reference = _endpoint_errors(
        context,
        initial.state_vector,
        campaign.restart.state_vector,
        fixed_runs[64]["restart"].state_vector,
    )
    endpoint_audit = (
        audit_causal_five_field_endpoint_with_reference_uncertainty(
            adaptive_to_reference,
            reference_uncertainty,
            dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1),
        )
    )
    campaign_data = _campaign_json(campaign, initial)
    work = campaign_data["work"]
    fixed_work = fixed_runs[64]["summary"]["work"]
    work_fraction = (
        work["jacobian_evaluations"]
        / fixed_work["jacobian_evaluations"]
    )
    work_passed = bool(
        work_fraction
        <= MAXIMUM_ADAPTIVE_TO_FIXED_S64_JACOBIAN_FRACTION
    )
    ledger = causal_five_field_bdf_physical_ledger_from_restart(
        campaign.restart
    )
    ledger_data = _ledger_json(ledger)
    ledger_passed = bool(
        ledger_data["maximum_relative_defect"]
        <= MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
    )
    attempts = [
        attempt
        for result in campaign.steps
        for attempt in result.attempts
    ]
    audits = [
        attempt.independent_audit
        for attempt in attempts
        if attempt.independent_audit is not None
    ]
    audits_passed = bool(
        audits and all(audit.passed for audit in audits)
    )
    estimator_passed = all(
        attempt.local_gate_audit is None
        or attempt.local_gate_audit["passed"]
        for attempt in attempts
        if attempt.accepted
    )
    temporal_passed = bool(
        campaign.passed
        and replay.passed
        and endpoint_audit["passed"]
        and ledger_passed
        and audits_passed
        and estimator_passed
        and split_roundtrip
        and replay_bitwise
        and final_roundtrip
        and work_passed
    )
    spatial = _cross_mesh_response(
        context,
        initial,
        campaign.restart,
    )
    decision = (
        (
            "certify_matched_n32_bdf2_and_spatial_response"
            if spatial["passed"]
            else (
                "certify_matched_n32_bdf2_stop_on_spatial_response"
            )
        )
        if temporal_passed
        else "stop_wp10c7d_temporal_gate_failed"
    )
    output = {
        **base_output,
        "adaptive_campaign": campaign_data,
        "adaptive_endpoint": {
            "adaptive_to_fixed_s64": adaptive_to_reference,
            "reference_uncertainty_s32_to_s64": (
                reference_uncertainty
            ),
            "combined_audit": endpoint_audit,
        },
        "adaptive_physical_ledger": {
            **ledger_data,
            "gate": MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT,
            "passed": ledger_passed,
        },
        "independent_audits": {
            "count": len(audits),
            "maximum_normalized_error": max(
                float(
                    audit.temporal_gate_audit[
                        "maximum_normalized_error"
                    ]
                )
                for audit in audits
                if audit.temporal_gate_audit is not None
            ),
            "passed": audits_passed,
        },
        "local_estimator_passed": estimator_passed,
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
            "fixed_s64": fixed_work,
            "adaptive_to_fixed_s64_jacobian_fraction": work_fraction,
            "maximum_fraction": (
                MAXIMUM_ADAPTIVE_TO_FIXED_S64_JACOBIAN_FRACTION
            ),
            "passed": work_passed,
        },
        "spatial_qualification": spatial,
        "temporal_controller_passed": temporal_passed,
        "authorization": {
            "wp10c7d_matched_n32_temporal_certified": (
                temporal_passed
            ),
            "bounded_n16_n32_spatial_response_certified": bool(
                spatial["passed"]
            ),
            "n64_n128_production_authorized": False,
            "long_evolution_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_hot_state_or_cycle_certified": False,
        },
        "decision": decision,
        "passed": temporal_passed,
    }
    _write(output_path, output)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "decision": decision,
                "temporal_controller_passed": temporal_passed,
                "spatial_qualification_passed": spatial["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
