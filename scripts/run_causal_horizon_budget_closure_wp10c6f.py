"""Close backward Euler against the certified N16 temporal reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION,
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    CausalFiveFieldAdaptiveRestart,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldDAEContext,
    CausalFiveFieldTemporalCampaignState,
    CausalFiveFieldTemporalControllerConfig,
    audit_causal_five_field_endpoint_with_reference_uncertainty,
    audit_causal_five_field_reference_convergence,
    audit_causal_five_field_state_gates,
    causal_five_field_temporal_campaign_states_equal,
    compare_causal_five_field_endpoint_vectors,
    evolve_causal_five_field_horizon_budget,
    load_causal_five_field_adaptive_restart,
    make_causal_five_field_regression_context,
    save_causal_five_field_adaptive_restart,
)


ROOT = Path(__file__).resolve().parents[1]
N_CELLS = 16
BASE_COMMIT = "7d9fe8d300222b7a387f7d05344b3cd739230742"
REFERENCE_BASE_COMMIT = (
    "e4c32bcf04cf1ebe62c46261d41a84bc9377bebb"
)
INITIAL_CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c5k"
    / "causal_wp10c5q_N016_final.npz"
)
REFERENCE_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c6e"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c6f"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables"
    / "causal_horizon_budget_closure_wp10c6f_N016.json"
)
SHARED_PASSING_CEILING_SECONDS = 1.9218219974586337e-3
FIRST_FAILING_TIMESTEP_SECONDS = 3.8436439949172674e-3
INITIAL_TIMESTEP_SECONDS = 0.5 * SHARED_PASSING_CEILING_SECONDS
TARGET_DURATION_SECONDS = 8.0 * SHARED_PASSING_CEILING_SECONDS
REFERENCE_SUBDIVISIONS = (128, 256, 512)
MAXIMUM_REFERENCE_UNCERTAINTY_FRACTION = 0.25
MINIMUM_OBSERVED_ORDER = 0.75
ORDER_FLOOR_FRACTION = 1.0e-3
COOLING_INNER_CUTOFF_RG = 6.0
FINITE_DIFFERENCE_STEP = 2.0e-6
RESTART_SPLIT_ACCEPTED_STEPS = 3
TARGET_TIME_RELATIVE_TOLERANCE = 5.0e-14
MAXIMUM_CONTROLLER_TO_REFERENCE_JACOBIAN_FRACTION = 0.5


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Validate the saved references without running the controller.",
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


def _reference_path(subdivisions: int) -> Path:
    return (
        REFERENCE_CHECKPOINT_DIRECTORY
        / f"causal_wp10c6e_N016_reference_S{subdivisions:04d}.npz"
    )


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


def _attempt_summary(attempt) -> dict:
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
        "second_half_contract": (
            None
            if attempt.second_half_contract is None
            else _contract_summary(attempt.second_half_contract)
        ),
    }


def _attempt_work(attempts) -> dict[str, int]:
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


def _campaign_work(campaign) -> dict[str, int]:
    names = (
        "implicit_solves",
        "function_evaluations",
        "jacobian_evaluations",
        "newton_iterations",
    )
    rows = [
        _attempt_work(record.result.attempts)
        for record in campaign.records
    ]
    return {
        name: int(sum(row[name] for row in rows))
        for name in names
    }


def _campaign_summary(campaign) -> dict:
    state = campaign.final_state
    return {
        "elapsed_time_seconds": float(state.elapsed_time),
        "target_elapsed_time_seconds": float(
            campaign.target_elapsed_time
        ),
        "dt_next_seconds": float(state.dt_next),
        "previous_dt_seconds": float(state.previous_dt),
        "extension_accepted_steps": int(state.accepted_steps),
        "extension_rejected_trials": int(state.rejected_trials),
        "cumulative_budget_fraction": float(
            state.cumulative_budget_fraction
        ),
        "target_reached": bool(campaign.target_reached),
        "budget_sum_passed": bool(campaign.budget_sum_passed),
        "state_adapter_requested": bool(
            campaign.state_adapter_requested
        ),
        "state_adapter_performed": bool(
            campaign.state_adapter_performed
        ),
        "state_gates": campaign.state_gates,
        "records": [
            {
                "elapsed_time_before_seconds": float(
                    record.elapsed_time_before
                ),
                "requested_timestep_seconds": float(
                    record.requested_timestep
                ),
                "accepted": bool(record.result.accepted),
                "dt_used_seconds": float(record.result.dt_used),
                "dt_next_seconds": float(record.result.dt_next),
                "normalized_error": record.result.normalized_error,
                "rejected_trials": int(record.rejected_trials),
                "message": str(record.result.message),
                "attempts": [
                    _attempt_summary(attempt)
                    for attempt in record.result.attempts
                ],
                "work": _attempt_work(record.result.attempts),
            }
            for record in campaign.records
        ],
        "work": _campaign_work(campaign),
        "message": str(campaign.message),
        "passed": bool(campaign.passed),
    }


def _campaign_initial_state(
    restart: CausalFiveFieldAdaptiveRestart,
) -> CausalFiveFieldTemporalCampaignState:
    return CausalFiveFieldTemporalCampaignState(
        state_vector=np.asarray(restart.state_vector, dtype=float),
        previous_physical_increment=np.asarray(
            restart.previous_physical_increment,
            dtype=float,
        ),
        elapsed_time=float(restart.elapsed_time),
        dt_next=INITIAL_TIMESTEP_SECONDS,
        previous_dt=float(restart.previous_dt),
    )


def _make_restart(
    initial: CausalFiveFieldAdaptiveRestart,
    state: CausalFiveFieldTemporalCampaignState,
    *,
    role: str,
    reference_hashes: dict[str, str],
) -> CausalFiveFieldAdaptiveRestart:
    return CausalFiveFieldAdaptiveRestart(
        state_vector=np.asarray(state.state_vector, dtype=float),
        previous_physical_increment=np.asarray(
            state.previous_physical_increment,
            dtype=float,
        ),
        elapsed_time=float(state.elapsed_time),
        dt_next=float(state.dt_next),
        previous_dt=float(state.previous_dt),
        accepted_steps=initial.accepted_steps + state.accepted_steps,
        rejected_attempts=(
            initial.rejected_attempts + state.rejected_trials
        ),
        provenance={
            "work_package": "WP10c6f",
            "role": role,
            "base_commit": BASE_COMMIT,
            "n_cells": N_CELLS,
            "source": "exact circularized regression stream",
            "output_horizon_seconds": TARGET_DURATION_SECONDS,
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "controller": (
                "dt-weighted global observable budget with accepted "
                "two-half-step state"
            ),
            "reference_checkpoint_sha256": reference_hashes,
            "campaign_state": {
                "accepted_steps": state.accepted_steps,
                "rejected_trials": state.rejected_trials,
                "cumulative_budget_fraction": (
                    state.cumulative_budget_fraction
                ),
            },
        },
    )


def _load_reference(
    context: CausalFiveFieldDAEContext,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha256: str,
    subdivisions: int,
) -> dict:
    path = _reference_path(subdivisions)
    if not path.exists():
        raise FileNotFoundError(path)
    restart = load_causal_five_field_adaptive_restart(path, context)
    provenance = restart.provenance
    summary = provenance.get("reference_summary")
    expected_elapsed = initial.elapsed_time + TARGET_DURATION_SECONDS
    elapsed_tolerance = max(
        1.0e-20,
        TARGET_TIME_RELATIVE_TOLERANCE * expected_elapsed,
    )
    expected_dt = TARGET_DURATION_SECONDS / subdivisions
    state_gates = audit_causal_five_field_state_gates(
        context,
        restart.state_vector,
    )
    valid = bool(
        provenance.get("work_package") == "WP10c6e"
        and provenance.get("role")
        == "direct_fixed_backward_euler_reference"
        and provenance.get("base_commit") == REFERENCE_BASE_COMMIT
        and provenance.get("n_cells") == N_CELLS
        and provenance.get("subdivisions") == subdivisions
        and provenance.get("target_duration_seconds")
        == TARGET_DURATION_SECONDS
        and provenance.get("initial_checkpoint_sha256")
        == initial_sha256
        and provenance.get("observable_schema")
        == CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
        and isinstance(summary, dict)
        and summary.get("passed", False)
        and abs(restart.elapsed_time - expected_elapsed)
        <= elapsed_tolerance
        and restart.dt_next == expected_dt
        and restart.previous_dt == expected_dt
        and state_gates["passed"]
    )
    if not valid:
        raise RuntimeError(
            f"S{subdivisions} reference checkpoint failed provenance"
        )
    return {
        "restart": restart,
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "summary": summary,
        "state_gates": state_gates,
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


def _progress(mode: str):
    accepted = 0

    def progress(record) -> None:
        nonlocal accepted
        if record.result.accepted:
            accepted += 1
        print(
            json.dumps(
                {
                    "mode": mode,
                    "accepted_step": (
                        accepted if record.result.accepted else None
                    ),
                    "requested_dt": record.requested_timestep,
                    "dt_used": record.result.dt_used,
                    "dt_next": record.result.dt_next,
                    "normalized_error": (
                        record.result.normalized_error
                    ),
                    "rejected_trials": record.rejected_trials,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return progress


def _write_output(path: Path, output: dict) -> None:
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
    if not INITIAL_CHECKPOINT.exists():
        raise FileNotFoundError(INITIAL_CHECKPOINT)
    initial_sha256 = _sha256(INITIAL_CHECKPOINT)
    initial = load_causal_five_field_adaptive_restart(
        INITIAL_CHECKPOINT,
        context,
    )
    initial_state_gates = audit_causal_five_field_state_gates(
        context,
        initial.state_vector,
    )
    initial_provenance_passed = bool(
        initial.provenance.get("work_package") == "WP10c5q"
        and initial.provenance.get("n_cells") == N_CELLS
        and "exact circularized regression stream"
        in str(initial.provenance.get("source", ""))
        and initial_state_gates["passed"]
    )
    if not initial_provenance_passed:
        raise RuntimeError("WP10c5q restart prerequisite failed")

    references = {
        subdivisions: _load_reference(
            context,
            initial,
            initial_sha256,
            subdivisions,
        )
        for subdivisions in REFERENCE_SUBDIVISIONS
    }
    reference_hashes = {
        str(subdivisions): reference["sha256"]
        for subdivisions, reference in references.items()
    }
    coarse_errors = _endpoint_errors(
        context,
        initial.state_vector,
        references[128]["restart"].state_vector,
        references[256]["restart"].state_vector,
    )
    fine_errors = _endpoint_errors(
        context,
        initial.state_vector,
        references[256]["restart"].state_vector,
        references[512]["restart"].state_vector,
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
    reference_passed = bool(reference_convergence["passed"])
    output = {
        "work_package": "WP10c6f",
        "scope": (
            "single N16 horizon-budget backward-Euler closure against "
            "the certified S512 reference"
        ),
        "base_commit": BASE_COMMIT,
        "construction": {
            "n_cells": N_CELLS,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "initial_timestep_seconds": INITIAL_TIMESTEP_SECONDS,
            "reference_subdivisions": list(
                REFERENCE_SUBDIVISIONS
            ),
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "temporal_accuracy_gates": dict(
                CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
            ),
            "accepted_state": "two_half_step_state",
            "local_budget_rule": (
                "global_gate * dt / output_horizon"
            ),
            "factor_rule": (
                "clip(0.8 / normalized_budget_error, 0.25, 2.0)"
            ),
            "maximum_controller_to_reference_jacobian_fraction": (
                MAXIMUM_CONTROLLER_TO_REFERENCE_JACOBIAN_FRACTION
            ),
            "reference_only_requested": bool(args.reference_only),
            "no_n32_n64_n128_production_run": True,
            "no_bdf2_disk_run": True,
            "no_long_timescale_or_physics_run": True,
        },
        "initial_checkpoint": {
            "path": str(INITIAL_CHECKPOINT.relative_to(ROOT)),
            "sha256": initial_sha256,
            "elapsed_time_seconds": initial.elapsed_time,
            "provenance": initial.provenance,
            "provenance_passed": initial_provenance_passed,
            "state_gates": initial_state_gates,
        },
        "references": {
            str(subdivisions): {
                key: value
                for key, value in reference.items()
                if key != "restart"
            }
            for subdivisions, reference in references.items()
        },
        "reference_errors": {
            "128_to_256": coarse_errors,
            "256_to_512": fine_errors,
        },
        "reference_convergence": reference_convergence,
        "reference_gate_passed": reference_passed,
        "horizon_budget": None,
        "restart_replay": None,
        "endpoint_audit": None,
        "work_audit": None,
        "final_checkpoint": None,
        "authorization": {
            "wp10c7a_bdf_method_work_authorized": False,
            "bdf2_disk_certification_authorized": False,
            "n32_controller_run_authorized": False,
            "n64_n128_production_run_authorized": False,
            "long_evolution_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_hot_state_or_cycle_certified": False,
        },
        "decision": (
            "reference_gate_passed_controller_not_requested"
            if args.reference_only and reference_passed
            else (
                "stop_reference_gate_failed"
                if not reference_passed
                else "run_single_n16_horizon_budget_closure"
            )
        ),
        "passed": False,
    }
    if args.reference_only or not reference_passed:
        _write_output(output_path, output)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "decision": output["decision"],
                    "passed": output["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    controller_config = _controller_config(context)
    target_elapsed = initial.elapsed_time + TARGET_DURATION_SECONDS
    uninterrupted = evolve_causal_five_field_horizon_budget(
        context,
        _campaign_initial_state(initial),
        target_elapsed,
        controller_config,
        target_time_relative_tolerance=(
            TARGET_TIME_RELATIVE_TOLERANCE
        ),
        progress=_progress("horizon_budget"),
    )

    split_audit: dict[str, object] = {
        "path": None,
        "sha256": None,
        "roundtrip_bitwise": False,
    }

    def restart_adapter(
        state: CausalFiveFieldTemporalCampaignState,
    ) -> CausalFiveFieldTemporalCampaignState:
        restart = _make_restart(
            initial,
            state,
            role="interrupted_horizon_budget_replay",
            reference_hashes=reference_hashes,
        )
        split_path = (
            OUTPUT_CHECKPOINT_DIRECTORY
            / "causal_wp10c6f_N016_split.npz"
        )
        save_causal_five_field_adaptive_restart(
            split_path,
            context,
            restart,
        )
        restored = load_causal_five_field_adaptive_restart(
            split_path,
            context,
        )
        roundtrip = _restart_is_bitwise(restart, restored)
        split_audit.update(
            {
                "path": str(split_path.relative_to(ROOT)),
                "sha256": _sha256(split_path),
                "roundtrip_bitwise": roundtrip,
            }
        )
        if not roundtrip:
            raise RuntimeError(
                "split horizon-budget restart is not bitwise"
            )
        campaign_state = restored.provenance.get("campaign_state")
        if not isinstance(campaign_state, dict):
            raise RuntimeError(
                "split restart omitted temporal campaign metadata"
            )
        return CausalFiveFieldTemporalCampaignState(
            state_vector=np.asarray(
                restored.state_vector,
                dtype=float,
            ),
            previous_physical_increment=np.asarray(
                restored.previous_physical_increment,
                dtype=float,
            ),
            elapsed_time=restored.elapsed_time,
            dt_next=restored.dt_next,
            previous_dt=restored.previous_dt,
            accepted_steps=int(campaign_state["accepted_steps"]),
            rejected_trials=int(campaign_state["rejected_trials"]),
            cumulative_budget_fraction=float(
                campaign_state["cumulative_budget_fraction"]
            ),
        )

    replay = evolve_causal_five_field_horizon_budget(
        context,
        _campaign_initial_state(initial),
        target_elapsed,
        controller_config,
        target_time_relative_tolerance=(
            TARGET_TIME_RELATIVE_TOLERANCE
        ),
        state_adapter_after_accepted_steps=(
            RESTART_SPLIT_ACCEPTED_STEPS
        ),
        state_adapter=restart_adapter,
        progress=_progress("budget_restart_replay"),
    )
    replay_final_bitwise = (
        causal_five_field_temporal_campaign_states_equal(
            uninterrupted.final_state,
            replay.final_state,
        )
    )
    controller_errors = _endpoint_errors(
        context,
        initial.state_vector,
        uninterrupted.final_state.state_vector,
        references[512]["restart"].state_vector,
    )
    endpoint_audit = (
        audit_causal_five_field_endpoint_with_reference_uncertainty(
            controller_errors,
            fine_errors,
            dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1),
        )
    )
    reference_work = references[512]["summary"]["work"]
    controller_work = _campaign_work(uninterrupted)
    jacobian_fraction = float(
        controller_work["jacobian_evaluations"]
        / reference_work["jacobian_evaluations"]
    )
    efficiency_passed = bool(
        jacobian_fraction
        <= MAXIMUM_CONTROLLER_TO_REFERENCE_JACOBIAN_FRACTION
    )
    work_audit = {
        "horizon_budget": controller_work,
        "reference_512": reference_work,
        "controller_to_reference": {
            name: float(
                controller_work[name] / reference_work[name]
            )
            for name in reference_work
        },
        "reference_over_controller": {
            name: float(
                reference_work[name] / controller_work[name]
            )
            for name in reference_work
        },
        "controller_per_simulated_second": {
            name: float(
                controller_work[name] / TARGET_DURATION_SECONDS
            )
            for name in controller_work
        },
        "maximum_jacobian_fraction": (
            MAXIMUM_CONTROLLER_TO_REFERENCE_JACOBIAN_FRACTION
        ),
        "jacobian_fraction": jacobian_fraction,
        "efficiency_passed": efficiency_passed,
    }
    final_restart = _make_restart(
        initial,
        uninterrupted.final_state,
        role="bounded_horizon_budget_closure",
        reference_hashes=reference_hashes,
    )
    final_path = (
        OUTPUT_CHECKPOINT_DIRECTORY
        / "causal_wp10c6f_N016_final.npz"
    )
    save_causal_five_field_adaptive_restart(
        final_path,
        context,
        final_restart,
    )
    restored_final = load_causal_five_field_adaptive_restart(
        final_path,
        context,
    )
    final_roundtrip_bitwise = _restart_is_bitwise(
        final_restart,
        restored_final,
    )
    accuracy_passed = bool(
        uninterrupted.passed
        and replay.passed
        and split_audit["roundtrip_bitwise"]
        and replay_final_bitwise
        and final_roundtrip_bitwise
        and endpoint_audit["passed"]
    )
    output["horizon_budget"] = _campaign_summary(uninterrupted)
    output["restart_replay"] = {
        **_campaign_summary(replay),
        "split_checkpoint": split_audit,
        "final_bitwise_equal_to_uninterrupted": (
            replay_final_bitwise
        ),
    }
    output["endpoint_audit"] = {
        "errors_against_512_step_reference": controller_errors,
        "reference_uncertainty_256_to_512": fine_errors,
        "combined_audit": endpoint_audit,
    }
    output["work_audit"] = work_audit
    output["final_checkpoint"] = {
        "path": str(final_path.relative_to(ROOT)),
        "sha256": _sha256(final_path),
        "roundtrip_bitwise": final_roundtrip_bitwise,
    }
    output["authorization"][
        "wp10c7a_bdf_method_work_authorized"
    ] = True
    output["decision"] = (
        "be_accuracy_failed_frozen_start_wp10c7a"
        if not accuracy_passed
        else (
            "be_accurate_inefficient_frozen_start_wp10c7a"
            if not efficiency_passed
            else "be_accurate_frozen_start_wp10c7a"
        )
    )
    output["passed"] = accuracy_passed
    _write_output(output_path, output)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "decision": output["decision"],
                "accuracy_passed": accuracy_passed,
                "efficiency_passed": efficiency_passed,
                "passed": output["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
