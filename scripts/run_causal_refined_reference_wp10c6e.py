"""Persist and certify the N16 128/256/512 temporal reference."""

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
    audit_causal_five_field_reference_convergence,
    audit_causal_five_field_state_gates,
    compare_causal_five_field_endpoint_vectors,
    evolve_causal_five_field_fixed_reference,
    load_causal_five_field_adaptive_restart,
    make_causal_five_field_regression_context,
    save_causal_five_field_adaptive_restart,
)


ROOT = Path(__file__).resolve().parents[1]
N_CELLS = 16
BASE_COMMIT = "e4c32bcf04cf1ebe62c46261d41a84bc9377bebb"
INITIAL_CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c5k"
    / "causal_wp10c5q_N016_final.npz"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c6e"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables"
    / "causal_refined_reference_wp10c6e_N016.json"
)
SHARED_PASSING_CEILING_SECONDS = 1.9218219974586337e-3
TARGET_DURATION_SECONDS = 8.0 * SHARED_PASSING_CEILING_SECONDS
REFERENCE_SUBDIVISIONS = (128, 256, 512)
MAXIMUM_REFERENCE_UNCERTAINTY_FRACTION = 0.25
MINIMUM_OBSERVED_ORDER = 0.75
ORDER_FLOOR_FRACTION = 1.0e-3
COOLING_INNER_CUTOFF_RG = 6.0
FINITE_DIFFERENCE_STEP = 2.0e-6
TARGET_TIME_RELATIVE_TOLERANCE = 5.0e-14


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subdivisions",
        type=int,
        action="append",
        choices=REFERENCE_SUBDIVISIONS,
        default=None,
        help="Repeat to run a subset; default selects the full ladder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected references instead of reusing checkpoints.",
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
        OUTPUT_CHECKPOINT_DIRECTORY
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
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        maximum_newton_iterations=12,
    ).validated()


def _result_summary(result) -> dict:
    return {
        "subdivisions": int(result.subdivisions),
        "timestep_seconds": float(result.timestep_seconds),
        "completed_steps": int(result.completed_steps),
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
        "maximum_conservation_telescoping_relative_defect": float(
            result.maximum_conservation_telescoping_relative_defect
        ),
        "maximum_physical_ledger_relative_defect": float(
            result.maximum_physical_ledger_relative_defect
        ),
        "maximum_linear_residual": float(
            result.maximum_linear_residual
        ),
        "maximum_newton_iterations": int(
            result.maximum_newton_iterations
        ),
        "work": {
            "implicit_solves": int(result.completed_steps),
            "function_evaluations": int(
                result.function_evaluations
            ),
            "jacobian_evaluations": int(
                result.jacobian_evaluations
            ),
            "newton_iterations": int(result.newton_iterations),
        },
        "passed": bool(result.passed),
        "message": str(result.message),
    }


def _progress_callback(subdivisions: int):
    interval = max(16, subdivisions // 4)

    def progress(completed: int, total: int) -> None:
        if completed % interval == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "mode": f"reference_{subdivisions}",
                        "completed_steps": completed,
                        "total_steps": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return progress


def _save_reference(
    context: CausalFiveFieldDAEContext,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha256: str,
    result,
) -> dict:
    summary = _result_summary(result)
    if not summary["passed"]:
        raise RuntimeError("cannot persist a failed temporal reference")
    target_elapsed = initial.elapsed_time + TARGET_DURATION_SECONDS
    restart = CausalFiveFieldAdaptiveRestart(
        state_vector=np.asarray(result.state_vector, dtype=float),
        previous_physical_increment=np.asarray(
            result.previous_physical_increment,
            dtype=float,
        ),
        elapsed_time=target_elapsed,
        dt_next=result.timestep_seconds,
        previous_dt=result.timestep_seconds,
        accepted_steps=initial.accepted_steps + result.completed_steps,
        rejected_attempts=initial.rejected_attempts,
        provenance={
            "work_package": "WP10c6e",
            "role": "direct_fixed_backward_euler_reference",
            "base_commit": BASE_COMMIT,
            "n_cells": N_CELLS,
            "subdivisions": result.subdivisions,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "source": "exact circularized regression stream",
            "initial_checkpoint_sha256": initial_sha256,
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "reference_summary": summary,
        },
    )
    path = _reference_path(result.subdivisions)
    save_causal_five_field_adaptive_restart(path, context, restart)
    restored = load_causal_five_field_adaptive_restart(path, context)
    roundtrip = _restart_is_bitwise(restart, restored)
    if not roundtrip:
        raise RuntimeError("reference checkpoint round trip is not bitwise")
    return {
        "state_vector": np.asarray(restored.state_vector, dtype=float),
        "previous_increment": np.asarray(
            restored.previous_physical_increment,
            dtype=float,
        ),
        "summary": summary,
        "checkpoint": {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "roundtrip_bitwise": roundtrip,
            "reused": False,
        },
    }


def _load_reference(
    context: CausalFiveFieldDAEContext,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha256: str,
    subdivisions: int,
) -> dict:
    path = _reference_path(subdivisions)
    restart = load_causal_five_field_adaptive_restart(path, context)
    provenance = restart.provenance
    expected_elapsed = initial.elapsed_time + TARGET_DURATION_SECONDS
    elapsed_tolerance = max(
        1.0e-20,
        TARGET_TIME_RELATIVE_TOLERANCE * expected_elapsed,
    )
    valid = bool(
        provenance.get("work_package") == "WP10c6e"
        and provenance.get("role")
        == "direct_fixed_backward_euler_reference"
        and provenance.get("base_commit") == BASE_COMMIT
        and provenance.get("n_cells") == N_CELLS
        and provenance.get("subdivisions") == subdivisions
        and provenance.get("target_duration_seconds")
        == TARGET_DURATION_SECONDS
        and provenance.get("initial_checkpoint_sha256")
        == initial_sha256
        and provenance.get("observable_schema")
        == CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
        and abs(restart.elapsed_time - expected_elapsed)
        <= elapsed_tolerance
        and restart.previous_dt
        == TARGET_DURATION_SECONDS / subdivisions
    )
    summary = provenance.get("reference_summary")
    state_gates = audit_causal_five_field_state_gates(
        context,
        restart.state_vector,
    )
    if (
        not valid
        or not isinstance(summary, dict)
        or not summary.get("passed", False)
        or not state_gates["passed"]
    ):
        raise RuntimeError(
            f"existing S{subdivisions} reference provenance failed"
        )
    return {
        "state_vector": np.asarray(restart.state_vector, dtype=float),
        "previous_increment": np.asarray(
            restart.previous_physical_increment,
            dtype=float,
        ),
        "summary": summary,
        "checkpoint": {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "roundtrip_bitwise": True,
            "reused": True,
        },
    }


def _run_or_load_reference(
    context: CausalFiveFieldDAEContext,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha256: str,
    subdivisions: int,
    *,
    force: bool,
) -> dict:
    path = _reference_path(subdivisions)
    if path.exists() and not force:
        print(
            json.dumps(
                {
                    "mode": f"reference_{subdivisions}",
                    "checkpoint": str(path),
                    "reused": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return _load_reference(
            context,
            initial,
            initial_sha256,
            subdivisions,
        )
    result = evolve_causal_five_field_fixed_reference(
        context,
        initial.state_vector,
        initial.previous_physical_increment,
        initial.previous_dt,
        TARGET_DURATION_SECONDS,
        subdivisions,
        _step_config(),
        physical_ledger_tolerance=1.0e-10,
        progress=_progress_callback(subdivisions),
    )
    if not result.passed:
        return {
            "state_vector": np.asarray(result.state_vector, dtype=float),
            "previous_increment": np.asarray(
                result.previous_physical_increment,
                dtype=float,
            ),
            "summary": _result_summary(result),
            "checkpoint": None,
        }
    return _save_reference(
        context,
        initial,
        initial_sha256,
        result,
    )


def _public_reference(reference: dict) -> dict:
    return {
        "summary": reference["summary"],
        "checkpoint": reference["checkpoint"],
    }


def main() -> None:
    args = _arguments()
    selected = (
        list(REFERENCE_SUBDIVISIONS)
        if args.subdivisions is None
        else list(dict.fromkeys(args.subdivisions))
    )
    context = make_causal_five_field_regression_context(N_CELLS)
    if not INITIAL_CHECKPOINT.exists():
        raise FileNotFoundError(INITIAL_CHECKPOINT)
    initial_sha256 = _sha256(INITIAL_CHECKPOINT)
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
        subdivisions: _run_or_load_reference(
            context,
            initial,
            initial_sha256,
            subdivisions,
            force=args.force,
        )
        for subdivisions in selected
    }
    all_selected_passed = all(
        reference["summary"]["passed"]
        and reference["checkpoint"] is not None
        for reference in references.values()
    )
    complete_ladder = set(references) == set(REFERENCE_SUBDIVISIONS)
    coarse_errors = None
    fine_errors = None
    convergence = None
    if complete_ladder and all_selected_passed:
        coarse_errors = compare_causal_five_field_endpoint_vectors(
            context,
            initial.state_vector,
            references[128]["state_vector"],
            references[256]["state_vector"],
            cooling_inner_cutoff=(
                COOLING_INNER_CUTOFF_RG
                * context.grid.gravitational_radius
            ),
        )
        fine_errors = compare_causal_five_field_endpoint_vectors(
            context,
            initial.state_vector,
            references[256]["state_vector"],
            references[512]["state_vector"],
            cooling_inner_cutoff=(
                COOLING_INNER_CUTOFF_RG
                * context.grid.gravitational_radius
            ),
        )
        convergence = audit_causal_five_field_reference_convergence(
            coarse_errors,
            fine_errors,
            dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1),
            maximum_reference_uncertainty_fraction=(
                MAXIMUM_REFERENCE_UNCERTAINTY_FRACTION
            ),
            minimum_observed_order=MINIMUM_OBSERVED_ORDER,
            order_floor_fraction=ORDER_FLOOR_FRACTION,
        )
    passed = bool(
        complete_ladder
        and all_selected_passed
        and convergence is not None
        and convergence["passed"]
    )
    output = {
        "work_package": "WP10c6e",
        "scope": (
            "bounded N16 direct 128/256/512 backward-Euler "
            "reference refinement and persistence"
        ),
        "base_commit": BASE_COMMIT,
        "construction": {
            "n_cells": N_CELLS,
            "selected_subdivisions": selected,
            "required_subdivisions": list(REFERENCE_SUBDIVISIONS),
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "temporal_accuracy_gates": dict(
                CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
            ),
            "maximum_reference_uncertainty_fraction": (
                MAXIMUM_REFERENCE_UNCERTAINTY_FRACTION
            ),
            "minimum_observed_order": MINIMUM_OBSERVED_ORDER,
            "order_floor_fraction": ORDER_FLOOR_FRACTION,
            "force_recompute": bool(args.force),
            "no_adaptive_controller_run": True,
            "no_n32_n64_n128_production_run": True,
            "no_long_timescale_or_physics_run": True,
        },
        "initial_checkpoint": {
            "path": str(INITIAL_CHECKPOINT.relative_to(ROOT)),
            "sha256": initial_sha256,
            "elapsed_time_seconds": initial.elapsed_time,
            "provenance": initial.provenance,
            "provenance_passed": provenance_passed,
            "state_gates": initial_state_gates,
        },
        "references": {
            str(subdivisions): _public_reference(reference)
            for subdivisions, reference in references.items()
        },
        "complete_ladder": complete_ladder,
        "all_selected_references_passed": all_selected_passed,
        "reference_errors": {
            "128_to_256": coarse_errors,
            "256_to_512": fine_errors,
        },
        "reference_convergence": convergence,
        "authorization": {
            "n16_refined_reference_certified": passed,
            "n16_horizon_budget_closure_authorized": passed,
            "bdf2_disk_certification_authorized": False,
            "n32_controller_run_authorized": False,
            "n64_n128_production_run_authorized": False,
            "long_evolution_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_hot_state_or_cycle_certified": False,
        },
        "decision": (
            "n16_refined_reference_passed"
            if passed
            else (
                "partial_reference_ladder_persisted"
                if not complete_ladder and all_selected_passed
                else "stop_refined_reference_gate_failed"
            )
        ),
        "passed": passed,
    }
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
                "selected_subdivisions": selected,
                "complete_ladder": complete_ladder,
                "decision": output["decision"],
                "passed": output["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
