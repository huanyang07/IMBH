"""Run the fixed-step N16 BDF2 certification ladder."""

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
    CausalFiveFieldBDFRestart,
    audit_causal_five_field_endpoint_with_reference_uncertainty,
    audit_causal_five_field_state_gates,
    causal_five_field_bdf_restarts_equal,
    compare_causal_five_field_endpoint_vectors,
    evolve_causal_five_field_fixed_bdf2,
    load_causal_five_field_adaptive_restart,
    load_causal_five_field_bdf_restart,
    make_causal_five_field_regression_context,
    save_causal_five_field_bdf_restart,
)


ROOT = Path(__file__).resolve().parents[1]
N_CELLS = 16
BASE_COMMIT = "6a298c69c3c398239e6198c1f07472697929aa2e"
REFERENCE_BASE_COMMIT = "e4c32bcf04cf1ebe62c46261d41a84bc9377bebb"
INITIAL_CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c5k"
    / "causal_wp10c5q_N016_final.npz"
)
REFERENCE_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c6e"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c7b"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_fixed_bdf2_wp10c7b_N016.json"
)
SHARED_PASSING_CEILING_SECONDS = 1.9218219974586337e-3
TARGET_DURATION_SECONDS = 8.0 * SHARED_PASSING_CEILING_SECONDS
BDF_SUBDIVISIONS = (8, 16, 32, 64)
REFERENCE_SUBDIVISIONS = (256, 512)
COOLING_INNER_CUTOFF_RG = 6.0
FINITE_DIFFERENCE_STEP = 2.0e-6
TARGET_TIME_RELATIVE_TOLERANCE = 5.0e-14
MINIMUM_OBSERVED_ORDER = 1.7
MAXIMUM_OBSERVED_ORDER = 2.3
ORDER_FLOOR_GATE_FRACTION = 1.0e-3
MAXIMUM_FINE_PHYSICAL_LEDGER_RELATIVE_DEFECT = 1.0e-3
PHYSICAL_LEDGER_ORDER_FLOOR = 1.0e-10


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subdivisions",
        type=int,
        action="append",
        choices=BDF_SUBDIVISIONS,
        default=None,
        help="Repeat to run a subset; default selects the full ladder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected BDF2 trajectories.",
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


def _reference_path(subdivisions: int) -> Path:
    return (
        REFERENCE_DIRECTORY
        / f"causal_wp10c6e_N016_reference_S{subdivisions:04d}.npz"
    )


def _bdf_path(subdivisions: int) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / f"causal_wp10c7b_N016_bdf2_S{subdivisions:04d}.npz"
    )


def _split_path(subdivisions: int) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / f"causal_wp10c7b_N016_bdf2_S{subdivisions:04d}_split.npz"
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


def _endpoint_errors(
    context,
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


def _ledger_summary(ledger) -> dict[str, list[float]]:
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
    }


def _result_summary(result) -> dict:
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
        "cumulative_physical_ledger": _ledger_summary(
            result.cumulative_physical_ledger
        ),
        "cumulative_physical_ledger_relative_defect": float(
            result.cumulative_physical_ledger_relative_defect
        ),
        "cumulative_physical_component_relative_defects": [
            float(value)
            for value in (
                result.cumulative_physical_component_relative_defects
            )
        ],
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
    if not INITIAL_CHECKPOINT.exists():
        raise FileNotFoundError(INITIAL_CHECKPOINT)
    checksum = _sha256(INITIAL_CHECKPOINT)
    initial = load_causal_five_field_adaptive_restart(
        INITIAL_CHECKPOINT,
        context,
    )
    state_gates = audit_causal_five_field_state_gates(
        context,
        initial.state_vector,
    )
    valid = bool(
        initial.provenance.get("work_package") == "WP10c5q"
        and initial.provenance.get("n_cells") == N_CELLS
        and "exact circularized regression stream"
        in str(initial.provenance.get("source", ""))
        and state_gates["passed"]
    )
    if not valid:
        raise RuntimeError("WP10c5q restart prerequisite failed")
    return initial, checksum


def _load_reference(
    context,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha256: str,
    subdivisions: int,
) -> dict:
    path = _reference_path(subdivisions)
    if not path.exists():
        raise FileNotFoundError(path)
    restart = load_causal_five_field_adaptive_restart(path, context)
    provenance = restart.provenance
    expected_elapsed = initial.elapsed_time + TARGET_DURATION_SECONDS
    elapsed_tolerance = max(
        1.0e-20,
        TARGET_TIME_RELATIVE_TOLERANCE * expected_elapsed,
    )
    summary = provenance.get("reference_summary")
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
        and restart.previous_dt
        == TARGET_DURATION_SECONDS / subdivisions
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


def _make_bdf_restart(
    initial: CausalFiveFieldAdaptiveRestart,
    result,
    initial_sha256: str,
    *,
    role: str,
) -> CausalFiveFieldBDFRestart:
    if result.history is None:
        raise RuntimeError("cannot persist BDF trajectory without history")
    return CausalFiveFieldBDFRestart(
        state_vector=np.asarray(result.state_vector, dtype=float),
        history=result.history,
        elapsed_time=initial.elapsed_time + TARGET_DURATION_SECONDS,
        dt_next=result.timestep_seconds,
        next_order=2,
        accepted_steps=initial.accepted_steps + result.completed_steps,
        rejected_attempts=initial.rejected_attempts,
        provenance={
            "work_package": "WP10c7b",
            "role": role,
            "base_commit": BASE_COMMIT,
            "n_cells": N_CELLS,
            "subdivisions": result.subdivisions,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "source": "exact circularized regression stream",
            "initial_checkpoint_sha256": initial_sha256,
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "temporal_method": (
                "one BDF1 startup step then fixed equal-step BDF2"
            ),
            "result_summary": _result_summary(result),
        },
    )


def _save_bdf_result(
    context,
    initial: CausalFiveFieldAdaptiveRestart,
    result,
    initial_sha256: str,
) -> dict:
    if not result.passed:
        raise RuntimeError("cannot persist a failed BDF2 trajectory")
    restart = _make_bdf_restart(
        initial,
        result,
        initial_sha256,
        role="fixed_bdf2_reference",
    )
    path = _bdf_path(result.subdivisions)
    save_causal_five_field_bdf_restart(path, context, restart)
    restored = load_causal_five_field_bdf_restart(path, context)
    roundtrip = causal_five_field_bdf_restarts_equal(
        restart,
        restored,
    )
    if not roundtrip:
        raise RuntimeError("BDF2 checkpoint round trip is not bitwise")
    return {
        "restart": restored,
        "summary": _result_summary(result),
        "checkpoint": {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "roundtrip_bitwise": True,
            "reused": False,
        },
    }


def _load_bdf_result(
    context,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha256: str,
    subdivisions: int,
) -> dict:
    path = _bdf_path(subdivisions)
    restart = load_causal_five_field_bdf_restart(path, context)
    provenance = restart.provenance
    expected_elapsed = initial.elapsed_time + TARGET_DURATION_SECONDS
    elapsed_tolerance = max(
        1.0e-20,
        TARGET_TIME_RELATIVE_TOLERANCE * expected_elapsed,
    )
    summary = provenance.get("result_summary")
    valid = bool(
        provenance.get("work_package") == "WP10c7b"
        and provenance.get("role") == "fixed_bdf2_reference"
        and provenance.get("base_commit") == BASE_COMMIT
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
        and restart.next_order == 2
        and restart.dt_next == TARGET_DURATION_SECONDS / subdivisions
        and restart.history.previous_timestep_seconds
        == TARGET_DURATION_SECONDS / subdivisions
        and audit_causal_five_field_state_gates(
            context,
            restart.state_vector,
        )["passed"]
    )
    if not valid:
        raise RuntimeError(
            f"existing BDF2 S{subdivisions} provenance failed"
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


def _progress(subdivisions: int):
    interval = max(2, subdivisions // 4)

    def progress(completed, total, _state, _history) -> None:
        if completed % interval == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "mode": f"bdf2_{subdivisions}",
                        "completed_steps": completed,
                        "total_steps": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return progress


def _run_or_load_bdf(
    context,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha256: str,
    subdivisions: int,
    *,
    force: bool,
) -> dict:
    path = _bdf_path(subdivisions)
    if path.exists() and not force:
        print(
            json.dumps(
                {
                    "mode": f"bdf2_{subdivisions}",
                    "checkpoint": str(path),
                    "reused": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return _load_bdf_result(
            context,
            initial,
            initial_sha256,
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
        progress=_progress(subdivisions),
    )
    if not result.passed:
        return {
            "restart": None,
            "summary": _result_summary(result),
            "checkpoint": None,
        }
    return _save_bdf_result(
        context,
        initial,
        result,
        initial_sha256,
    )


def _observable_order_audit(
    adjacent_errors: dict[str, dict[str, float | list[float]]],
) -> dict:
    gates = dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1)
    rows = {}
    for name, gate in gates.items():
        coarse = float(adjacent_errors["8_to_16"][name])
        middle = float(adjacent_errors["16_to_32"][name])
        fine = float(adjacent_errors["32_to_64"][name])
        coarse_order = (
            float(np.log2(coarse / middle))
            if coarse > 0.0 and middle > 0.0
            else None
        )
        fine_order = (
            float(np.log2(middle / fine))
            if middle > 0.0 and fine > 0.0
            else None
        )
        below_floor = fine <= ORDER_FLOOR_GATE_FRACTION * gate
        order_passed = bool(
            below_floor
            or (
                fine_order is not None
                and MINIMUM_OBSERVED_ORDER
                <= fine_order
                <= MAXIMUM_OBSERVED_ORDER
            )
        )
        rows[name] = {
            "8_to_16_error": coarse,
            "16_to_32_error": middle,
            "32_to_64_error": fine,
            "coarse_observed_order": coarse_order,
            "fine_observed_order": fine_order,
            "below_order_floor": below_floor,
            "passed": order_passed,
        }
    failed = sorted(
        name for name, row in rows.items() if not row["passed"]
    )
    return {
        "minimum_observed_order": MINIMUM_OBSERVED_ORDER,
        "maximum_observed_order": MAXIMUM_OBSERVED_ORDER,
        "order_floor_gate_fraction": ORDER_FLOOR_GATE_FRACTION,
        "observables": rows,
        "failed_observables": failed,
        "passed": not failed,
    }


def _physical_ledger_audit(runs: dict[int, dict]) -> dict:
    relative = {
        subdivisions: np.asarray(
            runs[subdivisions]["summary"][
                "cumulative_physical_component_relative_defects"
            ],
            dtype=float,
        )
        for subdivisions in BDF_SUBDIVISIONS
    }
    rows = {}
    for component in range(5):
        values = [
            float(relative[subdivisions][component])
            for subdivisions in BDF_SUBDIVISIONS
        ]
        fine_order = (
            float(np.log2(values[2] / values[3]))
            if values[2] > 0.0 and values[3] > 0.0
            else None
        )
        below_floor = values[3] <= PHYSICAL_LEDGER_ORDER_FLOOR
        rows[str(component)] = {
            "relative_defects": values,
            "fine_observed_order": fine_order,
            "below_order_floor": below_floor,
            "order_passed": bool(
                below_floor
                or (
                    fine_order is not None
                    and fine_order >= MINIMUM_OBSERVED_ORDER
                )
            ),
        }
    finest_maximum = float(
        runs[64]["summary"][
            "cumulative_physical_ledger_relative_defect"
        ]
    )
    magnitude_passed = bool(
        finest_maximum
        <= MAXIMUM_FINE_PHYSICAL_LEDGER_RELATIVE_DEFECT
    )
    failed_components = sorted(
        name for name, row in rows.items() if not row["order_passed"]
    )
    return {
        "maximum_fine_relative_defect": (
            MAXIMUM_FINE_PHYSICAL_LEDGER_RELATIVE_DEFECT
        ),
        "minimum_observed_order": MINIMUM_OBSERVED_ORDER,
        "order_floor": PHYSICAL_LEDGER_ORDER_FLOOR,
        "finest_relative_defect": finest_maximum,
        "magnitude_passed": magnitude_passed,
        "components": rows,
        "failed_components": failed_components,
        "passed": bool(magnitude_passed and not failed_components),
    }


def _restart_replay_audit(
    context,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha256: str,
    subdivisions: int,
    full_run: dict,
) -> dict:
    timestep = TARGET_DURATION_SECONDS / subdivisions
    split_step = subdivisions // 2
    captured: dict[str, object] = {}

    def capture(completed, _total, state, history) -> None:
        if completed == split_step:
            captured["state"] = np.array(state, copy=True)
            captured["history"] = history

    prefix = evolve_causal_five_field_fixed_bdf2(
        context,
        initial.state_vector,
        initial.previous_physical_increment,
        initial.previous_dt,
        split_step * timestep,
        split_step,
        _step_config(),
        progress=capture,
    )
    if not prefix.passed or not captured:
        raise RuntimeError("BDF2 split prefix failed")
    restart = CausalFiveFieldBDFRestart(
        state_vector=np.asarray(captured["state"], dtype=float),
        history=captured["history"],
        elapsed_time=initial.elapsed_time + split_step * timestep,
        dt_next=timestep,
        next_order=2,
        accepted_steps=initial.accepted_steps + split_step,
        rejected_attempts=initial.rejected_attempts,
        provenance={
            "work_package": "WP10c7b",
            "role": "fixed_bdf2_split_replay",
            "base_commit": BASE_COMMIT,
            "n_cells": N_CELLS,
            "subdivisions": subdivisions,
            "split_step": split_step,
            "initial_checkpoint_sha256": initial_sha256,
        },
    )
    path = _split_path(subdivisions)
    save_causal_five_field_bdf_restart(path, context, restart)
    restored = load_causal_five_field_bdf_restart(path, context)
    roundtrip = causal_five_field_bdf_restarts_equal(
        restart,
        restored,
    )
    replay = evolve_causal_five_field_fixed_bdf2(
        context,
        restored.state_vector,
        restored.history.previous_physical_increment,
        restored.history.previous_timestep_seconds,
        (subdivisions - split_step) * timestep,
        subdivisions - split_step,
        _step_config(),
        startup_with_bdf1=False,
        initial_history=restored.history,
    )
    target = full_run["restart"]
    if target is None:
        raise RuntimeError("BDF2 replay target is unavailable")
    endpoint_bitwise = bool(
        replay.passed
        and np.array_equal(
            replay.state_vector,
            target.state_vector,
        )
        and replay.history is not None
        and np.array_equal(
            replay.history.previous_physical_increment,
            target.history.previous_physical_increment,
        )
        and np.array_equal(
            replay.history.previous_vertical_killing_increment,
            target.history.previous_vertical_killing_increment,
        )
    )
    return {
        "subdivisions": subdivisions,
        "split_step": split_step,
        "checkpoint": {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
        },
        "roundtrip_bitwise": roundtrip,
        "endpoint_replay_bitwise": endpoint_bitwise,
        "passed": bool(roundtrip and endpoint_bitwise),
    }


def _public_run(run: dict) -> dict:
    return {
        "summary": run["summary"],
        "checkpoint": run["checkpoint"],
    }


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
    selected = (
        list(BDF_SUBDIVISIONS)
        if args.subdivisions is None
        else list(dict.fromkeys(args.subdivisions))
    )
    output_path = _absolute(args.output)
    context = make_causal_five_field_regression_context(N_CELLS)
    initial, initial_sha256 = _load_initial(context)
    references = {
        subdivisions: _load_reference(
            context,
            initial,
            initial_sha256,
            subdivisions,
        )
        for subdivisions in REFERENCE_SUBDIVISIONS
    }
    reference_uncertainty = _endpoint_errors(
        context,
        initial.state_vector,
        references[256]["restart"].state_vector,
        references[512]["restart"].state_vector,
    )
    runs = {
        subdivisions: _run_or_load_bdf(
            context,
            initial,
            initial_sha256,
            subdivisions,
            force=args.force,
        )
        for subdivisions in selected
    }
    selected_passed = all(
        run["summary"]["passed"] and run["checkpoint"] is not None
        for run in runs.values()
    )
    complete_ladder = set(runs) == set(BDF_SUBDIVISIONS)
    adjacent_errors = None
    order_audit = None
    endpoint_errors = None
    endpoint_audit = None
    physical_ledger_audit = None
    restart_replay = None
    if complete_ladder and selected_passed:
        adjacent_errors = {
            "8_to_16": _endpoint_errors(
                context,
                initial.state_vector,
                runs[8]["restart"].state_vector,
                runs[16]["restart"].state_vector,
            ),
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
        order_audit = _observable_order_audit(adjacent_errors)
        endpoint_errors = _endpoint_errors(
            context,
            initial.state_vector,
            runs[64]["restart"].state_vector,
            references[512]["restart"].state_vector,
        )
        endpoint_audit = (
            audit_causal_five_field_endpoint_with_reference_uncertainty(
                endpoint_errors,
                reference_uncertainty,
                dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1),
            )
        )
        physical_ledger_audit = _physical_ledger_audit(runs)
        restart_replay = _restart_replay_audit(
            context,
            initial,
            initial_sha256,
            8,
            runs[8],
        )

    all_discrete_ledgers_passed = bool(
        selected_passed
        and all(
            run["summary"][
                "maximum_discrete_ledger_relative_defect"
            ]
            <= _step_config().conservation_tolerance
            for run in runs.values()
        )
    )
    full_passed = bool(
        complete_ladder
        and selected_passed
        and all_discrete_ledgers_passed
        and order_audit is not None
        and order_audit["passed"]
        and endpoint_audit is not None
        and endpoint_audit["passed"]
        and physical_ledger_audit is not None
        and physical_ledger_audit["passed"]
        and restart_replay is not None
        and restart_replay["passed"]
    )
    output = {
        "work_package": "WP10c7b",
        "scope": (
            "fixed-step N16 BDF2 certification against the certified "
            "S512 backward-Euler reference"
        ),
        "base_commit": BASE_COMMIT,
        "construction": {
            "n_cells": N_CELLS,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "bdf_subdivisions": list(BDF_SUBDIVISIONS),
            "selected_subdivisions": selected,
            "reference_subdivisions": list(REFERENCE_SUBDIVISIONS),
            "temporal_method": (
                "one BDF1 startup step then fixed equal-step BDF2"
            ),
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "temporal_accuracy_gates": dict(
                CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
            ),
            "minimum_observed_order": MINIMUM_OBSERVED_ORDER,
            "maximum_observed_order": MAXIMUM_OBSERVED_ORDER,
            "order_floor_gate_fraction": ORDER_FLOOR_GATE_FRACTION,
            "maximum_fine_physical_ledger_relative_defect": (
                MAXIMUM_FINE_PHYSICAL_LEDGER_RELATIVE_DEFECT
            ),
            "no_adaptivity": True,
            "no_n32_n64_n128_production_run": True,
            "no_long_timescale_or_physics_run": True,
        },
        "initial_checkpoint": {
            "path": str(INITIAL_CHECKPOINT.relative_to(ROOT)),
            "sha256": initial_sha256,
            "elapsed_time_seconds": initial.elapsed_time,
            "provenance": initial.provenance,
        },
        "references": {
            str(subdivisions): {
                key: value
                for key, value in reference.items()
                if key != "restart"
            }
            for subdivisions, reference in references.items()
        },
        "reference_uncertainty_256_to_512": reference_uncertainty,
        "runs": {
            str(subdivisions): _public_run(run)
            for subdivisions, run in runs.items()
        },
        "adjacent_endpoint_errors": adjacent_errors,
        "observable_order_audit": order_audit,
        "finest_endpoint_errors_to_s512": endpoint_errors,
        "finest_endpoint_with_reference_uncertainty": endpoint_audit,
        "physical_ledger_audit": physical_ledger_audit,
        "restart_replay": restart_replay,
        "all_discrete_ledgers_passed": all_discrete_ledgers_passed,
        "authorization": {
            "wp10c7c_adaptive_n16_bdf2_authorized": full_passed,
            "wp10c7d_matched_n32_bdf2_authorized": False,
            "n64_n128_production_authorized": False,
            "long_evolution_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_hot_state_or_cycle_certified": False,
        },
        "decision": (
            "authorize_wp10c7c_adaptive_n16_bdf2"
            if full_passed
            else (
                "partial_ladder_complete_no_authorization"
                if not complete_ladder
                else "stop_wp10c7b_gate_failed"
            )
        ),
        "passed": full_passed,
    }
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


if __name__ == "__main__":
    main()
