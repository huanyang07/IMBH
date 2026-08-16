#!/usr/bin/env python3
"""Execute the frozen primary fixed-Q bounded-continuation pilot."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

e1 = importlib.import_module(  # noqa: E402
    "run_causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1"
)
from imri_qpe.layer3_minidisk_1d.causal_inner_bdf import (  # noqa: E402
    causal_bdf_coefficients,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    CausalFiveFieldFixedQBackwardEulerResult,
    CausalFiveFieldFixedQContinuationState,
    causal_five_field_fixed_q_continuation_state,
    causal_five_field_fixed_q_continuation_states_equal,
    causal_five_field_fixed_q_nonlinear_solver_states_equal,
    load_causal_five_field_fixed_q_continuation_state,
    save_causal_five_field_fixed_q_continuation_state,
    solve_causal_five_field_fixed_q_bdf,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14d"
ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d"
)
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_bounded_continuation_manifest_"
    "wp10c9d6c7c3b5c4f24e14c"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
SEED_PATH = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_continuation_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14b/canonical_seed_continuation.npz"
)
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
    "scripts/run_causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "manifest_wp10c9d6c7c3b5c4f24e14c.py",
)
TIMESTEP_SECONDS = 1.0e-7
HALF_TIMESTEP_SECONDS = 5.0e-8
THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class BindingRootFailure(RuntimeError):
    """Carry a rejected binding root into the canonical failure evidence."""

    def __init__(self, label: str, result, metrics: dict):
        super().__init__(f"binding root {label} failed")
        self.label = label
        self.result = result
        self.metrics = metrics


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_tree_is_clean() -> bool:
    return bool(
        subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode == 0
        and subprocess.run(
            ("git", "diff", "--cached", "--quiet"),
            cwd=ROOT,
        ).returncode
        == 0
    )


def _validate_checksums(directory: Path) -> dict[str, str]:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum changed: {directory / name}")
    return entries


def _validate_frozen_contract(*, require_clean: bool) -> dict:
    checksums = _validate_checksums(MANIFEST_DIRECTORY)
    summary = _read(MANIFEST_DIRECTORY / "summary.json")
    contract = _read(MANIFEST_DIRECTORY / "execution_manifest.json")
    provenance = _read(MANIFEST_DIRECTORY / "provenance.json")
    seed_lock = _read(MANIFEST_DIRECTORY / "seed_lock.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["trajectory_executed"]
        or not summary["primary_bounded_continuation_execution_authorized"]
        or summary["heldout_continuation_authorized"]
        or summary["fixed_Q_micro_solver_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or contract["authorized_execution"]["case"] != "primary_20ms"
        or contract["authorized_execution"]["layout"] != "middle"
        or contract["authorized_execution"]["timestep_seconds"]
        != TIMESTEP_SECONDS
        or contract["authorized_execution"]["new_main_BDF2_roots"] != 4
        or contract["hard_stops"]["no_heldout_continuation"] is not True
        or contract["hard_stops"]["no_fixed_Q_micro_solver"] is not True
        or contract["hard_stops"]["no_reduced_slow_evolution"] is not True
    ):
        raise RuntimeError("frozen primary continuation authorization changed")
    for relative, digest in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != digest:
            raise RuntimeError(f"frozen source changed: {relative}")
    if _sha(SEED_PATH) != seed_lock["sha256"]:
        raise RuntimeError("canonical continuation seed changed")
    if require_clean and not _tracked_tree_is_clean():
        raise RuntimeError("primary continuation execution requires a clean tree")
    current_threads = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if current_threads != THREAD_ENVIRONMENT:
        raise RuntimeError("primary continuation thread environment is not pinned")
    return {
        "summary": summary,
        "contract": contract,
        "provenance": provenance,
        "seed_lock": seed_lock,
        "checksums": checksums,
    }


def _execution_identity() -> dict:
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "manifest_summary_sha256": _sha(
            MANIFEST_DIRECTORY / "summary.json"
        ),
        "manifest_contract_sha256": _sha(
            MANIFEST_DIRECTORY / "execution_manifest.json"
        ),
        "seed_sha256": _sha(SEED_PATH),
    }


def _load_seed(data: dict, identity: dict) -> CausalFiveFieldFixedQContinuationState:
    seed = load_causal_five_field_fixed_q_continuation_state(
        SEED_PATH,
        data["context"],
    )
    if (
        seed.current_order != 2
        or seed.next_order != 2
        or seed.completed_steps != 2
        or seed.history.previous_timestep_seconds != TIMESTEP_SECONDS
        or seed.nonlinear_solver_state is not None
        or seed.next_reaction_channel_basis != "frozen_normalized"
        or not np.array_equal(seed.q3_target, data["reaction"].q3_value)
        or not np.array_equal(
            seed.constraint_row_scales,
            data["reaction"].q3_derivative_norms,
        )
    ):
        raise RuntimeError("canonical continuation seed semantics changed")
    seed_roundtrip = SCRATCH_DIRECTORY / "seed_roundtrip.npz"
    timing = {}
    save_causal_five_field_fixed_q_continuation_state(
        seed_roundtrip,
        data["context"],
        seed,
        timing_accumulator=timing,
    )
    loaded = load_causal_five_field_fixed_q_continuation_state(
        seed_roundtrip,
        data["context"],
        timing_accumulator=timing,
    )
    if not causal_five_field_fixed_q_continuation_states_equal(seed, loaded):
        raise RuntimeError("canonical continuation seed roundtrip changed")
    _write_json(
        SCRATCH_DIRECTORY / "seed_validation.json",
        {
            "passed": True,
            "identity": identity,
            "seed_sha256": _sha(SEED_PATH),
            "roundtrip_sha256": _sha(seed_roundtrip),
            "roundtrip_bitwise": True,
            "checkpoint_timing": timing,
        },
    )
    return loaded


def _predictors(
    continuation: CausalFiveFieldFixedQContinuationState,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rate = (
        continuation.history.previous_primitive_increment / columns
    ).ravel() / continuation.history.previous_timestep_seconds
    multiplier = np.linalg.solve(
        continuation.next_reaction_channel_transform,
        continuation.raw_multiplier_predictor,
    )
    if np.any(~np.isfinite(rate)) or np.any(~np.isfinite(multiplier)):
        raise RuntimeError("fixed-Q continuation predictor is non-finite")
    return rate, multiplier


def _cold_top_left(data: dict, continuation, timestep: float) -> np.ndarray:
    coefficients = causal_bdf_coefficients(
        2,
        timestep,
        continuation.history.previous_timestep_seconds,
    )
    return (
        coefficients.current_increment_coefficient
        * data["reaction"].descriptor_scaled_matrix
        / timestep
        + data["tangent"].evolving_scaled_jacobian
    )


def _root_policy(label: str) -> dict:
    cold = label in {"cold_1", "cold_shadow", "half_1", "half_2"}
    return {
        "cold": cold,
        "initial_exact_jacobian_required": cold,
        "maximum_exact_jacobian_refreshes": 2 if cold else 1,
        "use_carried_solver_state": not cold,
    }


def _result_metrics(result, events: list[dict], wall: float, process: float) -> dict:
    refresh_reasons = [
        event["reason"]
        for event in events
        if event.get("stage") == "exact_jacobian_refresh"
    ]
    return {
        "accepted": result.accepted,
        "message": result.message,
        "failure_reasons": list(result.acceptance.failure_reasons),
        "iterations": result.iterations,
        "function_evaluations": result.function_evaluations,
        "exact_Jacobian_assemblies": result.exact_jacobian_assemblies,
        "exact_Jacobian_reasons": refresh_reasons,
        "Broyden_updates": result.broyden_updates,
        "linear_solves": result.linear_solves,
        "maximum_scaled_residual": result.maximum_scaled_residual,
        "residual_margin": result.maximum_scaled_residual / 1.0e-10,
        "maximum_Q3_relative_defect": (
            result.evaluation.maximum_constraint_relative_defect
        ),
        "maximum_storage_parity_relative_defect": (
            result.maximum_direct_rate_increment_parity_defect
        ),
        "minimum_path_reconstruction_factor": (
            result.minimum_path_reconstruction_factor
        ),
        "maximum_path_reconstruction_factor": (
            result.maximum_path_reconstruction_factor
        ),
        "maximum_reaction_ledger_relative_defect": (
            result.evaluation.reaction.maximum_reaction_ledger_relative_defect
        ),
        "maximum_constraint_action_ledger_relative_defect": (
            result.maximum_multiplier_weighted_action_ledger_relative_defect
        ),
        "raw_Schur_rank": result.evaluation.reaction.raw_schur_numerical_rank,
        "raw_Schur_condition_number": (
            result.evaluation.reaction.raw_schur_condition_number
        ),
        "maximum_raw_Schur_solve_relative_defect": (
            result.evaluation.reaction.maximum_raw_schur_solve_relative_defect
        ),
        "maximum_H_over_R": result.maximum_h_over_r,
        "minimum_scattering_optical_depth": (
            result.minimum_scattering_optical_depth
        ),
        "maximum_scaled_primitive_change": (
            result.maximum_scaled_primitive_change
        ),
        "incoming_excision_characteristics": (
            result.evaluation.monolithic_evaluation
            .incoming_excision_characteristics
        ),
        "matrix_age_steps": result.nonlinear_solver_state.matrix_age_steps,
        "Broyden_updates_since_exact": (
            result.nonlinear_solver_state.broyden_updates_since_exact
        ),
        "solver_profiling": asdict(result.profiling),
        "root_wall_seconds": wall,
        "root_process_seconds": process,
        "event_trace": events,
        "acceptance": asdict(result.acceptance),
    }


def _save_result(path: Path, result, metrics: dict) -> None:
    _write_npz(
        path,
        primitive_charts=result.primitive_charts,
        primitive_increment=result.primitive_increment,
        scaled_rate_per_s=result.scaled_rate_per_s,
        scaled_interval_rate_per_s=result.scaled_interval_rate_per_s,
        multipliers=result.multipliers,
        scaled_reaction_rate_action_per_s=(
            result.scaled_reaction_rate_action_per_s
        ),
        augmented_scaled_residual=result.evaluation.augmented_scaled_residual,
        raw_solver_matrix=(
            result.nonlinear_solver_state
            .bordered_matrix_raw_reaction_coordinates
        ),
        metrics_json=np.asarray(json.dumps(_plain(metrics), sort_keys=True)),
    )


def _solve_root(
    label: str,
    data: dict,
    continuation: CausalFiveFieldFixedQContinuationState,
    timestep: float,
    identity: dict,
    *,
    artifact_label: str | None = None,
) -> tuple[CausalFiveFieldFixedQBackwardEulerResult, dict]:
    stored_label = label if artifact_label is None else artifact_label
    policy = _root_policy(label)
    rate, multiplier = _predictors(continuation, data["columns"])
    top_left = (
        _cold_top_left(data, continuation, timestep)
        if policy["cold"]
        else None
    )
    carried = (
        continuation.nonlinear_solver_state
        if policy["use_carried_solver_state"]
        else None
    )
    events: list[dict] = []

    def progress(payload: dict) -> None:
        plain = _plain(payload)
        events.append(plain)
        print(f"e14d {label}: {plain}", flush=True)

    began_wall = time.perf_counter()
    began_process = time.process_time()
    result = solve_causal_five_field_fixed_q_bdf(
        data["context"],
        continuation.current_primitive_charts,
        timestep,
        rate,
        multiplier,
        top_left,
        order=2,
        history=continuation.history,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        q3_target=continuation.q3_target,
        constraint_row_scales=continuation.constraint_row_scales,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=(
            continuation.next_reaction_channel_transform
        ),
        residual_tolerance=1.0e-10,
        constraint_tolerance=1.0e-12,
        ledger_tolerance=1.0e-12,
        storage_parity_tolerance=1.0e-9,
        minimum_reconstruction_factor=1.0 - 1.0e-12,
        maximum_schur_condition_number=1.0e8,
        maximum_scaled_primitive_change=5.0e-3,
        maximum_newton_iterations=8,
        maximum_line_search_iterations=12,
        refresh_exact_jacobian=True,
        maximum_exact_jacobian_refreshes=(
            policy["maximum_exact_jacobian_refreshes"]
        ),
        exact_jacobian_refresh_policy="on_line_search_failure",
        initial_nonlinear_solver_state=carried,
        initial_exact_jacobian_required=(
            policy["initial_exact_jacobian_required"]
        ),
        solver_state_provenance=identity,
        physical_state_audit=e1._state_audit,
        require_physical_state_audit=True,
        maximum_h_over_r=0.12,
        minimum_scattering_optical_depth=1.0,
        progress_callback=progress,
    )
    process = time.process_time() - began_process
    wall = time.perf_counter() - began_wall
    metrics = _result_metrics(result, events, wall, process)
    metrics.update({"label": label, "policy": policy, "timestep_seconds": timestep})
    _save_result(
        SCRATCH_DIRECTORY / f"result_{stored_label}.npz",
        result,
        metrics,
    )
    _write_json(
        SCRATCH_DIRECTORY / f"metrics_{stored_label}.json",
        metrics,
    )
    return result, metrics


def _roundtrip_checkpoint(
    label: str,
    data: dict,
    continuation: CausalFiveFieldFixedQContinuationState,
    identity: dict,
) -> tuple[CausalFiveFieldFixedQContinuationState, dict]:
    path = SCRATCH_DIRECTORY / f"checkpoint_{label}.npz"
    timings = {}
    began_process = time.process_time()
    save_causal_five_field_fixed_q_continuation_state(
        path,
        data["context"],
        continuation,
        timing_accumulator=timings,
    )
    loaded = load_causal_five_field_fixed_q_continuation_state(
        path,
        data["context"],
        expected_provenance=identity,
        timing_accumulator=timings,
    )
    process = time.process_time() - began_process
    equal = causal_five_field_fixed_q_continuation_states_equal(
        continuation,
        loaded,
    )
    metrics = {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "bitwise_roundtrip": equal,
        "process_seconds": process,
        **timings,
    }
    _write_json(SCRATCH_DIRECTORY / f"checkpoint_{label}.json", metrics)
    return loaded, metrics


def _advance(
    label: str,
    data: dict,
    continuation: CausalFiveFieldFixedQContinuationState,
    timestep: float,
    identity: dict,
) -> tuple[
    CausalFiveFieldFixedQBackwardEulerResult,
    CausalFiveFieldFixedQContinuationState,
    dict,
]:
    result, metrics = _solve_root(
        label,
        data,
        continuation,
        timestep,
        identity,
    )
    policy = _root_policy(label)
    budget_passed = bool(
        result.exact_jacobian_assemblies
        <= policy["maximum_exact_jacobian_refreshes"]
    )
    metrics["exact_assembly_budget_passed"] = budget_passed
    if not result.accepted or not budget_passed:
        metrics["root_passed"] = False
        _write_json(SCRATCH_DIRECTORY / f"metrics_{label}.json", metrics)
        raise BindingRootFailure(label, result, metrics)
    next_continuation = causal_five_field_fixed_q_continuation_state(
        result,
        data["context"],
        continuation.current_primitive_charts,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        elapsed_time_seconds=continuation.elapsed_time_seconds + timestep,
        completed_steps=continuation.completed_steps + 1,
        provenance=identity,
    )
    loaded, checkpoint = _roundtrip_checkpoint(
        label,
        data,
        next_continuation,
        identity,
    )
    metrics["checkpoint"] = checkpoint
    metrics["root_passed"] = bool(checkpoint["bitwise_roundtrip"])
    _write_json(SCRATCH_DIRECTORY / f"metrics_{label}.json", metrics)
    if not metrics["root_passed"]:
        raise BindingRootFailure(label, result, metrics)
    return result, loaded, metrics


def _bitwise_results_equal(
    left: CausalFiveFieldFixedQBackwardEulerResult,
    right: CausalFiveFieldFixedQBackwardEulerResult,
    left_metrics: dict,
    right_metrics: dict,
) -> bool:
    arrays = (
        (left.primitive_charts, right.primitive_charts),
        (left.primitive_increment, right.primitive_increment),
        (left.scaled_rate_per_s, right.scaled_rate_per_s),
        (left.scaled_interval_rate_per_s, right.scaled_interval_rate_per_s),
        (left.multipliers, right.multipliers),
        (
            left.scaled_reaction_rate_action_per_s,
            right.scaled_reaction_rate_action_per_s,
        ),
        (
            left.evaluation.augmented_scaled_residual,
            right.evaluation.augmented_scaled_residual,
        ),
    )
    return bool(
        all(np.array_equal(first, second) for first, second in arrays)
        and left.acceptance == right.acceptance
        and left.message == right.message
        and left.iterations == right.iterations
        and left.function_evaluations == right.function_evaluations
        and left.exact_jacobian_assemblies == right.exact_jacobian_assemblies
        and left.broyden_updates == right.broyden_updates
        and left.linear_solves == right.linear_solves
        and causal_five_field_fixed_q_nonlinear_solver_states_equal(
            left.nonlinear_solver_state,
            right.nonlinear_solver_state,
        )
        and left_metrics["event_trace"] == right_metrics["event_trace"]
    )


def _relative(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _scaled_state_absolute(
    first: np.ndarray,
    second: np.ndarray,
    columns: np.ndarray,
) -> float:
    return float(np.max(np.abs((first - second) / columns)))


def _scaled_endpoint_difference(
    refined: np.ndarray,
    full: np.ndarray,
    start: np.ndarray,
    columns: np.ndarray,
) -> float:
    difference = ((refined - full) / columns).ravel()
    change = ((full - start) / columns).ravel()
    return float(
        np.linalg.norm(difference)
        / max(float(np.linalg.norm(change)), np.finfo(float).tiny)
    )


def _classification(scientific_passed: bool, cost_passed: bool) -> str:
    if not scientific_passed:
        return "bounded_continuation_failed"
    if not cost_passed:
        return "bounded_continuation_valid_cost_failed"
    return "bounded_continuation_and_reuse_passed"


def _run() -> dict:
    frozen = _validate_frozen_contract(require_clean=True)
    identity = _execution_identity()
    if SCRATCH_DIRECTORY.exists():
        raise RuntimeError("primary continuation scratch directory already exists")
    SCRATCH_DIRECTORY.mkdir(parents=True)
    _write_json(SCRATCH_DIRECTORY / "execution_identity.json", identity)
    data = e1._state_data("primary_20ms")
    continuation = _load_seed(data, identity)
    main_results = {}
    main_continuations = {}
    main_metrics = {}
    start_checkpoints = {"cold_1": continuation}
    scientific_passed = True
    failure_stage = None
    try:
        for label in ("cold_1", "warm_1", "warm_2", "warm_3"):
            start_checkpoints[label] = continuation
            result, continuation, metrics = _advance(
                label,
                data,
                continuation,
                TIMESTEP_SECONDS,
                identity,
            )
            main_results[label] = result
            main_continuations[label] = continuation
            main_metrics[label] = metrics
    except BindingRootFailure as error:
        main_results[error.label] = error.result
        main_metrics[error.label] = error.metrics
        scientific_passed = False
        failure_stage = str(error)

    replay_metrics = {"executed": False, "passed": False}
    shadow_metrics = {"executed": False, "scientific_passed": False}
    half_metrics = {"executed": False, "passed": False}
    if scientific_passed:
        replay_start = load_causal_five_field_fixed_q_continuation_state(
            SCRATCH_DIRECTORY / "checkpoint_warm_1.npz",
            data["context"],
            expected_provenance=identity,
        )
        replay_bitwise = True
        replay_roots = {}
        replay_continuation = replay_start
        for label in ("warm_2", "warm_3"):
            replay_label = f"replay_{label}"
            result, metrics = _solve_root(
                label,
                data,
                replay_continuation,
                TIMESTEP_SECONDS,
                identity,
                artifact_label=replay_label,
            )
            if (
                not result.accepted
                or result.exact_jacobian_assemblies
                > _root_policy(label)["maximum_exact_jacobian_refreshes"]
            ):
                replay_bitwise = False
                replay_roots[label] = {
                    "result_bitwise": False,
                    "continuation_bitwise": False,
                    "accepted": result.accepted,
                    "metrics": metrics,
                }
                break
            next_replay = causal_five_field_fixed_q_continuation_state(
                result,
                data["context"],
                replay_continuation.current_primitive_charts,
                primitive_column_scales=data["columns"],
                conservation_row_scales=data["rows"],
                parent_cell_indices=data["layout"].parent_cell_indices,
                refinement_ratio=data["layout"].refinement_ratio,
                elapsed_time_seconds=(
                    replay_continuation.elapsed_time_seconds + TIMESTEP_SECONDS
                ),
                completed_steps=replay_continuation.completed_steps + 1,
                provenance=identity,
            )
            root_equal = _bitwise_results_equal(
                main_results[label],
                result,
                main_metrics[label],
                metrics,
            )
            continuation_equal = causal_five_field_fixed_q_continuation_states_equal(
                main_continuations[label],
                next_replay,
            )
            replay_bitwise = bool(
                replay_bitwise and root_equal and continuation_equal
            )
            replay_roots[label] = {
                "result_bitwise": root_equal,
                "continuation_bitwise": continuation_equal,
                "accepted": result.accepted,
                "metrics": metrics,
            }
            replay_continuation = next_replay
        replay_metrics = {
            "executed": True,
            "passed": replay_bitwise,
            "roots": replay_roots,
        }
        scientific_passed = bool(scientific_passed and replay_bitwise)
        if not replay_bitwise:
            failure_stage = "bitwise_suffix_replay"

    if scientific_passed:
        shadow_start = start_checkpoints["warm_2"]
        shadow, shadow_root_metrics = _solve_root(
            "cold_shadow",
            data,
            shadow_start,
            TIMESTEP_SECONDS,
            identity,
        )
        state_defect = _scaled_state_absolute(
            shadow.primitive_charts,
            main_results["warm_2"].primitive_charts,
            data["columns"],
        )
        action_defect = _relative(
            shadow.scaled_reaction_rate_action_per_s,
            main_results["warm_2"].scaled_reaction_rate_action_per_s,
        )
        wall_ratio = (
            main_metrics["warm_2"]["root_wall_seconds"]
            / max(shadow_root_metrics["root_wall_seconds"], np.finfo(float).tiny)
        )
        shadow_scientific = bool(
            shadow.accepted
            and shadow.exact_jacobian_assemblies <= 2
            and state_defect <= 1.0e-8
            and action_defect <= 1.0e-8
        )
        shadow_metrics = {
            "executed": True,
            "scientific_passed": shadow_scientific,
            "scaled_state_absolute_defect": state_defect,
            "reaction_action_relative_defect": action_defect,
            "warm_to_cold_wall_time_ratio": wall_ratio,
            "cost_passed": wall_ratio <= 0.75,
            "root": shadow_root_metrics,
        }
        scientific_passed = bool(scientific_passed and shadow_scientific)
        if not shadow_scientific:
            failure_stage = "same_history_cold_shadow"

    if replay_metrics["passed"]:
        half_start = start_checkpoints["warm_3"]
        try:
            half_1, half_continuation, half_1_metrics = _advance(
                "half_1",
                data,
                half_start,
                HALF_TIMESTEP_SECONDS,
                identity,
            )
            half_2, _, half_2_metrics = _advance(
                "half_2",
                data,
                half_continuation,
                HALF_TIMESTEP_SECONDS,
                identity,
            )
            state_defect = _scaled_endpoint_difference(
                half_2.primitive_charts,
                main_results["warm_3"].primitive_charts,
                half_start.current_primitive_charts,
                data["columns"],
            )
            action_defect = _relative(
                half_2.scaled_reaction_rate_action_per_s,
                main_results["warm_3"].scaled_reaction_rate_action_per_s,
            )
            half_passed = bool(
                state_defect <= 0.1 and action_defect <= 0.1
            )
            half_metrics = {
                "executed": True,
                "passed": half_passed,
                "state_difference_relative_to_full_step_change": state_defect,
                "reaction_action_relative_difference": action_defect,
                "half_1": half_1_metrics,
                "half_2": half_2_metrics,
            }
        except BindingRootFailure as error:
            half_passed = False
            half_metrics = {
                "executed": True,
                "passed": False,
                "failed_root": error.label,
                "failed_root_metrics": error.metrics,
            }
        scientific_passed = bool(scientific_passed and half_passed)
        if not half_passed:
            failure_stage = "matched_endpoint_half_step_audit"

    cumulative_ledger = sum(
        max(
            metrics["maximum_reaction_ledger_relative_defect"],
            metrics["maximum_constraint_action_ledger_relative_defect"],
        )
        for metrics in main_metrics.values()
    )
    ledger_passed = bool(
        len(main_metrics) == 4 and cumulative_ledger <= 4.0e-12
    )
    scientific_passed = bool(scientific_passed and ledger_passed)
    if not ledger_passed and failure_stage is None:
        failure_stage = "cumulative_ledger"
    warm_without_refresh = sum(
        main_results[label].exact_jacobian_assemblies == 0
        for label in ("warm_1", "warm_2", "warm_3")
        if label in main_results
    )
    cost_passed = bool(
        scientific_passed
        and warm_without_refresh >= 2
        and shadow_metrics.get("cost_passed", False)
    )
    classification = _classification(scientific_passed, cost_passed)
    metrics = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "scientific_passed": scientific_passed,
        "cost_passed": cost_passed,
        "failure_stage": failure_stage,
        "main_roots": main_metrics,
        "replay": replay_metrics,
        "same_history_cold_shadow": shadow_metrics,
        "matched_endpoint_half_step_audit": half_metrics,
        "cumulative_absolute_ledger_budget": cumulative_ledger,
        "cumulative_ledger_passed": ledger_passed,
        "warm_roots_without_exact_refresh": warm_without_refresh,
        "identity": identity,
        "frozen_manifest": frozen["summary"],
    }
    _write_json(SCRATCH_DIRECTORY / "execution_metrics.json", metrics)
    _canonicalize(metrics, data, main_results)
    return metrics


def _catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": (
                        "SUPPORTED" if summary["scientific_passed"] else "REJECTED"
                    ),
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["scientific_passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, data: dict, main_results: dict) -> None:
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("primary continuation canonical package already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    for path in sorted(SCRATCH_DIRECTORY.iterdir()):
        if path.is_file():
            shutil.copy2(path, CANONICAL_DIRECTORY / path.name)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["scientific_passed"],
        "scientific_passed": metrics["scientific_passed"],
        "cost_passed": metrics["cost_passed"],
        "trajectory_executed": True,
        "new_main_BDF2_roots": len(metrics["main_roots"]),
        "new_main_horizon_seconds": (
            len(metrics["main_roots"]) * TIMESTEP_SECONDS
        ),
        "heldout_continuation_manifest_authorized": bool(
            metrics["classification"]
            == "bounded_continuation_and_reuse_passed"
        ),
        "operational_timestep_manifest_authorized": bool(
            metrics["classification"]
            == "bounded_continuation_and_reuse_passed"
        ),
        "solver_optimization_manifest_authorized": bool(
            metrics["classification"]
            == "bounded_continuation_valid_cost_failed"
        ),
        "heldout_continuation_execution_authorized": False,
        "operational_timestep_execution_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", metrics)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in SOURCE_FILES
            },
            "manifest_sha256": _sha(
                MANIFEST_DIRECTORY / "execution_manifest.json"
            ),
            "seed_sha256": _sha(SEED_PATH),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT
            },
        },
    )
    if main_results:
        labels = [
            label
            for label in ("cold_1", "warm_1", "warm_2", "warm_3")
            if label in main_results
        ]
        _write_npz(
            CANONICAL_DIRECTORY / "decisive_arrays.npz",
            labels=np.asarray(labels),
            primitive_charts=np.stack(
                [main_results[label].primitive_charts for label in labels]
            ),
            scaled_rates_per_s=np.stack(
                [main_results[label].scaled_rate_per_s for label in labels]
            ),
            reaction_actions_per_s=np.stack(
                [
                    main_results[label].scaled_reaction_rate_action_per_s
                    for label in labels
                ]
            ),
            q3_target=np.asarray(data["reaction"].q3_value),
        )
    checksum_files = [
        path
        for path in sorted(CANONICAL_DIRECTORY.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in checksum_files),
        encoding="utf-8",
    )
    _catalog(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.validate:
        print(
            json.dumps(
                _plain(_validate_frozen_contract(require_clean=False)),
                indent=2,
                sort_keys=True,
            )
        )
        return
    metrics = _run()
    print(json.dumps(_plain(metrics), indent=2, sort_keys=True))
    if not metrics["scientific_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
