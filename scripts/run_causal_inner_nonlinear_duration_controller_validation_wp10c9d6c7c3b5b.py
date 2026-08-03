#!/usr/bin/env python3
"""Validate the frozen variable-step monolithic controller at short time."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402
import run_causal_inner_nonlinear_duration_controller_manifest_wp10c9d6c7c3b5a as c3b5a  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_controller_manifest_wp10c9d6c7c3b4a as c3b4a  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_export_face_audit_wp10c9d6c7c3b4d as c3b4d  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_temporal_wp10c9d6c7c3b4b2 as c3b4b2  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402
import run_causal_inner_nonlinear_temporal_coarse_screen_wp10c9d6c7c3b3b1 as c3b3b1  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFRestart,
    advance_causal_five_field_monolithic_bdf,
    causal_five_field_monolithic_bdf_restarts_equal,
    causal_five_field_monolithic_frozen_tangent,
    load_causal_five_field_monolithic_bdf_restart,
    save_causal_five_field_monolithic_bdf_restart,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5b"
ANALYZED_BASE_COMMIT = "d2ece4ec850905e6e5ae7a673dde48fa6b414c99"
ANALYZED_BASE_PARENT = "e3c4550ed5588db93ca0784a5c7827f2d07c590f"
ANALYZED_BASE_TREE = "b2e792cca5a2c68a9365ca28e1a2407ae654f572"

LAYOUT = c3b5a.CONTROLLER_LAYOUT
PROFILE = c3b5a.CONTROLLER_PROFILE
PROFILE_KIND = "primary_physical"
OUTPUT_TIMES = np.asarray(c3b4a.COMMON_OUTPUT_TIMES_SECONDS, dtype=float)
HORIZON_SECONDS = float(c3b5a.VALIDATION_HORIZON_SECONDS)
MIDPOINT_SECONDS = 2.0e-5

ARTIFACT = (
    "causal_inner_nonlinear_duration_controller_validation_"
    "wp10c9d6c7c3b5b"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_duration_controller_validation_"
    "wp10c9d6c7c3b5b.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_duration_controller_validation_"
    "wp10c9d6c7c3b5b.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_DURATION_CONTROLLER_VALIDATION_"
    "WP10C9D6C7C3B5B_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

PARENT_DIRECTORY = c3b5a.CANONICAL_DIRECTORY
REFERENCE_DIRECTORY = c3b4b2.CANONICAL_DIRECTORY
SPATIAL_PILOT_DIRECTORY = c3b2b.CANONICAL_DIRECTORY
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CHECKPOINT_JSON = CHECKPOINT_DIRECTORY / "progress.json"
CHECKPOINT_ARRAYS = CHECKPOINT_DIRECTORY / "progress_arrays.npz"


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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        c3b5a.THIS_RUNNER,
        c3b4d.THIS_RUNNER,
        c3b3b1.THIS_RUNNER,
    )
    return {
        path: _sha256(ROOT / path)
        for path in paths
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    manifest = _read_json(c3b5a.MANIFEST_PATH)
    if (
        parent["classification"]
        != "variable_step_monolithic_duration_controller_manifest_frozen_"
        "short_horizon_controller_validation_authorized"
        or not parent["passed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5b_short_horizon_variable_step_controller_validation"
        or manifest["propagation_executed"]
        or manifest["stage_authorization"]["duration_rungs_authorized_now"]
    ):
        raise RuntimeError("b5b authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("b5b analyzed identity changed")
    return parent, manifest


def _step_passed(step, contract: dict) -> bool:
    gates = contract["step_method_gates"]
    return bool(
        step.accepted
        and step.history is not None
        and step.maximum_scaled_residual <= gates["maximum_scaled_residual"]
        and step.maximum_discrete_ledger_defect
        <= gates["maximum_discrete_ledger_defect"]
        and step.evaluation.maximum_mapped_endpoint_path_closure_defect
        <= gates["maximum_mapped_endpoint_path_closure"]
        and step.minimum_path_reconstruction_factor
        >= gates["minimum_reconstruction_factor"] - 1.0e-12
        and step.incoming_excision_characteristics
        <= gates["maximum_incoming_excision_characteristics"]
    )


def _state_estimate(full: np.ndarray, fine: np.ndarray, scales: np.ndarray) -> float:
    scaled = (np.asarray(fine) - np.asarray(full)) / scales[None, :]
    rms = float(np.sqrt(np.mean(scaled * scaled)))
    maximum = float(np.max(np.abs(scaled)))
    return (4.0 / 3.0) * max(rms, maximum)


def _export_value(context, state: np.ndarray, coupling_face: int):
    values, ledger, incoming = c3b4d._export_history(
        context, np.asarray(state, dtype=float)[None, ...], coupling_face
    )
    return values[0], float(ledger), int(incoming)


def _export_estimate(
    full: np.ndarray,
    fine: np.ndarray,
    scales: np.ndarray,
) -> float:
    scaled = (np.asarray(fine) - np.asarray(full)) / scales
    return (4.0 / 3.0) * float(np.max(np.abs(scaled)))


def _next_timestep(current: float, error: float, contract: dict) -> float:
    proposal = contract["proposal"]
    tolerance = contract["error_estimator"]["local_tolerance"]
    if error <= np.finfo(float).tiny:
        factor = proposal["maximum_factor"]
    else:
        factor = proposal["safety_factor"] * (
            tolerance / error
        ) ** proposal["error_exponent"]
        factor = min(
            proposal["maximum_factor"],
            max(proposal["minimum_factor"], factor),
        )
    return float(
        min(
            contract["maximum_timestep_seconds"],
            max(contract["minimum_timestep_seconds"], current * factor),
        )
    )


def _step_record(step) -> dict:
    return {
        "accepted": bool(step.accepted),
        "order": int(step.order),
        "timestep_seconds": float(step.timestep_seconds),
        "maximum_scaled_residual": float(step.maximum_scaled_residual),
        "maximum_scaled_algebraic_residual": float(
            step.maximum_scaled_algebraic_residual
        ),
        "maximum_discrete_ledger_defect": float(
            step.maximum_discrete_ledger_defect
        ),
        "maximum_mapped_endpoint_path_closure_defect": float(
            step.evaluation.maximum_mapped_endpoint_path_closure_defect
        ),
        "minimum_path_reconstruction_factor": float(
            step.minimum_path_reconstruction_factor
        ),
        "incoming_excision_characteristics": int(
            step.incoming_excision_characteristics
        ),
        "iterations": int(step.iterations),
        "function_evaluations": int(step.function_evaluations),
    }


def _controller_segment(
    configuration: dict,
    tangent,
    state: np.ndarray,
    history,
    elapsed: float,
    candidate_timestep: float,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    coupling_face: int,
    contract: dict,
    *,
    stop_time: float,
    include_initial_output: bool,
) -> dict:
    context = configuration["context"]
    output_targets = OUTPUT_TIMES[
        (OUTPUT_TIMES >= elapsed - 1.0e-15)
        & (OUTPUT_TIMES <= stop_time + 1.0e-15)
    ]
    output_states = []
    output_exports = []
    output_times = []
    accepted_times = [float(elapsed)]
    accepted_timesteps = []
    local_state_estimates = []
    local_export_estimates = []
    local_error_estimates = []
    retries = []
    step_records = []
    maximum_export_ledger = 0.0
    maximum_export_incoming = 0

    def append_output(time_value: float, state_value: np.ndarray) -> None:
        nonlocal maximum_export_ledger, maximum_export_incoming
        export, ledger, incoming = _export_value(
            context, state_value, coupling_face
        )
        output_times.append(float(time_value))
        output_states.append(np.array(state_value, copy=True))
        output_exports.append(np.asarray(export, dtype=float))
        maximum_export_ledger = max(maximum_export_ledger, ledger)
        maximum_export_incoming = max(maximum_export_incoming, incoming)

    target_index = 0
    if include_initial_output and output_targets.size:
        if abs(float(output_targets[0]) - elapsed) <= 1.0e-15:
            append_output(elapsed, state)
            target_index = 1

    checkpoint = None
    checkpoint_next_timestep = None
    while elapsed < stop_time - 1.0e-15:
        next_output = (
            float(output_targets[target_index])
            if target_index < output_targets.size
            else float(stop_time)
        )
        previous_timestep = float(history.previous_timestep_seconds)
        timestep = min(
            float(candidate_timestep),
            next_output - elapsed,
            contract["maximum_BDF2_step_ratio"] * previous_timestep,
        )
        if timestep < contract["minimum_timestep_seconds"] - 1.0e-15:
            raise RuntimeError("controller exact landing fell below minimum step")
        attempt_count = 0
        accepted = False
        while attempt_count <= contract["proposal"]["maximum_retries"]:
            attempt_count += 1
            full = advance_causal_five_field_monolithic_bdf(
                context,
                state,
                timestep,
                tangent,
                order=2,
                history=history,
                residual_tolerance=1.0e-10,
                ledger_tolerance=1.0e-12,
                maximum_scaled_primitive_change=5.0e-3,
            )
            half_first = advance_causal_five_field_monolithic_bdf(
                context,
                state,
                0.5 * timestep,
                tangent,
                order=2,
                history=history,
                residual_tolerance=1.0e-10,
                ledger_tolerance=1.0e-12,
                maximum_scaled_primitive_change=5.0e-3,
            )
            step_records.extend(
                [_step_record(full), _step_record(half_first)]
            )
            half_second = None
            if _step_passed(half_first, contract):
                half_second = advance_causal_five_field_monolithic_bdf(
                    context,
                    half_first.primitive_charts,
                    0.5 * timestep,
                    tangent,
                    order=2,
                    history=half_first.history,
                    residual_tolerance=1.0e-10,
                    ledger_tolerance=1.0e-12,
                    maximum_scaled_primitive_change=5.0e-3,
                )
                step_records.append(_step_record(half_second))
            method_passed = bool(
                _step_passed(full, contract)
                and _step_passed(half_first, contract)
                and half_second is not None
                and _step_passed(half_second, contract)
            )
            if method_passed:
                full_export, full_ledger, full_incoming = _export_value(
                    context, full.primitive_charts, coupling_face
                )
                fine_export, fine_ledger, fine_incoming = _export_value(
                    context, half_second.primitive_charts, coupling_face
                )
                maximum_export_ledger = max(
                    maximum_export_ledger, full_ledger, fine_ledger
                )
                maximum_export_incoming = max(
                    maximum_export_incoming, full_incoming, fine_incoming
                )
                state_error = _state_estimate(
                    full.primitive_charts,
                    half_second.primitive_charts,
                    field_scales,
                )
                export_error = _export_estimate(
                    full_export, fine_export, export_scales
                )
                local_error = max(state_error, export_error)
            else:
                state_error = float("inf")
                export_error = float("inf")
                local_error = float("inf")
            print(
                f"b5b: t={elapsed:.8e} dt={timestep:.3e} "
                f"attempt={attempt_count} error={local_error:.3e}",
                flush=True,
            )
            if method_passed and local_error <= contract["error_estimator"]["local_tolerance"]:
                accepted = True
                break
            timestep *= contract["proposal"]["minimum_factor"]
            if timestep < contract["minimum_timestep_seconds"] - 1.0e-15:
                break
        if not accepted or full.history is None:
            raise RuntimeError("variable-step controller exhausted retries")
        state = np.array(full.primitive_charts, copy=True)
        history = full.history
        elapsed = float(elapsed + timestep)
        candidate_timestep = _next_timestep(timestep, local_error, contract)
        accepted_times.append(elapsed)
        accepted_timesteps.append(timestep)
        local_state_estimates.append(state_error)
        local_export_estimates.append(export_error)
        local_error_estimates.append(local_error)
        retries.append(attempt_count - 1)

        state_audit = c3b1a._state_audit(context, state)
        if (
            state_audit["maximum_h_over_r"] > c3b3b1.MAXIMUM_H_OVER_R
            or state_audit["minimum_scattering_optical_depth"]
            <= c3b3b1.MINIMUM_SCATTERING_OPTICAL_DEPTH
            or state_audit["minimum_reconstruction_factor"]
            < c3b3b1.MINIMUM_RECONSTRUCTION_FACTOR
        ):
            raise RuntimeError("accepted controller state failed readiness")

        if abs(elapsed - next_output) <= 1.0e-15:
            append_output(elapsed, state)
            target_index += 1
        if abs(elapsed - MIDPOINT_SECONDS) <= 1.0e-15:
            checkpoint = CausalFiveFieldMonolithicBDFRestart(
                primitive_charts=np.array(state, copy=True),
                history=history,
                elapsed_time_seconds=elapsed,
                completed_steps=len(accepted_timesteps) + 1,
                next_order=2,
                provenance={
                    "work_package": WORK_PACKAGE,
                    "layout": LAYOUT,
                    "profile": PROFILE,
                },
            )
            checkpoint_next_timestep = float(candidate_timestep)

    if not np.array_equal(np.asarray(output_times), output_targets):
        raise RuntimeError("controller missed a frozen output")
    return {
        "final_state": state,
        "final_history": history,
        "final_candidate_timestep": float(candidate_timestep),
        "output_times": np.asarray(output_times, dtype=float),
        "output_states": np.asarray(output_states, dtype=float),
        "output_exports": np.asarray(output_exports, dtype=float),
        "accepted_times": np.asarray(accepted_times, dtype=float),
        "accepted_timesteps": np.asarray(accepted_timesteps, dtype=float),
        "local_state_estimates": np.asarray(local_state_estimates, dtype=float),
        "local_export_estimates": np.asarray(local_export_estimates, dtype=float),
        "local_error_estimates": np.asarray(local_error_estimates, dtype=float),
        "retries": np.asarray(retries, dtype=int),
        "step_records": step_records,
        "maximum_export_ledger_defect": maximum_export_ledger,
        "maximum_export_incoming": maximum_export_incoming,
        "checkpoint": checkpoint,
        "checkpoint_next_timestep": checkpoint_next_timestep,
    }


def _trajectory(
    configuration: dict,
    tangent,
    initial_state: np.ndarray,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    coupling_face: int,
    contract: dict,
    trajectory_id: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = configuration["context"]
    started = time.perf_counter()
    initial = np.asarray(initial_state, dtype=float)
    initial_export, initial_ledger, initial_incoming = _export_value(
        context, initial, coupling_face
    )
    predictor = c3b3b1._scaled_linear_predictor(
        configuration,
        tangent,
        initial,
        contract["initial_timestep_seconds"],
    )
    startup = advance_causal_five_field_monolithic_bdf(
        context,
        initial,
        contract["initial_timestep_seconds"],
        tangent,
        order=1,
        initial_primitive_increment=predictor,
        residual_tolerance=1.0e-10,
        ledger_tolerance=1.0e-12,
        maximum_scaled_primitive_change=5.0e-3,
    )
    if not _step_passed(startup, contract):
        raise RuntimeError(f"{trajectory_id} BDF1 startup failed")
    segment = _controller_segment(
        configuration,
        tangent,
        startup.primitive_charts,
        startup.history,
        contract["initial_timestep_seconds"],
        contract["initial_timestep_seconds"],
        field_scales,
        export_scales,
        coupling_face,
        contract,
        stop_time=HORIZON_SECONDS,
        include_initial_output=False,
    )
    states = np.concatenate(
        (initial[None, ...], segment["output_states"]), axis=0
    )
    exports = np.concatenate(
        (initial_export[None, ...], segment["output_exports"]), axis=0
    )
    times = np.concatenate(([0.0], segment["output_times"]))
    if not np.array_equal(times, OUTPUT_TIMES):
        raise RuntimeError("complete controller output grid changed")

    checkpoint = segment["checkpoint"]
    if checkpoint is None or segment["checkpoint_next_timestep"] is None:
        raise RuntimeError("controller midpoint checkpoint was not formed")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "restart.npz"
        save_causal_five_field_monolithic_bdf_restart(path, context, checkpoint)
        restored = load_causal_five_field_monolithic_bdf_restart(
            path,
            context,
            expected_provenance=checkpoint.provenance,
        )
    roundtrip = causal_five_field_monolithic_bdf_restarts_equal(
        checkpoint, restored
    )
    replay = _controller_segment(
        configuration,
        tangent,
        restored.primitive_charts,
        restored.history,
        restored.elapsed_time_seconds,
        segment["checkpoint_next_timestep"],
        field_scales,
        export_scales,
        coupling_face,
        contract,
        stop_time=HORIZON_SECONDS,
        include_initial_output=True,
    )
    midpoint_index = int(np.flatnonzero(OUTPUT_TIMES == MIDPOINT_SECONDS)[0])
    replay_bitwise = bool(
        np.array_equal(replay["output_times"], OUTPUT_TIMES[midpoint_index:])
        and np.array_equal(replay["output_states"], states[midpoint_index:])
        and np.array_equal(replay["output_exports"], exports[midpoint_index:])
    )
    records = [_step_record(startup), *segment["step_records"]]
    state_audit = c3b1a._state_audit(context, segment["final_state"])
    method_passed = bool(
        all(item["accepted"] for item in records)
        and max(item["maximum_scaled_residual"] for item in records) <= 1.0e-10
        and max(item["maximum_discrete_ledger_defect"] for item in records) <= 1.0e-12
        and max(
            item["maximum_mapped_endpoint_path_closure_defect"]
            for item in records
        )
        <= 1.0e-9
        and min(item["minimum_path_reconstruction_factor"] for item in records)
        >= 1.0 - 1.0e-12
        and max(item["incoming_excision_characteristics"] for item in records)
        == 0
        and segment["maximum_export_ledger_defect"] <= 1.0e-9
        and segment["maximum_export_incoming"] == 0
        and float(np.sum(segment["local_error_estimates"]))
        <= contract["error_estimator"]["short_horizon_sum_of_accepted_error_estimates"]
        and roundtrip
        and replay_bitwise
    )
    report = {
        "trajectory_id": trajectory_id,
        "passed": method_passed,
        "accepted_BDF1_steps": 1,
        "accepted_BDF2_steps": int(segment["accepted_timesteps"].size),
        "rejected_attempts": int(np.sum(segment["retries"])),
        "minimum_accepted_timestep_seconds": float(
            np.min(segment["accepted_timesteps"])
        ),
        "maximum_accepted_timestep_seconds": float(
            np.max(segment["accepted_timesteps"])
        ),
        "maximum_local_state_error_estimate": float(
            np.max(segment["local_state_estimates"])
        ),
        "maximum_local_export_error_estimate": float(
            np.max(segment["local_export_estimates"])
        ),
        "maximum_local_error_estimate": float(
            np.max(segment["local_error_estimates"])
        ),
        "sum_local_error_estimates": float(
            np.sum(segment["local_error_estimates"])
        ),
        "maximum_scaled_residual": max(
            item["maximum_scaled_residual"] for item in records
        ),
        "maximum_discrete_ledger_defect": max(
            item["maximum_discrete_ledger_defect"] for item in records
        ),
        "maximum_mapped_endpoint_path_closure_defect": max(
            item["maximum_mapped_endpoint_path_closure_defect"]
            for item in records
        ),
        "minimum_path_reconstruction_factor": min(
            item["minimum_path_reconstruction_factor"] for item in records
        ),
        "maximum_incoming_excision_characteristics": max(
            item["incoming_excision_characteristics"] for item in records
        ),
        "maximum_export_ledger_defect": segment[
            "maximum_export_ledger_defect"
        ],
        "checkpoint_roundtrip_bitwise": roundtrip,
        "split_restart_replay_bitwise": replay_bitwise,
        "final_state_audit": state_audit,
        "elapsed_seconds": time.perf_counter() - started,
    }
    arrays = {
        "times_seconds": times,
        "states": states,
        "direct_exports": exports,
        "accepted_times_seconds": segment["accepted_times"],
        "accepted_timesteps_seconds": segment["accepted_timesteps"],
        "local_state_error_estimates": segment["local_state_estimates"],
        "local_export_error_estimates": segment["local_export_estimates"],
        "local_error_estimates": segment["local_error_estimates"],
        "retries": segment["retries"],
    }
    return report, arrays


def _load_progress() -> tuple[dict, dict[str, np.ndarray]]:
    if not CHECKPOINT_JSON.exists():
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "work_package": WORK_PACKAGE,
                "analyzed_base_commit": ANALYZED_BASE_COMMIT,
                "source_identity": _source_identity(),
                "completed": [],
                "reports": {},
            },
            {},
        )
    progress = _read_json(CHECKPOINT_JSON)
    if (
        progress.get("work_package") != WORK_PACKAGE
        or progress.get("analyzed_base_commit") != ANALYZED_BASE_COMMIT
        or progress.get("source_identity") != _source_identity()
    ):
        raise RuntimeError("saved b5b progress belongs to different code")
    return progress, _load_npz(CHECKPOINT_ARRAYS)


def _save_progress(progress: dict, arrays: dict[str, np.ndarray]) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CHECKPOINT_JSON, progress)
    np.savez_compressed(CHECKPOINT_ARRAYS, **arrays)


def _run(configuration: dict, tangent, manifest: dict, field_scales, export_scales):
    progress, arrays = _load_progress()
    completed = set(progress["completed"])
    controller = manifest["controller_contract"]
    coupling_face = int(controller["coupling_face_contract"][LAYOUT])
    packets = _load_npz(c3b4a.DECISIVE_ARRAYS)
    initial_states = {
        "base": np.asarray(configuration["base"], dtype=float),
        "perturbed": np.asarray(configuration["base"], dtype=float)
        + np.asarray(
            packets[f"{PROFILE}__{LAYOUT}__{PROFILE_KIND}"], dtype=float
        ),
    }
    for name, initial in initial_states.items():
        if name in completed:
            continue
        print(f"b5b: run {name}", flush=True)
        report, trajectory = _trajectory(
            configuration,
            tangent,
            initial,
            field_scales,
            export_scales,
            coupling_face,
            controller,
            name,
        )
        progress["reports"][name] = report
        for key, values in trajectory.items():
            arrays[f"{name}__{key}"] = np.asarray(values)
        completed.add(name)
        progress["completed"] = sorted(completed)
        _save_progress(progress, arrays)
        if not report["passed"]:
            break
    return progress, arrays


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def _comparison(arrays: dict[str, np.ndarray], field_scales, export_scales, contract):
    reference = _load_npz(REFERENCE_DIRECTORY / "decisive_arrays.npz")
    state = arrays["perturbed__states"] - arrays["base__states"]
    instant = arrays["perturbed__direct_exports"] - arrays["base__direct_exports"]
    cumulative = c3b2b._cumulative(instant)
    reference_state = np.asarray(
        reference[f"{PROFILE}__h4__state_response"], dtype=float
    )
    reference_instant = np.asarray(
        reference[f"{PROFILE}__h4__instantaneous_export_response"], dtype=float
    )
    reference_cumulative = np.asarray(
        reference[f"{PROFILE}__h4__cumulative_export_response"], dtype=float
    )
    scaled_state = (state - reference_state) / field_scales[None, None, :]
    scaled_instant = (instant - reference_instant) / export_scales[None, :]
    scaled_cumulative = (cumulative - reference_cumulative) / (
        HORIZON_SECONDS * export_scales[None, :]
    )
    state_maximum = float(np.max(np.abs(scaled_state)))
    state_rms = float(np.sqrt(np.mean(scaled_state * scaled_state)))
    instant_maximum = float(np.max(np.abs(scaled_instant)))
    cumulative_maximum = float(np.max(np.abs(scaled_cumulative)))
    state_cosine = _cosine(
        state / field_scales[None, None, :],
        reference_state / field_scales[None, None, :],
    )
    instant_cosine = _cosine(
        instant / export_scales[None, :],
        reference_instant / export_scales[None, :],
    )
    cumulative_cosine = _cosine(
        cumulative / (HORIZON_SECONDS * export_scales[None, :]),
        reference_cumulative / (HORIZON_SECONDS * export_scales[None, :]),
    )
    passed = bool(
        max(state_maximum, state_rms)
        <= contract["maximum_controller_to_reference_scaled_state_difference"]
        and max(instant_maximum, cumulative_maximum)
        <= contract["maximum_controller_to_reference_scaled_Tier_I_difference"]
        and min(state_cosine, instant_cosine, cumulative_cosine)
        >= contract["minimum_history_cosine"]
    )
    decisive = {
        "controller_state_response": state,
        "controller_instantaneous_export_response": instant,
        "controller_cumulative_export_response": cumulative,
        "reference_state_response": reference_state,
        "reference_instantaneous_export_response": reference_instant,
        "reference_cumulative_export_response": reference_cumulative,
    }
    return (
        {
            "passed": passed,
            "maximum_scaled_state_difference": state_maximum,
            "scaled_state_RMS_difference": state_rms,
            "maximum_scaled_instantaneous_Tier_I_difference": instant_maximum,
            "maximum_scaled_cumulative_Tier_I_difference": cumulative_maximum,
            "state_history_cosine": state_cosine,
            "instantaneous_Tier_I_history_cosine": instant_cosine,
            "cumulative_Tier_I_history_cosine": cumulative_cosine,
        },
        decisive,
    )


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "CERTIFIED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": status,
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _report(summary: dict) -> str:
    result = summary["validation"]
    lines = [
        "# Nonlinear variable-step controller validation WP10c9d6c7c3b5b",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        "The frozen controller evolved independent background and generic "
        "five-field trajectories on the coarse physical embedded layout and "
        "was compared with the committed fixed `dt=2.5e-6 s` reference.",
        "",
        "## Controller/reference result",
        "",
        f"- maximum/RMS scaled state difference: "
        f"`{result['maximum_scaled_state_difference']:.3e}` / "
        f"`{result['scaled_state_RMS_difference']:.3e}`",
        f"- maximum instantaneous/cumulative Tier-I difference: "
        f"`{result['maximum_scaled_instantaneous_Tier_I_difference']:.3e}` / "
        f"`{result['maximum_scaled_cumulative_Tier_I_difference']:.3e}`",
        f"- state/instantaneous/cumulative history cosines: "
        f"`{result['state_history_cosine']:.9f}` / "
        f"`{result['instantaneous_Tier_I_history_cosine']:.9f}` / "
        f"`{result['cumulative_Tier_I_history_cosine']:.9f}`",
        "",
        "## Method",
        "",
    ]
    for name, item in summary["trajectory_reports"].items():
        lines.extend(
            [
                f"- `{name}`: BDF2 steps `{item['accepted_BDF2_steps']}`, "
                f"rejections `{item['rejected_attempts']}`, step range "
                f"`{item['minimum_accepted_timestep_seconds']:.3e}-"
                f"{item['maximum_accepted_timestep_seconds']:.3e} s`, "
                f"maximum local estimate `{item['maximum_local_error_estimate']:.3e}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Authorized next",
            "",
            f"`{summary['authorized_next']}`",
            "",
            "Only the definitions-only first duration-rung manifest is authorized. "
            "Longer rungs, fixed-Q experiments and reduced evolution remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def _package(parent: dict, manifest: dict, progress: dict, arrays: dict, field_scales, export_scales) -> int:
    validation_contract = manifest["short_horizon_validation_contract"]
    expected = {"base", "perturbed"}
    methods_passed = bool(
        set(progress["reports"]) == expected
        and all(item["passed"] for item in progress["reports"].values())
    )
    if methods_passed:
        comparison, comparison_arrays = _comparison(
            arrays,
            field_scales,
            export_scales,
            validation_contract,
        )
        arrays.update(comparison_arrays)
    else:
        comparison = {"passed": False}
    passed = bool(methods_passed and comparison["passed"])
    classification = (
        "short_horizon_variable_step_controller_certified_first_duration_rung_manifest_authorized"
        if passed
        else "short_horizon_variable_step_controller_failed_duration_ladder_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b5c1a_first_duration_rung_manifest"
        if passed
        else "WP10c9d6c7c3b5b1_controller_failure_localization"
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": LAYOUT,
        "profile": PROFILE,
        "output_times_seconds": OUTPUT_TIMES,
        "controller_contract": manifest["controller_contract"],
        "validation_contract": validation_contract,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "parent_classification_preserved": parent["classification"],
        "operator_changed": False,
        "production_defaults_changed": False,
        "trajectory_reports": progress["reports"],
        "all_trajectory_methods_passed": methods_passed,
        "validation": comparison,
        "first_duration_rung_manifest_authorized": passed,
        "first_duration_rung_propagation_authorized": False,
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(_plain(config)),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values) for name, values in arrays.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    input_paths = {
        "parent_summary": PARENT_DIRECTORY / "summary.json",
        "controller_manifest": c3b5a.MANIFEST_PATH,
        "profile_arrays": c3b4a.DECISIVE_ARRAYS,
        "reference_arrays": REFERENCE_DIRECTORY / "decisive_arrays.npz",
        "spatial_pilot_arrays": SPATIAL_PILOT_DIRECTORY / "decisive_arrays.npz",
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "command": (
                "PYTHONPATH=src /Users/huanyang/.cache/codex-runtimes/"
                "codex-primary-runtime/dependencies/python/bin/python3 "
                + THIS_RUNNER
            ),
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
            "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
            "implementation_parent_commit": _git_value("rev-parse", "HEAD"),
            "implementation_parent_tree_sha": _git_value("rev-parse", "HEAD^{tree}"),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "implementation_source_hashes": _source_identity(),
            "input_hashes": {
                name: _sha256(path) for name, path in input_paths.items()
            },
        },
    )
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    names = ("config.json", "summary.json", "provenance.json", "decisive_arrays.npz")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parent, manifest = _validate_parent()
    configuration = c3b1a._configurations()[LAYOUT]
    pilot = _load_npz(SPATIAL_PILOT_DIRECTORY / "decisive_arrays.npz")
    field_scales = np.asarray(pilot["field_scales"], dtype=float)
    export_scales = np.asarray(
        pilot["fixed_physical_observable_scales"], dtype=float
    )
    print(f"b5b: build tangent {LAYOUT}", flush=True)
    started = time.perf_counter()
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    print(f"b5b: tangent built in {time.perf_counter() - started:.2f}s", flush=True)
    progress, arrays = _run(
        configuration, tangent, manifest, field_scales, export_scales
    )
    return _package(
        parent, manifest, progress, arrays, field_scales, export_scales
    )


if __name__ == "__main__":
    raise SystemExit(main())
