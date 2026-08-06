#!/usr/bin/env python3
"""Continue the frozen nonlinear duration ladder through 1e-3 seconds."""

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
import run_causal_inner_nonlinear_duration_controller_validation_wp10c9d6c7c3b5b as c3b5b  # noqa: E402
import run_causal_inner_nonlinear_first_duration_rung_wp10c9d6c7c3b5c1 as c3b5c1  # noqa: E402
import run_causal_inner_nonlinear_second_duration_rung_manifest_wp10c9d6c7c3b5c2a as c3b5c2a  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFRestart,
    advance_causal_five_field_monolithic_bdf,
    causal_five_field_monolithic_bdf_history,
    causal_five_field_monolithic_bdf_restarts_equal,
    causal_five_field_monolithic_frozen_tangent,
    causal_five_field_monolithic_storage_increment,
    load_causal_five_field_monolithic_bdf_restart,
    save_causal_five_field_monolithic_bdf_restart,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c2"
ANALYZED_BASE_COMMIT = "0d6df95057829f75c40004c44722d9ab664c81d1"
ANALYZED_BASE_PARENT = "9bac242a727b5c7dc254735b2c37b88057a60f45"
ANALYZED_BASE_TREE = "ee2a9a4bcaa2e94d7fb621491f16f2768025e766"

LAYOUT = c3b5c2a.LAYOUT
PROFILE = c3b5c2a.PROFILE
PROFILE_KIND = "primary_physical"
OUTPUT_TIMES = np.asarray(c3b5c2a.OUTPUT_TIMES_SECONDS, dtype=float)
CONTINUATION_OUTPUT_TIMES = np.asarray(
    c3b5c2a.CONTINUATION_OUTPUT_TIMES_SECONDS, dtype=float
)
HORIZON_SECONDS = c3b5c2a.HORIZON_SECONDS
CONTINUATION_START_SECONDS = c3b5c2a.CONTINUATION_START_SECONDS
PREVIOUS_HISTORY_TIME_SECONDS = c3b5c2a.PREVIOUS_HISTORY_TIME_SECONDS
RESTART_TIME_SECONDS = c3b5c2a.RESTART_TIME_SECONDS
STRICT_SHADOW_START_SECONDS = c3b5c2a.STRICT_SHADOW_START_SECONDS
COUPLING_FACE = c3b5c2a.COUPLING_FACE

ARTIFACT = "causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_second_duration_rung_"
    "wp10c9d6c7c3b5c2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_second_duration_rung_"
    "wp10c9d6c7c3b5c2.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_SECOND_DURATION_RUNG_"
    "WP10C9D6C7C3B5C2_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c3b5c2a.CANONICAL_DIRECTORY
FIRST_RUNG_DIRECTORY = c3b5c1.CANONICAL_DIRECTORY
SPATIAL_PILOT_DIRECTORY = c3b2b.CANONICAL_DIRECTORY
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CHECKPOINT_JSON = CHECKPOINT_DIRECTORY / "progress.json"
CHECKPOINT_ARRAYS = CHECKPOINT_DIRECTORY / "progress_arrays.npz"


def _plain(value):
    return c3b5b._plain(value)


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
    return {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST, c3b5c2a.THIS_RUNNER, c3b5c1.THIS_RUNNER)
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    manifest = _read_json(c3b5c2a.MANIFEST_PATH)
    first_rung = _read_json(FIRST_RUNG_DIRECTORY / "summary.json")
    if (
        parent["classification"]
        != "second_nonlinear_duration_rung_manifest_frozen_"
        "one_e_minus_three_second_propagation_authorized"
        or not parent["passed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c2_second_duration_rung_propagation"
        or not parent["second_duration_rung_propagation_authorized"]
        or manifest["propagation_executed"]
        or manifest["stage_authorization"]["later_duration_rungs_authorized_now"]
        or first_rung["classification"]
        != "first_nonlinear_duration_rung_certified_"
        "second_rung_manifest_authorized"
        or first_rung["decisive_arrays_sha256"]
        != _sha256(FIRST_RUNG_DIRECTORY / "decisive_arrays.npz")
    ):
        raise RuntimeError("c2 propagation authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c2 analyzed identity changed")
    return parent, manifest


def _save_restore(context, checkpoint, label: str):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{label}.npz"
        save_causal_five_field_monolithic_bdf_restart(path, context, checkpoint)
        restored = load_causal_five_field_monolithic_bdf_restart(
            path,
            context,
            expected_provenance=checkpoint.provenance,
        )
    return restored, causal_five_field_monolithic_bdf_restarts_equal(
        checkpoint, restored
    )


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
    output_times: np.ndarray,
    stop_time: float,
    checkpoint_times: tuple[float, ...] = (),
    include_initial_output: bool,
    record_accepted_steps: bool = False,
    log_prefix: str = "c2",
) -> dict:
    context = configuration["context"]
    output_targets = np.asarray(output_times, dtype=float)
    output_targets = output_targets[
        (output_targets >= elapsed - 1.0e-15)
        & (output_targets <= stop_time + 1.0e-15)
    ]
    output_states = []
    output_exports = []
    output_times_found = []
    accepted_times = [float(elapsed)]
    accepted_timesteps = []
    local_state_estimates = []
    local_export_estimates = []
    local_error_estimates = []
    retries = []
    step_records = []
    maximum_export_ledger = 0.0
    maximum_export_incoming = 0
    checkpoints = {}
    accepted_states = [np.array(state, copy=True)] if record_accepted_steps else []
    accepted_primitive_histories = (
        [np.array(history.previous_primitive_increment, copy=True)]
        if record_accepted_steps
        else []
    )
    accepted_mapped_histories = (
        [np.array(history.previous_mapped_storage_increment, copy=True)]
        if record_accepted_steps
        else []
    )
    accepted_height_histories = (
        [
            np.array(
                history.previous_responsive_height_storage_increment,
                copy=True,
            )
        ]
        if record_accepted_steps
        else []
    )
    accepted_previous_timesteps = (
        [float(history.previous_timestep_seconds)]
        if record_accepted_steps
        else []
    )
    accepted_step_wall_seconds = []

    def append_output(time_value: float, state_value: np.ndarray) -> None:
        nonlocal maximum_export_ledger, maximum_export_incoming
        export, ledger, incoming = c3b5b._export_value(
            context, state_value, coupling_face
        )
        output_times_found.append(float(time_value))
        output_states.append(np.array(state_value, copy=True))
        output_exports.append(np.asarray(export, dtype=float))
        maximum_export_ledger = max(maximum_export_ledger, ledger)
        maximum_export_incoming = max(maximum_export_incoming, incoming)

    target_index = 0
    if include_initial_output and output_targets.size:
        if abs(float(output_targets[0]) - elapsed) <= 1.0e-15:
            elapsed = float(output_targets[0])
            append_output(elapsed, state)
            target_index = 1

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
            raise RuntimeError("second rung exact landing fell below minimum step")
        attempt_count = 0
        accepted = False
        step_started = time.perf_counter()
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
                [c3b5b._step_record(full), c3b5b._step_record(half_first)]
            )
            if not (
                c3b5b._step_passed(full, contract)
                and c3b5b._step_passed(half_first, contract)
            ):
                raise RuntimeError("second rung full or first-half method gate failed")
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
            step_records.append(c3b5b._step_record(half_second))
            if not c3b5b._step_passed(half_second, contract):
                raise RuntimeError("second rung second-half method gate failed")
            full_export, full_ledger, full_incoming = c3b5b._export_value(
                context, full.primitive_charts, coupling_face
            )
            fine_export, fine_ledger, fine_incoming = c3b5b._export_value(
                context, half_second.primitive_charts, coupling_face
            )
            maximum_export_ledger = max(
                maximum_export_ledger, full_ledger, fine_ledger
            )
            maximum_export_incoming = max(
                maximum_export_incoming, full_incoming, fine_incoming
            )
            state_error = c3b5b._state_estimate(
                full.primitive_charts,
                half_second.primitive_charts,
                field_scales,
            )
            export_error = c3b5b._export_estimate(
                full_export, fine_export, export_scales
            )
            local_error = max(state_error, export_error)
            print(
                f"{log_prefix}: t={elapsed:.8e} dt={timestep:.3e} "
                f"attempt={attempt_count} error={local_error:.3e}",
                flush=True,
            )
            if local_error <= contract["error_estimator"]["local_tolerance"]:
                accepted = True
                break
            timestep *= contract["proposal"]["minimum_factor"]
            if timestep < contract["minimum_timestep_seconds"] - 1.0e-15:
                break
        if not accepted or full.history is None:
            raise RuntimeError("second rung controller exhausted retries")
        state = np.array(full.primitive_charts, copy=True)
        history = full.history
        elapsed = float(elapsed + timestep)
        if abs(elapsed - next_output) <= 1.0e-15:
            elapsed = next_output
        candidate_timestep = c3b5b._next_timestep(
            timestep, local_error, contract
        )
        accepted_times.append(elapsed)
        accepted_timesteps.append(timestep)
        local_state_estimates.append(state_error)
        local_export_estimates.append(export_error)
        local_error_estimates.append(local_error)
        retries.append(attempt_count - 1)
        accepted_step_wall_seconds.append(time.perf_counter() - step_started)
        if record_accepted_steps:
            accepted_states.append(np.array(state, copy=True))
            accepted_primitive_histories.append(
                np.array(history.previous_primitive_increment, copy=True)
            )
            accepted_mapped_histories.append(
                np.array(history.previous_mapped_storage_increment, copy=True)
            )
            accepted_height_histories.append(
                np.array(
                    history.previous_responsive_height_storage_increment,
                    copy=True,
                )
            )
            accepted_previous_timesteps.append(
                float(history.previous_timestep_seconds)
            )

        readiness = c3b1a._state_audit(context, state)
        if (
            readiness["maximum_h_over_r"] > 0.12
            or readiness["minimum_scattering_optical_depth"] <= 1.0
            or readiness["minimum_reconstruction_factor"] < 1.0
        ):
            raise RuntimeError("second rung accepted state failed readiness")

        if abs(elapsed - next_output) <= 1.0e-15:
            append_output(elapsed, state)
            target_index += 1
        for checkpoint_time in checkpoint_times:
            if (
                checkpoint_time not in checkpoints
                and abs(elapsed - checkpoint_time) <= 1.0e-15
            ):
                checkpoints[checkpoint_time] = {
                    "restart": CausalFiveFieldMonolithicBDFRestart(
                        primitive_charts=np.array(state, copy=True),
                        history=history,
                        elapsed_time_seconds=elapsed,
                        completed_steps=len(accepted_timesteps) + 1,
                        next_order=2,
                        provenance={
                            "work_package": WORK_PACKAGE,
                            "layout": LAYOUT,
                            "profile": PROFILE,
                            "checkpoint_time_seconds": checkpoint_time,
                        },
                    ),
                    "next_timestep": float(candidate_timestep),
                }

    if not np.array_equal(
        np.asarray(output_times_found, dtype=float), output_targets
    ):
        raise RuntimeError("second rung controller missed a frozen output")
    return {
        "final_state": state,
        "final_history": history,
        "output_times": np.asarray(output_times_found, dtype=float),
        "output_states": np.asarray(output_states, dtype=float),
        "output_exports": np.asarray(output_exports, dtype=float),
        "accepted_times": np.asarray(accepted_times, dtype=float),
        "accepted_timesteps": np.asarray(accepted_timesteps, dtype=float),
        "local_state_estimates": np.asarray(local_state_estimates, dtype=float),
        "local_export_estimates": np.asarray(local_export_estimates, dtype=float),
        "local_error_estimates": np.asarray(local_error_estimates, dtype=float),
        "retries": np.asarray(retries, dtype=int),
        "step_records": step_records,
        "accepted_step_wall_seconds": np.asarray(
            accepted_step_wall_seconds,
            dtype=float,
        ),
        "accepted_states": np.asarray(accepted_states, dtype=float),
        "accepted_primitive_histories": np.asarray(
            accepted_primitive_histories,
            dtype=float,
        ),
        "accepted_mapped_histories": np.asarray(
            accepted_mapped_histories,
            dtype=float,
        ),
        "accepted_height_histories": np.asarray(
            accepted_height_histories,
            dtype=float,
        ),
        "accepted_previous_timesteps": np.asarray(
            accepted_previous_timesteps,
            dtype=float,
        ),
        "next_candidate_timestep": float(candidate_timestep),
        "maximum_export_ledger_defect": maximum_export_ledger,
        "maximum_export_incoming": maximum_export_incoming,
        "checkpoints": checkpoints,
    }


def _segment_passed(segment: dict, contract: dict, sum_budget: float) -> bool:
    records = segment["step_records"]
    return bool(
        records
        and all(item["accepted"] for item in records)
        and max(item["maximum_scaled_residual"] for item in records) <= 1.0e-10
        and max(item["maximum_discrete_ledger_defect"] for item in records)
        <= 1.0e-12
        and max(
            item["maximum_mapped_endpoint_path_closure_defect"]
            for item in records
        )
        <= 1.0e-9
        and min(
            item["minimum_path_reconstruction_factor"] for item in records
        )
        >= 1.0
        and max(
            item["incoming_excision_characteristics"] for item in records
        )
        == 0
        and segment["maximum_export_ledger_defect"] <= 1.0e-9
        and segment["maximum_export_incoming"] == 0
        and float(np.sum(segment["local_error_estimates"])) <= sum_budget
    )


def _trajectory(
    configuration: dict,
    tangent,
    parent_times: np.ndarray,
    parent_states: np.ndarray,
    parent_exports: np.ndarray,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    manifest: dict,
    trajectory_id: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = configuration["context"]
    started = time.perf_counter()
    main_contract = manifest["main_controller"]
    strict_contract = manifest["strict_shadow"]["controller"]
    parent_times = np.asarray(parent_times, dtype=float)
    parent_states = np.asarray(parent_states, dtype=float)
    parent_exports = np.asarray(parent_exports, dtype=float)
    if not (
        parent_times.shape == (11,)
        and parent_states.shape == (11, 64, 5)
        and parent_exports.shape == (11, 13)
        and np.isclose(
            parent_times[-2], PREVIOUS_HISTORY_TIME_SECONDS, rtol=0.0, atol=1e-18
        )
        and np.isclose(
            parent_times[-1], CONTINUATION_START_SECONDS, rtol=0.0, atol=1e-18
        )
    ):
        raise RuntimeError("second-rung parent history is incomplete")
    previous = np.array(parent_states[-2], copy=True)
    initial = np.array(parent_states[-1], copy=True)
    initial_export, initial_ledger, initial_incoming = c3b5b._export_value(
        context, initial, COUPLING_FACE
    )
    initial_export_defect = float(
        np.max(
            np.abs(
                (np.asarray(initial_export) - parent_exports[-1])
                / export_scales
            )
        )
    )
    storage = causal_five_field_monolithic_storage_increment(
        context,
        previous,
        initial,
    )
    history = causal_five_field_monolithic_bdf_history(
        initial - previous,
        storage,
        CONTINUATION_START_SECONDS - PREVIOUS_HISTORY_TIME_SECONDS,
    )
    initial_restart = CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(initial, copy=True),
        history=history,
        elapsed_time_seconds=CONTINUATION_START_SECONDS,
        completed_steps=12,
        next_order=2,
        provenance={
            "work_package": WORK_PACKAGE,
            "layout": LAYOUT,
            "profile": PROFILE,
            "trajectory_id": trajectory_id,
            "continuation_start_seconds": CONTINUATION_START_SECONDS,
        },
    )
    restored_initial, initial_roundtrip = _save_restore(
        context, initial_restart, f"{trajectory_id}_continuation_start"
    )
    main = _controller_segment(
        configuration,
        tangent,
        restored_initial.primitive_charts,
        restored_initial.history,
        CONTINUATION_START_SECONDS,
        main_contract["initial_timestep_seconds"],
        field_scales,
        export_scales,
        COUPLING_FACE,
        main_contract,
        output_times=CONTINUATION_OUTPUT_TIMES,
        stop_time=HORIZON_SECONDS,
        checkpoint_times=(
            RESTART_TIME_SECONDS,
            STRICT_SHADOW_START_SECONDS,
        ),
        include_initial_output=True,
    )
    parent_output_indices = np.array([0, 5], dtype=int)
    states = np.concatenate(
        (parent_states[parent_output_indices], main["output_states"]), axis=0
    )
    exports = np.concatenate(
        (parent_exports[parent_output_indices], main["output_exports"]), axis=0
    )
    times = np.concatenate(
        (parent_times[parent_output_indices], main["output_times"])
    )
    if not np.allclose(times, OUTPUT_TIMES, rtol=0.0, atol=1e-18):
        raise RuntimeError("second rung complete output grid changed")

    restored = {}
    roundtrips = {}
    for checkpoint_time, payload in main["checkpoints"].items():
        restored[checkpoint_time], roundtrips[checkpoint_time] = _save_restore(
            context, payload["restart"], f"{trajectory_id}_{checkpoint_time:.8e}"
        )
    if set(restored) != {RESTART_TIME_SECONDS, STRICT_SHADOW_START_SECONDS}:
        raise RuntimeError("second rung checkpoint set is incomplete")

    replay_checkpoint = main["checkpoints"][RESTART_TIME_SECONDS]
    replay = _controller_segment(
        configuration,
        tangent,
        restored[RESTART_TIME_SECONDS].primitive_charts,
        restored[RESTART_TIME_SECONDS].history,
        RESTART_TIME_SECONDS,
        replay_checkpoint["next_timestep"],
        field_scales,
        export_scales,
        COUPLING_FACE,
        main_contract,
        output_times=OUTPUT_TIMES,
        stop_time=HORIZON_SECONDS,
        include_initial_output=True,
    )
    replay_index = int(
        np.flatnonzero(
            np.isclose(
                OUTPUT_TIMES, RESTART_TIME_SECONDS, rtol=0.0, atol=1e-18
            )
        )[0]
    )
    replay_bitwise = bool(
        np.array_equal(replay["output_times"], OUTPUT_TIMES[replay_index:])
        and np.array_equal(replay["output_states"], states[replay_index:])
        and np.array_equal(replay["output_exports"], exports[replay_index:])
    )

    strict_checkpoint = main["checkpoints"][STRICT_SHADOW_START_SECONDS]
    strict_candidate = min(
        strict_checkpoint["next_timestep"],
        strict_contract["maximum_timestep_seconds"],
    )
    strict = _controller_segment(
        configuration,
        tangent,
        restored[STRICT_SHADOW_START_SECONDS].primitive_charts,
        restored[STRICT_SHADOW_START_SECONDS].history,
        STRICT_SHADOW_START_SECONDS,
        strict_candidate,
        field_scales,
        export_scales,
        COUPLING_FACE,
        strict_contract,
        output_times=OUTPUT_TIMES,
        stop_time=HORIZON_SECONDS,
        include_initial_output=True,
    )

    main_passed = _segment_passed(
        main,
        main_contract,
        manifest["main_rung_error_budget"][
            "maximum_sum_of_accepted_error_estimates"
        ],
    )
    replay_passed = _segment_passed(
        replay,
        main_contract,
        manifest["main_rung_error_budget"][
            "maximum_sum_of_accepted_error_estimates"
        ],
    )
    strict_sum_budget = strict_contract["error_estimator"][
        "rung_sum_of_accepted_error_estimates"
    ]
    strict_passed = _segment_passed(strict, strict_contract, strict_sum_budget)
    readiness = c3b1a._state_audit(context, main["final_state"])
    all_roundtrips = all(roundtrips.values())
    passed = bool(
        main_passed
        and replay_passed
        and strict_passed
        and initial_roundtrip
        and all_roundtrips
        and replay_bitwise
        and initial_export_defect <= 1.0e-12
        and initial_ledger <= 1.0e-9
        and initial_incoming == 0
    )
    all_records = list(main["step_records"])
    report = {
        "trajectory_id": trajectory_id,
        "passed": passed,
        "accepted_BDF1_steps": 0,
        "continued_from_committed_BDF2_history": True,
        "continuation_history_roundtrip_bitwise": initial_roundtrip,
        "continuation_export_reconstruction_defect": initial_export_defect,
        "continuation_storage_mapped_closure_defect": float(
            storage.maximum_mapped_path_closure_defect
        ),
        "continuation_storage_minimum_reconstruction_factor": float(
            storage.minimum_path_reconstruction_factor
        ),
        "accepted_main_BDF2_steps": int(main["accepted_timesteps"].size),
        "accepted_replay_BDF2_steps": int(replay["accepted_timesteps"].size),
        "accepted_strict_shadow_BDF2_steps": int(
            strict["accepted_timesteps"].size
        ),
        "main_rejected_attempts": int(np.sum(main["retries"])),
        "strict_rejected_attempts": int(np.sum(strict["retries"])),
        "minimum_main_timestep_seconds": float(
            np.min(main["accepted_timesteps"])
        ),
        "maximum_main_timestep_seconds": float(
            np.max(main["accepted_timesteps"])
        ),
        "maximum_main_local_error_estimate": float(
            np.max(main["local_error_estimates"])
        ),
        "sum_main_local_error_estimates": float(
            np.sum(main["local_error_estimates"])
        ),
        "maximum_strict_local_error_estimate": float(
            np.max(strict["local_error_estimates"])
        ),
        "sum_strict_local_error_estimates": float(
            np.sum(strict["local_error_estimates"])
        ),
        "maximum_scaled_residual": max(
            item["maximum_scaled_residual"] for item in all_records
        ),
        "maximum_discrete_ledger_defect": max(
            item["maximum_discrete_ledger_defect"] for item in all_records
        ),
        "maximum_mapped_endpoint_path_closure_defect": max(
            item["maximum_mapped_endpoint_path_closure_defect"]
            for item in all_records
        ),
        "minimum_path_reconstruction_factor": min(
            item["minimum_path_reconstruction_factor"] for item in all_records
        ),
        "maximum_incoming_excision_characteristics": max(
            item["incoming_excision_characteristics"] for item in all_records
        ),
        "maximum_export_ledger_defect": main[
            "maximum_export_ledger_defect"
        ],
        "checkpoint_roundtrips_bitwise": all_roundtrips,
        "split_restart_replay_bitwise": replay_bitwise,
        "final_state_audit": readiness,
        "elapsed_seconds": time.perf_counter() - started,
    }
    arrays = {
        "times_seconds": times,
        "states": states,
        "direct_exports": exports,
        "main_accepted_times_seconds": main["accepted_times"],
        "main_accepted_timesteps_seconds": main["accepted_timesteps"],
        "main_local_state_error_estimates": main["local_state_estimates"],
        "main_local_export_error_estimates": main["local_export_estimates"],
        "main_local_error_estimates": main["local_error_estimates"],
        "main_retries": main["retries"],
        "strict_times_seconds": strict["output_times"],
        "strict_states": strict["output_states"],
        "strict_direct_exports": strict["output_exports"],
        "strict_accepted_timesteps_seconds": strict["accepted_timesteps"],
        "strict_local_error_estimates": strict["local_error_estimates"],
        "strict_retries": strict["retries"],
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
        raise RuntimeError("saved second-rung progress belongs to different code")
    return progress, _load_npz(CHECKPOINT_ARRAYS)


def _save_progress(progress: dict, arrays: dict[str, np.ndarray]) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CHECKPOINT_JSON, progress)
    np.savez_compressed(CHECKPOINT_ARRAYS, **arrays)


def _run(configuration, tangent, manifest, field_scales, export_scales):
    progress, arrays = _load_progress()
    if any(not report.get("passed", False) for report in progress["reports"].values()):
        return progress, arrays
    completed = set(progress["completed"])
    parent = _load_npz(FIRST_RUNG_DIRECTORY / "decisive_arrays.npz")
    for name in ("base", "perturbed"):
        if name in completed:
            continue
        print(f"c2: run {name}", flush=True)
        report, trajectory = _trajectory(
            configuration,
            tangent,
            parent[f"{name}__times_seconds"],
            parent[f"{name}__states"],
            parent[f"{name}__direct_exports"],
            field_scales,
            export_scales,
            manifest,
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


def _cumulative(times: np.ndarray, history: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    values = np.asarray(history, dtype=float)
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5 * np.diff(times)[:, None] * (values[:-1] + values[1:]),
        axis=0,
    )
    return result


def _shadow_comparison(arrays, field_scales, export_scales, manifest):
    shadow_times = arrays["base__strict_times_seconds"]
    start_index = int(
        np.flatnonzero(
            np.isclose(
                OUTPUT_TIMES,
                STRICT_SHADOW_START_SECONDS,
                rtol=0.0,
                atol=1e-18,
            )
        )[0]
    )
    main_state_response = (
        arrays["perturbed__states"] - arrays["base__states"]
    )[start_index:]
    strict_state_response = (
        arrays["perturbed__strict_states"] - arrays["base__strict_states"]
    )
    main_export_response = (
        arrays["perturbed__direct_exports"]
        - arrays["base__direct_exports"]
    )[start_index:]
    strict_export_response = (
        arrays["perturbed__strict_direct_exports"]
        - arrays["base__strict_direct_exports"]
    )
    main_cumulative = _cumulative(shadow_times, main_export_response)
    strict_cumulative = _cumulative(shadow_times, strict_export_response)
    state_scaled = (
        main_state_response - strict_state_response
    ) / field_scales[None, None, :]
    instant_scaled = (
        main_export_response - strict_export_response
    ) / export_scales[None, :]
    window = HORIZON_SECONDS - STRICT_SHADOW_START_SECONDS
    cumulative_scaled = (
        main_cumulative - strict_cumulative
    ) / (window * export_scales[None, :])
    state_maximum = float(np.max(np.abs(state_scaled)))
    state_rms = float(np.sqrt(np.mean(state_scaled * state_scaled)))
    instant_maximum = float(np.max(np.abs(instant_scaled)))
    cumulative_maximum = float(np.max(np.abs(cumulative_scaled)))
    state_cosine = c3b5b._cosine(
        main_state_response / field_scales[None, None, :],
        strict_state_response / field_scales[None, None, :],
    )
    instant_cosine = c3b5b._cosine(
        main_export_response / export_scales[None, :],
        strict_export_response / export_scales[None, :],
    )
    cumulative_cosine = c3b5b._cosine(
        main_cumulative / (window * export_scales[None, :]),
        strict_cumulative / (window * export_scales[None, :]),
    )
    contract = manifest["strict_shadow"]
    passed = bool(
        max(state_maximum, state_rms)
        <= contract["maximum_scaled_state_response_difference"]
        and max(instant_maximum, cumulative_maximum)
        <= contract["maximum_scaled_Tier_I_response_difference"]
        and min(state_cosine, instant_cosine, cumulative_cosine)
        >= contract["minimum_state_and_Tier_I_history_cosine"]
    )
    decisive = {
        "main_shadow_state_response": main_state_response,
        "strict_shadow_state_response": strict_state_response,
        "main_shadow_instantaneous_export_response": main_export_response,
        "strict_shadow_instantaneous_export_response": strict_export_response,
        "main_shadow_cumulative_export_response": main_cumulative,
        "strict_shadow_cumulative_export_response": strict_cumulative,
    }
    return {
        "passed": passed,
        "maximum_scaled_state_response_difference": state_maximum,
        "scaled_state_response_RMS_difference": state_rms,
        "maximum_scaled_instantaneous_Tier_I_response_difference": instant_maximum,
        "maximum_scaled_cumulative_Tier_I_response_difference": cumulative_maximum,
        "state_response_history_cosine": state_cosine,
        "instantaneous_Tier_I_response_history_cosine": instant_cosine,
        "cumulative_Tier_I_response_history_cosine": cumulative_cosine,
    }, decisive


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
    comparison = summary["strict_shadow_comparison"]
    lines = [
        "# Second nonlinear duration rung WP10c9d6c7c3b5c2",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        f"The executed coarse trajectory set was evaluated through "
        f"`{HORIZON_SECONDS:.1e} s` under the frozen variable-step controller.",
        "",
        "## Trajectories",
        "",
    ]
    for name, report in summary["trajectory_reports"].items():
        lines.append(
            f"- `{name}`: main steps `{report['accepted_main_BDF2_steps']}`, "
            f"retries `{report['main_rejected_attempts']}`, maximum local error "
            f"`{report['maximum_main_local_error_estimate']:.3e}`, bitwise replay "
            f"`{report['split_restart_replay_bitwise']}`"
        )
    lines.extend(["", "## Strict shadow", ""])
    if comparison.get("passed") and "maximum_scaled_state_response_difference" in comparison:
        lines.extend(
            [
                f"- maximum/RMS state response difference: "
                f"`{comparison['maximum_scaled_state_response_difference']:.3e}` / "
                f"`{comparison['scaled_state_response_RMS_difference']:.3e}`",
                f"- maximum instantaneous/cumulative Tier-I response difference: "
                f"`{comparison['maximum_scaled_instantaneous_Tier_I_response_difference']:.3e}` / "
                f"`{comparison['maximum_scaled_cumulative_Tier_I_response_difference']:.3e}`",
                f"- state/instantaneous/cumulative cosines: "
                f"`{comparison['state_response_history_cosine']:.9f}` / "
                f"`{comparison['instantaneous_Tier_I_response_history_cosine']:.9f}` / "
                f"`{comparison['cumulative_Tier_I_response_history_cosine']:.9f}`",
            ]
        )
    else:
        lines.append(
            "- not evaluated because the fail-fast trajectory-method contract "
            "did not pass"
        )
    lines.extend(
        [
            "",
            "## Authorized next",
            "",
            f"`{summary['authorized_next']}`",
            "",
            "Later duration, fixed-Q, and reduced evolution remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def _package(parent, manifest, progress, arrays, field_scales, export_scales):
    methods_passed = bool(
        set(progress["reports"]) == {"base", "perturbed"}
        and all(item["passed"] for item in progress["reports"].values())
    )
    if methods_passed:
        shadow, decisive = _shadow_comparison(
            arrays, field_scales, export_scales, manifest
        )
        arrays.update(decisive)
    else:
        shadow = {"passed": False}
    passed = bool(methods_passed and shadow["passed"])
    classification = (
        "second_nonlinear_duration_rung_certified_third_rung_manifest_authorized"
        if passed
        else "second_nonlinear_duration_rung_failed_later_duration_work_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b5c3a_third_duration_rung_manifest"
        if passed
        else "WP10c9d6c7c3b5c2b_second_duration_rung_localization"
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": LAYOUT,
        "profile": PROFILE,
        "coupling_face": COUPLING_FACE,
        "horizon_seconds": HORIZON_SECONDS,
        "output_times_seconds": OUTPUT_TIMES,
        "continuation_start_seconds": CONTINUATION_START_SECONDS,
        "restart_time_seconds": RESTART_TIME_SECONDS,
        "strict_shadow_start_seconds": STRICT_SHADOW_START_SECONDS,
        "main_controller": manifest["main_controller"],
        "strict_shadow": manifest["strict_shadow"],
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
        "strict_shadow_comparison": shadow,
        "third_duration_rung_manifest_authorized": passed,
        "third_duration_rung_propagation_authorized": False,
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
        "second_rung_manifest": c3b5c2a.MANIFEST_PATH,
        "first_rung_summary": FIRST_RUNG_DIRECTORY / "summary.json",
        "first_rung_arrays": FIRST_RUNG_DIRECTORY / "decisive_arrays.npz",
        "spatial_pilot_arrays": SPATIAL_PILOT_DIRECTORY / "decisive_arrays.npz",
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "command": (
                "PYTHONPATH=src:scripts /Users/huanyang/.cache/codex-runtimes/"
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
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
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
    progress, arrays = _load_progress()
    if any(not report.get("passed", False) for report in progress["reports"].values()):
        return _package(
            parent, manifest, progress, arrays, field_scales, export_scales
        )
    print(f"c2: build tangent {LAYOUT}", flush=True)
    started = time.perf_counter()
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    print(f"c2: tangent built in {time.perf_counter() - started:.2f}s", flush=True)
    progress, arrays = _run(
        configuration, tangent, manifest, field_scales, export_scales
    )
    return _package(
        parent, manifest, progress, arrays, field_scales, export_scales
    )


if __name__ == "__main__":
    raise SystemExit(main())
