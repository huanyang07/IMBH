#!/usr/bin/env python3
"""Execute the optimized middle-layout continuation from 6 to 20 ms."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_optimized_middle_20ms_completion_manifest_wp10c9d6c7c3b5c4e2 as c4e2  # noqa: E402
import run_causal_inner_nonlinear_middle_6ms_spatial_cost_pilot_wp10c9d6c7c3b5c4e1 as c4e1  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    advance_causal_five_field_monolithic_bdf,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_fluctuation import (  # noqa: E402
    causal_five_field_radial_candidate_ledger,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e3"
ANALYZED_BASE_COMMIT = c4e2.ANALYZED_BASE_COMMIT
ANALYZED_BASE_PARENT = c4e2.ANALYZED_BASE_PARENT
ANALYZED_BASE_TREE = c4e2.ANALYZED_BASE_TREE

OUTPUT_TARGET_MICROSECONDS = tuple(c4e2.OUTPUT_TARGET_MICROSECONDS)
AUDIT_TARGET_MICROSECONDS = tuple(c4e2.AUDIT_TARGET_MICROSECONDS)
OUTPUT_TARGET_SECONDS = np.asarray(OUTPUT_TARGET_MICROSECONDS, dtype=float) * 1.0e-6
AUDIT_TARGET_SECONDS = np.asarray(AUDIT_TARGET_MICROSECONDS, dtype=float) * 1.0e-6
LANDMARK_SECONDS = np.unique(
    np.concatenate((OUTPUT_TARGET_SECONDS, AUDIT_TARGET_SECONDS))
)
MAXIMUM_TIMESTEP_SECONDS = max(c4e2.TIMESTEP_CAP_CANDIDATES_SECONDS)
ROUTINE_ERROR_BOUND_SAFETY_FACTOR = 4.0
PRE_CAP_FALLBACK_RUNNER_SHA256 = (
    "4c8c8dc2afd2c494b80b5dfbb287cff199c21fa94ed7fcf52fbed7a600c54094"
)
EXTRACTION_JVP_RELATIVE_STEP = c4e2.EXTRACTION_JVP_RELATIVE_STEP
EXTRACTION_JVP_STEP_SWEEP = tuple(c4e2.EXTRACTION_JVP_STEP_SWEEP)
GENERIC_INDEX = c4e1.h2b1.GENERIC_INDEX
PROFILES = tuple(c4e1.h2b1.PROFILES)
EXTRACTION_FACE = c4e1.EXTRACTION_FACE
COUPLING_FACE = c4e1.COUPLING_FACE

ARTIFACT = (
    "causal_inner_nonlinear_optimized_middle_20ms_completion_"
    "wp10c9d6c7c3b5c4e3"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_optimized_middle_20ms_completion_"
    "wp10c9d6c7c3b5c4e3.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_optimized_middle_20ms_completion_"
    "wp10c9d6c7c3b5c4e3.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_OPTIMIZED_MIDDLE_20MS_"
    "COMPLETION_WP10C9D6C7C3B5C4E3_2026-08-10.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
PROGRESS_PATH = CHECKPOINT_DIRECTORY / "progress.json"
BASE_PATH = CHECKPOINT_DIRECTORY / "base.npz"
TANGENT_PATH = CHECKPOINT_DIRECTORY / "tangent.npz"
ANCHOR_PATH = CHECKPOINT_DIRECTORY / "anchor.npz"
EXTRACTION_PATH = CHECKPOINT_DIRECTORY / "extraction.npz"

h2b1 = c4e1.h2b1
controller = h2b1.controller
extraction5 = c4e1.extraction5


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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    dependencies = (
        THIS_RUNNER,
        THIS_TEST,
        c4e2.THIS_RUNNER,
        c4e2.THIS_TEST,
        h2b1.CONTROLLER_RELATIVE,
        h2b1.MODULE_RELATIVE,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py",
        "scripts/run_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_audit_wp10c9d6c7c3b5c3h2j1.py",
    )
    return {
        path: _sha256(ROOT / path) for path in dependencies if (ROOT / path).exists()
    }


def _checkpoint_hashes() -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in (BASE_PATH, TANGENT_PATH, ANCHOR_PATH, EXTRACTION_PATH)
        if path.exists()
    }


def _save_progress(progress: dict) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    payload = dict(progress)
    payload["checkpoint_hashes"] = _checkpoint_hashes()
    _write_json(PROGRESS_PATH, payload)
    progress["checkpoint_hashes"] = payload["checkpoint_hashes"]


def _progress() -> dict:
    identity = _source_identity()
    manifest_hash = _sha256(c4e2.MANIFEST_PATH)
    if PROGRESS_PATH.exists():
        payload = _read_json(PROGRESS_PATH)
        if payload.get("source_identity") != identity:
            raise RuntimeError("c4e3 checkpoint source identity changed")
        if payload.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("c4e3 checkpoint manifest changed")
        if payload.get("checkpoint_hashes") != _checkpoint_hashes():
            raise RuntimeError("c4e3 checkpoint payload hash changed")
        return payload
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "source_identity": identity,
        "manifest_sha256": manifest_hash,
        "base_steps_completed": 0,
        "tangent_steps_completed": 0,
        "anchor_steps_completed": 0,
        "extraction_times_completed": 0,
        "reports": {},
        "checkpoint_hashes": {},
    }


def _migrate_cap_fallback_checkpoint_identity() -> None:
    """Retain accepted arrays after hardening future method-gate fallback."""

    if not PROGRESS_PATH.exists():
        return
    payload = _read_json(PROGRESS_PATH)
    prior = dict(payload.get("source_identity", {}))
    current = _source_identity()
    runner_key = THIS_RUNNER
    if prior == current:
        return
    if prior.get(runner_key) != PRE_CAP_FALLBACK_RUNNER_SHA256:
        raise RuntimeError("c4e3 checkpoint is not the pre-cap-fallback run")
    if {key: value for key, value in prior.items() if key != runner_key} != {
        key: value for key, value in current.items() if key != runner_key
    }:
        raise RuntimeError("c4e3 scientific checkpoint dependencies changed")
    base = _load_npz(BASE_PATH)
    if (
        int(base["accepted_timesteps"].size) != 2
        or int(payload["reports"]["base"]["rejected_attempts"]) != 1
    ):
        raise RuntimeError("c4e3 pre-cap-fallback checkpoint shape changed")
    base["selected_maximum_timestep"] = np.asarray([4.0e-4])
    base["next_candidate_timestep"] = np.asarray(
        [min(float(base["next_candidate_timestep"][0]), 4.0e-4)]
    )
    np.savez_compressed(BASE_PATH, **base)
    payload["reports"]["base"]["cap_fallbacks"] = [
        {
            "time_seconds": 6.4e-3,
            "failed_timestep_seconds": 8.0e-4,
            "selected_cap_seconds": 4.0e-4,
            "reason": "full_step_method_gate",
            "accepted_arrays_recomputed": False,
        }
    ]
    payload["source_identity"] = current
    payload["cap_fallback_identity_migration"] = {
        "prior_runner_sha256": PRE_CAP_FALLBACK_RUNNER_SHA256,
        "accepted_arrays_recomputed": False,
        "scientific_results_changed": False,
        "future_method_gate_failure_now_locks_largest_passing_cap": True,
    }
    payload["checkpoint_hashes"] = _checkpoint_hashes()
    _write_json(PROGRESS_PATH, payload)


def _validate_parent() -> tuple[dict, dict]:
    manifest_summary = _read_json(c4e2.SUMMARY_PATH)
    manifest = _read_json(c4e2.MANIFEST_PATH)
    pilot = _read_json(c4e1.SUMMARY_PATH)
    if (
        not manifest_summary["passed"]
        or not manifest_summary["middle_twenty_ms_optimized_propagation_authorized"]
        or manifest_summary["fine_twenty_ms_propagation_authorized"]
        or manifest_summary["fixed_q_micro_solver_authorized"]
        or manifest_summary["reduced_slow_evolution_authorized"]
        or manifest_summary["authorized_next"]
        != f"{WORK_PACKAGE}_optimized_middle_6_to_20ms_completion"
        or not pilot["passed"]
        or not pilot["scientific_gates_passed"]
    ):
        raise RuntimeError("c4e3 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e3 analyzed identity changed")
    return manifest_summary, manifest


def _patch_shared_modules() -> None:
    h2b1.WORK_PACKAGE = WORK_PACKAGE
    h2b1.ARTIFACT = ARTIFACT
    h2b1.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
    h2b1.PROGRESS_PATH = PROGRESS_PATH
    h2b1.TANGENT_PATH = TANGENT_PATH
    h2b1.ANCHOR_PATH = ANCHOR_PATH
    h2b1._source_identity = _source_identity
    h2b1._save_progress = _save_progress
    h2b1._initial_tangent = _initial_tangent
    h2b1._initial_anchor = _initial_anchor
    h2b1._tangent_audit_indices = _declared_audit_indices
    h2b1._anchor_sample_indices = _declared_audit_indices


def _parent_arrays() -> dict[str, np.ndarray]:
    return _load_npz(c4e1.DECISIVE_ARRAYS)


def _append(existing: np.ndarray, new: np.ndarray) -> np.ndarray:
    value = np.asarray(new)
    if value.size == 0:
        return np.asarray(existing)
    return np.concatenate((np.asarray(existing), value), axis=0)


def _time_us(value: float) -> int:
    return int(np.rint(float(value) * 1.0e6))


def _declared_audit_indices(base: dict[str, np.ndarray]) -> set[int]:
    return {
        index
        for index, value in enumerate(base["accepted_times"][1:])
        if _time_us(value) in set(AUDIT_TARGET_MICROSECONDS)
    }


def _initial_base(configuration: dict) -> tuple[dict[str, np.ndarray], dict]:
    parent = _parent_arrays()
    state = np.asarray(parent["base__accepted_states"][-1], dtype=float)
    value, ledger, incoming = controller._export_value(
        configuration["context"], state, COUPLING_FACE
    )
    arrays = {
        "accepted_times": np.asarray([6.0e-3]),
        "accepted_timesteps": np.empty(0, dtype=float),
        "accepted_states": state[None, ...],
        "accepted_primitive_histories": parent[
            "base__accepted_primitive_histories"
        ][-1:],
        "accepted_mapped_histories": parent["base__accepted_mapped_histories"][-1:],
        "accepted_height_histories": parent["base__accepted_height_histories"][-1:],
        "accepted_previous_timesteps": parent[
            "base__accepted_previous_timesteps"
        ][-1:],
        "accepted_step_wall_seconds": np.empty(0, dtype=float),
        "audit_flags": np.empty(0, dtype=bool),
        "local_state_estimates": np.empty(0, dtype=float),
        "local_Tier_I_estimates": np.empty(0, dtype=float),
        "local_extraction_estimates": np.empty(0, dtype=float),
        "local_error_bounds": np.empty(0, dtype=float),
        "retries": np.empty(0, dtype=np.int64),
        "step_maximum_scaled_residuals": np.empty(0, dtype=float),
        "step_maximum_discrete_ledger_defects": np.empty(0, dtype=float),
        "step_maximum_mapped_closure_defects": np.empty(0, dtype=float),
        "step_minimum_reconstruction_factors": np.empty(0, dtype=float),
        "step_incoming_excision_characteristics": np.empty(0, dtype=np.int64),
        "step_export_ledger_defects": np.empty(0, dtype=float),
        "step_extraction_identity_defects": np.empty(0, dtype=float),
        "output_times": np.asarray([6.0e-3]),
        "output_states": state[None, ...],
        "output_exports": np.asarray(value, dtype=float)[None, :],
        "next_candidate_timestep": np.asarray(
            [float(parent["base__next_candidate_timestep"][-1])]
        ),
        "selected_maximum_timestep": np.asarray([MAXIMUM_TIMESTEP_SECONDS]),
        "last_audit_timestep": np.asarray(
            [float(parent["base__accepted_timesteps"][-1])]
        ),
        "last_audit_error": np.asarray(
            [float(parent["base__local_error_estimates"][-1])]
        ),
    }
    report = {
        "passed_so_far": bool(ledger <= 1.0e-9 and incoming == 0),
        "accepted_steps": 0,
        "audited_steps": 0,
        "rejected_attempts": 0,
        "wall_seconds": 0.0,
        "maximum_export_ledger_defect": float(ledger),
        "maximum_export_incoming_characteristics": int(incoming),
        "cap_fallbacks": [],
    }
    return arrays, report


def _initial_tangent() -> tuple[dict[str, np.ndarray], dict]:
    parent = _parent_arrays()
    arrays = {
        "state_directions": parent["tangent__state_directions"][-1:],
        "export_directions": parent["tangent__export_directions"][-1:],
        "primitive_history_directions": parent[
            "tangent__primitive_history_directions"
        ][-1:],
        "mapped_history_directions": parent[
            "tangent__mapped_history_directions"
        ][-1:],
        "height_history_directions": parent[
            "tangent__height_history_directions"
        ][-1:],
        "matrix_assembly_wall_seconds": np.empty(0, dtype=float),
        "block_step_wall_seconds": np.empty(0, dtype=float),
        "audit_flags": np.empty(0, dtype=bool),
        "step_ratios": np.empty(0, dtype=float),
        "field_scales": parent["tangent__field_scales"],
        "export_scales": parent["tangent__export_scales"],
    }
    report = {
        "maximum_step_matrix_jvp_relative_defect": 0.0,
        "maximum_linear_solve_relative_defect": 0.0,
        "maximum_matrix_component_closure_defect": 0.0,
        "maximum_incoming_excision_characteristics": 0,
        "maximum_export_active_prefix_ledger_defect": 0.0,
        "maximum_export_transport_telescoping_defect": 0.0,
    }
    return arrays, report


def _initial_anchor(configuration: dict, base: dict[str, np.ndarray]):
    parent = _parent_arrays()
    state = np.asarray(parent["anchor__anchor_states"][-1], dtype=float)
    base_value, base_ledger, base_incoming = controller._export_value(
        configuration["context"], base["accepted_states"][0], COUPLING_FACE
    )
    anchor_value, anchor_ledger, anchor_incoming = controller._export_value(
        configuration["context"], state, COUPLING_FACE
    )
    arrays = {
        "anchor_states": state[None, ...],
        "anchor_primitive_histories": parent[
            "anchor__anchor_primitive_histories"
        ][-1:],
        "anchor_mapped_histories": parent["anchor__anchor_mapped_histories"][-1:],
        "anchor_height_histories": parent["anchor__anchor_height_histories"][-1:],
        "anchor_previous_timesteps": parent[
            "anchor__anchor_previous_timesteps"
        ][-1:],
        "anchor_predictors": np.empty((0, *state.shape), dtype=float),
        "anchor_step_wall_seconds": np.empty(0, dtype=float),
        "sampled_flags": np.empty(0, dtype=bool),
        "sampled_state_error_estimates": np.empty(0, dtype=float),
        "sampled_export_error_estimates": np.empty(0, dtype=float),
        "base_exports": np.asarray(base_value, dtype=float)[None, :],
        "anchor_exports": np.asarray(anchor_value, dtype=float)[None, :],
    }
    report = {
        "maximum_scaled_residual": 0.0,
        "maximum_discrete_ledger_defect": 0.0,
        "maximum_mapped_endpoint_path_closure_defect": 0.0,
        "minimum_path_reconstruction_factor": 1.0,
        "maximum_incoming_excision_characteristics": 0,
        "maximum_export_ledger_defect": max(base_ledger, anchor_ledger),
        "maximum_export_incoming_characteristics": max(
            base_incoming, anchor_incoming
        ),
    }
    return arrays, report


def _extraction_value(context, state: np.ndarray):
    ledger = causal_five_field_radial_candidate_ledger(context, state)
    value, identity = extraction5._observable_from_ledger(
        ledger, EXTRACTION_FACE, COUPLING_FACE
    )
    audit = np.asarray(
        (
            ledger.interfaces.shared_conservative_face_defect,
            ledger.local_block_ledger_defect,
            ledger.source_double_count_defect,
            ledger.interfaces.incoming_excision_characteristics,
        ),
        dtype=float,
    )
    return np.asarray(value, dtype=float), float(identity), audit


def _scaled_difference(left, right, scales) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right)) / scales))


def _step_metrics(records: list[dict]) -> tuple[float, float, float, float, int]:
    return (
        max(item["maximum_scaled_residual"] for item in records),
        max(item["maximum_discrete_ledger_defect"] for item in records),
        max(item["maximum_mapped_endpoint_path_closure_defect"] for item in records),
        min(item["minimum_path_reconstruction_factor"] for item in records),
        max(item["incoming_excision_characteristics"] for item in records),
    )


def _next_landmark(elapsed: float) -> float:
    later = LANDMARK_SECONDS[LANDMARK_SECONDS > elapsed + 1.0e-15]
    if later.size == 0:
        return 20.0e-3
    return float(later[0])


def _is_audit_endpoint(value: float) -> bool:
    return _time_us(value) in set(AUDIT_TARGET_MICROSECONDS)


def _is_output_endpoint(value: float) -> bool:
    return _time_us(value) in set(OUTPUT_TARGET_MICROSECONDS)


def _run_base(
    progress: dict,
    configuration: dict,
    frozen_tangent,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    extraction_scales: np.ndarray,
    contract: dict,
    manifest: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    if BASE_PATH.exists():
        arrays = _load_npz(BASE_PATH)
        report = dict(progress["reports"]["base"])
    else:
        arrays, report = _initial_base(configuration)
        CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(BASE_PATH, **arrays)
        progress["reports"]["base"] = report
        _save_progress(progress)
    context = configuration["context"]
    tolerance = float(manifest["method_gates"]["maximum_local_error_estimate"])
    margin = float(manifest["method_gates"]["minimum_audit_error_margin_factor"])
    minimum_dt = float(contract["minimum_timestep_seconds"])
    start_index = int(progress["base_steps_completed"])
    del start_index
    while float(arrays["accepted_times"][-1]) < 20.0e-3 - 1.0e-15:
        elapsed = float(arrays["accepted_times"][-1])
        state = np.asarray(arrays["accepted_states"][-1], dtype=float)
        history = h2b1.h2a2._history(
            arrays["accepted_primitive_histories"][-1],
            arrays["accepted_mapped_histories"][-1],
            arrays["accepted_height_histories"][-1],
            arrays["accepted_previous_timesteps"][-1],
        )
        landmark = _next_landmark(elapsed)
        selected_cap = float(arrays["selected_maximum_timestep"][0])
        candidate = float(arrays["next_candidate_timestep"][0])
        previous_dt = float(history.previous_timestep_seconds)
        dt = min(
            candidate,
            landmark - elapsed,
            2.0 * previous_dt,
            selected_cap,
        )
        if dt < minimum_dt - 1.0e-15:
            raise RuntimeError("c4e3 exact landing fell below the minimum timestep")
        endpoint = elapsed + dt
        audited = _is_audit_endpoint(endpoint)
        attempts = 0
        accepted = False
        began = time.perf_counter()
        while attempts <= int(contract["proposal"]["maximum_retries"]):
            attempts += 1
            full = advance_causal_five_field_monolithic_bdf(
                context,
                state,
                dt,
                frozen_tangent,
                order=2,
                history=history,
                residual_tolerance=1.0e-10,
                ledger_tolerance=1.0e-12,
                maximum_scaled_primitive_change=5.0e-3,
            )
            records = [controller._step_record(full)]
            if full.history is None or not controller._step_passed(full, contract):
                failed_dt = dt
                smaller = [
                    cap
                    for cap in c4e2.TIMESTEP_CAP_CANDIDATES_SECONDS
                    if cap < failed_dt - 1.0e-15
                ]
                if smaller:
                    selected_cap = max(smaller)
                    arrays["selected_maximum_timestep"] = np.asarray([selected_cap])
                    report["cap_fallbacks"].append(
                        {
                            "time_seconds": elapsed,
                            "failed_timestep_seconds": failed_dt,
                            "selected_cap_seconds": selected_cap,
                            "reason": "full_step_method_gate",
                            "failed_step_record": records[0],
                        }
                    )
                dt = min(0.5 * failed_dt, selected_cap)
                audited = True
                if dt < minimum_dt - 1.0e-15:
                    break
                continue
            full_export, full_ledger, full_incoming = controller._export_value(
                context, full.primitive_charts, COUPLING_FACE
            )
            full_extraction, full_identity, _full_extraction_audit = (
                _extraction_value(context, full.primitive_charts)
            )
            state_error = 0.0
            tier_error = 0.0
            extraction_error = 0.0
            if audited:
                half_first = advance_causal_five_field_monolithic_bdf(
                    context,
                    state,
                    0.5 * dt,
                    frozen_tangent,
                    order=2,
                    history=history,
                    residual_tolerance=1.0e-10,
                    ledger_tolerance=1.0e-12,
                    maximum_scaled_primitive_change=5.0e-3,
                )
                if half_first.history is None:
                    raise RuntimeError("c4e3 first half step has no history")
                half_second = advance_causal_five_field_monolithic_bdf(
                    context,
                    half_first.primitive_charts,
                    0.5 * dt,
                    frozen_tangent,
                    order=2,
                    history=half_first.history,
                    residual_tolerance=1.0e-10,
                    ledger_tolerance=1.0e-12,
                    maximum_scaled_primitive_change=5.0e-3,
                )
                records.extend(
                    [controller._step_record(half_first), controller._step_record(half_second)]
                )
                if not (
                    controller._step_passed(half_first, contract)
                    and controller._step_passed(half_second, contract)
                ):
                    dt *= 0.5
                    if dt < minimum_dt - 1.0e-15:
                        break
                    continue
                fine_export, fine_ledger, fine_incoming = controller._export_value(
                    context, half_second.primitive_charts, COUPLING_FACE
                )
                fine_extraction, _fine_identity, _fine_audit = _extraction_value(
                    context, half_second.primitive_charts
                )
                report["maximum_export_ledger_defect"] = max(
                    report["maximum_export_ledger_defect"], full_ledger, fine_ledger
                )
                report["maximum_export_incoming_characteristics"] = max(
                    report["maximum_export_incoming_characteristics"],
                    full_incoming,
                    fine_incoming,
                )
                state_error = controller._state_estimate(
                    full.primitive_charts, half_second.primitive_charts, field_scales
                )
                tier_error = controller._export_estimate(
                    full_export, fine_export, export_scales
                )
                extraction_error = _scaled_difference(
                    full_extraction, fine_extraction, extraction_scales
                )
                local_bound = max(state_error, tier_error, extraction_error)
                preflight = any(
                    abs(dt - cap) <= 1.0e-15
                    for cap in c4e2.TIMESTEP_CAP_CANDIDATES_SECONDS
                )
                accepted_limit = tolerance / margin if preflight else tolerance
                if local_bound > accepted_limit:
                    smaller = [
                        cap
                        for cap in c4e2.TIMESTEP_CAP_CANDIDATES_SECONDS
                        if cap < selected_cap - 1.0e-15
                    ]
                    if preflight and smaller:
                        selected_cap = max(smaller)
                        arrays["selected_maximum_timestep"] = np.asarray([selected_cap])
                        report["cap_fallbacks"].append(
                            {
                                "time_seconds": elapsed,
                                "failed_timestep_seconds": dt,
                                "selected_cap_seconds": selected_cap,
                                "local_error": local_bound,
                            }
                        )
                    dt = min(0.5 * dt, selected_cap)
                    audited = True
                    if dt < minimum_dt - 1.0e-15:
                        break
                    continue
                arrays["last_audit_timestep"] = np.asarray([dt])
                arrays["last_audit_error"] = np.asarray([local_bound])
            else:
                local_bound = (
                    ROUTINE_ERROR_BOUND_SAFETY_FACTOR
                    * float(arrays["last_audit_error"][0])
                    * (dt / float(arrays["last_audit_timestep"][0])) ** 3
                )
                if local_bound > tolerance:
                    audited = True
                    continue
            accepted = True
            break
        if not accepted or full.history is None:
            raise RuntimeError("c4e3 optimized base exhausted retries")
        wall = time.perf_counter() - began
        endpoint = elapsed + dt
        if abs(endpoint - landmark) <= 1.0e-15:
            endpoint = landmark
        readiness = h2b1.h2a2.h2.h1.b1a._state_audit(context, full.primitive_charts)
        if (
            readiness["maximum_h_over_r"] > 0.12
            or readiness["minimum_scattering_optical_depth"] <= 1.0
            or readiness["minimum_reconstruction_factor"] < 1.0
        ):
            raise RuntimeError("c4e3 optimized base failed physical readiness")
        metrics = _step_metrics(records)
        arrays["accepted_times"] = _append(arrays["accepted_times"], [endpoint])
        arrays["accepted_timesteps"] = _append(arrays["accepted_timesteps"], [dt])
        arrays["accepted_states"] = _append(
            arrays["accepted_states"], full.primitive_charts[None, ...]
        )
        arrays["accepted_primitive_histories"] = _append(
            arrays["accepted_primitive_histories"],
            full.history.previous_primitive_increment[None, ...],
        )
        arrays["accepted_mapped_histories"] = _append(
            arrays["accepted_mapped_histories"],
            full.history.previous_mapped_storage_increment[None, ...],
        )
        arrays["accepted_height_histories"] = _append(
            arrays["accepted_height_histories"],
            full.history.previous_responsive_height_storage_increment[None, ...],
        )
        arrays["accepted_previous_timesteps"] = _append(
            arrays["accepted_previous_timesteps"],
            [full.history.previous_timestep_seconds],
        )
        arrays["accepted_step_wall_seconds"] = _append(
            arrays["accepted_step_wall_seconds"], [wall]
        )
        arrays["audit_flags"] = _append(arrays["audit_flags"], [audited])
        arrays["local_state_estimates"] = _append(
            arrays["local_state_estimates"], [state_error]
        )
        arrays["local_Tier_I_estimates"] = _append(
            arrays["local_Tier_I_estimates"], [tier_error]
        )
        arrays["local_extraction_estimates"] = _append(
            arrays["local_extraction_estimates"], [extraction_error]
        )
        arrays["local_error_bounds"] = _append(
            arrays["local_error_bounds"], [local_bound]
        )
        arrays["retries"] = _append(arrays["retries"], [attempts - 1])
        arrays["step_maximum_scaled_residuals"] = _append(
            arrays["step_maximum_scaled_residuals"], [metrics[0]]
        )
        arrays["step_maximum_discrete_ledger_defects"] = _append(
            arrays["step_maximum_discrete_ledger_defects"], [metrics[1]]
        )
        arrays["step_maximum_mapped_closure_defects"] = _append(
            arrays["step_maximum_mapped_closure_defects"], [metrics[2]]
        )
        arrays["step_minimum_reconstruction_factors"] = _append(
            arrays["step_minimum_reconstruction_factors"], [metrics[3]]
        )
        arrays["step_incoming_excision_characteristics"] = _append(
            arrays["step_incoming_excision_characteristics"], [metrics[4]]
        )
        arrays["step_export_ledger_defects"] = _append(
            arrays["step_export_ledger_defects"], [full_ledger]
        )
        arrays["step_extraction_identity_defects"] = _append(
            arrays["step_extraction_identity_defects"], [full_identity]
        )
        candidate = controller._next_timestep(dt, local_bound, contract)
        arrays["next_candidate_timestep"] = np.asarray(
            [min(candidate, float(arrays["selected_maximum_timestep"][0]))]
        )
        if _is_output_endpoint(endpoint):
            arrays["output_times"] = _append(arrays["output_times"], [endpoint])
            arrays["output_states"] = _append(
                arrays["output_states"], full.primitive_charts[None, ...]
            )
            arrays["output_exports"] = _append(
                arrays["output_exports"], np.asarray(full_export)[None, ...]
            )
        report["accepted_steps"] += 1
        report["audited_steps"] += int(audited)
        report["rejected_attempts"] += attempts - 1
        report["wall_seconds"] += wall
        report["maximum_export_ledger_defect"] = max(
            report["maximum_export_ledger_defect"], full_ledger
        )
        report["maximum_export_incoming_characteristics"] = max(
            report["maximum_export_incoming_characteristics"], full_incoming
        )
        progress["base_steps_completed"] = int(arrays["accepted_timesteps"].size)
        progress["reports"]["base"] = report
        np.savez_compressed(BASE_PATH, **arrays)
        _save_progress(progress)
        print(
            f"c4e3-base: step={arrays['accepted_timesteps'].size} "
            f"t={endpoint:.8e} dt={dt:.3e} audit={audited} "
            f"bound={local_bound:.3e} wall={wall:.1f}s",
            flush=True,
        )
    output_ids = tuple(_time_us(value) for value in arrays["output_times"])
    if output_ids != OUTPUT_TARGET_MICROSECONDS:
        raise RuntimeError("c4e3 output target identity changed")
    report.update(
        {
            "passed": bool(
                report["passed_so_far"]
                and float(np.max(arrays["step_maximum_scaled_residuals"])) <= 1.0e-10
                and float(np.max(arrays["step_maximum_discrete_ledger_defects"]))
                <= 1.0e-12
                and float(np.max(arrays["step_maximum_mapped_closure_defects"]))
                <= 1.0e-9
                and float(np.min(arrays["step_minimum_reconstruction_factors"])) >= 1.0
                and int(np.max(arrays["step_incoming_excision_characteristics"])) == 0
                and report["maximum_export_ledger_defect"] <= 1.0e-9
                and report["maximum_export_incoming_characteristics"] == 0
                and float(np.max(arrays["step_extraction_identity_defects"]))
                <= 1.0e-12
                and float(np.max(arrays["local_error_bounds"])) <= tolerance
                and float(np.sum(arrays["local_error_bounds"]))
                <= manifest["method_gates"]["maximum_sum_of_local_error_bounds"]
            ),
            "selected_maximum_timestep_seconds": float(
                arrays["selected_maximum_timestep"][0]
            ),
            "minimum_timestep_seconds": float(np.min(arrays["accepted_timesteps"])),
            "maximum_timestep_seconds": float(np.max(arrays["accepted_timesteps"])),
            "maximum_local_error_bound": float(np.max(arrays["local_error_bounds"])),
            "sum_local_error_bounds": float(np.sum(arrays["local_error_bounds"])),
            "maximum_audited_state_error": float(
                np.max(arrays["local_state_estimates"])
            ),
            "maximum_audited_Tier_I_error": float(
                np.max(arrays["local_Tier_I_estimates"])
            ),
            "maximum_audited_extraction_error": float(
                np.max(arrays["local_extraction_estimates"])
            ),
            "median_routine_step_wall_seconds": float(
                np.median(arrays["accepted_step_wall_seconds"][~arrays["audit_flags"]])
            ),
            "median_audit_step_wall_seconds": float(
                np.median(arrays["accepted_step_wall_seconds"][arrays["audit_flags"]])
            ),
        }
    )
    progress["reports"]["base"] = report
    _save_progress(progress)
    return report, arrays


def _extraction_direction(context, state, direction, relative_step):
    plus, plus_identity, plus_audit = _extraction_value(
        context, state + relative_step * direction
    )
    minus, minus_identity, minus_audit = _extraction_value(
        context, state - relative_step * direction
    )
    return (
        (plus - minus) / (2.0 * relative_step),
        max(plus_identity, minus_identity),
        np.maximum(plus_audit, minus_audit),
    )


def _initial_extraction() -> dict[str, np.ndarray]:
    return {
        "accepted_times": np.empty(0, dtype=float),
        "base_values": np.empty((0, 13), dtype=float),
        "anchor_values": np.empty((0, 13), dtype=float),
        "tangent_directions": np.empty((0, len(PROFILES), 13), dtype=float),
        "maximum_identity_defects": np.empty(0, dtype=float),
        "maximum_ledger_audits": np.empty((0, 4), dtype=float),
        "step_sweep_times": np.empty(0, dtype=float),
        "step_sweep_directions": np.empty(
            (0, len(EXTRACTION_JVP_STEP_SWEEP), 13), dtype=float
        ),
        "evaluation_wall_seconds": np.empty(0, dtype=float),
    }


def _cumulative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    for index in range(1, times.size):
        dt = float(times[index] - times[index - 1])
        result[index] = result[index - 1] + 0.5 * dt * (
            values[index - 1] + values[index]
        )
    return result


def _window_means(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    windows = ((0.010, 0.020), (0.016, 0.020))
    means = []
    for start, stop in windows:
        selected = (times >= start - 1.0e-15) & (times <= stop + 1.0e-15)
        window_times = times[selected]
        window_values = values[selected]
        if window_times.size < 2 or abs(window_times[0] - start) > 1.0e-15 or abs(
            window_times[-1] - stop
        ) > 1.0e-15:
            raise RuntimeError("c4e3 extraction mean window lacks exact endpoints")
        means.append(np.trapz(window_values, window_times, axis=0) / (stop - start))
    return np.asarray(means)


def _response_metrics(predicted, actual, scales) -> dict:
    metrics = h2b1.h2a3.h2a1.h1._response_metrics(predicted, actual, scales)
    metrics["discrepancy_fraction_of_observable_response"] = float(
        metrics["maximum_scaled_discrepancy"]
        / max(metrics["maximum_scaled_actual_response"], np.finfo(float).tiny)
    )
    return metrics


def _run_extraction(
    progress: dict,
    configuration: dict,
    base: dict[str, np.ndarray],
    tangent: dict[str, np.ndarray],
    anchor: dict[str, np.ndarray],
    extraction_scales: np.ndarray,
    manifest: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    arrays = _load_npz(EXTRACTION_PATH) if EXTRACTION_PATH.exists() else _initial_extraction()
    context = configuration["context"]
    start = int(progress["extraction_times_completed"])
    for index in range(start, base["accepted_times"].size):
        began = time.perf_counter()
        base_state = base["accepted_states"][index]
        anchor_state = anchor["anchor_states"][index]
        base_value, base_identity, base_audit = _extraction_value(context, base_state)
        anchor_value, anchor_identity, anchor_audit = _extraction_value(
            context, anchor_state
        )
        directions = []
        identity = max(base_identity, anchor_identity)
        audit = np.maximum(base_audit, anchor_audit)
        for direction in tangent["state_directions"][index]:
            value, direction_identity, direction_audit = _extraction_direction(
                context, base_state, direction, EXTRACTION_JVP_RELATIVE_STEP
            )
            directions.append(value)
            identity = max(identity, direction_identity)
            audit = np.maximum(audit, direction_audit)
        arrays["accepted_times"] = _append(
            arrays["accepted_times"], [base["accepted_times"][index]]
        )
        arrays["base_values"] = _append(arrays["base_values"], base_value[None, :])
        arrays["anchor_values"] = _append(
            arrays["anchor_values"], anchor_value[None, :]
        )
        arrays["tangent_directions"] = _append(
            arrays["tangent_directions"], np.asarray(directions)[None, ...]
        )
        arrays["maximum_identity_defects"] = _append(
            arrays["maximum_identity_defects"], [identity]
        )
        arrays["maximum_ledger_audits"] = _append(
            arrays["maximum_ledger_audits"], audit[None, :]
        )
        if index in (0, base["accepted_times"].size - 1):
            sweep = []
            for relative_step in EXTRACTION_JVP_STEP_SWEEP:
                value, direction_identity, direction_audit = _extraction_direction(
                    context,
                    base_state,
                    tangent["state_directions"][index, GENERIC_INDEX],
                    relative_step,
                )
                sweep.append(value)
                identity = max(identity, direction_identity)
                audit = np.maximum(audit, direction_audit)
            arrays["step_sweep_times"] = _append(
                arrays["step_sweep_times"], [base["accepted_times"][index]]
            )
            arrays["step_sweep_directions"] = _append(
                arrays["step_sweep_directions"], np.asarray(sweep)[None, ...]
            )
        wall = time.perf_counter() - began
        arrays["evaluation_wall_seconds"] = _append(
            arrays["evaluation_wall_seconds"], [wall]
        )
        progress["extraction_times_completed"] = index + 1
        np.savez_compressed(EXTRACTION_PATH, **arrays)
        _save_progress(progress)
        print(
            f"c4e3-extraction: {index + 1}/{base['accepted_times'].size} "
            f"t={base['accepted_times'][index]:.8e} wall={wall:.1f}s",
            flush=True,
        )
    actual = arrays["anchor_values"] - arrays["base_values"]
    predicted = arrays["tangent_directions"][:, GENERIC_INDEX]
    predicted_cumulative = _cumulative(predicted, arrays["accepted_times"])
    actual_cumulative = _cumulative(actual, arrays["accepted_times"])
    predicted_means = _window_means(predicted, arrays["accepted_times"])
    actual_means = _window_means(actual, arrays["accepted_times"])
    arrays["actual_generic_response"] = actual
    arrays["predicted_generic_response"] = predicted
    arrays["actual_generic_cumulative_response"] = actual_cumulative
    arrays["predicted_generic_cumulative_response"] = predicted_cumulative
    arrays["actual_generic_window_mean_response"] = actual_means
    arrays["predicted_generic_window_mean_response"] = predicted_means
    instantaneous = _response_metrics(predicted, actual, extraction_scales)
    cumulative = _response_metrics(
        predicted_cumulative, actual_cumulative, extraction_scales
    )
    means = _response_metrics(predicted_means, actual_means, extraction_scales)
    selected = EXTRACTION_JVP_STEP_SWEEP.index(EXTRACTION_JVP_RELATIVE_STEP)
    selected_sweep = arrays["step_sweep_directions"][:, selected]
    sweep_difference = 0.0
    for position in range(len(EXTRACTION_JVP_STEP_SWEEP)):
        difference = _scaled_difference(
            arrays["step_sweep_directions"][:, position],
            selected_sweep,
            extraction_scales,
        )
        response = float(np.max(np.abs(selected_sweep) / extraction_scales))
        sweep_difference = max(
            sweep_difference, difference / max(response, np.finfo(float).tiny)
        )
    maximum_audit = np.max(arrays["maximum_ledger_audits"], axis=0)
    gate = manifest["extraction_tangent_contract"]
    report = {
        "passed": bool(
            instantaneous["discrepancy_fraction_of_observable_response"]
            <= gate["maximum_generic_discrepancy_fraction_of_response"]
            and cumulative["discrepancy_fraction_of_observable_response"]
            <= gate["maximum_generic_discrepancy_fraction_of_response"]
            and means["discrepancy_fraction_of_observable_response"]
            <= gate["maximum_generic_discrepancy_fraction_of_response"]
            and sweep_difference
            <= gate["maximum_step_sensitivity_fraction_of_response"]
            and float(np.max(arrays["maximum_identity_defects"])) <= 1.0e-12
            and maximum_audit[0] <= 1.0e-12
            and maximum_audit[1] <= 1.0e-11
            and maximum_audit[2] <= 1.0e-12
            and int(maximum_audit[3]) == 0
        ),
        "instantaneous": instantaneous,
        "cumulative": cumulative,
        "window_mean": means,
        "maximum_step_sensitivity_fraction_of_response": sweep_difference,
        "maximum_identity_defect": float(
            np.max(arrays["maximum_identity_defects"])
        ),
        "maximum_shared_conservative_face_defect": float(maximum_audit[0]),
        "maximum_local_block_ledger_defect": float(maximum_audit[1]),
        "maximum_source_double_count_defect": float(maximum_audit[2]),
        "maximum_incoming_excision_characteristics": int(maximum_audit[3]),
        "median_evaluation_wall_seconds": float(
            np.median(arrays["evaluation_wall_seconds"])
        ),
    }
    progress["reports"]["extraction"] = report
    np.savez_compressed(EXTRACTION_PATH, **arrays)
    _save_progress(progress)
    return report, arrays


def _serialized_replays(configuration, frozen_tangent, base, anchor):
    return {
        "base": h2b1._serialized_last_step_replay(
            "base",
            configuration,
            frozen_tangent,
            base["accepted_states"],
            base["accepted_primitive_histories"],
            base["accepted_mapped_histories"],
            base["accepted_height_histories"],
            base["accepted_previous_timesteps"],
            base["accepted_timesteps"],
            base["accepted_times"],
            None,
        ),
        "anchor": h2b1._serialized_last_step_replay(
            "anchor",
            configuration,
            frozen_tangent,
            anchor["anchor_states"],
            anchor["anchor_primitive_histories"],
            anchor["anchor_mapped_histories"],
            anchor["anchor_height_histories"],
            anchor["anchor_previous_timesteps"],
            base["accepted_timesteps"],
            base["accepted_times"],
            anchor["anchor_predictors"][-1],
        ),
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
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


def main() -> int:
    _manifest_summary, manifest = _validate_parent()
    _patch_shared_modules()
    _migrate_cap_fallback_checkpoint_identity()
    progress = _progress()
    configuration = h2b1._configuration()
    print("c4e3: build middle frozen tangent", flush=True)
    frozen_tangent, setup_seconds = h2b1._build_frozen_tangent(configuration)
    parent = _parent_arrays()
    field_scales = np.asarray(parent["tangent__field_scales"], dtype=float)
    export_scales = np.asarray(parent["tangent__export_scales"], dtype=float)
    with np.load(extraction5.DECISIVE_ARRAYS, allow_pickle=False) as extraction_parent:
        extraction_scales = np.asarray(extraction_parent["export_scales"], dtype=float)
    contract, _strict = h2b1.h2a2.h2.g._controller_contracts()
    contract = dict(contract)
    contract["maximum_timestep_seconds"] = MAXIMUM_TIMESTEP_SECONDS
    base_report, base = _run_base(
        progress,
        configuration,
        frozen_tangent,
        field_scales,
        export_scales,
        extraction_scales,
        contract,
        manifest,
    )
    tangent_report, tangent = h2b1._run_tangent(progress, configuration, base)
    anchor_report, anchor = h2b1._run_anchor(
        progress,
        configuration,
        frozen_tangent,
        base,
        tangent,
        field_scales,
        export_scales,
        contract,
    )
    extraction_report, extraction = _run_extraction(
        progress,
        configuration,
        base,
        tangent,
        anchor,
        extraction_scales,
        manifest,
    )
    replays = _serialized_replays(configuration, frozen_tangent, base, anchor)
    replay_passed = all(
        item["checkpoint_roundtrip_bitwise"]
        and item["last_step_replay_bitwise"]
        and item["maximum_scaled_residual"] <= 1.0e-10
        for item in replays.values()
    )
    passed = bool(
        base_report["passed"]
        and tangent_report["passed"]
        and anchor_report["passed"]
        and extraction_report["passed"]
        and replay_passed
    )
    classification = (
        "optimized_middle_20ms_completion_passed_coarse_middle_checkpoint_analysis_authorized"
        if passed
        else "optimized_middle_20ms_completion_failed_fine_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b5c4e4_coarse_middle_20ms_checkpoint_analysis"
        if passed
        else "optimized_middle_completion_failure_localization_only"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "setup_wall_seconds": setup_seconds,
        "base": base_report,
        "tangent": tangent_report,
        "anchor": anchor_report,
        "extraction_tangent": extraction_report,
        "serialized_replays": replays,
        "coarse_middle_twenty_ms_checkpoint_analysis_authorized": passed,
        "fine_twenty_ms_propagation_authorized": False,
        "twenty_ms_spatial_checkpoint_certified": False,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    combined = {
        **{f"base__{key}": value for key, value in base.items()},
        **{f"tangent__{key}": value for key, value in tangent.items()},
        **{f"anchor__{key}": value for key, value in anchor.items()},
        **{f"extraction__{key}": value for key, value in extraction.items()},
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **combined)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "layout": h2b1.MIDDLE_LAYOUT,
            "profiles": PROFILES,
            "generic_profile": h2b1.GENERIC_PROFILE,
            "coupling_face": COUPLING_FACE,
            "extraction_face": EXTRACTION_FACE,
            "extraction_radius_rg": c4e1.c4e.EXTRACTION_RADIUS_RG,
            "output_target_microseconds": OUTPUT_TARGET_MICROSECONDS,
            "audit_target_microseconds": AUDIT_TARGET_MICROSECONDS,
            "controller": contract,
            "extraction_jvp_relative_step": EXTRACTION_JVP_RELATIVE_STEP,
        },
    )
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "optimized_manifest": _sha256(c4e2.MANIFEST_PATH),
                "middle_6ms_summary": _sha256(c4e1.SUMMARY_PATH),
                "middle_6ms_arrays": _sha256(c4e1.DECISIVE_ARRAYS),
                "five_ms_extraction_certificate": _sha256(extraction5.SUMMARY_PATH),
            },
            "implementation_source_hashes": _source_identity(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Optimized middle 20 ms completion WP10c9d6c7c3b5c4e3",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"The middle nonlinear base completed 6 to 20 ms in `{base_report['accepted_steps']}` accepted steps, including `{base_report['audited_steps']}` full-versus-two-half audits. The selected timestep cap was `{base_report['selected_maximum_timestep_seconds']:.6e} s`.",
                "",
                f"The maximum bounded local temporal error was `{base_report['maximum_local_error_bound']:.6e}` and the maximum nonlinear residual was `{float(np.max(base['step_maximum_scaled_residuals'])):.6e}`.",
                "",
                f"The generic tangent discrepancy was `{anchor_report['state']['discrepancy_fraction_of_observable_response']:.6e}` of the state response and `{anchor_report['instantaneous_Tier_I']['discrepancy_fraction_of_observable_response']:.6e}` of the Tier-I response.",
                "",
                f"For the certified extraction partition, instantaneous, cumulative, and window-mean tangent discrepancies were `{extraction_report['instantaneous']['discrepancy_fraction_of_observable_response']:.6e}`, `{extraction_report['cumulative']['discrepancy_fraction_of_observable_response']:.6e}`, and `{extraction_report['window_mean']['discrepancy_fraction_of_observable_response']:.6e}` of the nonlinear response.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
                "Fine propagation, 50 ms propagation, fixed-Q experiments, and reduced slow evolution remain blocked pending the coarse-middle spatial analysis.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
