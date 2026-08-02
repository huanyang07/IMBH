#!/usr/bin/env python3
"""Run the frozen coarse-layout nonlinear temporal-refinement screen."""

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
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402
import run_causal_inner_nonlinear_temporal_refinement_manifest_wp10c9d6c7c3b3a as c3b3a  # noqa: E402
import run_causal_inner_physical_background_nonlinear_readiness_manifest_wp10c9d6c7c3a1 as c3a1  # noqa: E402

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
WORK_PACKAGE = "WP10c9d6c7c3b3b1"
ANALYZED_BASE_COMMIT = "ecab6edec3fa760655cfc7ab8b7488afe26cca39"
ANALYZED_BASE_PARENT = "bf1ed9e6a11a7e687f8544ae6daae6c3e1cd9203"
ANALYZED_BASE_TREE = "cc17b0c89faae43c01f4e4506d56c36916735069"

LAYOUT = c3b3a.COARSE_LAYOUT
PROFILES = (c3b3a.PRIMARY_PROFILE, c3b3a.SECONDARY_PROFILE)
VARIANT_MULTIPLIER = 1.0
TIMESTEP_LEVELS_SECONDS = np.asarray(
    c3b3a.TIMESTEP_LEVELS_SECONDS,
    dtype=float,
)
REFINED_TIMESTEPS_SECONDS = TIMESTEP_LEVELS_SECONDS[1:]
HORIZON_SECONDS = c3b3a.HORIZON_SECONDS
COMMON_OUTPUT_TIMES_SECONDS = np.asarray(
    c3b3a.COMMON_OUTPUT_TIMES_SECONDS,
    dtype=float,
)
OBSERVABLE_NAMES = c3b2b.OBSERVABLE_NAMES

MAXIMUM_SCALED_RESIDUAL = c3b1a.MAXIMUM_SCALED_RESIDUAL
MAXIMUM_SCALED_ALGEBRAIC_RESIDUAL = (
    c3b1a.MAXIMUM_SCALED_ALGEBRAIC_RESIDUAL
)
MAXIMUM_DISCRETE_LEDGER_DEFECT = c3b1a.MAXIMUM_DISCRETE_LEDGER_DEFECT
MAXIMUM_SCALED_PRIMITIVE_CHANGE = c3b1a.MAXIMUM_SCALED_PRIMITIVE_CHANGE
MAXIMUM_H_OVER_R = c3b1a.MAXIMUM_H_OVER_R
MINIMUM_SCATTERING_OPTICAL_DEPTH = c3b1a.MINIMUM_SCATTERING_OPTICAL_DEPTH
MINIMUM_RECONSTRUCTION_FACTOR = c3b1a.MINIMUM_RECONSTRUCTION_FACTOR
MAXIMUM_MAPPED_ENDPOINT_PATH_CLOSURE = (
    c3b1a.MAXIMUM_MAPPED_ENDPOINT_PATH_CLOSURE
)

ARTIFACT = (
    "causal_inner_nonlinear_temporal_coarse_screen_"
    "wp10c9d6c7c3b3b1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_temporal_coarse_screen_"
    "wp10c9d6c7c3b3b1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_temporal_coarse_screen_"
    "wp10c9d6c7c3b3b1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_TEMPORAL_COARSE_SCREEN_"
    "WP10C9D6C7C3B3B1_2026-08-01.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

MANIFEST_DIRECTORY = c3b3a.CANONICAL_DIRECTORY
PARENT_PILOT_DIRECTORY = c3b2b.CANONICAL_DIRECTORY
C3A1_DIRECTORY = c3a1.CANONICAL_DIRECTORY
C7A_DIRECTORY = c3b2b.C7A_DIRECTORY
PREFLIGHT_DIRECTORY = c3b2b.STEP4_DIRECTORY

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
        c3b1a.THIS_MODULE,
        c3b3a.THIS_RUNNER,
    )
    return {
        path: _sha256(ROOT / path)
        for path in paths
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict]:
    manifest_summary = _read_json(MANIFEST_DIRECTORY / "summary.json")
    manifest = _read_json(
        MANIFEST_DIRECTORY / "temporal_refinement_manifest.json"
    )
    if (
        not manifest_summary["passed"]
        or not manifest_summary["coarse_temporal_screen_authorized"]
        or manifest_summary["authorized_next"]
        != "WP10c9d6c7c3b3b1_coarse_inward_outward_temporal_screen"
        or manifest["classification"]
        != "nonlinear_temporal_refinement_manifest_frozen_"
        "coarse_temporal_screen_authorized"
        or manifest["propagation_executed"]
    ):
        raise RuntimeError("c3b3a coarse-screen authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3b3b1 analyzed identity changed")
    return manifest_summary, manifest


def _case_id(profile: str) -> str:
    return f"{profile}__p1"


def _task_id(timestep: float, trajectory: str) -> str:
    timestep_token = f"{float(timestep):.1e}".replace(".", "p")
    timestep_token = timestep_token.replace("-", "m")
    return f"dt_{timestep_token}__{trajectory}"


def _scaled_linear_predictor(
    configuration: dict,
    tangent,
    state: np.ndarray,
    timestep_seconds: float,
) -> np.ndarray:
    columns = np.asarray(configuration["columns"], dtype=float)
    scaled_difference = (
        (np.asarray(state, dtype=float) - configuration["base"]).ravel()
        / columns
    )
    scaled_rate = (
        np.asarray(tangent.scaled_base_rate_per_s, dtype=float)
        + np.asarray(tangent.scaled_generator_per_s, dtype=float)
        @ scaled_difference
    )
    return (
        float(timestep_seconds) * columns * scaled_rate
    ).reshape(state.shape)


def _trajectory(
    configuration: dict,
    tangent,
    initial_state: np.ndarray,
    timestep_seconds: float,
    trajectory_id: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = configuration["context"]
    timestep = float(timestep_seconds)
    step_count = int(round(HORIZON_SECONDS / timestep))
    current = np.array(initial_state, copy=True)
    history = None
    states = [np.array(current, copy=True)]
    residuals = []
    algebraic = []
    ledgers = []
    mapped_closures = []
    reconstruction = []
    incoming = []
    iterations = []
    evaluations = []
    checkpoint = None
    restart_roundtrip = True
    split_replay = True
    provenance = {
        "work_package": WORK_PACKAGE,
        "layout": LAYOUT,
        "trajectory_id": trajectory_id,
        "timestep_seconds": timestep,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
    }
    started = time.perf_counter()
    for index in range(step_count):
        order = 1 if index == 0 else 2
        predictor = (
            _scaled_linear_predictor(
                configuration,
                tangent,
                current,
                timestep,
            )
            if index == 0
            else None
        )
        print(
            f"c3b3b1: {trajectory_id} dt={timestep:.1e} "
            f"step {index + 1}/{step_count}",
            flush=True,
        )
        step = advance_causal_five_field_monolithic_bdf(
            context,
            current,
            timestep,
            tangent,
            order=order,
            history=history,
            initial_primitive_increment=predictor,
            residual_tolerance=MAXIMUM_SCALED_RESIDUAL,
            ledger_tolerance=MAXIMUM_DISCRETE_LEDGER_DEFECT,
            maximum_scaled_primitive_change=(
                MAXIMUM_SCALED_PRIMITIVE_CHANGE
            ),
        )
        residuals.append(step.maximum_scaled_residual)
        algebraic.append(step.maximum_scaled_algebraic_residual)
        ledgers.append(step.maximum_discrete_ledger_defect)
        mapped_closures.append(
            step.evaluation.maximum_mapped_endpoint_path_closure_defect
        )
        reconstruction.append(step.minimum_path_reconstruction_factor)
        incoming.append(step.incoming_excision_characteristics)
        iterations.append(step.iterations)
        evaluations.append(step.function_evaluations)
        if not step.accepted or step.history is None:
            break
        current = np.array(step.primitive_charts, copy=True)
        history = step.history
        states.append(np.array(current, copy=True))
        if index == 1:
            checkpoint = CausalFiveFieldMonolithicBDFRestart(
                primitive_charts=np.array(current, copy=True),
                history=history,
                elapsed_time_seconds=2.0 * timestep,
                completed_steps=2,
                next_order=2,
                provenance=provenance,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "restart.npz"
                save_causal_five_field_monolithic_bdf_restart(
                    path,
                    context,
                    checkpoint,
                )
                restored = load_causal_five_field_monolithic_bdf_restart(
                    path,
                    context,
                    expected_provenance=provenance,
                )
            restart_roundtrip = (
                causal_five_field_monolithic_bdf_restarts_equal(
                    checkpoint,
                    restored,
                )
            )
        if index == 2 and checkpoint is not None:
            replay = advance_causal_five_field_monolithic_bdf(
                context,
                checkpoint.primitive_charts,
                timestep,
                tangent,
                order=2,
                history=checkpoint.history,
                residual_tolerance=MAXIMUM_SCALED_RESIDUAL,
                ledger_tolerance=MAXIMUM_DISCRETE_LEDGER_DEFECT,
                maximum_scaled_primitive_change=(
                    MAXIMUM_SCALED_PRIMITIVE_CHANGE
                ),
            )
            split_replay = bool(
                replay.accepted
                and replay.history is not None
                and np.array_equal(
                    replay.primitive_charts,
                    step.primitive_charts,
                )
                and np.array_equal(
                    replay.history.previous_primitive_increment,
                    step.history.previous_primitive_increment,
                )
                and np.array_equal(
                    replay.history.previous_mapped_storage_increment,
                    step.history.previous_mapped_storage_increment,
                )
                and np.array_equal(
                    replay.history
                    .previous_responsive_height_storage_increment,
                    step.history
                    .previous_responsive_height_storage_increment,
                )
            )

    state_audit = c3b1a._state_audit(context, current)
    completed = len(states) == step_count + 1
    passed = bool(
        completed
        and max(residuals, default=float("inf"))
        <= MAXIMUM_SCALED_RESIDUAL
        and max(algebraic, default=float("inf"))
        <= MAXIMUM_SCALED_ALGEBRAIC_RESIDUAL
        and max(ledgers, default=float("inf"))
        <= MAXIMUM_DISCRETE_LEDGER_DEFECT
        and max(mapped_closures, default=float("inf"))
        <= MAXIMUM_MAPPED_ENDPOINT_PATH_CLOSURE
        and min(reconstruction, default=0.0)
        >= MINIMUM_RECONSTRUCTION_FACTOR
        and max(incoming, default=1) == 0
        and state_audit["maximum_h_over_r"] <= MAXIMUM_H_OVER_R
        and state_audit["minimum_scattering_optical_depth"]
        > MINIMUM_SCATTERING_OPTICAL_DEPTH
        and state_audit["minimum_reconstruction_factor"]
        >= MINIMUM_RECONSTRUCTION_FACTOR
        and restart_roundtrip
        and split_replay
    )
    return (
        {
            "trajectory_id": trajectory_id,
            "layout": LAYOUT,
            "timestep_seconds": timestep,
            "completed_steps": len(states) - 1,
            "expected_steps": step_count,
            "maximum_scaled_residual": max(residuals, default=None),
            "maximum_scaled_algebraic_residual": max(
                algebraic,
                default=None,
            ),
            "maximum_discrete_ledger_defect": max(
                ledgers,
                default=None,
            ),
            "maximum_mapped_endpoint_path_closure_defect": max(
                mapped_closures,
                default=None,
            ),
            "minimum_path_reconstruction_factor": min(
                reconstruction,
                default=None,
            ),
            "maximum_incoming_excision_characteristics": max(
                incoming,
                default=None,
            ),
            "maximum_iterations": max(iterations, default=None),
            "function_evaluations": sum(evaluations),
            "checkpoint_roundtrip_bitwise": restart_roundtrip,
            "split_restart_replay_bitwise": split_replay,
            "final_state_audit": state_audit,
            "elapsed_seconds": time.perf_counter() - started,
            "passed": passed,
        },
        {
            "states": np.asarray(states),
            "scaled_residuals": np.asarray(residuals),
            "algebraic_residuals": np.asarray(algebraic),
            "ledger_defects": np.asarray(ledgers),
            "mapped_endpoint_path_closures": np.asarray(mapped_closures),
            "reconstruction_factors": np.asarray(reconstruction),
            "incoming_excision": np.asarray(incoming),
        },
    )


def _export_history(context, states: np.ndarray) -> tuple[np.ndarray, dict]:
    values = []
    maxima = {
        "maximum_local_block_ledger_defect": 0.0,
        "maximum_source_double_count_defect": 0.0,
        "maximum_shared_conservative_face_defect": 0.0,
        "maximum_split_closure_defect": 0.0,
        "maximum_incoming_excision_characteristics": 0,
    }
    for index, state in enumerate(states):
        started = time.perf_counter()
        observable, audit = c3b2b._direct_observable(
            context,
            state,
            c3b2b.COUPLING_PARENT_FACE,
        )
        values.append(np.asarray(observable, dtype=float))
        for key in (
            "local_block_ledger_defect",
            "source_double_count_defect",
            "shared_conservative_face_defect",
            "split_closure_defect",
        ):
            target = f"maximum_{key}"
            maxima[target] = max(maxima[target], float(audit[key]))
        maxima["maximum_incoming_excision_characteristics"] = max(
            maxima["maximum_incoming_excision_characteristics"],
            int(audit["incoming_excision_characteristics"]),
        )
        print(
            f"c3b3b1: direct export {index + 1}/{len(states)} "
            f"{time.perf_counter() - started:.2f}s",
            flush=True,
        )
    maxima["passed"] = bool(
        maxima["maximum_local_block_ledger_defect"]
        <= MAXIMUM_DISCRETE_LEDGER_DEFECT
        and maxima["maximum_source_double_count_defect"]
        <= MAXIMUM_DISCRETE_LEDGER_DEFECT
        and maxima["maximum_shared_conservative_face_defect"]
        <= MAXIMUM_DISCRETE_LEDGER_DEFECT
        and maxima["maximum_split_closure_defect"]
        <= MAXIMUM_MAPPED_ENDPOINT_PATH_CLOSURE
        and maxima["maximum_incoming_excision_characteristics"] == 0
    )
    return np.asarray(values), maxima


def _load_progress() -> tuple[dict, dict[str, np.ndarray]]:
    if not CHECKPOINT_JSON.exists():
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "work_package": WORK_PACKAGE,
                "analyzed_base_commit": ANALYZED_BASE_COMMIT,
                "source_identity": _source_identity(),
                "completed_tasks": [],
                "trajectory_reports": [],
                "export_audits": {},
                "failed": False,
            },
            {},
        )
    progress = _read_json(CHECKPOINT_JSON)
    if (
        progress.get("work_package") != WORK_PACKAGE
        or progress.get("analyzed_base_commit") != ANALYZED_BASE_COMMIT
        or progress.get("source_identity") != _source_identity()
    ):
        raise RuntimeError("saved c3b3b1 progress belongs to different code")
    return progress, _load_npz(CHECKPOINT_ARRAYS)


def _save_progress(progress: dict, arrays: dict[str, np.ndarray]) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CHECKPOINT_JSON, progress)
    np.savez_compressed(CHECKPOINT_ARRAYS, **arrays)


def _run_refined_trajectories(
    configuration: dict,
    tangent,
) -> tuple[dict, dict[str, np.ndarray]]:
    progress, arrays = _load_progress()
    completed = set(progress["completed_tasks"])
    packet_arrays = _load_npz(
        c3a1.C7C0_DIRECTORY / "decisive_arrays.npz"
    )
    initial_states = {"base": np.asarray(configuration["base"], dtype=float)}
    for profile in PROFILES:
        packet = np.asarray(
            packet_arrays[f"{profile}__{LAYOUT}__{c3a1.PROFILE_KIND}"],
            dtype=float,
        )
        initial_states[_case_id(profile)] = initial_states["base"] + packet

    for timestep in REFINED_TIMESTEPS_SECONDS:
        for trajectory_id, initial_state in initial_states.items():
            task = _task_id(float(timestep), trajectory_id)
            if task in completed:
                continue
            report, trajectory_arrays = _trajectory(
                configuration,
                tangent,
                initial_state,
                float(timestep),
                trajectory_id,
            )
            prefix = task
            for name, values in trajectory_arrays.items():
                arrays[f"{prefix}__{name}"] = values
            if report["passed"]:
                exports, audit = _export_history(
                    configuration["context"],
                    trajectory_arrays["states"],
                )
                arrays[f"{prefix}__direct_exports"] = exports
                progress["export_audits"][task] = audit
                report["export_audit_passed"] = audit["passed"]
                report["passed"] = bool(report["passed"] and audit["passed"])
            else:
                report["export_audit_passed"] = False
            progress["trajectory_reports"].append(report)
            completed.add(task)
            progress["completed_tasks"] = sorted(completed)
            progress["failed"] = bool(
                progress["failed"] or not report["passed"]
            )
            _save_progress(progress, arrays)
            if not report["passed"]:
                return progress, arrays
    return progress, arrays


def _common_indices(timestep_seconds: float) -> np.ndarray:
    indices = np.rint(
        COMMON_OUTPUT_TIMES_SECONDS / float(timestep_seconds)
    ).astype(np.int64)
    if not np.allclose(
        indices * float(timestep_seconds),
        COMMON_OUTPUT_TIMES_SECONDS,
        rtol=0.0,
        atol=1.0e-18,
    ):
        raise RuntimeError("common output time is not on temporal level")
    return indices


def _native_cumulative(values: np.ndarray, timestep: float) -> np.ndarray:
    history = np.asarray(values, dtype=float)
    result = np.zeros_like(history)
    result[1:] = np.cumsum(
        0.5 * float(timestep) * (history[:-1] + history[1:]),
        axis=0,
    )
    return result


def _temporal_gates(manifest: dict) -> dict:
    source = manifest["temporal_binding_contract"]["gates"]
    return {
        "minimum_rms_order": source["minimum_rms_order"],
        "minimum_maximum_order": source["minimum_maximum_order"],
        "minimum_significant_component_order": source[
            "minimum_significant_component_order"
        ],
        "maximum_fine_normalized_difference": source[
            "maximum_fine_normalized_temporal_difference"
        ],
        "minimum_history_cosine": source["minimum_history_cosine"],
        "minimum_refinement_error_cosine": source[
            "minimum_observable_refinement_error_cosine"
        ],
        "minimum_relative_activity": source["minimum_relative_activity"],
        "maximum_selected_step_richardson_error": source[
            "maximum_selected_step_richardson_error"
        ],
        "observability_factor": source["observability_factor"],
    }


def _selected_step_error(
    coarse: np.ndarray,
    medium: np.ndarray,
    scales: np.ndarray,
) -> float:
    normalized = (
        np.asarray(medium, dtype=float) - np.asarray(coarse, dtype=float)
    ) / np.asarray(scales, dtype=float)
    return float((4.0 / 3.0) * np.max(np.abs(normalized)))


def _state_metric(
    histories: tuple[np.ndarray, np.ndarray, np.ndarray],
    configuration: dict,
    field_scales: np.ndarray,
    gates: dict,
    numerical_floor: float,
) -> dict:
    raw = c3b2b._state_metrics(
        histories,
        configuration["context"].grid,
        field_scales,
        gates,
    )
    normalized = tuple(
        values / field_scales[None, None, :] for values in histories
    )
    coarse_medium = normalized[1] - normalized[0]
    medium_fine = normalized[2] - normalized[1]
    coarse_norm = float(np.sqrt(np.mean(coarse_medium**2)))
    fine_norm = float(np.sqrt(np.mean(medium_fine**2)))
    threshold = gates["observability_factor"] * numerical_floor
    observable = bool(coarse_norm > threshold and fine_norm > threshold)
    selected_error = _selected_step_error(
        histories[0],
        histories[1],
        field_scales[None, None, :],
    )
    upper_bound_passed = bool(
        raw["maximum_fine_normalized_difference"]
        <= gates["maximum_fine_normalized_difference"]
        and selected_error
        <= gates["maximum_selected_step_richardson_error"]
        and raw["history_cosine"] >= gates["minimum_history_cosine"]
    )
    passed = bool((raw["passed"] if observable else True) and upper_bound_passed)
    return {
        **raw,
        "raw_contract_passed": raw["passed"],
        "refinement_error_observable": observable,
        "numerical_uncertainty_floor": numerical_floor,
        "observability_threshold": threshold,
        "selected_step_richardson_error": selected_error,
        "upper_bound_route_used": not observable,
        "passed": passed,
    }


def _export_metric(
    histories: tuple[np.ndarray, np.ndarray, np.ndarray],
    scales: np.ndarray,
    gates: dict,
    numerical_floor: float,
) -> dict:
    metrics = c3b2b._packet_metrics(histories, scales, gates)
    payload = c3b2b._metric_payload(metrics)
    normalized = tuple(values / scales[None, :] for values in histories)
    response = np.max(np.abs(np.asarray(normalized)), axis=(0, 1))
    significant = response >= gates["minimum_relative_activity"]
    coarse_medium = normalized[1][:, significant] - normalized[0][
        :, significant
    ]
    medium_fine = normalized[2][:, significant] - normalized[1][
        :, significant
    ]
    coarse_norm = float(np.sqrt(np.mean(coarse_medium**2)))
    fine_norm = float(np.sqrt(np.mean(medium_fine**2)))
    threshold = gates["observability_factor"] * numerical_floor
    observable = bool(coarse_norm > threshold and fine_norm > threshold)
    selected_error = _selected_step_error(
        histories[0],
        histories[1],
        scales[None, :],
    )
    upper_bound_passed = bool(
        payload["maximum_fine_normalized_difference"]
        <= gates["maximum_fine_normalized_difference"]
        and selected_error
        <= gates["maximum_selected_step_richardson_error"]
        and payload["history_cosine"] >= gates["minimum_history_cosine"]
    )
    passed = bool((payload["passed"] if observable else True) and upper_bound_passed)
    return {
        **payload,
        "raw_contract_passed": payload["passed"],
        "refinement_error_observable": observable,
        "numerical_uncertainty_floor": numerical_floor,
        "observability_threshold": threshold,
        "selected_step_richardson_error": selected_error,
        "upper_bound_route_used": not observable,
        "passed": passed,
    }


def _response_histories(
    arrays: dict[str, np.ndarray],
    parent_arrays: dict[str, np.ndarray],
    case: str,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    state_histories = [
        np.asarray(
            parent_arrays[f"{LAYOUT}__{case}__state_response"],
            dtype=float,
        )
    ]
    instantaneous = [
        np.asarray(
            parent_arrays[
                f"{LAYOUT}__{case}__instantaneous_export_response"
            ],
            dtype=float,
        )
    ]
    cumulative = [
        np.asarray(
            parent_arrays[
                f"{LAYOUT}__{case}__cumulative_export_response"
            ],
            dtype=float,
        )
    ]
    for timestep in REFINED_TIMESTEPS_SECONDS:
        base_task = _task_id(float(timestep), "base")
        case_task = _task_id(float(timestep), case)
        indices = _common_indices(float(timestep))
        base_states = arrays[f"{base_task}__states"]
        case_states = arrays[f"{case_task}__states"]
        state_histories.append(case_states[indices] - base_states[indices])
        base_exports = arrays[f"{base_task}__direct_exports"]
        case_exports = arrays[f"{case_task}__direct_exports"]
        response = case_exports - base_exports
        instantaneous.append(response[indices])
        cumulative.append(
            _native_cumulative(response, float(timestep))[indices]
        )
    return (
        tuple(state_histories),
        tuple(instantaneous),
        tuple(cumulative),
    )


def _analyze(
    progress: dict,
    arrays: dict[str, np.ndarray],
    configuration: dict,
    manifest: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    parent_arrays = _load_npz(PARENT_PILOT_DIRECTORY / "decisive_arrays.npz")
    gates = _temporal_gates(manifest)
    field_scales = np.asarray(parent_arrays["field_scales"], dtype=float)
    observable_scales = np.asarray(
        parent_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    reports = list(progress["trajectory_reports"])
    preflight = _read_json(PREFLIGHT_DIRECTORY / "summary.json")
    method_floor = max(
        [
            float(preflight["maximum_scaled_residual"]),
            float(preflight["maximum_discrete_ledger_defect"]),
        ]
        + [
            float(report["maximum_scaled_residual"])
            for report in reports
        ]
        + [
            float(report["maximum_discrete_ledger_defect"])
            for report in reports
        ]
    )
    case_reports = {}
    decisive = {
        "timestep_levels_seconds": TIMESTEP_LEVELS_SECONDS,
        "common_output_times_seconds": COMMON_OUTPUT_TIMES_SECONDS,
        "field_scales": field_scales,
        "fixed_physical_observable_scales": observable_scales,
    }
    all_passed = not progress["failed"]
    for profile in PROFILES:
        case = _case_id(profile)
        state, instantaneous, cumulative = _response_histories(
            arrays,
            parent_arrays,
            case,
        )
        state_metric = _state_metric(
            state,
            configuration,
            field_scales,
            gates,
            method_floor,
        )
        instantaneous_metric = _export_metric(
            instantaneous,
            observable_scales,
            gates,
            method_floor,
        )
        cumulative_metric = _export_metric(
            cumulative,
            observable_scales * HORIZON_SECONDS,
            gates,
            method_floor,
        )
        passed = bool(
            state_metric["passed"]
            and instantaneous_metric["passed"]
            and cumulative_metric["passed"]
        )
        case_reports[case] = {
            "state": state_metric,
            "instantaneous_exports": instantaneous_metric,
            "cumulative_exports": cumulative_metric,
            "passed": passed,
        }
        all_passed = bool(all_passed and passed)
        for level, label in enumerate(("h", "h2", "h4")):
            decisive[f"{case}__{label}__state_response"] = state[level]
            decisive[
                f"{case}__{label}__instantaneous_export_response"
            ] = instantaneous[level]
            decisive[
                f"{case}__{label}__cumulative_export_response"
            ] = cumulative[level]
    return (
        {
            "passed": all_passed,
            "numerical_uncertainty_floor": method_floor,
            "case_reports": case_reports,
            "all_refined_trajectory_methods_passed": all(
                report["passed"] for report in reports
            ),
            "trajectory_reports": reports,
            "maximum_scaled_residual": max(
                report["maximum_scaled_residual"] for report in reports
            ),
            "maximum_discrete_ledger_defect": max(
                report["maximum_discrete_ledger_defect"]
                for report in reports
            ),
            "all_checkpoint_roundtrips_bitwise": all(
                report["checkpoint_roundtrip_bitwise"]
                for report in reports
            ),
            "all_split_restart_replays_bitwise": all(
                report["split_restart_replay_bitwise"]
                for report in reports
            ),
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
            fieldnames=[
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ],
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
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _report(summary: dict) -> str:
    lines = [
        "# Nonlinear coarse temporal screen WP10c9d6c7c3b3b1",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        "The unchanged coarse embedded operator was compared at "
        "`dt=1e-5/5e-6/2.5e-6 s` through the common `4e-5 s` horizon. "
        "Each level used its own BDF1 startup and BDF2 history.",
        "",
        "## Results",
        "",
    ]
    for case, report in summary["temporal_screen"]["case_reports"].items():
        state = report["state"]
        instant = report["instantaneous_exports"]
        cumulative = report["cumulative_exports"]
        lines.extend(
            [
                f"### `{case}`",
                "",
                "- state RMS/max/component order: "
                f"`{state['observed_rms_order']:.6f}` / "
                f"`{state['observed_maximum_order']:.6f}` / "
                f"`{state['minimum_significant_component_order']:.6f}`",
                "- instantaneous export RMS/max/component order: "
                f"`{instant['observed_rms_order']:.6f}` / "
                f"`{instant['observed_maximum_order']:.6f}` / "
                f"`{instant['minimum_significant_component_order']:.6f}`",
                "- cumulative export RMS/max/component order: "
                f"`{cumulative['observed_rms_order']:.6f}` / "
                f"`{cumulative['observed_maximum_order']:.6f}` / "
                f"`{cumulative['minimum_significant_component_order']:.6f}`",
                "- selected-step Richardson errors "
                "(state/instantaneous/cumulative): "
                f"`{state['selected_step_richardson_error']:.3e}` / "
                f"`{instant['selected_step_richardson_error']:.3e}` / "
                f"`{cumulative['selected_step_richardson_error']:.3e}`",
                f"- result: `{'pass' if report['passed'] else 'fail'}`",
                "",
            ]
        )
    screen = summary["temporal_screen"]
    lines.extend(
        [
            "## Method and uncertainty",
            "",
            "- maximum scaled nonlinear residual: "
            f"`{screen['maximum_scaled_residual']:.3e}`",
            "- maximum discrete ledger defect: "
            f"`{screen['maximum_discrete_ledger_defect']:.3e}`",
            "- all checkpoint roundtrips bitwise: "
            f"`{screen['all_checkpoint_roundtrips_bitwise']}`",
            "- all split/restart replays bitwise: "
            f"`{screen['all_split_restart_replays_bitwise']}`",
            "- conservative numerical uncertainty floor: "
            f"`{screen['numerical_uncertainty_floor']:.3e}`",
            "",
            "## Authorized next",
            "",
            f"`{summary['authorized_next']}`",
            "",
            "Long nonlinear evolution, fixed-Q experiments and reduced "
            "slow evolution remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def _package(
    parent: dict,
    manifest: dict,
    screen: dict,
    decisive: dict[str, np.ndarray],
    progress_arrays: dict[str, np.ndarray],
) -> int:
    passed = bool(screen["passed"])
    classification = (
        "coarse_inward_outward_nonlinear_temporal_screen_certified_"
        "middle_primary_confirmation_authorized"
        if passed
        else "coarse_nonlinear_temporal_screen_failed_"
        "duration_extension_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b3b2_middle_primary_temporal_confirmation"
        if passed
        else "WP10c9d6c7c3b3b1_temporal_failure_localization"
    )
    gates = _temporal_gates(manifest)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": LAYOUT,
        "profiles": list(PROFILES),
        "variant_multiplier": VARIANT_MULTIPLIER,
        "timestep_levels_seconds": TIMESTEP_LEVELS_SECONDS.tolist(),
        "horizon_seconds": HORIZON_SECONDS,
        "common_output_times_seconds": (
            COMMON_OUTPUT_TIMES_SECONDS.tolist()
        ),
        "temporal_gates": gates,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    combined_arrays = {**progress_arrays, **decisive}
    np.savez_compressed(DECISIVE_ARRAYS, **combined_arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "operator_changed": False,
        "production_defaults_changed": False,
        "parent_classification": parent["classification"],
        "temporal_screen": screen,
        "middle_primary_temporal_confirmation_authorized": passed,
        "temporal_convergence_certified": False,
        "meaningfully_nonlinear_dynamics_certified": False,
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(config),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values)
            for name, values in combined_arrays.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    input_paths = {
        "manifest_summary": MANIFEST_DIRECTORY / "summary.json",
        "manifest_contract": MANIFEST_DIRECTORY
        / "temporal_refinement_manifest.json",
        "parent_pilot_arrays": PARENT_PILOT_DIRECTORY
        / "decisive_arrays.npz",
        "physical_profile_arrays": c3a1.C7C0_DIRECTORY
        / "decisive_arrays.npz",
        "preflight_summary": PREFLIGHT_DIRECTORY / "summary.json",
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "command": (
                "PYTHONPATH=src "
                "/Users/huanyang/.cache/codex-runtimes/"
                "codex-primary-runtime/dependencies/python/bin/python3 "
                + THIS_RUNNER
            ),
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
            "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
            "implementation_parent_commit": _git_value(
                "rev-parse", "HEAD"
            ),
            "implementation_parent_tree_sha": _git_value(
                "rev-parse", "HEAD^{tree}"
            ),
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
    names = (
        "config.json",
        "summary.json",
        "provenance.json",
        "decisive_arrays.npz",
    )
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
    print(f"c3b3b1: build tangent {LAYOUT}", flush=True)
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    progress, progress_arrays = _run_refined_trajectories(
        configuration,
        tangent,
    )
    if progress["failed"]:
        screen = {
            "passed": False,
            "numerical_uncertainty_floor": None,
            "case_reports": {},
            "all_refined_trajectory_methods_passed": False,
            "trajectory_reports": progress["trajectory_reports"],
            "maximum_scaled_residual": max(
                report["maximum_scaled_residual"]
                for report in progress["trajectory_reports"]
            ),
            "maximum_discrete_ledger_defect": max(
                report["maximum_discrete_ledger_defect"]
                for report in progress["trajectory_reports"]
            ),
            "all_checkpoint_roundtrips_bitwise": all(
                report["checkpoint_roundtrip_bitwise"]
                for report in progress["trajectory_reports"]
            ),
            "all_split_restart_replays_bitwise": all(
                report["split_restart_replay_bitwise"]
                for report in progress["trajectory_reports"]
            ),
        }
        decisive = {}
    else:
        screen, decisive = _analyze(
            progress,
            progress_arrays,
            configuration,
            manifest,
        )
    return _package(
        parent,
        manifest,
        screen,
        decisive,
        progress_arrays,
    )


if __name__ == "__main__":
    raise SystemExit(main())
