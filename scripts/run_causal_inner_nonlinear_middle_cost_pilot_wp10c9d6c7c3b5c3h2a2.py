#!/usr/bin/env python3
"""Run the resumable 0.2 ms middle-layout cost and science pilot."""

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
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_middle_tangent_hardening_audit_wp10c9d6c7c3b5c3h2a1 as h2a1  # noqa: E402
import run_causal_inner_nonlinear_middle_cost_bounded_anchor_manifest_wp10c9d6c7c3b5c3h2 as h2  # noqa: E402
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_duration_controller_validation_wp10c9d6c7c3b5b as controller  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistory,
    CausalFiveFieldMonolithicBDFRestart,
    advance_causal_five_field_monolithic_bdf,
    causal_five_field_monolithic_bdf_history_direction,
    causal_five_field_monolithic_bdf_history_from_interval,
    causal_five_field_monolithic_bdf_restarts_equal,
    causal_five_field_monolithic_discrete_export_directions,
    causal_five_field_monolithic_discrete_step_matrix,
    causal_five_field_monolithic_discrete_tangent_step,
    causal_five_field_monolithic_frozen_tangent,
    load_causal_five_field_monolithic_bdf_restart,
    save_causal_five_field_monolithic_bdf_restart,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2a2"
ANALYZED_BASE_COMMIT = "69a8e4b9ab9e56d08e32ad94803696b8d153d852"
ANALYZED_BASE_PARENT = "dfd01c145bd1ff01df5ac69736430abf7f2525f2"
ANALYZED_BASE_TREE = "7b30292ab8d46226145cbd7ed642a7e00c924aba"

MIDDLE_LAYOUT = h2.MIDDLE_LAYOUT
GENERIC_PROFILE = h2.GENERIC_PROFILE
PROFILES = tuple(h2.PROFILES)
COUPLING_FACE = int(h2.ACTIVE_COUPLING_FACE)
START_SECONDS = 4.0e-5
STOP_SECONDS = 2.0e-4
OUTPUT_TIMES = np.asarray((START_SECONDS, STOP_SECONDS), dtype=float)
INITIAL_PREVIOUS_TIMESTEP_SECONDS = 1.0e-5
MINIMUM_PROJECTION_STEPS = 5
PROJECTION_SAFETY_FACTOR = 2.0
PACKAGING_ALLOWANCE_SECONDS = 900.0
FUTURE_AUDIT_JVP_COUNT = 3
FUTURE_REPLAY_COUNT = 2

ARTIFACT = "causal_inner_nonlinear_middle_cost_pilot_wp10c9d6c7c3b5c3h2a2"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_cost_pilot_"
    "wp10c9d6c7c3b5c3h2a2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_cost_pilot_"
    "wp10c9d6c7c3b5c3h2a2.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_COST_PILOT_"
    "WP10C9D6C7C3B5C3H2A2_2026-08-06.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CONTROLLER_RELATIVE = (
    "scripts/run_causal_inner_nonlinear_second_duration_rung_"
    "wp10c9d6c7c3b5c2.py"
)
MODULE_RELATIVE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_discrete_tangent.py"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
PROGRESS_PATH = CHECKPOINT_DIRECTORY / "progress.json"
BASE_ARRAYS_PATH = CHECKPOINT_DIRECTORY / "base_stage.npz"
TANGENT_ARRAYS_PATH = CHECKPOINT_DIRECTORY / "tangent_stage.npz"
ANCHOR_ARRAYS_PATH = CHECKPOINT_DIRECTORY / "anchor_stage.npz"


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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    return {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST, CONTROLLER_RELATIVE, MODULE_RELATIVE)
        if (ROOT / path).exists()
    }


def _validate_parent() -> None:
    parent = _read_json(h2a1.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["middle_0p2ms_cost_pilot_authorized"]
        or parent["middle_1ms_propagation_authorized"]
    ):
        raise RuntimeError("h2a2 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2a2 analyzed identity changed")


def _progress() -> dict:
    identity = _source_identity()
    if not PROGRESS_PATH.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "source_identity": identity,
            "completed_stages": [],
            "reports": {},
        }
    progress = _read_json(PROGRESS_PATH)
    if (
        progress.get("work_package") != WORK_PACKAGE
        or progress.get("analyzed_base_commit") != ANALYZED_BASE_COMMIT
        or progress.get("source_identity") != identity
    ):
        raise RuntimeError("saved h2a2 progress belongs to different code")
    return progress


def _save_progress(progress: dict) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(PROGRESS_PATH, progress)


def _history(
    primitive: np.ndarray,
    mapped: np.ndarray,
    height: np.ndarray,
    previous_timestep: float,
) -> CausalFiveFieldMonolithicBDFHistory:
    return CausalFiveFieldMonolithicBDFHistory(
        previous_primitive_increment=np.asarray(primitive, dtype=float),
        previous_mapped_storage_increment=np.asarray(mapped, dtype=float),
        previous_responsive_height_storage_increment=np.asarray(
            height,
            dtype=float,
        ),
        previous_timestep_seconds=float(previous_timestep),
    ).validated(n_cells=primitive.shape[0])


def _short_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_arrays = _load_npz(h2.h1.b1a.DECISIVE_ARRAYS)
    spatial_arrays = _load_npz(h2.h1.b4b3.DECISIVE_ARRAYS)
    scale_arrays = _load_npz(h2.h1.c3d.DECISIVE_ARRAYS)
    base = np.asarray(base_arrays[f"{MIDDLE_LAYOUT}__states"], dtype=float)
    task = f"{MIDDLE_LAYOUT}__{GENERIC_PROFILE}__p1__dt_1e-5"
    anchor = np.asarray(spatial_arrays[f"{task}__states"], dtype=float)
    return (
        base,
        anchor,
        np.asarray(scale_arrays["field_scales"], dtype=float),
        np.asarray(scale_arrays["export_scales"], dtype=float),
    )


def _build_frozen_tangent(configuration: dict):
    began = time.perf_counter()
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    return tangent, time.perf_counter() - began


def _base_stage(
    configuration: dict,
    frozen_tangent,
    base_short: np.ndarray,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    main_contract: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = configuration["context"]
    history = causal_five_field_monolithic_bdf_history_from_interval(
        context,
        base_short[3],
        base_short[4],
        INITIAL_PREVIOUS_TIMESTEP_SECONDS,
    )
    began = time.perf_counter()
    segment = c2._controller_segment(
        configuration,
        frozen_tangent,
        np.array(base_short[4], copy=True),
        history,
        START_SECONDS,
        main_contract["initial_timestep_seconds"],
        field_scales,
        export_scales,
        COUPLING_FACE,
        main_contract,
        output_times=OUTPUT_TIMES,
        stop_time=STOP_SECONDS,
        include_initial_output=True,
        record_accepted_steps=True,
        log_prefix="h2a2-base",
    )
    elapsed = time.perf_counter() - began
    passed = c2._segment_passed(
        segment,
        main_contract,
        main_contract["error_estimator"][
            "short_horizon_sum_of_accepted_error_estimates"
        ],
    )
    sample_sufficient = bool(
        segment["accepted_timesteps"].size >= MINIMUM_PROJECTION_STEPS
    )
    report = {
        "passed": bool(passed and sample_sufficient),
        "method_gates_passed": passed,
        "projection_sample_sufficient": sample_sufficient,
        "accepted_steps": int(segment["accepted_timesteps"].size),
        "rejected_attempts": int(np.sum(segment["retries"])),
        "minimum_timestep_seconds": float(np.min(segment["accepted_timesteps"])),
        "maximum_timestep_seconds": float(np.max(segment["accepted_timesteps"])),
        "next_candidate_timestep_seconds": segment["next_candidate_timestep"],
        "maximum_local_error_estimate": float(
            np.max(segment["local_error_estimates"])
        ),
        "sum_local_error_estimates": float(
            np.sum(segment["local_error_estimates"])
        ),
        "wall_seconds": elapsed,
        "median_accepted_step_wall_seconds": float(
            np.median(segment["accepted_step_wall_seconds"])
        ),
        "maximum_scaled_residual": max(
            item["maximum_scaled_residual"] for item in segment["step_records"]
        ),
        "maximum_discrete_ledger_defect": max(
            item["maximum_discrete_ledger_defect"]
            for item in segment["step_records"]
        ),
        "maximum_mapped_endpoint_path_closure_defect": max(
            item["maximum_mapped_endpoint_path_closure_defect"]
            for item in segment["step_records"]
        ),
        "minimum_path_reconstruction_factor": min(
            item["minimum_path_reconstruction_factor"]
            for item in segment["step_records"]
        ),
        "maximum_incoming_excision_characteristics": max(
            item["incoming_excision_characteristics"]
            for item in segment["step_records"]
        ),
        "maximum_export_ledger_defect": segment["maximum_export_ledger_defect"],
    }
    arrays = {
        "accepted_times": segment["accepted_times"],
        "accepted_timesteps": segment["accepted_timesteps"],
        "accepted_states": segment["accepted_states"],
        "accepted_primitive_histories": segment["accepted_primitive_histories"],
        "accepted_mapped_histories": segment["accepted_mapped_histories"],
        "accepted_height_histories": segment["accepted_height_histories"],
        "accepted_previous_timesteps": segment["accepted_previous_timesteps"],
        "accepted_step_wall_seconds": segment["accepted_step_wall_seconds"],
        "local_state_estimates": segment["local_state_estimates"],
        "local_export_estimates": segment["local_export_estimates"],
        "local_error_estimates": segment["local_error_estimates"],
        "retries": segment["retries"],
        "output_times": segment["output_times"],
        "output_states": segment["output_states"],
        "output_exports": segment["output_exports"],
        "next_candidate_timestep": np.asarray(
            [segment["next_candidate_timestep"]],
            dtype=float,
        ),
    }
    return report, arrays


def _tangent_stage(
    configuration: dict,
    base_short: np.ndarray,
    anchor_short: np.ndarray,
    base_arrays: dict[str, np.ndarray],
    field_scales: np.ndarray,
    export_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = configuration["context"]
    columns = configuration["columns"]
    rows = configuration["rows"]
    base_states = base_arrays["accepted_states"]
    timesteps = base_arrays["accepted_timesteps"]
    previous_timesteps = base_arrays["accepted_previous_timesteps"]
    initial_response_old = anchor_short[3] - base_short[3]
    initial_response = anchor_short[4] - base_short[4]

    began = time.perf_counter()
    history_matrix = causal_five_field_monolithic_discrete_step_matrix(
        context,
        base_short[3],
        base_short[4],
        INITIAL_PREVIOUS_TIMESTEP_SECONDS,
        INITIAL_PREVIOUS_TIMESTEP_SECONDS,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    initial_matrix_seconds = time.perf_counter() - began
    history_direction = causal_five_field_monolithic_bdf_history_direction(
        context,
        base_short[3],
        base_short[4],
        initial_response_old[None, :, :],
        initial_response[None, :, :],
        analytic_step_matrix=history_matrix,
    )
    direction = initial_response[None, :, :]
    state_directions = [np.array(initial_response, copy=True)]
    export_direction, initial_export_audit = (
        causal_five_field_monolithic_discrete_export_directions(
            history_matrix,
            direction,
            COUPLING_FACE,
        )
    )
    export_directions = [np.array(export_direction[0], copy=True)]
    matrix_seconds = []
    step_seconds = []
    jvp_defects = []
    linear_defects = []
    component_defects = []
    incoming = []
    export_ledgers = [initial_export_audit.active_prefix_ledger_defect]
    export_telescoping = [
        initial_export_audit.conservative_transport_telescoping_defect
    ]
    step_ratios = timesteps / previous_timesteps[:-1]
    nonunit = np.flatnonzero(np.abs(step_ratios - 1.0) > 1.0e-14)
    audit_indices = {0, int(timesteps.size - 1)}
    if nonunit.size:
        audit_indices.add(int(nonunit[0]))
    audit_flags = []
    for index, dt in enumerate(timesteps):
        base_history = _history(
            base_arrays["accepted_primitive_histories"][index],
            base_arrays["accepted_mapped_histories"][index],
            base_arrays["accepted_height_histories"][index],
            previous_timesteps[index],
        )
        began = time.perf_counter()
        matrix = causal_five_field_monolithic_discrete_step_matrix(
            context,
            base_states[index],
            base_states[index + 1],
            float(dt),
            float(previous_timesteps[index]),
            primitive_column_scales=columns,
            conservation_row_scales=rows,
        )
        matrix_seconds.append(time.perf_counter() - began)
        audit = index in audit_indices
        began = time.perf_counter()
        step = causal_five_field_monolithic_discrete_tangent_step(
            context,
            base_states[index],
            base_states[index + 1],
            float(dt),
            base_history,
            direction,
            history_direction,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            analytic_step_matrix=matrix,
            audit_complete_residual=audit,
        )
        step_seconds.append(time.perf_counter() - began)
        audit_flags.append(audit)
        direction = step.new_primitive_directions
        history_direction = step.new_history_directions
        state_directions.append(np.array(direction[0], copy=True))
        export_direction, export_audit = (
            causal_five_field_monolithic_discrete_export_directions(
                matrix,
                direction,
                COUPLING_FACE,
            )
        )
        export_directions.append(np.array(export_direction[0], copy=True))
        if np.isfinite(step.maximum_step_matrix_jvp_relative_defect):
            jvp_defects.append(step.maximum_step_matrix_jvp_relative_defect)
        linear_defects.append(step.maximum_linear_solve_relative_defect)
        component_defects.append(matrix.maximum_component_closure_defect)
        incoming.append(matrix.incoming_excision_characteristics)
        export_ledgers.append(export_audit.active_prefix_ledger_defect)
        export_telescoping.append(
            export_audit.conservative_transport_telescoping_defect
        )
        print(
            f"h2a2-tangent: {index + 1}/{timesteps.size} "
            f"t={base_arrays['accepted_times'][index + 1]:.8e} "
            f"matrix={matrix_seconds[-1]:.1f}s step={step_seconds[-1]:.1f}s "
            f"audit={audit}",
            flush=True,
        )

    state_array = np.asarray(state_directions, dtype=float)
    export_array = np.asarray(export_directions, dtype=float)
    report = {
        "passed": bool(
            max(jvp_defects, default=0.0)
            <= h2a1.GATES[
                "maximum_internal_discrete_residual_jvp_relative_defect"
            ]
            and max(linear_defects, default=0.0)
            <= h2a1.GATES["maximum_linear_solve_relative_defect"]
            and max(component_defects, default=0.0)
            <= h2a1.GATES["maximum_matrix_component_closure_defect"]
            and max(incoming, default=0) == 0
            and max(export_ledgers, default=0.0)
            <= h2a1.GATES["maximum_export_active_prefix_ledger_defect"]
            and max(export_telescoping, default=0.0)
            <= h2a1.GATES["maximum_export_transport_telescoping_defect"]
        ),
        "initial_history_matrix_wall_seconds": initial_matrix_seconds,
        "accepted_steps": int(timesteps.size),
        "audit_step_indices": sorted(audit_indices),
        "maximum_step_matrix_jvp_relative_defect": max(jvp_defects, default=0.0),
        "maximum_linear_solve_relative_defect": max(linear_defects, default=0.0),
        "maximum_matrix_component_closure_defect": max(
            component_defects,
            default=0.0,
        ),
        "maximum_incoming_excision_characteristics": max(incoming, default=0),
        "maximum_export_active_prefix_ledger_defect": max(
            export_ledgers,
            default=0.0,
        ),
        "maximum_export_transport_telescoping_defect": max(
            export_telescoping,
            default=0.0,
        ),
        "matrix_assembly_wall_seconds": matrix_seconds,
        "block_step_wall_seconds": step_seconds,
        "routine_block_step_median_wall_seconds": float(
            np.median(
                [
                    value
                    for value, audited in zip(step_seconds, audit_flags, strict=True)
                    if not audited
                ]
                or [h2.MIDDLE_CALIBRATION["block_step_wall_seconds"]]
            )
        ),
    }
    arrays = {
        "state_directions": state_array,
        "export_directions": export_array,
        "matrix_assembly_wall_seconds": np.asarray(matrix_seconds),
        "block_step_wall_seconds": np.asarray(step_seconds),
        "audit_flags": np.asarray(audit_flags, dtype=bool),
        "step_ratios": step_ratios,
        "field_scales": field_scales,
        "export_scales": export_scales,
    }
    return report, arrays


def _anchor_stage(
    configuration: dict,
    frozen_tangent,
    base_short: np.ndarray,
    anchor_short: np.ndarray,
    base_arrays: dict[str, np.ndarray],
    tangent_arrays: dict[str, np.ndarray],
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    main_contract: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = configuration["context"]
    state = np.array(anchor_short[4], copy=True)
    history = causal_five_field_monolithic_bdf_history_from_interval(
        context,
        anchor_short[3],
        anchor_short[4],
        INITIAL_PREVIOUS_TIMESTEP_SECONDS,
    )
    base_states = base_arrays["accepted_states"]
    timesteps = base_arrays["accepted_timesteps"]
    tangent_states = tangent_arrays["state_directions"]
    anchor_states = [np.array(state, copy=True)]
    anchor_primitive_histories = [
        np.array(history.previous_primitive_increment, copy=True)
    ]
    anchor_mapped_histories = [
        np.array(history.previous_mapped_storage_increment, copy=True)
    ]
    anchor_height_histories = [
        np.array(
            history.previous_responsive_height_storage_increment,
            copy=True,
        )
    ]
    anchor_previous_timesteps = [float(history.previous_timestep_seconds)]
    anchor_step_seconds = []
    anchor_predictors = []
    sampled_state_errors = []
    sampled_export_errors = []
    step_records = []
    base_exports = []
    anchor_exports = []
    maximum_export_ledger = 0.0
    maximum_export_incoming = 0
    for initial_state, collection in (
        (base_states[0], base_exports),
        (state, anchor_exports),
    ):
        value, ledger, incoming = controller._export_value(
            context,
            initial_state,
            COUPLING_FACE,
        )
        collection.append(np.asarray(value, dtype=float))
        maximum_export_ledger = max(maximum_export_ledger, ledger)
        maximum_export_incoming = max(maximum_export_incoming, incoming)

    sampled_indices = {0, 1}
    for index, dt in enumerate(timesteps):
        predictor = (
            base_states[index + 1]
            + tangent_states[index + 1]
            - state
        )
        anchor_predictors.append(np.array(predictor, copy=True))
        began = time.perf_counter()
        full = advance_causal_five_field_monolithic_bdf(
            context,
            state,
            float(dt),
            frozen_tangent,
            order=2,
            history=history,
            initial_primitive_increment=predictor,
            residual_tolerance=1.0e-10,
            ledger_tolerance=1.0e-12,
            maximum_scaled_primitive_change=5.0e-3,
        )
        step_records.append(controller._step_record(full))
        if not controller._step_passed(full, main_contract) or full.history is None:
            raise RuntimeError("middle pilot generic anchor full step failed")
        if index in sampled_indices:
            half_first = advance_causal_five_field_monolithic_bdf(
                context,
                state,
                0.5 * float(dt),
                frozen_tangent,
                order=2,
                history=history,
                residual_tolerance=1.0e-10,
                ledger_tolerance=1.0e-12,
                maximum_scaled_primitive_change=5.0e-3,
            )
            if half_first.history is None:
                raise RuntimeError("middle pilot anchor first half has no history")
            half_second = advance_causal_five_field_monolithic_bdf(
                context,
                half_first.primitive_charts,
                0.5 * float(dt),
                frozen_tangent,
                order=2,
                history=half_first.history,
                residual_tolerance=1.0e-10,
                ledger_tolerance=1.0e-12,
                maximum_scaled_primitive_change=5.0e-3,
            )
            step_records.extend(
                [
                    controller._step_record(half_first),
                    controller._step_record(half_second),
                ]
            )
            if not (
                controller._step_passed(half_first, main_contract)
                and controller._step_passed(half_second, main_contract)
            ):
                raise RuntimeError("middle pilot anchor sampled halves failed")
            full_export, full_ledger, full_incoming = controller._export_value(
                context,
                full.primitive_charts,
                COUPLING_FACE,
            )
            fine_export, fine_ledger, fine_incoming = controller._export_value(
                context,
                half_second.primitive_charts,
                COUPLING_FACE,
            )
            maximum_export_ledger = max(
                maximum_export_ledger,
                full_ledger,
                fine_ledger,
            )
            maximum_export_incoming = max(
                maximum_export_incoming,
                full_incoming,
                fine_incoming,
            )
            sampled_state_errors.append(
                controller._state_estimate(
                    full.primitive_charts,
                    half_second.primitive_charts,
                    field_scales,
                )
            )
            sampled_export_errors.append(
                controller._export_estimate(
                    full_export,
                    fine_export,
                    export_scales,
                )
            )
        anchor_step_seconds.append(time.perf_counter() - began)
        state = np.array(full.primitive_charts, copy=True)
        history = full.history
        anchor_states.append(np.array(state, copy=True))
        anchor_primitive_histories.append(
            np.array(history.previous_primitive_increment, copy=True)
        )
        anchor_mapped_histories.append(
            np.array(history.previous_mapped_storage_increment, copy=True)
        )
        anchor_height_histories.append(
            np.array(
                history.previous_responsive_height_storage_increment,
                copy=True,
            )
        )
        anchor_previous_timesteps.append(float(history.previous_timestep_seconds))
        base_value, base_ledger, base_incoming = controller._export_value(
            context,
            base_states[index + 1],
            COUPLING_FACE,
        )
        anchor_value, anchor_ledger, anchor_incoming = controller._export_value(
            context,
            state,
            COUPLING_FACE,
        )
        base_exports.append(np.asarray(base_value, dtype=float))
        anchor_exports.append(np.asarray(anchor_value, dtype=float))
        maximum_export_ledger = max(
            maximum_export_ledger,
            base_ledger,
            anchor_ledger,
        )
        maximum_export_incoming = max(
            maximum_export_incoming,
            base_incoming,
            anchor_incoming,
        )
        print(
            f"h2a2-anchor: {index + 1}/{timesteps.size} "
            f"t={base_arrays['accepted_times'][index + 1]:.8e} "
            f"wall={anchor_step_seconds[-1]:.1f}s "
            f"sampled={index in sampled_indices}",
            flush=True,
        )

    anchor_states_array = np.asarray(anchor_states, dtype=float)
    base_exports_array = np.asarray(base_exports, dtype=float)
    anchor_exports_array = np.asarray(anchor_exports, dtype=float)
    actual_state_response = anchor_states_array - base_states
    actual_export_response = anchor_exports_array - base_exports_array
    state_metrics = h2.h1._response_metrics(
        tangent_states,
        actual_state_response,
        field_scales,
    )
    export_metrics = h2.h1._response_metrics(
        tangent_arrays["export_directions"],
        actual_export_response,
        export_scales,
    )
    for metrics in (state_metrics, export_metrics):
        metrics["discrepancy_fraction_of_observable_response"] = float(
            metrics["maximum_scaled_discrepancy"]
            / max(
                metrics["maximum_scaled_actual_response"],
                np.finfo(float).tiny,
            )
        )
    gates = h2a1.GATES
    readiness = h2.h1.b1a._state_audit(context, anchor_states_array[-1])
    passed = bool(
        state_metrics["maximum_scaled_discrepancy"]
        <= gates["maximum_absolute_scaled_state_discrepancy"]
        and export_metrics["maximum_scaled_discrepancy"]
        <= gates["maximum_absolute_scaled_Tier_I_discrepancy"]
        and state_metrics["discrepancy_fraction_of_observable_response"]
        <= gates["maximum_discrepancy_fraction_of_observable_response"]
        and export_metrics["discrepancy_fraction_of_observable_response"]
        <= gates["maximum_discrepancy_fraction_of_observable_response"]
        and state_metrics["history_cosine"] >= gates["minimum_state_history_cosine"]
        and export_metrics["history_cosine"]
        >= gates["minimum_Tier_I_history_cosine"]
        and max(sampled_state_errors, default=0.0)
        <= main_contract["error_estimator"]["local_tolerance"]
        and max(sampled_export_errors, default=0.0)
        <= main_contract["error_estimator"]["local_tolerance"]
        and max(
            item["maximum_scaled_residual"] for item in step_records
        )
        <= 1.0e-10
        and max(
            item["maximum_discrete_ledger_defect"] for item in step_records
        )
        <= 1.0e-12
        and maximum_export_ledger <= 1.0e-9
        and maximum_export_incoming == 0
        and readiness["maximum_h_over_r"] <= 0.12
        and readiness["minimum_scattering_optical_depth"] > 1.0
        and readiness["minimum_reconstruction_factor"] >= 1.0
    )
    report = {
        "passed": passed,
        "state": state_metrics,
        "instantaneous_Tier_I": export_metrics,
        "sampled_step_indices": sorted(sampled_indices),
        "maximum_sampled_state_error_estimate": max(
            sampled_state_errors,
            default=0.0,
        ),
        "maximum_sampled_export_error_estimate": max(
            sampled_export_errors,
            default=0.0,
        ),
        "maximum_scaled_residual": max(
            item["maximum_scaled_residual"] for item in step_records
        ),
        "maximum_discrete_ledger_defect": max(
            item["maximum_discrete_ledger_defect"] for item in step_records
        ),
        "maximum_mapped_endpoint_path_closure_defect": max(
            item["maximum_mapped_endpoint_path_closure_defect"]
            for item in step_records
        ),
        "minimum_path_reconstruction_factor": min(
            item["minimum_path_reconstruction_factor"] for item in step_records
        ),
        "maximum_incoming_excision_characteristics": max(
            item["incoming_excision_characteristics"] for item in step_records
        ),
        "maximum_export_ledger_defect": maximum_export_ledger,
        "maximum_export_incoming_characteristics": maximum_export_incoming,
        "final_state_audit": readiness,
        "anchor_step_wall_seconds": anchor_step_seconds,
        "median_unsampled_anchor_step_wall_seconds": float(
            np.median(
                [
                    value
                    for index, value in enumerate(anchor_step_seconds)
                    if index not in sampled_indices
                ]
                or anchor_step_seconds
            )
        ),
        "median_sampled_anchor_step_wall_seconds": float(
            np.median(
                [
                    value
                    for index, value in enumerate(anchor_step_seconds)
                    if index in sampled_indices
                ]
            )
        ),
    }
    arrays = {
        "anchor_states": anchor_states_array,
        "anchor_primitive_histories": np.asarray(anchor_primitive_histories),
        "anchor_mapped_histories": np.asarray(anchor_mapped_histories),
        "anchor_height_histories": np.asarray(anchor_height_histories),
        "anchor_previous_timesteps": np.asarray(anchor_previous_timesteps),
        "anchor_predictors": np.asarray(anchor_predictors),
        "anchor_step_wall_seconds": np.asarray(anchor_step_seconds),
        "base_exports": base_exports_array,
        "anchor_exports": anchor_exports_array,
        "actual_state_response": actual_state_response,
        "actual_export_response": actual_export_response,
        "sampled_state_error_estimates": np.asarray(sampled_state_errors),
        "sampled_export_error_estimates": np.asarray(sampled_export_errors),
    }
    return report, arrays


def _serialized_last_step_replay(
    label: str,
    configuration: dict,
    frozen_tangent,
    states: np.ndarray,
    primitive_histories: np.ndarray,
    mapped_histories: np.ndarray,
    height_histories: np.ndarray,
    previous_timesteps: np.ndarray,
    timesteps: np.ndarray,
    predictor: np.ndarray | None,
) -> dict:
    context = configuration["context"]
    index = int(timesteps.size - 1)
    history = _history(
        primitive_histories[index],
        mapped_histories[index],
        height_histories[index],
        previous_timesteps[index],
    )
    checkpoint = CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(states[index], copy=True),
        history=history,
        elapsed_time_seconds=float(STOP_SECONDS - timesteps[index]),
        completed_steps=index,
        next_order=2,
        provenance={"work_package": WORK_PACKAGE, "label": label},
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{label}.npz"
        save_causal_five_field_monolithic_bdf_restart(path, context, checkpoint)
        restored = load_causal_five_field_monolithic_bdf_restart(
            path,
            context,
            expected_provenance=checkpoint.provenance,
        )
    roundtrip = causal_five_field_monolithic_bdf_restarts_equal(
        checkpoint,
        restored,
    )
    began = time.perf_counter()
    result = advance_causal_five_field_monolithic_bdf(
        context,
        restored.primitive_charts,
        float(timesteps[index]),
        frozen_tangent,
        order=2,
        history=restored.history,
        initial_primitive_increment=predictor,
        residual_tolerance=1.0e-10,
        ledger_tolerance=1.0e-12,
        maximum_scaled_primitive_change=5.0e-3,
    )
    wall = time.perf_counter() - began
    target_history = _history(
        primitive_histories[index + 1],
        mapped_histories[index + 1],
        height_histories[index + 1],
        previous_timesteps[index + 1],
    )
    bitwise = bool(
        result.history is not None
        and np.array_equal(result.primitive_charts, states[index + 1])
        and np.array_equal(
            result.history.previous_primitive_increment,
            target_history.previous_primitive_increment,
        )
        and np.array_equal(
            result.history.previous_mapped_storage_increment,
            target_history.previous_mapped_storage_increment,
        )
        and np.array_equal(
            result.history.previous_responsive_height_storage_increment,
            target_history.previous_responsive_height_storage_increment,
        )
        and result.history.previous_timestep_seconds
        == target_history.previous_timestep_seconds
    )
    return {
        "checkpoint_roundtrip_bitwise": roundtrip,
        "last_step_replay_bitwise": bitwise,
        "wall_seconds": wall,
        "maximum_scaled_residual": result.maximum_scaled_residual,
    }


def _simulate_remaining_steps(
    elapsed: float,
    previous_timestep: float,
    candidate_timestep: float,
    local_error: float,
    contract: dict,
) -> int:
    targets = np.asarray(
        (1.0e-3, 2.0e-3, 2.4e-3, 2.8e-3, 3.2e-3, 3.6e-3, 4.0e-3, 4.4e-3, 4.8e-3, 5.0e-3),
        dtype=float,
    )
    count = 0
    target_index = 0
    while elapsed < 5.0e-3 - 1.0e-15:
        while target_index < targets.size and targets[target_index] <= elapsed + 1.0e-15:
            target_index += 1
        target = float(targets[target_index]) if target_index < targets.size else 5.0e-3
        dt = min(
            candidate_timestep,
            target - elapsed,
            contract["maximum_BDF2_step_ratio"] * previous_timestep,
            contract["maximum_timestep_seconds"],
        )
        elapsed = target if abs(elapsed + dt - target) <= 1.0e-15 else elapsed + dt
        previous_timestep = dt
        candidate_timestep = controller._next_timestep(dt, local_error, contract)
        count += 1
    return count


def _projection(
    setup_seconds: float,
    base_report: dict,
    base_arrays: dict[str, np.ndarray],
    tangent_report: dict,
    anchor_report: dict,
    replay_reports: dict,
    main_contract: dict,
) -> dict:
    remaining = _simulate_remaining_steps(
        STOP_SECONDS,
        float(base_arrays["accepted_timesteps"][-1]),
        float(base_arrays["next_candidate_timestep"][0]),
        float(base_arrays["local_error_estimates"][-1]),
        main_contract,
    )
    total_steps = int(base_report["accepted_steps"] + remaining)
    base_seconds = total_steps * base_report["median_accepted_step_wall_seconds"]
    anchor_seconds = (
        total_steps * anchor_report["median_unsampled_anchor_step_wall_seconds"]
        + 3.0
        * (
            anchor_report["median_sampled_anchor_step_wall_seconds"]
            - anchor_report["median_unsampled_anchor_step_wall_seconds"]
        )
    )
    matrix_median = float(
        np.median(tangent_report["matrix_assembly_wall_seconds"])
    )
    tangent_seconds = total_steps * (
        matrix_median + tangent_report["routine_block_step_median_wall_seconds"]
    )
    audited_steps = tangent_report["audit_step_indices"]
    audited_wall = [
        tangent_report["block_step_wall_seconds"][index]
        for index in audited_steps
    ]
    routine = tangent_report["routine_block_step_median_wall_seconds"]
    audit_extra = FUTURE_AUDIT_JVP_COUNT * max(
        float(np.median(audited_wall)) - routine,
        0.0,
    )
    replay_median = float(
        np.median([item["wall_seconds"] for item in replay_reports.values()])
    )
    replay_seconds = FUTURE_REPLAY_COUNT * replay_median
    raw = (
        setup_seconds
        + base_seconds
        + anchor_seconds
        + tangent_seconds
        + audit_extra
        + replay_seconds
        + PACKAGING_ALLOWANCE_SECONDS
    )
    projected = PROJECTION_SAFETY_FACTOR * raw
    projected_hours = projected / 3600.0
    tier = (
        "automatic_continuation"
        if projected_hours <= 24.0
        else "optimization_review"
        if projected_hours <= 48.0
        else "explicit_cost_benefit_decision"
    )
    return {
        "pilot_accepted_steps": base_report["accepted_steps"],
        "simulated_remaining_steps": remaining,
        "projected_total_steps": total_steps,
        "setup_seconds": setup_seconds,
        "projected_base_seconds_before_safety": base_seconds,
        "projected_anchor_seconds_before_safety": anchor_seconds,
        "projected_tangent_seconds_before_safety": tangent_seconds,
        "projected_JVP_audit_seconds_before_safety": audit_extra,
        "projected_replay_seconds_before_safety": replay_seconds,
        "packaging_allowance_seconds_before_safety": PACKAGING_ALLOWANCE_SECONDS,
        "safety_factor": PROJECTION_SAFETY_FACTOR,
        "projected_total_wall_seconds": projected,
        "projected_total_wall_hours": projected_hours,
        "resource_tier": tier,
        "cost_projection_is_not_a_scientific_gate": True,
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


def _report(summary: dict) -> str:
    projection = summary["cost_projection"]
    anchor = summary["anchor"]
    return "\n".join(
        (
            "# Middle 0.2 ms cost pilot WP10c9d6c7c3b5c3h2a2",
            "",
            "## Classification",
            "",
            f"`{summary['classification']}`",
            "",
            f"The middle nonlinear base, generic nonlinear anchor, and five-profile discrete-tangent block reached 0.2 ms. The tangent/anchor state discrepancy is `{anchor['state']['maximum_scaled_discrepancy']:.6e}` and the Tier-I discrepancy is `{anchor['instantaneous_Tier_I']['maximum_scaled_discrepancy']:.6e}`.",
            "",
            f"The measured cost model projects `{projection['projected_total_wall_hours']:.2f}` wall hours through 5 ms after the frozen factor-two safety allowance, placing the campaign in the `{projection['resource_tier']}` tier. Cost alone is not a scientific rejection.",
            "",
            "A pass authorizes only a fresh definitions-only middle continuation manifest. Fine work, the 5 ms spatial certificate, fixed-Q experiments, and reduced slow evolution remain blocked.",
            "",
        )
    )


def main() -> int:
    _validate_parent()
    progress = _progress()
    base_short, anchor_short, field_scales, export_scales = _short_inputs()
    configuration = h2.h1.b1a._configurations()[MIDDLE_LAYOUT]
    print("h2a2: build frozen nonlinear tangent", flush=True)
    frozen_tangent, setup_seconds = _build_frozen_tangent(configuration)
    main_contract, _strict_contract = h2.g._controller_contracts()

    if "base" not in progress["completed_stages"]:
        report, arrays = _base_stage(
            configuration,
            frozen_tangent,
            base_short,
            field_scales,
            export_scales,
            main_contract,
        )
        CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(BASE_ARRAYS_PATH, **arrays)
        progress["reports"]["base"] = report
        progress["completed_stages"].append("base")
        _save_progress(progress)
    base_arrays = _load_npz(BASE_ARRAYS_PATH)
    if not progress["reports"]["base"]["passed"]:
        raise RuntimeError("middle pilot base stage failed")

    if "tangent" not in progress["completed_stages"]:
        report, arrays = _tangent_stage(
            configuration,
            base_short,
            anchor_short,
            base_arrays,
            field_scales,
            export_scales,
        )
        np.savez_compressed(TANGENT_ARRAYS_PATH, **arrays)
        progress["reports"]["tangent"] = report
        progress["completed_stages"].append("tangent")
        _save_progress(progress)
    tangent_arrays = _load_npz(TANGENT_ARRAYS_PATH)
    if not progress["reports"]["tangent"]["passed"]:
        raise RuntimeError("middle pilot tangent stage failed")

    if "anchor" not in progress["completed_stages"]:
        report, arrays = _anchor_stage(
            configuration,
            frozen_tangent,
            base_short,
            anchor_short,
            base_arrays,
            tangent_arrays,
            field_scales,
            export_scales,
            main_contract,
        )
        np.savez_compressed(ANCHOR_ARRAYS_PATH, **arrays)
        progress["reports"]["anchor"] = report
        progress["completed_stages"].append("anchor")
        _save_progress(progress)
    anchor_arrays = _load_npz(ANCHOR_ARRAYS_PATH)
    if not progress["reports"]["anchor"]["passed"]:
        raise RuntimeError("middle pilot anchor stage failed")

    replay_reports = {
        "base": _serialized_last_step_replay(
            "base",
            configuration,
            frozen_tangent,
            base_arrays["accepted_states"],
            base_arrays["accepted_primitive_histories"],
            base_arrays["accepted_mapped_histories"],
            base_arrays["accepted_height_histories"],
            base_arrays["accepted_previous_timesteps"],
            base_arrays["accepted_timesteps"],
            None,
        ),
        "anchor": _serialized_last_step_replay(
            "anchor",
            configuration,
            frozen_tangent,
            anchor_arrays["anchor_states"],
            anchor_arrays["anchor_primitive_histories"],
            anchor_arrays["anchor_mapped_histories"],
            anchor_arrays["anchor_height_histories"],
            anchor_arrays["anchor_previous_timesteps"],
            base_arrays["accepted_timesteps"],
            anchor_arrays["anchor_predictors"][-1],
        ),
    }
    projection = _projection(
        setup_seconds,
        progress["reports"]["base"],
        base_arrays,
        progress["reports"]["tangent"],
        progress["reports"]["anchor"],
        replay_reports,
        main_contract,
    )
    replay_passed = all(
        report["checkpoint_roundtrip_bitwise"]
        and report["last_step_replay_bitwise"]
        and report["maximum_scaled_residual"] <= 1.0e-10
        for report in replay_reports.values()
    )
    passed = bool(
        progress["reports"]["base"]["passed"]
        and progress["reports"]["tangent"]["passed"]
        and progress["reports"]["anchor"]["passed"]
        and replay_passed
    )
    tier = projection["resource_tier"]
    classification = (
        "middle_0p2ms_pilot_passed_1ms_continuation_manifest_authorized"
        if passed and tier == "automatic_continuation"
        else "middle_0p2ms_pilot_passed_optimization_review_required"
        if passed and tier == "optimization_review"
        else "middle_0p2ms_pilot_passed_explicit_cost_benefit_decision_required"
        if passed
        else "middle_0p2ms_pilot_failed_later_middle_and_fine_blocked"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "base": progress["reports"]["base"],
        "tangent": progress["reports"]["tangent"],
        "anchor": progress["reports"]["anchor"],
        "serialized_replays": replay_reports,
        "cost_projection": projection,
        "middle_1ms_continuation_manifest_authorized": passed,
        "middle_1ms_propagation_authorized": False,
        "middle_5ms_spatial_confirmation_certified": False,
        "fine_cost_bounded_propagation_authorized": False,
        "third_duration_rung_spatial_convergence_certified": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2b0_middle_1ms_continuation_manifest"
            if passed
            else None
        ),
    }
    combined = {
        **{f"base__{key}": value for key, value in base_arrays.items()},
        **{f"tangent__{key}": value for key, value in tangent_arrays.items()},
        **{f"anchor__{key}": value for key, value in anchor_arrays.items()},
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "layout": MIDDLE_LAYOUT,
            "generic_profile": GENERIC_PROFILE,
            "start_seconds": START_SECONDS,
            "stop_seconds": STOP_SECONDS,
            "coupling_face": COUPLING_FACE,
            "projection_safety_factor": PROJECTION_SAFETY_FACTOR,
            "main_controller": main_contract,
            "surrogate_gates": h2a1.GATES,
        },
    )
    np.savez_compressed(DECISIVE_ARRAYS, **combined)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent": ANALYZED_BASE_PARENT,
            "analyzed_base_tree": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "h2a1_summary": _sha256(h2a1.SUMMARY_PATH),
                "base_short_arrays": _sha256(h2.h1.b1a.DECISIVE_ARRAYS),
                "generic_short_arrays": _sha256(h2.h1.b4b3.DECISIVE_ARRAYS),
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
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
