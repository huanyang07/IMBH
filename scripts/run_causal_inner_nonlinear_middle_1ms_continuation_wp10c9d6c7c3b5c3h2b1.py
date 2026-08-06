#!/usr/bin/env python3
"""Run the resumable cost-bounded middle continuation from 0.2 to 1 ms."""

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

import run_causal_inner_nonlinear_middle_1ms_continuation_manifest_wp10c9d6c7c3b5c3h2b0 as h2b0  # noqa: E402
import run_causal_inner_nonlinear_middle_pilot_breadth_correction_wp10c9d6c7c3b5c3h2a3 as h2a3  # noqa: E402
import run_causal_inner_nonlinear_middle_cost_pilot_wp10c9d6c7c3b5c3h2a2 as h2a2  # noqa: E402
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_duration_controller_validation_wp10c9d6c7c3b5b as controller  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistoryDirection,
    CausalFiveFieldMonolithicBDFRestart,
    advance_causal_five_field_monolithic_bdf,
    causal_five_field_monolithic_bdf_restarts_equal,
    causal_five_field_monolithic_discrete_export_directions,
    causal_five_field_monolithic_discrete_step_matrix,
    causal_five_field_monolithic_discrete_tangent_step,
    causal_five_field_monolithic_frozen_tangent,
    load_causal_five_field_monolithic_bdf_restart,
    save_causal_five_field_monolithic_bdf_restart,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2b1"
ANALYZED_BASE_COMMIT = "546408525b199ef57564a41916396efe60840f33"
ANALYZED_BASE_PARENT = "d3172f717a70a0721f43f451fd3137d9f934e54e"
ANALYZED_BASE_TREE = "7fb7fe33f8733c9d046fee17b0ef0b78a4a6f431"

MIDDLE_LAYOUT = h2b0.MIDDLE_LAYOUT
GENERIC_PROFILE = h2b0.GENERIC_PROFILE
PROFILES = tuple(h2b0.PROFILES)
GENERIC_INDEX = PROFILES.index(GENERIC_PROFILE)
COUPLING_FACE = int(h2b0.COUPLING_FACE)
START_SECONDS = 2.0e-4
STOP_SECONDS = 1.0e-3
TARGET_MICROSECONDS = tuple(h2b0.TARGET_MICROSECONDS)
TARGET_SECONDS = np.asarray(TARGET_MICROSECONDS, dtype=float) * 1.0e-6
PROJECTION_SAFETY_FACTOR = 2.0
PACKAGING_ALLOWANCE_SECONDS = 900.0

ARTIFACT = "causal_inner_nonlinear_middle_1ms_continuation_wp10c9d6c7c3b5c3h2b1"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_1ms_continuation_"
    "wp10c9d6c7c3b5c3h2b1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_1ms_continuation_"
    "wp10c9d6c7c3b5c3h2b1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_1MS_"
    "CONTINUATION_WP10C9D6C7C3B5C3H2B1_2026-08-06.md"
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
BASE_PATH = CHECKPOINT_DIRECTORY / "base.npz"
TANGENT_PATH = CHECKPOINT_DIRECTORY / "tangent.npz"
ANCHOR_PATH = CHECKPOINT_DIRECTORY / "anchor.npz"


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
    correction = _read_json(h2a3.SUMMARY_PATH)
    manifest = _read_json(h2b0.SUMMARY_PATH)
    if (
        not correction["passed"]
        or not correction["middle_1ms_propagation_authorized"]
        or not manifest["middle_1ms_propagation_authorized"]
        or correction["middle_2ms_propagation_authorized"]
    ):
        raise RuntimeError("h2b1 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2b1 analyzed identity changed")


def _split_payload(path: Path, prefix: str) -> dict[str, np.ndarray]:
    payload = _load_npz(path)
    marker = f"{prefix}__"
    return {
        key.removeprefix(marker): value
        for key, value in payload.items()
        if key.startswith(marker)
    }


def _configuration():
    return h2a2.h2.h1.b1a._configurations()[MIDDLE_LAYOUT]


def _build_frozen_tangent(configuration: dict):
    began = time.perf_counter()
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    return tangent, time.perf_counter() - began


def _progress() -> dict:
    identity = _source_identity()
    if PROGRESS_PATH.exists():
        payload = _read_json(PROGRESS_PATH)
        if payload.get("source_identity") != identity:
            raise RuntimeError("h2b1 checkpoint source identity changed")
        return payload
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "source_identity": identity,
        "base_targets_completed": [],
        "tangent_steps_completed": 0,
        "anchor_steps_completed": 0,
        "reports": {},
    }


def _save_progress(progress: dict) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(PROGRESS_PATH, progress)


def _append(existing: np.ndarray, new: np.ndarray, *, drop_first: bool = False):
    addition = np.asarray(new)[1:] if drop_first else np.asarray(new)
    if addition.size == 0:
        return np.asarray(existing)
    return np.concatenate((np.asarray(existing), addition), axis=0)


def _initial_base(configuration: dict) -> tuple[dict[str, np.ndarray], dict]:
    pilot = _split_payload(h2a2.DECISIVE_ARRAYS, "base")
    state = np.asarray(pilot["accepted_states"][-1], dtype=float)
    value, ledger, incoming = controller._export_value(
        configuration["context"],
        state,
        COUPLING_FACE,
    )
    arrays = {
        "accepted_times": np.asarray([START_SECONDS]),
        "accepted_timesteps": np.empty(0, dtype=float),
        "accepted_states": state[None, :, :],
        "accepted_primitive_histories": pilot["accepted_primitive_histories"][-1: ],
        "accepted_mapped_histories": pilot["accepted_mapped_histories"][-1: ],
        "accepted_height_histories": pilot["accepted_height_histories"][-1: ],
        "accepted_previous_timesteps": pilot["accepted_previous_timesteps"][-1: ],
        "accepted_step_wall_seconds": np.empty(0, dtype=float),
        "local_state_estimates": np.empty(0, dtype=float),
        "local_export_estimates": np.empty(0, dtype=float),
        "local_error_estimates": np.empty(0, dtype=float),
        "retries": np.empty(0, dtype=np.int64),
        "output_times": np.asarray([START_SECONDS]),
        "output_states": state[None, :, :],
        "output_exports": np.asarray(value, dtype=float)[None, :],
        "next_candidate_timestep": pilot["next_candidate_timestep"],
    }
    report = {
        "passed_so_far": bool(ledger <= 1.0e-9 and incoming == 0),
        "accepted_steps": 0,
        "rejected_attempts": 0,
        "wall_seconds": 0.0,
        "maximum_scaled_residual": 0.0,
        "maximum_discrete_ledger_defect": 0.0,
        "maximum_mapped_endpoint_path_closure_defect": 0.0,
        "minimum_path_reconstruction_factor": 1.0,
        "maximum_incoming_excision_characteristics": incoming,
        "maximum_export_ledger_defect": ledger,
    }
    return arrays, report


def _update_base_report(report: dict, segment: dict, elapsed: float) -> dict:
    updated = dict(report)
    updated["accepted_steps"] += int(segment["accepted_timesteps"].size)
    updated["rejected_attempts"] += int(np.sum(segment["retries"]))
    updated["wall_seconds"] += elapsed
    records = segment["step_records"]
    updated["maximum_scaled_residual"] = max(
        updated["maximum_scaled_residual"],
        max(item["maximum_scaled_residual"] for item in records),
    )
    updated["maximum_discrete_ledger_defect"] = max(
        updated["maximum_discrete_ledger_defect"],
        max(item["maximum_discrete_ledger_defect"] for item in records),
    )
    updated["maximum_mapped_endpoint_path_closure_defect"] = max(
        updated["maximum_mapped_endpoint_path_closure_defect"],
        max(item["maximum_mapped_endpoint_path_closure_defect"] for item in records),
    )
    updated["minimum_path_reconstruction_factor"] = min(
        updated["minimum_path_reconstruction_factor"],
        min(item["minimum_path_reconstruction_factor"] for item in records),
    )
    updated["maximum_incoming_excision_characteristics"] = max(
        updated["maximum_incoming_excision_characteristics"],
        max(item["incoming_excision_characteristics"] for item in records),
    )
    updated["maximum_export_ledger_defect"] = max(
        updated["maximum_export_ledger_defect"],
        segment["maximum_export_ledger_defect"],
    )
    return updated


def _run_base_targets(
    progress: dict,
    configuration: dict,
    frozen_tangent,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    contract: dict,
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
    completed = set(int(value) for value in progress["base_targets_completed"])
    for target_us in TARGET_MICROSECONDS[1:]:
        if target_us in completed:
            continue
        target = float(target_us) * 1.0e-6
        start = float(arrays["accepted_times"][-1])
        state = np.array(arrays["accepted_states"][-1], copy=True)
        history = h2a2._history(
            arrays["accepted_primitive_histories"][-1],
            arrays["accepted_mapped_histories"][-1],
            arrays["accepted_height_histories"][-1],
            arrays["accepted_previous_timesteps"][-1],
        )
        began = time.perf_counter()
        segment = c2._controller_segment(
            configuration,
            frozen_tangent,
            state,
            history,
            start,
            float(arrays["next_candidate_timestep"][0]),
            field_scales,
            export_scales,
            COUPLING_FACE,
            contract,
            output_times=np.asarray((start, target), dtype=float),
            stop_time=target,
            include_initial_output=True,
            record_accepted_steps=True,
            log_prefix=f"h2b1-base-{target_us}us",
        )
        wall = time.perf_counter() - began
        if not c2._segment_passed(
            segment,
            contract,
            contract["error_estimator"][
                "short_horizon_sum_of_accepted_error_estimates"
            ],
        ):
            raise RuntimeError(f"h2b1 base segment to {target_us} us failed")
        for key in (
            "accepted_times",
            "accepted_states",
            "accepted_primitive_histories",
            "accepted_mapped_histories",
            "accepted_height_histories",
            "accepted_previous_timesteps",
        ):
            arrays[key] = _append(arrays[key], segment[key], drop_first=True)
        for key in (
            "accepted_timesteps",
            "accepted_step_wall_seconds",
            "local_state_estimates",
            "local_export_estimates",
            "local_error_estimates",
            "retries",
        ):
            arrays[key] = _append(arrays[key], segment[key])
        for key in ("output_times", "output_states", "output_exports"):
            arrays[key] = _append(arrays[key], segment[key], drop_first=True)
        arrays["next_candidate_timestep"] = np.asarray(
            [segment["next_candidate_timestep"]],
            dtype=float,
        )
        report = _update_base_report(report, segment, wall)
        progress["base_targets_completed"].append(target_us)
        progress["reports"]["base"] = report
        np.savez_compressed(BASE_PATH, **arrays)
        _save_progress(progress)
        print(
            f"h2b1: durable base target {target_us} us "
            f"steps={arrays['accepted_timesteps'].size} wall={wall:.1f}s",
            flush=True,
        )
    report.update(
        {
            "passed": bool(
                report["passed_so_far"]
                and report["maximum_scaled_residual"] <= 1.0e-10
                and report["maximum_discrete_ledger_defect"] <= 1.0e-12
                and report["maximum_mapped_endpoint_path_closure_defect"]
                <= 1.0e-9
                and report["minimum_path_reconstruction_factor"] >= 1.0
                and report["maximum_incoming_excision_characteristics"] == 0
                and report["maximum_export_ledger_defect"] <= 1.0e-9
                and float(np.sum(arrays["local_error_estimates"]))
                <= contract["error_estimator"][
                    "short_horizon_sum_of_accepted_error_estimates"
                ]
            ),
            "minimum_timestep_seconds": float(
                np.min(arrays["accepted_timesteps"])
            ),
            "maximum_timestep_seconds": float(
                np.max(arrays["accepted_timesteps"])
            ),
            "next_candidate_timestep_seconds": float(
                arrays["next_candidate_timestep"][0]
            ),
            "maximum_local_error_estimate": float(
                np.max(arrays["local_error_estimates"])
            ),
            "sum_local_error_estimates": float(
                np.sum(arrays["local_error_estimates"])
            ),
            "median_accepted_step_wall_seconds": float(
                np.median(arrays["accepted_step_wall_seconds"])
            ),
        }
    )
    progress["reports"]["base"] = report
    _save_progress(progress)
    return report, arrays


def _initial_tangent() -> tuple[dict[str, np.ndarray], dict]:
    correction = _load_npz(h2a3.DECISIVE_ARRAYS)
    arrays = {
        "state_directions": correction["state_directions"][-1: ],
        "export_directions": correction["Tier_I_export_directions"][-1: ],
        "primitive_history_directions": correction[
            "primitive_history_directions"
        ][-1: ],
        "mapped_history_directions": correction[
            "mapped_history_directions"
        ][-1: ],
        "height_history_directions": correction[
            "height_history_directions"
        ][-1: ],
        "matrix_assembly_wall_seconds": np.empty(0, dtype=float),
        "block_step_wall_seconds": np.empty(0, dtype=float),
        "audit_flags": np.empty(0, dtype=bool),
        "step_ratios": np.empty(0, dtype=float),
        "field_scales": correction["field_scales"],
        "export_scales": correction["export_scales"],
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


def _tangent_audit_indices(base: dict[str, np.ndarray]) -> set[int]:
    ratios = base["accepted_timesteps"] / base["accepted_previous_timesteps"][:-1]
    indices = {0, int(ratios.size - 1)}
    transitions = np.flatnonzero(
        np.abs(ratios - 2.0) > 1.0e-12
    )
    if transitions.size:
        indices.add(int(transitions[0]))
    return indices


def _run_tangent(
    progress: dict,
    configuration: dict,
    base: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    if TANGENT_PATH.exists():
        arrays = _load_npz(TANGENT_PATH)
        report = dict(progress["reports"]["tangent"])
    else:
        arrays, report = _initial_tangent()
        np.savez_compressed(TANGENT_PATH, **arrays)
        progress["reports"]["tangent"] = report
        _save_progress(progress)
    context = configuration["context"]
    columns = configuration["columns"]
    rows = configuration["rows"]
    audit_indices = _tangent_audit_indices(base)
    start_index = int(progress["tangent_steps_completed"])
    for index in range(start_index, base["accepted_timesteps"].size):
        direction = np.asarray(arrays["state_directions"][-1], dtype=float)
        history_direction = CausalFiveFieldMonolithicBDFHistoryDirection(
            previous_primitive_increment=np.asarray(
                arrays["primitive_history_directions"][-1],
                dtype=float,
            ),
            previous_mapped_storage_increment=np.asarray(
                arrays["mapped_history_directions"][-1],
                dtype=float,
            ),
            previous_responsive_height_storage_increment=np.asarray(
                arrays["height_history_directions"][-1],
                dtype=float,
            ),
        ).validated(n_directions=len(PROFILES), n_cells=direction.shape[1])
        base_history = h2a2._history(
            base["accepted_primitive_histories"][index],
            base["accepted_mapped_histories"][index],
            base["accepted_height_histories"][index],
            base["accepted_previous_timesteps"][index],
        )
        dt = float(base["accepted_timesteps"][index])
        previous_dt = float(base["accepted_previous_timesteps"][index])
        began = time.perf_counter()
        matrix = causal_five_field_monolithic_discrete_step_matrix(
            context,
            base["accepted_states"][index],
            base["accepted_states"][index + 1],
            dt,
            previous_dt,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
        )
        matrix_wall = time.perf_counter() - began
        audited = index in audit_indices
        began = time.perf_counter()
        step = causal_five_field_monolithic_discrete_tangent_step(
            context,
            base["accepted_states"][index],
            base["accepted_states"][index + 1],
            dt,
            base_history,
            direction,
            history_direction,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            analytic_step_matrix=matrix,
            audit_complete_residual=audited,
        )
        step_wall = time.perf_counter() - began
        export, export_audit = causal_five_field_monolithic_discrete_export_directions(
            matrix,
            step.new_primitive_directions,
            COUPLING_FACE,
        )
        arrays["state_directions"] = _append(
            arrays["state_directions"],
            step.new_primitive_directions[None, ...],
        )
        arrays["export_directions"] = _append(
            arrays["export_directions"],
            export[None, ...],
        )
        arrays["primitive_history_directions"] = _append(
            arrays["primitive_history_directions"],
            step.new_history_directions.previous_primitive_increment[None, ...],
        )
        arrays["mapped_history_directions"] = _append(
            arrays["mapped_history_directions"],
            step.new_history_directions.previous_mapped_storage_increment[None, ...],
        )
        arrays["height_history_directions"] = _append(
            arrays["height_history_directions"],
            step.new_history_directions.previous_responsive_height_storage_increment[
                None, ...
            ],
        )
        arrays["matrix_assembly_wall_seconds"] = _append(
            arrays["matrix_assembly_wall_seconds"],
            np.asarray([matrix_wall]),
        )
        arrays["block_step_wall_seconds"] = _append(
            arrays["block_step_wall_seconds"],
            np.asarray([step_wall]),
        )
        arrays["audit_flags"] = _append(
            arrays["audit_flags"],
            np.asarray([audited]),
        )
        arrays["step_ratios"] = _append(
            arrays["step_ratios"],
            np.asarray([dt / previous_dt]),
        )
        if np.isfinite(step.maximum_step_matrix_jvp_relative_defect):
            report["maximum_step_matrix_jvp_relative_defect"] = max(
                report["maximum_step_matrix_jvp_relative_defect"],
                step.maximum_step_matrix_jvp_relative_defect,
            )
        report["maximum_linear_solve_relative_defect"] = max(
            report["maximum_linear_solve_relative_defect"],
            step.maximum_linear_solve_relative_defect,
        )
        report["maximum_matrix_component_closure_defect"] = max(
            report["maximum_matrix_component_closure_defect"],
            matrix.maximum_component_closure_defect,
        )
        report["maximum_incoming_excision_characteristics"] = max(
            report["maximum_incoming_excision_characteristics"],
            matrix.incoming_excision_characteristics,
        )
        report["maximum_export_active_prefix_ledger_defect"] = max(
            report["maximum_export_active_prefix_ledger_defect"],
            export_audit.active_prefix_ledger_defect,
        )
        report["maximum_export_transport_telescoping_defect"] = max(
            report["maximum_export_transport_telescoping_defect"],
            export_audit.conservative_transport_telescoping_defect,
        )
        progress["tangent_steps_completed"] = index + 1
        progress["reports"]["tangent"] = report
        np.savez_compressed(TANGENT_PATH, **arrays)
        _save_progress(progress)
        print(
            f"h2b1-tangent: {index + 1}/{base['accepted_timesteps'].size} "
            f"t={base['accepted_times'][index + 1]:.8e} "
            f"matrix={matrix_wall:.1f}s step={step_wall:.1f}s audit={audited}",
            flush=True,
        )
    gates = h2a3.h2a1.GATES
    routine = arrays["block_step_wall_seconds"][~arrays["audit_flags"]]
    report.update(
        {
            "passed": bool(
                report["maximum_step_matrix_jvp_relative_defect"]
                <= gates["maximum_internal_discrete_residual_jvp_relative_defect"]
                and report["maximum_linear_solve_relative_defect"]
                <= gates["maximum_linear_solve_relative_defect"]
                and report["maximum_matrix_component_closure_defect"]
                <= gates["maximum_matrix_component_closure_defect"]
                and report["maximum_incoming_excision_characteristics"] == 0
                and report["maximum_export_active_prefix_ledger_defect"]
                <= gates["maximum_export_active_prefix_ledger_defect"]
                and report["maximum_export_transport_telescoping_defect"]
                <= gates["maximum_export_transport_telescoping_defect"]
            ),
            "accepted_steps": int(base["accepted_timesteps"].size),
            "audit_step_indices": sorted(audit_indices),
            "median_matrix_assembly_wall_seconds": float(
                np.median(arrays["matrix_assembly_wall_seconds"])
            ),
            "median_routine_block_step_wall_seconds": float(
                np.median(routine) if routine.size else 0.0
            ),
        }
    )
    progress["reports"]["tangent"] = report
    _save_progress(progress)
    return report, arrays


def _initial_anchor(configuration: dict, base: dict[str, np.ndarray]):
    pilot = _split_payload(h2a2.DECISIVE_ARRAYS, "anchor")
    state = np.asarray(pilot["anchor_states"][-1], dtype=float)
    base_value, base_ledger, base_incoming = controller._export_value(
        configuration["context"],
        base["accepted_states"][0],
        COUPLING_FACE,
    )
    anchor_value, anchor_ledger, anchor_incoming = controller._export_value(
        configuration["context"],
        state,
        COUPLING_FACE,
    )
    arrays = {
        "anchor_states": state[None, ...],
        "anchor_primitive_histories": pilot["anchor_primitive_histories"][-1: ],
        "anchor_mapped_histories": pilot["anchor_mapped_histories"][-1: ],
        "anchor_height_histories": pilot["anchor_height_histories"][-1: ],
        "anchor_previous_timesteps": pilot["anchor_previous_timesteps"][-1: ],
        "anchor_predictors": np.empty((0, *state.shape), dtype=float),
        "anchor_step_wall_seconds": np.empty(0, dtype=float),
        "sampled_flags": np.empty(0, dtype=bool),
        "sampled_state_error_estimates": np.empty(0, dtype=float),
        "sampled_export_error_estimates": np.empty(0, dtype=float),
        "base_exports": np.asarray(base_value)[None, :],
        "anchor_exports": np.asarray(anchor_value)[None, :],
    }
    report = {
        "maximum_scaled_residual": 0.0,
        "maximum_discrete_ledger_defect": 0.0,
        "maximum_mapped_endpoint_path_closure_defect": 0.0,
        "minimum_path_reconstruction_factor": 1.0,
        "maximum_incoming_excision_characteristics": 0,
        "maximum_export_ledger_defect": max(base_ledger, anchor_ledger),
        "maximum_export_incoming_characteristics": max(
            base_incoming,
            anchor_incoming,
        ),
    }
    return arrays, report


def _anchor_sample_indices(base: dict[str, np.ndarray]) -> set[int]:
    ratios = base["accepted_timesteps"] / base["accepted_previous_timesteps"][:-1]
    indices = {0, int(ratios.size - 1)}
    transitions = np.flatnonzero(np.abs(ratios - 2.0) > 1.0e-12)
    if transitions.size:
        indices.add(int(transitions[0]))
    return indices


def _cumulative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    for index in range(1, times.size):
        dt = float(times[index] - times[index - 1])
        result[index] = result[index - 1] + 0.5 * dt * (
            values[index - 1] + values[index]
        )
    return result


def _run_anchor(
    progress: dict,
    configuration: dict,
    frozen_tangent,
    base: dict[str, np.ndarray],
    tangent: dict[str, np.ndarray],
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    contract: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    if ANCHOR_PATH.exists():
        arrays = _load_npz(ANCHOR_PATH)
        report = dict(progress["reports"]["anchor"])
    else:
        arrays, report = _initial_anchor(configuration, base)
        np.savez_compressed(ANCHOR_PATH, **arrays)
        progress["reports"]["anchor"] = report
        _save_progress(progress)
    context = configuration["context"]
    sampled_indices = _anchor_sample_indices(base)
    start_index = int(progress["anchor_steps_completed"])
    for index in range(start_index, base["accepted_timesteps"].size):
        state = np.asarray(arrays["anchor_states"][-1], dtype=float)
        history = h2a2._history(
            arrays["anchor_primitive_histories"][-1],
            arrays["anchor_mapped_histories"][-1],
            arrays["anchor_height_histories"][-1],
            arrays["anchor_previous_timesteps"][-1],
        )
        predictor = (
            base["accepted_states"][index + 1]
            + tangent["state_directions"][index + 1, GENERIC_INDEX]
            - state
        )
        dt = float(base["accepted_timesteps"][index])
        began = time.perf_counter()
        full = advance_causal_five_field_monolithic_bdf(
            context,
            state,
            dt,
            frozen_tangent,
            order=2,
            history=history,
            initial_primitive_increment=predictor,
            residual_tolerance=1.0e-10,
            ledger_tolerance=1.0e-12,
            maximum_scaled_primitive_change=5.0e-3,
        )
        if full.history is None or not controller._step_passed(full, contract):
            raise RuntimeError(f"h2b1 anchor full step {index} failed")
        records = [controller._step_record(full)]
        sampled = index in sampled_indices
        sampled_state_error = None
        sampled_export_error = None
        if sampled:
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
                raise RuntimeError("h2b1 anchor first half has no history")
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
                raise RuntimeError(f"h2b1 anchor sampled halves {index} failed")
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
            report["maximum_export_ledger_defect"] = max(
                report["maximum_export_ledger_defect"],
                full_ledger,
                fine_ledger,
            )
            report["maximum_export_incoming_characteristics"] = max(
                report["maximum_export_incoming_characteristics"],
                full_incoming,
                fine_incoming,
            )
            sampled_state_error = controller._state_estimate(
                full.primitive_charts,
                half_second.primitive_charts,
                field_scales,
            )
            sampled_export_error = controller._export_estimate(
                full_export,
                fine_export,
                export_scales,
            )
            if max(sampled_state_error, sampled_export_error) > contract[
                "error_estimator"
            ]["local_tolerance"]:
                raise RuntimeError(f"h2b1 anchor sampled error {index} failed")
        wall = time.perf_counter() - began
        base_value, base_ledger, base_incoming = controller._export_value(
            context,
            base["accepted_states"][index + 1],
            COUPLING_FACE,
        )
        anchor_value, anchor_ledger, anchor_incoming = controller._export_value(
            context,
            full.primitive_charts,
            COUPLING_FACE,
        )
        arrays["anchor_states"] = _append(
            arrays["anchor_states"], full.primitive_charts[None, ...]
        )
        arrays["anchor_primitive_histories"] = _append(
            arrays["anchor_primitive_histories"],
            full.history.previous_primitive_increment[None, ...],
        )
        arrays["anchor_mapped_histories"] = _append(
            arrays["anchor_mapped_histories"],
            full.history.previous_mapped_storage_increment[None, ...],
        )
        arrays["anchor_height_histories"] = _append(
            arrays["anchor_height_histories"],
            full.history.previous_responsive_height_storage_increment[None, ...],
        )
        arrays["anchor_previous_timesteps"] = _append(
            arrays["anchor_previous_timesteps"],
            np.asarray([full.history.previous_timestep_seconds]),
        )
        arrays["anchor_predictors"] = _append(
            arrays["anchor_predictors"], predictor[None, ...]
        )
        arrays["anchor_step_wall_seconds"] = _append(
            arrays["anchor_step_wall_seconds"], np.asarray([wall])
        )
        arrays["sampled_flags"] = _append(
            arrays["sampled_flags"], np.asarray([sampled])
        )
        if sampled:
            arrays["sampled_state_error_estimates"] = _append(
                arrays["sampled_state_error_estimates"],
                np.asarray([sampled_state_error]),
            )
            arrays["sampled_export_error_estimates"] = _append(
                arrays["sampled_export_error_estimates"],
                np.asarray([sampled_export_error]),
            )
        arrays["base_exports"] = _append(
            arrays["base_exports"], np.asarray(base_value)[None, :]
        )
        arrays["anchor_exports"] = _append(
            arrays["anchor_exports"], np.asarray(anchor_value)[None, :]
        )
        report["maximum_scaled_residual"] = max(
            report["maximum_scaled_residual"],
            max(item["maximum_scaled_residual"] for item in records),
        )
        report["maximum_discrete_ledger_defect"] = max(
            report["maximum_discrete_ledger_defect"],
            max(item["maximum_discrete_ledger_defect"] for item in records),
        )
        report["maximum_mapped_endpoint_path_closure_defect"] = max(
            report["maximum_mapped_endpoint_path_closure_defect"],
            max(item["maximum_mapped_endpoint_path_closure_defect"] for item in records),
        )
        report["minimum_path_reconstruction_factor"] = min(
            report["minimum_path_reconstruction_factor"],
            min(item["minimum_path_reconstruction_factor"] for item in records),
        )
        report["maximum_incoming_excision_characteristics"] = max(
            report["maximum_incoming_excision_characteristics"],
            max(item["incoming_excision_characteristics"] for item in records),
        )
        report["maximum_export_ledger_defect"] = max(
            report["maximum_export_ledger_defect"],
            base_ledger,
            anchor_ledger,
        )
        report["maximum_export_incoming_characteristics"] = max(
            report["maximum_export_incoming_characteristics"],
            base_incoming,
            anchor_incoming,
        )
        progress["anchor_steps_completed"] = index + 1
        progress["reports"]["anchor"] = report
        np.savez_compressed(ANCHOR_PATH, **arrays)
        _save_progress(progress)
        print(
            f"h2b1-anchor: {index + 1}/{base['accepted_timesteps'].size} "
            f"t={base['accepted_times'][index + 1]:.8e} "
            f"wall={wall:.1f}s sampled={sampled}",
            flush=True,
        )
    actual_state = arrays["anchor_states"] - base["accepted_states"]
    actual_export = arrays["anchor_exports"] - arrays["base_exports"]
    predicted_state = tangent["state_directions"][:, GENERIC_INDEX]
    predicted_export = tangent["export_directions"][:, GENERIC_INDEX]
    state_metrics = h2a3.h2a1.h1._response_metrics(
        predicted_state,
        actual_state,
        field_scales,
    )
    export_metrics = h2a3.h2a1.h1._response_metrics(
        predicted_export,
        actual_export,
        export_scales,
    )
    predicted_cumulative = _cumulative(predicted_export, base["accepted_times"])
    actual_cumulative = _cumulative(actual_export, base["accepted_times"])
    cumulative_metrics = h2a3.h2a1.h1._response_metrics(
        predicted_cumulative,
        actual_cumulative,
        export_scales,
    )
    for metrics in (state_metrics, export_metrics, cumulative_metrics):
        metrics["discrepancy_fraction_of_observable_response"] = float(
            metrics["maximum_scaled_discrepancy"]
            / max(metrics["maximum_scaled_actual_response"], np.finfo(float).tiny)
        )
    arrays["actual_state_response"] = actual_state
    arrays["actual_export_response"] = actual_export
    arrays["predicted_cumulative_export_response"] = predicted_cumulative
    arrays["actual_cumulative_export_response"] = actual_cumulative
    readiness = h2a2.h2.h1.b1a._state_audit(
        configuration["context"], arrays["anchor_states"][-1]
    )
    gates = h2a3.h2a1.GATES
    sampled_state_max = float(
        np.max(arrays["sampled_state_error_estimates"], initial=0.0)
    )
    sampled_export_max = float(
        np.max(arrays["sampled_export_error_estimates"], initial=0.0)
    )
    passed = bool(
        all(
            metrics["maximum_scaled_discrepancy"]
            <= (
                gates["maximum_absolute_scaled_state_discrepancy"]
                if metrics is state_metrics
                else gates["maximum_absolute_scaled_Tier_I_discrepancy"]
            )
            and metrics["discrepancy_fraction_of_observable_response"]
            <= gates["maximum_discrepancy_fraction_of_observable_response"]
            and metrics["history_cosine"]
            >= (
                gates["minimum_state_history_cosine"]
                if metrics is state_metrics
                else gates["minimum_Tier_I_history_cosine"]
            )
            for metrics in (state_metrics, export_metrics, cumulative_metrics)
        )
        and max(sampled_state_max, sampled_export_max)
        <= contract["error_estimator"]["local_tolerance"]
        and report["maximum_scaled_residual"] <= 1.0e-10
        and report["maximum_discrete_ledger_defect"] <= 1.0e-12
        and report["maximum_mapped_endpoint_path_closure_defect"] <= 1.0e-9
        and report["minimum_path_reconstruction_factor"] >= 1.0
        and report["maximum_incoming_excision_characteristics"] == 0
        and report["maximum_export_ledger_defect"] <= 1.0e-9
        and report["maximum_export_incoming_characteristics"] == 0
        and readiness["maximum_h_over_r"] <= 0.12
        and readiness["minimum_scattering_optical_depth"] > 1.0
        and readiness["minimum_reconstruction_factor"] >= 1.0
    )
    routine = arrays["anchor_step_wall_seconds"][~arrays["sampled_flags"]]
    sampled_walls = arrays["anchor_step_wall_seconds"][arrays["sampled_flags"]]
    report.update(
        {
            "passed": passed,
            "state": state_metrics,
            "instantaneous_Tier_I": export_metrics,
            "cumulative_Tier_I": cumulative_metrics,
            "sampled_step_indices": sorted(sampled_indices),
            "maximum_sampled_state_error_estimate": sampled_state_max,
            "maximum_sampled_export_error_estimate": sampled_export_max,
            "final_state_audit": readiness,
            "median_routine_step_wall_seconds": float(
                np.median(routine) if routine.size else 0.0
            ),
            "median_sampled_step_wall_seconds": float(
                np.median(sampled_walls) if sampled_walls.size else 0.0
            ),
        }
    )
    progress["reports"]["anchor"] = report
    np.savez_compressed(ANCHOR_PATH, **arrays)
    _save_progress(progress)
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
    accepted_times: np.ndarray,
    predictor: np.ndarray | None,
) -> dict:
    context = configuration["context"]
    index = int(timesteps.size - 1)
    history = h2a2._history(
        primitive_histories[index],
        mapped_histories[index],
        height_histories[index],
        previous_timesteps[index],
    )
    checkpoint = CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(states[index], copy=True),
        history=history,
        elapsed_time_seconds=float(accepted_times[index]),
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
    target_history = h2a2._history(
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


def _remaining_projection(
    base_report: dict,
    base: dict[str, np.ndarray],
    tangent_report: dict,
    tangent: dict[str, np.ndarray],
    anchor_report: dict,
    replays: dict,
    setup_seconds: float,
    contract: dict,
) -> dict:
    remaining = h2a2._simulate_remaining_steps(
        STOP_SECONDS,
        float(base["accepted_timesteps"][-1]),
        float(base["next_candidate_timestep"][0]),
        float(base["local_error_estimates"][-1]),
        contract,
    )
    replay_median = float(np.median([item["wall_seconds"] for item in replays.values()]))
    routine_anchor = anchor_report["median_routine_step_wall_seconds"]
    routine_tangent = tangent_report["median_matrix_assembly_wall_seconds"] + tangent_report[
        "median_routine_block_step_wall_seconds"
    ]
    audit_extra = max(
        float(np.median(tangent["block_step_wall_seconds"][tangent["audit_flags"]]))
        - tangent_report["median_routine_block_step_wall_seconds"],
        0.0,
    )
    raw = (
        setup_seconds
        + remaining * base_report["median_accepted_step_wall_seconds"]
        + remaining * routine_anchor
        + 3.0 * max(
            anchor_report["median_sampled_step_wall_seconds"] - routine_anchor,
            0.0,
        )
        + remaining * routine_tangent
        + 3.0 * audit_extra
        + 2.0 * replay_median
        + PACKAGING_ALLOWANCE_SECONDS
    )
    projected = PROJECTION_SAFETY_FACTOR * raw
    hours = projected / 3600.0
    tier = (
        "automatic_continuation"
        if hours <= 24.0
        else "optimization_review"
        if hours <= 48.0
        else "explicit_cost_benefit_decision"
    )
    return {
        "remaining_steps_to_5ms": remaining,
        "safety_factor": PROJECTION_SAFETY_FACTOR,
        "projected_remaining_wall_seconds": projected,
        "projected_remaining_wall_hours": hours,
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
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
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
    h2a2._write_json(CANONICAL_SUMMARY, catalog)


def _report(summary: dict) -> str:
    anchor = summary["anchor"]
    return "\n".join(
        (
            "# Middle 1 ms continuation WP10c9d6c7c3b5c3h2b1",
            "",
            "## Classification",
            "",
            f"`{summary['classification']}`",
            "",
            f"The middle nonlinear base, generic nonlinear anchor, and corrected five-profile tangent reached 1 ms in `{summary['base']['accepted_steps']}` accepted continuation steps with `{summary['base']['rejected_attempts']}` rejected attempts.",
            "",
            f"The generic tangent/anchor discrepancies are `{anchor['state']['maximum_scaled_discrepancy']:.6e}` in state, `{anchor['instantaneous_Tier_I']['maximum_scaled_discrepancy']:.6e}` in instantaneous Tier-I response, and `{anchor['cumulative_Tier_I']['maximum_scaled_discrepancy']:.6e}` in the cumulative response. Base and anchor last-step serialized replays are bitwise.",
            "",
            f"The conservative factor-two projection for the remaining middle work to 5 ms is `{summary['remaining_cost_projection']['projected_remaining_wall_hours']:.2f}` hours in the `{summary['remaining_cost_projection']['resource_tier']}` tier.",
            "",
            "A pass authorizes only a fresh definitions-only middle 2 ms continuation manifest. Fine work, the 5 ms spatial certificate, fixed-Q experiments, and reduced slow evolution remain blocked.",
            "",
        )
    )


def main() -> int:
    _validate_parent()
    progress = _progress()
    configuration = _configuration()
    print("h2b1: build frozen nonlinear tangent", flush=True)
    frozen_tangent, setup_seconds = _build_frozen_tangent(configuration)
    correction = _load_npz(h2a3.DECISIVE_ARRAYS)
    field_scales = np.asarray(correction["field_scales"], dtype=float)
    export_scales = np.asarray(correction["export_scales"], dtype=float)
    contract, _strict = h2a2.h2.g._controller_contracts()

    base_report, base = _run_base_targets(
        progress,
        configuration,
        frozen_tangent,
        field_scales,
        export_scales,
        contract,
    )
    if not base_report["passed"]:
        raise RuntimeError("h2b1 base failed")
    tangent_report, tangent = _run_tangent(progress, configuration, base)
    if not tangent_report["passed"]:
        raise RuntimeError("h2b1 tangent failed")
    anchor_report, anchor = _run_anchor(
        progress,
        configuration,
        frozen_tangent,
        base,
        tangent,
        field_scales,
        export_scales,
        contract,
    )
    if not anchor_report["passed"]:
        raise RuntimeError("h2b1 anchor failed")
    replays = {
        "base": _serialized_last_step_replay(
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
        "anchor": _serialized_last_step_replay(
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
    replay_passed = all(
        item["checkpoint_roundtrip_bitwise"]
        and item["last_step_replay_bitwise"]
        and item["maximum_scaled_residual"] <= 1.0e-10
        for item in replays.values()
    )
    projection = _remaining_projection(
        base_report,
        base,
        tangent_report,
        tangent,
        anchor_report,
        replays,
        setup_seconds,
        contract,
    )
    passed = bool(
        base_report["passed"]
        and tangent_report["passed"]
        and anchor_report["passed"]
        and replay_passed
    )
    classification = (
        "middle_1ms_continuation_passed_2ms_manifest_authorized"
        if passed
        else "middle_1ms_continuation_failed_later_middle_and_fine_blocked"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "base": base_report,
        "tangent": tangent_report,
        "anchor": anchor_report,
        "serialized_replays": replays,
        "remaining_cost_projection": projection,
        "middle_2ms_continuation_manifest_authorized": passed,
        "middle_2ms_propagation_authorized": False,
        "middle_5ms_spatial_confirmation_certified": False,
        "fine_cost_bounded_propagation_authorized": False,
        "third_duration_rung_spatial_convergence_certified": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2c0_middle_2ms_continuation_manifest"
            if passed
            else None
        ),
    }
    combined = {
        **{f"base__{key}": value for key, value in base.items()},
        **{f"tangent__{key}": value for key, value in tangent.items()},
        **{f"anchor__{key}": value for key, value in anchor.items()},
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    h2a2._write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "layout": MIDDLE_LAYOUT,
            "profiles": PROFILES,
            "generic_profile": GENERIC_PROFILE,
            "coupling_face": COUPLING_FACE,
            "target_microseconds": TARGET_MICROSECONDS,
            "controller": contract,
            "surrogate_gates": h2a3.h2a1.GATES,
        },
    )
    np.savez_compressed(DECISIVE_ARRAYS, **combined)
    h2a2._write_json(SUMMARY_PATH, summary)
    h2a2._write_json(
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
                "continuation_manifest": _sha256(h2b0.MANIFEST_PATH),
                "pilot_summary": _sha256(h2a2.SUMMARY_PATH),
                "pilot_arrays": _sha256(h2a2.DECISIVE_ARRAYS),
                "breadth_correction_summary": _sha256(h2a3.SUMMARY_PATH),
                "breadth_correction_arrays": _sha256(h2a3.DECISIVE_ARRAYS),
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
