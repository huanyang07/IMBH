#!/usr/bin/env python3
"""Execute the durable, resumable coarse nonlinear 10 ms screen."""

from __future__ import annotations

import argparse
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

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402
import run_causal_inner_nonlinear_5ms_extraction_surface_certificate_wp10c9d6c7c3b5c3h2i1 as h2i1  # noqa: E402
import run_causal_inner_nonlinear_fourth_duration_rung_manifest_wp10c9d6c7c3b5c4a as c4a  # noqa: E402
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402
import run_causal_inner_nonlinear_ten_ms_cost_pilot_wp10c9d6c7c3b5c4b as c4b  # noqa: E402
import run_causal_inner_nonlinear_ten_ms_screen_manifest_wp10c9d6c7c3b5c4b1 as c4b1  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_screen_wp10c9d6c7c3b5c3b as c3b  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFRestart,
    causal_five_field_monolithic_bdf_history,
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
WORK_PACKAGE = "WP10c9d6c7c3b5c4b2"
ANALYZED_BASE_COMMIT = "aa1cce0c8a05dba0d5501530995c9ae280168878"
ANALYZED_BASE_PARENT = "b2aa5d5c956dd36599c154b096552457a2925f2e"
ANALYZED_BASE_TREE = "81411619f6349f69f97223d84599b128f1893e79"

ARTIFACT = "causal_inner_nonlinear_ten_ms_screen_wp10c9d6c7c3b5c4b2"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_ten_ms_screen_"
    "wp10c9d6c7c3b5c4b2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_ten_ms_screen_"
    "wp10c9d6c7c3b5c4b2.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_TEN_MS_SCREEN_"
    "WP10C9D6C7C3B5C4B2_2026-08-08.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROGRESS_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
STAGE_ORDER = (
    "base_main",
    "perturbed_main",
    "base_replay",
    "perturbed_replay",
    "base_strict",
    "perturbed_strict",
)
REPLAY_TARGET_MICROSECONDS = c4b1.MASTER_TARGET_MICROSECONDS[
    c4b1.REPLAY_TARGET_INDICES
]
STRICT_TARGET_MICROSECONDS = c4b1.MASTER_TARGET_MICROSECONDS[
    c4b1.STRICT_TARGET_INDICES
]


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
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


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
    return {
        "runner": _sha256(ROOT / THIS_RUNNER),
        "manifest": _sha256(c4b1.MANIFEST_PATH),
        "pilot_arrays": _sha256(c4b.DECISIVE_ARRAYS),
        "pilot_summary": _sha256(c4b.SUMMARY_PATH),
    }


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(c4b1.SUMMARY_PATH)
    manifest = _read_json(c4b1.MANIFEST_PATH)
    if (
        not parent["passed"]
        or not parent["ten_ms_screen_propagation_authorized"]
        or parent["authorized_next"] != f"{WORK_PACKAGE}_ten_ms_screen"
        or parent["twenty_ms_propagation_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("c4b2 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4b2 analyzed identity changed")
    return parent, manifest


def _pilot_seed(name: str, context) -> tuple[CausalFiveFieldMonolithicBDFRestart, dict]:
    pilot = _load_npz(c4b.DECISIVE_ARRAYS)
    states = np.asarray(pilot[f"{name}__output_states"], dtype=float)
    pilot_times = np.asarray(pilot["pilot_times_seconds"], dtype=float)
    accepted_timestep = float(pilot[f"{name}__accepted_timesteps"][0])
    storage = causal_five_field_monolithic_storage_increment(
        context, states[0], states[1]
    )
    history = causal_five_field_monolithic_bdf_history(
        states[1] - states[0], storage, accepted_timestep
    )
    restart = CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(states[1], copy=True),
        history=history,
        elapsed_time_seconds=float(pilot_times[-1]),
        completed_steps=1,
        next_order=2,
        provenance={
            "work_package": WORK_PACKAGE,
            "trajectory": name,
            "source": "committed_c4b_pilot_endpoint_history_reconstruction",
        },
    )
    arrays = {
        "output_times": pilot_times,
        "output_states": states,
        "output_raw_Tier_I": np.asarray(pilot[f"{name}__output_raw_Tier_I"]),
        "output_extraction_partition": np.asarray(
            pilot[f"{name}__output_extraction_partition"]
        ),
        "output_extraction_audits": np.asarray(
            pilot[f"{name}__extraction_partition_audits"]
        ),
        "accepted_times": np.asarray(
            (c4a.RUNG_START_SECONDS, c4a.PILOT_HORIZON_SECONDS), dtype=float
        ),
        "accepted_timesteps": np.asarray(pilot[f"{name}__accepted_timesteps"]),
        "local_error_estimates": np.asarray(
            (
                _read_json(c4b.SUMMARY_PATH)["trajectory_reports"][name]["method"]
                ["maximum_local_error_estimate"],
            ),
            dtype=float,
        ),
        "retries": np.zeros(1, dtype=int),
        "accepted_step_wall_seconds": np.asarray(
            pilot[f"{name}__accepted_step_wall_seconds"]
        ),
    }
    return restart, arrays


def _stage_paths(stage: str) -> tuple[Path, Path, Path]:
    directory = PROGRESS_DIRECTORY / stage
    return directory / "progress.json", directory / "arrays.npz", directory


def _target_restart_path(stage: str, microseconds: int) -> Path:
    return PROGRESS_DIRECTORY / stage / f"restart_{int(microseconds)}us.npz"


def _save_stage(
    stage: str,
    progress: dict,
    arrays: dict[str, np.ndarray],
    context,
    restart: CausalFiveFieldMonolithicBDFRestart,
) -> None:
    progress_path, arrays_path, directory = _stage_paths(stage)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    target_us = int(progress["current_target_microseconds"])
    save_causal_five_field_monolithic_bdf_restart(
        _target_restart_path(stage, target_us), context, restart
    )
    progress["arrays_sha256"] = _sha256(arrays_path)
    progress["restart_sha256"] = _sha256(_target_restart_path(stage, target_us))
    _write_json(progress_path, progress)


def _load_stage(stage: str, context) -> tuple[dict, dict, CausalFiveFieldMonolithicBDFRestart] | None:
    progress_path, arrays_path, _directory = _stage_paths(stage)
    if not (progress_path.exists() and arrays_path.exists()):
        return None
    progress = _read_json(progress_path)
    if (
        progress.get("work_package") != WORK_PACKAGE
        or progress.get("stage") != stage
        or progress.get("source_identity") != _source_identity()
        or progress.get("arrays_sha256") != _sha256(arrays_path)
    ):
        raise RuntimeError(f"{stage} progress identity changed")
    restart_path = _target_restart_path(
        stage, int(progress["current_target_microseconds"])
    )
    if (
        not restart_path.exists()
        or progress.get("restart_sha256") != _sha256(restart_path)
    ):
        raise RuntimeError(f"{stage} durable restart is missing")
    return (
        progress,
        _load_npz(arrays_path),
        load_causal_five_field_monolithic_bdf_restart(restart_path, context),
    )


def _append(array: np.ndarray, values: np.ndarray, *, skip_first: bool = False) -> np.ndarray:
    values = np.asarray(values)
    if skip_first:
        values = values[1:]
    return np.concatenate((np.asarray(array), values), axis=0)


def _exterior_history(context, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = []
    audits = []
    for state in np.asarray(states, dtype=float):
        value, audit = h2i1._exterior_observable(
            context,
            state,
            c4a.SELECTED_EXTRACTION_LAYOUT_FACE_INDICES[0],
            c2.COUPLING_FACE,
        )
        values.append(value)
        audits.append(audit)
    return np.asarray(values), np.asarray(audits)


def _audit_passed(audits: np.ndarray, gates: dict) -> bool:
    audits = np.asarray(audits, dtype=float)
    return bool(
        np.max(audits[:, 0]) <= gates["maximum_shared_conservative_face_defect"]
        and np.max(audits[:, 1]) <= 1.0e-12
        and np.max(audits[:, 2]) <= 1.0e-12
        and int(np.max(audits[:, 4])) <= gates["maximum_incoming_excision_characteristics"]
        and np.max(audits[:, 6]) <= gates["maximum_exterior_prefix_identity_defect"]
    )


def _segment_passed(report: dict, manifest: dict, *, strict: bool) -> bool:
    gates = manifest["binding_gates"]
    local_gate = (
        gates["strict_local_error_maximum"]
        if strict
        else gates["main_local_error_maximum"]
    )
    return bool(
        report["method_passed"]
        and report["maximum_local_error_estimate"] <= local_gate
        and report["maximum_scaled_residual"] <= gates["maximum_scaled_residual"]
        and report["maximum_discrete_ledger_defect"]
        <= gates["maximum_discrete_ledger_defect"]
        and report["maximum_incoming_excision_characteristics"]
        <= gates["maximum_incoming_excision_characteristics"]
    )


def _new_progress(stage: str, current_us: int, candidate: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "stage": stage,
        "source_identity": _source_identity(),
        "current_target_microseconds": int(current_us),
        "next_candidate_timestep_seconds": float(candidate),
        "segment_reports": {},
        "next_timestep_by_target_microseconds": {str(int(current_us)): float(candidate)},
        "complete": False,
    }


def _run_progression(
    stage: str,
    configuration: dict,
    tangent,
    restart: CausalFiveFieldMonolithicBDFRestart,
    arrays: dict[str, np.ndarray],
    progress: dict,
    targets_us: np.ndarray,
    field_scales: np.ndarray,
    raw_export_scales: np.ndarray,
    contract: dict,
    manifest: dict,
    *,
    strict: bool,
) -> tuple[dict, dict, CausalFiveFieldMonolithicBDFRestart]:
    context = configuration["context"]
    targets_us = np.asarray(targets_us, dtype=int)
    current_us = int(progress["current_target_microseconds"])
    remaining = targets_us[targets_us > current_us]
    for target_us in remaining:
        target_seconds = float(target_us) * 1.0e-6
        print(f"c4b2: {stage} {current_us} -> {int(target_us)} us", flush=True)
        segment = c2._controller_segment(
            configuration,
            tangent,
            restart.primitive_charts,
            restart.history,
            float(current_us) * 1.0e-6,
            progress["next_candidate_timestep_seconds"],
            field_scales,
            raw_export_scales,
            c2.COUPLING_FACE,
            contract,
            output_times=np.asarray((float(current_us) * 1.0e-6, target_seconds)),
            stop_time=target_seconds,
            include_initial_output=True,
            log_prefix=f"c4b2-{stage}",
        )
        report = c3b._segment_report(segment, contract)
        exterior, audits = _exterior_history(context, segment["output_states"][-1:])
        if not _segment_passed(report, manifest, strict=strict) or not _audit_passed(
            audits, manifest["binding_gates"]
        ):
            raise RuntimeError(f"{stage} failed at {int(target_us)} us")
        arrays["output_times"] = _append(
            arrays["output_times"], segment["output_times"], skip_first=True
        )
        arrays["output_states"] = _append(
            arrays["output_states"], segment["output_states"], skip_first=True
        )
        arrays["output_raw_Tier_I"] = _append(
            arrays["output_raw_Tier_I"], segment["output_exports"], skip_first=True
        )
        arrays["output_extraction_partition"] = _append(
            arrays["output_extraction_partition"], exterior
        )
        arrays["output_extraction_audits"] = _append(
            arrays["output_extraction_audits"], audits
        )
        arrays["accepted_times"] = _append(
            arrays["accepted_times"], segment["accepted_times"], skip_first=True
        )
        for key in (
            "accepted_timesteps",
            "local_error_estimates",
            "retries",
            "accepted_step_wall_seconds",
        ):
            arrays[key] = _append(arrays[key], segment[key])
        restart = CausalFiveFieldMonolithicBDFRestart(
            primitive_charts=np.array(segment["final_state"], copy=True),
            history=segment["final_history"],
            elapsed_time_seconds=target_seconds,
            completed_steps=int(arrays["accepted_timesteps"].size),
            next_order=2,
            provenance={
                "work_package": WORK_PACKAGE,
                "stage": stage,
                "target_microseconds": int(target_us),
            },
        )
        current_us = int(target_us)
        progress["current_target_microseconds"] = current_us
        progress["next_candidate_timestep_seconds"] = float(
            segment["next_candidate_timestep"]
        )
        progress["next_timestep_by_target_microseconds"][str(current_us)] = float(
            segment["next_candidate_timestep"]
        )
        progress["segment_reports"][str(current_us)] = report
        progress["complete"] = bool(current_us == int(targets_us[-1]))
        _save_stage(stage, progress, arrays, context, restart)
    return progress, arrays, restart


def _main_stage(
    name: str,
    configuration: dict,
    tangent,
    field_scales: np.ndarray,
    raw_export_scales: np.ndarray,
    manifest: dict,
) -> dict:
    stage = f"{name}_main"
    loaded = _load_stage(stage, configuration["context"])
    if loaded is None:
        restart, arrays = _pilot_seed(name, configuration["context"])
        progress = _new_progress(stage, 5400, 4.0e-4)
        _save_stage(stage, progress, arrays, configuration["context"], restart)
    else:
        progress, arrays, restart = loaded
    progress, arrays, restart = _run_progression(
        stage,
        configuration,
        tangent,
        restart,
        arrays,
        progress,
        c4b1.MASTER_TARGET_MICROSECONDS,
        field_scales,
        raw_export_scales,
        manifest["main_controller"],
        manifest,
        strict=False,
    )
    if not progress["complete"]:
        raise RuntimeError(f"{stage} did not complete")
    return {"progress": progress, "arrays": arrays, "restart": restart}


def _slice_main_at_targets(main: dict, targets_us: np.ndarray) -> dict[str, np.ndarray]:
    main_us = np.rint(main["arrays"]["output_times"] * 1.0e6).astype(int)
    indices = [int(np.flatnonzero(main_us == int(value))[0]) for value in targets_us]
    return {
        key: main["arrays"][key][indices]
        for key in (
            "output_times",
            "output_states",
            "output_raw_Tier_I",
            "output_extraction_partition",
            "output_extraction_audits",
        )
    }


def _aux_seed(
    name: str,
    kind: str,
    main: dict,
    configuration: dict,
    candidate: float,
) -> tuple[dict, dict, CausalFiveFieldMonolithicBDFRestart]:
    targets_us = (
        REPLAY_TARGET_MICROSECONDS
        if kind == "replay"
        else STRICT_TARGET_MICROSECONDS
    )
    start_us = int(targets_us[0])
    source = _slice_main_at_targets(main, np.asarray((start_us,), dtype=int))
    restart = load_causal_five_field_monolithic_bdf_restart(
        _target_restart_path(f"{name}_main", start_us), configuration["context"]
    )
    arrays = {
        **source,
        "accepted_times": np.asarray((float(start_us) * 1.0e-6,), dtype=float),
        "accepted_timesteps": np.empty(0, dtype=float),
        "local_error_estimates": np.empty(0, dtype=float),
        "retries": np.empty(0, dtype=int),
        "accepted_step_wall_seconds": np.empty(0, dtype=float),
    }
    progress = _new_progress(f"{name}_{kind}", start_us, candidate)
    return progress, arrays, restart


def _aux_stage(
    name: str,
    kind: str,
    main: dict,
    configuration: dict,
    tangent,
    field_scales: np.ndarray,
    raw_export_scales: np.ndarray,
    manifest: dict,
) -> dict:
    stage = f"{name}_{kind}"
    loaded = _load_stage(stage, configuration["context"])
    targets_us = (
        REPLAY_TARGET_MICROSECONDS
        if kind == "replay"
        else STRICT_TARGET_MICROSECONDS
    )
    contract = (
        manifest["main_controller"] if kind == "replay" else manifest["strict_controller"]
    )
    if loaded is None:
        if kind == "replay":
            start_us = int(targets_us[0])
            candidate = main["progress"]["next_timestep_by_target_microseconds"][
                str(start_us)
            ]
        else:
            candidate = contract["initial_timestep_seconds"]
        progress, arrays, restart = _aux_seed(
            name, kind, main, configuration, candidate
        )
        _save_stage(stage, progress, arrays, configuration["context"], restart)
    else:
        progress, arrays, restart = loaded
    progress, arrays, restart = _run_progression(
        stage,
        configuration,
        tangent,
        restart,
        arrays,
        progress,
        targets_us,
        field_scales,
        raw_export_scales,
        contract,
        manifest,
        strict=(kind == "strict"),
    )
    return {"progress": progress, "arrays": arrays, "restart": restart}


def _replay_report(name: str, main: dict, replay: dict) -> dict:
    targets = REPLAY_TARGET_MICROSECONDS
    reference = _slice_main_at_targets(main, targets)
    start_seconds = float(targets[0]) * 1.0e-6
    main_mask = main["arrays"]["accepted_times"] >= start_seconds - 1.0e-15
    main_endpoint_mask = main["arrays"]["accepted_times"][1:] > start_seconds + 1.0e-15
    comparisons = {
        f"{key}_bitwise": np.array_equal(replay["arrays"][key], reference[key])
        for key in reference
    }
    comparisons.update(
        {
            "accepted_times_bitwise": np.array_equal(
                replay["arrays"]["accepted_times"],
                main["arrays"]["accepted_times"][main_mask],
            ),
            "accepted_timesteps_bitwise": np.array_equal(
                replay["arrays"]["accepted_timesteps"],
                main["arrays"]["accepted_timesteps"][main_endpoint_mask],
            ),
            "final_restart_numerics_bitwise": _restart_numerics_bitwise(
                main["restart"], replay["restart"]
            ),
        }
    )
    return {"trajectory": name, "passed": all(comparisons.values()), **comparisons}


def _restart_numerics_bitwise(left, right) -> bool:
    return bool(
        np.array_equal(left.primitive_charts, right.primitive_charts)
        and np.array_equal(
            left.history.previous_primitive_increment,
            right.history.previous_primitive_increment,
        )
        and np.array_equal(
            left.history.previous_mapped_storage_increment,
            right.history.previous_mapped_storage_increment,
        )
        and np.array_equal(
            left.history.previous_responsive_height_storage_increment,
            right.history.previous_responsive_height_storage_increment,
        )
        and left.history.previous_timestep_seconds
        == right.history.previous_timestep_seconds
        and left.elapsed_time_seconds == right.elapsed_time_seconds
        and left.next_order == right.next_order
    )


def _maximum_scaled_difference(left: np.ndarray, right: np.ndarray, scales: np.ndarray) -> float:
    return float(np.max(np.abs((np.asarray(left) - np.asarray(right)) / scales)))


def _cosine(left: np.ndarray, right: np.ndarray, scales: np.ndarray) -> float:
    a = (np.asarray(left) / scales).reshape(-1)
    b = (np.asarray(right) / scales).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


def _strict_response_report(
    main: dict[str, dict],
    strict: dict[str, dict],
    field_scales: np.ndarray,
    exterior_scales: np.ndarray,
    manifest: dict,
) -> dict:
    targets = STRICT_TARGET_MICROSECONDS
    main_base = _slice_main_at_targets(main["base"], targets)
    main_perturbed = _slice_main_at_targets(main["perturbed"], targets)
    main_state = main_perturbed["output_states"] - main_base["output_states"]
    strict_state = (
        strict["perturbed"]["arrays"]["output_states"]
        - strict["base"]["arrays"]["output_states"]
    )
    main_export = (
        main_perturbed["output_extraction_partition"]
        - main_base["output_extraction_partition"]
    )
    strict_export = (
        strict["perturbed"]["arrays"]["output_extraction_partition"]
        - strict["base"]["arrays"]["output_extraction_partition"]
    )
    report = {
        "maximum_scaled_state_difference": _maximum_scaled_difference(
            main_state, strict_state, field_scales[None, None, :]
        ),
        "maximum_scaled_extraction_partition_difference": _maximum_scaled_difference(
            main_export, strict_export, exterior_scales[None, :]
        ),
        "state_history_cosine": _cosine(
            main_state, strict_state, field_scales[None, None, :]
        ),
        "extraction_partition_history_cosine": _cosine(
            main_export, strict_export, exterior_scales[None, :]
        ),
    }
    gates = manifest["binding_gates"]
    report["passed"] = bool(
        report["maximum_scaled_state_difference"]
        <= gates["strict_response_maximum_scaled_state_difference"]
        and report["maximum_scaled_extraction_partition_difference"]
        <= gates["strict_response_maximum_scaled_extraction_partition_difference"]
        and report["state_history_cosine"]
        >= gates["strict_response_minimum_history_cosine"]
        and report["extraction_partition_history_cosine"]
        >= gates["strict_response_minimum_history_cosine"]
    )
    return report


def _aggregate_stage(stage: dict) -> dict:
    reports = tuple(stage["progress"]["segment_reports"].values())
    report = {
        "accepted_comparisons": int(stage["arrays"]["accepted_timesteps"].size),
        "maximum_local_error_estimate": float(
            np.max(stage["arrays"]["local_error_estimates"])
        ),
        "sum_local_error_estimates": float(
            np.sum(stage["arrays"]["local_error_estimates"])
        ),
        "maximum_scaled_residual": max(
            float(report["maximum_scaled_residual"]) for report in reports
        ),
        "maximum_discrete_ledger_defect": max(
            float(report["maximum_discrete_ledger_defect"]) for report in reports
        ),
        "maximum_incoming_excision_characteristics": max(
            int(report["maximum_incoming_excision_characteristics"])
            for report in reports
        ),
        "rejected_attempts": int(np.sum(stage["arrays"]["retries"])),
        "measured_wall_seconds": float(
            np.sum(stage["arrays"]["accepted_step_wall_seconds"])
        ),
        "maximum_shared_conservative_face_defect": float(
            np.max(stage["arrays"]["output_extraction_audits"][:, 0])
        ),
        "maximum_exterior_prefix_identity_defect": float(
            np.max(stage["arrays"]["output_extraction_audits"][:, 6])
        ),
    }
    report["passed"] = bool(
        stage["progress"]["complete"]
        and reports
        and report["sum_local_error_estimates"] <= 5.0e-3
        and report["maximum_scaled_residual"] <= 1.0e-10
        and report["maximum_discrete_ledger_defect"] <= 1.0e-12
        and report["maximum_incoming_excision_characteristics"] == 0
        and report["maximum_shared_conservative_face_defect"] <= 1.0e-12
        and report["maximum_exterior_prefix_identity_defect"] <= 1.0e-12
    )
    return report


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


def _finalize(
    parent: dict,
    manifest: dict,
    stages: dict[str, dict],
    field_scales: np.ndarray,
    exterior_scales: np.ndarray,
    context,
    started: float,
) -> int:
    main = {name: stages[f"{name}_main"] for name in ("base", "perturbed")}
    replay = {name: stages[f"{name}_replay"] for name in ("base", "perturbed")}
    strict = {name: stages[f"{name}_strict"] for name in ("base", "perturbed")}
    replay_reports = {
        name: _replay_report(name, main[name], replay[name])
        for name in ("base", "perturbed")
    }
    strict_response = _strict_response_report(
        main, strict, field_scales, exterior_scales, manifest
    )
    stage_reports = {stage: _aggregate_stage(payload) for stage, payload in stages.items()}
    readiness = {
        name: c3b1a._state_audit(
            context,
            stages[f"{name}_main"]["restart"].primitive_charts,
        )
        for name in ("base", "perturbed")
    }
    passed = bool(
        all(report["passed"] for report in stage_reports.values())
        and all(report["passed"] for report in replay_reports.values())
        and strict_response["passed"]
        and all(
            audit["minimum_scattering_optical_depth"]
            >= manifest["binding_gates"]["minimum_scattering_optical_depth"]
            and audit["maximum_h_over_r"]
            <= manifest["binding_gates"]["maximum_h_over_r"]
            and audit["minimum_reconstruction_factor"]
            >= manifest["binding_gates"]["minimum_reconstruction_factor"]
            for audit in readiness.values()
        )
    )
    branch = manifest["positive_branch"] if passed else manifest["negative_branch"]
    physical_failure = any(
        audit["minimum_scattering_optical_depth"]
        < manifest["binding_gates"]["minimum_scattering_optical_depth"]
        or audit["maximum_h_over_r"]
        > manifest["binding_gates"]["maximum_h_over_r"]
        or audit["minimum_reconstruction_factor"]
        < manifest["binding_gates"]["minimum_reconstruction_factor"]
        for audit in readiness.values()
    )
    arrays = {
        "field_scales": field_scales,
        "extraction_partition_scales": exterior_scales,
    }
    for stage, payload in stages.items():
        for key, value in payload["arrays"].items():
            arrays[f"{stage}__{key}"] = value
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "main_targets_seconds": c4b1.MAIN_TARGETS_SECONDS,
        "replay_targets_seconds": c4b1.REPLAY_TARGETS_SECONDS,
        "strict_targets_seconds": c4b1.STRICT_TARGETS_SECONDS,
        "selected_extraction_radius_rg": c4a.SELECTED_EXTRACTION_RADIUS_RG,
        "selected_extraction_face": c4a.SELECTED_EXTRACTION_LAYOUT_FACE_INDICES[0],
    }
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": branch["classification"],
        "passed": passed,
        "physical_failure_detected": physical_failure,
        "stage_reports": stage_reports,
        "replay_reports": replay_reports,
        "strict_response": strict_response,
        "final_state_readiness": readiness,
        "elapsed_seconds": time.perf_counter() - started,
        "ten_ms_screen_certified": passed,
        "twenty_ms_completion_manifest_authorized": passed,
        "twenty_ms_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": branch["authorized_next"],
        "parent_classification_preserved": parent["classification"],
        "pointwise_horizon_flux_convergence_claimed": False,
        "raw_inner_face_rejection_preserved": True,
        "config_sha256": causal_canonical_json_sha256(_plain(config)),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(value) for name, value in arrays.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
            "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
            "implementation_parent_commit": _git_value("rev-parse", "HEAD"),
            "implementation_parent_tree_sha": _git_value("rev-parse", "HEAD^{tree}"),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "implementation_source_hashes": {
                path: _sha256(ROOT / path)
                for path in (THIS_RUNNER, THIS_TEST)
                if (ROOT / path).exists()
            },
            "input_hashes": _source_identity(),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 10 ms screen WP10c9d6c7c3b5c4b2",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                f"Screen passed: `{passed}`.",
                "",
                f"Strict extraction-partition response difference: `{strict_response['maximum_scaled_extraction_partition_difference']:.6e}`.",
                "",
                f"Authorized next: `{summary['authorized_next']}`.",
                "",
                "The binding slow export is the certified exterior-domain extraction partition at `R=1.9531594414758637 r_g`, not the raw pointwise horizon flux. The excision-to-extraction buffer remains part of the microdomain.",
                "",
                "The 20 ms propagation still requires a fresh manifest. Fixed-Q experiments and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=STAGE_ORDER, default=STAGE_ORDER[-1])
    arguments = parser.parse_args(argv)
    started = time.perf_counter()
    parent, manifest = _validate_parent()
    configuration = c3b1a._configurations()[c2.LAYOUT]
    pilot = _load_npz(c4b.DECISIVE_ARRAYS)
    field_scales = np.asarray(pilot["field_scales"], dtype=float)
    raw_export_scales = np.asarray(pilot["export_scales"], dtype=float)
    h2i_arrays = _load_npz(h2i1.DECISIVE_ARRAYS)
    exterior_scales = np.asarray(h2i_arrays["export_scales"], dtype=float)
    print("c4b2: build frozen nonlinear tangent", flush=True)
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    stages: dict[str, dict] = {}
    for stage in STAGE_ORDER:
        name, kind = stage.split("_", maxsplit=1)
        if kind == "main":
            stages[stage] = _main_stage(
                name,
                configuration,
                tangent,
                field_scales,
                raw_export_scales,
                manifest,
            )
        else:
            main_stage = stages.get(f"{name}_main")
            if main_stage is None:
                loaded = _load_stage(f"{name}_main", configuration["context"])
                if loaded is None or not loaded[0]["complete"]:
                    raise RuntimeError(f"{name} main stage is incomplete")
                main_stage = {
                    "progress": loaded[0],
                    "arrays": loaded[1],
                    "restart": loaded[2],
                }
                stages[f"{name}_main"] = main_stage
            stages[stage] = _aux_stage(
                name,
                kind,
                main_stage,
                configuration,
                tangent,
                field_scales,
                raw_export_scales,
                manifest,
            )
        if stage == arguments.through:
            if stage != STAGE_ORDER[-1]:
                print(
                    json.dumps(
                        {
                            "work_package": WORK_PACKAGE,
                            "completed_through": stage,
                            "progress_directory": str(PROGRESS_DIRECTORY),
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            break
    for stage in STAGE_ORDER:
        if stage not in stages:
            loaded = _load_stage(stage, configuration["context"])
            if loaded is None or not loaded[0]["complete"]:
                raise RuntimeError(f"{stage} is incomplete before finalization")
            stages[stage] = {
                "progress": loaded[0],
                "arrays": loaded[1],
                "restart": loaded[2],
            }
    return _finalize(
        parent,
        manifest,
        stages,
        field_scales,
        exterior_scales,
        configuration["context"],
        started,
    )


if __name__ == "__main__":
    raise SystemExit(main())
