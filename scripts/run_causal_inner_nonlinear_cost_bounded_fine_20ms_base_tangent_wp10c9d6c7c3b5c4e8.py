#!/usr/bin/env python3
"""Run the cost-bounded fine-layout base and block tangent to 20 ms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_cost_bounded_fine_20ms_manifest_wp10c9d6c7c3b5c4e7 as c4e7  # noqa: E402
import run_causal_inner_nonlinear_optimized_middle_20ms_completion_wp10c9d6c7c3b5c4e3 as c4e3  # noqa: E402
import run_causal_inner_nonlinear_fine_5ms_completion_wp10c9d6c7c3b5c3h2e1 as fine5  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    advance_causal_five_field_monolithic_bdf,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e8"
ANALYZED_BASE_COMMIT = c4e7.ANALYZED_BASE_COMMIT
ANALYZED_BASE_PARENT = c4e7.ANALYZED_BASE_PARENT
ANALYZED_BASE_TREE = c4e7.ANALYZED_BASE_TREE

FINE_LAYOUT = c4e7.FINE_LAYOUT
PROFILES = tuple(c4e7.PROFILES)
GENERIC_INDEX = PROFILES.index(c4e7.GENERIC_PROFILE)
COUPLING_FACE = c4e7.COUPLING_FACE
EXTRACTION_FACE = c4e7.EXTRACTION_FACE
TARGET_MICROSECONDS = tuple(c4e7.TARGET_MICROSECONDS)
AUDIT_TARGET_MICROSECONDS = tuple(c4e7.AUDIT_TARGET_MICROSECONDS)
TARGET_SECONDS = np.asarray(TARGET_MICROSECONDS, dtype=float) * 1.0e-6
AUDIT_SECONDS = np.asarray(AUDIT_TARGET_MICROSECONDS, dtype=float) * 1.0e-6
LANDMARK_SECONDS = np.unique(np.concatenate((TARGET_SECONDS, AUDIT_SECONDS)))
PILOT_STOP_SECONDS = c4e7.PILOT_STOP_MICROSECONDS * 1.0e-6
STOP_SECONDS = c4e7.STOP_MICROSECONDS * 1.0e-6
MAXIMUM_TIMESTEP_SECONDS = c4e7.MAXIMUM_TIMESTEP_SECONDS

ARTIFACT = (
    "causal_inner_nonlinear_cost_bounded_fine_20ms_base_tangent_"
    "wp10c9d6c7c3b5c4e8"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_cost_bounded_fine_20ms_base_tangent_"
    "wp10c9d6c7c3b5c4e8.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_cost_bounded_fine_20ms_base_tangent_"
    "wp10c9d6c7c3b5c4e8.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_COST_BOUNDED_FINE_20MS_"
    "BASE_TANGENT_WP10C9D6C7C3B5C4E8_2026-08-11.md"
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
EXTRACTION_PATH = CHECKPOINT_DIRECTORY / "extraction.npz"
REMAINDER_PATH = CHECKPOINT_DIRECTORY / "remainder.npz"
PILOT_SUMMARY_PATH = CHECKPOINT_DIRECTORY / "pilot_summary.json"
GENERATION_DIRECTORY = CHECKPOINT_DIRECTORY / "generations"

h2b1 = c4e3.h2b1
controller = c4e3.controller
_PILOT_STOP_ACTIVE = False


class _PilotComplete(RuntimeError):
    pass


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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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
        c4e7.THIS_RUNNER,
        c4e7.THIS_TEST,
        c4e3.THIS_RUNNER,
        c4e3.THIS_TEST,
        fine5.THIS_RUNNER,
        h2b1.CONTROLLER_RELATIVE,
        h2b1.MODULE_RELATIVE,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py",
    )
    return {
        path: _sha256(ROOT / path) for path in dependencies if (ROOT / path).exists()
    }


def _checkpoint_hashes() -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in (BASE_PATH, TANGENT_PATH, EXTRACTION_PATH, REMAINDER_PATH)
        if path.exists()
    }


def _snapshot_generation(progress: dict) -> None:
    generation = (
        int(progress.get("base_steps_completed", 0))
        + int(progress.get("tangent_steps_completed", 0))
        + int(progress.get("extraction_times_completed", 0))
        + int(progress.get("remainder_steps_completed", 0))
    )
    GENERATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary = GENERATION_DIRECTORY / f"generation_{generation:05d}.tmp"
    final = GENERATION_DIRECTORY / f"generation_{generation:05d}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    for path in (BASE_PATH, TANGENT_PATH, EXTRACTION_PATH, REMAINDER_PATH):
        if path.exists():
            shutil.copyfile(path, temporary / path.name)
    (temporary / "progress.json").write_text(
        json.dumps(_plain(progress), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if final.exists():
        shutil.rmtree(final)
    os.replace(temporary, final)
    latest_temporary = GENERATION_DIRECTORY / "LATEST.tmp"
    latest_temporary.write_text(final.name + "\n", encoding="utf-8")
    os.replace(latest_temporary, GENERATION_DIRECTORY / "LATEST")
    completed = sorted(GENERATION_DIRECTORY.glob("generation_[0-9]*"))
    for old in completed[:-2]:
        shutil.rmtree(old)


def _save_progress(progress: dict) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    payload = dict(progress)
    payload["checkpoint_hashes"] = _checkpoint_hashes()
    _write_json(PROGRESS_PATH, payload)
    progress["checkpoint_hashes"] = payload["checkpoint_hashes"]
    _snapshot_generation(payload)
    if _PILOT_STOP_ACTIVE and BASE_PATH.exists():
        base = _load_npz(BASE_PATH)
        if float(base["accepted_times"][-1]) >= PILOT_STOP_SECONDS - 1.0e-15:
            raise _PilotComplete


def _progress() -> dict:
    identity = _source_identity()
    manifest_hash = _sha256(c4e7.MANIFEST_PATH)
    if PROGRESS_PATH.exists():
        payload = _read_json(PROGRESS_PATH)
        if payload.get("source_identity") != identity:
            raise RuntimeError("c4e8 checkpoint source identity changed")
        if payload.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("c4e8 checkpoint manifest changed")
        if payload.get("checkpoint_hashes") != _checkpoint_hashes():
            raise RuntimeError("c4e8 checkpoint payload hash changed")
        return payload
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "source_identity": identity,
        "manifest_sha256": manifest_hash,
        "base_steps_completed": 0,
        "tangent_steps_completed": 0,
        "extraction_times_completed": 0,
        "remainder_steps_completed": 0,
        "reports": {},
        "checkpoint_hashes": {},
    }


def _validate_parent() -> tuple[dict, dict]:
    summary = _read_json(c4e7.SUMMARY_PATH)
    manifest = _read_json(c4e7.MANIFEST_PATH)
    if (
        not summary["passed"]
        or not summary["fine_base_block_tangent_propagation_authorized"]
        or summary["full_fine_generic_anchor_authorized"]
        or summary["fine_twenty_ms_spatial_certificate_issued"]
        or summary["physical_failure_detected"]
    ):
        raise RuntimeError("c4e8 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e8 analyzed identity changed")
    return summary, manifest


def _parent_arrays() -> dict[str, np.ndarray]:
    return _load_npz(fine5.DECISIVE_ARRAYS)


def _initial_base(configuration: dict):
    parent = _parent_arrays()
    state = np.asarray(parent["base__accepted_states"][-1], dtype=float)
    value, ledger, incoming = controller._export_value(
        configuration["context"], state, COUPLING_FACE
    )
    arrays = {
        "accepted_times": np.asarray([5.0e-3]),
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
        "output_times": np.asarray([5.0e-3]),
        "output_states": state[None, ...],
        "output_exports": np.asarray(value, dtype=float)[None, :],
        "next_candidate_timestep": np.asarray(
            [min(float(parent["base__next_candidate_timestep"][-1]), 4.0e-4)]
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


def _initial_tangent():
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


def _patch_modules() -> None:
    c4e3.WORK_PACKAGE = WORK_PACKAGE
    c4e3.ARTIFACT = ARTIFACT
    c4e3.OUTPUT_TARGET_MICROSECONDS = TARGET_MICROSECONDS
    c4e3.AUDIT_TARGET_MICROSECONDS = AUDIT_TARGET_MICROSECONDS
    c4e3.OUTPUT_TARGET_SECONDS = TARGET_SECONDS
    c4e3.AUDIT_TARGET_SECONDS = AUDIT_SECONDS
    c4e3.LANDMARK_SECONDS = LANDMARK_SECONDS
    c4e3.MAXIMUM_TIMESTEP_SECONDS = MAXIMUM_TIMESTEP_SECONDS
    c4e3.EXTRACTION_FACE = EXTRACTION_FACE
    c4e3.COUPLING_FACE = COUPLING_FACE
    c4e3.PROFILES = PROFILES
    c4e3.GENERIC_INDEX = GENERIC_INDEX
    c4e3.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
    c4e3.PROGRESS_PATH = PROGRESS_PATH
    c4e3.BASE_PATH = BASE_PATH
    c4e3.TANGENT_PATH = TANGENT_PATH
    c4e3.ANCHOR_PATH = CHECKPOINT_DIRECTORY / "forbidden_anchor.npz"
    c4e3.EXTRACTION_PATH = EXTRACTION_PATH
    c4e3._source_identity = _source_identity
    c4e3._save_progress = _save_progress
    c4e3._initial_base = _initial_base
    c4e3._initial_tangent = _initial_tangent
    c4e3.c4e2.MANIFEST_PATH = c4e7.MANIFEST_PATH
    c4e3.c4e2.TIMESTEP_CAP_CANDIDATES_SECONDS = (MAXIMUM_TIMESTEP_SECONDS,)
    c4e3._patch_shared_modules()
    h2b1.MIDDLE_LAYOUT = FINE_LAYOUT
    h2b1.COUPLING_FACE = COUPLING_FACE
    h2b1.PROFILES = PROFILES
    h2b1.GENERIC_INDEX = GENERIC_INDEX
    h2b1._initial_tangent = _initial_tangent
    h2b1._tangent_audit_indices = c4e3._declared_audit_indices


def _pilot_report(base: dict[str, np.ndarray], progress: dict) -> dict:
    routine = base["accepted_step_wall_seconds"][~base["audit_flags"]]
    audit = base["accepted_step_wall_seconds"][base["audit_flags"]]
    accepted = int(base["accepted_timesteps"].size)
    gates_passed = bool(
        accepted >= 3
        and routine.size >= 2
        and audit.size >= 1
        and float(np.max(base["step_maximum_scaled_residuals"])) <= 1.0e-10
        and float(np.max(base["step_maximum_discrete_ledger_defects"])) <= 1.0e-12
        and float(np.max(base["step_maximum_mapped_closure_defects"])) <= 1.0e-9
        and float(np.min(base["step_minimum_reconstruction_factors"])) >= 1.0
        and int(np.max(base["step_incoming_excision_characteristics"])) == 0
        and float(np.max(base["step_extraction_identity_defects"])) <= 1.0e-12
        and float(np.max(base["local_error_bounds"])) <= 2.5e-4
    )
    remaining_steps = max(c4e7.PROJECTED_ACCEPTED_STEPS - accepted, 0)
    routine_wall = float(np.median(routine))
    audit_wall = float(np.median(audit))
    remaining_audits = len(AUDIT_TARGET_MICROSECONDS) - int(np.sum(base["audit_flags"]))
    remaining_routine = max(remaining_steps - remaining_audits, 0)
    manifest_cost = _read_json(c4e7.MANIFEST_PATH)["cost_control"]
    extra = (
        float(manifest_cost["projected_tangent_matrix_wall_seconds"])
        + float(manifest_cost["projected_sampled_anchor_wall_seconds"])
        + float(manifest_cost["projected_setup_replay_io_wall_seconds"])
    )
    raw = remaining_routine * routine_wall + remaining_audits * audit_wall + extra
    report = {
        "passed": gates_passed,
        "stop_time_seconds": float(base["accepted_times"][-1]),
        "accepted_steps": accepted,
        "routine_steps": int(routine.size),
        "audit_steps": int(audit.size),
        "median_routine_step_wall_seconds": routine_wall,
        "median_audit_step_wall_seconds": audit_wall,
        "maximum_local_error_bound": float(np.max(base["local_error_bounds"])),
        "maximum_scaled_residual": float(
            np.max(base["step_maximum_scaled_residuals"])
        ),
        "rejected_attempts": int(np.sum(base["retries"])),
        "projected_remaining_raw_wall_hours": raw / 3600.0,
        "projected_remaining_safe_wall_hours": (
            c4e7.COST_SAFETY_FACTOR * raw / 3600.0
        ),
        "full_continuation_authorized": gates_passed,
        "physical_failure_detected": False,
        "progress_base_report": progress["reports"].get("base", {}),
    }
    _write_json(PILOT_SUMMARY_PATH, report)
    return report


def _initial_extraction() -> dict[str, np.ndarray]:
    return {
        "accepted_times": np.empty(0, dtype=float),
        "base_values": np.empty((0, 13), dtype=float),
        "tangent_directions": np.empty((0, len(PROFILES), 13), dtype=float),
        "maximum_identity_defects": np.empty(0, dtype=float),
        "maximum_ledger_audits": np.empty((0, 4), dtype=float),
        "evaluation_wall_seconds": np.empty(0, dtype=float),
    }


def _run_extraction(progress, configuration, base, tangent):
    if EXTRACTION_PATH.exists():
        arrays = _load_npz(EXTRACTION_PATH)
    else:
        arrays = _initial_extraction()
    context = configuration["context"]
    start = int(progress.get("extraction_times_completed", 0))
    for index in range(start, base["accepted_times"].size):
        began = time.perf_counter()
        base_value, identity, audit = c4e3._extraction_value(
            context, base["accepted_states"][index]
        )
        directions = []
        for direction in tangent["state_directions"][index]:
            value, local_identity, local_audit = c4e3._extraction_direction(
                context,
                base["accepted_states"][index],
                direction,
                c4e3.EXTRACTION_JVP_RELATIVE_STEP,
            )
            directions.append(value)
            identity = max(identity, local_identity)
            audit = np.maximum(audit, local_audit)
        arrays["accepted_times"] = np.append(
            arrays["accepted_times"], base["accepted_times"][index]
        )
        arrays["base_values"] = np.concatenate(
            (arrays["base_values"], base_value[None, :]), axis=0
        )
        arrays["tangent_directions"] = np.concatenate(
            (arrays["tangent_directions"], np.asarray(directions)[None, ...]),
            axis=0,
        )
        arrays["maximum_identity_defects"] = np.append(
            arrays["maximum_identity_defects"], identity
        )
        arrays["maximum_ledger_audits"] = np.concatenate(
            (arrays["maximum_ledger_audits"], audit[None, :]), axis=0
        )
        arrays["evaluation_wall_seconds"] = np.append(
            arrays["evaluation_wall_seconds"], time.perf_counter() - began
        )
        progress["extraction_times_completed"] = index + 1
        np.savez_compressed(EXTRACTION_PATH, **arrays)
        _save_progress(progress)
        print(
            f"c4e8-extraction: {index + 1}/{base['accepted_times'].size} "
            f"t={base['accepted_times'][index]:.8e}",
            flush=True,
        )
    report = {
        "passed": bool(
            float(np.max(arrays["maximum_identity_defects"])) <= 1.0e-12
            and float(np.max(arrays["maximum_ledger_audits"][:, :3])) <= 1.0e-12
            and int(np.max(arrays["maximum_ledger_audits"][:, 3])) == 0
        ),
        "accepted_times": int(arrays["accepted_times"].size),
        "maximum_identity_defect": float(
            np.max(arrays["maximum_identity_defects"])
        ),
        "maximum_ledger_audit": float(
            np.max(arrays["maximum_ledger_audits"][:, :3])
        ),
        "maximum_incoming_excision_characteristics": int(
            np.max(arrays["maximum_ledger_audits"][:, 3])
        ),
        "wall_seconds": float(np.sum(arrays["evaluation_wall_seconds"])),
    }
    progress["reports"]["extraction"] = report
    _save_progress(progress)
    return report, arrays


def _run_remainder(progress, configuration, frozen_tangent, base, tangent, extraction):
    if REMAINDER_PATH.exists():
        arrays = _load_npz(REMAINDER_PATH)
    else:
        arrays = {
            "endpoint_times": np.empty(0, dtype=float),
            "state_correction_fractions": np.empty(0, dtype=float),
            "extraction_correction_fractions": np.empty(0, dtype=float),
            "maximum_scaled_residuals": np.empty(0, dtype=float),
            "wall_seconds": np.empty(0, dtype=float),
        }
    audit_indices = [
        index
        for index, value in enumerate(base["accepted_times"][1:], start=1)
        if c4e3._time_us(value) in set(AUDIT_TARGET_MICROSECONDS)
    ]
    context = configuration["context"]
    field_scales = tangent["field_scales"]
    extraction_scales = _load_npz(c4e3.extraction5.DECISIVE_ARRAYS)["export_scales"]
    start = int(progress.get("remainder_steps_completed", 0))
    for position in range(start, len(audit_indices)):
        endpoint_index = audit_indices[position]
        old_index = endpoint_index - 1
        old_state = (
            base["accepted_states"][old_index]
            + tangent["state_directions"][old_index, GENERIC_INDEX]
        )
        history = h2b1.h2a2._history(
            base["accepted_primitive_histories"][old_index]
            + tangent["primitive_history_directions"][old_index, GENERIC_INDEX],
            base["accepted_mapped_histories"][old_index]
            + tangent["mapped_history_directions"][old_index, GENERIC_INDEX],
            base["accepted_height_histories"][old_index]
            + tangent["height_history_directions"][old_index, GENERIC_INDEX],
            base["accepted_previous_timesteps"][old_index],
        )
        began = time.perf_counter()
        result = advance_causal_five_field_monolithic_bdf(
            context,
            old_state,
            float(base["accepted_timesteps"][old_index]),
            frozen_tangent,
            order=2,
            history=history,
            residual_tolerance=1.0e-10,
            ledger_tolerance=1.0e-12,
            maximum_scaled_primitive_change=5.0e-3,
        )
        if result.history is None:
            raise RuntimeError("c4e8 sampled nonlinear shadow lacks history")
        record = controller._step_record(result)
        actual_state_response = (
            result.primitive_charts - base["accepted_states"][endpoint_index]
        )
        predicted_state_response = tangent["state_directions"][
            endpoint_index, GENERIC_INDEX
        ]
        state_correction = float(
            np.max(
                np.abs(actual_state_response - predicted_state_response)
                / field_scales[None, :]
            )
            / max(
                np.max(np.abs(actual_state_response) / field_scales[None, :]),
                np.finfo(float).tiny,
            )
        )
        actual_extraction, _identity, _audit = c4e3._extraction_value(
            context, result.primitive_charts
        )
        actual_extraction -= extraction["base_values"][endpoint_index]
        predicted_extraction = extraction["tangent_directions"][
            endpoint_index, GENERIC_INDEX
        ]
        extraction_correction = float(
            np.max(
                np.abs(actual_extraction - predicted_extraction) / extraction_scales
            )
            / max(
                np.max(np.abs(actual_extraction) / extraction_scales),
                np.finfo(float).tiny,
            )
        )
        arrays["endpoint_times"] = np.append(
            arrays["endpoint_times"], base["accepted_times"][endpoint_index]
        )
        arrays["state_correction_fractions"] = np.append(
            arrays["state_correction_fractions"], state_correction
        )
        arrays["extraction_correction_fractions"] = np.append(
            arrays["extraction_correction_fractions"], extraction_correction
        )
        arrays["maximum_scaled_residuals"] = np.append(
            arrays["maximum_scaled_residuals"], record["maximum_scaled_residual"]
        )
        arrays["wall_seconds"] = np.append(
            arrays["wall_seconds"], time.perf_counter() - began
        )
        progress["remainder_steps_completed"] = position + 1
        np.savez_compressed(REMAINDER_PATH, **arrays)
        _save_progress(progress)
        print(
            f"c4e8-shadow: t={base['accepted_times'][endpoint_index]:.8e} "
            f"state={state_correction:.3e} extraction={extraction_correction:.3e}",
            flush=True,
        )
    maximum = _read_json(c4e7.MANIFEST_PATH)["nonlinear_remainder"]
    report = {
        "passed": bool(
            float(np.max(arrays["state_correction_fractions"]))
            <= maximum["maximum_correction_fraction_of_generic_response"]
            and float(np.max(arrays["extraction_correction_fractions"]))
            <= maximum["maximum_correction_fraction_of_generic_response"]
            and float(np.max(arrays["maximum_scaled_residuals"])) <= 1.0e-10
        ),
        "sampled_shadows": int(arrays["endpoint_times"].size),
        "maximum_state_correction_fraction": float(
            np.max(arrays["state_correction_fractions"])
        ),
        "maximum_extraction_correction_fraction": float(
            np.max(arrays["extraction_correction_fractions"])
        ),
        "maximum_scaled_residual": float(
            np.max(arrays["maximum_scaled_residuals"])
        ),
        "wall_seconds": float(np.sum(arrays["wall_seconds"])),
        "middle_fine_fraction_deferred_to_spatial_analysis": True,
    }
    progress["reports"]["remainder"] = report
    _save_progress(progress)
    return report, arrays


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
    setup_seconds,
    base_report,
    base,
    tangent_report,
    tangent,
    extraction_report,
    extraction,
    remainder_report,
    remainder,
    replay,
):
    replay_passed = bool(
        replay["checkpoint_roundtrip_bitwise"]
        and replay["last_step_replay_bitwise"]
        and replay["maximum_scaled_residual"] <= 1.0e-10
    )
    passed = bool(
        base_report["passed"]
        and tangent_report["passed"]
        and extraction_report["passed"]
        and remainder_report["passed"]
        and replay_passed
    )
    classification = (
        "fine_20ms_base_block_tangent_passed_three_grid_spatial_analysis_authorized"
        if passed
        else "fine_20ms_base_or_tangent_failed_spatial_certificate_blocked"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "setup_wall_seconds": setup_seconds,
        "base": base_report,
        "tangent": tangent_report,
        "extraction_tangent": extraction_report,
        "sampled_nonlinear_remainder": remainder_report,
        "serialized_base_replay": replay,
        "fine_twenty_ms_computation_completed": passed,
        "fine_twenty_ms_spatial_certificate_issued": False,
        "three_grid_spatial_analysis_authorized": passed,
        "full_fine_generic_anchor_required": not remainder_report["passed"],
        "full_fine_generic_anchor_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4e9_three_grid_20ms_spatial_analysis"
            if passed
            else "fine_20ms_failure_localization_only"
        ),
    }
    combined = {
        **{f"base__{key}": value for key, value in base.items()},
        **{f"tangent__{key}": value for key, value in tangent.items()},
        **{f"extraction__{key}": value for key, value in extraction.items()},
        **{f"remainder__{key}": value for key, value in remainder.items()},
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **combined)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "layout": FINE_LAYOUT,
            "profiles": PROFILES,
            "generic_profile": PROFILES[GENERIC_INDEX],
            "coupling_face": COUPLING_FACE,
            "extraction_face": EXTRACTION_FACE,
            "target_microseconds": TARGET_MICROSECONDS,
            "audit_target_microseconds": AUDIT_TARGET_MICROSECONDS,
            "maximum_timestep_seconds": MAXIMUM_TIMESTEP_SECONDS,
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
                "fine_manifest": _sha256(c4e7.MANIFEST_PATH),
                "fine_5ms_summary": _sha256(fine5.SUMMARY_PATH),
                "fine_5ms_arrays": _sha256(fine5.DECISIVE_ARRAYS),
            },
            "implementation_source_hashes": _source_identity(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER} --through complete",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Cost-bounded fine 20 ms base and tangent WP10c9d6c7c3b5c4e8",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"The fine nonlinear base completed 5 to 20 ms in `{base_report['accepted_steps']}` accepted steps with `{base_report['audited_steps']}` full-versus-two-half audits.",
                "",
                f"The maximum bounded local temporal error was `{base_report['maximum_local_error_bound']:.6e}`; the block tangent and conservative extraction tangent passed their method ledgers.",
                "",
                f"The sampled one-step nonlinear remainder was `{remainder_report['maximum_state_correction_fraction']:.6e}` of the state response and `{remainder_report['maximum_extraction_correction_fraction']:.6e}` of the extraction response.",
                "",
                f"Authorized next: `{summary['authorized_next']}`.",
                "",
                "The three-grid spatial certificate, 50 ms propagation, fixed-Q experiments, and reduced slow evolution remain blocked pending the separate spatial analysis.",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--through", choices=("pilot", "base", "tangent", "complete"), default="complete"
    )
    arguments = parser.parse_args(argv)
    _validate_parent()
    _patch_modules()
    progress = _progress()
    configuration = h2b1._configuration()
    print("c4e8: build fine frozen tangent", flush=True)
    frozen_tangent, setup_seconds = h2b1._build_frozen_tangent(configuration)
    parent = _parent_arrays()
    field_scales = np.asarray(parent["tangent__field_scales"], dtype=float)
    export_scales = np.asarray(parent["tangent__export_scales"], dtype=float)
    extraction_scales = _load_npz(c4e3.extraction5.DECISIVE_ARRAYS)["export_scales"]
    contract, _strict = h2b1.h2a2.h2.g._controller_contracts()
    contract = dict(contract)
    contract["maximum_timestep_seconds"] = MAXIMUM_TIMESTEP_SECONDS
    manifest = _read_json(c4e7.MANIFEST_PATH)

    global _PILOT_STOP_ACTIVE
    _PILOT_STOP_ACTIVE = arguments.through == "pilot"
    try:
        base_report, base = c4e3._run_base(
            progress,
            configuration,
            frozen_tangent,
            field_scales,
            export_scales,
            extraction_scales,
            contract,
            manifest,
        )
    except _PilotComplete:
        base = _load_npz(BASE_PATH)
        report = _pilot_report(base, progress)
        print(json.dumps(_plain(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 2
    finally:
        _PILOT_STOP_ACTIVE = False
    if arguments.through == "base":
        print(json.dumps(_plain(base_report), indent=2, sort_keys=True))
        return 0 if base_report["passed"] else 2

    tangent_report, tangent = h2b1._run_tangent(progress, configuration, base)
    if arguments.through == "tangent":
        print(json.dumps(_plain(tangent_report), indent=2, sort_keys=True))
        return 0 if tangent_report["passed"] else 2

    extraction_report, extraction = _run_extraction(
        progress, configuration, base, tangent
    )
    remainder_report, remainder = _run_remainder(
        progress,
        configuration,
        frozen_tangent,
        base,
        tangent,
        extraction,
    )
    replay = h2b1._serialized_last_step_replay(
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
    )
    return _finalize(
        setup_seconds,
        base_report,
        base,
        tangent_report,
        tangent,
        extraction_report,
        extraction,
        remainder_report,
        remainder,
        replay,
    )


if __name__ == "__main__":
    raise SystemExit(main())
