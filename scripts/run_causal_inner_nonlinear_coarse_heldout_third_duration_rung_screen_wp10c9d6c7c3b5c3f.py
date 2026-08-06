#!/usr/bin/env python3
"""Run the frozen coarse held-out nonlinear duration screen through 5 ms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_screen_wp10c9d6c7c3b5c3b as c3b  # noqa: E402
import run_causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d as c3d  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_breadth_manifest_wp10c9d6c7c3b5c3e as c3e  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_temporal_wp10c9d6c7c3b4b2 as temporal  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFRestart,
    causal_five_field_monolithic_bdf_history,
    causal_five_field_monolithic_frozen_tangent,
    evaluate_causal_five_field_monolithic_backward_euler,
    load_causal_five_field_monolithic_bdf_restart,
    save_causal_five_field_monolithic_bdf_restart,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3f"
ANALYZED_BASE_COMMIT = "1c97bea290c0d0e649be0aedf94293074b76013d"
ANALYZED_BASE_PARENT = "bafc167cfc0871c892982dc13bbb74e2fae5db22"
ANALYZED_BASE_TREE = "b543dc71a70b10466dd173a52cd37700715c81d7"

ARTIFACT = (
    "causal_inner_nonlinear_coarse_heldout_third_duration_rung_screen_"
    "wp10c9d6c7c3b5c3f"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_coarse_heldout_third_duration_rung_"
    "screen_wp10c9d6c7c3b5c3f.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_coarse_heldout_third_duration_rung_"
    "screen_wp10c9d6c7c3b5c3f.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_COARSE_HELDOUT_THIRD_DURATION_RUNG_"
    "SCREEN_WP10C9D6C7C3B5C3F_2026-08-04.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PROGRESS_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT


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
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
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
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(c3e.SUMMARY_PATH)
    manifest = _read_json(c3e.MANIFEST_PATH)
    if (
        not parent["passed"]
        or parent["propagation_executed"]
        or not parent["coarse_heldout_duration_screen_authorized"]
        or parent["coarse_heldout_duration_propagation_executed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c3f_coarse_heldout_third_duration_rung_screen"
        or manifest["classification"]
        != "third_duration_rung_breadth_manifest_frozen_coarse_heldout_duration_screen_authorized"
    ):
        raise RuntimeError("c3f authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3f analyzed identity changed")
    return parent, manifest


def _initial_restart(profile: str, configuration: dict) -> CausalFiveFieldMonolithicBDFRestart:
    arrays = _load_npz(temporal.DECISIVE_ARRAYS)
    key = f"{profile}__p1__dt_2p5em06__states"
    states = np.asarray(arrays[key], dtype=float)
    previous = np.array(states[-2], copy=True)
    current = np.array(states[-1], copy=True)
    dt = c3e.INITIAL_HELDOUT_PREVIOUS_TIMESTEP_SECONDS
    evaluation = evaluate_causal_five_field_monolithic_backward_euler(
        previous,
        current,
        dt,
        configuration["context"],
    )
    history = causal_five_field_monolithic_bdf_history(
        current - previous,
        evaluation.storage_increment,
        dt,
    )
    return CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=current,
        history=history,
        elapsed_time_seconds=float(c3e.INITIAL_HELDOUT_HISTORY_SECONDS[-1]),
        completed_steps=1,
        next_order=2,
        provenance={
            "work_package": WORK_PACKAGE,
            "profile": profile,
            "source": "committed_b4b2_dt_2p5em06_final_two_states",
            "source_array": key,
        },
    )


def _cache_directory(profile: str) -> Path:
    return PROGRESS_DIRECTORY / profile


def _save_stage(profile: str, context, payload: dict) -> None:
    directory = _cache_directory(profile)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for segment_name in ("main", "replay", "strict"):
        segment = payload[segment_name]
        for name in (
            "output_times",
            "output_states",
            "output_exports",
            "accepted_times",
            "accepted_timesteps",
            "local_error_estimates",
            "retries",
        ):
            arrays[f"{segment_name}__{name}"] = segment[name]
    np.savez_compressed(directory / "arrays.npz", **arrays)
    save_causal_five_field_monolithic_bdf_restart(
        directory / "final_restart.npz", context, payload["final_restart"]
    )
    _write_json(
        directory / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "profile": profile,
            "passed": payload["passed"],
            "main_report": payload["main_report"],
            "replay_report": payload["replay_report"],
            "strict_report": payload["strict_report"],
            "replay_bitwise": payload["replay_bitwise"],
            "strict_response_comparison": payload["strict_response_comparison"],
            "final_state_audit": payload["final_state_audit"],
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
        },
    )


def _load_stage(profile: str, context) -> dict | None:
    directory = _cache_directory(profile)
    summary_path = directory / "summary.json"
    arrays_path = directory / "arrays.npz"
    restart_path = directory / "final_restart.npz"
    if not (summary_path.exists() and arrays_path.exists() and restart_path.exists()):
        return None
    summary = _read_json(summary_path)
    if not summary["passed"] or summary["runner_sha256"] != _sha256(
        ROOT / THIS_RUNNER
    ):
        return None
    source = _load_npz(arrays_path)
    payload = {
        "passed": summary["passed"],
        "main_report": summary["main_report"],
        "replay_report": summary["replay_report"],
        "strict_report": summary["strict_report"],
        "replay_bitwise": summary["replay_bitwise"],
        "strict_response_comparison": summary["strict_response_comparison"],
        "final_state_audit": summary["final_state_audit"],
        "final_restart": load_causal_five_field_monolithic_bdf_restart(
            restart_path, context
        ),
    }
    for segment_name in ("main", "replay", "strict"):
        payload[segment_name] = {
            name: source[f"{segment_name}__{name}"]
            for name in (
                "output_times",
                "output_states",
                "output_exports",
                "accepted_times",
                "accepted_timesteps",
                "local_error_estimates",
                "retries",
            )
        }
    return payload


def _base_segments() -> dict:
    arrays = _load_npz(c3d.DECISIVE_ARRAYS)
    return {
        segment: {
            "output_times": arrays[f"base__{segment}__output_times"],
            "output_states": arrays[f"base__{segment}__output_states"],
            "output_exports": arrays[f"base__{segment}__output_exports"],
        }
        for segment in ("main", "replay", "strict")
    }


def _response_report(
    profile_payload: dict,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    manifest: dict,
) -> dict:
    base = _base_segments()
    main_state = (
        profile_payload["main"]["output_states"] - base["main"]["output_states"]
    )
    main_export = (
        profile_payload["main"]["output_exports"]
        - base["main"]["output_exports"]
    )
    strict_state = (
        profile_payload["strict"]["output_states"]
        - base["strict"]["output_states"]
    )
    strict_export = (
        profile_payload["strict"]["output_exports"]
        - base["strict"]["output_exports"]
    )
    indices = c3d._target_indices(
        c3e.MAIN_TARGET_MICROSECONDS,
        np.asarray(
            (c3e.STRICT_TARGET_MICROSECONDS[0], c3e.STRICT_TARGET_MICROSECONDS[-1])
        ),
    )
    main_state = main_state[indices]
    main_export = main_export[indices]
    strict_state = strict_state[[0, -1]]
    strict_export = strict_export[[0, -1]]
    report = {
        "maximum_scaled_state_difference": c3b._scaled_max(
            main_state, strict_state, field_scales[None, None, :]
        ),
        "maximum_scaled_Tier_I_difference": c3b._scaled_max(
            main_export, strict_export, export_scales[None, :]
        ),
        "state_history_cosine": c3b._cosine(
            main_state / field_scales[None, None, :],
            strict_state / field_scales[None, None, :],
        ),
        "Tier_I_history_cosine": c3b._cosine(
            main_export / export_scales[None, :],
            strict_export / export_scales[None, :],
        ),
    }
    stage = manifest["coarse_heldout_duration_stage"]
    report["passed"] = bool(
        report["maximum_scaled_state_difference"]
        <= stage["strict_response_maximum_scaled_state_difference"]
        and report["maximum_scaled_Tier_I_difference"]
        <= stage["strict_response_maximum_scaled_Tier_I_difference"]
        and report["state_history_cosine"]
        >= stage["strict_response_history_cosine_minimum"]
        and report["Tier_I_history_cosine"]
        >= stage["strict_response_history_cosine_minimum"]
    )
    return report


def _trajectory_stage(
    profile: str,
    configuration: dict,
    tangent,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    manifest: dict,
) -> dict:
    cached = _load_stage(profile, configuration["context"])
    if cached is not None:
        print(f"c3f: reuse durable {profile}", flush=True)
        return cached
    restart = _initial_restart(profile, configuration)
    common = manifest["common_contract"]
    stage = manifest["coarse_heldout_duration_stage"]
    main_controller = common["main_controller"]
    strict_controller = common["strict_controller"]
    coupling_face = int(stage["active_coupling_face"])
    print(f"c3f: {profile} main", flush=True)
    main = c2._controller_segment(
        configuration,
        tangent,
        restart.primitive_charts,
        restart.history,
        restart.elapsed_time_seconds,
        main_controller["initial_timestep_seconds"],
        field_scales,
        export_scales,
        coupling_face,
        main_controller,
        output_times=c3e.MAIN_TARGETS_SECONDS,
        stop_time=c3e.HORIZON_SECONDS,
        checkpoint_times=(
            float(c3e.REPLAY_TARGETS_SECONDS[0]),
            float(c3e.STRICT_TARGETS_SECONDS[0]),
        ),
        include_initial_output=False,
    )
    main_report = c3b._segment_report(main, main_controller)
    main_report["passed"] = bool(
        main_report["method_passed"]
        and main_report["accepted_comparisons"] > 0
        and main_report["maximum_local_error_estimate"]
        <= main_controller["error_estimator"]["local_tolerance"]
        and main_report["sum_local_error_estimates"]
        <= main_controller["error_estimator"]
        ["short_horizon_sum_of_accepted_error_estimates"]
    )
    if not main_report["passed"]:
        raise RuntimeError(f"{profile} main duration stage failed")

    replay_start = float(c3e.REPLAY_TARGETS_SECONDS[0])
    strict_start = float(c3e.STRICT_TARGETS_SECONDS[0])
    replay_key = min(main["checkpoints"], key=lambda value: abs(value - replay_start))
    strict_key = min(main["checkpoints"], key=lambda value: abs(value - strict_start))
    replay_restart, replay_roundtrip = c2._save_restore(
        configuration["context"],
        main["checkpoints"][replay_key]["restart"],
        f"c3f_{profile}_replay",
    )
    print(f"c3f: {profile} replay", flush=True)
    replay = c2._controller_segment(
        configuration,
        tangent,
        replay_restart.primitive_charts,
        replay_restart.history,
        replay_start,
        main["checkpoints"][replay_key]["next_timestep"],
        field_scales,
        export_scales,
        coupling_face,
        main_controller,
        output_times=c3e.REPLAY_TARGETS_SECONDS,
        stop_time=c3e.HORIZON_SECONDS,
        include_initial_output=True,
    )
    replay_report = c3b._segment_report(replay, main_controller)
    main_indices = c3d._target_indices(
        c3e.MAIN_TARGET_MICROSECONDS, c3e.REPLAY_TARGET_MICROSECONDS
    )
    replay_bitwise = {
        "initial_restart_roundtrip_bitwise": replay_roundtrip,
        "target_labels_bitwise": np.array_equal(
            replay["output_times"], c3e.REPLAY_TARGETS_SECONDS
        ),
        "states_bitwise": np.array_equal(
            replay["output_states"], main["output_states"][main_indices]
        ),
        "Tier_I_exports_bitwise": np.array_equal(
            replay["output_exports"], main["output_exports"][main_indices]
        ),
        **c3d._history_bitwise(main["final_history"], replay["final_history"]),
        "final_state_bitwise": np.array_equal(
            main["final_state"], replay["final_state"]
        ),
    }
    replay_report["passed"] = bool(
        replay_report["method_passed"] and all(replay_bitwise.values())
    )
    if not replay_report["passed"]:
        raise RuntimeError(f"{profile} replay failed")

    strict_restart, strict_roundtrip = c2._save_restore(
        configuration["context"],
        main["checkpoints"][strict_key]["restart"],
        f"c3f_{profile}_strict",
    )
    print(f"c3f: {profile} strict", flush=True)
    strict = c2._controller_segment(
        configuration,
        tangent,
        strict_restart.primitive_charts,
        strict_restart.history,
        strict_start,
        strict_controller["initial_timestep_seconds"],
        field_scales,
        export_scales,
        coupling_face,
        strict_controller,
        output_times=c3e.STRICT_TARGETS_SECONDS,
        stop_time=c3e.HORIZON_SECONDS,
        include_initial_output=True,
    )
    strict_report = c3b._segment_report(strict, strict_controller)
    strict_report["initial_restart_roundtrip_bitwise"] = strict_roundtrip
    strict_report["passed"] = bool(
        strict_report["method_passed"]
        and strict_report["maximum_local_error_estimate"]
        <= strict_controller["error_estimator"]["local_tolerance"]
        and strict_roundtrip
    )
    if not strict_report["passed"]:
        raise RuntimeError(f"{profile} strict stage failed")

    final_audit = c3b1a._state_audit(configuration["context"], main["final_state"])
    readiness_passed = bool(
        final_audit["minimum_scattering_optical_depth"]
        >= common["minimum_scattering_optical_depth"]
        and final_audit["maximum_h_over_r"] <= common["maximum_h_over_r"]
        and final_audit["minimum_reconstruction_factor"]
        >= common["minimum_reconstruction_factor"] - 1.0e-12
    )
    final_audit["passed"] = readiness_passed
    final_restart = CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(main["final_state"], copy=True),
        history=main["final_history"],
        elapsed_time_seconds=c3e.HORIZON_SECONDS,
        completed_steps=len(main["accepted_timesteps"]),
        next_order=2,
        provenance={
            "work_package": WORK_PACKAGE,
            "profile": profile,
            "source": "canonical_main_coarse_heldout_duration_stage",
        },
    )
    payload = {
        "passed": True,
        "main": main,
        "replay": replay,
        "strict": strict,
        "main_report": main_report,
        "replay_report": replay_report,
        "strict_report": strict_report,
        "replay_bitwise": replay_bitwise,
        "final_state_audit": final_audit,
        "final_restart": final_restart,
    }
    response = _response_report(payload, field_scales, export_scales, manifest)
    payload["strict_response_comparison"] = response
    payload["passed"] = bool(readiness_passed and response["passed"])
    if not payload["passed"]:
        raise RuntimeError(f"{profile} response or readiness gate failed")
    _save_stage(profile, configuration["context"], payload)
    return payload


def _run_one_profile(profile: str) -> int:
    _, manifest = _validate_parent()
    if profile not in c3e.COARSE_EXECUTION_ORDER:
        raise ValueError(f"profile is not authorized: {profile}")
    c3d_arrays = _load_npz(c3d.DECISIVE_ARRAYS)
    field_scales = np.asarray(c3d_arrays["field_scales"], dtype=float)
    export_scales = np.asarray(c3d_arrays["export_scales"], dtype=float)
    configuration = c3b1a._configurations()[c3e.COARSE_LAYOUT]
    print(f"c3f: build tangent for {profile}", flush=True)
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    payload = _trajectory_stage(
        profile,
        configuration,
        tangent,
        field_scales,
        export_scales,
        manifest,
    )
    print(
        json.dumps(
            _plain(
                {
                    "profile": profile,
                    "passed": payload["passed"],
                    "main_report": payload["main_report"],
                    "replay_bitwise": payload["replay_bitwise"],
                    "strict_response": payload["strict_response_comparison"],
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


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


def _assemble(completed: list[str], failed_profile: str | None, elapsed: float) -> int:
    parent, manifest = _validate_parent()
    configuration = c3b1a._configurations()[c3e.COARSE_LAYOUT]
    payloads = {
        profile: _load_stage(profile, configuration["context"])
        for profile in completed
    }
    if any(payload is None for payload in payloads.values()):
        raise RuntimeError("completed profile cache is incomplete")
    passed = bool(
        failed_profile is None
        and tuple(completed) == tuple(c3e.COARSE_EXECUTION_ORDER)
        and all(payload["passed"] for payload in payloads.values())
    )
    branch = manifest["positive_branch"] if passed else manifest["negative_branch"]
    arrays = {
        "main_times_seconds": c3e.MAIN_TARGETS_SECONDS,
        "replay_times_seconds": c3e.REPLAY_TARGETS_SECONDS,
        "strict_times_seconds": c3e.STRICT_TARGETS_SECONDS,
    }
    profile_reports = {}
    for profile, payload in payloads.items():
        profile_reports[profile] = {
            "passed": payload["passed"],
            "main_report": payload["main_report"],
            "replay_report": payload["replay_report"],
            "strict_report": payload["strict_report"],
            "replay_bitwise": payload["replay_bitwise"],
            "strict_response_comparison": payload["strict_response_comparison"],
            "final_state_audit": payload["final_state_audit"],
        }
        for segment_name in ("main", "replay", "strict"):
            for name in ("output_states", "output_exports"):
                arrays[f"{profile}__{segment_name}__{name}"] = payload[segment_name][name]
        restart = payload["final_restart"]
        arrays[f"{profile}__final_state"] = restart.primitive_charts
        arrays[f"{profile}__final_primitive_history"] = (
            restart.history.previous_primitive_increment
        )
        arrays[f"{profile}__final_mapped_history"] = (
            restart.history.previous_mapped_storage_increment
        )
        arrays[f"{profile}__final_height_history"] = (
            restart.history.previous_responsive_height_storage_increment
        )
        arrays[f"{profile}__final_previous_timestep"] = np.asarray(
            restart.history.previous_timestep_seconds
        )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "profiles": c3e.COARSE_EXECUTION_ORDER,
        "completed_profiles": completed,
        "failed_profile": failed_profile,
        "layout": c3e.COARSE_LAYOUT,
        "active_coupling_face": c3e.ACTIVE_COUPLING_FACE_INDICES[c3e.COARSE_LAYOUT],
        "initial_history_seconds": c3e.INITIAL_HELDOUT_HISTORY_SECONDS,
        "main_targets_seconds": c3e.MAIN_TARGETS_SECONDS,
        "replay_targets_seconds": c3e.REPLAY_TARGETS_SECONDS,
        "strict_targets_seconds": c3e.STRICT_TARGETS_SECONDS,
    }
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": branch["classification"],
        "passed": passed,
        "authorized_next": branch["authorized_next"],
        "parent_classification_preserved": parent["classification"],
        "historical_c2d_classification_preserved": (
            "second_rung_perturbed_completion_failed_later_duration_blocked"
        ),
        "operator_changed": False,
        "production_defaults_changed": False,
        "completed_profiles": completed,
        "failed_profile": failed_profile,
        "profile_reports": profile_reports,
        "elapsed_seconds": elapsed,
        "coarse_heldout_duration_breadth_certified": passed,
        "third_duration_rung_spatial_confirmation_manifest_authorized": passed,
        "third_duration_rung_spatial_confirmation_propagation_authorized": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
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
            "command": f"PYTHONPATH=src:scripts python3 {THIS_RUNNER}",
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
                for path in (THIS_RUNNER, THIS_TEST, c3e.THIS_RUNNER, c3d.THIS_RUNNER)
                if (ROOT / path).exists()
            },
            "input_hashes": {
                "parent_summary": _sha256(c3e.SUMMARY_PATH),
                "parent_manifest": _sha256(c3e.MANIFEST_PATH),
                "certified_base_arrays": _sha256(c3d.DECISIVE_ARRAYS),
                "heldout_temporal_arrays": _sha256(temporal.DECISIVE_ARRAYS),
            },
        },
    )
    REPORT_PATH.write_text(
        "# Coarse held-out third nonlinear duration-rung screen "
        "WP10c9d6c7c3b5c3f\n\n"
        "## Classification\n\n"
        f"`{summary['classification']}`\n\n"
        f"Completed profiles: `{', '.join(completed)}`.\n\n"
        f"Failed profile: `{failed_profile}`.\n\n"
        f"Authorized next: `{summary['authorized_next']}`.\n\n"
        "Middle/fine propagation, the fourth duration rung, fixed-Q "
        "experiments and reduced evolution remain blocked.\n",
        encoding="utf-8",
    )
    names = ("config.json", "summary.json", "provenance.json", "decisive_arrays.npz")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=c3e.COARSE_EXECUTION_ORDER)
    arguments = parser.parse_args()
    if arguments.profile:
        return _run_one_profile(arguments.profile)
    _validate_parent()
    started = time.perf_counter()
    completed = []
    failed_profile = None
    for profile in c3e.COARSE_EXECUTION_ORDER:
        configuration = c3b1a._configurations()[c3e.COARSE_LAYOUT]
        if _load_stage(profile, configuration["context"]) is not None:
            print(f"c3f: reuse completed profile {profile}", flush=True)
            completed.append(profile)
            continue
        command = (sys.executable, str(ROOT / THIS_RUNNER), "--profile", profile)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts'}"
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if result.returncode != 0:
            failed_profile = profile
            break
        completed.append(profile)
    return _assemble(completed, failed_profile, time.perf_counter() - started)


if __name__ == "__main__":
    raise SystemExit(main())
