#!/usr/bin/env python3
"""Run the fail-fast 2e-3 s screen for the third nonlinear duration rung."""

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

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402
import run_causal_inner_nonlinear_canonical_time_replay_audit_wp10c9d6c7c3b5c2e1 as c2e1  # noqa: E402
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_manifest_wp10c9d6c7c3b5c3a as c3a  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistory,
    CausalFiveFieldMonolithicBDFRestart,
    causal_five_field_monolithic_frozen_tangent,
    load_causal_five_field_monolithic_bdf_restart,
    save_causal_five_field_monolithic_bdf_restart,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3b"
ANALYZED_BASE_COMMIT = "baee3ba4b3b3c42e9d0ea3b444b0157e443e8275"
ANALYZED_BASE_PARENT = "7620f5819a3a3bdd629a3a7f7c2fe75044f47fcf"
ANALYZED_BASE_TREE = "91315c8e6843b70431456960a6d4b720c141864d"

ARTIFACT = "causal_inner_nonlinear_third_duration_rung_screen_wp10c9d6c7c3b5c3b"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_third_duration_rung_screen_"
    "wp10c9d6c7c3b5c3b.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_third_duration_rung_screen_"
    "wp10c9d6c7c3b5c3b.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_THIRD_DURATION_RUNG_SCREEN_"
    "WP10C9D6C7C3B5C3B_2026-08-04.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c3a.CANONICAL_DIRECTORY
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


def _write_json(path: Path, payload: dict) -> None:
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
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    manifest = _read_json(c3a.MANIFEST_PATH)
    if (
        not parent["passed"]
        or not parent["third_duration_rung_screen_authorized"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c3b_third_duration_rung_screen"
        or parent["propagation_executed"]
        or manifest["classification"]
        != "third_nonlinear_duration_rung_manifest_frozen_two_e_minus_three_second_screen_authorized"
    ):
        raise RuntimeError("c3b authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3b analyzed identity changed")
    return parent, manifest


def _restart_from_c2e1(
    arrays: dict[str, np.ndarray], trajectory: str
) -> CausalFiveFieldMonolithicBDFRestart:
    return CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(
            arrays[f"{trajectory}__canonical_final_state"], copy=True
        ),
        history=CausalFiveFieldMonolithicBDFHistory(
            previous_primitive_increment=np.array(
                arrays[f"{trajectory}__canonical_final_primitive_history"],
                copy=True,
            ),
            previous_mapped_storage_increment=np.array(
                arrays[f"{trajectory}__canonical_final_mapped_history"],
                copy=True,
            ),
            previous_responsive_height_storage_increment=np.array(
                arrays[f"{trajectory}__canonical_final_height_history"],
                copy=True,
            ),
            previous_timestep_seconds=float(
                arrays[f"{trajectory}__canonical_final_previous_timestep"]
            ),
        ),
        elapsed_time_seconds=c3a.RUNG_START_SECONDS,
        completed_steps=1,
        next_order=2,
        provenance={
            "work_package": WORK_PACKAGE,
            "trajectory": trajectory,
            "source": "committed_c2e1_canonical_final_restart",
        },
    )


def _segment_report(segment: dict, contract: dict) -> dict:
    records = segment["step_records"]
    gates = contract["step_method_gates"]
    report = {
        "accepted_comparisons": len(segment["accepted_timesteps"]),
        "rejected_attempts": int(np.sum(segment["retries"])),
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
            item["maximum_mapped_endpoint_path_closure_defect"] for item in records
        ),
        "minimum_path_reconstruction_factor": min(
            item["minimum_path_reconstruction_factor"] for item in records
        ),
        "maximum_incoming_excision_characteristics": max(
            item["incoming_excision_characteristics"] for item in records
        ),
        "maximum_export_ledger_defect": float(
            segment["maximum_export_ledger_defect"]
        ),
    }
    report["method_passed"] = bool(
        all(item["accepted"] for item in records)
        and report["maximum_scaled_residual"] <= gates["maximum_scaled_residual"]
        and report["maximum_discrete_ledger_defect"]
        <= gates["maximum_discrete_ledger_defect"]
        and report["maximum_mapped_endpoint_path_closure_defect"]
        <= gates["maximum_mapped_endpoint_path_closure"]
        and report["minimum_path_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"] - 1.0e-12
        and report["maximum_incoming_excision_characteristics"]
        <= gates["maximum_incoming_excision_characteristics"]
    )
    return report


def _save_trajectory_stage(
    trajectory: str,
    context,
    payload: dict,
) -> None:
    directory = PROGRESS_DIRECTORY / trajectory
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
            "trajectory": trajectory,
            "passed": payload["passed"],
            "main_report": payload["main_report"],
            "replay_report": payload["replay_report"],
            "strict_report": payload["strict_report"],
            "replay_bitwise": payload["replay_bitwise"],
            "final_state_audit": payload["final_state_audit"],
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
        },
    )


def _load_trajectory_stage(trajectory: str, context) -> dict | None:
    directory = PROGRESS_DIRECTORY / trajectory
    summary_path = directory / "summary.json"
    arrays_path = directory / "arrays.npz"
    restart_path = directory / "final_restart.npz"
    if not (summary_path.exists() and arrays_path.exists() and restart_path.exists()):
        return None
    summary = _read_json(summary_path)
    if (
        not summary["passed"]
        or summary["runner_sha256"] != _sha256(ROOT / THIS_RUNNER)
    ):
        return None
    arrays = _load_npz(arrays_path)
    payload = {
        "passed": summary["passed"],
        "main_report": summary["main_report"],
        "replay_report": summary["replay_report"],
        "strict_report": summary["strict_report"],
        "replay_bitwise": summary["replay_bitwise"],
        "final_state_audit": summary["final_state_audit"],
        "final_restart": load_causal_five_field_monolithic_bdf_restart(
            restart_path, context
        ),
    }
    for segment_name in ("main", "replay", "strict"):
        payload[segment_name] = {
            name: arrays[f"{segment_name}__{name}"]
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


def _trajectory_stage(
    trajectory: str,
    configuration: dict,
    tangent,
    restart: CausalFiveFieldMonolithicBDFRestart,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    manifest: dict,
) -> dict:
    cached = _load_trajectory_stage(trajectory, configuration["context"])
    if cached is not None:
        print(f"c3b: reuse durable {trajectory} stage", flush=True)
        return cached
    print(f"c3b: {trajectory} main", flush=True)
    main = c2._controller_segment(
        configuration,
        tangent,
        restart.primitive_charts,
        restart.history,
        c3a.RUNG_START_SECONDS,
        manifest["main_controller"]["initial_timestep_seconds"],
        field_scales,
        export_scales,
        c2.COUPLING_FACE,
        manifest["main_controller"],
        output_times=c3a.MAIN_TARGETS_SECONDS,
        stop_time=c3a.SCREEN_HORIZON_SECONDS,
        checkpoint_times=(
            float(c3a.REPLAY_TARGETS_SECONDS[0]),
            float(c3a.STRICT_TARGETS_SECONDS[0]),
        ),
        include_initial_output=True,
    )
    main_report = _segment_report(main, manifest["main_controller"])
    gates = manifest["binding_gates"]
    main_report["passed"] = bool(
        main_report["method_passed"]
        and main_report["maximum_local_error_estimate"]
        <= gates["main_local_error_maximum"]
        and main_report["sum_local_error_estimates"]
        <= gates["main_local_error_sum_maximum"]
    )
    if not main_report["passed"]:
        raise RuntimeError(f"{trajectory} main screen failed")

    replay_key = min(
        main["checkpoints"],
        key=lambda value: abs(value - float(c3a.REPLAY_TARGETS_SECONDS[0])),
    )
    strict_key = min(
        main["checkpoints"],
        key=lambda value: abs(value - float(c3a.STRICT_TARGETS_SECONDS[0])),
    )
    replay_checkpoint, replay_roundtrip = c2._save_restore(
        configuration["context"],
        main["checkpoints"][replay_key]["restart"],
        f"c3b_{trajectory}_replay",
    )
    print(f"c3b: {trajectory} replay", flush=True)
    replay = c2._controller_segment(
        configuration,
        tangent,
        replay_checkpoint.primitive_charts,
        replay_checkpoint.history,
        float(c3a.REPLAY_TARGETS_SECONDS[0]),
        main["checkpoints"][replay_key]["next_timestep"],
        field_scales,
        export_scales,
        c2.COUPLING_FACE,
        manifest["main_controller"],
        output_times=c3a.REPLAY_TARGETS_SECONDS,
        stop_time=c3a.SCREEN_HORIZON_SECONDS,
        include_initial_output=True,
    )
    replay_report = _segment_report(replay, manifest["main_controller"])
    main_indices = [
        int(
            np.flatnonzero(
                np.isclose(c3a.MAIN_TARGETS_SECONDS, target, rtol=0.0, atol=1e-18)
            )[0]
        )
        for target in c3a.REPLAY_TARGETS_SECONDS
    ]
    replay_bitwise = {
        "initial_restart_roundtrip_bitwise": replay_roundtrip,
        "target_labels_bitwise": np.array_equal(
            replay["output_times"], c3a.REPLAY_TARGETS_SECONDS
        ),
        "states_bitwise": np.array_equal(
            replay["output_states"], main["output_states"][main_indices]
        ),
        "Tier_I_exports_bitwise": np.array_equal(
            replay["output_exports"], main["output_exports"][main_indices]
        ),
    }
    replay_report["passed"] = bool(
        replay_report["method_passed"] and all(replay_bitwise.values())
    )
    if not replay_report["passed"]:
        raise RuntimeError(f"{trajectory} replay screen failed")

    strict_checkpoint, strict_roundtrip = c2._save_restore(
        configuration["context"],
        main["checkpoints"][strict_key]["restart"],
        f"c3b_{trajectory}_strict",
    )
    print(f"c3b: {trajectory} strict", flush=True)
    strict = c2._controller_segment(
        configuration,
        tangent,
        strict_checkpoint.primitive_charts,
        strict_checkpoint.history,
        float(c3a.STRICT_TARGETS_SECONDS[0]),
        manifest["strict_controller"]["initial_timestep_seconds"],
        field_scales,
        export_scales,
        c2.COUPLING_FACE,
        manifest["strict_controller"],
        output_times=c3a.STRICT_TARGETS_SECONDS,
        stop_time=c3a.SCREEN_HORIZON_SECONDS,
        include_initial_output=True,
    )
    strict_report = _segment_report(strict, manifest["strict_controller"])
    strict_report["initial_restart_roundtrip_bitwise"] = strict_roundtrip
    strict_report["passed"] = bool(
        strict_report["method_passed"]
        and strict_report["maximum_local_error_estimate"]
        <= gates["strict_local_error_maximum"]
        and strict_roundtrip
    )
    if not strict_report["passed"]:
        raise RuntimeError(f"{trajectory} strict screen failed")
    readiness = c3b1a._state_audit(configuration["context"], main["final_state"])
    final_restart = CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(main["final_state"], copy=True),
        history=main["final_history"],
        elapsed_time_seconds=c3a.SCREEN_HORIZON_SECONDS,
        completed_steps=len(main["accepted_timesteps"]),
        next_order=2,
        provenance={
            "work_package": WORK_PACKAGE,
            "trajectory": trajectory,
            "source": "canonical_main_screen",
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
        "final_state_audit": readiness,
        "final_restart": final_restart,
    }
    _save_trajectory_stage(trajectory, configuration["context"], payload)
    return payload


def _scaled_max(left: np.ndarray, right: np.ndarray, scales: np.ndarray) -> float:
    return float(np.max(np.abs((np.asarray(left) - np.asarray(right)) / scales)))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def _response_report(
    base: dict,
    perturbed: dict,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    manifest: dict,
) -> dict:
    main_state = perturbed["main"]["output_states"] - base["main"]["output_states"]
    main_export = perturbed["main"]["output_exports"] - base["main"]["output_exports"]
    strict_state = (
        perturbed["strict"]["output_states"] - base["strict"]["output_states"]
    )
    strict_export = (
        perturbed["strict"]["output_exports"] - base["strict"]["output_exports"]
    )
    main_indices = [
        int(
            np.flatnonzero(
                np.isclose(c3a.MAIN_TARGETS_SECONDS, target, rtol=0.0, atol=1e-18)
            )[0]
        )
        for target in (c3a.STRICT_TARGETS_SECONDS[0], c3a.STRICT_TARGETS_SECONDS[-1])
    ]
    strict_indices = (0, strict_state.shape[0] - 1)
    main_state_endpoints = main_state[main_indices]
    main_export_endpoints = main_export[main_indices]
    strict_state_endpoints = strict_state[list(strict_indices)]
    strict_export_endpoints = strict_export[list(strict_indices)]
    report = {
        "maximum_scaled_state_difference": _scaled_max(
            main_state_endpoints,
            strict_state_endpoints,
            field_scales[None, None, :],
        ),
        "maximum_scaled_Tier_I_difference": _scaled_max(
            main_export_endpoints,
            strict_export_endpoints,
            export_scales[None, :],
        ),
        "state_history_cosine": _cosine(
            main_state_endpoints / field_scales[None, None, :],
            strict_state_endpoints / field_scales[None, None, :],
        ),
        "Tier_I_history_cosine": _cosine(
            main_export_endpoints / export_scales[None, :],
            strict_export_endpoints / export_scales[None, :],
        ),
    }
    gates = manifest["binding_gates"]
    report["passed"] = bool(
        report["maximum_scaled_state_difference"]
        <= gates["strict_response_maximum_scaled_state_difference"]
        and report["maximum_scaled_Tier_I_difference"]
        <= gates["strict_response_maximum_scaled_Tier_I_difference"]
        and report["state_history_cosine"]
        >= gates["strict_response_history_cosine_minimum"]
        and report["Tier_I_history_cosine"]
        >= gates["strict_response_history_cosine_minimum"]
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


def main() -> int:
    started = time.perf_counter()
    parent, manifest = _validate_parent()
    source = _load_npz(c2e1.DECISIVE_ARRAYS)
    pilot = _load_npz(c3b2b.CANONICAL_DIRECTORY / "decisive_arrays.npz")
    field_scales = pilot["field_scales"]
    export_scales = pilot["fixed_physical_observable_scales"]
    configuration = c3b1a._configurations()[c2.LAYOUT]
    print("c3b: build tangent", flush=True)
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    base = _trajectory_stage(
        "base",
        configuration,
        tangent,
        _restart_from_c2e1(source, "base"),
        field_scales,
        export_scales,
        manifest,
    )
    perturbed = _trajectory_stage(
        "perturbed",
        configuration,
        tangent,
        _restart_from_c2e1(source, "perturbed"),
        field_scales,
        export_scales,
        manifest,
    )
    response = _response_report(base, perturbed, field_scales, export_scales, manifest)
    readiness = {}
    for trajectory, payload in (("base", base), ("perturbed", perturbed)):
        audit = payload["final_state_audit"]
        gates = manifest["binding_gates"]
        readiness[trajectory] = {
            **audit,
            "passed": bool(
                audit["minimum_scattering_optical_depth"]
                >= gates["minimum_scattering_optical_depth"]
                and audit["maximum_h_over_r"] <= gates["maximum_h_over_r"]
                and audit["minimum_reconstruction_factor"]
                >= gates["minimum_reconstruction_factor"] - 1.0e-12
            ),
        }
    passed = bool(
        base["passed"]
        and perturbed["passed"]
        and response["passed"]
        and all(item["passed"] for item in readiness.values())
    )
    classification = (
        manifest["positive_branch"]["classification"]
        if passed
        else manifest["negative_branch"]["classification"]
    )
    authorized_next = (
        manifest["positive_branch"]["authorized_next"]
        if passed
        else manifest["negative_branch"]["authorized_next"]
    )
    arrays = {
        "main_times_seconds": base["main"]["output_times"],
        "strict_times_seconds": base["strict"]["output_times"],
        "field_scales": field_scales,
        "export_scales": export_scales,
    }
    trajectory_reports = {}
    for trajectory, payload in (("base", base), ("perturbed", perturbed)):
        trajectory_reports[trajectory] = {
            "passed": payload["passed"],
            "main_report": payload["main_report"],
            "replay_report": payload["replay_report"],
            "strict_report": payload["strict_report"],
            "replay_bitwise": payload["replay_bitwise"],
            "final_state_audit": readiness[trajectory],
        }
        for segment_name in ("main", "replay", "strict"):
            segment = payload[segment_name]
            for name in ("output_times", "output_states", "output_exports"):
                arrays[f"{trajectory}__{segment_name}__{name}"] = segment[name]
        restart = payload["final_restart"]
        arrays[f"{trajectory}__final_state"] = restart.primitive_charts
        arrays[f"{trajectory}__final_primitive_history"] = (
            restart.history.previous_primitive_increment
        )
        arrays[f"{trajectory}__final_mapped_history"] = (
            restart.history.previous_mapped_storage_increment
        )
        arrays[f"{trajectory}__final_height_history"] = (
            restart.history.previous_responsive_height_storage_increment
        )
        arrays[f"{trajectory}__final_previous_timestep"] = np.asarray(
            restart.history.previous_timestep_seconds
        )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "screen_horizon_seconds": c3a.SCREEN_HORIZON_SECONDS,
        "main_targets_seconds": c3a.MAIN_TARGETS_SECONDS,
        "replay_targets_seconds": c3a.REPLAY_TARGETS_SECONDS,
        "strict_targets_seconds": c3a.STRICT_TARGETS_SECONDS,
    }
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "parent_classification_preserved": parent["classification"],
        "historical_c2d_classification_preserved": (
            "second_rung_perturbed_completion_failed_later_duration_blocked"
        ),
        "operator_changed": False,
        "production_defaults_changed": False,
        "trajectory_reports": trajectory_reports,
        "strict_response_comparison": response,
        "elapsed_seconds": time.perf_counter() - started,
        "third_duration_rung_completion_manifest_authorized": passed,
        "third_duration_rung_completion_propagation_authorized": False,
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
                for path in (THIS_RUNNER, THIS_TEST, c3a.THIS_RUNNER, c2e1.THIS_RUNNER)
                if (ROOT / path).exists()
            },
            "input_hashes": {
                "parent_summary": _sha256(PARENT_DIRECTORY / "summary.json"),
                "parent_manifest": _sha256(c3a.MANIFEST_PATH),
                "canonical_restart_arrays": _sha256(c2e1.DECISIVE_ARRAYS),
            },
        },
    )
    REPORT_PATH.write_text(
        "# Third nonlinear duration-rung screen WP10c9d6c7c3b5c3b\n\n"
        "## Classification\n\n"
        f"`{classification}`\n\n"
        f"Base stage passes: `{base['passed']}`.\n\n"
        f"Perturbed stage passes: `{perturbed['passed']}`.\n\n"
        f"Strict response passes: `{response['passed']}`.\n\n"
        f"Authorized next: `{authorized_next}`.\n\n"
        "Fixed-Q experiments and reduced evolution remain blocked.\n",
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


if __name__ == "__main__":
    raise SystemExit(main())
