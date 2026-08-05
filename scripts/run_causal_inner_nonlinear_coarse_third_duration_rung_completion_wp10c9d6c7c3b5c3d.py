#!/usr/bin/env python3
"""Complete the coarse generic nonlinear duration trajectory through 5e-3 s."""

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
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_completion_manifest_wp10c9d6c7c3b5c3c as c3c  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_screen_wp10c9d6c7c3b5c3b as c3b  # noqa: E402

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
WORK_PACKAGE = "WP10c9d6c7c3b5c3d"
ANALYZED_BASE_COMMIT = "762b49394d6b568b837ac3198212e4803f863ace"
ANALYZED_BASE_PARENT = "79e5aef347028cef396a933897d5f00e250fb49a"
ANALYZED_BASE_TREE = "1a7a9bed7b8ffc151fff54b69be3278598bbb48d"

ARTIFACT = (
    "causal_inner_nonlinear_coarse_third_duration_rung_completion_"
    "wp10c9d6c7c3b5c3d"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_coarse_third_duration_rung_completion_"
    "wp10c9d6c7c3b5c3d.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_coarse_third_duration_rung_completion_"
    "wp10c9d6c7c3b5c3d.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_COARSE_THIRD_DURATION_RUNG_COMPLETION_"
    "WP10C9D6C7C3B5C3D_2026-08-04.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c3c.CANONICAL_DIRECTORY
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
    manifest = _read_json(c3c.MANIFEST_PATH)
    if (
        not parent["passed"]
        or parent["propagation_executed"]
        or not parent["coarse_third_duration_rung_completion_propagation_authorized"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c3d_coarse_third_duration_rung_completion"
        or manifest["classification"]
        != (
            "third_duration_rung_completion_manifest_frozen_coarse_"
            "five_e_minus_three_second_completion_authorized"
        )
    ):
        raise RuntimeError("c3d authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3d analyzed identity changed")
    return parent, manifest


def _restart_from_c3b(
    arrays: dict[str, np.ndarray], trajectory: str
) -> CausalFiveFieldMonolithicBDFRestart:
    return CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(arrays[f"{trajectory}__final_state"], copy=True),
        history=CausalFiveFieldMonolithicBDFHistory(
            previous_primitive_increment=np.array(
                arrays[f"{trajectory}__final_primitive_history"], copy=True
            ),
            previous_mapped_storage_increment=np.array(
                arrays[f"{trajectory}__final_mapped_history"], copy=True
            ),
            previous_responsive_height_storage_increment=np.array(
                arrays[f"{trajectory}__final_height_history"], copy=True
            ),
            previous_timestep_seconds=float(
                arrays[f"{trajectory}__final_previous_timestep"]
            ),
        ),
        elapsed_time_seconds=c3c.RUNG_START_SECONDS,
        completed_steps=1,
        next_order=2,
        provenance={
            "work_package": WORK_PACKAGE,
            "trajectory": trajectory,
            "source": "committed_c3b_final_restart_at_2e-3_s",
        },
    )


def _history_bitwise(left, right) -> dict:
    return {
        "primitive_history_bitwise": np.array_equal(
            left.previous_primitive_increment,
            right.previous_primitive_increment,
        ),
        "mapped_history_bitwise": np.array_equal(
            left.previous_mapped_storage_increment,
            right.previous_mapped_storage_increment,
        ),
        "height_history_bitwise": np.array_equal(
            left.previous_responsive_height_storage_increment,
            right.previous_responsive_height_storage_increment,
        ),
        "previous_timestep_bitwise": (
            left.previous_timestep_seconds == right.previous_timestep_seconds
        ),
    }


def _save_trajectory_stage(trajectory: str, context, payload: dict) -> None:
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
    if not summary["passed"] or summary["runner_sha256"] != _sha256(
        ROOT / THIS_RUNNER
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


def _target_indices(main_microseconds: np.ndarray, selected: np.ndarray) -> list[int]:
    return [
        int(np.flatnonzero(main_microseconds == int(target))[0])
        for target in selected
    ]


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
        print(f"c3d: reuse durable {trajectory} stage", flush=True)
        return cached
    gates = manifest["binding_gates"]
    execution = manifest["coarse_completion_execution"]
    print(f"c3d: {trajectory} main", flush=True)
    main = c2._controller_segment(
        configuration,
        tangent,
        restart.primitive_charts,
        restart.history,
        c3c.RUNG_START_SECONDS,
        manifest["main_controller"]["initial_timestep_seconds"],
        field_scales,
        export_scales,
        c2.COUPLING_FACE,
        manifest["main_controller"],
        output_times=c3c.MAIN_TARGETS_SECONDS,
        stop_time=c3c.RUNG_HORIZON_SECONDS,
        checkpoint_times=(
            float(c3c.REPLAY_TARGETS_SECONDS[0]),
            float(c3c.STRICT_TARGETS_SECONDS[0]),
        ),
        include_initial_output=True,
    )
    main_report = c3b._segment_report(main, manifest["main_controller"])
    main_report["passed"] = bool(
        main_report["method_passed"]
        and main_report["accepted_comparisons"]
        == execution["main_expected_comparisons_per_trajectory"]
        and main_report["maximum_local_error_estimate"]
        <= gates["main_local_error_maximum"]
        and main_report["sum_local_error_estimates"]
        <= gates["main_local_error_sum_maximum"]
    )
    if not main_report["passed"]:
        raise RuntimeError(f"{trajectory} main completion failed")

    replay_start = float(c3c.REPLAY_TARGETS_SECONDS[0])
    strict_start = float(c3c.STRICT_TARGETS_SECONDS[0])
    replay_key = min(main["checkpoints"], key=lambda value: abs(value - replay_start))
    strict_key = min(main["checkpoints"], key=lambda value: abs(value - strict_start))
    replay_checkpoint, replay_roundtrip = c2._save_restore(
        configuration["context"],
        main["checkpoints"][replay_key]["restart"],
        f"c3d_{trajectory}_replay",
    )
    print(f"c3d: {trajectory} replay", flush=True)
    replay = c2._controller_segment(
        configuration,
        tangent,
        replay_checkpoint.primitive_charts,
        replay_checkpoint.history,
        replay_start,
        main["checkpoints"][replay_key]["next_timestep"],
        field_scales,
        export_scales,
        c2.COUPLING_FACE,
        manifest["main_controller"],
        output_times=c3c.REPLAY_TARGETS_SECONDS,
        stop_time=c3c.RUNG_HORIZON_SECONDS,
        include_initial_output=True,
    )
    replay_report = c3b._segment_report(replay, manifest["main_controller"])
    main_indices = _target_indices(
        c3c.MAIN_TARGET_MICROSECONDS, c3c.REPLAY_TARGET_MICROSECONDS
    )
    replay_bitwise = {
        "initial_restart_roundtrip_bitwise": replay_roundtrip,
        "target_labels_bitwise": np.array_equal(
            replay["output_times"], c3c.REPLAY_TARGETS_SECONDS
        ),
        "states_bitwise": np.array_equal(
            replay["output_states"], main["output_states"][main_indices]
        ),
        "Tier_I_exports_bitwise": np.array_equal(
            replay["output_exports"], main["output_exports"][main_indices]
        ),
        **_history_bitwise(main["final_history"], replay["final_history"]),
        "final_state_bitwise": np.array_equal(
            main["final_state"], replay["final_state"]
        ),
    }
    replay_report["passed"] = bool(
        replay_report["method_passed"]
        and replay_report["accepted_comparisons"]
        == execution["replay_expected_comparisons_per_trajectory"]
        and all(replay_bitwise.values())
    )
    if not replay_report["passed"]:
        raise RuntimeError(f"{trajectory} replay completion failed")

    strict_checkpoint, strict_roundtrip = c2._save_restore(
        configuration["context"],
        main["checkpoints"][strict_key]["restart"],
        f"c3d_{trajectory}_strict",
    )
    print(f"c3d: {trajectory} strict", flush=True)
    strict = c2._controller_segment(
        configuration,
        tangent,
        strict_checkpoint.primitive_charts,
        strict_checkpoint.history,
        strict_start,
        manifest["strict_controller"]["initial_timestep_seconds"],
        field_scales,
        export_scales,
        c2.COUPLING_FACE,
        manifest["strict_controller"],
        output_times=c3c.STRICT_TARGETS_SECONDS,
        stop_time=c3c.RUNG_HORIZON_SECONDS,
        include_initial_output=True,
    )
    strict_report = c3b._segment_report(strict, manifest["strict_controller"])
    strict_report["initial_restart_roundtrip_bitwise"] = strict_roundtrip
    strict_report["passed"] = bool(
        strict_report["method_passed"]
        and strict_report["accepted_comparisons"]
        == execution["strict_expected_comparisons_per_trajectory"]
        and strict_report["maximum_local_error_estimate"]
        <= gates["strict_local_error_maximum"]
        and strict_roundtrip
    )
    if not strict_report["passed"]:
        raise RuntimeError(f"{trajectory} strict completion failed")

    readiness = c3b1a._state_audit(configuration["context"], main["final_state"])
    final_restart = CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(main["final_state"], copy=True),
        history=main["final_history"],
        elapsed_time_seconds=c3c.RUNG_HORIZON_SECONDS,
        completed_steps=len(main["accepted_timesteps"]),
        next_order=2,
        provenance={
            "work_package": WORK_PACKAGE,
            "trajectory": trajectory,
            "source": "canonical_main_coarse_completion",
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


def _response_report(
    base: dict,
    perturbed: dict,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    manifest: dict,
) -> dict:
    main_state = perturbed["main"]["output_states"] - base["main"]["output_states"]
    main_export = (
        perturbed["main"]["output_exports"] - base["main"]["output_exports"]
    )
    strict_state = (
        perturbed["strict"]["output_states"] - base["strict"]["output_states"]
    )
    strict_export = (
        perturbed["strict"]["output_exports"] - base["strict"]["output_exports"]
    )
    main_indices = _target_indices(
        c3c.MAIN_TARGET_MICROSECONDS,
        np.asarray((c3c.STRICT_TARGET_MICROSECONDS[0], c3c.STRICT_TARGET_MICROSECONDS[-1])),
    )
    main_state = main_state[main_indices]
    main_export = main_export[main_indices]
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
    source = _load_npz(c3b.DECISIVE_ARRAYS)
    pilot = _load_npz(c3b2b.CANONICAL_DIRECTORY / "decisive_arrays.npz")
    field_scales = pilot["field_scales"]
    export_scales = pilot["fixed_physical_observable_scales"]
    configuration = c3b1a._configurations()[c2.LAYOUT]
    print("c3d: build tangent", flush=True)
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
        _restart_from_c3b(source, "base"),
        field_scales,
        export_scales,
        manifest,
    )
    perturbed = _trajectory_stage(
        "perturbed",
        configuration,
        tangent,
        _restart_from_c3b(source, "perturbed"),
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
    branch = manifest["positive_branch"] if passed else manifest["negative_branch"]
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
        "rung_start_seconds": c3c.RUNG_START_SECONDS,
        "rung_horizon_seconds": c3c.RUNG_HORIZON_SECONDS,
        "main_targets_seconds": c3c.MAIN_TARGETS_SECONDS,
        "replay_targets_seconds": c3c.REPLAY_TARGETS_SECONDS,
        "strict_targets_seconds": c3c.STRICT_TARGETS_SECONDS,
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
        "trajectory_reports": trajectory_reports,
        "strict_response_comparison": response,
        "elapsed_seconds": time.perf_counter() - started,
        "coarse_third_duration_rung_completion_certified": passed,
        "third_duration_rung_breadth_manifest_authorized": passed,
        "third_duration_rung_breadth_propagation_authorized": False,
        "third_duration_rung_spatial_confirmation_authorized": False,
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
                for path in (THIS_RUNNER, THIS_TEST, c3c.THIS_RUNNER, c3b.THIS_RUNNER)
                if (ROOT / path).exists()
            },
            "input_hashes": {
                "parent_summary": _sha256(PARENT_DIRECTORY / "summary.json"),
                "parent_manifest": _sha256(c3c.MANIFEST_PATH),
                "screen_restart_arrays": _sha256(c3b.DECISIVE_ARRAYS),
            },
        },
    )
    REPORT_PATH.write_text(
        "# Coarse third nonlinear duration-rung completion "
        "WP10c9d6c7c3b5c3d\n\n"
        "## Classification\n\n"
        f"`{summary['classification']}`\n\n"
        f"Base stage passes: `{base['passed']}`.\n\n"
        f"Perturbed stage passes: `{perturbed['passed']}`.\n\n"
        f"Strict response passes: `{response['passed']}`.\n\n"
        f"Authorized next: `{summary['authorized_next']}`.\n\n"
        "The third-rung breadth and spatial gates, fourth duration rung, "
        "fixed-Q experiments and reduced evolution remain blocked.\n",
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
