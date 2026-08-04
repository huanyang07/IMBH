#!/usr/bin/env python3
"""Complete the missing perturbed second-rung trajectory under corrected replay gates."""

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
import run_causal_inner_nonlinear_paired_base_replay_validation_wp10c9d6c7c3b5c2c1 as c2c1  # noqa: E402
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFRestart,
    causal_five_field_monolithic_bdf_history,
    causal_five_field_monolithic_frozen_tangent,
    causal_five_field_monolithic_storage_increment,
    save_causal_five_field_monolithic_bdf_restart,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c2d"
ANALYZED_BASE_COMMIT = "38f01107ede022526624a041137903d880e31e63"
ANALYZED_BASE_PARENT = "2577f42a2252bc1822d4c1e1de729767864ba31a"
ANALYZED_BASE_TREE = "732fc36833cb6825245119fa5ea08fc88b4a4ad9"
LAYOUT = c2.LAYOUT
PROFILE = c2.PROFILE
COUPLING_FACE = c2.COUPLING_FACE
REPLAY_TARGETS = c2c1.REPLAY_TARGETS

ARTIFACT = (
    "causal_inner_nonlinear_second_rung_perturbed_completion_"
    "wp10c9d6c7c3b5c2d"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_second_rung_perturbed_completion_"
    "wp10c9d6c7c3b5c2d.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_second_rung_perturbed_completion_"
    "wp10c9d6c7c3b5c2d.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_SECOND_RUNG_PERTURBED_COMPLETION_"
    "WP10C9D6C7C3B5C2D_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c2c1.CANONICAL_DIRECTORY
PROGRESS_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
PROGRESS_JSON = PROGRESS_DIRECTORY / "main_replay_progress.json"
PROGRESS_ARRAYS = PROGRESS_DIRECTORY / "main_replay_arrays.npz"
STRICT_RESTART = PROGRESS_DIRECTORY / "strict_restart.npz"


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


def _validate_parent() -> tuple[dict, dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    historical = _read_json(c2.CANONICAL_DIRECTORY / "summary.json")
    manifest = _read_json(c2.c3b5c2a.MANIFEST_PATH)
    if (
        not parent["passed"]
        or not parent["perturbed_second_rung_authorized"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c2d_second_rung_perturbed_completion"
        or historical["classification"]
        != "second_nonlinear_duration_rung_failed_later_duration_work_blocked"
        or historical["trajectory_reports"]["base"]["passed"]
    ):
        raise RuntimeError("c2d perturbed completion authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c2d analyzed identity changed")
    return parent, historical, manifest


def _history_arrays(checkpoints: dict, targets: np.ndarray) -> dict[str, np.ndarray]:
    restarts = [checkpoints[float(target)]["restart"] for target in targets]
    return {
        "primitive": np.asarray(
            [restart.history.previous_primitive_increment for restart in restarts]
        ),
        "mapped": np.asarray(
            [restart.history.previous_mapped_storage_increment for restart in restarts]
        ),
        "height": np.asarray(
            [
                restart.history.previous_responsive_height_storage_increment
                for restart in restarts
            ]
        ),
        "previous_timestep": np.asarray(
            [restart.history.previous_timestep_seconds for restart in restarts]
        ),
    }


def _checkpoint_key(checkpoints: dict, target: float) -> float:
    key = min(checkpoints, key=lambda value: abs(float(value) - target))
    if abs(float(key) - target) > 1.0e-15:
        raise RuntimeError(f"checkpoint near {target:.8e} is absent")
    return key


def _save_main_replay_progress(
    context,
    main: dict,
    replay: dict,
    states: np.ndarray,
    exports: np.ndarray,
    main_history: dict,
    replay_history: dict,
    separate_replay: dict,
    strict_key: float,
) -> None:
    PROGRESS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROGRESS_ARRAYS,
        states=states,
        exports=exports,
        main_output_times=main["output_times"],
        main_accepted_times=main["accepted_times"],
        main_accepted_timesteps=main["accepted_timesteps"],
        main_local_error_estimates=main["local_error_estimates"],
        replay_output_times=replay["output_times"],
        replay_output_states=replay["output_states"],
        replay_output_exports=replay["output_exports"],
        main_primitive_history=main_history["primitive"],
        replay_primitive_history=replay_history["primitive"],
        main_mapped_history=main_history["mapped"],
        replay_mapped_history=replay_history["mapped"],
        main_height_history=main_history["height"],
        replay_height_history=replay_history["height"],
    )
    save_causal_five_field_monolithic_bdf_restart(
        STRICT_RESTART,
        context,
        main["checkpoints"][strict_key]["restart"],
    )
    _write_json(
        PROGRESS_JSON,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            "main_replay_complete": True,
            "separate_replay_report": separate_replay,
            "strict_checkpoint_key": strict_key,
            "strict_next_timestep": main["checkpoints"][strict_key][
                "next_timestep"
            ],
        },
    )


def _trajectory(configuration, tangent, parent, field_scales, export_scales, manifest):
    started = time.perf_counter()
    context = configuration["context"]
    times = parent["perturbed__times_seconds"]
    states_parent = parent["perturbed__states"]
    exports_parent = parent["perturbed__direct_exports"]
    previous = np.array(states_parent[-2], copy=True)
    initial = np.array(states_parent[-1], copy=True)
    initial_export, initial_ledger, initial_incoming = c2.c3b5b._export_value(
        context, initial, COUPLING_FACE
    )
    initial_export_defect = float(
        np.max(np.abs((initial_export - exports_parent[-1]) / export_scales))
    )
    storage = causal_five_field_monolithic_storage_increment(
        context, previous, initial
    )
    history = causal_five_field_monolithic_bdf_history(
        initial - previous,
        storage,
        c2.CONTINUATION_START_SECONDS - c2.PREVIOUS_HISTORY_TIME_SECONDS,
    )
    initial_restart = CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=initial,
        history=history,
        elapsed_time_seconds=c2.CONTINUATION_START_SECONDS,
        completed_steps=12,
        next_order=2,
        provenance={
            "work_package": WORK_PACKAGE,
            "layout": LAYOUT,
            "profile": PROFILE,
            "trajectory_id": "perturbed",
        },
    )
    restored_initial, initial_roundtrip = c2._save_restore(
        context, initial_restart, "c2d_perturbed_start"
    )
    main_targets = tuple(float(value) for value in REPLAY_TARGETS)
    main = c2._controller_segment(
        configuration,
        tangent,
        restored_initial.primitive_charts,
        restored_initial.history,
        c2.CONTINUATION_START_SECONDS,
        manifest["main_controller"]["initial_timestep_seconds"],
        field_scales,
        export_scales,
        COUPLING_FACE,
        manifest["main_controller"],
        output_times=c2.CONTINUATION_OUTPUT_TIMES,
        stop_time=c2.HORIZON_SECONDS,
        checkpoint_times=main_targets,
        include_initial_output=True,
    )
    parent_indices = np.array([0, 5], dtype=int)
    states = np.concatenate((states_parent[parent_indices], main["output_states"]))
    exports = np.concatenate((exports_parent[parent_indices], main["output_exports"]))
    complete_times = np.array(c2.OUTPUT_TIMES, copy=True)
    restored = {}
    roundtrips = {}
    for checkpoint_time, payload in main["checkpoints"].items():
        restored[checkpoint_time], roundtrips[checkpoint_time] = c2._save_restore(
            context, payload["restart"], f"c2d_{checkpoint_time:.8e}"
        )
    replay_start = float(REPLAY_TARGETS[0])
    replay_checkpoint = main["checkpoints"][replay_start]
    replay = c2._controller_segment(
        configuration,
        tangent,
        restored[replay_start].primitive_charts,
        restored[replay_start].history,
        replay_start,
        replay_checkpoint["next_timestep"],
        field_scales,
        export_scales,
        COUPLING_FACE,
        manifest["main_controller"],
        output_times=c2.OUTPUT_TIMES,
        stop_time=c2.HORIZON_SECONDS,
        checkpoint_times=tuple(float(value) for value in REPLAY_TARGETS[1:]),
        include_initial_output=True,
    )
    replay_index = int(
        np.flatnonzero(
            np.isclose(c2.OUTPUT_TIMES, replay_start, rtol=0.0, atol=1.0e-18)
        )[0]
    )
    main_history = _history_arrays(main["checkpoints"], REPLAY_TARGETS[1:])
    replay_history = _history_arrays(replay["checkpoints"], REPLAY_TARGETS[1:])
    time_delta = replay["output_times"] - main["output_times"][4:]
    time_ulp = float(
        np.max(
            np.abs(time_delta)
            / np.maximum(
                np.spacing(np.abs(c2.OUTPUT_TIMES[replay_index:])),
                np.finfo(float).tiny,
            )
        )
    )
    separate_replay = {
        "canonical_time_labels_bitwise": np.array_equal(
            c2.OUTPUT_TIMES[replay_index:], c2.OUTPUT_TIMES[replay_index:]
        ),
        "accumulated_time_labels_bitwise": np.array_equal(
            replay["output_times"], main["output_times"][4:]
        ),
        "maximum_accumulated_time_spacing_units": time_ulp,
        "primitive_states_bitwise": np.array_equal(
            replay["output_states"], states[replay_index:]
        ),
        "direct_Tier_I_exports_bitwise": np.array_equal(
            replay["output_exports"], exports[replay_index:]
        ),
        "primitive_history_bitwise": np.array_equal(
            main_history["primitive"], replay_history["primitive"]
        ),
        "mapped_history_bitwise": np.array_equal(
            main_history["mapped"], replay_history["mapped"]
        ),
        "height_history_bitwise": np.array_equal(
            main_history["height"], replay_history["height"]
        ),
        "previous_timesteps_bitwise": np.array_equal(
            main_history["previous_timestep"], replay_history["previous_timestep"]
        ),
    }
    strict_start = c2.STRICT_SHADOW_START_SECONDS
    strict_key = _checkpoint_key(main["checkpoints"], strict_start)
    strict_checkpoint = main["checkpoints"][strict_key]
    _save_main_replay_progress(
        context,
        main,
        replay,
        states,
        exports,
        main_history,
        replay_history,
        separate_replay,
        strict_key,
    )
    strict_contract = manifest["strict_shadow"]["controller"]
    strict = c2._controller_segment(
        configuration,
        tangent,
        restored[strict_key].primitive_charts,
        restored[strict_key].history,
        strict_start,
        min(
            strict_checkpoint["next_timestep"],
            strict_contract["maximum_timestep_seconds"],
        ),
        field_scales,
        export_scales,
        COUPLING_FACE,
        strict_contract,
        output_times=c2.OUTPUT_TIMES,
        stop_time=c2.HORIZON_SECONDS,
        include_initial_output=True,
    )
    main_passed = c2._segment_passed(
        main,
        manifest["main_controller"],
        manifest["main_rung_error_budget"]["maximum_sum_of_accepted_error_estimates"],
    )
    replay_passed = c2._segment_passed(
        replay,
        manifest["main_controller"],
        manifest["main_rung_error_budget"]["maximum_sum_of_accepted_error_estimates"],
    )
    strict_passed = c2._segment_passed(
        strict,
        strict_contract,
        strict_contract["error_estimator"]["rung_sum_of_accepted_error_estimates"],
    )
    bitwise_fields = (
        separate_replay["primitive_states_bitwise"]
        and separate_replay["direct_Tier_I_exports_bitwise"]
        and separate_replay["primitive_history_bitwise"]
        and separate_replay["mapped_history_bitwise"]
        and separate_replay["height_history_bitwise"]
        and separate_replay["previous_timesteps_bitwise"]
    )
    all_records = main["step_records"] + replay["step_records"] + strict["step_records"]
    readiness = c3b1a._state_audit(context, main["final_state"])
    passed = bool(
        main_passed
        and replay_passed
        and strict_passed
        and initial_roundtrip
        and all(roundtrips.values())
        and bitwise_fields
        and time_ulp <= 1.0
        and initial_export_defect <= 1.0e-12
        and initial_ledger <= 1.0e-9
        and initial_incoming == 0
    )
    report = {
        "passed": passed,
        "continued_from_committed_BDF2_history": True,
        "continuation_history_roundtrip_bitwise": initial_roundtrip,
        "checkpoint_roundtrips_bitwise": all(roundtrips.values()),
        "separate_replay_report": separate_replay,
        "accepted_main_BDF2_steps": int(main["accepted_timesteps"].size),
        "accepted_replay_BDF2_steps": int(replay["accepted_timesteps"].size),
        "accepted_strict_shadow_BDF2_steps": int(strict["accepted_timesteps"].size),
        "main_rejected_attempts": int(np.sum(main["retries"])),
        "strict_rejected_attempts": int(np.sum(strict["retries"])),
        "maximum_main_local_error_estimate": float(np.max(main["local_error_estimates"])),
        "sum_main_local_error_estimates": float(np.sum(main["local_error_estimates"])),
        "maximum_strict_local_error_estimate": float(
            np.max(strict["local_error_estimates"])
        ),
        "sum_strict_local_error_estimates": float(np.sum(strict["local_error_estimates"])),
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
        "maximum_export_ledger_defect": max(
            main["maximum_export_ledger_defect"],
            replay["maximum_export_ledger_defect"],
            strict["maximum_export_ledger_defect"],
        ),
        "continuation_export_reconstruction_defect": initial_export_defect,
        "continuation_storage_mapped_closure_defect": float(
            storage.maximum_mapped_path_closure_defect
        ),
        "final_state_audit": readiness,
        "elapsed_seconds": time.perf_counter() - started,
    }
    arrays = {
        "times_seconds": complete_times,
        "states": states,
        "direct_exports": exports,
        "main_accepted_times_seconds": main["accepted_times"],
        "main_accepted_timesteps_seconds": main["accepted_timesteps"],
        "main_local_error_estimates": main["local_error_estimates"],
        "strict_times_seconds": np.array(c2.OUTPUT_TIMES[-3:], copy=True),
        "strict_states": strict["output_states"],
        "strict_direct_exports": strict["output_exports"],
        "strict_accepted_timesteps_seconds": strict["accepted_timesteps"],
        "strict_local_error_estimates": strict["local_error_estimates"],
        "replay_accumulated_times_seconds": replay["output_times"],
        "main_replay_accumulated_times_seconds": main["output_times"][4:],
        "main_replay_primitive_history": main_history["primitive"],
        "replay_primitive_history": replay_history["primitive"],
        "main_replay_mapped_history": main_history["mapped"],
        "replay_mapped_history": replay_history["mapped"],
        "main_replay_height_history": main_history["height"],
        "replay_height_history": replay_history["height"],
    }
    return report, arrays


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
    parent, historical, manifest = _validate_parent()
    first = _load_npz(c2.FIRST_RUNG_DIRECTORY / "decisive_arrays.npz")
    historical_arrays = _load_npz(c2.CANONICAL_DIRECTORY / "decisive_arrays.npz")
    pilot = _load_npz(c3b2b.CANONICAL_DIRECTORY / "decisive_arrays.npz")
    field_scales = pilot["field_scales"]
    export_scales = pilot["fixed_physical_observable_scales"]
    configuration = c3b1a._configurations()[LAYOUT]
    print("c2d: build tangent", flush=True)
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    report, perturbed = _trajectory(
        configuration, tangent, first, field_scales, export_scales, manifest
    )
    arrays = {key: value for key, value in historical_arrays.items() if key.startswith("base__")}
    arrays.update({f"perturbed__{key}": value for key, value in perturbed.items()})
    shadow, shadow_arrays = c2._shadow_comparison(
        arrays, field_scales, export_scales, manifest
    )
    arrays.update(shadow_arrays)
    passed = bool(report["passed"] and shadow["passed"] and parent["passed"])
    classification = (
        "second_nonlinear_duration_rung_response_certified_"
        "third_rung_manifest_authorized"
        if passed
        else "second_rung_perturbed_completion_failed_later_duration_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b5c3a_third_duration_rung_manifest" if passed else "none"
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": LAYOUT,
        "profile": PROFILE,
        "coupling_face": COUPLING_FACE,
        "horizon_seconds": c2.HORIZON_SECONDS,
        "canonical_output_times_seconds": c2.OUTPUT_TIMES,
        "reuse_committed_base_by_hash": True,
        "run_only_missing_perturbed_trajectory": True,
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
        "historical_c2_classification_preserved": historical["classification"],
        "operator_changed": False,
        "production_defaults_changed": False,
        "committed_base_reused_by_hash": True,
        "perturbed_trajectory_report": report,
        "strict_shadow_comparison": shadow,
        "third_duration_rung_manifest_authorized": passed,
        "third_duration_rung_propagation_authorized": False,
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
                for path in (THIS_RUNNER, THIS_TEST, c2c1.THIS_RUNNER, c2.THIS_RUNNER)
                if (ROOT / path).exists()
            },
            "input_hashes": {
                "parent_summary": _sha256(PARENT_DIRECTORY / "summary.json"),
                "historical_base_arrays": _sha256(
                    c2.CANONICAL_DIRECTORY / "decisive_arrays.npz"
                ),
                "first_rung_arrays": _sha256(
                    c2.FIRST_RUNG_DIRECTORY / "decisive_arrays.npz"
                ),
                "second_rung_manifest": _sha256(c2.c3b5c2a.MANIFEST_PATH),
            },
        },
    )
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Second-rung perturbed completion WP10c9d6c7c3b5c2d",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Perturbed method gates pass: `{report['passed']}`.",
                f"Strict-shadow response gates pass: `{shadow['passed']}`.",
                "The historical c2 failure remains unchanged.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
                "Fixed-Q experiments and reduced evolution remain blocked.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    names = ("config.json", "summary.json", "provenance.json", "decisive_arrays.npz")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names)
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
