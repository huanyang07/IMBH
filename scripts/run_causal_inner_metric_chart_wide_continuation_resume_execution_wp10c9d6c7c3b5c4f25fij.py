#!/usr/bin/env python3
"""Resume the wide autonomous path with adaptive conservative metric patches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_metric_chart_wide_continuation_resume_manifest_wp10c9d6c7c3b5c4f25fii as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.conservative_metric_chart_atlas import ConservativeMetricChart  # noqa: E402


source = manifest.parent
execution = source.execution
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fij"
CYCLE_CLASSIFICATION = "metric_wide_resume_cycle_observed"
EQUILIBRIUM_CLASSIFICATION = "metric_wide_resume_equilibrium_candidate"
BUDGET_CLASSIFICATION = "metric_wide_resume_clean_budget_exhausted"
PHYSICAL_FAILURE_CLASSIFICATION = "metric_wide_resume_original_physical_gate_failed"
NUMERICAL_FAILURE_CLASSIFICATION = "metric_wide_resume_numerical_or_restart_failed"
AUTHORIZED_CYCLE_NEXT = (
    "WP10c9d6c7c3b5c4f25fik_metric_matched_path_global_cycle_manifest"
)
AUTHORIZED_EQUILIBRIUM_NEXT = (
    "WP10c9d6c7c3b5c4f25fik_metric_equilibrium_stability_manifest"
)
AUTHORIZED_BUDGET_NEXT = (
    "WP10c9d6c7c3b5c4f25fik_metric_wide_continuation_next_tranche_manifest"
)
ARTIFACT = (
    "causal_inner_metric_chart_wide_continuation_resume_execution_"
    "wp10c9d6c7c3b5c4f25fij"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_METRIC_CHART_WIDE_CONTINUATION_"
    "RESUME_EXECUTION_WP10C9D6C7C3B5C4F25FIJ_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_metric_chart_wide_continuation_resume_execution_"
    "wp10c9d6c7c3b5c4f25fij.py"
)
THIS_TEST = (
    "tests/test_causal_inner_metric_chart_wide_continuation_resume_execution_"
    "wp10c9d6c7c3b5c4f25fij.py"
)


def _helper():
    return manifest._helper()


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(manifest.CANONICAL_DIRECTORY / "resume_contract.json")
    cost = helper._read(manifest.CANONICAL_DIRECTORY / "cost_projection.json")
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["metric_chart_wide_resume_authorized"]
        or summary["metric_chart_wide_resume_executed"]
        or summary["new_trajectory"]
        or summary["authorized_next"] != manifest.AUTHORIZED_NEXT
        or contract["authorized_execution"] != manifest.AUTHORIZED_NEXT
        or contract["scope"]["maximum_accepted_segments"] != 64
        or contract["scope"]["maximum_exact_free_field_calls"] != 88
        or contract["adaptive_policy"]["maximum_segment_seconds"] != 2.0e-3
        or not contract["adaptive_policy"]["failed_candidate_is_never_propagated"]
        or not cost["cost_gate_passed"]
        or cost["reserved_projected_wall_hours"]
        > manifest.MAXIMUM_EXECUTION_WALL_HOURS
    ):
        raise RuntimeError("wide-resume authorization changed")
    for relative, frozen_hash in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != frozen_hash:
            raise RuntimeError(f"frozen wide-resume source changed: {relative}")
    parent_lock = manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("wide-resume execution requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "cost": cost,
        "parent_lock": parent_lock,
    }


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        source.THIS_RUNNER,
        source.parent.ATLAS_SOURCE,
        execution.source.THIS_RUNNER,
        "src/imri_qpe/layer3_minidisk_1d/conservative_free_field_rom.py",
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _prepare_scratch(lock: dict) -> dict:
    helper = _helper()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": _source_hashes(),
        "manifest_hashes": lock["hashes"],
    }
    path = SCRATCH_DIRECTORY / "execution_identity.json"
    if SCRATCH_DIRECTORY.exists():
        if not path.exists() or helper._read(path) != identity:
            raise RuntimeError("wide-resume scratch identity mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(path, identity)
    return identity


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "resume_seed.npz")


def _initial_progress() -> dict:
    seed = _seed()
    current = np.asarray(seed["current_coordinate470"])
    start = np.asarray(seed["start_coordinate470"])
    normal = np.asarray(seed["section_normal470"])
    return {
        "previous_coordinate": np.asarray(seed["previous_coordinate470"]).copy(),
        "current_coordinate": current.copy(),
        "previous_state": np.asarray(seed["previous_primitive_state"]).copy(),
        "current_state": np.asarray(seed["current_primitive_state"]).copy(),
        "previous_rate": np.asarray(seed["previous_coordinate_rate470_per_s"]).copy(),
        "current_rate": np.asarray(seed["current_coordinate_rate470_per_s"]).copy(),
        "previous_span": float(seed["previous_span_seconds"]),
        "next_span": float(seed["next_span_seconds"]),
        "elapsed_seconds": float(seed["elapsed_seconds"]),
        "accepted_segments_total": int(seed["accepted_segments_total"]),
        "accepted_segments_new": 0,
        "attempts": 0,
        "accepted_since_growth": int(seed["accepted_since_growth"]),
        "metric_transform": np.asarray(seed["current_metric_transform470x470"]).copy(),
        "metric_augmented": np.asarray(seed["current_metric_augmented560x560"]).copy(),
        "gauge_basis": np.asarray(seed["current_gauge_basis560x90"]).copy(),
        "section_normal": normal.copy(),
        "start_coordinate": start.copy(),
        "previous_section": float(normal @ (current - start)),
        "seen_negative": bool(seed["seen_negative_section"]),
        "initial_speed": float(seed["initial_resume_speed_per_second"]),
        "cycle_event": None,
        "nonclosing_events": [],
        "stop_classification": None,
    }


def _attempt_directory(index: int) -> Path:
    return SCRATCH_DIRECTORY / f"attempt_{index:04d}"


def _pair(directory: Path, stem: str) -> tuple[Path, Path]:
    return directory / f"{stem}.json", directory / f"{stem}.npz"


def _load_pair(
    directory: Path, stem: str
) -> tuple[dict, dict[str, np.ndarray]] | None:
    metrics_path, arrays_path = _pair(directory, stem)
    if metrics_path.exists() != arrays_path.exists():
        raise RuntimeError(f"partial wide-resume pair: {directory.name}/{stem}")
    if not metrics_path.exists():
        return None
    return _helper()._read(metrics_path), _load_npz(arrays_path)


def _write_pair(
    directory: Path,
    stem: str,
    metrics: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    metrics_path, arrays_path = _pair(directory, stem)
    _helper()._write_json(metrics_path, metrics)
    _save_npz(arrays_path, arrays)


def _inventory() -> dict:
    attempts = sorted(SCRATCH_DIRECTORY.glob("attempt_[0-9][0-9][0-9][0-9]"))
    completed = []
    for directory in attempts:
        pair = _pair(directory, "attempt")
        if pair[0].exists() != pair[1].exists():
            raise RuntimeError(f"partial completed attempt: {directory.name}")
        if pair[0].exists():
            completed.append((directory, pair[0], pair[1]))
    return {
        "attempt_directories": attempts,
        "completed_attempts": completed,
        "exact_field_paths": sorted(SCRATCH_DIRECTORY.glob("**/*_field.json")),
    }


def _checkpoint_arrays(progress: dict) -> dict[str, np.ndarray]:
    return {
        "previous_coordinate470": np.asarray(progress["previous_coordinate"]),
        "current_coordinate470": np.asarray(progress["current_coordinate"]),
        "previous_primitive_state": np.asarray(progress["previous_state"]),
        "current_primitive_state": np.asarray(progress["current_state"]),
        "previous_coordinate_rate470_per_s": np.asarray(progress["previous_rate"]),
        "current_coordinate_rate470_per_s": np.asarray(progress["current_rate"]),
        "previous_span_seconds": np.asarray(progress["previous_span"]),
        "next_span_seconds": np.asarray(progress["next_span"]),
        "elapsed_seconds": np.asarray(progress["elapsed_seconds"]),
        "accepted_segments_total": np.asarray(progress["accepted_segments_total"]),
        "accepted_segments_new": np.asarray(progress["accepted_segments_new"]),
        "attempts": np.asarray(progress["attempts"]),
        "accepted_since_growth": np.asarray(progress["accepted_since_growth"]),
        "current_metric_transform470x470": np.asarray(progress["metric_transform"]),
        "current_metric_augmented560x560": np.asarray(progress["metric_augmented"]),
        "current_gauge_basis560x90": np.asarray(progress["gauge_basis"]),
        "section_normal470": np.asarray(progress["section_normal"]),
        "start_coordinate470": np.asarray(progress["start_coordinate"]),
        "previous_section": np.asarray(progress["previous_section"]),
        "seen_negative": np.asarray(progress["seen_negative"]),
        "initial_speed_per_second": np.asarray(progress["initial_speed"]),
    }


def _progress_from_checkpoint(
    arrays: dict[str, np.ndarray], metadata: dict
) -> dict:
    return {
        "previous_coordinate": arrays["previous_coordinate470"].copy(),
        "current_coordinate": arrays["current_coordinate470"].copy(),
        "previous_state": arrays["previous_primitive_state"].copy(),
        "current_state": arrays["current_primitive_state"].copy(),
        "previous_rate": arrays["previous_coordinate_rate470_per_s"].copy(),
        "current_rate": arrays["current_coordinate_rate470_per_s"].copy(),
        "previous_span": float(arrays["previous_span_seconds"]),
        "next_span": float(arrays["next_span_seconds"]),
        "elapsed_seconds": float(arrays["elapsed_seconds"]),
        "accepted_segments_total": int(arrays["accepted_segments_total"]),
        "accepted_segments_new": int(arrays["accepted_segments_new"]),
        "attempts": int(arrays["attempts"]),
        "accepted_since_growth": int(arrays["accepted_since_growth"]),
        "metric_transform": arrays["current_metric_transform470x470"].copy(),
        "metric_augmented": arrays["current_metric_augmented560x560"].copy(),
        "gauge_basis": arrays["current_gauge_basis560x90"].copy(),
        "section_normal": arrays["section_normal470"].copy(),
        "start_coordinate": arrays["start_coordinate470"].copy(),
        "previous_section": float(arrays["previous_section"]),
        "seen_negative": bool(arrays["seen_negative"]),
        "initial_speed": float(arrays["initial_speed_per_second"]),
        "cycle_event": metadata.get("cycle_event"),
        "nonclosing_events": list(metadata.get("nonclosing_events", [])),
        "stop_classification": metadata.get("stop_classification"),
    }


def _write_checkpoint(directory: Path, progress: dict) -> bool:
    arrays_path = directory / "accepted_checkpoint.npz"
    metadata_path = directory / "accepted_checkpoint.json"
    arrays = _checkpoint_arrays(progress)
    metadata = {
        "cycle_event": progress["cycle_event"],
        "nonclosing_events": progress["nonclosing_events"],
        "stop_classification": progress["stop_classification"],
    }
    if arrays_path.exists() != metadata_path.exists():
        raise RuntimeError("partial accepted wide-resume checkpoint")
    if not arrays_path.exists():
        _save_npz(arrays_path, arrays)
        _helper()._write_json(metadata_path, metadata)
    loaded = _load_npz(arrays_path)
    return bool(
        set(loaded) == set(arrays)
        and all(np.array_equal(loaded[name], arrays[name]) for name in arrays)
        and _helper()._read(metadata_path) == metadata
    )


def _apply_record(
    progress: dict, metrics: dict, arrays: dict[str, np.ndarray]
) -> dict:
    result = {
        **progress,
        "attempts": int(progress["attempts"]) + 1,
        "next_span": float(metrics["next_span_seconds"]),
        "accepted_since_growth": int(metrics["accepted_since_growth_after"]),
        "seen_negative": bool(metrics["seen_negative_after"]),
        "previous_section": float(metrics["section_after"]),
        "stop_classification": metrics["stop_classification"],
    }
    if metrics.get("nonclosing_event") is not None:
        result["nonclosing_events"] = [
            *progress["nonclosing_events"],
            metrics["nonclosing_event"],
        ]
    if metrics.get("cycle_event") is not None:
        result["cycle_event"] = metrics["cycle_event"]
    if not metrics["accepted"]:
        return result
    result.update({
        "previous_coordinate": np.asarray(progress["current_coordinate"]),
        "current_coordinate": np.asarray(arrays["accepted_coordinate470"]),
        "previous_state": np.asarray(progress["current_state"]),
        "current_state": np.asarray(arrays["accepted_primitive_state"]),
        "previous_rate": np.asarray(progress["current_rate"]),
        "current_rate": np.asarray(arrays["accepted_coordinate_rate470_per_s"]),
        "previous_span": float(metrics["span_seconds"]),
        "elapsed_seconds": float(metrics["elapsed_seconds_after"]),
        "accepted_segments_total": int(progress["accepted_segments_total"]) + 1,
        "accepted_segments_new": int(progress["accepted_segments_new"]) + 1,
        "metric_transform": np.asarray(
            arrays["accepted_metric_transform470x470"]
        ),
        "metric_augmented": np.asarray(
            arrays["accepted_metric_augmented560x560"]
        ),
        "gauge_basis": np.asarray(arrays["accepted_gauge_basis560x90"]),
    })
    return result


def _restore_progress() -> tuple[dict, list[tuple[dict, dict[str, np.ndarray]]]]:
    progress = _initial_progress()
    records = []
    for expected, (directory, metrics_path, arrays_path) in enumerate(
        _inventory()["completed_attempts"]
    ):
        if int(directory.name.split("_")[-1]) != expected:
            raise RuntimeError("wide-resume attempt scratch is not contiguous")
        metrics = _helper()._read(metrics_path)
        arrays = _load_npz(arrays_path)
        candidate = execution._variable_step_ab2(
            progress["current_coordinate"],
            progress["current_rate"],
            progress["previous_rate"],
            float(progress["next_span"]),
            float(progress["previous_span"]),
        )
        np.testing.assert_array_equal(candidate, arrays["candidate_target470"])
        progress = _apply_record(progress, metrics, arrays)
        if metrics["accepted"] and not _write_checkpoint(directory, progress):
            raise RuntimeError("restored accepted checkpoint failed roundtrip")
        records.append((metrics, arrays))
    return progress, records


_RUN_STARTED = 0.0
_PRIOR_WALL = 0.0


def _current_total_wall() -> float:
    return float(_PRIOR_WALL + max(0.0, time.perf_counter() - _RUN_STARTED))


def _remaining_capacity(required_calls: int, lock: dict) -> bool:
    used = len(_inventory()["exact_field_paths"])
    call_ok = used + int(required_calls) + 1 <= manifest.MAXIMUM_EXACT_FREE_FIELD_CALLS
    projected = (
        _current_total_wall()
        + (required_calls + 1)
        * float(lock["cost"]["observed_suffix_wall_seconds_per_exact_call"])
        * manifest.COST_RESERVE_FACTOR
    )
    wall_ok = projected <= 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
    return bool(call_ok and wall_ok)


def _transition_after_attempt(
    *,
    span: float,
    accepted: bool,
    blind_required: bool,
    physical_failure: bool,
    chart_failure: bool,
    accepted_since_growth: int,
) -> dict:
    if physical_failure:
        return {
            "next_span": span,
            "accepted_since_growth": 0,
            "stop_classification": PHYSICAL_FAILURE_CLASSIFICATION,
        }
    if chart_failure:
        return {
            "next_span": span,
            "accepted_since_growth": 0,
            "stop_classification": NUMERICAL_FAILURE_CLASSIFICATION,
        }
    if accepted:
        count = accepted_since_growth + 1
        next_span = span
        if (
            blind_required
            and count >= manifest.ACCEPTED_SEGMENTS_BEFORE_GROWTH
            and span < manifest.MAXIMUM_SEGMENT_SECONDS
        ):
            next_span = min(
                manifest.GROWTH_FACTOR_MAXIMUM * span,
                manifest.MAXIMUM_SEGMENT_SECONDS,
            )
            count = 0
        return {
            "next_span": next_span,
            "accepted_since_growth": count,
            "stop_classification": None,
        }
    if span > manifest.MINIMUM_SEGMENT_SECONDS:
        return {
            "next_span": max(0.5 * span, manifest.MINIMUM_SEGMENT_SECONDS),
            "accepted_since_growth": 0,
            "stop_classification": None,
        }
    return {
        "next_span": span,
        "accepted_since_growth": 0,
        "stop_classification": NUMERICAL_FAILURE_CLASSIFICATION,
    }


def _retraction_physical_passed(metrics: dict) -> bool:
    return bool(
        metrics["minimum_reconstruction_factor"] >= 1.0 - 1.0e-12
        and metrics["maximum_height_ratio"] <= 0.5
        and metrics["minimum_scattering_optical_depth"] >= 1.0
    )


def _retraction_chart_failed(metrics: dict) -> bool:
    return bool(
        metrics["maximum_metric_augmented_condition_number"]
        > manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
    )


def _trajectory_so_far(
    seed: dict[str, np.ndarray],
    records: list[tuple[dict, dict[str, np.ndarray]]],
) -> np.ndarray:
    accepted = [arrays["accepted_coordinate470"] for metrics, arrays in records if metrics["accepted"]]
    if not accepted:
        return np.asarray(seed["trajectory_coordinates"])
    return np.vstack((seed["trajectory_coordinates"], np.stack(accepted)))


def _hidden_return_defect(
    inputs: dict,
    path: np.ndarray,
    event_coordinate: np.ndarray,
) -> float:
    full = np.vstack((path, np.asarray(event_coordinate)[None, :]))
    hidden = np.stack([inputs["split"].split(value)[1] for value in full])
    length = float(np.sum(np.linalg.norm(np.diff(hidden, axis=0), axis=1)))
    return float(
        np.linalg.norm(hidden[-1] - hidden[0])
        / max(length, np.finfo(float).tiny)
    )


def _attempt_segment(
    *,
    lock: dict,
    inputs: dict,
    exact_chart,
    seed: dict[str, np.ndarray],
    progress: dict,
    records: list[tuple[dict, dict[str, np.ndarray]]],
) -> tuple[dict, dict[str, np.ndarray]]:
    index = int(progress["attempts"])
    directory = _attempt_directory(index)
    directory.mkdir(exist_ok=True)
    completed = _load_pair(directory, "attempt")
    span = float(progress["next_span"])
    tentative = int(progress["accepted_segments_total"]) + 1
    blind_required = tentative % manifest.BLIND_MIDPOINT_FREQUENCY == 0
    candidate = execution._variable_step_ab2(
        progress["current_coordinate"],
        progress["current_rate"],
        progress["previous_rate"],
        span,
        progress["previous_span"],
    )
    if completed is not None:
        np.testing.assert_array_equal(completed[1]["candidate_target470"], candidate)
        print(f"{directory.name}: reused completed attempt", flush=True)
        return completed
    required = 1 + int(blind_required)
    if not _remaining_capacity(required, lock):
        raise RuntimeError("acquisition_budget")
    anchor_chart = ConservativeMetricChart(
        progress["current_coordinate"],
        progress["metric_transform"],
        source._block_sizes(),
    )
    retraction, retraction_arrays = source._metric_retraction(
        directory=directory,
        stem="endpoint_retraction",
        exact_chart=exact_chart,
        model=inputs["model"],
        progress=progress,
        target=candidate,
    )
    endpoint_coordinate = np.asarray(
        retraction_arrays["recovered_original_coordinate470"]
    )
    endpoint_state = np.asarray(retraction_arrays["primitive_state"])
    physical_failure = not _retraction_physical_passed(retraction)
    chart_failure = _retraction_chart_failed(retraction)
    retraction_closed = bool(retraction["passed"])
    endpoint_field = None
    endpoint_field_arrays = None
    endpoint_rate = np.full(470, np.nan)
    endpoint_defect = float("inf")
    if retraction_closed and not chart_failure and not physical_failure:
        endpoint_field, endpoint_field_arrays = source._metric_field(
            directory=directory,
            stem="endpoint_field",
            inputs=inputs,
            exact_chart=exact_chart,
            state=endpoint_state,
            coordinate=endpoint_coordinate,
            retraction=retraction,
            anchor_chart=anchor_chart,
        )
        endpoint_rate = np.asarray(
            endpoint_field_arrays["coordinate_free_rate470_per_s"]
        )
        endpoint_defect = execution._endpoint_integral_defect(
            progress["current_coordinate"],
            progress["current_rate"],
            endpoint_coordinate,
            endpoint_rate,
            span,
        )
    physical_failure = bool(
        physical_failure
        or (endpoint_field is not None and not endpoint_field["physical_passed"])
    )
    numerical_passed = bool(
        retraction_closed
        and not chart_failure
        and not physical_failure
        and endpoint_defect <= manifest.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
    )

    midpoint_target = np.full(470, np.nan)
    midpoint_hermite_rate = np.full(470, np.nan)
    midpoint_coordinate = np.full(470, np.nan)
    midpoint_state = np.full((112, 5), np.nan)
    midpoint_rate = np.full(470, np.nan)
    midpoint_retraction = None
    midpoint_field = None
    blind_defect = None
    if numerical_passed and blind_required:
        midpoint_target, midpoint_hermite_rate = execution._hermite(
            progress["current_coordinate"],
            progress["current_rate"],
            endpoint_coordinate,
            endpoint_rate,
            span,
            0.5,
        )
        midpoint_retraction, midpoint_retraction_arrays = source._metric_retraction(
            directory=directory,
            stem="midpoint_retraction",
            exact_chart=exact_chart,
            model=inputs["model"],
            progress=progress,
            target=midpoint_target,
        )
        midpoint_coordinate = np.asarray(
            midpoint_retraction_arrays["recovered_original_coordinate470"]
        )
        midpoint_state = np.asarray(midpoint_retraction_arrays["primitive_state"])
        if not _retraction_physical_passed(midpoint_retraction):
            physical_failure = True
            numerical_passed = False
        elif _retraction_chart_failed(midpoint_retraction):
            chart_failure = True
            numerical_passed = False
        elif not midpoint_retraction["passed"]:
            numerical_passed = False
        else:
            midpoint_field, midpoint_field_arrays = source._metric_field(
                directory=directory,
                stem="midpoint_field",
                inputs=inputs,
                exact_chart=exact_chart,
                state=midpoint_state,
                coordinate=midpoint_coordinate,
                retraction=midpoint_retraction,
                anchor_chart=anchor_chart,
            )
            midpoint_rate = np.asarray(
                midpoint_field_arrays["coordinate_free_rate470_per_s"]
            )
            blind_defect = source._relative(midpoint_hermite_rate, midpoint_rate)
            if not midpoint_field["physical_passed"]:
                physical_failure = True
                numerical_passed = False
            else:
                numerical_passed = bool(
                    numerical_passed
                    and blind_defect <= manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
                )

    section_after = float(progress["previous_section"])
    seen_negative_after = bool(progress["seen_negative"])
    event_fraction = None
    event_target = np.full(470, np.nan)
    event_hermite_rate = np.full(470, np.nan)
    event_coordinate = np.full(470, np.nan)
    event_state = np.full((112, 5), np.nan)
    event_rate = np.full(470, np.nan)
    event_record = None
    nonclosing_event = None
    cycle_event = None
    if numerical_passed:
        section_after = float(
            progress["section_normal"]
            @ (endpoint_coordinate - progress["start_coordinate"])
        )
        if section_after < 0.0:
            seen_negative_after = True
        if seen_negative_after:
            event_fraction = execution._section_root_fraction(
                progress["current_coordinate"],
                progress["current_rate"],
                endpoint_coordinate,
                endpoint_rate,
                span,
                progress["start_coordinate"],
                progress["section_normal"],
            )
        if event_fraction is not None:
            if not _remaining_capacity(1, lock):
                raise RuntimeError("acquisition_budget")
            event_target, event_hermite_rate = execution._hermite(
                progress["current_coordinate"],
                progress["current_rate"],
                endpoint_coordinate,
                endpoint_rate,
                span,
                event_fraction,
            )
            event_retraction, event_retraction_arrays = source._metric_retraction(
                directory=directory,
                stem="event_retraction",
                exact_chart=exact_chart,
                model=inputs["model"],
                progress=progress,
                target=event_target,
            )
            event_coordinate = np.asarray(
                event_retraction_arrays["recovered_original_coordinate470"]
            )
            event_state = np.asarray(event_retraction_arrays["primitive_state"])
            if not _retraction_physical_passed(event_retraction):
                physical_failure = True
                numerical_passed = False
            elif _retraction_chart_failed(event_retraction):
                chart_failure = True
                numerical_passed = False
            elif not event_retraction["passed"]:
                numerical_passed = False
            else:
                event_field, event_field_arrays = source._metric_field(
                    directory=directory,
                    stem="event_field",
                    inputs=inputs,
                    exact_chart=exact_chart,
                    state=event_state,
                    coordinate=event_coordinate,
                    retraction=event_retraction,
                    anchor_chart=anchor_chart,
                )
                event_rate = np.asarray(
                    event_field_arrays["coordinate_free_rate470_per_s"]
                )
                if not event_field["physical_passed"]:
                    physical_failure = True
                    numerical_passed = False
                else:
                    orientation = float(event_rate @ progress["section_normal"])
                    rate_defect = source._relative(event_hermite_rate, event_rate)
                    hidden_return = _hidden_return_defect(
                        inputs,
                        _trajectory_so_far(seed, records),
                        event_coordinate,
                    )
                    event_record = {
                        "fraction": event_fraction,
                        "elapsed_seconds": float(progress["elapsed_seconds"] + event_fraction * span),
                        "section_value": float(
                            progress["section_normal"]
                            @ (event_coordinate - progress["start_coordinate"])
                        ),
                        "orientation_per_second": orientation,
                        "hermite_to_exact_rate_defect": rate_defect,
                        "hidden_return_defect": hidden_return,
                        "physical_passed": True,
                    }
                    valid = bool(
                        orientation > 0.0
                        and rate_defect <= manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
                    )
                    if not valid:
                        numerical_passed = False
                    elif hidden_return <= manifest.CYCLE_HIDDEN_RETURN_DEFECT_MAXIMUM:
                        cycle_event = event_record
                    else:
                        nonclosing_event = event_record
                        seen_negative_after = False

    accepted = bool(numerical_passed and not physical_failure and not chart_failure)
    transition = _transition_after_attempt(
        span=span,
        accepted=accepted,
        blind_required=blind_required,
        physical_failure=physical_failure,
        chart_failure=chart_failure,
        accepted_since_growth=int(progress["accepted_since_growth"]),
    )
    if not accepted:
        section_after = float(progress["previous_section"])
        seen_negative_after = bool(progress["seen_negative"])
        nonclosing_event = None
        cycle_event = None
        event_record = None
    elapsed_after = (
        float(progress["elapsed_seconds"] + span)
        if accepted
        else float(progress["elapsed_seconds"])
    )
    metrics = {
        "attempt_index": index,
        "tentative_segment_number": tentative,
        "span_seconds": span,
        "previous_span_seconds": float(progress["previous_span"]),
        "blind_midpoint_required": blind_required,
        "endpoint_integral_defect": endpoint_defect,
        "blind_midpoint_rate_defect": blind_defect,
        "endpoint_physical_passed": (
            False if endpoint_field is None else endpoint_field["physical_passed"]
        ),
        "midpoint_physical_passed": (
            None if midpoint_field is None else midpoint_field["physical_passed"]
        ),
        "physical_failure": physical_failure,
        "chart_failure": chart_failure,
        "numerical_passed": numerical_passed,
        "accepted": accepted,
        "section_before": float(progress["previous_section"]),
        "section_after": section_after,
        "seen_negative_after": seen_negative_after,
        "accepted_since_growth_after": transition["accepted_since_growth"],
        "next_span_seconds": transition["next_span"],
        "elapsed_seconds_after": elapsed_after,
        "event_fraction": event_fraction,
        "nonclosing_event": nonclosing_event,
        "cycle_event": cycle_event,
        "stop_classification": transition["stop_classification"],
        "endpoint_field": endpoint_field,
        "midpoint_field": midpoint_field,
        "event_record": event_record,
    }
    accepted_coordinate = endpoint_coordinate if accepted else progress["current_coordinate"]
    accepted_state = endpoint_state if accepted else progress["current_state"]
    accepted_rate = endpoint_rate if accepted else progress["current_rate"]
    accepted_transform = (
        endpoint_field_arrays["metric_transform470x470"]
        if accepted else progress["metric_transform"]
    )
    accepted_augmented = (
        endpoint_field_arrays["metric_augmented560x560"]
        if accepted else progress["metric_augmented"]
    )
    accepted_gauge = (
        endpoint_field_arrays["gauge_basis560x90"]
        if accepted else progress["gauge_basis"]
    )
    arrays = {
        "candidate_target470": candidate,
        "endpoint_coordinate470": endpoint_coordinate,
        "endpoint_primitive_state": endpoint_state,
        "endpoint_coordinate_rate470_per_s": endpoint_rate,
        "accepted_coordinate470": np.asarray(accepted_coordinate),
        "accepted_primitive_state": np.asarray(accepted_state),
        "accepted_coordinate_rate470_per_s": np.asarray(accepted_rate),
        "accepted_metric_transform470x470": np.asarray(accepted_transform),
        "accepted_metric_augmented560x560": np.asarray(accepted_augmented),
        "accepted_gauge_basis560x90": np.asarray(accepted_gauge),
        "midpoint_target470": midpoint_target,
        "midpoint_hermite_rate470_per_s": midpoint_hermite_rate,
        "midpoint_coordinate470": midpoint_coordinate,
        "midpoint_primitive_state": midpoint_state,
        "midpoint_coordinate_rate470_per_s": midpoint_rate,
        "event_target470": event_target,
        "event_hermite_rate470_per_s": event_hermite_rate,
        "event_coordinate470": event_coordinate,
        "event_primitive_state": event_state,
        "event_coordinate_rate470_per_s": event_rate,
    }
    _write_pair(directory, "attempt", metrics, arrays)
    return metrics, arrays


def _restart_replay(
    records: list[tuple[dict, dict[str, np.ndarray]]],
    final_progress: dict,
) -> tuple[bool, bool, int | None]:
    accepted_positions = [
        index for index, record in enumerate(records) if record[0]["accepted"]
    ]
    if not accepted_positions:
        return False, False, None
    selected_accepted = min(
        manifest.RESTART_CHECKPOINT_ACCEPTED_INDEX,
        len(accepted_positions),
    )
    record_position = accepted_positions[selected_accepted - 1]
    directory = _attempt_directory(records[record_position][0]["attempt_index"])
    arrays_path = directory / "accepted_checkpoint.npz"
    metadata_path = directory / "accepted_checkpoint.json"
    if not arrays_path.exists() or not metadata_path.exists():
        return False, False, None
    first = _load_npz(arrays_path)
    second = _load_npz(arrays_path)
    metadata = _helper()._read(metadata_path)
    metadata_replay = _helper()._read(metadata_path)
    roundtrip = bool(
        set(first) == set(second)
        and all(np.array_equal(first[name], second[name]) for name in first)
        and metadata == metadata_replay
    )
    progress = _progress_from_checkpoint(first, metadata)
    replay = True
    for metrics, arrays in records[record_position + 1 :]:
        candidate = execution._variable_step_ab2(
            progress["current_coordinate"],
            progress["current_rate"],
            progress["previous_rate"],
            progress["next_span"],
            progress["previous_span"],
        )
        replay = bool(replay and np.array_equal(candidate, arrays["candidate_target470"]))
        if metrics["blind_midpoint_required"] and metrics["midpoint_field"] is not None:
            midpoint_target, midpoint_rate = execution._hermite(
                progress["current_coordinate"],
                progress["current_rate"],
                arrays["endpoint_coordinate470"],
                arrays["endpoint_coordinate_rate470_per_s"],
                metrics["span_seconds"],
                0.5,
            )
            replay = bool(
                replay
                and np.array_equal(midpoint_target, arrays["midpoint_target470"])
                and np.array_equal(midpoint_rate, arrays["midpoint_hermite_rate470_per_s"])
            )
        if metrics["event_fraction"] is not None and metrics["event_record"] is not None:
            event_target, event_rate = execution._hermite(
                progress["current_coordinate"],
                progress["current_rate"],
                arrays["endpoint_coordinate470"],
                arrays["endpoint_coordinate_rate470_per_s"],
                metrics["span_seconds"],
                metrics["event_fraction"],
            )
            replay = bool(
                replay
                and np.array_equal(event_target, arrays["event_target470"])
                and np.array_equal(event_rate, arrays["event_hermite_rate470_per_s"])
            )
        progress = _apply_record(progress, metrics, arrays)
    if final_progress["stop_classification"] in (
        BUDGET_CLASSIFICATION,
        EQUILIBRIUM_CLASSIFICATION,
    ):
        progress["stop_classification"] = final_progress["stop_classification"]
    scalar_names = (
        "elapsed_seconds",
        "accepted_segments_total",
        "accepted_segments_new",
        "attempts",
        "next_span",
        "previous_span",
        "accepted_since_growth",
        "previous_section",
        "seen_negative",
    )
    suffix = bool(
        replay
        and all(progress[name] == final_progress[name] for name in scalar_names)
        and np.array_equal(progress["current_coordinate"], final_progress["current_coordinate"])
        and np.array_equal(progress["current_state"], final_progress["current_state"])
        and np.array_equal(progress["current_rate"], final_progress["current_rate"])
        and progress["cycle_event"] == final_progress["cycle_event"]
        and progress["nonclosing_events"] == final_progress["nonclosing_events"]
        and progress["stop_classification"] == final_progress["stop_classification"]
    )
    return roundtrip, suffix, records[record_position][0]["attempt_index"]


def _execute(lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    del identity
    global _RUN_STARTED, _PRIOR_WALL
    helper = _helper()
    wall_path = SCRATCH_DIRECTORY / "cumulative_wall_seconds.json"
    _PRIOR_WALL = float(helper._read(wall_path)["wall_seconds"]) if wall_path.exists() else 0.0
    _RUN_STARTED = time.perf_counter()
    seed = _seed()
    inputs = execution.source._initial_inputs()
    exact_chart = execution.source.arclength._exact_chart()
    progress, records = _restore_progress()
    while progress["stop_classification"] is None and progress["cycle_event"] is None:
        inventory = _inventory()
        if (
            progress["accepted_segments_new"] >= manifest.MAXIMUM_ACCEPTED_SEGMENTS
            or progress["attempts"] >= manifest.MAXIMUM_ATTEMPTED_SEGMENTS
            or len(inventory["exact_field_paths"]) >= manifest.MAXIMUM_EXACT_FREE_FIELD_CALLS
            or _current_total_wall() >= 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
        ):
            progress["stop_classification"] = BUDGET_CLASSIFICATION
            break
        try:
            metrics, arrays = _attempt_segment(
                lock=lock,
                inputs=inputs,
                exact_chart=exact_chart,
                seed=seed,
                progress=progress,
                records=records,
            )
        except RuntimeError as error:
            if str(error) == "acquisition_budget":
                progress["stop_classification"] = BUDGET_CLASSIFICATION
                break
            raise
        records.append((metrics, arrays))
        progress = _apply_record(progress, metrics, arrays)
        if metrics["accepted"]:
            if not _write_checkpoint(_attempt_directory(metrics["attempt_index"]), progress):
                progress["stop_classification"] = NUMERICAL_FAILURE_CLASSIFICATION
                progress["cycle_event"] = None
                break
        helper._write_json(wall_path, {"wall_seconds": _current_total_wall()})
        print(
            f"attempt={metrics['attempt_index']:04d} accepted={metrics['accepted']} "
            f"h={metrics['span_seconds']:.3e}s endpoint={metrics['endpoint_integral_defect']:.3e} "
            f"blind={metrics['blind_midpoint_rate_defect']} "
            f"accepted_new={progress['accepted_segments_new']} "
            f"elapsed={progress['elapsed_seconds']:.6e}s next_h={progress['next_span']:.3e}s",
            flush=True,
        )
        if progress["cycle_event"] is not None or progress["stop_classification"] is not None:
            break
        if np.linalg.norm(progress["current_rate"]) <= manifest.EQUILIBRIUM_SPEED_RATIO * progress["initial_speed"]:
            progress["stop_classification"] = EQUILIBRIUM_CLASSIFICATION
            break

    total_wall = _current_total_wall()
    helper._write_json(wall_path, {"wall_seconds": total_wall})
    restart_roundtrip, suffix_replay, checkpoint_attempt = _restart_replay(records, progress)
    field_paths = _inventory()["exact_field_paths"]
    field_integrity = bool(
        field_paths
        and all(helper._read(path)["physical_passed"] for path in field_paths)
    )
    execution_integrity = bool(
        any(metrics["accepted"] for metrics, _arrays in records)
        and restart_roundtrip
        and suffix_replay
        and field_integrity
        and len(field_paths) <= manifest.MAXIMUM_EXACT_FREE_FIELD_CALLS
        and total_wall <= 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
    )
    if any(metrics["accepted"] for metrics, _arrays in records) and not execution_integrity:
        progress["stop_classification"] = NUMERICAL_FAILURE_CLASSIFICATION
        progress["cycle_event"] = None

    if progress["cycle_event"] is not None:
        classification = CYCLE_CLASSIFICATION
        passed = execution_integrity
        authorized_next = AUTHORIZED_CYCLE_NEXT if passed else None
    elif progress["stop_classification"] == EQUILIBRIUM_CLASSIFICATION:
        classification = EQUILIBRIUM_CLASSIFICATION
        passed = execution_integrity
        authorized_next = AUTHORIZED_EQUILIBRIUM_NEXT if passed else None
    elif progress["stop_classification"] == BUDGET_CLASSIFICATION:
        classification = BUDGET_CLASSIFICATION
        passed = execution_integrity
        authorized_next = AUTHORIZED_BUDGET_NEXT if passed else None
    else:
        classification = progress["stop_classification"] or NUMERICAL_FAILURE_CLASSIFICATION
        passed = False
        authorized_next = None

    accepted_records = [record for record in records if record[0]["accepted"]]
    exact_fields = []
    for metrics, _arrays in records:
        for name in ("endpoint_field", "midpoint_field"):
            if metrics.get(name) is not None:
                exact_fields.append(metrics[name])
        event_path = _attempt_directory(metrics["attempt_index"]) / "event_field.json"
        if event_path.exists():
            exact_fields.append(helper._read(event_path))
    raw_conditions = [field["free_field"]["coordinate_jacobian_condition_number"] for field in exact_fields]
    metric_conditions = [field["metric_chart"]["metric_jacobian_condition_number"] for field in exact_fields]
    transition_conditions = [field["metric_chart"]["patch_transition_condition_number"] for field in exact_fields]
    endpoint_defects = [metrics["endpoint_integral_defect"] for metrics, _arrays in records]
    accepted_endpoint_defects = [
        metrics["endpoint_integral_defect"]
        for metrics, _arrays in accepted_records
    ]
    blind_defects = [metrics["blind_midpoint_rate_defect"] for metrics, _arrays in records if metrics["blind_midpoint_rate_defect"] is not None]
    accepted_blind_defects = [
        metrics["blind_midpoint_rate_defect"]
        for metrics, _arrays in accepted_records
        if metrics["blind_midpoint_rate_defect"] is not None
    ]
    accepted_spans = np.asarray([metrics["span_seconds"] for metrics, _arrays in accepted_records])
    trajectory = np.vstack((
        seed["trajectory_coordinates"],
        np.stack([arrays["accepted_coordinate470"] for _metrics, arrays in accepted_records])
        if accepted_records else np.empty((0, 470)),
    ))
    trajectory_states = np.concatenate((
        seed["trajectory_primitive_states"],
        np.stack([arrays["accepted_primitive_state"] for _metrics, arrays in accepted_records])
        if accepted_records else np.empty((0, 112, 5)),
    ))
    section_values = (trajectory - progress["start_coordinate"]) @ progress["section_normal"]
    ledger_maxima = [
        max(
            value
            for name, value in field["free_field"]["reaction_free_ledger_values"].items()
            if name != "incoming_excision_characteristics"
        )
        for field in exact_fields
    ]
    gates = {
        "initial_elapsed_seconds": manifest.INITIAL_ELAPSED_SECONDS,
        "terminal_elapsed_seconds": float(progress["elapsed_seconds"]),
        "new_accepted_horizon_seconds": float(progress["elapsed_seconds"] - manifest.INITIAL_ELAPSED_SECONDS),
        "attempted_segments": len(records),
        "accepted_segments": len(accepted_records),
        "rejected_segments": len(records) - len(accepted_records),
        "exact_free_field_calls": len(_inventory()["exact_field_paths"]),
        "execution_wall_seconds": total_wall,
        "minimum_accepted_span_seconds": None if not len(accepted_spans) else float(np.min(accepted_spans)),
        "maximum_accepted_span_seconds": None if not len(accepted_spans) else float(np.max(accepted_spans)),
        "maximum_accepted_endpoint_integral_defect": max(
            accepted_endpoint_defects, default=0.0
        ),
        "maximum_attempted_endpoint_integral_defect": max(
            endpoint_defects, default=0.0
        ),
        "maximum_accepted_blind_midpoint_rate_defect": max(
            accepted_blind_defects, default=0.0
        ),
        "maximum_attempted_blind_midpoint_rate_defect": max(
            blind_defects, default=0.0
        ),
        "maximum_raw_coordinate_jacobian_condition": max(raw_conditions, default=0.0),
        "maximum_metric_coordinate_jacobian_condition": max(metric_conditions, default=0.0),
        "maximum_patch_transition_condition": max(transition_conditions, default=0.0),
        "all_exact_fields_physical_passed": bool(exact_fields and all(field["physical_passed"] for field in exact_fields)),
        "minimum_reconstruction_factor": min((field["free_field"]["minimum_reconstruction_factor"] for field in exact_fields), default=1.0),
        "maximum_height_ratio": max((field["free_field"]["maximum_height_ratio"] for field in exact_fields), default=0.0),
        "minimum_scattering_optical_depth": min((field["free_field"]["minimum_scattering_optical_depth"] for field in exact_fields), default=float("inf")),
        "maximum_reaction_free_ledger_defect": max(ledger_maxima, default=0.0),
        "minimum_section_value": float(np.min(section_values)),
        "terminal_section_value": float(section_values[-1]),
        "nonclosing_poincare_returns": len(progress["nonclosing_events"]),
        "cycle_observed": progress["cycle_event"] is not None,
        "cycle_duration_seconds": None if progress["cycle_event"] is None else progress["cycle_event"]["elapsed_seconds"],
        "equilibrium_candidate_observed": classification == EQUILIBRIUM_CLASSIFICATION,
        "restart_roundtrip_bitwise": restart_roundtrip,
        "suffix_history_replay_bitwise": suffix_replay,
        "restart_checkpoint_attempt": checkpoint_attempt,
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "gate_values": gates,
        "cycle_event": progress["cycle_event"],
        "nonclosing_events": progress["nonclosing_events"],
        "fixed_Q_physical_rate_calls": 0,
        "fixed_Q_reaction_calls": 0,
        "nonlinear_roots": 0,
        "BDF_microsteps": 0,
    }
    arrays = {
        **_checkpoint_arrays(progress),
        "trajectory_coordinates": trajectory,
        "trajectory_primitive_states": trajectory_states,
        "accepted_endpoint_rates470_per_s": (
            np.stack([arrays["accepted_coordinate_rate470_per_s"] for _metrics, arrays in accepted_records])
            if accepted_records else np.empty((0, 470))
        ),
        "accepted_segment_seconds": accepted_spans,
        "attempted_endpoint_integral_defects": np.asarray(endpoint_defects),
        "blind_midpoint_rate_defects": np.asarray(blind_defects),
        "section_values": section_values,
    }
    cycle_record = next((record for record in accepted_records if record[0]["cycle_event"] is not None), None)
    if cycle_record is not None:
        arrays.update({
            "cycle_event_coordinate470": cycle_record[1]["event_coordinate470"],
            "cycle_event_primitive_state": cycle_record[1]["event_primitive_state"],
            "cycle_event_coordinate_rate470_per_s": cycle_record[1]["event_coordinate_rate470_per_s"],
        })
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": status,
            })
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(summary_path)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(summary_path, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray], lock: dict, identity: dict) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("wide-resume result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "resume_execution_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "resume_execution_arrays.npz", arrays)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", {
        "manifest_hashes": lock["hashes"],
        "manifest_classification": lock["summary"]["classification"],
        "execution_identity": identity,
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "cycle_observed": metrics["gate_values"]["cycle_observed"],
        "equilibrium_candidate_observed": metrics["gate_values"]["equilibrium_candidate_observed"],
        "clean_budget_exhausted": metrics["classification"] == BUDGET_CLASSIFICATION,
        "new_accepted_segments": metrics["gate_values"]["accepted_segments"],
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": identity["implementation_commit"],
        "implementation_tree": identity["implementation_tree"],
        "source_hashes": identity["source_hashes"],
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Metric-chart wide-continuation resume execution",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"Accepted `{values['accepted_segments']}` segments over `{values['new_accepted_horizon_seconds']:.6f}` s, reaching `{values['terminal_elapsed_seconds']:.6f}` s. Exact calls: `{values['exact_free_field_calls']}`; wall time: `{values['execution_wall_seconds'] / 3600.0:.3f}` h.",
            "",
            f"Maximum accepted endpoint/blind defects were `{values['maximum_accepted_endpoint_integral_defect']:.6e}` and `{values['maximum_accepted_blind_midpoint_rate_defect']:.6e}`. Maximum raw/metric conditions were `{values['maximum_raw_coordinate_jacobian_condition']:.6e}` and `{values['maximum_metric_coordinate_jacobian_condition']:.6e}`.",
            "",
            f"Cycle observed: `{values['cycle_observed']}`. Equilibrium candidate: `{values['equilibrium_candidate_observed']}`. Restart/replay: `{values['restart_roundtrip_bitwise']}`/`{values['suffix_history_replay_bitwise']}`.",
            "",
            f"Authorized next artifact: `{metrics['authorized_next']}`.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("--run is required")
    lock = _validate_manifest(require_clean=True)
    identity = _prepare_scratch(lock)
    metrics, arrays = _execute(lock, identity)
    summary = _canonicalize(metrics, arrays, lock, identity)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
