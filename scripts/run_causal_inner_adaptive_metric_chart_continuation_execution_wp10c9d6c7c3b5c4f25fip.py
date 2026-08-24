#!/usr/bin/env python3
"""Execute a short repeated adaptive conservative metric-chart tranche."""

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

import run_causal_inner_adaptive_metric_chart_continuation_manifest_wp10c9d6c7c3b5c4f25fio as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.adaptive_metric_chart_continuation import (  # noqa: E402
    AdaptiveMetricChartPolicy,
    blind_midpoint_required,
    strict_chart_failure_is_retryable,
    transition_after_attempt,
)
from imri_qpe.layer3_minidisk_1d.conservative_metric_chart_atlas_v2 import (  # noqa: E402
    ConservativeMetricChart,
    metric_transport_retract_strict,
)


parent = manifest.parent
wide = parent.wide
suffix = parent.suffix
execution = parent.execution
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fip"
PASS_CLASSIFICATION = "adaptive_metric_chart_continuation_passed"
PHYSICAL_FAILURE_CLASSIFICATION = (
    "adaptive_metric_chart_continuation_original_physical_gate_failed"
)
NUMERICAL_FAILURE_CLASSIFICATION = (
    "adaptive_metric_chart_continuation_numerical_or_restart_failed"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fiq_adaptive_metric_chart_cycle_readiness_manifest"
)
ARTIFACT = (
    "causal_inner_adaptive_metric_chart_continuation_execution_"
    "wp10c9d6c7c3b5c4f25fip"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_METRIC_CHART_"
    "CONTINUATION_EXECUTION_WP10C9D6C7C3B5C4F25FIP_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_adaptive_metric_chart_continuation_execution_"
    "wp10c9d6c7c3b5c4f25fip.py"
)
THIS_TEST = (
    "tests/test_causal_inner_adaptive_metric_chart_continuation_execution_"
    "wp10c9d6c7c3b5c4f25fip.py"
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
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "continuation_contract.json"
    )
    cost = helper._read(manifest.CANONICAL_DIRECTORY / "cost_projection.json")
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["adaptive_metric_chart_continuation_authorized"]
        or summary["adaptive_metric_chart_continuation_executed"]
        or summary["new_trajectory"]
        or summary["authorized_next"] != manifest.AUTHORIZED_NEXT
        or contract["authorized_execution"] != manifest.AUTHORIZED_NEXT
        or contract["scope"]["maximum_accepted_segments"] != 8
        or contract["scope"]["maximum_attempted_segments"] != 12
        or contract["scope"]["maximum_exact_free_field_calls"] != 12
        or not contract["adaptive_policy"][
            "physically_admissible_chart_failure_halves_span"
        ]
        or not contract["adaptive_policy"]["physical_failure_stops"]
        or not contract["adaptive_policy"]["rejected_candidate_is_never_propagated"]
        or not cost["cost_gate_passed"]
    ):
        raise RuntimeError("adaptive continuation authorization changed")
    for relative, frozen_hash in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != frozen_hash:
            raise RuntimeError(f"frozen adaptive continuation source changed: {relative}")
    parent_lock = manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("adaptive continuation execution requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "cost": cost,
        "parent_lock": parent_lock,
    }


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "continuation_seed.npz")


def _policy() -> AdaptiveMetricChartPolicy:
    return AdaptiveMetricChartPolicy(
        minimum_span_seconds=manifest.MINIMUM_SEGMENT_SECONDS,
        maximum_span_seconds=manifest.MAXIMUM_SEGMENT_SECONDS,
        growth_factor=manifest.GROWTH_FACTOR,
        accepted_segments_before_growth=manifest.ACCEPTED_SEGMENTS_BEFORE_GROWTH,
        blind_midpoint_frequency=manifest.BLIND_MIDPOINT_FREQUENCY,
    )


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        manifest.POLICY_SOURCE,
        manifest.POLICY_TEST,
        parent.THIS_RUNNER,
        parent.diagnosis.manifest.STRICT_ATLAS_SOURCE,
        suffix.THIS_RUNNER,
        execution.source.THIS_RUNNER,
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _identity(lock: dict) -> dict:
    helper = _helper()
    return {
        "work_package": WORK_PACKAGE,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "manifest_hashes": lock["hashes"],
        "source_hashes": _source_hashes(),
        "policy": manifest._contract()["adaptive_policy"],
    }


def _prepare_scratch(lock: dict) -> dict:
    helper = _helper()
    identity = _identity(lock)
    path = SCRATCH_DIRECTORY / "execution_identity.json"
    if SCRATCH_DIRECTORY.exists():
        if not path.exists() or helper._read(path) != identity:
            raise RuntimeError("adaptive continuation scratch identity mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(path, identity)
    return identity


def _attempt_directory(index: int) -> Path:
    return SCRATCH_DIRECTORY / f"attempt_{int(index):04d}"


def _inventory() -> dict:
    attempts = []
    fields = []
    retractions = []
    if SCRATCH_DIRECTORY.exists():
        for directory in sorted(SCRATCH_DIRECTORY.glob("attempt_[0-9][0-9][0-9][0-9]")):
            metrics = directory / "attempt.json"
            arrays = directory / "attempt.npz"
            if metrics.exists() != arrays.exists():
                raise RuntimeError("incomplete adaptive attempt cache")
            if metrics.exists():
                attempts.append((directory, metrics, arrays))
            fields.extend(sorted(directory.glob("*_field.json")))
            retractions.extend(sorted(directory.glob("*_retraction.json")))
    return {"attempts": attempts, "fields": fields, "retractions": retractions}


def _initial_progress() -> dict:
    seed = _seed()
    return {
        "previous_coordinate": seed["previous_coordinate470"].copy(),
        "current_coordinate": seed["current_coordinate470"].copy(),
        "previous_state": seed["previous_primitive_state"].copy(),
        "current_state": seed["current_primitive_state"].copy(),
        "previous_rate": seed["previous_coordinate_rate470_per_s"].copy(),
        "current_rate": seed["current_coordinate_rate470_per_s"].copy(),
        "previous_span": float(seed["previous_span_seconds"]),
        "next_span": float(seed["next_span_seconds"]),
        "elapsed_seconds": float(seed["elapsed_seconds"]),
        "accepted_segments_total": int(seed["accepted_segments_total"]),
        "accepted_segments_new": 0,
        "attempts": 0,
        "accepted_since_growth": int(seed["accepted_since_growth"]),
        "metric_transform": seed["current_metric_transform470x470"].copy(),
        "metric_augmented": seed["current_metric_augmented560x560"].copy(),
        "gauge_basis": seed["current_gauge_basis560x90"].copy(),
        "section_normal": seed["section_normal470"].copy(),
        "start_coordinate": seed["start_coordinate470"].copy(),
        "stop_reason": None,
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
        "metric_transform470x470": np.asarray(progress["metric_transform"]),
        "metric_augmented560x560": np.asarray(progress["metric_augmented"]),
        "gauge_basis560x90": np.asarray(progress["gauge_basis"]),
        "section_normal470": np.asarray(progress["section_normal"]),
        "start_coordinate470": np.asarray(progress["start_coordinate"]),
    }


def _progress_from_checkpoint(arrays: dict[str, np.ndarray]) -> dict:
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
        "metric_transform": arrays["metric_transform470x470"].copy(),
        "metric_augmented": arrays["metric_augmented560x560"].copy(),
        "gauge_basis": arrays["gauge_basis560x90"].copy(),
        "section_normal": arrays["section_normal470"].copy(),
        "start_coordinate": arrays["start_coordinate470"].copy(),
        "stop_reason": None,
    }


def _write_checkpoint(directory: Path, progress: dict) -> bool:
    path = directory / "accepted_checkpoint.npz"
    arrays = _checkpoint_arrays(progress)
    _save_npz(path, arrays)
    replay = _load_npz(path)
    return bool(
        set(arrays) == set(replay)
        and all(np.array_equal(arrays[name], replay[name]) for name in arrays)
    )


def _apply_record(progress: dict, metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    result = dict(progress)
    result["attempts"] = int(progress["attempts"]) + 1
    result["next_span"] = float(metrics["next_span_seconds"])
    result["accepted_since_growth"] = int(metrics["accepted_since_growth_after"])
    result["stop_reason"] = metrics["stop_reason"]
    if not metrics["accepted"]:
        return result
    result.update(
        {
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
            "metric_transform": np.asarray(arrays["accepted_metric_transform470x470"]),
            "metric_augmented": np.asarray(arrays["accepted_metric_augmented560x560"]),
            "gauge_basis": np.asarray(arrays["accepted_gauge_basis560x90"]),
        }
    )
    return result


def _restore_progress() -> tuple[dict, list[tuple[dict, dict[str, np.ndarray]]]]:
    progress = _initial_progress()
    records = []
    for expected, (directory, metrics_path, arrays_path) in enumerate(
        _inventory()["attempts"]
    ):
        if int(directory.name.split("_")[-1]) != expected:
            raise RuntimeError("adaptive attempt scratch is not contiguous")
        metrics = _helper()._read(metrics_path)
        arrays = _load_npz(arrays_path)
        if (
            metrics["attempt_index"] != expected
            or metrics["span_seconds"] != progress["next_span"]
            or metrics["tentative_segment_number"]
            != progress["accepted_segments_total"] + 1
        ):
            raise RuntimeError("adaptive attempt metadata changed")
        candidate = execution._variable_step_ab2(
            progress["current_coordinate"],
            progress["current_rate"],
            progress["previous_rate"],
            progress["next_span"],
            progress["previous_span"],
        )
        np.testing.assert_array_equal(candidate, arrays["candidate_target470"])
        progress = _apply_record(progress, metrics, arrays)
        if metrics["accepted"] and not _write_checkpoint(directory, progress):
            raise RuntimeError("restored adaptive checkpoint failed roundtrip")
        records.append((metrics, arrays))
    return progress, records


_RUN_STARTED = 0.0
_PRIOR_WALL = 0.0


def _total_wall() -> float:
    return float(_PRIOR_WALL + max(0.0, time.perf_counter() - _RUN_STARTED))


def _strict_retraction(
    *,
    directory: Path,
    stem: str,
    target: np.ndarray,
    progress: dict,
    inputs: dict,
    exact_chart,
    anchor_chart: ConservativeMetricChart,
) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    metrics_path = directory / f"{stem}.json"
    arrays_path = directory / f"{stem}.npz"
    if metrics_path.exists() or arrays_path.exists():
        if not metrics_path.exists() or not arrays_path.exists():
            raise RuntimeError("incomplete adaptive retraction cache")
        metrics = helper._read(metrics_path)
        arrays = _load_npz(arrays_path)
        np.testing.assert_array_equal(arrays["target_original_coordinate470"], target)
        print(f"{directory.name}/{stem}: reused", flush=True)
        return metrics, arrays
    initial = suffix.parent.parent._initial_state(
        inputs["model"],
        progress["current_state"],
        progress["current_coordinate"],
        target,
    )
    state, matrix, metrics = metric_transport_retract_strict(
        exact_chart=exact_chart,
        model=inputs["model"],
        initial_state=initial,
        target_original_coordinate=target,
        gauge_basis=progress["gauge_basis"],
        anchor_delta=exact_chart._delta(inputs["model"], progress["current_state"]),
        anchor_metric_augmented=progress["metric_augmented"],
        chart=anchor_chart,
        policy=suffix._policy(),
    )
    recovered, factors = inputs["model"].coordinate(state)
    arrays = {
        "target_original_coordinate470": np.asarray(target),
        "recovered_original_coordinate470": np.asarray(recovered),
        "primitive_state": np.asarray(state),
        "final_metric_broyden560x560": np.asarray(matrix),
        "decoder_reconstruction_factors": np.asarray(factors),
    }
    helper._write_json(metrics_path, metrics)
    _save_npz(arrays_path, arrays)
    print(
        f"{directory.name}/{stem}: strict={metrics['passed']} "
        f"condition={metrics['maximum_metric_augmented_condition_number']:.6g}",
        flush=True,
    )
    return metrics, arrays


def _attempt(
    *,
    progress: dict,
    inputs: dict,
    exact_chart,
) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    index = int(progress["attempts"])
    directory = _attempt_directory(index)
    directory.mkdir(exist_ok=True)
    metrics_path = directory / "attempt.json"
    arrays_path = directory / "attempt.npz"
    span = float(progress["next_span"])
    tentative = int(progress["accepted_segments_total"]) + 1
    blind = blind_midpoint_required(tentative, _policy())
    candidate = execution._variable_step_ab2(
        progress["current_coordinate"],
        progress["current_rate"],
        progress["previous_rate"],
        span,
        progress["previous_span"],
    )
    if metrics_path.exists() or arrays_path.exists():
        if not metrics_path.exists() or not arrays_path.exists():
            raise RuntimeError("incomplete adaptive attempt pair")
        metrics = helper._read(metrics_path)
        arrays = _load_npz(arrays_path)
        if (
            metrics["attempt_index"] != index
            or metrics["span_seconds"] != span
            or metrics["tentative_segment_number"] != tentative
            or metrics["blind_midpoint_required"] != blind
        ):
            raise RuntimeError("cached adaptive attempt metadata changed")
        np.testing.assert_array_equal(arrays["candidate_target470"], candidate)
        print(f"{directory.name}: reused completed attempt", flush=True)
        return metrics, arrays
    inventory = _inventory()
    required_fields = 1 + int(blind)
    if (
        len(inventory["fields"]) + required_fields
        > manifest.MAXIMUM_EXACT_FREE_FIELD_CALLS
        or len(inventory["retractions"]) + required_fields
        > manifest.MAXIMUM_RETRACTIONS
    ):
        raise RuntimeError("acquisition_budget")
    anchor_chart = ConservativeMetricChart(
        progress["current_coordinate"],
        progress["metric_transform"],
        suffix._block_sizes(),
    )
    endpoint_retraction, endpoint_retraction_arrays = _strict_retraction(
        directory=directory,
        stem="endpoint_retraction",
        target=candidate,
        progress=progress,
        inputs=inputs,
        exact_chart=exact_chart,
        anchor_chart=anchor_chart,
    )
    endpoint_coordinate = np.asarray(
        endpoint_retraction_arrays["recovered_original_coordinate470"]
    )
    endpoint_state = np.asarray(endpoint_retraction_arrays["primitive_state"])
    physical_failure = not endpoint_retraction["physical_passed"]
    retryable_chart_failure = strict_chart_failure_is_retryable(
        endpoint_retraction
    )
    endpoint_field = None
    endpoint_field_arrays = None
    endpoint_rate = np.full(470, np.nan)
    endpoint_defect = float("inf")
    numerical_passed = bool(endpoint_retraction["passed"] and not physical_failure)
    if numerical_passed:
        endpoint_field, endpoint_field_arrays = suffix._metric_field(
            directory=directory,
            stem="endpoint_field",
            inputs=inputs,
            exact_chart=exact_chart,
            state=endpoint_state,
            coordinate=endpoint_coordinate,
            retraction=endpoint_retraction,
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
        if not endpoint_field["physical_passed"]:
            physical_failure = True
            numerical_passed = False
        else:
            numerical_passed = bool(
                endpoint_defect <= manifest.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
            )
    midpoint_target = np.full(470, np.nan)
    midpoint_hermite_rate = np.full(470, np.nan)
    midpoint_coordinate = np.full(470, np.nan)
    midpoint_state = np.full((112, 5), np.nan)
    midpoint_rate = np.full(470, np.nan)
    midpoint_retraction = None
    midpoint_field = None
    midpoint_field_arrays = None
    midpoint_defect = None
    if numerical_passed and blind:
        midpoint_target, midpoint_hermite_rate = execution._hermite(
            progress["current_coordinate"],
            progress["current_rate"],
            endpoint_coordinate,
            endpoint_rate,
            span,
            0.5,
        )
        midpoint_retraction, midpoint_retraction_arrays = _strict_retraction(
            directory=directory,
            stem="midpoint_retraction",
            target=midpoint_target,
            progress=progress,
            inputs=inputs,
            exact_chart=exact_chart,
            anchor_chart=anchor_chart,
        )
        midpoint_coordinate = np.asarray(
            midpoint_retraction_arrays["recovered_original_coordinate470"]
        )
        midpoint_state = np.asarray(midpoint_retraction_arrays["primitive_state"])
        if not midpoint_retraction["physical_passed"]:
            physical_failure = True
            numerical_passed = False
        elif not midpoint_retraction["passed"]:
            retryable_chart_failure = bool(
                retryable_chart_failure
                or strict_chart_failure_is_retryable(midpoint_retraction)
            )
            numerical_passed = False
        else:
            midpoint_field, midpoint_field_arrays = suffix._metric_field(
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
            midpoint_defect = suffix._relative(midpoint_hermite_rate, midpoint_rate)
            if not midpoint_field["physical_passed"]:
                physical_failure = True
                numerical_passed = False
            else:
                numerical_passed = bool(
                    midpoint_defect
                    <= manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
                )
    accepted = bool(numerical_passed and not physical_failure)
    transition = transition_after_attempt(
        policy=_policy(),
        span_seconds=span,
        tentative_segment_number=tentative,
        accepted=accepted,
        physical_failure=physical_failure,
        accepted_since_growth=int(progress["accepted_since_growth"]),
    )
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
        "blind_midpoint_required": blind,
        "endpoint_integral_defect": endpoint_defect,
        "blind_midpoint_rate_defect": midpoint_defect,
        "endpoint_retraction_passed": endpoint_retraction["passed"],
        "endpoint_retraction_physical_passed": endpoint_retraction[
            "physical_passed"
        ],
        "endpoint_metric_augmented_condition": endpoint_retraction[
            "maximum_metric_augmented_condition_number"
        ],
        "retryable_chart_failure": retryable_chart_failure,
        "endpoint_physical_passed": (
            None if endpoint_field is None else endpoint_field["physical_passed"]
        ),
        "midpoint_retraction_passed": (
            None if midpoint_retraction is None else midpoint_retraction["passed"]
        ),
        "midpoint_physical_passed": (
            None if midpoint_field is None else midpoint_field["physical_passed"]
        ),
        "physical_failure": physical_failure,
        "numerical_passed": numerical_passed,
        "accepted": accepted,
        "accepted_since_growth_after": transition["accepted_since_growth"],
        "next_span_seconds": transition["next_span_seconds"],
        "stop_reason": transition["stop_reason"],
        "elapsed_seconds_after": elapsed_after,
        "endpoint_field": endpoint_field,
        "midpoint_field": midpoint_field,
    }
    arrays = {
        "candidate_target470": candidate,
        "endpoint_coordinate470": endpoint_coordinate,
        "endpoint_primitive_state": endpoint_state,
        "endpoint_coordinate_rate470_per_s": endpoint_rate,
        "midpoint_target470": midpoint_target,
        "midpoint_hermite_rate470_per_s": midpoint_hermite_rate,
        "midpoint_coordinate470": midpoint_coordinate,
        "midpoint_primitive_state": midpoint_state,
        "midpoint_coordinate_rate470_per_s": midpoint_rate,
        "accepted_coordinate470": (
            endpoint_coordinate if accepted else progress["current_coordinate"]
        ),
        "accepted_primitive_state": (
            endpoint_state if accepted else progress["current_state"]
        ),
        "accepted_coordinate_rate470_per_s": (
            endpoint_rate if accepted else progress["current_rate"]
        ),
        "accepted_metric_transform470x470": (
            endpoint_field_arrays["metric_transform470x470"]
            if accepted
            else progress["metric_transform"]
        ),
        "accepted_metric_augmented560x560": (
            endpoint_field_arrays["metric_augmented560x560"]
            if accepted
            else progress["metric_augmented"]
        ),
        "accepted_gauge_basis560x90": (
            endpoint_field_arrays["gauge_basis560x90"]
            if accepted
            else progress["gauge_basis"]
        ),
    }
    helper._write_json(metrics_path, metrics)
    _save_npz(arrays_path, arrays)
    return metrics, arrays


def _restart_replay(
    records: list[tuple[dict, dict[str, np.ndarray]]],
    final_progress: dict,
) -> tuple[bool, int | None]:
    accepted_positions = [
        index for index, (metrics, _arrays) in enumerate(records) if metrics["accepted"]
    ]
    if len(accepted_positions) < 4:
        return False, None
    position = accepted_positions[3]
    attempt = records[position][0]["attempt_index"]
    path = _attempt_directory(attempt) / "accepted_checkpoint.npz"
    if not path.exists():
        return False, None
    progress = _progress_from_checkpoint(_load_npz(path))
    replay = True
    for metrics, arrays in records[position + 1 :]:
        candidate = execution._variable_step_ab2(
            progress["current_coordinate"],
            progress["current_rate"],
            progress["previous_rate"],
            progress["next_span"],
            progress["previous_span"],
        )
        replay = bool(replay and np.array_equal(candidate, arrays["candidate_target470"]))
        if (
            metrics["blind_midpoint_required"]
            and metrics["midpoint_retraction_passed"] is not None
        ):
            target, rate = execution._hermite(
                progress["current_coordinate"],
                progress["current_rate"],
                arrays["endpoint_coordinate470"],
                arrays["endpoint_coordinate_rate470_per_s"],
                metrics["span_seconds"],
                0.5,
            )
            replay = bool(
                replay
                and np.array_equal(target, arrays["midpoint_target470"])
                and np.array_equal(rate, arrays["midpoint_hermite_rate470_per_s"])
            )
        progress = _apply_record(progress, metrics, arrays)
    scalar_names = (
        "previous_span",
        "next_span",
        "elapsed_seconds",
        "accepted_segments_total",
        "accepted_segments_new",
        "attempts",
        "accepted_since_growth",
        "stop_reason",
    )
    array_names = (
        "previous_coordinate",
        "current_coordinate",
        "previous_state",
        "current_state",
        "previous_rate",
        "current_rate",
        "metric_transform",
        "metric_augmented",
        "gauge_basis",
        "section_normal",
        "start_coordinate",
    )
    replay = bool(
        replay
        and all(progress[name] == final_progress[name] for name in scalar_names)
        and all(
            np.array_equal(progress[name], final_progress[name])
            for name in array_names
        )
    )
    return replay, attempt


def _execute(lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    del identity
    global _RUN_STARTED, _PRIOR_WALL
    helper = _helper()
    wall_path = SCRATCH_DIRECTORY / "cumulative_wall_seconds.json"
    _PRIOR_WALL = (
        float(helper._read(wall_path)["wall_seconds"])
        if wall_path.exists()
        else 0.0
    )
    _RUN_STARTED = time.perf_counter()
    inputs = execution.source._initial_inputs()
    exact_chart = execution.source.arclength._exact_chart()
    progress, records = _restore_progress()
    checkpoint_roundtrips = [
        metrics["accepted"]
        and (_attempt_directory(metrics["attempt_index"]) / "accepted_checkpoint.npz").exists()
        for metrics, _arrays in records
        if metrics["accepted"]
    ]
    while (
        progress["accepted_segments_new"] < manifest.MAXIMUM_ACCEPTED_SEGMENTS
        and progress["stop_reason"] is None
    ):
        inventory = _inventory()
        if (
            progress["attempts"] >= manifest.MAXIMUM_ATTEMPTED_SEGMENTS
            or len(inventory["fields"]) >= manifest.MAXIMUM_EXACT_FREE_FIELD_CALLS
            or len(inventory["retractions"]) >= manifest.MAXIMUM_RETRACTIONS
            or _total_wall() >= 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
        ):
            progress["stop_reason"] = "execution_budget"
            break
        try:
            metrics, arrays = _attempt(
                progress=progress,
                inputs=inputs,
                exact_chart=exact_chart,
            )
        except RuntimeError as error:
            if str(error) == "acquisition_budget":
                progress["stop_reason"] = "acquisition_budget"
                break
            raise
        records.append((metrics, arrays))
        progress = _apply_record(progress, metrics, arrays)
        if metrics["accepted"]:
            roundtrip = _write_checkpoint(
                _attempt_directory(metrics["attempt_index"]), progress
            )
            checkpoint_roundtrips.append(roundtrip)
            if not roundtrip:
                progress["stop_reason"] = "checkpoint_roundtrip"
        helper._write_json(wall_path, {"wall_seconds": _total_wall()})
        print(
            f"attempt={metrics['attempt_index']:02d} accepted={metrics['accepted']} "
            f"h={metrics['span_seconds']:.3e}s "
            f"chart_retry={metrics['retryable_chart_failure']} "
            f"endpoint={metrics['endpoint_integral_defect']} "
            f"blind={metrics['blind_midpoint_rate_defect']} "
            f"accepted_new={progress['accepted_segments_new']} "
            f"elapsed={progress['elapsed_seconds']:.6f}s "
            f"next_h={progress['next_span']:.3e}s",
            flush=True,
        )
    total_wall = _total_wall()
    helper._write_json(wall_path, {"wall_seconds": total_wall})
    replay, replay_attempt = _restart_replay(records, progress)
    inventory = _inventory()
    field_metrics = [helper._read(path) for path in inventory["fields"]]
    accepted_records = [record for record in records if record[0]["accepted"]]
    physical_failure = any(metrics["physical_failure"] for metrics, _arrays in records)
    completed = progress["accepted_segments_new"] == manifest.MAXIMUM_ACCEPTED_SEGMENTS
    integrity = bool(
        completed
        and checkpoint_roundtrips
        and all(checkpoint_roundtrips)
        and replay
        and field_metrics
        and all(field["physical_passed"] for field in field_metrics)
        and len(inventory["fields"]) <= manifest.MAXIMUM_EXACT_FREE_FIELD_CALLS
        and len(inventory["retractions"]) <= manifest.MAXIMUM_RETRACTIONS
        and total_wall <= 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
    )
    if integrity:
        classification = PASS_CLASSIFICATION
        passed = True
        authorized_next = AUTHORIZED_NEXT
    elif physical_failure:
        classification = PHYSICAL_FAILURE_CLASSIFICATION
        passed = False
        authorized_next = None
    else:
        classification = NUMERICAL_FAILURE_CLASSIFICATION
        passed = False
        authorized_next = None
    endpoint_defects = [
        metrics["endpoint_integral_defect"]
        for metrics, _arrays in accepted_records
    ]
    blind_defects = [
        metrics["blind_midpoint_rate_defect"]
        for metrics, _arrays in accepted_records
        if metrics["blind_midpoint_rate_defect"] is not None
    ]
    raw_conditions = [
        field["free_field"]["coordinate_jacobian_condition_number"]
        for field in field_metrics
    ]
    metric_conditions = [
        field["metric_chart"]["metric_jacobian_condition_number"]
        for field in field_metrics
    ]
    ledger_values = [
        max(
            value
            for name, value in field["free_field"][
                "reaction_free_ledger_values"
            ].items()
            if name != "incoming_excision_characteristics"
        )
        for field in field_metrics
    ]
    accepted_coordinates = (
        np.stack([arrays["accepted_coordinate470"] for _metrics, arrays in accepted_records])
        if accepted_records
        else np.empty((0, 470))
    )
    accepted_states = (
        np.stack([arrays["accepted_primitive_state"] for _metrics, arrays in accepted_records])
        if accepted_records
        else np.empty((0, 112, 5))
    )
    accepted_rates = (
        np.stack([arrays["accepted_coordinate_rate470_per_s"] for _metrics, arrays in accepted_records])
        if accepted_records
        else np.empty((0, 470))
    )
    section_values = (
        (accepted_coordinates - progress["start_coordinate"])
        @ progress["section_normal"]
        if len(accepted_coordinates)
        else np.empty(0)
    )
    gates = {
        "initial_elapsed_seconds": manifest.INITIAL_ELAPSED_SECONDS,
        "terminal_elapsed_seconds": progress["elapsed_seconds"],
        "new_accepted_horizon_seconds": (
            progress["elapsed_seconds"] - manifest.INITIAL_ELAPSED_SECONDS
        ),
        "attempted_segments": len(records),
        "accepted_segments": len(accepted_records),
        "rejected_segments": len(records) - len(accepted_records),
        "retryable_chart_failures": sum(
            bool(metrics["retryable_chart_failure"]) for metrics, _arrays in records
        ),
        "accepted_segment_seconds": [
            metrics["span_seconds"] for metrics, _arrays in accepted_records
        ],
        "terminal_next_span_seconds": progress["next_span"],
        "maximum_accepted_endpoint_integral_defect": max(endpoint_defects, default=0.0),
        "maximum_accepted_blind_midpoint_rate_defect": max(blind_defects, default=0.0),
        "maximum_raw_coordinate_jacobian_condition": max(raw_conditions, default=0.0),
        "maximum_metric_coordinate_jacobian_condition": max(metric_conditions, default=0.0),
        "minimum_reconstruction_factor": min(
            (field["free_field"]["minimum_reconstruction_factor"] for field in field_metrics),
            default=1.0,
        ),
        "maximum_height_ratio": max(
            (field["free_field"]["maximum_height_ratio"] for field in field_metrics),
            default=0.0,
        ),
        "minimum_scattering_optical_depth": min(
            (field["free_field"]["minimum_scattering_optical_depth"] for field in field_metrics),
            default=float("inf"),
        ),
        "maximum_reaction_free_ledger_defect": max(ledger_values, default=0.0),
        "minimum_new_section_value": (
            None if not len(section_values) else float(np.min(section_values))
        ),
        "terminal_section_value": (
            None if not len(section_values) else float(section_values[-1])
        ),
        "all_accepted_checkpoint_roundtrips_bitwise": bool(
            checkpoint_roundtrips and all(checkpoint_roundtrips)
        ),
        "suffix_history_replay_bitwise": replay,
        "restart_checkpoint_attempt": replay_attempt,
        "exact_free_field_calls": len(inventory["fields"]),
        "retractions": len(inventory["retractions"]),
        "fixed_Q_calls": 0,
        "reaction_calls": 0,
        "nonlinear_roots": 0,
        "BDF_microsteps": 0,
        "execution_wall_seconds": total_wall,
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "stop_reason": progress["stop_reason"],
        "gate_values": gates,
    }
    arrays = {
        **_checkpoint_arrays(progress),
        "accepted_endpoint_coordinates470": accepted_coordinates,
        "accepted_endpoint_primitive_states": accepted_states,
        "accepted_endpoint_coordinate_rates470_per_s": accepted_rates,
        "accepted_segment_seconds": np.asarray(
            [metrics["span_seconds"] for metrics, _arrays in accepted_records]
        ),
        "attempted_segment_seconds": np.asarray(
            [metrics["span_seconds"] for metrics, _arrays in records]
        ),
        "attempted_acceptance": np.asarray(
            [metrics["accepted"] for metrics, _arrays in records], dtype=bool
        ),
        "attempted_retryable_chart_failure": np.asarray(
            [metrics["retryable_chart_failure"] for metrics, _arrays in records],
            dtype=bool,
        ),
        "new_section_values": section_values,
    }
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
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": helper._sha(path),
                    "scientific_status": status,
                }
            )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(summary_path)
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
            "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(summary_path, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray], lock: dict, identity: dict) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("adaptive continuation result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "continuation_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "continuation_arrays.npz", arrays)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_hashes": lock["hashes"],
            "manifest_classification": lock["summary"]["classification"],
            "execution_identity": identity,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "adaptive_metric_chart_continuation_passed": metrics["passed"],
        "new_accepted_segments": metrics["gate_values"]["accepted_segments"],
        "terminal_elapsed_seconds": metrics["gate_values"]["terminal_elapsed_seconds"],
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "implementation_commit": identity["implementation_commit"],
            "implementation_tree": identity["implementation_tree"],
            "source_hashes": identity["source_hashes"],
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Adaptive metric-chart continuation execution",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Accepted `{values['accepted_segments']}` of `{values['attempted_segments']}` attempted segments over `{values['new_accepted_horizon_seconds']:.6f}` s, reaching `{values['terminal_elapsed_seconds']:.6f}` s.",
                "",
                f"Retryable chart failures: `{values['retryable_chart_failures']}`. Accepted spans: `{values['accepted_segment_seconds']}`.",
                "",
                f"Maximum endpoint/blind defects: `{values['maximum_accepted_endpoint_integral_defect']:.6e}` / `{values['maximum_accepted_blind_midpoint_rate_defect']:.6e}`. Maximum raw/metric conditions: `{values['maximum_raw_coordinate_jacobian_condition']:.6e}` / `{values['maximum_metric_coordinate_jacobian_condition']:.6e}`.",
                "",
                f"Checkpoint/suffix replay: `{values['all_accepted_checkpoint_roundtrips_bitwise']}` / `{values['suffix_history_replay_bitwise']}`.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`.",
                "",
            )
        ),
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
