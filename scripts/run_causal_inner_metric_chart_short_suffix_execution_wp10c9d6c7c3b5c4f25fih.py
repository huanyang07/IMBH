#!/usr/bin/env python3
"""Execute four fixed segments with conservative metric-chart reanchoring."""

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

import run_causal_inner_metric_chart_short_suffix_manifest_wp10c9d6c7c3b5c4f25fig as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.conservative_metric_chart_atlas import (  # noqa: E402
    ConservativeMetricChart,
    block_whitening_transform,
    metric_augmented_jacobian,
    metric_transport_retract,
)


parent = manifest.parent
execution = parent.execution
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fih"
PASS_CLASSIFICATION = "metric_chart_short_suffix_passed"
PHYSICAL_FAILURE_CLASSIFICATION = (
    "metric_chart_short_suffix_original_physical_gate_failed"
)
NUMERICAL_FAILURE_CLASSIFICATION = (
    "metric_chart_short_suffix_numerical_or_restart_failed"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fii_metric_chart_wide_continuation_resume_manifest"
)
ARTIFACT = (
    "causal_inner_metric_chart_short_suffix_execution_"
    "wp10c9d6c7c3b5c4f25fih"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_METRIC_CHART_SHORT_SUFFIX_"
    "EXECUTION_WP10C9D6C7C3B5C4F25FIH_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_metric_chart_short_suffix_execution_"
    "wp10c9d6c7c3b5c4f25fih.py"
)
THIS_TEST = (
    "tests/test_causal_inner_metric_chart_short_suffix_execution_"
    "wp10c9d6c7c3b5c4f25fih.py"
)


def _helper():
    return manifest._helper()


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return parent._relative(left, right)


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
    contract = helper._read(manifest.CANONICAL_DIRECTORY / "suffix_contract.json")
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["metric_chart_short_suffix_authorized"]
        or summary["metric_chart_short_suffix_executed"]
        or summary["new_trajectory"]
        or summary["authorized_next"] != manifest.AUTHORIZED_NEXT
        or contract["authorized_execution"] != manifest.AUTHORIZED_NEXT
        or contract["suffix"]["new_segments"] != 4
        or contract["scope"]["new_exact_free_field_calls"] != 5
        or contract["scope"]["new_metric_retractions"] != 5
        or contract["restart"]["checkpoint_after_new_segments"] != 2
    ):
        raise RuntimeError("short-suffix authorization changed")
    for relative, frozen_hash in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != frozen_hash:
            raise RuntimeError(f"frozen suffix source changed: {relative}")
    parent_lock = manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("short-suffix execution requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "parent_lock": parent_lock,
    }


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        parent.THIS_RUNNER,
        parent.ATLAS_SOURCE,
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
            raise RuntimeError("short-suffix scratch identity mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(path, identity)
    return identity


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "suffix_seed.npz")


def _block_sizes() -> tuple[int, ...]:
    return parent._block_sizes()


def _policy():
    policy = parent._retraction_policy()
    if (
        policy.original_coordinate_tolerance
        != manifest.ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
        or policy.metric_coordinate_tolerance
        != manifest.METRIC_COORDINATE_RESIDUAL_TOLERANCE
        or policy.gauge_tolerance != manifest.GAUGE_RESIDUAL_TOLERANCE
        or policy.maximum_metric_augmented_condition
        != manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
    ):
        raise RuntimeError("short-suffix metric policy changed")
    return policy


def _initial_progress() -> dict[str, np.ndarray | float | int]:
    seed = _seed()
    return {
        "previous_coordinate": seed["previous_coordinate470"].copy(),
        "current_coordinate": seed["current_coordinate470"].copy(),
        "previous_state": seed["previous_primitive_state"].copy(),
        "current_state": seed["current_primitive_state"].copy(),
        "previous_rate": seed["previous_coordinate_rate470_per_s"].copy(),
        "current_rate": seed["current_coordinate_rate470_per_s"].copy(),
        "previous_span": float(seed["previous_span_seconds"]),
        "elapsed_seconds": float(seed["elapsed_seconds"]),
        "accepted_segments_total": int(seed["accepted_segments_total"]),
        "metric_transform": seed["current_metric_transform470x470"].copy(),
        "metric_augmented": seed["current_metric_augmented560x560"].copy(),
        "gauge_basis": seed["current_gauge_basis560x90"].copy(),
        "section_normal": seed["section_normal470"].copy(),
        "start_coordinate": seed["start_coordinate470"].copy(),
    }


def _segment_directory(index: int) -> Path:
    return SCRATCH_DIRECTORY / f"segment_{index:02d}"


def _pair(directory: Path, stem: str) -> tuple[Path, Path]:
    return directory / f"{stem}.json", directory / f"{stem}.npz"


def _load_pair(
    directory: Path, stem: str
) -> tuple[dict, dict[str, np.ndarray]] | None:
    metrics_path, arrays_path = _pair(directory, stem)
    if metrics_path.exists() != arrays_path.exists():
        raise RuntimeError(f"partial suffix scratch pair: {directory.name}/{stem}")
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


def _metric_retraction(
    *,
    directory: Path,
    stem: str,
    exact_chart,
    model,
    progress: dict,
    target: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    cached = _load_pair(directory, stem)
    if cached is not None:
        np.testing.assert_array_equal(
            cached[1]["target_original_coordinate470"], target
        )
        print(f"{directory.name}/{stem}: reused retraction", flush=True)
        return cached
    chart = ConservativeMetricChart(
        progress["current_coordinate"],
        progress["metric_transform"],
        _block_sizes(),
    )
    initial = parent.parent._initial_state(
        model,
        progress["current_state"],
        progress["current_coordinate"],
        target,
    )
    state, matrix, metrics = metric_transport_retract(
        exact_chart=exact_chart,
        model=model,
        initial_state=initial,
        target_original_coordinate=target,
        gauge_basis=progress["gauge_basis"],
        anchor_delta=exact_chart._delta(model, progress["current_state"]),
        anchor_metric_augmented=progress["metric_augmented"],
        chart=chart,
        policy=_policy(),
    )
    recovered, factors = model.coordinate(state)
    arrays = {
        "target_original_coordinate470": np.asarray(target),
        "recovered_original_coordinate470": np.asarray(recovered),
        "primitive_state": np.asarray(state),
        "final_metric_broyden560x560": np.asarray(matrix),
        "decoder_reconstruction_factors": np.asarray(factors),
    }
    _write_pair(directory, stem, metrics, arrays)
    print(
        f"{directory.name}/{stem}: original="
        f"{metrics['original_coordinate_residual_infinity']:.3e} "
        f"metric={metrics['metric_coordinate_residual_infinity']:.3e} "
        f"gauge={metrics['gauge_residual_infinity']:.3e} "
        f"passed={metrics['passed']}",
        flush=True,
    )
    return metrics, arrays


def _metric_field(
    *,
    directory: Path,
    stem: str,
    inputs: dict,
    exact_chart,
    state: np.ndarray,
    coordinate: np.ndarray,
    retraction: dict,
    anchor_chart: ConservativeMetricChart,
) -> tuple[dict, dict[str, np.ndarray]]:
    cached = _load_pair(directory, stem)
    if cached is not None:
        np.testing.assert_array_equal(cached[1]["requested_coordinate470"], coordinate)
        print(f"{directory.name}/{stem}: reused exact field", flush=True)
        return cached
    began = time.perf_counter()
    free_metrics, free_arrays = execution.source._free_field(inputs, state)
    free_wall = float(time.perf_counter() - began)
    began = time.perf_counter()
    jacobian, jacobian_metrics = exact_chart._coordinate_jacobian(
        inputs["model"], state
    )
    transform, transform_metrics = block_whitening_transform(
        jacobian, _block_sizes()
    )
    chart = ConservativeMetricChart(coordinate, transform, _block_sizes())
    gauge = exact_chart._canonical_null_basis(jacobian)
    augmented, augmented_condition = metric_augmented_jacobian(
        jacobian, gauge, chart
    )
    independent_wall = float(time.perf_counter() - began)
    transition = transform @ anchor_chart.inverse_transform
    raw_reproduction = abs(
        float(jacobian_metrics["condition_number"])
        - float(free_metrics["coordinate_jacobian_condition_number"])
    ) / max(
        float(jacobian_metrics["condition_number"]),
        float(free_metrics["coordinate_jacobian_condition_number"]),
        np.finfo(float).tiny,
    )
    metric = {
        **transform_metrics,
        "metric_augmented_condition_number": float(augmented_condition),
        "patch_transition_condition_number": float(np.linalg.cond(transition)),
        "transform_inverse_closure_defect": chart.inverse_closure_defect,
        "raw_condition_reproduction_relative_defect": float(raw_reproduction),
        "independent_metric_jacobian_wall_seconds": independent_wall,
    }
    physical_passed = parent._field_physical_passed(
        free_metrics, retraction, metric
    )
    metrics = {
        "free_field": free_metrics,
        "retraction": retraction,
        "metric_chart": metric,
        "physical_passed": physical_passed,
        "exact_free_field_call_wall_seconds": free_wall,
        "total_call_wall_seconds": free_wall + independent_wall,
        "historical_raw_condition_exceeded": bool(
            free_metrics["coordinate_jacobian_condition_number"]
            > parent.manifest.HISTORICAL_RAW_CONDITION
        ),
    }
    arrays = {
        **free_arrays,
        "requested_coordinate470": np.asarray(coordinate),
        "coordinate_jacobian470x560": np.asarray(jacobian),
        "metric_transform470x470": np.asarray(transform),
        "metric_augmented560x560": np.asarray(augmented),
        "gauge_basis560x90": np.asarray(gauge),
    }
    _write_pair(directory, stem, metrics, arrays)
    print(
        f"{directory.name}/{stem}: raw_kappa="
        f"{free_metrics['coordinate_jacobian_condition_number']:.3f} "
        f"metric_kappa={metric['metric_jacobian_condition_number']:.3f} "
        f"wall={metrics['total_call_wall_seconds']:.1f}s "
        f"physical={physical_passed}",
        flush=True,
    )
    return metrics, arrays


def _blind_required(tentative_segment: int) -> bool:
    return int(tentative_segment) == manifest.BLIND_MIDPOINT_SEGMENT


def _attempt_segment(
    *,
    index: int,
    progress: dict,
    inputs: dict,
    exact_chart,
) -> tuple[dict, dict[str, np.ndarray]]:
    directory = _segment_directory(index)
    directory.mkdir(exist_ok=True)
    completed = _load_pair(directory, "segment")
    tentative = int(progress["accepted_segments_total"]) + 1
    if tentative != manifest.TENTATIVE_SEGMENT_NUMBERS[index]:
        raise RuntimeError("suffix tentative segment counter changed")
    candidate = execution._variable_step_ab2(
        progress["current_coordinate"],
        progress["current_rate"],
        progress["previous_rate"],
        manifest.SEGMENT_SECONDS,
        progress["previous_span"],
    )
    if completed is not None:
        np.testing.assert_array_equal(completed[1]["candidate_target470"], candidate)
        print(f"{directory.name}: reused completed segment", flush=True)
        return completed
    anchor_chart = ConservativeMetricChart(
        progress["current_coordinate"],
        progress["metric_transform"],
        _block_sizes(),
    )
    retraction, retraction_arrays = _metric_retraction(
        directory=directory,
        stem="endpoint_retraction",
        exact_chart=exact_chart,
        model=inputs["model"],
        progress=progress,
        target=candidate,
    )
    coordinate = np.asarray(retraction_arrays["recovered_original_coordinate470"])
    state = np.asarray(retraction_arrays["primitive_state"])
    field, field_arrays = _metric_field(
        directory=directory,
        stem="endpoint_field",
        inputs=inputs,
        exact_chart=exact_chart,
        state=state,
        coordinate=coordinate,
        retraction=retraction,
        anchor_chart=anchor_chart,
    )
    rate = np.asarray(field_arrays["coordinate_free_rate470_per_s"])
    endpoint_defect = execution._endpoint_integral_defect(
        progress["current_coordinate"],
        progress["current_rate"],
        coordinate,
        rate,
        manifest.SEGMENT_SECONDS,
    )
    endpoint_passed = bool(
        retraction["passed"]
        and field["physical_passed"]
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
    blind_required = _blind_required(tentative)
    if endpoint_passed and blind_required:
        midpoint_target, midpoint_hermite_rate = execution._hermite(
            progress["current_coordinate"],
            progress["current_rate"],
            coordinate,
            rate,
            manifest.SEGMENT_SECONDS,
            0.5,
        )
        midpoint_retraction, midpoint_retraction_arrays = _metric_retraction(
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
        midpoint_field, midpoint_field_arrays = _metric_field(
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
        blind_defect = _relative(midpoint_hermite_rate, midpoint_rate)
    blind_passed = bool(
        not blind_required
        or (
            midpoint_retraction is not None
            and midpoint_retraction["passed"]
            and midpoint_field is not None
            and midpoint_field["physical_passed"]
            and blind_defect is not None
            and blind_defect <= manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
        )
    )
    accepted = bool(endpoint_passed and blind_passed)
    physical_failure = bool(
        not field["physical_passed"]
        or (midpoint_field is not None and not midpoint_field["physical_passed"])
    )
    metrics = {
        "index": index,
        "tentative_segment_number": tentative,
        "span_seconds": manifest.SEGMENT_SECONDS,
        "blind_midpoint_required": blind_required,
        "endpoint_integral_defect": endpoint_defect,
        "blind_midpoint_rate_defect": blind_defect,
        "endpoint_physical_passed": field["physical_passed"],
        "midpoint_physical_passed": (
            None if midpoint_field is None else midpoint_field["physical_passed"]
        ),
        "physical_failure": physical_failure,
        "accepted": accepted,
        "elapsed_seconds_after": (
            float(progress["elapsed_seconds"]) + manifest.SEGMENT_SECONDS
            if accepted
            else float(progress["elapsed_seconds"])
        ),
        "endpoint_field": field,
        "midpoint_field": midpoint_field,
    }
    arrays = {
        "candidate_target470": candidate,
        "endpoint_coordinate470": coordinate,
        "endpoint_primitive_state": state,
        "endpoint_coordinate_rate470_per_s": rate,
        "endpoint_metric_transform470x470": field_arrays[
            "metric_transform470x470"
        ],
        "endpoint_metric_augmented560x560": field_arrays[
            "metric_augmented560x560"
        ],
        "endpoint_gauge_basis560x90": field_arrays["gauge_basis560x90"],
        "midpoint_target470": midpoint_target,
        "midpoint_hermite_rate470_per_s": midpoint_hermite_rate,
        "midpoint_coordinate470": midpoint_coordinate,
        "midpoint_primitive_state": midpoint_state,
        "midpoint_coordinate_rate470_per_s": midpoint_rate,
    }
    _write_pair(directory, "segment", metrics, arrays)
    print(
        f"{directory.name}: accepted={accepted} "
        f"endpoint={endpoint_defect:.3e} blind={blind_defect}",
        flush=True,
    )
    return metrics, arrays


def _advance(progress: dict, metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if not metrics["accepted"]:
        return progress
    return {
        **progress,
        "previous_coordinate": np.asarray(progress["current_coordinate"]),
        "current_coordinate": np.asarray(arrays["endpoint_coordinate470"]),
        "previous_state": np.asarray(progress["current_state"]),
        "current_state": np.asarray(arrays["endpoint_primitive_state"]),
        "previous_rate": np.asarray(progress["current_rate"]),
        "current_rate": np.asarray(arrays["endpoint_coordinate_rate470_per_s"]),
        "previous_span": float(metrics["span_seconds"]),
        "elapsed_seconds": float(metrics["elapsed_seconds_after"]),
        "accepted_segments_total": int(progress["accepted_segments_total"]) + 1,
        "metric_transform": np.asarray(
            arrays["endpoint_metric_transform470x470"]
        ),
        "metric_augmented": np.asarray(
            arrays["endpoint_metric_augmented560x560"]
        ),
        "gauge_basis": np.asarray(arrays["endpoint_gauge_basis560x90"]),
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
        "elapsed_seconds": np.asarray(progress["elapsed_seconds"]),
        "accepted_segments_total": np.asarray(progress["accepted_segments_total"]),
        "current_metric_transform470x470": np.asarray(progress["metric_transform"]),
        "current_metric_augmented560x560": np.asarray(progress["metric_augmented"]),
        "current_gauge_basis560x90": np.asarray(progress["gauge_basis"]),
        "section_normal470": np.asarray(progress["section_normal"]),
        "start_coordinate470": np.asarray(progress["start_coordinate"]),
    }


def _write_checkpoint(index: int, progress: dict) -> bool:
    path = _segment_directory(index) / "accepted_checkpoint.npz"
    expected = _checkpoint_arrays(progress)
    if not path.exists():
        _save_npz(path, expected)
    loaded = _load_npz(path)
    return bool(
        set(loaded) == set(expected)
        and all(np.array_equal(loaded[name], expected[name]) for name in expected)
    )


def _progress_from_checkpoint(arrays: dict[str, np.ndarray]) -> dict:
    return {
        "previous_coordinate": arrays["previous_coordinate470"].copy(),
        "current_coordinate": arrays["current_coordinate470"].copy(),
        "previous_state": arrays["previous_primitive_state"].copy(),
        "current_state": arrays["current_primitive_state"].copy(),
        "previous_rate": arrays["previous_coordinate_rate470_per_s"].copy(),
        "current_rate": arrays["current_coordinate_rate470_per_s"].copy(),
        "previous_span": float(arrays["previous_span_seconds"]),
        "elapsed_seconds": float(arrays["elapsed_seconds"]),
        "accepted_segments_total": int(arrays["accepted_segments_total"]),
        "metric_transform": arrays["current_metric_transform470x470"].copy(),
        "metric_augmented": arrays["current_metric_augmented560x560"].copy(),
        "gauge_basis": arrays["current_gauge_basis560x90"].copy(),
        "section_normal": arrays["section_normal470"].copy(),
        "start_coordinate": arrays["start_coordinate470"].copy(),
    }


def _suffix_replay(
    records: list[tuple[dict, dict[str, np.ndarray]]],
    final_progress: dict,
) -> tuple[bool, bool]:
    checkpoint_index = manifest.RESTART_AFTER_NEW_SEGMENTS - 1
    path = _segment_directory(checkpoint_index) / "accepted_checkpoint.npz"
    if not path.exists():
        return False, False
    first = _load_npz(path)
    second = _load_npz(path)
    roundtrip = bool(
        set(first) == set(second)
        and all(np.array_equal(first[name], second[name]) for name in first)
    )
    progress = _progress_from_checkpoint(first)
    replay = True
    for metrics, arrays in records[manifest.RESTART_AFTER_NEW_SEGMENTS :]:
        candidate = execution._variable_step_ab2(
            progress["current_coordinate"],
            progress["current_rate"],
            progress["previous_rate"],
            manifest.SEGMENT_SECONDS,
            progress["previous_span"],
        )
        replay = bool(
            replay and np.array_equal(candidate, arrays["candidate_target470"])
        )
        if metrics["blind_midpoint_required"]:
            midpoint_target, midpoint_rate = execution._hermite(
                progress["current_coordinate"],
                progress["current_rate"],
                arrays["endpoint_coordinate470"],
                arrays["endpoint_coordinate_rate470_per_s"],
                manifest.SEGMENT_SECONDS,
                0.5,
            )
            replay = bool(
                replay
                and np.array_equal(midpoint_target, arrays["midpoint_target470"])
                and np.array_equal(
                    midpoint_rate, arrays["midpoint_hermite_rate470_per_s"]
                )
            )
        progress = _advance(progress, metrics, arrays)
    suffix = bool(
        replay
        and np.array_equal(progress["current_coordinate"], final_progress["current_coordinate"])
        and np.array_equal(progress["current_state"], final_progress["current_state"])
        and np.array_equal(progress["current_rate"], final_progress["current_rate"])
        and progress["elapsed_seconds"] == final_progress["elapsed_seconds"]
        and progress["accepted_segments_total"]
        == final_progress["accepted_segments_total"]
    )
    return roundtrip, suffix


def _execute(lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    del identity
    began = time.perf_counter()
    inputs = execution.source._initial_inputs()
    exact_chart = execution.source.arclength._exact_chart()
    progress = _initial_progress()
    records: list[tuple[dict, dict[str, np.ndarray]]] = []
    checkpoint_roundtrips = []
    for index in range(manifest.NEW_SEGMENTS):
        metrics, arrays = _attempt_segment(
            index=index,
            progress=progress,
            inputs=inputs,
            exact_chart=exact_chart,
        )
        records.append((metrics, arrays))
        if not metrics["accepted"]:
            break
        progress = _advance(progress, metrics, arrays)
        checkpoint_roundtrips.append(_write_checkpoint(index, progress))

    accepted_records = [record for record in records if record[0]["accepted"]]
    all_required_accepted = len(accepted_records) == manifest.NEW_SEGMENTS
    restart_roundtrip = False
    suffix_replay = False
    if all_required_accepted:
        restart_roundtrip, suffix_replay = _suffix_replay(records, progress)
    exact_fields = []
    for metrics, _arrays in records:
        exact_fields.append(metrics["endpoint_field"])
        if metrics["midpoint_field"] is not None:
            exact_fields.append(metrics["midpoint_field"])
    endpoint_defects = [record[0]["endpoint_integral_defect"] for record in records]
    blind_defects = [
        record[0]["blind_midpoint_rate_defect"]
        for record in records
        if record[0]["blind_midpoint_rate_defect"] is not None
    ]
    physical_failure = any(record[0]["physical_failure"] for record in records)
    execution_wall = float(time.perf_counter() - began)
    raw_conditions = [
        field["free_field"]["coordinate_jacobian_condition_number"]
        for field in exact_fields
    ]
    metric_conditions = [
        field["metric_chart"]["metric_jacobian_condition_number"]
        for field in exact_fields
    ]
    transition_conditions = [
        field["metric_chart"]["patch_transition_condition_number"]
        for field in exact_fields
    ]
    ledgers = [
        max(
            value
            for name, value in field["free_field"]["reaction_free_ledger_values"].items()
            if name != "incoming_excision_characteristics"
        )
        for field in exact_fields
    ]
    passed = bool(
        all_required_accepted
        and all(checkpoint_roundtrips)
        and restart_roundtrip
        and suffix_replay
        and len(exact_fields) == manifest.MAXIMUM_EXACT_FREE_FIELD_CALLS
        and all(field["physical_passed"] for field in exact_fields)
        and max(endpoint_defects, default=float("inf"))
        <= manifest.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
        and len(blind_defects) == 1
        and max(blind_defects, default=float("inf"))
        <= manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
        and max(metric_conditions, default=float("inf"))
        <= manifest.MAXIMUM_METRIC_JACOBIAN_CONDITION
        and max(transition_conditions, default=float("inf"))
        <= manifest.MAXIMUM_PATCH_TRANSITION_CONDITION
        and execution_wall <= 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
    )
    if passed:
        classification = PASS_CLASSIFICATION
    elif physical_failure:
        classification = PHYSICAL_FAILURE_CLASSIFICATION
    else:
        classification = NUMERICAL_FAILURE_CLASSIFICATION
    section_values = [
        float(
            progress["section_normal"]
            @ (record[1]["endpoint_coordinate470"] - progress["start_coordinate"])
        )
        for record in accepted_records
    ]
    gates = {
        "initial_elapsed_seconds": manifest.INITIAL_ELAPSED_SECONDS,
        "terminal_elapsed_seconds": float(progress["elapsed_seconds"]),
        "new_accepted_segments": len(accepted_records),
        "new_accepted_horizon_seconds": (
            len(accepted_records) * manifest.SEGMENT_SECONDS
        ),
        "exact_free_field_calls": len(exact_fields),
        "independent_metric_jacobian_audits": len(exact_fields),
        "metric_retractions": len(records) + len(blind_defects),
        "all_segment_checkpoint_roundtrips_bitwise": bool(
            checkpoint_roundtrips and all(checkpoint_roundtrips)
        ),
        "restart_checkpoint_roundtrip_bitwise": restart_roundtrip,
        "remaining_suffix_history_replay_bitwise": suffix_replay,
        "maximum_endpoint_integral_defect": max(endpoint_defects, default=float("inf")),
        "maximum_blind_midpoint_rate_defect": max(blind_defects, default=float("inf")),
        "maximum_raw_coordinate_jacobian_condition": max(raw_conditions, default=float("inf")),
        "minimum_raw_coordinate_jacobian_condition": min(raw_conditions, default=float("inf")),
        "maximum_metric_coordinate_jacobian_condition": max(metric_conditions, default=float("inf")),
        "maximum_patch_transition_condition": max(transition_conditions, default=float("inf")),
        "all_exact_fields_physical_passed": bool(
            exact_fields and all(field["physical_passed"] for field in exact_fields)
        ),
        "minimum_reconstruction_factor": min(
            (field["free_field"]["minimum_reconstruction_factor"] for field in exact_fields),
            default=0.0,
        ),
        "maximum_height_ratio": max(
            (field["free_field"]["maximum_height_ratio"] for field in exact_fields),
            default=float("inf"),
        ),
        "minimum_scattering_optical_depth": min(
            (field["free_field"]["minimum_scattering_optical_depth"] for field in exact_fields),
            default=0.0,
        ),
        "maximum_reaction_free_ledger_defect": max(ledgers, default=float("inf")),
        "minimum_section_value": min(section_values, default=float("inf")),
        "terminal_section_value": (
            section_values[-1] if section_values else float("inf")
        ),
        "execution_wall_seconds": execution_wall,
        "maximum_execution_wall_seconds": 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS,
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "gate_values": gates,
        "segments": [record[0] for record in records],
        "input_lock": {
            "manifest_hashes": lock["hashes"],
            "manifest_classification": lock["summary"]["classification"],
        },
        "fixed_Q_physical_rate_calls": 0,
        "fixed_Q_reaction_calls": 0,
        "nonlinear_roots": 0,
        "BDF_microsteps": 0,
    }
    terminal = _checkpoint_arrays(progress)
    arrays = {
        **terminal,
        "accepted_endpoint_coordinates470": (
            np.stack([record[1]["endpoint_coordinate470"] for record in accepted_records])
            if accepted_records else np.empty((0, 470))
        ),
        "accepted_endpoint_primitive_states": (
            np.stack([record[1]["endpoint_primitive_state"] for record in accepted_records])
            if accepted_records else np.empty((0, 112, 5))
        ),
        "accepted_endpoint_rates470_per_s": (
            np.stack([record[1]["endpoint_coordinate_rate470_per_s"] for record in accepted_records])
            if accepted_records else np.empty((0, 470))
        ),
        "endpoint_integral_defects": np.asarray(endpoint_defects),
        "blind_midpoint_rate_defects": np.asarray(blind_defects),
        "section_values": np.asarray(section_values),
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
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": status,
            })
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
        raise RuntimeError("short-suffix result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "suffix_execution_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "suffix_execution_arrays.npz", arrays)
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
        "metric_chart_short_suffix_passed": bool(metrics["passed"]),
        "new_accepted_segments": metrics["gate_values"]["new_accepted_segments"],
        "wide_resume_manifest_authorized": bool(metrics["passed"]),
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
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
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Metric-chart short-suffix execution",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"Accepted `{values['new_accepted_segments']}` of four prospective 0.25 ms segments, reaching `{values['terminal_elapsed_seconds']:.9f}` s. Maximum endpoint and blind defects were `{values['maximum_endpoint_integral_defect']:.6e}` and `{values['maximum_blind_midpoint_rate_defect']:.6e}`.",
            "",
            f"Maximum raw and metric Jacobian conditions were `{values['maximum_raw_coordinate_jacobian_condition']:.6e}` and `{values['maximum_metric_coordinate_jacobian_condition']:.6e}`. Checkpoint roundtrip and remaining-suffix replay were `{values['restart_checkpoint_roundtrip_bitwise']}` and `{values['remaining_suffix_history_replay_bitwise']}`.",
            "",
            "Every endpoint used a new exact local metric patch while physics and ledgers remained in the original representation.",
            "",
            f"Authorized next artifact: `{AUTHORIZED_NEXT if metrics['passed'] else None}`.",
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
