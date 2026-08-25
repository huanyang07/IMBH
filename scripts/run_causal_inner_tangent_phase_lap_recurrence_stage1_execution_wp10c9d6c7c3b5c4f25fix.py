#!/usr/bin/env python3
"""Execute stage 1 of tangent-phase-lap and recurrence acquisition."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imri_qpe.layer3_minidisk_1d.tangent_phase_atlas import (  # noqa: E402
    fit_tangent_phase_chart,
    normalized_metric_tangents,
)
import run_causal_inner_adaptive_metric_chart_continuation_execution_wp10c9d6c7c3b5c4f25fip as engine  # noqa: E402
import run_causal_inner_conservative_tangent_phase_atlas_holdout_execution_wp10c9d6c7c3b5c4f25fiv as holdout  # noqa: E402
import run_causal_inner_tangent_phase_lap_recurrence_manifest_wp10c9d6c7c3b5c4f25fiw as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "tangent_phase_lap_acquisition_stage1_passed_no_lap"
COARSE_RECURRENCE_CLASSIFICATION = "coarse_tangent_phase_recurrence_candidate_observed"
OPEN_CLASSIFICATION = "tangent_phase_lap_without_coarse_state_recurrence"
PHASE_FAILURE_CLASSIFICATION = "tangent_phase_lap_stage1_geometry_failed"
PHYSICAL_FAILURE_CLASSIFICATION = "tangent_phase_lap_stage1_physical_gate_failed"
NUMERICAL_FAILURE_CLASSIFICATION = "tangent_phase_lap_stage1_numerical_or_restart_failed"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fiy_"
    "tangent_phase_lap_recurrence_stage2_manifest"
)
COARSE_RECURRENCE_AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fiy_"
    "registered_tangent_phase_return_refinement_manifest"
)
ARTIFACT = (
    "causal_inner_tangent_phase_lap_recurrence_stage1_execution_"
    "wp10c9d6c7c3b5c4f25fix"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TANGENT_PHASE_LAP_RECURRENCE_"
    "STAGE1_EXECUTION_WP10C9D6C7C3B5C4F25FIX_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_tangent_phase_lap_recurrence_stage1_execution_"
    "wp10c9d6c7c3b5c4f25fix.py"
)
THIS_TEST = (
    "tests/test_causal_inner_tangent_phase_lap_recurrence_stage1_execution_"
    "wp10c9d6c7c3b5c4f25fix.py"
)

# Adapter constants consumed by the certified continuation engine.
_LEGACY = holdout._LEGACY
_CONTRACT_FILE = manifest.CANONICAL_DIRECTORY / "acquisition_contract.json"
INITIAL_ELAPSED_SECONDS = manifest.INITIAL_ELAPSED_SECONDS
MINIMUM_SEGMENT_SECONDS = 1.25e-4
MAXIMUM_SEGMENT_SECONDS = manifest.SEGMENT_SECONDS
GROWTH_FACTOR = 2.0
ACCEPTED_SEGMENTS_BEFORE_GROWTH = 4
BLIND_MIDPOINT_FREQUENCY = manifest.BLIND_MIDPOINT_FREQUENCY
MAXIMUM_ACCEPTED_SEGMENTS = manifest.STAGE_ACCEPTED_SEGMENTS
MAXIMUM_ATTEMPTED_SEGMENTS = manifest.MAXIMUM_STAGE_ATTEMPTS
MAXIMUM_EXACT_FREE_FIELD_CALLS = manifest.STAGE_EXACT_FIELD_BUDGET
MAXIMUM_RETRACTIONS = manifest.STAGE_RETRACTION_BUDGET
MAXIMUM_EXECUTION_WALL_HOURS = manifest.STAGE_WALL_HOURS
MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = _LEGACY.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT = _LEGACY.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT


_BASE_HELPER_MODULE = manifest._helper()
_ORIGINAL_ENGINE_ATTEMPT = engine._attempt


def _helper():
    return manifest._helper()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _contract() -> dict:
    return _helper()._read(_CONTRACT_FILE)


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _contract()
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    stage = contract["staged_scope"]["stage1"]
    gates = contract["binding_stage1_gates"]
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["stage1_execution_authorized"]
        or summary["stage1_execution_executed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or contract["authorized_execution"] != WORK_PACKAGE
        or not contract["staged_scope"]["only_stage1_is_authorized_now"]
        or stage["accepted_segments"] != MAXIMUM_ACCEPTED_SEGMENTS
        or stage["maximum_attempted_segments"] != MAXIMUM_ATTEMPTED_SEGMENTS
        or stage["maximum_exact_free_field_calls"]
        != MAXIMUM_EXACT_FREE_FIELD_CALLS
        or stage["maximum_retractions"] != MAXIMUM_RETRACTIONS
        or stage["maximum_wall_hours"] != MAXIMUM_EXECUTION_WALL_HOURS
        or gates["minimum_cumulative_phase_advance_radians"]
        != manifest.MINIMUM_STAGE_PHASE_ADVANCE
        or gates["maximum_direction_prediction_defect_radians"] != 0.005
    ):
        raise RuntimeError("phase-lap stage1 authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"phase-lap manifest source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("phase-lap stage1 requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "provenance": provenance,
    }


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "continuation_seed.npz")


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
        "metric_transform": seed["metric_transform470x470"].copy(),
        "metric_augmented": seed["metric_augmented560x560"].copy(),
        "gauge_basis": seed["gauge_basis560x90"].copy(),
        "section_normal": seed["section_normal470"].copy(),
        "start_coordinate": seed["start_coordinate470"].copy(),
        "stop_reason": None,
    }


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        holdout.THIS_RUNNER,
        holdout.THIS_TEST,
        holdout.manifest.PHASE_SOURCE,
        holdout.manifest.PHASE_TEST,
        engine.THIS_RUNNER,
        engine.THIS_TEST,
        engine.suffix.THIS_RUNNER,
        engine.execution.source.THIS_RUNNER,
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
        "contract": lock["contract"],
    }


def _prepare_scratch(lock: dict) -> dict:
    helper = _helper()
    identity = _identity(lock)
    path = SCRATCH_DIRECTORY / "execution_identity.json"
    if SCRATCH_DIRECTORY.exists():
        if not path.exists() or helper._read(path) != identity:
            raise RuntimeError("phase-lap stage1 scratch mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(path, identity)
    return identity


def _accepted_attempts() -> list[tuple[dict, dict[str, np.ndarray]]]:
    accepted = []
    if not SCRATCH_DIRECTORY.exists():
        return accepted
    for directory in sorted(SCRATCH_DIRECTORY.glob("attempt_[0-9][0-9][0-9][0-9]")):
        metrics_path = directory / "attempt.json"
        arrays_path = directory / "attempt.npz"
        if not metrics_path.exists() or not arrays_path.exists():
            continue
        metrics = _helper()._read(metrics_path)
        if metrics.get("accepted"):
            accepted.append((metrics, _load_npz(arrays_path)))
    return accepted


def _accepted_phase_history() -> np.ndarray:
    training = _seed()["phase_training_raw_rates470_per_s"].copy()
    accepted = [
        arrays["accepted_coordinate_rate470_per_s"]
        for _metrics, arrays in _accepted_attempts()
    ]
    if accepted:
        training = np.vstack((training, np.stack(accepted)))
    return training[-holdout.manifest.SELECTED_WINDOW :]


def _prior_accumulation() -> dict:
    phase = 0.0
    path_length = 0.0
    section_value = 0.0
    for metrics, _arrays in _accepted_attempts():
        phase = float(metrics["recurrence_geometry"]["cumulative_phase_advance_radians"])
        path_length = float(
            metrics["recurrence_geometry"]["cumulative_metric_path_length"]
        )
        section_value = float(
            metrics["recurrence_geometry"]["endpoint_registered_section_value"]
        )
    return {
        "cumulative_phase_advance_radians": phase,
        "cumulative_metric_path_length": path_length,
        "registered_section_value": section_value,
    }


def _predicted_unit_tangent(chart, phase_increment: float) -> np.ndarray:
    return holdout._predicted_unit_tangent(chart, phase_increment)


def _prediction(progress: dict) -> tuple[dict, dict[str, np.ndarray], object]:
    history = _accepted_phase_history()
    transform = _seed()["phase_observer_metric_transform470x470"]
    unit = normalized_metric_tangents(history, transform)
    chart = fit_tangent_phase_chart(
        unit,
        predictor_increment_count=holdout.manifest.PREDICTOR_INCREMENT_COUNT,
    )
    span_ratio = float(progress["next_span"] / progress["previous_span"])
    increment = float(chart.predicted_phase_increment * span_ratio)
    predicted = _predicted_unit_tangent(chart, increment)
    prior = _prior_accumulation()
    metrics = {
        "attempt_index": int(progress["attempts"]),
        "tentative_segment_number": int(progress["accepted_segments_total"] + 1),
        "history_samples": int(len(history)),
        "span_seconds": float(progress["next_span"]),
        "previous_span_seconds": float(progress["previous_span"]),
        "span_ratio": span_ratio,
        "predicted_phase_increment": increment,
        "prior_cumulative_phase_advance_radians": prior[
            "cumulative_phase_advance_radians"
        ],
        "training_two_plane_energy_fraction": chart.two_plane_energy_fraction,
        "training_relative_radial_rms": chart.training_relative_radial_rms,
        "circle_solve_condition_number": chart.circle_solve_condition_number,
        "frozen_before_exact_endpoint": True,
    }
    arrays = {
        "training_raw_rates470_per_s": history,
        "training_unit_tangents470": unit,
        "mean_tangent470": chart.mean_tangent,
        "plane_basis470x2": chart.plane_basis,
        "circle_center2": chart.circle_center,
        "circle_radius": np.asarray(chart.circle_radius),
        "orientation_sign": np.asarray(chart.orientation_sign),
        "oriented_angle_origin": np.asarray(chart.oriented_angle_origin),
        "training_phases": chart.training_phases,
        "predicted_unit_tangent470": predicted,
        "phase_observer_metric_transform470x470": transform,
    }
    return metrics, arrays, chart


def _phase_gate(
    prediction: dict,
    prediction_arrays: dict[str, np.ndarray],
    chart,
    exact_raw_rate: np.ndarray,
) -> dict:
    transform = prediction_arrays["phase_observer_metric_transform470x470"]
    exact = normalized_metric_tangents(exact_raw_rate[None, :], transform)[0]
    geometry = chart.evaluate(exact)
    direction = float(
        np.arccos(
            np.clip(
                prediction_arrays["predicted_unit_tangent470"] @ exact,
                -1.0,
                1.0,
            )
        )
    )
    gates = _contract()["binding_stage1_gates"]
    passed = bool(
        geometry["phase_increment"]
        > gates["minimum_local_phase_increment_strictly_greater_than"]
        and geometry["phase_increment"] <= gates["maximum_local_phase_increment"]
        and prediction["training_two_plane_energy_fraction"]
        >= gates["minimum_training_two_plane_energy_fraction"]
        and prediction["training_relative_radial_rms"]
        <= gates["maximum_training_relative_radial_rms"]
        and geometry["relative_radial_defect"]
        <= gates["maximum_holdout_relative_radial_defect"]
        and geometry["out_of_plane_defect"]
        <= gates["maximum_holdout_out_of_plane_defect"]
        and direction <= gates["maximum_direction_prediction_defect_radians"]
    )
    return {
        **geometry,
        "direction_prediction_defect_radians": direction,
        "predicted_phase_increment": prediction["predicted_phase_increment"],
        "passed": passed,
    }


def _recurrence_gate(
    *,
    progress: dict,
    endpoint_coordinate: np.ndarray,
    endpoint_rate: np.ndarray,
    phase_geometry: dict,
) -> dict:
    seed = _seed()
    prior = _prior_accumulation()
    transform = seed["phase_observer_metric_transform470x470"]
    reference = seed["phase_lap_reference_coordinate470"]
    reference_tangent = seed["phase_lap_reference_unit_tangent470"]
    covector = seed["registered_section_covector470"]
    reference_speed = float(seed["reference_metric_speed_per_s"])
    step = np.asarray(endpoint_coordinate) - progress["current_coordinate"]
    step_length = float(np.linalg.norm(transform @ step))
    cumulative_path = float(prior["cumulative_metric_path_length"] + step_length)
    cumulative_phase = float(
        prior["cumulative_phase_advance_radians"]
        + phase_geometry["phase_increment"]
    )
    previous_section = float(covector @ (progress["current_coordinate"] - reference))
    endpoint_section = float(covector @ (endpoint_coordinate - reference))
    section_derivative_fraction = float(
        (endpoint_section - previous_section)
        / (progress["next_span"] * reference_speed)
    )
    endpoint_tangent = normalized_metric_tangents(
        np.asarray(endpoint_rate)[None, :], transform
    )[0]
    tangent_cosine = float(reference_tangent @ endpoint_tangent)
    return_distance = float(np.linalg.norm(transform @ (endpoint_coordinate - reference)))
    return_ratio = float(return_distance / max(cumulative_path, np.finfo(float).tiny))
    section_bracket = bool(previous_section < 0.0 <= endpoint_section)
    crossing_fraction = None
    crossing_phase = None
    crossing_return_ratio = None
    if section_bracket:
        crossing_fraction = float(
            -previous_section / (endpoint_section - previous_section)
        )
        crossing_coordinate = (
            progress["current_coordinate"] + crossing_fraction * step
        )
        crossing_phase = float(
            prior["cumulative_phase_advance_radians"]
            + crossing_fraction * phase_geometry["phase_increment"]
        )
        crossing_path = float(
            prior["cumulative_metric_path_length"]
            + crossing_fraction * step_length
        )
        crossing_distance = float(
            np.linalg.norm(transform @ (crossing_coordinate - reference))
        )
        crossing_return_ratio = float(
            crossing_distance / max(crossing_path, np.finfo(float).tiny)
        )
    gates = _contract()["coarse_recurrence_candidate_requires"]
    candidate = bool(
        section_bracket
        and abs(crossing_phase - manifest.PHASE_LAP_RADIANS)
        <= gates["unwrapped_phase_at_crossing_within_radians_of_2pi"]
        and crossing_return_ratio
        <= gates[
            "maximum_metric_state_return_distance_over_accumulated_path_length"
        ]
        and tangent_cosine >= gates["minimum_metric_tangent_cosine"]
        and section_derivative_fraction
        >= gates[
            "minimum_positive_section_derivative_fraction_of_reference_speed"
        ]
    )
    phase_lap = bool(cumulative_phase >= manifest.PHASE_LAP_RADIANS)
    return {
        "step_metric_path_length": step_length,
        "cumulative_metric_path_length": cumulative_path,
        "cumulative_phase_advance_radians": cumulative_phase,
        "phase_lap_observed": phase_lap,
        "phase_lap_crossed_this_step": bool(
            prior["cumulative_phase_advance_radians"]
            < manifest.PHASE_LAP_RADIANS
            <= cumulative_phase
        ),
        "previous_registered_section_value": previous_section,
        "endpoint_registered_section_value": endpoint_section,
        "section_derivative_fraction_of_reference_speed": (
            section_derivative_fraction
        ),
        "registered_section_bracket": section_bracket,
        "crossing_fraction": crossing_fraction,
        "crossing_phase_advance_radians": crossing_phase,
        "endpoint_metric_return_distance": return_distance,
        "endpoint_return_distance_over_path_length": return_ratio,
        "crossing_return_distance_over_path_length": crossing_return_ratio,
        "endpoint_metric_tangent_cosine": tangent_cosine,
        "coarse_recurrence_candidate": candidate,
    }


def _phase_attempt(*, progress: dict, inputs: dict, exact_chart):
    helper = _helper()
    directory = engine._attempt_directory(int(progress["attempts"]))
    directory.mkdir(exist_ok=True)
    prediction_path = directory / "phase_prediction.json"
    prediction_arrays_path = directory / "phase_prediction.npz"
    if prediction_path.exists() or prediction_arrays_path.exists():
        if not prediction_path.exists() or not prediction_arrays_path.exists():
            raise RuntimeError("incomplete frozen phase prediction")
        prediction = helper._read(prediction_path)
        prediction_arrays = _load_npz(prediction_arrays_path)
        new_prediction, new_arrays, chart = _prediction(progress)
        if prediction != new_prediction:
            raise RuntimeError("frozen phase prediction metadata changed")
        for name, value in new_arrays.items():
            np.testing.assert_array_equal(prediction_arrays[name], value)
    else:
        prediction, prediction_arrays, chart = _prediction(progress)
        helper._write_json(prediction_path, prediction)
        _save_npz(prediction_arrays_path, prediction_arrays)
    metrics, arrays = _ORIGINAL_ENGINE_ATTEMPT(
        progress=progress,
        inputs=inputs,
        exact_chart=exact_chart,
    )
    if "phase_geometry" in metrics:
        return metrics, arrays
    if metrics["accepted"]:
        phase = _phase_gate(
            prediction,
            prediction_arrays,
            chart,
            arrays["endpoint_coordinate_rate470_per_s"],
        )
        metrics["phase_geometry"] = phase
        if phase["passed"]:
            recurrence = _recurrence_gate(
                progress=progress,
                endpoint_coordinate=arrays["endpoint_coordinate470"],
                endpoint_rate=arrays["endpoint_coordinate_rate470_per_s"],
                phase_geometry=phase,
            )
            metrics["recurrence_geometry"] = recurrence
            if recurrence["coarse_recurrence_candidate"]:
                metrics["stop_reason"] = "coarse_recurrence_candidate"
            elif recurrence["phase_lap_observed"]:
                metrics["stop_reason"] = "phase_lap_without_coarse_recurrence"
        else:
            metrics.update(
                {
                    "accepted": False,
                    "numerical_passed": False,
                    "stop_reason": "phase_geometry",
                    "recurrence_geometry": None,
                }
            )
            arrays.update(
                {
                    "accepted_coordinate470": progress["current_coordinate"],
                    "accepted_primitive_state": progress["current_state"],
                    "accepted_coordinate_rate470_per_s": progress["current_rate"],
                    "accepted_metric_transform470x470": progress["metric_transform"],
                    "accepted_metric_augmented560x560": progress[
                        "metric_augmented"
                    ],
                    "accepted_gauge_basis560x90": progress["gauge_basis"],
                }
            )
    else:
        metrics["phase_geometry"] = None
        metrics["recurrence_geometry"] = None
    helper._write_json(directory / "attempt.json", metrics)
    _save_npz(directory / "attempt.npz", arrays)
    return metrics, arrays


_ENGINE_NAMES = (
    "manifest",
    "WORK_PACKAGE",
    "PASS_CLASSIFICATION",
    "PHYSICAL_FAILURE_CLASSIFICATION",
    "NUMERICAL_FAILURE_CLASSIFICATION",
    "AUTHORIZED_NEXT",
    "SCRATCH_DIRECTORY",
    "_initial_progress",
    "_helper",
    "_attempt",
)


def _stable_engine_helper():
    return _BASE_HELPER_MODULE


@contextmanager
def _engine_context():
    saved = {name: getattr(engine, name) for name in _ENGINE_NAMES}
    replacements = {
        "manifest": sys.modules[__name__],
        "WORK_PACKAGE": WORK_PACKAGE,
        "PASS_CLASSIFICATION": PASS_CLASSIFICATION,
        "PHYSICAL_FAILURE_CLASSIFICATION": PHYSICAL_FAILURE_CLASSIFICATION,
        "NUMERICAL_FAILURE_CLASSIFICATION": NUMERICAL_FAILURE_CLASSIFICATION,
        "AUTHORIZED_NEXT": AUTHORIZED_NEXT,
        "SCRATCH_DIRECTORY": SCRATCH_DIRECTORY,
        "_initial_progress": _initial_progress,
        "_helper": _stable_engine_helper,
        "_attempt": _phase_attempt,
    }
    try:
        for name, value in replacements.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(engine, name, value)


def _phase_records() -> list[dict]:
    records = []
    if not SCRATCH_DIRECTORY.exists():
        return records
    for directory in sorted(SCRATCH_DIRECTORY.glob("attempt_[0-9][0-9][0-9][0-9]")):
        path = directory / "attempt.json"
        if path.exists():
            records.append(_helper()._read(path))
    return records


def _classify(
    metrics: dict,
    arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    result_metrics = dict(metrics)
    result_arrays = dict(arrays)
    records = _phase_records()
    phase_records = [
        item for item in records if item.get("phase_geometry") is not None
    ]
    accepted = [
        item
        for item in records
        if item.get("accepted") and item.get("recurrence_geometry") is not None
    ]
    phase_failed = any(not item["phase_geometry"]["passed"] for item in phase_records)
    recurrence = [item["recurrence_geometry"] for item in accepted]
    final_recurrence = recurrence[-1] if recurrence else None
    cumulative_phase = (
        final_recurrence["cumulative_phase_advance_radians"]
        if final_recurrence is not None
        else 0.0
    )
    cumulative_path = (
        final_recurrence["cumulative_metric_path_length"]
        if final_recurrence is not None
        else 0.0
    )
    phase_lap = any(item["phase_lap_observed"] for item in recurrence)
    candidate = any(item["coarse_recurrence_candidate"] for item in recurrence)
    stage_phase_pass = bool(
        len(accepted) == MAXIMUM_ACCEPTED_SEGMENTS
        and cumulative_phase >= manifest.MINIMUM_STAGE_PHASE_ADVANCE
        and not phase_failed
    )
    values = dict(metrics["gate_values"])
    values.update(
        {
            "phase_predictions_evaluated": len(phase_records),
            "accepted_phase_endpoints": len(accepted),
            "all_phase_geometry_gates_passed": bool(
                phase_records
                and all(item["phase_geometry"]["passed"] for item in phase_records)
            ),
            "minimum_phase_increment": min(
                (item["phase_geometry"]["phase_increment"] for item in accepted),
                default=None,
            ),
            "maximum_phase_increment": max(
                (item["phase_geometry"]["phase_increment"] for item in accepted),
                default=None,
            ),
            "maximum_phase_radial_defect": max(
                (
                    item["phase_geometry"]["relative_radial_defect"]
                    for item in accepted
                ),
                default=None,
            ),
            "maximum_phase_out_of_plane_defect": max(
                (
                    item["phase_geometry"]["out_of_plane_defect"]
                    for item in accepted
                ),
                default=None,
            ),
            "maximum_phase_direction_prediction_defect_radians": max(
                (
                    item["phase_geometry"][
                        "direction_prediction_defect_radians"
                    ]
                    for item in accepted
                ),
                default=None,
            ),
            "cumulative_phase_advance_radians": cumulative_phase,
            "cumulative_metric_path_length": cumulative_path,
            "phase_lap_observed": phase_lap,
            "coarse_recurrence_candidate_observed": candidate,
            "minimum_registered_section_value": min(
                (
                    item["endpoint_registered_section_value"]
                    for item in recurrence
                ),
                default=None,
            ),
            "terminal_registered_section_value": (
                None
                if final_recurrence is None
                else final_recurrence["endpoint_registered_section_value"]
            ),
            "minimum_endpoint_tangent_cosine": min(
                (item["endpoint_metric_tangent_cosine"] for item in recurrence),
                default=None,
            ),
            "terminal_return_distance_over_path_length": (
                None
                if final_recurrence is None
                else final_recurrence[
                    "endpoint_return_distance_over_path_length"
                ]
            ),
        }
    )
    result_metrics["gate_values"] = values
    if candidate:
        result_metrics.update(
            {
                "classification": COARSE_RECURRENCE_CLASSIFICATION,
                "passed": True,
                "authorized_next": COARSE_RECURRENCE_AUTHORIZED_NEXT,
            }
        )
    elif phase_lap:
        result_metrics.update(
            {
                "classification": OPEN_CLASSIFICATION,
                "passed": False,
                "authorized_next": None,
            }
        )
    elif metrics["passed"] and stage_phase_pass:
        result_metrics.update(
            {
                "classification": PASS_CLASSIFICATION,
                "passed": True,
                "authorized_next": AUTHORIZED_NEXT,
            }
        )
    elif phase_failed:
        result_metrics.update(
            {
                "classification": PHASE_FAILURE_CLASSIFICATION,
                "passed": False,
                "authorized_next": None,
            }
        )
    phase_increments = np.asarray(
        [item["phase_geometry"]["phase_increment"] for item in accepted]
    )
    result_arrays.update(
        {
            "accepted_phase_increments": phase_increments,
            "accepted_predicted_phase_increments": np.asarray(
                [
                    item["phase_geometry"]["predicted_phase_increment"]
                    for item in accepted
                ]
            ),
            "accepted_phase_radial_defects": np.asarray(
                [
                    item["phase_geometry"]["relative_radial_defect"]
                    for item in accepted
                ]
            ),
            "accepted_phase_out_of_plane_defects": np.asarray(
                [
                    item["phase_geometry"]["out_of_plane_defect"]
                    for item in accepted
                ]
            ),
            "accepted_phase_direction_prediction_defects_radians": np.asarray(
                [
                    item["phase_geometry"][
                        "direction_prediction_defect_radians"
                    ]
                    for item in accepted
                ]
            ),
            "accepted_cumulative_phase_advance_radians": np.cumsum(
                phase_increments
            ),
            "accepted_cumulative_metric_path_lengths": np.asarray(
                [item["cumulative_metric_path_length"] for item in recurrence]
            ),
            "accepted_registered_section_values": np.asarray(
                [item["endpoint_registered_section_value"] for item in recurrence]
            ),
            "accepted_return_distance_over_path_lengths": np.asarray(
                [
                    item["endpoint_return_distance_over_path_length"]
                    for item in recurrence
                ]
            ),
            "accepted_metric_tangent_cosines": np.asarray(
                [item["endpoint_metric_tangent_cosine"] for item in recurrence]
            ),
            "accepted_section_derivative_fractions": np.asarray(
                [
                    item["section_derivative_fraction_of_reference_speed"]
                    for item in recurrence
                ]
            ),
            "accepted_endpoint_metric_return_distances": np.asarray(
                [item["endpoint_metric_return_distance"] for item in recurrence]
            ),
            "accepted_phase_lap_flags": np.asarray(
                [item["phase_lap_observed"] for item in recurrence], dtype=bool
            ),
            "accepted_coarse_recurrence_candidate_flags": np.asarray(
                [item["coarse_recurrence_candidate"] for item in recurrence],
                dtype=bool,
            ),
        }
    )
    return result_metrics, result_arrays


def _execute(lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    with _engine_context():
        metrics, arrays = engine._execute(lock, identity)
    return _classify(metrics, arrays)


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": helper._sha(path),
                    "scientific_status": (
                        "SUPPORTED" if summary["passed"] else "REJECTED"
                    ),
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


def _canonicalize(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    lock: dict,
    identity: dict,
) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("phase-lap stage1 result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "stage1_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "stage1_arrays.npz", arrays)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_hashes": lock["hashes"],
            "manifest_classification": lock["summary"]["classification"],
            "execution_identity": identity,
        },
    )
    values = metrics["gate_values"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "accepted_segments": values["accepted_segments"],
        "terminal_elapsed_seconds": values["terminal_elapsed_seconds"],
        "cumulative_phase_advance_radians": values[
            "cumulative_phase_advance_radians"
        ],
        "phase_lap_observed": values["phase_lap_observed"],
        "coarse_recurrence_candidate_observed": values[
            "coarse_recurrence_candidate_observed"
        ],
        "complete_cycle_execution_authorized": False,
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
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Tangent-phase-lap recurrence acquisition: stage 1",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Accepted `{values['accepted_segments']}` of `{values['attempted_segments']}` attempts and advanced from `{INITIAL_ELAPSED_SECONDS:.6f}` s to `{values['terminal_elapsed_seconds']:.6f}` s.",
                "",
                f"Cumulative prospectively registered tangent phase: `{values['cumulative_phase_advance_radians']:.9f}` rad. Phase lap observed: `{values['phase_lap_observed']}`. Coarse recurrence candidate: `{values['coarse_recurrence_candidate_observed']}`.",
                "",
                f"Phase increment range: `{values['minimum_phase_increment']}` to `{values['maximum_phase_increment']}`. Maximum radial, out-of-plane, and direction defects: `{values['maximum_phase_radial_defect']}`, `{values['maximum_phase_out_of_plane_defect']}`, and `{values['maximum_phase_direction_prediction_defect_radians']}` rad.",
                "",
                f"Terminal registered-section value: `{values['terminal_registered_section_value']}`. Terminal return distance/path length: `{values['terminal_return_distance_over_path_length']}`. Minimum tangent cosine: `{values['minimum_endpoint_tangent_cosine']}`.",
                "",
                f"Exact fields/retractions/wall seconds: `{values['exact_free_field_calls']}` / `{values['retractions']}` / `{values['execution_wall_seconds']:.3f}`. Checkpoint and suffix replay remained bitwise.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`. A phase lap is not a cycle; complete-cycle execution and reduced slow evolution remain unauthorized.",
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
