#!/usr/bin/env python3
"""Execute the bounded autonomous endpoint/collocation continuation."""

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

import run_causal_inner_curvature_adaptive_arclength_continuation_manifest_wp10c9d6c7c3b5c4f25fh as manifest  # noqa: E402


source = manifest.diagnosis.parent
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fi"
PASS_CLASSIFICATION = "wide_continuation_cycle_observed_local_transport_passed"
EQUILIBRIUM_CLASSIFICATION = (
    "wide_continuation_equilibrium_candidate_requires_certificate"
)
BUDGET_CLASSIFICATION = (
    "wide_continuation_inconclusive_acquisition_budget_exhausted"
)
PHYSICAL_CLASSIFICATION = (
    "wide_continuation_original_free_field_physical_gate_failed"
)
VALIDATION_CLASSIFICATION = (
    "wide_continuation_endpoint_or_blind_validation_failed"
)
RESTART_CLASSIFICATION = "wide_continuation_restart_or_replay_failed"
AUTHORIZED_CYCLE_NEXT = (
    "WP10c9d6c7c3b5c4f25fj_matched_path_global_cycle_map_manifest"
)
AUTHORIZED_EQUILIBRIUM_NEXT = (
    "WP10c9d6c7c3b5c4f25fj_equilibrium_stability_manifest"
)
ARTIFACT = (
    "causal_inner_curvature_adaptive_arclength_continuation_execution_"
    "wp10c9d6c7c3b5c4f25fi"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CURVATURE_ADAPTIVE_ARCLENGTH_"
    "CONTINUATION_EXECUTION_WP10C9D6C7C3B5C4F25FI_2026-08-23.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_curvature_adaptive_arclength_continuation_"
    "execution_wp10c9d6c7c3b5c4f25fi.py"
)
THIS_TEST = (
    "tests/test_causal_inner_curvature_adaptive_arclength_continuation_"
    "execution_wp10c9d6c7c3b5c4f25fi.py"
)

INITIAL_PARENT_DURATION_SECONDS = 1.6e-2
PARENT_STEP_SECONDS = 2.5e-4
EQUILIBRIUM_SPEED_RATIO = 1.0e-1
CYCLE_HIDDEN_RETURN_DEFECT_MAXIMUM = 5.0e-2


def _helper():
    return manifest._helper()


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    source._save_npz(path, arrays)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    return _helper()._load_npz(path)


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "continuation_execution_contract.json"
    )
    if (
        not summary["passed"]
        or summary["classification"] != manifest.CLASSIFICATION
        or not summary["curvature_adaptive_continuation_execution_authorized"]
        or summary["curvature_adaptive_continuation_executed"]
        or summary["authorized_next"] != manifest.AUTHORIZED_NEXT
        or contract["authorized_execution"] != manifest.AUTHORIZED_NEXT
        or not contract["truth_system"]["autonomous"]
        or contract["truth_system"]["external_clock_or_phase"] != "forbidden"
        or contract["truth_system"]["fixed_Q_rate_or_reaction"] != "forbidden"
        or not contract["segment_validation"]["endpoint_exact_before_propagation"]
        or not contract["segment_validation"]["failed_candidate_is_never_propagated"]
    ):
        raise RuntimeError("curvature-adaptive execution authorization changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("continuation execution requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        ROOT / THIS_RUNNER,
        ROOT / THIS_TEST,
        ROOT / manifest.THIS_RUNNER,
        ROOT / manifest.diagnosis.THIS_RUNNER,
        ROOT / source.THIS_RUNNER,
        ROOT / "src/imri_qpe/layer3_minidisk_1d/conservative_free_field_rom.py",
    )
    return {str(path.relative_to(ROOT)): helper._sha(path) for path in paths}


def _prepare_scratch(parent_lock: dict) -> dict:
    helper = _helper()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": _source_hashes(),
        "manifest_hashes": parent_lock["hashes"],
    }
    identity_path = SCRATCH_DIRECTORY / "execution_identity.json"
    if SCRATCH_DIRECTORY.exists():
        if not identity_path.exists():
            raise RuntimeError("continuation scratch exists without identity")
        if helper._read(identity_path) != identity:
            raise RuntimeError("continuation scratch identity mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(identity_path, identity)
    return identity


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(a - b)
        / max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(float).tiny)
    )


def _variable_step_ab2(
    coordinate: np.ndarray,
    current_rate: np.ndarray,
    previous_rate: np.ndarray,
    span_seconds: float,
    previous_span_seconds: float,
) -> np.ndarray:
    h = float(span_seconds)
    hp = float(previous_span_seconds)
    if h <= 0.0 or hp <= 0.0:
        raise ValueError("AB2 spans must be positive")
    return (
        np.asarray(coordinate)
        + h * np.asarray(current_rate)
        + h * h / (2.0 * hp)
        * (np.asarray(current_rate) - np.asarray(previous_rate))
    )


def _hermite(
    left: np.ndarray,
    left_rate: np.ndarray,
    right: np.ndarray,
    right_rate: np.ndarray,
    span_seconds: float,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    u = float(fraction)
    h = float(span_seconds)
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    coordinate = (
        h00 * np.asarray(left)
        + h10 * h * np.asarray(left_rate)
        + h01 * np.asarray(right)
        + h11 * h * np.asarray(right_rate)
    )
    dh00 = 6.0 * u**2 - 6.0 * u
    dh10 = 3.0 * u**2 - 4.0 * u + 1.0
    dh01 = -6.0 * u**2 + 6.0 * u
    dh11 = 3.0 * u**2 - 2.0 * u
    rate = (
        dh00 * np.asarray(left)
        + dh10 * h * np.asarray(left_rate)
        + dh01 * np.asarray(right)
        + dh11 * h * np.asarray(right_rate)
    ) / h
    return coordinate, rate


def _endpoint_integral_defect(
    left: np.ndarray,
    left_rate: np.ndarray,
    right: np.ndarray,
    right_rate: np.ndarray,
    span_seconds: float,
) -> float:
    displacement_rate = (np.asarray(right) - np.asarray(left)) / float(span_seconds)
    trapezoid_rate = 0.5 * (np.asarray(left_rate) + np.asarray(right_rate))
    return _relative(displacement_rate, trapezoid_rate)


def _section_root_fraction(
    left: np.ndarray,
    left_rate: np.ndarray,
    right: np.ndarray,
    right_rate: np.ndarray,
    span_seconds: float,
    start: np.ndarray,
    normal: np.ndarray,
) -> float | None:
    def value(fraction: float) -> float:
        coordinate, _rate = _hermite(
            left, left_rate, right, right_rate, span_seconds, fraction
        )
        return float(np.asarray(normal) @ (coordinate - np.asarray(start)))

    a = value(0.0)
    b = value(1.0)
    if not (a < 0.0 <= b):
        return None
    lo = 0.0
    hi = 1.0
    for _index in range(80):
        middle = 0.5 * (lo + hi)
        if value(middle) < 0.0:
            lo = middle
        else:
            hi = middle
    return float(0.5 * (lo + hi))


def _pair(directory: Path, stem: str) -> tuple[Path, Path]:
    return directory / f"{stem}.json", directory / f"{stem}.npz"


def _load_pair(directory: Path, stem: str) -> tuple[dict, dict[str, np.ndarray]] | None:
    metrics_path, arrays_path = _pair(directory, stem)
    if metrics_path.exists() != arrays_path.exists():
        raise RuntimeError(f"partial scratch pair: {directory.name}/{stem}")
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


def _parent_data() -> dict:
    cycle = _load_npz(
        source.CANONICAL_DIRECTORY / "cycle_execution_arrays.npz"
    )
    witnesses = _load_npz(
        source.CANONICAL_DIRECTORY / "exact_witness_arrays.npz"
    )
    anchor_mask = np.asarray(witnesses["kinds"]).astype(str) == "anchor"
    anchor_coordinates = np.asarray(witnesses["coordinates"])[anchor_mask]
    anchor_rates = np.asarray(witnesses["coordinate_rates_per_s"])[anchor_mask]
    trajectory = np.asarray(cycle["trajectory_coordinates"])
    states = np.asarray(cycle["trajectory_primitive_states"])
    if (
        trajectory.shape != (65, 470)
        or states.shape != (65, 112, 5)
        or anchor_rates.shape != (64, 470)
    ):
        raise RuntimeError("parent continuation seed shape changed")
    np.testing.assert_array_equal(anchor_coordinates, trajectory[:-1])
    return {
        "start_coordinate": np.asarray(cycle["start_coordinate470"]),
        "start_state": np.asarray(cycle["start_primitive_state"]),
        "trajectory": trajectory,
        "states": states,
        "section_normal": np.asarray(cycle["section_normal470"]),
        "current_coordinate": trajectory[-1].copy(),
        "current_state": states[-1].copy(),
        "previous_coordinate": trajectory[-2].copy(),
        "previous_state": states[-2].copy(),
        "previous_rate": anchor_rates[-1].copy(),
        "previous_span_seconds": PARENT_STEP_SECONDS,
    }


def _evaluate_field(
    inputs: dict,
    directory: Path,
    stem: str,
    state: np.ndarray,
    coordinate: np.ndarray,
    retraction: dict | None,
    kind: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    cached = _load_pair(directory, stem)
    if cached is not None:
        metrics, arrays = cached
        np.testing.assert_array_equal(arrays["requested_coordinate470"], coordinate)
        print(f"{directory.name}/{stem}: reused exact field", flush=True)
        return metrics, arrays
    began = time.perf_counter()
    free_metrics, free_arrays = source._free_field(inputs, np.asarray(state))
    wall = float(time.perf_counter() - began)
    physical_passed = source._physical_passed(free_metrics, retraction)
    metrics = {
        "kind": kind,
        "total_call_wall_seconds": wall,
        "free_field": free_metrics,
        "retraction": retraction,
        "physical_passed": physical_passed,
    }
    arrays = {
        **free_arrays,
        "requested_coordinate470": np.asarray(coordinate),
    }
    _write_pair(directory, stem, metrics, arrays)
    print(
        f"{directory.name}/{stem}: |f|="
        f"{free_metrics['coordinate_free_rate_norm_per_second']:.6e}/s "
        f"wall={wall:.3f}s physical={physical_passed}",
        flush=True,
    )
    return metrics, arrays


def _initial_field(inputs: dict, parent_data: dict) -> tuple[dict, dict[str, np.ndarray]]:
    directory = SCRATCH_DIRECTORY / "initial"
    directory.mkdir(exist_ok=True)
    return _evaluate_field(
        inputs,
        directory,
        "terminal_field",
        parent_data["current_state"],
        parent_data["current_coordinate"],
        None,
        "initial_terminal",
    )


def _inventory() -> dict:
    field_paths = sorted(SCRATCH_DIRECTORY.glob("**/*_field.json"))
    attempts = sorted(SCRATCH_DIRECTORY.glob("attempt_[0-9][0-9][0-9][0-9]"))
    completed = []
    for directory in attempts:
        metrics_path, arrays_path = _pair(directory, "attempt")
        if metrics_path.exists() != arrays_path.exists():
            raise RuntimeError(f"partial completed attempt: {directory.name}")
        if metrics_path.exists():
            completed.append((directory, metrics_path, arrays_path))
    return {
        "exact_field_paths": field_paths,
        "attempt_directories": attempts,
        "completed_attempts": completed,
    }


def _initial_progress(parent_data: dict, initial_arrays: dict[str, np.ndarray]) -> dict:
    section = float(
        parent_data["section_normal"]
        @ (parent_data["current_coordinate"] - parent_data["start_coordinate"])
    )
    initial_rate = np.asarray(initial_arrays["coordinate_free_rate470_per_s"])
    return {
        "current_coordinate": parent_data["current_coordinate"].copy(),
        "current_state": parent_data["current_state"].copy(),
        "current_rate": initial_rate.copy(),
        "previous_coordinate": parent_data["previous_coordinate"].copy(),
        "previous_state": parent_data["previous_state"].copy(),
        "previous_rate": parent_data["previous_rate"].copy(),
        "previous_span_seconds": PARENT_STEP_SECONDS,
        "next_span_seconds": manifest.INITIAL_SEGMENT_SECONDS,
        "elapsed_seconds": INITIAL_PARENT_DURATION_SECONDS,
        "accepted_segments": 0,
        "attempts": 0,
        "accepted_since_growth": 0,
        "seen_negative": False,
        "previous_section": section,
        "initial_speed": float(np.linalg.norm(initial_rate)),
        "cycle_event": None,
        "nonclosing_events": [],
        "stop_classification": None,
    }


def _restore_progress(
    parent_data: dict,
    initial_arrays: dict[str, np.ndarray],
) -> tuple[dict, list[tuple[dict, dict[str, np.ndarray]]]]:
    helper = _helper()
    progress = _initial_progress(parent_data, initial_arrays)
    records = []
    inventory = _inventory()
    expected_attempt = 0
    for directory, metrics_path, arrays_path in inventory["completed_attempts"]:
        if int(directory.name.split("_")[-1]) != expected_attempt:
            raise RuntimeError("attempt scratch is not contiguous")
        metrics = helper._read(metrics_path)
        arrays = _load_npz(arrays_path)
        records.append((metrics, arrays))
        progress["attempts"] += 1
        progress["next_span_seconds"] = float(metrics["next_span_seconds"])
        progress["accepted_since_growth"] = int(
            metrics["accepted_since_growth_after"]
        )
        progress["seen_negative"] = bool(metrics["seen_negative_after"])
        progress["previous_section"] = float(metrics["section_after"])
        if metrics["accepted"]:
            progress["previous_coordinate"] = progress["current_coordinate"]
            progress["previous_state"] = progress["current_state"]
            progress["previous_rate"] = progress["current_rate"]
            progress["current_coordinate"] = np.asarray(
                arrays["accepted_coordinate470"]
            )
            progress["current_state"] = np.asarray(arrays["accepted_primitive_state"])
            progress["current_rate"] = np.asarray(
                arrays["accepted_coordinate_rate470_per_s"]
            )
            progress["previous_span_seconds"] = float(metrics["span_seconds"])
            progress["elapsed_seconds"] = float(metrics["elapsed_seconds_after"])
            progress["accepted_segments"] += 1
        if metrics.get("nonclosing_event") is not None:
            progress["nonclosing_events"].append(metrics["nonclosing_event"])
        if metrics.get("cycle_event") is not None:
            progress["cycle_event"] = metrics["cycle_event"]
        if metrics.get("stop_classification") is not None:
            progress["stop_classification"] = metrics["stop_classification"]
        expected_attempt += 1
    return progress, records


def _remaining_call_capacity(required: int) -> bool:
    used = len(_inventory()["exact_field_paths"])
    return used + int(required) + 1 <= manifest.MAXIMUM_EXACT_FREE_FIELD_CALLS


def _attempt_segment(
    inputs: dict,
    parent_data: dict,
    progress: dict,
    attempt_index: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    directory = SCRATCH_DIRECTORY / f"attempt_{attempt_index:04d}"
    directory.mkdir(exist_ok=True)
    span = float(progress["next_span_seconds"])
    tentative_number = int(progress["accepted_segments"]) + 1
    blind_required = bool(
        tentative_number % manifest.BLIND_MIDPOINT_FREQUENCY == 0
    )
    required_calls = 1 + int(blind_required)
    if not _remaining_call_capacity(required_calls):
        raise RuntimeError("exact_call_budget")
    candidate_target = _variable_step_ab2(
        progress["current_coordinate"],
        progress["current_rate"],
        progress["previous_rate"],
        span,
        progress["previous_span_seconds"],
    )
    context = source._build_retraction_context(
        inputs, progress["current_state"], progress["current_coordinate"]
    )
    endpoint_cached = _load_pair(directory, "endpoint_field")
    if endpoint_cached is None:
        endpoint_state, endpoint_retraction = source._retract(
            inputs, context, candidate_target
        )
        if not (
            endpoint_retraction["passed"]
            and endpoint_retraction["coordinate_residual_infinity"]
            <= source.COORDINATE_RETRACTION_TOLERANCE
            and endpoint_retraction["gauge_residual_infinity"]
            <= source.GAUGE_RETRACTION_TOLERANCE
        ):
            endpoint_metrics = {
                "kind": "endpoint",
                "physical_passed": False,
                "retraction": endpoint_retraction,
                "free_field": None,
            }
            endpoint_arrays = {
                "requested_coordinate470": candidate_target,
                "primitive_state": endpoint_state,
            }
        else:
            recovered, _factors = inputs["model"].coordinate(endpoint_state)
            endpoint_metrics, endpoint_arrays = _evaluate_field(
                inputs,
                directory,
                "endpoint_field",
                endpoint_state,
                np.asarray(recovered),
                endpoint_retraction,
                "endpoint",
            )
    else:
        endpoint_metrics, endpoint_arrays = endpoint_cached

    endpoint_rate = None
    endpoint_coordinate = np.asarray(endpoint_arrays["requested_coordinate470"])
    endpoint_state = np.asarray(endpoint_arrays["primitive_state"])
    endpoint_defect = float("inf")
    blind_defect = None
    midpoint_metrics = None
    midpoint_arrays = None
    endpoint_passed = bool(endpoint_metrics["physical_passed"])
    if endpoint_passed:
        endpoint_rate = np.asarray(
            endpoint_arrays["coordinate_free_rate470_per_s"]
        )
        endpoint_defect = _endpoint_integral_defect(
            progress["current_coordinate"],
            progress["current_rate"],
            endpoint_coordinate,
            endpoint_rate,
            span,
        )
    numerical_passed = bool(
        endpoint_passed
        and endpoint_defect <= manifest.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
    )
    if numerical_passed and blind_required:
        midpoint_target, midpoint_hermite_rate = _hermite(
            progress["current_coordinate"],
            progress["current_rate"],
            endpoint_coordinate,
            endpoint_rate,
            span,
            0.5,
        )
        midpoint_cached = _load_pair(directory, "midpoint_field")
        if midpoint_cached is None:
            midpoint_state, midpoint_retraction = source._retract(
                inputs, context, midpoint_target
            )
            if not (
                midpoint_retraction["passed"]
                and midpoint_retraction["coordinate_residual_infinity"]
                <= source.COORDINATE_RETRACTION_TOLERANCE
                and midpoint_retraction["gauge_residual_infinity"]
                <= source.GAUGE_RETRACTION_TOLERANCE
            ):
                midpoint_metrics = {
                    "kind": "blind_midpoint",
                    "physical_passed": False,
                    "retraction": midpoint_retraction,
                    "free_field": None,
                }
                midpoint_arrays = {
                    "requested_coordinate470": midpoint_target,
                    "primitive_state": midpoint_state,
                }
                helper._write_json(
                    directory / "midpoint_retraction_failure.json",
                    midpoint_metrics,
                )
            else:
                midpoint_recovered, _factors = inputs["model"].coordinate(
                    midpoint_state
                )
                midpoint_metrics, midpoint_arrays = _evaluate_field(
                    inputs,
                    directory,
                    "midpoint_field",
                    midpoint_state,
                    np.asarray(midpoint_recovered),
                    midpoint_retraction,
                    "blind_midpoint",
                )
        else:
            midpoint_metrics, midpoint_arrays = midpoint_cached
        if midpoint_metrics["physical_passed"]:
            blind_defect = _relative(
                midpoint_hermite_rate,
                midpoint_arrays["coordinate_free_rate470_per_s"],
            )
        else:
            blind_defect = float("inf")
        numerical_passed = bool(
            numerical_passed
            and midpoint_metrics["physical_passed"]
            and blind_defect <= manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
        )

    accepted = bool(endpoint_passed and numerical_passed)
    physical_failure = bool(
        not endpoint_passed
        or (
            midpoint_metrics is not None
            and not midpoint_metrics["physical_passed"]
        )
    )
    section_after = float(progress["previous_section"])
    seen_negative_after = bool(progress["seen_negative"])
    nonclosing_event = None
    cycle_event = None
    event_arrays = {}
    if accepted:
        section_after = float(
            parent_data["section_normal"]
            @ (endpoint_coordinate - parent_data["start_coordinate"])
        )
        if section_after < 0.0:
            seen_negative_after = True
        fraction = None
        if seen_negative_after:
            fraction = _section_root_fraction(
                progress["current_coordinate"],
                progress["current_rate"],
                endpoint_coordinate,
                endpoint_rate,
                span,
                parent_data["start_coordinate"],
                parent_data["section_normal"],
            )
        if fraction is not None:
            event_target, event_hermite_rate = _hermite(
                progress["current_coordinate"],
                progress["current_rate"],
                endpoint_coordinate,
                endpoint_rate,
                span,
                fraction,
            )
            event_state, event_retraction = source._retract(
                inputs, context, event_target
            )
            event_retraction_passed = bool(
                event_retraction["passed"]
                and event_retraction["coordinate_residual_infinity"]
                <= source.COORDINATE_RETRACTION_TOLERANCE
                and event_retraction["gauge_residual_infinity"]
                <= source.GAUGE_RETRACTION_TOLERANCE
            )
            event_recovered, _factors = inputs["model"].coordinate(event_state)
            if event_retraction_passed:
                event_metrics, event_field_arrays = _evaluate_field(
                    inputs,
                    directory,
                    "event_field",
                    event_state,
                    np.asarray(event_recovered),
                    event_retraction,
                    "poincare_event",
                )
                exact_event_rate = np.asarray(
                    event_field_arrays["coordinate_free_rate470_per_s"]
                )
            else:
                event_metrics = {
                    "kind": "poincare_event",
                    "physical_passed": False,
                    "retraction": event_retraction,
                    "free_field": None,
                }
                exact_event_rate = np.asarray(event_hermite_rate)
                helper._write_json(
                    directory / "event_retraction_failure.json", event_metrics
                )
            orientation = float(exact_event_rate @ parent_data["section_normal"])
            start_hidden = inputs["split"].split(parent_data["start_coordinate"])[1]
            parent_hidden = np.stack([
                inputs["split"].split(value)[1]
                for value in parent_data["trajectory"]
            ])
            hidden_path = float(
                np.sum(np.linalg.norm(np.diff(parent_hidden, axis=0), axis=1))
            )
            event_hidden = inputs["split"].split(np.asarray(event_recovered))[1]
            hidden_return = float(
                np.linalg.norm(event_hidden - start_hidden)
                / max(hidden_path, np.finfo(float).tiny)
            )
            event_record = {
                "fraction": fraction,
                "elapsed_seconds": progress["elapsed_seconds"] + fraction * span,
                "section_value": float(
                    parent_data["section_normal"]
                    @ (np.asarray(event_recovered) - parent_data["start_coordinate"])
                ),
                "orientation_per_second": orientation,
                "hermite_to_exact_rate_defect": _relative(
                    event_hermite_rate, exact_event_rate
                ),
                "hidden_return_defect": hidden_return,
                "physical_passed": bool(event_metrics["physical_passed"]),
            }
            event_arrays = {
                "event_coordinate470": np.asarray(event_recovered),
                "event_primitive_state": np.asarray(event_state),
                "event_coordinate_rate470_per_s": exact_event_rate,
            }
            event_valid = bool(
                event_metrics["physical_passed"]
                and orientation > 0.0
                and event_record["hermite_to_exact_rate_defect"]
                <= manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
            )
            if not event_valid:
                accepted = False
                physical_failure = not event_metrics["physical_passed"]
                numerical_passed = False
            elif hidden_return <= CYCLE_HIDDEN_RETURN_DEFECT_MAXIMUM:
                cycle_event = event_record
            else:
                nonclosing_event = event_record
                seen_negative_after = False

    stop_classification = None
    accepted_since_growth_after = int(progress["accepted_since_growth"])
    if accepted:
        accepted_since_growth_after += 1
        next_span = span
        if (
            blind_required
            and accepted_since_growth_after
            >= manifest.ACCEPTED_SEGMENTS_BEFORE_GROWTH
            and span < manifest.MAXIMUM_SEGMENT_SECONDS
        ):
            next_span = min(
                manifest.GROWTH_FACTOR_MAXIMUM * span,
                manifest.MAXIMUM_SEGMENT_SECONDS,
            )
            accepted_since_growth_after = 0
    elif span > manifest.MINIMUM_SEGMENT_SECONDS:
        next_span = max(0.5 * span, manifest.MINIMUM_SEGMENT_SECONDS)
        accepted_since_growth_after = 0
    else:
        next_span = span
        stop_classification = (
            PHYSICAL_CLASSIFICATION if physical_failure else VALIDATION_CLASSIFICATION
        )
    if not accepted:
        section_after = float(progress["previous_section"])
        seen_negative_after = bool(progress["seen_negative"])
        nonclosing_event = None
        cycle_event = None
        event_arrays = {}

    elapsed_after = (
        progress["elapsed_seconds"] + span if accepted else progress["elapsed_seconds"]
    )
    metrics = {
        "attempt_index": int(attempt_index),
        "tentative_segment_number": tentative_number,
        "span_seconds": span,
        "previous_span_seconds": float(progress["previous_span_seconds"]),
        "blind_midpoint_required": blind_required,
        "endpoint_physical_passed": endpoint_passed,
        "endpoint_integral_defect": endpoint_defect,
        "blind_midpoint_rate_defect": blind_defect,
        "numerical_passed": numerical_passed,
        "accepted": accepted,
        "section_before": float(progress["previous_section"]),
        "section_after": section_after,
        "seen_negative_after": seen_negative_after,
        "accepted_since_growth_after": accepted_since_growth_after,
        "next_span_seconds": next_span,
        "elapsed_seconds_after": elapsed_after,
        "nonclosing_event": nonclosing_event,
        "cycle_event": cycle_event,
        "stop_classification": stop_classification,
    }
    accepted_coordinate = (
        endpoint_coordinate if accepted else progress["current_coordinate"]
    )
    accepted_state = endpoint_state if accepted else progress["current_state"]
    accepted_rate = endpoint_rate if accepted else progress["current_rate"]
    arrays = {
        "candidate_target470": candidate_target,
        "endpoint_coordinate470": endpoint_coordinate,
        "endpoint_primitive_state": endpoint_state,
        "endpoint_coordinate_rate470_per_s": (
            np.full(470, np.nan) if endpoint_rate is None else endpoint_rate
        ),
        "accepted_coordinate470": np.asarray(accepted_coordinate),
        "accepted_primitive_state": np.asarray(accepted_state),
        "accepted_coordinate_rate470_per_s": np.asarray(accepted_rate),
        **event_arrays,
    }
    _write_pair(directory, "attempt", metrics, arrays)
    helper._write_json(
        SCRATCH_DIRECTORY / "cumulative_wall_seconds.json",
        {"wall_seconds": _current_total_wall()},
    )
    return metrics, arrays


_RUN_STARTED = 0.0
_PRIOR_WALL = 0.0


def _current_total_wall() -> float:
    return float(_PRIOR_WALL + max(0.0, time.perf_counter() - _RUN_STARTED))


def _checkpoint_roundtrip(
    parent_data: dict,
    initial_arrays: dict[str, np.ndarray],
    progress: dict,
    records: list[tuple[dict, dict[str, np.ndarray]]],
) -> tuple[bool, bool, int]:
    accepted_indices = [
        index for index, (metrics, _arrays) in enumerate(records)
        if metrics["accepted"]
    ]
    if not accepted_indices:
        return True, True, 0
    midpoint_accepted = len(accepted_indices) // 2
    checkpoint_attempt = accepted_indices[midpoint_accepted]
    checkpoint_progress = _initial_progress(parent_data, initial_arrays)
    for earlier_metrics, earlier_arrays in records[: checkpoint_attempt + 1]:
        checkpoint_progress["next_span_seconds"] = float(
            earlier_metrics["next_span_seconds"]
        )
        checkpoint_progress["accepted_since_growth"] = int(
            earlier_metrics["accepted_since_growth_after"]
        )
        checkpoint_progress["seen_negative"] = bool(
            earlier_metrics["seen_negative_after"]
        )
        checkpoint_progress["previous_section"] = float(
            earlier_metrics["section_after"]
        )
        checkpoint_progress["attempts"] += 1
        if earlier_metrics["accepted"]:
            checkpoint_progress["previous_coordinate"] = checkpoint_progress[
                "current_coordinate"
            ]
            checkpoint_progress["previous_state"] = checkpoint_progress[
                "current_state"
            ]
            checkpoint_progress["previous_rate"] = checkpoint_progress[
                "current_rate"
            ]
            checkpoint_progress["current_coordinate"] = np.asarray(
                earlier_arrays["accepted_coordinate470"]
            )
            checkpoint_progress["current_state"] = np.asarray(
                earlier_arrays["accepted_primitive_state"]
            )
            checkpoint_progress["current_rate"] = np.asarray(
                earlier_arrays["accepted_coordinate_rate470_per_s"]
            )
            checkpoint_progress["previous_span_seconds"] = float(
                earlier_metrics["span_seconds"]
            )
            checkpoint_progress["elapsed_seconds"] = float(
                earlier_metrics["elapsed_seconds_after"]
            )
            checkpoint_progress["accepted_segments"] += 1
    checkpoint_directory = SCRATCH_DIRECTORY / "restart_audit"
    checkpoint_directory.mkdir(exist_ok=True)
    checkpoint_arrays = {
        "current_coordinate470": np.asarray(checkpoint_progress["current_coordinate"]),
        "current_primitive_state": np.asarray(checkpoint_progress["current_state"]),
        "current_coordinate_rate470_per_s": np.asarray(checkpoint_progress["current_rate"]),
        "previous_coordinate470": np.asarray(checkpoint_progress["previous_coordinate"]),
        "previous_primitive_state": np.asarray(checkpoint_progress["previous_state"]),
        "previous_coordinate_rate470_per_s": np.asarray(checkpoint_progress["previous_rate"]),
    }
    checkpoint_metrics = {
        "attempt_index": checkpoint_attempt,
        "elapsed_seconds": checkpoint_progress["elapsed_seconds"],
        "next_span_seconds": checkpoint_progress["next_span_seconds"],
        "previous_span_seconds": checkpoint_progress["previous_span_seconds"],
        "seen_negative": checkpoint_progress["seen_negative"],
        "section": checkpoint_progress["previous_section"],
        "accepted_segments": checkpoint_progress["accepted_segments"],
        "accepted_since_growth": checkpoint_progress["accepted_since_growth"],
    }
    _write_pair(
        checkpoint_directory, "checkpoint", checkpoint_metrics, checkpoint_arrays
    )
    restored_metrics, restored_arrays = _load_pair(
        checkpoint_directory, "checkpoint"
    )
    roundtrip = bool(
        restored_metrics == checkpoint_metrics
        and all(
            np.array_equal(checkpoint_arrays[name], restored_arrays[name])
            for name in checkpoint_arrays
        )
    )
    replay_current_coordinate = np.asarray(restored_arrays["current_coordinate470"])
    replay_current_state = np.asarray(restored_arrays["current_primitive_state"])
    replay_current_rate = np.asarray(
        restored_arrays["current_coordinate_rate470_per_s"]
    )
    replay_previous_rate = np.asarray(
        restored_arrays["previous_coordinate_rate470_per_s"]
    )
    replay_previous_span = float(restored_metrics["previous_span_seconds"])
    replay_next_span = float(restored_metrics["next_span_seconds"])
    suffix = True
    for later_metrics, later_arrays in records[checkpoint_attempt + 1 :]:
        suffix = bool(
            suffix
            and float(later_metrics["span_seconds"]) == replay_next_span
            and np.array_equal(
                _variable_step_ab2(
                    replay_current_coordinate,
                    replay_current_rate,
                    replay_previous_rate,
                    replay_next_span,
                    replay_previous_span,
                ),
                np.asarray(later_arrays["candidate_target470"]),
            )
        )
        if later_metrics["accepted"]:
            replay_previous_rate = replay_current_rate
            replay_current_coordinate = np.asarray(
                later_arrays["accepted_coordinate470"]
            )
            replay_current_state = np.asarray(
                later_arrays["accepted_primitive_state"]
            )
            replay_current_rate = np.asarray(
                later_arrays["accepted_coordinate_rate470_per_s"]
            )
            replay_previous_span = float(later_metrics["span_seconds"])
        replay_next_span = float(later_metrics["next_span_seconds"])
    suffix = bool(
        suffix
        and np.array_equal(replay_current_coordinate, progress["current_coordinate"])
        and np.array_equal(replay_current_state, progress["current_state"])
        and np.array_equal(replay_current_rate, progress["current_rate"])
    )
    return roundtrip, suffix, checkpoint_attempt


def _execute(parent_lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    del parent_lock, identity
    global _RUN_STARTED, _PRIOR_WALL
    helper = _helper()
    wall_path = SCRATCH_DIRECTORY / "cumulative_wall_seconds.json"
    _PRIOR_WALL = (
        float(helper._read(wall_path)["wall_seconds"])
        if wall_path.exists()
        else 0.0
    )
    _RUN_STARTED = time.perf_counter()
    inputs = source._initial_inputs()
    parent_data = _parent_data()
    initial_metrics, initial_arrays = _initial_field(inputs, parent_data)
    helper._write_json(wall_path, {"wall_seconds": _current_total_wall()})
    if not initial_metrics["physical_passed"]:
        raise RuntimeError("initial terminal exact field failed physical audit")
    progress, records = _restore_progress(parent_data, initial_arrays)

    while progress["stop_classification"] is None and progress["cycle_event"] is None:
        inventory = _inventory()
        if (
            progress["accepted_segments"] >= manifest.MAXIMUM_ACCEPTED_SEGMENTS
            or len(inventory["exact_field_paths"])
            >= manifest.MAXIMUM_EXACT_FREE_FIELD_CALLS
            or _current_total_wall() / 3600.0
            >= manifest.MAXIMUM_EXECUTION_WALL_HOURS
        ):
            progress["stop_classification"] = BUDGET_CLASSIFICATION
            break
        try:
            metrics, arrays = _attempt_segment(
                inputs, parent_data, progress, progress["attempts"]
            )
        except RuntimeError as error:
            if str(error) == "exact_call_budget":
                progress["stop_classification"] = BUDGET_CLASSIFICATION
                break
            raise
        records.append((metrics, arrays))
        progress, records = _restore_progress(parent_data, initial_arrays)
        print(
            f"attempt={metrics['attempt_index']:04d} "
            f"accepted={metrics['accepted']} h={metrics['span_seconds']:.3e}s "
            f"endpoint={metrics['endpoint_integral_defect']:.3e} "
            f"blind={metrics['blind_midpoint_rate_defect']} "
            f"accepted_segments={progress['accepted_segments']} "
            f"elapsed={progress['elapsed_seconds']:.6e}s "
            f"next_h={progress['next_span_seconds']:.3e}s",
            flush=True,
        )
        if progress["cycle_event"] is not None:
            break
        if (
            np.linalg.norm(progress["current_rate"])
            <= EQUILIBRIUM_SPEED_RATIO * progress["initial_speed"]
        ):
            progress["stop_classification"] = EQUILIBRIUM_CLASSIFICATION
            break

    total_wall = _current_total_wall()
    helper._write_json(wall_path, {"wall_seconds": total_wall})
    restart_roundtrip, suffix_replay, checkpoint_attempt = _checkpoint_roundtrip(
        parent_data, initial_arrays, progress, records
    )
    if any(record[0]["accepted"] for record in records) and (
        not restart_roundtrip or not suffix_replay
    ):
        progress["stop_classification"] = RESTART_CLASSIFICATION
        progress["cycle_event"] = None

    if progress["cycle_event"] is not None:
        classification = PASS_CLASSIFICATION
        passed = True
    else:
        classification = progress["stop_classification"] or BUDGET_CLASSIFICATION
        passed = False

    accepted_records = [record for record in records if record[0]["accepted"]]
    attempted_endpoint_defects = np.asarray([
        record[0]["endpoint_integral_defect"] for record in records
    ])
    attempted_blind_defects = np.asarray([
        record[0]["blind_midpoint_rate_defect"]
        for record in records
        if record[0]["blind_midpoint_rate_defect"] is not None
    ])
    accepted_blind_defects = np.asarray([
        record[0]["blind_midpoint_rate_defect"]
        for record in accepted_records
        if record[0]["blind_midpoint_rate_defect"] is not None
    ])
    trajectory = np.vstack((
        parent_data["trajectory"],
        np.stack([
            record[1]["accepted_coordinate470"] for record in accepted_records
        ]) if accepted_records else np.empty((0, 470)),
    ))
    trajectory_states = np.concatenate((
        parent_data["states"],
        np.stack([
            record[1]["accepted_primitive_state"] for record in accepted_records
        ]) if accepted_records else np.empty((0, 112, 5)),
    ))
    accepted_rates = np.stack([
        record[1]["accepted_coordinate_rate470_per_s"]
        for record in accepted_records
    ]) if accepted_records else np.empty((0, 470))
    accepted_spans = np.asarray([
        record[0]["span_seconds"] for record in accepted_records
    ])
    exact_calls = len(_inventory()["exact_field_paths"])
    accepted_field_paths = [
        SCRATCH_DIRECTORY / "initial" / "terminal_field.json"
    ]
    for record in accepted_records:
        attempt_directory = (
            SCRATCH_DIRECTORY / f"attempt_{record[0]['attempt_index']:04d}"
        )
        accepted_field_paths.append(attempt_directory / "endpoint_field.json")
        if record[0]["blind_midpoint_required"]:
            accepted_field_paths.append(attempt_directory / "midpoint_field.json")
        if (
            record[0]["cycle_event"] is not None
            or record[0]["nonclosing_event"] is not None
        ):
            accepted_field_paths.append(attempt_directory / "event_field.json")
    accepted_field_metrics = [helper._read(path) for path in accepted_field_paths]
    accepted_free_metrics = [item["free_field"] for item in accepted_field_metrics]
    ledger_maxima = [
        max(
            value
            for name, value in item["reaction_free_ledger_values"].items()
            if name != "incoming_excision_characteristics"
        )
        for item in accepted_free_metrics
    ]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "gate_values": {
            "parent_duration_seconds": INITIAL_PARENT_DURATION_SECONDS,
            "terminal_elapsed_seconds": float(progress["elapsed_seconds"]),
            "new_accepted_horizon_seconds": float(
                progress["elapsed_seconds"] - INITIAL_PARENT_DURATION_SECONDS
            ),
            "attempted_segments": len(records),
            "accepted_segments": len(accepted_records),
            "rejected_segments": len(records) - len(accepted_records),
            "exact_free_field_calls": exact_calls,
            "execution_wall_seconds": total_wall,
            "minimum_accepted_span_seconds": (
                None if not len(accepted_spans) else float(np.min(accepted_spans))
            ),
            "maximum_accepted_span_seconds": (
                None if not len(accepted_spans) else float(np.max(accepted_spans))
            ),
            "maximum_accepted_endpoint_integral_defect": max(
                (
                    record[0]["endpoint_integral_defect"]
                    for record in accepted_records
                ),
                default=0.0,
            ),
            "maximum_blind_midpoint_rate_defect": (
                0.0
                if not len(accepted_blind_defects)
                else float(np.max(accepted_blind_defects))
            ),
            "maximum_attempted_blind_midpoint_rate_defect": (
                0.0
                if not len(attempted_blind_defects)
                else float(np.max(attempted_blind_defects))
            ),
            "minimum_section_value": float(
                np.min((trajectory - parent_data["start_coordinate"])
                       @ parent_data["section_normal"])
            ),
            "terminal_section_value": float(
                parent_data["section_normal"]
                @ (progress["current_coordinate"] - parent_data["start_coordinate"])
            ),
            "nonclosing_poincare_returns": len(progress["nonclosing_events"]),
            "cycle_observed": progress["cycle_event"] is not None,
            "cycle_duration_seconds": (
                None if progress["cycle_event"] is None
                else progress["cycle_event"]["elapsed_seconds"]
            ),
            "restart_roundtrip_bitwise": restart_roundtrip,
            "suffix_replay_bitwise": suffix_replay,
            "restart_checkpoint_attempt": checkpoint_attempt,
            "all_accepted_exact_fields_physical_passed": all(
                item["physical_passed"] for item in accepted_field_metrics
            ),
            "minimum_accepted_reconstruction_factor": float(
                min(item["minimum_reconstruction_factor"] for item in accepted_free_metrics)
            ),
            "maximum_accepted_height_ratio": float(
                max(item["maximum_height_ratio"] for item in accepted_free_metrics)
            ),
            "minimum_accepted_scattering_optical_depth": float(
                min(
                    item["minimum_scattering_optical_depth"]
                    for item in accepted_free_metrics
                )
            ),
            "maximum_accepted_reaction_free_ledger_defect": float(
                max(ledger_maxima, default=0.0)
            ),
        },
        "cycle_event": progress["cycle_event"],
        "nonclosing_events": progress["nonclosing_events"],
        "fixed_Q_physical_rate_calls": 0,
        "fixed_Q_reaction_calls": 0,
        "nonlinear_roots": 0,
        "BDF_microsteps": 0,
    }
    arrays = {
        "trajectory_coordinates": trajectory,
        "trajectory_primitive_states": trajectory_states,
        "accepted_endpoint_rates470_per_s": accepted_rates,
        "accepted_segment_seconds": accepted_spans,
        "attempted_endpoint_integral_defects": attempted_endpoint_defects,
        "blind_midpoint_rate_defects": attempted_blind_defects,
        "section_normal470": parent_data["section_normal"],
        "start_coordinate470": parent_data["start_coordinate"],
        "terminal_coordinate470": progress["current_coordinate"],
        "terminal_primitive_state": progress["current_state"],
        "terminal_coordinate_rate470_per_s": progress["current_rate"],
    }
    cycle_record = next(
        (record for record in accepted_records if record[0]["cycle_event"] is not None),
        None,
    )
    if cycle_record is not None:
        arrays.update({
            "cycle_event_coordinate470": cycle_record[1]["event_coordinate470"],
            "cycle_event_primitive_state": cycle_record[1]["event_primitive_state"],
            "cycle_event_coordinate_rate470_per_s": (
                cycle_record[1]["event_coordinate_rate470_per_s"]
            ),
        })
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = (
        source.manifest.parent.manifest.parent.arclength
        ._source()._post().manifest.transition.manifest.cold.manifest
    )
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED",
            })
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
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
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _canonicalize(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    identity: dict,
    parent_lock: dict,
) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("continuation execution result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(
        CANONICAL_DIRECTORY / "continuation_execution_metrics.json", metrics
    )
    _save_npz(CANONICAL_DIRECTORY / "continuation_execution_arrays.npz", arrays)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", parent_lock)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        **identity,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    cycle = metrics["classification"] == PASS_CLASSIFICATION
    equilibrium = metrics["classification"] == EQUILIBRIUM_CLASSIFICATION
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "autonomous_original_free_field_preserved": True,
        "cycle_observed": cycle,
        "equilibrium_candidate_observed": equilibrium,
        "local_transport_certificate_passed": cycle,
        "global_cycle_map_certificate_passed": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            AUTHORIZED_CYCLE_NEXT if cycle
            else AUTHORIZED_EQUILIBRIUM_NEXT if equilibrium
            else None
        ),
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
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
            "# Curvature-adaptive arclength continuation execution",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"The autonomous original-free-field continuation accepted `{values['accepted_segments']}` of `{values['attempted_segments']}` attempted wide segments and advanced `{values['new_accepted_horizon_seconds']:.6e}` s beyond the committed 16 ms prefix using `{values['exact_free_field_calls']}` exact calls in `{values['execution_wall_seconds'] / 3600.0:.3f}` wall hours.",
            "",
            f"The largest accepted span was `{values['maximum_accepted_span_seconds']}` s, the maximum accepted endpoint integral defect was `{values['maximum_accepted_endpoint_integral_defect']:.6e}`, and the maximum blind midpoint rate defect was `{values['maximum_blind_midpoint_rate_defect']:.6e}`. Restart roundtrip and suffix replay were `{values['restart_roundtrip_bitwise']}` and `{values['suffix_replay_bitwise']}`.",
            "",
            "No fixed-Q physical clock, fixed-Q reaction, nonlinear root, or BDF microstep was used. Even a detected cycle remains subject to matched-path/global cycle-map refinement before slow closure or reduced evolution.",
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
    parent_lock = _validate_manifest(require_clean=True)
    identity = _prepare_scratch(parent_lock)
    metrics, arrays = _execute(parent_lock, identity)
    summary = _canonicalize(metrics, arrays, identity, parent_lock)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
