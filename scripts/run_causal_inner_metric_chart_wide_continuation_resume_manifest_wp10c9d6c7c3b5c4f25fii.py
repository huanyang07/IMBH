#!/usr/bin/env python3
"""Freeze a bounded adaptive continuation resume using metric chart patches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_metric_chart_short_suffix_execution_wp10c9d6c7c3b5c4f25fih as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fii"
CLASSIFICATION = "metric_chart_wide_continuation_resume_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fij_metric_chart_wide_continuation_resume_execution"
)
ARTIFACT = (
    "causal_inner_metric_chart_wide_continuation_resume_manifest_"
    "wp10c9d6c7c3b5c4f25fii"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_METRIC_CHART_WIDE_CONTINUATION_"
    "RESUME_MANIFEST_WP10C9D6C7C3B5C4F25FII_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_metric_chart_wide_continuation_resume_manifest_"
    "wp10c9d6c7c3b5c4f25fii.py"
)
THIS_TEST = (
    "tests/test_causal_inner_metric_chart_wide_continuation_resume_manifest_"
    "wp10c9d6c7c3b5c4f25fii.py"
)

INITIAL_ELAPSED_SECONDS = parent.manifest.TERMINAL_ELAPSED_SECONDS
INITIAL_ACCEPTED_SEGMENTS = parent.manifest.INITIAL_ACCEPTED_SEGMENTS + parent.manifest.NEW_SEGMENTS
INITIAL_SEGMENT_SECONDS = 5.0e-4
MINIMUM_SEGMENT_SECONDS = 2.5e-4
MAXIMUM_SEGMENT_SECONDS = 2.0e-3
GROWTH_FACTOR_MAXIMUM = 2.0
ACCEPTED_SEGMENTS_BEFORE_GROWTH = 4
BLIND_MIDPOINT_FREQUENCY = 4
MAXIMUM_ACCEPTED_SEGMENTS = 64
MAXIMUM_ATTEMPTED_SEGMENTS = 72
MAXIMUM_EXACT_FREE_FIELD_CALLS = 88
MAXIMUM_EXECUTION_WALL_HOURS = 10.0
COST_RESERVE_FACTOR = 1.25
MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = parent.manifest.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT = parent.manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
MAXIMUM_METRIC_JACOBIAN_CONDITION = parent.manifest.MAXIMUM_METRIC_JACOBIAN_CONDITION
MAXIMUM_METRIC_AUGMENTED_CONDITION = parent.manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
MAXIMUM_PATCH_TRANSITION_CONDITION = parent.manifest.MAXIMUM_PATCH_TRANSITION_CONDITION
MAXIMUM_TRANSFORM_INVERSE_CLOSURE = parent.manifest.MAXIMUM_TRANSFORM_INVERSE_CLOSURE
MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT = parent.manifest.MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT
ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE = parent.manifest.ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
METRIC_COORDINATE_RESIDUAL_TOLERANCE = parent.manifest.METRIC_COORDINATE_RESIDUAL_TOLERANCE
GAUGE_RESIDUAL_TOLERANCE = parent.manifest.GAUGE_RESIDUAL_TOLERANCE
CYCLE_HIDDEN_RETURN_DEFECT_MAXIMUM = 5.0e-2
EQUILIBRIUM_SPEED_RATIO = 1.0e-1
RESTART_CHECKPOINT_ACCEPTED_INDEX = 8


def _helper():
    return parent._helper()


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(parent.CANONICAL_DIRECTORY / "suffix_execution_metrics.json")
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["metric_chart_short_suffix_passed"]
        or summary["new_accepted_segments"] != parent.manifest.NEW_SEGMENTS
        or not summary["wide_resume_manifest_authorized"]
        or summary["authorized_next"] != parent.AUTHORIZED_NEXT
        or metrics["classification"] != parent.PASS_CLASSIFICATION
        or not metrics["passed"]
        or values["terminal_elapsed_seconds"] != INITIAL_ELAPSED_SECONDS
        or values["new_accepted_segments"] != parent.manifest.NEW_SEGMENTS
        or not values["all_segment_checkpoint_roundtrips_bitwise"]
        or not values["restart_checkpoint_roundtrip_bitwise"]
        or not values["remaining_suffix_history_replay_bitwise"]
        or not values["all_exact_fields_physical_passed"]
        or values["maximum_metric_coordinate_jacobian_condition"]
        > MAXIMUM_METRIC_JACOBIAN_CONDITION
        or metrics["fixed_Q_physical_rate_calls"] != 0
        or metrics["fixed_Q_reaction_calls"] != 0
    ):
        raise RuntimeError("metric short-suffix evidence changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("wide-resume manifest requires a clean tracked tree")
    return {"hashes": hashes, "classification": summary["classification"]}


def _load_original_trajectory() -> tuple[np.ndarray, np.ndarray]:
    path = (
        parent.execution.CANONICAL_DIRECTORY / "continuation_execution_arrays.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["trajectory_coordinates"], dtype=float)
        states = np.asarray(payload["trajectory_primitive_states"], dtype=float)
    if coordinates.shape != (136, 470) or states.shape != (136, 112, 5):
        raise RuntimeError("original accepted trajectory shape changed")
    return coordinates, states


def _seed() -> dict[str, np.ndarray]:
    helper = _helper()
    suffix = helper._load_npz(
        parent.CANONICAL_DIRECTORY / "suffix_execution_arrays.npz"
    )
    boundary = helper._load_npz(
        parent.parent.CANONICAL_DIRECTORY / "boundary_execution_arrays.npz"
    )
    old_coordinates, old_states = _load_original_trajectory()
    new_coordinates = np.asarray(suffix["accepted_endpoint_coordinates470"])
    new_states = np.asarray(suffix["accepted_endpoint_primitive_states"])
    if new_coordinates.shape != (4, 470) or new_states.shape != (4, 112, 5):
        raise RuntimeError("metric suffix trajectory shape changed")
    boundary_coordinate = np.asarray(boundary["current_coordinate470"])
    boundary_state = np.asarray(boundary["current_primitive_state"])
    trajectory_coordinates = np.vstack((
        old_coordinates,
        boundary_coordinate[None, :],
        new_coordinates,
    ))
    trajectory_states = np.concatenate((
        old_states,
        boundary_state[None, :, :],
        new_states,
    ))
    np.testing.assert_array_equal(
        trajectory_coordinates[-1], suffix["current_coordinate470"]
    )
    np.testing.assert_array_equal(
        trajectory_states[-1], suffix["current_primitive_state"]
    )
    section_values = (
        trajectory_coordinates - suffix["start_coordinate470"]
    ) @ suffix["section_normal470"]
    if float(np.min(section_values)) < 0.0:
        raise RuntimeError("pre-resume history unexpectedly crossed negative section")
    names = (
        "previous_coordinate470",
        "current_coordinate470",
        "previous_primitive_state",
        "current_primitive_state",
        "previous_coordinate_rate470_per_s",
        "current_coordinate_rate470_per_s",
        "previous_span_seconds",
        "elapsed_seconds",
        "accepted_segments_total",
        "current_metric_transform470x470",
        "current_metric_augmented560x560",
        "current_gauge_basis560x90",
        "section_normal470",
        "start_coordinate470",
    )
    seed = {name: np.asarray(suffix[name]) for name in names}
    seed.update({
        "trajectory_coordinates": trajectory_coordinates,
        "trajectory_primitive_states": trajectory_states,
        "seen_negative_section": np.asarray(False),
        "initial_resume_speed_per_second": np.asarray(
            np.linalg.norm(suffix["current_coordinate_rate470_per_s"])
        ),
        "next_span_seconds": np.asarray(INITIAL_SEGMENT_SECONDS),
        "accepted_since_growth": np.asarray(0),
    })
    if (
        float(seed["elapsed_seconds"]) != INITIAL_ELAPSED_SECONDS
        or int(seed["accepted_segments_total"]) != INITIAL_ACCEPTED_SEGMENTS
        or float(seed["previous_span_seconds"]) != parent.manifest.SEGMENT_SECONDS
    ):
        raise RuntimeError("metric suffix terminal counters changed")
    return seed


def _cost_projection() -> dict:
    metrics = _helper()._read(
        parent.CANONICAL_DIRECTORY / "suffix_execution_metrics.json"
    )
    values = metrics["gate_values"]
    seconds_per_call = (
        float(values["execution_wall_seconds"])
        / int(values["exact_free_field_calls"])
    )
    raw_hours = MAXIMUM_EXACT_FREE_FIELD_CALLS * seconds_per_call / 3600.0
    reserved_hours = COST_RESERVE_FACTOR * raw_hours
    no_rejection_calls = (
        MAXIMUM_ACCEPTED_SEGMENTS
        + MAXIMUM_ACCEPTED_SEGMENTS // BLIND_MIDPOINT_FREQUENCY
    )
    maximum_horizon = (
        4 * INITIAL_SEGMENT_SECONDS
        + 4 * 1.0e-3
        + (MAXIMUM_ACCEPTED_SEGMENTS - 8) * MAXIMUM_SEGMENT_SECONDS
    )
    return {
        "observed_suffix_wall_seconds_per_exact_call": seconds_per_call,
        "maximum_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
        "no_rejection_exact_field_calls": no_rejection_calls,
        "rejection_or_event_call_reserve": (
            MAXIMUM_EXACT_FREE_FIELD_CALLS - no_rejection_calls
        ),
        "raw_projected_wall_hours": raw_hours,
        "reserve_factor": COST_RESERVE_FACTOR,
        "reserved_projected_wall_hours": reserved_hours,
        "maximum_execution_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
        "cost_gate_passed": reserved_hours <= MAXIMUM_EXECUTION_WALL_HOURS,
        "maximum_no_rejection_horizon_seconds": maximum_horizon,
        "maximum_no_rejection_terminal_elapsed_seconds": (
            INITIAL_ELAPSED_SECONDS + maximum_horizon
        ),
    }


def _contract() -> dict:
    cost = _cost_projection()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "authorized_execution": AUTHORIZED_NEXT,
        "truth_system": {
            "field": "autonomous original reaction-free field dy/dt=f_free(y)",
            "state": "original primitive state u",
            "physical_coordinate": "original q=C(u)",
            "metric_chart": "new exact block-whitened patch at every accepted endpoint",
            "all_physics_ledgers_events_and_interpolation_in_original_q": True,
            "fixed_Q_rate_or_reaction": "forbidden",
            "external_clock_or_phase": "forbidden",
        },
        "adaptive_policy": {
            "initial_segment_seconds": INITIAL_SEGMENT_SECONDS,
            "minimum_segment_seconds": MINIMUM_SEGMENT_SECONDS,
            "maximum_segment_seconds": MAXIMUM_SEGMENT_SECONDS,
            "growth_factor_maximum": GROWTH_FACTOR_MAXIMUM,
            "accepted_segments_before_growth": ACCEPTED_SEGMENTS_BEFORE_GROWTH,
            "blind_midpoint_frequency": BLIND_MIDPOINT_FREQUENCY,
            "growth_requires_blind_midpoint_pass": True,
            "numerical_rejection_halves_span": True,
            "original_physical_or_metric_chart_failure_stops_immediately": True,
            "failed_candidate_is_never_propagated": True,
        },
        "scope": {
            "initial_elapsed_seconds": INITIAL_ELAPSED_SECONDS,
            "maximum_accepted_segments": MAXIMUM_ACCEPTED_SEGMENTS,
            "maximum_attempted_segments": MAXIMUM_ATTEMPTED_SEGMENTS,
            "maximum_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
            "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
            "maximum_no_rejection_horizon_seconds": cost[
                "maximum_no_rejection_horizon_seconds"
            ],
            "fixed_Q_calls": 0,
            "reaction_calls": 0,
            "nonlinear_roots": 0,
            "BDF_microsteps": 0,
        },
        "gates": {
            "original_coordinate_residual_tolerance": ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE,
            "metric_coordinate_residual_tolerance": METRIC_COORDINATE_RESIDUAL_TOLERANCE,
            "gauge_residual_tolerance": GAUGE_RESIDUAL_TOLERANCE,
            "maximum_metric_jacobian_condition": MAXIMUM_METRIC_JACOBIAN_CONDITION,
            "maximum_metric_augmented_condition": MAXIMUM_METRIC_AUGMENTED_CONDITION,
            "maximum_patch_transition_condition": MAXIMUM_PATCH_TRANSITION_CONDITION,
            "maximum_transform_inverse_closure": MAXIMUM_TRANSFORM_INVERSE_CLOSURE,
            "maximum_coordinate_reconstruction_defect": MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT,
            "maximum_endpoint_integral_defect": MAXIMUM_ENDPOINT_INTEGRAL_DEFECT,
            "maximum_blind_midpoint_rate_defect": MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT,
            "rank": 470,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_height_ratio": 0.5,
            "minimum_scattering_optical_depth": 1.0,
            "all_reaction_free_ledgers": True,
            "checkpoint_roundtrip_bitwise": True,
            "suffix_history_replay_bitwise": True,
        },
        "events": {
            "poincare_section": "original committed section in original q",
            "cycle_requires_negative_then_positive_crossing": True,
            "positive_orientation_required": True,
            "maximum_hidden_return_defect": CYCLE_HIDDEN_RETURN_DEFECT_MAXIMUM,
            "equilibrium_speed_ratio": EQUILIBRIUM_SPEED_RATIO,
        },
        "cost_projection": cost,
        "decision": {
            "cycle": "metric_wide_resume_cycle_observed_matched_path_manifest_authorized",
            "equilibrium": "metric_wide_resume_equilibrium_candidate_requires_certificate",
            "clean_budget": "metric_wide_resume_clean_budget_exhausted_next_tranche_manifest_authorized",
            "physical_failure": "metric_wide_resume_original_physical_gate_failed",
            "numerical_failure": "metric_wide_resume_numerical_or_restart_failed",
        },
        "forbidden": [
            "evaluate physics or ledgers in metric coordinates",
            "increase span above 2 ms",
            "retry a genuine original physical or metric-chart failure",
            "propagate a rejected candidate",
            "authorize reduced slow evolution directly",
        ],
    }


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": "SUPPORTED",
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
        "passed": True,
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


def _freeze() -> dict:
    helper = _helper()
    lock = _validate_parent(require_clean=True)
    seed = _seed()
    cost = _cost_projection()
    if not cost["cost_gate_passed"]:
        raise RuntimeError("wide-resume cost projection failed")
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("wide-resume manifest already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "resume_contract.json", _contract())
    helper._write_json(CANONICAL_DIRECTORY / "cost_projection.json", cost)
    _save_npz(CANONICAL_DIRECTORY / "resume_seed.npz", seed)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", lock)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "metric_chart_wide_resume_authorized": True,
        "metric_chart_wide_resume_executed": False,
        "new_trajectory": False,
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "definition_commit": helper._git("rev-parse", "HEAD"),
        "definition_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {
            THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER),
            THIS_TEST: helper._sha(ROOT / THIS_TEST),
            parent.THIS_RUNNER: helper._sha(ROOT / parent.THIS_RUNNER),
            parent.parent.ATLAS_SOURCE: helper._sha(ROOT / parent.parent.ATLAS_SOURCE),
            parent.execution.source.THIS_RUNNER: helper._sha(
                ROOT / parent.execution.source.THIS_RUNNER
            ),
        },
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
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Metric-chart wide-continuation resume manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "This definitions-only package resumes the autonomous original free-field path from 112.50 ms with fresh conservative metric patches. It grows from 0.5 ms to at most 2 ms only after four accepted segments ending in a blind midpoint pass.",
            "",
            f"The tranche is capped at `{MAXIMUM_ACCEPTED_SEGMENTS}` accepted segments, `{MAXIMUM_EXACT_FREE_FIELD_CALLS}` exact calls, and `{MAXIMUM_EXECUTION_WALL_HOURS}` wall-hours. The suffix-calibrated reserved projection is `{cost['reserved_projected_wall_hours']:.3f}` hours for up to `{cost['maximum_no_rejection_horizon_seconds']:.3f}` s of new physical time.",
            "",
            f"Authorized next artifact: `{AUTHORIZED_NEXT}`.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("--freeze is required")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
