#!/usr/bin/env python3
"""Freeze a bounded orientation-based cycle-readiness discrimination tranche."""

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

import run_causal_inner_adaptive_metric_chart_cycle_readiness_execution_wp10c9d6c7c3b5c4f25fir as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fis"
CLASSIFICATION = (
    "adaptive_metric_chart_cycle_readiness_reforecast_manifest_frozen"
)
AUTHORIZED_BY = (
    f"{WORK_PACKAGE}_adaptive_metric_chart_cycle_readiness_reforecast_manifest"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fit_"
    "adaptive_metric_chart_cycle_readiness_discrimination_execution"
)
ARTIFACT = (
    "causal_inner_adaptive_metric_chart_cycle_readiness_reforecast_manifest_"
    "wp10c9d6c7c3b5c4f25fis"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_METRIC_CHART_"
    "CYCLE_READINESS_REFORECAST_MANIFEST_WP10C9D6C7C3B5C4F25FIS_"
    "2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_adaptive_metric_chart_cycle_readiness_"
    "reforecast_manifest_wp10c9d6c7c3b5c4f25fis.py"
)
THIS_TEST = (
    "tests/test_causal_inner_adaptive_metric_chart_cycle_readiness_"
    "reforecast_manifest_wp10c9d6c7c3b5c4f25fis.py"
)

INITIAL_ELAPSED_SECONDS = 0.1520000000000001
INITIAL_ACCEPTED_SEGMENTS = 132
INITIAL_SEGMENT_SECONDS = 2.5e-4
MINIMUM_SEGMENT_SECONDS = 1.25e-4
MAXIMUM_SEGMENT_SECONDS = 2.5e-4
GROWTH_FACTOR = 2.0
ACCEPTED_SEGMENTS_BEFORE_GROWTH = 4
BLIND_MIDPOINT_FREQUENCY = 4
MAXIMUM_ACCEPTED_SEGMENTS = 48
MAXIMUM_ATTEMPTED_SEGMENTS = 51
MAXIMUM_EXACT_FREE_FIELD_CALLS = 63
MAXIMUM_RETRACTIONS = 63
MAXIMUM_EXECUTION_WALL_HOURS = 8.0
FORECAST_WINDOWS = (4, 8, 12, 16, 24, 32, 40)

MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = (
    parent.manifest.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
)
MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT = (
    parent.manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
)
MAXIMUM_METRIC_JACOBIAN_CONDITION = (
    parent.manifest.MAXIMUM_METRIC_JACOBIAN_CONDITION
)
MAXIMUM_METRIC_AUGMENTED_CONDITION = (
    parent.manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
)
MAXIMUM_PATCH_TRANSITION_CONDITION = (
    parent.manifest.MAXIMUM_PATCH_TRANSITION_CONDITION
)
MAXIMUM_TRANSFORM_INVERSE_CLOSURE = (
    parent.manifest.MAXIMUM_TRANSFORM_INVERSE_CLOSURE
)
MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT = (
    parent.manifest.MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT
)
ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE = (
    parent.manifest.ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
)
METRIC_COORDINATE_RESIDUAL_TOLERANCE = (
    parent.manifest.METRIC_COORDINATE_RESIDUAL_TOLERANCE
)
GAUGE_RESIDUAL_TOLERANCE = parent.manifest.GAUGE_RESIDUAL_TOLERANCE


def _helper():
    return parent._helper()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "cycle_readiness_metrics.json"
    )
    provenance = helper._read(parent.CANONICAL_DIRECTORY / "provenance.json")
    arrays = _load_npz(
        parent.CANONICAL_DIRECTORY / "cycle_readiness_arrays.npz"
    )
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.OPEN_CLASSIFICATION
        or not summary["passed"]
        or not summary["open_transient_extended"]
        or summary["cycle_readiness_turn_bracketed"]
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or summary["authorized_next"] != AUTHORIZED_BY
        or metrics["classification"] != parent.OPEN_CLASSIFICATION
        or not metrics["passed"]
        or values["accepted_segments"] != 32
        or values["attempted_segments"] != 39
        or values["retryable_chart_failures"] != 7
        or values["exact_free_field_calls"] != 40
        or values["retractions"] != 47
        or values["terminal_elapsed_seconds"] != INITIAL_ELAPSED_SECONDS
        or values["terminal_section_velocity_per_second"] <= 0.0
        or values["section_turning_point_observed"]
        or values["section_negative_observed"]
        or not values["all_accepted_checkpoint_roundtrips_bitwise"]
        or not values["suffix_history_replay_bitwise"]
        or values["minimum_reconstruction_factor"] < 1.0 - 1.0e-12
        or values["maximum_height_ratio"] > 0.5
        or values["minimum_scattering_optical_depth"] < 1.0
        or int(arrays["accepted_segments_total"]) != INITIAL_ACCEPTED_SEGMENTS
        or float(arrays["previous_span_seconds"]) != INITIAL_SEGMENT_SECONDS
        or float(arrays["next_span_seconds"]) != 5.0e-4
        or not np.array_equal(
            arrays["current_coordinate470"],
            arrays["accepted_endpoint_coordinates470"][-1],
        )
        or not np.array_equal(
            arrays["current_primitive_state"],
            arrays["accepted_endpoint_primitive_states"][-1],
        )
    ):
        raise RuntimeError("cycle-readiness open-transient evidence changed")
    for relative, frozen_hash in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != frozen_hash:
            raise RuntimeError(f"parent cycle-readiness source changed: {relative}")
    parent._validate_manifest(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cycle-readiness reforecast requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
        "arrays": arrays,
    }


def _seed() -> dict[str, np.ndarray]:
    arrays = _load_npz(
        parent.CANONICAL_DIRECTORY / "cycle_readiness_arrays.npz"
    )
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
        "metric_transform470x470",
        "metric_augmented560x560",
        "gauge_basis560x90",
        "section_normal470",
        "start_coordinate470",
    )
    seed = {name: np.asarray(arrays[name]) for name in names}
    seed["next_span_seconds"] = np.asarray(INITIAL_SEGMENT_SECONDS)
    seed["accepted_since_growth"] = np.asarray(0)
    if (
        float(seed["elapsed_seconds"]) != INITIAL_ELAPSED_SECONDS
        or int(seed["accepted_segments_total"]) != INITIAL_ACCEPTED_SEGMENTS
        or float(seed["previous_span_seconds"]) != INITIAL_SEGMENT_SECONDS
        or float(seed["next_span_seconds"]) != INITIAL_SEGMENT_SECONDS
        or seed["current_coordinate470"].shape != (470,)
        or seed["current_primitive_state"].shape != (112, 5)
    ):
        raise RuntimeError("cycle-readiness reforecast seed changed")
    return seed


def _trajectory_observations() -> dict[str, np.ndarray]:
    earlier_module = parent.manifest.parent
    earlier = _load_npz(
        earlier_module.CANONICAL_DIRECTORY / "continuation_arrays.npz"
    )
    latest = _load_npz(
        parent.CANONICAL_DIRECTORY / "cycle_readiness_arrays.npz"
    )
    normal = np.asarray(latest["section_normal470"], dtype=float)
    earlier_spans = np.asarray(earlier["accepted_segment_seconds"], dtype=float)
    earlier_times = (
        float(earlier["elapsed_seconds"])
        - float(np.sum(earlier_spans))
        + np.cumsum(earlier_spans)
    )
    earlier_rates = np.asarray(
        earlier["accepted_endpoint_coordinate_rates470_per_s"], dtype=float
    )
    latest_rates = np.asarray(
        latest["accepted_endpoint_coordinate_rates470_per_s"], dtype=float
    )
    times = np.concatenate(
        (
            earlier_times,
            np.asarray(latest["accepted_endpoint_elapsed_seconds"], dtype=float),
        )
    )
    rates = np.concatenate((earlier_rates, latest_rates), axis=0)
    velocities = rates @ normal
    speeds = np.linalg.norm(rates, axis=1)
    orientations = velocities / speeds
    section_values = np.concatenate(
        (
            np.asarray(earlier["new_section_values"], dtype=float),
            np.asarray(latest["accepted_endpoint_section_values"], dtype=float),
        )
    )
    if (
        times.shape != (40,)
        or not np.all(np.diff(times) > 0.0)
        or not np.all(velocities > 0.0)
        or not np.all(orientations > 0.0)
        or not np.all(np.diff(velocities) < 0.0)
    ):
        raise RuntimeError("cycle-readiness trajectory observations changed")
    return {
        "times_seconds": times,
        "section_values": section_values,
        "section_velocities_per_second": velocities,
        "speeds_per_second": speeds,
        "orientation_cosines": orientations,
    }


def _linear_forecasts(times: np.ndarray, values: np.ndarray) -> dict:
    forecasts = {}
    for count in FORECAST_WINDOWS:
        slope, intercept = np.polyfit(times[-count:], values[-count:], 1)
        zero_time = float(-intercept / slope) if slope < 0.0 else None
        prediction = slope * times[-count:] + intercept
        forecasts[str(count)] = {
            "sample_count": count,
            "slope_per_second": float(slope),
            "forecast_zero_time_seconds": zero_time,
            "forecast_additional_seconds": (
                None if zero_time is None else zero_time - float(times[-1])
            ),
            "root_mean_square_fit_defect": float(
                np.sqrt(np.mean((prediction - values[-count:]) ** 2))
            ),
        }
    return forecasts


def _geometry_reforecast() -> dict:
    observations = _trajectory_observations()
    times = observations["times_seconds"]
    velocities = observations["section_velocities_per_second"]
    speeds = observations["speeds_per_second"]
    orientations = observations["orientation_cosines"]
    velocity_forecasts = _linear_forecasts(times, velocities)
    orientation_forecasts = _linear_forecasts(times, orientations)
    velocity_zeros = np.asarray(
        [item["forecast_zero_time_seconds"] for item in velocity_forecasts.values()]
    )
    orientation_zeros = np.asarray(
        [item["forecast_zero_time_seconds"] for item in orientation_forecasts.values()]
    )
    secant_accelerations = np.diff(velocities) / np.diff(times)
    acceleration_changes = np.diff(secant_accelerations)
    relaxation_run = 0
    for change in acceleration_changes[::-1]:
        if change > 0.0:
            relaxation_run += 1
        else:
            break
    terminal_orientation = float(orientations[-1])
    return {
        "truth_field_autonomous": True,
        "sample_count": int(times.size),
        "initial_observation_seconds": float(times[0]),
        "terminal_observation_seconds": float(times[-1]),
        "section_turning_point_observed": False,
        "section_negative_observed": False,
        "section_velocity_strictly_decreasing": True,
        "terminal_section_value": float(observations["section_values"][-1]),
        "terminal_section_velocity_per_second": float(velocities[-1]),
        "terminal_speed_per_second": float(speeds[-1]),
        "terminal_orientation_cosine": terminal_orientation,
        "terminal_orientation_angle_degrees": float(
            np.degrees(np.arccos(terminal_orientation))
        ),
        "minimum_recent_secant_acceleration_per_second2": float(
            np.min(secant_accelerations[-16:])
        ),
        "terminal_secant_acceleration_per_second2": float(
            secant_accelerations[-1]
        ),
        "consecutive_acceleration_relaxation_intervals": relaxation_run,
        "raw_velocity_linear_forecasts": velocity_forecasts,
        "orientation_linear_forecasts": orientation_forecasts,
        "raw_velocity_forecast_range_seconds": [
            float(np.min(velocity_zeros)),
            float(np.max(velocity_zeros)),
        ],
        "orientation_forecast_range_seconds": [
            float(np.min(orientation_zeros)),
            float(np.max(orientation_zeros)),
        ],
        "raw_velocity_forecast_spread_seconds": float(np.ptp(velocity_zeros)),
        "orientation_forecast_spread_seconds": float(np.ptp(orientation_zeros)),
        "event_equivalence": (
            "orientation cosine and section velocity have exactly the same "
            "zero because speed is strictly positive"
        ),
        "interpretation": (
            "the raw-velocity forecast is confounded by growing total speed; "
            "the normalized orientation forecast is diagnostic only and the "
            "observed sign change remains binding"
        ),
        "times_seconds": times.tolist(),
        "section_velocities_per_second": velocities.tolist(),
        "speeds_per_second": speeds.tolist(),
        "orientation_cosines": orientations.tolist(),
    }


def _cost_projection() -> dict:
    helper = _helper()
    latest = helper._read(
        parent.CANONICAL_DIRECTORY / "cycle_readiness_metrics.json"
    )["gate_values"]
    prior = helper._read(
        parent.manifest.CANONICAL_DIRECTORY / "cost_projection.json"
    )
    actual_seconds_per_acquisition = float(latest["execution_wall_seconds"]) / (
        int(latest["exact_free_field_calls"]) + int(latest["retractions"])
    )
    binding_seconds_per_acquisition = max(
        actual_seconds_per_acquisition,
        float(prior["binding_seconds_per_field_or_retraction"]),
    )
    no_rejection_fields = MAXIMUM_ACCEPTED_SEGMENTS + (
        MAXIMUM_ACCEPTED_SEGMENTS // BLIND_MIDPOINT_FREQUENCY
    )
    no_rejection_retractions = no_rejection_fields
    maximum_acquisitions = MAXIMUM_EXACT_FREE_FIELD_CALLS + MAXIMUM_RETRACTIONS
    maximum_projected_wall_hours = (
        maximum_acquisitions * binding_seconds_per_acquisition / 3600.0
    )
    return {
        "latest_actual_seconds_per_field_or_retraction": actual_seconds_per_acquisition,
        "worst_observed_seconds_per_field_or_retraction": binding_seconds_per_acquisition,
        "no_rejection_exact_free_field_calls": no_rejection_fields,
        "no_rejection_retractions": no_rejection_retractions,
        "actual_rate_no_rejection_projected_wall_hours": (
            (no_rejection_fields + no_rejection_retractions)
            * actual_seconds_per_acquisition
            / 3600.0
        ),
        "worst_observed_no_rejection_projected_wall_hours": (
            (no_rejection_fields + no_rejection_retractions)
            * binding_seconds_per_acquisition
            / 3600.0
        ),
        "maximum_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
        "maximum_retractions": MAXIMUM_RETRACTIONS,
        "maximum_projected_wall_hours": maximum_projected_wall_hours,
        "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
        "cost_gate_passed": maximum_projected_wall_hours <= MAXIMUM_EXECUTION_WALL_HOURS,
        "complete_cycle_runtime_identifiable": False,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "authorized_execution": AUTHORIZED_NEXT,
        "truth_system": {
            "field": "autonomous original reaction-free field dy/dt=f_free(y)",
            "state": "original primitive state u",
            "physical_coordinate": "original q=C(u)",
            "chart": "fresh strict conservative metric patch at every accepted endpoint",
            "fixed_Q_rate_or_reaction": "forbidden",
        },
        "adaptive_policy": {
            "initial_segment_seconds": INITIAL_SEGMENT_SECONDS,
            "minimum_segment_seconds": MINIMUM_SEGMENT_SECONDS,
            "maximum_segment_seconds": MAXIMUM_SEGMENT_SECONDS,
            "growth_factor": GROWTH_FACTOR,
            "accepted_segments_before_growth": ACCEPTED_SEGMENTS_BEFORE_GROWTH,
            "blind_midpoint_frequency": BLIND_MIDPOINT_FREQUENCY,
            "maximum_span_frozen_from_three_consecutive_rejected_growth_probes": True,
            "physically_admissible_chart_failure_halves_span": True,
            "endpoint_or_midpoint_numerical_failure_halves_span": True,
            "physical_failure_stops": True,
            "minimum_span_failure_stops": True,
            "rejected_candidate_is_never_propagated": True,
        },
        "scope": {
            "initial_elapsed_seconds": INITIAL_ELAPSED_SECONDS,
            "initial_accepted_segments": INITIAL_ACCEPTED_SEGMENTS,
            "maximum_accepted_segments": MAXIMUM_ACCEPTED_SEGMENTS,
            "maximum_attempted_segments": MAXIMUM_ATTEMPTED_SEGMENTS,
            "maximum_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
            "maximum_retractions": MAXIMUM_RETRACTIONS,
            "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
            "minimum_new_horizon_seconds": (
                MAXIMUM_ACCEPTED_SEGMENTS * MINIMUM_SEGMENT_SECONDS
            ),
            "nominal_new_horizon_seconds": (
                MAXIMUM_ACCEPTED_SEGMENTS * INITIAL_SEGMENT_SECONDS
            ),
            "fixed_Q_calls": 0,
            "reaction_calls": 0,
            "nonlinear_roots": 0,
            "BDF_microsteps": 0,
        },
        "forecast_discrimination": {
            "binding_event": "accepted section velocity changes from positive to nonpositive",
            "normalized_diagnostic": (
                "orientation cosine=(n0 dot f)/norm(f); same zero as binding event"
            ),
            "raw_velocity_extrapolation_binding": False,
            "orientation_extrapolation_binding": False,
            "record_secant_acceleration": True,
            "record_orientation_cosine": True,
            "record_speed": True,
            "turn_may_only_be_declared_from_observed_accepted_sign_change": True,
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
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_height_ratio": 0.5,
            "minimum_scattering_optical_depth": 1.0,
            "all_reaction_free_ledgers": True,
            "all_accepted_checkpoints_roundtrip_bitwise": True,
            "suffix_history_replay_bitwise": True,
        },
        "outcomes": {
            "turn_bracketed": (
                "authorize only a definitions-only negative-section acquisition manifest"
            ),
            "open_transient": (
                "update orientation and raw-velocity forecasts before another tranche"
            ),
            "physical_failure": "original physical gate failed",
            "numerical_failure": "minimum-span, replay, acquisition, or wall gate failed",
        },
        "forbidden": [
            "authorize a complete cycle before a negative section is observed",
            "infer a turn from either forecast",
            "propagate a rejected endpoint or midpoint",
            "increase span above 0.25 ms in this tranche",
            "alter or lag the original free field",
            "invoke a fixed-Q physical clock or reaction",
            "authorize reduced slow evolution",
        ],
    }


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        parent.manifest.THIS_RUNNER,
        parent.manifest.THIS_TEST,
        parent.manifest.parent.THIS_RUNNER,
        parent.manifest.parent.THIS_TEST,
        parent.manifest.parent.manifest.POLICY_SOURCE,
        parent.manifest.parent.manifest.parent.diagnosis.manifest.STRICT_ATLAS_SOURCE,
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _update_catalog() -> None:
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
                    "scientific_status": "SUPPORTED",
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
        "classification": CLASSIFICATION,
        "passed": True,
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


def _freeze() -> dict:
    helper = _helper()
    lock = _validate_parent(require_clean=True)
    geometry = _geometry_reforecast()
    cost = _cost_projection()
    if not cost["cost_gate_passed"]:
        raise RuntimeError("cycle-readiness reforecast exceeds frozen wall budget")
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("cycle-readiness reforecast manifest already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(
        CANONICAL_DIRECTORY / "forecast_discrimination_contract.json",
        _contract(),
    )
    with (CANONICAL_DIRECTORY / "continuation_seed.npz").open("wb") as handle:
        np.savez_compressed(handle, **_seed())
    helper._write_json(
        CANONICAL_DIRECTORY / "geometry_reforecast.json", geometry
    )
    helper._write_json(CANONICAL_DIRECTORY / "cost_projection.json", cost)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_hashes": lock["hashes"],
            "parent_classification": lock["summary"]["classification"],
            "terminal_span_override": {
                "parent_proposed_next_span_seconds": 5.0e-4,
                "frozen_next_span_seconds": INITIAL_SEGMENT_SECONDS,
                "reason": (
                    "three consecutive accepted-streak growth probes at 0.5 ms "
                    "failed the strict chart condition and were never propagated"
                ),
            },
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "forecast_discrimination_execution_authorized": True,
        "forecast_discrimination_execution_executed": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "definition_commit": helper._git("rev-parse", "HEAD"),
            "definition_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": _source_hashes(),
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
    velocity_range = geometry["raw_velocity_forecast_range_seconds"]
    orientation_range = geometry["orientation_forecast_range_seconds"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Adaptive metric-chart cycle-readiness reforecast manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The exact original-free-field trajectory is certified through 152.0 ms. The section and section velocity remain positive, so no turn, negative section, or cycle is established.",
                "",
                f"Raw section-velocity linear windows forecast a zero over `{velocity_range[0]:.6f}`--`{velocity_range[1]:.6f}` s. Normalizing by the strictly positive speed gives the equivalent orientation cosine, whose linear windows forecast `{orientation_range[0]:.6f}`--`{orientation_range[1]:.6f}` s. Both forecasts are diagnostic; only an observed accepted sign change is binding.",
                "",
                f"The terminal orientation is `{geometry['terminal_orientation_cosine']:.6e}` (angle `{geometry['terminal_orientation_angle_degrees']:.3f}` degrees). The final secant acceleration is `{geometry['terminal_secant_acceleration_per_second2']:.3f}` per second squared after `{geometry['consecutive_acceleration_relaxation_intervals']}` consecutive relaxation intervals, so constant-acceleration extrapolation is rejected.",
                "",
                f"The next tranche fixes the span ceiling at 0.25 ms, permits `{MAXIMUM_ACCEPTED_SEGMENTS}` accepted segments and a nominal 12 ms horizon, and caps fields/retractions at `{MAXIMUM_EXACT_FREE_FIELD_CALLS}` / `{MAXIMUM_RETRACTIONS}`. Projected wall time is `{cost['actual_rate_no_rejection_projected_wall_hours']:.3f}` hours at the latest measured rate and `{cost['maximum_projected_wall_hours']:.3f}` hours at the worst observed acquisition rate.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`. Complete-cycle execution and reduced slow evolution remain unauthorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog()
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
