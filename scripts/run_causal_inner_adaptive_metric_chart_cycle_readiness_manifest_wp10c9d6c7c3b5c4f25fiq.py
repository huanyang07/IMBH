#!/usr/bin/env python3
"""Freeze the next bounded exact-field tranche toward cycle readiness."""

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

import run_causal_inner_adaptive_metric_chart_continuation_execution_wp10c9d6c7c3b5c4f25fip as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fiq"
CLASSIFICATION = "adaptive_metric_chart_cycle_readiness_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fir_adaptive_metric_chart_cycle_readiness_execution"
)
ARTIFACT = (
    "causal_inner_adaptive_metric_chart_cycle_readiness_manifest_"
    "wp10c9d6c7c3b5c4f25fiq"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_METRIC_CHART_"
    "CYCLE_READINESS_MANIFEST_WP10C9D6C7C3B5C4F25FIQ_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_adaptive_metric_chart_cycle_readiness_manifest_"
    "wp10c9d6c7c3b5c4f25fiq.py"
)
THIS_TEST = (
    "tests/test_causal_inner_adaptive_metric_chart_cycle_readiness_manifest_"
    "wp10c9d6c7c3b5c4f25fiq.py"
)

INITIAL_ELAPSED_SECONDS = 0.1405000000000001
INITIAL_ACCEPTED_SEGMENTS = 100
INITIAL_SEGMENT_SECONDS = 5.0e-4
MINIMUM_SEGMENT_SECONDS = 1.25e-4
MAXIMUM_SEGMENT_SECONDS = 1.0e-3
GROWTH_FACTOR = 2.0
ACCEPTED_SEGMENTS_BEFORE_GROWTH = 4
BLIND_MIDPOINT_FREQUENCY = 4
MAXIMUM_ACCEPTED_SEGMENTS = 32
MAXIMUM_ATTEMPTED_SEGMENTS = 40
MAXIMUM_EXACT_FREE_FIELD_CALLS = 48
MAXIMUM_RETRACTIONS = 56
MAXIMUM_EXECUTION_WALL_HOURS = 8.0
COST_RESERVE_FACTOR = 1.2
MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = parent.manifest.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
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


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "continuation_metrics.json"
    )
    provenance = helper._read(parent.CANONICAL_DIRECTORY / "provenance.json")
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["adaptive_metric_chart_continuation_passed"]
        or summary["new_accepted_segments"] != 8
        or summary["terminal_elapsed_seconds"] != INITIAL_ELAPSED_SECONDS
        or summary["authorized_next"] != parent.AUTHORIZED_NEXT
        or summary["cycle_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or metrics["classification"] != parent.PASS_CLASSIFICATION
        or not metrics["passed"]
        or values["accepted_segments"] != 8
        or values["attempted_segments"] != 10
        or values["retryable_chart_failures"] != 2
        or values["terminal_elapsed_seconds"] != INITIAL_ELAPSED_SECONDS
        or values["exact_free_field_calls"] != 10
        or values["retractions"] != 12
        or not values["all_accepted_checkpoint_roundtrips_bitwise"]
        or not values["suffix_history_replay_bitwise"]
        or values["minimum_reconstruction_factor"] < 1.0 - 1.0e-12
        or values["maximum_height_ratio"] > 0.5
        or values["minimum_scattering_optical_depth"] < 1.0
    ):
        raise RuntimeError("adaptive metric-chart continuation evidence changed")
    for relative, frozen_hash in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != frozen_hash:
            raise RuntimeError(f"parent continuation source changed: {relative}")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cycle-readiness manifest requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
    }


def _parent_arrays() -> dict[str, np.ndarray]:
    return parent._load_npz(
        parent.CANONICAL_DIRECTORY / "continuation_arrays.npz"
    )


def _seed() -> dict[str, np.ndarray]:
    arrays = _parent_arrays()
    names = (
        "previous_coordinate470",
        "current_coordinate470",
        "previous_primitive_state",
        "current_primitive_state",
        "previous_coordinate_rate470_per_s",
        "current_coordinate_rate470_per_s",
        "previous_span_seconds",
        "next_span_seconds",
        "elapsed_seconds",
        "accepted_segments_total",
        "accepted_since_growth",
        "metric_transform470x470",
        "metric_augmented560x560",
        "gauge_basis560x90",
        "section_normal470",
        "start_coordinate470",
    )
    seed = {name: np.asarray(arrays[name]) for name in names}
    if (
        float(seed["elapsed_seconds"]) != INITIAL_ELAPSED_SECONDS
        or int(seed["accepted_segments_total"]) != INITIAL_ACCEPTED_SEGMENTS
        or float(seed["previous_span_seconds"]) != INITIAL_SEGMENT_SECONDS
        or float(seed["next_span_seconds"]) != INITIAL_SEGMENT_SECONDS
        or int(seed["accepted_since_growth"]) != 2
        or seed["current_coordinate470"].shape != (470,)
        or seed["metric_augmented560x560"].shape != (560, 560)
    ):
        raise RuntimeError("cycle-readiness continuation seed changed")
    return seed


def _geometry_forecast() -> dict:
    arrays = _parent_arrays()
    normal = np.asarray(arrays["section_normal470"])
    start = np.asarray(arrays["start_coordinate470"])
    coordinates = np.asarray(arrays["accepted_endpoint_coordinates470"])
    rates = np.asarray(arrays["accepted_endpoint_coordinate_rates470_per_s"])
    spans = np.asarray(arrays["accepted_segment_seconds"])
    times = (
        INITIAL_ELAPSED_SECONDS - float(np.sum(spans)) + np.cumsum(spans)
    )
    section = (coordinates - start) @ normal
    velocity = rates @ normal
    speed = np.linalg.norm(rates, axis=1)
    forecasts = {}
    for count in (4, 6, 8):
        slope, intercept = np.polyfit(times[-count:], velocity[-count:], 1)
        zero_time = float(-intercept / slope) if slope < 0.0 else None
        forecasts[str(count)] = {
            "sample_count": count,
            "section_acceleration_per_second2": float(slope),
            "forecast_zero_velocity_time_seconds": zero_time,
            "forecast_additional_seconds": (
                None if zero_time is None else zero_time - INITIAL_ELAPSED_SECONDS
            ),
        }
    zero_values = [
        value["forecast_zero_velocity_time_seconds"]
        for value in forecasts.values()
        if value["forecast_zero_velocity_time_seconds"] is not None
    ]
    return {
        "truth_field_autonomous": True,
        "cycle_observed": False,
        "section_negative_observed": False,
        "section_turning_point_observed": False,
        "terminal_section_value": float(section[-1]),
        "terminal_section_velocity_per_second": float(velocity[-1]),
        "terminal_speed_per_second": float(speed[-1]),
        "recent_section_values": section,
        "recent_section_velocities_per_second": velocity,
        "recent_speeds_per_second": speed,
        "linear_zero_velocity_forecasts": forecasts,
        "forecast_spread_seconds": (
            None if not zero_values else max(zero_values) - min(zero_values)
        ),
        "interpretation": (
            "the first section turning point is forecast near 0.240 s, but "
            "neither a negative section nor a cycle return has been observed"
        ),
    }


def _cost_projection() -> dict:
    helper = _helper()
    continuation = helper._read(
        parent.CANONICAL_DIRECTORY / "continuation_metrics.json"
    )["gate_values"]
    recovery_directory = parent.manifest.parent.CANONICAL_DIRECTORY
    recovery = helper._read(recovery_directory / "recovery_metrics.json")[
        "gate_values"
    ]
    continuation_seconds = float(continuation["execution_wall_seconds"]) / (
        int(continuation["exact_free_field_calls"])
        + int(continuation["retractions"])
    )
    recovery_seconds = float(recovery["execution_wall_seconds"]) / 4.0
    seconds_per_acquisition = max(continuation_seconds, recovery_seconds)
    no_rejection_fields = MAXIMUM_ACCEPTED_SEGMENTS + (
        MAXIMUM_ACCEPTED_SEGMENTS // BLIND_MIDPOINT_FREQUENCY
    )
    no_rejection_retractions = no_rejection_fields
    maximum_acquisitions = (
        MAXIMUM_EXACT_FREE_FIELD_CALLS + MAXIMUM_RETRACTIONS
    )
    reserved_hours = (
        maximum_acquisitions
        * seconds_per_acquisition
        * COST_RESERVE_FACTOR
        / 3600.0
    )
    geometry = _geometry_forecast()
    forecast_seconds = np.median(
        [
            item["forecast_additional_seconds"]
            for item in geometry["linear_zero_velocity_forecasts"].values()
        ]
    )
    stable_half_millisecond_acquisitions_per_second = (
        2.0 + 2.0 / BLIND_MIDPOINT_FREQUENCY + 1.0 / BLIND_MIDPOINT_FREQUENCY
    ) / INITIAL_SEGMENT_SECONDS
    return {
        "continuation_wall_seconds_per_field_or_retraction": continuation_seconds,
        "recovery_wall_seconds_per_field_or_retraction": recovery_seconds,
        "binding_seconds_per_field_or_retraction": seconds_per_acquisition,
        "no_rejection_exact_free_field_calls": no_rejection_fields,
        "no_rejection_retractions": no_rejection_retractions,
        "no_rejection_projected_wall_hours": (
            (no_rejection_fields + no_rejection_retractions)
            * seconds_per_acquisition
            / 3600.0
        ),
        "maximum_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
        "maximum_retractions": MAXIMUM_RETRACTIONS,
        "reserve_factor": COST_RESERVE_FACTOR,
        "reserved_projected_wall_hours": reserved_hours,
        "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
        "cost_gate_passed": reserved_hours <= MAXIMUM_EXECUTION_WALL_HOURS,
        "forecast_additional_seconds_to_first_section_turn": float(
            forecast_seconds
        ),
        "diagnostic_projected_wall_hours_to_first_section_turn": float(
            forecast_seconds
            * stable_half_millisecond_acquisitions_per_second
            * seconds_per_acquisition
            / 3600.0
        ),
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
            "role": "offline physical-cycle truth acquisition for a later event-to-event slow map",
            "fixed_Q_rate_or_reaction": "forbidden",
        },
        "adaptive_policy": {
            "initial_segment_seconds": INITIAL_SEGMENT_SECONDS,
            "minimum_segment_seconds": MINIMUM_SEGMENT_SECONDS,
            "maximum_segment_seconds": MAXIMUM_SEGMENT_SECONDS,
            "growth_factor": GROWTH_FACTOR,
            "accepted_segments_before_growth": ACCEPTED_SEGMENTS_BEFORE_GROWTH,
            "blind_midpoint_frequency": BLIND_MIDPOINT_FREQUENCY,
            "growth_requires_blind_midpoint_pass": True,
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
            "initial_span_new_horizon_seconds": (
                MAXIMUM_ACCEPTED_SEGMENTS * INITIAL_SEGMENT_SECONDS
            ),
            "fixed_Q_calls": 0,
            "reaction_calls": 0,
            "nonlinear_roots": 0,
            "BDF_microsteps": 0,
        },
        "cycle_readiness": {
            "section": "n0 dot (q-q0), with n0 the normalized initial physical rate",
            "current_status": "departed positive; section and section velocity remain positive",
            "first_readiness_event": "section velocity changes from positive to nonpositive",
            "cycle_return_event": (
                "only after the section becomes negative, a later negative-to-"
                "nonnegative crossing with positive orientation"
            ),
            "cycle_return_may_not_be_inferred_from_forecast": True,
            "record_each_endpoint": [
                "section value",
                "section velocity",
                "speed",
                "span",
                "chart conditions",
                "original physical ledgers",
            ],
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
                "cycle-readiness turning point bracketed; authorize only a "
                "definitions-only negative-section acquisition manifest"
            ),
            "open_transient": (
                "bounded exact continuation passed but section velocity remains "
                "positive; update forecast and cost before another tranche"
            ),
            "physical_failure": "original physical gate failed",
            "numerical_failure": "minimum-span, replay, or budget gate failed",
        },
        "forbidden": [
            "authorize a complete cycle before a negative section is observed",
            "infer a cycle from a linear turning forecast",
            "treat a physical failure as retryable chart failure",
            "propagate a rejected endpoint or midpoint",
            "increase span above 1 ms",
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
        parent.manifest.POLICY_SOURCE,
        parent.manifest.parent.diagnosis.manifest.STRICT_ATLAS_SOURCE,
    )
    return {name: helper._sha(ROOT / name) for name in paths}


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
    cost = _cost_projection()
    if not cost["cost_gate_passed"]:
        raise RuntimeError("cycle-readiness cost projection exceeded frozen budget")
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("cycle-readiness manifest already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    contract = _contract()
    geometry = _geometry_forecast()
    helper._write_json(
        CANONICAL_DIRECTORY / "cycle_readiness_contract.json", contract
    )
    with (CANONICAL_DIRECTORY / "continuation_seed.npz").open("wb") as handle:
        np.savez_compressed(handle, **_seed())
    helper._write_json(CANONICAL_DIRECTORY / "geometry_forecast.json", geometry)
    helper._write_json(CANONICAL_DIRECTORY / "cost_projection.json", cost)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_hashes": lock["hashes"],
            "parent_classification": lock["summary"]["classification"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "cycle_readiness_execution_authorized": True,
        "cycle_readiness_execution_executed": False,
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
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Adaptive metric-chart cycle-readiness manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The exact autonomous original-free-field trajectory is accepted through 140.50 ms. Its Poincare section and section velocity remain positive; no cycle or negative section has been observed.",
                "",
                f"Recent 4/6/8-point forecasts place the first section-velocity zero near 240 ms with spread `{geometry['forecast_spread_seconds']:.3e}` s. This forecast is diagnostic and cannot establish a cycle.",
                "",
                f"The next tranche permits `{MAXIMUM_ACCEPTED_SEGMENTS}` accepted segments, `{MAXIMUM_EXACT_FREE_FIELD_CALLS}` exact fields, `{MAXIMUM_RETRACTIONS}` retractions, and `{MAXIMUM_EXECUTION_WALL_HOURS}` wall-hours. At the current 0.5 ms span its nominal horizon is 16 ms; the reserved cost is `{cost['reserved_projected_wall_hours']:.3f}` hours.",
                "",
                "A pass may bracket the first section turning point or extend the certified open transient. It may not authorize complete-cycle execution until a negative section is observed, and it never authorizes reduced slow evolution.",
                "",
            )
        ),
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
