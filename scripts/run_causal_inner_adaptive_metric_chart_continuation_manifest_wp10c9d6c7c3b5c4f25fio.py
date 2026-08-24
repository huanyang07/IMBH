#!/usr/bin/env python3
"""Freeze a short repeated adaptive metric-chart continuation tranche."""

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

import run_causal_inner_adaptive_metric_chart_radius_recovery_execution_wp10c9d6c7c3b5c4f25fin as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fio"
CLASSIFICATION = "adaptive_metric_chart_continuation_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fip_adaptive_metric_chart_continuation_execution"
)
ARTIFACT = (
    "causal_inner_adaptive_metric_chart_continuation_manifest_"
    "wp10c9d6c7c3b5c4f25fio"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_METRIC_CHART_"
    "CONTINUATION_MANIFEST_WP10C9D6C7C3B5C4F25FIO_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_adaptive_metric_chart_continuation_manifest_"
    "wp10c9d6c7c3b5c4f25fio.py"
)
THIS_TEST = (
    "tests/test_causal_inner_adaptive_metric_chart_continuation_manifest_"
    "wp10c9d6c7c3b5c4f25fio.py"
)
POLICY_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/adaptive_metric_chart_continuation.py"
)
POLICY_TEST = "tests/test_adaptive_metric_chart_continuation.py"

INITIAL_ELAPSED_SECONDS = parent.manifest.ENDPOINT_ELAPSED_SECONDS
INITIAL_ACCEPTED_SEGMENTS = parent.manifest.EXPECTED_NEXT_TENTATIVE_SEGMENT
INITIAL_SEGMENT_SECONDS = 1.0e-3
MINIMUM_SEGMENT_SECONDS = 2.5e-4
MAXIMUM_SEGMENT_SECONDS = 2.0e-3
GROWTH_FACTOR = 2.0
ACCEPTED_SEGMENTS_BEFORE_GROWTH = 4
BLIND_MIDPOINT_FREQUENCY = 4
MAXIMUM_ACCEPTED_SEGMENTS = 8
MAXIMUM_ATTEMPTED_SEGMENTS = 12
MAXIMUM_EXACT_FREE_FIELD_CALLS = 12
MAXIMUM_RETRACTIONS = 16
MAXIMUM_EXECUTION_WALL_HOURS = 3.0
COST_RESERVE_FACTOR = 1.5
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
    metrics = helper._read(parent.CANONICAL_DIRECTORY / "recovery_metrics.json")
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["adaptive_metric_chart_radius_recovery_passed"]
        or summary["new_accepted_segments"] != 1
        or summary["endpoint_elapsed_seconds"] != INITIAL_ELAPSED_SECONDS
        or summary["authorized_next"] != parent.AUTHORIZED_NEXT
        or metrics["classification"] != parent.PASS_CLASSIFICATION
        or not metrics["passed"]
        or not values["accepted"]
        or values["endpoint_elapsed_seconds"] != INITIAL_ELAPSED_SECONDS
        or values["new_accepted_segments"] != 1
        or not values["endpoint_retraction_passed"]
        or not values["midpoint_retraction_passed"]
        or not values["endpoint_physical_passed"]
        or not values["midpoint_physical_passed"]
        or not values["checkpoint_roundtrip_bitwise"]
        or not values["history_replay_bitwise"]
        or values["exact_free_field_calls"] != 2
    ):
        raise RuntimeError("adaptive chart-radius recovery evidence changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("adaptive continuation manifest requires a clean tracked tree")
    return {"hashes": hashes, "classification": summary["classification"]}


def _seed() -> dict[str, np.ndarray]:
    arrays = _helper()._load_npz(
        parent.CANONICAL_DIRECTORY / "recovery_arrays.npz"
    )
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
        "current_metric_transform470x470",
        "current_metric_augmented560x560",
        "current_gauge_basis560x90",
        "section_normal470",
        "start_coordinate470",
    )
    seed = {name: np.asarray(arrays[name]) for name in names}
    if (
        float(seed["elapsed_seconds"]) != INITIAL_ELAPSED_SECONDS
        or int(seed["accepted_segments_total"]) != INITIAL_ACCEPTED_SEGMENTS
        or float(seed["previous_span_seconds"]) != INITIAL_SEGMENT_SECONDS
        or float(seed["next_span_seconds"]) != INITIAL_SEGMENT_SECONDS
        or int(seed["accepted_since_growth"]) != 0
        or seed["current_coordinate470"].shape != (470,)
        or seed["current_metric_augmented560x560"].shape != (560, 560)
    ):
        raise RuntimeError("adaptive continuation seed changed")
    return seed


def _cost_projection() -> dict:
    values = _helper()._read(
        parent.CANONICAL_DIRECTORY / "recovery_metrics.json"
    )["gate_values"]
    seconds_per_call = float(values["execution_wall_seconds"]) / int(
        values["exact_free_field_calls"]
    )
    projected = (
        MAXIMUM_EXACT_FREE_FIELD_CALLS
        * seconds_per_call
        * COST_RESERVE_FACTOR
        / 3600.0
    )
    return {
        "observed_recovery_wall_seconds_per_exact_call": seconds_per_call,
        "maximum_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
        "reserve_factor": COST_RESERVE_FACTOR,
        "reserved_projected_wall_hours": projected,
        "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
        "cost_gate_passed": projected <= MAXIMUM_EXECUTION_WALL_HOURS,
        "no_rejection_exact_field_calls": 10,
        "rejection_reserve_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS - 10,
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
            "metric_chart": "fresh exact block-whitened patch at every accepted endpoint",
            "strict_retraction_status": "physical AND closure AND chart condition",
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
            "minimum_new_horizon_seconds": 8 * MINIMUM_SEGMENT_SECONDS,
            "maximum_no_rejection_horizon_seconds": 0.012,
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
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_height_ratio": 0.5,
            "minimum_scattering_optical_depth": 1.0,
            "all_reaction_free_ledgers": True,
            "all_accepted_checkpoints_roundtrip_bitwise": True,
            "suffix_history_replay_bitwise": True,
        },
        "cost_projection": cost,
        "decision": {
            "pass": "repeated adaptive metric chart continuation passed",
            "pass_authorizes": "definitions-only next wide or complete-cycle readiness manifest",
            "physical_failure": "original physical gate failed",
            "numerical_failure": "minimum-span, replay, or budget gate failed",
        },
        "forbidden": [
            "treat a physical failure as retryable chart failure",
            "propagate a rejected endpoint or midpoint",
            "grow without four accepts ending in a blind midpoint",
            "increase span above 2 ms",
            "alter the original field or ledgers",
            "authorize complete-cycle execution directly",
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
        "classification": summary["classification"],
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
    seed = _seed()
    cost = _cost_projection()
    if not cost["cost_gate_passed"]:
        raise RuntimeError("adaptive continuation cost projection failed")
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("adaptive continuation manifest already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "continuation_contract.json", _contract())
    helper._write_json(CANONICAL_DIRECTORY / "cost_projection.json", cost)
    _save_npz(CANONICAL_DIRECTORY / "continuation_seed.npz", seed)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", lock)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "adaptive_metric_chart_continuation_authorized": True,
        "adaptive_metric_chart_continuation_executed": False,
        "new_trajectory": False,
        "cycle_authorized": False,
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
            "source_hashes": {
                THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER),
                THIS_TEST: helper._sha(ROOT / THIS_TEST),
                POLICY_SOURCE: helper._sha(ROOT / POLICY_SOURCE),
                POLICY_TEST: helper._sha(ROOT / POLICY_TEST),
                parent.THIS_RUNNER: helper._sha(ROOT / parent.THIS_RUNNER),
                parent.manifest.parent.manifest.STRICT_ATLAS_SOURCE: helper._sha(
                    ROOT / parent.manifest.parent.manifest.STRICT_ATLAS_SOURCE
                ),
            },
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
                "# Adaptive metric-chart continuation manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "This definitions-only package advances at most eight accepted segments from 133.50 ms. Physically admissible strict chart failures and integration-defect failures halve the span; original physical failures stop immediately.",
                "",
                f"The package is capped at `{MAXIMUM_EXACT_FREE_FIELD_CALLS}` exact calls and `{MAXIMUM_EXECUTION_WALL_HOURS}` hours; the reserved projection is `{cost['reserved_projected_wall_hours']:.3f}` hours.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`.",
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
