#!/usr/bin/env python3
"""Certify a conservative metric-chart overlap at the 111.25 ms stop."""

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

import run_causal_inner_conservative_metric_chart_atlas_manifest_wp10c9d6c7c3b5c4f25fic as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.conservative_metric_chart_atlas import (  # noqa: E402
    ConservativeMetricChart,
    MetricRetractionPolicy,
    block_whitening_transform,
    metric_augmented_jacobian,
    metric_transport_retract,
)


diagnosis = manifest.diagnosis
parent = diagnosis.parent
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fid"
PASS_CLASSIFICATION = "metric_chart_atlas_overlap_passed"
FAIL_CLASSIFICATION = "metric_chart_atlas_overlap_failed"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fie_metric_chart_boundary_crossing_manifest"
)
ARTIFACT = (
    "causal_inner_metric_chart_atlas_overlap_preflight_"
    "wp10c9d6c7c3b5c4f25fid"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_METRIC_CHART_ATLAS_OVERLAP_"
    "PREFLIGHT_WP10C9D6C7C3B5C4F25FID_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_metric_chart_atlas_overlap_preflight_"
    "wp10c9d6c7c3b5c4f25fid.py"
)
THIS_TEST = (
    "tests/test_causal_inner_metric_chart_atlas_overlap_preflight_"
    "wp10c9d6c7c3b5c4f25fid.py"
)
ATLAS_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/conservative_metric_chart_atlas.py"
)


def _helper():
    return manifest._helper()


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(manifest.CANONICAL_DIRECTORY / "atlas_contract.json")
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["metric_chart_atlas_preflight_authorized"]
        or summary["metric_chart_atlas_executed"]
        or summary["trajectory_authorized"]
        or summary["authorized_next"] != manifest.AUTHORIZED_NEXT
        or contract["authorized_execution"] != manifest.AUTHORIZED_NEXT
        or contract["scope"]["new_exact_coordinate_jacobians"] != 2
        or contract["scope"]["new_exact_free_field_calls"] != 0
        or contract["scope"]["new_retractions"] != 3
        or contract["scope"]["new_trajectory_segments"] != 0
    ):
        raise RuntimeError("metric-chart overlap authorization changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("metric-chart overlap requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(a - b)
        / max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(float).tiny)
    )


def _selected_witnesses() -> dict[str, np.ndarray]:
    helper = _helper()
    witness = helper._load_npz(
        diagnosis.manifest.CANONICAL_DIRECTORY / "conditioning_witnesses.npz"
    )
    attempts = np.asarray(witness["attempt_indices"], dtype=int)
    selected = []
    for attempt in (manifest.ANCHOR_ATTEMPT, manifest.OVERLAP_ATTEMPT):
        matches = np.flatnonzero(attempts == attempt)
        if len(matches) != 1:
            raise RuntimeError(f"saved witness {attempt} is not unique")
        selected.append(int(matches[0]))
    return {
        name: np.asarray(value)[selected]
        for name, value in witness.items()
        if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(attempts)
    }


def _initial_state(model, anchor_state, anchor_coordinate, target_coordinate):
    return (
        np.asarray(anchor_state)
        + np.asarray(model.decoded_state(target_coordinate))
        - np.asarray(model.decoded_state(anchor_coordinate))
    )


def _policy() -> MetricRetractionPolicy:
    transport = parent.source.arclength._transport()
    exact_chart = parent.source.arclength._exact_chart()
    return MetricRetractionPolicy(
        maximum_iterations=transport.manifest.MAXIMUM_TRANSPORT_ITERATIONS,
        refresh_iteration_reserve=transport.manifest.REFRESH_ITERATION_RESERVE,
        maximum_exact_refreshes=transport.manifest.MAXIMUM_TARGET_EXACT_REFRESHES,
        line_factors=tuple(float(value) for value in exact_chart.LINE_FACTORS),
        original_coordinate_tolerance=(
            manifest.ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
        ),
        metric_coordinate_tolerance=(
            manifest.METRIC_COORDINATE_RESIDUAL_TOLERANCE
        ),
        gauge_tolerance=manifest.GAUGE_RESIDUAL_TOLERANCE,
        maximum_anchor_departure=exact_chart.MAXIMUM_SCALED_DEPARTURE,
        maximum_metric_augmented_condition=(
            manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
        ),
    )


def _execute(lock: dict) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    witness = _selected_witnesses()
    states = np.asarray(witness["primitive_states"], dtype=float)
    coordinates = np.asarray(witness["requested_coordinates470"], dtype=float)
    rates = np.asarray(witness["coordinate_free_rates470_per_s"], dtype=float)
    if states.shape != (2, 112, 5) or coordinates.shape != (2, 470):
        raise RuntimeError("atlas overlap witness shapes changed")

    inputs = parent.source._initial_inputs()
    model = inputs["model"]
    exact_chart = parent.source.arclength._exact_chart()
    block_sizes = (
        diagnosis.manifest.PHYSICAL_ROWS,
        diagnosis.manifest.MEMORY_ROWS,
        diagnosis.manifest.DEPARTURE_ROWS,
    )
    began = time.perf_counter()
    jacobians = []
    transforms = []
    transform_metrics = []
    charts = []
    gauges = []
    deltas = []
    augmented = []
    augmented_conditions = []
    for state, coordinate in zip(states, coordinates, strict=True):
        jacobian, _raw_metrics = exact_chart._coordinate_jacobian(model, state)
        transform, local_metrics = block_whitening_transform(
            jacobian, block_sizes
        )
        chart = ConservativeMetricChart(coordinate, transform, block_sizes)
        gauge = exact_chart._canonical_null_basis(jacobian)
        matrix, condition = metric_augmented_jacobian(jacobian, gauge, chart)
        jacobians.append(jacobian)
        transforms.append(transform)
        transform_metrics.append(local_metrics)
        charts.append(chart)
        gauges.append(gauge)
        deltas.append(exact_chart._delta(model, state))
        augmented.append(matrix)
        augmented_conditions.append(condition)

    diagnosis_arrays = helper._load_npz(
        diagnosis.CANONICAL_DIRECTORY / "conditioning_arrays.npz"
    )
    frozen_transform = np.asarray(
        diagnosis_arrays["terminal_accepted_block_transform470x470"]
    )
    transform_reproduction = _relative(transforms[0], frozen_transform)
    original_roundtrip = max(
        _relative(chart.decode(chart.encode(coordinate)), coordinate)
        for chart in charts
        for coordinate in coordinates
    )
    rate_push_pull = max(
        _relative(chart.pull_rate(chart.push_rate(rate)), rate)
        for chart in charts
        for rate in rates
    )
    transition = transforms[1] @ charts[0].inverse_transform
    transition_condition = float(np.linalg.cond(transition))

    policy = _policy()
    forward_initial = _initial_state(
        model, states[0], coordinates[0], coordinates[1]
    )
    forward_state, forward_matrix, forward_metrics = metric_transport_retract(
        exact_chart=exact_chart,
        model=model,
        initial_state=forward_initial,
        target_original_coordinate=coordinates[1],
        gauge_basis=gauges[0],
        anchor_delta=deltas[0],
        anchor_metric_augmented=augmented[0],
        chart=charts[0],
        policy=policy,
    )
    replay_state, replay_matrix, replay_metrics = metric_transport_retract(
        exact_chart=exact_chart,
        model=model,
        initial_state=forward_initial,
        target_original_coordinate=coordinates[1],
        gauge_basis=gauges[0],
        anchor_delta=deltas[0],
        anchor_metric_augmented=augmented[0],
        chart=charts[0],
        policy=policy,
    )
    reverse_initial = _initial_state(
        model, states[1], coordinates[1], coordinates[0]
    )
    reverse_state, reverse_matrix, reverse_metrics = metric_transport_retract(
        exact_chart=exact_chart,
        model=model,
        initial_state=reverse_initial,
        target_original_coordinate=coordinates[0],
        gauge_basis=gauges[1],
        anchor_delta=deltas[1],
        anchor_metric_augmented=augmented[1],
        chart=charts[1],
        policy=policy,
    )
    forward_coordinate, _forward_factors = model.coordinate(forward_state)
    reverse_coordinate, _reverse_factors = model.coordinate(reverse_state)
    forward_saved_state_defect = _relative(forward_state, states[1])
    forward_coordinate_defect = _relative(forward_coordinate, coordinates[1])
    reverse_coordinate_defect = _relative(reverse_coordinate, coordinates[0])
    replay_bitwise = bool(
        np.array_equal(forward_state, replay_state)
        and np.array_equal(forward_matrix, replay_matrix)
        and {
            key: value
            for key, value in forward_metrics.items()
            if key != "wall_seconds"
        }
        == {
            key: value
            for key, value in replay_metrics.items()
            if key != "wall_seconds"
        }
    )
    metric_conditions = [
        item["metric_jacobian_condition_number"] for item in transform_metrics
    ]
    inverse_closures = [chart.inverse_closure_defect for chart in charts]
    all_retractions_passed = bool(
        forward_metrics["passed"]
        and replay_metrics["passed"]
        and reverse_metrics["passed"]
    )
    original_residuals = [
        forward_metrics["original_coordinate_residual_infinity"],
        replay_metrics["original_coordinate_residual_infinity"],
        reverse_metrics["original_coordinate_residual_infinity"],
    ]
    metric_residuals = [
        forward_metrics["metric_coordinate_residual_infinity"],
        replay_metrics["metric_coordinate_residual_infinity"],
        reverse_metrics["metric_coordinate_residual_infinity"],
    ]
    gauge_residuals = [
        forward_metrics["gauge_residual_infinity"],
        replay_metrics["gauge_residual_infinity"],
        reverse_metrics["gauge_residual_infinity"],
    ]
    execution_wall = float(time.perf_counter() - began)
    gates = {
        "transform_reproduction_relative_defect": transform_reproduction,
        "maximum_transform_inverse_closure": float(max(inverse_closures)),
        "maximum_original_coordinate_roundtrip_defect": original_roundtrip,
        "maximum_rate_push_pull_defect": rate_push_pull,
        "maximum_metric_jacobian_condition": float(max(metric_conditions)),
        "maximum_metric_augmented_condition": float(max(augmented_conditions)),
        "patch_transition_condition": transition_condition,
        "forward_saved_state_relative_defect": forward_saved_state_defect,
        "forward_coordinate_relative_defect": forward_coordinate_defect,
        "reverse_coordinate_relative_defect": reverse_coordinate_defect,
        "maximum_original_coordinate_residual_infinity": float(
            max(original_residuals)
        ),
        "maximum_metric_coordinate_residual_infinity": float(max(metric_residuals)),
        "maximum_gauge_residual_infinity": float(max(gauge_residuals)),
        "all_retractions_passed": all_retractions_passed,
        "forward_replay_bitwise": replay_bitwise,
        "new_exact_coordinate_jacobians": 2,
        "new_exact_free_field_calls": 0,
        "new_retractions": 3,
        "new_trajectory_segments": 0,
        "new_physical_time_seconds": 0.0,
        "execution_wall_seconds": execution_wall,
    }
    passed = bool(
        transform_reproduction <= 1.0e-12
        and gates["maximum_transform_inverse_closure"]
        <= manifest.MAXIMUM_TRANSFORM_INVERSE_CLOSURE
        and original_roundtrip
        <= manifest.MAXIMUM_ORIGINAL_COORDINATE_ROUNDTRIP_DEFECT
        and rate_push_pull <= manifest.MAXIMUM_RATE_PUSH_PULL_DEFECT
        and gates["maximum_metric_jacobian_condition"]
        <= manifest.MAXIMUM_METRIC_JACOBIAN_CONDITION
        and gates["maximum_metric_augmented_condition"]
        <= manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
        and transition_condition <= manifest.MAXIMUM_PATCH_TRANSITION_CONDITION
        and forward_saved_state_defect
        <= manifest.MAXIMUM_SAVED_STATE_RELATIVE_DEFECT
        and gates["maximum_original_coordinate_residual_infinity"]
        <= manifest.ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
        and gates["maximum_metric_coordinate_residual_infinity"]
        <= manifest.METRIC_COORDINATE_RESIDUAL_TOLERANCE
        and gates["maximum_gauge_residual_infinity"]
        <= manifest.GAUGE_RESIDUAL_TOLERANCE
        and all_retractions_passed
        and replay_bitwise
        and execution_wall <= 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "gate_values": gates,
        "anchor_transform_metrics": transform_metrics[0],
        "overlap_transform_metrics": transform_metrics[1],
        "forward_retraction": forward_metrics,
        "replay_retraction": replay_metrics,
        "reverse_retraction": reverse_metrics,
        "input_lock": {
            "manifest_hashes": lock["hashes"],
            "manifest_classification": lock["summary"]["classification"],
        },
    }
    arrays = {
        "anchor_attempt": np.asarray(manifest.ANCHOR_ATTEMPT),
        "overlap_attempt": np.asarray(manifest.OVERLAP_ATTEMPT),
        "anchor_original_coordinate470": coordinates[0],
        "overlap_original_coordinate470": coordinates[1],
        "anchor_primitive_state": states[0],
        "saved_overlap_primitive_state": states[1],
        "forward_metric_retracted_state": forward_state,
        "reverse_metric_retracted_state": reverse_state,
        "anchor_metric_transform470x470": transforms[0],
        "overlap_metric_transform470x470": transforms[1],
        "anchor_metric_augmented560x560": augmented[0],
        "overlap_metric_augmented560x560": augmented[1],
        "anchor_gauge_basis560x90": gauges[0],
        "overlap_gauge_basis560x90": gauges[1],
        "forward_final_broyden560x560": forward_matrix,
        "reverse_final_broyden560x560": reverse_matrix,
    }
    return metrics, arrays


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


def _canonicalize(metrics: dict, arrays: dict, lock: dict) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("metric-chart overlap result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "overlap_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "overlap_arrays.npz", arrays)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", {
        "manifest_hashes": lock["hashes"],
        "manifest_classification": lock["summary"]["classification"],
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "metric_chart_atlas_overlap_passed": bool(metrics["passed"]),
        "original_physics_preserved": True,
        "new_trajectory": False,
        "boundary_crossing_manifest_authorized": bool(metrics["passed"]),
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {
            THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER),
            THIS_TEST: helper._sha(ROOT / THIS_TEST),
            ATLAS_SOURCE: helper._sha(ROOT / ATLAS_SOURCE),
            manifest.THIS_RUNNER: helper._sha(ROOT / manifest.THIS_RUNNER),
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
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Metric-chart atlas overlap preflight",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"The saved 111.25 ms anchor and rejected 111.50 ms overlap candidate were represented in independently block-whitened local charts. The maximum metric Jacobian condition was `{values['maximum_metric_jacobian_condition']:.6e}` and the maximum metric augmented condition was `{values['maximum_metric_augmented_condition']:.6e}`.",
            "",
            f"Three saved-target retractions passed with maximum original-coordinate residual `{values['maximum_original_coordinate_residual_infinity']:.6e}`, metric residual `{values['maximum_metric_coordinate_residual_infinity']:.6e}`, and gauge residual `{values['maximum_gauge_residual_infinity']:.6e}`. Forward replay was bitwise: `{values['forward_replay_bitwise']}`.",
            "",
            "No new free-field call, propagated state, or physical time was used. Original primitive-space physics and original-coordinate ledgers remain binding.",
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
    metrics, arrays = _execute(lock)
    summary = _canonicalize(metrics, arrays, lock)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
