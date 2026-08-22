#!/usr/bin/env python3
"""Diagnose an arclength phase and hot-regime events from accepted atlas data.

This package performs no new truth evaluation.  It checksum-validates the
five accepted post-transition windows, rewrites their stored nodal data in a
weighted-coordinate arclength parameter, and selects the next prospective
continuation architecture.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imri_qpe.layer3_minidisk_1d.phase_collocation import (  # noqa: E402
    lagrange_differentiation_matrix,
)
import run_causal_inner_exact_retracted_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25f0 as source  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f1"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25f2"
CLASSIFICATION = (
    "weighted_coordinate_arclength_selected_hot_branch_continues_"
    "legacy_exit_not_approached"
)
ARTIFACT = (
    "causal_inner_hot_mode_arclength_event_diagnosis_"
    "wp10c9d6c7c3b5c4f25f1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_hot_mode_arclength_event_diagnosis_"
    "wp10c9d6c7c3b5c4f25f1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_hot_mode_arclength_event_diagnosis_"
    "wp10c9d6c7c3b5c4f25f1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HOT_MODE_ARCLENGTH_EVENT_"
    "DIAGNOSIS_WP10C9D6C7C3B5C4F25F1_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

MINIMUM_WITHIN_WINDOW_PHASE_SPEED_RATIO = 0.5
MINIMUM_CHORD_EFFICIENCY = 0.995
MAXIMUM_ARCLENGTH_TANGENT_DEFECT = 2.5e-2
MINIMUM_ARCLENGTH_DIRECTION_COSINE = 0.9995
MAXIMUM_FOUR_NODE_DIRECTION_HOLDOUT_DEFECT = 1.0e-4
MAXIMUM_FOUR_NODE_INVERSE_SPEED_HOLDOUT_DEFECT = 1.0e-4
MAXIMUM_REPARAMETERIZED_TIME_DEFECT = 5.0e-2
LEGACY_EXIT_DISTANCE_MINIMUM = 0.5
SATURATED_GATE_FRACTION = 0.95
FOUR_NODE_TRAINING_INDICES = np.asarray((0, 2, 5, 7), dtype=int)


def _helper():
    return source._helper()


def _window_directories() -> tuple[Path, ...]:
    adaptive = source._adaptive()
    return (
        adaptive._stage_directory(1),
        adaptive._stage_directory(2),
        source._stage_directory(3),
        source._stage_directory(4),
        source._stage_directory(5),
    )


def _validate_inputs(*, require_clean: bool) -> dict:
    helper = _helper()
    directories = _window_directories()
    hashes = {}
    summaries = []
    for index, directory in enumerate(directories, 1):
        hashes[f"window_{index:02d}"] = helper._validate_checksums(directory)
        summary = helper._read(directory / "summary.json")
        metrics = helper._read(directory / "phase_window_metrics.json")
        if (
            not summary["passed"]
            or not metrics["passed"]
            or int(summary["window_index"]) != index
            or metrics["hot_exit_observed"]
            or int(metrics["new_nonlinear_fixed_Q_roots"]) != 0
            or int(metrics["new_BDF_microsteps"]) != 0
        ):
            raise RuntimeError(f"accepted phase Window {index} changed")
        summaries.append(summary)
    for left, right in zip(directories[:-1], directories[1:], strict=True):
        left_arrays = helper._load_npz(left / "phase_window_arrays.npz")
        right_arrays = helper._load_npz(right / "phase_window_arrays.npz")
        np.testing.assert_array_equal(
            left_arrays["endpoint_primitive_state"], right_arrays["start_primitive_state"]
        )
        left_metrics = helper._read(left / "phase_window_metrics.json")
        right_metrics = helper._read(right / "phase_window_metrics.json")
        if float(left_metrics["end_time_seconds"]) != float(
            right_metrics["start_time_seconds"]
        ):
            raise RuntimeError("accepted phase windows are not time-contiguous")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("arclength diagnosis requires a clean tracked tree")
    return {"window_hashes": hashes, "window_summaries": summaries}


def _polynomial_holdout(
    phase: np.ndarray, values: np.ndarray, training_indices: np.ndarray
) -> np.ndarray:
    all_indices = np.arange(len(phase))
    heldout = all_indices[~np.isin(all_indices, training_indices)]
    vandermonde = np.vander(
        phase[training_indices], N=len(training_indices), increasing=True
    )
    coefficients = np.linalg.solve(vandermonde, values[training_indices])
    return np.vander(
        phase[heldout], N=len(training_indices), increasing=True
    ) @ coefficients - values[heldout]


def _window_diagnostic(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    metrics = helper._read(directory / "phase_window_metrics.json")
    arrays = helper._load_npz(directory / "phase_window_arrays.npz")
    coordinates = np.asarray(arrays["coordinates470"], dtype=float)
    rates = np.asarray(arrays["final_rates470_per_s"], dtype=float)
    speeds = np.linalg.norm(rates, axis=1)
    if np.any(speeds <= 0.0):
        raise RuntimeError("arclength phase speed is not strictly positive")
    increments = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    arclength_nodes = np.concatenate(([0.0], np.cumsum(increments)))
    arclength = float(arclength_nodes[-1])
    if arclength <= 0.0 or np.any(np.diff(arclength_nodes) <= 0.0):
        raise RuntimeError("stored coordinate path does not define arclength")
    phase = arclength_nodes / arclength
    differentiation = lagrange_differentiation_matrix(phase)
    arclength_tangent = (differentiation @ coordinates) / arclength
    unit_rate = rates / speeds[:, None]
    tangent_defects = np.linalg.norm(arclength_tangent - unit_rate, axis=1)
    tangent_norms = np.linalg.norm(arclength_tangent, axis=1)
    direction_cosines = np.sum(arclength_tangent * unit_rate, axis=1) / np.maximum(
        tangent_norms, np.finfo(float).tiny
    )
    times = float(metrics["start_time_seconds"]) + (
        float(metrics["duration_seconds"]) * np.asarray(arrays["nodes"], dtype=float)
    )
    time_derivative = (differentiation @ times) / arclength
    inverse_speeds = 1.0 / speeds
    time_defects = np.abs(time_derivative - inverse_speeds) / inverse_speeds
    direction_holdout = _polynomial_holdout(
        phase, unit_rate, FOUR_NODE_TRAINING_INDICES
    )
    inverse_speed_holdout = _polynomial_holdout(
        phase, inverse_speeds, FOUR_NODE_TRAINING_INDICES
    )
    heldout_indices = np.arange(len(phase))[
        ~np.isin(np.arange(len(phase)), FOUR_NODE_TRAINING_INDICES)
    ]
    inverse_speed_relative = np.abs(inverse_speed_holdout) / inverse_speeds[
        heldout_indices
    ]
    chord = float(
        np.linalg.norm(
            np.asarray(arrays["endpoint_coordinate470"])
            - np.asarray(arrays["start_coordinate470"])
        )
    )
    record = {
        "window_index": int(metrics["window_index"]),
        "start_time_seconds": float(metrics["start_time_seconds"]),
        "end_time_seconds": float(metrics["end_time_seconds"]),
        "duration_seconds": float(metrics["duration_seconds"]),
        "coordinate_arclength": arclength,
        "coordinate_chord": chord,
        "chord_efficiency": chord / arclength,
        "minimum_phase_speed_per_second": float(np.min(speeds)),
        "maximum_phase_speed_per_second": float(np.max(speeds)),
        "phase_speed_ratio": float(np.min(speeds) / np.max(speeds)),
        "maximum_arclength_tangent_defect": float(np.max(tangent_defects)),
        "minimum_arclength_direction_cosine": float(np.min(direction_cosines)),
        "maximum_arclength_tangent_norm_defect": float(
            np.max(np.abs(tangent_norms - 1.0))
        ),
        "maximum_reparameterized_time_defect": float(np.max(time_defects)),
        "maximum_four_node_direction_holdout_defect": float(
            np.max(np.linalg.norm(direction_holdout, axis=1))
        ),
        "maximum_four_node_inverse_speed_holdout_defect": float(
            np.max(inverse_speed_relative)
        ),
        "hidden_secant_fraction": float(
            metrics["event_metrics"]["hidden_secant_fraction"]
        ),
        "rank16_hidden_amplitude": float(
            metrics["event_metrics"]["rank16_hidden_amplitude_from_20ms_anchor"]
        ),
        "macro_drift": float(
            metrics["event_metrics"]["macro_drift_from_warm3_seed"]
        ),
        "full_collocation_gate_fraction": float(
            metrics["gate_values"]["maximum_full_collocation_defect"]
            / source.manifest.MAXIMUM_FULL_COLLOCATION_DEFECT
        ),
        "coordinate_residual_gate_fraction": float(
            metrics["gate_values"].get("maximum_coordinate_residual_infinity", 0.0)
            / source.manifest.COORDINATE_TOLERANCE
        ),
        "anchor_departure_gate_fraction": float(
            metrics["gate_values"].get("maximum_scaled_anchor_departure", 0.0)
            / source.manifest.MAXIMUM_SCALED_ANCHOR_DEPARTURE
        ),
    }
    data = {
        "phase_nodes": phase,
        "arclength_nodes": arclength_nodes,
        "unit_rates470": unit_rate,
        "arclength_tangents470": arclength_tangent,
        "phase_speeds_per_second": speeds,
        "inverse_phase_speeds_seconds": inverse_speeds,
        "tangent_defects": tangent_defects,
        "time_mapping_defects": time_defects,
        "start_coordinate470": np.asarray(arrays["start_coordinate470"]),
        "endpoint_coordinate470": np.asarray(arrays["endpoint_coordinate470"]),
        "endpoint_primitive_state": np.asarray(arrays["endpoint_primitive_state"]),
    }
    return record, data


def _diagnose(locked: dict) -> tuple[dict, dict[str, np.ndarray]]:
    records = []
    data = []
    for directory in _window_directories():
        record, arrays = _window_diagnostic(directory)
        records.append(record)
        data.append(arrays)
    speeds = np.asarray([item["maximum_phase_speed_per_second"] for item in records])
    hidden = np.asarray([item["hidden_secant_fraction"] for item in records])
    amplitudes = np.asarray([item["rank16_hidden_amplitude"] for item in records])
    macro = np.asarray([item["macro_drift"] for item in records])
    segment_lengths = np.asarray([item["coordinate_arclength"] for item in records])
    points = np.vstack((data[0]["start_coordinate470"], *(
        item["endpoint_coordinate470"] for item in data
    )))
    nonadjacent = [
        float(np.linalg.norm(points[right] - points[left]))
        for left in range(len(points))
        for right in range(left + 2, len(points))
    ]
    endpoint_rates = np.vstack([item["unit_rates470"][-1] for item in data])
    interface_cosines = np.sum(endpoint_rates[:-1] * endpoint_rates[1:], axis=1)
    final = records[-1]
    saturated = {
        "full_collocation": final["full_collocation_gate_fraction"]
        >= SATURATED_GATE_FRACTION,
        "coordinate_residual": final["coordinate_residual_gate_fraction"]
        >= SATURATED_GATE_FRACTION,
        "anchor_departure": final["anchor_departure_gate_fraction"]
        >= SATURATED_GATE_FRACTION,
    }
    gates = {
        "accepted_history_contiguous": True,
        "strictly_positive_arclength": bool(np.all(segment_lengths > 0.0)),
        "phase_speed_conditioning": min(
            item["phase_speed_ratio"] for item in records
        ) >= MINIMUM_WITHIN_WINDOW_PHASE_SPEED_RATIO,
        "chord_efficiency": min(item["chord_efficiency"] for item in records)
        >= MINIMUM_CHORD_EFFICIENCY,
        "arclength_tangent": max(
            item["maximum_arclength_tangent_defect"] for item in records
        ) <= MAXIMUM_ARCLENGTH_TANGENT_DEFECT,
        "arclength_direction": min(
            item["minimum_arclength_direction_cosine"] for item in records
        ) >= MINIMUM_ARCLENGTH_DIRECTION_COSINE,
        "four_node_direction_holdout": max(
            item["maximum_four_node_direction_holdout_defect"] for item in records
        ) <= MAXIMUM_FOUR_NODE_DIRECTION_HOLDOUT_DEFECT,
        "four_node_inverse_speed_holdout": max(
            item["maximum_four_node_inverse_speed_holdout_defect"] for item in records
        ) <= MAXIMUM_FOUR_NODE_INVERSE_SPEED_HOLDOUT_DEFECT,
        "reparameterized_time_consistency": max(
            item["maximum_reparameterized_time_defect"] for item in records
        ) <= MAXIMUM_REPARAMETERIZED_TIME_DEFECT,
        "legacy_exit_not_approached": bool(
            np.all(hidden - source.manifest.HIDDEN_SECANT_FRACTION_MAX
                   >= LEGACY_EXIT_DISTANCE_MINIMUM)
        ),
        "hot_amplitude_strictly_increasing": bool(np.all(np.diff(amplitudes) > 0.0)),
        "current_path_not_recurrent": min(nonadjacent)
        > float(np.median(segment_lengths)),
        "fixed_time_window_contract_saturated": sum(saturated.values()) >= 2,
    }
    passed = bool(all(gates.values()))
    if not passed:
        classification = "arclength_event_diagnosis_rejected_no_new_execution_authorized"
    else:
        classification = CLASSIFICATION
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "windows": records,
        "gates": gates,
        "gate_values": {
            "cumulative_coordinate_arclength": float(np.sum(segment_lengths)),
            "minimum_phase_speed_ratio": min(item["phase_speed_ratio"] for item in records),
            "minimum_chord_efficiency": min(item["chord_efficiency"] for item in records),
            "maximum_arclength_tangent_defect": max(item["maximum_arclength_tangent_defect"] for item in records),
            "minimum_arclength_direction_cosine": min(item["minimum_arclength_direction_cosine"] for item in records),
            "maximum_four_node_direction_holdout_defect": max(item["maximum_four_node_direction_holdout_defect"] for item in records),
            "maximum_four_node_inverse_speed_holdout_defect": max(item["maximum_four_node_inverse_speed_holdout_defect"] for item in records),
            "maximum_reparameterized_time_defect": max(item["maximum_reparameterized_time_defect"] for item in records),
            "minimum_nonadjacent_endpoint_distance": min(nonadjacent),
            "median_segment_arclength": float(np.median(segment_lengths)),
            "minimum_interface_direction_cosine": float(np.min(interface_cosines)),
            "final_hidden_secant_fraction": float(hidden[-1]),
            "legacy_hidden_exit_threshold": float(source.manifest.HIDDEN_SECANT_FRACTION_MAX),
            "final_rank16_hidden_amplitude": float(amplitudes[-1]),
            "final_macro_drift": float(macro[-1]),
            "final_phase_speed_per_second": float(speeds[-1]),
            "saturated_final_gate_count": int(sum(saturated.values())),
        },
        "final_gate_saturation": saturated,
        "regime_diagnosis": {
            "legacy_hot_exit_observed": False,
            "equilibrium_approach_observed": False,
            "recurrence_observed": False,
            "hot_amplitude_growth_observed": True,
            "classification_scope": "accepted_windows_01_through_05_only",
        },
        "selected_architecture": {
            "primary_phase": "Euclidean arclength in nondimensional exact 470-coordinate space",
            "phase_equations": {
                "nu": "||f_Q(y,t)||_2",
                "dy_ds": "f_Q(y,t)/nu",
                "dt_ds": "1/nu",
            },
            "moving_exact_retraction_retained": True,
            "legacy_hidden_fraction_is_diagnostic_not_sole_termination": True,
            "next_truth_node_count_recommendation": 5,
            "initial_arclength_span_recommendation": 2.5e-2,
            "fixed_time_window_06_superseded_before_execution": True,
        },
        "new_exact_truth_calls": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "new_BDF_microsteps": 0,
        "input_lock": locked,
    }
    arrays = {
        "window_coordinate_arclengths": segment_lengths,
        "window_phase_speed_minima": np.asarray([item["minimum_phase_speed_per_second"] for item in records]),
        "window_phase_speed_maxima": np.asarray([item["maximum_phase_speed_per_second"] for item in records]),
        "window_hidden_secant_fractions": hidden,
        "window_rank16_hidden_amplitudes": amplitudes,
        "window_macro_drifts": macro,
        "endpoint_coordinates470": points[1:],
        "endpoint_unit_rates470": endpoint_rates,
        "interface_direction_cosines": interface_cosines,
        **{
            f"window_{index:02d}__{name}": value
            for index, item in enumerate(data, 1)
            for name, value in item.items()
            if isinstance(value, np.ndarray)
        },
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = source._post().manifest.transition.manifest.cold.manifest
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


def _run() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("arclength event diagnosis already exists")
    locked = _validate_inputs(require_clean=True)
    metrics, arrays = _diagnose(locked)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "arclength_event_metrics.json", metrics)
    with (CANONICAL_DIRECTORY / "arclength_event_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    input_hashes = {
        str(path.relative_to(ROOT)): helper._sha(path)
        for directory in _window_directories()
        for path in (
            directory / "summary.json",
            directory / "phase_window_metrics.json",
            directory / "phase_window_arrays.npz",
        )
    }
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", input_hashes)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "truth_free_diagnosis": True,
        "accepted_windows_analyzed": [1, 2, 3, 4, 5],
        "weighted_coordinate_arclength_selected": metrics["passed"],
        "legacy_hot_exit_observed": False,
        "fixed_time_window_06_authorized": False,
        "arclength_manifest_authorized": metrics["passed"],
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "runner_sha256": helper._sha(ROOT / THIS_RUNNER),
        "test_sha256": helper._sha(ROOT / THIS_TEST),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Hot-mode arclength and event diagnosis",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            "No new truth rate, nonlinear fixed-Q root, or BDF microstep was executed. The five accepted phase windows are bitwise contiguous in primitive state and physical time.",
            "",
            f"The accumulated nondimensional coordinate arclength is `{values['cumulative_coordinate_arclength']:.6e}`. The worst within-window speed ratio is `{values['minimum_phase_speed_ratio']:.6e}`, the worst arclength tangent defect is `{values['maximum_arclength_tangent_defect']:.6e}`, and the worst four-node held-out direction defect is `{values['maximum_four_node_direction_holdout_defect']:.6e}`.",
            "",
            f"The final hidden secant fraction is `{values['final_hidden_secant_fraction']:.9e}` versus the legacy exit threshold `{values['legacy_hidden_exit_threshold']:.6e}`. The rank-16 hidden amplitude grows monotonically to `{values['final_rank16_hidden_amplitude']:.6e}`; neither exit, equilibrium approach, nor recurrence is selected by the available evidence.",
            "",
            "Fixed physical-time Window 6 is prospectively superseded. The authorized next artifact is a definitions-only moving exact weighted-arclength continuation and regime-classification manifest with a five-node first preflight.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    print(json.dumps(_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
