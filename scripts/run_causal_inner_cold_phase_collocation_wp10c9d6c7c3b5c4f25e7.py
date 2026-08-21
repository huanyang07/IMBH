#!/usr/bin/env python3
"""Replay the cold phase chart against saved exact continuous rates."""

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

from imri_qpe.layer3_minidisk_1d.phase_collocation import (  # noqa: E402
    PiecewisePhaseCollocation,
    PolynomialPhaseSegment,
    direction_cosine,
    relative_vector_defect,
)
import run_causal_inner_phase_collocation_preflight_manifest_wp10c9d6c7c3b5c4f25e6 as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25e7"
PASS_CLASSIFICATION = "cold_phase_collocation_exact_saved_vector_field_replay_passed"
FAIL_CLASSIFICATION = "cold_phase_collocation_rejected"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25e8"

ARTIFACT = "causal_inner_cold_phase_collocation_wp10c9d6c7c3b5c4f25e7"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_cold_phase_collocation_wp10c9d6c7c3b5c4f25e7.py"
THIS_TEST = "tests/test_causal_inner_cold_phase_collocation_wp10c9d6c7c3b5c4f25e7.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COLD_PHASE_COLLOCATION_"
    "WP10C9D6C7C3B5C4F25E7_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper():
    return manifest._helper()


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    contract = helper._read(manifest.CANONICAL_DIRECTORY / "phase_collocation_contract.json")
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("phase-collocation manifest changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen phase-collocation source changed: {relative}")
    for label, path in (
        ("candidate_arrays", manifest.CANDIDATE_ARRAYS),
        ("cold_rate_arrays", manifest.COLD_RATE_ARRAYS),
        ("tangent_arrays", manifest.TANGENT_ARRAYS),
        ("transition_geometry", manifest.TRANSITION_GEOMETRY),
    ):
        if helper._sha(path) != contract["input_hashes"][label]:
            raise RuntimeError(f"frozen phase-collocation input changed: {label}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cold phase-collocation replay requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _load_evidence() -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    helper = _helper()
    candidates = helper._load_npz(manifest.CANDIDATE_ARRAYS)
    cold = helper._load_npz(manifest.COLD_RATE_ARRAYS)
    tangent = helper._load_npz(manifest.TANGENT_ARRAYS)
    times = np.asarray(candidates["candidate_times_seconds"], dtype=float)
    coordinates = np.asarray(candidates["candidate_absolute_y470_coordinates"], dtype=float)
    rates = {}
    for milliseconds in (2, 5, 8, 12):
        prefix = f"candidate_{milliseconds:02d}ms"
        rates[milliseconds] = np.asarray(
            cold[f"{prefix}__coordinate_jacobian470x560"]
            @ cold[f"{prefix}__scaled_free_rate560_per_s"],
            dtype=float,
        )
    return (
        times,
        coordinates,
        rates,
        np.asarray(tangent["macro_restriction_R82"], dtype=float),
        np.asarray(tangent["hidden_dual_Q388"], dtype=float),
    )


def _build_models(times: np.ndarray, coordinates: np.ndarray, rates: dict[int, np.ndarray]):
    index = {int(round(time * 1000)): position for position, time in enumerate(times)}
    first = PolynomialPhaseSegment.from_constraints(
        start_time_seconds=0.002,
        end_time_seconds=0.008,
        value_times_seconds=np.asarray((0.002, 0.008)),
        values=np.stack((coordinates[index[2]], coordinates[index[8]])),
        rate_times_seconds=np.asarray((0.002, 0.008)),
        rates_per_second=np.stack((rates[2], rates[8])),
    )
    second = PolynomialPhaseSegment.from_constraints(
        start_time_seconds=0.008,
        end_time_seconds=0.020,
        value_times_seconds=np.asarray((0.008, 0.016, 0.020)),
        values=np.stack((coordinates[index[8]], coordinates[index[16]], coordinates[index[20]])),
        rate_times_seconds=np.asarray((0.008,)),
        rates_per_second=np.stack((rates[8],)),
    )
    refined = PiecewisePhaseCollocation((first, second))
    coarse = PolynomialPhaseSegment.from_constraints(
        start_time_seconds=0.002,
        end_time_seconds=0.020,
        value_times_seconds=np.asarray((0.002, 0.008, 0.016, 0.020)),
        values=np.stack(
            (coordinates[index[2]], coordinates[index[8]], coordinates[index[16]], coordinates[index[20]])
        ),
        rate_times_seconds=np.empty(0),
        rates_per_second=np.empty((0, coordinates.shape[1])),
    )
    return refined, coarse, index


def _evaluate() -> tuple[dict, dict[str, np.ndarray]]:
    times, coordinates, rates, restriction, hidden_dual = _load_evidence()
    refined, coarse, index = _build_models(times, coordinates, rates)
    heldout_ms = (5, 12)
    cold_path = float(np.sum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1)))
    state_errors = []
    rate_defects = []
    rate_cosines = []
    macro_rate_defects = []
    hidden_rate_defects = []
    refinement_state_defects = []
    refinement_rate_defects = []
    predictions = []
    predicted_rates = []
    for milliseconds in heldout_ms:
        time_seconds = milliseconds / 1000.0
        predicted = refined.value(time_seconds)
        predicted_rate = refined.rate(time_seconds)
        truth = coordinates[index[milliseconds]]
        truth_rate = rates[milliseconds]
        predictions.append(predicted)
        predicted_rates.append(predicted_rate)
        state_errors.append(float(np.linalg.norm(predicted - truth) / cold_path))
        rate_defects.append(relative_vector_defect(predicted_rate, truth_rate))
        rate_cosines.append(direction_cosine(predicted_rate, truth_rate))
        macro_rate_defects.append(
            relative_vector_defect(restriction @ predicted_rate, restriction @ truth_rate)
        )
        hidden_rate_defects.append(
            relative_vector_defect(hidden_dual @ predicted_rate, hidden_dual @ truth_rate)
        )
        refinement_state_defects.append(
            float(np.linalg.norm(predicted - coarse.value(time_seconds)) / cold_path)
        )
        refinement_rate_defects.append(
            float(
                np.linalg.norm(predicted_rate - coarse.rate(time_seconds))
                / max(float(np.linalg.norm(truth_rate)), np.finfo(float).tiny)
            )
        )
    interface_defects = refined.interface_value_defects()
    condition_numbers = np.asarray(
        [segment.constraint_condition_number for segment in refined.segments]
        + [coarse.constraint_condition_number]
    )
    gate_values = {
        "maximum_state_error_over_cold_path": max(state_errors),
        "maximum_full_rate_relative_defect": max(rate_defects),
        "minimum_full_rate_direction_cosine": min(rate_cosines),
        "maximum_macro_rate_relative_defect": max(macro_rate_defects),
        "maximum_hidden_rate_relative_defect": max(hidden_rate_defects),
        "maximum_one_versus_two_window_state_defect": max(refinement_state_defects),
        "maximum_one_versus_two_window_rate_defect": max(refinement_rate_defects),
        "maximum_interface_value_defect": float(np.max(interface_defects)),
        "maximum_constraint_condition_number": float(np.max(condition_numbers)),
    }
    gates = {
        "heldout_state": gate_values["maximum_state_error_over_cold_path"]
        <= manifest.MAXIMUM_STATE_ERROR_OVER_LOCAL_PATH,
        "full_vector_field": gate_values["maximum_full_rate_relative_defect"]
        <= manifest.MAXIMUM_RATE_RELATIVE_DEFECT,
        "rate_direction": gate_values["minimum_full_rate_direction_cosine"]
        >= manifest.MINIMUM_RATE_DIRECTION_COSINE,
        "window_state_refinement": gate_values["maximum_one_versus_two_window_state_defect"]
        <= manifest.MAXIMUM_ONE_VERSUS_TWO_WINDOW_STATE_DEFECT,
        "window_rate_refinement": gate_values["maximum_one_versus_two_window_rate_defect"]
        <= manifest.MAXIMUM_ONE_VERSUS_TWO_WINDOW_RATE_DEFECT,
        "multiple_shooting_continuity": gate_values["maximum_interface_value_defect"]
        <= manifest.MAXIMUM_INTERFACE_VALUE_DEFECT,
        "constraint_conditioning": gate_values["maximum_constraint_condition_number"]
        <= manifest.MAXIMUM_CONSTRAINT_CONDITION_NUMBER,
        "saved_exact_rates_only": True,
        "new_truth_calls_zero": True,
    }
    passed = bool(all(gates.values()))
    metrics = {
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "gates": gates,
        "gate_values": gate_values,
        "heldout_times_seconds": [value / 1000.0 for value in heldout_ms],
        "segment_count": len(refined.segments),
        "polynomial_degrees": [segment.degree for segment in refined.segments],
        "exact_saved_full_model_rate_witnesses": 4,
        "new_truth_calls": 0,
        "transition_collocation_authorized": passed,
        "post_transition_segment_authorized": False,
        "predictive_cycle_authorized": False,
    }
    arrays = {
        "heldout_times_seconds": np.asarray(metrics["heldout_times_seconds"]),
        "heldout_true_coordinates470": coordinates[[index[5], index[12]]],
        "heldout_predicted_coordinates470": np.asarray(predictions),
        "heldout_true_exact_rates470_per_s": np.stack((rates[5], rates[12])),
        "heldout_predicted_rates470_per_s": np.asarray(predicted_rates),
        "state_errors_over_cold_path": np.asarray(state_errors),
        "full_rate_relative_defects": np.asarray(rate_defects),
        "full_rate_direction_cosines": np.asarray(rate_cosines),
        "macro_rate_relative_defects": np.asarray(macro_rate_defects),
        "hidden_rate_relative_defects": np.asarray(hidden_rate_defects),
        "one_versus_two_window_state_defects": np.asarray(refinement_state_defects),
        "one_versus_two_window_rate_defects": np.asarray(refinement_rate_defects),
        "interface_value_defects": interface_defects,
        "constraint_condition_numbers": condition_numbers,
        "first_segment_coefficients": refined.segments[0].coefficients,
        "second_segment_coefficients": refined.segments[1].coefficients,
        "coarse_segment_coefficients": coarse.coefficients,
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    with manifest.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
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
    with manifest.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(manifest.CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": manifest.PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(manifest.CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("cold phase-collocation result already exists")
    locked = _validate_manifest(require_clean=True)
    metrics, arrays = _evaluate()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "cold_collocation_metrics.json", metrics)
    with (CANONICAL_DIRECTORY / "cold_collocation_model_and_replay.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", locked)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "cold_full_vector_field_replay_passed": metrics["passed"],
        "transition_collocation_manifest_authorized": metrics["passed"],
        "post_transition_segment_authorized": False,
        "predictive_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Cold phase collocation WP10c9d6c7c3b5c4f25e7",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"The two-window phase chart predicts the held-out 5 ms and 12 ms states with a maximum path-relative error of {metrics['gate_values']['maximum_state_error_over_cold_path']:.6e}. Against saved exact full-model continuous rates, its maximum relative defect is {metrics['gate_values']['maximum_full_rate_relative_defect']:.6e} and its minimum direction cosine is {metrics['gate_values']['minimum_full_rate_direction_cosine']:.12f}.",
                "",
                "This certifies the cold observed mode only. It makes no transition, post-transition, hot-exit, or complete-cycle claim.",
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
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    payload = _run()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
