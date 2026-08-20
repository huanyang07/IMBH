#!/usr/bin/env python3
"""Validate the frozen direct 470D field on fresh exact-rate holdouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_direct_coordinate_field_manifest_wp10c9d6c7c3b5c4f25cn as manifest  # noqa: E402
import run_causal_inner_shell_gated_atlas_rate_validation_wp10c9d6c7c3b5c4f25ck as prior_rate  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25co"
MANIFEST_COMMIT = "5d47f8c849c221506dbf3040ae40ca045dc184fb"
MANIFEST_PARENT = "2ecc74e1a7367b4d1e1e2070444ee0e22570cb61"
MANIFEST_TREE = "27b9c9f6bd341e5d2da075cc4caf8be89be739f3"

PASS_CLASSIFICATION = "direct_470_coordinate_field_independently_validated"
FAIL_CLASSIFICATION = "direct_470_coordinate_field_independent_validation_failed"
PASS_AUTHORIZED_NEXT = (
    "definitions_only_one_recentered_transition_forecast_execution_manifest"
)
FAIL_AUTHORIZED_NEXT = "definitions_only_direct_coordinate_field_revision_manifest"

ARTIFACT = (
    "causal_inner_direct_coordinate_field_validation_"
    "wp10c9d6c7c3b5c4f25co"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_direct_coordinate_field_validation_"
    "wp10c9d6c7c3b5c4f25co.py"
)
THIS_TEST = (
    "tests/test_causal_inner_direct_coordinate_field_validation_"
    "wp10c9d6c7c3b5c4f25co.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_DIRECT_COORDINATE_"
    "FIELD_VALIDATION_WP10C9D6C7C3B5C4F25CO_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PROGRESS_JSON = SCRATCH_DIRECTORY / "progress.json"
PROGRESS_NPZ = SCRATCH_DIRECTORY / "progress.npz"

rate_engine = prior_rate.rate_engine
vector_field = prior_rate.vector_field

_plain = manifest._plain
_read = manifest._read
_write_json = manifest._write_json
_sha = manifest._sha
_checksums = manifest._checksums
_load_npz = manifest._load_npz


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(expected))
        / max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    )


def _append(array: np.ndarray, value: np.ndarray) -> np.ndarray:
    item = np.asarray(value, dtype=float)
    return np.concatenate((array, item.reshape((1,) + item.shape)), axis=0)


def _thread_environment() -> dict[str, str]:
    return (
        vector_field.manifest.parent.geometry.chart_tools.coordinate_tools.THREAD_ENVIRONMENT
    )


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("direct coordinate-field manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("direct coordinate-field manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("direct coordinate-field manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    lock = _read(manifest.CANONICAL_DIRECTORY / "parent_lock.json")
    closure = _load_npz(
        manifest.CANONICAL_DIRECTORY / "direct_coordinate_field.npz"
    )
    geometry = _load_npz(manifest.HOLDOUT_GEOMETRY)
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_independent_exact_rate_evaluations"]
        != manifest.PLANNED_RATE_EVALUATIONS
        or not summary["coefficients_frozen_before_holdout_truth"]
        or summary["new_truth_rate_calls"] != 0
        or summary["state_dependent_coordinate_Jacobian_online"]
        or summary["trajectory_authorized"]
        or contract["decision"]["pass_classification"] != PASS_CLASSIFICATION
        or contract["decision"]["fail_classification"] != FAIL_CLASSIFICATION
        or closure["q_rate_centers"].shape != (16, 28)
        or closure["q_rate_coefficients"].shape != (16, 162)
        or geometry["candidate_primitive_states"].shape != (8, 112, 5)
    ):
        raise RuntimeError("direct coordinate-field exact validation contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"direct field manifest source changed: {relative}")
    if (
        _sha(manifest.HOLDOUT_GEOMETRY) != lock["holdout_geometry_sha256"]
        or _sha(manifest.CANONICAL_DIRECTORY / "direct_coordinate_field.npz")
        != hashes["direct_coordinate_field.npz"]
    ):
        raise RuntimeError("frozen direct-field input changed")
    for name, expected in _thread_environment().items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("direct coordinate-field validation requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "hashes": hashes,
        "closure": closure,
        "geometry": geometry,
    }


def _load_inputs(frozen: dict) -> dict:
    model = vector_field.ReducedVectorField()
    geometry = frozen["geometry"]
    states = np.asarray(geometry["candidate_primitive_states"], dtype=float)
    deltas = np.asarray(geometry["candidate_scaled_deltas"], dtype=float)
    departures = np.asarray(
        geometry["candidate_departure_coordinates"], dtype=float
    )
    bounds = np.asarray(geometry["candidate_component_bounds"], dtype=float)
    direction_indices = np.asarray(
        geometry["candidate_direction_indices"], dtype=int
    )
    recorded_coordinates = np.asarray(geometry["online_coordinates"], dtype=float)
    labels = tuple(
        _read(
            manifest.parent.manifest.CANONICAL_DIRECTORY / "holdout_design.json"
        )["labels"]
    )
    coordinates = np.asarray(
        [
            np.concatenate(
                (
                    np.zeros(manifest.PHYSICAL_DIMENSION),
                    model.memory_basis.T @ delta,
                    departure,
                )
            )
            for delta, departure in zip(deltas, departures)
        ]
    )
    if (
        states.shape != (8, 112, 5)
        or deltas.shape != (8, 560)
        or departures.shape != (8, 28)
        or coordinates.shape != (8, 470)
        or not np.array_equal(coordinates, recorded_coordinates)
        or tuple(direction_indices) != (0, 1, 2, 3, 0, 1, 2, 3)
        or tuple(bounds[:4]) != (0.0125,) * 4
        or tuple(bounds[4:]) != (0.015,) * 4
        or len(labels) != 4
    ):
        raise RuntimeError("direct coordinate-field holdout inputs changed")
    direct = manifest.DirectCoordinateField(frozen["closure"], model=model)
    return {
        "model": model,
        "direct": direct,
        "states": states,
        "deltas": deltas,
        "departures": departures,
        "coordinates": coordinates,
        "component_bounds": bounds,
        "direction_indices": direction_indices,
        "direction_labels": labels,
    }


def _online_prediction_without_coordinate_jacobian(
    direct: manifest.DirectCoordinateField, coordinate: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    chart_tools = vector_field.manifest.parent.geometry.chart_tools
    original = chart_tools._coordinate_jacobian

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("online direct field attempted a coordinate-Jacobian build")

    chart_tools._coordinate_jacobian = forbidden
    began = time.perf_counter()
    try:
        coordinate_rate = direct.field(coordinate)
        full_rate = direct.full_state_rate(coordinate)
    finally:
        wall = time.perf_counter() - began
        chart_tools._coordinate_jacobian = original
    return coordinate_rate, full_rate, wall


def _progress_array_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "total_rates_per_second": (560,),
        "free_rates_per_second": (560,),
        "physical_reaction_actions_per_second": (560,),
        "multiplier_coordinates_per_second": (3,),
        "exact_online_470_coordinate_rates_per_second": (470,),
        "predicted_online_470_coordinate_rates_per_second": (470,),
        "predicted_full_state_rates_per_second": (560,),
        "online_coordinates": (470,),
        "repaired_decoded_scaled_deltas": (560,),
        "decoded_online_coordinates": (470,),
    }


def _progress_identity() -> dict:
    return {
        "work_package": WORK_PACKAGE,
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_hashes": _checksums(manifest.CANONICAL_DIRECTORY),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "holdout_geometry_sha256": _sha(manifest.HOLDOUT_GEOMETRY),
    }


def _empty_progress() -> dict:
    progress = {"identity": _progress_identity(), "evaluations": [], "failures": []}
    for name, shape in _progress_array_shapes().items():
        progress[name] = np.empty((0,) + shape, dtype=float)
    return progress


def _save_progress(progress: dict) -> None:
    _write_json(
        PROGRESS_JSON,
        {
            "identity": progress["identity"],
            "evaluations": progress["evaluations"],
            "failures": progress["failures"],
        },
    )
    _write_npz(
        PROGRESS_NPZ,
        {name: progress[name] for name in _progress_array_shapes()},
    )


def _load_or_create_progress() -> dict:
    if not PROGRESS_JSON.exists() and not PROGRESS_NPZ.exists():
        return _empty_progress()
    if not PROGRESS_JSON.exists() or not PROGRESS_NPZ.exists():
        raise RuntimeError("direct-field validation checkpoint is incomplete")
    recorded = _read(PROGRESS_JSON)
    if recorded["identity"] != _progress_identity():
        raise RuntimeError("direct-field validation checkpoint identity changed")
    progress = {
        "identity": recorded["identity"],
        "evaluations": recorded["evaluations"],
        "failures": recorded["failures"],
        **_load_npz(PROGRESS_NPZ),
    }
    count = len(progress["evaluations"])
    if any(
        progress[name].shape != (count,) + shape
        for name, shape in _progress_array_shapes().items()
    ):
        raise RuntimeError("direct-field validation checkpoint dimensions changed")
    if [item["candidate_index"] for item in progress["evaluations"]] != list(
        range(count)
    ):
        raise RuntimeError("direct-field validation checkpoint ordering changed")
    return progress


def _evaluate_one(inputs: dict, progress: dict, index: int, data: dict) -> None:
    model = inputs["model"]
    direct = inputs["direct"]
    state = inputs["states"][index]
    exact_delta = inputs["deltas"][index]
    departure = inputs["departures"][index]
    coordinate = inputs["coordinates"][index]

    predicted_online, predicted_full, online_wall = (
        _online_prediction_without_coordinate_jacobian(direct, coordinate)
    )
    item, arrays = rate_engine.manifest.prior_screen._continuous_rate(data, state)
    coordinate_jacobian, coordinate_metrics = (
        vector_field.manifest.parent.geometry.chart_tools._coordinate_jacobian(
            state, model.components
        )
    )
    total_rate = np.asarray(arrays["total_rate"], dtype=float)
    exact_online = np.concatenate(
        (
            coordinate_jacobian @ total_rate,
            model.memory_basis.T @ total_rate,
            model.departure_basis.T @ total_rate,
        )
    )
    decoded_delta = direct.decoded_delta(coordinate)
    decoded_state = direct.decoded_state(coordinate)
    decoded_coordinate, decoded_factors = model.coordinate(decoded_state)
    decoded_physical = vector_field.manifest.parent.geometry.chart_tools._state_audit(
        model.components["context"], decoded_state
    )
    exact_a = exact_online[-manifest.DEPARTURE_DIMENSION :]
    predicted_a = predicted_online[-manifest.DEPARTURE_DIMENSION :]
    radius = float(np.linalg.norm(departure))
    exact_radial_speed = float(departure @ exact_a / radius)
    predicted_radial_speed = float(departure @ predicted_a / radius)
    direction_index = int(inputs["direction_indices"][index])
    blocks = {
        "q162": slice(0, manifest.PHYSICAL_DIMENSION),
        "z280": slice(
            manifest.PHYSICAL_DIMENSION,
            manifest.PHYSICAL_DIMENSION + manifest.MEMORY_DIMENSION,
        ),
        "a28": slice(-manifest.DEPARTURE_DIMENSION, None),
    }
    item.update(
        {
            "candidate_index": index,
            "rung_index": int(index // 4),
            "direction_index": direction_index,
            "direction_label": inputs["direction_labels"][direction_index],
            "component_bound": float(inputs["component_bounds"][index]),
            "departure_coordinate_norm": radius,
            "full_state_rate_relative_error": _relative_error(
                predicted_full, total_rate
            ),
            "full_coordinate_rate_relative_error": _relative_error(
                predicted_online, exact_online
            ),
            "q162_rate_relative_error": _relative_error(
                predicted_online[blocks["q162"]], exact_online[blocks["q162"]]
            ),
            "z280_rate_relative_error": _relative_error(
                predicted_online[blocks["z280"]], exact_online[blocks["z280"]]
            ),
            "a28_rate_relative_error": _relative_error(
                predicted_online[blocks["a28"]], exact_online[blocks["a28"]]
            ),
            "decoder_full_state_relative_error": _relative_error(
                decoded_delta, exact_delta
            ),
            "decoder_coordinate_relative_mismatch": _relative_error(
                decoded_coordinate, coordinate
            ),
            "exact_radial_speed_per_second": exact_radial_speed,
            "predicted_radial_speed_per_second": predicted_radial_speed,
            "radial_sign_agrees": bool(
                np.sign(exact_radial_speed) == np.sign(predicted_radial_speed)
            ),
            "coordinate_Jacobian_rank": coordinate_metrics["rank"],
            "coordinate_Jacobian_condition_number": coordinate_metrics[
                "condition_number"
            ],
            "online_field_wall_seconds": online_wall,
            "online_state_dependent_coordinate_Jacobian_calls": 0,
            "offline_truth_coordinate_Jacobian_calls": 1,
            "decoded_minimum_reconstruction_factor": min(
                float(np.min(decoded_factors)),
                decoded_physical["minimum_reconstruction_factor"],
            ),
            "decoded_maximum_H_over_R": decoded_physical["maximum_h_over_r"],
            "decoded_minimum_scattering_optical_depth": decoded_physical[
                "minimum_scattering_optical_depth"
            ],
        }
    )
    progress["evaluations"].append(item)
    values = {
        "total_rates_per_second": total_rate,
        "free_rates_per_second": arrays["free_rate"],
        "physical_reaction_actions_per_second": arrays["reaction_action"],
        "multiplier_coordinates_per_second": arrays["multiplier"],
        "exact_online_470_coordinate_rates_per_second": exact_online,
        "predicted_online_470_coordinate_rates_per_second": predicted_online,
        "predicted_full_state_rates_per_second": predicted_full,
        "online_coordinates": coordinate,
        "repaired_decoded_scaled_deltas": decoded_delta,
        "decoded_online_coordinates": decoded_coordinate,
    }
    for name, value in values.items():
        progress[name] = _append(progress[name], value)


def _aggregate(progress: dict, resumed: int, began: float) -> dict:
    evaluations = progress["evaluations"]

    def values(name: str) -> list[float]:
        return [float(item[name]) for item in evaluations]

    def maximum(name: str, default=math.inf) -> float:
        entries = values(name)
        return max(entries) if entries else float(default)

    def minimum(name: str, default=-math.inf) -> float:
        entries = values(name)
        return min(entries) if entries else float(default)

    def median(name: str, default=math.inf) -> float:
        entries = values(name)
        return float(np.median(entries)) if entries else float(default)

    return {
        "planned_nonbase_rate_evaluations": manifest.PLANNED_RATE_EVALUATIONS,
        "completed_nonbase_rate_evaluations": len(evaluations),
        "failed_rate_evaluations": len(progress["failures"]),
        "failures": progress["failures"],
        "minimum_reconstruction_factor": minimum(
            "minimum_reconstruction_factor", math.inf
        ),
        "maximum_reconstruction_factor": maximum("maximum_reconstruction_factor"),
        "maximum_raw_Schur_condition_number": maximum(
            "raw_Schur_condition_number"
        ),
        "maximum_reaction_identity_defect": maximum("reaction_identity_defect"),
        "maximum_rate_tangency_relative_defect": maximum(
            "rate_tangency_relative_defect"
        ),
        "maximum_coordinate_Jacobian_condition_number": maximum(
            "coordinate_Jacobian_condition_number"
        ),
        "maximum_H_over_R": maximum("maximum_H_over_R"),
        "minimum_scattering_optical_depth": minimum(
            "minimum_scattering_optical_depth"
        ),
        "maximum_incoming_excision_characteristics": maximum(
            "incoming_excision_characteristics"
        ),
        "maximum_full_state_rate_relative_error": maximum(
            "full_state_rate_relative_error"
        ),
        "median_full_state_rate_relative_error": median(
            "full_state_rate_relative_error"
        ),
        "maximum_full_coordinate_rate_relative_error": maximum(
            "full_coordinate_rate_relative_error"
        ),
        "median_full_coordinate_rate_relative_error": median(
            "full_coordinate_rate_relative_error"
        ),
        "maximum_q162_rate_relative_error": maximum("q162_rate_relative_error"),
        "median_q162_rate_relative_error": median("q162_rate_relative_error"),
        "maximum_z280_rate_relative_error": maximum("z280_rate_relative_error"),
        "maximum_a28_rate_relative_error": maximum("a28_rate_relative_error"),
        "maximum_decoder_full_state_relative_error": maximum(
            "decoder_full_state_relative_error"
        ),
        "maximum_decoder_coordinate_relative_mismatch": maximum(
            "decoder_coordinate_relative_mismatch"
        ),
        "radial_sign_disagreement_count": int(
            sum(not item["radial_sign_agrees"] for item in evaluations)
        ),
        "online_state_dependent_coordinate_Jacobian_calls": int(
            sum(item["online_state_dependent_coordinate_Jacobian_calls"] for item in evaluations)
        ),
        "offline_truth_coordinate_Jacobian_calls": int(
            sum(item["offline_truth_coordinate_Jacobian_calls"] for item in evaluations)
        ),
        "total_online_field_wall_seconds": float(
            sum(values("online_field_wall_seconds"))
        ),
        "median_online_field_wall_seconds": median("online_field_wall_seconds"),
        "maximum_online_field_wall_seconds": maximum("online_field_wall_seconds"),
        "minimum_decoded_reconstruction_factor": minimum(
            "decoded_minimum_reconstruction_factor", math.inf
        ),
        "maximum_decoded_H_over_R": maximum("decoded_maximum_H_over_R"),
        "minimum_decoded_scattering_optical_depth": minimum(
            "decoded_minimum_scattering_optical_depth"
        ),
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "resumed_evaluation_count": resumed,
        "wall_seconds_this_process": time.perf_counter() - began,
        "evaluations": evaluations,
    }


def _execute(inputs: dict) -> tuple[dict, dict[str, np.ndarray]]:
    progress = _load_or_create_progress()
    resumed = len(progress["evaluations"])
    data = rate_engine.manifest.prior_screen.manifest.parent.manifest.failed_screen._anchor_data(
        "primary"
    )
    began = time.perf_counter()
    for index in range(resumed, manifest.PLANNED_RATE_EVALUATIONS):
        try:
            _evaluate_one(inputs, progress, index, data)
            status = "accepted"
        except Exception as error:  # fail closed on first exact-rate failure
            progress["failures"].append(
                {
                    "candidate_index": index,
                    "direction_index": int(inputs["direction_indices"][index]),
                    "direction_label": inputs["direction_labels"][
                        int(inputs["direction_indices"][index])
                    ],
                    "component_bound": float(inputs["component_bounds"][index]),
                    "reason": type(error).__name__,
                    "message": str(error),
                }
            )
            status = "failed"
        _save_progress(progress)
        print(
            json.dumps(
                {
                    "exact_rate_evaluation": index + 1,
                    "total": manifest.PLANNED_RATE_EVALUATIONS,
                    "direction": inputs["direction_labels"][
                        int(inputs["direction_indices"][index])
                    ],
                    "component_bound": float(inputs["component_bounds"][index]),
                    "status": status,
                    "elapsed_this_process_seconds": time.perf_counter() - began,
                }
            ),
            flush=True,
        )
        if progress["failures"]:
            break
    metrics = _aggregate(progress, resumed, began)
    count = len(progress["evaluations"])
    arrays = {
        "candidate_primitive_states": inputs["states"][:count],
        "candidate_scaled_deltas": inputs["deltas"][:count],
        "candidate_departure_coordinates": inputs["departures"][:count],
        "candidate_component_bounds": inputs["component_bounds"][:count],
        "candidate_direction_indices": inputs["direction_indices"][:count],
        **{name: progress[name] for name in _progress_array_shapes()},
    }
    return metrics, arrays


def _truth_checks(metrics: dict, gates: dict) -> dict:
    return {
        "completed": metrics["completed_nonbase_rate_evaluations"]
        == gates["completed_nonbase_rate_evaluations_equal"],
        "failed": metrics["failed_rate_evaluations"]
        == gates["failed_rate_evaluations_equal"],
        "reconstruction_minimum": metrics["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"],
        "reconstruction_maximum": metrics["maximum_reconstruction_factor"]
        <= gates["maximum_reconstruction_factor"],
        "Schur_condition": metrics["maximum_raw_Schur_condition_number"]
        <= gates["maximum_raw_Schur_condition_number"],
        "reaction_identity": metrics["maximum_reaction_identity_defect"]
        <= gates["maximum_reaction_identity_defect"],
        "rate_tangency": metrics["maximum_rate_tangency_relative_defect"]
        <= gates["maximum_rate_tangency_relative_defect"],
        "coordinate_condition": metrics[
            "maximum_coordinate_Jacobian_condition_number"
        ] <= gates["maximum_coordinate_Jacobian_condition_number"],
        "height": metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"],
        "optical_depth": metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"],
        "incoming_excision": metrics["maximum_incoming_excision_characteristics"]
        == gates["maximum_incoming_excision_characteristics_equal"],
        "generator_budget": metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": metrics["new_nonlinear_roots"] == 0,
        "propagation_budget": metrics["propagated_states"] == 0,
    }


def _field_checks(metrics: dict, gates: dict) -> dict:
    return {
        "maximum_full_state_rate": metrics[
            "maximum_full_state_rate_relative_error"
        ] <= gates["maximum_full_state_rate_relative_error"],
        "median_full_state_rate": metrics[
            "median_full_state_rate_relative_error"
        ] <= gates["median_full_state_rate_relative_error"],
        "maximum_full_coordinate_rate": metrics[
            "maximum_full_coordinate_rate_relative_error"
        ] <= gates["maximum_full_coordinate_rate_relative_error"],
        "median_full_coordinate_rate": metrics[
            "median_full_coordinate_rate_relative_error"
        ] <= gates["median_full_coordinate_rate_relative_error"],
        "maximum_q162_rate": metrics["maximum_q162_rate_relative_error"]
        <= gates["maximum_q162_rate_relative_error"],
        "median_q162_rate": metrics["median_q162_rate_relative_error"]
        <= gates["median_q162_rate_relative_error"],
        "maximum_z280_rate": metrics["maximum_z280_rate_relative_error"]
        <= gates["maximum_z280_rate_relative_error"],
        "maximum_a28_rate": metrics["maximum_a28_rate_relative_error"]
        <= gates["maximum_a28_rate_relative_error"],
        "radial_sign": metrics["radial_sign_disagreement_count"]
        == gates["radial_sign_disagreement_count_equal"],
        "decoder_full_state": metrics["maximum_decoder_full_state_relative_error"]
        <= gates["maximum_decoder_full_state_relative_error"],
        "decoder_coordinate": metrics[
            "maximum_decoder_coordinate_relative_mismatch"
        ] <= gates["maximum_decoder_coordinate_relative_mismatch"],
        "no_online_coordinate_Jacobian": metrics[
            "online_state_dependent_coordinate_Jacobian_calls"
        ] == gates["state_dependent_coordinate_Jacobian_calls_equal"],
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": MANIFEST_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("direct coordinate-field validation already canonicalized")
    inputs = _load_inputs(frozen)
    metrics, arrays = _execute(inputs)
    truth_checks = _truth_checks(
        metrics, frozen["contract"]["binding_exact_truth_gates"]
    )
    field_checks = _field_checks(
        metrics, frozen["contract"]["binding_independent_field_gates"]
    )
    truth_passed = all(truth_checks.values())
    field_passed = all(field_checks.values())
    passed = bool(truth_passed and field_passed)
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = PASS_AUTHORIZED_NEXT if passed else FAIL_AUTHORIZED_NEXT
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "truth_passed": truth_passed,
        "independent_coordinate_field_passed": field_passed,
        "completed_exact_rate_evaluations": metrics[
            "completed_nonbase_rate_evaluations"
        ],
        "failed_rate_evaluations": metrics["failed_rate_evaluations"],
        "maximum_full_state_rate_relative_error": metrics[
            "maximum_full_state_rate_relative_error"
        ],
        "maximum_full_coordinate_rate_relative_error": metrics[
            "maximum_full_coordinate_rate_relative_error"
        ],
        "maximum_q162_rate_relative_error": metrics[
            "maximum_q162_rate_relative_error"
        ],
        "maximum_z280_rate_relative_error": metrics[
            "maximum_z280_rate_relative_error"
        ],
        "maximum_a28_rate_relative_error": metrics[
            "maximum_a28_rate_relative_error"
        ],
        "radial_sign_disagreement_count": metrics[
            "radial_sign_disagreement_count"
        ],
        "maximum_decoder_full_state_relative_error": metrics[
            "maximum_decoder_full_state_relative_error"
        ],
        "median_online_field_wall_seconds": metrics[
            "median_online_field_wall_seconds"
        ],
        "online_state_dependent_coordinate_Jacobian_calls": metrics[
            "online_state_dependent_coordinate_Jacobian_calls"
        ],
        "offline_truth_coordinate_Jacobian_calls": metrics[
            "offline_truth_coordinate_Jacobian_calls"
        ],
        "coefficients_refit_after_holdout_truth": False,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "geometry_candidate_became_atlas_center": False,
        "stable_memory_remains_dynamic": True,
        "stationary_fast_graph_used": False,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "rate_metrics.json",
        {
            "truth_checks": truth_checks,
            "truth_passed": truth_passed,
            "field_checks": field_checks,
            "field_passed": field_passed,
            **metrics,
        },
    )
    _write_npz(CANONICAL_DIRECTORY / "rate_arrays.npz", arrays)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
            "holdout_geometry_sha256": _sha(manifest.HOLDOUT_GEOMETRY),
            "direct_coordinate_field_sha256": _sha(
                manifest.CANONICAL_DIRECTORY / "direct_coordinate_field.npz"
            ),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        manifest.parent.THIS_RUNNER,
        manifest.parent.THIS_TEST,
        prior_rate.THIS_RUNNER,
        prior_rate.THIS_TEST,
        vector_field.THIS_RUNNER,
        vector_field.THIS_TEST,
        rate_engine.THIS_RUNNER,
        rate_engine.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "resumed_from_evaluation_count": metrics["resumed_evaluation_count"],
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "exact_rate_engine": rate_engine.manifest.prior_screen.THIS_RUNNER,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name) for name in _thread_environment()
            },
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Independent direct coordinate-field validation WP10c9d6c7c3b5c4f25co",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{summary['completed_exact_rate_evaluations']}` of `{manifest.PLANNED_RATE_EVALUATIONS}` frozen exact-rate evaluations with `{summary['failed_rate_evaluations']}` failures. Truth admissibility passed: `{truth_passed}`; independent direct-field validation passed: `{field_passed}`.",
                "",
                f"Maximum full-state/full-coordinate/q162/z280/a28 rate errors are `{metrics['maximum_full_state_rate_relative_error']:.6e}`, `{metrics['maximum_full_coordinate_rate_relative_error']:.6e}`, `{metrics['maximum_q162_rate_relative_error']:.6e}`, `{metrics['maximum_z280_rate_relative_error']:.6e}`, and `{metrics['maximum_a28_rate_relative_error']:.6e}`. There were `{metrics['radial_sign_disagreement_count']}` radial-sign disagreements.",
                "",
                f"Median direct online field time was `{metrics['median_online_field_wall_seconds']:.6e}` seconds. It made `{metrics['online_state_dependent_coordinate_Jacobian_calls']}` online coordinate-Jacobian calls; `{metrics['offline_truth_coordinate_Jacobian_calls']}` coordinate Jacobians were built solely to construct independent truth.",
                "",
                f"The compensated decoder maximum full-state error is `{metrics['maximum_decoder_full_state_relative_error']:.6e}`. Coefficients were not refit after holdout truth.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No state was propagated and no physical microburst, predictive cycle, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    _run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
