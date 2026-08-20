#!/usr/bin/env python3
"""Validate the frozen shell-gated local atlas on independent exact rates."""

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

import run_causal_inner_departure28_short_vector_field_validation_wp10c9d6c7c3b5c4f25bz as vector_field  # noqa: E402
import run_causal_inner_expanded_departure_rate_screen_wp10c9d6c7c3b5c4f25be as rate_engine  # noqa: E402
import run_causal_inner_shell_gated_atlas_rate_manifest_wp10c9d6c7c3b5c4f25cj as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ck"
MANIFEST_COMMIT = "6f2c3cc6398330ef636bd8c876aa76153a50336c"
MANIFEST_PARENT = "dddeed7d7358daaa1edf2965f8dcb22a354688ea"
MANIFEST_TREE = "f3c44bbb8b7ce3e495dd1e22b0578b3cc14c30a3"

PASS_CLASSIFICATION = "shell_gated_degree45_atlas_field_independently_validated"
FAIL_CLASSIFICATION = (
    "shell_gated_degree45_atlas_field_independent_validation_failed"
)
PASS_AUTHORIZED_NEXT = (
    "definitions_only_authentic_recentered_transition_forecast_manifest"
)
FAIL_AUTHORIZED_NEXT = "definitions_only_alternative_local_rate_extension_manifest"

ARTIFACT = (
    "causal_inner_shell_gated_atlas_rate_validation_"
    "wp10c9d6c7c3b5c4f25ck"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_shell_gated_atlas_rate_validation_"
    "wp10c9d6c7c3b5c4f25ck.py"
)
THIS_TEST = (
    "tests/test_causal_inner_shell_gated_atlas_rate_validation_"
    "wp10c9d6c7c3b5c4f25ck.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SHELL_GATED_ATLAS_"
    "RATE_VALIDATION_WP10C9D6C7C3B5C4F25CK_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PROGRESS_JSON = SCRATCH_DIRECTORY / "progress.json"
PROGRESS_NPZ = SCRATCH_DIRECTORY / "progress.npz"

atlas = manifest.parent.manifest.parent

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


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("independent atlas rate manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("independent atlas rate manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("independent atlas rate manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    lock = _read(manifest.CANONICAL_DIRECTORY / "parent_lock.json")
    geometry = _load_npz(manifest.GEOMETRY_ARRAYS)
    extension = _load_npz(manifest.EXTENSION_ARRAYS)
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_exact_rate_evaluations"]
        != manifest.PLANNED_RATE_EVALUATIONS
        or not summary["coefficients_frozen_before_truth"]
        or summary["new_truth_calls"] != 0
        or summary["trajectory_authorized"]
        or contract["decision"]["pass"]["classification"]
        != PASS_CLASSIFICATION
        or contract["decision"]["fail"]["classification"]
        != FAIL_CLASSIFICATION
        or geometry["candidate_primitive_states"].shape != (8, 112, 5)
        or extension["full_state_rate_even4_coefficients"].shape != (4, 560)
    ):
        raise RuntimeError("independent atlas exact-rate contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"atlas rate manifest source changed: {relative}")
    if _sha(manifest.GEOMETRY_ARRAYS) != lock["holdout_geometry_sha256"]:
        raise RuntimeError("mixed holdout geometry changed")
    if _sha(manifest.EXTENSION_ARRAYS) != lock["extension_coefficients_sha256"]:
        raise RuntimeError("frozen atlas extension changed")
    thread_environment = (
        vector_field.manifest.parent.geometry.chart_tools.coordinate_tools.THREAD_ENVIRONMENT
    )
    for name, expected in thread_environment.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("independent atlas rate validation requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "hashes": hashes,
        "geometry": geometry,
        "extension": extension,
    }


def _load_inputs(frozen: dict) -> dict:
    model = vector_field.ReducedVectorField()
    geometry = frozen["geometry"]
    extension = frozen["extension"]
    labels = tuple(
        _read(atlas.CANONICAL_DIRECTORY / "holdout_design.json")["labels"]
    )
    states = np.asarray(geometry["candidate_primitive_states"], dtype=float)
    deltas = np.asarray(geometry["candidate_scaled_deltas"], dtype=float)
    departures = np.asarray(
        geometry["candidate_departure_coordinates"], dtype=float
    )
    bounds = np.asarray(geometry["candidate_component_bounds"], dtype=float)
    direction_indices = np.asarray(
        geometry["candidate_direction_indices"], dtype=int
    )
    if (
        states.shape != (8, 112, 5)
        or deltas.shape != (8, 560)
        or departures.shape != (8, 28)
        or bounds.shape != (8,)
        or direction_indices.shape != (8,)
        or tuple(direction_indices) != (0, 1, 2, 3, 0, 1, 2, 3)
        or tuple(bounds[:4]) != (0.0125,) * 4
        or tuple(bounds[4:]) != (0.015,) * 4
        or len(labels) != 4
        or model.generator.shape != (560, 560)
        or model.memory_basis.shape != (560, 280)
        or model.departure_basis.shape != (560, 28)
    ):
        raise RuntimeError("independent atlas rate input dimensions changed")
    return {
        "model": model,
        "states": states,
        "deltas": deltas,
        "departures": departures,
        "component_bounds": bounds,
        "direction_indices": direction_indices,
        "direction_labels": labels,
        "extension": extension,
    }


def _predict_atlas(
    model: vector_field.ReducedVectorField,
    extension: dict[str, np.ndarray],
    exact_delta: np.ndarray,
    departure: np.ndarray,
) -> dict:
    coordinate = np.concatenate(
        (
            np.zeros(162, dtype=float),
            model.memory_basis.T @ np.asarray(exact_delta, dtype=float),
            np.asarray(departure, dtype=float),
        )
    )
    old_delta = model.decoded_delta(coordinate)
    load = float(np.max(np.abs(old_delta)))
    weight = atlas._shell_weight(load)
    centers = np.asarray(extension["extension_center_directions"], dtype=float)
    decoder_correction = atlas._extension_value(
        departure,
        centers,
        np.asarray(extension["decoder_even4_coefficients"], dtype=float),
        np.asarray(extension["decoder_odd5_coefficients"], dtype=float),
    )
    decoded_delta = old_delta + weight * decoder_correction
    decoded_state = model.base_state + (
        model.columns.ravel() * decoded_delta
    ).reshape(model.base_state.shape)
    decoded_coordinate, factors = model.coordinate(decoded_state)
    physical = vector_field.manifest.parent.geometry.chart_tools._state_audit(
        model.components["context"], decoded_state
    )
    baseline_rate = (
        model.base_rate
        + model.generator @ decoded_delta
        + model.departure_basis @ model.nonlinear_departure(departure)
    )
    rate_correction = atlas._extension_value(
        departure,
        centers,
        np.asarray(
            extension["full_state_rate_even4_coefficients"], dtype=float
        ),
        np.asarray(
            extension["full_state_rate_odd5_coefficients"], dtype=float
        ),
    )
    predicted_rate = baseline_rate + weight * rate_correction
    return {
        "online_coordinate": coordinate,
        "old_decoded_delta": old_delta,
        "decoded_delta": decoded_delta,
        "decoded_state": decoded_state,
        "decoded_coordinate": decoded_coordinate,
        "predicted_full_state_rate": predicted_rate,
        "predicted_a28_rate": model.departure_basis.T @ predicted_rate,
        "shell_load": load,
        "shell_weight": weight,
        "minimum_reconstruction_factor": min(
            float(np.min(factors)), physical["minimum_reconstruction_factor"]
        ),
        "maximum_H_over_R": physical["maximum_h_over_r"],
        "minimum_scattering_optical_depth": physical[
            "minimum_scattering_optical_depth"
        ],
    }


def _progress_array_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "total_rates_per_second": (560,),
        "free_rates_per_second": (560,),
        "physical_reaction_actions_per_second": (560,),
        "multiplier_coordinates_per_second": (3,),
        "exact_online_470_coordinate_rates_per_second": (470,),
        "exact_a28_rates_per_second": (28,),
        "predicted_full_state_rates_per_second": (560,),
        "predicted_a28_rates_per_second": (28,),
        "online_coordinates": (470,),
        "old_decoded_scaled_deltas": (560,),
        "extended_decoded_scaled_deltas": (560,),
        "decoded_online_coordinates": (470,),
    }


def _progress_identity() -> dict:
    return {
        "work_package": WORK_PACKAGE,
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_hashes": _checksums(manifest.CANONICAL_DIRECTORY),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "holdout_geometry_sha256": _sha(manifest.GEOMETRY_ARRAYS),
        "extension_coefficients_sha256": _sha(manifest.EXTENSION_ARRAYS),
    }


def _empty_progress() -> dict:
    progress = {
        "identity": _progress_identity(),
        "evaluations": [],
        "failures": [],
    }
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
        raise RuntimeError("independent atlas rate scratch checkpoint is incomplete")
    recorded = _read(PROGRESS_JSON)
    if recorded["identity"] != _progress_identity():
        raise RuntimeError("independent atlas rate scratch identity changed")
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
        raise RuntimeError("independent atlas rate scratch dimensions changed")
    if [item["candidate_index"] for item in progress["evaluations"]] != list(
        range(count)
    ):
        raise RuntimeError("independent atlas rate scratch ordering changed")
    return progress


def _evaluate_one(
    inputs: dict, progress: dict, index: int, data: dict
) -> None:
    model = inputs["model"]
    state = inputs["states"][index]
    exact_delta = inputs["deltas"][index]
    departure = inputs["departures"][index]
    item, arrays = rate_engine.manifest.prior_screen._continuous_rate(data, state)
    coordinate_jacobian, coordinate_metrics = (
        vector_field.manifest.parent.geometry.chart_tools._coordinate_jacobian(
            state, model.components
        )
    )
    prediction = _predict_atlas(
        model, inputs["extension"], exact_delta, departure
    )
    total_rate = np.asarray(arrays["total_rate"], dtype=float)
    exact_a28_rate = model.departure_basis.T @ total_rate
    predicted_rate = prediction["predicted_full_state_rate"]
    predicted_a28_rate = prediction["predicted_a28_rate"]
    radius = float(np.linalg.norm(departure))
    exact_radial_speed = float(departure @ exact_a28_rate / radius)
    predicted_radial_speed = float(departure @ predicted_a28_rate / radius)
    direction_index = int(inputs["direction_indices"][index])
    coordinate_mismatch = _relative_error(
        prediction["decoded_coordinate"], prediction["online_coordinate"]
    )
    item.update(
        {
            "candidate_index": index,
            "rung_index": int(index // 4),
            "direction_index": direction_index,
            "direction_label": inputs["direction_labels"][direction_index],
            "component_bound": float(inputs["component_bounds"][index]),
            "departure_coordinate_norm": radius,
            "shell_load": prediction["shell_load"],
            "shell_weight": prediction["shell_weight"],
            "full_state_rate_relative_error": _relative_error(
                predicted_rate, total_rate
            ),
            "a28_rate_relative_error": _relative_error(
                predicted_a28_rate, exact_a28_rate
            ),
            "decoder_full_state_relative_error": _relative_error(
                prediction["decoded_delta"], exact_delta
            ),
            "old_decoder_full_state_relative_error": _relative_error(
                prediction["old_decoded_delta"], exact_delta
            ),
            "decoder_coordinate_relative_mismatch": coordinate_mismatch,
            "exact_radial_speed_per_second": exact_radial_speed,
            "predicted_radial_speed_per_second": predicted_radial_speed,
            "exact_radial_direction_cosine": float(
                exact_radial_speed
                / max(float(np.linalg.norm(exact_a28_rate)), np.finfo(float).tiny)
            ),
            "predicted_radial_direction_cosine": float(
                predicted_radial_speed
                / max(
                    float(np.linalg.norm(predicted_a28_rate)),
                    np.finfo(float).tiny,
                )
            ),
            "radial_sign_agrees": bool(
                np.sign(exact_radial_speed) == np.sign(predicted_radial_speed)
            ),
            "coordinate_Jacobian_rank": coordinate_metrics["rank"],
            "coordinate_Jacobian_condition_number": coordinate_metrics[
                "condition_number"
            ],
            "decoded_minimum_reconstruction_factor": prediction[
                "minimum_reconstruction_factor"
            ],
            "decoded_maximum_H_over_R": prediction["maximum_H_over_R"],
            "decoded_minimum_scattering_optical_depth": prediction[
                "minimum_scattering_optical_depth"
            ],
        }
    )
    exact_online_rate = np.concatenate(
        (
            coordinate_jacobian @ total_rate,
            model.memory_basis.T @ total_rate,
            exact_a28_rate,
        )
    )
    progress["evaluations"].append(item)
    values = {
        "total_rates_per_second": total_rate,
        "free_rates_per_second": arrays["free_rate"],
        "physical_reaction_actions_per_second": arrays["reaction_action"],
        "multiplier_coordinates_per_second": arrays["multiplier"],
        "exact_online_470_coordinate_rates_per_second": exact_online_rate,
        "exact_a28_rates_per_second": exact_a28_rate,
        "predicted_full_state_rates_per_second": predicted_rate,
        "predicted_a28_rates_per_second": predicted_a28_rate,
        "online_coordinates": prediction["online_coordinate"],
        "old_decoded_scaled_deltas": prediction["old_decoded_delta"],
        "extended_decoded_scaled_deltas": prediction["decoded_delta"],
        "decoded_online_coordinates": prediction["decoded_coordinate"],
    }
    for name, value in values.items():
        progress[name] = _append(progress[name], value)


def _aggregate(inputs: dict, progress: dict, resumed: int, began: float) -> dict:
    evaluations = progress["evaluations"]

    def maximum(name: str, default=math.inf) -> float:
        values = [item[name] for item in evaluations]
        return float(max(values)) if values else float(default)

    def minimum(name: str, default=-math.inf) -> float:
        values = [item[name] for item in evaluations]
        return float(min(values)) if values else float(default)

    rate_errors = [item["full_state_rate_relative_error"] for item in evaluations]
    return {
        "planned_nonbase_rate_evaluations": manifest.PLANNED_RATE_EVALUATIONS,
        "completed_nonbase_rate_evaluations": len(evaluations),
        "failed_rate_evaluations": len(progress["failures"]),
        "failures": progress["failures"],
        "minimum_reconstruction_factor": minimum(
            "minimum_reconstruction_factor", math.inf
        ),
        "maximum_reconstruction_factor": maximum(
            "maximum_reconstruction_factor"
        ),
        "maximum_raw_Schur_condition_number": maximum(
            "raw_Schur_condition_number"
        ),
        "maximum_reaction_identity_defect": maximum(
            "reaction_identity_defect"
        ),
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
        "median_full_state_rate_relative_error": (
            float(np.median(rate_errors)) if rate_errors else math.inf
        ),
        "maximum_a28_rate_relative_error": maximum("a28_rate_relative_error"),
        "maximum_decoder_full_state_relative_error": maximum(
            "decoder_full_state_relative_error"
        ),
        "maximum_old_decoder_full_state_relative_error": maximum(
            "old_decoder_full_state_relative_error"
        ),
        "maximum_decoder_coordinate_relative_mismatch": maximum(
            "decoder_coordinate_relative_mismatch"
        ),
        "radial_sign_disagreement_count": int(
            sum(not item["radial_sign_agrees"] for item in evaluations)
        ),
        "outward_candidate_count": int(
            sum(item["exact_radial_speed_per_second"] > 0.0 for item in evaluations)
        ),
        "inward_candidate_count": int(
            sum(item["exact_radial_speed_per_second"] < 0.0 for item in evaluations)
        ),
        "minimum_shell_weight": minimum("shell_weight", math.inf),
        "maximum_shell_weight": maximum("shell_weight"),
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
        except Exception as error:  # fail closed on the first exact-rate failure
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
    metrics = _aggregate(inputs, progress, resumed, began)
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
        "generator_budget": metrics["new_complete_generator_assemblies"]
        == gates["new_complete_generator_assemblies_equal"],
        "root_budget": metrics["new_nonlinear_roots"]
        == gates["new_nonlinear_roots_equal"],
        "propagation_budget": metrics["propagated_states"]
        == gates["propagated_states_equal"],
    }


def _model_checks(metrics: dict, gates: dict) -> dict:
    return {
        "maximum_full_state_rate_error": metrics[
            "maximum_full_state_rate_relative_error"
        ] <= gates["maximum_full_state_rate_relative_error"],
        "median_full_state_rate_error": metrics[
            "median_full_state_rate_relative_error"
        ] <= gates["median_full_state_rate_relative_error"],
        "maximum_a28_rate_error": metrics["maximum_a28_rate_relative_error"]
        <= gates["maximum_a28_rate_relative_error"],
        "radial_sign": metrics["radial_sign_disagreement_count"]
        == gates["radial_sign_disagreement_count_equal"],
        "decoder_full_state_error": metrics[
            "maximum_decoder_full_state_relative_error"
        ] <= gates["maximum_decoder_full_state_relative_error"],
        "decoder_coordinate_mismatch": metrics[
            "maximum_decoder_coordinate_relative_mismatch"
        ] <= gates["maximum_decoder_coordinate_relative_mismatch"],
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
            fieldnames=[
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ],
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
        raise RuntimeError("independent atlas rate validation already canonicalized")
    inputs = _load_inputs(frozen)
    metrics, arrays = _execute(inputs)
    truth_checks = _truth_checks(
        metrics, frozen["contract"]["binding_exact_rate_gates"]
    )
    model_checks = _model_checks(
        metrics, frozen["contract"]["binding_independent_model_gates"]
    )
    truth_passed = all(truth_checks.values())
    model_passed = all(model_checks.values())
    passed = bool(truth_passed and model_passed)
    if passed:
        classification = PASS_CLASSIFICATION
        authorized_next = PASS_AUTHORIZED_NEXT
    else:
        classification = FAIL_CLASSIFICATION
        authorized_next = FAIL_AUTHORIZED_NEXT
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "truth_passed": truth_passed,
        "independent_model_passed": model_passed,
        "completed_exact_rate_evaluations": metrics[
            "completed_nonbase_rate_evaluations"
        ],
        "failed_rate_evaluations": metrics["failed_rate_evaluations"],
        "maximum_full_state_rate_relative_error": metrics[
            "maximum_full_state_rate_relative_error"
        ],
        "median_full_state_rate_relative_error": metrics[
            "median_full_state_rate_relative_error"
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
        "maximum_decoder_coordinate_relative_mismatch": metrics[
            "maximum_decoder_coordinate_relative_mismatch"
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
            "model_checks": model_checks,
            "model_passed": model_passed,
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
            "holdout_geometry_sha256": _sha(manifest.GEOMETRY_ARRAYS),
            "extension_coefficients_sha256": _sha(manifest.EXTENSION_ARRAYS),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        manifest.parent.THIS_RUNNER,
        manifest.parent.THIS_TEST,
        atlas.THIS_RUNNER,
        atlas.THIS_TEST,
        vector_field.THIS_RUNNER,
        vector_field.THIS_TEST,
        rate_engine.THIS_RUNNER,
        rate_engine.THIS_TEST,
    )
    thread_environment = (
        vector_field.manifest.parent.geometry.chart_tools.coordinate_tools.THREAD_ENVIRONMENT
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
                name: os.environ.get(name) for name in thread_environment
            },
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Independent shell-gated atlas rate validation WP10c9d6c7c3b5c4f25ck",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{summary['completed_exact_rate_evaluations']}` of `{manifest.PLANNED_RATE_EVALUATIONS}` frozen mixed-state exact continuous-rate evaluations with `{summary['failed_rate_evaluations']}` failures.",
                "",
                f"Truth admissibility passed: `{truth_passed}`. Independent model validation passed: `{model_passed}`. The maximum/median full-state rate errors are `{metrics['maximum_full_state_rate_relative_error']:.6e}` and `{metrics['median_full_state_rate_relative_error']:.6e}`; maximum a28 rate error is `{metrics['maximum_a28_rate_relative_error']:.6e}` with `{metrics['radial_sign_disagreement_count']}` radial-sign disagreements.",
                "",
                f"The extended decoder maximum full-state error is `{metrics['maximum_decoder_full_state_relative_error']:.6e}` and its maximum online-coordinate mismatch is `{metrics['maximum_decoder_coordinate_relative_mismatch']:.6e}`. Coefficients were not refit after holdout truth.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No state was propagated, no geometry holdout became a chart center, the 280D memory remains dynamic, and no physical microburst, cycle evolution, or reduced slow evolution is authorized.",
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
