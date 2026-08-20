#!/usr/bin/env python3
"""Evaluate five frozen rates and fit the authentic-center local field."""

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

import run_causal_inner_authentic_center_exact_rate_training_manifest_wp10c9d6c7c3b5c4f25ct as manifest  # noqa: E402
import run_causal_inner_direct_coordinate_field_validation_wp10c9d6c7c3b5c4f25co as direct_validation  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cu"
MANIFEST_COMMIT = "ff54ea0fc99d227b099c9e4842add420ecd4f2ac"
MANIFEST_PARENT = "c72fb45b95546e13c0d8feb7c29645bb8e6e41e1"
MANIFEST_TREE = "5cb6b3ad5210aab795afa40e669ab00957c480bd"

PASS_CLASSIFICATION = "authentic_center_local_field_coefficients_frozen"
FAIL_CLASSIFICATION = "authentic_center_local_field_training_failed"
PASS_AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25cv"
FAIL_AUTHORIZED_NEXT = "definitions_only_local_field_fit_revision"

EXACT_RATE_COUNT = 5
REVEALED_COUNT = manifest.REVEALED_COUNT
PHYSICAL_DIMENSION = 162
MEMORY_DIMENSION = 280
DEPARTURE_DIMENSION = 28

ARTIFACT = (
    "causal_inner_authentic_center_exact_rate_training_"
    "wp10c9d6c7c3b5c4f25cu"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_authentic_center_exact_rate_training_"
    "wp10c9d6c7c3b5c4f25cu.py"
)
THIS_TEST = (
    "tests/test_causal_inner_authentic_center_exact_rate_training_"
    "wp10c9d6c7c3b5c4f25cu.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_AUTHENTIC_CENTER_EXACT_RATE_"
    "TRAINING_WP10C9D6C7C3B5C4F25CU_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PROGRESS_JSON = SCRATCH_DIRECTORY / "progress.json"
PROGRESS_NPZ = SCRATCH_DIRECTORY / "progress.npz"

FROZEN_DESIGN = manifest.CANONICAL_DIRECTORY / "frozen_rate_training_design.npz"
DIRECT_FIELD = manifest.DIRECT_FIELD

rate_engine = direct_validation.rate_engine
vector_field = direct_validation.vector_field
direct_manifest = manifest.parent.manifest.direct_manifest

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


def _append(array: np.ndarray, value: np.ndarray) -> np.ndarray:
    item = np.asarray(value, dtype=float)
    return np.concatenate((array, item.reshape((1,) + item.shape)), axis=0)


def _relative_rows(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    return np.linalg.norm(left - right, axis=1) / np.maximum(
        np.linalg.norm(right, axis=1), np.finfo(float).tiny
    )


def _thread_environment() -> dict[str, str]:
    return manifest.parent.manifest.parent.THREAD_ENVIRONMENT


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("exact-rate training manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("exact-rate training manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("exact-rate training manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    lock = _read(manifest.CANONICAL_DIRECTORY / "parent_lock.json")
    design = _load_npz(FROZEN_DESIGN)
    expected_order = [
        "authentic_center_exact_continuous_rate",
        "training_0_exact_continuous_rate",
        "training_1_exact_continuous_rate",
        "training_2_exact_continuous_rate",
        "training_3_exact_continuous_rate",
        "fit_and_hash_local_field_coefficients",
    ]
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_new_training_exact_rate_calls"] != EXACT_RATE_COUNT
        or summary["planned_future_blind_holdout_exact_rate_calls"] != 4
        or not summary["coefficients_frozen_before_holdout_truth"]
        or summary["new_truth_rate_calls"] != 0
        or contract["execution_order"] != expected_order
        or contract["cost_budget"]["new_exact_continuous_rate_calls_equal"]
        != EXACT_RATE_COUNT
        or contract["cost_budget"]["holdout_rate_calls_equal"] != 0
        or contract["decision"]["pass_classification"] != PASS_CLASSIFICATION
        or contract["decision"]["fail_classification"] != FAIL_CLASSIFICATION
        or design["authentic_center_primitive_state"].shape != (112, 5)
        or design["training_primitive_states"].shape != (4, 112, 5)
        or design["holdout_primitive_states"].shape != (4, 112, 5)
        or design["revealed_overlap_exact_full_rates_per_second"].shape
        != (REVEALED_COUNT, 560)
        or design["revealed_overlap_exact_coordinate_rates_per_second"].shape
        != (REVEALED_COUNT, 470)
        or design["decoder_affine_coefficients"].shape != (3, 560)
    ):
        raise RuntimeError("exact-rate training execution contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"exact-rate training source changed: {relative}")
    if (
        _sha(FROZEN_DESIGN) != hashes["frozen_rate_training_design.npz"]
        or _sha(DIRECT_FIELD) != lock["direct_field_sha256"]
    ):
        raise RuntimeError("frozen local-field training input changed")
    for name, expected in _thread_environment().items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("exact-rate training requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "hashes": hashes,
        "design": design,
    }


def _load_inputs(frozen: dict) -> dict:
    design = frozen["design"]
    model = vector_field.ReducedVectorField()
    direct = direct_manifest.DirectCoordinateField(
        _load_npz(DIRECT_FIELD), model=model
    )
    states = np.concatenate(
        (
            design["authentic_center_primitive_state"][None, ...],
            design["training_primitive_states"],
        ),
        axis=0,
    )
    coordinates = np.vstack(
        (
            design["authentic_center_absolute_coordinate"],
            design["training_absolute_coordinates"],
        )
    )
    local_coordinates = np.vstack(
        (np.zeros(470), design["training_local_coordinates"])
    )
    reconstructed = []
    factors = []
    for state in states:
        coordinate, reconstruction = model.coordinate(state)
        reconstructed.append(coordinate)
        factors.append(np.asarray(reconstruction, dtype=float))
    reconstructed = np.asarray(reconstructed)
    mismatch = _relative_rows(reconstructed, coordinates)
    translation_defect = float(
        np.max(
            np.abs(
                coordinates[1:]
                - coordinates[0]
                - local_coordinates[1:]
            )
        )
    )
    if (
        states.shape != (EXACT_RATE_COUNT, 112, 5)
        or coordinates.shape != (EXACT_RATE_COUNT, 470)
        or local_coordinates.shape != (EXACT_RATE_COUNT, 470)
        or float(np.max(mismatch)) > 1.0e-8
        or translation_defect > 1.0e-15
    ):
        raise RuntimeError("frozen center/training inputs changed")
    return {
        "model": model,
        "direct": direct,
        "states": states,
        "absolute_coordinates": coordinates,
        "local_coordinates": local_coordinates,
        "coordinate_roundtrip_relative_errors": mismatch,
        "maximum_local_translation_absolute_defect": translation_defect,
        "minimum_input_reconstruction_factor": float(
            min(np.min(value) for value in factors)
        ),
        "labels": ("authentic_center", "training_0", "training_1", "training_2", "training_3"),
    }


def _progress_array_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "total_rates_per_second": (560,),
        "free_rates_per_second": (560,),
        "physical_reaction_actions_per_second": (560,),
        "multiplier_coordinates_per_second": (3,),
        "exact_coordinate_rates_per_second": (470,),
        "coordinate_jacobians": (PHYSICAL_DIMENSION, 560),
    }


def _progress_identity() -> dict:
    return {
        "work_package": WORK_PACKAGE,
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_hashes": _checksums(manifest.CANONICAL_DIRECTORY),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "frozen_design_sha256": _sha(FROZEN_DESIGN),
    }


def _empty_progress() -> dict:
    progress = {"identity": _progress_identity(), "evaluations": [], "failures": []}
    for name, shape in _progress_array_shapes().items():
        progress[name] = np.empty((0,) + shape, dtype=float)
    return progress


def _save_progress(progress: dict) -> None:
    SCRATCH_DIRECTORY.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError("exact-rate training checkpoint is incomplete")
    recorded = _read(PROGRESS_JSON)
    if recorded["identity"] != _progress_identity():
        raise RuntimeError("exact-rate training checkpoint identity changed")
    progress = {
        "identity": recorded["identity"],
        "evaluations": recorded["evaluations"],
        "failures": recorded["failures"],
        **_load_npz(PROGRESS_NPZ),
    }
    count = len(progress["evaluations"])
    if (
        count > EXACT_RATE_COUNT
        or [item["candidate_index"] for item in progress["evaluations"]]
        != list(range(count))
        or any(
            progress[name].shape != (count,) + shape
            for name, shape in _progress_array_shapes().items()
        )
    ):
        raise RuntimeError("exact-rate training checkpoint dimensions changed")
    return progress


def _evaluate_one(inputs: dict, progress: dict, index: int, data: dict) -> None:
    state = inputs["states"][index]
    model = inputs["model"]
    item, arrays = rate_engine.manifest.prior_screen._continuous_rate(data, state)
    coordinate_jacobian, coordinate_metrics = (
        vector_field.manifest.parent.geometry.chart_tools._coordinate_jacobian(
            state, model.components
        )
    )
    total_rate = np.asarray(arrays["total_rate"], dtype=float)
    exact_coordinate = np.concatenate(
        (
            coordinate_jacobian @ total_rate,
            model.memory_basis.T @ total_rate,
            model.departure_basis.T @ total_rate,
        )
    )
    item.update(
        {
            "candidate_index": index,
            "candidate_label": inputs["labels"][index],
            "center_local_scaled_load": float(
                np.max(np.abs(inputs["local_coordinates"][index]))
            ),
            "coordinate_roundtrip_relative_error": float(
                inputs["coordinate_roundtrip_relative_errors"][index]
            ),
            "coordinate_Jacobian_rank": coordinate_metrics["rank"],
            "coordinate_Jacobian_condition_number": coordinate_metrics[
                "condition_number"
            ],
            "offline_truth_coordinate_Jacobian_calls": 1,
        }
    )
    progress["evaluations"].append(item)
    values = {
        "total_rates_per_second": total_rate,
        "free_rates_per_second": arrays["free_rate"],
        "physical_reaction_actions_per_second": arrays["reaction_action"],
        "multiplier_coordinates_per_second": arrays["multiplier"],
        "exact_coordinate_rates_per_second": exact_coordinate,
        "coordinate_jacobians": coordinate_jacobian,
    }
    for name, value in values.items():
        progress[name] = _append(progress[name], value)


def _execute(inputs: dict) -> tuple[dict, dict[str, np.ndarray]]:
    progress = _load_or_create_progress()
    resumed = len(progress["evaluations"])
    data = (
        rate_engine.manifest.prior_screen.manifest.parent.manifest.failed_screen._anchor_data(
            "primary"
        )
    )
    began = time.perf_counter()
    for index in range(resumed, EXACT_RATE_COUNT):
        try:
            _evaluate_one(inputs, progress, index, data)
            status = "accepted"
        except Exception as error:  # preserve the first failed truth evaluation
            progress["failures"].append(
                {
                    "candidate_index": index,
                    "candidate_label": inputs["labels"][index],
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
                    "total": EXACT_RATE_COUNT,
                    "candidate": inputs["labels"][index],
                    "status": status,
                    "elapsed_this_process_seconds": time.perf_counter() - began,
                }
            ),
            flush=True,
        )
        if progress["failures"]:
            break
    count = len(progress["evaluations"])
    arrays = {
        "evaluated_primitive_states": inputs["states"][:count],
        "evaluated_absolute_coordinates": inputs["absolute_coordinates"][:count],
        "evaluated_local_coordinates": inputs["local_coordinates"][:count],
        **{name: progress[name] for name in _progress_array_shapes()},
    }
    evaluations = progress["evaluations"]

    def values(name: str) -> list[float]:
        return [float(item[name]) for item in evaluations]

    def maximum(name: str, default=math.inf) -> float:
        entries = values(name)
        return max(entries) if entries else float(default)

    def minimum(name: str, default=-math.inf) -> float:
        entries = values(name)
        return min(entries) if entries else float(default)

    metrics = {
        "planned_exact_rate_calls": EXACT_RATE_COUNT,
        "completed_exact_rate_calls": count,
        "failed_exact_rate_calls": len(progress["failures"]),
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
        "maximum_incoming_excision_characteristics": int(
            maximum("incoming_excision_characteristics", 1.0e9)
        ),
        "maximum_coordinate_roundtrip_relative_error": maximum(
            "coordinate_roundtrip_relative_error"
        ),
        "new_exact_continuous_rate_calls": count - resumed,
        "resumed_exact_rate_calls": resumed,
        "total_available_exact_rate_calls": count,
        "offline_truth_coordinate_Jacobian_calls": count,
        "holdout_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "wall_seconds_this_process": time.perf_counter() - began,
        "evaluations": evaluations,
    }
    return metrics, arrays


class AuthenticCenterLocalField:
    """Frozen center-local affine correction over the validated direct field."""

    def __init__(
        self,
        closure: dict[str, np.ndarray],
        *,
        model=None,
        direct=None,
    ):
        self.model = model or vector_field.ReducedVectorField()
        self.direct = direct or direct_manifest.DirectCoordinateField(
            _load_npz(DIRECT_FIELD), model=self.model
        )
        self.center_coordinate = np.asarray(
            closure["authentic_center_absolute_coordinate"], dtype=float
        )
        self.center_delta = np.asarray(
            closure["authentic_center_scaled_delta"], dtype=float
        )
        self.center_direct_delta = np.asarray(
            closure["authentic_center_direct_decoded_scaled_delta"], dtype=float
        )
        self.active_basis = np.asarray(
            closure["active_departure_basis"], dtype=float
        )
        self.decoder_coefficients = np.asarray(
            closure["decoder_affine_coefficients"], dtype=float
        )
        self.full_rate_coefficients = np.asarray(
            closure["full_rate_affine_coefficients"], dtype=float
        )
        self.q_rate_coefficients = np.asarray(
            closure["q162_rate_affine_coefficients"], dtype=float
        )
        self.restriction = np.asarray(
            closure["authentic_center_fixed_restriction"], dtype=float
        )

    def active_coordinates(self, local_coordinate: np.ndarray) -> np.ndarray:
        eta = np.asarray(local_coordinate, dtype=float)
        return eta[-DEPARTURE_DIMENSION:] @ self.active_basis / manifest.ACTIVE_SCALE

    def _features(self, local_coordinate: np.ndarray) -> np.ndarray:
        return np.concatenate(([1.0], self.active_coordinates(local_coordinate)))

    def decoded_delta(self, local_coordinate: np.ndarray) -> np.ndarray:
        eta = np.asarray(local_coordinate, dtype=float)
        absolute = self.center_coordinate + eta
        translated = self.direct.decoded_delta(absolute) - self.center_direct_delta
        correction = self.active_coordinates(eta) @ self.decoder_coefficients
        return self.center_delta + translated + correction

    def decoded_state(self, local_coordinate: np.ndarray) -> np.ndarray:
        delta = self.decoded_delta(local_coordinate)
        return self.model.base_state + (
            self.model.columns.ravel() * delta
        ).reshape(self.model.base_state.shape)

    def full_state_rate(self, local_coordinate: np.ndarray) -> np.ndarray:
        eta = np.asarray(local_coordinate, dtype=float)
        absolute = self.center_coordinate + eta
        return (
            self.direct.full_state_rate(absolute)
            + self._features(eta) @ self.full_rate_coefficients
        )

    def field(self, local_coordinate: np.ndarray) -> np.ndarray:
        full = self.full_state_rate(local_coordinate)
        result = self.restriction @ full
        result[:PHYSICAL_DIMENSION] += (
            self._features(local_coordinate) @ self.q_rate_coefficients
        )
        return result


def _fit_local_field(
    frozen: dict, inputs: dict, truth_arrays: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict]:
    design = frozen["design"]
    revealed_local = np.asarray(
        design["revealed_overlap_local_coordinates"], dtype=float
    )
    new_local = np.asarray(inputs["local_coordinates"], dtype=float)
    local = np.vstack((revealed_local, new_local))
    active = manifest._active_coordinates(
        local, design["active_departure_basis"]
    )
    exact_full = np.vstack(
        (
            design["revealed_overlap_exact_full_rates_per_second"],
            truth_arrays["total_rates_per_second"],
        )
    )
    exact_coordinate = np.vstack(
        (
            design["revealed_overlap_exact_coordinate_rates_per_second"],
            truth_arrays["exact_coordinate_rates_per_second"],
        )
    )
    old_full = np.vstack(
        (
            design["revealed_overlap_old_predicted_full_rates_per_second"],
            np.asarray(
                [
                    inputs["direct"].full_state_rate(coordinate)
                    for coordinate in inputs["absolute_coordinates"]
                ]
            ),
        )
    )
    weights = np.concatenate(
        (
            np.full(REVEALED_COUNT, 1.0 / REVEALED_COUNT),
            np.ones(1),
            np.full(4, 1.0 / 4.0),
        )
    )
    full_coefficients, full_fit = manifest._weighted_affine_fit(
        active,
        exact_full - old_full,
        weights,
        intercept=True,
    )
    features = manifest._affine_features(active)
    corrected_full = old_full + features @ full_coefficients
    restriction = np.asarray(
        design["authentic_center_fixed_restriction"], dtype=float
    )
    predicted_coordinate = corrected_full @ restriction.T
    q_coefficients, q_fit = manifest._weighted_affine_fit(
        active,
        exact_coordinate[:, :PHYSICAL_DIMENSION]
        - predicted_coordinate[:, :PHYSICAL_DIMENSION],
        weights,
        intercept=True,
    )
    predicted_coordinate[:, :PHYSICAL_DIMENSION] += features @ q_coefficients
    full_errors = _relative_rows(corrected_full, exact_full)
    coordinate_errors = _relative_rows(predicted_coordinate, exact_coordinate)
    q_errors = _relative_rows(
        predicted_coordinate[:, :PHYSICAL_DIMENSION],
        exact_coordinate[:, :PHYSICAL_DIMENSION],
    )
    z_slice = slice(PHYSICAL_DIMENSION, PHYSICAL_DIMENSION + MEMORY_DIMENSION)
    a_slice = slice(-DEPARTURE_DIMENSION, None)
    z_errors = _relative_rows(
        predicted_coordinate[:, z_slice], exact_coordinate[:, z_slice]
    )
    a_errors = _relative_rows(
        predicted_coordinate[:, a_slice], exact_coordinate[:, a_slice]
    )
    groups = {
        "revealed_overlap": np.arange(REVEALED_COUNT),
        "authentic_center": np.arange(REVEALED_COUNT, REVEALED_COUNT + 1),
        "new_training": np.arange(REVEALED_COUNT + 1, REVEALED_COUNT + 5),
        "new_exact": np.arange(REVEALED_COUNT, REVEALED_COUNT + 5),
    }

    def group_maximum(values: np.ndarray, group: str) -> float:
        return float(np.max(values[groups[group]]))

    metrics = {
        "fit_sample_count": int(local.shape[0]),
        "revealed_overlap_sample_count": REVEALED_COUNT,
        "authentic_center_sample_count": 1,
        "new_training_sample_count": 4,
        "group_total_weights": {
            "revealed_overlap": float(np.sum(weights[groups["revealed_overlap"]])),
            "authentic_center": float(np.sum(weights[groups["authentic_center"]])),
            "new_forward_training": float(np.sum(weights[groups["new_training"]])),
        },
        "full_fit": full_fit,
        "q_fit": q_fit,
        "maximum_regularized_normal_condition_number": float(
            max(
                full_fit["regularized_normal_condition_number"],
                q_fit["regularized_normal_condition_number"],
            )
        ),
        "full_rate_affine_coefficient_norm": float(np.linalg.norm(full_coefficients)),
        "q162_rate_affine_coefficient_norm": float(np.linalg.norm(q_coefficients)),
        "maximum_training_full_state_rate_relative_error": group_maximum(
            full_errors, "new_exact"
        ),
        "maximum_training_full_coordinate_rate_relative_error": group_maximum(
            coordinate_errors, "new_exact"
        ),
        "maximum_training_q162_rate_relative_error": group_maximum(
            q_errors, "new_exact"
        ),
        "maximum_training_z280_rate_relative_error": group_maximum(
            z_errors, "new_exact"
        ),
        "maximum_training_a28_rate_relative_error": group_maximum(
            a_errors, "new_exact"
        ),
        "maximum_revealed_full_state_rate_relative_error": group_maximum(
            full_errors, "revealed_overlap"
        ),
        "maximum_revealed_full_coordinate_rate_relative_error": group_maximum(
            coordinate_errors, "revealed_overlap"
        ),
        "maximum_revealed_q162_rate_relative_error": group_maximum(
            q_errors, "revealed_overlap"
        ),
        "maximum_revealed_z280_rate_relative_error": group_maximum(
            z_errors, "revealed_overlap"
        ),
        "maximum_revealed_a28_rate_relative_error": group_maximum(
            a_errors, "revealed_overlap"
        ),
    }
    closure = {
        "authentic_center_absolute_coordinate": design[
            "authentic_center_absolute_coordinate"
        ],
        "authentic_center_scaled_delta": design["authentic_center_scaled_delta"],
        "authentic_center_direct_decoded_scaled_delta": inputs[
            "direct"
        ].decoded_delta(design["authentic_center_absolute_coordinate"]),
        "authentic_center_fixed_restriction": restriction,
        "active_departure_basis": design["active_departure_basis"],
        "decoder_affine_coefficients": design["decoder_affine_coefficients"],
        "full_rate_affine_coefficients": full_coefficients,
        "q162_rate_affine_coefficients": q_coefficients,
        "fit_local_coordinates": local,
        "fit_active_coordinates": active,
        "fit_weights": weights,
        "fit_exact_full_rates_per_second": exact_full,
        "fit_old_full_rates_per_second": old_full,
        "fit_corrected_full_rates_per_second": corrected_full,
        "fit_exact_coordinate_rates_per_second": exact_coordinate,
        "fit_predicted_coordinate_rates_per_second": predicted_coordinate,
        "fit_full_state_relative_errors": full_errors,
        "fit_full_coordinate_relative_errors": coordinate_errors,
        "fit_q162_relative_errors": q_errors,
        "fit_z280_relative_errors": z_errors,
        "fit_a28_relative_errors": a_errors,
    }
    implementation = AuthenticCenterLocalField(
        closure, model=inputs["model"], direct=inputs["direct"]
    )
    repeated_full = np.asarray(
        [implementation.full_state_rate(value) for value in local]
    )
    repeated_coordinate = np.asarray([implementation.field(value) for value in local])
    metrics["maximum_full_rate_implementation_relative_defect"] = float(
        np.max(_relative_rows(repeated_full, corrected_full))
    )
    metrics["maximum_coordinate_field_implementation_relative_defect"] = float(
        np.max(_relative_rows(repeated_coordinate, predicted_coordinate))
    )
    metrics["online_state_dependent_coordinate_Jacobian_calls"] = 0
    return closure, metrics


def _truth_checks(metrics: dict, gates: dict) -> dict:
    return {
        "completed": metrics["completed_exact_rate_calls"]
        == gates["completed_exact_rate_calls_equal"],
        "failed": metrics["failed_exact_rate_calls"]
        == gates["failed_exact_rate_calls_equal"],
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
        "exact_rate_budget": metrics["total_available_exact_rate_calls"]
        == gates["completed_exact_rate_calls_equal"],
        "holdout_budget": metrics["holdout_rate_calls"] == 0,
        "generator_budget": metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": metrics["new_nonlinear_fixed_Q_roots"] == 0,
        "propagation_budget": metrics["propagated_states"] == 0,
    }


def _field_checks(metrics: dict, gates: dict) -> dict:
    return {
        "fit_condition": metrics["maximum_regularized_normal_condition_number"]
        <= gates["maximum_regularized_normal_condition_number"],
        "training_full_state": metrics[
            "maximum_training_full_state_rate_relative_error"
        ] <= gates["maximum_training_full_state_rate_relative_error"],
        "training_full_coordinate": metrics[
            "maximum_training_full_coordinate_rate_relative_error"
        ] <= gates["maximum_training_full_coordinate_rate_relative_error"],
        "training_q162": metrics["maximum_training_q162_rate_relative_error"]
        <= gates["maximum_training_q162_rate_relative_error"],
        "training_z280": metrics["maximum_training_z280_rate_relative_error"]
        <= gates["maximum_training_z280_rate_relative_error"],
        "training_a28": metrics["maximum_training_a28_rate_relative_error"]
        <= gates["maximum_training_a28_rate_relative_error"],
        "full_implementation": metrics[
            "maximum_full_rate_implementation_relative_defect"
        ] <= 1.0e-14,
        "coordinate_implementation": metrics[
            "maximum_coordinate_field_implementation_relative_defect"
        ] <= 1.0e-14,
        "no_online_coordinate_Jacobian": metrics[
            "online_state_dependent_coordinate_Jacobian_calls"
        ] == 0,
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
        raise RuntimeError("authentic-center exact-rate training already canonicalized")
    inputs = _load_inputs(frozen)
    truth_metrics, truth_arrays = _execute(inputs)
    gates = frozen["contract"]["training_execution_gates"]
    truth_checks = _truth_checks(truth_metrics, gates)
    truth_passed = all(truth_checks.values())
    closure: dict[str, np.ndarray] = {}
    field_metrics: dict = {}
    field_checks: dict[str, bool] = {"fit_available": False}
    if truth_passed:
        closure, field_metrics = _fit_local_field(frozen, inputs, truth_arrays)
        field_checks = _field_checks(field_metrics, gates)
    field_passed = bool(truth_passed and all(field_checks.values()))
    passed = bool(truth_passed and field_passed)
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = PASS_AUTHORIZED_NEXT if passed else FAIL_AUTHORIZED_NEXT
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "exact_truth_passed": truth_passed,
        "local_field_training_passed": field_passed,
        "completed_exact_rate_calls": truth_metrics["completed_exact_rate_calls"],
        "failed_exact_rate_calls": truth_metrics["failed_exact_rate_calls"],
        "holdout_rate_calls": 0,
        "coefficients_frozen_before_holdout_truth": passed,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "online_state_dependent_coordinate_Jacobian_calls": 0,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "training_metrics.json",
        {
            "truth_checks": truth_checks,
            "truth_passed": truth_passed,
            "field_checks": field_checks,
            "field_passed": field_passed,
            "truth": truth_metrics,
            "field": field_metrics,
        },
    )
    _write_npz(CANONICAL_DIRECTORY / "training_truth_arrays.npz", truth_arrays)
    if closure:
        _write_npz(CANONICAL_DIRECTORY / "authentic_center_local_field.npz", closure)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "input_execution_contract.json",
        frozen["contract"],
    )
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
            "frozen_design_sha256": _sha(FROZEN_DESIGN),
            "direct_field_sha256": _sha(DIRECT_FIELD),
            "blind_holdout_state_count": 4,
            "blind_holdout_rate_calls": 0,
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        manifest.parent.THIS_RUNNER,
        manifest.parent.THIS_TEST,
        direct_validation.THIS_RUNNER,
        direct_validation.THIS_TEST,
        rate_engine.manifest.prior_screen.THIS_RUNNER,
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
            "resumed_from_exact_rate_count": truth_metrics[
                "resumed_exact_rate_calls"
            ],
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
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
                "# Authentic-center exact-rate training WP10c9d6c7c3b5c4f25cu",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{summary['completed_exact_rate_calls']}` of `{EXACT_RATE_COUNT}` exact center/training rate calls with `{summary['failed_exact_rate_calls']}` failures. Exact truth admissibility passed: `{truth_passed}`; local-field training passed: `{field_passed}`.",
                "",
                (
                    "Maximum center/training full-state/full-coordinate/q162/z280/a28 errors are "
                    f"`{field_metrics.get('maximum_training_full_state_rate_relative_error', math.inf):.6e}`, "
                    f"`{field_metrics.get('maximum_training_full_coordinate_rate_relative_error', math.inf):.6e}`, "
                    f"`{field_metrics.get('maximum_training_q162_rate_relative_error', math.inf):.6e}`, "
                    f"`{field_metrics.get('maximum_training_z280_rate_relative_error', math.inf):.6e}`, and "
                    f"`{field_metrics.get('maximum_training_a28_rate_relative_error', math.inf):.6e}`."
                ),
                "",
                "The four frozen holdout states were not rate-evaluated. Coefficients are immutable before their separate blind validation. The online local field uses fixed affine maps and no state-dependent coordinate Jacobian.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No trajectory, physical microburst, predictive cycle, or reduced slow evolution is authorized.",
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
