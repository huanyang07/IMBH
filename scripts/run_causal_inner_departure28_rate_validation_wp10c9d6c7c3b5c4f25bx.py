#!/usr/bin/env python3
"""Independently validate the frozen departure-28 dual-polynomial closure."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_departure28_independent_validation_manifest_wp10c9d6c7c3b5c4f25bv as manifest  # noqa: E402
import run_causal_inner_departure28_validation_geometry_wp10c9d6c7c3b5c4f25bw as geometry  # noqa: E402
import run_causal_inner_departure28_dual_polynomial_diagnosis_wp10c9d6c7c3b5c4f25bu as diagnosis  # noqa: E402
import run_causal_inner_active8_projective_kernel_rate_validation_wp10c9d6c7c3b5c4f25bt as prior_rate  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bx"
GEOMETRY_COMMIT = "2e93a85cefa19cc1c7ec5ae60035e83451ea3877"
GEOMETRY_PARENT = "72134c782cd2efa262d8cb75c7d45d8a24c5e5d9"
GEOMETRY_TREE = "212551a3abb4f81fd09825f8927e3cf39d99767b"

PASS_CLASSIFICATION = (
    "departure28_dual_polynomial_rate_and_rank4_decoder_"
    "independently_validated"
)
FAIL_CLASSIFICATION = "departure28_dual_polynomial_independent_validation_failed"
AUTHORIZED_NEXT = (
    "definitions_only_departure28_short_reduced_vector_field_validation_manifest"
)

ARTIFACT = (
    "causal_inner_departure28_rate_validation_wp10c9d6c7c3b5c4f25bx"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_departure28_rate_validation_"
    "wp10c9d6c7c3b5c4f25bx.py"
)
THIS_TEST = (
    "tests/test_causal_inner_departure28_rate_validation_"
    "wp10c9d6c7c3b5c4f25bx.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_DEPARTURE28_RATE_"
    "VALIDATION_WP10C9D6C7C3B5C4F25BX_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

ENGINE_RUNNER = prior_rate.THIS_RUNNER
ENGINE_TEST = prior_rate.THIS_TEST
DATABASE_PATH = geometry.CANONICAL_DIRECTORY / "departure28_geometry_database.npz"
ARCHITECTURE_PATH = (
    diagnosis.CANONICAL_DIRECTORY / "departure28_architecture.npz"
)
PRIOR_DECODER_PATH = prior_rate.CANONICAL_DIRECTORY / "frozen_coefficients.npz"
ONLINE_GEOMETRY_PATH = prior_rate.ONLINE_GEOMETRY_PATH
GENERATOR_PATH = prior_rate.GENERATOR_PATH
CURVATURE_DESIGN_PATH = prior_rate.CURVATURE_DESIGN_PATH
FIT_ARRAY_PATH = SCRATCH_DIRECTORY / "frozen_coefficients.npz"
FIT_LOCK_PATH = SCRATCH_DIRECTORY / "coefficient_lock.json"

TRAINING_DIRECTION_COUNT = manifest.REVEALED_HIGH_DIRECTION_COUNT
DEPARTURE_DIMENSION = diagnosis.DEPARTURE_DIMENSION
EVEN_KERNEL_POWER = diagnosis.EVEN_KERNEL_POWER
ODD_KERNEL_POWER = diagnosis.ODD_KERNEL_POWER
EVEN_TARGET_WEIGHT_EXPONENT = diagnosis.EVEN_TARGET_WEIGHT_EXPONENT
ODD_TARGET_WEIGHT_EXPONENT = diagnosis.ODD_TARGET_WEIGHT_EXPONENT
EVEN_TIKHONOV_REGULARIZATION = diagnosis.EVEN_TIKHONOV_REGULARIZATION
ODD_TIKHONOV_REGULARIZATION = diagnosis.ODD_TIKHONOV_REGULARIZATION
RATE_COEFFICIENT_COUNT = diagnosis.RATE_COEFFICIENT_COUNT
CURVATURE_COEFFICIENT_COUNT = diagnosis.CURVATURE_COEFFICIENT_COUNT
TOTAL_NONLINEAR_COEFFICIENT_COUNT = diagnosis.TOTAL_NONLINEAR_COEFFICIENT_COUNT


_plain = prior_rate._plain
_read = prior_rate._read
_write_json = prior_rate._write_json
_write_npz = prior_rate._write_npz
_load_npz = prior_rate._load_npz
_sha = prior_rate._sha
_checksums = prior_rate._checksums
_relative_error = prior_rate._relative_error


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _evaluation_order() -> tuple[int, ...]:
    return tuple(range(manifest.PLANNED_CANDIDATES))


def _validate_geometry(*, require_clean: bool) -> dict:
    if _git("rev-parse", GEOMETRY_COMMIT) != GEOMETRY_COMMIT:
        raise RuntimeError("departure-28 geometry result commit changed")
    if _git("rev-parse", f"{GEOMETRY_COMMIT}^") != GEOMETRY_PARENT:
        raise RuntimeError("departure-28 geometry result lineage changed")
    if _git("rev-parse", f"{GEOMETRY_COMMIT}^{{tree}}") != GEOMETRY_TREE:
        raise RuntimeError("departure-28 geometry result tree changed")
    geometry_hashes = _checksums(geometry.CANONICAL_DIRECTORY)
    summary = _read(geometry.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(geometry.CANONICAL_DIRECTORY / "metrics.json")
    provenance = _read(geometry.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["classification"] != geometry.PASS_CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["completed_candidate_count"] != manifest.PLANNED_CANDIDATES
        or summary["failed_candidate_count"] != 0
        or summary["nonbase_continuous_rate_evaluations"] != 0
        or not all(metrics["checks"].values())
    ):
        raise RuntimeError("departure-28 rate authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"geometry source changed: {relative}")

    manifest_hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    if (
        not contract["leakage_control"][
            "all_rate_coefficients_frozen_and_hashed_before_new_rate_truth"
        ]
        or contract["mathematical_architecture"][
            "stored_total_nonlinear_coefficients"
        ]
        != TOTAL_NONLINEAR_COEFFICIENT_COUNT
        or contract["mathematical_architecture"]["departure_rate_input_dimension"]
        != DEPARTURE_DIMENSION
    ):
        raise RuntimeError("departure-28 leakage contract changed")

    diagnosis_hashes = _checksums(diagnosis.CANONICAL_DIRECTORY)
    diagnosis_summary = _read(diagnosis.CANONICAL_DIRECTORY / "summary.json")
    diagnosis_metrics = _read(diagnosis.CANONICAL_DIRECTORY / "metrics.json")
    if (
        not diagnosis_summary["passed"]
        or diagnosis_summary["classification"] != diagnosis.PASS_CLASSIFICATION
        or not diagnosis_summary["diagnostic_only"]
        or diagnosis_summary["new_truth_evaluations"] != 0
        or diagnosis_summary["revealed_direction_count"] != TRAINING_DIRECTION_COUNT
        or not all(diagnosis_metrics["checks"].values())
    ):
        raise RuntimeError("departure-28 selected architecture changed")

    prior_hashes = _checksums(prior_rate.CANONICAL_DIRECTORY)
    prior_summary = _read(prior_rate.CANONICAL_DIRECTORY / "summary.json")
    if (
        prior_summary["classification"] != prior_rate.FAIL_CLASSIFICATION
        or prior_summary["passed"]
        or not prior_summary["truth_database_passed"]
        or prior_summary["completed_nonbase_rate_evaluations"] != 48
        or prior_summary["failed_rate_evaluations"] != 0
    ):
        raise RuntimeError("certified rank-4 decoder source changed")

    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("departure-28 rate validation requires a clean tree")
    for name, expected in geometry.chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {
        "summary": summary,
        "metrics": metrics,
        "contract": contract,
        "geometry_hashes": geometry_hashes,
        "manifest_hashes": manifest_hashes,
        "diagnosis_hashes": diagnosis_hashes,
        "prior_hashes": prior_hashes,
    }


def _load_inputs() -> dict:
    database = _load_npz(DATABASE_PATH)
    online_geometry = _load_npz(ONLINE_GEOMETRY_PATH)
    generator = _load_npz(GENERATOR_PATH)
    architecture_database = _load_npz(ARCHITECTURE_PATH)
    prior_decoder = _load_npz(PRIOR_DECODER_PATH)
    curvature_design = _load_npz(CURVATURE_DESIGN_PATH)
    metrics = _read(geometry.CANONICAL_DIRECTORY / "metrics.json")
    states = np.asarray(database["candidate_primitive_states"], dtype=float)
    deltas = np.asarray(database["candidate_scaled_deltas"], dtype=float)
    coordinates = np.asarray(
        database["candidate_departure_coordinates"], dtype=float
    )
    candidates = metrics["candidates"]
    frozen_decoder = np.asarray(
        architecture_database["frozen_rank4_curvature_decoder_coefficients"],
        dtype=float,
    )
    if (
        states.shape != (manifest.PLANNED_CANDIDATES, 112, 5)
        or deltas.shape != (manifest.PLANNED_CANDIDATES, 560)
        or coordinates.shape != (manifest.PLANNED_CANDIDATES, DEPARTURE_DIMENSION)
        or len(candidates) != manifest.PLANNED_CANDIDATES
        or generator["complete_fixed_Q_generator"].shape != (560, 560)
        or online_geometry["online_coordinate_restriction"].shape != (470, 560)
        or online_geometry["online_coordinate_lifting"].shape != (560, 470)
        or curvature_design["rank4_curvature_basis"].shape != (560, 4)
        or architecture_database["directions"].shape
        != (TRAINING_DIRECTION_COUNT, DEPARTURE_DIMENSION)
        or architecture_database["quadratic_targets"].shape
        != (TRAINING_DIRECTION_COUNT, DEPARTURE_DIMENSION)
        or architecture_database["cubic_targets"].shape
        != (TRAINING_DIRECTION_COUNT, DEPARTURE_DIMENSION)
        or frozen_decoder.shape != (120, 4)
    ):
        raise RuntimeError("departure-28 rate input dimensions changed")
    if not np.array_equal(
        frozen_decoder,
        np.asarray(prior_decoder["curvature_cubic_coefficients"], dtype=float),
    ):
        raise RuntimeError("certified rank-4 curvature decoder changed")
    if [item["candidate_index"] for item in candidates] != list(
        range(manifest.PLANNED_CANDIDATES)
    ):
        raise RuntimeError("departure-28 geometry candidate ordering changed")
    if [item["split"] for item in candidates[:32]] != ["holdout"] * 32:
        raise RuntimeError("independent departure-28 holdout split changed")
    if [item["split"] for item in candidates[32:]] != ["tuning_low"] * 16:
        raise RuntimeError("independent departure-28 radial split changed")
    return {
        "database": database,
        "online_geometry": online_geometry,
        "generator": np.asarray(generator["complete_fixed_Q_generator"], dtype=float),
        "base_rate": np.asarray(generator["fixed_Q_rate"], dtype=float),
        "architecture_database": architecture_database,
        "prior_decoder": prior_decoder,
        "curvature_basis": np.asarray(
            curvature_design["rank4_curvature_basis"], dtype=float
        ),
        "states": states,
        "deltas": deltas,
        "coordinates": coordinates,
        "candidates": candidates,
    }


def _training_targets(inputs: dict) -> dict[str, np.ndarray]:
    source = inputs["architecture_database"]
    targets = {
        "directions": np.asarray(source["directions"], dtype=float),
        "radii": np.asarray(source["radii"], dtype=float),
        "quadratic_targets": np.asarray(source["quadratic_targets"], dtype=float),
        "cubic_targets": np.asarray(source["cubic_targets"], dtype=float),
        "curvature_decoder": np.asarray(
            source["frozen_rank4_curvature_decoder_coefficients"], dtype=float
        ),
    }
    expected = {
        "directions": (TRAINING_DIRECTION_COUNT, DEPARTURE_DIMENSION),
        "radii": (TRAINING_DIRECTION_COUNT,),
        "quadratic_targets": (TRAINING_DIRECTION_COUNT, DEPARTURE_DIMENSION),
        "cubic_targets": (TRAINING_DIRECTION_COUNT, DEPARTURE_DIMENSION),
        "curvature_decoder": (120, 4),
    }
    if any(targets[name].shape != shape for name, shape in expected.items()):
        raise RuntimeError("revealed departure-28 training dimensions changed")
    if not all(np.all(np.isfinite(values)) for values in targets.values()):
        raise RuntimeError("revealed departure-28 training data are nonfinite")
    return targets


def _fit_coefficients(targets: dict[str, np.ndarray]) -> tuple[dict, dict]:
    directions = targets["directions"]
    even, even_fit = diagnosis._fit(
        directions,
        targets["quadratic_targets"],
        power=EVEN_KERNEL_POWER,
        weight_exponent=EVEN_TARGET_WEIGHT_EXPONENT,
        regularization=EVEN_TIKHONOV_REGULARIZATION,
    )
    odd, odd_fit = diagnosis._fit(
        directions,
        targets["cubic_targets"],
        power=ODD_KERNEL_POWER,
        weight_exponent=ODD_TARGET_WEIGHT_EXPONENT,
        regularization=ODD_TIKHONOV_REGULARIZATION,
    )
    even_norms = np.linalg.norm(targets["quadratic_targets"], axis=1)
    odd_norms = np.linalg.norm(targets["cubic_targets"], axis=1)
    even_scale = float(np.median(even_norms))
    odd_scale = float(np.median(odd_norms))
    even_weights = (
        even_scale / np.maximum(even_norms, np.finfo(float).tiny)
    ) ** EVEN_TARGET_WEIGHT_EXPONENT
    odd_weights = (
        odd_scale / np.maximum(odd_norms, np.finfo(float).tiny)
    ) ** ODD_TARGET_WEIGHT_EXPONENT
    metrics = {
        "training_direction_count": TRAINING_DIRECTION_COUNT,
        "even_system_rank": even_fit["rank"],
        "even_system_condition_number": even_fit["condition_number"],
        "odd_system_rank": odd_fit["rank"],
        "odd_system_condition_number": odd_fit["condition_number"],
        "even_target_norm_median": even_scale,
        "odd_target_norm_median": odd_scale,
        "even_target_weight_minimum": float(np.min(even_weights)),
        "even_target_weight_maximum": float(np.max(even_weights)),
        "odd_target_weight_minimum": float(np.min(odd_weights)),
        "odd_target_weight_maximum": float(np.max(odd_weights)),
        "even_target_weight_exponent": EVEN_TARGET_WEIGHT_EXPONENT,
        "odd_target_weight_exponent": ODD_TARGET_WEIGHT_EXPONENT,
        "even_Tikhonov_regularization": EVEN_TIKHONOV_REGULARIZATION,
        "odd_Tikhonov_regularization": ODD_TIKHONOV_REGULARIZATION,
        "stored_rate_coefficient_count": RATE_COEFFICIENT_COUNT,
        "stored_curvature_coefficient_count": CURVATURE_COEFFICIENT_COUNT,
        "stored_total_nonlinear_coefficient_count": TOTAL_NONLINEAR_COEFFICIENT_COUNT,
    }
    arrays = {
        "training_directions_departure28": directions,
        "training_radii": targets["radii"],
        "rate_quadratic_targets": targets["quadratic_targets"],
        "rate_cubic_targets": targets["cubic_targets"],
        "even_target_norms": even_norms,
        "odd_target_norms": odd_norms,
        "even_target_weights": even_weights,
        "odd_target_weights": odd_weights,
        "even_dual_coefficients": even,
        "odd_dual_coefficients": odd,
        "frozen_rank4_curvature_decoder_coefficients": targets[
            "curvature_decoder"
        ],
    }
    selected = _load_npz(ARCHITECTURE_PATH)
    stored_even = np.asarray(
        selected["all_revealed_even_coefficients"], dtype=float
    )
    stored_odd = np.asarray(
        selected["all_revealed_odd_coefficients"], dtype=float
    )
    if not np.array_equal(even, stored_even) and not np.allclose(
        even, stored_even, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError("revealed departure-28 even refit changed")
    if not np.array_equal(odd, stored_odd) and not np.allclose(
        odd, stored_odd, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError("revealed departure-28 odd refit changed")
    if not all(np.all(np.isfinite(values)) for values in arrays.values()):
        raise RuntimeError("departure-28 frozen coefficients are nonfinite")
    return metrics, arrays


def _freeze_or_validate_coefficients(inputs: dict) -> tuple[dict, dict]:
    if FIT_LOCK_PATH.exists() != FIT_ARRAY_PATH.exists():
        raise RuntimeError("departure-28 coefficient lock is incomplete")
    targets = _training_targets(inputs)
    if FIT_LOCK_PATH.exists():
        lock = _read(FIT_LOCK_PATH)
        if (
            lock["coefficient_sha256"] != _sha(FIT_ARRAY_PATH)
            or lock["training_direction_count"] != TRAINING_DIRECTION_COUNT
            or lock["validation_rate_evaluations_at_freeze"] != 0
            or lock["runner_sha256"] != _sha(ROOT / THIS_RUNNER)
            or lock["geometry_database_sha256"] != _sha(DATABASE_PATH)
            or lock["architecture_database_sha256"] != _sha(ARCHITECTURE_PATH)
            or lock["prior_decoder_sha256"] != _sha(PRIOR_DECODER_PATH)
        ):
            raise RuntimeError("frozen departure-28 coefficient lock changed")
        arrays = _load_npz(FIT_ARRAY_PATH)
        for source_name, frozen_name in (
            ("directions", "training_directions_departure28"),
            ("radii", "training_radii"),
            ("quadratic_targets", "rate_quadratic_targets"),
            ("cubic_targets", "rate_cubic_targets"),
            ("curvature_decoder", "frozen_rank4_curvature_decoder_coefficients"),
        ):
            if not np.array_equal(targets[source_name], arrays[frozen_name]):
                raise RuntimeError(
                    "revealed departure-28 training truth changed after coefficient freeze"
                )
        return lock["fit_metrics"], arrays
    progress_paths = (
        SCRATCH_DIRECTORY / "progress.json",
        SCRATCH_DIRECTORY / "progress.npz",
    )
    if any(path.exists() for path in progress_paths):
        raise RuntimeError("new validation truth exists before coefficient freeze")
    fit_metrics, arrays = _fit_coefficients(targets)
    _write_npz(FIT_ARRAY_PATH, arrays)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "training_direction_count": TRAINING_DIRECTION_COUNT,
        "validation_rate_evaluations_at_freeze": 0,
        "coefficient_sha256": _sha(FIT_ARRAY_PATH),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "geometry_database_sha256": _sha(DATABASE_PATH),
        "architecture_database_sha256": _sha(ARCHITECTURE_PATH),
        "prior_decoder_sha256": _sha(PRIOR_DECODER_PATH),
        "fit_metrics": fit_metrics,
    }
    _write_json(FIT_LOCK_PATH, lock)
    print(
        json.dumps(
            {
                "coefficient_lock_frozen": True,
                "training_directions": TRAINING_DIRECTION_COUNT,
                "validation_rate_evaluations_at_freeze": 0,
                "coefficient_sha256": lock["coefficient_sha256"],
            }
        ),
        flush=True,
    )
    return fit_metrics, arrays


def _predict_rate(coordinate: np.ndarray, coefficients: dict) -> np.ndarray:
    coordinate = np.asarray(coordinate, dtype=float)
    return diagnosis._predict(
        coordinate.reshape(1, DEPARTURE_DIMENSION),
        coefficients["training_directions_departure28"],
        coefficients["even_dual_coefficients"],
        coefficients["odd_dual_coefficients"],
    )[0]


def _predict_curvature(active: np.ndarray, coefficients: dict) -> np.ndarray:
    active = np.asarray(active, dtype=float)
    radius = float(np.linalg.norm(active))
    if radius <= np.finfo(float).tiny:
        return np.zeros(4)
    direction = active / radius
    features = prior_rate.architecture._cubic_features(direction.reshape(1, 8))[0]
    return (
        radius**3
        * features
        @ coefficients["frozen_rank4_curvature_decoder_coefficients"]
    )


def _validation_metrics(
    inputs: dict, truth: dict[str, np.ndarray], coefficients: dict
) -> tuple[dict, dict[str, np.ndarray]]:
    energy = np.asarray(inputs["database"]["energy_directions"], dtype=float)
    restriction = inputs["online_geometry"]["online_coordinate_restriction"]
    lifting = inputs["online_geometry"]["online_coordinate_lifting"]
    components = geometry.base.high_chart._prepare_components()
    nonlinear_truth = (
        truth["departure_rate_increments_per_second"]
        - truth["departure_linear_references_per_second"]
    )
    records = []
    predicted_nonlinear = np.full(
        (manifest.PLANNED_CANDIDATES, DEPARTURE_DIMENSION), np.nan
    )
    predicted_curvature = np.full((manifest.PLANNED_CANDIDATES, 4), np.nan)
    predicted_deltas = np.full((manifest.PLANNED_CANDIDATES, 560), np.nan)
    for index, candidate in enumerate(inputs["candidates"]):
        if candidate["split"] != "holdout":
            continue
        rate = _predict_rate(inputs["coordinates"][index], coefficients)
        active = energy.T @ inputs["coordinates"][index]
        curvature = _predict_curvature(active, coefficients)
        departure = truth["departure_linear_references_per_second"][index] + rate
        online = lifting @ (restriction @ inputs["deltas"][index])
        predicted_delta = online + inputs["curvature_basis"] @ curvature
        true_curvature = inputs["deltas"][index] @ inputs["curvature_basis"]
        state = components["state"] + (
            components["columns"].ravel() * predicted_delta
        ).reshape(components["state"].shape)
        coordinate, coordinate_factors = geometry.chart_tools._coordinate_value_with_factors(
            state, components
        )
        physical = geometry.chart_tools._state_audit(components["context"], state)
        records.append(
            {
                "candidate_index": index,
                "pair_index": candidate["pair_index"],
                "split": candidate["split"],
                "amplitude_label": candidate["amplitude_label"],
                "sign": candidate["sign"],
                "nonlinear_departure_rate_relative_error": _relative_error(
                    rate, nonlinear_truth[index]
                ),
                "full_departure_rate_relative_error": _relative_error(
                    departure, truth["departure_rate_increments_per_second"][index]
                ),
                "curvature_prediction_error_over_full_state_delta": float(
                    np.linalg.norm(curvature - true_curvature)
                    / max(
                        float(np.linalg.norm(inputs["deltas"][index])),
                        np.finfo(float).tiny,
                    )
                ),
                "full_scaled_state_decoder_relative_error": _relative_error(
                    predicted_delta, inputs["deltas"][index]
                ),
                "reconstructed_C_phys_residual_infinity": float(
                    np.max(np.abs(coordinate - components["coordinate_target"]))
                ),
                "minimum_reconstructed_state_reconstruction_factor": min(
                    float(np.min(coordinate_factors)),
                    physical["minimum_reconstruction_factor"],
                ),
                "maximum_reconstructed_H_over_R": physical["maximum_h_over_r"],
                "minimum_reconstructed_scattering_optical_depth": physical[
                    "minimum_scattering_optical_depth"
                ],
            }
        )
        predicted_nonlinear[index] = rate
        predicted_curvature[index] = curvature
        predicted_deltas[index] = predicted_delta
    if len(records) != manifest.NEW_HIGH_CANDIDATE_COUNT:
        raise RuntimeError("independent departure-28 holdout count changed")

    def aggregate(field: str, operation) -> float:
        return float(operation([item[field] for item in records]))

    metrics = {
        "holdout_candidate_count": len(records),
        "holdout_median_nonlinear_departure_rate_relative_error": aggregate(
            "nonlinear_departure_rate_relative_error", np.median
        ),
        "holdout_maximum_nonlinear_departure_rate_relative_error": aggregate(
            "nonlinear_departure_rate_relative_error", np.max
        ),
        "holdout_median_full_departure_rate_relative_error": aggregate(
            "full_departure_rate_relative_error", np.median
        ),
        "holdout_maximum_full_departure_rate_relative_error": aggregate(
            "full_departure_rate_relative_error", np.max
        ),
        "maximum_curvature_prediction_error_over_full_state_delta": aggregate(
            "curvature_prediction_error_over_full_state_delta", np.max
        ),
        "maximum_full_scaled_state_decoder_relative_error": aggregate(
            "full_scaled_state_decoder_relative_error", np.max
        ),
        "maximum_reconstructed_C_phys_residual_infinity": aggregate(
            "reconstructed_C_phys_residual_infinity", np.max
        ),
        "minimum_reconstructed_state_reconstruction_factor": aggregate(
            "minimum_reconstructed_state_reconstruction_factor", np.min
        ),
        "maximum_reconstructed_H_over_R": aggregate(
            "maximum_reconstructed_H_over_R", np.max
        ),
        "minimum_reconstructed_scattering_optical_depth": aggregate(
            "minimum_reconstructed_scattering_optical_depth", np.min
        ),
        "validation": records,
    }
    return metrics, {
        "predicted_nonlinear_departure_rates_per_second": predicted_nonlinear,
        "predicted_curvature_coordinates": predicted_curvature,
        "predicted_scaled_deltas": predicted_deltas,
    }


def _radial_metrics(inputs: dict, truth: dict[str, np.ndarray]) -> dict:
    energy = np.asarray(inputs["database"]["energy_directions"], dtype=float)

    def rate_targets(start: int, stop: int) -> dict[str, np.ndarray]:
        return diagnosis._pair_targets(
            inputs["coordinates"][start:stop],
            truth["departure_rate_increments_per_second"][start:stop],
            truth["departure_linear_references_per_second"][start:stop],
        )

    def curvature_targets(start: int, stop: int) -> np.ndarray:
        return prior_rate._pair_targets(
            deltas=inputs["deltas"][start:stop],
            coordinates=inputs["coordinates"][start:stop],
            departure_increments=truth["departure_rate_increments_per_second"][start:stop],
            departure_linear=truth["departure_linear_references_per_second"][start:stop],
            energy_directions=energy,
            curvature_basis=inputs["curvature_basis"],
        )["curvature_cubic_targets"]

    high_start = 0
    high_stop = 2 * manifest.NEW_RADIAL_DIRECTION_COUNT
    low_start = manifest.NEW_HIGH_CANDIDATE_COUNT
    low_stop = manifest.PLANNED_CANDIDATES
    high_rate = rate_targets(high_start, high_stop)
    low_rate = rate_targets(low_start, low_stop)
    high_curvature = curvature_targets(high_start, high_stop)
    low_curvature = curvature_targets(low_start, low_stop)

    def differences(high: np.ndarray, low: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                _relative_error(high[index], low[index])
                for index in range(manifest.NEW_RADIAL_DIRECTION_COUNT)
            ],
            dtype=float,
        )

    quadratic = differences(
        high_rate["quadratic_targets"], low_rate["quadratic_targets"]
    )
    cubic = differences(high_rate["cubic_targets"], low_rate["cubic_targets"])
    curvature = differences(high_curvature, low_curvature)
    return {
        "maximum_quadratic_target_high_low_relative_difference": float(
            np.max(quadratic)
        ),
        "maximum_cubic_rate_target_high_low_relative_difference": float(
            np.max(cubic)
        ),
        "maximum_curvature_cubic_target_high_low_relative_difference": float(
            np.max(curvature)
        ),
        "quadratic_directionwise_high_low_relative_difference": quadratic,
        "cubic_rate_directionwise_high_low_relative_difference": cubic,
        "curvature_cubic_directionwise_high_low_relative_difference": curvature,
    }


def _model_gate_checks(model: dict, radial: dict, fit: dict, contract: dict) -> dict:
    model_gates = contract["binding_independent_model_gates"]
    radial_gates = contract["binding_radial_consistency_gates"]
    fit_gates = contract["binding_fit_gates"]
    checks = {
        name: model[name] <= threshold
        for name, threshold in model_gates.items()
        if name
        not in (
            "minimum_reconstructed_state_reconstruction_factor",
            "minimum_reconstructed_scattering_optical_depth",
        )
    }
    checks["minimum_reconstructed_state_reconstruction_factor"] = model[
        "minimum_reconstructed_state_reconstruction_factor"
    ] >= model_gates["minimum_reconstructed_state_reconstruction_factor"]
    checks["minimum_reconstructed_scattering_optical_depth"] = model[
        "minimum_reconstructed_scattering_optical_depth"
    ] >= model_gates["minimum_reconstructed_scattering_optical_depth"]
    checks.update(
        {name: radial[name] <= threshold for name, threshold in radial_gates.items()}
    )
    checks.update(
        {
            "even_system_full_rank": fit["even_system_rank"]
            == fit_gates["even_system_rank_equal"],
            "odd_system_full_rank": fit["odd_system_rank"]
            == fit_gates["odd_system_rank_equal"],
            "even_system_condition": math.isfinite(
                fit["even_system_condition_number"]
            )
            and fit["even_system_condition_number"]
            <= fit_gates["even_system_condition_number"],
            "odd_system_condition": math.isfinite(
                fit["odd_system_condition_number"]
            )
            and fit["odd_system_condition_number"]
            <= fit_gates["odd_system_condition_number"],
            "coefficient_count": fit["stored_total_nonlinear_coefficient_count"]
            == fit_gates["stored_total_nonlinear_coefficient_count_equal"],
        }
    )
    return checks


def _fresh_engine():
    path = ROOT / ENGINE_RUNNER
    spec = importlib.util.spec_from_file_location(
        "_departure28_rate_validation_engine", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load certified exact-rate engine")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    replacements = {
        "manifest": manifest,
        "geometry": geometry,
        "WORK_PACKAGE": WORK_PACKAGE,
        "GEOMETRY_COMMIT": GEOMETRY_COMMIT,
        "GEOMETRY_PARENT": GEOMETRY_PARENT,
        "GEOMETRY_TREE": GEOMETRY_TREE,
        "PASS_CLASSIFICATION": PASS_CLASSIFICATION,
        "FAIL_CLASSIFICATION": FAIL_CLASSIFICATION,
        "AUTHORIZED_NEXT": AUTHORIZED_NEXT,
        "ARTIFACT": ARTIFACT,
        "CANONICAL_DIRECTORY": CANONICAL_DIRECTORY,
        "SCRATCH_DIRECTORY": SCRATCH_DIRECTORY,
        "THIS_RUNNER": THIS_RUNNER,
        "THIS_TEST": THIS_TEST,
        "REPORT_RELATIVE": REPORT_RELATIVE,
        "REPORT_PATH": REPORT_PATH,
        "CANONICAL_MANIFEST": CANONICAL_MANIFEST,
        "CANONICAL_SUMMARY": CANONICAL_SUMMARY,
        "DATABASE_PATH": DATABASE_PATH,
        "ONLINE_GEOMETRY_PATH": ONLINE_GEOMETRY_PATH,
        "GENERATOR_PATH": GENERATOR_PATH,
        "CURVATURE_DESIGN_PATH": CURVATURE_DESIGN_PATH,
        "FIT_ARRAY_PATH": FIT_ARRAY_PATH,
        "FIT_LOCK_PATH": FIT_LOCK_PATH,
        "TRAINING_DIRECTION_COUNT": TRAINING_DIRECTION_COUNT,
        "TOTAL_NONLINEAR_COEFFICIENT_COUNT": TOTAL_NONLINEAR_COEFFICIENT_COUNT,
        "_evaluation_order": _evaluation_order,
        "_validate_geometry": _validate_geometry,
        "_load_inputs": _load_inputs,
        "_training_targets": _training_targets,
        "_fit_coefficients": _fit_coefficients,
        "_freeze_or_validate_coefficients": _freeze_or_validate_coefficients,
        "_validation_metrics": _validation_metrics,
        "_radial_metrics": _radial_metrics,
        "_model_gate_checks": _model_gate_checks,
    }
    for name, value in replacements.items():
        setattr(engine, name, value)
    return engine


def _run() -> dict:
    frozen = _validate_geometry(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("departure-28 rate validation already canonicalized")
    inputs = _load_inputs()
    engine = _fresh_engine()
    truth_metrics, truth_arrays, fit_metrics, fit_arrays = engine._execute_truth(inputs)
    truth_checks = engine._truth_gate_checks(
        truth_metrics, frozen["contract"]["binding_truth_rate_gates"]
    )
    truth_passed = all(truth_checks.values())
    model_metrics = {}
    radial_metrics = {}
    model_arrays = {}
    model_checks = {"truth_database_complete": False}
    if truth_passed:
        radial_metrics = _radial_metrics(inputs, truth_arrays)
        model_metrics, model_arrays = _validation_metrics(
            inputs, truth_arrays, fit_arrays
        )
        model_checks = _model_gate_checks(
            model_metrics, radial_metrics, fit_metrics, frozen["contract"]
        )
    passed = truth_passed and all(model_checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = AUTHORIZED_NEXT if passed else None

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "metrics.json",
        {
            "truth_checks": truth_checks,
            "model_checks": model_checks,
            "truth": truth_metrics,
            "fit": fit_metrics,
            "radial": radial_metrics,
            "model": model_metrics,
        },
    )
    _write_npz(
        CANONICAL_DIRECTORY / "departure28_closure.npz",
        {
            "candidate_primitive_states": inputs["states"],
            "candidate_scaled_deltas": inputs["deltas"],
            "candidate_departure_coordinates": inputs["coordinates"],
            "base_fixed_Q_rate_per_second": inputs["base_rate"],
            **truth_arrays,
            **fit_arrays,
            **model_arrays,
        },
    )
    if FIT_LOCK_PATH.exists():
        shutil.copy2(FIT_LOCK_PATH, CANONICAL_DIRECTORY / "coefficient_lock.json")
        shutil.copy2(FIT_ARRAY_PATH, CANONICAL_DIRECTORY / "frozen_coefficients.npz")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "truth_database_passed": truth_passed,
        "independent_model_validation_passed": passed,
        "completed_nonbase_rate_evaluations": truth_metrics[
            "completed_nonbase_rate_evaluations"
        ],
        "failed_rate_evaluations": truth_metrics["failed_rate_evaluations"],
        "coefficient_lock_preceded_validation": bool(
            FIT_LOCK_PATH.exists()
            and _read(FIT_LOCK_PATH)["validation_rate_evaluations_at_freeze"] == 0
        ),
        "stored_nonlinear_coefficients": TOTAL_NONLINEAR_COEFFICIENT_COUNT,
        "dynamic_state_dimension": 470,
        "dynamic_curvature_augmentation": False,
        "online_truth_calls_per_macrostep": 0,
        "online_Newton_retractions_per_macrostep": 0,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "geometry_commit": GEOMETRY_COMMIT,
            "geometry_parent": GEOMETRY_PARENT,
            "geometry_tree": GEOMETRY_TREE,
            "geometry_hashes": frozen["geometry_hashes"],
            "manifest_hashes": frozen["manifest_hashes"],
            "diagnosis_hashes": frozen["diagnosis_hashes"],
            "prior_projective_rate_hashes": frozen["prior_hashes"],
            "architecture_database_sha256": _sha(ARCHITECTURE_PATH),
            "prior_decoder_sha256": _sha(PRIOR_DECODER_PATH),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        geometry.THIS_RUNNER,
        geometry.THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        diagnosis.THIS_RUNNER,
        diagnosis.THIS_TEST,
        ENGINE_RUNNER,
        ENGINE_TEST,
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
            "resumed_from_evaluation_count": truth_metrics[
                "resumed_evaluation_count"
            ],
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "certified_exact_rate_engine": ENGINE_RUNNER,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": geometry.chart_tools.coordinate_tools.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    validation_text = "not evaluated"
    if model_metrics:
        validation_text = (
            f"holdout nonlinear median/max `{model_metrics['holdout_median_nonlinear_departure_rate_relative_error']:.6e}` / "
            f"`{model_metrics['holdout_maximum_nonlinear_departure_rate_relative_error']:.6e}`; full median/max "
            f"`{model_metrics['holdout_median_full_departure_rate_relative_error']:.6e}` / "
            f"`{model_metrics['holdout_maximum_full_departure_rate_relative_error']:.6e}`"
        )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Departure-28 rate validation WP10c9d6c7c3b5c4f25bx",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{truth_metrics['completed_nonbase_rate_evaluations']}` of `{manifest.PLANNED_CANDIDATES}` new exact truth-rate evaluations; failures: `{truth_metrics['failed_rate_evaluations']}`.",
                "",
                "All 9,440 nonlinear/decoder coefficients were frozen and hashed from 160 previously revealed directions before any new rate response was read.",
                "",
                f"Independent validation: {validation_text}.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No trajectory, predictive cycle, or reduced slow evolution is authorized directly.",
                "",
            )
        ),
        encoding="utf-8",
    )
    engine._update_catalog(summary)
    if SCRATCH_DIRECTORY.exists():
        shutil.rmtree(SCRATCH_DIRECTORY)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
