#!/usr/bin/env python3
"""Freeze exact-rate training and blind holdout work at the authentic center."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_authentic_center_geometry_preflight_wp10c9d6c7c3b5c4f25cs as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ct"
PARENT_COMMIT = "c72fb45b95546e13c0d8feb7c29645bb8e6e41e1"
PARENT_PARENT = "8921147bd311600bfcd17efde8bf8ff59b793990"
PARENT_TREE = "ac6ead0ee03a2a9a3658337cdf048112ce36a5a9"
CLASSIFICATION = "authentic_center_exact_rate_training_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25cu"

ACTIVE_SCALE = 1.5e-2
AFFINE_RIDGE = 1.0e-8
REVEALED_COUNT = 16
NEW_TRAINING_COUNT = 4
NEW_HOLDOUT_COUNT = 4
NEW_CENTER_COUNT = 1

ARTIFACT = (
    "causal_inner_authentic_center_exact_rate_training_manifest_"
    "wp10c9d6c7c3b5c4f25ct"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_authentic_center_exact_rate_training_manifest_"
    "wp10c9d6c7c3b5c4f25ct.py"
)
THIS_TEST = (
    "tests/test_causal_inner_authentic_center_exact_rate_training_manifest_"
    "wp10c9d6c7c3b5c4f25ct.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_AUTHENTIC_CENTER_EXACT_RATE_"
    "TRAINING_MANIFEST_WP10C9D6C7C3B5C4F25CT_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

GEOMETRY_ARRAYS = parent.CANONICAL_DIRECTORY / "geometry_arrays.npz"
DESIGN_ARRAYS = parent.manifest.CANONICAL_DIRECTORY / "center_local_field_design.npz"
DIRECT_FIELD = parent.manifest.DIRECT_FIELD

_plain = parent._plain
_read = parent._read
_write_json = parent._write_json
_sha = parent._sha
_checksums = parent._checksums
_load_npz = parent._load_npz


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _relative_rows(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    return np.linalg.norm(left - right, axis=1) / np.maximum(
        np.linalg.norm(right, axis=1), np.finfo(float).tiny
    )


def _active_coordinates(
    local_coordinates: np.ndarray, active_basis: np.ndarray
) -> np.ndarray:
    local = np.asarray(local_coordinates, dtype=float)
    return local[:, -parent.manifest.DEPARTURE_DIMENSION :] @ np.asarray(
        active_basis, dtype=float
    ) / ACTIVE_SCALE


def _affine_features(active_coordinates: np.ndarray) -> np.ndarray:
    active = np.asarray(active_coordinates, dtype=float)
    if active.ndim == 1:
        active = active.reshape(1, -1)
    return np.column_stack((np.ones(active.shape[0]), active))


def _weighted_affine_fit(
    active_coordinates: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    intercept: bool,
    regularization: float = AFFINE_RIDGE,
) -> tuple[np.ndarray, dict]:
    active = np.asarray(active_coordinates, dtype=float)
    design = _affine_features(active) if intercept else active
    target = np.asarray(targets, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if design.shape[0] != target.shape[0] or weight.shape != (design.shape[0],):
        raise ValueError("weighted affine fit dimensions changed")
    normal = design.T @ (weight[:, None] * design)
    penalty = float(regularization) * np.eye(design.shape[1])
    if intercept:
        penalty[0, 0] = 0.0
    regularized = normal + penalty
    coefficients = np.linalg.solve(
        regularized, design.T @ (weight[:, None] * target)
    )
    singular = np.linalg.svd(regularized, compute_uv=False)
    return coefficients, {
        "design_rank": int(np.linalg.matrix_rank(design)),
        "regularized_normal_rank": int(np.linalg.matrix_rank(regularized)),
        "regularized_normal_condition_number": float(singular[0] / singular[-1]),
    }


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("authentic-center geometry commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("authentic-center geometry lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("authentic-center geometry tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "geometry_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    geometry = _load_npz(GEOMETRY_ARRAYS)
    design = _load_npz(DESIGN_ARRAYS)
    if (
        not summary["passed"]
        or summary["classification"] != parent.FULL_CLASSIFICATION
        or summary["largest_passing_component_bound"] != 0.015
        or summary["completed_candidate_count"] != 8
        or summary["failed_candidate_count"] != 0
        or summary["authorized_next"]
        != "definitions_only_authentic_center_exact_rate_training_manifest"
        or summary["new_truth_rate_calls"] != 0
        or summary["new_generator_assemblies"] != 0
        or summary["new_nonlinear_roots"] != 0
        or metrics["passing_role_count"] != 2
        or not all(
            check
            for record in metrics["role_records"]
            for check in record["checks"].values()
        )
        or geometry["candidate_primitive_states"].shape != (8, 112, 5)
        or geometry["candidate_local_coordinates"].shape != (8, 470)
        or tuple(geometry["candidate_role_codes"]) != (0, 0, 0, 0, 1, 1, 1, 1)
        or design["revealed_overlap_local_coordinates"].shape != (16, 470)
    ):
        raise RuntimeError("authentic-center exact-rate authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"authentic-center geometry source changed: {relative}")
    for name, expected in parent.manifest.parent.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("exact-rate training manifest requires a clean tracked tree")
    return {
        "summary": summary,
        "metrics": metrics,
        "hashes": hashes,
        "geometry": geometry,
        "design": design,
    }


def _decoder_design(frozen: dict) -> tuple[dict[str, np.ndarray], dict]:
    design = frozen["design"]
    geometry = frozen["geometry"]
    direct_module = parent.manifest.direct_manifest
    direct = direct_module.DirectCoordinateField(_load_npz(DIRECT_FIELD))
    center_coordinate = np.asarray(
        design["authentic_center_absolute_coordinate"], dtype=float
    )
    center_delta = np.asarray(
        design["authentic_center_scaled_delta"], dtype=float
    )
    decoded_center = direct.decoded_delta(center_coordinate)
    absolute_coordinates = np.vstack(
        (
            design["revealed_overlap_absolute_coordinates"],
            geometry["candidate_absolute_coordinates"],
        )
    )
    absolute_deltas = np.vstack(
        (
            design["revealed_overlap_absolute_scaled_deltas"],
            geometry["candidate_absolute_scaled_deltas"],
        )
    )
    local_coordinates = absolute_coordinates - center_coordinate
    exact_local_deltas = absolute_deltas - center_delta
    translated = np.asarray(
        [
            direct.decoded_delta(coordinate) - decoded_center
            for coordinate in absolute_coordinates
        ]
    )
    residuals = exact_local_deltas - translated
    active = _active_coordinates(
        local_coordinates, design["active_departure_basis"]
    )
    fit_indices = np.arange(REVEALED_COUNT + NEW_TRAINING_COUNT)
    holdout_indices = np.arange(
        REVEALED_COUNT + NEW_TRAINING_COUNT,
        REVEALED_COUNT + NEW_TRAINING_COUNT + NEW_HOLDOUT_COUNT,
    )
    weights = np.concatenate(
        (
            np.full(REVEALED_COUNT, 1.0 / REVEALED_COUNT),
            np.full(NEW_TRAINING_COUNT, 1.0 / NEW_TRAINING_COUNT),
        )
    )
    coefficients, fit_metrics = _weighted_affine_fit(
        active[fit_indices],
        residuals[fit_indices],
        weights,
        intercept=False,
    )
    corrected = translated + active @ coefficients
    errors = _relative_rows(corrected, exact_local_deltas)
    baseline_errors = _relative_rows(translated, exact_local_deltas)
    decoded_holdout_coordinates = []
    minimum_factors = []
    maximum_h_over_r = []
    minimum_optical_depth = []
    for index in holdout_indices:
        state = direct.model.base_state + (
            direct.model.columns.ravel() * (center_delta + corrected[index])
        ).reshape(direct.model.base_state.shape)
        coordinate, factors = direct.model.coordinate(state)
        decoded_holdout_coordinates.append(coordinate - center_coordinate)
        physical = (
            direct_module.parent.manifest.parent.vector_field.manifest.parent.geometry.chart_tools._state_audit(
                direct.model.components["context"], state
            )
        )
        minimum_factors.append(
            min(float(np.min(factors)), physical["minimum_reconstruction_factor"])
        )
        maximum_h_over_r.append(physical["maximum_h_over_r"])
        minimum_optical_depth.append(
            physical["minimum_scattering_optical_depth"]
        )
    decoded_holdout_coordinates = np.asarray(decoded_holdout_coordinates)
    coordinate_errors = _relative_rows(
        decoded_holdout_coordinates, local_coordinates[holdout_indices]
    )
    groups = {
        "revealed": np.arange(REVEALED_COUNT),
        "new_training": np.arange(REVEALED_COUNT, REVEALED_COUNT + NEW_TRAINING_COUNT),
        "new_holdout": holdout_indices,
    }
    metrics = {
        **fit_metrics,
        "center_anchor_infinity_defect": 0.0,
        "coefficient_norm": float(np.linalg.norm(coefficients)),
        "maximum_revealed_baseline_relative_error": float(
            np.max(baseline_errors[groups["revealed"]])
        ),
        "maximum_revealed_corrected_relative_error": float(
            np.max(errors[groups["revealed"]])
        ),
        "maximum_new_training_baseline_relative_error": float(
            np.max(baseline_errors[groups["new_training"]])
        ),
        "maximum_new_training_corrected_relative_error": float(
            np.max(errors[groups["new_training"]])
        ),
        "maximum_new_holdout_baseline_relative_error": float(
            np.max(baseline_errors[groups["new_holdout"]])
        ),
        "maximum_new_holdout_corrected_relative_error": float(
            np.max(errors[groups["new_holdout"]])
        ),
        "median_new_holdout_corrected_relative_error": float(
            np.median(errors[groups["new_holdout"]])
        ),
        "maximum_new_holdout_coordinate_relative_mismatch": float(
            np.max(coordinate_errors)
        ),
        "minimum_new_holdout_reconstruction_factor": float(
            np.min(minimum_factors)
        ),
        "maximum_new_holdout_H_over_R": float(np.max(maximum_h_over_r)),
        "minimum_new_holdout_scattering_optical_depth": float(
            np.min(minimum_optical_depth)
        ),
    }
    arrays = {
        "decoder_affine_coefficients": coefficients,
        "decoder_training_weights": weights,
        "decoder_active_coordinates": active,
        "translated_decoder_local_deltas": translated,
        "corrected_decoder_local_deltas": corrected,
        "exact_local_scaled_deltas": exact_local_deltas,
        "decoder_relative_errors": errors,
        "decoder_baseline_relative_errors": baseline_errors,
        "decoded_holdout_local_coordinates": decoded_holdout_coordinates,
        "decoder_holdout_coordinate_relative_errors": coordinate_errors,
    }
    return arrays, metrics


def _revealed_rate_readiness(frozen: dict) -> tuple[dict[str, np.ndarray], dict]:
    design = frozen["design"]
    local_coordinates = np.asarray(
        design["revealed_overlap_local_coordinates"], dtype=float
    )
    active = _active_coordinates(
        local_coordinates, design["active_departure_basis"]
    )
    exact_full = np.asarray(
        design["revealed_overlap_exact_full_rates_per_second"], dtype=float
    )
    old_full = np.asarray(
        design["revealed_overlap_old_predicted_full_rates_per_second"], dtype=float
    )
    exact_coordinate = np.asarray(
        design["revealed_overlap_exact_coordinate_rates_per_second"], dtype=float
    )
    old_coordinate = np.asarray(
        design["revealed_overlap_old_predicted_coordinate_rates_per_second"], dtype=float
    )
    restriction = np.asarray(
        design["authentic_center_fixed_restriction"], dtype=float
    )
    training = np.arange(8)
    controls = np.arange(8, 16)
    uniform = np.ones(training.size)
    full_coefficients, full_fit = _weighted_affine_fit(
        active[training],
        exact_full[training] - old_full[training],
        uniform,
        intercept=True,
    )
    features = _affine_features(active)
    corrected_full = old_full + features @ full_coefficients
    physical = restriction[: parent.manifest.PHYSICAL_DIMENSION]
    q_coefficients, q_fit = _weighted_affine_fit(
        active[training],
        exact_coordinate[training, : parent.manifest.PHYSICAL_DIMENSION]
        - (physical @ corrected_full[training].T).T,
        uniform,
        intercept=True,
    )
    corrected_coordinate = np.empty_like(exact_coordinate)
    corrected_coordinate[:, : parent.manifest.PHYSICAL_DIMENSION] = (
        physical @ corrected_full.T
    ).T + features @ q_coefficients
    corrected_coordinate[
        :,
        parent.manifest.PHYSICAL_DIMENSION : (
            parent.manifest.PHYSICAL_DIMENSION + parent.manifest.MEMORY_DIMENSION
        ),
    ] = (restriction[162:442] @ corrected_full.T).T
    corrected_coordinate[:, -parent.manifest.DEPARTURE_DIMENSION :] = (
        restriction[-parent.manifest.DEPARTURE_DIMENSION :] @ corrected_full.T
    ).T
    blocks = {
        "full": slice(None),
        "q162": slice(0, 162),
        "z280": slice(162, 442),
        "a28": slice(442, 470),
    }
    control_metrics = {
        "maximum_full_state_rate_relative_error": float(
            np.max(_relative_rows(corrected_full[controls], exact_full[controls]))
        ),
        "median_full_state_rate_relative_error": float(
            np.median(_relative_rows(corrected_full[controls], exact_full[controls]))
        ),
    }
    for name, selection in blocks.items():
        errors = _relative_rows(
            corrected_coordinate[controls, selection],
            exact_coordinate[controls, selection],
        )
        control_metrics[f"maximum_{name}_rate_relative_error"] = float(
            np.max(errors)
        )
        control_metrics[f"median_{name}_rate_relative_error"] = float(
            np.median(errors)
        )
    metrics = {
        "training_sample_count": int(training.size),
        "revealed_control_sample_count": int(controls.size),
        "full_fit": full_fit,
        "q_fit": q_fit,
        **control_metrics,
    }
    arrays = {
        "readiness_full_rate_coefficients": full_coefficients,
        "readiness_q_rate_coefficients": q_coefficients,
        "readiness_corrected_full_rates_per_second": corrected_full,
        "readiness_corrected_coordinate_rates_per_second": corrected_coordinate,
        "readiness_active_coordinates": active,
    }
    return arrays, metrics


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "frozen_local_field": {
            "active_coordinates": "three_center_local_departure_coordinates_divided_by_0p015",
            "active_scale": ACTIVE_SCALE,
            "full_rate_prior": "independently_validated_old_direct_full_physical_rate",
            "full_rate_correction": "weighted_affine_constant_plus_three_active_coordinates",
            "q162_rate_prior": "fixed_authentic_center_physical_Jacobian_times_corrected_full_rate",
            "q162_rate_correction": "weighted_affine_constant_plus_three_active_coordinates",
            "z280_and_a28_rates": "fixed_projections_of_corrected_full_rate",
            "decoder_prior": "translated_compensated_old_decoder_exactly_reanchored_at_authentic_center",
            "decoder_correction": "center_zero_affine_three_active_coordinate_residual",
            "affine_ridge": AFFINE_RIDGE,
            "online_state_dependent_coordinate_Jacobian_calls": 0,
            "new_complete_generator_assemblies": 0,
        },
        "fit_database": {
            "revealed_overlap_samples": REVEALED_COUNT,
            "authentic_center_new_exact_rate_samples": NEW_CENTER_COUNT,
            "forward_training_new_exact_rate_samples": NEW_TRAINING_COUNT,
            "group_total_weights": {
                "revealed_overlap": 1.0,
                "authentic_center": 1.0,
                "new_forward_training": 1.0,
            },
            "coefficients_frozen_before_holdout_truth": True,
        },
        "execution_order": [
            "authentic_center_exact_continuous_rate",
            "training_0_exact_continuous_rate",
            "training_1_exact_continuous_rate",
            "training_2_exact_continuous_rate",
            "training_3_exact_continuous_rate",
            "fit_and_hash_local_field_coefficients",
        ],
        "coefficient_blind_holdout": {
            "count": NEW_HOLDOUT_COUNT,
            "rate_truth_forbidden_during_training_package": True,
            "states_and_geometry_hash_locked_before_training_truth": True,
            "separate_future_work_package_required": True,
        },
        "binding_manifest_gates": {
            "decoder_regularized_normal_condition_number_max": 1.0e2,
            "decoder_center_anchor_infinity_defect_max": 1.0e-14,
            "decoder_revealed_relative_error_max": 5.0e-2,
            "decoder_new_training_relative_error_max": 2.0e-2,
            "decoder_new_holdout_relative_error_max": 5.0e-2,
            "decoder_new_holdout_coordinate_relative_mismatch_max": 7.5e-2,
            "minimum_decoder_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_decoder_H_over_R": 0.12,
            "minimum_decoder_scattering_optical_depth": 1.0,
            "readiness_regularized_normal_condition_number_max": 1.0e3,
            "readiness_full_state_rate_relative_error_max": 5.0e-2,
            "readiness_full_coordinate_rate_relative_error_max": 5.0e-2,
            "readiness_q162_rate_relative_error_max": 5.0e-2,
            "readiness_z280_rate_relative_error_max": 5.0e-2,
            "readiness_a28_rate_relative_error_max": 5.0e-2,
        },
        "training_execution_gates": {
            "completed_exact_rate_calls_equal": 5,
            "failed_exact_rate_calls_equal": 0,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_raw_Schur_condition_number": 1.0e6,
            "maximum_reaction_identity_defect": 1.0e-9,
            "maximum_rate_tangency_relative_defect": 1.0e-8,
            "maximum_coordinate_Jacobian_condition_number": 5.0e3,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_incoming_excision_characteristics_equal": 0,
            "maximum_training_full_state_rate_relative_error": 7.5e-2,
            "maximum_training_full_coordinate_rate_relative_error": 7.5e-2,
            "maximum_training_q162_rate_relative_error": 7.5e-2,
            "maximum_training_z280_rate_relative_error": 7.5e-2,
            "maximum_training_a28_rate_relative_error": 7.5e-2,
            "maximum_regularized_normal_condition_number": 1.0e4,
        },
        "cost_budget": {
            "new_exact_continuous_rate_calls_equal": 5,
            "new_complete_generator_assemblies_equal": 0,
            "new_nonlinear_fixed_Q_roots_equal": 0,
            "propagated_states_equal": 0,
            "holdout_rate_calls_equal": 0,
        },
        "decision": {
            "pass_classification": "authentic_center_local_field_coefficients_frozen",
            "pass_authorizes_only": "WP10c9d6c7c3b5c4f25cv",
            "fail_classification": "authentic_center_local_field_training_failed",
            "fail_authorizes_only": "definitions_only_local_field_fit_revision",
        },
        "authorization_boundaries": {
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "fast_average_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _checks(decoder: dict, readiness: dict, gates: dict) -> dict:
    maximum_readiness_condition = max(
        readiness["full_fit"]["regularized_normal_condition_number"],
        readiness["q_fit"]["regularized_normal_condition_number"],
    )
    return {
        "decoder_condition": decoder["regularized_normal_condition_number"]
        <= gates["decoder_regularized_normal_condition_number_max"],
        "decoder_anchor": decoder["center_anchor_infinity_defect"]
        <= gates["decoder_center_anchor_infinity_defect_max"],
        "decoder_revealed": decoder["maximum_revealed_corrected_relative_error"]
        <= gates["decoder_revealed_relative_error_max"],
        "decoder_training": decoder[
            "maximum_new_training_corrected_relative_error"
        ] <= gates["decoder_new_training_relative_error_max"],
        "decoder_holdout": decoder[
            "maximum_new_holdout_corrected_relative_error"
        ] <= gates["decoder_new_holdout_relative_error_max"],
        "decoder_coordinate": decoder[
            "maximum_new_holdout_coordinate_relative_mismatch"
        ] <= gates["decoder_new_holdout_coordinate_relative_mismatch_max"],
        "decoder_reconstruction": decoder[
            "minimum_new_holdout_reconstruction_factor"
        ] >= gates["minimum_decoder_reconstruction_factor"],
        "decoder_height": decoder["maximum_new_holdout_H_over_R"]
        <= gates["maximum_decoder_H_over_R"],
        "decoder_optical_depth": decoder[
            "minimum_new_holdout_scattering_optical_depth"
        ] >= gates["minimum_decoder_scattering_optical_depth"],
        "readiness_condition": maximum_readiness_condition
        <= gates["readiness_regularized_normal_condition_number_max"],
        "readiness_full_state": readiness[
            "maximum_full_state_rate_relative_error"
        ] <= gates["readiness_full_state_rate_relative_error_max"],
        "readiness_full_coordinate": readiness[
            "maximum_full_rate_relative_error"
        ] <= gates["readiness_full_coordinate_rate_relative_error_max"],
        "readiness_q162": readiness["maximum_q162_rate_relative_error"]
        <= gates["readiness_q162_rate_relative_error_max"],
        "readiness_z280": readiness["maximum_z280_rate_relative_error"]
        <= gates["readiness_z280_rate_relative_error_max"],
        "readiness_a28": readiness["maximum_a28_rate_relative_error"]
        <= gates["readiness_a28_rate_relative_error_max"],
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
                    "scientific_status": "DEFINITIONS_ONLY",
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
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("exact-rate training manifest already exists")
    decoder_arrays, decoder_metrics = _decoder_design(frozen)
    readiness_arrays, readiness_metrics = _revealed_rate_readiness(frozen)
    contract = _contract()
    checks = _checks(
        decoder_metrics,
        readiness_metrics,
        contract["binding_manifest_gates"],
    )
    if not all(checks.values()):
        raise RuntimeError(f"exact-rate training readiness failed: {checks}")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(
        CANONICAL_DIRECTORY / "frozen_rate_training_design.npz",
        **decoder_arrays,
        **readiness_arrays,
        authentic_center_primitive_state=frozen["design"][
            "authentic_center_primitive_state"
        ],
        authentic_center_scaled_delta=frozen["design"][
            "authentic_center_scaled_delta"
        ],
        authentic_center_absolute_coordinate=frozen["design"][
            "authentic_center_absolute_coordinate"
        ],
        authentic_center_fixed_restriction=frozen["design"][
            "authentic_center_fixed_restriction"
        ],
        active_departure_basis=frozen["design"]["active_departure_basis"],
        training_primitive_states=frozen["geometry"][
            "candidate_primitive_states"
        ][:4],
        training_local_coordinates=frozen["geometry"][
            "candidate_local_coordinates"
        ][:4],
        training_absolute_coordinates=frozen["geometry"][
            "candidate_absolute_coordinates"
        ][:4],
        holdout_primitive_states=frozen["geometry"][
            "candidate_primitive_states"
        ][4:],
        holdout_local_coordinates=frozen["geometry"][
            "candidate_local_coordinates"
        ][4:],
        holdout_absolute_coordinates=frozen["geometry"][
            "candidate_absolute_coordinates"
        ][4:],
        revealed_overlap_exact_full_rates_per_second=frozen["design"][
            "revealed_overlap_exact_full_rates_per_second"
        ],
        revealed_overlap_exact_coordinate_rates_per_second=frozen["design"][
            "revealed_overlap_exact_coordinate_rates_per_second"
        ],
        revealed_overlap_old_predicted_full_rates_per_second=frozen["design"][
            "revealed_overlap_old_predicted_full_rates_per_second"
        ],
        revealed_overlap_local_coordinates=frozen["design"][
            "revealed_overlap_local_coordinates"
        ],
    )
    _write_json(
        CANONICAL_DIRECTORY / "readiness_metrics.json",
        {
            "checks": checks,
            "passed": True,
            "decoder": decoder_metrics,
            "revealed_rate_cross_validation": readiness_metrics,
        },
    )
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "geometry_arrays_sha256": _sha(GEOMETRY_ARRAYS),
            "design_arrays_sha256": _sha(DESIGN_ARRAYS),
            "direct_field_sha256": _sha(DIRECT_FIELD),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "decoder_geometry_passed": True,
        "revealed_rate_readiness_passed": True,
        "planned_new_training_exact_rate_calls": 5,
        "planned_future_blind_holdout_exact_rate_calls": 4,
        "coefficients_frozen_before_holdout_truth": True,
        "new_truth_rate_calls": 0,
        "new_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        parent.manifest.THIS_RUNNER,
        parent.manifest.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
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
                "# Authentic-center exact-rate training manifest WP10c9d6c7c3b5c4f25ct",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                f"The center-local affine decoder is independently geometry-checked on four 0.015 holdouts. Its maximum holdout error is `{decoder_metrics['maximum_new_holdout_corrected_relative_error']:.6e}`, versus `{decoder_metrics['maximum_new_holdout_baseline_relative_error']:.6e}` for untranslated correction-free continuation.",
                "",
                f"A revealed-only eight-versus-eight rate cross-validation reduces the maximum full-coordinate control error to `{readiness_metrics['maximum_full_rate_relative_error']:.6e}` and a28 error to `{readiness_metrics['maximum_a28_rate_relative_error']:.6e}`. This is readiness evidence, not independent validation.",
                "",
                "The next execution may evaluate exactly the authentic center and four training rates, then fit and hash the local coefficients. The four holdout rates remain forbidden until a separate package; no complete generator or BDF root is authorized.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`. No physical microburst, predictive cycle, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
