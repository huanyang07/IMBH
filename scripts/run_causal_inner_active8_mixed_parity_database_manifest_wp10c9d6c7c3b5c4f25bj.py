#!/usr/bin/env python3
"""Freeze the active-8 mixed-direction parity database contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_parity_low_rank_architecture_audit_wp10c9d6c7c3b5c4f25bi as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bj"
CLASSIFICATION = (
    "active8_mixed_direction_parity_database_manifest_frozen_"
    "geometry_preflight_authorized"
)
PARENT_COMMIT = "f097547d485e7476ad9049922b9d920b2fb995a2"
PARENT_PARENT = "0cde0a6336361401763352cf6eccbd1773ea7a41"
PARENT_TREE = "1ea6628d5df2fadc50b22ec4b9ee744b89e35d09"

ARTIFACT = (
    "causal_inner_active8_mixed_parity_database_manifest_"
    "wp10c9d6c7c3b5c4f25bj"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_active8_mixed_parity_database_manifest_"
    "wp10c9d6c7c3b5c4f25bj.py"
)
THIS_TEST = (
    "tests/test_causal_inner_active8_mixed_parity_database_manifest_"
    "wp10c9d6c7c3b5c4f25bj.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_active8_mixed_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25bk.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_active8_mixed_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25bk.py"
)
RATE_RUNNER = (
    "scripts/run_causal_inner_active8_mixed_parity_rate_fit_"
    "wp10c9d6c7c3b5c4f25bl.py"
)
RATE_TEST = (
    "tests/test_causal_inner_active8_mixed_parity_rate_fit_"
    "wp10c9d6c7c3b5c4f25bl.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ACTIVE8_MIXED_PARITY_"
    "DATABASE_MANIFEST_WP10C9D6C7C3B5C4F25BJ_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

GEOMETRY_PATH = parent.manifest.GEOMETRY_PATH
LOW_CHART_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_exact_geometric_departure_chart_"
    "preflight_wp10c9d6c7c3b5c4f25ay"
)
LOW_CHART_PATH = LOW_CHART_DIRECTORY / "geometric_departure_chart.npz"
HIGH_CHART_DIRECTORY = parent.manifest.rate_0p01.manifest.parent.CANONICAL_DIRECTORY
HIGH_CHART_PATH = HIGH_CHART_DIRECTORY / "expanded_departure_chart.npz"
LOW_RATE_PATH = parent.manifest.RATE_0P005_PATH
HIGH_RATE_PATH = parent.manifest.RATE_0P01_PATH

ACTIVE_DIMENSION = 8
DEPARTURE_DIMENSION = 28
HIDDEN_DIMENSION = 90
TRAINING_DIRECTION_COUNT = 40
TUNING_DIRECTION_COUNT = 8
HOLDOUT_DIRECTION_COUNT = 8
SCALING_DIRECTION_COUNT = TUNING_DIRECTION_COUNT
HIGH_COMPONENT_BOUND = 1.0e-2
LOW_COMPONENT_BOUND = 5.0e-3
PLANNED_HIGH_DIRECTIONS = (
    TRAINING_DIRECTION_COUNT + TUNING_DIRECTION_COUNT + HOLDOUT_DIRECTION_COUNT
)
PLANNED_LOW_DIRECTIONS = SCALING_DIRECTION_COUNT
PLANNED_CANDIDATES = 2 * (PLANNED_HIGH_DIRECTIONS + PLANNED_LOW_DIRECTIONS)
DESIGN_SEED = 2502
DESIGN_POOL_SIZE = 20_000


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _quadratic_features(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=float)
    columns = [directions[:, index] ** 2 for index in range(ACTIVE_DIMENSION)]
    columns.extend(
        np.sqrt(2.0) * directions[:, left] * directions[:, right]
        for left in range(ACTIVE_DIMENSION)
        for right in range(left + 1, ACTIVE_DIMENSION)
    )
    return np.asarray(columns, dtype=float).T


def _direction_design() -> tuple[dict, dict[str, np.ndarray]]:
    rng = np.random.default_rng(DESIGN_SEED)
    pool = rng.normal(size=(DESIGN_POOL_SIZE, ACTIVE_DIMENSION))
    pool /= np.linalg.norm(pool, axis=1)[:, None]
    pivots = np.argmax(np.abs(pool), axis=1)
    signs = np.sign(pool[np.arange(pool.shape[0]), pivots])
    pool *= signs[:, None]
    pool = pool[np.max(np.abs(pool), axis=1) <= 0.75]
    _unique, unique_indices = np.unique(
        np.round(pool, decimals=12), axis=0, return_index=True
    )
    pool = pool[np.sort(unique_indices)]
    features = _quadratic_features(pool)

    selected = [int(np.argmin(np.max(np.abs(pool), axis=1)))]
    for _ in range(35):
        basis = np.linalg.qr(features[selected].T, mode="reduced")[0]
        residual = features.T - basis @ (basis.T @ features.T)
        scores = np.sum(residual * residual, axis=0)
        scores[selected] = -1.0
        selected.append(int(np.argmax(scores)))
    for _ in range(TRAINING_DIRECTION_COUNT - len(selected)):
        coherence = np.max(np.abs(pool @ pool[selected].T), axis=1)
        scores = 1.0 - coherence
        scores[selected] = -1.0
        selected.append(int(np.argmax(scores)))
    training = pool[selected]

    remaining_selected = list(selected)
    validation_indices = []
    for _ in range(TUNING_DIRECTION_COUNT + HOLDOUT_DIRECTION_COUNT):
        coherence = np.max(
            np.abs(pool @ pool[remaining_selected].T), axis=1
        )
        scores = 1.0 - coherence
        scores[remaining_selected] = -1.0
        chosen = int(np.argmax(scores))
        validation_indices.append(chosen)
        remaining_selected.append(chosen)
    tuning = pool[validation_indices[:TUNING_DIRECTION_COUNT]]
    holdout = pool[validation_indices[TUNING_DIRECTION_COUNT:]]

    quadratic = _quadratic_features(training)
    cubic_kernel = (training @ training.T) ** 3
    quartic_kernel = (training @ training.T) ** 4

    def separation(left: np.ndarray, right: np.ndarray, same: bool) -> float:
        values = 1.0 - np.abs(left @ right.T)
        if same:
            values = values + 2.0 * np.eye(left.shape[0])
        return float(np.min(values))

    metrics = {
        "design_seed": DESIGN_SEED,
        "raw_pool_size": DESIGN_POOL_SIZE,
        "filtered_pool_size": int(pool.shape[0]),
        "training_direction_count": int(training.shape[0]),
        "tuning_direction_count": int(tuning.shape[0]),
        "holdout_direction_count": int(holdout.shape[0]),
        "maximum_absolute_direction_component": float(
            np.max(np.abs(np.vstack((training, tuning, holdout))))
        ),
        "training_projective_separation": separation(training, training, True),
        "tuning_projective_separation": separation(tuning, tuning, True),
        "holdout_projective_separation": separation(holdout, holdout, True),
        "validation_to_training_projective_separation": separation(
            np.vstack((tuning, holdout)), training, False
        ),
        "quadratic_feature_rank": int(np.linalg.matrix_rank(quadratic)),
        "quadratic_feature_condition_number": float(np.linalg.cond(quadratic)),
        "cubic_kernel_rank": int(np.linalg.matrix_rank(cubic_kernel)),
        "cubic_kernel_condition_number": float(np.linalg.cond(cubic_kernel)),
        "quartic_kernel_rank": int(np.linalg.matrix_rank(quartic_kernel)),
        "quartic_kernel_condition_number": float(np.linalg.cond(quartic_kernel)),
    }
    arrays = {
        "training_directions_active8": training.T,
        "tuning_directions_active8": tuning.T,
        "holdout_directions_active8": holdout.T,
        "training_quadratic_feature_matrix": quadratic,
        "training_cubic_kernel_matrix": cubic_kernel,
        "training_quartic_kernel_matrix": quartic_kernel,
    }
    return metrics, arrays


def _decoder_diagnosis() -> dict:
    with np.load(GEOMETRY_PATH, allow_pickle=False) as source:
        restriction = np.asarray(source["online_coordinate_restriction"], dtype=float)
        lifting = np.asarray(source["online_coordinate_lifting"], dtype=float)

    def parity(directory: Path, amplitude_index: int):
        metrics = _read(directory / "metrics.json")
        chart_path = next(directory.glob("*departure_chart.npz"))
        with np.load(chart_path, allow_pickle=False) as source:
            deltas = np.asarray(source["candidate_scaled_deltas"], dtype=float)
            coordinates = np.asarray(
                source["candidate_departure_coordinates"], dtype=float
            )
        hidden = deltas - (lifting @ (restriction @ deltas.T)).T
        radii = []
        even = []
        odd = []
        fractions = []
        for direction_index in range(ACTIVE_DIMENSION):
            indices = [
                index
                for index, item in enumerate(metrics["candidates"])
                if item["amplitude_index"] == amplitude_index
                and item["direction_index"] == direction_index
            ]
            negative, positive = sorted(
                indices, key=lambda index: metrics["candidates"][index]["sign"]
            )
            radius = float(
                np.linalg.norm(0.5 * (coordinates[positive] - coordinates[negative]))
            )
            radii.append(radius)
            even.append(0.5 * (hidden[positive] + hidden[negative]))
            odd.append(0.5 * (hidden[positive] - hidden[negative]))
            for index in (negative, positive):
                fractions.append(
                    float(
                        np.linalg.norm(hidden[index])
                        / max(np.linalg.norm(deltas[index]), np.finfo(float).tiny)
                    )
                )
        return tuple(np.asarray(value) for value in (radii, even, odd, fractions))

    low_radius, low_even, low_odd, low_fraction = parity(LOW_CHART_DIRECTORY, 2)
    high_radius, high_even, high_odd, high_fraction = parity(
        HIGH_CHART_DIRECTORY, 0
    )

    def exponents(low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return np.log(
            np.linalg.norm(high, axis=1) / np.linalg.norm(low, axis=1)
        ) / np.log(high_radius / low_radius)

    def balanced_energy(values: np.ndarray) -> list[float]:
        balanced = values / np.linalg.norm(values, axis=1)[:, None]
        singular = np.linalg.svd(balanced, compute_uv=False)
        return np.cumsum(singular * singular).tolist()

    cubic_energy = np.asarray(balanced_energy(high_odd), dtype=float)
    cubic_energy /= cubic_energy[-1]
    quartic_energy = np.asarray(balanced_energy(high_even), dtype=float)
    quartic_energy /= quartic_energy[-1]
    odd_exponents = exponents(low_odd, high_odd)
    even_exponents = exponents(low_even, high_even)
    return {
        "post_result_diagnosis": True,
        "independent_validation_claimed": False,
        "hidden_dimension": HIDDEN_DIMENSION,
        "median_hidden_state_fraction_at_0p005": float(np.median(low_fraction)),
        "maximum_hidden_state_fraction_at_0p005": float(np.max(low_fraction)),
        "median_hidden_state_fraction_at_0p01": float(np.median(high_fraction)),
        "maximum_hidden_state_fraction_at_0p01": float(np.max(high_fraction)),
        "median_odd_decoder_exponent": float(np.median(odd_exponents)),
        "minimum_odd_decoder_exponent": float(np.min(odd_exponents)),
        "maximum_odd_decoder_exponent": float(np.max(odd_exponents)),
        "median_even_decoder_exponent": float(np.median(even_exponents)),
        "minimum_even_decoder_exponent": float(np.min(even_exponents)),
        "maximum_even_decoder_exponent": float(np.max(even_exponents)),
        "cubic_decoder_rank4_energy": float(cubic_energy[3]),
        "cubic_decoder_rank6_energy": float(cubic_energy[5]),
        "quartic_decoder_rank4_energy": float(quartic_energy[3]),
        "quartic_decoder_rank6_energy": float(quartic_energy[5]),
        "selected_decoder": (
            "linear_470_lifting_plus_full_output_active8_cubic_odd_and_"
            "quartic_even_polynomial_kernel_corrections"
        ),
    }


def _validate_parent() -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("mixed-database parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("mixed-database parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("mixed-database parent tree changed")
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or summary["authorized_next"]
        != "definitions_only_active8_mixed_direction_parity_database_manifest"
        or summary["mixed_direction_coefficients_identified"]
        or summary["predictive_cycle_authorized"]
    ):
        raise RuntimeError("mixed-database authorization changed")
    packages = (
        parent.CANONICAL_DIRECTORY,
        LOW_CHART_DIRECTORY,
        HIGH_CHART_DIRECTORY,
        parent.manifest.rate_0p005.CANONICAL_DIRECTORY,
        parent.manifest.rate_0p01.CANONICAL_DIRECTORY,
        parent.manifest.architecture.CANONICAL_DIRECTORY,
    )
    return {
        "summary": summary,
        "package_hashes": {
            directory.name: _checksums(directory) for directory in packages
        },
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "identify_and_independently_test_a_local_zero_truth_call_"
            "nonlinear_closure_and_decoder_on_mixed_active8_departures"
        ),
        "fixed_online_architecture": {
            "state": "q162_plus_z280_plus_a28_equals_470",
            "active_nonlinear_input": "xi8_equals_W8_transpose_times_a28",
            "truncated_hidden_state": 90,
            "physical_update": "exact_conservative_finite_volume",
            "stable_memory_update": "inherited_energy_stable_IMEX_or_ETD",
            "truth_calls_per_online_macrostep": 0,
        },
        "database_design": {
            "active_dimension": ACTIVE_DIMENSION,
            "training_directions_at_0p01": TRAINING_DIRECTION_COUNT,
            "tuning_directions_at_0p01": TUNING_DIRECTION_COUNT,
            "same_tuning_directions_at_0p005": SCALING_DIRECTION_COUNT,
            "untouched_holdout_directions_at_0p01": HOLDOUT_DIRECTION_COUNT,
            "signs": [-1, 1],
            "planned_high_amplitude_candidates": 2 * PLANNED_HIGH_DIRECTIONS,
            "planned_low_amplitude_candidates": 2 * PLANNED_LOW_DIRECTIONS,
            "planned_total_candidates": PLANNED_CANDIDATES,
            "deterministic_design_seed": DESIGN_SEED,
            "direction_equivalence": "projective_because_each_direction_uses_both_signs",
        },
        "exact_geometry": {
            "method": "inherited_exact_Newton_retraction_on_C_phys",
            "rate_or_reaction_lift_used": False,
            "propagated_states": 0,
            "candidate_history_used": False,
            "failure_policy": "stop_before_any_rate_evaluation",
        },
        "parity_targets": {
            "departure_rate_even": "quadratic",
            "departure_rate_odd": "cubic",
            "hidden_decoder_odd": "cubic",
            "hidden_decoder_even": "quartic",
            "higher_order_terms_are_measured_validation_error_not_silently_discarded": True,
        },
        "closure_models": {
            "rate_quadratic": (
                "full_28_output_least_squares_on_all_36_symmetric_"
                "homogeneous_quadratic_features"
            ),
            "rate_cubic": (
                "full_28_output_degree3_polynomial_kernel_on_40_training_centers"
            ),
            "decoder_cubic": (
                "full_90_output_degree3_polynomial_kernel_on_40_training_centers"
            ),
            "decoder_quartic": (
                "full_90_output_degree4_polynomial_kernel_on_40_training_centers"
            ),
            "interpolation_regularization": 0.0,
            "stored_coefficient_upper_bound": 9_328,
            "online_truth_calls": 0,
            "online_Newton_retractions": 0,
        },
        "design_gates": {
            "quadratic_feature_rank_equal": 36,
            "quadratic_feature_condition_number_max": 8.0,
            "cubic_kernel_rank_equal": TRAINING_DIRECTION_COUNT,
            "cubic_kernel_condition_number_max": 10.0,
            "quartic_kernel_rank_equal": TRAINING_DIRECTION_COUNT,
            "quartic_kernel_condition_number_max": 10.0,
            "maximum_absolute_direction_component": 0.75,
            "minimum_projective_separation": 0.30,
        },
        "binding_geometry_gates": {
            "completed_candidate_count_equal": PLANNED_CANDIDATES,
            "failed_candidate_count_equal": 0,
            "maximum_coordinate_residual_infinity": 1.0e-10,
            "maximum_normalized_Q3_defect": 1.0e-10,
            "maximum_final_scaled_component": HIGH_COMPONENT_BOUND,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_coordinate_Jacobian_condition_number": 5.0e3,
            "minimum_departure_direction_alignment_cosine": 0.995,
            "maximum_departure_transverse_fraction": 0.05,
            "maximum_pair_coordinate_odd_symmetry_defect": 0.02,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "nonbase_continuous_rate_evaluations_equal": 0,
            "new_full_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "binding_truth_rate_gates": {
            "completed_nonbase_rate_evaluations_equal": PLANNED_CANDIDATES,
            "failed_rate_evaluations_equal": 0,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_raw_Schur_condition_number": 1.0e6,
            "maximum_reaction_identity_defect": 1.0e-9,
            "maximum_rate_tangency_relative_defect": 1.0e-8,
            "maximum_coordinate_Jacobian_condition_number": 5.0e3,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_incoming_excision_characteristics_equal": 0,
            "new_complete_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "binding_model_validation_gates": {
            "tuning_median_departure_rate_relative_error": 0.10,
            "tuning_maximum_departure_rate_relative_error": 0.25,
            "holdout_median_departure_rate_relative_error": 0.10,
            "holdout_maximum_departure_rate_relative_error": 0.25,
            "tuning_median_hidden_decoder_relative_error": 0.15,
            "tuning_maximum_hidden_decoder_relative_error": 0.35,
            "holdout_median_hidden_decoder_relative_error": 0.15,
            "holdout_maximum_hidden_decoder_relative_error": 0.35,
            "maximum_full_scaled_state_decoder_relative_error": 2.5e-3,
            "maximum_reconstructed_C_phys_residual_infinity": 2.5e-4,
            "minimum_reconstructed_state_reconstruction_factor": 1.0 - 1.0e-12,
        },
        "decision": {
            "geometry_pass": {
                "classification": "active8_mixed_geometry_passed_rate_fit_authorized",
                "authorizes_only": "WP10c9d6c7c3b5c4f25bl",
            },
            "geometry_fail": {
                "classification": "active8_mixed_geometry_failed_rate_fit_blocked",
                "authorizes_only": None,
            },
            "model_pass": {
                "classification": (
                    "active8_mixed_nonlinear_closure_and_decoder_locally_validated"
                ),
                "authorizes_only": (
                    "definitions_only_local_470_closure_short_trajectory_manifest"
                ),
            },
            "model_fail": {
                "classification": (
                    "active8_mixed_model_validation_failed_adaptive_database_"
                    "extension_required"
                ),
                "authorizes_only": None,
            },
        },
        "claim_boundary": {
            "decoder_degree_selection_is_post_result": True,
            "mixed_direction_thresholds_selected_before_mixed_results": True,
            "mixed_direction_coefficients_identified": False,
            "local_closure_validated": False,
            "multi_anchor_atlas_built": False,
            "online_integrator_implemented": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE",
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
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
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
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    parent_data = _validate_parent()
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("mixed-database manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("mixed-database manifest is already frozen")
    design_metrics, design_arrays = _direction_design()
    design_gates = _contract()["design_gates"]
    design_checks = {
        "quadratic_rank": design_metrics["quadratic_feature_rank"]
        == design_gates["quadratic_feature_rank_equal"],
        "quadratic_condition": design_metrics[
            "quadratic_feature_condition_number"
        ]
        <= design_gates["quadratic_feature_condition_number_max"],
        "cubic_rank": design_metrics["cubic_kernel_rank"]
        == design_gates["cubic_kernel_rank_equal"],
        "cubic_condition": design_metrics["cubic_kernel_condition_number"]
        <= design_gates["cubic_kernel_condition_number_max"],
        "quartic_rank": design_metrics["quartic_kernel_rank"]
        == design_gates["quartic_kernel_rank_equal"],
        "quartic_condition": design_metrics["quartic_kernel_condition_number"]
        <= design_gates["quartic_kernel_condition_number_max"],
        "component": design_metrics["maximum_absolute_direction_component"]
        <= design_gates["maximum_absolute_direction_component"],
        "training_separation": design_metrics["training_projective_separation"]
        >= design_gates["minimum_projective_separation"],
        "tuning_separation": design_metrics["tuning_projective_separation"]
        >= design_gates["minimum_projective_separation"],
        "holdout_separation": design_metrics["holdout_projective_separation"]
        >= design_gates["minimum_projective_separation"],
        "validation_separation": design_metrics[
            "validation_to_training_projective_separation"
        ]
        >= design_gates["minimum_projective_separation"],
    }
    if not all(design_checks.values()):
        raise RuntimeError(f"mixed-direction design failed: {design_checks}")
    decoder = _decoder_diagnosis()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "planned_candidate_count": PLANNED_CANDIDATES,
        "planned_truth_rate_evaluations": PLANNED_CANDIDATES,
        "planned_propagated_states": 0,
        "training_direction_count": TRAINING_DIRECTION_COUNT,
        "tuning_direction_count": TUNING_DIRECTION_COUNT,
        "holdout_direction_count": HOLDOUT_DIRECTION_COUNT,
        "post_result_decoder_diagnosis": True,
        "mixed_direction_results_seen": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25bk",
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(
        ARTIFACT_DIRECTORY / "design_metrics.json",
        {"checks": design_checks, **design_metrics},
    )
    np.savez_compressed(ARTIFACT_DIRECTORY / "mixed_direction_design.npz", **design_arrays)
    _write(ARTIFACT_DIRECTORY / "decoder_diagnosis.json", decoder)
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "package_hashes": parent_data["package_hashes"],
            "decisive_input_hashes": {
                "parity_architecture": _sha(
                    parent.CANONICAL_DIRECTORY / "parity_low_rank_architecture.npz"
                ),
                "online_470_geometry": _sha(GEOMETRY_PATH),
                "low_chart": _sha(LOW_CHART_PATH),
                "high_chart": _sha(HIGH_CHART_PATH),
                "low_rate": _sha(LOW_RATE_PATH),
                "high_rate": _sha(HIGH_RATE_PATH),
            },
        },
    )
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PROSPECTIVE",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "authorized_geometry_runner": NEXT_RUNNER,
            "authorized_geometry_test": NEXT_TEST,
            "prospective_rate_runner": RATE_RUNNER,
            "prospective_rate_test": RATE_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": (
                parent.manifest.rate_0p01.manifest.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT
            ),
        },
    )
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Active-8 mixed parity database manifest WP10c9d6c7c3b5c4f25bj",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "This prospective contract freezes 40 training, 8 tuning, and 8 untouched holdout mixed directions. The tuning directions are evaluated at both 0.005 and 0.01, for 128 signed exact-retraction states and 128 planned nonbase truth-rate evaluations.",
                "",
                f"The quadratic feature condition number is `{design_metrics['quadratic_feature_condition_number']:.6e}`; degree-3 and degree-4 kernel condition numbers are `{design_metrics['cubic_kernel_condition_number']:.6e}` and `{design_metrics['quartic_kernel_condition_number']:.6e}`.",
                "",
                f"Existing axial state data show a cubic odd decoder exponent `{decoder['median_odd_decoder_exponent']:.6f}` and quartic even exponent `{decoder['median_even_decoder_exponent']:.6f}`. This decoder-degree choice is explicitly post-result; the new mixed holdout thresholds are frozen before any mixed result.",
                "",
                "The closure keeps full 28- and 90-dimensional outputs. Its 9,328 stored coefficient upper bound is negligible beside the 470-state linear/stable kernels, avoiding unnecessary output-rank truncation during the first predictive test.",
                "",
                "Only the exact mixed-geometry preflight is authorized next. No rate fit, trajectory, predictive cycle, or reduced slow evolution is yet certified.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("pass --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
