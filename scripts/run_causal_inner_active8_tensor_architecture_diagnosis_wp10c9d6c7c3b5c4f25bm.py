#!/usr/bin/env python3
"""Diagnose the failed active-8 kernel model and select its replacement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import qr


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_active8_mixed_parity_rate_fit_wp10c9d6c7c3b5c4f25bl as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bm"
PARENT_COMMIT = "cd67f25ee2983bf94d70837f324f4e850fb96367"
PARENT_PARENT = "8426540a70f21871ae62d8eb0c4b07e749d6480b"
PARENT_TREE = "6311366b3bd54ef3ca1e4625802f4304a6c99812"

CLASSIFICATION = (
    "active8_kernel_failure_diagnosed_full_tensor_rate_and_rank4_"
    "slaved_curvature_architecture_selected"
)
AUTHORIZED_NEXT = (
    "definitions_only_active8_full_cubic_rank4_curvature_"
    "database_extension_manifest"
)
ARTIFACT = (
    "causal_inner_active8_tensor_architecture_diagnosis_"
    "wp10c9d6c7c3b5c4f25bm"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_active8_tensor_architecture_diagnosis_"
    "wp10c9d6c7c3b5c4f25bm.py"
)
THIS_TEST = (
    "tests/test_causal_inner_active8_tensor_architecture_diagnosis_"
    "wp10c9d6c7c3b5c4f25bm.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ACTIVE8_TENSOR_ARCHITECTURE_"
    "DIAGNOSIS_WP10C9D6C7C3B5C4F25BM_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

DESIGN_SEED = 2503
DESIGN_POOL_SIZE = 50_000
EXISTING_REVEALED_DIRECTION_COUNT = 56
ADDITIONAL_TRAINING_DIRECTION_COUNT = 64
TOTAL_TRAINING_DIRECTION_COUNT = 120
NEW_TUNING_DIRECTION_COUNT = 8
NEW_HOLDOUT_DIRECTION_COUNT = 16
CURVATURE_RANK = 4


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
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


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


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


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


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("active-8 rejection commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("active-8 rejection lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("active-8 rejection tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != parent.MODEL_FAIL_CLASSIFICATION
        or summary["passed"]
        or not summary["truth_database_passed"]
        or summary["local_closure_validated"]
        or summary["authorized_next"] is not None
        or not all(metrics["truth_checks"].values())
        or all(metrics["model_checks"].values())
    ):
        raise RuntimeError("active-8 rejected-model classification changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"active-8 source changed: {relative}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("tensor architecture diagnosis requires a clean tree")
    for name, expected in parent.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _cubic_features(directions: np.ndarray) -> np.ndarray:
    """Orthonormal symmetric-tensor features satisfying phi(x).phi(y)=(x.y)^3."""

    directions = np.asarray(directions, dtype=float)
    columns = [directions[:, index] ** 3 for index in range(8)]
    columns.extend(
        np.sqrt(3.0) * directions[:, repeated] ** 2 * directions[:, single]
        for repeated in range(8)
        for single in range(8)
        if repeated != single
    )
    columns.extend(
        np.sqrt(6.0)
        * directions[:, first]
        * directions[:, second]
        * directions[:, third]
        for first in range(8)
        for second in range(first + 1, 8)
        for third in range(second + 1, 8)
    )
    return np.asarray(columns, dtype=float).T


def _existing_revealed_directions() -> np.ndarray:
    design = _load_npz(parent.manifest.ARTIFACT_DIRECTORY / "mixed_direction_design.npz")
    return np.vstack(
        (
            design["training_directions_active8"].T,
            design["tuning_directions_active8"].T,
            design["holdout_directions_active8"].T,
        )
    )


def _candidate_pool() -> np.ndarray:
    rng = np.random.default_rng(DESIGN_SEED)
    pool = rng.normal(size=(DESIGN_POOL_SIZE, 8))
    pool /= np.linalg.norm(pool, axis=1)[:, None]
    pivots = np.argmax(np.abs(pool), axis=1)
    pool *= np.sign(pool[np.arange(pool.shape[0]), pivots])[:, None]
    pool = pool[np.max(np.abs(pool), axis=1) <= 0.75]
    return np.unique(np.round(pool, decimals=12), axis=0)


def _extended_direction_design() -> tuple[dict, dict[str, np.ndarray]]:
    revealed = _existing_revealed_directions()
    pool = _candidate_pool()
    revealed_features = _cubic_features(revealed)
    row_basis = np.linalg.qr(revealed_features.T, mode="reduced")[0]
    residual = _cubic_features(pool)
    residual -= (residual @ row_basis) @ row_basis.T
    _q, _r, pivots = qr(residual.T, pivoting=True, mode="economic")
    additional = pool[pivots[:ADDITIONAL_TRAINING_DIRECTION_COUNT]]
    training = np.vstack((revealed, additional))

    distance = np.min(1.0 - np.abs(pool @ training.T), axis=1)
    validation_indices = []
    selection_distances = []
    validation_count = NEW_TUNING_DIRECTION_COUNT + NEW_HOLDOUT_DIRECTION_COUNT
    for _index in range(validation_count):
        selected = int(np.argmax(distance))
        validation_indices.append(selected)
        selection_distances.append(float(distance[selected]))
        distance = np.minimum(distance, 1.0 - np.abs(pool @ pool[selected]))
        distance[selected] = -1.0
    validation = pool[validation_indices]
    tuning = validation[:NEW_TUNING_DIRECTION_COUNT]
    holdout = validation[NEW_TUNING_DIRECTION_COUNT:]
    cubic = _cubic_features(training)
    quadratic = parent.manifest._quadratic_features(training)
    metrics = {
        "design_seed": DESIGN_SEED,
        "raw_pool_size": DESIGN_POOL_SIZE,
        "filtered_pool_size": int(pool.shape[0]),
        "revealed_direction_count": int(revealed.shape[0]),
        "additional_training_direction_count": int(additional.shape[0]),
        "total_training_direction_count": int(training.shape[0]),
        "new_tuning_direction_count": int(tuning.shape[0]),
        "new_holdout_direction_count": int(holdout.shape[0]),
        "quadratic_feature_rank": int(np.linalg.matrix_rank(quadratic)),
        "quadratic_feature_condition_number": float(np.linalg.cond(quadratic)),
        "cubic_feature_rank": int(np.linalg.matrix_rank(cubic)),
        "cubic_feature_condition_number": float(np.linalg.cond(cubic)),
        "minimum_new_validation_to_training_projective_separation": float(
            np.min(1.0 - np.abs(validation @ training.T))
        ),
        "minimum_new_validation_mutual_projective_separation": float(
            np.min(
                (1.0 - np.abs(validation @ validation.T))
                + np.eye(validation.shape[0])
            )
        ),
        "maximum_absolute_new_direction_component": float(
            np.max(np.abs(np.vstack((additional, validation))))
        ),
        "validation_selection_distances": selection_distances,
        "new_candidate_count": int(
            2 * additional.shape[0]
            + 4 * tuning.shape[0]
            + 2 * holdout.shape[0]
        ),
    }
    arrays = {
        "revealed_directions_active8": revealed.T,
        "additional_training_directions_active8": additional.T,
        "total_training_directions_active8": training.T,
        "new_tuning_directions_active8": tuning.T,
        "new_holdout_directions_active8": holdout.T,
        "total_training_quadratic_features": quadratic,
        "total_training_cubic_features": cubic,
    }
    return metrics, arrays


def _relative(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(reference))
        / max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    )


def _split_ranges() -> tuple[tuple[str, int, int], ...]:
    return (
        ("revealed_training", 0, 80),
        ("revealed_tuning_high", 80, 96),
        ("revealed_holdout", 96, 112),
        ("revealed_tuning_low", 112, 128),
    )


def _capacity_analysis() -> tuple[dict, dict[str, np.ndarray]]:
    closure = _load_npz(parent.CANONICAL_DIRECTORY / "mixed_parity_closure.npz")
    geometry = _load_npz(parent.manifest.GEOMETRY_PATH)
    generator = _load_npz(parent.GENERATOR_PATH)["complete_fixed_Q_generator"]
    deltas = closure["candidate_scaled_deltas"]
    hidden_basis = geometry["hidden_stable_remainder_basis"]
    hidden = deltas @ hidden_basis
    _left, singular_values, right = np.linalg.svd(hidden[:80], full_matrices=False)
    curvature_coordinates = hidden @ right[:CURVATURE_RANK].T
    curvature_basis = hidden_basis @ right[:CURVATURE_RANK].T

    rank_metrics = {}
    for rank in (0, 1, 2, 4, 6, 8, 12):
        if rank:
            projected = (hidden @ right[:rank].T) @ right[:rank]
        else:
            projected = np.zeros_like(hidden)
        error = np.linalg.norm(hidden - projected, axis=1) / np.linalg.norm(
            deltas, axis=1
        )
        rank_metrics[str(rank)] = {
            name: {
                "median_full_scaled_state_relative_error": float(
                    np.median(error[start:stop])
                ),
                "maximum_full_scaled_state_relative_error": float(
                    np.max(error[start:stop])
                ),
            }
            for name, start, stop in _split_ranges()
        }

    odd_prediction = np.zeros_like(hidden)
    for pair in range(64):
        negative = 2 * pair
        positive = negative + 1
        odd = 0.5 * (hidden[positive] - hidden[negative])
        projected_odd = (odd @ right[:CURVATURE_RANK].T) @ right[:CURVATURE_RANK]
        odd_prediction[negative] = -projected_odd
        odd_prediction[positive] = projected_odd
    odd_error = np.linalg.norm(hidden - odd_prediction, axis=1) / np.linalg.norm(
        deltas, axis=1
    )
    odd_metrics = {
        name: {
            "median_full_scaled_state_relative_error": float(
                np.median(odd_error[start:stop])
            ),
            "maximum_full_scaled_state_relative_error": float(
                np.max(odd_error[start:stop])
            ),
        }
        for name, start, stop in _split_ranges()
    }

    stable_basis = geometry["stable_memory_coordinate_basis"]
    augmented = np.column_stack((stable_basis, curvature_basis))
    augmented_operator = augmented.T @ generator @ augmented
    augmented_eigenvalues = np.linalg.eigvals(augmented_operator)
    curvature_operator = curvature_basis.T @ generator @ curvature_basis

    radial_metrics = {}
    for name in (
        "rate_quadratic_targets",
        "rate_cubic_targets",
        "decoder_cubic_targets",
        "decoder_quartic_targets",
    ):
        values = closure[name]
        defects = np.asarray(
            [_relative(values[40 + index], values[56 + index]) for index in range(8)]
        )
        radial_metrics[name] = {
            "median_high_low_normalized_target_difference": float(
                np.median(defects)
            ),
            "maximum_high_low_normalized_target_difference": float(
                np.max(defects)
            ),
            "directionwise_high_low_normalized_target_difference": defects,
        }

    target_energy = {}
    for name in (
        "rate_quadratic_targets",
        "rate_cubic_targets",
        "decoder_cubic_targets",
        "decoder_quartic_targets",
    ):
        singular = np.linalg.svd(closure[name][:40], compute_uv=False)
        energy = np.cumsum(singular * singular) / np.sum(singular * singular)
        target_energy[name] = {
            f"rank_{rank}_energy_fraction": float(energy[rank - 1])
            for rank in (1, 2, 4, 6, 8)
        }

    metrics = {
        "post_result_diagnosis": True,
        "independent_validation_claimed": False,
        "all_prior_tuning_and_holdout_directions_now_revealed": True,
        "training_hidden_singular_energy_rank4": float(
            np.sum(singular_values[:4] ** 2) / np.sum(singular_values**2)
        ),
        "rank_capacity": rank_metrics,
        "rank4_odd_only_capacity": odd_metrics,
        "rank4_curvature_basis_orthogonality_defect": float(
            np.linalg.norm(
                curvature_basis.T @ curvature_basis - np.eye(CURVATURE_RANK),
                ord=np.inf,
            )
        ),
        "dynamic_augmentation_spectral_abscissa_per_second": float(
            np.max(augmented_eigenvalues.real)
        ),
        "dynamic_augmentation_unstable_eigenvalue_count": int(
            np.count_nonzero(augmented_eigenvalues.real >= 0.0)
        ),
        "curvature_self_block_spectral_abscissa_per_second": float(
            np.max(np.linalg.eigvals(curvature_operator).real)
        ),
        "radial_consistency": radial_metrics,
        "training_target_output_energy": target_energy,
    }
    arrays = {
        "training_hidden_singular_values": singular_values,
        "training_hidden_right_singular_vectors": right,
        "rank4_curvature_basis": curvature_basis,
        "candidate_rank4_curvature_coordinates": curvature_coordinates,
        "candidate_rank4_odd_only_hidden_predictions": odd_prediction,
        "dynamic_augmented_operator_per_second": augmented_operator,
        "dynamic_augmented_eigenvalues_per_second": augmented_eigenvalues,
    }
    return metrics, arrays


def _capacity_state_audit(
    capacity_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    closure = _load_npz(parent.CANONICAL_DIRECTORY / "mixed_parity_closure.npz")
    geometry = _load_npz(parent.manifest.GEOMETRY_PATH)
    components = parent.parent.high_chart._prepare_components()
    deltas = closure["candidate_scaled_deltas"]
    online = (geometry["online_coordinate_lifting"] @ (
        geometry["online_coordinate_restriction"] @ deltas.T
    )).T
    hidden_prediction = capacity_arrays[
        "candidate_rank4_odd_only_hidden_predictions"
    ]
    predicted = online + hidden_prediction @ geometry[
        "hidden_stable_remainder_basis"
    ].T
    records = []
    reconstructed_states = []
    began = time.perf_counter()
    for index in range(80, 128):
        state = components["state"] + (
            components["columns"].ravel() * predicted[index]
        ).reshape(components["state"].shape)
        coordinate, coordinate_factors = (
            parent.parent.chart_tools._coordinate_value_with_factors(
                state, components
            )
        )
        physical = parent.parent.chart_tools._state_audit(
            components["context"], state
        )
        if index < 96:
            split = "revealed_tuning_high"
        elif index < 112:
            split = "revealed_holdout"
        else:
            split = "revealed_tuning_low"
        records.append(
            {
                "candidate_index": index,
                "split": split,
                "full_scaled_state_relative_error": _relative(
                    predicted[index], deltas[index]
                ),
                "C_phys_residual_infinity": float(
                    np.max(
                        np.abs(
                            coordinate - components["coordinate_target"]
                        )
                    )
                ),
                "minimum_reconstruction_factor": min(
                    float(np.min(coordinate_factors)),
                    physical["minimum_reconstruction_factor"],
                ),
                "maximum_H_over_R": physical["maximum_h_over_r"],
                "minimum_scattering_optical_depth": physical[
                    "minimum_scattering_optical_depth"
                ],
            }
        )
        reconstructed_states.append(state)
        print(
            json.dumps(
                {
                    "capacity_audit_candidate": index + 1,
                    "total": 128,
                    "split": split,
                    "elapsed_seconds": time.perf_counter() - began,
                }
            ),
            flush=True,
        )
    metrics = {
        "evaluated_candidate_count": len(records),
        "maximum_full_scaled_state_relative_error": float(
            max(item["full_scaled_state_relative_error"] for item in records)
        ),
        "maximum_C_phys_residual_infinity": float(
            max(item["C_phys_residual_infinity"] for item in records)
        ),
        "minimum_reconstruction_factor": float(
            min(item["minimum_reconstruction_factor"] for item in records)
        ),
        "maximum_H_over_R": float(
            max(item["maximum_H_over_R"] for item in records)
        ),
        "minimum_scattering_optical_depth": float(
            min(item["minimum_scattering_optical_depth"] for item in records)
        ),
        "records": records,
        "wall_seconds": time.perf_counter() - began,
    }
    arrays = {
        "rank4_odd_only_predicted_scaled_deltas": predicted,
        "rank4_odd_only_reconstructed_validation_states": np.asarray(
            reconstructed_states
        ),
    }
    return metrics, arrays


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
                    "scientific_status": "CERTIFIED",
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
        raise RuntimeError("tensor architecture diagnosis already canonicalized")
    design_metrics, design_arrays = _extended_direction_design()
    capacity_metrics, capacity_arrays = _capacity_analysis()
    state_metrics, state_arrays = _capacity_state_audit(capacity_arrays)
    checks = {
        "parent_truth_database_passed": frozen["summary"][
            "truth_database_passed"
        ],
        "parent_model_rejection_preserved": not frozen["summary"]["passed"],
        "rank4_hidden_energy": capacity_metrics[
            "training_hidden_singular_energy_rank4"
        ]
        >= 0.99,
        "rank4_odd_capacity_state_error": state_metrics[
            "maximum_full_scaled_state_relative_error"
        ]
        <= 1.0e-3,
        "rank4_odd_capacity_C_phys": state_metrics[
            "maximum_C_phys_residual_infinity"
        ]
        <= 2.5e-4,
        "rank4_odd_capacity_reconstruction": state_metrics[
            "minimum_reconstruction_factor"
        ]
        >= 1.0 - 1.0e-12,
        "naive_dynamic_augmentation_rejected": capacity_metrics[
            "dynamic_augmentation_unstable_eigenvalue_count"
        ]
        >= 1,
        "quadratic_feature_rank": design_metrics["quadratic_feature_rank"]
        == 36,
        "cubic_feature_rank": design_metrics["cubic_feature_rank"] == 120,
        "cubic_feature_condition": design_metrics[
            "cubic_feature_condition_number"
        ]
        <= 25.0,
        "new_validation_separation": design_metrics[
            "minimum_new_validation_to_training_projective_separation"
        ]
        >= 0.27,
        "truth_call_budget": True,
        "root_budget": True,
        "propagation_budget": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"tensor architecture diagnosis failed: {checks}")

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    metrics = {
        "checks": checks,
        "parent_failed_model_metrics": frozen["metrics"]["model"],
        "design": design_metrics,
        "capacity": capacity_metrics,
        "capacity_state_audit": state_metrics,
        "new_nonbase_continuous_rate_evaluations": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
    }
    _write_json(CANONICAL_DIRECTORY / "metrics.json", metrics)
    _write_npz(
        CANONICAL_DIRECTORY / "tensor_architecture_design.npz",
        {**design_arrays, **capacity_arrays, **state_arrays},
    )
    contract = {
        "selected_online_state": "q162_plus_z280_plus_a28_equals_470",
        "stable_memory": "retain_inherited_certified_280D_descriptor_kernel",
        "unstable_departure": "retain_all_28_explicit_nonlinear_coordinates",
        "state_decoder": (
            "rank4_slaved_curvature_basis_with_full_homogeneous_cubic_"
            "active8_coefficient_map; omit_revealed_small_even_and_"
            "remaining_hidden_terms_as_validation_error"
        ),
        "rate_closure": (
            "full_28_output_homogeneous_quadratic_36_feature_plus_"
            "homogeneous_cubic_120_feature_tensor"
        ),
        "online_truth_calls_per_macrostep": 0,
        "online_Newton_retractions_per_macrostep": 0,
        "coefficient_count": {
            "rate_quadratic": 36 * 28,
            "rate_cubic": 120 * 28,
            "rank4_curvature_cubic": 120 * CURVATURE_RANK,
            "total": 36 * 28 + 120 * 28 + 120 * CURVATURE_RANK,
        },
        "new_database": {
            "reuse_revealed_high_amplitude_directions_as_training": 56,
            "additional_high_amplitude_training_directions": 64,
            "total_high_amplitude_training_directions": 120,
            "new_tuning_directions_at_0p01_and_0p005": 8,
            "new_untouched_holdout_directions_at_0p01": 16,
            "new_signed_exact_geometry_and_rate_candidates": design_metrics[
                "new_candidate_count"
            ],
        },
        "claim_boundary": {
            "post_result_architecture_diagnosis": True,
            "old_tuning_and_holdout_are_revealed_training_only": True,
            "new_tuning_and_holdout_must_remain_untouched_until_fit_is_frozen": True,
            "no_predictive_trajectory_authorized": True,
        },
    }
    _write_json(CANONICAL_DIRECTORY / "selected_architecture.json", contract)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "parent_model_rejection_preserved": True,
        "selected_online_state_dimension": 470,
        "selected_curvature_decoder_rank": CURVATURE_RANK,
        "selected_rate_quadratic_feature_dimension": 36,
        "selected_rate_cubic_feature_dimension": 120,
        "new_candidate_count": design_metrics["new_candidate_count"],
        "independent_validation_claimed": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
        },
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED_POST_RESULT_DIAGNOSIS",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": parent.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT,
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
                "# Active-8 tensor architecture diagnosis WP10c9d6c7c3b5c4f25bm",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The 40-center quadratic/cubic plus 90-output cubic/quartic kernel model remains rejected. All prior tuning and holdout data are now revealed and are used only for post-result diagnosis.",
                "",
                f"A train-only rank-{CURVATURE_RANK} odd curvature subspace captures `{capacity_metrics['training_hidden_singular_energy_rank4']:.8f}` of hidden snapshot energy. Its revealed validation capacity has maximum full-state error `{state_metrics['maximum_full_scaled_state_relative_error']:.6e}` and C_phys residual `{state_metrics['maximum_C_phys_residual_infinity']:.6e}`.",
                "",
                f"Naively evolving these curvature modes is rejected: the augmented memory projection has `{capacity_metrics['dynamic_augmentation_unstable_eigenvalue_count']}` unstable eigenvalues and spectral abscissa `{capacity_metrics['dynamic_augmentation_spectral_abscissa_per_second']:.6e} s^-1`. They must remain an algebraic/slaved decoder.",
                "",
                f"The replacement closure uses complete homogeneous quadratic/cubic tensors of dimensions 36/120. Retaining 56 revealed directions and adding 64 new training directions gives cubic rank `{design_metrics['cubic_feature_rank']}` and condition `{design_metrics['cubic_feature_condition_number']:.6f}`. Fresh validation remains separated by `{design_metrics['minimum_new_validation_to_training_projective_separation']:.6f}`.",
                "",
                f"The next definitions-only extension contains `{design_metrics['new_candidate_count']}` new signed exact-geometry/rate candidates. No predictive trajectory, cycle, or reduced slow evolution is authorized.",
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
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
