#!/usr/bin/env python3
"""Freeze the full-tensor/rank-4-curvature active-8 database extension."""

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

import run_causal_inner_active8_tensor_architecture_diagnosis_wp10c9d6c7c3b5c4f25bm as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bn"
CLASSIFICATION = (
    "active8_full_cubic_rank4_curvature_database_extension_manifest_"
    "frozen_geometry_authorized"
)
PARENT_COMMIT = "4c61537d72b9ccbbafec10e9899b3788edfc5e77"
PARENT_PARENT = "f1e17183fd09129094e4bb75576903e11eda57ef"
PARENT_TREE = "5b0e6fa40134b53296e6bb8af123be6378d6d61f"

ARTIFACT = (
    "causal_inner_active8_tensor_database_extension_manifest_"
    "wp10c9d6c7c3b5c4f25bn"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_active8_tensor_database_extension_manifest_"
    "wp10c9d6c7c3b5c4f25bn.py"
)
THIS_TEST = (
    "tests/test_causal_inner_active8_tensor_database_extension_manifest_"
    "wp10c9d6c7c3b5c4f25bn.py"
)
GEOMETRY_RUNNER = (
    "scripts/run_causal_inner_active8_tensor_geometry_extension_"
    "wp10c9d6c7c3b5c4f25bo.py"
)
GEOMETRY_TEST = (
    "tests/test_causal_inner_active8_tensor_geometry_extension_"
    "wp10c9d6c7c3b5c4f25bo.py"
)
RATE_RUNNER = (
    "scripts/run_causal_inner_active8_tensor_rate_validation_"
    "wp10c9d6c7c3b5c4f25bp.py"
)
RATE_TEST = (
    "tests/test_causal_inner_active8_tensor_rate_validation_"
    "wp10c9d6c7c3b5c4f25bp.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ACTIVE8_TENSOR_DATABASE_"
    "EXTENSION_MANIFEST_WP10C9D6C7C3B5C4F25BN_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

HIGH_COMPONENT_BOUND = 1.0e-2
LOW_COMPONENT_BOUND = 5.0e-3
REVEALED_TRAINING_DIRECTIONS = 56
ADDITIONAL_TRAINING_DIRECTIONS = 64
TOTAL_TRAINING_DIRECTIONS = 120
NEW_TUNING_DIRECTIONS = 8
NEW_HOLDOUT_DIRECTIONS = 16
NEW_TRAINING_CANDIDATES = 2 * ADDITIONAL_TRAINING_DIRECTIONS
NEW_TUNING_CANDIDATES = 4 * NEW_TUNING_DIRECTIONS
NEW_HOLDOUT_CANDIDATES = 2 * NEW_HOLDOUT_DIRECTIONS
PLANNED_CANDIDATES = (
    NEW_TRAINING_CANDIDATES + NEW_TUNING_CANDIDATES + NEW_HOLDOUT_CANDIDATES
)


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


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("tensor-architecture parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("tensor-architecture parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("tensor-architecture parent tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    architecture = _read(
        parent.CANONICAL_DIRECTORY / "selected_architecture.json"
    )
    if (
        not summary["passed"]
        or summary["classification"] != parent.CLASSIFICATION
        or summary["authorized_next"] != parent.AUTHORIZED_NEXT
        or summary["independent_validation_claimed"]
        or summary["predictive_cycle_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or architecture["online_truth_calls_per_macrostep"] != 0
        or architecture["online_Newton_retractions_per_macrostep"] != 0
    ):
        raise RuntimeError("tensor-architecture authorization changed")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("database-extension manifest requires a clean tree")
    return {"summary": summary, "architecture": architecture, "hashes": hashes}


def _design() -> tuple[dict, dict[str, np.ndarray]]:
    path = parent.CANONICAL_DIRECTORY / "tensor_architecture_design.npz"
    with np.load(path, allow_pickle=False) as source:
        arrays = {
            "revealed_directions_active8": np.asarray(
                source["revealed_directions_active8"], dtype=float
            ),
            "additional_training_directions_active8": np.asarray(
                source["additional_training_directions_active8"], dtype=float
            ),
            "total_training_directions_active8": np.asarray(
                source["total_training_directions_active8"], dtype=float
            ),
            "new_tuning_directions_active8": np.asarray(
                source["new_tuning_directions_active8"], dtype=float
            ),
            "new_holdout_directions_active8": np.asarray(
                source["new_holdout_directions_active8"], dtype=float
            ),
            "total_training_quadratic_features": np.asarray(
                source["total_training_quadratic_features"], dtype=float
            ),
            "total_training_cubic_features": np.asarray(
                source["total_training_cubic_features"], dtype=float
            ),
            "rank4_curvature_basis": np.asarray(
                source["rank4_curvature_basis"], dtype=float
            ),
        }
    training = arrays["total_training_directions_active8"].T
    validation = np.vstack(
        (
            arrays["new_tuning_directions_active8"].T,
            arrays["new_holdout_directions_active8"].T,
        )
    )
    cubic = parent._cubic_features(training)
    quadratic = parent.parent.manifest._quadratic_features(training)
    separation = 1.0 - np.abs(validation @ training.T)
    mutual = 1.0 - np.abs(validation @ validation.T) + np.eye(
        validation.shape[0]
    )
    metrics = {
        "revealed_training_direction_count": int(
            arrays["revealed_directions_active8"].shape[1]
        ),
        "additional_training_direction_count": int(
            arrays["additional_training_directions_active8"].shape[1]
        ),
        "total_training_direction_count": int(training.shape[0]),
        "new_tuning_direction_count": int(
            arrays["new_tuning_directions_active8"].shape[1]
        ),
        "new_holdout_direction_count": int(
            arrays["new_holdout_directions_active8"].shape[1]
        ),
        "quadratic_feature_rank": int(np.linalg.matrix_rank(quadratic)),
        "quadratic_feature_condition_number": float(np.linalg.cond(quadratic)),
        "cubic_feature_rank": int(np.linalg.matrix_rank(cubic)),
        "cubic_feature_condition_number": float(np.linalg.cond(cubic)),
        "minimum_validation_to_training_projective_separation": float(
            np.min(separation)
        ),
        "minimum_validation_mutual_projective_separation": float(
            np.min(mutual)
        ),
        "maximum_absolute_new_direction_component": float(
            np.max(
                np.abs(
                    np.vstack(
                        (
                            arrays[
                                "additional_training_directions_active8"
                            ].T,
                            validation,
                        )
                    )
                )
            )
        ),
        "planned_candidate_count": PLANNED_CANDIDATES,
    }
    return metrics, arrays


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "independently_validate_a_zero_truth_call_active8_vector_field_"
            "using_full_quadratic_cubic_tensors_and_rank4_slaved_curvature"
        ),
        "mathematical_architecture": {
            "online_state": "q162_plus_z280_plus_a28_equals_470",
            "q_update": "exact_conservative_finite_volume",
            "z_update": "inherited_certified_280D_stable_descriptor_kernel",
            "a_update": (
                "inherited_linear_term_plus_full_28_output_homogeneous_"
                "quadratic36_and_cubic120_tensor_closure"
            ),
            "state_reconstruction": (
                "linear_470_lifting_plus_frozen_rank4_curvature_basis_times_"
                "a_full_homogeneous_cubic120_coefficient_map"
            ),
            "curvature_is_algebraic_not_dynamic": True,
            "online_truth_calls_per_macrostep": 0,
            "online_Newton_retractions_per_macrostep": 0,
            "stored_nonlinear_coefficients": 4_848,
        },
        "database_design": {
            "revealed_high_amplitude_directions_reused_as_training": (
                REVEALED_TRAINING_DIRECTIONS
            ),
            "additional_high_amplitude_training_directions": (
                ADDITIONAL_TRAINING_DIRECTIONS
            ),
            "total_high_amplitude_training_directions": (
                TOTAL_TRAINING_DIRECTIONS
            ),
            "new_tuning_directions_at_0p01_and_0p005": NEW_TUNING_DIRECTIONS,
            "new_untouched_holdout_directions_at_0p01": (
                NEW_HOLDOUT_DIRECTIONS
            ),
            "signs": [-1, 1],
            "high_component_bound": HIGH_COMPONENT_BOUND,
            "low_component_bound": LOW_COMPONENT_BOUND,
            "new_training_candidates": NEW_TRAINING_CANDIDATES,
            "new_tuning_candidates": NEW_TUNING_CANDIDATES,
            "new_holdout_candidates": NEW_HOLDOUT_CANDIDATES,
            "planned_total_candidates": PLANNED_CANDIDATES,
            "direction_order": (
                "additional_training_high_then_tuning_high_then_holdout_high_"
                "then_tuning_low;_negative_before_positive"
            ),
        },
        "leakage_control": {
            "old_tuning_and_holdout": "revealed_training_only",
            "geometry_stage_may_test_only_admissibility_not_response_accuracy": True,
            "rate_evaluation_order": [
                "additional_training_high",
                "freeze_and_hash_all_three_coefficient_maps",
                "new_tuning_high_and_low",
                "new_holdout_high",
            ],
            "fit_may_read_new_training_responses_only": True,
            "coefficient_hash_precedes_any_new_validation_response_read": True,
            "tuning_cannot_change_architecture_features_rank_or_gates": True,
            "holdout_cannot_change_any_coefficient_or_threshold": True,
        },
        "coefficient_fit": {
            "rate_quadratic": (
                "least_squares_120_by_36_symmetric_orthonormal_features_to_"
                "28_outputs"
            ),
            "rate_cubic": "solve_120_by_120_cubic_features_to_28_outputs",
            "curvature_cubic": (
                "solve_120_by_120_cubic_features_to_4_frozen_curvature_"
                "coordinates"
            ),
            "regularization": 0.0,
            "online_truth_calls": 0,
            "online_Newton_retractions": 0,
        },
        "design_gates": {
            "quadratic_feature_rank_equal": 36,
            "quadratic_feature_condition_number_max": 5.0,
            "cubic_feature_rank_equal": 120,
            "cubic_feature_condition_number_max": 25.0,
            "minimum_validation_to_training_projective_separation": 0.27,
            "minimum_validation_mutual_projective_separation": 0.27,
            "maximum_absolute_new_direction_component": 0.75,
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
        "binding_radial_consistency_gates": {
            "maximum_quadratic_target_high_low_relative_difference": 0.10,
            "maximum_cubic_rate_target_high_low_relative_difference": 0.15,
            "maximum_curvature_cubic_target_high_low_relative_difference": 0.10,
        },
        "binding_independent_model_gates": {
            "tuning_median_nonlinear_departure_rate_relative_error": 0.10,
            "tuning_maximum_nonlinear_departure_rate_relative_error": 0.25,
            "holdout_median_nonlinear_departure_rate_relative_error": 0.10,
            "holdout_maximum_nonlinear_departure_rate_relative_error": 0.25,
            "tuning_median_full_departure_rate_relative_error": 0.02,
            "tuning_maximum_full_departure_rate_relative_error": 0.05,
            "holdout_median_full_departure_rate_relative_error": 0.02,
            "holdout_maximum_full_departure_rate_relative_error": 0.05,
            "maximum_curvature_prediction_error_over_full_state_delta": 1.5e-3,
            "maximum_full_scaled_state_decoder_relative_error": 2.5e-3,
            "maximum_reconstructed_C_phys_residual_infinity": 2.5e-4,
            "minimum_reconstructed_state_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstructed_H_over_R": 0.12,
            "minimum_reconstructed_scattering_optical_depth": 1.0,
        },
        "decision": {
            "geometry_pass": {
                "classification": "active8_tensor_geometry_extension_passed",
                "authorizes_only": "WP10c9d6c7c3b5c4f25bp",
            },
            "geometry_fail": {
                "classification": "active8_tensor_geometry_extension_failed",
                "authorizes_only": None,
            },
            "model_pass": {
                "classification": (
                    "active8_full_tensor_rate_and_rank4_slaved_curvature_"
                    "independently_validated"
                ),
                "authorizes_only": (
                    "definitions_only_active8_short_reduced_vector_field_"
                    "validation_manifest"
                ),
            },
            "truth_or_model_fail": {
                "classification": (
                    "active8_tensor_database_or_independent_validation_failed"
                ),
                "authorizes_only": None,
            },
        },
        "claim_boundary": {
            "old_kernel_model_remains_rejected": True,
            "architecture_selected_post_old_result": True,
            "all_new_directions_and_thresholds_frozen_before_new_results": True,
            "independent_validation_complete": False,
            "trajectory_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _design_checks(metrics: dict) -> dict:
    gates = _contract()["design_gates"]
    return {
        "revealed_count": metrics["revealed_training_direction_count"]
        == REVEALED_TRAINING_DIRECTIONS,
        "additional_count": metrics["additional_training_direction_count"]
        == ADDITIONAL_TRAINING_DIRECTIONS,
        "total_count": metrics["total_training_direction_count"]
        == TOTAL_TRAINING_DIRECTIONS,
        "tuning_count": metrics["new_tuning_direction_count"]
        == NEW_TUNING_DIRECTIONS,
        "holdout_count": metrics["new_holdout_direction_count"]
        == NEW_HOLDOUT_DIRECTIONS,
        "quadratic_rank": metrics["quadratic_feature_rank"]
        == gates["quadratic_feature_rank_equal"],
        "quadratic_condition": metrics["quadratic_feature_condition_number"]
        <= gates["quadratic_feature_condition_number_max"],
        "cubic_rank": metrics["cubic_feature_rank"]
        == gates["cubic_feature_rank_equal"],
        "cubic_condition": metrics["cubic_feature_condition_number"]
        <= gates["cubic_feature_condition_number_max"],
        "validation_separation": metrics[
            "minimum_validation_to_training_projective_separation"
        ]
        >= gates["minimum_validation_to_training_projective_separation"],
        "validation_mutual_separation": metrics[
            "minimum_validation_mutual_projective_separation"
        ]
        >= gates["minimum_validation_mutual_projective_separation"],
        "component": metrics["maximum_absolute_new_direction_component"]
        <= gates["maximum_absolute_new_direction_component"],
        "candidate_count": metrics["planned_candidate_count"]
        == PLANNED_CANDIDATES,
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
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
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
    _write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("database-extension manifest is already frozen")
    design_metrics, arrays = _design()
    checks = _design_checks(design_metrics)
    if not all(checks.values()):
        raise RuntimeError(f"database-extension design failed: {checks}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "planned_candidate_count": PLANNED_CANDIDATES,
        "planned_truth_rate_evaluation_count": PLANNED_CANDIDATES,
        "total_training_direction_count": TOTAL_TRAINING_DIRECTIONS,
        "new_tuning_direction_count": NEW_TUNING_DIRECTIONS,
        "new_untouched_holdout_direction_count": NEW_HOLDOUT_DIRECTIONS,
        "new_results_seen": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25bo",
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "contract.json", _contract())
    _write_json(
        CANONICAL_DIRECTORY / "design_metrics.json",
        {"checks": checks, **design_metrics},
    )
    with (CANONICAL_DIRECTORY / "extension_design.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "decisive_input_hashes": {
                "selected_architecture": _sha(
                    parent.CANONICAL_DIRECTORY / "selected_architecture.json"
                ),
                "tensor_architecture_design": _sha(
                    parent.CANONICAL_DIRECTORY / "tensor_architecture_design.npz"
                ),
                "old_rejected_closure": _sha(
                    parent.parent.CANONICAL_DIRECTORY / "mixed_parity_closure.npz"
                ),
                "old_exact_geometry": _sha(parent.parent.manifest.GEOMETRY_PATH),
            },
        },
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PROSPECTIVE",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "authorized_geometry_runner": GEOMETRY_RUNNER,
            "authorized_geometry_test": GEOMETRY_TEST,
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
                parent.parent.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT
            ),
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
                "# Active-8 tensor database extension manifest WP10c9d6c7c3b5c4f25bn",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "This prospective package freezes a complete 36-feature quadratic and 120-feature cubic closure for all 28 active departure rates, plus a four-coordinate cubic algebraic curvature decoder. The online dimension remains 470 and both truth calls and Newton retractions remain zero.",
                "",
                f"The frozen design reuses {REVEALED_TRAINING_DIRECTIONS} revealed directions as training, adds {ADDITIONAL_TRAINING_DIRECTIONS} training directions, and reserves {NEW_TUNING_DIRECTIONS} new tuning plus {NEW_HOLDOUT_DIRECTIONS} untouched holdout directions. It contains {PLANNED_CANDIDATES} new signed exact states.",
                "",
                f"The complete cubic feature matrix has rank {design_metrics['cubic_feature_rank']} and condition number {design_metrics['cubic_feature_condition_number']:.6f}. New validation separation from training is {design_metrics['minimum_validation_to_training_projective_separation']:.6f}.",
                "",
                "Leakage is fail-closed: training truth is evaluated first, all coefficient maps are hashed, and only then may tuning and holdout responses be read. The holdout cannot alter coefficients, architecture, or gates.",
                "",
                "Only the exact-geometry extension is authorized next. No rate validation, reduced trajectory, predictive cycle, or reduced slow evolution is authorized by this manifest.",
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
