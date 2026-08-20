#!/usr/bin/env python3
"""Freeze an authentic-center local field and exact chart-overlap contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_direct_coordinate_field_manifest_wp10c9d6c7c3b5c4f25cn as direct_manifest  # noqa: E402
import run_causal_inner_direct_coordinate_field_validation_wp10c9d6c7c3b5c4f25co as direct_validation  # noqa: E402
import run_causal_inner_recenter_transition_validation_wp10c9d6c7c3b5c4f25cq as parent  # noqa: E402
import run_causal_inner_shell_gated_atlas_rate_validation_wp10c9d6c7c3b5c4f25ck as mixed_validation  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cr"
PARENT_COMMIT = "88c23290d83130177cb6982a07dc8a678a1e05cc"
PARENT_PARENT = "a0314ad01e98a722474fed3ec07f17ca17b06520"
PARENT_TREE = "4132f7c9079d8379c7c305af4d1325374965e642"
CLASSIFICATION = "authentic_center_local_field_overlap_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25cs"

PHYSICAL_DIMENSION = 162
MEMORY_DIMENSION = 280
DEPARTURE_DIMENSION = 28
ONLINE_DIMENSION = 470
TRAINING_DIRECTION_COUNT = 4
HOLDOUT_DIRECTION_COUNT = 4
TRAINING_COMPONENT_BOUND = 1.25e-2
HOLDOUT_COMPONENT_BOUND = 1.5e-2
TRANSVERSE_MIXING = 0.35
HOLDOUT_AXIS_MIXING = 0.50
OVERLAP_ACTIVATION_LOAD = 1.0e-2
OVERLAP_FULL_NEW_LOAD = 1.3e-2
OLD_HARD_LOAD = 1.5e-2
NEW_HARD_LOAD = 1.5e-2
AFFINE_RIDGE = 1.0e-8

ARTIFACT = (
    "causal_inner_authentic_center_local_field_overlap_manifest_"
    "wp10c9d6c7c3b5c4f25cr"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_authentic_center_local_field_overlap_manifest_"
    "wp10c9d6c7c3b5c4f25cr.py"
)
THIS_TEST = (
    "tests/test_causal_inner_authentic_center_local_field_overlap_manifest_"
    "wp10c9d6c7c3b5c4f25cr.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_AUTHENTIC_CENTER_LOCAL_FIELD_"
    "OVERLAP_MANIFEST_WP10C9D6C7C3B5C4F25CR_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

TRANSITION_ARRAYS = parent.CANONICAL_DIRECTORY / "transition_arrays.npz"
DIRECT_FIELD = direct_manifest.CANONICAL_DIRECTORY / "direct_coordinate_field.npz"
MIXED_RATE_ARRAYS = mixed_validation.CANONICAL_DIRECTORY / "rate_arrays.npz"
REVEALED_HOLDOUT_RATE_ARRAYS = (
    direct_validation.CANONICAL_DIRECTORY / "rate_arrays.npz"
)

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


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(float).tiny:
        raise RuntimeError("center-local direction vanished")
    return value / norm


def _transverse_axes(forward: np.ndarray, count: int = 2) -> np.ndarray:
    """Choose deterministic well-conditioned axes transverse to forward."""

    axis = _normalize(forward)
    selected: list[np.ndarray] = []
    coordinate_axes = np.eye(axis.size)
    while len(selected) < count:
        best = None
        best_norm = -1.0
        for candidate in coordinate_axes:
            residual = candidate - float(candidate @ axis) * axis
            for prior in selected:
                residual -= float(residual @ prior) * prior
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm > best_norm + 1.0e-15:
                best = residual
                best_norm = residual_norm
        if best is None or best_norm <= 1.0e-12:
            raise RuntimeError("center-local transverse design lost rank")
        selected.append(best / best_norm)
    return np.asarray(selected, dtype=float)


def _direction_design(forward: np.ndarray) -> dict[str, np.ndarray]:
    v = _normalize(forward)
    t1, t2 = _transverse_axes(v, 2)
    signs = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    training = np.asarray(
        [
            _normalize(v + TRANSVERSE_MIXING * (s1 * t1 + s2 * t2))
            for s1, s2 in signs
        ]
    )
    holdout = np.asarray(
        [
            _normalize(v + HOLDOUT_AXIS_MIXING * t1),
            _normalize(v - HOLDOUT_AXIS_MIXING * t1),
            _normalize(v + HOLDOUT_AXIS_MIXING * t2),
            _normalize(v - HOLDOUT_AXIS_MIXING * t2),
        ]
    )
    active_basis = np.column_stack((v, t1, t2))
    return {
        "forward_direction": v,
        "transverse_axes": np.asarray((t1, t2)),
        "active_departure_basis": active_basis,
        "training_directions": training,
        "holdout_directions": holdout,
        "training_component_bounds": np.full(
            TRAINING_DIRECTION_COUNT, TRAINING_COMPONENT_BOUND
        ),
        "holdout_component_bounds": np.full(
            HOLDOUT_DIRECTION_COUNT, HOLDOUT_COMPONENT_BOUND
        ),
    }


def _smoothstep_weight(old_load: float) -> float:
    value = float(old_load)
    if value <= OVERLAP_ACTIVATION_LOAD:
        return 0.0
    if value >= OVERLAP_FULL_NEW_LOAD:
        return 1.0
    t = (value - OVERLAP_ACTIVATION_LOAD) / (
        OVERLAP_FULL_NEW_LOAD - OVERLAP_ACTIVATION_LOAD
    )
    return float(t**3 * (10.0 - 15.0 * t + 6.0 * t**2))


def _affine_features(active_coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(active_coordinates, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return np.column_stack((np.ones(values.shape[0]), values))


def _fit_affine_residual(
    active_coordinates: np.ndarray,
    residuals: np.ndarray,
    regularization: float = AFFINE_RIDGE,
) -> np.ndarray:
    design = _affine_features(active_coordinates)
    penalty = float(regularization) * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ residuals)


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("authentic transition commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("authentic transition lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("authentic transition tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "validation_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    transition = _load_npz(TRANSITION_ARRAYS)
    if (
        not summary["passed"]
        or summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["authentic_center_established"]
        or summary["accepted_truth_roots"] != 2
        or summary["authorized_next"]
        != "definitions_only_authentic_center_local_field_and_overlap_manifest"
        or summary["physical_microburst_authorized"]
        or summary["predictive_cycle_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or not metrics["passed"]
        or not all(metrics["checks"].values())
        or transition["authentic_center_primitive_state"].shape != (112, 5)
        or transition["authentic_center_scaled_delta"].shape != (560,)
        or transition["authentic_center_old_coordinate"].shape != (470,)
    ):
        raise RuntimeError("authentic center authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"authentic transition source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("authentic-center manifest requires a clean tracked tree")
    return {
        "summary": summary,
        "metrics": metrics,
        "hashes": hashes,
        "transition": transition,
    }


def _revealed_overlap_database() -> dict[str, np.ndarray]:
    mixed = _load_npz(MIXED_RATE_ARRAYS)
    holdout = _load_npz(REVEALED_HOLDOUT_RATE_ARRAYS)
    return {
        "coordinates": np.vstack(
            (mixed["online_coordinates"], holdout["online_coordinates"])
        ),
        "scaled_deltas": np.vstack(
            (mixed["candidate_scaled_deltas"], holdout["candidate_scaled_deltas"])
        ),
        "exact_full_rates": np.vstack(
            (mixed["total_rates_per_second"], holdout["total_rates_per_second"])
        ),
        "exact_coordinate_rates": np.vstack(
            (
                mixed["exact_online_470_coordinate_rates_per_second"],
                holdout["exact_online_470_coordinate_rates_per_second"],
            )
        ),
        "old_predicted_full_rates": np.vstack(
            (
                mixed["predicted_full_state_rates_per_second"],
                holdout["predicted_full_state_rates_per_second"],
            )
        ),
        "old_predicted_coordinate_rates": np.vstack(
            (
                direct_manifest._load_npz(DIRECT_FIELD)[
                    "training_direct_predicted_online_rates_per_second"
                ][-8:],
                holdout["predicted_online_470_coordinate_rates_per_second"],
            )
        ),
        "source_group": np.concatenate(
            (np.zeros(8, dtype=int), np.ones(8, dtype=int))
        ),
    }


def _translation_defect(points: np.ndarray, center: np.ndarray) -> float:
    restored = (np.asarray(points) - np.asarray(center)) + np.asarray(center)
    return float(np.max(np.abs(restored - np.asarray(points))))


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "mathematical_architecture": {
            "absolute_online_state": "y470_equals_q162_z280_a28_relative_to_original_authentic_anchor",
            "local_patch_state": "eta_i_equals_y_minus_y_center_i",
            "chart_transition": "exact_affine_translation_with_identity_tangent",
            "fixed_coordinates": {
                "q162": "exact_physical_coordinate_difference",
                "z280": "fixed_memory_basis_projection_of_scaled_state_difference",
                "a28": "fixed_departure_basis_projection_of_scaled_state_difference",
            },
            "decoder_patch": (
                "exact_center_plus_translated_compensated_decoder_difference_"
                "plus_center_anchored_low_rank_affine_geometry_residual"
            ),
            "full_physical_rate_patch": (
                "old_independently_validated_full_rate_plus_low_rank_affine_"
                "residual_in_three_center_local_departure_coordinates"
            ),
            "q162_rate_patch": (
                "fixed_authentic_center_physical_Jacobian_times_corrected_full_"
                "rate_plus_low_rank_affine_q_residual"
            ),
            "z280_rate_patch": "fixed_memory_basis_transpose_times_corrected_full_rate",
            "a28_rate_patch": "fixed_departure_basis_transpose_times_corrected_full_rate",
            "online_state_dependent_coordinate_Jacobian_calls": 0,
            "new_complete_generator_assemblies": 0,
        },
        "overlap": {
            "old_weight_zero_through_load": OVERLAP_ACTIVATION_LOAD,
            "new_weight_one_from_load": OVERLAP_FULL_NEW_LOAD,
            "old_hard_load": OLD_HARD_LOAD,
            "new_hard_load": NEW_HARD_LOAD,
            "weight": "C2_quintic_smoothstep_of_old_decoder_load",
            "single_absolute_state_prevents_coordinate_jump": True,
            "decoder_and_field_are_blended_as_global_atlas_maps": True,
            "hysteretic_primary_patch_selection": True,
        },
        "local_residual_fit": {
            "active_dimension": 3,
            "features": "constant_plus_three_scaled_active_departure_coordinates",
            "ridge": AFFINE_RIDGE,
            "recycled_revealed_exact_overlap_samples": 16,
            "center_exact_rate_samples": 1,
            "new_forward_training_rate_samples": TRAINING_DIRECTION_COUNT,
            "new_forward_holdout_rate_samples": HOLDOUT_DIRECTION_COUNT,
            "coefficients_frozen_before_holdout_truth": True,
        },
        "prospective_geometry": {
            "training_direction_count": TRAINING_DIRECTION_COUNT,
            "training_component_bound": TRAINING_COMPONENT_BOUND,
            "holdout_direction_count": HOLDOUT_DIRECTION_COUNT,
            "holdout_component_bound": HOLDOUT_COMPONENT_BOUND,
            "all_directions_are_forward_sector": True,
            "fixed_q162_center_target": True,
            "geometry_before_rate_truth": True,
        },
        "binding_manifest_gates": {
            "center_coordinate_infinity_defect_max": 1.0e-13,
            "center_scaled_delta_infinity_defect_max": 1.0e-14,
            "translation_roundtrip_infinity_defect_max": 1.0e-14,
            "center_old_load_below": OLD_HARD_LOAD,
            "warm5_new_scaled_load_below": 6.0e-3,
            "revealed_overlap_sample_count_equal": 16,
            "revealed_overlap_maximum_new_scaled_load": 1.2e-2,
            "center_physical_coordinate_Jacobian_rank_equal": PHYSICAL_DIMENSION,
            "center_full_restriction_rank_equal": ONLINE_DIMENSION,
            "center_full_restriction_condition_number_max": 5.0e3,
            "minimum_forward_direction_cosine": 0.85,
            "maximum_holdout_to_training_absolute_cosine": 0.95,
            "active_basis_orthogonality_defect_max": 1.0e-12,
        },
        "next_geometry_preflight_budget": {
            "candidate_count": TRAINING_DIRECTION_COUNT + HOLDOUT_DIRECTION_COUNT,
            "new_continuous_rate_calls": 0,
            "new_complete_generator_assemblies": 0,
            "new_nonlinear_fixed_Q_roots": 0,
            "propagated_states": 0,
        },
        "prospective_total_rate_budget_after_geometry_pass": {
            "center_exact_continuous_rate_calls": 1,
            "new_training_exact_continuous_rate_calls": TRAINING_DIRECTION_COUNT,
            "new_holdout_exact_continuous_rate_calls": HOLDOUT_DIRECTION_COUNT,
            "total_new_exact_continuous_rate_calls": 9,
            "new_complete_generator_assemblies": 0,
            "authentic_BDF_roots": 0,
        },
        "decision": {
            "pass_classification": CLASSIFICATION,
            "pass_authorizes_only": AUTHORIZED_NEXT,
            "fail_classification": "authentic_center_local_field_overlap_manifest_failed",
            "fail_authorizes_only": "definitions_only_local_field_architecture_revision",
        },
        "authorization_boundaries": {
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "fast_average_authorized": False,
            "reduced_slow_evolution_authorized": False,
            "repeated_authentic_recenter_roots_required_by_architecture": False,
        },
    }


def _design(frozen: dict) -> tuple[dict[str, np.ndarray], dict]:
    transition = frozen["transition"]
    center_state = np.asarray(
        transition["authentic_center_primitive_state"], dtype=float
    )
    center_delta = np.asarray(
        transition["authentic_center_scaled_delta"], dtype=float
    )
    center_coordinate = np.asarray(
        transition["authentic_center_old_coordinate"], dtype=float
    )
    closure = _load_npz(DIRECT_FIELD)
    direct = direct_manifest.DirectCoordinateField(closure)
    model = direct.model
    recomputed_coordinate, factors = model.coordinate(center_state)
    recomputed_delta = (
        (center_state - model.base_state) / model.columns
    ).ravel()
    physical_jacobian, physical_metrics = (
        parent.manifest.warm4.manifest.parent.geometry.chart_tools._coordinate_jacobian(
            center_state, model.components
        )
    )
    center_restriction = np.vstack(
        (physical_jacobian, model.memory_basis.T, model.departure_basis.T)
    )
    singular = np.linalg.svd(center_restriction, compute_uv=False)
    overlap = _revealed_overlap_database()
    local_overlap_coordinates = overlap["coordinates"] - center_coordinate
    local_overlap_deltas = overlap["scaled_deltas"] - center_delta
    warm5_coordinate = np.asarray(
        transition["warm_5_truth_coordinate"], dtype=float
    )
    warm5_delta = np.asarray(
        transition["warm_5_truth_scaled_delta"], dtype=float
    )
    warm6_a = center_coordinate[-DEPARTURE_DIMENSION:]
    warm5_a = warm5_coordinate[-DEPARTURE_DIMENSION:]
    direction_design = _direction_design(warm6_a - warm5_a)
    train_holdout_cosines = np.abs(
        direction_design["holdout_directions"]
        @ direction_design["training_directions"].T
    )
    active_gram = (
        direction_design["active_departure_basis"].T
        @ direction_design["active_departure_basis"]
    )
    translation_points = np.vstack(
        (overlap["coordinates"], warm5_coordinate, center_coordinate)
    )
    metrics = {
        "center_coordinate_infinity_defect": float(
            np.max(np.abs(recomputed_coordinate - center_coordinate))
        ),
        "center_scaled_delta_infinity_defect": float(
            np.max(np.abs(recomputed_delta - center_delta))
        ),
        "translation_roundtrip_infinity_defect": _translation_defect(
            translation_points, center_coordinate
        ),
        "center_old_scaled_state_load": float(np.max(np.abs(center_delta))),
        "center_old_decoder_load": float(
            np.max(np.abs(direct.decoded_delta(center_coordinate)))
        ),
        "warm5_new_scaled_state_load": float(
            np.max(np.abs(warm5_delta - center_delta))
        ),
        "revealed_overlap_sample_count": int(overlap["coordinates"].shape[0]),
        "revealed_overlap_maximum_new_scaled_load": float(
            np.max(np.abs(local_overlap_deltas))
        ),
        "revealed_overlap_maximum_local_coordinate_load": float(
            np.max(np.abs(local_overlap_coordinates))
        ),
        "center_physical_coordinate_Jacobian_rank": int(
            physical_metrics["rank"]
        ),
        "center_physical_coordinate_Jacobian_condition_number": float(
            physical_metrics["condition_number"]
        ),
        "center_full_restriction_rank": int(
            np.linalg.matrix_rank(center_restriction)
        ),
        "center_full_restriction_condition_number": float(
            singular[0] / singular[-1]
        ),
        "center_minimum_coordinate_reconstruction_factor": float(
            np.min(factors)
        ),
        "minimum_training_forward_cosine": float(
            np.min(
                direction_design["training_directions"]
                @ direction_design["forward_direction"]
            )
        ),
        "minimum_holdout_forward_cosine": float(
            np.min(
                direction_design["holdout_directions"]
                @ direction_design["forward_direction"]
            )
        ),
        "maximum_holdout_to_training_absolute_cosine": float(
            np.max(train_holdout_cosines)
        ),
        "active_basis_orthogonality_defect": float(
            np.max(np.abs(active_gram - np.eye(3)))
        ),
        "new_continuous_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
    }
    arrays = {
        "authentic_center_primitive_state": center_state,
        "authentic_center_scaled_delta": center_delta,
        "authentic_center_absolute_coordinate": center_coordinate,
        "authentic_center_physical_coordinate_Jacobian": physical_jacobian,
        "authentic_center_fixed_restriction": center_restriction,
        "warm5_absolute_coordinate": warm5_coordinate,
        "warm5_local_coordinate": warm5_coordinate - center_coordinate,
        "warm5_local_scaled_delta": warm5_delta - center_delta,
        "revealed_overlap_absolute_coordinates": overlap["coordinates"],
        "revealed_overlap_local_coordinates": local_overlap_coordinates,
        "revealed_overlap_absolute_scaled_deltas": overlap["scaled_deltas"],
        "revealed_overlap_local_scaled_deltas": local_overlap_deltas,
        "revealed_overlap_exact_full_rates_per_second": overlap[
            "exact_full_rates"
        ],
        "revealed_overlap_exact_coordinate_rates_per_second": overlap[
            "exact_coordinate_rates"
        ],
        "revealed_overlap_old_predicted_full_rates_per_second": overlap[
            "old_predicted_full_rates"
        ],
        "revealed_overlap_old_predicted_coordinate_rates_per_second": overlap[
            "old_predicted_coordinate_rates"
        ],
        "revealed_overlap_source_group": overlap["source_group"],
        **direction_design,
    }
    return arrays, metrics


def _checks(metrics: dict, gates: dict) -> dict:
    return {
        "center_coordinate": metrics["center_coordinate_infinity_defect"]
        <= gates["center_coordinate_infinity_defect_max"],
        "center_scaled_delta": metrics["center_scaled_delta_infinity_defect"]
        <= gates["center_scaled_delta_infinity_defect_max"],
        "translation": metrics["translation_roundtrip_infinity_defect"]
        <= gates["translation_roundtrip_infinity_defect_max"],
        "center_inside_old_hard_limit": metrics["center_old_scaled_state_load"]
        < gates["center_old_load_below"],
        "warm5_inside_new_overlap": metrics["warm5_new_scaled_state_load"]
        < gates["warm5_new_scaled_load_below"],
        "overlap_count": metrics["revealed_overlap_sample_count"]
        == gates["revealed_overlap_sample_count_equal"],
        "overlap_load": metrics["revealed_overlap_maximum_new_scaled_load"]
        <= gates["revealed_overlap_maximum_new_scaled_load"],
        "physical_rank": metrics[
            "center_physical_coordinate_Jacobian_rank"
        ] == gates["center_physical_coordinate_Jacobian_rank_equal"],
        "full_rank": metrics["center_full_restriction_rank"]
        == gates["center_full_restriction_rank_equal"],
        "full_condition": metrics[
            "center_full_restriction_condition_number"
        ] <= gates["center_full_restriction_condition_number_max"],
        "training_forward_sector": metrics["minimum_training_forward_cosine"]
        >= gates["minimum_forward_direction_cosine"],
        "holdout_forward_sector": metrics["minimum_holdout_forward_cosine"]
        >= gates["minimum_forward_direction_cosine"],
        "holdout_separation": metrics[
            "maximum_holdout_to_training_absolute_cosine"
        ] <= gates["maximum_holdout_to_training_absolute_cosine"],
        "active_basis": metrics["active_basis_orthogonality_defect"]
        <= gates["active_basis_orthogonality_defect_max"],
        "rate_budget": metrics["new_continuous_rate_calls"] == 0,
        "generator_budget": metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": metrics["new_nonlinear_roots"] == 0,
        "propagation_budget": metrics["propagated_states"] == 0,
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
        raise RuntimeError("authentic-center local-field manifest already exists")
    contract = _contract()
    arrays, metrics = _design(frozen)
    checks = _checks(metrics, contract["binding_manifest_gates"])
    if not all(checks.values()):
        raise RuntimeError(f"authentic-center local-field design failed: {checks}")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "center_local_field_design.npz", **arrays)
    _write_json(
        CANONICAL_DIRECTORY / "design_metrics.json",
        {"checks": checks, "passed": True, **metrics},
    )
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "transition_arrays_sha256": _sha(TRANSITION_ARRAYS),
            "direct_field_sha256": _sha(DIRECT_FIELD),
            "mixed_rate_arrays_sha256": _sha(MIXED_RATE_ARRAYS),
            "revealed_holdout_rate_arrays_sha256": _sha(
                REVEALED_HOLDOUT_RATE_ARRAYS
            ),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "authentic_center_used": True,
        "exact_affine_chart_translation": True,
        "revealed_overlap_sample_count": metrics[
            "revealed_overlap_sample_count"
        ],
        "prospective_geometry_candidate_count": (
            TRAINING_DIRECTION_COUNT + HOLDOUT_DIRECTION_COUNT
        ),
        "prospective_total_new_exact_rate_calls": 9,
        "online_state_dependent_coordinate_Jacobian_calls": 0,
        "new_truth_rate_calls": 0,
        "new_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "repeated_authentic_recenter_roots_required": False,
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
        direct_manifest.THIS_RUNNER,
        direct_manifest.THIS_TEST,
        direct_validation.THIS_RUNNER,
        direct_validation.THIS_TEST,
        mixed_validation.THIS_RUNNER,
        mixed_validation.THIS_TEST,
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
                "# Authentic-center local-field overlap manifest WP10c9d6c7c3b5c4f25cr",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The authentic warm-6 state is the second atlas center. One global 470D absolute coordinate is retained; every local chart is the exact affine translation `eta = y - y_center`, so switching charts changes neither the represented state nor the coordinate tangent.",
                "",
                "The new patch uses the already validated direct field as a prior and fits only a three-coordinate affine residual in the authentic forward departure sector. The physical rate, q162 rate, and decoder corrections are separated; z280 and a28 remain fixed projections of the corrected physical rate. No online state-dependent coordinate Jacobian and no new complete generator are permitted.",
                "",
                f"Sixteen already revealed exact-rate states lie in the old/new overlap with maximum center-relative scaled load `{metrics['revealed_overlap_maximum_new_scaled_load']:.6e}`. They become fit data, never fresh validation. Eight new geometry states are frozen prospectively: four training states at `{TRAINING_COMPONENT_BOUND:.4f}` and four independent forward-sector holdouts at `{HOLDOUT_COMPONENT_BOUND:.4f}`.",
                "",
                "The next package is geometry-only. Even after a geometry pass, the complete prospective rate budget is nine exact continuous-rate calls: one authentic center, four training, and four coefficient-blind holdouts. No authentic BDF root is part of the local-field construction.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`. No physical microburst, predictive cycle, fast average, or reduced slow evolution is authorized.",
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
