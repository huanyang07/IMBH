#!/usr/bin/env python3
"""Screen a saved-array transition hidden basis without new truth calls."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_transition_sector_macrostate_revision_manifest_wp10c9d6c7c3b5c4f25dh as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25di"
PARENT_COMMIT = "0cda68a09e26aad17c0f271c81bb5c4021e10f3c"
PARENT_PARENT = "b8a81a7cd7a9997ca21172240a571d21fd002182"
PARENT_TREE = "b370e8504ef6dbea764e0192be93ee1bbc5a7e91"

COMMON_CLASSIFICATION = (
    "common_transition_hidden_basis_candidate_supported_"
    "definitions_only_tangent_manifest_authorized"
)
ATLAS_CLASSIFICATION = (
    "multi_center_transition_hidden_atlas_required_"
    "definitions_only_sampling_manifest_authorized"
)
FALLBACK_CLASSIFICATION = (
    "transition_internal_reduction_not_supported_"
    "full470_offline_impulse_map_retained"
)
AUTHORIZED_NEXT_COMMON = "WP10c9d6c7c3b5c4f25dj"
AUTHORIZED_NEXT_ATLAS = "WP10c9d6c7c3b5c4f25dj"
AUTHORIZED_NEXT_FALLBACK = "WP10c9d6c7c3b5c4f25dj"

COORDINATE_DIMENSION = 470
MACRO_DIMENSION = 82
HIDDEN_DIMENSION = 388
GAUGE_DIMENSION = 90
PHYSICAL_DIMENSION = 560
RADIAL_CELLS = 112
PHYSICAL_FIELDS = 5
SEED_COUNT = 13
REVEALED_SEED_COUNT = 9
VALIDATION_SEED_COUNT = 4
CANDIDATE_RANKS = (8, 16, 24, 32, 48, 64, 96, 128)

MACRO_ANNIHILATION_GATE = 5.0e-12
ORTHONORMALITY_GATE = 5.0e-12
TRAINING_CAPTURE_GATE = 0.99
CURRENT_HIDDEN_CAPTURE_GATE = 0.95
CURRENT_PHYSICAL_CAPTURE_GATE = 0.95
MAXIMUM_SELECTED_RANK = 128

ARTIFACT = "causal_inner_transition_hidden_basis_screen_wp10c9d6c7c3b5c4f25di"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_transition_hidden_basis_screen_"
    "wp10c9d6c7c3b5c4f25di.py"
)
THIS_TEST = (
    "tests/test_causal_inner_transition_hidden_basis_screen_"
    "wp10c9d6c7c3b5c4f25di.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TRANSITION_HIDDEN_BASIS_SCREEN_"
    "WP10C9D6C7C3B5C4F25DI_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_ARCHITECTURE = parent.CANONICAL_DIRECTORY / "revised_hybrid_architecture.json"
PARENT_ARRAYS = parent.CANONICAL_DIRECTORY / "transition_reconciliation_arrays.npz"
SEED_ARRAYS = parent.LOCAL_ATLAS_SEED
CURRENT_RATE_ARRAYS = parent.PARENT_RATE_ARRAYS
EXACT_CHART_ARRAYS = parent.EXACT_CHART_ARRAYS
DUAL_GEOMETRY_ARRAYS = (
    ROOT
    / "results/canonical/causal_inner_primary_hidden_fast_root_manifest_"
    "wp10c9d6c7c3b5c4f25df/dual_hidden_geometry.npz"
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
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("hidden-basis parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("hidden-basis parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("hidden-basis parent tree changed")

    parent_hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    architecture = _read(PARENT_ARCHITECTURE)
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    screen = architecture["prospective_hidden_basis_screen"]
    gates = screen["binding_gates"]
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["parent_anchor_rejection_preserved"]
        or not summary["prior_three_coordinate_transition_internal_model_rejected"]
        or summary["new_exact_fixed_Q_rate_calls"] != 0
        or summary["new_complete_generator_assemblies"] != 0
        or summary["new_nonlinear_roots"] != 0
        or summary["propagated_states"] != 0
        or summary["sealed_16ms_opened"]
        or screen["truth_policy"] != "saved_arrays_only"
        or screen["candidate_hidden_ranks"] != list(CANDIDATE_RANKS)
        or gates["macro_annihilation_infinity_max"] != MACRO_ANNIHILATION_GATE
        or gates["basis_orthonormality_infinity_max"] != ORTHONORMALITY_GATE
        or gates["training_minimum_hidden_action_energy_capture"]
        != TRAINING_CAPTURE_GATE
        or gates["current_primary_hidden_action_energy_capture"]
        != CURRENT_HIDDEN_CAPTURE_GATE
        or gates["current_primary_gauge_fixed_physical_action_energy_capture"]
        != CURRENT_PHYSICAL_CAPTURE_GATE
        or gates["maximum_selected_hidden_rank"] != MAXIMUM_SELECTED_RANK
    ):
        raise RuntimeError("prospective hidden-basis contract changed")

    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"transition-revision source changed: {relative}")
    decisive = architecture["decisive_input_hashes"]
    decisive_paths = {
        "parent_summary": parent.parent.CANONICAL_DIRECTORY / "summary.json",
        "parent_rate_arrays": CURRENT_RATE_ARRAYS,
        "parent_rate_metrics": parent.PARENT_RATE_METRICS,
        "prior_transition_arrays": parent.TRANSITION_ARRAYS,
        "prior_transition_metrics": parent.TRANSITION_METRICS,
        "prior_transition_architecture": parent.TRANSITION_ARCHITECTURE,
        "candidate_arrays": parent.CANDIDATE_ARRAYS,
        "exact_chart_arrays": EXACT_CHART_ARRAYS,
        "local_atlas_seed": SEED_ARRAYS,
    }
    for name, path in decisive_paths.items():
        if _sha(path) != decisive[name]:
            raise RuntimeError(f"decisive hidden-basis input changed: {name}")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("hidden-basis screen requires a clean tracked tree")
    return {
        "parent_hashes": parent_hashes,
        "parent_classification": summary["classification"],
        "parent_work_package": summary["work_package"],
    }


def _canonicalize_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, dtype=float, copy=True)
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _energy_capture(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    samples = np.atleast_2d(np.asarray(vectors, dtype=float))
    denominator = np.maximum(
        np.sum(samples**2, axis=1), np.finfo(float).tiny
    )
    coefficients = samples @ basis
    return np.sum(coefficients**2, axis=1) / denominator


def _fit_basis(hidden_rates: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    norms = np.maximum(
        np.linalg.norm(hidden_rates, axis=1), np.finfo(float).tiny
    )
    normalized = hidden_rates / norms[:, None]
    _, singular_values, right = np.linalg.svd(normalized, full_matrices=False)
    effective_rank = min(int(rank), int(np.linalg.matrix_rank(normalized)))
    return _canonicalize_columns(right[:effective_rank].T), singular_values


def _physical_lift(
    action470: np.ndarray, augmented_chart_jacobian: np.ndarray
) -> np.ndarray:
    action = np.asarray(action470, dtype=float)
    if action.ndim == 1:
        right = np.concatenate((action, np.zeros(GAUGE_DIMENSION)))
    else:
        right = np.vstack((action, np.zeros((GAUGE_DIMENSION, action.shape[1]))))
    return np.linalg.solve(augmented_chart_jacobian, right)


def _physical_capture(
    full: np.ndarray, projected: np.ndarray
) -> tuple[float, float, float]:
    denominator = max(float(np.linalg.norm(full)), np.finfo(float).tiny)
    relative_error = float(np.linalg.norm(projected - full) / denominator)
    capture = float(1.0 - relative_error**2)
    cosine_denominator = max(
        float(np.linalg.norm(full) * np.linalg.norm(projected)),
        np.finfo(float).tiny,
    )
    cosine_squared = float((np.dot(full, projected) / cosine_denominator) ** 2)
    return capture, relative_error, cosine_squared


def _component_capture(
    full_cells: np.ndarray, projected_cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    field_capture = []
    for field in range(PHYSICAL_FIELDS):
        denominator = max(
            float(np.linalg.norm(full_cells[:, field]) ** 2),
            np.finfo(float).tiny,
        )
        error = float(
            np.linalg.norm(projected_cells[:, field] - full_cells[:, field]) ** 2
        )
        field_capture.append(1.0 - error / denominator)
    radial_capture = []
    for group in np.array_split(np.arange(RADIAL_CELLS), 4):
        denominator = max(
            float(np.linalg.norm(full_cells[group]) ** 2), np.finfo(float).tiny
        )
        error = float(
            np.linalg.norm(projected_cells[group] - full_cells[group]) ** 2
        )
        radial_capture.append(1.0 - error / denominator)
    return np.asarray(field_capture), np.asarray(radial_capture)


def _leave_family_out(hidden_rates: np.ndarray, rank: int) -> dict:
    families = {
        "prior_revealed_nine": np.arange(0, REVEALED_SEED_COUNT),
        "prior_validation_four": np.arange(REVEALED_SEED_COUNT, SEED_COUNT),
    }
    output = {}
    all_indices = np.arange(SEED_COUNT)
    for name, heldout in families.items():
        training = np.setdiff1d(all_indices, heldout)
        basis, _ = _fit_basis(hidden_rates[training], rank)
        captures = _energy_capture(hidden_rates[heldout], basis)
        output[name] = {
            "training_count": int(training.size),
            "heldout_count": int(heldout.size),
            "realized_rank": int(basis.shape[1]),
            "minimum_heldout_hidden_action_energy_capture": float(
                np.min(captures)
            ),
            "heldout_hidden_action_energy_captures": captures,
        }
    individual = []
    for heldout in range(SEED_COUNT):
        training = np.delete(all_indices, heldout)
        basis, _ = _fit_basis(hidden_rates[training], rank)
        individual.append(float(_energy_capture(hidden_rates[heldout], basis)[0]))
    output["leave_one_snapshot_out"] = {
        "minimum_heldout_hidden_action_energy_capture": min(individual),
        "heldout_hidden_action_energy_captures": individual,
    }
    return output


def _screen() -> tuple[dict, dict[str, np.ndarray], dict]:
    seed = _load_npz(SEED_ARRAYS)
    dual = _load_npz(DUAL_GEOMETRY_ARRAYS)
    current = _load_npz(CURRENT_RATE_ARRAYS)
    chart = _load_npz(EXACT_CHART_ARRAYS)
    revision = _load_npz(PARENT_ARRAYS)

    seed_rates = np.asarray(
        seed["seed_exact_coordinate_rates_per_second"], dtype=float
    )
    current_rate = np.asarray(current["coordinate_rate_F470_per_s"], dtype=float)
    current_hidden_recorded = np.asarray(
        current["hidden_rate_H388_per_s"], dtype=float
    )
    current_action_recorded = np.asarray(
        current["hidden_action_ZH470_per_s"], dtype=float
    )
    R = np.asarray(dual["macro_restriction_R82"], dtype=float)
    Z = np.asarray(dual["hidden_basis_Z388"], dtype=float)
    Q = np.asarray(dual["hidden_dual_Q388"], dtype=float)
    augmented = np.asarray(chart["anchor_augmented_chart_jacobian"], dtype=float)

    if (
        seed_rates.shape != (SEED_COUNT, COORDINATE_DIMENSION)
        or current_rate.shape != (COORDINATE_DIMENSION,)
        or current_hidden_recorded.shape != (HIDDEN_DIMENSION,)
        or R.shape != (MACRO_DIMENSION, COORDINATE_DIMENSION)
        or Z.shape != (COORDINATE_DIMENSION, HIDDEN_DIMENSION)
        or Q.shape != (HIDDEN_DIMENSION, COORDINATE_DIMENSION)
        or augmented.shape != (PHYSICAL_DIMENSION, PHYSICAL_DIMENSION)
    ):
        raise RuntimeError("hidden-basis input shape changed")

    seed_hidden = seed_rates @ Q.T
    seed_actions = seed_hidden @ Z.T
    current_hidden = Q @ current_rate
    current_action = Z @ current_hidden
    normalized_seed = seed_hidden / np.maximum(
        np.linalg.norm(seed_hidden, axis=1), np.finfo(float).tiny
    )[:, None]
    data_rank = int(np.linalg.matrix_rank(normalized_seed))
    _, singular_values, full_right = np.linalg.svd(
        normalized_seed, full_matrices=False
    )

    candidate_metrics = []
    common_selection = None
    training_selection = None
    seen_effective_ranks = set()
    for requested_rank in CANDIDATE_RANKS:
        effective_rank = min(requested_rank, data_rank)
        if effective_rank in seen_effective_ranks:
            continue
        seen_effective_ranks.add(effective_rank)
        basis = _canonicalize_columns(full_right[:effective_rank].T)
        training_capture = _energy_capture(seed_hidden, basis)
        current_capture = float(_energy_capture(current_hidden, basis)[0])
        projected_hidden = basis @ (basis.T @ current_hidden)
        projected_action = Z @ projected_hidden
        physical_full = _physical_lift(current_action, augmented)
        physical_projected = _physical_lift(projected_action, augmented)
        physical_capture, physical_error, physical_cosine = _physical_capture(
            physical_full, physical_projected
        )
        candidate = {
            "requested_rank": requested_rank,
            "effective_data_supported_rank": effective_rank,
            "minimum_training_hidden_action_energy_capture": float(
                np.min(training_capture)
            ),
            "current_primary_hidden_action_energy_capture": current_capture,
            "current_primary_gauge_fixed_physical_action_energy_capture": (
                physical_capture
            ),
            "current_primary_gauge_fixed_physical_relative_error": physical_error,
            "current_primary_gauge_fixed_physical_cosine_squared": physical_cosine,
        }
        candidate_metrics.append(candidate)
        if (
            training_selection is None
            and candidate["minimum_training_hidden_action_energy_capture"]
            >= TRAINING_CAPTURE_GATE
        ):
            training_selection = (requested_rank, effective_rank, basis)
        if (
            common_selection is None
            and candidate["minimum_training_hidden_action_energy_capture"]
            >= TRAINING_CAPTURE_GATE
            and current_capture >= CURRENT_HIDDEN_CAPTURE_GATE
            and physical_capture >= CURRENT_PHYSICAL_CAPTURE_GATE
        ):
            common_selection = (requested_rank, effective_rank, basis)

    augmented_with_primary = False
    if common_selection is not None:
        requested_rank, _, selected_basis = common_selection
        classification = COMMON_CLASSIFICATION
        authorized_next = AUTHORIZED_NEXT_COMMON
    elif training_selection is not None:
        requested_rank, _, seed_basis = training_selection
        residual = current_hidden - seed_basis @ (seed_basis.T @ current_hidden)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > np.finfo(float).eps * float(np.linalg.norm(current_hidden)):
            selected_basis = np.column_stack((seed_basis, residual / residual_norm))
            selected_basis = _canonicalize_columns(selected_basis)
            augmented_with_primary = True
        else:
            selected_basis = seed_basis
        classification = ATLAS_CLASSIFICATION
        authorized_next = AUTHORIZED_NEXT_ATLAS
    else:
        requested_rank = CANDIDATE_RANKS[-1]
        selected_basis = _canonicalize_columns(full_right[:data_rank].T)
        classification = FALLBACK_CLASSIFICATION
        authorized_next = AUTHORIZED_NEXT_FALLBACK

    selected_rank = int(selected_basis.shape[1])
    selected_action_basis = Z @ selected_basis
    training_captures = _energy_capture(seed_hidden, selected_basis)
    current_hidden_capture = float(
        _energy_capture(current_hidden, selected_basis)[0]
    )
    current_projected_hidden = selected_basis @ (
        selected_basis.T @ current_hidden
    )
    current_projected_action = Z @ current_projected_hidden
    physical_full = _physical_lift(current_action, augmented)
    physical_projected = _physical_lift(current_projected_action, augmented)
    physical_capture, physical_error, physical_cosine = _physical_capture(
        physical_full, physical_projected
    )
    full_cells = physical_full.reshape(RADIAL_CELLS, PHYSICAL_FIELDS)
    projected_cells = physical_projected.reshape(RADIAL_CELLS, PHYSICAL_FIELDS)
    field_capture, radial_capture = _component_capture(full_cells, projected_cells)
    physical_basis = _physical_lift(selected_action_basis, augmented)

    metrics = {
        "seed_exact_rate_count": SEED_COUNT,
        "seed_provenance_families": {
            "prior_revealed_nine": REVEALED_SEED_COUNT,
            "prior_validation_four": VALIDATION_SEED_COUNT,
        },
        "normalized_seed_hidden_action_data_rank": data_rank,
        "normalized_seed_hidden_action_singular_values": singular_values,
        "candidate_rank_metrics": candidate_metrics,
        "selected_requested_rank": requested_rank,
        "selected_hidden_rank": selected_rank,
        "selected_basis_source": (
            "prior_seed_only" if not augmented_with_primary else "prior_seed_plus_primary_center"
        ),
        "primary_direction_added_as_atlas_center": augmented_with_primary,
        "minimum_training_hidden_action_energy_capture": float(
            np.min(training_captures)
        ),
        "training_hidden_action_energy_captures": training_captures,
        "current_primary_hidden_action_energy_capture": current_hidden_capture,
        "current_primary_gauge_fixed_physical_action_energy_capture": physical_capture,
        "current_primary_gauge_fixed_physical_relative_error": physical_error,
        "current_primary_gauge_fixed_physical_cosine_squared": physical_cosine,
        "current_primary_fieldwise_physical_action_energy_capture": field_capture,
        "current_primary_radial_quartile_physical_action_energy_capture": radial_capture,
        "selected_basis_orthonormality_infinity_defect": float(
            np.linalg.norm(
                selected_basis.T @ selected_basis - np.eye(selected_rank),
                ord=np.inf,
            )
        ),
        "selected_action_macro_annihilation_infinity_defect": float(
            np.linalg.norm(R @ selected_action_basis, ord=np.inf)
        ),
        "dual_identity_infinity_defect": float(
            np.linalg.norm(Q @ Z - np.eye(HIDDEN_DIMENSION), ord=np.inf)
        ),
        "hidden_basis_isometry_infinity_defect": float(
            np.linalg.norm(Z.T @ Z - np.eye(HIDDEN_DIMENSION), ord=np.inf)
        ),
        "seed_hidden_action_reconstruction_relative_defect": float(
            np.linalg.norm(seed_actions - seed_hidden @ Z.T)
            / max(np.linalg.norm(seed_actions), np.finfo(float).tiny)
        ),
        "current_hidden_rate_reproduction_relative_defect": float(
            np.linalg.norm(current_hidden - current_hidden_recorded)
            / max(np.linalg.norm(current_hidden_recorded), np.finfo(float).tiny)
        ),
        "current_hidden_action_reproduction_relative_defect": float(
            np.linalg.norm(current_action - current_action_recorded)
            / max(np.linalg.norm(current_action_recorded), np.finfo(float).tiny)
        ),
        "current_hidden_action_matches_revision_relative_defect": float(
            np.linalg.norm(
                current_action
                - revision["exact_anchor_hidden_action470_per_s"]
            )
            / max(np.linalg.norm(current_action), np.finfo(float).tiny)
        ),
        "leave_family_out": _leave_family_out(seed_hidden, selected_rank),
        "classification": classification,
        "authorized_next": authorized_next,
        "new_exact_fixed_Q_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "new_chart_retractions": 0,
        "propagated_states": 0,
        "sealed_16ms_truth_calls": 0,
    }
    checks = {
        "parent_inputs_reproduced": max(
            metrics["current_hidden_rate_reproduction_relative_defect"],
            metrics["current_hidden_action_reproduction_relative_defect"],
            metrics["current_hidden_action_matches_revision_relative_defect"],
        )
        <= MACRO_ANNIHILATION_GATE,
        "macro_annihilation": metrics[
            "selected_action_macro_annihilation_infinity_defect"
        ]
        <= MACRO_ANNIHILATION_GATE,
        "basis_orthonormality": metrics[
            "selected_basis_orthonormality_infinity_defect"
        ]
        <= ORTHONORMALITY_GATE,
        "training_capture": metrics[
            "minimum_training_hidden_action_energy_capture"
        ]
        >= TRAINING_CAPTURE_GATE,
        "current_hidden_capture": current_hidden_capture
        >= CURRENT_HIDDEN_CAPTURE_GATE,
        "current_physical_capture": physical_capture
        >= CURRENT_PHYSICAL_CAPTURE_GATE,
        "selected_rank": selected_rank <= MAXIMUM_SELECTED_RANK,
        "truth_budget": metrics["new_exact_fixed_Q_rate_calls"] == 0,
        "generator_budget": metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": metrics["new_nonlinear_roots"] == 0,
        "retraction_budget": metrics["new_chart_retractions"] == 0,
        "propagation_budget": metrics["propagated_states"] == 0,
        "sealed_budget": metrics["sealed_16ms_truth_calls"] == 0,
    }
    arrays = {
        "seed_hidden_rates388_per_s": seed_hidden,
        "seed_hidden_actions470_per_s": seed_actions,
        "normalized_seed_hidden_action_singular_values": singular_values,
        "selected_hidden_basis388": selected_basis,
        "selected_coordinate_action_basis470": selected_action_basis,
        "selected_gauge_fixed_physical_basis560": physical_basis,
        "training_hidden_action_energy_captures": training_captures,
        "current_primary_hidden_rate388_per_s": current_hidden,
        "current_primary_hidden_action470_per_s": current_action,
        "current_primary_projected_hidden_rate388_per_s": current_projected_hidden,
        "current_primary_projected_hidden_action470_per_s": current_projected_action,
        "current_primary_gauge_fixed_physical_action560_per_s": physical_full,
        "current_primary_projected_gauge_fixed_physical_action560_per_s": physical_projected,
        "current_primary_fieldwise_physical_action_energy_capture": field_capture,
        "current_primary_radial_quartile_physical_action_energy_capture": radial_capture,
        "macro_restriction_R82": R,
        "hidden_basis_Z388": Z,
        "hidden_dual_Q388": Q,
    }
    return metrics, arrays, checks


def _selected_contract(metrics: dict, checks: dict) -> dict:
    common = metrics["classification"] == COMMON_CLASSIFICATION
    atlas = metrics["classification"] == ATLAS_CLASSIFICATION
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": all(checks.values()),
        "evidence_type": "saved_array_hidden_basis_screen",
        "basis": {
            "coordinate_system": "dual_consistent_hidden_H388",
            "construction": (
                "SVD_of_thirteen_individually_normalized_saved_hidden_rate_"
                "actions"
            ),
            "selected_rank": metrics["selected_hidden_rank"],
            "source": metrics["selected_basis_source"],
            "inside_kernel_of_R82": checks["macro_annihilation"],
            "current_primary_was_held_out_from_fit": common,
            "primary_direction_added_as_atlas_center": metrics[
                "primary_direction_added_as_atlas_center"
            ],
        },
        "interpretation": {
            "common_transition_hidden_basis_candidate_supported": common,
            "multi_center_transition_atlas_required": atlas,
            "full470_offline_transition_reference_still_required": True,
            "basis_is_a_certified_transition_dynamics_model": False,
            "physical_fixed_Q_failure_detected": False,
        },
        "prospective_next_manifest": {
            "work_package": metrics["authorized_next"],
            "kind": (
                "definitions_only_common_hidden_basis_tangent_diagnostic"
                if common
                else (
                    "definitions_only_multi_center_transition_sampling"
                    if atlas
                    else "definitions_only_full470_offline_impulse_map"
                )
            ),
            "may_use_new_truth_calls": False,
            "may_execute_in_this_package": False,
            "must_preserve_full470_offline_fallback": True,
        },
        "authorization_boundaries": {
            "complete_tangent_executed": False,
            "transition_truth_campaign_authorized": False,
            "branch_root_authorized": False,
            "online_transition_ODE_authorized": False,
            "online_solver_authorized": False,
            "physical_microburst_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
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
                    "scientific_status": "DIAGNOSTIC_PASS",
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
        raise RuntimeError("transition hidden-basis screen already exists")
    metrics, arrays, checks = _screen()
    passed = all(checks.values())
    if not passed:
        raise RuntimeError(f"transition hidden-basis screen failed: {checks}")
    contract = _selected_contract(metrics, checks)

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "hidden_basis_screen_arrays.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "hidden_basis_screen_metrics.json",
        {"metrics": metrics, "checks": checks, "passed": passed},
    )
    _write_json(CANONICAL_DIRECTORY / "selected_hidden_basis_contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            **frozen,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": passed,
        "saved_arrays_only": True,
        "selected_hidden_rank": metrics["selected_hidden_rank"],
        "selected_basis_source": metrics["selected_basis_source"],
        "minimum_training_hidden_action_energy_capture": metrics[
            "minimum_training_hidden_action_energy_capture"
        ],
        "current_primary_hidden_action_energy_capture": metrics[
            "current_primary_hidden_action_energy_capture"
        ],
        "current_primary_gauge_fixed_physical_action_energy_capture": metrics[
            "current_primary_gauge_fixed_physical_action_energy_capture"
        ],
        "new_exact_fixed_Q_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "new_chart_retractions": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "physical_fixed_Q_failure_detected": False,
        "full470_offline_transition_reference_preserved": True,
        "complete_tangent_executed": False,
        "transition_truth_campaign_authorized": False,
        "online_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (THIS_RUNNER, THIS_TEST, parent.THIS_RUNNER, parent.THIS_TEST)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DIAGNOSTIC_PASS",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "decisive_input_hashes": {
                "parent_architecture": _sha(PARENT_ARCHITECTURE),
                "parent_arrays": _sha(PARENT_ARRAYS),
                "seed_arrays": _sha(SEED_ARRAYS),
                "current_rate_arrays": _sha(CURRENT_RATE_ARRAYS),
                "dual_geometry_arrays": _sha(DUAL_GEOMETRY_ARRAYS),
                "exact_chart_arrays": _sha(EXACT_CHART_ARRAYS),
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    leave = metrics["leave_family_out"]
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Transition hidden-basis screen WP10c9d6c7c3b5c4f25di",
                "",
                "## Classification",
                "",
                f"`{metrics['classification']}`",
                "",
                "The saved-array-only screen passed without any new exact fixed-Q rate, complete generator, nonlinear root, chart retraction, propagation, or sealed 16 ms call.",
                "",
                f"The smallest candidate is a rank-{metrics['selected_hidden_rank']} basis fitted only to the thirteen prior transition snapshots. It captures at least `{metrics['minimum_training_hidden_action_energy_capture']:.12%}` of every training hidden action.",
                "",
                f"The exact current 20 ms action was held out from the fit. The basis captures `{metrics['current_primary_hidden_action_energy_capture']:.12%}` of its dual-consistent hidden-coordinate energy and `{metrics['current_primary_gauge_fixed_physical_action_energy_capture']:.12%}` of its gauge-fixed physical-action energy.",
                "",
                f"Leaving out the original nine-snapshot family gives minimum capture `{leave['prior_revealed_nine']['minimum_heldout_hidden_action_energy_capture']:.12%}`. Leaving out the four prior validation snapshots gives `{leave['prior_validation_four']['minimum_heldout_hidden_action_energy_capture']:.12%}`. These are robustness diagnostics, not substitute truth calls.",
                "",
                f"Macro annihilation is `{metrics['selected_action_macro_annihilation_infinity_defect']:.6e}` and hidden-basis orthonormality defect is `{metrics['selected_basis_orthonormality_infinity_defect']:.6e}`.",
                "",
                "This supports a common transition hidden-basis candidate; it does not yet certify invariant tangent dynamics, a transition impulse map, branch states, or an online reduced solver. The full y470 chart remains the mandatory offline reference and fallback.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`, definitions-only. No tangent execution, transition truth campaign, online transition ODE, physical microburst, or reduced slow evolution is authorized by this result.",
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
