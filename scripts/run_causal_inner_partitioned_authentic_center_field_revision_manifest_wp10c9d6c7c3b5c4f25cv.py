#!/usr/bin/env python3
"""Freeze a partitioned authentic-center field after the pooled fit failed."""

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

import run_causal_inner_authentic_center_exact_rate_training_wp10c9d6c7c3b5c4f25cu as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cv"
PARENT_COMMIT = "1cd3398bd1fb6338156cefac0f7792e0071ad244"
PARENT_PARENT = "1668fc55f70fd35852c8884f388a2859aec10090"
PARENT_TREE = "fdf870c46a4126287ecb39ccc679c5c8171584eb"

CLASSIFICATION = "partitioned_authentic_center_tangent_field_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25cw"

CENTER_CORE_FULL_RADIUS = 2.5e-3
CENTER_CORE_ZERO_RADIUS = 1.8e-2
FORWARD_ZERO_COORDINATE = 4.5e-1
FORWARD_FULL_COORDINATE = 1.25

REVEALED_COUNT = 16
NEW_EXACT_COUNT = 5
TRAINING_COUNT = 4
HOLDOUT_COUNT = 4
PHYSICAL_DIMENSION = 162
MEMORY_DIMENSION = 280
DEPARTURE_DIMENSION = 28

ARTIFACT = (
    "causal_inner_partitioned_authentic_center_field_revision_manifest_"
    "wp10c9d6c7c3b5c4f25cv"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_partitioned_authentic_center_field_revision_manifest_"
    "wp10c9d6c7c3b5c4f25cv.py"
)
THIS_TEST = (
    "tests/test_causal_inner_partitioned_authentic_center_field_revision_manifest_"
    "wp10c9d6c7c3b5c4f25cv.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PARTITIONED_AUTHENTIC_CENTER_"
    "FIELD_REVISION_MANIFEST_WP10C9D6C7C3B5C4F25CV_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_TRUTH = parent.CANONICAL_DIRECTORY / "training_truth_arrays.npz"
PARENT_FAILED_FIELD = (
    parent.CANONICAL_DIRECTORY / "authentic_center_local_field.npz"
)
FROZEN_DESIGN = parent.manifest.CANONICAL_DIRECTORY / "frozen_rate_training_design.npz"
OVERLAP_DESIGN = parent.manifest.DESIGN_ARRAYS
DIRECT_FIELD = parent.DIRECT_FIELD
OLD_GEOMETRY = parent.direct_manifest.ONLINE_GEOMETRY

direct_manifest = parent.direct_manifest
vector_field = parent.vector_field

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


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _relative_rows(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    return np.linalg.norm(left - right, axis=1) / np.maximum(
        np.linalg.norm(right, axis=1), np.finfo(float).tiny
    )


def _smoothstep(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    t = np.clip((values - float(lower)) / (float(upper) - float(lower)), 0.0, 1.0)
    return np.clip(t**3 * (10.0 - 15.0 * t + 6.0 * t**2), 0.0, 1.0)


def _partition_weights(
    local_coordinates: np.ndarray, active_basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local = np.asarray(local_coordinates, dtype=float)
    if local.ndim == 1:
        local = local.reshape(1, -1)
    active = parent.manifest._active_coordinates(local, active_basis)
    radius = np.linalg.norm(local, axis=1)
    center = 1.0 - _smoothstep(
        radius, CENTER_CORE_FULL_RADIUS, CENTER_CORE_ZERO_RADIUS
    )
    forward = _smoothstep(
        active[:, 0], FORWARD_ZERO_COORDINATE, FORWARD_FULL_COORDINATE
    )
    combined = 1.0 - (1.0 - center) * (1.0 - forward)
    return combined, center, forward, active


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("failed pooled-field certificate commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("failed pooled-field certificate lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("failed pooled-field certificate tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "training_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    truth = _load_npz(PARENT_TRUTH)
    failed = _load_npz(PARENT_FAILED_FIELD)
    design = _load_npz(FROZEN_DESIGN)
    overlap = _load_npz(OVERLAP_DESIGN)
    if (
        summary["passed"]
        or summary["classification"] != parent.FAIL_CLASSIFICATION
        or not summary["exact_truth_passed"]
        or summary["local_field_training_passed"]
        or summary["completed_exact_rate_calls"] != NEW_EXACT_COUNT
        or summary["failed_exact_rate_calls"] != 0
        or summary["holdout_rate_calls"] != 0
        or summary["authorized_next"] != parent.FAIL_AUTHORIZED_NEXT
        or not metrics["truth_passed"]
        or metrics["field_passed"]
        or metrics["field_checks"]["training_q162"]
        or truth["total_rates_per_second"].shape != (NEW_EXACT_COUNT, 560)
        or truth["exact_coordinate_rates_per_second"].shape
        != (NEW_EXACT_COUNT, 470)
        or truth["coordinate_jacobians"].shape
        != (NEW_EXACT_COUNT, PHYSICAL_DIMENSION, 560)
        or failed["fit_local_coordinates"].shape
        != (REVEALED_COUNT + NEW_EXACT_COUNT, 470)
        or design["holdout_primitive_states"].shape != (HOLDOUT_COUNT, 112, 5)
        or overlap["revealed_overlap_old_predicted_coordinate_rates_per_second"].shape
        != (REVEALED_COUNT, 470)
    ):
        raise RuntimeError("partitioned-field revision authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"failed pooled-field source changed: {relative}")
    if (
        _sha(PARENT_TRUTH) != hashes["training_truth_arrays.npz"]
        or _sha(PARENT_FAILED_FIELD) != hashes["authentic_center_local_field.npz"]
        or _sha(FROZEN_DESIGN)
        != parent.manifest._checksums(parent.manifest.CANONICAL_DIRECTORY)[
            "frozen_rate_training_design.npz"
        ]
    ):
        raise RuntimeError("partitioned-field revision input changed")
    for name, expected in parent._thread_environment().items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("partitioned-field revision requires a clean tracked tree")
    return {
        "summary": summary,
        "metrics": metrics,
        "hashes": hashes,
        "truth": truth,
        "failed": failed,
        "design": design,
        "overlap": overlap,
    }


def _old_coordinate_field(
    full_rates: np.ndarray,
    absolute_coordinates: np.ndarray,
    old_restriction: np.ndarray,
    direct_closure: dict[str, np.ndarray],
) -> np.ndarray:
    result = np.asarray(full_rates, dtype=float) @ np.asarray(
        old_restriction, dtype=float
    ).T
    result[:, :PHYSICAL_DIMENSION] += np.asarray(
        [
            direct_manifest._q_correction(
                coordinate[-DEPARTURE_DIMENSION:],
                direct_closure["q_rate_centers"],
                direct_closure["q_rate_coefficients"],
            )
            for coordinate in absolute_coordinates
        ]
    )
    return result


def _known_database(frozen: dict) -> dict:
    failed = frozen["failed"]
    truth = frozen["truth"]
    overlap = frozen["overlap"]
    design = frozen["design"]
    local = np.asarray(failed["fit_local_coordinates"], dtype=float)
    absolute = np.vstack(
        (
            overlap["revealed_overlap_absolute_coordinates"],
            truth["evaluated_absolute_coordinates"],
        )
    )
    exact_full = np.asarray(failed["fit_exact_full_rates_per_second"], dtype=float)
    old_full = np.asarray(failed["fit_old_full_rates_per_second"], dtype=float)
    exact_coordinate = np.asarray(
        failed["fit_exact_coordinate_rates_per_second"], dtype=float
    )
    direct_closure = _load_npz(DIRECT_FIELD)
    old_geometry = _load_npz(OLD_GEOMETRY)
    old_coordinate = _old_coordinate_field(
        old_full,
        absolute,
        old_geometry["online_coordinate_restriction"],
        direct_closure,
    )
    old_revealed_defect = float(
        np.max(
            np.abs(
                old_coordinate[:REVEALED_COUNT]
                - overlap[
                    "revealed_overlap_old_predicted_coordinate_rates_per_second"
                ]
            )
        )
    )
    if (
        local.shape != (REVEALED_COUNT + NEW_EXACT_COUNT, 470)
        or absolute.shape != (REVEALED_COUNT + NEW_EXACT_COUNT, 470)
        or exact_full.shape != (REVEALED_COUNT + NEW_EXACT_COUNT, 560)
        or exact_coordinate.shape != (REVEALED_COUNT + NEW_EXACT_COUNT, 470)
        or old_revealed_defect > 5.0e-10
    ):
        raise RuntimeError("revealed/training field database changed")
    return {
        "local_coordinates": local,
        "absolute_coordinates": absolute,
        "exact_full_rates": exact_full,
        "old_full_rates": old_full,
        "exact_coordinate_rates": exact_coordinate,
        "old_coordinate_rates": old_coordinate,
        "new_coordinate_jacobians": truth["coordinate_jacobians"],
        "old_revealed_coordinate_implementation_infinity_defect": old_revealed_defect,
        "direct_closure": direct_closure,
        "old_restriction": old_geometry["online_coordinate_restriction"],
        "center_restriction": design["authentic_center_fixed_restriction"],
        "active_basis": design["active_departure_basis"],
    }


def _fit_revision(
    frozen: dict, database: dict
) -> tuple[dict[str, np.ndarray], dict]:
    local = database["local_coordinates"]
    active_basis = database["active_basis"]
    weights, center_weights, forward_weights, active = _partition_weights(
        local, active_basis
    )
    features = parent.manifest._affine_features(active)
    new = np.arange(REVEALED_COUNT, REVEALED_COUNT + NEW_EXACT_COUNT)
    revealed = np.arange(REVEALED_COUNT)
    training = np.arange(REVEALED_COUNT + 1, REVEALED_COUNT + NEW_EXACT_COUNT)
    uniform = np.ones(NEW_EXACT_COUNT)
    full_coefficients, full_fit = parent.manifest._weighted_affine_fit(
        active[new],
        database["exact_full_rates"][new] - database["old_full_rates"][new],
        uniform,
        intercept=True,
    )
    jacobian_coefficients_flat, jacobian_fit = parent.manifest._weighted_affine_fit(
        active[new],
        database["new_coordinate_jacobians"].reshape(NEW_EXACT_COUNT, -1),
        uniform,
        intercept=True,
    )
    jacobian_coefficients = jacobian_coefficients_flat.reshape(
        4, PHYSICAL_DIMENSION, 560
    )
    new_full = database["old_full_rates"] + features @ full_coefficients
    transported_jacobians = np.einsum(
        "nf,fij->nij", features, jacobian_coefficients
    )
    new_coordinate = new_full @ database["center_restriction"].T
    new_coordinate[:, :PHYSICAL_DIMENSION] = np.einsum(
        "nij,nj->ni", transported_jacobians, new_full
    )
    atlas_full = database["old_full_rates"] + weights[:, None] * (
        new_full - database["old_full_rates"]
    )
    atlas_coordinate = database["old_coordinate_rates"] + weights[:, None] * (
        new_coordinate - database["old_coordinate_rates"]
    )
    full_errors = _relative_rows(atlas_full, database["exact_full_rates"])
    coordinate_errors = _relative_rows(
        atlas_coordinate, database["exact_coordinate_rates"]
    )
    q_errors = _relative_rows(
        atlas_coordinate[:, :PHYSICAL_DIMENSION],
        database["exact_coordinate_rates"][:, :PHYSICAL_DIMENSION],
    )
    z_slice = slice(PHYSICAL_DIMENSION, PHYSICAL_DIMENSION + MEMORY_DIMENSION)
    a_slice = slice(-DEPARTURE_DIMENSION, None)
    z_errors = _relative_rows(
        atlas_coordinate[:, z_slice], database["exact_coordinate_rates"][:, z_slice]
    )
    a_errors = _relative_rows(
        atlas_coordinate[:, a_slice], database["exact_coordinate_rates"][:, a_slice]
    )
    jacobian_errors = _relative_rows(
        transported_jacobians[new].reshape(NEW_EXACT_COUNT, -1),
        database["new_coordinate_jacobians"].reshape(NEW_EXACT_COUNT, -1),
    )

    loo_full = []
    loo_coordinate = []
    loo_q = []
    loo_jacobian = []
    for held in training:
        fit_indices = np.asarray([index for index in new if index != held])
        local_jacobian_indices = fit_indices - REVEALED_COUNT
        held_jacobian_index = held - REVEALED_COUNT
        loo_full_coefficients, _ = parent.manifest._weighted_affine_fit(
            active[fit_indices],
            database["exact_full_rates"][fit_indices]
            - database["old_full_rates"][fit_indices],
            np.ones(len(fit_indices)),
            intercept=True,
        )
        loo_jacobian_flat, _ = parent.manifest._weighted_affine_fit(
            active[fit_indices],
            database["new_coordinate_jacobians"][local_jacobian_indices].reshape(
                len(fit_indices), -1
            ),
            np.ones(len(fit_indices)),
            intercept=True,
        )
        held_full = (
            database["old_full_rates"][held]
            + features[held] @ loo_full_coefficients
        )
        held_jacobian = (
            features[held] @ loo_jacobian_flat
        ).reshape(PHYSICAL_DIMENSION, 560)
        held_coordinate = database["center_restriction"] @ held_full
        held_coordinate[:PHYSICAL_DIMENSION] = held_jacobian @ held_full
        loo_full.append(
            _relative_rows(
                held_full[None], database["exact_full_rates"][held : held + 1]
            )[0]
        )
        loo_coordinate.append(
            _relative_rows(
                held_coordinate[None],
                database["exact_coordinate_rates"][held : held + 1],
            )[0]
        )
        loo_q.append(
            _relative_rows(
                held_coordinate[None, :PHYSICAL_DIMENSION],
                database["exact_coordinate_rates"][
                    held : held + 1, :PHYSICAL_DIMENSION
                ],
            )[0]
        )
        loo_jacobian.append(
            _relative_rows(
                held_jacobian.reshape(1, -1),
                database["new_coordinate_jacobians"][
                    held_jacobian_index : held_jacobian_index + 1
                ].reshape(1, -1),
            )[0]
        )

    design = frozen["design"]
    failed = frozen["failed"]
    geometry_local = np.vstack(
        (
            design["revealed_overlap_local_coordinates"],
            design["training_local_coordinates"],
            design["holdout_local_coordinates"],
        )
    )
    geometry_weights, geometry_center, geometry_forward, geometry_active = (
        _partition_weights(geometry_local, active_basis)
    )
    center_direct_delta = failed["authentic_center_direct_decoded_scaled_delta"]
    center_delta = design["authentic_center_scaled_delta"]
    old_absolute_deltas = center_direct_delta + design[
        "translated_decoder_local_deltas"
    ]
    new_absolute_deltas = center_delta + design[
        "corrected_decoder_local_deltas"
    ]
    exact_absolute_deltas = center_delta + design["exact_local_scaled_deltas"]
    decoder_absolute = old_absolute_deltas + geometry_weights[:, None] * (
        new_absolute_deltas - old_absolute_deltas
    )
    decoder_errors = _relative_rows(decoder_absolute, exact_absolute_deltas)
    geometry_revealed = np.arange(REVEALED_COUNT)
    geometry_training = np.arange(REVEALED_COUNT, REVEALED_COUNT + TRAINING_COUNT)
    geometry_holdout = np.arange(
        REVEALED_COUNT + TRAINING_COUNT,
        REVEALED_COUNT + TRAINING_COUNT + HOLDOUT_COUNT,
    )

    def maximum(values: np.ndarray, indices: np.ndarray) -> float:
        return float(np.max(np.asarray(values)[indices]))

    metrics = {
        "fit_sample_count": NEW_EXACT_COUNT,
        "active_dimension": 3,
        "full_fit": full_fit,
        "jacobian_fit": jacobian_fit,
        "maximum_fit_condition_number": float(
            max(
                full_fit["regularized_normal_condition_number"],
                jacobian_fit["regularized_normal_condition_number"],
            )
        ),
        "maximum_new_exact_full_state_rate_relative_error": maximum(
            full_errors, new
        ),
        "maximum_new_exact_full_coordinate_rate_relative_error": maximum(
            coordinate_errors, new
        ),
        "maximum_new_exact_q162_rate_relative_error": maximum(q_errors, new),
        "maximum_new_exact_z280_rate_relative_error": maximum(z_errors, new),
        "maximum_new_exact_a28_rate_relative_error": maximum(a_errors, new),
        "maximum_revealed_full_state_rate_relative_error": maximum(
            full_errors, revealed
        ),
        "maximum_revealed_full_coordinate_rate_relative_error": maximum(
            coordinate_errors, revealed
        ),
        "maximum_revealed_q162_rate_relative_error": maximum(q_errors, revealed),
        "maximum_revealed_z280_rate_relative_error": maximum(z_errors, revealed),
        "maximum_revealed_a28_rate_relative_error": maximum(a_errors, revealed),
        "maximum_training_coordinate_Jacobian_relative_error": float(
            np.max(jacobian_errors)
        ),
        "maximum_leave_one_training_out_full_state_rate_relative_error": float(
            np.max(loo_full)
        ),
        "maximum_leave_one_training_out_full_coordinate_rate_relative_error": float(
            np.max(loo_coordinate)
        ),
        "maximum_leave_one_training_out_q162_rate_relative_error": float(
            np.max(loo_q)
        ),
        "maximum_leave_one_training_out_coordinate_Jacobian_relative_error": float(
            np.max(loo_jacobian)
        ),
        "maximum_revealed_partition_weight": float(np.max(weights[revealed])),
        "center_partition_weight": float(weights[REVEALED_COUNT]),
        "minimum_training_partition_weight": float(np.min(weights[training])),
        "maximum_revealed_decoder_relative_error": maximum(
            decoder_errors, geometry_revealed
        ),
        "maximum_training_decoder_relative_error": maximum(
            decoder_errors, geometry_training
        ),
        "maximum_holdout_geometry_decoder_relative_error": maximum(
            decoder_errors, geometry_holdout
        ),
        "maximum_revealed_geometry_partition_weight": float(
            np.max(geometry_weights[geometry_revealed])
        ),
        "minimum_training_geometry_partition_weight": float(
            np.min(geometry_weights[geometry_training])
        ),
        "minimum_holdout_geometry_partition_weight": float(
            np.min(geometry_weights[geometry_holdout])
        ),
        "minimum_revealed_coordinate_radius": float(
            np.min(np.linalg.norm(local[revealed], axis=1))
        ),
        "maximum_revealed_forward_coordinate": float(
            np.max(active[revealed, 0])
        ),
        "minimum_training_forward_coordinate": float(
            np.min(active[training, 0])
        ),
        "minimum_holdout_forward_coordinate": float(
            np.min(geometry_active[geometry_holdout, 0])
        ),
        "old_revealed_coordinate_implementation_infinity_defect": database[
            "old_revealed_coordinate_implementation_infinity_defect"
        ],
        "online_state_dependent_coordinate_Jacobian_calls": 0,
        "new_exact_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
    }
    closure = {
        "authentic_center_absolute_coordinate": design[
            "authentic_center_absolute_coordinate"
        ],
        "authentic_center_scaled_delta": center_delta,
        "authentic_center_direct_decoded_scaled_delta": center_direct_delta,
        "authentic_center_fixed_restriction": database["center_restriction"],
        "active_departure_basis": active_basis,
        "decoder_affine_coefficients": design["decoder_affine_coefficients"],
        "full_rate_affine_coefficients": full_coefficients,
        "q162_Jacobian_affine_coefficients": jacobian_coefficients,
        "center_core_full_radius": np.asarray(CENTER_CORE_FULL_RADIUS),
        "center_core_zero_radius": np.asarray(CENTER_CORE_ZERO_RADIUS),
        "forward_zero_coordinate": np.asarray(FORWARD_ZERO_COORDINATE),
        "forward_full_coordinate": np.asarray(FORWARD_FULL_COORDINATE),
        "known_local_coordinates": local,
        "known_partition_weights": weights,
        "known_center_weights": center_weights,
        "known_forward_weights": forward_weights,
        "known_active_coordinates": active,
        "known_atlas_full_rates_per_second": atlas_full,
        "known_atlas_coordinate_rates_per_second": atlas_coordinate,
        "known_transported_q162_Jacobians": transported_jacobians,
        "leave_one_out_full_state_relative_errors": np.asarray(loo_full),
        "leave_one_out_full_coordinate_relative_errors": np.asarray(
            loo_coordinate
        ),
        "leave_one_out_q162_relative_errors": np.asarray(loo_q),
        "leave_one_out_coordinate_Jacobian_relative_errors": np.asarray(
            loo_jacobian
        ),
        "geometry_local_coordinates": geometry_local,
        "geometry_partition_weights": geometry_weights,
        "geometry_center_weights": geometry_center,
        "geometry_forward_weights": geometry_forward,
        "geometry_active_coordinates": geometry_active,
        "geometry_partitioned_decoder_scaled_deltas": decoder_absolute,
        "geometry_decoder_relative_errors": decoder_errors,
        "holdout_primitive_states": design["holdout_primitive_states"],
        "holdout_local_coordinates": design["holdout_local_coordinates"],
        "holdout_absolute_coordinates": design["holdout_absolute_coordinates"],
    }
    return closure, metrics


class PartitionedAuthenticCenterField:
    """Cheap partition-of-unity decoder and tangent-transported field."""

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
        self.full_coefficients = np.asarray(
            closure["full_rate_affine_coefficients"], dtype=float
        )
        self.q_jacobian_coefficients = np.asarray(
            closure["q162_Jacobian_affine_coefficients"], dtype=float
        )
        self.center_restriction = np.asarray(
            closure["authentic_center_fixed_restriction"], dtype=float
        )

    def _active(self, local_coordinate: np.ndarray) -> np.ndarray:
        eta = np.asarray(local_coordinate, dtype=float)
        return eta[-DEPARTURE_DIMENSION:] @ self.active_basis / parent.manifest.ACTIVE_SCALE

    def _features(self, local_coordinate: np.ndarray) -> np.ndarray:
        return np.concatenate(([1.0], self._active(local_coordinate)))

    def weight(self, local_coordinate: np.ndarray) -> float:
        return float(
            _partition_weights(
                np.asarray(local_coordinate, dtype=float), self.active_basis
            )[0][0]
        )

    def decoded_delta(self, local_coordinate: np.ndarray) -> np.ndarray:
        eta = np.asarray(local_coordinate, dtype=float)
        absolute = self.center_coordinate + eta
        old = self.direct.decoded_delta(absolute)
        translated = old - self.center_direct_delta
        new = (
            self.center_delta
            + translated
            + self._active(eta) @ self.decoder_coefficients
        )
        weight = self.weight(eta)
        return old + weight * (new - old)

    def decoded_state(self, local_coordinate: np.ndarray) -> np.ndarray:
        delta = self.decoded_delta(local_coordinate)
        return self.model.base_state + (
            self.model.columns.ravel() * delta
        ).reshape(self.model.base_state.shape)

    def full_state_rate(self, local_coordinate: np.ndarray) -> np.ndarray:
        eta = np.asarray(local_coordinate, dtype=float)
        absolute = self.center_coordinate + eta
        old = self.direct.full_state_rate(absolute)
        new = old + self._features(eta) @ self.full_coefficients
        return old + self.weight(eta) * (new - old)

    def _new_coordinate_rate(self, local_coordinate: np.ndarray) -> np.ndarray:
        eta = np.asarray(local_coordinate, dtype=float)
        absolute = self.center_coordinate + eta
        full = (
            self.direct.full_state_rate(absolute)
            + self._features(eta) @ self.full_coefficients
        )
        result = self.center_restriction @ full
        jacobian = np.einsum(
            "f,fij->ij", self._features(eta), self.q_jacobian_coefficients
        )
        result[:PHYSICAL_DIMENSION] = jacobian @ full
        return result

    def field(self, local_coordinate: np.ndarray) -> np.ndarray:
        eta = np.asarray(local_coordinate, dtype=float)
        absolute = self.center_coordinate + eta
        old = self.direct.field(absolute)
        new = self._new_coordinate_rate(eta)
        return old + self.weight(eta) * (new - old)


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "superseded_failed_model": {
            "classification": parent.FAIL_CLASSIFICATION,
            "preserved_as_binding_negative_result": True,
            "failure_localization": "pooled_backward_and_forward_q162_affine_residual",
            "physical_truth_failure": False,
        },
        "mathematical_architecture": {
            "absolute_state": "one_global_y470_equal_q162_z280_a28",
            "local_state": "eta_equals_y_minus_authentic_center",
            "chart_transition": "exact_affine_translation_identity_tangent",
            "atlas": "C2_partition_of_unity_between_old_field_and_authentic_center_patch",
            "center_weight": "compact_radial_C2_core",
            "forward_weight": "C2_gate_in_first_active_center_coordinate",
            "combined_weight": "one_minus_one_minus_center_times_one_minus_forward",
            "decoder": "partitioned_old_decoder_and_center_reanchored_affine_decoder",
            "full_physical_rate": "partitioned_old_rate_and_new_five_sample_affine_residual",
            "q162_rate": "affinely_transported_q162_state_Jacobian_times_new_full_rate",
            "z280_a28_rate": "fixed_center_projections_of_new_full_rate",
            "online_state_dependent_coordinate_Jacobian_calls": 0,
            "online_exact_rate_calls": 0,
            "new_complete_generator_assemblies": 0,
        },
        "partition": {
            "center_core_full_radius": CENTER_CORE_FULL_RADIUS,
            "center_core_zero_radius": CENTER_CORE_ZERO_RADIUS,
            "forward_zero_coordinate": FORWARD_ZERO_COORDINATE,
            "forward_full_coordinate": FORWARD_FULL_COORDINATE,
            "C2_quintic_endpoints": True,
        },
        "coefficient_fit": {
            "exact_samples": NEW_EXACT_COUNT,
            "revealed_overlap_samples_used_as_coefficients": 0,
            "revealed_overlap_samples_used_as_partition_controls": REVEALED_COUNT,
            "features": "constant_plus_three_active_coordinates",
            "ridge": parent.manifest.AFFINE_RIDGE,
            "full_rate_coefficients": [4, 560],
            "q162_Jacobian_coefficients": [4, PHYSICAL_DIMENSION, 560],
            "leave_one_training_direction_out_bound": True,
            "coefficients_frozen_before_blind_holdout_truth": True,
        },
        "binding_revision_gates": {
            "maximum_fit_condition_number": 100.0,
            "maximum_new_exact_full_state_rate_relative_error": 1.0e-2,
            "maximum_new_exact_full_coordinate_rate_relative_error": 1.0e-2,
            "maximum_new_exact_q162_rate_relative_error": 1.0e-2,
            "maximum_new_exact_z280_rate_relative_error": 1.0e-2,
            "maximum_new_exact_a28_rate_relative_error": 1.0e-2,
            "maximum_revealed_full_state_rate_relative_error": 5.0e-2,
            "maximum_revealed_full_coordinate_rate_relative_error": 5.0e-2,
            "maximum_revealed_q162_rate_relative_error": 5.0e-2,
            "maximum_revealed_z280_rate_relative_error": 5.0e-2,
            "maximum_revealed_a28_rate_relative_error": 5.0e-2,
            "maximum_training_coordinate_Jacobian_relative_error": 1.0e-3,
            "maximum_leave_one_training_out_full_state_rate_relative_error": 2.0e-2,
            "maximum_leave_one_training_out_full_coordinate_rate_relative_error": 2.0e-2,
            "maximum_leave_one_training_out_q162_rate_relative_error": 2.0e-2,
            "maximum_leave_one_training_out_coordinate_Jacobian_relative_error": 1.0e-3,
            "maximum_revealed_partition_weight": 0.0,
            "center_partition_weight": 1.0,
            "minimum_training_partition_weight": 1.0,
            "maximum_revealed_decoder_relative_error": 1.0e-2,
            "maximum_training_decoder_relative_error": 1.0e-2,
            "maximum_holdout_geometry_decoder_relative_error": 5.0e-2,
            "minimum_holdout_geometry_partition_weight": 1.0,
            "new_exact_rate_calls_equal": 0,
        },
        "blind_holdout_execution": {
            "work_package": AUTHORIZED_NEXT,
            "count": HOLDOUT_COUNT,
            "coefficients_may_not_change": True,
            "maximum_full_state_rate_relative_error": 7.5e-2,
            "maximum_full_coordinate_rate_relative_error": 7.5e-2,
            "maximum_q162_rate_relative_error": 7.5e-2,
            "maximum_z280_rate_relative_error": 7.5e-2,
            "maximum_a28_rate_relative_error": 7.5e-2,
            "maximum_q162_Jacobian_relative_error": 5.0e-3,
            "maximum_decoder_relative_error": 5.0e-2,
            "maximum_decoder_coordinate_relative_mismatch": 7.5e-2,
            "completed_exact_rate_calls_equal": HOLDOUT_COUNT,
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
            "online_state_dependent_coordinate_Jacobian_calls_equal": 0,
            "new_complete_generator_assemblies_equal": 0,
            "new_nonlinear_fixed_Q_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "decision": {
            "pass_classification": "partitioned_authentic_center_field_independently_validated",
            "fail_classification": "partitioned_authentic_center_field_blind_validation_failed",
            "pass_authorizes_only": "definitions_only_reduced_slow_atlas_integrator_manifest",
            "fail_authorizes_only": "definitions_only_partitioned_field_revision",
        },
        "authorization_boundaries": {
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "fast_average_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _checks(metrics: dict, gates: dict) -> dict:
    scalar = {
        key: metrics[key] <= value
        for key, value in gates.items()
        if key.startswith("maximum_") and key in metrics
    }
    scalar.update(
        {
            key: metrics[key] >= value
            for key, value in gates.items()
            if key.startswith("minimum_") and key in metrics
        }
    )
    scalar["center_partition_weight"] = (
        metrics["center_partition_weight"] == gates["center_partition_weight"]
    )
    scalar["new_exact_rate_budget"] = (
        metrics["new_exact_rate_calls"] == gates["new_exact_rate_calls_equal"]
    )
    return scalar


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
        raise RuntimeError("partitioned-field revision manifest already exists")
    database = _known_database(frozen)
    closure, metrics = _fit_revision(frozen, database)
    contract = _contract()
    checks = _checks(metrics, contract["binding_revision_gates"])
    if not all(checks.values()):
        raise RuntimeError(f"partitioned-field revision readiness failed: {checks}")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "partitioned_local_field.npz", closure)
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
            "training_truth_sha256": _sha(PARENT_TRUTH),
            "failed_field_sha256": _sha(PARENT_FAILED_FIELD),
            "frozen_design_sha256": _sha(FROZEN_DESIGN),
            "overlap_design_sha256": _sha(OVERLAP_DESIGN),
            "direct_field_sha256": _sha(DIRECT_FIELD),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "failed_pooled_affine_result_preserved": True,
        "partitioned_field_readiness_passed": True,
        "new_exact_rate_calls": 0,
        "blind_holdout_rate_calls": 0,
        "coefficients_frozen_before_blind_holdout_truth": True,
        "online_state_dependent_coordinate_Jacobian_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
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
        parent.direct_manifest.THIS_RUNNER,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
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
                name: os.environ.get(name) for name in parent._thread_environment()
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
                "# Partitioned authentic-center field revision WP10c9d6c7c3b5c4f25cv",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The failed pooled affine field remains rejected. The revised atlas preserves the old validated field on all backward overlap controls and applies a compact-center/forward-sector partition to the new patch.",
                "",
                f"Maximum new exact full-coordinate and q162 errors are `{metrics['maximum_new_exact_full_coordinate_rate_relative_error']:.6e}` and `{metrics['maximum_new_exact_q162_rate_relative_error']:.6e}`. Maximum revealed overlap full-coordinate error is `{metrics['maximum_revealed_full_coordinate_rate_relative_error']:.6e}`.",
                "",
                f"Worst leave-one-training-direction-out full-coordinate, q162, and physical-Jacobian errors are `{metrics['maximum_leave_one_training_out_full_coordinate_rate_relative_error']:.6e}`, `{metrics['maximum_leave_one_training_out_q162_rate_relative_error']:.6e}`, and `{metrics['maximum_leave_one_training_out_coordinate_Jacobian_relative_error']:.6e}`.",
                "",
                "No new exact rate was evaluated. The four holdout rates remain blind and the coefficients are now immutable.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`. No trajectory, physical microburst, predictive cycle, or reduced slow evolution is authorized.",
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
