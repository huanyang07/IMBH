#!/usr/bin/env python3
"""Freeze the minimal forward-quadratic authentic-center field revision."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_partitioned_authentic_center_field_blind_validation_wp10c9d6c7c3b5c4f25cw as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cx"
PARENT_COMMIT = "75d3083671f36dcd69524df0ead95ea3a9c9f516"
PARENT_PARENT = "c8e92126f56c3eddfab04849e68db7a8b99ec5c4"
PARENT_TREE = "b6587627ca5acba0d9315931d94298b34e4bb1de"

CLASSIFICATION = "forward_quadratic_authentic_center_field_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25cy"

REVEALED_SAMPLE_COUNT = 9
NEW_GEOMETRY_COUNT = 4
PHYSICAL_DIMENSION = 162
MEMORY_DIMENSION = 280
DEPARTURE_DIMENSION = 28
ACTIVE_SCALE = 1.5e-2
RIDGE = 1.0e-8
COMPONENT_BOUND = 1.5e-2

# These directions deliberately vary cone radius. They distinguish forward
# curvature from a transverse-radius surrogate before any new rate is seen.
BLIND_MIXING_MAGNITUDES = np.asarray((0.0, 0.25, 0.50, 0.75))
BLIND_AZIMUTHS_RADIANS = np.asarray((0.0, 0.0, math.pi / 8.0, 5.0 * math.pi / 8.0))

ARTIFACT = (
    "causal_inner_forward_quadratic_field_revision_manifest_"
    "wp10c9d6c7c3b5c4f25cx"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_forward_quadratic_field_revision_manifest_"
    "wp10c9d6c7c3b5c4f25cx.py"
)
THIS_TEST = (
    "tests/test_causal_inner_forward_quadratic_field_revision_manifest_"
    "wp10c9d6c7c3b5c4f25cx.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_FORWARD_QUADRATIC_FIELD_"
    "REVISION_MANIFEST_WP10C9D6C7C3B5C4F25CX_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

FAILED_VALIDATION_ARRAYS = parent.CANONICAL_DIRECTORY / "validation_arrays.npz"
PARTITIONED_FIELD = parent.manifest.CANONICAL_DIRECTORY / "partitioned_local_field.npz"
TRAINED_FIELD = parent.manifest.parent.CANONICAL_DIRECTORY / "authentic_center_local_field.npz"
TRAINING_TRUTH = parent.manifest.PARENT_TRUTH
FROZEN_DESIGN = parent.manifest.FROZEN_DESIGN
PRIOR_GEOMETRY = parent.manifest.parent.manifest.GEOMETRY_ARRAYS

partition = parent.manifest
training = partition.parent
vector_field = partition.vector_field

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


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(right)), np.finfo(float).tiny)
    )


def _normalize(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).tiny:
        raise RuntimeError("blind direction vanished")
    return vector / norm


def _full_features(active_coordinates: np.ndarray) -> np.ndarray:
    active = np.asarray(active_coordinates, dtype=float)
    if active.ndim == 1:
        active = active.reshape(1, -1)
    return np.column_stack((np.ones(len(active)), active, active[:, 0] ** 2))


def _q_jacobian_features(active_coordinates: np.ndarray) -> np.ndarray:
    active = np.asarray(active_coordinates, dtype=float)
    if active.ndim == 1:
        active = active.reshape(1, -1)
    return np.column_stack((np.ones(len(active)), active))


def _ridge_fit(design: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, dict]:
    matrix = np.asarray(design, dtype=float)
    target = np.asarray(targets, dtype=float)
    penalty = RIDGE * np.eye(matrix.shape[1])
    penalty[0, 0] = 0.0
    regularized = matrix.T @ matrix + penalty
    coefficients = np.linalg.solve(regularized, matrix.T @ target)
    singular = np.linalg.svd(regularized, compute_uv=False)
    return coefficients, {
        "design_rank": int(np.linalg.matrix_rank(matrix)),
        "regularized_normal_rank": int(np.linalg.matrix_rank(regularized)),
        "regularized_normal_condition_number": float(singular[0] / singular[-1]),
    }


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("affine blind-failure certificate commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("affine blind-failure lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("affine blind-failure tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "validation_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    validation = _load_npz(FAILED_VALIDATION_ARRAYS)
    closure = _load_npz(PARTITIONED_FIELD)
    trained = _load_npz(TRAINED_FIELD)
    truth = _load_npz(TRAINING_TRUTH)
    design = _load_npz(FROZEN_DESIGN)
    geometry = _load_npz(PRIOR_GEOMETRY)
    required_true_checks = (
        "Schur_condition",
        "coordinate_condition",
        "decoder",
        "decoder_coordinate",
        "height",
        "incoming_excision",
        "optical_depth",
        "partition",
        "q162_Jacobian",
        "rate_tangency",
        "reaction_identity",
        "reconstruction_maximum",
        "reconstruction_minimum",
        "z280",
    )
    if (
        summary["passed"]
        or summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["completed_exact_rate_calls"] != NEW_GEOMETRY_COUNT
        or summary["failed_exact_rate_calls"] != 0
        or summary["coefficients_refit_after_holdout_truth"]
        or summary["authorized_next"] != parent.FAIL_AUTHORIZED_NEXT
        or metrics["passed"]
        or not all(metrics["checks"][name] for name in required_true_checks)
        or metrics["checks"]["full_state"]
        or metrics["checks"]["full_coordinate"]
        or metrics["checks"]["q162"]
        or validation["total_rates_per_second"].shape != (NEW_GEOMETRY_COUNT, 560)
        or validation["exact_q162_Jacobians"].shape
        != (NEW_GEOMETRY_COUNT, PHYSICAL_DIMENSION, 560)
        or trained["fit_local_coordinates"].shape[0] != 21
        or truth["coordinate_jacobians"].shape != (5, PHYSICAL_DIMENSION, 560)
        or design["authentic_center_primitive_state"].shape != (112, 5)
        or geometry["training_directions"].shape != (4, DEPARTURE_DIMENSION)
        or geometry["holdout_directions"].shape != (4, DEPARTURE_DIMENSION)
    ):
        raise RuntimeError("forward-quadratic revision authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"affine blind-validation source changed: {relative}")
    if _sha(FAILED_VALIDATION_ARRAYS) != hashes["validation_arrays.npz"]:
        raise RuntimeError("affine blind-validation arrays changed")
    for name, expected in training._thread_environment().items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("forward-quadratic manifest requires a clean tracked tree")
    return {
        "summary": summary,
        "metrics": metrics,
        "hashes": hashes,
        "validation": validation,
        "closure": closure,
        "trained": trained,
        "truth": truth,
        "design": design,
        "geometry": geometry,
    }


def _revealed_database(frozen: dict) -> dict[str, np.ndarray]:
    trained = frozen["trained"]
    validation = frozen["validation"]
    closure = frozen["closure"]
    truth = frozen["truth"]
    local = np.vstack(
        (trained["fit_local_coordinates"][16:], validation["holdout_local_coordinates"])
    )
    active = np.vstack(
        (
            trained["fit_active_coordinates"][16:],
            local[5:, -DEPARTURE_DIMENSION:]
            @ closure["active_departure_basis"]
            / ACTIVE_SCALE,
        )
    )
    exact_full = np.vstack(
        (trained["fit_exact_full_rates_per_second"][16:], validation["total_rates_per_second"])
    )
    exact_coordinate = np.vstack(
        (
            trained["fit_exact_coordinate_rates_per_second"][16:],
            validation["exact_coordinate_rates_per_second"],
        )
    )
    exact_jacobian = np.vstack(
        (truth["coordinate_jacobians"], validation["exact_q162_Jacobians"])
    )
    old_full = np.empty_like(exact_full)
    old_full[:5] = trained["fit_old_full_rates_per_second"][16:]
    old_full[5:] = validation["predicted_full_rates_per_second"] - (
        _q_jacobian_features(active[5:]) @ closure["full_rate_affine_coefficients"]
    )
    if (
        local.shape != (REVEALED_SAMPLE_COUNT, 470)
        or active.shape != (REVEALED_SAMPLE_COUNT, 3)
        or exact_full.shape != (REVEALED_SAMPLE_COUNT, 560)
        or exact_coordinate.shape != (REVEALED_SAMPLE_COUNT, 470)
        or exact_jacobian.shape
        != (REVEALED_SAMPLE_COUNT, PHYSICAL_DIMENSION, 560)
    ):
        raise RuntimeError("revealed forward-sector database changed")
    return {
        "local_coordinates": local,
        "active_coordinates": active,
        "exact_full_rates": exact_full,
        "old_full_rates": old_full,
        "exact_coordinate_rates": exact_coordinate,
        "exact_q162_Jacobians": exact_jacobian,
    }


def _blind_directions(active_basis: np.ndarray) -> np.ndarray:
    forward, transverse_1, transverse_2 = np.asarray(active_basis, dtype=float).T
    directions = []
    for mixing, azimuth in zip(BLIND_MIXING_MAGNITUDES, BLIND_AZIMUTHS_RADIANS):
        transverse = math.cos(float(azimuth)) * transverse_1 + math.sin(
            float(azimuth)
        ) * transverse_2
        directions.append(_normalize(forward + float(mixing) * transverse))
    return np.asarray(directions)


def _fit_revision(frozen: dict, database: dict) -> tuple[dict[str, np.ndarray], dict]:
    closure = frozen["closure"]
    active = database["active_coordinates"]
    full_design = _full_features(active)
    jacobian_design = _q_jacobian_features(active)
    full_coefficients, full_fit = _ridge_fit(
        full_design, database["exact_full_rates"] - database["old_full_rates"]
    )
    jacobian_flat, jacobian_fit = _ridge_fit(
        jacobian_design,
        database["exact_q162_Jacobians"].reshape(REVEALED_SAMPLE_COUNT, -1),
    )
    jacobian_coefficients = jacobian_flat.reshape(4, PHYSICAL_DIMENSION, 560)
    predicted_full = database["old_full_rates"] + full_design @ full_coefficients
    predicted_jacobian = np.einsum(
        "nf,fij->nij", jacobian_design, jacobian_coefficients
    )
    restriction = closure["authentic_center_fixed_restriction"]
    predicted_coordinate = predicted_full @ restriction.T
    predicted_coordinate[:, :PHYSICAL_DIMENSION] = np.einsum(
        "nij,nj->ni", predicted_jacobian, predicted_full
    )

    def grouped_errors(
        full: np.ndarray, coordinate: np.ndarray, indices: np.ndarray
    ) -> tuple[float, ...]:
        q_slice = slice(0, PHYSICAL_DIMENSION)
        z_slice = slice(PHYSICAL_DIMENSION, PHYSICAL_DIMENSION + MEMORY_DIMENSION)
        a_slice = slice(-DEPARTURE_DIMENSION, None)
        exact_full = database["exact_full_rates"][indices]
        exact_coordinate = database["exact_coordinate_rates"][indices]
        return (
            _relative_error(full, exact_full),
            _relative_error(coordinate, exact_coordinate),
            _relative_error(coordinate[:, q_slice], exact_coordinate[:, q_slice]),
            _relative_error(coordinate[:, z_slice], exact_coordinate[:, z_slice]),
            _relative_error(coordinate[:, a_slice], exact_coordinate[:, a_slice]),
        )

    per_fit = []
    for index in range(REVEALED_SAMPLE_COUNT):
        per_fit.append(
            grouped_errors(
                predicted_full[index : index + 1],
                predicted_coordinate[index : index + 1],
                np.asarray((index,)),
            )
        )

    leave_one_out = []
    loo_conditions = []
    for held in range(1, REVEALED_SAMPLE_COUNT):
        fit_indices = np.asarray(
            [index for index in range(REVEALED_SAMPLE_COUNT) if index != held]
        )
        held_full_coefficients, held_full_fit = _ridge_fit(
            full_design[fit_indices],
            database["exact_full_rates"][fit_indices]
            - database["old_full_rates"][fit_indices],
        )
        held_jacobian_flat, held_jacobian_fit = _ridge_fit(
            jacobian_design[fit_indices],
            database["exact_q162_Jacobians"][fit_indices].reshape(
                len(fit_indices), -1
            ),
        )
        held_full = (
            database["old_full_rates"][held]
            + full_design[held] @ held_full_coefficients
        )
        held_jacobian = (
            jacobian_design[held] @ held_jacobian_flat
        ).reshape(PHYSICAL_DIMENSION, 560)
        held_coordinate = restriction @ held_full
        held_coordinate[:PHYSICAL_DIMENSION] = held_jacobian @ held_full
        errors = grouped_errors(
            held_full.reshape(1, -1),
            held_coordinate.reshape(1, -1),
            np.asarray((held,)),
        )
        jacobian_error = _relative_error(
            held_jacobian, database["exact_q162_Jacobians"][held]
        )
        leave_one_out.append((*errors, jacobian_error))
        loo_conditions.append(
            max(
                held_full_fit["regularized_normal_condition_number"],
                held_jacobian_fit["regularized_normal_condition_number"],
            )
        )

    fit_errors = np.asarray(per_fit)
    loo_errors = np.asarray(leave_one_out)
    directions = _blind_directions(closure["active_departure_basis"])
    direction_singular = np.linalg.svd(directions, compute_uv=False)
    previous_directions = np.vstack(
        (
            frozen["geometry"]["training_directions"],
            frozen["geometry"]["holdout_directions"],
        )
    )
    maximum_previous_alignment = float(
        np.max(np.abs(directions @ previous_directions.T))
    )
    metrics = {
        "revealed_exact_sample_count": REVEALED_SAMPLE_COUNT,
        "new_exact_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "full_feature_count": int(full_design.shape[1]),
        "q162_Jacobian_feature_count": int(jacobian_design.shape[1]),
        "full_fit": full_fit,
        "q162_Jacobian_fit": jacobian_fit,
        "maximum_fit_condition_number": float(
            max(
                full_fit["regularized_normal_condition_number"],
                jacobian_fit["regularized_normal_condition_number"],
            )
        ),
        "maximum_leave_one_out_condition_number": float(max(loo_conditions)),
        "maximum_fit_full_state_rate_relative_error": float(np.max(fit_errors[:, 0])),
        "maximum_fit_full_coordinate_rate_relative_error": float(np.max(fit_errors[:, 1])),
        "maximum_fit_q162_rate_relative_error": float(np.max(fit_errors[:, 2])),
        "maximum_fit_z280_rate_relative_error": float(np.max(fit_errors[:, 3])),
        "maximum_fit_a28_rate_relative_error": float(np.max(fit_errors[:, 4])),
        "maximum_leave_one_out_full_state_rate_relative_error": float(np.max(loo_errors[:, 0])),
        "maximum_leave_one_out_full_coordinate_rate_relative_error": float(np.max(loo_errors[:, 1])),
        "maximum_leave_one_out_q162_rate_relative_error": float(np.max(loo_errors[:, 2])),
        "maximum_leave_one_out_z280_rate_relative_error": float(np.max(loo_errors[:, 3])),
        "maximum_leave_one_out_a28_rate_relative_error": float(np.max(loo_errors[:, 4])),
        "maximum_leave_one_out_q162_Jacobian_relative_error": float(np.max(loo_errors[:, 5])),
        "blind_direction_count": int(len(directions)),
        "blind_direction_rank": int(np.linalg.matrix_rank(directions)),
        "minimum_blind_direction_singular_value": float(direction_singular[2]),
        "maximum_previous_direction_alignment": maximum_previous_alignment,
        "prior_maximum_blind_decoder_relative_error": frozen["summary"][
            "maximum_decoder_relative_error"
        ],
        "prior_maximum_blind_q162_Jacobian_relative_error": frozen["summary"][
            "maximum_q162_Jacobian_relative_error"
        ],
    }
    output = {
        "authentic_center_primitive_state": frozen["design"][
            "authentic_center_primitive_state"
        ],
        "authentic_center_absolute_coordinate": closure[
            "authentic_center_absolute_coordinate"
        ],
        "authentic_center_scaled_delta": closure["authentic_center_scaled_delta"],
        "authentic_center_direct_decoded_scaled_delta": closure[
            "authentic_center_direct_decoded_scaled_delta"
        ],
        "authentic_center_fixed_restriction": restriction,
        "active_departure_basis": closure["active_departure_basis"],
        "decoder_affine_coefficients": closure["decoder_affine_coefficients"],
        "full_rate_forward_quadratic_coefficients": full_coefficients,
        "q162_Jacobian_affine_coefficients": jacobian_coefficients,
        "center_core_full_radius": closure["center_core_full_radius"],
        "center_core_zero_radius": closure["center_core_zero_radius"],
        "forward_zero_coordinate": closure["forward_zero_coordinate"],
        "forward_full_coordinate": closure["forward_full_coordinate"],
        "revealed_local_coordinates": database["local_coordinates"],
        "revealed_active_coordinates": active,
        "revealed_exact_full_rates_per_second": database["exact_full_rates"],
        "revealed_old_full_rates_per_second": database["old_full_rates"],
        "revealed_exact_coordinate_rates_per_second": database[
            "exact_coordinate_rates"
        ],
        "revealed_exact_q162_Jacobians": database["exact_q162_Jacobians"],
        "revealed_predicted_full_rates_per_second": predicted_full,
        "revealed_predicted_coordinate_rates_per_second": predicted_coordinate,
        "revealed_predicted_q162_Jacobians": predicted_jacobian,
        "revealed_fit_relative_errors": fit_errors,
        "leave_one_out_relative_errors": loo_errors,
        "blind_directions": directions,
        "blind_component_bounds": np.full(NEW_GEOMETRY_COUNT, COMPONENT_BOUND),
        "blind_mixing_magnitudes": BLIND_MIXING_MAGNITUDES,
        "blind_azimuths_radians": BLIND_AZIMUTHS_RADIANS,
    }
    return output, metrics


class ForwardQuadraticAuthenticCenterField:
    """Partitioned field with a minimal forward-quadratic physical residual."""

    def __init__(self, closure: dict[str, np.ndarray], *, model=None, direct=None):
        self.model = model or vector_field.ReducedVectorField()
        self.direct = direct or partition.direct_manifest.DirectCoordinateField(
            _load_npz(partition.DIRECT_FIELD), model=self.model
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
        self.active_basis = np.asarray(closure["active_departure_basis"], dtype=float)
        self.decoder_coefficients = np.asarray(
            closure["decoder_affine_coefficients"], dtype=float
        )
        self.full_coefficients = np.asarray(
            closure["full_rate_forward_quadratic_coefficients"], dtype=float
        )
        self.q_jacobian_coefficients = np.asarray(
            closure["q162_Jacobian_affine_coefficients"], dtype=float
        )
        self.center_restriction = np.asarray(
            closure["authentic_center_fixed_restriction"], dtype=float
        )

    def _active(self, local_coordinate: np.ndarray) -> np.ndarray:
        eta = np.asarray(local_coordinate, dtype=float)
        return eta[-DEPARTURE_DIMENSION:] @ self.active_basis / ACTIVE_SCALE

    def _full_features(self, local_coordinate: np.ndarray) -> np.ndarray:
        active = self._active(local_coordinate)
        return np.concatenate(([1.0], active, [active[0] ** 2]))

    def _q_features(self, local_coordinate: np.ndarray) -> np.ndarray:
        return np.concatenate(([1.0], self._active(local_coordinate)))

    def weight(self, local_coordinate: np.ndarray) -> float:
        return float(
            partition._partition_weights(
                np.asarray(local_coordinate, dtype=float), self.active_basis
            )[0][0]
        )

    def decoded_delta(self, local_coordinate: np.ndarray) -> np.ndarray:
        eta = np.asarray(local_coordinate, dtype=float)
        absolute = self.center_coordinate + eta
        old = self.direct.decoded_delta(absolute)
        translated = old - self.center_direct_delta
        new = self.center_delta + translated + self._active(eta) @ self.decoder_coefficients
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
        new = old + self._full_features(eta) @ self.full_coefficients
        return old + self.weight(eta) * (new - old)

    def q162_jacobian(self, local_coordinate: np.ndarray) -> np.ndarray:
        return np.einsum(
            "f,fij->ij", self._q_features(local_coordinate), self.q_jacobian_coefficients
        )

    def _new_coordinate_rate(self, local_coordinate: np.ndarray) -> np.ndarray:
        eta = np.asarray(local_coordinate, dtype=float)
        absolute = self.center_coordinate + eta
        full = self.direct.full_state_rate(absolute) + self._full_features(eta) @ self.full_coefficients
        result = self.center_restriction @ full
        result[:PHYSICAL_DIMENSION] = self.q162_jacobian(eta) @ full
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
        "preserved_negative_result": {
            "classification": parent.FAIL_CLASSIFICATION,
            "physical_truth_failure": False,
            "failure_localization": "affine_physical_rate_residual",
            "geometry_decoder_and_q162_Jacobian_supported": True,
        },
        "mathematical_architecture": {
            "state": "one_global_y470_equal_q162_z280_a28",
            "atlas": "unchanged_C2_partition_of_unity",
            "decoder": "unchanged_partitioned_affine_decoder",
            "full_physical_rate": "old_field_plus_constant_linear_active_and_forward_squared_residual",
            "q162_rate": "affinely_transported_q162_state_Jacobian_times_forward_quadratic_full_rate",
            "z280_a28_rate": "fixed_center_projections_of_forward_quadratic_full_rate",
            "online_state_dependent_coordinate_Jacobian_calls": 0,
            "online_exact_rate_calls": 0,
        },
        "coefficient_fit": {
            "revealed_exact_samples": REVEALED_SAMPLE_COUNT,
            "features": ["1", "xi_forward", "xi_transverse_1", "xi_transverse_2", "xi_forward_squared"],
            "ridge": RIDGE,
            "full_rate_coefficients": [5, 560],
            "q162_Jacobian_features": ["1", "xi_forward", "xi_transverse_1", "xi_transverse_2"],
            "q162_Jacobian_coefficients": [4, PHYSICAL_DIMENSION, 560],
            "noncenter_leave_one_out": True,
        },
        "binding_revision_gates": {
            "maximum_fit_condition_number": 1.0e3,
            "maximum_leave_one_out_condition_number": 1.1e3,
            "maximum_fit_full_state_rate_relative_error": 1.0e-2,
            "maximum_fit_full_coordinate_rate_relative_error": 1.0e-2,
            "maximum_fit_q162_rate_relative_error": 3.5e-2,
            "maximum_fit_z280_rate_relative_error": 5.0e-3,
            "maximum_fit_a28_rate_relative_error": 1.5e-2,
            "maximum_leave_one_out_full_state_rate_relative_error": 2.0e-2,
            "maximum_leave_one_out_full_coordinate_rate_relative_error": 2.0e-2,
            "maximum_leave_one_out_q162_rate_relative_error": 7.0e-2,
            "maximum_leave_one_out_z280_rate_relative_error": 5.0e-3,
            "maximum_leave_one_out_a28_rate_relative_error": 2.5e-2,
            "maximum_leave_one_out_q162_Jacobian_relative_error": 5.0e-4,
            "minimum_blind_direction_singular_value": 2.5e-1,
            "new_exact_rate_calls_equal": 0,
        },
        "geometry_preflight": {
            "work_package": AUTHORIZED_NEXT,
            "count": NEW_GEOMETRY_COUNT,
            "component_bound": COMPONENT_BOUND,
            "directions_may_not_change": True,
            "coefficients_may_not_change": True,
            "minimum_partition_weight": 1.0,
            "maximum_decoder_relative_error": 5.0e-2,
            "maximum_decoder_coordinate_relative_mismatch": 7.5e-2,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "new_exact_rate_calls_equal": 0,
        },
        "blind_rate_validation": {
            "count": NEW_GEOMETRY_COUNT,
            "maximum_full_state_rate_relative_error": 7.5e-2,
            "maximum_full_coordinate_rate_relative_error": 7.5e-2,
            "maximum_q162_rate_relative_error": 7.5e-2,
            "maximum_z280_rate_relative_error": 7.5e-2,
            "maximum_a28_rate_relative_error": 7.5e-2,
            "maximum_q162_Jacobian_relative_error": 5.0e-3,
            "completed_exact_rate_calls_equal": NEW_GEOMETRY_COUNT,
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
            "geometry_pass_authorizes_only": "WP10c9d6c7c3b5c4f25cz",
            "blind_rate_pass_classification": "forward_quadratic_authentic_center_field_independently_validated",
            "blind_rate_fail_classification": "forward_quadratic_authentic_center_field_blind_validation_failed",
            "blind_rate_pass_authorizes_only": "definitions_only_reduced_slow_atlas_integrator_manifest",
            "blind_rate_fail_authorizes_only": "definitions_only_nonlinear_local_field_revision",
        },
        "authorization_boundaries": {
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "fast_average_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _checks(metrics: dict, gates: dict) -> dict[str, bool]:
    checks = {
        key: metrics[key] <= value
        for key, value in gates.items()
        if key.startswith("maximum_") and key in metrics
    }
    checks.update(
        {
            key: metrics[key] >= value
            for key, value in gates.items()
            if key.startswith("minimum_") and key in metrics
        }
    )
    checks["exact_rate_budget"] = (
        metrics["new_exact_rate_calls"] == gates["new_exact_rate_calls_equal"]
    )
    checks["blind_direction_count"] = metrics["blind_direction_count"] == NEW_GEOMETRY_COUNT
    checks["blind_direction_rank"] = metrics["blind_direction_rank"] == 3
    return checks


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
        raise RuntimeError("forward-quadratic revision manifest already exists")
    database = _revealed_database(frozen)
    closure, metrics = _fit_revision(frozen, database)
    contract = _contract()
    checks = _checks(metrics, contract["binding_revision_gates"])
    if not all(checks.values()):
        raise RuntimeError(f"forward-quadratic readiness failed: {checks}")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "forward_quadratic_local_field.npz", closure)
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
            "failed_validation_arrays_sha256": _sha(FAILED_VALIDATION_ARRAYS),
            "partitioned_field_sha256": _sha(PARTITIONED_FIELD),
            "trained_field_sha256": _sha(TRAINED_FIELD),
            "training_truth_sha256": _sha(TRAINING_TRUTH),
            "frozen_design_sha256": _sha(FROZEN_DESIGN),
            "prior_geometry_sha256": _sha(PRIOR_GEOMETRY),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "affine_blind_failure_preserved": True,
        "forward_quadratic_readiness_passed": True,
        "revealed_exact_sample_count": REVEALED_SAMPLE_COUNT,
        "new_exact_rate_calls": 0,
        "prospective_geometry_candidate_count": NEW_GEOMETRY_COUNT,
        "coefficients_frozen_before_new_geometry_and_truth": True,
        "directions_frozen_before_geometry_and_truth": True,
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
        partition.THIS_RUNNER,
        partition.THIS_TEST,
        training.THIS_RUNNER,
        training.THIS_TEST,
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
            "source_hashes": {relative: _sha(ROOT / relative) for relative in source_files},
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name) for name in training._thread_environment()
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
                "# Forward-quadratic authentic-center field revision WP10c9d6c7c3b5c4f25cx",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The four-point affine blind failure remains binding. Exact geometry, the partitioned decoder, and affine physical-Jacobian transport remain supported; only the physical-rate residual is revised.",
                "",
                "The selected residual basis is `[1, xi_forward, xi_transverse_1, xi_transverse_2, xi_forward^2]`. It is the smallest tested nonlinear basis and avoids the overfitting observed for richer quadratics and radial-basis kernels.",
                "",
                f"Worst noncenter leave-one-out full-state, full-coordinate, q162, and a28 errors are `{metrics['maximum_leave_one_out_full_state_rate_relative_error']:.6e}`, `{metrics['maximum_leave_one_out_full_coordinate_rate_relative_error']:.6e}`, `{metrics['maximum_leave_one_out_q162_rate_relative_error']:.6e}`, and `{metrics['maximum_leave_one_out_a28_rate_relative_error']:.6e}`.",
                "",
                "Four new cone-radius-discriminating directions are frozen before geometry retraction and before truth. No new exact rate, generator, nonlinear root, or propagated state was evaluated.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}` geometry-only preflight. No trajectory, microburst, predictive cycle, or reduced slow evolution is authorized.",
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
