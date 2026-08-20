#!/usr/bin/env python3
"""Independently validate the frozen active-8 projective-kernel closure."""

from __future__ import annotations

import argparse
import csv
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

import run_causal_inner_active8_projective_kernel_validation_manifest_wp10c9d6c7c3b5c4f25br as manifest  # noqa: E402
import run_causal_inner_active8_projective_kernel_geometry_wp10c9d6c7c3b5c4f25bs as geometry  # noqa: E402
import run_causal_inner_active8_tensor_rate_validation_wp10c9d6c7c3b5c4f25bp as previous  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bt"
GEOMETRY_COMMIT = "a58607389d854d5a48f723f153b997a918a5d076"
GEOMETRY_PARENT = "eeb1ae9f236f818254d33b7888ccfd9337ed9971"
GEOMETRY_TREE = "7e63c184b87d445c65107414207fb8c5007289d1"

PASS_CLASSIFICATION = (
    "active8_projective_kernel_rate_and_rank4_decoder_"
    "independently_validated"
)
FAIL_CLASSIFICATION = "active8_projective_kernel_independent_validation_failed"
AUTHORIZED_NEXT = (
    "definitions_only_active8_short_reduced_vector_field_validation_manifest"
)

ARTIFACT = (
    "causal_inner_active8_projective_kernel_rate_validation_"
    "wp10c9d6c7c3b5c4f25bt"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_active8_projective_kernel_rate_validation_"
    "wp10c9d6c7c3b5c4f25bt.py"
)
THIS_TEST = (
    "tests/test_causal_inner_active8_projective_kernel_rate_validation_"
    "wp10c9d6c7c3b5c4f25bt.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ACTIVE8_PROJECTIVE_KERNEL_RATE_"
    "VALIDATION_WP10C9D6C7C3B5C4F25BT_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

DATABASE_PATH = geometry.CANONICAL_DIRECTORY / "projective_kernel_geometry_database.npz"
PREVIOUS_CLOSURE_PATH = previous.CANONICAL_DIRECTORY / "tensor_closure.npz"
PREVIOUS_FROZEN_PATH = previous.CANONICAL_DIRECTORY / "frozen_coefficients.npz"
ONLINE_GEOMETRY_PATH = previous.ONLINE_GEOMETRY_PATH
GENERATOR_PATH = previous.GENERATOR_PATH
CURVATURE_DESIGN_PATH = previous.DESIGN_PATH
FIT_ARRAY_PATH = SCRATCH_DIRECTORY / "frozen_coefficients.npz"
FIT_LOCK_PATH = SCRATCH_DIRECTORY / "coefficient_lock.json"

TRAINING_DIRECTION_COUNT = manifest.REVEALED_HIGH_DIRECTION_COUNT
EVEN_TARGET_WEIGHT_EXPONENT = 2.0
EVEN_TIKHONOV_REGULARIZATION = 1.0 / 64.0
EVEN_QUARTIC_KERNEL_WEIGHT = 1.0 / 320.0
EVEN_KERNEL_COEFFICIENT_COUNT = TRAINING_DIRECTION_COUNT * 28
ODD_CUBIC_COEFFICIENT_COUNT = 120 * 28
CURVATURE_COEFFICIENT_COUNT = 120 * 4
TOTAL_NONLINEAR_COEFFICIENT_COUNT = (
    EVEN_KERNEL_COEFFICIENT_COUNT
    + ODD_CUBIC_COEFFICIENT_COUNT
    + CURVATURE_COEFFICIENT_COUNT
)


_plain = previous._plain
_read = previous._read
_write_json = previous._write_json
_write_npz = previous._write_npz
_load_npz = previous._load_npz
_sha = previous._sha
_checksums = previous._checksums
_relative_error = previous._relative_error
_append = previous._append
_pair_targets = previous._pair_targets
architecture = previous.architecture


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _evaluation_order() -> tuple[int, ...]:
    return tuple(range(manifest.PLANNED_CANDIDATES))


def _validate_geometry(*, require_clean: bool) -> dict:
    if _git("rev-parse", GEOMETRY_COMMIT) != GEOMETRY_COMMIT:
        raise RuntimeError("projective-kernel geometry result commit changed")
    if _git("rev-parse", f"{GEOMETRY_COMMIT}^") != GEOMETRY_PARENT:
        raise RuntimeError("projective-kernel geometry result lineage changed")
    if _git("rev-parse", f"{GEOMETRY_COMMIT}^{{tree}}") != GEOMETRY_TREE:
        raise RuntimeError("projective-kernel geometry result tree changed")
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
        raise RuntimeError("projective-kernel rate authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"geometry source changed: {relative}")
    manifest_hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    if (
        not contract["leakage_control"][
            "all_coefficients_frozen_and_hashed_before_new_rate_truth"
        ]
        or contract["mathematical_architecture"]["stored_nonlinear_coefficients_after_refit"]
        != TOTAL_NONLINEAR_COEFFICIENT_COUNT
    ):
        raise RuntimeError("projective-kernel leakage contract changed")
    previous_hashes = _checksums(previous.CANONICAL_DIRECTORY)
    previous_summary = _read(previous.CANONICAL_DIRECTORY / "summary.json")
    previous_metrics = _read(previous.CANONICAL_DIRECTORY / "metrics.json")
    if (
        previous_summary["truth_database_passed"] is not True
        or previous_summary["completed_nonbase_rate_evaluations"] != 192
        or previous_summary["failed_rate_evaluations"] != 0
        or not all(previous_metrics["truth_checks"].values())
    ):
        raise RuntimeError("revealed tensor truth database changed")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("projective-kernel rate validation requires a clean tree")
    for name, expected in geometry.chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {
        "summary": summary,
        "metrics": metrics,
        "contract": contract,
        "geometry_hashes": geometry_hashes,
        "manifest_hashes": manifest_hashes,
        "previous_hashes": previous_hashes,
    }


def _load_inputs() -> dict:
    database = _load_npz(DATABASE_PATH)
    online_geometry = _load_npz(ONLINE_GEOMETRY_PATH)
    generator = _load_npz(GENERATOR_PATH)
    previous_closure = _load_npz(PREVIOUS_CLOSURE_PATH)
    previous_frozen = _load_npz(PREVIOUS_FROZEN_PATH)
    curvature_design = _load_npz(CURVATURE_DESIGN_PATH)
    metrics = _read(geometry.CANONICAL_DIRECTORY / "metrics.json")
    states = np.asarray(database["candidate_primitive_states"], dtype=float)
    deltas = np.asarray(database["candidate_scaled_deltas"], dtype=float)
    coordinates = np.asarray(database["candidate_departure_coordinates"], dtype=float)
    candidates = metrics["candidates"]
    if (
        states.shape != (manifest.PLANNED_CANDIDATES, 112, 5)
        or deltas.shape != (manifest.PLANNED_CANDIDATES, 560)
        or coordinates.shape != (manifest.PLANNED_CANDIDATES, 28)
        or len(candidates) != manifest.PLANNED_CANDIDATES
        or generator["complete_fixed_Q_generator"].shape != (560, 560)
        or online_geometry["online_coordinate_restriction"].shape != (470, 560)
        or curvature_design["rank4_curvature_basis"].shape != (560, 4)
        or previous_closure["candidate_scaled_deltas"].shape != (192, 560)
        or previous_frozen["directions"].shape != (120, 8)
    ):
        raise RuntimeError("projective-kernel rate input dimensions changed")
    if [item["candidate_index"] for item in candidates] != list(
        range(manifest.PLANNED_CANDIDATES)
    ):
        raise RuntimeError("projective-kernel geometry candidate ordering changed")
    if [item["split"] for item in candidates[:32]] != ["holdout"] * 32:
        raise RuntimeError("independent high-radius holdout split changed")
    if [item["split"] for item in candidates[32:]] != ["tuning_low"] * 16:
        raise RuntimeError("independent radial split changed")
    return {
        "database": database,
        "online_geometry": online_geometry,
        "generator": np.asarray(generator["complete_fixed_Q_generator"], dtype=float),
        "base_rate": np.asarray(generator["fixed_Q_rate"], dtype=float),
        "previous_closure": previous_closure,
        "previous_frozen": previous_frozen,
        "curvature_basis": np.asarray(
            curvature_design["rank4_curvature_basis"], dtype=float
        ),
        "states": states,
        "deltas": deltas,
        "coordinates": coordinates,
        "candidates": candidates,
    }


def _training_targets(inputs: dict) -> dict[str, np.ndarray]:
    old = inputs["previous_frozen"]
    revealed = inputs["previous_closure"]
    extra = _pair_targets(
        deltas=revealed["candidate_scaled_deltas"][128:176],
        coordinates=revealed["candidate_departure_coordinates"][128:176],
        departure_increments=revealed["departure_rate_increments_per_second"][128:176],
        departure_linear=revealed["departure_linear_references_per_second"][128:176],
        energy_directions=inputs["database"]["energy_directions"],
        curvature_basis=inputs["curvature_basis"],
    )
    targets = {
        name: np.concatenate((np.asarray(old[name], dtype=float), extra[name]), axis=0)
        for name in (
            "directions",
            "radii",
            "rate_quadratic_targets",
            "rate_cubic_targets",
            "curvature_cubic_targets",
        )
    }
    expected = {
        "directions": (TRAINING_DIRECTION_COUNT, 8),
        "radii": (TRAINING_DIRECTION_COUNT,),
        "rate_quadratic_targets": (TRAINING_DIRECTION_COUNT, 28),
        "rate_cubic_targets": (TRAINING_DIRECTION_COUNT, 28),
        "curvature_cubic_targets": (TRAINING_DIRECTION_COUNT, 4),
    }
    if any(targets[name].shape != shape for name, shape in expected.items()):
        raise RuntimeError("revealed projective-kernel training dimensions changed")
    if not all(np.all(np.isfinite(values)) for values in targets.values()):
        raise RuntimeError("revealed projective-kernel training data are nonfinite")
    return targets


def _even_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    gram = np.asarray(left, dtype=float) @ np.asarray(right, dtype=float).T
    return gram**2 + EVEN_QUARTIC_KERNEL_WEIGHT * gram**4


def _fit_coefficients(targets: dict[str, np.ndarray]) -> tuple[dict, dict]:
    directions = targets["directions"]
    norms = np.linalg.norm(targets["rate_quadratic_targets"], axis=1)
    target_scale = float(np.median(norms))
    weights = (
        target_scale / np.maximum(norms, np.finfo(float).tiny)
    ) ** EVEN_TARGET_WEIGHT_EXPONENT
    kernel = _even_kernel(directions, directions)
    regularized = kernel + EVEN_TIKHONOV_REGULARIZATION * np.diag(1.0 / weights)
    cubic = architecture._cubic_features(directions)
    even_rank = int(np.linalg.matrix_rank(regularized))
    cubic_rank = int(np.linalg.matrix_rank(cubic))
    if even_rank != TRAINING_DIRECTION_COUNT:
        raise RuntimeError("regularized even kernel lost rank")
    if cubic_rank != 120:
        raise RuntimeError("overdetermined cubic design lost rank")
    coefficients = {
        "even_kernel_coefficients": np.linalg.solve(
            regularized, targets["rate_quadratic_targets"]
        ),
        "odd_cubic_coefficients": np.linalg.lstsq(
            cubic, targets["rate_cubic_targets"], rcond=None
        )[0],
        "curvature_cubic_coefficients": np.linalg.lstsq(
            cubic, targets["curvature_cubic_targets"], rcond=None
        )[0],
    }
    metrics = {
        "training_direction_count": int(directions.shape[0]),
        "regularized_even_kernel_rank": even_rank,
        "regularized_even_kernel_condition_number": float(np.linalg.cond(regularized)),
        "odd_cubic_feature_rank": cubic_rank,
        "odd_cubic_feature_condition_number": float(np.linalg.cond(cubic)),
        "even_target_norm_median": target_scale,
        "even_target_weight_minimum": float(np.min(weights)),
        "even_target_weight_maximum": float(np.max(weights)),
        "even_target_weight_exponent": EVEN_TARGET_WEIGHT_EXPONENT,
        "even_Tikhonov_regularization": EVEN_TIKHONOV_REGULARIZATION,
        "even_quartic_kernel_weight": EVEN_QUARTIC_KERNEL_WEIGHT,
        "stored_nonlinear_coefficient_count": TOTAL_NONLINEAR_COEFFICIENT_COUNT,
    }
    arrays = {
        "training_directions_active8": directions,
        "training_radii": targets["radii"],
        "rate_quadratic_targets": targets["rate_quadratic_targets"],
        "rate_cubic_targets": targets["rate_cubic_targets"],
        "curvature_cubic_targets": targets["curvature_cubic_targets"],
        "even_target_norms": norms,
        "even_target_weights": weights,
        **coefficients,
    }
    if not all(np.all(np.isfinite(values)) for values in arrays.values()):
        raise RuntimeError("projective-kernel coefficients are nonfinite")
    return metrics, arrays


def _freeze_or_validate_coefficients(inputs: dict) -> tuple[dict, dict]:
    if FIT_LOCK_PATH.exists() != FIT_ARRAY_PATH.exists():
        raise RuntimeError("projective-kernel coefficient lock is incomplete")
    targets = _training_targets(inputs)
    if FIT_LOCK_PATH.exists():
        lock = _read(FIT_LOCK_PATH)
        if (
            lock["coefficient_sha256"] != _sha(FIT_ARRAY_PATH)
            or lock["training_direction_count"] != TRAINING_DIRECTION_COUNT
            or lock["validation_rate_evaluations_at_freeze"] != 0
            or lock["runner_sha256"] != _sha(ROOT / THIS_RUNNER)
            or lock["geometry_database_sha256"] != _sha(DATABASE_PATH)
            or lock["previous_closure_sha256"] != _sha(PREVIOUS_CLOSURE_PATH)
            or lock["previous_frozen_sha256"] != _sha(PREVIOUS_FROZEN_PATH)
        ):
            raise RuntimeError("frozen projective-kernel coefficient lock changed")
        arrays = _load_npz(FIT_ARRAY_PATH)
        for source_name, frozen_name in (
            ("directions", "training_directions_active8"),
            ("radii", "training_radii"),
            ("rate_quadratic_targets", "rate_quadratic_targets"),
            ("rate_cubic_targets", "rate_cubic_targets"),
            ("curvature_cubic_targets", "curvature_cubic_targets"),
        ):
            if not np.array_equal(targets[source_name], arrays[frozen_name]):
                raise RuntimeError("revealed training truth changed after coefficient freeze")
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
        "previous_closure_sha256": _sha(PREVIOUS_CLOSURE_PATH),
        "previous_frozen_sha256": _sha(PREVIOUS_FROZEN_PATH),
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


def _progress_array_names() -> tuple[str, ...]:
    return previous._progress_array_names()


def _progress_identity() -> dict:
    return {
        "execution_commit": _git("rev-parse", "HEAD"),
        "geometry_commit": GEOMETRY_COMMIT,
        "geometry_database_sha256": _sha(DATABASE_PATH),
        "coefficient_sha256": _sha(FIT_ARRAY_PATH),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
    }


def _empty_progress(identity: dict) -> dict:
    return {
        "identity": identity,
        "evaluations": [],
        "failures": [],
        "total_rates_per_second": np.empty((0, 560), dtype=float),
        "free_rates_per_second": np.empty((0, 560), dtype=float),
        "physical_reaction_actions_per_second": np.empty((0, 560), dtype=float),
        "multiplier_coordinates_per_second": np.empty((0, 3), dtype=float),
        "online_470_coordinate_rates_per_second": np.empty((0, 470), dtype=float),
        "departure_rate_increments_per_second": np.empty((0, 28), dtype=float),
        "linear_rate_references_per_second": np.empty((0, 560), dtype=float),
        "departure_linear_references_per_second": np.empty((0, 28), dtype=float),
    }


def _save_progress(progress: dict) -> None:
    SCRATCH_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        SCRATCH_DIRECTORY / "progress.json",
        {
            "identity": progress["identity"],
            "evaluations": progress["evaluations"],
            "failures": progress["failures"],
        },
    )
    _write_npz(
        SCRATCH_DIRECTORY / "progress.npz",
        {name: progress[name] for name in _progress_array_names()},
    )


def _load_or_create_progress() -> dict:
    identity = _progress_identity()
    json_path = SCRATCH_DIRECTORY / "progress.json"
    npz_path = SCRATCH_DIRECTORY / "progress.npz"
    if not json_path.exists() and not npz_path.exists():
        return _empty_progress(identity)
    if not json_path.exists() or not npz_path.exists():
        raise RuntimeError("projective-kernel rate checkpoint is incomplete")
    recorded = _read(json_path)
    if recorded["identity"] != identity:
        raise RuntimeError("projective-kernel rate checkpoint identity changed")
    progress = {
        "identity": identity,
        "evaluations": recorded["evaluations"],
        "failures": recorded["failures"],
        **_load_npz(npz_path),
    }
    count = len(progress["evaluations"])
    if any(progress[name].shape[0] != count for name in _progress_array_names()):
        raise RuntimeError("projective-kernel rate checkpoint dimensions changed")
    if [item["candidate_index"] for item in progress["evaluations"]] != list(
        _evaluation_order()[:count]
    ):
        raise RuntimeError("projective-kernel rate evaluation order changed")
    return progress


def _evaluate_candidate(
    inputs: dict, progress: dict, index: int, data: dict, components: dict
) -> None:
    previous._evaluate_candidate(inputs, progress, index, data, components)


def _ordered_truth_arrays(progress: dict) -> dict[str, np.ndarray]:
    count = len(progress["evaluations"])
    indices = np.asarray(
        [item["candidate_index"] for item in progress["evaluations"]], dtype=int
    )
    arrays = {}
    for name in _progress_array_names():
        source = progress[name]
        ordered = np.full((count,) + source.shape[1:], np.nan, dtype=float)
        if count:
            if np.min(indices) < 0 or np.max(indices) >= count:
                raise RuntimeError("cannot order incomplete projective-kernel truth arrays")
            ordered[indices] = source
        arrays[name] = ordered
    return arrays


def _truth_metrics(progress: dict, began: float) -> tuple[dict, dict[str, np.ndarray]]:
    evaluations = progress["evaluations"]

    def maximum(name: str, default=math.inf) -> float:
        values = [item[name] for item in evaluations]
        return float(max(values)) if values else float(default)

    def minimum(name: str, default=-math.inf) -> float:
        values = [item[name] for item in evaluations]
        return float(min(values)) if values else float(default)

    metrics = {
        "planned_nonbase_rate_evaluations": manifest.PLANNED_CANDIDATES,
        "completed_nonbase_rate_evaluations": len(evaluations),
        "failed_rate_evaluations": len(progress["failures"]),
        "failures": progress["failures"],
        "maximum_state_rate_linear_relative_defect": maximum(
            "state_rate_linear_relative_defect"
        ),
        "maximum_departure_rate_linear_relative_defect": maximum(
            "departure_rate_linear_relative_defect"
        ),
        "minimum_reconstruction_factor": minimum("minimum_reconstruction_factor", math.inf),
        "maximum_reconstruction_factor": maximum("maximum_reconstruction_factor"),
        "maximum_raw_Schur_condition_number": maximum("raw_Schur_condition_number"),
        "maximum_reaction_identity_defect": maximum("reaction_identity_defect"),
        "maximum_rate_tangency_relative_defect": maximum(
            "rate_tangency_relative_defect"
        ),
        "maximum_coordinate_Jacobian_condition_number": maximum(
            "coordinate_Jacobian_condition_number"
        ),
        "maximum_H_over_R": maximum("maximum_H_over_R"),
        "minimum_scattering_optical_depth": minimum("minimum_scattering_optical_depth"),
        "maximum_incoming_excision_characteristics": maximum(
            "incoming_excision_characteristics"
        ),
        "wall_seconds_this_process": time.perf_counter() - began,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "evaluations": evaluations,
    }
    arrays = {}
    if len(evaluations) == manifest.PLANNED_CANDIDATES:
        arrays = _ordered_truth_arrays(progress)
    return metrics, arrays


def _execute_truth(
    inputs: dict,
) -> tuple[dict, dict[str, np.ndarray], dict, dict]:
    fit_metrics, fit_arrays = _freeze_or_validate_coefficients(inputs)
    progress = _load_or_create_progress()
    resumed = len(progress["evaluations"])
    began = time.perf_counter()
    data = previous.rate_tools.manifest.prior_screen.manifest.parent.manifest.failed_screen._anchor_data(
        "primary"
    )
    components = geometry.base.high_chart._prepare_components()
    order = _evaluation_order()
    for position in range(len(progress["evaluations"]), len(order)):
        index = order[position]
        candidate = inputs["candidates"][index]
        try:
            _evaluate_candidate(inputs, progress, index, data, components)
            status = "accepted"
        except Exception as error:  # fail closed on first truth failure
            progress["failures"].append(
                {
                    "candidate_index": index,
                    "pair_index": candidate["pair_index"],
                    "split": candidate["split"],
                    "sign": candidate["sign"],
                    "reason": type(error).__name__,
                    "message": str(error),
                }
            )
            status = "failed"
        _save_progress(progress)
        print(
            json.dumps(
                {
                    "truth_evaluation": position + 1,
                    "candidate_index": index,
                    "total": manifest.PLANNED_CANDIDATES,
                    "split": candidate["split"],
                    "direction": candidate["split_direction_index"],
                    "sign": candidate["sign"],
                    "status": status,
                    "elapsed_this_process_seconds": time.perf_counter() - began,
                }
            ),
            flush=True,
        )
        if progress["failures"]:
            break
    metrics, arrays = _truth_metrics(progress, began)
    metrics["resumed_evaluation_count"] = resumed
    metrics["coefficient_lock_frozen"] = FIT_LOCK_PATH.exists()
    return metrics, arrays, fit_metrics, fit_arrays


def _truth_gate_checks(metrics: dict, gates: dict) -> dict:
    return previous._truth_gate_checks(metrics, gates)


def _predict(active: np.ndarray, coefficients: dict) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(active, dtype=float)
    radius = float(np.linalg.norm(active))
    if radius <= np.finfo(float).tiny:
        return np.zeros(28), np.zeros(4)
    direction = active / radius
    quadratic = (
        _even_kernel(direction.reshape(1, 8), coefficients["training_directions_active8"])[0]
        @ coefficients["even_kernel_coefficients"]
    )
    cubic_features = architecture._cubic_features(direction.reshape(1, 8))[0]
    cubic = cubic_features @ coefficients["odd_cubic_coefficients"]
    curvature = (
        radius**3
        * cubic_features
        @ coefficients["curvature_cubic_coefficients"]
    )
    return radius**2 * quadratic + radius**3 * cubic, curvature


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
    predicted_nonlinear = np.full((manifest.PLANNED_CANDIDATES, 28), np.nan)
    predicted_curvature = np.full((manifest.PLANNED_CANDIDATES, 4), np.nan)
    predicted_deltas = np.full((manifest.PLANNED_CANDIDATES, 560), np.nan)
    for index, candidate in enumerate(inputs["candidates"]):
        if candidate["split"] != "holdout":
            continue
        active = energy.T @ inputs["coordinates"][index]
        rate, curvature = _predict(active, coefficients)
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
                    / max(float(np.linalg.norm(inputs["deltas"][index])), np.finfo(float).tiny)
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
        raise RuntimeError("independent holdout validation count changed")

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

    def targets(start: int, stop: int) -> dict[str, np.ndarray]:
        return _pair_targets(
            deltas=inputs["deltas"][start:stop],
            coordinates=inputs["coordinates"][start:stop],
            departure_increments=truth["departure_rate_increments_per_second"][start:stop],
            departure_linear=truth["departure_linear_references_per_second"][start:stop],
            energy_directions=energy,
            curvature_basis=inputs["curvature_basis"],
        )

    high = targets(0, 2 * manifest.NEW_RADIAL_DIRECTION_COUNT)
    low = targets(manifest.NEW_HIGH_CANDIDATE_COUNT, manifest.PLANNED_CANDIDATES)

    def differences(name: str) -> np.ndarray:
        return np.asarray(
            [
                _relative_error(high[name][index], low[name][index])
                for index in range(manifest.NEW_RADIAL_DIRECTION_COUNT)
            ],
            dtype=float,
        )

    quadratic = differences("rate_quadratic_targets")
    cubic = differences("rate_cubic_targets")
    curvature = differences("curvature_cubic_targets")
    return {
        "maximum_quadratic_target_high_low_relative_difference": float(np.max(quadratic)),
        "maximum_cubic_rate_target_high_low_relative_difference": float(np.max(cubic)),
        "maximum_curvature_cubic_target_high_low_relative_difference": float(np.max(curvature)),
        "quadratic_directionwise_high_low_relative_difference": quadratic,
        "cubic_rate_directionwise_high_low_relative_difference": cubic,
        "curvature_cubic_directionwise_high_low_relative_difference": curvature,
    }


def _model_gate_checks(model: dict, radial: dict, fit: dict, contract: dict) -> dict:
    model_gates = contract["binding_independent_model_gates"]
    radial_gates = contract["binding_radial_consistency_gates"]
    checks = {
        name: model[name] <= threshold
        for name, threshold in model_gates.items()
        if name != "minimum_reconstructed_state_reconstruction_factor"
        and name != "minimum_reconstructed_scattering_optical_depth"
    }
    checks["minimum_reconstructed_state_reconstruction_factor"] = model[
        "minimum_reconstructed_state_reconstruction_factor"
    ] >= model_gates["minimum_reconstructed_state_reconstruction_factor"]
    checks["minimum_reconstructed_scattering_optical_depth"] = model[
        "minimum_reconstructed_scattering_optical_depth"
    ] >= model_gates["minimum_reconstructed_scattering_optical_depth"]
    checks.update({name: radial[name] <= threshold for name, threshold in radial_gates.items()})
    checks.update(
        {
            "regularized_even_kernel_full_rank": fit["regularized_even_kernel_rank"]
            == TRAINING_DIRECTION_COUNT,
            "odd_cubic_feature_full_rank": fit["odd_cubic_feature_rank"] == 120,
            "coefficient_count": fit["stored_nonlinear_coefficient_count"]
            == TOTAL_NONLINEAR_COEFFICIENT_COUNT,
            "regularized_even_kernel_condition_finite": math.isfinite(
                fit["regularized_even_kernel_condition_number"]
            ),
            "odd_cubic_feature_condition_finite": math.isfinite(
                fit["odd_cubic_feature_condition_number"]
            ),
        }
    )
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
                    "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
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
            "latest_source_parent_commit": GEOMETRY_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_geometry(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("projective-kernel rate validation already canonicalized")
    inputs = _load_inputs()
    truth_metrics, truth_arrays, fit_metrics, fit_arrays = _execute_truth(inputs)
    truth_checks = _truth_gate_checks(
        truth_metrics, frozen["contract"]["binding_truth_rate_gates"]
    )
    truth_passed = all(truth_checks.values())
    model_metrics = {}
    radial_metrics = {}
    model_arrays = {}
    model_checks = {"truth_database_complete": False}
    if truth_passed:
        radial_metrics = _radial_metrics(inputs, truth_arrays)
        model_metrics, model_arrays = _validation_metrics(inputs, truth_arrays, fit_arrays)
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
        CANONICAL_DIRECTORY / "projective_kernel_closure.npz",
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
            "previous_tensor_rate_hashes": frozen["previous_hashes"],
            "previous_closure_sha256": _sha(PREVIOUS_CLOSURE_PATH),
            "previous_frozen_sha256": _sha(PREVIOUS_FROZEN_PATH),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        geometry.THIS_RUNNER,
        geometry.THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        previous.THIS_RUNNER,
        previous.THIS_TEST,
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
            "resumed_from_evaluation_count": truth_metrics["resumed_evaluation_count"],
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {relative: _sha(ROOT / relative) for relative in source_files},
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
                "# Active-8 projective-kernel rate validation WP10c9d6c7c3b5c4f25bt",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{truth_metrics['completed_nonbase_rate_evaluations']}` of `{manifest.PLANNED_CANDIDATES}` new exact truth-rate evaluations; failures: `{truth_metrics['failed_rate_evaluations']}`.",
                "",
                "All 7,872 nonlinear coefficients were frozen and hashed from 144 previously revealed high-radius directions before any new rate response was read.",
                "",
                f"Independent validation: {validation_text}.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No trajectory, predictive cycle, or reduced slow evolution is authorized directly.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
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
